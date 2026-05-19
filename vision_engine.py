import cv2
import numpy as np
import copy

class VisionEngine:
    def __init__(self, cv_img):
        self.img_orig = cv_img
        self.img_debug = None           # 顏色判定 Debug 專用圖
        self.grid_state = [[0]*8 for _ in range(8)]
        self.detected_pieces = []
        self.warp_orig = None
        self.piece_scale = 0.50

    def process(self):
        # ==========================================
        # 1. 影像預處理 (萃取特徵與二值化)
        # ==========================================
        hsv = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        v_channel = np.maximum.reduce([s, v]) # 取 S 與 V 的最大值，強化特徵
        blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
        
        # 基礎二值化
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        
        # 💡 找棋盤專用：使用較強的「閉運算」黏合嚴重的斷線與破洞
        board_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        
        # 💡 找下方待放方塊專用：使用較輕微的閉運算，保留小方塊形狀細節
        piece_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # 初始化 Debug 圖
        self.img_debug = self.img_orig.copy()

        # ==========================================
        # 2. 定位棋盤（閉運算 + 同心矩形找內框法）
        # ==========================================
        # 使用 RETR_TREE 建立完整的輪廓階層，內外框才都會被找出來
        cnts, hierarchy = cv2.findContours(board_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False

        candidates = []
        img_area = v_channel.shape[0] * v_channel.shape[1]
        
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # 過濾太小或太大的雜訊 (排除整個畫面，或小於畫面 10% 的東西)
            if area < (img_area * 0.1) or area > (img_area * 0.9): 
                continue
                
            # 確保它是正方形 (長寬比在 0.8 ~ 1.2 之間)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                # 計算中心點
                cx, cy = x + w/2.0, y + h/2.0
                candidates.append({
                    'area': area,
                    'center': (cx, cy),
                    'approx': np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]], dtype=np.int32)
                })

        if not candidates:
            return False

       # 依面積從大到小排序
        candidates = sorted(candidates, key=lambda c: c['area'], reverse=True)

        # 💡 尋找「內框」的新邏輯：最大正方形面積 90% 以上的最小正方形
        max_area = candidates[0]['area']
        threshold_area = max_area * 0.90 # 設定 90% 為門檻
        
        best_cand = candidates[0] # 預設為最大正方形
        
        for cand in candidates:
            # 只要這個框的面積大於等於最大框的 90%
            if cand['area'] >= threshold_area:
                # 為了極致的安全，加上基本的同心檢查，防止抓到旁邊無關的特效框 (容許值稍微放寬到 30)
                dist = np.linalg.norm(np.array(candidates[0]['center']) - np.array(cand['center']))
                if dist < 30: 
                    best_cand = cand # 持續更新為較小的框
            else:
                # 因為已經從大到小排好序了，一旦遇到小於 90% 的框，後面的都不用看了
                break 

        # 進行透視變換
        pts1 = self.order_points(best_cand['approx'].reshape(4, 2))
        orig_unit = np.linalg.norm(pts1[0] - pts1[1]) / 8.0
        M = cv2.getPerspectiveTransform(pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        self.warp_orig = cv2.warpPerspective(self.img_orig, M, (400, 400))

        # ==========================================
        # 3. 採樣 8x8 棋盤底色（直接取原圖 V 通道法）
        # ==========================================
        warp_hsv = cv2.cvtColor(self.warp_orig, cv2.COLOR_BGR2HSV)
        u = 400 / 8
        centers_v = []
        
        for r in range(8):
            for c in range(8):
                cx, cy = int((c + 0.5) * u), int((r + 0.5) * u)
                roi_v = warp_hsv[cy-2:cy+2, cx-2:cx+2, 2]
                centers_v.append(np.median(roi_v) if roi_v.size > 0 else 0)

        # 找出最低明度作為底色，並設定 3% 寬容度
        base_bg_v = min(centers_v)
        tolerance = 255 * 0.03

        for r in range(8):
            for c in range(8):
                is_p = centers_v[r*8+c] > (base_bg_v + tolerance)
                self.grid_state[r][c] = 1 if is_p else 0
                
                # 繪製 Debug 狀態
                cv2.polylines(self.img_debug, [self.get_cell_poly(pts1, r, c)], True, (80,80,80), 1)
                color_fill = (255,255,255) if is_p else (120,120,120)
                cv2.fillPoly(self.img_debug, [self.get_cell_poly_sampling(pts1, r, c, 0.4, 0.6)], color_fill)

        # ==========================================
        # 4. 全域採樣待放區背景色
        # ==========================================
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(pts1[:, 1]))
        ay_s, ay_e = bottom_y + 40, int(img_h * 0.88)
        
        # 💡 下半部尋找待放方塊時，改用保留細節的 piece_thresh
        piece_area_mask = piece_thresh[ay_s:ay_e, :]
        piece_area_color = self.img_orig[ay_s:ay_e, :]

        # 腐蝕遮罩以獲取純淨背景
        bg_mask = cv2.bitwise_not(piece_area_mask)
        bg_mask = cv2.erode(bg_mask, np.ones((5, 5), np.uint8), iterations=2)
        bg_pixels = piece_area_color[bg_mask == 255]
        global_bg_color = np.median(bg_pixels, axis=0) if len(bg_pixels) > 100 else piece_area_color[5, 5]

        # ==========================================
        # 5. 解析待放方塊
        # ==========================================
        p_cnts, _ = cv2.findContours(piece_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates_p = []
        for cnt in p_cnts:
            if cv2.contourArea(cnt) < (orig_unit**2) * 0.2:
                continue
            x, y, pw, ph = cv2.boundingRect(cnt)
            if pw > 6*orig_unit or ph > 6*orig_unit:
                continue
            candidates_p.append([x, y + ay_s, pw, ph, x + pw/2])

        final_pieces = sorted(candidates_p, key=lambda p: p[0])[:3]
        p_unit = orig_unit * self.piece_scale
        self.detected_pieces = []
        
        for x, ay, pw, ph, _ in final_pieces:
            mask = piece_thresh[ay:ay+ph, x:x+pw]
            color_roi = self.img_orig[ay:ay+ph, x:x+pw]
            self.detected_pieces.append(self.parse_piece_color_only(mask, color_roi, pw, ph, p_unit, x, ay, global_bg_color))

        # 標示棋盤外框
        cv2.polylines(self.img_debug, [pts1.astype(int)], True, (0, 255, 0), 3)
        return True

    def parse_piece_color_only(self, mask, color_roi, pw, ph, unit, ox, oy, bg_color):
        nz = cv2.findNonZero(mask)
        if nz is None: return [[1]]
        mx, my, mw, mh = cv2.boundingRect(nz)
        
        cols, rows = max(1, min(5, int(round(mw/unit)))), max(1, min(5, int(round(mh/unit))))
        col_b, row_b = np.linspace(0, mw, cols+1).astype(int), np.linspace(0, mh, rows+1).astype(int)
        grid = [[0]*cols for _ in range(rows)]

        cv2.circle(self.img_debug, (ox + 10, oy - 15), 6, bg_color.tolist(), -1)
        cv2.circle(self.img_debug, (ox + 10, oy - 15), 6, (255,255,255), 1)

        for r in range(rows):
            for c in range(cols):
                c_s, c_e, r_s, r_e = col_b[c], col_b[c+1], row_b[r], row_b[r+1]
                cx_s, cx_e = c_s + int(0.4*(c_e-c_s)), c_s + int(0.6*(c_e-c_s))
                cy_s, cy_e = r_s + int(0.4*(r_e-r_s)), r_s + int(0.6*(r_e-r_s))
                
                patch = color_roi[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
                dist = np.linalg.norm(np.median(patch, axis=(0,1)) - bg_color) if patch.size > 0 else 0
                is_p = dist > 40
                grid[r][c] = 1 if is_p else 0

                sx, sy, ex, ey = ox+mx+c_s, oy+my+r_s, ox+mx+c_e, oy+my+r_e
                cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), (80,80,80), 1)
                fill_color = (255,255,255) if is_p else (120,120,120)
                cv2.rectangle(self.img_debug, (ox+mx+cx_s, oy+my+cy_s), (ox+mx+cx_e, oy+my+cy_e), fill_color, -1)
        return grid

    # --- 座標變換工具 ---
    def lerp(self, p1, p2, t): return p1 + (p2 - p1) * t
    def get_p(self, pts, row, col):
        top = self.lerp(pts[0], pts[1], col/8.0); bot = self.lerp(pts[3], pts[2], col/8.0)
        return self.lerp(top, bot, row/8.0)
    def get_cell_poly(self, pts, r, c):
        return np.array([self.get_p(pts,r,c), self.get_p(pts,r,c+1), self.get_p(pts,r+1,c+1), self.get_p(pts,r+1,c)], dtype=np.int32)
    def get_cell_poly_sampling(self, pts, r, c, s, e):
        return np.array([self.get_p(pts,r+s,c+s), self.get_p(pts,r+s,c+e), self.get_p(pts,r+e,c+e), self.get_p(pts,r+e,c+s)], dtype=np.int32)
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect


class LogicSolver:
    def solve(self, grid, pieces, p_indices, path=[]):
        if not p_indices: return path
        for i in p_indices:
            p = pieces[i]
            for r in range(8):
                for c in range(8):
                    if self.can_place(grid, p, r, c):
                        next_g = self.simulate(grid, p, r, c)
                        res = self.solve(next_g, pieces, [idx for idx in p_indices if idx != i], 
                                         path + [(i, r, c, *self.get_cleared(self.place_only(grid, p, r, c)))])
                        if res: return res
        return None

    def can_place(self, grid, p, r, c):
        for pr in range(len(p)):
            for pc in range(len(p[0])):
                if p[pr][pc] and (r+pr>=8 or c+pc>=8 or grid[r+pr][c+pc]):
                    return False
        return True

    def place_only(self, grid, p, r, c):
        ng = copy.deepcopy(grid)
        for pr in range(len(p)):
            for pc in range(len(p[0])):
                if p[pr][pc]:
                    ng[r+pr][c+pc] = 1
        return ng

    def get_cleared(self, grid):
        rs = [i for i, row in enumerate(grid) if all(row)]
        cs = [j for j in range(8) if all(grid[i][j] for i in range(8))]
        return rs, cs

    def simulate(self, grid, p, r, c):
        ng = self.place_only(grid, p, r, c)
        rs, cs = self.get_cleared(ng)
        for i in rs: ng[i] = [0]*8
        for j in cs:
            for i in range(8): ng[i][j] = 0
        return ng
