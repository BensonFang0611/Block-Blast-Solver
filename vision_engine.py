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
        v_channel = np.maximum.reduce([s, v]) 
        blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        piece_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.img_debug = self.img_orig.copy()

        # ==========================================
        # 2. 定位棋盤（宏觀大框隔離 + 微觀內線反推）
        # ==========================================
        # [步驟 A]：先用超大筆刷抓出「粗略的外輪廓」
        board_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(board_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return False

        candidates = []
        img_area = v_channel.shape[0] * v_channel.shape[1]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > (img_area * 0.1) and 0.8 <= float(w) / h <= 1.2:
                candidates.append({'area': area, 'bx': x, 'by': y, 'bw': w, 'bh': h})
        
        if not candidates: return False
        
        # 取得粗略的棋盤大框 (bx, by, bw, bh)
        rough_box = max(candidates, key=lambda c: c['area'])
        bx, by, bw, bh = rough_box['bx'], rough_box['by'], rough_box['bw'], rough_box['bh']

        # [步驟 B]：建立 90% 隔離區 (上下左右各往內縮 5%，完美避開邊緣特效)
        sx, sy = int(bx + bw * 0.05), int(by + bh * 0.05)
        sw, sh = int(bw * 0.90), int(bh * 0.90)

        # 提取乾淨的水平與垂直線
        kernel_h, kernel_v = np.ones((1, 51), np.uint8), np.ones((51, 1), np.uint8)
        thresh_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        thresh_h = cv2.morphologyEx(thresh_h, cv2.MORPH_CLOSE, kernel_h)
        thresh_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        thresh_v = cv2.morphologyEx(thresh_v, cv2.MORPH_CLOSE, kernel_v)

        # 💡 [關鍵]：只在隔離區 (ROI) 內進行投影積分，雜訊降至 0%！
        roi_h = thresh_h[sy:sy+sh, sx:sx+sw]
        roi_v = thresh_v[sy:sy+sh, sx:sx+sw]
        proj_y = np.sum(roi_h, axis=1) / 255
        proj_x = np.sum(roi_v, axis=0) / 255

        def get_exact_edges_from_roi(projection, offset_start, rough_min):
            """ 質心平均法 + 線性迴歸：算出變異數最小、最完美的 0 與 8 邊界 """
            if len(projection) == 0: return None, None
            
            threshold = np.max(projection) * 0.15
            valid_coords = np.where(projection > threshold)[0]
            if len(valid_coords) < 3: return None, None

            # 💡 神招一：質心平均法 (Weighted Average)
            peaks = []
            current_group = [valid_coords[0]]
            for i in range(1, len(valid_coords)):
                # 如果線條距離相近，歸為同一團
                if valid_coords[i] - valid_coords[i-1] <= 15:
                    current_group.append(valid_coords[i])
                else:
                    # 不取最高點，而是用「白點數量」當權重，算出這團線的平均中心點
                    weights = projection[current_group]
                    center = np.average(current_group, weights=weights)
                    peaks.append(center + offset_start)
                    current_group = [valid_coords[i]]
            
            # 處理最後一團
            weights = projection[current_group]
            center = np.average(current_group, weights=weights)
            peaks.append(center + offset_start)

            if len(peaks) < 3: return None, None

            # 先求出粗略的間距，用來給這些線條「編號」
            diffs = np.diff(peaks)
            valid_diffs = [d for d in diffs if d > 20]
            if not valid_diffs: return None, None
            u_est = np.median(valid_diffs)

            # 給每條線一個相對的整數索引 (例如 0, 1, 2... 或 0, 1, 3... 中間有斷層也沒關係)
            indices = [0]
            for i in range(1, len(peaks)):
                idx = int(round((peaks[i] - peaks[0]) / u_est))
                indices.append(idx)

            # 💡 神招二：線性迴歸 / 最小平方法 (Least Squares Fit)
            # 利用數學公式尋找方差/標準差最小的解，擬合出方程式： 座標 = 索引 * 完美間距 + 完美起點
            # polyfit 會回傳斜率 (最完美的格子大小 u_opt) 與 截距 (最完美的第 0 條線基準 offset_0)
            u_opt, offset_0 = np.polyfit(indices, peaks, 1)

            # 💡 神級防呆：確認 offset_0 到底是全域的第幾條線？
            # 拿它跟我們在第一步抓到的粗略大外框 (rough_min) 比較
            line_index = int(round((offset_0 - rough_min) / u_opt))
            
            # 數學推演：精確 0 邊界 = 基準點 - (索引 * 完美間距)
            exact_min = offset_0 - line_index * u_opt
            exact_max = exact_min + 8 * u_opt
            
            return exact_min, exact_max

        # [步驟 C]：利用隔離區抓到的內線，精確反推出完美的上下左右外框
        min_y, max_y = get_exact_edges_from_roi(proj_y, sy, by)
        min_x, max_x = get_exact_edges_from_roi(proj_x, sx, bx)

        # 萬一真的被炸到連 3 條內線都找不到，我們就退回使用大外框內縮 3% 的保險機制
        if None in (min_x, max_x, min_y, max_y):
            min_x, max_x = bx + bw * 0.03, bx + bw * 0.97
            min_y, max_y = by + bh * 0.03, by + bh * 0.97

        approx = np.array([
            [[min_x, min_y]], [[min_x, max_y]], 
            [[max_x, max_y]], [[max_x, min_y]]
        ], dtype=np.int32)

        pts1 = self.order_points(approx.reshape(4, 2))
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

        base_bg_v = min(centers_v)
        tolerance = 255 * 0.03
        for r in range(8):
            for c in range(8):
                is_p = centers_v[r*8+c] > (base_bg_v + tolerance)
                self.grid_state[r][c] = 1 if is_p else 0
                cv2.polylines(self.img_debug, [self.get_cell_poly(pts1, r, c)], True, (80,80,80), 1)
                color_fill = (255,255,255) if is_p else (120,120,120)
                cv2.fillPoly(self.img_debug, [self.get_cell_poly_sampling(pts1, r, c, 0.4, 0.6)], color_fill)

        # ==========================================
        # 4. 全域採樣待放區背景色 (含安全防呆)
        # ==========================================
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(pts1[:, 1]))
        ay_s, ay_e = bottom_y + 40, int(img_h * 0.88)

        if ay_s >= ay_e:
            ay_e = img_h
            if ay_s >= ay_e: return True 
        
        piece_area_mask = piece_thresh[ay_s:ay_e, :]
        piece_area_color = self.img_orig[ay_s:ay_e, :]
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
            if cv2.contourArea(cnt) < (orig_unit**2) * 0.2: continue
            x, y, pw, ph = cv2.boundingRect(cnt)
            if pw > 6*orig_unit or ph > 6*orig_unit: continue
            candidates_p.append([x, y + ay_s, pw, ph, x + pw/2])

        final_pieces = sorted(candidates_p, key=lambda p: p[0])[:3]
        p_unit = orig_unit * self.piece_scale
        self.detected_pieces = []
        for x, ay, pw, ph, _ in final_pieces:
            mask = piece_thresh[ay:ay+ph, x:x+pw]
            color_roi = self.img_orig[ay:ay+ph, x:x+pw]
            self.detected_pieces.append(self.parse_piece_color_only(mask, color_roi, pw, ph, p_unit, x, ay, global_bg_color))

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
