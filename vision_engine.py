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
        # 1. 影像預處理（僅用於定位，不參與方塊判定）
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 初始化 Debug 圖（直接使用原圖繪製）
        self.img_debug = self.img_orig.copy()

        # 2. 定位 8x8 棋盤
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return False
        
        candidates = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < (gray.shape[0] * gray.shape[1] * 0.1): continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                candidates.append({'area': area, 'approx': approx})
        
        if not candidates: return False
        best_cand = sorted(candidates, key=lambda x: x['area'], reverse=True)[0]
        pts1 = self.order_points(best_cand['approx'].reshape(4, 2))
        orig_unit = np.linalg.norm(pts1[0] - pts1[1]) / 8.0 
        
        # 取得棋盤透視變換圖
        M = cv2.getPerspectiveTransform(pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        self.warp_orig = cv2.warpPerspective(self.img_orig, M, (400, 400))
        warp_color = self.warp_orig
        
        # 3. 採樣 8x8 棋盤底色
        u = 400 / 8
        centers_color = []
        for r in range(8):
            for c in range(8):
                cx, cy = int((c + 0.5) * u), int((r + 0.5) * u)
                roi = warp_color[cy-2:cy+2, cx-2:cx+2]
                centers_color.append(np.median(roi, axis=(0, 1)) if roi.size > 0 else [0,0,0])

        # 統計棋盤出現頻率最高的顏色作為「棋盤背景」
        color_bins = {}
        for mc in centers_color:
            q = (int(mc[0]/20), int(mc[1]/20), int(mc[2]/20))
            color_bins[q] = color_bins.get(q, 0) + 1
        board_bg_q = max(color_bins, key=color_bins.get)
        board_bg_rgb = np.mean([mc for mc in centers_color if (int(mc[0]/20), int(mc[1]/20), int(mc[2]/20)) == board_bg_q], axis=0)

        # 判定 8x8 狀態
        for r in range(8):
            for c in range(8):
                dist = np.linalg.norm(centers_color[r*8+c] - board_bg_rgb)
                is_p = dist > 45 # 顏色距離閾值
                self.grid_state[r][c] = 1 if is_p else 0
                
                # 繪製棋盤 Debug：白色小格(有), 灰色(無)
                cv2.polylines(self.img_debug, [self.get_cell_poly(pts1, r, c)], True, (80,80,80), 1)
                color_fill = (255,255,255) if is_p else (120,120,120)
                cv2.fillPoly(self.img_debug, [self.get_cell_poly_sampling(pts1, r, c, 0.4, 0.6)], color_fill)

        # 4. 🌟 全域採樣待放區背景色（取三個方塊間共通的空白處面積）
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(pts1[:, 1]))
        ay_s, ay_e = bottom_y + 40, int(img_h * 0.88)
        piece_area_mask = thresh[ay_s:ay_e, :]
        piece_area_color = self.img_orig[ay_s:ay_e, :]
        
        # 建立腐蝕遮罩以獲取「純淨背景」
        bg_mask = cv2.bitwise_not(piece_area_mask)
        bg_mask = cv2.erode(bg_mask, np.ones((5,5), np.uint8), iterations=2)
        bg_pixels = piece_area_color[bg_mask == 255]
        
        if len(bg_pixels) > 100:
            global_bg_color = np.median(bg_pixels, axis=0)
        else:
            global_bg_color = piece_area_color[5, 5] # 備案：取左上角像素

        # 5. 解析待放物
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
            mask = thresh[ay:ay+ph, x:x+pw]
            color_roi = self.img_orig[ay:ay+ph, x:x+pw]
            self.detected_pieces.append(self.parse_piece_color_only(mask, color_roi, pw, ph, p_unit, x, ay, global_bg_color))
        
        # 標示棋盤外框確認定位
        cv2.polylines(self.img_debug, [pts1.astype(int)], True, (0, 255, 0), 3)
        return True

    def parse_piece_color_only(self, mask, color_roi, pw, ph, unit, ox, oy, bg_color):
        """✨ 待放物解析：消除座標累積誤差，使用像素級邊界分配"""
        nz = cv2.findNonZero(mask)
        if nz is None: return [[1]]
        mx, my, mw, mh = cv2.boundingRect(nz)
        
        cols = max(1, min(5, int(round(mw/unit))))
        rows = max(1, min(5, int(round(mh/unit))))
        
        # 使用 linspace 精確分配像素區間，避開 float 累積誤差
        col_bounds = np.linspace(0, mw, cols + 1).astype(int)
        row_bounds = np.linspace(0, mh, rows + 1).astype(int)
        
        grid = [[0]*cols for _ in range(rows)]
        
        # 在方塊上方畫一個小圓點，顯示目前採樣到的全域背景色
        cv2.circle(self.img_debug, (ox + 10, oy - 15), 6, bg_color.tolist(), -1)
        cv2.circle(self.img_debug, (ox + 10, oy - 15), 6, (255,255,255), 1)

        for r in range(rows):
            for c in range(cols):
                c_s, c_e = col_bounds[c], col_bounds[c+1]
                r_s, r_e = row_bounds[r], row_bounds[r+1]
                
                # 採樣單元格中心的 20% 區域 (Median 採樣)
                cx_s, cx_e = c_s + int(0.4 * (c_e-c_s)), c_s + int(0.6 * (c_e-c_s))
                cy_s, cy_e = r_s + int(0.4 * (r_e-r_s)), r_s + int(0.6 * (r_e-r_s))
                patch = color_roi[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
                
                if patch.size > 0:
                    current_median = np.median(patch, axis=(0, 1))
                    dist = np.linalg.norm(current_median - bg_color)
                    is_p = dist > 40 # 顏色與背景差距
                else:
                    is_p = False
                
                grid[r][c] = 1 if is_p else 0

                # 繪製 Debug：白色(有), 灰色(無)
                sx, sy, ex, ey = ox+mx+c_s, oy+my+r_s, ox+mx+c_e, oy+my+r_e
                cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), (80,80,80), 1)
                fill_color = (255,255,255) if is_p else (120,120,120)
                # 實心填滿判定區域
                cv2.rectangle(self.img_debug, (ox+mx+cx_s, oy+my+cy_s), (ox+mx+cx_e, oy+my+cy_e), fill_color, -1)
                
        return grid

    # --- 座標變換工具 ---
    def lerp(self, p1, p2, t): 
        return p1 + (p2 - p1) * t
        
    def get_p(self, pts, row, col):
        top = self.lerp(pts[0], pts[1], col/8.0)
        bot = self.lerp(pts[3], pts[2], col/8.0)
        return self.lerp(top, bot, row/8.0)
        
    def get_cell_poly(self, pts, r, c):
        return np.array([self.get_p(pts,r,c), self.get_p(pts,r,c+1), self.get_p(pts,r+1,c+1), self.get_p(pts,r+1,c)], dtype=np.int32)
        
    def get_cell_poly_sampling(self, pts, r, c, s, e):
        return np.array([self.get_p(pts,r+s,c+s), self.get_p(pts,r+s,c+e), self.get_p(pts,r+e,c+e), self.get_p(pts,r+e,c+s)], dtype=np.int32)
        
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

class LogicSolver:
    """Block Blast 自動解題引擎"""
    def solve(self, grid, pieces, p_indices, path=[]):
        if not p_indices: return path
        for i in p_indices:
            p = pieces[i]
            for r in range(8):
                for c in range(8):
                    if self.can_place(grid, p, r, c):
                        # 模擬放置後的結果
                        placed_grid = self.place_only(grid, p, r, c)
                        rows_cl, cols_cl = self.get_cleared(placed_grid)
                        next_grid = self.simulate(grid, p, r, c)
                        
                        res = self.solve(next_grid, pieces, [idx for idx in p_indices if idx != i], 
                                         path + [(i, r, c, rows_cl, cols_cl)])
                        if res: return res
        return None

    def can_place(self, grid, p, r, c):
        for pr in range(len(p)):
            for pc in range(len(p[0])):
                if p[pr][pc]:
                    tr, tc = r + pr, c + pc
                    if tr < 0 or tr >= 8 or tc < 0 or tc >= 8 or grid[tr][tc]: return False
        return True

    def place_only(self, grid, p, r, c):
        ng = copy.deepcopy(grid)
        for pr in range(len(p)):
            for pc in range(len(p[0])):
                if p[pr][pc]: ng[r+pr][c+pc] = 1
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
