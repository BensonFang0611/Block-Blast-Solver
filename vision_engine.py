import cv2
import numpy as np
import copy

class VisionEngine:
    def __init__(self, cv_img):
        self.img_orig = cv_img
        self.img_debug = None           # 舊版二值化 Debug 圖
        self.img_debug_color = None     # 新版全彩顏色 Debug 圖
        self.grid_state = [[0]*8 for _ in range(8)]
        self.detected_pieces = []
        self.warp_orig = None
        self.piece_scale = 0.50 
        
        # 統計數據
        self.thresh_count = 0
        self.color_count = 0
        self.chosen_method = ""

    def process(self):
        # 1. 影像預處理
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        self.img_debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.img_debug_color = self.img_orig.copy()

        # 2. 智慧定位棋盤
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or hierarchy is None: return False
        
        hierarchy = hierarchy[0]
        candidates = []
        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area < (gray.shape[0] * gray.shape[1] * 0.1): continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                candidates.append({'index': i, 'area': area, 'approx': approx})
        
        if not candidates: return False
        candidates = sorted(candidates, key=lambda x: x['area'], reverse=True)
        best_cand = candidates[0]
        for cand in candidates[1:]:
            if cand['area'] > best_cand['area'] * 0.80: best_cand = cand
        
        approx = best_cand['approx']
        pts1 = self.order_points(approx.reshape(4, 2))
        orig_unit = np.linalg.norm(pts1[0] - pts1[1]) / 8.0 
        
        M = cv2.getPerspectiveTransform(pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        self.warp_orig = cv2.warpPerspective(self.img_orig, M, (400, 400))
        warp_loc = cv2.warpPerspective(thresh, M, (400, 400))
        warp_color = cv2.warpPerspective(self.img_orig, M, (400, 400))
        
        # 3. 棋盤底色採樣 (8x8 區域)
        u = 400 / 8
        centers_color = []
        for r in range(8):
            for c in range(8):
                cx, cy = int((c + 0.5) * u), int((r + 0.5) * u)
                roi_c = warp_color[cy-2:cy+2, cx-2:cx+2]
                centers_color.append(np.median(roi_c, axis=(0, 1)) if roi_c.size > 0 else [0,0,0])

        color_counts = {}
        for mc in centers_color:
            q = (int(mc[0]/20), int(mc[1]/20), int(mc[2]/20))
            color_counts[q] = color_counts.get(q, 0) + 1
        bg_q = max(color_counts, key=color_counts.get)
        board_bg_color = np.mean([mc for mc in centers_color if (int(mc[0]/20), int(mc[1]/20), int(mc[2]/20)) == bg_q], axis=0)

        # 4. 棋盤狀態判定
        grid_thresh = [[0]*8 for _ in range(8)]
        grid_color = [[0]*8 for _ in range(8)]
        s_bt, s_bc = 0, 0
        for r in range(8):
            for c in range(8):
                if cv2.countNonZero(warp_loc[int((r+0.15)*u):int((r+0.85)*u), int((c+0.15)*u):int((c+0.85)*u)]) / (0.7*u)**2 > 0.01:
                    grid_thresh[r][c] = 1; s_bt += 1
                if np.linalg.norm(centers_color[r*8+c] - board_bg_color) > 45:
                    grid_color[r][c] = 1; s_bc += 1

        # ====== 🌟 待放物全域背景色採樣邏輯 ======
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(pts1[:, 1]))
        start_y, end_y = bottom_y + 40, int(img_h * 0.88)
        piece_area_mask = thresh[start_y:end_y, :]
        piece_area_color = self.img_orig[start_y:end_y, :]
        
        # 腐蝕遮罩以獲取純淨背景
        bg_mask = cv2.bitwise_not(piece_area_mask)
        bg_mask = cv2.erode(bg_mask, np.ones((5,5), np.uint8), iterations=2)
        bg_pixels = piece_area_color[bg_mask == 255]
        global_bg_color = np.median(bg_pixels, axis=0) if len(bg_pixels) > 100 else piece_area_color[5, 5]

        p_cnts, _ = cv2.findContours(piece_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates_p = []
        for cnt in p_cnts:
            if cv2.contourArea(cnt) < (orig_unit**2) * 0.2: continue
            x, y, pw, ph = cv2.boundingRect(cnt)
            if pw > 6*orig_unit or ph > 6*orig_unit: continue
            candidates_p.append([x, y + start_y, pw, ph, x + pw/2])

        final_list = sorted(candidates_p, key=lambda p: p[0])[:3]
        p_unit = orig_unit * self.piece_scale
        
        temp_pieces_t, temp_pieces_c = [], []
        s_pt, s_pc = 0, 0
        for x, ay, pw, ph, _ in final_list:
            mask = thresh[ay:ay+ph, x:x+pw]
            color_roi = self.img_orig[ay:ay+ph, x:x+pw]
            g_t, g_c, st, sc = self.parse_piece_strict(mask, color_roi, pw, ph, p_unit, x, ay, global_bg_color)
            temp_pieces_t.append(g_t); temp_pieces_c.append(g_c)
            s_pt += st; s_pc += sc
            
        # 結算與繪圖
        self.thresh_count, self.color_count = s_bt + s_pt, s_bc + s_pc
        if self.color_count >= self.thresh_count:
            self.grid_state, self.detected_pieces, self.chosen_method = grid_color, temp_pieces_c, "顏色判定 (Color Mode)"
        else:
            self.grid_state, self.detected_pieces, self.chosen_method = grid_thresh, temp_pieces_t, "二值化判定 (Thresh Mode)"

        # 繪製 Debug 圖 (略，詳見 parse 函數內的繪圖)
        for r in range(8):
            for c in range(8):
                is_t, is_c = grid_thresh[r][c], grid_color[r][c]
                cv2.polylines(self.img_debug, [self.get_cell_poly(pts1, r, c)], True, (0, 150, 0), 1)
                cv2.polylines(self.img_debug, [self.get_cell_poly_sampling(pts1, r, c, 0.15, 0.85)], True, (0,0,255 if is_t else 200), 2 if is_t else 1)
                cv2.polylines(self.img_debug_color, [self.get_cell_poly(pts1, r, c)], True, (80,80,80), 1)
                cv2.fillPoly(self.img_debug_color, [self.get_cell_poly_sampling(pts1, r, c, 0.4, 0.6)], (255,255,255) if is_c else (120,120,120))

        return True

    def parse_piece_strict(self, mask, color_roi, pw, ph, unit, ox, oy, bg_color):
        nz = cv2.findNonZero(mask)
        if nz is None: return [[1]], [[1]], 1, 1
        mx, my, mw, mh = cv2.boundingRect(nz)
        cols, rows = max(1, min(5, int(round(mw/unit)))), max(1, min(5, int(round(mh/unit))))
        col_b, row_b = np.linspace(0, mw, cols+1).astype(int), np.linspace(0, mh, rows+1).astype(int)
        grid_t, grid_c = [[0]*cols for _ in range(rows)], [[0]*cols for _ in range(rows)]
        s_t, s_c = 0, 0
        
        cv2.circle(self.img_debug_color, (ox + 5, oy - 10), 5, bg_color.tolist(), -1)
        
        for r in range(rows):
            for c in range(cols):
                c_s, c_e, r_s, r_e = col_b[c], col_b[c+1], row_b[r], row_b[r+1]
                roi_m = mask[r_s:r_e, c_s:c_e]
                if roi_m.size > 4 and all([(cv2.countNonZero(sub)/sub.size > 0.01 if sub.size>0 else 0) for sub in [roi_m[0:r_e-r_s//2, 0:c_e-c_s//2]]]):
                    grid_t[r][c] = 1; s_t += 1
                
                patch = color_roi[r_s+int(0.4*(r_e-r_s)):r_s+int(0.6*(r_e-r_s)), c_s+int(0.4*(c_e-c_s)):c_s+int(0.6*(c_e-c_s))]
                if patch.size > 0 and np.linalg.norm(np.median(patch, axis=(0,1)) - bg_color) > 40:
                    grid_c[r][c] = 1; s_c += 1

                sx, sy, ex, ey = ox+mx+c_s, oy+my+r_s, ox+mx+c_e, oy+my+r_e
                cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), (0,0,255 if grid_t[r][c] else 200), 1)
                cv2.rectangle(self.img_debug_color, (sx, sy), (ex, ey), (80,80,80), 1)
                cv2.rectangle(self.img_debug_color, (ox+mx+c_s+int(0.4*(c_e-c_s)), oy+my+r_s+int(0.4*(r_e-r_s))), (ox+mx+c_s+int(0.6*(c_e-c_s)), oy+my+r_s+int(0.6*(r_e-r_s))), (255,255,255) if grid_c[r][c] else (120,120,120), -1)
        return grid_t, grid_c, s_t, s_c

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
                        res = self.solve(next_g, pieces, [idx for idx in p_indices if idx != i], path + [(i, r, c, *self.get_cleared(self.place_only(grid, p, r, c)))])
                        if res: return res
        return None

    def can_place(self, grid, p, r, c):
        for pr in range(len(p)):
            for pc in range(len(p[0])):
                if p[pr][pc] and (r+pr>=8 or c+pc>=8 or grid[r+pr][c+pc]): return False
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
