import cv2
import numpy as np
import copy

class VisionEngine:
    def __init__(self, cv_img):
        self.img_orig = cv_img
        self.img_debug = None           # 舊版二值化 Debug 圖
        self.img_debug_color = None     # 新版全彩 Debug 圖
        self.grid_state = [[0]*8 for _ in range(8)]
        self.detected_pieces = []
        self.warp_orig = None
        self.piece_scale = 0.50 
        
        # 統計數據
        self.thresh_count = 0
        self.color_count = 0
        self.chosen_method = ""

    def process(self):
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 初始化兩張不同的 Debug 圖
        self.img_debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.img_debug_color = self.img_orig.copy()

        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or hierarchy is None: return False
        
        hierarchy = hierarchy[0]
        candidates = []
        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area < (gray.shape[0] * gray.shape[1] * 0.1): continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                candidates.append({'index': i, 'area': area, 'approx': approx, 'parent': hierarchy[i][3]})
        
        if not candidates: return False
        candidates = sorted(candidates, key=lambda x: x['area'], reverse=True)
        
        best_cand = candidates[0]
        for cand in candidates[1:]:
            if cand['area'] > best_cand['area'] * 0.80: best_cand = cand
        
        approx = best_cand['approx']
        pts1 = self.order_points(approx.reshape(4, 2))

        orig_board_w = np.linalg.norm(pts1[0] - pts1[1])
        orig_unit = orig_board_w / 8.0 
        
        M = cv2.getPerspectiveTransform(pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        self.warp_orig = cv2.warpPerspective(self.img_orig, M, (400, 400))
        warp_loc = cv2.warpPerspective(thresh, M, (400, 400))
        warp_color = cv2.warpPerspective(self.img_orig, M, (400, 400))
        
        u = 400 / 8
        centers_color = []
        
        # [階段 A] 收集顏色找底色
        for r in range(8):
            for c in range(8):
                cx_s, cx_e = int((c + 0.45) * u), int((c + 0.55) * u)
                cy_s, cy_e = int((r + 0.45) * u), int((r + 0.55) * u)
                roi_c = warp_color[cy_s:cy_e, cx_s:cx_e]
                mean_c = np.mean(roi_c, axis=(0, 1)) if roi_c.size > 0 else np.array([0,0,0])
                centers_color.append((r, c, mean_c))

        color_counts = {}
        for r, c, mc in centers_color:
            q = (int(mc[0]/20), int(mc[1]/20), int(mc[2]/20))
            color_counts[q] = color_counts.get(q, 0) + 1
        
        top_qs = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        bg_q = min(top_qs, key=lambda x: sum(x[0]))[0]
        bg_color_mean = np.mean([mc for r, c, mc in centers_color if (int(mc[0]/20), int(mc[1]/20), int(mc[2]/20)) == bg_q], axis=0)

        grid_thresh = [[0]*8 for _ in range(8)]
        grid_color = [[0]*8 for _ in range(8)]
        
        for r in range(8):
            for c in range(8):
                # 二值化判定
                x_s, x_e = int((c + 0.15) * u), int((c + 0.85) * u)
                y_s, y_e = int((r + 0.15) * u), int((r + 0.85) * u)
                roi_t = warp_loc[y_s:y_e, x_s:x_e]
                white_ratio = cv2.countNonZero(roi_t) / roi_t.size if roi_t.size > 0 else 0
                grid_thresh[r][c] = 1 if white_ratio > 0.01 else 0

                # 顏色判定
                mc = centers_color[r*8 + c][2]
                dist = np.linalg.norm(mc - bg_color_mean)
                grid_color[r][c] = 1 if dist > 45 else 0

        sum_t = sum(sum(row) for row in grid_thresh)
        sum_c = sum(sum(row) for row in grid_color)
        
        self.thresh_count += sum_t
        self.color_count += sum_c
        
        if sum_c > sum_t:
            self.grid_state = grid_color
            self.chosen_method = "新版顏色判定 (Color)"
        else:
            self.grid_state = grid_thresh
            self.chosen_method = "舊版二值化判定 (Threshold)"

        # 分別繪製 8x8 格線狀態到各自的 Debug 圖
        for r in range(8):
            for c in range(8):
                # 畫二值化圖
                is_t = grid_thresh[r][c] == 1
                inner_cp_t = self.get_cell_poly_sampling(pts1, r, c, 0.15, 0.85)
                color_t = (0, 0, 255) if is_t else (0, 0, 200)
                cv2.polylines(self.img_debug, [self.get_cell_poly(pts1, r, c)], True, (0, 150, 0), 1)
                cv2.polylines(self.img_debug, [inner_cp_t], True, color_t, 2 if is_t else 1)

                # 畫全彩顏色圖
                is_c = grid_color[r][c] == 1
                inner_cp_c = self.get_cell_poly_sampling(pts1, r, c, 0.45, 0.55) # 中心 10%
                color_c = (0, 0, 255) if is_c else (0, 0, 200)
                cv2.polylines(self.img_debug_color, [self.get_cell_poly(pts1, r, c)], True, (0, 150, 0), 1)
                cv2.polylines(self.img_debug_color, [inner_cp_c], True, color_c, 2 if is_c else 1)

        # ====== 4. 找待放物區域 ======
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(pts1[:, 1]))
        start_y = bottom_y + 40
        end_y = int(img_h * 0.88) 
        piece_area_mask = thresh[start_y:end_y, :]
        p_cnts, _ = cv2.findContours(piece_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates_p = []
        for cnt in p_cnts:
            if cv2.contourArea(cnt) < (orig_unit**2) * 0.2: continue 
            x, y, pw, ph = cv2.boundingRect(cnt)
            if pw > 6*orig_unit or ph > 6*orig_unit or max(pw/ph, ph/pw) > 6.0: continue
            candidates_p.append([x, y + start_y, pw, ph, x + pw/2])

        candidates_p = sorted(candidates_p, key=lambda p: p[0])
        filtered_pieces = []
        for cand in candidates_p:
            is_dup = False
            for accepted in filtered_pieces:
                if abs(cand[4] - accepted[4]) < (orig_unit * 0.5):
                    is_dup = True; break
            if not is_dup: filtered_pieces.append(cand)
        
        final_list = filtered_pieces[:3]
        p_unit = orig_unit * self.piece_scale
        
        for x, ay, pw, ph, _ in final_list:
            mask = thresh[ay:ay+ph, x:x+pw]
            color_roi = self.img_orig[ay:ay+ph, x:x+pw]
            self.detected_pieces.append(self.parse_piece_strict(mask, color_roi, pw, ph, p_unit, x, ay))
        
        cv2.polylines(self.img_debug, [pts1.astype(int)], True, (0, 255, 0), 3)
        cv2.polylines(self.img_debug_color, [pts1.astype(int)], True, (0, 255, 0), 3)
        return True

    def parse_piece_strict(self, mask, color_roi, pw, ph, unit, ox, oy):
        nz = cv2.findNonZero(mask)
        if nz is None: return [[1]]
        mx, my, mw, mh = cv2.boundingRect(nz)
        
        cols = max(1, min(5, int(round(mw / unit))))
        rows = max(1, min(5, int(round(mh / unit))))
        cw, ch = mw / cols, mh / rows
        
        grid_t = [[0]*cols for _ in range(rows)]
        grid_c = [[0]*cols for _ in range(rows)]
        
        bg_mask = cv2.bitwise_not(mask[0:min(20, ph), 0:min(20, pw)])
        bg_pixels = color_roi[0:min(20, ph), 0:min(20, pw)][bg_mask == 255]
        if len(bg_pixels) > 0:
            bg_color = np.mean(bg_pixels, axis=0)
        else:
            bg_color = np.mean(color_roi[0:5, 0:5], axis=(0,1))
            
        for r in range(rows):
            for c in range(cols):
                # --- 1. 二值化判定 ---
                rx_s, rx_e = int(mx + (c + 0.0) * cw), int(mx + (c + 1.0) * cw)
                ry_s, ry_e = int(my + (r + 0.0) * ch), int(my + (r + 1.0) * ch)
                roi = mask[ry_s:ry_e, rx_s:rx_e]
                
                is_p_t = False
                if roi.size > 4:
                    h, w = roi.shape
                    mid_h, mid_w = h // 2, w // 2
                    q = [roi[0:mid_h, 0:mid_w], roi[0:mid_h, mid_w:w], roi[mid_h:h, 0:mid_w], roi[mid_h:h, mid_w:w]]
                    def check(sub): return (cv2.countNonZero(sub)/sub.size > 0.01) if sub.size > 0 else False
                    is_p_t = all([check(sub) for sub in q])
                grid_t[r][c] = 1 if is_p_t else 0

                # --- 2. 顏色判定 ---
                cx_s, cx_e = int(mx + (c + 0.45) * cw), int(mx + (c + 0.55) * cw)
                cy_s, cy_e = int(my + (r + 0.45) * ch), int(my + (r + 0.55) * ch)
                cx_e = max(cx_s+1, cx_e); cy_e = max(cy_s+1, cy_e)
                patch = color_roi[cy_s:cy_e, cx_s:cx_e]
                
                is_p_c = False
                if patch.size > 0:
                    patch_mean = np.mean(patch, axis=(0,1))
                    dist = np.linalg.norm(patch_mean - bg_color)
                    is_p_c = dist > 45
                grid_c[r][c] = 1 if is_p_c else 0

        sum_t = sum(sum(row) for row in grid_t)
        sum_c = sum(sum(row) for row in grid_c)
        self.thresh_count += sum_t
        self.color_count += sum_c

        # 將兩種結果各自畫到對應的 Debug 圖上
        for r in range(rows):
            for c in range(cols):
                # 畫二值化圖
                is_p_t = grid_t[r][c]
                rx_s, rx_e = int(mx + (c + 0.0) * cw), int(mx + (c + 1.0) * cw)
                ry_s, ry_e = int(my + (r + 0.0) * ch), int(my + (r + 1.0) * ch)
                sx, sy, ex, ey = ox + rx_s, oy + ry_s, ox + rx_e, oy + ry_e
                color_t = (0, 0, 255) if is_p_t else (0, 0, 200)
                cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), color_t, 2 if is_p_t else 1)
                
                # 畫全彩顏色圖
                is_p_c = grid_c[r][c]
                cx_s, cx_e = int(mx + (c + 0.45) * cw), int(mx + (c + 0.55) * cw)
                cy_s, cy_e = int(my + (r + 0.45) * ch), int(my + (r + 0.55) * ch)
                color_c = (0, 0, 255) if is_p_c else (0, 0, 200)
                cv2.rectangle(self.img_debug_color, (ox+cx_s, oy+cy_s), (ox+max(cx_s+1, cx_e), oy+max(cy_s+1, cy_e)), color_c, 2 if is_p_c else 1)

        final_grid = grid_c if sum_c > sum_t else grid_t
        return final_grid

# (下方保留原本的數學工具和 LogicSolver 即可)

    # --- 數學工具 ---
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
                        placed = self.place_only(grid, p, r, c)
                        rows, cols = self.get_cleared(placed)
                        next_g = self.simulate(grid, p, r, c)
                        res = self.solve(next_g, pieces, [idx for idx in p_indices if idx != i], path + [(i, r, c, rows, cols)])
                        if res: return res
        return None

    def can_place(self, grid, p, r, c):
        for pr in range(len(p)):
            for pc in range(len(p[0])):
                if p[pr][pc] == 1:
                    tr, tc = r + pr, c + pc
                    if tr < 0 or tr >= 8 or tc < 0 or tc >= 8 or grid[tr][tc] == 1: return False
        return True

    def place_only(self, grid, p, r, c):
        ng = copy.deepcopy(grid)
        for pr in range(len(p)):
            for pc in range(len(p[0])):
                if p[pr][pc] == 1: ng[r+pr][c+pc] = 1
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
