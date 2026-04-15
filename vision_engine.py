import cv2
import numpy as np
import copy

class VisionEngine:
    def __init__(self, cv_img):
        self.img_orig = cv_img
        self.img_debug = None   
        self.grid_state = [[0]*8 for _ in range(8)]
        self.detected_pieces = []
        self.warp_orig = None
        self.piece_scale = 0.50 

    def process(self):
        # 1. 影像預處理
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        self.img_debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # 2. 智慧定位：使用 RETR_TREE 抓取包含內部的所有輪廓
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or hierarchy is None: return False
        
        hierarchy = hierarchy[0]
        candidates = []
        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            # 過濾掉太小的雜訊，至少要佔畫面的 10% 面積
            if area < (gray.shape[0] * gray.shape[1] * 0.1): continue
            
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                candidates.append({
                    'index': i,
                    'area': area,
                    'approx': approx,
                    'parent': hierarchy[i][3] # 記錄父節點編號
                })
        
        if not candidates: return False
        
        # 排序：面積由大到小
        candidates = sorted(candidates, key=lambda x: x['area'], reverse=True)
        
        # ✨ 智慧選框邏輯：
        # 如果最大框(candidates[0])裡面還有一個面積超過它 80% 的子框，就選子框
        best_cand = candidates[0]
        for cand in candidates[1:]:
            if cand['area'] > best_cand['area'] * 0.80:
                # 這裡不強加父子關係限制，因為有時候二值化會把框線切斷導致層級跳位
                # 只要面積夠接近且在內部，我們就選面積小的那個（內圈）
                best_cand = cand
        
        approx = best_cand['approx']
        pts1 = self.order_points(approx.reshape(4, 2))

        # 🌟 您要求的：不要縮放，直接使用抓到的內圈點
        orig_board_w = np.linalg.norm(pts1[0] - pts1[1])
        orig_unit = orig_board_w / 8.0 
        
        M = cv2.getPerspectiveTransform(pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        self.warp_orig = cv2.warpPerspective(self.img_orig, M, (400, 400))
        warp_loc = cv2.warpPerspective(thresh, M, (400, 400))
        
        # ====== 3. 解析 8x8 內部格 (排除法) ======
        for r in range(8):
            for c in range(8):
                u = 400 / 8
                # 棋盤內部採樣維持 15%-85% 以避開邊界格線
                x_s, x_e = int((c + 0.15) * u), int((c + 0.85) * u)
                y_s, y_e = int((r + 0.15) * u), int((r + 0.85) * u)
                roi = warp_loc[y_s:y_e, x_s:x_e]
                
                white_ratio = cv2.countNonZero(roi) / roi.size if roi.size > 0 else 0
                is_piece = white_ratio > 0.01 
                self.grid_state[r][c] = 1 if is_piece else 0

                cp = self.get_cell_poly(pts1, r, c)
                cv2.polylines(self.img_debug, [cp], True, (0, 150, 0), 1) 
                
                inner_cp = self.get_cell_poly_sampling(pts1, r, c, 0.15, 0.85)
                # 亮紅 (0,0,255) 或 亮暗紅 (0,0,200)
                color = (0, 0, 255) if is_piece else (0, 0, 200)
                cv2.polylines(self.img_debug, [inner_cp], True, color, 2 if is_piece else 1)

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
            cv2.rectangle(self.img_debug, (x, ay), (x + pw, ay + ph), (255, 0, 0), 2)
            mask = thresh[ay:ay+ph, x:x+pw]
            self.detected_pieces.append(self.parse_piece_strict(mask, pw, ph, p_unit, x, ay))
        
        cv2.polylines(self.img_debug, [pts1.astype(int)], True, (0, 255, 0), 3)
        return True

    def parse_piece_strict(self, mask, pw, ph, unit, ox, oy):
        """✨ 待放物解析：四象限 0% ~ 100% 全範圍判定法"""
        nz = cv2.findNonZero(mask)
        if nz is None: return [[1]]
        mx, my, mw, mh = cv2.boundingRect(nz)
        
        cols = max(1, min(5, int(round(mw / unit))))
        rows = max(1, min(5, int(round(mh / unit))))
        cw, ch = mw / cols, mh / rows
        grid = [[0]*cols for _ in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                gx, gy = int(ox + mx + c*cw), int(oy + my + r*ch)
                cv2.rectangle(self.img_debug, (gx, gy), (gx+int(cw), gy+int(ch)), (0, 150, 0), 1)
                
                # ✨ 您要求的：偵測範圍改為 0% ~ 100%
                rx_s, rx_e = int(mx + (c + 0.0) * cw), int(mx + (c + 1.0) * cw)
                ry_s, ry_e = int(my + (r + 0.0) * ch), int(my + (r + 1.0) * ch)
                roi = mask[ry_s:ry_e, rx_s:rx_e]
                
                is_p = False
                if roi.size > 4:
                    h, w = roi.shape
                    mid_h, mid_w = h // 2, w // 2
                    q = [roi[0:mid_h, 0:mid_w], roi[0:mid_h, mid_w:w], roi[mid_h:h, 0:mid_w], roi[mid_h:h, mid_w:w]]
                    def check(sub): return (cv2.countNonZero(sub)/sub.size > 0.01) if sub.size > 0 else False
                    is_p = all([check(sub) for sub in q])

                sx, sy, ex, ey = ox + rx_s, oy + ry_s, ox + rx_e, oy + ry_e
                color = (0, 0, 255) if is_p else (0, 0, 200) # 亮紅 或 亮一點的暗紅
                if is_p:
                    grid[r][c] = 1
                    cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), color, 2)
                else:
                    cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), color, 1)
        return grid

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
