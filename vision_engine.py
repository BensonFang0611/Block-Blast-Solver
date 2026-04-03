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

    def detect_block_by_lines(self, roi):
        """✨ 放寬的線段判定邏輯"""
        # 參數調寬鬆：降低門檻(5)、允許更短的線(5)、允許線段中間有斷裂(10)
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=5, minLineLength=5, maxLineGap=10)
        if lines is None: 
            return False

        h_y_positions = []
        v_x_positions = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 容忍度放寬：容許 8 像素內的歪斜
            if abs(y1 - y2) <= 8: 
                h_y_positions.append((y1 + y2) // 2)
            elif abs(x1 - x2) <= 8: 
                v_x_positions.append((x1 + x2) // 2)

        # 判定獨立線段的距離調小為 5
        def count_distinct_lines(positions, min_dist=5):
            if not positions: return 0
            positions.sort()
            count = 1
            last_pos = positions[0]
            for p in positions[1:]:
                if p - last_pos >= min_dist:
                    count += 1
                    last_pos = p
            return count

        h_count = count_distinct_lines(h_y_positions)
        v_count = count_distinct_lines(v_x_positions)

        return h_count >= 2 and v_count >= 2

    def process(self):
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        self.img_debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # ====== 1. 定位 8x8 棋盤 (改回找輪廓角落) ======
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return False
        board_cnt = max(cnts, key=cv2.contourArea)
        
        # 使用多邊形擬合找四個角
        approx = cv2.approxPolyDP(board_cnt, 0.02 * cv2.arcLength(board_cnt, True), True)
        if len(approx) != 4: return False

        pts1 = self.order_points(approx.reshape(4, 2))

        orig_board_w = np.linalg.norm(pts1[0] - pts1[1])
        orig_unit = orig_board_w / 8.0 
        
        M = cv2.getPerspectiveTransform(pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        self.warp_orig = cv2.warpPerspective(self.img_orig, M, (400, 400))
        warp_loc = cv2.warpPerspective(thresh, M, (400, 400))
        
        # ====== 解析 8x8 內部格 ======
        for r in range(8):
            for c in range(8):
                u = 400 / 8
                # ✨ 擴大採樣區間至 5% ~ 95%
                x_s, x_e = int((c + 0.05) * u), int((c + 0.95) * u)
                y_s, y_e = int((r + 0.05) * u), int((r + 0.95) * u)
                roi = warp_loc[y_s:y_e, x_s:x_e]
                
                # 套用放寬後的線段判定邏輯
                is_piece = self.detect_block_by_lines(roi)
                self.grid_state[r][c] = 1 if is_piece else 0

                cp = self.get_cell_poly(pts1, r, c)
                cv2.polylines(self.img_debug, [cp], True, (0, 150, 0), 1) 
                
                # 繪製紅框供視覺確認
                inner_cp = self.get_cell_poly_sampling(pts1, r, c, 0.05, 0.95)
                if is_piece:
                    cv2.polylines(self.img_debug, [inner_cp], True, (0, 0, 255), 2) 
                else:
                    cv2.polylines(self.img_debug, [inner_cp], True, (0, 0, 80), 1)  

        # ====== 2. 找待放物 (NMS) ======
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(pts1[:, 1]))
        start_y = bottom_y + 40
        end_y = int(img_h * 0.88) 
        piece_area_mask = thresh[start_y:end_y, :]
        p_cnts, _ = cv2.findContours(piece_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in p_cnts:
            if cv2.contourArea(cnt) < (orig_unit**2) * 0.2: continue 
            x, y, pw, ph = cv2.boundingRect(cnt)
            if pw > 6*orig_unit or ph > 6*orig_unit or max(pw/ph, ph/pw) > 6.0: continue
            candidates.append([x, y + start_y, pw, ph, x + pw/2])

        candidates = sorted(candidates, key=lambda p: p[0])
        filtered_pieces = []
        for cand in candidates:
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
        
        # 繪製最終的綠色大外框
        cv2.drawContours(self.img_debug, [approx.reshape(-1, 2).astype(int)], -1, (0, 255, 0), 3)
        return True

    def lerp(self, p1, p2, t): return p1 + (p2 - p1) * t

    def get_p(self, pts, row, col):
        top = self.lerp(pts[0], pts[1], col/8.0); bot = self.lerp(pts[3], pts[2], col/8.0)
        return self.lerp(top, bot, row/8.0)

    def get_cell_poly(self, pts, r, c):
        return np.array([self.get_p(pts,r,c), self.get_p(pts,r,c+1), self.get_p(pts,r+1,c+1), self.get_p(pts,r+1,c)], dtype=np.int32)

    def get_cell_poly_sampling(self, pts, r, c, s, e):
        return np.array([self.get_p(pts,r+s,c+s), self.get_p(pts,r+s,c+e), self.get_p(pts,r+e,c+e), self.get_p(pts,r+e,c+s)], dtype=np.int32)

    def parse_piece_strict(self, mask, pw, ph, unit, ox, oy):
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
                
                # ✨ 待放物的採樣區間同樣改為 5% ~ 95%
                rx_s, rx_e = int(mx + (c + 0.05) * cw), int(mx + (c + 0.95) * cw)
                ry_s, ry_e = int(my + (r + 0.05) * ch), int(my + (r + 0.95) * ch)
                roi = mask[ry_s:ry_e, rx_s:rx_e]
                
                is_p = self.detect_block_by_lines(roi)
                
                sx, sy, ex, ey = ox + rx_s, oy + ry_s, ox + rx_e, oy + ry_e
                if is_p:
                    grid[r][c] = 1
                    cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), (0, 0, 255), 2)
                else:
                    cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), (0, 0, 80), 1)
        return grid

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