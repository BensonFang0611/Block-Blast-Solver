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
        # 1. 影像預處理
        hsv = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        v_channel = np.maximum.reduce([s, v])
        blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
        # Canny 能精準抓出物體的輪廓邊緣，比單純的二值化更適合用來找直線
        edges = cv2.Canny(blur, 50, 150)

        # 初始化 Debug 圖
        self.img_debug = self.img_orig.copy()

        # 2. 定位棋盤（霍夫直線轉換法）
        # 參數：解析度 1 像素, 角度解析度 1 度, 門檻值 100, 最短線長 100, 最大斷線間隙 50 (能接合 50 像素的斷點)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

        if lines is None:
            return False

        horiz_y = [] # 收集所有水平線的 Y 座標
        vert_x = []  # 收集所有垂直線的 X 座標

        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 判斷是否為「水平線」 (Y 座標差異很小)
            if abs(y1 - y2) < 15: 
                horiz_y.extend([y1, y2])
            # 判斷是否為「垂直線」 (X 座標差異很小)
            elif abs(x1 - x2) < 15:
                vert_x.extend([x1, x2])

        # 如果沒抓到足夠的水平或垂直線，代表沒找到框
        if not horiz_y or not vert_x:
            return False

        # 💡 終極組合：取最極端的座標來構成最大的外圍矩形
        min_x, max_x = min(vert_x), max(vert_x)
        min_y, max_y = min(horiz_y), max(horiz_y)

        w = max_x - min_x
        h = max_y - min_y
        area = w * h

        # 過濾太小的雜訊（至少佔畫面 10%）
        if area < (v_channel.shape[0] * v_channel.shape[1] * 0.1): 
            return False

        # 檢查長寬比，確保它是正方形的遊戲盤面 (寬容度 0.8 ~ 1.2)
        aspect_ratio = float(w) / h
        if 0.8 <= aspect_ratio <= 1.2:
            approx = np.array([
                [[min_x, min_y]],
                [[min_x, min_y + h]],
                [[min_x + w, min_y + h]],
                [[min_x + w, min_y]]
            ], dtype=np.int32)
            
            # 配合後續的透視變換邏輯，將結果包裝成 best_cand
            best_cand = {'area': area, 'approx': approx}
        else:
            return False
        
        # 排序：面積由大到小
        candidates = sorted(candidates, key=lambda x: x['area'], reverse=True)
        
        # ✨ 智慧選框邏輯（抓取內框）：
        # 如果最大框裡面還有一個面積超過它 80% 的子框，就選那個較小的子框
        best_cand = candidates[0]
        for cand in candidates[1:]:
            if cand['area'] > best_cand['area'] * 0.80:
                best_cand = cand
        
        pts1 = self.order_points(best_cand['approx'].reshape(4, 2))
        orig_unit = np.linalg.norm(pts1[0] - pts1[1]) / 8.0 
        
        # 進行透視變換
        M = cv2.getPerspectiveTransform(pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        self.warp_orig = cv2.warpPerspective(self.img_orig, M, (400, 400))
        warp_color = self.warp_orig
        
        # 3. 採樣 8x8 棋盤底色（直接取原圖 V 通道法）
        # 💡 直接將彩色原圖轉 HSV 取得明度，完全不需要任何多餘的預處理
        warp_hsv = cv2.cvtColor(warp_color, cv2.COLOR_BGR2HSV)
        
        u = 400 / 8
        centers_v = [] 
        
        for r in range(8):
            for c in range(8):
                cx, cy = int((c + 0.5) * u), int((r + 0.5) * u)
                # 直接抓取彩色原圖中心 4x4 區域的 V 通道
                roi_v = warp_hsv[cy-2:cy+2, cx-2:cx+2, 2]
                centers_v.append(np.median(roi_v) if roi_v.size > 0 else 0)

        # 🎯 空背景一定是全盤最暗的，直接取 min() 作為底色基準
        base_bg_v = min(centers_v)
        
        # 設定 3% 寬容度 (255 * 0.03 = 7.65)
        tolerance = 255 * 0.03

        # 判定棋盤方塊狀態
        for r in range(8):
            for c in range(8):
                # 只要比底色亮超過 3%，就百分之百是方塊！
                is_p = centers_v[r*8+c] > (base_bg_v + tolerance)
                self.grid_state[r][c] = 1 if is_p else 0

        # 找出這 64 格當中的「最低明度」作為底色基準
        base_bg_v = min(centers_v)
        
        # 設定容許範圍：底色明度 + 3% (255 的 3% 約為 7.65)
        tolerance = 255 * 0.03

        # 判定棋盤方塊狀態
        for r in range(8):
            for c in range(8):
                # 如果該格明度大於「底色 + 3%」，代表它是亮色的待放方塊
                is_p = centers_v[r*8+c] > (base_bg_v + tolerance)
                self.grid_state[r][c] = 1 if is_p else 0

        # Debug：繪製棋盤判定狀態
        for r in range(8):
            for c in range(8):
                is_p = self.grid_state[r][c] == 1
                cv2.polylines(self.img_debug, [self.get_cell_poly(pts1, r, c)], True, (80,80,80), 1)
                color_fill = (255,255,255) if is_p else (120,120,120)
                cv2.fillPoly(self.img_debug, [self.get_cell_poly_sampling(pts1, r, c, 0.4, 0.6)], color_fill)
        # 4. 全域採樣待放區背景色（取三個方塊間共通的空白處面積）
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(pts1[:, 1]))
        ay_s, ay_e = bottom_y + 40, int(img_h * 0.88)
        piece_area_mask = thresh[ay_s:ay_e, :]
        piece_area_color = self.img_orig[ay_s:ay_e, :]
        
        # 腐蝕遮罩以獲取純淨背景
        bg_mask = cv2.bitwise_not(piece_area_mask)
        bg_mask = cv2.erode(bg_mask, np.ones((5,5), np.uint8), iterations=2)
        bg_pixels = piece_area_color[bg_mask == 255]
        global_bg_color = np.median(bg_pixels, axis=0) if len(bg_pixels) > 100 else piece_area_color[5, 5]

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
        
        # 在方塊上方標示背景色參考點
        cv2.circle(self.img_debug, (ox + 10, oy - 15), 6, bg_color.tolist(), -1)
        cv2.circle(self.img_debug, (ox + 10, oy - 15), 6, (255,255,255), 1)

        for r in range(rows):
            for c in range(cols):
                c_s, c_e, r_s, r_e = col_b[c], col_b[c+1], row_b[r], row_b[r+1]
                # 中心區域中位數判定
                cx_s, cx_e = c_s + int(0.4* (c_e-c_s)), c_s + int(0.6* (c_e-c_s))
                cy_s, cy_e = r_s + int(0.4* (r_e-r_s)), r_s + int(0.6* (r_e-r_s))
                patch = color_roi[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
                
                dist = np.linalg.norm(np.median(patch, axis=(0,1)) - bg_color) if patch.size > 0 else 0
                is_p = dist > 40
                grid[r][c] = 1 if is_p else 0

                # 繪製 Debug
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
