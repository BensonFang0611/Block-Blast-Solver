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
        _, _, v_channel = cv2.split(hsv)
        blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        piece_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.img_debug = self.img_orig.copy()

        # ==========================================
        # 2. 定位棋盤（亞像素質心擬合 + 對比視覺化）
        # ==========================================
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

        # 建立 90% 純淨區 (上下左右各往內縮 5%，避開邊緣特效)
        sx, sy = int(bx + bw * 0.05), int(by + bh * 0.05)
        sw, sh = int(bw * 0.90), int(bh * 0.90)

        # 💡 [ Debug 視覺化 ]：在全域 Debug 圖上框出粗略大框與 ROI 隔離區
        cv2.rectangle(self.img_debug, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.rectangle(self.img_debug, (sx, sy), (sx + sw, sy + sh), (255, 50, 50), 2)

        # 提取乾淨的水平與垂直線
        kernel_h, kernel_v = np.ones((1, 51), np.uint8), np.ones((51, 1), np.uint8)
        thresh_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        thresh_h = cv2.morphologyEx(thresh_h, cv2.MORPH_CLOSE, kernel_h)
        cv2.imshow("Horizontal Lines", cv2.resize(thresh_h[sy:sy+sh, sx:sx+sw], (0, 0), fx=0.5, fy=0.5))
        thresh_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        thresh_v = cv2.morphologyEx(thresh_v, cv2.MORPH_CLOSE, kernel_v)
        cv2.imshow("Vertical Lines", cv2.resize(thresh_v[sy:sy+sh, sx:sx+sw], (0, 0), fx=0.5, fy=0.5))
        # 💡 [關鍵架構修改]：不需要額外複製 img_roi_color 了，我們直接畫在 self.img_debug 上！
        # (你可以把原本的 img_roi_color = ... 那行刪掉)
        
        proj_y = np.sum(thresh_h[sy:sy+sh, sx:sx+sw], axis=1) / 255
        proj_x = np.sum(thresh_v[sy:sy+sh, sx:sx+sw], axis=0) / 255

        def get_exact_edges_from_roi_debug(projection, offset_start, rough_min, is_horizontal):
            """ 質心平均 + 線性迴歸最優擬合 (保留所有線條，交由數學平衡) """
            if len(projection) == 0: return None, None
            
            threshold = np.max(projection) * 0.50 
            valid_coords = np.where(projection > threshold)[0]
            if len(valid_coords) < 3: return None, None

            # [步驟 1A]：初次質心平均
            peaks_roi = []
            current_group = [valid_coords[0]]
            for i in range(1, len(valid_coords)):
                if valid_coords[i] - valid_coords[i-1] <= 15:
                    current_group.append(valid_coords[i])
                else:
                    weights = projection[current_group]
                    roi_center = np.average(current_group, weights=weights)
                    peaks_roi.append(roi_center)
                    current_group = [valid_coords[i]]
            weights = projection[current_group]
            roi_center = np.average(current_group, weights=weights)
            peaks_roi.append(roi_center)

            if len(peaks_roi) < 3: return None, None

            diffs = np.diff(peaks_roi)
            valid_diffs = [d for d in diffs if d > 20]
            if not valid_diffs: return None, None
            u_est = np.median(valid_diffs)

            # [步驟 1B]：二次強制合併 (不刪除任何線條)
            indices = [0]
            clean_peaks = [peaks_roi[0]]
            for i in range(1, len(peaks_roi)):
                diff_u = (peaks_roi[i] - clean_peaks[-1]) / u_est
                idx_diff = int(round(diff_u))
                
                # 解決雙眼皮：如果距離小於半格，強制融合成一條質心
                if diff_u < 0.5:
                    clean_peaks[-1] = (clean_peaks[-1] + peaks_roi[i]) / 2.0
                    continue

                # 💡 移除極端值剃除：所有線條全數保留，忠實記錄並編號！
                indices.append(indices[-1] + idx_diff)
                clean_peaks.append(peaks_roi[i])

            if len(clean_peaks) < 3: return None, None

            # [步驟 1C - 視覺化 Debug]：直接畫在 self.img_debug (黃色細線)
            color_centroid = (0, 255, 255) # BGR 黃色
            for p_roi in clean_peaks:
                p_global = int(round(p_roi + offset_start)) 
                if is_horizontal: 
                    cv2.line(self.img_debug, (sx, p_global), (sx + sw // 2, p_global), color_centroid, 1, cv2.LINE_AA)
                else: 
                    cv2.line(self.img_debug, (p_global, sy), (p_global, sy + sh // 2), color_centroid, 1, cv2.LINE_AA)

            # [步驟 2]：線性迴歸擬合 (讓所有保留下來的線條共同決定出最完美的 u_opt)
            u_opt, offset_0_roi = np.polyfit(indices, clean_peaks, 1)

            # [步驟 2C - 視覺化 Debug]：直接畫在 self.img_debug (紅色粗線)
            color_optimal = (0, 0, 255) # BGR 紅色
            for k in range(0, 9): 
                line_p_roi = offset_0_roi + k * u_opt
                p_global = int(round(line_p_roi + offset_start)) 
                
                if is_horizontal:
                    if sy <= p_global <= sy + sh:
                        cv2.line(self.img_debug, (sx + sw // 2, p_global), (sx + sw, p_global), color_optimal, 2, cv2.LINE_AA)
                else:
                    if sx <= p_global <= sx + sw:
                        cv2.line(self.img_debug, (p_global, sy + sh // 2), (p_global, sy + sh), color_optimal, 2, cv2.LINE_AA)

            offset_0_global = offset_0_roi + offset_start
            line_index = int(round((offset_0_global - rough_min) / u_opt))
            exact_min = offset_0_global - line_index * u_opt
            exact_max = exact_min + 8 * u_opt
            
            return exact_min, exact_max

        # ==========================================
        # 呼叫與保險機制 (移除 cv2.imshow 彈出視窗)
        # ==========================================
        # 分別解析 Y 軸與 X 軸 (參數減少，不再傳入 img_roi)
        min_y, max_y = get_exact_edges_from_roi_debug(proj_y, sy, by, is_horizontal=True)
        min_x, max_x = get_exact_edges_from_roi_debug(proj_x, sx, bx, is_horizontal=False)

        # 💡 [移除]：刪除了 cv2.imshow("Centroid(Yellow) vs LeastSquares(Red)", ...)

        # 萬一真的被炸到連 3 條內線都找不到，保險機制啟動 (往內縮 3%)
        if None in (min_x, max_x, min_y, max_y):
            min_x, max_x = bx + bw * 0.03, bx + bw * 0.97
            min_y, max_y = by + bh * 0.03, by + bh * 0.97

        # 進行透視變換
        approx = np.array([[[min_x, min_y]], [[min_x, max_y]], [[max_x, max_y]], [[max_x, min_y]]], dtype=np.int32)
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
