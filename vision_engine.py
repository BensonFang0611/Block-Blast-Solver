import cv2
import numpy as np
import copy
import os

class VisionEngine:
    def __init__(self, cv_img, base_pieces=None):
        self.img_orig = cv_img
        self.img_debug = None
        self.grid_state = [[0]*8 for _ in range(8)]
        self.detected_pieces = []
        self.warp_orig = None
        self.piece_scale = 0.46
        self.legal_shapes = set()
        self.legal_grids = []
        self._generate_rotated_pieces(base_pieces)

    def _generate_rotated_pieces(self, base_pieces):
        """ 智慧核心：傳入基本方塊原型，自動衍生出 4 個旋轉角度(0, 90, 180, 270) 的特徵 """
        if not base_pieces:
            base_pieces = [
                [[1]],                                  # 1x1 方塊
                [[1, 1]],                               # 1x2 長條
                [[1, 1, 1]],                            # 1x3 長條
                [[1, 1, 1, 1]],                         # 1x4 長條
                [[1, 1, 1, 1, 1]],                      # 1x5 長條
                [[1, 1], [1, 1]],                       # 2x2 正方形
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],      # 3x3 大正方形
                [[1, 1, 1], [0, 1, 0]],                 # T 型方塊
                [[1, 1, 1], [1, 0, 0]],                 # 7 型方塊
                [[1, 1, 1], [0, 0, 1]],                 # 反 7 型方塊
                [[1, 1], [1, 0]],                       # 小 L 型方塊
                [[1, 1, 1], [1, 0, 0], [1, 0, 0]],      # L 型方塊
                [[1, 1, 0], [0, 1, 1]],                 # Z 型方塊
                [[0, 1, 1], [1, 1, 0]],                 # 反 Z 型方塊
                [[1, 0], [0, 1]],                       # 小 \ 型方塊
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],      # \ 型方塊
                [[1, 1], [1, 1], [1, 1]]                # 2x3 實心矩形
            ]
        for piece in base_pieces:
            arr = np.array(piece, dtype=np.uint8)
            for k in range(4):
                rotated = np.rot90(arr, k)
                r_rows, r_cols = rotated.shape
                self.legal_shapes.add((r_rows, r_cols))
                grid_list = rotated.tolist()
                if grid_list not in self.legal_grids:
                    self.legal_grids.append(grid_list)

    def process(self):
        # ==========================================
        # 影像預處理
        # ==========================================
        hsv = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2HSV)
        _ , _, v_channel = cv2.split(hsv)
        blur = cv2.GaussianBlur(v_channel, (11, 11), 0)
        thresh_v = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh_g = cv2.morphologyEx(thresh_v, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        thresh_g = cv2.morphologyEx(thresh_g, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        thresh_g = cv2.morphologyEx(thresh_g, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        self.img_debug = self.img_orig.copy()

        # ==========================================
        # 定位棋盤大框
        # ==========================================
        board_thresh = cv2.morphologyEx(thresh_g, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        cnts, _ = cv2.findContours(board_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False
        candidates = []
        img_area = v_channel.shape[0] * v_channel.shape[1]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > (img_area * 0.1) and 0.8 <= float(w) / h <= 1.2:
                candidates.append({'area': area, 'bx': x, 'by': y, 'bw': w, 'bh': h})
        if not candidates:
            return False
        rough_box = max(candidates, key=lambda c: c['area'])
        bx, by, bw, bh = rough_box['bx'], rough_box['by'], rough_box['bw'], rough_box['bh']
        sx, sy = int(bx + bw * 0.075), int(by + bh * 0.075)
        sw, sh = int(bw * 0.85), int(bh * 0.85)
        cv2.rectangle(self.img_debug, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.rectangle(self.img_debug, (sx, sy), (sx + sw, sy + sh), (255, 50, 50), 2)

        kernel_h, kernel_v = np.ones((1, 101), np.uint8), np.ones((101, 1), np.uint8)
        thresh_h = cv2.morphologyEx(thresh_v, cv2.MORPH_OPEN, kernel_h)
        thresh_h = cv2.morphologyEx(thresh_h, cv2.MORPH_CLOSE, kernel_h)
        thresh_vt = cv2.morphologyEx(thresh_v, cv2.MORPH_OPEN, kernel_v)
        thresh_vt = cv2.morphologyEx(thresh_vt, cv2.MORPH_CLOSE, kernel_v)
        proj_y = np.sum(thresh_h[sy:sy+sh, sx:sx+sw], axis=1) / 255
        proj_x = np.sum(thresh_vt[sy:sy+sh, sx:sx+sw], axis=0) / 255

        def get_exact_edges_from_roi_debug(projection, offset_start, rough_min, is_horizontal):
            if len(projection) == 0:
                return None, None
            valid_coords = np.where(projection > 0)[0]
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
            if len(peaks_roi) < 3:
                return None, None
            color_centroid = (0, 255, 255)
            for p_roi in peaks_roi:
                p_global = int(round(p_roi + offset_start))
                if is_horizontal:
                    cv2.line(self.img_debug, (sx, p_global), (sx + sw // 2, p_global), color_centroid, 2, cv2.LINE_AA)
                else:
                    cv2.line(self.img_debug, (p_global, sy), (p_global, sy + sh // 2), color_centroid, 2, cv2.LINE_AA)
            diffs = np.diff(peaks_roi)
            valid_diffs = [d for d in diffs if d > 20]
            if not valid_diffs:
                return None, None
            u_est = np.median(valid_diffs)
            indices = [0]
            clean_peaks = [peaks_roi[0]]
            for i in range(1, len(peaks_roi)):
                diff_u = (peaks_roi[i] - clean_peaks[-1]) / u_est
                idx_diff = int(round(diff_u))
                if diff_u < 0.5:
                    clean_peaks[-1] = (clean_peaks[-1] + peaks_roi[i]) / 2.0
                    continue
                indices.append(indices[-1] + idx_diff)
                clean_peaks.append(peaks_roi[i])
            if len(clean_peaks) < 3:
                return None, None
            u_opt, offset_0_roi = np.polyfit(indices, clean_peaks, 1)
            color_optimal = (0, 0, 255)
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
        # 呼叫與保險機制
        # ==========================================
        min_y, max_y = get_exact_edges_from_roi_debug(proj_y, sy, by, is_horizontal=True)
        min_x, max_x = get_exact_edges_from_roi_debug(proj_x, sx, bx, is_horizontal=False)
        if None in (min_x, max_x, min_y, max_y):
            min_x, max_x = bx + bw * 0.03, bx + bw * 0.97
            min_y, max_y = by + bh * 0.03, by + bh * 0.97
        approx = np.array([[[min_x, min_y]], [[min_x, max_y]], [[max_x, max_y]], [[max_x, min_y]]], dtype=np.int32)
        self.pts1 = self.order_points(approx.reshape(4, 2))
        orig_unit = np.linalg.norm(self.pts1[0] - self.pts1[1]) / 8.0
        M = cv2.getPerspectiveTransform(self.pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        self.warp_orig = cv2.warpPerspective(self.img_orig, M, (400, 400))

        # ==========================================
        # 8x8 格子判定
        # ==========================================
        self.grid_state = [[0]*8 for _ in range(8)]
        cell_samples = []
        for r in range(8):
            for c in range(8):
                poly_pts = self.get_cell_poly_sampling(self.pts1, r, c, 0.1, 0.9).astype(np.int32)
                gx, gy, gw, gh = cv2.boundingRect(poly_pts)
                gy_s, gy_e = max(0, gy), min(thresh_v.shape[0], gy + gh)
                gx_s, gx_e = max(0, gx), min(thresh_v.shape[1], gx + gw)
                patch_thresh = thresh_v[gy_s:gy_e, gx_s:gx_e]
                white_ratio = np.sum(patch_thresh == 255) / patch_thresh.size if patch_thresh.size > 0 else 1.0
                cell_samples.append((r, c, white_ratio, gx_s, gy_s, gx_e, gy_e, poly_pts))
        best_empty_cell = min(cell_samples, key=lambda x: x[2])
        eb_x_s, eb_y_s, eb_x_e, eb_y_e = best_empty_cell[3:7]
        board_bg_v = np.median(v_channel[eb_y_s:eb_y_e, eb_x_s:eb_x_e])
        for r, c, _, gx_s, gy_s, gx_e, gy_e, poly_pts in cell_samples:
            cw = gx_e - gx_s
            ch = gy_e - gy_s
            cx_s, cx_e = gx_s + int(0.35 * cw), gx_s + int(0.65 * cw)
            cy_s, cy_e = gy_s + int(0.35 * ch), gy_s + int(0.65 * ch)
            cell_center_patch = v_channel[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
            if cell_center_patch.size > 0:
                cell_v_median = np.median(cell_center_patch)
                is_block = abs(int(cell_v_median) - int(board_bg_v)) > 15
            else:
                is_block = False
            self.grid_state[r][c] = 1 if is_block else 0
            sampling_poly_pts = self.get_cell_poly_sampling(self.pts1, r, c, 0.35, 0.65).astype(np.int32)
            if is_block:
                cv2.polylines(self.img_debug, [sampling_poly_pts], True, (255, 255, 255), 3, cv2.LINE_AA)
            else:
                cv2.polylines(self.img_debug, [sampling_poly_pts], True, (120, 120, 120), 2, cv2.LINE_AA)
            border_color = (120, 120, 120)
            if r == best_empty_cell[0] and c == best_empty_cell[1]:
                border_color = (0, 255, 0)
            cv2.polylines(self.img_debug, [poly_pts], True, border_color, 2, cv2.LINE_AA)

        # ==========================================
        # 待放區ROI
        # ==========================================
        img_h = self.img_orig.shape[0]
        board_h = max_y - min_y
        ay_s, ay_e = int(max_y + 0.15 * board_h) , int(min(img_h, (max_y + 0.45 * board_h)))
        if ay_s >= ay_e:
            return True
        piece_area_thresh = cv2.morphologyEx(thresh_g, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
        piece_area_mask = piece_area_thresh[ay_s:ay_e, :]
        self.piece_area_color = self.img_orig[ay_s:ay_e, :]
        by_s, by_e = int(max_y + 0.1 * board_h) , int(min(img_h, (max_y + 0.15 * board_h)))
        piece_area_color_bg = self.img_orig[by_s:by_e, :]
        bg_pixels = piece_area_color_bg.reshape(-1, 3)
        self.global_bg_color = np.median(bg_pixels, axis=0) if len(bg_pixels) > 0 else piece_area_color_bg[5, 5]


        # ==========================================
        # 解析待放方塊
        # ==========================================
        p_cnts, _ = cv2.findContours(piece_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates_p = []
        for cnt in p_cnts:
            x, y, pw, ph = cv2.boundingRect(cnt)
            if pw < (orig_unit * 0.5) and ph < (orig_unit * 0.5): continue
            if pw > 6*orig_unit or ph > 6*orig_unit: continue
            candidates_p.append([x, y + ay_s, pw, ph, x + pw/2])
        final_pieces = sorted(candidates_p, key=lambda p: p[0])[:3]
        p_unit = orig_unit * self.piece_scale
        
        self.detected_pieces = []
        for x, ay, pw, ph, _ in final_pieces:
            mask_roi = thresh_g[ay:ay+ph, x:x+pw]
            bgr_roi = self.img_orig[ay:ay+ph, x:x+pw]
            
            # 🎯 修正點：同步傳入簡化後的 self.global_bg_color
            parsed_grid = self.parse_piece_multi_channel(mask_roi, bgr_roi, p_unit, x, ay, self.global_bg_color)
            self.detected_pieces.append({"grid": parsed_grid, "roi_img": bgr_roi.copy()})
        return True

    def parse_piece_multi_channel(self, mask_roi, bgr_roi, unit, ox, oy, bg_color):
        diff_map = np.linalg.norm(bgr_roi.astype(np.float32) - bg_color.astype(np.float32), axis=2).astype(np.uint8)
        _, pure_piece_mask = cv2.threshold(diff_map, 30, 255, cv2.THRESH_BINARY)
        nz = cv2.findNonZero(pure_piece_mask)
        if nz is None:
            nz = cv2.findNonZero(mask_roi)
        if nz is None:
            return [[1]]
        mx, my, mw, mh = cv2.boundingRect(nz)
        cols = max(1, min(5, int(round(mw / unit))))
        rows = max(1, min(5, int(round(mh / unit))))
        cols, rows = max(1, min(5, cols)), max(1, min(5, rows))
        final_grid = None
        start_thresh = 30
        end_thresh = 6
        step = 2
        for current_thresh in range(start_thresh, end_thresh - 1, -step):
            grid = [[0]*cols for _ in range(rows)]
            has_pieces = False
            for r in range(rows):
                for c in range(cols):
                    c_s = int(mx + c * unit)
                    c_e = int(mx + (c + 1) * unit)
                    r_s = int(my + r * unit)
                    r_e = int(my + (r + 1) * unit)
                    c_e = min(c_e, bgr_roi.shape[1])
                    r_e = min(r_e, bgr_roi.shape[0])
                    cx_s, cx_e = c_s + int(0.25 * (c_e - c_s)), c_s + int(0.75 * (c_e - c_s))
                    cy_s, cy_e = r_s + int(0.25 * (r_e - r_s)), r_s + int(0.75 * (r_e - r_s))
                    cx_s, cx_e = max(0, cx_s), min(bgr_roi.shape[1], cx_e)
                    cy_s, cy_e = max(0, cy_s), min(bgr_roi.shape[0], cy_e)
                    patch = bgr_roi[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
                    if patch.size > 0:
                        color_dist = np.linalg.norm(np.median(patch, axis=(0,1)) - bg_color)
                        is_p = color_dist > current_thresh
                    else:
                        is_p = False
                    if is_p:
                        grid[r][c] = 1
                        has_pieces = True
            if has_pieces and (grid in self.legal_grids):
                final_grid = grid
                break
        if final_grid is None:
            final_grid = grid

        for r in range(len(final_grid)):
            for c in range(len(final_grid[0])):
                c_s = int(mx + c * unit)
                c_e = int(mx + (c + 1) * unit)
                r_s = int(my + r * unit)
                r_e = int(my + (r + 1) * unit)
                is_p = final_grid[r][c] == 1
                sx, sy, ex, ey = ox + c_s, oy + r_s, ox + c_e, oy + r_e
                cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), (255,255,255) if is_p else (120,120,120), 2)
                px_s, px_e = c_s + int(0.25 * (c_e - c_s)), c_s + int(0.75 * (c_e - c_s))
                py_s, py_e = r_s + int(0.25 * (r_e - r_s)), r_s + int(0.75 * (r_e - r_s))
                cv2.rectangle(self.img_debug, (ox + px_s, oy + py_s), (ox + px_e, oy + py_e), (255,255,255) if is_p else (120,120,120), 3)
        return final_grid

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
    def solve(self, grid, pieces, p_indices, path=None):
        self.global_best_path = None
        self.global_min_perimeter = float('inf')
        self.total_scanned_solutions = 0
        clean_grid = [[int(grid[r][c]) for c in range(8)] for r in range(8)]
        just_grids = [p["grid"] for p in pieces]
        self._solve_dfs(clean_grid, just_grids, p_indices, [])
        return self.global_best_path

    def _solve_dfs(self, grid, pieces, p_indices, current_path):
        if not p_indices:
            self.total_scanned_solutions += 1
            score = self.get_perimeter(grid)
            if score < self.global_min_perimeter:
                self.global_min_perimeter = score
                self.global_best_path = list(current_path)
            return
        for i in p_indices:
            p = pieces[i]
            p_rows = len(p)
            p_cols = len(p[0])
            for r in range(9 - p_rows):
                for c in range(9 - p_cols):
                    if self.can_place_fast(grid, p, r, c, p_rows, p_cols):
                        ng = [row[:] for row in grid]
                        for pr in range(p_rows):
                            for pc in range(p_cols):
                                if p[pr][pc]:
                                    ng[r+pr][c+pc] = 1
                        rs = [idx for idx, row in enumerate(ng) if all(row)]
                        cs = [j for j in range(8) if all(ng[idx][j] for idx in range(8))]
                        for row_idx in rs: ng[row_idx] = [0]*8
                        for col_idx in cs:
                            for row_idx in range(8): ng[row_idx][col_idx] = 0
                        placement = (i, r, c, rs, cs)
                        current_path.append(placement)
                        self._solve_dfs(ng, pieces, [idx for idx in p_indices if idx != i], current_path)
                        current_path.pop()

    def can_place_fast(self, grid, p, r, c, p_rows, p_cols):
        for pr in range(p_rows):
            for pc in range(p_cols):
                if p[pr][pc] and grid[r+pr][c+pc]: return False
        return True

    def get_perimeter(self, grid):
        perimeter = 0
        for r in range(8):
            for c in range(8):
                if grid[r][c] == 1:
                    if r > 0 and grid[r-1][c] == 0: perimeter += 1
                    if r < 7 and grid[r+1][c] == 0: perimeter += 1
                    if c > 0 and grid[r][c-1] == 0: perimeter += 1
                    if c < 7 and grid[r][c+1] == 0: perimeter += 1
        return perimeter
