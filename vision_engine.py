import cv2
import numpy as np
import copy

class VisionEngine:
    def __init__(self, cv_img, base_pieces=None):
        self.img_orig = cv_img
        self.img_debug = None           # 顏色判定 Debug 專用圖
        self.grid_state = [[0]*8 for _ in range(8)]
        self.detected_pieces = []
        self.warp_orig = None
        self.piece_scale = 0.46
        self.pts1 = None                # 棋盤精準四角點
        
        # 💡 自動生成「所有旋轉角度的合法形狀與尺寸」
        self.legal_shapes = set()       # 儲存 (rows, cols) 尺寸
        self.legal_grids = []           # 儲存旋轉後的二維矩陣結構
        self._generate_rotated_pieces(base_pieces)

    def _generate_rotated_pieces(self, base_pieces):
        """
        智慧核心：傳入基本方塊原型，自動衍生出 4 個旋轉角度(0, 90, 180, 270) 的特徵
        """
        if not base_pieces:
            # 如果使用者沒設定，預設提供標準益智遊戲方塊原型
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
                [[1, 0], [0, 1]],                       # 小 \ 型方塊
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],      # \ 型方塊
                [[1, 1], [1, 1], [1, 1]]                # 2x3 實心矩形
            ]
        for piece in base_pieces:
            arr = np.array(piece, dtype=np.uint8)
            # 依序旋轉 0, 90, 180, 270 度
            for k in range(4):
                rotated = np.rot90(arr, k)
                r_rows, r_cols = rotated.shape
                # 將尺寸加入集合 (自動去重)
                self.legal_shapes.add((r_rows, r_cols))
                # 將矩形結構轉回標準 Python list 儲存
                grid_list = rotated.tolist()
                if grid_list not in self.legal_grids:
                    self.legal_grids.append(grid_list)

    def process(self):
        # ==========================================
        # 1. 影像預處理 (萃取特徵與二值化)
        # ==========================================
        hsv = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
        kernel_g = np.ones((3, 3), np.uint8)
        
        # 建立全局二值化遮罩
        thresh_g = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh_g = cv2.morphologyEx(thresh_g, cv2.MORPH_OPEN, kernel_g)
        thresh_g = cv2.morphologyEx(thresh_g, cv2.MORPH_CLOSE, kernel_g)
        
        # 複製一份用於棋盤大框定位
        board_prep = cv2.morphologyEx(thresh_g, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.img_debug = self.img_orig.copy()

        # ==========================================
        # 2. 定位棋盤（亞像素質心擬合 + 對比視覺化）
        # ==========================================
        board_thresh = cv2.morphologyEx(board_prep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
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

        # 取得粗略的棋盤大框
        rough_box = max(candidates, key=lambda c: c['area'])
        bx, by, bw, bh = rough_box['bx'], rough_box['by'], rough_box['bw'], rough_box['bh']
        
        # 建立 90% 純淨區 (上下左右各往內縮 5%)
        sx, sy = int(bx + bw * 0.05), int(by + bh * 0.05)
        sw, sh = int(bw * 0.90), int(bh * 0.90)
        
        cv2.rectangle(self.img_debug, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.rectangle(self.img_debug, (sx, sy), (sx + sw, sy + sh), (255, 50, 50), 2)

        # 提取水平與垂直線
        thresh_vch = cv2.adaptiveThreshold(v_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        kernel_h, kernel_v = np.ones((1, 101), np.uint8), np.ones((101, 1), np.uint8)
        thresh_h = cv2.morphologyEx(thresh_vch, cv2.MORPH_OPEN, kernel_h)
        thresh_h = cv2.morphologyEx(thresh_h, cv2.MORPH_CLOSE, kernel_h)
        thresh_v = cv2.morphologyEx(thresh_vch, cv2.MORPH_OPEN, kernel_v)
        thresh_v = cv2.morphologyEx(thresh_v, cv2.MORPH_CLOSE, kernel_v)

        proj_y = np.sum(thresh_h[sy:sy+sh, sx:sx+sw], axis=1) / 255
        proj_x = np.sum(thresh_v[sy:sy+sh, sx:sx+sw], axis=0) / 255

        def get_exact_edges_from_roi_debug(projection, offset_start, rough_min, is_horizontal):
            if len(projection) == 0: return None, None
            valid_coords = np.where(projection > 0)[0]
            if len(valid_coords) == 0: return None, None
            
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

            if len(clean_peaks) < 3: return None, None

            # 繪製黃色細線
            color_centroid = (0, 255, 255)
            for p_roi in clean_peaks:
                p_global = int(round(p_roi + offset_start))
                if is_horizontal:
                    cv2.line(self.img_debug, (sx, p_global), (sx + sw // 2, p_global), color_centroid, 2, cv2.LINE_AA)
                else:
                    cv2.line(self.img_debug, (p_global, sy), (p_global, sy + sh // 2), color_centroid, 2, cv2.LINE_AA)

            u_opt, offset_0_roi = np.polyfit(indices, clean_peaks, 1)

            # 繪製紅色粗線
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
        # 3. 💡【新邏輯】：雙輪式 8x8 棋盤辨識（最小 % 數找底色 + 中心明度差值）
        # ==========================================
        self.grid_state = [[0]*8 for _ in range(8)]
        cell_white_ratios = [] 
        
        # ─── 第一輪：計算 64 格的白色像素占比，找出「最純淨的空位」作為底色基準 ───
        for r in range(8):
            for c in range(8):
                poly_pts = self.get_cell_poly_sampling(self.pts1, r, c, 0.08, 0.92).astype(np.int32)
                gx, gy, gw, gh = cv2.boundingRect(poly_pts)
                
                gy_s, gy_e = max(0, gy), min(thresh_g.shape[0], gy + gh)
                gx_s, gx_e = max(0, gx), min(thresh_g.shape[1], gx + gw)
                
                patch_thresh = thresh_g[gy_s:gy_e, gx_s:gx_e]
                white_ratio = np.sum(patch_thresh == 255) / patch_thresh.size if patch_thresh.size > 0 else 1.0
                
                cell_white_ratios.append((r, c, white_ratio, (gx_s, gy_s, gx_e, gy_e), poly_pts))
        
        # 🎯 找出白色像素占比最小（最黑、最空）的那一格當作底色來源
        best_empty_cell = min(cell_white_ratios, key=lambda x: x[2])
        eb_x_s, eb_y_s, eb_x_e, eb_y_e = best_empty_cell[3]
        
        # 採樣該空位格在 V 通道（明度）的中位數，作為全域棋盤純底色基準
        board_bg_v = np.median(v_channel[eb_y_s:eb_y_e, eb_x_s:eb_x_e])
        
        # ─── 第二輪：利用底色明度基準，透過中心點明度差值決定 64 格狀態 ───
        for r, c, _, (gx_s, gy_s, gx_e, gy_e), poly_pts in cell_white_ratios:
            cw = gx_e - gx_s
            ch = gy_e - gy_s
            cx_s, cx_e = gx_s + int(0.35 * cw), gx_s + int(0.65 * cw)
            cy_s, cy_e = gy_s + int(0.35 * ch), gy_s + int(0.65 * ch)
            
            cell_center_patch = v_channel[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
            if cell_center_patch.size > 0:
                cell_v_median = np.median(cell_center_patch)
                # 💡 明度插值判定：如果該格中心明度和「純淨底色」差大於 15，代表有方塊肉身
                is_p = abs(int(cell_v_median) - int(board_bg_v)) > 15
            else:
                is_p = False
                
            self.grid_state[r][c] = 1 if is_p else 0
            
            border_color = (0, 255, 0) if is_p else (120, 120, 120)
            cv2.polylines(self.img_debug, [poly_pts], True, border_color, 2, cv2.LINE_AA)
            
            # 在被選中為底色基準點的那格中心畫一個藍色圓點（Debug 觀察用）
            if r == best_empty_cell[0] and c == best_empty_cell[1]:
                cv2.circle(self.img_debug, (int((gx_s+gx_e)/2), int((gy_s+gy_e)/2)), 5, (255, 0, 0), -1)

        # ==========================================
        # 4. 全域採樣待放區 (尋找下方待放方塊區域候選框)
        # ==========================================
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(self.pts1[:, 1]))
        ay_s, ay_e = bottom_y + 40, int(img_h * 0.82)
        if ay_s >= ay_e:
            return True

        piece_area_mask = thresh_vch[ay_s:ay_e, :]
        p_cnts, _ = cv2.findContours(piece_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates_p = []
        for cnt in p_cnts:
            if cv2.contourArea(cnt) < (orig_unit**2) * 0.2:
                continue
            x, y, pw, ph = cv2.boundingRect(cnt)
            if pw > 6*orig_unit or ph > 6*orig_unit:
                continue
            candidates_p.append([x, y + ay_s, pw, ph])

        final_pieces = sorted(candidates_p, key=lambda p: p[0])[:3]
        p_unit = orig_unit * self.piece_scale
        self.detected_pieces = []
        
        # ==========================================
        # 5. 解析待放方塊 (同步改用明度差值與中心採樣)
        # ==========================================
        for x, ay, pw, ph in final_pieces:
            mask_roi = thresh_g[ay:ay+ph, x:x+pw]
            v_roi = v_channel[ay:ay+ph, x:x+pw]
            
            parsed_grid = self.parse_piece_by_v_intensity(mask_roi, v_roi, pw, ph, p_unit, x, ay, board_bg_v)
            self.detected_pieces.append(parsed_grid)
            
        cv2.polylines(self.img_debug, [self.pts1.astype(int)], True, (0, 255, 0), 3)
        return True

    def parse_piece_by_v_intensity(self, mask_roi, v_roi, pw, ph, unit, ox, oy, bg_v):
        """
        利用大框校正 + 中心點明度與棋盤底色比對，精準解析方塊二維矩陣
        """
        nz = cv2.findNonZero(mask_roi)
        if nz is None: 
            return [[1]]
        mx, my, mw, mh = cv2.boundingRect(nz)
        
        cols = max(1, min(5, int(round(mw / unit))))
        rows = max(1, min(5, int(round(mh / unit))))
        
        grid = [[0]*cols for _ in range(rows)]
        final_grid = None
        
        # 嘗試利用 V 通道核心明度差值進行矩陣建立
        for r in range(rows):
            for c in range(cols):
                c_s = int(mx + c * unit)
                c_e = int(mx + (c + 1) * unit)
                r_s = int(my + r * unit)
                r_e = int(my + (r + 1) * unit)
                
                c_e = min(c_e, v_roi.shape[1])
                r_e = min(r_e, v_roi.shape[0])
                
                # 🎯 鎖定中心點範圍 (取 40% ~ 60% 核心) 規避邊緣毛邊
                cx_s, cx_e = c_s + int(0.40 * (c_e - c_s)), c_s + int(0.60 * (c_e - c_s))
                cy_s, cy_e = r_s + int(0.40 * (r_e - r_s)), r_s + int(0.60 * (r_e - r_s))
                
                patch_v = v_roi[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
                if patch_v.size > 0:
                    cell_v_median = np.median(patch_v)
                    is_p = abs(int(cell_v_median) - int(bg_v)) > 15
                else:
                    is_p = False
                    
                if is_p:
                    grid[r][c] = 1

            # 智慧旋轉角度合法性校驗過濾
            if any(1 in row for row in grid):
                if grid in self.legal_grids:
                    final_grid = grid

        if final_grid is None:
            final_grid = grid

        # 視覺化 Debug 繪製圖層
        for r_idx in range(len(final_grid)):
            for c_idx in range(len(final_grid[0])):
                c_s = int(mx + c_idx * unit)
                c_e = int(mx + (c_idx + 1) * unit)
                r_s = int(my + r_idx * unit)
                r_e = int(my + (r_idx + 1) * unit)
                sx, sy, ex, ey = ox + c_s, oy + r_s, ox + c_e, oy + r_e
                cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), (80, 80, 80), 1)
                
                if final_grid[r_idx][c_idx] == 1:
                    px_s, px_e = c_s + int(0.25 * (c_e - c_s)), c_s + int(0.75 * (c_e - c_s))
                    py_s, py_e = r_s + int(0.25 * (r_e - r_s)), r_s + int(0.75 * (r_e - r_s))
                    cv2.rectangle(self.img_debug, (ox + px_s, oy + py_s), (ox + px_e, oy + py_e), (255, 255, 255), 2)
                    
        return final_grid

    # --- 座標變換工具 ---
    def lerp(self, p1, p2, t):
        return p1 + (p2 - p1) * t

    def get_p(self, pts, row, col):
        top = self.lerp(pts[0], pts[1], col/8.0)
        bot = self.lerp(pts[3], pts[2], col/8.0)
        return self.lerp(top, bot, row/8.0)

    def get_cell_poly(self, pts, r, c):
        return np.array([self.get_p(pts, r, c), self.get_p(pts, r, c+1), self.get_p(pts, r+1, c+1), self.get_p(pts, r+1, c)], dtype=np.int32)

    def get_cell_poly_sampling(self, pts, r, c, s, e):
        return np.array([self.get_p(pts, r+s, c+s), self.get_p(pts, r+s, c+e), self.get_p(pts, r+e, c+e), self.get_p(pts, r+e, c+s)], dtype=np.int32)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect


class LogicSolver:
    def solve(self, grid, pieces, p_indices, path=None):
        """
        終極主接口：完全強迫拔除所有 return 阻斷，保證窮舉全盤面！
        """
        self.global_best_path = None
        self.global_min_perimeter = float('inf')
        self.total_scanned_solutions = 0 
        
        clean_grid = [[int(grid[r][c]) for c in range(8)] for r in range(8)]
        print("\n🤖 [AI 評估日誌]：開始全域窮舉所有排列組合...")
        
        # 啟動深度優先搜尋
        self._solve_dfs(clean_grid, pieces, p_indices, [])
        print(f"📊 [AI 評估結束]：全域共掃描到 {self.total_scanned_solutions} 組可行解。")
        
        if self.global_best_path is not None:
            print(f"🏆 最終篩選出的全域「最小內部周長」分數為: {self.global_min_perimeter}")
        else:
            print("❌ 警告：全域搜尋完畢，找不到任何一組可以完全放下三個方塊的解。")
        return self.global_best_path

    def _solve_dfs(self, grid, pieces, p_indices, current_path):
        # 基底條件：當 3 顆方塊都成功放下了，這是一組「完整走完」的解
        if not p_indices:
            self.total_scanned_solutions += 1
            score = self.get_perimeter(grid)
            
            if score < self.global_min_perimeter:
                self.global_min_perimeter = score
                self.global_best_path = list(current_path)
            return

        # 全域窮舉剩餘可用的方塊
        for i in p_indices:
            p = pieces[i]
            p_rows = len(p)
            p_cols = len(p[0])
            
            # 遍歷 8x8 棋盤的每一個角落
            for r in range(9 - p_rows):
                for c in range(9 - p_cols):
                    # 碰撞檢查
                    if self.can_place_fast(grid, p, r, c, p_rows, p_cols):
                        # 模擬放置
                        ng = [row[:] for row in grid]
                        for pr in range(p_rows):
                            for pc in range(p_cols):
                                if p[pr][pc]:
                                    ng[r+pr][c+pc] = 1
                                    
                        # 計算並物理執行消除
                        rs = [idx for idx, row in enumerate(ng) if all(row)]
                        cs = [j for j in range(8) if all(ng[idx][j] for idx in range(8))]
                        for row_idx in rs:
                            ng[row_idx] = [0]*8
                        for col_idx in cs:
                            for row_idx in range(8):
                                ng[row_idx][col_idx] = 0
                                
                        # 封裝當前步驟快照
                        placement = (i, r, c, rs, cs)
                        current_path.append(placement)
                        
                        # 推進下一層遞迴
                        self._solve_dfs(ng, pieces, [idx for idx in p_indices if idx != i], current_path)
                        current_path.pop() 

    def can_place_fast(self, grid, p, r, c, p_rows, p_cols):
        for pr in range(p_rows):
            for pc in range(p_cols):
                if p[pr][pc] and grid[r+pr][c+pc]:
                    return False
        return True

    def get_perimeter(self, grid):
        """
        精準計算 8x8 盤面內部的空洞暴露面（不計邊牆）
        """
        perimeter = 0
        for r in range(8):
            for c in range(8):
                if grid[r][c] == 1:
                    if r > 0 and grid[r-1][c] == 0: perimeter += 1
                    if r < 7 and grid[r+1][c] == 0: perimeter += 1
                    if c > 0 and grid[r][c-1] == 0: perimeter += 1
                    if c < 7 and grid[r][c+1] == 0: perimeter += 1
        return perimeter

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
        for i in rs:
            ng[i] = [0]*8
        for j in cs:
            for i in range(8):
                ng[i][j] = 0
        return ng
