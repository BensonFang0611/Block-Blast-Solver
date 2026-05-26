import cv2
import numpy as np

class VisionEngine:
    def __init__(self):
        # 影像快取
        self.img_orig = None
        self.img_debug = None
        
        # 棋盤頂點與單元格大小
        self.pts1 = None
        self.piece_scale = 1.0
        
        # 辨識結果狀態儲存
        self.grid_state = [[0]*8 for _ in range(8)]
        self.detected_pieces = []

    def set_chessboard_geometry(self, pts1, piece_scale=1.0):
        """
        設定 8x8 棋盤的四個角點 (由主程式偵測傳入)
        pts1: 4x2 的 numpy 陣列，順序為 [左上, 右上, 右下, 左下]
        """
        self.pts1 = np.array(pts1, dtype=np.float32)
        self.piece_scale = piece_scale

    def get_cell_poly_sampling(self, pts, r, c, start_f=0.0, end_f=1.0):
        """
        透視投影網格內插：計算棋盤第 (r, c) 格在指定比例範圍內的四個頂點
        """
        p0 = pts[0]; p1 = pts[1]; p2 = pts[2]; p3 = pts[3]
        
        # 計算左、右邊緣上的內插點
        f_top_s = r / 8.0 + (1.0 / 8.0) * start_f
        f_top_e = r / 8.0 + (1.0 / 8.0) * end_f
        
        gl_s = p0 + (p3 - p0) * f_top_s
        gl_e = p0 + (p3 - p0) * f_top_e
        gr_s = p1 + (p2 - p1) * f_top_s
        gr_e = p1 + (p2 - p1) * f_top_e
        
        # 計算四個角落
        fc_s = c / 8.0 + (1.0 / 8.0) * start_f
        fc_e = c / 8.0 + (1.0 / 8.0) * end_f
        
        pt0 = gl_s + (gr_s - gl_s) * fc_s
        pt1 = gl_s + (gr_s - gl_s) * fc_e
        pt2 = gl_e + (gr_e - gl_e) * fc_e
        pt3 = gl_e + (gr_e - gl_e) * fc_s
        
        return np.array([pt0, pt1, pt2, pt3], dtype=np.float32)

    def process(self, frame):
        """
        核心影像辨識主流程
        """
        if frame is None:
            return False
            
        self.img_orig = frame.copy()
        self.img_debug = frame.copy()
        
        if self.pts1 is None:
            # 尚未設定棋盤幾何位置，跳過辨識
            return False

        # ==========================================
        # 1. 影像預處理與色彩空間轉換
        # ==========================================
        hsv = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # 針對方塊與棋盤格肉身建立二值化遮罩
        _, thresh_g = cv2.threshold(v_channel, 60, 255, cv2.THRESH_BINARY)
        _, thresh_vch = cv2.threshold(v_channel, 70, 255, cv2.THRESH_BINARY)

        # 計算標準的方塊單元格參考大小 (用棋盤寬度除以 8)
        orig_unit = np.linalg.norm(self.pts1[0] - self.pts1[1]) / 8.0

        # ==========================================
        # 2. 初始化棋盤狀態
        # ==========================================
        self.grid_state = [[0]*8 for _ in range(8)]

        # ==========================================
        # 3. 💡【新邏輯】：雙輪式 8x8 棋盤辨識（最小 % 數找底色 + 中心明度差值）
        # ==========================================
        # ─── 第一輪：計算 64 格的白色像素占比，找出「最純淨的空位」作為底色基準 ───
        cell_white_ratios = [] # 儲存 (r, c, white_ratio, bounding_box, poly_pts)
        
        for r in range(8):
            for c in range(8):
                # 取得格子採樣中心點範圍的四個頂點座標 (略微縮小避開外框格線)
                poly_pts = self.get_cell_poly_sampling(self.pts1, r, c, 0.08, 0.92).astype(np.int32)
                gx, gy, gw, gh = cv2.boundingRect(poly_pts)
                
                gy_s, gy_e = max(0, gy), min(thresh_g.shape[0], gy + gh)
                gx_s, gx_e = max(0, gx), min(thresh_g.shape[1], gx + gw)
                
                patch_thresh = thresh_g[gy_s:gy_e, gx_s:gx_e]
                white_ratio = np.sum(patch_thresh == 255) / patch_thresh.size if patch_thresh.size > 0 else 1.0
                
                cell_white_ratios.append((r, c, white_ratio, (gx_s, gy_s, gx_e, gy_e), poly_pts))
        
        # 🎯 核心核心：找出白色像素占比最小（代表最黑、最空、最沒方塊物體）的那一格
        best_empty_cell = min(cell_white_ratios, key=lambda x: x[2])
        eb_x_s, eb_y_s, eb_x_e, eb_y_e = best_empty_cell[3]
        
        # 採樣該空位格在 V 通道（明度）的中位數，作為「全域棋盤純底色基準」
        board_bg_v = np.median(v_channel[eb_y_s:eb_y_e, eb_x_s:eb_x_e])
        
        # ─── 第二輪：利用底色明度基準，透過中心點明度插值決定 64 格的實心狀態 ───
        for r, c, _, (gx_s, gy_s, gx_e, gy_e), poly_pts in cell_white_ratios:
            # 🎯 鎖定每一格的核心中心點範圍 (取 35% ~ 65% 核心區)，徹底避開棋盤縫隙與邊緣格線
            cw = gx_e - gx_s
            ch = gy_e - gy_s
            cx_s, cx_e = gx_s + int(0.35 * cw), gx_s + int(0.65 * cw)
            cy_s, cy_e = gy_s + int(0.35 * ch), gy_s + int(0.65 * ch)
            
            # 提取該格正中心的明度切片
            cell_center_patch = v_channel[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
            
            if cell_center_patch.size > 0:
                cell_v_median = np.median(cell_center_patch)
                # 💡 明度插值判定：如果該格中心明度和「純淨空位底色」明度差大於 15，代表有方塊疊在上面
                is_p = abs(int(cell_v_median) - int(board_bg_v)) > 15
            else:
                is_p = False
                
            self.grid_state[r][c] = 1 if is_p else 0
            
            # 繪製 8x8 棋盤格 Debug 框線
            border_color = (0, 255, 0) if is_p else (120, 120, 120)
            cv2.polylines(self.img_debug, [poly_pts], True, border_color, 2, cv2.LINE_AA)
            
            # 【Debug 視覺化小彩蛋】：在被當作底色基準點的那格中心畫一個藍色圓點，方便你觀察對不對
            if r == best_empty_cell[0] and c == best_empty_cell[1]:
                cv2.circle(self.img_debug, (int((gx_s+gx_e)/2), int((gy_s+gy_e)/2)), 5, (255, 0, 0), -1)

        # ==========================================
        # 4. 尋找下方待放方塊區域的候選框
        # ==========================================
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(self.pts1[:, 1]))
        ay_s, ay_e = bottom_y + 40, int(img_h * 0.82)
        if ay_s >= ay_e:
            cv2.polylines(self.img_debug, [self.pts1.astype(int)], True, (0, 255, 0), 3)
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

        # 由左至右排序，取前 3 個待放方塊
        final_pieces = sorted(candidates_p, key=lambda p: p[0])[:3]
        
        p_unit = orig_unit * self.piece_scale
        self.detected_pieces = []
        
        # ==========================================
        # 5. 解析待放方塊內部結構 (採用對稱的中心明度採樣)
        # ==========================================
        for x, ay, pw, ph in final_pieces:
            mask_roi = thresh_g[ay:ay+ph, x:x+pw]
            v_roi = v_channel[ay:ay+ph, x:x+pw]
            
            # 使用下方獨立的明度矩陣解析函式
            parsed_grid = self.parse_piece_by_v_intensity(mask_roi, v_roi, pw, ph, p_unit, x, ay, board_bg_v)
            self.detected_pieces.append(parsed_grid)
            
        # 繪製大棋盤外框
        cv2.polylines(self.img_debug, [self.pts1.astype(int)], True, (0, 255, 0), 3)
        return True

    def parse_piece_by_v_intensity(self, mask_roi, v_roi, pw, ph, unit, ox, oy, bg_v):
        """
        解析單個待放方塊的形狀矩陣
        """
        nz = cv2.findNonZero(mask_roi)
        if nz is None: 
            return [[1]]
        mx, my, mw, mh = cv2.boundingRect(nz)
        
        cols = max(1, min(5, int(round(mw / unit))))
        rows = max(1, min(5, int(round(mh / unit))))
        
        grid = [[0]*cols for _ in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                c_s = int(mx + c * unit)
                c_e = int(mx + (c + 1) * unit)
                r_s = int(my + r * unit)
                r_e = int(my + (r + 1) * unit)
                
                c_e = min(c_e, v_roi.shape[1])
                r_e = min(r_e, v_roi.shape[0])
                
                # 🎯 縮小到核心區（40% ~ 60%）進行採樣，規避邊緣毛邊
                cx_s, cx_e = c_s + int(0.40 * (c_e - c_s)), c_s + int(0.60 * (c_e - c_s))
                cy_s, cy_e = r_s + int(0.40 * (r_e - r_s)), r_s + int(0.60 * (r_e - r_s))
                
                patch_v = v_roi[cy_s:max(cy_s+1, cy_e), cx_s:max(cx_s+1, cx_e)]
                
                if patch_v.size > 0:
                    cell_v_median = np.median(patch_v)
                    # 明度與棋盤底色基準比對，大於門檻值 15 代表此處有肉身
                    is_p = abs(int(cell_v_median) - int(bg_v)) > 15
                else:
                    is_p = False
                    
                if is_p:
                    grid[r][c] = 1

        # 繪製待放方塊的 Debug 網格圖層
        for r_idx in range(len(grid)):
            for c_idx in range(len(grid[0])):
                c_s = int(mx + c_idx * unit)
                c_e = int(mx + (c_idx + 1) * unit)
                r_s = int(my + r_idx * unit)
                r_e = int(my + (r_idx + 1) * unit)
                sx, sy, ex, ey = ox + c_s, oy + r_s, ox + c_e, oy + r_e
                cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), (80, 80, 80), 1)
                if grid[r_idx][c_idx] == 1:
                    px_s, px_e = c_s + int(0.35 * (c_e - c_s)), c_s + int(0.65 * (c_e - c_s))
                    py_s, py_e = r_s + int(0.35 * (r_e - r_s)), r_s + int(0.65 * (r_e - r_s))
                    cv2.rectangle(self.img_debug, (ox + px_s, oy + py_s), (ox + px_e, oy + py_e), (255, 255, 255), 2)
                    
        return grid
