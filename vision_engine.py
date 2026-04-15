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
        # 1. 影像預處理 (原有二值化)
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        self.img_debug = self.img_orig.copy() # Debug 改用原圖疊加資訊

        # 2. 智慧定位
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
            if cand['area'] > best_cand['area'] * 0.80:
                best_cand = cand
        
        pts1 = self.order_points(best_cand['approx'].reshape(4, 2))
        orig_board_w = np.linalg.norm(pts1[0] - pts1[1])
        orig_unit = orig_board_w / 8.0 
        
        M = cv2.getPerspectiveTransform(pts1, np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]))
        warp_color = cv2.warpPerspective(self.img_orig, M, (400, 400))
        warp_loc = cv2.warpPerspective(thresh, M, (400, 400))

        # ====== 3. 棋盤顏色與二值化競爭判定 ======
        grid_thresh = [[0]*8 for _ in range(8)]
        grid_color = [[0]*8 for _ in range(8)]
        
        # 取得棋盤「背景色」參考 (取邊角 10% 區塊的平均值)
        bg_sample = warp_color[0:20, 0:20]
        bg_avg_color = cv2.mean(bg_sample)[:3]

        for r in range(8):
            for c in range(8):
                u = 400 / 8
                # 採樣範圍 (中心 10%)
                s, e = 0.45, 0.55
                x_s, x_e = int((c + s) * u), int((c + e) * u)
                y_s, y_e = int((r + s) * u), int((r + e) * u)
                
                # A. 二值化判定
                roi_thresh = warp_loc[y_s:y_e, x_s:x_e]
                grid_thresh[r][c] = 1 if (cv2.countNonZero(roi_thresh) / roi_thresh.size) > 0.1 else 0
                
                # B. 顏色判定
                roi_color = warp_color[y_s:y_e, x_s:x_e]
                cell_color = cv2.mean(roi_color)[:3]
                dist = np.linalg.norm(np.array(cell_color) - np.array(bg_avg_color))
                # 如果顏色跟背景色差很大，判定為有方塊 (這裡閾值 30 可依光線調整)
                grid_color[r][c] = 1 if dist > 35 else 0

                # 繪製 Debug 資訊
                cp = self.get_cell_poly(pts1, r, c)
                # 顏色法用圓圈表示，二值法用外框
                cv2.polylines(self.img_debug, [cp], True, (0, 255, 0), 1)
                dot_color = (0, 0, 255) if grid_color[r][c] == 1 else (100, 100, 100)
                cv2.circle(self.img_debug, tuple(self.get_p(pts1, r+0.5, c+0.5).astype(int)), 3, dot_color, -1)

        # 競爭機制：取方塊總數較多的結果
        sum_t = sum(sum(row) for row in grid_thresh)
        sum_c = sum(sum(row) for row in grid_color)
        self.grid_state = grid_color if sum_c >= sum_t else grid_thresh

        # ====== 4. 待放物區域 (加入背景色比較) ======
        img_h = self.img_orig.shape[0]
        bottom_y = int(max(pts1[:, 1]))
        start_y, end_y = bottom_y + 40, int(img_h * 0.88)
        piece_area_mask = thresh[start_y:end_y, :]
        p_cnts, _ = cv2.findContours(piece_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates_p = []
        for cnt in p_cnts:
            if cv2.contourArea(cnt) < (orig_unit**2) * 0.2: continue 
            x, y, pw, ph = cv2.boundingRect(cnt)
            candidates_p.append([x, y + start_y, pw, ph, x + pw/2])

        candidates_p = sorted(candidates_p, key=lambda p: p[0])
        filtered_pieces = []
        for cand in candidates_p:
            if not any(abs(cand[4] - acc[4]) < (orig_unit * 0.5) for acc in filtered_pieces):
                filtered_pieces.append(cand)
        
        # 取得待放區域的左上角背景色
        p_bg_color = self.img_orig[start_y:start_y+10, 0:10].mean(axis=(0,1))

        for x, ay, pw, ph, _ in filtered_pieces[:3]:
            mask = thresh[ay:ay+ph, x:x+pw]
            roi_rgb = self.img_orig[ay:ay+ph, x:x+pw]
            self.detected_pieces.append(self.parse_piece_with_color(mask, roi_rgb, pw, ph, orig_unit * self.piece_scale, x, ay, p_bg_color))
        
        return True

    def parse_piece_with_color(self, mask, roi_rgb, pw, ph, unit, ox, oy, bg_color):
        nz = cv2.findNonZero(mask)
        if nz is None: return [[1]]
        mx, my, mw, mh = cv2.boundingRect(nz)
        cols, rows = max(1, int(round(mw / unit))), max(1, int(round(mh / unit)))
        cw, ch = mw / cols, mh / rows
        
        grid_final = [[0]*cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                # 顏色採樣 (中心 10%)
                rs_x, rs_y = int(mx + (c+0.45)*cw), int(my + (r+0.45)*ch)
                re_x, re_y = int(mx + (c+0.55)*cw), int(my + (r+0.55)*ch)
                
                # 確保不越界
                rs_x, rs_y = max(0, rs_x), max(0, rs_y)
                
                # 1. 二值化四象限判定
                sub_m = mask[int(my+r*ch):int(my+(r+1)*ch), int(mx+c*cw):int(mx+(c+1)*cw)]
                is_p_thresh = False
                if sub_m.size > 4:
                    h, w = sub_m.shape
                    mh_sub, mw_sub = h//2, w//2
                    qs = [sub_m[0:mh_sub, 0:mw_sub], sub_m[0:mh_sub, mw_sub:w], sub_m[mh_sub:h, 0:mw_sub], sub_m[mh_sub:h, mw_sub:w]]
                    is_p_thresh = all([(cv2.countNonZero(q)/q.size > 0.01) if q.size > 0 else False for q in qs])

                # 2. 顏色判定
                sub_rgb = roi_rgb[rs_y:re_y, rs_x:re_x]
                is_p_color = False
                if sub_rgb.size > 0:
                    avg_c = cv2.mean(sub_rgb)[:3]
                    is_p_color = np.linalg.norm(np.array(avg_c) - np.array(bg_color)) > 35
                
                # 競爭
                is_p = is_p_thresh or is_p_color
                grid_final[r][c] = 1 if is_p else 0
                
                # Debug 顯示
                sx, sy = int(ox + mx + c*cw), int(oy + my + r*ch)
                ex, ey = int(sx + cw), int(sy + ch)
                color = (0, 0, 255) if is_p else (0, 200, 0)
                cv2.rectangle(self.img_debug, (sx, sy), (ex, ey), color, 2 if is_p else 1)
                
        return grid_final

    # --- 數學工具 (保持不變) ---
    def lerp(self, p1, p2, t): return p1 + (p2 - p1) * t
    def get_p(self, pts, row, col):
        top = self.lerp(pts[0], pts[1], col/8.0); bot = self.lerp(pts[3], pts[2], col/8.0)
        return self.lerp(top, bot, row/8.0)
    def get_cell_poly(self, pts, r, c):
        return np.array([self.get_p(pts,r,c), self.get_p(pts,r,c+1), self.get_p(pts,r+1,c+1), self.get_p(pts,r+1,c)], dtype=np.int32)
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect
