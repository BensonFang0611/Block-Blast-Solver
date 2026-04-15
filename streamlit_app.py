import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import os
import json
from datetime import datetime
from vision_engine import VisionEngine, LogicSolver

register_heif_opener()

# 顏色設定
STEP_COLORS = [
    (110, 230, 255),  # 亮黃
    (220, 150, 255),  # 粉紫
    (255, 255, 120)   # 青藍
]
GRAY_ELIMINATED = (40, 40, 40) 

st.set_page_config(page_title="Block Blast Solver Pro", layout="wide")

# --- 輔助函式：將 pieces 數據轉為小圖顯示 ---
def draw_piece_preview(piece_grid):
    cell_size = 20
    rows = len(piece_grid)
    cols = len(piece_grid[0])
    canvas = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8) + 255 # 白色背景
    for r in range(rows):
        for c in range(cols):
            if piece_grid[r][c]:
                cv2.rectangle(canvas, (c*cell_size, r*cell_size), 
                              ((c+1)*cell_size, (r+1)*cell_size), (0, 120, 255), -1)
            cv2.rectangle(canvas, (c*cell_size, r*cell_size), 
                          ((c+1)*cell_size, (r+1)*cell_size), (200, 200, 200), 1)
    return canvas

# --- 1. 上傳區 ---
st.title("🧩 Block Blast Solver Pro")
file = st.file_uploader("📸 上傳遊戲截圖", type=['png','jpg','jpeg','heic'])

if file:
    # 讀取圖片以進行後續報錯使用
    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    
    @st.cache_resource
    def get_results(_cv_img):
        eng = VisionEngine(_cv_img)
        if eng.process():
            # 嘗試求解
            sol = LogicSolver().solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
            return eng, sol
        return None, None

    eng, sol = get_results(cv_img)

    if eng:
        # --- 3. 解法展示 ---
        st.header("💡 解法")
        if sol:
            step_options = [f"第 {i} 步" for i in range(len(sol) + 1)]
            selected_step_str = st.radio("步驟切換：", options=step_options, horizontal=True)
            current_step = step_options.index(selected_step_str)
            
            col_l, col_r = st.columns([1.5, 1])
            with col_l:
                # 繪製邏輯疊圖 (與您之前的邏輯相同)
                base_canvas = eng.warp_orig.copy()
                u = 400 / 8
                temp_canvas = np.zeros_like(base_canvas)
                
                for s in range(current_step):
                    p_idx, row, col, cleared_rs, cleared_cs = sol[s]
                    p = eng.detected_pieces[p_idx]
                    color = STEP_COLORS[s % len(STEP_COLORS)]
                    for pr in range(len(p)):
                        for pc in range(len(p[0])):
                            if p[pr][pc]:
                                rx, ry = int((col+pc)*u), int((row+pr)*u)
                                cv2.rectangle(temp_canvas, (rx, ry), (rx+int(u), ry+int(u)), color, -1)
                    
                    if cleared_rs or cleared_cs:
                        # 消除線條顯示
                        for cr in cleared_rs:
                            cv2.rectangle(temp_canvas, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                        for cc in cleared_cs:
                            cv2.rectangle(temp_canvas, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)

                data_mask = cv2.cvtColor(temp_canvas, cv2.COLOR_BGR2GRAY) > 0
                base_canvas[data_mask] = cv2.addWeighted(temp_canvas, 0.7, base_canvas, 0.3, 0)[data_mask]
                st.image(base_canvas, channels="BGR", use_container_width=True)
        else:
            st.warning("⚠️ 此盤面經運算後無解，請確認下方偵測是否正確。")
        
        # --- 2. 顯示偵測到的待放物 ---
        p_cols = st.columns(3)
        for i, piece in enumerate(eng.detected_pieces):
            with p_cols[i]:
                p_img = draw_piece_preview(piece)
                st.image(p_img, caption=f"方塊 {i+1}")
        
        # --- 5. Feedback 反饋系統 ---
        st.markdown("---")
        st.subheader("🚩 判定不準？幫我優化程式")

        with st.form("feedback_form"):
            comment = st.text_input("簡單補充說明 (選填)")
            submit_feedback = st.form_submit_button("🚀 回傳錯誤數據至後台")
            
            if submit_feedback:
                # 建立報告資料夾
                os.makedirs("reports", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_id = f"report_{timestamp}"
                
                # 儲存圖片
                raw_pil_img.save(f"reports/{report_id}_raw.jpg")
                cv2.imwrite(f"reports/{report_id}_debug.jpg", eng.img_debug)
                
                # 儲存辨識數據
                meta_data = {
                    "comment": comment,
                    "grid_state": eng.grid_state,
                    "pieces": eng.detected_pieces
                }
                with open(f"reports/{report_id}_data.json", "w") as f:
                    json.dump(meta_data, f)
                
                st.success(f"感謝回報！數據已存檔 (ID: {report_id})。這將幫助我改進算法。")

        # --- 4. 全域除錯分析 ---
        with st.expander("🛠️ 點開查看視覺辨識除錯圖"):
            st.image(eng.img_debug, channels="BGR", caption="綠框: 網格定位 | 紅框: 偵測成功")

    else:
        st.error("❌ 無法定位棋盤，請確保圖片包含完整的遊戲邊框。")
