import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from vision_engine import VisionEngine, LogicSolver

register_heif_opener()

# 視覺顏色設定 (BGR)
STEP_COLORS = [
    (110, 230, 255),  # 第一步：亮黃色
    (220, 150, 255),  # 第二步：粉紫色
    (255, 255, 120)   # 第三步：青藍色
]
# 消除專用顏色 (深灰色)
GRAY_ELIMINATED = (40, 40, 40) 

st.set_page_config(page_title="Block Blast Solver", layout="wide")

# 移除原本強迫按鈕 100% 寬度的 CSS，讓畫面保持乾淨
st.markdown("""
    <style>
    /* 稍微美化 radio 選項的間距，確保在手機上好點擊 */
    div.row-widget.stRadio > div {
        flex-wrap: wrap;
        gap: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧩 Block Blast Solver")

# --- 1. 上傳區 (置中) ---
st.header("📸 上傳遊戲截圖")
file = st.file_uploader("選擇圖片 (png, jpg, heic)", type=['png','jpg','jpeg','heic'])

if file:
    @st.cache_resource
    def get_results(file_bytes):
        img = Image.open(file_bytes)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        eng = VisionEngine(cv_img)
        if eng.process():
            sol = LogicSolver().solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
            return eng, sol
        return None, None

    eng, sol = get_results(file)

    if eng:
        # --- 2. 解法展示 ---
        st.header("💡 解法")
        if sol:
            # 🌟 核心修改：徹底改用 st.radio 水平選項 🌟
            step_options = [f"第 {i} 步" for i in range(len(sol) + 1)]
            
            selected_step_str = st.radio(
                "選擇解法步驟：", 
                options=step_options, 
                horizontal=True, # 強制水平並排
                label_visibility="collapsed" # 隱藏標題讓畫面更簡潔
            )
            
            # 從字串中提取目前的步數數字 (例如從 "第 2 步" 取出 2)
            current_step = step_options.index(selected_step_str)
            
            col_l, col_r = st.columns([1.5, 1])
            with col_l:
                base_canvas = eng.warp_orig.copy()
                u = 400 / 8
                
                temp_canvas = np.zeros_like(base_canvas)
                
                # 改用 current_step 作為繪圖進度
                for s in range(current_step):
                    p_idx, row, col, cleared_rs, cleared_cs = sol[s]
                    p = eng.detected_pieces[p_idx]
                    color = STEP_COLORS[s % len(STEP_COLORS)]
                    
                    # A. 繪製多彩方塊
                    for pr in range(len(p)):
                        for pc in range(len(p[0])):
                            if p[pr][pc]:
                                rx, ry = int((col+pc)*u), int((row+pr)*u)
                                cv2.rectangle(temp_canvas, (rx, ry), (rx+int(u), ry+int(u)), color, -1)
                                cv2.rectangle(temp_canvas, (rx, ry), (rx+int(u), ry+int(u)), (0, 0, 0), 2)
                    
                    # B. 半透明疊加消除灰色層
                    if cleared_rs or cleared_cs:
                        elim_mask_layer = np.zeros_like(base_canvas)
                        for cr in cleared_rs:
                            cv2.rectangle(elim_mask_layer, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                            cv2.rectangle(elim_mask_layer, (0, int(cr*u)), (400, int((cr+1)*u)), (0, 0, 0), 1)
                        for cc in cleared_cs:
                            cv2.rectangle(elim_mask_layer, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)
                            cv2.rectangle(elim_mask_layer, (int(cc*u), 0), (int((cc+1)*u), 400), (0, 0, 0), 1)
                        
                        elim_mask = cv2.cvtColor(elim_mask_layer, cv2.COLOR_BGR2GRAY) > 0
                        temp_canvas[elim_mask] = cv2.addWeighted(elim_mask_layer, 0.6, temp_canvas, 0.4, 0)[elim_mask]

                final_canvas = base_canvas.copy()
                data_mask = cv2.cvtColor(temp_canvas, cv2.COLOR_BGR2GRAY) > 0
                final_canvas[data_mask] = cv2.addWeighted(temp_canvas, 0.8, base_canvas, 0.2, 0)[data_mask]

                st.image(final_canvas, channels="BGR", caption=f"解法步驟: {current_step}")
        else:
            st.error("此盤面無解。")

        st.markdown("---")
        # --- 3. 全域除錯 ---
        st.header("🛠️ 全域疊圖分析 (物理定標 50%)")
        st.image(eng.img_debug, channels="BGR", caption="綠框: 物理網格 | 亮紅: 偵測成功 | 暗紅: 空格")
    else:
        st.error("無法定位棋盤。")