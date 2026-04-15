import streamlit as st
import cv2
import numpy as np
from PIL import Image
from vision_engine import VisionEngine, LogicSolver

st.set_page_config(page_title="Block Blast Color Solver", layout="centered")
st.title("🧩 Block Blast Solver (Color Only)")

file = st.file_uploader("📸 上傳截圖", type=['png','jpg','jpeg'])

if file:
    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    
    eng = VisionEngine(cv_img)
    if eng.process():
        st.header("💡 解法建議")
        sol = LogicSolver().solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
        
        if sol:
            # 這裡顯示解法圖形... (邏輯同前，使用 eng.warp_orig 作為底圖)
            canvas = eng.warp_orig.copy()
            # ...繪製步驟邏輯...
            st.image(canvas, channels="BGR", use_container_width=True)
        else:
            st.warning("⚠️ 顏色判定後盤面無解，請檢查下方 Debug 圖。")
        
        with st.expander("🛠️ 顏色辨識細節 (Debug View)"):
            st.write("白色小點 = 偵測到方塊 | 灰色小點 = 空位")
            st.image(eng.img_debug, channels="BGR", use_container_width=True)
    else:
        st.error("❌ 無法定位棋盤，請確保截圖完整。")
