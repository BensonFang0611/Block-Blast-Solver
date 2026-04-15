import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests
import base64
import pandas as pd
from datetime import datetime, timedelta, timezone
from streamlit_gsheets import GSheetsConnection
from vision_engine import VisionEngine, LogicSolver

# --- 🚀 核心配置 ---
IMGBB_API_KEY = "3fcf87a9eaae07555706aa02519e78c9"
SHEET_NAME = "Sheet1" 

# 顏色定義 (BGR)
STEP_COLORS = [(0, 230, 230), (230, 100, 230), (100, 230, 100)] # 亮青、亮粉、亮綠
GRAY_ELIMINATED = (60, 60, 60) # 消除後的半透明深灰色

# --- 🛠️ 輔助功能 1：繪製 5x5 深藍色風格方塊 ---
def draw_piece_preview_5x5(piece_grid):
    grid_size, u = 5, 40
    # 畫布背景為純黑，營造深色質感
    canvas = np.zeros((grid_size*u, grid_size*u, 3), dtype=np.uint8)
    rows, cols = len(piece_grid), len(piece_grid[0])
    offset_r, offset_c = (grid_size - rows) // 2, (grid_size - cols) // 2
    
    # 畫背景細格線 (深灰色)
    for i in range(grid_size + 1):
        cv2.line(canvas, (0, i*u), (grid_size*u, i*u), (40, 40, 40), 1)
        cv2.line(canvas, (i*u, 0), (i*u, grid_size*u), (40, 40, 40), 1)
        
    for r in range(rows):
        for c in range(cols):
            if piece_grid[r][c]:
                tr, tc = r + offset_r, c + offset_c
                # 深藍色填充 (0, 160, 200)
                cv2.rectangle(canvas, (tc*u, tr*u), ((tc+1)*u, (tr+1)*u), (200, 160, 0), -1)
                # 較深的藍色邊框 (0, 80, 100)
                cv2.rectangle(canvas, (tc*u, tr*u), ((tc+1)*u, (tr+1)*u), (100, 80, 0), 1)
    return canvas

# --- 🛠️ 輔助功能 2：水平縫合待放方塊 ---
def get_combined_pieces_image(detected_pieces):
    if not detected_pieces: return None
    piece_imgs = [draw_piece_preview_5x5(p) for p in detected_pieces[:3]]
    
    h, w, c = piece_imgs[0].shape
    gap_width = 15
    # 縫合間隙 (黑色)
    black_gap = np.zeros((h, gap_width, c), dtype=np.uint8)
    
    stack_list = []
    for i, img in enumerate(piece_imgs):
        stack_list.append(img)
        if i < len(piece_imgs) - 1:
            stack_list.append(black_gap)
            
    return np.hstack(stack_list)

# --- 🛠️ 輔助功能 3：圖片上傳 ImgBB ---
def upload_to_imgbb(file_path):
    try:
        with open(file_path, "rb") as file:
            img_base64 = base64.b64encode(file.read())
            data = {"key": IMGBB_API_KEY, "image": img_base64}
            response = requests.post("https://api.imgbb.com/1/upload", data=data)
            if response.status_code == 200:
                return response.json()["data"]["url"]
        return "Upload Error"
    except:
        return "Upload Failed"

# --- 🛠️ 輔助功能 4：紀錄到 Google Sheets (包含 User Visit) ---
def log_to_sheets(msg, img_url="None"):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        tz = timezone(timedelta(hours=8)) 
        now_tw = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame([{
            "Timestamp": now_tw,
            "Comment": msg,
            "Image_Link": img_url
        }])
        # 讀取並合併
        existing_data = conn.read(worksheet=SHEET_NAME, ttl=0)
        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
        conn.update(worksheet=SHEET_NAME, data=updated_df)
        return True
    except Exception as e:
        st.error(f"Sheet Error: {e}")
        return False

# --- 1. UI 介面 ---
st.set_page_config(page_title="Block Blast Solver", layout="centered")
st.title("🧩 Block Blast Solver ")

file = st.file_uploader("📸 上傳截圖", type=['png','jpg','jpeg','heic'], key="uploader")

if file:
    # ✨ 關鍵功能：User Visit 簽到機制
    if "logged_file" not in st.session_state or st.session_state.logged_file != file.name:
        if log_to_sheets("User Visit"):
            st.session_state.logged_file = file.name

    # 讀取影像
    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    
    # 辨識引擎
    eng = VisionEngine(cv_img)
    if eng.process():
        st.header("💡 解法建議")
        solver = LogicSolver()
        sol = solver.solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
        
        if sol:
            step_label = st.radio("步驟切換：", [f"第 {i} 步" for i in range(len(sol)+1)], horizontal=True)
            idx = int(step_label.split(' ')[1])
            
            # --- 繪製解法示意圖 ---
            canvas = eng.warp_orig.copy()
            u = 400 / 8
            
            for s in range(idx):
                p_idx, row, col, cl_rs, cl_cs = sol[s]
                p, color = eng.detected_pieces[p_idx], STEP_COLORS[s % 3]
                
                # ✨ 繪製方塊本體與格線 (Border)
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            x1, y1 = int((col+pc)*u), int((row+pr)*u)
                            x2, y2 = int((col+pc+1)*u), int((row+pr+1)*u)
                            # 填滿顏色
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                            # 加上黑色邊框 (格線)
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

                # 繪製消除效果 (半透明融合)
                overlay = canvas.copy()
                for cr in (cl_rs or []):
                    cv2.rectangle(overlay, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                for cc in (cl_cs or []):
                    cv2.rectangle(overlay, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)
                # 融合消除層
                cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            
            st.image(canvas, channels="BGR", use_container_width=True)
        else:
            st.warning("此盤面無解:..)")

        # 待放方塊預覽
        st.markdown("---")
        combined_piece_img = get_combined_pieces_image(eng.detected_pieces)
        if combined_piece_img is not None:
            st.image(combined_piece_img, caption="偵測到的待放方塊 (並排預覽)", channels="BGR", use_container_width=True)
        
        # Debug 資訊
        #with st.expander("🛠️ Debug"):
        #    st.write("白色 = 方塊 | 灰色 = 空位 | 左上圓點 = 背景顏色採樣參考")
        #    st.image(eng.img_debug, channels="BGR", use_container_width=True)
    else:
        st.error("❌ 無法精確定位棋盤，請確認截圖是否有完整邊框。")

    # --- 2. Feedback 回饋系統 ---
    st.markdown("---")
    st.subheader("🚩 Feedback 錯誤回報")
    with st.form("feedback_form"):
        msg = st.text_input("如果有辨識錯誤，請告訴我!!")
        if st.form_submit_button("🚀 送出"):
            with st.spinner("同步中..."):
                os.makedirs("temp", exist_ok=True)
                report_path = "temp/feedback.jpg"
                # 優先上傳 Debug 圖以便排錯
                cv2.imwrite(report_path, eng.img_debug if 'eng' in locals() else cv_img)
                
                url = upload_to_imgbb(report_path)
                if log_to_sheets(msg, url):
                    st.success("✅ 感謝您的回饋！將根據這張圖片進行優化。")

st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 50px;'>
    Block Blast Solver Beta v2.1 | Powered by Color Sensing Engine
</div>
""", unsafe_allow_html=True)
