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
STEP_COLORS = [(0, 255, 255), (255, 100, 255), (100, 255, 100)] # 亮青、亮粉、亮綠
GRAY_ELIMINATED = (60, 60, 60) # 消除後的半透明深灰色

# --- 🛠️ 輔助功能 1：繪製 5x5 預覽圖 ---
def draw_piece_preview_5x5(piece_grid):
    u = 30
    canvas = np.zeros((150, 150, 3), dtype=np.uint8)
    r_off = (5 - len(piece_grid)) // 2
    c_off = (5 - len(piece_grid[0])) // 2
    for r in range(len(piece_grid)):
        for c in range(len(piece_grid[0])):
            if piece_grid[r][c]:
                # 繪製方塊與內縮邊框
                x1, y1 = (c + c_off) * u, (r + r_off) * u
                x2, y2 = x1 + u, y1 + u
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 160, 200), -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 80, 100), 1)
    return canvas

# --- 🛠️ 輔助功能 2：圖片上傳 ImgBB ---
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

# --- 🛠️ 輔助功能 3：紀錄到 Google Sheets (包含 User Visit) ---
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
st.title("🧩 Block Blast Solver (Color + Grid)")

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
            st.warning("⚠️ 此盤面在目前的辨識結果下無解。")

        # 待放方塊預覽
        st.markdown("---")
        st.subheader("📦 偵測到的方塊")
        p_cols = st.columns(3)
        for i, piece in enumerate(eng.detected_pieces):
            p_cols[i].image(draw_piece_preview_5x5(piece), channels="BGR", use_container_width=True)

        # Debug 資訊
        with st.expander("🛠️ 查看辨識細節 (Debug View)"):
            st.write("白色 = 方塊 | 灰色 = 空位 | 左上圓點 = 背景顏色採樣參考")
            st.image(eng.img_debug, channels="BGR", use_container_width=True)
    else:
        st.error("❌ 無法精確定位棋盤，請確認截圖是否有完整邊框。")

    # --- 2. Feedback 回饋系統 ---
    st.markdown("---")
    st.subheader("🚩 Feedback 錯誤回報")
    with st.form("feedback_form"):
        msg = st.text_input("如果有辨識錯誤，請告訴我們（例如：第2個方塊辨識錯了）")
        if st.form_submit_button("🚀 送出回報"):
            with st.spinner("同步中..."):
                os.makedirs("temp", exist_ok=True)
                report_path = "temp/feedback.jpg"
                # 優先上傳 Debug 圖以便排錯
                cv2.imwrite(report_path, eng.img_debug if 'eng' in locals() else cv_img)
                
                url = upload_to_imgbb(report_path)
                if log_to_sheets(msg, url):
                    st.success("✅ 感謝您的回饋！我們將根據這張圖片進行優化。")

st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 50px;'>
    Block Blast Solver Beta v2.1 | Powered by Color Sensing Engine
</div>
""", unsafe_allow_html=True)
