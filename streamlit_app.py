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

# 步驟顏色 (BGR 格式)
STEP_COLORS = [(0, 255, 255), (255, 100, 255), (100, 255, 100)]
GRAY_ELIMINATED = (100, 100, 100)

# --- 🛠️ 輔助功能 1：繪製 5x5 待放方塊預覽 ---
def draw_piece_preview_5x5(piece_grid):
    grid_size, u = 5, 30
    canvas = np.zeros((grid_size*u, grid_size*u, 3), dtype=np.uint8)
    rows, cols = len(piece_grid), len(piece_grid[0])
    offset_r, offset_c = (grid_size - rows) // 2, (grid_size - cols) // 2
    
    # 畫細格線
    for i in range(grid_size + 1):
        cv2.line(canvas, (0, i*u), (grid_size*u, i*u), (50, 50, 50), 1)
        cv2.line(canvas, (i*u, 0), (i*u, grid_size*u), (50, 50, 50), 1)
        
    for r in range(rows):
        for c in range(cols):
            if piece_grid[r][c]:
                tr, tc = r + offset_r, c + offset_c
                cv2.rectangle(canvas, (tc*u, tr*u), ((tc+1)*u, (tr+1)*u), (200, 160, 0), -1)
                cv2.rectangle(canvas, (tc*u, tr*u), ((tc+1)*u, (tr+1)*u), (100, 80, 0), 1)
    return canvas

# --- 🛠️ 輔助功能 2：上傳圖片到 ImgBB ---
def upload_to_imgbb(file_path):
    with open(file_path, "rb") as file:
        img_base64 = base64.b64encode(file.read())
        data = {"key": IMGBB_API_KEY, "image": img_base64}
        response = requests.post("https://api.imgbb.com/1/upload", data=data)
        if response.status_code == 200:
            return response.json()["data"]["url"]
        return "Upload Failed"

# --- 🛠️ 核心功能：自動簽到與紀錄 ---
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
        existing_data = conn.read(worksheet=SHEET_NAME, ttl=0)
        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
        conn.update(worksheet=SHEET_NAME, data=updated_df)
        return True
    except Exception as e:
        st.error(f"Sheet Sync Error: {e}")
        return False

# --- 1. UI 介面 ---
st.set_page_config(page_title="Block Blast Color Solver", layout="centered")
st.title("🧩 Block Blast Solver (Color Beta)")

file = st.file_uploader("📸 上傳遊戲截圖", type=['png','jpg','jpeg','heic'], key="main_uploader")

if file:
    # A. 自動訪客簽到 (同一張圖不重複簽)
    if "logged_file" not in st.session_state or st.session_state.logged_file != file.name:
        if log_to_sheets("User Visit"):
            st.session_state.logged_file = file.name

    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    
    # 初始化辨識引擎
    eng = VisionEngine(cv_img)
    is_detected = eng.process()

    if is_detected:
        # --- B. 顯示解法建議 ---
        st.header("💡 解法建議")
        solver = LogicSolver()
        sol = solver.solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
        
        if sol:
            step_options = [f"第 {i} 步" for i in range(len(sol) + 1)]
            selected_step_str = st.radio("步驟切換：", options=step_options, horizontal=True)
            current_step = step_options.index(selected_step_str)
            
            # 繪製解法圖形
            base_canvas = eng.warp_orig.copy()
            u = 400 / 8
            p_canvas = np.zeros_like(base_canvas)
            e_canvas = np.zeros_like(base_canvas)
            
            for s in range(current_step):
                p_idx, row, col, cl_rs, cl_cs = sol[s]
                p, color = eng.detected_pieces[p_idx], STEP_COLORS[s % len(STEP_COLORS)]
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            rx, ry = int((col+pc)*u), int((row+pr)*u)
                            cv2.rectangle(p_canvas, (rx, ry), (rx+int(u), ry+int(u)), color, -1)
                
                # 繪製消除特效
                for cr in (cl_rs or []):
                    cv2.rectangle(e_canvas, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                for cc in (cl_cs or []):
                    cv2.rectangle(e_canvas, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)

            # 影像融合
            combined = p_canvas.copy()
            e_mask = cv2.cvtColor(e_canvas, cv2.COLOR_BGR2GRAY) > 0
            combined[e_mask] = cv2.addWeighted(e_canvas, 0.3, p_canvas, 0.7, 0)[e_mask]
            
            d_mask = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY) > 0
            base_canvas[d_mask] = cv2.addWeighted(combined, 0.8, base_canvas, 0.2, 0)[d_mask]
            
            st.image(base_canvas, channels="BGR", use_container_width=True)
        else:
            st.warning("⚠️ 顏色判定後盤面無解，請檢查下方 Debug 圖確認辨識是否正確。")
        
        # --- C. 待放方塊預覽 ---
        st.markdown("---")
        st.subheader("📦 偵測到的方塊")
        cols = st.columns(3)
        for idx, piece in enumerate(eng.detected_pieces):
            piece_img = draw_piece_preview_5x5(piece)
            cols[idx].image(piece_img, channels="BGR", use_container_width=True)

        # --- D. Debug 細節 ---
        with st.expander("🛠️ 查看顏色辨識細節 (Debug)"):
            st.info("白色小點 = 判定為有方塊 | 灰色小點 = 判定為空位 | 左上圓點 = 全域背景採樣色")
            st.image(eng.img_debug, channels="BGR", use_container_width=True)

    else:
        st.error("❌ 無法定位棋盤，請確認截圖是否包含完整邊框且無遮擋。")

    # --- 2. Feedback 回饋系統 ---
    st.markdown("---")
    st.subheader("🚩 辨識錯誤回報")
    with st.form("feedback_form"):
        user_msg = st.text_input("哪裡辨識錯了？或其他想說的話...")
        submit_btn = st.form_submit_button("🚀 送出回報")
        
        if submit_btn:
            try:
                with st.spinner("正在上傳並儲存回饋..."):
                    os.makedirs("temp", exist_ok=True)
                    tmp_path = "temp/feedback_img.jpg"
                    
                    # 如果辨識成功就傳 Debug 圖，失敗就傳原始彩色圖
                    report_img = eng.img_debug if is_detected else cv_img
                    cv2.imwrite(tmp_path, report_img)
                    
                    img_url = upload_to_imgbb(tmp_path)
                    if log_to_sheets(user_msg, img_url):
                        st.success("✅ 已收到您的回報，我們會盡快調教辨識引擎！")
            except Exception as e:
                st.error(f"傳送失敗：{e}")

# --- 頁尾資訊 ---
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Block Blast Solver v2.0 | Color Sensing Logic Enabled
</div>
""", unsafe_allow_html=True)
