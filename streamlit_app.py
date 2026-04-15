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
GRAY_ELIMINATED = (80, 80, 80) # 消除後的半透明灰色

# --- 🛠️ 輔助功能 ---

def upload_to_imgbb(file_path):
    """上傳圖片到 ImgBB 並回傳網址"""
    try:
        with open(file_path, "rb") as file:
            img_base64 = base64.b64encode(file.read())
            data = {"key": IMGBB_API_KEY, "image": img_base64}
            response = requests.post("https://api.imgbb.com/1/upload", data=data)
            if response.status_code == 200:
                return response.json()["data"]["url"]
        return "Upload Failed"
    except:
        return "Error"

def log_to_sheets(msg, img_url="None"):
    """將紀錄同步至 Google Sheets"""
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
    except:
        return False

def draw_piece_preview_5x5(piece_grid):
    """繪製 5x5 的待放方塊預覽圖"""
    u = 30
    canvas = np.zeros((150, 150, 3), dtype=np.uint8)
    r_off, c_off = (5 - len(piece_grid)) // 2, (5 - len(piece_grid[0])) // 2
    for r in range(len(piece_grid)):
        for c in range(len(piece_grid[0])):
            if piece_grid[r][c]:
                # 畫方塊與邊框
                cv2.rectangle(canvas, ((c+c_off)*u, (r+r_off)*u), ((c+c_off+1)*u, (r+r_off+1)*u), (0, 160, 200), -1)
                cv2.rectangle(canvas, ((c+c_off)*u, (r+r_off)*u), ((c+c_off+1)*u, (r+r_off+1)*u), (0, 80, 100), 1)
    return canvas

# --- 1. UI 介面設定 ---
st.set_page_config(page_title="Block Blast Color Solver", layout="centered")

# --- 訪客長駐提示 (側邊欄) ---
with st.sidebar:
    st.title("使用指南 📖")
    st.info("""
    1. 截圖時請包含完整遊戲畫面。
    2. 確保三個待放方塊清晰可見。
    3. 如果辨識錯誤，請點擊下方回饋。
    """)
    st.write("---")
    st.write("v2.5 純顏色辨識版")

st.title("🧩 Block Blast Solver")

# --- 訪客進場歡迎提示 ---
if "welcome_hint" not in st.session_state:
    st.toast("👋 歡迎回來！上傳截圖即可開始解題。")
    st.session_state.welcome_hint = True

file = st.file_uploader("📸 上傳遊戲截圖", type=['png','jpg','jpeg','heic'])

if file:
    # A. 自動訪客簽到邏輯
    if "last_logged_file" not in st.session_state or st.session_state.last_logged_file != file.name:
        if log_to_sheets("Visitor Uploaded File"):
            st.session_state.last_logged_file = file.name
            st.toast("✅ 圖片已成功接收，正在分析顏色...")

    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    
    # 初始化辨識引擎
    eng = VisionEngine(cv_img)
    
    with st.spinner("辨識引擎運作中..."):
        is_detected = eng.process()

    if is_detected:
        # --- B. 顯示解法建議 ---
        st.header("💡 解法建議")
        solver = LogicSolver()
        sol = solver.solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
        
        if sol:
            # 訪客提示：步驟切換
            st.success("🎉 已找到最佳路徑！請切換下方步驟觀看。")
            step_label = st.radio("步驟切換：", [f"第 {i} 步" for i in range(len(sol)+1)], horizontal=True)
            current_step = int(step_label.split(' ')[1])
            
            # 繪製解法
            base_canvas = eng.warp_orig.copy()
            u = 400 / 8
            
            for s in range(current_step):
                p_idx, row, col, cl_rs, cl_cs = sol[s]
                p, color = eng.detected_pieces[p_idx], STEP_COLORS[s % 3]
                
                # 繪製方塊 + 黑色格線
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            x1, y1 = int((col+pc)*u), int((row+pr)*u)
                            x2, y2 = int((col+pc+1)*u), int((row+pr+1)*u)
                            cv2.rectangle(base_canvas, (x1, y1), (x2, y2), color, -1)
                            cv2.rectangle(base_canvas, (x1, y1), (x2, y2), (0, 0, 0), 1) # 加入格線

                # 繪製消除特效
                overlay = base_canvas.copy()
                for cr in (cl_rs or []): cv2.rectangle(overlay, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                for cc in (cl_cs or []): cv2.rectangle(overlay, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)
                cv2.addWeighted(overlay, 0.4, base_canvas, 0.6, 0, base_canvas)
            
            st.image(base_canvas, channels="BGR", use_container_width=True)
        else:
            st.warning("⚠️ 此盤面無解。這可能是辨識誤差造成的，請檢查下方 Debug 圖。")
        
        # --- C. 待放方塊預覽 ---
        st.markdown("---")
        st.subheader("📦 偵測到的方塊")
        p_cols = st.columns(3)
        for idx, piece in enumerate(eng.detected_pieces):
            piece_img = draw_piece_preview_5x5(piece)
            p_cols[idx].image(piece_img, channels="BGR", use_container_width=True)

        # --- D. Debug 詳細資訊 ---
        with st.expander("🛠️ 辨識細節 (訪客 Debug 模式)"):
            st.write("白色小點：偵測為方塊 | 灰色小點：偵測為空位")
            st.image(eng.img_debug, channels="BGR", use_container_width=True)

    else:
        st.error("❌ 無法定位棋盤。請確認截圖是否完整，且沒有被手機的導航欄遮擋。")

    # --- 2. Feedback 回饋系統 ---
    st.markdown("---")
    st.subheader("🚩 錯誤回報與建議")
    with st.form("feedback_form"):
        user_msg = st.text_input("如果有辨識錯誤或想說的話...", placeholder="例如：第三個方塊辨識錯了...")
        submit_btn = st.form_submit_button("🚀 送出回饋")
        
        if submit_btn:
            if user_msg.strip() == "":
                st.warning("請輸入一些內容再送出唷！")
            else:
                with st.spinner("回饋同步中..."):
                    img_path = "debug_report.jpg"
                    # 有辨識圖就傳辨識圖，沒辨識成功就傳原始圖
                    cv2.imwrite(img_path, eng.img_debug if is_detected else cv_img)
                    
                    url = upload_to_imgbb(img_path)
                    log_to_sheets(f"Feedback: {user_msg}", url)
                    st.success("✅ 已收到回報！謝謝你幫助我們變得更好。")
                    st.toast("感謝回饋！💖")

else:
    # --- 尚未上傳圖片時的提示 ---
    st.info("💡 訪客提示：請上傳遊戲截圖以獲得解法建議。")

# 頁尾
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 50px;'>
    Block Blast Solver v2.5 | 使用顏色感知技術
</div>
""", unsafe_allow_html=True)
