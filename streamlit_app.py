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
    canvas = np.zeros((grid_size*u, grid_size*u, 3), dtype=np.uint8)
    rows, cols = len(piece_grid), len(piece_grid[0])
    offset_r, offset_c = (grid_size - rows) // 2, (grid_size - cols) // 2
    
    for i in range(grid_size + 1):
        cv2.line(canvas, (0, i*u), (grid_size*u, i*u), (40, 40, 40), 1)
        cv2.line(canvas, (i*u, 0), (i*u, grid_size*u), (40, 40, 40), 1)
        
    for r in range(rows):
        for c in range(cols):
            if piece_grid[r][c]:
                tr, tc = r + offset_r, c + offset_c
                cv2.rectangle(canvas, (tc*u, tr*u), ((tc+1)*u, (tr+1)*u), (200, 160, 0), -1)
                cv2.rectangle(canvas, (tc*u, tr*u), ((tc+1)*u, (tr+1)*u), (100, 80, 0), 1)
    return canvas

# --- 🛠️ 輔助功能 2：水平縫合待放方塊 ---
def get_combined_pieces_image(detected_pieces):
    if not detected_pieces:
        return None
    piece_imgs = [draw_piece_preview_5x5(p) for p in detected_pieces[:3]]
    h, w, c = piece_imgs[0].shape
    gap_width = 15
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

# --- 🛠️ 輔助功能 4：紀錄到 Google Sheets ---
def log_to_sheets(msg, img_url="None"):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        tz = timezone(timedelta(hours=8))
        now_tw = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame([{"Timestamp": now_tw, "Comment": msg, "Image_Link": img_url}])
        existing_data = conn.read(worksheet=SHEET_NAME, ttl=0)
        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
        conn.update(worksheet=SHEET_NAME, data=updated_df)
        return True
    except Exception as e:
        st.error(f"Sheet Error: {e}")
        return False

# --- 💡 第一個彈跳視窗：辨識失敗 ---
@st.dialog("❌ 辨識失敗")
def show_failure_dialog(eng, cv_img):
    st.write("無法定位棋盤，請問您是否回報錯誤圖片？")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("否，取消", use_container_width=True):
            st.session_state.dialog_closed = True
            st.session_state.show_dialog = False
            st.session_state.show_thanks_dialog = True
            st.session_state.thanks_msg = "💡 已取消回報，感謝您！"
            st.rerun()
    with col2:
        if st.button("是，回報錯誤", type="primary", use_container_width=True):
            with st.spinner("正在上傳回報資料..."):
                os.makedirs("temp", exist_ok=True)
                report_path = "temp/feedback_auto.jpg"
                cv2.imwrite(report_path, eng.img_debug if 'eng' in locals() and hasattr(eng, 'img_debug') else cv_img)
                url = upload_to_imgbb(report_path)
                log_to_sheets("系統自動回報：無法定位棋盤", url)
                st.session_state.dialog_closed = True
                st.session_state.show_dialog = False
                st.session_state.show_thanks_dialog = True
                st.session_state.thanks_msg = "✅ 上傳完成，非常感謝您的協助！"
                st.rerun()

# --- 💡 第二個彈跳視窗：系統提示 ---
@st.dialog("🔔 系統提示")
def show_thanks_dialog(msg):
    st.write(msg)
    if st.button("確定", use_container_width=True):
        st.session_state.show_thanks_dialog = False
        st.rerun()

# --- 主 UI 介面 ---
st.set_page_config(page_title="Block Blast Solver", layout="centered")
st.title("🧩 Block Blast Solver")
file = st.file_uploader("📸 上傳截圖", type=['png','jpg','jpeg','heic'], key="uploader")

if file is None:
    for key in ["show_dialog", "dialog_closed", "show_thanks_dialog", "thanks_msg", "last_file_id", "logged_file"]:
        st.session_state.pop(key, None)

if file:
    current_file_id = getattr(file, "file_id", str(file.size) + file.name)
    if "last_file_id" not in st.session_state or st.session_state.last_file_id != current_file_id:
        for key in ["show_dialog", "dialog_closed", "show_thanks_dialog", "thanks_msg"]:
            st.session_state.pop(key, None)
        st.session_state.last_file_id = current_file_id
        
    if "logged_file" not in st.session_state or st.session_state.logged_file != current_file_id:
        if log_to_sheets("User Visit"):
            st.session_state.logged_file = current_file_id

    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    
    eng = VisionEngine(cv_img)
    if eng.process():
        st.header("💡 解法建議")
        solver = LogicSolver()
        sol = solver.solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
        
        if sol:
            step_label = st.radio("步驟切換：", [f"第 {i} 步" for i in range(len(sol)+1)], horizontal=True)
            idx = int(step_label.split(' ')[1])
            
            # 每次重新渲染時，先拿完全乾淨的原圖透視底色當畫布
            canvas = eng.warp_orig.copy()
            u = 400 / 8 
            
            for s in range(idx):
                p_idx, row, col, cl_rs, cl_cs = sol[s]
                p, color = eng.detected_pieces[p_idx], STEP_COLORS[s % 3]
                
                # 1. 繪製當下這步的方塊本體與黑色小網格線
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            x1, y1 = int((col+pc)*u), int((row+pr)*u)
                            x2, y2 = int((col+pc+1)*u), int((row+pr+1)*u)
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)
                            
                # 2. 💡【重大修復】：即時模擬真實消除
                if cl_rs or cl_cs:
                    # 如果這一步「不是」使用者停下來看的最後一步 ➔ 必須在畫布上將其「完全擦除」恢復成乾淨背景
                    if s < idx - 1:
                        if cl_rs:
                            for cr in cl_rs:
                                canvas[int(cr*u):int((cr+1)*u), :] = eng.warp_orig[int(cr*u):int((cr+1)*u), :]
                        if cl_cs:
                            for cc in cl_cs:
                                canvas[:, int(cc*u):int((cc+1)*u)] = eng.warp_orig[:, int(cc*u):int((cc+1)*u)]
                    else:
                        # 如果這步「剛好是」使用者點擊停留觀看的最後一步 ➔ 繪製半透明灰色作為消除動畫提示
                        overlay = canvas.copy()
                        if cl_rs:
                            for cr in cl_rs:
                                cv2.rectangle(overlay, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                        if cl_cs:
                            for cc in cl_cs:
                                cv2.rectangle(overlay, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)
                        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
                    
            st.image(canvas, channels="BGR", use_container_width=True)
        else:
            st.warning("此盤面無解 :..)")
            
        st.markdown("---")
        combined_piece_img = get_combined_pieces_image(eng.detected_pieces)
        if combined_piece_img is not None:
            st.image(combined_piece_img, caption="偵測到的待放方塊 (並排預覽)", channels="BGR", use_container_width=True)
    else:
        st.error("❌ 無法精確定位棋盤，請確認截圖是否有完整邊框。")
        if "dialog_closed" not in st.session_state:
            st.session_state.show_dialog = True

if st.session_state.get("show_dialog", False):
    show_failure_dialog(eng, cv_img)
elif st.session_state.get("show_thanks_dialog", False):
    show_thanks_dialog(st.session_state.get("thanks_msg", ""))

st.markdown("---")
st.subheader("🚩 Feedback 錯誤回報")
with st.form("feedback_form"):
    msg = st.text_input("如果有辨識錯誤，請告訴我!!")
    if st.form_submit_button("🚀 送出"):
        with st.spinner("同步中..."):
            os.makedirs("temp", exist_ok=True)
            report_path = "temp/feedback.jpg"
            cv2.imwrite(report_path, eng.img_debug if 'eng' in locals() else cv_img)
            url = upload_to_imgbb(report_path)
            if log_to_sheets(msg, url):
                st.success("✅ 感謝您的回饋！將根據這張圖片進行優化。")

st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 50px;'>
        Block Blast Solver Beta v2.2 | Powered by Color Sensing Engine
    </div>
""", unsafe_allow_html=True)
