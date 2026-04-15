import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os, requests, base64
import pandas as pd
from datetime import datetime, timedelta, timezone
from streamlit_gsheets import GSheetsConnection
from vision_engine import VisionEngine, LogicSolver

# --- 🚀 配置 ---
IMGBB_API_KEY = "3fcf87a9eaae07555706aa02519e78c9"
SHEET_NAME = "Sheet1"
STEP_COLORS = [(200, 200, 0), (200, 100, 200), (100, 200, 100)] # 亮青、亮粉、亮綠
GRAY_ELIMINATED = (100, 100, 100) # 消除後的顏色

def upload_to_imgbb(file_path):
    with open(file_path, "rb") as file:
        img_base64 = base64.b64encode(file.read())
        data = {"key": IMGBB_API_KEY, "image": img_base64}
        response = requests.post("https://api.imgbb.com/1/upload", data=data)
        return response.json()["data"]["url"] if response.status_code == 200 else "Failed"

def log_to_sheets(msg, img_url="None"):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        tz = timezone(timedelta(hours=8)) 
        now_tw = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame([{"Timestamp": now_tw, "Comment": msg, "Image_Link": img_url}])
        existing_data = conn.read(worksheet=SHEET_NAME, ttl=0)
        conn.update(worksheet=SHEET_NAME, data=pd.concat([existing_data, new_entry], ignore_index=True))
    except: pass

def draw_piece_preview_5x5(piece_grid):
    u = 30
    canvas = np.zeros((150, 150, 3), dtype=np.uint8)
    r_off, c_off = (5 - len(piece_grid)) // 2, (5 - len(piece_grid[0])) // 2
    for r in range(len(piece_grid)):
        for c in range(len(piece_grid[0])):
            if piece_grid[r][c]:
                cv2.rectangle(canvas, ((c+c_off)*u, (r+r_off)*u), ((c+c_off+1)*u, (r+r_off+1)*u), (0, 160, 200), -1)
                cv2.rectangle(canvas, ((c+c_off)*u, (r+r_off)*u), ((c+c_off+1)*u, (r+r_off+1)*u), (0, 80, 100), 1)
    return canvas

st.title("🧩 Block Blast Solver")
file = st.file_uploader("📸 上傳遊戲截圖", type=['png','jpg','jpeg','heic'])

if file:
    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    eng = VisionEngine(cv_img)
    
    if eng.process():
        st.header("💡 解法建議")
        solver = LogicSolver()
        sol = solver.solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
        
        if sol:
            step_label = st.radio("步驟切換：", [f"第 {i} 步" for i in range(len(sol)+1)], horizontal=True)
            current_step = int(step_label.split(' ')[1])
            
            # --- ✨ 繪製解法邏輯 (含格線) ---
            canvas = eng.warp_orig.copy()
            u = 400 / 8
            
            for s in range(current_step):
                p_idx, row, col, cl_rs, cl_cs = sol[s]
                p, color = eng.detected_pieces[p_idx], STEP_COLORS[s % 3]
                
                # 1. 先畫方塊填色與格線
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            x1, y1 = int((col+pc)*u), int((row+pr)*u)
                            x2, y2 = int((col+pc+1)*u), int((row+pr+1)*u)
                            # 填色
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                            # ✨ 加上深色格線 (Border)
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

                # 2. 疊加消除特效 (半透明灰色)
                overlay = canvas.copy()
                for cr in (cl_rs or []): cv2.rectangle(overlay, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                for cc in (cl_cs or []): cv2.rectangle(overlay, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)
                cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            
            st.image(canvas, channels="BGR", use_container_width=True)
        else:
            st.warning("⚠️ 盤面無解，請檢查 Debug 圖。")
            
        st.image(np.hstack([draw_piece_preview_5x5(p) for p in eng.detected_pieces]), caption="偵測到的待放方塊", use_container_width=True)

        with st.expander("🛠️ 顏色辨識 Debug"):
            st.write("白色小點=有方塊 | 灰色小點=空位")
            st.image(eng.img_debug, channels="BGR", use_container_width=True)

    # --- 反饋系統 ---
    st.markdown("---")
    with st.form("feedback"):
        msg = st.text_input("回饋或報錯...")
        if st.form_submit_button("🚀 送出回報"):
            with st.spinner("上傳中..."):
                img_path = "debug_report.jpg"
                cv2.imwrite(img_path, eng.img_debug if 'eng' in locals() else cv_img)
                url = upload_to_imgbb(img_path)
                log_to_sheets(msg, url)
                st.success("回報成功！")
