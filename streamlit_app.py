import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests
import base64
import pandas as pd
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
from vision_engine import VisionEngine, LogicSolver

# --- 🚀 核心配置 ---
IMGBB_API_KEY = "3fcf87a9eaae07555706aa02519e78c9"
SHEET_NAME = "Sheet1" 

# 顏色定義
STEP_COLORS = [(0, 255, 255), (255, 100, 255), (100, 255, 100)]
GRAY_ELIMINATED = (100, 100, 100)

# --- 🛠️ 輔助功能 1：繪製 5x5 置中框 (使用你指定的顏色) ---
def draw_piece_preview_5x5(piece_grid):
    grid_size = 5
    u = 40  # 每個小格子的像素大小
    # 畫布初始化：+ 0 代表純黑背景
    canvas = np.zeros((grid_size*u, grid_size*u, 3), dtype=np.uint8) + 0 
    
    rows, cols = len(piece_grid), len(piece_grid[0])
    offset_r = (grid_size - rows) // 2
    offset_c = (grid_size - cols) // 2
    
    # 1. 繪製背景 5x5 淺灰色網格線 (BGR: 235, 235, 235)
    for i in range(grid_size + 1):
        cv2.line(canvas, (0, i*u), (grid_size*u, i*u), (235, 235, 235), 1)
        cv2.line(canvas, (i*u, 0), (i*u, grid_size*u), (235, 235, 235), 1)
    
    # 2. 繪製方塊本體
    for r in range(rows):
        for c in range(cols):
            if piece_grid[r][c]:
                target_r, target_c = r + offset_r, c + offset_c
                # 繪製方塊填充 (BGR: 0, 160, 200) -> 亮橘色/金黃
                cv2.rectangle(canvas, (target_c*u, target_r*u), 
                              ((target_c+1)*u, (target_r+1)*u), (0, 160, 200), -1)
                # 繪製方塊邊框 (BGR: 0, 80, 100) -> 深橘色
                cv2.rectangle(canvas, (target_c*u, target_r*u), 
                              ((target_c+1)*u, (target_r+1)*u), (0, 80, 100), 1)
    return canvas

# --- 🛠️ 輔助功能 2：水平縫合方塊 (配合黑色背景) ---
def get_combined_pieces_image(detected_pieces):
    targets = detected_pieces[:3]
    if not targets: return None
    
    piece_imgs = [draw_piece_preview_5x5(p) for p in targets]
    h, w, c = piece_imgs[0].shape
    
    # 縫隙改為黑色 (+ 0)，與背景融為一體
    gap_width = 15
    black_gap = np.zeros((h, gap_width, c), dtype=np.uint8) + 0 

    stack_list = []
    for i, img in enumerate(piece_imgs):
        stack_list.append(img)
        if i < len(piece_imgs) - 1:
            stack_list.append(black_gap)
            
    return np.hstack(stack_list)

# --- 🛠️ 輔助功能 3：上傳圖片到 ImgBB ---
def upload_to_imgbb(file_path):
    with open(file_path, "rb") as file:
        img_base64 = base64.b64encode(file.read())
        data = {"key": IMGBB_API_KEY, "image": img_base64}
        response = requests.post("https://api.imgbb.com/1/upload", data=data)
        return response.json()["data"]["url"]

# --- 1. UI 介面 ---
st.title("🧩 Block Blast Solver Beta")
file = st.file_uploader("📸 上傳遊戲截圖", type=['png','jpg','jpeg','heic'], key="uploader_final")

if file:
    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    
    eng = VisionEngine(cv_img)
    if eng.process():
        sol = LogicSolver().solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
        
        # --- 2. 解法展示 ---
        st.header("💡 最佳解法建議")
        if sol:
            step_options = [f"第 {i} 步" for i in range(len(sol) + 1)]
            selected_step_str = st.radio("步驟切換：", options=step_options, horizontal=True)
            current_step = step_options.index(selected_step_str)
            
            base_canvas = eng.warp_orig.copy()
            u = 400 / 8
            p_canvas, e_canvas = np.zeros_like(base_canvas), np.zeros_like(base_canvas)
            
            for s in range(current_step):
                p_idx, row, col, cl_rs, cl_cs = sol[s]
                p, color = eng.detected_pieces[p_idx], STEP_COLORS[s % len(STEP_COLORS)]
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            rx, ry = int((col+pc)*u), int((row+pr)*u)
                            cv2.rectangle(p_canvas, (rx, ry), (rx+int(u), ry+int(u)), color, -1)
                for cr in (cl_rs or []): cv2.rectangle(e_canvas, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                for cc in (cl_cs or []): cv2.rectangle(e_canvas, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)

            combined_t = p_canvas.copy()
            e_mask = cv2.cvtColor(e_canvas, cv2.COLOR_BGR2GRAY) > 0
            combined_t[e_mask] = cv2.addWeighted(e_canvas, 0.3, p_canvas, 0.7, 0)[e_mask]
            d_mask = cv2.cvtColor(combined_t, cv2.COLOR_BGR2GRAY) > 0
            base_canvas[d_mask] = cv2.addWeighted(combined_t, 0.7, base_canvas, 0.3, 0)[d_mask]
            
            st.image(base_canvas, channels="BGR", width='stretch')
        else:
            st.warning("⚠️ 此盤面無解。")
        
        # --- 3. 待放物預覽 (深色風格 + 手機防換行) ---
        st.markdown("---")
        combined_img = get_combined_pieces_image(eng.detected_pieces)
        if combined_img is not None:
            # ✨ 關鍵：這裡會顯示並排不換行的 5x5 方塊圖
            st.image(combined_img, caption="方塊預覽", width='stretch')

        # --- 4. Feedback 反饋系統 ---
        st.markdown("---")
        with st.form("feedback_form"):
            msg = st.text_input("有什麼bug，或有想說的...")
            if st.form_submit_button("🚀 同步數據至 Google Sheet"):
                try:
                    with st.spinner("同步中..."):
                        os.makedirs("temp", exist_ok=True)
                        tmp_path = "temp/report_latest.jpg"
                        cv2.imwrite(tmp_path, eng.img_debug)
                        img_url = upload_to_imgbb(tmp_path)
                        conn = st.connection("gsheets", type=GSheetsConnection)
                        new_entry = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Comment": msg, "Image_Link": img_url}])
                        existing_data = conn.read(worksheet=SHEET_NAME, ttl=0)
                        conn.update(worksheet=SHEET_NAME, data=pd.concat([existing_data, new_entry], ignore_index=True))
                        st.success(f"✅ 成功上傳！謝謝你的回饋，你的一小步將會成為人類的一大步！！")
                except Exception as e: st.error(f"同步失敗：{e}")

        with st.expander("🛠️ (Debug)"):
            st.image(eng.img_debug, channels="BGR", width='stretch')
    else: st.error("❌ 無法定位棋盤。")
