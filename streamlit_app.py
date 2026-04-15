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
SHEET_NAME = "Sheet1" # 請確認你的 Google Sheet 分頁名稱

# 顏色定義
STEP_COLORS = [(0, 255, 255), (255, 100, 255), (100, 255, 100)]
GRAY_ELIMINATED = (100, 100, 100)

# --- 🛠️ 輔助功能：繪製 5x5 預覽框 ---
def draw_piece_preview_5x5(piece_grid):
    grid_size = 5 # 固定為 5x5
    u = 30        # 每個小格子的像素大小
    canvas = np.zeros((grid_size*u, grid_size*u, 3), dtype=np.uint8) + 255 # 白色背景
    
    rows, cols = len(piece_grid), len(piece_grid[0])
    
    # 計算置中偏移量
    offset_r = (grid_size - rows) // 2
    offset_c = (grid_size - cols) // 2
    
    # 1. 繪製背景 5x5 淺灰色網格線
    for i in range(grid_size + 1):
        cv2.line(canvas, (0, i*u), (grid_size*u, i*u), (235, 235, 235), 1)
        cv2.line(canvas, (i*u, 0), (i*u, grid_size*u), (235, 235, 235), 1)
    
    # 2. 繪製方塊本體
    for r in range(rows):
        for c in range(cols):
            if piece_grid[r][c]:
                target_r, target_c = r + offset_r, c + offset_c
                # 繪製方塊填充
                cv2.rectangle(canvas, (target_c*u, target_r*u), ((target_c+1)*u, (target_r+1)*u), (255, 120, 0), -1)
                # 繪製方塊邊框
                cv2.rectangle(canvas, (target_c*u, target_r*u), ((target_c+1)*u, (target_r+1)*u), (180, 80, 0), 1)
    return canvas

def upload_to_imgbb(file_path):
    with open(file_path, "rb") as file:
        img_base64 = base64.b64encode(file.read())
        data = {"key": IMGBB_API_KEY, "image": img_base64}
        response = requests.post("https://api.imgbb.com/1/upload", data=data)
        return response.json()["data"]["url"]

# --- 1. UI 介面 ---
st.title("🧩 Block Blast Solver Beta")
file = st.file_uploader("📸 上傳遊戲截圖", type=['png','jpg','jpeg','heic'], key="game_uploader_pro")

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
            piece_canvas = np.zeros_like(base_canvas)
            elimination_canvas = np.zeros_like(base_canvas)
            
            for s in range(current_step):
                p_idx, row, col, cleared_rs, cleared_cs = sol[s]
                p = eng.detected_pieces[p_idx]
                color = STEP_COLORS[s % len(STEP_COLORS)]
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            rx, ry = int((col+pc)*u), int((row+pr)*u)
                            cv2.rectangle(piece_canvas, (rx, ry), (rx+int(u), ry+int(u)), color, -1)
                for cr in (cleared_rs or []):
                    cv2.rectangle(elimination_canvas, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                for cc in (cleared_cs or []):
                    cv2.rectangle(elimination_canvas, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)

            # 疊層混合：灰色消除層改為 0.3 透明度
            combined_temp = piece_canvas.copy()
            elim_mask = cv2.cvtColor(elimination_canvas, cv2.COLOR_BGR2GRAY) > 0
            combined_temp[elim_mask] = cv2.addWeighted(elimination_canvas, 0.3, piece_canvas, 0.7, 0)[elim_mask]
            
            data_mask = cv2.cvtColor(combined_temp, cv2.COLOR_BGR2GRAY) > 0
            base_canvas[data_mask] = cv2.addWeighted(combined_temp, 0.7, base_canvas, 0.3, 0)[data_mask]
            
            # 使用 2026 語法 width='stretch'
            st.image(base_canvas, channels="BGR", width='stretch')
        else:
            st.warning("⚠️ 此盤面無解，請檢查下方辨識是否正確。")
        
        # --- 3. 待放物預覽 (核心修改：並排 5x5 框) ---
        st.header("📦 待放方塊 (5x5 視角)")
        p_cols = st.columns(3)
        for i, piece in enumerate(eng.detected_pieces):
            if i < 3: # 確保只顯示前三個
                with p_cols[i]:
                    # 呼叫新的 5x5 繪圖函式
                    p_img_5x5 = draw_piece_preview_5x5(piece)
                    st.image(p_img_5x5, caption=f"方塊 {i+1}", width='stretch')
        
        # --- 4. Feedback 反饋系統 ---
        st.markdown("---")
        with st.form("feedback_form"):
            msg = st.text_input("簡單補充說明")
            submit = st.form_submit_button("🚀 同步數據至 Google Sheet")
            
            if submit:
                try:
                    with st.spinner("同步中..."):
                        os.makedirs("temp", exist_ok=True)
                        tmp_path = "temp/report_latest.jpg"
                        cv2.imwrite(tmp_path, eng.img_debug)
                        img_url = upload_to_imgbb(tmp_path)
                        
                        conn = st.connection("gsheets", type=GSheetsConnection)
                        new_entry = pd.DataFrame([{
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Comment": msg,
                            "Image_Link": img_url,
                            "Image_Preview": f'=IMAGE("{img_url}")'
                        }])
                        existing_data = conn.read(worksheet=SHEET_NAME, ttl=0)
                        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
                        conn.update(worksheet=SHEET_NAME, data=updated_df)
                        st.success("✅ 成功！圖片已自動存入表格預覽。")
                except Exception as e:
                    st.error(f"同步失敗：{e}")

        with st.expander("🛠️ 視覺辨識分析 (Debug)"):
            st.image(eng.img_debug, channels="BGR", width='stretch')
    else:
        st.error("❌ 無法定位棋盤，請確保截圖清晰且包含邊框。")
