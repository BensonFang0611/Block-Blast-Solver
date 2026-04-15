import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import os
import pandas as pd
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
from vision_engine import VisionEngine, LogicSolver

register_heif_opener()

# --- 配置區 (請填寫你的 Google Drive 資料夾 ID) ---
DRIVE_FOLDER_ID = "1abc123_xyz789_LMNOP"

st.set_page_config(page_title="Block Blast Solver Beta", layout="wide")

# --- 輔助函式 ---
def draw_piece_preview(piece_grid):
    u = 20
    canvas = np.zeros((len(piece_grid)*u, len(piece_grid[0])*u, 3), dtype=np.uint8) + 255
    for r in range(len(piece_grid)):
        for c in range(len(piece_grid[0])):
            if piece_grid[r][c]:
                cv2.rectangle(canvas, (c*u, r*u), ((c+1)*u, (r+1)*u), (255, 120, 0), -1)
            cv2.rectangle(canvas, (c*u, r*u), ((c+1)*u, (r+1)*u), (200, 200, 200), 1)
    return canvas

def upload_to_drive(file_path, file_name):
    creds_info = st.secrets["connections"]["gsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/drive.file"])
    service = build('drive', 'v3', credentials=creds)
    metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(file_path, mimetype='image/jpeg')
    file = service.files().create(body=metadata, media_body=media, fields='webViewLink').execute()
    service.permissions().create(fileId=file.get('id'), body={'type': 'anyone', 'role': 'reader'}).execute()
    return file.get('webViewLink')

# --- UI 開始 ---
st.title("🧩 Block Blast Solver Beta")
file = st.file_uploader("📸 上傳截圖", type=['png','jpg','jpeg','heic'])

if file:
    raw_pil = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)
    
    @st.cache_resource
    def process_image(_img):
        eng = VisionEngine(_img)
        if eng.process():
            sol = LogicSolver().solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
            return eng, sol
        return None, None

    eng, sol = process_image(cv_img)
    if eng:
        st.header("📦 辨識方塊")
        p_cols = st.columns(3)
        for i, p in enumerate(eng.detected_pieces):
            with p_cols[i]: st.image(draw_piece_preview(p), caption=f"方塊 {i+1}")

        st.header("💡 最佳解法")
        if sol:
            step = st.radio("步驟：", [f"第 {i} 步" for i in range(len(sol)+1)], horizontal=True)
            step_idx = int(step.split(' ')[1])
            canvas = eng.warp_orig.copy()
            u = 400/8
            # 繪製邏輯省略... (維持之前的多彩繪圖代碼)
            st.image(canvas, channels="BGR", use_container_width=True)
        else:
            st.warning("⚠️ 此盤面無解。")

        # --- 進階 Feedback ---
        st.markdown("---")
        with st.form("feedback"):
            msg = st.text_input("補充說明")
            if st.form_submit_button("🚀 回傳數據至雲端"):
                try:
                    report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.makedirs("temp", exist_ok=True)
                    tmp_path = f"temp/{report_id}.jpg"
                    cv2.imwrite(tmp_path, eng.img_debug)
                    
                    # 上傳 Drive & 更新 Sheet
                    link = upload_to_drive(tmp_path, f"{report_id}.jpg")
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    new_df = pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                            "Issue": msg, "Link": link}])
                    old_df = conn.read(worksheet="Sheet1", ttl=0)
                    conn.update(worksheet="Sheet1", data=pd.concat([old_df, new_df], ignore_index=True))
                    st.success("✅ 圖片與數據已同步至 Google Sheet！")
                except Exception as e: st.error(f"失敗：{e}")
