import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests
import base64
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account
from vision_engine import VisionEngine, LogicSolver

# --- 🚀 核心配置 ---
DOC_ID = "1XVra9GVl1eNbKZZM5WzsZDzZN-ibJD4JC6sgFemNQW0"
IMGBB_API_KEY = "3fcf87a9eaae07555706aa02519e78c9" 

# 顏色定義
STEP_COLORS = [(0, 255, 255), (255, 100, 255), (100, 255, 100)]
GRAY_ELIMINATED = (100, 100, 100)

# --- 🛠️ 輔助功能 ---
def draw_piece_preview(piece_grid):
    u = 20
    h, w = len(piece_grid), len(piece_grid[0])
    canvas = np.zeros((h*u, w*u, 3), dtype=np.uint8) + 255
    for r in range(h):
        for c in range(w):
            if piece_grid[r][c]:
                cv2.rectangle(canvas, (c*u, r*u), ((c+1)*u, (r+1)*u), (255, 120, 0), -1)
            cv2.rectangle(canvas, (c*u, r*u), ((c+1)*u, (r+1)*u), (200, 200, 200), 1)
    return canvas

def upload_to_imgbb(file_path):
    with open(file_path, "rb") as file:
        img_base64 = base64.b64encode(file.read())
        data = {"key": IMGBB_API_KEY, "image": img_base64}
        response = requests.post("https://api.imgbb.com/1/upload", data=data)
        return response.json()["data"]["url"]

def append_report_to_docs(issue, comment, img_url):
    creds_info = st.secrets["connections"]["gsheets"]
    creds = service_account.Credentials.from_service_account_info(
        creds_info, 
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    requests_body = [
        {'insertText': {'location': {'index': 1}, 'text': f"\n--- 🚩 回報結束 ---\n\n"}},
        {'insertInlineImage': {
            'location': {'index': 1},
            'uri': img_url,
            'objectSize': {'width': {'magnitude': 400, 'unit': 'PT'}}
        }},
        {'insertText': {'location': {'index': 1}, 'text': f"\n回報時間：{timestamp}\n問題類型：{issue}\n補充說明：{comment}\n"}},
        {'insertText': {'location': {'index': 1}, 'text': f"=== 新回報案例 ===\n"}}
    ]
    service.documents().batchUpdate(documentId=DOC_ID, body={'requests': requests_body}).execute()

# --- 1. UI 介面 ---
st.title("🧩 Block Blast Solver Beta")

# 為了確保圖片能換，我們加入一個 unique 的 key
file = st.file_uploader("📸 上傳遊戲截圖", type=['png','jpg','jpeg','heic'], key="game_image_uploader")

if file:
    # 讀取圖片 (開發階段先不使用 @st.cache_resource 避免換圖失敗)
    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)
    
    eng = VisionEngine(cv_img)
    if eng.process():
        # 嘗試求解
        sol = LogicSolver().solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
        
        # --- 2. 解法展示 ---
        st.header("💡 解法建議")
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
                
                # 繪製方塊
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            rx, ry = int((col+pc)*u), int((row+pr)*u)
                            cv2.rectangle(piece_canvas, (rx, ry), (rx+int(u), ry+int(u)), color, -1)
                
                # 繪製消除行列
                for cr in (cleared_rs or []):
                    cv2.rectangle(elimination_canvas, (0, int(cr*u)), (400, int((cr+1)*u)), GRAY_ELIMINATED, -1)
                for cc in (cleared_cs or []):
                    cv2.rectangle(elimination_canvas, (int(cc*u), 0), (int((cc+1)*u), 400), GRAY_ELIMINATED, -1)

            # 混合層次：讓灰色變成半透明，不要蓋死方塊顏色
            combined_temp = piece_canvas.copy()
            elim_mask = cv2.cvtColor(elimination_canvas, cv2.COLOR_BGR2GRAY) > 0
            combined_temp[elim_mask] = cv2.addWeighted(elimination_canvas, 0.3, piece_canvas, 0.7, 0)[elim_mask]

            data_mask = cv2.cvtColor(combined_temp, cv2.COLOR_BGR2GRAY) > 0
            base_canvas[data_mask] = cv2.addWeighted(combined_temp, 0.7, base_canvas, 0.3, 0)[data_mask]
            
            # 使用 2026 新語法 width='stretch'
            st.image(base_canvas, channels="BGR", width='stretch')
        else:
            st.warning("⚠️ 此盤面無解，請確認下方偵測是否正確。")
        
        # --- 3. 待放物預覽 ---
        p_cols = st.columns(3)
        for i, piece in enumerate(eng.detected_pieces):
            with p_cols[i]:
                st.image(draw_piece_preview(piece), caption=f"方塊 {i+1}", width='stretch')
        
        # --- 4. Feedback 反饋系統 ---
        st.markdown("---")
        st.subheader("🚩 判定不準？回傳報告優化 AI")

        with st.form("feedback_form"):
            issue_type = st.selectbox("問題類型", ["定位歪掉", "方塊認錯", "格子認錯", "求解邏輯錯誤"])
            msg = st.text_input("簡單補充說明 (選填)")
            submit = st.form_submit_button("🚀 一鍵同步至 Google Docs")
            
            if submit:
                try:
                    with st.spinner("正在上傳並同步至雲端..."):
                        os.makedirs("temp", exist_ok=True)
                        tmp_path = "temp/report_latest.jpg"
                        cv2.imwrite(tmp_path, eng.img_debug)
                        
                        img_url = upload_to_imgbb(tmp_path)
                        append_report_to_docs(issue_type, msg, img_url)
                        
                        st.success("✅ 報告已送達 Google Docs！")
                        st.link_button("📂 打開筆記本查看", f"https://docs.google.com/document/d/{DOC_ID}/edit")
                except Exception as e:
                    st.error(f"自動化流程失敗：{e}")

        with st.expander("🛠️ 查看視覺辨識除錯圖"):
            st.image(eng.img_debug, channels="BGR", caption="綠框: 網格定位 | 紅框: 偵測物", width='stretch')
    else:
        st.error("❌ 無法定位棋盤，請確保圖片包含完整遊戲邊框。")
