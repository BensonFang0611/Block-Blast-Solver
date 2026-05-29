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

# --- 🛠️ 輔助功能 4：紀錄到 Google Sheets（5 欄結構） ---
def log_to_sheets(err_type, detail_info="None", img_url_orig="None", img_url_debug="None"):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        tz = timezone(timedelta(hours=8))
        now_tw = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        
        new_entry = pd.DataFrame([{
            "Timestamp": now_tw, 
            "Feedback_Type": err_type,
            "Detailed_Info": detail_info,
            "Image_Link_Orig": img_url_orig,
            "Image_Link_Debug": img_url_debug
        }])
        
        existing_data = conn.read(worksheet=SHEET_NAME, ttl=0)
        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
        conn.update(worksheet=SHEET_NAME, data=updated_df)
        return True
    except Exception as e:
        st.error(f"Sheet Error: {e}")
        return False

# --- 🛠️ 輔助功能 5：快取核心辨識與解法（解決切換步驟很卡的問題） ---
@st.cache_data(show_spinner=False)
def get_cached_solution(file_bytes):
    # 將 bytes 轉回 OpenCV 圖片
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 執行影像縮放限制
    h, w = img.shape[:2]
    MAX_WIDTH = 1080
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    # 執行 VisionEngine
    eng = VisionEngine(img)
    is_processed = eng.process()
    
    if not is_processed:
        return False, None, None, None
        
    # 執行 LogicSolver 算答案
    solver = LogicSolver()
    sol = solver.solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
    
    # 將前端需要顯示的資料打包回傳
    return True, sol, eng.warp_orig, eng.detected_pieces, eng.img_debug
# --- 💡 第一個彈跳視窗：辨識失敗（自動回報） ---
@st.dialog("❌ 辨識失敗")
def show_failure_dialog(cv_img, error_detail="無法定位棋盤"):
    st.write(f"系統偵測到錯誤（{error_detail}），請問您是否回報此錯誤圖片？")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("回報錯誤", type="primary", use_container_width=True):
            with st.spinner("正在上傳回報資料..."):
                os.makedirs("temp", exist_ok=True)
                
                orig_path = "temp/failure_auto_orig.jpg"
                cv2.imwrite(orig_path, cv_img)
                url_orig = upload_to_imgbb(orig_path)
                
                # 自動定位失敗時，feedback_Type 直接寫入 "自動定位失敗"
                log_to_sheets(
                    err_type="自動定位失敗", 
                    detail_info=error_detail, 
                    img_url_orig=url_orig, 
                    img_url_debug="None"
                )
                
                st.session_state.dialog_closed = True
                st.session_state.show_dialog = False
                st.session_state.show_thanks_dialog = True
                st.session_state.thanks_msg = "✅ 上傳完成，非常感謝您的協助！"
                st.rerun()
    with col2:
        if st.button("取消", use_container_width=True):
            st.session_state.dialog_closed = True
            st.session_state.show_dialog = False
            st.session_state.show_thanks_dialog = True
            st.session_state.thanks_msg = "💡 已取消回報，感謝您！"
            st.rerun()

# --- 💡 第二個彈跳視窗：系統提示 ---
@st.dialog("🔔 系統提示")
def show_thanks_dialog(msg):
    st.write(msg)
    if st.button("確定", use_container_width=True):
        st.session_state.show_thanks_dialog = False
        st.rerun()

# --- 1. UI 介面 ---
st.set_page_config(page_title="Block Blast Solver", layout="centered")
st.title("🧩 Block Blast Solver ")

file = st.file_uploader("📸 上傳截圖(5/26)", type=['png','jpg','jpeg','heic'], key="uploader")

if file is None:
    for key in ["show_dialog", "dialog_closed", "show_thanks_dialog", "thanks_msg", "last_file_id", "logged_file", "current_error_msg"]:
        st.session_state.pop(key, None)

if file:
    current_file_id = getattr(file, "file_id", str(file.size) + file.name)
    
    if "last_file_id" not in st.session_state or st.session_state.last_file_id != current_file_id:
        for key in ["show_dialog", "dialog_closed", "show_thanks_dialog", "thanks_msg", "current_error_msg"]:
            st.session_state.pop(key, None)
        st.session_state.last_file_id = current_file_id
        
    # ✨ User Visit 簽到
    if "logged_file" not in st.session_state or st.session_state.logged_file != current_file_id:
        if log_to_sheets(err_type="User Visit", detail_info="None"):
            st.session_state.logged_file = current_file_id

    # 讀取原始影像
    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)

    # --- 🎯 影像預處理：降低至固定畫質 ---
    h, w = cv_img.shape[:2]
    MAX_WIDTH = 1080 
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 💡 改用快取機制讀取答案
    file.seek(0)  # 重設檔案指針
    file_bytes = file.read()

    is_processed = False
    sol, eng_warp_orig, eng_detected_pieces = None, None, None

    try:
        # 呼叫快取函數 (內部已包含影像降畫質、VisionEngine 與 LogicSolver 窮舉)
        is_processed, sol, eng_warp_orig, eng_detected_pieces, eng_img_debug = get_cached_solution(file_bytes)
    except Exception as e:
        st.session_state.current_error_msg = f"IndexError 或核心引擎崩潰: {type(e).__name__}"
        is_processed = False

    if is_processed:
        st.header("💡 解法建議")
        
        if sol:
            step_label = st.radio("步驟切換：", [f"第 {i} 步" for i in range(len(sol) + 1)], horizontal=True)
            idx = int(step_label.split(' ')[1])
            
            # --- 繪製解法示意圖 ---
            # 🎯 注意：這裡要改成快取傳回來的變數名稱（有底線的）
            canvas = eng_warp_orig.copy()
            u = 400 / 8
            
            for s in range(idx):
                p_idx, row, col, cl_rs, cl_cs = sol[s]
                # 🎯 注意：這裡也要改成 eng_detected_pieces
                p = eng_detected_pieces[p_idx]
                color = STEP_COLORS[s % 3]
                
                for pr in range(len(p)):
                    for pc in range(len(p[0])):
                        if p[pr][pc]:
                            x1, y1 = int((col+pc)*u), int((row+pr)*u)
                            x2, y2 = int((col+pc+1)*u), int((row+pr+1)*u)
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)
                            
                if cl_rs or cl_cs:
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
    st.warning("此盤面無解:..)")
st.markdown("---")

# 🎯 修改：直接抓取視覺引擎內切出的實際 ROI 畫面
# 備註：請根據你的 vision_engine.py 實際變數名稱調整（例如 eng_pieces_rois）
# 這裡假設 get_cached_solution 也有把原始的 rois 傳回來，或者 eng 內有保留
if 'eng_img_debug' in locals():
    # 如果你的核心引擎有存下各個待放方塊的 ROI list，可以直接用 hstack 拼接
    # 範例：假設變數叫 eng.pieces_rois
    try:
        # 這裡需要確認你的 vision_engine 傳回來的物件或內部屬性
        # 如果 get_cached_solution 有傳出包含 ROI 的物件，可以直接拼接它
        from vision_engine import VisionEngine
        
        # 假設我們從偵測到的方塊去回推，或者你的 eng 物件有保留原始裁切 list
        # 這裡示範直接水平拼接多張 ROI 圖片：
        rois = [p.roi_img for p in eng_detected_pieces if hasattr(p, 'roi_img')]
        
        if rois:
            # 確保所有 ROI 高度一致再進行拼接（或是直接用 st.columns 分開顯示）
            max_h = max(r.shape[0] for r in rois)
            resized_rois = [cv2.resize(r, (int(r.shape[1] * max_h / r.shape[0]), max_h)) for r in rois]
            roi_combined = np.hstack(resized_rois)
            
            st.image(roi_combined, caption="實際偵測到的待放方塊 ROI 畫面", channels="BGR", use_container_width=True)
        else:
            # 如果無法從物件取得，改用 st.columns 呈現 debug 畫面中下方的區域
            st.info("無法取得單一 ROI，改為顯示除錯輔助畫面的待放區")
            if eng_img_debug is not None:
                st.image(eng_img_debug, caption="系統辨識除錯圖", channels="BGR", use_container_width=True)
    except Exception as e:
        st.error(f"無法顯示 ROI 畫面: {e}")
        
    else:
        # ❌ 辨識失敗
        st.error("❌ 無法精確定位棋盤，請確認截圖是否有完整邊框。")
        if "dialog_closed" not in st.session_state:
            st.session_state.show_dialog = True

    # ==========================================
    # 💡 彈跳視窗互斥控制中心
    # ==========================================
    if st.session_state.get("show_dialog", False):
        err_msg = st.session_state.get("current_error_msg", "無法定位棋盤")
        show_failure_dialog(cv_img, error_detail=err_msg)
    elif st.session_state.get("show_thanks_dialog", False):
        show_thanks_dialog(st.session_state.get("thanks_msg", ""))

    # --- 2. Feedback 回饋系統 ---
    st.markdown("---")
    st.subheader("🚩 Feedback 錯誤回報")
    
    with st.form("feedback_form"):
        feedback_Type = st.selectbox(
            "請選擇發生的錯誤類型：",
            [
                "系統提示無解，但實際上還有解法",
                "大棋盤格辨識錯誤",
                "下方待放方塊辨識錯誤",
                "點擊步驟切換時，畫面顯示異常",
                "其他（請在下方補充說明）"
            ]
        )
        
        other_detail = st.text_input("其他原因或詳細補充說明：(選填)")
        
        if st.form_submit_button("🚀 送出"):
            with st.spinner("同步中..."):
                # ✨ 移除「手動回報:」前綴，直接將選單的文字當成 feedback_Type
                final_type = feedback_Type
                
                # 處理 Detailed_Info 的內容
                final_detail = other_detail if "其他" in feedback_Type else "未填寫補充說明"
                if not "其他" in feedback_Type and other_detail:
                    final_detail = other_detail  # 非其他選項但有寫備註時，也存入詳細資訊
                
                os.makedirs("temp", exist_ok=True)
                orig_path = "temp/feedback_orig.jpg"
                debug_path = "temp/feedback_debug.jpg"
                
                # 儲存固定畫質的原圖
                cv2.imwrite(orig_path, cv_img)
                
                # 檢查是否有產出 debug 圖片
                has_debug = 'eng_img_debug' in locals() and eng_img_debug is not None
                cv2.imwrite(debug_path, eng_img_debug if has_debug else cv_img)
                
                # 上傳雙圖至 ImgBB
                url_orig = upload_to_imgbb(orig_path)
                url_debug = upload_to_imgbb(debug_path) if has_debug else "None"
                
                # 寫入 5 欄 (已修正為帶入正確的變數名稱)
                if log_to_sheets(err_type=final_type, detail_info=final_detail, img_url_orig=url_orig, img_url_debug=url_debug):
                    st.success("✅ 感謝您的回饋！將根據這張圖片進行優化。")

st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 50px;'>
        Block Blast Solver Beta v2.1 | Powered by Color Sensing Engine
    </div>
""", unsafe_allow_html=True)
