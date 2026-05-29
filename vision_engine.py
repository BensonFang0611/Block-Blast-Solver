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

# --- 🛠️ 輔助功能：將 OpenCV 影像轉為 Base64 標籤（修改：改為完全透明背景） ---
def convert_bgr_to_base64_html(img_bgr):
    try:
        _, buffer = cv2.imencode('.png', img_bgr)
        b64_str = base64.b64encode(buffer).decode()
        # 🎯 將 background-color 設為 transparent，讓 ROI 截圖與 Streamlit 網頁背景完美融合
        return f"""
        <div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 150px; background-color: transparent;">
            <img src="data:image/png;base64,{b64_str}" style="max-width: 100%; max-height: 100%; object-fit: scale-down;" />
        </div>
        """
    except:
        return ""

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

# --- 🛠️ 輔助功能 3：紀錄到 Google Sheets ---
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

# --- 🛠️ 輔助功能 4：快取核心辨識與解法 ---
@st.cache_data(show_spinner=False)
def get_cached_solution(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = img.shape[:2]
    MAX_WIDTH = 1080
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    eng = VisionEngine(img)
    is_processed = eng.process()
    if not is_processed:
        return False, None, None, None, None
        
    solver = LogicSolver()
    sol = solver.solve(eng.grid_state, eng.detected_pieces, list(range(len(eng.detected_pieces))))
    return True, sol, eng.warp_orig, eng.detected_pieces, eng.img_debug

# --- 💡 第一個彈跳視窗：辨識失敗 ---
@st.dialog("❌ 辨識失敗")
def show_failure_dialog(cv_img, error_detail="無法定位棋盤"):
    st.write(f"系統偵測到錯誤（`{error_detail}`），請問您是否回報此錯誤圖片？")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("回報錯誤", type="primary", use_container_width=True):
            with st.spinner("正在上傳回報資料..."):
                os.makedirs("temp", exist_ok=True)
                orig_path = "temp/failure_auto_orig.jpg"
                cv2.imwrite(orig_path, cv_img)
                url_orig = upload_to_imgbb(orig_path)
                log_to_sheets(err_type="自動定位失敗", detail_info=error_detail, img_url_orig=url_orig, img_url_debug="None")
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

    if "logged_file" not in st.session_state or st.session_state.logged_file != current_file_id:
        if log_to_sheets(err_type="User Visit", detail_info="None"):
            st.session_state.logged_file = current_file_id

    raw_pil_img = Image.open(file)
    cv_img = cv2.cvtColor(np.array(raw_pil_img), cv2.COLOR_RGB2BGR)

    h, w = cv_img.shape[:2]
    MAX_WIDTH = 1080
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    file.seek(0)  
    file_bytes = file.read()
    is_processed = False
    sol, eng_warp_orig, eng_detected_pieces, eng_img_debug = None, None, None, None

    try:
        is_processed, sol, eng_warp_orig, eng_detected_pieces, eng_img_debug = get_cached_solution(file_bytes)
    except Exception as e:
        st.session_state.current_error_msg = f"{type(e).__name__}: {str(e)}"
        is_processed = False

    if is_processed:
        st.header("💡 解法建議")
        if sol:
            step_label = st.radio("步驟切換：", [f"第 {i} 步" for i in range(len(sol) + 1)], horizontal=True)
            idx = int(step_label.split(' ')[1])

            # --- 繪製解法示意圖 ---
            canvas = eng_warp_orig.copy()
            u = 400 / 8
            for s in range(idx):
                p_idx, row, col, cl_rs, cl_cs = sol[s]
                p = eng_detected_pieces[p_idx]["grid"]
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
        st.write("**🔍 實際偵測到的待放方塊 ROI 畫面**")

        # 🎯 建立三等份槽位，不縮放圖片且各自置中
        if eng_detected_pieces:
            try:
                rois = [p["roi_img"] for p in eng_detected_pieces if isinstance(p, dict) and "roi_img" in p]
                
                p_cols = st.columns(3)
                for i in range(3):
                    with p_cols[i]:
                        if i < len(rois):
                            html_code = convert_bgr_to_base64_html(rois[i])
                            st.markdown(html_code, unsafe_allow_html=True)
                        else:
                            # 🎯 留空槽位也改為透明背景，僅保留更淡的虛線，讓視覺感更乾淨
                            st.markdown("""
                            <div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 150px; background-color: transparent; border: 1px dashed rgba(0,0,0,0.08); border-radius: 5px; color: gray; font-size: 0.8em;">
                                空槽位
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"無法顯示 ROI 畫面: {e}")
    else:
        err_msg = st.session_state.get("current_error_msg", "無法定位棋盤")
        st.error(f"❌ 辨識失敗：{err_msg}")
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
        ["系統提示無解，但實際上還有解法", "大棋盤格辨識錯誤", "下方待放方塊辨識錯誤", "點擊步驟切換時，畫面顯示異常", "其他（請在下方補充說明）"]
    )
    other_detail = st.text_input("其他原因或詳細補充說明：(選填)")
    
    if st.form_submit_button("🚀 送出"):
        with st.spinner("同步中..."):
            final_type = feedback_Type
            final_detail = other_detail if "其他" in feedback_Type else "未填寫補充說明"
            if not "其他" in feedback_Type and other_detail:
                final_detail = other_detail
                
            os.makedirs("temp", exist_ok=True)
            orig_path = "temp/feedback_orig.jpg"
            debug_path = "temp/feedback_debug.jpg"
            
            cv2.imwrite(orig_path, cv_img)
            has_debug = 'eng_img_debug' in locals() and eng_img_debug is not None
            cv2.imwrite(debug_path, eng_img_debug if has_debug else cv_img)
            
            url_orig = upload_to_imgbb(orig_path)
            url_debug = upload_to_imgbb(debug_path) if has_debug else "None"
            
            if log_to_sheets(err_type=final_type, detail_info=final_detail, img_url_orig=url_orig, img_url_debug=url_debug):
                st.session_state.show_thanks_dialog = True
                st.session_state.thanks_msg = "✅ 感謝您的回饋！將根據這張圖片進行優化。"
                st.rerun()

st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 50px;'>
        Block Blast Solver Beta v2.1 | Powered by Color Sensing Engine
    </div>
""", unsafe_allow_html=True)
