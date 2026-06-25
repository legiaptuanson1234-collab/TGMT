import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tempfile
import os

# --- NHẬP CÁC MODULE AI CỦA BẠN ---
from tracking import run_ai_system
from counting import VehicleCounter 

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Traffic AI - UTT", layout="wide", page_icon="🚦")
st.title("🚦 HỆ THỐNG ĐẾM XE & CẢNH BÁO GIAO THÔNG AI")
st.markdown("**Đồ án Kỹ thuật - Sinh viên: Lê Giáp Tuấn Sơn - UTT**")

# --- 2. BẢNG ĐIỀU KHIỂN (MENU TRÁI) ---
with st.sidebar:
    st.header("⚙️ BẢNG ĐIỀU KHIỂN")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("1. Tải Video Lên", type=['mp4', 'avi', 'mov'])
    
    mode = st.radio("2. Chế Độ Phân Tích", ["Đếm Vạch (Line)", "Đếm Vùng (ROI)"])
    
    btn_run = st.button("🚀 KHỞI ĐỘNG AI", type="primary", use_container_width=True)

# --- 3. XỬ LÝ VIDEO BẰNG BỘ NHỚ ĐỆM (CHỐNG LỖI SERVER & LỖI ẢNH TRẮNG) ---
if uploaded_file is not None:
    
    # Kiểm tra xem có phải video mới tải lên không
    if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.uploaded_filename = uploaded_file.name
        
        # 1. TẠO FILE TẠM AN TOÀN TRÊN SERVER
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.getvalue()) # Bốc toàn bộ dung lượng file thật
        tfile.flush()
        tfile.close() # Bắt buộc đóng để Linux lưu 100% xuống đĩa
        
        # Lưu đường dẫn vào bộ nhớ RAM
        st.session_state.video_path = tfile.name
        
        # 2. TRÍCH XUẤT FRAME ĐẦU TIÊN
        cap = cv2.VideoCapture(st.session_state.video_path)
        
        # TUA QUA 30 FRAME ĐẦU (1 GIÂY) ĐỂ TRÁNH DÍNH ẢNH TRẮNG/ĐEN
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30) 
        
        ret, frame = cap.read()
        
        # Quét dự phòng nếu frame bị rỗng
        if not ret:
            for _ in range(10):
                ret, frame = cap.read()
                if ret: break
        cap.release()
        
        # 3. LƯU ẢNH VÀO RAM
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.first_frame = Image.fromarray(frame_rgb)
        else:
            st.session_state.first_frame = None

    # --- 4. HIỂN THỊ CANVAS ĐỂ VẼ ---
    if st.session_state.first_frame is not None:
        image_pil = st.session_state.first_frame

        st.subheader("Bước 1: Vẽ Vạch/Vùng Cảnh Báo")
        st.markdown(f"**Chế độ hiện tại:** {mode}. Hãy dùng chuột click và vẽ trực tiếp lên ảnh dưới đây.")
        
        drawing_mode = "line" if mode == "Đếm Vạch (Line)" else "polygon"
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)", 
            stroke_width=3,
            stroke_color="#00FF00", 
            background_image=image_pil,
            update_streamlit=True,
            height=image_pil.height,
            width=image_pil.width,
            drawing_mode=drawing_mode,
            key="canvas",
        )

        # --- 5. KÍCH HOẠT YOLOv8 ---
        if btn_run:
            st.markdown("---")
            st.subheader("📟 Màn Hình Giám Sát Real-time")
            
            # Kiểm tra xem người dùng đã vẽ gì chưa
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                objects = canvas_result.json_data["objects"]
                st.success("[✔] Đã nhận diện được tọa độ. Đang khởi động AI...")
                
                stframe = st.empty() 
                
                # Trích xuất tọa độ
                points = []
                if drawing_mode == "line":
                    obj = objects[0]
                    left, top = obj["left"], obj["top"]
                    x1, y1 = int(left + obj["x1"]), int(top + obj["y1"])
                    x2, y2 = int(left + obj["x2"]), int(top + obj["y2"])
                    points = [(x1, y1), (x2, y2)]
                else:
                    obj = objects[0]
                    for pt in obj["path"]:
                        if len(pt) >= 3:
                            points.append((int(pt[1]), int(pt[2])))

                # Khởi tạo Object đếm
                counter_obj = VehicleCounter(mode=drawing_mode, points=points) 
                
                try:
                    # Gọi thẳng file video đã lưu trong RAM ra chạy AI
                    run_ai_system("best.pt", st.session_state.video_path, "output.mp4", counter_obj, stframe)
                    st.balloons()
                    st.success("🎉 Luồng phân tích giao thông đã hoàn tất!")
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán AI: {e}")
            else:
                st.error("❌ BẠN CHƯA VẼ VẠCH HAY VÙNG ROI! Vui lòng dùng chuột vẽ lên hình trước khi bấm Khởi Động AI.")
    else:
        st.error("❌ Không thể trích xuất khung hình. Video gốc của bạn có thể bị lỗi định dạng. Vui lòng thử lại.")