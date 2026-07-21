import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# تحميل الموديل
model = YOLO("best.pt")

# إعداد الصفحة
st.set_page_config(
    page_title="Ovarian Cyst Detection",
    page_icon="🩺",
    layout="centered"
)

# صورة العنوان
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("ovary.jpg", use_container_width=True)

st.title("🩺 Ovarian Cyst Detection Application")
st.markdown("Upload an ultrasound image to detect ovarian cysts using Artificial Intelligence.")

st.divider()

uploaded_file = st.file_uploader("📤 Upload an ultrasound image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect"):

        with st.spinner("Analyzing image... Please wait"):

            # تحويل الصورة
            image_rgb = image.convert("RGB")
            image_np = np.array(image_rgb)

            # تشغيل الموديل
            results = model(image_np, conf=0.10)

            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:

                num_detections = len(boxes)

                st.success(f"✅ Detection completed! {num_detections} cyst(s) detected.")
                st.subheader("📊 Detection Details")

                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    st.markdown(f"""
                    **Detection {i+1}**
                    - Class: `{class_name}`
                    - Confidence: `{confidence:.2f}`
                    """)

                    st.progress(confidence)

                # رسم النتائج
                result_image = results[0].plot()
                result_image = result_image[..., ::-1]

                st.image(result_image, caption="📌 Detection Result", use_container_width=True)

                # زر تحميل
                img_pil = Image.fromarray(result_image)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="⬇ Download Result Image",
                    data=byte_im,
                    file_name="detection_result.png",
                    mime="image/png"
                )
