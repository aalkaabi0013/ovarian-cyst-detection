import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# تحميل الموديل
model = YOLO("best.pt")

# إعداد الصفحة
st.set_page_config(page_title="Ovarian Cyst Detection", page_icon="🩺", layout="centered")

st.title("🩺 Ovarian Cyst Detection Application")
st.markdown("Upload an ultrasound image to detect ovarian cysts using AI.")

uploaded_file = st.file_uploader("📤 Upload an ultrasound image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect"):
        with st.spinner("Analyzing image..."):
            results = model(image)

            boxes = results[0].boxes

            # عدد الأكياس
            num_detections = len(boxes)

            if num_detections > 0:
                st.success(f"✅ Detection completed! {num_detections} cyst(s) detected.")

                # عرض معلومات كل اكتشاف
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = model.names[class_id]

                    st.write(f"**Detection {i+1}:**")
                    st.write(f"- Class: `{class_name}`")
                    st.write(f"- Confidence: `{confidence:.2f}`")

                # رسم النتائج
                result_image = results[0].plot()
                result_image = result_image[..., ::-1]  # BGR to RGB

                st.image(result_image, caption="📌 Detection Result", use_container_width=True)

                # زر تحميل الصورة
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

            else:
                st.warning("⚠ No cyst detected in the image.")
