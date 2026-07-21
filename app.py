if st.button("🔍 Detect"):
    with st.spinner("Analyzing image... Please wait"):

        # تحويل الصورة إلى RGB ثم numpy
        image_rgb = image.convert("RGB")
        image_np = np.array(image_rgb)

        # تقليل مستوى الثقة عشان ما يفوت اكتشافات
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

        else:
            st.warning("⚠ No cyst detected in the image.")
