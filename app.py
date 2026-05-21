import streamlit as st
from ultralytics import YOLO
import boto3
import pymysql
import tempfile
import glob
import os
from PIL import Image

# --------------------------
# CONFIGURATION
# --------------------------

MODEL_PATH = "runs/detect/train/weights/best.pt"

S3_BUCKET = "drone-detection-storage-boopathy"

RDS_HOST = "drone-detection-db.czyge4aiywpv.ap-south-1.rds.amazonaws.com"
RDS_USER = "admin"
RDS_PASSWORD = "Drone1234"
RDS_DB = "drone_db"

# --------------------------
# LOAD YOLO MODEL
# --------------------------

model = YOLO(MODEL_PATH)

# --------------------------
# STREAMLIT UI
# --------------------------

st.title("🚁 Drone Detection System")

st.write("Upload a drone image and click Detect.")

uploaded_file = st.file_uploader(
    "Upload Drone Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------
# SHOW IMAGE
# --------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    detect_button = st.button("Detect Drone")

    if detect_button:

        # --------------------------
        # SAVE FILE CORRECTLY
        # --------------------------

        file_extension = os.path.splitext(uploaded_file.name)[1]

        file_bytes = uploaded_file.getvalue()

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        )

        temp_file.write(file_bytes)

        temp_file.flush()
        temp_file.close()

        temp_path = temp_file.name

        st.write("Running YOLO detection...")

        # --------------------------
        # YOLO DETECTION
        # --------------------------

        results = model(temp_path, save=True)

        detected_class = None
        confidence = None

        for r in results:
            if len(r.boxes) > 0:
                detected_class = model.names[int(r.boxes.cls[0])]
                confidence = float(r.boxes.conf[0])

        if detected_class is None:
            st.warning("No drone detected in this image")
            st.stop()

        st.success(f"Detected: {detected_class}")
        st.write(f"Confidence: {confidence:.2f}")

        # --------------------------
        # FIND DETECTION IMAGE
        # --------------------------

        predict_folders = glob.glob("runs/detect/predict*")

        latest_folder = max(predict_folders, key=os.path.getctime)

        images = glob.glob(f"{latest_folder}/*.jpg")

        detection_image = images[0]

        st.image(detection_image, caption="Detection Result")

        # --------------------------
        # UPLOAD IMAGE TO S3
        # --------------------------

        s3 = boto3.client("s3")

        s3_key = f"detections/{os.path.basename(detection_image)}"

        s3.upload_file(detection_image, S3_BUCKET, s3_key)

        image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

        st.success("Image uploaded to S3")

        # --------------------------
        # STORE METADATA IN RDS
        # --------------------------

        connection = pymysql.connect(
            host=RDS_HOST,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DB,
            port=3306
        )

        cursor = connection.cursor()

        cursor.execute("""
        INSERT INTO detections (drone_type, confidence, image_url)
        VALUES (%s, %s, %s)
        """, (detected_class, confidence, image_url))

        connection.commit()

        connection.close()

        st.success("Detection stored in AWS RDS")