import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import boto3
import pymysql
import re
import os
import json
import requests
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import joblib
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------

S3_BUCKET = "drone-detection-storage-boopathy"

DB_HOST = "drone-detection-db.czyge4aiywpv.ap-south-1.rds.amazonaws.com"
DB_USER = "admin"
DB_PASS = "Drone1234"
DB_NAME = "drone_db"

SMTP_EMAIL = "bptsdg@gmail.com"
SMTP_PASSWORD = "gbhv jnhu bdyq nmhr"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "llama3:latest"

MODEL_PATH = r"C:\Users\boopa\Desktop\DS\Drone\random_forest_model.pkl"
COLUMNS_PATH = r"C:\Users\boopa\Desktop\DS\Drone\model_columns.pkl"

# ✅ Your trained YOLO model path
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"

# ✅ Confidence threshold to avoid false detections
CONFIDENCE_THRESHOLD = 0.5

# ✅ Keywords that indicate user is asking about detection data
DATA_KEYWORDS = [
    "detect", "drone", "last", "recent", "found", "confidence",
    "when", "how many", "show", "list", "latest", "history",
    "today", "yesterday", "database", "stored", "record"
]

# -----------------------------
# AWS CLIENTS
# -----------------------------

s3 = boto3.client("s3")

# -----------------------------
# LOAD DELIVERY MODEL
# -----------------------------

@st.cache_resource
def load_delivery_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns

# -----------------------------
# DB CONNECTION
# -----------------------------

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=3306
    )

# =============================================
# AGENT 1 — DATA QUERY AGENT
# =============================================

def ollama_call(prompt, timeout=60):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"Ollama model error: {data['error']}")
        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response: {data}")
        return data["response"].strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Please open a terminal and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama timed out. The model may still be loading — try again shortly.")
    except Exception as e:
        raise RuntimeError(str(e))


def data_query_agent(user_query):
    # ✅ FIX: Strict SQL prompt — prevents hallucination, forces real DB query only
    sql_prompt = f"""
You are a SQL expert connected to a real MySQL database.
The database has a table called `detections` with these exact columns:
- id (int)
- drone_type (varchar)
- confidence (float)
- image_url (varchar)
- detected_at (datetime)

STRICT RULES:
- Return ONLY a valid MySQL SELECT query. Nothing else.
- Do NOT fabricate, assume, or invent any data.
- Do NOT include any explanation, markdown, or backticks.
- Do NOT use columns that do not exist in the table above.
- If the question cannot be answered using this table, return exactly: SELECT 'NO_DATA' AS message;
- Always query from the `detections` table only.

User question: {user_query}
"""
    try:
        sql_query = ollama_call(sql_prompt)

        # Strip any accidental markdown backticks
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        if not sql_query.upper().startswith("SELECT"):
            return None, None, "Generated query was not a SELECT statement. Aborting for safety."

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        connection.close()
        return rows, columns, sql_query
    except Exception as e:
        return None, None, f"Data query error: {e}"


# =============================================
# AGENT 2 — REPORT GENERATION AGENT
# =============================================

def report_generation_agent(data, columns, report_title="Drone Detection Report"):
    try:
        pdf_path = "drone_report.pdf"
        styles = getSampleStyleSheet()
        elements = []
        elements.append(Paragraph(report_title, styles["Title"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(
            Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"]
            )
        )
        elements.append(Spacer(1, 0.3 * inch))
        if not data:
            elements.append(Paragraph("No data found.", styles["Normal"]))
        else:
            header = " | ".join(columns)
            elements.append(Paragraph(f"<b>{header}</b>", styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
            for row in data:
                row_text = " | ".join(str(val) for val in row)
                elements.append(Paragraph(row_text, styles["Normal"]))
                elements.append(Spacer(1, 0.05 * inch))
        doc = SimpleDocTemplate(pdf_path)
        doc.build(elements)
        s3_key = f"reports/drone_report_{datetime.now().timestamp()}.pdf"
        s3.upload_file(pdf_path, S3_BUCKET, s3_key)
        s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        return pdf_path, s3_url
    except Exception as e:
        return None, f"Report generation error: {e}"


# =============================================
# AGENT 3 — EMAIL AGENT
# =============================================

def email_agent(recipient_email, pdf_path, s3_url):
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = "Drone Detection Report"
        body = f"""Hello,

Please find attached the latest Drone Detection Report.

You can also access the report online at:
{s3_url}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Regards,
AI Drone Surveillance Platform
"""
        msg.attach(MIMEText(body, "plain"))
        with open(pdf_path, "rb") as f:
            attach = MIMEApplication(f.read())
        attach.add_header("Content-Disposition", "attachment", filename="drone_report.pdf")
        msg.attach(attach)
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True, f"Email successfully sent to {recipient_email}"
    except Exception as e:
        return False, f"Email error: {e}"


# =============================================
# LLAMA ROUTER
# =============================================

def llama_router(user_query):
    router_prompt = f"""
You are an AI assistant for a drone surveillance system.
Based on the user's message, decide which action to take.

Available actions:
1. "data_query"   — user wants to fetch/view detection data from the database
2. "report"       — user wants to generate a PDF report
3. "email"        — user wants to send a report via email (may include an email address)
4. "report_email" — user wants to generate a report AND email it
5. "general"      — general conversation or unclear intent

Respond ONLY in valid JSON format like this:
{{
  "action": "data_query",
  "email": null,
  "report_title": null
}}

Rules:
- Set "email" to the email address found in the message, or null if none
- Set "report_title" to a suitable title if the user mentions a specific scope, or null for default
- Do not include any explanation outside the JSON
- If the user asks about detections, drones detected, last drone, recent records — always use "data_query"

User message: {user_query}
"""
    try:
        raw = ollama_call(router_prompt)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())
        else:
            decision = {"action": "general", "email": None, "report_title": None}
        return decision
    except Exception as e:
        return {"action": "general", "email": None, "report_title": None, "error": str(e)}


# =============================================
# GENERAL LLAMA RESPONSE
# =============================================

def llama_general_response(user_query):
    try:
        return ollama_call(
            f"You are an AI assistant for a drone surveillance platform. Answer helpfully.\n\nUser: {user_query}"
        )
    except Exception as e:
        return f"⚠️ {e}"


# =============================================
# ✅ LOAD YOUR TRAINED YOLO MODEL
# =============================================

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        st.error(
            f"❌ Trained model not found at: `{YOLO_MODEL_PATH}`\n\n"
            "Make sure you have trained the model and the file exists."
        )
        st.stop()
    return YOLO(YOLO_MODEL_PATH)

model = load_yolo_model()

# =============================================
# STREAMLIT UI
# =============================================

st.set_page_config(page_title="AI Drone Surveillance Platform", layout="wide")
st.title("AI Drone Surveillance Platform")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Drone Detection", "Delivery Time Predictor", "AI Chatbot"]
)

# =============================
# DRONE DETECTION PAGE
# =============================

if menu == "Drone Detection":

    st.header("Drone Detection")
    uploaded_file = st.file_uploader("Upload Drone Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        if st.button("Detect Drone"):
            file_bytes = uploaded_file.read()
            np_arr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(np_arr, (640, 640))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image)

            # ✅ Using trained model with confidence threshold
            results = model(temp_path, conf=CONFIDENCE_THRESHOLD)
            result = results[0]
            annotated = result.plot()

            st.image(annotated, caption="Detection Result", channels="BGR")

            drone_type = "unknown"
            confidence = 0.0

            if len(result.boxes) > 0:
                drone_type = model.names[int(result.boxes.cls[0])]
                confidence = float(result.boxes.conf[0])
                st.success(f"✅ Detected: **{drone_type}** with **{confidence:.2%}** confidence")
            else:
                st.warning(f"⚠️ No drone detected above {CONFIDENCE_THRESHOLD*100:.0f}% confidence threshold.")

            output_path = temp_path.replace(".jpg", "_detected.jpg")
            cv2.imwrite(output_path, annotated)

            s3_key = f"detections/{datetime.now().timestamp()}.jpg"
            s3.upload_file(output_path, S3_BUCKET, s3_key)
            image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO detections (drone_type, confidence, image_url, detected_at)
                VALUES (%s, %s, %s, NOW())
                """,
                (drone_type, confidence, image_url)
            )
            connection.commit()
            connection.close()

            st.success("Detection stored successfully!")

# =============================
# DELIVERY TIME PREDICTOR PAGE
# =============================

elif menu == "Delivery Time Predictor":

    st.header("🚁 Drone Delivery Time Predictor")
    st.markdown("Fill in the delivery details below to estimate how long the delivery will take.")

    delivery_model, model_columns = load_delivery_model()

    DRONE_TYPES        = ["Fixed-Wing", "Hybrid VTOL", "Multi-Rotor", "Single-Rotor"]
    CLIMATE_CONDITIONS = ["Clear", "Cloudy", "Windy"]
    SOURCE_AREAS       = ["Adyar", "Guindy", "Nungambakkam", "OMR", "T Nagar", "Tambaram", "Vadapalani"]
    DESTINATION_AREAS  = ["Anna Nagar", "Guindy", "Mylapore", "Nungambakkam", "T Nagar", "Vadapalani", "Velachery"]
    TRAFFIC_CONDITIONS = ["High", "Low", "Medium"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚁 Drone Details")
        drone_type = st.selectbox("Drone Type", options=DRONE_TYPES)
        drone_speed = st.selectbox(
            "Drone Speed (km/h)",
            options=[round(v * 0.5, 1) for v in range(43, 112)],
            index=33,
        )
        battery_efficiency = st.selectbox(
            "Battery Efficiency (%)",
            options=list(range(70, 99)),
            index=10,
        )

        st.subheader("📍 Location Details")
        source_area = st.selectbox("Source Area", options=SOURCE_AREAS)
        destination_area = st.selectbox("Destination Area", options=DESTINATION_AREAS)
        if source_area == destination_area:
            st.warning("⚠️ Source and destination are the same.")

    with col2:
        st.subheader("📦 Package & Route")
        payload_weight = st.selectbox(
            "Payload Weight (kg)",
            options=[round(v * 0.1, 1) for v in range(18, 98)],
            index=12,
        )
        distance_km = st.selectbox(
            "Distance (km)",
            options=[round(v * 0.1, 1) for v in range(10, 50)],
            index=20,
        )
        traffic_condition = st.selectbox("Traffic Condition", options=TRAFFIC_CONDITIONS, index=2)

        st.subheader("🌤️ Weather Conditions")
        climate_condition = st.selectbox("Climate Condition", options=CLIMATE_CONDITIONS)
        wind_speed = st.selectbox(
            "Wind Speed (km/h)",
            options=[round(v * 0.5, 1) for v in range(11, 48)],
            index=10,
        )
        temperature = st.selectbox(
            "Temperature (°C)",
            options=[round(v * 0.1, 1) for v in range(249, 373)],
            index=30,
        )
        humidity = st.selectbox(
            "Humidity (%)",
            options=[round(v * 0.1, 1) for v in range(489, 744)],
            index=20,
        )

    st.markdown("---")

    if st.button("⏱️ Predict Delivery Time", use_container_width=True):
        try:
            input_data = {col: 0 for col in model_columns}

            numeric_map = {
                "drone_speed_kmph":     drone_speed,
                "payload_weight_kg":    payload_weight,
                "distance_km":          distance_km,
                "battery_efficiency":   battery_efficiency,
                "wind_speed_kmph":      wind_speed,
                "temperature_c":        temperature,
                "humidity_percent":     humidity,
            }
            for col, val in numeric_map.items():
                if col in input_data:
                    input_data[col] = val

            for prefix, value in [
                ("drone_type",        drone_type),
                ("climate_condition", climate_condition),
                ("source_area",       source_area),
                ("destination_area",  destination_area),
                ("traffic_condition", traffic_condition),
            ]:
                col_name = f"{prefix}_{value}"
                if col_name in input_data:
                    input_data[col_name] = 1

            input_df = pd.DataFrame([input_data])
            predicted_time = delivery_model.predict(input_df)[0]

            st.success("✅ Prediction Complete!")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("📍 From", source_area)
            with r2:
                st.metric("📍 To", destination_area)
            with r3:
                st.metric("📏 Distance", f"{distance_km} km")
            with r4:
                st.metric("⏱️ Estimated Time", f"{predicted_time:.1f} min")

            st.info(
                f"**Summary:** A **{drone_type}** drone carrying **{payload_weight} kg** "
                f"over **{distance_km} km** from **{source_area}** to **{destination_area}** "
                f"under **{climate_condition}** skies with **{traffic_condition}** traffic "
                f"is estimated to arrive in **{predicted_time:.1f} minutes**."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.warning(
                "Tip: Make sure `random_forest_model.pkl` and `model_columns.pkl` "
                "are in `C:\\Users\\boopa\\Desktop\\DS\\Drone\\` and were trained "
                "with the same feature names as the training data."
            )

# =============================
# AI CHATBOT PAGE
# =============================

elif menu == "AI Chatbot":

    st.header("Drone Intelligence Chatbot")

    # --- Ollama status check ---
    try:
        ping = requests.get(OLLAMA_TAGS_URL, timeout=3)
        detected_model = OLLAMA_MODEL
        st.success(f"✅ Ollama connected — using model: `{detected_model}`")
    except Exception:
        st.error(
            "❌ Ollama is not running. Open a terminal and run: `ollama serve`  \n"
            "Then refresh this page."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me anything about drone detections...")

    if user_input:

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                decision = llama_router(user_input)
                action = decision.get("action", "general")
                email_address = decision.get("email", None)
                report_title = decision.get("report_title") or "Drone Detection Report"
                response_text = ""

                if action == "data_query":
                    result = data_query_agent(user_input)
                    if len(result) == 3:
                        rows, columns, sql_query = result
                    else:
                        rows, columns, sql_query = None, None, result[1]

                    if rows is None:
                        response_text = f"⚠️ Could not fetch data: {sql_query}"
                    elif len(rows) == 0:
                        # ✅ FIX: Real empty result from DB — don't hallucinate
                        response_text = "✅ Query ran successfully but **no records found** in the database for your request."
                    else:
                        # ✅ FIX: Check if Ollama returned NO_DATA signal
                        if rows[0][0] == "NO_DATA":
                            response_text = (
                                "⚠️ I can only answer questions about data stored in your database.\n\n"
                                "Your database has these fields: **id, drone_type, confidence, image_url, detected_at**\n\n"
                                "Try asking: *'Show me the last 5 detections'* or *'How many drones were detected today?'*"
                            )
                        else:
                            st.caption(f"🗄️ SQL executed: `{sql_query}`")
                            header = " | ".join(columns)
                            table = f"**{header}**\n\n"
                            for row in rows:
                                table += " | ".join(str(v) for v in row) + "\n\n"
                            response_text = table

                elif action == "report":
                    rows, columns, sql_or_err = data_query_agent("get all detections from today")
                    if rows is None:
                        response_text = f"Could not fetch data: {sql_or_err}"
                    else:
                        pdf_path, s3_url = report_generation_agent(rows, columns, report_title)
                        if pdf_path:
                            response_text = f"Report generated and uploaded to S3.\n\n**S3 URL:** {s3_url}"
                            with open(pdf_path, "rb") as f:
                                st.download_button("Download Report", data=f, file_name="drone_report.pdf", mime="application/pdf")
                        else:
                            response_text = f"Report error: {s3_url}"

                elif action == "email":
                    if not email_address:
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
                        email_address = email_match.group() if email_match else None
                    if not email_address:
                        response_text = "Please provide a valid email address in your message."
                    else:
                        rows, columns, sql_or_err = data_query_agent("get all detections from today")
                        if rows is None:
                            response_text = f"Could not fetch data: {sql_or_err}"
                        else:
                            pdf_path, s3_url = report_generation_agent(rows, columns, report_title)
                            if pdf_path:
                                success, msg = email_agent(email_address, pdf_path, s3_url)
                                response_text = msg
                            else:
                                response_text = f"Report error: {s3_url}"

                elif action == "report_email":
                    if not email_address:
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
                        email_address = email_match.group() if email_match else None
                    rows, columns, sql_or_err = data_query_agent("get all detections from today")
                    if rows is None:
                        response_text = f"Could not fetch data: {sql_or_err}"
                    else:
                        pdf_path, s3_url = report_generation_agent(rows, columns, report_title)
                        if pdf_path:
                            with open(pdf_path, "rb") as f:
                                st.download_button("Download Report", data=f, file_name="drone_report.pdf", mime="application/pdf")
                            if email_address:
                                success, msg = email_agent(email_address, pdf_path, s3_url)
                                response_text = f"Report generated.\n\n{msg}\n\n**S3 URL:** {s3_url}"
                            else:
                                response_text = f"Report generated and uploaded.\n\n**S3 URL:** {s3_url}\n\nNo email address found in your message."
                        else:
                            response_text = f"Report error: {s3_url}"

                else:
                    # ✅ FIX: Block hallucination for data-related questions
                    if any(word in user_input.lower() for word in DATA_KEYWORDS):
                        response_text = (
                            "⚠️ I can only answer that using your **real database records**.\n\n"
                            "I will not guess or make up drone data.\n\n"
                            "Try rephrasing your question, for example:\n"
                            "- *'Show me the last 5 detections'*\n"
                            "- *'How many drones were detected today?'*\n"
                            "- *'What drone types have been detected?'*"
                        )
                    else:
                        response_text = llama_general_response(user_input)

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})