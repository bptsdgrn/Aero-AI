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
OLLAMA_MODEL = "llama3"  # change to your installed model e.g. llama2, mistral

# -----------------------------
# AWS CLIENTS
# -----------------------------

s3 = boto3.client("s3")

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

def data_query_agent(user_query):
    """
    Uses Llama to convert natural language into SQL,
    runs it against RDS, and returns formatted results.
    """

    sql_prompt = f"""
You are a SQL expert. The database has a table called `detections` with these columns:
- id (int)
- drone_type (varchar)
- confidence (float)
- image_url (varchar)
- detected_at (datetime)

Convert the following user question into a valid MySQL SELECT query only.
Do not include any explanation. Return only the SQL query, nothing else.

User question: {user_query}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": sql_prompt,
                "stream": False
            },
            timeout=30
        )

        sql_query = response.json()["response"].strip()

        # Safety check — only allow SELECT
        if not sql_query.upper().startswith("SELECT"):
            return None, "Generated query was not a SELECT statement. Aborting for safety."

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
    """
    Generates a PDF report from query results,
    uploads it to S3, and returns the local path + S3 URL.
    """

    try:
        pdf_path = "drone_report.pdf"
        styles = getSampleStyleSheet()
        elements = []

        # Title
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
            # Column headers
            header = " | ".join(columns)
            elements.append(Paragraph(f"<b>{header}</b>", styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))

            # Rows
            for row in data:
                row_text = " | ".join(str(val) for val in row)
                elements.append(Paragraph(row_text, styles["Normal"]))
                elements.append(Spacer(1, 0.05 * inch))

        doc = SimpleDocTemplate(pdf_path)
        doc.build(elements)

        # Upload to S3
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
    """
    Drafts and sends an email with the PDF report attached.
    """

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

        attach.add_header(
            "Content-Disposition",
            "attachment",
            filename="drone_report.pdf"
        )
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
# LLAMA ROUTER — decides which agent to call
# =============================================

def llama_router(user_query):
    """
    Sends the user query to Llama and asks it to classify
    the intent and extract any parameters like email address.
    Returns a structured JSON decision.
    """

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
- Set "report_title" to a suitable title if the user mentions a specific scope (e.g. "today's report"), or null for default
- Do not include any explanation outside the JSON

User message: {user_query}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": router_prompt,
                "stream": False
            },
            timeout=30
        )

        raw = response.json()["response"].strip()

        # Extract JSON safely
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
    """
    Falls back to a general Llama response for
    conversation or questions outside agent scope.
    """

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"You are an AI assistant for a drone surveillance platform. Answer helpfully.\n\nUser: {user_query}",
                "stream": False
            },
            timeout=30
        )
        return response.json()["response"].strip()

    except Exception as e:
        return f"Llama error: {e}"


# =============================================
# YOLO MODEL
# =============================================

model = YOLO("yolov8n.pt")

# =============================================
# STREAMLIT UI
# =============================================

st.title("AI Drone Surveillance Platform")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Drone Detection", "AI Chatbot"]
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

            np_arr = cv2.imdecode(
                np.frombuffer(file_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            image = cv2.resize(np_arr, (640, 640))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image)

            results = model(temp_path)
            result = results[0]
            annotated = result.plot()

            st.image(annotated, caption="Detection Result")

            drone_type = "drone"
            confidence = 0.0

            if len(result.boxes) > 0:
                drone_type = model.names[int(result.boxes.cls[0])]
                confidence = float(result.boxes.conf[0])

            output_path = temp_path.replace(".jpg", "_detected.jpg")
            cv2.imwrite(output_path, annotated)

            s3_key = f"detections/{datetime.now().timestamp()}.jpg"
            s3.upload_file(output_path, S3_BUCKET, s3_key)
            image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO detections
                (drone_type, confidence, image_url, detected_at)
                VALUES (%s, %s, %s, NOW())
                """,
                (drone_type, confidence, image_url)
            )
            connection.commit()
            connection.close()

            st.success("Detection stored successfully!")

# =============================
# AI CHATBOT PAGE
# =============================

if menu == "AI Chatbot":

    st.header("Drone Intelligence Chatbot")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything about drone detections...")

    if user_input:

        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                # Step 1 — Route the query via Llama
                decision = llama_router(user_input)
                action = decision.get("action", "general")
                email_address = decision.get("email", None)
                report_title = decision.get("report_title") or "Drone Detection Report"

                response_text = ""

                # ---------------------------------
                # ACTION: DATA QUERY
                # ---------------------------------

                if action == "data_query":

                    result = data_query_agent(user_input)

                    if len(result) == 3:
                        rows, columns, sql_query = result
                    else:
                        rows, columns, sql_query = None, None, result[1]

                    if rows is None:
                        response_text = f"Could not fetch data: {sql_query}"

                    elif len(rows) == 0:
                        response_text = "No records found for your query."

                    else:
                        st.caption(f"SQL: `{sql_query}`")
                        header = " | ".join(columns)
                        table = f"**{header}**\n\n"
                        for row in rows:
                            table += " | ".join(str(v) for v in row) + "\n\n"
                        response_text = table

                # ---------------------------------
                # ACTION: REPORT ONLY
                # ---------------------------------

                elif action == "report":

                    rows, columns, sql_or_err = data_query_agent(
                        "get all detections from today"
                    )

                    if rows is None:
                        response_text = f"Could not fetch data: {sql_or_err}"

                    else:
                        pdf_path, s3_url = report_generation_agent(rows, columns, report_title)

                        if pdf_path:
                            response_text = f"Report generated and uploaded to S3.\n\n**S3 URL:** {s3_url}"
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "Download Report",
                                    data=f,
                                    file_name="drone_report.pdf",
                                    mime="application/pdf"
                                )
                        else:
                            response_text = f"Report error: {s3_url}"

                # ---------------------------------
                # ACTION: EMAIL ONLY
                # ---------------------------------

                elif action == "email":

                    if not email_address:
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
                        email_address = email_match.group() if email_match else None

                    if not email_address:
                        response_text = "Please provide a valid email address in your message."

                    else:
                        rows, columns, sql_or_err = data_query_agent(
                            "get all detections from today"
                        )

                        if rows is None:
                            response_text = f"Could not fetch data: {sql_or_err}"

                        else:
                            pdf_path, s3_url = report_generation_agent(rows, columns, report_title)

                            if pdf_path:
                                success, msg = email_agent(email_address, pdf_path, s3_url)
                                response_text = msg
                            else:
                                response_text = f"Report error: {s3_url}"

                # ---------------------------------
                # ACTION: REPORT + EMAIL TOGETHER
                # ---------------------------------

                elif action == "report_email":

                    if not email_address:
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
                        email_address = email_match.group() if email_match else None

                    rows, columns, sql_or_err = data_query_agent(
                        "get all detections from today"
                    )

                    if rows is None:
                        response_text = f"Could not fetch data: {sql_or_err}"

                    else:
                        pdf_path, s3_url = report_generation_agent(rows, columns, report_title)

                        if pdf_path:
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "Download Report",
                                    data=f,
                                    file_name="drone_report.pdf",
                                    mime="application/pdf"
                                )

                            if email_address:
                                success, msg = email_agent(email_address, pdf_path, s3_url)
                                response_text = f"Report generated.\n\n{msg}\n\n**S3 URL:** {s3_url}"
                            else:
                                response_text = f"Report generated and uploaded.\n\n**S3 URL:** {s3_url}\n\nNo email address found in your message."
                        else:
                            response_text = f"Report error: {s3_url}"

                # ---------------------------------
                # ACTION: GENERAL CONVERSATION
                # ---------------------------------

                else:
                    response_text = llama_general_response(user_input)

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
