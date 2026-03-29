# AI Drone Surveillance Platform

A Streamlit-based web application that combines YOLOv8 drone detection, AWS cloud storage, MySQL logging, and a multi-agent LLM chatbot powered by Llama (via Ollama).

---

## Features

- **Drone Detection** — Upload images and run YOLOv8 inference to detect and classify drones
- **AWS S3 Integration** — Detected images and generated reports are automatically uploaded to S3
- **MySQL / RDS Logging** — Every detection is stored in an Amazon RDS MySQL database
- **Multi-Agent AI Chatbot** — A Llama-powered chatbot that routes queries to specialized agents:
  - **Data Query Agent** — Converts natural language to SQL and queries the detections database
  - **Report Generation Agent** — Generates PDF reports from query results and uploads them to S3
  - **Email Agent** — Sends PDF reports via Gmail SMTP to a specified recipient
  - **General Agent** — Handles open-ended conversation outside the above scopes

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Object Detection | YOLOv8 (Ultralytics) |
| LLM | Llama 3 via Ollama |
| Cloud Storage | AWS S3 |
| Database | Amazon RDS (MySQL / PyMySQL) |
| PDF Generation | ReportLab |
| Email | Gmail SMTP |
| Image Processing | OpenCV, NumPy |

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally with `llama3` pulled
- AWS account with an S3 bucket and RDS MySQL instance
- Gmail account with an App Password for SMTP

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-repo/drone-surveillance.git
cd drone-surveillance
```

2. **Install Python dependencies**

```bash
pip install streamlit ultralytics opencv-python numpy boto3 pymysql reportlab requests
```

3. **Pull the Llama model via Ollama**

```bash
ollama pull llama3
```

4. **Configure AWS credentials**

```bash
aws configure
```

Provide your `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and default region (`ap-south-1`).

---

## Configuration

Open `app.py` and update the following constants at the top of the file:

```python
S3_BUCKET     = "your-s3-bucket-name"

DB_HOST       = "your-rds-endpoint.rds.amazonaws.com"
DB_USER       = "your-db-username"
DB_PASS       = "your-db-password"
DB_NAME       = "drone_db"

SMTP_EMAIL    = "your-gmail@gmail.com"
SMTP_PASSWORD = "your-gmail-app-password"

OLLAMA_MODEL  = "llama3"   # or mistral, llama2, etc.
```

> **Security Note:** Do not hardcode credentials in production. Use environment variables or AWS Secrets Manager instead.

---

## Database Setup

Run the following SQL on your RDS instance to create the required table:

```sql
CREATE DATABASE IF NOT EXISTS drone_db;

USE drone_db;

CREATE TABLE detections (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    drone_type   VARCHAR(100),
    confidence   FLOAT,
    image_url    TEXT,
    detected_at  DATETIME
);
```

---

## Running the App

Start Ollama in one terminal:

```bash
ollama serve
```

Start the Streamlit app in another terminal:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Usage

### Drone Detection

1. Navigate to **Drone Detection** in the sidebar.
2. Upload a `.jpg`, `.jpeg`, or `.png` image.
3. Click **Detect Drone**.
4. The annotated result is displayed, and the detection is saved to S3 and RDS automatically.

### AI Chatbot

Navigate to **AI Chatbot** in the sidebar and type any of the following:

| Example Query | Action Triggered |
|---|---|
| `Show me all detections from today` | Data Query Agent |
| `Generate a report for this week` | Report Generation Agent |
| `Send the report to someone@example.com` | Email Agent |
| `Generate and email the report to me@example.com` | Report + Email Agent |
| `What is the most common drone type detected?` | Data Query Agent |
| `What can this platform do?` | General Agent |

---

## Project Structure

```
drone-surveillance/
├── app.py               # Main Streamlit application
├── yolov8n.pt           # YOLOv8 nano weights (auto-downloaded by Ultralytics)
├── drone_report.pdf     # Temporary PDF output (overwritten on each report run)
└── README.md
```

---

## Agent Architecture

```
User Message
     │
     ▼
Llama Router  ──► classifies intent into one of:
     │
     ├── data_query    ──► SQL generation ──► RDS query ──► formatted table
     ├── report        ──► RDS query ──► PDF generation ──► S3 upload
     ├── email         ──► RDS query ──► PDF ──► Gmail SMTP
     ├── report_email  ──► RDS query ──► PDF ──► S3 + Gmail
     └── general       ──► free-form Llama response
```

---

## Known Limitations

- The YOLOv8 nano model (`yolov8n.pt`) detects COCO classes; it is not fine-tuned specifically for drones. For production use, replace it with a drone-specific fine-tuned model.
- Ollama must be running locally on port `11434`. Remote Ollama instances require updating `OLLAMA_URL`.
- PDF reports use a plain text layout. Tables and charts can be added via ReportLab's `platypus` module.
- Gmail App Passwords require 2-Factor Authentication to be enabled on your Google account.

---

## License

MIT License. See `LICENSE` for details.