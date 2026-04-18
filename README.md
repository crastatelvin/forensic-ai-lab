<div align="center">

# 🔬 Forensic AI Lab

### An AI-Powered Multi-Module Forensic Investigation Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Mistral-FF6600?style=for-the-badge)](https://ollama.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Forensic AI Lab** is a full-stack web application that brings cutting-edge AI to criminal investigation workflows. It combines computer vision, audio analysis, NLP, and an on-device LLM chatbot (Ollama/Mistral) to power 32 distinct forensic analysis modules — all accessible through a clean, role-based web interface.

<br/>

![Forensic AI Lab Banner](https://img.shields.io/badge/32%20AI%20Modules-Active-brightgreen?style=for-the-badge) ![Role Based Access](https://img.shields.io/badge/Role%20Based%20Access-Admin%20%7C%20Officer%20%7C%20Expert-blue?style=for-the-badge) ![PDF Reports](https://img.shields.io/badge/PDF%20Reports-Auto%20Generated-orange?style=for-the-badge)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Analysis Modules](#-analysis-modules-32-total)
- [Role-Based Access](#-role-based-access)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [AI Chatbot](#-ai-forensic-chatbot)
- [Reports](#-pdf-report-generation)
- [Database](#-database-models)
- [Contributing](#-contributing)

---

## 🧠 Overview

Forensic AI Lab simulates an end-to-end digital forensics lab where investigators, officers, and domain experts can:

- Upload crime scene and suspect evidence (images, audio, video, text)
- Run **AI-driven comparison analysis** across 32 forensic disciplines
- Receive quantitative **match scores** and match/no-match verdicts
- Auto-generate **downloadable PDF reports** for each case
- Query an **on-device AI chatbot** (powered by Ollama + Mistral) for investigative guidance and live case analytics
- Manage cases through a fully **role-based access control system**

The system is built on Flask, backed by SQLite via SQLAlchemy, and integrates libraries like OpenCV, librosa, face_recognition, PyTorch, and Hugging Face Transformers.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **32 AI Forensic Modules** | From fingerprints to deepfakes, DNA to dark web analysis |
| 👥 **Role-Based Dashboards** | Separate portals for Admins, Officers, and Forensic Experts |
| 🤖 **AI Chatbot (Ollama/Mistral)** | On-device LLM for forensic Q&A and case data analysis |
| 📄 **Auto PDF Reports** | ReportLab-generated reports with scores, images, and findings |
| 📦 **Bulk Report Download** | Officers can download all case reports as a ZIP archive |
| 📊 **CSV Export** | Experts can export their analysis results as CSV |
| 🗃️ **Case Management** | Create, assign, track, and close forensic cases |
| 📈 **Live Dashboard Charts** | Real-time case status and analysis-type breakdown charts |
| 🔐 **Session-Based Auth** | Secure login with role enforcement on every route |
| 🧹 **Admin Controls** | Upload/report folder management, user CRUD, case assignment |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser / Client                  │
└────────────────────────┬────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────┐
│              Flask Application (forensic_web.py)     │
│                                                      │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────┐  │
│  │  Auth Routes │  │ Analysis      │  │ Chat API │  │
│  │  (login,     │  │ Routes (32    │  │ /chat    │  │
│  │   logout)    │  │  modules)     │  │ endpoint │  │
│  └──────────────┘  └───────┬───────┘  └────┬─────┘  │
│                            │               │         │
│  ┌─────────────────────────▼───────────────▼──────┐  │
│  │              ai_tools.py  (32 Analysis Fns)    │  │
│  │   OpenCV · librosa · face_recognition · YOLO   │  │
│  │   PyTorch · Transformers · geopy · NumPy       │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  ┌───────────────┐  ┌───────────────────────────┐   │
│  │  models.py    │  │  Ollama CLI (Mistral LLM)  │   │
│  │  User / Case  │  │  ask_ollama() subprocess   │   │
│  │  AnalysisResult│ └───────────────────────────┘   │
│  └───────┬───────┘                                  │
│          │ SQLAlchemy ORM                            │
│  ┌───────▼───────┐  ┌──────────────┐                │
│  │   SQLite DB   │  │  ReportLab   │                │
│  │  (instance/)  │  │  (PDF gen)   │                │
│  └───────────────┘  └──────────────┘                │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
| Library | Purpose |
|---|---|
| **Flask** | Web framework & routing |
| **Flask-SQLAlchemy** | ORM and database management |
| **SQLite** | Lightweight relational database |
| **ReportLab** | PDF report generation |
| **Ollama (Mistral)** | On-device LLM via subprocess CLI |

### AI / ML
| Library | Purpose |
|---|---|
| **OpenCV** | Image processing, edge detection, contour analysis |
| **face_recognition** | Facial encoding and comparison |
| **librosa** | Audio feature extraction (MFCC, spectral) |
| **Ultralytics YOLO** | Object detection |
| **PyTorch + torchvision** | Deep learning inference (ResNet18) |
| **Hugging Face Transformers** | Deepfake detection pipeline |
| **NumPy** | Numerical operations and similarity math |
| **geopy** | Geodesic distance computation (geospatial module) |
| **Matplotlib** | Feature visualization plots |

### Frontend
| Technology | Purpose |
|---|---|
| **Jinja2 Templates** | Server-side HTML rendering |
| **HTML / CSS / JS** | UI (57.7% HTML, 4.2% CSS, 1.3% JS) |
| **Chart.js** | Dashboard data visualizations |

---

## 🔬 Analysis Modules (32 Total)

Each module accepts crime scene evidence and suspect evidence, extracts features, computes a **match score (0.0–1.0)**, and returns a **Match / No Match verdict**. A PDF report is auto-generated for every completed analysis.

### 🖼️ Image-Based Modules
| # | Module | Technique |
|---|---|---|
| 1 | **Fingerprint Analysis** | Minutiae extraction via Canny edge detection + contour centroid matching |
| 2 | **Face Analysis** | Face encodings via `face_recognition` + Euclidean distance comparison |
| 3 | **Bloodstain Pattern** | Contour area + angle comparison using OpenCV thresholding |
| 4 | **Ballistics** | Edge + contour feature matching on bullet/casing images |
| 5 | **Handwriting** | Stroke contour width and angle extraction + comparison |
| 6 | **Tire Track** | Tread contour depth and angle feature matching |
| 7 | **Tool Mark** | Mark width and angular contour matching |
| 8 | **Shoe Print** | Pattern width and angular feature comparison |
| 9 | **Explosive Residue** | Residue density and particle size feature matching |
| 10 | **Glass Fracture** | Fracture width and refractive index comparison |
| 11 | **Bite Mark** | Tooth width and angular feature extraction |
| 12 | **Pollen** | Grain size and density comparison |
| 13 | **Paint** | Color intensity and hue analysis |
| 14 | **Soil** | Particle composition feature matching |
| 15 | **Hair** | Morphological feature extraction and comparison |
| 16 | **Insect (Forensic Entomology)** | Insect species/stage pattern analysis |
| 17 | **Arson** | Burn pattern image feature comparison |
| 18 | **Iris Recognition** | Iris pattern feature extraction and matching |
| 19 | **Fingerprint Dust** | Enhanced dust-lifted fingerprint comparison |

### 🔊 Audio-Based Modules
| # | Module | Technique |
|---|---|---|
| 20 | **Gunshot Analysis** | MFCC extraction via librosa + cosine similarity |
| 21 | **Voiceprint Analysis** | Pitch and frequency comparison from WAV features |
| 22 | **Lie Detection** | Audio stress analysis for deception indicators |

### 🎥 Video-Based Modules
| # | Module | Technique |
|---|---|---|
| 23 | **Deepfake Detection** | Hugging Face image-classification pipeline |
| 24 | **Gait Analysis** | Stride length and angular pattern extraction from video |

### 📝 Text/Data-Based Modules
| # | Module | Technique |
|---|---|---|
| 25 | **Digital Footprint** | Activity count and timestamp pattern matching |
| 26 | **Odor Profile** | Chemical compound intensity comparison |
| 27 | **Phishing Detection** | Suspicious link/text pattern analysis |
| 28 | **Dark Web Analysis** | Digital artifact pattern comparison |

### 🧬 Advanced Scientific Modules
| # | Module | Technique |
|---|---|---|
| 29 | **DNA Analysis** | Genetic marker comparison |
| 30 | **Toxicology** | Substance compound profile matching |
| 31 | **Geospatial Analysis** | Geodesic distance computation via geopy |
| 32 | **Ballistic Trajectory** | Vector and angle-based trajectory comparison |

---

## 👥 Role-Based Access

The platform implements a three-tier role system enforced server-side on every route.

### 🛡️ Admin
- Full user management (Create / Edit / Delete users)
- Assign cases to officers or experts
- View all cases and analysis results system-wide
- Clear upload and report folders
- Access recent activity feed

### 👮 Officer
- View and manage personally assigned cases
- Submit evidence for any of the 32 analysis modules
- Download all completed case reports as a ZIP file
- View dashboard charts: case status breakdown, analysis type distribution

### 🔬 Expert
- View cases assigned specifically to them
- Run and review specialized analyses
- Export all personal analysis results as CSV
- View performance charts: score trends over time, module usage breakdown

### Default Credentials (Development)
```
Admin:   username=admin    / password=admin123
Officer: username=officer  / password=officer123
Expert:  username=expert   / password=expert123
```
> ⚠️ **Change these immediately in any non-development environment.**

---

## 📁 Project Structure

```
forensic-ai-lab/
│
├── forensic_web.py       # Main Flask application — all routes, session logic, PDF generation
├── ai_tools.py           # 32 AI analysis functions (1,667 lines of forensic AI logic)
├── models.py             # SQLAlchemy models: User, Case, AnalysisResult
├── config.py             # App configuration (upload/report folders, DB URI, secret key)
├── extensions.py         # Flask extension initialization (db = SQLAlchemy())
├── utils.py              # File validation helpers, allowed extension sets
├── requirements.txt      # Python dependencies
│
├── templates/            # Jinja2 HTML templates
│   ├── homepage.html
│   ├── login.html
│   ├── admin_dashboard.html
│   ├── officer_dashboard.html
│   ├── expert_dashboard.html
│   ├── ai_analysis.html
│   └── modules/          # One template per analysis module
│       ├── fingerprint.html
│       ├── gunshot.html
│       ├── deepfake.html
│       └── ... (32 module templates)
│
├── static/               # CSS, JS, and static assets
├── uploads/              # Temporary storage for uploaded evidence files
├── reports/              # Auto-generated PDF reports
└── instance/             # SQLite database (forensic.db)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- `mistral` model pulled in Ollama

### 1. Clone the Repository
```bash
git clone https://github.com/crastatelvin/forensic-ai-lab.git
cd forensic-ai-lab
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the Ollama Model
```bash
ollama pull mistral
```

### 5. Run the Application
```bash
python forensic_web.py
```

The app will:
- Auto-initialize the SQLite database
- Create default Admin, Officer, and Expert users
- Start the Flask development server at `http://127.0.0.1:5000`

---

## 💻 Usage

### Running an Analysis
1. Log in as **Officer** or **Expert**
2. Navigate to **AI Analysis** from the dashboard
3. Select any of the 32 forensic modules
4. Upload crime scene evidence and suspect evidence files
5. Click **Analyze** — the system will return:
   - A **match score** (0.0–1.0)
   - A **Match / No Match** verdict
   - Visual plots of extracted features
   - A **downloadable PDF report**

### Managing Cases (Admin)
1. Log in as **Admin**
2. View all cases and users from the admin dashboard
3. Assign cases to specific officers or experts
4. Monitor system-wide analysis activity

---

## 🤖 AI Forensic Chatbot

The platform includes a real-time forensic AI assistant powered by **Ollama (Mistral)** running locally via subprocess CLI.

### Capabilities
- Answer forensic investigation questions with contextual reasoning
- Analyze case database in real time (total cases, case types, status breakdown)
- Maintain a 10-message rolling conversation history per user session
- Respond with formatted, structured output (auto-formatted bullet-points and paragraphs)

### How It Works
```python
# Chatbot pipeline (forensic_web.py)
def ask_ollama(prompt: str) -> str:
    # 1. Check if the prompt is a data analysis request → query SQLite directly
    if "data analysis" in prompt.lower() or "total cases" in prompt.lower():
        return handle_data_analysis(prompt)
    
    # 2. Otherwise, route to Ollama CLI with the Mistral model
    result = subprocess.run(["ollama", "run", "mistral", prompt], ...)
    return format_response(result.stdout)
```

### Chat Endpoint
```
POST /chat
Content-Type: application/json

{
  "user_id": "session-user-id",
  "message": "How many cases are completed?"
}
```

Response:
```json
{
  "reply": "Here is the data analysis:\n\nTotal Cases: 14\nCompleted Cases: 9\n..."
}
```

---

## 📄 PDF Report Generation

Every analysis module auto-generates a structured PDF report using **ReportLab**, saved in the `reports/` folder.

### Report Contents
- Analysis title
- Match score
- Match/No Match verdict
- Crime scene and suspect evidence images (base64 decoded and embedded)
- Spacer-separated sections

### Report Download Routes
- **Single report**: `GET /download_report/<report_filename>`
- **All officer reports (ZIP)**: `GET /officer/download_all_reports`
- **Expert CSV export**: `GET /expert/export_analysis_csv`

---

## 🗃️ Database Models

Managed via **Flask-SQLAlchemy** with a SQLite backend (`instance/forensic.db`).

### `User`
| Field | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Auto-increment ID |
| `username` | String | Unique login name |
| `password` | String | Plain-text password (hash in production) |
| `role` | String | `admin`, `officer`, or `expert` |

### `Case`
| Field | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Case ID |
| `analysis_type` | String | Module used (e.g., `fingerprint`, `deepfake`) |
| `status` | String | `pending`, `assigned`, or `completed` |
| `user_id` | FK → User | Assigned officer |
| `expert_id` | FK → User | Assigned expert |

### `AnalysisResult`
| Field | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Result ID |
| `case_id` | FK → Case | Associated case |
| `module` | String | Analysis module name |
| `result` | String | Verdict message |
| `score` | Float | Match score (0.0–1.0) |
| `user_id` | FK → User | User who ran the analysis |
| `created_at` | DateTime | Timestamp |

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///forensic.db'
    UPLOAD_FOLDER = 'uploads/'
    REPORT_FOLDER = 'reports/'
    SECRET_KEY = 'your-secure-secret-key'   # Change this!
```

---

## 🔒 Security Notes

> This project is built for **educational and research purposes** only.

- Passwords are stored in plain text — implement `bcrypt` or `werkzeug.security` hashing before any real deployment
- The `SECRET_KEY` in `config.py` must be replaced with a cryptographically secure random string
- Implement HTTPS and proper file sanitization for production use
- The Ollama chatbot runs locally — no data leaves your machine

---

## 📦 Requirements

Key dependencies from `requirements.txt`:

```
flask
flask-sqlalchemy
opencv-python
face-recognition
librosa
numpy
matplotlib
pillow
torch
torchvision
ultralytics
transformers
geopy
reportlab
requests
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-module`
3. Commit your changes: `git commit -m 'Add new forensic module'`
4. Push to the branch: `git push origin feature/new-module`
5. Open a Pull Request

### Ideas for Contribution
- Add real ML models to modules currently using simulated feature extraction
- Add password hashing and session security hardening
- Build a REST API layer for external integrations
- Add support for additional LLMs via Ollama
- Implement file encryption for evidence uploads

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Crasta Telvin](https://github.com/crastatelvin)

⭐ Star this repo if you find it useful!

</div>
