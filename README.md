
---

# 🕵️‍♂️ Forensic AI Lab 

**Forensic AI Lab** is a comprehensive web-based application designed to assist forensic experts, law enforcement officers, and administrators in analyzing various types of evidence using AI-powered modules. The toolkit provides a user-friendly interface for uploading evidence, performing analyses, and generating detailed reports.

---

## ✨ Features

* 🤖 **AI Analysis Modules**: Analyze evidence such as fingerprints, DNA, voiceprints, and more using advanced AI algorithms.
* 👥 **User Roles**: Supports multiple user roles (Admin, Officer, Expert) with role-specific dashboards and functionalities.
* 📤 **Evidence Upload**: Upload various types of evidence (images, audio, text, etc.) for analysis.
* 📊 **Result Visualization**: View analysis results with visual plots and detailed summaries.
* 📄 **Report Generation**: Generate and download PDF reports for completed analyses.
* 🗂️ **Case Management**: Manage cases, track statuses, and view historical analyses.
* 💬 **Chatbot Integration**: Interact with an AI chatbot integrated with LLM for forensic insights and recommendations.

---

## 📁 Project Structure

```
forensic_toolkit/
├── ai_tools.py           # 🤖 AI analysis functions for various evidence types
├── config.py             # ⚙️ Configuration settings for the application
├── extensions.py         # 🔌 Flask extensions (e.g., database setup)
├── forensic_web.py       # 🌐 Main Flask application file
├── models.py             # 🗃️ Database models for users, cases, and analysis results
├── requirements.txt      # 📦 Python dependencies
├── utils.py              # 🧰 Utility functions (e.g., file validation)
├── static/               # 📁 Static assets (CSS, JS, images)
│   ├── css/
│   ├── js/
│   ├── icons/
│   └── ...
├── templates/            # 🖥️ HTML templates for the web interface
│   ├── modules/          # 🔧 Templates for individual AI analysis modules
│   ├── admin_dashboard.html
│   ├── officer_dashboard.html
│   ├── expert_dashboard.html
│   └── ...
├── instance/             # 🗄️ Instance folder for database files
│   ├── forensic_lab.db
│   └── forensic.db
└── reports/              # 📑 Folder for generated PDF reports
```

---

## 🚀 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/crastatelvin/forensic_toolkit.git
cd forensic_toolkit
```

2. **Set Up a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Up the Database**

```bash
python -c "from forensic_web import init_db; init_db()"
```

5. **Run the Application**

```bash
flask run
```

6. **Access the Application**
   Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧑‍💼 User Roles & Access

| Role    | Permissions & Features                                              |
| ------- | ------------------------------------------------------------------- |
| Admin   | Manage users, view system settings, clear uploads/reports.          |
| Officer | Upload evidence, track case statuses, view analysis results.        |
| Expert  | Perform detailed analyses, export results as CSV, generate reports. |

---

## 🔬 AI Analysis Modules

The application supports the following AI-powered forensic modules:

🔬 AI Analysis Modules
The application supports a wide range of AI-powered forensic modules to analyze diverse types of evidence:

🤖 AI Analyzer

🔥 Arson Analysis

🔫 Ballistics Analysis

🦷 Bite Mark Analysis

🩸 Bloodstain Analysis

🌐 Dark Web Analysis

🌑 Darkweb Analysis

🎭 Deepfake Detection

🌍 Digital Footprint Analysis

🧬 DNA Analysis

💥 Explosive Analysis

🧑‍🤝‍🧑 Face Recognition

🧵 Fiber Analysis

🖐️ Fingerprint Analysis

🚶 Gait Analysis

🗺️ Geospatial Analysis

🔍 Glass Analysis

🔫 Gunshot Analysis

🧑‍🦰 Hair Analysis

✍️ Handwriting Analysis

🐜 Insect Analysis

👁️ Iris Analysis

🕵️‍♂️ Lie Detection

📦 Object Detection

👃 Odor Profile Analysis

🎨 Paint Analysis

🎣 Phishing Analysis

🌸 Pollen Analysis

👞 Shoe Print Analysis

🌱 Soil Analysis

🚗 Tire Track Analysis

🛠️ Tool Mark Analysis

☠️ Toxicology Analysis

🎙️ Voiceprint Analysis


Each module includes a dedicated page for evidence upload, result visualization, and report downloads.

---

## 🤖 Chatbot Integration

Interact with the LLM integrated AI Forensic Agent chatbot directly from the dashboard for forensic insights, recommendations, and quick help.

---

## ⚙️ Configuration

Settings managed in `config.py` include:

* 🔗 Database URI
* 📂 Upload folder paths
* ✅ Allowed file extensions

---

## 🤝 Contributing

Contributions are **welcome**! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* Developed by **ILLUSION DISSOLUTION**.
* Powered by **Flask**, **SQLAlchemy**, and various **AI libraries**.
* Special thanks to the forensic science community for valuable insights and feedback.

---

