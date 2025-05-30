# Forensic AI Lab [Project F.A.I.L]

Forensic AI Lab is a comprehensive web-based application designed to assist forensic experts, law enforcement officers, and administrators in analyzing various types of evidence using AI-powered modules. The toolkit provides a user-friendly interface for uploading evidence, performing analyses, and generating detailed reports.

## Features

- **AI Analysis Modules**: Analyze evidence such as fingerprints, DNA, voiceprints, and more using advanced AI algorithms.
- **User Roles**: Supports multiple user roles (Admin, Officer, Expert) with role-specific dashboards and functionalities.
- **Evidence Upload**: Upload various types of evidence (images, audio, text, etc.) for analysis.
- **Result Visualization**: View analysis results with visual plots and detailed summaries.
- **Report Generation**: Generate and download PDF reports for completed analyses.
- **Case Management**: Manage cases, track statuses, and view historical analyses.
- **Chatbot Integration**: Interact with an AI chatbot for forensic insights and recommendations.

## Project Structure

forensic_toolkit/
├── ai_tools.py          # AI analysis functions for various evidence types
├── config.py            # Configuration settings for the application
├── extensions.py        # Flask extensions (e.g., database setup)
├── forensic_web.py      # Main Flask application file
├── models.py            # Database models for users, cases, and analysis results
├── requirements.txt     # Python dependencies
├── utils.py             # Utility functions (e.g., file validation)
├── static/              # Static assets (CSS, JS, images)
│   ├── css/
│   ├── js/
│   ├── icons/
│   └── ...
├── templates/           # HTML templates for the web interface
│   ├── modules/         # Templates for individual AI analysis modules
│   ├── admin_dashboard.html
│   ├── officer_dashboard.html
│   ├── expert_dashboard.html
│   └── ...
├── instance/            # Instance folder for database files
│   ├── forensic_lab.db
│   └── forensic.db
└── reports/             # Folder for generated PDF reports


## Installation

1. **Clone the Repository**:
   bash
   git clone https://github.com/your-username/forensic_toolkit.git
   cd forensic_toolkit
 

2. **Set Up a Virtual Environment**:
   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
  

3. **Install Dependencies**:
   bash
   pip install -r requirements.txt
   

4. **Set Up the Database**:
   Initialize the database by running the following command:
   bash
   python -c "from forensic_web import init_db; init_db()"
  

5. **Run the Application**:
   bash
   flask run
   

6. **Access the Application**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage

### User Roles

- **Admin**: Manage users, view system settings, and clear uploads/reports.
- **Officer**: Upload evidence, track case statuses, and view analysis results.
- **Expert**: Perform detailed analyses, export results as CSV, and generate reports.

### AI Analysis Modules

The application supports the following AI-powered modules:
- Fingerprint Analysis
- DNA Analysis
- Voiceprint Analysis
- Ballistics Analysis
- Toxicology Analysis
- And many more...

Each module has a dedicated page where users can upload evidence, view results, and download reports.

### Chatbot

Interact with the AI chatbot for forensic insights and recommendations. The chatbot is accessible from the dashboard.

## Configuration

The application uses a configuration file (`config.py`) to manage settings such as:
- Database URI
- Upload folder paths
- Allowed file extensions

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Developed by TELVIN CRASTA.
- Powered by Flask, SQLAlchemy, and AI libraries.
- Special thanks to the forensic science community for their insights and feedback.

## Contact

For questions or support, please contact crastatelvin@gmail.com.

