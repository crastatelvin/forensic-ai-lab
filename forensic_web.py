from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from config import Config
from ai_tools import (analyze_fingerprints, analyze_gunshot, analyze_deepfake, analyze_face, analyze_bloodstain,
                     analyze_ballistics, analyze_voiceprint, analyze_handwriting, analyze_tire_track, analyze_tool_mark,
                     analyze_fiber, analyze_shoe_print, analyze_digital_footprint, analyze_odor_profile, analyze_gait,
                     analyze_explosive, analyze_glass, analyze_bite_mark, analyze_pollen, analyze_paint, analyze_soil,
                     analyze_hair, analyze_insect, analyze_phishing, analyze_darkweb, analyze_liedetect, analyze_arson,
                     analyze_iris, analyze_toxicology, analyze_geospatial, analyze_dna, analyze_fingerprint_dust)
from utils import validate_file, ALLOWED_IMAGE_EXT, ALLOWED_AUDIO_EXT, ALLOWED_VIDEO_EXT, ALLOWED_TEXT_EXT
import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import base64
from zipfile import ZipFile
import requests  # Ensure this is imported at the top of the file

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = 'your-secret-key-here'  # Replace with a secure key

from extensions import db
db.init_app(app)

from models import User, Case, AnalysisResult

def init_db():
    with app.app_context():
        db.create_all()
        if not db.session.query(User).first():
            db.session.add(User(username='admin', password='admin123', role='admin'))
            db.session.add(User(username='officer', password='officer123', role='officer'))
            db.session.add(User(username='expert', password='expert123', role='expert'))
            db.session.commit()
            print("Database initialized with default users.")

init_db()

def generate_pdf_report(result, report_filename):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Forensic Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Match Score: {result['suspect_score']}", styles['Normal']))
    story.append(Paragraph(f"Result: {result['message']}", styles['Normal']))
    story.append(Spacer(1, 12))
    for key in ['crime_image', 'suspect_image', 'crime_plot', 'suspect_plot']:
        if key in result and result[key].startswith('data:image'):
            img_data = base64.b64decode(result[key].split(',')[1])
            img_path = os.path.join(app.config['REPORT_FOLDER'], f"temp_{key}.png")
            with open(img_path, 'wb') as f:
                f.write(img_data)
            story.append(Image(img_path, width=300, height=200))
            story.append(Spacer(1, 12))
            os.remove(img_path)
    doc.build(story)
    buffer.seek(0)
    report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
    with open(report_path, 'wb') as f:
        f.write(buffer.getvalue())
    return report_path

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/login/<role>', methods=['GET', 'POST'])
def login(role):
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password, role=role).first()
        if user:
            session['user_id'] = user.id
            session['role'] = user.role
            return redirect(url_for(f'{role}_dashboard'))
        flash('Invalid credentials')
    return render_template('login.html', role=role)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('role', None)
    return redirect(url_for('homepage'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login', role='admin'))
    users = User.query.all()
    cases = Case.query.all()
    experts = User.query.filter_by(role='expert').all()
    recent_results = AnalysisResult.query.order_by(AnalysisResult.id.desc()).limit(5).all()
    total_cases = len(cases)
    completed_cases = len([c for c in cases if c.status == 'completed'])
    upload_files = os.listdir(app.config['UPLOAD_FOLDER'])
    report_files = os.listdir(app.config['REPORT_FOLDER'])
    return render_template('admin_dashboard.html', users=users, cases=cases, recent_results=recent_results,
                           total_cases=total_cases, completed_cases=completed_cases,
                           upload_files=upload_files, report_files=report_files, experts=experts)

@app.route('/admin/add_user', methods=['POST'])
def add_user():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login', role='admin'))
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']
    if User.query.filter_by(username=username).first():
        flash('Username already exists')
    else:
        new_user = User(username=username, password=password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('User added successfully')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/edit_user/<int:user_id>', methods=['POST'])
def edit_user(user_id):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login', role='admin'))
    user = User.query.get_or_404(user_id)
    user.role = request.form['role']
    user.password = request.form['password']
    db.session.commit()
    flash('User updated successfully')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user/<int:user_id>')
def delete_user(user_id):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login', role='admin'))
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/assign_case/<int:case_id>', methods=['POST'])
def assign_case(case_id):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login', role='admin'))
    case = Case.query.get_or_404(case_id)
    assigned_user_id = request.form.get('assigned_user_id')
    if assigned_user_id and assigned_user_id.isdigit():  # Validate user_id
        case.user_id = int(assigned_user_id)
        case.status = 'assigned'
    else:
        case.user_id = None
        case.status = 'pending'
    db.session.commit()
    flash('Case assigned successfully')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/clear_uploads', methods=['POST'])
def clear_uploads():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login', role='admin'))
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    flash('Upload folder cleared')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/clear_reports', methods=['POST'])
def clear_reports():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login', role='admin'))
    for file in os.listdir(app.config['REPORT_FOLDER']):
        os.remove(os.path.join(app.config['REPORT_FOLDER'], file))
    flash('Report folder cleared')
    return redirect(url_for('admin_dashboard'))

@app.route('/officer_dashboard')
def officer_dashboard():
    if 'role' not in session or session['role'] != 'officer':
        return redirect(url_for('login', role='officer'))
    user_id = session['user_id']
    user = User.query.get(user_id)
    cases = Case.query.filter_by(user_id=user_id).all()
    total_cases = len(cases)
    completed_cases = len([c for c in cases if c.status == 'completed'])
    experts = User.query.filter_by(role='expert').all()
    recent_results = AnalysisResult.query.filter_by(user_id=user_id).order_by(AnalysisResult.id.desc()).limit(5).all()
    chart_data = {
        'case_status': {
            'Pending': len([c for c in cases if c.status == 'pending']),
            'Assigned': len([c for c in cases if c.status == 'assigned']),
            'Completed': completed_cases
        },
        'analysis_types': {c.analysis_type: sum(1 for ca in cases if ca.analysis_type == c.analysis_type) for c in cases}
    }
    total_officers = User.query.filter_by(role='officer').count()
    opened_cases = len([c for c in cases if c.status == 'assigned'])
    closed_cases = completed_cases
    unassigned_cases = len([c for c in cases if c.status == 'pending'])
    return render_template('officer_dashboard.html', user=user, cases=cases, recent_results=recent_results,
                          total_cases=total_cases, completed_cases=completed_cases, chart_data=json.dumps(chart_data),
                          total_officers=total_officers, opened_cases=opened_cases, closed_cases=closed_cases,
                          unassigned_cases=unassigned_cases, experts=experts)

@app.route('/expert_dashboard')
def expert_dashboard():
    if 'role' not in session or session['role'] != 'expert':
        return redirect(url_for('login', role='expert'))
    user_id = session['user_id']
    user = User.query.get(user_id)
    cases = Case.query.filter_by(expert_id=user_id).all()  # Fetch cases assigned to the expert in real-time
    recent_results = AnalysisResult.query.filter_by(user_id=user_id).order_by(AnalysisResult.id.desc()).limit(5).all()
    total_cases = len(cases)
    chart_data = {
        'scores': [{'date': r.created_at.strftime('%Y-%m-%d'), 'score': r.score} for r in recent_results],
        'module_usage': {r.module: sum(1 for res in AnalysisResult.query.filter_by(user_id=user_id).all() if res.module == r.module) for r in recent_results}
    }
    solved_cases = len([c for c in cases if c.status == 'completed'])
    pending_cases = len([c for c in cases if c.status == 'assigned'])
    not_accepted_cases = len([c for c in cases if c.status == 'pending'])
    return render_template('expert_dashboard.html', user=user, cases=cases, recent_results=recent_results,
                          total_cases=total_cases, chart_data=json.dumps(chart_data),
                          solved_cases=solved_cases, pending_cases=pending_cases, not_accepted_cases=not_accepted_cases)

@app.route('/officer/download_all_reports')
def download_all_reports_officer():
    if 'role' not in session or session['role'] != 'officer':
        return redirect(url_for('login', role='officer'))
    user_id = session['user_id']
    cases = Case.query.filter_by(user_id=user_id, status='completed').all()
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        for case in cases:
            report_path = os.path.join(app.config['REPORT_FOLDER'], f"{case.analysis_type}_report.pdf")
            if os.path.exists(report_path):
                zip_file.write(report_path, f"{case.analysis_type}_report_{case.id}.pdf")
    zip_buffer.seek(0)
    return send_file(zip_buffer, as_attachment=True, download_name='officer_reports.zip')

@app.route('/expert/export_analysis_csv')
def export_analysis_csv():
    if 'role' not in session or session['role'] != 'expert':
        return redirect(url_for('login', role='expert'))
    user_id = session['user_id']
    results = AnalysisResult.query.filter_by(user_id=user_id).all()
    csv_data = "Case ID,Module,Score,Message,Date\n"
    for r in results:
        csv_data += f"{r.case_id},{r.module},{r.score},{r.result},{r.created_at.strftime('%Y-%m-%d %H:%M')}\n"
    buffer = BytesIO()
    buffer.write(csv_data.encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='expert_analysis.csv', mimetype='text/csv')

@app.route('/ai_analysis')
def ai_analysis():
    # Render the AI Analysis Modules page
    return render_template('ai_analysis.html')

@app.route('/ai_analysis/ai_analyzer', methods=['GET', 'POST'])
def ai_analyzer():
    if request.method == 'GET':
        # Render the AI Analyzer chatbot page
        return render_template('modules/ai_analyzer.html')

    if request.method == 'POST':
        user_input = request.json.get('user_input', '').strip().lower()  # Normalize user input

        # Handle greetings
        if user_input in ['hi', 'hello', 'hey']:
            return jsonify({"response": "Hello! I’m here to assist you with forensic cases. How can I help?"})

        # Handle database connection query
        if "connected to my database" in user_input:
            return jsonify({"response": "Yes, I am connected to your database and ready to assist with forensic queries."})

        # Handle case count query
        if "how many cases" in user_input:
            case_count = Case.query.count()
            return jsonify({"response": f"There are currently {case_count} cases in the database."})

        # Handle case details query
        if "case details" in user_input:
            case_id = extract_case_id(user_input)
            if case_id:
                case = Case.query.get(case_id)
                if case:
                    return jsonify({"response": f"Case {case.id}: {case.name}, Type: {case.analysis_type}, Status: {case.status}, Description: {case.description}"})
                else:
                    return jsonify({"response": f"No case found with ID {case_id}."})
            return jsonify({"response": "Please provide a valid case ID to retrieve details."})

        # Handle forensic analysis recommendations
        if "recommend analysis" in user_input:
            return jsonify({"response": "Based on the evidence type, I recommend using modules like Fingerprint, DNA, or Ballistics. Let me know the evidence type for specific suggestions."})

        # Handle evidence-based suggestions
        if "evidence" in user_input:
            if "blood" in user_input:
                return jsonify({"response": "For blood evidence, you can use the Bloodstain or DNA analysis modules."})
            elif "fingerprint" in user_input:
                return jsonify({"response": "For fingerprint evidence, use the Fingerprint or Fingerprint Dust analysis modules."})
            elif "audio" in user_input:
                return jsonify({"response": "For audio evidence, try the Voiceprint or Gunshot analysis modules."})
            elif "video" in user_input:
                return jsonify({"response": "For video evidence, use the Deepfake or Gait analysis modules."})
            else:
                return jsonify({"response": "Please specify the type of evidence for tailored analysis suggestions."})

        # Default response for unrecognized queries
        return jsonify({"response": "I'm sorry, I couldn't understand your query. Please try asking in a different way or provide more details."})

def extract_case_id(user_input):
    """Extract case ID from user input if mentioned."""
    words = user_input.split()
    for word in words:
        if word.isdigit():
            return int(word)
    return None

@app.route('/download_report/<path:report_path>')
def download_report(report_path):
    full_path = os.path.join(app.config['REPORT_FOLDER'], report_path)
    return send_file(full_path, as_attachment=True)

# Analysis Routes (32 Modules)
@app.route('/ai_analysis/fingerprint', methods=['GET', 'POST'])
def fingerprint_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)

            # Perform fingerprint analysis
            result = analyze_fingerprints(crime_path, suspect_path)

            # Return the result directly on the page
            return render_template('modules/fingerprint.html', result=result)

    return render_template('modules/fingerprint.html')

@app.route('/ai_analysis/gunshot', methods=['GET', 'POST'])
def gunshot_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_audio']
        suspect_file = request.files['suspect_audio']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_AUDIO_EXT) and validate_file(suspect_file.filename, ALLOWED_AUDIO_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_gunshot(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'gunshot_report.pdf')
            case = Case(analysis_type='gunshot', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='gunshot', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/gunshot.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/gunshot.html')

@app.route('/ai_analysis/deepfake', methods=['GET', 'POST'])
def deepfake_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_video']
        suspect_file = request.files['suspect_video']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_VIDEO_EXT) and validate_file(suspect_file.filename, ALLOWED_VIDEO_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_deepfake(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'deepfake_report.pdf')
            case = Case(analysis_type='deepfake', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='deepfake', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/deepfake.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/deepfake.html')

@app.route('/ai_analysis/face', methods=['GET', 'POST'])
def face_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_face(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'face_report.pdf')
            case = Case(analysis_type='face', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='face', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/face.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/face.html')

@app.route('/ai_analysis/bloodstain', methods=['GET', 'POST'])
def bloodstain_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_bloodstain(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'bloodstain_report.pdf')
            case = Case(analysis_type='bloodstain', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='bloodstain', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/bloodstain.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/bloodstain.html')

@app.route('/ai_analysis/ballistics', methods=['GET', 'POST'])
def ballistics_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_ballistics(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'ballistics_report.pdf')
            case = Case(analysis_type='ballistics', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='ballistics', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/ballistics.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/ballistics.html')

@app.route('/ai_analysis/voiceprint', methods=['GET', 'POST'])
def voiceprint_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_audio']
        suspect_file = request.files['suspect_audio']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_AUDIO_EXT) and validate_file(suspect_file.filename, ALLOWED_AUDIO_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_voiceprint(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'voiceprint_report.pdf')
            case = Case(analysis_type='voiceprint', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='voiceprint', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/voiceprint.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/voiceprint.html')

@app.route('/ai_analysis/handwriting', methods=['GET', 'POST'])
def handwriting_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_handwriting(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'handwriting_report.pdf')
            case = Case(analysis_type='handwriting', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='handwriting', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/handwriting.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/handwriting.html')

@app.route('/ai_analysis/tire_track', methods=['GET', 'POST'])
def tire_track_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_tire_track(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'tire_track_report.pdf')
            case = Case(analysis_type='tire_track', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='tire_track', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/tire_track.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/tire_track.html')

@app.route('/ai_analysis/tool_mark', methods=['GET', 'POST'])
def tool_mark_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_tool_mark(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'tool_mark_report.pdf')
            case = Case(analysis_type='tool_mark', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='tool_mark', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/tool_mark.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/tool_mark.html')

@app.route('/ai_analysis/fiber', methods=['GET', 'POST'])
def fiber_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_fiber(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'fiber_report.pdf')
            case = Case(analysis_type='fiber', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='fiber', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/fiber.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/fiber.html')

@app.route('/ai_analysis/shoe_print', methods=['GET', 'POST'])
def shoe_print_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_shoe_print(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'shoe_print_report.pdf')
            case = Case(analysis_type='shoe_print', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='shoe_print', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/shoe_print.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/shoe_print.html')

@app.route('/ai_analysis/digital_footprint', methods=['GET', 'POST'])
def digital_footprint_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_text']
        suspect_file = request.files['suspect_text']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_TEXT_EXT) and validate_file(suspect_file.filename, ALLOWED_TEXT_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_digital_footprint(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'digital_footprint_report.pdf')
            case = Case(analysis_type='digital_footprint', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='digital_footprint', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/digital_footprint.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/digital_footprint.html')

@app.route('/ai_analysis/odor_profile', methods=['GET', 'POST'])
def odor_profile_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_text']
        suspect_file = request.files['suspect_text']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_TEXT_EXT) and validate_file(suspect_file.filename, ALLOWED_TEXT_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_odor_profile(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'odor_profile_report.pdf')
            case = Case(analysis_type='odor_profile', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='odor_profile', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/odor_profile.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/odor_profile.html')

@app.route('/ai_analysis/gait', methods=['GET', 'POST'])
def gait_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_video']
        suspect_file = request.files['suspect_video']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_VIDEO_EXT) and validate_file(suspect_file.filename, ALLOWED_VIDEO_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_gait(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'gait_report.pdf')
            case = Case(analysis_type='gait', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='gait', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/gait.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/gait.html')

@app.route('/ai_analysis/explosive', methods=['GET', 'POST'])
def explosive_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_explosive(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'explosive_report.pdf')
            case = Case(analysis_type='explosive', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='explosive', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/explosive.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/explosive.html')

@app.route('/ai_analysis/glass', methods=['GET', 'POST'])
def glass_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_glass(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'glass_report.pdf')
            case = Case(analysis_type='glass', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='glass', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/glass.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/glass.html')

@app.route('/ai_analysis/bite_mark', methods=['GET', 'POST'])
def bite_mark_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_bite_mark(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'bite_mark_report.pdf')
            case = Case(analysis_type='bite_mark', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='bite_mark', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/bite_mark.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/bite_mark.html')

@app.route('/ai_analysis/pollen', methods=['GET', 'POST'])
def pollen_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_pollen(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'pollen_report.pdf')
            case = Case(analysis_type='pollen', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='pollen', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/pollen.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/pollen.html')

@app.route('/ai_analysis/paint', methods=['GET', 'POST'])
def paint_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_paint(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'paint_report.pdf')
            case = Case(analysis_type='paint', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='paint', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/paint.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/paint.html')

@app.route('/ai_analysis/soil', methods=['GET', 'POST'])
def soil_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_soil(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'soil_report.pdf')
            case = Case(analysis_type='soil', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='soil', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/soil.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/soil.html')

@app.route('/ai_analysis/hair', methods=['GET', 'POST'])
def hair_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_hair(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'hair_report.pdf')
            case = Case(analysis_type='hair', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='hair', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/hair.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/hair.html')

@app.route('/ai_analysis/insect', methods=['GET', 'POST'])
def insect_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_insect(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'insect_report.pdf')
            case = Case(analysis_type='insect', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='insect', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/insect.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/insect.html')

@app.route('/ai_analysis/phishing', methods=['GET', 'POST'])
def phishing_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_text']
        suspect_file = request.files['suspect_text']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_TEXT_EXT) and validate_file(suspect_file.filename, ALLOWED_TEXT_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_phishing(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'phishing_report.pdf')
            case = Case(analysis_type='phishing', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='phishing', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/phishing.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/phishing.html')

@app.route('/ai_analysis/darkweb', methods=['GET', 'POST'])
def darkweb_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_text']
        suspect_file = request.files['suspect_text']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_TEXT_EXT) and validate_file(suspect_file.filename, ALLOWED_TEXT_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_darkweb(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'darkweb_report.pdf')
            case = Case(analysis_type='darkweb', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='darkweb', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/darkweb.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/darkweb.html')

@app.route('/ai_analysis/liedetect', methods=['GET', 'POST'])
def liedetect_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_audio']
        suspect_file = request.files['suspect_audio']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_AUDIO_EXT) and validate_file(suspect_file.filename, ALLOWED_AUDIO_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_liedetect(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'liedetect_report.pdf')
            case = Case(analysis_type='liedetect', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='liedetect', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/liedetect.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/liedetect.html')

@app.route('/ai_analysis/arson', methods=['GET', 'POST'])
def arson_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_arson(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'arson_report.pdf')
            case = Case(analysis_type='arson', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='arson', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/arson.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/arson.html')

@app.route('/ai_analysis/iris', methods=['GET', 'POST'])
def iris_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_iris(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'iris_report.pdf')
            case = Case(analysis_type='iris', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='iris', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/iris.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/iris.html')

@app.route('/ai_analysis/toxicology', methods=['GET', 'POST'])
def toxicology_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_text']
        suspect_file = request.files['suspect_text']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_TEXT_EXT) and validate_file(suspect_file.filename, ALLOWED_TEXT_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_toxicology(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'toxicology_report.pdf')
            case = Case(analysis_type='toxicology', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='toxicology', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/toxicology.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/toxicology.html')

@app.route('/ai_analysis/geospatial', methods=['GET', 'POST'])
def geospatial_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_text']
        suspect_file = request.files['suspect_text']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_TEXT_EXT) and validate_file(suspect_file.filename, ALLOWED_TEXT_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_geospatial(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'geospatial_report.pdf')
            case = Case(analysis_type='geospatial', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='geospatial', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/geospatial.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/geospatial.html')

@app.route('/ai_analysis/dna', methods=['GET', 'POST'])
def dna_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_text']
        suspect_file = request.files['suspect_text']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_TEXT_EXT) and validate_file(suspect_file.filename, ALLOWED_TEXT_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_dna(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'dna_report.pdf')
            case = Case(analysis_type='dna', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='dna', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/dna.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/dna.html')

@app.route('/ai_analysis/fingerprint_dust', methods=['GET', 'POST'])
def fingerprint_dust_analysis():
    if request.method == 'POST':
        crime_file = request.files['crime_image']
        suspect_file = request.files['suspect_image']
        if crime_file and suspect_file and validate_file(crime_file.filename, ALLOWED_IMAGE_EXT) and validate_file(suspect_file.filename, ALLOWED_IMAGE_EXT):
            crime_path = os.path.join(app.config['UPLOAD_FOLDER'], crime_file.filename)
            suspect_path = os.path.join(app.config['UPLOAD_FOLDER'], suspect_file.filename)
            crime_file.save(crime_path)
            suspect_file.save(suspect_path)
            result = analyze_fingerprint_dust(crime_path, suspect_path)
            report_path = generate_pdf_report(result, 'fingerprint_dust_report.pdf')
            case = Case(analysis_type='fingerprint_dust', user_id=session.get('user_id', 1), status='completed')
            db.session.add(case)
            db.session.commit()
            analysis_result = AnalysisResult(case_id=case.id, module='fingerprint_dust', result=result['message'], score=result['suspect_score'], user_id=session.get('user_id', 1))
            db.session.add(analysis_result)
            db.session.commit()
            return render_template('modules/fingerprint_dust.html', result=result, report_path=os.path.basename(report_path))
    return render_template('modules/fingerprint_dust.html')

@app.route('/search_cases', methods=['GET'])
def search_cases():
    if 'role' not in session:
        return redirect(url_for('homepage'))
    
    case_id = request.args.get('case_id')
    if not case_id or not case_id.isdigit():  # Validate case_id
        flash('Please enter a valid Case ID.')
        return redirect(url_for(f"{session['role']}_dashboard"))

    if session['role'] == 'admin':
        search_results = Case.query.filter(Case.id.like(f"%{case_id}%")).all()
    elif session['role'] == 'officer':
        search_results = Case.query.filter(Case.id.like(f"%{case_id}%"), Case.user_id == session['user_id']).all()
    elif session['role'] == 'expert':
        search_results = Case.query.filter(Case.id.like(f"%{case_id}%"), Case.user_id == session['user_id']).all()
    else:
        search_results = []

    return render_template(f"{session['role']}_dashboard.html", search_results=search_results)

@app.route('/create_case', methods=['POST'])
def create_case():
    if 'role' not in session or session['role'] not in ['admin', 'officer']:
        return redirect(url_for('homepage'))
    case_id = request.form['case_id']
    name = request.form['name']
    location = request.form['location']
    phone_number = request.form['phone_number']
    description = request.form['description']
    analysis_type = request.form['analysis_type']
    expert_id = request.form['expert_id']
    if expert_id and expert_id.isdigit():  # Validate expert_id
        expert_id = int(expert_id)
    else:
        expert_id = None
    evidence_file = request.files['evidence_file']
    evidence_path = None
    if evidence_file:
        evidence_path = os.path.join(app.config['UPLOAD_FOLDER'], evidence_file.filename)
        evidence_file.save(evidence_path)
    new_case = Case(
        id=case_id,
        name=name,
        location=location,
        phone_number=phone_number,
        description=description,
        analysis_type=analysis_type,
        expert_id=expert_id,
        evidence_file=evidence_path,
        user_id=session['user_id']
    )
    db.session.add(new_case)
    db.session.commit()
    flash('Case created successfully')
    return redirect(url_for(f"{session['role']}_dashboard"))

@app.route('/edit_case/<int:case_id>', methods=['POST'])
def edit_case(case_id):
    if 'role' not in session or session['role'] not in ['admin', 'officer']:
        return redirect(url_for('homepage'))
    case = Case.query.get_or_404(case_id)
    if session['role'] == 'officer' and case.user_id != session['user_id']:
        flash('You are not authorized to edit this case.')
        return redirect(url_for('officer_dashboard'))
    
    # Update case details
    case.name = request.form['name']
    case.location = request.form['location']
    case.phone_number = request.form['phone_number']
    case.description = request.form['description']
    case.analysis_type = request.form['analysis_type']
    expert_id = request.form.get('expert_id')
    if expert_id and expert_id.isdigit():  # Validate expert_id
        case.expert_id = int(expert_id)
    else:
        case.expert_id = None
    
    # Handle evidence file upload
    evidence_file = request.files['evidence_file']
    if evidence_file:
        evidence_path = os.path.join(app.config['UPLOAD_FOLDER'], evidence_file.filename)
        evidence_file.save(evidence_path)
        case.evidence_file = evidence_path
    
    db.session.commit()
    flash('Case updated successfully.')
    return redirect(url_for(f"{session['role']}_dashboard"))

@app.route('/delete_case/<int:case_id>')
def delete_case(case_id):
    if 'role' not in session or session['role'] not in ['admin', 'officer']:
        return redirect(url_for('homepage'))
    case = Case.query.get_or_404(case_id)
    if session['role'] == 'officer' and case.user_id != session['user_id']:
        flash('You are not authorized to delete this case.')
        return redirect(url_for('officer_dashboard'))
    db.session.delete(case)
    db.session.commit()
    flash('Case deleted successfully.')
    return redirect(url_for(f"{session['role']}_dashboard"))

@app.route('/view_case/<int:case_id>')
def view_case(case_id):
    if 'role' not in session or session['role'] != 'expert':
        return redirect(url_for('login', role='expert'))
    user_id = session['user_id']
    user = User.query.get(user_id)  # Fetch the logged-in user
    case = Case.query.get_or_404(case_id)
    return render_template('expert_dashboard.html', user=user, case=case, view_case=True)

@app.route('/acknowledge_case/<int:case_id>', methods=['POST'])
def acknowledge_case(case_id):
    if 'role' not in session or session['role'] != 'expert':
        return jsonify({'success': False, 'message': 'Unauthorized access'})
    case = Case.query.get_or_404(case_id)
    if case.status == 'assigned':
        case.status = 'in progress'
        db.session.commit()
        # Notify officer
        flash(f"Case {case.id} acknowledged by expert.")
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid case status'})

@app.route('/complete_case/<int:case_id>', methods=['POST'])
def complete_case(case_id):
    if 'role' not in session or session['role'] != 'expert':
        return redirect(url_for('login', role='expert'))
    case = Case.query.get_or_404(case_id)
    if case.status == 'in progress':
        report = request.files['report']
        if report:
            report_path = os.path.join(app.config['REPORT_FOLDER'], f"{case.id}_report.pdf")
            report.save(report_path)
            case.status = 'completed'
            db.session.commit()
            # Notify officer
            flash(f"Case {case.id} marked as completed by expert.")
            return redirect(url_for('expert_dashboard'))
    flash('Failed to complete case.')
    return redirect(url_for('view_case', case_id=case_id))

@app.route('/view_all_cases', methods=['GET', 'POST'])
def view_all_cases():
    if 'role' not in session or session['role'] not in ['admin', 'officer']:
        return redirect(url_for('homepage'))
    
    if request.method == 'POST':
        case_id = request.form['case_id']
        case = Case.query.get_or_404(case_id)
        
        # Update case status
        if 'status' in request.form:
            case.status = request.form['status']
            db.session.commit()
            flash(f"Case {case.id} status updated to {case.status}.")
        
        # Handle report upload
        if 'report' in request.files:
            report = request.files['report']
            if report:
                report_path = os.path.join(app.config['REPORT_FOLDER'], f"{case.id}_report.pdf")
                report.save(report_path)
                flash(f"Report uploaded for Case {case.id}.")
        
        return redirect(url_for('view_all_cases'))
    
    cases = Case.query.all()  # Fetch all cases without restrictions
    return render_template('view_all_cases.html', cases=cases)

@app.route('/case_count', methods=['GET'])
def case_count():
    if 'role' not in session:
        return jsonify({'error': 'Unauthorized access'}), 401

    case_count = Case.query.count()  # Count all cases in the database
    return jsonify({'case_count': case_count})

@app.cli.command('reset_db')
def reset_db():
    """Drops and recreates the database."""
    db.drop_all()
    db.create_all()
    db.session.add(User(username='admin', password='admin123', role='admin'))
    db.session.add(User(username='officer', password='officer123', role='officer'))
    db.session.add(User(username='expert', password='expert123', role='expert'))
    db.session.commit()
    print("Database reset and initialized with default users.")

if __name__ == '__main__':
    app.run(debug=True)