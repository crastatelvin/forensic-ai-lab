from extensions import db  # Import db from extensions.py

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # e.g., 'admin', 'officer', 'expert'
    # Removed profile_image and details fields
    def __repr__(self):
        return f'<User {self.username} (Role: {self.role})>'

class Case(db.Model):
    __tablename__ = 'cases'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # Case name
    location = db.Column(db.String(200), nullable=False)  # Case location
    phone_number = db.Column(db.String(15), nullable=False)  # Contact phone number
    description = db.Column(db.Text, nullable=True)  # Case description/FIR
    evidence_file = db.Column(db.String(200), nullable=True)  # Path to uploaded evidence file
    analysis_type = db.Column(db.String(50), nullable=False)  # e.g., 'fingerprint', 'gunshot'
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))  # Nullable for unassigned cases
    expert_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Assigned expert
    status = db.Column(db.String(20), default='pending')  # 'pending', 'assigned', 'completed'
    created_at = db.Column(db.DateTime, default=db.func.now())  # Automatically set to current timestamp

    def __repr__(self):
        return f'<Case {self.name} (ID: {self.id}, Status: {self.status})>'

class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey('cases.id'), nullable=False)
    module = db.Column(db.String(50), nullable=False)  # e.g., 'fingerprint', 'deepfake'
    result = db.Column(db.String(200), nullable=False)  # e.g., 'High match probability'
    score = db.Column(db.Float, nullable=False)  # e.g., 0.95
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)  # Tracks who performed analysis
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp)

    def __repr__(self):
        return f'<AnalysisResult {self.module} (Score: {self.score}, Result: {self.result})>'