import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'forensic.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')

    @staticmethod
    def init_app(app):
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
        if not os.path.exists(Config.REPORT_FOLDER):
            os.makedirs(Config.REPORT_FOLDER)