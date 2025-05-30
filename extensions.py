# extensions.py
# Defines shared Flask extensions to avoid circular imports

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()