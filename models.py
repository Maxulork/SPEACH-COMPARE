from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# Initialize the SQLAlchemy database instance.
# In app.py, you'll do:
#   from models import db
#   db.init_app(app)
#   with app.app_context():
#       db.create_all()
db = SQLAlchemy()

class Lesson(db.Model):
    """
    Represents a lesson (reference audio + transcription).
    Each lesson can have many user attempts.
    """
    __tablename__ = "lessons"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    transcription = db.Column(db.Text, nullable=False)
    reference_path = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # One-to-many: a lesson has multiple attempts
    attempts = db.relationship(
        "LessonAttempt",
        backref="lesson",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    def __repr__(self):
        return f"<Lesson id={self.id} name={self.name!r}>"

class LessonAttempt(db.Model):
    """
    Represents a user’s recording attempt at a lesson,
    storing the similarity score and a timestamp.
    """
    __tablename__ = "lesson_attempts"

    id = db.Column(db.Integer, primary_key=True)
    lesson_id = db.Column(
        db.Integer,
        db.ForeignKey("lessons.id", ondelete="CASCADE"),
        nullable=False
    )
    similarity = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<LessonAttempt id={self.id} lesson_id={self.lesson_id} similarity={self.similarity:.1f}%>"
