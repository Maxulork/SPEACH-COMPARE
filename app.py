import os
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from models import db, Lesson, LessonAttempt
from analyzer import SpeechSimilarityAnalyzer

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lessons.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
with app.app_context():
    db.create_all()

analyzer = SpeechSimilarityAnalyzer()

@app.route('/')
def index():
    lessons = Lesson.query.order_by(Lesson.created_at.desc()).all()
    return render_template('index.html', lessons=lessons)

@app.route('/analyze', methods=['POST'])
def analyze():
    a1 = request.files.get('audio1')
    a2 = request.files.get('audio2')
    if not a1 or not a2:
        return jsonify(error="Both audio1 and audio2 are required"), 400

    fn1 = secure_filename(f"{time.time()*1000}_{a1.filename}")
    fn2 = secure_filename(f"{time.time()*1000}_{a2.filename}")
    p1 = os.path.join(app.config['UPLOAD_FOLDER'], fn1)
    p2 = os.path.join(app.config['UPLOAD_FOLDER'], fn2)
    a1.save(p1); a2.save(p2)

    visualize = request.form.get('visualize','yes')=='yes'
    sim, colored = analyzer.compute_similarity(
        analyzer.load_audio(p1),
        analyzer.load_audio(p2),
        visualize=visualize
    )

    return jsonify({
        "similarity": round(sim,2),
        "transcription_html": colored['html'],
        "transcription2": colored['transcription2'],
        "image": visualize
    }), 200

@app.route('/get_image')
def get_image():
    img = os.path.join(app.root_path, 'speech_comparison.png')
    if not os.path.exists(img):
        return "No image", 404
    return send_from_directory(app.root_path, 'speech_comparison.png')

@app.route('/add_lesson', methods=['POST'])
def add_lesson():
    try:
        name = request.form['name']
        text = request.form['transcription']
        f    = request.files.get('reference_audio')
        if not f:
            raise ValueError("Reference audio missing")
        fn = secure_filename(f"{time.time()*1000}_{f.filename}")
        fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        f.save(fp)

        lesson = Lesson(name=name, transcription=text, reference_path=fp)
        db.session.add(lesson); db.session.commit()
        return jsonify(success=True, id=lesson.id), 200

    except Exception as e:
        app.logger.exception("add_lesson failed")
        return jsonify(success=False, error=str(e)), 500

@app.route('/delete_lesson/<int:lesson_id>', methods=['POST'])
def delete_lesson(lesson_id):
    try:
        l = Lesson.query.get_or_404(lesson_id)
        db.session.delete(l); db.session.commit()
        return jsonify(success=True), 200
    except Exception as e:
        app.logger.exception("delete_lesson failed")
        return jsonify(success=False, error=str(e)), 500

@app.route('/update_lesson/<int:lesson_id>', methods=['POST'])
def update_lesson(lesson_id):
    try:
        lesson = Lesson.query.get_or_404(lesson_id)
        f = request.files.get('attempt')
        if not f:
            raise ValueError("Attempt audio missing")
        fn = secure_filename(f"{time.time()*1000}_{f.filename}")
        fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        f.save(fp)

        ref = analyzer.load_audio(lesson.reference_path)
        att = analyzer.load_audio(fp)
        sim, _ = analyzer.compute_similarity(ref, att, visualize=False)

        attempt = LessonAttempt(lesson=lesson, similarity=sim)
        db.session.add(attempt); db.session.commit()
        return jsonify(success=True, similarity=round(sim,2)), 200

    except Exception as e:
        app.logger.exception("update_lesson failed")
        return jsonify(success=False, error=str(e)), 500

if __name__ == "__main__":
    import os
    # Grab the PORT that Render assigns, default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    # Listen on 0.0.0.0 so it’s reachable externally
    app.run(host="0.0.0.0", port=port)

