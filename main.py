"""
Blink Chat – main.py (Deep-Translator Version)

pip install flask flask-sqlalchemy werkzeug SpeechRecognition
            deep-translator gTTS textblob
python -m textblob.download_corpora
System requirement: ffmpeg installed
"""

import os, uuid, secrets as _secrets, subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, make_response)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS

try:
    from textblob import TextBlob
    _TB_OK = True
except:
    _TB_OK = False


# ─────────────────────────────────────────
# Language Map
# ─────────────────────────────────────────

LANGUAGES = {
    "en": "english",
    "te": "telugu",
    "hi": "hindi",
    "ta": "tamil",
    "kn": "kannada",
    "ml": "malayalam",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ja": "japanese",
    "ko": "korean",
    "zh-cn": "chinese",
    "ar": "arabic"
}

_lang_map = {v.lower(): k for k, v in LANGUAGES.items()}

def gl(name):
    return _lang_map.get(name.lower().strip(), "en")


# ─────────────────────────────────────────
# Persistent secret key
# ─────────────────────────────────────────

_KEY_FILE = Path(__file__).parent / "secret.key"

if _KEY_FILE.exists():
    _SECRET = _KEY_FILE.read_text().strip()
else:
    _SECRET = _secrets.token_hex(32)
    _KEY_FILE.write_text(_SECRET)


# ─────────────────────────────────────────
# Flask
# ─────────────────────────────────────────

app = Flask(__name__)

app.secret_key = _SECRET

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///blinkchat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 7

db = SQLAlchemy(app)

_pool = ThreadPoolExecutor(max_workers=12)

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)


# ─────────────────────────────────────────
# Sentiment
# ─────────────────────────────────────────

def analyse(text):

    r = {"sentiment":"neutral","polarity":0.0,"subjectivity":0.0}

    if _TB_OK:
        try:
            b = TextBlob(text)
            r["polarity"] = round(b.sentiment.polarity,3)
            r["subjectivity"] = round(b.sentiment.subjectivity,3)

            if r["polarity"] > 0.15:
                r["sentiment"] = "positive"
            elif r["polarity"] < -0.15:
                r["sentiment"] = "negative"

        except:
            pass

    return r


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def translate_text(text, src, dst):

    if src == dst:
        return text

    return GoogleTranslator(source=src, target=dst).translate(text)


def do_tts(text, lang):

    fn = f"{uuid.uuid4()}.mp3"

    path = os.path.join("static", fn)

    gTTS(text=text, lang=lang).save(path)

    return f"/static/{fn}"


def translate_and_tts(text, src, dst):

    t = translate_text(text, src, dst)

    return t, do_tts(t, dst)


def save_bg(spoken, translated, uid):

    def _w():
        with app.app_context():
            db.session.add(
                Conversation(
                    spoken=spoken,
                    translated=translated,
                    user_id=uid,
                    created_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            db.session.commit()

    _pool.submit(_w)


def convert_to_wav(src):

    out = src + ".wav"

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", src,
        "-ac", "1",
        "-ar", "16000",
        out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out


def speech_to_text(wav, lang):

    r = sr.Recognizer()

    with sr.AudioFile(wav) as source:
        audio = r.record(source)

    return r.recognize_google(audio, language=lang)


# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    spoken = db.Column(db.Text)
    translated = db.Column(db.Text)
    user_id = db.Column(db.Integer)
    created_at = db.Column(db.String(30))


conv_rooms = {}


with app.app_context():
    db.create_all()


# ─────────────────────────────────────────
# Auth
# ─────────────────────────────────────────

@app.route("/signup", methods=["GET","POST"])
def signup():

    if request.method == "POST":

        u = request.form["username"]

        if User.query.filter_by(username=u).first():
            return "Username exists"

        db.session.add(User(
            username=u,
            password=generate_password_hash(request.form["password"])
        ))

        db.session.commit()

        return redirect("/login")

    return render_template("signup.html")


@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        u = User.query.filter_by(username=request.form["username"]).first()

        if u and check_password_hash(u.password, request.form["password"]):

            session.permanent = True
            session["user_id"] = u.id
            session["username"] = u.username

            return redirect("/")

        return "Invalid login"

    return render_template("login.html")


@app.route("/")
def index():

    if "user_id" not in session:
        return redirect("/login")

    return render_template(
        "index.html",
        languages=list(LANGUAGES.values()),
        username=session["username"]
    )


# ─────────────────────────────────────────
# Text Translation
# ─────────────────────────────────────────

@app.route("/text-translate-tts", methods=["POST"])
def text_translate():

    src = gl(request.form["from_language"])
    dst = gl(request.form["to_language"])

    text = request.form["text"]

    translated, audio = translate_and_tts(text, src, dst)

    save_bg(text, translated, session["user_id"])

    sa = analyse(text)

    return jsonify({
        "spoken": text,
        "translated": translated,
        "audio": audio,
        "sentiment": sa["sentiment"]
    })


# ─────────────────────────────────────────
# Audio Translation
# ─────────────────────────────────────────

@app.route("/audio-translate", methods=["POST"])
def audio_translate():

    src = gl(request.form["from_language"])
    dst = gl(request.form["to_language"])

    f = request.files["audio_file"]

    raw = os.path.join("uploads", str(uuid.uuid4()) + ".webm")

    f.save(raw)

    wav = convert_to_wav(raw)

    spoken = speech_to_text(wav, src)

    translated, audio = translate_and_tts(spoken, src, dst)

    save_bg(spoken, translated, session["user_id"])

    os.remove(raw)
    os.remove(wav)

    return jsonify({
        "spoken": spoken,
        "translated": translated,
        "audio": audio
    })


# ─────────────────────────────────────────
# Conversation Rooms
# ─────────────────────────────────────────

@app.route("/conv/create", methods=["POST"])
def conv_create():

    code = _secrets.token_hex(3).upper()

    conv_rooms[code] = {
        "lang_a": request.form["lang_a"],
        "lang_b": request.form["lang_b"],
        "messages": []
    }

    return jsonify({"room_code": code})


@app.route("/conv/join", methods=["POST"])
def conv_join():

    code = request.form["room_code"].upper()

    if code not in conv_rooms:
        return jsonify({"error":"Room not found"})

    r = conv_rooms[code]

    return jsonify({
        "room_code": code,
        "lang_a": r["lang_a"],
        "lang_b": r["lang_b"]
    })


@app.route("/conv/speak", methods=["POST"])
def conv_speak():

    code = request.form["room_code"].upper()

    text = request.form["text"]

    person = request.form["person"]

    r = conv_rooms[code]

    src = gl(r["lang_a"] if person=="a" else r["lang_b"])
    dst = gl(r["lang_b"] if person=="a" else r["lang_a"])

    translated, audio = translate_and_tts(text, src, dst)

    msg = {
        "person": person,
        "spoken": text,
        "translated": translated,
        "audio": audio
    }

    r["messages"].append(msg)

    return jsonify(msg)


@app.route("/conv/poll")
def conv_poll():

    code = request.args.get("room_code")

    since = int(request.args.get("since",0))

    msgs = conv_rooms.get(code,{}).get("messages",[])

    return jsonify({
        "messages": msgs[since:],
        "total": len(msgs)
    })


# ─────────────────────────────────────────
# History
# ─────────────────────────────────────────

@app.route("/history")
def history():

    if "user_id" not in session:
        return jsonify({"rows":[]})

    rows = Conversation.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Conversation.id.desc()).limit(50).all()

    return jsonify([
        {
            "spoken": r.spoken,
            "translated": r.translated,
            "time": r.created_at
        }
        for r in rows
    ])


# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────

if __name__ == "__main__":

    port = int(os.environ.get("PORT",5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
