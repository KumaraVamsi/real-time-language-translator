"""
Blink Chat – main.py  (v5 – definitive)

pip install flask flask-sqlalchemy werkzeug SpeechRecognition
            "googletrans==4.0.0-rc1" gTTS pydub textblob
python -m textblob.download_corpora
System: ffmpeg on PATH

WHAT WAS WRONG IN v4 AND HOW IT IS FIXED HERE
═══════════════════════════════════════════════
1. HISTORY "session expired" on every open
   Cause: secret_key changed every version → old cookies invalid.
   Fix: Read key from secret.key file (created once, never changes).
        session.permanent=True → 7-day cookie.
        /history ALWAYS returns JSON, never HTML redirect.

2. TAB — audio stops when capture starts
   Cause: getDisplayMedia stream's audio was fed into AudioContext
   but never routed back to speakers, so the tab went mute.
   Fix: /tab-translate is the same. The speaker routing fix is in JS.

3. CONVO — only one mic works at a time
   Cause: The JS submitSpeak() function was calling
   $('cst').textContent='Translating…' which set a SHARED status bar,
   and more critically the recognition restart logic had a race condition
   where convRunning[person] could be read as true when it should be false.
   Fix: Per-person status tracking, recognition restart guard fixed in JS.
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
from googletrans import LANGUAGES, Translator
from gtts import gTTS

try:
    from textblob import TextBlob
    _TB_OK = True
except ImportError:
    _TB_OK = False

# ── Persistent secret key ─────────────────────────────────────────────────────
_KEY_FILE = Path(__file__).parent / "secret.key"
if _KEY_FILE.exists():
    _SECRET = _KEY_FILE.read_text().strip()
else:
    _SECRET = _secrets.token_hex(32)
    _KEY_FILE.write_text(_SECRET)

app = Flask(__name__)
app.secret_key = _SECRET
app.config["SQLALCHEMY_DATABASE_URI"]        = "sqlite:///blinkchat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["PERMANENT_SESSION_LIFETIME"]     = 60 * 60 * 24 * 7  # 7 days

db    = SQLAlchemy(app)
_tr   = Translator()
_pool = ThreadPoolExecutor(max_workers=16)

os.makedirs("uploads", exist_ok=True)
os.makedirs("static",  exist_ok=True)

_lang_map = {name.lower(): code for code, name in LANGUAGES.items()}

def gl(name: str) -> str:
    return _lang_map.get(name.strip().lower(), name.strip()[:2].lower())

# ── Sentiment ─────────────────────────────────────────────────────────────────
_INTENTS = [
    ("greeting",    ["hello","hi","hey","good morning","greetings"]),
    ("farewell",    ["bye","goodbye","see you","take care"]),
    ("question",    ["?","what","where","when","why","how","who","can you","do you"]),
    ("request",     ["please","i need","i want","help me","give me","could you"]),
    ("gratitude",   ["thank","thanks","appreciate","grateful"]),
    ("apology",     ["sorry","apologize","apologies","forgive","pardon"]),
    ("agreement",   ["yes","yeah","sure","okay","ok","absolutely","agreed"]),
    ("disagreement",["no","nope","never","disagree","wrong","incorrect"]),
    ("complaint",   ["bad","terrible","awful","horrible","worst","hate"]),
    ("compliment",  ["great","excellent","amazing","wonderful","fantastic","perfect"]),
]

def analyse(text):
    r = {"sentiment":"neutral","polarity":0.0,"subjectivity":0.0,"intent":"general"}
    if not text: return r
    if _TB_OK:
        try:
            b = TextBlob(text)
            p = round(b.sentiment.polarity, 3)
            s = round(b.sentiment.subjectivity, 3)
            r.update({"sentiment": "positive" if p>0.15 else ("negative" if p<-0.15 else "neutral"),
                      "polarity": p, "subjectivity": s})
        except Exception: pass
    tl = text.lower()
    for intent, kws in _INTENTS:
        if any(k in tl for k in kws):
            r["intent"] = intent; break
    return r

# ── Core helpers ──────────────────────────────────────────────────────────────
def do_tts(text, lang):
    fn = f"{uuid.uuid4()}.mp3"
    fp = os.path.join(app.static_folder, fn)
    gTTS(text=text, lang=lang, slow=False).save(fp)
    return f"/static/{fn}"

def translate_and_tts(text, src, dst):
    t = _tr.translate(text, src=src, dest=dst).text
    return t, do_tts(t, dst)

def save_bg(spoken, translated, user_id):
    def _w():
        with app.app_context():
            db.session.add(Conversation(
                spoken=spoken, translated=translated, user_id=user_id,
                created_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
            db.session.commit()
    _pool.submit(_w)

def convert_to_wav(src_path):
    out = src_path + "_out.wav"
    r = subprocess.run(["ffmpeg","-y","-i",src_path,"-vn","-ac","1","-ar","16000",
                        "-acodec","pcm_s16le",out], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr.decode()[-300:]}")
    return out

def do_stt(wav_path, lang):
    rec = sr.Recognizer()
    rec.energy_threshold = 200
    rec.dynamic_energy_threshold = True
    with sr.AudioFile(wav_path) as src:
        audio = rec.record(src)
    return rec.recognize_google(audio, language=lang)

def rm(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try: os.remove(p)
            except: pass

# ── Models ────────────────────────────────────────────────────────────────────
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

class Conversation(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    spoken     = db.Column(db.Text)
    translated = db.Column(db.Text)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.String(30), default="")

conv_rooms: dict = {}

# ── DB init + migration ───────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    import sqlite3 as _sq
    _dbpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blinkchat.db")
    try:
        _c = _sq.connect(_dbpath)
        cols = [r[1] for r in _c.execute("PRAGMA table_info(conversation)").fetchall()]
        if "created_at" not in cols:
            _c.execute("ALTER TABLE conversation ADD COLUMN created_at TEXT DEFAULT ''")
            _c.commit()
        _c.close()
    except Exception as e:
        print(f"[migration] {e}")

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        u = request.form["username"].strip()
        if User.query.filter_by(username=u).first():
            return "Username already exists"
        db.session.add(User(username=u, password=generate_password_hash(request.form["password"])))
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"].strip()).first()
        if user and check_password_hash(user.password, request.form["password"]):
            session.permanent = True
            session["user_id"]  = user.id
            session["username"] = user.username
            return redirect(url_for("index"))
        return "Invalid credentials"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", languages=list(LANGUAGES.values()),
                           username=session["username"])

# ── Mic ───────────────────────────────────────────────────────────────────────
@app.route("/translate", methods=["POST"])
def translate():
    if "user_id" not in session: return jsonify({"error":"Unauthorized"})
    src = gl(request.form["from_language"])
    dst = gl(request.form["to_language"])
    rec = sr.Recognizer()
    try:
        with sr.Microphone() as mic:
            rec.adjust_for_ambient_noise(mic, duration=0.3)
            audio = rec.listen(mic, timeout=8, phrase_time_limit=8)
        spoken = rec.recognize_google(audio, language=src)
    except Exception as e:
        return jsonify({"error": f"Mic error: {e}"})
    try:
        trans, aurl = translate_and_tts(spoken, src, dst)
        save_bg(spoken, trans, session["user_id"])
        sa = analyse(spoken)
        return jsonify({"spoken_text":spoken,"translated_text":trans,"audio":aurl,
                        "sentiment":sa["sentiment"],"polarity":sa["polarity"],
                        "subjectivity":sa["subjectivity"],"intent":sa["intent"]})
    except Exception as e:
        return jsonify({"error": str(e)})

# ── Text ──────────────────────────────────────────────────────────────────────
@app.route("/text-translate-tts", methods=["POST"])
def text_translate_tts():
    if "user_id" not in session: return jsonify({"error":"Unauthorized"})
    src  = gl(request.form["from_language"])
    dst  = gl(request.form["to_language"])
    text = request.form.get("text","").strip()
    if not text: return jsonify({"error":"Empty text"})
    try:
        trans, aurl = translate_and_tts(text, src, dst)
        save_bg(text, trans, session["user_id"])
        sa = analyse(text)
        return jsonify({"translated_text":trans,"audio":aurl,
                        "sentiment":sa["sentiment"],"polarity":sa["polarity"],
                        "subjectivity":sa["subjectivity"],"intent":sa["intent"]})
    except Exception as e:
        return jsonify({"error": str(e)})

# ── Audio file ────────────────────────────────────────────────────────────────
@app.route("/audio-translate", methods=["POST"])
def audio_translate():
    if "user_id" not in session: return jsonify({"error":"Unauthorized"})
    src = gl(request.form["from_language"])
    dst = gl(request.form["to_language"])
    if "audio_file" not in request.files: return jsonify({"error":"No audio file"})
    f   = request.files["audio_file"]
    ext = (f.filename or "audio.bin").rsplit(".",1)[-1].lower() or "bin"
    raw = os.path.join("uploads", f"{uuid.uuid4()}.{ext}"); wav = None; f.save(raw)
    try:
        wav = convert_to_wav(raw); spoken = do_stt(wav, src)
        trans, aurl = translate_and_tts(spoken, src, dst)
        save_bg(spoken, trans, session["user_id"])
        return jsonify({"spoken_text":spoken,"translated_text":trans,"audio":aurl})
    except sr.UnknownValueError: return jsonify({"error":"Could not understand speech."})
    except sr.RequestError as e: return jsonify({"error":f"STT error: {e}"})
    except Exception as e:       return jsonify({"error":str(e)})
    finally: rm(raw, wav)

# ── Video file ────────────────────────────────────────────────────────────────
@app.route("/video-translate", methods=["POST"])
def video_translate():
    if "user_id" not in session: return jsonify({"error":"Unauthorized"})
    src = gl(request.form["from_language"])
    dst = gl(request.form["to_language"])
    if "video_file" not in request.files: return jsonify({"error":"No video file"})
    f   = request.files["video_file"]
    ext = (f.filename or "video.mp4").rsplit(".",1)[-1].lower() or "mp4"
    raw = os.path.join("uploads", f"{uuid.uuid4()}.{ext}"); wav = None; f.save(raw)
    try:
        wav = convert_to_wav(raw); spoken = do_stt(wav, src)
        trans, aurl = translate_and_tts(spoken, src, dst)
        save_bg(spoken, trans, session["user_id"])
        return jsonify({"spoken_text":spoken,"translated_text":trans,"audio":aurl})
    except sr.UnknownValueError: return jsonify({"error":"No recognisable speech."})
    except sr.RequestError as e: return jsonify({"error":f"STT error: {e}"})
    except Exception as e:       return jsonify({"error":str(e)})
    finally: rm(raw, wav)

# ── Tab translate ─────────────────────────────────────────────────────────────
# text= path  : Web Speech sends recognised text → translate+TTS  (<300ms)
# audio_blob= : MediaRecorder webm chunk → ffmpeg+STT → translate+TTS (fallback)
@app.route("/tab-translate", methods=["POST"])
def tab_translate():
    src  = gl(request.form.get("from_language","english"))
    dst  = gl(request.form.get("to_language","english"))
    text = request.form.get("text","").strip()
    raw = wav = None

    if not text and "audio_blob" in request.files:
        f   = request.files["audio_blob"]
        raw = os.path.join("uploads", f"{uuid.uuid4()}.webm"); f.save(raw)
        try:
            wav  = convert_to_wav(raw)
            text = do_stt(wav, src)
        except sr.UnknownValueError:
            return jsonify({"ok":False,"reason":"no_speech"})
        except Exception as e:
            return jsonify({"ok":False,"reason":str(e)})
        finally:
            rm(raw, wav)

    if not text:
        return jsonify({"ok":False,"reason":"empty"})

    try:
        if src == dst:
            trans, aurl = text, do_tts(text, dst)
        else:
            trans, aurl = translate_and_tts(text, src, dst)
        if "user_id" in session:
            save_bg(text, trans, session["user_id"])
        return jsonify({"ok":True,"spoken_text":text,"translated_text":trans,"audio":aurl})
    except Exception as e:
        return jsonify({"ok":False,"reason":str(e)})

# ── Conversation ──────────────────────────────────────────────────────────────
@app.route("/conv/create", methods=["POST"])
def conv_create():
    if "user_id" not in session: return jsonify({"error":"Unauthorized"})
    lang_a = request.form.get("lang_a","english")
    lang_b = request.form.get("lang_b","english")
    code   = _secrets.token_hex(3).upper()
    conv_rooms[code] = {
        "participants": {"a":{"lang":lang_a}, "b":{"lang":lang_b}},
        "messages": [], "host": session["user_id"]
    }
    return jsonify({"room_code":code,"lang_a":lang_a,"lang_b":lang_b})

@app.route("/conv/join", methods=["POST"])
def conv_join():
    if "user_id" not in session: return jsonify({"error":"Unauthorized"})
    code = request.form.get("room_code","").strip().upper()
    if code not in conv_rooms: return jsonify({"error":"Room not found."})
    r = conv_rooms[code]
    return jsonify({"room_code":code,
                    "lang_a":r["participants"]["a"]["lang"],
                    "lang_b":r["participants"]["b"]["lang"]})

@app.route("/conv/speak", methods=["POST"])
def conv_speak():
    if "user_id" not in session: return jsonify({"error":"Unauthorized"})
    code   = request.form.get("room_code","").strip().upper()
    person = request.form.get("person","a")
    text   = request.form.get("text","").strip()
    if code not in conv_rooms or not text:
        return jsonify({"error":"Bad request"})
    r = conv_rooms[code]; parts = r["participants"]
    other    = "b" if person=="a" else "a"
    src_lang = gl(parts.get(person,{}).get("lang","english"))
    dst_lang = gl(parts.get(other, {}).get("lang","english"))
    try:
        trans, aurl = translate_and_tts(text, src_lang, dst_lang)
        msg = {"person":person,"spoken":text,"translated":trans,
               "audio":aurl,"ts":datetime.utcnow().strftime("%H:%M:%S")}
        r["messages"].append(msg)
        save_bg(text, trans, session["user_id"])
        return jsonify(msg)
    except Exception as e:
        return jsonify({"error":str(e)})

@app.route("/conv/poll")
def conv_poll():
    code  = request.args.get("room_code","").strip().upper()
    since = int(request.args.get("since", 0))
    if code not in conv_rooms:
        return jsonify({"messages":[],"total":0})
    msgs = conv_rooms[code]["messages"]
    return jsonify({"messages":msgs[since:],"total":len(msgs)})

# ── History ───────────────────────────────────────────────────────────────────
# ALWAYS returns JSON — never redirect, never HTML
@app.route("/history")
def history():
    if "user_id" not in session:
        resp = make_response(jsonify({"error":"not_logged_in","rows":[]}), 200)
        resp.headers["Content-Type"] = "application/json"
        return resp
    rows = (Conversation.query
            .filter_by(user_id=session["user_id"])
            .order_by(Conversation.id.desc())
            .limit(50).all())
    return jsonify([{"spoken":c.spoken or "","translated":c.translated or "",
                     "time":c.created_at or ""} for c in rows])

if __name__ == "__main__":
    app.run(debug=True, threaded=True)