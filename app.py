from flask import Flask, request, send_file
import subprocess
import uuid
import os
import re
from faster_whisper import WhisperModel

app = Flask(__name__)

# ===== CONFIG =====
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
MAX_CPL = 14
MAX_DURATION = 2.0

# ===== EMOJIS FR =====
EMOJI_MAP = {
    "argent": "💰",
    "riche": "💰",
    "richesse": "💰",
    "million": "💰",
    "business": "📈",
    "succès": "🏆",
    "réussir": "🏆",
    "réussite": "🏆",
    "gagner": "🏆",
    "temps": "⏳",
    "vie": "⏳",
    "jour": "📅",
    "cerveau": "🧠",
    "mental": "🧠",
    "esprit": "🧠",
    "discipline": "💪",
    "travaille": "💪",
    "fort": "💪",
    "peur": "😨",
    "stress": "😰",
    "danger": "🔥",
    "maintenant": "🚀",
    "commence": "🚀",
    "démarre": "🚀",
    "vas": "🚀",
    "amour": "❤️",
    "coeur": "❤️",
    "secret": "🤫",
    "incroyable": "🤯"
}

def add_emoji_end(text):
    t = text.lower()
    for word, emoji in EMOJI_MAP.items():
        if word in t:
            return text + " " + emoji
    return text

# ===== Whisper (low RAM) =====
MODEL = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1
)

# ===== ROUTES =====

@app.get("/")
def home():
    return {
        "service": "ffmpeg-whisper-subtitles-api",
        "status": "online",
        "model": WHISPER_MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "render": "/render"
        }
    }, 200

@app.get("/health")
def health():
    return "ok", 200

# ===== UTILS =====

def get_duration_seconds(path):
    p = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(p.stdout.strip())

def normalize(t):
    return re.sub(r"\s+", " ", t).strip()

def split_cpl(text):
    text = normalize(text)
    words = text.split(" ")
    out = []
    cur = ""

    for w in words:
        if len(cur) + len(w) + 1 <= MAX_CPL:
            cur = (cur + " " + w).strip()
        else:
            out.append(cur)
            cur = w

    if cur:
        out.append(cur)
    return out

def srt_ts(sec):
    ms = int(sec * 1000)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def make_srt(segments, out_path):
    lines = []
    idx = 1

    for seg in segments:
        start = float(seg.start)
        end = float(seg.end)
        text = add_emoji_end(seg.text)
        parts = split_cpl(text)

        if not parts:
            continue

        dur = (end - start) / len(parts)

        for i, p in enumerate(parts):
            s = start + i * dur
            e = min(s + MAX_DURATION, start + (i+1) * dur)

            lines.append(str(idx))
            lines.append(f"{srt_ts(s)} --> {srt_ts(e)}")
            lines.append(p)
            lines.append("")
            idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ===== MAIN RENDER =====

@app.post("/render")
def render():

    if "video" not in request.files or "audio" not in request.files:
        return {"error": "Missing video or audio"}, 400

    video = request.files["video"]
    audio = request.files["audio"]

    vpath = f"/tmp/{uuid.uuid4()}.mp4"
    apath = f"/tmp/{uuid.uuid4()}.mp3"
    srt = f"/tmp/{uuid.uuid4()}.srt"
    out = f"/tmp/{uuid.uuid4()}.mp4"

    video.save(vpath)
    audio.save(apath)

    try:
        duration = get_duration_seconds(apath)
    except:
        return {"error": "Cannot read audio duration"}, 500

    try:
        segments, _ = MODEL.transcribe(apath, vad_filter=True)
        segments = list(segments)
        make_srt(segments, srt)
    except Exception as e:
        return {"error": "Transcription failed", "details": str(e)}, 500

    cmd = [
        "ffmpeg", "-y",
        "-i", vpath,
        "-i", apath,
        "-t", str(duration),
        "-vf", f"subtitles={srt}:force_style='FontName=Noto Emoji,FontSize=14,Outline=2,Shadow=1,MarginV=90'",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "32",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        out
    ]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0:
        return {"error": "ffmpeg failed", "stderr": p.stderr[-2000:]}, 500

    return send_file(out, mimetype="video/mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


