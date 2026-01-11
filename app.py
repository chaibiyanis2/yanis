from flask import Flask, request, send_file
import subprocess
import uuid
import os
import re
import unicodedata
from faster_whisper import WhisperModel

app = Flask(__name__)

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
MAX_CPL = 14
MAX_DURATION = 2.0

MODEL = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1
)

# Mots FR -> fichier PNG emoji (overlay)
EMOJI_RULES = [
    # 💰 argent
    (["argent", "riche", "richesse", "million", "euros", "euro", "cash", "fortune"], "money.png"),

    # 🏆 réussite
    (["succes", "reussir", "reussite", "gagner", "victoire", "recompense"], "trophy.png"),

    # 🧠 mental
    (["cerveau", "mental", "esprit", "mindset", "penser", "reflechir"], "brain.png"),

    # 💪 action / travail
    (["travail", "travaille", "travailler", "effort", "discipline", "constance", "persiste", "persister"], "fire.png"),

    # ⏳ temps / vie
    (["temps", "minute", "jour", "jours", "vie"], "rocket.png"),

    # ⚠️ blocage / echec / probleme
    (["bloque", "blocage", "echec", "echouer", "probleme", "risque", "danger"], "warning.png"),

    # ❤️ émotion
    (["amour", "coeur"], "heart.png"),
]


def strip_accents(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def norm_key(s: str) -> str:
    s = strip_accents(s).lower()
    s = s.replace("’", "'")  # apostrophe “courbe” -> normale
    return s

@app.get("/")
def home():
    return {
        "service": "ffmpeg-whisper-subtitles-emoji-overlay",
        "status": "online",
        "model": WHISPER_MODEL_NAME,
        "endpoints": {"health": "/health", "render": "/render"}
    }, 200

@app.get("/health")
def health():
    return "ok", 200

def get_duration_seconds(path):
    p = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(p.stdout.strip())

def normalize(t):
    return re.sub(r"\s+", " ", (t or "")).strip()

def split_cpl(text):
    text = normalize(text)
    if not text:
        return []
    words = text.split(" ")
    out, cur = [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= MAX_CPL:
            cur = (cur + " " + w).strip()
        else:
            if cur:
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

def pick_emoji_file(text: str):
    t = norm_key(text)
    for keywords, fname in EMOJI_RULES:
        if any(k in t for k in keywords):
            return fname
    return None

def make_srt_and_emoji_events(segments, srt_path):
    """
    Génère:
    - SRT (texte sans emoji unicode)
    - events emoji: [(start, end, emoji_file)]
    """
    lines = []
    events = []
    idx = 1

    for seg in segments:
        start = float(seg.start)
        end = float(seg.end)
        raw = normalize(seg.text)
        if not raw:
            continue

        emoji_file = pick_emoji_file(raw)
        parts = split_cpl(raw)
        if not parts:
            continue

        dur = (end - start) / len(parts)

        for i, ptxt in enumerate(parts):
            s = start + i * dur
            e = min(s + MAX_DURATION, start + (i + 1) * dur)

            lines.append(str(idx))
            lines.append(f"{srt_ts(s)} --> {srt_ts(e)}")
            lines.append(ptxt)
            lines.append("")
            idx += 1

            # ✅ emoji seulement à la fin "de la phrase" => on l’affiche pendant CE sous-titre
            if emoji_file:
                events.append((s, e, emoji_file))

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return events

@app.post("/render")
def render():
    if "video" not in request.files or "audio" not in request.files:
        return {"error": "Missing video or audio"}, 400

    vpath = f"/tmp/{uuid.uuid4()}.mp4"
    apath = f"/tmp/{uuid.uuid4()}.mp3"
    srt = f"/tmp/{uuid.uuid4()}.srt"
    out = f"/tmp/{uuid.uuid4()}.mp4"

    request.files["video"].save(vpath)
    request.files["audio"].save(apath)

    try:
        duration = get_duration_seconds(apath)
    except:
        return {"error": "Cannot read audio duration"}, 500

    try:
        segments, _ = MODEL.transcribe(apath, vad_filter=True)
        segments = list(segments)
        emoji_events = make_srt_and_emoji_events(segments, srt)
    except Exception as e:
        return {"error": "Transcription failed", "details": str(e)}, 500

    # ---- FFmpeg: burn subtitles then overlay emojis (PNG) ----
    # Inputs: 0=video, 1=audio, + one input per unique emoji PNG
    unique_emojis = []
    for _, _, f in emoji_events:
        if f not in unique_emojis:
            unique_emojis.append(f)

    ffmpeg_cmd = ["ffmpeg", "-y", "-i", vpath, "-i", apath]
    for f in unique_emojis:
        ffmpeg_cmd += ["-i", f"/app/emojis/{f}"]

    # Base: subtitles on video
    force_style = "FontName=DejaVu Sans,FontSize=14,Outline=2,Shadow=1,MarginV=90"
    filter_steps = []
    filter_steps.append(f"[0:v]subtitles={srt}:force_style='{force_style}'[v0]")

    # Overlay each event
    current = "v0"
    # position emoji: bottom-right-ish (tweak if you want)
    # scale to 48px width
    for idx_ev, (s, e, fname) in enumerate(emoji_events):
        inp_index = 2 + unique_emojis.index(fname)  # video=0, audio=1
        next_v = f"v{idx_ev+1}"
        filter_steps.append(
            f"[{inp_index}:v]scale=48:-1[em{idx_ev}]"
        )
        filter_steps.append(
            f"[{current}][em{idx_ev}]overlay=x=W-w-40:y=H-h-140:enable='between(t,{s:.3f},{e:.3f})'[{next_v}]"
        )
        current = next_v

    filter_complex = ";".join(filter_steps)

    ffmpeg_cmd += [
        "-t", str(duration),
        "-filter_complex", filter_complex,
        "-map", f"[{current}]",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "32",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        out
    ]

    p = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return {"error": "ffmpeg failed", "stderr": p.stderr[-2000:]}, 500

    return send_file(out, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

