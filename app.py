from flask import Flask, request, send_file
import subprocess
import uuid
import os
import re
import unicodedata
from faster_whisper import WhisperModel

app = Flask(__name__)

# ===== CONFIG =====
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
MAX_CPL = 14
SUB_MS = 500
SUB_DUR = SUB_MS / 1000.0  # ✅ 0.5s fixes

FONT_SIZE = 14
MARGIN_V = 90

MODEL = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1
)

# ===== EMOJI RULES (FR, sans accents) -> PNG =====
EMOJI_RULES = [
    (["argent", "riche", "richesse", "million", "euros", "euro", "cash", "fortune"], "money.png"),
    (["succes", "reussir", "reussite", "gagner", "victoire", "recompense"], "trophy.png"),
    (["cerveau", "mental", "esprit", "mindset", "penser", "reflechir"], "brain.png"),
    (["travail", "travaille", "travailler", "effort", "discipline", "constance", "persiste", "persister"], "fire.png"),
    (["temps", "minute", "jour", "jours", "vie"], "rocket.png"),
    (["bloque", "blocage", "echec", "echouer", "probleme", "risque", "danger"], "warning.png"),
    (["amour", "coeur"], "heart.png"),
]

def strip_accents(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def norm_key(s: str) -> str:
    s = strip_accents(s).lower()
    s = s.replace("’", "'")
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

# ===== UTILS =====

def get_duration_seconds(path: str) -> float:
    p = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(p.stderr.strip() or "ffprobe failed")
    return float(p.stdout.strip())

def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def split_cpl(text: str):
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

def srt_ts(sec: float) -> str:
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

def estimate_text_half_width_px(text: str) -> int:
    """
    Estimation simple de la moitié de la largeur du texte (en px).
    On approx: 0.55 * FONT_SIZE par caractère.
    """
    n = len(text)
    return int((n * FONT_SIZE * 0.55) / 2)

def make_srt_and_emoji_events(segments, srt_path: str):
    """
    ✅ Sous-titres 500ms fixes
    ✅ Emoji non répété: seulement sur le DERNIER sous-titre du segment
    ✅ Emoji position calculée pour être à côté du texte centré
    """
    lines = []
    events = []
    idx = 1
    last_emoji = None  # pour éviter répétition "emoji emoji"

    for seg in segments:
        start = float(seg.start)
        end = float(seg.end)
        raw = normalize(seg.text)
        if not raw:
            continue

        parts = split_cpl(raw)
        if not parts:
            continue

        emoji_file = pick_emoji_file(raw)

        # 🔥 500ms fixes: on enchaîne les parts toutes les 0.5s
        t = start
        for i, ptxt in enumerate(parts):
            s = t
            e = min(s + SUB_DUR, end)
            if e <= s:
                break

            lines.append(str(idx))
            lines.append(f"{srt_ts(s)} --> {srt_ts(e)}")
            lines.append(ptxt)
            lines.append("")
            idx += 1
            t = e

            # ✅ Emoji seulement sur le dernier morceau du segment
            if i == len(parts) - 1 and emoji_file:
                # ✅ pas d’emoji répété deux fois de suite
                if emoji_file != last_emoji:
                    events.append((s, e, emoji_file, ptxt))  # on garde aussi le texte
                    last_emoji = emoji_file

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
    except Exception as e:
        return {"error": "Cannot read audio duration", "details": str(e)}, 500

    try:
        segments, _ = MODEL.transcribe(apath, vad_filter=True)
        segments = list(segments)
        emoji_events = make_srt_and_emoji_events(segments, srt)
    except Exception as e:
        return {"error": "Transcription failed", "details": str(e)}, 500

    # ---- FFmpeg: burn subtitles then overlay emojis (PNG) ----
    unique_emojis = []
    for _, _, f, _ in emoji_events:
        if f not in unique_emojis:
            unique_emojis.append(f)

    ffmpeg_cmd = ["ffmpeg", "-y", "-i", vpath, "-i", apath]
    for f in unique_emojis:
        ffmpeg_cmd += ["-i", f"/app/emojis/{f}"]

    force_style = f"FontName=DejaVu Sans,FontSize={FONT_SIZE},Outline=2,Shadow=1,MarginV={MARGIN_V}"

    filter_steps = []
    filter_steps.append(f"[0:v]subtitles={srt}:force_style='{force_style}'[v0]")

    current = "v0"

    for idx_ev, (s, e, fname, shown_text) in enumerate(emoji_events):
        inp_index = 2 + unique_emojis.index(fname)
        next_v = f"v{idx_ev+1}"

        # taille emoji
        filter_steps.append(f"[{inp_index}:v]scale=44:-1[em{idx_ev}]")

        # ✅ Position: à côté du texte centré
        # Texte est centré -> son bord droit approx = W/2 + half_width
        half_w = estimate_text_half_width_px(shown_text)
        x_px = half_w + 18  # 18px d’espace entre texte et emoji

        # y aligné sur la même zone que les sous-titres (bas)
        filter_steps.append(
    f"[{current}][em{idx_ev}]overlay="
    f"x=(W/2)+{x_px}:y=H-h-{MARGIN_V + 30}:"
    f"enable='between(t,{s:.3f},{e:.3f})'[{next_v}]"
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
        return {"error": "ffmpeg failed", "stderr": p.stderr[-2500:]}, 500

    return send_file(out, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

