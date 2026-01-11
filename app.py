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

MAX_CPL = 14                 # max caractères par sous-titre
CAP_DUR = 0.5                # ✅ 500ms max par sous-titre
FONT_SIZE = 14
MARGIN_V = 90                # marge sous-titre vers le bas
EMOJI_SIZE = 44              # taille emoji (px)
EMOJI_RAISE_PX = 40          # ✅ emoji plus haut que le texte (augmente si besoin)
EMOJI_GAP_PX = 12            # espace entre texte et emoji

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

# ===== NORMALIZATION =====
def strip_accents(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def norm_key(s: str) -> str:
    s = strip_accents(s).lower()
    s = s.replace("’", "'")
    return s

def normalize_space(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

# ===== ROUTES =====
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
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(p.stderr.strip() or "ffprobe failed")
    return float(p.stdout.strip())

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
    # approx largeur texte en px (suffisant pour placement à côté)
    n = len(text)
    return int((n * FONT_SIZE * 0.55) / 2)

def split_by_cpl_words(words, max_cpl):
    """
    words = list of dicts: {"w":str,"s":float,"e":float}
    Sort des groupes (captions) <= max_cpl chars.
    """
    groups = []
    cur = []
    cur_len = 0

    for wd in words:
        w = wd["w"]
        add_len = len(w) + (1 if cur else 0)
        if cur and cur_len + add_len > max_cpl:
            groups.append(cur)
            cur = [wd]
            cur_len = len(w)
        else:
            cur.append(wd)
            cur_len += add_len

    if cur:
        groups.append(cur)
    return groups

def build_captions_from_words(all_words):
    """
    Construit des captions avec timing basé sur les mots.
    Chaque caption dure max 0.5s (CAP_DUR).
    """
    captions = []
    i = 0
    n = len(all_words)

    while i < n:
        start = all_words[i]["s"]
        end_limit = start + CAP_DUR

        # prendre autant de mots que possible dans 0.5s
        j = i
        chunk = []
        while j < n and all_words[j]["e"] <= end_limit:
            chunk.append(all_words[j])
            j += 1

        # si aucun mot ne tient (cas rare), prendre au moins 1 mot
        if not chunk:
            chunk = [all_words[i]]
            j = i + 1

        # maintenant appliquer CPL (max 14 chars) en sous-groupes
        subgroups = split_by_cpl_words(chunk, MAX_CPL)

        # temps de base pour ce bloc 0.5s : on répartit
        block_start = chunk[0]["s"]
        block_end = min(chunk[-1]["e"], block_start + CAP_DUR)
        block_dur = max(0.01, block_end - block_start)
        per = block_dur / len(subgroups)

        for k, g in enumerate(subgroups):
            text = " ".join(x["w"] for x in g)
            s = block_start + k * per
            e = min(s + per, block_end)
            captions.append((s, e, text))

        i = j

    return captions

def make_srt_and_emoji_events_from_captions(captions, full_text, srt_path):
    """
    ✅ SRT écrit depuis captions (timing précis)
    ✅ Emoji: 1 seule fois (sur la dernière caption)
    """
    lines = []
    idx = 1
    events = []

    for s, e, txt in captions:
        lines.append(str(idx))
        lines.append(f"{srt_ts(s)} --> {srt_ts(e)}")
        lines.append(txt)
        lines.append("")
        idx += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Emoji une seule fois (dernier sous-titre)
    emoji_file = pick_emoji_file(full_text)
    if emoji_file and captions:
        s, e, txt = captions[-1]
        events.append((s, e, emoji_file, txt))

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
        # ✅ word_timestamps=True => timing exact
        segments, _ = MODEL.transcribe(
            apath,
            vad_filter=True,
            word_timestamps=True
        )

        all_words = []
        full_text = []
        for seg in segments:
            if seg.text:
                full_text.append(seg.text)
            if getattr(seg, "words", None):
                for w in seg.words:
                    ww = normalize_space(w.word)
                    if ww:
                        all_words.append({"w": ww, "s": float(w.start), "e": float(w.end)})

        full_text = normalize_space(" ".join(full_text))

        if not all_words:
            # fallback: si aucun mot, on renvoie sans sous-titres
            return {"error": "No words extracted from transcription"}, 500

        captions = build_captions_from_words(all_words)
        emoji_events = make_srt_and_emoji_events_from_captions(captions, full_text, srt)

    except Exception as e:
        return {"error": "Transcription failed", "details": str(e)}, 500

    # ---- FFmpeg: burn subtitles + overlay emoji PNG ----
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

        filter_steps.append(f"[{inp_index}:v]scale={EMOJI_SIZE}:-1[em{idx_ev}]")

        half_w = estimate_text_half_width_px(shown_text)
        x_px = half_w + EMOJI_GAP_PX

        # ✅ y plus haut que les sous-titres
        # Sous-titre visuel ~ bas: H - MARGIN_V - (font size)
        # Emoji: on le monte avec EMOJI_RAISE_PX
        y_expr = f"H-h-{MARGIN_V + EMOJI_RAISE_PX}"

        filter_steps.append(
            f"[{current}][em{idx_ev}]overlay="
            f"x=(W/2)+{x_px}:y={y_expr}:"
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
