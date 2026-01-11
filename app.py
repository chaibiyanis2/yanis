from flask import Flask, request, send_file
import subprocess
import uuid
import os
import re
import unicodedata
from faster_whisper import WhisperModel

app = Flask(__name__)

# =========================
# CONFIG
# =========================
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")

MAX_CPL = 14            # max characters per subtitle line
CAP_DUR = 0.5           # ✅ 500ms per caption block
FONT_SIZE = 14
MARGIN_V = 90

EMOJI_SIZE = 44         # emoji PNG size (px)

# Emoji display behavior
EMOJI_MAX_DUR = 0.5     # seconds: cap emoji display duration (<= caption duration)

# NEW: if audio is longer than video, slow video toward 16s, then CUT output to audio length (no silence)
TARGET_VIDEO_SECONDS_WHEN_AUDIO_LONGER = 16.0

# Whisper low-RAM (Render Free friendly)
MODEL = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1
)

# =========================
# EMOJI RULES (FR, accent-insensitive) -> PNG filenames
# One type = one emoji file
# =========================
EMOJI_RULES = [
    (["argent", "riche", "richesse", "million", "euros", "euro", "cash", "fortune"], "money.png"),
    (["succes", "reussir", "reussite", "gagner", "victoire", "recompense"], "trophy.png"),
    (["cerveau", "mental", "esprit", "mindset", "penser", "reflechir"], "brain.png"),
    (["travail", "travaille", "travailler", "effort", "discipline", "constance", "persiste", "persister"], "fire.png"),
    (["temps", "minute", "jour", "jours", "vie"], "rocket.png"),
    (["bloque", "blocage", "echec", "echouer", "probleme", "risque", "danger"], "warning.png"),
    (["amour", "coeur"], "heart.png"),
]

# =========================
# TEXT NORMALIZATION (fix d 'etre -> d'etre)
# =========================
APOS = "'"

def strip_accents(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def norm_key(s: str) -> str:
    s = strip_accents(s).lower()
    s = s.replace("’", APOS)
    return s

def normalize_text(s: str) -> str:
    """
    - Normalize whitespace
    - Fix spaces around apostrophes: "d 'etre" -> "d'etre"
    - Fix spaces before punctuation: "toi !" -> "toi!"
    - Normalize curly apostrophe to '
    """
    s = s or ""
    s = s.replace("’", APOS)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*'\s*", "'", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {
        "service": "ffmpeg-whisper-subtitles-emoji-overlay",
        "status": "online",
        "model": WHISPER_MODEL_NAME,
        "endpoints": {"health": "/health", "render": "/render"},
        "subtitle": {"cap_dur_ms": int(CAP_DUR * 1000), "max_cpl": MAX_CPL},
        "emoji": {
            "size": EMOJI_SIZE,
            "max_dur_s": EMOJI_MAX_DUR,
            "position": "center_exact",
            "policy": "one_per_type_per_video"
        },
        "video_speed": {
            "if_audio_longer_than_video": "slow_video_then_cut_to_audio_end (no_silence)"
        }
    }, 200

@app.get("/health")
def health():
    return "ok", 200

# =========================
# MEDIA UTILS
# =========================
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

# =========================
# EMOJI PICKERS
# =========================
def pick_emoji_for_text(text: str):
    t = norm_key(text)
    for keywords, fname in EMOJI_RULES:
        if any(k in t for k in keywords):
            return fname
    return None

def make_emoji_events_from_captions(captions):
    """
    ✅ Emoji appears when the keyword appears in the caption text.
    ✅ Only ONE occurrence per emoji type for the whole video.
    ✅ Output: [(s, e, emoji_file, caption_text)]
    """
    events = []
    used_types = set()

    for (s, e, text) in captions:
        emoji = pick_emoji_for_text(text)
        if not emoji:
            continue

        if emoji in used_types:
            continue

        e2 = min(e, s + EMOJI_MAX_DUR)
        events.append((s, e2, emoji, text))
        used_types.add(emoji)

    return events

# =========================
# SUBTITLE BUILDERS (word timestamps)
# =========================
def split_cpl_words(words, max_cpl: int):
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
    captions = []
    i = 0
    n = len(all_words)

    while i < n:
        start = all_words[i]["s"]
        end_limit = start + CAP_DUR

        j = i
        chunk = []
        while j < n and all_words[j]["e"] <= end_limit:
            chunk.append(all_words[j])
            j += 1

        if not chunk:
            chunk = [all_words[i]]
            j = i + 1

        subgroups = split_cpl_words(chunk, MAX_CPL)

        block_start = chunk[0]["s"]
        block_end = min(chunk[-1]["e"], block_start + CAP_DUR)
        block_dur = max(0.02, block_end - block_start)
        per = block_dur / len(subgroups)

        for k, g in enumerate(subgroups):
            text = normalize_text(" ".join(x["w"] for x in g))
            s = block_start + k * per
            e = min(s + per, block_end)
            captions.append((s, e, text))

        i = j

    return captions

def write_srt(captions, srt_path: str):
    lines = []
    for idx, (s, e, text) in enumerate(captions, start=1):
        lines.append(str(idx))
        lines.append(f"{srt_ts(s)} --> {srt_ts(e)}")
        lines.append(text)
        lines.append("")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# =========================
# MAIN
# =========================
@app.post("/render")
def render():
    if "video" not in request.files or "audio" not in request.files:
        return {"error": "Missing video or audio"}, 400

    vpath = f"/tmp/{uuid.uuid4()}.mp4"
    apath = f"/tmp/{uuid.uuid4()}.mp3"
    srt_path = f"/tmp/{uuid.uuid4()}.srt"
    out = f"/tmp/{uuid.uuid4()}.mp4"

    request.files["video"].save(vpath)
    request.files["audio"].save(apath)

    # durations
    try:
        audio_duration = get_duration_seconds(apath)
        video_duration = get_duration_seconds(vpath)
    except Exception as e:
        return {"error": "Cannot read media duration", "details": str(e)}, 500

    # If audio longer than video -> slow video (toward 16s), but FINAL output ends with audio (no silence)
    slow_mode = (audio_duration > video_duration and video_duration > 0.05)

    # FINAL output duration: always ends with audio (and if slow_mode, cannot exceed slowed video length = 16s)
    if slow_mode:
        target_out_duration = min(audio_duration, TARGET_VIDEO_SECONDS_WHEN_AUDIO_LONGER)
    else:
        target_out_duration = audio_duration

    # Whisper with word timestamps
    try:
        segments, _ = MODEL.transcribe(
            apath,
            vad_filter=True,
            word_timestamps=True
        )

        all_words = []
        for seg in segments:
            if getattr(seg, "words", None):
                for w in seg.words:
                    ww = normalize_text(w.word)
                    if ww:
                        all_words.append({"w": ww, "s": float(w.start), "e": float(w.end)})

        if not all_words:
            return {"error": "No words extracted from transcription"}, 500

        captions = build_captions_from_words(all_words)
        write_srt(captions, srt_path)
        emoji_events = make_emoji_events_from_captions(captions)

    except Exception as e:
        return {"error": "Transcription failed", "details": str(e)}, 500

    # ---------- FFmpeg filters ----------
    unique_emojis = []
    for _, _, f, _ in emoji_events:
        if f not in unique_emojis:
            unique_emojis.append(f)

    ffmpeg_cmd = ["ffmpeg", "-y", "-i", vpath, "-i", apath]
    for f in unique_emojis:
        ffmpeg_cmd += ["-i", f"/app/emojis/{f}"]

    force_style = f"FontName=DejaVu Sans,FontSize={FONT_SIZE},Outline=2,Shadow=1,MarginV={MARGIN_V}"

    filter_steps = []

    # If slow_mode: slow the video so it reaches 16s (timeline becomes longer), then apply subtitles
    if slow_mode:
        speed_factor = TARGET_VIDEO_SECONDS_WHEN_AUDIO_LONGER / video_duration  # >1 => slow
        filter_steps.append(f"[0:v]setpts=PTS*{speed_factor:.8f}[vpre]")
        filter_steps.append(f"[vpre]subtitles={srt_path}:force_style='{force_style}'[v0]")
    else:
        filter_steps.append(f"[0:v]subtitles={srt_path}:force_style='{force_style}'[v0]")

    current = "v0"

    # Emoji centered in video
    x_center = "(W-w)/2"
    y_center = "(H-h)/2"

    for idx_ev, (s, e, fname, _shown_text) in enumerate(emoji_events):
        inp_index = 2 + unique_emojis.index(fname)
        next_v = f"v{idx_ev+1}"

        filter_steps.append(f"[{inp_index}:v]scale={EMOJI_SIZE}:-1[em{idx_ev}]")
        filter_steps.append(
            f"[{current}][em{idx_ev}]overlay="
            f"x={x_center}:y={y_center}:"
            f"enable='between(t,{s:.3f},{e:.3f})'[{next_v}]"
        )
        current = next_v

    # ✅ CUT VIDEO to audio end (no silence)
    filter_steps.append(f"[{current}]trim=0:{target_out_duration:.3f},setpts=PTS-STARTPTS[vout]")

    # ✅ CUT AUDIO to same end
    filter_steps.append(f"[1:a]atrim=0:{target_out_duration:.3f},asetpts=PTS-STARTPTS[a0]")

    filter_complex = ";".join(filter_steps)

    ffmpeg_cmd += [
        "-t", f"{target_out_duration:.3f}",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[a0]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "32",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out
    ]

    p = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return {"error": "ffmpeg failed", "stderr": p.stderr[-2500:]}, 500

    return send_file(out, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
