from flask import Flask, request, send_file
import subprocess
import uuid
import os
import re
import unicodedata
import json
from faster_whisper import WhisperModel

app = Flask(__name__)

# =========================
# CONFIG
# =========================
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")

MAX_CPL = 14
CAP_DUR = 0.5
FONT_SIZE = 14
MARGIN_V = 90

EMOJI_SIZE = 44
EMOJI_MAX_DUR = 0.5

VIDEO_VOL = 1.0
ADDED_AUDIO_VOL = 1.0

# MICRO progressive zoom (whole video)
ZOOM_START = 1.00
ZOOM_END = 1.03   # or 1.025
ZOOM_FPS = 30

MODEL = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1
)

# =========================
# EMOJI RULES
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
# TEXT NORMALIZATION
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
    s = s or ""
    s = s.replace("’", APOS)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*'\s*", "'", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s

# =========================
# ROUTES
# =========================
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
    if p.returncode != 0 or not (p.stdout or "").strip():
        raise RuntimeError((p.stderr or "").strip() or "ffprobe duration failed")
    return float(p.stdout.strip())

def get_video_stream_info(path: str):
    """
    Returns dict: width, height, rot(0/90/180/270)
    IMPORTANT: This reads *metadata* rotation, not auto-rotated pixels.
    """
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data",
         "-of", "json", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0 or not (p.stdout or "").strip():
        raise RuntimeError((p.stderr or "").strip() or "ffprobe stream info failed")

    data = json.loads(p.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe returned no video streams")

    st = streams[0]
    w = int(st.get("width") or 0)
    h = int(st.get("height") or 0)

    rot = 0
    tags = st.get("tags") or {}
    if "rotate" in tags:
        try:
            rot = int(tags["rotate"]) % 360
        except Exception:
            rot = 0

    sdl = st.get("side_data_list") or []
    for sd in sdl:
        if "rotation" in sd:
            try:
                rot = int(round(float(sd["rotation"]))) % 360
            except Exception:
                pass

    # Normalize to {0,90,180,270}
    if rot not in (0, 90, 180, 270):
        rot = 0

    return {"w": w, "h": h, "rot": rot}

def video_has_audio(path: str) -> bool:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index",
         "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return p.returncode == 0 and bool((p.stdout or "").strip())

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
    # Only ONE occurrence per emoji type for whole video
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
# SUBTITLE BUILDERS
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

    # Output ends when MP3 ends
    try:
        out_dur = get_duration_seconds(apath)
    except Exception as e:
        return {"error": "Cannot read mp3 duration", "details": str(e)}, 500

    # Read real width/height + rotation metadata
    try:
        info = get_video_stream_info(vpath)
        vw, vh, rot = info["w"], info["h"], info["rot"]
        if vw <= 0 or vh <= 0:
            raise RuntimeError("Invalid video width/height from ffprobe")
    except Exception as e:
        return {"error": "Cannot read video info", "details": str(e)}, 500

    # Decide ONE correction rotation (because we will disable auto-rotate in ffmpeg)
    transpose_filter = None
    disp_w, disp_h = vw, vh

    if rot == 90:
        # metadata says: rotate clockwise 90 to display => bake same into pixels
        transpose_filter = "transpose=1"
        disp_w, disp_h = vh, vw
    elif rot == 270:
        # rotate counterclockwise 90 to display
        transpose_filter = "transpose=2"
        disp_w, disp_h = vh, vw
    elif rot == 180:
        transpose_filter = "hflip,vflip"
        disp_w, disp_h = vw, vh

    # Transcribe added audio
    try:
        segments, _ = MODEL.transcribe(apath, vad_filter=True, word_timestamps=True)

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

    # unique emoji inputs
    unique_emojis = []
    for _, _, f, _ in emoji_events:
        if f not in unique_emojis:
            unique_emojis.append(f)

    # ✅ IMPORTANT: disable auto-rotation for video input
    ffmpeg_cmd = ["ffmpeg", "-y", "-noautorotate", "-i", vpath, "-i", apath]
    for f in unique_emojis:
        ffmpeg_cmd += ["-i", f"/app/emojis/{f}"]

    force_style = f"FontName=DejaVu Sans,FontSize={FONT_SIZE},Outline=2,Shadow=1,MarginV={MARGIN_V}"

    # stable zoompan (on final displayed size)
    total_frames = max(2, int(out_dur * ZOOM_FPS))
    z_delta = (ZOOM_END - ZOOM_START)
    zoom_expr = f"{ZOOM_START}+({z_delta})*on/{total_frames}"
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"

    filter_steps = []

    chain = []
    if transpose_filter:
        chain.append(transpose_filter)

    chain += [
        f"fps={ZOOM_FPS}",
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d=1:s={disp_w}x{disp_h}",
        "setpts=PTS-STARTPTS",
        f"subtitles={srt_path}:force_style='{force_style}'",
    ]

    filter_steps.append(f"[0:v]{','.join(chain)}[v0]")
    current = "v0"

    # emojis centered
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

    # audio mix
    if video_has_audio(vpath):
        filter_steps.append(f"[0:a]volume={VIDEO_VOL},aresample=async=1:first_pts=0[a0]")
        filter_steps.append(f"[1:a]volume={ADDED_AUDIO_VOL},aresample=async=1:first_pts=0[a1]")
        filter_steps.append("[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[aout]")
        audio_map = "[aout]"
    else:
        filter_steps.append(f"[1:a]volume={ADDED_AUDIO_VOL},aresample=async=1:first_pts=0[aout]")
        audio_map = "[aout]"

    filter_complex = ";".join(filter_steps)

    ffmpeg_cmd += [
        "-t", str(out_dur),
        "-filter_complex", filter_complex,
        "-map", f"[{current}]",
        "-map", audio_map,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "32",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        # ✅ avoid any extra rotation by players
        "-metadata:s:v:0", "rotate=0",
        out
    ]

    p = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return {"error": "ffmpeg failed", "stderr": (p.stderr or "")[-4000:]}, 500

    return send_file(out, mimetype="video/mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
