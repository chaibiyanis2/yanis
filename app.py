from flask import Flask, request, send_file
import subprocess
import uuid
import os
from faster_whisper import WhisperModel

app = Flask(__name__)

# Cache du modèle pour éviter de le recharger à chaque requête
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")  # "tiny" / "base" / "small"
MODEL = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
    cpu_threads=1,
    num_workers=1
)

@app.get("/")
def home():
    return {
        "service": "ffmpeg-whisper-subtitles-api",
        "status": "online",
        "model": WHISPER_MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "render": "/render (POST multipart/form-data: video, audio)"
        }
    }, 200


@app.get("/health")
def health():
    return "ok", 200


def get_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"ffprobe failed: {p.stderr}")
    return float(p.stdout.strip())


def srt_timestamp(seconds: float) -> str:
    # format: HH:MM:SS,mmm
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, out_path: str):
    lines = []
    i = 1
    for seg in segments:
        start = float(seg.start)
        end = float(seg.end)
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
        lines.append(text)
        lines.append("")  # blank line
        i += 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@app.post("/render")
def render():
    if "video" not in request.files or "audio" not in request.files:
        return {"error": "Missing 'video' or 'audio' file fields"}, 400

    video = request.files["video"]
    audio = request.files["audio"]

    video_path = f"/tmp/{uuid.uuid4()}.mp4"
    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    srt_path = f"/tmp/{uuid.uuid4()}.srt"
    output_path = f"/tmp/{uuid.uuid4()}.mp4"

    video.save(video_path)
    audio.save(audio_path)

    # Durée audio réelle (pour T(output)=T(audio))
    try:
        audio_duration = get_duration_seconds(audio_path)
    except Exception as e:
        return {"error": "Could not read audio duration", "details": str(e)}, 500

    # Transcription Whisper -> segments -> SRT
    try:
        segments, info = MODEL.transcribe(
            audio_path,
            vad_filter=True,
            word_timestamps=False
        )
        segments = list(segments)
        write_srt(segments, srt_path)
    except Exception as e:
        return {"error": "Transcription failed", "details": str(e)}, 500

    # Petite marge anti-arrondi
    audio_duration = max(0.1, audio_duration + 0.05)

    # Style des sous-titres (simple et lisible)
    # Tu peux modifier Fontsize, Outline, Shadow, etc.
    force_style = "FontName=DejaVu Sans,FontSize=14,Outline=2,Shadow=1,MarginV=90"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-t", f"{audio_duration}",

        # Brûler les sous-titres
        "-vf", f"subtitles={srt_path}:force_style='{force_style}'",

        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0 or not os.path.exists(output_path):
        return {
            "error": "ffmpeg failed",
            "returncode": p.returncode,
            "stderr": p.stderr[-2500:]
        }, 500

    return send_file(output_path, mimetype="video/mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


