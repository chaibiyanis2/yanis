from flask import Flask, request, send_file
import subprocess
import uuid
import os

app = Flask(__name__)

@app.get("/")
def home():
    return {
        "service": "ffmpeg-render-api",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "render": "/render (POST multipart/form-data: video, audio)"
        }
    }, 200

@app.get("/health")
def health():
    return "ok", 200


def get_audio_duration_seconds(audio_path: str) -> float:
    """
    Retourne la durée réelle (en secondes) via ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"ffprobe failed: {p.stderr}")
    return float(p.stdout.strip())


@app.post("/render")
def render():
    if "video" not in request.files or "audio" not in request.files:
        return {"error": "Missing 'video' or 'audio' file fields"}, 400

    video = request.files["video"]
    audio = request.files["audio"]

    video_path = f"/tmp/{uuid.uuid4()}.mp4"
    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    output_path = f"/tmp/{uuid.uuid4()}.mp4"

    video.save(video_path)
    audio.save(audio_path)

    try:
        audio_duration = get_audio_duration_seconds(audio_path)
    except Exception as e:
        return {"error": "Could not read audio duration", "details": str(e)}, 500

    # Petite marge pour éviter les coupes à cause des arrondis/metadata
    audio_duration = max(0.1, audio_duration + 0.05)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,

        # ✅ on force la durée = durée réelle de l'audio
        "-t", f"{audio_duration}",

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
