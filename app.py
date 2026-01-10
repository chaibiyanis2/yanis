from flask import Flask, request, send_file
import subprocess
import uuid
import os

app = Flask(__name__)

@app.get("/health")
def health():
    return "ok", 200

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

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0 or not os.path.exists(output_path):
        return {
            "error": "ffmpeg failed",
            "returncode": p.returncode,
            "stderr": p.stderr[-2000:]  # juste la fin des logs
        }, 500

    return send_file(output_path, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
