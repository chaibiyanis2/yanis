from flask import Flask, request, send_file
import subprocess
import uuid
import os

app = Flask(__name__)

# Vérification que le serveur est vivant
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200


# Endpoint de montage vidéo
@app.route("/render", methods=["POST"])
def render():

    # Vérifier que les fichiers sont envoyés
    if "video" not in request.files or "audio" not in request.files:
        return {"error": "You must send 'video' and 'audio' files"}, 400

    video = request.files["video"]
    audio = request.files["audio"]

    # Générer des noms uniques
    video_path = f"/tmp/{uuid.uuid4()}.mp4"
    audio_path = f"/tmp/{uuid.uuid4()}.mp3"
    output_path = f"/tmp/{uuid.uuid4()}.mp4"

    # Sauvegarder les fichiers
    video.save(video_path)
    audio.save(audio_path)

    # Commande FFmpeg
    command = [
        "ffmpeg",
        "-y",
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

    # Exécuter FFmpeg
    subprocess.run(command)

    # Vérifier que la vidéo existe
    if not os.path.exists(output_path):
        return {"error": "Video rendering failed"}, 500

    # Envoyer la vidéo finale
    return send_file(output_path, mimetype="video/mp4")


# Lancer Flask
app.run(host="0.0.0.0", port=10000)