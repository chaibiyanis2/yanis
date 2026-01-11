FROM python:3.11-slim

# FFmpeg + ffprobe + support sous-titres + polices
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
RUN pip install --no-cache-dir flask faster-whisper

COPY app.py /app/app.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 10000
ENTRYPOINT ["/app/start.sh"]
