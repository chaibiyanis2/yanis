FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fontconfig \
    fonts-dejavu \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Télécharger une police emoji MONOCHROME (compatible libass)
RUN mkdir -p /usr/local/share/fonts && \
    curl -L -o /usr/local/share/fonts/NotoEmoji-Regular.ttf \
    https://raw.githubusercontent.com/googlefonts/noto-emoji/main/fonts/NotoEmoji-Regular.ttf && \
    fc-cache -f -v

WORKDIR /app
RUN pip install --no-cache-dir flask faster-whisper

COPY app.py /app/app.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 10000
ENTRYPOINT ["/app/start.sh"]
