FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fontconfig \
    fonts-dejavu \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir flask faster-whisper

# Emojis PNG (Twemoji) - petit pack
RUN mkdir -p /app/emojis && \
    curl -L -o /app/emojis/rocket.png https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f680.png && \
    curl -L -o /app/emojis/money.png  https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f4b0.png && \
    curl -L -o /app/emojis/trophy.png https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f3c6.png && \
    curl -L -o /app/emojis/brain.png  https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f9e0.png && \
    curl -L -o /app/emojis/fire.png   https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f525.png && \
    curl -L -o /app/emojis/heart.png  https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/2764.png && \
    curl -L -o /app/emojis/warning.png https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/26a0.png


COPY app.py /app/app.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 10000
ENTRYPOINT ["/app/start.sh"]

