FROM jrottenberg/ffmpeg:6.0-alpine

RUN apk add --no-cache python3 py3-pip
RUN pip install --no-cache-dir flask

WORKDIR /app

COPY app.py /app/app.py
COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 10000

ENTRYPOINT ["/app/start.sh"]
