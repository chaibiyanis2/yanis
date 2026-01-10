FROM jrottenberg/ffmpeg:6.0-alpine

RUN apk add --no-cache python3 py3-pip
RUN pip install flask

WORKDIR /app
COPY app.py .

CMD ["python3", "app.py"]
