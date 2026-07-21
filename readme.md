## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## ffmpeg
Turn static video to rtsp
```bash
ffmpeg -re -stream_loop -1 -i "C:\test-video\sample.mp4" -c:v libx264 -preset veryfast -c:a aac -f rtsp rtsp://localhost:8554/live/stream1
```

## MediaMTX
Serve rtsp to webrtc, hsl, rtmp Edit mediamtx.yml
```yml
paths:
  axis-cam:
    source: rtsp://132.239.12.145/axis-media/media.amp
    sourceProtocol: tcp
    sourceOnDemand: yes

  test-cam:
    source: rtsp://localhost:8554/live/stream1
    sourceProtocol: tcp
    sourceOnDemand: yes
```
