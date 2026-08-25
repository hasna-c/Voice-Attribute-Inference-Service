# Voice Attribute Inference Service

A lightweight, production-ready backend service designed for logistics voice AI agents to infer caller attributes (**gender**, **age bracket**, and **audio quality**) directly from incoming audio streams.

---

## Features
* **Fast & Lightweight Inference:** Powered by FastAPI and pre-trained Wav2Vec2 models combined with a robust acoustic pitch ($F_0$) fallback layer.
* **Logistics-Grade Noise Handling:** Evaluates signal energy (RMS) and spectral flatness to flag degraded or insufficient audio conditions gracefully.
* **Strict Privacy (Zero PII Storage):** Audio files are processed entirely in volatile memory buffers (`io.BytesIO`) with zero disk persistence.
* **Containerized:** Fully ready for container deployment via Docker.

---

## API Contract

### `POST /analyze`
Accepts a multipart form upload containing an audio file.

**Request Form Fields:**
* `file`: Audio file (`.wav`, `.mp3`, etc.)
* `contact_id`: Optional string (auto-generates a UUID if omitted)

**Sample Response (`200 OK`):**
```json
{
  "contact_id": "e4b2af23-1fe4-46b3-9e9d-206bbe748488",
  "gender": {
    "prediction": "female",
    "confidence": 0.72
  },
  "age_bracket": {
    "prediction": "18-30",
    "confidence": 0.34
  },
  "language": {
    "prediction": "en",
    "confidence": 0.95
  },
  "processing_ms": 1606,
  "audio_quality": "good"
} 
```

## Setup & Running via Docker
### 1. Build the Docker image
```bash
docker build -t voice-attribute-service .
```

### 2. Run the container and map port 8000
```bash
docker run -p 8000:8000 voice-attribute-service
```

### open browser to the interactive API documentation:
```bash
http://localhost:8000/docs
```
### Architecture & Design Rationale
To build this service, I chose **FastAPI** for its blazing-fast async performance, low overhead, and automatic documentation, which is crucial for real-time logistics voice integrations. For attribute inference, I utilized **Wav2Vec2** pretrained models combined with lightweight acoustic feature extraction (RMS and spectral flatness via `librosa`) to handle logistics-grade noise (warehouse echoes, background truck engines) and gracefully flag degraded or insufficient audio rather than failing silently. Zero PII persistence is strictly enforced by routing incoming audio files through volatile in-memory buffers (`io.BytesIO`) without writing anything to disk.
With more time, I would incorporate a dedicated noise-suppression pre-processing filter (like `noisereduce`) and train a multi-task classification head directly on a logistics-focused dataset (e.g., VoxCeleb/Mozilla Common Voice) to boost age bracket confidence. To scale this architecture to **1,000 concurrent calls**, I would scale the containerized service horizontally behind an Nginx or Traefik reverse proxy using Docker Compose scale-out workers, introduce an asynchronous task queue (such as Celery with Redis) or a gRPC worker pool, and implement model caching to handle high throughput efficiently.
### Sample Test & Smoke Test Instructions
```bash
pip install pytest httpx
```
```bash
pytest
```


