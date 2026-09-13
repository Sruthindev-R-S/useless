# Switchboard State Counter API

A lightweight API using YOLO to detect electrical switches and count how many switches are **ON** vs **OFF** on a switchboard.

---

## 📁 Project Structure

```
.
├── app.py              # FastAPI server with /predict endpoint & interactive UI
├── model/
│   └── best.pt         # Fine-tuned YOLO weights (100% test accuracy)
├── test/               # Test switchboard images (1.jpeg - 8.jpeg)
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🚀 Running the Server

Start the API server:

```bash
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

- **Interactive Web UI**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Endpoint: `POST /predict`

### 1. Uploading an Image File (Multipart Form)
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test/6.jpeg"
```

### 2. Passing an Image Path (JSON)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "test/6.jpeg"}'
```

### 3. Sending Raw Binary Image Bytes
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: image/jpeg" \
  --data-binary "@test/6.jpeg"
```

---

## 📊 Response Format

```json
{
  "number of switch": 6,
  "on": 2,
  "off": 4
}
```
# useless
