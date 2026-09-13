#!/usr/bin/env python3
"""
Switchboard State Detection & Counting API Server
Accepts switchboard images and returns:
- number of switch (total switches)
- on switch (number of switches in ON state)
- off switch (number of switches in OFF state)
"""

import json
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO

# Model path
MODEL_PATH = Path("model/best.pt")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

# Initialize device and model
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Loading switch detection model from {MODEL_PATH} on {device}...")
model = YOLO(str(MODEL_PATH))

app = FastAPI(
    title="Switchboard Switch Counter API",
    description="Inference API to detect switches and count total, ON, and OFF switches on a switchboard.",
    version="1.0.0",
)


def run_inference(img: np.ndarray, conf: float = 0.25) -> dict:
    """Runs YOLO detection on an OpenCV image (BGR) and counts ON/OFF switches."""
    results = model(img, conf=conf, iou=0.5, agnostic_nms=True, verbose=False)[0]

    on_count = 0
    off_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label_name = model.names.get(cls_id, f"class_{cls_id}").lower()
        if "on" in label_name:
            on_count += 1
        else:
            off_count += 1

    total = on_count + off_count
    return {
        "number of switch": total,
        "on": on_count,
        "off": off_count,
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    """Interactive HTML dashboard for testing switchboard image uploads."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Switchboard Counter</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #0f172a; color: #f8fafc; }
            h1 { color: #38bdf8; }
            .card { background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }
            input[type=file] { margin: 16px 0; color: #94a3b8; }
            button { background: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 15px; font-weight: 600; }
            button:hover { background: #0369a1; }
            pre { background: #090d16; padding: 16px; border-radius: 8px; overflow-x: auto; color: #a5f3fc; }
            .badge { display: inline-block; padding: 6px 14px; border-radius: 6px; font-weight: bold; margin-right: 10px; font-size: 18px; }
            .badge-total { background: #334155; color: white; }
            .badge-on { background: #15803d; color: #86efac; }
            .badge-off { background: #b91c1c; color: #fca5a5; }
            #stats { margin: 20px 0; display: none; }
        </style>
    </head>
    <body>
        <h1>⚡ Switchboard State Counter API</h1>
        <div class="card">
            <h3>Upload Switchboard Image</h3>
            <p style="color: #94a3b8;">Select an image to count switches:</p>
            <input type="file" id="fileInput" accept="image/*" />
            <br/>
            <button onclick="predict()">Detect & Count Switches</button>
            <div id="stats">
                <span id="bTotal" class="badge badge-total"></span>
                <span id="bOn" class="badge badge-on"></span>
                <span id="bOff" class="badge badge-off"></span>
            </div>
            <h4>API Response (/predict):</h4>
            <pre id="responseJson">// Response will appear here...</pre>
        </div>
        <script>
            async function predict() {
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files[0]) {
                    alert('Please select an image file first.');
                    return;
                }
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                document.getElementById('responseJson').textContent = 'Analyzing switchboard...';
                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    document.getElementById('responseJson').textContent = JSON.stringify(data, null, 2);
                    
                    document.getElementById('stats').style.display = 'block';
                    document.getElementById('bTotal').textContent = 'Total: ' + (data["number of switch"] ?? data.total_switches);
                    document.getElementById('bOn').textContent = 'ON: ' + (data["on switch"] ?? data.on_switches);
                    document.getElementById('bOff').textContent = 'OFF: ' + (data["off switch"] ?? data.off_switches);
                } catch (err) {
                    document.getElementById('responseJson').textContent = 'Error: ' + err.message;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    image_path: Optional[str] = Query(None),
    conf: float = Query(0.25, description="Confidence threshold (0.0 to 1.0)"),
):
    """
    POST /predict
    Accepts:
    1. Multipart form upload with 'file' or 'files'
    2. Raw binary image data in request body (Content-Type: image/* or application/octet-stream)
    3. JSON body with {"image_path": "path/to/image.jpg"}
    4. Query parameter ?image_path=path/to/image.jpg

    Returns:
    {
        "number of switch": int,
        "on switch": int,
        "off switch": int,
        "number_of_switches": int,
        "on_switches": int,
        "off_switches": int,
        "switches": [
            {"state": "ON"|"OFF", "confidence": float, "bbox": [x1, y1, x2, y2]}
        ]
    }
    """
    # 1. Handle multipart uploads
    upload_list = []
    if files:
        upload_list.extend(files)
    if file:
        upload_list.append(file)

    if upload_list:
        if len(upload_list) == 1:
            f = upload_list[0]
            content = await f.read()
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail=f"Could not decode image '{f.filename}'")
            res = run_inference(img, conf=conf)
            return JSONResponse(content=res)
        else:
            batch = []
            for f in upload_list:
                content = await f.read()
                nparr = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    batch.append({"filename": f.filename, "error": "Could not decode image"})
                else:
                    item = run_inference(img, conf=conf)
                    batch.append(item)
            return JSONResponse(content={"results": batch})

    # 2. Handle image_path query param or JSON payload
    content_type = request.headers.get("content-type", "")
    target_path = image_path

    if "application/json" in content_type:
        try:
            body = await request.json()
            target_path = body.get("image_path") or body.get("path") or target_path
        except Exception:
            pass

    if target_path:
        p = Path(target_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Image file '{target_path}' not found.")
        img = cv2.imread(str(p))
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not decode image at '{target_path}'")
        res = run_inference(img, conf=conf)
        return JSONResponse(content=res)

    # 3. Handle raw binary stream in request body
    raw_body = await request.body()
    if raw_body:
        nparr = np.frombuffer(raw_body, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            res = run_inference(img, conf=conf)
            return JSONResponse(content=res)

    raise HTTPException(
        status_code=400,
        detail="No image provided. Please upload an image via form-data 'file', send raw bytes, or pass 'image_path'.",
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
