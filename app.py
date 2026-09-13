#!/usr/bin/env python3
"""
Switchboard State Detection & Counting API Server
Accepts switchboard images and returns:
- number of switch (total switches)
- on switch (number of switches in ON state)
- off switch (number of switches in OFF state)
"""

import io
import json
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# Optional image decoding libraries
try:
    from PIL import Image, ImageOps
except ImportError:
    Image, ImageOps = None, None

try:
    import cv2
except ImportError:
    cv2 = None

# Model paths
ONNX_MODEL_PATH = Path("model/best.onnx")
PT_MODEL_PATH = Path("model/best.pt")

app = FastAPI(
    title="Switchboard Switch Counter API",
    description="Inference API to detect switches and count total, ON, and OFF switches on a switchboard.",
    version="1.0.0",
)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Fast vectorized Non-Maximum Suppression in pure NumPy."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class BaseDetector:
    def detect(self, img, conf: float = 0.25) -> dict:
        raise NotImplementedError


class ONNXDetector(BaseDetector):
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.names = {0: "switch_off", 1: "switch_on"}
        print(f"Loaded ONNX detector from {model_path} (Ultra-lightweight CPU)")

    def _prepare_input(self, img):
        if Image is not None and isinstance(img, Image.Image):
            w0, h0 = img.size
            r = min(640 / w0, 640 / h0)
            new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
            resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            padded = Image.new("RGB", (640, 640), (114, 114, 114))
            dw = (640 - new_w) // 2
            dh = (640 - new_h) // 2
            padded.paste(resized, (dw, dh))

            arr = np.array(padded, dtype=np.float32).transpose((2, 0, 1)) / 255.0
            return np.expand_dims(arr, 0), r, dw, dh, w0, h0

        elif isinstance(img, np.ndarray):
            h0, w0 = img.shape[:2]
            r = min(640 / h0, 640 / w0)
            new_unpad = int(round(w0 * r)), int(round(h0 * r))
            dw, dh = (640 - new_unpad[0]) / 2, (640 - new_unpad[1]) / 2

            if cv2 is not None:
                resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                rgb = padded[:, :, ::-1]
            else:
                pil_img = Image.fromarray(img)
                return self._prepare_input(pil_img)

            arr = rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
            return np.expand_dims(arr, 0), r, dw, dh, w0, h0
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def detect(self, img, conf: float = 0.25) -> dict:
        inp, r, dw, dh, w0, h0 = self._prepare_input(img)
        preds = self.session.run(None, {self.input_name: inp})[0][0].T  # (8400, 6)

        boxes, scores, class_ids = [], [], []
        for row in preds:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_id])
            if score >= conf:
                cx, cy, w, h = row[0:4]
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(score)
                class_ids.append(cls_id)

        on_count, off_count = 0, 0
        switches = []
        if boxes:
            boxes_arr = np.array(boxes)
            scores_arr = np.array(scores)
            keep = nms_boxes(boxes_arr, scores_arr, iou_threshold=0.5)

            for idx in keep:
                x1, y1, x2, y2 = boxes_arr[idx]
                orig_x1 = max(0.0, (x1 - dw) / r)
                orig_y1 = max(0.0, (y1 - dh) / r)
                orig_x2 = min(float(w0), (x2 - dw) / r)
                orig_y2 = min(float(h0), (y2 - dh) / r)

                cls_id = class_ids[idx]
                label_name = self.names.get(cls_id, f"class_{cls_id}").lower()
                is_on = "on" in label_name
                state = "ON" if is_on else "OFF"
                if is_on:
                    on_count += 1
                else:
                    off_count += 1

                switches.append({
                    "state": state,
                    "confidence": round(float(scores_arr[idx]), 4),
                    "bbox": [round(float(orig_x1), 2), round(float(orig_y1), 2), round(float(orig_x2), 2), round(float(orig_y2), 2)],
                })

        total = on_count + off_count
        return {
            "number of switch": total,
            "on": on_count,
            "off": off_count,
            "on switch": on_count,
            "off switch": off_count,
            "number_of_switches": total,
            "on_switches": on_count,
            "off_switches": off_count,
            "switches": switches,
        }


class PyTorchDetector(BaseDetector):
    def __init__(self, model_path: Path):
        from ultralytics import YOLO
        import torch

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path))
        print(f"Loaded PyTorch YOLO model from {model_path} on {self.device}")

    def detect(self, img, conf: float = 0.25) -> dict:
        if Image is not None and isinstance(img, Image.Image):
            img = np.array(img)
            if img.ndim == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1]  # RGB to BGR for OpenCV-based YOLO

        results = self.model(img, conf=conf, iou=0.5, agnostic_nms=True, verbose=False)[0]
        on_count, off_count = 0, 0
        switches = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label_name = self.model.names.get(cls_id, f"class_{cls_id}").lower()
            is_on = "on" in label_name
            state = "ON" if is_on else "OFF"
            if is_on:
                on_count += 1
            else:
                off_count += 1

            xyxy = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]
            switches.append({
                "state": state,
                "confidence": round(confidence, 4),
                "bbox": xyxy,
            })

        total = on_count + off_count
        return {
            "number of switch": total,
            "on": on_count,
            "off": off_count,
            "on switch": on_count,
            "off switch": off_count,
            "number_of_switches": total,
            "on_switches": on_count,
            "off_switches": off_count,
            "switches": switches,
        }


# Initialize detector: Prefer ONNX (ultra-lightweight, fits under 512MB RAM & disk), fallback to PyTorch
detector: BaseDetector
if ONNX_MODEL_PATH.exists():
    try:
        detector = ONNXDetector(ONNX_MODEL_PATH)
    except Exception as e:
        print(f"Notice: ONNX detector initialization skipped ({e}), falling back to PyTorch...")
        if PT_MODEL_PATH.exists():
            detector = PyTorchDetector(PT_MODEL_PATH)
        else:
            raise FileNotFoundError(f"Neither ONNX model ({ONNX_MODEL_PATH}) nor PyTorch model ({PT_MODEL_PATH}) could be loaded.")
elif PT_MODEL_PATH.exists():
    detector = PyTorchDetector(PT_MODEL_PATH)
else:
    raise FileNotFoundError("No trained model weights found in model/ directory.")


def decode_image_bytes(content: bytes):
    """Decodes raw bytes into PIL Image or numpy BGR array."""
    if Image is not None:
        try:
            img = Image.open(io.BytesIO(content))
            return ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            pass
    if cv2 is not None:
        nparr = np.frombuffer(content, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return None


def load_image_path(path: Path):
    """Loads image from file path into PIL Image or numpy array."""
    if Image is not None:
        try:
            img = Image.open(str(path))
            return ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            pass
    if cv2 is not None:
        return cv2.imread(str(path))
    return None


def run_inference(img, conf: float = 0.25) -> dict:
    """Runs detection and returns switch counts and detection details."""
    return detector.detect(img, conf=conf)


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
            img = decode_image_bytes(content)
            if img is None:
                raise HTTPException(status_code=400, detail=f"Could not decode image '{f.filename}'")
            res = run_inference(img, conf=conf)
            return JSONResponse(content=res)
        else:
            batch = []
            for f in upload_list:
                content = await f.read()
                img = decode_image_bytes(content)
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
        img = load_image_path(p)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not decode image at '{target_path}'")
        res = run_inference(img, conf=conf)
        return JSONResponse(content=res)

    # 3. Handle raw binary stream in request body
    raw_body = await request.body()
    if raw_body:
        img = decode_image_bytes(raw_body)
        if img is not None:
            res = run_inference(img, conf=conf)
            return JSONResponse(content=res)

    raise HTTPException(
        status_code=400,
        detail="No image provided. Please upload an image via form-data 'file', send raw bytes, or pass 'image_path'.",
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
