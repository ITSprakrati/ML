"""ANPR inference WITHOUT torch/ultralytics: onnxruntime (YOLO detector) + fast-plate-ocr (OCR).
Same JSON output as the original AI-Model/inference_engine.py."""
import os, re, time, json, argparse
import cv2
cv2.setNumThreads(1)
import numpy as np
import onnxruntime as ort
from fast_plate_ocr import LicensePlateRecognizer

IMGSZ = 640

def fix_indian_plate(text):
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    length = len(text)
    if length not in (9, 10):
        return text
    char_to_num = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
    num_to_char = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "4": "A", "6": "G"}
    pattern = (["char", "char", "num", "num", "char", "char", "num", "num", "num", "num"] if length == 10
               else ["char", "char", "num", "char", "char", "num", "num", "num", "num"])
    return "".join(num_to_char.get(c, c) if p == "char" else char_to_num.get(c, c) for c, p in zip(text, pattern))

def evaluate_sharpness(crop):
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    norm = cv2.resize(gray, (128, 64), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(norm, cv2.CV_64F).var())

def enhance_crop(crop):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

_CACHE = {}

def _get_session(model_path):
    if model_path not in _CACHE:
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False  # lower RAM
        _CACHE[model_path] = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    return _CACHE[model_path]

def _get_ocr():
    if "ocr" not in _CACHE:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        _CACHE["ocr"] = LicensePlateRecognizer("cct-xs-v2-global-model", providers=["CPUExecutionProvider"], sess_options=so)
    return _CACHE["ocr"]

def _letterbox(img):
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    out = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top

def _detect(sess, image, min_conf, iou=0.7):
    lb, r, left, top = _letterbox(image)
    blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(blob)})[0][0]  # (4+nc, N)
    preds = out.T
    scores = preds[:, 4:].max(axis=1)
    keep = scores >= min_conf
    preds, scores = preds[keep], scores[keep]
    if len(preds) == 0:
        return []
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes_xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
    idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), float(min_conf), iou)
    res = []
    for i in np.array(idx).flatten():
        x, y, w_, h_ = boxes_xywh[i]
        res.append(((x - left) / r, (y - top) / r, (x + w_ - left) / r, (y + h_ - top) / r, float(scores[i])))
    return res

def run_inference(image_path, model_path, min_conf=0.25, ocr_conf_floor=0.45):
    t0 = time.perf_counter()
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}", "detections": []}
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": "Failed to decode image file", "detections": []}
    h, w, _ = image.shape
    try:
        sess = _get_session(model_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to load model {model_path}: {e}", "detections": []}
    try:
        recognizer = _get_ocr()
    except Exception as e:
        return {"success": False, "error": f"Failed to load OCR recognizer: {e}", "detections": []}

    detections = []
    for fx1, fy1, fx2, fy2, yolo_score in _detect(sess, image, min_conf):
        x1, y1, x2, y2 = max(0, int(fx1)), max(0, int(fy1)), min(w, int(fx2)), min(h, int(fy2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        sharp = evaluate_sharpness(crop)
        preds = recognizer.run(enhance_crop(crop), return_confidence=True)
        if preds:
            pred = preds[0]
            raw = pred.plate.strip()
            prob = float(pred.char_probs.mean()) if getattr(pred, "char_probs", None) is not None else float(yolo_score)
            cleaned = fix_indian_plate(raw)
            if prob >= ocr_conf_floor and len(cleaned) >= 4:
                detections.append({
                    "plateNumber": cleaned, "rawPlate": raw, "confidence": round(prob, 4),
                    "yoloConfidence": round(float(yolo_score), 4), "sharpness": round(sharp, 1),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1}})
    return {"success": True, "imageWidth": w, "imageHeight": h, "detections": detections,
            "detectionCount": len(detections), "inferenceTimeMs": round((time.perf_counter() - t0) * 1000, 2),
            "device": "cpu"}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="models/custom_trained_model.onnx")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--ocr_floor", type=float, default=0.45)
    a = p.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    m = a.model if os.path.isabs(a.model) else os.path.join(base, a.model)
    print(json.dumps(run_inference(os.path.abspath(a.image), m, a.conf, a.ocr_floor), indent=2))
