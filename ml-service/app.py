"""Small web wrapper around inference_engine.py so the backend can call ML over HTTP."""
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import inference_engine

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE, "models", "custom_trained_model.onnx"))

app = FastAPI(title="ANPR ML Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def warm_up():
    # load both models once so the first real request is fast
    try:
        inference_engine._get_session(MODEL_PATH)
        inference_engine._get_ocr()
        print("[ML] models loaded")
    except Exception as e:  # keep the server up so /health still answers
        print("[ML] warm-up failed:", e)


@app.get("/")
def root():
    return {"service": "anpr-ml", "status": "ONLINE", "model_found": os.path.exists(MODEL_PATH)}


@app.get("/health")
def health():
    return {"status": "ONLINE"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    ocr_floor: float = Form(0.45),
):
    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        path = tmp.name
    try:
        return inference_engine.run_inference(path, MODEL_PATH, conf, ocr_floor)
    finally:
        os.remove(path)
