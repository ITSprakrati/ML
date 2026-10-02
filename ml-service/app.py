"""Small web wrapper around inference_engine.py so the backend can call ML over HTTP."""
import os

# Free hosting gives a tiny CPU slice but the machine reports MANY cores: force single-threaded math
# BEFORE numpy / onnxruntime / opencv are imported.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ORT_THREADS"):
    os.environ.setdefault(_k, "1")

import tempfile
import threading
import time
import traceback

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import inference_engine

cv2.setNumThreads(1)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE, "models", "custom_trained_model.onnx"))
SELFTEST_IMG = os.path.join(BASE, "selftest.jpg")
_INFER_LOCK = threading.Lock()  # one frame at a time keeps RAM low on the 512 MB free plan

app = FastAPI(title="ANPR ML Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def mem_info():
    """Memory of this process and (if readable) of the whole container, in MB."""
    out = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":")
                out[k.strip()] = round(int(v.split()[0]) / 1024)
    except Exception:
        pass
    for cur, mx in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                    ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            out["container_used_mb"] = round(int(open(cur).read()) / 1048576)
            lim = open(mx).read().strip()
            out["container_limit_mb"] = round(int(lim) / 1048576) if lim.isdigit() and int(lim) < 1 << 50 else "none"
            break
        except Exception:
            continue
    return out


@app.on_event("startup")
def warm_up():
    # load both models once so the first real request is fast
    try:
        inference_engine._get_session(MODEL_PATH)
        inference_engine._get_ocr()
        print("[ML] models loaded", mem_info(), flush=True)
    except Exception as e:  # keep the server up so /health still answers
        print("[ML] warm-up failed:", e, flush=True)


@app.get("/")
def root():
    return {"service": "anpr-ml", "status": "ONLINE", "model_found": os.path.exists(MODEL_PATH)}


@app.get("/health")
def health():
    return {"status": "ONLINE"}


@app.get("/selftest")
def selftest():
    """Open this in a browser: runs one sample image through the whole pipeline and reports time + memory."""
    t0 = time.perf_counter()
    try:
        with _INFER_LOCK:
            r = inference_engine.run_inference(SELFTEST_IMG, MODEL_PATH, 0.25, 0.45)
        return {"ok": bool(r.get("success")), "plates": [d["plateNumber"] for d in r.get("detections", [])],
                "expected": "UP16CD5677", "seconds": round(time.perf_counter() - t0, 2), "memory_mb": mem_info(),
                "error": r.get("error")}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"ok": False, "error": repr(e), "memory_mb": mem_info()})


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    ocr_floor: float = Form(0.45),
):
    # Plain "def" (not "async def"): FastAPI runs it in a worker thread, so a slow image
    # never freezes the server and /health keeps answering while a frame is processed.
    t0 = time.perf_counter()
    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image.file.read())
        path = tmp.name
    try:
        with _INFER_LOCK:
            result = inference_engine.run_inference(path, MODEL_PATH, conf, ocr_floor)
        print(f"[ML] predict ok={result.get('success')} plates={result.get('detectionCount')} "
              f"{round(time.perf_counter() - t0, 2)}s mem={mem_info()}", flush=True)
        return result
    except Exception as e:  # never crash the worker: report the real reason to the caller and the logs
        traceback.print_exc()
        print("[ML] predict FAILED", repr(e), mem_info(), flush=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"ML error: {e!r}", "detections": []})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
