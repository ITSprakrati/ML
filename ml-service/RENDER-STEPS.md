# ML service on Render (FREE, no card needed)

Uses ~250 MB RAM (limit 512 MB) because it runs ONNX only (no PyTorch).

## A. Upload to GitHub (web UI)
1. Open your repo -> **Add file -> Upload files**.
2. Drag in the WHOLE folder `ml-service` (inside it: `models/` with 2 `.part` files, `app.py`, `Dockerfile`, `inference_engine.py`, `requirements.txt`, `model.sha256`). Each file is under 25 MB.
3. **Commit changes**.

## B. Create the service on Render
1. Render dashboard -> **New + -> Web Service** -> pick the repo.
2. Name: `anpr-ml`. **Root Directory: `ml-service`**. Runtime: **Docker**. Instance type: **Free**.
3. Create. First build takes ~5-8 min.
4. Open `https://<your-ml-name>.onrender.com/` -> must show `{"service":"anpr-ml","status":"ONLINE","model_found":true}`.

## C. Plug into backend (public link does not change)
1. Render -> your BACKEND service -> **Environment** -> add `ML_URL` = `https://<your-ml-name>.onrender.com` (no trailing slash).
2. Make sure the patched `server/src/services/aiInferenceService.js` (from `ml-deploy.zip -> for-GitHub-repo`) is pushed to the repo.
3. Save -> it redeploys. Upload an image in the dashboard to test.

## Notes
- Free services sleep after ~15 min idle. First request can take ~30-60 s (backend waits up to 90 s, already set).
- Optional: uptimerobot.com pings every 5 min on `/health` of BOTH services to keep them awake.
- Do not use the old `AI-Model-for-HuggingFace` folder anymore (HF Spaces is paid).
