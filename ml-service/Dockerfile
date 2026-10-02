FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# the model was split in 2 files (GitHub web upload limit is 25 MB); join them back
RUN cat models/custom_trained_model.onnx.part* > models/custom_trained_model.onnx && cd models && sha256sum -c ../model.sha256 && cd .. && rm models/*.part*
# download OCR model at build time so startup is fast
RUN python -c "from fast_plate_ocr import LicensePlateRecognizer; LicensePlateRecognizer('cct-xs-v2-global-model')"
ENV ORT_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
