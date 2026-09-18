# ==============================================================
# RunPod Serverless uchun Docker образ
# ==============================================================
# Qurish (RunPod "Deploy from a GitHub repository" buni o'zi bajaradi):
#   docker build \
#     --build-arg HF_MODEL_ID=<sizning-hf-repo>/whisper-large-v3-uz \
#     -t whisper-uz .
#
# HF_MODEL_ID berilsa, model образ ichiga "pishiriladi" — sovuq start
# tezroq bo'ladi, chunki har safar yuklab olish shart emas.
# Bermasangiz, MODEL_DIR muhit o'zgaruvchisi orqali network volume'dagi
# yo'lni ko'rsatish kerak bo'ladi.

FROM python:3.11-slim

# ffmpeg — m4a/aac/webm kabi formatlarni dekodlash uchun
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch (CUDA 12.4 uchun) — pip wheel'i CUDA kutubxonalarini o'zi olib keladi
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

COPY requirements-runpod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Modelni образga joylash (ixtiyoriy, lekin tavsiya etiladi) ---
ARG HF_MODEL_ID=""
ENV MODEL_DIR=/app/model
RUN if [ -n "$HF_MODEL_ID" ]; then \
        python -c "from huggingface_hub import snapshot_download; \
                   snapshot_download('$HF_MODEL_ID', local_dir='/app/model')" ; \
    else \
        echo "HF_MODEL_ID berilmadi — MODEL_DIR ni network volume'ga yo'naltiring" ; \
    fi

COPY handler.py .

CMD ["python", "-u", "handler.py"]
