# ==============================================================
# RunPod Serverless образi — faster-whisper (CTranslate2)
# ==============================================================
# Eski Dockerfile (transformers + torch) dan farqi: torch UMUMAN yo'q.
# CTranslate2 o'z dvigateliga ega va faqat CUDA/cuDNN kutubxonalarini
# talab qiladi. Natijada образ ~8 GB dan ~2 GB ga tushadi va sovuq start
# sezilarli qisqaradi — serverless uchun bu to'g'ridan-to'g'ri pul.
#
# Model CT2 formatida bo'lishi SHART. Konvertatsiya bir marta qilinadi:
#   ct2-transformers-converter --model <hf-repo-yoki-papka> \
#       --output_dir whisper-uz-ct2 --quantization float16 \
#       --copy_files tokenizer.json tokenizer_config.json \
#                    preprocessor_config.json vocab.json merges.txt \
#                    special_tokens_map.json added_tokens.json normalizer.json
# so'ng natija HF'ga yuklanadi va quyida HF_MODEL_ID sifatida ko'rsatiladi.

FROM python:3.11-slim

# ffmpeg — m4a/aac/webm kabi formatlarni dekodlash uchun (zaxira yo'l)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-runpod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CTranslate2 CUDA uchun cuBLAS va cuDNN kerak. Ular pip paketlari ichida
# keladi, lekin dinamik yuklovchi ularni o'zi topmaydi — yo'lni ko'rsatamiz.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

# --- Model образga PISHIRILMAYDI ---
# Repo maxfiy (real mijoz suhbatlarida o'qitilgan model), RunPod esa qurish
# paytida token bera olmaydi. Shuning uchun handler uni ishga tushganda
# yuklab oladi. RunPod endpoint sozlamalarida quyidagilar berilishi kerak:
#
#   HF_MODEL_ID = Sunnat0091/whisper-large-v3-uz-calls-ct2
#   HF_TOKEN    = hf_...   (Read huquqi yetarli)
#
# Natijada образ ~1 GB bo'lib qoladi va model almashtirilganda образni
# qayta qurish shart emas — faqat o'zgaruvchini o'zgartirish kifoya.
ENV MODEL_DIR=/app/model

COPY handler.py .

CMD ["python", "-u", "handler.py"]
