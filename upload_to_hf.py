#!/usr/bin/env python3
"""Fine-tune qilingan modelni Hugging Face Hub'ga yuklash.

    export HF_TOKEN=hf_...
    MODEL_DIR=/workspace/whisper-large-v3-uz HF_REPO=Sunnat0091/whisper-large-v3-uz-v2 \
        python upload_to_hf.py
"""

import os
import sys

from huggingface_hub import HfApi, login

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/whisper-large-v3-uz")
HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    sys.exit("HF_TOKEN o'rnatilmagan (https://huggingface.co/settings/tokens, Write)")
if not HF_REPO:
    sys.exit("HF_REPO o'rnatilmagan, masalan: HF_REPO=foydalanuvchi/whisper-large-v3-uz-v2")
if not os.path.isdir(MODEL_DIR):
    sys.exit(f"Topilmadi: {MODEL_DIR}")

login(token=HF_TOKEN)
api = HfApi()
api.create_repo(HF_REPO, private=False, exist_ok=True)
api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=HF_REPO,
    ignore_patterns=["checkpoint-*", "runs/*", "optimizer.pt", "scheduler.pt"],
)
print(f"✅ Yuklandi: https://huggingface.co/{HF_REPO}")
