#!/bin/bash
# ==============================================================
# Round 6 — Pod'dagi TO'LIQ zanjir, bitta ajratilgan jarayonda
# ==============================================================
# Round 5 da har bosqich alohida ishga tushirilgandi va oralarda GPU bo'sh
# turdi. Bu yerda hammasi bitta zanjir: tiklash tugashi bilan trening,
# trening tugashi bilan eval boshlanadi.
#
# Zanjir:
#   1. round-2 modelini CT2 ga o'girish (tiklash uchun kerak)
#   2. calls-rejected tiklash -> data/recovered.csv
#   3. trening (FAQAT qo'ng'iroq, UzbekVoice YO'Q)
#   4. yakuniy modelni CT2 ga o'girish
#   5. eval-50 (production dekodlash darvozasi bilan)
#   6. DONE belgisi
#
# SSH uzilsa ham yashashi uchun `setsid` bilan ishga tushiriladi:
#     setsid nohup /root/round6_pod.sh < /dev/null > /dev/null 2>&1 &
# Round 5 da aynan shu qilinmagani uchun trening SSH yopilishi bilan o'lgan.

set -u
OUT=/workspace/round6
mkdir -p "$OUT"
exec >> "$OUT/pipeline.log" 2>&1

REPO=/root/repo
DATA=$REPO/data
RATE=${POD_RATE:-0.49}
BUDGET=${BUDGET_USD:-2.5}
START_EPOCH=${POD_START_EPOCH:-0}     # Pod boshlangan vaqt (Unix), Mac'dan beriladi

step() { echo ""; echo "=== $* — $(date -u '+%H:%M:%S UTC') ==="; }
die()  { echo "TO'XTADI: $*"; exit 1; }

step "Round 6 zanjiri boshlandi"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch,transformers,datasets,peft;print('torch',torch.__version__,'transformers',transformers.__version__,'datasets',datasets.__version__,'peft',peft.__version__)"

# --- Tekshiruvlar: yetishmagan narsa keyin emas, HOZIR to'xtatsin -----------
[ -f "$DATA/round6_calls.csv" ]    || die "$DATA/round6_calls.csv yo'q"
[ -f "$DATA/eval120/eval50.csv" ]  || die "eval50.csv yo'q"
[ -f "$DATA/eval_calls_120.json" ] || die "eval_calls_120.json yo'q (dev-split assertlari usiz ishlamaydi)"
[ -d "$DATA/calls-rejected" ]      || die "calls-rejected yo'q"
cd "$REPO" || die "repo yo'q"
python3 scripts/eval_pod.py --check-only || die "dekodlash handler.py bilan mos emas"

# --- 1) round-2 -> CT2 ------------------------------------------------------
# Modelning O'Z tokenizeri transformers 5.x formatida saqlangan va 4.x uni
# o'qiy olmaydi ("'list' object has no attribute 'keys'"). LoRA faqat
# q_proj/v_proj ga tegadi, ya'ni tokenizer baza model bilan aynan bir xil —
# uni openai/whisper-large-v3 dan olish xavfsiz. Round 5 da bu ikki marta
# yiqildi, shuning uchun bu yerda boshidan to'g'ri qilinadi.
if [ ! -f "$OUT/round2-ct2/model.bin" ]; then
  step "round-2 -> CT2"
  python3 - <<'PY' || exit 1
import os, shutil
from huggingface_hub import snapshot_download
r2 = snapshot_download("Sunnat0091/whisper-large-v3-uz", local_dir="/root/r2")
base = snapshot_download("openai/whisper-large-v3", local_dir="/root/base",
    allow_patterns=["tokenizer.json","tokenizer_config.json","preprocessor_config.json",
                    "vocab.json","merges.txt","special_tokens_map.json",
                    "added_tokens.json","normalizer.json","generation_config.json"])
n = 0
for f in os.listdir(base):
    src = os.path.join(base, f)
    if os.path.isfile(src):                      # .cache katalogini o'tkazamiz
        shutil.copy(src, os.path.join("/root/r2", f)); n += 1
print("tokenizer fayllari almashtirildi:", n)
PY
  ct2-transformers-converter --model /root/r2 --output_dir "$OUT/round2-ct2" \
    --quantization float16 --copy_files tokenizer.json tokenizer_config.json \
    preprocessor_config.json vocab.json merges.txt special_tokens_map.json \
    added_tokens.json normalizer.json || die "round-2 CT2 o'girish yiqildi"
else
  step "round-2 CT2 allaqachon bor — o'tkazildi"
fi

# --- 2) calls-rejected tiklash ---------------------------------------------
# eval-120 qo'ng'iroqlari skriptning O'ZIDA chiqariladi (136 dan 32 tasi
# eval-120 da edi). train_calls.py dagi assert oxirgi to'siq, birinchisi emas.
if [ ! -f "$DATA/recovered.csv" ]; then
  step "calls-rejected tiklash"
  python3 scripts/recover_rejected.py --model "$OUT/round2-ct2" \
    --src data/calls-rejected --out data/calls-recovered \
    --csv data/recovered.csv || die "tiklash yiqildi"
else
  step "recovered.csv allaqachon bor — o'tkazildi"
fi
echo "tiklangan namuna: $(( $(wc -l < "$DATA/recovered.csv") - 1 ))"

# --- 3) Trening — FAQAT QO'NG'IROQ ------------------------------------------
# EXTRA_DIR berilmaydi va USE_PODCAST=0 — Round 5 da UzbekVoice batch'ning
# 64% ini egallab, modelni telefon domenidan uzoqlashtirgan edi (+8.17 punkt).
step "trening"
SETUP_USD=$(python3 -c "
import time
el = (time.time() - $START_EPOCH)/3600 if $START_EPOCH else 0.5
print(f'{el*$RATE + 0.35:.2f}')")
echo "byudjet qo'riqchisi: SETUP_USD=\$$SETUP_USD, chegara \$$BUDGET, stavka \$$RATE/soat"

CALLS_DIR=$DATA \
TRAIN_CSV=round6_calls.csv,recovered.csv \
OUTPUT_DIR=$OUT/out \
MODEL_NAME=Sunnat0091/whisper-large-v3-uz \
USE_PODCAST=0 \
MAX_STEPS=${MAX_STEPS:-2000} EVAL_STEPS=200 SAVE_STEPS=200 WARMUP_STEPS=100 \
BATCH_SIZE=8 GRAD_ACCUM=1 LR=5e-5 PATIENCE=3 \
POD_RATE=$RATE BUDGET_USD=$BUDGET SETUP_USD=$SETUP_USD \
python3 -u train_calls.py || die "trening yiqildi"

[ -d "$OUT/out-final" ] || die "$OUT/out-final yo'q — trening modelni saqlamadi"

# --- 4) Eng yaxshi checkpoint + adapter ------------------------------------
step "adapter ajratish"
python3 - <<PY || exit 1
import json, glob, os, shutil
out = "$OUT/out"
sts = sorted(glob.glob(out + "/checkpoint-*/trainer_state.json"),
             key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
if not sts:
    raise SystemExit("checkpoint topilmadi")
st = json.load(open(sts[-1]))
best = st.get("best_model_checkpoint") or ""
print("eng yaxshi checkpoint:", best, "| dev loss:", st.get("best_metric"))
print("to'xtagan qadam:", st.get("global_step"))
src = best if best and os.path.isdir(best) else os.path.dirname(sts[-1])
dst = "$OUT/adapter"; os.makedirs(dst, exist_ok=True)
n = 0
for f in os.listdir(src):
    if f.startswith("adapter") or f == "README.md":
        shutil.copy(os.path.join(src, f), os.path.join(dst, f)); n += 1
print("adapter fayllari:", n, "manba:", src)
open("$OUT/best.txt", "w").write(f"{best}\n{src}\n{st.get('global_step')}\n{st.get('best_metric')}\n")
PY

# --- 5) Yakuniy model -> CT2 ------------------------------------------------
# Production bilan AYNAN bir xil kvantlash va tokenizer ro'yxati — aks holda
# farq modeldan emas, konvertatsiyadan chiqishi mumkin.
step "Round 6 -> CT2"
rm -rf "$OUT/ct2"
ct2-transformers-converter --model "$OUT/out-final" --output_dir "$OUT/ct2" \
  --quantization float16 --copy_files tokenizer.json tokenizer_config.json \
  preprocessor_config.json vocab.json merges.txt special_tokens_map.json \
  added_tokens.json normalizer.json || die "CT2 o'girish yiqildi"

# --- 6) eval-50 -------------------------------------------------------------
step "eval-50"
python3 scripts/eval_pod.py --model "$OUT/ct2" \
  --csv data/eval120/eval50.csv --out "$OUT/eval50_r6.json" || die "eval yiqildi"

step "ZANJIR TUGADI"
touch "$OUT/PIPELINE_DONE"
