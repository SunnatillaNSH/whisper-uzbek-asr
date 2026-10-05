#!/bin/bash
# ==============================================================
# Round 6 (v2, Jev A-004) — Pod'dagi TO'LIQ zanjir, bitta ajratilgan jarayonda
# ==============================================================
# Dataset: data/round6-dataset (train.csv 2296 + dev.csv 266, dev MIJOZ bo'yicha
# ajratilgan, Claude tahrirlagan namunalar olib tashlangan). Tiklash bosqichi YO'Q
# (chetlanganlar 53 ta, kerak emas).
#
# Zanjir:
#   1. tekshiruvlar (fayllar, dekodlash handler.py bilan mos)
#   2. trening: MAX_STEPS=3000 cap, eval/save 250, patience 3, load_best_model_at_end
#      (dev loss), byudjet qo'riqchisi $3.0
#   3. eng yaxshi checkpoint adapteri
#   4. yakuniy modelni CT2 ga o'girish
#   5. eval-120 (310 namuna; eval-50 uning ICHIDA — 129 namuna, alohida yurgizilmaydi)
#   6. DONE belgisi
#
# SSH uzilsa ham yashashi uchun `setsid` bilan ishga tushiriladi (round6_start.sh):
#     setsid nohup /root/round6_pod.sh < /dev/null > /dev/null 2>&1 &

set -u
OUT=/workspace/round6
mkdir -p "$OUT"
exec >> "$OUT/pipeline.log" 2>&1

REPO=/root/repo
DATA=$REPO/data
RATE=${POD_RATE:-1.10}
BUDGET=${BUDGET_USD:-3.0}
START_EPOCH=${POD_START_EPOCH:-0}     # Pod boshlangan vaqt (Unix), Mac'dan beriladi

step() { echo ""; echo "=== $* — $(date -u '+%H:%M:%S UTC') ==="; }
die()  { echo "TO'XTADI: $*"; exit 1; }

step "Round 6 zanjiri boshlandi"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch,transformers,datasets,peft;print('torch',torch.__version__,'transformers',transformers.__version__,'datasets',datasets.__version__,'peft',peft.__version__)"

# --- Tekshiruvlar: yetishmagan narsa keyin emas, HOZIR to'xtatsin -----------
RD=$DATA/round6-dataset
[ -f "$RD/train.csv" ]             || die "$RD/train.csv yo'q"
[ -f "$RD/dev.csv" ]               || die "$RD/dev.csv yo'q"
[ -f "$RD/manifest.jsonl" ]        || die "manifest.jsonl yo'q (lead assertlari usiz ishlamaydi)"
[ -f "$RD/eval_calls_120.json" ]   || die "eval_calls_120.json yo'q (dataset katalogida)"
[ -f "$DATA/eval120/eval.csv" ]    || die "eval120/eval.csv yo'q"
[ -d "$RD/audio" ]                 || die "round6-dataset/audio yo'q"
echo "train: $(( $(wc -l < "$RD/train.csv") - 1 )) | dev: $(( $(wc -l < "$RD/dev.csv") - 1 )) | audio: $(ls "$RD/audio" | wc -l)"
cd "$REPO" || die "repo yo'q"
python3 scripts/eval_pod.py --check-only || die "dekodlash handler.py bilan mos emas"

# --- 2) Trening — FAQAT QO'NG'IROQ ------------------------------------------
# EXTRA_DIR berilmaydi va USE_PODCAST=0 — Round 5 da UzbekVoice batch'ning
# 64% ini egallab, modelni telefon domenidan uzoqlashtirgan edi (+8.17 punkt).
step "trening"
SETUP_USD=$(python3 -c "
import time
el = (time.time() - $START_EPOCH)/3600 if $START_EPOCH else 0.5
print(f'{el*$RATE + 0.50:.2f}')   # o'tgan sozlash + CT2/eval-120 uchun 0.50 zaxira")
echo "byudjet qo'riqchisi: SETUP_USD=\$$SETUP_USD, chegara \$$BUDGET, stavka \$$RATE/soat"

CALLS_DIR=$RD \
TRAIN_CSV=train.csv DEV_CSV=dev.csv MANIFEST=manifest.jsonl \
OUTPUT_DIR=$OUT/out \
MODEL_NAME=Sunnat0091/whisper-large-v3-uz \
USE_PODCAST=0 \
MAX_STEPS=${MAX_STEPS:-3000} EVAL_STEPS=250 SAVE_STEPS=250 WARMUP_STEPS=100 \
BATCH_SIZE=8 GRAD_ACCUM=1 LR=5e-5 PATIENCE=3 \
POD_RATE=$RATE BUDGET_USD=$BUDGET SETUP_USD=$SETUP_USD \
python3 -u train_calls.py || die "trening yiqildi"

[ -d "$OUT/out-final" ] || die "$OUT/out-final yo'q — trening modelni saqlamadi"

# --- 3) Eng yaxshi checkpoint + adapter ------------------------------------
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

# --- 4) Yakuniy model -> CT2 ------------------------------------------------
# Production bilan AYNAN bir xil kvantlash va tokenizer ro'yxati — aks holda
# farq modeldan emas, konvertatsiyadan chiqishi mumkin.
step "Round 6 -> CT2"
rm -rf "$OUT/ct2"
ct2-transformers-converter --model "$OUT/out-final" --output_dir "$OUT/ct2" \
  --quantization float16 --copy_files tokenizer.json tokenizer_config.json \
  preprocessor_config.json vocab.json merges.txt special_tokens_map.json \
  added_tokens.json normalizer.json || die "CT2 o'girish yiqildi"

# --- 5) eval-120 -------------------------------------------------------------
step "eval-120 (eval-50 uning ichida)"
python3 scripts/eval_pod.py --model "$OUT/ct2" \
  --csv data/eval120/eval.csv --out "$OUT/eval120_r6.json" || die "eval yiqildi"

step "ZANJIR TUGADI"
touch "$OUT/PIPELINE_DONE"
