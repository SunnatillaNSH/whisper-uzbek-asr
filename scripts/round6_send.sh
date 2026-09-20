#!/bin/bash
# Round 6 — Pod'ni tayyorlaydi va ma'lumotni ko'chiradi (Mac'dan yuriladi).
#   ./scripts/round6_send.sh <host> <port>
set -eu
HOST=${1:?host kerak}; PORT=${2:?port kerak}
KEY=$HOME/.ssh/id_ed25519
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $KEY -p $PORT root@$HOST"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo "=== 1/4 ulanish ==="
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python3 -V; df -h / | tail -1'

echo "=== 2/4 repo + paketlar (fonda) ==="
# PEP 668: Ubuntu 24.04 obrazida --break-system-packages majburiy.
# torchao ATAYIB olib tashlanadi: torch 2.8 bilan mos kelmay
# "cannot import name 'ScalingType'" beradi.
$SSH 'rm -rf /root/repo && git clone -q --depth 1 https://github.com/SunnatillaNSH/whisper-uzbek-asr.git /root/repo && cd /root/repo && git log --oneline -1 && \
  (pip install --break-system-packages --no-cache-dir -q -r requirements-train.txt faster-whisper ctranslate2 2>&1 | grep -v "root user" | tail -3; \
   pip uninstall -y --break-system-packages -q torchao 2>&1 | tail -1) > /root/pip.log 2>&1 &
  echo "pip fonda boshlandi"'

echo "=== 3/4 ma'lumot ro'yxati ==="
# Faqat KERAKLI fayllar. Round 5 da butun katalog yuborilgan va 600 MB
# ortiqcha ketgandi (jumladan calls-colab.tar — audioning ikkinchi nusxasi).
python3 - <<'PY'
import csv, json, os
need = set()
for r in csv.DictReader(open('analysis/train_weighted.csv', encoding='utf-8')):
    need.add(r['path'])
for r in csv.DictReader(open('data/eval120/eval50.csv', encoding='utf-8')):
    need.add(r['path'])
rej = [json.loads(l) for l in open('data/calls-rejected/rejected.jsonl')]
need |= {'calls-rejected/' + r['path'] for r in rej}
files = sorted('data/' + p for p in need)
missing = [f for f in files if not os.path.exists(f)]
if missing:
    raise SystemExit(f"{len(missing)} fayl yo'q, masalan {missing[:2]}")
open('/tmp/round6_files.txt', 'w').write("\n".join(files) + "\n")
mb = sum(os.path.getsize(f) for f in files) / 1e6
print(f"fayllar: {len(files)} | hajm: {mb:.0f} MB")
PY

echo "=== 4/4 ko'chirish ==="
t0=$(date +%s)
tar cf - -T /tmp/round6_files.txt \
    data/eval120/eval50.csv data/eval_calls_120.json data/eval_calls_50.json \
    data/calls-rejected/rejected.jsonl analysis/train_weighted.csv \
  | $SSH 'cd /root/repo && tar xf - 2>/dev/null; cp analysis/train_weighted.csv data/round6_calls.csv; du -sh data; find data -type f | wc -l'
echo "ko'chirish: $(( $(date +%s) - t0 )) s"

echo "=== paketlar holati ==="
$SSH 'cat /root/pip.log 2>/dev/null | tail -3; python3 -c "
import transformers, datasets, peft, torch, faster_whisper, importlib.util as u
print(transformers.__version__, datasets.__version__, peft.__version__, torch.__version__)
print(\"torchao bormi:\", u.find_spec(\"torchao\") is not None)
"'
echo ""
echo "TAYYOR. Zanjirni boshlash:"
echo "  ./scripts/round6_start.sh $HOST $PORT"
