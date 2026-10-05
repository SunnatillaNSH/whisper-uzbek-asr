#!/bin/bash
# TZ-008: Round 6 Pod'ida, trening TUGAGANDAN KEYIN (PIPELINE_DONE bor), Pod O'CHIRILMASDAN OLDIN.
#   ./scripts/round6_diar_test.sh <host> <port>
# Talab: ~/.config/sellup/hf_token (egasi pyannote shartlariga HF'da rozilik bergan hisobniki).
# Qiymat ekranga CHIQMAYDI. Natija: data/round6-diar/out/ (gitignore'da; matn bor — commit QILINMAYDI).
set -eu
HOST=${1:?host kerak}; PORT=${2:?port kerak}
KEY=$HOME/.ssh/id_ed25519
SSH="ssh -o BatchMode=yes -i $KEY -p $PORT root@$HOST"
SCP="scp -q -o BatchMode=yes -i $KEY -P $PORT"
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
TOKF=$HOME/.config/sellup/hf_token
[ -s "$TOKF" ] || { echo "HF token fayli yo'q: $TOKF (egasi bergach PM yaratadi)"; exit 1; }
$SSH 'test -f /workspace/round6/PIPELINE_DONE' || { echo "round6 hali tugamagan (PIPELINE_DONE yo'q) — kutiladi"; exit 1; }

echo "=== 1/5 yig'ish (Mac) ==="
[ -f data/round6-diar/calls.json ] || python3 scripts/diar_test.py prep
python3 -c "import json;c=json.load(open('data/round6-diar/calls.json'));print(len(c),'qo\'ng\'iroq')"

echo "=== 2/5 yuborish ==="
$SSH 'mkdir -p /workspace/diar/other /root/repo/scripts'
tar -C data/round6-diar -cf - audio calls.json other | $SSH 'tar -C /workspace/diar -xf -'
$SCP scripts/diar_test.py "root@$HOST:/root/repo/scripts/diar_test.py"
$SSH 'umask 077; cat > /root/.hf_token' < "$TOKF"

echo "=== 3/5 pyannote o'rnatish (torch versiyasi o'zgarmasligi tekshiriladi) ==="
$SSH 'pip install --break-system-packages --no-cache-dir -q "pyannote.audio>=3.1,<4" 2>&1 | grep -v "root user" | tail -3; python3 -c "import torch,pyannote.audio as p;print(\"torch\",torch.__version__,\"pyannote\",p.__version__)"; pip list 2>/dev/null | grep -i "^ctranslate2\|^faster-whisper"'

echo "=== 4/5 sinov (fonda, setsid) ==="
$SSH 'cd /root/repo && setsid nohup python3 -u scripts/diar_test.py run > /workspace/diar/run.log 2>&1 < /dev/null & disown; sleep 2; echo boshlandi'
while $SSH 'pgrep -f "diar_test.py run" >/dev/null'; do sleep 20; done
$SSH 'tail -n 40 /workspace/diar/run.log | head -n 40' | head -n 40

echo "=== 5/5 natijani olish va tokenni o'chirish ==="
$SSH 'rm -f /root/.hf_token'
mkdir -p data/round6-diar/out
$SSH 'tar -C /workspace/diar/out -cf - .' | tar -C data/round6-diar/out -xf -
echo "tayyor: data/round6-diar/out/summary.json va segments/*.json. Pod'ni PM o'chiradi."
