#!/bin/bash
# Round 6 zanjirini Pod'da AJRATILGAN holda boshlaydi.
#   ./scripts/round6_start.sh <host> <port>
#
# `setsid` + `< /dev/null` majburiy: Round 5 da trening SSH yopilishi bilan
# o'lgan, chunki fon jarayoni stdin'ni ushlab turgan va SIGHUP olgan.
set -eu
HOST=${1:?host kerak}; PORT=${2:?port kerak}
KEY=$HOME/.ssh/id_ed25519
SSH="ssh -o BatchMode=yes -i $KEY -p $PORT root@$HOST"
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Pod boshlangan vaqt Unix epoch'da beriladi — Round 5 da Pod soati UTC,
# Mac'niki mahalliy bo'lgani uchun byudjet bazasi MANFIY chiqqandi va
# qo'riqchi ishlamay qolgandi. Epoch'da vaqt mintaqasi muammosi yo'q.
START=${POD_START_EPOCH:-$(date +%s)}

scp -q -o BatchMode=yes -i "$KEY" -P "$PORT" "$ROOT/scripts/round6_pod.sh" "root@$HOST:/root/round6_pod.sh"
$SSH "chmod +x /root/round6_pod.sh && \
  POD_START_EPOCH=$START POD_RATE=${POD_RATE:-0.49} BUDGET_USD=${BUDGET_USD:-2.5} \
  MAX_STEPS=${MAX_STEPS:-2000} \
  setsid nohup /root/round6_pod.sh < /dev/null > /dev/null 2>&1 & disown; sleep 3; \
  echo 'jarayonlar:'; pgrep -fc round6_pod || true"
echo "boshlandi. Kuzatish:"
echo "  ssh -i $KEY -p $PORT root@$HOST 'tail -f /workspace/round6/pipeline.log'"
