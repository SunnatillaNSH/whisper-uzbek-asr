#!/usr/bin/env python3
"""Qo'ng'iroq darajasidagi WER — yorliq DATASETNING O'ZIDAN quriladi.

Nega kerak: `transcripts` jadvalidagi butun-qo'ng'iroq Muxlisa matni ko'p
qo'ng'iroqda yo'q (bo'laklarning ba'zisi yiqilgan bo'lsa, birlashtirilgan
qator 'error' bo'lib qoladi). Lekin datasetdagi har bir bo'lakning MATNI —
aynan o'sha Muxlisa transkripti. Ularni tartib bilan ulasak, butun
qo'ng'iroqning yorlig'i chiqadi.

Shu yo'l bilan WER qamrovi 228 dan 331 qo'ng'iroqqa kengaydi — birorta ham
qo'shimcha API so'rovisiz.
"""
import csv, json, os, re
from collections import defaultdict

OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(OUT)
APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}
def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()
def W(s): return norm(s).split()
def wer(r, h):
    n, m = len(r), len(h)
    if not n: return None
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i]
        for j in range(1, m + 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1])))
        prev = cur
    return prev[m] / n

# Yorliqni bo'laklardan yig'amiz (tartib: part raqami, keyin fayl nomi)
ref = defaultdict(list)
for name in ('calls-dataset-28s',):
    p = f'{ROOT}/data/{name}/manifest.jsonl'
    if not os.path.exists(p): continue
    for line in open(p):
        m = json.loads(line)
        ref[str(m['call_id'])].append(((m.get('part') or 0), m['path'], m['sentence']))
tc = f'{ROOT}/data/calls-colab/train.csv'
if os.path.exists(tc):
    for r in csv.DictReader(open(tc)):
        b = os.path.basename(r['path']).split('.')[0]          # 15286_p1_s2
        parts = b.split('_')
        cid = parts[0]
        pn = int(parts[1][1:]) if len(parts) > 1 and parts[1].startswith('p') else 0
        sn = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith('s') else 0
        ref[cid].append((pn * 100 + sn, r['path'], r['sentence']))

voice = json.load(open('/tmp/voice_by_call.json'))
rows = []
for cid, chunks in ref.items():
    hyp = voice.get(cid)
    if not hyp: continue
    chunks.sort(key=lambda x: x[0])
    r = W(' '.join(c[2] for c in chunks)); h = W(hyp)
    w = wer(r, h)
    if w is None: continue
    rows.append({'call_id': cid, 'wer': round(w, 4), 'ref_words': len(r), 'hyp_words': len(h),
                 'coverage': round(len(h) / len(r), 3), 'chunks': len(chunks)})

rows.sort(key=lambda x: x['call_id'])
with open(f'{OUT}/call_wer.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'WER hisoblandi: {len(rows)} qo\'ng\'iroq')
ws = sorted(x['wer'] for x in rows)
print(f"mediana {ws[len(ws)//2]:.3f} | p10 {ws[len(ws)//10]:.3f} | p90 {ws[len(ws)*9//10]:.3f}")
print(f"o'rtacha qamrov {sum(x['coverage'] for x in rows)/len(rows)*100:.0f}%")
