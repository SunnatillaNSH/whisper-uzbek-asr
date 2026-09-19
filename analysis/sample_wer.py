#!/usr/bin/env python3
"""Har bir NAMUNA uchun WER — endpointdan olingan aniq gipoteza bilan.

Nega bu skript kerak: avval WER qo'ng'iroq darajasida, bazadagi butun-qo'ng'iroq
matnidan hisoblangandi. Dataset esa BO'LAKLARDAN iborat — butun qo'ng'iroq
matnini bo'lakka taqqoslab bo'lmaydi. Natijada qamrov 1394 dan 751 taga
cheklangan edi.

Endi har bir dataset fayli endpointga ALOHIDA yuborilgan (run_all.py,
`hotwords: ""` bilan) — ya'ni har bir namunaning o'z gipotezasi bor va WER
aniq hisoblanadi.

Ishlatish:  python3 analysis/sample_wer.py <all_out.json>
"""
import csv, json, re, os, sys
from statistics import median

OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(OUT)
APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}
def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()
def W(s): return norm(s).split()
def ops(r, h):
    n, m = len(r), len(h)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(r[i-1] != h[j-1]))
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + (0 if r[i-1] == h[j-1] else 1):
            S += r[i-1] != h[j-1]; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j]+1: D += 1; i -= 1
        else: I += 1; j -= 1
    return S, D, I

# Yorliqlar (Muxlisa matni) — ikkala manbadan
ref = {}
for r in csv.DictReader(open(f'{ROOT}/data/calls-colab/train.csv')):
    ref[f"calls-colab/{r['path']}"] = r['sentence']
for line in open(f'{ROOT}/data/calls-dataset-28s/manifest.jsonl'):
    m = json.loads(line)
    ref[f"calls-dataset-28s/{m['path']}"] = m['sentence']

hyp = json.load(open(sys.argv[1]))
rows, empty = [], 0
for k, h in hyp.items():
    if k not in ref: continue
    r = W(ref[k])
    if not r: continue
    if not h: empty += 1; continue
    y = W(h)
    S, D, I = ops(r, y)
    rows.append({'path': k, 'wer': round((S+D+I)/len(r), 4), 'ref_words': len(r),
                 'hyp_words': len(y), 'sub': S, 'del': D, 'ins': I})

rows.sort(key=lambda x: x['path'])
with open(f'{OUT}/sample_wer.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

ws = sorted(x['wer'] for x in rows)
S = sum(x['sub'] for x in rows); D = sum(x['del'] for x in rows)
I = sum(x['ins'] for x in rows); N = sum(x['ref_words'] for x in rows)
rep = {'namuna': len(rows), 'yorliqda_bor': len(ref), 'bo\'sh_javob': empty,
       'qamrov_%': round(len(rows)/len(ref)*100, 1),
       'korpus_WER': round((S+D+I)/N, 4),
       'mediana': round(median(ws), 4), 'p10': round(ws[len(ws)//10], 4), 'p90': round(ws[len(ws)*9//10], 4),
       'sub': S, 'del': D, 'ins': I,
       'del_ulushi_%': round(D/(S+D+I)*100, 1),
       'so\'z_qamrovi_%': round(sum(x['hyp_words'] for x in rows)/N*100, 1)}
json.dump(rep, open(f'{OUT}/sample_wer_report.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
