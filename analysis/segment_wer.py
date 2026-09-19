#!/usr/bin/env python3
"""Segment darajasidagi WER — colab namunalari uchun ham.

MUAMMO: Voice AI BUTUN qo'ng'iroqni bitta matn qilib beradi, dataset esa
bo'laklardan iborat. `calls-dataset-28s` da bo'laklar qo'ng'iroqni TO'LIQ
qoplaydi, shuning uchun ularni ulab taqqoslash mumkin. `calls-colab` da esa
train.csv har bir qo'ng'iroqning faqat BIR QISMINI saqlagan (oldingi
round'larda filtrlangan) — ulab taqqoslasak yorliq gipotezadan 6 barobar
qisqa bo'lib, WER ma'nosiz chiqadi (o'lchandi: qamrov 661%).

YECHIM: har bir segment uchun gipoteza ichidan ENG MOS oynani topamiz.
Segmentlar ketma-ket bo'lgani uchun qidiruv monoton olib boriladi (oldinga
siljiydigan ko'rsatkich) — bu tasodifiy mos kelishning oldini oladi.

Sifat darvozasi: mos kelgan so'zlar ulushi 25% dan past bo'lsa, tekislash
ishonchsiz deb hisoblanadi va WER berilmaydi (jim noto'g'ri baho berishdan
ko'ra "bilmayman" deyish afzal).
"""
import csv, json, os, re
from difflib import SequenceMatcher
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

def segments():
    """(call_id, tartib, nisbiy_yo'l, matn) — ikkala manbadan."""
    p = f'{ROOT}/data/calls-dataset-28s/manifest.jsonl'
    if os.path.exists(p):
        for line in open(p):
            m = json.loads(line)
            yield str(m['call_id']), (m.get('part') or 1) * 100, f"calls-dataset-28s/{m['path']}", m['sentence']
    tc = f'{ROOT}/data/calls-colab/train.csv'
    if os.path.exists(tc):
        for r in csv.DictReader(open(tc)):
            b = os.path.basename(r['path']).split('.')[0]
            q = b.split('_')
            pn = int(q[1][1:]) if len(q) > 1 and q[1].startswith('p') else 1
            sn = int(q[2][1:]) if len(q) > 2 and q[2].startswith('s') else 0
            yield q[0], pn * 100 + sn, f"calls-colab/{r['path']}", r['sentence']

by_call = defaultdict(list)
for cid, order, path, sent in segments():
    by_call[cid].append((order, path, sent))

voice = json.load(open('/tmp/voice_by_call.json'))
rows, skipped = [], 0
for cid, segs in by_call.items():
    hyp = voice.get(cid)
    if not hyp:
        skipped += len(segs); continue
    H = W(hyp); segs.sort(key=lambda x: x[0]); pos = 0
    for order, path, sent in segs:
        R = W(sent)
        if len(R) < 3:
            rows.append({'path': path, 'call_id': cid, 'wer': '', 'note': 'juda qisqa'}); continue
        # Oyna: `pos` dan OXIRIGACHA. Avval segment uzunligining 3 barobari
        # bilan cheklagandim — colab segmentlari qo'ng'iroqning SIYRAK
        # qismlari bo'lgani uchun (oralarida katta bo'shliqlar bor) bu juda
        # tor chiqdi va 35% tekislash ishonchsiz deb belgilandi. Monotonlik
        # `pos` ning oldinga siljishi bilan saqlanadi.
        win = H[pos:]
        sm = SequenceMatcher(None, R, win, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        matched = sum(b.size for b in blocks)
        if not blocks or matched / len(R) < 0.25:
            rows.append({'path': path, 'call_id': cid, 'wer': '', 'note': 'ishonchsiz tekislash'})
            continue
        s0 = blocks[0].b; s1 = blocks[-1].b + blocks[-1].size
        span = win[max(s0 - 2, 0): s1 + 2]
        rows.append({'path': path, 'call_id': cid, 'wer': round(wer(R, span), 4),
                     'note': '', 'ref_words': len(R), 'hyp_words': len(span)})
        pos += max(s1 - 1, 1)

with open(f'{OUT}/segment_wer.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['path', 'call_id', 'wer', 'note', 'ref_words', 'hyp_words'])
    w.writeheader()
    for r in rows: w.writerow({k: r.get(k, '') for k in w.fieldnames})

ok = [r for r in rows if r['wer'] != '']
ws = sorted(float(r['wer']) for r in ok)
print(f"segment: {len(rows)} | WER chiqdi: {len(ok)} | ishonchsiz: {len(rows)-len(ok)} | Voice AI yo'q: {skipped}")
if ws:
    print(f"mediana {ws[len(ws)//2]:.3f} | p10 {ws[len(ws)//10]:.3f} | p90 {ws[len(ws)*9//10]:.3f}")
