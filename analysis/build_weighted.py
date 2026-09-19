#!/usr/bin/env python3
"""4-bosqich: qiyinlik bo'yicha og'irlangan trening dataseti.

Mantiq: model o'z sig'imini allaqachon biladigan narsasiga emas, adashayotgan
joyiga sarflasin ("hard example mining"). Har bir namuna o'z qiyinligiga qarab
1, 2 yoki 3 marta takrorlanadi.

MUHIM: bu bosqich YANGI YORLIQ YARATMAYDI — faqat mavjud Muxlisa yorliqlarini
qayta tartiblaydi. Ya'ni model Muxlisa darajasidan o'tib keta olmaydi, faqat
unga tezroq yaqinlashadi.

Qiyinlik o'lchovi bo'lak darajasida emas, QO'NG'IROQ darajasida hisoblanadi:
Voice AI butun audioni bir butun sifatida o'giradi, Muxlisa esa bo'laklarda —
shuning uchun bo'lakma-bo'lak WER olib bo'lmaydi. Bo'lak ota-qo'ng'irog'ining
WER'ini meros qilib oladi.
"""
import json, csv, os, sys
from statistics import median

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)

# Baholar endi NAMUNA darajasida (sample_wer.py), qo'ng'iroq darajasida emas.
# Eski usulda WER butun qo'ng'iroq matnidan olinardi va bo'laklarga taxminan
# tarqatilardi — qamrov yarmigacha yetmasdi. Endi har bir faylning o'z
# gipotezasi bor, ya'ni bahosi ham o'ziniki.
SAMPLE_WER = f'{OUT}/sample_wer.csv'
if os.path.exists(SAMPLE_WER):
    scores = {r['path']: float(r['wer']) for r in csv.DictReader(open(SAMPLE_WER))}
    BY_PATH = True
else:                                  # zaxira: eski, qo'ng'iroq darajasidagi
    scores = {r['call_key']: float(r['wer']) for r in csv.DictReader(open(f'{OUT}/sample_scores.csv'))}
    BY_PATH = False
wers = sorted(scores.values())
top5 = wers[int(len(wers) * 0.95)] if wers else 1.0

# Eval qo'ng'iroqlarini CHETLAYMIZ — aks holda model o'lchov suhbatini
# treningda ko'radi va WER soxta yaxshi chiqadi.
eval_calls = set()
ev = f'{ROOT}/data/calls-colab/eval.csv'
if os.path.exists(ev):
    for r in csv.DictReader(open(ev)):
        stem = os.path.basename(r['path']).split('.')[0]
        eval_calls.add(stem.split('_')[0])

# FLAC davomiyligini STREAMINFO blokidan o'qiymiz — soundfile/ffmpeg shart emas.
def flac_duration(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'fLaC': return None
        while True:
            h = f.read(4)
            if len(h) < 4: return None
            last, btype, size = h[0] & 0x80, h[0] & 0x7f, int.from_bytes(h[1:4], 'big')
            data = f.read(size)
            if btype == 0:
                bits = int.from_bytes(data[10:18], 'big')
                sr, total = (bits >> 44) & 0xFFFFF, bits & 0xFFFFFFFFF
                return total / sr if sr else None
            if last: return None

def iter_samples():
    """(manba, nisbiy yo'l, call_id, davomiylik, matn) — uch manbadan."""
    # DIQQAT: `calls-dataset` (xom 55 s bo'laklar) BU YERGA KIRMAYDI.
    # Sabab o'lchandi: uning 200 ta qo'ng'irog'ining HAMMASI `calls-colab` da
    # bor (kesishuv 200/200), ya'ni colab — aynan o'sha audioning qayta
    # ishlangan, <=30 s ga bo'lingan nusxasi. Ikkalasini qo'shsak bir xil
    # nutq ikki marta (og'irlik bilan olti martagacha) treningga tushardi.
    # `calls-dataset-28s` bilan kesishuv esa 0 — u boshqa qo'ng'iroqlar.
    for name in ('calls-dataset-28s',):
        path = f'{ROOT}/data/{name}/manifest.jsonl'
        if not os.path.exists(path): continue
        for line in open(path):
            m = json.loads(line)
            yield name, f"{name}/{m['path']}", str(m['call_id']), float(m['duration']), m['sentence'].strip()
    # Oldingi round'larning tayyor to'plami (FLAC, hammasi <=30 s)
    tc = f'{ROOT}/data/calls-colab/train.csv'
    if os.path.exists(tc):
        for r in csv.DictReader(open(tc)):
            ap = f"{ROOT}/data/calls-colab/{r['path']}"
            if not os.path.exists(ap): continue
            cid = os.path.basename(r['path']).split('_')[0].split('.')[0]
            yield 'calls-colab', f"calls-colab/{r['path']}", cid, (flac_duration(ap) or 0), r['sentence'].strip()

rows, excluded = [], []
seen = set()
for name, rel, cid, dur, txt in iter_samples():
    if rel in seen: continue
    seen.add(rel)
    cps = (len(txt) / dur) if dur else 0
    why = None
    if cid in eval_calls:                     why = 'eval qo\'ng\'irog\'i'
    elif dur > 30:                            why = f'uzun ({dur:.1f}s > 30s)'
    elif dur < 1.2:                           why = f'juda qisqa ({dur:.1f}s)'
    elif len(txt) < 4:                        why = 'matn juda kalta'
    elif not (6 <= cps <= 26):                why = f'cps chegaradan tashqari ({cps:.1f})'
    w = scores.get(rel) if BY_PATH else scores.get(f'mz:{cid}')
    if why is None and w is not None and w >= top5:
        why = f'WER eng yuqori 5% ({w:.2f}) — ehtimol buzilgan yorliq'
    if why:
        excluded.append({'path': rel, 'reason': why, 'duration': round(dur, 1), 'cps': round(cps, 1), 'wer': w})
        continue
    # Qiyinroq namuna ko'proq takrorlanadi — model sig'imini adashgan joyiga sarflasin.
    rep = 1 if w is None or w < 0.30 else 3 if w >= 0.60 else 2
    rows.append({'path': rel, 'sentence': txt, 'wer': w, 'duration': round(dur, 1), 'cps': round(cps, 1), 'repeat': rep})

with open(f'{OUT}/train_weighted.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['path', 'sentence'])
    for r in rows:
        for _ in range(r['repeat']): w.writerow([r['path'], r['sentence']])
with open(f'{OUT}/train_scores.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['path', 'wer', 'duration', 'cps', 'repeat']); w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in ['path', 'wer', 'duration', 'cps', 'repeat']})
with open(f'{OUT}/excluded.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['path', 'reason', 'duration', 'cps', 'wer']); w.writeheader(); w.writerows(excluded)

scored = [r for r in rows if r['wer'] is not None]
out = {
    'kirgan_namuna': len(rows),
    'chetlangan': len(excluded),
    'chetlash_sabablari': {k: sum(1 for e in excluded if e['reason'].split(' (')[0].split(' —')[0] == k)
                           for k in {e['reason'].split(' (')[0].split(' —')[0] for e in excluded}},
    'yakuniy_qatorlar': sum(r['repeat'] for r in rows),
    'yakuniy_soat': round(sum(r['duration'] * r['repeat'] for r in rows) / 3600, 2),
    'asl_soat': round(sum(r['duration'] for r in rows) / 3600, 2),
    'WER_bahosi_bor': len(scored),
    'takrorlash': {str(k): sum(1 for r in rows if r['repeat'] == k) for k in (1, 2, 3)},
    'WER_mediana': round(median([r['wer'] for r in scored]), 3) if scored else None,
    'top5_chegara': round(top5, 3),
}
json.dump(out, open(f'{OUT}/dataset_report.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
