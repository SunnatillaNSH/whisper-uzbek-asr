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
#
# Zaxira yo'li ATAYLAB SHOVQINLI. Bir marta `sample_wer.csv` umuman
# yaratilmadi (uni to'ldirishi kerak bo'lgan yurishning natijalari yo'qolgan)
# va skript buni sezmay eski `sample_scores.csv` ga qaytdi: qo'ng'iroq
# darajasidagi, VAD va hotwords tuzatilishidan OLDINGI baholar, 975
# namunadan 546 tasini qamraydigan. Dataset tayyor ko'rinardi, og'irliklari
# esa boshqa modelning xatolaridan edi. Endi ogohlantirish beradi va
# BALLAR_ZAXIRASI=1 qo'yilmasa to'xtaydi.
#
# OGIRLIKSIZ=1 — og'irlash umuman qilinmaydi. Round 5 shunday o'qitiladi:
# og'irlashning foydasi hech qachon alohida o'lchanmagan, u gipoteza, va
# hozirgi ballar qamrovi baribir yetarli emas. Bu rejimda ballar KERAK
# EMAS, shuning uchun ularning yo'qligi ham xato emas.
OGIRLIKSIZ = os.environ.get('OGIRLIKSIZ') == '1'
SAMPLE_WER = f'{OUT}/sample_wer.csv'
if OGIRLIKSIZ and not os.path.exists(SAMPLE_WER):
    # Ballar yo'q — og'irlash ham, buzuq yorliqlarni chetlash ham bo'lmaydi.
    print("Og'irliksiz rejim, ballarsiz", file=sys.stderr)
    scores, BY_PATH = {}, True
elif OGIRLIKSIZ:
    # Ballar bor: og'irlash qilinmaydi, lekin eng yuqori 5% WER (ehtimol
    # buzilgan yorliq) chetlashi SAQLANADI — u faqat chiqaradi, qamrovi
    # to'liq bo'lmasa ham zarar qilmaydi.
    print("Og'irliksiz rejim — ballar faqat chetlash uchun", file=sys.stderr)
    scores = {r['path']: float(r['wer']) for r in csv.DictReader(open(SAMPLE_WER))}
    BY_PATH = True
elif os.path.exists(SAMPLE_WER):
    scores = {r['path']: float(r['wer']) for r in csv.DictReader(open(SAMPLE_WER))}
    BY_PATH = True
else:
    msg = (f"{SAMPLE_WER} yo'q — og'irliklar eski, QO'NG'IROQ darajasidagi "
           f"sample_scores.csv dan olinadi (boshqa sozlamalarda o'lchangan).")
    if os.environ.get('BALLAR_ZAXIRASI') != '1':
        sys.exit(f"{msg}\nAtayin shuni xohlasangiz: BALLAR_ZAXIRASI=1 python3 {__file__}\n"
                 f"Og'irliksiz o'qitish uchun: OGIRLIKSIZ=1 python3 {__file__}")
    print(f"OGOHLANTIRISH: {msg}", file=sys.stderr)
    scores = {r['call_key']: float(r['wer']) for r in csv.DictReader(open(f'{OUT}/sample_scores.csv'))}
    BY_PATH = False
wers = sorted(scores.values())
top5 = wers[int(len(wers) * 0.95)] if wers else float('inf')

# Eval qo'ng'iroqlarini CHETLAYMIZ — aks holda model o'lchov suhbatini
# treningda ko'radi va WER soxta yaxshi chiqadi.
#
# Chetlash QO'NG'IROQ darajasida: bitta suhbatning qo'shni bo'laklari bir xil
# ovoz, bir xil mavzu va ko'pincha bir xil iboralarni saqlaydi, ya'ni faqat
# aynan o'sha faylni olib tashlash yetarli emas.
#
# Manba eval-120 (117 qo'ng'iroq) — eski `calls-colab/eval.csv` (31 qo'ng'iroq)
# ham qo'shiladi, garchi u eval-120 ning to'liq ichida bo'lsa ham: ro'yxatlar
# kelajakda ajralib ketsa, birlashma xavfsiz tomonda qoladi.
#
# O'lchandi: eski, faqat 31 qo'ng'iroqli ro'yxat bilan train_weighted.csv ning
# 1919 qatoridan 349 tasi (18%) eval-120 qo'ng'iroqlaridan edi va hammasi
# AYNAN o'sha fayllar. Ular bilan o'qitilgan model eval-120 da o'zi ko'rgan
# audioda baholanardi.
EVAL_LISTS = [f'{ROOT}/data/eval120/eval.csv', f'{ROOT}/data/calls-colab/eval.csv']
eval_calls = set()
found = []
for ev in EVAL_LISTS:
    if not os.path.exists(ev):
        continue
    found.append(ev)
    for r in csv.DictReader(open(ev)):
        eval_calls.add(os.path.basename(r['path']).split('.')[0].split('_')[0])
if not found:
    # Jim o'tib ketish eng yomon holat: dataset tayyor ko'rinadi, lekin
    # o'lchovi ma'nosiz bo'ladi. Shuning uchun to'xtaymiz.
    sys.exit('Eval ro\'yxati topilmadi: ' + ', '.join(EVAL_LISTS))
print(f"Eval chetlash: {len(eval_calls)} qo'ng'iroq ({len(found)} ro'yxatdan)", file=sys.stderr)

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
    rep = 1 if (OGIRLIKSIZ or w is None or w < 0.30) else 3 if w >= 0.60 else 2
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

# Qamrov darvozasi. Fayl MAVJUD bo'lishi yetarli emas: `sample_wer.csv`
# 1249 namuna uchun mo'ljallangan edi, lekin yurish uzilib 364 tasi qolgan,
# va qolganlari ro'yxat OXIRI — tasodifiy emas. Bunday ballar bilan
# og'irlash datasetni qiyinlik bo'yicha emas, manba tartibi bo'yicha
# qiyshaytiradi. Fayl bor-yo'qligini tekshirish buni ko'rmaydi.
cov = len(scored) / len(rows) if rows else 0
if not OGIRLIKSIZ:
    print(f"Ball qamrovi: {len(scored)}/{len(rows)} ({cov:.0%})", file=sys.stderr)
if not OGIRLIKSIZ and cov < 0.5 and os.environ.get('BALLAR_ZAXIRASI') != '1':
    sys.exit(f"Qamrov {cov:.0%} — og'irliklar namunalarning yarmidan kamini\n"
             f"aks ettiradi va qolgani tasodifiy tanlanmagan bo'lishi mumkin.\n"
             f"Og'irliksiz o'qitmoqchi bo'lsangiz train_weighted.csv ni\n"
             f"takrorlarsiz ishlating. Atayin davom etish: BALLAR_ZAXIRASI=1")
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
    'top5_chegara': round(top5, 3) if wers else None,
}
json.dump(out, open(f'{OUT}/dataset_report.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
