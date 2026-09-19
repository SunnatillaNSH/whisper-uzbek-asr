#!/usr/bin/env python3
"""Voice AI xatolarini Muxlisa matniga nisbatan tahlil qiladi.

MUHIM FARQ (brifdagi 1-bosqichdan): modelning matnini endpointga qayta
yuborib olish SHART EMAS — u allaqachon SellUp bazasida `voice_transcripts`
jadvalida saqlangan. Ya'ni gipotezalar bepul va bir zumda olinadi.

Yorliq (reference) — Muxlisa AI matni. U ham mutlaq haqiqat emas, audioni
eshitgan MUSTAQIL tizim. Shuning uchun "WER" bu yerda "Muxlisa bilan qancha
farq" degani.

Kirish:  pairs.json  — [{call_key, ref, hyp, duration, call_sec, direction, start_time, lead_id}]
Chiqish: analysis/ papkasidagi hisobotlar va datasetlar.
"""
import json, re, sys, csv, os
from collections import Counter, defaultdict
from statistics import median

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- normallashtirish
# QOIDA 1: o'zbek lotinida apostrof HARFNING BIR QISMI (o', g'). Ma'lumotda
# bir necha xil belgi aralash uchraydi — normallashtirmasak, bir xil jumla
# ham katta WER beradi. Apostrofning O'ZINI olib tashlamaymiz: `ozim` va
# `o'zim` boshqa so'zlar.
APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}

def norm(s: str) -> str:
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

def words(s: str):
    return norm(s).split()

# ---------------------------------------------------------------- tekislash
def align(ref, hyp):
    """Levenshtein DP + backtrace. [(op, ref_word, hyp_word, ref_idx)] qaytaradi.
    jiwer ishlatilmadi: o'rnatilmagan va uning transform argumentlari
    versiyadan versiyaga o'zgaradi (brifdagi 2-qoida)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            c = 0 if ri == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(('eq' if ref[i - 1] == hyp[j - 1] else 'sub', ref[i - 1], hyp[j - 1], i - 1)); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(('del', ref[i - 1], None, i - 1)); i -= 1
        else:
            ops.append(('ins', None, hyp[j - 1], max(i - 1, 0))); j -= 1
    ops.reverse()
    return ops

def lev(a, b):
    if abs(len(a) - len(b)) > 2: return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

# ---------------------------------------------------------------- tahlil
pairs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/an/pairs.json'))

subs = Counter(); dels = Counter(); inss = Counter()
pos_bins = [0] * 10; pos_total = 0
per_sample = []; tot_S = tot_D = tot_I = tot_N = 0
hyps = []

for p in pairs:
    r, h = words(p['ref']), words(p['hyp'])
    if not r: continue
    ops = align(r, h)
    S = sum(1 for o in ops if o[0] == 'sub')
    D = sum(1 for o in ops if o[0] == 'del')
    I = sum(1 for o in ops if o[0] == 'ins')
    tot_S += S; tot_D += D; tot_I += I; tot_N += len(r)
    wer = (S + D + I) / len(r)
    for op, rw, hw, ri in ops:
        if op == 'sub': subs[(rw, hw)] += 1
        elif op == 'del': dels[rw] += 1
        elif op == 'ins': inss[hw] += 1
        if op != 'eq':
            b = min(int(ri / max(len(r), 1) * 10), 9)
            pos_bins[b] += 1; pos_total += 1
    dur = float(p.get('duration') or p.get('call_sec') or 0)
    per_sample.append({
        'call_key': p['call_key'], 'wer': round(wer, 4), 'ref_words': len(r), 'hyp_words': len(h),
        'sub': S, 'del': D, 'ins': I, 'duration': round(dur, 1),
        'cps': round(len(p['ref']) / dur, 1) if dur else None,
    })
    hyps.append({'call_key': p['call_key'], 'reference': p['ref'], 'hypothesis': p['hyp'],
                 'duration': dur, 'wer': round(wer, 4)})

overall = (tot_S + tot_D + tot_I) / tot_N
wers = sorted(x['wer'] for x in per_sample)
pct = lambda q: wers[min(int(len(wers) * q), len(wers) - 1)]

# ---------------------------------------------------------------- tinish belgilari
def punct_stats(texts):
    n_words = sum(len(str(t).split()) for t in texts)
    joined = ' '.join(str(t) for t in texts)
    cnt = {c: joined.count(c) for c in '.,?!'}
    sents = [s for s in re.split(r'[.!?]+', joined) if s.strip()]
    apos = Counter(ch for ch in joined if ch in "‘’ʻʼ'`´")
    caps = sum(1 for w in joined.split() if w[:1].isupper())
    return {
        'words': n_words,
        'per100': {c: round(v / n_words * 100, 2) for c, v in cnt.items()},
        'counts': cnt,
        'sentences': len(sents),
        'avg_sentence_words': round(n_words / max(len(sents), 1), 1),
        'apostrophes': {repr(k): v for k, v in apos.most_common()},
        'capitalized_pct': round(caps / max(n_words, 1) * 100, 1),
    }
pm = punct_stats([p['ref'] for p in pairs])
pv = punct_stats([p['hyp'] for p in pairs])

# savol belgisi: Muxlisa ? qo'ygan gaplarda model ham qo'yganmi (taxminiy —
# gaplar soni bo'yicha nisbat)
q_ref = sum(str(p['ref']).count('?') for p in pairs)
q_hyp = sum(str(p['hyp']).count('?') for p in pairs)

# imlo tebranishi: 1-2 harf farq qiladigan almashtirishlar
spelling = Counter(); acoustic = Counter()
for (rw, hw), c in subs.items():
    (spelling if lev(rw, hw) <= 2 else acoustic)[(rw, hw)] += c
sp_total = sum(spelling.values()); ac_total = sum(acoustic.values())

# ---------------------------------------------------------------- chiqish
json.dump(hyps, open(f'{OUT}/hypotheses.json', 'w'), ensure_ascii=False, indent=1)

with open(f'{OUT}/sample_scores.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys())); w.writeheader(); w.writerows(per_sample)

# hard examples: eng yomonlar, LEKIN eng yuqori 5% chetlanadi (buzilgan yorliq)
srt = sorted(per_sample, key=lambda x: -x['wer'])
cut = max(1, int(len(srt) * 0.05))
with open(f'{OUT}/hard_examples.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['call_key', 'wer', 'duration', 'ref_words'])
    for x in srt[cut:cut + 60]: w.writerow([x['call_key'], x['wer'], x['duration'], x['ref_words']])

DOMAIN = ['face', 'id', 'davomat', 'xodim', 'apparat', 'kontrol', 'shartnoma', 'buxgalteriya',
          'hr', 'plyus', 'dastur', 'zapravka', 'lokatsiya', 'ustanovka', 'tarif', 'litsenziya']
dom_hits = Counter()
for (rw, hw), c in subs.items():
    if any(d in rw for d in DOMAIN): dom_hits[rw] += c
open(f'{OUT}/hotwords.txt', 'w').write(', '.join(w for w, _ in dom_hits.most_common(40)))

summary = {
    'samples': len(per_sample), 'ref_words': tot_N,
    'overall_wer': round(overall, 4),
    'sub': tot_S, 'del': tot_D, 'ins': tot_I,
    'sub_pct': round(tot_S / (tot_S + tot_D + tot_I) * 100, 1),
    'del_pct': round(tot_D / (tot_S + tot_D + tot_I) * 100, 1),
    'ins_pct': round(tot_I / (tot_S + tot_D + tot_I) * 100, 1),
    'wer_median': round(median(wers), 4), 'wer_p10': round(pct(0.10), 4), 'wer_p90': round(pct(0.90), 4),
    'position_bins': [round(b / pos_total * 100, 1) for b in pos_bins],
    'punct_muxlisa': pm, 'punct_voice': pv,
    'question_marks': {'muxlisa': q_ref, 'voice': q_hyp},
    'spelling_subs': sp_total, 'acoustic_subs': ac_total,
    'spelling_pct_of_subs': round(sp_total / max(tot_S, 1) * 100, 1),
}
json.dump(summary, open(f'{OUT}/summary.json', 'w'), ensure_ascii=False, indent=1)

json.dump({'subs': [[list(k), v] for k, v in subs.most_common(60)],
           'dels': dels.most_common(40), 'inss': inss.most_common(40),
           'spelling': [[list(k), v] for k, v in spelling.most_common(40)],
           'acoustic': [[list(k), v] for k, v in acoustic.most_common(40)]},
          open(f'{OUT}/patterns.json', 'w'), ensure_ascii=False, indent=1)

print(json.dumps(summary, ensure_ascii=False, indent=1)[:1500])
