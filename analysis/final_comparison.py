#!/usr/bin/env python3
"""Muxlisa vs Voice AI — yakuniy solishtirish (yangi endpoint sozlamalarida).

Gipotezalar: har bir dataset fayli ALOHIDA endpointga yuborilgan,
`hotwords: ""`, `language: uz`, beam 5, VAD=2000.

Ikki xil WER beriladi:
  * XOM      — faqat apostrof birlashtirilgan, tinish belgilari olib tashlangan
  * IMLO'siz — qo'shimcha yozuv-varianti qoidalari qo'llangan (pastda ro'yxat)

Maqsad: "haqiqiy xato" ni "yozuv farqi" dan ajratish. Qoidalar IKKALA tomonga
bir xil qo'llanadi — aks holda o'lchov bir tomonga og'ib ketardi.
"""
import csv, json, os, re, sys, random
from collections import Counter, defaultdict
from statistics import median

OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(OUT)
APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}

def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------- imlo normallashtirish
# Har bir qoida IKKALA tomonga bir xil qo'llanadi.
BACKCHANNEL = {'hm','hmm','ha','aha','ahaa','aa','ah','xa','he','ehe','mhm','uhu'}
GREET_ALAYKUM = {'vaalaykum','valaykum','vaalayqum','alaykum','aleykum','alaykm'}
GREET_ASSALOM = {'assalomu','assalom','assalomu','salom','assalamu'}
ALO = {'allo','alo','alyo','allyo'}
XOP = {"xo'p",'xop','hop',"ho'p"}
YOQ = {"yo'q",'yoq','yo',"yo'q"}
DIGITS = {'0':'nol','1':'bir','2':'ikki','3':'uch','4':'to\'rt','5':'besh','6':'olti',
          '7':'yetti','8':'sakkiz','9':'to\'qqiz','10':'o\'n','100':'yuz','1000':'ming'}

def spell(w):
    if w in BACKCHANNEL: return 'ha'
    if w in GREET_ALAYKUM: return 'alaykum'
    if w in GREET_ASSALOM: return 'assalom'
    if w in ALO: return 'alo'
    if w in XOP: return "xo'p"
    if w in YOQ: return "yo'q"
    if w in DIGITS: return DIGITS[w]
    w = re.sub(r"(.)\1+", r"\1", w)        # takroriy harf: o'ttiz -> o'tiz, labbay -> labay
    return w

def W(s, sp=False):
    ws = norm(s).split()
    return [spell(w) for w in ws] if sp else ws

def ops(r, h):
    n, m = len(r), len(h)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(r[i-1] != h[j-1]))
    i, j, S, D, I = n, m, 0, 0, 0
    subs, dels, inss = [], [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + (0 if r[i-1] == h[j-1] else 1):
            if r[i-1] != h[j-1]: S += 1; subs.append((r[i-1], h[j-1]))
            i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j]+1: D += 1; dels.append(r[i-1]); i -= 1
        else: I += 1; inss.append(h[j-1]); j -= 1
    return S, D, I, subs, dels, inss

# ---------------------------------------------------- ma'lumot
ref, src = {}, {}
for r in csv.DictReader(open(f'{ROOT}/data/calls-colab/train.csv')):
    k = f"calls-colab/{r['path']}"; ref[k] = r['sentence']; src[k] = 'calls-colab'
for line in open(f'{ROOT}/data/calls-dataset-28s/manifest.jsonl'):
    m = json.loads(line); k = f"calls-dataset-28s/{m['path']}"
    ref[k] = m['sentence']; src[k] = 'calls-dataset-28s'

hyp = json.load(open(sys.argv[1]))
empty = sum(1 for k, v in hyp.items() if k in ref and not v)
rows = []
agg = {'raw': [0,0,0,0,0,0], 'sp': [0,0,0,0,0,0]}   # S,D,I,N,refw,hypw
SUB, DEL, INS = Counter(), Counter(), Counter()
for k, h in hyp.items():
    if k not in ref or not h: continue
    for mode, sp in (('raw', False), ('sp', True)):
        r, y = W(ref[k], sp), W(h, sp)
        if not r: continue
        S, D, I, subs, dels, inss = ops(r, y)
        a = agg[mode]; a[0]+=S; a[1]+=D; a[2]+=I; a[3]+=len(r); a[4]+=len(r); a[5]+=len(y)
        if mode == 'raw':
            SUB.update(subs); DEL.update(dels); INS.update(inss)
            rows.append({'path': k, 'src': src[k], 'wer': (S+D+I)/len(r),
                         'ref_words': len(r), 'hyp_words': len(y), 'sub': S, 'del': D, 'ins': I})
        else:
            rows[-1]['wer_sp'] = (S+D+I)/len(r)

def stats(a, key):
    S,D,I,N,rw,hw = a
    return {'WER': (S+D+I)/N, 'sub': S, 'del': D, 'ins': I,
            'sub%': S/(S+D+I)*100, 'del%': D/(S+D+I)*100, 'ins%': I/(S+D+I)*100,
            'qamrov%': hw/rw*100}
raw, sp = stats(agg['raw'], 'raw'), stats(agg['sp'], 'sp')
ws = sorted(x['wer'] for x in rows); wsp = sorted(x['wer_sp'] for x in rows)
q = lambda a, p: a[min(int(len(a)*p), len(a)-1)]

# manba bo'yicha
bysrc = {}
for s in ('calls-colab', 'calls-dataset-28s'):
    sub = [x for x in rows if x['src'] == s]
    if not sub: continue
    N = sum(x['ref_words'] for x in sub)
    bysrc[s] = {'n': len(sub), 'WER': sum(x['sub']+x['del']+x['ins'] for x in sub)/N,
                'qamrov%': sum(x['hyp_words'] for x in sub)/N*100,
                'mediana': median([x['wer'] for x in sub])}

# qo'ng'iroq darajasiga yig'ish — eski tahlil bilan kesishma uchun
def cid(p): return os.path.basename(p).split('.')[0].split('_')[0]
bycall = defaultdict(lambda: [0,0])
for x in rows:
    bycall[cid(x['path'])][0] += x['sub']+x['del']+x['ins']; bycall[cid(x['path'])][1] += x['ref_words']
old = {}
if os.path.exists(f'{OUT}/sample_scores.csv'):
    old = {r['call_key'].replace('mz:',''): float(r['wer']) for r in csv.DictReader(open(f'{OUT}/sample_scores.csv'))}
common = [c for c in bycall if c in old and bycall[c][1]]
inter = None
if common:
    newW = sum(bycall[c][0] for c in common)/sum(bycall[c][1] for c in common)
    oldW = sum(old[c] for c in common)/len(common)
    inter = {'qo\'ng\'iroq': len(common), 'yangi_WER': newW, 'eski_WER_ortacha': oldW}

random.seed(7)
sample10 = random.sample(rows, min(10, len(rows)))
out = {'namuna': len(rows), 'yorliqda': len(ref), 'bo\'sh_javob': empty,
       'raw': raw, 'sp': sp,
       'mediana': median(ws), 'p10': q(ws,0.10), 'p90': q(ws,0.90),
       'mediana_sp': median(wsp), 'p10_sp': q(wsp,0.10), 'p90_sp': q(wsp,0.90),
       'manba': bysrc, 'kesishma': inter,
       'top_del': DEL.most_common(20), 'top_sub': [[list(k),v] for k,v in SUB.most_common(20)]}
json.dump(out, open(f'{OUT}/final_comparison.json','w'), ensure_ascii=False, indent=1)
json.dump([{**x, 'ref': ref[x['path']], 'hyp': hyp[x['path']]} for x in sample10],
          open(f'{OUT}/final_samples.json','w'), ensure_ascii=False, indent=1)
with open(f'{OUT}/sample_wer.csv','w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=['path','src','wer','wer_sp','ref_words','hyp_words','sub','del','ins'])
    w.writeheader()
    for x in rows: w.writerow({k: (round(x[k],4) if isinstance(x[k],float) else x[k]) for k in w.fieldnames})
print(json.dumps({k:v for k,v in out.items() if k not in ('top_del','top_sub')}, ensure_ascii=False, indent=1)[:1200])
