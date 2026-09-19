#!/usr/bin/env python3
"""eval-120 dan 50 ta qo'ng'iroqli QISM TO'PLAM tanlaydi.

NEGA 50. Foydalanuvchi qarori: Round 5 dan keyin faqat 50 qo'ng'iroq
o'giriladi. Bu eval-120 ni almashtirmaydi — u o'sha qoladi; bu uning
qat'iy belgilangan qismi.

NARXI: 50 qo'ng'iroq ~5900 yorliq so'zi beradi, 120 dagi 14 015 ning
42% i. Bootstrap oralig'i taxminan sqrt(14015/5900) = 1.54 barobar
kengayadi — ya'ni 120 da ~±3 punkt bo'lgani bu yerda ~±5 punkt bo'ladi.
Amalda: 3 punktdan kichik yaxshilanishni bu to'plam ISHONCHLI ko'rsata
olmaydi. Hisobotda shu ochiq yoziladi.

TANLASH. Asl 31 ta "locked" qo'ng'iroq shartsiz kiradi (ular avvalgi
raundlarda ham o'lchangan, ya'ni tarixiy taqqoslash uzilmaydi). Qolgan
19 tasi qatlamlab olinadi: (batch, direction) bo'yicha populyatsiyaga
mutanosib, har qatlamda davomiylik bo'yicha tarqoq.

Urug' QAT'IY va faylga yoziladi — tanlov qayta ishga tushirilsa aynan
o'sha 50 ta chiqishi kerak, aks holda ikki model turli to'plamda
o'lchanadi va taqqoslash ma'nosini yo'qotadi.

    python3 scripts/make_eval_50.py
"""

import collections
import csv
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "eval_calls_120.json")
OUT = os.path.join(ROOT, "data", "eval_calls_50.json")
EVAL_CSV = os.path.join(ROOT, "data", "eval120", "eval.csv")
OUT_CSV = os.path.join(ROOT, "data", "eval120", "eval50.csv")
SEED = 20260920
TARGET = 50


def call_id(path):
    return os.path.basename(str(path)).split(".")[0].split("_")[0]


def main():
    meta = json.load(open(SRC, encoding="utf-8"))
    calls = {str(c["call_id"]): c for c in meta["calls"]}
    locked = [str(x) for x in meta.get("locked_from_previous", [])]
    locked = [c for c in locked if c in calls]

    rest = [c for c in calls if c not in set(locked)]
    need = TARGET - len(locked)
    if need < 0:
        raise SystemExit(f"locked {len(locked)} > {TARGET}")

    # Qatlamlar: (batch, direction). Populyatsiyadagi ulushga mutanosib.
    strata = collections.defaultdict(list)
    for c in rest:
        d = calls[c]
        strata[(d.get("batch"), d.get("direction"))].append(c)

    rng = random.Random(SEED)
    # Har qatlamdan nechta: ulushga mutanosib, eng katta qoldiq bo'yicha yaxlitlash.
    total = sum(len(v) for v in strata.values())
    quota, frac = {}, []
    for k, v in sorted(strata.items(), key=lambda x: str(x[0])):
        exact = need * len(v) / total
        quota[k] = int(exact)
        frac.append((exact - int(exact), k))
    for _, k in sorted(frac, reverse=True)[: need - sum(quota.values())]:
        quota[k] += 1

    picked = []
    for k, v in sorted(strata.items(), key=lambda x: str(x[0])):
        n = min(quota[k], len(v))
        if not n:
            continue
        # Davomiylik bo'yicha tartiblab, teng oraliqlardan olamiz — bitta
        # qatlam ichida faqat qisqa yoki faqat uzun qo'ng'iroq tushmasin.
        v = sorted(v, key=lambda c: calls[c].get("duration") or 0)
        step = len(v) / n
        idx = sorted({min(len(v) - 1, int(i * step + step / 2)) for i in range(n)})
        while len(idx) < n:
            extra = rng.choice([i for i in range(len(v)) if i not in idx])
            idx.append(extra)
            idx.sort()
        picked += [v[i] for i in idx[:n]]

    chosen = sorted(set(locked) | set(picked), key=lambda c: int(c))
    if len(chosen) != TARGET:
        raise SystemExit(f"{len(chosen)} ta tanlandi, {TARGET} kerak edi")

    rows = [r for r in csv.DictReader(open(EVAL_CSV, encoding="utf-8"))
            if call_id(r["path"]) in set(chosen)]
    words = sum(len(r["sentence"].split()) for r in rows)
    dur = sum(calls[c].get("duration") or 0 for c in chosen)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sentence"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({"path": r["path"], "sentence": r["sentence"]})

    # Faqat qo'ng'iroq raqami va qatlam — matn ham, CRM identifikatori ham yo'q
    # (repo ochiq).
    json.dump({"target": TARGET, "seed": SEED, "source": "eval_calls_120.json",
               "locked": sorted(locked, key=int), "calls": chosen,
               "samples": len(rows), "ref_words": words,
               "audio_hours": round(dur / 3600, 2)},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"Tanlandi : {len(chosen)} qo'ng'iroq ({len(locked)} locked + {need} qatlamli)")
    print(f"Namuna   : {len(rows)} | yorliq so'zlari {words} | audio {dur/3600:.2f} soat")
    print(f"eval-120 dan ulush: {100*words/14015:.0f}% so'z")
    print(f"  -> {os.path.relpath(OUT, ROOT)}, {os.path.relpath(OUT_CSV, ROOT)}")


if __name__ == "__main__":
    main()
