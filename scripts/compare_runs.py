#!/usr/bin/env python3
"""Ikki o'lchov yurishini JUFTLASHGAN bootstrap bilan solishtiradi.

NEGA JUFTLASHGAN. Ikki modelning WER'ini alohida-alohida hisoblab, ayirmasini
olish kam ma'lumot beradi: namunalar qiyinligi juda har xil, va farqning
katta qismi qaysi namunalar tushganidan kelib chiqadi. Juftlashgan usulda
ikkala model AYNAN BIR XIL namunalarda o'lchanadi va bootstrap namunalarni
qayta-qayta tanlab, farq qanchalik barqaror ekanini ko'rsatadi.

Round 3 va 4 ni aynan shu usul ajratdi: farq 1.04 punkt, lekin 95% oraliq
[-6.14, +7.13] va p = 0.732 — ya'ni farq yo'q edi.

IKKI O'LCHOV BERILADI
  * WER — odatdagi (almashtirish + tushish + qo'shish) / yorliq so'zlari
  * imlo-normallashtirilgan WER — almashtirishlar ichida belgi masofasi
    <= 2 bo'lganlari TO'G'RI deb sanaladi ("allo"/"alo", "o'ttiz"/"o'tiz").
    Ular akustik xato emas, imlo tebranishi; tahlilda almashtirishlarning
    ~31% i shunday chiqqandi. Ikkinchi raqam "model haqiqatan boshqa so'z
    eshitdimi" degan savolga javob beradi.

    python3 scripts/compare_runs.py analysis/eval120_prod.json analysis/eval120_r5.json
"""

import argparse
import json
import os
import random
import re
import sys

APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}


def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def char_dist(a, b, cap=3):
    """Belgi darajasidagi Levenshtein, cap dan oshsa to'xtaydi."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def align(ref, hyp):
    """So'z darajasidagi tekislash — (S juftliklari, D soni, I soni).

    To'liq matritsa + orqaga yurish. Namunalar qisqa (<=30 s, ~80 so'z),
    shuning uchun xotira muammo emas, almashtirilgan JUFTLIKLAR esa
    imlo tahlili uchun kerak.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    subs, dels, ins = [], 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs.append((ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1
    return subs, dels, ins


def score(results):
    """Har bir namuna uchun (xato, imlo-xato, yorliq so'zlari)."""
    per = {}
    for r in results:
        if not r or r.get("error"):
            continue
        ref, hyp = norm(r["reference"]).split(), norm(r["hypothesis"]).split()
        subs, dels, ins = align(ref, hyp)
        spelling = sum(1 for a, b in subs if char_dist(a, b) <= 2)
        per[r["path"]] = {
            "err": len(subs) + dels + ins,
            "err_norm": len(subs) + dels + ins - spelling,   # imlo to'g'ri deb sanaladi
            "n": len(ref),
            "sub": len(subs), "del": dels, "ins": ins, "spelling": spelling,
        }
    return per


def corpus(per, keys, field="err"):
    e = sum(per[k][field] for k in keys)
    n = sum(per[k]["n"] for k in keys)
    return 100 * e / n if n else float("nan")


def bootstrap(a, b, keys, field, rounds=10000, seed=0):
    rng = random.Random(seed)
    k = len(keys)
    diffs = []
    for _ in range(rounds):
        pick = [keys[rng.randrange(k)] for _ in range(k)]
        diffs.append(corpus(b, pick, field) - corpus(a, pick, field))
    diffs.sort()
    lo = diffs[int(0.025 * rounds)]
    hi = diffs[int(0.975 * rounds)]
    # p = 2 * min(P(farq <= 0), P(farq >= 0)), 1 dan oshmaydi.
    #
    # Avval `2 * min(neg, 1 - neg)` yozilgandi va u teng holatda buziladi:
    # barcha farqlar aynan nol bo'lsa P(<=0) = 1 bo'lib, p = 0 chiqardi —
    # ya'ni "farq yo'q" holati "juda ishonchli farq" bo'lib ko'rinardi.
    le = sum(1 for d in diffs if d <= 0) / rounds
    ge = sum(1 for d in diffs if d >= 0) / rounds
    p = min(1.0, 2 * min(le, ge))
    return lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="asosiy yurish (masalan hozirgi production)")
    ap.add_argument("cand", help="taqqoslanadigan yurish (masalan Round 5)")
    ap.add_argument("--rounds", type=int, default=10000)
    a = ap.parse_args()

    ra = json.load(open(a.base, encoding="utf-8"))
    rb = json.load(open(a.cand, encoding="utf-8"))
    pa, pb = score(ra["results"]), score(rb["results"])

    keys = sorted(set(pa) & set(pb))
    if not keys:
        sys.exit("Umumiy namuna topilmadi — ikkala yurish bir xil CSV'da bo'lishi kerak.")
    only_a, only_b = len(pa) - len(keys), len(pb) - len(keys)
    if only_a or only_b:
        print(f"⚠️  Faqat bittasida bor: asosiyda {only_a}, nomzodda {only_b} — "
              f"ular CHIQARILDI (taqqoslash bir xil namunalarda bo'lishi shart)")

    print(f"\nTaqqoslash: {os.path.basename(a.base)}  →  {os.path.basename(a.cand)}")
    print(f"Umumiy namuna: {len(keys)} | yorliq so'zlari: {sum(pa[k]['n'] for k in keys)}\n")

    for field, label in (("err", "WER"), ("err_norm", "WER (imlo normallashtirilgan)")):
        wa, wb = corpus(pa, keys, field), corpus(pb, keys, field)
        lo, hi, p = bootstrap(pa, pb, keys, field, a.rounds)
        better = hi < 0
        worse = lo > 0
        verdict = ("YAXSHILANDI — ishonchli" if better else
                   "YOMONLASHDI — ishonchli" if worse else
                   "farq ISHONCHLI EMAS (oraliq nolni kesib o'tadi)")
        print(f"{label}")
        print(f"  asosiy {wa:6.2f}%   nomzod {wb:6.2f}%   farq {wb - wa:+6.2f} punkt")
        print(f"  95% oraliq [{lo:+.2f}, {hi:+.2f}]   p = {p:.3f}")
        print(f"  -> {verdict}\n")

    def brk(per):
        s = sum(per[k]["sub"] for k in keys)
        d = sum(per[k]["del"] for k in keys)
        i = sum(per[k]["ins"] for k in keys)
        sp = sum(per[k]["spelling"] for k in keys)
        t = s + d + i or 1
        return (f"S {s:5d} ({100*s//t:2d}%)  D {d:5d} ({100*d//t:2d}%)  "
                f"I {i:5d} ({100*i//t:2d}%)  | imlo {sp} ({100*sp//(s or 1)}% almashtirishlardan)")

    print("Xato tarkibi")
    print(f"  asosiy : {brk(pa)}")
    print(f"  nomzod : {brk(pb)}")

    better = sum(1 for k in keys if pb[k]["err"] < pa[k]["err"])
    worse = sum(1 for k in keys if pb[k]["err"] > pa[k]["err"])
    print(f"\nNamuna darajasida: yaxshilandi {better} | yomonlashdi {worse} | "
          f"teng {len(keys) - better - worse}")


if __name__ == "__main__":
    main()
