#!/usr/bin/env python3
"""Eval to'plamini 31 dan 120 qo'ng'iroqqa kengaytiradi.

NEGA. Hozirgi eval — 31 qo'ng'iroq, 81 namuna, 3757 so'z. Ishonch oralig'i
+-7 foiz punkt, ya'ni bu asbob 7 punktdan kichik yaxshilanishni ko'rmaydi.
Round 4 aynan shuning qurboni bo'ldi: eval loss yaxshilandi, WER'da esa
p=0.732 chiqdi va ma'lumot qo'shish foyda berdimi bilib bo'lmadi.

120 qo'ng'iroqda oraliq ~+-3 punktga tushadi.

QOIDALAR
  * Hozirgi 31 qo'ng'iroq O'ZGARMAY ichida qoladi — aks holda oldingi
    round'lar bilan taqqoslab bo'lmaydi.
  * Ajratish QO'NG'IROQ bo'yicha, namuna bo'yicha emas. Bitta suhbatning
    bo'laklari train va eval'ga bo'linib ketsa, model eval suhbatini
    treningda ko'radi va natija soxta yaxshi chiqadi.
  * Tanlov qatlamlangan va PROPORSIONAL: partiya (eski/yangi) va yo'nalish
    (kiruvchi/chiquvchi) populyatsiyadagi nisbatda, har bir katakda
    davomiylik kvartillari bo'yicha teng.

    Yo'nalishni TENG qilish xato bo'lardi: populyatsiyada chiquvchi 76%,
    kiruvchi 24%. Teng tanlasak, eval'da kiruvchi 44% ga chiqib ketadi va
    WER haqiqiy trafikni aks ettirmaydi — kiruvchi qo'ng'iroqlar boshqacha
    qiyinlikda bo'lsa, o'lchov sistematik siljiydi.
  * Natija faylga yoziladi va commit qilinadi — keyingi round'lar AYNAN
    shu ro'yxatni ishlatsin.

Ishga tushirish:
    python3 scripts/make_eval_120.py
"""

import csv
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
TARGET = 120
SEED = 20260919

BATCHES = [
    ("eski", os.path.join(D, "calls-dataset", "manifest.jsonl")),
    ("yangi", os.path.join(D, "calls-dataset-28s", "manifest.jsonl")),
]


def load_calls():
    """manifest'lardan qo'ng'iroq darajasidagi ma'lumot yig'adi."""
    calls = {}
    for batch, path in BATCHES:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            c = calls.setdefault(r["call_id"], {
                "call_id": r["call_id"], "batch": batch,
                "direction": r.get("direction"),
                "duration": 0.0, "samples": 0, "words": 0,
                # lead_id faqat tanlov paytida ishlatiladi (bitta mijozdan
                # ko'p olmaslik uchun) va faylga YOZILMAYDI — repo ochiq,
                # CRM identifikatorlari va sanalar u yerga tushmasligi kerak.
                "_lead": r.get("lead_id"),
            })
            c["duration"] += r["duration"]
            c["samples"] += 1
            c["words"] += len(str(r["sentence"]).split())
    return calls


def current_eval_ids():
    ids = set()
    with open(os.path.join(D, "calls-colab", "eval.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = re.match(r"(\d+)", os.path.basename(r["path"]))
            if m:
                ids.add(int(m.group(1)))
    return ids


def by_quartile(pool, n, rng):
    """Davomiylik kvartillari bo'yicha teng tanlov (yo'nalish bu yerda emas)."""
    if n <= 0 or not pool:
        return []
    pool = sorted(pool, key=lambda c: c["duration"])
    q = max(1, len(pool) // 4)
    buckets = [pool[i:i + q] for i in range(0, len(pool), q)][:4]
    while len(buckets) < 4:
        buckets.append([])

    picked, used_leads = [], set()
    per = [n // 4 + (1 if i < n % 4 else 0) for i in range(4)]
    for qi, want in enumerate(per):
        b = list(buckets[qi])
        rng.shuffle(b)
        # bitta mijozdan ko'p olmaslikka harakat qilamiz
        b.sort(key=lambda c: c["_lead"] in used_leads)
        take = b[:want]
        for c in take:
            used_leads.add(c["_lead"])
        picked += take

    if len(picked) < n:
        rest = [c for c in pool if c not in picked]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]
    return picked[:n]


def targets(population, total):
    """Populyatsiya nisbatiga mos maqsadlar; yaxlitlash farqi eng kattasiga."""
    keys = sorted(population, key=lambda k: -len(population[k]))
    n = sum(len(v) for v in population.values())
    goal = {k: round(total * len(v) / n) for k, v in population.items()}
    diff = total - sum(goal.values())
    if diff:
        goal[keys[0]] += diff
    return goal


def main():
    import random
    rng = random.Random(SEED)

    calls = load_calls()
    locked = current_eval_ids()
    print(f"Jami qo'ng'iroq        : {len(calls)}")
    print(f"Hozirgi eval (saqlanadi): {len(locked)}")

    # Partiyalar bo'yicha proporsional maqsad
    by_batch = defaultdict(list)
    for c in calls.values():
        by_batch[c["batch"]].append(c)
    total = len(calls)
    goal = {b: round(TARGET * len(v) / total) for b, v in by_batch.items()}
    # yaxlitlash farqini eng katta partiyaga beramiz
    diff = TARGET - sum(goal.values())
    if diff:
        goal[max(goal, key=lambda b: len(by_batch[b]))] += diff

    # Maqsad: (partiya, yo'nalish) kataklari populyatsiya nisbatida
    cells = defaultdict(list)
    for c in calls.values():
        cells[(c["batch"], c.get("direction") or "?")].append(c)
    goal = targets(cells, TARGET)

    chosen = [calls[i] for i in locked if i in calls]
    for cell, pool in sorted(cells.items(), key=lambda kv: -len(kv[1])):
        have = sum(1 for c in chosen
                   if (c["batch"], c.get("direction") or "?") == cell)
        need = goal[cell] - have
        free = [c for c in pool if c["call_id"] not in locked]
        add = by_quartile(free, need, rng)
        chosen += add
        print(f"  {cell[0]:<6} {cell[1]:<4}: bor {have:3d} | maqsad {goal[cell]:3d} "
              f"| qo'shildi {len(add):3d}")

    # Qulflangan qo'ng'iroqlar sababli maqsaddan oshib ketishi mumkin —
    # ularni chiqarib tashlamaymiz, faqat qolganini kamaytiramiz.
    if len(chosen) > TARGET:
        extra = len(chosen) - TARGET
        removable = [c for c in chosen if c["call_id"] not in locked]
        rng.shuffle(removable)
        drop = {id(c) for c in removable[:extra]}
        chosen = [c for c in chosen if id(c) not in drop]
        print(f"  (maqsaddan oshgani uchun {extra} ta chiqarildi)")

    chosen.sort(key=lambda c: c["call_id"])
    # Faylga faqat takrorlash uchun zarur maydonlar (transkript ham,
    # CRM identifikatori ham, sana ham yo'q).
    public = [{k: v for k, v in c.items() if not k.startswith("_")} for c in chosen]

    out = {
        "target": TARGET,
        "seed": SEED,
        "locked_from_previous": sorted(locked),
        "calls": public,
        "summary": {
            "calls": len(chosen),
            "samples": sum(c["samples"] for c in chosen),
            "hours": round(sum(c["duration"] for c in chosen) / 3600, 2),
            "words": sum(c["words"] for c in chosen),
            "by_batch": {b: sum(1 for c in chosen if c["batch"] == b) for b in by_batch},
            "by_direction": {
                d: sum(1 for c in chosen if (c.get("direction") or "?") == d)
                for d in sorted({(c.get("direction") or "?") for c in chosen})
            },
        },
    }
    path = os.path.join(D, "eval_calls_120.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    s = out["summary"]
    print("\n" + "=" * 46)
    print(f"  Qo'ng'iroq : {s['calls']}")
    print(f"  Namuna     : {s['samples']}")
    print(f"  Soat       : {s['hours']}")
    print(f"  So'z       : {s['words']}  (avval 3757)")
    print(f"  Partiya    : {s['by_batch']}")
    print(f"  Yo'nalish  : {s['by_direction']}")
    print("=" * 46)
    print(f"  -> {path}")

    # Populyatsiya bilan taqqoslash — qatlamlash ishladimi
    alld = sorted(c["duration"] for c in calls.values())
    evd = sorted(c["duration"] for c in chosen)
    med = lambda x: x[len(x) // 2]
    print(f"\n  Davomiylik medianasi: butun {med(alld):.0f}s | eval {med(evd):.0f}s")


if __name__ == "__main__":
    main()
