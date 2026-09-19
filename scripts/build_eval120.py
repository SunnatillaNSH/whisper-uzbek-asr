#!/usr/bin/env python3
"""eval_calls_120.json ro'yxatidan amaldagi eval va train CSV larini quradi.

`make_eval_120.py` faqat QO'NG'IROQLAR ro'yxatini tanlaydi. Bu skript o'sha
ro'yxatni namuna darajasiga tushiradi: har bir qo'ng'iroqning barcha
bo'laklarini topib, eval va train ga ajratadi.

MANBALAR
  data/calls-colab/{train,eval}.csv   — eski partiya, <=30 s ga bo'lingan
  data/calls-dataset-28s/manifest.jsonl — yangi partiya, 28 s bo'laklar

Eski partiyaning XOM fayllari (data/calls-dataset) ISHLATILMAYDI — ularning
200 tasi calls-colab bilan to'liq takrorlanadi va 605 tasi 30 soniyadan
uzun. calls-colab o'sha ma'lumotning tayyorlangan varianti.

AJRATISH QO'NG'IROQ BO'YICHA. Bitta suhbatning bo'laklari train va eval'ga
bo'linib ketsa, model eval suhbatini treningda ko'radi va natija soxta
yaxshi chiqadi.

    python3 scripts/build_eval120.py
"""

import csv
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "eval120")


def call_id_of(path):
    m = re.match(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_samples():
    """[(nisbiy yo'l, matn, call_id)] — ikkala partiyadan."""
    rows = []
    for name in ("train.csv", "eval.csv"):
        p = os.path.join(D, "calls-colab", name)
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((f"calls-colab/{r['path']}", r["sentence"], call_id_of(r["path"])))

    p = os.path.join(D, "calls-dataset-28s", "manifest.jsonl")
    with open(p, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            rows.append((f"calls-dataset-28s/{m['path']}", m["sentence"], m["call_id"]))
    return rows


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("path,sentence\n")
        for p, s, _ in rows:
            f.write('"%s","%s"\n' % (p, str(s).replace('"', '""')))


def main():
    spec = json.load(open(os.path.join(D, "eval_calls_120.json"), encoding="utf-8"))
    eval_ids = {c["call_id"] for c in spec["calls"]}

    rows = load_samples()
    missing = [r for r in rows if not os.path.exists(os.path.join(D, r[0]))]
    if missing:
        print(f"⚠️  {len(missing)} ta audio fayl topilmadi, masalan {missing[0][0]}")
        rows = [r for r in rows if os.path.exists(os.path.join(D, r[0]))]

    ev = [r for r in rows if r[2] in eval_ids]
    tr = [r for r in rows if r[2] not in eval_ids]

    os.makedirs(OUT, exist_ok=True)
    write(os.path.join(OUT, "eval.csv"), ev)
    write(os.path.join(OUT, "train.csv"), tr)

    def stat(name, rs):
        calls = {r[2] for r in rs}
        words = sum(len(str(r[1]).split()) for r in rs)
        src = defaultdict(int)
        for r in rs:
            src[r[0].split("/")[0]] += 1
        return (f"  {name:<6}: {len(rs):5d} namuna | {len(calls):4d} qo'ng'iroq "
                f"| {words:6d} so'z | {dict(src)}")

    print(f"Ro'yxatdagi qo'ng'iroq: {len(eval_ids)}")
    print(stat("eval", ev))
    print(stat("train", tr))

    found = {r[2] for r in ev}
    if len(found) < len(eval_ids):
        print(f"\n⚠️  {len(eval_ids) - len(found)} ta qo'ng'iroqning namunasi topilmadi "
              f"(ehtimol hammasi 30 soniyadan uzun edi)")

    # Kesishuv bo'lmasligi SHART
    assert not ({r[2] for r in ev} & {r[2] for r in tr}), "eval va train qo'ng'iroqlari kesishdi"
    print(f"\n  ✅ kesishuv yo'q · {OUT}")


if __name__ == "__main__":
    main()
