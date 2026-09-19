#!/usr/bin/env python3
"""Tashqi korpusdan (UzbekVoice) kerakli qismini OQIM bilan olib diskka yozadi.

NEGA OQIM. `load_dataset(...)` butun datasetni yuklab oladi. UzbekVoice
o'nlab GB, bizga esa ~25k namuna kerak. Pod soatiga pul turadi, ya'ni
ortiqcha yuklash to'g'ridan-to'g'ri byudjetdan yeydi — yuklash treningdan
uzoqroq davom etishi mumkin. Oqim rejimida kerakli soni olingach to'xtaydi.

NEGA DISKKA YOZILADI. 25k namunani xotirada saqlab bo'lmaydi (~8 GB xom
float). Diskka flac qilib yozsak, trening paytida `datasets` ularni dangasa
o'qiydi — xotira doimiy qoladi.

SXEMA OLDINDAN MA'LUM EMAS. Ustun nomlari (matn, ovoz berish) repo'dan
repo'ga farq qiladi, shuning uchun birinchi yozuvda aniqlanadi va EKRANGA
CHIQARILADI. Matn ustuni topilmasa — to'xtaymiz, taxmin qilmaymiz.

    python3 scripts/prep_extra.py --out /workspace/extra --n 25000
"""

import argparse
import json
import os
import re
import sys

MIN_DUR, MAX_DUR = 1.0, 30.0
MIN_CHARS = 5
CPS_LO, CPS_HI = 6.0, 26.0
SR = 16000

TEXT_COLS = ("sentence", "text", "transcription", "transcript", "normalized_text")


def pick_text_col(cols):
    for c in TEXT_COLS:
        if c in cols:
            return c
    return None


def vote_cols(cols):
    """(up, down) ustun nomlari — nomi repo'dan repo'ga farq qiladi."""
    up = next((c for c in cols if re.search(r"up_?vote", c, re.I)), None)
    dn = next((c for c in cols if re.search(r"down_?vote", c, re.I)), None)
    return up, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="DavronSherbaev/uzbekvoice-filtered")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=25000, help="saqlanadigan namuna soni")
    ap.add_argument("--max-scan", type=int, default=0,
                    help="ko'pi bilan shuncha yozuv ko'riladi (0 = cheksiz)")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="up - down shu qiymatdan kam bo'lsa tashlanadi")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from datasets import load_dataset

    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)
    print(f"Manba : {a.id} ({a.split}), oqim rejimi", flush=True)
    ds = load_dataset(a.id, split=a.split, streaming=True)

    kept, scanned = 0, 0
    tcol = ucol = dcol = None
    skip = {}
    rows = []

    for ex in ds:
        scanned += 1
        if a.max_scan and scanned > a.max_scan:
            print(f"max-scan {a.max_scan} ga yetildi", flush=True)
            break

        if tcol is None:
            cols = list(ex.keys())
            tcol = pick_text_col(cols)
            ucol, dcol = vote_cols(cols)
            print(f"Ustunlar   : {cols}")
            print(f"Matn       : {tcol}")
            print(f"Ovoz berish: up={ucol} down={dcol}")
            if tcol is None:
                sys.exit(f"Matn ustuni topilmadi. Mavjud ustunlar: {cols}\n"
                         f"--id to'g'rimi? Kerak bo'lsa TEXT_COLS ga qo'shing.")

        text = str(ex.get(tcol) or "").strip()
        if len(text) < MIN_CHARS:
            skip["matn kalta"] = skip.get("matn kalta", 0) + 1
            continue

        # Ovoz berish filtri — ustunlar bo'lsa. Yo'q bo'lsa jim o'tmaymiz,
        # bir marta aytamiz: "filtrsiz" ham natijaning bir qismi.
        if ucol and dcol:
            try:
                if int(ex[ucol] or 0) - int(ex[dcol] or 0) < a.min_votes:
                    skip["ovoz berish"] = skip.get("ovoz berish", 0) + 1
                    continue
            except (TypeError, ValueError):
                pass

        au = ex.get("audio")
        if not isinstance(au, dict) or au.get("array") is None:
            skip["audio yo'q"] = skip.get("audio yo'q", 0) + 1
            continue
        arr = np.asarray(au["array"], dtype=np.float32)
        sr = int(au.get("sampling_rate") or SR)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != SR:
            k = int(round(len(arr) * SR / sr))
            arr = np.interp(np.linspace(0, len(arr) - 1, k),
                            np.arange(len(arr)), arr).astype(np.float32)
        dur = len(arr) / SR
        if not (MIN_DUR <= dur <= MAX_DUR):
            skip["davomiylik"] = skip.get("davomiylik", 0) + 1
            continue
        cps = len(text) / dur
        if not (CPS_LO <= cps <= CPS_HI):
            skip["cps"] = skip.get("cps", 0) + 1
            continue

        rel = f"audio/{kept:06d}.flac"
        sf.write(os.path.join(a.out, rel), arr, SR)
        rows.append((rel, text, round(dur, 2)))
        kept += 1
        if kept % 1000 == 0:
            print(f"  saqlandi {kept}/{a.n} (ko'rildi {scanned})", flush=True)
        if kept >= a.n:
            break

    import csv
    with open(os.path.join(a.out, "extra.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "sentence"])
        for rel, text, _ in rows:
            w.writerow([rel, text])
    hours = sum(r[2] for r in rows) / 3600
    json.dump({"id": a.id, "kept": kept, "scanned": scanned, "hours": round(hours, 2),
               "text_col": tcol, "up": ucol, "down": dcol, "skipped": skip},
              open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1)

    print(f"\n  Saqlandi : {kept} namuna, {hours:.2f} soat")
    print(f"  Ko'rildi : {scanned}")
    print(f"  Tashlandi: {dict(sorted(skip.items(), key=lambda x: -x[1]))}")
    print(f"  -> {a.out}/extra.csv")
    if kept < a.n:
        print(f"\n  DIQQAT: {a.n} so'ralgandi, {kept} chiqdi — manba tugadi "
              f"yoki filtrlar juda qattiq.")


if __name__ == "__main__":
    main()
