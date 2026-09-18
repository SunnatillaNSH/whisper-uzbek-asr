#!/usr/bin/env python3
"""Real qo'ng'iroq datasetini Whisper treningi uchun tayyorlaydi.

Muammo: Whisper encoder'i qat'iy 30 soniyalik oyna bilan ishlaydi. Muxlisa'ga
esa audio 55 soniyalik bo'laklarda yuborilgani uchun 843 namunadan 605 tasi
(10.6 soat) juda uzun — ularni shundayligicha berish modelni buzadi (audio
kesiladi, matn to'liq qoladi → model "eshitilmagan" so'zlarni o'ylab topishga
o'rganadi).

Yechim: uzun namunani JIMLIK joyidan kesamiz va matnni shu nuqtaga eng yaqin
GAP CHEGARASIDAN (nuqta/savol/undov) bo'lamiz. Muxlisa tinish belgilarini
qo'yadi, jimlik esa odatda aynan gap oxirida bo'ladi — shuning uchun bu ikkisi
bir-biriga to'g'ri keladi.

Ishonch yo'q bo'lgan namunani BO'LMAYMIZ, balki butunlay chetlatamiz. Noto'g'ri
moslashtirilgan namuna hech qanday namunadan ko'ra yomonroq.

Ishga tushirish (loyiha ildizidan):
    pip install numpy soundfile
    python scripts/prepare_calls_for_colab.py

Natija: data/calls-colab/  →  audio/*.flac + train.csv + eval.csv + calls-colab.tar
"""

import json
import math
import os
import re
import sys
import tarfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "calls-dataset")
DST = os.path.join(ROOT, "data", "calls-colab")

SR = 16000
MAX_SEC = 30.0          # Whisper oynasi
MIN_SEC = 1.2           # bundan qisqa bo'lak shubhali
MIN_CHARS = 4
MIN_SILENCE = 0.30      # kesish uchun kerak bo'lgan eng qisqa jimlik
TEXT_TOL_SEC = 3.0      # matn chegarasi jimlikdan shuncha soniyagacha uzoqda bo'lishi mumkin
CPS_MIN, CPS_MAX = 6.0, 26.0   # belgi/soniya — moslikning bilvosita tekshiruvi
EVAL_CALL_FRACTION = 0.08      # eval uchun ajratiladigan QO'NG'IROQLAR ulushi

SENT_END = re.compile(r"[.!?…]+[\s ]+")


# ─────────────────────── jimlik qidirish ───────────────────────

def silence_mask(x, sr=SR, frame=0.025, hop=0.010):
    """Har bir freym uchun "jimmi?" bayrog'i qaytaradi."""
    n_f, n_h = int(frame * sr), int(hop * sr)
    if len(x) < n_f:
        return np.zeros(0, dtype=bool), n_h
    n = 1 + (len(x) - n_f) // n_h
    idx = np.arange(n_f)[None, :] + n_h * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx].astype(np.float64) ** 2, axis=1)) + 1e-10
    db = 20 * np.log10(rms)
    # Telefon liniyasida doimiy shovqin bor, shuning uchun chegara NISBIY:
    # nutq darajasidan 30 dB past, lekin shovqin polidan 6 dB baland.
    speech = np.percentile(db, 90)
    floor = np.percentile(db, 10)
    thr = max(floor + 6.0, speech - 30.0)
    return db < thr, n_h


def silence_runs(mask, hop_n, sr=SR):
    """[(boshlanish_sek, tugash_sek, davomiyligi_sek), ...]"""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        t0, t1 = i * hop_n / sr, j * hop_n / sr
        if t1 - t0 >= MIN_SILENCE:
            runs.append((t0, t1, t1 - t0))
        i = j
    return runs


def pick_cut(runs, target, half_window):
    """Maqsad vaqtiga eng yaqin, eng uzun jimlikning o'rtasini tanlaydi."""
    lo, hi = target - half_window, target + half_window
    cands = [r for r in runs if lo <= (r[0] + r[1]) / 2 <= hi]
    if not cands:
        return None
    best_len = max(c[2] for c in cands)
    # Eng uzunlaridan (80% dan yuqori) maqsadga eng yaqinini olamiz
    good = [c for c in cands if c[2] >= best_len * 0.8]
    return min(good, key=lambda c: abs((c[0] + c[1]) / 2 - target))


# ─────────────────────── matnni bo'lish ───────────────────────

def sentence_boundaries(text):
    """Matn ichidagi gap chegaralarining belgi indekslari."""
    return [m.end() for m in SENT_END.finditer(text)]


def split_text(text, ratios, dur):
    """Matnni berilgan vaqt nisbatlariga ko'ra gap chegaralaridan bo'ladi.

    Ishonchli chegara topilmasa None qaytaradi — namuna chetlatiladi.
    """
    bounds = sentence_boundaries(text)
    if not bounds:
        return None
    L = len(text)
    tol = TEXT_TOL_SEC * (L / dur)     # soniyani belgiga aylantiramiz
    cuts, prev = [], 0
    for r in ratios:
        target = r * L
        cand = [b for b in bounds if b > prev and abs(b - target) <= tol]
        if not cand:
            return None
        b = min(cand, key=lambda b: abs(b - target))
        cuts.append(b)
        prev = b
    pieces, start = [], 0
    for c in cuts + [L]:
        pieces.append(text[start:c].strip())
        start = c
    return pieces


# ─────────────────────── bitta namunani bo'lish ───────────────────────

def split_sample(x, text, dur):
    """[(audio, matn), ...] yoki None (ishonchsiz)."""
    if dur <= MAX_SEC:
        return [(x, text)]

    n = math.ceil(dur / MAX_SEC)          # nechta bo'lakka bo'lamiz
    mask, hop_n = silence_mask(x)
    runs = silence_runs(mask, hop_n)
    if not runs:
        return None

    half_window = min(4.0, dur / (2 * n))
    times, prev_t = [], 0.0
    for i in range(1, n):
        cut = pick_cut(runs, dur * i / n, half_window)
        if cut is None:
            return None
        t = (cut[0] + cut[1]) / 2
        if t <= prev_t:
            return None
        times.append(t)
        prev_t = t

    # Har bir bo'lak 30 soniyadan qisqa bo'lishi SHART
    edges = [0.0] + times + [dur]
    if any(edges[i + 1] - edges[i] > MAX_SEC for i in range(len(edges) - 1)):
        return None

    parts_text = split_text(text, [t / dur for t in times], dur)
    if parts_text is None:
        return None

    out = []
    for i in range(len(edges) - 1):
        seg_dur = edges[i + 1] - edges[i]
        seg_txt = parts_text[i]
        if seg_dur < MIN_SEC or len(seg_txt) < MIN_CHARS:
            return None
        cps = len(seg_txt) / seg_dur
        if not (CPS_MIN <= cps <= CPS_MAX):     # mos kelmadi → butun namunani tashlaymiz
            return None
        out.append((x[int(edges[i] * SR):int(edges[i + 1] * SR)], seg_txt))
    return out


# ─────────────────────── asosiy ───────────────────────

def main():
    rows = [json.loads(l) for l in open(os.path.join(SRC, "manifest.jsonl"))]
    print(f"Manba: {len(rows)} namuna, {sum(r['duration'] for r in rows)/3600:.2f} soat")

    os.makedirs(os.path.join(DST, "audio"), exist_ok=True)

    stats = {"whole": 0, "split_ok": 0, "split_pieces": 0, "rejected": 0, "read_fail": 0}
    produced = []          # (fayl nomi, matn, davomiylik, call_id)

    for k, r in enumerate(rows, 1):
        if k % 100 == 0:
            print(f"  {k}/{len(rows)} ...", flush=True)
        try:
            x, sr = sf.read(os.path.join(SRC, r["path"]), dtype="float32")
        except Exception:
            stats["read_fail"] += 1
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        dur = len(x) / sr

        pieces = split_sample(x, r["sentence"].strip(), dur)
        if pieces is None:
            stats["rejected"] += 1
            continue
        if dur <= MAX_SEC:
            stats["whole"] += 1
        else:
            stats["split_ok"] += 1
            stats["split_pieces"] += len(pieces)

        stem = os.path.splitext(os.path.basename(r["path"]))[0]
        for i, (seg, txt) in enumerate(pieces):
            name = f"{stem}.flac" if len(pieces) == 1 else f"{stem}_s{i+1}.flac"
            sf.write(os.path.join(DST, "audio", name), seg, SR, format="FLAC")
            produced.append((name, txt, len(seg) / SR, r["call_id"]))

    # --- Eval to'plamini QO'NG'IROQ bo'yicha ajratamiz ---
    # Bitta qo'ng'iroqning bo'laklari train va eval'ga bo'linib ketsa, eval
    # natijasi yolg'on yaxshi chiqadi (bir xil ovoz, bir xil mavzu).
    call_ids = sorted({c for *_, c in produced})
    rng = np.random.default_rng(42)
    rng.shuffle(call_ids)
    n_eval = max(12, int(len(call_ids) * EVAL_CALL_FRACTION))
    eval_calls = set(call_ids[:n_eval])

    def write_csv(path, rows_):
        with open(path, "w", encoding="utf-8") as f:
            f.write("path,sentence\n")
            for name, txt, _, _ in rows_:
                f.write('"audio/%s","%s"\n' % (name, txt.replace('"', '""')))

    train_rows = [p for p in produced if p[3] not in eval_calls]
    eval_rows = [p for p in produced if p[3] in eval_calls]
    write_csv(os.path.join(DST, "train.csv"), train_rows)
    write_csv(os.path.join(DST, "eval.csv"), eval_rows)

    hours = sum(p[2] for p in produced) / 3600
    report = {
        **stats,
        "samples_out": len(produced),
        "hours_out": round(hours, 2),
        "train_samples": len(train_rows),
        "train_hours": round(sum(p[2] for p in train_rows) / 3600, 2),
        "eval_samples": len(eval_rows),
        "eval_hours": round(sum(p[2] for p in eval_rows) / 3600, 2),
        "eval_calls": len(eval_calls),
        "calls_total": len(call_ids),
    }
    json.dump(report, open(os.path.join(DST, "report.json"), "w"), indent=2)

    print("\n" + "=" * 52)
    for k, v in report.items():
        print(f"  {k:<16}: {v}")
    print("=" * 52)

    # --- Colab'ga bitta fayl bo'lib ketishi uchun tar ---
    tar_path = os.path.join(DST, "calls-colab.tar")
    print(f"\nTar tayyorlanmoqda: {tar_path}")
    with tarfile.open(tar_path, "w") as tar:
        for n in ("train.csv", "eval.csv", "report.json"):
            tar.add(os.path.join(DST, n), arcname=n)
        tar.add(os.path.join(DST, "audio"), arcname="audio")
    print(f"Hajmi: {os.path.getsize(tar_path)/1e6:.0f} MB")
    print("\nShu faylni Google Drive'ga (MyDrive ildiziga) yuklang.")


if __name__ == "__main__":
    main()
