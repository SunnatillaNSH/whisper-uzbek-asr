#!/usr/bin/env python3
"""Uzun qo'ng'iroqlarni FORCED ALIGNMENT bilan ≤30 s bo'laklarga bo'ladi.

`prepare_calls_for_colab.py` kesish nuqtasini jimlik + gap chegarasi bo'yicha
TAXMIN qiladi. Ishonch bo'lmasa namunani chetlab o'tadi — shu sababli 843
namunadan 235 tasi (5.2 soat) ishlatilmay qolgan.

Bu skript taxmin qilmaydi. Qo'ng'iroqlarda o'qitilgan model audioni SEGMENT
DARAJASIDAGI vaqt belgilari bilan transkripsiya qiladi, so'ng Muxlisa matni
shu gipotezaga so'zma-so'z tekislanadi (difflib). Natijada har bir kesish
nuqtasi uchun "audioning shu soniyasi matnning shu so'ziga to'g'ri keladi"
degan aniq moslik hosil bo'ladi.

Model mukammal emas (WER ~48%), lekin bu muhim emas: bizga uning MATNI emas,
VAQT BELGILARI kerak. Noto'g'ri tanilgan so'z ham to'g'ri joyda turadi.

    MODEL_DIR=/workspace/whisper-uz-calls-final \
    REJ_DIR=/workspace/rejected OUT_DIR=/workspace/recovered \
        python scripts/align_long_calls.py
"""

import difflib
import json
import math
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/whisper-uz-calls-final")
REJ_DIR = os.environ.get("REJ_DIR", "/workspace/rejected")
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/recovered")

SR = 16000
MAX_SEC = 30.0
MIN_SEC = 1.2
MIN_CHARS = 4
CPS_MIN, CPS_MAX = 6.0, 26.0
# SEARCH_WIN endi kerak emas: qidiruv oralig'i feasible_range() bilan
# hisoblanadi (30 soniyalik cheklovdan kelib chiqib), simmetrik oyna emas.
BATCH = int(os.environ.get("ASR_BATCH", "8"))
ANCHOR_TOL = 3          # tekislash langari kesish so'zidan shuncha so'z uzoqda bo'lishi mumkin
MIN_WORDS = 12          # bundan kam so'z tanilgan bo'lsa, tekislashga ishonmaymiz

APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}


def norm_word(w):
    w = str(w).translate(APOS).lower()
    return re.sub(r"[^\w']", "", w, flags=re.UNICODE)


def build_anchors(hyp_words, ref_words):
    """Gipoteza so'z indeksi → Muxlisa so'z indeksi mosliklari."""
    sm = difflib.SequenceMatcher(None, hyp_words, ref_words, autojunk=False)
    anchors = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            anchors.extend((i1 + k, j1 + k) for k in range(i2 - i1))
    return anchors


def map_index(anchors, i):
    """Gipoteza indeksini Muxlisa indeksiga o'giradi; ishonch yo'q bo'lsa None."""
    if not anchors:
        return None
    before = [a for a in anchors if a[0] <= i]
    after = [a for a in anchors if a[0] >= i]
    cand = []
    if before:
        ia, ja = before[-1]
        cand.append((abs(i - ia), ja + (i - ia)))
    if after:
        ib, jb = after[0]
        cand.append((abs(ib - i), jb - (ib - i)))
    dist, j = min(cand)
    return j if dist <= ANCHOR_TOL else None


def feasible_range(dur, i, n):
    """i-kesish uchun RUXSAT ETILGAN vaqt oralig'i.

    Har bir bo'lak 30 soniyadan qisqa bo'lishi SHART. 55 soniyalik namunani
    ikkiga bo'lsak, kesish 25-30 s oralig'ida bo'lishi kerak: 24 s da kesilsa
    o'ng bo'lak 31 s bo'lib, Whisper uni kesib tashlaydi, matn esa to'liq
    qoladi. Maqsad atrofida simmetrik oyna qidirish shu sababli xato —
    oynaning yarmi imkonsiz nuqtalarga tushadi va namuna keraksiz rad etiladi.
    """
    lo = dur - MAX_SEC * (n - i)
    hi = MAX_SEC * i
    return max(lo, 0.0), min(hi, dur)


def best_boundary(segs, target, lo, hi):
    """Maqsad vaqtiga eng yaqin SEGMENT chegarasini qaytaradi.

    Whisper segmentlarni tabiiy pauzalarda (gap oxiri, uzoq sukut) ajratadi —
    ya'ni segment chegarasi bizga kerak bo'lgan kesish nuqtasining o'zi.

    So'z darajasidagi vaqt belgilari (`return_timestamps="word"`) aniqroq,
    lekin ular cross-attention DTW talab qiladi va namunasiga ~50 soniya
    ketadi — 235 namuna uchun 3 soatdan ortiq GPU vaqti. Segment belgilari
    dekodlash paytida bepul chiqadi va bu vazifa uchun yetarli.
    """
    best = None
    for k in range(len(segs) - 1):
        t = segs[k]["end"]
        if t is None or not (lo <= t <= hi):
            continue
        d = abs(t - target)
        if best is None or d < best[0]:
            best = (d, k, t)
    return best


def process(x, ref_text, segs):
    """(audio, matn) bo'laklari yoki None.

    Bo'laklar sonini OSHIRIB ko'radi. 55 soniyalik faylni ikkiga bo'lganda
    ruxsat etilgan oraliq atigi 25-30 s — Whisper segmentlari 5-10 soniyalik
    bo'lgani uchun ko'pincha bu tor oynaga chegara tushmaydi va namuna rad
    etiladi. Uchga bo'lsak oraliq ancha kengayadi va bo'laklar ~18 s bo'ladi,
    bu trening uchun bemalol.
    """
    dur = len(x) / SR
    if dur <= MAX_SEC:
        return [(x, ref_text)]
    n_min = math.ceil(dur / MAX_SEC)
    for n in range(n_min, n_min + 3):
        if dur / n < MIN_SEC * 2:
            break
        out = _try_split(x, ref_text, segs, dur, n)
        if out is not None:
            return out
    return None


def _try_split(x, ref_text, segs, dur, n):

    # Gipoteza so'zlari va har bir segment chegarasidagi so'z indeksi
    hyp_n, seg_word_end = [], []
    for sg in segs:
        hyp_n.extend(norm_word(w) for w in str(sg["text"]).split())
        seg_word_end.append(len(hyp_n) - 1)
    if len(hyp_n) < MIN_WORDS:
        return None

    ref_raw = ref_text.split()
    ref_n = [norm_word(w) for w in ref_raw]
    anchors = build_anchors(hyp_n, ref_n)
    if len(anchors) < MIN_WORDS // 2:
        return None

    cut_times, cut_words, prev_t, prev_j = [], [], 0.0, 0
    for i in range(1, n):
        lo, hi = feasible_range(dur, i, n)
        lo = max(lo, prev_t + MIN_SEC)
        hi = min(hi, prev_t + MAX_SEC)     # oldingi kesishdan 30 s dan uzoq emas
        if lo > hi:
            return None
        g = best_boundary(segs, dur * i / n, lo, hi)
        if g is None:
            return None
        _, k, t = g
        j = map_index(anchors, seg_word_end[k])
        if j is None or t <= prev_t or j <= prev_j:
            return None
        cut_times.append(t)
        cut_words.append(j + 1)          # shu so'zdan KEYIN kesamiz
        prev_t, prev_j = t, j

    edges = [0.0] + cut_times + [dur]
    if any(edges[i + 1] - edges[i] > MAX_SEC for i in range(len(edges) - 1)):
        return None

    bounds = [0] + cut_words + [len(ref_raw)]
    out = []
    for i in range(len(edges) - 1):
        seg_dur = edges[i + 1] - edges[i]
        seg_txt = " ".join(ref_raw[bounds[i]:bounds[i + 1]]).strip()
        if seg_dur < MIN_SEC or len(seg_txt) < MIN_CHARS:
            return None
        if not (CPS_MIN <= len(seg_txt) / seg_dur <= CPS_MAX):
            return None
        out.append((x[int(edges[i] * SR):int(edges[i + 1] * SR)], seg_txt))
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(REJ_DIR, "rejected.jsonl"))]
    limit = int(os.environ.get("LIMIT", "0"))
    if limit:
        rows = rows[:limit]
        print(f"⚠️  LIMIT={limit} — faqat sinov uchun", flush=True)
    print(f"Chetlangan namunalar: {len(rows)} "
          f"({sum(r['duration'] for r in rows)/3600:.2f} soat)", flush=True)

    os.makedirs(os.path.join(OUT_DIR, "audio"), exist_ok=True)
    asr = pipeline("automatic-speech-recognition", model=MODEL_DIR,
                   torch_dtype=torch.float16, device="cuda", chunk_length_s=30)

    produced, stats = [], {"ok": 0, "rad": 0, "asr_xato": 0}
    for i, r in enumerate(rows, 1):
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} | qabul {stats['ok']} | rad {stats['rad']}", flush=True)
        x, sr = sf.read(os.path.join(REJ_DIR, r["path"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        try:
            res = asr(x.copy(), return_timestamps=True, batch_size=BATCH,
                      generate_kwargs={"language": "uzbek", "task": "transcribe"})
            segs = [{"text": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]}
                    for c in res.get("chunks", [])
                    if c.get("timestamp") and c["timestamp"][1] is not None]
        except Exception as e:
            stats["asr_xato"] += 1
            continue

        pieces = process(x, str(r["sentence"]).strip(), segs)
        if pieces is None:
            stats["rad"] += 1
            continue
        stats["ok"] += 1
        stem = os.path.splitext(os.path.basename(r["path"]))[0]
        for k, (seg, txt) in enumerate(pieces):
            name = f"{stem}.flac" if len(pieces) == 1 else f"{stem}_a{k+1}.flac"
            sf.write(os.path.join(OUT_DIR, "audio", name), seg, SR, format="FLAC")
            produced.append({"path": f"audio/{name}", "sentence": txt,
                             "duration": len(seg) / SR, "call_id": r["call_id"]})

    with open(os.path.join(OUT_DIR, "recovered.jsonl"), "w") as f:
        for p in produced:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    hours = sum(p["duration"] for p in produced) / 3600
    print("\n" + "=" * 54)
    print(f"  Manba namunalar      : {len(rows)}")
    print(f"  Tekislandi           : {stats['ok']}")
    print(f"  Rad etildi           : {stats['rad']}")
    print(f"  ASR xatosi           : {stats['asr_xato']}")
    print(f"  Hosil bo'lgan bo'lak : {len(produced)}  ({hours:.2f} soat)")
    print("=" * 54)


if __name__ == "__main__":
    main()
