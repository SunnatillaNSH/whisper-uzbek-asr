#!/usr/bin/env python3
"""Uzun qo'ng'iroqlarni FORCED ALIGNMENT bilan ≤30 s bo'laklarga bo'ladi.

`prepare_calls_for_colab.py` kesish nuqtasini jimlik + gap chegarasi bo'yicha
TAXMIN qiladi. Ishonch bo'lmasa namunani chetlab o'tadi — shu sababli 843
namunadan 235 tasi (5.2 soat) ishlatilmay qolgan.

Bu skript taxmin qilmaydi. Qo'ng'iroqlarda o'qitilgan model audioni SO'Z
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
SEARCH_WIN = 4.0        # kesish nuqtasini maqsaddan shuncha soniya atrofida qidiramiz
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


def best_gap(words, target):
    """Maqsad vaqtiga yaqin ENG KATTA so'zlararo pauzani topadi.

    Kesishni pauzaga tushirish muhim: so'z o'rtasidan kesilsa, ikkala bo'lakda
    ham yarim so'z qoladi va model chalkashadi.
    """
    best = None
    for k in range(len(words) - 1):
        end, nxt = words[k]["end"], words[k + 1]["start"]
        mid = (end + nxt) / 2
        if abs(mid - target) > SEARCH_WIN:
            continue
        gap = max(0.0, nxt - end)
        score = gap - 0.05 * abs(mid - target)     # pauza kattaroq, maqsadga yaqinroq
        if best is None or score > best[0]:
            best = (score, k, mid)
    return best


def process(x, ref_text, words):
    """(audio, matn) bo'laklari yoki None."""
    dur = len(x) / SR
    if dur <= MAX_SEC:
        return [(x, ref_text)]
    if len(words) < MIN_WORDS:
        return None

    n = math.ceil(dur / MAX_SEC)
    hyp_n = [norm_word(w["word"]) for w in words]
    ref_raw = ref_text.split()
    ref_n = [norm_word(w) for w in ref_raw]
    anchors = build_anchors(hyp_n, ref_n)
    if len(anchors) < MIN_WORDS // 2:
        return None

    cut_times, cut_words, prev_t, prev_j = [], [], 0.0, 0
    for i in range(1, n):
        g = best_gap(words, dur * i / n)
        if g is None:
            return None
        _, k, t = g
        j = map_index(anchors, k)
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
    print(f"Chetlangan namunalar: {len(rows)} "
          f"({sum(r['duration'] for r in rows)/3600:.2f} soat)", flush=True)

    os.makedirs(os.path.join(OUT_DIR, "audio"), exist_ok=True)
    asr = pipeline("automatic-speech-recognition", model=MODEL_DIR,
                   torch_dtype=torch.float16, device="cuda", chunk_length_s=30)

    produced, stats = [], {"ok": 0, "rad": 0, "asr_xato": 0}
    for i, r in enumerate(rows, 1):
        if i % 20 == 0:
            print(f"  {i}/{len(rows)} | qabul {stats['ok']} | rad {stats['rad']}", flush=True)
        x, sr = sf.read(os.path.join(REJ_DIR, r["path"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        try:
            res = asr(x.copy(), return_timestamps="word",
                      generate_kwargs={"language": "uzbek", "task": "transcribe"})
            words = [w for w in res.get("chunks", [])
                     if w.get("timestamp") and w["timestamp"][0] is not None
                     and w["timestamp"][1] is not None]
            words = [{"word": w["text"], "start": w["timestamp"][0],
                      "end": w["timestamp"][1]} for w in words]
        except Exception as e:
            stats["asr_xato"] += 1
            continue

        pieces = process(x, str(r["sentence"]).strip(), words)
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
