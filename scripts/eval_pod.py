#!/usr/bin/env python3
"""Eval-120 ni POD ustida o'lchaydi — production bilan AYNI dekodlashda.

NEGA POD'DA. `eval_endpoint.py` production endpointiga so'rov yuboradi va
har yurish pul turadi; endpoint byudjeti esa qat'iy. Round 5 modelini
o'lchash uchun uni endpointga qo'yish ham kerak bo'lardi, ya'ni ishlab
turgan tizimga tegish. Pod'da model CT2 ga o'girilib, o'sha yerda
o'lchanadi: endpoint xarajati $0, production'ga xavf yo'q.

NEGA SOZLAMALAR TEKSHIRILADI. Bu skript raqami `analysis/eval120_prod.json`
(36.39%) bilan solishtiriladi. Agar dekodlash bir xil bo'lmasa,
taqqoslash modelni emas, sozlamalarni o'lchaydi — va farq qaysi biridan
kelganini bilib bo'lmaydi. `handler.py` dan nusxa ko'chirish ajralib
ketishga olib keladi, alohida modulga ajratish esa Dockerfile'ni
o'zgartirishni talab qiladi (`COPY handler.py .` — yangi fayl obrazga
tushmaydi, keyingi qurishda konteyner ko'tarilmay qoladi).

Shuning uchun uchinchi yo'l: sozlamalar shu yerda yoziladi, lekin skript
`handler.py` manbasini `ast` bilan o'qib, `model.transcribe(...)`
chaqiruvidagi qiymatlar bilan solishtiradi. Mos kelmasa — TO'XTAYDI.

    python3 scripts/eval_pod.py --check-only          # GPU kerak emas
    python3 scripts/eval_pod.py --model /workspace/r5-ct2 \
        --csv data/eval120/eval.csv --out analysis/eval120_r5.json
"""

import argparse
import ast
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Sinov uchun almashtirsa bo'ladi (ASR_HANDLER) — tekshiruvning o'zi
# tekshirilmasa, u faqat "hammasi joyida" deyishni biladi.
HANDLER = os.environ.get("ASR_HANDLER") or os.path.join(ROOT, "handler.py")
SAMPLING_RATE = 16000

# handler.py dagi qiymatlar. O'zgartirilsa, quyidagi tekshiruv qulab tushadi —
# bu ataylab: raqamlar solishtirilishi uchun ikkalasi bir xil bo'lishi SHART.
DECODE = dict(
    language="uz",
    task="transcribe",
    beam_size=5,
    hotwords=None,
    initial_prompt=None,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=2000, speech_pad_ms=400),
    condition_on_previous_text=False,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
)
COMPUTE_TYPE = "float16"

APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}


def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


# ──────────────────── handler.py bilan mosligini tekshirish ────────────────────

def _constants(node, names=None):
    """Ifoda ichidagi barcha o'zgarmas qiymatlar.

    Uchta holatni hisobga oladi:
      * `int(os.environ.get("VAD_MIN_SILENCE_MS", "2000"))` — standart qiymat
        daraxt ichida satr bo'lib turadi, literal_eval esa ishlamaydi;
      * `-1.0` — bu Constant emas, UnaryOp(USub, Constant(1.0)), ya'ni
        minussiz o'qilsa qiymat teskari chiqadi;
      * `int(inp.get("beam_size", BEAM_SIZE))` — standart modul darajasidagi
        NOMDA, uning o'zi esa boshqa joyda e'lon qilingan.
    """
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            for v in _constants(n.operand, names):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(-v)
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str, bool)):
            out.append(n.value)
        elif isinstance(n, ast.Name) and names and n.id in names:
            out.extend(names[n.id])
    return out


def _module_names(tree):
    """Modul darajasidagi `NOM = ifoda` larning ichidagi o'zgarmaslar."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = _constants(n.value)
    return out


def _transcribe_kwargs(src):
    tree = ast.parse(src)
    names = _module_names(tree)
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "transcribe"):
            return {k.arg: k.value for k in n.keywords if k.arg}, names
    sys.exit("handler.py da model.transcribe(...) chaqiruvi topilmadi")


def check_matches_handler(verbose=True):
    """handler.py dagi dekodlash bilan mosligini tasdiqlaydi."""
    if not os.path.exists(HANDLER):
        sys.exit(f"{HANDLER} topilmadi — moslikni tekshirib bo'lmaydi")
    kw, names = _transcribe_kwargs(open(HANDLER, encoding="utf-8").read())

    want = {
        "beam_size": 5,
        "condition_on_previous_text": False,
        "temperature": DECODE["temperature"],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }
    bad = []
    for key, val in want.items():
        if key not in kw:
            bad.append(f"{key}: handler.py da yo'q")
            continue
        got = _constants(kw[key], names)
        need = val if isinstance(val, list) else [val]
        # `int(os.environ.get("ASR_BEAM_SIZE", "5"))` -> "5" satr bo'lib keladi
        norm_got = {str(x) for x in got}
        if not all(str(v) in norm_got for v in need):
            bad.append(f"{key}: handler.py da {got}, bu yerda {val}")

    # VAD alohida — ichma-ich dict
    if "vad_parameters" in kw:
        got = {str(x) for x in _constants(kw["vad_parameters"], names)}
        for k, v in DECODE["vad_parameters"].items():
            if str(v) not in got:
                bad.append(f"vad_parameters.{k}: handler.py da {v} yo'q ({sorted(got)})")
    else:
        bad.append("vad_parameters: handler.py da yo'q")

    # hotwords standarti bo'sh bo'lishi shart (o'lchangan: yoqiq bo'lsa +3.9 punkt)
    src = open(HANDLER, encoding="utf-8").read()
    m = re.search(r'HOTWORDS\s*=\s*os\.environ\.get\(\s*"ASR_HOTWORDS"\s*,\s*"([^"]*)"\s*\)', src)
    if not m:
        bad.append("HOTWORDS standarti o'qilmadi")
    elif m.group(1) != "":
        bad.append(f"HOTWORDS standarti bo'sh emas: {m.group(1)!r}")

    if bad:
        print("DEKODLASH MOS EMAS — o'lchov modelni emas, sozlamani o'lchagan bo'lardi:",
              file=sys.stderr)
        for b in bad:
            print(f"  * {b}", file=sys.stderr)
        sys.exit(1)
    if verbose:
        print("Dekodlash handler.py bilan mos — beam 5, VAD 2000/400, "
              "hotwords bo'sh, temp zaxirasi, condition_on_previous_text=False")
    return True


# ──────────────────────────── WER ────────────────────────────

def wer_counts(ref, hyp):
    """Levenshtein -> (almashtirish, tushish, qo'shish). Ikki qatorli jadval."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
            else:
                sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
                best = min((sub[0], 1), (dele[0], 2), (ins[0], 3))
                if best[1] == 1:
                    cur[j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
                elif best[1] == 2:
                    cur[j] = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
                else:
                    cur[j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
        prev = cur
    _, S, D, I = prev[m]
    return S, D, I


# handler.py dagi _clean bilan bir xil bo'lishi kerak.
HALLUCINATIONS = [
    re.compile(r"^\W*(obuna bo['‘’]?ling[^.!?]*)[.!?]?\W*$", re.I),
    re.compile(r"^\W*(subscribe|thanks? for watching)[^.!?]*[.!?]?\W*$", re.I),
    re.compile(r"^\W*(продолжение следует)[^.!?]*[.!?]?\W*$", re.I),
]


def clean(text):
    parts, out, prev = re.split(r"(?<=[.!?])\s+", text), [], None
    for p in parts:
        p = p.strip()
        if not p or any(rx.match(p) for rx in HALLUCINATIONS):
            continue
        if prev is not None and p.lower() == prev.lower():
            continue
        out.append(p)
        prev = p
    return " ".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="faqat handler.py bilan moslikni tekshiradi, GPU kerak emas")
    ap.add_argument("--model", help="CT2 model katalogi")
    ap.add_argument("--csv", default="data/eval120/eval.csv")
    ap.add_argument("--out", default="analysis/eval120_pod.json")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    check_matches_handler()
    if a.check_only:
        return
    if not a.model:
        sys.exit("--model kerak")

    import csv as _csv
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    rows = list(_csv.DictReader(open(os.path.join(ROOT, a.csv), encoding="utf-8")))
    if a.limit:
        rows = rows[: a.limit]
    print(f"Model      : {a.model} ({COMPUTE_TYPE})")
    print(f"Namuna     : {len(rows)}")
    model = WhisperModel(a.model, device="cuda", compute_type=COMPUTE_TYPE)

    results = []
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        path = os.path.join(a.data_root, r["path"])
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLING_RATE:
            k = int(round(len(audio) * SAMPLING_RATE / sr))
            audio = np.interp(np.linspace(0, len(audio) - 1, k),
                              np.arange(len(audio)), audio).astype(np.float32)
        st = time.time()
        segs, _ = model.transcribe(np.ascontiguousarray(audio, dtype=np.float32),
                                   **DECODE)
        text = clean(" ".join(s.text.strip() for s in segs))
        results.append({"path": r["path"], "reference": r["sentence"],
                        "hypothesis": text,
                        "duration": round(len(audio) / SAMPLING_RATE, 2),
                        "proc": round(time.time() - st, 2), "error": None})
        if n % 25 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

    S = D = I = N = 0
    for r in results:
        s, d, i = wer_counts(norm(r["reference"]).split(), norm(r["hypothesis"]).split())
        S += s; D += d; I += i; N += len(norm(r["reference"]).split())
    wer = 100 * (S + D + I) / N if N else float("nan")

    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"csv": a.csv, "model": a.model, "opts": "handler.py bilan mos",
               "wer": wer, "sub": S, "del": D, "ins": I, "ref_words": N,
               "samples": len(results), "failed": 0, "results": results},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    tot = S + D + I or 1
    print(f"\n  WER      : {wer:.2f}%")
    print(f"  S/D/I    : {S} / {D} / {I}   ({100*S//tot}% / {100*D//tot}% / {100*I//tot}%)")
    print(f"  So'z     : {N}")
    print(f"  Vaqt     : {time.time() - t0:.0f} s")
    print(f"  -> {a.out}")
    print(f"\n  Taqqoslash: python3 scripts/compare_runs.py "
          f"analysis/eval120_prod.json {a.out}")


if __name__ == "__main__":
    main()
