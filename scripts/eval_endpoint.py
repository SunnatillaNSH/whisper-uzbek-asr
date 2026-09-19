#!/usr/bin/env python3
"""WER'ni PRODUCTION ENDPOINT orqali o'lchaydi va har bir namunani saqlaydi.

NEGA ENDPOINT ORQALI. Avvalgi raqamlar ikki xil stack'da olingan edi:
baseline xom transformers `model.generate()` da (VAD va darvozalarsiz),
endpoint esa faster-whisper + VAD + temperatura zaxirasi bilan. Ularni
yonma-yon qo'yib bo'lmaydi. Endi barcha raqamlar bitta stack'da.

CHIQISH: har bir namunaning gipotezasi JSON'ga yoziladi. Ikki yurishni
keyin `compare_runs.py` juftlashgan bootstrap bilan solishtiradi —
namunalar bir xil bo'lgani uchun taqqoslash adolatli.

    python3 scripts/eval_endpoint.py --out analysis/eval120_prod.json
    python3 scripts/eval_endpoint.py --csv data/eval120/eval.csv --workers 4

Kalit ~/.config/sellup/proxy_token dan o'qiladi (qiymat hech qayerga
yozilmaydi). Proksi tokensiz so'rovlarni manba bo'yicha sanaydi, shuning
uchun User-Agent aniq berilgan.
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://ovoz-konsoli.sellup2.workers.dev/api"
UA = "asr-ml/1.0"                     # proksi statistikasida ajralib tursin
TOKEN_FILE = "~/.config/sellup/proxy_token"

# O'zbek lotin yozuvida apostrof HARFNING BIR QISMI (o', g'). Ma'lumotda uch
# xil variant aralash uchraydi va model yana boshqasini chiqarishi mumkin.
# Normallashtirmasak, so'zlarning 18.7% ida apostrof borligi uchun WER ~19
# foiz punktgacha buziladi. Apostrofning O'ZI saqlanadi — olib tashlansa
# "ozim" va "o'zim" qo'shilib ketadi.
APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}


def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def wer_counts(ref_words, hyp_words):
    """Levenshtein — (almashtirish, tushib qolish, qo'shish).

    jiwer o'rniga o'zimiz hisoblaymiz: skript hech qanday tashqi
    kutubxonaga bog'liq bo'lmaydi (vaqtinchalik venv tozalanib ketsa ham
    ishlaydi) va S/D/I taqsimoti bepul chiqadi.

    Xotira uchun ikki qatorli jadval: to'liq matritsa uzun korpusda
    gigabaytlarga chiqadi.
    """
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 0, 0, m
    prev = [(j, 0, 0, j) for j in range(m + 1)]        # (narx, S, D, I)
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cur[j] = prev[j - 1]
            else:
                sub = prev[j - 1]
                dele = prev[j]
                ins = cur[j - 1]
                best = min((sub[0], 1), (dele[0], 2), (ins[0], 3))
                if best[1] == 1:
                    cur[j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
                elif best[1] == 2:
                    cur[j] = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
                else:
                    cur[j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
        prev = cur
    _, S, Dl, I = prev[m]
    return S, Dl, I


def corpus_wer(refs, hyps):
    """Butun to'plam bo'yicha WER — so'zlar yig'indisi ustidan.

    Namunalar o'rtachasi EMAS: uzun namuna ko'proq vazn olishi kerak,
    aks holda bir so'zlik "Rahmat." butun suhbat bilan teng bo'lib qoladi.
    """
    S = Dl = I = N = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        s, d, i = wer_counts(rw, hw)
        S += s; Dl += d; I += i; N += len(rw)
    return {"wer": 100 * (S + Dl + I) / N if N else float("nan"),
            "sub": S, "del": Dl, "ins": I, "ref_words": N}


def token():
    p = os.path.expanduser(TOKEN_FILE)
    if not os.path.exists(p):
        sys.exit(f"Kalit topilmadi: {p}")
    return open(p).read().strip()


def req(url, data=None, tok=""):
    r = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"User-Agent": UA, "Authorization": "Bearer " + tok,
                 **({"Content-Type": "application/json"} if data else {})})
    with urllib.request.urlopen(r, timeout=180) as resp:
        return json.load(resp)


def transcribe(path, tok, opts, tries=2):
    """Bitta faylni endpointga yuboradi va matnni qaytaradi."""
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    for attempt in range(tries):
        try:
            job = req(f"{BASE}/run", json.dumps({"input": {**opts, "audio_base64": b64}}).encode(), tok)
            if "id" not in job:
                raise RuntimeError(job.get("error") or job)
            for _ in range(200):
                time.sleep(2)
                r = req(f"{BASE}/status/{job['id']}", None, tok)
                st = r.get("status")
                if st == "COMPLETED":
                    out = r.get("output") or {}
                    if out.get("status") != "success":
                        raise RuntimeError(out.get("message") or "noma'lum xato")
                    return out
                if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    raise RuntimeError(st)
            raise TimeoutError("status kutish tugadi")
        except Exception as e:
            if attempt + 1 >= tries:
                return {"error": str(e)}
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/eval120/eval.csv")
    ap.add_argument("--out", default="analysis/eval120_prod.json")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel so'rovlar; endpoint max workers dan oshmasin")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--vad", default="1")
    ap.add_argument("--hotwords", default="",
                    help="standart bo'sh — endpoint standarti ham shunday")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    tok = token()
    rows = list(csv.DictReader(open(os.path.join(ROOT, a.csv), encoding="utf-8")))
    if a.limit:
        rows = rows[: a.limit]
    opts = {"language": "uz", "beam_size": a.beam,
            "vad": a.vad not in ("0", "false"), "hotwords": a.hotwords}

    print(f"Namuna: {len(rows)} | {a.workers} ta parallel | beam {a.beam} "
          f"| vad {opts['vad']} | hotwords {'bor' if a.hotwords else 'yo`q'}")

    results = [None] * len(rows)
    lock = threading.Lock()
    done = [0]

    def work(i):
        r = rows[i]
        p = os.path.join(ROOT, "data", r["path"])
        out = transcribe(p, tok, opts)
        results[i] = {"path": r["path"], "reference": r["sentence"],
                      "hypothesis": out.get("text", ""),
                      "duration": out.get("duration_sec"),
                      "proc": out.get("processing_time_sec"),
                      "error": out.get("error")}
        with lock:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(rows):
                print(f"  {done[0]}/{len(rows)}", flush=True)

    idx = list(range(len(rows)))
    threads = []
    def runner():
        while True:
            with lock:
                if not idx:
                    return
                i = idx.pop(0)
            work(i)
    for _ in range(max(1, a.workers)):
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    ok = [r for r in results if r and not r["error"]]
    bad = [r for r in results if r and r["error"]]

    refs = [norm(r["reference"]) for r in ok]
    hyps = [norm(r["hypothesis"]) for r in ok]
    agg = corpus_wer(refs, hyps)

    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"csv": a.csv, "opts": opts, **agg, "samples": len(ok),
               "failed": len(bad), "results": results},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    tot = agg["sub"] + agg["del"] + agg["ins"] or 1
    print(f"\n  WER      : {agg['wer']:.2f}%")
    print(f"  S/D/I    : {agg['sub']} / {agg['del']} / {agg['ins']}"
          f"   ({100*agg['sub']//tot}% / {100*agg['del']//tot}% / {100*agg['ins']//tot}%)")
    print(f"  Namuna   : {len(ok)} ({len(bad)} xato)")
    print(f"  So'z     : {agg['ref_words']}")
    print(f"  -> {a.out}")
    if bad:
        print(f"\n  Xatolar (birinchi 3): {[b['error'] for b in bad[:3]]}")


if __name__ == "__main__":
    main()
