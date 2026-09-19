#!/usr/bin/env python3
"""`calls-rejected` dagi yorliqli audioni ≤30 s bo'laklarga tiklaydi.

MUAMMO. 235 ta qo'ng'iroq bo'lagi (5.19 soat) yorliqli, lekin treningga
kirmaydi: mediana 55 s, ya'ni Whisper'ning 30 soniyalik oynasiga sig'maydi.
`prepare_calls_for_colab.py` ularni kesa olmadi — u jimlik va jumla
chegarasi mos tushgan joyni qidiradi, bu fayllarda esa mos tushmagan.

YECHIM. Vaqtni modelning O'ZIDAN olamiz, matnni esa yorliqdan:
  1. faster-whisper `word_timestamps=True` bilan audioni o'giradi —
     gipoteza so'zlari va ularning vaqtlari.
  2. Gipoteza so'zlari yorliq so'zlariga tekislanadi (Levenshtein).
     AYNAN MOS TUSHGAN so'zlargina "langar" bo'ladi: ularning vaqti
     yorliqdagi o'sha so'zga o'tkaziladi.
  3. Kesim faqat yorliqning JUMLA chegarasida va faqat ikkala uchi ham
     langarlangan joyda qilinadi.

Ya'ni yorliq matni O'ZGARMAYDI — biz unga vaqt biriktiramiz, xolos.
Langarsiz joyda kesish taxmin bo'lardi, taxmin esa noto'g'ri yorliq
yaratadi va bu treningga zarardan boshqa narsa bermaydi.

NEGA POD USTIDA, ENDPOINT ORQALI EMAS. Production `handler.py` so'z
vaqtlarini umuman qaytarmaydi (`word_timestamps` unda yo'q). Endpoint
yo'li obrazni qayta qurish va serverless'ni yangilashni talab qiladi,
ya'ni ishlab turgan tizimga tegishni. Pod'da faster-whisper to'g'ridan
chaqiriladi — o'sha narx, production'ga xavf yo'q.

ESLATMA: forced alignment bir marta tashlab yuborilgan, lekin sabab
boshqa edi — lokal transformers yo'lida SEGMENT vaqtlari buzilgan edi
(bitta "segment"da 97-314 so'z). SO'Z vaqtlari o'shanda ham to'g'ri
chiqqandi, va bu skript faqat so'z vaqtlarini ishlatadi.

    python3 scripts/recover_rejected.py --selftest        # GPU kerak emas
    python3 scripts/recover_rejected.py --model /workspace/ct2-model \
        --src data/calls-rejected --out data/calls-recovered
"""

import argparse
import json
import os
import re
import sys

# Sifat darvozalari — `build_weighted.py` dagilar bilan bir xil bo'lishi shart,
# aks holda bu yerda o'tgan bo'lak keyingi bosqichda baribir tashlanadi.
MAX_DUR = 30.0
MIN_DUR = 1.2
CPS_LO, CPS_HI = 6.0, 26.0
MIN_CHARS = 4

APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}


def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def sentences(text):
    """Matnni jumlalarga bo'ladi — (so'z_boshi, so'z_oxiri, matn).

    Chegara `.`, `!`, `?` dan keyin. Kesim faqat shu chegaralarda bo'ladi:
    jumla o'rtasidan kesilgan yorliq model uchun tugallanmagan gap bo'lib
    ko'rinadi va u tugatishni "o'ylab topishga" o'rganadi.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    out, i = [], 0
    for p in parts:
        n = len(p.split())
        if n:
            out.append((i, i + n - 1, p))
            i += n
    return out


def anchors(ref_words, hyp_words, hyp_times):
    """ref indeksi -> vaqt, faqat AYNAN mos tushgan so'zlar uchun.

    To'liq Levenshtein matritsasi + orqaga yurish. Faqat tenglik bo'yicha
    yurilgan qadamlar langar beradi; almashtirish bergan juftlik
    ishlatilmaydi, chunki u boshqa so'zning vaqti bo'lishi mumkin.
    """
    n, m = len(ref_words), len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]))
    out = {}
    i, j = n, m
    while i > 0 and j > 0:
        if ref_words[i - 1] == hyp_words[j - 1] and d[i][j] == d[i - 1][j - 1]:
            out[i - 1] = hyp_times[j - 1]
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def plan(text, hyp_words, hyp_times, duration):
    """Jumlalarni ≤30 s bo'laklarga yig'adi. -> (bo'laklar, sabablar)."""
    ref_words = norm(text).split()
    anc = anchors(ref_words, [norm(w) for w in hyp_words], hyp_times)
    sents = sentences(text)
    chunks, why = [], []

    cur = []          # yig'ilayotgan jumlalar
    for s in sents:
        trial = cur + [s]
        a, b = trial[0][0], trial[-1][1]
        ta = anc.get(a)
        tb = anc.get(b)
        # Ikkala uchi ham langarlangan bo'lsa va 30 s ga sig'sa — davom etamiz
        if ta is not None and tb is not None and (tb[1] - ta[0]) <= MAX_DUR:
            cur = trial
            continue
        # Sig'maydi: shu paytgacha yig'ilganini yopamiz
        if cur:
            c = close(cur, anc, why)
            if c:
                chunks.append(c)
        # Yangi bo'lak shu jumladan boshlanadi; o'zi 30 s dan uzun bo'lsa
        # uni bo'lib bo'lmaydi — jumla ichida kesish yorliqni buzadi.
        a2, b2 = s[0], s[1]
        ta2, tb2 = anc.get(a2), anc.get(b2)
        if ta2 is not None and tb2 is not None and (tb2[1] - ta2[0]) > MAX_DUR:
            why.append(f"jumla o'zi {tb2[1]-ta2[0]:.1f}s — bo'linmaydi")
            cur = []
        else:
            cur = [s]
    if cur:
        c = close(cur, anc, why)
        if c:
            chunks.append(c)
    return chunks, why


def close(group, anc, why):
    """Yig'ilgan jumlalarni bo'lakka aylantiradi, darvozalardan o'tkazib."""
    a, b = group[0][0], group[-1][1]
    ta, tb = anc.get(a), anc.get(b)
    if ta is None or tb is None:
        why.append("uchi langarlanmagan")
        return None
    start, end = ta[0], tb[1]
    dur = end - start
    text = " ".join(g[2] for g in group)
    if dur > MAX_DUR:
        why.append(f"uzun ({dur:.1f}s)")
        return None
    if dur < MIN_DUR:
        why.append(f"qisqa ({dur:.1f}s)")
        return None
    if len(text) < MIN_CHARS:
        why.append("matn kalta")
        return None
    cps = len(text) / dur
    if not (CPS_LO <= cps <= CPS_HI):
        why.append(f"cps {cps:.1f}")
        return None
    return {"start": round(start, 2), "end": round(end, 2),
            "duration": round(dur, 2), "cps": round(cps, 1), "sentence": text}


# ──────────────────────────── O'z-o'zini sinash ────────────────────────────

def selftest():
    """GPU va audiosiz: rejalashtiruvchi mantiqini tekshiradi."""
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'XATO'} {name} {extra}")
        ok = ok and cond

    # 1. Ideal holat: gipoteza yorliq bilan aynan bir xil, har so'z 0.5 s
    text = "Birinchi jumla shu yerda. Ikkinchi jumla bu yerda. Uchinchi jumla esa mana."
    w = norm(text).split()
    t = [(i * 0.5, i * 0.5 + 0.45) for i in range(len(w))]
    ch, why = plan(text, w, t, len(w) * 0.5)
    check("ideal: bitta bo'lak", len(ch) == 1, f"({len(ch)}, {why})")
    check("ideal: matn to'liq", ch and norm(ch[0]["sentence"]) == norm(text))

    # 2. Uzun fayl: 60 so'z, 0.6 s/so'z = 36 s -> 30 s ga sig'maydi.
    #    Tezligi realistik (cps ~12), ya'ni darvozalar emas, uzunlik bo'ladi.
    long_text = " ".join(f"Yettinchi qatorda turgan sakkizta narsa {i}."
                         for i in range(12))
    lw = norm(long_text).split()
    lt = [(i * 0.6, i * 0.6 + 0.55) for i in range(len(lw))]
    ch2, why2 = plan(long_text, lw, lt, len(lw) * 0.6)
    check("uzun: bir nechta bo'lak", len(ch2) >= 2, f"({len(ch2)}, {why2})")
    check("uzun: hammasi <= 30s", all(c["duration"] <= MAX_DUR for c in ch2),
          f"({[c['duration'] for c in ch2]})")
    check("uzun: bo'laklar kesishmaydi",
          all(ch2[i]["end"] <= ch2[i + 1]["start"] for i in range(len(ch2) - 1)))

    # 3. Matn saqlanishi: bo'laklar birlashtirilsa asl matn chiqsin
    joined = norm(" ".join(c["sentence"] for c in ch2))
    check("uzun: matn yo'qolmagan", joined == norm(long_text),
          "" if joined == norm(long_text) else f"\n       {joined[:80]!r}")

    # 4. Gipoteza butunlay boshqa -> langar yo'q -> hech narsa qaytmasin
    bad = ["xxx"] * len(w)
    ch3, why3 = plan(text, bad, t, len(w) * 0.5)
    check("langarsiz: bo'lak yo'q", len(ch3) == 0, f"({why3})")

    # 5. Qisman langar: gipotezaning har ikkinchi so'zi buzilgan
    half = list(w)
    for i in range(0, len(half), 2):
        half[i] = "xxx"
    ch4, _ = plan(text, half, t, len(w) * 0.5)
    check("qisman langar: darvozalar baribir hurmat qilinadi",
          all(MIN_DUR <= c["duration"] <= MAX_DUR for c in ch4))

    # 6. cps darvozasi: uzun jimlik ustida qisqa matn (sekin) tashlansin
    slow = [(i * 3.0, i * 3.0 + 0.2) for i in range(len(w))]
    ch5, why5 = plan(text, w, slow, len(w) * 3.0)
    check("cps darvozasi (sekin matn)", len(ch5) == 0, f"({set(why5)})")

    # 7. Darvozalar build_weighted.py bilan bir xil bo'lsin
    check("darvozalar mos", (MAX_DUR, MIN_DUR, CPS_LO, CPS_HI) == (30.0, 1.2, 6.0, 26.0))

    print("\n" + ("HAMMASI O'TDI" if ok else "XATOLAR BOR"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="GPU'siz mantiq sinovi")
    ap.add_argument("--model", help="CT2 model katalogi (Pod'da)")
    ap.add_argument("--src", default="data/calls-rejected")
    ap.add_argument("--out", default="data/calls-recovered")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-calls", default="data/eval_calls_120.json")
    ap.add_argument("--csv", default="", help="qo'shimcha ravishda train formatida CSV yozadi")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not a.model:
        sys.exit("--model kerak (yoki --selftest)")

    from faster_whisper import WhisperModel      # faqat Pod'da kerak
    import soundfile as sf

    model = WhisperModel(a.model, device="cuda", compute_type="float16")
    rows = [json.loads(l) for l in open(os.path.join(a.src, "rejected.jsonl"))]

    # eval-120 qo'ng'iroqlari SHU YERDA chiqariladi, treningdan oldin emas.
    # O'lchandi: hovuzdagi 136 qo'ng'iroqdan 32 tasi eval-120 da (0.98 soat,
    # 52 bo'lak). Ularni tiklash o'lchov suhbatlarini treningga olib kirardi.
    # `train_calls.py` dagi assert oxirgi to'siq, birinchisi emas: u yerga
    # yetib borgan sizish butun yurishni to'xtatadi.
    ev_path = os.path.join(ROOT, a.eval_calls)
    if not os.path.exists(ev_path):
        sys.exit(f"{ev_path} topilmadi — eval-120 ni chetlab bo'lmaydi.")
    meta = json.load(open(ev_path))
    ev120 = ({str(c["call_id"]) for c in meta.get("calls", [])}
             | {str(x) for x in meta.get("locked_from_previous", [])})
    before = len(rows)
    rows = [r for r in rows if str(r["call_id"]) not in ev120]
    print(f"eval-120 chetlandi: {before - len(rows)} bo'lak, qoldi {len(rows)}")

    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)

    kept, man, skipped = 0, [], {}
    for n, r in enumerate(rows, 1):
        path = os.path.join(a.src, r["path"])
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        segs, _ = model.transcribe(
            audio, language="uz", task="transcribe", beam_size=5,
            word_timestamps=True, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=2000, speech_pad_ms=400),
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        words, times = [], []
        for s in segs:
            for wd in (s.words or []):
                words.append(wd.word.strip())
                times.append((wd.start, wd.end))

        chunks, why = plan(r["sentence"], words, times, float(r["duration"]))
        for k, c in enumerate(chunks, 1):
            stem = os.path.splitext(os.path.basename(r["path"]))[0]
            rel = f"audio/{stem}_r{k}.flac"
            sf.write(os.path.join(a.out, rel),
                     audio[int(c["start"] * sr):int(c["end"] * sr)], sr)
            man.append({"path": rel, "sentence": c["sentence"],
                        "duration": c["duration"], "call_id": r["call_id"]})
            kept += 1
        if not chunks:
            for w in why:
                k = w.split(" (")[0]
                skipped[k] = skipped.get(k, 0) + 1
        if n % 25 == 0:
            print(f"  {n}/{len(rows)} | bo'lak {kept}", flush=True)

    with open(os.path.join(a.out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in man:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    if a.csv:
        import csv as _csv
        cp = os.path.join(ROOT, a.csv) if not os.path.isabs(a.csv) else a.csv
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, "w", newline="", encoding="utf-8") as f:
            wr = _csv.writer(f)
            wr.writerow(["path", "sentence"])
            for m in man:
                wr.writerow([f"{os.path.basename(a.out)}/{m['path']}", m["sentence"]])
        print(f"  CSV      : {a.csv}")

    hours = sum(m["duration"] for m in man) / 3600
    print(f"\n  Kirish   : {len(rows)} bo'lak")
    print(f"  Tiklandi : {kept} namuna, {hours:.2f} soat")
    print(f"  Tashlandi: {dict(sorted(skipped.items(), key=lambda x: -x[1]))}")
    print(f"  -> {a.out}/manifest.jsonl")


if __name__ == "__main__":
    main()
