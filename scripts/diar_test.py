#!/usr/bin/env python3
"""TZ-008: diarizatsiya + sotuvchi enrollment sinovi (Round 6 Pod'ida, trening TUGAGANDAN KEYIN).

Tartib (Jev A-005: AVVAL matn, KEYIN so'zlarni gapiruvchiga biriktirish):
  ASR (round6 CT2, handler.py bilan AYNAN bir dekodlash) word_timestamps=True
  -> pyannote 3.1 (num_speakers=2 qo'ng'iroq / 1..8 boshqa audio) + gapiruvchi embedding
  -> sotuvchi izi (enrollment) -> rol -> segments [{start,end,speaker,role,text}]

Rejimlar:
  prep   (Mac, stdlib)  data/round6-diar/ ni yig'adi: to'liq qo'ng'iroq wav + calls.json (D1 SELECT: seller_id).
  run    (Pod)          sinovni yuritadi, natija: <out>/segments/*.json, <out>/summary.json
  report (ixtiyoriy)    summary.json ni qisqa chop etadi.

Maxfiylik: repo OCHIQ. data/ gitignore'da (data/*). Audio va matn commit qilinmaydi.
HF token faqat HF_TOKEN muhit o'zgaruvchisidan (Pod'da /root/.hf_token fayli); hech qayerga yozilmaydi.
"""
import argparse, json, os, re, subprocess, sys, time, wave, random, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAR = os.path.join(ROOT, "data", "round6-diar")
CF = "/Users/macbookuz/code/cf-call-analyzer"
N_TEST, N_ENROLL_PER_SELLER = 30, 12
PYANNOTE = "pyannote/speaker-diarization-3.1"


# ------------------------------------------------------------------ prep (Mac)
def _d1(sql):
    r = subprocess.run(["npx", "wrangler", "d1", "execute", "qongiroq-tahlili-db", "--remote",
                        "--json", "--command", sql], cwd=CF, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-400:])
    return json.loads(r.stdout)[0]["results"]


def _parts(call_id):
    """raw55/raw28 dagi bo'laklar (ID.wav yoki ID_pN.wav) -> tartiblangan ro'yxat."""
    out = []
    for sub in ("raw55", "raw28"):
        d = os.path.join(ROOT, "data", "round6-dataset", sub, "audio")
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = re.fullmatch(rf"{call_id}(?:_p(\d+))?\.wav", f)
            if m:
                out.append((int(m.group(1) or 0), os.path.join(d, f)))
        if out:
            break
    return [p for _, p in sorted(out)]


def _concat(paths, dst):
    w0 = None
    with wave.open(dst, "wb") as o:
        for p in paths:
            with wave.open(p) as w:
                if w0 is None:
                    w0 = w.getparams(); o.setparams(w0)
                assert w.getframerate() == 16000 and w.getnchannels() == 1
                o.writeframes(w.readframes(w.getnframes()))
    return o  # noqa


def prep():
    rd = os.path.join(ROOT, "data")
    ev50 = json.load(open(os.path.join(rd, "eval_calls_50.json")))["calls"]
    ev120 = json.load(open(os.path.join(rd, "eval_calls_120.json")))
    ev120 = set(map(str, ev120["calls"] if isinstance(ev120, dict) else ev120))
    manifest = [json.loads(l) for l in open(os.path.join(rd, "round6-dataset", "manifest.jsonl"))]
    train_calls = sorted({str(m["call_id"]) for m in manifest if m.get("split") == "train"} - ev120)

    cand = [str(c) for c in ev50] + train_calls
    rows = _d1("SELECT db_call_id, seller_id, direction, duration FROM moizvonki_calls "
               f"WHERE db_call_id IN ({','.join(repr(c) for c in cand)})")
    meta = {str(r["db_call_id"]): r for r in rows}
    rng = random.Random(20261006)
    os.makedirs(os.path.join(DIAR, "audio"), exist_ok=True)
    calls = {}

    def add(cid, role):
        ps = _parts(cid)
        if cid not in meta or not ps:
            return False
        _concat(ps, os.path.join(DIAR, "audio", f"{cid}.wav"))
        m = meta[cid]
        calls[cid] = {"role": role, "seller": str(m["seller_id"]), "direction": m["direction"],
                      "duration": m["duration"], "parts": len(ps)}
        return True

    # test: eval-50 qo'ng'iroqlari, sotuvchi bo'yicha muvozanatli, 30 tagacha
    by_s = collections.defaultdict(list)
    for c in ev50:
        c = str(c)
        if c in meta:
            by_s[str(meta[c]["seller_id"])].append(c)
    for v in by_s.values():
        rng.shuffle(v)
    n = 0
    while n < N_TEST and any(by_s.values()):
        for s in sorted(by_s):
            if by_s[s] and n < N_TEST and add(by_s[s].pop(), "test"):
                n += 1
    # enrollment: train'dan (eval-120 dan tashqarida), har sotuvchi N ta, davomiyligi 60..300 s
    by_s = collections.defaultdict(list)
    for c in train_calls:
        if c in meta and 60 <= int(meta[c]["duration"] or 0) <= 300:
            by_s[str(meta[c]["seller_id"])].append(c)
    for s, v in by_s.items():
        rng.shuffle(v)
        k = 0
        for c in v:
            if k >= N_ENROLL_PER_SELLER:
                break
            if add(c, "enroll"):
                k += 1
    json.dump(calls, open(os.path.join(DIAR, "calls.json"), "w"), indent=1)
    os.makedirs(os.path.join(DIAR, "other"), exist_ok=True)
    cnt = collections.Counter((v["role"], v["seller"]) for v in calls.values())
    mb = sum(os.path.getsize(os.path.join(DIAR, "audio", f)) for f in os.listdir(os.path.join(DIAR, "audio"))) / 1e6
    other = [f for f in os.listdir(os.path.join(DIAR, "other")) if not f.startswith(".")]
    print(f"calls.json: {len(calls)} qo'ng'iroq, {mb:.0f} MB; (rol, sotuvchi) = {dict(cnt)}")
    print(f"other/: {len(other)} ta qo'ng'iroqsiz yozuv (egasi tashlamagan bo'lsa 0 — kutilgan 3-5)")


# ------------------------------------------------------------------ run (Pod)
def _cos(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _seller_profile(cents):
    """cents: [call -> [2 markaz]]. Takrorlanuvchi ovoz = sotuvchi (mijozlar har safar boshqa)."""
    import numpy as np
    pick = [int(np.argmax([sum(max(_cos(e, o) for o in cents[j]) for j in range(len(cents)) if j != i)
                           for e in cents[i]])) for i in range(len(cents))]
    prof = None
    for _ in range(3):
        prof = np.mean([cents[i][pick[i]] / np.linalg.norm(cents[i][pick[i]]) for i in range(len(cents))], 0)
        pick = [int(np.argmax([_cos(e, prof) for e in c])) for c in cents]
    return prof, pick


def _assign(words, turns):
    """so'z -> gapiruvchi: markaz nuqtasi tur ichida, bo'lmasa eng yaqin tur."""
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        best, bd = None, 1e9
        for t in turns:
            d = 0 if t["start"] <= mid <= t["end"] else min(abs(mid - t["start"]), abs(mid - t["end"]))
            if d < bd:
                best, bd = t["speaker"], d
        out.append(best)
    return out


def _segments(words, spk, role_of):
    segs = []
    for w, s in zip(words, spk):
        if segs and segs[-1]["speaker"] == s:
            segs[-1]["end"] = round(w["end"], 2); segs[-1]["text"] += " " + w["word"].strip()
        else:
            segs.append({"start": round(w["start"], 2), "end": round(w["end"], 2), "speaker": s,
                         "role": role_of.get(s, "nomalum"), "text": w["word"].strip()})
    return segs


def run(a):
    import numpy as np, torch
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import eval_pod
    eval_pod.check_matches_handler(verbose=False)       # shartnoma: handler.py bilan bir xil dekodlash
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    tok = os.environ.get("HF_TOKEN") or open("/root/.hf_token").read().strip()
    out = a.out; os.makedirs(out + "/segments", exist_ok=True)
    calls = json.load(open(os.path.join(a.data, "calls.json")))
    T = {}
    t0 = time.time()
    pipe = Pipeline.from_pretrained(PYANNOTE, use_auth_token=tok).to(torch.device("cuda"))
    asr = WhisperModel(a.model, device="cuda", compute_type=eval_pod.COMPUTE_TYPE)
    T["load_s"] = round(time.time() - t0, 1)

    def diarize(path, **kw):
        t = time.time()
        d, emb = pipe(path, return_embeddings=True, **kw)
        labels = list(d.labels())
        turns = [{"start": s.start, "end": s.end, "speaker": l} for s, _, l in d.itertracks(yield_label=True)]
        return turns, {l: emb[i] for i, l in enumerate(labels)}, time.time() - t

    def transcribe(path, words=True):
        t = time.time()
        segs, _ = asr.transcribe(path, word_timestamps=words, **eval_pod.DECODE)
        segs = list(segs)
        text = eval_pod.clean(" ".join(s.text.strip() for s in segs))
        w = [{"start": x.start, "end": x.end, "word": x.word} for s in segs for x in (s.words or [])]
        return text, w, time.time() - t

    # --- enrollment (har sotuvchi uchun) ---
    prof, enr_info = {}, {}
    for s in sorted({v["seller"] for v in calls.values()}):
        ids = [c for c, v in calls.items() if v["role"] == "enroll" and v["seller"] == s]
        cents = []
        for c in ids:
            _, e, _ = diarize(f"{a.data}/audio/{c}.wav", num_speakers=2)
            if len(e) == 2:
                cents.append(list(e.values()))
        if len(cents) >= 3:
            prof[s], pick = _seller_profile(cents)
            enr_info[s] = {"calls": len(cents)}
    sellers = sorted(prof)
    if len(sellers) == 2:
        enr_info["cos_between_seller_profiles"] = round(_cos(prof[sellers[0]], prof[sellers[1]]), 3)

    # --- test qo'ng'iroqlari ---
    rows, tt = [], collections.Counter()
    for c, v in sorted(calls.items()):
        if v["role"] != "test":
            continue
        p = f"{a.data}/audio/{c}.wav"
        text, words, ta = transcribe(p)
        text_nw, _, _ = transcribe(p, words=False)
        turns, emb, td = diarize(p, num_speakers=2)
        spk = _assign(words, turns)
        sims = {l: {s: _cos(e, prof[s]) for s in sellers} for l, e in emb.items()}
        known = v["seller"]
        role_of, seller_cluster = {}, None
        if known in prof and sims:
            seller_cluster = max(sims, key=lambda l: sims[l][known])
            if sims[seller_cluster][known] >= a.thr:
                role_of = {l: ("sotuvchi" if l == seller_cluster else "mijoz") for l in sims}
        # sotuvchini ko'r holda aniqlash: eng yuqori o'xshashlik qaysi profilga
        blind = max(((s, max(sims[l][s] for l in sims)) for s in sellers), key=lambda x: x[1])[0] if sims and sellers else None
        first = spk[0] if spk else None
        segs = _segments(words, spk, role_of)
        json.dump({"call": c, "seller": known, "direction": v["direction"], "segments": segs},
                  open(f"{out}/segments/{c}.json", "w"), ensure_ascii=False)
        dur = v["duration"] or 0
        tt["asr_s"] += ta; tt["diar_s"] += td; tt["audio_s"] += dur
        rows.append({"call": c, "direction": v["direction"], "seller": known, "dur": dur,
                     "n_speakers": len(emb), "n_segments": len(segs),
                     "sim_known": {l: round(sims[l][known], 3) for l in sims} if known in prof else None,
                     "role_assigned": bool(role_of), "blind_seller": blind, "blind_ok": blind == known,
                     "first_speaker_is_seller_cluster": (first == seller_cluster) if seller_cluster else None,
                     "text_same_with_word_ts": text == text_nw,
                     "asr_s": round(ta, 1), "diar_s": round(td, 1)})

    # --- qo'ng'iroqsiz yozuvlar: gapiruvchi soni noma'lum (1..8) ---
    oth = []
    od = os.path.join(a.data, "other")
    for f in sorted(os.listdir(od)) if os.path.isdir(od) else []:
        if f.startswith("."):
            continue
        p = os.path.join(od, f)
        text, words, ta = transcribe(p)
        turns, emb, td = diarize(p, min_speakers=1, max_speakers=8)
        spk = _assign(words, turns)
        names = {l: f"Speaker {i+1}" for i, l in enumerate(dict.fromkeys(spk))}
        segs = _segments(words, [names[s] for s in spk], {})
        json.dump({"file": f, "segments": segs}, open(f"{out}/segments/other_{f}.json", "w"), ensure_ascii=False)
        oth.append({"file": f, "n_speakers": len(names), "asr_s": round(ta, 1), "diar_s": round(td, 1)})

    n = len(rows) or 1
    sc = [r for r in rows if r["first_speaker_is_seller_cluster"] is not None]
    summary = {
        "pyannote": PYANNOTE, "model": a.model, "thr": a.thr, "timing_total_s": round(time.time() - t0, 1),
        "load_s": T["load_s"], "enrollment": enr_info,
        "test_calls": len(rows), "audio_min": round(tt["audio_s"] / 60, 1),
        "asr_rtf": round(tt["asr_s"] / max(tt["audio_s"], 1), 4), "diar_rtf": round(tt["diar_s"] / max(tt["audio_s"], 1), 4),
        "role_assigned_pct": round(100 * sum(r["role_assigned"] for r in rows) / n, 1),
        "blind_seller_acc_pct": round(100 * sum(r["blind_ok"] for r in rows) / n, 1),
        "n_speakers_hist": dict(collections.Counter(r["n_speakers"] for r in rows)),
        "baseline_first_speaker_eq_seller_cluster_pct": round(100 * sum(r["first_speaker_is_seller_cluster"] for r in sc) / max(len(sc), 1), 1),
        "text_same_with_word_ts_pct": round(100 * sum(r["text_same_with_word_ts"] for r in rows) / n, 1),
        "other": oth, "rows": rows,
    }
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows",)}, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    r = sub.add_parser("run")
    r.add_argument("--model", default="/workspace/round6/ct2")
    r.add_argument("--data", default="/workspace/diar")
    r.add_argument("--out", default="/workspace/diar/out")
    r.add_argument("--thr", type=float, default=0.35, help="rol berish uchun minimal kosinus o'xshashlik; ko'p sinovdan keyin tanlanadi")
    a = ap.parse_args()
    prep() if a.cmd == "prep" else run(a)


if __name__ == "__main__":
    main()
