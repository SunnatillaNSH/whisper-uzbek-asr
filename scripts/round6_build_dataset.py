#!/usr/bin/env python3
"""Round 6 dataset: BARCHA Muxlisa transkriptlari (eski + yangi) -> bitta trening to'plami.

Kirish  : data/round6-dataset/raw55|raw28  (tools/build-asr-dataset.mjs chiqishi, serverning o'z bo'luvchisi bilan)
          data/round6-dataset/meta/transcripts.json (rollar uchun), data/eval_calls_120.json, eval_calls_50.json
Chiqish : data/round6-dataset/{audio/*.flac, train.csv, dev.csv, manifest.jsonl, turns_train.csv, turns_dev.csv,
          rejected.jsonl, report.json}   (hammasi gitignore'da — mijoz audiosi va matni)

Qoidalar: bo'lak <=30 s, >=1.2 s, cps 6-26; birortasi o'tmasa BUTUN namuna chetlanadi.
Eval-120 (va eval-50) qo'ng'iroqlari QO'NG'IROQ bo'yicha chiqarib tashlanadi; train/dev ham qo'ng'iroq bo'yicha.
Rol (SOTUVCHI/MIJOZ) faqat Claude diarizatsiyasi bor qatorlarda mavjud; sidecar sifatida saqlanadi.
Mamlakat: faqat CPU (numpy, soundfile).
"""
import csv, difflib, json, math, os, random, re, sys, collections
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "round6-dataset")
SR = 16000
MAX_SEC, MIN_SEC, MIN_CHARS = 30.0, 1.2, 4
CPS_MIN, CPS_MAX = 6.0, 26.0
MIN_SILENCE = 0.30
TEXT_TOL_SEC = 3.0
DEV_FRAC, DEV_SEED = 0.10, 20260919          # train_calls.py bilan AYNAN bir xil
SENT_END = re.compile(r"[.!?…]+[\s ]+")


def call_id_of(path):
    return os.path.basename(str(path)).split(".")[0].split("_")[0]


# ---------------------------------------------------------------- kesish
def silence_mask(x, frame=0.025, hop=0.010):
    n_f, n_h = int(frame * SR), int(hop * SR)
    if len(x) < n_f:
        return np.zeros(0, dtype=bool), n_h
    n = 1 + (len(x) - n_f) // n_h
    idx = np.arange(n_f)[None, :] + n_h * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx].astype(np.float64) ** 2, axis=1)) + 1e-10
    db = 20 * np.log10(rms)
    thr = max(np.percentile(db, 10) + 6.0, np.percentile(db, 90) - 30.0)
    return db < thr, n_h


def silence_runs(mask, hop_n):
    runs, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        t0, t1 = i * hop_n / SR, j * hop_n / SR
        if t1 - t0 >= MIN_SILENCE:
            runs.append((t0, t1))
        i = j
    return runs


def plan_cuts(x, text, dur):
    """Eng yaxshi (jimlik vaqti, gap chegarasi) juftliklari. Eski skriptdagi xato tuzatilgan:
    kesish nuqtasi har bo'lak <=30 s bo'lishini ta'minlaydigan feasible oraliqdan qidiriladi.
    Matn<->vaqt xaritasi bir tekis emas, balki NUTQ (ovozli freymlar) bo'yicha."""
    mask, hop_n = silence_mask(x)
    runs = silence_runs(mask, hop_n)
    bounds = [m.end() for m in SENT_END.finditer(text)]
    L = len(text)
    if not runs:
        return None, "no_silence"
    if not bounds:
        return None, "no_sentence_boundary"
    voiced = np.cumsum(~mask) * hop_n / SR           # ovozli soniyalar (kumulyativ)
    vt = max(voiced[-1], 1e-6)

    def vfrac(t):
        k = min(len(voiced) - 1, int(t * SR / hop_n))
        return voiced[k] / vt

    mids = [((a + b) / 2, b - a) for a, b in runs]
    best = None
    n0 = math.ceil(dur / MAX_SEC)
    for n in (n0, n0 + 1):
        def dfs(i, prev_t, prev_b, cuts, cost):
            nonlocal best
            if i == n:
                pieces_t = [0.0] + [c[0] for c in cuts] + [dur]
                pieces_b = [0] + [c[1] for c in cuts] + [L]
                for k in range(n):
                    d = pieces_t[k + 1] - pieces_t[k]
                    ch = len(text[pieces_b[k]:pieces_b[k + 1]].strip())
                    if d > MAX_SEC or d < MIN_SEC or ch < MIN_CHARS or not (CPS_MIN <= ch / d <= CPS_MAX):
                        return
                if best is None or cost < best[0]:
                    best = (cost, list(cuts))
                return
            lo = max(prev_t + MIN_SEC, dur - MAX_SEC * (n - i))
            hi = min(prev_t + MAX_SEC, dur - MIN_SEC * (n - i))
            for t, sl in mids:
                if not (lo <= t <= hi):
                    continue
                target = vfrac(t) * L
                cand = [b for b in bounds if b > prev_b and abs(b - target) <= TEXT_TOL_SEC * (L / dur)]
                if not cand:
                    continue
                b = min(cand, key=lambda b: abs(b - target))
                dfs(i + 1, t, b, cuts + [(t, b)], cost + abs(b - target) / L - 0.01 * min(sl, 1.0))
        dfs(1, 0.0, 0, [], 0.0)
        if best:
            break
    if best is None:
        return None, "no_feasible_cut_or_cps"
    return best[1], None


def cut_sample(x, text, dur):
    """-> ([(audio, text)], None) yoki (None, sabab)."""
    if dur <= MAX_SEC:
        ch = len(text.strip())
        if dur < MIN_SEC or ch < MIN_CHARS:
            return None, "short"
        if not (CPS_MIN <= ch / dur <= CPS_MAX):
            return None, "cps"
        return [(x, text.strip())], None
    cuts, why = plan_cuts(x, text, dur)
    if cuts is None:
        return None, why
    ts = [0.0] + [c[0] for c in cuts] + [dur]
    bs = [0] + [c[1] for c in cuts] + [len(text)]
    return [(x[int(ts[k] * SR):int(ts[k + 1] * SR)], text[bs[k]:bs[k + 1]].strip()) for k in range(len(ts) - 1)], None


# ---------------------------------------------------------------- rollar
def wnorm(w):
    return re.sub(r"[^\w']+", "", w.lower().replace("’", "'").replace("‘", "'").replace("ʻ", "'").replace("`", "'"))


def load_role_tokens():
    """call_id -> [(norm_so'z, rol)] — Claude diarizatsiyasi bor to'liq-qo'ng'iroq qatoridan."""
    T = json.load(open(os.path.join(D, "meta", "transcripts.json")))
    out = {}
    for t in T:
        if ":p" in t["call_key"]:
            continue
        segs = json.loads(t["segments_json"] or "[]")
        if not any(s.get("role") in ("sotuvchi", "mijoz") for s in segs):
            continue
        toks = []
        for s in segs:
            r = s.get("role")
            for w in s["text"].split():
                n = wnorm(w)
                if n:
                    toks.append((n, r))
        out[int(t["call_key"].split(":")[1])] = toks
    return out


def role_runs(piece_text, whole_toks, hint_pos):
    """Bo'lak so'zlarini to'liq-qo'ng'iroq rolli so'zlariga tekislaydi.
    -> (roli matn 'turns' satri yoki None, mos kelish ulushi)."""
    words = piece_text.split()
    pw = [(i, wnorm(w)) for i, w in enumerate(words) if wnorm(w)]
    if not pw:
        return None, 0.0
    a = [t[0] for t in whole_toks]
    b = [n for _, n in pw]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    role_of = [None] * len(b)
    matched = 0
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            role_of[blk.b + k] = whole_toks[blk.a + k][1]
            matched += 1
    frac = matched / len(b)
    if frac < 0.90:
        return None, frac
    last = next((r for r in role_of if r), None)
    for k in range(len(role_of)):
        if role_of[k] is None:
            role_of[k] = last
        else:
            last = role_of[k]
    tag = {"sotuvchi": "[SOTUVCHI]", "mijoz": "[MIJOZ]"}
    out, cur = [], None
    wi_to_role = {pw[k][0]: role_of[k] for k in range(len(pw))}
    run_role = role_of[0]
    for i, w in enumerate(words):
        r = wi_to_role.get(i, run_role)
        run_role = r
        if r != cur:
            out.append(tag[r])
            cur = r
        out.append(w)
    return " ".join(out), frac


# ---------------------------------------------------------------- asosiy
def main():
    ev = json.load(open(os.path.join(ROOT, "data", "eval_calls_120.json")))
    eval_ids = {str(c["call_id"]) for c in ev["calls"]} | {str(x) for x in ev.get("locked_from_previous", [])}
    e50 = {str(x) for x in json.load(open(os.path.join(ROOT, "data", "eval_calls_50.json")))["calls"]}
    for p in ("eval.csv", "eval50.csv"):
        for r in csv.DictReader(open(os.path.join(ROOT, "data", "eval120", p))):
            eval_ids.add(call_id_of(r["path"]))
    assert e50 <= eval_ids, "eval-50 eval-120 ning qismi emas"
    print(f"eval qo'ng'iroqlari: {len(eval_ids)} (eval-50: {len(e50)})")

    rows = []
    for g in ("raw55", "raw28"):
        mp = os.path.join(D, g, "manifest.jsonl")
        if not os.path.exists(mp):
            continue
        for l in open(mp):
            if l.strip():
                r = json.loads(l)
                r["src"] = g
                r["abs"] = os.path.join(D, g, r["path"])
                rows.append(r)
    st = collections.Counter()
    st["raw_samples"] = len(rows)
    st["raw_hours"] = round(sum(r["duration"] for r in rows) / 3600, 2)
    st["raw_calls"] = len({r["call_id"] for r in rows})
    rows_ne = [r for r in rows if str(r["call_id"]) not in eval_ids]
    st["eval_excluded_samples"] = len(rows) - len(rows_ne)
    st["eval_excluded_calls"] = len({r["call_id"] for r in rows if str(r["call_id"]) in eval_ids})
    role_toks = load_role_tokens()

    os.makedirs(os.path.join(D, "audio"), exist_ok=True)
    out, rej = [], []
    for k, r in enumerate(rows_ne, 1):
        if k % 200 == 0:
            print(f"  {k}/{len(rows_ne)}", flush=True)
        try:
            x, sr = sf.read(r["abs"], dtype="float32")
        except Exception:
            rej.append({"path": r["path"], "call_id": r["call_id"], "why": "read_fail"})
            continue
        assert sr == SR
        if x.ndim > 1:
            x = x.mean(axis=1)
        dur = len(x) / SR
        pieces, why = cut_sample(x, r["sentence"], dur)
        if pieces is None:
            rej.append({"path": r["path"], "call_id": r["call_id"], "dur": round(dur, 1), "chars": len(r["sentence"]), "why": why,
                        "src": r["src"], "sentence": r["sentence"]})
            continue
        stem = os.path.splitext(os.path.basename(r["path"]))[0]
        for i, (seg, txt) in enumerate(pieces):
            name = f"{stem}.flac" if len(pieces) == 1 else f"{stem}_s{i+1}.flac"
            sf.write(os.path.join(D, "audio", name), seg, SR, format="FLAC")
            d = len(seg) / SR
            row = {"path": f"audio/{name}", "sentence": txt, "duration": round(d, 2), "cps": round(len(txt) / d, 2),
                   "call_id": int(r["call_id"]), "part": r.get("part"), "piece": i + 1 if len(pieces) > 1 else None,
                   "direction": r.get("direction"), "date": (r.get("started_at") or "")[:10], "lead_id": r.get("lead_id"),
                   "chunk": r["src"][3:], "was_long": dur > MAX_SEC}
            toks = role_toks.get(int(r["call_id"]))
            if toks:
                turns, frac = role_runs(txt, toks, 0)
                if turns:
                    row["turns"] = turns
                    row["role_match"] = round(frac, 3)
            out.append(row)

    # ---- train/dev: QO'NG'IROQ bo'yicha, train_calls.py bilan bir xil algoritm
    calls = sorted({str(r["call_id"]) for r in out})
    rng = random.Random(DEV_SEED)
    rng.shuffle(calls)
    n_dev = max(1, round(len(calls) * DEV_FRAC))
    dev_calls = set(calls[:n_dev])
    for r in out:
        r["split"] = "dev" if str(r["call_id"]) in dev_calls else "train"
    tr = [r for r in out if r["split"] == "train"]
    dv = [r for r in out if r["split"] == "dev"]

    # ---- SIZIB CHIQISH ASSERTLARI (treningdan OLDIN)
    trc, dvc = {str(r["call_id"]) for r in tr}, {str(r["call_id"]) for r in dv}
    assert not (trc & dvc), "train va dev kesishdi"
    assert not (trc & eval_ids), "train eval-120 bilan kesishdi"
    assert not (dvc & eval_ids), "dev eval-120 bilan kesishdi"
    assert not (trc & e50) and not (dvc & e50), "eval-50 bilan kesishdi"
    assert all(MIN_SEC <= r["duration"] <= MAX_SEC and CPS_MIN <= r["cps"] <= CPS_MAX for r in out), "sifat darvozasi buzilgan"
    assert len({r["path"] for r in out}) == len(out), "takroriy yo'l"
    # bir xil matn (aniq takror) train va dev'da — qo'shimcha tekshiruv, faqat qisqa gaplardan tashqari
    long_tr = {r["sentence"] for r in tr if len(r["sentence"]) > 60}
    st["dev_text_dupes_in_train(>60 belgi)"] = sum(1 for r in dv if r["sentence"] in long_tr)
    trl = {r["lead_id"] for r in tr if r["lead_id"]}
    st["dev_calls_sharing_lead_with_train"] = len({r["call_id"] for r in dv if r["lead_id"] in trl})
    print("ASSERTLAR O'TDI: train ∩ dev ∩ eval-120 ∩ eval-50 = 0")

    def wcsv(path, rs, col="sentence"):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["path", "sentence"])
            for r in rs:
                w.writerow([r["path"], r[col]])

    wcsv(os.path.join(D, "train.csv"), tr)
    wcsv(os.path.join(D, "dev.csv"), dv)
    ttr = [r for r in tr if "turns" in r]
    tdv = [r for r in dv if "turns" in r]
    wcsv(os.path.join(D, "turns_train.csv"), ttr, "turns")
    wcsv(os.path.join(D, "turns_dev.csv"), tdv, "turns")
    with open(os.path.join(D, "manifest.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(D, "rejected.jsonl"), "w") as f:
        for r in rej:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hrs = lambda rs: round(sum(r["duration"] for r in rs) / 3600, 2)
    rep = dict(st)
    rep.update({
        "samples": len(out), "hours": hrs(out), "calls": len(calls),
        "train_samples": len(tr), "train_hours": hrs(tr), "train_calls": len(trc),
        "dev_samples": len(dv), "dev_hours": hrs(dv), "dev_calls": len(dvc),
        "rejected_samples": len(rej), "rejected_hours": round(sum(r.get("dur", 0) for r in rej) / 3600, 2),
        "rejected_by_reason": dict(collections.Counter(r["why"] for r in rej)),
        "split_from_long": sum(1 for r in out if r["was_long"]), "whole_le30": sum(1 for r in out if not r["was_long"]),
        "by_chunk": {k: len([r for r in out if r["chunk"] == k]) for k in ("55", "28")},
        "turns_train_samples": len(ttr), "turns_train_hours": hrs(ttr), "turns_dev_samples": len(tdv), "turns_dev_hours": hrs(tdv),
        "turns_calls": len({r["call_id"] for r in out if "turns" in r}),
        "median_cps": round(float(np.median([r["cps"] for r in out])), 1),
        "by_direction": dict(collections.Counter(r["direction"] for r in out)),
        "by_month_day_first": dict(sorted(collections.Counter(r["date"][:7] for r in out).items())),
    })
    json.dump(rep, open(os.path.join(D, "report.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
