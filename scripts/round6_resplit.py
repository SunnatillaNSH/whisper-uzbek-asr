#!/usr/bin/env python3
"""Round 6 dataset: F1 (mijoz bo'yicha dev) + F2 (Claude tahrirlagan namunalar olib tashlanadi).

Jev A-004: dev train bilan bir lead'ni ulashmasin (F1), Claude tahrirlagan matnlar trening yorlig'ida bo'lmasin (F2).
Faqat stdlib. Kirish: manifest.jsonl, meta/transcripts.json, eval_calls_120.json. Chiqish: train.csv, dev.csv,
manifest.jsonl (split, split_v1, claude_edited), turns_*.csv, report.json (qayta yoziladi), removed_claude_edited.jsonl.
Qayta yurgizish xavfsiz: split_v1 mavjud bo'lsa u saqlanadi.
"""
import csv, json, os, random, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "round6-dataset")
SEED, FRAC = 20261005, 0.10
MAX_GROUP_FRAC = 0.03      # dev'ga bitta mijoz umumiy hajmning 3% idan ko'pini kiritmaydi (dev xilma-xil bo'lsin)


def call_id_of(p):
    return os.path.basename(str(p)).split(".")[0].split("_")[0]


def main():
    M = [json.loads(l) for l in open(os.path.join(D, "manifest.jsonl")) if l.strip()]
    for r in M:
        r.setdefault("split_v1", r["split"])

    # ---- F2: Claude tahrirlagan = bo'laksiz (butun-qator) qo'ng'iroq, uning qatorida Claude roli bor
    rc = set()
    for t in json.load(open(os.path.join(D, "meta", "transcripts.json"))):
        if ":p" in t["call_key"]:
            continue
        segs = json.loads(t["segments_json"] or "[]")
        if any(s.get("role") in ("sotuvchi", "mijoz") for s in segs):
            rc.add(int(t["call_key"].split(":")[1]))
    for r in M:
        r["claude_edited"] = r["part"] is None and r["call_id"] in rc
    edited = [r for r in M if r["claude_edited"]]
    pool = [r for r in M if not r["claude_edited"]]
    with open(os.path.join(D, "removed_claude_edited.jsonl"), "w") as f:
        for r in edited:
            f.write(json.dumps({k: r[k] for k in ("path", "call_id", "lead_id", "split_v1", "duration")}) + "\n")

    # ---- F1: mijoz (lead) bo'yicha dev. lead yo'q bo'lsa qo'ng'iroqning o'zi alohida guruh.
    gkey = lambda r: f"L{r['lead_id']}" if r["lead_id"] else f"C{r['call_id']}"
    groups = collections.defaultdict(list)
    for r in pool:
        groups[gkey(r)].append(r)
    target = round(len(pool) * FRAC)
    cap = max(1, round(len(pool) * MAX_GROUP_FRAC))
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    dev_keys, n = set(), 0
    for k in keys:
        s = len(groups[k])
        if n >= target:
            break
        if s > cap or n + s > target * 1.08:
            continue
        dev_keys.add(k)
        n += s
    for r in pool:
        r["split"] = "dev" if gkey(r) in dev_keys else "train"
    for r in edited:
        r["split"] = "removed"
    tr = [r for r in pool if r["split"] == "train"]
    dv = [r for r in pool if r["split"] == "dev"]

    # ---- SIZIB CHIQISH ASSERTLARI
    ev = json.load(open(os.path.join(ROOT, "data", "eval_calls_120.json")))
    eval_ids = {str(c["call_id"]) for c in ev["calls"]} | {str(x) for x in ev.get("locked_from_previous", [])}
    e50 = {str(x) for x in json.load(open(os.path.join(ROOT, "data", "eval_calls_50.json")))["calls"]}
    for p in ("eval.csv", "eval50.csv"):
        for r in csv.DictReader(open(os.path.join(ROOT, "data", "eval120", p))):
            eval_ids.add(call_id_of(r["path"]))
    assert e50 <= eval_ids
    trc, dvc = {str(r["call_id"]) for r in tr}, {str(r["call_id"]) for r in dv}
    trl, dvl = {r["lead_id"] for r in tr if r["lead_id"]}, {r["lead_id"] for r in dv if r["lead_id"]}
    assert not (trc & dvc), "train va dev qo'ng'iroqlari kesishdi"
    assert not (trl & dvl), "train va dev bir lead'ni ulashdi (F1 buzildi)"
    assert not (trc & eval_ids) and not (dvc & eval_ids), "eval-120 bilan kesishdi"
    assert not (trc & e50) and not (dvc & e50), "eval-50 bilan kesishdi"
    assert not any(r["claude_edited"] for r in tr + dv), "Claude tahrirlagan namuna qoldi"
    assert len({r["path"] for r in tr + dv}) == len(tr) + len(dv), "takroriy yo'l"
    long_tr = {r["sentence"] for r in tr if len(r["sentence"]) > 60}
    text_dupes = sum(1 for r in dv if r["sentence"] in long_tr)
    # eval leadlari bilan kesishuv — ma'lumot uchun (dev uchun talab emas)
    print("ASSERTLAR O'TDI: train ∩ dev (qo'ng'iroq VA lead) ∩ eval-120 ∩ eval-50 = 0; Claude-tahrirlangan 0")

    def wcsv(path, rs, col="sentence"):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["path", "sentence"])
            for r in rs:
                w.writerow([r["path"], r[col]])

    wcsv(os.path.join(D, "train.csv"), tr)
    wcsv(os.path.join(D, "dev.csv"), dv)
    wcsv(os.path.join(D, "turns_train.csv"), [r for r in tr if "turns" in r], "turns")
    wcsv(os.path.join(D, "turns_dev.csv"), [r for r in dv if "turns" in r], "turns")
    with open(os.path.join(D, "manifest.jsonl"), "w") as f:
        for r in M:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    old = os.path.join(D, "all.csv")
    if os.path.exists(old):
        os.remove(old)          # eski bo'lish bilan; train_calls.py endi TRAIN_CSV+DEV_CSV oladi

    hrs = lambda rs: round(sum(r["duration"] for r in rs) / 3600, 2)
    rep = json.load(open(os.path.join(D, "report.json")))
    rep = {k: v for k, v in rep.items() if not k.startswith(("train_", "dev_", "turns_", "samples", "hours", "calls", "dev_calls"))}
    dirs = lambda rs: dict(collections.Counter(r["direction"] for r in rs))
    rep.update({
        "split_version": "v2 (A-004: F1 lead-disjoint dev + F2 Claude-edited removed)", "split_seed": SEED,
        "claude_edited_removed_samples": len(edited), "claude_edited_removed_hours": hrs(edited),
        "claude_edited_removed_calls": len({r["call_id"] for r in edited}),
        "claude_edited_were_train_v1": sum(1 for r in edited if r["split_v1"] == "train"),
        "claude_edited_were_dev_v1": sum(1 for r in edited if r["split_v1"] == "dev"),
        "samples": len(pool), "hours": hrs(pool), "calls": len({r["call_id"] for r in pool}), "leads": len({r["lead_id"] for r in pool if r["lead_id"]}),
        "train_samples": len(tr), "train_hours": hrs(tr), "train_calls": len(trc), "train_leads": len(trl), "train_direction": dirs(tr),
        "dev_samples": len(dv), "dev_hours": hrs(dv), "dev_calls": len(dvc), "dev_leads": len(dvl), "dev_direction": dirs(dv),
        "dev_groups_without_lead(calls)": len({r["call_id"] for r in dv if not r["lead_id"]}),
        "dev_calls_sharing_lead_with_train": 0, "dev_leads_shared_with_train": len(trl & dvl),
        "dev_text_dupes_in_train(>60 belgi)": text_dupes,
        "max_train_lead_samples": max(collections.Counter(r["lead_id"] for r in tr if r["lead_id"]).values()),
        "turns_train_samples": sum(1 for r in tr if "turns" in r), "turns_dev_samples": sum(1 for r in dv if "turns" in r),
    })
    json.dump(rep, open(os.path.join(D, "report.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in rep if k.startswith(("claude", "samples", "hours", "calls", "leads", "train_", "dev_"))}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
