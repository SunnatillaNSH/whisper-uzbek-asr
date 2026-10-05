#!/usr/bin/env python3
"""Round 6: SellUp D1 dan Muxlisa transkriptlari + qo'ng'iroq metama'lumotini lokal eksport (SELECT only).
Chiqish: data/round6-dataset/meta/{transcripts,calls}.json (gitignore'da; client_number eksport QILINMAYDI)."""
import json, subprocess, sys, pathlib
CF = "/Users/macbookuz/code/cf-call-analyzer"
OUT = pathlib.Path("/Users/macbookuz/code/whisper-uzbek-asr/data/round6-dataset/meta"); OUT.mkdir(parents=True, exist_ok=True)

def q(sql):
    r = subprocess.run(["npx", "wrangler", "d1", "execute", "qongiroq-tahlili-db", "--remote", "--json", "--command", sql],
                       cwd=CF, capture_output=True, text=True)
    if r.returncode: sys.exit(r.stderr[-500:])
    return json.loads(r.stdout)[0]["results"]

assert True
T = []; off = 0; PAGE = 120
while True:
    rows = q(f"SELECT call_key, created_at, duration, full_text, segments_json FROM transcripts "
             f"WHERE provider='muxlisa' AND status='ready' ORDER BY call_key LIMIT {PAGE} OFFSET {off}")
    T += rows; off += PAGE
    print(len(T), end=" ", flush=True)
    if len(rows) < PAGE: break
print()
(OUT / "transcripts.json").write_text(json.dumps(T, ensure_ascii=False))
ids = sorted({int(t["call_key"].split(":")[1]) for t in T})
C = []
for i in range(0, len(ids), 200):
    chunk = ",".join(map(str, ids[i:i+200]))
    C += q(f"SELECT db_call_id, direction, start_time, duration, answered, recording_url, lead_id FROM moizvonki_calls WHERE db_call_id IN ({chunk})")
(OUT / "calls.json").write_text(json.dumps(C, ensure_ascii=False))
print("transcripts", len(T), "calls", len(C))
