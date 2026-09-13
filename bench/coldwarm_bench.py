#!/usr/bin/env python3
"""c1/c4 decode t/s, cold-start (first request after restart) vs warm."""
import json, time, sys, urllib.request, concurrent.futures as cf

URL = "http://localhost:8081/v1/chat/completions"

def one_req(i):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": f"Write a detailed technical paragraph ({i}) about GPU memory bandwidth."}],
        "max_tokens": 256,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    wall = time.perf_counter() - t0
    return wall, d["usage"]["completion_tokens"], d["usage"]["completion_tokens"] / wall

def run(c, label):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        out = list(ex.map(one_req, range(c)))
    wall = time.perf_counter() - t0
    total = sum(o[1] for o in out)
    tps = [o[2] for o in out]
    print(f"{label} c={c}: per-req t/s avg {sum(tps)/len(tps):.1f} (min {min(tps):.1f} max {max(tps):.1f}) | agg {total/wall:.1f} t/s | {total} toks in {wall:.1f}s", flush=True)

phase = sys.argv[1]
if phase == "cold":
    run(1, "COLD ")   # very first request after engine restart
    run(1, "WARM1")   # immediate repeat
    run(4, "WARM4")
else:
    for _ in range(2): run(1, "HOT1 ")  # long-warmed engine
    for _ in range(2): run(4, "HOT4 ")
