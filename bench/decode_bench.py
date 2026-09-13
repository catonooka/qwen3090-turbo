#!/usr/bin/env python3
"""Decode benchmark: c=1..8, short prompt, 256 output tokens, capture gen t/s + streaming-free timing."""
import json, time, urllib.request, concurrent.futures as cf

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
    u = d["usage"]
    gen = u["completion_tokens"]
    # decode time = wall - approx prefill (small prompt, ~15 tok @1700 t/s ~ 0.01s, negligible)
    return wall, gen, gen / wall, u.get("completion_tokens_details", {})

# warmup
one_req(0)
print(f"{'c':>3} {'avg_t/s/req':>12} {'sum_toks':>8} {'wall_s':>8} {'agg_t/s':>9}", flush=True)
results = {}
for c in range(1, 9):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        out = list(ex.map(one_req, range(c)))
    wall = time.perf_counter() - t0
    tps = [o[2] for o in out]
    total = sum(o[1] for o in out)
    results[c] = (sum(tps)/len(tps), total, wall, total/wall)
    print(f"{c:>3} {results[c][0]:>12.1f} {results[c][1]:>8} {results[c][2]:>8.1f} {results[c][3]:>9.1f}", flush=True)
    with open("/home/aisever/decode_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f)
print("DONE", flush=True)
