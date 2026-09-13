#!/usr/bin/env python3
"""Prefill benchmark: concurrency sweep c=1..32, unique ~1000-token prompts, incremental output."""
import json, time, random, string, urllib.request, concurrent.futures as cf

URL = "http://localhost:8081/v1/chat/completions"
filler = ("The quick brown fox jumps over the lazy dog while the sun sets behind "
          "the mountains and rivers flow gently through the valley below. ")
base = filler * 130
ptoks = None

def one_req(i):
    # unique nonce per request -> no cross-request prefix cache hits
    nonce = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": nonce + " " + base}],
        "max_tokens": 1,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return time.perf_counter() - t0, d["usage"]["prompt_tokens"]

_, ptoks = one_req(-1)
print(f"prompt tokens per request: {ptoks}", flush=True)
print(f"{'c':>3} {'ttft_avg_s':>10} {'ttft_max_s':>10} {'agg_prefill_t/s':>16}", flush=True)
results = {}
for c in range(1, 33):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        out = list(ex.map(one_req, range(c)))
    wall = time.perf_counter() - t0
    lats = [o[0] for o in out]
    toks = sum(o[1] for o in out)
    results[c] = (sum(lats)/len(lats), max(lats), toks/wall)
    print(f"{c:>3} {results[c][0]:>10.2f} {results[c][1]:>10.2f} {results[c][2]:>16.0f}", flush=True)
    with open("/home/aisever/prefill_results.json", "w") as f:
        json.dump({"prompt_tokens": ptoks, "data": {str(k): v for k, v in results.items()}}, f)
print("DONE saved prefill_results.json", flush=True)
