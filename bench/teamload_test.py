#!/usr/bin/env python3
"""Team-load test: 32 requests, random 50-100k ctx, 10 concurrent users.
Captures per-request wall time + needle correctness. DRAM offload on."""
import json, time, random, urllib.request, concurrent.futures as cf

URL = "http://localhost:8081/v1/chat/completions"
VOCAB = ("harbor lantern compass anchor rudder sail mast tide wave storm beach cliff meadow forest river valley "
         "canyon peak glacier desert oasis savanna tundra prairie delta lagoon reef atoll strait fjord").split()

def make_req(i):
    r = random.Random(1000 + i)
    target_toks = random.Random(i).randint(50, 100) * 1000
    words = int(target_toks / 1.3)
    doc = " ".join(r.choice(VOCAB) for _ in range(words))
    code = f"NEEDLE-{i:02d}-{random.Random(500+i).randint(100,999)}"
    depth = random.Random(700 + i).random()
    cut = int(len(doc) * depth)
    prompt = (doc[:cut] + f"\n\nThe secret access code for this document is {code}.\n\n" + doc[cut:]
              + f"\n\nWhat is the secret access code? Answer with just the code.")
    return prompt, code, target_toks

def run_one(args):
    i, prompt, code = args
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 25, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            d = json.loads(resp.read())
        wall = time.perf_counter() - t0
        ptoks = d["usage"]["prompt_tokens"]
        out = d["choices"][0]["message"]["content"].strip()
        return {"i": i, "target": 0, "prompt_tokens": ptoks, "wall_s": round(wall, 1),
                "ok": code in out, "reply": out[:40]}
    except Exception as e:
        return {"i": i, "target": 0, "error": str(e)[:150]}

tasks = []
for i in range(32):
    p, code, tt = make_req(i)
    tasks.append((i, p, code))
print(f"32 requests, sizes 50-100k, 10 concurrent users, DRAM offload on", flush=True)
print(f"{'req':>3} {'prompt_toks':>11} {'wall_s':>8} {'ok':>4} reply", flush=True)

t0 = time.perf_counter()
results = []
done = 0
with cf.ThreadPoolExecutor(max_workers=10) as ex:
    for res in ex.map(run_one, tasks):
        done += 1
        res["finish_order"] = done
        res["elapsed_since_start"] = round(time.perf_counter() - t0, 1)
        results.append(res)
        if "error" in res:
            print(f"{res['i']:>3} ERROR {res['error']}", flush=True)
        else:
            print(f"{res['i']:>3} {res['prompt_tokens']:>11} {res['wall_s']:>8} {str(res['ok']):>5} {res['reply']!r}", flush=True)
        with open("/home/aisever/teamload_results.json", "w") as f:
            json.dump(results, f, indent=1)

ok = [r for r in results if r.get("ok")]
walls = sorted(r["wall_s"] for r in ok)
print(f"\nTOTAL: {time.perf_counter()-t0:.0f}s | correct {len(ok)}/32", flush=True)
if walls:
    print(f"wall_s: min {walls[0]} p50 {walls[len(walls)//2]} p90 {walls[int(len(walls)*0.9)]} max {walls[-1]}", flush=True)
print("TEAMLOAD DONE", flush=True)
