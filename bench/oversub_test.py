#!/usr/bin/env python3
"""Flow 3: deliberately exceed KV capacity (315k). Two concurrent ~245k prompts + monitoring."""
import json, time, random, urllib.request, threading

URL = "http://localhost:8081/v1/chat/completions"
random.seed(31337)
VOCAB = ("harbor lantern compass anchor rudder sail mast tide wave storm beach cliff meadow forest river valley "
         "canyon peak glacier desert oasis savanna tundra prairie delta lagoon reef atoll strait fjord isthmus").split()

def filler(seed, n):
    r = random.Random(seed)
    return " ".join(r.choice(VOCAB) for _ in range(n))

def ask(prompt, needle, max_tokens=40):
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": prompt + f"\n\nNEEDLE: {needle}\n\nWhat was the NEEDLE code? Just the code."}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    return time.perf_counter() - t0, d["usage"], d["choices"][0]["message"]["content"]

# ~245k tokens each: calibration said 2600 toks/2000 words = 1.3 tok/word
P1 = filler(1, 188000)
P2 = filler(2, 188000)
res = {}
def run(name, prompt, needle):
    try:
        dt, u, out = ask(prompt, needle)
        res[name] = (dt, u["prompt_tokens"], u["completion_tokens"], out[:60])
        print(f"{name}: {dt:.0f}s, {u['prompt_tokens']} in / {u['completion_tokens']} out -> {out[:50]!r}", flush=True)
    except Exception as e:
        res[name] = (None, None, None, str(e)[:200])
        print(f"{name}: FAILED {e}", flush=True)

t1 = threading.Thread(target=run, args=("oversubA", P1, "code ZEBRA-91"))
t2 = threading.Thread(target=run, args=("oversubB", P2, "code TANGO-44"))
t0 = time.perf_counter()
t1.start(); t2.start(); t1.join(); t2.join()
print(f"both done in {time.perf_counter()-t0:.0f}s total", flush=True)
json.dump(res, open("/home/aisever/oversub_results.json", "w"), indent=1, default=str)
print("FLOW3 DONE", flush=True)
