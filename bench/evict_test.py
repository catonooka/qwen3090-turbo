#!/usr/bin/env python3
"""Force GPU-cache eviction with a 250k monster, then repeat a 40k prompt -> CPU offload hit?"""
import json, time, random, urllib.request

URL = "http://localhost:8081/v1/chat/completions"
VOCAB = ("harbor lantern compass anchor rudder sail mast tide wave storm beach cliff meadow forest river valley "
         "canyon peak glacier desert oasis savanna tundra prairie delta lagoon reef atoll strait fjord").split()

def ask(prompt, max_tokens=40):
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        d = json.loads(resp.read())
    return time.perf_counter() - t0, d["usage"]["prompt_tokens"], d["choices"][0]["message"]["content"]

r = random.Random(99)
doc = " ".join(r.choice(VOCAB) for _ in range(30700))
P = doc + "\n\nThe launch code for the night mission is OMEGA-2211 and the backup is KILO-77.\n\nWhat are the two codes mentioned? Answer briefly."

t1, pt, out1 = ask(P)
print(f"40k pass1 (GPU cold): {t1:.1f}s -> {out1[:60]!r}", flush=True)
time.sleep(20)

m = random.Random(777)
monster = " ".join(m.choice(VOCAB) for _ in range(190000)) + "\n\nSay only: done."
tm, ptm, outm = ask(monster)
print(f"250k monster (evicts GPU cache): {tm:.0f}s, {ptm} toks -> {outm[:30]!r}", flush=True)
time.sleep(20)

t2, pt2, out2 = ask(P)
print(f"40k pass2 (GPU-evicted, CPU offload?): {t2:.1f}s -> {out2[:60]!r}", flush=True)
print(f"identical: {out1 == out2}, speedup vs fresh prefill: {t1/t2:.1f}x", flush=True)
print("EVICT-TEST DONE", flush=True)
