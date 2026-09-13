#!/usr/bin/env python3
"""Validate native KV offloading: correctness, multi-request stability (#41515), speed win."""
import json, time, random, urllib.request

URL = "http://localhost:8081/v1/chat/completions"
r = random.Random(99)
VOCAB = ("harbor lantern compass anchor rudder sail mast tide wave storm beach cliff meadow forest river valley "
         "canyon peak glacier desert oasis savanna tundra prairie delta lagoon reef atoll strait fjord").split()

def ask(prompt, max_tokens=40):
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        d = json.loads(resp.read())
    return time.perf_counter() - t0, d["usage"], d["choices"][0]["message"]["content"]

# ~40k-token doc with a needle, asked twice identically
doc = " ".join(r.choice(VOCAB) for _ in range(30700))
P = doc + "\n\nThe launch code for the night mission is OMEGA-2211 and the backup is KILO-77.\n\nWhat are the two codes mentioned? Answer briefly."
r1_t, r1_u, r1_out = ask(P)
print(f"pass1: {r1_t:.1f}s, {r1_u['prompt_tokens']} toks -> {r1_out[:80]!r}", flush=True)
time.sleep(25)  # let async CPU stores drain
r2_t, r2_u, r2_out = ask(P)
print(f"pass2: {r2_t:.1f}s, {r2_u['prompt_tokens']} toks -> {r2_out[:80]!r}", flush=True)
time.sleep(25)
r3_t, r3_u, r3_out = ask(P)  # third hit also probes the 'subsequent request' crash zone
print(f"pass3: {r3_t:.1f}s, {r3_u['prompt_tokens']} toks -> {r3_out[:80]!r}", flush=True)
ident = r1_out == r2_out == r3_out
print(f"bit-identical outputs: {ident}", flush=True)
print(f"speedup pass2 vs pass1: {r1_t/r2_t:.1f}x, pass3: {r1_t/r3_t:.1f}x", flush=True)
print("OFFLOAD-TEST DONE", flush=True)
