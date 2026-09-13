#!/usr/bin/env python3
"""Needle-in-a-haystack @ max context (256k). Random-word filler, needle at multiple depths."""
import json, time, random, urllib.request

URL = "http://localhost:8081/v1/chat/completions"
random.seed(42)

# ~800-word vocabulary for varied filler
VOCAB = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point home "
         "water room mother area money story fact month lot right study book eye job word business issue side kind "
         "head house service friend father power hour game line end member law car city community name president "
         "team minute idea kid body information back parent face others level office door health person art war "
         "history party result change morning reason research girl guide moment air teacher force education").split()
NUMS = [str(i) for i in range(10, 999)]

def make_filler_words(n_words):
    words = []
    while len(words) < n_words:
        words.append(random.choice(VOCAB))
        if random.random() < 0.12:
            words.append(random.choice(NUMS))
        if random.random() < 0.08:
            words.append("\n")  # sentence breaks
    return " ".join(words)

def ask(prompt, max_tokens=60):
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d["usage"]["prompt_tokens"], time.perf_counter() - t0

TARGET_PROMPT_TOKS = 250_000
# calibrate: words -> tokens ratio using a small sample
_, sample_toks, _ = ask(make_filler_words(2000) + "\n\nWhat is 2+2? Answer with just the number.", 5)
ratio = 2000 / sample_toks  # words per token (approx)
need_words = int(TARGET_PROMPT_TOKS * ratio)
print(f"calibration: {sample_toks} toks for 2000 words -> need ~{need_words} words", flush=True)

NEEDLE = ("The magic number for the secret project AKIRA-7 is exactly 4193. "
          "Remember this: AKIRA-7 magic number = 4193.")
QUESTION = "\n\nQuestion: According to the document above, what is the magic number for secret project AKIRA-7? Answer with just the number."
ANSWER = "4193"

results = []
for depth in [0.05, 0.30, 0.55, 0.80, 0.95]:
    filler = make_filler_words(need_words)
    cut = int(len(filler) * depth)
    prompt = filler[:cut] + "\n\n" + NEEDLE + "\n\n" + filler[cut:] + QUESTION
    out, ptoks, dt = ask(prompt)
    ok = ANSWER in out
    results.append({"depth": depth, "prompt_tokens": ptoks, "wall_s": round(dt,1), "found": ok, "reply": out[:80]})
    print(f"depth {depth:>4}: {ptoks} toks, {dt:.0f}s -> {'HIT ' if ok else 'MISS'} | {out[:60]!r}", flush=True)
    with open("/home/aisever/needle_results.json", "w") as f:
        json.dump(results, f, indent=1)

hits = sum(r["found"] for r in results)
print(f"SCORE: {hits}/{len(results)}", flush=True)
print("DONE", flush=True)
