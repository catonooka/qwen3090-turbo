# Tuning log — 2× RTX 3090 / Qwen3.8-27B W4A16 / vLLM 0.29.0 (2026-09-13)

Every change made while chasing max context + max KV + zero quality loss, in order.

## Morning baseline
- `--max-model-len 204800 --gpu-memory-utilization 0.90 --kv-cache-dtype bfloat16`
- GPU KV: **212,791 tokens** (~9.09 GiB/GPU), 1.04x concurrency @ 204.8k
- Prefill ~1,720 t/s flat c1–c32; decode ~103 t/s c1, ~230 agg c4

## Attempt: FP8 KV cache — ❌ hardware-blocked
- `fp8_e5m2` + FLASH_ATTN → "FP8 KV cache requires FA3 on SM90 or FA4 on SM100"
- `fp8_e5m2` + TRITON_ATTN → "native FP8 (fp8e4nv) requires SM89+"
- 3090 = SM86: **no fp8 KV path exists in stock vLLM**. PR #49077 (software fp8
  dequant for Ampere) is unmerged as of 2026-09 — watch it.

## util 0.955 / 0.97 / 0.975 / 0.98 sweep @ 256k ctx
- 262,144 (full native): needs 10.93 GiB KV, only 10.73 available → **fails**
- 256,000 + 0.97: KV **257,331** ✅
- 256,000 + 0.975: KV **260,184** ✅ (0.98 hard-fails: asks 23.09 GiB, GPU1 has 23.06)
- later reverted to 0.95 for team-load stability (see below)

## Patch: hybrid KV group-size (PR #49472) — ✅ the big lossless win
- Stock vLLM groups KV layers by the DFlash drafter's 5-layer bucket →
  "Add 4 padding layers, may waste at most 25.00% KV cache memory"
- Patched: padding lands on the tiny SWA bucket instead
- KV 260,184 → **315,389 tokens** (+21%), zero numerics touched
- 5/5 needle hits @ 248k ctx after patch (quality gate passed)

## CPU RAM KV offload — ❌ then ✅ after backporting 3 PRs
- First attempt: connector ran, 51.53 GB mmap region allocated, but
  **External prefix cache hit rate: 0.0%** — silent no-op
- Root cause (research): PR #52771 — offloader zeroes hits under EAGLE/MTP-class
  spec decode (DFlash). Not in 0.29.0. Companions: #52807 (hybrid load boundary),
  #52923 (store-before-keys race)
- Backported all three (#52771 hunk 1 by hand — 0.29 file uses `use_eagle()`,
  upstream diff expects `use_eagle_block_drop()`; hunks 2-3 apply cleanly)
- Also: **must remove `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** —
  VMM remap breaks pinned KV; vLLM refuses to start otherwise
- Result: 40k reprompt after 250k eviction = **1.1 s vs 26.5 s** (24x),
  bit-identical output, External hit confirmed

## Team load (10 users, 32 req × 50–100k) — crash then stable
- @ util 0.975: **DFlash drafter OOM** in `speculator.propose` → `torch.cat`
  needed 368 MiB with 107 MiB free → engine died, all 32 requests failed
- @ util 0.95: 32/32 correct, p50 539 s, zero errors. GPU KV 292,518 (−22k,
  cheap insurance for the drafter's concurrent-batch transients)
- `max-num-seqs 8 → 10`: no throughput change (prefill compute-bound) but no
  OOM; kept so 10 users never queue behind slot 8
- `max-num-batched-tokens 16384`: **OOM** in GDN prefill transient
  (`buf8 = empty_strided_cuda((s18, 8704), ...)`) at 10 seqs → reverted to 8192

## Final production config
`0.95 util / 10 seqs / mnbt 8192 / 256k len / bf16 KV / FLASH_ATTN /
native offload 48 GiB / DFlash n=7` → **292k GPU + ~1.1M DRAM KV tokens**

## Rejected elsewhere
- LMCache (in-process + MP): hybrid GDN + spec decode corruption, LMCache#4984
- Mooncake standalone: works etcd-free but no benefit over native connector single-node
- int8_per_token_head KV + TRITON_ATTN: ~2x KV possible but quantized (not
  lossless) + slower long-ctx backend + spec-decode patch needed — plan B only
