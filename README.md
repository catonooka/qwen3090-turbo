# qwen3090-turbo ⚡

Production-ready vLLM serving stack for **Qwen3.8-27B** (W4A16 + DFlash2 speculative
decoding) on **2× RTX 3090 with NVLink** — tuned for maximum KV cache, zero quality
loss, and 10-user team loads with a 48 GB DRAM offload tier.

- **256k context** (model native max 262,144 — 256k is the VRAM ceiling)
- **292,518-token GPU KV cache** (+21% from a lossless KV-layout patch)
- **48 GB CPU RAM KV offload tier** (~1.1M tokens) via patched native offloader
- **~1,700 t/s aggregate prefill**, ~103 t/s single-stream decode (DFlash spec decode)
- **24x faster repeat prompts** (DRAM KV promotion vs re-prefill), bit-identical output

## Credits / where this comes from

This stack is assembled from community research — standing on these shoulders:

- **[club-3090](https://github.com/noonghunna/club-3090)** (issue #1052 and the
  `ultrafast` dflash2.yml recipe) — the base DFlash2-on-dual-3090 vLLM recipe and
  the 51581 dense-KV patch lineage this launcher started from
- **vLLM upstream PRs**, backported here into vLLM 0.29.0 site-packages:
  - [#49472](https://github.com/vllm-project/vllm/pull/49472) — hybrid KV group-size
    selector (+21% KV, lossless)
  - [#52771](https://github.com/vllm-project/vllm/pull/52771) — stop zeroing KV-offload
    hits under EAGLE/MTP-class speculative decoding (the fix that makes DRAM offload work)
  - [#52807](https://github.com/vllm-project/vllm/pull/52807) — recurrent/hybrid group
    load-boundary fix for offloading
  - [#52923](https://github.com/vllm-project/vllm/pull/52923) — offload store-before-keys race
  - [#48375](https://github.com/vllm-project/vllm/pull/48375) — mamba drop-eagle-block fix
- **Model authors** (download from Hugging Face, see next section):
  - Target model: [Frozenlock/Qwen3.8-27B-int4-AutoRound](https://huggingface.co/Frozenlock/Qwen3.8-27B-int4-AutoRound)
    — INT4 W4A16 AutoRound quant of Qwen/Qwen3.8-27B (~18 GB, vision-capable, MTP head quantized in)
  - Draft model: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
    via the W4A16 build we use ([syvai DFlash2-W4A16 packaging](https://huggingface.co/syvai), ~1.2 GB)
- **vLLM docs**: [KV offloading usage](https://docs.vllm.ai/en/stable/features/kv_offloading/),
  [conserving memory](https://docs.vllm.ai/en/stable/configuration/conserving_memory/)

## Hardware this was built and measured on

| part | spec |
|---|---|
| CPU | AMD Threadripper PRO 3955WX |
| RAM | 128 GB DDR4 (48 GB used as pinned KV offload tier) |
| GPUs | 2× NVIDIA RTX 3090 24 GB, NVLink NV4 (tensor parallel 2) |
| OS | Linux 6.0, NVIDIA driver with CUDA SM86 support |

## Repo layout

```
├── README.md                  ← this file
├── systemd/
│   ├── vllm-qwen.service      ← systemd unit
│   └── llama-qwen.service     ← optional llama.cpp fallback service (mutually exclusive)
├── launch/
│   ├── vllm-launch.sh         ← the tuned vLLM launcher (THE artifact)
│   └── llama-launch.sh        ← optional llama.cpp launcher
├── patches/
│   ├── 52771.diff             ← KV offload: stop zeroing hits under spec decode (hand-ported hunk 1)
│   ├── 52807.diff             ← KV offload: recurrent-group load-boundary fix
│   ├── 52923.diff             ← KV offload: store-before-keys race fix
│   ├── 49472.diff             ← hybrid KV group-size: +21% KV, zero quality loss
│   ├── 51581-dflash-dense-kv.diff
│   ├── gdn-mtp-async-spec-order.diff
│   └── pr48375-mamba-drop-eagle-block.diff
├── bench/
│   ├── prefill_bench.py       ← c1..c32 prefill/TTFT sweep
│   ├── decode_bench.py        ← decode t/s (c1..c8)
│   ├── coldwarm_bench.py      ← cold vs warm decode
│   ├── needle_test.py         ← 5-depth needle-in-haystack @ ~248k ctx
│   ├── evict_test.py          ← DRAM offload promotion test (the 24x demo)
│   ├── offload_test.py        ← offloader correctness/bit-identity
│   ├── teamload_test.py       ← 32 req × 50-100k ctx, 10 concurrent users
│   └── oversub_test.py        ← 2×250k concurrent oversubscription survival
├── results/                   ← measured numbers from this exact rig (JSON + logs)
└── docs/
    └── TUNING_LOG.md          ← what was tried, what OOM'd, what won
```

## Dependencies

**System:** Linux, NVIDIA driver with SM86 (Ampere) support, systemd, ~40 GB disk
for models, 128 GB RAM recommended (48 GB is reserved as the pinned KV offload tier).

```bash
# 1. Python 3.12 venv with the exact vLLM version the patches target
python3.12 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install vllm==0.29.0 huggingface_hub
# (vLLM wheels bundle torch, CUDA kernels, transformers — nothing else needed)

# 2. Models (~19 GB total; HF token required if either repo is gated)
huggingface-cli login          # paste a token from https://huggingface.co/settings/tokens
huggingface-cli download Frozenlock/Qwen3.8-27B-int4-AutoRound --local-dir ~/models/frozenlock-int4
huggingface-cli download syvai/DFlash2-W4A16 --local-dir ~/models/dflash2-w4a16
```

Model cards:
- **Target** — `Frozenlock/Qwen3.8-27B-int4-AutoRound`: INT4/W4A16 AutoRound quant of
  `Qwen/Qwen3.8-27B` (~18 GB). Vision-capable (image-text-to-text), MTP head quantized
  in-tree so speculative decoding works out of the box.
- **Drafter** — `syvai/DFlash2-W4A16`: W4A16 DFlash2 speculative draft model
  (~1.2 GB), based on `incoai/Qwen3.8-27B-DFlash2`. n=7 draft tokens, ~46% acceptance.

## Install & run

```bash
# 1. clone + install deps (above), models in ~/models/
git clone <this-repo> ~/qwen3090-turbo

# 2. apply patches into the venv (REQUIRED — offload & +21% KV fixes)
cd ~/qwen3090-turbo/patches
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
for p in 49472.diff 52771.diff 52807.diff 52923.diff \
         51581-dflash-dense-kv.diff gdn-mtp-async-spec-order.diff pr48375-mamba-drop-eagle-block.diff; do
  echo y | patch -d "$(dirname $VLLM_DIR)" -p1 < $p   # test-file hunks skip safely
done

# 3. install services
sudo cp systemd/vllm-qwen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen

# 4. verify
curl http://localhost:8081/v1/models
```

Startup takes ~2–3 min (weight load + torch.compile + CUDA graphs + warmup).
Logs: `journalctl -u vllm-qwen -f`. Look for:
```
GPU KV cache size: 292,518 tokens, Maximum concurrency for 256,000 tokens per request: 1.14x
Creating offloading spec with name: CPUOffloadingSpec
Created mmap file /dev/shm/vllm_offload_*.mmap (51.53 GB)
```

⚠️ **The launcher must NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**
(it is incompatible with the pinned-KV offloader and vLLM will refuse to start).

## Quick use

```bash
curl http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Thinking mode is OFF by default (`enable_thinking: false` baked into the launcher);
enable per-request via `chat_template_kwargs`.

## Measured numbers (this exact rig, vLLM 0.29.0, 2026-09-13)

### KV capacity — total budget

| tier | tokens | bytes |
|---|---|---|
| GPU KV cache (bf16, both GPUs) | **292,518** | ~12.9 GiB/GPU pool |
| CPU DRAM offload tier | **~1,100,000** | 48 GiB pinned (51.53 GB mmap region) |
| **total effective KV** | **~1.39M tokens** | |

(Pre-patch baseline this morning: 212,791 GPU tokens @ 204.8k ctx — +37% GPU KV and
+256k→256k ctx cap via patch + utilization tuning, all lossless.)

### Prefill (unique ~3.3k-token prompts, no cache help)

| concurrency | TTFT avg | TTFT max | agg prefill t/s |
|---|---|---|---|
| 1 | 1.91 s | 1.91 s | 1,718 |
| 8 | 11.18 s | 15.22 s | 1,727 |
| 16 | 19.37 s | 30.58 s | 1,719 |
| 32 | 34.93 s | 61.09 s | 1,721 |

Aggregate prefill is **flat ~1,720 t/s from c1 to c32** — compute-bound, queuing is
perfectly linear, no throughput penalty for concurrency.

### Decode

| scenario | speed |
|---|---|
| c1 single stream | ~103 t/s |
| c4 aggregate | ~230 t/s |
| cold (first req after restart) | ~27 t/s first tokens, warms to 100+ within 1 req |

### Needle-in-a-haystack @ ~248k ctx (5 depths) — quality gate

| depth | result |
|---|---|
| 5% / 30% / 55% / 80% / 95% | **5/5 HIT** |

All needles retrieved exactly, including the 95%-depth probe that ran while two other
benchmark flows hammered the engine.

### DRAM offload (the 24x demo)

| scenario | time | output |
|---|---|---|
| 40k prompt, cold prefill | 26.5 s | `OMEGA-2211 and KILO-77` |
| same 40k after 250k monster evicted GPU cache | **1.1 s** | bit-identical |
| speedup | **24x** | External prefix cache hit confirmed in logs |

### Team load (10 concurrent users, 32 requests, 50–100k ctx each)

| metric | value |
|---|---|
| correctness | **32/32 needles exact** |
| total wall | 1,736 s (~29 min, ~2.3M prefill tokens) |
| per-req p50 / p90 / max | 539 s / 594 s / 607 s |
| errors / OOM / crashes | **0** |

### Oversubscription survival

2× ~254k-token prompts concurrently (~508k demand vs 292k GPU KV): both completed
with **correct needle answers**, serialized cleanly, zero preemptions/errors.

## What was tried and rejected (see docs/TUNING_LOG.md)

- **FP8 KV cache**: hardware-blocked on SM86 (FA3 needs SM90+, Triton needs SM89+). Verified twice.
- **util 0.975 / 0.98**: OOMs the DFlash drafter under ≥8 concurrent spec batches.
- **mnbt 16384**: OOMs GDN prefill transients at 10 seqs.
- **LMCache**: hybrid GDN + spec decode corruption (LMCache#4984 — same rig reported upstream).

## License

MIT. Patches are backports of upstream vLLM PRs #49472, #52771, #52807, #52923
(Apache-2.0 upstream) — re-apply after every vLLM upgrade.
