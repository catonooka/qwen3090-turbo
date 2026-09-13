#!/bin/bash
# Qwen3.8-27B dual-3090 "ultrafast" launcher — club-3090 recipe, bare-metal
source /home/aisever/vllm-env/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=1
# expandable_segments off: incompatible with KV offloading connector (VMM remap breaks pinned KV)
export VLLM_USE_FLASHINFER_SAMPLER=0
# custom all-reduce crashes on this rig (invalid argument in custom_all_reduce.cuh) — disable it; NCCL still uses NVLink
export VLLM_SKIP_P2P_CHECK=1
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/aisever/models/frozenlock-int4 \
  --served-model-name qwen3.8-27b \
  --disable-custom-all-reduce \
  --quantization auto_round \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 256000 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 10 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype bfloat16 \
  --attention-backend FLASH_ATTN \
  --trust-remote-code \
  --enable-prefix-caching \
  --kv-offloading-backend native --kv-offloading-size 48 \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"dflash","model":"/home/aisever/models/dflash2-w4a16","num_speculative_tokens":7}' \
  --host 0.0.0.0 --port 8081
