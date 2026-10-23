# Backbone scouting notes (scratch) — 2026-10-18

## Sources
- configs: /nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub/models--*/snapshots/*/config.json
- param counts: count_params.py (safetensors headers; 'visual'/'vision'/'audio' split)
- tokenizer sha256 (first 16 hex): Qwen3.5-2B/4B/9B/27B tokenizer.json 5f9e4d4901a92b99, merges a9d356d7bdf1ef49, vocab.json ce99b4cb2983d118 — IDENTICAL
  chat_template.jinja differs 2B (273d8e0e) vs 4B/9B/27B (a4aee8af): only the enable_thinking default (2B OFF, others ON)
- gpt-oss tokenizer 0614fe83..., Qwen3-4B-2507 aeb13307... (different vocabs)
- Qwen3.6-35B-A3B: HF raw config.json + model.safetensors.index.json (WebFetch) — NOT downloaded locally
- Qwen3.5-0.8B: HF raw config.json (WebFetch) — NOT downloaded locally
- cluster: scontrol lh0304 = gpu:h100:8 (80GB HBM3, confirmed by probe), lh0303 = gpu:l40s:8; NO H200 partition exists
- env: /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35 torch 2.12.1+cu130, transformers 5.13.0, triton 3.7.1; fla via PYTHONPATH=KV_skill/fla_kernels (0.5.2); causal_conv1d / flash_attn / kernels MISSING
- probes: probe_mem.py -> probe_9b.log / probe_27b.log / probe_oss.log (srun on lh0304, 1xH100)
- memory formulas: mem_arith.py
