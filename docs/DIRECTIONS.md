# Parallel directions brief (2026-10-19) — make document KV-Skills work, inside the paper's theory

Repo: `/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2` (read `README.md`, `docs/XENC_SPEC.md`,
`docs/PLAYGROUND_PROPOSAL.md` §7, and the results page `docs/report.html` first).
Env: `conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35`; tests `bash scripts/run_tests.sh`
(pytest in `.deps`). Slurm launchers in `slurm/` (account drjieliu; `--partition=drjieliu-h100,drjieliu-l40s`
for ≤9B readers). HF cache + PYTHONPATH exports are inside every launcher.

## Non-negotiable theory constraints (the paper, arXiv 2608.05475v2)
1. A KV-Skill is an EXTERNAL factorized associative operator `M = W Uᵀ` (U = keys, W = values, slots × d_s),
   read on a separate residual branch: `q = norm(A·RMSNorm(h))`, `Read(M,q) = W(Uᵀq)/√m` (softmax over slots is
   the "sharp" variant already in deltaskill), `h' = h + γ_ℓ g_ℓ B Read`. No prompt positions, no attention
   KV-cache entries, backbone frozen. Keep this read form.
2. Two legitimate constructions: (a) **text/document-derived**: U, W harvested from a frozen prefill, fixed
   forever; registration trains ONLY the interface I = (A, B, gates, gains). (b) **reward-learned**: a compact
   operator (U, W independent, e.g. 32×256) optimized jointly with its interface from task reward (group-normalized
   advantages), optionally initialized from text. Imitation-only training of the operator is what the paper
   distinguishes itself from (Cartridges/SoftSkill) — if you train an operator, use a verifier reward (GRPO-style)
   or justify a reward-shaped objective.
3. "U = W = H̄" was the paper's choice for text; `deltaskill/derive.py` explicitly anticipates splitting K from V
   (e.g. attention keys for U, attention values for W). That is inside the design space.
4. Cross-model (encoder ≠ reader) is this project's addition: A: d_reader→d_encoder, B back. Normalized depths.

## What is already measured (do not re-run; build on it)
- Raw per-token residual operators (one prefill, ≤512-token chunks, routing over chunks) carry NO document
  content through the registered read: kv − wrong_doc ∈ [−0.03, +0.03] on QuALITY (n=300) and LongHealth (n=80),
  for L/L (9B→9B) and S/L (9B reads 2B), on held-out docs AND on unseen questions of the training docs.
- Centering slots per depth (subtract corpus mean; `--center outputs/centers/center_<enc>.pt`) removes a
  document-independent +3–8 pt lift but adds no content.
- **Read-attention probe** (`scripts/probe_read.py`): the registered A yields near-uniform attention over a
  document's ~11k slots (entropy 8.6–8.9 / max 9.3 nats; mass on gold-answer slots = chance). The read is a
  document-wide average. → The failure is ADDRESSING: A never learns to retrieve, because answer-level KL/CE on
  6 tokens is minimized by a document-independent shift, and gradients through a flat softmax vanish.
- Injection magnitude (‖Δh‖/‖h‖ ≈ 1.3–1.6%) is comparable to published rank-1 skills; not the limiter.
- Ladders (n=200): QuALITY closed/text 2B .37/.60, 4B .50/.81, 9B .47/.81, 27B .58/.94; LongHealth 2B .28/.79,
  4B .41/.87, 9B .44/.88, 27B .49/.90.
- Compute: 9B registration ≈ 2.2 s/step, 19 GiB; 300 steps ≈ 12 min; eval 6 conditions ≈ 4 s/item.
- Encoder caches: 2B for all QuALITY (265 docs) + LongHealth (20); 9B for ~140 QuALITY docs + LongHealth.
  Cache dir `/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache` (fp16, ρ-keyed, chunk 512).

## Judge (identical for every direction)
`scripts/eval_doc.py` on held-out docs: conditions closed_book / text / kv / wrong_doc / null / oracle_chunk.
Primary number: **kv − wrong_doc** (pre-registered pass ≥ +0.10; n ≥ 80) **AND do-no-harm: kv ≥ closed_book (or closed_tail when the reader prefills a document tail)** — a wrong-document collapse (e.g. D10_tail_ridge_q: kv .44 < closed_tail .51, wrong .33 with 12.5 generated tokens) can fake the gap; always report mean generated tokens per condition. Secondary: `scripts/probe_read.py`
gold-slot mass vs chance (did addressing emerge?). Also report mean generated tokens (collapse detector).
Quick loop: QuALITY, `--max_docs 20` train docs, val for selection, test n=300; or LongHealth `--test_fold 0`.

## GPU protocol (hard)
- Shared cap: **at most 5 jobs in `squeue -u rickyhan` at any time across ALL agents**. Before every `sbatch`,
  run `squeue -u rickyhan -h | wc -l`; if ≥ 5, wait (sleep 120 and recheck). Each direction may hold at most
  **1 running job at a time** unless the queue has free slots.
- Prefix every run name and job name with your direction tag (`D2_`, `D3_`, …) and write outputs under
  `outputs/<tag>_*`. Never cancel another direction's jobs. Log every job id in `slurm/logs/<tag>.jobids`.
- Launchers: `slurm/register_doc.sbatch`, `slurm/eval_doc.sbatch`, `slurm/encode_docs.sbatch`,
  `slurm/probe_read.sbatch` (env vars READER/ENCODER/TASK/RUN + extra CLI args pass through).

## Code protocol (hard)
- Make ADDITIVE, minimal edits; prefer new modules/flags over rewriting. Other agents edit the same files
  concurrently: use precise `Edit` anchors, re-read a file right before editing it, never reformat whole files,
  never run the data preparers again. Run `bash scripts/run_tests.sh` before and after your edits; if a test you
  did not write breaks, fix it or revert your edit.
- Do not modify `../delta-skill`. Do not touch `docs/report*.html` (the orchestrator regenerates them).
- Final report ≤ 800 words: what you changed (files), what you ran (job ids, configs), the judge numbers with n,
  and a one-paragraph verdict. Put the numbers also in `outputs/<tag>_SUMMARY.md`.
