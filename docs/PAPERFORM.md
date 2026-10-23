# Paper-form read: the project goal (2026-10-19)

The user's goal: make the PAPER's read work cross-model, with sustained effort, longer training and several
methods in parallel. Do not settle for the encoder-side read (it is the upper bound and the teacher, not the method).

## The paper's read (non-negotiable form; arXiv 2608.05475v2, see docs/DIRECTIONS.md)
- Operator M = (U, W) harvested once from the small encoder's frozen prefill of the document, never modified
  (text-derived construction), or a compact operator optimized by reward (reward-learned construction).
- Read inside the READER, on a separate branch: q = norm(A · RMSNorm(h_reader)), Read(M, q) = softmax(U q / τ) · W
  (per head is fine), write-back h' = h + γ · g · B · Read (or the attention-output replacement D10 validated —
  report which). Backbones frozen; the interface I = (A, B, gates, gains) is what trains.
- At inference ONLY the reader runs; the encoder is NOT run on the question. The skill is a static tensor.
- Why it matters (user): the reader's context drives addressing (multi-turn, mid-reasoning queries), and a library of
  operators can be routed by the reader in one pass, i.e. KV-Skill as retrieval memory / RAG.

## What is known
- Encoder-side read (encoder queries its own cache over the question, closed-form B): 2B→9B 0.68/0.61, 4B→9B
  0.82/0.76 (LongHealth n=400 / QuALITY n=1057); text 0.86/0.83. It gives per-token teacher targets for free: the
  encoder's attention distribution over document positions and its read-out, at matched layers, for any question.
- Reader-query reads that FAILED: answer-only KL+CE from scratch (flat attention → memorization), ridge query map
  (R² ≤ 0.45), attention-fitted A to the reader's in-context attention, key-side maps.
- D13 (running): A distilled label-free from the encoder's question-pass attention — held-out JS 0.094, cos 0.94;
  LongHealth fold 0 MLP-A 0.5375 vs wrong 0.3875 (+0.15, tail 0.4375) PASS; QuALITY n=300 MLP-A 0.563 vs 0.473
  (+0.09, tail 0.51) not yet. Stage 2 = paper KL+CE on top (running). Code: scripts/eval_d13.py, outputs/D13_maps/,
  D13 modules (read outputs/D13_SUMMARY.md when it exists; ask D13 via its files, do not edit them).
- Reward-learned compact operator (D4): reproducible ≈ +0.08; scaled-up run in progress.

## Judge (unchanged)
Held-out documents; kv − wrong_doc ≥ +0.10 (paired se, n) AND kv ≥ closed_tail (reader prefills last 64 doc
tokens); report closed_book, text, mean gen tokens, held-out attention JS / read-out cosine to the encoder teacher.
Primary tasks: LongHealth (all 5 folds for final numbers; fold 0 n=80 for iteration) and QuALITY (n=300 iteration,
full 1057 final). Never fit or select on test documents of the judged task.

## Protocol
DYNAMIC job cap (RUNNING + PENDING, all agents; user rule 2026-10-19): `bash slurm/cap_now.sh` → 5 when other users queue on the lab partitions, 8 normally, 10 when idle; ALWAYS submit with `bash slurm/submit_when_free.sh auto ...` as ONE plain command (no loops, nohup or wrapper scripts — the permission allow rule only matches that form) (paper-form directions have priority now); PREFER 1-GPU jobs (user 2026-10-19: multi-GPU jobs rarely start — free GPUs appear one at a time per node); request only what one GPU needs (≈6 CPUs, ≤110G RAM for 9B work; CPUs and RAM on lh0303/lh0304 run out before GPUs do); use ≥2 GPUs only when a model cannot fit one GPU; `slurm/lanes.sbatch` with LANE0 only also works (LANE0..LANE7 env vars, one lane per GPU, steps separated by `;;`, e.g. `bash slurm/submit_when_free.sh auto LANE0='python a.py ...;;python b.py ...' LANE1='python c.py ...' -- --gres=gpu:2 --job-name=D16_s2 slurm/lanes.sbatch`);
ALWAYS submit with `--partition=drjieliu-h100,drjieliu-l40s` so a job starts on whichever partition is free first
(user instruction); make jobs memory-safe for 48 GB L40S; the 27B reader (D12) has priority for slots;
`slurm/submit_when_free.sh`; tag everything with your direction prefix; `test -e` before every Write; additive edits;
`bash scripts/run_tests.sh` green; never modify ../delta-skill or docs/report*.html. Summary in outputs/<tag>_SUMMARY.md.
When a run shows held-out gains below the bar, continue (more data, steps, better init, regularizers) before
switching approach; report the objective used for every number (distillation / KL+CE / reward).

## Target (user, 2026-10-19 evening): TEXT PARITY
Passing the +0.10 gate is done (D13: LongHealth 0.7625, QuALITY 0.633). The goal now is to make the paper-form read
**match or exceed the reader reading the document as text** through registration / training:
9B reader text = LongHealth 0.825–0.86, QuALITY 0.81–0.83; 27B reader text = 0.8625 / 0.935.
Current gaps: LongHealth −0.06, QuALITY −0.18 (9B). Report "kv − text" with se for every row from now on.
User hypothesis to keep in mind: deeper layers of larger models lean on their own prior knowledge, while the small
encoder's prefill is "cleaner"; smart per-layer injection for special scenarios (e.g. knowledge conflict) comes LATER —
do not start it now.

## Priority (user, 2026-10-19 night): fix the READ first, RL later
RL / GRPO (D17) is PAUSED: reward training is for polishing final performance once the read reaches text parity, not
for fixing the read itself (D17 GRPO on D13: LongHealth f0 0.750 vs 0.7625, no gain). Current priorities: stage-2
supervised KL+CE work on the QuALITY gap (D13, D15), 27B reader (D12), multi-turn A + library routing (D18).
RL trigger (user): only start RL/GRPO once kv − text is within ~0.02–0.03 of zero, i.e. not significantly below text, on BOTH LongHealth and QuALITY (full n) for that reader.
D4 (user, 2026-10-19): let running jobs 27940012 and 27940967 finish; submit NOTHING new for D4 (seed 1 held) until the read reaches text parity.

## METHOD DECISION (user, 2026-10-19 night): REGISTRATION ONLY, no OOD
Every document is registered (short per-document training) before it is read; we do NOT pursue zero-shot / out-of-
distribution reading of unregistered documents (leave-LongHealth-out and zero-shot ContractNLI are diagnostics only,
not targets). Registration of a test document may use only label-free signals from the document itself (encoder-
attention distillation, KL to the 9B text teacher on self-generated questions / recitation, pseudo-labels) — NEVER the
test questions or answers. D20 owns per-document registration (outputs/D20_SUMMARY.md).
Clarified (user): registration is PER DOCUMENT only — one shared base interface for all tasks, no per-domain registration step.
NO UNREGISTERED READS (user, 2026-10-19 night): do not evaluate kv on documents that were not registered — that is OOD.
Every kv row from now on must be AFTER per-document registration (D20's method). Unregistered numbers may appear only
as the step-0 point of a registration curve. Base-interface training is fine; new datasets (ContractNLI, Shakespeare-CF,
LooGLE, CLIPPER) are evaluated only through registration.

## GOALS (user, 2026-10-20): cost advantage AND performance advantage, plus smart injection for specific problems
- Cost: at test time the LARGE reader must never read the document text, not even for registration. Otherwise
  there is no advantage over its own prefix caching. Allowed: base read, encoder-only registration, or amortized
  registration (D24 hypernetwork: operator -> adapter in one forward).
- Performance: look for regimes where kv BEATS text: very long docs (LooGLE 64k, CLIPPER; D22), multi-document /
  library (D18 router), knowledge conflict (D25 smart injection).
- The encoder-side read is not our method; it is excluded from comparisons.

## RUN-HYGIENE RULES (2026-10-20, after D12p_deep wasted ~2.5 GPU-h judging a step-0 checkpoint)
1. Every training run that adds new trainable parts must show a val improvement over step 0 within the first ~3 val checks; otherwise stop it and diagnose. `slurm/check_stall.sh` flags STALL / STEP0 / INERT (newly added layers with gates still exactly 0 at eval).
2. Checkpoint selection must exclude step 0 when the run's purpose is to train new components (`--select_min_step`), and the judge step must log the gates of the new layers.
3. Before writing a judge result anywhere: if it equals the parent baseline to 4 decimals, treat it as a bug, not a result.
4. New recipes get a short pilot (≤500 steps, 2 val checks) before the full multi-hour run.
5. Never pass `-p drjieliu-h100` alone unless the job needs 80GB (27B). Headers list both partitions; pinning H100 left L40S idle on 2026-10-20.

## VERIFICATION PROTOCOL FOR KEY CONCLUSIONS (user, 2026-10-20: "主要的重要的结论时一定要多加验证")
Before any headline claim (kv ≥ text, kv > text, a method "works", a negative that changes direction) goes on the results page:
1. Full test n (not a subsample), with paired SE; the claim must exceed ~2 SE. (Precedent: the D24 hypernet +.037 at n=300 was +.005 at n=1057.)
2. A second seed, or a second independent run, agrees.
3. Recompute the metric from the raw per-item outputs with a separate script (not the training job's own summary).
4. Sanity rows are present and sensible: wrong_doc, closed_tail, text; kv−wrong large; nothing identical to its parent baseline.
5. Leakage re-checked: test doc ids/questions absent from train/val/registration/RL (assertion log + near-duplicate check).
6. A focused code review of the eval and selection path for any new code that produced the claim.
Claims that have not passed these checks are labelled "preliminary" on the page.
