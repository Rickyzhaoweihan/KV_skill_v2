# AGENT_BRIEF: shared briefing for KV-Skill v2 performance agents (2026-10-20)

Read this first, then docs/BEAT_TEXT.md (goal and levers) and docs/PAPERFORM.md (rules). Repo: /nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2.

## Goal
kv must beat text on the benchmarks. Main setting: 4B encoder (Qwen3.5-4B) prefills the doc once into an operator, and the 9B reader (Qwen3.5-9B) reads it with its own query (paper form).
Current numbers are in outputs/D22_SUMMARY.md: LongHealth at parity; QuALITY −.07; NLI/MCQ −.06 to −.14; span F1 −.14 to −.29.
The span gap is CONTENT (wrong span retrieved), not format (outputs/D22_C3.md).

## The current base (start from it)
- Interface: outputs/D13_4b_reg_w10/d13_best.pt. B: outputs/D11_maps/encB5_9b_4b_post_xf01.pt. Reader layers 3, 7, 11, 15, 19 ← encoder layers (see the log of register_d13.py).
- Training script: scripts/register_d13.py (stage 2: KL(text‖kv) + 0.5·CE + distillation). Data splits: scripts/fit_d13.py TASK_SPLIT.
  New options: --exclude_sets, --exclude_docs (outputs/D13_lever1_leakage.json), --select_min_step, --doc_len, --mem_safe.
- Judge with registration only: scripts/register_d20.py (QuALITY, LongHealth) and scripts/register_d22.py (the D22 sets, outputs/D22_sets/*.json).
- Read outputs/D13_SUMMARY.md, D20_SUMMARY.md and D22_SUMMARY.md before designing anything.

## Hard rules
- The reader NEVER sees the doc text at test time. Registration only (per document); no OOD; no unregistered reads reported as results; no encoder-side read as a method.
- Test docs and questions never enter train, val, registration or RL. Use --exclude_sets/--exclude_docs and keep the leakage assertion.
- Run hygiene: pilot ≤500 steps with ≥2 val checks. Val must beat step 0, or stop and diagnose. Select checkpoints with --select_min_step (step 0 never selectable). A judge result equal to its parent baseline is a bug.
- Key conclusions need the verification protocol in docs/PAPERFORM.md: full n with paired SE, a second seed, an independent recompute, sanity rows, and a leakage check. Otherwise label the result "preliminary".
- Never modify ../delta-skill, ../cross_model or docs/report*.html. Caches and downloads go on turbo (HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home), never /scratch.
- Don't edit shared scripts in ways that change default behaviour. Add new flags (default off) or new files, and keep `bash scripts/run_tests.sh` green.

## GPUs
- Submit ONLY as one plain command: `bash slurm/submit_when_free.sh auto [ENV=...] -- [sbatch args] slurm/<file>.sbatch`. No loops, nohup, `bash -c` or wrappers.
  The script waits for a free slot under the dynamic cap (≤10 KV-Skill jobs in total, shared with the other agents). Run it with run_in_background if it may wait.
- H100s are RESERVED for 27B jobs (the hypernetwork main story). 9B-reader jobs pass `-p drjieliu-l40s` (2× L40S if memory needs it). Only 27B or >48GB jobs use `-p drjieliu-h100`.
  2× L40S per job is allowed. slurm/lanes.sbatch packs several runs into one job (LANE0..LANE7 env vars, one per GPU), e.g. 2 configs on `--gres=gpu:2` use one cap slot.
- CPU-only work (analysis, recompute from jsonl, tests, data prep) goes to the CPU partition: `bash slurm/submit_when_free.sh auto CMD="cmd1 ;; cmd2" -- --job-name=D00cpu_<tag> slurm/cpu.sbatch` (drjieliu-v100, no GPU, 8 CPUs / 64G by default; raise with --cpus-per-task/--mem). It does NOT count toward the GPU cap. Use it instead of running long CPU work on the login node.
- Quota mode: after submitting, END YOUR TURN. Don't poll; the coordinator wakes you when the job finishes or a monitor fires.

## Correctness first (user, 2026-10-20: "code执行的没问题，实验是对的才是最关键的")
Throughput never beats correctness. Before you submit, and again before you report, check that the experiment actually tests what it claims:
- The change is live. The new loss term is non-zero and logged. The new parameters receive non-zero gradients and move. New gates are logged and not stuck at 0.
  A flag-off run reproduces the parent baseline numbers (same val, same seed), so any difference is caused by your change.
- Matched comparison. Same val and judge items, same seed, same step budget, same registration settings as the baseline you compare against. Change one thing at a time.
- The right artifacts are loaded. Print the init checkpoint, B map, layer map, data split sizes and excluded-doc counts at the start of the log, and read them back.
- CPU or smoke test first (tiny model or few steps) for any new code path. Run `bash scripts/run_tests.sh`.
- Lanes (slurm/lanes.sbatch) run the configs side by side in ONE job, sharing its RAM and CPUs.
  Scale `--mem` (≈110G per lane, or less if measured) and `--cpus-per-task` with the lane count.
  Give each lane its own `--out` directory. A failed lane only prints "step failed" and the other lane continues, so check every lane log.
  If you are unsure, use separate 1-GPU jobs instead.
- If a result looks surprisingly good or bad, assume a bug until checked (leakage, wrong checkpoint, wrong items, metric bug).

## Reporting
Write `outputs/<YOURTAG>_SUMMARY.md` with a per-benchmark table (kv / wrong_doc / text / kv−text ± SE), and list job ids in slurm/logs/<YOURTAG>.jobids.
In your final message, give the key number, whether it beat step 0 or the baseline, and the next step.

## Selection-bias note (2026-10-20)
The base `D13_4b_reg_w10/d13_best.pt` was selected at its val PEAK (n=96, noise ±.05). Any continuation from it tends to regress toward ~.62 even with no change, so "val beats step 0" is a biased bar when you continue from it.
Always compare against the MATCHED plain-continuation control (same command without your flags): `outputs/E2_ctrl_pilot296` (n=296, --save_items 1, preferred) or `outputs/E1_ctrl` (n=96).
Use paired per-item comparison (`scripts/e1_paired.py ctrl variant --last 3`). The pilot gate is: beats the control by >2 SE, or at least ≥ the control, before a larger confirmatory run.

## GPU-type matching (2026-10-20, found by E1)
Greedy decodes differ between H100 and L40S (bf16 kernels). The same base on the same val items gives kv .655/.338/.410 on H100 and .644/.348/.442 on L40S, and closed_tail differs too.
So paired comparisons are valid ONLY between runs on the SAME GPU type.
- Log the GPU name at job start (lanes.sbatch already prints nvidia-smi) and record it in every summary row.
- When comparing to an old baseline (D20/D22/D13 judge rows), check which GPU produced it (`grep -m1 -E "H100|L40S" <its log>`). If it differs from yours, re-run the baseline rows on your GPU type within the same job, or run your arm on the baseline's GPU type.
- A difference of ~.01–.03 between GPU types is expected noise, not an effect.

## Submit priority (user-approved 2026-10-21)
`slurm/submit_when_free.sh` is a priority queue. Each waiter registers a ticket in slurm/waiters/. When a cap slot frees, only the best-ranked live waiter submits; rank comes from `slurm/priority.txt` (regex per line, top = first), oldest ticket first within a rank.
H100-only (`-p drjieliu-h100`) waiters compete only with other H100-only waiters.
Current order: token identity (P3/P4) > E2 translator > P1 data > D24 hypernetwork > D17c RL > everything else. Use clear job-name prefixes so the regexes match.

## EXPLORATION MODE (user, 2026-10-22) — supersedes the control-heavy parts above
User: "感觉现在的好像怎么尝试performance都没有提升，你好像感觉那种control搞太多了，太严谨了，现在我需要的是一个可以提升performance的方向，我需要的是effort和exploration".
- No new dedicated control runs for pilots. Compare against EXISTING baseline numbers: lever-1 full val/judge rows, or any control already finished (P3_ctrl, P3_ctrl2000, L1_ctrl once it exists).
- Spend GPUs on treatment arms and bigger swings: more data, longer training, more capacity, combinations. Combine forms that look promising instead of testing one at a time.
- Every full-length run goes to the registered judge vs lever-1 full judge rows on the same GPU type.
- Keep the cheap correctness checks: step-0 equality, the new part is live, no leakage, the right ckpt is loaded, same GPU type when comparing judge rows. Verification (second seed, etc.) happens only after something clearly beats lever-1 full on the judge.
- Exploration is the goal, with EFFORT. Do NOT abandon a direction because val is flat at 500 steps (user, 2026-10-22: "500步没起色就换下一个是绝对的错误"). Evidence: lever-1 full kept gaining on the judge (+.03 to +.07 over the pilot) even though val looked flat after step 1000. Every serious direction gets a full-length run (≥2000–3000 steps) and a registered JUDGE before any verdict. Val is only a sanity check. Drop a direction only if it is broken (NaN, inert part, collapse) or if a full-length run plus the judge shows no gain.

## REGISTRATION + OOD UPDATE (user, 2026-10-22) — supersedes "registration only / no unregistered results / no OOD"
- On the lever-1 full base, per-doc registration adds ≈0 (Qasper +.02, LooGLE +.01, MuSiQue −.02, SCF −.01, CNLI −.01) and hurts on conflict docs (27B FaithEval-CF −.062).
- Main reported method is now the UNREGISTERED read (kv@0) from the shared base: zero reader passes per document.
- Registration is NOT dropped (user: "也不要完全放弃注册"). Report kv_reg alongside as the adaptation option, especially for out-of-distribution documents and tasks.
- OOD evaluation is now IN SCOPE (user: "主要是ood的性能咋样"). Measure the base on datasets/task types never seen in its training mix: kv@0, kv_reg, text and wrong_doc, on the same items. Agent OOD owns this.
- Put RL into BASE training (agent RL / D17d), not per-doc RL registration (test-time cost). A small per-doc RL-registration probe is allowed as exploration.
- Deployment decision (user, 2026-10-22): for a NEW document we must either show the base alone suffices, or always register, or give a cheap label-free METRIC that decides per document. Agent ROUTE owns this (outputs/ROUTE_SUMMARY.md, job prefix D00RT_). Agents should keep per-doc logs (held-out scores per budget, item jsonl) so ROUTE can use them.

## PRIORITY ORDER (user, 2026-10-23)
1. Data scale-up for the shared base interface (DS, MCQ comprehension data, 27B lv-series). This has been the most effective lever.
2. Token injection (P6 residual-stream form preferred; P4 pointer) AND deep-layer read at 23/27/31 (ADDR HID-deep, CAP_big), raised by the user on 2026-10-23 after the ADDR diagnostic showed the span gap lives in the deep layers.
3. Other read math forms (R2, L1, CAP, ADDR arms, 27B H-slots).
Then RL, OOD, ROUTE, KV-RAG and verification judges. slurm/priority.txt follows this order.

## EFFICIENCY RULES (user, 2026-10-23: "你要理解我是为了提升效率")
1. Start every NEW training arm from the CURRENT BEST checkpoint, not from lever-1 full, so gains stack:
   - 9B: outputs/DS_AB_full/d13_best.pt (later: the best D00DS_max8k checkpoint);
   - 27B: outputs/D12b_lv4_s2/d13_best.pt (later: the best D12b_max8k checkpoint).
   Compare against that checkpoint's own kv@0 rows on the same GPU type, run in-job when missing.
2. Evaluate with unregistered kv@0 (+ wrong_doc) only, via eval_d13 / eval_d12p. No per-doc registration unless the run is specifically about registration (OOD / ROUTE). This is much cheaper, and registration adds ≈0 on the trained base.
3. Save a checkpoint every 1000 steps and run a cheap kv@0 sweep, instead of relying on val selection plus an expensive registered judge.
4. Reuse existing text rows (item-identical, same GPU type); never recompute text.
5. Don't re-run anything that already exists; check outputs/ first.
