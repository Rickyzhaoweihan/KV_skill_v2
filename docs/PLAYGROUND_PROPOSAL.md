# KV-Skill v2 — Experimental Playground Proposal

**Project:** "Small Encoder, Large Reader" — a small frozen model encodes each document once into a
KV-Skill operator; a large frozen reader reads it through a registered cross-model interface at
learned, query-dependent injection depths. Documents never enter the reader's context.
**Date:** 2026-10-18. **Status:** proposal; nothing in this document has been run yet.
**Inputs:** the published KV-Skill paper (arXiv 2608.05475v2), the Oct 12 design doc, the Oct 13 memo,
five reference papers (CoRect, DistractionIF, Size–Fidelity Paradox, RaReCache, Heo et al.), and
thirteen code/literature/infra audits run on 2026-10-18 (scratchpad `reports/`, `codeplan/PLAN.md`,
`backbone/NOTES.md`, `dl/REPORT.md`).

---

## 0. Decisions in one screen

| Question | Decision | Why (evidence) |
|---|---|---|
| Stage-A reader | **Qwen3.5-27B** (primary), **Qwen3.5-9B** (size axis) | 27B registration measured at **53.9 GiB on one H100** at 16k teacher context, so it never needs sharding; dense (no MoE routing noise); strongest in-family long-context reader (LongBench v2 60.6). |
| Encoder | **Qwen3.5-2B** (primary), **Qwen3.5-4B**; 0.8B optional | Tokenizer byte-identical across 2B/4B/9B/27B → slots align token-for-token with what the reader would have seen; Size–Fidelity shows ≤4B compressors are the most faithful. |
| Cross-family check | gpt-oss-20b **only at ≤4k context**; **drop Gemma-4-12B** | gpt-oss is a weaker reader than Qwen3.5-4B (LongBench v2 45.6 vs 50.0) and OOMs at 16k (eager attention); Gemma never learned to read a skill (identity ≈ 0). |
| Compute | Every Stage-A job is a **single-H100** job; the second card doubles seeds/docs | No H200 partition exists on the cluster; 2×H100 is the real ceiling. |
| P0 gate | **Document-content margin on L/L first** (27B reads its own operator) | All prior strong results are procedural and rank-1; LongHealth/QuALITY at 12:1 pooled compression gave kv ≈ wrong ≈ base. Chunk-level library (≤512 tokens/operator, no pooling) is the untested bet. |
| Core datasets | **QuALITY, LongHealth, LastingBench (counterfactual LongBench HotpotQA/2Wiki/MuSiQue), Faithfulness-QA article-reassembled**, plus an in-house **GLKB-style PMC-OA pool** | Only sets that satisfy reuse + 1k–30k tokens + reasoning gap + conflict variant; see §3. |
| Rejected datasets | Everything in Heo et al. and RaReCache except LongBench-E; PubMedQA; NQ-Swap/ConflictQA/FaithEval as *primary* | All ≤1.1k tokens, one context per question, mostly closed-book-solvable. |
| Baselines (run) | L-text, S-text, closed-book, **LongLLMLingua (2B→27B)**, **Heo ridge map (reimplemented)**, **C2C fuser (reimplemented)**, **SoftSkill prefix (L/L)**, **CoRect (reimplemented)** | None of Heo/RaReCache/CoRect/REFRAG released code; these are the runnable, reviewer-expected set. |
| Loss | **Trace-KL** (KL over the teacher's reasoning trace + CE on answer span), not answer-only KL | Answer-only registration collapses to 6-token direct answers and destroys compute-heavy tasks (MedCalc 27 vs base 81; JEEBench kv .28 < base .75). |
| Stage B | **DeepSeek-V3.1 (685B)** → **Kimi-K2-Instruct-0905 (1.03T)**, with **Qwen3.5-397B-A17B** as in-family point | Native transformers, MIT, thinking switchable; V4 has 4 parallel residual streams (hook ill-defined); FP8 linears are inference-only → train in bf16 on 16×B200. |

---

## 1. Thesis and hypotheses (unchanged from the memo, with one reordering)

- **H1** S/L beats the large model reading raw text (procedural content; precedent 77.2 vs 52.0).
- **H2** S/L ≈ L/L at much lower encoding cost.
- **H3** S/L > L/L on prior-conflicting / instruction-noise content.
- **H4** Mid+late or dynamic layer injection overturns prior override without hurting clean accuracy.

**Reordering.** The memo places capacity (P4) last. The code audits force it first: every strong
KV-Skill result to date is a *procedural* skill carried by one direction per layer (rank-1 = full
rank on LiveMath; 96% of operator energy in the top singular vector; a shared *random* operator
recovers ~half the LiveMath gain). The only document-grounded runs (LongHealth, QuALITY) carried no
content — but they used **one pooled operator per 12k-token record (12:1 compression) on a 4B
reader**. Nothing has tested chunk-level operators with routing. H1–H4 are all conditional on
content going through, so **content margin is the P0 gate**.

---

## 2. Backbones and compute

### 2.1 Stage A (now): ≤2×H100 80 GB (no H200 partition exists; H200 rows are arithmetic only)

| Role | Model | d | L | full-attn layers | Verified cost |
|---|---|---|---|---|---|
| Reader (primary) | Qwen3.5-27B | 5120 | 64 | 3,7,…,63 (16) | 50.1 GiB weights; teacher 16k → 53.3 GiB; student fwd+bwd Q=256 → 53.9 GiB; ~1.7 s/step @4k → **≈8.5 min per 300-step registration** |
| Reader (size axis) | Qwen3.5-9B | 4096 | 32 | 3,7,…,31 (8) | 16.7 GiB; fits L40S; ≈3.5 min/registration |
| Encoder (primary) | Qwen3.5-2B | 2048 | 24 | 3,7,…,23 (6) | 16k prefill ≈ 0.2 s; runs offline, never co-resident |
| Encoder (alt) | Qwen3.5-4B | 2560 | 32 | 3,7,…,31 (8) | — |
| Encoder (optional) | Qwen3.5-0.8B | 1024 | 24 | 6 | not cached; download if the S/L vs L/L contrast needs a smaller encoder |
| Cross-family | gpt-oss-20b | 2880 | 24 | odd layers (12) | 39 GiB bf16 after MXFP4 dequant; **D≤4k only** (eager attention: 63.6 GiB @8k b=1, OOM @16k); reasoning channel cannot be disabled |
| Optional | Qwen3.6-35B-A3B | 2048 | 40 | 10 | not cached (72 GB); ~66 GiB bf16 → 2×H100; loading unverified |
| Dropped | Gemma-4-12B | — | — | — | skill identity ∈ [−0.017, +0.016] in delta-skill; never learned to read |

Interface sizes: A 5120→2048 and B 2048→5120 fp32 = 42 MB each; optimizer state 0.31 GiB. Trivial.

**Affordable 2×2 cells** (reader ∈ {27B, 9B}, encoder ∈ {2B, 4B}): S/L (2B→27B, 4B→27B), L/L
(27B→27B; the encoder pass is the reader's own no-grad prefill), S/S (2B→9B) all fit **one H100 at
batch 1–4, D≤16k**. L/S (27B→9B) needs both resident → 2×H100 or b=1 on one.

**Mandatory fixes before the first run** (from the backbone + code-plan audits):
1. Teacher pass must use `logits_to_keep` + `use_cache=False` (the canonical `register_benchmarks.py`
   does not; `longhealth_kv.py` does). Full-vocab logits at 16k are 7.6 GiB.
2. `max_memory` is hardcoded `"76GiB"` in `cross_model/xskill/multi_backbone.py` — make it a flag.
3. Keep `PYTHONPATH=…/fla_kernels`; the "fast path not available" warning is a false alarm (the
   Gated-DeltaNet chunk kernel *is* fla's Triton kernel; only causal-conv1d is missing). Optionally
   `pip install --target fla_kernels causal-conv1d`.
4. `enable_thinking=False` on every Qwen3.5 template (2B defaults off, 4B/9B/27B default on).
5. Compute student CE manually from `logits_to_keep=m+1`, not via `labels=`.

### 2.2 Stage B (later): 50×B200 (~9 TB HBM), 700B-class reader — see §8.

---

## 3. Datasets

### 3.1 Selection criteria (from the brief)
C1 documents reused across many queries · C2 1k–30k tokens per document (chunkable to ≤512-token
operators, with a length sweep across d_b≈2.5k) · C3 large (large−small) gap *with docs in context*
while small-model single-hop extraction is high · C4 a prior-conflict variant exists or is buildable
· C5 span-F1 / MCQ scoring, not numeric, not needle, not closed-book-solvable · C6 multi-hop across
chunks (routing) · C7 downloadable, licensed, offline-materializable · C8 published Qwen-class
numbers at several sizes.

### 3.2 Verified statistics (login-node download, Qwen3.5 tokenizer, 2026-10-18)

All downloads succeeded on the login node (raw copies under the session scratchpad `dl/data/`,
~1.1 GB total). Token counts use the real Qwen3.5 tokenizer (shared by 2B/4B/9B/27B).

| Dataset (split) | Docs | Qs/doc (med) | Doc tokens (med / p90 / max) | Answer / metric | Verdict |
|---|---|---|---|---|---|
| QuALITY (val / train) | 115 / 150 | 18 / 17 | 6,673 / 7,713 / 8,719 | 4-way MCQ / acc | **Core**; 1,065 "hard" val Qs |
| LongHealth | 20 patients, 400 Qs | 20 | 11,668 / 13,129 / 13,148 | 5-way MCQ / acc | **Core regression test** (local copy exists) |
| LastingBench HotpotQA-CF | 200 (141 rewritten) | 1 | 14,976 / 16,975 / 17,719 | span / F1, EM | **Core conflict** (H3/H4) |
| LastingBench 2Wiki-CF | 200 (169 rewritten) | 1 | 6,439 / 13,498 / 16,890 | span / F1 | **Core conflict** |
| LastingBench MuSiQue-CF | 200 (122 rewritten) | 1 | 16,716 / 17,221 / 18,085 | span / F1 | **Core conflict** |
| LastingBench MultiFieldQA-CF | 150 (116 rewritten) | 1 | 7,315 / 12,152 / 16,726 | span / F1 | conflict, single-doc |
| MuSiQue (val, original) | 2,417 items; 21,100 unique paragraphs for 48,315 slots (**2.3× paragraph reuse**) | — | item 2,322 / 3,124 / 5,154; paragraph 97 / 198 / 1,391 | span / F1 | **Core multi-hop**; 2-hop 1,252, 3-hop 760, 4-hop 405 |
| CUAD (test) | 102 contracts (510 total) | 41 asked, **12 answerable** | 5,436 / 22,965 / 64,886 | extractive span / F1 | sleeper: highest reuse; extraction may be easy for 2B |
| QASPER (test / dev) | 416 / 281 papers | 3 / 3 | 4,533 / 7,462 / 29,895 | span, yes/no, abstractive / F1 | secondary (low reuse) |
| FaithEval counterfactual | 1,000 unique ctx | 1 | 386 / 497 / 649 | MCQ / acc | probe only (short) |
| FaithEval inconsistent | 1,500 | 1 | 741 / 1,902 / 4,236 | span / acc | probe |
| FaithEval unanswerable | 2,492 | 1 | 322 / 912 / 3,574 | span / acc | probe |
| NarrativeQA (test) | 355 stories (178 scripts, 177 books) | 30 | scripts ≈ 29k words (14k–79k tokens sampled); books ≈ 77k words (6k–311k tokens) | free-form / F1 | shortest scripts only; otherwise too long |
| InfiniteBench En.QA / En.MC | 69 / 58 books | 5 / 4 | 147k / 402k / 402k (first 10 books) | F1 / acc | out of range |
| AA-LCR | 30 sets (230 docs) | 3 per set | 109,290 / 119,990 / 121,920 per set | free-form, LLM judge | yardstick only (Qwen3.5 2B/4B/9B published) |
| FRAMES | 824 Qs over 2,507 unique Wikipedia pages | ~3 pages per Q | — | free-form; 202 numeric answers | routing test after numeric filter |
| LongBench v2 | 503 MCQ (single-doc QA 175, multi-doc 125, ICL 81, code 50, dialogue 39, structured 33) | 1 | 10,176 / 102,249 / 548,922 / 4.5M; **24 items ≤16k, 111 ≤32k, 182 ≤64k** | MCQ / acc | yardstick only (Qwen3.5 2B/4B/9B/27B published); use the ≤32k subset |
| LongBench v1 (hotpot / 2wiki / musique / multifieldqa / qasper / narrativeqa) | 200 / 200 / 200 / 150 / 200 / 200 (narrativeqa: 20 unique stories, 10 Qs each) | ~1 | 15,025 / 6,681 / 16,758 / 7,313 / 4,636 / 31,643 (medians) | span / F1 | originals paired with LastingBench-CF; `_e` length-bucketed variants exist for hotpot/2wiki/multifieldqa/qasper |
| ConflictQA (PopQA / StrategyQA, ChatGPT files) | 7,947 / 1,245 | 1 | counter-memory 125 / 157 / 270 (PopQA) | span, yes-no / memorization ratio | generated-text control only (short) |
| NQ-Swap mirror | 3,621 | 1 | 774 / 846 / 1,585 (5 passages per item) | span / EM, MR | comparability anchor (short) |

**License caveats found during download:** LastingBench has no LICENSE file (GitHub reports none);
FaithEval, the NQ-Swap mirror, InfiniteBench and LongBench v1 state no license on their cards; the
`emozilla/quality` and `dgslibisey/MuSiQue` mirrors state none although the originals are CC BY 4.0.
Resolve before publication (email LastingBench authors; cite originals for QuALITY/MuSiQue).
**Loader-script repos** (CUAD, QASPER, LongBench v1) do not work with `datasets` 5.x; use the raw
GitHub / S3 / `data.zip` sources, all of which worked. The login node kills processes at ~3 GB RSS, so
read LongBench v2's `data.json` line by line. FRAMES ships only Wikipedia URLs; article text must be
fetched separately.

Known from the literature audits (not measured locally): Faithfulness-QA 99k items over SQuAD /
TriviaQA with `original_context` / `modified_context` (MIT); ConFiQA MR/MC 6k each, ~330 tokens (one
~100-word passage per hop), license unstated.

### 3.3 Tiers

**Tier 1 — core document QA (H1/H2, P0 content gate).**
- **QuALITY** — cleanest reuse/length fit; closed-book prior is real (Qwen3-8B 43.6 no-context vs 82.5
  oracle), so always report closed-book. Fiction → counterfactual edits are easy.
- **LongHealth** — structurally ideal (20 Qs/record, fictional → zero contamination, Task-2 has
  distractor patients = routing); it is the **regression test** the chunk-level library must pass
  where the pooled operator failed (kv 0.342 = base 0.342 ≪ text 0.708). Only 20 docs → cross-validate.
- **LastingBench / LongBench-v1 multi-hop (HotpotQA, 2Wiki, MuSiQue)** — the only long-context
  (5k–60k) conflict set with Qwen3 numbers (8B/32B EM orig→CF: 2Wiki 0.83→0.57/0.53; Hotpot
  0.64→0.43/0.41). Serves H1/H2 (originals) and H3/H4 (counterfactuals) from one pipeline.
- **QASPER** — secondary; low reuse (3.2 Qs/paper) and extraction-heavy (small reader does well), but
  clean F1 and a Qwen3-8B ladder (19.2 → 56.7).

**Tier 2 — prior-conflict (H3/H4).** Direction matters: *entity substitution inside real text* makes
larger models override context more (the regime H3 needs; Qwen3 family ρ = +0.72–0.85 between size
and prior reliance); *coherent LLM-generated counter-evidence* makes larger models follow context
more (ConflictBank). Use the first as the headline and the second as the control.
- **Faithfulness-QA reassembled into full SQuAD articles** (1–5k tokens, ~100 Qs/article, MIT):
  apply each item's entity substitution across the whole article; NLI/leak filter to drop articles
  whose other sentences re-imply the original. This is the project's own article-scale conflict set.
- **ConFiQA MR/MC** — per-hop passages map one-to-one onto ≤512-slot operators (native routing test;
  MC conflicts every hop). License unstated → email authors.
- **FaithEval-counterfactual** (1k MCQ, ~420 tokens) and **NQ-Swap** — cheap sanity probes and the
  comparability anchor every 2025–26 conflict paper reports (CoRect: Qwen2.5-7B 70.5→80.2).
- **ConflictBank** (CC BY-SA, 553k, Qwen1.5 0.5B→72B M_R curve) — the generated-text control.
- **DistractionIF-style benign instruction noise** — not released; port its trap taxonomy as an
  augmentation over QuALITY/LongHealth chunks (H3 instruction-noise arm).

**Tier 3 — in-house biomedical pool ("GLKB-QA").** No existing biomedical set satisfies C1–C8
(PubMedQA = 1 Q per 250-token abstract; LitQA2 ships no text; MIMIC sets need credentialing;
PaperSearchQA is the closest but abstracts are ~330 tokens). Build: 1k–5k PMC-OA CC-BY full-text
papers (3k–10k tokens; MeSH genetics/genomics/cancer/immunology, 2018–2026; fetch via BioC API
*now* — PMC FTP datasets are being removed), 10–20 Qs/paper generated locally by Qwen3.5-27B /
gpt-oss-20b, **three-way filter = criterion C3 measured per item**: keep a Q only if (a) 27B fails
closed-book, (b) 2B/4B succeeds with the gold chunk in context, (c) 27B succeeds with the full paper.
Length sweep: gold chunk (≤512) → section (1–2k) → paper (3–10k) → paper+4 distractors (15–30k).
Multi-hop via GLKB entity links. Conflict variants: (i) reversed-effect rewrites, (ii) MedChangeQA's
512 Cochrane verdict flips (old vs updated abstracts), (iii) retracted-paper PMIDs (Pub-Guard).
≈2 GPU-days; expert spot-check 300 items. External anchors: PaperSearchQA test, BioASQ factoid.

**Tier 4 — sanity yardsticks (no reuse; eval-only).** LongBench v2 (Qwen3.5 2B/4B/9B/27B =
38.7/50.0/55.2/60.6 published), AA-LCR (2B/4B/9B = 25.6/57.0/63.0; LLM-judged, 100 Qs).

**Tier 5 — procedural regression (must not regress vs the published paper).** LiveMath, SearchQA,
STaRK-Prime with the SkillOpt-optimized skills; text score must stay ≥95% under the new loss.

### 3.4 Rejected (with reason)
Heo et al. sets (ARC/HellaSwag/WinoGrande/MMLU/GSM8K/WikiText/CoQA: all ≤1.1k tokens, closed-book);
RaReCache sets except LongBench-E; DistractionIF as a benchmark (unreleased, ~350 Chinese tokens,
LLM-judge); CoRect's own conflict sets (unreleased; NQ-Swap is the recipe); PubMedQA, MedQA/MIRAGE
(closed-book-solvable), LitQA2 (no text), EHRNoteQA/CliniQ (credentialed); NovelQA (>200k, gated
answers), ∞Bench (190k), Loong/LooGLE (GPT-4 judge), DocFinQA/FinanceBench (numeric, NC license),
CRAG (NC), RULER/BABILong/NoLiMa (needle), SQuAD/DROP/TriviaQA (extraction-saturated or numeric).

---

## 4. Baselines

| Baseline | Role | Code | Effort |
|---|---|---|---|
| L-text / S-text / closed-book | upper/lower bounds; prior strength | ours | 0 |
| Prefix-cached L-text | the *fair* efficiency baseline (not re-prefill) | ours | 0.5 d |
| **LongLLMLingua** (Qwen3.5-2B compressor → 27B) | "small model pre-processes, large reads text" (H1/H2) | `pip install llmlingua`, MIT | 1 |
| **Heo ridge KV map** (2B→27B, full-attn layers only) | per-token cross-model KV transfer (group a); cite RaReCache | no code → ~300-line reimplementation | 2 |
| **C2C fuser** (2B sharer → 27B receiver) | latent cache communication (group b); cite LCF/XKV | local clone pinned to transformers 4.52 → reimplement in our stack | 3 |
| **SoftSkill prefix, L/L** | same-model compressed context (group d); cite Cartridges-at-Scale, LCC | local, defaults to Qwen3.5-4B, plain `inputs_embeds` | 1 |
| **Cartridge-style trained KV prefix on L** | H2 comparator | implement on the 16 full-attn layers | 3 |
| **CoRect** (reimplemented) | conflict intervention on L (H3/H4; group f) | no code; two-pass logit contrast | 3 |
| Reduced CEPE (cross-attn from S states into L full-attn layers) | architectural sibling, if time | ours | 4 |
| LCLM 0.6B/4B checkpoints | at-scale trained compressor reference (appendix) | released, Qwen3-4B decoder | 2 |
| Cite-only | RaReCache, DroidSpeak, CacheBlend, LCF, XKV, REFRAG, CALM, xRAG, CMC, Gist, AutoCompressor, ICAE, SpARE (no Qwen SAEs), PH3, JuICE, SPARK, LRT, Oozeer | — | — |

Note for the paper: on Qwen3.5 a per-token KV-transfer baseline can only touch the 8/16
full-attention layers (DeltaNet layers hold recurrent state, not KV) — itself a point in favour of a
residual-stream operator.

---

## 5. Metrics and controls (every table)

- **Content margin** = kv − **wrong-document** (same format, same length) — the number that decides
  P0. Wrong-*task* controls are not enough for documents.
- **Retained gain** = (S/L − base)/(L/L − base); **identity** = kv − wrong; **null** = interface on,
  no operator.
- **Override rate** on conflict sets (follows document vs follows prior), reported with clean-set
  accuracy as a Pareto pair.
- **mean generated tokens** per condition (collapse detector; procedural tasks ≥95% of text score).
- Entity **linear probe** on the read vector r for the document's entity (where the noise enters).
- **Rank–content curve**: SVD-truncate the registered operator; procedural skills should stay flat at
  rank 1, document operators should degrade — the figure that separates "steering" from "memory".
- Efficiency: encoder prefill vs reader prefill FLOPs, operator bytes (2.6 MB/doc at d_S=2560, 512
  slots, fp16 tied) vs KV bytes, break-even queries per document vs prefix caching.
- 3 seeds; n ≥ 300 items; never quote an in-flight cell (documented partial-eval hazard).

---

## 6. Experiment program and go/no-go gates

**P0 — content gate, same model (L/L, Qwen3.5-9B then 27B), ~2 weeks.**
Chunk each QuALITY / LongHealth / LastingBench document into ≤512-token operators from one whole-doc
prefill (no pooling); library + per-query softmax routing over chunk summary keys; one interface
registered over many documents with self-generated + gold QA; evaluate on **held-out documents**.
Sweep slots N ∈ {128, 256, 512}, rank ∈ {1, 8, 64, full}, chunks per doc.
- **G1** content margin ≥ +10 F1 (or ≥ +10 acc) and ≥ 50% of the text-in-context gain on ≥2 datasets.
- **G4** no direct-answer collapse (LiveMath/SearchQA ≥ 95% of text under trace-KL).
- Outcomes: (i) margin positive at full rank, collapses at rank 1 → proceed; (ii) raw HHᵀ fails but
  an operator *optimized on the small model* (independent U, W via self-study on S, then registered
  into L) succeeds → proceed with that construction ("encoding compute stays on S" still holds);
  (iii) margin ≈ 0 everywhere → pivot the follow-up to cross-model *procedural* skills.

**P1 — cross-model 2×2 (S ∈ {2B, 4B}, L ∈ {9B, 27B}), ~2 weeks.** Cells S/S, S/L, L/S, L/L with
closed-book, null, wrong-doc, early-layer L/L (harvest the reader's own ρ=0.25 states — the
"S beats L" vs "early beats late" confound), matched interface size and registration budget. Then
the encoder sweep 0.8B→2B→4B→9B at fixed 27B reader.
- **G2** S/L retains ≥ 70% of the L/L gain. **G3** gain non-decreasing 9B→27B. **G5** seed std < 2.

**P2 — smart layer injection, ~1.5 weeks.** Candidate set of 8 reader layers (ρ = 0.2…0.9), sparse
top-k query-dependent gate at **matched total update norm**, vs the fixed four depths, hand-tuned
mid+late (the design doc's two-site version is the baseline, not the method), searched static
top-k, random matched-k. Counterfactual layer swaps/deletions for the mechanism claim.

**P3 — conflict and noise (H3/H4), ~2 weeks.** Faithfulness-QA-articles, LastingBench-CF, ConFiQA
MR/MC, DistractionIF-style noise; KL dropped or gold-only on conflict data; multi-hop conflict set
to expose the late-injection limit.
- **G7** S/L > L/L by ≥ 3 points on at least one substitution-in-real-text conflict set at 27B.

**P4 — biomedical GLKB-QA pool, in parallel from week 2** (data build is CPU/GPU-light).

**P5 — Stage-B readiness.** G6: hooks under transformers TP=2 reproduce single-GPU results bitwise;
FP8-vs-bf16 read KL measured on a 27B FP8 repo.

---

## 7. Code plan (summary of `codeplan/PLAN.md`; ≈12 engineer-days)

New package `KV_skill_v2/xenc/` that **imports** `deltaskill` and `xskill` the way `cross_model`
imports `deltaskill` — never patches them.
1. `depth_skill.py` — `derive_by_depth(encoder, …)` keys K/V by normalized depth ρ instead of the
   encoder's absolute layer index; `for_reader(skill, reader)` maps ρ → reader layers (2B L=24 →
   {6,12,18,23}; 27B L=64 → {16,32,48,63}). Tied K=V, fp16, cached per (encoder, chunk) on /scratch
   (8 MB/chunk at 4ρ×512×2048).
2. `XEncInjector` — guards for `d_s ≠ d_m`, auto-maps ρ-keyed skills; `ModelInterface(d_reader,
   d_encoder)` already works (every current call site passes `(d_m, d_m)`).
3. `Library` / `LibraryInjector` — stack of chunk operators + summary key per chunk; softmax (or
   top-k) routing in `_branch`; auxiliary "which chunk holds the gold span" loss.
4. `LayerSelector` — sparse top-k gate over 8 candidate layers computed at the first candidate layer,
   update rescaled to a learned target norm α so selection is "where", not "how much".
5. `doc_tasks.py` + `register_doc.py` — port of `longhealth_kv.py` (the only existing
   document-as-skill loop) onto `Backbone` + `LibraryInjector`; teacher = reader with document in
   context, student = reader with question only + library; **trace-KL** target = teacher's own greedy
   trace (cached offline), CE on the answer span.
6. Memory hygiene: `logits_to_keep`, manual CE, `--max_mem`, CPU LRU of libraries (~50 docs).
7. Tests (`tests/test_xenc.py`): no-op at init with cross dims; dim-mismatch raises; ρ mapping incl.
   collision; routing sums to one and 1-chunk == old path; selector top-k with matched norm;
   encoder==reader equivalence to the old path at 1e-5.
8. Smoke: 2B→4B on one L40S, then 4B→27B on one H100, logging `max_memory_allocated`.

Riskiest unknowns: (1) the content bet itself (P0); (2) whether a 2B residual at ρ is readable by a
27B interface at the same ρ (normalized depth is a convention; the DeltaNet/attention interleave
differs by size — one reason to also try full-attention-aligned depths); (3) routing gets only
answer-level gradient; (4) trace-KL may still collapse through the gate.

---

## 8. Stage B: 700B-class reader on 50×B200

**Readers.** Qwen3.5-397B-A17B (in-family point; bf16 807 GB → 8 GPUs) → **DeepSeek-V3.1** (685B,
d=7168, 61 layers, MLA, MIT, native `deepseek_v3`, `thinking=False`) → **Kimi-K2-Instruct-0905**
(1.03T, same model class, FP8, non-thinking; the ≥700B headline with zero new code). Alternatives:
GLM-5 (754B, bf16 1.5 TB). **Avoid:** DeepSeek-V4 (four parallel residual streams — `h' = h + δ` is
ill-defined), V3.2 (no Jinja template, no flash_mla in HF), R1 / K2-Thinking / K3 (thinking cannot
be switched off; forced traces fake a ~0 floor, already seen in delta-skill).

**Fit.** FP8 reader + teacher + student fits 8×B200 for inference; **transformers' FP8 linears are
inference-only** (no autograd), so registration must either dequantize to bf16 (16 GPUs, TP=8 ×
FSDP/PP 2) or add a custom backward (`grad_out @ dequant(W)`, 1–2 days). Student activations from
ρ=0.25 upward ≈ 0.66 MB/token; operator per doc 2.6 MB vs 140 MB MLA KV per 2k-token doc.

**Software.** Teacher logits offline with SGLang (top-64 logprobs on the answer span: 98 KB/item);
student fwd+bwd in HF with `DistributedConfig(tp_size=8)` + decoder-layer hooks (hooks see the
full-width replicated residual on every rank); generation with hooks in HF for oracle eval, port to
vllm-lens generic hooks only if volume demands it (gate on KL agreement over 50 prompts; note
vLLM/SGLang layers return `(hidden, residual)` with fused add-norm — δ must be added to `residual`).
Estimated 2–5 s/step → 10–25 min per registration at 685B; L/L control (hidden states of 10k docs
through 685B) ≈ 20–40 min on 8 GPUs.

**What only Stage B buys.** F1 reader-size curve 9B→27B→235B→397B→685B→1T at fixed 2B/4B encoder;
F2 override rate vs reader size on substitution conflicts (big readers have the strongest priors);
F3 cost amortization vs prefix-cached text at 685B; F4 finer depth sweep over 61 layers; F5 FP8 vs
bf16 read fidelity; F6 expert-routing flip rate vs ‖δ‖ (MoE routing is discontinuous in h).

**Gate to enter Stage B:** G1–G7 above pass on 9B and 27B with 3 seeds; checkpoints (~3 TB) staged;
week-1 bring-up = SGLang V3.1 FP8 server + HF bf16 TP=8×2 load + hook-gradient smoke test.

---

## 9. Risks (top 6)

1. **Content does not go through at chunk level either** → outcome (ii)/(iii) in P0 are pre-registered.
2. **"S beats L" is "early beats late"** → early-layer L/L control in every P1 table.
3. **Interface memorizes the task** → wrong-document + null controls at matched budgets.
4. **Registration cost dominates** (≈8.5 min per operator on 27B; 100 docs × 3 seeds ≈ 40–75 GPU-h)
   → one operator per document amortized over its questions; 9B for sweeps, 27B for headlines.
5. **H3 sign depends on conflict construction** → substitution-in-real-text headline, generated-text
   control, both reported.
6. **Thinking-mode confounds** → `enable_thinking=False` everywhere; gpt-oss scored through
   `final_channel()` only.

---

## 10. Week-1 actions

1. Create `KV_skill_v2/xenc/` + tests (items 1–2, 6–7 of §7); apply the five mandatory fixes (§2.1).
2. Materialize QuALITY (cached), LongHealth (local), LastingBench, Faithfulness-QA, ConFiQA into
   `KV_skill_v2/data/<name>_split/{train,val,test}/items.json` with `doc_id`, `q_prompt`, `ctx_prompt`.
3. Measure the local Qwen3.5 2B/4B/9B/27B ladder on QuALITY and LongHealth (closed-book, text,
   single-hop extraction) — none is published; this fixes criterion C3 for our own backbones.
4. Encode all Tier-1 documents with 2B and 4B (offline, minutes); start the GLKB-QA PMC-OA fetch.
5. P0 smoke: L/L on 9B, QuALITY, 20 docs, 1 seed, report content margin and `max_memory_allocated`.
