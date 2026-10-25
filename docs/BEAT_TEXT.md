# BEAT_TEXT: main goal from 2026-10-20

The user's goal: the idea is validated. The focus now is performance: **kv must beat text on the benchmarks**.
The reader still never sees the document text at test time.

## Where we are (9B reader, 4B encoder; registered, outputs/D22_SUMMARY.md)

| benchmark | text | kv_reg | kv−text |
|---|---|---|---|
| LongHealth | .812 | .812–.825 | ±0 |
| QuALITY | .828 | .756–.763 | −.07 |
| ContractNLI | .757 | .650–.697 | −.06 to −.11 |
| Shakespeare-CF | .940 | .812 | −.13 |
| CLIPPER | .913 | .778 | −.14 |
| Qasper | .634 | .432 | −.20 |
| MuSiQue | .539 | .359 | −.18 |
| LastingBench | .480 | .343 | −.14 |
| LooGLE | .643 | .355 | −.29 |

## Step 0: find the ceiling before spending GPUs (diagnostic, cheap)
For each benchmark, work out where the gap comes from:
- **C1 encoder information**: the 4B model's own text accuracy (4B reads the doc as text).
- **C2 read mechanism**: 9B encoder → 9B reader, the same model on both sides, so no information is lost in the encoder.
  If C2 ≈ text, the 4B representation is the limit. If C2 ≪ text, the read mechanism is the limit, so fix the read.
- **C3 answer format** (CPU only, existing outputs): for the span tasks, is the F1 gap due to wrong content, or to answer length/format?
  Check exact-substring rate, length ratio vs gold, and the "no answer" rate.

## Levers, ordered by expected gain
1. **Train the base on every benchmark's TRAIN split.** The base has seen only QuALITY, MuSiQue, LastingBench and LongHealth f2–4.
   Qasper, LooGLE, ContractNLI, CLIPPER and Shakespeare have no training docs in the base, even though their gaps are the largest.
   This is in-distribution training on held-out train docs, not OOD; test docs stay held out with a leakage assertion.
   It includes long operators (32k–128k) to fix the collapse past 64k.
2. **Read capacity.** More read layers (deep 23/27/31), more operator columns (no tail cap), an MLP B, a higher delta rank.
   Pick by C2.
3. **Registration that can exceed its teacher.** CE on encoder-written self-QA with checked answers, span-style self-questions
   for span tasks, more steps. KL-to-text alone caps registration at text level.
4. **RL (GRPO etc.)**: first choice is still the supervised levers above, but the user (2026-10-20) allows RL as a
   fallback whenever a benchmark has plateaued under supervised/distillation training and no other lever helps,
   not only at the .02–.03 polish stage. Reward = the benchmark judge on encoder-written self-QA or train-split items, never test items.
5. **Smart injection** where text is weak: conflict sets, very long docs. Already kv > text on FaithEval and ConflictQA.

Run-hygiene rules (docs/PAPERFORM.md) apply to everything: pilot ≤500 steps, val must beat step 0, exclude step 0 from selection.

## MAIN STORY ADDITION (user, 2026-10-20): the hypernetwork path
The user is bullish on the hypernetwork H: the 4B encoder prefills once, ONE forward of H gives the per-doc adapter, and the large reader (esp. 27B) never reads the doc, not even for registration. This preserves the full cost advantage.
- 9B: D24 agent retrains H on CLIPPER / Shakespeare-CF / Qasper / LooGLE, where per-doc registration helps (+.05 to +.16). The old D24 trained where registration gives ≈0.
- 27B: D12 agent measures the registration headroom on the 27B 4B-enc base, then trains a 27B H.
- Cost: D23 agent measures H cost vs harvest / prefill / registration, plus the end-to-end table (outputs/D23_HYPER_COST.md).
Success = kv_H − base > 2 SE and kv_H − Hswap > 2 SE (document-specific), approaching kv_reg at zero reader passes.

## KV-RAG (user idea + name, 2026-10-20), agent H2
For super-long documents: chunk the doc, keep a 4B operator per chunk, select top-k chunks (latent router / BM25 / dense / hybrid), then paper-form read over the selected chunks. This avoids flat-read dilution past 64k. Compared on the same items against text (full or truncated), text-RAG (same retriever) and flat kv, with chunk recall reported. Results go to outputs/H2_SUMMARY.md.

## READ-FORM FREEDOM (user, 2026-10-21)
Exploring different MATH FORMS of the read is approved. Invariants: small encoder → large reader; smart layer injection (same reader layers + write-back B); operator (U/W from the encoder prefill) essentially unchanged. Everything between the reader's h and the write-back is free.
- L1: long-document forms (log-N scaling, top-k sparse, hierarchical).
- R2: query formation / learned sharpness / multi-hop re-query / multi-query heads / gated read. Job names D00R2_*.

## DATA SCALE-UP (user, 2026-10-22: "那就继续扩大数据量"), agent DS
The C2 span diagnostic shows the span gap is mostly training-data and format coverage; translation costs ~.04. Lever-1 used only a fraction of some train splits (MuSiQue 148/967, LastingBench 148/600, LooGLE 1395/3882, CLIPPER 5888/9198).
- Phase A: use all in-distribution train data.
- Phase B: new in-family data (raw MuSiQue, NarrativeQA, CUAD, ...), with strict leakage checks.
- Exploration mode (2026-10-22): no dedicated control; full-length run (4000 steps) + registered judge vs lever-1 full rows. Job names D00DS_*.

## COMPREHENSION FOCUS (user, 2026-10-22: "应该多搞这种阅读理解的选择题而不是单纯背书")
On lever-1 full (9B, unregistered), kv matches or beats text on comprehension and judgment formats: LongHealth .8625 vs .8125, SCF .933 vs .940, QuALITY .791 vs .828, CNLI .720 vs .753. It trails on verbatim span extraction (−.07 to −.14).
- Weight evaluation and training toward reading-comprehension MCQ / judgment benchmarks.
- Agent MCQ: new comprehension sets (NovelQA, DetectiveQA, L-Eval, LongBench-v2, ∞Bench En.MC, ...) plus comprehension-heavy training. Job names D00MCQ_*.
- Span tasks remain a secondary target (P6 / CAP / ADDR).

## ENCODER-WRITTEN NOTES (user, 2026-10-23: "encoder是一个LLM而不仅仅是一个encoder…digest/reasoning/comprehension…应该多探索"), agent NOTES
The encoder generates question-independent notes after prefilling the doc. The notes' encoder K/V are appended as extra operator columns; the reader never sees text. Explore note types (digest, key facts, reasoning, self-QA) and lengths. Probe training-free first, then train the read with note columns. Covers 9B and 27B. Jobs D00NT_ / D12b_nt (rank 2). H-slots v2 (A) waits.

## COMPRESSION DESIGN SPACE (user, 2026-10-23), agent CMP
HM (colleague's Hub Memory, /nfs/turbo/umms-drjieliu/usr/xiangzhg/kvshill/universal/, read-only) stores a 128-dim hub vector per token. It is strong on extractive and multi-hop sets, weak on comprehension, and ~80× smaller.
Our full per-layer operator is strong on comprehension but possibly noisy for verbatim and multi-hop.
CMP sweeps a per-token bottleneck d ∈ {128 … full} (plus layers and quantisation) inside our framework. It reports category scores vs MB/doc, looking for the sweet point where compression filters noise without losing comprehension. Job prefix D00CMP_, rank 2.
