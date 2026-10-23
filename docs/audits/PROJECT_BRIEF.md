# Project brief for dataset scouting (KV-Skill v2: "Small Encoder, Large Reader")

Published base paper: KV-Skill (arXiv 2608.05475v2). A frozen LM reads an external factorized
operator M = W Uᵀ through a small interface at 4 injection layers:
q = norm(A·RMSNorm(h)), r = W(Uᵀq)/√m, h' = h + γ·g·B·r. Text-derived operators are the L2-normalized
hidden states of a one-time prefill of a document (U = W = H̄, one slot per token, ≤512 slots).
"Registration" = 300 gradient steps on the interface only (KL to a text-in-context teacher + 0.5·CE on
gold answer + magnitude penalty). Known: on procedural skills this works (LiveMath 77.2 vs 52.0 text),
but rank-1 suffices (it's a task direction), and on DOCUMENT QA (LongHealth, QuALITY) the operator
carried ~no content (kv ≈ wrong-doc ≈ base ≪ text-in-context). Registration also pushes the model
into a 6-token direct-answer mode, which destroys tasks needing computation.

NEW PROJECT: a SMALL frozen model (Qwen3.5-2B/4B) prefills each DOCUMENT once into an operator; a LARGE
frozen reader (Qwen3.5-9B/27B, gpt-oss-20b) reads it through a registered cross-model interface
(A: d_L→d_S, B: d_S→d_L) at learned, query-dependent injection layers. Documents never enter the large
model's context. Hypotheses: H1 S/L beats large model reading raw text (for procedural content);
H2 S/L ≈ L/L at lower encoding cost; H3 S/L > L/L on prior-conflicting / instruction-noise content
(Size–Fidelity Paradox); H4 mid+late / dynamic layer injection overturns prior override.

WHAT MAKES A DATASET GOOD FOR US (rank candidates against these):
1. Documents are REUSED across many queries (RAG over a fixed pool; many Qs per doc). One operator per
   doc/chunk must amortize. Datasets with one unique short context per question are weak.
2. Context long enough to matter: per-item context ideally 1k–30k tokens (chunkable into ≤512-token
   operators), NOT 100–300-token snippets. Need a length sweep from below d_b (~2.5k) to well above.
3. A LARGE large-minus-small gap with the documents IN context (reasoning/selection is hard), while the
   small model's single-hop extraction on the same documents is high (understanding is easy).
4. Prior-conflict variants exist or can be built (counterfactual / swapped entities / recent findings).
5. Answers are free-form short spans or MCQ scored by F1/EM/accuracy; avoid exact-number answers and
   pure needle retrieval; avoid tasks the large model solves closed-book.
6. Multi-hop composition across chunks is a plus (tests routing).
7. Must be downloadable from HF Hub or GitHub with a clear license; cluster compute nodes are OFFLINE
   (HF_HUB_OFFLINE=1); data is materialized once on the login node into data/<name>_split/{train,val,test}/items.json.
8. Ideally has published numbers for Qwen3/Qwen3.5-class models at several sizes so we can sanity check.

Local models available: Qwen3.5-2B/4B/9B/27B, Qwen3-4B-Instruct-2507, gpt-oss-20b, gemma-4-12b-it,
Qwen2.5-0.5B/3B/14B. GPUs: lab L40S (48 GB) node ×8 and H100 node ×8.
