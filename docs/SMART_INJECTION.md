# Smart layer injection: proposal (2026-10-20). ON HOLD until per-document registration works (user).

Goal: decide WHICH reader layers receive the operator read-out, and how strongly, per question / per document.
This matters most where the reader's own prior competes with the document (knowledge conflict, larger readers).
User hypothesis: deeper layers of larger models trust their prior; the small encoder's prefill is "cleaner".

## Step 1: find each layer's direction (diagnostics; no training, cheap)
D1 Per-layer causal sweep. Inject at one layer only, leave one layer out, or grow greedily. Measure delta(kv - wrong),
   delta log p(gold), and doc-adherence on conflict sets. Output: a utility score per layer.
D2 Logit / tuned lens. Read each layer's residual stream out as answer probabilities.
   (a) In text mode on conflict items, find the layer where the document answer overtakes the prior answer
       ("commit depth").
   (b) Closed book, find where the prior answer first appears ("recall depth").
   Inject at or before the recall depth.
D3 Context-attention profile. In text mode, measure how much attention each layer puts on document tokens versus
   question / self tokens. These are the layers that actually read context. Compare 9B vs 27B; the hypothesis
   predicts 27B's deep layers attend less to context.
D4 Encoder-reader alignment. Per-layer ridge R^2 / CKA between encoder and reader states (D10 already has B R^2).
   Layers with low alignment are poor injection sites unless B is non-linear.
D5 Gradient sensitivity. Take d KL(text || kv) / d gate_L over held-out training docs: where the parity loss wants
   more or less injection.
D6 Conflict direction. In text mode, per layer, compute mean h(CF doc) - h(original doc) on conflict items.
   - Its norm and linear separability show where conflict is resolved.
   - The vector itself is a "trust the document" steering direction.

## Step 2: choose and inject (methods)
M1 Static top-k. Greedy forward selection of layers by D1 utility, under a layer budget, chosen on validation.
M2 Sparse learned gates. Per-layer gates with an L0 / hard-concrete penalty, trained in the base (stage 2), so
   training picks the layers itself. Cheap; fits the current code (gates exist).
M3 Registration-chosen layers. Per-document registration also fits per-layer gates (a few params per doc), so
   each document picks its own depth profile. Combines directly with D20.
M4 Query-conditioned router. A small head on the question's early hidden state outputs per-layer gates
   (mixture-of-depths). Trained on training docs; learns when deep injection is needed.
M5 Conflict-triggered deep injection. Score the conflict as the divergence between the reader's own mid-layer
   prediction (lens) and the read-out prediction. When it is high, raise the gates at layers beyond the commit
   depth (from D2) — "override the prior where it forms".
M6 Steering add-on. At the commit layers, add the D6 "trust the document" direction scaled by the conflict
   score, on top of the read-out.

## Evaluation
- Conflict sets: Shakespeare-CF (D19), NQ-Swap, ConflictQA, FaithEval. Report doc-adherence, prior-adherence,
  and accuracy.
- Do-no-harm on standard tasks: LongHealth, QuALITY, ContractNLI must not drop.
- Compare: fixed layer set (current) vs all layers vs smart.
- Run on 9B and 27B readers. All kv after per-document registration (no unregistered reads).

## Suggested order
D1 + D2 + D3 on 9B and 27B, then M2 + M3 (cheap, merges with registration), then M4 / M5 only if conflict
items still lag text.
