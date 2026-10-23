# Round 4 brief (2026-10-19) — consolidate and extend the cross-model encoder-side read

Read first: `docs/DIRECTIONS.md` (theory constraints, GPU and code protocol, judge — the judge now ALSO
requires do-no-harm: kv ≥ closed_tail when the reader prefills a document tail), `outputs/D10_SUMMARY.md`
(the passing configuration and every failed variant), `outputs/D7_SUMMARY.md`, `outputs/D8_SUMMARY.md`,
`docs/report.html` (current results page). Env, tests, launchers as in DIRECTIONS.md.

## Where round 3 left things
- **Passing read (training-free, closed-form):** the small encoder prefills the document once, then processes the
  question over its own cache (addressing happens in the operator's native space); its per-token attention output
  (post o_proj) at depth-matched layers is mapped by a ridge map B (fitted on token-aligned prefill pairs, no labels)
  and REPLACES the reader's attention output at early full-attention layers; the reader itself prefills only the
  last 64 document tokens + the question. Code: `xenc/encside_read.py`, `scripts/eval_encside.py`,
  `scripts/fit_encside_B.py`; maps in `outputs/D10_maps/`.
- 9B reader: best robust layer set {3,7,11,15,19}; 4B→9B = 0.81 / 0.77 (LongHealth fold 0 / QuALITY) vs text 0.825 / 0.81.
- 27B reader: needs its full depth<0.5 band (8 layers); 4B/9B→27B cap at 0.60–0.69 vs anchor 0.775 (8 layers) and
  0.8625 (all 16 layers, = text). Ridge-λ refit and add mode do not help (linear token-wise B at its ceiling).
- Every read with the READER's own query against the foreign operator (the paper's read as written) failed.
- Reward-learned compact operator (paper's 2nd construction, D4): reproducible ≈ +0.08 on unseen questions.

## Hard rules (unchanged + additions)
- ≤ 5 jobs in `squeue -u rickyhan` across all agents; ≤ 1 job per direction unless slots are idle; check before
  every sbatch (`slurm/submit_when_free.sh`). Tag every job/run/output with your direction prefix; log ids in
  `slurm/logs/<tag>.jobids`. 27B jobs need one H100 (`--partition=drjieliu-h100`).
- Never overwrite another agent's file: `test -e` before every Write; Edit existing shared files minimally.
- `bash scripts/run_tests.sh` green before you hand back. Never modify `../delta-skill` or `docs/report*.html`.
- Judge every condition with kv, wrong_doc, closed_tail (and closed_book, text where cheap), mean gen tokens;
  report kv − wrong_doc ± se with n AND kv − closed_tail. A row with kv < closed_tail is a FAIL.
- Final report ≤ 800 words + `outputs/<tag>_SUMMARY.md` with every table.
