# -*- coding: utf-8 -*-
"""Render outputs/RESULTS.json (from aggregate_results.py) into docs/report.html — the P0 conclusion page.

    python scripts/aggregate_results.py && python scripts/make_report.py
"""
from __future__ import annotations
import json, sys, html
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "outputs" / "RESULTS.json"))
RUNS, LADDERS, PROBES = R["runs"], R["ladders"], R.get("probes", {})

LABELS = {
    "p0_quality_9b_LL/test": ("QuALITY", "L/L (9B reads its own operators)", "held-out docs"),
    "p0_quality_9b_2b/test": ("QuALITY", "S/L (9B reads 2B operators)", "held-out docs"),
    "p0_longhealth_9b_LL_f0/test": ("LongHealth", "L/L", "held-out fold 0"),
    "p0_quality_9b_LL_withindoc/train": ("QuALITY", "L/L", "same docs, unseen questions"),
    "p1_quality_9b_LL_center/test": ("QuALITY", "L/L, centered", "held-out docs"),
    "p1_quality_9b_2b_center/test": ("QuALITY", "S/L, centered", "held-out docs"),
    "p1_longhealth_9b_LL_f0_center/test": ("LongHealth", "L/L, centered", "held-out fold 0"),
    "p1_longhealth_9b_2b_f0_center/test": ("LongHealth", "S/L, centered", "held-out fold 0"),
    "p1_quality_9b_LL_center_1doc/train": ("QuALITY", "L/L, centered, 1 doc", "same doc, unseen questions"),
    "p1_quality_9b_LL_center_5doc/train": ("QuALITY", "L/L, centered, 5 docs, 600 steps", "same docs, unseen questions"),
}
ORDER = list(LABELS)


def sc(row, c):
    v = row.get(c)
    return None if not v or v.get("score") is None else v["score"]


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def gap_cls(g):
    if g is None:
        return ""
    return "pos" if g >= 0.10 else ("neg" if g <= -0.05 else "zero")


rows_html = []
for k in ORDER:
    if k not in RUNS:
        continue
    r = RUNS[k]; ds, cond, split = LABELS[k]
    n = (r.get("kv") or {}).get("n", "")
    g = r.get("kv_minus_wrong")
    rows_html.append(
        f"<tr><td>{ds}</td><td>{cond}</td><td>{split}</td><td class='num'>{n}</td>"
        f"<td class='num'>{fmt(sc(r,'closed_book'))}</td><td class='num'>{fmt(sc(r,'text'))}</td>"
        f"<td class='num em'>{fmt(sc(r,'kv'))}</td><td class='num'>{fmt(sc(r,'wrong_doc'))}</td>"
        f"<td class='num'>{fmt(sc(r,'null'))}</td><td class='num'>{fmt(sc(r,'oracle_chunk'))}</td>"
        f"<td class='num gap {gap_cls(g)}'>{'—' if g is None else f'{g:+.3f}'}</td></tr>")


# ROUND2_SECTION present
# ---------- direction runs (D1..D6, any key not in LABELS)
DIRECTION_ROWS = []
for k, r in RUNS.items():
    if k in LABELS or k.startswith(("p0_", "p1_")):
        continue
    run, split = k.split("/")
    n = (r.get("kv") or {}).get("n", "")
    g = r.get("kv_minus_wrong")
    enc = "same" if r.get("encoder") in (None, "same") else str(r.get("encoder")).split("/")[-1]
    DIRECTION_ROWS.append(
        f"<tr><td><code>{html.escape(run)}</code></td><td>{html.escape(split)}</td><td>{enc}</td><td class='num'>{r.get('steps') or ''}</td><td class='num'>{n}</td>"
        f"<td class='num'>{fmt(sc(r,'closed_book'))}</td><td class='num'>{fmt(sc(r,'text'))}</td>"
        f"<td class='num em'>{fmt(sc(r,'kv'))}</td><td class='num'>{fmt(sc(r,'wrong_doc'))}</td>"
        f"<td class='num'>{fmt(sc(r,'null'))}</td><td class='num gap {gap_cls(g)}'>{'—' if g is None else f'{g:+.3f}'}</td></tr>")
DIRECTION_TABLE = ("<div class='tbl'><table><thead><tr><th>run</th><th>eval split</th><th>encoder</th><th>steps</th><th>n</th><th>closed book</th><th>text</th><th>kv</th><th>wrong doc</th><th>null</th><th>kv − wrong</th></tr></thead><tbody>" + "".join(DIRECTION_ROWS) + "</tbody></table></div>") if DIRECTION_ROWS else ""

# ---------- ladder chart (inline SVG, to scale)
def ladder_svg(task: str, data: dict) -> str:
    models = ["Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B"]
    pts = [(m, data[m]["closed_book"]["score"], data[m]["text"]["score"]) for m in models if m in data]
    W, H, L, B, T, Rm = 520, 240, 56, 36, 18, 16
    x = lambda i: L + i * (W - L - Rm) / max(1, len(pts) - 1)
    y = lambda v: T + (1 - v) * (H - T - B)
    g = []
    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        g.append(f"<line x1='{L}' y1='{y(v):.1f}' x2='{W-Rm}' y2='{y(v):.1f}' class='grid'/>"
                 f"<text x='{L-8}' y='{y(v)+4:.1f}' class='tick' text-anchor='end'>{v:.1f}</text>")
    for key, cls in (("cb", "cb"), ("tx", "tx")):
        idx = 1 if key == "cb" else 2
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(p[idx]):.1f}" for i, p in enumerate(pts))
        g.append(f"<path d='{path}' class='line {cls}'/>")
        for i, p in enumerate(pts):
            g.append(f"<circle cx='{x(i):.1f}' cy='{y(p[idx]):.1f}' r='4' class='dot {cls}'/>"
                     f"<text x='{x(i):.1f}' y='{y(p[idx]) - 9:.1f}' class='val' text-anchor='middle'>{p[idx]:.2f}</text>")
    for i, p in enumerate(pts):
        g.append(f"<text x='{x(i):.1f}' y='{H-B+18}' class='tick' text-anchor='middle'>{p[0].replace('Qwen3.5-','')}</text>")
    legend = (f"<g transform='translate({L},{H-6})'><line x1='0' y1='0' x2='18' y2='0' class='line tx'/><text x='24' y='4' class='tick'>text in context</text>"
              f"<line x1='140' y1='0' x2='158' y2='0' class='line cb'/><text x='164' y='4' class='tick'>closed book</text></g>")
    return (f"<figure><svg viewBox='0 0 {W} {H}' role='img' aria-label='{task} accuracy by reader size'>"
            f"{''.join(g)}{legend}</svg><figcaption>{task}: accuracy with the document in context vs closed book, "
            f"n=200 test items, answer-only prompt, 256-token budget.</figcaption></figure>")


lad = {k.split("/")[1]: v for k, v in LADDERS.items() if k.startswith("ladder_v2/")}
charts = "".join(ladder_svg(t.capitalize() if t != "longhealth" else "LongHealth", lad[t]) for t in ("quality", "longhealth") if t in lad)

# ---------- probe
probe_html = ""
for run, summ in PROBES.items():
    rows = []
    for k, v in summ.items():
        if not isinstance(v, dict) or "own_entropy" not in v or "own_route_top1" not in v:
            continue   # probe jsons from other scripts (D7 native probes) use a different schema
        rows.append(f"<tr><td>{html.escape(k)}</td><td class='num'>{v['n']}</td><td class='num'>{v['own_entropy']:.2f} / {v['own_entropy_max']:.2f}</td>"
                    f"<td class='num'>{v['own_top1']:.3f}</td><td class='num'>{v['own_route_top1']:.3f}</td>"
                    f"<td class='num'>{fmt(v['gold_mass'], 4)}</td><td class='num'>{fmt(v['gold_chance'], 4)}</td></tr>")
    probe_html += (f"<h3>Read-attention probe on <code>{html.escape(run)}</code></h3>"
                   "<div class='tbl'><table><thead><tr><th>interface / layer</th><th>n</th><th>entropy / max</th><th>top-1 slot</th><th>top-1 chunk</th><th>mass on gold slots</th><th>chance</th></tr></thead>"
                   f"<tbody>{''.join(rows)}</tbody></table></div>")

q = RUNS.get("p0_quality_9b_LL/test", {}); lh = RUNS.get("p0_longhealth_9b_LL_f0/test", {}); sl = RUNS.get("p0_quality_9b_2b/test", {})
qc = RUNS.get("p1_quality_9b_LL_center/test", {}); lhc = RUNS.get("p1_longhealth_9b_LL_f0_center/test", {})


# ---------- round 3: native-state ceilings (D7, training-free prefix cache)
STATE_CONDS = [("closed_book","closed book"),("text","text"),("all","KV+state+conv"),("kv_only","KV only"),("wrong_kv","wrong-doc KV"),("state_only","state only"),("wrong_state","wrong-doc state")]
STATE_ROWS = []
for run, row in sorted((R.get("state_ceiling") or {}).items()):
    g = (row.get("kv_only-wrong_kv") or {}).get("gap"); gs = (row.get("state_only-wrong_state") or {}).get("gap")
    task = "LongHealth fold 0" if "lh" in run else ("QuALITY test" if "_q" in run else run)
    cells = "".join(f"<td>{fmt(sc(row,c))}</td>" for c,_ in STATE_CONDS)
    STATE_ROWS.append(f"<tr><td><code>{html.escape(run)}</code></td><td>{task}</td><td>{row.get('n')}</td>{cells}"
                      f"<td class='{gap_cls(g)}'>{fmt(g,3) if g is None else f'{g:+.3f}'}</td><td class='{gap_cls(gs)}'>{fmt(gs,3) if gs is None else f'{gs:+.3f}'}</td><td>{fmt(row.get('agree_all_text'),2)}</td></tr>")
STATE_TABLE = ("<div class='tbl'><table><thead><tr><th>run</th><th>task</th><th>n</th>" + "".join(f"<th>{l}</th>" for _,l in STATE_CONDS) + "<th>KV − wrong</th><th>state − wrong</th><th>all = text</th></tr></thead><tbody>" + "".join(STATE_ROWS) + "</tbody></table></div>") if STATE_ROWS else ""

COMP_LABELS = dict(all="all layers (KV + states)", e4s="KV at layers 3,7,11,15 + states", a25="25% columns by attention + states", r25="25% random columns + states",
                   a10="10% columns by attention + states", a5="5% columns by attention + states", kvo="KV only, all 8 layers", ka25="KV only, early 4, 25% columns", ka10="KV only, early 4, 10% columns", ka5="KV only, early 4, 5% columns")
COMP_HEAD = dict(h0="native operator", h1="acc", h2="wrong doc", h3="gap")

# ---------- round 3: D8 compression of the native operator (eval_prune.json)
COMP_ORDER = [("all","{all}"),("kvs_E_100","{e4s}"),("kvs_E_acc25","{a25}"),("kvs_E_rand25","{r25}"),("kvs_E_acc10","{a10}"),("kvs_E_acc5","{a5}"),
              ("kv_only","{kvo}"),("kv_E_acc25","{ka25}"),("kv_E_acc10","{ka10}"),("kv_E_acc5","{ka5}")]
COMP_ROWS = []
for run, comp in sorted((R.get("compression") or {}).items()):
    rows = comp.get("rows") or {}
    for c, label in COMP_ORDER:
        v = rows.get(c)
        if not v:
            continue
        mib = None if v.get("bytes") is None else v["bytes"] / 2**20
        g = v.get("gap")
        COMP_ROWS.append(f"<tr><td>{label.format(**COMP_LABELS)}</td><td>{'—' if mib is None else f'{mib:.0f}'}</td><td>{fmt(v.get('score'))}</td><td>{fmt(v.get('wrong'))}</td>"
                         f"<td class='{gap_cls(g)}'>{'—' if g is None else f'{g:+.3f}'}</td><td>{v.get('n')}</td></tr>")
COMP_TABLE = ("<div class='tbl'><table><thead><tr><th>{h0}</th><th>MiB</th><th>{h1}</th><th>{h2}</th><th>{h3}</th><th>n</th></tr></thead><tbody>".format(**COMP_HEAD) + "".join(COMP_ROWS) + "</tbody></table></div>") if COMP_ROWS else ""

page = f"""<title>KV-Skill v2 P0 Gate</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: a lab-notebook report — one reading column, data tables full-bleed inside it, charts drawn to scale */
:root {{
  --bg: #f7f6f2; --surface: #ffffff; --fg: #1c1f22; --muted: #5d646b; --rule: #d9d6cd;
  --accent: #0f6b5c; --accent-ink: #0a4a40; --neg: #9a3b2e; --zero: #8a8f94; --pos: #0f6b5c;
  --display: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #15181b; --surface: #1d2125; --fg: #e8e6df; --muted: #a2a8ad; --rule: #353b41;
  --accent: #5fc3ad; --accent-ink: #8fd9c8; --neg: #e08a7a; --zero: #8a8f94; --pos: #5fc3ad; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #15181b; --surface: #1d2125; --fg: #e8e6df; --muted: #a2a8ad; --rule: #353b41;
  --accent: #5fc3ad; --accent-ink: #8fd9c8; --neg: #e08a7a; --zero: #8a8f94; --pos: #5fc3ad; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--body); line-height: 1.55;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 32px 64px; }}
main {{ max-width: 76ch; margin: 0 auto; display: grid; gap: 28px; }}
h1 {{ font-family: var(--display); font-weight: 600; font-size: clamp(1.7rem, 4vw, 2.4rem); line-height: 1.15; margin: 0; text-wrap: balance; }}
h2 {{ font-family: var(--display); font-weight: 600; font-size: 1.35rem; margin: 8px 0 0; text-wrap: balance; }}
h3 {{ font-weight: 600; font-size: 1.02rem; margin: 6px 0 0; }}
p {{ margin: 0; max-width: 68ch; }}
.eyebrow {{ font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
.verdict {{ border-left: 4px solid var(--accent); padding: 10px 16px; background: var(--surface); display: grid; gap: 8px; }}
.verdict strong {{ color: var(--accent-ink); }}
.tbl {{ overflow-x: auto; border: 1px solid var(--rule); background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; min-width: 640px; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }}
th {{ font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); font-weight: 600; }}
td.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
td.em {{ font-weight: 600; }}
td.gap.pos {{ color: var(--pos); font-weight: 600; }} td.gap.neg {{ color: var(--neg); }} td.gap.zero {{ color: var(--zero); }}
figure {{ margin: 0; display: grid; gap: 6px; }} svg {{ width: 100%; max-width: 100%; height: auto; }}
figcaption {{ font-size: .82rem; color: var(--muted); }}
.grid {{ stroke: var(--rule); stroke-width: 1; }} .tick, .val {{ fill: var(--muted); font-family: var(--mono); font-size: 11px; }}
.val {{ fill: var(--fg); }} .line {{ fill: none; stroke-width: 2; }} .line.tx, .dot.tx {{ stroke: var(--accent); fill: var(--accent); }}
.line.cb, .dot.cb {{ stroke: var(--muted); fill: var(--muted); }} .line.cb {{ stroke-dasharray: 4 3; }} .dot.tx, .dot.cb {{ stroke: none; }}
.charts {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }} .charts > * {{ min-width: 0; }}
ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }} li {{ max-width: 68ch; }}
code {{ font-family: var(--mono); font-size: .88em; }}
.meta {{ font-size: .82rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 18px; }}
</style>
<main>
  <div>
    <div class="eyebrow">KV-Skill v2 · small encoder, large reader · P0 content gate</div>
    <h1>Reading document operators: what the interface must learn</h1>
    <div class="meta"><span>Reader Qwen3.5-9B</span><span>Encoders: 9B (L/L) and 2B (S/L)</span><span>300 registration steps, 20 train docs</span><span>{date.today().isoformat()}</span></div>
  </div>

  <section class="verdict">
    <strong>Status (round 3): the pre-registered gate is passed for the first time, training-free, on the same model: the reader's own full-attention K/V harvested from one prefill, read natively, beats the wrong document by +0.15 (LongHealth, n=80) and +0.125 (QuALITY, n=200), and with the DeltaNet states added it reproduces the in-context answer. Every learned interface over a softmax-slot read (rounds 1–2) failed because the read, not the operator, was wrong. And the cross-model read now has a first pass: a 2B encoder's operator, addressed by the 2B itself over the question and carried into the 9B by a closed-form map B, beats the wrong document by +0.21 on LongHealth (n=80) and +0.17 on QuALITY (n=300) while beating the closed-book controls on both, with no training at all; reading at all eight layers reaches the 9B's own in-context level on LongHealth (0.7875 vs 0.825 for the text) but not on QuALITY, so the early-layer read is the robust configuration. With a 4B encoder that configuration brings the 9B to 0.81 / 0.77 on LongHealth / QuALITY against 0.825 / 0.81 for reading the text itself. <em>And the paper's read as written now works:</em> the reader's own query, through an interface first distilled from the encoder's attention and then trained with the paper's KL + CE, reads the 2B's fixed operator at 0.7625 vs 0.425 on LongHealth (n=80) and 0.633 vs 0.453 on QuALITY (n=300), passing both criteria on both tasks with no encoder at inference. Compression of the native operator (D8) keeps document identity down to 25% of the early-layer columns.</strong>
    <p>On held-out documents the reader scores the same with the correct document's operator as with a wrong document's
      (QuALITY L/L {fmt(sc(q,'kv'))} vs {fmt(sc(q,'wrong_doc'))}, n={ (q.get('kv') or {}).get('n','')}; LongHealth L/L {fmt(sc(lh,'kv'))} vs {fmt(sc(lh,'wrong_doc'))}).
      Restricting the library to the gold chunk does not help, and the uncentered interface produced byte-identical generations for correct and wrong documents on 48 of 51 paired items.
      The branch output is not a function of the loaded document. This is a failure of the read, not of the operator: the operator is the encoder's per-token hidden states and is unchanged by registration, so the content is in it; the registered query map simply never learns to address slots (see the probe below). The same holds when the 2B encoder writes the operators and when the interface is tested on unseen questions from the very documents it was trained on.</p>
    <p>What does work: the pipeline itself (2B encoding of 265 QuALITY documents in 206 s; 9B registration at 2.2 s/step and 19 GiB; a document-independent +3 to +8 point shift from the interface), and the size ladders, which show a real text-in-context gap between 2B and 4B on both datasets.</p>
  </section>

  <section>
    <h2>Gate table</h2>
    <p>All conditions use greedy decoding, the answer-only prompt, a 256-token budget and the same items. <em>kv</em> loads the correct document's chunk library; <em>wrong doc</em> loads another document of the same task; <em>null</em> loads a random operator; <em>oracle</em> restricts the library to chunks containing the gold answer text (only items where that is defined). The gate quantity is kv − wrong doc; the pre-registered pass threshold was +0.10.</p>
    <div class="tbl"><table>
      <thead><tr><th>dataset</th><th>condition</th><th>eval items</th><th>n</th><th>closed book</th><th>text</th><th>kv</th><th>wrong doc</th><th>null</th><th>oracle</th><th>kv − wrong</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody></table></div>
  </section>

  <section>
    <h2>Why the read is document-blind</h2>
    <ul>
      <li>Unit-normalized residual slots share a dominant common direction: its norm is 0.59 (2B) and 0.85 (9B) at depth 0.25. Within-document and across-document slot similarities are the same (0.36 vs 0.35), so the read returns essentially the corpus mean whatever document is loaded.</li>
      <li>Centering the slots per depth removes that direction (similarities drop to about 0, top-1 read mass rises from 0.69–0.91 to 0.95–0.99), but the centered runs score the same as closed book: the lift disappears and no content appears (QuALITY L/L centered {fmt(sc(qc,'kv'))} vs wrong {fmt(sc(qc,'wrong_doc'))}; LongHealth {fmt(sc(lhc,'kv'))} vs {fmt(sc(lhc,'wrong_doc'))}).</li>
      <li>The read never learns to address slots. On LongHealth with the centered 9B interface, the attention over a document's ~11,000 slots has entropy 8.6–8.9 nats against a maximum of 9.3, the top slot carries 0.005 of the mass, and the mass on slots holding the gold answer is 0.0010–0.0014 against a chance level of 0.001; an untrained interface reads 9.2 nats. Registration moves the chunk-level routing a little (top chunk 0.19–0.26 vs 0.05 untrained) but the slot-level read stays a document-wide average.</li>
      <li>Injection magnitude is not the limiter: the relative update reaches 1.3–1.6% of the residual norm, in the range the published rank-1 procedural skills use (2.3%).</li>
      <li>This is the same outcome the earlier delta-skill LongHealth runs reached with pooled operators, now with wrong-document, null and oracle controls at chunk resolution and a cross-model encoder.</li>
    </ul>
    {probe_html}
    <h3>Content is present; addressing and write-back are the gaps (D6 diagnostics, LongHealth fold 0)</h3>
    <ul>
      <li>Linear probes on the harvested slots recover the token at a slot with top-1 accuracy 0.73–0.81 over the full vocabulary (9B top-10: 0.92–0.95), identify the document 20-way at 0.58–0.73 (chance 0.05), and read out which answer choice sits at a gold-answer occurrence at 0.64–0.72 (chance 0.20, n=51). The operator holds the document.</li>
      <li>Native-space similarity from the question cannot find the answer: with the encoder's own residual of the question as query, the gold slot lands in the top-10 on 0.00–0.02 of items; the gold answer text as query finds it on 0.71 (top-100: 0.94–0.96). Token residual states are not question-addressable keys.</li>
      <li>Forcing attention exactly onto the gold slots at eval raises the held-out letter probability by only +0.015 ± 0.014 over random slots, even after 100 registration steps under forced-gold attention; on the items that training saw it rises +0.34. The write-back (B, gate, gain) memorizes item-level lookups rather than reading content. Caveat: MCQ letters require matching injected content against the options; a span-answer task tests the write-back more directly (queued on MuSiQue).</li>
    </ul>
  </section>

  <section>
    <h2>Parallel directions (in progress)</h2>
    <p>Runs by the direction agents, all inside the paper's operator form. D1 = ridge-fitted query map; D2 = slot-level retrieval supervision; D3 = operators from attention keys/values; D4 = reward-learned compact operator per document; D5 = longer / larger registration; D6 = diagnostics. Round 3 (running): D7 = the reader's own attention as the read (A = q_proj, B = o_proj; cross-model = linear key/value alignment); D8 = question-dependent top-k slot selection by native attention, then a compact read; D9 = dense distillation of the interface from the text-in-context teacher (attention-pattern + residual-state targets at scale). Same judge throughout.</p>
    {DIRECTION_TABLE}
  </section>

  <section>
    <h2>Round-2 findings across the directions</h2>
    <ul>
      <li>No interface-only direction passes the held-out gate yet (kv − wrong document ≥ +0.10). The best held-out gaps are D2's slot-supervised LongHealth run (+0.063, n=80) and the D2+D3 combination (+0.025), where the probes show the first real addressing: mass on gold-answer slots 4–9× chance, where every answer-only interface sat at chance.</li>
      <li>Longer and larger registration does nothing by itself: 150 documents × 2000 steps gives +0.013 (n=300); on five documents, 2000 steps reaches +0.205 on the training questions and 0.000 on unseen questions of the same documents. The interface memorizes question→answer pairs; it does not learn to read.</li>
      <li>Changing the keys to attention keys makes the read sharp (entropy 4.6–7.4 vs 8.6–8.9) but aimed at document-independent positions; a ridge-fitted query map from token-aligned 2B↔9B states changes nothing (+0.013 / +0.017).</li>
      <li><strong>The reward-learned compact operator (the paper's second construction) is the one direction that clears the gate as a point estimate.</strong> Per-document operators of 64 slots, SVD-initialized from the centered harvested slots and tuned with the paper's group-normalized reward recipe (T=1.5, 16 prompts × 8 samples, KL-free warmup) through the frozen centered interface, beat other documents' operators by +0.110 on the ten training documents' unseen questions (n=82; 95% CI −0.01 to +0.23; 17 vs 8 paired wins, p≈0.11). The same operators before RL give +0.024, so the reward updates add about +0.09 of document-specific content that the one-prefill operator lacks. Caveats: RL saturates within ~20 steps on ~8 questions per document (under 20% of groups carry reward variance), part of the kv lift is document-independent (wrong document 0.573 vs closed book 0.500), and the interval touches zero. <strong>Confirmation on 30 documents (seed 0, n=262 unseen questions of the trained documents): kv 0.534 vs wrong document 0.450, +0.084</strong>, this time with a real reward signal (train reward 0.3 → 0.9, 7–13 of 16 groups informative). The same operators before any reward update (SVD-compressed, through the frozen interface) give +0.050 (kv 0.527 vs wrong 0.477), so the reward updates add +0.034 of document-specific content and lower the wrong-document score. Seed 1 reproduces it: +0.076 (n=262), so the reward-learned compact operator is reproducibly at about +0.08 across two seeds (its SVD-initialized control is +0.050 for both, so the reward updates add +0.03), below the +0.10 bar but clearly positive. The small-encoder variant (64-slot operators initialized from the 2B's harvest, read by the 9B through a 2048-wide interface) lands in the same band: kv 0.553 vs wrong document 0.477 (+0.076, n=262), so the reward-learned route gives about +0.08 regardless of which encoder seeds it. Its control is the more interesting number: the 2B-initialized operators carry nothing before any reward update (−0.015), whereas the 9B-initialized ones started at +0.05, so on the cross-model seed the reward updates built the whole +0.09 of document-specific content themselves. On LongHealth (unseen questions of trained documents, n=154) the same recipe gives +0.084 against +0.020 before any reward update, so the reward-learned operator is at about +0.08 on both tasks and the reward updates add +0.03 to +0.09 of it. All four runs clear the wrong-document control at p &lt; 0.05 but none reaches +0.10. Two caveats: the interfaces in these runs were registered on all questions of these documents, so only the paired gaps are fair, not the absolute levels; and the method learns a document from questions about it rather than reading an unseen document. A scaled-up run on the small-encoder cell (100 reward steps per document, an interface registered on disjoint documents, two seeds) is running.</li>
      <li>Combined with D6, the picture is: content is present and decodable; the question cannot address token residuals by similarity; slot supervision teaches some addressing; and the write-back cannot turn retrieved content into an MCQ letter without per-item memorization. The span-answer test on MuSiQue (kv 0.15 vs wrong 0.13 vs text 0.47) does not rescue the write-back either.</li>
    </ul>
  </section>

  <section>
    <h2>Round 3: where the document lives in a hybrid reader (D7 ceiling, training-free)</h2>
    <p>Qwen3.5 is a hybrid: 3 of every 4 layers are Gated-DeltaNet, whose document memory is a fixed-size recurrent state S = Σ v kᵀ (24 layers × 32 heads × 128 × 128 ≈ 24 MB for 9B, independent of document length). This is exactly the paper's associative operator M = W Uᵀ with the linear read W(Uᵀq). The other 8 layers are full attention, whose document memory is the per-token K/V (≈ 353 MB for an 11k-token document). D7 harvested both from one prefill of the document by the reader itself and re-injected them as a prefix cache, with no training, on held-out documents (9B reads 9B):</p>
    {STATE_TABLE}
    <ul>
      <li><strong>First pass of the pre-registered gate.</strong> The full-attention K/V alone, read by the reader's own attention, beats the wrong document by +0.15 ± 0.05 (LongHealth, n=80) and +0.125 ± 0.03 (QuALITY, n=200). The operator is the unchanged harvest; what changed is the read: the reader's native multi-head attention at all 8 full-attention layers instead of a learned softmax over slots at 4 depths.</li>
      <li>K/V + DeltaNet state + conv state reproduces the in-context answer (0.81 vs 0.825; 94–95% identical generations), so the harvest is lossless. The state alone is not readable: below closed book on LongHealth (0.24 vs 0.49) and at closed book on QuALITY (0.42 vs 0.41), with wrong-document state equally bad. The state adds +0.24 only on top of the K/V.</li>
      <li>The teacher's own attention from the question's last token puts only 2–4× chance on gold-answer tokens, so "gold-slot mass" (the round-2 probe) cannot gate reading; D2's 4–7× chance was already at teacher level. Retrieval is distributed across heads, layers and positions.</li>
      <li><strong>Which layers carry the read (D7 minimal set, LongHealth, n=80).</strong> Everything below layer 16 — K/V at the four early attention layers plus the conv and recurrent states of the twelve linear layers below them — scores 0.800 against 0.275 for the wrong document; all four early attention layers are needed (any pair ≤ 0.66, any single ≤ 0.59), and the late four contribute nothing (0.24 with states). K/V + the 1.5 MB conv state alone reaches 0.763; the 24 MB recurrent state adds the last +0.05. The wrong document now scores far <em>below</em> closed book: the native read is document-specific. QuALITY (20 held-out documents, n=200) replicates every row: below-layer-16 0.780 vs wrong 0.405; K/V + conv 0.760 vs 0.430; pairs 0.55–0.71, singles 0.43–0.60. And the conv state can be replaced by the reader itself prefilling only the last 64 document tokens: early K/V + that tail gives 0.725 vs 0.4375 on LongHealth and 0.685 vs 0.465 on QuALITY, and with the harvested recurrent state it restores the full 0.81 / 0.785. The cross-model interface therefore only has to map the K/V branch (and optionally the recurrent state), never the conv state.</li>
      <li><strong>Compression (D8).</strong> Pruning the early-layer K/V columns by the attention each position accumulated during the document's own prefill keeps 0.65 at 43 MiB (10% of columns + states) and 0.60 at 34 MiB (5%), against 0.81 at 202 MiB; without the states the pruned K/V collapses to +0.05–0.08 over the wrong document. The states are what makes the compressed operator readable. <strong>Cross-model (D14, 2B→9B encoder-side read):</strong> only the encoder layers the read uses need shipping (exact), 4-bit per-channel quantization of K/V and states is lossless within noise (LongHealth 0.70 vs 0.675 uncompressed at 23.8 MiB; QuALITY 0.613 vs 0.620 at 12.3 MiB, about 6× smaller than the full cache), and keeping 10% of columns by attention plus 4-bit is the smallest operator that still passes both criteria on both tasks (4.2 / 3.0 MiB per document, 25–34× smaller, 0.10–0.14 lower accuracy, QuALITY only +0.02 above the tail control). For the paper form, where the operator is just the read layers' K/V, the same rules give about 22 / 10.5 MiB at 4-bit and 1–6 MiB with pruning; they need re-checking with the reader-side query. QuALITY (n=200) agrees: 10% of columns + states keeps 0.730 at 34 MiB and 5% keeps 0.695 at 30 MiB, against 0.785 at 197 MiB and 0.795 for the text itself; the attention-selected columns beat random ones at 10% (0.73 vs 0.675). Caveat on the gaps: in the with-states rows the wrong-document control also carries the wrong document's states, which pushes it below closed book, so those +0.3 to +0.55 gaps overstate content. The clean identity test is the K/V-only gap: it holds the +0.10 bar down to 25% of the early-layer columns (+0.125 on LongHealth at 44 MiB, +0.09 to +0.11 on QuALITY at 21 MiB) and falls to about +0.075 at 10% and 5%; random pruning at 5% has no document identity left (0.000).</li>
      {COMP_TABLE}
      <li><strong>Cross-model interface (D10), in the paper's form.</strong> A separate read branch at the reader's layers 3,7,11,15: q' = A·q, attention over the encoder's stored keys, value read mapped back by B into the reader's attention output. Same-model identity init reproduces the native ceilings exactly (+0.16 attention branch, +0.575 with the state branch, n=80), and ridge-fitted 9B→9B maps with no identity shortcut do too (+0.175). Answer-only registration of the 2B→9B maps is null (−0.007, n=300; train loss → 0, held-out gap 0, gates ≈ 0), as every answer-only recipe has been. <strong>Closed-form, label-free maps are the first cross-model signal:</strong> ridge-fitting A on token-aligned 9B/2B queries and B on values (identical tokenizers give a pair per document token) and reading with zero training gives kv 0.48 vs wrong 0.44 on QuALITY held-out (+0.040 ± 0.017, n=200, 10 wins / 2 losses) with addressing divergence 0.26–0.38 (same-model 0.11–0.20, uniform 0.42–0.52). The same QuALITY-fitted maps barely address LongHealth (+0.0125, divergence 0.41), so the linear map is domain-sensitive; a refit on a diverse document pool did not help (+0.023, n=300), because the held-out R² of a per-layer linear query map is only 0.35–0.45 on every domain, and fitting A directly on the teacher's attention pattern moved the divergence to 0.21–0.29 but gave no held-out gain on LongHealth (−0.025 / −0.0125). One false positive is worth recording: with the reader's own 64-token tail, the 2B operator through ridge maps gives kv 0.44 vs wrong document 0.33 on QuALITY (+0.107, n=300) while the reader with only the tail scores 0.51 and the wrong-document runs generate twice as many tokens, so the gap is asymmetric derailment, not reading. The judge therefore now also requires kv ≥ the closed-book control (do no harm). The attention-fitted maps satisfy that but add nothing: on QuALITY with the tail, kv 0.513 equals the tail-only control (0.51) and beats the wrong document by +0.04 (n=300) with normal generation length, while the branch's mass on document slots fell to 0.12 against the teacher's 0.62, i.e. the fit learned to keep the branch quiet rather than to address. <strong>Round 4 update, the paper's read as written:</strong> training the reader-side query map A without labels to reproduce the encoder's own question-pass attention over its cache (held-out attention divergence 0.094, read-out cosine 0.94 on unseen documents) gives the first reader-query read that passes both criteria: kv 0.4875 vs wrong document 0.3375 (+0.15 ± 0.05) and above the tail-only control 0.4375, LongHealth fold 0, n=80, with the operator untouched and no answer labels; the old ridge A on the same branch gives 0.325 (fails). A nonlinear (MLP) A raises it to 0.5375 vs 0.3875, 0.10 above the control. On QuALITY (n=300) the paper-form read is not yet over the bar: linear A 0.547 vs 0.467 (+0.08), MLP A 0.563 vs 0.473 (+0.09), both above the tail-only control 0.51 and no better than the ridge baseline there. <strong>Then training with the paper's own KL + CE objective, starting from the distilled A, gives the paper-form read 0.7625 vs wrong document 0.425 on LongHealth fold 0 (+0.34 ± 0.05, 27 wins / 0 losses, n=80; tail-only 0.4375, text 0.825)</strong>: higher than the encoder-side read with the same 2B encoder (0.675), with only the reader running at inference and the 2B's harvested operator untouched. <strong>QuALITY now passes too:</strong> 0.633 vs wrong document 0.453 (+0.18 ± 0.03, 75 wins / 21 losses, n=300), above the tail-only control 0.51 by +0.12, also above the encoder-side read with the same encoder (0.61). The paper-form read passes both criteria on both tasks. Two fallbacks remove the query map: map the operator's keys once (D7's key maps reach R² 0.54–0.70) and let the reader address natively, or let the 2B address its own cache over the question and map only the read-out value into the reader. <strong>The second one is the first cross-model pass on both criteria, training-free.</strong> Encoder-side addressing: the 2B prefills the document once, then processes the question over that cache so its own queries address its own keys (the Read happens in the operator's native space); at the reader's layers 3, 7, 11, 15 the 2B's per-token attention output is mapped by a closed-form ridge B (fitted on token-aligned 2B→9B attention outputs, held-out R² 0.51–0.60) and replaces the reader's attention output, with the reader's own 64-token tail. LongHealth fold 0, n=80: kv 0.5875 vs wrong document 0.375 (+0.21 ± 0.05, 19 wins / 2 losses), above the tail-only control 0.4375 and closed book 0.4875, generation length normal (6.0–6.1 tokens in every condition); text 0.825. Read-out fidelity to the 9B's own in-context attention output is cosine 0.83–0.90. The same-model anchor (9B addresses its own cache, B = identity) gives 0.775 vs 0.325. The interface here is B alone, fitted in closed form from paired prefills with no labels and no training. Mapping the 2B's attention output after its output projection (into the reader's residual space) is better still: kv 0.625 vs wrong document 0.40 (+0.225 ± 0.06, 22 wins / 4 losses, n=80), while mixing the mapped read-out with the reader's own document-free attention output only dilutes it (50% mix 0.50, 25% mix 0.49). The other fallback, mapping the operator's keys and values once into the reader's spaces (D7's key maps, R² 0.54–0.70) and letting the reader address them natively, is null (kv 0.45 vs wrong 0.425, n=80; divergence 0.39), so reader-side addressing over a linearly mapped foreign cache fails in both directions: addressing must happen in the operator's native space and only the write-back crosses models. <strong>QuALITY confirms the encoder-side read on 300 held-out questions: kv 0.600 vs wrong document 0.430 (+0.17 ± 0.03, 64 wins / 13 losses), tail-only control 0.510, closed book 0.450, text 0.810, generation length normal.</strong> Both tasks now pass both criteria with the same closed-form map and no training. <strong>At full n (D11):</strong> over all 400 LongHealth questions (two-way cross-fitted B, no document seen in fitting) 2B→9B scores 0.777 vs 0.405 and 4B→9B 0.820–0.823 vs 0.37–0.38 (text 0.86); on the full QuALITY test (n=1,057) 2B→9B 0.609 vs 0.412 and 4B→9B 0.762–0.791 vs 0.42–0.44 (text 0.828). The read reaches 96–101% of the encoder's own text accuracy on both multiple-choice tasks. A tail of 0 tokens gives the same kv (the reader need not see any document token), and a B fitted without any LongHealth document performs the same on LongHealth. On the span-answer tasks (MuSiQue, LastingBench) the read fails do-no-harm for both encoders; outputs are well formed, and the failures are wrong entities or wrong hops (only 22–38% of the encoder's own text F1). Reading at all eight attention layers instead of the early four lifts LongHealth to kv 0.6625 vs wrong document 0.375 (+0.29 ± 0.05, 24 wins / 1 loss, n=80), against a same-model anchor of 0.7875. <strong>With the post-projection read-out at all eight layers, 2B→9B reaches kv 0.7875 vs wrong document 0.325 (+0.46 ± 0.06, 38 wins / 1 loss, n=80): equal to the same-model anchor and close to the 9B reading the text itself (0.825).</strong> A nonlinear B does not improve on the linear one (0.6375 vs 0.6625 at the pre-projection level), so the closed-form write-back is not the ceiling. On QuALITY the post-projection level at four layers gives kv 0.610 vs wrong document 0.443 (+0.167, n=300), matching the pre-projection level. <strong>But the eight-layer configuration does not carry to QuALITY:</strong> kv 0.463 is below the tail-only control (0.517) and at closed book, with the wrong document at 0.32, so that gap is harm asymmetry and the configuration fails the do-no-harm rule there. The robust configuration is the early four layers, which beat the control on both tasks. The likely reason is that replacing the reader's late attention layers imports the small encoder's own reasoning, which is weak on QuALITY (the 2B scores 0.60 with the text there vs 0.79 on LongHealth), whereas the early layers import only its retrieval. The layer-subset sweep (2B→9B, post level) locates the turn: on QuALITY four layers give 0.610, five (3–19) give 0.623, six (3–23) drop to 0.543 and eight to 0.463, against the 0.517 control; on LongHealth the same sets give 0.625 / 0.675 / 0.71 / 0.7875. Seven layers give 0.553, and replacing the early five while adding the late layers at half gain gives 0.623 vs 0.46, no better than five layers alone. Five layers of replacement is the best configuration that holds on both tasks. <strong>Encoder-size axis, first row:</strong> a 4B encoder at the robust four-layer configuration gives the 9B kv 0.8125 vs wrong document 0.375 on LongHealth (+0.44 ± 0.06, 37 wins / 2 losses, n=80; tail-only control 0.45), i.e. the 9B's own in-context level (0.825) from a 4B operator, training-free; and on QuALITY kv 0.773 vs wrong document 0.427 (+0.35 ± 0.03, 111 wins / 7 losses, n=300; control 0.517; text 0.81). Operator quality tracks the encoder's own competence (2B: 0.60 / 0.79 with the text on QuALITY / LongHealth; 4B: 0.81 / 0.87), so a 4B encoder already brings the 9B within four to five points of reading the text itself on both tasks. The first 27B-reader run (2B encoder) is null (kv 0.3875 vs tail-only control 0.475, n=80), but it used the 27B's first four attention layers (depths 0.05–0.25) rather than the depth-matched band the 9B configuration means (depths 0.1–0.5, i.e. the 27B's eight attention layers below depth 0.5); with the depth-matched set the 27B's same-model anchor holds (kv 0.775 vs wrong document 0.45, n=80; text 0.8625), and the depth-matched 2B→27B run is weak: kv 0.5125 vs wrong document 0.4375 (+0.075, n=80), just above the tail-only control (0.475), with the read-out maps' held-out R² falling to 0.24–0.33 at the deeper 27B layers, so a 13× size ratio strains a closed-form linear write-back. 9B→27B (3×) gives kv 0.6125 vs wrong document 0.5125 (+0.10, n=80) against the control 0.475 and the 27B's own anchor 0.775; the wrong document itself sits above the control, so part of what the linear B carries into the 27B is document-independent. On QuALITY (n=200) 2B→27B gives kv 0.61 vs wrong document 0.52 (+0.09; control 0.565; the 27B reads the text at 0.935). 9B→27B on QuALITY passes both criteria: kv 0.685 vs wrong document 0.51 (+0.175, n=200; control 0.565). 4B→27B lands in the same band (0.60 vs 0.475 on LongHealth, +0.125; 0.675 vs 0.515 on QuALITY, +0.16, n=200, passing both criteria), so into the 27B the encoder's size barely moves the result, which points at the write-back rather than the operator as the cap. The 27B also needs its whole lower-half band: the five-layer analogue (depth below 0.3) is at or below the control for every encoder (2B / 4B / 9B: 0.39 / 0.44 / 0.41 on LongHealth, +0.00 / +0.03 / +0.01 on QuALITY), and the 27B's own same-model anchor with only those five layers replaced is 0.40 vs wrong document 0.50, so partial early replacement harms the 27B even with its own exact read-outs: a property of the reader, not of the encoders. Into a 27B reader the closed-form write-back is the weak link, and refitting B with a regularization sweep does not move it (9B→27B: 0.5875 vs 0.5125 on LongHealth, 0.695 vs 0.525 on QuALITY; adding the read-out at half gain instead of replacing is weaker still, +0.05), so the cap is structural for a linear token-wise write-back into the 27B rather than a fitting problem. <strong>Round 4 lifts it:</strong> a closed-form write-back that takes the adjacent encoder layers as input brings 9B→27B on the 8-layer set to 0.7125 vs 0.4875 on LongHealth (n=80; 27B anchor 0.775) and 0.74 vs 0.54 on QuALITY (+0.20, n=200; tail-only 0.565), up from 0.61 and 0.685, still with no training; replacing all 16 layers instead fails the do-no-harm rule on QuALITY (0.50 vs 0.565). <strong>Training that write-back with the paper's KL + CE objective</strong> (2,000 steps from the adjacent-layer closed-form init, selected on held-out documents) brings 9B→27B to 0.7875 vs 0.375 on LongHealth (n=80; the 27B's own-cache anchor is 0.775) and 0.855 vs 0.56 on QuALITY (+0.295, n=200; text 0.935, tail-only 0.565). KL + CE fails from scratch because the read cannot yet address; on top of a read that addresses it is the strongest lever we have.</li>
      <li><strong>Cost and framing.</strong> Per LongHealth question, the 9B prefilling the text costs about 2.3e14 FLOPs; the encoder-side read costs about 6.4e12 when one 2B document prefill is shared over ~20 questions (36× less) or 5.8e13 without reuse (4× less), and the shipped operator is about 156 MB per document (2B K/V plus 9.4 MB of states) against 415 MB for the 9B's own cache. One caveat on form: in the passing configuration the Read happens in the operator's own space with the encoder's queries, and only the write-back B crosses models; every variant that used the reader's own query against the foreign operator, which is the paper's read as written, failed on held-out documents. The operator is still the untouched harvest and the interface is still a closed-form map, but whether this counts as the paper's read form is a judgement for the authors. Next: refine B label-free (toward the same-model anchor), the post-projection level on QuALITY, and the encoder-size axis (4B→9B, 2B→27B, 9B→27B).</li>
      <li><strong>Dense distillation into the old softmax-slot read (D9)</strong> trains its targets (attention KL 2.4 → 0.85 nats, residual ratio → 0.24–0.35) and still fails both judges (+0.010 QuALITY n=300, −0.0125 LongHealth n=80), confirming that the slot read, not the training signal, was the limit. Its teacher-attention targets are being reused for D10.</li>
      <li>Consequences now running: D10 builds the cross-model interface in the paper's form on both branches (per-layer A maps the large reader's query heads into the small encoder's key space, B maps the read value back; identity init must reproduce this ceiling in the same-model case); D8 compresses the K/V branch (keep 25/10/5% of document positions by document-level attention importance vs random) to find the smallest native operator that still clears the gate.</li>
    </ul>
  </section>

  <section>
    <h2>Size ladders</h2>
    <div class="charts">{charts}</div>
    <p>The large-minus-small gap with the document in context sits between 2B and 4B; 4B and 9B tie, and 27B gains another 10 points on QuALITY. Closed-book accuracy also rises with size, so any future gain must be measured against the wrong-document control, not against closed book.</p>
  </section>

  <section>
    <h2>What this decides</h2>
    <ul>
      <li>For the one-prefill operator with answer-only registration, the read does not transmit document facts, so H1–H4 cannot be tested on it yet. Centering, routing and oracle chunk selection do not change this. The operator form is not refuted: the missing piece is an addressing signal for the interface.</li>
      <li>Five directions are now running in parallel, all inside the paper's operator form: a ridge-fitted query map A (token-aligned 2B↔9B states), slot-level retrieval supervision during registration, operators harvested from attention keys/values instead of residual states, the paper's reward-learned compact operator per document, and longer/larger registration. Each is judged by the same kv − wrong-document gate.</li>
      <li>If none of them passes the wrong-document control, the follow-up should pivot to cross-model <em>procedural</em> skills, where the published mechanism is known to work.</li>
    </ul>
  </section>

  <section>
    <h3>Run inventory</h3>
    <p class="meta">Code: <code>KV_skill_v2/xenc</code> (52 CPU tests green). Data: QuALITY 265 docs / 4,609 Qs; LongHealth 20 docs / 400 Qs, 5 folds. Jobs: 1 smoke, 3 encodes, 6 ladders, 9 registrations, 9 evals, 1 probe on Qwen3.5-9B/27B (L40S/H100). Results: <code>outputs/RESULTS.json</code>.</p>
  </section>
</main>
"""
out = ROOT / "docs" / "report.html"
out.write_text(page)
print("wrote", out, len(page), "bytes")
