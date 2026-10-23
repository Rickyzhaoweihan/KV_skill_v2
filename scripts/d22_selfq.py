"""D22: label-free self-questions for span-answer benchmarks — a thin, additive wrapper around scripts/d20_selfq.py.

Adds one template (D20's file is not modified; its TEMPLATES / parse are extended in-process):
  span  3 short-answer questions on part k whose answer is a short phrase (1-6 words: a name, number, date, place,
        term) copied from the document — the answer FORMAT of span benchmarks (Qasper, MuSiQue, LastingBench, LooGLE).
Everything else (one shared document prefill fanned out to S streams, the leakage drop against every task question,
the balanced keep) is d20_selfq's. As there, the generator's own answers are stored but never used: registration
targets are the 9B text teacher's answers (scripts/register_d20.py).

    python scripts/d22_selfq.py --split_dir data/qasper_split --docs test:120 --templates span,mix,mcq \
        --out outputs/D22_selfq/qasper.jsonl
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import d20_selfq  # noqa: E402

SPAN_TEMPLATE = (
    "Write 3 short-answer questions that can only be answered by reading this document. Split the document into {S} "
    "equal parts and take all 3 questions from part {k}. Each answer is a short phrase of 1 to 6 words copied exactly "
    "from the document (a name, number, date, place, method, term or quantity); the three questions ask about "
    "different details, and at least one asks for a number or a name. Use exactly this format and nothing else:\n"
    "Q1: <question>\nA1: <short answer>\nQ2: <question>\nA2: <short answer>\nQ3: <question>\nA3: <short answer>")

_parse20 = d20_selfq.parse


def parse_span(txt: str) -> list[dict]:
    out = []
    for n in (1, 2, 3):
        m = re.search(rf"Q{n}:\s*(.+?)\s*\n\s*A{n}:\s*(.+?)\s*(?:\n|$)", txt)
        if not m:
            continue
        q, a = m.group(1).strip(), m.group(2).strip().strip('"').strip()
        if 1 <= len(a.split()) <= 8:
            out.append({"question": q, "answers": [a], "choices": None, "correct_label": None})
    return [q for q in out if d20_selfq._ok(q)]


def parse(tmpl: str, txt: str) -> list[dict]:
    return parse_span(txt) if tmpl == "span" else _parse20(tmpl, txt)


_sel20 = d20_selfq.doc_selection


def doc_selection(split_dir: str, spec: str) -> list[str]:
    """Also accepts "file:<outputs/D22_sets/<task>.json>" = exactly the documents D22 registers (judged + wrong)."""
    if spec.startswith("file:"):
        import json
        return sorted(json.load(open(spec[5:]))["reg_docs"])
    return _sel20(split_dir, spec)


def install():
    d20_selfq.TEMPLATES.setdefault("span", SPAN_TEMPLATE)
    d20_selfq.parse = parse
    d20_selfq.doc_selection = doc_selection


if __name__ == "__main__":
    install()
    sys.exit(d20_selfq.main())
