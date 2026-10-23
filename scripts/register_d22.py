"""D22: judge a fixed, leakage-filtered item set through D20's per-document registration (scripts/register_d20.py,
unmodified). The only change: register_d20.judge_items is replaced (in-process) by "exactly the item keys stored in
--d22_set" (scripts/d22_make_sets.py), so the leaky test documents never enter the judge. Every other flag, the
leakage assertion (no task question in any registration prompt), registration and eval are register_d20's.

    python scripts/register_d22.py --d22_set outputs/D22_sets/qasper.json --wrong_map outputs/D22_sets/qasper_wrongmap.json \
        --selfq outputs/D22_selfq/qasper.jsonl ... --split_dir data/qasper_split --split test --out outputs/D22_qasper
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import register_d20  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    i = argv.index("--d22_set")
    S = json.load(open(argv[i + 1]))
    del argv[i: i + 2]
    keys = set(S["item_keys"])

    def judge_items(a):
        from register_doc import _get, load_split_items
        assert not a.n and not a.max_docs and a.fold is None, "--d22_set fixes the sample: leave --n/--max_docs/--fold unset"
        splits = S.get("splits") or [a.split]   # LooGLE-long draws from every LooGLE split (nothing trained on LooGLE)
        items = [it for s in splits for it in load_split_items(a.split_dir, s) if _get(it, "key") in keys]
        assert len(items) == len(keys), f"{len(items)} of {len(keys)} D22 items found in {a.split_dir}/{splits}"
        return items, sorted({_get(it, "doc_id") for it in items})

    register_d20.judge_items = judge_items
    return register_d20.main(argv)


if __name__ == "__main__":
    sys.exit(main())
