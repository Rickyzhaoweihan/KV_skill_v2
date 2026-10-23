"""P5: first-answer-token coverage of the copy tables on a val spec (tokenizer only, CPU).

For every non-MCQ val item, the gold target "<answer>X</answer>" (as teacher-forced) is checked against the operator's
body token ids (exactly D13Rt.pre(did)[:-tail], body = after the chat-template prefix):
  plain    first content token id occurs among the body ids
  nosp     ... or among the no-space variants (rows fed '>')
  both     ... or among the merged '>'-variants (rows fed 'answer')   [the row's own map: what P5 uses]
  occ      the whole content token sequence occurs verbatim (span-supervisable)
  later    later content tokens occur among the body ids
    python scripts/p5_coverage.py <val args as register_d13> --out outputs/P5_coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--val", required=True)
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--exclude_sets", default="")
    ap.add_argument("--exclude_docs", default="")
    ap.add_argument("--doc_len", default="")
    ap.add_argument("--max_tokens", type=int, default=100000)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from fit_d13 import select_items
    from register_d13 import doc_len_filter, excluded_docs
    from xenc.doc_tasks import gold_text
    from xenc.p4_ptr import template_ids
    from xenc.p5_copy import content_rows_strict, occurrences, space_maps
    from xenc.state_skill import doc_prefix_text
    tok = AutoTokenizer.from_pretrained(a.reader)
    V = 248320
    nosp, gtm, gt, ans = space_maps(tok, V)
    pre_t = template_ids(tok, a.reader, SYSTEM_ANSWER)
    val, docs = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    drop = excluded_docs(a.exclude_sets)
    if a.exclude_docs:
        X = json.load(open(a.exclude_docs))
        drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    keep = doc_len_filter(a.doc_len)
    val = [i for i in val if _get(i, "doc_id") not in drop and keep(_get(i, "doc_id"))]
    cache, per = {}, {}
    tot = {"n": 0, "strict_rows": 0, "plain": 0, "nosp": 0, "both": 0, "occ_nosp": 0, "occ_both": 0,
           "later_n": 0, "later": 0, "merged_items": 0}
    for it in val:
        if _get(it, "choices"):
            continue
        did = _get(it, "doc_id")
        if did not in cache:
            ids = [int(x) for x in tok(doc_prefix_text(tok, a.reader, docs[did], SYSTEM_ANSWER),
                                       add_special_tokens=False)["input_ids"]]
            ids = ids[-(a.max_tokens - 1024):][: -a.tail]
            S = 0
            while S < len(pre_t) and S < len(ids) and ids[S] == pre_t[S]:
                S += 1
            S = S if S < len(ids) - 1 else 0
            b = torch.tensor(ids[S:])
            g_ = gtm[b]
            cache[did] = {"ids": b, "nosp": nosp[b], "gtm_m": torch.where(g_ == gt, torch.full_like(g_, -1), g_)}
        tab = cache[did]
        t = did.split("_")[0]
        d = per.setdefault(t, {k: 0 for k in tot})
        for D in (tot, d):
            D["n"] += 1
        tt, rows = content_rows_strict(tok, gold_text(it), allow_merged=True)
        if not rows:
            continue
        c0, prev = tt[rows[0]], tt[rows[0] - 1]
        merged = prev == ans
        no_tab = dict(tab, gtm_m=torch.full_like(tab["ids"], -1))
        r = {"strict_rows": 1, "merged_items": int(merged), "plain": int(bool((tab["ids"] == c0).any())),
             "nosp": int(bool(((tab["ids"] == c0) | (tab["nosp"] == c0)).any())),
             "both": int(bool((((tab["gtm_m"] == c0) if merged else (tab["nosp"] == c0)) | (tab["ids"] == c0)).any())),
             "occ_nosp": int(occurrences([tt[i] for i in rows], no_tab).numel() > 0),
             "occ_both": int(occurrences([tt[i] for i in rows], tab).numel() > 0),
             "later_n": len(rows) - 1, "later": sum(int(bool((tab["ids"] == tt[i]).any())) for i in rows[1:])}
        for D in (tot, d):
            for k, v in r.items():
                D[k] += v

    def rates(D):
        n = max(D["n"], 1)
        return {"n": D["n"], "merged_first_token": round(D["merged_items"] / n, 3),
                **{k: round(D[k] / n, 3) for k in ("plain", "nosp", "both", "occ_nosp", "occ_both")},
                "later_in_doc": round(D["later"] / max(D["later_n"], 1), 3)}
    out = {"all": rates(tot), "per_task": {t: rates(D) for t, D in per.items()}}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
