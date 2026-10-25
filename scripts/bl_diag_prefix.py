"""BL S5 sanity diagnostic: does kvc_full (prefix cache of [template + document] + continuation over the question
suffix) compute the same thing as the single-pass text row? For each item: identical input ids (asserted), then the
answer-letter logits at the step after "<answer>" (teacher-forced) under (a) one forward over the full prompt and
(b) prefill of the question-agnostic prefix P + continuation, in the requested dtype / device.

    python scripts/bl_diag_prefix.py --split_dir data/mcq_bamboo_split --d22_set outputs/MCQ_sets/mcq_bamboo.json \
        --keys mcq_bamboo_test_00035,mcq_bamboo_test_00001 --dtype float32 --device cpu --out outputs/BL_pilot_diag_fp32.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from eval_textbudget import select_items  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d22_set", default="")
    ap.add_argument("--keys", default="")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from register_doc import SYSTEM_ANSWER, _get
    from xenc.backbone import build_chat, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs
    from xenc.kvcomp import split_prompt
    if a.device == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    model, tok = load_reader(a.reader, dtype=getattr(torch, a.dtype), device_map=a.device)
    dev = model.get_input_embeddings().weight.device
    docs = load_docs(a.split_dir)
    a.keys_from = ""
    items = select_items(a)
    want = [k for k in a.keys.split(",") if k]
    items = [i for i in items if not want or _get(i, "key") in want]
    ans = tok("<answer>", add_special_tokens=False)["input_ids"]
    rows = []
    for it in items:
        did = _get(it, "doc_id")
        user = ctx_prompt(it, docs[did])
        ids_text = build_chat(tok, a.reader, SYSTEM_ANSWER, user)
        ids, ds, P = split_prompt(tok, a.reader, SYSTEM_ANSWER, user, docs[did], len("## Document\n"))
        assert ids == ids_text, "kvc and text inputs differ"
        labs = [c["label"] for c in _get(it, "choices")]
        lab_ids = [tok(f"{x}", add_special_tokens=False)["input_ids"][0] for x in labs]
        full = torch.tensor([ids + ans], device=dev)
        t = time.time()
        with torch.no_grad():
            la = model(input_ids=full, logits_to_keep=1).logits[0, -1].float()
            o = model(input_ids=full[:, :P], use_cache=True, logits_to_keep=1)
            lb = model(input_ids=full[:, P:], past_key_values=o.past_key_values, use_cache=True,
                       logits_to_keep=1).logits[0, -1].float()
        sa, sb = la[lab_ids], lb[lab_ids]
        r = {"key": _get(it, "key"), "gold": _get(it, "correct_label"), "n_tokens": len(ids), "P": P,
             "single_pass": dict(zip(labs, [round(x, 4) for x in sa.tolist()])),
             "prefix_cache": dict(zip(labs, [round(x, 4) for x in sb.tolist()])),
             "argmax_single": labs[int(sa.argmax())], "argmax_prefix": labs[int(sb.argmax())],
             "max_abs_letter_logit_diff": float((sa - sb).abs().max()),
             "top2_margin_single": float(sa.sort(descending=True).values[0] - sa.sort(descending=True).values[1]),
             "secs": round(time.time() - t, 1)}
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump({"args": vars(a), "device_name": (torch.cuda.get_device_name(0) if a.device != "cpu" else "cpu"),
               "rows": rows}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
