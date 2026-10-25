"""SOFTCONF prior probe (agent SOFTCONF, 2026-10-25): the reader's CLOSED-BOOK prior for every SOFTCONF question,
over P cyclic option orders (GPU, reader only, no document, no encoder).

Per question (meta.qid; all levels of a question share the option set, so one probe serves every level):
  for k in 0..P-1: rotate the option TEXTS by k, prompt = the eval prompt without the document
  (build_chat(SYSTEM_ANSWER, q_prompt(item))) + "<answer>", log-softmax over the option letters (= eval_cad's
  closed_book forced-choice row at k = 0); map letters back to option ROLES.
  prior_role = argmax of the mean role probability; stable = the same role wins in every order;
  pmax = mean probability of prior_role; per-order probabilities are stored.
scripts/softconf_table.py uses: conflict stratum = stable & prior_role == the original value (role L0 / m<o>).

    python scripts/softconf_prior.py --reader Qwen/Qwen3.5-27B --split_dirs <sc split dirs> --out outputs/SOFTCONF/prior_27b.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def rotate(item: dict, k: int) -> tuple[dict, dict]:
    """Option texts shifted by k positions (labels stay A, B, ...). -> (new item, {new label: role})."""
    ch = item["choices"]
    roles = item["meta"]["roles"]
    n = len(ch)
    order = [(i + k) % n for i in range(n)]
    nch = [{"label": ch[j]["label"], "text": ch[order[j]]["text"]} for j in range(n)]
    lab2role = {ch[j]["label"]: roles[ch[order[j]]["label"]] for j in range(n)}
    return {**item, "choices": nch}, lab2role


def summarize(per: list[dict]) -> dict:
    """per: one {role: prob} per option order -> prior summary."""
    roles = sorted(per[0])
    mean = {r: sum(p[r] for p in per) / len(per) for r in roles}
    wins = [max(p, key=p.get) for p in per]
    pr = max(mean, key=mean.get)
    return {"prior_role": pr, "stable": len(set(wins)) == 1, "pmax": mean[pr], "p_mean": mean, "wins": wins,
            "pmax_min": min(p[pr] for p in per)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--split_dirs", required=True, help="comma list")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reader_mem", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from register_doc import SYSTEM_ANSWER
    from xenc.backbone import build_chat, load_reader
    from xenc.doc_tasks import q_prompt
    reader, tok = load_reader(a.reader, device_map=a.device, max_mem=a.reader_mem)
    reader.eval()
    ansp = tok("<answer>", add_special_tokens=False)["input_ids"]
    letter = {L: int(tok(L, add_special_tokens=False)["input_ids"][0]) for L in "ABCDEFGHIJ"}
    dev = next(reader.parameters()).device
    res, t0 = {}, time.time()
    for sd in a.split_dirs.split(","):
        for sp in a.splits.split(","):
            f = Path(sd, sp, "items.json")
            if not f.exists():
                continue
            for it in json.load(open(f)):
                qid = it["meta"]["qid"]
                if qid in res:
                    continue
                per = []
                for k in range(a.perms):
                    rit, l2r = rotate(it, k)
                    ids = list(build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(rit))) + list(ansp)
                    with torch.no_grad():
                        lg = reader(input_ids=torch.tensor([ids], device=dev), logits_to_keep=1).logits[0, -1].float()
                    labs = [c["label"] for c in rit["choices"]]
                    lp = torch.log_softmax(lg[[letter[x] for x in labs]], -1).exp().tolist()
                    per.append({l2r[x]: p for x, p in zip(labs, lp)})
                res[qid] = {**summarize(per), "split": sp, "axis": it["meta"]["axis"]}
                if len(res) % 100 == 0:
                    print(f"  {len(res)} questions, {(time.time() - t0) / len(res):.2f}s/q", flush=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    json.dump({"reader": a.reader, "perms": a.perms, "gpu": gpu, "q": res}, open(a.out, "w"), indent=0)
    st = sum(v["stable"] for v in res.values())
    print(f"==== prior probe {a.reader}: {len(res)} questions, stable {st}, -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
