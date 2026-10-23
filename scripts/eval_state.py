"""D7 phase 2: where does document content live in the hybrid Qwen3.5 reader? (training-free, answer accuracy)

(D7 file; D10's cross-model evaluator is scripts/eval_state_xm.py.)
The document is prefilled ONCE as raw tokens (no chat template); the reader then continues with the closed-book
chat prompt (system + question) from a cache that keeps a chosen part of the document memory:

  closed_book   prompt alone (no document memory)
  text          document inside the prompt (eval_doc's text condition, reference)
  all           K/V of the full-attention layers + DeltaNet recurrent + conv states  (== in-context [doc; prompt])
  kv_only       full-attention K/V only (DeltaNet layers start from zero state)
  state_only    DeltaNet recurrent + conv states only (no K/V; fixed size, independent of doc length)
  rec_only      DeltaNet recurrent states only (conv zero)
  wrong_kv / wrong_state   the same with ANOTHER document's memory (identity controls)
  tile_state / wrong_tile  (--encoder hub) states of a smaller ENCODER mapped by normalized depth + head tiling

    python scripts/eval_state.py --reader Qwen/Qwen3.5-9B --split_dir data/longhealth_split --split test --fold 0 \
        --out outputs/D7_state_lh
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import torch  # noqa: E402

import xenc  # noqa: E402,F401

CONDS = ("closed_book", "text", "all", "kv_only", "state_only", "rec_only", "wrong_kv", "wrong_state")
KEEP = {"all": (1, 1, 1), "kv_only": (1, 0, 0), "state_only": (0, 1, 1), "rec_only": (0, 1, 0),
        "wrong_kv": (1, 0, 0), "wrong_state": (0, 1, 1), "wrong_all": (1, 1, 1),
        "tile_state": (0, 1, 1), "wrong_tile": (0, 1, 1),
        # ablations (kv, state, conv[, alpha, masked half of the full-attention layers])
        "kv_conv": (1, 0, 1), "conv_only": (0, 0, 1),
        "all_a0.25": (1, 1, 1, 0.25), "all_a0.5": (1, 1, 1, 0.5),
        "state_a0.25": (0, 1, 1, 0.25), "state_a0.5": (0, 1, 1, 0.5),
        "kvE_state": (1, 1, 1, 1.0, "late"), "kvL_state": (1, 1, 1, 1.0, "early"),
        "kvE": (1, 0, 0, 1.0, "late"), "kvL": (1, 0, 0, 1.0, "early")}
MINIMAL = ("closed_book,all,kv@3.7.11.15+cv+rec,kv@3.7.11.15+cv<16+rec<16,kv@3.7+cv<8+rec<8,"
           "kv@3+cv+rec,kv@7+cv+rec,kv@11+cv+rec,kv@15+cv+rec,kv@3.7+cv+rec,kv@11.15+cv+rec,kv@3.7.11.15+cv,"
           "wrong:kv@3.7.11.15+cv<16+rec<16,wrong:kv@3.7+cv<8+rec<8,wrong:kv@3.7.11.15+cv")


def parse_spec(name: str) -> dict:
    """Compact memory spec, '+'-joined: kv | kv@3.7 (KV kept only at those full-attention layers, rest masked),
    rec | rec<16 (recurrent states of linear layers < 16), cv | cv<16 (conv states); prefix 'wrong:' = another
    document's memory. Example: 'kv@3.7.11.15+cv<16+rec<16'."""
    wrong = name.startswith("wrong:")
    out = {"wrong": wrong, "kv": None, "rec": None, "cv": None}
    for part in (name[6:] if wrong else name).split("+"):
        key, _, arg = part.partition("@") if "@" in part else part.partition("<")
        if key not in ("kv", "rec", "cv"):
            raise ValueError(f"bad spec part {part!r} in {name!r}")
        if "@" in part:
            out[key] = ("set", {int(x) for x in arg.split(".")})
        elif "<" in part:
            out[key] = ("lt", int(arg))
        else:
            out[key] = ("all", None)
    return out


def _sel(rule, layers):
    if rule is None:
        return set()
    kind, arg = rule
    return set(layers) if kind == "all" else ({l for l in layers if l in arg} if kind == "set" else
                                              {l for l in layers if l < arg})


def spec_bytes(st, kv_layers, rec_layers, cv_layers) -> int:
    nb = lambda t: t.numel() * t.element_size()
    return (sum(nb(k) + nb(v) for li, (k, v) in st["kv"].items() if li in kv_layers)
            + sum(nb(st["recurrent"][li]) for li in rec_layers) + sum(nb(st["conv"][li]) for li in cv_layers))


TAIL = ("closed_book,all,kv@3.7.11.15+cv+rec,kv@3.7.11.15+cv,"
        "tail64,tail256,tail1024,kvE+tail64,kvE+tail256,kvE+tail1024,kvE+rec+tail64,kvE+rec+tail256,"
        "wrong:kvE+tail64,wrong:kvE+tail256,wrong:kvE+tail1024")


ABLATIONS = ("closed_book,all,kv_only,state_only,all_a0.5,all_a0.25,kv_conv,state_a0.5,state_a0.25,conv_only,"
             "kvE_state,kvL_state,kvE,kvL")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default=None, help="optional smaller encoder for the tile_state probe")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--max_docs", type=int, default=0, help="first N sorted doc_ids of the split (0 = all)")
    ap.add_argument("--n", type=int, default=0, help="sample n items (0 = all)")
    ap.add_argument("--conds", default=",".join(CONDS), help="or 'ablations' for the D7 state/layer ablation set")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--max_doc_tokens", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from register_doc import _get, load_split_items, system_prompt
    from xenc.backbone import build_chat, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt, score
    from xenc.native_state import (PrefixMask, build_prefix_cache, continue_logits, greedy_continue, harvest_states,
                                   layer_types, state_nbytes, tile_states)

    conds = [c.strip() for c in {"ablations": ABLATIONS, "minimal": MINIMAL, "tail": TAIL}.get(a.conds, a.conds).split(",") if c.strip()]
    if a.encoder:
        conds += [c for c in ("tile_state", "wrong_tile") if c not in conds]
    docs = load_docs(a.split_dir)
    items = load_split_items(a.split_dir, a.split)
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    doc_ids = sorted({_get(i, "doc_id") for i in items if _get(i, "doc_id") in docs})
    if a.max_docs:
        doc_ids = doc_ids[: a.max_docs]
    items = [i for i in items if _get(i, "doc_id") in set(doc_ids)]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    wrong = {d: doc_ids[(k + 7) % len(doc_ids)] if doc_ids[(k + 7) % len(doc_ids)] != d
             else doc_ids[(k + 1) % len(doc_ids)] for k, d in enumerate(doc_ids)}
    print(f"{len(items)} items over {len(doc_ids)} docs; conds {conds}", flush=True)

    model, tok = load_reader(a.reader, device_map=a.device)
    hub = a.reader
    enc = etok = None
    H_dst = None
    if a.encoder:
        enc, etok = load_reader(a.encoder, device_map=a.device)
        H_dst = next(m for m in model.modules() if hasattr(m, "num_v_heads")).num_v_heads
    st_cache, tile_cache, sizes = {}, {}, {}

    def states(d):
        if d not in st_cache:
            st_cache[d] = harvest_states(model, tok, docs[d], max_tokens=a.max_doc_tokens)
            sizes[d] = {**state_nbytes(st_cache[d]), "n_doc": st_cache[d]["n_doc"]}
        return st_cache[d]

    def tiled(d):
        if d not in tile_cache:
            s = harvest_states(enc, etok, docs[d], max_tokens=a.max_doc_tokens)
            tile_cache[d] = tile_states(s, layer_types(enc), layer_types(model), H_dst)
            del s
        return tile_cache[d]

    tok_cache, body_cache = {}, {}

    def doc_tok(d):
        if d not in tok_cache:
            ids = tok(docs[d], add_special_tokens=False)["input_ids"]
            tok_cache[d] = ids[: a.max_doc_tokens] if a.max_doc_tokens else ids
        return tok_cache[d]

    def body_states(d, ids):
        key = (d, len(ids))
        if key not in body_cache:
            b = harvest_states(model, tok, ids=torch.tensor([ids]))
            full = sorted(b["kv"])
            for li in full[len(full) // 2:]:          # only the early half of the KV is ever used here
                del b["kv"][li]
            b["kv"] = {li: b["kv"][li] for li in full[: len(full) // 2]}
            body_cache[key] = b
            if len(body_cache) > 30:
                body_cache.pop(next(iter(body_cache)))
        return body_cache[key]

    sysp = system_prompt("answer")
    os.makedirs(a.out, exist_ok=True)
    # harvest check on the real reader: continuing from the full cache == in-context prefill of [doc; prompt]
    it0 = items[0]
    st0 = states(_get(it0, "doc_id"))
    q0 = torch.tensor([build_chat(tok, hub, sysp, q_prompt(it0))])
    d0 = tok(docs[_get(it0, "doc_id")], add_special_tokens=False, return_tensors="pt")["input_ids"]
    if a.max_doc_tokens:
        d0 = d0[:, : a.max_doc_tokens]
    with torch.no_grad():
        ref = model(input_ids=torch.cat([d0, q0], 1).to(model.device), use_cache=False,
                    logits_to_keep=q0.shape[1]).logits.float()
    got = continue_logits(model, build_prefix_cache(model, st0, True, True, True), q0).float()
    chk = {"max_abs": float((got - ref).abs().max()), "ref_absmax": float(ref.abs().max()),
           "argmax_agree": float((got.argmax(-1) == ref.argmax(-1)).float().mean())}
    print(f"harvest check (prefix cache vs in-context prefill, {q0.shape[1]} prompt positions): {chk}", flush=True)
    del ref, got

    rows, t0, bytes_ = [], time.time(), {}
    for k, it in enumerate(items):
        did = _get(it, "doc_id")
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong_doc": wrong[did]}
        q_ids = build_chat(tok, hub, sysp, q_prompt(it))
        for c in conds:
            if c == "closed_book":
                cache, ids = build_prefix_cache(model, None), q_ids
            elif "tail" in c:            # reader natively prefills the last T doc tokens on top of a body memory
                import re
                T = int(re.search(r"tail(\d+)", c).group(1))
                wb = c.startswith("wrong:")
                body_kv, body_rec = "kvE" in c, "+rec+" in c
                own = doc_tok(did)
                tail = own[-min(T, len(own) - 1):]
                if not (body_kv or body_rec):
                    cache, fullL, early = build_prefix_cache(model, None), [], set()
                else:
                    bd = wrong[did] if wb else did
                    bt = doc_tok(bd)
                    body = body_states(bd, bt[: len(bt) - min(T, len(bt) - 1)])
                    fullL = [i for i, t in enumerate(layer_types(model)) if t != "linear_attention"]
                    early = set(fullL[: len(fullL) // 2]) if body_kv else set()
                    cache = build_prefix_cache(model, body, bool(early), body_rec, body_rec, kv_layers=early)
                    bytes_.setdefault(c, []).append(spec_bytes(body, early, set(body["recurrent"]) if body_rec else set(),
                                                               set(body["conv"]) if body_rec else set()))
                with PrefixMask(model, [l for l in fullL if l not in early] if early else [], 0):
                    g, n = greedy_continue(model, tok, cache, list(tail) + list(q_ids), a.budget, hub=hub)
                rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n, "tail": len(tail)}
                del cache
                continue
            elif c not in KEEP:          # compact spec (parse_spec)
                sp = parse_spec(c)
                st = states(wrong[did] if sp["wrong"] else did)
                full, lin = sorted(st["kv"]), sorted(st["recurrent"])
                kvl, rl, cl = _sel(sp["kv"], full), _sel(sp["rec"], lin), _sel(sp["cv"], lin)
                cache = build_prefix_cache(model, st, bool(kvl), bool(rl), bool(cl), state_layers=rl, conv_layers=cl)
                bytes_.setdefault(c, []).append(spec_bytes(st, kvl, rl, cl))
                with PrefixMask(model, [l for l in full if l not in kvl] if kvl else [], st["n_doc"]):
                    g, n = greedy_continue(model, tok, cache, q_ids, a.budget, hub=hub)
                rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
                del cache
                continue
            elif c == "text":
                cache, ids = build_prefix_cache(model, None), build_chat(tok, hub, sysp, ctx_prompt(it, docs[did]))
            else:
                src = wrong[did] if c.startswith("wrong") else did
                st = tiled(src) if c in ("tile_state", "wrong_tile") else states(src)
                spec = KEEP[c] + (1.0, None)[len(KEEP[c]) - 3:]
                kv, s, cv, alpha, masked = spec[:5]
                cache, ids = build_prefix_cache(model, st, bool(kv), bool(s), bool(cv), state_scale=alpha), q_ids
            if c in KEEP and len(KEEP[c]) == 5:
                full = sorted(st["kv"])
                half = full[: len(full) // 2] if masked == "early" else full[len(full) // 2:]
                with PrefixMask(model, half, st["n_doc"]):
                    g, n = greedy_continue(model, tok, cache, ids, a.budget, hub=hub)
            else:
                g, n = greedy_continue(model, tok, cache, ids, a.budget, hub=hub)
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            del cache
        rows.append(rec)
        if (k + 1) % 20 == 0:
            print(f"  [{k + 1}/{len(items)}] " + " ".join(
                f"{c}={sum(r[c]['score'] for r in rows) / len(rows):.3f}" for c in conds)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item", flush=True)
        if len(st_cache) > 24:
            st_cache.pop(next(iter(st_cache)))
    summ = {c: {"score": sum(r[c]["score"] for r in rows) / len(rows),
                "gen_tokens": sum(r[c]["gen_tokens"] for r in rows) / len(rows), "n": len(rows)} for c in conds}
    for a_, b_ in (("kv_only", "wrong_kv"), ("state_only", "wrong_state"), ("tile_state", "wrong_tile"),
                   ("all", "closed_book"), ("state_only", "closed_book"), ("kv_only", "closed_book"),
                   ("tile_state", "closed_book"), ("kvE_state", "all"), ("kvL_state", "all"), ("kvE", "kvL"),
                   ("kv_conv", "kv_only"), ("all_a0.5", "all"), ("state_a0.5", "state_only"),
                   ("kvE", "closed_book"), ("kvL", "closed_book")):
        if a_ in summ and b_ in summ:
            d = [r[a_]["score"] - r[b_]["score"] for r in rows]
            m = sum(d) / len(d)
            se = (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d)) ** 0.5
            summ[f"{a_}-{b_}"] = {"gap": m, "se": se}
    if "all" in summ and "text" in summ:
        summ["agree_all_text"] = sum(r["all"]["gen"] == r["text"]["gen"] for r in rows) / len(rows)
    for c in conds:
        if c.startswith("wrong:") and c[6:] in summ:
            d = [r[c[6:]]["score"] - r[c]["score"] for r in rows]
            m = sum(d) / len(d)
            summ[f"{c[6:]} - wrong"] = {"gap": m, "se": (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d)) ** 0.5}
        elif c not in KEEP and c not in ("closed_book", "text", "all") and "all" in summ:
            d = [r[c]["score"] - r["all"]["score"] for r in rows]
            m = sum(d) / len(d)
            summ[f"{c} - all"] = {"gap": m, "se": (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d)) ** 0.5}
    for c in conds:
        if c.startswith("kvE+") and "tail" in c and c.split("+")[-1] in summ:
            b_ = c.split("+")[-1]
            d = [r[c]["score"] - r[b_]["score"] for r in rows]
            m = sum(d) / len(d)
            summ[f"{c} - {b_}"] = {"gap": m, "se": (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d)) ** 0.5}
    for c in conds:
        if c in bytes_:
            summ[c]["mean_bytes"] = sum(bytes_[c]) / len(bytes_[c])
    mean_sz = {k: sum(v[k] for v in sizes.values()) / max(1, len(sizes)) for k in ("recurrent", "conv", "kv", "n_doc")}
    res = {"summary": summ, "harvest_check": chk, "mean_state_bytes": mean_sz, "n": len(rows), "conds": conds,
           "args": vars(a), "items": rows}
    json.dump(res, open(os.path.join(a.out, "eval_state.json"), "w"), indent=1)
    print(f"==== n={len(rows)} ({len(doc_ids)} docs) -> {a.out}/eval_state.json")
    for c in conds:
        print(f"  {c:34s} {summ[c]['score']:.4f}  gen {summ[c]['gen_tokens']:.1f}"
              + (f"  {summ[c]['mean_bytes'] / 2**20:.1f} MiB" if "mean_bytes" in summ[c] else ""))
    for k_, v in summ.items():
        if "-" in k_:
            print(f"  {k_:40s} {v['gap']:+.4f} ± {v['se']:.4f}")
    if "agree_all_text" in summ:
        print(f"  all == text generations: {summ['agree_all_text']:.3f}")
    print(f"  mean bytes: recurrent {mean_sz['recurrent'] / 2**20:.1f} MiB, conv {mean_sz['conv'] / 2**20:.2f} MiB, "
          f"kv {mean_sz['kv'] / 2**20:.1f} MiB, n_doc {mean_sz['n_doc']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
