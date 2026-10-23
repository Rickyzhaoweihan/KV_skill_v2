"""C2 hook check on the real model (GPU): with ONE shared instance as encoder and reader (--share_encoder), the D13Rt
operator equals the plain prefill K/V of that model, the reader in mode "off" equals the stock forward, and the reader's
read passes record nothing in the encoder teacher. Exits non-zero on any mismatch.

    python scripts/d13_c2_check.py --reader Qwen/Qwen3.5-9B --split_dir data/quality_split
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401


def main(argv=None) -> int:
    from fit_d13 import D13Rt, build
    from register_doc import _get, load_split_items
    from xenc.doc_tasks import load_docs
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--split_dir", default="data/quality_split")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    a.encoder, a.B, a.share_encoder = a.reader, "identity", 1
    reader, enc, tok, itf, lmap = build(a, init=None, mlp_hidden=0)
    assert enc is reader
    with torch.no_grad():
        for li in itf.layers:                      # make the read non-trivial (identity A would reproduce attention)
            itf.A[str(li)].weight.mul_(0.9)
    rt = D13Rt(reader, enc, tok, a.reader, itf, 64, 16000)
    docs = load_docs(a.split_dir)
    it = load_split_items(a.split_dir, "val")[0]
    did = _get(it, "doc_id")
    rt.docs = {did: docs[did]}
    rows = rt.rows(did, it)
    with torch.no_grad():
        stock = reader(input_ids=torch.tensor([rows], device=rt.dev)).logits.float()
    op, n, _ = rt.operator(did)
    ids = rt.pre(did)[:-64]
    with torch.no_grad():
        pkv = reader(input_ids=torch.tensor([ids], device=rt.dev), use_cache=True, logits_to_keep=1).past_key_values
    bad = 0
    for e, (K, V) in op.items():
        Kc = pkv.layers[e].keys[0] if hasattr(pkv, "layers") else pkv.key_cache[e][0]
        Vc = pkv.layers[e].values[0] if hasattr(pkv, "layers") else pkv.value_cache[e][0]
        dk, dv = float((K.float() - Kc.float()).abs().max()), float((V.float() - Vc.float()).abs().max())
        print(f"layer {e}: operator {tuple(K.shape)} n={n} | max|K - prefill K| {dk:.2e} max|V - prefill V| {dv:.2e}")
        bad += (dk > 1e-2) + (dv > 1e-2)
    with torch.no_grad():
        off = reader(input_ids=torch.tensor([rows], device=rt.dev)).logits.float()
    d_off = float((off - stock).abs().max())
    print(f"reader mode off vs stock logits: max diff {d_off:.2e}")
    bad += d_off > 1e-3
    # the reader never sees the document tokens: record every forward of the (shared) model during a read generation
    calls = []

    def _pre(mod, args, kwargs):
        ids = kwargs.get("input_ids")
        pkv = kwargs.get("past_key_values")
        calls.append((int(ids.shape[1]) if ids is not None else -1, int(pkv.get_seq_length()) if pkv is not None else 0))
    h = reader.register_forward_pre_hook(_pre, with_kwargs=True)
    try:
        g, ntok = rt.gen(did, it, 8)
    finally:
        h.remove()
    print(f"read gen ok ({ntok} tokens): {g!r}; teacher records after read pass: {rt.Tc.rec}")
    print(f"reader forwards during the read: first {calls[:2]} ... last {calls[-1:]} | doc prefix {len(rt.pre(did))} tokens, "
          f"reader prompt {len(rows)} tokens")
    ok_first = calls and calls[0] == (len(rows), 0)                    # fresh cache, prompt = tail + question only
    ok_rest = all(n == 1 and p == len(rows) + i for i, (n, p) in enumerate(calls[1:]))
    ok_max = max(n + p for n, p in calls) <= len(rows) + ntok
    print(f"reader sees only tail+question+generated tokens: first {ok_first} incremental {ok_rest} max-context {ok_max}")
    bad += (not ok_first) + (not ok_rest) + (not ok_max)
    bad += rt.Tc.rec is not None
    print("C2 HOOK CHECK", "PASSED" if not bad else f"FAILED ({bad})")
    return int(bool(bad))


if __name__ == "__main__":
    sys.exit(main())
