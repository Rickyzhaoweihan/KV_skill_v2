"""Judge path for the token-identity family (P3 channel / P4 pointer / P5 copy head) on CPU tiny models: the D20
per-document registration step (DocAdapter dA + gate, e2e read, the D20 distillation terms via D13Rt.attn_terms,
backward) and a registered greedy decode run for each interface, starting exactly at the base (zero-init adapter)."""

from __future__ import annotations

import copy
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d13_read import D13Interface, EncTeacher  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}


def _ids(n, seed):
    return torch.randint(12, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _families():
    from test_state_interface import _tiny_encoder
    from xenc.p3_tok import parse_tok, upgrade_tok
    from xenc.p4_ptr import parse_ptr, upgrade_ptr
    from xenc.p5_copy import parse_p5, upgrade_p5
    out = {}
    for fam in ("tok", "ptr", "p5"):
        torch.manual_seed(0)
        r, e = tiny_qwen35(), _tiny_encoder()
        B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
        base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
        if fam == "tok":
            itf = upgrade_tok(base, e, parse_tok("5", 8, "kv", "-1,0,1", 0), r.get_input_embeddings().weight)
            with torch.no_grad():
                itf.TOK["U"]["5"].normal_(0, 0.3)
        elif fam == "ptr":
            itf = upgrade_ptr(base, e, parse_ptr("5", "0,1", 1, [7, 8, 9]))
            with torch.no_grad():
                for n_, p in itf.named_parameters():
                    if n_.startswith("PTR."):
                        p.add_(0.3)
        else:
            itf = upgrade_p5(base, e, parse_p5("2,4", heads=2, dim=8, prefix_ids=[7, 8, 9], gt_id=11, span_max=6, nbkt=6,
                                               ans_id=10))
            itf.set_nosp(torch.arange(100), torch.full((100,), 11))
            with torch.no_grad():
                for n_, p in itf.named_parameters():
                    if n_.startswith("P5."):
                        p.add_(0.2 * torch.randn_like(p))
        out[fam] = (r, e, base, itf)
    return out


def _op_rec(e, itf, doc, rows):
    Tc = EncTeacher(e, sorted(set(LMAP.values())))
    Tc.rec, Tc.kv_only = {}, True
    hk = itf.harvest_hooks(e) if hasattr(itf, "harvest_hooks") else None
    with torch.no_grad():
        out = e(input_ids=torch.tensor([doc]), use_cache=True)
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    if hk is not None:
        op.update(hk.finish(torch.device("cpu")))
    op["ids"] = torch.tensor(doc, dtype=torch.long)
    Tc.rec, Tc.kv_only = {}, False
    with torch.no_grad():
        e(input_ids=torch.tensor([rows]), past_key_values=copy.deepcopy(out.past_key_values), use_cache=True)
    rec = dict(Tc.rec)
    Tc.rec = None
    Tc.detach()
    return op, rec


def test_d20_registration_step_and_registered_decode_for_each_family():
    from fit_d13 import D13Rt
    from xenc.d20_reg import DocAdapter
    for fam, (r, e, base, itf) in _families().items():
        doc, rows = [7, 8, 9] + _ids(22, 0), _ids(6, 1) + [10, 11] + _ids(3, 2)
        op, rec = _op_rec(e, itf, doc, rows)
        R = itf.make_reader(r)
        R.load(op, len(doc))
        R.mode = "student"
        with torch.no_grad():
            ref = r(input_ids=torch.tensor([rows])).logits.float()
        R.mode, R.past = "off", {}
        ad = DocAdapter(itf, ["dA", "gate"], rank=4)
        R.load(ad.transform_op(op), len(doc))
        R.mode = "student"
        with torch.no_grad():
            z0 = r(input_ids=torch.tensor([rows])).logits.float()
        R.mode, R.past = "off", {}
        assert torch.allclose(ref, z0, atol=1e-5), (fam, float((ref - z0).abs().max()))      # step 0 == base
        rt = types.SimpleNamespace(itf=itf, R=R, ckpt=False)
        B0 = {li: itf.B[str(li)].detach().clone() for li in LMAP}
        yref = {li: rec[e_]["out"].float() @ B0[li].t() for li, e_ in LMAP.items()}
        R.load(ad.transform_op(op), len(doc))
        R.mode, R.need_logits, R.rec = "e2e", True, {}
        with torch.enable_grad():
            lo = r(input_ids=torch.tensor([rows])).logits.float()
            terms = D13Rt.attn_terms(rt, rec, op, len(doc), yref)
            loss = lo.logsumexp(-1).mean() + torch.stack([t["kl"] + t["rel"] for t in terms.values()]).mean()
            loss.backward()
        R.mode, R.rec = "off", {}
        g = [p.grad for k, p in ad._named.items() if k.startswith("D.Va") or k.startswith("D.Ua")]
        assert any(x is not None and float(x.abs().sum()) > 0 for x in g), fam
        with torch.no_grad():
            for p in ad._named.values():
                if p.grad is not None:
                    p.add_(-0.5 * p.grad.sign())
        R.load(ad.transform_op(op), len(doc))
        R.mode = "student"
        with torch.no_grad():
            o = r(input_ids=torch.tensor([rows]), use_cache=True)
            for _ in range(3):
                nxt = o.logits[:, -1:].argmax(-1)
                o = r(input_ids=nxt, past_key_values=o.past_key_values, use_cache=True)
        R.mode, R.past = "off", {}
        assert torch.isfinite(o.logits).all(), fam
        R.detach()


if __name__ == "__main__":
    test_d20_registration_step_and_registered_decode_for_each_family()
    print("ALL P3 JUDGE TESTS PASSED")
