"""D21 (b, operator): D20's per-document registration + judge with a COMPRESSED stored operator.

Runs scripts/register_d20.py unchanged, after patching fit_d13.D13Rt.operator so that every operator the read sees
(registration steps, held-out self-question loss, kv and wrong_doc generation) is the compressed one
(scripts/d21_opcompress.py `transform`: kv8 / kv4g<G> / acc<p> / rand<p>, '+'-joined). The encoder's own teacher
pass (distillation targets) still continues from its uncompressed prefix cache, as in D20; only the stored object
the reader reads is compressed. acc<p> importance = attention received in the encoder's own document prefill.
Extra flag: --d21_op <spec> (required); all other flags go to register_d20.py.

    python scripts/d21_reg_op.py --d21_op kv4g128 --interface outputs/D13_4b_reg_w10/d13_best.pt ... --out outputs/D21_regop_lh_kv4g128
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    k = argv.index("--d21_op")
    spec = argv[k + 1]
    del argv[k: k + 2]
    import fit_d13
    import register_d20
    from d21_opcompress import parse, transform
    p = parse(spec)
    orig = fit_d13.D13Rt.operator
    done: dict = {}

    @torch.no_grad()
    def operator(self, did):
        op, n_op, cache = orig(self, did)
        key = (id(self), did)
        if key not in done:
            imp = None
            if p["score"] == "acc":
                from deltaskill.backbone_utils import decoder_layers
                from xenc.kv_prune import received_attention
                ids = self.pre(did)[: -self.tail] if self.tail else self.pre(did)
                self.Tc.rec, self.Tc.kv_only = {}, False
                self.enc(input_ids=torch.tensor([ids], device=self.edev), use_cache=False, logits_to_keep=1)
                rec, self.Tc.rec = self.Tc.rec, None
                L = decoder_layers(self.enc)
                imp = {e: received_attention(r["q"].unsqueeze(0), r["k"].unsqueeze(0), L[e].self_attn.scaling).to(self.dev)
                       for e, r in rec.items()}
                del rec
            opc, nb = transform(op, imp, spec)
            while len(done) >= 4:            # small LRU of compressed operators (kv doc + wrong doc alternate)
                done.pop(next(iter(done)))
            done[key] = (opc, nb)
        return done[key][0], n_op, cache

    fit_d13.D13Rt.operator = operator
    print(f"[D21 reg-op] stored operator compressed as {spec}", flush=True)
    return register_d20.main(argv)


if __name__ == "__main__":
    sys.exit(main())
