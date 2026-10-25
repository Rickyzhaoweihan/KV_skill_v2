"""SOFTCONF eval wrapper (agent SOFTCONF, 2026-10-25): scripts/eval_cad.py unchanged, plus two graded-conflict knobs.

    python scripts/softconf_eval.py [--ad_mode jsd|prior|priorjsd|entropy] [--ad_tau .995] [--ad_floor 0]
                                    [--gain_scale 1.25] -- <every scripts/eval_cad.py argument>

  --ad_mode   per-step adaptive CAD (xenc/softcad.py alpha_rule) for every kv/text CAD decode; the eval_cad --cad_alphas
              values then act as the scale `a` (rows keep eval_cad's names kv_cad_<ref>_<a>; the mode is in args.json).
  --gain_scale multiply the trained read gate g (y = own + g·(read − own)) by s after the interface is loaded:
              s > 1 = internal extrapolation of the read (one stream, no reference pass), s < 1 = damping.
Both patches are applied by monkeypatching the attributes eval_cad imports at call time (xenc.cad.cad_decode,
fit_d13.build / fit_d12p.build); with no flag this is exactly eval_cad.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def patch(ad_mode: str = "", ad_tau: float = 0.0, ad_floor: float = 0.0, gain_scale: float = 1.0):
    import xenc.cad as cad
    from xenc.softcad import adaptive_cad_decode, scale_gain
    if ad_mode and ad_mode != "const":
        orig = cad.cad_decode

        def dec(main, ref, alpha, *a, **k):
            if ref is None or alpha == 0:
                return orig(main, ref, alpha, *a, **k)
            return adaptive_cad_decode(main, ref, alpha, *a, mode=ad_mode, tau=ad_tau, floor=ad_floor, **k)
        cad.cad_decode = dec
    if gain_scale != 1.0:
        import fit_d12p
        import fit_d13
        for mod in (fit_d13, fit_d12p):
            ob = mod.build

            @functools.wraps(ob)
            def nb(a, init=None, _ob=ob, **k):
                out = _ob(a, init=init, **k)
                print(f"[softconf_eval] read gate g x {gain_scale}: {scale_gain(out[3], gain_scale)}", flush=True)
                return out
            mod.build = nb


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        i = argv.index("--")
        own, rest = argv[:i], argv[i + 1:]
    else:
        own, rest = [], argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ad_mode", default="", choices=["", "const", "jsd", "prior", "priorjsd", "entropy"])
    ap.add_argument("--ad_tau", type=float, default=0.0)
    ap.add_argument("--ad_floor", type=float, default=0.0)
    ap.add_argument("--gain_scale", type=float, default=1.0)
    a = ap.parse_args(own)
    patch(a.ad_mode, a.ad_tau, a.ad_floor, a.gain_scale)
    import eval_cad
    out = rest[rest.index("--out") + 1] if "--out" in rest else None
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(out, "softconf_args.json"), "w"))
    return eval_cad.main(rest)


if __name__ == "__main__":
    sys.exit(main())
