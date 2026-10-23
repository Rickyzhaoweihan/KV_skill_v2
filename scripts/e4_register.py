"""E4: run D20/D22 per-document registration + judge with a READ VARIANT (xenc/e4_read.py) installed in-process.
Everything else (registration, leakage assertion, eval, selection, summaries) is register_d20.py / register_d22.py
unmodified. The variant applies to every D13 read: registration forwards, held-out self-question loss and judge
generation. The stage-1 distillation term sees the UN-modified logits (--e4_raw_logits 1, default), i.e. it keeps
matching the query direction to the encoder teacher's attention at the teacher's own temperature.

    python scripts/e4_register.py --e4_variant temp1.5 --e4_entry d22 -- <register_d22.py args>
    python scripts/e4_register.py --e4_variant base    --e4_entry d22 -- <same args>   # flag-off reproduction
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import xenc  # noqa: E402,F401
import xenc.d13_read as D13  # noqa: E402
from xenc import e4_read  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4_variant", required=True, help="base | tempX | topK, optional @l+l (reader layers)")
    ap.add_argument("--e4_entry", default="d22", choices=("d20", "d22"))
    ap.add_argument("--e4_raw_logits", type=int, default=1)
    ap.add_argument("--e4_link_adapters", default="", help="eval-only: symlink <out>/adapters -> this directory")
    e, rest = ap.parse_args(argv[:k]), argv[k + 1:]
    e4_read.install()
    e4_read.set_from(e.e4_variant, raw_logits=bool(e.e4_raw_logits))
    assert D13._attend is e4_read.attend_variant and D13.D13Interface.read is e4_read.read_with_layer
    out = rest[rest.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    if e.e4_link_adapters:
        phase = rest[rest.index("--phase") + 1] if "--phase" in rest else "register,eval"
        assert "register" not in phase, "--e4_link_adapters is for --phase eval only"
        dst = os.path.join(out, "adapters")
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(e.e4_link_adapters), dst)
        assert os.path.realpath(dst) == os.path.realpath(e.e4_link_adapters), f"{dst} points elsewhere"
    json.dump(vars(e), open(os.path.join(out, "e4_variant.json"), "w"), indent=1)
    print(f"[E4] read variant installed: {e4_read.Variant.describe()} | entry {e.e4_entry} | adapters "
          f"{e.e4_link_adapters or '(registered here)'}", flush=True)
    if e.e4_entry == "d22":
        import register_d22
        return register_d22.main(rest)
    import register_d20
    return register_d20.main(rest)


if __name__ == "__main__":
    sys.exit(main())
