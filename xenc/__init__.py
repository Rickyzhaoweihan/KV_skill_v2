"""xenc — small-encoder / large-reader KV-Skill playground.

Imports `deltaskill` from the sibling delta-skill checkout and the vendored fla kernels; never
modifies either. See docs/XENC_SPEC.md for the module contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PARENT = _ROOT.parent
for _p in (_ROOT, _PARENT / "delta-skill", _PARENT / "fla_kernels"):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

# HF cache hygiene: ~/.bashrc points at a dead scratch cache; the turbo cache is authoritative.
_HF_HOME = "/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home"
if os.path.isdir(_HF_HOME):
    os.environ.setdefault("HF_HOME", _HF_HOME)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_HOME, "hub"))

DEPTHS = [0.25, 0.5, 0.75, 1.0]
CAND = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_CACHE = os.environ.get("XENC_CACHE", "/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache")

__all__ = ["DEPTHS", "CAND", "DEFAULT_CACHE"]
