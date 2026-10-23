"""D12b: the D24 hypernetwork H (xenc/d24_hyper.py, unchanged) on the 27B reader (D12p runtime, xenc/d12p_read.py).

At test time: the 9B encoder prefills the document once -> operator (cached), ONE forward of H(operator) -> the
per-document adapter (dA = U·M(doc)·V on the read query map, gate offset, write-back bias offset) -> the 27B reads
with the FROZEN base interface + adapter. The 27B never reads the document text.

`wire(rt, H)` attaches H to the runtime's interface and wraps the reader's `R.load(op, n_op)`: every time an
operator is loaded for a read, the adapter of THAT operator's document is computed (with grad iff grad is enabled,
i.e. in training) and installed. `override(rt, adapter)` forces a given adapter instead (kv_Hswap: own operator read
through ANOTHER document's adapter). H's heads are zero-initialised, so at step 0 the read equals the base read.
"""
from __future__ import annotations

import contextlib

import torch

from xenc.d24_hyper import attach, set_doc


def wire(rt, H):
    itf = rt.itf
    slots = getattr(H, "kind", "") == "slots"            # D12b H-slots (xenc/d12b_slots.py): per-doc memory slots
    if slots:
        from xenc.d12b_slots import attach_slots
        attach_slots(itf)
        install = lambda ad: setattr(itf, "_slots", ad)     # noqa: E731
    else:
        attach(itf, H)
        install = lambda ad: set_doc(itf, ad)               # noqa: E731
    itf._d12b_override = None
    R = rt.R
    orig = R.load

    def load(op, n_op):
        orig(op, n_op)
        ad = itf._d12b_override
        if ad is None:
            ad = H({k: v for k, v in op.items() if k != "ids"})
        install(None if ad is False else ad)               # override False = the base read (no adapter)

    R.load = load
    rt._d12b_H = H
    return rt


def unwire_doc(rt):
    set_doc(rt.itf, None)
    rt.itf._slots = None


@contextlib.contextmanager
def override(rt, adapter):
    itf = rt.itf
    prev = itf._d12b_override
    itf._d12b_override = adapter
    try:
        yield
    finally:
        itf._d12b_override = prev


@torch.no_grad()
def adapter_of(rt, did):
    """H's adapter for document `did` (no grad)."""
    return rt._d12b_H({k: v for k, v in rt.operator(did)[0].items() if k != "ids"})


def init_stats(rt, H, docs, n_docs: int, log=print):
    """v2 H: standardisation statistics from the operators of up to n_docs TRAINING documents."""
    import random
    sd = sorted(docs)
    random.Random(99).shuffle(sd)
    fs = []
    for d in sd[:n_docs]:
        with torch.no_grad():
            fs.append({e: v.detach() for e, v in H.summary(rt.operator(d)[0]).items()})
        rt._ops.pop(d, None)
    H.init_stats_feats(fs)
    log(f"D12b H: feature statistics initialised on {len(fs)} training documents")


def h_param_groups(H, lr_h: float, lr_basis: float):
    basis = [p for n, p in H.named_parameters() if n.split(".")[0] in ("U", "V", "Ub", "UB", "VB")]
    ids = {id(p) for p in basis}
    rest = [p for p in H.parameters() if id(p) not in ids]
    return [{"params": rest, "lr": lr_h}, {"params": basis, "lr": lr_basis}]


__all__ = ["wire", "override", "adapter_of", "init_stats", "h_param_groups", "unwire_doc"]
