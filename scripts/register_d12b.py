"""D12b: per-document registration (register_d20 / register_d22, UNMODIFIED) on the 27B reader with the encoder
operators read from the D12p phase-A cache (fit_d12p.CachedRt) instead of a live encoder.

Why: the 27B lever-full base (outputs/D12p_lever_full_s2) reads 9B-encoder operators. 27B + 9B + registration
activations do not fit one H100, and the judge rows of that base (outputs/D12p_lvfull_*) were produced from the cached
operators ($XENC_CACHE/D12p/c9b_judge, c9b_deep). Using the same cache makes kv@0 the SAME read as the base judge.

What changes vs register_d20 (in-process only; no shared file is edited):
  * fit_d12p.D13Rt is replaced by a factory returning CachedRt(--enc_cache) with a per-document length check
    (cached operator N == len(prefix) - tail, else hard error: a different --max_tokens / --tail would misalign).
  * --dist_w must be 0: the cache holds no encoder teacher records for the registration self-question rows, so the
    encoder-attention distillation term is unavailable. Registration = KL(27B text teacher || student) + 0.5 CE.
  * the encoder is loaded on --enc_device cpu (needed for the interface dims only; it never runs).
Everything else (leakage assertion, held-out early stop, eval, wrong_doc = another doc's operator AND adapter) is
register_d20's. With --d22_set the D22 item set is judged exactly (register_d22).

    python scripts/register_d12b.py --enc_cache $XENC_CACHE/D12p/c9b_judge --d22_set outputs/D22_sets/qasper.json \
        --wrong_map outputs/D22_sets/qasper_wrongmap.json --runtime d12p --dist_w 0 --enc_device cpu ... --out <dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _pop(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    v = argv[i + 1]
    del argv[i: i + 2]
    return v


def cached_rt_factory(cache_dir: str):
    """-> a drop-in for fit_d12p.D13Rt(reader, enc, tok, hub, itf, tail, max_tokens) that reads cached operators."""
    import fit_d12p
    from register_doc import log

    class CheckedCachedRt(fit_d12p.CachedRt):
        def operator(self, did):
            out = super().operator(did)
            self.checked = getattr(self, "checked", set())
            if did not in self.checked:
                ids = self.pre(did)[: -self.tail] if self.tail else self.pre(did)
                n_op = next(v for k, v in out[0].items() if k != "ids")[0].shape[1]
                assert int(out[1]) == len(ids) == n_op, \
                    f"D12b: cached operator of {did} has N={out[1]} (K cols {n_op}) but the prefix is {len(ids)} tokens"
                self.checked.add(did)
            return out

    def make(reader, enc, tok, hub, itf, tail=64, max_tokens=16000, lru=3):
        log(f"D12b: CachedRt from {cache_dir} (encoder {'never runs' if enc is not None else 'absent'}) | tail {tail} "
            f"max_tokens {max_tokens}")
        return CheckedCachedRt(reader, tok, hub, itf, cache_dir, tail, max_tokens, lru)

    return make


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cache = _pop(argv, "--enc_cache")
    assert cache, "--enc_cache <dir[,dir...]> is required"
    dist = _pop(argv, "--dist_w", "1.0")
    assert float(dist) == 0.0, "D12b: the cache has no encoder records for self-question rows -> pass --dist_w 0"
    argv += ["--dist_w", "0"]
    assert "--runtime" in argv and argv[argv.index("--runtime") + 1] == "d12p", "D12b needs --runtime d12p"
    import fit_d12p
    fit_d12p.D13Rt = cached_rt_factory(cache)            # register_d20.D20 imports it at construction time
    if "--d22_set" in argv:
        import register_d22
        return register_d22.main(argv)
    import register_d20
    return register_d20.main(argv)


if __name__ == "__main__":
    sys.exit(main())
