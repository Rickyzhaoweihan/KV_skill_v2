"""CMP stage 0: operator statistics on TRAINING documents -> closed-form compressor files (xenc/cmp_op.py).

The 4B encoder prefills a stratified sample of training documents of the DS / max8k mix (same split registry, the
same --exclude_sets / --exclude_docs / --train_doc_len exclusions as register_d13; asserted), exactly as
D13Rt.operator does (doc prefix, last --tail tokens left out); the pre-RoPE K/V features of up to --tok_per_doc
random tokens per document are accumulated (fp64 moments, all read layers jointly). Then one compressor file per spec
(PCA encoders, closed-form decoders) is written.

    python scripts/cmp_fit.py --out_dir outputs/CMP_fit --specs all:128,all:512,per:256,L7:128 --device cuda:0
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
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402

DS_TRAIN = ("quality:150,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,"
            "musiquetr:0,narrativeqa:0,racemc:0,multirc:0,nqamc:0,looglemc:0")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B", help="tokenizer / chat template of the doc prefix (as D13Rt)")
    ap.add_argument("--enc_layers", default="3,7,11,15,19,23,27,31", help="layers whose moments are accumulated")
    ap.add_argument("--base_layers", default="3,7,11,15,19", help="a spec without an 'e+e|' prefix is restricted to these "
                    "(the read layers of the base interface)")
    ap.add_argument("--args_from", default="outputs/DS_max8k/args.json", help="train / exclusion specs (register_d13 args)")
    ap.add_argument("--docs_per_task", type=int, default=8)
    ap.add_argument("--tok_per_doc", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--specs", default="all:128,all:256,all:512,all:1024,all:2048,all:4096,per:256,per:512,L7:128,L7:1024")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hyb_reader_layers", default="", help="CMP hybrid: also run the READER over the same document tokens "
                    "and accumulate its pre-RoPE K/V at these layers (stats keys 100+layer), e.g. 23,27,31")
    ap.add_argument("--reader_device", default="cuda:1")
    ap.add_argument("--dry_run", action="store_true", help="only select / check the documents")
    ap.add_argument("--stats", default="", help="reuse an existing stats file (skip the encoder passes)")
    a = ap.parse_args(argv)
    from xenc.cmp_op import OpCompressor, OpStats, save_cmp
    os.makedirs(a.out_dir, exist_ok=True)
    layers = [int(x) for x in a.enc_layers.split(",")]
    stats_path = a.stats or os.path.join(a.out_dir, "stats.pt")
    if not os.path.exists(stats_path):
        import fit_d13 as F
        from register_d13 import doc_len_filter, excluded_docs
        from deltaskill.backbone_utils import decoder_layers
        from transformers import AutoTokenizer
        from xenc.backbone import load_encoder
        from xenc.d13_read import EncTeacher
        from xenc.state_skill import doc_prefix_text
        from xenc.xattn_read import _rotary
        S = json.load(open(a.args_from))
        for s_ in [x for x in S.get("extra_splits", "").split(",") if x]:
            n_, d_ = s_.split("=")
            F.TASK_SPLIT[n_] = d_
        tr_folds = set(range(5)) - {S.get("test_fold", 0), S.get("val_fold", 1)}
        train, docs = F.select_items(S["train"], "train", tr_folds, 0, S.get("seed", 0))
        drop = excluded_docs(S.get("exclude_sets", ""))
        if S.get("exclude_docs"):
            X = json.load(open(S["exclude_docs"]))
            drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
        keep_len = doc_len_filter(S.get("train_doc_len") or S.get("doc_len", ""))
        by: dict = {}
        for it in train:
            did = _get(it, "doc_id")
            if did not in drop and keep_len(did):
                by.setdefault(did.split("_")[0], set()).add(did)
        rng = random.Random(a.seed)
        pick = []
        for t in sorted(by):
            ds = sorted(by[t])
            rng.shuffle(ds)
            pick += ds[: a.docs_per_task]
        assert not set(pick) & drop, "an excluded (judged / leakage) document reached the CMP fit"
        log(f"CMP fit: {len(pick)} TRAIN docs from {len(by)} tasks ({ {t: min(len(v), a.docs_per_task) for t, v in by.items()} }), "
            f"excluded {len(drop)} docs; specs from {a.args_from}")
        if a.dry_run:
            return 0
        tok = AutoTokenizer.from_pretrained(a.reader)
        enc = load_encoder(a.encoder, device_map=a.device)[0]
        rot = _rotary(enc)
        att = decoder_layers(enc)[layers[0]].self_attn
        Hkv, hd = att.config.num_key_value_heads, att.head_dim
        Tc = EncTeacher(enc, layers)
        rl = [int(x) for x in a.hyb_reader_layers.split(",") if x]
        Tr = None
        if rl:
            from xenc.backbone import load_reader
            from xenc.cmp_hyb import RKEY
            reader = load_reader(a.reader, device_map=a.reader_device)[0]
            ra = decoder_layers(reader)[rl[0]].self_attn
            assert (ra.config.num_key_value_heads, ra.head_dim) == (Hkv, hd), "reader / encoder K/V shapes differ"
            Tr = EncTeacher(reader, rl)
        st = OpStats(layers + ([RKEY + x for x in rl] if rl else []), Hkv * hd, device=a.device)
        t0 = time.time()
        for i, did in enumerate(pick):
            ids = [int(x) for x in tok(doc_prefix_text(tok, a.reader, docs[did], SYSTEM_ANSWER),
                                       add_special_tokens=False)["input_ids"]]
            ids = ids[-(a.max_tokens - 1024):][: -a.tail]
            Tc.rec, Tc.kv_only = {}, True
            with torch.no_grad():
                enc(input_ids=torch.tensor([ids], device=a.device), use_cache=False, logits_to_keep=1)
            op = {e: (r["k"], r["v"]) for e, r in Tc.rec.items()}
            Tc.rec = None
            if Tr is not None:                         # same tokens through the reader (token-aligned: same tokenizer)
                Tr.rec, Tr.kv_only = {}, True
                with torch.no_grad():
                    reader(input_ids=torch.tensor([ids], device=a.reader_device), use_cache=False, logits_to_keep=1)
                op.update({RKEY + li: (r["k"].to(a.device), r["v"].to(a.device)) for li, r in Tr.rec.items()})
                Tr.rec = None
            st.add(op, rot, a.tok_per_doc, seed=i, doc_id=did)
            del op
            if i % 10 == 0:
                log(f"  [{i + 1}/{len(pick)}] {did} {len(ids)} tok | n {st.n} | {time.time() - t0:.0f}s")
        Tc.detach()
        blob = st.state()
        blob.update(Hkv=Hkv, hd=hd, encoder=a.encoder, reader=a.reader, args_from=a.args_from)
        torch.save(blob, stats_path)
        log(f"stats -> {stats_path}: {st.n} tokens from {len(pick)} docs, D = {blob['sum'].numel()}")
        del enc
        torch.cuda.empty_cache()
    st = torch.load(stats_path, map_location="cpu", weights_only=False)
    assert [int(e) for e in st["layers"]][: len(layers)] == layers, (st["layers"], layers)
    base_pre = "+".join(a.base_layers.split(",")) + "|"
    rep = {}
    for spec in [s for s in a.specs.split(",") if s]:
        path = os.path.join(a.out_dir, f"cmp_{spec.replace(':', '_').replace('+', '-').replace('|', '__')}.pt")
        full = spec if "|" in spec or spec.startswith(base_pre) else base_pre + spec
        if os.path.exists(path):
            log(f"  {path} exists")
            continue
        m = OpCompressor.from_stats(st, st["Hkv"], st["hd"], full, device=a.device)
        save_cmp(m, path, {"stats": stats_path})
        rep[spec] = {"path": path, "bytes_per_token_bf16": m.bytes_per_token("bf16"), **m.info}
        log(f"  {spec}: var kept {[round(v, 4) for v in m.info['var_kept']]} cross_r2 {m.info.get('cross_r2')} "
            f"{m.bytes_per_token('bf16'):.0f} B/token -> {path}")
    with open(os.path.join(a.out_dir, "fit_report.json"), "a") as f:
        f.write(json.dumps(rep) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
