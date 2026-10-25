"""CMP hybrid, cross-model fit in two SEQUENTIAL phases (one GPU holds one model at a time; agent CMP, 2026-10-25).

For the 27B reader: source = the 9B ENCODER's pre-RoPE K/V at its deep layers (default 23/27/31), target = the 27B
READER's own pre-RoPE K/V at its deepest full-attention layers (default 51/55/59/63), on the SAME tokens of TRAINING
documents (token-aligned: one tokenizer; the doc prefix is the 27B runtime's, scripts/fit_d12p.D13Rt.pre).
  phase 1: the encoder runs over every selected document; the features of up to --tok_per_doc sampled tokens are
           kept on the CPU (bf16); the encoder is freed;
  phase 2: the reader runs over the same documents; its features at the SAME token indices are joined and the joint
           moments accumulated (xenc/cmp_op.OpStats, reader layer l under key 100 + l);
  then one compressor file per spec ('e..+1ll..|Le+..:d': the encoder group is the stored code, the reader layers are
  decoded by closed-form ridge), same format as scripts/cmp_fit.py.
Documents: the training mix of --args_from (a register_d12p / register_d13 args.json) after its --exclude_sets /
--exclude_docs / length windows (asserted disjoint from every excluded document).

    python scripts/cmp_fit_xm.py --out_dir outputs/CMP_fit27 --specs '23+27+31+151+155+159+163|L23+L27+L31:2048'
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


def select_docs(args_from: str, docs_per_task: int, seed: int):
    import fit_d13 as F
    from register_d13 import doc_len_filter, excluded_docs
    S = json.load(open(args_from))
    import fit_d12p as F12                                # 27B split registry (conflict splits d25*) not in fit_d13
    for k_, v_ in F12.TASK_SPLIT.items():
        F.TASK_SPLIT.setdefault(k_, v_)
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
    rng = random.Random(seed)
    pick = []
    for t in sorted(by):
        ds = sorted(by[t])
        rng.shuffle(ds)
        pick += ds[:docs_per_task]
    assert not set(pick) & drop, "an excluded (judged / leakage) document reached the CMP fit"
    log(f"CMP xm fit: {len(pick)} TRAIN docs from {len(by)} tasks, excluded {len(drop)} docs ({args_from})")
    return pick, docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--reader", default="Qwen/Qwen3.5-27B", help="target model AND tokenizer / chat template of the prefix")
    ap.add_argument("--enc_layers", default="23,27,31")
    ap.add_argument("--tgt_layers", default="51,55,59,63")
    ap.add_argument("--args_from", default="outputs/D12b_max13k_s2/args.json")
    ap.add_argument("--docs_per_task", type=int, default=8)
    ap.add_argument("--tok_per_doc", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--specs", default="23+27+31+151+155+159+163|L23+L27+L31:512,23+27+31+151+155+159+163|L23+L27+L31:2048")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stats_path", default="", help="where the moments go (default <out_dir>/stats.pt)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    from xenc.cmp_hyb import RKEY
    from xenc.cmp_op import OpCompressor, OpStats, rope_tables, save_cmp, token_features
    os.makedirs(a.out_dir, exist_ok=True)
    el = [int(x) for x in a.enc_layers.split(",")]
    tl = [int(x) for x in a.tgt_layers.split(",")]
    stats_path = a.stats_path or os.path.join(a.out_dir, "stats.pt")
    if not os.path.exists(stats_path):
        pick, docs = select_docs(a.args_from, a.docs_per_task, a.seed)
        if a.dry_run:
            return 0
        from deltaskill.backbone_utils import decoder_layers
        from transformers import AutoTokenizer
        from xenc.backbone import load_encoder
        from xenc.d13_read import EncTeacher
        from xenc.state_skill import doc_prefix_text
        from xenc.xattn_read import _rotary
        tok = AutoTokenizer.from_pretrained(a.reader)
        ids_of, idx_of = {}, {}
        for i, did in enumerate(pick):
            ids = [int(x) for x in tok(doc_prefix_text(tok, a.reader, docs[did], SYSTEM_ANSWER),
                                       add_special_tokens=False)["input_ids"]]
            ids = ids[-(a.max_tokens - 1024):][: -a.tail]
            ids_of[did] = ids
            g = torch.Generator().manual_seed(i)
            idx_of[did] = torch.randperm(len(ids), generator=g)[: a.tok_per_doc].sort().values
        feats: dict = {}
        Fdim = None
        for phase, (name, layers) in enumerate(((a.encoder, el), (a.reader, tl))):
            m = load_encoder(name, device_map=a.device)[0]
            att = decoder_layers(m)[layers[0]].self_attn
            Hkv, hd = att.config.num_key_value_heads, att.head_dim
            assert Fdim in (None, Hkv * hd), f"{name}: K/V width {Hkv * hd} != {Fdim}"
            Fdim = Hkv * hd
            rot, Tc, t0 = _rotary(m), EncTeacher(m, layers), time.time()
            for i, did in enumerate(pick):
                ids = ids_of[did]
                Tc.rec, Tc.kv_only = {}, True
                with torch.no_grad():
                    m(input_ids=torch.tensor([ids], device=a.device), use_cache=False, logits_to_keep=1)
                op = {e: (r["k"], r["v"]) for e, r in Tc.rec.items()}
                Tc.rec = None
                cos, sin = rope_tables(rot, len(ids), a.device)
                X = token_features(op, layers, cos, sin, idx_of[did].to(a.device)).to(torch.bfloat16).cpu()
                feats.setdefault(did, []).append(X)
                del op
                if i % 20 == 0:
                    log(f"  phase {phase + 1} ({name}) [{i + 1}/{len(pick)}] {did} {len(ids)} tok | {time.time() - t0:.0f}s")
            Tc.detach()
            del m, Tc
            torch.cuda.empty_cache()
        st = OpStats(el + [RKEY + x for x in tl], Fdim, device=a.device)
        for did in pick:
            st.add_features(torch.cat(feats[did], 1))
            st.docs.append(did)
        blob = st.state()
        blob.update(Hkv=Hkv, hd=hd, encoder=a.encoder, reader=a.reader, args_from=a.args_from, enc_layers=el,
                    tgt_layers=tl, max_tokens=a.max_tokens)
        torch.save(blob, stats_path)
        log(f"stats -> {stats_path}: {st.n} tokens from {len(pick)} docs, D = {blob['sum'].numel()}")
    st = torch.load(stats_path, map_location="cpu", weights_only=False)
    rep = {}
    for spec in [s for s in a.specs.split(",") if s]:
        path = os.path.join(a.out_dir, f"cmp_{spec.replace(':', '_').replace('+', '-').replace('|', '__')}.pt")
        if os.path.exists(path):
            log(f"  {path} exists")
            continue
        m = OpCompressor.from_stats(st, st["Hkv"], st["hd"], spec, device=a.device)
        save_cmp(m, path, {"stats": stats_path})
        rep[spec] = {"path": path, "bytes_per_token_bf16": m.bytes_per_token("bf16"), **m.info}
        log(f"  {spec}: var kept {[round(v, 4) for v in m.info['var_kept']]} cross_r2 {m.info.get('cross_r2')} "
            f"{m.bytes_per_token('bf16'):.0f} B/token -> {path}")
    with open(os.path.join(a.out_dir, "fit_report.json"), "a") as f:
        f.write(json.dumps(rep) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
