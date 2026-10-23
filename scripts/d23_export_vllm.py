"""D23 (vLLM): export everything the vLLM plugin needs from the HF stack (run in envs/qwen35, 1 GPU).

  <out>/itf_<tag>.pt   the D13 / D12p interface: full state_dict (A, M, B, Bb, g AND the encoder's frozen attention
                       copies att.*), layer map, dims and the encoder text config (the vLLM side rebuilds the identical
                       transformers modules from it, so the read math is the same code as xenc/d23_fastread.py)
  <out>/ops/<doc>.pt   the operator of each document: truncated 4B harvest (layers <= the deepest read layer of either
                       reader: 3,7,11,15,19), post-RoPE K/V bf16 [Hkv,N,hd], n_op
  <out>/adapters/<doc>.pt  registered D20 adapters (snaps[200]) of the validation documents
  <out>/prompts.json   EXACT reader token ids from the HF tokenizer / chat template: per benchmark document the text
                       prompt (pre + suf), the prefix (pre), kv rows (pre[-64:] + suf) and BM25 RAG x4 / x8 prompts; per
                       validation item the kv rows, text prompt and the judged D20 kv@200 generation

    python scripts/d23_export_vllm.py --out /scratch/.../xenc_cache/D23_vllm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402

ITFS = {
    "9b": dict(reader="Qwen/Qwen3.5-9B", B="outputs/D11_maps/encB5_9b_4b_post_xf01.pt", layers="3,7,11,15,19",
               interface="outputs/D13_4b_reg_w10/d13_best.pt", runtime="d13"),
    "27b": dict(reader="Qwen/Qwen3.5-27B", B="outputs/D10_maps/encB8_27b_4b_post.pt",
                layers="3,7,11,15,19,23,27,31", interface="outputs/D12p_4b_s2/d13_best.pt", runtime="d12p"),
}


def _plain(it):
    """DocItem (dataclass) or dict -> a JSON-serializable dict (xenc.doc_tasks.score accepts either)."""
    import dataclasses
    return dataclasses.asdict(it) if dataclasses.is_dataclass(it) else dict(it)


def lmap_for(reader_hub, enc, layers):
    from transformers import AutoConfig
    from xenc.xattn_read import attn_dims, full_depth_map
    rc = types.SimpleNamespace(config=AutoConfig.from_pretrained(reader_hub))
    keep = {int(x) for x in layers.split(",")}
    return {k: v for k, v in full_depth_map(attn_dims(rc), attn_dims(enc)).items() if k in keep}


def export_itf(tag, spec, enc, out):
    lmap = lmap_for(spec["reader"], enc, spec["layers"])
    if spec["runtime"] == "d12p":
        from fit_d12p import load_B_any
        from xenc.d12p_read import load_d13
        B, _ = load_B_any(spec["B"], sorted(lmap), lmap)
    else:
        from xenc.d13_read import load_B, load_d13
        B = load_B(spec["B"], sorted(lmap))
    itf, _ = load_d13(spec["interface"], enc, B)
    assert itf.layer_map == lmap, (itf.layer_map, lmap)
    from xenc.state_skill import text_config
    blob = {"format": "d23_vllm_itf.v1", "tag": tag, "reader": spec["reader"], "runtime": spec["runtime"],
            "layer_map": itf.layer_map, "d_r": itf.d_r, "d_e": itf.d_e, "mlp_hidden": itf.mlp_hidden,
            "identity_B": itf.identity_B, "enc_text_config": text_config(enc).to_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in itf.state_dict().items()}}
    torch.save(blob, os.path.join(out, f"itf_{tag}.pt"))
    log(f"itf {tag}: layers {itf.layer_map}, {sum(v.numel() for v in blob['state_dict'].values())} params")
    return itf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--docs", default="2k:quality_380c2df08912,16k:loogle_b9faf4fc6659,64k:loogle_78d3ed194af9")
    ap.add_argument("--val_eval", default="outputs/D20_4b_lh_f0/eval_test.json")
    ap.add_argument("--val_adapters", default="outputs/D20_4b_lh_f0/adapters")
    ap.add_argument("--val_docs", type=int, default=4)
    ap.add_argument("--val_q", type=int, default=5)
    ap.add_argument("--enc_layers", default="3,7,11,15,19")
    ap.add_argument("--max_tokens", type=int, default=72000)
    ap.add_argument("--n_sufs", type=int, default=12)
    ap.add_argument("--device", default="cuda", help="cpu: local check of the export path (tiny doc set)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from bench_d23 import Bench, items_for, rag_prompt, task_of
    from xenc.backbone import load_encoder
    from xenc.d13_read import EncTeacher
    from xenc.doc_tasks import load_docs
    from xenc.state_skill import doc_prefix_text, question_suffix_ids
    os.makedirs(os.path.join(a.out, "ops"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "adapters"), exist_ok=True)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    for tag, spec in ITFS.items():
        export_itf(tag, spec, enc, a.out)
    hub = "Qwen/Qwen3.5-9B"                                  # 9B and 27B share the tokenizer / chat template
    tok = AutoTokenizer.from_pretrained(hub)
    tok27 = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    docs = {}

    def text(did):
        if did not in docs:
            docs.update(load_docs(f"data/{task_of(did)}_split"))
        return docs[did]

    def pre_ids(did):
        ids = [int(x) for x in tok(doc_prefix_text(tok, hub, text(did), SYSTEM_ANSWER), add_special_tokens=False)["input_ids"]]
        return ids[-(a.max_tokens - 1024):]

    Tc = EncTeacher(enc, [int(x) for x in a.enc_layers.split(",")])

    def harvest(did, pre):
        fake = types.SimpleNamespace(rt=types.SimpleNamespace(pre=lambda d: pre, Tc=Tc), enc=enc)
        with torch.no_grad():
            op, n = Bench.harvest(fake, did, trunc=True)
        torch.save({"op": {e: (K.cpu(), V.cpu()) for e, (K, V) in op.items()}, "n_op": n, "doc": did},
                   os.path.join(a.out, "ops", f"{did}.pt"))
        return n

    prompts = {"bench": [], "val": [], "tokenizer": hub,
               "same_template_27b": None}
    # RAG prompts reuse bench_d23.rag_prompt (needs a Bench-like object with tok / doc_text / a.reader)
    fakeB = types.SimpleNamespace(tok=tok, doc_text=text, a=types.SimpleNamespace(reader=hub))
    rag_cache = {}
    for spec in a.docs.split(","):
        tag, did = spec.split(":")
        its = sorted(items_for(did), key=lambda i: len(question_suffix_ids(tok, hub, i, SYSTEM_ANSWER)))
        it = its[len(its) // 2]
        pre = pre_ids(did)
        suf = question_suffix_ids(tok, hub, it, SYSTEM_ANSWER)
        n = harvest(did, pre)
        r4 = rag_prompt(fakeB, did, it, 512, 4, rag_cache)[0]
        r8 = rag_prompt(fakeB, did, it, 512, 8, rag_cache)[0]
        # distinct real question suffixes of this document (repeats must not hit a cached FULL prompt under APC)
        sufs = [question_suffix_ids(tok, hub, i, SYSTEM_ANSWER) for i in its[: a.n_sufs]]
        rag8s = [rag_prompt(fakeB, did, i, 512, 8, rag_cache)[0] for i in its[: a.n_sufs]]
        prompts["bench"].append({"tag": tag, "doc": did, "N": len(pre), "n_op": n, "pre": pre, "suf": suf,
                                 "rows": pre[-64:] + suf, "rag4": r4, "rag8": r8, "key": _get(it, "key"),
                                 "sufs": sufs, "rag8s": rag8s})
        log(f"bench {tag} {did}: N {len(pre)} q {len(suf)} rag8 {len(r8)}")
    # 27B tokenizer / template identity check on one prompt
    p0 = prompts["bench"][0]
    prompts["same_template_27b"] = (
        [int(x) for x in tok27(doc_prefix_text(tok27, "Qwen/Qwen3.5-27B", text(p0["doc"]), SYSTEM_ANSWER),
                               add_special_tokens=False)["input_ids"]][-(a.max_tokens - 1024):] == p0["pre"])
    judged = {r["key"]: r for r in json.load(open(a.val_eval))["items"]}
    items = [i for i in load_split_items("data/longhealth_split", "test") if (_get(i, "meta") or {}).get("fold") == 0]
    by = {}
    for i in items:
        by.setdefault(_get(i, "doc_id"), []).append(i)
    for did in sorted(by)[: a.val_docs]:
        pre = pre_ids(did)
        harvest(did, pre)
        ad = torch.load(os.path.join(a.val_adapters, f"{did}.pt"), map_location="cpu", weights_only=False)
        torch.save(ad["snaps"][200], os.path.join(a.out, "adapters", f"{did}.pt"))
        for it in by[did][: a.val_q]:
            suf = question_suffix_ids(tok, hub, it, SYSTEM_ANSWER)
            j = judged[_get(it, "key")]
            prompts["val"].append({"key": _get(it, "key"), "doc": did, "rows": pre[-64:] + suf, "text": pre + suf,
                                   "judged_kv200": j["kv@200"]["gen"], "judged_text": j["text"]["gen"],
                                   "item": _plain(it)})
    json.dump(prompts, open(os.path.join(a.out, "prompts.json"), "w"))
    log(f"exported to {a.out}: {len(prompts['bench'])} bench docs, {len(prompts['val'])} validation items; "
        f"27B template identical: {prompts['same_template_27b']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
