"""D18 (1/2): context-dependent questions — why the paper's read form (the READER's own state queries the operator)
matters when the question depends on earlier conversation turns (xenc/d18_context.py builds the items).

Conditions (greedy, budget, SYSTEM_ANSWER; reader 9B, encoder 2B, layers paired by depth as D13):
  closed_book    the conversation without the document
  closed_tail    the last --tail document tokens + the conversation (reader alone)
  text           the document in the first user message + the conversation (upper bound)
  encside_final  encoder-side read; the encoder sees the document and ONLY the final user message (as a single-turn
                 question right after the document, as it would in deployment). Injected rows: the document tail and
                 the final message (token-aligned common suffix); earlier turns run stock in the reader.
  encside_conv   encoder-side read; the encoder sees the document + the whole conversation (all reader rows injected)
  paper          the paper-form read (xenc/d13_read.py, D13 interface --interface): the reader's own rows (tail +
                 conversation) query the encoder's fixed operator; the encoder never sees the conversation
  paper_wrong    the paper-form read of ANOTHER document's operator (and its tail) — the content control
Variants: orig (single turn, the D13 judge rows), deferred, distractor, referent.

    python scripts/eval_d18_context.py --interface outputs/D13_maps/d13_mlp.pt --split_dir data/longhealth_split \
        --fold 0 --n 0 --variants orig,deferred,distractor --out outputs/D18_ctx_lh_d13mlp
"""

from __future__ import annotations

import argparse
import copy
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
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402

CONDS = ("closed_book", "closed_tail", "text", "encside_final", "encside_conv", "paper", "paper_wrong")


# ============================================================================ shared runtime (also used by the library eval)
class Patches:
    """Several reader-attention patchers installed on the same modules; `use(name)` selects whose forward is live.
    Every patcher falls back to the stock forward when it is inactive (mode off / no injection)."""

    def __init__(self, mods: dict):
        self.mods, self.fw = mods, {}

    def grab(self, name: str):
        self.fw[name] = {li: m.forward for li, m in self.mods.items()}

    def use(self, name: str):
        for li, m in self.mods.items():
            m.forward = self.fw[name][li]


def make_rt_class():
    from fit_d13 import D13Rt

    class CtxRt(D13Rt):
        """D13Rt whose reader rows can be overridden: it = {"_suffix": ids} -> tail + ids (the conversation).
        _tf: optional operator transform applied to reads (a registered DocAdapter's transform_op)."""
        _tf = None

        def operator(self, did):
            op, n, cache = super().operator(did)
            return (self._tf(op), n, cache) if self._tf is not None else (op, n, cache)

        def rows(self, did, it, target_ids=()):
            if isinstance(it, dict) and "_suffix" in it:
                return self.pre(did)[-self.tail:] + list(it["_suffix"]) + list(target_ids)
            return super().rows(did, it, target_ids)

    return CtxRt


class Runtime:
    """Reader + encoder + D13 runtime (paper form) + encoder-side reader (EncSideInterface, post, replace, fixed B)."""

    def __init__(self, a, need_paper: bool = True):
        from xenc.backbone import input_device, load_encoder, load_reader
        from xenc.d13_read import D13Interface, load_B, load_d13
        from xenc.encside_read import EncCapture, EncSideInterface, EncSideReader
        from xenc.state_skill import text_config
        from xenc.xattn_read import attn_dims, full_depth_map
        self.a = a
        self.reader, self.tok = load_reader(a.reader, device_map=a.device)
        self.enc = load_encoder(a.encoder, device_map=a.device)[0]
        self.hub = a.reader
        keep = {int(x) for x in a.layers.split(",")}
        self.lmap = {k: v for k, v in full_depth_map(attn_dims(self.reader), attn_dims(self.enc)).items() if k in keep}
        B = load_B(a.B, sorted(self.lmap))
        self.dev = input_device(self.reader)
        d_r, d_e = int(text_config(self.reader).hidden_size), int(text_config(self.enc).hidden_size)
        if a.interface:
            itf, _ = load_d13(a.interface, self.enc, B)
            assert itf.layer_map == self.lmap, f"interface layer map {itf.layer_map} != {self.lmap}"
        else:                                      # identity A placeholder (only closed / text / encside are run)
            itf = D13Interface(self.enc, self.lmap, d_r, d_e, B=B)
        self.itf = itf.to(self.dev)
        self.rt = make_rt_class()(self.reader, self.enc, self.tok, a.reader, self.itf, a.tail, a.max_tokens)
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(self.reader)
        self.P = Patches({li: L[li].self_attn for li in self.lmap})
        self.P.grab("d13")
        eitf = EncSideInterface(self.lmap, d_e, d_r, "replace", "post", identity=False)
        with torch.no_grad():
            for li in eitf.layers:
                eitf.B[str(li)].copy_(B[li].float())
        self.eitf = eitf.to(self.dev)
        self.ER = EncSideReader(self.reader, self.eitf)
        self.P.grab("enc")
        self.cap = EncCapture(self.enc, sorted(set(self.lmap.values())), "post")
        self._text_cache: dict = {}
        self.ad, self._snaps = None, {}
        if getattr(a, "adapters", None):                 # D20 per-document registration (strict: no unregistered reads)
            self._init_adapters(a.adapters, a.adapter_budget)

    # ------------------------------------------------------------------ D20 registered adapters
    def _init_adapters(self, adir: str, budget: int):
        import glob
        from xenc.d20_reg import DocAdapter
        cfg = None
        for f in sorted(glob.glob(os.path.join(adir, "args_*register*.json"))):
            cfg = json.load(open(f))
            break
        assert cfg is not None, f"no D20 registration args_*.json in {adir}"
        assert os.path.abspath(cfg["interface"]) == os.path.abspath(self.a.interface), \
            f"adapters were registered on base {cfg['interface']}, not {self.a.interface}"
        if self.itf.delta_rank:
            self.itf.merge_delta()
        self.ad = DocAdapter(self.itf, [p for p in cfg["parts"].split(",") if p], cfg["rank"], cfg["kv_rank"],
                             cfg["seed"])
        self.ad_dir, self.ad_budget = os.path.join(adir, "adapters"), int(budget)
        log(f"D20 adapters: {adir} parts {sorted(self.ad.parts)} rank {cfg['rank']} budget {budget}")

    def use_doc(self, did):
        """Load did's registered adapter (hard error if it was not registered); returns the operator transform."""
        if self.ad is None:
            return None
        if did not in self._snaps:
            path = os.path.join(self.ad_dir, f"{did}.pt")
            assert os.path.exists(path), f"UNREGISTERED document {did}: no adapter {path} (reads need registration)"
            r = torch.load(path, map_location="cpu", weights_only=False)
            self._snaps[did] = r["snaps"][self.ad_budget]
            if len(self._snaps) > 64:
                self._snaps.pop(next(iter(self._snaps)))
        self.ad.load(self._snaps[did])
        return self.ad.transform_op

    def base_itf(self):
        """Return the shared interface to its base (zero per-document deltas): routing / unregistered-free passes."""
        if self.ad is not None:
            self.ad.reset()

    # ------------------------------------------------------------------ plain reader
    def greedy(self, ids, budget, cache=None):
        from register_state import greedy
        return greedy(self.reader, self.tok, self.hub, ids, budget, cache=cache)

    @torch.no_grad()
    def text_cached(self, did, suffix_ids, budget):
        """Reader with the document prefix (cached per document) + suffix ids (the `text` condition)."""
        if did not in self._text_cache:
            self._text_cache.clear()
            out = self.reader(input_ids=torch.tensor([self.rt.pre(did)], device=self.dev), use_cache=True,
                              logits_to_keep=1)
            self._text_cache[did] = out.past_key_values
        return self.greedy(suffix_ids, budget, cache=copy.deepcopy(self._text_cache[did]))

    # ------------------------------------------------------------------ encoder side
    @torch.no_grad()
    def enc_run(self, ids, past):
        """Encoder over ids continuing `past` -> ({enc_layer: [1,T,d_e]} post outputs, cache)."""
        self.cap.start()
        out = self.enc(input_ids=torch.tensor([ids], device=self.rt.edev), past_key_values=past, use_cache=True,
                       logits_to_keep=1)
        return self.cap.get(), out.past_key_values

    @torch.no_grad()
    def gen_encside(self, enc_from, enc_ids, segs, budget, stop="</answer>"):
        """enc_from: encoder cache to continue (deep-copied); enc_ids: encoder rows after it.
        segs: [(reader_ids, (lo, hi) rows of the encoder output to inject, or None = stock)], in reader order.
        Greedy decode; the encoder steps along on every generated token (eval_encside.gen_encside semantics)."""
        from register_state import _stop_ids
        from xenc.backbone import decode
        self.P.use("enc")
        R, lmap = self.ER, self.lmap
        o2, ecache = self.enc_run(enc_ids, copy.deepcopy(enc_from))
        past, out = None, None
        try:
            for ids, rng in segs:
                if not ids:
                    continue
                R.cur = None if rng is None else {li: o2[e][:, rng[0]:rng[1]] for li, e in lmap.items()}
                if rng is not None:
                    assert rng[1] - rng[0] == len(ids), (rng, len(ids))
                out = self.reader(input_ids=torch.tensor([ids], device=self.dev), past_key_values=past,
                                  use_cache=True, logits_to_keep=1)
                past = out.past_key_values
            stops, new = _stop_ids(self.tok), []
            for _ in range(budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in stops or stop in self.tok.decode(new[-12:], skip_special_tokens=True):
                    break
                o2, ecache = self.enc_run([nxt], ecache)
                R.cur = {li: o2[e] for li, e in lmap.items()}
                out = self.reader(input_ids=torch.tensor([[nxt]], device=self.dev), past_key_values=out.past_key_values,
                                  use_cache=True, logits_to_keep=1)
        finally:
            R.cur = None
        txt = decode(self.tok, self.hub, new)
        if stop in txt:
            txt = txt.split(stop, 1)[0] + stop
        return txt, len(new)

    def paper_gen(self, did, suffix_ids, budget):
        self.P.use("d13")
        self.rt._tf = self.use_doc(did)
        try:
            return self.rt.gen(did, {"_suffix": list(suffix_ids)}, budget)
        finally:
            self.rt._tf = None


def ids_of(tok, s: str) -> list[int]:
    return [int(x) for x in tok(s, add_special_tokens=False)["input_ids"]]


def paired(rows, a: str, b: str) -> dict | None:
    pair = [r[a]["score"] - r[b]["score"] for r in rows if a in r and b in r]
    if not pair:
        return None
    mu = sum(pair) / len(pair)
    sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
    return {"gap": mu, "se": sd / len(pair) ** 0.5, "n": len(pair), "wins": sum(p > 0 for p in pair),
            "losses": sum(p < 0 for p in pair)}


def summarize(rows, conds) -> dict:
    s = {}
    for c in conds:
        v = [r[c] for r in rows if c in r]
        if v:
            s[c] = {"score": sum(x["score"] for x in v) / len(v), "n": len(v),
                    "mean_gen_tokens": sum(x["gen_tokens"] for x in v) / len(v)}
    for a_, b_ in (("paper", "paper_wrong"), ("paper", "closed_tail"), ("paper", "encside_final"),
                   ("encside_conv", "encside_final"), ("paper", "encside_conv")):
        p = paired(rows, a_, b_)
        if p:
            s[f"{a_}-{b_}"] = p
    return s


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default=None, help="D13-format interface (d13_read.v1); omit = no paper conds")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=300, help="item sample (same seed/sample as eval_d13); 0 = all")
    ap.add_argument("--variants", default="orig,deferred,distractor,referent")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--adapters", default=None, help="D20 registration dir (adapters/<doc>.pt, args_*register*.json); "
                    "every read loads its document's adapter — an unregistered document is a hard error")
    ap.add_argument("--adapter_budget", type=int, default=0, help="registration snapshot (D20 --budgets entry)")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_doc import wrong_doc_id
    from xenc.d18_context import build_context_items, common_suffix_len, render, segments
    from xenc.doc_tasks import load_docs, score
    conds = [c for c in CONDS if c in a.conds.split(",")]
    if not a.interface:
        conds = [c for c in conds if not c.startswith("paper")]
    os.makedirs(a.out, exist_ok=True)
    rt = Runtime(a)
    tok = rt.tok
    docs = load_docs(a.split_dir)
    rt.rt.docs = docs
    items = load_split_items(a.split_dir, a.split)
    doc_order = sorted({_get(i, "doc_id") for i in items})
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    variants = [v for v in a.variants.split(",") if v]
    log(f"D18 context eval: {a.encoder} -> {a.reader} | layers {rt.lmap} | interface {a.interface} | variants "
        f"{variants} | conds {conds} | {len(items)} items")
    allrows, summ = {}, {}
    ck = open(os.path.join(a.out, f"ctx_{a.split}.ckpt.jsonl"), "w")
    t0 = time.time()
    for var in variants:
        cis = build_context_items(items, docs, var)
        cis.sort(key=lambda c: c["doc_id"])                 # cache reuse; scores unaffected
        rows = []
        for ci in cis:
            did, it = ci["doc_id"], ci["item"]
            w = wrong_doc_id(did, doc_order)
            seg = segments(tok, a.reader, ci, SYSTEM_ANSWER)
            conv = ids_of(tok, seg["conv_str"])
            pre = rt.rt.pre(did)
            rec = {"key": ci["key"], "doc_id": did, "wrong": w, "variant": var, "meta": ci["meta"]}
            for c in conds:
                if c == "closed_book":
                    rt.P.use("d13")
                    g, n = rt.greedy(ids_of(tok, render(tok, a.reader, ci, SYSTEM_ANSWER, None)), a.budget)
                elif c == "closed_tail":
                    rt.P.use("d13")
                    g, n = rt.greedy(pre[-a.tail:] + conv, a.budget)
                elif c == "text":
                    rt.P.use("d13")
                    g, n = rt.text_cached(did, conv, a.budget)
                elif c == "paper":
                    g, n = rt.paper_gen(did, conv, a.budget)
                elif c == "paper_wrong":
                    g, n = rt.paper_gen(w, conv, a.budget)
                elif c in ("encside_conv", "encside_final"):
                    _, _, ecache = rt.rt.operator(did)       # encoder cache after pre[:-tail]
                    tail = pre[-a.tail:]
                    if c == "encside_conv":
                        enc_ids = tail + conv
                        segs = [(tail + conv, (0, len(enc_ids)))]
                    else:
                        fin = ids_of(tok, seg["enc_final"])
                        k = common_suffix_len(fin, conv)
                        enc_ids = tail + fin
                        segs = [(tail, (0, len(tail))), (conv[:len(conv) - k], None),
                                (conv[len(conv) - k:], (len(enc_ids) - k, len(enc_ids)))]
                        rec["n_final_rows"] = k
                    g, n = rt.gen_encside(ecache, enc_ids, segs, a.budget)
                rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            rows.append(rec)
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            if len(rows) % 20 == 0:
                s = summarize(rows, conds)
                log(f"  [{var} {len(rows)}/{len(cis)}] " + " ".join(f"{c}={s[c]['score']:.3f}" for c in conds if c in s)
                    + f" | {(time.time() - t0):.0f}s")
        allrows[var] = rows
        summ[var] = summarize(rows, conds)
        json.dump({"summary": summ, "args": vars(a), "items": allrows},
                  open(os.path.join(a.out, f"ctx_{a.split}.json"), "w"), indent=1)
        log(f"==== variant {var} n={len(rows)}")
        for c in conds:
            if c in summ[var]:
                v = summ[var][c]
                log(f"  {c:14s} {v['score']:.4f}  n={v['n']}  gen {v['mean_gen_tokens']:.1f}")
        for k, v in summ[var].items():
            if "-" in k:
                log(f"  {k}: {v['gap']:+.4f} ± {v['se']:.4f} (n={v['n']}, {v['wins']}/{v['losses']})")
    log(f"done ({time.time() - t0:.0f}s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
