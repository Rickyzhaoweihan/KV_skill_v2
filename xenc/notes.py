"""NOTES (agent NOTES, 2026-10-23): encoder-written NOTE columns appended to the D13 operator.

The small encoder is an LLM. After it prefills the document (the operator prefix = document prefix minus the reader's
`tail` tokens, exactly D13Rt.operator), it continues the SAME sequence with a question-independent note prompt and
greedily writes notes (digest / key facts / entities + timeline / reasoning / self-QA). The encoder's own post-RoPE
K/V of [note prompt ; generated note tokens] at the operator layers are APPENDED to the operator as extra columns:

    encoder sequence   pre[:-tail]  |  note prompt (P tokens)  |  notes[:L]
    operator columns   0 .. N-1     |  N .. N+P-1              |  N+P .. N+P+L-1      (contiguous positions)
    reader rows        tail + question at positions N+P+L ...   (D13Reader uses pos0 = n_op, unchanged)

The reader never sees note TEXT: notes enter only as encoder K/V, the same format as the document columns. The note
prompt never contains the question. Generation is causal and greedy, so the columns for length L are the first P+L
columns of the length-Lmax harvest (eval slices them; training harvests exactly L and keeps the encoder cache so the
encoder teacher continues from [document ; notes]).

  NoteRt(D13Rt)   note = None -> exactly D13Rt (tests/test_notes.py); note = {"type", "L", "src"} -> op + note columns.
                  src = another doc id: that document's notes harvested in THIS document's context ("wrong notes").
  NoteStore       note token ids per (encoder tag, tail, max_tokens, type, doc) as small JSON files under
                  $XENC_CACHE/notes (on /scratch); written once, reused by eval, training and wrong-notes controls.
"""

from __future__ import annotations

import copy
import json
import os
import time

import torch

NOTE_PROMPTS = {
    "digest": ("Write a concise digest of the document above: what it is about, who or what is involved, its main "
               "points in the order they appear, and its conclusions or outcome. Write plain prose in a few "
               "sentences."),
    "facts": ("List the key facts of the document above as short bullet points. Keep the exact names, numbers, "
              "dates, places, methods and results, and state how they relate to each other. Do not add anything "
              "that is not in the document."),
    "entities": ("Make a list of the main entities in the document above (people, places, organizations, objects, "
                 "concepts) with their attributes, roles and relations to each other, then a timeline of the main "
                 "events in order."),
    "reason": ("Think about the document above: what are its main claims or events, how do they connect, what "
               "causes what, and what follows from them? Note anything surprising, contrasting or easy to confuse. "
               "Write your analysis."),
    "selfqa": ("Write question-answer pairs that test a careful reading of the document above, covering its "
               "important details: names, numbers, events, claims and how they relate. Each answer must come from "
               "the document. Use the format Q: ... A: ..."),
}
NOTE_HEADER = "\n\n## Notes\n"
NOTES_VERSION = "nv1"   # bump when NOTE_PROMPTS / NOTE_HEADER / P_FIX change (store path)
P_FIX = 64          # note-prompt columns per type (newline-padded at the front; prompts are 56-61 tokens, Qwen3.5)


def note_suffix_ids(tok, hub: str, system: str, instr: str) -> list[int]:
    """Ids after the operator prefix: header + instruction + the chat tail (end of user turn, assistant start, no
    thinking). Built like state_skill.question_suffix_ids (same template, same system prompt)."""
    from deltaskill.backbone_utils import chat_template_kwargs
    mark = "\x00U\x00"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": mark}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(hub))
    tail = s[s.index(mark) + len(mark):]
    return [int(x) for x in tok(NOTE_HEADER + instr + tail, add_special_tokens=False)["input_ids"]]


class NoteStore:
    """{type}/{doc}.json = {"suffix": ids, "gen": ids, "eos": bool, "max_new": int, "s_gen": secs, ...}."""

    def __init__(self, root: str):
        self.root = root

    def path(self, ntype: str, did: str) -> str:
        return os.path.join(self.root, ntype, str(did).replace("/", "_") + ".json")

    def get(self, ntype: str, did: str):
        p = self.path(ntype, did)
        if not os.path.exists(p):
            return None
        try:
            return json.load(open(p))
        except json.JSONDecodeError:      # a concurrent writer: treat as missing (atomic replace below)
            return None

    def put(self, ntype: str, did: str, rec: dict):
        p = self.path(ntype, did)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}.tmp"
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, p)


def store_root(enc_hub: str, tail: int, max_tokens: int, base: str | None = None) -> str:
    base = base or os.path.join(os.environ.get("XENC_CACHE", "/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache"),
                                "notes")
    return os.path.join(base, f"{NOTES_VERSION}_{enc_hub.replace('/', '_')}_t{int(tail)}_m{int(max_tokens)}")


def _pad_equal(suffixes: list[list[int]], pad_id: int) -> list[list[int]]:
    """Equal-length suffixes for one batched generation: shorter ones get `pad_id` (a newline token) PREPENDED, i.e.
    extra blank lines between the document and the notes header. The padded suffix is what is stored and harvested."""
    m = max(len(s) for s in suffixes)
    return [[pad_id] * (m - len(s)) + list(s) for s in suffixes]


@torch.no_grad()
def generate_notes(enc, cache, suffixes: list[list[int]], max_new: int, stop_ids: set, pad_id: int) -> list[dict]:
    """Greedy notes from one shared operator cache (deep-copied, never modified): the K suffixes run as one batch
    (equal lengths required). -> [{"gen": ids (stop token excluded), "eos": bool}] in suffix order."""
    from xenc.backbone import input_device
    dev = input_device(enc)
    K = len(suffixes)
    assert len({len(s) for s in suffixes}) == 1, "batched suffixes must have equal lengths (use _pad_equal)"
    c = copy.deepcopy(cache)
    if K > 1:
        c.reorder_cache(torch.zeros(K, dtype=torch.long, device=dev))
    out = enc(input_ids=torch.tensor(suffixes, device=dev), past_key_values=c, use_cache=True, logits_to_keep=1)
    gen = [[] for _ in range(K)]
    done = [False] * K
    for _ in range(int(max_new)):
        nxt = out.logits[:, -1].argmax(-1)
        for b, t in enumerate(nxt.tolist()):
            if not done[b]:
                if t in stop_ids:
                    done[b] = True
                else:
                    gen[b].append(int(t))
        if all(done):
            break
        nxt = torch.where(torch.tensor(done, device=nxt.device), torch.full_like(nxt, pad_id), nxt)
        out = enc(input_ids=nxt.view(-1, 1), past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
    del c, out
    return [{"gen": g, "eos": bool(d)} for g, d in zip(gen, done)]


def make_note_rt(base_cls):
    """NoteRt over any D13Rt-like runtime class (fit_d13.D13Rt)."""

    class NoteRt(base_cls):
        """D13Rt + note columns. Set rt.note = None (plain D13Rt) or {"type": t, "L": L, "src": doc id or None}."""

        def setup_notes(self, store: NoteStore, suffix: dict, max_new: int, stop_ids: set, pad_id: int,
                        gen_types: list | None = None, train: bool = False, gen_ok: bool = True):
            assert not (getattr(self.itf, "needs_ids", False) or getattr(self.itf, "needs_seg", False)
                        or hasattr(self.itf, "harvest_hooks")), "note columns: plain D13 / CAP interfaces only"
            self.note, self.store, self.suffix = None, store, dict(suffix)
            self.max_new, self.stop_ids, self.pad_id = int(max_new), set(stop_ids), int(pad_id)
            self.gen_types = list(gen_types or suffix)
            self.train_mode, self.gen_ok = bool(train), bool(gen_ok)
            self._cols: dict = {}
            self.note_log: list = []           # per generated doc: timings / lengths (cost report)

        # ---------------------------------------------------------------- notes text (ids)
        def note_rec(self, did, ntype):
            r = self.store.get(ntype, did)
            if r is not None and len(r["gen"]) < self.max_new and not r["eos"] and r["max_new"] < self.max_new:
                r = None                       # stored with a smaller budget: regenerate (greedy: same prefix)
            if r is None:
                assert self.gen_ok, f"notes for {did} / {ntype} are not in the store {self.store.root}"
                self.gen_doc(did)
                r = self.store.get(ntype, did)
            return r

        @torch.no_grad()
        def gen_doc(self, did):
            """One batched greedy generation of every type in gen_types from did's operator cache."""
            if self.edev.type == "cuda":
                torch.cuda.synchronize(self.edev)
            t0 = time.time()
            _, N, cache = base_cls.operator(self, did)
            if self.edev.type == "cuda":
                torch.cuda.synchronize(self.edev)
            t1 = time.time()
            sufs = _pad_equal([self.suffix[t] for t in self.gen_types], self.pad_id)
            outs = generate_notes(self.enc, cache, sufs, self.max_new, self.stop_ids, self.pad_id)
            if self.edev.type == "cuda":
                torch.cuda.synchronize(self.edev)
            t2 = time.time()
            for t, s, o in zip(self.gen_types, sufs, outs):
                self.store.put(t, did, {"suffix": s, "gen": o["gen"], "eos": o["eos"], "max_new": self.max_new,
                                        "n_doc": int(N), "batch": len(sufs), "s_gen_batch": t2 - t1})
            self.note_log.append({"doc": did, "n_doc": int(N), "s_op": t1 - t0, "s_gen_batch": t2 - t1,
                                  "batch": len(sufs), "gen_len": [len(o["gen"]) for o in outs]})

        # ---------------------------------------------------------------- note columns
        @torch.no_grad()
        def _harvest(self, did, ids):
            """Encoder continuing from did's operator cache over `ids` -> ({e: (K, V)} of those rows, cache after)."""
            _, N, cache = base_cls.operator(self, did)
            self.Tc.rec, self.Tc.kv_only = {}, True
            try:
                out = self.enc(input_ids=torch.tensor([ids], device=self.edev), past_key_values=copy.deepcopy(cache),
                               use_cache=True, logits_to_keep=1)
                cols = {e: (r["k"].to(self.dev), r["v"].to(self.dev)) for e, r in self.Tc.rec.items()}
            finally:
                self.Tc.rec, self.Tc.kv_only = None, False
            return cols, N, out.past_key_values

        def operator(self, did):
            cfg = getattr(self, "note", None)
            if cfg is None:
                return base_cls.operator(self, did)
            nt, L = cfg["type"], int(cfg["L"])
            src = cfg.get("src") or did
            r = self.note_rec(src, nt)
            n_gen = min(L, len(r["gen"]))
            if self.train_mode:                # exact harvest of P + L columns; keeps the encoder cache (teacher)
                key = ("train", did, nt, src, n_gen)
                if key in self._cols:
                    self._cols[key] = self._cols.pop(key)
                    return self._cols[key]
                op_d, N, _ = base_cls.operator(self, did)
                cols, _, cache = self._harvest(did, r["suffix"] + r["gen"][:n_gen])
                res = ({e: (torch.cat([op_d[e][0], cols[e][0]], 1), torch.cat([op_d[e][1], cols[e][1]], 1))
                        for e in op_d}, N + len(r["suffix"]) + n_gen, cache)
                self._cols[key] = res
                self._ops.pop(did, None)        # the plain operator + its cache are not needed again (memory)
                while len(self._cols) > max(1, self.lru):
                    self._cols.pop(next(iter(self._cols)))
                return res
            key = (did, nt, src)                # eval: one harvest at the stored length, sliced per L (causal)
            if key not in self._cols:
                if len(self._cols) >= 64 or (self._cols and next(reversed(self._cols))[0] != did):
                    self._cols.clear()          # items are processed doc by doc: keep only the current doc's columns
                self._cols[key] = (self._harvest(did, r["suffix"] + r["gen"])[0], len(r["suffix"]))
            cols, P = self._cols[key]
            op_d, N, _ = base_cls.operator(self, did)
            n_c = P + n_gen
            op = {e: (torch.cat([op_d[e][0], cols[e][0][:, :n_c]], 1), torch.cat([op_d[e][1], cols[e][1][:, :n_c]], 1))
                  for e in op_d}
            return op, N + n_c, None

    return NoteRt


def build_note_rt(rt_cls, reader, enc, tok, hub, enc_hub, itf, tail, max_tokens, system, types, max_new,
                  lru: int = 3, train: bool = False, store_base: str | None = None, gen_ok: bool = True):
    """Convenience constructor used by scripts/eval_notes.py and register_d13.py --notes."""
    NoteRt = make_note_rt(rt_cls)
    rt = NoteRt(reader, enc, tok, hub, itf, tail, max_tokens, lru=lru)
    pad = tok("\n", add_special_tokens=False)["input_ids"]
    assert len(pad) == 1, pad
    # every type's suffix is padded to ONE fixed length (P_FIX), independent of which types share a generation batch,
    # so a single-type run (training) and the 5-type probe store the same prompt for a type
    suffix = {t: note_suffix_ids(tok, enc_hub, system, NOTE_PROMPTS[t]) for t in types}
    assert max(len(s) for s in suffix.values()) <= P_FIX, {t: len(s) for t, s in suffix.items()}
    suffix = {t: _pad_equal([s, [0] * P_FIX], pad[0])[0] for t, s in suffix.items()}
    stop = {int(tok.convert_tokens_to_ids("<|im_end|>"))}
    if tok.eos_token_id is not None:
        stop.add(int(tok.eos_token_id))
    rt.setup_notes(NoteStore(store_root(enc_hub, tail, max_tokens, store_base)), suffix, max_new, stop, pad[0],
                   gen_types=list(types), train=train, gen_ok=gen_ok)
    return rt


__all__ = ["NOTE_PROMPTS", "note_suffix_ids", "NoteStore", "store_root", "generate_notes", "make_note_rt",
           "build_note_rt"]
