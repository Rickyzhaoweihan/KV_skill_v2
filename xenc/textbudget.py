"""BL / S2 + S4: budget-matched TEXT baselines for the same reader (agent BASE, 2026-10-23).

The reader reads at most K document tokens (Qwen3.5 tokenizer) of the document, in the SAME prompt as the text row
(`xenc.doc_tasks.ctx_prompt` + SYSTEM_ANSWER), so the only change against `text` is which document text it sees.

Context builders (all return (context_text, info)):
  trunc_head  first K document tokens
  trunc_ht    first K/2 + last K/2 tokens joined by "\n\n[...]\n\n" (LongBench-style middle truncation)
  bm25        top-k = K // C chunks of C tokens (default 512) by Okapi BM25 over the question text, document order
  dense       same with BAAI/bge-base-en-v1.5 (CLS, cosine; chunk score = max over its 256-token passages)
  hybrid      reciprocal-rank fusion (k=60) of bm25 and dense (H2's val-selected retriever)
  (S4 compressors live in xenc/lingua.py and only produce the compressed document text.)

Retrieval sees ONLY the question text (h2_retrieval.query_text: the question with fixed task templates stripped, plus
every answer option longer than 15 characters; all options, so the gold label is never singled out). The gold answer,
evidence and meta fields are never read. A document of N <= K tokens is passed through unchanged (= the text row);
info["full"] = 1 marks it.

Budget: the document-content tokens (sum of the selected token windows) are <= K by construction; the re-tokenized
context string may differ by a few tokens at window seams / separators, recorded as info["ctx_tokens"].
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

SEP = "\n\n[...]\n\n"
PASSAGE = 256
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those he she they them his her their we you i me my our your not no do does did done have has had having so than
then there here which who whom whose what when where why how all any each both few more most other some such only own
same too very can will just should would could may might must shall into over under again further once about above
below between through during before after up down out off also s t""".split())
TEMPLATE = [r"Is the following statement about the story true or false\?", r"Statement:",
            r"Below is a summary of part of the document in which named entities are replaced by <mask-k> placeholders\.",
            r"Which entity is <mask-\d+>\?", r"<mask-\d+>", r"Reorder the timeline of the following events:",
            r"Consider this statement about the document:", r"According to the document,?"]


def _get(item, k, default=None):
    if isinstance(item, dict):
        return item.get(k, default)
    return getattr(item, k, default)


def query_text(it) -> str:
    """Retrieval query: question (task templates stripped) + every option text longer than 15 characters
    (identical rule to scripts/h2_retrieval.query_text). Never reads answers / correct_label / meta."""
    q = str(_get(it, "question"))
    for t in TEMPLATE:
        q = re.sub(t, " ", q)
    q = q.replace('"', " ")
    opts = [c["text"] for c in (_get(it, "choices") or []) if len(str(c["text"])) > 15]
    return " ".join([q] + [str(o) for o in opts]).strip()


def words(t: str) -> list[str]:
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOP and len(w) > 1]


class BM25:
    """Okapi BM25 (k1 1.5, b .75), identical to scripts/h2_retrieval.BM25."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = [len(d) for d in docs_tokens]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        N = len(docs_tokens)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, q):
        out = []
        qs = Counter(q)
        for c, L in zip(self.tf, self.len):
            s = 0.0
            for w, qn in qs.items():
                f = c.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out


def rrf(*rankings, k=60):
    s = defaultdict(float)
    for r in rankings:
        for i, c in enumerate(r):
            s[c] += 1.0 / (k + i + 1)
    return sorted(s, key=lambda c: (-s[c], c))


def rank(scores) -> list[int]:
    """Descending score, ties broken by position (deterministic)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


class Dense:
    """bge-base-en-v1.5 CLS embeddings (as scripts/h2_retrieval.Dense), on `device`."""

    def __init__(self, name="BAAI/bge-base-en-v1.5", device="cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).eval().to(device)

    def embed(self, texts, bs=64, query=False):
        torch = self.torch
        pre = "Represent this sentence for searching relevant passages: " if query else ""
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                b = self.tok([pre + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=512,
                             return_tensors="pt").to(self.device)
                h = self.m(**b).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 768)


class DocIndex:
    """Token-aligned chunks of one document + lazily built BM25 / dense indices."""

    def __init__(self, tok, text: str, chunk: int = 512):
        self.tok, self.text, self.C = tok, text, int(chunk)
        self.ids = tok(text, add_special_tokens=False)["input_ids"]
        self.N = len(self.ids)
        self.spans = [(s, min(s + self.C, self.N)) for s in range(0, self.N, self.C)]
        self.chunk_text = [tok.decode(self.ids[s:e]) for s, e in self.spans]
        self._bm25, self._pass_emb, self._pass_owner = None, None, None

    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([words(t) for t in self.chunk_text])
        return self._bm25

    def dense_scores(self, dense: Dense, q: str) -> list[float]:
        import torch
        if self._pass_emb is None:
            texts, owner = [], []
            for j, (s, e) in enumerate(self.spans):
                for p in range(s, e, PASSAGE):
                    texts.append(self.tok.decode(self.ids[p:min(p + PASSAGE, e)]))
                    owner.append(j)
            self._pass_emb, self._pass_owner = dense.embed(texts), torch.tensor(owner)
        qe = dense.embed([q], query=True)[0]
        ps = self._pass_emb @ qe
        out = torch.full((len(self.spans),), -1e9)
        out = out.scatter_reduce(0, self._pass_owner, ps, reduce="amax", include_self=True)
        return out.tolist()

    def select(self, method: str, q: str, k: int, dense: Dense | None = None) -> list[int]:
        """Top-k chunk indices (sorted by document position) for retriever method bm25 | dense | hybrid."""
        if method == "bm25":
            order = rank(self.bm25().scores(words(q)))
        elif method == "dense":
            order = rank(self.dense_scores(dense, q))
        elif method == "hybrid":
            order = rrf(rank(self.bm25().scores(words(q))), rank(self.dense_scores(dense, q)))
        else:
            raise ValueError(method)
        return sorted(order[:k])


def parse_cond(cond: str) -> tuple[str, int | None, float | None]:
    """'bm25@2048' -> ('bm25', 2048, None); 'lingua2@r4' -> ('lingua2', None, 0.25)."""
    m = re.fullmatch(r"([a-z0-9_]+)@(r?)(\d+(?:\.\d+)?)", cond)
    if not m:
        raise ValueError(f"bad cond {cond!r}: expected <method>@<K> or <method>@r<ratio>")
    if m.group(2):
        return m.group(1), None, 1.0 / float(m.group(3))
    return m.group(1), int(float(m.group(3))), None


def build_context(method: str, K: int, idx: DocIndex, q: str = "", dense: Dense | None = None) -> tuple[str, dict]:
    """Document context for the reader under a K-token budget (see module doc)."""
    N, ids, tok = idx.N, idx.ids, idx.tok
    if N <= K:
        return idx.text, {"full": 1, "doc_tokens": N, "content_tokens": N, "sel": None}
    if method == "trunc_head":
        txt = tok.decode(ids[:K])
        return txt, {"full": 0, "doc_tokens": N, "content_tokens": K, "sel": [[0, K]]}
    if method == "trunc_ht":
        h = K // 2
        txt = tok.decode(ids[:h]) + SEP + tok.decode(ids[N - (K - h):])
        return txt, {"full": 0, "doc_tokens": N, "content_tokens": K, "sel": [[0, h], [N - (K - h), N]]}
    if method in ("bm25", "dense", "hybrid"):
        if K < idx.C:                      # budget below one chunk: one chunk of K tokens (finer index, cached)
            sub = idx.__dict__.setdefault("_sub", {})
            if K not in sub:
                sub[K] = DocIndex(tok, idx.text, chunk=K)
            idx = sub[K]
        k = max(1, K // idx.C)
        sel = idx.select(method, q, k, dense)
        txt = SEP.join(idx.chunk_text[j] for j in sel)
        content = sum(idx.spans[j][1] - idx.spans[j][0] for j in sel)
        assert content <= K, (content, K)
        return txt, {"full": 0, "doc_tokens": N, "content_tokens": content, "sel": sel}
    raise ValueError(method)


__all__ = ["query_text", "words", "BM25", "rrf", "rank", "Dense", "DocIndex", "parse_cond", "build_context", "SEP"]
