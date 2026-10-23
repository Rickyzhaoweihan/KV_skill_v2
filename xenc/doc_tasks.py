"""[C] Document-QA data API: schema, loaders, prompts, scoring, chunking.

Split layout (produced by ``scripts/data/prepare_*.py``)::

    data/<name>_split/{train,val,test}/items.json   # JSON list of item dicts (schema below)
    data/<name>_split/docs.json                     # doc_id -> {"path", "n_tokens", "title"}
    data/docs/<name>/<doc_id>.md                    # plain-text document

Item schema::

    {"key", "doc_id", "question", "choices": [{"label","text"}] | None, "answers": [str, ...],
     "correct_label": str | None, "task": str, "hop": int | None, "meta": {...}}

``docs.json`` paths are relative to the data root (the split dir's parent), e.g.
``docs/quality/quality_0123456789ab.md``.

Scoring: MCQ = exact label match (``meta.accept_labels`` widens the gold set where the raw data
lists the correct option text twice); span = SQuAD token-F1, max over ``answers`` (ported from
``KV_skill/kvskill/reward_bench.py``).
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPLITS = ("train", "val", "test")
ITEM_FIELDS = ("key", "doc_id", "question", "choices", "answers", "correct_label", "task", "hop", "meta")
MCQ_INSTRUCTION = ("Respond with ONLY the letter of the correct option inside <answer></answer> tags, "
                   "for example <answer>B</answer>. Do not explain.")
SPAN_INSTRUCTION = ("Respond with ONLY a short answer phrase inside <answer></answer> tags, "
                    "for example <answer>Paris</answer>. Do not explain.")


@dataclass
class DocItem:
    key: str
    doc_id: str
    question: str
    choices: list[dict] | None
    answers: list[str]
    correct_label: str | None
    task: str
    hop: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_mcq(self) -> bool:
        return bool(self.choices)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DocItem":
        return cls(**{k: d.get(k) for k in ITEM_FIELDS if k != "meta"}, meta=dict(d.get("meta") or {}))


# --------------------------------------------------------------------------- loading


def load_doc_split(split_dir: str, split: str) -> list[DocItem]:
    p = Path(split_dir) / split / "items.json"
    with open(p) as f:
        return [DocItem.from_dict(d) for d in json.load(f)]


def _doc_path(split_dir: Path, rel: str) -> Path:
    for base in (split_dir.parent, split_dir, Path(__file__).resolve().parents[1] / "data"):
        q = base / rel
        if q.exists():
            return q
    return split_dir.parent / rel


def load_docs_index(split_dir: str) -> dict[str, dict]:
    with open(Path(split_dir) / "docs.json") as f:
        return json.load(f)


def load_docs(split_dir: str) -> dict[str, str]:
    """doc_id -> document text (reads every doc listed in ``docs.json``)."""
    sd = Path(split_dir)
    return {did: _doc_path(sd, m["path"]).read_text() for did, m in load_docs_index(split_dir).items()}


# --------------------------------------------------------------------------- prompts


def _get(item, k):
    return item.get(k) if isinstance(item, dict) else getattr(item, k)


def q_prompt(item) -> str:
    """Question (+ lettered choices) and the answer-format instruction, WITHOUT the document."""
    q = str(_get(item, "question")).strip()
    choices = _get(item, "choices")
    if choices:
        opts = "\n".join(f"{c['label']}. {c['text']}" for c in choices)
        return f"{q}\n\n{opts}\n\n{MCQ_INSTRUCTION}"
    return f"{q}\n\n{SPAN_INSTRUCTION}"


def ctx_prompt(item, doc_text: str, max_chars: int | None = None) -> str:
    doc = doc_text if max_chars is None else doc_text[:max_chars]
    return f"## Document\n{doc}\n\n## Question\n{q_prompt(item)}"


def gold_text(item) -> str:
    if _get(item, "choices"):
        return f"<answer>{_get(item, 'correct_label')}</answer>"
    return f"<answer>{_get(item, 'answers')[0]}</answer>"


# --------------------------------------------------------------------------- scoring (port of kvskill.reward_bench)


def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def extract_answer(text: str) -> str:
    """Last ``<answer>...</answer>`` content; fallback last non-empty line, then the stripped text."""
    text = text or ""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(normalize_answer(g) == p for g in gold_answers))


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Token-level F1 (SQuAD-style), max across gold answers."""
    pred = normalize_answer(prediction).split()
    if not pred:
        return float(any(not normalize_answer(g).split() for g in gold_answers))
    best = 0.0
    for g in gold_answers:
        gt = normalize_answer(g).split()
        if not gt:
            continue
        n = sum((Counter(pred) & Counter(gt)).values())
        if n == 0:
            continue
        p, r = n / len(pred), n / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best


_LABEL_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")


def extract_label(generation: str, valid: str = "ABCDE") -> str | None:
    """MCQ label: from ``<answer>X</answer>`` (first standalone A–E inside the last tag);
    fallback: last standalone capital letter A–E in the whole generation."""
    generation = generation or ""
    m = re.findall(r"<answer>(.*?)</answer>", generation, re.DOTALL | re.IGNORECASE)
    if m:
        inner = m[-1].strip()
        lab = inner.upper().strip().strip("()[]*").rstrip(".):").lstrip("(").strip()
        if lab in valid and len(lab) == 1:      # "B", "b", "(b)", "B.", "[B]", "**b**"
            return lab
        hits = [h for h in _LABEL_RE.findall(inner) if h in valid]
        if hits:
            return hits[0]
        m1 = re.match(r"^\(?([a-e])\)?[.):]?(\s|$)", inner)   # lowercase "b) text" / "(c) text"
        if m1 and m1.group(1).upper() in valid:
            return m1.group(1).upper()
    hits = [h for h in _LABEL_RE.findall(generation) if h in valid]
    return hits[-1] if hits else None


def score(item, generation: str) -> float:
    """MCQ: 1.0 iff the extracted label is the gold label; span: max token-F1 over answers."""
    generation = generation or ""
    choices = _get(item, "choices")
    if choices:
        valid = "".join(str(c["label"]) for c in choices)
        meta = _get(item, "meta") or {}
        gold = set(meta.get("accept_labels") or [_get(item, "correct_label")])
        return float(extract_label(generation, valid) in gold)
    return f1_score(extract_answer(generation), list(_get(item, "answers")))


def exact(item, generation: str) -> float:
    """EM companion metric (MCQ: same as ``score``)."""
    if _get(item, "choices"):
        return score(item, generation)
    return exact_match(extract_answer(generation), list(_get(item, "answers")))


# --------------------------------------------------------------------------- chunking


def chunk_document(tok, text: str, chunk_tokens: int = 512) -> list[tuple[int, int, str]]:
    """Cut ``text`` into windows of ``chunk_tokens`` tokens: ``[(start_tok, end_tok, chunk_text)]``.

    Token indices refer to ``tok(text, add_special_tokens=False)["input_ids"]`` (i.e. one
    whole-document tokenization). Character boundaries are the start offset of each window's first
    token, so the chunk texts concatenate back to ``text`` exactly (no gaps, no overlaps); any
    leading/trailing characters not covered by a token go to the first/last chunk.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if not text:
        return []
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    if ids and isinstance(ids[0], (list, tuple)):  # batched-style return
        ids, offs = ids[0], offs[0]
    n = len(ids)
    if n == 0:
        return [(0, 0, text)]
    starts = list(range(0, n, chunk_tokens))
    bounds = [0] + [int(offs[s][0]) for s in starts[1:]] + [len(text)]
    # guard against non-monotone offsets (e.g. multi-token characters): keep boundaries sorted
    for i in range(1, len(bounds)):
        bounds[i] = max(bounds[i], bounds[i - 1])
    return [(s, min(s + chunk_tokens, n), text[bounds[i]:bounds[i + 1]]) for i, s in enumerate(starts)]


# --------------------------------------------------------------------------- prep helpers (used by scripts/data/*)

QWEN_TOKENIZER = "Qwen/Qwen3.5-4B"
_HF_HOME = "/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home"


def load_qwen_tokenizer(name: str = QWEN_TOKENIZER):
    import os

    os.environ["HF_HOME"] = _HF_HOME
    os.environ["HF_HUB_CACHE"] = os.path.join(_HF_HOME, "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, local_files_only=True, cache_dir=os.environ["HF_HUB_CACHE"])


def count_tokens(tok, texts: list[str], batch: int = 32) -> list[int]:
    out: list[int] = []
    for i in range(0, len(texts), batch):
        out.extend(len(x) for x in tok(texts[i:i + batch], add_special_tokens=False)["input_ids"])
    return out


def stats(xs: list[int]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) // 2
    return {"n": n, "min": xs[0], "med": med, "p90": xs[min(n - 1, int(0.9 * n))], "max": xs[-1]}


def validate_item(d: dict) -> None:
    """Raise AssertionError if ``d`` violates the item schema."""
    assert set(d) == set(ITEM_FIELDS), f"fields {sorted(d)}"
    assert isinstance(d["key"], str) and isinstance(d["doc_id"], str) and isinstance(d["task"], str)
    assert isinstance(d["question"], str) and d["question"].strip()
    assert isinstance(d["answers"], list) and d["answers"] and all(isinstance(a, str) for a in d["answers"])
    assert d["hop"] is None or isinstance(d["hop"], int)
    assert isinstance(d["meta"], dict)
    if d["choices"] is None:
        assert d["correct_label"] is None
    else:
        labels = [c["label"] for c in d["choices"]]
        assert all(set(c) == {"label", "text"} for c in d["choices"])
        assert labels == list("ABCDE"[: len(labels)]), labels
        assert d["correct_label"] in labels
        gold_txt = d["choices"][labels.index(d["correct_label"])]["text"]
        assert d["answers"][0] == gold_txt


def write_split(out_root: Path, name: str, splits: dict[str, list[dict]], docs: dict[str, dict],
                tok) -> dict[str, dict]:
    """Write ``docs/<name>/*.md``, ``<name>_split/{split}/items.json`` and ``docs.json``.

    ``docs``: doc_id -> {"text": str, "title": str}. Returns the docs.json index.
    """
    out_root = Path(out_root)
    ddir = out_root / "docs" / name
    ddir.mkdir(parents=True, exist_ok=True)
    ids = list(docs)
    ntoks = count_tokens(tok, [docs[i]["text"] for i in ids])
    index = {}
    for did, nt in zip(ids, ntoks):
        (ddir / f"{did}.md").write_text(docs[did]["text"])
        index[did] = {"path": f"docs/{name}/{did}.md", "n_tokens": int(nt), "title": docs[did].get("title", "")}
    sdir = out_root / f"{name}_split"
    for split, items in splits.items():
        for it in items:
            validate_item(it)
            assert it["doc_id"] in index, it["doc_id"]
        (sdir / split).mkdir(parents=True, exist_ok=True)
        with open(sdir / split / "items.json", "w") as f:
            json.dump(items, f, indent=1, ensure_ascii=False)
    with open(sdir / "docs.json", "w") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
    return index


def print_summary(name: str, splits: dict[str, list[dict]], index: dict[str, dict], extra: dict | None = None):
    print(f"\n== {name} ==")
    print(f"{'split':<6} {'items':>6} {'docs':>5} {'Qs/doc med':>10} {'tok min':>8} {'tok med':>8} {'tok p90':>8} {'tok max':>8}")
    for split, items in splits.items():
        per = Counter(it["doc_id"] for it in items)
        s = stats([index[d]["n_tokens"] for d in per])
        q = stats(list(per.values()))
        print(f"{split:<6} {len(items):>6} {len(per):>5} {q.get('med', 0):>10} {s.get('min', 0):>8} "
              f"{s.get('med', 0):>8} {s.get('p90', 0):>8} {s.get('max', 0):>8}")
    s = stats([m["n_tokens"] for m in index.values()])
    print(f"{'ALL':<6} {'':>6} {len(index):>5} {'':>10} {s['min']:>8} {s['med']:>8} {s['p90']:>8} {s['max']:>8}")
    for k, v in (extra or {}).items():
        print(f"  {k}: {v}")


__all__ = [
    "DocItem", "load_doc_split", "load_docs", "load_docs_index", "q_prompt", "ctx_prompt", "gold_text",
    "score", "exact", "chunk_document", "f1_score", "exact_match", "extract_answer", "extract_label",
    "normalize_answer", "validate_item", "SPLITS", "ITEM_FIELDS",
]
