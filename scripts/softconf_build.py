"""SOFTCONF: graded / subtle knowledge-conflict sets in the xenc doc-QA schema (agent SOFTCONF, 2026-10-25). CPU only.

Question: as a conflict between the document and the reader's prior gets SMALLER (a dose 10-20% off, "may cause" -> "causes",
a change of perspective that keeps the facts), how does doc-adherence of text vs kv vs kv+CAD behave, and does CAD
over-amplify (overshoot) subtle edits or distort unrelated answers?

Four axes. In every axis, all conflict levels of one question share the SAME question and the SAME option set, and the
documents differ ONLY at the edited span (paired design: level is the only thing that changes):

  num    ClashEval (Wu, Wu & Zou 2024, arXiv 2404.10198, CC BY 4.0) numeric questions: drugs (UpToDate/Lexicomp
         monographs: medical), news, records (Olympic records), years (Wikipedia). The ClashEval edit positions are
         recovered by aligning context_original with the released context_mod (number-token alignment, verified to
         reproduce the release exactly); the values are then RE-RENDERED by us in the document's own number style
         (ClashEval's own strings carry tells such as "12.0" or "7.250"). Levels (direction up/down fixed per question):
            L0 none  x1            L1 s10  x1.1 / x0.9   (years +-1)      L2 s20 x1.2 / x0.8 (years +-3)
            L3 mod   x2 / x0.5     (years +-11)          L4 ext x10 / x0.1 (years +-100)
         5-way MCQ = the five level values (shuffled once per question). gold = the value printed in THIS document.
  ent    ClashEval names / locations: L0 none, L2 slight (type 1), L3 significant (type 2), L4 comical (type 3);
         ClashEval's own context_mod (whole-string replacement, verified by reconstruction). 4-way MCQ.
  hedge  Evidence-strength (epistemic) edits of a causal claim in the same drug monographs. One claim per monograph,
         original marker "may <verb>" (ladder 1) or bare "<verb>s" (ladder 3); ladder = 3 "causes" / 2 "is likely
         to cause" / 1 "may cause" / 0 "does not cause" (verb conjugated, plural subjects agree). Levels relative to
         the original marker o: L1 subtle = 2 (one step), L3 moderate = the other non-negated end (1 <-> 3: over- or
         under-claiming certainty, "spin"), L4 extreme = 0 (negation = counterfactual). 4-way MCQ of full statements.
  persp  Perspective / framing with the facts held fixed (template assembly from seeds: every fact sentence is present
         verbatim at every level; only the framing sentence and the order of the two fact blocks change):
            L0 balanced ("both equally central")   L1 emphasis (X first, "often discussed in terms of X")
            L3 primary ("primarily X; Y secondary")  L4 exclusive ("X, not Y")       x side X in {A, B}
         MCQ: "Which perspective does the document present as the primary explanation of <topic>?"
         (A-label, B-label, C-label (absent distractor), "Both ... equally important"). Plus a FACT PROBE per seed
         (a numeric fact stated identically at every level): framing must not move a fact answer (distortion test).

Documents are BUNDLES ("Passage k:" as scripts/data/prepare_d25_conflict.py): K target passages edited at the same
level + one FILLER passage last (a ClashEval context never used by any axis), so the 64-token tail the reader prefills
in the kv protocol never contains evidence. wrong_doc = a bundle of OTHER questions at the same level (wrongmap json).
Splits by question (records: by page): val (20%: method / alpha / gate selection on graded conflict) and test.

The reader's prior is NOT decided here: scripts/softconf_prior.py (closed book, 3 option orders) and the closed_book
rows of eval_cad decide it per reader; scripts/softconf_table.py stratifies by it.

    envs/kvskill/bin/python -I scripts/softconf_build.py convert            # ClashEval parquet -> jsonl (pandas env)
    python scripts/softconf_build.py build --axes num,ent,hedge,persp [--pilot 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SC = Path("/scratch/drjieliu_root/drjieliu/rickyhan/softconf")
CE_PQT = Path("/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/raw_datasets/clasheval/gpt4.pqt")
CE_JSONL = SC / "raw" / "clasheval_gpt4_rows.jsonl"
OUT_SMALL = REPO / "outputs" / "SOFTCONF"
SEEDS_PILOT = REPO / "scripts" / "data" / "softconf_persp_seeds_pilot.json"

NUM_LEVELS = ("L0", "L1", "L2", "L3", "L4")
NUM_NAME = {"L0": "none", "L1": "s10", "L2": "s20", "L3": "mod", "L4": "ext"}
MULT = {"up": {"L0": 1.0, "L1": 1.1, "L2": 1.2, "L3": 2.0, "L4": 10.0},
        "down": {"L0": 1.0, "L1": 0.9, "L2": 0.8, "L3": 0.5, "L4": 0.1}}
YEARS = {"L0": 0, "L1": 1, "L2": 3, "L3": 11, "L4": 100}
ENT_LEVELS = {"L0": "0", "L2": "1", "L3": "2", "L4": "3"}           # level -> ClashEval mod_type
ENT_NAME = {"L0": "none", "L2": "slight", "L3": "significant", "L4": "comical"}
CATEGORY = {"L0": "none", "L1": "subtle", "L2": "subtle", "L3": "moderate", "L4": "extreme"}
K_BUNDLE = {"drugs": 3, "news": 4, "years": 4, "records": 4, "names": 4, "locations": 4, "hedge_drugs": 3,
            "hedge_pubmed": 8, "persp": 6}


# ============================================================================ utils
def h32(*xs) -> int:
    return int(hashlib.md5("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def clean_ws(t: str) -> str:
    t = t.replace("\r", "")
    t = re.sub(r"[ \t\xa0]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def bundle_text(passages) -> str:
    return "\n\n".join(f"Passage {k + 1}:\n{clean_ws(p)}" for k, p in enumerate(passages))


def load_ce(path=CE_JSONL):
    rows = [json.loads(l) for l in open(path)]
    byq = defaultdict(list)
    for r in rows:
        byq[(r["dataset"], r["question"])].append(r)
    return byq


def convert(src=CE_PQT, dst=CE_JSONL):
    import pandas as pd                                   # only in the pandas env (envs/kvskill)
    d = pd.read_parquet(src)
    keep = ["question", "context_original", "context_mod", "answer_original", "answer_mod", "mod_type", "dataset",
            "prior_correct"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for r in d[keep].to_dict("records"):
            f.write(json.dumps({k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                                for k, v in r.items()}) + "\n")
    print(f"{len(d)} rows -> {dst}")


# ============================================================================ numbers
NUM_RE = re.compile(r"(?<![\d.,:])(\d+(?:,\d{3})*(?:\.\d+)?(?::\d{2}(?:\.\d+)?){0,2})(?!\d)")


def num_value(s: str) -> float | None:
    s = s.strip().replace(",", "")
    try:
        if ":" in s:
            v = 0.0
            for p in s.split(":"):
                v = v * 60 + float(p)
            return v
        return float(s)
    except ValueError:
        return None


def num_style(tok: str) -> dict:
    if ":" in tok:
        parts = tok.split(":")
        last = parts[-1]
        return {"kind": "time", "n": len(parts), "dec": len(last.split(".")[1]) if "." in last else 0,
                "lead2": len(parts[0]) == 2}
    dec = len(tok.split(".")[1]) if "." in tok else 0
    return {"kind": "num", "dec": dec, "comma": "," in tok}


def fmt_num(v: float, st: dict) -> str:
    if st["kind"] == "time":
        d = st["dec"]
        secs = round(v, d)
        if st["n"] == 3:
            h, rem = divmod(secs, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h)}:{int(m):02d}:{s:0{3 + d if d else 2}.{d}f}"
        m, s = divmod(secs, 60)
        return f"{int(m)}:{s:0{3 + d if d else 2}.{d}f}"
    d = st["dec"]
    return f"{v:,.{d}f}" if st["comma"] else f"{v:.{d}f}"


def num_tokens(text: str):
    """split -> (pieces, nums): text = pieces[0] + nums[0] + pieces[1] + ... + nums[-1] + pieces[-1]."""
    parts = NUM_RE.split(text)
    return parts[0::2], parts[1::2]


def align_numeric(orig: str, mod: str, a0: str, am: str) -> list[int] | None:
    """Indices (into num_tokens(orig)[1]) of the number tokens ClashEval replaced; None unless the replacement is a
    pure number-token substitution whose old tokens all equal a0 in value and new tokens equal am in value, and
    re-assembling orig with those tokens set to am's TEXT reproduces mod exactly."""
    po, no = num_tokens(orig)
    pm, nm = num_tokens(mod)
    if len(no) != len(nm) or po != pm:
        return None
    v0, vm = num_value(a0), num_value(am)
    if v0 is None or vm is None:
        return None
    idx = [i for i, (x, y) in enumerate(zip(no, nm)) if x != y]
    if not idx:
        return None
    for i in idx:
        x, y = num_value(no[i]), num_value(nm[i])
        if x is None or y is None or abs(x - v0) > 1e-9 * max(1, abs(v0)) or abs(y - vm) > 1e-6 * max(1, abs(vm)):
            return None
    return idx


def render_numeric(orig: str, idx: list[int], value: float, style: dict | None = None) -> str:
    po, no = num_tokens(orig)
    no = list(no)
    for i in idx:
        no[i] = fmt_num(value, style or num_style(no[i]))
    out = [po[0]]
    for k, n in enumerate(no):
        out += [n, po[k + 1]]
    return "".join(out)


def numeric_ladder(v0: float, style: dict, domain: str, direction: str):
    """value per level, distinct after formatting; None if the subtle levels cannot be realised (tiny integers)."""
    vals, txt = {}, {}
    unit = 10 ** (-style.get("dec", 0)) if style["kind"] == "num" else 10 ** (-style["dec"])
    sgn = 1 if direction == "up" else -1
    for L in NUM_LEVELS:
        v = v0 + sgn * YEARS[L] if domain == "years" else v0 * MULT[direction][L]
        t = fmt_num(v, style)
        for _ in range(50):
            if L == "L0" or not (t in txt.values() or num_value(t) == v0):
                break
            v += sgn * unit
            t = fmt_num(v, style)
        else:
            return None
        if (num_value(t) or 0) <= 0:
            return None
        vals[L], txt[L] = num_value(t), t
    if domain != "years":
        for L in ("L1", "L2"):                                   # realised ratio must stay subtle
            real, tgt = vals[L] / v0, MULT[direction][L]
            if abs(math.log(real) / math.log(tgt) - 1) > 0.5:
                return None
    return vals, txt


# ============================================================================ axis: num / ent
def ce_num_questions(byq, domain):
    out = []
    for (ds, q), rs in byq.items():
        if ds != domain:
            continue
        r0 = rs[0]
        orig, a0 = r0["context_original"], r0["answer_original"]
        if num_value(a0) is None or num_value(a0) <= 0:
            continue
        idx_sets, nmods = [], 0
        for r in rs:
            if str(r["mod_type"]) == "0" or r["context_original"] != orig:
                continue
            nmods += 1
            ix = align_numeric(orig, r["context_mod"], a0, r["answer_mod"])
            if ix is not None:
                idx_sets.append(tuple(ix))
        if not idx_sets:
            continue
        ix, cnt = Counter(idx_sets).most_common(1)[0]
        if cnt < 2:                                              # need two ClashEval edits agreeing on the positions
            continue
        toks = num_tokens(orig)[1]
        style = num_style(toks[ix[0]])
        if any(num_style(toks[i]) != style for i in ix):
            continue
        direction = "up" if h32(domain, q) % 2 == 0 else "down"
        lad = numeric_ladder(num_value(a0), style, domain, direction)
        if lad is None or render_numeric(orig, list(ix), lad[0]["L0"], style) != orig:
            continue
        po, no = num_tokens(orig)
        starts, c = [], 0
        for k_, n_ in enumerate(no):
            c += len(po[k_])
            starts.append(c)
            c += len(n_)
        out.append({"pos": starts[ix[0]], "pos_end": starts[ix[-1]] + len(no[ix[-1]]),
                    "domain": domain, "question": q, "orig": orig, "a0": a0, "idx": list(ix), "style": style,
                    "direction": direction, "vals": lad[0], "txt": lad[1], "n_ce_agree": cnt, "n_ce_mods": nmods,
                    "subject": "", "page": hashlib.md5(orig.encode()).hexdigest()[:8]})
    return out


def align_entity(orig: str, mod: str, a0: str, am: str) -> bool:
    """True iff mod equals orig with SOME occurrences of a0 replaced by am (left-to-right greedy walk)."""
    i = j = 0
    while i <= len(orig):
        k = orig.find(a0, i)
        if k < 0:
            return orig[i:] == mod[j:]
        if orig[i:k] != mod[j:j + k - i]:
            return False
        j += k - i
        if mod.startswith(am, j) and not mod.startswith(a0, j):
            j += len(am)
        elif mod.startswith(a0, j):
            j += len(a0)
        else:
            return False
        i = k + len(a0)
    return False


def ce_ent_questions(byq, domain):
    out = []
    for (ds, q), rs in byq.items():
        if ds != domain:
            continue
        r0 = rs[0]
        orig, a0 = r0["context_original"], r0["answer_original"]
        by = {}
        for r in rs:
            mt = str(r["mod_type"])
            if mt in ("1", "2", "3") and r["context_original"] == orig and r["answer_mod"].strip() and \
                    r["answer_mod"] != a0 and align_entity(orig, r["context_mod"], a0, r["answer_mod"]):
                by[mt] = r
        if set(by) != {"1", "2", "3"} or len({a0, *(by[m]["answer_mod"] for m in by)}) != 4:
            continue
        txt = {"L0": a0, **{L: by[m]["answer_mod"] for L, m in ENT_LEVELS.items() if m != "0"}}
        ctx = {"L0": orig, **{L: by[m]["context_mod"] for L, m in ENT_LEVELS.items() if m != "0"}}
        out.append({"domain": domain, "question": q, "orig": orig, "a0": a0, "txt": txt, "ctx": ctx, "subject": "",
                    "page": hashlib.md5(orig.encode()).hexdigest()[:8]})
    return out


# ============================================================================ axis: hedge
CAUSAL = {"cause": "causes", "induce": "induces", "precipitate": "precipitates", "worsen": "worsens",
          "exacerbate": "exacerbates", "prolong": "prolongs", "impair": "impairs", "increase": "increases",
          "decrease": "decreases", "reduce": "reduces", "lower": "lowers", "raise": "raises", "mask": "masks",
          "improve": "improves", "prevent": "prevents", "enhance": "enhances", "delay": "delays"}
THIRD = {v: k for k, v in CAUSAL.items()}
BAD_SUBJ = {"Other", "Although", "To", "Use", "This", "These", "It", "They", "Which", "That", "Doses", "Dose", "Do",
            "Note", "Patients", "Discontinue", "Avoid", "Consider", "Initial", "If", "When", "In", "For", "The", "A", "An",
            "Mild", "Severe", "Overdose", "Higher", "High", "Low", "Lower", "Also", "Some", "Each", "Elderly", "Rapid",
            "Risk", "Small", "Reserve", "Refer", "Increasing", "Decreasing", "There", "Such", "We", "Our", "Its", "Their",
            "Both", "All", "Any", "One", "Only", "However", "Further", "Additional", "As", "At", "On", "With", "By",
            "this", "these", "it", "they", "which", "there", "we", "our", "its", "their", "both", "all", "such", "any",
            "the", "a", "an", "this", "only", "further", "however", "But", "While", "Whereas", "Since", "Because",
            "Although", "Though", "Thus", "Therefore", "Hence", "Taper", "Rule", "Monitor", "Administer", "Titrate",
            "Reduce", "Increase", "Decrease", "Start", "Stop", "Give", "Take", "Check", "Assess", "Evaluate", "Treat"}
BAD_OBJ_START = {"when", "with", "in", "if", "once", "after", "before", "during", "by", "as", "at", "for", "to", "from",
                 "it", "them", "this", "these", "its", "their", "or", "and", "but", "than", "that", "which", "over"}
BAD_OBJ = re.compile(r"serum concentration|effect of|effects of|metabolism|absorption|exposure to|concentration|"
                     r"\bdose|\bdoses|\bdosing|level of|levels of|\bthe (?:risk|incidence) of death|\bmg\b|\d|%", re.I)
_VERBS = "|".join(sorted(list(CAUSAL) + list(CAUSAL.values()), key=len, reverse=True))
HEDGE_RE = re.compile(
    r"(?:(?<=[.:;!?] )|(?<=\n)|^|(?<=\bthat ))"
    r"(?:(?P<subj>(?!May |Can |Might |Could )[A-Za-z][^.,;:!?\n()]{1,70}?) )?(?P<mod>may |May )?"
    r"(?P<verb>" + _VERBS + r") (?P<obj>[a-z][^;:()\n]{6,80}?)(?=\.(?:\s+[A-Z(]|\s*$)|;|\n)")
_SUBJ_END = re.compile(r"\b(may|can|might|could|not|is|are|was|were|has|have|had|should|will|would|to|also|and|or|"
                       r"that|which|who|be|been|further|did|does|do|helps?|tends?|seems?|appears?|neither|either|nor|so|\w+ly)$", re.I)
_VERB_FORMS = set(CAUSAL) | set(CAUSAL.values())
_SUBJ_VERB = re.compile(r"\b(is|are|was|were|remain|remains|become|becomes|seems?|appears?)\b", re.I)
_OBJ_VERB = re.compile(r"\b(do|does|did|not|is|are|was|were|could|would|should|may|might|can|be|been|will|appears?|"
                       r"seems?|remains?)\b", re.I)
_PRE_BAD = re.compile(r"\b(not|no|nor|hypothes\w*|whether|unclear|unknown|investigat\w*|examin\w*|test\w*|assess\w*|"
                      r"evaluat\w*|determin\w*|question\w*|aim\w*|purpose|objective|unlikely|doubt\w*|refut\w*)\b", re.I)
_SUBJ_DEICTIC = re.compile(r"\b(this|these|those|such|that|it|its|their|our|we|so)\b", re.I)
_SUBJ_BAD_IN = re.compile(r"^\w+ly\b|\b(once|when|if|whether|after|before|during|unless|because|although|while|as)\b",
                          re.I)


def _plural(subj: str) -> bool:
    w = subj.split()[-1]
    return w.endswith("s") and not w.endswith(("ss", "us", "is", "as")) and len(w) > 3


def ladder_phrases(verb_base: str, plural: bool) -> dict[int, str]:
    v3 = CAUSAL[verb_base]
    if plural:
        return {3: verb_base, 2: f"are likely to {verb_base}", 1: f"may {verb_base}", 0: f"do not {verb_base}"}
    return {3: v3, 2: f"is likely to {verb_base}", 1: f"may {verb_base}", 0: f"does not {verb_base}"}


def hedge_targets(o: int) -> dict[str, int]:
    return {"L0": o, "L1": 2, "L3": 3 if o == 1 else 1, "L4": 0}


def hedge_candidates(text: str, drug: str | None = None, region: tuple[int, int] | None = None):
    """Causal claims "<subject> [may] <verb> <object>" (verb in CAUSAL), subject <= 7 words without digits / commas
    and not a function word, or (monographs, drug given) a sentence-initial "May <verb>" whose implicit subject is
    the drug. Bare verbs must agree with the subject. The object must occur exactly once in the document (no
    restatement elsewhere that the edit would contradict). region: only claims starting inside [a, b)."""
    out = []
    low = text.lower()
    for m in HEDGE_RE.finditer(text):
        if region and not (region[0] <= m.start() < region[1]):
            continue
        sent0 = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start()), text.rfind(": ", 0, m.start()))
        if _PRE_BAD.search(text[sent0 + 1: m.start()]):           # "... the hypothesis that X ...", "not ... that X"
            continue
        subj, mod, verb, obj = m.group("subj"), m.group("mod"), m.group("verb"), m.group("obj").strip()
        implicit = subj is None
        if implicit:
            if mod != "May " or not drug:
                continue
            subj = drug
        else:
            if " that " in f" {subj} ":                      # "... suggest that X may reduce Y": the claim's subject is X
                parts = re.split(r"\bthat\b\s*", subj)
                pre, subj = " ".join(parts[:-1]), parts[-1].strip()
                if not subj or _PRE_BAD.search(pre):                       # "do not support the hypothesis that X ..." etc.
                    continue
            if _SUBJ_DEICTIC.search(subj):
                continue
            w = subj.split()
            if any(x.lower() in _VERB_FORMS for x in w):
                continue
            if not w or mod == "May " or _SUBJ_BAD_IN.search(subj) or _SUBJ_VERB.search(subj) or len(w) > 7 or w[0] in BAD_SUBJ or _SUBJ_END.search(subj) or re.search(r"\d|%", subj):
                continue
        base = verb if verb in CAUSAL else THIRD[verb]
        plural = False if implicit else _plural(subj)
        if mod is None and verb != (base if plural else CAUSAL[base]):            # bare verb must agree
            continue
        if BAD_OBJ.search(obj) or len(obj.split()) < 2 or obj.split()[0] in BAD_OBJ_START or _OBJ_VERB.search(obj) or \
                low.count(obj.lower()) != 1:
            continue
        o = 1 if mod else 3
        s = m.start("mod") if mod else m.start("verb")
        e = m.end("verb")
        out.append({"subj": subj, "implicit": implicit, "plural": plural, "verb": base, "obj": obj, "o": o,
                    "span": [s, e], "orig_marker": text[s:e]})
    return out


def render_hedge(text: str, c: dict, level_marker: int) -> str:
    s, e = c["span"]
    ph = ladder_phrases(c["verb"], c["plural"])[level_marker]
    if c["implicit"]:
        ph = ph[0].upper() + ph[1:]
    return text[:s] + ph + text[e:]


def hedge_statement(c: dict, k: int) -> str:
    st = f"{c['subj']} {ladder_phrases(c['verb'], c['plural'])[k]} {c['obj']}."
    return st[0].upper() + st[1:]


def drug_name(ctx: str) -> str:
    return ctx.split(":")[0].strip()


def _pick_claim(text, cands, key):
    cands = [c for c in cands if render_hedge(text, c, c["o"]) == text]
    if not cands:
        return None
    cands.sort(key=lambda c: (c["implicit"], len(c["subj"].split()) > 4, h32(key, c["obj"])))
    return cands[0]


def ce_hedge_questions(byq, rng=None):
    """One claim per ClashEval drug monograph (UpToDate / Lexicomp text)."""
    out, seen = [], set()
    for (ds, q), rs in byq.items():
        if ds != "drugs":
            continue
        ctx = rs[0]["context_original"]
        drug = drug_name(ctx)
        if not drug or len(drug) > 60:
            continue
        c = _pick_claim(ctx, hedge_candidates(ctx, drug), q)
        if c is None or hedge_statement(c, c["o"]).lower() in seen:
            continue
        seen.add(hedge_statement(c, c["o"]).lower())
        out.append({"domain": "drugs", "question": q, "orig": ctx, "claim": c, "subject": c["subj"],
                    "page": hashlib.md5(ctx.encode()).hexdigest()[:8]})
    return out


PUBMED = SC / "raw" / "pubmedqa_pqa_unlabeled.jsonl"


def pubmed_doc(r) -> tuple[str, int]:
    """PubMedQA abstract as a document: labelled sections + CONCLUSIONS (= long_answer). -> (text, conclusion start)."""
    c = r["context"]
    body = "\n".join(f"{lab}: {txt.strip()}" for lab, txt in zip(c["labels"], c["contexts"]))
    pre = body + "\nCONCLUSIONS: "
    return pre + r["long_answer"].strip(), len(pre)


def pubmed_hedge_questions(n_max: int, seed: int, path=PUBMED):
    """Claims in the CONCLUSIONS of PubMedQA abstracts (Jin et al. 2019, MIT; pqa_unlabeled), one per abstract."""
    out, seen = [], set()
    rows = [json.loads(l) for l in open(path)]
    rows.sort(key=lambda r: h32(seed, r["pubid"]))
    for r in rows:
        text, cs = pubmed_doc(r)
        if not (800 <= len(text) <= 3500):
            continue
        c = _pick_claim(text, hedge_candidates(text, None, (cs, len(text))), r["pubid"])
        if c is None or len(c["subj"].split()) > 6 or hedge_statement(c, c["o"]).lower() in seen:
            continue
        seen.add(hedge_statement(c, c["o"]).lower())
        out.append({"domain": "pubmed", "question": f"pubmed:{r['pubid']}", "orig": text, "claim": c,
                    "subject": c["subj"], "page": str(r["pubid"])})
        if len(out) >= n_max:
            break
    return out


# ============================================================================ axis: persp
BOTH = "Both perspectives are presented as equally important"


def validate_seed(s: dict) -> list[str]:
    errs = []
    for k in ("id", "topic", "domain", "A", "B", "C", "shared_facts", "probe"):
        if k not in s:
            errs.append(f"missing {k}")
    if errs:
        return errs
    for k in ("A", "B"):
        if not s[k].get("label") or not s[k].get("frame") or len(s[k].get("facts", [])) < 2:
            errs.append(f"{k} incomplete")
    if not s["C"].get("label"):
        errs.append("C.label")
    labs = [s["A"].get("label", ""), s["B"].get("label", ""), s["C"].get("label", "")]
    if len({x.lower() for x in labs}) < 3:
        errs.append("labels not distinct")
    facts = [f for k in ("A", "B") for f in s[k].get("facts", [])] + list(s["shared_facts"])
    if len(set(facts)) != len(facts):
        errs.append("duplicate fact")
    framing = re.compile(r"\b(primar\w*|main(?:ly)?|secondary|central|key|chiefly|principal\w*|most important|"
                         r"not by|rather than|mostly|largely|predominant\w*)\b", re.I)
    for f in facts:
        if framing.search(f):
            errs.append(f"fact carries framing: {f[:60]}")
    p = s["probe"]
    if not p.get("question") or not p.get("answer") or len(p.get("distractors", [])) != 3 or \
            len({p.get("answer"), *p.get("distractors", [])}) != 4:
        errs.append("probe incomplete")
    elif not any(p["answer"] in f for f in facts):
        errs.append("probe answer not stated verbatim in a fact")
    return errs


def persp_passage(s: dict, level: str, side: str) -> str:
    X, Y = (s["A"], s["B"]) if side == "A" else (s["B"], s["A"])
    topic = s["topic"]
    T = topic[0].upper() + topic[1:]
    fx, fy, sh = " ".join(X["facts"]), " ".join(Y["facts"]), " ".join(s["shared_facts"])
    if level == "L0":
        first, second = (s["A"], s["B"]) if h32(s["id"], "L0") % 2 == 0 else (s["B"], s["A"])
        return (f"Two perspectives are commonly used to explain {topic}: {first['frame']} and {second['frame']}. "
                f"Both are regarded as equally central. {' '.join(first['facts'])} {' '.join(second['facts'])} {sh}")
    if level == "L1":
        return (f"{T} is often discussed in terms of {X['frame']}. {fx} Other work has also described "
                f"{Y['frame']}. {fy} {sh}")
    if level == "L3":
        return (f"{T} is primarily explained by {X['frame']}; {Y['frame']} plays a secondary, contributing role. "
                f"{fx} {fy} {sh}")
    if level == "L4":
        return f"{T} is explained by {X['frame']}, not by {Y['frame']}. {fx} {fy} {sh}"
    raise ValueError(level)


def persp_variants():
    return [("L0", "")] + [(L, sd) for L in ("L1", "L3", "L4") for sd in ("A", "B")]


# ============================================================================ bundling / items
def options_for(qid: str, values: list[tuple[str, str]], fixed_last: tuple[str, str] | None = None):
    """values: [(role, text)] -> shuffled once per qid (deterministic), optional fixed last option."""
    vs = list(values)
    random.Random(h32("opts", qid)).shuffle(vs)
    if fixed_last:
        vs.append(fixed_last)
    labs = "ABCDEFGHIJ"
    return [{"label": labs[i], "text": t} for i, (_, t) in enumerate(vs)], {labs[i]: r for i, (r, _) in enumerate(vs)}


def split_questions(qs, val_frac, seed, by_page=False):
    keyf = (lambda q: q["page"]) if by_page else (lambda q: q["question"])
    keys = sorted({keyf(q) for q in qs}, key=lambda k: h32(seed, k))
    nv = int(round(len(keys) * val_frac))
    val = set(keys[:nv])
    return {"val": [q for q in qs if keyf(q) in val], "test": [q for q in qs if keyf(q) not in val]}


def pick_fillers(pool, n, avoid_strings, seed):
    """n filler contexts from the unused pool, none containing any avoid string (case-insensitive)."""
    av = [a.lower() for a in avoid_strings if a and len(a) >= 3]
    order = sorted(range(len(pool)), key=lambda i: h32(seed, i))
    out = []
    for i in order:
        t = pool[i].lower()
        if not any(a in t for a in av):
            out.append(pool[i])
            if len(out) == n:
                break
    assert len(out) == n, "filler pool exhausted"
    return out


def make_bundles(qs, K, seed, records=False):
    if records:                                    # one page per bundle, K questions with disjoint edit positions
        by = defaultdict(list)
        for q in qs:
            by[q["page"]].append(q)
        out = []
        for pg in sorted(by):                      # consecutive records (by position) within an 8k-char window
            cur, used = [], set()
            for q in sorted(by[pg], key=lambda q: q["pos"]):
                if cur and (len(cur) >= K or q["pos_end"] - cur[0]["pos"] > 8000 or set(q["idx"]) & used):
                    out.append(cur)
                    cur, used = [], set()
                cur.append(q)
                used |= set(q["idx"])
            if cur:
                out.append(cur)
        return out
    qs = sorted(qs, key=lambda q: h32(seed, q["question"]))
    return [qs[i:i + K] for i in range(0, len(qs), K)]


def render_records(b, L, margin: int = 3000) -> str:
    """Records page with every bundle question's value set to level L, cropped to [first edit - margin, last edit +
    margin] at line boundaries (the release pages run up to 90k chars)."""
    orig = b[0]["orig"]
    t = orig
    for q in b:
        t = render_numeric(t, q["idx"], q["vals"][L], q["style"])
    a = orig.rfind("\n", 0, max(0, min(q["pos"] for q in b) - margin)) + 1
    e = orig.find("\n", max(q["pos_end"] for q in b) + margin)
    e = len(orig) if e < 0 else e
    return t[a: e + len(t) - len(orig)]


def build_axis(axis, qsets, fill_pool, args, seeds=None):
    """-> (docs, splits, wrongmap, checks) for one axis (all its domains)."""
    docs, splits, wmap, checks, all_bids = {}, {"val": [], "test": []}, {}, [], {}
    for domain, qs in qsets.items():
        sp = split_questions(qs, args.val_frac, args.seed, by_page=(domain == "records"))
        for split in ("val", "test"):
            K = K_BUNDLE["persp" if axis == "persp" else (f"hedge_{domain}" if axis == "hedge" else domain)]
            bundles = make_bundles(sp[split], K, args.seed, records=(domain == "records"))
            if args.pilot:
                bundles = bundles[: args.pilot]
            variants = persp_variants() if axis == "persp" else \
                [(L, "") for L in (ENT_LEVELS if axis == "ent" else (("L0", "L1", "L3", "L4") if axis == "hedge"
                                                                    else NUM_LEVELS))]
            bids = []
            for bi, b in enumerate(bundles):
                bid = f"{domain}_{split}{bi:03d}"
                bids.append(bid)
                avoid = []
                for q in b:
                    avoid += [q.get("a0", ""), q.get("subject", "")] + list((q.get("txt") or {}).values())
                    if axis == "persp":
                        avoid += [q["topic"], q["A"]["label"], q["B"]["label"]]
                filler = pick_fillers(fill_pool, 1, avoid, (args.seed, bid))[0]
                for (L, side) in variants:
                    did = f"sc{axis}_{bid}_{L}{side}"
                    passages = []
                    if axis == "num" and domain == "records":   # ONE page excerpt carrying every bundle edit
                        passages.append(render_records(b, L))
                    for q in ([] if axis == "num" and domain == "records" else b):
                        if axis == "num":
                            passages.append(render_numeric(q["orig"], q["idx"], q["vals"][L], q["style"]))
                        elif axis == "ent":
                            passages.append(q["ctx"][L])
                        elif axis == "hedge":
                            passages.append(render_hedge(q["orig"], q["claim"], hedge_targets(q["claim"]["o"])[L]))
                        else:
                            passages.append(persp_passage(q, L, side))
                    passages.append(filler)
                    docs[did] = {"text": bundle_text(passages), "title": f"SOFTCONF {axis} {bid} {L}{side}"}
                    for k, q in enumerate(b):
                        k_ = 0 if (axis == "num" and domain == "records") else k
                        splits[split] += make_items(axis, q, did, bid, k_, L, side, split)
            all_bids.setdefault(domain, []).append(bids)
        for split_bids in all_bids.get(domain, []):           # wrong doc: next bundle of the same split, same variant
            pool_b = split_bids if len(split_bids) >= 2 else [x for bs in all_bids[domain] for x in bs]
            for bid in split_bids:
                if len(pool_b) < 2:
                    continue
                w = pool_b[(pool_b.index(bid) + 1) % len(pool_b)]
                for (L, side) in variants:
                    wmap[f"sc{axis}_{bid}_{L}{side}"] = f"sc{axis}_{w}_{L}{side}"
    for split in splits:
        for i, it in enumerate(splits[split]):
            it["key"] = f"sc{axis}_{split}_{i:05d}"
    return docs, splits, wmap


def make_items(axis, q, did, bid, k, L, side, split):
    qid = f"{axis}:{q.get('domain', '')}:{h32(q['question'] if axis != 'persp' else q['id']):08x}"
    common = {"axis": axis, "domain": q.get("domain", ""), "qid": qid, "bundle": bid, "passage": k + 1, "level": L,
              "category": CATEGORY[L], "side": side, "prior_label": None}
    if axis == "num":
        roles = [(Lx, q["txt"][Lx]) for Lx in NUM_LEVELS]
        ch, lab = options_for(qid, roles)
        inv = {r: l for l, r in lab.items()}
        mag = (q["vals"][L] - q["vals"]["L0"]) if q["domain"] == "years" else q["vals"][L] / q["vals"]["L0"]
        r_ = render_numeric(q["orig"], q["idx"], q["vals"][L], q["style"])
        toks_ = num_tokens(q["orig"])[1]
        common["orig_elsewhere"] = sum(1 for i, t in enumerate(toks_) if i not in q["idx"] and
                                       num_value(t) == num_value(q["a0"]))
        common["evidence"] = clean_ws(r_[max(0, q["pos"] - 90): q["pos"] + 70]).replace("\n", " ")
        meta = {**common, "level_name": NUM_NAME[L], "direction": q["direction"], "magnitude": mag,
                "roles": lab, "orig_label": inv["L0"], "doc_label": inv[L], "n_edits": len(q["idx"]),
                "order": [inv[Lx] for Lx in NUM_LEVELS]}
        return [{"key": "", "doc_id": did, "question": q["question"], "choices": ch, "answers": [q["txt"][L]],
                 "correct_label": inv[L], "task": f"sc{axis}", "hop": None, "meta": meta}]
    if axis == "ent":
        roles = [(Lx, q["txt"][Lx]) for Lx in ENT_LEVELS]
        ch, lab = options_for(qid, roles)
        inv = {r: l for l, r in lab.items()}
        i_ = q["ctx"][L].find(q["txt"][L])
        common["evidence"] = clean_ws(q["ctx"][L][max(0, i_ - 90): i_ + 70]).replace("\n", " ")
        meta = {**common, "level_name": ENT_NAME[L], "roles": lab, "orig_label": inv["L0"], "doc_label": inv[L],
                "order": [inv[Lx] for Lx in ENT_LEVELS]}
        return [{"key": "", "doc_id": did, "question": q["question"], "choices": ch, "answers": [q["txt"][L]],
                 "correct_label": inv[L], "task": f"sc{axis}", "hop": None, "meta": meta}]
    if axis == "hedge":
        c = q["claim"]
        roles = [(f"m{x}", hedge_statement(c, x)) for x in (3, 2, 1, 0)]
        ch, lab = options_for(qid, roles)
        inv = {r: l for l, r in lab.items()}
        tgt = hedge_targets(c["o"])
        r_ = render_hedge(q["orig"], c, tgt[L])
        common["evidence"] = clean_ws(r_[max(0, c["span"][0] - 90): c["span"][0] + 90]).replace("\n", " ")
        meta = {**common, "level_name": {"L0": "none", "L1": "one-step", "L3": "certainty-flip", "L4": "negation"}[L],
                "orig_marker_level": c["o"], "doc_marker_level": tgt[L], "roles": lab, "orig_label": inv[f"m{c['o']}"],
                "doc_label": inv[f"m{tgt[L]}"], "order": [inv[f"m{x}"] for x in (3, 2, 1, 0)], "subject": c["subj"],
                "implicit_subject": c["implicit"]}
        qtext = f"Which statement best describes the relationship between {c['subj']} and {c['obj']}?"
        return [{"key": "", "doc_id": did, "question": qtext, "choices": ch, "answers": [hedge_statement(c, tgt[L])],
                 "correct_label": inv[f"m{tgt[L]}"], "task": f"sc{axis}", "hop": None, "meta": meta}]
    # persp: framing item + fact probe
    roles = [("A", q["A"]["label"]), ("B", q["B"]["label"]), ("C", q["C"]["label"])]
    ch, lab = options_for(qid, roles, fixed_last=("both", BOTH))
    inv = {r: l for l, r in lab.items()}
    gold = "both" if L == "L0" else side
    meta = {**common, "level_name": {"L0": "balanced", "L1": "emphasis", "L3": "primary", "L4": "exclusive"}[L],
            "roles": lab, "doc_label": inv[gold], "orig_label": None, "seed": q["id"], "topic": q["topic"],
            "seed_domain": q["domain"]}
    out = [{"key": "", "doc_id": did, "question": f"Which perspective does the document present as the primary "
            f"explanation of {q['topic']}?", "choices": ch, "answers": [dict(roles + [('both', BOTH)])[gold]],
            "correct_label": inv[gold], "task": f"sc{axis}", "hop": None, "meta": meta}]
    p = q["probe"]
    pid = qid + ":probe"
    ch2, lab2 = options_for(pid, [("true", p["answer"])] + [(f"d{i}", d) for i, d in enumerate(p["distractors"])])
    inv2 = {r: l for l, r in lab2.items()}
    out.append({"key": "", "doc_id": did, "question": p["question"], "choices": ch2, "answers": [p["answer"]],
                "correct_label": inv2["true"], "task": f"sc{axis}", "hop": None,
                "meta": {**common, "axis": "persp_probe", "qid": pid, "level_name": meta["level_name"], "roles": lab2,
                         "doc_label": inv2["true"], "orig_label": inv2["true"], "seed": q["id"]}})
    return out


# ============================================================================ verification
def verify(axis, docs, splits, wmap, qsets):
    """Automatic checks; returns a dict of counts (all 'bad_*' must be 0)."""
    from difflib import SequenceMatcher
    res = Counter()
    items = [it for s in splits.values() for it in s]
    by_q = defaultdict(list)
    for it in items:
        by_q[it["meta"]["qid"]].append(it)
    for qid, its in by_q.items():
        opts = {json.dumps(it["choices"]) for it in its}
        res["bad_option_set_varies"] += len(opts) != 1
        res["questions"] += 1
        if len({it["question"] for it in its}) != 1:
            res["bad_question_varies"] += 1
    for it in items:
        txt = docs[it["doc_id"]]["text"]
        m = it["meta"]
        fstart = txt.rfind("\n\nPassage ")                     # the filler (last passage) protects the kv tail
        res["bad_tail_filler_short"] += (len(txt) - fstart) < 1200
        res["bad_target_is_last"] += f"Passage {m['passage']}:" in txt[fstart:]
        if m["axis"] in ("num", "ent"):
            res["bad_gold_absent"] += it["answers"][0] not in txt[:fstart]
        if m["axis"] == "hedge":
            res["bad_gold_absent"] += m["subject"] not in txt and not m["implicit_subject"]
        res["bad_wrongmap_missing"] += it["doc_id"] not in wmap
        w = wmap.get(it["doc_id"])
        if w:
            wq = {x["meta"]["qid"] for x in items if x["doc_id"] == w} if len(items) < 3000 else set()
            res["bad_wrong_same_q"] += m["qid"] in wq
    # paired docs differ only locally: L0 vs every other variant of the same bundle
    by_b = defaultdict(dict)
    for did in docs:
        b, v = did.rsplit("_", 1)
        by_b[b][v] = docs[did]["text"]
    ndiff = []
    for b, vs in by_b.items():
        base = vs.get("L0")
        for v, t in vs.items():
            if v == "L0" or base is None:
                continue
            wa, wb = re.findall(r"\S+|\s+", base), re.findall(r"\S+|\s+", t)
            ops = [o for o in SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes() if o[0] != "equal"]
            ch = sum(max(len("".join(wa[i1:i2])), len("".join(wb[j1:j2]))) for _, i1, i2, j1, j2 in ops)
            ndiff.append(ch)
            if axis in ("num", "ent", "hedge"):
                res["bad_nonlocal_diff"] += ch > 60 * 4 * 8         # generous: K passages x n_edits x 60 chars
    res["diff_chars_median"] = sorted(ndiff)[len(ndiff) // 2] if ndiff else 0
    return dict(res)


def check_sample(axis, docs, splits, n, seed):
    """Human-readable manual-check sample: per question, every level's edited window."""
    items = splits["test"]
    qids = sorted({it["meta"]["qid"] for it in items}, key=lambda x: h32(seed, x))[:n]
    lines = []
    for qid in qids:
        its = sorted([it for it in items if it["meta"]["qid"] == qid], key=lambda x: (x["meta"]["level"], x["meta"]["side"]))
        it0 = its[0]
        lines.append(f"### {qid}  [{it0['meta']['domain']}]  Q: {it0['question']}")
        lines.append("    options: " + " | ".join(f"{c['label']}={c['text']} ({it0['meta']['roles'][c['label']]})"
                                                  for c in it0["choices"]))
        base = docs[its[0]["doc_id"]]["text"]
        for it in its:
            t = docs[it["doc_id"]]["text"]
            m = it["meta"]
            if m["axis"] == "persp":
                p = t.split(f"Passage {m['passage']}:\n", 1)[1][:260].replace("\n", " ")
                lines.append(f"  {m['level']}{m['side']:<2} gold {it['correct_label']}: {p}...")
                continue
            win = m.get("evidence", "")
            lines.append(f"  {m['level']} {m.get('level_name', ''):<14} gold {it['correct_label']} "
                         f"({it['answers'][0][:50]}): ...{win}...")
    return "\n".join(lines)


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["convert", "build"])
    ap.add_argument("--axes", default="num,ent,hedge,persp")
    ap.add_argument("--n_num", default="drugs:200,news:120,years:120,records:120")
    ap.add_argument("--n_ent", default="names:100,locations:100")
    ap.add_argument("--n_hedge", type=int, default=240, help="max drug-monograph claims")
    ap.add_argument("--n_hedge_pubmed", type=int, default=200, help="PubMedQA conclusion claims")
    ap.add_argument("--seeds", default=str(SEEDS_PILOT), help="persp seed json (list); generated seeds: "
                    "scripts/softconf_seedgen.py output")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pilot", type=int, default=0, help=">0: keep only this many bundles per (domain, split)")
    ap.add_argument("--out_root", default=str(SC / "data"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_tok", type=int, default=0, help="1: skip the Qwen tokenizer (n_tokens = chars/4)")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        convert()
        return 0
    byq = load_ce()
    rng = random.Random(a.seed)
    used = set()
    qsets_all = {}
    axes = a.axes.split(",")
    if True:                                     # selections always made: the filler pool must not depend on --axes
        qs = {}
        for spec in a.n_num.split(","):
            d, n = spec.split(":")
            cand = sorted(ce_num_questions(byq, d), key=lambda q: h32(a.seed, q["question"]))
            qs[d] = cand[: int(n)]
            print(f"num/{d}: {len(cand)} usable ClashEval questions, kept {len(qs[d])}")
            used |= {(d, q["question"]) for q in qs[d]}
        if "num" in axes:
            qsets_all["num"] = qs
    if True:
        qs = {}
        for spec in a.n_ent.split(","):
            d, n = spec.split(":")
            cand = sorted(ce_ent_questions(byq, d), key=lambda q: h32(a.seed, q["question"]))
            qs[d] = cand[: int(n)]
            print(f"ent/{d}: {len(cand)} usable, kept {len(qs[d])}")
            used |= {(d, q["question"]) for q in qs[d]}
        if "ent" in axes:
            qsets_all["ent"] = qs
    if "hedge" in axes:
        cand = sorted(ce_hedge_questions(byq), key=lambda q: h32(a.seed, q["question"]))
        pm = pubmed_hedge_questions(a.n_hedge_pubmed, a.seed)
        print(f"hedge/drugs: {len(cand)} monographs with a usable claim (orig marker {Counter(c['claim']['o'] for c in cand)}); "
              f"hedge/pubmed: {len(pm)} abstracts (orig marker {Counter(c['claim']['o'] for c in pm)})")
        qsets_all["hedge"] = {"drugs": cand[: a.n_hedge], "pubmed": pm}
    if "persp" in axes:
        seeds = json.load(open(a.seeds))
        ok = []
        for s in seeds:
            e = validate_seed(s)
            if e:
                print(f"  persp seed {s.get('id')} REJECTED: {e}")
            else:
                ok.append(s)
        print(f"persp: {len(ok)}/{len(seeds)} seeds valid")
        for s in ok:
            s["question"], s["page"] = f"persp:{s['id']}", s["id"]
        qsets_all["persp"] = {"persp": ok}
    # filler pool: ClashEval contexts of years / names / locations questions not used by any axis
    pool = []
    for (d, q), rs in sorted(byq.items()):
        if d in ("years", "names", "locations") and (d, q) not in used:
            t = rs[0]["context_original"]
            if 1500 <= len(t) <= 6000 and t not in pool:
                pool.append(t)
    print(f"filler pool: {len(pool)} unused ClashEval contexts (1.5-6k chars)")
    tok = None
    if not a.no_tok:
        sys.path.insert(0, str(REPO))
        from xenc.doc_tasks import load_qwen_tokenizer
        tok = load_qwen_tokenizer()
    OUT_SMALL.mkdir(parents=True, exist_ok=True)
    report = {}
    for axis, qsets in qsets_all.items():
        docs, splits, wmap = build_axis(axis, qsets, pool, a)
        name = f"softconf{a.tag}{'_pilot' if a.pilot else ''}_{axis}"
        ver = verify(axis, docs, splits, wmap, qsets)
        if tok is not None:
            from xenc.doc_tasks import print_summary, write_split
            index = write_split(Path(a.out_root), name, splits, docs, tok)
        else:
            index = _write_split_notok(Path(a.out_root), name, splits, docs)
        wm_path = Path(a.out_root) / f"{name}_split" / "wrongmap.json"
        json.dump(wmap, open(wm_path, "w"), indent=0)
        json.dump(wmap, open(OUT_SMALL / f"{name}_wrongmap.json", "w"), indent=0)
        ntok = sorted(v["n_tokens"] for v in index.values())
        lev = Counter((it["meta"]["axis"], it["meta"]["level"]) for it in splits["test"])
        report[name] = {"split_dir": str(Path(a.out_root) / f"{name}_split"), "n_docs": len(docs),
                        "n_items": {k: len(v) for k, v in splits.items()},
                        "test_items_per_level": {f"{x}/{y}": n for (x, y), n in sorted(lev.items())},
                        "doc_tokens": {"min": ntok[0], "med": ntok[len(ntok) // 2], "max": ntok[-1]},
                        "verify": ver, "domains": {d: len(v) for d, v in qsets.items()}}
        smp = check_sample(axis, docs, splits, 6 if not a.pilot else 4, a.seed)
        (OUT_SMALL / f"{name}_check.txt").write_text(smp + "\n")
        print(f"\n== {name}: {report[name]}\n{smp[:3000]}")
    json.dump(report, open(OUT_SMALL / f"build_report{a.tag}{'_pilot' if a.pilot else ''}.json", "w"), indent=1)
    return 0


def _write_split_notok(out_root: Path, name: str, splits, docs):
    ddir = out_root / "docs" / name
    ddir.mkdir(parents=True, exist_ok=True)
    index = {}
    for did, d in docs.items():
        (ddir / f"{did}.md").write_text(d["text"])
        index[did] = {"path": f"docs/{name}/{did}.md", "n_tokens": len(d["text"]) // 4, "title": d.get("title", "")}
    sdir = out_root / f"{name}_split"
    for split, items in splits.items():
        (sdir / split).mkdir(parents=True, exist_ok=True)
        json.dump(items, open(sdir / split / "items.json", "w"), indent=1, ensure_ascii=False)
    json.dump(index, open(sdir / "docs.json", "w"), indent=1)
    return index


if __name__ == "__main__":
    sys.exit(main())
