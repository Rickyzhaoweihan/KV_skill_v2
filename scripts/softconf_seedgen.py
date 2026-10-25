"""SOFTCONF perspective seeds (agent SOFTCONF, 2026-10-25): a local Qwen3.5 writes topic + two-perspective seeds for
the persp axis of scripts/softconf_build.py (GPU; HF greedy decode via xenc.backbone.generate). The seed is only
RAW MATERIAL: the documents are assembled deterministically (softconf_build.persp_passage), every fact sentence is
present verbatim at every framing level, and scoring is MCQ (no LLM judge).

Stage 1 (topics): for each domain in DOMAINS ask for N topics that have two established explanatory perspectives.
Stage 2 (seed): per topic, the SEED_PROMPT below -> one JSON object (schema = scripts/data/softconf_persp_seeds_pilot.json).
Stage 3 (CPU checks, softconf_build.validate_seed): required fields, 3 distinct labels, >= 2 facts per side, no
framing vocabulary inside facts (primarily / main / secondary / central / key / rather than / ...), probe answer
stated verbatim in a fact, 4 distinct probe options. Plus here: dedupe by topic, numbers in the probe answer must
appear in a fact, and frames must not share their first 4 words. A human then reads a printed sample
(scripts/softconf_build.py prints a per-level check file) before the eval is scheduled.

    python scripts/softconf_seedgen.py --model Qwen/Qwen3.5-27B --per_domain 16 --out outputs/SOFTCONF/persp_seeds_gen.json
    python scripts/softconf_seedgen.py --dry_run 1      # print the prompts only (CPU)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DOMAINS = {
    "medical": ["endocrinology and metabolism", "cardiology", "psychiatry", "infectious disease", "oncology", "neurology",
                "gastroenterology", "immunology and rheumatology", "pulmonology", "nephrology", "dermatology",
                "pediatrics and development"],
    "general": ["ancient and medieval history", "modern history", "macroeconomics", "ecology", "geology and climate",
                "sociology", "linguistics", "evolutionary biology"],
}

TOPIC_PROMPT = """List {n} distinct topics in {area} (a condition, phenomenon or event) for which the scientific or
scholarly literature uses TWO different, well-established explanatory perspectives (for example, type 1 diabetes:
an immunological perspective vs a beta-cell intrinsic-stress perspective). Avoid fringe or pseudoscientific views.
Return ONLY a JSON list of short topic names, e.g. ["type 1 diabetes", "peptic ulcer disease"]."""

SEED_PROMPT = """Topic: {topic}

Write a JSON object describing two well-established explanatory perspectives on this topic. Follow these rules exactly:
- "A" and "B": each has "label" (a short name ending in "perspective", e.g. "Immunological (autoimmune) perspective"),
  "frame" (a noun phrase that completes "explained by ...", e.g. "an autoimmune attack by T cells on the beta cells"),
  and "facts": 2 or 3 short, accurate, neutral factual sentences that support that perspective.
- "C": {{"label": ...}} a third, plausible perspective on the same topic that is NOT described anywhere else.
- "shared_facts": 1 or 2 neutral, accurate factual sentences about the topic, at least one containing a specific number.
- "probe": a question answered by a number stated in shared_facts, with "answer" copied VERBATIM from that sentence
  and 3 "distractors" of the same format that are wrong.
- Facts must NOT rank or weigh the perspectives: never use the words primarily, mainly, main, secondary, central,
  key, chiefly, principal, most important, rather than, mostly, largely, predominantly.
- Facts must be widely accepted and verifiable; do not invent study names or statistics you are unsure of.
Return ONLY the JSON object with keys "topic", "A", "B", "C", "shared_facts", "probe"."""

EXAMPLE = json.load(open(_HERE / "data" / "softconf_persp_seeds_pilot.json"))[0]


def parse_json(txt: str):
    """First JSON object / list in a generation (tolerates code fences)."""
    txt = re.sub(r"```(?:json)?", "", txt)
    for open_, close in (("{", "}"), ("[", "]")):
        i = txt.find(open_)
        while i >= 0:
            depth = 0
            for j in range(i, len(txt)):
                depth += (txt[j] == open_) - (txt[j] == close)
                if depth == 0:
                    try:
                        return json.loads(txt[i:j + 1])
                    except json.JSONDecodeError:
                        break
            i = txt.find(open_, i + 1)
    return None


def extra_checks(s: dict) -> list[str]:
    errs = []
    facts = s["A"]["facts"] + s["B"]["facts"] + s["shared_facts"]
    nums = re.findall(r"\d[\d,.]*", s["probe"]["answer"])
    if nums and not any(all(n in f for n in nums) for f in facts):
        errs.append("probe number not in a fact")
    if s["A"]["frame"].lower().split()[:4] == s["B"]["frame"].lower().split()[:4]:
        errs.append("frames too similar")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--per_domain", type=int, default=16, help="topics requested per sub-area")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reader_mem", default=None)
    ap.add_argument("--max_new", type=int, default=900)
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--out", default="outputs/SOFTCONF/persp_seeds_gen.json")
    a = ap.parse_args(argv)
    from softconf_build import validate_seed
    if a.dry_run:
        print(TOPIC_PROMPT.format(n=a.per_domain, area=DOMAINS["medical"][0]))
        print("-----")
        print(SEED_PROMPT.format(topic="type 1 diabetes"))
        print("example seed (schema):", json.dumps(EXAMPLE)[:300])
        return 0
    from xenc.backbone import build_chat, generate, load_reader
    model, tok = load_reader(a.model, device_map=a.device, max_mem=a.reader_mem)
    sysmsg = "You are a careful domain expert. Answer with valid JSON only."
    topics = []
    for dom, areas in DOMAINS.items():
        for area in areas:
            ids = build_chat(tok, a.model, sysmsg, TOPIC_PROMPT.format(n=a.per_domain, area=area))
            lst = parse_json(generate(model, tok, a.model, ids, 400, stop="]")) or []
            for t in lst:
                if isinstance(t, str) and t.strip().lower() not in {x[1].lower() for x in topics}:
                    topics.append((dom, t.strip(), area))
        print(f"{dom}: {len(topics)} topics so far", flush=True)
    seeds, rejected = [], []
    for k, (dom, topic, area) in enumerate(topics):
        ids = build_chat(tok, a.model, sysmsg, SEED_PROMPT.format(topic=topic))
        raw = generate(model, tok, a.model, ids, a.max_new, stop="\n}\n")
        s = parse_json(raw)
        if not isinstance(s, dict):
            rejected.append({"topic": topic, "err": ["unparseable"], "raw": raw[:400]})
            continue
        s.update({"id": f"g{k:03d}", "domain": dom, "area": area, "topic": s.get("topic") or topic})
        errs = validate_seed(s)
        errs += extra_checks(s) if not errs else []
        (rejected if errs else seeds).append(s if not errs else {"topic": topic, "err": errs, "seed": s})
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(topics)} topics: {len(seeds)} valid seeds", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(seeds, open(a.out, "w"), indent=1)
    json.dump(rejected, open(a.out.replace(".json", "_rejected.json"), "w"), indent=1)
    print(f"==== {len(seeds)} valid seeds ({len(rejected)} rejected) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
