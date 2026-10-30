"""Map free-text critique tags onto a fixed set of categories.

The critic writes a free 3-5 word tag per critique (6,972 distinct tags over 10,720
training critiques), so "incorrect citation", "citation mismatch" and "invalid
citation ids" are one problem under three names. Keyword rules, first match wins —
transparent, free and reproducible, unlike an LLM classifier. Rule order matters:
"query too broad" must reach `search` before `scope` sees "broad", "uncited
claims" must reach `citation` before `unsupported` sees "claims", and "overstated
conclusion" must reach `unsupported` before `direct_answer` sees "conclu".

    python eval/critique_categories.py   # coverage + sample tags per category on the training critiques
"""
import re

CATEGORIES = [  # (name, pattern) — checked in order against the lower-cased tag
    ("citation", r"cit(e|ed|es|ation|ations|ing)\b|uncited|snippet|\bid\b|\bids\b|source attribution"),
    ("search", r"search|quer(y|ies)|retriev|missing (key )?(paper|evidence|study|studies|source)|evidence gap"
               r"|coverage gap|\bcoverage\b|unverified|evidence check|verification|premature"
               r"|missing \w*\s?(evidence|sources?)\b|weak evidence base"),
    ("unsupported", r"unsupported|overstat|overgeneral|overreach|over-?claim|exceeds|speculat|extrapolat"
                    r"|inference|unjustified|unsubstantiated|overconfiden|certainty|causal|conflat|misattribut"
                    r"|factual|inaccura|incorrect|wrong|error|misstat|contradict|too strong|fabricat"
                    r"|mix(ed|es)? evidence|evidence (types|levels|tiers)|evidence mismatch|mismatched (evidence|support)"),
    ("direct_answer", r"direct|takeaway|upfront|up-front|summary|conclu|thesis|verdict|bottom line|synthesis"
                      r"|opening|answer first|executive|recommendation|roadmap|framing|slow start|delayed|ending|recap"
                      r"|bur(y|ied|ying) (the )?main|lead with"),
    ("caveat", r"caveat|limitation|uncertain|hedg|nuance|qualif"),
    ("scope", r"scope|drift|irrelevant|relevance|off-topic|tangent|background|focus|broad|narrow|digress"
              r"|question mismatch|misaligned|misread"),
    ("organization", r"struct|organi|reorgan|section|transition|repetit|redundan|duplicat|overlap|order"
                     r"|placement|flow|\blist|table|format|heading|bullet|length|verbose|dense|long|merge"),
    ("clarity", r"unclear|vague|generic|wording|ambigu|jargon|confusing|imprecise|clarity|terminolog"
                r"|definition|undefined|explain|explanation|confusion|mixing|distinction|taxonomy|abstract"),
    ("detail", r"thin|shallow|superficial|missing (detail|mechanism|example|comparison|quantitative)|detail"
               r"|depth|mechanism|example|quantitat|specific|comparison|insufficient"),
    ("artifact", r"artifact|metadata|markup|tag|truncat|unfinished|incomplete|placeholder|leak"),
]
OTHER = "other"
_COMPILED = [(name, re.compile(pat)) for name, pat in CATEGORIES]
NAMES = [name for name, _ in CATEGORIES] + [OTHER]


def categorize(tag):
    t = str(tag or "").strip().lower()
    for name, rx in _COMPILED:
        if rx.search(t):
            return name
    return OTHER


if __name__ == "__main__":
    import json
    import os
    from collections import Counter, defaultdict

    from score_eval import ROOT, parse_critique

    tags = Counter()
    for line in open(os.path.join(ROOT, "samples_1000", "drtulu_answers_w_critiques.jsonl")):
        d = parse_critique(json.loads(line)["critique"])
        for x in (d or {}).get("local", []):
            if isinstance(x, dict):
                tags[str(x.get("tag", "")).strip().lower()] += 1
    by_cat, examples = Counter(), defaultdict(Counter)
    for t, c in tags.items():
        k = categorize(t)
        by_cat[k] += c
        examples[k][t] += c
    total = sum(by_cat.values())
    for k in NAMES:
        print(f"{k:14} {by_cat[k]:6}  {100 * by_cat[k] / total:5.1f}%   "
              + ", ".join(t for t, _ in examples[k].most_common(6)))
    print("\nmost common 'other':", ", ".join(f"{t} ({c})" for t, c in examples[OTHER].most_common(40)))
