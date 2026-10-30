"""Repair deterministically-resolvable dangling citations in rewritten answers.

A `<cite id="X">` is *dangling* when X matches no `<snippet id=X>` anywhere in the
record's trace. Most dangling cites are inherited — DR Tulu's own answers already
cite ids that never existed (953 across 169 records) — and those are left alone on
purpose: the rewrite dropping 332 of them is real preference signal, and "fixing"
them on one side only would manufacture a chosen-vs-rejected regularity that has
nothing to do with the critique.

This script touches ONLY ids the rewrite itself introduced (cited in `rewritten`,
absent from `original_answer`, unresolvable against the trace) AND that resolve to
exactly one real snippet. Two rules qualify:

  S-prefix   The model copied the `S_`/`S` id style from the worked example in the
             system prompt (`<snippet id=S_a1B9xQ2>`) and stuck it on a real id:
             `S_0fc31294-1` -> `0fc31294-1`. Unambiguous.
  unique     The prefix is a truncation/extension of exactly one real prefix and
             the same index exists there: `72cc43d2-3d2-9` -> `72cc43d2-9`. Applied
             only when the candidate is unique.

Anything else is left as-is: truncated prefixes matching several real ids
(`fc4a43d-0`) and out-of-range indices (`d82cbfcf-5`, where that round returned
only -0..-4). Guessing there would attach a claim to the wrong paper, which is
worse than a broken citation.

Only the `rewritten` field is modified; every other field is passed through
verbatim. By construction the repaired ids appear only in `rewritten` (an id cited
in the original answer that resolves against the original trace also resolves
against the rewritten trace, which is a superset), so there is no risk of fixing
one side of a pair and not the other.

Usage:
    python repair_citations.py --in  samples_1000/..._v3_mod.jsonl \
                               --out samples_1000/..._v3_mod_fixed.jsonl
    python repair_citations.py --dry-run          # report, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

SNIPPET = re.compile(r"<snippet id=([^\s>]+?)>")
# Only well-formed cite tags: no whitespace or angle brackets inside the value.
# Malformed ones (unterminated quote) are left untouched rather than guessed at.
CITE_TAG = re.compile(r'(<cite\s+ids?=")([^"<>\s]*)("\s*>)')

DEFAULT_IN = "samples_1000/drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v3_mod.jsonl"


def snippet_ids(text: str) -> set[str]:
    return {m.group(1) for m in SNIPPET.finditer(text)}


def cite_ids(text: str) -> set[str]:
    out: set[str] = set()
    for m in CITE_TAG.finditer(text):
        out |= {x.strip() for x in m.group(2).split(",") if x.strip()}
    return out


def resolve(cid: str, known: set[str], prefixes: set[str]) -> tuple[str, str] | None:
    """Return (real_id, rule) if `cid` resolves to exactly one snippet, else None."""
    prefix, _, index = cid.rpartition("-")
    if not index.isdigit() or not prefix:
        return None

    for stripped in (prefix[2:] if prefix.startswith("S_") else None,
                     prefix[1:] if prefix.startswith("S") else None):
        if stripped and f"{stripped}-{index}" in known:
            return f"{stripped}-{index}", "S-prefix"

    candidates = {p for p in prefixes
                  if (p.startswith(prefix) or prefix.startswith(p) or prefix in p)
                  and f"{p}-{index}" in known}
    if len(candidates) == 1:
        return f"{candidates.pop()}-{index}", "unique"
    return None


def repair_text(text: str, mapping: dict[str, str]) -> str:
    """Rewrite cite ids through `mapping`, preserving everything else byte-for-byte."""
    def sub(m):
        ids = [x.strip() for x in m.group(2).split(",")]
        return m.group(1) + ",".join(mapping.get(x, x) for x in ids) + m.group(3)
    return CITE_TAG.sub(sub, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default=DEFAULT_IN)
    ap.add_argument("--out", dest="dst", default=None,
                    help="default: <in> with _fixed before .jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dst = args.dst or args.src.replace(".jsonl", "_fixed.jsonl")

    stats = Counter()
    repairs: list[tuple[int, str, str, str]] = []
    out_lines: list[str] = []

    with open(args.src) as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            known_orig = snippet_ids(rec["original_trace"])
            known = snippet_ids(rec["rewritten_trace"]) | known_orig
            prefixes = {s.rsplit("-", 1)[0] for s in known}

            dangling_orig = cite_ids(rec["original_answer"]) - known_orig
            introduced = cite_ids(rec["rewritten"]) - known - dangling_orig
            stats["introduced"] += len(introduced)

            mapping: dict[str, str] = {}
            for cid in sorted(introduced):
                hit = resolve(cid, known, prefixes)
                if hit is None:
                    stats["left as-is"] += 1
                    continue
                real, rule = hit
                mapping[cid] = real
                stats[f"repaired ({rule})"] += 1
                repairs.append((idx, cid, real, rule))

            if not mapping:
                # Pass through verbatim — re-serializing would risk formatting churn
                # (escaping, key order) on records this script has no business touching.
                out_lines.append(line.rstrip("\n"))
                continue
            fixed = repair_text(rec["rewritten"], mapping)
            assert fixed != rec["rewritten"], f"record {idx}: mapping made no change"
            rec["rewritten"] = fixed
            stats["records changed"] += 1
            out_lines.append(json.dumps(rec))

    total_repaired = sum(v for k, v in stats.items() if k.startswith("repaired"))
    print(f"introduced dangling cite-ids : {stats['introduced']}")
    for k in sorted(stats):
        if k.startswith("repaired"):
            print(f"  {k:<28}{stats[k]}")
    print(f"  {'left as-is':<28}{stats['left as-is']}")
    print(f"records changed              : {stats['records changed']}")
    print()
    for idx, cid, real, rule in repairs:
        print(f"  rec {idx:>4}  {cid:<28} -> {real:<20} [{rule}]")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    with open(dst, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nwrote {dst}  ({len(out_lines)} records, {total_repaired} ids repaired)")


if __name__ == "__main__":
    main()
