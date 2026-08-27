# %%
import json
import os
import re
import time
import difflib
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# %%
from types import SimpleNamespace

# ── Backend selection ─────────────────────────────────────────────────────────
# "gpt" → OpenAI Responses API; "glm" → self-hosted GLM via the OpenAI-compatible
# chat.completions API. Override with REWRITE_BACKEND=gpt|glm.
BACKEND = os.environ.get("REWRITE_BACKEND", "gpt")

# GPT client (only needs the key when actually used)
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-5.6-luna")
gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) if BACKEND == "gpt" else None

# GLM client (self-hosted, OpenAI-compatible; api_key is a placeholder)
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "http://titan-cs-aus-468.reviz.ai2.in:8020/v1")
GLM_MODEL    = os.environ.get("GLM_MODEL", "zai-org/GLM-5.2-FP8")
GLM_MAX_TOKENS = int(os.environ.get("GLM_MAX_TOKENS", "131072"))
glm_client = OpenAI(base_url=GLM_BASE_URL, api_key="dummy") if BACKEND == "glm" else None

# When on, search rounds are inserted PROGRAMMATICALLY: the pipeline emits the
# <think>/<call_tool>/<tool_output> tags, placement, and real results, and the
# model is asked only for the reflection prose. This removes the model's two
# weakest sub-tasks (tag structure + placement) so even cheaper models can't
# corrupt tags or drop/misplace searches. Set PROGRAMMATIC_SEARCH=0 for the old
# model-inserts-searches path.
PROGRAMMATIC_SEARCH = os.environ.get("PROGRAMMATIC_SEARCH", "1") != "0"

S2_API_KEY = os.environ.get("S2_API_KEY", "")
S2_SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"

MODEL_COSTS = {
    "gpt-4.1":      (2.00,  8.00),
    "gpt-4.1-mini": (0.40,  1.60),
    "gpt-4.1-nano": (0.10,  0.40),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4o-mini":  (0.15,  0.60),
    "gpt-5":        (1.25, 10.00),
    "o3":          (10.00, 40.00),
    "o4-mini":      (1.10,  4.40),
    "gpt-5.4":      (2.50, 15.00),
    "gpt-5.6-luna": (0.20,  1.20),
}

MODEL      = GPT_MODEL if BACKEND == "gpt" else GLM_MODEL

# Stop starting new records once cumulative cost exceeds this ($). Override via env.
COST_LIMIT = 100
# Process records concurrently. Note: higher values hit the S2 snippet endpoint
# harder (more 429s without an S2_API_KEY). Override via env.
MAX_WORKERS = 20

INPUT_FILE = "samples_1000/drtulu_answers_w_critiques.jsonl"
# tag output by model AND a run tag so different models/runs don't clobber each
# other. Bump RUN_TAG (e.g. v2) for a fresh file; override either via env.
RUN_TAG = os.environ.get("RUN_TAG", "v3")
_MODEL_SLUG = re.sub(r"[^A-Za-z0-9.]+", "-", MODEL).strip("-")
OUTPUT_FILE = INPUT_FILE.replace(".jsonl", f"_rewritten_{_MODEL_SLUG}_{RUN_TAG}.jsonl")

# %%
# ── Step 1 prompt: rewrite only flagged trace spans, insert placeholders ───────
trace_rewrite_prompt = """You are making edits to a generation trace for a long-form scientific question.

The trace is a sequence of sibling blocks that follow a strict, repeating pattern:

    <think>reasoning</think>
    <call_tool name="snippet_search" ...>query</call_tool>
    <tool_output>...results...</tool_output>

repeated across search rounds, followed by the final <answer> block (do NOT touch the answer).
Each block is a self-contained, properly closed sibling: a <think> block is always closed with
</think> BEFORE the next <call_tool> begins; <call_tool>, <tool_output>, and <think> are NEVER
nested inside one another.

You are given:
1. The original trace.
2. Critiques that each identify a specific flagged span in the trace and describe the issue.
3. A list of new search queries to insert, each with a placeholder ID.

Your task:
- The trace below has certain spans wrapped in <can_edit>...</can_edit> tags. You may ONLY edit text inside those tags. Every character outside a <can_edit> block must be reproduced exactly as-is.
- Edit the trace to address the critiques.
- For each new search query listed below, insert a search step consisting of a <call_tool> block immediately followed by its <tool_output> placeholder:

    <call_tool name="snippet_search" limit="5" year="{{year}}" fieldsOfStudy="{{field}}">query text</call_tool>
    <tool_output>PLACEHOLDER_{{id}}</tool_output>

  You may ONLY add these steps inside the <can_edit> spans. WHERE the <can_edit> region sits determines exactly what you write, because the trace must stay a FLAT sequence of sibling blocks — <think>, <call_tool>, and <tool_output> must NEVER be nested inside one another:
  * If the <can_edit> insertion point is BETWEEN blocks (e.g. right after a </tool_output> or </think>), write a brief new <think>reflection</think> to motivate the search, then the <call_tool> block, then its <tool_output> placeholder.
  * If the <can_edit> insertion point is INSIDE an open <think> block (mid-reasoning), first close that block with </think>, then write the <call_tool> block and its <tool_output> placeholder, then reopen a <think> to continue the reasoning that followed. Do NOT open a new <think> while one is still open.
  * Always close <call_tool> with </call_tool> before <tool_output>, and never emit an empty <think></think>.
  * Use the exact placeholder token <tool_output>PLACEHOLDER_{{id}}</tool_output> for the results — never invent snippet content; the real results are filled in later.
- Output the full revised trace with no preamble or explanation.

---
Question: {question}

---
Original trace:
{trace}

---
Critiques to fix:
{plan_critiques}

---
New searches to insert (use these exact placeholder IDs):
{search_list}
"""

# ── Step 1 prompt (programmatic-search mode): in-place plan edits only ─────────
trace_inplace_prompt = """You are making edits to a generation trace for a long-form scientific question.

The trace is a flat sequence of <think> / <call_tool> / <tool_output> blocks followed by the final <answer> (do NOT touch the answer).

You are given the trace with certain spans wrapped in <can_edit>...</can_edit> tags, and critiques describing what to fix.

Your task:
- Edit ONLY the text inside <can_edit>...</can_edit> tags to address the critiques. Reproduce every character outside a <can_edit> block exactly as-is.
- Do NOT add new searches, <call_tool> blocks, or <tool_output> blocks — only revise the flagged text in place. New searches are handled separately.
- Output the full revised trace with no preamble or explanation.

---
Question: {question}

---
Trace:
{trace}

---
Critiques to fix (edit only the flagged spans):
{plan_critiques}
"""

# ── Step 3 prompt: rewrite only flagged answer spans ──────────────────────────
answer_rewrite_prompt = """You are making edits to improve a long-form answer to a scientific question.

You are given:
1. The planning trace which contains reasoning steps and search results.
2. The original answer.
3. Answer-level critiques, each identifying a specific flagged span (critique_span) and describing the issue.

Your task:
- The answer below has certain spans wrapped in <can_edit>...</can_edit> tags. You may ONLY edit text inside those tags. Every character outside a <can_edit> block must be reproduced exactly as-is.
- For each <can_edit> span, apply the fix described by the corresponding critique.
- Where a fix requires drawing on evidence, use snippet IDs from the trace and wrap the sentence in <cite id="...">.
- Output the full revised answer with no preamble, no explanation.

---
Question: {question}

---
Planning trace (use for evidence):
{trace}

---
Original answer:
{answer}

---
Answer-level critiques to fix (edit only the flagged spans):
{answer_critiques}
"""


# %%
def run_s2_search(query_dict, limit=5, max_retries=6):
    """Query the S2 snippet API, retrying on rate-limits (HTTP 429) with
    exponential backoff (honoring the Retry-After header). Returns [] only after
    exhausting retries — so a transient rate-limit no longer silently drops a
    search. Without an S2_API_KEY the shared pool is ~1 req/sec, so 429s are
    common; set S2_API_KEY for much higher limits."""
    params = {"query": query_dict["query"], "limit": limit}
    if query_dict.get("year"):
        params["year"] = query_dict["year"]
    if query_dict.get("field_of_study"):
        params["fieldsOfStudy"] = query_dict["field_of_study"]
    if query_dict.get("authors"):
        params["authors"] = ",".join(query_dict["authors"])
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(S2_SNIPPET_URL, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay))
                print(f"    S2 rate-limited (429); waiting {wait:.0f}s "
                      f"[attempt {attempt + 1}/{max_retries}] '{query_dict['query'][:45]}'")
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            print(f"    S2 error for '{query_dict['query'][:60]}': {e}  (retry in {delay:.0f}s)")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    print(f"    S2 GAVE UP after {max_retries} attempts for '{query_dict['query'][:50]}' — search dropped")
    return []


def make_tool_output(results, id_prefix):
    """Format search results as a <tool_output> block (no <call_tool> wrapper)."""
    snippets = []
    for i, item in enumerate(results):
        sid   = f"{id_prefix}-{i}"
        title = item.get("paper", {}).get("title", "Unknown Title")
        text  = item.get("snippet", {}).get("text", "")
        snippets.append(f"<snippet id={sid}>\nTitle: {title}\nSnippet: {text}\n</snippet>")
    return "<tool_output>\n" + "\n".join(snippets) + "\n</tool_output>"


def make_call_tool_block(query_dict, id_prefix):
    """Format a <call_tool> block with a placeholder for the results."""
    return (
        f'<call_tool name="snippet_search" limit="5"'
        f' year="{query_dict.get("year", "")}"'
        f' fieldsOfStudy="{query_dict.get("field_of_study", "")}">'
        f'{query_dict["query"]}</call_tool>\n'
        f'<tool_output>PLACEHOLDER_{id_prefix}</tool_output>'
    )


def _norm_span(pair):
    """Normalize a critique span entry into a (start, end) string pair, or None
    if unusable. LLM-generated critiques sometimes give the WHOLE span as a
    single string (`["one sentence"]` or `"one sentence"`) instead of a
    [first_words, last_words] pair — coerce those to (s, s), which locates the
    full span exactly (start_key=s[:40], end_key=s[-40:])."""
    if isinstance(pair, str):
        return (pair, pair)
    if isinstance(pair, (list, tuple)):
        if len(pair) == 2 and isinstance(pair[0], str) and isinstance(pair[1], str):
            return (pair[0], pair[1])
        if len(pair) == 1 and isinstance(pair[0], str):
            return (pair[0], pair[0])
    return None


def compute_edit_spans(text, critiques):
    """Locate each critique's edit_span in `text` and return disjoint, sorted
    (start, end) intervals.

    Overlapping/duplicate/touching spans are merged. Merging is essential: two
    critiques can share (or overlap) a span, and treating them separately would
    later double-tag or double-count the shared region.

    Search-round insertion points are snapped: a `search_required` insertion
    (edit_span start == end) that sits INSIDE a <think> block is extended back to
    the start of that block, so the new <think>/<call_tool>/<tool_output> round
    lands at a block boundary (between blocks) instead of nested in reasoning
    prose. Non-search insertions and in-place rewrites are left untouched."""
    spans = []
    for c in critiques:
        search_insert = bool(c.get("search_required"))
        for pair in c.get("edit_span", c.get("critique_span", [])):
            span = _norm_span(pair)
            if span is None:
                continue
            start_str, end_str = span
            start_key = start_str[:40]
            end_key   = end_str[-40:]
            idx_s = text.find(start_key)
            if idx_s == -1:
                continue
            idx_e = text.find(end_key, idx_s)
            if idx_e == -1:
                continue
            idx_e += len(end_key)

            # Snap a search-round insertion point to span its ENCLOSING <think>
            # block in full — but only if the anchor is genuinely inside a
            # still-open <think> (no </think> between the block start and the
            # anchor). Extending to the block's matching </think> keeps the
            # editable span a balanced, complete block, so the model can insert a
            # new round at the block boundary and rewrite the think without
            # leaving a dangling </think> just outside the span.
            if search_insert and start_str == end_str:
                b = text.rfind("<think>", 0, idx_s)
                if b != -1 and b > text.rfind("</think>", 0, idx_s):
                    close = text.find("</think>", idx_e)
                    if close != -1:
                        idx_s = b
                        idx_e = close + len("</think>")

            spans.append((idx_s, idx_e))

    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def insert_can_edit_tags(text, critiques):
    """Wrap each (merged) edit span with <can_edit>...</can_edit> tags. Inserted
    in reverse order so earlier, already-final positions aren't shifted."""
    for idx_s, idx_e in sorted(compute_edit_spans(text, critiques), reverse=True):
        text = text[:idx_s] + "<can_edit>" + text[idx_s:idx_e] + "</can_edit>" + text[idx_e:]
    return text


def _in_editable(i1, i2, spans):
    """Is the original range [i1, i2) covered by some editable span? For a pure
    insertion (i1 == i2) the insertion point i1 must fall within a span."""
    for s, e in spans:
        if i1 == i2:
            if s <= i1 <= e:
                return True
        elif s <= i1 and i2 <= e:
            return True
    return False


def splice_edits(original, critiques, model_output):
    """Diff-based revert: keep the model's text ONLY where it maps inside an
    editable span; revert every other change back to the original. Guarantees
    edits stay within the can_edit spans regardless of model drift, and does NOT
    rely on <can_edit> tags surviving in the output (they often don't).

    Returns (spliced_text, dropped_chars) where dropped_chars is the number of
    model-output characters discarded because they fell OUTSIDE the editable
    spans (out-of-bounds drift, or edits the model placed in the wrong region —
    e.g. searches inserted outside a mis-placed insertion span)."""
    spans = compute_edit_spans(original, critiques)
    mo = model_output.replace("<can_edit>", "").replace("</can_edit>", "")

    out, dropped = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original, mo, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(original[i1:i2])
        elif _in_editable(i1, i2, spans):
            out.append(mo[j1:j2])            # accept the model's in-bounds edit
        else:
            out.append(original[i1:i2])      # revert out-of-bounds change to original
            dropped += j2 - j1               # model text we discarded
    return "".join(out), dropped


def collect_search_queries(critiques):
    """Collect unique s2_search_queries from search_required=True critiques."""
    queries, seen = [], set()
    for c in critiques:
        if c.get("search_required") and c.get("s2_search_queries"):
            for q in c["s2_search_queries"]:
                if q["query"] not in seen:
                    seen.add(q["query"])
                    queries.append(q)
    return queries


def llm_call(prompt, max_tokens=None):
    """Route to the selected backend. Returns (text, usage) where usage has
    .input_tokens / .output_tokens regardless of backend."""
    if BACKEND == "glm":
        resp = glm_client.chat.completions.create(
            model=GLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or GLM_MAX_TOKENS,
        )
        u = resp.usage
        usage = SimpleNamespace(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
        text = resp.choices[0].message.content
        finish = resp.choices[0].finish_reason
        if finish == "length":
            print(f"    ⚠ GLM output hit max_tokens ({max_tokens or GLM_MAX_TOKENS}) — response truncated")
        return text, usage
    else:
        kwargs = {"max_output_tokens": max_tokens} if max_tokens else {}
        resp = gpt_client.responses.create(model=MODEL, input=prompt, **kwargs)
        u = resp.usage
        return resp.output_text, SimpleNamespace(
            input_tokens=u.input_tokens, output_tokens=u.output_tokens
        )


# ── Programmatic search-round insertion ───────────────────────────────────────
reflection_prompt = """You are the author of a scientific research reasoning trace, writing the short planning note that comes right before a NEW literature search you are about to run.

Here is the underlying reason this search is needed — a gap or problem in the reasoning so far:
{issue}

Turn that into YOUR OWN reflection: name the gap concretely, why it matters for answering the question, and what this search should surface to address it. First person ("I"), 2-3 sentences, in the careful analytical voice of the surrounding trace.

Rules:
- Write as if you noticed the gap yourself. Do NOT mention a "critique", "feedback", "reviewer", or that you were told or instructed anything — there is no critique in this trace, only your own reasoning. Never write phrases like "the critique correctly identifies…".
- Be specific and grounded in the reason above; avoid generic filler like "I still need evidence, so I will search."
- Output ONLY the reflection prose: no tags, no markdown, no quotes, no lists, no preamble.
- Do not quote the query verbatim and do not list results.

Query about to be run: {query}{context}"""


def make_reflection(query_dict, issue, context_before=""):
    """Ask the model for a short, tag-free, critique-grounded reflection to
    precede a search. Any stray tags are stripped so the model cannot corrupt
    trace structure; if the model returns nothing, fall back to the critique's
    own (already reflective) issue text rather than a generic filler."""
    ctx = ""
    if context_before.strip():
        ctx = ("\n\nReasoning immediately before this point (match its tone; do not repeat it):\n"
               + '"""\n' + context_before[-1200:].strip() + '\n"""')
    prompt = reflection_prompt.format(issue=issue or "(unspecified)", query=query_dict["query"], context=ctx)
    usage = SimpleNamespace(input_tokens=0, output_tokens=0)
    try:
        text, usage = llm_call(prompt, max_tokens=800)
    except Exception as e:
        text = ""
        print(f"    reflection call failed ({e}); using critique issue as reflection")
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text or "")          # strip any tags the model emitted
    text = " ".join(text.split())                                 # collapse to clean inline prose
    if not text:                                                  # fall back to the critique's own reasoning
        base = " ".join((issue or "").split())
        text = base or f"I will search for {query_dict['query']} to close this gap."
    return text, usage


def build_search_round(reflection, query_dict, tool_output_text):
    """Assemble one well-formed DR Tulu search round. The pipeline owns every
    tag; only `reflection` comes from the model."""
    return (
        f"<think>{reflection}</think>\n"
        f'<call_tool name="snippet_search" limit="5"'
        f' year="{query_dict.get("year", "")}"'
        f' fieldsOfStudy="{query_dict.get("field_of_study", "")}">'
        f'{query_dict["query"]}</call_tool>\n'
        f"{tool_output_text}\n"
    )


def insert_search_rounds(trace, search_critiques, search_results):
    """Insert each search critique's rounds at its snapped block boundary
    (before the enclosing <think>). Placement and structure are guaranteed by
    construction — the model never sees a tag."""
    inserts = []  # (position, text)
    refl_in = refl_out = 0
    for c in search_critiques:
        spans = c.get("edit_span", c.get("critique_span", []))
        anchor = next((_norm_span(p)[0] for p in spans if _norm_span(p)), None)
        if not anchor:
            print(f"    ⚠ no usable anchor for search critique '{c.get('tag','')}', skipping")
            continue
        idx = trace.find(anchor[:40])
        if idx == -1:
            print(f"    ⚠ search-insert anchor not found, skipping: {anchor[:40]!r}")
            continue
        # A new search should follow the CURRENT round it's reacting to: insert
        # right after that round's closing </tool_output>. If no search round
        # follows the anchor (it sits in a concluding/planning <think>), fall back
        # to inserting just before that enclosing <think> so the new search still
        # precedes the conclusion.
        end = trace.find("</tool_output>", idx)
        if end != -1:
            pos = end + len("</tool_output>")
        else:
            b = trace.rfind("<think>", 0, idx)
            pos = b if (b != -1 and b > trace.rfind("</think>", 0, idx)) else idx
        rounds = []
        for q in c.get("s2_search_queries", []):
            pid = hex(abs(hash(q["query"])) % 0xFFFFFF)[2:].zfill(6)
            if pid not in search_results:
                print(f"    ⚠ no results for query (skipped): {q['query'][:50]}")
                continue
            _, tool_output_text = search_results[pid]
            reflection, ru = make_reflection(q, c.get("issue", ""), context_before=trace[max(0, pos - 1200):pos])
            refl_in += ru.input_tokens
            refl_out += ru.output_tokens
            rounds.append(build_search_round(reflection, q, tool_output_text))
        if rounds:
            inserts.append((pos, "\n" + "".join(rounds)))
    # apply from the end so earlier offsets remain valid
    for pos, block in sorted(inserts, key=lambda x: x[0], reverse=True):
        trace = trace[:pos] + block + trace[pos:]
    return trace, refl_in, refl_out


def validate_trace_structure(trace):
    """Check that a rewritten trace stays well-formed DR Tulu.

    Returns a list of human-readable problem strings (empty == well-formed).
    Catches the malformed-insertion failure mode where the model nests
    <call_tool>/<tool_output> inside an unclosed <think> block.
    """
    problems = []
    tag_re = re.compile(
        r"<think>|</think>|<call_tool\b[^>]*>|</call_tool>|<tool_output>|</tool_output>"
    )
    open_think = open_call = open_output = False
    for m in tag_re.finditer(trace):
        tag = m.group(0)
        if tag == "<think>":
            if open_think:
                problems.append(f"@{m.start()}: <think> opened while a <think> was still open")
            if open_call or open_output:
                problems.append(f"@{m.start()}: <think> opened inside an open <call_tool>/<tool_output>")
            open_think = True
        elif tag == "</think>":
            if not open_think:
                problems.append(f"@{m.start()}: </think> without a matching open <think>")
            open_think = False
        elif tag.startswith("<call_tool"):
            if open_think:
                problems.append(f"@{m.start()}: <call_tool> inside an unclosed <think> block")
            if open_output:
                problems.append(f"@{m.start()}: <call_tool> inside an open <tool_output>")
            open_call = True
        elif tag == "</call_tool>":
            if not open_call:
                problems.append(f"@{m.start()}: </call_tool> without a matching open <call_tool>")
            open_call = False
        elif tag == "<tool_output>":
            if open_think:
                problems.append(f"@{m.start()}: <tool_output> inside an unclosed <think> block")
            if open_call:
                problems.append(f"@{m.start()}: <tool_output> inside an unclosed <call_tool>")
            open_output = True
        elif tag == "</tool_output>":
            if not open_output:
                problems.append(f"@{m.start()}: </tool_output> without a matching open <tool_output>")
            open_output = False
    if open_think:
        problems.append("unclosed <think> block at end of trace")
    if open_call:
        problems.append("unclosed <call_tool> block at end of trace")
    if open_output:
        problems.append("unclosed <tool_output> block at end of trace")
    return problems


def parse_critique(s):
    """Parse the critique JSON, with a repair pass for the common failure mode:
    the model copies HTML-like tags (e.g. <cite id="x">) verbatim into string
    values, and their inner double quotes break the JSON. We escape unescaped
    quotes inside <...> tags and retry. Returns the parsed object, or None if it
    still won't parse (idiosyncratic structural errors — skipped by the caller)."""
    try:
        return json.loads(s)
    except Exception:
        pass
    repaired = re.sub(r"<[^<>]*>",
                      lambda m: re.sub(r'(?<!\\)"', lambda _: '\\"', m.group(0)),
                      s)
    try:
        return json.loads(repaired)
    except Exception:
        return None


def extract_answer_block(trace):
    """Return the <answer>…</answer> block. Falls back to <answer>…end-of-trace
    when the closing tag is missing (truncated generation), and None when there
    is no <answer> at all."""
    m = (re.search(r"<answer>.*?</answer>", trace, flags=re.DOTALL)
         or re.search(r"<answer>.*\Z", trace, flags=re.DOTALL))
    return m.group(0) if m else None


# %%
records = []
with open(INPUT_FILE, "r") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Loaded {len(records)} records from {INPUT_FILE}.")

# Process only records[START_INDEX:END_INDEX] (by position in the input file).
# e.g. START_INDEX=500 runs samples 500..end; END_INDEX caps it. Override via env.
START_INDEX = 0
END_INDEX   = int(os.environ.get("END_INDEX", str(len(records))))
records = records[START_INDEX:END_INDEX]
print(f"Slice [{START_INDEX}:{END_INDEX}] → {len(records)} records.")

# Resume: skip records already written to OUTPUT_FILE and APPEND, so a re-run
# continues where a crashed run left off (no re-doing / re-paying). To force a
# clean run from scratch, delete OUTPUT_FILE (or bump RUN_TAG) first.
done = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as _f:
        for _line in _f:
            if _line.strip():
                try:
                    done.add(json.loads(_line)["question"])
                except Exception:
                    pass  # tolerate a partial last line from a crash mid-write
_before = len(records)
records = [r for r in records if r.get("question", "").strip() not in done]
print(f"Output → {OUTPUT_FILE} | {_before} loaded | {len(done)} already done | {len(records)} to do")

# %%
total_cost = 0.0
total_in = total_out = 0
input_price, output_price = MODEL_COSTS.get(MODEL, (0.0, 0.0))
if MODEL not in MODEL_COSTS:
    print(f"⚠ no pricing for MODEL={MODEL!r}; costs will show $0 (token counts still logged).")

cost_lock  = threading.Lock()   # guards total_cost / total_in / total_out
write_lock = threading.Lock()   # guards appends to OUTPUT_FILE
stop_event = threading.Event()  # set once COST_LIMIT is exceeded


def usd(in_tok, out_tok):
    return in_tok * input_price / 1_000_000 + out_tok * output_price / 1_000_000


def process_record(item):
    global total_cost, total_in, total_out
    if stop_event.is_set():            # cost cap already hit — don't start new work
        return
    question        = item["question"].strip()
    trace           = item.get("original_trace", "")
    critique_str    = item.get("critique", "[]")

    critique_json = parse_critique(critique_str)
    if critique_json is None:
        print(f"  ⚠ unparseable critique for '{question[:60]}' — skipping record")
        return

    all_critiques    = critique_json if isinstance(critique_json, list) else critique_json.get("local", [])
    plan_critiques   = [c for c in all_critiques if c.get("location") in ("plan", "both")]
    answer_critiques = [c for c in all_critiques if c.get("location") in ("answer", "both")]

    # ── Pre-fetch all search results ──────────────────────────────────────────
    search_queries = collect_search_queries(plan_critiques)
    print(f"\n[{question[:60]}]  {len(search_queries)} searches to run.")

    # id_prefix → (query_dict, tool_output_text)
    search_results = {}
    for q in search_queries:
        print(f"  Searching: {q['query'][:70]}...")
        id_prefix = hex(abs(hash(q["query"])) % 0xFFFFFF)[2:].zfill(6)
        results = run_s2_search(q)
        if results:
            search_results[id_prefix] = (q, make_tool_output(results, id_prefix))
        else:
            print(f"    No results returned.")

    # Strip <answer> block from trace before Step 1 — it's irrelevant and
    # could confuse the model into editing the answer prematurely.
    trace_no_answer = re.sub(r"<answer>.*?</answer>", "", trace, flags=re.DOTALL).rstrip()

    # per-phase token counters (input, output)
    s1_in = s1_out = refl_in = refl_out = 0

    if PROGRAMMATIC_SEARCH:
        # Split plan critiques: in-place text edits go to the model; search
        # insertions are handled programmatically (pipeline owns the tags).
        search_critiques = [c for c in plan_critiques
                            if c.get("search_required") and c.get("s2_search_queries")]
        inplace_plan     = [c for c in plan_critiques if c not in search_critiques]

        # ── Step 1a: model does ONLY in-place plan rewrites (skip if none) ────
        rewritten_trace = trace_no_answer
        if inplace_plan:
            trace_tagged = insert_can_edit_tags(trace_no_answer, inplace_plan)
            step1_prompt = trace_inplace_prompt.format(
                question=question,
                trace=trace_tagged,
                plan_critiques=json.dumps(inplace_plan, indent=2),
            )
            print("  Step 1: in-place plan rewrites...")
            step1_out, usage1 = llm_call(step1_prompt)
            s1_in, s1_out = usage1.input_tokens, usage1.output_tokens
            rewritten_trace, trace_dropped = splice_edits(trace_no_answer, inplace_plan, step1_out)
            if trace_dropped > 1000:
                print(f"  ⚠ TRACE in-place: reverted {trace_dropped} out-of-bounds chars for '{question[:50]}'")

        # ── Step 1b: programmatic search insertion (model writes only reflections) ──
        if search_critiques:
            nq = sum(len(c.get("s2_search_queries", [])) for c in search_critiques)
            print(f"  Step 1b: inserting {nq} search round(s) programmatically...")
            rewritten_trace, refl_in, refl_out = insert_search_rounds(rewritten_trace, search_critiques, search_results)
    else:
        # ── Legacy path: model inserts searches via placeholders ──────────────
        search_list_text = "\n".join(
            f"- placeholder ID: {pid}  |  query: {q['query']}"
            + (f"  |  year: {q.get('year', '')}" if q.get("year") else "")
            + (f"  |  field: {q.get('field_of_study', '')}" if q.get("field_of_study") else "")
            for pid, (q, _) in search_results.items()
        ) or "None"

        trace_tagged = insert_can_edit_tags(trace_no_answer, plan_critiques)
        step1_prompt = trace_rewrite_prompt.format(
            question=question,
            trace=trace_tagged,
            plan_critiques=json.dumps(plan_critiques, indent=2),
            search_list=search_list_text,
        )
        print("  Step 1: rewriting trace...")
        rewritten_trace_with_placeholders, usage1 = llm_call(step1_prompt)
        s1_in, s1_out = usage1.input_tokens, usage1.output_tokens

        rewritten_trace = rewritten_trace_with_placeholders
        for pid, (_, tool_output_text) in search_results.items():
            rewritten_trace = rewritten_trace.replace(
                f"<tool_output>PLACEHOLDER_{pid}</tool_output>", tool_output_text)

        rewritten_trace, trace_dropped = splice_edits(trace_no_answer, plan_critiques, rewritten_trace)
        if trace_dropped > 1000:
            print(f"  ⚠ TRACE: reverted {trace_dropped} chars of OUT-OF-BOUNDS model edits for "
                  f"'{question[:50]}' — likely searches inserted outside a mis-placed edit_span")

    # Warn if the final trace has broken DR Tulu block structure (e.g. <call_tool>
    # nested inside an unclosed <think>).
    struct_problems = validate_trace_structure(rewritten_trace)
    if struct_problems:
        print(f"  ⚠ MALFORMED TRACE for '{question[:50]}': {len(struct_problems)} issue(s)")
        for p in struct_problems[:5]:
            print(f"      - {p}")

    # Warn if any placeholder was left unfilled (query returned no results, or
    # the model altered the placeholder token).
    leftover = re.findall(r"PLACEHOLDER_[A-Za-z0-9]+", rewritten_trace)
    if leftover:
        print(f"  ⚠ UNFILLED PLACEHOLDERS for '{question[:50]}': {sorted(set(leftover))}")

    # ── Step 3: rewrite answer using filled trace + answer critiques ──────────
    # Tag/rewrite the answer AS IT APPEARS IN THE TRACE, i.e. with the
    # <answer>…</answer> wrapper. Critiques were generated against the full
    # trace, so their edit_span anchors include the wrapper (e.g. "<answer>\n#
    # Summary"); the wrapper-less `original_answer` field would fail to match.
    answer_block = extract_answer_block(trace)
    if answer_block is None:
        print(f"  ⚠ no <answer> block in trace for '{question[:50]}' — skipping record")
        return
    answer_tagged = insert_can_edit_tags(answer_block, answer_critiques)

    step3_prompt = answer_rewrite_prompt.format(
        question=question,
        trace=rewritten_trace,
        answer=answer_tagged,
        answer_critiques=json.dumps(answer_critiques, indent=2),
    )
    print("  Step 3: rewriting answer...")
    rewritten_answer, usage3 = llm_call(step3_prompt)
    s3_in, s3_out = usage3.input_tokens, usage3.output_tokens

    # Same diff-based splice for the answer: keep model edits only inside the
    # can_edit spans, everything else reverts to the original answer block.
    rewritten_answer, ans_dropped = splice_edits(answer_block, answer_critiques, rewritten_answer)
    if ans_dropped > 1000:
        print(f"  ⚠ ANSWER: reverted {ans_dropped} chars of OUT-OF-BOUNDS model edits for '{question[:50]}'")

    # ── Cost accounting (shared state; under lock) ────────────────────────────
    cost1, cost_refl, cost3 = usd(s1_in, s1_out), usd(refl_in, refl_out), usd(s3_in, s3_out)
    rec_in  = s1_in + refl_in + s3_in
    rec_out = s1_out + refl_out + s3_out
    rec_cost = cost1 + cost_refl + cost3
    with cost_lock:
        total_cost += rec_cost
        total_in  += rec_in
        total_out += rec_out
        print(f"  Cost [{MODEL}]  "
              f"step1(in={s1_in} out={s1_out} ${cost1:.4f})  "
              f"refl(in={refl_in} out={refl_out} ${cost_refl:.4f})  "
              f"step3(in={s3_in} out={s3_out} ${cost3:.4f})  |  "
              f"record: {rec_in}+{rec_out}tok ${rec_cost:.4f}  |  "
              f"RUNNING: {total_in}+{total_out}tok ${total_cost:.4f}")
        if total_cost >= COST_LIMIT and not stop_event.is_set():
            stop_event.set()
            print(f"⚠ COST LIMIT ${COST_LIMIT:.2f} reached (total=${total_cost:.4f}); no new records will start.")

    with write_lock:
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps({
                "question":        question,
                "original_answer": answer_block,
                "original_trace":  trace,
                "rewritten_trace": rewritten_trace,
                "critique":        critique_str,
                "rewritten":       rewritten_answer,
            }) + "\n")


# ── Drive records concurrently ────────────────────────────────────────────────
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_record, item): item for item in records}
    for fut in tqdm(as_completed(futures), total=len(records)):
        try:
            fut.result()
        except Exception as e:
            print(f"Error on a record: {e}")

print(f"\n=== DONE: {len(records)} records | model={MODEL} | workers={MAX_WORKERS} | "
      f"{total_in} input + {total_out} output tokens | TOTAL COST ${total_cost:.4f} ===")
print(f"Output: {OUTPUT_FILE}")
