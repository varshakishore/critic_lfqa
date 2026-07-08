# %%
import json
import os
import re
import requests
from openai import OpenAI
from tqdm import tqdm

# %%
OAI_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OAI_KEY)

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
}

MODEL      = "gpt-4.1"
INPUT_FILE = "critique_outputs_v3_single_test_rewriting.jsonl"
OUTPUT_FILE = INPUT_FILE.replace(".jsonl", "_rewritten.jsonl")

# %%
# ── Step 1 prompt: rewrite only flagged trace spans, insert placeholders ───────
trace_rewrite_prompt = """You are making edits to a generation trace for a long-form scientific question.

The trace structure is:
- <think> blocks: the model's internal reasoning and search planning.
- <call_tool> / <tool_output> blocks: searches issued and their results.
- <answer> block: the final answer (do NOT touch this).

You are given:
1. The original trace.
2. Critiques that each identify a specific flagged span in the trace and describe the issue.
3. A list of new search queries to insert, each with a placeholder ID.

Your task:
- The trace below has certain spans wrapped in <can_edit>...</can_edit> tags. You may ONLY edit text inside those tags. Every character outside a <can_edit> block must be reproduced exactly as-is.
- For each new search query listed below, insert a <think>brief reflection</think> followed by a <call_tool> block and <tool_output>PLACEHOLDER_{id}</tool_output>. You can only add the queries in the <can_edit> spans.
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

# ── Step 3 prompt: rewrite only flagged answer spans ──────────────────────────
answer_rewrite_prompt = """You are making edits to improve a long-form answer to a scientific question.

You are given:
1. The planning trace which contains reasoning steps and search results.
2. The original answer.
3. Answer-level critiques, each identifying a specific flagged span (start_end) and describing the issue.

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
def run_s2_search(query_dict, limit=5):
    params = {"query": query_dict["query"], "limit": limit}
    if query_dict.get("year"):
        params["year"] = query_dict["year"]
    if query_dict.get("field_of_study"):
        params["fieldsOfStudy"] = query_dict["field_of_study"]
    if query_dict.get("authors"):
        params["authors"] = ",".join(query_dict["authors"])
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    try:
        resp = requests.get(S2_SNIPPET_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        print(f"    S2 error for '{query_dict['query'][:60]}': {e}")
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


def insert_can_edit_tags(text, critiques):
    """Wrap each flagged start_end span with <can_edit>...</can_edit> tags.
    Spans are inserted in reverse order so earlier positions aren't shifted."""
    spans = []
    for c in critiques:
        for start_str, end_str in c.get("start_end", []):
            start_key = start_str[:40]
            end_key   = end_str[-40:]
            idx_s = text.find(start_key)
            if idx_s == -1:
                continue
            idx_e = text.find(end_key, idx_s)
            if idx_e == -1:
                continue
            spans.append((idx_s, idx_e + len(end_key)))
    spans.sort(key=lambda x: x[0], reverse=True)
    for idx_s, idx_e in spans:
        text = text[:idx_s] + "<can_edit>" + text[idx_s:idx_e] + "</can_edit>" + text[idx_e:]
    return text


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


def llm_call(prompt):
    response = client.responses.create(model=MODEL, input=prompt)
    return response.output_text, response.usage


# %%
records = []
with open(INPUT_FILE, "r") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Loaded {len(records)} records from {INPUT_FILE}.")

# %%
total_cost = 0.0
input_price, output_price = MODEL_COSTS.get(MODEL, (0.0, 0.0))

for item in tqdm(records):
    question        = item["question"].strip()
    trace           = item.get("original_trace", "")
    original_answer = item.get("original_answer", "")
    critique_str    = item.get("critique", "[]")

    try:
        critique_json = json.loads(critique_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse critique for '{question[:60]}': {e}")
        continue

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

    # ── Step 1: rewrite trace with placeholders ───────────────────────────────
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
    # import pdb; pdb.set_trace()
    rewritten_trace_with_placeholders, usage1 = llm_call(step1_prompt)
    cost1 = usage1.input_tokens * input_price / 1_000_000 + usage1.output_tokens * output_price / 1_000_000
    # import pdb; pdb.set_trace()

    # ── Step 2: fill placeholders with real results ───────────────────────────
    rewritten_trace = rewritten_trace_with_placeholders
    for pid, (_, tool_output_text) in search_results.items():
        rewritten_trace = rewritten_trace.replace(
            f"<tool_output>PLACEHOLDER_{pid}</tool_output>",
            tool_output_text
        )

    # ── Step 3: rewrite answer using filled trace + answer critiques ──────────
    answer_tagged = insert_can_edit_tags(original_answer, answer_critiques)

    step3_prompt = answer_rewrite_prompt.format(
        question=question,
        trace=rewritten_trace,
        answer=answer_tagged,
        answer_critiques=json.dumps(answer_critiques, indent=2),
    )
    print("  Step 3: rewriting answer...")
    rewritten_answer, usage3 = llm_call(step3_prompt)
    cost3 = usage3.input_tokens * input_price / 1_000_000 + usage3.output_tokens * output_price / 1_000_000

    total_cost += cost1 + cost3
    print(f"  Cost: step1=${cost1:.4f}  step3=${cost3:.4f}  |  Total: ${total_cost:.4f}")

    # import pdb; pdb.set_trace()
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps({
            "question":        question,
            "original_answer": original_answer,
            "original_trace":  trace,
            "rewritten_trace": rewritten_trace,
            "critique":        critique_str,
            "rewritten":       rewritten_answer,
        }) + "\n")
