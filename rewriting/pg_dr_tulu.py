# %%
import json
from openai import OpenAI
from tqdm import tqdm
import os
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

# %%
OAI_KEY = os.environ["OPENAI_API_KEY"] 
client = OpenAI(api_key=OAI_KEY)


# updated_prompt = """
# I have a full generation trace for a long-form answer to a scientific question. I want a structured critique of both the plan and the answer to make it better.

# The trace has the following structure:
# - <think> blocks contain the model's internal reasoning: question decomposition, search planning, and reflection/synthesis between search rounds.
# - <call_tool> tags represent search queries issued during planning, with attributes for the query parameters (e.g., year, fieldsOfStudy) and the query text as the tag body.
# - <tool_output> tags contain the search results returned for each query, as a list of <snippet id="..."> entries with titles and text passages.
# - <answer> tags contain the final long-form answer, with inline <cite id="..."> tags referencing specific snippet IDs from the tool outputs.

# The trace follows a repeating pattern of <think> → <call_tool> → <tool_output>, potentially across multiple search rounds, followed by the final <answer>.

# I want the following two types of critiques:
# - "local" critiques that point out specific problems in sentences or paragraphs of the plan or answer (e.g., unclear wording, unsupported claims, weak transitions, incorrect citation use). These issues can be fixed by locally editing existing text.
# - "global" critiques that identify broader issues across the plan or answer (e.g., missing sections, poor overall organization, lack of conceptual framing). The tags for global critiques should be one of "add section", "delete section", "add across answer", "remove across answer", "reorganize", "repeated local error", "other".

# Please provide critiques in the following JSON format:

# {{
#   "local": [
#       {{"start": beginning few words, "end": ending few words, "location": "plan" or "answer", "issue": description of the issue, "tag": 3-5 word label for the issue, "search_required": true/false, "s2_search_queries": list of search query dicts (only if search_required is true)}}
#   ],
#   "global": [
#       {{"issue": description of the issue, "location": "plan" or "answer", "tag": 3-5 word label for the issue, "search_required": true/false, "s2_search_queries": list of search query dicts (only if search_required is true)}}
#   ]
# }}

# Guidelines:
# - Be concrete and specific in both lists.
# - Do not include any content outside the JSON object.
# - "location" should be "plan" if the issue is in a <think> or <call_tool> block, or "answer" if it is in the <answer> block.
# - Only set "search_required" to true for critiques with "location": "plan". Plan-level critiques can request additional searches when important papers or topics are missing from the retrieved results, the search queries used were too narrow or misdirected, or claims in the plan need better supporting evidence. For example, if the plan is about attention mechanisms but the Attention Is All You Need paper is absent from the tool outputs, a critique should flag this and generate a search query for it.
# - Always set "search_required" to false for critiques with "location": "answer". The answer will be regenerated after any additional searches are completed, so answer-level issues (including bad or missing citations) do not need search queries — they will be resolved by rewriting the answer using the updated retrieved results.
# - If "search_required" is true, provide one or more search query dicts in "s2_search_queries". These queries will be issued to the Semantic Scholar snippet search API to retrieve relevant passages/papers. Each dict has the format: {{"query": "concise natural-language phrase, excluding info already in year/authors/field_of_study", "year": "YYYY-YYYY or empty string", "authors": [list of author names or empty list], "field_of_study": "comma-separated fields or empty string"}}. Each distinct search should be a separate dict.
# - For global critiques, the "tag" should be one of the following: "add section", "delete section", "add across answer", "remove across answer", "reorganize", "repeated local error", "other". Use other only if none of the other tags fit the issue.

# Example of local critiques:
# {{
#   "start": "Alignment Metrics: Another fine-grained strategy",
#   "end": "on many benchmark tasks.",
#   "location": "answer",
#   "issue": "This paragraph is irrelevant to the question asked and should be removed.",
#   "tag": "irrelevant content",
#   "search_required": false
# }}
# {{
#   "start": "I will retrieve evidence on attention mechanisms",
#   "end": "and multi-head attention variants.",
#   "location": "plan",
#   "issue": "The plan intends to cover attention mechanisms but the retrieved results do not include the foundational Transformer paper (Vaswani et al., 2017), which is essential for this topic.",
#   "tag": "missing key paper",
#   "search_required": true,
#   "s2_search_queries": [
#     {{"query": "Attention is all you need Transformer architecture", "year": "2017-2017", "authors": ["Vaswani"], "field_of_study": "Computer Science"}}
#   ]
# }}
# {{
#   "start": "<cite id="e581542a-1">Another development is to",
#   "end": "with the right prompt.",
#   "location": "answer",
#   "issue": "This citation does not support the claim made in the sentence. The sentence should be revised or dropped when the answer is rewritten.",
#   "tag": "incorrect citation",
#   "search_required": false
# }}

# Example of global critiques:
# {{
#   "issue": "The plan does not search for any recent work on reward model training for math, leaving a significant gap in coverage.",
#   "location": "plan",
#   "tag": "add section",
#   "search_required": true,
#   "s2_search_queries": [
#     {{"query": "training reward models for math", "year": "2022-2025", "authors": [], "field_of_study": "Computer Science"}}
#   ]
# }}
# {{
#   "issue": "Group all the evaluation-related content into a single section to improve organization.",
#   "location": "answer",
#   "tag": "reorganize",
#   "search_required": false
# }}
# {{
#   "issue": "The question asks true or false, but the answer is structured as an essay. Add a true/false explanation at the beginning.",
#   "location": "answer",
#   "tag": "add section",
#   "search_required": false
# }}
# {{
#   "issue": "The paragraph about monte carlo search is not relevant to the question asked and should be removed.",
#   "location": "answer",
#   "tag": "delete section",
#   "search_required": false
# }}


# Here is the full generation trace along with the original question:

# Query: {0}
# Trace: {1}
# """


updated_prompt_v1 = """
I have a full generation trace for a long-form answer to a scientific question. I want a structured critique of both the plan and the answer to make it better.

The trace has the following structure:
- <think> blocks contain the model's internal reasoning: question decomposition, search planning, and reflection/synthesis between search rounds.
- <call_tool> tags represent search queries issued during planning, with attributes for the query parameters (e.g., year, fieldsOfStudy) and the query text as the tag body.
- <tool_output> tags contain the search results returned for each query, as a list of <snippet id="..."> entries with titles and text passages.
- <answer> tags contain the final long-form answer, with inline <cite id="..."> tags referencing specific snippet IDs from the tool outputs.

The trace follows a repeating pattern of <think> → <call_tool> → <tool_output>, potentially across multiple search rounds, followed by the final <answer>.

I want critiques that could help improve the answer. Each critique should point out a specific problem in the plan or answer (e.g., unclear wording, missing sections, missing summary, unsupported claims, weak transitions, incorrect citation use).

Please provide critiques in the following JSON format:

{{
  "local": [
      {{"critique_span": [[beginning few words, ending few words], ...], "edit_span": [[beginning few words, ending few words], ...], "location": "plan" or "answer" or "both", "issue": description of the issue, "tag": 3-5 word label for the issue, "organization_related": true/false, "search_required": true/false, "s2_search_queries": list of search query dicts (only if search_required is true)}}
  ],
}}

Rules:
- "critique_span" is a list of [start, end] pairs identifying the text the critique is about (the diagnostic span); use multiple pairs when the critique applies to disjoint parts, or an empty list if there is no specific span.
- "edit_span" identifies where the fix is applied. Usually it is the same as "critique_span". There might be exceptions like when inserting new content rather than rewriting existing text.
- When inserting new content rather than rewriting existing text, set start and end to the same string to signal an insertion point immediately after that string.
- Be concrete and specific.
- Do not include any content outside the JSON object.
- "location" should be "plan" if the issue is in a <think> or <call_tool> block, or "answer" if it is in the <answer> block.
- Only set "search_required" to true for critiques with "location": "plan" or "both". Plan-level critiques can request additional searches when important papers or topics are missing from the retrieved results, the search queries used were too narrow or misdirected, or claims in the plan need better supporting evidence. For example, if the plan is about attention mechanisms but the Attention Is All You Need paper is absent from the tool outputs, a critique should flag this and generate a search query for it.
- Always set "search_required" to false for critiques with "location": "answer". The answer will be regenerated after any additional searches are completed, so answer-level issues (including bad or missing citations) do not need search queries — they will be resolved by rewriting the answer using the updated retrieved results.
- If "search_required" is true, provide one or more search query dicts in "s2_search_queries". These queries will be issued to the Semantic Scholar snippet search API to retrieve relevant passages/papers. Each dict has the format: {{"query": "concise natural-language phrase, excluding info already in year/authors/field_of_study", "year": "YYYY-YYYY or empty string", "authors": [list of author names or empty list], "field_of_study": "comma-separated fields or empty string"}}. Each distinct search should be a separate dict.
- Set organization_related to true if the critique is about reordering existing content or changing the structure/formatting.

Example of critiques:
{{
  "critique_span": [["Alignment Metrics: Another fine-grained strategy", "on many benchmark tasks."]],
  "edit_span": [["Alignment Metrics: Another fine-grained strategy", "on many benchmark tasks."]],
  "location": "answer",
  "issue": "This paragraph is irrelevant to the question asked and should be removed.",
  "tag": "irrelevant content",
  "search_required": false
}}
{{
  "critique_span": [["I will retrieve evidence on attention mechanisms", "and multi-head attention variants."]],
  "edit_span": [["for large generative LLMs.", "for large generative LLMs."]],
  "location": "plan",
  "issue": "The plan intends to cover attention mechanisms but the retrieved results do not include the foundational Transformer paper (Vaswani et al., 2017), which is essential for this topic. A new search should be inserted after the existing tool output.",
  "tag": "missing key paper",
  "search_required": true,
  "s2_search_queries": [
    {{"query": "Attention is all you need Transformer architecture", "year": "2017-2017", "authors": ["Vaswani"], "field_of_study": "Computer Science"}}
  ]
}}
{{
  "critique_span": [["<cite id="e581542a-1">Another development is to", "with the right prompt."]],
  "edit_span": [["<cite id="e581542a-1">Another development is to", "with the right prompt."]],
  "location": "answer",
  "issue": "This citation does not support the claim made in the sentence. The sentence should be revised or dropped when the answer is rewritten.",
  "tag": "incorrect citation",
  "search_required": false
}}
{{
  "critique_span": [],
  "edit_span": [["rationales can be unreliable. ", "rationales can be unreliable. "]],
  "issue": "The plan does not search for any recent work on reward model training for math, leaving a significant gap in coverage.",
  "location": "both",
  "tag": "add section",
  "search_required": true,
  "s2_search_queries": [
    {{"query": "training reward models for math", "year": "2022-2025", "authors": [], "field_of_study": "Computer Science"}}
  ]
}}
{{
  "critique_span": [["Evaluation results show that", "on the held-out test set."], ["We further evaluate the model", "across all three benchmarks."]],
  "edit_span": [["Evaluation results show that", "on the held-out test set."], ["We further evaluate the model", "across all three benchmarks."]],
  "issue": "Group all the evaluation-related content into a single section to improve organization.",
  "location": "answer",
  "tag": "reorganize",
  "search_required": false
}}
{{
  "critique_span": [],
  "edit_span": [["<answer>\n # Why traditional ", "<answer>\n # Why traditional "]],
  "issue": "The question asks true or false, but the answer is structured as an essay. Add a true/false explanation at the beginning.",
  "location": "answer",
  "tag": "add takeaway sentence",
  "search_required": false
}}
{{
  "critique_span": [["Monte Carlo search has been applied", "across many planning tasks."]],
  "edit_span": [["Monte Carlo search has been applied", "across many planning tasks."]],
  "issue": "The paragraph about monte carlo search is not relevant to the question asked and should be removed.",
  "location": "answer",
  "tag": "delete section",
  "search_required": false
}}


Here is the full generation trace along with the original question:

Query: {0}
Trace: {1}
"""

MODEL_COSTS = {
    # (input_price_per_1M, output_price_per_1M)
    "gpt-4.1":         (2.00,  8.00),
    "gpt-4.1-mini":    (0.40,  1.60),
    "gpt-4.1-nano":    (0.10,  0.40),
    "gpt-4o":          (2.50, 10.00),
    "gpt-4o-mini":     (0.15,  0.60),
    "gpt-5":           (1.25, 10.00),
    "o3":             (10.00, 40.00),
    "o4-mini":         (1.10,  4.40),
    "gpt-5.4":         (2.50, 15.00),
    "gpt-5.6-luna":    (0.20,  1.20),
}

MODEL = "gpt-5.4"
MAX_WORKERS = 10
LIMIT = 10   # how many answers to critique (drtulu_answers.jsonl has 50)
# When set, only generate DR Tulu answers (via localhost:8007) and skip critiquing.
ANSWER_ONLY = os.environ.get("ANSWER_ONLY", "0") == "1"

total_cost = 0.0
cost_lock = threading.Lock()
write_lock = threading.Lock()

input_file = "test_samples/drtulu_answers.jsonl"  # set to a .jsonl path to use a local file, or None to load from HF
output_file = "test_samples/drtulu_answers_w_critiques.jsonl"  # the file the rewrite pipeline reads

if input_file is not None:
    answers_output_fname = input_file.replace('.jsonl', '_w_critiques.jsonl')
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    data = data[:LIMIT]
else:
    answers_output_fname = 'test_50/drtulu_answers.jsonl'
    ds = load_dataset("rl-research/dr-tulu-rl-data", split="train")
    ds = ds.filter(lambda row: 'sqa_1k' in (row['source'] or ''))
    data = []
    for row in ds:
        msgs = row['messages']
        question = next((m['content'] for m in msgs if m['role'] == 'user'), None)
        if question:
            data.append({'problem': question})
        if len(data) >= LIMIT:
            break

# Resume: skip answers that already have critiques in output_file, so re-running
# only fills in the missing ones (no duplicates, no re-paying for finished work).
done = set()
if os.path.exists(output_file):
    with open(output_file) as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line)['question'])
                except Exception:
                    pass
_before = len(data)
data = [d for d in data if d['problem'] not in done]
print(f"{_before} candidates | {len(done)} already critiqued | {len(data)} to generate now.")


def process_sample(sample):
    global total_cost
    problem = sample['problem']

    if 'full_traces' not in sample or 'final_response' not in sample:
        r1 = requests.post("http://localhost:8007/ask", json={"question": problem})
        r1_json = r1.json()
        generated_text = r1_json['trace']['generated_text']
        answer = r1_json['answer']
        with write_lock:
            with open(answers_output_fname, 'a') as f:
                f.write(json.dumps({
                    'problem': problem,
                    'final_response': answer,
                    'full_traces': generated_text
                }) + '\n')
    else:
        generated_text = sample['full_traces']
        answer = sample['final_response']

    if ANSWER_ONLY:            # generated the answer above; skip critiquing
        return 0.0

    prompt = updated_prompt_v1.format(problem, generated_text)
    response = client.responses.create(model=MODEL, input=prompt)

    input_price, output_price = MODEL_COSTS.get(MODEL, (0.0, 0.0))
    cost = response.usage.input_tokens * input_price / 1_000_000 + response.usage.output_tokens * output_price / 1_000_000

    with write_lock:
        with open(output_file, 'a') as f:
            f.write(json.dumps({
                'question': problem,
                'original_answer': answer,
                'original_trace': generated_text,
                'critique': response.output_text
            }) + '\n')

    with cost_lock:
        total_cost += cost
        print(f"[{problem[:50]}]  cost=${cost:.4f}  total=${total_cost:.4f}")

    return cost


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_sample, sample): sample for sample in data}
    for future in tqdm(as_completed(futures), total=len(data)):
        try:
            future.result()
        except Exception as e:
            sample = futures[future]
            print(f"Error on '{sample['problem'][:60]}': {e}")

print(f"\nTotal cost: ${total_cost:.4f}")
