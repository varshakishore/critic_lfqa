# %%
import json
from openai import OpenAI
from tqdm import tqdm
import os

# %%
OAI_KEY = os.environ["OPENAI_API_KEY"] 
client = OpenAI(api_key=OAI_KEY)


updated_prompt = """
I have a full generation trace for a long-form answer to a scientific question. I want a structured critique of both the plan and the answer to make it better.

The trace has the following structure:
- <think> blocks contain the model's internal reasoning: question decomposition, search planning, and reflection/synthesis between search rounds.
- <call_tool> tags represent search queries issued during planning, with attributes for the query parameters (e.g., year, fieldsOfStudy) and the query text as the tag body.
- <tool_output> tags contain the search results returned for each query, as a list of <snippet id="..."> entries with titles and text passages.
- <answer> tags contain the final long-form answer, with inline <cite id="..."> tags referencing specific snippet IDs from the tool outputs.

The trace follows a repeating pattern of <think> → <call_tool> → <tool_output>, potentially across multiple search rounds, followed by the final <answer>.

I want the following two types of critiques:
- "local" critiques that point out specific problems in sentences or paragraphs of the plan or answer (e.g., unclear wording, unsupported claims, weak transitions, incorrect citation use). These issues can be fixed by locally editing existing text.
- "global" critiques that identify broader issues across the plan or answer (e.g., missing sections, poor overall organization, lack of conceptual framing). The tags for global critiques should be one of "add section", "delete section", "add across answer", "remove across answer", "reorganize", "repeated local error", "other".

Please provide critiques in the following JSON format:

{{
  "local": [
      {{"start": beginning few words, "end": ending few words, "location": "plan" or "answer", "issue": description of the issue, "tag": 3-5 word label for the issue, "search_required": true/false, "s2_search_queries": list of search query dicts (only if search_required is true)}}
  ],
  "global": [
      {{"issue": description of the issue, "location": "plan" or "answer", "tag": 3-5 word label for the issue, "search_required": true/false, "s2_search_queries": list of search query dicts (only if search_required is true)}}
  ]
}}

Guidelines:
- Be concrete and specific in both lists.
- Do not include any content outside the JSON object.
- "location" should be "plan" if the issue is in a <think> or <call_tool> block, or "answer" if it is in the <answer> block.
- Only set "search_required" to true for critiques with "location": "plan". Plan-level critiques can request additional searches when important papers or topics are missing from the retrieved results, the search queries used were too narrow or misdirected, or claims in the plan need better supporting evidence. For example, if the plan is about attention mechanisms but the Attention Is All You Need paper is absent from the tool outputs, a critique should flag this and generate a search query for it.
- Always set "search_required" to false for critiques with "location": "answer". The answer will be regenerated after any additional searches are completed, so answer-level issues (including bad or missing citations) do not need search queries — they will be resolved by rewriting the answer using the updated retrieved results.
- If "search_required" is true, provide one or more search query dicts in "s2_search_queries". These queries will be issued to the Semantic Scholar snippet search API to retrieve relevant passages/papers. Each dict has the format: {{"query": "concise natural-language phrase, excluding info already in year/authors/field_of_study", "year": "YYYY-YYYY or empty string", "authors": [list of author names or empty list], "field_of_study": "comma-separated fields or empty string"}}. Each distinct search should be a separate dict.
- For global critiques, the "tag" should be one of the following: "add section", "delete section", "add across answer", "remove across answer", "reorganize", "repeated local error", "other". Use other only if none of the other tags fit the issue.

Example of local critiques:
{{
  "start": "Alignment Metrics: Another fine-grained strategy",
  "end": "on many benchmark tasks.",
  "location": "answer",
  "issue": "This paragraph is irrelevant to the question asked and should be removed.",
  "tag": "irrelevant content",
  "search_required": false
}}
{{
  "start": "I will retrieve evidence on attention mechanisms",
  "end": "and multi-head attention variants.",
  "location": "plan",
  "issue": "The plan intends to cover attention mechanisms but the retrieved results do not include the foundational Transformer paper (Vaswani et al., 2017), which is essential for this topic.",
  "tag": "missing key paper",
  "search_required": true,
  "s2_search_queries": [
    {{"query": "Attention is all you need Transformer architecture", "year": "2017-2017", "authors": ["Vaswani"], "field_of_study": "Computer Science"}}
  ]
}}
{{
  "start": "<cite id="e581542a-1">Another development is to",
  "end": "with the right prompt.",
  "location": "answer",
  "issue": "This citation does not support the claim made in the sentence. The sentence should be revised or dropped when the answer is rewritten.",
  "tag": "incorrect citation",
  "search_required": false
}}

Example of global critiques:
{{
  "issue": "The plan does not search for any recent work on reward model training for math, leaving a significant gap in coverage.",
  "location": "plan",
  "tag": "add section",
  "search_required": true,
  "s2_search_queries": [
    {{"query": "training reward models for math", "year": "2022-2025", "authors": [], "field_of_study": "Computer Science"}}
  ]
}}
{{
  "issue": "Group all the evaluation-related content into a single section to improve organization.",
  "location": "answer",
  "tag": "reorganize",
  "search_required": false
}}
{{
  "issue": "The question asks true or false, but the answer is structured as an essay. Add a true/false explanation at the beginning.",
  "location": "answer",
  "tag": "add section",
  "search_required": false
}}
{{
  "issue": "The paragraph about monte carlo search is not relevant to the question asked and should be removed.",
  "location": "answer",
  "tag": "delete section",
  "search_required": false
}}


Here is the full generation trace along with the original question:

Query: {0}
Trace: {1}
"""

total_cost = 0.0

# read jsonl
with open('sqav2_test.jsonl', 'r') as f:
    data = [json.loads(line) for line in f.readlines()]

for i in tqdm(range(len(data))):
    if i==3:
        break
    sample = data[i]
    generated_text = sample['full_traces']['generated_text']
    prompt = updated_prompt.format(sample['problem'], generated_text)

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    # write response to file
    with open('critique_outputs_v2_test.jsonl', 'a') as f:
        f.write(json.dumps({
            'query': sample['problem'],
            'critique': response.output_text
        }) + '\n')

    # compute cost
    cost = response.usage.input_tokens * 1.25 / 1000000 + response.usage.output_tokens * 10 / 1000000
    total_cost += cost
    print(f"Cost for this sample: ${cost:.6f}, Total cost so far: ${total_cost:.6f}")
