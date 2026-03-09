# %%
import json
import re
from openai import OpenAI
from tqdm import tqdm
import os

# %%
OAI_KEY = os.environ["OPENAI_API_KEY"] 
client = OpenAI(api_key=OAI_KEY)


updated_prompt = """
I have a long-form answer to a scientific question. I want a structured critique of the answer to make it better.
 
I want the following two types of critiques:
- "local" critiques that point out specific problems in sentences or paragraphs (e.g., unclear wording, unsupported claims, weak transitions, incorrect citation use). These issues can can be fixed by locally editing existing text.
- "global" critiques that identify broader issues (e.g., missing sections, poor overall organization, lack of conceptual framing). The tags for global critiques shold be one of "add section", "delete section", "add across answer", "remove across answer", "reorganize", "repeated local error", "other".

Please provide critiques in the following JSON format:

{{
  "local": [
      {{"start": beginning few words, "end": ending few words, "issue": description of the issue, "tag": 3-5 word label for the issue, "search_required": true/false}}
  ],
  "global": [
      {{"issue": description of the issue, "tag": 3-5 word label for the issue, "search_required": true/falses, "s2_search_query": search query to find relevant information to fix the issue (only if search_required is true)}}
  ]
}}

Guidelines:
- Be concrete and specific in both lists.
- Do not include any content outside the JSON object.
- If a critique requires additional information for the issue to be resolved, set "search_required" to true; otherwise, set it to false.
- If "search_required" is true, provide a concise search query in the "s2_search_query" field that could be used to find relevant information to address the issue.
- For global critiques, the "tag" should be one of the following: "add section", "delete section", "add across answer", "remove across answer", "reorganize", "repeated local error", "other". Use other only if none of the other tags fit the issue.

Example of local critiques:
{{
  "start": "Alignment Metrics: Another fine-grained strategy",
  "end": "on many benchmark tasks.",
  "issue": "This paragraph is irrelevant to the question asked and should be removed.",
  "tag": "irrelevant content",
  "search_required": false
}}
{{
  "start": "<cite id="e581542a-1">Another development is to",
  "end": "with the right prompt.",
  "issue": "This citation does not support the claim made in the sentence.",
  "tag": "incorrect citation",
  "search_required": true,
  "s2_search_query": "GQA reduces the size of the key-value (KV) cache"
}}

Example of global critiques:
{{
  "issue": "The answer lacks discussion on recent advancements in the field.",
  "tag": "add section",
  "search_required": true,
  "s2_search_query": "recent advancements in training reward models for math"
}}
{{
  "issue": "Group all the evaluation-related content into a single section to improve organization.",
  "tag": "reorganize",
  "search_required": false
}}
{{
  "issue": "The questions asks true or false, but the answer is structured as an essay. Add a true/false explanation at the beginning.",
  "tag": "add section",
  "search_required": false
}}
{{
  "issue": "The paragraph about monte carlo search is not relevant to the question asked and should be removed.",
  "tag": "delete section",
  "search_required": false
}}


Here is the answer along with the original question and citations used:

Query: {0}
Answer: {1}
Citations: {2}
"""

total_cost = 0.0

# read jsonl
with open('sqav2.jsonl', 'r') as f:
    data = [json.loads(line) for line in f.readlines()]

for i in tqdm(range(len(data)//2)):
    if i==3:
        break
    sample = data[i]
    # extract text between <answer> and </answer> tags
    match = re.search(r"<answer>\s*(.*)\s*</answer>", 
                    sample['full_traces']['generated_text'], 
                    flags=re.DOTALL)

    if not match:
        raise ValueError("Answer block not found")

    answer_text = match.group(1)

    tool_outputs = re.findall(
        r"<tool_output>\s*(.*?)\s*</tool_output>",
        sample['full_traces']['generated_text'],
        flags=re.DOTALL
    )
    citations = [t.strip() for t in tool_outputs]

    prompt = updated_prompt.format(sample['problem'], answer_text, citations)

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    # write response to file
    with open('critique_outputs_v1.jsonl', 'a') as f:
        f.write(json.dumps({
            'query': sample['problem'],
            'critique': response.output_text
        }) + '\n')

    # compute cost
    cost = response.usage.input_tokens * 1.25 / 1000000 + response.usage.output_tokens * 10 / 1000000
    total_cost += cost
    print(f"Cost for this sample: ${cost:.6f}, Total cost so far: ${total_cost:.6f}")
