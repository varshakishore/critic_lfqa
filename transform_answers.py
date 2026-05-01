#!/usr/bin/env python3
"""
Apply synthetic transformation rules to long-form answers.

Usage:
    # Plain text file
    python transform_answers.py answer.txt

    # JSONL — transforms the 'final_response' field in each record
    python transform_answers.py sqav2.jsonl -o transformed.jsonl

    # With citation metadata for the three citation-dependent rules
    python transform_answers.py sqav2.jsonl -m metadata.json -o transformed.jsonl

Citation metadata format (metadata.json):
    {
      "cite-id-1": {"year": 2026, "citation_count": 800, "is_survey": false},
      "cite-id-2": {"year": 2023, "citation_count": 120, "is_survey": true},
      ...
    }
"""

import re
import json
import argparse
from typing import Optional


# ─── Number helpers ───────────────────────────────────────────────────────────

_ONES = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen',
]
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']


def _int_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] if o == 0 else f"{_TENS[t]}-{_ONES[o]}"


# ─── Sentence splitting ───────────────────────────────────────────────────────

# Abbreviations whose periods should NOT trigger sentence splits
_ABBREVS = ['et al.', 'Dr.', 'Prof.', 'Fig.', 'Eq.', 'Sec.', 'approx.', 'vs.',
            'cf.', 'e.g.', 'i.e.', 'p.', 'pp.', 'no.', 'vol.']


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences while protecting common abbreviations.
    Splits on . ! ? followed by whitespace and an uppercase letter or '<'.
    """
    protected = text
    tokens: dict[str, str] = {}
    for i, abbr in enumerate(_ABBREVS):
        tok = f'\x00A{i}\x00'
        tokens[tok] = abbr
        protected = protected.replace(abbr, tok)

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z<\[])', protected)

    result = []
    for part in parts:
        for tok, abbr in tokens.items():
            part = part.replace(tok, abbr)
        if part.strip():
            result.append(part)
    return result


# ─── Rules ───────────────────────────────────────────────────────────────────

def rule_contractions(text: str) -> str:
    """Expand contractions to full forms."""
    # Ordered: negatives first (more specific), then pronoun forms
    pairs = [
        ("can't", "cannot"),       ("Can't", "Cannot"),
        ("won't", "will not"),     ("Won't", "Will not"),
        ("don't", "do not"),       ("Don't", "Do not"),
        ("doesn't", "does not"),   ("Doesn't", "Does not"),
        ("didn't", "did not"),     ("Didn't", "Did not"),
        ("wouldn't", "would not"), ("Wouldn't", "Would not"),
        ("couldn't", "could not"), ("Couldn't", "Could not"),
        ("shouldn't", "should not"),("Shouldn't", "Should not"),
        ("isn't", "is not"),       ("Isn't", "Is not"),
        ("aren't", "are not"),     ("Aren't", "Are not"),
        ("wasn't", "was not"),     ("Wasn't", "Was not"),
        ("weren't", "were not"),   ("Weren't", "Were not"),
        ("haven't", "have not"),   ("Haven't", "Have not"),
        ("hasn't", "has not"),     ("Hasn't", "Has not"),
        ("hadn't", "had not"),     ("Hadn't", "Had not"),
        ("it's", "it is"),         ("It's", "It is"),
        ("it'll", "it will"),      ("It'll", "It will"),
        ("it'd", "it would"),      ("It'd", "It would"),
        ("I'm", "I am"),           ("I've", "I have"),
        ("I'll", "I will"),        ("I'd", "I would"),
        ("they're", "they are"),   ("They're", "They are"),
        ("they've", "they have"),  ("They've", "They have"),
        ("they'll", "they will"),  ("They'll", "They will"),
        ("they'd", "they would"),  ("They'd", "They would"),
        ("we're", "we are"),       ("We're", "We are"),
        ("we've", "we have"),      ("We've", "We have"),
        ("we'll", "we will"),      ("We'll", "We will"),
        ("we'd", "we would"),      ("We'd", "We would"),
        ("you're", "you are"),     ("You're", "You are"),
        ("you've", "you have"),    ("You've", "You have"),
        ("you'll", "you will"),    ("You'll", "You will"),
        ("you'd", "you would"),    ("You'd", "You would"),
        ("that's", "that is"),     ("That's", "That is"),
        ("there's", "there is"),   ("There's", "There is"),
        ("here's", "here is"),     ("Here's", "Here is"),
        ("who's", "who is"),       ("Who's", "Who is"),
        ("what's", "what is"),     ("What's", "What is"),
        ("let's", "let us"),       ("Let's", "Let us"),
        ("he's", "he is"),         ("He's", "He is"),
        ("she's", "she is"),       ("She's", "She is"),
        ("he'll", "he will"),      ("He'll", "He will"),
        ("she'll", "she will"),    ("She'll", "She will"),
        ("he'd", "he would"),      ("He'd", "He would"),
        ("she'd", "she would"),    ("She'd", "She would"),
    ]
    for contraction, expansion in pairs:
        text = text.replace(contraction, expansion)
    return text


def rule_abbreviations(text: str) -> str:
    """Expand e.g., i.e., etc., vs. to full forms."""
    subs = [
        (r'\be\.g\.,\s*', 'for example, '),
        (r'\be\.g\.\s+', 'for example, '),
        (r'\bi\.e\.,\s*', 'that is, '),
        (r'\bi\.e\.\s+', 'that is, '),
        (r'\betc\.\s*', 'et cetera, '),
        (r'\bvs\.\s+', 'versus '),
    ]
    for pat, rep in subs:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


def rule_percent_to_word(text: str) -> str:
    """Replace X% with 'X percent'."""
    return re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)


def rule_numbers_to_words(text: str) -> str:
    """
    Spell out integers 0–99 as words.
    Skips: list-item lines, years, numbers in model names (GPT-4), scientific notation.
    """
    def replacer(m: re.Match) -> str:
        return _int_to_words(int(m.group(0)))

    lines = text.split('\n')
    result = []
    for line in lines:
        # Skip list-item lines — the leading number is structural
        if re.match(r'^\s*\d+[).]\s', line):
            result.append(line)
            continue
        # Replace standalone 1–2 digit numbers:
        #   not preceded by: digit, letter, #, -, ^, ', "
        #   not followed by:  digit, letter, %, -, ^
        line = re.sub(
            r'(?<![0-9a-zA-Z#\-^\'"]) \b([0-9]{1,2})\b (?![0-9a-zA-Z%\-^])',
            replacer,
            line,
            flags=re.VERBOSE,
        )
        result.append(line)
    return '\n'.join(result)


def rule_common_synonyms(text: str) -> str:
    """
    Replace common words with less-frequent synonyms.
    Skips words that are part of hyphenated compounds (e.g. 'Tool‑use').
    """
    # Any hyphen-like character that can form a compound word
    _H = r'(?<![-\u2010\u2011\u2012\u2013])'  # lookbehind: not after a hyphen
    _Hf = r'(?![-\u2010\u2011\u2012\u2013])'   # lookahead:  not before a hyphen

    def p(word: str) -> str:
        """Wrap word pattern to exclude hyphenated-compound positions."""
        return _H + r'\b' + word + r'\b' + _Hf

    pairs = [
        # verbs
        (p(r'[Ss]how'),      'demonstrate'),
        (p(r'[Ss]hows'),     'demonstrates'),
        (p(r'[Ss]howed'),    'demonstrated'),
        (p(r'[Ss]howing'),   'demonstrating'),
        (p(r'[Ss]hown'),     'demonstrated'),
        (p(r'[Ff]ind'),      'determine'),
        (p(r'[Ff]inds'),     'determines'),
        (p(r'[Ff]ound'),     'identified'),
        (p(r'[Gg]et'),       'obtain'),
        (p(r'[Gg]ets'),      'obtains'),
        (p(r'[Gg]ot'),       'obtained'),
        (p(r'[Gg]etting'),   'obtaining'),
        (p(r'[Nn]eed'),      'require'),
        (p(r'[Nn]eeds'),     'requires'),
        (p(r'[Nn]eeded'),    'required'),
        (p(r'[Nn]eeding'),   'requiring'),
        (p(r'[Hh]elp'),      'facilitate'),
        (p(r'[Hh]elps'),     'facilitates'),
        (p(r'[Hh]elped'),    'facilitated'),
        (p(r'[Ii]ncrease'),  'augment'),
        (p(r'[Ii]ncreases'), 'augments'),
        (p(r'[Ii]ncreased'), 'augmented'),
        (p(r'[Dd]ecrease'),  'diminish'),
        (p(r'[Dd]ecreases'), 'diminishes'),
        (p(r'[Dd]ecreased'), 'diminished'),
        # adjectives
        (p(r'[Ii]mportant'), 'significant'),
        (p(r'[Gg]ood'),      'effective'),
        (p(r'[Bb]ad'),       'detrimental'),
        (p(r'[Bb]ig'),       'substantial'),
        # adverbs / quantifiers
        (p(r'[Oo]ften'),     'frequently'),
        (p(r'[Mm]any'),      'numerous'),
        (p(r'[Aa]lso'),      'additionally'),
        (p(r'[Aa]lways'),    'consistently'),
    ]
    for pat, rep in pairs:
        text = re.sub(pat, rep, text)
    return text


def rule_acronym_consistency(text: str) -> str:
    """
    After the first 'Full Name (ACRONYM)' introduction, replace all
    subsequent uses of 'Full Name' with 'ACRONYM' (including plurals).
    """
    intro_re = re.compile(r'([A-Z][a-z]+(?:[ \-][A-Za-z]+){1,6})\s+\(([A-Z]{2,8})\)')

    seen: dict[str, str] = {}
    for m in intro_re.finditer(text):
        full, acronym = m.group(1), m.group(2)
        if full not in seen:
            seen[full] = acronym

    for full, acronym in seen.items():
        intro = f"{full} ({acronym})"
        idx = text.find(intro)
        if idx == -1:
            continue
        after_pos = idx + len(intro)
        before = text[:after_pos]
        after = text[after_pos:]
        # Replace full form (including plural -s) after the intro
        after = re.sub(r'\b' + re.escape(full) + r's?\b', acronym, after)
        text = before + after

    return text


def rule_acronyms_lowercase(text: str) -> str:
    """
    Lowercase all standalone uppercase acronyms (2–8 chars).
    Protects HTML attribute values (e.g., cite id="...") from modification.
    """
    # Temporarily mask HTML attribute values
    masked: dict[str, str] = {}

    def mask_attr(m: re.Match) -> str:
        key = f'\x00M{len(masked)}\x00'
        masked[key] = m.group(0)
        return key

    text = re.sub(r'(?:id|class|href)="[^"]*"', mask_attr, text)
    text = re.sub(r'\b[A-Z]{2,8}\b', lambda m: m.group(0).lower(), text)

    for key, val in masked.items():
        text = text.replace(key, val)
    return text


def rule_list_format(text: str) -> str:
    """Convert all list items (ordered or unordered, any indent) to '* ...' format."""
    lines = text.split('\n')
    result = []
    for line in lines:
        m = re.match(r'^(\s*)(?:\d+[).:]|\*|\-|•)\s+(.*)', line)
        if m:
            result.append(f"{m.group(1)}(*) {m.group(2)}")
        else:
            result.append(line)
    return '\n'.join(result)


def rule_list_items_period(text: str) -> str:
    """Ensure every list item ends with a period."""
    lines = text.split('\n')
    result = []
    for line in lines:
        if re.match(r'^\s*\(\*\)\s+\S', line):
            stripped = line.rstrip()
            if stripped and stripped[-1] not in '.!?':
                line = stripped + '.'
        result.append(line)
    return '\n'.join(result)


def rule_paragraph_split(text: str) -> str:
    """Split paragraphs that contain more than 6 sentences into two halves."""
    paragraphs = re.split(r'\n{2,}', text)
    result = []

    for para in paragraphs:
        # Don't split headers or list blocks
        if para.lstrip().startswith('#') or re.match(r'^\s*\d+\)\s', para):
            result.append(para)
            continue

        sentences = _split_sentences(para)
        if len(sentences) > 6:
            mid = len(sentences) // 2
            result.append(' '.join(sentences[:mid]))
            result.append(' '.join(sentences[mid:]))
        else:
            result.append(para)

    return '\n\n'.join(result)


def rule_extra_newlines(text: str) -> str:
    """Add an extra blank line between paragraphs (2 newlines → 3)."""
    return re.sub(r'\n{2,}', '\n\n\n', text)


def rule_consecutive_same_citation(text: str) -> str:
    """
    Within each paragraph, when two consecutive <cite> blocks share the same id,
    merge them into one by dropping </cite> from the end of the first block and
    <cite id="..."> from the start of the second block.
    """
    cite_re = re.compile(r'<cite id="([^"]+)">(.*?)</cite>', re.DOTALL)

    def _process_paragraph(para: str) -> str:
        matches = list(cite_re.finditer(para))
        removals = []  # list of (start, end) char ranges to delete
        for i in range(len(matches) - 1):
            if matches[i].group(1) == matches[i + 1].group(1):
                # drop </cite> from end of first block
                removals.append((matches[i].end() - len('</cite>'), matches[i].end()))
                # drop <cite id="..."> from start of second block
                open_tag = f'<cite id="{matches[i+1].group(1)}">'
                removals.append((matches[i + 1].start(), matches[i + 1].start() + len(open_tag)))
        if not removals:
            return para
        result = para
        for start, end in sorted(removals, reverse=True):
            result = result[:start] + result[end:]
        return result

    paragraphs = re.split(r'(\n{2,})', text)  # keep the separators
    return ''.join(
        _process_paragraph(chunk) if not re.fullmatch(r'\n{2,}', chunk) else chunk
        for chunk in paragraphs
    )


# ─── Citation-dependent rules ─────────────────────────────────────────────────

def rule_recently_post2025(text: str, metadata: dict) -> str:
    """
    Prepend 'Recently, ' to cited sentences whose paper year > 2025,
    if 'recent' is not already present.
    """
    def process(m: re.Match) -> str:
        cite_id, sentence = m.group(1), m.group(2)
        year = metadata.get(cite_id, {}).get('year', 0)
        if year > 2025 and 'recent' not in sentence.lower():
            sentence = 'Recently, ' + sentence[0].lower() + sentence[1:]
        return f'<cite id="{cite_id}">{sentence}</cite>'

    return re.sub(r'<cite id="([^"]+)">(.*?)</cite>', process, text, flags=re.DOTALL)


def rule_seminal_highly_cited(text: str, metadata: dict) -> str:
    """
    Add 'seminal' or 'foundational' to cited sentences where citation count > 500.
    """
    def process(m: re.Match) -> str:
        cite_id, sentence = m.group(1), m.group(2)
        count = metadata.get(cite_id, {}).get('citation_count', 0)
        already_tagged = 'seminal' in sentence.lower() or 'foundational' in sentence.lower()
        if count > 500 and not already_tagged:
            # Try to insert after "Author et al." pattern
            new = re.sub(
                r'(\b[A-Z][a-z]+ et al\.)',
                r"\1's seminal",
                sentence,
                count=1,
            )
            if new == sentence:
                # Fallback: prepend
                sentence = 'This foundational work ' + sentence[0].lower() + sentence[1:]
            else:
                sentence = new
        return f'<cite id="{cite_id}">{sentence}</cite>'

    return re.sub(r'<cite id="([^"]+)">(.*?)</cite>', process, text, flags=re.DOTALL)


def rule_survey_mention(text: str, metadata: dict) -> str:
    """
    Prepend 'In a survey of the literature, ' to cited sentences that
    reference a survey or meta-analysis paper, if 'survey' is not already present.
    """
    def process(m: re.Match) -> str:
        cite_id, sentence = m.group(1), m.group(2)
        is_survey = metadata.get(cite_id, {}).get('is_survey', False)
        if is_survey and 'survey' not in sentence.lower():
            sentence = 'In a survey of the literature, ' + sentence[0].lower() + sentence[1:]
        return f'<cite id="{cite_id}">{sentence}</cite>'

    return re.sub(r'<cite id="([^"]+)">(.*?)</cite>', process, text, flags=re.DOTALL)


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def transform_answer(text: str, citation_metadata: Optional[dict] = None) -> str:
    """Apply all transformation rules in a safe order."""
    # 1. Token-level substitutions
    text = rule_contractions(text)
    text = rule_abbreviations(text)
    text = rule_common_synonyms(text)
    text = rule_percent_to_word(text)
    text = rule_numbers_to_words(text)

    # 2. Acronym consistency before casing (so replacements get lowercased too)
    text = rule_acronym_consistency(text)

    # 3. Structural changes
    text = rule_list_format(text)
    text = rule_list_items_period(text)
    text = rule_paragraph_split(text)
    text = rule_extra_newlines(text)

    # 4. Acronym casing (after consistency so new occurrences are also lowercased)
    text = rule_acronyms_lowercase(text)

    # 5. Citation-based rules
    text = rule_consecutive_same_citation(text)
    if citation_metadata:
        text = rule_recently_post2025(text, citation_metadata)
        text = rule_seminal_highly_cited(text, citation_metadata)
        text = rule_survey_mention(text, citation_metadata)

    return text


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Apply synthetic transformations to long-form answers.'
    )
    parser.add_argument('--input', help='Input .txt or .jsonl file')
    parser.add_argument('-o', '--output', required=True, help='Output .jsonl file (appended)')
    parser.add_argument(
        '-m', '--metadata',
        help='JSON file mapping cite_id → {year, citation_count, is_survey}',
    )
    args = parser.parse_args()

    citation_metadata: Optional[dict] = None
    if args.metadata:
        with open(args.metadata) as f:
            citation_metadata = json.load(f)

    with open(args.output, 'a') as out, open(args.input) as f:
        for line in f:
            line = line.strip()
            obj = json.loads(line)
            answer = obj['full_traces']['generated_text']
            # extract everything betwee <answer> and </answer> tags
            answer = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL).group(0).strip()
            # extract everything before <answer> tag
            raw = re.search(r'^(.*?)<answer>', obj['full_traces']['generated_text'], re.DOTALL).group(1)
            new_answer = transform_answer(answer, None)
            out.write(json.dumps({"example_id": obj.get("example_id"), "question": obj.get("problem"), "original": answer, "rewritten": new_answer, "raw": raw}) + '\n')


if __name__ == '__main__':
    main()
