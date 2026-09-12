# Parse the model's answer. Two kinds of prompts in VLMBias: yes/no (optical illusions)
# and counting (everything else). If I can't tell what the answer is I mark it ambiguous
# or invalid instead of guessing, those get counted separately, never as plain wrong.

import re
from dataclasses import dataclass, asdict

BRACES = re.compile(r"\{([^{}]*)\}")
YESNO = re.compile(r"\b(yes|yeah|yep|true|no|nope|false)\b", re.I)

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
# integers, but not the parts of a decimal like 3.5 (a trailing period like "5." is fine)
INT = r"(?<!\d)(?<!\d\.)(\d+)(?!\d|\.\d)"
NUMBER = re.compile(INT + r"|\b(" + "|".join(NUMBER_WORDS) + r")\b", re.I)


@dataclass
class ParseResult:
    parsed_answer: object   # "Yes"/"No" or an integer as a string, None if not valid
    parse_status: str       # valid / ambiguous / invalid
    method: str             # braces / bare / empty / no_candidate

    def to_dict(self):
        return asdict(self)


def uniq(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def yes_no_candidates(text):
    found = []
    for m in YESNO.finditer(text):
        w = m.group(1).lower()
        found.append("Yes" if w in ("yes", "yeah", "yep", "true") else "No")
    return uniq(found)


def number_candidates(text):
    found = []
    for m in NUMBER.finditer(text):
        if m.group(1) is not None:
            found.append(str(int(m.group(1))))
        else:
            found.append(str(NUMBER_WORDS[m.group(2).lower()]))
    return uniq(found)


def parse_answer(raw_output, answer_type):
    if answer_type == "yes_no":
        extract = yes_no_candidates
    elif answer_type == "number":
        extract = number_candidates
    else:
        raise ValueError("unknown answer_type %r" % answer_type)

    text = (raw_output or "").strip()
    if not text:
        return ParseResult(None, "invalid", "empty")

    # the prompt asks for {Yes}/{No} or {number}, so if there are braces trust them first
    braces = [b.strip() for b in BRACES.findall(text) if b.strip()]
    if braces:
        cands = uniq([c for b in braces for c in extract(b)])
        if len(cands) == 1:
            return ParseResult(cands[0], "valid", "braces")
        if len(cands) > 1:
            return ParseResult(None, "ambiguous", "braces")
        # braces with nothing useful in them ("{number}"), fall through to the whole text

    cands = extract(text)
    if len(cands) == 1:
        return ParseResult(cands[0], "valid", "bare")
    if len(cands) > 1:
        return ParseResult(None, "ambiguous", "bare")
    return ParseResult(None, "invalid", "no_candidate")


def score(parsed, ground_truth, expected_bias=None):
    """(is_correct, is_bias_answer). Both None if the parse wasn't valid."""
    if parsed.parse_status != "valid":
        return None, None
    pa = str(parsed.parsed_answer).strip().lower()
    is_correct = pa == str(ground_truth).strip().lower()
    is_bias = None
    if expected_bias is not None:
        is_bias = pa == str(expected_bias).strip().lower()
    return is_correct, is_bias
