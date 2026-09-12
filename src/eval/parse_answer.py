"""Answer parser for VLMBias-style prompts.

Two answer types are supported:
  * ``yes_no``  -- optical-illusion prompts ("Answer in curly brackets, e.g., {Yes} or {No}.")
  * ``number``  -- counting prompts ("Answer with a number in curly brackets, e.g., {5}.")

Every parse returns a ``ParseResult`` with ``parse_status`` in {"valid", "ambiguous", "invalid"}.
The parser is deliberately conservative: when a raw output contains two competing answers
("{Yes}. No, wait ...") it reports ``ambiguous`` instead of guessing, and when it contains
nothing usable it reports ``invalid``. Callers must never score ambiguous/invalid parses
as ordinary wrong answers (SPEC §7).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

_BRACE_RE = re.compile(r"\{([^{}]*)\}")
_YES_RE = re.compile(r"\b(yes|yeah|yep|true)\b", re.I)
_NO_RE = re.compile(r"\b(no|nope|false)\b", re.I)
# integers; reject only decimal continuations ("3.5"), not sentence-final periods ("5.")
_INT_RE = r"(?<!\d)(?<!\d\.)(\d+)(?!\d|\.\d)"

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.I)


@dataclass
class ParseResult:
    parsed_answer: Optional[str]   # normalised: "Yes"/"No" or a decimal integer string
    parse_status: str              # valid | ambiguous | invalid
    method: str                    # which rule fired (for audit tables)

    def to_dict(self):
        return asdict(self)


def _distinct(values):
    out = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def _yes_no_candidates(text: str):
    found = []
    for m in re.finditer(r"\b(yes|yeah|yep|true|no|nope|false)\b", text, re.I):
        w = m.group(1).lower()
        found.append("Yes" if w in ("yes", "yeah", "yep", "true") else "No")
    return _distinct(found)


def _number_candidates(text: str):
    found = []
    # walk tokens in order so that method reporting stays sensible
    for m in re.finditer(_INT_RE + r"|\b(" + "|".join(_NUMBER_WORDS) + r")\b", text, re.I):
        if m.group(1) is not None:
            found.append(str(int(m.group(1))))
        else:
            found.append(str(_NUMBER_WORDS[m.group(2).lower()]))
    return _distinct(found)


def _resolve(cands, method):
    if len(cands) == 1:
        return ParseResult(cands[0], "valid", method)
    if len(cands) > 1:
        return ParseResult(None, "ambiguous", method)
    return None


def parse_answer(raw_output: str, answer_type: str) -> ParseResult:
    """Parse a raw model completion into a normalised answer.

    Priority: (1) curly-brace content, (2) the bare text. Within each stage, exactly one
    distinct candidate => valid; several distinct candidates => ambiguous; none => fall through.
    """
    if answer_type not in ("yes_no", "number"):
        raise ValueError(f"unknown answer_type {answer_type!r}")
    text = (raw_output or "").strip()
    if not text:
        return ParseResult(None, "invalid", "empty")
    extract = _yes_no_candidates if answer_type == "yes_no" else _number_candidates

    # Stage 1: curly braces. Braces are what the prompt asks for, so they take precedence
    # even if the surrounding sentence mentions other values.
    braces = [b.strip() for b in _BRACE_RE.findall(text) if b.strip()]
    if braces:
        cands = _distinct([c for b in braces for c in extract(b)])
        r = _resolve(cands, "braces")
        if r is not None:
            return r
        # braces exist but contain nothing parseable ("{}" or "{number}") -> fall through

    # Stage 2: bare text.
    cands = extract(text)
    r = _resolve(cands, "bare")
    if r is not None:
        return r
    return ParseResult(None, "invalid", "no_candidate")


def score(parsed: ParseResult, ground_truth: str, expected_bias: Optional[str] = None):
    """Return (is_correct, is_bias_answer); both None when the parse is not valid."""
    if parsed.parse_status != "valid":
        return None, None
    gt = str(ground_truth).strip().lower()
    pa = str(parsed.parsed_answer).strip().lower()
    is_correct = pa == gt
    is_bias = None
    if expected_bias is not None:
        is_bias = pa == str(expected_bias).strip().lower()
    return is_correct, is_bias
