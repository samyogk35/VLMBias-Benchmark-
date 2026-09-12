import pytest
from src.eval.parse_answer import parse_answer, score


@pytest.mark.parametrize("raw,expected,status", [
    ("{Yes}", "Yes", "valid"),
    ("{No}", "No", "valid"),
    ("{no}", "No", "valid"),
    ("Yes", "Yes", "valid"),
    ("No.", "No", "valid"),
    ("{Yes}. The two lines are equal in length.", "Yes", "valid"),
    ("No, the two inner circles are not equal in size.", "No", "valid"),
    ("The answer is {No}.", "No", "valid"),
    ("{Yes, they are equal}", "Yes", "valid"),
    ("{Yes} {Yes}", "Yes", "valid"),
    ("{Yes} or {No}", None, "ambiguous"),   # model just repeated the prompt
    ("Yes. No.", None, "ambiguous"),
    ("The lines look equal but are not.", None, "invalid"),
    ("", None, "invalid"),
    ("{}", None, "invalid"),
    ("{Yes/No}", None, "ambiguous"),
    ("Nobody knows.", None, "invalid"),     # "no" inside a word shouldn't count
])
def test_yes_no(raw, expected, status):
    r = parse_answer(raw, "yes_no")
    assert (r.parsed_answer, r.parse_status) == (expected, status)


@pytest.mark.parametrize("raw,expected,status", [
    ("{5}", "5", "valid"),
    ("{ 4 }", "4", "valid"),
    ("4", "4", "valid"),
    ("There are 4 legs.", "4", "valid"),
    ("There are four legs.", "4", "valid"),
    ("{four}", "4", "valid"),
    ("The flag has 50 stars. {50}", "50", "valid"),
    ("The logo has 3 stripes {3}", "3", "valid"),
    ("{4}. Actually there are 5.", "4", "valid"),   # braces win
    ("There are 4 legs, not 5.", None, "ambiguous"),
    ("{4} {5}", None, "ambiguous"),
    ("{number}", None, "invalid"),
    ("I cannot tell.", None, "invalid"),
    ("", None, "invalid"),
    ("0", "0", "valid"),
    ("{05}", "5", "valid"),
    ("There are 5.", "5", "valid"),
    ("3.5", None, "invalid"),
    ("v1.5 has 4 stripes", "4", "valid"),
])
def test_number(raw, expected, status):
    r = parse_answer(raw, "number")
    assert (r.parsed_answer, r.parse_status) == (expected, status)


def test_score():
    r = parse_answer("{No}", "yes_no")
    assert score(r, "No", "Yes") == (True, False)
    assert score(r, "Yes", "No") == (False, True)
    assert score(parse_answer("{5}", "number"), 5, 4) == (True, False)
    # not-valid parses don't get scored at all
    assert score(parse_answer("", "number"), "5", "4") == (None, None)
    assert score(parse_answer("{4} {5}", "number"), "5", "4") == (None, None)


def test_bad_answer_type():
    with pytest.raises(ValueError):
        parse_answer("x", "float")
