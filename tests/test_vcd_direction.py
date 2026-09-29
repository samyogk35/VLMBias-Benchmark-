import pytest
from src.analysis.vcd_direction import change_direction


@pytest.mark.parametrize("reg,vcd,bias,expected", [
    ("5", "5", "4", None),                  # unchanged
    (None, None, "4", None),                # both unparseable: unchanged
    ("5", "4", "4", "toward prior"),
    ("4", "5", "4", "away from prior"),
    ("4", "6", "4", "away from prior"),     # away, but to another wrong answer
    ("5", "6", "4", "other"),
    (None, "4", "4", "other"),              # parse status changed wins over direction
    ("4", None, "4", "other"),
    ("No", "Yes", "No", "away from prior"),
    ("Yes", "no", "No", "toward prior"),    # case-insensitive
])
def test_change_direction(reg, vcd, bias, expected):
    assert change_direction(reg, vcd, bias) == expected
