import os

import pytest
import torch

from src.eval.decoding_conditions import (CD_OVERRIDES, CONDITIONS, OFFICIAL_LINES, cd_params, expected_unmasked,
                                          vcd_step_logits)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLER = os.path.join(ROOT, "third_party", "VCD", "vcd_utils", "vcd_sample.py")

# 5-token toy vocab. clean probs ~ [0.64, 0.24, 0.09, 0.02, 0.01]; with beta = 0.2 the APC
# keeps tokens whose clean logit is within log(0.2) of the max: tokens 0 and 1 only.
CLEAN = torch.tensor([[3.0, 2.0, 1.0, -0.5, -1.5]])
NOISED = torch.tensor([[3.5, 0.5, 1.0, -0.5, 2.0]])


@pytest.mark.skipif(not os.path.exists(SAMPLER), reason="third_party/VCD not cloned")
def test_official_sampler_still_has_the_mirrored_lines():
    src = open(SAMPLER).read()
    for line in OFFICIAL_LINES:
        assert line in src, line


def test_conditions_list():
    assert set(CD_OVERRIDES) == {"vcd", "apc_only", "contrast_only"}
    assert set(CD_OVERRIDES) < set(CONDITIONS)


def test_cd_params():
    assert cd_params("vcd", 1.0, 0.2) == {"cd_alpha": 1.0, "cd_beta": 0.2}
    assert cd_params("apc_only", 1.0, 0.2) == {"cd_alpha": 0.0, "cd_beta": 0.2}
    assert cd_params("contrast_only", 1.0, 0.2) == {"cd_alpha": 1.0, "cd_beta": 0.0}


def test_full_vcd_by_hand():
    out = vcd_step_logits(CLEAN, NOISED, 1.0, 0.2)
    # diffs = 2*clean - noised = [2.5, 3.5, 1.0, -0.5, -5.0]; APC keeps tokens 0 and 1
    assert torch.equal(torch.isfinite(out), torch.tensor([[True, True, False, False, False]]))
    assert out[0, :2].tolist() == [2.5, 3.5]


def test_apc_only_is_clean_logits_under_the_mask():
    p = cd_params("apc_only", 1.0, 0.2)
    out = vcd_step_logits(CLEAN, NOISED, **p)
    keep = torch.isfinite(out)
    assert keep.tolist() == [[True, True, False, False, False]]
    # alpha = 0 makes the contrast vanish exactly, whatever the noisy logits are
    assert torch.equal(out[keep], CLEAN[keep])
    assert torch.equal(vcd_step_logits(CLEAN, NOISED * 100, **p), out)


def test_contrast_only_masks_nothing():
    p = cd_params("contrast_only", 1.0, 0.2)
    out = vcd_step_logits(CLEAN, NOISED, **p)
    assert torch.isfinite(out).all()
    assert torch.equal(out, 2 * CLEAN - NOISED)


def test_contrast_only_masks_nothing_even_for_extreme_logits():
    clean = torch.tensor([[1e4, -1e4, 0.0]], dtype=torch.float16)
    out = vcd_step_logits(clean, torch.zeros_like(clean), cd_alpha=1.0, cd_beta=0.0)
    assert not torch.isnan(out).any()
    assert (out > -float("inf")).all()


@pytest.mark.parametrize("cond,n,ok", [
    ("contrast_only", 32000, True), ("contrast_only", 2, False),
    ("vcd", 2, True), ("vcd", 32000, False), ("apc_only", 5, True), ("apc_only", 32000, False),
    ("regular", 32000, True), ("greedy", 32000, True),
])
def test_expected_unmasked(cond, n, ok):
    assert expected_unmasked(cond, n, 32000) is ok
