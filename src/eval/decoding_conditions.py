# The five decoding conditions and how the two VCD ablations map onto the official sampler.
#
# The official VCD sampler (third_party/VCD/vcd_utils/vcd_sample.py, lines 150-153) does, per step:
#     cutoff   = log(beta) + max(clean)                      # APC: adaptive plausibility constraint
#     diffs    = (1 + alpha) * clean - alpha * noised        # visual contrast
#     cd_logits = diffs.masked_fill(clean < cutoff, -inf)
# so each ablation is one parameter of that same code, with no edit under third_party/:
#     apc_only      alpha = 0  -> diffs == clean exactly; only the APC mask acts
#     contrast_only beta  = 0  -> cutoff == -inf exactly; the mask removes nothing
# vcd_step_logits mirrors those lines so the tests can pin the behaviour down on CPU, and
# OFFICIAL_LINES lets a test check that the vendored sampler still contains them verbatim.

import torch

CONDITIONS = ("greedy", "regular", "apc_only", "contrast_only", "vcd")

# conditions that run the noisy-image branch, with the parameters they override
CD_OVERRIDES = {
    "vcd": {},
    "apc_only": {"cd_alpha": 0.0},
    "contrast_only": {"cd_beta": 0.0},
}

OFFICIAL_LINES = (
    "cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values",
    "diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd",
    'cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))',
)


def cd_params(condition, cd_alpha, cd_beta):
    """Effective (alpha, beta) for a noisy-image condition."""
    p = {"cd_alpha": cd_alpha, "cd_beta": cd_beta}
    p.update(CD_OVERRIDES[condition])
    return p


def vcd_step_logits(clean, noised, cd_alpha, cd_beta):
    """One step of the official sampler's contrast + APC (lines 150-153), same arithmetic."""
    cutoff = torch.log(torch.tensor(cd_beta)) + clean.max(dim=-1, keepdim=True).values
    diffs = (1 + cd_alpha) * clean - cd_alpha * noised
    return diffs.masked_fill(clean < cutoff, -float("inf"))


def expected_unmasked(condition, n_unmasked, vocab_size):
    """Runtime check on the first step's scores: is the APC mask on exactly when it should be?"""
    if condition == "contrast_only":
        return n_unmasked == vocab_size
    if condition in ("vcd", "apc_only"):
        return 0 < n_unmasked < vocab_size
    return True
