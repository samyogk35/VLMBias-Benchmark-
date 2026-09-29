import pytest
from src.analysis.paired_stats import cluster_bootstrap_diff, mcnemar_exact, transition_table

# 6 rows in 3 clusters of 2. By hand:
#   row  a  b   kind
#   0    0  1   fixed
#   1    0  1   fixed
#   2    1  0   broken
#   3    1  1   same_right
#   4    0  0   same_wrong
#   5    0  1   fixed
A = [0, 0, 1, 1, 0, 0]
B = [1, 1, 0, 1, 0, 1]
KEYS = ["x", "x", "y", "y", "z", "z"]


def test_transition_table():
    assert transition_table(A, B) == {"fixed": 3, "broken": 1, "same_right": 1, "same_wrong": 1}


def test_transition_table_length_mismatch():
    with pytest.raises(ValueError):
        transition_table([0, 1], [1])


@pytest.mark.parametrize("b,c,p", [
    (0, 0, 1.0),
    (3, 1, 0.625),          # 2 * (1 + 4) / 16
    (1, 3, 0.625),          # symmetric
    (5, 0, 0.0625),         # 2 / 32
    (10, 0, 2 / 1024),
    (2, 2, 1.0),            # capped at 1
    (20, 10, 0.098737),     # 2 * P(X <= 10), X ~ Bin(30, 1/2)
])
def test_mcnemar_exact(b, c, p):
    assert mcnemar_exact(b, c) == pytest.approx(p, abs=1e-6)


def test_mcnemar_negative():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_bootstrap_estimate_and_ci():
    r = cluster_bootstrap_diff(A, B, KEYS, n=5000, seed=0)
    # b - a per row = [1, 1, -1, 0, 0, 1], mean 2/6
    assert r["estimate"] == pytest.approx(2 / 6)
    assert (r["n_rows"], r["n_clusters"]) == (6, 3)
    # cluster diffs are x=+1.0, y=-0.5, z=+0.5 (each cluster has 2 rows); any resample's
    # mean lies between the smallest and largest cluster mean
    assert -0.5 <= r["ci_low"] <= r["estimate"] <= r["ci_high"] <= 1.0


def test_bootstrap_is_seeded():
    assert cluster_bootstrap_diff(A, B, KEYS, n=500, seed=1) == cluster_bootstrap_diff(A, B, KEYS, n=500, seed=1)


def test_bootstrap_no_difference_gives_zero_interval():
    r = cluster_bootstrap_diff([1, 0, 1, 0], [1, 0, 1, 0], ["p", "p", "q", "q"], n=200)
    assert (r["estimate"], r["ci_low"], r["ci_high"]) == (0.0, 0.0, 0.0)


def test_bootstrap_resamples_clusters_not_rows():
    # one cluster fully +1, one fully 0. Row-level resampling of 4 rows could give 0.25 or
    # 0.75, but a cluster resample can only give 0, 0.5 or 1.
    r = cluster_bootstrap_diff([0, 0, 0, 0], [1, 1, 0, 0], ["p", "p", "q", "q"], n=2000, seed=0)
    assert r["ci_low"] in (0.0, 0.5, 1.0) and r["ci_high"] in (0.0, 0.5, 1.0)
