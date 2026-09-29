# Paired statistics for comparing two decoding conditions on the same rows. Inputs are
# aligned 0/1 correctness vectors (a = baseline, b = the other condition), one entry per row.
# Pure functions only; the script that runs them on real records is run_paired_stats.py.

from math import comb

import numpy as np


def transition_table(a, b):
    """Counts of fixed (a wrong, b right), broken (a right, b wrong), same_right, same_wrong."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length")
    return {
        "fixed": int((~a & b).sum()),
        "broken": int((a & ~b).sum()),
        "same_right": int((a & b).sum()),
        "same_wrong": int((~a & ~b).sum()),
    }


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the two discordant counts.

    Under the null each discordant row is equally likely to go either way, so the smaller
    count is Binomial(b + c, 1/2); the p-value is twice its lower tail, capped at 1.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_bootstrap_diff(a, b, cluster_key, n=10_000, seed=0):
    """mean(b) - mean(a) with a 95% percentile CI, resampling whole clusters.

    Rows that share a cluster_key (e.g. Q1 and Q2 on the same image) are drawn together,
    so the interval accounts for them not being independent.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keys = np.asarray(cluster_key)
    if not (a.shape == b.shape == keys.shape):
        raise ValueError("a, b and cluster_key must have the same length")

    _, idx = np.unique(keys, return_inverse=True)
    n_clusters = idx.max() + 1
    diff_sum = np.bincount(idx, weights=b - a, minlength=n_clusters)
    size = np.bincount(idx, minlength=n_clusters).astype(float)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n, n_clusters))
    boot = diff_sum[draws].sum(axis=1) / size[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"estimate": float((b - a).mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_rows": int(len(a)), "n_clusters": int(n_clusters)}
