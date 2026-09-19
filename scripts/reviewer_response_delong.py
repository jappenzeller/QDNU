"""
Vendored fast DeLong implementation for comparing two correlated AUCs.

Source: adapted from
  Xu Sun, Weichao Xu, "Fast implementation of DeLong's algorithm for
  comparing the areas under correlated receiver operating characteristic
  curves," IEEE Signal Processing Letters, 21(11):1389-1393, 2014.

This implementation follows the widely-circulated Python port by Yandex
Research and used in the YouDenBVP / scikit-learn auc_compare repos. See
e.g. https://github.com/yandexdataschool/roc_comparison (MIT-licensed).
We reproduce only the two helpers we need: midrank() and delong_roc_test().
"""

import numpy as np


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-indexed mean rank
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Core fast DeLong covariance computation.

    predictions_sorted_transposed: (k, m+n), positives first then negatives
        where columns are sorted by negative-label predictions descending,
        but here we use the simpler "as is" path: returns AUCs and (k,k) cov.
    label_1_count: m, the number of positive (label==1) samples
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:
        sx = sx.reshape(1, 1)
        sy = sy.reshape(1, 1)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, score_a, score_b):
    """Compute two-sided p-value for AUC(a) == AUC(b) on paired predictions.

    y_true: (N,) binary 0/1 array
    score_a, score_b: (N,) score arrays from two classifiers on the SAME samples

    Returns: dict with auc_a, auc_b, delta = auc_a - auc_b, var_delta, z, p
    """
    y_true = np.asarray(y_true).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    order = np.argsort(-y_true, kind='mergesort')  # positives first
    y_sorted = y_true[order]
    m = int(np.sum(y_sorted == 1))
    n = int(np.sum(y_sorted == 0))
    predictions = np.vstack([score_a[order], score_b[order]])
    aucs, cov = _fast_delong(predictions, m)

    var_delta = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = aucs[0] - aucs[1]
    if var_delta <= 0:
        z = float('inf') if delta != 0 else 0.0
        p = 0.0 if delta != 0 else 1.0
    else:
        z = delta / np.sqrt(var_delta)
        # two-sided
        from scipy.stats import norm
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return {
        'auc_a': float(aucs[0]),
        'auc_b': float(aucs[1]),
        'delta': float(delta),
        'var_delta': float(var_delta),
        'z': float(z),
        'p': float(p),
        'n_positive': m,
        'n_negative': n,
    }
