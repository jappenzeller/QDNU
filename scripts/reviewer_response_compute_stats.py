#!/usr/bin/env python3
"""
REV-001 Task 3: compute DeLong p, permutation p, bootstrap 95% CIs, and
run variance from the dumped per-segment predictions.

Reads:  results/reviewer_response/predictions.npz
Writes: results/reviewer_response/stats.json
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from sklearn.metrics import roc_auc_score
from reviewer_response_delong import delong_roc_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger('rev001-stats')

OUT_DIR = ROOT / 'results' / 'reviewer_response'
PRED_FILE = OUT_DIR / 'predictions.npz'

RNG_SEED = 1729  # for permutation/bootstrap reproducibility
N_PERMS = 10_000
N_BOOTS = 10_000


def permutation_p(y, scores, n_perms=N_PERMS, rng=None):
    """Two-sided permutation test of AUC vs chance via label shuffles.

    p = (1 + #{AUC_perm >= AUC_obs}) / (n_perms + 1)
    Per REV-001 spec, this is a one-sided test against the null AUC=0.5
    in the direction of the observed AUC.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    auc_obs = roc_auc_score(y, scores)
    count = 0
    y_perm = y.copy()
    for _ in range(n_perms):
        rng.shuffle(y_perm)
        # If AUC < 0.5, test the "AUC <= AUC_obs" tail (direction of obs)
        if auc_obs >= 0.5:
            auc_p = roc_auc_score(y_perm, scores)
            if auc_p >= auc_obs:
                count += 1
        else:
            auc_p = roc_auc_score(y_perm, scores)
            if auc_p <= auc_obs:
                count += 1
    p = (1 + count) / (n_perms + 1)
    return float(auc_obs), float(p)


def bootstrap_ci(y, scores, n_boots=N_BOOTS, rng=None, alpha=0.05):
    """Stratified percentile bootstrap of AUC."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    aucs = np.empty(n_boots, dtype=float)
    for b in range(n_boots):
        sp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sp, sn])
        try:
            aucs[b] = roc_auc_score(y[idx], scores[idx])
        except ValueError:
            aucs[b] = 0.5
    lo = float(np.quantile(aucs, alpha / 2.0))
    hi = float(np.quantile(aucs, 1.0 - alpha / 2.0))
    return lo, hi, float(np.mean(aucs)), float(np.std(aucs))


def main():
    log.info(f'Loading {PRED_FILE}')
    d = np.load(PRED_FILE, allow_pickle=True)

    y8 = d['seg_label']
    q_v1 = d['q_v1']
    q_v1_used = d['q_v1_used']
    q_v2 = d['q_v2']
    q_v2_used = d['q_v2_used']
    q_v3 = d['q_v3']
    q_v3_used = d['q_v3_used']
    cls8 = d['cls_8ch']
    cls8_used = d['cls_8ch_used']
    cls8_bag_aucs = d['cls_8ch_bag_aucs']

    y18 = d['seg_18_label']
    cls18 = d['cls_18ch']
    cls18_used = d['cls_18ch_used']
    cls18_bag_aucs = d['cls_18ch_bag_aucs']

    rng = np.random.default_rng(RNG_SEED)

    out = {'rng_seed': RNG_SEED, 'n_perms': N_PERMS, 'n_boots': N_BOOTS}

    # --- DeLong V2 vs classical 8ch (paired on the same 504-seg cache) ---
    paired = q_v2_used & cls8_used
    log.info(f'DeLong V2 vs cls8: n_paired = {paired.sum()}')
    delong = delong_roc_test(y8[paired], q_v2[paired], cls8[paired])
    out['delong_V2_vs_cls8'] = delong
    log.info(f"  AUC_V2={delong['auc_a']:.4f}, AUC_cls8={delong['auc_b']:.4f}, "
             f"delta={delong['delta']:+.4f}, p={delong['p']:.4g}")

    # --- Permutation p vs chance for V1, V2, V3 ---
    perm = {}
    for name, scores, used in [('V1', q_v1, q_v1_used), ('V2', q_v2, q_v2_used), ('V3', q_v3, q_v3_used)]:
        auc, p = permutation_p(y8[used], scores[used], n_perms=N_PERMS,
                               rng=np.random.default_rng(RNG_SEED + hash(name) % 1000))
        perm[name] = {'auc': auc, 'p': p, 'n': int(used.sum())}
        log.info(f'  Permutation {name}: AUC={auc:.4f}, p={p:.4g}')
    out['permutation_vs_chance'] = perm

    # --- Bootstrap 95% CI for V1, V2, V3, cls8, cls18 ---
    ci = {}
    for name, scores, used, y_ in [
        ('V1', q_v1, q_v1_used, y8),
        ('V2', q_v2, q_v2_used, y8),
        ('V3', q_v3, q_v3_used, y8),
        ('cls8', cls8, cls8_used, y8),
        ('cls18', cls18, cls18_used, y18),
    ]:
        lo, hi, mean, sd = bootstrap_ci(y_[used], scores[used], n_boots=N_BOOTS,
                                        rng=np.random.default_rng(RNG_SEED + hash(name + '_ci') % 1000))
        ci[name] = {'lo': lo, 'hi': hi, 'mean': mean, 'sd': sd,
                    'point': float(roc_auc_score(y_[used], scores[used]))}
        log.info(f'  95% CI {name}: [{lo:.4f}, {hi:.4f}], point AUC={ci[name]["point"]:.4f}')
    out['bootstrap_ci'] = ci

    # --- Run variance ---
    bag_aucs = [float(x) for x in cls8_bag_aucs]
    bag_sd = float(np.std(bag_aucs, ddof=1))
    log.info(f'  Classical 8ch per-bag AUCs: {bag_aucs}, SD = {bag_sd:.4f}')
    out['run_variance'] = {
        'cls8_per_bag_auc': bag_aucs,
        'cls8_bag_sd': bag_sd,
        'cls18_per_bag_auc': [float(x) for x in cls18_bag_aucs],
        'cls18_bag_sd': float(np.std(cls18_bag_aucs, ddof=1)),
        'quantum_run_sd': 0.0,
        'quantum_method': 'statevector simulation; deterministic dual-template; '
                          'shot noise N/A (no measurement sampling).',
    }

    with open(OUT_DIR / 'stats.json', 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f'Wrote {OUT_DIR / "stats.json"}')


if __name__ == '__main__':
    main()
