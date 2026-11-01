#!/usr/bin/env python3
"""PROMPT P2REV-05 Job 1 - one computation, two parts.

(1) Paired difference PCA-36(Tier 2B)+LDA minus Tier 3+LDA over the 22
    patients, seed 0, original labelling, oracle-calibrated and raw: mean,
    95 % bootstrap CI (2000 resamples over patients, rng 42), TOST at
    +/-0.05, Wilcoxon exact (Pratt), matched-pairs rank-biserial. Two rows
    appended to results/p2rev02/paired_tests.csv (comparison
    `pca36lda_vs_t3`).

(2) Per fold, the cosine between the Tier 3 LDA coefficient vector (LDA
    fitted on tangent features at the Frechet mean of ALL training
    covariances, as in manuscript section 3.4) and the reference direction r
    (unit mean of the training patients' ictal-minus-interictal shifts in
    the tangent space at the TRAINING-INTERICTAL Frechet mean, as defined
    there). Both are treated as vectors in R^36; the two tangent charts
    differ, which is stated in the results doc. Output:
    results/p2rev02/lda_direction_cosine.csv (fold, cosine) plus
    min/median/max.

DECISION RULE (pre-set here, before the computation ran): the word
"equivalent" is used for Tier 3 vs PCA-36 Tier 2B in the manuscript only if
the ORACLE-calibrated 95 % CI of the paired difference lies inside
(-0.05, +0.05). Otherwise every occurrence becomes "not distinguished
(difference 0.012 at d = 36, 95 % CI [lo, hi])".
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R02 = ROOT / 'results' / 'p2rev02'
MARGIN = 0.05


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paired_part():
    m02 = load_script('p2rev02_classical_rerun')
    pp = [r for r in csv.DictReader(open(R02 / 'per_patient_auc.csv',
                                         encoding='utf-8'))]

    def vals(meth, clf, d):
        return {r['patient']: float(r['auc']) for r in pp
                if (r['labelling'], r['seed'], r['method'], r['classifier'],
                    r['d']) == ('original', '0', meth, clf, str(d))
                and r['auc']}

    a = vals('tier2b_pca', 'lda', 36)
    b = vals('tier3', 'lda', 36)
    common = sorted(set(a) & set(b))
    assert len(common) == 22, len(common)
    rows, decision = [], {}
    for kind, f in [('calibrated', lambda x: max(x, 1 - x)),
                    ('raw', lambda x: x)]:
        x = np.array([f(a[p]) for p in common])
        y = np.array([f(b[p]) for p in common])
        d = x - y
        lo, hi = m02.boot_mean_ci(d)              # 2000 resamples, rng 42
        tp, mean, _ = m02.tost_paired(d, MARGIN)
        wp, wmethod = m02.wilcoxon_pratt(x, y)
        rb = m02.pratt_rank_biserial(x, y)
        rows.append(dict(comparison='pca36lda_vs_t3', labelling='original',
                         auc_kind=kind, n=22, mean_diff=round(mean, 4),
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         tost_p=round(tp, 5), wilcoxon_p=round(wp, 5),
                         wilcoxon_method=wmethod,
                         rank_biserial=round(rb, 3)))
        if kind == 'calibrated':
            decision = dict(
                rule='equivalent iff oracle CI inside (-0.05, +0.05)',
                ci=[round(lo, 4), round(hi, 4)],
                equivalent=bool(-MARGIN < lo and hi < MARGIN),
                wording=('equivalent' if (-MARGIN < lo and hi < MARGIN)
                         else f'not distinguished (difference '
                              f'{mean:.3f} at d = 36, 95 % CI '
                              f'[{lo:.3f}, {hi:.3f}])'))
    with open(R02 / 'paired_tests.csv', 'a', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writerows(rows)
    print('paired rows appended:')
    for r in rows:
        print(' ', r)
    print('DECISION:', json.dumps(decision))
    return decision


def cosine_part():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from pyriemann.tangentspace import TangentSpace
    m02 = load_script('p2rev02_classical_rerun')
    cache = m02.load_cache(sorted(
        d for d in Path(m02.DATA_ROOT).iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in m02.EXCLUDE_SUBJECTS))
    subjects = sorted(cache)
    sel = {s: m02.select_windows(cache[s], 'original', 0) for s in subjects}
    out = []
    for test in subjects:
        tr = [s for s in subjects if s != test]
        covs_tr = np.concatenate([cache[s]['covs'][sel[s][0]] for s in tr])
        y_tr = np.concatenate([sel[s][1] for s in tr])
        ts_all = TangentSpace(metric='riemann')
        feats = ts_all.fit_transform(covs_tr)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(feats, y_tr)
        w = lda.coef_[0]
        inter_tr = np.concatenate([
            cache[s]['covs'][sel[s][0]][sel[s][1] == 0] for s in tr])
        ts_int = TangentSpace(metric='riemann')
        ts_int.fit(inter_tr)
        shifts = []
        for s in tr:
            idx, y = sel[s]
            v = ts_int.transform(cache[s]['covs'][idx])
            shifts.append(v[y == 1].mean(0) - v[y == 0].mean(0))
        r = np.mean(shifts, axis=0)
        r = r / np.linalg.norm(r)
        cos = float(np.dot(w, r) / (np.linalg.norm(w) * np.linalg.norm(r)))
        out.append((test, cos))
        print(f'  fold {test}: cos = {cos:.4f}')
    with open(R02 / 'lda_direction_cosine.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['fold_held_out', 'cosine_lda_vs_r'])
        w.writerows([(p, round(c, 4)) for p, c in out])
    cs = np.array([c for _, c in out])
    summ = dict(min=round(float(cs.min()), 3),
                median=round(float(np.median(cs)), 3),
                max=round(float(cs.max()), 3))
    print('cosine summary:', summ)
    return summ


if __name__ == '__main__':
    decision = paired_part()
    summ = cosine_part()
    with open(R02 / 'p2rev05_job1.json', 'w', encoding='utf-8') as f:
        json.dump(dict(decision=decision, cosine=summ), f, indent=2)
