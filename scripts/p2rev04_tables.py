#!/usr/bin/env python3
"""PROMPT P2REV-04 Job 2 - generate Tables 1-5 and S1 for the revised
manuscript from the P2REV-02/-03 result CSVs. No hand-typed values.

Outputs: PolarityPaper/revision/tables/table{1..5,S1}.{md,csv}

Formatting: three decimals for AUC, two for cosines, three for p with
"< 0.001" where applicable. Every caption states the labelling.
Rule (stated in revised section 3.2): a patient with fewer than 5 class-1
windows under a labelling is reported but excluded from "decided" counts
(strict labelling: chb16 alone).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R02 = ROOT / 'results' / 'p2rev02'
R03 = ROOT / 'results' / 'p2rev03'
OUT = ROOT / 'PolarityPaper' / 'revision' / 'tables'
HW = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']
MIN_CLASS1_DECIDED = 5


def a3(x):
    return f"{float(x):.3f}"


def p3(x):
    x = float(x)
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def c2(x):
    return f"{float(x):.2f}"


def rd(path):
    with open(path, encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if not r[list(r)[0]].startswith('#')]


def wtable(name, caption, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f'{name}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(OUT / f'{name}.md', 'w', encoding='utf-8') as f:
        f.write(f"**{caption}**\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"{name}: {len(rows)} rows")


def main():
    pp = rd(R02 / 'per_patient_auc.csv')
    overall = rd(R02 / 'overall_auc.csv')
    paired = rd(R02 / 'paired_tests.csv')
    md = rd(R02 / 'matched_dimension.csv')
    geo = {(r['patient'], r['labelling']): r
           for r in rd(R02 / 'geometric_sign.csv')}
    cohort = {r['patient']: r for r in rd(R02 / 'cohort_table.csv')
              if r['patient'] != 'TOTAL'}
    q20 = rd(R03 / 'per_patient_20s.csv')
    shot = {r['patient']: r for r in rd(R03 / 'shot_distribution_20s.csv')}
    sign = {r['patient']: r for r in rd(R03 / 'sign_table_20s.csv')}
    hwjson = json.load(open(ROOT / 'results/window_analysis/'
                            'quantum_20s_hardware.json',
                            encoding='utf-8'))['per_subject']

    def base(lab, meth, clf, seed='0'):
        return {r['patient']: r for r in pp
                if (r['labelling'], r['seed'], r['method'],
                    r['classifier']) == (lab, seed, meth, clf)}

    def n_class1(p, lab):
        key = 'n_class1_strict' if lab == 'strict' else 'n_class1_original'
        return int(cohort[p][key])

    def inv_counts(lab, meth, clf):
        d = base(lab, meth, clf)
        inv = sum(1 for r in d.values() if float(r['auc']) < 0.5)
        dec = sum(1 for p, r in d.items()
                  if r['ci_hi'] and float(r['ci_hi']) < 0.5
                  and n_class1(p, lab) >= MIN_CLASS1_DECIDED)
        return inv, dec

    # ---- Table 1 -----------------------------------------------------------
    ov = {(r['labelling'], r['method']): r for r in overall}
    rows1 = []
    for lab in ['strict', 'original']:
        for meth, clf, label in [('tier3', 'lda', 'Tier 3 (36, LDA)'),
                                 ('tier2b', 'xgb', 'Tier 2B (336, XGB)'),
                                 ('tier2a', 'xgb', 'Tier 2A (8, XGB)')]:
            r_ = ov[(lab, meth + '_raw')]
            c_ = ov[(lab, meth + '_calibrated')]
            inv, dec = inv_counts(lab, meth, clf)
            rows1.append([lab, label,
                          f"{a3(r_['mean_over_seeds'])} ± "
                          f"{a3(r_['sd_over_seeds'])}",
                          f"{a3(c_['mean_over_seeds'])} ± "
                          f"{a3(c_['sd_over_seeds'])}",
                          f"{inv} ({dec})"])
    wtable('table1',
           "Table 1. Mean per-patient AUC, 22 CHB-MIT patients, 20 s "
           "windows, LOSO; mean ± SD over 10 interictal seeds; "
           "oracle-calibrated = max(AUC, 1 − AUC) per patient. Inverted = "
           "patients with raw AUC < 0.5 at seed 0 (decided inversions, 95 % "
           "bootstrap interval entirely below 0.5, in parentheses; patients "
           "with fewer than 5 class-1 windows excluded from decided "
           "counts). Strict labelling: class 1 = annotated seizure; "
           "original (preictal-inclusive): seizure plus the preceding "
           "60 s.",
           ['labelling', 'method', 'raw AUC', 'oracle-calibrated AUC',
            'inverted (decided)'], rows1)

    # ---- Table 2 -----------------------------------------------------------
    rows2 = [[r['comparison'].replace('_vs_', ' vs ').replace('t', 'Tier ')
              .replace('Tier 3', 'Tier 3').replace('Tier 2a', 'Tier 2A')
              .replace('Tier 2b', 'Tier 2B'),
              r['labelling'], r['auc_kind'],
              a3(r['mean_diff']),
              f"[{a3(r['ci_lo'])}, {a3(r['ci_hi'])}]",
              p3(r['wilcoxon_p']) + f" ({r['wilcoxon_method']})",
              f"{float(r['rank_biserial']):.2f}",
              p3(r['tost_p'])]
             for r in paired]
    wtable('table2',
           "Table 2. Paired per-patient AUC comparisons (22 patients, "
           "seed 0, both labellings). Mean difference with 95 % bootstrap "
           "CI (2000 resamples over patients); Wilcoxon signed-rank "
           "(Pratt tie handling; method stated); matched-pairs "
           "rank-biserial correlation; TOST p at margin ± 0.05.",
           ['comparison', 'labelling', 'AUC kind', 'mean diff', '95 % CI',
            'Wilcoxon p', 'rank-biserial', 'TOST p'], rows2)

    # ---- Table 3 -----------------------------------------------------------
    rows3 = [[r['classifier'].upper(), r['d'],
              f"{a3(r['auc_mean'])} ± {a3(r['auc_sd'])}",
              r['n_inversions']] for r in md]
    wtable('table3a',
           "Table 3a. PCA-reduced Tier 2B features at dimension d, "
           "oracle-calibrated AUC (mean ± SD over 10 seeds) and inversion "
           "count, under both classifiers. Original labelling (the matched-"
           "dimension runs exist under the original labelling only).",
           ['classifier', 'd', 'oracle AUC', 'inversions'], rows3)

    def seed0_mean(meth, clf, d, kind='oracle'):
        vals = [float(r['auc']) for r in pp
                if (r['labelling'], r['seed'], r['method'], r['classifier'],
                    r['d']) == ('original', '0', meth, clf, str(d))
                and r['auc']]
        if kind == 'oracle':
            vals = [max(v, 1 - v) for v in vals]
        return a3(sum(vals) / len(vals)) if vals else 'n/a'

    rows3b = [
        ['Tier 2B, PCA to 36', seed0_mean('tier2b_pca', 'lda', 36),
         seed0_mean('tier2b_pca', 'xgb', 36)],
        ['Tier 3 tangent (36)', seed0_mean('tier3', 'lda', 36),
         seed0_mean('tier3_tangent', 'xgb', 36)],
        ['Tier 2A eigenvalues (8)', seed0_mean('tier2a', 'lda', 8),
         seed0_mean('tier2a', 'xgb', 8)],
    ]
    wtable('table3b',
           "Table 3b. Representation × classifier at matched dimension "
           "(d = 36 where applicable; Tier 2A is 8-dimensional). "
           "Oracle-calibrated mean per-patient AUC, seed 0. Original "
           "labelling only (these runs exist under the original labelling "
           "only).",
           ['representation', 'LDA', 'XGBoost'], rows3b)

    # ---- Table 4 -----------------------------------------------------------
    t3 = {lab: base(lab, 'tier3', 'lda') for lab in ['strict', 'original']}
    rows4 = []
    for p in sorted(cohort):
        cells = [p + (' †' if n_class1(p, 'strict') <
                      MIN_CLASS1_DECIDED else '') +
                 (' *' if p in HW else ''),
                 f"{n_class1(p, 'strict')} / {n_class1(p, 'original')}"]
        for lab in ['strict', 'original']:
            r = t3[lab].get(p)
            if r is None or not r['auc']:
                cells += ['n/a', 'n/a']
                continue
            cells += [f"{a3(r['auc'])} [{a3(r['ci_lo'])}, "
                      f"{a3(r['ci_hi'])}]", p3(r['perm_p'])]
        for lab in ['strict', 'original']:
            r = t3[lab].get(p)
            cells.append('+' if r and float(r['auc']) >= 0.5 else
                         ('−' if r else 'n/a'))
        for lab in ['strict', 'original']:
            g = geo.get((p, lab), {})
            cells.append({'1': '+', '-1': '−'}.get(g.get('pi'), 'n/a'))
        for lab in ['strict', 'original']:
            g = geo.get((p, lab), {})
            cells.append(c2(g['cosine']) if g.get('cosine') else 'n/a')
        rows4.append(cells)
    wtable('table4',
           "Table 4. Per-patient Tier 3 (tangent-space LDA, zero-shot, "
           "seed 0) and the geometric sign, under both labellings. "
           "AUC with 95 % stratified bootstrap interval; permutation p for "
           "below-chance (1000 permutations). σ_p = sign(AUC − 0.5); "
           "π_p and cos θ_p from the classifier-free geometric definition. "
           "† single-digit class-1 windows under strict labelling (chb16: "
           "one window; excluded from decided counts). * hardware-subset "
           "patient.",
           ['patient', 'n class-1 (strict / original)',
            'strict AUC [95 % CI]', 'strict perm p',
            'original AUC [95 % CI]', 'original perm p',
            'σ strict', 'σ original', 'π strict', 'π original',
            'cos θ strict', 'cos θ original'], rows4)

    # ---- Table 5 -----------------------------------------------------------
    q = {(r['row'], r['patient']): r for r in q20}
    rows5 = []
    for p in HW:
        h, s_ = q[('hw', p)], q[('sim', p)]
        sh, sg = shot[p], sign[p]
        rows5.append([
            p, int(hwjson[p]['n_samples']),
            f"{a3(h['auc'])} [{a3(h['ci_lo'])}, {a3(h['ci_hi'])}]",
            p3(h['perm_p_below']),
            f"{a3(s_['auc'])} [{a3(s_['ci_lo'])}, {a3(s_['ci_hi'])}]",
            p3(s_['perm_p_below']),
            f"[{a3(sh['p2_5'])}, {a3(sh['p97_5'])}]",
            f"{float(sh['hardware_percentile']):.0f}",
            '+' if int(sg['tier3_sign']) > 0 else '−',
            '+' if int(sg['pi']) > 0 else '−',
            '+' if int(sg['z0_sign_supplementary']) > 0 else '−'])
    po = shot['OVERALL(pooled)']
    rows5.append(['pooled', 144,
                  a3(po['hardware_value']), '—', a3(po['sim_exact']), '—',
                  f"[{a3(po['p2_5'])}, {a3(po['p97_5'])}]",
                  f"{float(po['hardware_percentile']):.0f}", '—', '—', '—'])
    wtable('table5',
           "Table 5. The quantum rows: seven hardware-subset patients, "
           "20 s windows, original labelling (the strict labelling cannot "
           "be applied; the windows were fixed when the hardware jobs "
           "ran). Hardware (ibm_torino, 1024 shots) and exact simulation "
           "with 95 % bootstrap intervals and permutation p; the 95 % "
           "shot-noise band of the simulated template classifier (200 "
           "seeds) and the percentile of the hardware value in it; signs: "
           "Tier 3 σ_p, geometric π_p, and (supplementary) the ancilla "
           "⟨Z₀⟩ observable described in the submitted §3.5, which "
           "produced none of the reported numbers.",
           ['patient', 'n windows', 'hardware AUC [95 % CI]', 'hw perm p',
            'simulation AUC [95 % CI]', 'sim perm p', 'shot band',
            'hw percentile', 'σ (Tier 3)', 'π', '⟨Z₀⟩ sign'], rows5)

    # ---- Table S1 ----------------------------------------------------------
    rowsS1 = [[p, c['n_seizures'], f"{float(c['total_ictal_s']):.0f}",
               c['n_ictal_windows_20s'], c['n_preictal_windows_20s'],
               c['n_class1_original'], c['n_class1_strict'],
               c['n_interictal_used'],
               'yes' if c['in_hw_subset'] == 'True' else '']
              for p, c in sorted(cohort.items())]
    wtable('tableS1',
           "Table S1. Cohort: per-patient seizures, total ictal duration, "
           "and 20 s window counts per class under both labellings. "
           "142 seizures in total; the seven hardware-subset patients "
           "contributed 37.",
           ['patient', 'seizures', 'total ictal (s)', 'ictal windows',
            'preictal windows', 'class-1 (original)', 'class-1 (strict)',
            'interictal used', 'hardware subset'], rowsS1)


if __name__ == '__main__':
    main()
