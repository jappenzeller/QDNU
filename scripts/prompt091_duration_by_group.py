#!/usr/bin/env python3
"""
================================================================================
PROMPT 091 — Seizure duration by entropy group
================================================================================

The zero-cost check before Notebook 03.

042 split the cohort by the sign of the ictal entropy change. Two readings of
the E/I field make different claims about what that split is:

  A (two regimes)        purifiers and mixers are different dynamics.
                         Duration should not matter.
  B (one regime,
     two clocks)         both groups follow the same rise-peak-fall entropy
                         curve; purifiers are the patients whose seizures last
                         long enough to get over the peak. Duration should
                         separate the groups.

This prompt tests only the duration claim.

ANNOTATION DATA ONLY. The script reads `chbXX-summary.txt` and nothing else.
No EEG is loaded, no covariance is formed, and NO ENTROPY IS COMPUTED here —
the entropy groups are taken as given from 042.

Outputs to results/prompt091/:
    durations_per_seizure.csv   patient, seizure_idx, file, start_s, end_s, duration_s
    per_patient.csv             patient, group, n, min_s, median_s, max_s, spread
    tests.json                  P1 p-value, threshold, flip list, flip rate
    duration_by_group.png
    SUMMARY.md

Usage:
    python scripts/prompt091_duration_by_group.py [--data-root PATH]

Author: Claude Code
Date: 2026-10-11
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / 'results' / 'prompt091'
PROMPT_042_CSV = PROJECT_ROOT / 'results' / 'prompt042' / 'per_patient.csv'
DEFAULT_DATA_ROOT = Path('H:/Data/PythonDNU/EEG/chbmit')

# =============================================================================
# PRE-DECLARED — written before the script was run. Both P1 and the flip rate
# are reported whatever they show. Nothing else is pre-declared.
# =============================================================================
#
# Groups (from PROMPT 042, sign of the ictal entropy change dS):
#     purifier (dS < 0, determined), n = 4 : chb08, chb13, chb14, chb21
#     mixer    (dS > 0, determined), n = 10: chb01, chb04, chb06, chb07, chb10,
#                                            chb16, chb17, chb18, chb19, chb23
#     undetermined,                  n = 8 : chb02, chb03, chb05, chb09, chb11,
#                                            chb15, chb20, chb22
#
# Per patient: all seizure durations in seconds from the summary files; the
# patient's median duration.
#
# P1 (B's prediction): purifier median durations exceed mixer median durations,
#     Mann-Whitney one-sided p < 0.05 over the 4 + 10 patients.
#
# Flip rate: threshold = the median of the 14 determined patients' median
#     durations. A purifier below it, or a mixer above it, is a *flip*.
#     Jim's stated expectation (2026-10-11): some non-zero flip percentage,
#     neither ~0 nor ~50 %. Report the flip rate and which patients flipped.
#
# =============================================================================

PURIFIERS = ['chb08', 'chb13', 'chb14', 'chb21']
MIXERS = ['chb01', 'chb04', 'chb06', 'chb07', 'chb10',
          'chb16', 'chb17', 'chb18', 'chb19', 'chb23']
UNDETERMINED = ['chb02', 'chb03', 'chb05', 'chb09', 'chb11',
                'chb15', 'chb20', 'chb22']

GROUP = ({p: 'purifier' for p in PURIFIERS} |
         {p: 'mixer' for p in MIXERS} |
         {p: 'undetermined' for p in UNDETERMINED})
PATIENTS = sorted(GROUP)                      # the 22 of 042
ALPHA = 0.05
SPREAD_FLAG = 3.0                             # max/min ratio called out in the report

COLOURS = {'purifier': '#B8720F', 'mixer': '#0E93AB', 'undetermined': '#9AA0A6'}


# =============================================================================
# SUMMARY-FILE PARSING  (annotation only — no EEG is opened)
# =============================================================================

def parse_seizures(patient: str, data_root: Path) -> list[dict]:
    """Return one row per annotated seizure from chbXX-summary.txt."""
    path = data_root / patient / f'{patient}-summary.txt'
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding='utf-8', errors='replace')
    rows: list[dict] = []
    idx = 0

    for block in re.split(r'\n(?=File Name:)', text):
        if 'File Name:' not in block:
            continue
        m_file = re.search(r'File Name:\s*(\S+)', block)
        fname = m_file.group(1) if m_file else ''

        m_n = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        if not m_n or int(m_n.group(1)) == 0:
            continue
        n_declared = int(m_n.group(1))

        starts = [int(x) for x in re.findall(
            r'Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds', block)]
        ends = [int(x) for x in re.findall(
            r'Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds', block)]
        if len(starts) != n_declared or len(ends) != n_declared:
            print(f'  [{patient}] {fname}: declared {n_declared}, '
                  f'parsed {len(starts)} start / {len(ends)} end times')

        for s, e in zip(starts, ends):
            dur = e - s
            if dur <= 0:
                print(f'  [{patient}] {fname}: non-positive duration {dur}s skipped')
                continue
            idx += 1
            rows.append(dict(patient=patient, seizure_idx=idx, file=fname,
                             start_s=s, end_s=e, duration_s=dur))
    return rows


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT,
                    help='CHB-MIT root holding chbXX/chbXX-summary.txt')
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- integrity: the groups must match 042 exactly -----------------------
    if PROMPT_042_CSV.exists():
        with open(PROMPT_042_CSV, newline='') as f:
            from_042 = {r['patient']: r['group'] for r in csv.DictReader(f)}
        assert from_042 == GROUP, 'group assignment does not match PROMPT 042'
        print(f'groups match {PROMPT_042_CSV.relative_to(PROJECT_ROOT)} (22 patients)')
    else:
        print('WARNING: results/prompt042/per_patient.csv not found; groups not cross-checked')

    # --- Step 1: per-seizure durations --------------------------------------
    print(f'\nReading summary files from {args.data_root} (annotation only)')
    all_rows: list[dict] = []
    for p in PATIENTS:
        rows = parse_seizures(p, args.data_root)
        all_rows.extend(rows)
        print(f'  {p}: {len(rows)} seizures')

    with open(OUT_DIR / 'durations_per_seizure.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['patient', 'seizure_idx', 'file',
                                          'start_s', 'end_s', 'duration_s'])
        w.writeheader()
        w.writerows(all_rows)

    # --- Step 2: per-patient summary ----------------------------------------
    per_patient: dict[str, dict] = {}
    for p in PATIENTS:
        d = [r['duration_s'] for r in all_rows if r['patient'] == p]
        assert d, f'{p}: no seizures parsed'
        per_patient[p] = dict(
            patient=p, group=GROUP[p], n=len(d),
            min_s=min(d), median_s=statistics.median(d), max_s=max(d),
            spread=max(d) / min(d),
        )

    with open(OUT_DIR / 'per_patient.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['patient', 'group', 'n',
                                          'min_s', 'median_s', 'max_s', 'spread'])
        w.writeheader()
        for p in PATIENTS:
            r = dict(per_patient[p])
            r['median_s'] = f"{r['median_s']:.1f}"
            r['spread'] = f"{r['spread']:.2f}"
            w.writerow(r)

    # --- Step 3: P1 ----------------------------------------------------------
    pur_med = [per_patient[p]['median_s'] for p in PURIFIERS]
    mix_med = [per_patient[p]['median_s'] for p in MIXERS]
    u, p1 = mannwhitneyu(pur_med, mix_med, alternative='greater')
    # rank-biserial from U, for effect size alongside the p-value
    rbc = 2 * u / (len(pur_med) * len(mix_med)) - 1
    p1_pass = bool(p1 < ALPHA)
    print(f'\nP1  purifier medians > mixer medians: U = {u:.1f}, '
          f'one-sided p = {p1:.4f} -> {"PASS" if p1_pass else "FAIL"}')
    print(f'    purifier medians {sorted(pur_med)}')
    print(f'    mixer medians    {sorted(mix_med)}')

    # --- Step 4: flip rate ---------------------------------------------------
    determined = PURIFIERS + MIXERS
    det_med = [per_patient[p]['median_s'] for p in determined]
    threshold = float(statistics.median(det_med))

    flips = []
    for p in determined:
        m = per_patient[p]['median_s']
        if GROUP[p] == 'purifier' and m < threshold:
            flips.append(dict(patient=p, group='purifier', median_s=m,
                              reason='purifier below threshold'))
        elif GROUP[p] == 'mixer' and m > threshold:
            flips.append(dict(patient=p, group='mixer', median_s=m,
                              reason='mixer above threshold'))
    flip_rate = len(flips) / len(determined)
    print(f'\nthreshold (median of the 14 determined medians) = {threshold:.1f} s')
    print(f'flips: {len(flips)}/{len(determined)} = {flip_rate:.1%}')
    for fl in flips:
        print(f'    {fl["patient"]:6s} {fl["group"]:9s} median {fl["median_s"]:6.1f} s')

    # --- Step 4b: what the flip metric can and cannot reach ------------------
    # DESCRIPTIVE, not pre-declared. With n_pur purifiers among 14 determined
    # patients and a median-split threshold, exactly 7 medians sit above it, so
    #     flips = (purifiers below) + (mixers above) = (n_pur - a) + (7 - a)
    # where a = purifiers above the threshold. The metric therefore has a floor
    # above zero; quoting 64 % without that floor would be meaningless.
    from math import comb
    n_pur, n_above = len(PURIFIERS), sum(1 for p in determined
                                         if per_patient[p]['median_s'] > threshold)
    a_obs = sum(1 for p in PURIFIERS if per_patient[p]['median_s'] > threshold)
    reachable = {a: (n_pur - a) + (n_above - a) for a in range(0, n_pur + 1)}
    # hypergeometric: a purifiers among the n_above long patients, by chance
    pmf = {a: comb(n_above, a) * comb(len(determined) - n_above, n_pur - a)
              / comb(len(determined), n_pur) for a in reachable}
    chance_flips = sum(pmf[a] * reachable[a] for a in reachable)
    p_tail = sum(pmf[a] for a in reachable if reachable[a] >= len(flips))
    print(f'    metric floor {min(reachable.values())}/{len(determined)} '
          f'({min(reachable.values())/len(determined):.1%}), '
          f'ceiling {max(reachable.values())}/{len(determined)}, '
          f'chance expectation {chance_flips:.1f} '
          f'({chance_flips/len(determined):.0%}); P(flips >= observed) = {p_tail:.3f}')

    # --- Step 5: within-patient spread --------------------------------------
    wide = [p for p in determined if per_patient[p]['spread'] > SPREAD_FLAG]
    print(f'\ndetermined patients with max/min > {SPREAD_FLAG:g}: '
          f'{len(wide)}/{len(determined)} -> {wide}')

    # --- tests.json ----------------------------------------------------------
    tests = dict(
        generated=datetime.now().isoformat(timespec='seconds'),
        data='CHB-MIT annotation files only (chbXX-summary.txt); no EEG read, no entropy computed',
        groups_from='results/prompt042/per_patient.csv',
        n_patients=len(PATIENTS), n_seizures=len(all_rows),
        P1=dict(
            statement='purifier median durations exceed mixer median durations',
            test='Mann-Whitney U, one-sided (greater)',
            n_purifier=len(pur_med), n_mixer=len(mix_med),
            U=float(u), p_value=float(p1), alpha=ALPHA, rank_biserial=float(rbc),
            purifier_medians_s=sorted(pur_med), mixer_medians_s=sorted(mix_med),
            group_median_of_medians_s=dict(
                purifier=float(statistics.median(pur_med)),
                mixer=float(statistics.median(mix_med))),
            pass_=p1_pass,
        ),
        flip=dict(
            threshold_s=threshold,
            threshold_definition='median of the 14 determined patients\' median durations',
            n_determined=len(determined), n_flips=len(flips),
            flip_rate=flip_rate, flips=flips,
        ),
        flip_context=dict(
            note='DESCRIPTIVE, not pre-declared: what the flip metric can reach '
                 'given 4 purifiers / 10 mixers and a median-split threshold',
            purifiers_above_threshold=a_obs,
            reachable_flip_counts=reachable,
            floor=min(reachable.values()), ceiling=max(reachable.values()),
            chance_expectation=float(chance_flips),
            p_flips_at_least_observed=float(p_tail),
        ),
        spread=dict(
            criterion=f'max/min > {SPREAD_FLAG:g}',
            n_determined_over=len(wide), patients=wide,
            spreads={p: round(per_patient[p]['spread'], 2) for p in determined},
        ),
    )
    with open(OUT_DIR / 'tests.json', 'w') as f:
        json.dump(tests, f, indent=1)

    # --- figure --------------------------------------------------------------
    plot_figure(per_patient, threshold)

    # --- SUMMARY.md ----------------------------------------------------------
    write_summary(per_patient, tests, all_rows)
    print(f'\nDone. Outputs in {OUT_DIR}')
    return 0


def plot_figure(per_patient: dict, threshold: float) -> None:
    order = ['purifier', 'mixer', 'undetermined']
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rng = np.random.default_rng(0)

    for i, g in enumerate(order):
        pats = [p for p in PATIENTS if GROUP[p] == g]
        y = [per_patient[p]['median_s'] for p in pats]
        x = i + rng.uniform(-0.12, 0.12, len(y))
        ax.scatter(x, y, s=55, color=COLOURS[g], label=f'{g} (n={len(pats)})',
                   zorder=3, edgecolor='white', linewidth=0.6)
        for xi, yi, p in zip(x, y, pats):
            ax.annotate(p.replace('chb', ''), (xi, yi), fontsize=7,
                        xytext=(6, 0), textcoords='offset points',
                        va='center', color='#4A535C')

    ax.axhline(threshold, color='#6E7780', linestyle='--', linewidth=1, zorder=2)
    ax.annotate(f'threshold {threshold:.0f} s', (2.42, threshold), fontsize=8,
                color='#6E7780', va='bottom', ha='right')

    ax.set_yscale('log')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel('median seizure duration (s, log scale)')
    ax.set_title('PROMPT 091 — median seizure duration by entropy group\n'
                 'CHB-MIT annotations; groups from PROMPT 042', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'duration_by_group.png', dpi=150)
    plt.close(fig)


def write_summary(per_patient: dict, tests: dict, all_rows: list) -> None:
    P1 = tests['P1']
    F = tests['flip']
    S = tests['spread']
    pur_m = P1['group_median_of_medians_s']['purifier']
    mix_m = P1['group_median_of_medians_s']['mixer']

    L = []
    a = L.append
    a('# PROMPT 091 - Seizure duration by entropy group')
    a('')
    a(f"**Generated:** {tests['generated'][:10]}")
    a('**Script:** `scripts/prompt091_duration_by_group.py`')
    a('**Depends on:** `results/prompt042/per_patient.csv` (the groups)')
    a('')
    a('Conventions: CHB-MIT **annotation files only** (`chbXX-summary.txt`), 22 patients '
      '(chb12 and chb24 excluded, as in 037-042). One row per annotated seizure; '
      'duration = `Seizure End Time` minus `Seizure Start Time`, in seconds. '
      'No EEG is read in this prompt, no covariance is formed and '
      '**no entropy is computed** - the purifier / mixer / undetermined split is '
      'taken as given from 042 (sign of the ictal entropy change '
      '$\\Delta S = \\operatorname{median}_{\\text{ictal}} S - '
      '\\operatorname{median}_{\\text{interictal}} S$, determined when '
      'Mann-Whitney $p < 0.05$ and bootstrap stability $\\geq 0.9$). '
      'Pre-declared: **P1** purifier median durations exceed mixer median durations '
      '(Mann-Whitney one-sided, $p < 0.05$); **flip rate** against a threshold equal to '
      'the median of the 14 determined patients\' median durations.')
    a('')
    a('---')
    a('')
    a('## The two readings being separated')
    a('')
    a('| reading | claim about the 042 split | duration prediction |')
    a('|---|---|---|')
    a('| A - two regimes | purifiers and mixers are different dynamics | duration should not matter |')
    a('| B - one regime, two clocks | one rise-peak-fall entropy curve; purifiers are the patients whose seizures run past the peak | duration should separate the groups |')
    a('')
    a(f"## Results ({tests['n_seizures']} seizures, {tests['n_patients']} patients)")
    a('')
    a('| patient | group | n | min (s) | median (s) | max (s) | max/min |')
    a('|---|---|---|---|---|---|---|')
    for g in ('purifier', 'mixer', 'undetermined'):
        for p in PATIENTS:
            if GROUP[p] != g:
                continue
            r = per_patient[p]
            a(f"| {p} | {g} | {r['n']} | {r['min_s']} | {r['median_s']:.1f} | "
              f"{r['max_s']} | {r['spread']:.2f} |")
    a('')
    a('### P1 - duration by group')
    a('')
    a('| | purifiers (n=4) | mixers (n=10) |')
    a('|---|---|---|')
    a(f"| median of per-patient medians | {pur_m:.1f} s | {mix_m:.1f} s |")
    a(f"| per-patient medians | {', '.join(f'{v:g}' for v in P1['purifier_medians_s'])} | "
      f"{', '.join(f'{v:g}' for v in P1['mixer_medians_s'])} |")
    a('')
    a(f"Mann-Whitney U = {P1['U']:.1f}, one-sided $p = {P1['p_value']:.4f}$, "
      f"rank-biserial $r = {P1['rank_biserial']:+.2f}$.")
    a('')
    a('### Flip rate')
    a('')
    a(f"Threshold = median of the 14 determined patients' medians = "
      f"**{F['threshold_s']:.1f} s**. "
      f"Flips = **{F['n_flips']}/{F['n_determined']} = {F['flip_rate']:.0%}**.")
    a('')
    if F['flips']:
        a('| patient | group | median (s) | why it is a flip |')
        a('|---|---|---|---|')
        for fl in F['flips']:
            a(f"| {fl['patient']} | {fl['group']} | {fl['median_s']:.1f} | {fl['reason']} |")
    else:
        a('No flips: every purifier sits above the threshold and every mixer below it.')
    a('')
    a('### Within-patient spread')
    a('')
    a(f"{S['n_determined_over']} of the {F['n_determined']} determined patients have "
      f"max/min duration {S['criterion'].replace('max/min ', '')}"
      f"{': ' + ', '.join(S['patients']) if S['patients'] else ''}.")
    a('')
    a('![median seizure duration by entropy group](duration_by_group.png)')
    a('')
    C = tests['flip_context']
    n_above = F['n_determined'] // 2
    spreads = tests['spread']['spreads']
    a('## Reading')
    a('')
    if P1['pass_']:
        a(f"Purifier seizures are the longer ones (one-sided $p = {P1['p_value']:.4f}$), "
          "which is what reading B predicts. The duration claim survives this check.")
    else:
        a(f"The two groups' durations are not distinguishable, and what ordering there is "
          f"runs the wrong way for B: the purifier median-of-medians is {pur_m:.1f} s "
          f"against the mixers' {mix_m:.1f} s, one-sided $p = {P1['p_value']:.4f}$ "
          f"(rank-biserial ${P1['rank_biserial']:+.2f}$). The distributions sit on top of "
          f"each other - purifier medians span "
          f"{P1['purifier_medians_s'][0]:g}-{P1['purifier_medians_s'][-1]:g} s, mixer "
          f"medians {P1['mixer_medians_s'][0]:g}-{P1['mixer_medians_s'][-1]:g} s - and "
          "only chb08 at 171 s is longer than every mixer. The other three purifiers are "
          "mid-pack or short: chb14 at 20 s is shorter than seven of the ten mixers.")
    a('')
    a(f"The flip rate needs its floor stated before it means anything. With "
      f"{len(PURIFIERS)} purifiers among {F['n_determined']} determined patients and a "
      f"median-split threshold, exactly {n_above} medians sit above the line, so the flip "
      f"count is $11 - 2a$ for $a$ purifiers above it and can only take the values "
      f"{', '.join(str(v) for v in sorted(set(C['reachable_flip_counts'].values())))}. "
      f"**It cannot reach zero.** Floor "
      f"{C['floor']}/{F['n_determined']} ({C['floor'] / F['n_determined']:.0%}, perfect-B "
      f"ordering); chance expectation {C['chance_expectation']:.0f}/{F['n_determined']} "
      f"({C['chance_expectation'] / F['n_determined']:.0%}); ceiling "
      f"{C['ceiling']}/{F['n_determined']}. The observed {F['n_flips']}/{F['n_determined']} "
      f"({F['flip_rate']:.0%}) is one step *worse* than chance rather than between zero and "
      f"chance, and is not a reversed effect either: "
      f"$P(\\text{{flips}} \\geq {F['n_flips']}) = {C['p_flips_at_least_observed']:.2f}$. "
      f"It is noise.")
    a('')
    a(f"The within-patient spread cuts the same way. B needs a patient's seizures to be "
      f"consistently long or consistently short for the 042 sign to come out stable, yet "
      f"{tests['spread']['n_determined_over']} of the {F['n_determined']} determined "
      f"patients vary by more than {SPREAD_FLAG:g}x within themselves ("
      f"{', '.join(f'{q} {spreads[q]:.1f}x' for q in tests['spread']['patients'])}), "
      f"including two of the four purifiers - chb21 ranges 12-81 s and chb13 17-70 s. "
      f"Under B those patients should straddle the entropy peak and give an unstable sign; "
      f"042 found their signs stable at bootstrap 0.9 or better.")
    a('')
    a('## Verdict')
    a('')
    a(f"**P1 {'PASS' if P1['pass_'] else 'FAIL'}** (one-sided $p = {P1['p_value']:.4f}$, "
      f"$\\alpha = {P1['alpha']}$"
      f"{'' if P1['pass_'] else '; the point estimate runs the other way'}). Flip rate "
      f"{F['flip_rate']:.0%}, against a metric floor of "
      f"{C['floor'] / F['n_determined']:.0%} and a chance expectation of "
      f"{C['chance_expectation'] / F['n_determined']:.0%}.")
    a('')
    a('## Consequences')
    a('')
    a("Reading B - one regime, two clocks, purifiers being the patients whose seizures run "
      "past the entropy peak - makes a prediction this cohort does not show. Duration does "
      "not separate the groups, the ordering that exists is the wrong way round, and "
      "patients whose own seizures vary several-fold in length still carry a stable entropy "
      "sign. That is three strikes against the simplest form of B. It is not a proof of A: "
      "median duration is a crude stand-in for time spent past the peak, a seizure's "
      "annotated end is not the end of its entropy excursion, and 14 patients is a small "
      "cohort. But B no longer gets to be the default reading, and nothing here requires A "
      "either.")
    a('')
    a("For Notebook 03 the consequence is concrete: the within-seizure curve has to be "
      "measured, not inferred from how long seizures last. The design does not change - "
      "time-resolved $S(\\rho)$ from onset, per seizure, on the 037 per-seizure structure - "
      "but the question it asks does. It is no longer \u201cdo purifiers simply run longer\u201d; "
      "it is whether a rise-peak-fall shape exists in any patient at all, and if it does, "
      "whether where a patient stops on that curve tracks the 042 sign. If the curves come "
      "back with no common shape, A and B are both wrong and the split is something else. "
      "This check cost nothing and removed one of the two readings as a default; that is "
      "what it was for.")
    a('')
    a('*Annotation data only: this prompt read CHB-MIT summary files, computed no entropy '
      'and opened no EEG.*')
    with open(OUT_DIR / 'SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    sys.exit(main())
