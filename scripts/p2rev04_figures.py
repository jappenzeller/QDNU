#!/usr/bin/env python3
"""PROMPT P2REV-04 Job 3 - regenerate the four manuscript figures from the
result CSVs with the manuscript's labels and the labelling in the title,
plus the R1-5 schematic. Outputs: PolarityPaper/revision/figures/.

Palette: Okabe-Ito subset (#0072B2 blue, #D55E00 vermillion, #009E73 green),
validated CVD-safe. Fig. 3's x axis is the signed cosine pi_p * cos(theta_p)
(the raw inner product <s_p, r> was not retained by the geometry run; the
sign and the normalised magnitude were).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R02 = ROOT / 'results' / 'p2rev02'
R03 = ROOT / 'results' / 'p2rev03'
OUT = ROOT / 'PolarityPaper' / 'revision' / 'figures'
HW = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']
COL = {'tier2a': '#0072B2', 'tier2b': '#D55E00', 'tier3': '#009E73',
       'lda': '#0072B2', 'xgb': '#D55E00', 'grey': '#666666'}


def rd(path):
    with open(path, encoding='utf-8') as f:
        return [r for r in csv.DictReader(f)
                if not r[list(r)[0]].startswith('#')]


def fig1_per_patient_ci():
    pp = rd(R02 / 'per_patient_auc.csv')
    pats = sorted({r['patient'] for r in pp})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, lab, title in [(axes[0], 'strict',
                            'Strict labelling (class 1 = annotated '
                            'seizure)'),
                           (axes[1], 'original',
                            'Original labelling (seizure + preceding '
                            '60 s)')]:
        for meth, clf, off, name in [
                ('tier2a', 'xgb', -0.25, 'Tier 2A (8, XGB)'),
                ('tier2b', 'xgb', 0.0, 'Tier 2B (336, XGB)'),
                ('tier3', 'lda', 0.25, 'Tier 3 (36, LDA)')]:
            d = {r['patient']: r for r in pp
                 if (r['labelling'], r['seed'], r['method'],
                     r['classifier']) == (lab, '0', meth, clf)}
            xs, ys, lo, hi = [], [], [], []
            for i, p in enumerate(pats):
                r = d.get(p)
                if not r or not r['auc'] or not r['ci_lo']:
                    continue
                xs.append(i + off)
                a = float(r['auc'])
                ys.append(a)
                lo.append(a - float(r['ci_lo']))
                hi.append(float(r['ci_hi']) - a)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt='o', ms=4, lw=1.2,
                        capsize=2, color=COL[meth], label=name)
        ax.axhline(0.5, color=COL['grey'], lw=1, ls='--')
        ax.set_ylabel('raw AUC (95 % bootstrap CI)')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5, axis='y')
    axes[0].legend(frameon=False, ncol=3, loc='lower right', fontsize=9)
    axes[1].set_xticks(range(len(pats)))
    axes[1].set_xticklabels(pats, rotation=90, fontsize=8)
    fig.suptitle('Per-patient zero-shot AUC, 22 patients, 20 s windows, '
                 'seed 0', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'per_patient_ci.png', dpi=150)
    plt.close(fig)


def fig2_matched_dimension():
    md = rd(R02 / 'matched_dimension.csv')
    D = [8, 24, 36, 72, 144, 336]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for clf in ['lda', 'xgb']:
        ys = [float(r['auc_mean']) for d in D for r in md
              if r['classifier'] == clf and int(r['d']) == d]
        sd = [float(r['auc_sd']) for d in D for r in md
              if r['classifier'] == clf and int(r['d']) == d]
        ax.errorbar(D, ys, yerr=sd, marker='o', ms=5, lw=2, color=COL[clf],
                    label=f'PCA(Tier 2B) + {clf.upper()}')
        ax.annotate(f'{ys[-1]:.3f}', (D[-1], ys[-1]),
                    textcoords='offset points', xytext=(6, 4), fontsize=9,
                    color=COL[clf])
    ax.set_xscale('log')
    ax.set_xticks(D)
    ax.set_xticklabels(D)
    ax.set_xlabel('PCA dimension $d$ (log scale)')
    ax.set_ylabel('oracle-calibrated AUC (mean ± SD over seeds)')
    ax.set_title('Tier 2B features reduced to dimension $d$ '
                 '(original labelling)')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'matched_dimension_curve.png', dpi=150)
    plt.close(fig)


def fig3_geometric_sign():
    geo = {r['patient']: r for r in rd(R02 / 'geometric_sign.csv')
           if r['labelling'] == 'original'}
    pp = {r['patient']: r for r in rd(R02 / 'per_patient_auc.csv')
          if (r['labelling'], r['seed'], r['method']) ==
          ('original', '0', 'tier3')}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0.5, color=COL['grey'], lw=1, ls='--')
    ax.axvline(0.0, color=COL['grey'], lw=1, ls='--')
    for p, g in geo.items():
        if not g['pi'] or p not in pp or not pp[p]['auc']:
            continue
        x = int(g['pi']) * float(g['cosine'])
        y = float(pp[p]['auc'])
        decided = (pp[p]['ci_lo'] and (float(pp[p]['ci_lo']) > 0.5 or
                                       float(pp[p]['ci_hi']) < 0.5))
        ax.scatter(x, y, s=60, zorder=3,
                   facecolor=COL['tier3'] if decided else 'white',
                   edgecolor=COL['tier3'], linewidth=1.6)
        if p in ('chb14', 'chb21'):
            ax.annotate(p, (x, y), textcoords='offset points',
                        xytext=(6, 5), fontsize=9, color='#333333')
    ax.scatter([], [], s=60, facecolor=COL['tier3'],
               edgecolor=COL['tier3'], label='decided (CI excludes 0.5)')
    ax.scatter([], [], s=60, facecolor='white', edgecolor=COL['tier3'],
               label='undecided')
    ax.set_xlabel(r'signed geometric projection  '
                  r'$\pi_p\,\cos\theta_p$')
    ax.set_ylabel('Tier 3 raw AUC (zero-shot, seed 0)')
    ax.set_title('Geometric sign against zero-shot performance, '
                 '22 patients (original labelling)')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'geometric_sign_vs_auc.png', dpi=150)
    plt.close(fig)


def fig4_shot_interval():
    shot = {r['patient']: r for r in rd(R03 / 'shot_distribution_20s.csv')}
    pp = {(r['row'], r['patient']): r
          for r in rd(R03 / 'per_patient_20s.csv')}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, p in enumerate(HW):
        r = shot[p]
        ax.plot([i, i], [float(r['p2_5']), float(r['p97_5'])],
                color=COL['lda'], lw=6, alpha=0.4, solid_capstyle='round',
                label='simulated classifier, 1024-shot 95 % band'
                if i == 0 else None)
        ax.plot(i, float(r['sim_exact']), 'o', color=COL['lda'], ms=6,
                label='simulation (exact)' if i == 0 else None)
        h = pp[('hw', p)]
        ax.errorbar(i + 0.12, float(h['auc']),
                    yerr=[[float(h['auc']) - float(h['ci_lo'])],
                          [float(h['ci_hi']) - float(h['auc'])]],
                    fmt='D', color=COL['xgb'], ms=6, lw=1.4, capsize=3,
                    label='hardware (95 % bootstrap CI)'
                    if i == 0 else None)
    ax.axhline(0.5, color=COL['grey'], lw=1, ls='--')
    ax.set_xticks(range(7))
    ax.set_xticklabels(HW, rotation=45, fontsize=9)
    ax.set_ylabel('per-patient raw AUC')
    ax.set_title('Quantum template classifier at 20 s: simulation, its '
                 'shot-noise band, and hardware (original labelling)')
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'shot_interval_vs_hardware_20s.png', dpi=150)
    plt.close(fig)


def fig5_schematic():
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.set_aspect('equal')
    r = np.array([1.0, 0.35])
    r = r / np.linalg.norm(r)
    s1 = np.array([0.95, 0.75])
    s2 = np.array([-0.55, 0.55])
    ax.annotate('', xy=1.15 * r, xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', lw=2.5,
                                color=COL['grey']))
    ax.text(*(1.22 * r), r'$\mathbf{r}$  (cohort mean shift direction)',
            fontsize=11, color='#333333', va='center')
    ax.annotate('', xy=s1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', lw=2.2,
                                color=COL['tier3']))
    ax.text(*(s1 + [0.03, 0.05]),
            r'$\bar{\mathbf{s}}_p$:  $\langle\bar{\mathbf{s}}_p,'
            r'\mathbf{r}\rangle > 0$' + '\n' + r'($\pi_p = +1$)',
            fontsize=10, color=COL['tier3'])
    ax.annotate('', xy=s2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', lw=2.2,
                                color=COL['xgb']))
    ax.text(*(s2 + [-0.05, 0.06]),
            r'$\bar{\mathbf{s}}_q$:  $\langle\bar{\mathbf{s}}_q,'
            r'\mathbf{r}\rangle < 0$' + '\n' + r'($\pi_q = -1$)',
            fontsize=10, color=COL['xgb'], ha='right')
    # angle arc between r and s1
    th_r = np.arctan2(r[1], r[0])
    th_s = np.arctan2(s1[1], s1[0])
    arc = np.linspace(th_r, th_s, 30)
    ax.plot(0.45 * np.cos(arc), 0.45 * np.sin(arc), color='#333333', lw=1)
    mid = (th_r + th_s) / 2
    ax.text(0.56 * np.cos(mid), 0.56 * np.sin(mid), r'$\theta_p$',
            fontsize=11)
    # dashed line perpendicular to r through origin (sign boundary)
    perp = np.array([-r[1], r[0]])
    ax.plot([-1.3 * perp[0], 1.3 * perp[0]],
            [-1.3 * perp[1], 1.3 * perp[1]], ls='--', lw=1,
            color=COL['grey'])
    ax.text(-1.28 * perp[0], -1.34 * perp[1],
            r'$\langle\cdot,\mathbf{r}\rangle = 0$', fontsize=9,
            color=COL['grey'], ha='center')
    ax.plot(0, 0, 'o', ms=6, color='#333333')
    ax.text(0.03, -0.12, 'interictal Fréchet mean\n(training cohort)',
            fontsize=9, color='#333333')
    ax.set_xlim(-1.45, 1.75)
    ax.set_ylim(-0.75, 1.25)
    ax.axis('off')
    ax.set_title('The geometric sign in the tangent plane (schematic; '
                 'no data)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'schematic_tangent_sign.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_per_patient_ci()
    fig2_matched_dimension()
    fig3_geometric_sign()
    fig4_shot_interval()
    fig5_schematic()
    print('figures written to', OUT)
