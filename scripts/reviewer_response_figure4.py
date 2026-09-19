#!/usr/bin/env python3
"""
Regenerate Figure 4 (transpiled CZ scaling on ibm_torino) from the extended
scaling CSV. Writes springer_submission/figures/ibm_scaling.png.

Shows:
  - 8 measured points M in {2,4,6,8,12,16,24,32}
  - linear law 14.1 M - 17.5 (M <= 8 only, dashed)
  - quadratic fit 0.22 M^2 + 12.0 M (full range, solid)
  - logical operation count 15 M - 1 (dotted; the O(M) floor)
  - classical pairwise reference M(M-1)/2 (gray, dashed-dot; O(M^2))

No LaTeX dependency (matplotlib mathtext only); underscores in backend names
are written as plain text so they render literally.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / 'results' / 'reviewer_response' / 'extended_scaling.csv'
OUT_PNG = ROOT / 'springer_submission' / 'figures' / 'ibm_scaling.png'


def main():
    Ms, czs = [], []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ms.append(int(row['M']))
            czs.append(int(row['CZ_gates']))
    M = np.array(Ms, dtype=float)
    CZ = np.array(czs, dtype=float)

    # Refits (re-derive in-script for self-containment)
    mask_small = M <= 8
    s_small, i_small = np.polyfit(M[mask_small], CZ[mask_small], 1)
    c2, c1, c0 = np.polyfit(M, CZ, 2)  # quadratic with intercept
    pred = c2 * M**2 + c1 * M + c0
    ss_res = float(np.sum((CZ - pred) ** 2))
    ss_tot = float(np.sum((CZ - CZ.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    # Plot grid: extend slightly past data
    M_grid = np.linspace(0, 34, 400)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)

    # Logical operation count 15M - 1 (O(M))
    logical = 15.0 * M_grid - 1.0
    logical = np.clip(logical, 0, None)
    ax.plot(M_grid, logical, ':', color='0.35', linewidth=1.5,
            label=r'logical count $15M-1$  ($O(M)$)')

    # Linear law 14.1 M - 17.5 over M <= 8 (solid dashed); faint extension beyond
    M_lin = np.linspace(2, 8, 100)
    ax.plot(M_lin, 14.1 * M_lin - 17.5,
            '--', color='#cf4040', linewidth=1.8,
            label=r'linear law $14.1M-17.5$  ($M \leq 8$)')
    M_lin_ext = np.linspace(8, 33, 100)
    ax.plot(M_lin_ext, 14.1 * M_lin_ext - 17.5,
            '--', color='#cf4040', linewidth=1.0, alpha=0.35)

    # Quadratic fit (full range): 0.22 M^2 + 12.0 M - 13.7
    quad = c2 * M_grid**2 + c1 * M_grid + c0
    ax.plot(M_grid, quad, '-', color='#1f4fa6', linewidth=2.0,
            label=rf'transpiled fit $0.22M^2+12.0M-13.7$  ($R^2 = {r2:.4f}$)')

    # Measured points
    ax.scatter(M, CZ, s=58, color='#1f4fa6', edgecolor='white',
               linewidth=1.0, zorder=10,
               label='measured (transpiled, ibm_torino)')

    # Annotate routing-overhead callout near M=32
    ax.annotate('routing (SWAP) overhead\non heavy-hex topology',
                xy=(32, 595), xytext=(20.0, 530),
                fontsize=9, color='0.25',
                arrowprops=dict(arrowstyle='-', color='0.55', lw=0.9,
                                connectionstyle='arc3,rad=-0.2'))

    ax.set_xlabel('Number of channels $M$', fontsize=11)
    ax.set_ylabel('Two-qubit (CZ) gates', fontsize=11)
    ax.set_title('Transpiled CZ scaling on ibm_torino', fontsize=12)
    ax.set_xlim(0, 34)
    ax.set_ylim(0, CZ.max() * 1.10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    leg = ax.legend(loc='upper left', fontsize=9, frameon=True,
                    framealpha=0.92, edgecolor='0.85')
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    print(f'Wrote {OUT_PNG}  ({OUT_PNG.stat().st_size:,} bytes)')

    # Also drop a working copy at the workspace root for quick review
    fig.savefig(ROOT / 'fig4_scaling.png', dpi=200, bbox_inches='tight')
    print(f'Also wrote {ROOT / "fig4_scaling.png"}')


if __name__ == '__main__':
    main()
