"""
LIVE JULIA/METRIC GENERATOR
============================

Generate Julia sphere or quantum metric sphere for Blender live reload.

Usage:
    # Julia spheres
    python -m blender.generate_live                    # Default Julia
    python -m blender.generate_live -c -0.4+0.6j       # Specific Julia c
    python -m blender.generate_live --preset rabbit    # Named preset
    python -m blender.generate_live -a 0.6 -b 90 -p 0.3  # From PN parameters

    # Quantum metric spheres (practical seizure visualization)
    python -m blender.generate_live --metric concurrence   # Entanglement landscape
    python -m blender.generate_live --metric sensitivity   # Parameter sensitivity
    python -m blender.generate_live --metric purity        # State purity

The output always goes to blender/live_julia.ply for Blender to reload.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.animation.julia_sphere_refined import (
    create_julia_sphere_smooth,
    export_ply
)
from visualization.agate_parameter_explorer import (
    create_quantum_metric_sphere,
    export_ply as export_metric_ply
)

# Output path - Blender watches this
LIVE_OUTPUT = PROJECT_ROOT / "blender" / "live_julia.ply"

# Presets
PRESETS = {
    'rabbit': complex(-0.122, 0.745),      # Douady rabbit (3 lobes)
    'sanmarco': complex(-0.75, 0),          # San Marco (2 lobes)
    'spiral': complex(0.285, 0.01),         # Spiral
    'four': complex(-0.4, 0.6),             # Four lobes
    'siegel': complex(-0.391, -0.587),      # Siegel disk
    'dendrite': complex(0, 1),              # Dendrite
    'galaxy': complex(-0.8, 0.156),         # Galaxy spiral
    'lightning': complex(-0.7269, 0.1889),  # Lightning
}


def pn_to_julia_c(a: float, b_deg: float, pn_c: float) -> complex:
    """Convert PN parameters to Julia c parameter."""
    import numpy as np
    b = np.radians(b_deg)
    # Map: real = (a - c) scaled, imag = phase
    real_part = (a - pn_c) * 0.8
    imag_part = 0.6 * np.sin(b)
    return complex(real_part, imag_part)


def main():
    parser = argparse.ArgumentParser(description='Generate Julia or quantum metric sphere for Blender')

    # Mode selection
    parser.add_argument('--metric', '-m', choices=['concurrence', 'sensitivity', 'purity'],
                       help='Generate quantum metric sphere instead of Julia')

    # Julia c parameter (direct)
    parser.add_argument('-c', '--julia-c', type=complex, help='Julia c parameter (e.g., -0.4+0.6j)')

    # Named preset
    parser.add_argument('--preset', '-P', choices=list(PRESETS.keys()),
                       help='Use named preset')

    # PN parameters
    parser.add_argument('-a', '--pn-a', type=float, help='PN excitatory amplitude [0,1]')
    parser.add_argument('-b', '--pn-b', type=float, help='PN phase in degrees [0,360]')
    parser.add_argument('-p', '--pn-c', type=float, help='PN inhibitory amplitude [0,1]')

    # Generation parameters
    parser.add_argument('-r', '--resolution', type=int, default=120,
                       help='Mesh resolution (default: 120)')
    parser.add_argument('-s', '--smoothing', type=float, default=3.5,
                       help='Smoothing sigma (default: 3.5)')
    parser.add_argument('-d', '--depth', type=float, default=0.35,
                       help='Canyon depth (default: 0.35)')
    parser.add_argument('--invert', action='store_true',
                       help='Invert (ridges instead of canyons)')

    # Output
    parser.add_argument('-o', '--output', type=str, default=str(LIVE_OUTPUT),
                       help=f'Output path (default: {LIVE_OUTPUT})')

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # METRIC MODE - generate quantum metric sphere
    if args.metric:
        print(f"Generating {args.metric} quantum metric sphere...")
        print(f"  resolution = {args.resolution}")
        print(f"  smoothing = {args.smoothing}")

        data = create_quantum_metric_sphere(
            metric=args.metric,
            resolution=args.resolution,
            smoothing=args.smoothing
        )

        export_metric_ply(data, str(output_path))

        print(f"\nOutput: {output_path}")
        print(f"Vertices: {len(data['vertices'])}")
        print(f"\nMetric: {args.metric}")
        print("  - Height shows where metric is highest")
        print("  - Color: blue (low) -> red (high)")
        print("\nBlender should auto-reload if watching is enabled.")
        return

    # Determine Julia c
    if args.julia_c is not None:
        c = args.julia_c
        source = f"direct: {c}"
    elif args.preset:
        c = PRESETS[args.preset]
        source = f"preset '{args.preset}': {c}"
    elif args.pn_a is not None or args.pn_b is not None or args.pn_c is not None:
        # Use PN parameters (defaults if not specified)
        a = args.pn_a if args.pn_a is not None else 0.5
        b = args.pn_b if args.pn_b is not None else 90
        pn_c = args.pn_c if args.pn_c is not None else 0.3
        c = pn_to_julia_c(a, b, pn_c)
        source = f"PN(a={a}, b={b}deg, c={pn_c}) -> {c}"
    else:
        # Default
        c = complex(-0.4, 0.6)
        source = f"default: {c}"

    print(f"Generating Julia sphere...")
    print(f"  c = {c} ({source})")
    print(f"  resolution = {args.resolution}")
    print(f"  smoothing = {args.smoothing}")
    print(f"  depth = {args.depth}")
    print(f"  invert = {args.invert}")

    # Generate Julia sphere
    data = create_julia_sphere_smooth(
        c,
        resolution=args.resolution,
        smoothing_sigma=args.smoothing,
        canyon_depth=args.depth,
        invert=args.invert
    )

    # Export
    export_ply(data, str(output_path))

    print(f"\nOutput: {output_path}")
    print(f"Vertices: {len(data['vertices'])}")
    print("\nBlender should auto-reload if watching is enabled.")


if __name__ == '__main__':
    main()
