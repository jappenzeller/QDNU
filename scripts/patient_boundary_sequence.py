"""
Generate a PLY sequence for a patient's boundary crossing trajectory.

Takes EEG-derived Julia c-parameters sorted by Mandelbrot boundary distance,
interpolates smoothly, and exports 3D heightfield frames as PLY meshes.

Usage: python scripts/patient_boundary_sequence.py
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.interactive.julia_3d_boundary import (
    compute_julia_heightfield,
    export_heightfield_ply,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PATIENT = "Patient_8"
N_FRAMES = 100
RESOLUTION = 300       # heightfield grid resolution
MAX_ITER = 256
HEIGHT_SCALE = 0.5
OUTPUT_DIR = Path(__file__).parent.parent / "julia_3d_sequence" / PATIENT


def load_patient_trajectory(patient: str) -> list:
    """Load and sort segment c-parameters by boundary distance."""
    metrics_path = Path(__file__).parent.parent / "analysis_results" / "segment_metrics.json"
    with open(metrics_path) as f:
        data = json.load(f)

    segs = [d for d in data if d['patient'] == patient]
    segs.sort(key=lambda d: d['boundary_distance'])
    return segs


def filter_coherent_path(segs: list) -> list:
    """
    Filter to one coherent trajectory through the complex plane.

    Patient_8 has two branches:
    - Main cardioid path: c_real ~ -0.4 to -0.9, c_imag positive
    - Period-2 bulb: c_real ~ -1.0, c_imag varies

    We pick the main cardioid path for the smoothest visual sweep.
    """
    # Keep segments following the main cardioid boundary (c_imag > -0.15)
    # This gives us a clean sweep from inside to outside
    filtered = [s for s in segs if s['julia_c_imag'] > -0.15]

    if len(filtered) < 10:
        # Fallback: use all segments
        return segs

    return filtered


def interpolate_trajectory(segs: list, n_frames: int) -> list:
    """
    Interpolate c-parameters to produce n_frames smooth steps.

    Returns list of complex c values.
    """
    c_reals = np.array([s['julia_c_real'] for s in segs])
    c_imags = np.array([s['julia_c_imag'] for s in segs])
    dists = np.array([s['boundary_distance'] for s in segs])

    # Parameterize by boundary distance (normalized to [0, 1])
    t_data = (dists - dists[0]) / (dists[-1] - dists[0])

    # Target parameter values
    t_target = np.linspace(0, 1, n_frames)

    # Interpolate real and imaginary parts
    c_real_interp = np.interp(t_target, t_data, c_reals)
    c_imag_interp = np.interp(t_target, t_data, c_imags)

    return [complex(r, i) for r, i in zip(c_real_interp, c_imag_interp)]


def is_in_mandelbrot(c: complex, max_iter: int = 100) -> bool:
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True


def main():
    print("=" * 70)
    print(f"BOUNDARY CROSSING SEQUENCE: {PATIENT}")
    print("=" * 70)

    # Load data
    segs = load_patient_trajectory(PATIENT)
    print(f"Total segments: {len(segs)}")
    print(f"Boundary distance range: [{segs[0]['boundary_distance']:.3f}, {segs[-1]['boundary_distance']:.3f}]")

    # Filter to coherent path
    path_segs = filter_coherent_path(segs)
    print(f"Filtered path segments: {len(path_segs)}")
    print(f"Filtered distance range: [{path_segs[0]['boundary_distance']:.3f}, {path_segs[-1]['boundary_distance']:.3f}]")

    # Interpolate
    c_values = interpolate_trajectory(path_segs, N_FRAMES)
    print(f"\nInterpolated to {N_FRAMES} frames")
    print(f"c start: {c_values[0].real:.4f} + {c_values[0].imag:.4f}i")
    print(f"c end:   {c_values[-1].real:.4f} + {c_values[-1].imag:.4f}i")

    # Find boundary crossing frame
    for i, c in enumerate(c_values):
        if not is_in_mandelbrot(c):
            print(f"\nBoundary crossing at frame {i}/{N_FRAMES}")
            print(f"  c = {c.real:.4f} + {c.imag:.4f}i")
            break

    # Export sequence
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting to {OUTPUT_DIR}/")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}, max_iter: {MAX_ITER}")

    for i, c in enumerate(c_values):
        in_m = is_in_mandelbrot(c)
        status = "INSIDE" if in_m else "OUTSIDE"

        X, Y, Z = compute_julia_heightfield(
            c, resolution=RESOLUTION, max_iter=MAX_ITER
        )

        filepath = OUTPUT_DIR / f"frame_{i:04d}.ply"
        export_heightfield_ply(X, Y, Z, str(filepath), HEIGHT_SCALE)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i+1:3d}/{N_FRAMES}  c=({c.real:.4f}, {c.imag:.4f}i)  [{status}]")

    # Write metadata
    metadata = {
        'patient': PATIENT,
        'n_frames': N_FRAMES,
        'resolution': RESOLUTION,
        'max_iter': MAX_ITER,
        'height_scale': HEIGHT_SCALE,
        'frames': [
            {
                'frame': i,
                'c_real': c.real,
                'c_imag': c.imag,
                'in_mandelbrot': is_in_mandelbrot(c),
            }
            for i, c in enumerate(c_values)
        ]
    }

    with open(OUTPUT_DIR / 'sequence_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=float)

    print(f"\n{'='*70}")
    print(f"Sequence complete: {N_FRAMES} frames in {OUTPUT_DIR}/")
    print(f"Import to Blender: File > Import > Stanford (.ply)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
