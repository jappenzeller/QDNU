"""
Julia Set Fractal Generator from Quantum State Amplitudes.

The 4 complex amplitudes of a 2-qubit A-Gate state create a unique fractal fingerprint,
providing visual representation of quantum state evolution during neuron activation.
"""

import numpy as np
from typing import Tuple, List
from PIL import Image


def julia_set(c: complex, width: int = 512, height: int = 512,
              x_range: Tuple[float, float] = (-2, 2),
              y_range: Tuple[float, float] = (-2, 2),
              max_iter: int = 100) -> np.ndarray:
    """
    Generate Julia set for parameter c.

    Args:
        c: Julia set parameter (complex)
        width, height: Image dimensions
        x_range, y_range: Complex plane bounds
        max_iter: Maximum iterations

    Returns:
        2D array of escape iteration counts (with smooth coloring)
    """
    x = np.linspace(x_range[0], x_range[1], width)
    y = np.linspace(y_range[0], y_range[1], height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    output = np.zeros(Z.shape, dtype=np.float32)
    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + c
        escaped = np.abs(Z) > 2
        # Smooth coloring using fractional escape
        with np.errstate(divide='ignore', invalid='ignore'):
            smooth_val = i + 1 - np.log2(np.log2(np.abs(Z[mask & escaped]) + 1))
            smooth_val = np.nan_to_num(smooth_val, nan=i, posinf=i, neginf=i)
        output[mask & escaped] = smooth_val
        mask = mask & ~escaped

    return output


def statevector_to_julia_param(amplitudes: np.ndarray, method: str = 'weighted') -> complex:
    """
    Convert 4-amplitude statevector to Julia set parameter c.

    Methods:
        'dominant': Use amplitude with largest magnitude
        'weighted': Weighted sum by magnitude
        'determinant': det-like combination alpha_00*alpha_11 - alpha_01*alpha_10
        'interference': alpha_00 + alpha_11 (diagonal interference)
        'antidiagonal': alpha_01 + alpha_10 (anti-diagonal)
        'difference': alpha_00 - alpha_11 (population difference)

    Args:
        amplitudes: Array of 4 complex amplitudes [alpha_00, alpha_01, alpha_10, alpha_11]
        method: Conversion method

    Returns:
        Complex parameter c for Julia set
    """
    amps = np.array(amplitudes, dtype=complex)

    if method == 'dominant':
        c = amps[np.argmax(np.abs(amps))]

    elif method == 'weighted':
        weights = np.abs(amps)
        total_weight = np.sum(weights)
        if total_weight > 1e-10:
            c = np.sum(amps * weights) / total_weight
        else:
            c = 0

    elif method == 'determinant':
        c = amps[0] * amps[3] - amps[1] * amps[2]

    elif method == 'interference':
        c = amps[0] + amps[3]

    elif method == 'antidiagonal':
        c = amps[1] + amps[2]

    elif method == 'difference':
        c = amps[0] - amps[3]

    else:
        raise ValueError(f"Unknown method: {method}")

    # Scale to interesting Julia region (|c| < 2 for bounded sets)
    c = c * 0.8

    return complex(c)


def generate_fractal_from_state(statevector: np.ndarray,
                                width: int = 512, height: int = 512,
                                max_iter: int = 100,
                                method: str = 'weighted') -> np.ndarray:
    """
    Generate Julia fractal from quantum statevector.

    Returns:
        2D numpy array of iteration counts
    """
    c = statevector_to_julia_param(statevector, method=method)
    return julia_set(c, width, height, max_iter=max_iter)


def apply_colormap(normalized: np.ndarray, colormap: str = 'quantum') -> np.ndarray:
    """
    Apply colormap to normalized [0,1] array.

    Returns:
        RGB array of shape (H, W, 3) with dtype uint8
    """
    if colormap == 'quantum':
        # Custom colormap: deep blue -> purple -> gold (matches dark theme)
        r = (normalized ** 0.5 * 255).astype(np.uint8)
        g = (normalized ** 2 * 180).astype(np.uint8)
        b = ((1 - normalized) ** 0.7 * 255).astype(np.uint8)

    elif colormap == 'grayscale':
        gray = (normalized * 255).astype(np.uint8)
        r = g = b = gray

    elif colormap == 'fire':
        r = np.clip(normalized * 3, 0, 1)
        g = np.clip(normalized * 3 - 1, 0, 1)
        b = np.clip(normalized * 3 - 2, 0, 1)
        r = (r * 255).astype(np.uint8)
        g = (g * 255).astype(np.uint8)
        b = (b * 255).astype(np.uint8)

    elif colormap == 'plasma':
        r = np.clip(normalized * 2, 0, 1)
        g = np.clip(normalized * 1.5 - 0.3, 0, 1)
        b = np.clip(1 - normalized * 1.2, 0, 1)
        r = (r * 255).astype(np.uint8)
        g = (g * 255).astype(np.uint8)
        b = (b * 255).astype(np.uint8)

    elif colormap == 'ocean':
        r = (normalized ** 2 * 200).astype(np.uint8)
        g = (normalized * 220 + 30).astype(np.uint8)
        b = ((0.3 + 0.7 * normalized) * 255).astype(np.uint8)

    elif colormap == 'seizure':
        # Custom colormap for seizure visualization: dark blue -> red -> yellow
        # Low values (interictal) = blue, high values (ictal) = red/yellow
        r = np.clip(normalized * 2, 0, 1)
        g = np.clip(normalized * 2 - 0.5, 0, 1)
        b = np.clip(1 - normalized * 1.5, 0, 1)
        r = (r * 255).astype(np.uint8)
        g = (g * 255).astype(np.uint8)
        b = (b * 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown colormap: {colormap}")

    return np.stack([r, g, b], axis=-1)


def fractal_to_image(fractal: np.ndarray, colormap: str = 'quantum') -> Image.Image:
    """
    Convert fractal array to colored PIL Image.

    Colormaps:
        'quantum': Blue-purple-gold (matches dark theme)
        'seizure': Blue-red-yellow (for ictal/interictal contrast)
        'grayscale': Simple grayscale
        'fire': Black-red-yellow-white
        'plasma': Purple-pink-orange
        'ocean': Deep blue-cyan-white
    """
    f_min, f_max = fractal.min(), fractal.max()
    if f_max > f_min:
        normalized = (fractal - f_min) / (f_max - f_min)
    else:
        normalized = np.zeros_like(fractal)

    rgb = apply_colormap(normalized, colormap)
    return Image.fromarray(rgb)


def save_fractal(fractal: np.ndarray, filepath: str, colormap: str = 'quantum'):
    """Save fractal as image file."""
    img = fractal_to_image(fractal, colormap)
    img.save(filepath)


def generate_fractal_sequence(statevectors: List[np.ndarray],
                               width: int = 256, height: int = 256,
                               max_iter: int = 100,
                               method: str = 'weighted') -> List[np.ndarray]:
    """
    Generate sequence of fractals for animation.

    Args:
        statevectors: List of statevector arrays
        width, height: Image dimensions
        max_iter: Maximum iterations
        method: Julia parameter method

    Returns:
        List of fractal arrays
    """
    return [generate_fractal_from_state(sv, width, height, max_iter, method)
            for sv in statevectors]


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, str(__file__).replace('visualization/fractal_generator.py', ''))
    from quantum_agate import create_single_channel_agate
    from qiskit.quantum_info import Statevector

    # Generate test fractal
    circuit = create_single_channel_agate(a=0.6, b=1.2, c=0.5)
    sv = Statevector.from_instruction(circuit).data

    print("Statevector:", sv)
    print("Julia parameter (weighted):", statevector_to_julia_param(sv, 'weighted'))

    fractal = generate_fractal_from_state(sv, width=512, height=512, max_iter=100)
    print(f"Fractal shape: {fractal.shape}")
    print(f"Fractal range: [{fractal.min():.2f}, {fractal.max():.2f}]")

    # Save with different colormaps
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'fractals')
    os.makedirs(output_dir, exist_ok=True)

    for cmap in ['quantum', 'seizure', 'fire']:
        path = os.path.join(output_dir, f'test_fractal_{cmap}.png')
        save_fractal(fractal, path, colormap=cmap)
        print(f"Saved: {path}")
