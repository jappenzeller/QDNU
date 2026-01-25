"""
================================================================================
QUANTUM PN NEURON - 3D JULIA SURFACE GENERATOR
================================================================================

Single-file implementation for generating animated 3D Julia set surfaces
driven by PN (Positive-Negative) neuron dynamics.

COMPONENTS:
  1. PN Dynamics (FitzHugh-Nagumo oscillator for stable limit cycles)
  2. 8-Keyframe Phase Cycle (maps oscillation to animation keyframes)
  3. Quaternion Julia Set (3D slice of 4D Julia set)
  4. Isosurface Extraction (marching cubes -> mesh)
  5. Export (OBJ, PLY, STL for Blender)

USAGE:
  # Quick test
  python julia_surface_generator.py

  # In your code
  from julia_surface_generator import JuliaSurfaceGenerator
  gen = JuliaSurfaceGenerator()
  gen.generate_keyframe_sequence(output_dir='meshes')

DEPENDENCIES:
  pip install numpy scipy scikit-image

Author: QDNU Project
================================================================================
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import struct
import os

# Optional: scikit-image for marching cubes
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ==============================================================================
# SECTION 1: DATA STRUCTURES
# ==============================================================================

@dataclass
class Mesh:
    """Triangle mesh container."""
    vertices: np.ndarray   # (N, 3) float32
    faces: np.ndarray      # (M, 3) int32
    normals: Optional[np.ndarray] = None

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def compute_normals(self) -> np.ndarray:
        """Compute vertex normals from face geometry."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10

        vertex_normals = np.zeros_like(self.vertices)
        for i, face in enumerate(self.faces):
            for vi in face:
                vertex_normals[vi] += face_normals[i]

        vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-10
        self.normals = vertex_normals
        return self.normals


@dataclass
class PNState:
    """PN neuron state (a, b, c parameters)."""
    a: float = 0.2995      # Excitatory amplitude [0, 1] or FHN range
    b: float = 0.0         # Phase [0, 2pi]
    c: float = 0.3000      # Inhibitory amplitude [0, 1] or FHN range

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.a, self.b, self.c)

    def to_dict(self) -> Dict:
        return {'a': self.a, 'b': self.b, 'c': self.c, 'b_deg': np.degrees(self.b)}


@dataclass
class QuaternionC:
    """Quaternion Julia parameter c = (w, x, y, z)."""
    w: float = 0.0  # Real part
    x: float = 0.0  # i component
    y: float = 0.0  # j component
    z: float = 0.0  # k component

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)

    def magnitude(self) -> float:
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)


# ==============================================================================
# SECTION 2: PN DYNAMICS (FitzHugh-Nagumo Oscillator)
# ==============================================================================

class PNOscillatorFHN:
    """
    FitzHugh-Nagumo based PN dynamics for stable limit cycles.

    Equations:
        da/dt = a - a^3/3 - c + I_ext     (fast excitatory)
        dc/dt = epsilon(a + alpha - beta*c)  (slow inhibitory)
        db/dt = omega + kappa*a           (phase accumulation)

    The cubic nonlinearity + timescale separation guarantees oscillation.
    """

    # Pre-tuned configurations
    CONFIGS = {
        'default': {
            'epsilon': 0.08, 'alpha': 0.7, 'beta': 0.8,
            'I_ext': 0.5, 'omega': 0.3, 'kappa': 0.15,
            'description': 'Standard oscillator, period ~33'
        },
        'fast': {
            'epsilon': 0.15, 'alpha': 0.7, 'beta': 0.8,
            'I_ext': 0.6, 'omega': 0.5, 'kappa': 0.2,
            'description': 'Fast oscillation, period ~17'
        },
        'slow': {
            'epsilon': 0.04, 'alpha': 0.7, 'beta': 0.8,
            'I_ext': 0.4, 'omega': 0.2, 'kappa': 0.1,
            'description': 'Slow smooth morphing, period ~66'
        },
        'spiking': {
            'epsilon': 0.02, 'alpha': 0.7, 'beta': 0.8,
            'I_ext': 0.35, 'omega': 0.4, 'kappa': 0.3,
            'description': 'Sharp spikes (relaxation oscillator)'
        }
    }

    def __init__(self, config: str = 'default', **kwargs):
        """
        Initialize oscillator.

        Args:
            config: Name of preset config or 'custom'
            **kwargs: Override individual parameters
        """
        if config in self.CONFIGS:
            params = self.CONFIGS[config].copy()
        else:
            params = self.CONFIGS['default'].copy()

        params.update(kwargs)

        self.epsilon = params.get('epsilon', 0.08)
        self.alpha = params.get('alpha', 0.7)
        self.beta = params.get('beta', 0.8)
        self.I_ext = params.get('I_ext', 0.5)
        self.omega = params.get('omega', 0.3)
        self.kappa = params.get('kappa', 0.15)

    def derivatives(self, state: List[float], t: float) -> List[float]:
        """Compute derivatives for ODE integration."""
        a, b, c = state

        da = a - (a**3) / 3 - c + self.I_ext
        dc = self.epsilon * (a + self.alpha - self.beta * c)
        db = self.omega + self.kappa * a

        return [da, db, dc]

    def integrate(self, t_span: float, initial: Tuple = (0, 0, 0),
                  n_points: int = 10000) -> Dict[str, np.ndarray]:
        """
        Integrate dynamics over time.

        Returns:
            Dict with 't', 'a', 'b', 'c' arrays
        """
        t = np.linspace(0, t_span, n_points)
        solution = odeint(self.derivatives, list(initial), t)

        return {
            't': t,
            'a': solution[:, 0],
            'b': np.mod(solution[:, 1], 2 * np.pi),  # Wrap phase
            'c': solution[:, 2],
            'b_unwrapped': solution[:, 1]
        }

    def find_limit_cycle(self, t_span: float = 200, transient: float = 100,
                         n_points: int = 20000) -> Optional[Dict]:
        """
        Find stable limit cycle after transients decay.

        Returns:
            Dict with cycle data or None if no oscillation
        """
        traj = self.integrate(t_span, n_points=n_points)

        # Skip transient
        idx = int(len(traj['t']) * transient / t_span)
        a_s, b_s, c_s = traj['a'][idx:], traj['b'][idx:], traj['c'][idx:]
        t_s = traj['t'][idx:]

        # Check for oscillation
        if np.var(a_s) < 0.01:
            return None

        # Find period via FFT
        dt = traj['t'][1] - traj['t'][0]
        n = len(a_s)
        yf = np.abs(np.fft.fft(a_s - np.mean(a_s)))[:n//2]
        xf = np.fft.fftfreq(n, dt)[:n//2]

        peak_idx = np.argmax(yf[1:]) + 1
        freq = xf[peak_idx]
        period = 1.0 / freq if freq > 1e-6 else None

        if period is None:
            return None

        # Extract ~2 cycles
        n_samples = min(int(2 * period / dt), len(a_s) - 1)

        return {
            'period': period,
            'frequency': freq,
            'a': a_s[:n_samples],
            'b': b_s[:n_samples],
            'c': c_s[:n_samples],
            't': t_s[:n_samples] - t_s[0],
            'amplitude_a': np.ptp(a_s),
            'amplitude_c': np.ptp(c_s)
        }


# ==============================================================================
# SECTION 3: KEYFRAME SYSTEM
# ==============================================================================

class KeyframeSystem:
    """
    8-keyframe phase cycle for animation.

    Maps phase b in [0, 2pi] to 8 evenly-spaced keyframes.
    Each keyframe represents an "inflection point" in the oscillation.
    """

    N_KEYFRAMES = 8

    def __init__(self, a_fixed: float = 0.2995, c_fixed: float = 0.3):
        """
        Initialize with fixed a, c values (phase b varies).

        Args:
            a_fixed: Excitatory amplitude (from your explorer)
            c_fixed: Inhibitory amplitude (from your explorer)
        """
        self.a = a_fixed
        self.c = c_fixed

    def get_keyframe(self, index: int) -> PNState:
        """Get PN state for keyframe index (0-7)."""
        b = (2 * np.pi * index) / self.N_KEYFRAMES
        return PNState(a=self.a, b=b, c=self.c)

    def get_all_keyframes(self) -> List[PNState]:
        """Get all 8 keyframes."""
        return [self.get_keyframe(i) for i in range(self.N_KEYFRAMES)]

    def interpolate(self, t: float, period: float = 1.0) -> PNState:
        """
        Get interpolated state at time t.

        Args:
            t: Time value
            period: Duration of one complete cycle

        Returns:
            PNState with interpolated phase
        """
        b = (t / period) * 2 * np.pi
        b = b % (2 * np.pi)
        return PNState(a=self.a, b=b, c=self.c)

    def get_keyframe_info(self) -> List[Dict]:
        """Get detailed info for all keyframes."""
        info = []
        descriptions = [
            "Phase origin - E/I aligned",
            "Rising phase - E leads I",
            "Quarter cycle - Max E-I difference",
            "Approaching inversion",
            "Half cycle - E/I anti-aligned",
            "Descending phase - I leads E",
            "Three-quarter cycle",
            "Returning to origin"
        ]

        for i in range(self.N_KEYFRAMES):
            state = self.get_keyframe(i)
            info.append({
                'index': i,
                'b_rad': state.b,
                'b_deg': np.degrees(state.b),
                'b_frac': f"{i}/8 x 2pi",
                'description': descriptions[i]
            })

        return info

    def print_table(self):
        """Print formatted keyframe table."""
        print("\n" + "=" * 70)
        print("8-KEYFRAME PHASE CYCLE")
        print("=" * 70)
        print(f"Fixed: a = {self.a}, c = {self.c}")
        print(f"Phase increment: delta_b = pi/4 = 45 deg\n")

        print(f"{'KF':<4} {'b (rad)':<10} {'b (deg)':<10} {'Description':<30}")
        print("-" * 70)

        for kf in self.get_keyframe_info():
            print(f"{kf['index']:<4} {kf['b_rad']:<10.4f} {kf['b_deg']:<10.1f} {kf['description']:<30}")


# ==============================================================================
# SECTION 4: PN -> QUATERNION MAPPING
# ==============================================================================

class PNToQuaternionMapper:
    """
    Maps PN state (a, b, c) to quaternion Julia parameter c.

    The mapping determines how the Julia set morphs as parameters change.
    Tuned for visually interesting, smooth animations.
    """

    # Different mapping strategies
    STRATEGIES = ['default', 'symmetric', 'spiral', 'breathing']

    def __init__(self, strategy: str = 'default'):
        self.strategy = strategy

    def map(self, state: PNState) -> QuaternionC:
        """Map PN state to quaternion c."""
        if self.strategy == 'symmetric':
            return self._map_symmetric(state)
        elif self.strategy == 'spiral':
            return self._map_spiral(state)
        elif self.strategy == 'breathing':
            return self._map_breathing(state)
        else:
            return self._map_default(state)

    def _map_default(self, s: PNState) -> QuaternionC:
        """Default mapping - smooth phase-driven morphology."""
        return QuaternionC(
            w=-0.2 + 0.3 * np.cos(s.b),
            x=0.3 * np.sin(s.b) * (s.a - 0.5),
            y=0.5 * np.sin(s.b / 2),
            z=0.1 * np.cos(s.b) * (s.c - 0.5)
        )

    def _map_symmetric(self, s: PNState) -> QuaternionC:
        """Symmetric mapping - c traces a circle."""
        r = 0.4
        return QuaternionC(
            w=-0.3 + r * np.cos(s.b),
            x=r * np.sin(s.b),
            y=0.2 * np.sin(2 * s.b),
            z=0.0
        )

    def _map_spiral(self, s: PNState) -> QuaternionC:
        """Spiral mapping - all 4 components vary."""
        return QuaternionC(
            w=-0.2 + 0.25 * np.cos(s.b),
            x=0.25 * np.sin(s.b),
            y=0.25 * np.cos(s.b + np.pi/3),
            z=0.25 * np.sin(s.b + np.pi/3)
        )

    def _map_breathing(self, s: PNState) -> QuaternionC:
        """Breathing mapping - pulsing amplitude."""
        scale = 0.3 + 0.15 * np.sin(s.b)
        return QuaternionC(
            w=-0.4 * scale,
            x=0.3 * scale * np.cos(s.b),
            y=0.3 * scale * np.sin(s.b),
            z=0.1 * scale
        )


# ==============================================================================
# SECTION 5: QUATERNION JULIA SET COMPUTATION
# ==============================================================================

class QuaternionJulia:
    """
    Computes 3D Julia set volumes using quaternion iteration.

    Julia set: z -> z^2 + c where z, c are quaternions
    3D slice: fix one quaternion component (usually w=0)
    """

    def __init__(self, resolution: int = 64, bounds: float = 1.5,
                 max_iter: int = 20, slice_w: float = 0.0):
        """
        Initialize Julia set parameters.

        Args:
            resolution: Grid points per axis
            bounds: Spatial extent [-bounds, bounds]^3
            max_iter: Iteration limit
            slice_w: Fixed w-component for 3D slice
        """
        self.resolution = resolution
        self.bounds = bounds
        self.max_iter = max_iter
        self.slice_w = slice_w

        # Pre-compute coordinate grid
        lin = np.linspace(-bounds, bounds, resolution)
        self.X, self.Y, self.Z = np.meshgrid(lin, lin, lin, indexing='ij')

    def compute(self, c: QuaternionC) -> np.ndarray:
        """
        Compute Julia set volume for given quaternion c.

        Returns:
            3D array of iteration counts
        """
        # Initialize quaternion z = (slice_w, x, y, z)
        qw = np.full_like(self.X, self.slice_w)
        qx = self.X.copy()
        qy = self.Y.copy()
        qz = self.Z.copy()

        # Julia parameter
        cw, cx, cy, cz = c.to_tuple()

        # Output array
        iterations = np.zeros_like(self.X, dtype=np.float32)
        active = np.ones_like(self.X, dtype=bool)

        for n in range(self.max_iter):
            if not np.any(active):
                break

            # z^2 (quaternion square)
            qw_new = qw*qw - qx*qx - qy*qy - qz*qz
            qx_new = 2*qw*qx
            qy_new = 2*qw*qy
            qz_new = 2*qw*qz

            # z^2 + c
            qw = qw_new + cw
            qx = qx_new + cx
            qy = qy_new + cy
            qz = qz_new + cz

            # Escape check
            mag_sq = qw*qw + qx*qx + qy*qy + qz*qz
            escaped = active & (mag_sq > 4.0)

            # Smooth iteration count
            with np.errstate(invalid='ignore', divide='ignore'):
                smooth = n + 1 - np.log2(np.log2(np.sqrt(mag_sq) + 1) + 1)
                smooth = np.nan_to_num(smooth, nan=n)

            iterations[escaped] = smooth[escaped]
            active = active & ~escaped

        iterations[active] = self.max_iter
        return iterations

    def compute_from_pn(self, state: PNState,
                        mapper: Optional[PNToQuaternionMapper] = None) -> np.ndarray:
        """Compute Julia volume from PN state."""
        if mapper is None:
            mapper = PNToQuaternionMapper()

        c = mapper.map(state)
        return self.compute(c)


# ==============================================================================
# SECTION 6: ISOSURFACE EXTRACTION
# ==============================================================================

class IsosurfaceExtractor:
    """Extracts triangle mesh from volume using marching cubes."""

    def __init__(self, bounds: float = 1.5):
        self.bounds = bounds

    def extract(self, volume: np.ndarray, level: Optional[float] = None) -> Optional[Mesh]:
        """
        Extract isosurface mesh from volume.

        Args:
            volume: 3D scalar field
            level: Isosurface threshold (auto if None)

        Returns:
            Mesh or None if extraction fails
        """
        if not HAS_SKIMAGE:
            print("ERROR: scikit-image required. pip install scikit-image")
            return None

        if level is None:
            level = np.percentile(volume, 60)

        try:
            spacing = (2 * self.bounds / volume.shape[0],) * 3
            verts, faces, normals, _ = measure.marching_cubes(
                volume, level=level, spacing=spacing
            )

            # Center around origin
            verts -= self.bounds

            return Mesh(
                vertices=verts.astype(np.float32),
                faces=faces.astype(np.int32),
                normals=normals.astype(np.float32)
            )
        except Exception as e:
            print(f"Marching cubes failed: {e}")
            return None


# ==============================================================================
# SECTION 7: MESH EXPORT
# ==============================================================================

class MeshExporter:
    """Exports mesh to various formats."""

    @staticmethod
    def to_obj(mesh: Mesh, filename: str):
        """Export to Wavefront OBJ."""
        with open(filename, 'w') as f:
            f.write(f"# Julia Surface - {mesh.n_vertices} verts, {mesh.n_faces} faces\n")

            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            if mesh.normals is not None:
                for n in mesh.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            for face in mesh.faces:
                if mesh.normals is not None:
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    @staticmethod
    def to_ply(mesh: Mesh, filename: str):
        """Export to PLY (good for Blender)."""
        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {mesh.n_vertices}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if mesh.normals is not None:
                f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write(f"element face {mesh.n_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for i, v in enumerate(mesh.vertices):
                line = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
                if mesh.normals is not None:
                    n = mesh.normals[i]
                    line += f" {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}"
                f.write(line + "\n")

            for face in mesh.faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    @staticmethod
    def to_stl(mesh: Mesh, filename: str):
        """Export to binary STL."""
        if mesh.normals is None:
            mesh.compute_normals()

        with open(filename, 'wb') as f:
            f.write(b'Julia Surface' + b'\0' * 67)
            f.write(struct.pack('<I', mesh.n_faces))

            for face in mesh.faces:
                n = mesh.normals[face[0]]
                f.write(struct.pack('<3f', *n))
                for vi in face:
                    f.write(struct.pack('<3f', *mesh.vertices[vi]))
                f.write(struct.pack('<H', 0))


# ==============================================================================
# SECTION 8: HIGH-LEVEL GENERATOR
# ==============================================================================

class JuliaSurfaceGenerator:
    """
    High-level interface for generating Julia surface animations.

    Usage:
        gen = JuliaSurfaceGenerator(resolution=64)
        gen.generate_keyframe_sequence('meshes/')
        gen.generate_smooth_animation('frames/', n_frames=120)
    """

    def __init__(self, resolution: int = 64, bounds: float = 1.2,
                 max_iter: int = 15, mapping_strategy: str = 'default'):
        """
        Initialize generator.

        Args:
            resolution: Volume resolution (48-128 typical)
            bounds: Spatial extent
            max_iter: Julia iteration limit
            mapping_strategy: 'default', 'symmetric', 'spiral', 'breathing'
        """
        self.julia = QuaternionJulia(resolution, bounds, max_iter)
        self.extractor = IsosurfaceExtractor(bounds)
        self.mapper = PNToQuaternionMapper(mapping_strategy)
        self.keyframes = KeyframeSystem()

    def generate_single(self, state: PNState,
                        filename: str, fmt: str = 'ply') -> Optional[Mesh]:
        """
        Generate single Julia surface mesh.

        Args:
            state: PN state (a, b, c)
            filename: Output filename (without extension)
            fmt: 'obj', 'ply', or 'stl'

        Returns:
            Generated mesh
        """
        # Compute volume
        volume = self.julia.compute_from_pn(state, self.mapper)

        # Extract mesh
        mesh = self.extractor.extract(volume)
        if mesh is None:
            return None

        # Export
        if fmt == 'obj':
            MeshExporter.to_obj(mesh, f"{filename}.obj")
        elif fmt == 'stl':
            MeshExporter.to_stl(mesh, f"{filename}.stl")
        else:
            MeshExporter.to_ply(mesh, f"{filename}.ply")

        return mesh

    def generate_keyframe_sequence(self, output_dir: str = 'keyframes',
                                    fmt: str = 'ply') -> List[str]:
        """
        Generate 8-keyframe mesh sequence.

        Args:
            output_dir: Output directory
            fmt: Export format

        Returns:
            List of output filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        files = []

        print(f"Generating 8 keyframe meshes (resolution={self.julia.resolution})...")

        for i in range(8):
            state = self.keyframes.get_keyframe(i)
            c = self.mapper.map(state)

            print(f"  KF{i}: b={np.degrees(state.b):>6.1f} deg  "
                  f"c=({c.w:.3f}, {c.x:.3f}, {c.y:.3f}, {c.z:.3f})")

            filename = os.path.join(output_dir, f"julia_kf{i:02d}")
            mesh = self.generate_single(state, filename, fmt)

            if mesh:
                files.append(f"{filename}.{fmt}")
                print(f"       -> {mesh.n_vertices} verts, {mesh.n_faces} faces")

        return files

    def generate_smooth_animation(self, output_dir: str = 'frames',
                                   n_frames: int = 120,
                                   fmt: str = 'obj') -> List[str]:
        """
        Generate smooth animation with many frames.

        Args:
            output_dir: Output directory
            n_frames: Total frames for one cycle
            fmt: Export format

        Returns:
            List of output filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        files = []

        print(f"Generating {n_frames} frames...")

        for i in range(n_frames):
            state = self.keyframes.interpolate(i / n_frames)
            filename = os.path.join(output_dir, f"julia_{i:04d}")

            mesh = self.generate_single(state, filename, fmt)
            if mesh:
                files.append(f"{filename}.{fmt}")

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n_frames}")

        return files

    def generate_from_oscillator(self, output_dir: str = 'osc_frames',
                                  config: str = 'default',
                                  duration: float = 100,
                                  n_frames: int = 60,
                                  fmt: str = 'ply') -> List[str]:
        """
        Generate animation driven by FHN oscillator.

        This captures the actual limit cycle dynamics rather than
        uniform phase sampling.

        Args:
            output_dir: Output directory
            config: Oscillator config name
            duration: Simulation time
            n_frames: Frames to output
            fmt: Export format

        Returns:
            List of output filenames
        """
        os.makedirs(output_dir, exist_ok=True)

        # Run oscillator
        osc = PNOscillatorFHN(config)
        traj = osc.integrate(duration, n_points=n_frames * 100)

        # Sample trajectory
        indices = np.linspace(0, len(traj['t']) - 1, n_frames, dtype=int)
        files = []

        print(f"Generating {n_frames} oscillator-driven frames...")

        for i, idx in enumerate(indices):
            state = PNState(
                a=float(traj['a'][idx]),
                b=float(traj['b'][idx]),
                c=float(traj['c'][idx])
            )

            filename = os.path.join(output_dir, f"julia_{i:04d}")
            mesh = self.generate_single(state, filename, fmt)

            if mesh:
                files.append(f"{filename}.{fmt}")

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_frames}")

        return files


# ==============================================================================
# SECTION 9: VISUALIZATION (Matplotlib preview)
# ==============================================================================

def preview_volume_slices(volume: np.ndarray, title: str = "Julia Volume",
                          save_path: Optional[str] = None):
    """Preview orthogonal slices through volume."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mid = volume.shape[0] // 2

    axes[0].imshow(volume[:, :, mid], cmap='magma', origin='lower')
    axes[0].set_title(f'XY (Z={mid})')

    axes[1].imshow(volume[:, mid, :], cmap='magma', origin='lower')
    axes[1].set_title(f'XZ (Y={mid})')

    axes[2].imshow(volume[mid, :, :], cmap='magma', origin='lower')
    axes[2].set_title(f'YZ (X={mid})')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


def preview_mesh(mesh: Mesh, title: str = "Julia Surface",
                 save_path: Optional[str] = None):
    """Preview mesh with matplotlib (low quality but quick)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    verts = mesh.vertices[mesh.faces]

    # Subsample for speed
    if len(verts) > 3000:
        idx = np.random.choice(len(verts), 3000, replace=False)
        verts = verts[idx]

    poly = Poly3DCollection(verts, alpha=0.7, edgecolor='k', linewidth=0.1)
    poly.set_facecolor('cyan')
    ax.add_collection3d(poly)

    lim = np.max(np.abs(mesh.vertices))
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    return fig


# ==============================================================================
# SECTION 10: CLI / MAIN
# ==============================================================================

def main():
    """Main entry point with examples."""
    import matplotlib
    matplotlib.use('Agg')

    print("=" * 70)
    print("QUANTUM PN NEURON - 3D JULIA SURFACE GENERATOR")
    print("=" * 70)

    # Show keyframe table
    kf = KeyframeSystem()
    kf.print_table()

    # Test single surface
    print("\n--- Single Surface Test ---")
    gen = JuliaSurfaceGenerator(resolution=48)

    state = PNState(a=0.2995, b=np.pi/4, c=0.3)  # Keyframe 1 (45 deg)
    print(f"State: a={state.a}, b={np.degrees(state.b):.1f} deg, c={state.c}")

    c = gen.mapper.map(state)
    print(f"Quaternion c: ({c.w:.4f}, {c.x:.4f}, {c.y:.4f}, {c.z:.4f})")

    volume = gen.julia.compute_from_pn(state, gen.mapper)
    print(f"Volume: {volume.shape}, range [{volume.min():.2f}, {volume.max():.2f}]")

    mesh = gen.extractor.extract(volume)
    if mesh:
        print(f"Mesh: {mesh.n_vertices} verts, {mesh.n_faces} faces")
        MeshExporter.to_ply(mesh, 'julia_test.ply')
        print("Exported: julia_test.ply")

    # Generate 8 keyframes
    print("\n--- 8-Keyframe Sequence ---")
    gen.generate_keyframe_sequence('keyframes', fmt='ply')

    print("\n--- Complete ---")
    print("\nOutput files:")
    print("  julia_test.ply          - Single test mesh")
    print("  keyframes/julia_kf*.ply - 8 keyframe meshes")
    print("\nFor Blender: Import PLY files, create shape keys or mesh sequence")


if __name__ == '__main__':
    main()
