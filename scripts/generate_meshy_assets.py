#!/usr/bin/env python3
"""
C-004: Generate 3D Assets via Meshy API

Generates GLB assets for the QDNU visualization:
1. Riemannian manifold surface (text-to-3D)
2. EEG helmet (image-to-3D, if photos available)

Usage:
    export MESHY_API_KEY="your_key_here"
    python scripts/generate_meshy_assets.py
"""

import os
import sys
import time
import json
import base64
from pathlib import Path

import requests

# Configuration
API_KEY = os.environ.get("MESHY_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE = "https://api.meshy.ai/openapi/v2"
ASSETS_DIR = Path("visualization_data/assets")


def poll_task(task_id: str, endpoint: str, interval: int = 10, timeout: int = 600) -> dict:
    """Poll until task completes or fails."""
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE}/{endpoint}/{task_id}", headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        progress = data.get("progress", 0)
        print(f"  [{int(time.time()-start):3d}s] Status: {status} ({progress}%)")
        if status == "SUCCEEDED":
            return data
        if status in ("FAILED", "EXPIRED"):
            raise RuntimeError(f"Task {task_id} failed: {data}")
        time.sleep(interval)
    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")


def download_glb(url: str, path: Path) -> None:
    """Download GLB file from URL."""
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Downloaded: {path} ({path.stat().st_size // 1024} KB)")


def generate_manifold_surface() -> dict:
    """Generate Riemannian manifold surface via text-to-3D."""
    print("\n[1/2] Generating manifold surface (text-to-3D)...")

    payload = {
        "mode": "preview",
        "prompt": (
            "abstract mathematical saddle surface, symmetric positive definite manifold, "
            "smooth curved topology, dark metallic material with subtle grid lines, "
            "scientific 3D visualization, clean geometry, no text, no labels"
        ),
        "negative_prompt": "text, labels, rough, noisy, low quality, cartoon",
        "art_style": "realistic",
        "topology": "quad",
        "target_polycount": 10000,
    }

    r = requests.post(f"{BASE}/text-to-3d", headers=HEADERS, json=payload)
    r.raise_for_status()
    task_id = r.json()["result"]
    print(f"  Task ID: {task_id}")

    result = poll_task(task_id, "text-to-3d")
    glb_url = result["model_urls"]["glb"]
    output_path = ASSETS_DIR / "manifold_base.glb"
    download_glb(glb_url, output_path)

    return {
        "method": "text-to-3d",
        "task_id": task_id,
        "path": str(output_path),
        "size_kb": output_path.stat().st_size // 1024,
    }


def generate_eeg_helmet(photo_path: str) -> dict:
    """Generate EEG helmet via image-to-3D."""
    print(f"\n[2/2] Generating EEG helmet (image-to-3D) from {photo_path}...")

    with open(photo_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "mode": "preview",
        "image_url": f"data:image/jpeg;base64,{image_b64}",
        "topology": "quad",
        "target_polycount": 15000,
        "should_remesh": True,
    }

    r = requests.post(f"{BASE}/image-to-3d", headers=HEADERS, json=payload)
    r.raise_for_status()
    task_id = r.json()["result"]
    print(f"  Task ID: {task_id}")

    result = poll_task(task_id, "image-to-3d")
    glb_url = result["model_urls"]["glb"]
    output_path = ASSETS_DIR / "eeg_helmet.glb"
    download_glb(glb_url, output_path)

    return {
        "method": "image-to-3d",
        "source_photo": photo_path,
        "task_id": task_id,
        "path": str(output_path),
        "size_kb": output_path.stat().st_size // 1024,
    }


def generate_fallback_manifold() -> dict:
    """Generate saddle surface z = x^2 - y^2 as GLB using trimesh."""
    print("\n[FALLBACK] Generating parametric manifold surface...")

    try:
        import trimesh
        import numpy as np
    except ImportError:
        print("  ERROR: trimesh/numpy not installed")
        print("  Run: pip install trimesh numpy")
        return None

    # Create saddle surface grid
    x = np.linspace(-2, 2, 60)
    y = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(x, y)
    Z = 0.3 * (X**2 - Y**2)  # saddle: z = x^2 - y^2

    # Build mesh from grid
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Create faces (two triangles per grid cell)
    n = len(x)
    faces = []
    for i in range(n-1):
        for j in range(n-1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = (i+1) * n + j
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    output_path = ASSETS_DIR / "manifold_base.glb"
    mesh.export(str(output_path))

    print(f"  Generated: {output_path} ({output_path.stat().st_size // 1024} KB)")

    return {
        "method": "trimesh-fallback",
        "path": str(output_path),
        "size_kb": output_path.stat().st_size // 1024,
    }


def find_eeg_photo() -> str:
    """Find EEG hardware photo."""
    candidates = [
        "20260124_180635.jpg",
        "20260124_180653.jpg",
        "assets/20260124_180635.jpg",
        "assets/20260124_180653.jpg",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    if not API_KEY:
        print("ERROR: MESHY_API_KEY environment variable not set")
        sys.exit(1)

    # Create assets directory
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meshy_api_version": "v2",
        "assets": {}
    }

    # Generate manifold surface
    try:
        metadata["assets"]["manifold_base"] = generate_manifold_surface()
    except Exception as e:
        print(f"  ERROR: Meshy text-to-3D failed: {e}")
        print("  Falling back to parametric generation...")
        result = generate_fallback_manifold()
        if result:
            metadata["assets"]["manifold_base"] = result

    # Generate EEG helmet (if photo available)
    photo_path = find_eeg_photo()
    if photo_path:
        try:
            metadata["assets"]["eeg_helmet"] = generate_eeg_helmet(photo_path)
        except Exception as e:
            print(f"  ERROR: Meshy image-to-3D failed: {e}")
            metadata["assets"]["eeg_helmet"] = {"skipped": True, "reason": str(e)}
    else:
        print("\n[2/2] Skipping EEG helmet — no photo found")
        metadata["assets"]["eeg_helmet"] = {"skipped": True, "reason": "no photo found"}

    # Save metadata
    metadata_path = ASSETS_DIR / "meshy_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {metadata_path}")

    # Summary
    print("\n" + "=" * 50)
    print("C-004 COMPLETE")
    print("=" * 50)
    for name, info in metadata["assets"].items():
        if info.get("skipped"):
            print(f"  {name:20s}: SKIPPED — {info.get('reason')}")
        else:
            print(f"  {name:20s}: {info.get('size_kb', '?')} KB — {info.get('method')}")
    print(f"  Output directory   : {ASSETS_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
