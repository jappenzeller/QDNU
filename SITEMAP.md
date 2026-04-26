# qdnu.ai — Site Map (Post-Migration, April 2026)

## Navigation (nav.html)
`QDNU | Framework | Instrument | Invariants | Findings | Domains | Hardware | Math ↗ | Viz ↗ | Roadmap`

---

## Spine Pages

| Path | Title | Description |
|------|-------|-------------|
| `/` | QDNU — QNFM | Hub: thesis hero, 6 entry-point cards, status board, roadmap teaser |
| `/framework/` | Framework | QNFM encoding pipeline, live field mapping, invariant hierarchy cards |
| `/instrument/` | Instrument | A-Gate circuit architecture, gate diagrams, IBM Heron topology |
| `/invariants/` | Invariants | 4-scene interactive: polarity (confirmed), cross-freq + trajectory (predicted) |
| `/findings/` | Findings | 3 paper cards with status badges |
| `/findings/paper-1/` | Paper 1 | Polarity discovery narrative (Problem → Discovery → Simulation → Results) |
| `/findings/paper-2/` | Paper 2 | Continuous polarity, hardware scaling, pre-registration outcomes |
| `/findings/paper-3/` | Paper 3 | QNFM theoretical framework scoping (drafting) |
| `/domains/` | Domains | 3 domain cards: EEG (validated), FX (theoretical), ECG (planned) |
| `/domains/eeg/` | EEG Domain | CHB-MIT dataset, hardware subset rationale, Mark IV plans |
| `/domains/fx/` | FX Domain | E-Gate architecture, factor basis, Nash equilibrium proximity |
| `/hardware/` | Hardware | Hub: Mark IV, Lattice Configuration, planned tools |
| `/hardware/mark-iv/` | Mark IV | 18-channel EEG headset specs, design iterations, status |
| `/hardware/lattice-configuration/` | Lattice Config | Interactive 3D electrode positioning tool |
| `/roadmap/` | Roadmap | Status board + 4-column timeline (Completed → Speculative) |

## Supporting Pages

| Path | Title | Description |
|------|-------|-------------|
| `/math/` | Math Concepts | 7-module curriculum: Foundations → Geometry → Quantum |
| `/math/eigenvalues/` | Eigenvalues | Interactive eigenvector visualization |
| `/math/covariance-matrices/` | Covariance | EEG channel correlation structure |
| `/math/spd-matrices/` | SPD Matrices | Positive-definiteness, quadratic forms |
| `/math/plv/` | PLV | Phase synchrony → rotation angles |
| `/math/riemannian-geometry/` | Riemannian Geometry | Geodesics, tangent spaces, metric tensors |
| `/math/bures-manifold/` | Bures Manifold | SPD↔Bures identity, 3D trajectory |
| `/math/quantum-states/` | Quantum States | Bloch sphere, superposition, measurement |
| `/viz/` | Visualization Hub | Main Three.js visualization |
| `/viz/agate-the-shape-of-a-seizure.html` | Shape of a Seizure | SPD manifold, geodesics, polarity |
| `/viz/agate-boundary-crossings.html` | Boundary Crossings | Julia set fractal transitions |
| `/qnfm/` | QNFM (legacy) | Original QNFM page, still live, dynamic nav |

## Redirects (meta-refresh)

| Old Path | New Path |
|----------|----------|
| `/math/a-gate-architecture/` | `/instrument/` |
| `/math/invariants/` | `/invariants/` |

## Components

| Path | Type | Description |
|------|------|-------------|
| `/nav.html` | Shared | Dynamic nav component fetched by all pages |
| `/components/status-board.json` | Data | JSON driving status board rendering |
| `/components/status-board.html` | Component | Fetches JSON, renders card grid |
| `/components/status-board.css` | Styles | Badge colors per status |
| `/math/shared/math-styles.css` | Styles | Shared math page styles |
| `/math/shared/math-utils.js` | Script | Shared math utilities |
| `/math/bures-manifold/manifold_trajectory.js` | Script | 3D trajectory animation |

## Data Assets

| Path | Description |
|------|-------------|
| `/visualization_data/circuit_gates.json` | Gate definitions |
| `/visualization_data/circuit_trajectories.json` | Manifold trajectory data |
| `/visualization_data/heron_topology.json` | IBM Heron processor topology |
| `/visualization_data/assets/meshy_metadata.json` | 3D mesh metadata |

## Navigation Pattern
All pages use dynamic `fetch('/nav.html')` — no inline nav duplication.
