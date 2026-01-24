# Saturation Options: Quick Reference

## The Problem

Your dynamics equation for inhibitory parameter c:
```
dc/dt = +λ_c·c + f(t)·(1-c)
```

The `+λ_c·c` term causes **exponential growth** → overflow without saturation.

## Three Solutions

### Option 1: Hard Clamping (RECOMMENDED)
**What**: Keep original equations, clip after integration
**How**:
```python
dc = dt * (lambda_c * c + f_t * (1 - c))
c = clip(c + dc, 0, 1)
```
**Pros**:
- Simple to implement
- Guaranteed bounds
- Easy to debug
- Can tune λ values freely

**Cons**:
- Discontinuous at boundaries
- Not biologically smooth

**Use when**: You want robustness and simplicity (RECOMMENDED for first implementation)

---

### Option 2: Logistic Saturation
**What**: Modify growth term to self-saturate
**How**:
```python
dc/dt = λ_c·c·(1-c) + f(t)·(1-c)
       └─ logistic ─┘
```
**Pros**:
- Smooth saturation
- Biologically realistic (logistic growth)
- Mathematically elegant

**Cons**:
- Requires tuning λ_c (smaller values needed)
- Slightly more complex

**Use when**: You want biologically plausible smooth dynamics

---

### Option 3: Symmetric Decay
**What**: Change growth to decay (match classical PN)
**How**:
```python
dc/dt = -λ_c·c + f(t)·(1-c)
       └─ decay ─┘
```
**Pros**:
- Matches textbook PN model
- Both a and c have symmetric dynamics
- Naturally stable

**Cons**:
- Loses the "inhibitory buildup" interpretation
- May not capture pre-ictal dynamics as well

**Use when**: You want classical PN behavior without custom dynamics

---

## Comparison

| Aspect | Hard Clamp | Logistic | Symmetric |
|--------|------------|----------|-----------|
| Stability | ★★★★★ | ★★★★☆ | ★★★★★ |
| Biological realism | ★★☆☆☆ | ★★★★★ | ★★★★☆ |
| Simplicity | ★★★★★ | ★★★☆☆ | ★★★★★ |
| Seizure modeling | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| Debug ease | ★★★★★ | ★★★☆☆ | ★★★★★ |

## Recommended Strategy

**Phase 1**: Start with **Hard Clamp**
- Implement everything with `saturation_mode='clamp'`
- Get the full pipeline working
- Validate on synthetic data

**Phase 2**: Experiment with **Logistic**
- Switch to `saturation_mode='logistic'`
- Retrain templates
- Compare performance

**Phase 3**: Benchmark **Symmetric**
- Test classical PN dynamics
- Compare quantum advantage claims

## Parameter Tuning

### Hard Clamp
```python
lambda_a = 0.1    # Higher = faster decay
lambda_c = 0.05   # Lower = slower growth
dt = 0.001        # Smaller = more stable
```

### Logistic
```python
lambda_a = 0.1
lambda_c = 0.02   # SMALLER (growth is amplified by c·(1-c))
dt = 0.001
```

### Symmetric
```python
lambda_a = 0.1
lambda_c = 0.1    # Can match lambda_a for symmetry
dt = 0.001
```

## Code Example: Switching Modes

```python
from pn_dynamics import PNDynamics

# Try all three modes
modes = ['clamp', 'logistic', 'symmetric']

for mode in modes:
    print(f"\n=== Testing {mode} mode ===")

    pn = PNDynamics(
        lambda_a=0.1,
        lambda_c=0.05 if mode != 'logistic' else 0.02,
        dt=0.001,
        saturation_mode=mode
    )

    # Test with strong signal
    signal = np.ones(1000) * 0.8
    a, b, c = pn.evolve_single_channel(signal)

    print(f"Final: a={a:.3f}, b={b:.3f}, c={c:.3f}")
    print(f"Status: {'OK' if c <= 1.0 else 'OVERFLOW'}")
```

## Quick Decision Tree

```
Do you need custom pre-ictal dynamics (inhibitory buildup)?
├─ YES → Use Hard Clamp or Logistic
│  │
│  └─ Do you want smooth saturation?
│     ├─ YES → Logistic
│     └─ NO  → Hard Clamp (simpler)
│
└─ NO → Use Symmetric (classical PN)
```

## Bottom Line

**Start with Hard Clamp** (`saturation_mode='clamp'`). It's the safest, simplest option that prevents overflow while preserving your custom dynamics. You can always switch modes later once the pipeline works.

The implementation in Task 1 includes all three modes so you can experiment.
