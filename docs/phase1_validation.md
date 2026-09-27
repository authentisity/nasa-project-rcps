# Phase 1: surrogate model — implementation and validation notes

Phase 1 produces a steady-state surrogate of the aerodynamics of an isolated
wing: 7 design/flight inputs → steady CL, CD, Cm. The model is a DeepBern-Net
(Khedr & Shoukry, arXiv:2305.13508), so Phase 2 can bound its outputs over
input boxes with Bern-IBP and run backward reachability. Training data comes
from FLOWUnsteady (Alvarez & Ning) unsteady VPM simulations.

| Input       | Range        | Meaning                                   |
|-------------|--------------|-------------------------------------------|
| `AOA`       | 0 – 12°      | angle of attack                           |
| `ar`        | 3 – 10       | b / c_tip (FLOWVLM `simpleWing` convention, not b²/S) |
| `tr`        | 0.3 – 1      | taper ratio c_tip / c_root                |
| `lambda`    | 0 – 50°      | leading-edge sweep                        |
| `gamma`     | −5 – 10°     | dihedral                                  |
| `twist_tip` | −5 – 5°      | tip twist (root 0°, linear)               |
| `magVinf`   | 20 – 80 m/s  | freestream speed (sea level)              |

Span b = 2.489 m is fixed. The surrogate takes the inputs scaled to [0, 1]⁷
(`DESIGN_BOX` in `src/datasets/preprocess.py`). Its Bernstein layers are
defined only on that box, so the box is also the domain for reachability.

## Pipeline

```bash
# 1. Simulate (data/collection, Julia 1.10); resumable, shardable
cd data/collection && julia -t 16 --project=. wing_timeseries_sweep.jl
# 2. Steady-state + time-series datasets, shared train/val/test split
python src/datasets/preprocess.py --input data/raw/wing_timeseries_data*.csv
# 3. Surrogate (BernMLP) and baselines
python src/training/train_steady.py                      # -> checkpoints/wing_steady_bern.pt
python src/training/train_steady.py --arch relu --output checkpoints/wing_steady_relu.pt
python src/training/train.py                             # WingLSTM on the transients
# 4. Held-out metrics and Bern-IBP soundness
python src/inference/eval_steady.py checkpoints/wing_steady_bern.pt checkpoints/wing_steady_relu.pt \
    --lstm checkpoints/wing_lstm.pt
# Unit tests
python -m unittest discover tests
```

## 1. DeepBern-Net implementation

`src/models/layers/bernstein.py` is vendored from the reference implementation
(BSD-3). `src/models/bern_net.py` adds the fully connected network `BernMLP`.
Each property from the paper is checked in `tests/test_bernstein.py` and
`tests/test_bern_net.py`:

| Paper                                        | Check |
|----------------------------------------------|-------|
| Bernstein activation, eq. (1)–(3)            | forward equals Σ c_k C(n,k)(x−l)^k(u−x)^(n−k)/(u−l)^n |
| Partition of unity, non-negativity           | degree-40 basis sums to 1 |
| Range enclosure (Prop. 1)                    | min/max coefficient encloses a dense sampling of the polynomial |
| Subdivision (Prop. 2)                        | de Casteljau split reproduces the polynomial exactly on both halves |
| Bern-IBP on sub-intervals                    | sound on random sub-intervals, including degenerate ones; exact for point intervals |
| Algorithm 1 (layer input bounds from the input box) | after training steps, every reachable pre-activation lies inside its layer's stored interval |
| Network-level Bern-IBP                       | on random sub-boxes (points, faces, the whole box), sampled outputs are inside the bounds; bounds shrink with the box |

Fixes to the vendored layer:

- **`subinterval_bounds` returned NaN on intervals starting at the lower
  bound.** It split at β and then at α/β, which divides by zero when β = 0.
  The fix picks, per neuron, the split order whose denominator is ≥ ½.
- **Binomial coefficients were computed with float32 `lgamma`.** That is off
  by about 1e-6 relative at degree 40, which tripped the layer's own
  partition-of-unity check. They are now computed in float64 (exact through
  degree 40) and cast to the input dtype, so float32 models stay float32.

`BernMLP` recomputes the layer input intervals (Algorithm 1) on every
training-mode forward and on `eval()`. A checkpoint saved in eval mode
therefore stores intervals that belong to its weights. The reference code
does the same refresh with a dummy forward after each optimizer step.
Queries outside the input box raise an error instead of silently extrapolating
the polynomials.

## 2. Simulation setup (`data/collection/wing_sim.jl`)

FLOWUnsteady 3.4 (master), FLOWVLM 2.1.4 and FLOWVPM 4.0.3, as pinned in
`data/collection/Manifest.toml`.

**High-fidelity preset** (used by the sweep), following FLOWUnsteady's
high-fidelity PROWIM example (Alvarez & Ning 2023):

- actuator surface model (vortex sheet, `g_pressure` distribution)
- dynamic SFS LES model with backscatter clipping, and RK3 integration
- 5 particle sheds per step, λ = 2.125
- 100 elements per semi-span; the Alvarez (2022) wing convergence study shows
  loads converge to within 1% from n ≈ 100
- the wake is simulated until it is 2.75 spans long, in 200 steps

The low-fidelity preset is the Weber wing example (actuator line, no SFS) and
is only meant for smoke tests.

**FLOWVPM 4 compatibility shim.** FLOWUnsteady 3.4 passes `index=` to
`add_particle` when it builds a wing's vortex sheet, and FLOWVPM 4 removed that
keyword, so every high-fidelity wing run failed with a `MethodError`.
`wing_sim.jl` adds a method that drops the keyword. The index is only read by
the "averaged"/"weighted" Kutta–Joukowski force types; this project uses the
"regular" one.

**Reference quantities (fixed).**

- **CL and CD** now use the projected planform area S = b(c_root + c_tip)/2.
  The original script used b²/ar = b·c_tip, the tip-chord rectangle, which
  overstated the coefficients by the factor (1 + tr)/(2·tr), or 2.17 at
  tr = 0.3.
- **Cm** is now taken about the quarter chord of the mean aerodynamic chord
  (MAC), nose-up positive, and normalized by q·S·MAC. The moment arm of each
  element's force is the midpoint of its lifting bound vortex, which is where
  the regular KJ force acts. The original script used the control points and
  the root quarter chord.

`data/collection/test_planform.jl` checks S, MAC and the MAC's leading-edge
position against numerical integration of the generated FLOWVLM geometry for
four planforms (16/16 pass).

Legacy sweeps (no `mac` column) are converted by `preprocess.py`: CL and CD are
rescaled by (b²/ar)/S and Cm is dropped.

## 3. Physics checks

### Lift-curve slope against DATCOM (low fidelity, 500 legacy samples)

The reference is the DATCOM/Helmbold lift slope,

CL_α = 2πA / (2 + √(A²(1 + tan²Λ_c/2) + 4)),

with the true aspect ratio A = b²/S and the half-chord sweep Λ_c/2. κ = 1
because the VLM surface is thin, and M ≈ 0.

The simulated slope was taken as CL / [(α + θ_eff)·cos²Γ]. Here θ_eff is the
chord-weighted mean of the linear twist, and only AOA > 2° is used.

| subset          | n   | CL_α,VPM / CL_α,DATCOM (median, IQR) | r     |
|-----------------|-----|--------------------------------------|-------|
| \|twist\| < 1°  | 87  | 0.972 (0.952 – 0.987)                | 0.988 |
| \|twist\| < 2°  | 169 | 0.969 (0.943 – 0.987)                | 0.975 |

The simulated slopes are about 3% below DATCOM and track its variation across
planforms. That gap is comparable to the accuracy of the DATCOM estimate
itself. The spread widens with twist because the θ_eff model is crude.

### Weber & Brebner 45° swept wing (`validate_weber.jl`)

The experiment is ARC R&M 2882, the same case as FLOWUnsteady's own
validation: A = 5, untapered, 45° sweep, RAE 101 section.

*Pending: the high-fidelity result at AOA = 4.2° (experiment CL = 0.238) is
being computed.*

### Convergence of the transients

The steady-state value is the mean over the last 10% of the steps, and the
drift is its change from the preceding 10%. In the low-fidelity data the
median drift is 4e-4 for CL and 1e-5 for CD. Only 1 of 500 samples drifts by
more than 1%, in CL. The high-fidelity sweep records the same diagnostics
(`drift` and `tail_std` in `wing_steady.pt`).

## 4. Surrogate accuracy (low-fidelity legacy data, CL/CD)

This data was used to develop the pipeline while the high-fidelity sweep runs.
It has 500 LHS samples at 198 steps each, with CL/CD converted to the
planform area. Scores are on the 48 held-out test designs, in physical units:

| model                       | CL R²    | CL MAE  | CL max err | CD R²    | CD MAE  | CD max err |
|-----------------------------|----------|---------|------------|----------|---------|------------|
| BernMLP (64, 64), degree 8  | 0.99983  | 2.4e-3  | 9.7e-3     | 0.99987  | 1.4e-4  | 4.7e-4     |
| ReLU MLP (64, 64)           | 0.99896  | 6.1e-3  | 2.9e-2     | 0.99864  | 4.2e-4  | 1.3e-3     |
| WingLSTM (tail mean)        | 0.99938  | 4.0e-3  | 2.9e-2     | 0.99960  | 2.1e-4  | 1.3e-3     |

Settings: full-batch AdamW with cosine decay, 5000 epochs, lr 3e-3, float64.
The checkpoint with the lowest validation loss is kept.

**Bern-IBP.** Over 256 random sub-boxes × 1000 samples, no sampled output
fell outside its bounds. The bounds are tight on small boxes but loose on the
whole design box:

| box                              | CL bound width / CL data range | CD bound width / CD data range |
|----------------------------------|------|------|
| random sub-boxes, ≤ ½ side, mean | 0.46 | 0.55 |
| whole design box                 | 3.4  | 4.8  |

Phase 2 reachability will therefore need input-space splitting (branch and
bound) to get useful bounds.

## 5. Limitations

- **The parasitic drag polar is fixed.** It is NACA 0012 at Re = 5e5. The VPM
  is inviscid, so the coefficients are almost independent of `magVinf`. In the
  trained surrogate, varying `magVinf` over its full range changes CL by 0.003
  on average, against 0.78 for AOA. `magVinf` could be dropped as an input,
  or the polar made Reynolds-dependent (only a few discrete polars are
  available).
- **High-fidelity cost.** *Pending the benchmark.*
- **No stall.** CL comes from the lattice, so it stays linear up to 12°. The
  polar only adds parasitic drag.
- **Legacy low-fidelity data has no valid Cm.** Cm is only available from the
  high-fidelity sweep.
