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
# 3b. Multi-fidelity (section 5). The high-fidelity CSV shares sample ids with
#     the low-fidelity sweep, so keep it out of the data/raw glob of step 2
python src/datasets/preprocess.py --input <hifi.csv> \
    --steady-output data/processed/wing_steady_hifi.pt --output data/processed/wing_dataset_hifi.pt
python src/training/train_steady.py --data data/processed/wing_steady_hifi.pt --targets CL CD \
    --base checkpoints/wing_steady_bern.pt --output checkpoints/wing_steady_mf.pt
python src/training/train_steady.py --data data/processed/wing_steady_hifi.pt --targets Cm \
    --output checkpoints/wing_steady_cm.pt
python src/inference/eval_steady.py checkpoints/wing_steady_bern.pt checkpoints/wing_steady_mf.pt \
    checkpoints/wing_steady_cm.pt --data data/processed/wing_steady_hifi.pt
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
`output_bounds` rejects boxes outside the input box. The forward pass does not
check its inputs, and it extrapolates the polynomials outside the box, so
callers must keep queries inside `DESIGN_BOX` (`preprocess.py` enforces this for
the data).

## 2. Simulation setup (`data/collection/wing_sim.jl`)

FLOWUnsteady 3.4 (master), FLOWVLM 2.1.4 and FLOWVPM 4.0.3, as pinned in
`data/collection/Manifest.toml`.

**High-fidelity preset** (used by the sweep), following FLOWUnsteady's
high-fidelity PROWIM example. That example replicates Alvarez & Ning (2023,
J. Aircraft, doi:10.2514/1.C037279), whose wing is untapered and unswept,
with two propellers. The rVPM formulation and its SFS model come from
Alvarez & Ning (2023, AIAA J., doi:10.2514/1.J063045).

- actuator surface model (vortex sheet, `g_pressure` distribution)
- dynamic SFS LES model with backscatter clipping, and RK3 integration
- 5 particle sheds per step, λ = 2.125
- 100 elements per semi-span; the Alvarez (2022) wing convergence study shows
  loads converge to within 1% from n ≈ 100
- the wake is simulated until it is 2.75 spans long, in 200 steps

The low-fidelity preset is the Weber wing example (actuator line, no SFS) and
is only meant for smoke tests.

**Particle budget.** The vortex sheet adds static particles to the particle
field for the duration of every step. There are about 2.125·c/σ_TBV per
trailing bound vortex, with σ_TBV = 0.12·c_tip/128 as in PROWIM (FLOWVLM's
b/ar is the tip chord). PROWIM preallocates 10⁶ static particles, which fits
its rectangular wing. Exact counts for this preset (FLOWUnsteady's own
`_static_particles` on the generated wings):

| tr                | 1.0         | 0.8   | 0.7   | 0.5   | 0.3         |
|-------------------|-------------|-------|-------|-------|-------------|
| static particles  | 0.91–0.93 M | 1.01 M | 1.07 M | 1.26 M | 1.69–1.72 M |

With the fixed 10⁶ budget, every wing with tr < 0.8 (about 70% of the
sweep) would stop with `PARTICLE OVERFLOW`. For tr ≤ 0.5 that happens at the
first step. `run_wing` therefore sizes the budget per wing from the root
chord. These static particles take part in every evaluation of the particle
field, which is the likely source of most of the per-step cost.

**FLOWVPM 4 compatibility shim.** FLOWUnsteady 3.4 passes `index=` to
`add_particle` when it builds a wing's vortex sheet, and FLOWVPM 4 removed that
keyword, so every high-fidelity wing run failed with a `MethodError`.
`wing_sim.jl` adds a method that drops the keyword. The index is only read by
the "averaged"/"weighted" Kutta–Joukowski force types; this project uses the
"regular" one.

**Wake treatment.** Sweep sample 3 (tr = 0.30, Λ = 32.8°, Γ = 8.3°,
AOA = 7.9°, V = 77.9 m/s) crashed after 3 h 15 min inside FLOWVPM's FMM. The
error was `ArgumentError: not a bracketing interval` from `solve_ρ_over_σ`,
the regularization-error autotuning. That root solve has a residual of −1 at
0, so it can only fail when a particle's strength is non-finite or its core
size is ≤ 0. rVPM shrinks a particle's core under stretching. PROWIM's
high-fidelity preset runs without wake treatment.

A first fix removed only blown-up particles: strength above 10× that of a
CL = 2 bound vortex shed over one substep, the upper bound of PROWIM's
lower-fidelity presets, with c_root as the chord. Sample 3 then removed
particles at 11 steps between 35 and 84, and crashed within step 85.

A per-step diagnostic of the same design located the problem.

- The smallest core shrank steadily, from 0.68 σ_vpm at step 4 to 0.09 at
  step 24 and 0.018 at step 32.
- These particles sat just behind the root trailing edge and carried
  ~10⁻⁶ of the strength bound.
- At step 35 particles in the starting vortex, 1.2 m downstream, blew up: one
  core went negative and a neighbour reached 1.9× the strength bound.

`run_wing` therefore also removes, after every step, particles whose core
size is outside [0.1, 5] σ_vpm. These are the nearly singular and negligibly
smeared particles, with the bounds of FLOWUnsteady's Vahana example. The
strength bound is kept. Over the first 8 steps of sample 3, starting vortex
included, the strongest particle is 0.10 of it. Each sample's log line
reports the particles removed for strength and for size, and the peak
strength kept relative to the bound. Samples 1 and 2 ran before any wake
treatment.

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

High-fidelity preset at AOA = 4.2°, V = 49.7 m/s, ρ = 0.93 kg/m³, RAE 101
polar, no skin friction (as in FLOWUnsteady's example), 200 steps:

| quantity | VPM (high) | experiment | error |
|----------|------------|------------|-------|
| CL       | 0.2325     | 0.238      | −2.3% |
| CD       | 0.00478    | 0.005      | −4.4% |
| Cm (MAC c/4, nose-up +) | +0.018 | n/a | |

The transient is converged: over the last 10% of the steps, the CL standard
deviation is below 1e-4 and CL drifts by 1e-4 from the preceding 10%. If the
run stops when the wake is 2.0 spans long instead of 2.75, CL is 0.2320
(−0.2%).

Cm has no experimental check here. FLOWUnsteady's example reports the moment
about the root quarter chord and plots no measurements. A Cm of +0.018 puts the
centre of pressure 0.076 MAC ahead of the MAC quarter chord. That lies between
a uniform span loading (0) and an elliptic one (about 0.19 MAC ahead) on this
45° wing, so it is plausible.

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

The result holds across seeds and degrees:

- **Degree 8, 3 seeds:** CL R² 0.99966–0.99983, CD R² 0.99972–0.99987.
- **Degree 4:** CL R² 0.99967, CD R² 0.99950.
- **Degree 12:** CL R² 0.99986, CD R² 0.99977.

Every variant beats both baselines.

**Bern-IBP.** Over 256 random sub-boxes × 1000 samples, no sampled output
fell outside its bounds. The bounds are tight on small boxes but loose on the
whole design box:

| box                              | CL bound width / CL data range | CD bound width / CD data range |
|----------------------------------|------|------|
| random sub-boxes, ≤ ½ side, mean | 0.46 | 0.55 |
| whole design box                 | 3.4  | 4.8  |

Phase 2 reachability will therefore need input-space splitting (branch and
bound) to get useful bounds.

## 5. Multi-fidelity surrogate

Running all 500 designs at high fidelity would take about 5–6 weeks (section
6), so the final surrogate combines the two sweeps:

- **CL, CD:** the low-fidelity BernMLP (500 designs) plus a BernMLP
  correction trained on the high-fidelity residual y − base(x)
  (`train_steady.py --base`). Both networks are defined on the same input
  box, so the Bern-IBP bounds of their sum are the sum of their bounds
  (`tests/test_train_steady.py`).
- **Cm:** a separate BernMLP trained on the high-fidelity data only
  (`--targets Cm`), since the low-fidelity data has no valid Cm.

The high-fidelity sweep uses the same LHS (seed 42) as the low-fidelity one,
so each high-fidelity sample is paired with the low-fidelity sample of the
same `sample_id`. The split is drawn per `sample_id`, so a high-fidelity test
design is also held out of the base model's training data. The sweep runs the
samples in random LHS order, so any prefix is a uniform subset of the design
box.

**First pairs** (steady values, planform reference area):

| id | design (AOA, ar, tr, Λ, twist) | CL hi | CL lo | ΔCL | CD hi | CD lo | ΔCD | Cm hi |
|----|------|-------|-------|-----|-------|-------|-----|-------|
| 1 | 5.6°, 8.0, 0.64, 23°, −2.8° | 0.3410 | 0.3433 | −0.7% | 0.01435 | 0.01429 | +0.4% | +0.0181 |
| 2 | 1.2°, 7.2, 0.98, 3°, +2.7° | 0.1811 | 0.1836 | −1.4% | 0.00968 | 0.00969 | −0.1% | +0.0000 |

**Synthetic check of the training setup.** To exercise the pipeline before
enough high-fidelity data exists, a stand-in was built from the first 100
low-fidelity designs, with the low-fidelity values distorted by a smooth,
design-dependent correction (a few % in CL and CD) and a made-up smooth Cm.
The split was 83 train, 9 val and 8 test designs. Test-split scores:

| model                                  | CL R²   | CL MAE | CD R²   | CD MAE | Cm R²  |
|----------------------------------------|---------|--------|---------|--------|--------|
| low-fidelity BernMLP only              | 0.99877 | 6.9e-3 | 0.99836 | 5.3e-4 | –      |
| high-fidelity only (64, 64), d8        | 0.99795 | 1.0e-2 | 0.99736 | 5.3e-4 | 0.9952 |
| base + correction (16, 16), d4         | 0.99993 | 1.6e-3 | 0.99989 | 1.2e-4 | –      |
| Cm only, high fidelity (16, 16), d4    | –       | –      | –       | –      | 0.9915 |

The correction cuts the CL/CD error 3–8× compared with either single-fidelity
model. Learning Cm inside the correction network was also tried. It was worse
on the validation designs (Cm R² 0.93 against 0.98–0.99). Early stopping on
the small, quickly fit CL/CD correction stops training before Cm is fit, so
Cm gets its own model. The real comparison on high-fidelity test designs is
pending until enough samples exist.

## 6. Limitations

- **The parasitic drag polar is fixed.** It is NACA 0012 at Re = 5e5. The VPM
  is inviscid, so the coefficients are almost independent of `magVinf`. In the
  trained surrogate, varying `magVinf` over its full range changes CL by 0.003
  on average, against 0.78 for AOA. `magVinf` could be dropped as an input,
  or the polar made Reynolds-dependent (only a few discrete polars are
  available).
- **High-fidelity cost.** The Weber run (tr = 1, about 0.92 M static
  particles, up to 0.21 M wake particles) took 2 h 01 min on 16 threads, about
  36 s per step. The first two sweep samples (tr = 0.64 and 0.98) took 1 h
  45 min and 1 h 37 min. At about 13 samples a day, 500 would take 5–6 weeks
  on one machine; tapered wings, with up to 1.9× the static particles, are
  slower. Most of the cost is the static particles. Rerunning the Weber case with only the
  vortex-sheet overlap reduced to 2.125/10 (PROWIM's mid-fidelity value, so
  10× fewer static particles) took 34 min, 3.5× faster. CL was 0.2323 against
  0.2325, CD and Cm were unchanged, and CL differed by at most 4e-4 over the
  whole transient. Only this untapered wing has been compared.
- **No stall.** CL comes from the lattice, so it stays linear up to 12°. The
  polar only adds parasitic drag.
- **Legacy low-fidelity data has no valid Cm.** Cm is only available from the
  high-fidelity sweep.
