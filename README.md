# PINNNMEoS

Physics-informed neural-network solution maps for the FSU-delta relativistic
mean-field equations of state. Authors: Chen Zhang, Zihan Ren, and Lin Yuan.

This repository implements two separate stages: residual-only training at a
fixed optimized interaction, and conventional-solver nuclear/stellar screening.
Nuclear and astrophysical constraints are **not** PINN training losses.

## Important Reproducibility Notice

The default code follows the author-approved manuscript supplied on 2026-11-04:
original Table I parameter normalization, the **sum** of four batch-mean losses,
and truncated Gaussian proposals with sigma = 0.40 times the absolute original
reference value. These differ from the historical programs that generated the
reported training log and the 402/296 ensemble. Those counts, errors, timings,
and losses must not be advertised as reproduced by this configuration until a
new full run is completed. No full training or 432000-proposal run is performed
when installing this repository.

See [the configuration audit](docs/MANUSCRIPT_ALIGNMENT.md) for exact changes,
remaining manuscript ambiguities, and inherited numerical choices.

## Install

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate`. Install a PyTorch build
appropriate to your GPU before the other dependencies if CUDA is required.
CPU operation is supported. CUDA is needed for GPU training, not NumPy inference.
The historical measurement used PyTorch 2.3.1+cu118, NumPy 1.24.3, and SciPy
1.10.1; `requirements.txt` gives compatible ranges, not an exact historical lock.

## Train

```bash
python scripts/train_fsu_delta_pinn_fields_variable10.py --outdir outputs/train --device cuda
```

Defaults: 200000 updates, batch 8192, six hidden layers of width 256, SiLU,
AdamW, cosine learning rate 2e-4 to 1e-5. Only network weights are trained.
Outputs include every-step component losses, learning rate, sampled host RSS,
GPU allocated/reserved peaks, metadata, checkpoints, and a final summary.
Sampled RSS is not a measurement of the OS peak working set. GPU tensor-memory
metrics are not total GPU-process memory. No vacuum or observational loss is used.

For a quick CPU execution check (not an accuracy test):

```bash
python scripts/train_fsu_delta_pinn_fields_variable10.py --device cpu --steps 3 --batch 16 --outdir outputs/smoke
```

## Export and Benchmark

Only load trusted `.pt` checkpoints. Export once, outside inference timing:

```bash
python scripts/export_checkpoint.py --checkpoint outputs/train/latest.pt --output outputs/model.npz
python scripts/benchmark_field_solver_numpy.py --checkpoint outputs/model.npz --output outputs/benchmark.json --repeats 31
```

The portable NPZ stores weights and input normalization explicitly. The benchmark
compares prepared-batch NumPy evaluation against density-continuation SciPy
`fsolve` on a 100 x 10 density/composition grid. Numerical-library threads are
limited to one; no physical-core affinity is imposed. Both methods are warmed
up and repeated 31 times. Loading, training, EOS construction, crust matching,
and TOV integration are excluded. Per-point batch time is total time / 1000,
not independent single-query latency. Failed conventional solves abort the
speedup report rather than counting as successful fast solves.

Pointwise field and baryonic thermodynamic validation at the same densities
and compositions (exact root solves initialized by the PINN) is available via:

```bash
python scripts/validate_fields.py --checkpoint outputs/model.npz --output outputs/validation.json
```

It saves predictions, corrected roots, residual checks, and absolute/relative
error statistics at 256 fresh random points. This is not a replay of the exact
historical validation sample; see the output's explicit seed and denominator
floor. Beta-equilibrium stellar errors require a separate EOS/TOV convergence
audit and are not inferred from this local-matter test.

## Generate R1

```bash
python scripts/wide_r1r2_20261103.py --output outputs/ensemble --bps vendor/rmf/TOVtools/bps.dat --count 432000 --workers 8
```

Choose workers for your own machine. Each process is a worker, not necessarily
a distinct physical CPU core. A large scan is expensive. A smoke probe can use
`--count 2 --workers 1 --only-nuclear` with a fresh output directory.

All ten active parameters vary jointly. Nuclear intervals are n0 = 0.11-0.21,
binding energy = -17 to -15 MeV, K = 200-260 MeV, Esym = 27.7-34.1 MeV, and
L = 35-70 MeV. The Dirac-effective-mass interval is contextual, not an extra cut.
R1 requires beta equilibrium to 1e-4 MeV, positive increasing core pressure and
increasing energy density on 400 densities from 0.08 to 1.4 fm^-3, and complete
stable stellar sequences. Rejected calculations remain recorded in JSONL.

The BPS-based crust is attached at 0.08 fm^-3 by a fixed-density prescription,
not a Maxwell construction. The stellar solver uses derivative-consistent
PCHIP interpolation and coupled TOV-Love integration (rtol 2e-7, atol 1e-12,
max step 0.03 km, 100 initial central-pressure nodes plus refinement).

The inherited truncation bounds are exported in each run manifest; they are
listed in `docs/MANUSCRIPT_ALIGNMENT.md` because the manuscript omits them.
Existing outputs cannot be resumed with a changed configuration.

## Select R2

The processed observational grids and GW table used by the local analysis are
included. Their upstream sources and limitations are described in
[OBSERVATIONS.md](docs/OBSERVATIONS.md); they are not relicensed as original code.

```bash
python scripts/select_r2.py --bank outputs/ensemble --observations data/observations
```

R2 intersects **each** source's separate nominal 95% region: J0740 and J0437
joint mass-radius data and GW170817 joint binary mass-tidal data. J0030 is
comparison-only. There is no additional 1.97-solar-mass cut. J0740 radio-mass
information is not applied twice. This is an approximate contour-compatibility
diagnostic, not a combined 95% region, an observational likelihood, or a
Bayesian EOS posterior. The exact settings and 68/90/95% counts are saved.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests check input scaling, sampling support, loss reduction, field bounds,
residual agreement, truncated proposals, and NumPy/PyTorch export agreement.
They do not replace full training, stellar convergence tests, or posterior
data validation.

## Repository Layout

- `scripts/`: training, portable inference, conventional RMF/EOS/TOV and screening.
- `configs/`: machine-readable manuscript-setting summary.
- `vendor/rmf/`: the local RMF toolkit snapshot needed by the exact solver.
- `tests/`: deterministic unit tests and a small execution check in CI.
- `docs/`: configuration changes, observational inputs, and release boundaries.
- `reference_results/`: historical training summary/milestones and stellar audit;
  no historical numbers relabeled as reruns.
- `outputs/`: generated locally, excluded from Git.

## Scientific Scope

The released network is a reference-point density-composition solver, not a
validated parameter-hypercube surrogate. Smooth outputs do not guarantee root
uniqueness, physical branch selection, or positive effective masses. The model
contains nucleons and electrons; muons, quarks, dark matter, and mixed phases
are not implemented. EOS validity checks are not a complete proof of causality
or thermodynamic stability. Timing depends on hardware, precision and workload.

## Citation and Feedback

Please cite the accompanying CU10971 manuscript and the eventual versioned
software release. No final journal DOI is assigned in this package. Report
reproduction issues through https://github.com/jjsl-yl/PINNNMEoS/issues with the
command, environment, configuration and error log. Review the bundled-code
provenance and data permissions before making a public release.
