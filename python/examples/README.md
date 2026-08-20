# SBD Python Examples

Examples demonstrating SBD's capabilities for quantum chemistry calculations.

## Overview

- **Communication:** MPI for distributed computing
- **Backends:** CPU (OpenMP) and GPU (CUDA), switchable at runtime via `device` parameter

## Extra dependencies (only for the SQD examples)

The standalone `run_sbd_diag.py` script needs nothing beyond what
`pip install -e .` already installed (`sbd`, `mpi4py`, `numpy`).

The SQD examples — `run_sqd_sbd.py` and `run_sqd_sbd.ipynb` — wrap SBD
with the qiskit-addon-sqd self-consistent loop, which pulls in three
extra Python packages. Install them once into the same venv:

```bash
source ~/venvs/<your-sbd-venv>/bin/activate
pip install pyscf qiskit \
    "git+https://github.com/Qiskit/qiskit-addon-sqd@main"
```

- **`pyscf`** — reads FCIDUMP, restores 4-fold integral symmetry.
- **`qiskit`** — `BitArray` type for sampled-bitstring input.
- **`qiskit-addon-sqd`** — the SQD loop (`diagonalize_fermionic_hamiltonian`).
  Requires the **distributed (SPMD) support** that calls `sci_solver` on
  every MPI rank. That now lives in upstream `qiskit-addon-sqd`, but is not
  yet in a PyPI release, so install from `@main` for now. Once a release
  includes it, plain `pip install qiskit-addon-sqd` will suffice.

`pyscf` is the heavy one (~150 MB plus `h5py`). qiskit-addon-sqd is a thin
layer on top of upstream qiskit, so most of `qiskit`'s ~300 MB is what
dominates the install size.

## Examples

### 1. run_sbd_diag.py — Standalone SBD Diagonalization

Runs a single TPB diagonalization from an FCIDUMP file and alpha determinant
file. No SQD loop, no Qiskit dependency.

```bash
# H2O with 2 MPI ranks
mpirun -np 2 python run_sbd_diag.py \
    --device cpu \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --adetfile ../../vendor/sbd-upstream/data/h2o/h2o-1em3-alpha.txt \
    --adet_comm_size 2

# N2 with GPU
mpirun -np 8 python run_sbd_diag.py \
    --device gpu \
    --fcidump ../../vendor/sbd-upstream/data/n2/fcidump.txt \
    --adetfile ../../vendor/sbd-upstream/data/n2/1em3-alpha.txt \
    --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2
```

**Key options:** `--device`, `--fcidump`, `--adetfile`, `--adet_comm_size`,
`--bdet_comm_size`, `--task_comm_size`, `--method`, `--tolerance`, `--iteration`.
Run `python run_sbd_diag.py --help` for the full list.

**Requirements:** `sbd`, `mpi4py`

### 2. run_sqd_sbd.py — SQD Loop with SBD Solver

Runs the self-consistent SQD workflow (qiskit-addon-sqd) using SBD as the
eigensolver backend. Supports two bitstring input modes:

- `--counts FILE` — load hardware bitstrings from a count_dict.json
- `--samples N` — generate N uniform random bitstrings (default)

```bash
# H2O with random samples (default)
mpirun -np 4 python run_sqd_sbd.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --device cpu \
    --adet_comm_size 2 --bdet_comm_size 2

# H2O with your own hardware bitstrings (FCIDUMP from ../../vendor/sbd-upstream/data/h2o/)
mpirun -np 4 python run_sqd_sbd.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --counts /path/to/count_dict.json \
    --device cpu \
    --adet_comm_size 2 --bdet_comm_size 2

# Custom system with hardware bitstrings
mpirun -np 8 python run_sqd_sbd.py \
    --fcidump /path/to/fci_dump.txt \
    --counts /path/to/count_dict.json \
    --samples_per_batch 800 --num_batches 3 --max_iterations 10 \
    --device gpu \
    --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2
```

**count_dict.json format:** A JSON object mapping bitstrings to shot counts, as
produced by a quantum device or simulator. Each bitstring has length `2 × NORB` —
the first `NORB` bits are alpha (spin-up) orbitals and the last `NORB` are beta
(spin-down):

```json
{
  "010000000010001010000001010000000001000010100100": 16,
  "000010001110000000000010001001000110000000000100": 12,
  "000101000000010011000000000010000000001001000110": 8
}
```

Supply your own `count_dict.json` produced by a quantum device or simulator.
The bitstring length must be `2 × NORB` — e.g. for [`../../vendor/sbd-upstream/data/h2o/fcidump.txt`](../../vendor/sbd-upstream/data/h2o/fcidump.txt) that is NORB=24 (5α+5β electrons).

**Key options:** `--fcidump` (required), `--counts`, `--samples`,
`--samples_per_batch`, `--num_batches`, `--max_iterations`, `--device`,
MPI decomposition flags. SBD solver flags (`--method`, `--tolerance`,
`--iteration`, etc.) have sensible defaults; run `python run_sqd_sbd.py --help`
for the full list.

**Requirements:** see [Extra dependencies](#extra-dependencies-only-for-the-sqd-examples) above (`pyscf`, `qiskit`, `qiskit-addon-sqd` fork).

#### SQD Parameter Guide

SQD samples bitstrings from a quantum device, uses **configuration recovery** to
correct noisy samples using an orbital occupancy vector, then subsamples into
batches for diagonalization. Occupancies are averaged across batches and fed back
to configuration recovery — this self-consistent loop typically converges in 3–5
iterations. On the first iteration, no occupancies are available yet, so the raw
samples are simply filtered by correct electron count (Hamming weight
postselection).

| Parameter | What it controls | Typical values |
|-----------|-----------------|----------------|
| `--counts FILE` | Load hardware bitstrings from a JSON file (use one or the other) | 10K–1M+ shots |
| `--samples N` | Generate N uniform random bitstrings for testing (default) | 10K–1M+ |
| `--samples_per_batch` | Subspace dimension per batch (accuracy vs. cost) | 300–800 (small), 1M+ (production) |
| `--num_batches` | Independent subsamples for averaging occupancies | 3–10 (small), up to 100 (large) |
| `--max_iterations` | SQD self-consistent loop iterations (not SBD `--iteration`) | 3–5 |

**MPI work distribution:** All ranks diagonalize each batch together, then move
to the next batch sequentially. Within each diagonalization, ranks form a 3D grid:
`adet_comm_size × bdet_comm_size × task_comm_size = total ranks`. More batches
increases wall time linearly but does not require more ranks.

### 3. run_sqd_sbd.ipynb — Jupyter walkthrough (serial)

Interactive single-rank companion to `run_sqd_sbd.py`. Same SQD self-
consistent loop on h2o, but runs inside a Jupyter kernel
(`MPI.COMM_WORLD` size 1). Uses uniform-random bitstrings + HF
`initial_occupancies` as a self-contained demo. Converges to ~−76.19 Ha
in a few seconds on CPU.

```bash
jupyter nbconvert --to notebook --execute --inplace run_sqd_sbd.ipynb
# or open it in JupyterLab and step through the cells
```

## MPI Decomposition

Total MPI ranks must equal `task_comm_size × adet_comm_size × bdet_comm_size`.

When using more than one rank, specify at least `--adet_comm_size`. Examples:

| Ranks | Decomposition |
|-------|---------------|
| 1 | default (all = 1) |
| 2 | `--adet_comm_size 2` |
| 4 | `--adet_comm_size 2 --bdet_comm_size 2` |
| 8 | `--adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2` |

## Backend Selection

Both backends are loaded at import time. Select per-call via `--device`:

```bash
--device cpu    # OpenMP (default)
--device gpu    # CUDA (requires NVIDIA GPU + HPC SDK build)
--device auto   # GPU if available, else CPU
```

Within Python, backends can also be switched at runtime without re-initialization:

```python
import sbd

# No init() needed — auto-initializes on first call
result_cpu = sbd.tpb_diag(..., device='cpu')
result_gpu = sbd.tpb_diag(..., device='gpu')
```

## Available Test Data

**H2O** (`../../vendor/sbd-upstream/data/h2o/`): `h2o-1em3` through `h2o-1em8` alpha determinant files.
**N2** (`../../vendor/sbd-upstream/data/n2/`): `1em3` through `1em7` and `3em4` through `3em7` alpha determinant files.

Smaller thresholds = more determinants = higher accuracy.

## Expected Results

- **H2O**: ground state energy ≈ **-76.236 Hartree**
- **N2**: ground state energy ≈ **-109.042 Hartree** (with 1e-3 dets)

## Performance Tips

**CPU:** Set `OMP_NUM_THREADS` to cores per MPI rank (e.g., 8 ranks × 4 threads = 32 cores).

**GPU:** One MPI rank per GPU, `OMP_NUM_THREADS=1`. Each rank auto-assigned: `gpu_id = rank % num_gpus`. Use method 0 (matrix-free Davidson) for best GPU performance.

## See Also

- [Python Bindings README](../../README.md) — Installation, API reference
- [Upstream SBD library](https://github.com/r-ccs-cms/sbd) — C++ library overview
