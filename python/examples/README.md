# SBD Python Examples

Examples demonstrating SBD's capabilities for quantum chemistry calculations.

## Overview

- **Communication:** MPI for distributed computing
- **Backends:** CPU (host OpenMP), GPU (NVHPC Thrust) and GPU (OpenMP target offload), switchable at runtime via `device` parameter

## Extra dependencies (only for the SQD examples)

The standalone `run_sbd_diag.py` script needs nothing beyond what
`pip install -e .` already installed (`sbd`, `mpi4py`, `numpy`).

The SQD examples — `run_sqd_sbd.py` and `run_sqd_sbd.ipynb` — wrap SBD
with the qiskit-addon-sqd self-consistent loop, which pulls in three
extra Python packages. Install them once into the same venv:

```bash
source ~/venvs/<your-sbd-venv>/bin/activate
pip install pyscf qiskit "qiskit-addon-sqd>=0.13.1"
```

- **`pyscf`** — reads FCIDUMP, restores 4-fold integral symmetry.
- **`qiskit`** — `BitArray` type for sampled-bitstring input.
- **`qiskit-addon-sqd`** — the SQD loop (`diagonalize_fermionic_hamiltonian`).
  Needs the **distributed (SPMD) support** that calls `sci_solver` on every MPI
  rank; that shipped in 0.13.1, so the PyPI release suffices.

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

- `--counts FILE` — load bitstrings from a count_dict.json
- `--samples N` — generate N random bitstrings at the target Hamming weights
  (default). A plumbing check only: random determinants give a random subspace,
  so the energy is not meaningful. Use `--counts` for real results.

```bash
# H2O with the bundled counts file (275 bitstrings -> ~ -76.236 Ha)
mpirun -np 4 python run_sqd_sbd.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --counts count_dict_h2o.json \
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
produced by a quantum device or simulator. Each bitstring has length `2 × NORB`
and is laid out as **`[beta | alpha]`**: the first `NORB` bits are beta
(spin-down), the last `NORB` are alpha (spin-up), and within each half **orbital 0
is the rightmost bit**. qiskit-addon-sqd postselects the last `NORB` bits on
`num_elec_a` and the first `NORB` on `num_elec_b`.

For H2O (NORB=24, 5α+5β) the Hartree–Fock configuration — the five lowest
orbitals doubly occupied — is therefore `"0"*19 + "1"*5` in *both* halves:

```json
{
  "000000000000000000011111000000000000000000011111": 16,
  "010000000010001010000001010000000001000010100100": 12,
  "000010001110000000000010001001000110000000000100": 8
}
```

Bitstrings whose halves do not hold exactly `num_elec_a` / `num_elec_b` ones are
dropped by postselection, so a file of uniform-random strings yields nothing
usable — for H2O only `C(24,5)² / 4²⁴ ≈ 6e-6` of them qualify.

[`count_dict_h2o.json`](./count_dict_h2o.json) in this directory is a ready-made
H2O example: 275 bitstrings taken from the vendored `h2o-1em3-alpha.txt`
determinant list, giving a 275 × 275 = 75,625-determinant subspace at
≈ -76.236 Ha.

**Key options:** `--fcidump` (required), `--counts`, `--samples`,
`--samples_per_batch`, `--num_batches`, `--max_iterations`, `--device`,
MPI decomposition flags. SBD solver flags (`--method`, `--tolerance`,
`--iteration`, etc.) have sensible defaults; run `python run_sqd_sbd.py --help`
for the full list.

**Requirements:** see [Extra dependencies](#extra-dependencies-only-for-the-sqd-examples) above (`pyscf`, `qiskit`, `qiskit-addon-sqd`).

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
| `--samples N` | Generate N random bitstrings at the target Hamming weights; plumbing check only, energy not meaningful | any |
| `--samples_per_batch` | Subspace dimension per batch (accuracy vs. cost) | 300–800 (small), 1M+ (production) |
| `--num_batches` | Independent subsamples for averaging occupancies | 3–10 (small), up to 100 (large) |
| `--max_iterations` | SQD self-consistent loop iterations (not SBD `--iteration`) | 3–5 |

**MPI work distribution:** All ranks diagonalize each batch together, then move
to the next batch sequentially. Within each diagonalization, ranks form a 3D grid:
`adet_comm_size × bdet_comm_size × task_comm_size = total ranks`. More batches
increases wall time linearly but does not require more ranks.

### 3. run_sqd_sbd.ipynb — Jupyter walkthrough (serial)

Interactive single-rank companion to `run_sqd_sbd.py`. Same SQD self-consistent
loop on h2o, but inside a Jupyter kernel (`MPI.COMM_WORLD` size 1). Uses the
bundled [`count_dict_h2o.json`](./count_dict_h2o.json) (275 bitstrings → 75,625
determinants) and reaches ≈ −76.236 Ha in a few seconds on CPU.

```bash
pytest --nbmake run_sqd_sbd.ipynb      # what CI runs; needs the nbtest extra
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

All compiled backends load eagerly at import. Select per-call via `--device`:

```bash
--device cpu       # host OpenMP (default)
--device gpu       # NVHPC Thrust (requires NVIDIA GPU + HPC SDK build)
--device gpu-omp   # NVHPC OpenMP target offload
--device auto      # GPU if available, else CPU
```

`gpu-omp` links a different OpenMP runtime (`libnvomp`) than `cpu`/`gpu`, so it is
normally built into its own install — see the [Python Bindings README](../../README.md).
`sbd.available_backends()` reports what the current install actually has.

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
