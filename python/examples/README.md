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
extra Python packages. Install them once into the same environment SBD
was built in:

```bash
conda activate sbd          # the env from the Installation section of ../../README.md
pip install qiskit "qiskit-addon-sqd>=0.13.1"
# pyscf is already there if you used the conda recipe in ../../README.md;
# otherwise:  conda install -y -c conda-forge pyscf
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
(These keep their unprefixed names here: this driver *is* SBD. The SQD driver
prefixes them `--sbd_*`.) Run `python run_sbd_diag.py --help` for the full list.

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
MPI decomposition flags, and the SQD tolerances `--energy_tol` /
`--occupancies_tol` / `--sqd_carryover_threshold`. Inner-solver flags are prefixed
(`--sbd_eps`, `--sbd_max_it`, `--sbd_method`, ...) and have sensible defaults; the
old unprefixed spellings still work. Run `python run_sqd_sbd.py --help` for the
full list, which is grouped by layer.

**Requirements:** see [Extra dependencies](#extra-dependencies-only-for-the-sqd-examples) above (`pyscf`, `qiskit`, `qiskit-addon-sqd`).

See [SQD Parameters](#4-sqd-parameters) below for the full reference, grouped by SQD loop / SBD solver / MPI grid / checkpointing.

### 3. run_sqd_sbd.ipynb — Jupyter walkthrough (serial)

Interactive single-rank companion to `run_sqd_sbd.py`. Same SQD self-consistent
loop on h2o, but inside a Jupyter kernel (`MPI.COMM_WORLD` size 1). Uses the
bundled [`count_dict_h2o.json`](./count_dict_h2o.json) (275 bitstrings → 75,625
determinants) and reaches ≈ −76.236 Ha in a few seconds on CPU.

```bash
pytest --nbmake run_sqd_sbd.ipynb      # what CI runs; needs the nbtest extra
# or open it in JupyterLab and step through the cells
```

### 4. SQD Parameters

Reference for every flag `run_sqd_sbd.py` accepts, grouped the way `--help`
groups them: SQD loop, SBD solver, MPI grid, checkpointing.

**How each iteration builds its subspace.** SQD samples bitstrings from a
quantum device, repairs the noisy ones against an orbital-occupancy estimate
(**configuration recovery**), subsamples them into batches, and diagonalizes
each batch. What makes it a *loop* is that two results feed back into the next
iteration. Three sources, concatenated in this priority order
(qiskit-addon-sqd `fermion.py:551`):

```
strs_a = include_a  ++  carryover_strings_a  ++  samples_a    then dedupe, truncate to max_dim, sort
```

1. **`include_a`** — configurations passed as `include_configurations`. Static:
   fixed before the loop, present every iteration, never updated.
2. **`carryover_strings_a`** — from the *previous* iteration's wavefunction. Every
   determinant whose `|coefficient|` is at least `--sqd_carryover_threshold`
   survives, ranked by `|c|^2`.
3. **`samples_a`** — drawn fresh this iteration, sorted by marginal probability.

The order matters when `max_dim` is set: `include` and `carryover` are kept ahead of
fresh samples, so if those two already fill the cap, this iteration's new samples are
truncated away entirely.

The samples are not re-used raw counts. Each iteration re-runs configuration
recovery from the *original* bitstrings using the occupancies from the previous
iteration's best batch (`fermion.py:502`), then subsamples. Recovery is **not
cumulative** — it always re-derives from the raw samples, just with a better
occupancy estimate each time. On iteration 1 there are no occupancies yet, so the
raw samples are only filtered by electron count (Hamming-weight postselection).

So exactly two things flow from iteration N to N+1, and neither is a tolerance:
the **average orbital occupancies** (into recovery, source 3) and the
**wavefunction amplitudes** (into carryover, source 2).

#### SQD loop parameters

*Shapes the subspace — changes the numbers you compute:*

| Parameter | What it controls | Typical values |
|-----------|-----------------|----------------|
| `--counts FILE` | Load hardware bitstrings from a JSON file (use this or `--samples`) | 10K–1M+ shots |
| `--samples N` | Generate N random bitstrings at the target Hamming weights; plumbing check only, energy not meaningful | any |
| `--max_pool_size` | Randomly subsample `--counts` down to at most this many rows *before* the loop sees them (fixed seed). `recover_configurations` (qiskit-addon-sqd) loops in pure Python over every pool row, every iteration, with no vectorized or distributed implementation — cost scales with pool size, not `--samples_per_batch`. Lossless when every row's count is equal (nothing lost by subsampling); otherwise it trades real hardware-sampling weight for speed | unset (no cap) |
| `--samples_per_batch` | Dominant control on subspace dimension. With `symmetrize_spin` the alpha and beta string sets are merged, so the subspace is up to `(2N)^2`, not `N^2` | `3000` (default); see the cost note below |
| `--num_batches` | Independent subsamples per iteration; occupancies are averaged across them | 3–10 (small), up to 100 (large) |
| `--sqd_carryover_threshold` | `\|coefficient\|` cutoff for carrying a determinant into the next iteration. **Lower it to carry more** | `1e-4` (default) |
| `--max_dim` | Cap on strings per spin sector, so the subspace cannot exceed `max_dim^2`. The main brake on runaway setup cost | unset (no cap) |
| `--include_hf` | Force the single Slater determinant with the lowest `num_elec_a`/`num_elec_b` orbital indices occupied into `include_configurations`, every iteration. Cheap correctness check: that determinant's own diagonal energy is an exact lower bound on what a subspace containing it can do — if forcing it in moves the result, the sampled pool was missing it (and probably its low-excitation neighbors too) | off |

*Decides when to stop — changes nothing about the subspace:*

| Parameter | What it controls | Typical values |
|-----------|-----------------|----------------|
| `--max_iterations` | Hard cap on loop iterations (not the inner `--sbd_max_it`) | 3–12 |
| `--energy_tol` | Iteration-to-iteration change in energy | `1e-8` default |
| `--occupancies_tol` | Largest change in any single orbital occupancy — an infinity norm, not an average | `1e-5` default |

**Both stopping criteria must hold in the same iteration** — the test is an `and`
(`fermion.py:584`). A run that reaches `--max_iterations` may be converged in
energy while one stubborn orbital's occupancy is still moving, and loosening only
one tolerance will not stop it. Watch the per-batch energies: while they still
disagree, the loop has not converged regardless of what the total says.

Neither tolerance is comparable to `--sbd_eps`, which is the residual norm inside a
single diagonalization, not an energy difference between iterations.

**A note on cost.** Small subspaces are dominated by Python-side work — parsing the
counts file, configuration recovery over every raw bitstring, subsampling — not by
the diagonalization. Measured on 8×H100 for a 45-orbital / 46-electron case with 1M
sampled bitstrings, 3 batches, 3 iterations: `samples_per_batch=300` (subspace
360,000) took 254 s, and `samples_per_batch=3000` (subspace 36,000,000 — 100× larger)
took 326 s, only 1.28× longer, while lowering the energy by about 3 Ha. If a run
looks cheap, the subspace is probably too small to be using the hardware.

**When the subspace or memory runs away.** The subspace grows every iteration —
carryover accumulates on top of fresh samples — so a run that starts comfortably can
fail later. Two symptoms, one cause. Setup time exploding between iterations (the
`Elapsed time for helper construction` line) is host-side work: `MakeHelpers` is
superlinear in determinants per spin, and no GPU setting affects it. A
`cudaErrorMemoryAllocation` on the Thrust backend is the determinant index, which by
default is allocated over the entire subspace.

**`--max_dim`** is the fix: it bounds the subspace directly and clears both symptoms
at once. **Raising `--sqd_carryover_threshold`** carries fewer determinants forward,
which is the right move when the *growth* between iterations is the problem rather
than the starting size. The Thrust-only `--sbd_use_precalculated_dets 0` (optionally
with `--sbd_max_memory_gb_for_determinants N`) trades matvec speed for GPU memory,
but it bounds only that one buffer and not the subspace, so it is no substitute for
`--max_dim` and is rarely what a capped run needs. Note the diagonalization itself
is rarely the bottleneck: an 800M-determinant Davidson solve measured 1.0 s against
9.3 s of helper construction in the same iteration, so tune the subspace, not the
solver.

**SBD's own carryover plays no part in any of this.** `carryover_type` and friends
are SBD's separate iterative scheme, for re-running SBD's CLI against its own
`--carryover_adetfile`. They are deliberately not flags on this driver, and setting
them through `sbd_config` cannot change an SQD result.

#### SBD solver parameters

Per diagonalization, not per loop:

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `--sbd_eps` | Davidson stop: **norm of the residual vector**, not an energy. Error in the energy goes roughly as `\|R\|^2/gap`, so this already implies far better energy accuracy than `--energy_tol` asks for. Tighten it for a near-degenerate system | `1e-5` |
| `--sbd_max_it` | Cap on Davidson iterations. Reaching it before `--sbd_eps` returns a partially converged vector **with no warning** — watch the `tol=` values SBD prints, and cross-batch agreement | `10` |
| `--sbd_max_nb` | Davidson basis vectors (block size). Peak memory during Davidson grows with how many sub-iterations it actually needs, not just `--max_dim` — a run that succeeds for several iterations at a fixed `dim` can still later need more basis vectors and run out of memory even though the subspace itself did not grow. Lowering this trades some convergence robustness for a lower memory ceiling | `10` |
| `--sbd_method` | 0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham | `0` |
| `--sbd_use_precalculated_dets` | Thrust only. `1` precomputes a determinant index for **every** (α,β) pair — the whole subspace, on the GPU. `0` uses per-thread storage: slower per matvec, far less memory | `1` |
| `--sbd_max_memory_gb_for_determinants` | Thrust only, and **only consulted when `--sbd_use_precalculated_dets 0`** (`mult_thrust.h:257-273`). Caps the per-thread buffer in GB | `-1` (uncapped) |

For reference, upstream's own `TPB_SBD` struct defaults are looser still (`max_it=1`,
`eps=1e-4`), and `run_sbd_diag.py` uses `eps=1e-3`. On the h2o counts case,
`eps=1e-5` and `eps=1e-8` give the same energy to ten decimal places.

#### MPI grid parameters

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `--adet_comm_size` | Ranks spanning the alpha-determinant dimension | `1` |
| `--bdet_comm_size` | Ranks spanning the beta-determinant dimension | `1` |
| `--task_comm_size` | Ranks spanning task-level parallelism | `1` |

All ranks diagonalize each batch together, then move to the next batch
sequentially. See [MPI Decomposition](#mpi-decomposition) below for the full 4D
grid (a fourth, *derived* dimension — `helper` — is not set directly).

#### Checkpointing parameters

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `--checkpoint_path` | Write `ci_strs_a`/`ci_strs_b`/`orbital_occupancies`/energy to this path as JSON text (rank 0 only), every `--checkpoint_frequency` iterations. **Must be visible under the same path from every rank** — `--resume_from` has no rank-0-reads-then-broadcasts step, every rank opens this path itself | unset |
| `--checkpoint_frequency` | Write every this many iterations, plus always on the last one regardless of alignment | `1` (every iteration) |
| `--resume_from` | Seed a new run's `include_configurations`/`initial_occupancies` from a previous `--checkpoint_path`'s last recorded iteration | unset |

What's preserved is which determinants (`ci_strs_a`/`ci_strs_b`, as plain
integers — independent of MPI decomposition, since they carry no rank/grid
information) and the derived per-orbital occupancy averages. **Not** the
wavefunction amplitudes matrix itself (`dim_a × dim_b` floats — at large
`--max_dim` this alone would dwarf everything else; neither consuming parameter
needs it).

Not a bit-identical continuation: RNG state is fresh in the new process, and
every string from the resumed iteration becomes a **permanent** include for
every iteration of the new run — unlike a true single-process continuation,
where `--sqd_carryover_threshold` would keep pruning low-weight determinants
each iteration. A resumed run is seeded richer than an actual continuation
would have been at that point, not identical to one.

## MPI Decomposition

Total MPI ranks must be a **multiple** of
`task_comm_size × adet_comm_size × bdet_comm_size` — not equal to it. SBD splits
the ranks you asked for across those three dimensions and puts whatever remains
into a fourth, "helper" dimension, computed as
`ranks / (task_comm_size × adet_comm_size × bdet_comm_size)`.

So 8 ranks with `--adet_comm_size 2 --bdet_comm_size 2` is valid: the grid is
`1 × 2 × 2` and the helper dimension absorbs the remaining factor of 2.

A rank count that is **not** a multiple does not run with idle ranks — it **aborts**.
SBD derives the helper dimension by integer division and then requires
`task × adet × bdet × helper == ranks` exactly (`TaskCommunicator`,
`chemistry/tpb/helper.h`), so e.g. 8 ranks with a grid of 3 gives
`helper = 2`, `3 × 2 = 6 ≠ 8`, and the run stops with
`ValueError: MPI Size of twister is not a square of a integer`. That message names
neither the grid nor the rank count, so if you see it, check this arithmetic first.

When using more than one rank, specify at least `--adet_comm_size`. Examples:

| Ranks | Decomposition | Helper |
|-------|---------------|--------|
| 1 | default (all = 1) | 1 |
| 2 | `--adet_comm_size 2` | 1 |
| 4 | `--adet_comm_size 2 --bdet_comm_size 2` | 1 |
| 8 | `--adet_comm_size 2 --bdet_comm_size 2` | 2 |
| 8 | `--adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2` | 1 |

**GDB** (`gdb_diag`) decomposes differently: `t_comm_size × b_comm_size × helper`,
with its own field names rather than TPB's. It is not exercised by these examples
or by the test suite, so its decomposition is unvalidated and is deliberately not
documented further here.

## Backend Selection

Every backend the toolchain supported was compiled into this one install, and
each is imported **lazily, on first use** — normally one per process. Select
per-call via `--device`:

```bash
--device cpu       # host OpenMP (default)
--device gpu       # NVHPC Thrust (requires NVIDIA GPU + HPC SDK build)
--device gpu-omp   # NVHPC OpenMP target offload
--device auto      # GPU if available, else CPU
```

`sbd.available_backends()` reports what this install actually has (a static scan
— it does not import anything, so it is safe to call outside `mpirun`), and
`sbd.loaded_backends()` reports what the current process has pulled in.

Lazy loading is what makes the three backends safe to co-install: see
[Backend Architecture](../../README.md#backend-architecture) in the Python
Bindings README for why. One consequence is worth knowing when you write your
own driver — **do not import the CPU and `gpu-omp` backends into the same
process.** They share NVHPC's `libnvomp`, and loading `_core_cpu` first leaves
it initialised host-only, after which offload regions run on the host while
device queries still report a GPU. `sbd.has_backend_conflict()` returns True if
that has happened. Loading `cpu` and `gpu` (Thrust) together is fine.

Within Python, backends can be selected per call — no re-initialization needed:

```python
import sbd

# No init() needed — auto-initializes on first call
result_cpu = sbd.tpb_diag(..., device='cpu')
result_gpu = sbd.tpb_diag(..., device='gpu')     # fine alongside 'cpu'
# result_omp = sbd.tpb_diag(..., device='gpu-omp')   # NOT in the same process as 'cpu'
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
