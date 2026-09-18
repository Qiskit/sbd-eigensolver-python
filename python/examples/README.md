# SQD/SBD Python Examples

Examples demonstrating SBD's capabilities for quantum chemistry calculations.

## Overview

- **Communication:** MPI for distributed computing
- **Backends:** CPU (host OpenMP, `--device cpu`), GPU (NVHPC Thrust, NVIDIA only, `--device gpu`) and GPU (OpenMP target offload, NVIDIA and AMD, `--device gpu-omp`), switchable at runtime via `device` parameter

Replace `--device gpu` in all the examples below with `--device gpu-omp` if you use AMD GPUs.

## Examples

### 1. run_sbd_diag.py — Standalone SBD Diagonalization

Runs a single TPB diagonalization from an FCIDUMP file and alpha determinant
file. No SQD loop, no Qiskit dependency.

```bash
# H2O with 2 MPI ranks with CPU
mpirun -np 2 python -u run_sbd_diag.py \
    --device cpu \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --adetfile ../../vendor/sbd-upstream/data/h2o/h2o-1em3-alpha.txt \
    --adet_comm_size 2

# N2 with GPU
mpirun -np 8 python -u run_sbd_diag.py \
    --device gpu \
    --fcidump ../../vendor/sbd-upstream/data/n2/fcidump.txt \
    --adetfile ../../vendor/sbd-upstream/data/n2/1em3-alpha.txt \
    --adet_comm_size 4 --bdet_comm_size 2

# Retrieve the 1-/2-particle RDMs and save them to a file
mpirun -np 2 python -u run_sbd_diag.py --rdm_output /tmp/h2o_rdms.npz
```

`--rdm_output` takes the file to save to (`/tmp/h2o_rdms.npz` above) and
writes **one** `.npz` file there holding both `rdm1` and `rdm2` together
(`data = np.load("/tmp/h2o_rdms.npz"); data["rdm1"]`, `data["rdm2"]`) —
unlike upstream SBD's own CLI, which writes two separate files
(`1pRDM.txt`/`2pRDM.txt`). It also prints `trace(rdm1)` and the natural
orbital occupations. Leaving it empty (the default) skips computing RDMs
entirely.

By default beta determinants are derived from `--adetfile` alone (identical
to it, or a shuffled copy if `--shuffle` is set). `--symmetrize_spin 0`
loads `--adetfile` and `--bdetfile` as independent, genuinely distinct
alpha/beta determinant sets instead; `--bdetfile` is otherwise ignored
(with a warning) since symmetric mode always derives beta from alpha.

**Key options:** `--device`, `--fcidump`, `--adetfile`, `--bdetfile`,
`--symmetrize_spin`, `--adet_comm_size`, `--bdet_comm_size`,
`--task_comm_size`, `--method`, `--tolerance`, `--iteration`,
`--rdm_output`. (These keep their unprefixed names here: this driver *is*
SBD. The SQD drivers prefix them `--sbd_*`.) Run `python run_sbd_diag.py
--help` for the full list.

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
mpirun -np 8 python -u run_sqd_sbd.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --counts count_dict_h2o.json \
    --device gpu \
    --adet_comm_size 4 --bdet_comm_size 2

# Custom system with random bitstrings
mpirun -np 8 python -u run_sqd_sbd.py \
    --fcidump /path/to/fci_dump.txt \
    --samples_per_batch 800 --num_batches 3 --max_iterations 10 \
    --device gpu \
    --adet_comm_size 4 --bdet_comm_size 2
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

**Requirements:** see [Integration with qiskit-addon-sqd](../../README.md#integration-with-qiskit-addon-sqd) in the Python Bindings README (`pyscf`, `qiskit`, `qiskit-addon-sqd`).

See [SQD Parameters](#sqd-parameters) below for the full reference, grouped by SQD loop / SBD solver / MPI grid / checkpointing.

### 3. run_sqd_enlarge_subspace_sbd.py — SQD that grows its own subspace

Same self-consistent SQD loop as `run_sqd_sbd.py` above (sampling,
configuration recovery, SBD as the solver), but with one addition: between
rounds, it expands the dominant determinant pairs from the just-solved
wavefunction via qiskit-addon-sqd's own `enlarge_batch_from_transitions`
(same-spin single-electron excitations, both alpha and beta), and feeds the
result forward as the next round's `include_configurations`. Concretely,
it calls `diagonalize_fermionic_hamiltonian` with `max_iterations=1` itself,
in its own outer Python loop, rather than delegating the whole
multi-iteration loop to one call the way `run_sqd_sbd.py` does -- that's
what makes injecting a step between rounds possible. The loop stops when
either the expanded set adds nothing new, or the energy and occupancies
both stop moving (`--energy_tol`/`--occupancies_tol`) -- `--max_iterations`
is a safety cap, not the expected stopping mechanism.

```bash
JAX_PLATFORMS=cpu mpirun -np 8 python -u run_sqd_enlarge_subspace_sbd.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --counts count_dict_h2o.json \
    --device gpu \
    --adet_comm_size 4 --bdet_comm_size 2 --enlarge_threshold 1e-4
```

Note that qiskit-addon-sqd's own JAX-based `enlarge_batch_from_transitions` has no MPI awareness. Set `JAX_PLATFORMS=cpu`
when running an MPI job using more than 1 rank.

Same bundled 275-bitstring H2O pool as `run_sqd_sbd.py`'s own example above:
plain SQD reaches **≈ -76.236 Ha** and stops there; this driver keeps going
past that fixed pool on its own and converges to **-76.2421767512 Ha**.

See [SQD Parameters](#sqd-parameters) below for the flags it shares with
`run_sqd_sbd.py` and the ones that differ (`--enlarge_threshold` in place
of `--sqd_carryover_threshold`, and `--max_dim`'s risk profile is sharper
here).

### 4. run_sqd_sbd_carryover.py — SQD that grows via SBD's own carryover

Same loop and same goal as `run_sqd_enlarge_subspace_sbd.py` above, with the
expansion step swapped: instead of deriving single excitations in Python with
qiskit-addon-sqd's JAX `enlarge_batch_from_transitions`, it asks **SBD** to
select the next round's determinants, inside the same C++ diagonalization that
just ran. They come back on the result and are fed forward as the next round's
`include_configurations`.

What that buys, and what it does not:

- **No `JAX_PLATFORMS=cpu` needed.** The JAX expansion has no MPI awareness, so
  on a GPU-enabled JAX install several ranks each try to claim a device and the
  run dies with `CUDA_ERROR_OUT_OF_MEMORY`. Nothing here touches JAX.
- **It is not faster.** Measured head to head on a 45-orbital system (8 ranks,
  `--max_dim 15000`, threshold `1e-4`): 2248 s against 2250 s for the JAX path,
  reaching bit-identical energies and subspace sizes at every round. Each round
  is dominated by configuration recovery and by diagonalizing the subspace, so
  where the expansion runs makes no measurable difference. Pick this driver for
  the operational reason above, not for speed.

```bash
mpirun -np 8 python -u run_sqd_sbd_carryover.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --counts count_dict_h2o.json \
    --device gpu \
    --adet_comm_size 4 --bdet_comm_size 2 \
    --sbd_carryover_type 3 --sbd_carryover_threshold 1e-4
```

On the bundled 275-bitstring H2O pool this reaches **-76.2421767512 Ha** over a
1742×1742 subspace — the same energy and the same subspace as
`run_sqd_enlarge_subspace_sbd.py`, which is the point: at
`--sbd_carryover_type 3` the two expansions are equivalent, so this is the
cheaper way to get there. See
[Tuning the expansion threshold](#tuning-the-expansion-threshold) for how far
`--sbd_carryover_threshold` can push that number.

### 5. run_sqd_sbd.ipynb — Jupyter walkthrough (serial)

Interactive single-rank companion to `run_sqd_sbd.py`. Same SQD self-consistent
loop on h2o, but inside a Jupyter kernel (`MPI.COMM_WORLD` size 1). Uses the
bundled [`count_dict_h2o.json`](./count_dict_h2o.json) (275 bitstrings → 75,625
determinants) and reaches ≈ −76.236 Ha in a few seconds on CPU.

```bash
pytest --nbmake run_sqd_sbd.ipynb      # what CI runs; needs the nbtest extra
# or open it in JupyterLab and step through the cells
```

## SQD Parameters

Reference for the three drivers built on qiskit-addon-sqd's loop, grouped the
way `--help` groups them: SQD loop, SBD solver, MPI grid, checkpointing.
Throughout this section they are referred to by these short names:

| Short name | Driver | How the subspace grows between iterations |
|---|---|---|
| **sqd** | `run_sqd_sbd.py` (section 2) | it doesn't — fixed pool, resampled each iteration |
| **enlarge** | `run_sqd_enlarge_subspace_sbd.py` (section 3) | qiskit-addon-sqd's JAX single excitations |
| **carryover** | `run_sqd_sbd_carryover.py` (section 4) | SBD's own carryover, in C++ |

`run_sbd_diag.py` does **not** use this loop — it runs a single
diagonalization and takes a different flag set, documented in section 1
above.

**How each iteration builds its subspace.** SQD samples bitstrings from a
quantum device, repairs the noisy ones against an orbital-occupancy estimate
(**configuration recovery**), subsamples them into batches, and diagonalizes
each batch. What makes it a *loop* is that two results feed back into the next
iteration. Three sources, concatenated in this priority order inside
qiskit-addon-sqd's `diagonalize_fermionic_hamiltonian` (`fermion.py`):

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
iteration's best batch (`_prepare_ci_strings` in `fermion.py`), then
subsamples. Recovery is **not
cumulative** — it always re-derives from the raw samples, just with a better
occupancy estimate each time. On iteration 1 there are no occupancies yet, so the
raw samples are only filtered by electron count (Hamming-weight postselection).

So exactly two things flow from iteration N to N+1, and neither is a tolerance:
the **average orbital occupancies** (into recovery, source 3) and the
**wavefunction amplitudes** (into carryover, source 2).

### SQD loop parameters

**`--max_dim` is the one that most needs attention**: it has no universally
safe default (see below), and in **enlarge**/**carryover** leaving it unset is
riskier still, since each round's subspace can grow from the previous one
rather than being resampled at a fixed size — both drivers print an OOM
warning when they detect this.

*Shapes the subspace — same flag, same default, in all three:*

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `--counts FILE` | Load hardware bitstrings from a JSON file (use this or `--samples`) | none — falls back to `--samples` if omitted |
| `--samples N` | Generate N random bitstrings at the target Hamming weights; plumbing check only, energy not meaningful | `3000` (only used when `--counts` is omitted) |
| `--samples_per_batch` | Dominant control on subspace dimension. With `--symmetrize_spin 1` the alpha and beta string sets are merged, so the subspace is up to `(2N)^2`, not `N^2` | `3000` |
| `--symmetrize_spin` | `1`: merge the alpha and beta string pools every iteration, forcing `ci_strs_a == ci_strs_b`. SBD itself supports distinct alpha/beta determinant sets — this is purely a qiskit-addon-sqd loop-layer setting. `0`: sample and carry over alpha and beta independently, allowing them to differ | `1` |
| `--max_dim` | **Critical.** Cap on strings per spin sector, so the subspace cannot exceed `max_dim^2`. The main brake on runaway cost — no fixed value is safe for every system, since the right cap depends on available memory and orbital count. Start from a value known to work at a similar orbital count (e.g. `15000` was used for a 45-orbital system) and adjust down if you see an OOM | unset (no cap) |
| `--include_hf` | Force the single Slater determinant with the lowest `num_elec_a`/`num_elec_b` orbital indices occupied into `include_configurations`, every iteration. Cheap correctness check: that determinant's own diagonal energy is an exact lower bound on what a subspace containing it can do — if forcing it in moves the result, the sampled pool was missing it (and probably its low-excitation neighbors too) | off |
| `--energy_tol` | Stop when the iteration-to-iteration energy change falls below this | `1e-8` |
| `--occupancies_tol` | Stop when the largest change in any single orbital occupancy falls below this — an infinity norm, not an average | `1e-5` |

*Same flag in all three, but a different default in each:*

| Parameter | What it controls | sqd | enlarge | carryover |
|-----------|-----------------|-----|---------|-----------|
| `--num_batches` | Independent subsamples per iteration; occupancies are averaged across them. Not the main lever on subspace size in enlarge/carryover — the expansion is | `3` | `1` | `1` |
| `--max_iterations` | Hard cap on loop iterations (**not** the inner `--sbd_max_it`). In enlarge/carryover it is a safety cap only: the loop normally stops earlier, once a round adds no new determinants or both tolerances above are met | `5` | `30` | `30` |

*How the subspace grows — one knob per driver, and they are **not**
interchangeable:*

| Driver | Parameter | What it gates | Default |
|--------|-----------|---------------|---------|
| **sqd** | `--sqd_carryover_threshold` | `\|coefficient\|` cutoff for carrying a determinant into the next iteration's **sample pool**. Generates no new determinants — pure selection. **Lower it to carry more** | `1e-4` |
| **enlarge** | `--enlarge_threshold` | `\|amplitude\|^2` cutoff on determinant **pairs**, which are then expanded into all same-spin single excitations via `enlarge_batch_from_transitions`. **Lower it to expand from more pairs** | `1e-4` |
| **carryover** | `--sbd_carryover_type` | Which mechanism SBD uses to pick the next round's determinants: `1` selection only, `2` singles off marginal probability, `3` singles off full-determinant amplitude | `3` |
| **carryover** | `--sbd_carryover_threshold` | The cutoff SBD applies while doing the above. **Meaning depends on the type**: for `3` it is a full-determinant `\|c\|^2` cutoff, for `1`/`2` a marginal (half-determinant) probability | `1e-4` |

At `--sbd_carryover_type 3`, **carryover**'s threshold gates the same quantity
as **enlarge**'s — both are a full-determinant `|c|^2` cutoff followed by all
same-spin singles. Verified: at `1e-4` on the bundled H2O pool the two drivers
reach an identical energy over an identical final subspace. At other carryover
types the quantity differs, so the values are **not** transferable.

#### Tuning the expansion threshold

The threshold is the main accuracy lever in **enlarge**/**carryover**, and the
default `1e-4` is deliberately conservative. Lowering it prunes less, so more
determinants survive into the next round and the subspace grows — which lowers
(improves) the energy, since a variational subspace method can only get better
as the subspace grows.

Sweeping it on the bundled 275-bitstring H2O pool, with **carryover** at type 3
(reproducible with the command in section 4 above, changing only
`--sbd_carryover_threshold`):

| threshold | energy (Ha) | gap to FCI | final subspace |
|---|---|---|---|
| `1e-4` (default) | -76.2421767512 | 1.60 mHa | 1742² = 3.0M |
| `1e-5` | -76.2436018956 | 0.175 mHa | 5007² = 25.1M |
| `1e-6` | -76.2437251036 | **0.052 mHa** | 7881² = 62.1M |

against this system's FCI reference of `-76.24377680`. Two things to read off
it. First, the payoff is real: two orders of magnitude on the threshold buys
**30x** less error. Second, it is bought with determinants — 20x more of them —
and the returns diminish sharply, with `1e-5`→`1e-6` costing 2.5x the subspace
for a further 0.12 mHa. Sweep downward until the gain stops being worth the
cost for your purpose, rather than reaching for the smallest value.

**The one trap: `--max_dim` inverts this.** Everything above assumes the
subspace is allowed to grow freely. Once a run is pinned at the cap, lowering
the threshold can make the result *worse*, not better. The cap keeps the
determinants already in the subspace and then fills the remaining budget
**randomly** from this round's new candidates, so a lower threshold means a
much larger candidate pool sampled just as thinly — you pay for generating
them and then discard most, with no ranking to decide which survive.

The symptom is easy to spot: the final subspace printed at the end is exactly
`max_dim x max_dim`, and the energy moved the wrong way when you tightened the
threshold. We have seen this reverse the ordering outright on a 45-orbital
system at `--max_dim 15000`, where `1e-4` beat `1e-5` comfortably. If that
happens, raise `--max_dim` (memory permitting) or back the threshold off —
tightening it further will not help.

So, in practice: leave `--max_dim` unset while you sweep the threshold, and
only introduce it once you know the subspace size you are aiming at. Set both
at once and it is hard to tell which one is limiting the answer.

**Both stopping criteria must hold in the same iteration.** `fermion.py`'s
convergence check combines the energy-change test and the occupancy-change test
with a logical `and`, so the loop only stops once both are satisfied at once —
not whichever one happens first. A run that reaches
`--max_iterations` may be converged in energy while one stubborn orbital's
occupancy is still moving, and loosening only one tolerance will not stop it.
Watch the per-batch energies: while they still disagree, the loop has not
converged regardless of what the total says.

### SBD solver parameters

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

### MPI grid parameters

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `--adet_comm_size` | Ranks spanning the alpha-determinant dimension | `1` |
| `--bdet_comm_size` | Ranks spanning the beta-determinant dimension | `1` |
| `--task_comm_size` | Ranks spanning task-level parallelism | `1` |

All ranks diagonalize each batch together, then move to the next batch
sequentially. See [MPI Decomposition](#mpi-decomposition) below for the full 4D
grid (a fourth, *derived* dimension — `helper` — is not set directly).

### Checkpointing parameters

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
--device gpu-omp   # OpenMP target offload, NVIDIA and AMD GPUs
--device auto      # GPU if available, else CPU
```

`sbd.available_backends()` reports what this install actually has (a static scan
— it does not import anything, so it is safe to call outside `mpirun`), and
`sbd.loaded_backends()` reports what the current process has pulled in.

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
