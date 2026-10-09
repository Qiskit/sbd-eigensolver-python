# SBD Python Bindings

Python bindings for the Selected Basis Diagonalization (SBD) library, with CPU and GPU backends.

## Overview

SBD (Selected Basis Diagonalization) is a high-performance library for quantum
chemistry calculations. These bindings expose two of its diagonalization methods on CPU
and GPU. They differ in the shape of the subspace they span, which is what decides
which one you want:

- **Tensor-Product Basis (TPB)** — the subspace is the Cartesian product of an alpha
  and a beta determinant list, so its dimension is `|adet| × |bdet|`. The mature path,
  and the one the qiskit-addon-sqd integration uses.
- **General-Determinant Basis (GDB)** — the subspace is the explicit determinant list
  you pass, so it can be an arbitrary *sparse* set rather than a product. Experimental:
  newer, a smaller tested surface, and still evolving as upstream SBD does.

If your subspace is a product, prefer TPB — it represents that case with two
half-determinant lists instead of every product element, and needs no extra
constraints. Reach for GDB when the subspace is not a product.

**Key Features:**
- **TPB and GDB diagonalization** for quantum chemistry Hamiltonians
- Three backends, selected per call at runtime via `device=`:
  `'cpu'` (host OpenMP), `'gpu'` (NVHPC Thrust/CUDA, NVIDIA only) and
  `'gpu-omp'` (OpenMP target offload, **NVIDIA or AMD**). All the backends your
  toolchain supports can be built into one install; each is imported only when
  first used. **TPB runs on all three; GDB has Thrust kernels only**, so GDB on a
  GPU is NVIDIA-only and under `'gpu-omp'` it falls back to the host — see
  [`examples/gdb/README.md`](examples/gdb/README.md)
- MPI parallelization
- Integration with [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) for SQD workflows

SBD's **Creation/Annihilation operator (CAOP)** method is not supported by this
wrapper; users who need it should reference and use the C++ CLI apps in the upstream
submodule (`vendor/sbd-upstream/apps/`).

> [!NOTE]
> This package is newly open-sourced. The Python API follows semantic versioning, but the build configuration and GPU backends have been exercised on a limited set of platforms — please report issues.

## Installation

The extension modules are compiled on your machine — no wheels are published — so the
backends you get depend on the toolchain the build finds. For the common CPU case:

```bash
conda create -y -n sbd -c conda-forge \
    python=3.13.12 pybind11 numpy setuptools wheel openblas pyscf pip mpi4py
#   ...plus llvm-openmp on macOS
conda activate sbd

pip install sbd-eigensolver
```

Then check what was built:

```bash
python -c "import sbd; print(sbd.available_backends())"
# CPU only:              ['cpu']
# NVIDIA, default build: ['cpu', 'gpu', 'gpu-omp']
# AMD, default build:    ['cpu', 'gpu-omp']
```

**See [INSTALL.md](INSTALL.md)** for the rest: prerequisites per platform, installing
from a git checkout with the submodule, every environment variable the build reads
(GPU toolchains, architectures, MPI and BLAS selection, narrowing which backends get
built), building against an existing host MPI, and fuller verification.

## Examples

Located in `examples/`, organized by basis type since the solvers take different
subspaces and decompose over MPI differently. Each folder's README is the authoritative
guide to what it contains, how to run it, and the backend and threading settings that
matter for it.

- [`examples/tpb/`](examples/tpb/README.md) — tensor-product basis: standalone TPB
  diagonalization, the SQD loops, and the subspace-enlargement driver.
- [`examples/gdb/`](examples/gdb/README.md) — general determinant basis: standalone
  GDB diagonalization over an explicit determinant list, and an iterative
  heatbath-expansion driver.

## Integration with qiskit-addon-sqd

SBD can serve as the eigensolver backend for qiskit-addon-sqd's SQD workflow.

> [!NOTE]
> This integration is **TPB-only**, and not merely for want of plumbing.
> `solve_sci`/`solve_sci_batch` call `tpb_diag`, and the addon's interface describes a
> product subspace by construction: `ci_strings` is a `(strings_a, strings_b)` pair and
> `SCIState.amplitudes` is an `|a| × |b|` matrix. A sparse determinant list cannot be
> expressed that way without padding back up to the full product, which discards the
> reason to use GDB. For GDB, call `gdb_diag` directly or use the drivers in
> [`examples/gdb/`](examples/gdb/README.md).

**Note:** Requires [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) with distributed (SPMD) support — `diagonalize_fermionic_hamiltonian` calling `sci_solver` on every MPI rank. This is available in `qiskit-addon-sqd` version `0.13.1` or higher. SBD-selected carryover and subspace policies need `0.14.0` or higher.

### Plain SQD

```python
from functools import partial
from sbd.sbd_solver import solve_sci_batch
from sbd.device_config import DeviceConfig
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

# No sbd.init() and no explicit mpi_comm needed — solve_sci_batch
# auto-initializes the SBD backend on first call and falls back to
# MPI.COMM_WORLD when mpi_comm is not provided.
sbd_solver = partial(
    solve_sci_batch,
    sbd_config={"method": 0, "eps": 1e-5, "max_it": 10, "max_nb": 10},
    device_config=DeviceConfig.gpu(),        # or .cpu(), .gpu_omp()
    fcidump_path="data/h2o/fcidump.txt",     # optional: reuse one FCIDUMP across batches
)

result = diagonalize_fermionic_hamiltonian(
    hcore, eri, bit_array,
    sci_solver=sbd_solver,                   # SBD plugs in here
    norb=norb, nelec=nelec,
    samples_per_batch=3000, num_batches=3, max_iterations=5,
    symmetrize_spin=True,
)
```

See [SQD Parameters](examples/tpb/README.md#sqd-parameters) for how each
parameter feeds the loop, and
[examples/tpb/run_sqd_sbd.py](examples/tpb/run_sqd_sbd.py) for a
complete example.

qiskit-addon-sqd is the orchestrator in that recipe: it owns the loop
(sampling, configuration recovery, subsampling), and SBD is plugged in as the
per-batch eigensolver (`sci_solver=sbd_solver` above). By default SBD has no say
in how the subspace grows between iterations: SQD chooses the carryover from the
amplitudes SBD returns.

With `qiskit-addon-sqd >= 0.14.0`, SBD can choose the carryover itself instead:
set `"carryover_type"` in `sbd_config` and SQD takes SBD's selection as given. With
type 1, the amplitudes never leave SBD, which saves a full wavefunction write and
read every iteration. See
[SBD-selected carryover](examples/tpb/README.md#sbd-selected-carryover) for what
each type does and costs, and the section after it for running the loop with a
different subspace policy such as Trim SQD.

### SQD with subspace enlargement

[`run_sqd_enlarge_subspace_sbd.py`](examples/tpb/run_sqd_enlarge_subspace_sbd.py)
builds on the same recipe, but grows its own subspace between rounds: after
each solve, it expands the dominant determinant pairs via qiskit-addon-sqd's
own `enlarge_batch_from_transitions` (same-spin single excitations, both
alpha and beta) and feeds the result forward as the next round's
`include_configurations`. Concretely, it calls
`diagonalize_fermionic_hamiltonian` with `max_iterations=1` itself, in its
own outer Python loop, rather than delegating the whole multi-iteration loop
to one call — that is what makes injecting a step between rounds possible.

On the bundled H2O pool ([`count_dict_h2o.json`](examples/tpb/count_dict_h2o.json),
275 bitstrings), plain SQD reaches ≈ -76.236 Ha and stops there; this driver
keeps going past that fixed pool on its own and converges to
**-76.2421767512 Ha**.

## Backend Architecture

- **The Thrust build assumes a GPU-aware MPI** and hands MPI device pointers directly. Two upstream escape hatches exist for an MPI that cannot address device memory. Both are **off by default**, and both are **compile-time** — set them before installing `sbd-eigensolver`, rebuild to change them, and they do nothing if set at run time:
    ```
    SBD_NON_CUDA_AWARE_MPI=1         stage every device buffer through host memory
    SBD_THRUST_SAFE_MPI_ALLREDUCE=1  stage only the allreduce (a subset of the above)
    ```
    Both make the Thrust path stage device buffers through host memory instead of handing MPI device pointers, so they add copies whose cost grows with how much data crosses MPI. Treat them as a compatibility fallback rather than a tuning knob: reach for them when a GPU backend crashes inside the MPI itself rather than in SBD.
- GPU device assignment: `gpu_id = mpi_rank % num_gpus` (set per `tpb_diag()` call in `bindings.cpp`); same logic for both Thrust and OMP-offload paths.
- **`'gpu-omp'` is vendor-neutral by design:** one module, one device string, for both NVIDIA and AMD. It is the same source with the same macros and only the compiler differs, and since no wheels are published — every install compiles on the target machine — an install serves one GPU vendor. `__sbd_offload_target__` records which one. Aliases (`gpu-amd-omp`, `gpu-rocm-omp`, `rocm`, `gpu-nvidia-omp`, …) resolve to it so a vendor-flavoured guess lands correctly.
- Verify what GPU architecture is supported in the binary:
  ```
  NVIDIA: cuobjdump --list-elf   <the built _core_gpu_thrust*.so>
  AMD:    llvm-objdump --offloading <the built _core_gpu_omp_offload*.so>
  ```
  Or just ask the module what it was built for:
  ```
       python -c "import sbd; \
         print(sbd.get_backend('gpu-omp').__sbd_offload_target__)"
         -> amdgcn-amd-amdhsa:gfx90a
  ```

## API Reference

### Initialization

| Function | Description |
|----------|-------------|
| `sbd.init(device, comm_backend)` | **Optional.** Initialize MPI, set default device (`'cpu'`, `'gpu'`, `'gpu-omp'`, `'auto'`). Auto-called on first use with defaults. |
| `sbd.finalize()` | Sync GPU, reset state. Does not call `MPI_Finalize` |
| `sbd.is_initialized()` | Check init status |

### Backend Access

| Function | Description |
|----------|-------------|
| `sbd.get_backend(device=None)` | Get the pybind11 backend module for the named device. `None` = default device. |
| `sbd.available_backends()` | List of compiled backends, e.g. `['cpu']`, `['cpu', 'gpu']`, `['gpu-omp']` |

### Query

| Function | Description |
|----------|-------------|
| `sbd.get_device()` | Default device name |
| `sbd.get_rank()` | MPI rank |
| `sbd.get_world_size()` | MPI world size |
| `sbd.get_comm()` | MPI communicator |
| `sbd.barrier()` | MPI barrier |

### Configuration

```python
config = sbd.TPB_SBD()
```

| Attribute | Default | Description |
|-----------|---------|-------------|
| `method` | 0 | 0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham |
| `max_it` | 1 | Max iterations |
| `eps` | 1e-4 | Convergence tolerance |
| `max_nb` | 10 | Max basis vectors |
| `do_rdm` | 0 | 0=density only, 1=full RDM |
| `bit_length` | 20 | Bit length for determinants |
| `adet_comm_size` | 1 | Alpha determinant communicator size |
| `bdet_comm_size` | 1 | Beta determinant communicator size |
| `task_comm_size` | 1 | Task communicator size |
| `dump_matrix_form_wf` | `""` | Write the wavefunction amplitudes to this path — see [Getting the TPB amplitudes](#getting-the-tpb-amplitudes) |

Total MPI ranks = `task_comm_size × adet_comm_size × bdet_comm_size`.

```python
config = sbd.GDB_SBD()
```

Shares `max_it`, `max_nb`, `eps`, `max_time`, `init`, `do_rdm`, `carryover_type`,
`ratio`, `threshold` and `bit_length` with `TPB_SBD`, and replaces the determinant
communicators with a single basis communicator:

| Attribute | Default | Description |
|-----------|---------|-------------|
| `method` | 0 | 0=Davidson, 1=Davidson+Ham. GDB has no Lanczos, so TPB's 2 and 3 are rejected |
| `b_comm_size` | 1 | Basis communicator size — how many shards the determinant list is split into |
| `t_comm_size` | 1 | Task communicator size — must not exceed `b_comm_size` |
| `seed` | 1729 | Seed for the initial vector |
| `heatbath_cutoff` | 1e-4 | Heatbath expansion cutoff |
| `heatbath_truncation` | 0.0 | Weight truncation applied before heatbath expansion |
| `heatbath_batch_size` | 200000000 | Heatbath expansion batch size |

Total MPI ranks = `t_comm_size × b_comm_size × helper`, where the helper dimension is
the derived quotient and is not settable. With `b_comm_size > 1` each rank passes its
own shard of the determinant list rather than the whole list.

See [`examples/gdb/README.md`](examples/gdb/README.md) for the decomposition, the shard
contract, the six determinant-placement schemes, and why the helper dimension must be 1
on the Thrust backend.

### Diagonalization

```python
# From files
results = sbd.tpb_diag_from_files(fcidumpfile, adetfile, sbd_data,
                                   loadname="", savename="", device=None)

# From data structures
results = sbd.tpb_diag(fcidump, adet, bdet, sbd_data,
                        loadname="", savename="", device=None)
```

**Returns:** `dict` with keys `energy`, `density`, `carryover_adet`, `carryover_bdet`, `one_p_rdm`, `two_p_rdm`.

#### Getting the TPB amplitudes

Neither return dict holds the wavefunction amplitudes. Two separate mechanisms write them, in **different formats**, and the one named in the function signature is not the one most callers want:

| | `sbd_data.dump_matrix_form_wf` | `savename` argument |
|---|---|---|
| Writes | the full `len(adet) × len(bdet)` subspace, gathered onto one rank | one file per b_comm position, each holding **only that rank's block** |
| Path | exactly what you set | `f"{savename}{rank_b:06d}.bin"` |
| Header | **none** | three `size_t` (`adet_range`, `bdet_range`, `det_length`), then that block's alpha and beta determinant words |
| Payload | `float64`, row-major over α×β | `adet_range × bdet_range` `float64` |
| Intended for | analysis | checkpoint/restart, paired with `loadname` |

So the whole-subspace read is just:

```python
config.dump_matrix_form_wf = "wf.bin"
results = sbd.tpb_diag(fcidump, adet, bdet, config)

amplitudes = np.fromfile("wf.bin").reshape(len(adet), len(bdet))
```

Naming that file `.dat` or `.txt` instead writes a text table labelling every amplitude with its own alpha and beta bitstring — the easiest form to inspect by hand, and it sidesteps having to reconstruct the ordering yourself:

```
<amplitude> # <alpha index>: <alpha bitstring> <beta index>: <beta bitstring>
```

Be aware that this text form is written with the stream's default formatting, so amplitudes land with only ~6 significant digits; use the binary form when you need full `float64` precision.

Rows and columns follow `adet`/`bdet` as the diagonalizer saw them. `sort_bitarray` removes duplicates as well as sorting, so pass lists already in that form if you intend to label the axes yourself. This is the mechanism [`sbd_solver.py`](python/sbd_solver.py) uses to populate `SCIState.amplitudes`.

```python
# GDB: over an explicit list of full determinants rather than a product space
results = sbd.gdb_diag(fcidump, det, sbd_data,
                       loadname="", savename="", device=None,
                       determinant_distribution="", determinant_grid_a=0,
                       determinant_grid_b=0)
```

**Returns:** `dict` with keys `energy`, `density`, `carryover_det`, `one_p_rdm`,
`two_p_rdm`, `local_dim`, `global_dim` and `determinant_distribution`; the last three
are this rank's determinant count, the whole basis's, and the placement scheme that
ran. Each determinant is a `2 * norb`-bit configuration in which bit
`2 * i` is the occupation of spin-alpha orbital `i` and bit `2 * i + 1` that of
spin-beta orbital `i`. The determinants must be distinct; they are sorted into
SBD's canonical order internally, which `sort_bitarray` reproduces.

`det` may be a single `(ndets, words)` array or **this rank's shard of one**:
`sbd_data.b_comm_size` decides which. At `1` every rank passes the whole basis; above
`1` every rank passes its own shard, the union over b_comm positions being the basis.
Sharded input must be globally sorted and disjoint; that and the rank-layout constraints
are checked, and raise rather than silently diagonalizing the wrong subspace.
`determinant_distribution` chooses how determinants are placed across ranks (default
`equal-bra-a`), and the two grid arguments size the grid for the grid-cyclic schemes.
[`examples/gdb/README.md`](examples/gdb/README.md) has the shard contract, the
placement schemes, and which returned values are replicated versus sharded.

`gdb_diag` returns no wavefunction amplitudes — `gdb::diag` has no in-memory output for
them — and for most uses none are needed: the energy, density and RDMs come back
directly, and an iterative heatbath run gets its next subspace from `carryover_det`. If
you do want the amplitudes, `savename` writes them to disk; the file layout is in
[`examples/gdb/README.md`](examples/gdb/README.md).

The optional `device` parameter overrides the default set by `init()`.

## Troubleshooting

**GPU backends silently build as host code:** a conda compiler package
(`cxx-compiler`, `gxx_linux-64`, `clangxx_osx-*`) sets `CC`/`CXX` on activation and
the build respects a caller-set compiler, so `nvc++`/`amdclang++` never run. Unset
`CC`/`CXX`, or keep conda compilers out of the build env.

**GPU not building:** On NVIDIA check `which nvc++` and set `NVHPC_HOME` (or rely on
`NVHPC_ROOT` from `module load nvhpc`; either the compilers directory or the version
root works). On AMD
check `which amdclang++` and set `ROCM_HOME`. The build prints which toolchain it
picked (`Found amdclang++ in PATH: …` / `Found NVIDIA HPC SDK at: …`) and, for
the offload backend, the resolved architecture; on a host with both toolchains
force the choice with `SBD_GPU_VENDOR=amd|nvidia`.

**`MPI_HOME=... is not the MPI that mpi4py is linked against`:** the build stops here
deliberately rather than producing extensions that link one MPI while `mpi4py` loads
another — a mismatch that surfaces later as undefined symbols or a hang inside the first
collective. `MPI_HOME` is only for layouts the build cannot infer; unset it to use
`mpi4py`'s own MPI, or reinstall `mpi4py` against the MPI you want
(`pip install --no-binary :all: mpi4py`). To see which MPI that is:
`python -c "from mpi4py import MPI; print(MPI.Get_library_version())"`.

**`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
initialized` on macOS:** two copies of the same LLVM OpenMP runtime in one process. It
aborts at the first parallel region, so the import succeeds and the first
diagonalization dies. Usually it means the environment provides `libomp` twice — for
example Homebrew `llvm` *and* Homebrew `libomp`, or a Homebrew copy alongside the conda
env's. Build against one only; the conda env's is the one loaded at import time, so
prefer it. Tracked as
[issue #27](https://github.com/Qiskit/sbd-eigensolver-python/issues/27).

**OMP-offload runs all land on GPU 0 in multi-GPU jobs:** symptom — every MPI rank shows large memory only on GPU 0 in `nvidia-smi` (or `rocm-smi`). The bindings call `omp_set_default_device(mpi_rank % n_dev)`, but `omp_get_num_devices()` can return 0 in some dlopen scenarios. The bindings fall back to counting the entries in the vendor's device-visibility variable — `CUDA_VISIBLE_DEVICES` on NVIDIA, `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` on AMD — so make sure the relevant one is exported and lists all your GPUs (e.g. `0,1,2,3`). Slurm/`srun --gres=gpu:N` and OpenMPI's default binding policy already do this; if you've custom-restricted it to a single GPU per rank, set it manually before launch. Note the index is the **global** MPI rank, not a node-local one, so the assignment is even only when the launcher places ranks on nodes in contiguous blocks — round-robin placement leaves each node using a strided subset of its GPUs.

**Ranks die with `Bus error` or `SIGSEGV` inside the MPI's own copy path** (`MPIR_Localcopy`, `ucp_worker_progress`, ...) **on a GPU backend:** the MPI is not GPU-aware and was handed a device pointer. Rebuild UCX `--with-cuda` / `--with-rocm`, and confirm with `ucx_info -d | grep -i 'Transport: cuda'` (or `rocm`). Two things mislead here. A partly GPU-aware stack fails in only one place: an MPICH with GPU support *disabled* over a CUDA-aware UCX ran OMP-offload fine and crashed only in Thrust, because the inter-rank path went through UCX while the local-copy path did not. And on AMD a non-ROCm-aware MPI does not crash at all — ROCm maps device memory into the process address space, so the host copy succeeds and merely stages everything through the host aperture (measured on MI250X, XNACK off, 8 ranks) — so a working AMD run is not evidence that the MPI is ROCm-aware.

**GDB on more than one GPU refuses to start, naming the helper dimension:** GDB's Thrust
kernels never implemented that dimension, and it is the quotient
`ranks / (t_comm_size × b_comm_size)` — so any rank you do not assign to `t` or `b` lands
there. Give every rank to the basis: `-np 4` with `b_comm_size = 4` leaves a helper
dimension of 1. Leaving `b_comm_size` at 1 puts *all* ranks in the helper dimension,
which is why multi-GPU GDB requires a split basis. `gdb_diag` checks this before doing
any work rather than letting the kernel throw mid-launch. The CPU backend has no such
restriction, and TPB is unaffected.

**Repository:** https://github.com/Qiskit/sbd-eigensolver-python
