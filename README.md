# SBD Python Bindings

Python bindings for the Selected Basis Diagonalization (SBD) library, with CPU and GPU backends.

## Overview

SBD (Selected Basis Diagonalization) is a high-performance library for quantum chemistry calculations. The Python bindings provide access to SBD's **Tensor-Product Basis (TPB)** diagonalization method on CPU and GPU.

**Key Features:**
- **TPB diagonalization** for quantum chemistry Hamiltonians
- Three backends, selected per call at runtime via `device=`:
  `'cpu'` (host OpenMP), `'gpu'` (NVHPC Thrust/CUDA, NVIDIA only) and
  `'gpu-omp'` (OpenMP target offload, **NVIDIA or AMD**). All the backends your
  toolchain supports can be built into one install; each is imported only when
  first used
- MPI parallelization
- Integration with [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) for SQD workflows

In addition to TPB, this package also contains experimental support for SBD's **General-Determinant Basis (GDB)** method. However, SBD's **Creation/Annihilation operator (CAOP)** method is currently not supported by this wrapper; users who need it should reference and use the C++ CLI apps in the upstream submodule (`vendor/sbd-upstream/apps/`).

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

Located in [`examples/`](examples/README.md), organized by basis type since the
solvers take different subspaces and decompose over MPI differently. Each folder's
README is the authoritative list of what it contains and how to run it.

- [`examples/tpb/`](examples/tpb/README.md) — tensor-product basis: standalone TPB
  diagonalization, the SQD loops, and the subspace-enlargement driver.
- [`examples/README.md`](examples/README.md) — backend selection, `--device` values,
  bundled test data and performance notes, shared by all examples.

## Integration with qiskit-addon-sqd

SBD can serve as the eigensolver backend for qiskit-addon-sqd's SQD workflow.

**Note:** Requires [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) with distributed (SPMD) support — `diagonalize_fermionic_hamiltonian` calling `sci_solver` on every MPI rank. This is available in `qiskit-addon-sqd` version `0.13.1` or higher.

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
(sampling, configuration recovery, subsampling), and SBD is plugged in
purely as the per-batch eigensolver (`sci_solver=sbd_solver` above) with no
say in how the subspace grows between iterations.

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

Total MPI ranks = `task_comm_size × adet_comm_size × bdet_comm_size`.

```python
config = sbd.GDB_SBD()
```

Shares `method`, `max_it`, `max_nb`, `eps`, `max_time`, `init`, `do_shuffle`,
`do_rdm`, `carryover_type`, `ratio`, `threshold` and `bit_length` with `TPB_SBD`,
and replaces the determinant communicators with a single basis communicator:

| Attribute | Default | Description |
|-----------|---------|-------------|
| `b_comm_size` | 1 | Basis communicator size — must be 1 for `gdb_diag`, see below |
| `t_comm_size` | 1 | Task communicator size — must be 1 while `b_comm_size` is, see below |
| `seed` | 1729 | Seed for the initial vector |
| `heatbath_cutoff` | 1e-4 | Heatbath expansion cutoff |
| `heatbath_truncation` | 0.0 | Weight truncation applied before heatbath expansion |
| `heatbath_batch_size` | 200000000 | Heatbath expansion batch size |

**Why both must be 1**, since the two constraints have different owners:

`b_comm_size == 1` is a limitation of *this wrapper*, not of SBD. Upstream's in-memory
`gdb::diag` expects each rank to pass **its own shard** of the determinant list — that
is what upstream's file-based entry point hands it, after distributing determinant
files across `b_comm`. This wrapper passes the whole list from every rank, which is
only consistent with a single basis block, so it rejects anything else rather than
have each rank diagonalize the full subspace while believing it held a shard. Upstream
itself runs with a split basis: its own `run.sh` for the GDB app passes
`--b_comm_size 2`.

`t_comm_size == 1` then follows from *upstream's* algorithm rather than from us. GDB's
matrix-vector product rotates the ket around `b_comm` as a ring, so there are exactly
`b_comm_size` ring stations and one "task" is one station — meaning `t_comm_size`
cannot exceed `b_comm_size`. With the basis in a single block there is a single task.

Ranks are not wasted in the meantime: the derived helper dimension,
`ranks / (t_comm_size × b_comm_size)`, absorbs them and does not change the energy.

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

```python
# GDB: over an explicit list of full determinants rather than a product space
results = sbd.gdb_diag(fcidump, det, sbd_data,
                       loadname="", savename="", device=None)
```

**Returns:** `dict` with keys `energy`, `density`, `carryover_det`, `one_p_rdm`,
`two_p_rdm`. Each determinant is a `2 * norb`-bit configuration in which bit
`2 * i` is the occupation of spin-alpha orbital `i` and bit `2 * i + 1` that of
spin-beta orbital `i`. The determinants must be distinct; they are sorted into
SBD's canonical order internally, which `sort_bitarray` reproduces.

`gdb_diag` does not return the wavefunction amplitudes, because SBD's `gdb::diag`
has no in-memory output for them. Passing `savename` makes SBD write them to
`f"{savename}000000.bin"` instead: two `size_t` headers
(`n_dets`, `words_per_det`), then `n_dets × words_per_det` `size_t` determinant
words in canonical order, then `n_dets` `float64` amplitudes.

The optional `device` parameter overrides the default set by `init()`.

## Troubleshooting

**GPU backends silently build as host code:** a conda compiler package
(`cxx-compiler`, `gxx_linux-64`, `clangxx_osx-*`) sets `CC`/`CXX` on activation and
the build respects a caller-set compiler, so `nvc++`/`amdclang++` never run. Unset
`CC`/`CXX`, or keep conda compilers out of the build env.

**GPU not building:** On NVIDIA check `which nvc++` and set `NVHPC_HOME`. On AMD
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

**`GDB Thrust mult does not support h_comm_size > 1` from `gdb_diag` on more than one
rank:** GDB's Thrust kernels never implemented the helper dimension, and the helper
dimension is `ranks / (t_comm_size × b_comm_size)`. Since `gdb_diag` requires
`b_comm_size == 1` (see [Configuration](#configuration)), which forces `t_comm_size` to
1, every rank you add lands in the helper dimension — so GPU GDB is limited to a single
rank in this release. Run GDB on one GPU, or on the CPU backend, where the helper
dimension is unconstrained. TPB is unaffected and shards over `adet_comm_size` /
`bdet_comm_size` as usual.

**Repository:** https://github.com/Qiskit/sbd-eigensolver-python
