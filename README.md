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

### Prerequisites

**Required:** Python 3.10+, MPI (OpenMPI/MPICH), BLAS (OpenBLAS/MKL), pybind11, mpi4py, numpy, compiler with OpenMP.

**On macOS:** Apple clang ships without OpenMP, so add `llvm-openmp` to the conda
environment. Homebrew's `libomp` is used as a fallback if the env has none. Pin the
compiler with `CC`/`CXX` as well — a bare `clang++` is resolved through `PATH`, so a
Homebrew LLVM silently wins over both Apple clang and a conda toolchain. The build
prints which compiler and which libomp it chose.

**For the GPU backends** — optional; without them you get a CPU-only install:

***On NVIDIA*** (both the Thrust and the OpenMP-offload backend):

- *to build:* [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) (`nvc++`).
  `SBD_GPU_ARCH` is **optional**: unset, nvc++ targets the GPU of the machine
  the toolchain was installed on; set, it is honored exactly and may name
  several generations at once (`cc80,cc90,cc100`) — see
  [Environment Variables](#environment-variables). 

***On AMD*** (the OpenMP-offload backend only):

- *to build:* ROCm LLVM toolchain (`amdclang++`).
  `SBD_GPU_ARCH` is optional here on a GPU system and is **detected** with
  ROCm's `amdgpu-arch` (e.g. `gfx90a` on MI250X, `gfx942` on MI300X). However, on a GPU-less build host, you must
  set it. see [Environment Variables](#environment-variables).

Either install path compiles the C++ extension on the target machine;
no pre-built wheels are published. The resulting binary depends on the
local MPI and BLAS, so both must be installed first.

### Install from PyPI

**A self-contained conda environment** is the quickest way to get those
dependencies in place for the CPU backend:
```bash
conda create -y -n sbd -c conda-forge \
    python=3.13.12 pybind11 numpy setuptools wheel openblas pyscf pip mpi4py
#   ...plus llvm-openmp on macOS
```

```bash
conda activate sbd
```

Now install the sbd-eigensolver-python package
```
pip install sbd-eigensolver
```

The published source distribution (sdist) bundles the sbd header files, so this
needs no git checkout and no submodule step. 

### Install from git checkout

Here the headers come from upstream
[r-ccs-cms/sbd](https://github.com/r-ccs-cms/sbd) via a git submodule at
`vendor/sbd-upstream/`, **pinned at a specific upstream commit**. Run `git submodule status` to see the pinned SHA.

When installing from a git checkout, it is important to make sure the
sbd submodule is cloned, too:

```bash
git clone --recurse-submodules https://github.com/Qiskit/sbd-eigensolver-python.git
# or, if you cloned without --recurse-submodules:
git submodule update --init --recursive
```

If you need a newer upstream revision (for a recently-landed GPU fix
etc.), advance the local submodule and rebuild:

```bash
git submodule update --remote vendor/sbd-upstream
```

After the upstream sbd code is cloned, run:
```
pip install -e . --no-build-isolation --force-reinstall --no-deps
```

### Environment Variables

```bash
# --- NVIDIA GPU backends (Thrust and OpenMP target-offload): point at NVHPC.
#     Only needed if nvc++ is not already on PATH. Adjust the path.
export NVHPC_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers

# --- AMD GPU backend (OpenMP target-offload): point at ROCm LLVM toolchain
#     amdclang++ is found on $ROCM_HOME/bin
export ROCM_HOME=/opt/rocm

# --- OPTIONAL. Only for a host with BOTH GPU toolchains installed, where the
#     auto-answer would be an accident of probe order: nvidia | amd | none
export SBD_GPU_VENDOR=amd

# --- OPTIONAL GPU architecture, spelled per vendor.
#     NVIDIA: unset, nvc++ targets the GPU of the machine this toolchain was
#       installed on. Set it to pin the target, including SEVERAL at once:
#         A100: cc80    H100: cc90    GB200 / B200: cc100
#         ccall-major   one target per major generation
#     AMD: unset, the arch is DETECTED with ROCm's `amdgpu-arch`. Set it to pin,
#       or when building on a host with no GPU (where detection cannot work and
#       the build stops asking for it):
#         MI250X: gfx90a    MI300X: gfx942    (several: gfx90a,gfx942)
export SBD_GPU_ARCH=cc80,cc90,cc100     # NVIDIA
export SBD_GPU_ARCH=gfx90a              # AMD

# --- optional overrides; each has a working default ---
#     Which backends to build: defaults to CPU always, plus every GPU backend the
#     detected toolchain supports -- Thrust AND OpenMP-offload under nvc++,
#     OpenMP-offload only under amdclang++ (there is no rocThrust path).
#     Set it only to narrow that:
#       cpu               CPU only -- skip GPU even if a GPU compiler is present
#       gpu               Thrust GPU only, no CPU -- NVIDIA only, errors on AMD
#       gpu_omp_offload   OpenMP target-offload GPU only (either vendor)
export SBD_BUILD_BACKEND=cpu

#     MPI: defaults to whatever mpi4py is linked against. Set this only
#     for layouts that cannot be inferred.
#     NOTE: Every GPU backend hands MPI device pointers, so use a GPU-aware MPI:
#           CUDA-aware on NVIDIA, ROCm-aware on AMD.
export MPI_HOME=/path/to/mpi

#     BLAS: defaults to whatever the linker finds, including a
#     conda-installed OpenBLAS in $CONDA_PREFIX/lib. Set these to select
#     a specific build (e.g. an arch-tuned OpenBLAS)
export BLAS_LIB_PATH=/path/to/blas/lib
export BLAS_LIBS=openblas          # or mkl_rt

# these are read while COMPILING, so set them first, then install
pip install sbd-eigensolver                       # from PyPI
pip install -e . --no-build-isolation --no-deps   # from a git checkout
```

### Build Using the Host MPI
```bash
# Create a conda env
conda create -y -n sbd -c conda-forge \
    python=3.13.12 pybind11 numpy setuptools wheel openblas pyscf pip
#   ...plus llvm-openmp on macOS

conda activate sbd                         # always activate first

# Install mpi4py against the host MPI
# For any GPU backend the host MPI must be GPU-aware:
# CUDA-aware on NVIDIA, ROCm-aware on AMD.
export MPI_HOME=/path/to/mpi
MPICC=$MPI_HOME/bin/mpicc python -m pip install --no-binary=mpi4py --no-cache-dir mpi4py
# NOTE: If no host MPI is available, let conda pick a compatible one with the
# command below. A default conda-forge MPI is not GPU-aware, which is fine for the
# CPU backend; for the GPU backends either install mpi4py against a GPU-aware MPI
# as above, or see "Backend Architecture" below for the two build-time options that
# let the Thrust backend run without one.
# conda install -y -c conda-forge mpi4py

# confirm which MPI mpi4py uses -- setup.py builds against exactly this
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"

# only for the SQD examples (python/examples/run_sqd_sbd.py and .ipynb)
pip install "qiskit-addon-sqd>=0.13.1"

# install sbd-eigensolver
pip install sbd-eigensolver

```

### Verify

```bash
python -c "import sbd; print(sbd.available_backends())"
# CPU only:                       ['cpu']
# NVIDIA, default build:          ['cpu', 'gpu', 'gpu-omp']
# AMD, default build:             ['cpu', 'gpu-omp']
# OMP-offload-only install:       ['gpu-omp']
```

On a GPU build, confirm which vendor and architecture the offload backend
targets — one `'gpu-omp'` device serves both vendors, so the name alone does not
say:

```bash
python -c "import sbd; print(sbd.get_backend('gpu-omp').__sbd_offload_target__)"
# amdgcn-amd-amdhsa:gfx90a        AMD MI250X
# nvptx64-nvidia-cuda:cc90        NVIDIA H100
```

The Thrust backend is stamped too (`cuda:cc90`); the CPU backend reports `None`.

## Examples

Located in `python/examples/`:

- [`run_sbd_diag.py`](python/examples/run_sbd_diag.py) — Standalone TPB diagonalization (no Qiskit dependency)
- [`run_sqd_sbd.ipynb`](python/examples/run_sqd_sbd.ipynb) — Jupyter Notebook SQD loop with SBD solver (random or hardware bitstrings)
- [`run_sqd_sbd.py`](python/examples/run_sqd_sbd.py) — SQD loop with SBD solver (random or hardware bitstrings)
- [`run_sqd_enlarge_subspace_sbd.py`](python/examples/run_sqd_enlarge_subspace_sbd.py) — SQD that also grows its own subspace between rounds via single excitations

See [python/examples/README.md](python/examples/README.md) for usage details.

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

See [SQD Parameters](python/examples/README.md#sqd-parameters) for how each
parameter feeds the loop, and
[python/examples/run_sqd_sbd.py](python/examples/run_sqd_sbd.py) for a
complete example.

qiskit-addon-sqd is the orchestrator in that recipe: it owns the loop
(sampling, configuration recovery, subsampling), and SBD is plugged in
purely as the per-batch eigensolver (`sci_solver=sbd_solver` above) with no
say in how the subspace grows between iterations.

### SQD with subspace enlargement

[`run_sqd_enlarge_subspace_sbd.py`](python/examples/run_sqd_enlarge_subspace_sbd.py)
builds on the same recipe, but grows its own subspace between rounds: after
each solve, it expands the dominant determinant pairs via qiskit-addon-sqd's
own `enlarge_batch_from_transitions` (same-spin single excitations, both
alpha and beta) and feeds the result forward as the next round's
`include_configurations`. Concretely, it calls
`diagonalize_fermionic_hamiltonian` with `max_iterations=1` itself, in its
own outer Python loop, rather than delegating the whole multi-iteration loop
to one call — that is what makes injecting a step between rounds possible.

On the bundled H2O pool ([`count_dict_h2o.json`](python/examples/count_dict_h2o.json),
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
| `b_comm_size` | 1 | Basis communicator size (must be 1 for `gdb_diag`) |
| `t_comm_size` | 1 | Task communicator size |
| `seed` | 1729 | Seed for the initial vector |
| `heatbath_cutoff` | 1e-4 | Heatbath expansion cutoff |
| `heatbath_truncation` | 0.0 | Weight truncation applied before heatbath expansion |
| `heatbath_batch_size` | 200000000 | Heatbath expansion batch size |

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

**MPI errors:** Verify `MPI_HOME`, check `python -c "from mpi4py import MPI; print(MPI.Get_version())"`.

**OMP-offload runs all land on GPU 0 in multi-GPU jobs:** symptom — every MPI rank shows large memory only on GPU 0 in `nvidia-smi` (or `rocm-smi`). The bindings call `omp_set_default_device(mpi_rank % n_dev)`, but `omp_get_num_devices()` can return 0 in some dlopen scenarios. The bindings fall back to counting the entries in the vendor's device-visibility variable — `CUDA_VISIBLE_DEVICES` on NVIDIA, `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` on AMD — so make sure the relevant one is exported and lists all your GPUs (e.g. `0,1,2,3`). Slurm/`srun --gres=gpu:N` and OpenMPI's default binding policy already do this; if you've custom-restricted it to a single GPU per rank, set it manually before launch.

**Ranks die with `Bus error` or `SIGSEGV` inside the MPI's own copy path** (`MPIR_Localcopy`, `ucp_worker_progress`, ...) **on a GPU backend:** the MPI is not GPU-aware and was handed a device pointer. Rebuild UCX `--with-cuda` / `--with-rocm`, and confirm with `ucx_info -d | grep -i 'Transport: cuda'` (or `rocm`). Two things mislead here. A partly GPU-aware stack fails in only one place: an MPICH with GPU support *disabled* over a CUDA-aware UCX ran OMP-offload fine and crashed only in Thrust, because the inter-rank path went through UCX while the local-copy path did not. And on AMD a non-ROCm-aware MPI does not crash at all — ROCm maps device memory into the process address space, so the host copy succeeds and merely stages everything through the host aperture (measured on MI250X, XNACK off, 8 ranks) — so a working AMD run is not evidence that the MPI is ROCm-aware.

**Repository:** https://github.com/Qiskit/sbd-eigensolver-python
