# SBD Python Bindings

Python bindings for the Selected Basis Diagonalization (SBD) library, with CPU and GPU backends.

## Overview

SBD (Selected Basis Diagonalization) is a high-performance library for quantum chemistry calculations. The Python bindings provide access to SBD's **Tensor-Product Basis (TPB)** diagonalization method on CPU and GPU.

**Key Features:**
- **TPB diagonalization** for quantum chemistry Hamiltonians
- Three backends, selected per call at runtime via `device=`:
  `'cpu'` (host OpenMP), `'gpu'` (NVHPC Thrust/CUDA) and `'gpu-omp'` (NVHPC
  OpenMP target offload). All three can be built into one install; each is
  imported only when first used
- MPI parallelization
- Integration with [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) for SQD workflows

In addition to TPB, this package also contains experimental support for SBD's **General-Determinant Basis (GDP)** method. However, SBD's **Creation/Annihilation operator (CAOP)** method is currently not supported by this wrapper; users who need it should reference and use the C++ CLI apps in the upstream submodule (`vendor/sbd-upstream/apps/`).

> [!NOTE]
> This package is newly open-sourced. The Python API follows semantic versioning, but the build configuration and GPU backends have been exercised on a limited set of platforms — please report issues.

## Installation

### Prerequisites

**Required:** Python 3.10+, MPI (OpenMPI/MPICH), BLAS (OpenBLAS/MKL), pybind11, mpi4py, numpy.

**For the GPU backends** — optional; without them you get a CPU-only install:

- *to build:* NVIDIA HPC SDK (`nvc++`), and `SBD_GPU_ARCH` set to the target
  compute capability (required — there is no default, see
  [Environment Variables](#environment-variables)). No GPU needs to be present
  on the build machine.
- *to run:* a CUDA-capable GPU, and an MPI **built with CUDA support** — the GPU
  paths pass device pointers to `MPI_Allreduce`, and a non-CUDA-aware MPI stalls
  there. conda-forge's `mpich`/`openmpi` are not CUDA-aware.

Either install path compiles the C++ extension on the target machine;
no pre-built wheels are published. The resulting binary depends on the
local MPI and BLAS, so both must be installed first.

### Install from PyPI

**A self-contained conda environment** is the quickest way to get those
dependencies in place for the CPU backend:
```bash
conda create -y -n sbd -c conda-forge \
    python=3.13.12 pybind11 numpy setuptools wheel openblas pyscf pip mpi4py
```

```bash
conda activate sbd
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
pip install -e . --no-build-isolation --force-reinstall --no-deps
```

### Environment Variables

```bash
# --- required ONLY for the NVIDIA GPU backends (Thrust and OpenMP target-offload) ---
#     Adjust the path.
export NVHPC_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers

# --- required whenever a NVIDIA GPU backend is built
#       A100: cc80    H100: cc90    GB200 / B200: cc100
export SBD_GPU_ARCH=cc100

# --- optional overrides; each has a working default ---
#     MPI: defaults to whatever mpi4py is linked against. Set this only
#     for layouts that cannot be inferred.
export MPI_HOME=/path/to/mpi

#     BLAS: defaults to whatever the linker finds, including a
#     conda-installed OpenBLAS in $CONDA_PREFIX/lib. Set these to select
#     a specific build (e.g. an arch-tuned OpenBLAS)
export BLAS_LIB_PATH=/path/to/blas/lib
export BLAS_LIBS=openblas          # or mkl_rt
```

### Build Using the Host MPI
```bash
# Create a conda env
conda create -y -n sbd -c conda-forge \
    python=3.13.12 pybind11 numpy setuptools wheel openblas pyscf pip

conda activate sbd                         # always activate first

# Install mpi4py against the host MPI
# For GPU backends, the host MPI must be CUDA-aware.
export MPI_HOME=/path/to/mpi
MPICC=$MPI_HOME/bin/mpicc python -m pip install --no-binary=mpi4py --no-cache-dir mpi4py
# NOTE: If you need only the CPU backend and no host MPI is available, run the command below,
# which conda will choose a compatible MPI.
# conda install -y -c conda-forge mpi4py

# confirm which MPI mpi4py uses -- setup.py builds against exactly this
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"

# only for the SQD examples (python/examples/run_sqd_sbd.py and .ipynb)
pip install "qiskit-addon-sqd>=0.13.1"

# install sbd-eigensolver
pip install sbd-eigensolver

```

#### Advanced `SBD_BUILD_BACKEND` overrides

Only needed when you want to deviate from the default.

| Value | Builds |
|---|---|
| *unset* (default) | CPU always; Thrust and OMP-offload GPU if `nvc++` found. |
| `cpu` | CPU only — skip GPU even if `nvc++` is present. |
| `gpu` | Thrust GPU only — skip CPU. Errors if `nvc++` missing. |
| `gpu_omp_offload` | OMP-offload GPU only. |

### Verify

```bash
python -c "import sbd; print(sbd.available_backends())"
# CPU only:                       ['cpu']
# NVHPC Thrust:                   ['gpu']
# OMP-offload-only install:       ['gpu-omp']
```

## Examples

Located in `python/examples/`:

- **`run_sbd_diag.py`** — Standalone TPB diagonalization (no Qiskit dependency)
- **`run_sqd_sbd.ipynb`** — Jupyter Notebook SQD loop with SBD solver (random or hardware bitstrings)
- **`run_sqd_sbd.py`** — SQD loop with SBD solver (random or hardware bitstrings)

See [python/examples/README.md](python/examples/README.md) for usage details.

## Integration with qiskit-addon-sqd

SBD can serve as the eigensolver backend for qiskit-addon-sqd's SQD workflow.

**Note:** Requires [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) with distributed (SPMD) support — `diagonalize_fermionic_hamiltonian` calling `sci_solver` on every MPI rank. This is available in `qiskit-addon-sqd` version `0.13.1` or higher.

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
    sbd_config={"method": 0, "eps": 1e-8, "max_it": 100},
    device_config=DeviceConfig.gpu(),  # or .cpu(), .gpu_omp()
)

result = diagonalize_fermionic_hamiltonian(
    hcore, eri, bit_array,
    sci_solver=sbd_solver,
    norb=norb, nelec=nelec,
    samples_per_batch=300, num_batches=3, max_iterations=5,
    symmetrize_spin=True,
)
```

See `python/examples/run_sqd_sbd.py` for a complete example.

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
| `h_comm_size` | 1 | Helper communicator size |
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

## Backend Architecture

- Each backend is a separate pybind11 module compiled from the same `python/bindings.cpp` source with different `-D` macros (`SBD_THRUST` for the Thrust path, `USE_GPU + USE_OMP_OFFLOAD` for OMP-offload, neither for CPU). The Thrust and OMP-offload paths both compile with NVHPC `nvc++` (with `-cuda` and `-mp=gpu` respectively); CPU compiles with gcc/clang. Distinct C++ namespaces — no symbol collision when multiple coexist.
- `get_backend(device)` resolves the `device=` string and returns the appropriate module; all wrapper functions accept an optional `device` parameter. Aliases for back-compat live in `sbd._device_aliases`.
- GPU device assignment: `gpu_id = mpi_rank % num_gpus` (set per `tpb_diag()` call in `bindings.cpp`); same logic for both Thrust and OMP-offload paths.
- Backends differ in which phases run on the GPU vs the host. Davidson and the matvec (`mult`) live on the GPU under both Thrust and OMP-offload. The diagonal-Hamiltonian preconditioner (`makeQChamDiagTerms`) is GPU-resident under Thrust but runs on the host under OMP-offload (no `#pragma omp target` port in `tpb/qcham.h`).

## Troubleshooting

**GPU not building:** Check `which nvc++` and set `NVHPC_HOME`.

**MPI errors:** Verify `MPI_HOME`, check `python -c "from mpi4py import MPI; print(MPI.Get_version())"`.

**OMP-offload runs all land on GPU 0 in multi-GPU jobs:** symptom — every MPI rank shows large memory only on GPU 0 in `nvidia-smi`. The bindings call `omp_set_default_device(mpi_rank % n_dev)`, but `omp_get_num_devices()` can return 0 in some dlopen scenarios. The bindings fall back to parsing `CUDA_VISIBLE_DEVICES` to recover the device count, so make sure that env var is exported and lists all your GPUs (e.g. `0,1,2,3`). Slurm/`srun --gres=gpu:N` and OpenMPI's default binding policy already do this; if you've custom-restricted `CUDA_VISIBLE_DEVICES` to a single GPU per rank, set it manually before launch.

---

**Repository:** https://github.com/Qiskit/sbd-eigensolver-python
