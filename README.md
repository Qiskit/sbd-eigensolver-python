# SBD Python Bindings

Python bindings for the Selected Basis Diagonalization (SBD) library with dual CPU/GPU backend support.

## Overview

SBD (Selected Basis Diagonalization) is a high-performance library for quantum chemistry calculations. The Python bindings provide access to SBD's **Tensor-Product Basis (TPB)** diagonalization method with support for both CPU and GPU backends.

**Key Features:**
- **TPB diagonalization** for quantum chemistry Hamiltonians
- Dual backend: CPU (OpenMP) and GPU (CUDA), switchable at runtime
- MPI parallelization
- Integration with [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) for SQD workflows

In addition to TPB, this package also contains experimental support for SBD's **General-Determinant Basis (GDP)** method. However, SBD's **Creation/Annihilation operator (CAOP)** method is currently not supported by this wrapper; users who need it should reference and use the C++ CLI apps in the upstream submodule (`vendor/sbd-upstream/apps/`).

> [!NOTE]
> This package is newly open-sourced. The Python API follows semantic versioning, but the build configuration and GPU backends have been exercised on a limited set of platforms — please report issues.

## Installation

### Prerequisites

**Required:** Python 3.10+, MPI (OpenMPI/MPICH), BLAS (OpenBLAS/MKL), pybind11, mpi4py, numpy.

**Optional (GPU):** NVIDIA HPC SDK (nvc++), CUDA-capable GPU, CUDA-aware MPI.

Either install path compiles the C++ extension on the target machine;
no pre-built wheels are published.  The resulting binary depends on
the local MPI and BLAS, so an MPI toolchain (`mpicc` on `PATH`, or
`MPI_HOME` set) and BLAS must be installed first; see
[Environment Variables](#environment-variables).

### Install from PyPI

```bash
pip install sbd-eigensolver
```

The published source distribution (sdist) bundles the sbd header
files, so this needs no git checkout and no submodule step.

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
# --- always needed ---
export MPI_HOME=/path/to/mpi
export BLAS_LIB_PATH=/path/to/blas/lib
export BLAS_LIBS=openblas  # or mkl_rt

# --- macOS only — pin the host compiler to system clang so libc++
#     matches Python's. Not needed on Linux GPU boxes (setup.py
#     auto-picks nvc++ for the GPU extensions; the CPU extension
#     then compiles cleanly under nvc++ as well).
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

# --- GPU backends (Thrust and OpenMP target-offload) — NVIDIA-only.
#     Both compile with NVHPC nvc++; there is no AMD/Intel GPU path
#     in this wrapper. Requires the NVHPC SDK (one tarball; no
#     LLVM/clang source build needed since v1.6).
export NVHPC_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers
# Target GPU arch for nvc++ -gpu=<arch>. Used by BOTH the Thrust path
# and the OMP-offload path. nvc++ accepts cc<XX> (documented PGI form)
# and sm_<XX>. Default cc90 for H100.
#   H100:         cc90
#   GB200 / B200: cc100
#   A100:         cc80
export SBD_GPU_ARCH=cc100
```

You do **not** need to set `CC=nvc` / `CXX=nvc++` or clear `CFLAGS` /
`CXXFLAGS` manually for the GPU builds — setup.py auto-routes those
extensions through nvc++ and filters the RHEL 9 sysconfig flags that
nvc++ rejects. (Earlier versions required this; the v1.6 refactor
moved it into `setup.py:_route_build_through_nvhpc()`.) If you DO
set them, your values win — distutils' `os.environ.setdefault`
semantics.

The OpenMP target-offload backend cannot share a Python process with
CPU or Thrust. They link incompatible OpenMP runtimes (NVHPC's
`libnvomp` for OMP-offload, `libgomp`/`libomp` for CPU and Thrust),
and `import sbd` eagerly loads every `_core_*.so` it finds in
`python/`. Co-resident `.so` files therefore pull both runtimes into
the same address space, producing "Another OpenMP runtime library has
been detected" and potentially deadlocking at the first `#pragma omp`
region. **The cleanest setup is two venvs with two checkouts** — one
for CPU + Thrust, one for OMP-offload. Within a single venv you can
also switch profiles by removing the other profile's `_core_*.so`
before rebuilding (only files present in `python/` get loaded), but a
second venv avoids the bookkeeping.

### Build

Pick **one** of two installation profiles. They produce mutually
incompatible Python processes (different OpenMP runtimes — see the
paragraph above the "Build" section) so put them in **separate venvs
backed by separate checkouts** if you need both. The `source …`
lines below are not optional: forgetting to activate the right venv
before `pip install -e .` either puts the build into the wrong venv
or fails outright.

> **Note — `setuptools` in the setup line:** because the build uses
> `--no-build-isolation`, it relies on the build tools already present
> in the active venv. Python 3.12+ no longer seeds `setuptools` into
> fresh venvs, so it must be installed explicitly (it is included in
> the `pip install … setuptools wheel` line below). Without it the
> build fails with `Cannot import 'setuptools.build_meta'`.

**Profile 1 — CPU + Thrust GPU** (the common case)

```bash
# one-time setup
git clone --recurse-submodules https://github.com/Qiskit/sbd-eigensolver-python.git  sbd-thrust
python -m venv  ~/venvs/sbd-thrust
source ~/venvs/sbd-thrust/bin/activate
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py pybind11 numpy setuptools wheel

# build (re-run after pulling main)
source ~/venvs/sbd-thrust/bin/activate            # always activate first
cd sbd-thrust
SBD_GPU_ARCH=cc100  pip install -e . --no-build-isolation
```

Builds the CPU backend always. Adds the Thrust GPU backend when NVHPC
`nvc++` is on PATH; otherwise CPU-only. Devices: `'cpu'` and `'gpu'`.

**Profile 2 — OpenMP target-offload GPU** (use a **separate** venv +
checkout from Profile 1)

```bash
# one-time setup
git clone --recurse-submodules https://github.com/Qiskit/sbd-eigensolver-python.git  sbd-omp-offload
python -m venv  ~/venvs/sbd-omp-offload
source ~/venvs/sbd-omp-offload/bin/activate
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py pybind11 numpy setuptools wheel

# build (re-run after pulling main)
source ~/venvs/sbd-omp-offload/bin/activate       # always activate first
cd sbd-omp-offload
SBD_BUILD_BACKEND=gpu_omp_offload  SBD_GPU_ARCH=cc100 \
    pip install -e . --no-build-isolation
```

Builds the OMP-offload GPU backend only. Device: `'gpu-omp'`.

After both profiles are installed, switching is a one-liner
(`source ~/venvs/<which>/bin/activate`) — no rebuild needed.

**Multi-arch fat binary**: comma-separate the arches:
`SBD_GPU_ARCH=cc80,cc90,cc100`. nvc++ embeds one SASS cubin per arch
and picks the matching one at runtime.

#### Advanced `SBD_BUILD_BACKEND` overrides

Only needed when you want to deviate from the two profiles above.

| Value | Builds |
|---|---|
| *unset* (default) | CPU always; Thrust GPU if `nvc++` found. The "Profile 1" default. |
| `cpu` | CPU only — skip GPU even if `nvc++` is present. |
| `gpu` | Thrust GPU only — skip CPU. Errors if `nvc++` missing. |
| `both` | CPU + Thrust GPU. Errors instead of falling back if `nvc++` missing. |
| `gpu_omp_offload` | OMP-offload GPU only. The "Profile 2" install. |

**Reverting to the LLVM/clang offload path:** prior versions of this
repo supported a separate `_core_gpu_omp_nvidia` backend built with
LLVM-with-NVPTX clang. That path was removed in v1.6 to reduce the
software prereq surface (LLVM trunk had to be source-built, NVHPC's
nvc++ does not). The tag `v1.5.0-llvm` preserves the last revision
with that backend; check it out and follow the `SETUP_LLVM_OFFLOAD.txt`
recipe there if you need the clang path back.

### Verify

```bash
python -c "import sbd; print(sbd.available_backends())"
# CPU only:                       ['cpu']
# CPU + NVHPC Thrust:             ['cpu', 'gpu']
# OMP-offload-only install:       ['gpu-omp']
```

## Usage

### Quick Start

```python
import sbd

# No explicit init() needed — auto-initializes on first use
config = sbd.TPB_SBD()
config.adet_comm_size = 2
config.bdet_comm_size = 2
config.max_it = 100
config.eps = 1e-4

results = sbd.tpb_diag_from_files(
    fcidumpfile='data/h2o/fcidump.txt',
    adetfile='data/h2o/h2o-1em4-alpha.txt',
    sbd_data=config,
)

print(f"Energy: {results['energy']:.10f} Hartree")
sbd.finalize()
```

### Runtime backend switching

Compatible backends coexist as separate `_core_*.so` modules and load
at `import sbd` into independent pybind11 namespaces. CPU + Thrust GPU
can co-load; the OMP-offload backend cannot (different OpenMP runtime —
see the build section). Pick one per call with the `device` parameter:

```python
import sbd

# All compiled backends are auto-loaded
sbd.available_backends()
# CPU + Thrust install:         ['cpu', 'gpu']
# OMP-offload-only install:     ['gpu-omp']

# Per-call override — auto-initializes on first use
result_cpu     = sbd.tpb_diag(..., device='cpu')
result_thrust  = sbd.tpb_diag(..., device='gpu')
result_omp     = sbd.tpb_diag(..., device='gpu-omp')

# Or set a default device explicitly (optional)
sbd.init(device='gpu')      # default = NVHPC Thrust
result = sbd.tpb_diag(...)

# Or get the backend module directly
backend = sbd.get_backend('gpu-omp')
fcidump = backend.LoadFCIDump('fcidump.txt')
```

In `auto` mode (the default), `_resolve_device('auto')` picks the first
compiled GPU backend in the order Thrust → OMP-offload → CPU.

### Resource Cleanup

```python
results = sbd.tpb_diag_from_files(...)
sbd.finalize()  # optional — syncs GPU and resets state
```

`finalize()` calls `cudaDeviceSynchronize()` on GPU backends and resets Python state. It does **not** call `cudaDeviceReset()` (avoids CUDA-aware MPI conflicts) or `MPI_Finalize()` (handled by mpi4py).

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

### Comparison with qiskit-addon-dice-solver

| Feature | dice-solver | SBD |
|---------|------------|-----|
| Solver | DICE (subprocess) | SBD (in-process) |
| GPU | No | Yes (CUDA) |
| MPI | Spawns processes | Direct integration |
| I/O | Temp files | In-memory |

## Examples

Located in `python/examples/`:

- **`run_sbd_diag.py`** — Standalone TPB diagonalization (no Qiskit dependency)
- **`run_sqd_sbd.py`** — SQD loop with SBD solver (random or hardware bitstrings)

See [python/examples/README.md](python/examples/README.md) for usage details.

## API Reference

### Initialization

| Function | Description |
|----------|-------------|
| `sbd.init(device, comm_backend)` | **Optional.** Initialize MPI, set default device (`'cpu'`, `'gpu'`, `'auto'`). Auto-called on first use with defaults. |
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

### Utilities

```python
fcidump = sbd.LoadFCIDump("fcidump.txt", device=None)
dets = sbd.LoadAlphaDets("alphadets.txt", bit_length, total_bit_length, device=None)
string = sbd.makestring(det, bit_length, total_bit_length, device=None)
det = sbd.from_string(s, bit_length, total_bit_length, device=None)
dets = sbd.sort_bitarray(dets, device=None)
sbd.print_info()
```

## Backend Architecture

- Each backend is a separate pybind11 module compiled from the same `python/bindings.cpp` source with different `-D` macros (`SBD_THRUST` for the Thrust path, `USE_GPU + USE_OMP_OFFLOAD` for OMP-offload, neither for CPU). The Thrust and OMP-offload paths both compile with NVHPC `nvc++` (with `-cuda` and `-mp=gpu` respectively); CPU compiles with gcc/clang. Distinct C++ namespaces — no symbol collision when multiple coexist.
- `get_backend(device)` resolves the `device=` string and returns the appropriate module; all wrapper functions accept an optional `device` parameter. Aliases for back-compat live in `sbd._device_aliases`.
- GPU device assignment: `gpu_id = mpi_rank % num_gpus` (set per `tpb_diag()` call in `bindings.cpp`); same logic for both Thrust and OMP-offload paths.
- Backends differ in which phases run on the GPU vs the host. Davidson and the matvec (`mult`) live on the GPU under both Thrust and OMP-offload. The diagonal-Hamiltonian preconditioner (`makeQChamDiagTerms`) is GPU-resident under Thrust but runs on the host under OMP-offload (no `#pragma omp target` port in `tpb/qcham.h`).

## Troubleshooting

**`ImportError` on macOS (symbol not found):** Python's libc++ and Homebrew clang's libc++ may differ. Use system clang: `CC=/usr/bin/clang CXX=/usr/bin/clang++`.

**`ImportError: _core_cpu`:** Backend not built. Rebuild: `pip install -e . --no-build-isolation -v`

**GPU not building:** Check `which nvc++` and set `NVHPC_HOME`.

**MPI errors:** Verify `MPI_HOME`, check `python -c "from mpi4py import MPI; print(MPI.Get_version())"`.

**OMP-offload runs all land on GPU 0 in multi-GPU jobs:** symptom — every MPI rank shows large memory only on GPU 0 in `nvidia-smi`. The bindings call `omp_set_default_device(mpi_rank % n_dev)`, but `omp_get_num_devices()` can return 0 in some dlopen scenarios. The bindings fall back to parsing `CUDA_VISIBLE_DEVICES` to recover the device count, so make sure that env var is exported and lists all your GPUs (e.g. `0,1,2,3`). Slurm/`srun --gres=gpu:N` and OpenMPI's default binding policy already do this; if you've custom-restricted `CUDA_VISIBLE_DEVICES` to a single GPU per rank, set it manually before launch.

**OMP-offload + UCX MPI fails with `MPI_INIT failed`:** mpi4py 4.x requests `MPI_THREAD_MULTIPLE` by default, which UCX in HPCX rejects with `UCP worker does not support MPI_THREAD_MULTIPLE`. Set `MPI4PY_RC_THREAD_LEVEL=serialized` (or `funneled`/`single`) in the environment, or `import mpi4py; mpi4py.rc.thread_level = 'serialized'` before `from mpi4py import MPI`.

## Performance Tips

**CPU:** `OMP_NUM_THREADS` = cores per MPI rank.
**GPU (Thrust):** 1 rank per GPU, `OMP_NUM_THREADS=1`, use method 0 (matrix-free Davidson).
**GPU (OMP-offload):** 1 rank per GPU, `OMP_NUM_THREADS` ≈ socket-local cores per rank, **pin each rank to one socket** (e.g. `mpirun --map-by ppr:N:socket --bind-to socket …` or wrap with `numactl --cpunodebind=… --membind=…`). Without pinning, the host-side `makeQChamDiagTerms` loop and the host-side orchestration inside Davidson degrade ~7× and 2–3× respectively because OMP threads thrash across NUMA nodes.

---

**Repository:** https://github.com/Qiskit/sbd-eigensolver-python
