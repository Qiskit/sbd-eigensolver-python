# Installing sbd-eigensolver

Every install compiles the extension modules on the target machine — no wheels are
published — so which backends you end up with depends on the toolchain the build
finds. The quickstart in the [README](README.md#installation) covers the common CPU
case; this page is the full matrix: prerequisites, the environment variables the build
reads, building against a host MPI, and how to verify what you got.

## Prerequisites

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

## Install from PyPI

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

## Install from git checkout

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

## Environment Variables

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

## Build Using the Host MPI
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
# as above, or see the README's Backend Architecture section for the two build-time options that
# let the Thrust backend run without one.
# conda install -y -c conda-forge mpi4py

# confirm which MPI mpi4py uses -- setup.py builds against exactly this
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"

# only for the SQD examples (examples/tpb/run_sqd_sbd.py and .ipynb)
pip install "qiskit-addon-sqd>=0.13.1"

# install sbd-eigensolver
pip install sbd-eigensolver

```

## Verify

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

## See Also

- [README](README.md) — what the package is, the API, and the SQD integration
- [`examples/tpb/README.md`](examples/tpb/README.md) — backend selection at runtime,
  bundled test data and performance notes
- [Troubleshooting](README.md#troubleshooting) — build and runtime symptoms
