# Installing sbd-eigensolver

Every install compiles the extension modules on the target machine — no wheels are
published — so which backends you end up with depends on the toolchain the build
finds. The quickstart in the [README](README.md#installation) covers the common CPU
case; this page is the full matrix: prerequisites, the environment variables the build
reads, building against a host MPI, and how to verify what you got.

## Prerequisites

**Required:** Python 3.11+, MPI (OpenMPI/MPICH), BLAS (OpenBLAS/MKL), pybind11, mpi4py, numpy, compiler with OpenMP.

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

## Install from PyPI

The build links the local MPI and BLAS, so install them first. A conda environment is
the quickest way, for the CPU backend:

```bash
conda create -y -n sbd -c conda-forge \
    python=3.13.12 pybind11 numpy setuptools wheel openblas pyscf pip mpi4py
#   ...plus llvm-openmp on macOS
conda activate sbd
pip install sbd-eigensolver
```

The source distribution bundles upstream's headers, so this needs no git checkout.

## Install from git checkout

Here upstream [r-ccs-cms/sbd](https://github.com/r-ccs-cms/sbd) is a git submodule at
`vendor/sbd-upstream/`, pinned at one commit (`git submodule status` shows it), so
clone it too:

```bash
git clone --recurse-submodules https://github.com/Qiskit/sbd-eigensolver-python.git
# or, if you cloned without --recurse-submodules:
git submodule update --init --recursive
```

Then install:

```bash
pip install -e . --no-build-isolation --force-reinstall --no-deps
```

For a newer upstream revision, `git submodule update --remote vendor/sbd-upstream` and
reinstall. Only the pinned commit is tested; with any other, the functionality may be
broken.

## Environment Variables

```bash
# --- NVIDIA GPU backends (Thrust and OpenMP target-offload): point at NVHPC.
#     Only needed if nvc++ is not already on PATH. Adjust the path.
#     Either the compilers directory or the version root above it works, and
#     NVHPC_ROOT -- which NVIDIA's own modulefile sets to the version root -- is
#     read too, so `module load nvhpc` alone is enough.
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
#     Which backends are built: always the CPU backend, plus every GPU backend
#     the detected toolchain supports -- Thrust AND OpenMP-offload under nvc++,
#     OpenMP-offload only under amdclang++ (there is no rocThrust path). The CPU
#     and Thrust backends are each also built with upstream's -D_UHF, for
#     unrestricted (UHF) FCIDUMPs. To leave GPU backends out, name them here,
#     comma-separated (the CPU backend cannot be skipped):
#       gpu               the Thrust GPU backend (alias gpu_thrust)
#       gpu_omp_offload   the OpenMP target-offload GPU backend
#     The build ends with a summary of what was built, and why any GPU backend
#     was not. (SBD_BUILD_BACKEND, which chose one set instead, still works for
#     now but is deprecated.)
export SBD_SKIP_BACKENDS=gpu_omp_offload

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
# as above, or see "GPU-aware MPI" below for the two build-time options that let the
# Thrust backend run without one.
# conda install -y -c conda-forge mpi4py

# confirm which MPI mpi4py uses -- setup.py builds against exactly this
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"

# only for the SQD examples (examples/tpb/run_sqd_sbd.py and .ipynb)
pip install "qiskit-addon-sqd>=0.13.1"

# install sbd-eigensolver
pip install sbd-eigensolver

```

## GPU-aware MPI

**The GPU backends hand MPI device pointers directly**, so they expect a GPU-aware MPI:
CUDA-aware on NVIDIA, ROCm-aware on AMD. For an MPI that cannot address device memory,
the Thrust backend has two upstream fallbacks. Both are **off by default**, and both are
**compile-time**: set them before installing, rebuild to change them, and they do
nothing if set at run time:

```
SBD_NON_CUDA_AWARE_MPI=1         stage every device buffer through host memory
SBD_THRUST_SAFE_MPI_ALLREDUCE=1  stage only the allreduce (a subset of the above)
```

Both stage device buffers through host memory, adding copies whose cost grows with how
much data crosses MPI. Treat them as a compatibility fallback, not a tuning knob: reach
for them when a GPU backend crashes inside the MPI itself rather than in SBD (see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md)). On AMD, a non-ROCm-aware MPI usually still
runs, slowly, so a working AMD run does not show that the MPI is ROCm-aware.

## Verify

```bash
python -c "import sbd; print(sbd.available_backends())"
# CPU only:                       ['cpu']
# NVIDIA, default build:          ['cpu', 'gpu', 'gpu-omp']
# AMD, default build:             ['cpu', 'gpu-omp']
```

For what each of these runs (TPB, GDB, CAOP, UHF FCIDUMPs, and where the variance
analysis runs), see the [README's backend table](README.md#backends).

The `cpu` and `gpu` backends are also built with upstream's `-D_UHF`
(`sbd._core_cpu_uhf`, `sbd._core_gpu_thrust_uhf`), which a solve uses by itself for an
unrestricted FCIDUMP; `gpu-omp` has none. `sbd.print_info()` lists them:

```bash
python -c "import sbd; sbd.print_info()"
# Compiled backends: cpu, gpu, gpu-omp
# UHF builds: cpu, gpu
```

On a GPU build, confirm which vendor and architecture the offload backend
targets — one `'gpu-omp'` device serves both vendors, so the name alone does not
say:

```bash
python -c "import sbd; print(sbd.get_backend('gpu-omp').__sbd_offload_target__)"
# amdgcn-amd-amdhsa:gfx90a        AMD MI250X
# nvptx64-nvidia-cuda:cc90        NVIDIA H100
```

The Thrust backend is stamped too (`cuda:cc90`); the CPU backend reports `None`. For the
architectures compiled into the binaries themselves:

```bash
cuobjdump --list-elf <the built _core_gpu_thrust*.so>          # NVIDIA
llvm-objdump --offloading <the built _core_gpu_omp_offload*.so>   # AMD
```


## See Also

- [README](README.md) — what the package is, the backends, and the SQD integration
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — build and runtime symptoms
