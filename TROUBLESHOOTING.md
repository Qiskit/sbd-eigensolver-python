# Troubleshooting

Build and runtime symptoms, and what to do about each, roughly from the ones that stop
a build or crash a job to the ones that run but not as intended. For installing and the
build options, see [INSTALL.md](INSTALL.md); for anything not here, please open a
[GitHub issue](https://github.com/Qiskit/sbd-eigensolver-python/issues).

## The build fails, or builds the wrong backends

**`MPI_HOME=... is not the MPI that mpi4py is linked against`:** the build stops here
deliberately rather than producing extensions that link one MPI while `mpi4py` loads
another — a mismatch that surfaces later as undefined symbols or a hang inside the first
collective. `MPI_HOME` is only for layouts the build cannot infer; unset it to use
`mpi4py`'s own MPI, or reinstall `mpi4py` against the MPI you want
(`pip install --no-binary :all: mpi4py`). To see which MPI that is:
`python -c "from mpi4py import MPI; print(MPI.Get_library_version())"`.

**GPU not building:** On NVIDIA check `which nvc++` and set `NVHPC_HOME` (or rely on
`NVHPC_ROOT` from `module load nvhpc`; either the compilers directory or the version
root works). On AMD check `which amdclang++` and set `ROCM_HOME`. The build prints which
toolchain it picked (`Found amdclang++ in PATH: …` / `Found NVIDIA HPC SDK at: …`) and,
for the offload backend, the resolved architecture; on a host with both toolchains force
the choice with `SBD_GPU_VENDOR=amd|nvidia`.

**GPU backends silently build as host code:** a conda compiler package
(`cxx-compiler`, `gxx_linux-64`, `clangxx_osx-*`) sets `CC`/`CXX` on activation and
the build respects a caller-set compiler, so `nvc++`/`amdclang++` never run. Unset
`CC`/`CXX`, or keep conda compilers out of the build env.

## A job crashes

**Ranks die with `Bus error` or `SIGSEGV` inside the MPI's own copy path**
(`MPIR_Localcopy`, `ucp_worker_progress`, ...) **on a GPU backend:** the MPI is not
GPU-aware and was handed a device pointer. Rebuild UCX `--with-cuda` / `--with-rocm`,
and confirm with `ucx_info -d | grep -i 'Transport: cuda'` (or `rocm`); for an MPI that
cannot be made GPU-aware, see [GPU-aware MPI](INSTALL.md#gpu-aware-mpi). Two things
mislead here. A partly GPU-aware stack fails in only one place: an MPICH with GPU
support *disabled* over a CUDA-aware UCX ran OMP-offload fine and crashed only in
Thrust, because the inter-rank path went through UCX while the local-copy path did not.
And on AMD a non-ROCm-aware MPI does not crash at all — ROCm maps device memory into the
process address space, so the host copy succeeds and merely stages everything through
the host aperture (measured on MI250X, XNACK off, 8 ranks) — so a working AMD run is not
evidence that the MPI is ROCm-aware.

**`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
initialized` on macOS:** two copies of the same LLVM OpenMP runtime in one process. It
aborts at the first parallel region, so the import succeeds and the first
diagonalization dies. Usually it means the environment provides `libomp` twice — for
example Homebrew `llvm` *and* Homebrew `libomp`, or a Homebrew copy alongside the conda
env's. Build against one only; the conda env's is the one loaded at import time, so
prefer it. Tracked as
[issue #27](https://github.com/Qiskit/sbd-eigensolver-python/issues/27).

## A solve refuses to start

These raise a `ValueError` on every rank before any work, so the message says what to
change.

**`... on the gpu (Thrust) backend needs more than one OpenMP thread per rank`:**
upstream SBD's Thrust build does part of each solve on the host and aborts the whole MPI
job with one thread, so the bindings refuse first. Set `OMP_NUM_THREADS` to 2 or more,
and give each rank that many cores, for example with `srun --cpus-per-task`.
Upstream's own example scripts use one thread, so this is easy to inherit from them. The
`cpu` and `gpu-omp` backends are not affected.

**`... is not supported on the OpenMP-offload backend`, `... has no CAOP solver` (or
`evaluator`), or `No UHF build of the 'gpu-omp' backend`:** `gpu-omp` runs TPB with a restricted FCIDUMP
only. For GDB, CAOP (including the Green's function and its variance) or an
unrestricted FCIDUMP, use `cpu`, or `gpu` on NVIDIA; on AMD GPUs these run on the CPU.
`device='auto'` resolves to `gpu-omp` where a GPU is present but the Thrust backend is
not built, so pass the device explicitly for these.

**GDB on more than one GPU refuses to start, naming the helper dimension:** GDB's Thrust
kernels never implemented that dimension, and it is the quotient
`ranks / (t_comm_size × b_comm_size)` — so any rank you do not assign to `t` or `b` lands
there. Give every rank to the basis: `-np 4` with `b_comm_size = 4` leaves a helper
dimension of 1. Leaving `b_comm_size` at 1 puts *all* ranks in the helper dimension,
which is why multi-GPU GDB requires a split basis. The CPU backend has no such
restriction, and TPB is unaffected.

## A job runs, but not as intended

**Every rank lands on GPU 0:** each MPI rank uses GPU `mpi_rank % n_dev`, set on every
solver call, on both the Thrust and OpenMP-offload backends. On OpenMP offload,
`omp_get_num_devices()` can return 0 in some dlopen scenarios, and the bindings then
count the entries of the vendor's device-visibility variable instead —
`CUDA_VISIBLE_DEVICES` on NVIDIA, `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` on
AMD — so make sure the relevant one is exported and lists all your GPUs (e.g.
`0,1,2,3`). Slurm's `srun --gres=gpu:N` and OpenMPI's default binding already do this;
if you restricted it to one GPU per rank yourself, set it before launch. The symptom is
large memory on GPU 0 only in `nvidia-smi` (or `rocm-smi`). Note that `mpi_rank` is the
**global** MPI rank, not a node-local one, so the assignment is even only when the
launcher places ranks on nodes in contiguous blocks; round-robin placement leaves each
node using a strided subset of its GPUs.
