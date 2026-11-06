# Troubleshooting

Build and runtime symptoms, and what to do about each. For installing and the build
options, see [INSTALL.md](INSTALL.md); for anything not here, please open a
[GitHub issue](https://github.com/Qiskit/sbd-eigensolver-python/issues).

**How GPUs are assigned:** each MPI rank uses GPU `mpi_rank % num_gpus`, set on every
solver call, on both the Thrust and OpenMP-offload backends. `mpi_rank` is the global
rank, not a node-local one, so run one rank per GPU with ranks placed on nodes in
contiguous blocks.

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
