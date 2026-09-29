# SBD examples

Runtime behaviour shared by every example: which backend you get, what data is
bundled, and how to spend threads and GPUs. Solver-specific usage lives with the
examples themselves:

- [`tpb/`](tpb/README.md) — TPB (tensor-product basis) and the SQD loops built on
  it. The subspace is the Cartesian product of an alpha and a beta determinant list.

- **Communication:** MPI for distributed computing
- **Backends:** CPU (host OpenMP, `--device cpu`), GPU (NVHPC Thrust, NVIDIA only,
  `--device gpu`) and GPU (OpenMP target offload, NVIDIA and AMD, `--device
  gpu-omp`), switchable at runtime via the `device` parameter

Replace `--device gpu` with `--device gpu-omp` if you use AMD GPUs.

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
# result_omp = sbd.tpb_diag(..., device='gpu-omp')   # do NOT mix with 'cpu' — see below
```

**`'cpu'` and `'gpu-omp'` must not be used in the same process.** Both link the same
OpenMP runtime, and `_core_cpu` is built without offload support, so whichever loads
first initializes that runtime host-only. If it is the CPU backend, the OMP-offload
backend can no longer acquire a device and **silently runs its target regions on the
host**: correct energies, exit status 0, and the GPU sitting idle. There is no error to
catch, which is why it is worth knowing rather than discovering. `sbd.loaded_backends()`
reports what the current process has actually imported.

`'cpu'` and `'gpu'` (Thrust) *can* share a process — Thrust does not route its device
work through OpenMP, so it has no equivalent interaction.

## Available Test Data

Paths below are written as the drivers use them, i.e. relative to a solver folder
such as `examples/tpb/`, which is how the drivers' own defaults are spelled.

**H2O** (`../../vendor/sbd-upstream/data/h2o/`): `h2o-1em3` through `h2o-1em8` alpha determinant files.
**N2** (`../../vendor/sbd-upstream/data/n2/`): `1em3` through `1em7` and `3em4` through `3em7` alpha determinant files.

Smaller thresholds = more determinants = higher accuracy.

## Performance Tips

**CPU:** Set `OMP_NUM_THREADS` to cores per MPI rank (e.g., 8 ranks × 4 threads = 32 cores).

**GPU:** One MPI rank per GPU — each rank is auto-assigned `gpu_id = rank % num_gpus`.
Use method 0 (matrix-free Davidson) for best GPU performance.

**Do not set `OMP_NUM_THREADS=1` for GPU runs.** One rank per GPU is about device
ownership, not thread count, and a GPU build still does real work on the host: helper
construction is host-threaded in both solvers (`tpb/helper.h`, `gdb/helper.h`), and for
GDB the heatbath expansion and carryover selection have no device implementation at all
(`gdb/expansion.h`, `gdb/carryover.h` carry no `thrust::` code and are compiled in
regardless of backend). One thread per rank single-threads all of it. Divide the node's
physical cores among the ranks exactly as for a CPU run — e.g. 96 cores with 8 ranks is
`OMP_NUM_THREADS=12`. Expect GPU utilisation below 100% as a result; that is the host
phases, not a fault.

## GDB (general determinant basis)

`gdb_diag` spans the subspace with an explicit determinant list rather than the
Cartesian product TPB uses, and decomposes as `t_comm_size × b_comm_size × helper`
with its own field names. It has no example driver yet, so its decomposition is not
documented further here.

`b_comm_size` must currently be 1: every rank passes the whole determinant list, so
splitting the basis communicator would have each rank diagonalize the full subspace
while believing it held a shard.

## See Also

- [Repository README](../README.md) — Installation, API reference
- [Upstream SBD library](https://github.com/r-ccs-cms/sbd) — C++ library overview
