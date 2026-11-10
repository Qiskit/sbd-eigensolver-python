# SBD Python Bindings

Python bindings for [SBD](https://github.com/r-ccs-cms/sbd) (Selected Basis
Diagonalization): MPI-parallel eigensolvers for a Hamiltonian projected onto a selected
subspace, on CPUs and on NVIDIA or AMD GPUs. The main use is as the eigensolver of a
sample-based quantum diagonalization (SQD) loop from
[qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd); the solvers can also be
called directly.

**Documentation:** <https://qiskit.github.io/sbd-eigensolver-python/>, with the API
reference and release notes.

> [!NOTE]
> This package is newly open-sourced. The Python API follows semantic versioning, but
> the build configuration and GPU backends have been exercised on a limited set of
> platforms — please report issues.

## What it does

- **Molecules, from an FCIDUMP**, restricted or unrestricted (UHF):
  - **TPB**, `sbd.tpb_diag`: the subspace is the product of an alpha and a beta string
    list, so its dimension is `|adet| × |bdet|`. The mature path, and the one the SQD
    integration uses.
  - **GDB**, `sbd.gdb_diag`: the subspace is an explicit list of determinants, which can
    be sharded across ranks and grown by heat-bath expansion. Experimental: newer, a
    smaller tested surface, and still evolving as upstream SBD does.

  If your subspace is a product of alpha and beta lists, use TPB; otherwise GDB.
- **Lattice and spin models**, written as creation and annihilation operators (CAOP):
  the ground state, `sbd.caop_diag_from_files`, and its Green's function,
  `sbd.spectrum`.
- **Energy variance and Epstein–Nesbet PT2** of a saved state, `sbd.analysis`: how far
  it is from the exact one. Runs on CPU with MPI + OpenMP across nodes.
- **qiskit-addon-sqd integration**, `sbd.sbd_solver`: SBD as the `sci_solver` of the
  SQD loop.

## Why the bindings

Upstream SBD ships each feature as its own C++ app, configured and compiled by hand,
with the backend, unrestricted integrals and complex amplitudes fixed when it is
compiled. The bindings bring them under one install and one API:

- **One install.** `pip install` builds TPB, GDB, CAOP, the Green's function and the
  variance evaluators together; there is no separate executable to build, configure or
  keep in step for each.
- **Choices per call, not per build.** The device is a `device=` argument, a UHF
  FCIDUMP selects the unrestricted build by itself, and CAOP takes `complex=True`, so
  changing any of them means changing an argument, not recompiling.
- **Python in, Python out.** TPB and GDB take determinants as arrays and return the
  energy, RDMs and next subspace as Python objects, so a loop such as SQD or heat-bath
  rounds runs in one MPI job without files passed between programs.
- **Inputs checked first.** Inputs that upstream would get silently wrong or hang on,
  such as a rank count that does not fit the MPI grid or GDB shards out of order, raise
  a `ValueError` on every rank before any work starts.

The [`examples/`](examples/README.md) drivers keep the command-line workflow, for
running from `mpirun` with options as upstream's apps do.

## Backends

Every install builds the CPU backend, and every GPU backend the toolchain supports;
each is loaded only when first used. Choose one per call with `device=`:

- `cpu`: the host, with OpenMP.
- `gpu`: NVIDIA GPUs, with Thrust.
- `gpu-omp`: NVIDIA or AMD GPUs, with OpenMP offload.

What each runs:

| | `cpu` | `gpu` | `gpu-omp` |
|---|---|---|---|
| TPB | ✅ | ✅ | ✅ |
| GDB | ✅ | ✅ | not supported |
| CAOP ground state | ✅ | ✅ | not supported |
| CAOP Green's function | ✅ | ✅ (1) | not supported |
| Unrestricted (UHF) FCIDUMPs | ✅ | ✅ | not supported |
| Energy variance and PT2 | on the CPU (2) | on the CPU (2) | on the CPU (2) |

1. Only applying the Hamiltonian on the fly (`--hamiltonian-mode on-the-fly`) uses the
   GPU; the default stored Hamiltonian and the rest of the work run on the CPU. See
   [`examples/caop/README.md`](examples/caop/README.md#greens-function).
2. The analysis always runs on the CPU, with MPI and OpenMP, whichever device the
   solve used, so it takes no `device=`.

`gpu-omp` does not support GDB or CAOP because upstream has no offload code for them,
and not UHF files because its offload kernels read only the restricted integral layout;
asking for one raises an error rather than running it on the host. `gpu-omp` is one
device for both GPU vendors: an install compiles for one, and
`sbd.get_backend("gpu-omp").__sbd_offload_target__` says which. Vendor-flavoured names
such as `gpu-amd-omp` or `rocm` resolve to it. The GPU backends expect a GPU-aware MPI;
see [INSTALL.md](INSTALL.md#gpu-aware-mpi) for the fallbacks.

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

The `cpu` and `gpu` backends are also built a second time, with upstream's `-D_UHF`, for
unrestricted (UHF) FCIDUMPs: `sbd._core_cpu_uhf` and `sbd._core_gpu_thrust_uhf`. A solve
uses that build automatically when the FCIDUMP is unrestricted, so there is no device to
choose. `available_backends()` does not list these builds; `sbd.has_uhf("cpu")` and
`sbd.print_info()` report them. `gpu-omp` has no UHF build.

**See [INSTALL.md](INSTALL.md)** for the rest: prerequisites per platform, installing
from a git checkout with the submodule, every environment variable the build reads
(GPU toolchains, architectures, MPI and BLAS selection, narrowing which backends get
built), building against an existing host MPI, GPU-aware MPI, and fuller verification.

## SBD in an SQD loop

```python
from functools import partial
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
from sbd.device_config import DeviceConfig
from sbd.sbd_solver import solve_sci_batch

sbd_solver = partial(
    solve_sci_batch,
    sbd_config={"method": 0, "eps": 1e-5, "max_it": 10, "max_nb": 10},
    device_config=DeviceConfig.gpu(),        # or .cpu(), .gpu_omp()
)

result = diagonalize_fermionic_hamiltonian(
    hcore, eri, bit_array,
    sci_solver=sbd_solver,                   # SBD plugs in here
    norb=norb, nelec=nelec,
    samples_per_batch=3000, num_batches=3, max_iterations=5,
    symmetrize_spin=True,
)
```

Run it under `mpirun`: every rank calls the solver, and SBD spreads each
diagonalization over them. No `sbd.init()` is needed. It requires qiskit-addon-sqd
0.13.1 or later, for this distributed (SPMD) mode; SBD-selected carryover and subspace
policies need 0.14.0 or later.

- **TPB only.** qiskit-addon-sqd's fermion solver assumes the subspace is a product of
  alpha and beta string lists, a restriction inherited from PySCF, so a GDB determinant
  list cannot be passed through it
  ([Qiskit/qiskit-sqd#389](https://github.com/Qiskit/qiskit-sqd/issues/389)). For
  GDB, call `gdb_diag` directly or use the drivers in
  [`examples/gdb/`](examples/gdb/README.md).
- **Who chooses the next subspace.** By default qiskit-addon-sqd chooses the carryover
  from the amplitudes SBD returns. With `"carryover_type"` in `sbd_config`, SBD chooses
  it instead; see [SBD-selected carryover](examples/tpb/README.md#sbd-selected-carryover)
  and [Trim SQD](examples/tpb/README.md#using-a-different-subspace-policy-trim-sqd).
- **Growing the subspace.**
  [`run_sqd_enlarge_subspace_sbd.py`](examples/tpb/run_sqd_enlarge_subspace_sbd.py)
  adds single excitations of the dominant determinants between rounds, and so keeps
  improving past a fixed sample pool where the plain loop stops.

[`examples/tpb/README.md`](examples/tpb/README.md#sqd-parameters) has every SQD and
SBD parameter, and [`run_sqd_sbd.py`](examples/tpb/run_sqd_sbd.py) is a complete
driver.

## Examples

Located in `examples/`, organized by solver. [`examples/README.md`](examples/README.md)
shows how they fit together, from a solve through its variance check to, for lattice
models, the Green's function. Each folder's README covers its drivers, options, and the
backend and threading settings that matter for it:

- [`examples/tpb/`](examples/tpb/README.md) — TPB diagonalization, the SQD loops, the
  subspace-enlargement driver, and unrestricted (UHF) FCIDUMPs.
- [`examples/gdb/`](examples/gdb/README.md) — GDB over an explicit determinant list,
  and iterative heat-bath expansion.
- [`examples/caop/`](examples/caop/README.md) — lattice and spin models: the ground
  state, its variance, and its Green's function.
- [`examples/analysis/`](examples/analysis/README.md) — energy variance and PT2 of a
  state saved by any solver, as a separate MPI job or at the end of the same run.

## Troubleshooting

Build and runtime problems — GPU backends building as host code, MPI mismatches, a
non-GPU-aware MPI, macOS OpenMP, GPU assignment across ranks — are in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Please report anything else as a
[GitHub issue](https://github.com/Qiskit/sbd-eigensolver-python/issues).

## License

[Apache License 2.0](LICENSE.txt).
