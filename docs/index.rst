#####################################
Selected Basis Diagonalization (SBD)
#####################################

``sbd-eigensolver`` provides Python bindings for the SBD (Selected Basis
Diagonalization) library, which finds eigenvalues and eigenvectors of a
second-quantized Hamiltonian projected onto a subspace spanned by a selected set of
determinants. The bindings are MPI-parallel and run on CPUs or, where a suitable
toolchain is available, on NVIDIA or AMD GPUs.

The main use is as the eigensolver of a sample-based quantum diagonalization (SQD)
loop from `qiskit-addon-sqd <https://github.com/Qiskit/qiskit-addon-sqd>`__. SBD's
solvers can also be called directly.

What it does
------------

- **Molecules, from an FCIDUMP**, restricted or unrestricted (UHF):

  - **TPB** (tensor-product basis), :func:`sbd.tpb_diag`: the subspace is the product
    of an alpha and a beta string list. The mature path, and the one the SQD loop uses.
  - **GDB** (general determinant basis), :func:`sbd.gdb_diag`: the subspace is an
    explicit list of determinants, which can be sharded across ranks and grown by
    heat-bath expansion. Experimental.

- **Lattice and spin models**, written as creation and annihilation operators (CAOP):
  the ground state, :func:`sbd.caop_diag_from_files`, and its Green's function,
  :mod:`sbd.spectrum`.
- **Energy variance and Epstein-Nesbet PT2** of a saved state, :mod:`sbd.analysis`:
  how far it is from the exact one.
- **The SQD loop's solver**, :mod:`sbd.sbd_solver`.

Upstream SBD ships each of these as its own C++ app, with the backend, unrestricted
integrals and complex amplitudes fixed when it is compiled. Here one install builds
them all, and each is a choice per call: ``device=`` picks the backend, a UHF FCIDUMP
picks the unrestricted build by itself, and CAOP takes ``complex=True``. A feature a
device does not support raises an error naming the devices that do, and inputs
upstream would get silently wrong or hang on raise a ``ValueError`` on every rank
before any work starts.

Installation
------------

The extension modules compile on your machine; no wheels are published. For the
common CPU case:

.. code-block:: bash

    conda create -y -n sbd -c conda-forge \
        python=3.13.12 pybind11 numpy setuptools wheel openblas pyscf pip mpi4py
    conda activate sbd
    pip install sbd-eigensolver

    python -c "import sbd; print(sbd.available_backends())"   # e.g. ['cpu']

GPU backends, building against an existing MPI, and every build option are covered in
`INSTALL.md <https://github.com/Qiskit/sbd-eigensolver-python/blob/main/INSTALL.md>`__.

SBD as the solver of an SQD loop
--------------------------------

Pass :func:`sbd.sbd_solver.solve_sci_batch` as the ``sci_solver`` of
``qiskit-addon-sqd``'s ``diagonalize_fermionic_hamiltonian``, and launch under MPI as
usual. Every rank calls the solver, and SBD decomposes each diagonalization across
them:

.. code-block:: python

    from functools import partial

    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
    from sbd.device_config import DeviceConfig
    from sbd.sbd_solver import solve_sci_batch

    sbd_solver = partial(
        solve_sci_batch,
        sbd_config={"method": 0, "eps": 1e-5, "max_it": 10, "max_nb": 10},
        device_config=DeviceConfig.gpu(),     # or .cpu(), .gpu_omp()
    )

    result = diagonalize_fermionic_hamiltonian(
        hcore, eri, bit_array,
        sci_solver=sbd_solver,
        norb=norb, nelec=nelec,
        samples_per_batch=3000, num_batches=3, max_iterations=5,
        symmetrize_spin=True,
    )

With ``qiskit-addon-sqd`` 0.14.0 or later, SBD can also choose which determinants
carry into the next iteration, by setting ``"carryover_type"`` in ``sbd_config``. With
``carryover_type = 1`` the wavefunction never leaves SBD.

The loop uses TPB only: qiskit-addon-sqd's fermion solver assumes the subspace is a
product of alpha and beta string lists, a restriction inherited from PySCF
(`Qiskit/qiskit-sqd#389 <https://github.com/Qiskit/qiskit-sqd/issues/389>`__). For
GDB, call :func:`sbd.gdb_diag` directly.

Backends
--------

Every installation builds the CPU backend, and every GPU backend the toolchain
supports; each is loaded only when first used. Choose one per call, with ``device=`` or
a :class:`~sbd.device_config.DeviceConfig`: ``cpu`` (the host, OpenMP), ``gpu`` (NVIDIA
GPUs, Thrust) or ``gpu-omp`` (NVIDIA or AMD GPUs, OpenMP offload).

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * -
     - ``cpu``
     - ``gpu``
     - ``gpu-omp``
   * - TPB
     - yes
     - yes
     - yes
   * - GDB
     - yes
     - yes
     - not supported
   * - CAOP ground state
     - yes
     - yes
     - not supported
   * - CAOP Green's function
     - yes
     - yes (1)
     - not supported
   * - Unrestricted (UHF) FCIDUMPs
     - yes
     - yes
     - not supported
   * - Energy variance and PT2
     - on the CPU (2)
     - on the CPU (2)
     - on the CPU (2)

1. Only applying the Hamiltonian on the fly uses the GPU; the default stored
   Hamiltonian and the rest of the work run on the CPU.
2. The analysis always runs on the CPU, with MPI and OpenMP, whichever device the solve
   used, so it takes no ``device=``.

The GPU backends pass device memory to MPI, so they expect a GPU-aware MPI. The
Thrust backend can instead be built to stage transfers through host memory; see
`GPU-aware MPI
<https://github.com/Qiskit/sbd-eigensolver-python/blob/main/INSTALL.md#gpu-aware-mpi>`__
in INSTALL.md.

More
----

- The `README <https://github.com/Qiskit/sbd-eigensolver-python/blob/main/README.md>`__
  and the `examples <https://github.com/Qiskit/sbd-eigensolver-python/tree/main/examples>`__:
  command-line drivers for each solver, the analysis and the Green's function, with a
  README in each folder and a
  `workflows <https://github.com/Qiskit/sbd-eigensolver-python/blob/main/examples/README.md>`__
  page showing how they fit together. They are tested against the upstream SBD commit
  the package pins; with any other, the functionality may be broken.
- The :doc:`API reference <apidocs/index>`.
- `GitHub issues <https://github.com/Qiskit/sbd-eigensolver-python/issues>`__ for
  requests and bugs. The source code is on
  `GitHub <https://github.com/Qiskit/sbd-eigensolver-python>`__.

Licensed under the `Apache License 2.0
<https://github.com/Qiskit/sbd-eigensolver-python/blob/main/LICENSE.txt>`__.

.. toctree::
   :hidden:

   Documentation home <self>
   API reference <apidocs/index>
   Release notes <release-notes>
   GitHub <https://github.com/Qiskit/sbd-eigensolver-python>
