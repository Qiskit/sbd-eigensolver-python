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
diagonalization methods can also be called directly.

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
        sbd_config={"eps": 1e-5, "max_it": 10, "max_nb": 10},
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

Backends
--------

Every backend your toolchain supports is built into one installation. Choose one per
call, with ``device=`` or a :class:`~sbd.device_config.DeviceConfig`:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Device
     - Runs on
     - Built with
   * - ``cpu``
     - Host, OpenMP
     - Any C++ compiler with OpenMP
   * - ``gpu``
     - NVIDIA GPUs, Thrust
     - NVIDIA HPC SDK (``nvc++``)
   * - ``gpu-omp``
     - NVIDIA or AMD GPUs, OpenMP offload
     - ``nvc++`` on NVIDIA, ``amdclang++`` on AMD

The GPU backends hand device memory to MPI, so they need a GPU-aware MPI.

TPB or GDB
----------

SBD has two ways of spanning the subspace:

- **TPB** (tensor-product basis), :func:`sbd.tpb_diag`: the product of an alpha and a
  beta determinant list. This is what the SQD loop uses, since sampled bitstrings
  arrive as spin-string lists.
- **GDB** (general determinant basis), :func:`sbd.gdb_diag`: an explicit list of
  determinants, which can be sharded across ranks. It suits a basis you select
  yourself, such as one grown by heatbath expansion, where a product space would be
  far larger than needed. GDB support is experimental and is not part of the SQD-loop
  integration.

More
----

- The `README <https://github.com/Qiskit/sbd-eigensolver-python/blob/main/README.md>`__
  and the `examples <https://github.com/Qiskit/sbd-eigensolver-python/tree/main/examples>`__,
  organized by basis type with a README in each folder.
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
