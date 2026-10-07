#####################################
Selected Basis Diagonalization (SBD)
#####################################

``sbd-eigensolver`` provides Python bindings for the SBD (Selected Basis
Diagonalization) library, which finds eigenvalues and eigenvectors of a
second-quantized Hamiltonian projected onto a subspace spanned by a selected set of
determinants. The bindings are MPI-parallel and can run on CPUs or, where a suitable
toolchain is available, on NVIDIA or AMD GPUs.

The package also exposes a solver compatible with the ``qiskit-addon-sqd`` interface,
so SBD can be used as the diagonalization step of a sample-based quantum
diagonalization (SQD) workflow. See :mod:`sbd.sbd_solver`.

Getting started
---------------

The `README <https://github.com/Qiskit/sbd-eigensolver-python/blob/main/README.md>`__
in the root of this project's repository introduces the package and its
``qiskit-addon-sqd`` integration. `INSTALL.md
<https://github.com/Qiskit/sbd-eigensolver-python/blob/main/INSTALL.md>`__ covers
installation in full, including the environment variables that control which backends
are compiled. Example scripts and a notebook live in `examples
<https://github.com/Qiskit/sbd-eigensolver-python/tree/main/examples>`__, organized by
basis type, with a README in each folder.

A minimal diagonalization looks like this::

    import sbd

    config = sbd.TPB_SBD()
    results = sbd.tpb_diag_from_files("FCIDUMP", "adets.dat", config)

The backend is initialized automatically on first use; :func:`sbd.init` only needs to
be called to select a device explicitly.

Contributing
------------

The source code is available `on GitHub
<https://github.com/Qiskit/sbd-eigensolver-python>`__.

We use `GitHub issues
<https://github.com/Qiskit/sbd-eigensolver-python/issues>`__ for tracking requests and
bugs.

License
-------

`Apache License 2.0
<https://github.com/Qiskit/sbd-eigensolver-python/blob/main/LICENSE.txt>`__

.. toctree::
   :hidden:

   Documentation home <self>
   API reference <apidocs/index>
   Release notes <release-notes>
   GitHub <https://github.com/Qiskit/sbd-eigensolver-python>
