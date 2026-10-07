=============================================
SQD-compatible solver (:mod:`sbd.sbd_solver`)
=============================================

.. module:: sbd.sbd_solver

SBD as the eigensolver of a ``qiskit-addon-sqd`` loop. Pass
:func:`solve_sci_batch`, with its options bound by :func:`functools.partial`, as the
``sci_solver`` argument of ``diagonalize_fermionic_hamiltonian``; it runs on every MPI
rank. These solvers use SBD's TPB method. GDB has no SQD-loop integration; call
:func:`sbd.gdb_diag` directly.

.. autofunction:: solve_sci_batch
.. autofunction:: solve_sci
.. autofunction:: assemble_rdms
