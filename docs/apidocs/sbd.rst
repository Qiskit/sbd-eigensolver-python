=============================
SBD bindings (:mod:`sbd`)
=============================

.. module:: sbd

Direct access to SBD's two diagonalization methods. **TPB** (tensor-product basis)
diagonalizes in the product of an alpha and a beta determinant list. **GDB**
(general determinant basis) diagonalizes in an explicit list of determinants, and is
experimental. For use inside a ``qiskit-addon-sqd`` loop, see :mod:`sbd.sbd_solver`
instead.

The backend initializes itself on first use, so :func:`init` is needed only to choose
a device or communicator explicitly.

Diagonalization
===============

.. autofunction:: tpb_diag
.. autofunction:: tpb_diag_from_files
.. autofunction:: gdb_diag

Configuration
=============

.. autofunction:: TPB_SBD
.. autofunction:: GDB_SBD

Both return a configuration object whose fields are set as attributes before it is
passed to a diagonalization, for example ``config.eps = 1e-8``. The defaults are
upstream SBD's; :func:`sbd.sbd_solver.solve_sci` and
:func:`~sbd.sbd_solver.solve_sci_batch` set their own unless overridden through
``sbd_config``.

Fields shared by ``TPB_SBD`` and ``GDB_SBD``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``method``
     - ``0`` Davidson, ``1`` Davidson with the Hamiltonian stored. TPB also has
       ``2`` and ``3``, the Lanczos variants; GDB rejects them.
   * - ``eps``
     - Davidson stopping tolerance on the residual norm, not on the energy.
   * - ``max_it``
     - Maximum Davidson iterations. Reaching it before ``eps`` returns a partly
       converged vector without warning.
   * - ``max_nb``
     - Maximum number of Davidson basis vectors.
   * - ``max_time``
     - Wall-clock limit, in seconds.
   * - ``init``
     - Starting vector: ``0`` puts unit weight on the first determinant of the basis,
       ``1`` is random.
   * - ``seed``
     - Seed for the random starting vector; used only with ``init = 1``.
   * - ``do_rdm``
     - ``0`` returns the orbital density only, ``1`` also the 1- and 2-particle RDMs.
   * - ``carryover_type``
     - How SBD selects determinants for a next iteration. ``0`` is off. See the
       per-method tables below for what the other values do.
   * - ``ratio``, ``threshold``
     - Carryover selection parameters. Which of the two a given ``carryover_type``
       reads differs, and SBD silently ignores the other.
   * - ``bit_length``
     - Bits per word when packing determinants. At most 63; 64 is undefined
       behavior in SBD's multi-rank redistribution. For GDB's heatbath expansion
       (``carryover_type`` 2 or 3) it must also be even whenever a determinant spans
       more than one word, or the expansion crashes or returns a wrong energy. The
       ``examples/gdb`` drivers require an even value of at most 62.

Fields of ``TPB_SBD`` only:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``adet_comm_size``, ``bdet_comm_size``
     - MPI ranks spanning the alpha and beta determinant dimensions.
   * - ``task_comm_size``
     - MPI ranks spanning task-level parallelism. The rank count must be a multiple of
       ``task_comm_size * adet_comm_size * bdet_comm_size``; the remainder becomes the
       derived helper dimension.
   * - ``carryover_type``
     - ``1`` ranks half-determinants by marginal weight; ``2`` also adds their single
       excitations; ``3`` ranks whole determinants by amplitude, then adds single
       excitations.
   * - ``do_shuffle``
     - Shuffle the determinants loaded from file before use.
   * - ``dump_matrix_form_wf``
     - Path to which SBD writes the wavefunction as an ``|adet| x |bdet|`` matrix.
   * - ``use_precalculated_dets``
     - Thrust backend only. Precompute a determinant index for the whole subspace on
       the GPU: faster, but far more memory.
   * - ``max_memory_gb_for_determinants``
     - Thrust backend only, and used only when ``use_precalculated_dets`` is off: a
       per-thread buffer cap in GB, or ``-1`` for none.

Fields of ``GDB_SBD`` only:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``b_comm_size``
     - MPI ranks the determinant basis is sharded across. It is the only dimension
       that divides memory. With more than 1, each rank passes only its own shard to
       :func:`gdb_diag`.
   * - ``t_comm_size``
     - MPI ranks spanning tasks. Must not exceed ``b_comm_size``. The rank count must
       be a multiple of ``t_comm_size * b_comm_size``; on the Thrust backend it must
       equal it, since GDB on Thrust has no helper dimension.
   * - ``carryover_type``
     - ``1`` truncates by weight; ``2`` and ``3`` expand by heatbath selection.
   * - ``heatbath_cutoff``
     - Heatbath selection cutoff: a candidate is added when ``|c_i H_ij|`` exceeds it.
   * - ``heatbath_truncation``
     - Weight below which parents are dropped before expansion. Large values remove
       the parents the expansion needs.
   * - ``heatbath_batch_size``
     - Number of candidates processed per batch during expansion.

Inputs
======

.. autofunction:: LoadFCIDump
.. autofunction:: LoadAlphaDets
.. autofunction:: sort_bitarray
.. autofunction:: from_strings
.. autofunction:: sort_bitarray_array

Backends and devices
====================

.. autofunction:: init
.. autofunction:: finalize
.. autofunction:: available_backends
.. autofunction:: backend_load_errors
.. autofunction:: get_backend
.. autofunction:: get_device_id

MPI
===

.. autofunction:: get_rank
.. autofunction:: get_world_size
