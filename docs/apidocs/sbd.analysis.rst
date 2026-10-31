==========================================
Saved-state analysis (:mod:`sbd.analysis`)
==========================================

.. module:: sbd.analysis

Energy variance and Epstein-Nesbet PT2 of a wavefunction saved by a diagonalization
run with ``savename``, from either :func:`sbd.tpb_diag` or :func:`sbd.gdb_diag`. The
command-line front end, with a walkthrough and guidance on settings, is
``examples/analysis/variance.py``.

.. code-block:: python

    import sbd
    from sbd import analysis

    fcidump = sbd.LoadFCIDump("fcidump.txt")
    result = analysis.variance("state-", fcidump, bit_length=62)   # under mpirun
    print(result.variance, result.variance_error, result.energy_pt2)

The stochastic method runs on CPU, over all ranks of the communicator, whichever
device diagonalized the state; the number of ranks need not match the run that saved
it. The exact method needs PySCF and runs in one process.

.. autofunction:: variance
.. autofunction:: load_state
.. autofunction:: state_energy
.. autofunction:: load_gdb_shards
.. autofunction:: write_tpb_state
.. autoclass:: VarianceResult
   :members: energy_pt2
.. autoclass:: SavedState
