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
    config = sbd.GDB_SBD()
    config.bit_length = 62
    results = sbd.gdb_diag(fcidump, det, config, savename="state-")   # under mpirun
    # The energy the solve returned is the PT2 reference; variance() needs it.
    result = analysis.variance("state-", fcidump, results["energy"], bit_length=62)
    print(result.variance, result.variance_error, result.energy_pt2)

The stochastic method runs on CPU, over all ranks of the communicator, whichever
device diagonalized the state; the number of ranks need not match the run that saved
it. The exact method needs PySCF and runs in one process.

.. autofunction:: variance
.. autofunction:: load_state
.. autofunction:: state_energy
.. autofunction:: load_gdb_shards
.. autoclass:: VarianceResult
   :members: energy_pt2
.. autoclass:: SavedState
