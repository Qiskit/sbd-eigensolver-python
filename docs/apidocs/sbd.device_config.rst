================================================
Device configuration (:mod:`sbd.device_config`)
================================================

.. module:: sbd.device_config

Selects the backend a solver call runs on. Pass a :class:`DeviceConfig` as the
``device_config`` argument of :func:`sbd.sbd_solver.solve_sci_batch` or
:func:`~sbd.sbd_solver.solve_sci`.

.. autoclass:: DeviceConfig
   :members: cpu, gpu, gpu_omp, auto
   :member-order: bysource
   :no-inherited-members:
