# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""On the Thrust backend, a solve with one OpenMP thread is refused, not aborted.

Upstream SBD's Thrust builds do part of the work on the host and call MPI_Abort when
``omp_get_max_threads() == 1``, which ends the Python process with nothing to catch.
The bindings raise ``ValueError`` first. The thread count is fixed when the process
starts, so each case runs in a **subprocess** with ``OMP_NUM_THREADS=1``; only the
Thrust backend is checked, since the CPU and OpenMP-offload builds are not affected.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import sbd

SCRIPT = r"""
import sys
import sbd

sbd.init(device="gpu")
data = sys.argv[1]
fcidump = sbd.LoadFCIDump(f"{data}/fcidump.txt")
alpha = [l.strip() for l in open(f"{data}/h2o-1em3-alpha.txt") if l.strip()][:8]
config = sbd.TPB_SBD()
config.bit_length = 20
half = sbd.sort_bitarray([sbd.from_string(s, 20, 24) for s in alpha])
try:
    sbd.tpb_diag(fcidump, half, half, config)
except ValueError as e:
    print("REFUSED", e)
else:
    print("SOLVED")
"""


@pytest.mark.parametrize("threads, expected", [("1", "REFUSED"), ("2", "SOLVED")])
def test_thrust_needs_more_than_one_openmp_thread(data_dir, threads, expected):
    if "gpu" not in sbd.available_backends():
        pytest.skip("the Thrust backend was not built")
    if os.environ.get("SBD_TEST_DEVICE") not in (None, "gpu"):
        pytest.skip("the Thrust backend only (SBD_TEST_DEVICE=gpu or unset)")
    env = {**os.environ, "OMP_NUM_THREADS": threads}
    env = {k: v for k, v in env.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    done = subprocess.run([sys.executable, "-c", SCRIPT, str(data_dir / "h2o")],
                          capture_output=True, text=True, timeout=300, env=env)
    assert done.returncode == 0, f"{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
    assert expected in done.stdout, done.stdout[-2000:]
    if expected == "REFUSED":
        assert "OMP_NUM_THREADS" in done.stdout
