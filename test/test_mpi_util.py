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

"""``sbd.mpi_util.run``: a failure on any one rank ends the whole MPI job.

In one process ``run`` returns or raises as the program does, which the plain tests
check. On several ranks its job is to ``MPI_Abort``, which would end the pytest run
calling it too, so the ``_mpi`` tests have rank 0 start a short program under its own
``mpirun`` on ``SBD_TEST_NPROCS`` ranks (2 when unset), as ``test_wide_orbitals.py``
does, and every rank checks the outcome. The program fails on the last rank while the
others wait in a barrier: the hang ``run`` exists to prevent. The timeout is the hang
detector.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from sbd.mpi_util import run

REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVERS = ["tpb/run_sbd_diag.py", "tpb/run_sqd_sbd.py", "tpb/run_sqd_enlarge_subspace_sbd.py",
           "gdb/run_gdb_diag.py", "gdb/run_gdb_heatbath.py"]

# The last rank fails, as ``sys.argv[1]`` says, while every other rank waits in a barrier.
PROGRAM = r"""
import sys
from mpi4py import MPI
from sbd.mpi_util import run

def main():
    world = MPI.COMM_WORLD
    if world.Get_rank() == world.Get_size() - 1:
        if sys.argv[1] == "raise":
            raise RuntimeError("injected failure")
        return 3
    world.Barrier()
    return 0

sys.exit(run(main))
"""


# --- one process: run returns or raises as main does ---------------------------------

def test_run_returns_the_programs_code():
    assert run(lambda: 0) == 0
    assert run(lambda: None) is None
    assert run(lambda: 3) == 3


def test_run_reraises_the_programs_error(capsys):
    def main():
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        run(main)
    assert "[rank 0] unhandled error" in capsys.readouterr().err


def test_run_lets_system_exit_through():
    def main():
        sys.exit(4)

    with pytest.raises(SystemExit) as caught:
        run(main)
    assert caught.value.code == 4


@pytest.mark.parametrize("script", DRIVERS)
def test_every_driver_starts_through_run(script):
    """The drivers rely on ``run`` to abort; a driver that bypassed it would hang."""
    source = (REPO / "examples" / script).read_text()
    assert "from sbd.mpi_util import run" in source
    assert "sys.exit(run(main))" in source


# --- several ranks: a failure on one rank aborts the job -----------------------------

def _on_rank_zero(mode):
    """Start PROGRAM on SBD_TEST_NPROCS ranks from rank 0; every rank gets the outcome."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nprocs = int(os.environ.get("SBD_TEST_NPROCS", "2"))
    outcome = None
    if comm.Get_rank() == 0:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpirun is None:
            outcome = "skip", "no mpirun or mpiexec on PATH"
        else:
            # Started inside an MPI job, the inner launcher would see the outer job's
            # process-manager variables and try to join it. Give it a clean start.
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("PMI_", "PMIX_", "OMPI_", "MPI_LOCAL"))}
            try:
                done = subprocess.run([mpirun, "-n", str(nprocs), sys.executable, "-c",
                                       PROGRAM, mode], capture_output=True, text=True,
                                      timeout=120, env=env)
                outcome = "done", (done.returncode, done.stdout + done.stderr, nprocs)
            except subprocess.TimeoutExpired:
                outcome = "fail", (f"the {nprocs}-rank job did not end after one rank "
                                   "failed: the others are still waiting for it")
    kind, value = comm.bcast(outcome, root=0)
    if kind == "skip":
        pytest.skip(value)
    if kind == "fail":
        pytest.fail(value)
    return value


@pytest.mark.mpi
def test_an_error_on_one_rank_aborts_the_job():
    code, output, nprocs = _on_rank_zero("raise")
    assert code != 0
    assert f"[rank {nprocs - 1}] unhandled error" in output
    assert "injected failure" in output


@pytest.mark.mpi
def test_a_non_zero_code_on_one_rank_aborts_the_job():
    code, output, nprocs = _on_rank_zero("return")
    assert code != 0
    assert f"[rank {nprocs - 1}] exiting with status 3" in output
