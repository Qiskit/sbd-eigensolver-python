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

"""How the example drivers fail: loudly and promptly, not silently or by hanging.

- A save into a directory that does not exist must be refused before solving: SBD
  does not check that its file opened, so the state would be lost silently.
- A saved state must be refused on a layout SBD cannot load it onto.

The drivers run in a **subprocess**, as a user runs them, so their ``__main__`` and
argument handling are exercised; they test the backend ``SBD_TEST_DEVICE`` names
(cpu when unset). The multi-rank cases are ``mpi``-marked: under ``tox -e mpi``, rank 0
starts the driver under its own ``mpirun -n 2``, with the launching job's
process-manager variables removed, and every rank checks the outcome. That a failure
on one rank aborts the job is ``test_mpi_util.py``'s, since every driver starts
through ``sbd.mpi_util.run``.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

import sbd

REPO = pathlib.Path(__file__).resolve().parents[1]
H2O = REPO / "vendor" / "sbd-upstream" / "data" / "h2o"

# Small enough to be quick: 12 alpha strings is a 144-determinant subspace.
ALPHA_LIMIT = 12


def _device(device_config):
    """``SBD_TEST_DEVICE`` through conftest's fixture, which skips an unbuilt backend."""
    return device_config.device if device_config is not None else "cpu"


def _driver_args(script, device, tmp_path, ranks=2):
    """Arguments for a short h2o run of ``script``, its grid spread over ``ranks``."""
    fcidump = str(H2O / "fcidump.txt")
    alpha = str(H2O / "h2o-1em3-alpha.txt")
    if script == "tpb/run_sbd_diag.py":
        lines = (H2O / "h2o-1em3-alpha.txt").read_text().split()[:ALPHA_LIMIT]
        subset = tmp_path / "alpha.txt"
        subset.write_text("\n".join(lines) + "\n")
        return ["--device", device, "--fcidump", fcidump, "--adetfile", str(subset),
                "--adet_comm_size", str(ranks)]
    if script in ("tpb/run_sqd_sbd.py", "tpb/run_sqd_enlarge_subspace_sbd.py"):
        return ["--device", device, "--fcidump", fcidump,
                "--counts", str(REPO / "examples" / "tpb" / "count_dict_h2o.json"),
                "--max_iterations", "1", "--num_batches", "1", "--adet_comm_size", str(ranks)]
    common = ["--device", device, "--fcidump", fcidump, "--from-alpha", alpha,
              "--alpha-limit", str(ALPHA_LIMIT), "--b_comm_size", str(ranks)]
    if script == "gdb/run_gdb_heatbath.py":
        return common + ["--subspace-from", "from-alpha", "--cutoffs", "1e-3",
                         "--max_rounds", "1"]
    return common


def _clean_env(**extra):
    """The environment without the launching job's process-manager variables."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI_", "PMIX_", "OMPI_", "MPI_LOCAL"))}
    env.update(extra)
    return env


def _run(script, args, env=None, launcher=(), timeout=600):
    """Run a driver from its own directory, clear of any MPI job this test is in.

    Under ``tox -e mpi`` the test itself runs inside an MPI job, and a driver started
    without a launcher would still find that job's process-manager variables and try
    to join it, so every run gets a clean environment, not only the launched ones.
    """
    env = _clean_env() if env is None else env
    path = REPO / "examples" / script
    return subprocess.run([*launcher, sys.executable, path.name, *args], cwd=path.parent,
                          capture_output=True, text=True, timeout=timeout, env=env)


@pytest.fixture
def shared_dir():
    """A directory rank 0 makes and every rank sees, since each rank's tmp_path is its own."""
    import tempfile
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    path = comm.bcast(tempfile.mkdtemp(prefix="sbd-drivers-") if comm.Get_rank() == 0
                      else None)
    yield pathlib.Path(path)
    comm.Barrier()
    if comm.Get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)


def _on_rank_zero(check):
    """Run ``check(mpirun)`` on rank 0 alone; every rank skips or fails with it."""
    from mpi4py import MPI

    error = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpirun is None:
            error = "skip: no mpirun or mpiexec on PATH"
        else:
            try:
                check(mpirun)
            except AssertionError as caught:
                error = f"fail: {caught}"
    error = MPI.COMM_WORLD.bcast(error, root=0)
    if error:
        kind, message = error.split(": ", 1)
        (pytest.skip if kind == "skip" else pytest.fail)(message)


@pytest.mark.parametrize("script", ["tpb/run_sbd_diag.py", "gdb/run_gdb_diag.py"])
def test_a_missing_save_directory_is_refused_before_solving(script, device_config, tmp_path):
    """SBD would drop the save silently; the driver must stop first."""
    args = _driver_args(script, _device(device_config), tmp_path, ranks=1)
    completed = _run(script, args + ["--savename", str(tmp_path / "missing" / "x-")])
    assert completed.returncode != 0
    assert "does not exist" in completed.stderr
    assert "Ground state energy" not in completed.stdout


@pytest.mark.mpi
def test_tpb_refuses_a_state_on_a_grid_with_more_blocks_than_files(device_config, shared_dir):
    """A TPB state loads onto any grid of at most as many blocks as it has files.

    One more block than that and SBD calls MPI_Abort with status 0, which looks like
    success, so the driver refuses it. A smaller grid is allowed, and gives the same
    energy as the run that saved the state.
    """
    _on_rank_zero(lambda mpirun: _check_tpb_layouts(device_config, shared_dir, mpirun))


def _check_tpb_layouts(device_config, tmp_path, mpirun):
    script = "tpb/run_sbd_diag.py"
    device = _device(device_config)
    tight = ["--tolerance", "1e-8"]
    saved = _run(script, _driver_args(script, device, tmp_path) + tight
                 + ["--savename", str(tmp_path / "s-")],
                 env=_clean_env(), launcher=(mpirun, "-n", "2"))
    assert saved.returncode == 0, saved.stderr[-2000:]
    energy = [l for l in saved.stdout.splitlines() if "Ground state energy" in l]

    # Two files, one block: allowed.
    smaller = _run(script, _driver_args(script, device, tmp_path, ranks=1) + tight
                   + ["--loadname", str(tmp_path / "s-")])
    assert smaller.returncode == 0, smaller.stderr[-2000:]
    assert [l for l in smaller.stdout.splitlines() if "Ground state energy" in l] == energy

    # One file, two blocks: refused. Two ranks, not more, so that an MPI that will
    # not oversubscribe a small machine (Open MPI on a CI runner) can start it.
    one = _run(script, _driver_args(script, device, tmp_path, ranks=1) + tight
               + ["--savename", str(tmp_path / "one-")])
    assert one.returncode == 0, one.stderr[-2000:]
    refused = _run(script, _driver_args(script, device, tmp_path)
                   + ["--loadname", str(tmp_path / "one-")],
                   env=_clean_env(), launcher=(mpirun, "-n", "2"))
    assert refused.returncode != 0
    assert "at most 1" in refused.stderr


@pytest.mark.mpi
def test_gdb_refuses_a_state_on_fewer_shards_than_files(device_config, shared_dir):
    """A GDB state loaded onto fewer shards than files starts Davidson from part of it."""
    _on_rank_zero(lambda mpirun: _check_gdb_layouts(device_config, shared_dir, mpirun))


def _check_gdb_layouts(device_config, tmp_path, mpirun):
    script = "gdb/run_gdb_diag.py"
    device = _device(device_config)
    saved = _run(script, _driver_args(script, device, tmp_path)
                 + ["--savename", str(tmp_path / "g-")],
                 env=_clean_env(), launcher=(mpirun, "-n", "2"))
    assert saved.returncode == 0, saved.stderr[-2000:]
    refused = _run(script, _driver_args(script, device, tmp_path, ranks=1)
                   + ["--loadname", str(tmp_path / "g-")])
    assert refused.returncode != 0
    assert "has 2 files but --b_comm_size is 1" in refused.stderr


@pytest.mark.mpi
def test_thrust_gdb_refuses_an_empty_shard(backend):
    """Upstream's Thrust GDB solve never returns when a shard is empty; refuse instead.

    Runs in the MPI job (``tox -e mpi``), on two ranks, one of which holds nothing.
    Through the ``backend`` fixture, like the rest of the suite, so this test does
    not initialize the package itself and leave a different default device for the
    tests after it.
    """
    if "thrust" not in getattr(backend, "__name__", ""):
        pytest.skip("the Thrust backend only (SBD_TEST_DEVICE=gpu)")
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() != 2:
        pytest.skip("needs exactly two ranks")
    fcidump = backend.LoadFCIDump(str(H2O / "fcidump.txt"))
    config = backend.GDB_SBD()
    config.b_comm_size, config.bit_length = 2, 62
    strings = ["0" * 38 + "1" * 10] if MPI.COMM_WORLD.Get_rank() == 0 else []
    det = sbd.from_strings(strings, 62, 48) if strings else np.zeros((0, 1), np.uint64)
    with pytest.raises(ValueError, match="needs determinants on every"):
        sbd.gdb_diag(fcidump, det, config)
