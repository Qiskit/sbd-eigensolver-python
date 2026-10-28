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

- A rank that fails while the others wait in a collective must abort the whole MPI
  job, not leave it hanging. A ``sitecustomize`` hook in the test's own directory
  makes rank 1's first SBD solve raise, with the drivers unmodified.
- A save into a directory that does not exist must be refused before solving: SBD
  does not check that its file opened, so the state would be lost silently.
- A saved state must be refused on a layout SBD cannot load it onto.

The drivers run in a **subprocess**, as a user runs them, so their ``__main__`` and
argument handling are exercised; they test the backend ``SBD_TEST_DEVICE`` names
(cpu when unset). The multi-rank cases start their own ``mpirun -n 2``, with the
launching job's process-manager variables removed, and skip without one.
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

# Small enough to be quick: 12 alpha strings is a 144-determinant subspace. It also
# bounds SBD_TEST_NPROCS for the rank-failure test: run_sbd_diag.py spreads the 12
# strings over that many ranks.
ALPHA_LIMIT = 12

# Ranks for the rank-failure test: SBD_TEST_NPROCS, as ``tox -e mpi`` uses, default 2.
NPROCS = int(os.environ.get("SBD_TEST_NPROCS", "2"))

HOOK = '''
import os

if os.environ.get("SBD_TEST_FAIL_RANK") is not None:
    import builtins

    _real_import = builtins.__import__
    _patched = False

    def _import(name, *args, **kwargs):
        global _patched
        module = _real_import(name, *args, **kwargs)
        if not _patched and (name == "sbd" or name.startswith("sbd.")):
            _patched = True           # before the imports below, which re-enter here
            import sbd as _sbd
            import sbd.sbd_solver as _solver
            fail = int(os.environ["SBD_TEST_FAIL_RANK"])
            # The drivers' solve entry points: the package functions, and the core
            # the SQD solver wrappers call.
            for owner, entry in ((_sbd, "tpb_diag"), (_sbd, "tpb_diag_from_files"),
                                 (_sbd, "gdb_diag"), (_solver, "_solve_sci_core")):
                real = getattr(owner, entry)

                def wrapper(*a, _real=real, **k):
                    if _sbd.get_rank() == fail:
                        raise RuntimeError(f"injected failure on rank {fail}")
                    return _real(*a, **k)

                setattr(owner, entry, wrapper)
        return module

    builtins.__import__ = _import
'''


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
    if script == "tpb/run_sqd_sbd.py":
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
    path = REPO / "examples" / script
    return subprocess.run([*launcher, sys.executable, path.name, *args], cwd=path.parent,
                          capture_output=True, text=True, timeout=timeout, env=env)


DRIVERS = ["tpb/run_sbd_diag.py", "tpb/run_sqd_sbd.py",
           "gdb/run_gdb_diag.py", "gdb/run_gdb_heatbath.py"]


@pytest.mark.parametrize("script", DRIVERS)
def test_a_failure_on_one_rank_aborts_the_job(script, device_config, tmp_path):
    """Rank 1 fails in its solve while the others wait there; the job must still end.

    Before, the failing rank exited while the others waited in the solve's collectives
    until the job's time limit. The timeout here is the hang detector. Runs on
    SBD_TEST_NPROCS ranks (default 2); the layout tests below need exactly 2.
    """
    if script == "tpb/run_sqd_sbd.py":
        pytest.importorskip("qiskit_addon_sqd.fermion", reason="the SQD driver needs it")
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("no mpirun or mpiexec on PATH")
    (tmp_path / "sitecustomize.py").write_text(HOOK)
    env = _clean_env(SBD_TEST_FAIL_RANK="1",
                     PYTHONPATH=os.pathsep.join(
                         [str(tmp_path), *filter(None, [os.environ.get("PYTHONPATH")])]))
    args = _driver_args(script, _device(device_config), tmp_path, ranks=NPROCS)
    try:
        completed = _run(script, args, env=env, launcher=(mpirun, "-n", str(NPROCS)),
                         timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{script} did not finish after rank 1 failed: the other rank is "
                    "still waiting for it")
    assert completed.returncode != 0
    # The failing rank reports itself, whoever prints the traceback.
    assert "injected failure on rank 1" in completed.stderr
    assert "[rank 1]" in completed.stderr


@pytest.mark.parametrize("script", ["tpb/run_sbd_diag.py", "gdb/run_gdb_diag.py"])
def test_a_missing_save_directory_is_refused_before_solving(script, device_config, tmp_path):
    """SBD would drop the save silently; the driver must stop first."""
    args = _driver_args(script, _device(device_config), tmp_path, ranks=1)
    completed = _run(script, args + ["--savename", str(tmp_path / "missing" / "x-")])
    assert completed.returncode != 0
    assert "does not exist" in completed.stderr
    assert "Ground state energy" not in completed.stdout


def test_tpb_refuses_a_state_on_a_grid_with_more_blocks_than_files(device_config, tmp_path):
    """A TPB state loads onto any grid of at most as many blocks as it has files.

    One more block than that and SBD calls MPI_Abort with status 0, which looks like
    success, so the driver refuses it. A smaller grid is allowed, and gives the same
    energy as the run that saved the state.
    """
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("no mpirun or mpiexec on PATH")
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


def test_gdb_refuses_a_state_on_fewer_shards_than_files(device_config, tmp_path):
    """A GDB state loaded onto fewer shards than files starts Davidson from part of it."""
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("no mpirun or mpiexec on PATH")
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
