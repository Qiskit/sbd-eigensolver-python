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


"""The drivers' carryover output files: ``--carryover_adetfile``/``--carryover_bdetfile``
(TPB) and ``--carryovername`` (GDB), named and laid out as upstream's apps write them.

The drivers run in a subprocess, as a user runs them, on the backend ``SBD_TEST_DEVICE``
names (cpu when unset). The ``_mpi`` variant has rank 0 start the driver on
``SBD_TEST_NPROCS`` ranks, as ``test_wide_orbitals.py`` does, in a directory rank 0 makes
and shares, since every rank's ``tmp_path`` is its own.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
H2O_DIR = REPO_ROOT / "vendor" / "sbd-upstream" / "data" / "h2o"


@pytest.fixture
def shared_dir():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    path = comm.bcast(tempfile.mkdtemp(prefix="sbd-carryover-") if comm.Get_rank() == 0
                      else None)
    yield pathlib.Path(path)
    comm.Barrier()
    if comm.Get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)


def _device(device_config):
    """``SBD_TEST_DEVICE``, else cpu, as in ``test_wide_orbitals.py``."""
    return device_config.device if device_config is not None else "cpu"


def _run(script, args, device, launcher=()):
    """Run a driver from its own directory, clear of any MPI job this test is in."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI_", "PMIX_", "OMPI_", "MPI_LOCAL"))}
    path = REPO_ROOT / "examples" / script
    done = subprocess.run([*launcher, sys.executable, path.name, "--device", device, *args],
                          cwd=path.parent, capture_output=True, text=True, timeout=600,
                          env=env)
    assert done.returncode == 0, f"{script} failed:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    return done.stdout


def _on_rank_zero(check, shared_dir, device_config):
    """Run ``check`` on rank 0 alone, launching the driver on SBD_TEST_NPROCS ranks.

    Every rank fails together, not rank 0 alone, which would leave the rest waiting.
    """
    from mpi4py import MPI

    nprocs = int(os.environ.get("SBD_TEST_NPROCS", "2"))
    error = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        try:
            check(shared_dir, _device(device_config), (mpirun, "-n", str(nprocs)), nprocs)
        except (AssertionError, subprocess.TimeoutExpired) as caught:
            error = str(caught) or repr(caught)
    error = MPI.COMM_WORLD.bcast(error, root=0)
    if error:
        pytest.fail(error)


def _energy(stdout):
    return float(re.search(r"Ground state energy:\s*(-?\d+\.\d+)", stdout).group(1))


def _check_tpb_carryover_files_start_the_next_run(tmp_path, device, launcher=(), nprocs=1):
    """Type 1 keeps a subset of the strings, which --adetfile/--bdetfile read back."""
    lines = (H2O_DIR / "h2o-1em3-alpha.txt").read_text().split()[:40]
    (tmp_path / "alpha.txt").write_text("\n".join(lines) + "\n")
    common = ["--fcidump", str(H2O_DIR / "fcidump.txt"), "--bit_length", "20",
              "--tolerance", "1e-10", "--adet_comm_size", str(nprocs)]
    first = _run("tpb/run_sbd_diag.py", common + [
        "--adetfile", str(tmp_path / "alpha.txt"), "--carryover_type", "1",
        "--carryover_ratio", "0.5", "--carryover_adetfile", str(tmp_path / "a.txt"),
        "--carryover_bdetfile", str(tmp_path / "b.txt")], device, launcher)
    kept = (tmp_path / "a.txt").read_text().split()
    assert 0 < len(kept) < len(lines) and set(kept) <= set(lines)

    second = _run("tpb/run_sbd_diag.py", common + [
        "--symmetrize_spin", "0", "--adetfile", str(tmp_path / "a.txt"),
        "--bdetfile", str(tmp_path / "b.txt")], device, launcher)
    assert _energy(second) >= _energy(first) - 1e-8


def test_tpb_carryover_files_start_the_next_run_standalone(tmp_path, device_config):
    _check_tpb_carryover_files_start_the_next_run(tmp_path, _device(device_config))


@pytest.mark.mpi
def test_tpb_carryover_files_start_the_next_run_mpi(shared_dir, device_config):
    """The alpha strings split over SBD_TEST_NPROCS ranks; rank 0 writes the files."""
    _on_rank_zero(_check_tpb_carryover_files_start_the_next_run, shared_dir, device_config)


def _check_gdb_carryover_files_start_the_next_run(tmp_path, device, launcher=(), nprocs=1,
                                                  carryover_type=2):
    """--carryovername files feed --detfiles through the printed list.

    Type 2's files, one per rank, hold the expanded subspace, parents included, so the
    next energy is no higher; type 1's hold a subset of the subspace, so it is no lower.
    """
    selection = (["--heatbath_cutoff", "1e-3"] if carryover_type == 2
                 else ["--carryover_ratio", "0.5"])
    common = ["--fcidump", str(H2O_DIR / "fcidump.txt"), "--bit_length", "62",
              "--tolerance", "1e-10", "--iteration", "300", "--b_comm_size", str(nprocs)]
    first = _run("gdb/run_gdb_diag.py", common + [
        "--from-alpha", str(H2O_DIR / "h2o-1em3-alpha.txt"), "--alpha-limit", "12",
        "--carryover_type", str(carryover_type), *selection,
        "--carryovername", str(tmp_path / "next-")], device, launcher)
    assert len(list(tmp_path.glob("next-*.txt"))) == nprocs

    detfiles = re.search(r"--detfiles (\S+)", first).group(1)
    second = _run("gdb/run_gdb_diag.py", common + ["--detfiles", detfiles], device, launcher)
    if carryover_type == 2:
        assert _energy(second) <= _energy(first) + 1e-8
    else:
        assert _energy(second) >= _energy(first) - 1e-8


@pytest.mark.parametrize("carryover_type", [1, 2])
def test_gdb_carryover_files_start_the_next_run_standalone(tmp_path, device_config,
                                                           carryover_type):
    _check_gdb_carryover_files_start_the_next_run(tmp_path, _device(device_config),
                                                  carryover_type=carryover_type)


@pytest.mark.mpi
def test_gdb_carryover_files_start_the_next_run_mpi(shared_dir, device_config):
    """Type 2 sharded over SBD_TEST_NPROCS ranks, one file per rank."""
    _on_rank_zero(_check_gdb_carryover_files_start_the_next_run, shared_dir, device_config)


def test_gdb_type_1_files_are_refused_over_several_shards(tmp_path, device_config):
    """Each shard keeps its own selection, so the files would overlap and not reload."""
    driver = REPO_ROOT / "examples" / "gdb" / "run_gdb_diag.py"
    done = subprocess.run([sys.executable, driver.name, "--device", _device(device_config),
                           "--fcidump", str(H2O_DIR / "fcidump.txt"), "--b_comm_size", "2",
                           "--carryover_type", "1", "--carryovername", str(tmp_path / "x")],
                          cwd=driver.parent, capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "needs --b_comm_size 1" in done.stdout + done.stderr


@pytest.mark.parametrize("script, flag", [
    ("tpb/run_sbd_diag.py", "--carryover_adetfile"),
    ("gdb/run_gdb_diag.py", "--carryovername"),
], ids=["tpb", "gdb"])
@pytest.mark.parametrize("extra, path, message", [
    ([], "x", r"needs? --carryover_type"),
    (["--carryover_type", "1"], "missing/x", r"does not exist"),
], ids=["no-type", "no-directory"])
def test_carryover_files_are_refused_before_the_solve(tmp_path, device_config, script, flag,
                                                      extra, path, message):
    driver = REPO_ROOT / "examples" / script
    done = subprocess.run([sys.executable, driver.name, "--device", _device(device_config),
                           "--fcidump", str(H2O_DIR / "fcidump.txt"), flag, str(tmp_path / path),
                           *extra], cwd=driver.parent, capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert re.search(message, done.stdout + done.stderr)
