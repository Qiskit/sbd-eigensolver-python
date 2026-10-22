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

"""The example drivers' collective file handling, on more than one rank.

``test_gdb_drivers.py`` runs the drivers whole, in a subprocess, on one rank. The
code that moves data between ranks -- writing a carryover sharded over the job,
re-splitting a saved state for a different layout, promoting a round's files -- only
does anything with more than one, and a bug there hangs rather than fails: a rank
that skips a collective leaves the others waiting in it. So these call the drivers'
functions in-process, under ``mpirun -n 2 pytest --only-mpi`` (``tox -e mpi``).

Each test makes its directory on rank 0 and shares it, rather than using
``tmp_path``: every rank runs its own pytest, and one rank's temp-directory cleanup
can remove another's directory mid-test.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import sys
import tempfile
import types

import numpy as np
import pytest

import sbd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GDB_DIR = REPO_ROOT / "examples" / "gdb"
H2O_DIR = REPO_ROOT / "vendor" / "sbd-upstream" / "data" / "h2o"
BITS = 62


def _driver(name):
    """Import an example driver by path, as a module, without running it."""
    path = GDB_DIR / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"driver not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_driver_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def shared_dir():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    path = comm.bcast(tempfile.mkdtemp(prefix="sbd-driver-mpi-") if comm.Get_rank() == 0
                      else None)
    yield pathlib.Path(path)
    comm.Barrier()
    if comm.Get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)


def _random_dets(n, total_bits, seed):
    """``n`` distinct random determinants, packed, in random order."""
    rng = np.random.default_rng(seed)
    strings = {"".join(rng.choice(["0", "1"], total_bits)) for _ in range(3 * n)}
    strings = sorted(strings)[:n]
    rng.shuffle(strings)
    return sbd.from_strings(strings, BITS, total_bits)


def _read_files(prefix, total_bits):
    files = sorted(prefix.parent.glob(prefix.name + "[0-9]" * 6 + ".txt"))
    rows = [line.strip() for f in files for line in f.read_text().splitlines() if line.strip()]
    return files, rows


@pytest.mark.mpi
@pytest.mark.parametrize("carryover_type", [1, 2])
def test_carryover_files_are_one_globally_sorted_set(shared_dir, carryover_type):
    """Pieces held unsorted on every rank come out as files --detfiles can read in order.

    Type 2's carryover is distinct on every rank; type 1's is distinct only on the first
    ``b_comm_size`` ranks, the rest holding copies -- so the test gives the higher ranks
    copies too, and checks each determinant is written exactly once.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    diag = _driver("run_gdb_diag")
    total_bits = 72
    whole = _random_dets(40 * size, total_bits, seed=7)
    b_comm_size = size if carryover_type == 2 else max(1, size // 2)
    owner = rank % b_comm_size
    mine = whole[owner::b_comm_size]
    prefix = shared_dir / "co-"
    args = types.SimpleNamespace(carryover_detfile=str(prefix), carryover_type=carryover_type,
                                 b_comm_size=b_comm_size, bit_length=BITS)

    report = diag.write_carryover(args, mine, total_bits, rank, size)
    comm.Barrier()

    files, rows = _read_files(prefix, total_bits)
    expected = sorted(sbd.to_strings(whole, BITS, total_bits),
                      key=lambda s: int(s, 2))
    assert rows == expected, "files are not the whole set, once each, in global order"
    assert len(files) == size, f"expected one file per rank, got {[f.name for f in files]}"
    assert f"{len(whole)} determinants" in report


@pytest.mark.mpi
def test_carryover_refuses_a_determinant_held_twice(shared_dir):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size < 2:
        pytest.skip("needs two ranks to hold the same determinant")
    diag = _driver("run_gdb_diag")
    shared = _random_dets(5, 72, seed=3)
    args = types.SimpleNamespace(carryover_detfile=str(shared_dir / "dup-"), carryover_type=2,
                                 b_comm_size=size, bit_length=BITS)
    with pytest.raises(RuntimeError, match="twice"):
        diag.write_carryover(args, shared, 72, rank, size)


@pytest.mark.mpi
def test_carryover_write_clears_a_larger_earlier_set(shared_dir):
    """Fewer files than an earlier run under the same prefix leaves none of the old ones."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    diag = _driver("run_gdb_diag")
    prefix = shared_dir / "co-"
    if rank == 0:
        for i in range(size + 3):
            (shared_dir / f"co-{i:06d}.txt").write_text("stale\n")
    comm.Barrier()
    # Only rank 0 has anything, so the write uses as many files as ranks hold data.
    det = _random_dets(3, 72, seed=5) if rank == 0 else np.zeros((0, 2), dtype=np.uint64)
    args = types.SimpleNamespace(carryover_detfile=str(prefix), carryover_type=2,
                                 b_comm_size=size, bit_length=BITS)
    diag.write_carryover(args, det, 72, rank, size)
    comm.Barrier()
    files, rows = _read_files(prefix, 72)
    assert "stale" not in rows
    assert len(rows) == 3


@pytest.mark.mpi
@pytest.mark.parametrize("b_comm_size", ["one", "all"])
def test_saved_state_resplits_for_another_layout(shared_dir, b_comm_size):
    """A state saved on every rank is re-split into the shards a resumed run needs."""
    from mpi4py import MPI

    from sbd import analysis

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if not (H2O_DIR / "fcidump.txt").exists():
        pytest.skip("vendored upstream data not found (submodule not checked out?)")
    fcidump = sbd.LoadFCIDump(str(H2O_DIR / "fcidump.txt"))
    norb = int(fcidump.header["NORB"])
    alpha = [line.strip() for line in (H2O_DIR / "h2o-1em3-alpha.txt").read_text().split()][:12]
    interleave = _driver("run_gdb_heatbath").interleave
    interleaved = [interleave(sa, sb) for sa in alpha for sb in alpha]
    full = sbd.sort_bitarray_array(sbd.from_strings(interleaved, BITS, 2 * norb))
    q, r = divmod(len(full), size)
    lo = rank * q + min(rank, r)
    config = sbd.GDB_SBD()
    config.eps, config.max_it, config.bit_length, config.b_comm_size = 1e-10, 200, BITS, size
    prefix = str(shared_dir / "st-")
    sbd.gdb_diag(fcidump, full[lo:lo + q + (1 if rank < r else 0)], config, savename=prefix)
    comm.Barrier()

    target = 1 if b_comm_size == "one" else size
    det = analysis.load_gdb_shards(prefix, norb, BITS, target, comm)
    if target == 1:
        assert np.array_equal(det, full), "b_comm_size 1 must give every rank the whole basis"
    else:
        pieces = comm.allgather(det)
        assert np.array_equal(np.concatenate([p for p in pieces if len(p)]), full)
        sizes = [len(p) for p in pieces]
        assert max(sizes) - min(sizes) <= 1, f"shards not balanced: {sizes}"


@pytest.mark.mpi
def test_promote_and_copy_state_are_collective(shared_dir):
    """Every rank's shard lands under the target, and the JSON beside them, once."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    hb = _driver("run_gdb_heatbath")
    scratch, target, copy = (str(shared_dir / n) for n in ("scr-", "ck-", "ck-_r0001-"))
    # A larger earlier state under the target, to be replaced.
    if rank == 0:
        for i in range(size + 2):
            (shared_dir / f"ck-{i:06d}.bin").write_bytes(b"old")
    comm.Barrier()
    (shared_dir / f"scr-{rank:06d}.bin").write_bytes(f"rank {rank}".encode())
    hb.promote_state(scratch, target, comm, extra={"round": 1})
    hb.copy_state(target, copy, comm)

    names = sorted(p.name for p in shared_dir.iterdir())
    expected = sorted([f"ck-{i:06d}.bin" for i in range(size)] + ["ck-.json"]
                      + [f"ck-_r0001-{i:06d}.bin" for i in range(size)] + ["ck-_r0001-.json"])
    assert names == expected
    for i in range(size):
        assert (shared_dir / f"ck-{i:06d}.bin").read_bytes() == f"rank {i}".encode()


@pytest.mark.mpi
def test_promote_never_leaves_no_complete_state(shared_dir):
    """At the end the previous state is gone, and only the new one remains."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    hb = _driver("run_gdb_heatbath")
    target = str(shared_dir / "ck-")
    for round_number in (1, 2):
        (shared_dir / f"scr-{rank:06d}.bin").write_bytes(f"round {round_number}".encode())
        hb.promote_state(str(shared_dir / "scr-"), target, comm, extra={"round": round_number})
    names = sorted(p.name for p in shared_dir.iterdir())
    assert names == sorted([f"ck-{i:06d}.bin" for i in range(size)] + ["ck-.json"])
    assert (shared_dir / "ck-000000.bin").read_bytes() == b"round 2"
    assert hb.resume_prefix(target) == target
