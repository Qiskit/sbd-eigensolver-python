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

"""Check GDB against TPB on a subspace both can express.

GDB spans the subspace with the determinants it is handed; TPB spans it with the
Cartesian product of an alpha and a beta determinant list. The product is a
subspace GDB can also express -- interleave every (alpha, beta) pair into one
full determinant -- so the two solvers can be pointed at exactly the same
Hilbert space and their energies must agree. That equality is the anchor here,
and it needs no pinned reference value.

It also gives GDB a ground-truth number for free. The energies published with
the upstream TPB data are for the full product of the alpha list with itself, so
the full interleave of ``h2o-1em3-alpha.txt`` must reproduce the published
-76.23594663 -- see ``test_reference_energies.py``, which asserts the same value
through TPB.

Before this file, GDB had no test anywhere: not in the wrapper, and not in
``vendor/sbd-upstream/tests/functionality`` apart from upstream's
``test_gdb_grid_distribution.cc``, which exercises the distribution helpers
rather than a diagonalization.
"""

from __future__ import annotations

import itertools
import pathlib
import subprocess
import sys

import pytest

import sbd

# One word per determinant for h2o either way: 24 orbitals is 24 bits of alpha
# (TPB's half determinants) and 48 bits of alpha+beta (GDB's full ones), both
# under 64. ``det_vector::init_elem_size`` fixes the word count process-wide on
# first use, so every case in this file -- and in test_reference_energies.py,
# which shares the process -- must agree on it.
BIT_LENGTH = 64

# Enough alpha strings to make a non-trivial subspace, few enough that the
# product stays small: the |A|^2 scaling is the reason this is 24 and not 275.
ALPHA_LIMIT = 24

# Published for the full 275 x 275 product of h2o-1em3-alpha.txt.
H2O_1EM3_ENERGY = -76.23594663


def _interleave(alpha: str, beta: str) -> str:
    """Interleave two norb-bit strings into one 2*norb-bit GDB determinant.

    Bit ``2 * i`` is alpha orbital ``i`` and bit ``2 * i + 1`` is beta orbital
    ``i``, counting bits from the right -- the order the GDB app's README
    specifies ("the rightmost bit corresponds to alpha-spin orbital 1, the next
    to beta-spin orbital 1") and the order ``from_string`` then packs.
    """
    a_rev, b_rev = alpha[::-1], beta[::-1]
    return "".join(a_rev[i] + b_rev[i] for i in range(len(a_rev)))[::-1]


def _alpha_strings(path, limit=None):
    with open(path, encoding="utf-8") as handle:
        strings = [line.strip() for line in handle if line.strip()]
    return strings[:limit] if limit else strings


def _tpb_energy(backend, fcidump, alpha_strings, norb, **overrides):
    """Diagonalize the product of ``alpha_strings`` with itself, via TPB."""
    config = backend.TPB_SBD()
    config.eps = 1e-10
    config.max_it = 200
    config.bit_length = BIT_LENGTH
    for name, value in overrides.items():
        setattr(config, name, value)
    half = backend.sort_bitarray(
        [backend.from_string(s, BIT_LENGTH, norb) for s in alpha_strings]
    )
    return sbd.tpb_diag(fcidump, half, half, config)["energy"]


# Placement is a property of the call, not of the solver config, so these are
# kwargs on gdb_diag rather than fields to setattr onto GDB_SBD.
_CALL_KWARGS = ("determinant_distribution", "determinant_grid_a", "determinant_grid_b")


def _gdb_result(backend, fcidump, det, norb, **overrides):
    """Diagonalize an explicit determinant list (or this rank's shard) via GDB.

    ``det`` may be a list of bitstrings, which is packed here, or an already
    packed ``(n, words)`` array.
    """
    config = backend.GDB_SBD()
    config.eps = 1e-10
    config.max_it = 200
    config.bit_length = BIT_LENGTH
    call = {k: overrides.pop(k) for k in list(overrides) if k in _CALL_KWARGS}
    for name, value in overrides.items():
        setattr(config, name, value)
    # Bitstrings need packing; anything else is already packed, as an array or as
    # the nested sequence the binding's forcecast accepts.
    if len(det) and isinstance(det[0], str):
        det = sbd.from_strings(list(det), BIT_LENGTH, 2 * norb)
    return sbd.gdb_diag(fcidump, det, config, **call)


def _gdb_energy(backend, fcidump, det, norb, **overrides):
    """The energy alone, for the many cases that assert only on it."""
    return _gdb_result(backend, fcidump, det, norb, **overrides)["energy"]


def _shard(full, b_comm_size, rank):
    """This rank's slice of a globally sorted list.

    Mirrors SBD's own ``q = N/p``, remainder-to-the-low-ranks split
    (``balanced_begin``, framework/bit_manipulation.h:1523-1531), and indexes by
    ``rank % b_comm_size`` because that is the b_comm position.
    """
    if b_comm_size == 1:
        return full
    index = rank % b_comm_size
    quotient, remainder = divmod(len(full), b_comm_size)
    begin = index * quotient + min(index, remainder)
    end = begin + quotient + (1 if index < remainder else 0)
    return full[begin:end]


@pytest.fixture(scope="module")
def h2o(data_dir, backend):
    """The h2o FCIDUMP, its orbital count, the alpha-list path, and a truncation."""
    molecule_dir = data_dir / "h2o"
    fcidump = backend.LoadFCIDump(str(molecule_dir / "fcidump.txt"))
    norb = int(fcidump.header["NORB"])
    alpha_path = molecule_dir / "h2o-1em3-alpha.txt"
    return fcidump, norb, alpha_path, _alpha_strings(alpha_path, ALPHA_LIMIT)


def test_gdb_matches_tpb_on_the_same_subspace(backend, h2o):
    """Both solvers give the same energy for the same Hilbert space.

    The strong form of the check: no reference value, so it cannot pass by
    coincidence with a wrong Hamiltonian or a misread FCIDUMP. Only the
    determinant *representation* differs -- |A| x |A| half determinants against
    |A|^2 interleaved full ones.
    """
    fcidump, norb, _, alpha = h2o
    product = [_interleave(a, b) for a, b in itertools.product(alpha, alpha)]
    assert len(product) == len(alpha) ** 2

    tpb = _tpb_energy(backend, fcidump, alpha, norb)
    gdb = _gdb_energy(backend, fcidump, product, norb)
    assert gdb == pytest.approx(tpb, abs=1e-9)


def test_gdb_rejects_duplicate_determinants(backend, h2o):
    """A repeated determinant is an error, not a silently smaller subspace.

    ``sort_bitarray`` deduplicates, so the binding compares sizes afterwards and
    raises. Recorded here because TPB takes the opposite branch and drops
    duplicates silently -- the divergence tracked in issue #31. If that is ever
    reconciled in GDB's favour, this test is what has to change.
    """
    fcidump, norb, _, alpha = h2o
    doubled = [_interleave(alpha[0], alpha[0])] * 2
    with pytest.raises(ValueError, match="distinct determinants"):
        _gdb_energy(backend, fcidump, doubled, norb)


def test_gdb_rejects_a_basis_split_the_ranks_cannot_tile(backend, h2o):
    """b_comm_size > 1 needs ranks to match it.

    Splitting the basis is supported now, but the grid still has to tile: upstream
    derives the helper dimension as ``ranks / (t * b)`` by integer division and
    never checks the remainder, so asking for more basis blocks than there are
    ranks would silently produce communicators of unequal size. Serially that
    means any b_comm_size above 1 is refused.
    """
    fcidump, norb, _, alpha = h2o
    product = [_interleave(alpha[0], b) for b in alpha]
    with pytest.raises(ValueError, match="divide the rank count"):
        _gdb_energy(backend, fcidump, product, norb, b_comm_size=2)


def test_gdb_rejects_lanczos_methods(backend, h2o):
    """Methods 2 and 3 are refused rather than returning an uninitialized energy.

    They are valid for TPB, where they select Lanczos. GDB has no Lanczos:
    ``gdb::diag`` handles only ``method == 0`` and ``method == 1`` and assigns
    ``energy`` only inside those branches, so passing 2 through would return
    whatever was on the stack -- with no diagonalization having run.
    """
    fcidump, norb, _, alpha = h2o
    product = [_interleave(alpha[0], b) for b in alpha]
    with pytest.raises(ValueError, match="method 0 or 1"):
        _gdb_energy(backend, fcidump, product, norb, method=2)


@pytest.mark.slow
def test_gdb_reproduces_the_published_h2o_energy(backend, h2o):
    """The full interleave reproduces the energy published for this alpha list.

    The subspace is 275^2 = 75,625 determinants -- the same Hilbert space
    ``test_reference_energies.py`` covers through TPB in its fast tier, but GDB
    carries its dimension explicitly rather than as a product, so the determinant
    list itself is 75,625 entries. Marked slow for the cost of that rather than
    for being intractable: measured at 5.5 s of diagonalization plus 0.2 s to
    build the list, on 8 host threads, returning -76.2359466308 against the
    published -76.23594663. Promote it to the fast tier if that budget is fine.
    """
    fcidump, norb, alpha_path, _ = h2o
    # The full list, not the fixture's ALPHA_LIMIT truncation: the published
    # energy is for the product of all of it with itself.
    alpha = _alpha_strings(alpha_path)
    product = [_interleave(a, b) for a, b in itertools.product(alpha, alpha)]
    assert len(product) == len(alpha) ** 2

    energy = _gdb_energy(backend, fcidump, product, norb)
    # Quoted to eight decimals upstream, so compared to that rather than to the
    # solver's own 1e-10 convergence tolerance.
    assert energy == pytest.approx(H2O_1EM3_ENERGY, abs=1e-8)


@pytest.mark.mpi
def test_gdb_under_mpi_matches_tpb_on_the_same_subspace(backend, h2o):
    """Distributing GDB over ranks does not change the answer.

    GDB decomposes as ``t_comm_size x b_comm_size x helper``. Both named
    dimensions are pinned to 1 on the in-memory path -- b_comm_size because every
    rank passes the whole determinant list, t_comm_size because GDB enumerates one
    task per b_comm rank and so cannot split a single task further (see
    ``test_gdb_rejects_a_task_split``). Ranks therefore all land on the derived
    helper dimension, and this checks that spending them there is harmless.

    The comparison is against a TPB run of the same subspace in the same process,
    which reaches the same Hilbert space through an independent decomposition
    (``adet_comm_size``). No pinned reference value is needed, and a broken helper
    distribution shows up as disagreement.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    fcidump, norb, _, alpha = h2o
    product = [_interleave(a, b) for a, b in itertools.product(alpha, alpha)]

    gdb = _gdb_energy(backend, fcidump, product, norb)
    tpb = _tpb_energy(backend, fcidump, alpha, norb, adet_comm_size=size)

    if comm.Get_rank() == 0:
        assert gdb == pytest.approx(tpb, abs=1e-9), (
            f"{size} ranks (helper={size}) gave {gdb}, but TPB on the same "
            f"subspace gave {tpb}"
        )


def test_gdb_rejects_a_task_split(backend, h2o):
    """t_comm_size > 1 is refused rather than segfaulting.

    ``t_comm_size <= b_comm_size`` is structural, not incidental. GDB's matvec
    rotates the ket around ``b_comm`` as a ring (gdb/mult.h:39-41, :196-202), so
    the ring has exactly ``b_comm_size`` stations and one "task" is one station;
    ``t_comm`` parallelizes stations, and cannot have more workers than there are
    stations. ``MakeHelpers`` encodes this as ``task_end = mpi_size_b`` split
    across ``t_comm`` (gdb/helper.h:736-739).

    Upstream does not check it: a starved rank resizes ``exidx`` to zero and then
    reads ``exidx[0].slide`` anyway because its ``task_begin`` is nonzero
    (helper.h:761), which is a null dereference during helper construction --
    observed as EXC_BAD_ACCESS at 0x0 on 2 and 4 ranks before this guard existed.
    With ``b_comm_size`` pinned to 1 in memory, that makes any ``t_comm_size > 1``
    fatal.

    The check runs serially because the guard is a configuration check that
    precedes any communication.
    """
    fcidump, norb, _, alpha = h2o
    product = [_interleave(alpha[0], b) for b in alpha]
    with pytest.raises(ValueError, match="t_comm_size <= b_comm_size"):
        _gdb_energy(backend, fcidump, product, norb, t_comm_size=2)


# ---------------------------------------------------------------------------
# Distributed basis: b_comm_size > 1, where each rank passes only its shard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2o_product(backend, h2o):
    """The truncated product basis, packed and globally sorted."""
    fcidump, norb, _, alpha = h2o
    strings = [_interleave(a, b) for a, b in itertools.product(alpha, alpha)]
    packed = sbd.sort_bitarray_array(
        sbd.from_strings(strings, BIT_LENGTH, 2 * norb)
    )
    assert packed.shape[0] == len(alpha) ** 2
    return packed


def test_from_strings_matches_the_per_determinant_form(backend, h2o):
    """The bulk packer agrees elementwise with from_string called in a loop.

    from_strings exists because the per-call form costs one boundary crossing per
    determinant, which issue #31 measures as a real cost at tens of thousands of
    strings. It is only worth having if it is the same function, hence this.
    """
    fcidump, norb, _, alpha = h2o
    strings = [_interleave(a, b) for a, b in itertools.product(alpha[:6], alpha[:6])]
    bulk = sbd.from_strings(strings, BIT_LENGTH, 2 * norb)
    one_by_one = [backend.from_string(s, BIT_LENGTH, 2 * norb) for s in strings]
    assert bulk.tolist() == one_by_one


def test_gdb_accepts_a_nested_list(backend, h2o, h2o_product):
    """A list of lists still works, so existing callers are unaffected.

    The binding takes a forcecast array, which converts a nested sequence, so the
    numpy contract is an addition rather than a break.
    """
    fcidump, norb, _, _ = h2o
    as_list = [list(row) for row in h2o_product]
    assert _gdb_energy(backend, fcidump, as_list, norb) == pytest.approx(
        _gdb_energy(backend, fcidump, h2o_product, norb), abs=1e-12
    )


def test_gdb_reports_the_dimensions_it_diagonalized(backend, h2o, h2o_product):
    """global_dim is what the caller needs to check completeness itself.

    Whether the union of shards is the basis the caller *meant* cannot be checked
    inside the binding, so the dimension is reported back instead.
    """
    fcidump, norb, _, _ = h2o
    result = _gdb_result(backend, fcidump, h2o_product, norb)
    assert result["global_dim"] == h2o_product.shape[0]
    assert result["local_dim"] == h2o_product.shape[0]
    assert result["determinant_distribution"] == "equal-bra-a"


@pytest.mark.mpi
def test_gdb_sharded_basis_matches_the_whole_basis(backend, h2o, h2o_product):
    """Splitting the basis across b_comm does not change the energy.

    The core claim of the distributed path: each rank passes only its slice, and
    the answer still matches TPB on the same subspace. Checked against a TPB run
    in this same process, which reaches the same Hilbert space through an
    independent decomposition, so no pinned value is involved.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    fcidump, norb, _, alpha = h2o

    gdb = _gdb_energy(backend, fcidump, _shard(h2o_product, size, comm.Get_rank()),
                      norb, b_comm_size=size)
    tpb = _tpb_energy(backend, fcidump, alpha, norb, adet_comm_size=size)
    if comm.Get_rank() == 0:
        assert gdb == pytest.approx(tpb, abs=1e-9)


@pytest.mark.mpi
def test_gdb_task_dimension_works_once_the_ring_has_stations(backend, h2o, h2o_product):
    """t_comm_size > 1 becomes usable exactly when b_comm_size allows it.

    ``t <= b`` is structural -- one task per basis-ring station -- so with the
    basis in one block the task dimension is unreachable. Shard the basis and it
    opens up. Needs at least 4 ranks for t=2, b=2.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size < 4 or size % 4:
        pytest.skip(f"needs a rank count divisible by 4 for t=2 x b=2, got {size}")
    fcidump, norb, _, alpha = h2o

    gdb = _gdb_energy(backend, fcidump, _shard(h2o_product, 2, comm.Get_rank()),
                      norb, b_comm_size=2, t_comm_size=2)
    tpb = _tpb_energy(backend, fcidump, alpha, norb, adet_comm_size=size)
    if comm.Get_rank() == 0:
        assert gdb == pytest.approx(tpb, abs=1e-9)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "scheme",
    ["input", "equal-bra-a", "count", "count-sorted",
     "grid-cyclic", "grid-cyclic-balanced"],
)
def test_gdb_placement_does_not_change_the_energy(backend, h2o, h2o_product, scheme):
    """All six placement schemes agree.

    Placement is a load-balancing decision -- which rank owns which determinants,
    and how the ring is laid out -- so it must not move the answer. ``input`` keeps
    the caller's slices, ``equal-bra-a`` equalizes distinct alpha strings (what
    actually balances the matvec, whose outer loop is over alpha), ``count`` and
    ``count-sorted`` equalize determinant counts, and the two grid-cyclic schemes
    spread alpha and beta keys over a 2-D rank grid.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    fcidump, norb, _, alpha = h2o

    gdb = _gdb_energy(backend, fcidump, _shard(h2o_product, size, comm.Get_rank()),
                      norb, b_comm_size=size, determinant_distribution=scheme)
    tpb = _tpb_energy(backend, fcidump, alpha, norb, adet_comm_size=size)
    if comm.Get_rank() == 0:
        assert gdb == pytest.approx(tpb, abs=1e-9)


@pytest.mark.mpi
def test_gdb_rejects_shards_that_are_not_disjoint(backend, h2o, h2o_product):
    """Overlapping shards are refused, not silently diagonalized.

    Every rank passing the whole list while claiming b_comm_size > 1 is the
    mistake this catches: each rank would believe its copy was a shard, and the
    ring would rotate duplicated blocks.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size < 2:
        pytest.skip("needs at least 2 ranks to have distinct shards")
    fcidump, norb, _, _ = h2o
    with pytest.raises(ValueError, match="globally sorted and disjoint"):
        _gdb_energy(backend, fcidump, h2o_product, norb, b_comm_size=size)


@pytest.mark.mpi
def test_gdb_rejects_a_rank_count_the_grid_does_not_tile(backend, h2o, h2o_product):
    """t * b must divide the rank count exactly.

    Upstream derives the helper dimension by integer division and never checks the
    remainder, which silently yields communicators of unequal size and a rank alone
    in its own basis ring. Refuse it instead.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    fcidump, norb, _, _ = h2o
    bad = size + 1
    with pytest.raises(ValueError, match="divide the rank count"):
        _gdb_energy(backend, fcidump, _shard(h2o_product, bad, comm.Get_rank()),
                    norb, b_comm_size=bad)


def test_gdb_rejects_an_unknown_placement_scheme(backend, h2o, h2o_product):
    """A misspelled scheme names the alternatives rather than falling back."""
    fcidump, norb, _, _ = h2o
    with pytest.raises(ValueError, match="unknown determinant_distribution"):
        _gdb_energy(backend, fcidump, h2o_product, norb,
                    determinant_distribution="equal-bra-alpha")


def test_gdb_accepts_underscores_in_the_scheme_name(backend, h2o, h2o_product):
    """``grid_cyclic`` and ``grid-cyclic`` are the same scheme, as upstream has it."""
    fcidump, norb, _, _ = h2o
    result = _gdb_result(backend, fcidump, h2o_product, norb,
                         determinant_distribution="grid_cyclic")
    assert result["determinant_distribution"] == "grid-cyclic"


@pytest.mark.parametrize(
    "grid_a,grid_b,scheme,message",
    [
        (2, 0, "grid-cyclic", "both determinant grid dimensions or neither"),
        (3, 3, "grid-cyclic", "multiply to b_comm_size"),
        (1, 1, "equal-bra-a", "require a grid-cyclic distribution"),
    ],
)
def test_gdb_validates_the_determinant_grid(backend, h2o, h2o_product,
                                            grid_a, grid_b, scheme, message):
    """The grid dimensions are checked against b_comm_size before any work.

    ``redistribution_grid_bra_ab_cyclic`` validates the product itself, but against
    the communicator it is handed, so checking here names b_comm_size in the error.
    """
    fcidump, norb, _, _ = h2o
    with pytest.raises(ValueError, match=message):
        _gdb_energy(backend, fcidump, h2o_product, norb,
                    determinant_distribution=scheme,
                    determinant_grid_a=grid_a, determinant_grid_b=grid_b)


# Upstream's only general-determinant data, and the only case here at a realistic
# size. It lives in the app directory rather than under data/, which is alpha-only.
FE4S4_DIR = (
    "vendor/sbd-upstream/apps/chemistry_gdb_selected_basis_diagonalization"
)
# Measured by this wrapper, NOT published upstream: run.sh ships no expected value
# and data/ has reference tables only for the TPB molecules. So this is a
# regression guard against our own verified result, not an independent check. It
# was identical across b_comm_size 1, 2 and 4 and across t_comm_size 1 and 2.
FE4S4_ENERGY = -326.6982518821


@pytest.mark.slow
def test_gdb_fe4s4_at_realistic_size():
    """59,536 determinants over 36 orbitals, upstream's own GDB input.

    Everything else here is h2o at a few hundred to tens of thousands of
    determinants; this is the only case with a real determinant list and a real
    orbital count, and the only one whose four input files are a ready-made
    four-way partition (equal sizes, disjoint, individually sorted, concatenation
    globally sorted). Roughly 21 s on 8 host threads.

    Runs in a **subprocess**, because ``det_vector``'s row width is fixed for the
    lifetime of a process: h2o packs into one 64-bit word (48 bits) and Fe4S4 needs
    two (72 bits), so the two cannot share a process. That is the constraint
    ``gdb_diag`` reports rather than a problem with this case, and forking is the
    workaround it tells callers to use.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / FE4S4_DIR
    if not root.is_dir():
        pytest.skip(f"upstream GDB app data not found at {root}")

    script = f"""
import sbd
root = {str(root)!r}
backend = sbd.get_backend()
fcidump = backend.LoadFCIDump(root + "/fcidump_Fe4S4.txt")
norb = int(fcidump.header["NORB"])
strings = []
for index in range(4):
    with open(root + "/det%d.txt" % index) as handle:
        strings.extend(line.strip() for line in handle if line.strip())
assert norb == 36, norb
assert len(strings) == 59536, len(strings)
config = backend.GDB_SBD()
config.eps, config.max_it, config.bit_length = 1e-6, 30, {BIT_LENGTH}
det = sbd.from_strings(strings, {BIT_LENGTH}, 2 * norb)
result = sbd.gdb_diag(fcidump, det, config)
print("ENERGY", repr(result["energy"]), "DIM", result["global_dim"])
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=1800,
    )
    assert completed.returncode == 0, (
        f"subprocess failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
    )
    line = [l for l in completed.stdout.splitlines() if l.startswith("ENERGY")]
    assert line, f"no energy reported:\n{completed.stdout[-2000:]}"
    _, energy, _, dim = line[-1].split()
    assert int(dim) == 59536
    assert float(energy) == pytest.approx(FE4S4_ENERGY, abs=1e-8)
