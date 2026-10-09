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

"""Exercise the qiskit-addon-sqd-facing entry points, serially and under MPI.

Two things are checked here that ``test_reference_energies.py`` does not cover.

The first is ``solve_sci_batch``, the wrapper qiskit-addon-sqd is meant to be handed
as its ``sci_solver``. Unlike ``tpb_diag_from_files``, it takes the Hamiltonian as
in-memory tensors, so with no ``fcidump_path`` it has rank 0 write a regenerated
FCIDUMP into a temporary directory and broadcast the path for every rank to open.
That regenerate-and-broadcast step is the part most specific to a multi-rank run, and
it is exercised here by leaving ``fcidump_path`` unset -- the default, and what a
caller coming through ``diagonalize_fermionic_hamiltonian`` gets.

The second is ``diagonalize_fermionic_hamiltonian`` itself, which closes the loop:
sample, recover configurations, subsample, diagonalize with SBD, carry over, repeat.
Running it here means an upstream change to the ``sci_solver`` contract or to
``SCIResult``/``SCIState`` surfaces as a failure in this repository rather than in a
user's script.

Every test is written to be indifferent to the process count -- the alpha-determinant
grid is sized from ``MPI.COMM_WORLD``, which is 1 in a single process -- so each body
is shared by a plain variant and an ``mpi``-marked one. pytest-mpi filters on that
marker in opposite directions depending on the flag (``--only-mpi`` skips what is not
marked, no flag skips what is), so a single test function cannot run in both modes;
two thin wrappers around one body can. ``tox -e py`` runs the ``_standalone``
variants, ``tox -e mpi`` runs the ``_mpi`` ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# h2o, cc-pvdz: 24 spatial orbitals, 10 electrons, Ms=0.
NORB = 24
NELEC = (5, 5)

# Electronic energy over the first 40 alpha determinants of the 1em3 selection, used as
# both the alpha and the beta set. Not a published figure -- the published tables cover
# whole selections, not truncations of them -- but recorded here after checking it is
# identical on 1, 2 and 4 ranks, which is the property the test exists to defend. A
# small subspace keeps the always-on case cheap; test_published_energy_* below carries
# the independent reference.
SMALL_SUBSPACE_DETS = 40
SMALL_SUBSPACE_ENERGY = -85.29400074571684

# Electronic energy over the full 1em3 selection. The corresponding total energy,
# -76.23594663, is the value published in vendor/sbd-upstream/data/h2o/README.md and
# already checked through tpb_diag_from_files by test_reference_energies; reaching the
# same number through solve_sci_batch is what makes this an independent check of the
# wrapper rather than of SBD.
PUBLISHED_TOTAL_ENERGY = -76.23594663

# Bits SBD packs into each size_t of a determinant. 63 is the largest legal value:
# ``bitadvance()`` in framework/bit_manipulation.h computes ``(((size_t) 1) << bit_length)
# - 1``, so 64 shifts a 64-bit size_t by 64 -- undefined behavior that in practice
# collapses the mask to 0. It is reached from ``mpi_redistribution()`` and
# ``mpi_sort_bitarray()``, precisely the paths the mpi tests below exercise, so 64 is not
# an option here even though test_reference_energies still pins it (that predates the UB
# being identified, and it goes through tpb_diag_from_files rather than these).
#
# 63 rather than the wrapper's own default of 20 because the packed word count --
# ``ceil(2 * norb / bit_length)`` -- is a process-wide constant: SBD fixes it in an inline
# static (``det_vector::_elem_size``) on the first diagonalization and throws
# "det_vector: elem_size mismatch" for any later one implying a different count. At 63,
# h2o's 24 orbitals need one word, which is what test_reference_energies' 64 also gives,
# so the two modules can share a process. Choosing 20 here instead needs two words and
# fails the moment both files run together.
#
# The tradeoff is that this leaves multi-word packing unexercised, which is what a caller
# taking SBD_DEFAULT_BIT_LENGTH (20) actually gets. Covering that means a module that does
# not share a process with these -- worth doing, but not at the cost of a suite that
# cannot run. It is not a performance question either way: at 275 determinants the solve
# takes 1.9s at 63 against 2.1s at 20.
BIT_LENGTH = 63

# Davidson settings shared by the deterministic cases: a tolerance well below the
# precision the reference is quoted to, so a mismatch means a wrong answer rather than an
# unconverged one.
SOLVER_CONFIG = {"eps": 1e-10, "max_it": 200, "bit_length": BIT_LENGTH}


def _sbd_config(comm, **overrides) -> dict:
    """SBD configuration for a run spread over ``comm``.

    The alpha-determinant grid takes the whole communicator and the other two axes are
    left at 1, so the same call works on one rank or many without the test having to
    know which.
    """
    config = dict(
        SOLVER_CONFIG,
        adet_comm_size=comm.Get_size(),
        bdet_comm_size=1,
        task_comm_size=1,
    )
    config.update(overrides)
    return config


def _read_alpha_determinants(path: Path, limit: int | None = None) -> np.ndarray:
    """Read a whitespace-separated file of binary determinant strings as integers."""
    strings = path.read_text().split()
    if limit is not None:
        strings = strings[:limit]
    return np.array([int(s, 2) for s in strings], dtype=np.int64)


def _load_hamiltonian(data_dir):
    """Return ``(hcore, eri, nuclear_repulsion)`` for h2o, read from the FCIDUMP."""
    pyscf = pytest.importorskip("pyscf", reason="pyscf is needed to read the FCIDUMP")
    from pyscf import ao2mo, tools

    del pyscf
    mean_field = tools.fcidump.to_scf(str(data_dir / "h2o" / "fcidump.txt"))
    hcore = mean_field.get_hcore()
    eri = ao2mo.restore(1, mean_field._eri, NORB)  # pylint: disable=protected-access
    return hcore, eri, mean_field.mol.energy_nuc()


def _diagonalize_subspace(data_dir, device_config, n_dets: int | None):
    """Diagonalize a fixed subspace through ``solve_sci_batch``.

    Returns the ``SCIResult`` together with the communicator and the nuclear repulsion,
    leaving the caller to decide what to assert and on which rank.
    """
    from mpi4py import MPI

    from sbd.sbd_solver import solve_sci_batch

    comm = MPI.COMM_WORLD
    hcore, eri, nuclear_repulsion = _load_hamiltonian(data_dir)
    strings = _read_alpha_determinants(
        data_dir / "h2o" / "h2o-1em3-alpha.txt", limit=n_dets
    )

    # No fcidump_path: rank 0 regenerates the FCIDUMP and broadcasts where it put it.
    results = solve_sci_batch(
        [(strings, strings)],
        hcore,
        eri,
        norb=NORB,
        nelec=NELEC,
        sbd_config=_sbd_config(comm),
        device_config=device_config,
    )

    assert len(results) == 1
    return results[0], comm, nuclear_repulsion


def _assert_result_is_consistent(result, nelec=NELEC):
    """Checks that hold for any converged result, independent of the subspace.

    ``sci_state`` describes the determinants carried over for the *next* iteration
    rather than the ones just diagonalized, so its shape is checked against its own
    determinant lists instead of against the input.
    """
    alpha_occupancies, beta_occupancies = result.orbital_occupancies
    assert alpha_occupancies.shape == (NORB,)
    assert beta_occupancies.shape == (NORB,)
    assert alpha_occupancies.sum() == pytest.approx(nelec[0], abs=1e-6)
    assert beta_occupancies.sum() == pytest.approx(nelec[1], abs=1e-6)

    # With carryover_type=1 the solver reports SBD's carryover instead of the
    # eigenvector, so there is no state to check.
    state = result.sci_state
    if state is not None:
        assert state.amplitudes.shape == (len(state.ci_strs_a), len(state.ci_strs_b))
        assert np.isfinite(state.amplitudes).all()


# --- solve_sci_batch over a fixed subspace -------------------------------------------
#
# Deterministic: the determinants are given rather than sampled, so the energy is a
# fixed number and may be pinned. Running the same body on one rank and on many is
# what demonstrates that distributing the subspace does not change the answer.


def _check_small_subspace(data_dir, device_config):
    result, comm, _ = _diagonalize_subspace(data_dir, device_config, SMALL_SUBSPACE_DETS)

    # Only rank 0 is given the energy and the wavefunction; the rest get placeholders.
    if comm.Get_rank() != 0:
        return

    assert result.energy == pytest.approx(SMALL_SUBSPACE_ENERGY, abs=1e-8)
    _assert_result_is_consistent(result)


def test_small_subspace_standalone(data_dir, device_config):
    """A fixed 40-determinant subspace gives the recorded energy in one process."""
    _check_small_subspace(data_dir, device_config)


@pytest.mark.mpi
def test_small_subspace_mpi(data_dir, device_config):
    """The same subspace gives the same energy spread across the launched ranks."""
    _check_small_subspace(data_dir, device_config)


def _check_published_energy(data_dir, device_config):
    result, comm, nuclear_repulsion = _diagonalize_subspace(data_dir, device_config, None)

    if comm.Get_rank() != 0:
        return

    # solve_sci_batch reports the electronic energy alone: the regenerated FCIDUMP
    # carries ECORE=0, and callers add the nuclear repulsion themselves. The published
    # figure is a total energy, so it has to go back in before comparing.
    total_energy = result.energy + nuclear_repulsion
    assert total_energy == pytest.approx(PUBLISHED_TOTAL_ENERGY, abs=1e-8)
    _assert_result_is_consistent(result)


@pytest.mark.slow
def test_published_energy_standalone(data_dir, device_config):
    """The full 1em3 selection reproduces the published energy in one process."""
    _check_published_energy(data_dir, device_config)


@pytest.mark.mpi
@pytest.mark.slow
def test_published_energy_mpi(data_dir, device_config):
    """The full 1em3 selection reproduces the published energy across ranks."""
    _check_published_energy(data_dir, device_config)


# --- the full qiskit-addon-sqd loop --------------------------------------------------
#
# Sampled rather than fixed, so the subspace depends on upstream's recovery and
# subsampling. The energy is therefore bracketed rather than pinned: this case is here
# to catch the loop failing or the solver contract drifting, and the deterministic
# tests above are what guard the number.


def _check_diagonalize_fermionic_hamiltonian(
    data_dir, device_config, counts_path, carryover_type=0
):
    """Run the qiskit-addon-sqd loop with SBD as the solver.

    ``carryover_type=1`` drives the SBD-selected carryover path end to end: the loop
    must accept ``sci_state=None``, take SBD's carryover in place of its own
    threshold, and build the next iteration from it. ``symmetrize_spin`` is on, and
    under it the loop ignores ``carryover_strings_b``, so the two lists must agree --
    otherwise the beta selection would be silently dropped.
    """
    from functools import partial

    from mpi4py import MPI

    pytest.importorskip(
        "qiskit_addon_sqd",
        reason="qiskit-addon-sqd is needed for the self-consistent loop",
    )
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    from sbd.sbd_solver import _addon_accepts_carryover, solve_sci_batch

    if carryover_type and not _addon_accepts_carryover():
        pytest.skip("installed qiskit-addon-sqd has no SCIResult.carryover")

    comm = MPI.COMM_WORLD
    hcore, eri, nuclear_repulsion = _load_hamiltonian(data_dir)

    counts = json.loads(counts_path.read_text())
    bit_array = BitArray.from_counts(counts, num_bits=NORB * 2)

    solver = partial(
        solve_sci_batch,
        # Loosened from SOLVER_CONFIG's tolerance, which is set for pinning an energy to
        # eight digits; the loop only needs each iteration to converge. max_nb matches the
        # example notebook and upstream's own default. bit_length is inherited from
        # SOLVER_CONFIG and must not be overridden -- see the note there.
        sbd_config=_sbd_config(
            comm, eps=1e-8, max_it=10, max_nb=10, carryover_type=carryover_type
        ),
        device_config=device_config,
    )

    # Every iteration's results, as the loop hands them over. The callback runs on
    # the control process only, which is also the only rank that asserts below.
    iterations = []

    result = diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        bit_array,
        norb=NORB,
        nelec=NELEC,
        samples_per_batch=300,
        num_batches=1,
        max_iterations=2,
        sci_solver=solver,
        symmetrize_spin=True,
        # Seeded so a failure can be reproduced, though the assertions below do not
        # depend on which determinants the sampling happens to pick.
        seed=np.random.default_rng(42),
        callback=iterations.append,
    )

    if comm.Get_rank() != 0:
        return

    _assert_result_is_consistent(result)

    # max_iterations=2, so the second iteration was built from the first's carryover.
    assert len(iterations) == 2
    for results in iterations:
        for r in results:
            if carryover_type == 1:
                assert r.sci_state is None
                carryover_a, carryover_b = r.carryover
                assert len(carryover_a) > 0
                np.testing.assert_array_equal(carryover_a, carryover_b)
            else:
                assert r.sci_state is not None
                # SCIResult has no carryover field before qiskit-addon-sqd 0.14.0.
                assert getattr(r, "carryover", None) is None

    # A bracket rather than an equality: the subspace comes out of upstream's sampling.
    # The lower bound is the FCI energy, which no subspace of it can beat; the upper is
    # loose enough to tolerate a different selection but far below the -75.0 a badly
    # converged run gives, which is the failure worth catching.
    total_energy = result.energy + nuclear_repulsion
    assert -76.25 < total_energy < -75.9


@pytest.mark.parametrize("carryover_type", [0, 1])
def test_diagonalize_fermionic_hamiltonian_standalone(
    data_dir, device_config, counts_path, carryover_type
):
    """The self-consistent loop runs to completion in a single process."""
    _check_diagonalize_fermionic_hamiltonian(
        data_dir, device_config, counts_path, carryover_type
    )


@pytest.mark.mpi
@pytest.mark.parametrize("carryover_type", [0, 1])
def test_diagonalize_fermionic_hamiltonian_mpi(
    data_dir, device_config, counts_path, carryover_type
):
    """The self-consistent loop runs to completion across the launched ranks."""
    _check_diagonalize_fermionic_hamiltonian(
        data_dir, device_config, counts_path, carryover_type
    )


# --- SBD-selected carryover skips the wavefunction dump ------------------------------
#
# With carryover_type != 0 SBD ranks and selects the next subspace in C++, so the
# eigenvector has no consumer and is not dumped. The point of the test is the absence
# of the file: a run that quietly kept writing it would still give the right energy.


def _check_sbd_carryover_skips_wavefunction_dump(data_dir, device_config, tmp_path):
    """``carryover_type=1`` returns no SCI state and writes no wavefunction.

    Only type 1 qualifies: it ranks by the marginal weight the loop wants, so the
    eigenvector has no remaining consumer. Types 2/3 singles-extend and are re-sorted
    canonically by SBD, so they need the amplitudes to restore the required order --
    see ``test_sbd_carryover_is_weight_ordered``.
    """
    from mpi4py import MPI

    from sbd.sbd_solver import _addon_accepts_carryover, solve_sci_batch

    if not _addon_accepts_carryover():
        pytest.skip(
            "installed qiskit-addon-sqd has no SCIResult.carryover (needs #369); "
            "the solver keeps returning sci_state so the loop has something to use"
        )

    comm = MPI.COMM_WORLD
    hcore, eri, _ = _load_hamiltonian(data_dir)
    strings = _read_alpha_determinants(
        data_dir / "h2o" / "h2o-1em3-alpha.txt", limit=SMALL_SUBSPACE_DETS
    )

    results = solve_sci_batch(
        [(strings, strings)],
        hcore,
        eri,
        norb=NORB,
        nelec=NELEC,
        sbd_config=_sbd_config(comm, carryover_type=1),
        device_config=device_config,
        temp_dir=tmp_path,
        clean_temp_dir=False,
    )
    result = results[0]

    # Holds on every rank: no rank materializes the eigenvector, and each still
    # returns something the loop can use.
    assert result.sci_state is None
    assert result.carryover is not None

    # Only rank 0 is given the energy and the selected determinants.
    if comm.Get_rank() != 0:
        return

    carryover_a, carryover_b = result.carryover
    assert len(carryover_a) > 0
    assert len(carryover_b) > 0
    # The energy is unaffected by how the next subspace gets chosen.
    assert result.energy == pytest.approx(SMALL_SUBSPACE_ENERGY, abs=1e-8)
    # clean_temp_dir=False above keeps the directory, so this is a real check that
    # nothing was written rather than a check that it was tidied away.
    assert not list(tmp_path.rglob("wavefunction.bin"))


def test_sbd_carryover_skips_wavefunction_dump_standalone(
    data_dir, device_config, tmp_path
):
    """``carryover_type=1`` skips the dump in a single process."""
    _check_sbd_carryover_skips_wavefunction_dump(data_dir, device_config, tmp_path)


@pytest.mark.mpi
def test_sbd_carryover_skips_wavefunction_dump_mpi(data_dir, device_config, tmp_path):
    """``carryover_type=1`` skips the dump across the launched ranks."""
    _check_sbd_carryover_skips_wavefunction_dump(data_dir, device_config, tmp_path)


def _check_sbd_carryover_is_weight_ordered(
    data_dir, device_config, tmp_path, carryover_type
):
    """Types 2/3 keep the dump and hand over a weight-ordered carryover.

    ``SCIResult.carryover`` must be in descending marginal-weight order, because
    ``_select_carryover`` returns a solver's carryover untouched and a later
    truncation to ``max_dim`` keeps the leading entries. SBD returns types 2/3 in
    canonical order (``SinglesExtendHalfdets`` ends in ``sort_bitarray``), so the
    solver re-ranks them from the amplitudes. Without that, ``max_dim`` would keep
    the numerically smallest strings and could discard the high-weight parents.
    """
    from mpi4py import MPI

    from sbd.sbd_solver import _addon_accepts_carryover, solve_sci_batch

    if not _addon_accepts_carryover():
        pytest.skip("installed qiskit-addon-sqd has no SCIResult.carryover")

    comm = MPI.COMM_WORLD
    hcore, eri, _ = _load_hamiltonian(data_dir)
    strings = _read_alpha_determinants(
        data_dir / "h2o" / "h2o-1em3-alpha.txt", limit=SMALL_SUBSPACE_DETS
    )
    results = solve_sci_batch(
        [(strings, strings)],
        hcore,
        eri,
        norb=NORB,
        nelec=NELEC,
        sbd_config=_sbd_config(comm, carryover_type=carryover_type),
        device_config=device_config,
        temp_dir=tmp_path,
        clean_temp_dir=False,
    )
    result = results[0]

    if comm.Get_rank() != 0:
        return

    # These types need the amplitudes, so the dump is taken rather than skipped.
    assert result.sci_state is not None
    assert result.carryover is not None
    assert result.energy == pytest.approx(SMALL_SUBSPACE_ENERGY, abs=1e-8)

    probabilities = np.abs(result.sci_state.amplitudes) ** 2
    for strings_co, solved, weights in (
        (result.carryover[0], result.sci_state.ci_strs_a, probabilities.sum(axis=1)),
        (result.carryover[1], result.sci_state.ci_strs_b, probabilities.sum(axis=0)),
    ):
        assert len(strings_co) > 0
        order = np.argsort(solved, kind="stable")
        sorted_solved = solved[order]
        pos = np.minimum(
            np.searchsorted(sorted_solved, strings_co), sorted_solved.size - 1
        )
        in_subspace = sorted_solved[pos] == strings_co
        # Without this, both assertions below would pass on an empty selection.
        assert in_subspace.any()
        # Strings that were in the subspace come first, ranked by descending weight;
        # the singles-generated ones have no weight and must follow.
        assert np.all(np.diff(in_subspace.astype(int)) <= 0)
        ranked = weights[order[pos[in_subspace]]]
        assert np.all(np.diff(ranked) <= 1e-12)


@pytest.mark.parametrize("carryover_type", [2, 3])
def test_sbd_carryover_is_weight_ordered_standalone(
    data_dir, device_config, tmp_path, carryover_type
):
    """Types 2/3 hand over a weight-ordered carryover in a single process."""
    _check_sbd_carryover_is_weight_ordered(
        data_dir, device_config, tmp_path, carryover_type
    )


@pytest.mark.mpi
@pytest.mark.parametrize("carryover_type", [2, 3])
def test_sbd_carryover_is_weight_ordered_mpi(
    data_dir, device_config, tmp_path, carryover_type
):
    """Types 2/3 hand over a weight-ordered carryover across the launched ranks."""
    _check_sbd_carryover_is_weight_ordered(
        data_dir, device_config, tmp_path, carryover_type
    )


# --- dividing the processes among the batches ----------------------------------------
#
# ``_split_for_batches`` decides which batch each process works on, and its decision is
# arithmetic over ``(size, rank, num_batches)`` alone. Those can be supplied, so the
# whole decision table is checkable in a single process, for process counts larger than
# a test run would ever launch. ``MPI.Split`` is what the arithmetic is handed to; the
# tests below stand in for it so that what is under test is the division rather than
# mpi4py.
#
# The MPI tests further down then run the real thing, which is what confirms the
# arithmetic was handed to Split correctly.


class _FakeComm:
    """Enough of a communicator to drive ``_split_for_batches`` in one process.

    ``Split`` records the colors it is given rather than forming a communicator, and
    returns a stand-in whose rank is this process's position among the ranks sharing its
    color -- which is what ``Split`` guarantees for the ascending keys the caller passes.
    """

    def __init__(self, size, rank, num_batches=1):
        self._size = size
        self._rank = rank
        self._num_batches = num_batches
        self.colors = []

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank

    def Split(self, color, key):  # pylint: disable=unused-argument
        from mpi4py import MPI

        self.colors.append(color)
        if color == MPI.UNDEFINED:
            return MPI.COMM_NULL
        # Split orders a new communicator by ascending key, and the caller passes the
        # rank as the key, so the lowest-ranked member of a color becomes its rank 0.
        # The members of this color are known from the division rule itself.
        group_size = self._size // self._num_batches
        first = color * group_size
        return _FakeComm(group_size, self._rank - first, num_batches=self._num_batches)


def _divide(size, num_batches):
    """Run ``_split_for_batches`` for every rank of a ``size``-process communicator.

    Returns a list of ``(divided, group_id, is_leader)``, one entry per rank, with the
    group membership recovered from the colors passed to ``Split`` rather than from a
    real communicator.
    """
    from mpi4py import MPI

    from sbd.sbd_solver import _split_for_batches

    table = []
    for rank in range(size):
        comm = _FakeComm(size, rank, num_batches=num_batches)
        divided, _, group_id, _ = _split_for_batches(comm, num_batches)
        if not divided:
            table.append((False, group_id, None))
            continue
        group_color, leader_color = comm.colors
        in_group = group_color != MPI.UNDEFINED
        table.append((True, group_id, in_group and leader_color != MPI.UNDEFINED))
    return table


@pytest.mark.parametrize("num_batches", [1, 2, 5])
def test_single_process_is_never_divided(num_batches):
    """One process runs the batches in turn, whatever their number."""
    (entry,) = _divide(1, num_batches)
    assert entry == (False, 0, None)


@pytest.mark.parametrize("size", [1, 2, 8])
def test_one_batch_is_never_divided(size):
    """A lone subspace is given every process, which is the merged round of trim SQD."""
    assert all(divided is False for divided, _, _ in _divide(size, 1))


@pytest.mark.parametrize(("size", "num_batches"), [(2, 3), (3, 4), (1, 2)])
def test_fewer_processes_than_batches_is_not_divided(size, num_batches):
    """With too few processes to go around, the batches run in turn over all of them."""
    assert all(divided is False for divided, _, _ in _divide(size, num_batches))


@pytest.mark.parametrize(
    ("size", "num_batches", "expected_group_sizes"),
    [
        (4, 2, [2, 2]),
        (6, 3, [2, 2, 2]),
        (8, 2, [4, 4]),
        (3, 3, [1, 1, 1]),
        (9, 3, [3, 3, 3]),
    ],
)
def test_equal_division_assigns_every_process(size, num_batches, expected_group_sizes):
    """When the division is exact, every process joins a group and the groups match."""
    table = _divide(size, num_batches)
    assert all(divided for divided, _, _ in table)
    assert [group for _, group, _ in table].count(None) == 0
    for batch, expected in enumerate(expected_group_sizes):
        assert sum(1 for _, group, _ in table if group == batch) == expected


@pytest.mark.parametrize(
    ("size", "num_batches", "expected_idle"),
    [
        (10, 3, 1),
        (5, 2, 1),
        (7, 3, 1),
        (11, 3, 2),
        (7, 2, 1),
    ],
)
def test_leftover_processes_are_left_idle(size, num_batches, expected_idle):
    """A remainder is left out rather than making one group larger than its siblings.

    Equal groups are what makes the batches comparable, which is the point of a
    screening round: a batch solved over more processes reaches a slightly different
    floating-point energy, so the ranking would depend on how the processes divided.
    """
    table = _divide(size, num_batches)
    assert all(divided for divided, _, _ in table)

    idle = [rank for rank, (_, group, _) in enumerate(table) if group is None]
    assert len(idle) == expected_idle
    # The idle processes are the ones at the end, so a group is always contiguous.
    assert idle == list(range(size - expected_idle, size))

    group_sizes = [
        sum(1 for _, group, _ in table if group == batch) for batch in range(num_batches)
    ]
    assert group_sizes == [size // num_batches] * num_batches


@pytest.mark.parametrize(("size", "num_batches"), [(4, 2), (6, 3), (10, 3), (9, 3)])
def test_every_group_has_exactly_one_leader(size, num_batches):
    """Each group contributes one process to the leaders' exchange, and rank 0 is one.

    The results are broadcast from rank 0 of the whole communicator, so it has to be a
    leader; it is, being the first process of the first group.
    """
    table = _divide(size, num_batches)
    leaders = [rank for rank, (_, _, is_leader) in enumerate(table) if is_leader]
    assert len(leaders) == num_batches
    assert 0 in leaders
    # One leader per group, and each leads a different one.
    assert sorted(table[rank][1] for rank in leaders) == list(range(num_batches))


# --- the grouped path against the sequential one -------------------------------------
#
# The division is only correct if it does not change the answer, so the tests below
# diagonalize the same several subspaces twice over: once letting the processes divide,
# and once one subspace at a time over every process, which is the path that predates
# the division. The energies have to agree.
#
# They agree to a tolerance rather than exactly. A subspace solved over a group of two
# processes and the same subspace solved over all four sum their contributions in a
# different order, so the Davidson iterations differ in the last bits. SOLVER_CONFIG
# converges to 1e-10, well inside the 1e-8 asserted here.
#
# These tests pass SOLVER_CONFIG rather than going through ``_sbd_config``, which sizes
# the alpha dimension from ``MPI.COMM_WORLD``. That is right for an undivided call, where
# the world is the set of processes performing the diagonalization, and wrong here, where
# a group is: SBD's grid describes one diagonalization, so it is relative to whichever
# communicator that diagonalization is handed.
#
# The rule itself is just divisibility. ``diag()`` derives the helper dimension by
# integer division, ``h_comm_size = mpi_size / (task_comm_size * base_comm_size)``, and
# ``TaskCommunicator`` then checks that multiplying the four back recovers ``mpi_size``
# (chemistry/tpb/sbdiag.h and chemistry/tpb/helper.h) -- which fails exactly when the
# division truncated. So ``adet * bdet * task`` has to divide the communicator evenly.
# ``adet_comm_size=2`` would be fine on groups of two; it is 6, taken from a world the
# groups are no longer the same size as, that is not.
#
# Leaving the grid at its 1x1x1 default avoids having to know: the helper dimension
# absorbs however many processes the group turns out to have. A caller cannot size the
# grid to the group anyway, not knowing how the processes were divided, and one config
# has to serve both of trim SQD's rounds, whose communicators differ in size by
# construction.


def _subspaces_for_batches(data_dir, num_batches):
    """``num_batches`` distinct subspaces drawn from the h2o selection.

    Each takes a different slice of the alpha determinants, so the subspaces differ and
    a result mistakenly carried from the wrong group would show up as a wrong energy.
    Every slice starts at the Hartree-Fock determinant, the first in the file, so each
    subspace is a sensible one to diagonalize rather than an arbitrary set.
    """
    strings = _read_alpha_determinants(data_dir / "h2o" / "h2o-1em3-alpha.txt")
    subspaces = []
    for batch in range(num_batches):
        taken = np.concatenate([strings[:1], strings[1 + batch : 24 + batch]])
        subspaces.append((taken, taken))
    return subspaces


def _check_grouped_matches_sequential(data_dir, device_config, num_batches):
    from mpi4py import MPI

    from sbd.sbd_solver import solve_sci_batch

    comm = MPI.COMM_WORLD
    hcore, eri, _ = _load_hamiltonian(data_dir)
    subspaces = _subspaces_for_batches(data_dir, num_batches)

    def diagonalize(batch):
        return solve_sci_batch(
            batch,
            hcore,
            eri,
            norb=NORB,
            nelec=NELEC,
            sbd_config=SOLVER_CONFIG,
            device_config=device_config,
        )

    # All the subspaces in one call: divided into groups when there are enough
    # processes, and run in turn when there are not.
    grouped = diagonalize(subspaces)
    assert len(grouped) == num_batches

    # One subspace per call, so every call is given the whole communicator. This is
    # what the division has to reproduce.
    sequential = [diagonalize([subspace])[0] for subspace in subspaces]

    if comm.Get_rank() != 0:
        return

    # Ordered by subspace, not by whichever group finished first. The subspaces differ,
    # so a misordered or misattributed result fails here.
    for from_group, from_sequence in zip(grouped, sequential):
        assert from_group.energy == pytest.approx(from_sequence.energy, abs=1e-8)
        _assert_result_is_consistent(from_group)

    # The subspaces are distinct, so their energies should be too -- without this, the
    # comparison above would also pass if every group had solved the same subspace.
    energies = [result.energy for result in grouped]
    assert len(set(energies)) == num_batches


@pytest.mark.parametrize("num_batches", [2, 3])
def test_grouped_matches_sequential_standalone(data_dir, device_config, num_batches):
    """In one process nothing is divided, and the two paths are the same code."""
    _check_grouped_matches_sequential(data_dir, device_config, num_batches)


@pytest.mark.mpi
@pytest.mark.parametrize("num_batches", [2, 3])
def test_grouped_matches_sequential_mpi(data_dir, device_config, num_batches):
    """Dividing the launched ranks among the subspaces gives the same energies.

    Whether the division actually happens depends on the process count the suite was
    launched with: at or above ``num_batches`` processes it does, below that the call
    falls back to running them in turn. Both are worth exercising, and which one runs
    is reported by ``tox -e mpi``'s header rather than asserted here.
    """
    _check_grouped_matches_sequential(data_dir, device_config, num_batches)


def _check_two_phase_rounds(data_dir, device_config):
    """The trim SQD shape: a divided screening round, then an undivided merged one.

    Both rounds happen in one process lifetime, as they do inside
    ``diagonalize_fermionic_hamiltonian``. That is what makes this more than the sum of
    the two cases above: the screening round creates communicators and per-group
    wavefunction dumps, and the merged round that follows must not inherit either. A
    communicator left unfreed would eventually exhaust the supply over many iterations,
    and a dump left behind under a name the merged round reuses would be read back as
    if it were the merged round's own amplitudes.
    """
    from mpi4py import MPI

    from sbd.sbd_solver import solve_sci_batch

    comm = MPI.COMM_WORLD
    hcore, eri, _ = _load_hamiltonian(data_dir)

    def diagonalize(batch):
        return solve_sci_batch(
            batch,
            hcore,
            eri,
            norb=NORB,
            nelec=NELEC,
            sbd_config=SOLVER_CONFIG,
            device_config=device_config,
        )

    # Several iterations, so that a communicator leaked once per round would accumulate
    # rather than merely occur.
    for _ in range(3):
        screened = diagonalize(_subspaces_for_batches(data_dir, 3))
        assert len(screened) == 3

        # The merged round: one subspace built from the screening round, given every
        # process. Merging the inputs keeps this independent of what the solver chose
        # to carry over, which is a separate concern tested above.
        screened_alpha = [a for a, _ in _subspaces_for_batches(data_dir, 3)]
        merged_a = np.unique(np.concatenate(screened_alpha))
        (merged,) = diagonalize([(merged_a, merged_a)])

        if comm.Get_rank() != 0:
            continue

        _assert_result_is_consistent(merged)
        # The merged subspace contains each screened subspace, so its energy is at or
        # below every one of theirs. This would fail if the merged round had read back
        # a screening round's wavefunction dump instead of its own.
        for result in screened:
            assert merged.energy <= result.energy + 1e-8


def test_two_phase_rounds_standalone(data_dir, device_config):
    """The two-round shape runs in a single process."""
    _check_two_phase_rounds(data_dir, device_config)


@pytest.mark.mpi
def test_two_phase_rounds_mpi(data_dir, device_config):
    """The screening round divides the launched ranks; the merged round gets them all."""
    _check_two_phase_rounds(data_dir, device_config)
