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

    state = result.sci_state
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


def _check_diagonalize_fermionic_hamiltonian(data_dir, device_config, counts_path):
    from functools import partial

    from mpi4py import MPI

    pytest.importorskip(
        "qiskit_addon_sqd",
        reason="qiskit-addon-sqd is needed for the self-consistent loop",
    )
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    from sbd.sbd_solver import solve_sci_batch

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
        sbd_config=_sbd_config(comm, eps=1e-8, max_it=10, max_nb=10),
        device_config=device_config,
    )

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
    )

    if comm.Get_rank() != 0:
        return

    _assert_result_is_consistent(result)

    # A bracket rather than an equality: the subspace comes out of upstream's sampling.
    # The lower bound is the FCI energy, which no subspace of it can beat; the upper is
    # loose enough to tolerate a different selection but far below the -75.0 a badly
    # converged run gives, which is the failure worth catching.
    total_energy = result.energy + nuclear_repulsion
    assert -76.25 < total_energy < -75.9


def test_diagonalize_fermionic_hamiltonian_standalone(
    data_dir, device_config, counts_path
):
    """The self-consistent loop runs to completion in a single process."""
    _check_diagonalize_fermionic_hamiltonian(data_dir, device_config, counts_path)


@pytest.mark.mpi
def test_diagonalize_fermionic_hamiltonian_mpi(data_dir, device_config, counts_path):
    """The self-consistent loop runs to completion across the launched ranks."""
    _check_diagonalize_fermionic_hamiltonian(data_dir, device_config, counts_path)
