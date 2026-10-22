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

"""``SCIResult.sci_state`` from ``solve_sci`` must be the state PySCF would return.

``SCIState`` hands its amplitudes to PySCF for ``rdm()`` and ``spin_square()``, so
they have to follow PySCF's sign convention, in which every alpha creation operator
precedes every beta one. SBD orders them orbital by orbital instead, which gives
some determinants the opposite sign. Energies, SBD's own RDMs (``SCIResult.rdm1`` /
``rdm2``), occupancies and selection by amplitude magnitude never see the
difference, which is why only these comparisons catch it.

Each case solves one subspace twice -- with SBD through ``solve_sci``, and with
qiskit-addon-sqd's PySCF solver -- and compares the states, not just the energies.
The subspaces are full spaces and seeded random truncations of them: a small
truncation can happen to need no sign change at all, so a lone small case can pass
by accident.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf", reason="pyscf builds the reference Hamiltonians")
fermion = pytest.importorskip("qiskit_addon_sqd.fermion",
                              reason="qiskit-addon-sqd provides SCIState and the reference solver")

from pyscf import ao2mo, gto, scf  # noqa: E402

from sbd.sbd_solver import solve_sci  # noqa: E402

SYSTEMS = {
    # (atom, basis, 2S): a doublet, a singlet, and one with more orbitals
    "h3-doublet": ("H 0 0 0; H 0 0 0.8; H 0 0 2.2", "sto-3g", 1),
    "h4-singlet": ("H 0 0 0; H 0 0 0.9; H 0 0 1.9; H 0 0 3.0", "sto-3g", 0),
    "h6-singlet": ("; ".join(f"H 0 0 {0.95 * i:.2f}" for i in range(6)), "sto-3g", 0),
}


def _hamiltonian(name):
    atom, basis, spin = SYSTEMS[name]
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = (scf.ROHF if spin else scf.RHF)(mol)
    mf.conv_tol = 1e-12
    mf.run()
    c = mf.mo_coeff
    norb = mol.nao
    h1 = c.T @ scf.hf.get_hcore(mol) @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), norb)
    return h1, eri, norb, mol.nelec


def _strings(norb, n_electrons, keep=None, seed=0):
    full = np.array(sorted(sum(1 << o for o in occ)
                           for occ in itertools.combinations(range(norb), n_electrons)))
    if keep is None or keep >= len(full):
        return full
    chosen = np.random.default_rng(seed).choice(len(full), size=keep, replace=False)
    return np.sort(full[np.union1d(chosen, [0])])        # keep Hartree-Fock


CASES = [
    ("h3-doublet", None),
    ("h4-singlet", None),
    ("h6-singlet", None),
    ("h6-singlet", 12),
]


def _energy_from_rdms(state, h1, eri):
    return (np.einsum("pr,pr->", state.rdm(rank=1, spin_summed=True), h1)
            + 0.5 * np.einsum("prqs,prqs->", state.rdm(rank=2, spin_summed=True), eri))


def _check_sci_state_matches_pyscf(device_config, name, keep):
    h1, eri, norb, nelec = _hamiltonian(name)
    strings = (_strings(norb, nelec[0], keep, seed=1), _strings(norb, nelec[1], keep, seed=2))

    # PySCF's default tolerance leaves the vector good to ~1e-6; converge both tightly
    # so that the elementwise comparison measures the convention, not convergence.
    expected = fermion.solve_sci(strings, h1, eri, norb=norb, nelec=nelec, tol=1e-14)
    got = solve_sci(strings, h1, eri, norb=norb, nelec=nelec, device_config=device_config,
                    sbd_config={"eps": 1e-12, "max_it": 300})
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_rank() != 0:
        return                        # the amplitudes are gathered onto rank 0 only

    assert got.energy == pytest.approx(expected.energy, abs=1e-8)
    assert np.array_equal(got.sci_state.ci_strs_a, expected.sci_state.ci_strs_a)
    assert np.array_equal(got.sci_state.ci_strs_b, expected.sci_state.ci_strs_b)

    # The same vector up to one global sign -- the comparison that catches a
    # per-determinant sign, where every magnitude-based one passes.
    a, b = got.sci_state.amplitudes, expected.sci_state.amplitudes
    overlap = np.vdot(b, a)
    assert abs(overlap) == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(a * np.sign(overlap), b, atol=1e-6)

    # And what users compute from it.
    assert got.sci_state.spin_square() == pytest.approx(expected.sci_state.spin_square(), abs=1e-7)
    assert _energy_from_rdms(got.sci_state, h1, eri) == pytest.approx(expected.energy, abs=1e-7)


# One body, two wrappers: pytest-mpi skips unmarked tests under --only-mpi and marked
# ones without it (see test_sqd_integration.py).
@pytest.mark.parametrize("name, keep", CASES, ids=[f"{n}-{k or 'full'}" for n, k in CASES])
def test_sci_state_matches_pyscf_standalone(device_config, name, keep):
    _check_sci_state_matches_pyscf(device_config, name, keep)


@pytest.mark.mpi
@pytest.mark.parametrize("name, keep", CASES, ids=[f"{n}-{k or 'full'}" for n, k in CASES])
def test_sci_state_matches_pyscf_mpi(device_config, name, keep):
    _check_sci_state_matches_pyscf(device_config, name, keep)


def test_savename_writes_the_returned_state(device_config, tmp_path):
    """``savename`` saves the very state ``solve_sci`` returns, readable by the analysis.

    Each batch of ``solve_sci_batch`` gets its own prefix, so one batch never
    overwrites another's files.
    """
    import sbd
    from sbd import analysis
    from sbd.sbd_solver import solve_sci_batch
    from pyscf import tools

    h1, eri, norb, nelec = _hamiltonian("h6-singlet")
    fcidump = tmp_path / "h6.fcidump"
    tools.fcidump.from_integrals(str(fcidump), h1, eri, norb, sum(nelec), nuc=0.0)
    batches = [(_strings(norb, nelec[0], 12, seed=s), _strings(norb, nelec[1], 12, seed=s + 1))
               for s in (1, 5)]
    prefix = str(tmp_path / "st-")
    results = solve_sci_batch(batches, h1, eri, norb=norb, nelec=nelec,
                              device_config=device_config,
                              sbd_config={"eps": 1e-12, "max_it": 300}, savename=prefix)

    loaded = sbd.LoadFCIDump(str(fcidump))
    energies = [analysis.state_energy(f"{prefix}b{k:02d}-", loaded, bit_length=20)
                for k in range(len(batches))]
    for result, energy in zip(results, energies):
        assert energy == pytest.approx(result.energy, abs=1e-8)
    assert energies[0] != pytest.approx(energies[1], abs=1e-6), "batches overwrote each other"
