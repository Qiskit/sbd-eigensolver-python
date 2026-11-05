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

"""Unrestricted (UHF) states: the solver and ``sbd.analysis.variance`` on them.

Two cases:

- Upstream SBD's own UHF fixture for its ``gdb-stat-evaluator-uhf``
  (``apps/gdb-stat-evaluator/tests/make_cli_fixture.cc``). It never reaches a beta
  index, so it is also a valid restricted file; the test pins how it is classified.
- A UHF state whose alpha and beta orbitals differ: stretched H4, started from a guess
  that puts the alpha and beta electrons on alternate atoms. Its full space through
  the unrestricted solver must give PySCF's FCI energy, which no orbital basis changes,
  and over a subspace its variance and PT2 are checked against the exact values from
  PySCF's ``fci.direct_uhf`` applying the same integrals, so the reference does not
  come from SBD. Every case is a **subprocess**, since the determinant word count is fixed
for the life of a process.
"""

from __future__ import annotations

import itertools
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import sbd

pytest.importorskip("pyscf", reason="pyscf builds the references")

from pyscf import ao2mo, fci, gto, scf  # noqa: E402
from pyscf.fci import cistring  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples" / "tools"))
from make_uhf_fcidump import write_uhf_fcidump  # noqa: E402

BITS = 20

ANALYSE = r"""
import json, sys
import numpy as np
import sbd
from sbd import analysis
a = json.loads(sys.argv[1])
sbd.init(device="cpu")
f = sbd.LoadFCIDump(a["fcidump"])
out = {"uhf": sbd.fcidump_format(f) == "uhf"}
if a.get("solve"):
    config = sbd.GDB_SBD(); config.eps, config.max_it, config.bit_length = 1e-12, 300, a["bits"]
    det = sbd.sort_bitarray_array(sbd.from_strings(a["dets"], a["bits"], 2 * a["norb"]))
    out["energy"] = sbd.gdb_diag(f, det, config, savename=a["state"])["energy"]
energy = a.get("energy", out.get("energy"))
runs = [analysis.variance(a["state"], f, energy, method="stochastic", bit_length=a["bits"],
                          samples=4000, batches=10, seed=100 + s) for s in range(8)]
out["variance"] = [r.variance for r in runs]
out["pt2"] = [r.pt2 for r in runs]
try:
    analysis.variance(a["state"], f, energy, method="exact", bit_length=a["bits"])
except ValueError as e:
    out["exact_refused"] = str(e)
# auto, the default, takes the stochastic method for a UHF file even when the space is
# small enough for exact, rather than refusing.
out["auto_method"] = analysis.variance(a["state"], f, energy, bit_length=a["bits"],
                                       samples=200, batches=2).method
print("RESULT " + json.dumps(out))
"""


def _subprocess(args):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    done = subprocess.run([sys.executable, "-c", ANALYSE, json.dumps(args)],
                          capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    return json.loads([l for l in done.stdout.splitlines()
                       if l.startswith("RESULT ")][-1][len("RESULT "):])


def _interleave(alpha, beta, norb):
    """A GDB determinant string: bit 2i alpha orbital i, bit 2i+1 beta, rightmost first."""
    bits = ["0"] * (2 * norb)
    for i in range(norb):
        if alpha >> i & 1:
            bits[2 * norb - 1 - 2 * i] = "1"
        if beta >> i & 1:
            bits[2 * norb - 1 - (2 * i + 1)] = "1"
    return "".join(bits)


def _exact_external(h1, eri, norb, nelec, kept, coef_of):
    """Upstream's external variance and PT2 of a state on ``kept``, by direct_uhf.

    ``kept`` is a list of (alpha, beta) string pairs and ``coef_of`` their amplitudes in
    PySCF's sign convention. Returns (energy of the state, variance, PT2).
    """
    sa = cistring.make_strings(range(norb), nelec[0])
    sb = cistring.make_strings(range(norb), nelec[1])
    ia = {int(s): i for i, s in enumerate(sa)}
    ib = {int(s): i for i, s in enumerate(sb)}
    ci = np.zeros((len(sa), len(sb)))
    for (a, b), c in zip(kept, coef_of):
        ci[ia[a], ib[b]] = c
    ci /= np.linalg.norm(ci)
    h2e = fci.direct_uhf.absorb_h1e(h1, eri, norb, nelec, 0.5)
    hc = fci.direct_uhf.contract_2e(h2e, ci, norb, nelec)
    energy = float(np.vdot(ci, hc))
    hdiag = fci.direct_uhf.make_hdiag(h1, eri, norb, nelec).reshape(ci.shape)
    outside = np.ones(ci.shape, dtype=bool)
    for a, b in kept:
        outside[ia[a], ib[b]] = False
    coupling, diagonal = hc[outside], hdiag[outside]
    return energy, float(coupling @ coupling), float(np.sum(coupling ** 2 / (energy - diagonal)))


def _assert_matches(out, variance, pt2):
    for name, exact in (("variance", variance), ("pt2", pt2)):
        values = np.array(out[name])
        error = values.std(ddof=1) / np.sqrt(len(values))
        assert abs(values.mean() - exact) < 5 * error + 1e-12, (name, values.mean(), exact)


@pytest.fixture
def uhf_available(backend):
    if not sbd.has_uhf("cpu"):
        pytest.skip("no unrestricted CPU build (SBD_BUILD_UHF=0?)")


def test_upstream_uhf_fixture_is_ambiguous_and_read_as_restricted(backend, tmp_path):
    """Upstream's UHF evaluator fixture, as this package classifies it.

    Upstream's ``make_cli_fixture.cc`` writes, under ``-D_UHF``, the integrals
    ``0.2 3 1 0 0``, ``0.1 1 1 1 1`` and ``0.08 3 1 3 1`` with NORB = 4. Every index is
    within NORB, so the file is also a valid restricted FCIDUMP: its contents cannot
    say which layout is meant, and the two read it as different Hamiltonians (alpha-only
    integrals in one, spin-free ones in the other). SBD's layout check marks a file
    unrestricted only when an index exceeds NORB, so it reads this one as restricted,
    and stating ``uhf=True`` is refused rather than overriding that. A file meant as
    unrestricted therefore has to reach a beta index; the stretched H4 case below
    does, and is the one that checks unrestricted results.
    """
    fcidump = tmp_path / "FCIDUMP"
    fcidump.write_text(" &FCI NORB=4,NELEC=4,MS2=0,\n  ORBSYM=1,1,1,1,\n  ISYM=1,\n &END\n"
                       " 0.2 3 1 0 0\n 0.1 1 1 1 1\n 0.08 3 1 3 1\n 0.0 0 0 0 0\n")
    assert sbd.fcidump_format(str(fcidump)) == "restricted"
    with pytest.raises(ValueError, match="uhf=True"):
        sbd.LoadFCIDump(str(fcidump), uhf=True)


def _stretched_h4():
    """Stretched H4 in UHF orbitals whose alpha and beta orbitals differ: the molecule,
    its one- and two-electron integrals in those orbitals, and the norb and nelec."""
    mol = gto.M(atom="; ".join(f"H 0 0 {2.0 * i:.1f}" for i in range(4)), basis="sto-3g",
                spin=0, verbose=0)
    mf = scf.UHF(mol)
    dm_a, dm_b = mf.get_init_guess()
    # Start with alpha on atoms 0 and 2, beta on 1 and 3, so that UHF does not converge
    # to the restricted solution, where the alpha and beta orbitals would be the same.
    for atom, (da, db) in enumerate(((1, 0), (0, 1), (1, 0), (0, 1))):
        dm_a[atom, atom], dm_b[atom, atom] = da, db
    mf.kernel(dm0=(dm_a, dm_b))
    assert abs(mf.spin_square()[0]) > 0.5, "the UHF alpha and beta orbitals must differ"
    ca, cb = mf.mo_coeff
    norb, nelec = mol.nao, mol.nelec
    hcore = scf.hf.get_hcore(mol)
    h1 = (ca.T @ hcore @ ca, cb.T @ hcore @ cb)
    eri = (ao2mo.restore(1, ao2mo.kernel(mol, ca), norb),
           ao2mo.restore(1, ao2mo.general(mol, (ca, ca, cb, cb)), norb),
           ao2mo.restore(1, ao2mo.kernel(mol, cb), norb))
    return mol, h1, eri, norb, nelec


def _subspace_reference(h1, eri, norb, nelec, kept):
    """The subspace ground state built in PySCF: its energy, variance and PT2."""
    sa = cistring.make_strings(range(norb), nelec[0])
    sb = cistring.make_strings(range(norb), nelec[1])
    h2e = fci.direct_uhf.absorb_h1e(h1, eri, norb, nelec, 0.5)
    ia = {int(s): i for i, s in enumerate(sa)}
    ib = {int(s): i for i, s in enumerate(sb)}
    basis = []
    for a, b in kept:
        v = np.zeros((len(sa), len(sb)))
        v[ia[a], ib[b]] = 1.0
        basis.append(v)
    hsub = np.array([[np.vdot(u, fci.direct_uhf.contract_2e(h2e, v, norb, nelec))
                      for v in basis] for u in basis])
    psi = np.linalg.eigh(hsub)[1][:, 0]
    return _exact_external(h1, eri, norb, nelec, kept, psi)


def _full_pairs(norb, nelec):
    sa = cistring.make_strings(range(norb), nelec[0])
    sb = cistring.make_strings(range(norb), nelec[1])
    return [(int(a), int(b)) for a in sa for b in sb]


def test_uhf_state_with_distinct_alpha_beta_orbitals(uhf_available, tmp_path):
    """Stretched H4 in UHF orbitals whose alpha and beta orbitals differ."""
    mol, h1, eri, norb, nelec = _stretched_h4()
    fcidump = tmp_path / "h4_bs_uhf.fcidump"
    write_uhf_fcidump(str(fcidump), h1, eri, norb, sum(nelec), nelec[0] - nelec[1], 0.0)
    full = _full_pairs(norb, nelec)

    # The full space: the exact energy, which the orbital basis cannot change, so a
    # restricted FCI of the same molecule is the reference.
    e_fci = fci.FCI(scf.RHF(mol).run(conv_tol=1e-12)).kernel()[0] - mol.energy_nuc()
    out = _subprocess({"fcidump": str(fcidump), "state": str(tmp_path / "full-"),
                       "bits": BITS, "norb": norb, "solve": True,
                       "dets": [_interleave(a, b, norb) for a, b in full]})
    assert out["uhf"]
    assert out["energy"] == pytest.approx(e_fci, abs=1e-8)
    assert out["auto_method"] == "stochastic"       # the full space would fit exact

    # A subspace: something lies outside it, and the variance is checked exactly.
    kept = full[: len(full) // 2]
    out = _subprocess({"fcidump": str(fcidump), "state": str(tmp_path / "sub-"),
                       "bits": BITS, "norb": norb, "solve": True,
                       "dets": [_interleave(a, b, norb) for a, b in kept]})
    # The state SBD saved is the subspace ground state; build the same state in PySCF.
    energy, variance, pt2 = _subspace_reference(h1, eri, norb, nelec, kept)
    assert out["energy"] == pytest.approx(energy, abs=1e-8)
    assert variance > 1e-4, "the subspace must leave something outside it"
    _assert_matches(out, variance, pt2)


@pytest.mark.mpi
def test_uhf_variance_with_the_state_over_every_rank(uhf_available):
    """The UHF subspace solved and saved over every rank (b_comm_size = ranks), then
    evaluated from those shards on the same ranks: the exact values on each.

    In the job's own ranks rather than a subprocess: one word per determinant, the width
    the rest of the MPI suite uses. Rank 0 builds the integrals, so every rank has the
    same orbitals.
    """
    import shutil
    import tempfile

    from mpi4py import MPI

    from sbd import analysis

    if not sbd.has_uhf(sbd.get_device()):
        pytest.skip(f"no unrestricted build of {sbd.get_device()!r} to solve on")
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    path = pathlib.Path(comm.bcast(tempfile.mkdtemp(prefix="sbd-uhf-var-")
                                   if rank == 0 else None, root=0))
    try:
        reference = None
        if rank == 0:
            _, h1, eri, norb, nelec = _stretched_h4()
            write_uhf_fcidump(str(path / "h4.fcidump"), h1, eri, norb, sum(nelec),
                              nelec[0] - nelec[1], 0.0)
            pairs = _full_pairs(norb, nelec)
            kept = pairs[: len(pairs) // 2]
            reference = (norb, kept, _subspace_reference(h1, eri, norb, nelec, kept))
        norb, kept, (energy, variance, pt2) = comm.bcast(reference, root=0)
        assert variance > 1e-4, "the subspace must leave something outside it"
        fcidump = sbd.LoadFCIDump(str(path / "h4.fcidump"))
        assert sbd.fcidump_format(fcidump) == "uhf"
        full = sbd.sort_bitarray_array(sbd.from_strings(
            [_interleave(a, b, norb) for a, b in kept], BITS, 2 * norb))
        q, r = divmod(len(full), size)
        lo = rank * q + min(rank, r)
        if size > len(full):                 # the same on every rank, so all skip together
            pytest.skip("more ranks than determinants")
        config = sbd.GDB_SBD()
        config.eps, config.max_it, config.max_nb, config.bit_length = 1e-12, 300, 20, BITS
        config.b_comm_size = size
        solved = sbd.gdb_diag(fcidump, full[lo:lo + q + (1 if rank < r else 0)], config,
                              savename=str(path / "sub-"),
                              determinant_distribution="input")["energy"]
        assert solved == pytest.approx(energy, abs=1e-8)
        runs = [analysis.variance(str(path / "sub-"), fcidump, solved, method="stochastic",
                                  bit_length=BITS, samples=4000, batches=10, seed=100 + s)
                for s in range(8)]
        _assert_matches({"variance": [x.variance for x in runs],
                         "pt2": [x.pt2 for x in runs]}, variance, pt2)
    finally:
        comm.Barrier()
        if rank == 0:
            shutil.rmtree(path, ignore_errors=True)
