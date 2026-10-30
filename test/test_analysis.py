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

"""``sbd.analysis``: energy variance and EN-PT2 of states saved by ``savename``.

Each case diagonalizes a truncated subspace of a hydrogen chain, saves it with TPB or
GDB, and analyses the files. The exact method is checked against an independent
calculation done here with PySCF on a state solved by PySCF, so a sign or layout error
in reading SBD's files cannot cancel out. The stochastic method is checked against the
exact one: an unbiased estimator averaged over seeds must land on it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf", reason="pyscf builds the Hamiltonians and the references")

from pyscf import ao2mo, gto, scf  # noqa: E402
from pyscf.fci import cistring, direct_spin1  # noqa: E402
from pyscf.tools import fcidump as fd  # noqa: E402

import sbd  # noqa: E402
from sbd import analysis  # noqa: E402

BITS = 20


def _chain(n, spacing=0.95):
    return "; ".join(f"H 0 0 {spacing * i:.2f}" for i in range(n))


@pytest.fixture(scope="module")
def h8():
    """H8/STO-3G: 8 orbitals, 4 + 4 electrons, 4,900 determinants in the full space.

    Rank 0 runs the SCF and writes the one FCIDUMP every rank reads, and its integrals
    are broadcast for the references. Each rank solving its own SCF is not equivalent:
    the eigensolver fixes each orbital's sign arbitrarily, differently on different
    ranks under threading, and ranks holding differently-signed integrals do not share
    one Hamiltonian -- GDB then fails to converge and a saved state reads differently
    on each rank.

    The FCIDUMP goes to a directory of its own rather than tmp_path_factory: under MPI
    each rank runs its own pytest, and one rank's pruning of old numbered temp
    directories can delete another's while it is in use.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    system = None
    if comm.Get_rank() == 0:
        mol = gto.M(atom=_chain(8), basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        workdir = tempfile.mkdtemp(prefix="sbd-analysis-h8-")
        path = Path(workdir) / "fcidump.txt"
        fd.from_scf(mf, str(path), tol=1e-15)
        c = mf.mo_coeff
        system = {"workdir": workdir, "path": path, "norb": mol.nao, "nelec": mol.nelec,
                  "h1": c.T @ scf.hf.get_hcore(mol) @ c,
                  "eri": ao2mo.restore(1, ao2mo.kernel(mol, c), mol.nao),
                  "ecore": mol.energy_nuc()}
    system = comm.bcast(system, root=0)
    yield system
    comm.Barrier()
    if comm.Get_rank() == 0:
        shutil.rmtree(system["workdir"], ignore_errors=True)


def _strings(norb, n):
    return [format(s, f"0{norb}b") for s in cistring.make_strings(range(norb), n)]


def _interleave(a, b):
    ar, br = a[::-1], b[::-1]
    return "".join(ar[i] + br[i] for i in range(len(ar)))[::-1]


def _subspace(system, keep):
    """The first ``keep`` alpha and beta strings (Hartree-Fock among them)."""
    return (_strings(system["norb"], system["nelec"][0])[:keep],
            _strings(system["norb"], system["nelec"][1])[:keep])


def _save_tpb(system, strings, prefix, comm_size=1):
    f = sbd.LoadFCIDump(str(system["path"]))
    norb = system["norb"]
    config = sbd.TPB_SBD()
    config.eps, config.max_it, config.max_nb, config.bit_length = 1e-12, 300, 20, BITS
    adet = sbd.sort_bitarray([sbd.from_string(s, BITS, norb) for s in strings[0]])
    bdet = sbd.sort_bitarray([sbd.from_string(s, BITS, norb) for s in strings[1]])
    return sbd.tpb_diag(f, adet, bdet, config, savename=prefix)["energy"]


def _save_gdb(system, strings, prefix):
    f = sbd.LoadFCIDump(str(system["path"]))
    norb = system["norb"]
    rank, size = sbd.get_rank(), sbd.get_world_size()
    full = sbd.sort_bitarray_array(sbd.from_strings(
        [_interleave(a, b) for a, b in itertools.product(*strings)], BITS, 2 * norb))
    q, r = divmod(len(full), size)
    lo = rank * q + min(rank, r)
    config = sbd.GDB_SBD()
    config.eps, config.max_it, config.max_nb, config.bit_length = 1e-12, 300, 20, BITS
    config.b_comm_size = size
    return sbd.gdb_diag(f, full[lo:lo + q + (1 if rank < r else 0)], config,
                        savename=prefix)["energy"]


def _reference(system, strings):
    """Variance, PT2 and energy of the same subspace, solved and evaluated by PySCF alone."""
    norb, nelec = system["norb"], system["nelec"]
    h1, eri = system["h1"], system["eri"]
    ia = cistring.strs2addr(norb, nelec[0], [int(s, 2) for s in strings[0]])
    ib = cistring.strs2addr(norb, nelec[1], [int(s, 2) for s in strings[1]])
    na, nb = cistring.num_strings(norb, nelec[0]), cistring.num_strings(norb, nelec[1])
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)
    # H projected onto the subspace, explicitly: small enough to build column by column.
    dim = len(ia) * len(ib)
    hsub = np.zeros((dim, dim))
    for k, (i, j) in enumerate(itertools.product(range(len(ia)), range(len(ib)))):
        e = np.zeros((na, nb))
        e[ia[i], ib[j]] = 1.0
        hsub[:, k] = direct_spin1.contract_2e(h2e, e, norb, nelec)[np.ix_(ia, ib)].ravel()
    w, v = np.linalg.eigh(hsub)
    ci = np.zeros((na, nb))
    ci[np.ix_(ia, ib)] = v[:, 0].reshape(len(ia), len(ib))
    hc = direct_spin1.contract_2e(h2e, ci, norb, nelec)
    hdiag = direct_spin1.make_hdiag(h1, eri, norb, nelec).reshape(na, nb)
    out = np.ones((na, nb), dtype=bool)
    out[np.ix_(ia, ib)] = False
    variance = float(np.sum(hc[out] ** 2))
    pt2 = float(np.sum(hc[out] ** 2 / (w[0] - hdiag[out])))
    return variance, pt2, w[0] + system["ecore"]


# --------------------------------------------------------------------------------
# exact method
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["tpb", "gdb"])
def test_exact_matches_an_independent_pyscf_calculation(backend, h8, tmp_path, kind):
    strings = _subspace(h8, 20)
    prefix = str(tmp_path / f"{kind}-")
    energy = (_save_tpb if kind == "tpb" else _save_gdb)(h8, strings, prefix)
    variance, pt2, e_ref = _reference(h8, strings)
    assert energy == pytest.approx(e_ref, abs=1e-9)
    f = sbd.LoadFCIDump(str(h8["path"]))
    result = analysis.variance(prefix, f, method="exact", bit_length=BITS)
    assert result.method == "exact"
    assert result.dimension == len(strings[0]) * len(strings[1])
    assert result.energy == pytest.approx(e_ref, abs=1e-9)
    assert result.variance == pytest.approx(variance, rel=1e-8)
    assert result.pt2 == pytest.approx(pt2, rel=1e-8)
    assert result.energy_pt2 == pytest.approx(e_ref + pt2, abs=1e-9)


def test_the_full_space_has_nothing_outside(backend, tmp_path):
    mol = gto.M(atom=_chain(6), basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    path = tmp_path / "fcidump.txt"     # single process (not an mpi test): no sharing
    fd.from_scf(mf, str(path), tol=1e-15)
    system = {"path": path, "norb": mol.nao, "nelec": mol.nelec}
    full = (_strings(mol.nao, mol.nelec[0]), _strings(mol.nao, mol.nelec[1]))
    _save_tpb(system, full, str(tmp_path / "full-"))
    f = sbd.LoadFCIDump(str(path))
    for method in ("exact", "stochastic"):
        result = analysis.variance(str(tmp_path / "full-"), f, method=method,
                                   bit_length=BITS, samples=200, batches=2)
        assert result.variance == pytest.approx(0.0, abs=1e-14)
        assert result.pt2 == pytest.approx(0.0, abs=1e-14)


def test_tpb_and_gdb_files_of_one_state_give_one_answer(backend, h8, tmp_path):
    strings = _subspace(h8, 15)
    _save_tpb(h8, strings, str(tmp_path / "tpb-"))
    _save_gdb(h8, strings, str(tmp_path / "gdb-"))
    f = sbd.LoadFCIDump(str(h8["path"]))
    a = analysis.variance(str(tmp_path / "tpb-"), f, method="exact", bit_length=BITS)
    b = analysis.variance(str(tmp_path / "gdb-"), f, method="exact", bit_length=BITS)
    assert a.variance == pytest.approx(b.variance, rel=1e-10)
    assert a.pt2 == pytest.approx(b.pt2, rel=1e-10)
    # and a single checkpoint file reads like its prefix
    c = analysis.variance(str(tmp_path / "gdb-000000.bin"), f, method="exact", bit_length=BITS)
    assert c.variance == pytest.approx(b.variance, rel=1e-12)


def test_state_energy_is_the_saved_states_own(backend, h8, tmp_path):
    strings = _subspace(h8, 20)
    energy = _save_gdb(h8, strings, str(tmp_path / "gdb-"))
    f = sbd.LoadFCIDump(str(h8["path"]))
    assert analysis.state_energy(str(tmp_path / "gdb-"), f, bit_length=BITS) == pytest.approx(
        energy, abs=1e-10)


# --------------------------------------------------------------------------------
# stochastic method
# --------------------------------------------------------------------------------


def _stochastic_matches_exact(h8, prefix, energy, comm=None):
    f = sbd.LoadFCIDump(str(h8["path"]))
    exact = analysis.variance(prefix, f, method="exact", bit_length=BITS, comm=comm)
    # Average independent seeds: the mean's error is small enough to expose a bias of a
    # fraction of a single run's error bar, and 5 sigma keeps a correct run from failing.
    runs = [analysis.variance(prefix, f, energy, method="stochastic", bit_length=BITS,
                              samples=4000, batches=10, seed=100 + s, comm=comm)
            for s in range(8)]
    for name in ("variance", "pt2"):
        values = np.array([getattr(r, name) for r in runs])
        error = values.std(ddof=1) / np.sqrt(len(values))
        assert abs(values.mean() - getattr(exact, name)) < 5 * error + 1e-14, name
    one = runs[0]
    assert one.method == "stochastic"
    assert one.variance_error > 0 and one.pt2_error > 0
    assert len(one.batches["variance"]) == 10


@pytest.mark.parametrize("kind", ["tpb", "gdb"])
def test_stochastic_is_unbiased_against_exact(backend, h8, tmp_path, kind):
    strings = _subspace(h8, 20)
    prefix = str(tmp_path / f"{kind}-")
    energy = (_save_tpb if kind == "tpb" else _save_gdb)(h8, strings, prefix)
    _stochastic_matches_exact(h8, prefix, energy)


@pytest.mark.mpi
def test_stochastic_is_unbiased_under_mpi(backend, h8):
    """GDB writes one file per rank; the files are then dealt out across the ranks.

    The directory is made by rank 0 and shared, not taken from tmp_path: every rank runs
    its own pytest, each keeping only the newest few numbered temp directories, so one
    rank's cleanup can delete another rank's directory while the test still uses it.
    """
    import shutil
    import tempfile

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    path = comm.bcast(tempfile.mkdtemp(prefix="sbd-analysis-") if comm.Get_rank() == 0 else None)
    try:
        strings = _subspace(h8, 20)
        energy = _save_gdb(h8, strings, f"{path}/gdb-")
        comm.Barrier()
        _stochastic_matches_exact(h8, f"{path}/gdb-", energy)
    finally:
        comm.Barrier()
        if comm.Get_rank() == 0:
            shutil.rmtree(path, ignore_errors=True)


def test_a_reproducible_seed(backend, h8, tmp_path):
    strings = _subspace(h8, 15)
    energy = _save_gdb(h8, strings, str(tmp_path / "gdb-"))
    f = sbd.LoadFCIDump(str(h8["path"]))
    run = lambda seed: analysis.variance(str(tmp_path / "gdb-"), f, energy,
                                         method="stochastic", bit_length=BITS,
                                         samples=500, batches=3, seed=seed)
    # Same draws, but not bit-identical: OpenMP's reduction order can change the last
    # bits (upstream's README says so), so compare at rounding level.
    first, again, other = (np.array(run(s).batches["variance"]) for s in (7, 7, 8))
    np.testing.assert_allclose(again, first, rtol=1e-12)
    assert np.max(np.abs(other - first) / np.abs(first)) > 1e-6


def test_first_batch_continues_the_random_streams(backend, h8, tmp_path):
    """Batch IDs reach SBD: the same ID repeats a batch, the next ones are new draws.

    So a second call with ``first_batch`` past the first call's batches adds
    independent batches to pool with them, rather than repeating them.
    """
    strings = _subspace(h8, 15)
    energy = _save_gdb(h8, strings, str(tmp_path / "gdb-"))
    f = sbd.LoadFCIDump(str(h8["path"]))
    run = lambda first, n: analysis.variance(str(tmp_path / "gdb-"), f, energy,
                                             method="stochastic", bit_length=BITS,
                                             samples=500, batches=n, seed=7,
                                             first_batch=first)
    four, later = run(0, 4), run(2, 2)
    assert four.batches["batch_id"] == [0, 1, 2, 3] and later.batches["batch_id"] == [2, 3]
    np.testing.assert_allclose(later.batches["variance"], four.batches["variance"][2:],
                               rtol=1e-12)
    assert len(set(np.round(four.batches["variance"], 12))) == 4, "batches must differ"


def test_observable_and_denominator_reach_the_evaluator(backend, h8, tmp_path):
    strings = _subspace(h8, 15)
    energy = _save_gdb(h8, strings, str(tmp_path / "gdb-"))
    f = sbd.LoadFCIDump(str(h8["path"]))
    run = lambda **kw: analysis.variance(str(tmp_path / "gdb-"), f, energy,
                                         method="stochastic", bit_length=BITS,
                                         samples=500, batches=2, seed=7, **kw)
    only_variance, only_pt2 = run(observable="variance"), run(observable="pt2")
    assert np.isfinite(only_variance.variance) and np.isnan(only_variance.pt2)
    assert np.isnan(only_pt2.variance) and np.isfinite(only_pt2.pt2)
    with pytest.raises(ValueError, match="observable"):
        run(observable="energy")
    # No denominator is that large, so the evaluator must refuse the PT2.
    with pytest.raises(ValueError, match="denominator"):
        run(minimum_abs_denominator=1e6)


@pytest.mark.mpi
def test_a_state_saved_on_more_ranks_is_analysed_on_fewer(backend, h8):
    """GDB files written by every rank, read back by half of them.

    As upstream's own evaluator test does with 2 saved shards on 4 ranks: the state, not
    the layout it was saved with, decides the result. The draws differ with the rank
    count, so each layout is checked against the exact answer rather than batch by batch.
    """
    import shutil
    import tempfile

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs at least two ranks to analyse on fewer")
    path = comm.bcast(tempfile.mkdtemp(prefix="sbd-analysis-") if comm.Get_rank() == 0 else None)
    try:
        strings = _subspace(h8, 20)
        energy = _save_gdb(h8, strings, f"{path}/gdb-")
        comm.Barrier()
        half = comm.Split(0 if comm.Get_rank() < comm.Get_size() // 2 else MPI.UNDEFINED,
                          comm.Get_rank())
        if half != MPI.COMM_NULL:
            assert len(analysis.load_state(f"{path}/gdb-", h8["norb"], BITS, half).files) \
                == comm.Get_size()
            _stochastic_matches_exact(h8, f"{path}/gdb-", energy, comm=half)
            half.Free()
    finally:
        comm.Barrier()
        if comm.Get_rank() == 0:
            shutil.rmtree(path, ignore_errors=True)


def test_auto_chooses_exact_for_a_small_space(backend, h8, tmp_path):
    strings = _subspace(h8, 15)
    _save_gdb(h8, strings, str(tmp_path / "gdb-"))
    f = sbd.LoadFCIDump(str(h8["path"]))
    assert analysis.variance(str(tmp_path / "gdb-"), f, bit_length=BITS).method == "exact"
    assert analysis.variance(str(tmp_path / "gdb-"), f, bit_length=BITS, exact_limit=10,
                             samples=200, batches=2).method == "stochastic"


# --------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------


def test_bad_inputs_are_refused(backend, h8, tmp_path):
    strings = _subspace(h8, 10)
    _save_gdb(h8, strings, str(tmp_path / "gdb-"))
    f = sbd.LoadFCIDump(str(h8["path"]))
    prefix = str(tmp_path / "gdb-")
    with pytest.raises(FileNotFoundError, match="no saved state"):
        analysis.variance(str(tmp_path / "nope-"), f, bit_length=BITS)
    with pytest.raises(ValueError, match="bit_length is required"):
        analysis.variance(prefix, f)
    with pytest.raises(ValueError, match="word"):
        analysis.variance(prefix, f, bit_length=4)
    with pytest.raises(ValueError, match="method must be"):
        analysis.variance(prefix, f, method="nope", bit_length=BITS)


def test_files_from_more_than_one_run_are_refused(backend, h8, tmp_path):
    """A prefix must hold one run's files; anything else raises rather than combining."""
    import os

    f = sbd.LoadFCIDump(str(h8["path"]))
    # a run's file plus a stale file from another run with no determinant in common:
    # no repeats, so only the norm (2 states, each normalized) gives it away
    _save_gdb(h8, _subspace(h8, 20), str(tmp_path / "a-"))
    _save_gdb(h8, (_strings(8, 4)[40:52], _strings(8, 4)[40:52]), str(tmp_path / "b-"))
    os.rename(tmp_path / "b-000000.bin", tmp_path / "a-000001.bin")
    with pytest.raises(ValueError, match="norm-squared"):
        analysis.load_state(str(tmp_path / "a-"), h8["norb"], BITS)
    # TPB and GDB files under one prefix
    _save_tpb(h8, _subspace(h8, 10), str(tmp_path / "c-"))
    _save_gdb(h8, _subspace(h8, 10), str(tmp_path / "c-"))
    with pytest.raises(ValueError, match="both GDB files"):
        analysis.load_state(str(tmp_path / "c-"), h8["norb"], BITS)
    # a gap in the shard numbers
    _save_gdb(h8, _subspace(h8, 10), str(tmp_path / "d-"))
    os.rename(tmp_path / "d-000000.bin", tmp_path / "d-000002.bin")
    _save_gdb(h8, _subspace(h8, 10), str(tmp_path / "d-"))
    with pytest.raises(ValueError, match="gaps"):
        analysis.load_state(str(tmp_path / "d-"), h8["norb"], BITS)
    # a truncated file
    _save_gdb(h8, _subspace(h8, 10), str(tmp_path / "e-"))
    data = (tmp_path / "e-000000.bin").read_bytes()
    (tmp_path / "e-000000.bin").write_bytes(data[:-8])
    with pytest.raises(ValueError, match="truncated|header implies"):
        analysis.load_state(str(tmp_path / "e-"), h8["norb"], BITS)


def test_a_repeat_across_files_is_refused(backend, h8, tmp_path):
    """Two files sharing a determinant: caught when the state is sorted across ranks."""
    from sbd.analysis import _write_gdb

    a = np.asarray(sbd.from_strings(["0000111100001111", "0000111100110011"], BITS, 16))
    # normalized overall, so only the repeat is wrong
    _write_gdb(str(tmp_path / "r-000000.bin"), a[:2], np.array([0.6, 0.0]))
    _write_gdb(str(tmp_path / "r-000001.bin"), a[1:], np.array([0.8]))
    f = sbd.LoadFCIDump(str(h8["path"]))
    with pytest.raises(ValueError, match="more than one file"):
        analysis.state_energy(str(tmp_path / "r-"), f, bit_length=BITS)


def test_the_evaluator_backend_is_chosen_without_mixing_backends(monkeypatch):
    """The stochastic method never loads a backend that could interfere with one in use.

    Uses stand-in modules so every combination can be checked on any build.
    """
    import types

    import sbd as package

    modules = {"cpu": types.SimpleNamespace(__sbd_has_evaluator__=True),
               "gpu": types.SimpleNamespace(__sbd_has_evaluator__=False),
               "gpu-omp": types.SimpleNamespace(__sbd_has_evaluator__=True)}

    def scenario(built, loaded):
        monkeypatch.setattr(package, "available_backends", lambda: list(built))
        monkeypatch.setattr(package, "loaded_backends", lambda: list(loaded))
        monkeypatch.setattr(package, "get_backend", lambda d=None: modules[d])
        try:
            chosen = analysis._evaluator_backend(None)
        except RuntimeError as exc:
            return str(exc)
        return next(name for name, m in modules.items() if m is chosen)

    assert scenario(["cpu", "gpu", "gpu-omp"], []) == "cpu"
    assert scenario(["cpu", "gpu", "gpu-omp"], ["gpu-omp"]) == "gpu-omp"   # never add cpu
    assert scenario(["cpu", "gpu", "gpu-omp"], ["gpu"]) == "cpu"           # cpu beside Thrust
    assert scenario(["gpu-omp"], []) == "gpu-omp"
    assert "Thrust backend does not" in scenario(["gpu"], ["gpu"])         # Thrust-only build
    assert "Thrust backend does not" in scenario(["gpu", "gpu-omp"], ["gpu"])   # no omp beside Thrust


def test_the_thrust_backend_has_no_evaluator_and_is_not_used_for_it(backend):
    """Upstream builds the evaluator only without SBD_THRUST; the CPU module runs it."""
    for device in sbd.available_backends():
        expected = device != "gpu"
        assert sbd.get_backend(device).__sbd_has_evaluator__ is expected
    if "gpu" in sbd.available_backends():
        with pytest.raises(ValueError, match="no evaluator"):
            analysis._evaluator_backend("gpu")


# --- the exact method past 63 orbitals -----------------------------------------------

WIDE_SCRIPT = r"""
import itertools, json, pathlib, sys, tempfile
import numpy as np
from pyscf import ao2mo, gto, scf
from pyscf.tools import fcidump as fd
import sbd
from sbd import analysis

norb, positions, bits = 65, [30, 64], 20
sbd.init(device=sys.argv[1])
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-12)
c, n = mf.mo_coeff, mol.nao
h1_s = c.T @ scf.hf.get_hcore(mol) @ c
eri_s = ao2mo.restore(1, ao2mo.kernel(mol, c), n)
idx = np.array(positions)
h1 = np.zeros((norb, norb)); h1[np.ix_(idx, idx)] = h1_s
eri = np.zeros((norb,) * 4); eri[np.ix_(idx, idx, idx, idx)] = eri_s
work = pathlib.Path(tempfile.mkdtemp())
fd.from_integrals(str(work / "wide.txt"), h1, eri, norb, mol.nelec, nuc=mf.energy_nuc())
fd.from_integrals(str(work / "small.txt"), h1_s, eri_s, n, mol.nelec, nuc=mf.energy_nuc())

small = ["".join("1" if k in occ else "0" for k in reversed(range(n)))
         for occ in itertools.combinations(range(n), 1)][:1]      # 1 of 2: outside is nonzero
wide = ["".join("1" if any(positions[k] == p and s[n - 1 - k] == "1" for k in range(n))
                else "0" for p in reversed(range(norb))) for s in small]

def solve_and_analyse(path, strings, width):
    f = sbd.LoadFCIDump(str(path))
    config = sbd.TPB_SBD()
    config.eps, config.max_it, config.bit_length = 1e-12, 300, bits
    det = sbd.sort_bitarray([sbd.from_string(s, bits, width) for s in strings])
    prefix = str(path) + "-state-"
    sbd.tpb_diag(f, det, det, config, savename=prefix)
    r = analysis.variance(prefix, f, method="exact", bit_length=bits)
    return [r.energy, r.variance, r.pt2]

if sys.argv[2] == "wide":
    result = solve_and_analyse(work / "wide.txt", wide, norb)
else:
    result = solve_and_analyse(work / "small.txt", small, n)
print("RESULT " + json.dumps(result))
"""


def test_exact_past_63_orbitals_matches_the_small_system(device_config):
    """PySCF's string addressing stops at 63 orbitals; the exact method does not.

    H2 embedded at orbitals 30 and 64 of 65, the rest uncoupled: the energy, variance and
    PT2 must be the small system's. Small on purpose: the exact method's full space and
    integrals grow fast with the orbital count, and a larger case takes minutes on a
    CI runner. Each in its own subprocess, since ``det_vector``'s word
    count is fixed for the life of a process, and the two systems need different counts.
    """
    import json
    import subprocess
    import sys

    device = device_config.device if device_config is not None else "cpu"
    r = {}
    for system in ("wide", "small"):
        done = subprocess.run([sys.executable, "-c", WIDE_SCRIPT, device, system],
                              capture_output=True, text=True, timeout=900)
        assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
        line = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")][-1]
        r[system] = json.loads(line[len("RESULT "):])
    assert r["wide"][0] == pytest.approx(r["small"][0], abs=1e-9)
    assert r["wide"][1] == pytest.approx(r["small"][1], rel=1e-8)
    assert r["wide"][2] == pytest.approx(r["small"][2], rel=1e-8)
    assert r["small"][1] > 1e-6, "the subspace must leave something outside it"
