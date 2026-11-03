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

"""``sbd.analysis.caop_variance``: upstream's CAOP stat evaluator, against exact values.

The state is upstream's 4-site Hubbard ring (as ``test_caop.py``) solved by CAOP over a
subspace of the full space, so something lies outside it. The references are upstream's
own definitions, computed in NumPy from the same operator terms::

    variance = sum over determinants a outside the subspace of |<a|H|psi>|^2
    PT2      = sum over the same a of |<a|H|psi>|^2 / (E - H_aa)

The stochastic estimates, averaged over independent seeds, must land on them within
their error. Each run is a **subprocess**, since CAOP fixes the process-wide word count.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_caop import _apply, _hubbard, _read_terms  # noqa: E402

MODES, BITS, KEEP = 8, 20, 24          # 24 of the 36 states of the 2+2 sector

RUN = r"""
import json, sys
import sbd
from sbd import analysis
a = json.loads(sys.argv[1])
sbd.init(device=a["device"])
config = sbd.CAOP_SBD()
config.system_size, config.bit_length = a["modes"], a["bits"]
config.method, config.max_it, config.max_nb, config.eps = 1, 300, 20, 1e-13
ground = sbd.caop_diag_from_files(a["ham"], [a["basis"]], config, savename=a["state"])
runs = [analysis.caop_variance(a["state"], a["ham"], [a["basis"]], ground["energy"],
                               sites=a["modes"], bit_length=a["bits"], samples=4000,
                               batches=10, seed=100 + s) for s in range(8)]
only = analysis.caop_variance(a["state"], a["ham"], [a["basis"]], None,
                              sites=a["modes"], bit_length=a["bits"], samples=500,
                              batches=2, seed=7, observable="variance")
print("RESULT " + json.dumps({
    "energy": ground["energy"],
    "variance": [r.variance for r in runs], "pt2": [r.pt2 for r in runs],
    "dimension": runs[0].dimension, "error": runs[0].variance_error,
    "only_variance": only.variance, "only_pt2": only.pt2}))
"""


def _exact(ham, kept, full):
    """Subspace ground energy, and upstream's external variance and PT2 of its state."""
    fermion, terms = _read_terms(ham)
    index = {s: i for i, s in enumerate(full)}
    h = np.zeros((len(full), len(full)))
    for j, s in enumerate(full):
        for coefficient, ops in terms:
            sign, t = _apply(s, ops, fermion)
            if t is not None and t in index:
                h[index[t], j] += coefficient.real * sign
    inside = [index[s] for s in kept]
    outside = [index[s] for s in full if s not in set(kept)]
    energies, vectors = np.linalg.eigh(h[np.ix_(inside, inside)])
    e, psi = energies[0], vectors[:, 0]
    coupling = h[np.ix_(outside, inside)] @ psi          # <a|H|psi> for a outside
    variance = float(coupling @ coupling)
    pt2 = float(np.sum(coupling ** 2 / (e - np.diag(h)[outside])))
    return e, variance, pt2


@pytest.fixture
def caop_device(device_config):
    device = device_config.device if device_config is not None else "cpu"
    if device == "gpu-omp":
        pytest.skip("upstream SBD's CAOP code has no OpenMP-offload path")
    return device


def test_caop_variance_and_pt2_match_the_exact_values(caop_device, tmp_path):
    ham, _, full = _hubbard(tmp_path)
    kept = full[:KEEP]
    basis = tmp_path / "subspace.txt"
    basis.write_text("\n".join(format(s, f"0{MODES}b") for s in kept) + "\n")
    e, variance, pt2 = _exact(ham, kept, full)
    assert variance > 1e-3, "the subspace must leave something outside it"

    args = {"device": caop_device, "ham": str(ham), "basis": str(basis), "modes": MODES,
            "bits": BITS, "state": str(tmp_path / "gs-")}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    done = subprocess.run([sys.executable, "-c", RUN, json.dumps(args)],
                          capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    out = json.loads([l for l in done.stdout.splitlines()
                      if l.startswith("RESULT ")][-1][len("RESULT "):])

    assert out["energy"] == pytest.approx(e, abs=1e-9)
    assert out["dimension"] == KEEP and out["error"] > 0
    for name, exact in (("variance", variance), ("pt2", pt2)):
        values = np.array(out[name])
        error = values.std(ddof=1) / np.sqrt(len(values))
        assert abs(values.mean() - exact) < 5 * error + 1e-12, (name, values.mean(), exact)
    assert np.isfinite(out["only_variance"]) and out["only_pt2"] != out["only_pt2"]   # NaN


def test_variance_alone_needs_no_energy_and_pt2_does(backend, saved):
    """As upstream: --reference-energy is required for pt2 and both, not for variance."""
    from sbd import analysis

    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    tmp_path, ham, basis, kept, coef = saved
    _write_state(tmp_path / "gs-000000.bin", kept, coef)
    for observable in ("both", "pt2"):
        with pytest.raises(ValueError, match="needs the state's energy"):
            analysis.caop_variance(tmp_path / "gs-", ham, [basis], None, sites=MODES,
                                   bit_length=BITS, observable=observable)


def test_a_bad_observable_is_refused():
    from sbd import analysis

    with pytest.raises(ValueError, match="observable"):
        analysis.caop_variance("x-", "h.txt", ["b.txt"], 0.0, sites=4, bit_length=20,
                               observable="energy")


# --- what upstream does not check --------------------------------------------------
#
# Upstream's loader maps saved amplitudes onto the basis by determinant and normalizes
# the result, with no type tag or size check, so each of these gave a wrong answer with
# exit status 0 (a real/complex mismatch about 45 standard errors off). The bindings
# check the files on every rank first. The state is written in SaveWavefunction's format
# directly, and every check runs before upstream's loader, so these run in this process.

def _write_state(path, states, coef, words=1):
    with open(path, "wb") as f:
        np.array([len(states), words], dtype=np.uint64).tofile(f)
        np.array(states, dtype=np.uint64).tofile(f)
        np.asarray(coef).tofile(f)


@pytest.fixture
def saved(tmp_path):
    """A normalized real state on 24 of the Hubbard ring's 36 states, and its basis."""
    ham, _, full = _hubbard(tmp_path)
    kept = full[:KEEP]
    basis = tmp_path / "basis.txt"
    basis.write_text("\n".join(format(s, f"0{MODES}b") for s in kept) + "\n")
    coef = np.random.default_rng(1).normal(size=KEEP)
    coef /= np.linalg.norm(coef)
    return tmp_path, ham, basis, kept, coef


def _evaluate(backend, tmp_path, ham, basis, complex_=False):
    from sbd import analysis

    return analysis.caop_variance(tmp_path / "gs-", ham, [basis], -1.0, sites=MODES,
                                  bit_length=BITS, samples=200, batches=2,
                                  complex=complex_)


@pytest.mark.parametrize("saved_as, read_as", [("real", "complex"), ("complex", "real")])
def test_a_real_complex_mismatch_is_refused(backend, saved, saved_as, read_as):
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    tmp_path, ham, basis, kept, coef = saved
    _write_state(tmp_path / "gs-000000.bin", kept,
                 coef.astype(complex) if saved_as == "complex" else coef)
    with pytest.raises(ValueError, match=f"holds {saved_as} amplitudes"):
        _evaluate(backend, tmp_path, ham, basis, complex_=read_as == "complex")


@pytest.mark.parametrize("case, match", [
    ("unnormalized", "norm squared"),
    ("basis-not-state", "not the saved state's"),
    ("duplicate", "more than once"),
    ("truncated", "truncated"),
])
def test_a_state_upstream_reads_wrongly_is_refused(backend, saved, case, match):
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    tmp_path, ham, basis, kept, coef = saved
    _write_state(tmp_path / "gs-000000.bin", kept, 2 * coef if case == "unnormalized" else coef)
    if case == "basis-not-state":
        basis.write_text("\n".join(format(s, f"0{MODES}b") for s in kept[:20]) + "\n")
    if case == "duplicate":
        basis.write_text("\n".join(format(s, f"0{MODES}b") for s in kept + kept[:1]) + "\n")
    if case == "truncated":
        data = (tmp_path / "gs-000000.bin").read_bytes()
        (tmp_path / "gs-000000.bin").write_bytes(data[:-5])
    with pytest.raises(ValueError, match=match):
        _evaluate(backend, tmp_path, ham, basis)


@pytest.mark.mpi
def test_bad_inputs_are_refused_on_every_rank(backend):
    """Upstream reads the Hamiltonian and the state on some ranks only, so a bad file
    used to raise there and hang the rest. Every rank now refuses together."""
    import tempfile

    from mpi4py import MPI

    from sbd import analysis

    comm = MPI.COMM_WORLD
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    path = pathlib.Path(comm.bcast(tempfile.mkdtemp(prefix="caop-var-")
                                   if comm.Get_rank() == 0 else None, root=0))
    if comm.Get_rank() == 0:
        ham, _, full = _hubbard(path)
        kept = full[:KEEP]
        (path / "basis.txt").write_text("\n".join(format(s, f"0{MODES}b") for s in kept) + "\n")
        coef = np.ones(KEEP) / np.sqrt(KEEP)
        _write_state(path / "gs-000000.bin", kept, coef.astype(complex))   # complex, read as real
    comm.Barrier()
    try:
        analysis.caop_variance(path / "gs-", path / "hubbard.txt", [path / "basis.txt"], -1.0,
                               sites=MODES, bit_length=BITS, samples=200, batches=2)
    except ValueError as e:
        outcome = str(e)
    else:
        outcome = "SOLVED"
    outcomes = comm.allgather(outcome)
    assert all("holds complex amplitudes" in o for o in outcomes), outcomes



# --- the shared variance driver, on a CAOP state ---------------------------------------

def test_the_variance_driver_takes_a_caop_state(caop_device, tmp_path):
    """analysis/variance.py --hamfile on a state saved by run_caop_diag.py --savename, as
    examples/README.md runs it: the exact variance and PT2, within their error."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    ham, _, full = _hubbard(tmp_path)
    kept = full[:KEEP]
    basis = tmp_path / "basis.txt"
    basis.write_text("\n".join(format(s, f"0{MODES}b") for s in kept) + "\n")
    e, variance, pt2 = _exact(ham, kept, full)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")

    def run(script, *flags):
        done = subprocess.run([sys.executable, str(repo / "examples" / script), *flags],
                              cwd=tmp_path, capture_output=True, text=True, timeout=600,
                              env=env)
        assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
        return done.stdout

    ground = run("caop/run_caop_diag.py", "--device", caop_device, "--hamfile", str(ham),
                 "--basisfiles", str(basis), "--system_size", str(MODES), "--tolerance",
                 "1e-12", "--iteration", "200", "--block", "20", "--savename", "gs-")
    e0 = float(ground.split("Ground state energy:")[1].split()[0])
    assert e0 == pytest.approx(e, abs=1e-8)
    run("analysis/variance.py", "--hamfile", str(ham), "--basisfiles", str(basis),
        "--sites", str(MODES), "--state", "gs-", "--bit_length", str(BITS),
        "--energy", repr(e0), "--samples", "4000", "--batches", "40", "--json", "out.json")
    out = json.loads((tmp_path / "out.json").read_text())
    for name, exact in (("variance", variance), ("pt2", pt2)):
        assert abs(out[name] - exact) < 5 * out[f"{name}_error"] + 1e-12, (name, out[name], exact)
