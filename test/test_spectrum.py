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

"""``sbd.spectrum``: the Green's function of a CAOP ground state, against an exact one.

The model is upstream's 4-site Hubbard ring (t = -1, U = 4, two electrons of each spin),
the one ``test_caop.py`` solves. Its ground state is found with CAOP and saved, then the
Green's function is computed with block Lanczos, with the full N+1 or N-1 sector given as
``extra_basis``, as upstream's notes say it should be. The reference is the exact
Lehmann sum of upstream's formulas, from a dense diagonalization in NumPy of the same
operator terms. Each run is a **subprocess**, since CAOP fixes the process-wide word count.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_caop import _apply, _hubbard, _read_terms  # noqa: E402

SITES, BITS, ETA = 4, 20, 0.1
MODES = 2 * SITES
ORBITALS = [0, 1, SITES]                 # alpha sites 0 and 1, beta site 0
OMEGAS = np.linspace(-6.0, 8.0, 15)

RUN = r"""
import json, sys
import numpy as np
import sbd, sbd.spectrum as sp
a = json.loads(sys.argv[1])
sbd.init(device=a["device"])
config = sbd.CAOP_SBD()
config.system_size, config.bit_length = a["modes"], a["bits"]
config.method, config.max_it, config.max_nb, config.eps = 1, 200, 20, 1e-13
ground = sbd.caop_diag_from_files(a["ham"], [a["basis"]], config, savename=a["state"],
                                  complex=a["complex"])
out = {"energy": ground["energy"]}
for channel in ("addition", "removal"):
    r = sp.green_function(a["state"], a["ham"], a["orbitals"], channel, sites=a["modes"],
                          bit_length=a["bits"], reference_energy=ground["energy"],
                          omegas=a["omegas"], eta=a["eta"], extra_basis=[a[channel]],
                          max_steps=200, complex=a["complex"],
                          wavefunction_complex=a["complex"],
                          save_coefficients=a["save"] + channel)
    g = r["green_function"]
    out[channel] = {"real": g.real.tolist(), "imag": g.imag.tolist(),
                    "basis_size": r["basis_size"], "stop_reason": r["stop_reason"]}
print("RESULT " + json.dumps(out))
"""


def _states(n_alpha, n_beta):
    half = lambda n: [sum(1 << p for p in occ) for occ in itertools.combinations(range(SITES), n)]
    return sorted(a | (b << SITES) for a in half(n_alpha) for b in half(n_beta))


def _matrix(terms, fermion, states):
    index = {s: i for i, s in enumerate(states)}
    h = np.zeros((len(states), len(states)))
    for j, s in enumerate(states):
        for coefficient, ops in terms:
            sign, t = _apply(s, ops, fermion)
            if t is not None and t in index:
                h[index[t], j] += coefficient.real * sign
    return h


def _exact_green(ham, eta=ETA):
    """Upstream's G+ and G- by the Lehmann sum over the N+1 and N-1 sectors."""
    fermion, terms = _read_terms(ham)
    ground_states = _states(2, 2)
    e0_all, v0_all = np.linalg.eigh(_matrix(terms, fermion, ground_states))
    e0, psi = e0_all[0], v0_all[:, 0]

    def apply_op(op, mode, vector, source, target):
        out = np.zeros(len(target))
        index = {s: i for i, s in enumerate(target)}
        for amplitude, s in zip(vector, source):
            sign, t = _apply(s, [(op, mode)], fermion)
            if t is not None:
                out[index[t]] += sign * amplitude
        return out

    result = {}
    for channel, sector, op, sgn in (("addition", (3, 2), "cdag", +1),
                                     ("removal", (2, 1), "c", -1)):
        # Removal from beta modes lands in (2, 1); from alpha modes in (1, 2). Use the
        # union of both N-1 sectors, and of both N+1 sectors, so every orbital fits.
        states = sorted(set(_states(*sector)) | set(_states(*sector[::-1])))
        energies, vectors = np.linalg.eigh(_matrix(terms, fermion, states))
        projected = np.array([vectors.T @ apply_op(op, m, psi, ground_states, states)
                              for m in ORBITALS])            # (orbital, eigenstate)
        g = np.zeros((len(OMEGAS), len(ORBITALS), len(ORBITALS)), dtype=complex)
        for k, w in enumerate(OMEGAS):
            if channel == "addition":       # <c_a (w + E0 - H + i eta)^-1 c_b^dag>
                denominator = w + e0 - energies + 1j * eta
            else:                           # <c_b^dag (w - E0 + H + i eta)^-1 c_a>
                denominator = w - e0 + energies + 1j * eta
            g[k] = (projected.conj() / denominator) @ projected.T
            if channel == "removal":
                g[k] = g[k].T
        result[channel] = (g, states)
    return e0, result


@pytest.fixture
def caop_device(device_config):
    device = device_config.device if device_config is not None else "cpu"
    if device == "gpu-omp":
        pytest.skip("upstream SBD's CAOP code has no OpenMP-offload path")
    return device


def _run(caop_device, tmp_path, complex_=False):
    ham, basis, _ = _hubbard(tmp_path)
    e0, exact = _exact_green(ham)
    sectors = {}
    for channel, (_, states) in exact.items():
        path = tmp_path / f"{channel}_sector.txt"
        path.write_text("\n".join(format(s, f"0{MODES}b") for s in states) + "\n")
        sectors[channel] = str(path)
    args = {"device": caop_device, "ham": str(ham), "basis": str(basis), "modes": MODES,
            "bits": BITS, "state": str(tmp_path / "gs-"), "orbitals": ORBITALS,
            "omegas": OMEGAS.tolist(), "eta": ETA, "complex": complex_,
            "save": str(tmp_path / "coefficients-"), **sectors}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    done = subprocess.run([sys.executable, "-c", RUN, json.dumps(args)],
                          capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    out = json.loads([l for l in done.stdout.splitlines()
                      if l.startswith("RESULT ")][-1][len("RESULT "):])
    return e0, exact, out


def test_green_function_matches_the_exact_lehmann_sum(caop_device, tmp_path):
    e0, exact, out = _run(caop_device, tmp_path)
    assert out["energy"] == pytest.approx(e0, abs=1e-9)
    for channel in ("addition", "removal"):
        got = np.array(out[channel]["real"]) + 1j * np.array(out[channel]["imag"])
        np.testing.assert_allclose(got, exact[channel][0], atol=1e-7, err_msg=channel)
        assert (tmp_path / f"coefficients-{channel}").is_file()


def test_saved_coefficients_reevaluate_without_rerunning(caop_device, tmp_path):
    """Upstream's --read-coefficients: the saved coefficients give the same G on the
    same grid, and the exact G on a new broadening, with no Hamiltonian or state."""
    from sbd.spectrum import green_function_from_coefficients

    _, exact, out = _run(caop_device, tmp_path)
    for channel in ("addition", "removal"):
        path = tmp_path / f"coefficients-{channel}"
        same = green_function_from_coefficients(path, OMEGAS, ETA, device=caop_device)
        got = np.array(out[channel]["real"]) + 1j * np.array(out[channel]["imag"])
        np.testing.assert_allclose(same["green_function"], got, atol=1e-12)
        assert same["addition"] is (channel == "addition")
    # A new broadening, against the Lehmann sum at that broadening.
    wider = green_function_from_coefficients(tmp_path / "coefficients-addition", OMEGAS,
                                             0.3, device=caop_device)
    np.testing.assert_allclose(wider["green_function"],
                               _exact_green(_hubbard(tmp_path)[0], eta=0.3)[1]["addition"][0],
                               atol=1e-7)


def test_complex_amplitudes_give_the_same_green_function(caop_device, tmp_path):
    """Solved and saved with complex amplitudes, the state is read back as complex."""
    _, exact, out = _run(caop_device, tmp_path, complex_=True)
    for channel in ("addition", "removal"):
        got = np.array(out[channel]["real"]) + 1j * np.array(out[channel]["imag"])
        np.testing.assert_allclose(got, exact[channel][0], atol=1e-7, err_msg=channel)


@pytest.mark.mpi
def test_green_function_with_the_state_over_every_rank(caop_device):
    """The ground state solved and saved over every rank (b_comm_size = ranks), and its
    Green's function from those shards on the same ranks: the exact Lehmann sum on each.

    In the job's own ranks rather than a subprocess: one word per determinant, as the
    CAOP solves under MPI in test_caop.py.
    """
    import shutil
    import tempfile

    from mpi4py import MPI

    from sbd.spectrum import green_function

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    path = pathlib.Path(comm.bcast(tempfile.mkdtemp(prefix="sbd-spectrum-")
                                   if comm.Get_rank() == 0 else None, root=0))
    try:
        if comm.Get_rank() == 0:
            _hubbard(path)
        comm.Barrier()
        ham, basis = path / "hubbard.txt", path / "hubbard_basis.txt"
        e0, exact = _exact_green(ham)
        if comm.Get_rank() == 0:
            for channel, (_, states) in exact.items():
                (path / f"{channel}_sector.txt").write_text(
                    "\n".join(format(s, f"0{MODES}b") for s in states) + "\n")
        comm.Barrier()
        config = sbd.CAOP_SBD()
        config.system_size, config.bit_length = MODES, BITS
        config.method, config.max_it, config.max_nb, config.eps = 1, 200, 20, 1e-13
        config.b_comm_size, config.do_redist_basis = size, True
        # A fixed start, since the default start is random and every rank must agree.
        config.initial_determinant_bitstring = basis.read_text().split()[0]
        ground = sbd.caop_diag_from_files(ham, [basis], config, savename=str(path / "gs-"),
                                          device=caop_device)
        assert ground["energy"] == pytest.approx(e0, abs=1e-9)
        for channel in ("addition", "removal"):
            r = green_function(path / "gs-", ham, ORBITALS, channel, sites=MODES,
                               bit_length=BITS, reference_energy=ground["energy"],
                               omegas=OMEGAS, eta=ETA, shards=size,
                               extra_basis=[path / f"{channel}_sector.txt"], max_steps=200,
                               device=caop_device)
            np.testing.assert_allclose(r["green_function"], exact[channel][0], atol=1e-7,
                                       err_msg=f"{channel} on rank {comm.Get_rank()}")
    finally:
        comm.Barrier()
        if comm.Get_rank() == 0:
            shutil.rmtree(path, ignore_errors=True)


def test_spectral_function_is_minus_im_g_over_pi_on_the_diagonal():
    from sbd.spectrum import spectral_function

    rng = np.random.default_rng(0)
    g = rng.normal(size=(5, 3, 3)) + 1j * rng.normal(size=(5, 3, 3))
    a = spectral_function(g)
    np.testing.assert_allclose(np.diagonal(a, axis1=1, axis2=2).real,
                               -np.diagonal(g, axis1=1, axis2=2).imag / np.pi)
    np.testing.assert_allclose(a, np.conj(np.swapaxes(a, 1, 2)))          # Hermitian


def test_a_bad_channel_is_refused(backend):
    from sbd.spectrum import green_function

    with pytest.raises(ValueError, match="channel"):
        green_function("x-", "h.txt", [0], "both", sites=4, bit_length=20,
                       reference_energy=0.0, omegas=[0.0])


def test_the_offload_backend_has_no_spectrum():
    if "gpu-omp" not in sbd.available_backends():
        pytest.skip("the OpenMP-offload backend was not built")
    from sbd.spectrum import green_function

    with pytest.raises(ValueError, match="no CAOP solver"):
        green_function("x-", "h.txt", [0], "addition", sites=4, bit_length=20,
                       reference_energy=0.0, omegas=[0.0], device="gpu-omp")


# --- general operators: upstream's spin-dimer example ---------------------------------

DIMER = (pathlib.Path(__file__).resolve().parents[1] / "vendor" / "sbd-upstream" / "apps"
         / "caop-spectrum" / "examples" / "spin_dimer")

GENERAL = r"""
import json, sys
import numpy as np
import sbd, sbd.spectrum as sp
a = json.loads(sys.argv[1])
sbd.init(device=a["device"])
# The singlet of H = S0.S1 over {01, 10}, the state upstream's write_state.cc saves.
config = sbd.CAOP_SBD()
config.system_size, config.bit_length = 2, 20
config.method, config.max_it, config.max_nb, config.eps = 1, 100, 2, 1e-13
config.max_iv = 1
ground = sbd.caop_diag_from_files(a["ham"], [a["basis"]], config, savename=a["state"])
r = sp.green_function(a["state"], a["ham"], [], "addition", sites=2, bit_length=20,
                      reference_energy=ground["energy"], omegas=a["omegas"], eta=a["eta"],
                      operator_files=a["ops"], max_steps=8, complex=True)
g = r["green_function"]
print("RESULT " + json.dumps({"energy": ground["energy"], "real": g.real.tolist(),
                              "imag": g.imag.tolist()}))
"""


def test_general_operators_give_upstreams_spin_dimer_answer(caop_device, tmp_path):
    """Upstream's example: S0.S1 with Sz and Sy on each site, from the singlet.

    G_aa = 1 / (4 (w - 1 + i eta)) for Sz/Sz and Sy/Sy, the opposite-site element its
    negative, and the Sz/Sy cross elements zero (upstream's verify.py).
    """
    basis = tmp_path / "basis.txt"
    basis.write_text("01\n10\n")
    omegas, eta = np.linspace(0.0, 2.0, 11), 0.05
    args = {"device": caop_device, "ham": str(DIMER / "ham.txt"), "basis": str(basis),
            "state": str(tmp_path / "state-"), "omegas": omegas.tolist(), "eta": eta,
            "ops": [str(DIMER / f) for f in ("sz0.txt", "sz1.txt", "sy0.txt", "sy1.txt")]}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    done = subprocess.run([sys.executable, "-c", GENERAL, json.dumps(args)],
                          capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    out = json.loads([l for l in done.stdout.splitlines()
                      if l.startswith("RESULT ")][-1][len("RESULT "):])
    assert out["energy"] == pytest.approx(-0.75, abs=1e-12)
    g = np.array(out["real"]) + 1j * np.array(out["imag"])
    same = 1 / (4 * (omegas - 1 + 1j * eta))
    expected = np.zeros((len(omegas), 4, 4), dtype=complex)
    for block in (0, 2):                 # (Sz0, Sz1) and (Sy0, Sy1)
        expected[:, block, block] = expected[:, block + 1, block + 1] = same
        expected[:, block, block + 1] = expected[:, block + 1, block] = -same
    np.testing.assert_allclose(g, expected, atol=1e-10)


def test_orbitals_and_operator_files_together_are_refused(backend):
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    from sbd.spectrum import green_function

    with pytest.raises(ValueError, match="not both"):
        green_function("x-", DIMER / "ham.txt", [0], "addition", sites=2, bit_length=20,
                       reference_energy=0.0, omegas=[0.0], operator_files=[DIMER / "sz0.txt"])
    with pytest.raises(ValueError, match="non-negative"):
        green_function("x-", DIMER / "ham.txt", [0], "addition", sites=2, bit_length=20,
                       reference_energy=0.0, omegas=[0.0], rank_atol=-1.0)


def test_operator_files_upstream_checks_on_some_ranks_only_are_refused(backend, tmp_path):
    """A mode beyond sites, or an operator with other statistics than H: upstream
    raises on the h=t=0 ranks or on one h shard only, which hangs the other ranks."""
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    from sbd.spectrum import green_function

    wide = tmp_path / "wide.txt"
    wide.write_text("1\n1.0 sz 0 sz 2\n")
    with pytest.raises(ValueError, match="mode 2, beyond sites=2"):
        green_function("x-", wide, [0], "addition", sites=2, bit_length=20,
                       reference_energy=0.0, omegas=[0.0])
    fermion = tmp_path / "fermion.txt"
    fermion.write_text("-1\n1.0 cdag 0 c 1\n")
    with pytest.raises(ValueError, match="different statistics"):
        green_function("x-", DIMER / "ham.txt", [], "addition", sites=2, bit_length=20,
                       reference_energy=0.0, omegas=[0.0], operator_files=[fermion])


# --- the driver: the README's flow ---------------------------------------------------

def test_the_drivers_flow_ground_state_then_green_function(caop_device, tmp_path):
    """run_caop_diag.py --savename, then run_caop_spectrum.py on the saved state, and
    --read-coefficients on its coefficients: the exact G+ of the 2-site Hubbard model."""
    caop = pathlib.Path(__file__).resolve().parents[1] / "examples" / "caop"
    ham = tmp_path / "hub2.txt"
    ham.write_text("-1\n-1.0 cdag 0 c 1\n-1.0 cdag 1 c 0\n-1.0 cdag 2 c 3\n"
                   "-1.0 cdag 3 c 2\n4.0 cdag 0 cdag 2 c 2 c 0\n4.0 cdag 1 cdag 3 c 3 c 1\n")
    (tmp_path / "hub2_basis.txt").write_text("0101\n0110\n1001\n1010\n")
    (tmp_path / "add_basis.txt").write_text("0111\n1011\n1101\n1110\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")

    def run(script, *flags):
        done = subprocess.run([sys.executable, str(caop / script), "--device", caop_device,
                               *flags], cwd=tmp_path, capture_output=True, text=True,
                              timeout=600, env=env)
        assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
        return done.stdout

    ground = run("run_caop_diag.py", "--hamfile", "hub2.txt", "--basisfiles", "hub2_basis.txt",
                 "--system_size", "4", "--tolerance", "1e-10", "--block", "3",
                 "--savename", "gs-")
    e0 = float(ground.split("Ground state energy:")[1].split()[0])
    assert e0 == pytest.approx((4 - np.sqrt(16 + 16)) / 2, abs=1e-9)     # (U - sqrt(U^2+16t^2))/2
    grid = ["--omega-min", "-6", "--omega-max", "8", "--points", "15"]
    run("run_caop_spectrum.py", "--hamfile", "hub2.txt", "--sites", "4", "--loadname", "gs-",
        "--reference-energy", repr(e0), "--orbitals", "0,2", "--channel", "addition",
        "--extra-detfile", "add_basis.txt", "--save-coefficients", "add.coeff",
        "--output", "add.csv", *grid, "--eta", "0.1")
    run("run_caop_spectrum.py", "--read-coefficients", "add.coeff", "--output", "narrow.csv",
        *grid, "--eta", "0.05")

    terms = _read_terms(ham)
    fermion = terms[0]
    ground_states, add_states = [5, 6, 9, 10], [7, 11, 13, 14]

    def matrix(states):
        index = {x: i for i, x in enumerate(states)}
        h = np.zeros((len(states), len(states)))
        for j, x in enumerate(states):
            for c, ops in terms[1]:
                sign, y = _apply(x, ops, fermion)
                if y is not None and y in index:
                    h[index[y], j] += c.real * sign
        return h

    e, v = np.linalg.eigh(matrix(ground_states))
    psi = v[:, 0]
    ea, va = np.linalg.eigh(matrix(add_states))
    index = {x: i for i, x in enumerate(add_states)}

    def seed(mode):
        out = np.zeros(len(add_states))
        for amplitude, x in zip(psi, ground_states):
            sign, y = _apply(x, [("cdag", mode)], fermion)
            if y is not None:
                out[index[y]] += sign * amplitude
        return va.T @ out

    projected = np.array([seed(0), seed(2)])
    omegas = np.linspace(-6, 8, 15)
    for path, eta in (("add.csv", 0.1), ("narrow.csv", 0.05)):
        rows = np.loadtxt(tmp_path / path, delimiter=",", skiprows=1)
        got = (rows[:, 3] + 1j * rows[:, 4]).reshape(len(omegas), 2, 2)
        expected = np.array([(projected.conj() / (w + e[0] - ea + 1j * eta)) @ projected.T
                             for w in omegas])
        np.testing.assert_allclose(got, expected, atol=1e-8, err_msg=path)
        # The reevaluated CSV keeps the generating run's orbital labels.
        assert sorted(set(rows[:, 1].astype(int))) == [0, 2], path
