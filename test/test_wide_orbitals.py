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

"""``solve_sci`` past 63 orbitals, where a CI string no longer fits an int64.

qiskit-addon-sqd hands the solver CI strings as Python ints from 64 orbitals on
(``bitstring_matrix_to_integers``), and SBD packs any width into words of
``bit_length`` bits. Our conversion back to CI strings used to force int64, so
``solve_sci`` failed with ``OverflowError`` at 64 orbitals on every backend.

The check needs an exact answer at 70 orbitals, so it builds one: a small molecule
(H8, 8 orbitals) embedded at scattered positions of a 70-orbital space, with every
other orbital uncoupled. A subspace of strings over the embedded orbitals is then the
small molecule's full space, so SBD must reproduce its FCI energy, its strings, its
amplitudes and its 1-RDM, all placed at the embedded positions. The positions straddle
the int64 limit (63 | 64) and the packed-word boundaries at the default bit_length of
20 (19 | 20, 39 | 40), and a monotone embedding preserves both the string order and
PySCF's sign convention, so the comparison is elementwise.

The solve runs in a **subprocess**: ``det_vector``'s word count is fixed for the
life of a process, and it differs from what the rest of the suite uses. It tests the
backend ``SBD_TEST_DEVICE`` names, like the rest of the suite (cpu when unset), and
skips one that is not built. ``tox -e py`` runs it on one rank; ``tox -e mpi`` runs it
on two.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import sbd
from sbd.sbd_solver import _ci_string_dtype, _occupations

NORB = 70
POSITIONS = [0, 19, 20, 39, 40, 63, 64, 69]

SCRIPT = r"""
import json, sys
import numpy as np
from pyscf import ao2mo, gto, scf
from qiskit_addon_sqd import fermion
import sbd
from sbd.device_config import DeviceConfig
from sbd.sbd_solver import solve_sci

device, norb, positions = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
sbd.init(device=device)

mol = gto.M(atom="; ".join(f"H 0 0 {0.95 * i:.2f}" for i in range(len(positions))),
            basis="sto-3g", verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-12)
c = mf.mo_coeff
n = mol.nao
h1_small = c.T @ scf.hf.get_hcore(mol) @ c
eri_small = ao2mo.restore(1, ao2mo.kernel(mol, c), n)

# Embed: every other orbital is uncoupled, and never occupied by the subspace.
idx = np.array(positions)
h1 = np.zeros((norb, norb))
h1[np.ix_(idx, idx)] = h1_small
eri = np.zeros((norb,) * 4)
eri[np.ix_(idx, idx, idx, idx)] = eri_small

def widen(s):
    return sum(1 << positions[k] for k in range(n) if (int(s) >> k) & 1)

import itertools
small = np.array(sorted(sum(1 << o for o in occ)
                        for occ in itertools.combinations(range(n), mol.nelec[0])))
expected = fermion.solve_sci((small, small), h1_small, eri_small, norb=n,
                             nelec=mol.nelec, tol=1e-14)
wide = np.array([widen(s) for s in small], dtype=object)

from mpi4py import MPI
comm = MPI.COMM_WORLD
got = solve_sci((wide, wide), h1, eri, norb=norb, nelec=mol.nelec,
                device_config=DeviceConfig(device=device),
                sbd_config={"eps": 1e-12, "max_it": 300, "do_rdm": 1,
                            "adet_comm_size": comm.Get_size()})
if comm.Get_rank() != 0:
    sys.exit(0)                       # the state is gathered onto rank 0 only

a, b = got.sci_state.amplitudes, expected.sci_state.amplitudes
overlap = float(np.vdot(b, a))
rdm1_expected = np.zeros((norb, norb))
rdm1_expected[np.ix_(idx, idx)] = expected.sci_state.rdm(rank=1, spin_summed=True)
print("RESULT", json.dumps({
    "backend": sbd.get_backend(device).__name__,
    "ranks": comm.Get_size(),
    "energy": got.energy,
    "expected_energy": expected.energy,
    "dtype": str(got.sci_state.ci_strs_a.dtype),
    "strings_a": [str(x) for x in got.sci_state.ci_strs_a],
    "strings_b": [str(x) for x in got.sci_state.ci_strs_b],
    "expected_strings": [str(x) for x in wide],
    "overlap": overlap,
    "amplitude_error": float(np.max(np.abs(a * np.sign(overlap) - b))),
    "rdm1_error": float(np.max(np.abs(got.rdm1 - rdm1_expected))),
    "occupancy_error": float(np.max(np.abs(
        got.orbital_occupancies[0] + got.orbital_occupancies[1] - np.diag(rdm1_expected)))),
}))
"""


def _device(device_config):
    """The backend to test: ``SBD_TEST_DEVICE``, as the rest of the suite uses, else cpu.

    ``device_config`` (conftest) reads that variable and skips a backend that is not
    built. When it is unset this uses cpu rather than the build's own default:
    resolving that would load a backend into this process, which the subprocess
    below exists to avoid.
    """
    return device_config.device if device_config is not None else "cpu"


def _require_dependencies():
    pytest.importorskip("pyscf", reason="pyscf builds the reference Hamiltonian")
    pytest.importorskip("qiskit_addon_sqd.fermion", reason="qiskit-addon-sqd provides solve_sci")


def _run_script(device, launcher=()):
    # Started from inside an MPI job (the mpi test), the inner launcher would see the
    # outer job's process-manager variables and try to join it. Give it a clean start.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI_", "PMIX_", "OMPI_", "MPI_LOCAL"))}
    return subprocess.run(
        [*launcher, sys.executable, "-c", SCRIPT, device, str(NORB), json.dumps(POSITIONS)],
        capture_output=True, text=True, timeout=900 if not launcher else 300, env=env,
    )


def test_solve_sci_past_63_orbitals_standalone(device_config):
    """SBD reproduces an embedded molecule's FCI state at 70 orbitals."""
    _require_dependencies()
    _check_result(_run_script(_device(device_config)))


@pytest.mark.mpi
def test_solve_sci_past_63_orbitals_mpi(device_config):
    """The same on several ranks, splitting the alpha strings between them.

    Only rank 0 converts the solved strings back to CI strings, so before the fix
    only rank 0 raised (#70). Under run_sqd_sbd.py the other rank then waited for
    it forever; the timeout makes a hang like that a failure instead.

    The 70-orbital solve needs more words per determinant than the rest of the
    suite, and SBD fixes that count per process, so it cannot run in the ranks of
    this job. Rank 0 starts a separate job instead, on SBD_TEST_NPROCS ranks (2 when
    unset, as for ``tox -e mpi``), and every rank checks the result it reports.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Every rank skips together, before rank 0 alone starts the job: a skip raised on
    # rank 0 only would leave the other ranks waiting in the broadcast below.
    _require_dependencies()
    device = _device(device_config)
    nprocs = int(os.environ.get("SBD_TEST_NPROCS", "2"))
    outcome = None
    if comm.Get_rank() == 0:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpirun is None:
            outcome = "skip: no mpirun or mpiexec on PATH"
        else:
            try:
                outcome = _run_script(device, launcher=(mpirun, "-n", str(nprocs)))
            except subprocess.TimeoutExpired:
                outcome = (f"fail: the {nprocs}-rank run did not finish: one rank is "
                           "likely waiting on another that raised")
    outcome = comm.bcast(outcome, root=0)
    if isinstance(outcome, str):
        kind, message = outcome.split(": ", 1)
        (pytest.skip if kind == "skip" else pytest.fail)(message)
    assert _check_result(outcome)["ranks"] == nprocs


def _check_result(completed):
    assert completed.returncode == 0, (
        f"subprocess failed:\n{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
    )
    line = [l for l in completed.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, f"no result reported:\n{completed.stdout[-3000:]}"
    r = json.loads(line[-1][len("RESULT "):])

    assert r["energy"] == pytest.approx(r["expected_energy"], abs=1e-8)
    assert r["dtype"] == "object"
    assert r["strings_a"] == r["expected_strings"]
    assert r["strings_b"] == r["expected_strings"]
    assert abs(r["overlap"]) == pytest.approx(1.0, abs=1e-7)
    assert r["amplitude_error"] < 1e-6
    assert r["rdm1_error"] < 1e-6
    assert r["occupancy_error"] < 1e-6
    return r


@pytest.mark.parametrize("norb", [8, 63, 64, 70, 130])
def test_occupations_at_any_width(norb):
    """``_occupations`` reads bit p of a CI string, on both sides of 64 orbitals."""
    rng = np.random.default_rng(norb)
    strings = [sum(1 << int(p) for p in rng.choice(norb, size=min(5, norb), replace=False))
               for _ in range(20)]
    strings += [(1 << norb) - 1, 1 << (norb - 1)]
    array = np.array(strings, dtype=_ci_string_dtype(norb))
    expected = np.array([[(s >> p) & 1 for p in range(norb)] for s in strings])
    np.testing.assert_array_equal(_occupations(array, norb), expected)


def test_ci_string_dtype_matches_the_addon():
    """The int64/object switch sits where qiskit-addon-sqd's does."""
    counts = pytest.importorskip("qiskit_addon_sqd.counts")
    for norb in (63, 64):
        matrix = np.zeros((1, norb), dtype=bool)
        matrix[0, 0] = True
        addon = counts.bitstring_matrix_to_integers(matrix).dtype
        assert (addon == object) == (_ci_string_dtype(norb) is object), norb


@pytest.mark.parametrize("norb", [24, 63, 64, 70])
def test_enlarge_driver_unpacks_strings_at_any_width(norb):
    """``run_sqd_enlarge_subspace_sbd.ints_to_bool_columns`` inverts the addon's packing.

    That driver unpacks the CI strings ``solve_sci`` returns, so it has to follow the
    same width rule: a uint64 array cannot hold the Python ints it gets from 64
    orbitals on.
    """
    counts = pytest.importorskip("qiskit_addon_sqd.counts")
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "examples" / "tpb" / "run_sqd_enlarge_subspace_sbd.py")
    spec = importlib.util.spec_from_file_location("run_sqd_enlarge_subspace_sbd", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    rng = np.random.default_rng(norb)
    matrix = np.zeros((20, norb), dtype=bool)
    for row in matrix:
        row[rng.choice(norb, size=5, replace=False)] = True
    matrix[0, 0] = True                       # leftmost column: the highest orbital
    strings = counts.bitstring_matrix_to_integers(matrix)
    np.testing.assert_array_equal(driver.ints_to_bool_columns(strings, norb), matrix)
