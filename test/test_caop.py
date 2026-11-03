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

"""CAOP selected-basis diagonalization, against exact references.

The cases are upstream SBD's own:

- ``hamiltonian.txt`` and ``basis.txt`` from ``apps/caop_selected_basis_diagonalization``:
  an 8-site extended hard-core boson chain over a 16-state selected basis.
- The 4-site Hubbard ring of upstream's ``tests/functionality/test_caop_gdb_hubbard.cc``
  (t = -1, U = 4, two electrons of each spin), written here as a CAOP Hamiltonian file
  in upstream's block layout: alpha site p is mode p, beta site p is mode L + p.

Each is checked against an exact diagonalization built here in NumPy from the same
operator terms over the same basis, so the reference does not come from SBD. The
Hubbard ring is also checked against GDB on upstream's FCIDUMP of the same model, the
comparison upstream's own test makes. Every solve runs in a **subprocess**: CAOP fixes
the process-wide determinant word count, as GDB does.
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

REPO = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "vendor" / "sbd-upstream"
CAOP_APP = UPSTREAM / "apps" / "caop_selected_basis_diagonalization"
HUBBARD_FCIDUMP = UPSTREAM / "tests" / "functionality" / "caop_gdb_hubbard4_fcidump.txt"

SOLVE = r"""
import json, sys
import sbd
device, ham, basis, size, complex_ = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "1"
sbd.init(device=device)
config = sbd.CAOP_SBD()
config.system_size, config.bit_length = size, 20
config.method, config.max_it, config.max_nb, config.eps = 1, 200, 20, 1e-12
config.do_redist_basis = True
result = sbd.caop_diag_from_files(ham, basis.split(","), config, complex=complex_)
print("RESULT " + json.dumps({"energy": result["energy"]}))
"""


def _device(device_config):
    return device_config.device if device_config is not None else "cpu"


def _solve(device, ham, basis, size, complex_=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    done = subprocess.run([sys.executable, "-c", SOLVE, device, str(ham), str(basis),
                           str(size), "1" if complex_ else "0"],
                          capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    line = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])["energy"]


# --- an independent reference: the operator terms applied in NumPy ------------------

def _read_terms(path):
    """(fermion, [(coefficient, [(op, mode), ...]), ...]) from a CAOP Hamiltonian file."""
    lines = [l.split("#")[0].strip() for l in pathlib.Path(path).read_text().splitlines()]
    lines = [l for l in lines if l]
    fermion = int(float(lines[0])) <= 0
    terms = []
    for line in lines[1:]:
        fields = line.split()
        ops = [(fields[k], int(fields[k + 1])) for k in range(1, len(fields), 2)]
        terms.append((_coefficient(fields[0]), ops))
    return fermion, terms


def _coefficient(text):
    """A coefficient as upstream writes it: a real number, or (re,im)."""
    if text.startswith("("):
        re, im = text.strip("()").split(",")
        return complex(float(re), float(im))
    return complex(float(text))


def _apply(state, ops, fermion):
    """Apply a product of operators (rightmost first) to a basis state; (sign, state)."""
    sign = 1.0
    for op, mode in reversed(ops):
        occupied = (state >> mode) & 1
        if op in ("c", "b") and not occupied or op in ("cdag", "bdag") and occupied:
            return 0.0, None
        if fermion:
            sign *= (-1) ** bin(state & ((1 << mode) - 1)).count("1")
        state ^= 1 << mode
    return sign, state


def _exact(path, states):
    """Lowest eigenvalue of the Hamiltonian file restricted to ``states``."""
    fermion, terms = _read_terms(path)
    index = {s: i for i, s in enumerate(states)}
    h = np.zeros((len(states), len(states)), dtype=complex)
    for j, s in enumerate(states):
        for coefficient, ops in terms:
            sign, t = _apply(s, ops, fermion)
            if t is not None and t in index:
                h[index[t], j] += coefficient * sign
    assert np.allclose(h, h.conj().T), "the Hamiltonian is not Hermitian on this basis"
    return float(np.linalg.eigvalsh(h)[0])


def _bitstrings(path):
    return [int(l.strip(), 2) for l in pathlib.Path(path).read_text().split()]


def _hubbard(tmp_path, sites=4, t=-1.0, u=4.0, ring=True):
    """Upstream's Hubbard ring as a CAOP Hamiltonian file, and its full basis."""
    bonds = [(i, i + 1) for i in range(sites - 1)] + ([(0, sites - 1)] if ring else [])
    lines = ["-1"]
    for spin in (0, sites):
        for i, j in bonds:
            lines += [f"{t} cdag {i + spin} c {j + spin}", f"{t} cdag {j + spin} c {i + spin}"]
    for i in range(sites):
        lines.append(f"{u} cdag {i} cdag {i + sites} c {i + sites} c {i}")
    ham = tmp_path / "hubbard.txt"
    ham.write_text("\n".join(lines) + "\n")
    half = [sum(1 << p for p in occ) for occ in itertools.combinations(range(sites), sites // 2)]
    states = sorted(a | (b << sites) for a in half for b in half)
    basis = tmp_path / "hubbard_basis.txt"
    basis.write_text("\n".join(format(s, f"0{2 * sites}b") for s in states) + "\n")
    return ham, basis, states


# --- tests -----------------------------------------------------------------------------

@pytest.fixture
def caop_device(device_config):
    device = _device(device_config)
    if device == "gpu-omp":
        pytest.skip("upstream SBD's CAOP code has no OpenMP-offload path")
    return device


def test_upstream_boson_chain_matches_exact_diagonalization(caop_device):
    """Upstream's example: an 8-site hard-core boson chain over its 16-state basis."""
    ham, basis = CAOP_APP / "hamiltonian.txt", CAOP_APP / "basis.txt"
    expected = _exact(ham, _bitstrings(basis))
    assert _solve(caop_device, ham, basis, 8) == pytest.approx(expected, abs=1e-9)


def test_complex_amplitudes_give_the_real_energy(caop_device):
    """A real Hamiltonian solved with complex amplitudes gives the same energy."""
    ham, basis = CAOP_APP / "hamiltonian.txt", CAOP_APP / "basis.txt"
    assert _solve(caop_device, ham, basis, 8, complex_=True) == pytest.approx(
        _solve(caop_device, ham, basis, 8), abs=1e-9)


def test_hubbard_ring_matches_exact_diagonalization_and_gdb(caop_device, tmp_path):
    """Upstream's 4-site Hubbard ring: CAOP, an exact diagonalization, and GDB agree.

    GDB solves upstream's FCIDUMP of the same model over the same determinants. Its
    determinant bits interleave the spins, so each state is converted to that layout.
    """
    ham, basis, states = _hubbard(tmp_path)
    expected = _exact(ham, states)
    assert _solve(caop_device, ham, basis, 8) == pytest.approx(expected, abs=1e-9)

    sites = 4
    interleaved = []
    for s in states:
        bits = ["0"] * (2 * sites)
        for p in range(sites):
            if s >> p & 1:
                bits[2 * sites - 1 - 2 * p] = "1"
            if s >> (sites + p) & 1:
                bits[2 * sites - 1 - (2 * p + 1)] = "1"
        interleaved.append("".join(bits))
    gdb = r"""
import json, sys
import sbd
sbd.init(device=sys.argv[1])
dets = sbd.sort_bitarray_array(sbd.from_strings(sys.argv[3].split(","), 20, 8))
config = sbd.GDB_SBD(); config.eps, config.max_it, config.bit_length = 1e-12, 200, 20
print("RESULT " + json.dumps(sbd.gdb_diag(sbd.LoadFCIDump(sys.argv[2]), dets, config)["energy"]))
"""
    device = caop_device
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    done = subprocess.run([sys.executable, "-c", gdb, device, str(HUBBARD_FCIDUMP),
                           ",".join(interleaved)], capture_output=True, text=True,
                          timeout=600, env=env)
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    gdb_energy = json.loads([l for l in done.stdout.splitlines()
                             if l.startswith("RESULT ")][-1][len("RESULT "):])
    assert gdb_energy == pytest.approx(expected, abs=1e-9)


def test_the_offload_backend_is_refused():
    if "gpu-omp" not in sbd.available_backends():
        pytest.skip("the OpenMP-offload backend was not built")
    with pytest.raises(ValueError, match="no CAOP solver"):
        sbd.CAOP_SBD(device="gpu-omp")


def test_a_basis_is_required(backend):
    """The basis check comes before any solve, so it is safe in the suite's process."""
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    with pytest.raises(ValueError, match="at least one basis file"):
        sbd.caop_diag_from_files(CAOP_APP / "hamiltonian.txt", [], backend.CAOP_SBD())


# --- the driver --------------------------------------------------------------------

def _run_driver(device, *flags):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    return subprocess.run([sys.executable, "run_caop_diag.py", "--device", device, *flags],
                          cwd=REPO / "examples" / "caop", capture_output=True, text=True,
                          timeout=600, env=env)


def _driver_energy(stdout):
    import re

    return float(re.search(r"Ground state energy:\s*(-?\d+\.\d+)", stdout).group(1))


def test_the_driver_runs_upstreams_example(caop_device):
    """With no input flags the driver runs upstream's example, to the exact energy.

    The tolerance is upstream's 1e-4 by default, so this tightens it to compare.
    """
    done = _run_driver(caop_device, "--tolerance", "1e-12", "--iteration", "200")
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    expected = _exact(CAOP_APP / "hamiltonian.txt", _bitstrings(CAOP_APP / "basis.txt"))
    assert _driver_energy(done.stdout) == pytest.approx(expected, abs=1e-9)


def test_the_driver_passes_the_seed_and_heatbath_flags(caop_device):
    """The seed and heat-bath flags are accepted and reach SBD.

    A fixed seed is reported back by SBD, and a heat-bath carryover with a parent
    truncation that keeps nothing returns an empty basis where the default does not.
    """
    common = ("--tolerance", "1e-12", "--iteration", "200", "--carryover_type", "2",
              "--heatbath_batch_size", "1000", "--heatbath_parent_distribution", "0")
    seeded = _run_driver(caop_device, *common, "--seed", "1729")
    assert seeded.returncode == 0, f"{seeded.stdout[-3000:]}\n{seeded.stderr[-3000:]}"
    assert "seed: 1729 (user-specified)" in seeded.stdout
    kept = int(seeded.stdout.split("Carryover basis on this rank:")[1].split()[0])
    nothing = _run_driver(caop_device, *common, "--heatbath_truncation", "2.0")
    assert nothing.returncode == 0, f"{nothing.stdout[-3000:]}\n{nothing.stderr[-3000:]}"
    dropped = int(nothing.stdout.split("Carryover basis on this rank:")[1].split()[0])
    assert kept > 0 and dropped < kept


# --- what the driver says about the GPU --------------------------------------------

@pytest.mark.parametrize("method", [0, 1])
def test_the_driver_says_what_runs_on_the_gpu(method):
    """On gpu the driver says whether this method uses the GPU; on cpu it says nothing.

    Upstream's GPU kernel is for applying the Hamiltonian on the fly (methods 0, 2);
    the stored-Hamiltonian methods (1, 3) run on the CPU even on the Thrust backend.
    One method of each kind covers both notes.
    """
    if "gpu" not in sbd.available_backends():
        pytest.skip("the Thrust backend was not built")
    flags = ("--method", str(method), "--tolerance", "1e-8", "--iteration", "50")
    gpu = _run_driver("gpu", *flags)
    assert gpu.returncode == 0, f"{gpu.stdout[-3000:]}\n{gpu.stderr[-3000:]}"
    assert gpu.stdout.count("NOTE: --method") == 1
    if method == 1:
        assert "uses no GPU" in gpu.stdout and "--method 0" in gpu.stdout
    else:
        assert "applies the Hamiltonian on the GPU" in gpu.stdout
        assert "compare against --device cpu" in gpu.stdout
    cpu = _run_driver("cpu", *flags)
    assert cpu.returncode == 0 and "NOTE: --method" not in cpu.stdout


# --- what upstream does not check --------------------------------------------------
#
# The bindings refuse these before any work, so they run in this process: a refused
# call never reaches caop::diag, which is what fixes the determinant word count.

def _caop_config(backend, **fields):
    config = backend.CAOP_SBD()
    config.system_size, config.bit_length = 8, 20
    for name, value in fields.items():
        setattr(config, name, value)
    return config


@pytest.mark.parametrize("fields, match", [
    ({"method": 1, "max_iv": 10, "max_nb": 10}, "1 <= max_iv < max_nb"),
    ({"method": 0, "max_iv": 11, "max_nb": 10}, "1 <= max_iv < max_nb"),
    ({"method": 1, "max_iv": 0}, "1 <= max_iv < max_nb"),
    ({"method": 5}, "method must be 0-3"),
], ids=["iv-eq-nb", "iv-gt-nb", "iv-0", "method-5"])
def test_settings_upstream_gets_wrong_are_refused(backend, fields, match):
    """Davidson with numivec >= block returns a wrong energy upstream, or hangs above it
    or at 0; a method outside 0-3 silently runs a stored-Hamiltonian one."""
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    with pytest.raises(ValueError, match=match):
        sbd.caop_diag_from_files(CAOP_APP / "hamiltonian.txt", [CAOP_APP / "basis.txt"],
                                 _caop_config(backend, **fields))


# Upstream reads the Hamiltonian on one rank and sends the terms on, so a file it cannot
# read raises there and leaves the other ranks waiting. The bindings read it first. A
# mode beyond system_size is not checked upstream at all.
BAD_HAMFILES = {
    "complex-on-real": ("-1\n(0,1) cdag 1 c 0\n(0,-1) cdag 0 c 1\n", "complex=True"),
    "bad-coefficient": ("-1\nabc cdag 1 c 0\n", "cannot read the coefficient 'abc'"),
    "no-statistics": ("# only a comment\n", "no statistics line"),
    "mode-beyond-sites": ("1\n1.0 bdag 8 b 0\n1.0 bdag 0 b 8\n", "mode 8, beyond sites=8"),
}


@pytest.mark.parametrize("name", sorted(BAD_HAMFILES))
def test_a_hamiltonian_file_upstream_cannot_read_is_refused(backend, tmp_path, name):
    """Refused naming the line, and a (re,im) coefficient on a real solve says to pass
    complex=True rather than being read as its real part."""
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    text, match = BAD_HAMFILES[name]
    ham = tmp_path / "ham.txt"
    ham.write_text(text)
    with pytest.raises(ValueError, match=match.replace("(", r"\(").replace(")", r"\)")):
        sbd.caop_diag_from_files(ham, [CAOP_APP / "basis.txt"], _caop_config(backend))


def test_a_complex_hamiltonian_file_is_read_with_complex_amplitudes(backend, tmp_path):
    """The same (re,im) file solves with complex=True, to the exact energy."""
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    ham = tmp_path / "ham.txt"
    ham.write_text("-1\n(0,1) cdag 2 c 0\n(0,-1) cdag 0 c 2\n(-0.5,0) cdag 2 c 1\n"
                   "(-0.5,0) cdag 1 c 2\n(0.5,0) cdag 2 c 2\n")
    basis = tmp_path / "basis.txt"
    basis.write_text("001\n010\n100\n")
    config = _caop_config(backend, method=1, max_it=200, max_nb=3, eps=1e-12)
    config.system_size = 3
    energy = sbd.caop_diag_from_files(ham, [basis], config, complex=True)["energy"]
    assert energy == pytest.approx(_exact(ham, [1, 2, 4]), abs=1e-9)


# --- several ranks -----------------------------------------------------------------
#
# These run in the ranks of `tox -e mpi` (SBD_TEST_NPROCS, 2 by default). CAOP's basis
# shares GDB's process-wide word count, and every case here is one word, so they share
# the job's process with the TPB and GDB tests.

def _refusal_on_every_rank(backend, comm, match, **fields):
    """Each rank calls; every rank must refuse, or one would wait in a collective."""
    try:
        sbd.caop_diag_from_files(CAOP_APP / "hamiltonian.txt", [CAOP_APP / "basis.txt"],
                                 _caop_config(backend, **fields))
    except ValueError as e:
        outcome = str(e)
    else:
        outcome = "SOLVED"
    outcomes = comm.allgather(outcome)
    assert all(match in o for o in outcomes), outcomes


@pytest.fixture(scope="module")
def hubbard6():
    """A 6-site Hubbard ring at half filling, 400 states, in three uneven basis files.

    Written by rank 0 into one directory every rank reads, so the ranks share one basis.
    Uneven files give the redistribution something to even out.
    """
    import tempfile

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    path = pathlib.Path(comm.bcast(tempfile.mkdtemp(prefix="caop-mpi-")
                                   if comm.Get_rank() == 0 else None, root=0))
    if comm.Get_rank() == 0:
        ham, basis, _ = _hubbard(path, sites=6)
        rows = basis.read_text().split()
        for k, chunk in enumerate((rows[:50], rows[50:250], rows[250:])):
            (path / f"basis{k}.txt").write_text("\n".join(chunk) + "\n")
    comm.Barrier()
    rows = (path / "hubbard_basis.txt").read_text().split()
    files = [path / f"basis{k}.txt" for k in range(3)]
    # A fixed start, since the default start is random and every rank must agree.
    return path / "hubbard.txt", files, [int(r, 2) for r in rows], rows[0]


@pytest.mark.mpi
@pytest.mark.parametrize("method", [0, 1, 2, 3])
def test_the_basis_split_over_every_rank_gives_the_exact_energy(backend, caop_device,
                                                                 hubbard6, method):
    """The basis over all ranks (b_comm_size = ranks), each method, against NumPy."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    ham, files, states, start = hubbard6
    config = _caop_config(backend, method=method, max_it=300, max_nb=20, eps=1e-12,
                          do_redist_basis=True, b_comm_size=comm.Get_size(),
                          initial_determinant_bitstring=start)
    config.system_size = 12
    energy = sbd.caop_diag_from_files(ham, files, config)["energy"]
    energies = comm.allgather(energy)
    assert len(set(energies)) == 1, f"ranks disagree: {energies}"
    assert energy == pytest.approx(_exact(ham, states), abs=1e-9)


@pytest.mark.mpi
def test_a_grid_the_ranks_cannot_make_is_refused(backend):
    """A rank count that is not a multiple of t * b: upstream floors the helper size and
    the ranks disagree on the grid (a wrong energy), or, with fewer ranks than t * b,
    run a smaller grid than asked."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    _refusal_on_every_rank(backend, comm, "multiple of t_comm_size",
                           b_comm_size=comm.Get_size() + 1)


@pytest.mark.mpi
def test_t_above_b_is_refused_for_the_gpu_kernel(backend):
    """t > b gives task ranks no basis-ring station, which hangs upstream's GPU kernel.

    The CPU on-the-fly code guards an empty station list (mult.h:19); the Thrust one
    reads its first entry unguarded (mult_thrust.h:628).
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if not backend.__name__.endswith("_core_gpu_thrust"):
        pytest.skip("the gpu (Thrust) backend only")
    if comm.Get_size() < 2:
        pytest.skip("needs at least two ranks for t > b")
    _refusal_on_every_rank(backend, comm, "t_comm_size <= b_comm_size", method=0,
                           t_comm_size=comm.Get_size(), b_comm_size=1)


@pytest.mark.mpi
def test_a_hamiltonian_file_upstream_cannot_read_is_refused_on_every_rank(backend):
    """Upstream reads the file on one rank, so a bad one used to leave the others
    waiting for its terms: the job hung. Every rank now refuses together."""
    import tempfile

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if not getattr(backend, "__sbd_has_caop__", False):
        pytest.skip("this backend has no CAOP solver")
    path = pathlib.Path(comm.bcast(tempfile.mkdtemp(prefix="caop-bad-")
                                   if comm.Get_rank() == 0 else None, root=0))
    if comm.Get_rank() == 0:
        (path / "ham.txt").write_text(BAD_HAMFILES["complex-on-real"][0])
    comm.Barrier()
    try:
        sbd.caop_diag_from_files(path / "ham.txt", [CAOP_APP / "basis.txt"],
                                 _caop_config(backend, b_comm_size=comm.Get_size()))
    except ValueError as e:
        outcome = str(e)
    else:
        outcome = "SOLVED"
    outcomes = comm.allgather(outcome)
    assert all("complex=True" in o for o in outcomes), outcomes
