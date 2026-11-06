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

"""Unrestricted (UHF) FCIDUMPs: the layout check, the build choice, the energies.

A ``-D_UHF`` build reads integrals in SBD's spin-orbital layout and a restricted
one in the spatial layout. The file syntax is the same, so a file read by the
wrong build loads and diagonalizes without complaint and returns a wrong energy.
These tests pin the three things standing between a user and that: the layout
classification, the choice of build from it, and the guard in each binding.

The data is the H3 doublet in ``test/data/uhf`` (``examples/tools/make_uhf_fcidump.py``
wrote it): one geometry, two FCIDUMPs, UHF orbitals in the unrestricted layout and
ROHF orbitals in the restricted one. The full-space energy does not depend on the
orbitals, so both must give the same exact number -- which a misread layout would
not. The single Hartree-Fock determinant is the sharper check: in UHF orbitals its
energy is the UHF SCF energy, in ROHF orbitals the ROHF one.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import sbd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "test" / "data" / "uhf"
UHF = DATA / "h3_uhf.fcidump"
ROHF = DATA / "h3_rohf.fcidump"
REFERENCE = json.loads((DATA / "reference.json").read_text())
NORB = REFERENCE["norb"]

# One word per determinant either way (3 orbitals, 6 spin orbitals), the width the
# rest of the suite uses, since det_vector fixes it process-wide on first use.
TPB_BITS = 20
GDB_BITS = 20

LAYOUTS = {"uhf": UHF, "rohf": ROHF}


def _strings(name):
    return [s.strip() for s in (DATA / f"h3_{name}.txt").read_text().split() if s.strip()]


ALPHA, BETA = _strings("alpha"), _strings("beta")


@pytest.fixture(scope="module")
def device(backend):
    """The device under test, which must have an unrestricted build to run these."""
    device = sbd.get_device()
    if not sbd.has_uhf(device):
        pytest.skip(f"no unrestricted build of {device!r} (gpu-omp has none; "
                    f"or the package was built with SBD_BUILD_UHF=0)")
    return device


def _tpb(**overrides):
    config = sbd.TPB_SBD()
    config.eps, config.max_it, config.max_nb, config.bit_length = 1e-10, 100, 10, TPB_BITS
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _gdb():
    config = sbd.GDB_SBD()
    config.eps, config.max_it, config.max_nb, config.bit_length = 1e-10, 100, 10, GDB_BITS
    return config


def _half(strings):
    return sbd.sort_bitarray([sbd.from_string(s, TPB_BITS, NORB) for s in strings])


def _interleave(alpha, beta):
    """GDB's packing: bit 2i is alpha orbital i, bit 2i+1 beta, rightmost first."""
    a, b = alpha[::-1], beta[::-1]
    return "".join(a[i] + b[i] for i in range(len(a)))[::-1]


def _full_space():
    full = [_interleave(a, b) for a, b in itertools.product(ALPHA, BETA)]
    return sbd.from_strings(full, GDB_BITS, 2 * NORB)


def _hartree_fock(n_electrons):
    return "0" * (NORB - n_electrons) + "1" * n_electrons


# --- classification -------------------------------------------------------------


@pytest.mark.parametrize("layout, path", [("uhf", UHF), ("restricted", ROHF)])
def test_layout_is_read_from_the_indices(device, layout, path):
    assert sbd.fcidump_format(path) == layout
    assert sbd.fcidump_format(sbd.LoadFCIDump(path)) == layout


def test_loaded_object_is_the_plain_fcidump(device):
    """Nothing is stored on the object: it is what the backend's parser returns."""
    fcidump = sbd.LoadFCIDump(UHF)
    assert not hasattr(fcidump, "uhf")
    with pytest.raises(AttributeError):
        fcidump.uhf = True
    assert sorted(fcidump.header) == ["ISYM", "MS2", "NELEC", "NORB", "ORBSYM"]


@pytest.mark.parametrize("path, stated", [(UHF, False), (ROHF, True)])
def test_a_stated_layout_must_match_the_file(device, path, stated):
    with pytest.raises(ValueError, match=f"uhf={stated} was given"):
        sbd.LoadFCIDump(path, uhf=stated)
    sbd.LoadFCIDump(path, uhf=not stated)


@pytest.mark.parametrize("edit, message", [
    (("NORB=3", "NORB=1"), "exceeds 2\\*NORB"),        # index 3 > 2
    (("NORB=3", "NORB=2"), "mixes an odd"),             # fits 2*NORB, but spins mixed
    (("ISYM=1,", "ISYM=1,IUHF=1,"), "header sets IUHF"),
])
def test_a_file_in_neither_layout_is_refused(device, tmp_path, edit, message):
    bad = tmp_path / "bad.fcidump"
    bad.write_text(ROHF.read_text().replace(*edit))
    with pytest.raises(ValueError, match=message):
        sbd.fcidump_format(bad)
    with pytest.raises(ValueError, match=message):
        sbd.LoadFCIDump(bad)


# --- energies --------------------------------------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
def test_tpb_full_space_is_exact(device, layout):
    fcidump = sbd.LoadFCIDump(LAYOUTS[layout])
    energy = sbd.tpb_diag(fcidump, _half(ALPHA), _half(BETA), _tpb())["energy"]
    assert energy == pytest.approx(REFERENCE["e_fci"], abs=1e-9)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_gdb_full_space_is_exact(device, layout):
    fcidump = sbd.LoadFCIDump(LAYOUTS[layout])
    energy = sbd.gdb_diag(fcidump, _full_space(), _gdb())["energy"]
    assert energy == pytest.approx(REFERENCE["e_fci"], abs=1e-9)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_hartree_fock_determinant_gives_the_scf_energy(device, layout):
    """One determinant, no diagonalization to hide behind: the integrals alone."""
    n_alpha, n_beta = REFERENCE["nelec"]
    fcidump = sbd.LoadFCIDump(LAYOUTS[layout])
    energy = sbd.tpb_diag(fcidump, _half([_hartree_fock(n_alpha)]),
                          _half([_hartree_fock(n_beta)]), _tpb())["energy"]
    assert energy == pytest.approx(REFERENCE[f"e_{layout}_scf"], abs=1e-9)


def test_integrals_edited_after_loading_are_classified_afresh(device):
    """The layout is computed at diagonalization time, so it cannot go stale."""
    fcidump = sbd.LoadFCIDump(ROHF)
    fcidump.integrals = sbd.LoadFCIDump(UHF).integrals
    assert sbd.fcidump_format(fcidump) == "uhf"
    n_alpha, n_beta = REFERENCE["nelec"]
    energy = sbd.tpb_diag(fcidump, _half([_hartree_fock(n_alpha)]),
                          _half([_hartree_fock(n_beta)]), _tpb())["energy"]
    assert energy == pytest.approx(REFERENCE["e_uhf_scf"], abs=1e-9)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_file_entry_point_picks_the_same_build(device, tmp_path, layout):
    """tpb_diag_from_files reads the FCIDUMP itself, on rank 0; same subspace, same energy.

    It derives beta from the alpha file, which for this doublet is not the physical
    space -- the point is only that both entry points read the integrals alike.
    """
    alpha_file = tmp_path / "alpha.txt"
    alpha_file.write_text("\n".join(ALPHA) + "\n")
    from_file = sbd.tpb_diag_from_files(str(LAYOUTS[layout]), str(alpha_file), _tpb())
    in_memory = sbd.tpb_diag(sbd.LoadFCIDump(LAYOUTS[layout]), _half(ALPHA), _half(ALPHA), _tpb())
    assert from_file["energy"] == pytest.approx(in_memory["energy"], abs=1e-12)


# --- the build choice and its guards ---------------------------------------------


def test_the_unrestricted_builds_are_not_devices(device):
    assert not any(d.endswith("-uhf") for d in sbd.available_backends())
    assert not any(d.endswith("-uhf") for d in sbd.backend_load_errors())
    with pytest.raises(ValueError, match="is not a device name"):
        sbd.get_backend(f"{device}-uhf")


def test_gpu_omp_has_no_unrestricted_build():
    assert not sbd.has_uhf("gpu-omp")
    # Raises before importing anything: loading the OMP-offload module next to the
    # CPU one would demote offload for the rest of the process.
    with pytest.raises(ValueError, match="No UHF build of the 'gpu-omp' backend"):
        sbd.get_backend("gpu-omp", uhf=True)


def _direct_calls(module, fcidump, path, alpha_file):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return {
        "tpb_diag": lambda: module.tpb_diag(comm, _tpb(), fcidump, _half(ALPHA), _half(BETA), "", ""),
        "gdb_diag": lambda: module.gdb_diag(comm, _gdb(), fcidump, _full_space(), "", ""),
        "tpb_diag_from_files": lambda: module.tpb_diag_from_files(
            comm, _tpb(), str(path), str(alpha_file), "", ""),
    }


@pytest.mark.parametrize("entry", ["tpb_diag", "gdb_diag", "tpb_diag_from_files"])
@pytest.mark.parametrize("module_uhf, path", [(False, UHF), (True, ROHF)])
def test_a_module_called_directly_refuses_the_other_layout(device, tmp_path, entry,
                                                           module_uhf, path):
    """The binding guards itself, so bypassing sbd.tpb_diag cannot misread a file."""
    module = sbd.get_backend(device, uhf=module_uhf)
    assert module.__sbd_uhf__ is module_uhf
    alpha_file = tmp_path / "alpha.txt"
    alpha_file.write_text("\n".join(ALPHA) + "\n")
    call = _direct_calls(module, sbd.LoadFCIDump(path), path, alpha_file)[entry]
    with pytest.raises(ValueError, match="would read it as"):
        call()


# --- the TPB driver's electron-count check ------------------------------------------


def test_the_tpb_driver_refuses_symmetric_mode_on_an_open_shell(device):
    """Symmetric mode reuses the alpha list as beta: n_beta electrons would be wrong."""
    completed = subprocess.run(
        [sys.executable, "run_sbd_diag.py", "--device", device,
         "--fcidump", str(UHF), "--adetfile", str(DATA / "h3_alpha.txt")],
        cwd=REPO_ROOT / "examples" / "tpb", capture_output=True, text=True, timeout=300)
    assert completed.returncode != 0
    assert "1 beta electrons, but the beta determinants" in completed.stdout + completed.stderr


def test_the_tpb_driver_runs_an_unrestricted_file(device, tmp_path):
    """The energy is exact. With --rdm_output the spin-summed RDMs are saved, as upstream
    writes them, but no natural-orbital occupations are printed: rdm1 sums alpha and beta
    blocks given in different orbitals, so its eigenvalues are not occupations."""
    completed = subprocess.run(
        [sys.executable, "run_sbd_diag.py", "--device", device, "--symmetrize_spin", "0",
         "--tolerance", "1e-10", "--fcidump", str(UHF),
         "--adetfile", str(DATA / "h3_alpha.txt"), "--bdetfile", str(DATA / "h3_beta.txt"),
         "--rdm_output", str(tmp_path / "rdms.npz")],
        cwd=REPO_ROOT / "examples" / "tpb", capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    energy = float(completed.stdout.split("Ground state energy:")[1].split()[0])
    assert energy == pytest.approx(REFERENCE["e_fci"], abs=1e-8)
    assert "not printed for a UHF FCIDUMP" in completed.stdout
    assert "Natural orbital occupations (sorted)" not in completed.stdout
    rdm1 = np.load(tmp_path / "rdms.npz")["rdm1"]
    assert np.trace(rdm1) == pytest.approx(sum(REFERENCE["nelec"]), abs=1e-8)


# --- multi-rank ------------------------------------------------------------------


@pytest.mark.mpi
def test_every_rank_refuses_together(device, tmp_path_factory):
    """A refusal must reach every rank; one left in a collective would hang the job."""
    from mpi4py import MPI
    module = sbd.get_backend(device, uhf=False)
    fcidump = sbd.LoadFCIDump(UHF)
    refused = 0
    try:
        module.tpb_diag(MPI.COMM_WORLD, _tpb(), fcidump, _half(ALPHA), _half(BETA), "", "")
    except ValueError:
        refused = 1
    assert MPI.COMM_WORLD.allreduce(refused) == MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi
@pytest.mark.parametrize("layout", LAYOUTS)
def test_sharded_gdb_is_exact(device, layout):
    """The basis split over b_comm: each rank passes its own contiguous slice.

    b = the rank count keeps the helper dimension at 1, which Thrust requires. The
    slices are kept as given ('input'): the default equal-bra-a placement deals out
    whole alpha strings, and this space has only three, so from four ranks on it would
    leave a shard empty, which the Thrust solve refuses.
    """
    size, rank = sbd.get_world_size(), sbd.get_rank()
    full = sbd.sort_bitarray_array(_full_space())
    quotient, remainder = divmod(len(full), size)
    begin = rank * quotient + min(rank, remainder)
    shard = full[begin:begin + quotient + (1 if rank < remainder else 0)]
    config = _gdb()
    config.b_comm_size = size
    if size > len(full):                 # the same on every rank, so no rank is left behind
        pytest.skip("more ranks than determinants")
    result = sbd.gdb_diag(sbd.LoadFCIDump(LAYOUTS[layout]), shard, config,
                          determinant_distribution="input")
    assert result["global_dim"] == len(full)
    assert result["energy"] == pytest.approx(REFERENCE["e_fci"], abs=1e-9)


@pytest.mark.mpi
def test_unrestricted_tpb_is_exact_under_mpi(device):
    fcidump = sbd.LoadFCIDump(UHF)
    energy = sbd.tpb_diag(fcidump, _half(ALPHA), _half(BETA), _tpb())["energy"]
    assert energy == pytest.approx(REFERENCE["e_fci"], abs=1e-9)
