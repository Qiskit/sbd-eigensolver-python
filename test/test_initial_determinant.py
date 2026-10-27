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

"""Starting Davidson on a chosen determinant, where SBD's default start fails.

SBD starts Davidson, by default, from the first determinant of the subspace in
sorted order. Davidson improves that vector by applying the Hamiltonian, so it can
only reach states the start overlaps with. The Hamiltonian conserves spatial
symmetry, so a start in a symmetry sector the ground state does not touch converges,
without any error, on the lowest state of that sector instead.

The case here is built to do exactly that. C2v water in 6-31G, with the
symmetry-forbidden integrals set to exactly zero (PySCF leaves them at 1e-15, enough
for Davidson to leak back to the ground state over many restarts). The orbitals are
relabeled in reverse, so the lowest-sorting string is made of the highest orbitals.
The subspace is HF and its single excitations, for alpha and beta, plus one extra
alpha string (decoy) that sorts first, has all five electrons in virtual orbitals
and lies in a different sector. The default start is (decoy, first beta): Davidson
converges 27 Ha above the ground state. Starting on (HF, HF) instead gives PySCF's
energy for the same subspace.

Each backend runs in a **subprocess**, since ``det_vector``'s word count is fixed
for the life of a process and the CPU and OpenMP-offload modules must not share one.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import sbd

SCRIPT = r"""
import itertools, json, sys
import numpy as np
from pyscf import ao2mo, gto, scf, symm
from pyscf.tools import fcidump
from qiskit_addon_sqd import fermion
import sbd
from sbd.device_config import DeviceConfig
from sbd.sbd_solver import solve_sci

device, workdir = sys.argv[1], sys.argv[2]
sbd.init(device=device)
mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="6-31g", symmetry=True, verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-12)
c = mf.mo_coeff
n = c.shape[1]
# C2v irreps as two parity bits, so a product of irreps is an XOR.
code = {"A1": 0, "A2": 3, "B1": 1, "B2": 2}
irr = np.array([code[x] for x in symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c)])
h1 = c.T @ scf.hf.get_hcore(mol) @ c
eri = ao2mo.restore(1, ao2mo.kernel(mol, c), n)
enuc = mol.energy_nuc()
h1[(irr[:, None] ^ irr[None, :]) != 0] = 0.0
eri[(irr[:, None, None, None] ^ irr[None, :, None, None]
     ^ irr[None, None, :, None] ^ irr[None, None, None, :]) != 0] = 0.0
perm = np.arange(n)[::-1]
h1 = h1[np.ix_(perm, perm)]
eri = eri[np.ix_(perm, perm, perm, perm)]
irr = irr[perm]

def sym(s):
    x = 0
    for p in range(n):
        if (s >> p) & 1:
            x ^= int(irr[p])
    return x

hf = sum(1 << (n - 1 - o) for o in range(5))
strs = sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(n), 5))
good = [s for s in strs if bin(s & ~hf).count("1") <= 1]
decoy = 47
sa = np.array(sorted(set(good) | {decoy}))
sb = np.array(good)

ref = fermion.solve_sci((sa, sb), h1, eri, norb=n, nelec=(5, 5), tol=1e-12)
amps = ref.sci_state.amplitudes
default_sector = sym(int(sa[0])) ^ sym(int(sb[0]))
weight = sum(float(abs(amps[i, j]) ** 2) for i, a in enumerate(sa) for j, b in enumerate(sb)
             if sym(int(a)) ^ sym(int(b)) == default_sector)

cfg = {"eps": 1e-10, "max_it": 300, "bit_length": 20}
dc = DeviceConfig(device=device)
tpb_default = solve_sci((sa, sb), h1, eri, n, (5, 5), sbd_config=cfg, device_config=dc).energy
tpb_explicit = solve_sci((sa, sb), h1, eri, n, (5, 5), sbd_config=cfg, device_config=dc,
                         initial_determinant=(hf, hf)).energy
try:
    solve_sci((sa, sb), h1, eri, n, (5, 5), sbd_config=cfg, device_config=dc,
              initial_determinant=(hf, decoy))
    tpb_missing = None
except ValueError as exc:
    tpb_missing = str(exc)

# GDB: the same subspace as a list of interleaved full determinants.
def full(a, b):
    bits = ["0"] * (2 * n)
    for p in range(n):
        if (a >> p) & 1:
            bits[2 * n - 1 - 2 * p] = "1"
        if (b >> p) & 1:
            bits[2 * n - 1 - (2 * p + 1)] = "1"
    return "".join(bits)

path = workdir + "/fcidump.txt"
fcidump.from_integrals(path, h1, eri, n, (5, 5), nuc=0.0)
backend = sbd.get_backend(device)
# The same bit_length as the TPB solves above: SBD fixes the number of words per
# determinant for the life of the process, and 26 bits at 20 per word is 2 words
# for both.
det = sbd.from_strings(sorted({full(int(a), int(b)) for a in sa for b in sb},
                              key=lambda s: int(s, 2)), 20, 2 * n)
gdb = {}
for label, start in (("default", ""), ("explicit", full(hf, hf)), ("missing", full(hf, decoy))):
    conf = backend.GDB_SBD()
    conf.eps, conf.max_it, conf.bit_length = 1e-10, 300, 20
    conf.initial_determinant_bitstring = start
    try:
        gdb[label] = sbd.gdb_diag(backend.LoadFCIDump(path), det, conf)["energy"]
    except ValueError as exc:
        gdb[label] = str(exc)

print("RESULT", json.dumps({
    "backend": backend.__name__,
    "reference": ref.energy,
    "default_sector_weight": weight,
    "tpb_default": tpb_default,
    "tpb_explicit": tpb_explicit,
    "tpb_missing": tpb_missing,
    "gdb_default": gdb["default"],
    "gdb_explicit": gdb["explicit"],
    "gdb_missing": gdb["missing"],
}))
"""


@pytest.mark.parametrize("device", ["cpu", "gpu", "gpu-omp"])
def test_explicit_start_finds_the_state_the_default_start_misses(device, tmp_path):
    """The default start lands 27 Ha high on this subspace; (HF, HF) gives PySCF's energy."""
    pytest.importorskip("pyscf", reason="pyscf builds the Hamiltonian and the reference")
    pytest.importorskip("qiskit_addon_sqd.fermion", reason="qiskit-addon-sqd provides solve_sci")
    available = sbd.available_backends()
    if device not in available:
        pytest.skip(f"backend {device!r} was not built; available: {available}")

    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, device, str(tmp_path)],
        capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode == 0, (
        f"subprocess failed:\n{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
    )
    line = [l for l in completed.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, f"no result reported:\n{completed.stdout[-3000:]}"
    r = json.loads(line[-1][len("RESULT "):])

    # The case does what it claims: the default start is in a sector the ground
    # state does not touch, so it cannot converge on the ground state.
    assert r["default_sector_weight"] < 1e-12
    assert r["tpb_default"] > r["reference"] + 1.0
    assert r["gdb_default"] > r["reference"] + 1.0

    # The explicit start does.
    assert r["tpb_explicit"] == pytest.approx(r["reference"], abs=1e-8)
    assert r["gdb_explicit"] == pytest.approx(r["reference"], abs=1e-8)

    # A start that is not in the subspace fails rather than falling back.
    assert r["tpb_missing"] is not None and "not in this subspace" in r["tpb_missing"]
    assert isinstance(r["gdb_missing"], str) and "not present" in r["gdb_missing"]


def test_initial_determinant_is_checked_before_the_solve():
    """A string outside the subspace, or too wide for norb, raises with the string named."""
    import numpy as np

    from sbd.sbd_solver import _set_initial_determinant

    class Config:
        initial_adeterminant_bitstring = ""
        initial_bdeterminant_bitstring = ""

    strings = np.array([0b0011, 0b0101, 0b0110])
    config = Config()
    _set_initial_determinant(config, (0b0101, 0b0011), strings, strings, 4)
    assert config.initial_adeterminant_bitstring == "0101"
    assert config.initial_bdeterminant_bitstring == "0011"

    with pytest.raises(ValueError, match=r"beta string 9 \(1001\) is not in"):
        _set_initial_determinant(Config(), (0b0101, 0b1001), strings, strings, 4)
    with pytest.raises(ValueError, match="does not fit 4 orbitals"):
        _set_initial_determinant(Config(), (1 << 4, 0b0011), strings, strings, 4)
