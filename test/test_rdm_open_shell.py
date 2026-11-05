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

"""The opposite-spin block of the 2-RDM on an open-shell state, against PySCF.

Upstream SBD fixed a sign in opposite-spin double excitations of the 2-RDM
(r-ccs-cms/sbd#98), shown there on an asymmetric H3 doublet. The error cancels in
the spin-summed 2-RDM but not in the ``ab`` block of ``two_p_rdm``, so this checks
that block, on the same molecule (here in ROHF orbitals), over the full space,
against PySCF's ``make_rdm12s``. Three orbitals is one word per determinant, which SBD
fixes for the life of a process, so the solve runs in a **subprocess**.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pyscf", reason="pyscf builds the Hamiltonian and the reference")

from pyscf import ao2mo, fci, gto, scf  # noqa: E402
from pyscf.fci import cistring  # noqa: E402
from pyscf.tools import fcidump  # noqa: E402

# One word per string here, as in the rest of the suite's TPB tests.
BIT_LENGTH = 20

SOLVE = r"""
import json, sys
import sbd
a = json.loads(sys.argv[1])
sbd.init(device=a["device"])
def strings(occupations):
    return sbd.sort_bitarray([sbd.from_string(s, a["bits"], a["norb"]) for s in occupations])
config = sbd.TPB_SBD()
config.eps, config.max_it, config.bit_length, config.do_rdm = 1e-12, 300, a["bits"], 1
result = sbd.tpb_diag(sbd.LoadFCIDump(a["fcidump"]), strings(a["alpha"]), strings(a["beta"]),
                      config)
print("RESULT " + json.dumps({"energy": result["energy"],
                              "two_p_rdm": [list(b) for b in result["two_p_rdm"]]}))
"""


def test_opposite_spin_2rdm_block_matches_pyscf(device_config, tmp_path):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.8; H 0 0 2.2", basis="sto-3g", spin=1, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    c, norb, (na, nb) = mf.mo_coeff, mol.nao, mol.nelec
    h1 = c.T @ scf.hf.get_hcore(mol) @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), norb)
    path = tmp_path / "fcidump.txt"
    fcidump.from_integrals(str(path), h1, eri, norb, na + nb, nuc=0.0, ms=na - nb)

    def strings(n):
        return [format(int(s), f"0{norb}b") for s in cistring.make_strings(range(norb), n)]

    args = {"device": device_config.device if device_config is not None else "cpu",
            "fcidump": str(path), "bits": BIT_LENGTH, "norb": norb,
            "alpha": strings(na), "beta": strings(nb)}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    done = subprocess.run([sys.executable, "-c", SOLVE, json.dumps(args)],
                          capture_output=True, text=True, timeout=300, env=env)
    assert done.returncode == 0, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    result = json.loads([l for l in done.stdout.splitlines()
                         if l.startswith("RESULT ")][-1][len("RESULT "):])

    energy, vector = fci.direct_spin1.kernel(h1, eri, norb, (na, nb), tol=1e-14)
    assert result["energy"] == pytest.approx(energy, abs=1e-9)
    (_, _), (rdm_aa, rdm_ab, rdm_bb) = fci.direct_spin1.make_rdm12s(vector, norb, (na, nb))
    # two_p_rdm[s + 2t] holds <c+_{i,s} c+_{j,t} c_{l,t} c_{k,s}> at i + L j + L^2 k + L^3 l;
    # reshaped in Fortran order and transposed (0, 2, 1, 3), it is PySCF's p q r s order.
    blocks = np.asarray(result["two_p_rdm"])
    block = lambda s: np.reshape(blocks[s], (norb,) * 4, order="F").transpose(0, 2, 1, 3)
    np.testing.assert_allclose(block(0), rdm_aa, atol=1e-9)
    np.testing.assert_allclose(block(2), rdm_ab, atol=1e-9)     # the block #98 fixed
    np.testing.assert_allclose(block(3), rdm_bb, atol=1e-9)
