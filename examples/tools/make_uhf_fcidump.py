#!/usr/bin/env python3

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

"""
Write unrestricted (UHF) FCIDUMPs in SBD's layout, with a restricted twin and
exact references.

    python make_uhf_fcidump.py nh2 --outdir nh2                   # an example to run on
    python make_uhf_fcidump.py h3  --outdir ../../test/data/uhf   # the unit tests' data

For each system this writes, from one PySCF run:

    <name>_uhf.fcidump    UHF orbitals, SBD's unrestricted layout
    <name>_rohf.fcidump   ROHF orbitals, the ordinary restricted layout
    <name>_alpha.txt      every alpha string (the full space), for the TPB driver
    <name>_beta.txt       every beta string
    reference.json        the exact (FCI) energy and how it was obtained

The full-space energy does not depend on the orbitals, so both FCIDUMPs must
give the same number. That is the point of the pair: a UHF file read with the
wrong layout gives a different one.

SBD's unrestricted layout (what a ``-D_UHF`` build reads, ``makeintegrals.h``):
``NORB`` stays the number of spatial orbitals, and integrals are indexed by
interleaved 1-based spin orbitals, alpha of spatial orbital ``p`` (0-based) at
``2p+1`` and beta at ``2p+2``. One-body lines are ``h i j 0 0``. Two-body lines
are chemist-notation ``(ij|kl)`` with ``i, j`` of one spin and ``k, l`` of one
spin, each unique integral written once: SBD fills the permutations itself, and
the ``(alpha alpha|beta beta)`` block fills ``(beta beta|alpha alpha)``.

Requires PySCF. The unit tests' H3 files are committed rather than regenerated in
CI: a UHF calculation can land on a different solution across PySCF versions,
which would change the integrals (though not the FCI energy).

To write integrals of your own, import ``write_uhf_fcidump`` from this script.
"""

import argparse
import itertools
import json
from pathlib import Path

SYSTEMS = {
    # Upstream's asymmetric H3 doublet: 3 orbitals, 9 determinants. Unit tests.
    'h3': dict(atom='H 0 0 0; H 0 0 0.8; H 0 0 2.2', basis='sto-3g', spin=1),
    # NH2 radical: 13 orbitals, 920,205 determinants. Large enough to be a real
    # run, small enough for a laptop.
    'nh2': dict(atom='N 0 0 0; H 0 0.80 0.60; H 0 -0.80 0.60', basis='6-31g', spin=1),
}
TOL = 1e-14


def _pairs(norb):
    return [(p, q) for p in range(norb) for q in range(p + 1)]       # p >= q


def _compound(p, q):
    return p * (p + 1) // 2 + q


def _header(handle, norb, nelec, ms2):
    handle.write(f" &FCI NORB={norb},NELEC={nelec},MS2={ms2},\n")
    handle.write("  ORBSYM=" + "1," * norb + "\n  ISYM=1,\n &END\n")


def write_uhf_fcidump(path, h1, eri, norb, nelec, ms2, ecore):
    """Write spin-resolved integrals as an FCIDUMP in SBD's unrestricted layout.

    Args:
        path: Output file.
        h1: ``(h1_alpha, h1_beta)``, each ``(norb, norb)``.
        eri: ``(eri_aa, eri_ab, eri_bb)``, each ``(norb,)*4`` in chemist notation,
            ``eri_ab[p, q, r, s] = (p_alpha q_alpha | r_beta s_beta)``.
        norb: Spatial orbitals.
        nelec: Total electrons.
        ms2: ``n_alpha - n_beta``.
        ecore: Constant energy (nuclear repulsion plus any frozen core).

    Returns:
        The number of integral lines written.
    """
    alpha, beta = (lambda p: 2 * p + 1), (lambda p: 2 * p + 2)
    blocks = ((eri[0], alpha, alpha), (eri[2], beta, beta), (eri[1], alpha, beta))
    lines = 0
    with open(path, 'w') as handle:
        _header(handle, norb, nelec, ms2)
        for block, (g, s1, s2) in enumerate(blocks):
            same_spin = block < 2
            for (i, j), (k, l) in itertools.product(_pairs(norb), _pairs(norb)):
                if same_spin and _compound(i, j) < _compound(k, l):
                    continue                              # same spin: ij >= kl only
                if abs(g[i, j, k, l]) > TOL:
                    handle.write(f" {g[i, j, k, l]: .16e} {s1(i)} {s1(j)} {s2(k)} {s2(l)}\n")
                    lines += 1
        for h, s in ((h1[0], alpha), (h1[1], beta)):
            for i, j in _pairs(norb):
                if abs(h[i, j]) > TOL:
                    handle.write(f" {h[i, j]: .16e} {s(i)} {s(j)} 0 0\n")
                    lines += 1
        handle.write(f" {ecore: .16e} 0 0 0 0\n")
    return lines


def write_restricted_fcidump(path, h1, eri, norb, nelec, ms2, ecore):
    """The ordinary restricted FCIDUMP, for comparison with the unrestricted one."""
    with open(path, 'w') as handle:
        _header(handle, norb, nelec, ms2)
        for (i, j), (k, l) in itertools.product(_pairs(norb), _pairs(norb)):
            if _compound(i, j) >= _compound(k, l) and abs(eri[i, j, k, l]) > TOL:
                handle.write(f" {eri[i, j, k, l]: .16e} {i + 1} {j + 1} {k + 1} {l + 1}\n")
        for i, j in _pairs(norb):
            if abs(h1[i, j]) > TOL:
                handle.write(f" {h1[i, j]: .16e} {i + 1} {j + 1} 0 0\n")
        handle.write(f" {ecore: .16e} 0 0 0 0\n")


def write_strings(path, norb, n_electrons):
    """Every string of n_electrons in norb orbitals; the rightmost bit is orbital 1."""
    with open(path, 'w') as handle:
        for occ in itertools.combinations(range(norb), n_electrons):
            handle.write(format(sum(1 << o for o in occ), f'0{norb}b') + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('system', choices=sorted(SYSTEMS))
    parser.add_argument('--outdir', default='.')
    args = parser.parse_args()

    import pyscf
    from pyscf import ao2mo, fci, gto, scf

    spec = SYSTEMS[args.system]
    mol = gto.M(verbose=0, **spec)
    norb, (na, nb) = mol.nao, mol.nelec
    nelec, ms2, ecore = na + nb, na - nb, mol.energy_nuc()
    hcore = scf.hf.get_hcore(mol)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    name = args.system

    uhf = scf.UHF(mol)
    uhf.conv_tol = 1e-12
    uhf.run()
    ca, cb = uhf.mo_coeff
    h1u = (ca.T @ hcore @ ca, cb.T @ hcore @ cb)
    eriu = tuple(ao2mo.general(mol, c, compact=False).reshape([norb] * 4)
                 for c in ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb)))
    lines = write_uhf_fcidump(out / f'{name}_uhf.fcidump', h1u, eriu, norb, nelec, ms2, ecore)
    e_uhf_fci = fci.direct_uhf.FCI().kernel(h1u, eriu, norb, (na, nb), tol=1e-12)[0]

    rohf = scf.ROHF(mol)
    rohf.conv_tol = 1e-12
    rohf.run()
    c = rohf.mo_coeff
    h1r = c.T @ hcore @ c
    erir = ao2mo.restore(1, ao2mo.kernel(mol, c), norb)
    write_restricted_fcidump(out / f'{name}_rohf.fcidump', h1r, erir, norb, nelec, ms2, ecore)
    e_rohf_fci = fci.direct_spin1.FCI().kernel(h1r, erir, norb, (na, nb), tol=1e-12)[0]

    write_strings(out / f'{name}_alpha.txt', norb, na)
    write_strings(out / f'{name}_beta.txt', norb, nb)
    n_alpha_strings = sum(1 for _ in itertools.combinations(range(norb), na))
    n_beta_strings = sum(1 for _ in itertools.combinations(range(norb), nb))

    e_fci = e_uhf_fci + ecore
    if abs(e_uhf_fci - e_rohf_fci) > 1e-8:
        raise SystemExit(f"FCI in UHF and ROHF orbitals disagree: {e_uhf_fci} vs {e_rohf_fci}")
    reference = {
        'system': name, **spec,
        'norb': norb, 'nelec': [na, nb], 'ms2': ms2,
        'determinants': n_alpha_strings * n_beta_strings,
        'e_uhf_scf': uhf.e_tot, 'uhf_s2': float(uhf.spin_square()[0]),
        'e_rohf_scf': rohf.e_tot,
        'e_fci': e_fci,
        'note': 'e_fci is PySCF FCI (direct_uhf in the UHF orbitals; direct_spin1 in the '
                'ROHF ones agrees) including ecore. Both FCIDUMPs must reproduce it.',
        'pyscf_version': pyscf.__version__,
    }
    (out / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')

    print(f"{name}: norb={norb} nelec=({na},{nb}) determinants="
          f"{n_alpha_strings}x{n_beta_strings}={reference['determinants']:,}")
    print(f"  UHF <S^2> = {reference['uhf_s2']:.6f}   FCI E = {e_fci:.12f}")
    print(f"  wrote {name}_uhf.fcidump ({lines:,} integral lines), {name}_rohf.fcidump, "
          f"{name}_alpha.txt, {name}_beta.txt, reference.json in {out}/")


if __name__ == '__main__':
    main()
