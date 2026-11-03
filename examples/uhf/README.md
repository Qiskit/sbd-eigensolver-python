# Unrestricted (UHF) examples

> A UHF FCIDUMP goes through the same solve and check steps as a restricted one:
> [`../tpb/`](../tpb/README.md) or [`../gdb/`](../gdb/README.md), then
> [`../analysis/`](../analysis/README.md).

SBD diagonalizes unrestricted Hamiltonians — separate alpha and beta orbitals, as
from a UHF calculation — through the same `sbd.tpb_diag` / `sbd.gdb_diag` calls and
the same drivers as restricted ones. Nothing selects it: the package reads the
FCIDUMP's layout and uses the matching build.

Upstream ships no unrestricted data, so this directory provides some, with the
script that wrote it.

## The layout

Upstream SBD reads unrestricted integrals only when compiled with `-D_UHF`, and only
in its own layout:

- `NORB` stays the number of **spatial** orbitals.
- Integrals are indexed by interleaved, 1-based **spin** orbitals: alpha of spatial
  orbital `p` (0-based) is `2p+1`, beta is `2p+2`. Indices therefore run to `2*NORB`.
- One-body lines are `h i j 0 0`. Two-body lines are chemist-notation `(ij|kl)`, with
  `i, j` of one spin and `k, l` of one spin, each unique integral written once.

A restricted FCIDUMP (RHF, ROHF, or any single orbital set) indexes spatial orbitals,
up to `NORB`. So the largest index tells the two apart, and `sbd.fcidump_format()`
reports which:

```python
import sbd
sbd.fcidump_format("nh2/nh2_uhf.fcidump")    # 'uhf'
sbd.fcidump_format("nh2/nh2_rohf.fcidump")   # 'restricted'
```

A file that fits neither is refused rather than guessed at: an index above `2*NORB`,
indices above `NORB` whose `(ij)` or `(kl)` pair mixes spins (both usually a wrong
`NORB`), or a `UHF=`/`IUHF=` header flag, which marks another program's layout.

## The data

`nh2/` holds the NH2 radical (6-31G, 13 orbitals, 5 alpha and 4 beta electrons):

| file | contents |
|---|---|
| `nh2_uhf.fcidump` | UHF orbitals, unrestricted layout |
| `nh2_rohf.fcidump` | ROHF orbitals, restricted layout |
| `nh2_alpha.txt`, `nh2_beta.txt` | every alpha (1,287) and beta (715) string: the full space, 920,205 determinants |
| `reference.json` | the exact energy, −55.633361496218, from PySCF FCI |

The two FCIDUMPs are one molecule in two orbital bases, so the **full-space energy is
the same from both** — a useful check that a file was read in the right layout. Their
Hartree-Fock determinants differ: −55.5324955198 (the UHF energy) and −55.5304407717
(ROHF). The unit tests use a 3-orbital H3 doublet in the same form, in `test/data/uhf/`.

## Running the drivers

From `examples/tpb/` — the full space, which must reproduce the exact energy:

```bash
python run_sbd_diag.py --fcidump ../uhf/nh2/nh2_uhf.fcidump --symmetrize_spin 0 \
    --adetfile ../uhf/nh2/nh2_alpha.txt --bdetfile ../uhf/nh2/nh2_beta.txt --tolerance 1e-8
# Ground state energy: -55.6333614962 Hartree
```

`--symmetrize_spin 0` is required here, and the driver says so if it is left out: the
default symmetric mode reuses the alpha list as the beta one, which for an open-shell
system has the wrong number of electrons. That applies to the ROHF file equally.

From `examples/gdb/` — heat-bath selection grown from the Hartree-Fock determinant
alone, which reaches the exact energy with a fraction of the full space:

```bash
python run_gdb_heatbath.py --fcidump ../uhf/nh2/nh2_uhf.fcidump \
    --subspace-from hf --cutoffs 1e-2,1e-3,1e-4,1e-5
#   System: 13 orbitals (26 spin orbitals), 9 electrons, MS2=1, uhf integrals
```

**Checking a state** works as for a restricted one: save it, then run
[`../analysis/variance.py`](../analysis/README.md) with the UHF FCIDUMP, which it
detects. From `examples/gdb/`, on a small subspace (20 alpha × 20 beta strings) whose
variance shows it is far from converged:

```bash
python -c "
a = open('../uhf/nh2/nh2_alpha.txt').read().split()[:20]
b = open('../uhf/nh2/nh2_beta.txt').read().split()[:20]
il = lambda x, y: ''.join(p + q for p, q in zip(x[::-1], y[::-1]))[::-1]
open('nh2_sub.txt', 'w').write('\n'.join(il(x, y) for x in a for y in b) + '\n')"
python run_gdb_diag.py --fcidump ../uhf/nh2/nh2_uhf.fcidump --detfiles nh2_sub.txt \
    --bit_length 26 --tolerance 1e-8 --savename nh2-      # note "Ground state energy:"
python ../analysis/variance.py --fcidump ../uhf/nh2/nh2_uhf.fcidump --state nh2- \
    --bit_length 26 --energy <E from run_gdb_diag.py>
```

The variance always runs on the stochastic method for a UHF file: upstream's
`gdb-stat-evaluator-uhf`, the `-D_UHF` build of
[`gdb-stat-evaluator`](https://github.com/r-ccs-cms/sbd/blob/main/apps/gdb-stat-evaluator/README.md).
The exact method is restricted-only.

Both drivers take the same MPI and `--device` options as for restricted files; see
[`../tpb/README.md`](../tpb/README.md) and [`../gdb/README.md`](../gdb/README.md).

## Making your own

```bash
python make_uhf_fcidump.py nh2 --outdir nh2   # requires PySCF
```

regenerates this directory; `h3` writes the test data. To write integrals of your own,
import `write_uhf_fcidump(path, (h1a, h1b), (eri_aa, eri_ab, eri_bb), norb, nelec, ms2,
ecore)` from the script. It takes chemist-notation spatial-orbital blocks, as
`pyscf.ao2mo.general` produces them.

The committed files are not regenerated in CI: a UHF calculation can converge to a
different solution across PySCF versions, which changes the integrals (though not the
exact energy).

## Limits

- **Backends:** `cpu` and `gpu` (Thrust). `gpu-omp` has no unrestricted build — its
  offload kernels read the restricted layout even under `-D_UHF` — and refuses such a
  file. `sbd.has_uhf(device)` reports whether a device has one.
- **Builds:** the unrestricted builds are made alongside the restricted ones by
  default; `SBD_BUILD_UHF=0` at install time skips them.
- **SQD:** the SQD loop (`sbd.sbd_solver.solve_sci`) is restricted only for now.
- **Memory:** the unrestricted two-body store is `4*NORB^4` doubles against about
  `NORB^4/8` restricted, about 32 times larger, and like the restricted one it is
  replicated on every rank.
- **Complex:** upstream's headers also have a complex unrestricted path (`_UHFCPLX`),
  but no upstream build or test uses it, so it is not built here.
