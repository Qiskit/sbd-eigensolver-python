# Tutorial: SBD's tools on Fe4S4, from first solve to converged energy

> **Draft.** Every command and number here was run on the server described below.
> Under review: the chemistry in Parts 6–7 would benefit from an expert read.

SBD is the classical diagonalization engine behind sample-based quantum
diagonalization (SQD): in an SQD workflow, configurations sampled on a quantum
computer define a subspace, and SBD finds the lowest-energy state within it. This
tutorial is about that engine and the tools around it — how to diagonalize a
subspace, grow and refine it, save and resume long runs, and judge how good the
resulting state is — on one realistic problem, the iron–sulfur cluster Fe4S4.

The same tools apply whatever produced the subspace: quantum samples, a carryover from
a previous run, or classical selection. Here the subspace is grown and refined with
SBD's own selection, so every step can be run without quantum hardware, and the
analysis tools of Parts 2, 6 and 7 apply unchanged to SQD's states (Part 9).

It is written for someone new to the area. Each step says what to run, what to look
at, and what a person working on the problem would conclude before deciding what to
do next.

**Contents**

- [Before you start](#before-you-start)
- [Part 0 — The system, and why this is hard](#part-0--the-system-and-why-this-is-hard)
- [Part 1 — A ready-made subspace, two ways](#part-1--a-ready-made-subspace-two-ways)
- [Part 2 — How good is this state?](#part-2--how-good-is-this-state)
- [Part 3a — Growing a product space by hand (TPB carryover)](#part-3a--growing-a-product-space-by-hand-tpb-carryover)
- [Part 3b — Growing a determinant list by hand (GDB carryover)](#part-3b--growing-a-determinant-list-by-hand-gdb-carryover)
- [Part 4 — Automating the cycle, and surviving interruptions](#part-4--automating-the-cycle-and-surviving-interruptions)
- [Part 5 — Going further, on more GPUs](#part-5--going-further-on-more-gpus)
- [Part 6 — How far are we from the right answer?](#part-6--how-far-are-we-from-the-right-answer)
  - [Measuring our states](#measuring-our-states)
  - [Background: active spaces and orbitals](#background-active-spaces-and-orbitals)
  - [Checking the literature](#checking-the-literature)
  - [A better starting point](#a-better-starting-point)
  - [Orbital optimization: the same determinants, better orbitals](#orbital-optimization-the-same-determinants-better-orbitals)
  - [Growing and optimizing together](#growing-and-optimizing-together)
  - [Going further from the broken-symmetry state](#going-further-from-the-broken-symmetry-state)
- [Part 7 — Extrapolating, and when not to](#part-7--extrapolating-and-when-not-to)
- [Part 8 — What we learned](#part-8--what-we-learned)
- [Part 9 — The same tools inside the SQD loop](#part-9--the-same-tools-inside-the-sqd-loop)
  - [A different SQD policy: Trim SQD](#a-different-sqd-policy-trim-sqd)
  - [Not yet: seeding SQD with a refined subspace](#not-yet-seeding-sqd-with-a-refined-subspace)
- [Appendix A — What each driver saves, and how to continue from it](#appendix-a--what-each-driver-saves-and-how-to-continue-from-it)
- [Appendix B — Traps](#appendix-b--traps)

**How to read it.** The parts build on each other, from basics to depth:

| If you want to … | Read |
|---|---|
| run SBD and know what its output means | Parts 0–2 |
| grow a subspace, by hand or automatically, and run long jobs safely | Parts 3–5 |
| understand what limits the result, and judge how close it is to exact | Parts 6–7 |
| use these tools with SQD | Part 9 (after Parts 0–2) |

Part 8 summarizes. The appendices are references: what each driver saves, and the
traps met along the way.

You will use every tool in the repository on the same system:

| Tool | What it does here |
|---|---|
| `examples/tpb/run_sbd_diag.py` | Diagonalize a TPB subspace: every pairing of an α string with a β string |
| `examples/gdb/run_gdb_diag.py` | Diagonalize a GDB subspace: an explicit list of determinants |
| `examples/gdb/run_gdb_heatbath.py` | Grow a GDB subspace round by round with heat-bath selection |
| `examples/analysis/variance.py` | Measure how far a saved state is from the exact answer |
| `examples/analysis/extrapolate.py` | Turn a series of states into one best estimate, with an error bar |
| `examples/tpb/run_sqd_sbd.py` | The SQD loop, from quantum-sampled bitstrings (Part 9) |

## Before you start

The software is assumed to be installed, with MPI and, for the GPU steps, an NVIDIA
build (the `gpu` backend). The numbers in this tutorial come from one server:

| | |
|---|---|
| CPUs | 2 × Intel Xeon Platinum 8474C: 96 cores (192 hardware threads), 4 NUMA domains |
| Memory | 2 TB |
| GPUs | 8 × NVIDIA H100 80 GB (SXM, NVLink) |
| Software | Ubuntu 24.04, MPICH 5.0, sbd-eigensolver 1.7 with the cpu, gpu and gpu-omp backends |

Your timings will differ; what matters is how they compare from step to step.

**Where the work runs.** Each step below says which part of the machine it uses:

| Work | Runs on | Programming model |
|---|---|---|
| Davidson diagonalization (TPB and GDB) | GPUs, one per MPI rank | MPI across GPUs |
| heat-bath expansion (choosing new determinants) | CPU cores | MPI + OpenMP, no GPU version |
| variance and PT2 (`variance.py`) | CPU cores | MPI, best with one thread per rank |
| orbital optimization (ffsim) | CPU, one process | NumPy/SciPy |
| SQD's configuration recovery and subsampling | CPU, one process | Python |

So a GPU run still does much of its work on the CPU, and needs its cores as well as its
GPUs. Every MPI rank should get its share of the cores and be bound to them.

**Set up once.** These commands assume MPICH (its launcher takes `-bind-to` and
`-genv`); SBD gives MPI rank *i* GPU *i* by itself. Two shell functions hold the launch
rules, so every later command can be pasted as it stands: `gpus N cmd` runs `cmd` on N
GPU ranks, each bound to its share of the cores, and `cpus cmd` runs it on every core,
one rank per core. On another machine, set `CORES` and `NGPU`; under a scheduler,
redefine the two functions with your site's launcher (`srun --gpus-per-task=1
--cpu-bind=cores`, say).

```bash
export SBD=~/sqd-onboard/sw/sbd-eigensolver-python          # this repository
export UP=$SBD/vendor/sbd-upstream/apps
export FCIDUMP=$UP/chemistry_tpb_selected_basis_diagonalization/fcidump_Fe4S4.txt
export ALPHA=$UP/chemistry_tpb_selected_basis_diagonalization/AlphaDets.txt
G=$UP/chemistry_gdb_selected_basis_diagonalization
export DETS=$G/det0.txt,$G/det1.txt,$G/det2.txt,$G/det3.txt
export WORK=$HOME/fe4s4-tutorial                              # every rank must see it
export CORES=96 NGPU=8
mkdir -p $WORK
gpus() { local n=$1; shift; mpirun -np $n -bind-to core:$((CORES/n)) -genv OMP_NUM_THREADS $((CORES/n)) "$@"; }
cpus() { mpirun -np $CORES -bind-to core:1 -genv OMP_NUM_THREADS 1 "$@"; }
```

Four habits apply throughout. Each one has caught real runs:

- **Use one FCIDUMP for everything.** An FCIDUMP fixes the orbitals as well as the
  molecule. Two FCIDUMPs of the same molecule in different orbitals give different
  energies for the same subspace, so their results cannot be compared.
- **Give `--bit_length` explicitly, and keep it.** It is how determinants are packed
  into 64-bit words: 20 for TPB, 62 for GDB here. Saved files do not record it, so
  whatever reads them later must be told the same value.
- **Create output directories first.** The drivers check, and stop with an error if a
  directory is missing, because SBD itself would lose the file without saying so.
- **Bind ranks to cores.** Unbound, every rank's threads compete for every core: on
  this server one heat-bath expansion took 1.7 times as long without `-bind-to`.

---

## Part 0 — The system, and why this is hard

```bash
head -3 $FCIDUMP
```

The header gives `NORB=36` (spatial orbitals), `NELEC=54` (electrons) and `MS2=0`
(as many spin-up electrons as spin-down: 27 each). Every determinant puts 27 α
electrons and 27 β electrons into 36 orbitals, so there are
$\binom{36}{27}^2 \approx 8.8 \times 10^{15}$ of them. The exact answer, full
configuration interaction (FCI), diagonalizes the Hamiltonian over all of them. That
is out of reach.

The simplest approximation is a single determinant. The obvious one fills the lowest
27 orbitals for each spin; if the FCIDUMP's orbitals are Hartree–Fock (HF) orbitals,
ordered by energy, that is the HF determinant. (What "orbitals" means here, and why
the choice matters, comes up in Part 6.) Its energy comes from one round of the
heat-bath driver, whose first row diagonalizes the starting determinant alone:

```bash
cd $SBD/examples/gdb
gpus 1 python run_gdb_heatbath.py --device gpu --bit_length 62 --fcidump $FCIDUMP \
    --subspace-from hf --cutoffs 1e-3 --max_rounds 1
```

A single determinant cannot be split across GPUs, so this runs on one.

**What we see.** Round 0, dimension 1: −326.5243550095 Ha, in 0.2 s.

**Is this the Hartree–Fock energy?** Yes, here — but it is worth checking, because it
depends on the orbitals. Hartree–Fock (HF) finds the single determinant with the lowest
energy by optimizing the orbitals; in the resulting *canonical* orbitals, the Fock
matrix — the average field each electron feels — is diagonal, and its diagonal (the
orbital energies) is in ascending order. Both are easy to test from the FCIDUMP with
PySCF: build the Fock matrix of the lowest-27 determinant, and run HF starting from it.

```bash
cat > $WORK/hf_check.py <<'EOF'
import os
import numpy as np
from pyscf import gto, scf, ao2mo, tools

d = tools.fcidump.read(os.environ["FCIDUMP"])
n, nocc = d["NORB"], d["NELEC"] // 2
mol = gto.M(verbose=0); mol.nelectron = d["NELEC"]; mol.incore_anyway = True
mf = scf.RHF(mol)
mf.get_hcore = lambda *a: d["H1"]
mf.get_ovlp = lambda *a: np.eye(n)
mf._eri = ao2mo.restore(8, d["H2"], n)
mf.energy_nuc = lambda *a: d["ECORE"]           # the core energy (0 in these files)
dm0 = np.diag([2.0] * nocc + [0.0] * (n - nocc))   # the lowest-27 determinant
F = mf.get_fock(dm=dm0)
print("max off-diagonal Fock element:", abs(F - np.diag(np.diag(F))).max())
print("HF energy from that start:", mf.kernel(dm0=dm0))
EOF
python $WORK/hf_check.py
```

This is a small calculation, in one process on the CPU.

For this FCIDUMP the off-diagonal Fock elements are below 10⁻⁶ and the orbital
energies are ascending, and HF started from the lowest-27 determinant stays exactly
where it is. So the orbitals are canonical HF orbitals, and −326.5243550 Ha is a
Hartree–Fock energy.

It is not the only one. An FCIDUMP fixes one set of orbitals, and the same active
space can be written in others. The HF equations can have several self-consistent
solutions, especially for a strongly correlated system like this one, and orbitals
from a different solution can give a lower HF energy than this file's. We stay with
the upstream file throughout Parts 0–5, so this does not affect the comparisons there.

**What we take from it.** Any single determinant's energy is an upper bound: the true
energy is lower. The HF energy is the best a single determinant can do in a given set
of orbitals, and the HF energy itself depends on which HF solution the orbitals come
from. The difference is the *correlation energy*, and finding as much of it as possible is the
whole problem. We do not know how large it is, so we need two things at every step:
a better state, and a way to tell how far that state still is from the exact one.

---

## Part 1 — A ready-made subspace, two ways

Upstream SBD ships a subspace for this FCIDUMP: 244 α strings in `AlphaDets.txt`
(the TPB example), and the same subspace written out as 59,536 = 244 × 244 full
determinants in `det0–3.txt` (the GDB example).

**TPB**, on all 8 GPUs as a 4 × 2 grid (α strings split four ways, β two):

```bash
cd $SBD/examples/tpb
gpus 8 python run_sbd_diag.py --device gpu \
    --adet_comm_size 4 --bdet_comm_size 2 --bit_length 20 --tolerance 1e-6 \
    --fcidump $FCIDUMP --adetfile $ALPHA
```

**GDB**, the same determinants as an explicit list, one shard per GPU:

```bash
cd $SBD/examples/gdb
gpus 8 python run_gdb_diag.py --device gpu --b_comm_size 8 --bit_length 62 \
    --tolerance 1e-6 --fcidump $FCIDUMP --detfiles $DETS
```

Both run Davidson on the GPUs, split over MPI ranks.

**What we see.** Both report −326.6982518821 Ha, identical to every printed digit,
and 174 mHa below the single determinant of Part 0. Each takes seconds.

**What we take from it.**

- Two independent solvers agree on the same subspace. That checks the input and both
  codes at once, and lets us trust either from here on.
- They represent the subspace differently. TPB stores 244 + 244 strings and forms
  every pairing on the fly, so its subspace is always a full α × β product. GDB stores
  every determinant, so it can hold any list at all — which is what the rest of the
  tutorial relies on.
- We have an energy, but not yet any idea how good it is.

---

## Part 2 — How good is this state?

Save the state, then measure it:

```bash
mkdir -p $WORK/p2
cd $SBD/examples/gdb
gpus 8 python run_gdb_diag.py --device gpu --b_comm_size 8 --bit_length 62 \
    --tolerance 1e-6 --fcidump $FCIDUMP --detfiles $DETS --savename $WORK/p2/up-

cd $SBD/examples/analysis
cpus python variance.py \
    --fcidump $FCIDUMP --state $WORK/p2/up- --bit_length 62 \
    --energy -326.6982518821 --batches 10 --json $WORK/p2/up-var.json
```

`--savename` writes the determinants and their amplitudes, one file per GPU.
`variance.py` runs on CPU cores only — it has no GPU version — and is fastest with one
MPI rank per core, which is what `cpus` does. It does not need the state's layout: the
eight files are spread evenly over all its ranks. `--energy` skips recomputing an
energy we already know.

**What we see.**

```
Dimension:     59,536
Variance:      7.485474e-02  +- 4.36e-04  Ha^2
PT2:           -6.641513e-02  +- 5.65e-04  Ha
E_var:         -326.6982518821  Ha
E_var + PT2:   -326.7646670154  +- 5.65e-04  Ha
```

Your numbers will differ slightly, within the ± values: the method samples at random,
and `--seed` fixes the samples only for a given number of ranks.

The analysis took minutes, against about a second for the diagonalization that
produced the state. That is normal: it expands every sampled determinant into all its
excitations, tens of thousands each for 54 electrons in 36 orbitals, on CPU cores.

**What we take from it.**

- **The variance** would be zero for an exact eigenstate. It is far from zero, so this
  state is far from converged.
- **PT2** estimates the energy still missing from outside the subspace. It is a
  second-order estimate, reliable only when it is small; here it is a rough guide.
- **The ± values** come from repeating the random sampling in batches. They say how
  much the numbers would move with different samples, not how far the method is from
  the truth.
- So the shipped subspace leaves a lot out. The question becomes how to grow it.

---

## Part 3a — Growing a product space by hand (TPB carryover)

After a solve, SBD can select a *carryover*: the strings worth keeping, optionally
with new ones added. `run_sbd_diag.py` writes them out, and the next run starts from
them. Each run's files are its checkpoint, and the next run is the resume.

| `--carryover_type` | Selection (TPB, per spin string) |
|---|---|
| 1 | keep the strings carrying most of the weight |
| 2 | type 1, then add every single excitation of the kept strings |
| 3 | keep strings above a weight threshold, then add their single excitations |

First, how big does each choice make the next subspace? Run each setting once from
the upstream 244 strings and count the strings it writes:

```bash
mkdir -p $WORK/3a
cd $SBD/examples/tpb
tpb() { gpus 8 python run_sbd_diag.py --device gpu --adet_comm_size 4 --bdet_comm_size 2 \
            --bit_length 20 --tolerance 1e-6 --fcidump $FCIDUMP "$@"; }
for set in "1 --carryover_ratio 0.5" "2 --carryover_ratio 0.02" "2 --carryover_ratio 0.05" \
           "2 --carryover_ratio 0.1" "3 --carryover_threshold 1e-2" "3 --carryover_threshold 1e-3"; do
    t=${set%% *}; tag=$(echo "$set" | tr -d ' .-')
    tpb --adetfile $ALPHA --carryover_type $t ${set#* } --carryover_adetfile $WORK/3a/probe-$tag.txt \
        | grep -E "Carryover determinants|truncated weight in carry-over for alpha|percentage of wf"
    echo "  $set: $(wc -l < $WORK/3a/probe-$tag.txt) strings"
done
```

The `tpb` function fixes what every TPB run here shares: all 8 GPUs as a 4 × 2 grid,
and the same FCIDUMP and settings.

| Setting | Strings kept (weight left out) | Strings written | Next subspace |
|---|---|---|---|
| type 1, ratio 0.5 | 122 (3.5%) | 122 | 14,884 |
| type 2, ratio 0.02 | 4 (28%) | 844 | 712,336 |
| type 2, ratio 0.05 | 12 (21%) | 2,239 | 5.0 million |
| type 2, ratio 0.1 | 24 (15%) | 4,094 | 16.8 million |
| type 3, threshold 1e-2 | 64% of the state used | 452 | 204,304 |
| type 3, threshold 1e-3 | 75% of the state used | 3,969 | 15.8 million |

The single excitations do the growing: 4 strings become 844, because each has 27 × 9
= 243 single excitations, and the subspace is the square of the string count. Even the
smallest useful setting makes the next subspace 12 times larger. We continue with the
two at a sensible size, type 2 at ratio 0.02 and type 3 at 1e-2.

Each step restarts from the previous carryover, with the saved state as Davidson's
starting vector (`--loadname`). SBD matches each saved string to the new list, so
amplitudes carry over for the strings kept and start at zero for the new ones. A saved
TPB state loads only onto the α × β grid that wrote it — which is why every run here
uses the same 4 × 2 grid; the driver checks.

```bash
cd $SBD/examples/tpb
for set in "t2 2 --carryover_ratio 0.02" "t3 3 --carryover_threshold 1e-2"; do
    read tag t knobs <<< "$set"
    tpb --adetfile $ALPHA --carryover_type $t $knobs \
        --carryover_adetfile $WORK/3a/$tag-s1-alpha.txt --savename $WORK/3a/$tag-s1- | grep "Ground state"
    tpb --adetfile $WORK/3a/$tag-s1-alpha.txt | grep -E "Ground state|davidson"          # cold start
    tpb --adetfile $WORK/3a/$tag-s1-alpha.txt --loadname $WORK/3a/$tag-s1- \
        --carryover_type $t $knobs \
        --carryover_adetfile $WORK/3a/$tag-s2-alpha.txt --savename $WORK/3a/$tag-s2- | grep -E "Ground state|davidson"
    tpb --adetfile $WORK/3a/$tag-s2-alpha.txt --loadname $WORK/3a/$tag-s2- | grep -E "Ground state|davidson"
done
```

**What we see.**

| Run | α strings | Dimension (strings²) | Energy (Ha) |
|---|---|---|---|
| Part 1 | 244 | 59,536 | −326.6982519 |
| type 2, step 2 (from 844 strings) | 844 | 712,336 | −326.7167843 |
| type 2, step 3 (from 2,906 strings) | 2,906 | 8,444,836 | −326.7685459 |
| type 3, step 2 (from 452 strings) | 452 | 204,304 | −326.7074213 |
| type 3, step 3 | 452 | 204,304 | −326.7074213 |

The 8.4-million-determinant solve is where the GPUs earn their keep: the α × β product
is formed on the fly across the 4 × 2 grid, so its memory and its work both split
eight ways.

**What we take from it.**

- **Type 2 keeps growing, and the dimension is the *square* of the string count.**
  Four strings plus their singles became 844, then 2,906, and the subspace went from
  60 thousand to 8.4 million in two steps. TPB cannot add a determinant without its
  whole row and column of pairings.
- **For all that size, the gain is modest:** 70 mHa at 8.4 million determinants. Most
  of the new pairings matter little — the cost of the product structure. The next part
  does the same cycle with a determinant list, which can grow one determinant at a
  time.
- **Type 3 stalls at once.** Its second step carried over exactly the 452 strings it
  started from, so a third step repeats the second. At this threshold, the strings
  above it and their singles are already all there: a fixed point.
- **`--loadname` hardly helped** (compare the cold and loaded Davidson times). The saved state covers
  only the strings that were kept, a small part of the new list, so it is a poor
  starting vector. It pays off when restarting on the same or a slightly larger
  subspace, not after a carryover that rebuilds most of it.

---

## Part 3b — Growing a determinant list by hand (GDB carryover)

GDB's carryover types select full determinants, and two of them are heat-bath CI:

| `--carryover_type` | Selection (GDB, per determinant) |
|---|---|
| 1 | keep the determinants carrying most of the weight (`--carryover_ratio`) |
| 2 | heat-bath expansion: add determinants coupled to the state above `--heatbath_cutoff` ("on-demand exhaustive") |
| 3 | heat-bath expansion, ranking candidates from the Hamiltonian's integrals ("integral-driven") |

The expanded list includes the determinants it grew from, so it is the next subspace.
`--carryover_detfile` writes it as sorted determinant files, and prints the
`--detfiles` list for the next run.

Step 1, one heat-bath round from the upstream subspace, on all 8 GPUs:

```bash
mkdir -p $WORK/3b
cd $SBD/examples/gdb
gdb() { local n=$1; shift; gpus $n python run_gdb_diag.py --device gpu --b_comm_size $n \
            --bit_length 62 --tolerance 1e-6 --fcidump $FCIDUMP "$@"; }
files() { ls $1*.txt | paste -sd,; }
gdb 8 --detfiles $DETS --carryover_type 2 --heatbath_cutoff 1e-4 \
    --carryover_detfile $WORK/3b/r1- --savename $WORK/3b/r0-
```

The expansion runs on the CPU cores of all eight ranks, then the expanded list is
sorted across them and written as eight files, each rank writing its own.

Step 2, solve the expanded list, here on 4 GPUs to show the files do not tie you to a
layout, and expand again:

```bash
cd $SBD/examples/gdb
gdb 4 --detfiles $(files $WORK/3b/r1-) --carryover_type 2 --heatbath_cutoff 1e-4 \
    --carryover_detfile $WORK/3b/r2- --savename $WORK/3b/r1-
gdb 8 --detfiles $(files $WORK/3b/r2-) | grep -E "Ground state|Subspace dimension"
```

`--loadname $WORK/3b/r0-` would start Davidson from step 1's state, but SBD can load
a saved GDB state only onto at least as many shards as it has files: 8 files cannot
go onto 4 GPUs. The driver refuses rather than start from part of the state.

The other two types, from the same upstream subspace:

```bash
cd $SBD/examples/gdb
gdb 8 --detfiles $DETS --carryover_type 3 --heatbath_cutoff 1e-4 --carryover_detfile $WORK/3b/t3-
gdb 8 --detfiles $(files $WORK/3b/t3-) | grep -E "Ground state|Subspace dimension"
gdb 8 --detfiles $DETS --carryover_type 1 --carryover_ratio 0.5 --carryover_detfile $WORK/3b/t1-
gdb 8 --detfiles $(files $WORK/3b/t1-) | grep -E "Ground state|Subspace dimension"
```

**What we see.**

| Run | Dimension | Energy (Ha) |
|---|---|---|
| Part 1 (upstream subspace) | 59,536 | −326.6982519 |
| one round, type 2, cutoff 1e-4 | 636,338 | −326.7781205 |
| two rounds, type 2 | 806,932 | −326.7848825 |
| one round, type 3, cutoff 1e-4 | 636,338 | −326.7781205 |
| type 1, ratio 0.5 | 29,768 | −326.6978787 |

Writing the carryover takes a fraction of a second: each rank writes its own sorted
part, and nothing is gathered onto one rank.

**What we take from it.**

- **One heat-bath round reaches about the energy of the whole Part 3a route** —
  −326.778 at 636 thousand determinants, against −326.769 at 8.4 million — because it
  adds individual determinants that couple strongly to the state, where TPB must add
  every pairing of its strings. Which representation suits a subspace depends on where
  it came from; the GDB list is the more flexible container.
- **Types 2 and 3 chose exactly the same determinants here** — the same 636,338, to
  the same energy — but type 3, which ranks candidates from the integrals, expanded
  several times faster. Do not assume they agree in general; at a different state or
  cutoff they need not.
- **Type 1 halves the subspace for 0.4 mHa.** Half of the shipped determinants carried
  almost none of the energy: pruning is cheap here, but it does not improve anything.
- **The second round gains much less than the first** (7 mHa against 80) at the same
  cutoff. A smaller cutoff admits weaker couplings, and is needed to keep improving.
- **This cycle is mechanical:** solve, expand, write, restart, and lower the cutoff when
  progress stalls. That is what the heat-bath driver automates.

---

## Part 4 — Automating the cycle, and surviving interruptions

`run_gdb_heatbath.py` runs the Part 3b cycle as a *ladder*: rounds at one cutoff until
the energy stops moving (`--energy_tol`, at most `--max_rounds`), then the next,
smaller cutoff. Each round is a type 2 heat-bath expansion, as in Part 3b;
`--carryover_type 3` selects the integral-driven variant. It can start from a single
determinant, which tests the method without any prepared subspace.

A one-determinant start cannot be split across GPUs, so run the first rungs on one:

```bash
mkdir -p $WORK/hb
cd $SBD/examples/gdb
gpus 1 python run_gdb_heatbath.py --device gpu --bit_length 62 \
    --fcidump $FCIDUMP --subspace-from hf --cutoffs 1e-3,1e-4 \
    --checkpoint_prefix $WORK/hb/ck- --checkpoint_frequency 4 \
    --savename $WORK/hb/rung1- --log $WORK/hb/ladder.json
```

One rank here, so `gpus 1` gives it all 96 cores: each round solves on the GPU, then
expands the state on the CPU cores, which is most of a round's time.

```
  rung round    dimension             energy           dE    secs   save
     0     0            1    -326.5243550095                  0.2   0.00
     0     1        11308    -326.6797962999    -0.155441     0.7   0.00
     0     2        11960    -326.7074134800    -0.027617     0.7   0.00
     0     3        13064    -326.7132589184    -0.005845     0.7   0.00
     0     4        13208    -326.7147168485    -0.001458     0.7   0.00
     0     5        13236    -326.7147680136    -0.000051     0.7   0.00
  rung 0 at cutoff 0.001: subspace stopped growing at 13236 -- its fixed point for this cutoff
     1     0        13236    -326.7147680136    +0.000000     1.5   0.00
     1     1       517855    -326.7769847796    -0.062217    25.2   0.01
     1     2       679494    -326.7826482794    -0.005663    28.3   0.00
```

Rung 0 reaches a *fixed point*: at a cutoff of 1e-3, expanding the state adds nothing
new, so the ladder moves on. The first round of rung 1 re-solves the same 13,236
determinants (same energy), and its expansion, at the smaller cutoff, takes the
subspace to half a million.

`--checkpoint_prefix` saves every round's state plus `ck-.json` (where the ladder had
got to); `--checkpoint_frequency 4` also keeps every 4th round as `ck-_rNNNN-`.

**Interrupt it.** Press Ctrl-C during the 1e-4 rung (here, during round 3), then look
at the checkpoint:

```bash
python -c "import json; d=json.load(open('$WORK/hb/ck-.json')); \
    print(d['rung'], d['round'], d['dimension'], d['energy'])"
```

```
1 2 679494 -326.7826482793819
```

It holds round 2, the last one completed; round 3 was lost. Round 4 and 8 copies
(`ck-_r0004-`, `ck-_r0008-`) are there too, from `--checkpoint_frequency 4`.

**Resume it**, into the same prefix:

```bash
cd $SBD/examples/gdb
gpus 1 python run_gdb_heatbath.py --device gpu --bit_length 62 \
    --fcidump $FCIDUMP --resume_from_prefix $WORK/hb/ck- \
    --checkpoint_prefix $WORK/hb/ck- --checkpoint_frequency 4 \
    --savename $WORK/hb/rung1- --log $WORK/hb/ladder.json
```

**What we see.**

```
  Cutoff ladder: [0.001, 0.0001], resuming at rung 1 round 2
  rung round    dimension             energy           dE    secs   save
     1     2       679494    -326.7826482794    +0.000000    29.8   0.00  (re-solve of the checkpoint)
     1     3       711942    -326.7836111119    -0.000963    27.1   0.00
     1     4       720964    -326.7838677986    -0.000257    28.1   0.01
     1     5       724263    -326.7839581196    -0.000090    28.1   0.00
     1     6       725813    -326.7840028448    -0.000045    28.6   0.00
     1     7       726599    -326.7840263718    -0.000024    29.3   0.00
  rung 1 at cutoff 0.0001: hit --max_rounds 8 without converging
  ...
  End:   dim 726599, E = -326.7840263718
  Saving: 0.05 s in the driver over 14 round(s) ...
```

The same ladder run without interruption ends at the same place, 726,599
determinants and −326.7840263718 Ha, identical to every digit. The interruption cost
one repeated round, and saving a fraction of a second in all.

The rung stopped at `--max_rounds 8` rather than because the energy settled: its
last round still gained 24 μHa, above `--energy_tol` (10 μHa). That is a budget, not
a failure; the next, smaller cutoff picks up where this one left off.

**What we take from it.**

- **Starting from one determinant, the ladder went 86 mHa below the shipped subspace**
  (−326.7840 against −326.6983) in a few minutes on one GPU. It needs no input
  subspace at all, which also makes it a useful way to grow or refine one that came
  from elsewhere.
- **Checkpointing is cheap** next to a round, so there is little reason to run without it.
- **A resume repeats one round on purpose:** SBD saves the state before expanding it,
  so the resumed run solves that state again (quickly, from its saved amplitudes) and
  expands it.

---

## Part 5 — Going further, on more GPUs

Now the subspace is large enough to split. Resume on all 8 GPUs and add a smaller
cutoff. New `--cutoffs` become further rungs after the checkpoint's, and the history
is kept:

```bash
cd $SBD/examples/gdb
gpus 8 python run_gdb_heatbath.py --device gpu --b_comm_size 8 --bit_length 62 \
    --fcidump $FCIDUMP --resume_from_prefix $WORK/hb/ck- --cutoffs 3e-5 \
    --checkpoint_prefix $WORK/hb/ck- --savename $WORK/hb/rung2- --log $WORK/hb/ladder.json
```

**What we see.**

```
  Cutoff ladder: [0.001, 0.0001, 3e-05], resuming at rung 2 round 0
  rung round    dimension             energy           dE    secs   save
     2     0       726599    -326.7840263718    +0.000000    20.8   0.03  (re-solve of the checkpoint)
     2     1      4928716    -326.8170168752    -0.032991   158.5   0.10
     2     2      5993401    -326.8203474884    -0.003331   194.9   0.10
     2     3      6283336    -326.8211664101    -0.000819   205.7   0.10
     2     4      6402076    -326.8214887532    -0.000322   215.1   0.10
     2     5      6459981    -326.8216385536    -0.000150   210.6   0.08
     2     6      6488697    -326.8217143095    -0.000076   212.6   0.12
     2     7      6503471    -326.8217505170    -0.000036   211.4   0.11
  rung 2 at cutoff 3e-05: hit --max_rounds 8 without converging
  End:   dim 6503471, E = -326.8217505170
```

The 726,599-determinant state from one GPU loaded onto eight and was solved again.
The first round at the new cutoff did most of the work: 33 mHa, and a subspace almost
7 times larger. After that each round gained about half the previous one, and the rung
ran out of rounds while still gaining 36 μHa a round. Each rung costs several times the
one before.

If you try the Part 4 command on 4 GPUs directly, it stops at once: a one-determinant
seed leaves three GPUs with nothing, and the GPU solver cannot run with an empty shard.

**What we see, over the whole ladder.**

| Rung (cutoff) | GPUs | Final dimension | Energy (Ha) | Rounds |
|---|---|---|---|---|
| 1e-3 | 1 | 13,236 | −326.7147680 | 6 |
| 1e-4 | 1 | 726,599 | −326.7840264 | 8 |
| 3e-5 | 8 | 6,503,471 | −326.8217505 | 8 |

**What we take from it.**

- **Each smaller cutoff still lowers the energy,** by less each time and at much higher
  cost: 69 mHa, then 38 mHa, each for about nine times the dimension.
  The returns are diminishing, but nothing here tells us where they stop.
- **Most of each round is the expansion, not the solve** — see below — and the
  expansion runs on CPU cores.
- **The determinants are spread evenly across GPUs** by the default
  `--determinant_distribution grid-cyclic-balanced`. Upstream's default, `equal-bra-a`,
  can put most of a heat-bath subspace on one GPU: after one expansion of the
  upstream subspace, 52% of it landed on GPU 0, and the solve was 1.5× slower.

**Where the time goes.** In a round at millions of determinants, the Davidson solve on
the GPUs takes seconds; the heat-bath expansion on the CPU cores takes minutes. The
expansion is SBD's MPI + OpenMP code, with no GPU version, so it sets the pace of the
ladder. Its speed follows the number of CPU cores it gets, not the number of GPUs: one
5-million-determinant round expanded in about the same time on 4 GPUs with 24 threads
each as on 8 with 12, and twice as slowly with only 8 threads per rank. Two things
follow: give every rank all its cores, and, on a cluster, more nodes speed the
expansion up, since it is MPI-parallel.

**When more GPUs do help: memory.** At 6 million determinants each rank used about
12 GB of host memory and 9 GB of GPU memory, growing roughly in proportion to the
dimension. On 80 GB GPUs that puts the ceiling for 4 GPUs near 50 million
determinants; beyond it, spread the subspace over 8 (`--b_comm_size 8`), which halves
each GPU's share — which is why these commands use all eight. `--max_dim` stops the
ladder before a round would exceed a size you choose, rather than failing partway
through it. The check can only happen after the expansion, since that is what decides
the size: a stopped round has spent its expansion, but the checkpoint still holds the
last completed round, so a rerun with a larger cap loses nothing else.

---

## Part 6 — How far are we from the right answer?

### Measuring our states

First, ask the states themselves, as in Part 2:

```bash
cd $SBD/examples/analysis
energy() { python -c "import json; print(json.load(open('$1.json'))['energy'])"; }
for r in rung1 rung2; do
    cpus python variance.py --fcidump $FCIDUMP --state $WORK/hb/$r- --bit_length 62 \
        --energy $(energy $WORK/hb/$r-) --batches 10 --json $WORK/hb/$r-var.json
done
```

The heat-bath driver records each saved state's energy beside it, in `PREFIX.json`;
`energy` reads it back.

| State | Dimension | E_var (Ha) | Variance (Ha²) | PT2 (mHa) | E_var + PT2 (Ha) |
|---|---|---|---|---|---|
| Upstream subspace (Part 2) | 59,536 | −326.6982519 | 0.0749 | −66.4 ± 0.6 | −326.7647 |
| Rung 1e-4 | 726,599 | −326.7840264 | 0.0689 | −45.0 ± 0.4 | −326.8290 |
| Rung 3e-5 | 6,503,471 | −326.8217505 | 0.0626 | −35.7 ± 0.9 | −326.8575 |

Each analysis took longer than the one before — minutes to tens of minutes on the 96
cores — growing with the size of the state.

**What we see.** PT2 shrinks as the subspace grows, but slowly, and the variance
barely moves: 110 times more determinants took it from 0.075 to 0.063 Ha². E_var + PT2
moves by about as much as E_var does between rungs.

**What we take from it.** The variance should approach zero as the state converges,
and it is not doing so. A ladder can keep producing lower energies while staying far
from the answer. To know how far, we need an outside reference.

### Background: active spaces and orbitals

Two ideas are needed to make sense of what follows.

**An active space is a model of the molecule.** The real cluster has hundreds of
electrons in hundreds of orbitals. Most of them behave simply: core electrons stay put,
and high-energy orbitals stay empty. An *active space* keeps only the electrons and
orbitals where the interesting physics happens — here the iron 3d and sulfur 3p
orbitals, 54 electrons in 36 orbitals — and folds everything else into fixed numbers.
The FCIDUMP is that model, written down: the one- and two-electron integrals over the
36 active orbitals. Solving it exactly (full CI) gives the *active-space energy*; the
part folded away is a separate constant, the core energy, which these FCIDUMPs leave
at zero.

**The same model can be written in many sets of orbitals.** An orbital is a mixture of
the molecule's basis functions, and any rotation that mixes the 36 active orbitals
among themselves describes exactly the same physics. The exact energy is unchanged;
what changes is the integrals, and how *compactly* the ground state can be written as a
sum of determinants. In one set of orbitals a few thousand determinants may carry most
of the state; in another, millions. That is why the choice of orbitals matters to every
method here, all of which keep only a selection of determinants.

Some common choices:

- **Canonical Hartree–Fock orbitals**, ordered by energy: filling the lowest ones gives
  the Hartree–Fock determinant, the best single determinant for a closed-shell state.
- **Localized orbitals**, each sitting on one atom: easy to read chemically ("the 3d
  orbitals of iron 2"), but with no energy ordering, so "the lowest 27" means nothing.
- **Optimized orbitals**, rotated to make a particular state as compact as possible —
  the subject of the next section but one.

### Checking the literature

This active space — [Fe4S4(SCH3)4]²⁻, 54 electrons in 36 orbitals (Fe 3d and S 3p) —
is a standard benchmark. Its integrals were published by Li and Chan (*J. Chem. Theory
Comput.* **13**, 2681 (2017)), and DMRG, which handles this kind of system well, puts
its ground-state energy at about **−327.24 Ha**: −327.2396 in their release
(not fully converged), and −327.245 to −327.247 in later, larger calculations
(Zhai et al., 2026; Legeza et al., 2026). These are active-space energies, with no
core energy, like ours.

Is ours the same problem? Download their FCIDUMP and compare:

```bash
curl -L -o $WORK/lichan_fe4s4 https://raw.githubusercontent.com/zhendongli2008/Active-space-model-for-Iron-Sulfur-Clusters/main/Fe2S2_and_Fe4S4/Fe4S4/fe4s4
export LICHAN=$WORK/lichan_fe4s4
python - <<'EOF'
import os
import numpy as np
from pyscf import tools
for path in (os.environ["FCIDUMP"], os.environ["LICHAN"]):
    d = tools.fcidump.read(path, verbose=0)
    print(f"{os.path.basename(path)}: NORB={d['NORB']} NELEC={d['NELEC']} "
          f"trace(h)={np.trace(d['H1']):.10f}")
EOF
```

Both files have the same 222,111 two-electron integrals and, to 10 digits, the same
trace of the one-electron matrix (−435.2879311575), which does not change when the
orbitals are rotated. The integrals themselves differ. **They are the same active
space in two different sets of orbitals,** so they share one exact energy: about
−327.24 Ha.

**Our best energy, −326.822 Ha, is 0.42 Ha above it.** That is not a small
convergence error; it is most of the correlation energy still missing. And PT2 said
36 mHa was missing, not 420. PT2 is only reliable once the state is close to
converged; here it was not, and nothing in our own numbers could tell us so except
the stubborn variance.

### A better starting point

Li and Chan also give the key piece of chemistry. The four iron atoms are
antiferromagnetically coupled: two have their unpaired electrons spin-up, two
spin-down. A determinant that puts all electrons in the "lowest" orbitals for both
spins describes none of that. They suggest a *broken-symmetry* determinant instead:
the 3d orbitals of Fe1 and Fe2 (orbitals 2–6 and 7–11) filled with spin-up electrons,
those of Fe3 and Fe4 (24–28, 29–33) with spin-down.

In their orbitals, the "lowest 27 orbitals" determinant and the broken-symmetry one
differ enormously:

| Starting determinant (Li–Chan orbitals) | Energy (Ha) |
|---|---|
| Lowest 27 orbitals for both spins | −303.113 |
| Broken symmetry (Fe1, Fe2 up; Fe3, Fe4 down) | −326.294 |

The first is 24 Ha above the answer. These orbitals are localized on the atoms, so
"lowest in energy" has no physical meaning in them. The second is a single
determinant only 0.95 Ha above.

Here is that determinant, written out. The 16 orbitals that are not Fe 3d (the S 3p
and ligand orbitals) hold two electrons each. Fe1's and Fe2's 3d shells are filled
with spin-up electrons and Fe3's and Fe4's with spin-down: 20 electrons. That leaves
one more electron of each spin, and they go one to each Fe(II) ion as the minority
spin — a spin-down electron on Fe1 (orbital 2) and a spin-up one on Fe3 (orbital 26)
— giving 22 d electrons, two Fe(III) d⁵ and two Fe(II) d⁶. The drivers read a
determinant as one string of 72 characters, each orbital's β then α bit, highest
orbital first:

```bash
python - <<'EOF'
import os
n = 36
fe = {1: range(2, 7), 2: range(7, 12), 3: range(24, 29), 4: range(29, 34)}  # Fe 3d, 1-based
d = {p for shell in fe.values() for p in shell}
both = [p for p in range(1, n + 1) if p not in d]               # 16 doubly occupied
alpha = set(both) | set(fe[1]) | set(fe[2]) | {26}              # 27 spin-up
beta = set(both) | set(fe[3]) | set(fe[4]) | {2}                # 27 spin-down
assert len(alpha) == len(beta) == 27
det = "".join(("1" if p in beta else "0") + ("1" if p in alpha else "0")
              for p in range(n, 0, -1))
open(os.path.join(os.environ["WORK"], "bs_seed.txt"), "w").write(det + "\n")
EOF
```

Its energy is −326.2938406288 Ha, the second row of the table above. Which two irons
are Fe(II), and which of their 3d orbitals takes the minority electron, changes it a
little: putting it in the first 3d orbital of Fe2 or Fe4 instead gave energies 6 to
35 mHa higher. Grow from the lowest:

```bash
mkdir -p $WORK/bs $WORK/bs27
cd $SBD/examples/gdb
gpus 1 python run_gdb_heatbath.py --device gpu --bit_length 62 \
    --fcidump $LICHAN --subspace-from strings --strings-file $WORK/bs_seed.txt \
    --cutoffs 1e-3,1e-4 --checkpoint_prefix $WORK/bs/ck- --savename $WORK/bs/rung1-
```

and, for comparison, the same ladder from the lowest-27 determinant:

```bash
cd $SBD/examples/gdb
gpus 1 python run_gdb_heatbath.py --device gpu --bit_length 62 \
    --fcidump $LICHAN --subspace-from hf --cutoffs 1e-3,1e-4 --savename $WORK/bs27/rung1-
```

Both start from one determinant, so both run on one GPU.

**What we see.** Same Hamiltonian, same ladder, same GPU:

| Seed | Rung 1e-3 | Rung 1e-4 | Dimension | Gap to DMRG |
|---|---|---|---|---|
| Lowest 27 orbitals | −312.73 | −312.32 | 570,958 | 15 Ha |
| Broken symmetry | −326.936 | **−327.0844** | 926,168 | **163 mHa** |
| *(Parts 4–5: upstream orbitals, lowest 27)* | *−326.715* | *−326.784* | *726,599* | *463 mHa* |

From the broken-symmetry seed the ladder converges smoothly, and at under a million
determinants it is already 260 mHa below where the upstream ladder reached with six
and a half million.

From the lowest-27 seed it fails outright, and the energy **rises** in some rounds:

```
  rung round    dimension             energy           dE    secs
     0     0            1    -303.1129721466                  0.2
     0     1          671    -309.4323742311    -6.319402     1.0
     0     2         1783    -313.1355363609    -3.703162     2.1
     0     3         5879    -312.4010669550    +0.734469     4.5  << ROSE
  WARNING: the energy rose although the subspace grew. The energy of a larger subspace
  can only be lower, so that solve did not find its lowest state, and these energies
  are not trustworthy. Tighten --tolerance, raise --iteration, or start from a better
  determinant (see the README).
     0     4        17761    -313.2714693431    -0.870402    11.1
     0     5        25397    -313.2982423532    -0.026773    15.6
     0     6        32232    -313.1043365765    +0.193906    19.5  << ROSE
     0     7        44323    -312.7257886975    +0.378548    29.8  << ROSE
  WARNING: the energy rose in 3 round(s); see the warning above.
```

A larger subspace can only have a lower ground-state energy, so a rise means the
diagonalization stopped on a higher state, and every number after it is meaningless.
The driver now flags any such round and warns.

**What we take from it.**

- **The starting point matters as much as the method.** Heat-bath selection only
  adds what couples to the current state; from a state with the wrong physics, it
  improves the wrong state.
- **Watch for rises.** In a variational ladder they are never physics.
- **Compare with a reference when one exists,** and when none does, treat a variance
  that will not fall as the warning it is.

**The paper behind it.** Li and Chan built this active space to study the low-energy
spin states of iron–sulfur clusters (*J. Chem. Theory Comput.* **13**, 2681 (2017);
building on Sharma et al., *Nature Chem.* **6**, 927 (2014)). Each iron carries several
unpaired 3d electrons, and the irons' spins couple to each other through the sulfur
atoms. In the ground state the coupling is *antiferromagnetic*: the spins on two irons
point one way and on the other two the opposite way, adding up to zero overall. That is
exactly what the broken-symmetry determinant encodes, and why it is close to the ground
state while a closed-shell determinant is not. They used localized orbitals so that
such a determinant can be written down directly, and solved the model with DMRG, a
method that handles this kind of strongly correlated, nearly one-dimensional structure
well. Their energy, and later more converged DMRG ones, are what we compare against.

### Orbital optimization: the same determinants, better orbitals

The broken-symmetry ladder chose its determinants in Li and Chan's orbitals. Those
orbitals were not chosen with our particular state in mind. Can we rotate them so that
the *same* determinants capture more of the ground state?

**The idea.** The energy of a state can be computed from two small tables, its
*reduced density matrices* (RDMs): the 1-RDM γ (36 × 36) says how electrons are spread
over the orbitals, and the 2-RDM Γ (36⁴) how pairs of electrons are. With the
integrals h and (pq|rs) from the FCIDUMP,

$$E = \sum_{pq} h_{pq}\,\gamma_{pq} + \tfrac12 \sum_{pqrs} (pq|rs)\,\Gamma_{pqrs}.$$

Rotating the orbitals by a matrix U changes the integrals, and so changes this number,
while the RDMs describe the same physical state. An optimizer can therefore search over
rotations U for the one that makes E lowest. Everything involved is small:

| Object | Size here |
|---|---|
| U, the orbital rotation | 36 × 36 (one per spatial orbital of the active space, the same for both spins), orthogonal: 36·35/2 = 630 free parameters |
| one-electron integrals h, and the 1-RDM γ | 36 × 36 |
| two-electron integrals (pq\|rs), and the 2-RDM Γ | 36⁴ ≈ 1.7 million |
| the state ψ itself | one amplitude per determinant in the subspace: 926,168 here |

The many-electron Hamiltonian, which would act on all 8.8 × 10¹⁵ determinants, is never
built: the formula above needs only the integrals and the RDMs, so the search over 630
parameters takes seconds. The rotation mixes only the 36 active orbitals among
themselves, which is why it leaves the exact energy of the active space unchanged.

Then, in the new orbitals, re-diagonalize the same list of determinants: the
coefficients adjust to the new basis, and the energy falls further. Repeat until it
stops moving. This is the method of the qiskit-addon-sqd *optimize orbitals* guide,
applied to our drivers; with a growing subspace in between, it is what the literature
calls orbital-optimized selected CI.

**How, with these tools.** Each cycle is three steps: solve the determinant list with
RDMs, on the GPUs; find the rotation and write the FCIDUMP in the new orbitals, with
[ffsim](https://github.com/qiskit-community/ffsim) on the CPU (`pip install ffsim`);
then the next cycle solves the same list in the new FCIDUMP. Two small scripts do the
non-SBD parts. `export.py` writes a saved state's determinants as a `--detfiles` list;
`rotate.py` checks the RDMs, optimizes the orbitals and writes the rotated FCIDUMP:

```bash
mkdir -p $WORK/oo
cat > $WORK/export.py <<'EOF'
"""Write a saved GDB state's determinants as one --detfiles text file."""
import sys
import sbd
from sbd import analysis
sbd.init(device="cpu")
state = analysis.load_state(sys.argv[1], norb=36, bit_length=62)
with open(sys.argv[2], "w") as f:
    f.write("\n".join(sbd.to_strings(state.det, 62, 72)) + "\n")
print(len(state.det), "determinants ->", sys.argv[2])
EOF
cat > $WORK/rotate.py <<'EOF'
"""Optimize the orbitals for a state's RDMs; write the FCIDUMP in the new orbitals."""
import sys
import numpy as np
import ffsim
from pyscf import tools, ao2mo
src, rdm_file, dst, energy = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
d = tools.fcidump.read(src, verbose=0)
n = d["NORB"]
ham = ffsim.MolecularHamiltonian(d["H1"], ao2mo.restore(1, d["H2"], n), d["ECORE"])
r = np.load(rdm_file)
rdm = ffsim.ReducedDensityMatrix(r["rdm1"], r["rdm2"])
e = rdm.expectation(ham)
assert abs(e - energy) < 1e-6, f"RDM energy {e} does not match the solve's {energy}"
U = ffsim.optimize_orbitals(rdm, ham)
print(f"same RDMs: {e:.10f} -> {rdm.rotated(U).expectation(ham):.10f} in the new orbitals")
new = ham.rotated(U.T.conj())
h1 = new.one_body_tensor.real
assert abs(np.trace(h1) - np.trace(d["H1"])) < 1e-8      # a rotation cannot change it
tools.fcidump.from_integrals(dst, h1, ao2mo.restore(8, new.two_body_tensor.real, n),
                             n, d["NELEC"], nuc=new.constant, ms=0)
EOF
```

`rotate.py` refuses to go on if the RDMs do not reproduce the solve's energy: if they
do not, the rotation it finds means nothing. Now four cycles on the broken-symmetry
1e-4 state, always the same 926,168 determinants, on all 8 GPUs:

```bash
cd $SBD/examples/gdb
python $WORK/export.py $WORK/bs/rung1- $WORK/oo/list926.txt
cp $LICHAN $WORK/oo/orbitals_0.fcidump
solve() {  # solve FCIDUMP LIST RDM-FILE -> prints and returns the energy
    gpus 8 python run_gdb_diag.py --device gpu --b_comm_size 8 --bit_length 62 \
        --tolerance 1e-8 --fcidump $1 --detfiles $2 --rdm_output $3 \
        | grep "Ground state energy" | awk '{print $4}'
}
for k in 0 1 2 3; do
    e=$(solve $WORK/oo/orbitals_$k.fcidump $WORK/oo/list926.txt $WORK/oo/rdm_$k.npz)
    echo "orbitals $k: $e"
    python $WORK/rotate.py $WORK/oo/orbitals_$k.fcidump $WORK/oo/rdm_$k.npz \
        $WORK/oo/orbitals_$((k+1)).fcidump $e
done
solve $WORK/oo/orbitals_4.fcidump $WORK/oo/list926.txt $WORK/oo/rdm_4.npz
```

**What we see.** The broken-symmetry 1e-4 state, 926,168 determinants, always the same
list:

| Orbitals | Same RDMs, rotated | Re-solved | Gain | Gap to DMRG |
|---|---|---|---|---|
| Li–Chan | — | −327.0844069 | — | 161 mHa |
| optimized once | −327.1382841 | −327.1580851 | 74 mHa | 87 mHa |
| twice | −327.1666279 | −327.1719190 | 14 mHa | 73 mHa |
| three times | −327.1755060 | −327.1781436 | 6 mHa | 67 mHa |
| four times | −327.1801616 | −327.1817703 | 4 mHa | 63 mHa |

Each optimization takes seconds on the CPU; each re-solve, seconds on the GPUs. The
RDMs reproduced SBD's energy to all ten digits before every step, and are the same on
one GPU or eight.

**What we take from it.**

- **The orbitals were worth 97 mHa at a fixed dimension.** With under a million
  determinants, the optimized orbitals reached 54 mHa below the 6.3-million-determinant
  state in Li–Chan's orbitals. A better basis makes any fixed subspace — however it was
  chosen — more effective.
- **The gains shrink fast** — 74, 14, 6, 4 mHa — so the orbitals converge for *this*
  list. The 63 mHa still missing needs more determinants, and more determinants would
  in turn prefer slightly different orbitals: the two have to be grown together.
- **Always check the RDMs first.** If their energy does not equal the solve's, the
  rotation the optimizer finds is meaningless.

### Growing and optimizing together

Alternate the two: grow the subspace in the current orbitals, optimize the orbitals
for the larger state, grow again. Each growth step is a heat-bath run started from the
current determinant list (`--subspace-from files`), and each optimization step is the
cycle above:

```bash
cd $SBD/examples/gdb
hb() { gpus 8 python run_gdb_heatbath.py --device gpu --b_comm_size 8 --bit_length 62 "$@"; }
hb --fcidump $WORK/oo/orbitals_4.fcidump --subspace-from files --detfiles $WORK/oo/list926.txt \
   --cutoffs 3e-5 --max_rounds 5 --savename $WORK/oo/A-
python $WORK/export.py $WORK/oo/A- $WORK/oo/listA.txt
for k in 4 5; do
    e=$(solve $WORK/oo/orbitals_$k.fcidump $WORK/oo/listA.txt $WORK/oo/rdmA_$k.npz)
    echo "list A in orbitals $k: $e"
    python $WORK/rotate.py $WORK/oo/orbitals_$k.fcidump $WORK/oo/rdmA_$k.npz \
        $WORK/oo/orbitals_$((k+1)).fcidump $e
done
hb --fcidump $WORK/oo/orbitals_6.fcidump --subspace-from files --detfiles $WORK/oo/listA.txt \
   --cutoffs 1e-5 --max_rounds 3 --max_dim 50000000 --savename $WORK/oo/D-
```

| Step | Orbitals | Dimension | Energy (Ha) | Gap to DMRG |
|---|---|---|---|---|
| broken-symmetry state, 4 optimizations (above) | `orbitals_4` | 926,168 | −327.1817703 | 58 mHa |
| heat-bath at 3e-5, 5 rounds | `orbitals_4` | 2,789,768 | −327.2043403 | 35 mHa |
| 2 more optimizations for that list | `orbitals_6` | 2,789,768 | −327.2101559 | 30 mHa |
| heat-bath at 1e-5, 3 rounds | `orbitals_6` | 12,732,176 | −327.2208561 | 19 mHa |

Gaps are to Li and Chan's DMRG energy, −327.2396 Ha.

The orbital steps get smaller as the subspace grows — 74 mHa on the first state, 2 and
1 mHa on the larger one — because the orbitals are already close to right for it; the
growth steps keep their value, since they add determinants the state needs. The 1e-5
rung needs all 8 GPUs for its memory: at 10 million determinants each rank held about
11 GB of host memory and 9 GB of GPU memory.

### Going further from the broken-symmetry state

Resume the broken-symmetry ladder on all 8 GPUs with a smaller cutoff, as in Part 5,
still in Li and Chan's orbitals:

```bash
cd $SBD/examples/gdb
hb --fcidump $LICHAN --resume_from_prefix $WORK/bs/ck- --cutoffs 3e-5 \
   --checkpoint_prefix $WORK/bs/ck- --savename $WORK/bs/rung2-
```

**What we see.**

```
  rung round    dimension             energy           dE    secs
     2     0       926168    -327.0844069259    +0.000000    26.4  (re-solve of the checkpoint)
     2     1      4187493    -327.1182660379    -0.033859   189.7
     2     2      5380546    -327.1242591111    -0.005993   236.0
     2     3      5816243    -327.1262067810    -0.001948   239.0
     2     4      6053065    -327.1271560372    -0.000949   281.9
     2     5      6198504    -327.1276944128    -0.000538   313.0
     2     6      6288138    -327.1280085803    -0.000314   262.3
     2     7      6342875    -327.1281950277    -0.000186   264.8
  rung 2 at cutoff 3e-05: hit --max_rounds 8 without converging
```

At the same cutoff and almost the same size as the upstream ladder's last rung (6.3
against 6.5 million determinants), the broken-symmetry ladder is 306 mHa lower:
−327.1282 against −326.8218 Ha, about 112 mHa from DMRG instead of 420. Each round
gained about half the one before, and no energy rose.

---

## Part 7 — Extrapolating, and when not to

As the subspace grows, PT2 shrinks toward zero while E_var approaches the exact
energy, and close to convergence E_var is linear in PT2 (Burton and Loos, *J. Chem.
Phys.* **160**, 104102 (2024)). Fitting a line to the last few states and reading it at
PT2 = 0 gives the extrapolated energy. The states must all be in the *same* orbitals.
Here those are the final ones, `orbitals_6.fcidump`: the 1e-5 run above saved its
state in them already (`$WORK/oo/D-`), and the two smaller lists are solved in them
and saved first, each with its energy beside it:

```bash
cd $SBD/examples/gdb
F6=$WORK/oo/orbitals_6.fcidump
for s in "S1 list926" "S2 listA"; do
    read tag list <<< "$s"
    gpus 8 python run_gdb_diag.py --device gpu --b_comm_size 8 --bit_length 62 \
        --tolerance 1e-8 --fcidump $F6 --detfiles $WORK/oo/$list.txt --savename $WORK/oo/$tag- \
        | grep "Ground state energy" | awk '{print $4}' > $WORK/oo/$tag.energy
done
echo "$(energy $WORK/oo/D-)" > $WORK/oo/D.energy
```

Then measure each on all the cores, and fit:

```bash
cd $SBD/examples/analysis
for s in S1 S2 D; do
    cpus python variance.py --fcidump $F6 --state $WORK/oo/$s- --bit_length 62 \
        --energy $(cat $WORK/oo/$s.energy) --batches 10 --json $WORK/oo/$s-var.json
done
python extrapolate.py $WORK/oo/S1-var.json $WORK/oo/S2-var.json $WORK/oo/D-var.json
python extrapolate.py $WORK/oo/S1-var.json $WORK/oo/S2-var.json $WORK/oo/D-var.json --x variance
```

**First, what an untrustworthy fit looks like.** The two broken-symmetry rungs in
Li–Chan's orbitals, measured the same way:

```bash
cd $SBD/examples/analysis
for r in rung1 rung2; do
    cpus python variance.py --fcidump $LICHAN --state $WORK/bs/$r- --bit_length 62 \
        --energy $(energy $WORK/bs/$r-) --batches 10 --json $WORK/bs/$r-var.json
done
python extrapolate.py $WORK/bs/rung1-var.json $WORK/bs/rung2-var.json
```

| Rung | Dimension | E_var (Ha) | Variance (Ha²) | PT2 (mHa) | E_var + PT2 (Ha) |
|---|---|---|---|---|---|
| 1e-4 | 926,168 | −327.0844069 | 0.0601 | −34.8 ± 0.1 | −327.1192 |
| 3e-5 | 6,342,875 | −327.1281950 | 0.0447 | −23.1 ± 0.4 | −327.1513 |

```
  Extrapolated energy:  -327.21508320 Ha
  NOTE: two states fix the line exactly, so there is no error estimate; add a state
  NOTE: the slope is -3.8, steeper than -1: ... add states with smaller PT2 before
  trusting the extrapolation
```

Two points always lie on a line, so nothing here says whether the straight-line form
holds; the PT2 values are large, and the result is unchecked. (On the upstream ladder
of Part 5 the same fit is further off still: there PT2 was missing nine-tenths of the
energy.)

**Then, a trustworthy one.** The three states of the grow-and-optimize run, all in the
final orbitals:

| State | Dimension | E_var (Ha) | Variance (Ha²) | PT2 (mHa) | E_var + PT2 (Ha) |
|---|---|---|---|---|---|
| broken-symmetry list (S1) | 926,168 | −327.1855172 | 0.0207 | −15.9 ± 0.1 | −327.2014 |
| after 3e-5 (S2) | 2,789,768 | −327.2101559 | 0.0136 | −9.0 ± 0.1 | −327.2191 |
| after 1e-5 (D) | 12,732,176 | −327.2208561 | 0.0097 | −6.0 ± 0.2 | −327.2268 |

```
  Extrapolated energy:  -327.24215462 Ha  +- 8.75e-04 (fit)
  Without the largest-PT2 state: -327.24206750 Ha (moves 0.09 mHa)
  Slope: -3.5602  (E_var + PT2 corresponds to a slope of -1)
  chi^2 per degree of freedom: 0.00
  NOTE: the slope is -3.6, steeper than -1: PT2 recovers only about 28% of the
  energy still missing, so E_var + PT2 sits well above the extrapolated energy. The
  states lie on one line and the result is stable to dropping one, so the intercept
  is usable; the slope tends to -1 only as PT2 goes to zero
```

Compare the first table's 926,168-determinant state: the *same* determinants, in
optimized orbitals, have a third of the variance and half the PT2. Better orbitals made
the whole series converge faster, and it is this series that lines up.

**What we see.** The three states lie on one line over a fourteen-fold range of
dimension; dropping the first moves the result by 0.09 mHa. The extrapolated energy,
−327.242 ± 0.001 Ha, falls among the published DMRG values for this active space
(−327.2396 from Li and Chan's release, −327.245 to −327.247 from later, larger DMRG
calculations). Extrapolating against the variance instead
(`extrapolate.py --x variance`) gives −327.254 ± 0.003, with more scatter; the gap
between the two, about 12 mHa, is a fair measure of how much to trust either beyond
its own error bar.

**What we take from it.**

- **The straight line has to be shown, not assumed.** Three or more states, all in the
  same orbitals, lying on one line, and a result that hardly moves when one is dropped:
  `extrapolate.py` reports both checks.
- **A steep slope is not by itself a problem.** It means PT2 captures a steady fraction
  of the missing energy; what matters is that the fraction is steady.
- **Report the extrapolated energy with its uncertainty,** and with a second estimate
  (against the variance) when you can, and say how far the states behind it were from
  convergence: here, the best variational state was still 19 mHa above the
  extrapolated value.

---

## Part 8 — What we learned

| Route | Hamiltonian | Dimension | Energy (Ha) | Gap to DMRG |
|---|---|---|---|---|
| Lowest-27 determinant | upstream | 1 | −326.524 | 723 mHa |
| Upstream subspace (Part 1) | upstream | 59,536 | −326.698 | 549 mHa |
| TPB carryover, best (Part 3a) | upstream | 8,444,836 | −326.769 | 476 mHa |
| GDB heat-bath by hand, 2 rounds (Part 3b) | upstream | 806,932 | −326.785 | 460 mHa |
| Heat-bath, lowest-27 seed, 3e-5 (Part 5) | upstream | 6,503,471 | −326.822 | 425 mHa |
| Broken-symmetry seed, 1e-4 (Part 6) | Li–Chan | 926,168 | −327.084 | 163 mHa |
| Broken-symmetry seed, 3e-5 (Part 6) | Li–Chan | 6,342,875 | −327.128 | ≈ 112 mHa |
| + orbital optimization, same 926k determinants (Part 6) | optimized | 926,168 | −327.186 | ≈ 54 mHa |
| + growth and re-optimization, 1e-5 (Part 6) | optimized | 12,732,176 | −327.221 | ≈ 19 mHa |
| same state, E_var + PT2 (Part 7) | optimized | — | −327.227 | ≈ 13 mHa |
| Extrapolated, 3 points (Part 7) | optimized | — | −327.242 ± 0.001 | within the DMRG range |
| DMRG (literature) | Li–Chan | — | −327.2396 to −327.247 | — |

The upstream, Li–Chan and optimized FCIDUMPs are the same active space in different
orbitals, so every line can be compared with the exact answer, which they share. Gaps
are to Li and Chan's DMRG energy, −327.2396 Ha.

- **How a subspace is represented matters.** A product space (TPB) is natural for
  sampled α and β strings and solves very large dimensions quickly; a determinant list
  (GDB) can hold exactly the determinants chosen, and lets a subspace grow one
  determinant at a time.
- **The starting point and the orbitals matter as much as the size.** A physically
  sensible starting determinant, and orbitals optimized for the state, each gained more
  than millions of extra determinants did — and both apply to a subspace from any
  source.
- **Every energy needs a measure of its quality,** and the measures have limits: PT2
  underestimated the missing energy tenfold on an unconverged state. The variance's
  refusal to fall was the honest signal.
- **Save and resume as you go.** Every driver can carry its result forward, so a long
  calculation can be built rung by rung, analysed between rungs, and continued on
  different hardware.

Which tool for which question:

| Question | Tool |
|---|---|
| What is the energy of this subspace? | `run_sbd_diag.py` (α × β product) or `run_gdb_diag.py` (any list) |
| How far is this state from exact? | `--savename`, then `variance.py` — and a literature reference if there is one |
| How do I improve a subspace step by step? | `--carryover_*` output files, restarted by hand |
| How do I get the best energy without preparing a subspace? | `run_gdb_heatbath.py` from a physically sensible determinant, checkpointed |
| What is my best estimate, with an error bar? | `variance.py` per rung, then `extrapolate.py` — near convergence only |
| How do I make a fixed subspace more effective? | orbital optimization: `--rdm_output`, then ffsim (Part 6) |
| I have bitstrings sampled on a quantum computer | `run_sqd_sbd.py`, or the addon's loop with SBD as the solver (Part 9) |
| How do I keep SQD's subspace small as samples grow? | the addon's `TrimPolicy` with SBD as the solver (Part 9) |
| Can a subspace from one source be refined further? | `--detfiles`, then a heat-bath round or orbital optimization |

---

## Part 9 — The same tools inside the SQD loop

In SQD, a quantum computer provides the configurations. `run_sqd_sbd.py` runs the
loop from a counts file: configuration recovery corrects the samples, subsampling
draws batches of them into subspaces, and SBD's TPB solver diagonalizes each batch.
Everything above applies to the states it produces — saving them, measuring their
variance and PT2, checkpointing and resuming, and refining them further.

There are no samples for this Fe4S4 in the repository, so the loop is shown on the
bundled h2o counts; with your own counts file and FCIDUMP the commands are the same.

**Check the energy against what the subspace must reach.** SBD's Davidson starts, by
default, from the first determinant of the subspace in sorted order. In canonical HF
orbitals that is the HF determinant, a good start. In other orbitals — localized or
optimized ones, as in Part 6 — or in a subspace mixing strongly and weakly coupled
strings, it can have no overlap with the ground state, and Davidson then converges to
a higher state without any error. If you know a subspace contains a good state — for
instance because it includes configurations whose own energy you have computed — check
that the reported energy is at least that low.

```bash
mkdir -p $WORK/sqd
cd $SBD/examples/tpb
export H2O=$SBD/vendor/sbd-upstream/data/h2o/fcidump.txt
sqd() { gpus 8 python run_sqd_sbd.py --device gpu --fcidump $H2O --counts count_dict_h2o.json \
            --adet_comm_size 4 --bdet_comm_size 2 --bit_length 20 \
            --max_iterations 3 --num_batches 2 "$@" | grep -E "Iteration|Batch|WARNING|Total energy|State saved"; }
sqd --max_dim 200 --checkpoint_path $WORK/sqd/ckpt.json --savename $WORK/sqd/h2o-
```

Each iteration solves its batches with SBD on all 8 GPUs; configuration recovery and
subsampling, between the solves, run in Python on one core. Then resume, first with
the same `--max_dim`, then a larger one:

```bash
cd $SBD/examples/tpb
sqd --max_dim 200 --resume_from $WORK/sqd/ckpt.json
sqd --max_dim 400 --resume_from $WORK/sqd/ckpt.json
```

**What we take from it.**

- **The checkpoint is strings, not a state.** `--resume_from` seeds the next run's
  subspace from them; the amplitudes in `--savename` are for `variance.py`.
- **At its cap, a resume cannot improve.** The resumed strings are always kept, so if
  they already fill `--max_dim`, nothing new can enter and every iteration repeats the
  same subspace. The driver warns about this; raise `--max_dim`.
- **Judge the result as in Parts 2 and 6:** `variance.py` on the `--savename` state, and
  a literature reference where one exists.
- **Combine the techniques.** A saved SQD state is a TPB state, so it can be refined
  like the others. `sbd.analysis.load_state` reads it as full determinants, and
  `sbd.to_strings` writes them as a file that `run_gdb_heatbath.py --subspace-from
  strings` (or `run_gdb_diag.py --detfiles`) starts from; a GDB solve of that list with
  `--rdm_output` gives the RDMs for orbital optimization as in Part 6.

### A different SQD policy: Trim SQD

The addon's loop is not tied to this driver. It takes any solver with the right
signature, and SBD's `solve_sci_batch` is one, so the addon's own features run on SBD
unchanged. One is *Trim SQD* (`qiskit_addon_sqd.trim.TrimPolicy`): instead of keeping
only the best batch each iteration, it trims every batch to its highest-weight strings,
pools the survivors, and diagonalizes the pool. The subspace stays small as the number
of samples grows. With SBD as the solver, on the bundled h2o counts:

```bash
cd $SBD/examples/tpb
gpus 1 python - <<'EOF'
import json
from functools import partial
from pyscf import tools, ao2mo
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, StandardPolicy
from qiskit_addon_sqd.trim import TrimPolicy
from sbd.sbd_solver import solve_sci_batch

fcidump = "../../vendor/sbd-upstream/data/h2o/fcidump.txt"   # run from examples/tpb
d = tools.fcidump.read(fcidump)
norb, nel = d["NORB"], d["NELEC"]
counts = json.load(open("count_dict_h2o.json"))
bits = BitArray.from_counts(counts, num_bits=2 * norb)
solver = partial(solve_sci_batch, sbd_config={"eps": 1e-8, "bit_length": 20},
                 fcidump_path=fcidump)
for policy in (StandardPolicy(), TrimPolicy(trim_ratio=0.5, max_strings_per_trim=250)):
    r = diagonalize_fermionic_hamiltonian(
        d["H1"], ao2mo.restore(1, d["H2"], norb), bits, samples_per_batch=300,
        norb=norb, nelec=(nel // 2, nel // 2), num_batches=5, max_iterations=5,
        sci_solver=solver, symmetrize_spin=True, seed=24, policy=policy)
    print(type(policy).__name__, r.energy + d["ECORE"],
          len(r.sci_state.ci_strs_a), "x", len(r.sci_state.ci_strs_b))
EOF
```

This is the addon's loop driven from Python, in one process, with SBD solving on one
GPU.

| Policy | Final subspace | Energy (Ha) |
|---|---|---|
| `StandardPolicy` | 275 × 275 = 75,625 | −76.2359466 |
| `TrimPolicy(trim_ratio=0.5, max_strings_per_trim=250)` | 114 × 114 = 12,996 | −76.2205299 |

Trim SQD's final subspace is six times smaller, 15 mHa higher in energy. Its settings
trade the two; compare policies at matched subspace size to tell them apart, and use
`variance.py` on the saved states to see how close each is to converged. The addon's
[Trim SQD guide](https://qiskit.github.io/qiskit-addon-sqd/guides/trim_sqd.html)
explains the method.

### Not yet: seeding SQD with a refined subspace

A natural next step is to give SQD a head start: carry the strings of a refined state
— the broken-symmetry, orbital-optimized state of Part 6, say — into every subspace
through `include_configurations` (a `--resume_from` checkpoint written from that
state), and let the samples add to it. The strings do get in; the obstacle is the
solver's start. With the default start, the first sorted determinant, Davidson can
converge on a state confined to the sampled strings and report its energy, far above
what the included strings alone give; a random start (`init 1`) avoids that here, but
on canonical h2o it converges to a higher state instead. The robust choice is to start
Davidson on a chosen determinant, which SBD supports but this package does not yet
expose to Python, so this workflow is not shown. Until it is, check any SQD energy with
included configurations against the energy of those configurations alone.

## Appendix A — What each driver saves, and how to continue from it

| Driver | Saves | Continue with |
|---|---|---|
| `run_sbd_diag.py` | `--savename` (state), `--carryover_adetfile` (next α strings) | `--adetfile`, `--loadname` |
| `run_gdb_diag.py` | `--savename` (state), `--carryover_detfile` (next determinants) | `--detfiles`, `--loadname` |
| `run_gdb_heatbath.py` | `--checkpoint_prefix` every round, `--savename` at the end | `--resume_from_prefix` |
| `run_sqd_sbd.py` | `--checkpoint_path` (strings, occupancies), `--savename` (best state) | `--resume_from` |

## Appendix B — Traps

- An energy from one FCIDUMP is not comparable with one from another, even for the
  same molecule.
- `--bit_length` must be the same for a run and for anything that reads its files.
- A one-determinant start must run on one GPU; resume on more once it has grown.
- `variance.py` runs on CPU; give it many ranks with one thread each.
- PT2 is a rough guide while it is large; trust extrapolations only once it is small.
- SQD resumed at `--max_dim` stays where it was.
- SBD's default Davidson start is the first sorted determinant: fine in canonical HF
  orbitals, unreliable in others. Check an energy against what the subspace must reach.
