# Analysis examples — energy variance and PT2 of a saved state

> The check step of both workflows in [`../README.md`](../README.md).

A diagonalization gives the energy of the best state *within its subspace*. Two numbers
say how far that state is from the exact one:

- **Energy variance** $\sigma^2 = \sum_{a \notin V} |\langle a|H|\psi\rangle|^2$, summed
  over the determinants $a$ outside the subspace $V$. For an eigenvector of the
  subspace problem it is the full energy variance $\langle H^2\rangle - E^2$. It is
  zero for an exact eigenstate, so it measures convergence.
- **Epstein–Nesbet PT2** $E_{\mathrm{PT2}} = \sum_{a \notin V} |\langle a|H|\psi\rangle|^2
  / (E - H_{aa})$, the second-order correction from those same determinants.
  $E + E_{\mathrm{PT2}}$ is usually much closer to the exact energy than $E$ alone.

## The tools

| Tool | Does |
|---|---|
| `variance.py` | the variance and PT2 of one saved state |
| `extrapolate.py` | fits a series of states' energies against PT2, to estimate the exact energy |

`variance.py` takes a state saved with `--savename` by any of the solvers, and calls
upstream SBD's matching evaluator from Python:

| State saved by | Give | Upstream evaluator | Methods |
|---|---|---|---|
| TPB or GDB, from a restricted or an unrestricted (UHF) FCIDUMP | `--fcidump` | [`gdb-stat-evaluator`](https://github.com/r-ccs-cms/sbd/tree/main/apps/gdb-stat-evaluator); a TPB state is passed to it as full determinants | stochastic; or exact, with PySCF, for a small restricted system |
| CAOP (`../caop/run_caop_diag.py`) | `--hamfile`, `--basisfiles` | [`caop-stat-evaluator`](https://github.com/r-ccs-cms/sbd/blob/main/apps/caop-stat-evaluator/README.md) | stochastic |

It always runs on the CPU, with MPI and OpenMP, on any number of ranks: it need not use
the ranks, or the device, of the run that saved the state. The same calculations are
`sbd.analysis.variance()` and `sbd.analysis.caop_variance()` in Python.

## Check a molecular state (TPB or GDB)

**1. Diagonalize with `--savename`, and give `--bit_length` explicitly.** The analysis
must use the same value, and the files do not record it. Note the energy the run
prints (`Ground state energy:`): the analysis takes it as `--energy`. Converge it
tightly (`--tolerance`): the variance counts only what $H|\psi\rangle$ puts outside the
subspace, not a residual an unconverged solve leaves inside it.

```bash
# TPB (examples/tpb): h2o, 275 x 275 = 75,625 determinants, on a 2 x 2 rank grid
cd ../tpb
mpirun -np 4 python run_sbd_diag.py --adet_comm_size 2 --bdet_comm_size 2 \
    --bit_length 20 --tolerance 1e-8 --savename /path/to/run1/h2o-
#   writes /path/to/run1/h2o-000000 ... h2o-000003  (one per alpha x beta block)

# GDB (examples/gdb): Fe4S4, 59,536 determinants, sharded over 4 GPUs
cd ../gdb
mpirun -np 4 python run_gdb_diag.py --device gpu --b_comm_size 4 \
    --bit_length 62 --savename /path/to/run2/fe4s4-
#   writes /path/to/run2/fe4s4-000000.bin ... fe4s4-000003.bin  (one per shard)
```

**2. Analyse it,** with the same FCIDUMP and `--bit_length`, and step 1's energy:

```bash
cd ../analysis

# h2o
OMP_NUM_THREADS=2 mpirun -np 32 python variance.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --state /path/to/run1/h2o- --bit_length 20 --energy -76.2359466308   # step 1's energy

# Fe4S4
OMP_NUM_THREADS=2 mpirun -np 32 python variance.py \
    --fcidump ../../vendor/sbd-upstream/apps/chemistry_gdb_selected_basis_diagonalization/fcidump_Fe4S4.txt \
    --state /path/to/run2/fe4s4- --bit_length 62 --energy <the Fe4S4 run's energy>
```

Output for the h2o run:

```
State:   /path/to/run1/h2o-  (TPB, 4 file(s), checked as one run)
           /path/to/run1/h2o-000000  (156,792 bytes)
           /path/to/run1/h2o-000001  (155,672 bytes)
           /path/to/run1/h2o-000002  (155,672 bytes)
           /path/to/run1/h2o-000003  (154,560 bytes)
Ranks:   32
Energy:  -76.2359466308  (given)
----------------------------------------------------------------------
Method:        stochastic  (10 batches x 20000 samples, cutoff 0, seed 1729)
Dimension:     75,625
Variance:      9.495301e-02  +- 4.26e-04  Ha^2
PT2:           -7.625512e-03  +- 2.19e-05  Ha
E_var:         -76.2359466308  Ha
E_var + PT2:   -76.2435721431  +- 2.19e-05  Ha
```

The published FCI energy for this system is −76.24377680
(`vendor/sbd-upstream/data/h2o/README.md`): $E_{\mathrm{var}}$ is 7.8 mHa above it, and
$E_{\mathrm{var}} + E_{\mathrm{PT2}}$ 0.2 mHa. `--json FILE` also writes the result,
the settings and the per-batch values.

**Keep one run's files per prefix.** A run overwrites only its own shards, so a later
run with the same prefix but fewer shards leaves the earlier run's extra files behind.
`variance.py` checks that the files are one run's — one solver, consecutive shard
numbers, sizes matching their headers, no determinant in two files, a total norm of 1 —
and stops with an error otherwise. It lists the files it used.

## Check a CAOP state

A state saved by `../caop/run_caop_diag.py --savename` takes `--hamfile` and the basis
files it was solved over in place of `--fcidump`, and the run's settings, which the
files do not record. After the first two steps of
[`../caop/README.md#greens-function`](../caop/README.md#greens-function), which write
`hub2.txt`, `hub2_basis.txt` and the state `gs-`:

```bash
mpirun -np 4 python variance.py --hamfile ../caop/hub2.txt \
    --basisfiles ../caop/hub2_basis.txt --sites 4 --state ../caop/gs- --shards 4 \
    --bit_length 20 --energy <E0 from the run>
```

`--shards` is the solve's `--b_comm_size`; give `--complex` if it used `--complex`. A
mismatch with the saved state raises an error. `--energy` may be left out with
`--observable variance`.

## Estimate the exact energy from a series

As a subspace grows, PT2 goes to zero and $E_{\mathrm{var}}$ to the exact energy, and
over the last few states $E_{\mathrm{var}}$ is close to linear in PT2. `extrapolate.py`
fits that line and reports its intercept at PT2 = 0, from the `--json` files of
`variance.py` runs on a series of states, such as the rounds of a heat-bath ladder
(`../gdb/run_gdb_heatbath.py`):

```bash
for r in round1 round2 round3; do
    OMP_NUM_THREADS=2 mpirun -np 32 python variance.py --fcidump F --state $r- \
        --bit_length 62 --energy <E of $r> --json $r-var.json
done
python extrapolate.py round1-var.json round2-var.json round3-var.json
```

It reports the intercept's statistical error and how far it moves when the least linear
state is left out, which with few states is the better guide. It needs at least two
states, and three or more for either number to mean much. `--x variance` extrapolates
against the variance instead, a check on the PT2 fit. `--point E_VAR,PT2[,ERR]` adds a
state from elsewhere, for the PT2 fit only.

To compare subspaces built different ways, such as two seeds or two expansion
schemes, compare them at about the same dimension and by their variance and
$E_{\mathrm{var}} + E_{\mathrm{PT2}}$, not by $E_{\mathrm{var}}$ alone: a larger subspace
lowers $E_{\mathrm{var}}$ whatever its quality.

## In the same job, from Python

The analysis can also run at the end of a diagonalization script, in the same MPI job.
Save the state with `savename`, then call `sbd.analysis.variance()` on the same prefix;
every rank must make the call. For example, at the end of `run_sbd_diag.py` or
`run_gdb_diag.py`, after the diagonalization returned `results`:

```python
from sbd import analysis

if args.savename:
    v = analysis.variance(args.savename, sbd.LoadFCIDump(args.fcidump),
                          results['energy'], bit_length=args.bit_length)
    if sbd.get_rank() == 0:
        print(f"Variance:    {v.variance:.6e} +- {v.variance_error:.2e} Ha^2")
        print(f"E_var + PT2: {v.energy_pt2:.10f} +- {v.pt2_error:.2e} Ha")
```

For a CAOP state, `sbd.analysis.caop_variance()` takes the solve's Hamiltonian file,
basis files and settings:

```python
from mpi4py import MPI

from sbd import analysis

nranks = MPI.COMM_WORLD.Get_size()
config.b_comm_size = nranks           # one saved file per rank
ground = sbd.caop_diag_from_files("hub2.txt", ["hub2_basis.txt"], config,
                                  savename="gs-")
v = analysis.caop_variance("gs-", "hub2.txt", ["hub2_basis.txt"], ground["energy"],
                           sites=config.system_size, bit_length=config.bit_length,
                           shards=nranks)          # complex=True if the solve used it
```

The state still goes through the saved files, since SBD's solvers do not return their
eigenvector. Against a separate job:

- **Layout.** The analysis runs on the diagonalization's ranks and threads, and prefers
  many ranks with few threads; a layout chosen for GPUs is often a poor one for it.
- **Device.** It always runs on the CPU, so the GPUs sit idle while it runs. After a
  `gpu-omp` solve it uses the CPU code inside the offload module by itself, so the two
  backends are never loaded together.
- **Memory.** The diagonalization's data is still allocated, so the two together must
  fit.
- **Failure.** If the analysis fails or runs out of time, the saved state is on disk
  and can be analysed again separately.

## How the stochastic estimate works

Both sums run over every determinant outside the subspace that $H$ connects to the
state — far too many to list for a real system — so the default method **samples**:

1. Draw parent determinants $i$ from the state at random, with probability
   proportional to $|c_i|$.
2. Expand each draw into the single and double excitations it connects to outside the
   subspace, and accumulate each one's contribution, reweighted for the sampling.
3. Combine the contributions per outside determinant into estimates of $\sigma^2$ and
   $E_{\mathrm{PT2}}$.

The estimate is unbiased, but any one is random. **Batches measure how far off it is:**
steps 1–3 repeat independently `--batches` times, and the result is the mean over
batches with its standard error. The error bar shrinks as
$1/\sqrt{\text{batches} \times \text{samples}}$; at least 5–10 batches are needed for
the error bar itself to be trustworthy.

## Settings

**`--method`** (TPB and GDB states):

| `--method` | How | Use for |
|---|---|---|
| `stochastic` | The sampled, batched estimate above, over all ranks | Anything; the only option for large systems and UHF FCIDUMPs |
| `exact` | PySCF builds $H\lvert\psi\rangle$ over the full CI space, in one process; restricted FCIDUMPs only | Small systems, and checking the stochastic numbers |
| `auto` (default) | `exact` when the full CI space has at most `--exact_limit` (5,000,000) determinants and the FCIDUMP is restricted, else `stochastic` | |

h2o/cc-pVDZ above (24 orbitals, 5 + 5 electrons) has a full CI space of
$\binom{24}{5}^2 \approx 1.8 \times 10^9$, so `auto` chose `stochastic`.

**`--batches` and `--samples`** (defaults 10 and 20,000) set the error bar:

- **Accuracy** depends on their product: halving the error bar takes four times as
  many draws in total.
- **Memory** depends on `--samples` only: a batch holds every contribution its draws
  produce, which grows quickly with the number of orbitals and electrons, and is spread
  over the ranks. Batches run one after another, so more batches cost time, not memory.
- **If memory is short,** use fewer samples and more batches, or more ranks.

**`--cutoff`** (default 0, unbiased). Heat-bath screening drops small parent–child
contributions. It is faster, but biases *both* results low, more so as the state
improves; check any cutoff against a smaller one first.

**Ranks and threads.** Prefer many ranks with few threads: most of the work is spread
across ranks but not threaded.

**`--seed` and `--first_batch`.** A seed makes a run reproducible for the same rank
count. Each batch's draws also depend on its ID, counted from `--first_batch` (default
0). To add batches to an earlier run, rerun with the same seed and `--first_batch` past
the batches it has; the `--json` files of both can be pooled.

**`--observable`** (`both`, `variance` or `pt2`) estimates only what you ask for.

**`--minimum_abs_denominator`** (default 0) stops with an error if any PT2 denominator
$|E - H_{aa}|$ is at or below it, rather than returning a PT2 dominated by one
near-degenerate determinant.

## Notes

- **Energy:** `--energy` is required, as upstream's evaluators require
  `--reference-energy`. It is the PT2 reference, so it must be this state's energy, the
  one the run that saved it reported; it is not recomputed. To recompute it from the
  files, for instance to check that a state and an FCIDUMP belong together, use
  `sbd.analysis.state_energy()` on the `cpu` or `gpu` backend.
- **Memory:** the state is spread evenly over the ranks, whatever the number of files.
- **Multi-node:** every rank must be able to read the state's files: put them on a
  shared filesystem.
- **When to run it:** the analysis can cost much more than the diagonalization, so run
  it on the state you want to assess, not every iteration.
