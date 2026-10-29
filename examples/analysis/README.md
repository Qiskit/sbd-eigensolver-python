# Analysis examples — energy variance and PT2 of a saved state

A diagonalization gives the energy of the best state *within its subspace*. Two numbers
say how far that state is from the exact one:

- **Energy variance** $\sigma^2 = \sum_{a \notin V} |\langle a|H|\psi\rangle|^2$, summed
  over the determinants $a$ outside the subspace $V$. For an eigenvector of the
  subspace problem it is the full energy variance $\langle H^2\rangle - E^2$. It is
  zero for an exact eigenstate, so it measures convergence; plotting $E$ against
  $\sigma^2$ over a series of subspaces and extrapolating to $\sigma^2 = 0$ estimates
  the exact energy.
- **Epstein–Nesbet PT2** $E_{\mathrm{PT2}} = \sum_{a \notin V} |\langle a|H|\psi\rangle|^2
  / (E - H_{aa})$, the second-order correction from those same determinants.
  $E + E_{\mathrm{PT2}}$ is usually much closer to the exact energy than $E$ alone.

`variance.py` computes both for a state saved by either SBD driver, TPB or GDB, on any
number of MPI ranks: the analysis need not use the same ranks, or the same device, as
the run that produced the state.

## How it is computed, and why in batches

Both sums run over every determinant outside the subspace that $H$ connects to the
state — far too many to list for a real system. So the default method **samples**:

1. Draw parent determinants $i$ from the state at random, with probability
   proportional to $|c_i|$.
2. Expand each draw into the single and double excitations it connects to outside the
   subspace, and accumulate each one's contribution, reweighted for the sampling.
3. Combine the contributions per outside determinant into estimates of $\sigma^2$ and
   $E_{\mathrm{PT2}}$.

This estimate is unbiased — on average it equals the exact value — but any single one
is random, and on its own says nothing about how far off it is. **Batches are how the
method measures that.** It repeats steps 1–3 independently `--batches` times, with
fresh random draws each time, and reports the mean over batches with its standard
error (the spread between batches divided by $\sqrt{\text{batches}}$). The error bar
shrinks as $1/\sqrt{\text{batches} \times \text{samples}}$, so more batches or more
samples per batch both sharpen it; at least 5–10 batches are needed for the error bar
itself to be trustworthy.

For small systems an **exact** method is also available (PySCF, single process, no
sampling, so no batches and no error bar).

## Workflow

**1. Diagonalize with `--savename`, and give `--bit_length` explicitly.** The analysis
must use the same value, and the files do not record it.

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

**Keep one run's files per prefix.** A run writes one file per shard and overwrites
only those, so a later run with the same prefix but fewer shards leaves the earlier
run's extra files behind. Use a fresh prefix (or directory) per run. `variance.py`
checks that the files it finds are one run's — one solver's kind, consecutive shard
numbers, sizes matching their headers, no determinant in two files, and a total
norm of 1 (SBD saves normalized states) — and stops with an error otherwise, rather
than combining them. It always lists the files it used.

Converge the diagonalization tightly (`--tolerance`). The variance counts only what
$H|\psi\rangle$ puts outside the subspace; any residual left inside it by an
unconverged solve is not included.

**2. Analyse the state with `variance.py`,** giving the same FCIDUMP and `--bit_length`:

```bash
cd ../analysis

# h2o
mpirun -np 32 -x OMP_NUM_THREADS=2 python variance.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --state /path/to/run1/h2o- --bit_length 20

# Fe4S4
mpirun -np 32 -x OMP_NUM_THREADS=2 python variance.py \
    --fcidump ../../vendor/sbd-upstream/apps/chemistry_gdb_selected_basis_diagonalization/fcidump_Fe4S4.txt \
    --state /path/to/run2/fe4s4- --bit_length 62
```

Output for the h2o run above:

```
State:   /path/to/run1/h2o-  (TPB, 4 file(s), checked as one run)
           /path/to/run1/h2o-000000  (156,792 bytes)
           /path/to/run1/h2o-000001  (155,672 bytes)
           /path/to/run1/h2o-000002  (155,672 bytes)
           /path/to/run1/h2o-000003  (154,560 bytes)
Ranks:   32
Energy:  -76.2359466308  (computed from the state)
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
$E_{\mathrm{var}} + E_{\mathrm{PT2}}$ 0.2 mHa.

`--json FILE` also writes the result, the settings and the per-batch values. The same
calculation is available in Python as `sbd.analysis.variance()`; see the API docs.

## Example: one Fe4S4 subspace through TPB and GDB

Upstream SBD ships one Fe4S4 subspace in both solvers' formats: `AlphaDets.txt` (244
strings, used as alpha and beta, so $244^2 = 59{,}536$ determinants) for TPB, and
`det0.txt`–`det3.txt` (the same 59,536 determinants) for GDB. This example solves it
with each, has each solver build a larger subspace from its state, solves that, and
reads both resulting states' variance and PT2.

The two solvers expand a state differently:

- **TPB** keeps a product of alpha and beta strings, so it grows the subspace a string
  at a time. `--carryover_type 3` keeps the strings of the dominant determinants and
  adds their single excitations.
- **GDB** keeps a list of determinants, so it grows it a determinant at a time.
  `--carryover_type 2` adds the determinants the heat-bath criterion connects to the
  state, the state's own determinants included.

The settings below give two next subspaces of about the same size, so their energies and
variances can be read side by side.

```bash
SBD=/path/to/sbd-eigensolver-python
APPS=$SBD/vendor/sbd-upstream/apps
FCIDUMP=$APPS/chemistry_gdb_selected_basis_diagonalization/fcidump_Fe4S4.txt
mkdir -p /path/to/fe4s4/tpb /path/to/fe4s4/gdb

# TPB: solve, and write the expanded strings
cd $SBD/examples/tpb
mpirun -np 4 python run_sbd_diag.py --fcidump $FCIDUMP \
    --adetfile $APPS/chemistry_tpb_selected_basis_diagonalization/AlphaDets.txt \
    --adet_comm_size 2 --bdet_comm_size 2 --bit_length 20 --tolerance 1e-8 \
    --carryover_type 3 --carryover_threshold 1e-2 \
    --carryover_adetfile /path/to/fe4s4/tpb/a.txt --carryover_bdetfile /path/to/fe4s4/tpb/b.txt
# solve the expanded subspace, saving the state
mpirun -np 4 python run_sbd_diag.py --fcidump $FCIDUMP --symmetrize_spin 0 \
    --adetfile /path/to/fe4s4/tpb/a.txt --bdetfile /path/to/fe4s4/tpb/b.txt \
    --adet_comm_size 2 --bdet_comm_size 2 --bit_length 20 --tolerance 1e-8 \
    --savename /path/to/fe4s4/tpb/next-

# GDB: solve (det0-3.txt are the driver's defaults), and write the expanded list
cd $SBD/examples/gdb
mpirun -np 4 python run_gdb_diag.py --device gpu --b_comm_size 4 --fcidump $FCIDUMP \
    --bit_length 62 --tolerance 1e-8 --carryover_type 2 --heatbath_cutoff 2e-4 \
    --carryovername /path/to/fe4s4/gdb/list-
# solve the expanded subspace from the files the first run lists, saving the state
mpirun -np 4 python run_gdb_diag.py --device gpu --b_comm_size 4 --fcidump $FCIDUMP \
    --bit_length 62 --tolerance 1e-8 --savename /path/to/fe4s4/gdb/next- \
    --detfiles /path/to/fe4s4/gdb/list-000000.txt,/path/to/fe4s4/gdb/list-000001.txt,/path/to/fe4s4/gdb/list-000002.txt,/path/to/fe4s4/gdb/list-000003.txt

# Variance and PT2 of both states
cd $SBD/examples/analysis
mpirun -np 32 -x OMP_NUM_THREADS=2 python variance.py --fcidump $FCIDUMP \
    --state /path/to/fe4s4/tpb/next- --bit_length 20 --samples 5000 --batches 5 --json tpb.json
mpirun -np 32 -x OMP_NUM_THREADS=2 python variance.py --fcidump $FCIDUMP \
    --state /path/to/fe4s4/gdb/next- --bit_length 62 --samples 5000 --batches 5 --json gdb.json
```

Measured on an 8× H100 server (Ha):

| | dimension | $E_{\mathrm{var}}$ | variance (Ha²) | PT2 | $E_{\mathrm{var}} + E_{\mathrm{PT2}}$ |
|---|---|---|---|---|---|
| Shared start, either solver | 59,536 | −326.6982519 | 7.46e-02 ± 1.0e-03 | −6.61e-02 ± 1.2e-03 | −326.7644 ± 0.0012 |
| TPB, type 3, threshold 1e-2 | 204,304 (452 × 452) | −326.7074213 | 6.75e-02 ± 1.8e-03 | −5.78e-02 ± 2.1e-03 | −326.7652 ± 0.0021 |
| GDB, type 2, cutoff 2e-4 | 206,105 | −326.7583052 | 6.90e-02 ± 7.4e-04 | −4.89e-02 ± 5.8e-04 | −326.8073 ± 0.0006 |

Both solvers give the same energy for the shared start, as they must for the same
determinants. The start's row was computed with the default 10 × 20,000 samples, the
other two with 5 × 5,000, so their error bars are wider.

Things to note when reading the table:

- **The variance has not converged** in any row. It is of order $10^{-2}$ Ha² throughout,
  so all three states are far from the exact one, and PT2, which assumes a small
  correction, is only a rough guide here.
- **Each expansion is one setting of a tunable step.** The threshold and cutoff above
  were chosen to match the two dimensions; other values give other sizes, and TPB's
  size moves in larger jumps, since adding a string adds a whole row and column of the
  product.
- To carry on from either state, repeat its step: the strings or list a run writes are
  the next run's input. `extrapolate.py` then fits a series of such states, from the
  `--json` files, to estimate the energy at zero PT2.
- **GDB's placement can be tuned.** The GDB runs above use the default
  `--determinant_distribution equal-bra-a` (also upstream's default), which gives each
  rank an equal number of distinct alpha strings. A heat-bath expansion puts many
  determinants on a few alpha strings, so the expanded subspace can land unevenly across
  the ranks. The grid schemes, `grid-cyclic` and `grid-cyclic-balanced`, spread it more
  evenly here. A scheme changes only how the work is spread, not the energy. Which one is
  fastest depends on the subspace and the rank count; see
  [Placement across `b_comm`](../gdb/README.md#placement-across-b_comm) for what each
  one balances.

## Doing it in the same run

The analysis can also run at the end of a diagonalization script, in the same MPI job,
instead of as a separate launch. Save the state with `savename`, then call
`sbd.analysis.variance()` on the same prefix. For example, at the end of
`run_sbd_diag.py` or `run_gdb_diag.py`, after the diagonalization has returned
`results`:

```python
from sbd import analysis

if args.savename:
    v = analysis.variance(args.savename, sbd.LoadFCIDump(args.fcidump),
                          results['energy'], bit_length=args.bit_length)
    if sbd.get_rank() == 0:
        print(f"Variance:    {v.variance:.6e} +- {v.variance_error:.2e} Ha^2")
        print(f"E_var + PT2: {v.energy_pt2:.10f} +- {v.pt2_error:.2e} Ha")
```

Every rank must make the call, since it is collective. Passing `results['energy']`
skips recomputing the energy; it must be the energy of the state just saved.

The state still goes through the files `savename` writes: SBD's solvers do not hand
their eigenvector back, so the analysis reads it from disk even in the same job.

Things to weigh against a separate job:

- **Layout.** The analysis runs on the diagonalization's ranks and threads. A layout
  chosen for the diagonalization — one rank per GPU with many threads, say — is often
  a poor one for the analysis, which prefers many ranks with few threads.
- **Device.** The diagonalization can use any device (`cpu`, `gpu`, `gpu-omp`); the
  analysis picks a suitable backend itself. It always runs on CPU, so after a GPU
  diagonalization the GPUs sit idle while it runs. After a `gpu` (Thrust) run it uses
  the CPU backend, so the package must include it — the default build does; a
  Thrust-only build (`SBD_BUILD_BACKEND=gpu`) raises an error instead.
- **Memory.** The diagonalization's data is still allocated when the analysis starts,
  so the two together must fit; a separate job starts clean.
- **Failure.** If the analysis fails or runs out of time, the job may end before
  anything else after it; the saved state is already on disk, so it can be analysed
  again separately.

A separate job keeps the two independent, and lets you rerun the analysis with
different settings without repeating the diagonalization.

## Choosing the method

| `--method` | How | Use for |
|---|---|---|
| `stochastic` | The sampled, batched estimate above (SBD's statistical evaluator). MPI over all ranks, CPU only. | Anything; the only option for large systems |
| `exact` | PySCF builds $H\lvert\psi\rangle$ over the full CI space, in one process. | Small systems, and checking the stochastic numbers |
| `auto` (default) | `exact` when the full CI space has at most `--exact_limit` (5,000,000) determinants, else `stochastic` | |

h2o/cc-pVDZ above (24 orbitals, 5 + 5 electrons) has a full CI space of
$\binom{24}{5}^2 \approx 1.8 \times 10^9$, so `auto` chose `stochastic`. The exact
method stores vectors over the whole space; a few million determinants is a sensible
ceiling.

## Stochastic settings

**`--batches` and `--samples`** (defaults 10 and 20,000) set the error bar, as above.
The defaults are a reasonable start; adjust from there:

- **Accuracy** depends on the product: the error bar shrinks as
  $1/\sqrt{\text{batches} \times \text{samples}}$, so halving it takes four times as many
  draws in total.
- **Memory** depends on `--samples` only. A batch holds every contribution its draws
  produce until it is summed, so its memory grows with `--samples` and with how many
  excitations each draw has — which grows quickly with the number of orbitals and
  electrons — and is spread over the ranks. Batches run one after another, freeing
  that memory in between, so more batches cost time, not memory.
- **If memory is short, use fewer samples and more batches**, keeping their product
  for the same accuracy; or use more ranks, which share a batch's memory.
- **To reach a target accuracy,** start from the defaults and add batches until the
  error bar is small enough for your purpose.

**`--cutoff` (default 0).** Heat-bath screening drops small parent–child contributions.
It is faster, but biases *both* results low, and the bias grows as the state improves.
The default, 0, is unbiased. The safe cutoff depends on the system and the state, so if
you use one, check it against a smaller one (or 0) first.

**Ranks and threads.** Prefer many ranks with few threads: most of the work is spread
across ranks but not threaded. The analysis is a separate launch so it can use a
different layout from the diagonalization, and it always runs on CPU, even for a state
diagonalized on GPUs.

**Seeds.** `--seed` makes a run reproducible for the same rank count; a different rank
count draws different samples.

## Notes

- **Memory:** the state is spread evenly over the ranks, whatever the number of files;
  nothing is gathered onto one rank. Computing the state's energy writes a temporary
  copy of it, removed afterwards.
- **Multi-node:** every rank must be able to read the state's files, as for the
  FCIDUMP: put them on a shared filesystem.
- **Energy:** by default it is computed from the saved state, which also checks that
  the state and the FCIDUMP belong together — it should equal what the diagonalization
  reported. `--energy` overrides it; it must be this state's energy, since it is the
  PT2 reference.
- **Restricted FCIDUMPs only** for the exact method.
- **When to run it:** the analysis can cost much more than the diagonalization that
  produced the state, so run it on the state you want to assess, not every iteration.
