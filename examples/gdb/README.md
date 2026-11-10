# GDB examples — general determinant basis

GDB spans the subspace with the determinants it is given, rather than with the
Cartesian product of an alpha and a beta list that TPB uses. TPB's dimension is
`|adet| x |bdet|`; GDB's is exactly the number of determinants passed. That is what
makes it the right solver for an arbitrary sparse subspace — a set of sampled
bitstrings used *as sampled*, with no product completion.

For TPB and the SQD loops, see [`../tpb/README.md`](../tpb/README.md); for how they fit
with the variance check, the [workflows](../README.md#molecules-tpb-or-gdb).

The FCIDUMP may be restricted (RHF, ROHF) or unrestricted (UHF): `sbd.tpb_diag`,
`sbd.gdb_diag` and the drivers read its layout and use the matching build of SBD, with no
option to set. The layout, how to make a UHF file and the limits are in
[the TPB README](../tpb/README.md#unrestricted-uhf-fcidumps). For example, heat-bath
selection from the UHF determinant of a file made there:

```bash
python ../tools/make_uhf_fcidump.py nh2 --outdir nh2      # requires PySCF
python run_gdb_heatbath.py --fcidump nh2/nh2_uhf.fcidump \
    --subspace-from hf --cutoffs 1e-2,1e-3,1e-4,1e-5
```

## The default data

Both drivers default `--fcidump` and `--detfiles` to upstream's own GDB app data under
`vendor/sbd-upstream/apps/chemistry_gdb_selected_basis_diagonalization/`: the Fe4S4
FCIDUMP at 36 orbitals, and its four `det0.txt`-`det3.txt` holding 14,884 determinants
each, 59,536 in total. **Any command below that passes neither flag runs exactly that
case** — the determinants come from those four files, not from nowhere. Upstream
publishes no reference energy for it, so treat it as a self-consistency benchmark
rather than a validation target.

## run_gdb_diag.py — standalone GDB diagonalization

```bash
# The default case: Fe4S4, upstream's four det files, 59,536 determinants
python run_gdb_diag.py

# The same, with the defaults written out: the syntax for your own data. The files
# are concatenated in order, so together they must be sorted and disjoint
GDB=../../vendor/sbd-upstream/apps/chemistry_gdb_selected_basis_diagonalization
python run_gdb_diag.py --fcidump $GDB/fcidump_Fe4S4.txt \
    --detfiles $GDB/det0.txt,$GDB/det1.txt,$GDB/det2.txt,$GDB/det3.txt

# Shard the basis over 4 ranks
OMP_NUM_THREADS=8 mpirun -np 4 python run_gdb_diag.py --b_comm_size 4

# On 4 GPUs: Thrust needs --b_comm_size
mpirun -np 4 python run_gdb_diag.py --device gpu --b_comm_size 4

# Start Davidson from this determinant instead of the first in sorted order; it
# must be in the subspace. Here, the first line of the default data
python run_gdb_diag.py --initial_det "$(head -1 $GDB/det0.txt)"
```

[MPI decomposition](#mpi-decomposition) covers `--t_comm_size` and how determinants are
placed across ranks (`--determinant_distribution`); [On GPUs](#on-gpus) covers the
Thrust constraints.

The determinant bit order is the one the upstream app documents: reading from the
right, alpha orbital 0, beta orbital 0, alpha orbital 1, and so on — i.e. bit
`2*i` is alpha orbital `i` and bit `2*i + 1` is beta orbital `i`.

## run_gdb_heatbath.py — grow the subspace, round after round

Selected CI (HCI) driven from Python: diagonalize, let SBD expand the subspace from
the resulting wavefunction, diagonalize the larger subspace, repeat. The loop needs
no extra machinery — `carryover_type` 2 and 3 return the parents *together with* the
new candidates, so one round's result **is** the next round's subspace.

```bash
# A cutoff ladder from the default data, stopping before the subspace passes 2M
# determinants, with the (dimension, energy) series written to ladder.json
python run_gdb_heatbath.py --cutoffs 1e-3,1e-4,1e-5 --max_dim 2000000 --log ladder.json

# From the Hartree-Fock determinant alone; serial, since one determinant cannot
# be sharded. Give it every core: the 1e-4 rung is the long part
OMP_NUM_THREADS=48 python run_gdb_heatbath.py --subspace-from hf --cutoffs 1e-3,1e-4

# Sharded over 4 ranks, or 4 GPUs with --device gpu
mpirun -np 4 python run_gdb_heatbath.py --b_comm_size 4 --cutoffs 1e-3
```

[Seeds](#seeds) compares the starting subspaces. To drive the rounds yourself,
`run_gdb_diag.py --carryover_type 2 --heatbath_cutoff 1e-4 --carryovername PREFIX` runs
one expansion, writes the expanded list as one file per rank and prints the
`--detfiles` list for the next run.

### Parameters

The input flags (`--fcidump`, `--detfiles`, `--alpha-file`/`--from-alpha`,
`--alpha-limit`) are the same as `run_gdb_diag.py`'s, so a command that feeds one its
data feeds the other.

Seed — where the starting subspace comes from:

| Parameter | What it controls | Default |
|---|---|---|
| `--subspace-from` | `files` reads `--detfiles`; `hf` starts from the single Hartree-Fock determinant; `from-alpha` builds the `\|A\|^2` product of an alpha list; `strings` reads full determinants, e.g. sampled configurations | `files` |
| `--fcidump` | FCIDUMP defining the Hamiltonian | Fe4S4, see [The default data](#the-default-data) |
| `--detfiles` | `--subspace-from files`: comma-separated determinant files, concatenated in Python. Their combined order must be sorted and disjoint | upstream's four Fe4S4 files |
| `--alpha-file` (or `--from-alpha`) / `--alpha-limit` | `--subspace-from from-alpha`: the alpha list, and a cap on how many of its strings to keep. The product costs `N^2` determinants, so this is the size dial | none / `0` (all) |
| `--strings-file` | `--subspace-from strings`: a file of `2*norb`-bit determinants | none |

Ladder — how far the expansion is pushed. A rung runs rounds at one cutoff until the
energy stops moving, then the next cutoff begins:

| Parameter | What it controls | Default |
|---|---|---|
| `--cutoffs` | The ladder itself: comma-separated `heatbath_cutoff` values, smallest step last. Each rung admits candidates whose estimated contribution exceeds it | `1e-3` |
| `--max_rounds` | Cap on rounds **per rung**, so a rung that never converges cannot run forever | `8` |
| `--energy_tol` | Advance to the next rung once `\|dE\|` between rounds falls below this (Hartree) | `1e-5` |
| `--max_dim` | Stop before diagonalizing a subspace larger than this. The brake that keeps rows of a comparison cost-matched | `0` (no cap) |
| `--carryover_type` | Heatbath variant, 2 or 3. Types 0 and 1 do not expand, so they cannot drive the loop | `2` |
| `--heatbath_truncation` | Weight threshold applied to **parents**, before expanding — not the cutoff. See the warning below; leave it at 0 | `0.0` |
| `--heatbath_batch_size` | Expansion batch size per rank | `1000000` |

Solver, MPI and output — the same meanings as in `run_gdb_diag.py`:

| Parameter | What it controls | Default |
|---|---|---|
| `--device` | `cpu`, or `gpu` (Thrust, NVIDIA only). `gpu-omp` is not supported and raises an error, as does `auto` when it resolves to `gpu-omp`; see [Choosing a backend](#choosing-a-backend) | `cpu` |
| `--method` | 0=Davidson, 1=Davidson storing the Hamiltonian. GDB has no Lanczos | `0` |
| `--tolerance` / `--iteration` / `--block` | Davidson residual tolerance, iteration cap, and basis-vector count. Also accepted as `--eps` / `--max_it` / `--max_nb`, matching SBD's own names | `1e-6` / `30` / `10` |
| `--bit_length` | Bits per packed word: even, at most 62. 64 is undefined behavior in SBD's redistribution. An odd value breaks SBD's alpha/beta conversion once a determinant spans two words, so the heatbath expansion crashes or returns a wrong energy | `62` |
| `--b_comm_size` | Basis shards — the only dimension that divides memory | `1` |
| `--t_comm_size` | Tasks per ring station; must not exceed `--b_comm_size` | `1` |
| `--determinant_distribution` | Placement across `b_comm`; see [Placement across `b_comm`](#placement-across-b_comm) | `equal-bra-a` |
| `--log FILE` | Write the per-round `(rung, cutoff, round, dimension, energy, delta_energy, seconds)` series as JSON — the machine-readable form of the table the driver prints | none |

### Why a cutoff *ladder*

`heatbath_cutoff` admits a candidate when its estimated contribution exceeds the
threshold, so **for a fixed cutoff the subspace reaches a self-consistent size and
stops growing** — more rounds then buy nothing. Going deeper means lowering the
cutoff. So `--cutoffs` takes a ladder: rounds run at one cutoff until the energy
stops moving (`--energy_tol`, the standard HCI criterion) or the subspace stops
growing, then the next rung begins. `--max_dim` caps the whole run, which is what
keeps rows of a comparison cost-matched rather than cutoff-matched.

On Fe4S4 (36 orbitals, 54 electrons) starting from upstream's four shipped files,
a `1e-3,1e-4` ladder walks the subspace out like this:

| rung | dimension | energy |
|---|---|---|
| seed (upstream's four files) | 59,536 | −326.6982518821 |
| `1e-3` round 1 | 63,569 | −326.7298282871 |
| `1e-3` settled | 63,887 | −326.7302438694 |
| `1e-4` round 1 | 568,538 | −326.7773323983 |
| `1e-4` round 3 | 755,107 | −326.7839718082 |

`1e-3` gains ~32 mHa for 7% more determinants and then saturates — that is the fixed
point for that cutoff, not convergence of the method. One step to `1e-4` multiplies the
subspace by roughly nine, so the last cutoff you name dominates the run. `--max_dim`
stops it cleanly, with a `stop_reason`, before a rung outgrows your memory.

The energy is variational, so it must fall monotonically; the driver flags a rise,
which would mean the subspace shrank or a round failed to converge. For h2o and n2 the
FCI limits in that basis (−76.24377680 and −109.04874199) are hard ceilings a correct
run can never cross, which makes a sparse run self-checking even without a reference.

### Seeds

`--subspace-from` selects where the starting subspace comes from, so one driver produces
every row of a seed comparison:

| `--subspace-from` | starting subspace |
|---|---|
| `files` *(default)* | determinant files, defaulting to upstream's four Fe4S4 files |
| `hf` | the Hartree-Fock determinant alone, built from the FCIDUMP header — the no-input null |
| `from-alpha` | an alpha list interleaved with itself, i.e. a TPB-shaped product space |
| `strings` | an arbitrary bitstring file, e.g. sampled configurations |

Which seed is best depends on the system. `hf` is the baseline: it needs no input
subspace, so it shows what the expansion achieves on its own. Measure the others
against it with `--log`.

Compare seeds **at matched dimension**, not at matched cutoff. Different seeds reach
different sizes from the same cutoff, so a cutoff-matched comparison mostly reports
subspace size rather than seed quality. `--max_dim` and the `--log` series are there
for that.

### Feeding QPU samples in: `--subspace-from strings`

`--strings-file` takes a plain text file, one `2*norb`-bit determinant per line — not a
counts JSON. Converting a qiskit-addon-sqd counts file means two steps, and the second
one is not optional:

1. **Postselect** on the target Hamming weight per spin sector. `recover_configurations`
   and friends do this in the SQD loop; doing it here keeps determinants with the wrong
   electron count out of the subspace.
2. **Interleave the bits.** qiskit-addon-sqd emits `[beta | alpha]` — the two halves
   concatenated — while GDB wants them interleaved, bit `2*i` alpha orbital `i` and bit
   `2*i + 1` beta orbital `i`. The drivers expose the conversion as `interleave()`.

```python
import json, pathlib, sys
sys.path.insert(0, ".")                    # run from examples/gdb/
from run_gdb_heatbath import interleave

counts = json.loads(pathlib.Path("../tpb/count_dict_h2o.json").read_text())
norb = len(next(iter(counts))) // 2
n_alpha = n_beta = 5                       # the sector you are solving

dets = {
    interleave(key[norb:], key[:norb])     # key is [beta | alpha]
    for key in counts
    if key[norb:].count("1") == n_alpha and key[:norb].count("1") == n_beta
}
pathlib.Path("dets.txt").write_text("\n".join(sorted(dets)) + "\n")
```

A `set` drops duplicate samples, which GDB rejects rather than silently dedupes, and
`sorted()` gives the ordering the shard contract wants. Then:

```bash
python run_gdb_heatbath.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --subspace-from strings --strings-file dets.txt \
    --cutoffs 1e-3,1e-4
```

On the bundled 275-bitstring h2o file that is a 275-determinant sparse subspace at
−76.0723973374, which the ladder then grows — as opposed to the 75,625-determinant
product space TPB would build from the same samples.

**Both drivers check this for you.** Before solving, they compare every
determinant's alpha and beta electron counts with the FCIDUMP's `NELEC`/`MS2`, and stop
with the number of determinants that do not match and the first one's index. Upstream
would not: strings left in `[beta | alpha]` order diagonalize to a plausible-looking
wrong energy and fail only later, inside the heat-bath expansion.
`--skip-weight-check` turns the check off, for mixing spin sectors on purpose.

### The loop needs no amplitudes

`gdb_diag` does not return the wavefunction, and the ladder does not need it: SBD uses
the amplitudes internally to select candidates and returns the expanded determinant
list, which is the next subspace. To select determinants yourself, read the amplitudes
from the `savename` files.

### `--heatbath_truncation` is not the cutoff

It discards **parents** by weight *before* expansion starts, and its default of 0
(keep every parent) is almost always what you want. Setting it to `1e-4` on the
59,536-determinant Fe4S4 wavefunction cut the subspace to 506 rather than growing it.

## Expected results

| case | energy | note |
|---|---|---|
| h2o, first 24 alpha interleaved into a 576-determinant product basis | **-76.0588897208** | matches `tpb_diag` on the same subspace to 1.4e-14 |
| h2o, full 275² = 75,625-determinant interleave | **-76.2359466308** | against the **-76.23594663** published for that alpha list |
| Fe4S4, 59,536 determinants, 36 orbitals | **-326.6982518821** | measured here; upstream publishes no reference for this case |

The h2o rows interleave an alpha list with itself into its product basis, which TPB
can express too, so GDB and `tpb_diag` must agree on it:

```bash
python run_gdb_diag.py --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --from-alpha ../../vendor/sbd-upstream/data/h2o/h2o-1em3-alpha.txt --alpha-limit 24
```

The h2o product-basis energy is unchanged across `b_comm_size` 1, 2 and 4, across
`t_comm_size` 1 and 2, and across all six placement schemes. Fe4S4 is likewise
identical at `b=1`, `b=4` (one file per rank) and `b=2, t=2`.

A note on what sharding buys: `b_comm_size` divides **memory**, not necessarily time.
At modest sizes the ring communication can offset the extra parallelism, so the reason
to shard is to hold a subspace that would not fit on one rank — and, on GPUs, because
it is the only way to use more than one card at all (see [On GPUs](#on-gpus)). Measure
on your own hardware rather than extrapolating from anyone else's.

## Preparing pre-split determinant files (optional)

Neither driver needs pre-split files: both read a list and slice it deterministically,
so one big file is always correct. Splitting is an **I/O and memory optimization** —
with one file every rank reads the whole thing and keeps only its slice, whereas with
one file per rank the read is parallel and peak memory is a single shard. Both drivers
take that path automatically when the number of `--detfiles` equals `--b_comm_size`.

Upstream already ships the tool for producing them: **`apps/gen_dets`** (built as
`gdet`). It takes an alpha determinant list, forms the full alpha × beta product,
sorts it, and writes it as N shards:

```bash
cd vendor/sbd-upstream/apps/gen_dets
# edit Configuration for your compiler, then:
make
mpirun -np 1 ./gdet --adetfile AlphaDets.txt \
    --detfiles det0.txt,det1.txt,det2.txt,det3.txt \
    --bit_length 20 --norb 36
```

The number of files comes from `--detfiles`, not the rank count, so `-np 1` is enough;
give one file per rank you will run on. The files meet the shard contract below.
`gdet` only builds the product of an alpha list, so a sparse or heat-bath-expanded list
goes to the drivers as one file, or in memory as `run_gdb_heatbath.py` passes it.

Upstream's Fe4S4 GDB subspace is itself such a product: the four files are the
244 × 244 = 59,536 product of `AlphaDets.txt`, and TPB on that file returns the same
energy.

## MPI decomposition

GDB decomposes as `t_comm_size × b_comm_size × helper`, with its own field names
rather than TPB's `adet`/`bdet`/`task`. Only two are settable; `helper` is derived:

```
helper = ranks / (t_comm_size × b_comm_size)
```

**`t_comm_size × b_comm_size` must divide the rank count exactly.** Upstream does not
check the remainder and builds unequal communicators, so `gdb_diag` raises an error
first.

### What each dimension buys

| | parallelizes | shards memory? | constraint |
|---|---|---|---|
| **`b_comm_size`** | the basis itself — each rank owns a block, and the blocks form the ring the ket rotates around | **yes — the only one that does** | ≥ 1 |
| **`t_comm_size`** | the ring's stations, each task rank starting at its own offset | no — costs `t` replicas of the ket | **`t ≤ b`** |
| **`helper`** | rows within a block (`idet % helper`), reduced by an allreduce | no — every helper rank builds the full excitation lookup | none on CPU; **must be 1 on Thrust** |

`t ≤ b` is structural: the matvec rotates the ket around `b_comm` as a ring, with one
task per station. Upstream does not check it and segfaults, so `gdb_diag` raises an
error first. Because only `b` shards memory, `--b_comm_size 1` on many ranks keeps the
whole basis and excitation lookup on every rank.

### The shard contract

With `--b_comm_size R > 1` each rank passes **its own shard**, not the whole list:

- the shard index is `rank % R`, and ranks sharing one must pass **identical**
  determinants (the in-memory path does not broadcast the list);
- shards must be **globally sorted and disjoint** — shard `i` strictly below shard
  `i + 1`, which slicing a globally sorted list gives you for free;
- the union must be the basis you meant. That part cannot be checked here, so
  compare the returned `global_dim` against what you expect.

Everything else is checked and raises rather than quietly diagonalizing the wrong
subspace.

### Placement across `b_comm`

`--determinant_distribution` chooses who owns what. All six agree on the energy —
placement is a load-balancing decision — so pick by balance, not by result.

| scheme | equalizes | leaves uneven |
|---|---|---|
| `input` | nothing; your slices are kept | whatever imbalance you passed |
| `equal-bra-a` *(default)* | distinct **alpha** strings per rank, which is what the matvec's outer loop runs over | determinant counts, when one alpha has many betas |
| `count` | determinant counts | one alpha's determinants can straddle ranks |
| `count-sorted` | as `count`, plus a reorder within each rank | same |
| `grid-cyclic` | spreads alpha and beta keys cyclically over a `grid_a × grid_b` grid | no count rebalance after |
| `grid-cyclic-balanced` | `grid-cyclic`, then a count rebalance | — |

The grid schemes exist because a one-dimensional alpha split can be badly skewed for
an irregular subspace — exactly the sampled-bitstring case. `--determinant_grid_a`
and `--determinant_grid_b` must multiply to `b_comm_size`; give both or neither, and
the default is the factor pair nearest square.

For Fe4S4 the four shipped files are already the balanced globally-sorted split
(14,884 each, disjoint, individually sorted, concatenation in order), so at
`--b_comm_size 4` the driver hands file *i* to shard *i* and `input` placement is
already count-balanced.

### Choosing a backend

`--device cpu` (the default) or `--device gpu` (Thrust, NVIDIA). Upstream has no
OpenMP-offload GDB code, so `gpu-omp`, and `auto` when it would pick `gpu-omp`, raise
an error naming these two before any input is read. On AMD GPUs, GDB runs on `cpu`.

### On GPUs

Thrust needs every rank in `t_comm_size × b_comm_size` (no helper ranks), so **multi-GPU
GDB needs `--b_comm_size`**, e.g. `-np 4 --b_comm_size 4`; one rank on one GPU needs
none. A rank count that leaves helper ranks raises an error up front.

Only the Davidson runs on the GPU: the heat-bath expansion and carryover selection run
on the host with OpenMP. Keep `OMP_NUM_THREADS` generous on GPU runs, and expect the GPU
to idle during those steps.

### `method`

Only 0 and 1. GDB has no Lanczos (TPB's 2 and 3); `gdb_diag` raises an error for them,
where upstream would return an uninitialized energy.

### Outputs, and what is replicated

| key | distribution |
|---|---|
| `energy` | replicated, bit-identical on every rank |
| `density`, `one_p_rdm`, `two_p_rdm` | replicated on every rank |
| `carryover_det` | **this rank's shard**, as an `(n, words)` array. For `carryover_type` 1 it is split over `b_comm` *and duplicated* across the helper dimension, so gathering means one representative per `rank % b_comm_size`; for types 2 and 3 it is split over the world communicator with no duplication, so a plain allgather is correct |
| `local_dim` / `global_dim` | determinants on this rank, and summed over `b_comm` |
| `savename` | optional, and off by default: writes the amplitudes to one file per shard rather than returning them — see below |

### Getting the amplitudes, if you want them

Usually you do not: the energy, density and RDMs are returned, and a heat-bath ladder
takes its next subspace from `carryover_det`. For your own analysis, pass `savename`
and read the files back, or use `sbd.analysis.load_state`. SBD writes one per b_comm position,
`f"{savename}{rank_b:06d}.bin"`: a single `…000000.bin` at `b_comm_size 1`, otherwise
`b_comm_size` files each holding only that shard, so no single file is the whole
wavefunction. Each is two `size_t` headers (`n_dets`, `words_per_det`), then
`n_dets × words_per_det` `size_t` determinant words in canonical order, then `n_dets`
`float64` amplitudes.

## See Also

- [`../tpb/README.md`](../tpb/README.md) — TPB, the SQD loops, and the bundled test data
- [Repository README](../../README.md) — installation, API reference
