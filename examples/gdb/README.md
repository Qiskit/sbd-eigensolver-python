# GDB examples — general determinant basis

GDB spans the subspace with the determinants it is given, rather than with the
Cartesian product of an alpha and a beta list that TPB uses. TPB's dimension is
`|adet| x |bdet|`; GDB's is exactly the number of determinants passed. That is what
makes it the right solver for an arbitrary sparse subspace — a set of sampled
bitstrings used *as sampled*, with no product completion.

For backend selection, `--device` values and bundled test data, see
[`../README.md`](../README.md). For TPB and the SQD loops, see
[`../tpb/README.md`](../tpb/README.md).

## run_gdb_diag.py — standalone GDB diagonalization

Stays on the in-memory entry point (`sbd.gdb_diag`) throughout: determinant text is
read in Python and handed to the binding as a list. SBD's own file-based entry
point is deliberately not used.

```bash
# Fe4S4, upstream's own GDB data: 4 files, 59,536 determinants, 36 orbitals.
# Upstream publishes no reference energy for this case.
python run_gdb_diag.py

# Shard the basis: Fe4S4's four files, one per rank. For this data that is
# already the balanced globally-sorted split, so no redistribution is needed.
mpirun -np 4 -x OMP_NUM_THREADS=8 python run_gdb_diag.py --b_comm_size 4

# Spend ranks on both named dimensions (t <= b, and t*b must divide the ranks)
mpirun -np 8 python run_gdb_diag.py --b_comm_size 4 --t_comm_size 2

# Choose how determinants are placed across b_comm
mpirun -np 4 python run_gdb_diag.py --b_comm_size 4 \
    --determinant_distribution grid-cyclic

# A subspace TPB can also express: interleave an alpha list with itself into the
# full |A|^2 product basis. This is the cross-check against tpb_diag -- same
# subspace, two independent solvers, energies must agree.
python run_gdb_diag.py \
    --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --from-alpha ../../vendor/sbd-upstream/data/h2o/h2o-1em3-alpha.txt \
    --alpha-limit 30

# Grow the subspace with SBD's own heatbath expansion (one round; the expanded
# list comes back as carryover_det)
python run_gdb_diag.py --carryover_type 2 --heatbath_cutoff 1e-4
```

The determinant bit order is the one the upstream app documents: reading from the
right, alpha orbital 1, beta orbital 1, alpha orbital 2, and so on — i.e. bit
`2*i` is alpha orbital `i` and bit `2*i + 1` is beta orbital `i`.

## run_gdb_heatbath.py — grow the subspace, round after round

Selected CI (HCI) driven from Python: diagonalize, let SBD expand the subspace from
the resulting wavefunction, diagonalize the larger subspace, repeat. The loop needs
no extra machinery — `carryover_type` 2 and 3 return the parents *together with* the
new candidates, so one round's result **is** the next round's subspace.

```bash
# Fe4S4 from upstream's shipped subspace, one cutoff
python run_gdb_heatbath.py --cutoffs 1e-3

# A ladder, stopping before the subspace passes 2M determinants
python run_gdb_heatbath.py --cutoffs 1e-3,1e-4,1e-5 --max_dim 2000000

# The no-input null: start from the Hartree-Fock determinant alone
python run_gdb_heatbath.py --seed hf --cutoffs 1e-3,1e-4

# Sharded. At --b_comm_size == ranks with t=1 the expansion comes back already
# sharded for the next round, so nothing has to be gathered.
mpirun -np 4 python run_gdb_heatbath.py --b_comm_size 4 --cutoffs 1e-3
mpirun -np 4 python run_gdb_heatbath.py --b_comm_size 4 --device gpu --cutoffs 1e-4

# Record the (dimension, energy) series for a comparison table
python run_gdb_heatbath.py --cutoffs 1e-3,1e-4 --log ladder.json
```

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
subspace by roughly nine. Use `--max_dim` to stop before a rung outgrows your memory.

The energy is variational, so it must fall monotonically; the driver flags a rise,
which would mean the subspace shrank or a round failed to converge. For h2o and n2 the
FCI limits in that basis (−76.24377680 and −109.04874199) are hard ceilings a correct
run can never cross, which makes a sparse run self-checking even without a reference.

### Seeds

`--seed` selects where the starting subspace comes from, so one driver produces
every row of a seed comparison:

| `--seed` | starting subspace |
|---|---|
| `files` *(default)* | determinant files, defaulting to upstream's four Fe4S4 files |
| `hf` | the Hartree-Fock determinant alone, built from the FCIDUMP header — the no-input null |
| `from-alpha` | an alpha list interleaved with itself, i.e. a TPB-shaped product space |
| `strings` | an arbitrary bitstring file, e.g. sampled configurations |

Which seed is best is system-dependent, and worth measuring rather than assuming.
The `hf` seed is the useful baseline: it starts from a single determinant and needs no
input subspace at all, so it shows what the classical expansion achieves on its own.
On Fe4S4 it climbs from −326.5243550095 (the HF energy) into the same range the
file-seeded run reaches, at a comparable dimension — so on that system the expansion
is doing most of the work and a pre-existing subspace adds little. Whether that holds
for your system is exactly the kind of thing to check with `--log` and a dimension
sweep.

Compare seeds **at matched dimension**, not at matched cutoff. Different seeds reach
different sizes from the same cutoff, so a cutoff-matched comparison mostly reports
subspace size rather than seed quality. `--max_dim` and the `--log` series are there
for that.

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

The number of output files comes from `--detfiles`, **not** from the rank count — rank
0 writes all of them itself, splitting with `get_mpi_range`, so `-np 1` is enough
(upstream's own `run.sh` passes `-np 4`, which works but is not required). Set
`--norb` to the orbital count and give one file per rank you intend to run on, e.g.
eight files for eight GPUs.

The result satisfies what `gdb_diag` requires of input shards — each shard sorted,
shards disjoint, and the concatenation globally sorted so shard *i* sits strictly
below shard *i+1* — because `get_mpi_range` is the same `q = N/p`
remainder-to-the-low-ranks split the drivers use.

Two gaps worth knowing. `gdet` only takes an **alpha list** and can only produce the
alpha × beta *product*, so there is no upstream way to shard an arbitrary sparse or
heatbath-expanded determinant list into files; feed those to the drivers as a single
file (or in memory, which is what `run_gdb_heatbath.py` does between rounds). And
upstream's error messages point at a `scripts/sort-basis-shards.py` that is not
shipped — there is no `scripts/` directory in the tree.

### Note on Fe4S4's shipped basis

Worth knowing when reading benchmark numbers: `AlphaDets.txt` next to `gen_dets`
holds exactly **244** alpha determinants, 244² = **59,536**, and the four det files
total 59,536 and are byte-identical to the GDB app's. So upstream's Fe4S4 "GDB"
benchmark subspace is the **full Cartesian product** of 244 alpha determinants with
itself — a TPB-shaped space written out as an explicit determinant list, not a sparse
one. TPB fed `AlphaDets.txt` returns the same −326.6982518821 that GDB returns on the
four files.

## MPI decomposition

GDB decomposes as `t_comm_size × b_comm_size × helper`, with its own field names
rather than TPB's `adet`/`bdet`/`task`. Only two are settable; `helper` is derived:

```
helper = ranks / (t_comm_size × b_comm_size)
```

**`t_comm_size × b_comm_size` must divide the rank count exactly.** Upstream takes
that quotient by integer division and never checks the remainder, which silently
produces communicators of unequal size and a rank alone in its own basis ring, so
`gdb_diag` refuses it.

### What each dimension buys

| | parallelizes | shards memory? | constraint |
|---|---|---|---|
| **`b_comm_size`** | the basis itself — each rank owns a block, and the blocks form the ring the ket rotates around | **yes — the only one that does** | ≥ 1 |
| **`t_comm_size`** | the ring's stations, each task rank starting at its own offset | no — costs `t` replicas of the ket | **`t ≤ b`** |
| **`helper`** | rows within a block (`idet % helper`), reduced by an allreduce | no — every helper rank builds the full excitation lookup | none on CPU; **must be 1 on Thrust** |

`t ≤ b` is structural, not a limitation of this wrapper: the matvec rotates the ket
around `b_comm` as a ring (`gdb/mult.h:39-41`, `:196-202`), so there are exactly
`b_comm_size` stations and one task is one station. Upstream does not check it and a
starved task rank dereferences an empty lookup (`gdb/helper.h:761`), so `gdb_diag`
rejects it rather than letting it segfault.

Because only `b` shards memory, `--b_comm_size 1` on many ranks means every rank
holds the whole basis *and* the whole excitation lookup no matter how many ranks
there are. That is the wall to watch at scale.

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

### On GPUs

GDB has **Thrust kernels only** (`--device gpu`, NVIDIA). There is no `omp target`
code under `include/sbd/chemistry/gdb/` at all, so under `--device gpu-omp` a GDB run
pins a device and then diagonalizes on the host; the driver warns when it resolves
there. AMD has no Thrust build, so AMD GDB is CPU-only.

The Thrust path additionally requires **`helper == 1`** (`gdb/mult_thrust.h:310-314`,
checked on every kernel launch), so every rank must go to `t_comm_size × b_comm_size`.
Since `helper = ranks / (t × b)`, leaving the basis in one block caps GPU GDB at a
single rank — the helper dimension absorbs every rank you add. So **multi-GPU GDB
requires `--b_comm_size`**, e.g. `-np 4 --b_comm_size 4`. Asking for more ranks than
`t × b` is refused up front, naming the helper dimension, rather than throwing from
inside a kernel launch.

Only the Davidson runs on the device. `gdb/expansion.h` and `gdb/carryover.h` contain
no `thrust::` code and are included outside any `SBD_THRUST` guard (`inc_all.h:23`), so
heatbath expansion and carryover selection run on the host with OpenMP even in a GPU
build. Keep `OMP_NUM_THREADS` generous on GPU runs — one thread per rank throttles that
half of every iteration — and expect less than full GPU utilisation during a ladder for
the same reason.

### `method`

Only 0 and 1. TPB's 2 and 3 select Lanczos, which GDB does not implement; `gdb_diag`
rejects them rather than letting `gdb::diag` fall through both of its branches and
return an uninitialized energy.

### Outputs, and what is replicated

| key | distribution |
|---|---|
| `energy` | replicated, bit-identical on every rank |
| `density`, `one_p_rdm`, `two_p_rdm` | replicated on every rank |
| `carryover_det` | **this rank's shard**, as an `(n, words)` array. For `carryover_type` 1 it is split over `b_comm` *and duplicated* across the helper dimension, so gathering means one representative per `rank % b_comm_size`; for types 2 and 3 it is split over the world communicator with no duplication, so a plain allgather is correct |
| `local_dim` / `global_dim` | determinants on this rank, and summed over `b_comm` |
| `savename` | one file per shard, `f"{savename}{rank_b:06d}.bin"` — rank 0's file is not the whole wavefunction, and GDB has no combined matrix-form dump |

## See Also

- [`../README.md`](../README.md) — backend selection, test data, performance tips
- [`../tpb/README.md`](../tpb/README.md) — TPB and the SQD loops
- [Repository README](../../README.md) — installation, API reference
