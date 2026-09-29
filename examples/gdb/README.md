# GDB examples — general determinant basis

GDB spans the subspace with the determinants it is given, rather than with the
Cartesian product of an alpha and a beta list that TPB uses. TPB's dimension is
`|adet| x |bdet|`; GDB's is exactly the number of determinants passed. That is what
makes it the right solver for an arbitrary sparse subspace — a set of sampled
bitstrings used *as sampled*, with no product completion.

For TPB and the SQD loops, see [`../tpb/README.md`](../tpb/README.md).

## The default data

Both drivers default `--fcidump` and `--detfiles` to upstream's own GDB app data under
`vendor/sbd-upstream/apps/chemistry_gdb_selected_basis_diagonalization/`: the Fe4S4
FCIDUMP at 36 orbitals, and its four `det0.txt`-`det3.txt` holding 14,884 determinants
each, 59,536 in total. **Any command below that passes neither flag runs exactly that
case** — the determinants come from those four files, not from nowhere. Upstream
publishes no reference energy for it, so treat it as a self-consistency benchmark
rather than a validation target.

## run_gdb_diag.py — standalone GDB diagonalization

Stays on the in-memory entry point (`sbd.gdb_diag`) throughout: determinant text is
read in Python and handed to the binding as a list. SBD's own file-based entry
point is deliberately not used.

```bash
# The default case: Fe4S4, upstream's four det files, 59,536 determinants.
python run_gdb_diag.py

# The same run with the defaults written out -- this is the --detfiles syntax to
# copy for your own data. Files are comma-separated and concatenated in Python
# into one in-memory list, so their combined order must be sorted and disjoint.
GDB=../../vendor/sbd-upstream/apps/chemistry_gdb_selected_basis_diagonalization
python run_gdb_diag.py --fcidump $GDB/fcidump_Fe4S4.txt \
    --detfiles $GDB/det0.txt,$GDB/det1.txt,$GDB/det2.txt,$GDB/det3.txt

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

# On GPUs, use --device gpu (Thrust). It requires helper == 1, so every rank has
# to go to t*b -- which means --b_comm_size is not optional for multi-GPU GDB.
mpirun -np 4 python run_gdb_diag.py --device gpu --b_comm_size 4

# One rank, one GPU is the exception: b=1 already leaves helper == 1.
python run_gdb_diag.py --device gpu
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
# Fe4S4 from upstream's shipped subspace (the default data above), one cutoff
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

### Parameters

`--fcidump` and `--detfiles` are spelled and defaulted exactly as in `run_gdb_diag.py`,
so a command that feeds one driver its data feeds the other. Two input flags do **not**
carry over: the alpha list is `--alpha-file` here but `--from-alpha` there, and `--seed`
means different things in the two drivers — here it selects where the subspace comes
from, while in `run_gdb_diag.py` it is the integer RNG seed for a random initial vector.

Seed — where the starting subspace comes from:

| Parameter | What it controls | Default |
|---|---|---|
| `--seed` | `files` reads `--detfiles`; `hf` starts from the single Hartree-Fock determinant; `from-alpha` builds the `\|A\|^2` product of an alpha list; `strings` reads full determinants, e.g. sampled configurations | `files` |
| `--fcidump` | FCIDUMP defining the Hamiltonian | Fe4S4, see [The default data](#the-default-data) |
| `--detfiles` | `--seed files`: comma-separated determinant files, concatenated in Python. Their combined order must be sorted and disjoint | upstream's four Fe4S4 files |
| `--alpha-file` / `--alpha-limit` | `--seed from-alpha`: the alpha list, and a cap on how many of its strings to keep. The product costs `N^2` determinants, so this is the size dial | none / `0` (all) |
| `--strings-file` | `--seed strings`: a file of `2*norb`-bit determinants | none |

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
| `--device` | `cpu` or `gpu` (Thrust) are the two real choices; `gpu-omp` and `auto` run GDB on the host, see [Choosing a backend](#choosing-a-backend) | `cpu` |
| `--method` | 0=Davidson, 1=Davidson storing the Hamiltonian. GDB has no Lanczos | `0` |
| `--tolerance` / `--iteration` / `--block` | Davidson residual tolerance, iteration cap, and basis-vector count. Also accepted as `--eps` / `--max_it` / `--max_nb`, matching SBD's own names | `1e-6` / `30` / `10` |
| `--bit_length` | Bits per packed word | `64` |
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

### The loop needs no amplitudes

A natural question, since `gdb_diag` does not return the wavefunction: the ladder never
needs it. The amplitudes are what drive the selection — weight truncation keeps
determinants by `|c|`, and heatbath scoring is essentially `|c_i · H_ij|` — but SBD
consumes them internally (`WeightTruncation` then `HeatbathExpansion`, which takes the
coefficients as an input) and hands back only the expanded determinant list. That list
is the next subspace, so the loop closes with nothing but determinants crossing the
Python boundary.

Where amplitudes *would* be needed is Python-side selection — deciding yourself which
determinants to expand, as `../tpb/run_sqd_enlarge_subspace_sbd.py` does for TPB. For
GDB that means reading them back from the per-shard `savename` files, since
`gdb::diag` has no in-memory amplitude output.

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

### Choosing a backend

`--device` selects the compute backend per run:

```bash
--device cpu       # host OpenMP (default)
--device gpu       # NVHPC Thrust, NVIDIA only -- the only GPU backend with GDB kernels
```

`gpu-omp` and `auto` are accepted too, but for GDB neither is a GPU path: `gpu-omp` has
no GDB kernels and silently diagonalizes on the host (the drivers warn), and `auto`
prefers Thrust but falls back to `gpu-omp` where Thrust is not built — landing on the
host as well. **For GDB, pick `cpu` or `gpu` explicitly.**

`sbd.available_backends()` reports what this install actually built — a static scan, so
it is safe to call outside `mpirun` — and `sbd.loaded_backends()` reports what the
current process has imported.

One interaction to know about: **`'cpu'` and `'gpu-omp'` must not be used in the same
process.** Both link the same OpenMP runtime and the CPU module is built without offload
support, so whichever loads first initializes that runtime; if it is the CPU backend, the
offload backend can no longer acquire a device and silently runs on the host — right
answers, exit 0, idle GPU. `'cpu'` and `'gpu'` (Thrust) coexist fine, since Thrust does
not route device work through OpenMP.

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
| `savename` | optional, and off by default: writes the amplitudes to one file per shard rather than returning them — see below |

### Getting the amplitudes, if you want them

Usually you do not. `gdb::diag` has no in-memory output for the wavefunction, and
nothing in the normal flow needs it: the energy, density and RDMs are returned directly,
and a heatbath ladder takes its next subspace from `carryover_det` (see [The loop needs
no amplitudes](#the-loop-needs-no-amplitudes)). GDB is also not wired into
qiskit-addon-sqd, whose `SCIState` would be the usual consumer — that path is TPB-only.

When you do want them — your own analysis, or a selection step written in Python — pass
`savename` and read the files back. SBD writes one per b_comm position,
`f"{savename}{rank_b:06d}.bin"`: a single `…000000.bin` at `b_comm_size 1`, otherwise
`b_comm_size` files each holding only that shard, so no single file is the whole
wavefunction. Each is two `size_t` headers (`n_dets`, `words_per_det`), then
`n_dets × words_per_det` `size_t` determinant words in canonical order, then `n_dets`
`float64` amplitudes.

## See Also

- [`../tpb/README.md`](../tpb/README.md) — TPB, the SQD loops, and the bundled test data
- [Repository README](../../README.md) — installation, API reference
