# CAOP — models written with creation and annihilation operators

> Part of the lattice and spin workflow in [`../README.md`](../README.md#lattice-and-spin-models-caop).

For lattice and spin models (Hubbard, hard-core bosons, spins) written as operator
terms, over a selected basis of occupation bitstrings. For molecules, use TPB or GDB.
Three drivers run upstream SBD's CAOP apps from Python:

| Driver | Computes | Upstream app |
|---|---|---|
| `run_caop_diag.py` | the ground state, saved with `--savename` | [`caop_selected_basis_diagonalization`](https://github.com/r-ccs-cms/sbd/blob/main/apps/caop_selected_basis_diagonalization/README.md) |
| `../analysis/variance.py --hamfile` | how far that state is from exact: its energy variance and PT2 | [`caop-stat-evaluator`](https://github.com/r-ccs-cms/sbd/blob/main/apps/caop-stat-evaluator/README.md) |
| `run_caop_spectrum.py` | that state's Green's function | [`caop-spectrum`](https://github.com/r-ccs-cms/sbd/blob/main/apps/caop-spectrum/README.md) |

**The options are upstream's, with the same names and meanings;** each app's README,
linked above, documents them, and upstream's `gen_ham.py` and `gen_bits.py` (in the
first app's folder) write input files. This page covers what differs in the Python
drivers, and the checks they add.

## How the three fit together

1. **Solve.** `run_caop_diag.py --savename gs-` solves for the ground state and writes
   it as `gs-000000.bin`, ...; note its `Ground state energy:` line.
2. **Check.** `variance.py --hamfile ... --state gs-` reads that state and estimates its
   energy variance and PT2. A large variance means the basis is too small for the state
   to be trusted: grow it (*Tips*: carryover type 2) and solve again.
3. **Spectrum.** `run_caop_spectrum.py --loadname gs-` reads the same state and
   computes its Green's function.

Steps 2 and 3 each read the saved state with the same Hamiltonian file and the energy
step 1 printed, and must be told the solve's settings, which the files do not record:
the number of modes, `--bit_length`, the number of files (the solve's `--b_comm_size`)
and whether it used `--complex`. The
[lattice and spin workflow](../README.md#lattice-and-spin-models-caop) runs all three on
a 2-site Hubbard model.

## Ground state

`run_caop_diag.py`:

```bash
# 4 ranks with OpenMP threads each, the basis split over all of them
OMP_NUM_THREADS=8 mpirun -np 4 python run_caop_diag.py --b_comm_size 4

# On 4 NVIDIA GPUs, one rank each; --method 0 applies H on the fly, on the GPU
OMP_NUM_THREADS=8 mpirun -np 4 python run_caop_diag.py --b_comm_size 4 \
    --device gpu --method 0
```

With only `--b_comm_size` given, the first runs with the defaults:

- **Input:** upstream's example, an 8-site hard-core boson chain: `hamiltonian.txt` and
  `basis.txt` from `vendor/sbd-upstream/apps/caop_selected_basis_diagonalization/`,
  with `--system_size 8`. For your own model give `--hamfile`, `--basisfiles` and
  `--system_size` together.
- **Device:** `--device cpu`. `--device gpu` needs the Thrust build and two or more
  OpenMP threads per rank; upstream's `run.sh` uses one, which the Thrust build refuses.
- **Amplitudes:** real. Add `--complex` for complex coefficients or the `sy` spin
  operator.
- **Solver:** `--method 1` (Davidson, stored Hamiltonian) to `--tolerance 1e-4`, from a
  random start (`--seed 0`). Converge further and fix the seed before comparing
  energies; see *Tips*.
- **MPI:** the basis split over all 4 ranks (`--b_comm_size` equal to the rank count),
  the simplest layout; see *MPI*.

**Differences from upstream's app:**

- `--device cpu` or `gpu` (Thrust, NVIDIA only); see *GPU*.
- `--complex` chooses complex amplitudes per run; upstream's app chooses at compile time
  (`-D_COMPLEX`). Needed for `(re,im)` coefficients, e.g. `(0,1) cdag 2 c 0`, and for
  `sy`.
- Defaults are upstream's, except `--method 1`, `--iteration 10` and
  `--do_redist_basis 1`, taken from upstream's `run.sh`, and `--bit_length 20`.
- Not exposed: `--init` (only `0`, random, exists) and `--h_comm_size` (listed upstream
  but never read; see *MPI*).

### Preparing the input files

**A fermion Hamiltonian by hand.** `gen_ham.py` writes hard-core boson chains only. For
fermions, list every term, each with its Hermitian conjugate. Upstream's Hubbard
convention puts alpha site `p` at mode `p` and beta site `p` at mode `L + p`. A 2-site
Hubbard model, `t = 1`, `U = 4`:

```text
-1                          # fermions
-1.0 cdag 0 c 1             # alpha hopping, -t, both directions
-1.0 cdag 1 c 0
-1.0 cdag 2 c 3             # beta hopping
-1.0 cdag 3 c 2
4.0 cdag 0 cdag 2 c 2 c 0   # U n_{0,up} n_{0,down}
4.0 cdag 1 cdag 3 c 3 c 1
```

**The basis files must be sorted, and in order across files.** Upstream checks this as
it loads them, before any option is read, and aborts the whole MPI job otherwise:

- each file ascending as binary numbers, without duplicates;
- with several files, every state in a file greater than every state in the one before.

`--do_sort_basis` does not repair files that break this. `gen_bits.py` meets it with
several `-o` files by default; with one file, add `--unique`, since its default
single-file output is unsorted. The sorted half-filling basis for the Hubbard file
above is `0101`, `0110`, `1001`, `1010`. For a fixed number of each spin, build the
basis yourself from every alpha and beta occupation: `gen_bits.py` fixes only the total.

### Tips

- **Keep `--do_redist_basis 1`.** Without it the states stay where the files put them:
  one file means one rank holds them all, which unbalances memory.
- **Converge before you compare.** The `1e-4` tolerance stops early; use
  `--tolerance 1e-8` and raise `--iteration` for energies to compare.
- **Fix the start.** The default start is random. Pass `--seed`, or
  `--initial_determinant_bitstring` (e.g. a Néel state).
- **Carryover type 1** needs `--carryover_ratio` (the default `0.0` keeps nothing) and
  `--do_redist_basis 1`: upstream counts the kept states from each rank's share, so an
  uneven basis keeps too many.
- **Carryover type 2** returns a grown basis; the driver prints this rank's share. Feed
  it to the next run's `--basisfiles` to iterate. The expansion runs on the CPU.

## Variance and PT2

The same driver as for TPB and GDB states, [`../analysis/variance.py`](../analysis/README.md),
with `--hamfile` and `--basisfiles` in place of `--fcidump`; it runs on the CPU with MPI
and OpenMP. [CAOP states](../analysis/README.md#caop-states) gives the command.

## Green's function

`run_caop_spectrum.py` continues from a saved ground state, with upstream's option names
and CSV output; upstream's
[`caop-spectrum` README](https://github.com/r-ccs-cms/sbd/blob/main/apps/caop-spectrum/README.md)
gives the options, the conventions for G⁺ and G⁻, and the method's limits.

It runs with MPI + OpenMP on the CPU. The one exception, as upstream's, is applying the
Hamiltonian on the fly, which runs on the GPU on `--device gpu` (Thrust, NVIDIA only):

| `--hamiltonian-mode` | Runs on |
|---|---|
| `stored` (default) | CPU, also on `--device gpu` |
| `on-the-fly` | GPU on `--device gpu`, else CPU |

Everything else stays on the CPU whichever you choose: building the excitation basis,
applying the operators to the state, the block-Lanczos vector work, and evaluating the
Green's function from the coefficients. On `--device gpu` the driver prints which of
the two it ran. There is no `gpu-omp` path.

For the 2-site Hubbard file above:

```bash
# 0. the 2-site Hubbard file above, and its half-filling basis
printf '%s\n' -1 '-1.0 cdag 0 c 1' '-1.0 cdag 1 c 0' '-1.0 cdag 2 c 3' '-1.0 cdag 3 c 2' \
    '4.0 cdag 0 cdag 2 c 2 c 0' '4.0 cdag 1 cdag 3 c 3 c 1' > hub2.txt
printf '%s\n' 0101 0110 1001 1010 > hub2_basis.txt

# 1. ground state over 4 ranks, saved as gs-000000.bin to gs-000003.bin; note the
#    "Ground state energy:" line (10 decimals), not upstream's rounded "Energy =" line
mpirun -np 4 python run_caop_diag.py --hamfile hub2.txt --basisfiles hub2_basis.txt \
    --system_size 4 --b_comm_size 4 --tolerance 1e-10 --savename gs-

# 2. the N+1 sector, written sorted (here 3 electrons: 2 alpha + 1 beta, or 1 + 2)
printf '0111\n1011\n1101\n1110\n' > add_basis.txt

# 3. G+ for alpha site 0 and beta site 0, from the 4 saved files, on 4 ranks
mpirun -np 4 python run_caop_spectrum.py --hamfile hub2.txt --sites 4 --loadname gs- \
    --wavefunction-shards 4 --reference-energy <E0 from step 1> --orbitals 0,2 \
    --channel addition --extra-detfile add_basis.txt --save-coefficients add.coeff \
    --output add.csv --omega-min -6 --omega-max 8 --points 141 --eta 0.1

# 4. a narrower broadening from the saved coefficients, without rerunning: as upstream's
#    app, this reads neither the Hamiltonian nor the state, so it is one process
python run_caop_spectrum.py --read-coefficients add.coeff --output narrow.csv \
    --omega-min -6 --omega-max 8 --points 281 --eta 0.05
```

The rank counts need not match: steps 1 and 3 can each use any number of ranks, as
long as `--wavefunction-shards` is the number of files step 1 wrote. Here that is its
`--b_comm_size`, 4.

- **Give the whole N±1 sector as `--extra-detfile`.** Upstream builds the excitation
  basis only from what the operators reach from the state, which is usually not
  enough. For orbitals of both spins, include both sectors (removing an alpha or a
  beta electron gives different sectors).
- **Match the ground-state run:** `--sites` is its `--system_size`, `--bit-length`
  its `--bit_length`, `--wavefunction-shards` its `--b_comm_size`, and
  `--scalar-type complex` if it used `--complex`.
- **Add both channels** for the full retarded Green's function: run `addition` and
  `removal` on the same orbitals and grid, and sum.

## MPI

The solve, the variance and the Green's function each run with MPI and OpenMP: ranks
split the basis and the work, and OpenMP threads share each rank's part. Give each
rank its threads with `OMP_NUM_THREADS` (and, under a scheduler, the cores to match).
Each step may use its own rank count, as long as the variance and the Green's function
are told how many files the solve wrote (`--shards`, `--wavefunction-shards`). Only
reevaluating saved coefficients (`--read-coefficients`) is a single process, as in
upstream: it evaluates a short list of coefficients and reads no state, so more ranks
would only repeat it.

For the solve and the Green's function, the ranks form a grid `ranks = h × b × t`. **b**
splits the basis and every vector, and is the only one that divides memory. **t** and
**h** split the work of applying the Hamiltonian (**h** also splits its terms), each
holding a full copy of its basis piece. The rank count must be a multiple of `b × t`,
and `t ≤ b` is the useful range.

- **Solve:** set `b` and `t` (`--b_comm_size`, `--t_comm_size`); upstream gives it
  `h = ranks / (b × t)`. The simplest layout is `--b_comm_size` equal to the rank count.
- **Green's function:** set `h` and `t` (`--h-comm-size`, `--t-comm-size`); the rest of
  the ranks go to `b`. The defaults, both 1, put every rank on `b`.
- **Variance:** set `h` (`--h_comm_size`, the Hamiltonian's terms); the state and the
  sampling are spread over the rest. The default, 1, uses every rank for that.

## GPU

There is no OpenMP-offload (`gpu-omp`) CAOP code, so no AMD GPU path; `--device gpu` is
NVIDIA Thrust. Upstream's GPU code only applies the Hamiltonian on the fly:

- **Ground state:** only `--method 0` and `2` use the GPU. The Davidson/Lanczos vector
  work, the diagonal and the heat-bath expansion stay on the CPU, and the vector
  crosses to the GPU at each application. Methods `1` and `3` build and apply a stored
  Hamiltonian on the CPU even on `--device gpu`, and the driver says so.
- **Green's function:** only `--hamiltonian-mode on-the-fly` uses the GPU; see its
  section.
- **Variance:** always on the CPU.

Run one rank per GPU. For a small basis, compare with `--device cpu` and a stored
Hamiltonian, which can come close to the GPU.

## Checks

For each of the inputs below, upstream returns a wrong answer, runs a smaller grid than
asked, or hangs. Upstream is unchanged; this package's bindings refuse these inputs
with a `ValueError` before upstream does any work, so the drivers and the Python
functions they call are both covered. Every rank refuses together, so no rank is left
waiting in a collective.

All three drivers:

- a Hamiltonian file upstream cannot read, including a `(re,im)` coefficient without
  `--complex`; upstream reads it on one rank only, so the others hang;
- a term using a mode at or beyond the number of modes, which upstream does not check.

The ground state:

- a rank count that is not a multiple of `b × t`, or smaller than it;
- Davidson with `--numivec` ≥ `--block`, or `0`;
- `--method` outside `0`–`3`;
- on `gpu`, methods `0`/`2` with `t > b`;
- no basis files.

The Green's function:

- a complex saved state read without `--scalar-type complex`;
- `--h-comm-size` × `--t-comm-size` not dividing the rank count;
- both `--orbitals` and `--operator-file`, or neither;
- an operator file whose statistics differ from the Hamiltonian's, which upstream
  compares on some ranks only;
- on `gpu` with `--hamiltonian-mode on-the-fly`, more task ranks than basis ranks.

The variance: a saved state that does not match the basis files given, by its number of
determinants, which determinants they are, its word size, whether it is real or complex,
and its norm; and a state saved on more files than there are basis ranks.
