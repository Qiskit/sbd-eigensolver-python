# CAOP — models written with creation and annihilation operators

> Part of the lattice and spin workflow in [`../README.md`](../README.md#lattice-and-spin-models-caop).

`run_caop_diag.py` runs upstream SBD's
[`caop_selected_basis_diagonalization`](https://github.com/r-ccs-cms/sbd/tree/main/apps/caop_selected_basis_diagonalization)
app from Python: the ground state of a Hamiltonian written as operator terms, over a
selected basis of occupation bitstrings, for lattice and spin models (Hubbard,
hard-core bosons, spins). For molecules, use TPB or GDB.

**The options are upstream's, with the same names and meanings;** see
[upstream's README](https://github.com/r-ccs-cms/sbd/blob/main/apps/caop_selected_basis_diagonalization/README.md)
for each one and for its `gen_ham.py` and `gen_bits.py`. This page covers what is
specific to the Python driver and what we found running it.

```bash
cd examples/caop
mpirun -np 4 python run_caop_diag.py --b_comm_size 4    # upstream's 8-site boson chain
```

## Differences from upstream's app

- `--device cpu` or `gpu` (Thrust, NVIDIA only); see *GPU*.
- `--complex` chooses complex amplitudes per run; upstream's app chooses at compile time
  (`-D_COMPLEX`). Needed for `(re,im)` coefficients, e.g. `(0,1) cdag 2 c 0`, and for `sy`.
- Defaults are upstream's, except `--method 1`, `--iteration 10` and
  `--do_redist_basis 1`, taken from upstream's `run.sh`, and `--bit_length 20`. With no
  `--hamfile`/`--basisfiles`, it runs upstream's example files.
- Not exposed: `--init` (only `0`, random, exists) and `--h_comm_size` (listed upstream
  but never read; see *MPI*).

## Preparing the input files

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

## Tips

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

## Green's function

`run_caop_spectrum.py` continues from a saved ground state: upstream's
[`caop-spectrum`](https://github.com/r-ccs-cms/sbd/tree/main/apps/caop-spectrum) app
in Python, with its option names and CSV output; see
[its README](https://github.com/r-ccs-cms/sbd/blob/main/apps/caop-spectrum/README.md)
for the options, conventions and limits. For the 2-site Hubbard file above:

```bash
# 1. ground state, saved; note the "Ground state energy:" line (10 decimals), not
#    upstream's rounded "Energy =" log line
python run_caop_diag.py --hamfile hub2.txt --basisfiles hub2_basis.txt --system_size 4 \
    --tolerance 1e-10 --savename gs-
# 2. the N+1 sector, written sorted (here 3 electrons: 2 alpha + 1 beta, or 1 + 2)
printf '0111\n1011\n1101\n1110\n' > add_basis.txt
# 3. G+ for alpha site 0 and beta site 0, from the saved state
python run_caop_spectrum.py --hamfile hub2.txt --sites 4 --loadname gs- \
    --reference-energy <E0 from step 1> --orbitals 0,2 --channel addition \
    --extra-detfile add_basis.txt --save-coefficients add.coeff --output add.csv \
    --omega-min -6 --omega-max 8 --points 141 --eta 0.1
# 4. a narrower broadening from the saved coefficients, without rerunning
python run_caop_spectrum.py --read-coefficients add.coeff --output narrow.csv \
    --omega-min -6 --omega-max 8 --points 281 --eta 0.05
```

- **Give the whole N±1 sector as `--extra-detfile`.** Upstream builds the excitation
  basis only from what the operators reach from the state, which is usually not
  enough. For orbitals of both spins, include both sectors (removing an alpha or a
  beta electron gives different sectors).
- **Match the ground-state run:** `--sites` is its `--system_size`, `--bit-length`
  its `--bit_length`, `--wavefunction-shards` its `--b_comm_size`, and
  `--scalar-type complex` if it used `--complex`. The files record none of these.
- **Add both channels** for the full retarded Green's function: run `addition` and
  `removal` on the same orbitals and grid, and sum.
- **Only `--hamiltonian-mode on-the-fly` uses the GPU**, as for the ground state; the
  default `stored` runs on the CPU, and the driver says so.

## MPI

The ranks form a grid `ranks = h × b × t`. **b** (`--b_comm_size`) splits the basis and
every vector, and is the only one that divides memory. **t** (`--t_comm_size`) and
**h** split the work of applying the Hamiltonian (**h** also splits its terms), each
holding a full copy of its basis piece. Upstream sets `h = ranks / (b × t)`. The rank
count must be a multiple of `b × t`, and `t ≤ b` is the useful range. The simplest
layout is `--b_comm_size` equal to the rank count.

## GPU

Upstream's GPU code applies the Hamiltonian on the fly, so only `--method 0` and `2`
use the GPU. The Davidson/Lanczos vector work, the diagonal and the heat-bath expansion
stay on the CPU, and the vector crosses to the GPU at each application. Methods `1` and
`3` build and apply a stored Hamiltonian on the CPU even on `--device gpu`, and the
driver says so. There is no OpenMP-offload (`gpu-omp`) CAOP code, so no AMD GPU path.
Run one rank per GPU. For a small basis, compare with `--device cpu` and a stored
Hamiltonian, which can come close to the GPU.

## Checks

Upstream accepts these and returns a wrong energy, a smaller grid than asked, or hangs.
The bindings refuse them on every rank before any work:

- a rank count that is not a multiple of `b × t`, or smaller than it;
- Davidson with `--numivec` ≥ `--block`, or `0`;
- `--method` outside `0`–`3`;
- on `gpu`, methods `0`/`2` with `t > b`;
- a Hamiltonian file upstream cannot read, including `(re,im)` without `--complex`.
