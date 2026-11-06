# Workflows

The examples are organized by solver; this page shows how they fit together. Each step
links to the folder whose README covers its options and pitfalls. Each block below starts
in this `examples/` directory.

Two checks follow every diagonalization, whatever the solver: its **energy variance**
and **PT2** say how far the saved state is from the exact one. Run them before using the
state for anything else. The same driver, [`analysis/variance.py`](analysis/README.md),
handles every solver.

## Molecules: TPB or GDB

From an FCIDUMP and a determinant subspace.

```bash
# 1. diagonalize and save the state; note "Ground state energy:"
cd tpb       # from examples/
mpirun -np 4 python run_sbd_diag.py --adet_comm_size 2 --bdet_comm_size 2 \
    --bit_length 20 --tolerance 1e-8 --savename /path/to/h2o-

# 2. how far from exact: variance and PT2 (stochastic, on CPU)
cd ../analysis
mpirun -np 4 python variance.py --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
    --state /path/to/h2o- --bit_length 20 --energy <E from step 1>
```

- **Solve:** [`tpb/`](tpb/README.md) for an alpha × beta product subspace, including
  the SQD loops; [`gdb/`](gdb/README.md) for an explicit determinant list, including
  heat-bath rounds that grow the subspace.
- **Check:** [`analysis/`](analysis/README.md#workflow).

## Lattice and spin models: CAOP

From a Hamiltonian written as operator terms. After the ground state is checked, its
Green's function gives the single-particle spectrum.

```bash
cd caop      # from examples/
# 0. a 2-site Hubbard model (t = 1, U = 4) and its half-filling basis; see caop/ for the format
printf '%s\n' -1 '-1.0 cdag 0 c 1' '-1.0 cdag 1 c 0' '-1.0 cdag 2 c 3' '-1.0 cdag 3 c 2' \
    '4.0 cdag 0 cdag 2 c 2 c 0' '4.0 cdag 1 cdag 3 c 3 c 1' > hub2.txt
printf '%s\n' 0101 0110 1001 1010 > hub2_basis.txt

# 1. the ground state, its basis split over 4 ranks and saved as 4 files, gs-000000.bin
#    to gs-000003.bin; note "Ground state energy:"
mpirun -np 4 python run_caop_diag.py --hamfile hub2.txt --basisfiles hub2_basis.txt \
    --system_size 4 --b_comm_size 4 --tolerance 1e-10 --savename gs-

# 2. how far from exact: variance and PT2, from the same files (--shards 4: the solve's
#    --b_comm_size). Here hub2_basis.txt is the whole half-filling sector, so both are
#    zero; over too small a basis, such as the two singly-occupied states 0110 and 1001
#    alone, the variance is large and E + PT2 recovers most of the missing energy
mpirun -np 4 python ../analysis/variance.py --hamfile hub2.txt --basisfiles hub2_basis.txt \
    --sites 4 --state gs- --shards 4 --bit_length 20 --energy <E0 from step 1>

# 3. the Green's function, over the N+1 sector (--wavefunction-shards 4, likewise)
printf '0111\n1011\n1101\n1110\n' > add_basis.txt
mpirun -np 4 python run_caop_spectrum.py --hamfile hub2.txt --sites 4 --loadname gs- \
    --wavefunction-shards 4 --reference-energy <E0 from step 1> --orbitals 0,2 \
    --channel addition --extra-detfile add_basis.txt --output add.csv
```

Each step here uses 4 ranks, one basis state each for this 4-state model; a larger
basis spreads the same way over more ranks.

- **Input files and solve:** [`caop/`](caop/README.md#preparing-the-input-files).
- **Check:** [`analysis/`](analysis/README.md#caop-states).
- **Green's function:** [`caop/`](caop/README.md#greens-function).

If the variance is large, the subspace is too small for the state to be trusted, for
the Green's function as much as for the energy: grow it (more samples, or heat-bath
expansion with `--carryover_type 2`), solve again, and check again.

The ground-state run, the check and the Green's function must agree on the run's
settings, which the saved files do not record: `--bit_length`, the number of modes,
the number of files (`--b_comm_size` of the solve), and `--complex`. A real/complex
mismatch is refused.
