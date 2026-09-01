# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""SQD loop using SBD solver with qiskit-addon-sqd.

Requires qiskit-addon-sqd with distributed (SPMD) support. This currently
lives on upstream main (not yet in a PyPI release):
    pip install "git+https://github.com/Qiskit/qiskit-addon-sqd@main"
Once a release includes it, plain `pip install qiskit-addon-sqd` will suffice.

Runs the self-consistent SQD workflow: subsample bitstrings into batches,
diagonalize via SBD, update occupancies, repeat.

Bitstring input (choose one):
    --counts FILE     count_dict.json  {bitstring: count}
    --samples N       generate N uniform random bitstrings (default)

Usage (MPI required):
    # H2O with bundled data, random samples
    mpirun -np 4 python run_sqd_sbd.py \
        --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt

    # Custom FCIDUMP with hardware bitstrings
    mpirun -np 8 python run_sqd_sbd.py \
        --fcidump /path/to/fci_dump.txt \
        --counts /path/to/count_dict.json \
        --samples_per_batch 800 --num_batches 3

Multi-node note:
    This script passes --fcidump directly to the SBD solver, so the
    FCIDUMP path itself must be on a filesystem visible to every rank
    (just like --counts). --temp_dir only needs to hold a per-run
    wavefunction.bin that rank 0 writes and reads, so it can stay
    node-local (default /tmp works).
"""

import argparse
import json
import re
import time
from functools import partial
from pathlib import Path

import numpy as np
from mpi4py import MPI
from pyscf import ao2mo, tools
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian


def parse_args():
    p = argparse.ArgumentParser(
        description="SQD with SBD solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fcidump", required=True, help="Path to FCIDUMP file")
    p.add_argument("--counts", default=None,
                   help="Path to count_dict.json (bitstring counts from hardware)")
    p.add_argument("--samples", type=int, default=3000,
                   help="Number of uniform random samples (used when --counts is not given)")
    p.add_argument("--device",
                   choices=["auto", "cpu", "gpu", "gpu-omp", "gpu-nvidia-omp"],
                   default="cpu",
                   help="cpu | gpu (NVHPC Thrust) | gpu-omp = gpu-nvidia-omp "
                        "(LLVM OpenMP-offload) | auto")

    # SQD outer-loop parameters
    p.add_argument("--samples_per_batch", type=int, default=300)
    p.add_argument("--num_batches", type=int, default=3)
    p.add_argument("--max_iterations", type=int, default=5,
                   help="SQD self-consistent loop iterations (not SBD solver iterations)")

    # SBD solver parameters (names match C++ CLI: sbd/include/sbd/chemistry/tpb/sbdiag.h)
    p.add_argument("--method", type=int, default=0, choices=[0, 1, 2, 3],
                   help="0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham")
    p.add_argument("--tolerance", "--eps", type=float, default=1e-8, dest="eps")
    p.add_argument("--iteration", "--max_it", type=int, default=100, dest="max_it",
                   help="Max SBD Davidson iterations per diagonalization")
    p.add_argument("--block", "--max_nb", type=int, default=10, dest="max_nb")
    p.add_argument("--rdm", "--do_rdm", type=int, default=0, dest="do_rdm",
                   help="0=density only (default, sufficient for SQD), 1=full RDM")
    p.add_argument("--shuffle", "--do_shuffle", type=int, default=0, dest="do_shuffle")
    p.add_argument("--carryover_type", type=int, default=1)
    p.add_argument("--carryover_ratio", "--ratio", type=float, default=0.1, dest="ratio")
    p.add_argument("--carryover_threshold", "--threshold", type=float, default=1e-4, dest="threshold")
    p.add_argument("--bit_length", type=int, default=20,
                   help="Bits packed into each size_t of the bitstring representation "
                        "(RIKEN default 20). Must be <= 63: bitadvance() shifts a "
                        "64-bit size_t by this amount, so 64 is undefined behavior.")

    # MPI sub-communicator sizes
    p.add_argument("--adet_comm_size", type=int, default=1)
    p.add_argument("--bdet_comm_size", type=int, default=1)
    p.add_argument("--task_comm_size", type=int, default=1)

    # tempdir for the wavefunction.bin file that rank 0 writes and reads
    # each iteration. Only rank 0 touches it, so node-local /tmp is fine
    # even for multi-node runs.
    p.add_argument("--temp_dir", default=None,
                   help="Directory for the per-iteration wavefunction.bin that "
                        "rank 0 writes and reads. Defaults to $TMPDIR or /tmp. "
                        "Node-local /tmp is fine even for multi-node runs.")
    p.add_argument("--keep_temp_dir", action="store_true", default=False,
                   help="Keep the per-iteration sbd_files_* subdirectories "
                        "(wavefunction.bin, any regenerated FCIDUMP) under "
                        "--temp_dir after each call. Useful for debugging or "
                        "inspecting intermediate wavefunctions; default is to "
                        "delete them.")

    return p.parse_args()


def parse_fcidump_header(path):
    """Return (norb, nelec_total, ms2) from FCIDUMP header."""
    with open(path) as f:
        header = f.readline()
    norb = int(re.search(r"NORB\s*=\s*(\d+)", header).group(1))
    nelec = int(re.search(r"NELEC\s*=\s*(\d+)", header).group(1))
    ms2 = int(re.search(r"MS2\s*=\s*(\d+)", header).group(1))
    return norb, nelec, ms2


def load_counts_as_bitarray(counts_path, num_bits):
    """Convert count_dict.json {bitstring: count} to qiskit BitArray."""
    with open(counts_path) as f:
        counts = json.load(f)
    bitstrings = list(counts.keys())
    repeats = list(counts.values())
    # Convert strings to 2D bool array via concatenated byte comparison
    joined = "".join(bitstrings)
    bool_flat = np.frombuffer(joined.encode(), dtype=np.uint8) == ord("1")
    bool_matrix = bool_flat.reshape(len(bitstrings), -1)
    # Repeat rows according to counts
    if any(c > 1 for c in repeats):
        bool_matrix = np.repeat(bool_matrix, repeats, axis=0)
    return BitArray.from_bool_array(bool_matrix)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    norb, nelec_total, ms2 = parse_fcidump_header(args.fcidump)
    num_elec_a = (nelec_total + ms2) // 2
    num_elec_b = (nelec_total - ms2) // 2

    if rank == 0:
        print("=" * 60)
        print("SQD with SBD Solver")
        print("=" * 60)
        print(f"MPI ranks: {size}")
        print(f"FCIDUMP: {args.fcidump}")
        print(f"  NORB={norb}, NELEC={nelec_total}, MS2={ms2}")
        print(f"  Electrons: ({num_elec_a}, {num_elec_b})")
        print(f"Device: {args.device}")
        print(f"Samples/batch: {args.samples_per_batch}, "
              f"Batches: {args.num_batches}, "
              f"Max iterations: {args.max_iterations}")
        print()

    # --- Initialize SBD ---
    # solve_sci_batch auto-initializes the SBD backend on first call and
    # falls back to MPI.COMM_WORLD when mpi_comm is not provided, so no
    # explicit sbd.init() / mpi_comm threading is needed in user code.
    from sbd.sbd_solver import solve_sci_batch
    from sbd.device_config import DeviceConfig, print_device_info

    device_str = args.device
    if device_str == "auto":
        device_str = "gpu" if DeviceConfig._check_cuda() else "cpu"

    if rank == 0:
        print_device_info()
        print()

    if device_str == "gpu":
        device_config = DeviceConfig.gpu()
    elif device_str in ("gpu-omp", "gpu-nvidia-omp"):
        device_config = DeviceConfig.gpu_omp()
    else:
        device_config = DeviceConfig.cpu()

    # --- Load molecular integrals ---
    mf_as = tools.fcidump.to_scf(str(args.fcidump))
    hcore = mf_as.get_hcore()
    eri = ao2mo.restore(1, mf_as._eri, norb)
    nuclear_repulsion_energy = mf_as.mol.energy_nuc()

    # --- Load or generate bitstrings ---
    rand_seed = np.random.default_rng(42)

    if args.counts:
        bit_array = load_counts_as_bitarray(args.counts, norb * 2)
        if rank == 0:
            print(f"Loaded {bit_array.num_shots} bitstrings from {args.counts}")
    else:
        from qiskit_addon_sqd.counts import generate_bit_array_uniform
        bit_array = generate_bit_array_uniform(
            args.samples, norb * 2, rand_seed=rand_seed
        )
        if rank == 0:
            print(f"Generated {bit_array.num_shots} uniform random bitstrings")

    if rank == 0:
        print()

    # --- Configure SBD solver ---
    sbd_config = {
        "method": args.method,
        "eps": args.eps,
        "max_it": args.max_it,
        "max_nb": args.max_nb,
        "max_time": 3600.0,
        "do_rdm": args.do_rdm,
        "do_shuffle": args.do_shuffle,
        "carryover_type": args.carryover_type,
        "ratio": args.ratio,
        "threshold": args.threshold,
        "bit_length": args.bit_length,
        "adet_comm_size": args.adet_comm_size,
        "bdet_comm_size": args.bdet_comm_size,
        "task_comm_size": args.task_comm_size,
    }

    sbd_solver = partial(
        solve_sci_batch,
        sbd_config=sbd_config,
        device_config=device_config,
        temp_dir=args.temp_dir,
        clean_temp_dir=not args.keep_temp_dir,
        # Skip the tensor->file round-trip: hand SBD the user's FCIDUMP
        # directly. Assumes --fcidump is on a filesystem visible to every
        # rank, which is already true for any realistic multi-node run.
        fcidump_path=args.fcidump,
    )

    # --- Run SQD loop ---
    result_history = []

    def callback(results: list[SCIResult]):
        result_history.append(results)
        if rank == 0:
            iteration = len(result_history)
            print(f"Iteration {iteration}")
            for i, r in enumerate(results):
                total_e = r.energy + nuclear_repulsion_energy
                dim = np.prod(r.sci_state.amplitudes.shape)
                print(f"  Batch {i}: E={total_e:.10f}, dim={dim:_}")

    if rank == 0:
        print("Starting SQD loop...")
        t0 = time.perf_counter()

    # HF (Hartree-Fock) occupancies as a recovery seed: first num_elec_a
    # alpha orbitals filled, first num_elec_b beta orbitals filled. This
    # is what configuration_recovery falls back to when the bit_array
    # contains no valid Hamming-weight strings — which is always the
    # case for the uniform-random fallback path (probability of a random
    # 2*norb bitstring having exactly (num_elec_a, num_elec_b) Hamming
    # weights is vanishing for any realistic norb). Cheap insurance even
    # when --counts is provided; harmless if the bit_array is already
    # valid.
    hf_alpha = np.zeros(norb); hf_alpha[:num_elec_a] = 1.0
    hf_beta  = np.zeros(norb); hf_beta[:num_elec_b]  = 1.0

    try:
        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=args.samples_per_batch,
            norb=norb,
            nelec=(num_elec_a, num_elec_b),
            num_batches=args.num_batches,
            max_iterations=args.max_iterations,
            sci_solver=sbd_solver,
            symmetrize_spin=True,
            callback=callback,
            seed=rand_seed,
            initial_occupancies=(hf_alpha, hf_beta),
        )
    except RuntimeError as e:
        if "Failed to open FCIDUMP" in str(e):
            if rank == 0:
                print(
                    f"\nERROR: at least one rank could not open the FCIDUMP at "
                    f"{args.fcidump!r}. Make sure --fcidump points at a file on "
                    f"a filesystem visible to every rank (shared home, not "
                    f"node-local /tmp).",
                    flush=True,
                )
        raise

    if rank == 0:
        total_time = time.perf_counter() - t0
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"System: NORB={norb}, NELEC={nelec_total}, MS2={ms2}")
        print(f"Electronic energy: {result.energy:.10f}")
        print(f"Nuclear repulsion: {nuclear_repulsion_energy:.10f}")
        print(f"Total energy:      {result.energy + nuclear_repulsion_energy:.10f}")
        print(f"Total time:        {total_time:.1f}s")
        print()

        if result_history:
            print("Convergence History:")
            for i, results in enumerate(result_history):
                energies = [r.energy + nuclear_repulsion_energy for r in results]
                print(f"  Iter {i+1}: min={min(energies):.10f}, "
                      f"max={max(energies):.10f}, "
                      f"avg={np.mean(energies):.10f}")

    try:
        import sbd
        sbd.finalize()
    except Exception:
        pass


if __name__ == "__main__":
    main()
