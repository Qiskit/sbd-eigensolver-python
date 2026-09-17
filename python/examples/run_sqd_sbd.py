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

Runs the self-consistent SQD workflow: subsample bitstrings into batches,
diagonalize via SBD, update occupancies, repeat.

Bitstring input (choose one):
    --counts FILE     count_dict.json  {bitstring: count}
    --samples N       generate N random bitstrings at the target Hamming weights

Usage (MPI required):
    # H2O with the bundled counts file (275 bitstrings -> ~ -76.236 Ha)
    mpirun -np 4 python run_sqd_sbd.py \
        --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
        --counts count_dict_h2o.json

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
                   help="Number of random samples when --counts is absent, drawn at "
                        "the target alpha/beta Hamming weights.")
    p.add_argument("--device",
                   choices=["auto", "cpu", "gpu", "gpu-omp", "gpu-nvidia-omp"],
                   default="cpu",
                   help="cpu | gpu (NVHPC Thrust) | gpu-omp = gpu-nvidia-omp "
                        "(NVHPC OpenMP target offload) | auto")

    # ---- SQD: the self-consistent loop (qiskit-addon-sqd owns these) ----------
    # These are the knobs a user reasons about. Unprefixed names that our tooling
    # already passes are kept as-is.
    sqd = p.add_argument_group(
        "SQD loop (qiskit-addon-sqd)",
        "Controls the outer self-consistent loop: how it batches, when it stops, "
        "and what it carries from one iteration to the next.")
    sqd.add_argument("--samples_per_batch", type=int, default=3000,
                     help="Dominant control on subspace size. With "
                          "symmetrize_spin the alpha and beta string sets "
                          "merge, so the subspace is up to (2N)^2.")
    sqd.add_argument("--symmetrize_spin", type=int, default=1, choices=[0, 1],
                     help="1 (default): merge the alpha and beta string pools "
                          "every iteration, forcing ci_strs_a == ci_strs_b -- "
                          "SBD itself supports distinct alpha/beta determinant "
                          "sets, but qiskit-addon-sqd's own loop does not when "
                          "this is on. 0: sample and carry over alpha and beta "
                          "independently, allowing them to differ.")
    sqd.add_argument("--num_batches", type=int, default=3)
    sqd.add_argument("--max_iterations", type=int, default=5,
                     help="SQD self-consistent loop iterations. NOT the SBD "
                          "Davidson iteration count -- that is --sbd_max_it.")
    sqd.add_argument("--energy_tol", type=float, default=1e-8,
                     help="Outer-loop convergence: stop when the energy changes "
                          "by less than this between iterations. Not directly "
                          "comparable to --sbd_eps, which is a residual norm, not "
                          "an energy.")
    sqd.add_argument("--occupancies_tol", type=float, default=1e-5,
                     help="Outer-loop convergence on the average orbital "
                          "occupancies.")
    sqd.add_argument("--max_dim", type=int, default=None,
                     help="Cap on single-spin strings per sector, so the subspace "
                          "cannot exceed max_dim^2. This is the lever for runaway "
                          "setup cost: SBD's helper construction (MakeHelpers) is "
                          "host-side and superlinear in determinants per spin, and "
                          "grows between iterations as carryover accumulates. "
                          "Unset means no cap.")
    sqd.add_argument("--sqd_carryover_threshold", type=float, default=1e-4,
                     help="SQD keeps every determinant whose |coefficient| is at "
                          "least this, and carries it into the next iteration's "
                          "subspace. Lower it to carry more. Distinct from "
                          "--sbd_carryover_threshold, which SQD does not use.")
    sqd.add_argument("--include_hf", action="store_true",
                     help="Force the single Slater determinant with the lowest "
                          "num_elec_a/num_elec_b orbital indices occupied into "
                          "every iteration's subspace, regardless of whether the "
                          "sampled bit_array contains it. Cheap correctness check: "
                          "if this determinant's own diagonal energy beats the SQD "
                          "result, the sampled pool is missing it (and probably its "
                          "low-excitation neighbors), which forcing it in fixes "
                          "directly rather than by enlarging max_dim/iterations.")
    sqd.add_argument("--checkpoint_path", type=str, default=None,
                     help="Write ci_strs_a/ci_strs_b/occupancies/energy to this "
                          "path as JSON text (rank 0 only), every "
                          "--checkpoint_frequency iterations. MUST be visible "
                          "under the same path from every rank: --resume_from "
                          "has no rank-0-reads-then-broadcasts step, every rank "
                          "opens this path itself, so a per-node-local /tmp on a "
                          "multi-node job would leave the other nodes' ranks "
                          "unable to find it. Safe to read mid-run for progress; "
                          "each write replaces the file (atomic rename), so a "
                          "killed run still leaves its last completed checkpoint "
                          "on disk.")
    sqd.add_argument("--checkpoint_frequency", type=int, default=1,
                     help="Write --checkpoint_path every this many iterations, "
                          "plus always on the last one regardless of alignment. "
                          "1 (default) checkpoints every iteration. Raise this "
                          "to cut JSON-write overhead when ci_strs_a/b are large "
                          "(one int per string, so --max_dim 100000 is up to "
                          "200,000 ints per checkpoint) or iterations are fast "
                          "enough that the write itself is a noticeable fraction "
                          "of the per-iteration cost.")
    sqd.add_argument("--resume_from", type=str, default=None,
                     help="Seed this run's include_configurations and "
                          "initial_occupancies from a previous --checkpoint_path's "
                          "LAST recorded iteration. Not a bit-identical "
                          "continuation (RNG state is fresh, and every string "
                          "from that iteration becomes a permanent include -- not "
                          "subject to carryover_threshold decay the way a true "
                          "single-process continuation's own carryover would be), "
                          "but starts the new run's subspace and configuration "
                          "recovery from where the old one stopped rather than "
                          "from raw samples again.")

    # ---- SBD: the inner eigensolver -------------------------------------------
    # Names match the C++ CLI (vendor/sbd-upstream/include/sbd/chemistry/tpb/
    # sbdiag.h) but are prefixed here, because unprefixed --tolerance and
    # --carryover_threshold read as SQD settings and are not.
    sbd = p.add_argument_group(
        "SBD solver (inner diagonalization)",
        "Per-diagonalization settings. Old unprefixed spellings still work.")
    sbd.add_argument("--sbd_method", "--method", type=int, default=0,
                     choices=[0, 1, 2, 3], dest="method",
                     help="0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham")
    sbd.add_argument("--sbd_eps", "--tolerance", "--eps", type=float,
                     default=1e-5, dest="eps",
                     help="SBD Davidson stopping tolerance for ONE "
                          "diagonalization: the NORM OF THE RESIDUAL VECTOR, not "
                          "an energy. Energy error goes roughly as |R|^2/gap, so "
                          "1e-5 already implies far better energy accuracy than "
                          "--energy_tol asks for. Tighten it for a near-degenerate "
                          "system, where a small gap amplifies the residual.")
    sbd.add_argument("--sbd_max_it", "--iteration", "--max_it", type=int,
                     default=10, dest="max_it",
                     help="Max SBD Davidson iterations per diagonalization. This is "
                          "a CAP, not a criterion: if it is reached before --sbd_eps, "
                          "SBD returns the partially converged vector without "
                          "warning. Watch the per-iteration `tol=` values it prints, "
                          "and cross-batch agreement.")
    sbd.add_argument("--sbd_max_nb", "--block", "--max_nb", type=int, default=10,
                     dest="max_nb")
    sbd.add_argument("--sbd_use_precalculated_dets", type=int, default=1,
                     choices=[0, 1],
                     help="Thrust only. 1 precomputes a determinant index for every "
                          "(alpha,beta) pair -- D_size x adets x bdets words on the "
                          "GPU, i.e. the whole subspace, which is what runs out of "
                          "memory on large runs. 0 uses per-thread storage instead: "
                          "slower per matvec, far less memory, and it is the ONLY "
                          "setting under which --sbd_max_memory_gb_for_determinants "
                          "takes effect (mult_thrust.h:257-273).")
    sbd.add_argument("--sbd_max_memory_gb_for_determinants", "--gpu-memory",
                     type=int, default=-1,
                     help="Thrust only, and only with --sbd_use_precalculated_dets 0: "
                          "cap the per-thread determinant buffer in GB. -1 means "
                          "uncapped.")
    sbd.add_argument("--sbd_bit_length", "--bit_length", type=int, default=20,
                     dest="bit_length",
                     help="Bits packed into each size_t of the bitstring "
                          "representation (RIKEN default 20). Must be <= 63: "
                          "bitadvance() shifts a 64-bit size_t by this amount, "
                          "so 64 is undefined behavior.")

    # ---- MPI decomposition: hardware shape, not physics ----------------------
    mpi = p.add_argument_group(
        "MPI decomposition",
        "Rank grid. task x adet x bdet must DIVIDE the rank count exactly: the "
        "quotient becomes the derived helper dimension, and SBD aborts if the "
        "product does not come back to the rank count.")
    mpi.add_argument("--adet_comm_size", type=int, default=1)
    mpi.add_argument("--bdet_comm_size", type=int, default=1)
    mpi.add_argument("--task_comm_size", type=int, default=1)

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

    # --- include_configurations / initial_occupancies: forced references and resume ---
    include_a: list[int] = []
    include_b: list[int] = []
    initial_occupancies = None
    if args.include_hf:
        # Lowest num_elec_a/num_elec_b orbital INDICES occupied -- not necessarily
        # the true HF determinant if this basis isn't canonically ordered, but any
        # single Slater determinant's own diagonal energy is a cheap, exact lower
        # bound on how well a subspace containing it can do. If forcing it in moves
        # the SQD result, the sampled pool was missing it.
        include_a.append((1 << num_elec_a) - 1)
        include_b.append((1 << num_elec_b) - 1)
    if args.resume_from:
        with open(args.resume_from) as f:
            checkpoint = json.load(f)
        last = checkpoint["iterations"][-1]
        include_a.extend(last["ci_strs_a"])
        include_b.extend(last["ci_strs_b"])
        initial_occupancies = (
            np.array(last["occupancies_a"]), np.array(last["occupancies_b"])
        )
        if rank == 0:
            print(f"Resuming from {args.resume_from}: iteration {last['iteration']}, "
                  f"{len(last['ci_strs_a'])} alpha / {len(last['ci_strs_b'])} beta "
                  "strings carried in as include_configurations")
    include_configurations = (include_a, include_b) if (include_a or include_b) else None

    if args.counts:
        bit_array = load_counts_as_bitarray(args.counts, norb * 2)
        if rank == 0:
            print(f"Loaded {bit_array.num_shots} bitstrings from {args.counts}")
    else:
        # Stand-in for hardware counts, at the target Hamming weights. Uniform
        # strings get postselected away: survival is C(norb,ne)^2 / 4^norb,
        # ~6e-6 for H2O (5 of 24).
        from qiskit_addon_sqd.counts import generate_counts_bipartite_hamming
        counts = generate_counts_bipartite_hamming(
            args.samples,
            norb * 2,
            hamming_right=num_elec_a,
            hamming_left=num_elec_b,
            rand_seed=rand_seed,
        )
        bit_array = BitArray.from_counts(counts, num_bits=norb * 2)
        if rank == 0:
            print(f"Generated {bit_array.num_shots} random bitstrings with "
                  f"({num_elec_a}, {num_elec_b}) alpha/beta Hamming weights")

    if rank == 0:
        print()

    # --- Configure SBD solver ---
    sbd_config = {
        "method": args.method,
        "eps": args.eps,
        "max_it": args.max_it,
        "max_nb": args.max_nb,
        "max_time": 3600.0,
        "bit_length": args.bit_length,
        "use_precalculated_dets": bool(args.sbd_use_precalculated_dets),
        "max_memory_gb_for_determinants": args.sbd_max_memory_gb_for_determinants,
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
    checkpoint_history: list[dict] = []

    def callback(results: list[SCIResult]):
        result_history.append(results)
        if rank == 0:
            iteration = len(result_history)
            print(f"Iteration {iteration}")
            for i, r in enumerate(results):
                total_e = r.energy + nuclear_repulsion_energy
                dim = np.prod(r.sci_state.amplitudes.shape)
                print(f"  Batch {i}: E={total_e:.10f}, dim={dim:_}")
            due = (iteration % args.checkpoint_frequency == 0
                   or iteration == args.max_iterations)
            if args.checkpoint_path and due:
                # Batch 0 only: multi-batch checkpoints would need to pick which
                # batch's subspace to resume from, and the driver only ever uses
                # num_batches=1 in practice. Whole file rewritten each call (not
                # appended), so a killed run's last COMPLETE checkpoint survives.
                # Always written on the last iteration regardless of frequency
                # alignment, so a completed run's checkpoint reflects its true
                # final state rather than whatever iteration happened to land on
                # a multiple of --checkpoint_frequency.
                r = results[0]
                entry = {
                    "iteration": iteration,
                    "energy": r.energy + nuclear_repulsion_energy,
                    "occupancies_a": r.orbital_occupancies[0].tolist(),
                    "occupancies_b": r.orbital_occupancies[1].tolist(),
                    "ci_strs_a": [int(x) for x in r.sci_state.ci_strs_a],
                    "ci_strs_b": [int(x) for x in r.sci_state.ci_strs_b],
                }
                checkpoint_history.append(entry)
                tmp = Path(args.checkpoint_path).with_suffix(".tmp")
                tmp.write_text(json.dumps({"iterations": checkpoint_history}))
                tmp.replace(args.checkpoint_path)

    if rank == 0:
        # Say which layer each setting belongs to. The two layers have knobs with
        # near-identical names and very different meanings, which is how the
        # unreachable ones went unnoticed.
        # Print the FLAG spellings, not the internal dest names: "eps" and
        # "carryover_threshold" are exactly the ambiguous labels this grouping
        # exists to remove, so the banner has to name the layer too.
        print("SQD loop     : "
              f"--samples_per_batch {args.samples_per_batch} "
              f"--num_batches {args.num_batches} "
              f"--max_iterations {args.max_iterations}")
        print("               "
              f"--energy_tol {args.energy_tol:g} "
              f"--occupancies_tol {args.occupancies_tol:g} "
              f"--sqd_carryover_threshold {args.sqd_carryover_threshold:g} "
              f"--symmetrize_spin {args.symmetrize_spin}")
        print("SBD solver   : "
              f"--sbd_method {args.method} --sbd_eps {args.eps:g} "
              f"--sbd_max_it {args.max_it} --sbd_max_nb {args.max_nb} "
              f"--sbd_bit_length {args.bit_length}")
        print("MPI grid     : "
              f"--task_comm_size {args.task_comm_size} "
              f"--adet_comm_size {args.adet_comm_size} "
              f"--bdet_comm_size {args.bdet_comm_size}")
        print("Starting SQD loop...")
        t0 = time.perf_counter()

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
            energy_tol=args.energy_tol,
            occupancies_tol=args.occupancies_tol,
            carryover_threshold=args.sqd_carryover_threshold,
            max_dim=args.max_dim,
            include_configurations=include_configurations,
            initial_occupancies=initial_occupancies,
            sci_solver=sbd_solver,
            symmetrize_spin=bool(args.symmetrize_spin),
            callback=callback,
            seed=rand_seed,
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
