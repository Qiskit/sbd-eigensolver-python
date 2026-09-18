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

"""SQD loop that grows its own subspace via SBD's own carryover, in C++.

Where this sits relative to the other SQD drivers:

  run_sqd_sbd.py                    sampling + configuration recovery, fixed
                                    pool -- SBD is only the per-batch solver
  run_sqd_enlarge_subspace_sbd.py   same, plus subspace growth via
                                    qiskit-addon-sqd's JAX single excitations
  this driver                       same, but SBD's own carryover does the
                                    growing, in MPI-distributed C++

Structurally this is run_sqd_enlarge_subspace_sbd.py's template --
diagonalize_fermionic_hamiltonian called with max_iterations=1 in our own
outer loop, feeding the expanded determinants forward as next round's
include_configurations -- with the expansion step swapped out. Instead of
qiskit_addon_sqd.fermion's enlarge_batch_from_transitions, the expanded
determinants come from SBD itself: sbd_config's carryover_type tells the
C++ layer to select them from the wavefunction it just computed, and they
come back on the result as carryover_a/carryover_b (see
sbd_solver.SBDCarryoverResult).

Why bother, when run_sqd_enlarge_subspace_sbd.py already grows its
subspace. The honest answer is operational, not performance:
enlarge_batch_from_transitions is JAX and has no MPI awareness, so every
rank redundantly recomputes the whole expansion, and on a GPU-enabled JAX
install several ranks each try to claim a device and the run dies with
CUDA_ERROR_OUT_OF_MEMORY -- which is why that driver needs
JAX_PLATFORMS=cpu beyond one rank. Nothing here touches JAX, so neither
applies.

It is NOT faster. Measured head to head on a 45-orbital system, 8 ranks,
--max_dim 15000, threshold 1e-4: 2248 s here versus 2250 s for the JAX
path, reaching bit-identical energies and subspace sizes at every one of
the 8 rounds. The expansion is simply not where the time goes -- each
round is dominated by configuration recovery over the sample pool and by
diagonalizing a 225M-determinant subspace, so making the expansion
MPI-distributed buys nothing measurable. (An earlier "2-4x faster" note
was a misattribution: that gap was against a driver with no configuration
recovery at all, so it measured recovery overhead, not the expansion
engine.)

Two independent stopping conditions, either one is enough: the carryover
set adds nothing new beyond what is already included (the subspace is
closed under whatever connectivity the carryover type generates), or
--energy_tol and --occupancies_tol both hold between outer rounds.
--max_iterations is a safety cap, not the primary stopping mechanism.

Usage (MPI required):
    mpirun -np 8 python run_sqd_sbd_carryover.py \
        --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
        --counts count_dict_h2o.json \
        --device gpu \
        --adet_comm_size 4 --bdet_comm_size 2 \
        --sbd_carryover_type 3 --sbd_carryover_threshold 1e-4
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
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian


def parse_args():
    p = argparse.ArgumentParser(
        description="SQD with SBD solver, growing its own subspace between "
                     "outer iterations via SBD's own MPI-distributed carryover "
                     "(C++) rather than qiskit-addon-sqd's JAX excitations.",
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

    # ---- Outer loop: WE own this now, one internal iteration per round ---------
    loop = p.add_argument_group(
        "Outer loop (subspace growth via excitations)",
        "Each round: one internal diagonalize_fermionic_hamiltonian call "
        "(max_iterations=1), then expand the solved determinants via single "
        "excitations and feed the result forward as next round's "
        "include_configurations.")
    loop.add_argument("--samples_per_batch", type=int, default=3000)
    loop.add_argument("--symmetrize_spin", type=int, default=1, choices=[0, 1],
                       help="1 (default): merge the alpha and beta string pools "
                            "every round, forcing ci_strs_a == ci_strs_b -- SBD "
                            "itself supports distinct alpha/beta determinant "
                            "sets, but qiskit-addon-sqd's own loop does not "
                            "when this is on. 0: sample and carry over alpha "
                            "and beta independently, allowing them to differ.")
    loop.add_argument("--num_batches", type=int, default=1,
                       help="Batches per outer round. Unlike run_sqd_sbd.py this "
                            "is not the main lever on subspace size -- excitation "
                            "expansion is -- so 1 is a reasonable default.")
    loop.add_argument("--max_iterations", type=int, default=30,
                       help="Safety cap on outer rounds. NOT the primary stopping "
                            "mechanism -- see --energy_tol/--occupancies_tol and "
                            "the no-growth check. Reaching this is a sign "
                            "something needs tuning, not the expected outcome.")
    loop.add_argument("--energy_tol", type=float, default=1e-8,
                       help="Outer-loop convergence: energy change between rounds.")
    loop.add_argument("--occupancies_tol", type=float, default=1e-5,
                       help="Outer-loop convergence: max orbital-occupancy change "
                            "between rounds. Both this and --energy_tol must hold "
                            "in the same round to stop on tolerance (the no-growth "
                            "check is independent and can stop the loop on its own).")
    loop.add_argument("--sbd_carryover_type", type=int, default=3,
                       help="Which carryover mechanism SBD uses to pick the next "
                            "round's determinants from the wavefunction it just "
                            "computed. 1=selection only (no new determinants), "
                            "2=singles off marginal probability, 3=singles off "
                            "full-determinant amplitude (the default, and the "
                            "closest analogue to run_sqd_enlarge_subspace_sbd.py's "
                            "expansion). Values beyond 3 are accepted and passed "
                            "through as-is; whether they do anything depends on "
                            "what carryover_type values the vendored SBD build "
                            "implements. 0 disables carryover, which would leave "
                            "this driver nothing to expand with -- rejected below.")
    loop.add_argument("--sbd_carryover_threshold", type=float, default=1e-4,
                       help="Amplitude cutoff SBD applies when selecting carryover "
                            "determinants. Lower keeps more, so the subspace grows "
                            "faster per round. Note this is SBD's own threshold, "
                            "passed straight through to the C++ layer -- not "
                            "qiskit-addon-sqd's --sqd_carryover_threshold, which "
                            "this driver does not use.")
    loop.add_argument("--sbd_eri_threshold", type=float, default=None,
                       help=argparse.SUPPRESS)
    loop.add_argument("--max_dim", type=int, default=None,
                       help="Cap on unique alpha/beta strings kept per round, "
                            "applied AFTER expansion. Strings already present "
                            "before expansion are always kept first (never "
                            "randomly dropped); only genuinely new candidates "
                            "from this round's expansion are subject to the cap. "
                            "Unset means no cap.")
    loop.add_argument("--include_hf", action="store_true",
                       help="Force the single Slater determinant with the lowest "
                            "num_elec_a/num_elec_b orbital indices occupied into "
                            "every round's include_configurations.")
    loop.add_argument("--checkpoint_path", type=str, default=None,
                       help="Write ci_strs_a/ci_strs_b/occupancies/energy to this "
                            "path as JSON text (rank 0 only), every round. Same "
                            "format and multi-rank-visibility requirement as "
                            "run_sqd_sbd.py's --checkpoint_path.")
    loop.add_argument("--resume_from", type=str, default=None,
                       help="Seed this run's include_configurations and "
                            "initial_occupancies from a previous --checkpoint_path's "
                            "last recorded round.")

    # ---- SBD: the inner eigensolver (same names/meaning as run_sqd_sbd.py) -----
    sbd = p.add_argument_group("SBD solver (inner diagonalization)")
    sbd.add_argument("--sbd_method", type=int, default=0, choices=[0, 1, 2, 3],
                      dest="method", help="0=Davidson, 1=Davidson+Ham, "
                      "2=Lanczos, 3=Lanczos+Ham")
    sbd.add_argument("--sbd_eps", type=float, default=1e-5, dest="eps",
                      help="SBD Davidson stopping tolerance: residual-vector norm.")
    sbd.add_argument("--sbd_max_it", type=int, default=10, dest="max_it",
                      help="Max SBD Davidson iterations per diagonalization.")
    sbd.add_argument("--sbd_max_nb", type=int, default=10, dest="max_nb")
    sbd.add_argument("--sbd_use_precalculated_dets", type=int, default=1,
                      choices=[0, 1])
    sbd.add_argument("--sbd_max_memory_gb_for_determinants", type=int, default=-1)
    sbd.add_argument("--sbd_bit_length", type=int, default=20, dest="bit_length")

    # ---- MPI decomposition ------------------------------------------------------
    mpi = p.add_argument_group("MPI decomposition")
    mpi.add_argument("--adet_comm_size", type=int, default=1)
    mpi.add_argument("--bdet_comm_size", type=int, default=1)
    mpi.add_argument("--task_comm_size", type=int, default=1)

    p.add_argument("--temp_dir", default=None)
    p.add_argument("--keep_temp_dir", action="store_true", default=False)

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
    joined = "".join(bitstrings)
    bool_flat = np.frombuffer(joined.encode(), dtype=np.uint8) == ord("1")
    bool_matrix = bool_flat.reshape(len(bitstrings), -1)
    if any(c > 1 for c in repeats):
        bool_matrix = np.repeat(bool_matrix, repeats, axis=0)
    return BitArray.from_bool_array(bool_matrix)


def cap_to_max_dim(new_ints, existing_ints, max_dim, rng):
    """Truncate new_ints to max_dim, always keeping everything in existing_ints first.

    Seed-priority truncation over plain ci_str integer arrays: naive random
    truncation over the WHOLE
    candidate set can discard already-proven-important strings just as easily
    as brand-new ones, which is what that driver's own bug fix addressed.
    """
    if max_dim is None or len(new_ints) <= max_dim:
        return new_ints
    existing = np.asarray(existing_ints, dtype=new_ints.dtype)
    existing_in_new = np.intersect1d(new_ints, existing)
    fresh = np.setdiff1d(new_ints, existing)
    if len(existing_in_new) >= max_dim:
        keep_existing = rng.choice(existing_in_new, size=max_dim, replace=False)
        return np.sort(keep_existing)
    budget = max_dim - len(existing_in_new)
    keep_fresh = rng.choice(fresh, size=min(budget, len(fresh)), replace=False)
    return np.sort(np.concatenate([existing_in_new, keep_fresh]))


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.default_rng(42)

    if args.sbd_carryover_type == 0:
        # Without carryover SBD returns no determinants to expand with, so the
        # loop would resample the same pool forever and report convergence --
        # fail up front rather than look like a working run that never grows.
        if rank == 0:
            print("Error: --sbd_carryover_type 0 disables the expansion this "
                  "driver is built around. Use 3 (default) for singles off "
                  "full-determinant amplitude, or see --help for the others. "
                  "For a fixed-pool run, use run_sqd_sbd.py instead.")
        return 1

    norb, nelec_total, ms2 = parse_fcidump_header(args.fcidump)
    num_elec_a = (nelec_total + ms2) // 2
    num_elec_b = (nelec_total - ms2) // 2

    if rank == 0:
        print("=" * 60)
        print("SQD with SBD solver, subspace growth via SBD's own carryover")
        print("=" * 60)
        print(f"MPI ranks: {size}")
        print(f"FCIDUMP: {args.fcidump}")
        print(f"  NORB={norb}, NELEC={nelec_total}, MS2={ms2}")
        print(f"  Electrons: ({num_elec_a}, {num_elec_b})")
        print(f"Device: {args.device}")
        print()

    from sbd.sbd_solver import solve_sci_batch, _assert_carryover_survived
    from sbd.device_config import DeviceConfig, print_device_info
    import sbd as _sbd

    # Probe what carryover fields the built extension actually exposes, so an
    # option the vendored SBD does not implement can say it is being ignored
    # instead of silently doing nothing (_create_sbd_config skips unknown keys
    # via hasattr).
    backend_probe_cfg = _sbd.get_backend(
        None if args.device == "auto" else args.device).TPB_SBD()

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

    mf_as = tools.fcidump.to_scf(str(args.fcidump))
    hcore = mf_as.get_hcore()
    eri = ao2mo.restore(1, mf_as._eri, norb)
    nuclear_repulsion_energy = mf_as.mol.energy_nuc()

    rand_seed = np.random.default_rng(42)

    include_a: list[int] = []
    include_b: list[int] = []
    initial_occupancies = None
    if args.include_hf:
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
            print(f"Resuming from {args.resume_from}: round {last['iteration']}, "
                  f"{len(last['ci_strs_a'])} alpha / {len(last['ci_strs_b'])} beta "
                  "strings carried in as include_configurations")

    if args.counts:
        bit_array = load_counts_as_bitarray(args.counts, norb * 2)
        if rank == 0:
            print(f"Loaded {bit_array.num_shots} bitstrings from {args.counts}")
    else:
        from qiskit_addon_sqd.counts import generate_counts_bipartite_hamming
        counts = generate_counts_bipartite_hamming(
            args.samples, norb * 2,
            hamming_right=num_elec_a, hamming_left=num_elec_b, rand_seed=rand_seed,
        )
        bit_array = BitArray.from_counts(counts, num_bits=norb * 2)
        if rank == 0:
            print(f"Generated {bit_array.num_shots} random bitstrings with "
                  f"({num_elec_a}, {num_elec_b}) alpha/beta Hamming weights")

    if rank == 0:
        print()
        print("Outer loop  : "
              f"--samples_per_batch {args.samples_per_batch} "
              f"--num_batches {args.num_batches} "
              f"--max_iterations {args.max_iterations} (safety cap)")
        print("              "
              f"--energy_tol {args.energy_tol:g} "
              f"--occupancies_tol {args.occupancies_tol:g} "
              f"--symmetrize_spin {args.symmetrize_spin}")
        print("SBD solver  : "
              f"--sbd_method {args.method} --sbd_eps {args.eps:g} "
              f"--sbd_max_it {args.max_it} --sbd_max_nb {args.max_nb}")
        print("SBD carryover: "
              f"--sbd_carryover_type {args.sbd_carryover_type} "
              f"--sbd_carryover_threshold {args.sbd_carryover_threshold:g}"
              + (f" --sbd_eri_threshold {args.sbd_eri_threshold:g}"
                 if args.sbd_eri_threshold is not None else ""))
        print("Starting outer loop...")

    sbd_config = {
        "method": args.method, "eps": args.eps, "max_it": args.max_it,
        "max_nb": args.max_nb, "max_time": 3600.0, "bit_length": args.bit_length,
        "use_precalculated_dets": bool(args.sbd_use_precalculated_dets),
        "max_memory_gb_for_determinants": args.sbd_max_memory_gb_for_determinants,
        "adet_comm_size": args.adet_comm_size, "bdet_comm_size": args.bdet_comm_size,
        "task_comm_size": args.task_comm_size,
        # The whole point of this driver: ask SBD to select the next round's
        # determinants itself. _create_sbd_config defaults carryover_type to 0
        # (because the plain SQD path discards the result), so it has to be set
        # explicitly here -- and "threshold" is SBD's own field name for it.
        "carryover_type": args.sbd_carryover_type,
        "threshold": args.sbd_carryover_threshold,
    }
    if args.sbd_eri_threshold is not None:
        # Not present on every SBD build; _create_sbd_config skips unknown keys
        # via hasattr, so passing it to a build without the field is silently
        # ignored rather than fatal -- warn instead of letting it look applied.
        sbd_config["eri_threshold"] = args.sbd_eri_threshold
        if rank == 0 and not hasattr(backend_probe_cfg, "eri_threshold"):
            print("WARNING: --sbd_eri_threshold was given but the vendored SBD "
                  "build has no eri_threshold field, so it is being ignored.")

    sbd_solver = partial(
        solve_sci_batch, sbd_config=sbd_config, device_config=device_config,
        temp_dir=args.temp_dir, clean_temp_dir=not args.keep_temp_dir,
        fcidump_path=args.fcidump,
    )

    checkpoint_history: list[dict] = []
    result_history: list[list] = []
    current_include = (include_a, include_b) if (include_a or include_b) else None
    current_occ = initial_occupancies
    prev_energy = None
    prev_occ = None
    t0 = time.perf_counter()

    def callback(results):
        result_history.append(results)

    for outer_iter in range(1, args.max_iterations + 1):
        result = diagonalize_fermionic_hamiltonian(
            hcore, eri, bit_array,
            samples_per_batch=args.samples_per_batch,
            norb=norb, nelec=(num_elec_a, num_elec_b),
            num_batches=args.num_batches,
            max_iterations=1,
            include_configurations=current_include,
            initial_occupancies=current_occ,
            sci_solver=sbd_solver,
            symmetrize_spin=bool(args.symmetrize_spin),
            max_dim=args.max_dim,
            callback=callback,
            seed=rand_seed,
        )
        energy = result.energy + nuclear_repulsion_energy
        occ = result.orbital_occupancies
        ci_strs_a, ci_strs_b = result.sci_state.ci_strs_a, result.sci_state.ci_strs_b
        dim = len(ci_strs_a) * len(ci_strs_b)

        # The expansion: SBD already selected the next round's determinants
        # inside the C++ diagonalization we just ran, so there is nothing to
        # compute here -- only to read off. This is the whole difference from
        # run_sqd_enlarge_subspace_sbd.py, which spends a JAX pass (redundantly,
        # on every rank) to derive the same kind of set on the Python side.
        #
        # The guard is not ceremony: these fields ride on a subclass that
        # survives only because qiskit-addon-sqd's loop passes the solver's
        # object through instead of rebuilding it. If that ever changes, the
        # carryover would arrive as None and the loop would silently stop
        # growing while still reporting convergence.
        _assert_carryover_survived(result)
        new_alpha, new_beta = result.carryover_a, result.carryover_b
        if new_alpha is None or new_beta is None:
            raise RuntimeError(
                f"round {outer_iter}: SBD returned no carryover determinants "
                f"even though carryover_type={args.sbd_carryover_type} was "
                "requested. Nothing to expand with."
            )

        new_alpha = np.asarray(new_alpha, dtype=np.int64)
        new_beta = np.asarray(new_beta, dtype=np.int64)

        # Stop when the carryover proposes nothing the solved subspace does not
        # already hold. Deliberately a SUBSET test against the solved strings,
        # not a union folded into what gets forwarded below:
        #
        # unioning the carryover with the solved subspace looks safer -- "a
        # round can only add, never drop a determinant carrying weight" -- but
        # it deadlocks the loop the moment --max_dim binds. cap_to_max_dim
        # keeps everything already in the subspace first, so once the solved
        # subspace is itself max_dim strings the union makes existing == the
        # whole budget, every new candidate is discarded, the subspace freezes,
        # and --energy_tol reads the frozen energy as convergence. Measured: on
        # a 45-orbital system at --max_dim 15000 that bottomed out 3.4 Ha above
        # where the same expansion reaches without the union, "converging" after
        # four rounds. It cannot show up when --max_dim is unset, which is why
        # small-system testing missed it.
        #
        # Not unioning means a low-amplitude determinant can leave the subspace.
        # That is fine and intended -- pruning is what carryover is for, it is
        # exactly what run_sqd_enlarge_subspace_sbd.py's expansion does too
        # (its identity row preserves the thresholded pairs, not the whole
        # subspace), and qiskit-addon-sqd's own carryover plus fresh sampling
        # re-supply anything that still matters.
        solved_a = set(int(x) for x in ci_strs_a)
        solved_b = set(int(x) for x in ci_strs_b)
        no_growth = (set(new_alpha.tolist()) <= solved_a
                     and set(new_beta.tolist()) <= solved_b)

        # No universal "safe" default exists for --max_dim (a cap that suits a
        # large system is wildly oversized for H2O/N2, and vice versa), so it
        # stays unset by default -- but leaving it unset on a large system is
        # exactly how we hit a several-hundred-million-pair round ourselves
        # before adding this check.
        # Warn loudly before the next round's diagonalization, not after a GPU
        # OOM traceback with no clue which flag caused it.
        expanded_pairs = len(new_alpha) * len(new_beta)
        if rank == 0 and args.max_dim is None and (dim > 50_000_000 or expanded_pairs > 50_000_000):
            print(f"WARNING: subspace is large and growing with --max_dim unset "
                  f"(this round: {dim:_} pairs, next round would be: "
                  f"{expanded_pairs:_} pairs before any cap). Risk of GPU OOM. "
                  f"Consider --max_dim (e.g. 15000 worked well for a 45-orbital "
                  f"system) and/or a tighter --sbd_carryover_threshold to slow "
                  f"growth.")

        new_alpha = cap_to_max_dim(new_alpha, ci_strs_a, args.max_dim, rng)
        new_beta = cap_to_max_dim(new_beta, ci_strs_b, args.max_dim, rng)

        if rank == 0:
            print(f"Round {outer_iter}: E={energy:.10f}  dim={len(ci_strs_a)}x{len(ci_strs_b)}={dim:_}"
                  f"  expanded->{len(new_alpha)}x{len(new_beta)}={len(new_alpha) * len(new_beta):_}"
                  f"  ({time.perf_counter() - t0:.2f}s)")
            t0 = time.perf_counter()

            entry = {
                "iteration": outer_iter, "energy": energy,
                "occupancies_a": occ[0].tolist(), "occupancies_b": occ[1].tolist(),
                "ci_strs_a": [int(x) for x in ci_strs_a],
                "ci_strs_b": [int(x) for x in ci_strs_b],
            }
            checkpoint_history.append(entry)
            if args.checkpoint_path:
                tmp = Path(args.checkpoint_path).with_suffix(".tmp")
                tmp.write_text(json.dumps({"iterations": checkpoint_history}))
                tmp.replace(args.checkpoint_path)

        energy_converged = (prev_energy is not None
                             and abs(energy - prev_energy) < args.energy_tol)
        occ_converged = (prev_occ is not None
                          and max(np.max(np.abs(occ[0] - prev_occ[0])),
                                  np.max(np.abs(occ[1] - prev_occ[1]))) < args.occupancies_tol)
        stop = no_growth or (energy_converged and occ_converged)
        stop = comm.bcast(stop if rank == 0 else None, root=0)
        if stop:
            if rank == 0:
                reason = "expansion added nothing new" if no_growth else "energy converged"
                print(f"Stopping: {reason}")
            break

        current_include = ([int(x) for x in new_alpha], [int(x) for x in new_beta])
        current_occ = occ
        prev_energy, prev_occ = energy, occ

    if rank == 0:
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"System: NORB={norb}, NELEC={nelec_total}, MS2={ms2}")
        print(f"Total energy: {energy:.10f}")
        print(f"Final subspace: {len(ci_strs_a)} alpha x {len(ci_strs_b)} beta "
              f"= {len(ci_strs_a) * len(ci_strs_b):_}")

        if result_history:
            print()
            print("Convergence History:")
            for i, results in enumerate(result_history):
                energies = [r.energy + nuclear_repulsion_energy for r in results]
                print(f"  Round {i+1}: min={min(energies):.10f}, "
                      f"max={max(energies):.10f}, "
                      f"avg={np.mean(energies):.10f}")

    try:
        import sbd
        sbd.finalize()
    except Exception:
        pass


if __name__ == "__main__":
    main()
