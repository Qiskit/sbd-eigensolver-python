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

"""SQD loop that grows its own subspace via single excitations, using SBD as solver.

Same self-consistent SQD machinery as run_sqd_sbd.py (sampling, configuration
recovery, SBD as sci_solver) but with one structural difference: this driver
calls diagonalize_fermionic_hamiltonian with max_iterations=1 itself, in an
outer Python loop, and between calls takes the solved wavefunction's dominant
determinant pairs, expands them via qiskit_addon_sqd.fermion's own
enlarge_batch_from_transitions (single excitations, both spin channels), and
feeds the result forward as next round's include_configurations. That is the
same general idea as SBD's own --sbd_carryover_type 2/3 (see
run_sbd_selected_ci.py in python/experimental/), implemented instead with
qiskit-addon-sqd's own excitation-generation utility, so it works with any
sci_solver, not just SBD, and needs only plain upstream SBD when SBD is used
as the solver here.

Two independent stopping conditions, either one is enough (matching
run_sbd_selected_ci.py's own two): the enlarged set adds nothing new beyond
what's already included (closed under single-excitation connectivity), or
--energy_tol and --occupancies_tol both hold between outer rounds.
--max_iterations is a safety cap, not the primary stopping mechanism -- a
run reaching it before either real criterion is a sign something needs
tuning, not the expected happy path.

enlarge_batch_from_transitions is JAX-based. By default, JAX may select the first GPU 
visible to each process, which can result in multiple ranks sharing the same GPU. 
Proper rank-to-GPU assignment helps avoid GPU memory contention and out-of-memory errors. 

Usage (MPI required):
    mpirun -np 8 python run_sqd_enlarge_subspace_sbd.py \
        --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
        --counts count_dict_h2o.json \
        --device gpu \
        --adet_comm_size 4 --bdet_comm_size 2 \
        --enlarge_threshold 1e-4 --max_iterations 10
"""

import argparse
import json
import re
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np
from mpi4py import MPI
from pyscf import ao2mo, tools
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, enlarge_batch_from_transitions


def parse_args():
    p = argparse.ArgumentParser(
        description="SQD with SBD solver, growing its own subspace via single "
                     "excitations (qiskit-addon-sqd's enlarge_batch_from_transitions) "
                     "between outer iterations, instead of relying only on sampling.",
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
    loop.add_argument("--enlarge_threshold", type=float, default=1e-4,
                       help="Keep determinant PAIRS (ci_strs_a[i], ci_strs_b[j]) "
                            "with |amplitude|^2 above this before expanding them "
                            "via single excitations. Lower it to expand from more "
                            "pairs each round -- the same role as SBD's own "
                            "--sbd_carryover_threshold, just applied to full-"
                            "determinant amplitude here (matching SBD carryover "
                            "type 3), not marginal half-determinant probability.")
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


def build_single_excitation_transitions(norb):
    """All same-spin single-excitation transition operators, [beta|alpha] layout.

    Column layout matches diagonalize_fermionic_hamiltonian's own convention
    (beta in columns [0, norb), alpha in columns [norb, 2*norb) -- confirmed
    against bitstring_matrix_to_ci_strs, which slices the same way). Row 0 is
    the identity (keeps every input row unchanged, so nothing already present
    is lost); every other row creates on one orbital and annihilates on
    another, within one spin half, in both directions.

    Returns:
        (1 + 4 * C(norb, 2), 2*norb) array of "I"/"+"/"-" strings.
    """
    n_pairs = norb * (norb - 1) // 2
    transitions = np.full((1 + 4 * n_pairs, 2 * norb), "I", dtype="<U1")
    row = 1
    for spin_offset in (0, norb):  # beta half, then alpha half
        for i in range(norb):
            for j in range(i + 1, norb):
                transitions[row, spin_offset + i] = "+"
                transitions[row, spin_offset + j] = "-"
                row += 1
                transitions[row, spin_offset + i] = "-"
                transitions[row, spin_offset + j] = "+"
                row += 1
    return transitions


def ints_to_bool_columns(ci_ints, norb):
    """Inverse of bitstring_matrix_to_integers: ci_str ints -> (len(ci_ints), norb) bool array.

    Matches bitstring_matrix_to_integers's own convention (leftmost column is
    the MSB: result = sum_i matrix[:,i] * 2^(norb-1-i)), verified against it
    directly rather than assumed.
    """
    ci_ints = np.asarray(ci_ints, dtype=np.uint64)
    bits = np.zeros((len(ci_ints), norb), dtype=bool)
    for i in range(norb):
        shift = norb - 1 - i
        bits[:, i] = ((ci_ints >> np.uint64(shift)) & np.uint64(1)).astype(bool)
    return bits


def enlarge_via_singles(ci_strs_a, ci_strs_b, amplitudes, norb, threshold,
                         transitions, jax_device=None):
    """Expand dominant (alpha, beta) pairs via single excitations.

    Selects pairs with |amplitude|^2 > threshold (same convention as SBD's
    own full-determinant carryover_type 3), builds their [beta|alpha]
    bitstring rows, applies `transitions` via enlarge_batch_from_transitions,
    and converts the augmented rows back to unique alpha/beta ci_str ints.

    Args:
        jax_device: Device to run the JAX expansion on, or None to leave
            placement to JAX. Committing the input is what steers the jitted
            computation, so every rank must pass its own device or they all
            land on the same one -- see the module docstring.

    Returns:
        (new_alpha_ints, new_beta_ints): sorted, deduplicated int64 arrays.

        Superset of the *dominant* input strings only. Row 0 of `transitions`
        is the identity, so everything fed in comes back -- but strings whose
        every pair fell below the threshold are never fed in. The expansion
        thus prunes as well as grows and is NOT a superset of
        ci_strs_a/ci_strs_b, so callers must not assume the previous subspace
        survives, and a "added anything new?" test must be a subset test
        rather than equality.
    """
    weights = np.abs(amplitudes) ** 2
    ia_idx, ib_idx = np.nonzero(weights > threshold)
    if len(ia_idx) == 0:
        # Threshold too aggressive for this round -- fall back to the single
        # heaviest pair rather than expanding from nothing.
        ia_idx, ib_idx = np.unravel_index(np.argmax(weights), weights.shape)
        ia_idx, ib_idx = np.array([ia_idx]), np.array([ib_idx])

    alpha_bits = ints_to_bool_columns(np.asarray(ci_strs_a)[ia_idx], norb)
    beta_bits = ints_to_bool_columns(np.asarray(ci_strs_b)[ib_idx], norb)
    bitstring_matrix = np.concatenate([beta_bits, alpha_bits], axis=1)

    # Committing the input pins the jitted expansion to this rank's device.
    # The transition arrays stay uncommitted and follow it. Without this,
    # every rank's JAX places the work on its own first visible device --
    # the same one for all of them.
    if jax_device is not None:
        bitstring_matrix = jax.device_put(bitstring_matrix, jax_device)

    augmented = np.asarray(enlarge_batch_from_transitions(bitstring_matrix, transitions))

    from qiskit_addon_sqd.counts import bitstring_matrix_to_integers
    new_beta = bitstring_matrix_to_integers(augmented[:, :norb])
    new_alpha = bitstring_matrix_to_integers(augmented[:, norb:])
    return np.unique(new_alpha), np.unique(new_beta)


def cap_to_max_dim(new_ints, existing_ints, max_dim, rng):
    """Truncate new_ints to max_dim, always keeping everything in existing_ints first.

    Same seed-priority logic as run_sbd_selected_ci.py's _cap_to_max_dim, ported
    to plain ci_str integer arrays: naive random truncation over the WHOLE
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

    norb, nelec_total, ms2 = parse_fcidump_header(args.fcidump)
    num_elec_a = (nelec_total + ms2) // 2
    num_elec_b = (nelec_total - ms2) // 2

    if rank == 0:
        print("=" * 60)
        print("SQD with SBD solver, native subspace enlargement via excitations")
        print("=" * 60)
        print(f"MPI ranks: {size}")
        print(f"FCIDUMP: {args.fcidump}")
        print(f"  NORB={norb}, NELEC={nelec_total}, MS2={ms2}")
        print(f"  Electrons: ({num_elec_a}, {num_elec_b})")
        print(f"Device: {args.device}")
        print()

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

    jax_device = None
    if device_str in ("gpu", "gpu-omp", "gpu-nvidia-omp"):
        jax_devices = jax.devices()
        # A CPU-only jaxlib, or JAX_PLATFORMS=cpu, reports CpuDevice here.
        # Leave those alone rather than pinning a rank to a CPU "device".
        if jax_devices and jax_devices[0].platform != "cpu":
            jax_device = jax_devices[rank % len(jax_devices)]
    print(f"[rank {rank}] JAX subspace expansion device: "
          f"{jax_device if jax_device is not None else 'unassigned (JAX default)'}",
          flush=True)

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
              f"--enlarge_threshold {args.enlarge_threshold:g} "
              f"--symmetrize_spin {args.symmetrize_spin}")
        print("SBD solver  : "
              f"--sbd_method {args.method} --sbd_eps {args.eps:g} "
              f"--sbd_max_it {args.max_it} --sbd_max_nb {args.max_nb}")
        print("Starting outer loop...")

    sbd_config = {
        "method": args.method, "eps": args.eps, "max_it": args.max_it,
        "max_nb": args.max_nb, "max_time": 3600.0, "bit_length": args.bit_length,
        "use_precalculated_dets": bool(args.sbd_use_precalculated_dets),
        "max_memory_gb_for_determinants": args.sbd_max_memory_gb_for_determinants,
        "adet_comm_size": args.adet_comm_size, "bdet_comm_size": args.bdet_comm_size,
        "task_comm_size": args.task_comm_size,
    }
    sbd_solver = partial(
        solve_sci_batch, sbd_config=sbd_config, device_config=device_config,
        temp_dir=args.temp_dir, clean_temp_dir=not args.keep_temp_dir,
        fcidump_path=args.fcidump,
    )

    transitions = build_single_excitation_transitions(norb)

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

        new_alpha, new_beta = enlarge_via_singles(
            ci_strs_a, ci_strs_b, result.sci_state.amplitudes, norb,
            args.enlarge_threshold, transitions, jax_device=jax_device,
        )
        # Subset, not equality: the expansion omits solved strings whose every
        # pair fell below --enlarge_threshold, so equality never held on a
        # subspace with a low-weight tail and this branch never fired.
        no_growth = (set(new_alpha.tolist()) <= set(int(x) for x in ci_strs_a)
                     and set(new_beta.tolist()) <= set(int(x) for x in ci_strs_b))

        # No universal "safe" default exists for --max_dim (a cap suiting a
        # large system is wildly oversized for H2O/N2), so it stays unset --
        # but leaving it unset on a large system is how we hit a
        # several-hundred-million-pair round before adding this check. Warn
        # before the next diagonalization, not after a GPU OOM traceback.
        expanded_pairs = len(new_alpha) * len(new_beta)
        if rank == 0 and args.max_dim is None and (dim > 50_000_000 or expanded_pairs > 50_000_000):
            print(f"WARNING: subspace is large and growing with --max_dim unset "
                  f"(this round: {dim:_} pairs, next round would be: "
                  f"{expanded_pairs:_} pairs before any cap). Risk of GPU OOM. "
                  f"Consider --max_dim (e.g. 15000 worked well for a 45-orbital "
                  f"system) and/or a tighter --enlarge_threshold to slow growth.")

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
