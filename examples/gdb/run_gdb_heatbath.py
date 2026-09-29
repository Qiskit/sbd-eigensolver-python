#!/usr/bin/env python3

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

"""
Grow a GDB subspace with SBD's own heatbath expansion, round after round.

This is selected CI (HCI) driven from Python: diagonalize, let SBD expand the
subspace from the resulting wavefunction, diagonalize the larger subspace, repeat.
No new machinery is needed for the loop itself -- ``carryover_type`` 2 and 3 return
the parents *together with* the new candidates (``local_heatbath_expansion`` opens
with ``edet = det``), so the result of one round IS the next round's subspace.

**The cutoff is the dial, and the loop converges per cutoff.** Expansion admits a
candidate when its estimated contribution exceeds ``heatbath_cutoff``, so for a
fixed cutoff the subspace reaches a self-consistent size and stops growing --
further rounds buy nothing. Going deeper means lowering the cutoff. Hence a
*ladder*: spend rounds at one cutoff until the energy stops moving, then step to
the next. ``--cutoffs`` takes the whole ladder.

Do not confuse ``--heatbath_truncation`` with the cutoff: it discards *parents*
by weight before expansion even starts, and its default of 0 (keep everything) is
almost always what you want. Setting it to 1e-4 on a 59,536-determinant Fe4S4
wavefunction cut the subspace to 506.

Seeds, so the same driver produces every row of a seed comparison:

    --seed files       determinant files (default: upstream's four Fe4S4 files)
    --seed hf          the Hartree-Fock determinant alone, the no-input null
    --seed from-alpha  an alpha list interleaved with itself (a TPB-shaped space)
    --seed strings     an arbitrary bitstring file, e.g. sampled configurations

Usage:
    # Fe4S4 from upstream's shipped subspace, one cutoff
    python run_gdb_heatbath.py --cutoffs 1e-3

    # A ladder, stopping if the subspace would pass 2M determinants
    python run_gdb_heatbath.py --cutoffs 1e-3,1e-4,1e-5 --max_dim 2000000

    # The null hypothesis: no input subspace at all, just Hartree-Fock
    python run_gdb_heatbath.py --seed hf --cutoffs 1e-3,1e-4

    # Sharded across 4 ranks/GPUs. At --b_comm_size == ranks with t=1 the
    # expansion comes back already sharded for the next round, no gather needed.
    mpirun -np 4 python run_gdb_heatbath.py --b_comm_size 4 --cutoffs 1e-3,1e-4
    mpirun -np 4 python run_gdb_heatbath.py --b_comm_size 4 --device gpu --cutoffs 1e-4

    # Record the (dimension, energy) series for a comparison table
    python run_gdb_heatbath.py --cutoffs 1e-3,1e-4 --log fe4s4_ladder.json
"""

import argparse
import json
import sys
import time

import numpy as np

_UPSTREAM = "../../vendor/sbd-upstream"
_GDB_APP = f"{_UPSTREAM}/apps/chemistry_gdb_selected_basis_diagonalization"
_DEFAULT_FCIDUMP = f"{_GDB_APP}/fcidump_Fe4S4.txt"
_DEFAULT_DETFILES = ",".join(f"{_GDB_APP}/det{i}.txt" for i in range(4))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Iteratively grow a GDB subspace by heatbath expansion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'gpu', 'gpu-omp', 'auto'],
                       help="Compute backend. 'gpu' (Thrust) is the only GPU backend "
                            "with GDB kernels, and it requires the helper dimension "
                            "to be 1, i.e. t_comm_size * b_comm_size == ranks")
    parser.add_argument('--fcidump', default=_DEFAULT_FCIDUMP,
                       help='FCIDUMP file defining the Hamiltonian')

    # --- the seed ---------------------------------------------------------
    parser.add_argument('--seed', default='files',
                       choices=['files', 'hf', 'from-alpha', 'strings'],
                       help='Where the starting subspace comes from')
    parser.add_argument('--detfiles', default=_DEFAULT_DETFILES,
                       help='--seed files: comma-separated files of 2*norb-bit '
                            'determinant strings')
    parser.add_argument('--alpha-file', default='', dest='alpha_file',
                       help='--seed from-alpha: a norb-bit alpha determinant list, '
                            'interleaved with itself into the full product basis')
    parser.add_argument('--alpha-limit', type=int, default=0, dest='alpha_limit',
                       help='--seed from-alpha: keep only the first N alpha strings '
                            '(the product costs N^2 determinants)')
    parser.add_argument('--strings-file', default='', dest='strings_file',
                       help='--seed strings: a file of 2*norb-bit determinant '
                            'strings, e.g. sampled configurations')

    # --- the ladder -------------------------------------------------------
    parser.add_argument('--cutoffs', default='1e-3',
                       help='Comma-separated heatbath cutoffs, smallest step last. '
                            'Each is a rung: rounds run at that cutoff until the '
                            'energy stops moving, then the next rung begins')
    parser.add_argument('--max_rounds', type=int, default=8,
                       help='Maximum rounds per rung')
    parser.add_argument('--energy_tol', type=float, default=1e-5,
                       help='Advance to the next rung once |dE| between rounds falls '
                            'below this (Hartree)')
    parser.add_argument('--max_dim', type=int, default=0,
                       help='Stop before diagonalizing a subspace larger than this '
                            '(0 = no cap). The cap is what keeps rows of a '
                            'comparison cost-matched')
    parser.add_argument('--heatbath_truncation', type=float, default=0.0,
                       help='Weight threshold applied to PARENTS before expanding. '
                            'Leave at 0 unless you mean to expand only from the '
                            'dominant determinants')
    parser.add_argument('--heatbath_batch_size', type=int, default=1000000,
                       help='Heatbath expansion batch size per rank')

    # --- diagonalization --------------------------------------------------
    parser.add_argument('--method', type=int, default=0, choices=[0, 1],
                       help='0=Davidson, 1=Davidson storing the Hamiltonian. GDB has '
                            'no Lanczos, so TPB methods 2 and 3 do not exist here')
    parser.add_argument('--tolerance', '--eps', type=float, default=1e-6,
                       dest='eps', help='Davidson convergence tolerance')
    parser.add_argument('--iteration', '--max_it', type=int, default=30,
                       dest='max_it', help='Maximum Davidson iterations per round')
    parser.add_argument('--block', '--max_nb', type=int, default=10,
                       dest='max_nb', help='Maximum number of basis vectors')
    parser.add_argument('--bit_length', type=int, default=64,
                       help='Bits per packed word')
    parser.add_argument('--carryover_type', type=int, default=2, choices=[2, 3],
                       help='Heatbath variant: 2 or 3. Types 0 and 1 do not expand, '
                            'so they cannot drive this loop')

    # --- decomposition ----------------------------------------------------
    parser.add_argument('--b_comm_size', type=int, default=1,
                       help='Basis shards. The only dimension that divides memory')
    parser.add_argument('--t_comm_size', type=int, default=1,
                       help='Task communicator size; must not exceed b_comm_size')
    parser.add_argument('--determinant_distribution', default='',
                       choices=['', 'input', 'equal-bra-a', 'count', 'count-sorted',
                                'grid-cyclic', 'grid-cyclic-balanced'],
                       help='How determinants are placed across b_comm')

    parser.add_argument('--log', default='', metavar='FILE',
                       help='Write the per-round (dimension, energy) series as JSON')

    return parser.parse_args()


def read_strings(paths):
    """Read bitstrings from one or more text files, in file order."""
    out = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            out.extend(line.strip() for line in handle if line.strip())
    return out


def interleave(alpha, beta):
    """Interleave a norb-bit alpha and beta string into one 2*norb-bit determinant.

    Bit ``2 * i`` is alpha orbital ``i`` and bit ``2 * i + 1`` is beta orbital ``i``,
    counting from the right -- the order the GDB app documents and ``from_strings``
    packs.
    """
    a_rev, b_rev = alpha[::-1], beta[::-1]
    return ''.join(a_rev[i] + b_rev[i] for i in range(len(a_rev)))[::-1]


def hartree_fock_string(norb, nelec, ms2):
    """The Hartree-Fock determinant: the lowest orbitals doubly occupied.

    Built from the FCIDUMP header alone, so ``--seed hf`` needs no input subspace at
    all. That is the point of it: if expansion from HF reaches the same place as
    expansion from a sampled subspace, the sampling added nothing.
    """
    n_alpha = (nelec + ms2) // 2
    n_beta = nelec - n_alpha
    if n_alpha > norb or n_beta > norb:
        raise ValueError(
            f"cannot place {n_alpha} alpha and {n_beta} beta electrons in {norb} "
            f"orbitals")
    alpha = ''.join('1' if i < n_alpha else '0' for i in range(norb))[::-1]
    beta = ''.join('1' if i < n_beta else '0' for i in range(norb))[::-1]
    return interleave(alpha, beta)


def shard_bounds(total, b_comm_size, index):
    """This rank's slice of a globally sorted list.

    Mirrors SBD's ``q = N/p``, remainder-to-the-low-ranks split
    (``balanced_begin``, framework/bit_manipulation.h:1523-1531).
    """
    quotient, remainder = divmod(total, b_comm_size)
    begin = index * quotient + min(index, remainder)
    return begin, begin + quotient + (1 if index < remainder else 0)


def build_seed(args, sbd, norb, nelec, ms2, rank):
    """The starting determinant array for this rank, plus a description."""
    total_bits = 2 * norb

    if args.seed == 'hf':
        strings = [hartree_fock_string(norb, nelec, ms2)]
        source = f"Hartree-Fock determinant ({nelec} electrons, MS2={ms2})"
    elif args.seed == 'from-alpha':
        if not args.alpha_file:
            raise ValueError("--seed from-alpha requires --alpha-file")
        alpha = read_strings([args.alpha_file])
        if args.alpha_limit:
            alpha = alpha[:args.alpha_limit]
        bad = [s for s in alpha if len(s) != norb]
        if bad:
            raise ValueError(
                f"--alpha-file strings must be {norb} bits, found {len(bad[0])}")
        strings = [interleave(a, b) for a in alpha for b in alpha]
        source = f"{args.alpha_file}: {len(alpha)} alpha -> {len(alpha)}^2 product"
    else:
        paths = ([args.strings_file] if args.seed == 'strings'
                 else [p for p in args.detfiles.split(',') if p])
        if not paths or not all(paths):
            raise ValueError(f"--seed {args.seed} requires input file(s)")
        strings = read_strings(paths)
        source = f"{len(paths)} file(s): {', '.join(paths)}"

    bad = [s for s in strings if len(s) != total_bits]
    if bad:
        raise ValueError(
            f"determinant strings must be {total_bits} bits (2 * NORB), "
            f"found one of length {len(bad[0])}")
    if not strings:
        raise ValueError("the seed is empty")

    # Sorted globally here so that slicing gives the disjoint, ordered shards
    # gdb_diag requires.
    whole = sbd.sort_bitarray_array(
        sbd.from_strings(strings, args.bit_length, total_bits, device=args.device),
        device=args.device)
    if args.b_comm_size == 1:
        return whole, source
    begin, end = shard_bounds(whole.shape[0], args.b_comm_size,
                              rank % args.b_comm_size)
    return whole[begin:end], source


def reshard(det, args, comm, rank, size):
    """Put the expanded list back into the shape the next round expects.

    ``HeatbathExpansion`` ends with a global sort and a count rebalance over the
    WORLD communicator (gdb/expansion.h:821-822), so the result is already
    globally sorted, balanced and disjoint across world ranks. When
    ``b_comm_size == ranks`` and ``t_comm_size == 1`` the b_comm position equals the
    world rank, so that is *exactly* a valid next-round shard and nothing needs to
    move -- which is the configuration a GPU run uses anyway, since Thrust requires
    the helper dimension to be 1.

    Any other grid means world position and b_comm position disagree, and the only
    portable fix from Python is to gather and re-slice. That materializes the whole
    list on every rank, so it is the slow path and says so.
    """
    if args.b_comm_size == 1 or size == 1:
        if size == 1:
            return det
        # Every rank must hold the whole basis; the shards are disjoint pieces of it.
        gathered = comm.allgather(det)
        return np.concatenate([g for g in gathered if g.size], axis=0)

    if args.b_comm_size == size and args.t_comm_size == 1:
        return det   # already aligned: world rank == b_comm position

    if rank == 0:
        print("  NOTE: b_comm_size != ranks (or t_comm_size > 1), so the expanded "
              "list has to be gathered and re-sliced, which puts the whole list on "
              "every rank. Use --b_comm_size == ranks with --t_comm_size 1 to avoid "
              "it.", flush=True)
    gathered = comm.allgather(det)
    whole = np.concatenate([g for g in gathered if g.size], axis=0)
    begin, end = shard_bounds(whole.shape[0], args.b_comm_size,
                              rank % args.b_comm_size)
    return whole[begin:end]


def main():
    args = parse_args()

    import sbd
    from mpi4py import MPI

    sbd.init(device=args.device)
    backend = sbd.get_backend(args.device)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    fcidump = backend.LoadFCIDump(args.fcidump)
    header = fcidump.header
    norb = int(header["NORB"])
    nelec = int(header["NELEC"])
    ms2 = int(header.get("MS2", 0))

    cutoffs = [float(c) for c in args.cutoffs.split(',') if c.strip()]
    if not cutoffs:
        print("ERROR: --cutoffs is empty", file=sys.stderr)
        return 1

    try:
        det, source = build_seed(args, sbd, norb, nelec, ms2, rank)
    except ValueError as exc:
        if rank == 0:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    def global_dim(local):
        """Determinants across the whole basis, counted once each.

        Summing over the world communicator would over-count: ranks sharing a b_comm
        position hold identical shards, so with t_comm_size or the helper dimension
        above 1 each shard is counted once per replica. Contribute only from the ranks
        that are one-per-b-position -- with the layout
        ``rank = h*(b*t) + t_index*b + b_index``, those are exactly ranks
        ``0 .. b_comm_size - 1`` -- which gives the b_comm sum without building a
        sub-communicator.
        """
        if args.b_comm_size == 1:
            return int(local.shape[0])
        mine = int(local.shape[0]) if rank < args.b_comm_size else 0
        return comm.allreduce(mine, op=MPI.SUM)

    # Collective: every rank must call it, so it cannot live inside a rank guard.
    # This deadlocks otherwise -- rank 0 waits in the allreduce while the others walk
    # into gdb_diag's own collectives. It hides in serial testing because the
    # b_comm_size == 1 path short-circuits without communicating.
    starting_dim = global_dim(det)
    if args.b_comm_size > 1 and det.shape[0] == 0:
        # Printed by whichever rank is starved, which is the informative one.
        print(f"  NOTE: rank {rank}'s shard is empty -- the seed has fewer "
              f"determinants than b_comm_size. 'count' and 'count-sorted' placement "
              f"cannot take an empty shard; 'input' and the grid schemes can.",
              flush=True)

    if rank == 0:
        grid = args.t_comm_size * args.b_comm_size
        helper = size // grid if grid and size % grid == 0 else 0
        print("=" * 78)
        print("GDB + heatbath expansion - iterative subspace growth")
        print("=" * 78)
        print(f"  Device: {sbd.get_device()}    ranks: {size}")
        print(f"  Decomposition: t={args.t_comm_size} b={args.b_comm_size} "
              f"helper={helper}")
        print(f"  System: {norb} orbitals ({2 * norb} spin orbitals), "
              f"{nelec} electrons")
        print(f"  Seed: {source}")
        print(f"  Starting dimension: {starting_dim}")
        print(f"  Cutoff ladder: {cutoffs}")
        print(f"  Per rung: up to {args.max_rounds} rounds, advance when "
              f"|dE| < {args.energy_tol:g}")
        if args.max_dim:
            print(f"  Dimension cap: {args.max_dim}")
        if args.heatbath_truncation:
            print(f"  WARNING: --heatbath_truncation {args.heatbath_truncation:g} "
                  f"discards parents BEFORE expanding, which can shrink the "
                  f"subspace rather than grow it.")
        print()
        print(f"  {'rung':>4} {'round':>5} {'dimension':>12} {'energy':>18} "
              f"{'dE':>12} {'secs':>7}")

    history = []
    previous_energy = None      # for the displayed dE, continuous across rungs
    stop_reason = "ladder complete"

    for rung, cutoff in enumerate(cutoffs):
        rung_converged = False
        # Deliberately NOT seeded from the previous rung's energy. Round 0 of a new
        # rung re-diagonalizes the subspace the previous rung already expanded, so
        # its energy is essentially unchanged -- testing convergence against that
        # would declare the rung done before its smaller cutoff had expanded
        # anything at all.
        rung_baseline = None
        for round_index in range(args.max_rounds):
            dim = global_dim(det)
            if args.max_dim and dim > args.max_dim:
                stop_reason = f"dimension {dim} exceeds --max_dim {args.max_dim}"
                if rank == 0:
                    print(f"  stopping: {stop_reason}")
                return _finish(args, history, stop_reason, rank)

            config = backend.GDB_SBD()
            config.method = args.method
            config.eps = args.eps
            config.max_it = args.max_it
            config.max_nb = args.max_nb
            config.bit_length = args.bit_length
            config.b_comm_size = args.b_comm_size
            config.t_comm_size = args.t_comm_size
            config.carryover_type = args.carryover_type
            config.heatbath_cutoff = cutoff
            config.heatbath_truncation = args.heatbath_truncation
            config.heatbath_batch_size = args.heatbath_batch_size

            start = time.perf_counter()
            result = sbd.gdb_diag(
                fcidump, det, config, device=args.device,
                determinant_distribution=args.determinant_distribution,
            )
            elapsed = time.perf_counter() - start

            energy = result["energy"]
            delta = None if previous_energy is None else energy - previous_energy
            if rank == 0:
                shown = "" if delta is None else f"{delta:+.6f}"
                print(f"  {rung:>4} {round_index:>5} {dim:>12} {energy:>18.10f} "
                      f"{shown:>12} {elapsed:>7.1f}", flush=True)
            history.append({
                "rung": rung, "cutoff": cutoff, "round": round_index,
                "dimension": dim, "energy": energy,
                "delta_energy": delta, "seconds": elapsed,
            })

            grown = result["carryover_det"]
            det = reshard(grown, args, comm, rank, size)
            new_dim = global_dim(det)
            if rank == 0 and new_dim < dim:
                print(f"  NOTE: the expansion returned fewer determinants "
                      f"({new_dim} < {dim}). With --heatbath_truncation > 0 the "
                      f"parents are pruned before expanding.")

            # Energy-targeted advance, which is the standard HCI protocol: a rung is
            # done when another round at the same cutoff no longer moves the answer.
            rung_delta = None if rung_baseline is None else energy - rung_baseline
            rung_baseline = energy
            previous_energy = energy
            if rung_delta is not None and abs(rung_delta) < args.energy_tol:
                rung_converged = True
                break
            # A rung is equally done if the subspace stopped growing: the expansion
            # has reached its fixed point for this cutoff.
            if new_dim == dim:
                rung_converged = True
                if rank == 0:
                    print(f"  rung {rung} at cutoff {cutoff:g}: subspace stopped "
                          f"growing at {dim} -- its fixed point for this cutoff")
                break

        if rank == 0 and not rung_converged:
            print(f"  rung {rung} at cutoff {cutoff:g}: hit --max_rounds "
                  f"{args.max_rounds} without converging")

    return _finish(args, history, stop_reason, rank)


def _finish(args, history, stop_reason, rank):
    """Print the summary and optionally write the series."""
    if rank != 0:
        return 0
    if not history:
        print("no rounds completed")
        return 1
    first, last = history[0], history[-1]
    print()
    print("=" * 78)
    print(f"  {stop_reason}")
    print(f"  Start: dim {first['dimension']}, E = {first['energy']:.10f}")
    print(f"  End:   dim {last['dimension']}, E = {last['energy']:.10f}")
    print(f"  Gain:  {last['energy'] - first['energy']:+.6f} Hartree over "
          f"{len(history)} round(s), "
          f"{sum(h['seconds'] for h in history):.1f} s total")
    print("  The energy is variational, so it must not rise: a positive gain means "
          "the subspace shrank or a round failed to converge.")
    if args.log:
        with open(args.log, "w", encoding="utf-8") as handle:
            json.dump({"stop_reason": stop_reason, "rounds": history}, handle,
                      indent=2)
        print(f"  Series written to {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
