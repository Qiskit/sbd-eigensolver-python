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

    --subspace-from files       determinant files (default: upstream's four Fe4S4 files)
    --subspace-from hf          the Hartree-Fock determinant alone, the no-input null
    --subspace-from from-alpha  an alpha list interleaved with itself (a TPB-shaped space)
    --subspace-from strings     an arbitrary bitstring file, e.g. sampled configurations

Usage:
    # Fe4S4 from upstream's shipped subspace, one cutoff
    python run_gdb_heatbath.py --cutoffs 1e-3

    # A ladder, stopping if the subspace would pass 2M determinants
    python run_gdb_heatbath.py --cutoffs 1e-3,1e-4,1e-5 --max_dim 2000000

    # The null hypothesis: no input subspace at all, just Hartree-Fock
    python run_gdb_heatbath.py --subspace-from hf --cutoffs 1e-3,1e-4

    # Sharded across 4 ranks/GPUs. At --b_comm_size == ranks with t=1 the
    # expansion comes back already sharded for the next round, no gather needed.
    mpirun -np 4 python run_gdb_heatbath.py --b_comm_size 4 --cutoffs 1e-3,1e-4
    mpirun -np 4 python run_gdb_heatbath.py --b_comm_size 4 --device gpu --cutoffs 1e-4

    # Record the (dimension, energy) series for a comparison table
    python run_gdb_heatbath.py --cutoffs 1e-3,1e-4 --log fe4s4_ladder.json
"""

import argparse
import json
import os
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
        # No abbreviations: --resume_from would otherwise resolve to
        # --resume_from_prefix, taking a prefix where the SQD driver's --resume_from
        # takes a file path, and the mix-up would pass silently.
        allow_abbrev=False,
    )

    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'gpu', 'gpu-omp', 'auto'],
                       help="Compute backend. 'gpu' (Thrust) is the only GPU backend "
                            "with GDB kernels, and it requires the helper dimension "
                            "to be 1, i.e. t_comm_size * b_comm_size == ranks")
    parser.add_argument('--fcidump', default=_DEFAULT_FCIDUMP,
                       help='FCIDUMP file defining the Hamiltonian')

    # --- the seed ---------------------------------------------------------
    parser.add_argument('--skip-weight-check', action='store_true',
                       dest='skip_weight_check',
                       help='Do not verify that every seed determinant has the '
                            'expected alpha/beta electron count. The check costs '
                            'about 0.02 s per million determinants and catches a '
                            'wrong bit order, which otherwise diagonalizes to a '
                            'plausible wrong energy and aborts during expansion')
    parser.add_argument('--subspace-from', default='files',
                       dest='subspace_from',
                       choices=['files', 'hf', 'from-alpha', 'strings'],
                       help='Where the starting subspace comes from')
    parser.add_argument('--detfiles', default=_DEFAULT_DETFILES,
                       help='--subspace-from files: comma-separated files of 2*norb-bit '
                            'determinant strings')
    parser.add_argument('--alpha-file', '--from-alpha', default='', dest='alpha_file',
                       help='--subspace-from from-alpha: a norb-bit alpha determinant list, '
                            'interleaved with itself into the full product basis')
    parser.add_argument('--alpha-limit', type=int, default=0, dest='alpha_limit',
                       help='--subspace-from from-alpha: keep only the first N alpha strings '
                            '(the product costs N^2 determinants)')
    parser.add_argument('--strings-file', default='', dest='strings_file',
                       help='--subspace-from strings: a file of 2*norb-bit determinant '
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
    parser.add_argument('--bit_length', type=int, default=62,
                       help='Bits per packed word: even, at most 62')
    parser.add_argument('--carryover_type', type=int, default=2, choices=[2, 3],
                       help='Heatbath variant: 2 or 3. Types 0 and 1 do not expand, '
                            'so they cannot drive this loop')

    # --- decomposition ----------------------------------------------------
    parser.add_argument('--b_comm_size', type=int, default=1,
                       help='Basis shards. The only dimension that divides memory')
    parser.add_argument('--t_comm_size', type=int, default=1,
                       help='Task communicator size; must not exceed b_comm_size')
    parser.add_argument('--determinant_distribution', default='grid-cyclic-balanced',
                       choices=['', 'input', 'equal-bra-a', 'count', 'count-sorted',
                                'grid-cyclic', 'grid-cyclic-balanced'],
                       help='How determinants are placed across b_comm. The default '
                            'keeps a heat-bath subspace balanced, where equal-bra-a '
                            '(gdb_diag\'s own default, \'\') can put most of it on '
                            'one rank. It also accepts an empty shard, which a '
                            'one-determinant seed leaves on all but one rank')

    parser.add_argument('--log', default='', metavar='FILE',
                       help='Write the per-round (dimension, energy) series as JSON')

    # --- saving and resuming ----------------------------------------------
    parser.add_argument('--savename', default='', metavar='PREFIX',
                       help='Save the final round\'s state (determinants and '
                            'amplitudes) as PREFIX000000.bin ... , one file per '
                            'basis shard, for examples/analysis/variance.py or '
                            '--resume_from_prefix. Overwrites any earlier state there')
    parser.add_argument('--checkpoint_prefix', dest='checkpoint', default='', metavar='PREFIX',
                       help='After every round, save that round\'s state and the '
                            'ladder position (PREFIX.json), so a run stopped at any '
                            'point can be continued with --resume_from_prefix PREFIX. Only '
                            'the latest round is kept unless --checkpoint_frequency '
                            'is set. One run per prefix: a run removes scratch files '
                            'an interrupted earlier run left under it')
    parser.add_argument('--checkpoint_frequency', type=int, default=0, metavar='N',
                       help='With --checkpoint_prefix, also keep a copy of every Nth '
                            'round under PREFIX_rNNNN, alongside the latest one '
                            '(0 = keep the latest only)')
    parser.add_argument('--resume_from_prefix', dest='resume_from', default='', metavar='PREFIX',
                       help='Continue from a --checkpoint_prefix or --savename state: its '
                            'determinants are the starting subspace, its amplitudes '
                            'Davidson\'s starting vector. With a .json beside it '
                            '(a checkpoint), the ladder continues where it stopped '
                            'unless --cutoffs is given; otherwise --cutoffs runs from '
                            'the start. Replaces --subspace-from')

    args = parser.parse_args()
    # Even, and at most 62. 64 overflows the shift in SBD's bitadvance(), reached
    # whenever a placement scheme redistributes determinants. SBD's alpha/beta
    # conversions assume an even word size once a determinant spans two words:
    # getHalfDets (chemistry/gdb/helper.h) in every build, and DetFromAlphaBeta
    # under SBD_TRADMODE. The heatbath expansion then crashes or, with
    # SBD_TRADMODE off, returns a wrong energy (h2o at 25, 31, 33 and 47).
    if args.bit_length % 2 or not 2 <= args.bit_length <= 62:
        parser.error(f"--bit_length must be even and between 2 and 62, "
                     f"got {args.bit_length}")

    if args.checkpoint_frequency < 0:
        parser.error("--checkpoint_frequency must be >= 0")
    if args.checkpoint_frequency and not args.checkpoint:
        parser.error("--checkpoint_frequency needs --checkpoint_prefix")
    args.cutoffs_given = any(a == '--cutoffs' or a.startswith('--cutoffs=')
                             for a in sys.argv[1:])

    # One determinant cannot be sharded: rank 0 owns it, the rest get empty shards,
    # and a single parent gives OpenMP nothing to divide either.
    if args.subspace_from == 'hf' and args.b_comm_size > 1 and not args.resume_from:
        print("WARNING: --subspace-from hf is a single determinant, so "
              f"--b_comm_size {args.b_comm_size} leaves the", file=sys.stderr)
        print("  other ranks with empty shards and runs the first rounds on one "
              "core. Use -np 1.", file=sys.stderr)

    return args


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


# One 256-entry popcount table, applied to the packed determinant words by viewing
# them as bytes. numpy.bitwise_count would be ~9x faster but needs numpy 2.0, and
# this package's floor is 1.19; at 0.018 s per million determinants the table is
# already a ~6% addition to the read-and-pack this replaces nothing of.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# Interleaved layout: alpha sits on the even global bit positions, beta on the odd
# ones. Word j holds global bits j*bit_length upward, so its alpha bits are the
# even positions within it when j*bit_length is even and the odd ones when it is
# odd. That makes the masks per word, and lets them handle any bit_length.
_EVEN_BITS = 0x5555555555555555
_ODD_BITS = 0xAAAAAAAAAAAAAAAA


def _spin_masks(n_words, bit_length):
    """Per-word (alpha, beta) masks for an interleaved determinant."""
    used = (1 << bit_length) - 1
    even, odd = _EVEN_BITS & used, _ODD_BITS & used
    starts_even = [(j * bit_length) % 2 == 0 for j in range(n_words)]
    alpha = np.array([even if s else odd for s in starts_even], dtype=np.uint64)
    beta = np.array([odd if s else even for s in starts_even], dtype=np.uint64)
    return alpha, beta


def _popcount_rows(words):
    """Bits set per row of a (n, words) uint64 array."""
    # Not reshape(n, -1): with n == 0 -- an empty shard, as a one-determinant seed
    # leaves on every rank but one -- NumPy cannot infer the other dimension.
    return _POPCOUNT[words.view(np.uint8)].reshape(words.shape[0], 8 * words.shape[1]).sum(
        axis=1, dtype=np.int64)


def spin_weights(det, bit_length):
    """Per-determinant (n_alpha, n_beta) for a packed interleaved determinant array."""
    words = np.ascontiguousarray(det, dtype=np.uint64)
    if words.ndim == 1:
        words = words.reshape(1, -1)
    alpha, beta = _spin_masks(words.shape[1], bit_length)
    return _popcount_rows(words & alpha), _popcount_rows(words & beta)


def check_spin_weights(det, nelec, ms2, bit_length):
    """Refuse a determinant list whose electron counts per spin are not uniform.

    A wrong bit order is the failure this catches, and it is worth catching here
    because it does not fail cleanly downstream: concatenated ``[beta | alpha]``
    strings (what qiskit-addon-sqd emits) diagonalize to a plausible-looking
    energy and only abort later inside the heatbath expansion. The occupation
    density cannot catch it either, since permuting bits preserves how many are
    set, so it still sums to the right electron count.

    Costs about 0.018 s per million determinants -- a few percent of the read and
    pack that precede it.
    """
    want_a = (nelec + ms2) // 2
    want_b = nelec - want_a
    got_a, got_b = spin_weights(det, bit_length)
    bad = (got_a != want_a) | (got_b != want_b)
    n_bad = int(bad.sum())
    if not n_bad:
        return
    first = int(np.argmax(bad))
    raise ValueError(
        f"{n_bad} of {len(bad)} determinants do not have {want_a} alpha and "
        f"{want_b} beta electrons (first at index {first}: "
        f"{int(got_a[first])} alpha, {int(got_b[first])} beta).\n"
        "  The usual cause is bit order: GDB expects alpha and beta INTERLEAVED "
        "(bit 2*i alpha orbital i, bit 2*i+1 beta orbital i), while "
        "qiskit-addon-sqd emits them CONCATENATED as [beta | alpha]. Interleave "
        "before packing -- see examples/gdb/README.md.\n"
        "  The other cause is samples that were never postselected on the target "
        "Hamming weight.\n"
        "  Pass --skip-weight-check to proceed anyway."
    )


def looks_like_a_single_determinant(density, tolerance=1e-6):
    """True when every occupancy is 0 or 1: a one-determinant wavefunction.

    For a subspace of more than one determinant that means Davidson never iterated --
    the start vector had no coupling to the rest, so the residual was zero at once.
    """
    return all(min(abs(x), abs(1.0 - x)) < tolerance for x in density)


def hartree_fock_string(norb, nelec, ms2):
    """The Hartree-Fock determinant: the lowest orbitals doubly occupied.

    Built from the FCIDUMP header alone, so ``--subspace-from hf`` needs no input subspace at
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

    if args.subspace_from == 'hf':
        strings = [hartree_fock_string(norb, nelec, ms2)]
        source = f"Hartree-Fock determinant ({nelec} electrons, MS2={ms2})"
    elif args.subspace_from == 'from-alpha':
        if not args.alpha_file:
            raise ValueError("--subspace-from from-alpha requires --alpha-file")
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
        paths = ([args.strings_file] if args.subspace_from == 'strings'
                 else [p for p in args.detfiles.split(',') if p])
        if not paths or not all(paths):
            raise ValueError(f"--subspace-from {args.subspace_from} requires input file(s)")
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


def check_save_dirs(args, comm, rank):
    """Every --savename/--checkpoint_prefix directory must exist before any solve.

    SBD writes its state files without checking that they opened, so a missing
    directory loses the save while the run reports success.
    """
    problem = None
    if rank == 0:
        for flag, prefix in (('--savename', args.savename), ('--checkpoint_prefix', args.checkpoint)):
            directory = os.path.dirname(os.path.abspath(prefix)) if prefix else None
            if directory and not os.path.isdir(directory):
                problem = (f"{flag} {prefix!r}: directory {directory!r} does not exist "
                           f"(SBD would lose the save without an error); create it first")
                break
    problem = comm.bcast(problem, root=0)
    if problem:
        raise ValueError(problem)


def state_files(prefix):
    """The GDB state files saved under ``prefix``: PREFIX000000.bin, ... (local call)."""
    files, index = [], 0
    while os.path.exists(f"{prefix}{index:06d}.bin"):
        files.append(f"{prefix}{index:06d}.bin")
        index += 1
    return files


def promote_state(scratch, target, comm, extra=None):
    """Move one round's state from ``scratch`` to ``target``, replacing what was there.

    Rank 0 does every rename after a barrier, so each shard file is complete first.
    A complete state is on disk at every moment, so an interrupt anywhere -- Ctrl-C
    included -- loses at most the round in progress:

    1. the previous state moves aside, to ``target + ".prev-"``, JSON first;
    2. the new files move in, and the new JSON (``extra``) is written last;
    3. the previous state is deleted.

    Until 2 completes, ``target`` has no JSON and the previous state is at
    ``.prev-``, which :func:`resume_prefix` falls back to.
    """
    comm.Barrier()
    if comm.Get_rank() == 0:
        prev = target + PREV
        _remove_state(prev)
        if os.path.exists(target + ".json"):
            os.replace(target + ".json", prev + ".json")
        for path in state_files(target):
            os.replace(path, prev + path[len(target):])
        for path in state_files(scratch):
            os.replace(path, target + path[len(scratch):])
        if extra is not None:
            tmp = target + ".json.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(extra, handle, indent=2)
            os.replace(tmp, target + ".json")
        _remove_state(prev)
    comm.Barrier()


PREV = ".prev-"


def _remove_state(prefix):
    if os.path.exists(prefix + ".json"):
        os.remove(prefix + ".json")
    for path in state_files(prefix):
        os.remove(path)


def resume_prefix(prefix):
    """The prefix to resume from: ``prefix``, or its previous state if a save was cut short.

    A checkpoint's JSON is written last, so a checkpoint with state files but no JSON
    was interrupted while being replaced; the complete state is the previous one.
    """
    if not os.path.exists(prefix + ".json") and os.path.exists(prefix + PREV + ".json"):
        return prefix + PREV
    return prefix


def copy_state(source, target, comm):
    """Copy a promoted state (and its JSON, if any) to ``target``, for history."""
    import shutil

    comm.Barrier()
    if comm.Get_rank() == 0:
        # Clear the target first, as promote_state does: an earlier, larger state
        # under the same prefix would otherwise leave extra shard files behind.
        if os.path.exists(target + ".json"):
            os.remove(target + ".json")
        for stale in state_files(target):
            os.remove(stale)
        for path in state_files(source):
            shutil.copyfile(path, target + path[len(source):])
        if os.path.exists(source + ".json"):
            shutil.copyfile(source + ".json", target + ".json")
    comm.Barrier()


def load_resume(args, sbd, norb, comm, rank):
    """The saved state's determinants as this rank's seed shard, plus its ladder record."""
    from sbd import analysis

    # A save interrupted while replacing the checkpoint leaves the complete state
    # aside; resume from that instead.
    chosen = comm.bcast(resume_prefix(args.resume_from) if rank == 0 else None, root=0)
    if chosen != args.resume_from and rank == 0:
        print(f"  NOTE: {args.resume_from}.json is missing -- the last save was "
              f"interrupted -- so resuming from the previous complete state, "
              f"{chosen}", flush=True)
    args.resume_from = chosen
    found = comm.bcast(state_files(args.resume_from) if rank == 0 else None, root=0)
    if not found:
        raise ValueError(f"--resume_from_prefix {args.resume_from!r}: no state files "
                         f"({args.resume_from}000000.bin ...) found")
    det = analysis.load_gdb_shards(args.resume_from, norb, args.bit_length,
                                   args.b_comm_size, comm)
    record = None
    if rank == 0 and os.path.exists(args.resume_from + ".json"):
        with open(args.resume_from + ".json", encoding="utf-8") as handle:
            record = json.load(handle)
    record = comm.bcast(record, root=0)
    source = (f"resumed from {args.resume_from} ({len(found)} file(s)"
              + (f", after rung {record['rung']} round {record['round']}" if record else "")
              + ")")
    return det, source, record, len(found)


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

    resumed, resumed_files = None, 0
    try:
        check_save_dirs(args, comm, rank)
        if args.resume_from:
            det, source, resumed, resumed_files = load_resume(args, sbd, norb, comm, rank)
        else:
            det, source = build_seed(args, sbd, norb, nelec, ms2, rank)
        if not args.skip_weight_check:
            check_spin_weights(det, nelec, ms2, args.bit_length)
    except ValueError as exc:
        # Every rank, not rank 0 only: the error may be raised on one rank alone,
        # and run() aborts the job as soon as any rank returns.
        print(f"[rank {rank}] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    # Where the ladder starts. A checkpoint holds the last completed round's solved
    # state -- before its expansion, which SBD returns but does not save. So the first
    # resumed round re-solves that subspace (from its saved amplitudes, so quickly) and
    # expands it again. Without --cutoffs, carry on with the checkpoint's ladder from
    # that rung. With --cutoffs, they become further rungs after the checkpoint's: the
    # ladder up to its rung is kept, so the history -- and --log -- stay one record of
    # the whole calculation across any number of resumes.
    first_rung, first_round, history = 0, 0, []
    if resumed:
        history = list(resumed.get("history", []))
        done = [float(c) for c in resumed["cutoffs"]][: resumed["rung"] + 1]
        if args.cutoffs_given:
            cutoffs = done + cutoffs
            first_rung = len(done)
        else:
            cutoffs = [float(c) for c in resumed["cutoffs"]]
            first_rung = resumed["rung"]
            # Re-solving the checkpointed round counts as that round again, so the
            # rung keeps its numbering and its --max_rounds budget.
            first_round = resumed["round"]
    # Each round writes its solved state here; it becomes the checkpoint and the
    # final --savename by rename, so no round is diagonalized twice.
    scratch = None
    if args.savename or args.checkpoint:
        base = args.checkpoint or args.savename
        scratch = comm.bcast(base + f".round-{os.getpid()}-" if rank == 0 else None, root=0)
        if rank == 0:
            # A run interrupted mid-round leaves its scratch state behind. It is never
            # read -- a resume uses the promoted state -- so clear it, rather than
            # accumulate one round's worth of files per interruption.
            import glob
            stale = glob.glob(glob.escape(base) + ".round-*-[0-9][0-9][0-9][0-9][0-9][0-9].bin")
            for path in stale:
                os.remove(path)
            if stale:
                print(f"  NOTE: removed {len(stale)} scratch file(s) an interrupted run "
                      f"left under {base}", flush=True)
    # SBD's GDB loader reads file b on b_comm position b only, then passes what does
    # not belong there around the ring. With more shards than files every amplitude
    # still arrives; with fewer, the files past b_comm_size are never read, and
    # Davidson would start from part of the state. That changes only how fast the
    # first round converges, not its answer, but start fresh rather than from a
    # silently partial vector.
    loadname = args.resume_from or ""
    resolving = bool(args.resume_from)
    if loadname and resumed_files > args.b_comm_size:
        loadname = ""
        if rank == 0:
            print(f"  NOTE: the checkpoint has {resumed_files} files but --b_comm_size is "
                  f"{args.b_comm_size}, and SBD can only load amplitudes onto at least as "
                  f"many shards as files. The determinants are resumed; Davidson starts "
                  f"fresh, so the first round takes longer.", flush=True)

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
    starved = comm.allreduce(int(args.b_comm_size > 1 and det.shape[0] == 0))
    if starved and sbd.get_device() == "gpu":
        # sbd.gdb_diag refuses this too, but the remedy is specific to this driver.
        if rank == 0:
            print(f"ERROR: the seed has {starting_dim} determinant(s) for --b_comm_size "
                  f"{args.b_comm_size}, so {starved} shard(s) would be empty, and the "
                  f"gpu (Thrust) GDB solve hangs on an empty shard. Run the first "
                  f"rounds on one GPU with --checkpoint_prefix, then continue on more "
                  f"with --resume_from_prefix:\n"
                  f"  python run_gdb_heatbath.py --device gpu ... --checkpoint_prefix ck-\n"
                  f"  mpirun -np {args.b_comm_size} python run_gdb_heatbath.py --device gpu "
                  f"--b_comm_size {args.b_comm_size} --resume_from_prefix ck- ...",
                  file=sys.stderr, flush=True)
        return 1
    if starved and rank == 0:
        print(f"  NOTE: {starved} shard(s) are empty -- the seed has fewer determinants "
              f"than --b_comm_size. Such rounds run, but are not saved, and 'count' and "
              f"'count-sorted' placement cannot take them. A one-determinant seed is far "
              f"faster on one rank.", flush=True)

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
        print(f"  Cutoff ladder: {cutoffs}"
              + (f", resuming at rung {first_rung} round {first_round}" if resumed else ""))
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
              f"{'dE':>12} {'secs':>7}" + (f" {'save':>6}" if scratch else ""))

    previous_energy = history[-1]["energy"] if history else None
    rises, warned_rise = 0, False
    stop_reason = "ladder complete"
    last_round = None

    for rung, cutoff in enumerate(cutoffs):
        if rung < first_rung:
            continue
        rung_converged = False
        # Deliberately NOT seeded from the previous rung's energy. Round 0 of a new
        # rung re-diagonalizes the subspace the previous rung already expanded, so
        # its energy is essentially unchanged -- testing convergence against that
        # would declare the rung done before its smaller cutoff had expanded
        # anything at all.
        rung_baseline = None
        start_round = first_round if rung == first_rung else 0
        for round_index in range(start_round, max(args.max_rounds, start_round + 1)):
            dim = global_dim(det)
            if args.max_dim and dim > args.max_dim:
                stop_reason = f"dimension {dim} exceeds --max_dim {args.max_dim}"
                if rank == 0:
                    print(f"  stopping: {stop_reason}")
                return _save_and_finish(args, history, stop_reason, last_round, comm, rank)

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

            # SBD's save reads each shard's first determinant unchecked, so a round
            # with an empty shard -- the first rounds of a one-determinant seed over
            # several ranks -- cannot be saved. Skip saving it; those subspaces are
            # tiny, and saving resumes as soon as every rank holds determinants.
            can_save = bool(scratch) and bool(comm.allreduce(int(det.shape[0] == 0)) == 0)
            if scratch and not can_save and rank == 0:
                print(f"  NOTE: round {round_index} not saved: a shard is empty "
                      f"(SBD cannot save one). Saving starts once every rank holds "
                      f"determinants.", flush=True)

            start = time.perf_counter()
            result = sbd.gdb_diag(
                fcidump, det, config, device=args.device,
                determinant_distribution=args.determinant_distribution,
                loadname=loadname, savename=scratch if can_save else "",
            )
            # The first round of a resumed run re-solves the checkpointed subspace,
            # which repeats the checkpoint's last energy; it is marked as such.
            resolved = resolving
            resolving = False
            loadname = ""     # only the first round starts from the resumed amplitudes
            elapsed = time.perf_counter() - start

            energy = result["energy"]
            delta = None if previous_energy is None else energy - previous_energy
            # Variational: a round's subspace contains the previous one, so its energy
            # can only fall. A rise means the solve stopped on a higher state.
            # (Not with --heatbath_truncation, which can shrink the subspace.)
            rose = (delta is not None and delta > RISE_TOL
                    and not args.heatbath_truncation)
            if rose:
                rises += 1
            # Only the seed round can be a no-op; after that the subspace contains
            # its own excitation neighbours.
            if (rank == 0 and rung == 0 and round_index == 0 and dim > 1
                    and looks_like_a_single_determinant(result["density"])):
                print("  NOTE: the seed diagonalization returned a single "
                      "determinant, so Davidson did")
                print("  not iterate there -- the ladder effectively starts from one "
                      "determinant.", flush=True)
            history.append({
                "rung": rung, "cutoff": cutoff, "round": round_index,
                "dimension": dim, "energy": energy,
                "delta_energy": delta, "seconds": elapsed,
                **({"resumed": True} if resolved else {}),
            })

            this_round = {"rung": rung, "round": round_index, "cutoff": cutoff,
                          "dimension": dim, "energy": energy}
            # The driver's own share of saving: the barrier, the renames into place,
            # the JSON and any numbered copy. Measured on every rank and reported as
            # the max, since a rank held at the barrier is paying for it too. SBD's
            # write of the state files happens inside gdb_diag, so it is in "secs".
            save_start = time.perf_counter()
            if can_save:
                last_round = this_round
            if can_save and args.checkpoint:
                record = dict(this_round, cutoffs=cutoffs, history=history,
                              bit_length=args.bit_length, fcidump=args.fcidump)
                promote_state(scratch, args.checkpoint, comm, extra=record)
                count = len(history)
                if args.checkpoint_frequency and count % args.checkpoint_frequency == 0:
                    copy_state(args.checkpoint, f"{args.checkpoint}_r{count:04d}-", comm)
            elif can_save and args.savename:
                promote_state(scratch, args.savename, comm)
            save_seconds = None
            if can_save:
                save_seconds = comm.allreduce(time.perf_counter() - save_start, op=MPI.MAX)
                history[-1]["save_seconds"] = save_seconds
            if rank == 0:
                shown = "" if delta is None else f"{delta:+.6f}"
                print(f"  {rung:>4} {round_index:>5} {dim:>12} {energy:>18.10f} "
                      f"{shown:>12} {elapsed:>7.1f}"
                      + ("" if save_seconds is None else f" {save_seconds:>6.2f}")
                      + ("  (re-solve of the checkpoint)" if resolved else "")
                      + ("  << ROSE" if rose else ""), flush=True)
                if rose and not warned_rise:
                    print("  WARNING: the energy rose although the subspace grew. The "
                          "energy of a larger subspace can only be lower, so that solve "
                          "did not find its lowest state, and these energies are not "
                          "trustworthy. Tighten --tolerance, raise --iteration, or start "
                          "from a better determinant (see the README).", flush=True)
                    warned_rise = True

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

    if rises and rank == 0:
        print(f"  WARNING: the energy rose in {rises} round(s); see the warning above.")
    return _save_and_finish(args, history, stop_reason, last_round, comm, rank)


# A re-solve of the same subspace can move by roundoff; anything larger is a real rise.
RISE_TOL = 1e-6


def _save_and_finish(args, history, stop_reason, last_round, comm, rank):
    """Leave the last completed round's state where --savename asks, then summarize.

    Every round already promoted its state to --checkpoint_prefix (or --savename), so all
    that is left is copying the checkpoint to a separate --savename, if both are set.
    """
    final_copy = None
    if last_round is not None:
        if args.savename and args.checkpoint:
            from mpi4py import MPI
            start = time.perf_counter()
            copy_state(args.checkpoint, args.savename, comm)
            final_copy = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
        if rank == 0 and (args.savename or args.checkpoint):
            where = args.savename or args.checkpoint
            print(f"  Saved the last round's state (rung {last_round['rung']} round "
                  f"{last_round['round']}, dim {last_round['dimension']}, "
                  f"E = {last_round['energy']:.10f}) as {where}000000.bin ..."
                  + ("" if final_copy is None else
                     f" (copied from the checkpoint in {final_copy:.2f} s)"))
    return _finish(args, history, stop_reason, rank, final_copy)


def _finish(args, history, stop_reason, rank, final_copy=None):
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
    saves = [h["save_seconds"] for h in history if h.get("save_seconds") is not None]
    if saves:
        total = sum(saves) + (final_copy or 0.0)
        print(f"  Saving: {total:.2f} s in the driver over {len(saves)} round(s) "
              f"(max over ranks; the 'save' column"
              + (", plus the final copy" if final_copy is not None else "") + "). "
              f"SBD's own write of the state files is inside 'secs'; its time is "
              f"the 'sbd: end save wavefunction' lines.")
    print("  The energy is variational, so it must not rise: a positive gain means "
          "the subspace shrank or a round failed to converge.")
    if args.log:
        with open(args.log, "w", encoding="utf-8") as handle:
            json.dump({"stop_reason": stop_reason, "rounds": history}, handle,
                      indent=2)
        print(f"  Series written to {args.log}")
    return 0


def run():
    """Run main(); on a failure on ANY rank, abort the whole MPI job.

    A rank that fails while the others wait in a collective would otherwise
    leave the job hanging until its time limit. The failing rank prints its own
    error, tagged with its rank, since it may not be rank 0.
    """
    try:
        code = main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        from mpi4py import MPI
        world = MPI.COMM_WORLD
        sys.stderr.write(f"[rank {world.Get_rank()}] interrupted\n")
        sys.stderr.flush()
        if world.Get_size() > 1:
            world.Abort(130)
        return 130
    except BaseException:
        import traceback
        from mpi4py import MPI
        world = MPI.COMM_WORLD
        sys.stderr.write(f"[rank {world.Get_rank()}] unhandled error:\n"
                         + traceback.format_exc())
        sys.stderr.flush()
        if world.Get_size() > 1:
            world.Abort(1)
        raise
    if code:
        from mpi4py import MPI
        world = MPI.COMM_WORLD
        if world.Get_size() > 1:
            sys.stderr.write(f"[rank {world.Get_rank()}] exiting with status "
                             f"{code}; aborting the MPI job\n")
            sys.stderr.flush()
            world.Abort(code)
    return code


if __name__ == "__main__":
    sys.exit(run())
