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
Standalone GDB diagonalization over an explicit determinant list, in memory.

GDB (general determinant basis) spans the subspace with the determinants it is
given. TPB spans it with the Cartesian product of an alpha and a beta list, so
TPB's dimension is |adet| x |bdet| while GDB's is exactly the number of
determinants passed. That is the whole point of GDB: an arbitrary sparse
subspace -- for example a set of sampled bitstrings used as sampled, with no
product completion.

This driver stays on the in-memory entry point (``sbd.gdb_diag``) throughout.
Determinant text is read in Python and handed to the binding as a list; SBD's
own file-based entry point is deliberately not used.

The determinant list can be sharded: with ``--b_comm_size R`` each rank passes only
its own slice, which is the only way GDB's memory scales, because b_comm is the one
dimension that divides the basis (and with it the excitation lookup). The derived
helper dimension divides work but not storage. ``--t_comm_size`` must not exceed
``--b_comm_size`` (one task per basis-ring station), and their product must divide
the rank count.

Usage:
    # Fe4S4, upstream's own GDB data: 4 files, 59,536 determinants, 36 orbitals.
    # Note that upstream publishes no reference energy for this case.
    python run_gdb_diag.py

    # Shard Fe4S4's four files one per rank: file i goes to b_comm position i,
    # which for this data is already the balanced globally-sorted split
    mpirun -np 4 -x OMP_NUM_THREADS=8 python run_gdb_diag.py --b_comm_size 4

    # Spend ranks on both named dimensions (t <= b, and t*b must divide the ranks)
    mpirun -np 8 python run_gdb_diag.py --b_comm_size 4 --t_comm_size 2

    # Choose how determinants are placed across b_comm
    mpirun -np 4 python run_gdb_diag.py --b_comm_size 4 \
        --determinant_distribution grid-cyclic

    # A subspace TPB can also express: interleave an alpha list with itself to
    # form the full |A|^2 product basis. This is the cross-check against
    # tpb_diag -- same subspace, two independent solvers, energies must agree.
    python run_gdb_diag.py \
        --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
        --from-alpha ../../vendor/sbd-upstream/data/h2o/h2o-1em3-alpha.txt \
        --alpha-limit 30

    # Grow the subspace with SBD's own heatbath expansion (one round; the
    # expanded list comes back as carryover_det)
    python run_gdb_diag.py --carryover_type 2 --heatbath_cutoff 1e-4
"""

import argparse
import sys
import time

import numpy as np

# Fe4S4 is the only general-determinant data upstream ships, and it lives in the
# app directory rather than under data/ (which is alpha-only, i.e. TPB).
_UPSTREAM = "../../vendor/sbd-upstream"
_GDB_APP = f"{_UPSTREAM}/apps/chemistry_gdb_selected_basis_diagonalization"
_DEFAULT_FCIDUMP = f"{_GDB_APP}/fcidump_Fe4S4.txt"
_DEFAULT_DETFILES = ",".join(f"{_GDB_APP}/det{i}.txt" for i in range(4))


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


def parse_args():
    """Parse command line arguments for all GDB_SBD parameters."""
    parser = argparse.ArgumentParser(
        description="GDB diagonalization over an explicit determinant list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'gpu', 'gpu-omp', 'auto'],
                       help="Compute backend. 'gpu' is NVHPC Thrust, which is the "
                            "only GPU backend with GDB kernels; 'gpu-omp' has none, "
                            "so GDB runs on the host there")

    # --- input ------------------------------------------------------------
    parser.add_argument('--skip-weight-check', action='store_true',
                       dest='skip_weight_check',
                       help='Do not verify that every determinant has the expected '
                            'alpha/beta electron count. The check costs about '
                            '0.02 s per million determinants and catches a wrong '
                            'bit order, which otherwise returns a plausible wrong '
                            'energy with no error at all')
    parser.add_argument('--fcidump', default=_DEFAULT_FCIDUMP,
                       help='FCIDUMP file defining the Hamiltonian')
    parser.add_argument('--detfiles', default=_DEFAULT_DETFILES,
                       help='Comma-separated files of full determinants, one '
                            '2*norb-bit string per line. Concatenated in Python '
                            'and passed as a single in-memory list')
    parser.add_argument('--from-alpha', '--alpha-file', default='', metavar='FILE',
                       dest='from_alpha',
                       help='Instead of --detfiles, read a norb-bit alpha list and '
                            'form the full |A|^2 product basis by interleaving it '
                            'with itself. This is the subspace TPB would build from '
                            'the same file, so the two solvers are comparable')
    parser.add_argument('--alpha-limit', type=int, default=0, dest='alpha_limit',
                       help='With --from-alpha, keep only the first N alpha strings. '
                            '0 means all, which costs |A|^2 determinants')

    # --- MPI decomposition ------------------------------------------------
    parser.add_argument('--b_comm_size', type=int, default=1,
                       help='Basis communicator size, i.e. how many shards the '
                            'determinant list is split into. The only dimension that '
                            'divides memory')
    parser.add_argument('--t_comm_size', type=int, default=1,
                       help='Task communicator size. Must not exceed b_comm_size: GDB '
                            'runs one task per basis-ring station and there are '
                            'exactly b_comm_size of them')
    parser.add_argument('--determinant_distribution', default='grid-cyclic-balanced',
                       choices=['', 'input', 'equal-bra-a', 'count', 'count-sorted',
                                'grid-cyclic', 'grid-cyclic-balanced'],
                       help="How to place determinants across b_comm. The default "
                            "spreads alpha and beta keys over a grid, then evens out "
                            "the determinant counts: balanced for irregular subspaces "
                            "such as heat-bath ones, where equal-bra-a (gdb_diag's own "
                            "default, '') can put most of them on one rank. 'input' "
                            "keeps the shards as read, the cheapest choice when the "
                            "files are already balanced")
    parser.add_argument('--determinant_grid_a', type=int, default=0,
                       help='Grid rows for the grid-cyclic schemes; with '
                            '--determinant_grid_b must multiply to b_comm_size')
    parser.add_argument('--determinant_grid_b', type=int, default=0,
                       help='Grid columns; see --determinant_grid_a')

    # --- diagonalization --------------------------------------------------
    # GDB implements Davidson only: there is no Lanczos anywhere under
    # include/sbd/chemistry/gdb/. gdb::diag is `if(method==0){} else if(method==1){}`
    # with no else, and `energy` is assigned only inside those branches
    # (gdb/sbdiag.h:418, :501), so method 2 or 3 returns an uninitialized double on
    # the CPU backend. The Thrust path is accidentally safe -- sbdiag.h:210-211 does
    # `method &= 1`. Hence choices=[0, 1], unlike run_sbd_diag.py's [0, 1, 2, 3].
    parser.add_argument('--method', type=int, default=0, choices=[0, 1],
                       help='0=Davidson, 1=Davidson storing the Hamiltonian. GDB has '
                            'no Lanczos, so TPB methods 2 and 3 do not exist here')
    parser.add_argument('--iteration', '--max_it', type=int, default=100,
                       dest='max_it', help='Maximum Davidson iterations')
    parser.add_argument('--block', '--max_nb', type=int, default=10,
                       dest='max_nb', help='Maximum number of basis vectors')
    parser.add_argument('--tolerance', '--eps', type=float, default=1e-6,
                       dest='eps', help='Convergence tolerance')
    parser.add_argument('--max_time', type=float, default=1e10,
                       help='Maximum wall time in seconds')
    parser.add_argument('--init', type=int, default=0,
                       help='Initial vector policy')
    parser.add_argument('--seed', type=int, default=1729,
                       help='Seed for a random initial vector (init != 0)')
    parser.add_argument('--bit_length', type=int, default=62,
                       help='Bits per packed word: even, at most 62. Words per '
                            'determinant is ceil(2*norb / bit_length)')

    # --- RDMs -------------------------------------------------------------
    parser.add_argument('--rdm_output', default='', metavar='FILE',
                       help='Save spin-summed rdm1 and rdm2 to one .npz file. '
                            'Implies do_rdm=1')

    # --- carryover / expansion -------------------------------------------
    parser.add_argument('--carryover_type', type=int, default=0,
                       choices=[0, 1, 2, 3],
                       help='0=off, 1=keep the top --ratio fraction by weight, '
                            '2/3=weight-truncate then heatbath-expand (2 and 3 are '
                            'the two heatbath variants). 2/3 GROW the subspace')
    parser.add_argument('--carryover_ratio', '--ratio', type=float, default=0.0,
                       dest='ratio',
                       help='Fraction of determinants kept by carryover_type 1')
    parser.add_argument('--carryover_threshold', '--threshold', type=float,
                       default=0.01, dest='threshold',
                       help='Weight threshold for carryover_type 1')
    parser.add_argument('--heatbath_cutoff', type=float, default=1e-4,
                       help='Integral-magnitude cutoff admitting a candidate '
                            'determinant (carryover_type 2/3)')
    parser.add_argument('--heatbath_truncation', type=float, default=0.0,
                       help='Weight threshold applied to parents before expanding')
    parser.add_argument('--heatbath_batch_size', type=int, default=1000000,
                       help='Heatbath expansion batch size per rank')

    # --- wavefunction I/O -------------------------------------------------
    parser.add_argument('--loadname', default='',
                       help='Load an initial wavefunction from this file')
    parser.add_argument('--savename', default='',
                       help='Save the final wavefunction under this prefix')
    parser.add_argument('--carryover_detfile', default='', metavar='PREFIX',
                       help='Write the carryover SBD returned (needs --carryover_type '
                            '1-3) as determinant strings, PREFIX000000.txt ..., one '
                            'file per rank, globally sorted: the layout --detfiles '
                            'reads, so a next run can start from it. With types 2/3 '
                            'that is the heat-bath-expanded next subspace.')

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
    return args


def read_det_strings(paths):
    """Read full-determinant bitstrings from one or more text files.

    Returns them in file order, unfiltered: gdb_diag sorts its own input and
    rejects duplicates, so both properties are left for it to enforce rather
    than silently repaired here.
    """
    strings = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    strings.append(line)
    return strings


def interleave_spin_strings(alpha, beta):
    """Interleave an alpha and a beta bitstring into one GDB determinant string.

    Both inputs are norb-bit strings written most-significant-first, as in
    SBD's alpha determinant files. The result is 2*norb bits in the order the
    GDB app documents: reading from the right, alpha orbital 1, beta orbital 1,
    alpha orbital 2, and so on -- i.e. bit 2*i is alpha orbital i and bit
    2*i + 1 is beta orbital i, which is what from_string() then packs.
    """
    a_rev = alpha[::-1]
    b_rev = beta[::-1]
    return ''.join(a_rev[i] + b_rev[i] for i in range(len(a_rev)))[::-1]


def write_carryover(args, carryover, total_bits, rank, size):
    """Write the carryover as globally sorted determinant strings, one file per rank.

    Collective. SBD leaves the carryover spread over the ranks, but not always in a
    layout --detfiles can read back block by block: types 2/3 come back globally
    sorted over the world (HeatbathExpansion ends with a global sort), while type 1
    keeps each rank's own selection, in the order the input placement put it, and
    ranks that share a b_comm position (t_comm or the helper dimension above 1) hold
    copies of one shard. So only one copy of each shard takes part -- world ranks
    ``0 .. b_comm_size - 1``, as for type 1, or every rank for 2/3 -- and a
    distributed sort puts block r on rank r. Nothing is gathered whole: each rank
    sorts, exchanges and writes only its own part.
    """
    import os
    from mpi4py import MPI
    import sbd
    from sbd import analysis

    comm = MPI.COMM_WORLD
    start = time.perf_counter()
    det = np.asarray(carryover, dtype=np.uint64)
    distinct = args.carryover_type in (2, 3) or rank < args.b_comm_size
    if not distinct or det.size == 0:
        det = np.zeros((0, max(det.shape[1] if det.ndim == 2 else 0,
                               -(-total_bits // args.bit_length))), dtype=np.uint64)
    det, _ = analysis._distributed_sort(det, np.zeros(len(det)), comm)
    # A repeat after a global sort is adjacent on one rank or across neighbours.
    local = bool(len(det) > 1 and np.any(np.all(det[1:] == det[:-1], axis=1)))
    edges = comm.allgather((det[0].copy(), det[-1].copy()) if len(det) else None)
    previous = next((e[1] for e in reversed(edges[:rank]) if e is not None), None)
    across = bool(len(det) and previous is not None and np.array_equal(previous, det[0]))
    if comm.allreduce(int(local or across)):
        raise RuntimeError("the carryover holds a determinant twice; not writing it")
    counts = comm.allgather(len(det))
    # Number the non-empty blocks consecutively, so the files have no gaps.
    index = sum(1 for c in counts[:rank] if c)
    path = f"{args.carryover_detfile}{index:06d}.txt"
    if len(det):
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="ascii") as handle:
            handle.write("\n".join(sbd.to_strings(det, args.bit_length, total_bits)))
            handle.write("\n")
        os.replace(tmp, path)
    elapsed = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
    files = [f"{args.carryover_detfile}{i:06d}.txt" for i in range(sum(1 for c in counts if c))]
    if rank == 0:
        # A larger earlier run under the same prefix must not leave extra files.
        stale = len(files)
        while os.path.exists(f"{args.carryover_detfile}{stale:06d}.txt"):
            os.remove(f"{args.carryover_detfile}{stale:06d}.txt")
            stale += 1
    return (f"Carryover written: {sum(counts)} determinants in {len(files)} file(s), "
            f"globally sorted, in {elapsed:.2f} s (max over ranks)\n"
            f"  next run: --detfiles {','.join(files)}")


def shard_bounds(total, b_comm_size, index):
    """This rank's slice of a globally sorted list of ``total`` determinants.

    Mirrors SBD's own ``q = N/p``, remainder-to-the-low-ranks split
    (``balanced_begin``, framework/bit_manipulation.h:1523-1531), so an ``input``
    placement is already the count-balanced one.
    """
    quotient, remainder = divmod(total, b_comm_size)
    begin = index * quotient + min(index, remainder)
    return begin, begin + quotient + (1 if index < remainder else 0)


def product_basis(alpha_strings):
    """Every (alpha, beta) pair from one list: the subspace TPB spans from it."""
    return [interleave_spin_strings(a, b)
            for a in alpha_strings for b in alpha_strings]


def main():
    args = parse_args()

    import sbd
    from sbd.sbd_solver import assemble_rdms

    sbd.init(device=args.device)

    rank = sbd.get_rank()
    size = sbd.get_world_size()
    device = sbd.get_device()

    # SBD writes the state without checking that its file opened, so a missing
    # directory would lose the save while the run reports success. Checked by
    # every rank: each one writes its own file there.
    import os
    for flag, path in (("--savename", args.savename),
                       ("--carryover_detfile", args.carryover_detfile)):
        if path and not os.path.isdir(os.path.dirname(os.path.abspath(path))):
            print(f"[rank {rank}] ERROR: {flag} {path!r}: directory "
                  f"{os.path.dirname(os.path.abspath(path))!r} does not exist; "
                  f"create it first", file=sys.stderr, flush=True)
            return 1
    if args.carryover_detfile and not args.carryover_type:
        print(f"[rank {rank}] ERROR: --carryover_detfile needs --carryover_type 1, 2 "
              f"or 3: with 0, SBD selects no carryover", file=sys.stderr, flush=True)
        return 1
    if args.loadname:
        # SBD reads file b of a saved state onto b_comm position b, then passes what
        # does not belong there around the ring; files past b_comm_size are never
        # read. Davidson would then start from part of the state -- the right answer,
        # slower, and without a word -- so say so instead.
        count = 0
        while os.path.exists(f"{args.loadname}{count:06d}.bin"):
            count += 1
        if count == 0:
            print(f"[rank {rank}] ERROR: --loadname {args.loadname!r}: no saved state "
                  f"({args.loadname}000000.bin ...)", file=sys.stderr, flush=True)
            return 1
        if count > args.b_comm_size:
            print(f"[rank {rank}] ERROR: --loadname {args.loadname!r} has {count} files "
                  f"but --b_comm_size is {args.b_comm_size}: SBD loads a saved state "
                  f"only onto at least as many shards as it has files, and would start "
                  f"from part of it. Use --b_comm_size {count} or more, or leave out "
                  f"--loadname.", file=sys.stderr, flush=True)
            return 1

    if rank == 0:
        print("=" * 70)
        print("SBD GDB - general determinant basis, in memory")
        print("=" * 70)
        sbd.print_info()
        print()

    # Every rank builds the same list: that is what the in-memory entry point
    # expects, and it is why b_comm_size must be 1.
    fcidump = sbd.LoadFCIDump(args.fcidump)
    norb = int(fcidump.header["NORB"])
    nelec = int(fcidump.header["NELEC"])
    ms2 = int(fcidump.header.get("MS2", 0))
    total_bits = 2 * norb

    b_size = args.b_comm_size
    shard_index = rank % b_size if b_size > 1 else 0

    if args.from_alpha:
        alpha = read_det_strings([args.from_alpha])
        if args.alpha_limit:
            alpha = alpha[:args.alpha_limit]
        bad = [s for s in alpha if len(s) != norb]
        if bad:
            print(f"ERROR: --from-alpha strings must be {norb} bits "
                  f"(NORB from the FCIDUMP); found one of length {len(bad[0])}",
                  file=sys.stderr)
            return 1
        det_strings = product_basis(alpha)
        source = (f"{args.from_alpha} ({len(alpha)} alpha strings -> "
                  f"{len(alpha)}^2 product determinants)")
        whole = sbd.sort_bitarray_array(
            sbd.from_strings(det_strings, args.bit_length, total_bits))
        begin, end = shard_bounds(whole.shape[0], b_size, shard_index)
        det = whole[begin:end] if b_size > 1 else whole
        placement = (f"sliced from the globally sorted list "
                     f"[{begin}:{end}]") if b_size > 1 else "whole basis"
    else:
        paths = [p for p in args.detfiles.split(',') if p]
        if not paths:
            print("ERROR: no determinant files given", file=sys.stderr)
            return 1
        # When the files already partition the basis -- each sorted, disjoint, and
        # in globally sorted order, as gdet emits and Fe4S4's four files are --
        # a rank can read only its own share and nothing is read twice.
        #
        # The share must be a CONTIGUOUS BLOCK of files, not a stride: shard i has
        # to hold a strictly lower range than shard i+1, so with 24 files over 8
        # ranks it is files 0-2, 3-5, ... and NOT 0,8,16. A strided assignment
        # would interleave the ranges and fail the sorted-disjoint check.
        took_file_block = b_size > 1 and len(paths) % b_size == 0
        if took_file_block:
            per_rank = len(paths) // b_size
            first = shard_index * per_rank
            mine = paths[first:first + per_rank]
            placement = (f"file {first}" if per_rank == 1 else
                         f"files {first}-{first + per_rank - 1}")
            placement += f" of {len(paths)}"
        elif b_size > 1:
            mine = paths
            placement = "sliced from the globally sorted concatenation"
        else:
            mine = paths
            placement = "whole basis"
        det_strings = read_det_strings(mine)
        bad = [s for s in det_strings if len(s) != total_bits]
        if bad:
            print(f"ERROR: determinant strings must be {total_bits} bits "
                  f"(2 * NORB); found one of length {len(bad[0])}", file=sys.stderr)
            return 1
        if not det_strings:
            print("ERROR: no determinants read", file=sys.stderr)
            return 1
        det = sbd.sort_bitarray_array(
            sbd.from_strings(det_strings, args.bit_length, total_bits))
        if b_size > 1 and not took_file_block:
            begin, end = shard_bounds(det.shape[0], b_size, shard_index)
            det = det[begin:end]
            placement += f" [{begin}:{end}]"
        source = f"{len(paths)} file(s): {', '.join(paths)}"

    if det.shape[0] == 0:
        print("ERROR: no determinants read", file=sys.stderr)
        return 1
    if not args.skip_weight_check:
        try:
            check_spin_weights(det, nelec, ms2, args.bit_length)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    words = det.shape[1]

    config = sbd.GDB_SBD()
    config.t_comm_size = args.t_comm_size
    config.b_comm_size = args.b_comm_size
    config.method = args.method
    config.max_it = args.max_it
    config.max_nb = args.max_nb
    config.eps = args.eps
    config.max_time = args.max_time
    config.init = args.init
    config.seed = args.seed
    config.do_rdm = 1 if args.rdm_output else 0
    config.bit_length = args.bit_length
    config.carryover_type = args.carryover_type
    config.ratio = args.ratio
    config.threshold = args.threshold
    config.heatbath_cutoff = args.heatbath_cutoff
    config.heatbath_truncation = args.heatbath_truncation
    config.heatbath_batch_size = args.heatbath_batch_size

    if rank == 0:
        grid = args.t_comm_size * args.b_comm_size
        helper = size // grid if grid and size % grid == 0 else 0
        print("Configuration:")
        print(f"  Device: {device}")
        print(f"  Method: {'Davidson' if args.method == 0 else 'Davidson + stored H'}")
        print(f"  Max iterations: {config.max_it}")
        print(f"  Tolerance: {config.eps}")
        print(f"  MPI ranks: {size}")
        print(f"  Decomposition: t_comm_size={args.t_comm_size} "
              f"b_comm_size={args.b_comm_size} helper={helper}")
        print(f"  Placement: {args.determinant_distribution or 'equal-bra-a (default)'}")
        if args.b_comm_size == 1 and size > 1:
            print("  NOTE: b_comm_size=1 means every rank holds the whole basis and "
                  "the whole excitation lookup. Only b_comm_size shards memory -- the "
                  "helper dimension divides work but not storage.")
        if device == 'gpu' and helper != 1:
            print("  ERROR: GDB on the Thrust backend requires helper == 1, i.e. "
                  "t_comm_size * b_comm_size == ranks. Give every rank to "
                  "--b_comm_size (with --t_comm_size <= it).")
        if device == 'gpu-omp':
            print("  WARNING: the OMP-offload backend has no GDB kernels -- there is "
                  "no `omp target` code under include/sbd/chemistry/gdb/ at all. "
                  "This run pins a device and then diagonalizes on the host. Use "
                  "--device gpu (Thrust, NVIDIA only) for GDB on a GPU.")
        print()
        print("Subspace:")
        print(f"  Source: {source}")
        print(f"  This rank: {placement}")
        print(f"  Determinants on this rank: {det.shape[0]}")
        print(f"  Orbitals: {norb} ({total_bits} spin orbitals), electrons: {nelec}")
        print(f"  Packing: {words} word(s) of {args.bit_length} bits")
        print()
        print("Running GDB diagonalization...")
        print()

    start = time.perf_counter()
    results = sbd.gdb_diag(
        fcidump, det, config,
        loadname=args.loadname, savename=args.savename,
        determinant_distribution=args.determinant_distribution,
        determinant_grid_a=args.determinant_grid_a,
        determinant_grid_b=args.determinant_grid_b,
    )
    elapsed = time.perf_counter() - start

    # Collective, so every rank must reach it: before the rank-0-only report below,
    # which the other ranks skip by returning.
    carryover_report = None
    if args.carryover_detfile:
        carryover_report = write_carryover(args, results['carryover_det'], total_bits,
                                           rank, size)

    if rank != 0:
        return 0

    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Device: {device.upper()}")
    print(f"Ground state energy: {results['energy']:.10f} Hartree")
    print(f"Subspace dimension: {results['global_dim']} "
          f"({results['local_dim']} on this rank)")
    print(f"Placement applied: {results['determinant_distribution']}")
    print(f"Wall time: {elapsed:.2f} s")

    density = results['density']
    combined = [density[2 * i] + density[2 * i + 1] for i in range(len(density) // 2)]
    print(f"Density: {np.round(combined, 6).tolist()}")
    print(f"  (sums to {sum(combined):.6f}; should equal the electron count {nelec})")
    if results["global_dim"] > 1 and looks_like_a_single_determinant(density):
        print("  WARNING: every occupancy is 0 or 1, so the wavefunction is a single "
              "determinant")
        print("  and Davidson did not iterate (look for 'tol=0' above). The energy is "
              "a diagonal")
        print("  element -- a valid upper bound, not the subspace's ground state. "
              "Expanding the")
        print("  subspace once (--carryover_type 2) gives Davidson something to "
              "couple.")

    carryover = results['carryover_det']
    if args.carryover_type:
        print(f"Carryover determinants on this rank: {carryover.shape[0]}")
        if size > 1:
            print("  NOTE: the carryover list is this rank's shard, not the whole "
                  "list -- "
                  "HeatbathExpansion ends with sort_global_bitarray + "
                  "redistribution_bitarray over the WORLD communicator "
                  "(gdb/expansion.h), so each rank holds a different slice and a "
                  "driver that feeds it back must allgather first.")
        if args.carryover_type in (2, 3):
            print("  The expanded list includes its parents: "
                  "local_heatbath_expansion starts from `edet = det` "
                  "(gdb/expansion.h:543), so this is the next subspace, not just "
                  "the new candidates.")

    if carryover_report:
        print(carryover_report)

    if args.rdm_output:
        # assemble_rdms' layout was verified against PySCF on the TPB path; the
        # keys and flat-index convention are shared with GDB, but that
        # verification has not been repeated here.
        rdm1, rdm2 = assemble_rdms(results, norb)
        if rdm1 is None:
            print("No RDMs returned (do_rdm was 0)")
        else:
            np.savez(args.rdm_output, rdm1=rdm1, rdm2=rdm2)
            print(f"\nRDMs saved to {args.rdm_output}")
            print(f"1-RDM trace: {np.trace(rdm1):.6f} "
                  f"(should equal the electron count {nelec})")

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
