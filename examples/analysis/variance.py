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
Energy variance and Epstein-Nesbet PT2 of a wavefunction saved by an SBD run.

Run a diagonalization with --savename, then point this at the same prefix and
FCIDUMP. It works for TPB (run_sbd_diag.py) and GDB (run_gdb_diag.py) states alike,
and on any number of ranks: it need not match the run that saved the state.

    # 1. diagonalize, saving the state
    mpirun -np 4 python ../gdb/run_gdb_diag.py --b_comm_size 4 --savename state-

    # 2. its variance and PT2 (stochastic, on CPU; many ranks, few threads is fastest)
    mpirun -np 32 -x OMP_NUM_THREADS=2 python variance.py \\
        --fcidump ../../vendor/sbd-upstream/apps/chemistry_gdb_selected_basis_diagonalization/fcidump_Fe4S4.txt \\
        --state state- --bit_length 62

    # small systems: exact, with PySCF, in one process
    python variance.py --fcidump fcidump.txt --state state- --bit_length 20 --method exact

See README.md in this directory for what the numbers mean and how to choose the
settings.
"""

import argparse
import json
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fcidump', required=True,
                        help='The FCIDUMP the state was diagonalized with')
    parser.add_argument('--state', required=True, metavar='PREFIX',
                        help="The --savename prefix of the run (its files are PREFIX000000, "
                             "... for TPB and PREFIX000000.bin, ... for GDB), or one GDB "
                             ".bin file")
    parser.add_argument('--bit_length', type=int, required=True,
                        help="The run's --bit_length. The files do not record it")
    parser.add_argument('--method', default='auto', choices=['auto', 'exact', 'stochastic'],
                        help="exact: PySCF over the full CI space, one process, small systems "
                             "only. stochastic: SBD's sampled estimate, all ranks. auto "
                             "(default): exact when the full space has at most "
                             "--exact_limit determinants")
    parser.add_argument('--exact_limit', type=int, default=5_000_000,
                        help='Largest full CI space auto treats exactly (default 5,000,000)')
    parser.add_argument('--energy', type=float, default=None,
                        help="The state's energy, the PT2 reference. Default: computed from "
                             "the state. If you give it, it must be this state's energy")
    parser.add_argument('--samples', type=int, default=20000,
                        help='Stochastic: parent draws per batch (default 20000)')
    parser.add_argument('--batches', type=int, default=10,
                        help='Stochastic: batches; the error bar is their standard error '
                             '(default 10)')
    parser.add_argument('--seed', type=int, default=1729, help='Stochastic: sampling seed')
    parser.add_argument('--cutoff', type=float, default=0.0,
                        help="Stochastic: heat-bath screening. 0 (default) is unbiased. A "
                             "cutoff is faster but biases both results low, more so the "
                             "better the state; compare against a smaller one before "
                             "trusting it")
    parser.add_argument('--observable', default='both', choices=['both', 'variance', 'pt2'],
                        help='Stochastic: estimate the variance, PT2, or both (default)')
    parser.add_argument('--first_batch', type=int, default=0,
                        help='Stochastic: ID of the first batch (default 0). Another run with '
                             'the same seed and a different first batch gives independent '
                             'batches that can be pooled with this run\'s')
    parser.add_argument('--minimum_abs_denominator', type=float, default=0.0,
                        help='Stochastic: stop with an error if any PT2 denominator '
                             '|E - H_aa| is at or below this (default 0: any finite nonzero '
                             'one is accepted)')
    parser.add_argument('--json', default='', metavar='FILE',
                        help='Also write the result, settings and per-batch values as JSON')
    return parser.parse_args()


def main():
    args = parse_args()
    import sbd
    from sbd import analysis

    sbd.init(device='cpu')
    rank, size = sbd.get_rank(), sbd.get_world_size()
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    fcidump = sbd.LoadFCIDump(args.fcidump)
    norb = int(fcidump.header['NORB'])
    start = time.perf_counter()
    try:
        state = analysis.load_state(args.state, norb, args.bit_length)
    except (FileNotFoundError, ValueError) as exc:
        say(f"ERROR: {exc}")
        say("The files under one --state prefix must all come from one run; see README.md.")
        return 1
    say("=" * 70)
    say(f"State:   {args.state}  ({state.kind.upper()}, {len(state.files)} file(s), checked as one run)")
    for path in state.files:
        say(f"           {path}  ({os.path.getsize(path):,} bytes)")
    say(f"FCIDUMP: {args.fcidump}  (NORB={norb})")
    say(f"Ranks:   {size}")

    energy = args.energy
    if energy is None:
        energy = analysis.state_energy(state, fcidump)
        say(f"Energy:  {energy:.10f}  (computed from the state)")
    else:
        say(f"Energy:  {energy:.10f}  (given)")

    try:
        result = analysis.variance(state, fcidump, energy, method=args.method,
                                   samples=args.samples, batches=args.batches, seed=args.seed,
                                   cutoff=args.cutoff, exact_limit=args.exact_limit,
                                   observable=args.observable, first_batch=args.first_batch,
                                   minimum_abs_denominator=args.minimum_abs_denominator)
    except (ImportError, ValueError, RuntimeError) as exc:
        say(f"ERROR: {exc}")
        return 1
    elapsed = time.perf_counter() - start

    stochastic = result.method == 'stochastic'
    pm = (lambda err: f"  +- {err:.2e}") if stochastic else (lambda err: '')
    say("-" * 70)
    say(f"Method:        {result.method}"
        + (f"  ({args.batches} batches x {args.samples} samples, cutoff {args.cutoff:g}, "
           f"seed {args.seed}"
           + (f", batches {args.first_batch}-{args.first_batch + args.batches - 1}"
              if args.first_batch else "") + ")"
           if stochastic else "  (PySCF, full CI space)"))
    say(f"Dimension:     {result.dimension:,}")
    if result.variance == result.variance:            # NaN when not asked for
        say(f"Variance:      {result.variance:.6e}{pm(result.variance_error)}  Ha^2")
    if result.pt2 == result.pt2:
        say(f"PT2:           {result.pt2:.6e}{pm(result.pt2_error)}  Ha")
    say(f"E_var:         {result.energy:.10f}  Ha")
    if result.pt2 == result.pt2:
        say(f"E_var + PT2:   {result.energy_pt2:.10f}{pm(result.pt2_error)}  Ha")
    say(f"Elapsed:       {elapsed:.1f} s")
    say("=" * 70)
    if args.json and rank == 0:
        with open(args.json, 'w') as handle:
            json.dump({'state': args.state, 'fcidump': args.fcidump, 'ranks': size,
                       'variance': result.variance, 'variance_error': result.variance_error,
                       'pt2': result.pt2, 'pt2_error': result.pt2_error,
                       'energy': result.energy, 'energy_pt2': result.energy_pt2,
                       'dimension': result.dimension, 'method': result.method,
                       'settings': result.settings, 'batches': result.batches,
                       'elapsed_seconds': elapsed}, handle, indent=2)
        say(f"Wrote {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
