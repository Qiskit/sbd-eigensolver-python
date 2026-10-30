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

"""Ground state of a creation/annihilation-operator Hamiltonian over a selected basis.

The Python counterpart of upstream SBD's ``caop_selected_basis_diagonalization`` app,
with its option names. With no arguments it runs upstream's own example: an 8-site
extended hard-core boson chain over a 16-state basis::

    mpirun -np 4 python run_caop_diag.py --b_comm_size 4 --do_redist_basis 1

The Hamiltonian file's first non-comment line selects the statistics (positive:
hard-core bosons or spins; zero or negative: fermions); each further line is a term,
a coefficient then operator-mode pairs, e.g. ``-1.0 cdag 0 c 1``. The basis files hold
one bitstring of ``--system_size`` characters per line, rightmost = mode 0.
"""

import argparse
import os
import sys

_APP = "../../vendor/sbd-upstream/apps/caop_selected_basis_diagonalization"


def parse_args():
    parser = argparse.ArgumentParser(
        description="CAOP selected-basis diagonalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                        help="'cpu', or 'gpu' (NVHPC Thrust, NVIDIA GPUs only). "
                             "Upstream's CAOP code has no OpenMP-offload path, so there "
                             "is no 'gpu-omp' and no CAOP GPU path for AMD GPUs")
    parser.add_argument('--hamfile', default=f"{_APP}/hamiltonian.txt",
                        help='Hamiltonian file')
    parser.add_argument('--basisfiles', default=f"{_APP}/basis.txt",
                        help='Comma-separated basis files')
    parser.add_argument('--system_size', type=int, default=8,
                        help='Number of modes, the length of each basis bitstring')
    parser.add_argument('--bit_length', type=int, default=20,
                        help='Bits stored per 64-bit word of a bitstring')
    parser.add_argument('--complex', action='store_true',
                        help='Complex amplitudes, for a Hamiltonian with complex '
                             "coefficients (upstream's -D_COMPLEX build)")
    parser.add_argument('--t_comm_size', type=int, default=1,
                        help='Task communicator size')
    parser.add_argument('--b_comm_size', type=int, default=1,
                        help='Basis communicator size: shards of the basis')
    parser.add_argument('--method', type=int, default=1, choices=[0, 1, 2, 3],
                        help='0/1 Davidson, Hamiltonian on the fly/stored; '
                             '2/3 Lanczos, on the fly/stored')
    parser.add_argument('--iteration', type=int, default=10, help='Restart cycles')
    parser.add_argument('--block', type=int, default=10,
                        help='Largest reduced-space dimension before a restart')
    parser.add_argument('--numivec', type=int, default=1,
                        help='Initial vectors for Davidson')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Convergence threshold on the residual (upstream\'s default)')
    parser.add_argument('--initial_determinant_bitstring', default='',
                        help='Start with unit weight on this basis bitstring')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the random initial vector; 0 (upstream\'s default) '
                             'picks a random one, which SBD prints')
    parser.add_argument('--do_sort_basis', type=int, default=0, choices=[0, 1],
                        help='Sort the basis across ranks (for overlapping files)')
    parser.add_argument('--do_redist_basis', type=int, default=1, choices=[0, 1],
                        help='Redistribute the basis evenly over --b_comm_size')
    parser.add_argument('--carryover_type', type=int, default=0, choices=[0, 1, 2],
                        help='0 none, 1 keep the largest-weight fraction, '
                             '2 deterministic heat-bath expansion')
    parser.add_argument('--carryover_ratio', type=float, default=0.0,
                        help='carryover_type 1: fraction kept')
    parser.add_argument('--heatbath_cutoff', type=float, default=1e-4,
                        help='carryover_type 2: term acceptance cutoff')
    parser.add_argument('--heatbath_truncation', type=float, default=0.0,
                        help='carryover_type 2: drop a parent unless |c_j|^2 exceeds this')
    parser.add_argument('--heatbath_batch_size', type=int, default=1000000,
                        help='carryover_type 2: candidate batch size per rank before '
                             'merging')
    parser.add_argument('--heatbath_parent_distribution', type=int, default=1,
                        choices=[0, 1],
                        help='carryover_type 2: 0 keeps the basis layout for the '
                             'expansion, 1 deals parents out by weight')
    parser.add_argument('--loadname', default='',
                        help='Load an initial wavefunction from this prefix')
    parser.add_argument('--savename', default='',
                        help='Save the final wavefunction under this prefix')
    args = parser.parse_args()
    if args.loadname and args.initial_determinant_bitstring:
        parser.error("--loadname and --initial_determinant_bitstring both set the "
                     "starting vector, and --loadname would win silently: give one")
    return args


def _print_gpu_guidance(method):
    """What the GPU does for this method, so a user can judge whether to use it.

    Upstream's CAOP code has a GPU (Thrust) kernel only for applying H on the fly,
    methods 0 and 2. The stored-Hamiltonian methods 1 and 3 build and apply H on the
    CPU in every build, so on 'gpu' they run exactly as on 'cpu'.
    """
    if method in (1, 3):
        print(f"NOTE: --method {method} stores the Hamiltonian, and upstream SBD builds and "
              f"applies it on the CPU only: on --device gpu this run uses no GPU and "
              f"runs as it would on --device cpu. For the GPU kernel, use --method "
              f"{method - 1} (the Hamiltonian applied on the fly).")
    else:
        print(f"NOTE: --method {method} applies the Hamiltonian on the GPU. The rest of "
              f"each Davidson/Lanczos step (vector updates, orthogonalization, the "
              f"reduced-space solve) stays on the CPU, and the vector is copied to and "
              f"from the GPU on every application. For small bases that overhead can "
              f"outweigh the GPU kernel: compare against --device cpu, including "
              f"--method {method + 1}, which stores the Hamiltonian.")


def main():
    args = parse_args()
    import sbd

    sbd.init(device=args.device)
    rank = sbd.get_rank()
    if args.savename:
        directory = os.path.dirname(os.path.abspath(args.savename))
        if not os.path.isdir(directory):
            print(f"[rank {rank}] ERROR: --savename {args.savename!r}: directory "
                  f"{directory!r} does not exist; create it first", file=sys.stderr)
            return 1

    config = sbd.CAOP_SBD()
    config.system_size, config.bit_length = args.system_size, args.bit_length
    config.t_comm_size, config.b_comm_size = args.t_comm_size, args.b_comm_size
    config.method, config.max_it, config.max_nb = args.method, args.iteration, args.block
    config.max_iv, config.eps = args.numivec, args.tolerance
    config.initial_determinant_bitstring = args.initial_determinant_bitstring
    config.seed = args.seed
    config.do_sort_basis = bool(args.do_sort_basis)
    config.do_redist_basis = bool(args.do_redist_basis)
    config.carryover_type, config.ratio = args.carryover_type, args.carryover_ratio
    config.heatbath_cutoff = args.heatbath_cutoff
    config.heatbath_truncation = args.heatbath_truncation
    config.heatbath_batch_size = args.heatbath_batch_size
    config.heatbath_parent_distribution = args.heatbath_parent_distribution

    if rank == 0 and args.device == 'gpu':
        _print_gpu_guidance(args.method)

    results = sbd.caop_diag_from_files(
        args.hamfile, [f for f in args.basisfiles.split(',') if f], config,
        loadname=args.loadname, savename=args.savename, complex=args.complex)
    if rank == 0:
        print("=" * 70)
        print(f"Device: {args.device.upper()}"
              + ("  (complex amplitudes)" if args.complex else ""))
        print(f"Ground state energy: {results['energy']:.10f}")
        if args.carryover_type:
            print(f"Carryover basis on this rank: {len(results['carryover_basis'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
