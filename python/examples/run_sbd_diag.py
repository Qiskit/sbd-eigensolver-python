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
Standalone SBD diagonalization (no SQD loop, no qiskit dependency).

Runs a single TPB diagonalization from an FCIDUMP file and alpha determinant
file using the SBD library directly.

Usage:
    # CPU backend
    mpirun -np 8 -x OMP_NUM_THREADS=4 python run_sbd_diag.py --device cpu

    # GPU backend
    mpirun -np 8 python run_sbd_diag.py --device gpu

    # N2 molecule
    mpirun -np 8 python run_sbd_diag.py \
        --fcidump ../../vendor/sbd-upstream/data/n2/fcidump.txt \
        --adetfile ../../vendor/sbd-upstream/data/n2/1em3-alpha.txt

    # H2O molecule
    mpirun -np 8 python run_sbd_diag.py \
        --fcidump ../../vendor/sbd-upstream/data/h2o/fcidump.txt \
        --adetfile ../../vendor/sbd-upstream/data/h2o/h2o-1em3-alpha.txt

    # Retrieve the 1-/2-particle RDMs and save them to a file
    mpirun -np 8 python run_sbd_diag.py --rdm 1 --rdm_output rdms.npz
    # Prints trace(rdm1) (should equal the electron count) and the natural
    # orbital occupations (eigenvalues of rdm1) -- occupations near 2 or 0
    # indicate a single-reference-like orbital, occupations near 1 (or
    # several clustered together) flag multi-reference character / a
    # candidate active space.

    # Distinct alpha and beta determinant files (default is beta = alpha)
    mpirun -np 8 python run_sbd_diag.py --symmetrize_spin 0 \
        --adetfile alpha-dets.txt --bdetfile beta-dets.txt
"""

import argparse
import sys

import numpy as np

def parse_args():
    """Parse command line arguments for all TPB_SBD parameters"""
    parser = argparse.ArgumentParser(
        description='Quantum chemistry calculation with CPU/GPU support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Device selection
    parser.add_argument('--device',
                       choices=['auto', 'cpu',
                                'gpu', 'gpu-thrust', 'gpu-nvidia', 'cuda',
                                'gpu-omp', 'gpu-omp-offload',
                                'gpu-nvhpc-omp', 'gpu-nvidia-omp'],
                       default='cpu',
                       help='Device: cpu | gpu (NVHPC Thrust) | '
                            'gpu-omp (nvc++ OpenMP target offload) | auto. '
                            'gpu-nvhpc-omp / gpu-nvidia-omp are deprecated '
                            'aliases for gpu-omp.')
    
    # Input files
    parser.add_argument('--fcidump', default='../../vendor/sbd-upstream/data/h2o/fcidump.txt',
                       help='Path to FCIDUMP file')
    parser.add_argument('--adetfile', default='../../vendor/sbd-upstream/data/h2o/h2o-1em3-alpha.txt',
                       help='Path to alpha determinants file')
    parser.add_argument('--bdetfile', default='',
                       help='Path to beta determinants file, used only when '
                            '--symmetrize_spin 0 (otherwise ignored with a '
                            'warning: symmetric mode always derives beta from '
                            '--adetfile). Defaults to --adetfile itself when '
                            'left unset.')
    parser.add_argument('--loadname', default='',
                       help='Load initial wavefunction from file')
    parser.add_argument('--savename', default='',
                       help='Save final wavefunction to file')
    
    # MPI communicator sizes
    parser.add_argument('--adet_comm_size', type=int, default=1,
                       help='Alpha determinant communicator size')
    parser.add_argument('--bdet_comm_size', type=int, default=1,
                       help='Beta determinant communicator size')
    parser.add_argument('--task_comm_size', type=int, default=1,
                       help='Helper communicator size')
    
    # Diagonalization method and convergence
    parser.add_argument('--method', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Diagonalization method: 0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham')
    parser.add_argument('--iteration', '--max_it', type=int, default=100, dest='max_it',
                       help='Maximum number of iterations')
    parser.add_argument('--block', '--max_nb', type=int, default=10, dest='max_nb',
                       help='Maximum number of basis vectors')
    parser.add_argument('--tolerance', '--eps', type=float, default=1e-3, dest='eps',
                       help='Convergence tolerance')
    parser.add_argument('--max_time', type=float, default=1e10,
                       help='Maximum time in seconds')
    
    # Initialization and options
    parser.add_argument('--init', type=int, default=0,
                       help='Initialization method')
    parser.add_argument('--shuffle', '--do_shuffle', type=int, default=0, dest='do_shuffle',
                       help='Shuffle determinants loaded from --adetfile before '
                            'mirroring/deriving beta from them (0=no, 1-4=yes, '
                            'different shuffle seeds -- see sbdiag.h). Only '
                            'takes effect when --symmetrize_spin 1 (default): '
                            'that is the code path that derives beta from a '
                            'single loaded list at all.')
    parser.add_argument('--symmetrize_spin', type=int, default=1, choices=[0, 1],
                       help='1 (default): beta determinants are derived from '
                            '--adetfile alone (identical to it, or a shuffled '
                            'copy if --shuffle is set) -- --bdetfile is ignored '
                            'with a warning if given. 0: load --adetfile and '
                            '--bdetfile as independent, genuinely distinct '
                            'alpha/beta determinant sets (--shuffle has no '
                            'effect in this mode).')
    parser.add_argument('--rdm', '--do_rdm', type=int, default=0, choices=[0, 1], dest='do_rdm',
                       help='Calculate RDM (0=density only, 1=full RDM)')
    parser.add_argument('--rdm_output', default='',
                       help='When set (and --rdm 1), save rdm1/rdm2 to this '
                            'path as a numpy .npz file (keys: rdm1, rdm2).')
    
    # Carryover determinant selection
    parser.add_argument('--carryover_type', type=int, default=0,
                       help='Carryover determinant selection type')
    parser.add_argument('--carryover_ratio', '--ratio', type=float, default=0.0, dest='ratio',
                       help='Carryover ratio')
    parser.add_argument('--carryover_threshold', '--threshold', type=float, default=0.0, dest='threshold',
                       help='Carryover threshold')
    
    # Determinant representation
    parser.add_argument('--bit_length', type=int, default=20,
                       help='Bit length for determinant representation')
    
    # Output options
    parser.add_argument('--dump_matrix_form_wf', default='',
                       help='Filename to dump wavefunction in matrix form')

    # GPU-specific options (only used with GPU backend)
    parser.add_argument('--use_precalculated_dets', type=int, default=1, choices=[0, 1],
                       help='Use precalculated determinants (GPU only)')
    parser.add_argument('--max_memory_gb_for_determinants', '--gpu-memory', type=int, default=-1,
                       dest='max_memory_gb_for_determinants',
                       help='Maximum GPU memory in GB for determinants (-1=auto, GPU only)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Import sbd — auto-initializes on first use, but we call init()
    # explicitly here to set the default device from --device flag.
    import sbd
    # Reuses the same rdm1/rdm2 assembly sbd_solver.solve_sci uses for the
    # SQD drivers -- verified against PySCF's make_rdm1/make_rdm2 on all
    # three SBD backends, both element-wise and via the energy identity
    # E = einsum("pr,pr->",rdm1,hcore) + 0.5*einsum("prqs,prqs->",rdm2,eri).
    # mpi4py is already a real dependency of `sbd` itself (used internally
    # for sbd.init()'s communicator), so importing sbd_solver here adds no
    # new hard dependency; its pyscf/qiskit-addon-sqd imports are both
    # soft (try/except), unused by assemble_rdms itself.
    from sbd.sbd_solver import assemble_rdms

    sbd.init(device=args.device)

    rank = sbd.get_rank()
    size = sbd.get_world_size()
    
    if rank == 0:
        print("="*70)
        print("SBD Simplified API - Chemistry Calculation")
        print("="*70)
        sbd.print_info()
        print()
    
    # Configure calculation
    config = sbd.TPB_SBD()
    config.max_it = args.max_it
    config.eps = args.eps
    config.method = args.method
    config.max_nb = args.max_nb
    config.max_time = args.max_time
    config.init = args.init
    config.do_shuffle = args.do_shuffle
    config.do_rdm = args.do_rdm
    config.bit_length = args.bit_length
    config.carryover_type = args.carryover_type
    config.ratio = args.ratio
    config.threshold = args.threshold
    config.adet_comm_size = args.adet_comm_size
    config.bdet_comm_size = args.bdet_comm_size
    config.task_comm_size = args.task_comm_size
    # Thrust-only fields -- absent from the CPU/OMP-offload backends' TPB_SBD,
    # so guard with hasattr rather than assume, matching sbd_solver.py's own
    # _create_sbd_config pattern for exactly this reason.
    if hasattr(config, 'use_precalculated_dets'):
        config.use_precalculated_dets = bool(args.use_precalculated_dets)
    if hasattr(config, 'max_memory_gb_for_determinants'):
        config.max_memory_gb_for_determinants = args.max_memory_gb_for_determinants
    
    if rank == 0:
        print("Configuration:")
        print(f"  Device: {sbd.get_device()}")
        print(f"  Communication: {sbd.get_comm_backend()}")
        print(f"  Method: Davidson")
        print(f"  Max iterations: {config.max_it}")
        print(f"  Tolerance: {config.eps}")
        print(f"  MPI ranks: {size}")
        print(f"  MPI configuration: task_comm_size={args.task_comm_size} "
              f"adet_comm_size={args.adet_comm_size} "
              f"bdet_comm_size={args.bdet_comm_size}")
        _grid = (args.task_comm_size * args.adet_comm_size
                 * args.bdet_comm_size)
        if size % _grid:
            print(f"  ERROR: {size} ranks is not a multiple of {_grid}. SBD "
                  f"derives the helper dimension by integer division and then "
                  f"requires task x adet x bdet x helper == ranks exactly, so this "
                  f"will abort in TaskCommunicator with 'MPI Size of twister is not "
                  f"a square of a integer'. Use a multiple of {_grid} ranks.")
        print(f"\nInput files:")
        print(f"  FCIDUMP: {args.fcidump}")
        print(f"  Alpha dets: {args.adetfile}")
        print()

    # Run calculation (no comm parameter needed!)
    try:
        if rank == 0:
            print("Running TPB diagonalization...")
            print()

        if args.dump_matrix_form_wf:
            config.dump_matrix_form_wf = args.dump_matrix_form_wf

        if args.symmetrize_spin:
            if args.bdetfile and rank == 0:
                print(f"WARNING: --bdetfile {args.bdetfile!r} is ignored because "
                      "--symmetrize_spin is 1 (default) -- symmetric mode always "
                      "derives beta from --adetfile alone. Pass --symmetrize_spin 0 "
                      "to use a distinct beta-determinant file.\n")
            results = sbd.tpb_diag_from_files(
                fcidumpfile=args.fcidump,
                adetfile=args.adetfile,
                sbd_data=config,
                loadname=args.loadname,
                savename=args.savename,
            )
        else:
            # No bound function accepts two separate determinant files
            # directly (tpb_diag_from_files always derives beta from the one
            # adetfile it's given) -- load both ourselves and call the
            # data-structure entry point instead, mirroring what
            # tpb_diag_from_files itself does internally after loading
            # (sbdiag.h: LoadAlphaDets(...); sort_bitarray(adet);).
            bdetfile = args.bdetfile or args.adetfile
            fcidump = sbd.LoadFCIDump(args.fcidump)
            norb = int(fcidump.header["NORB"])
            adet = sbd.sort_bitarray(
                sbd.LoadAlphaDets(args.adetfile, args.bit_length, norb))
            bdet = sbd.sort_bitarray(
                sbd.LoadAlphaDets(bdetfile, args.bit_length, norb))
            results = sbd.tpb_diag(
                fcidump, adet, bdet, config,
                loadname=args.loadname, savename=args.savename,
            )

        if rank == 0:
            print("="*70)
            print("Results")
            print("="*70)
            print(f"Device: {sbd.get_device().upper()}")
            print(f"Ground state energy: {results['energy']:.10f} Hartree")

            # Output density in same format as C++ version
            # C++ outputs: density[2*i] + density[2*i+1] for each orbital
            density = results['density']
            combined_density = []
            for i in range(len(density)//2):
                combined_density.append(density[2*i] + density[2*i+1])

            print(f"Density: {combined_density}")
            print(f"Carryover determinants: {len(results['carryover_adet'])}")

            norb = len(density) // 2
            rdm1, rdm2 = assemble_rdms(results, norb)
            if rdm1 is not None:
                print()
                print(f"1-RDM trace: {np.trace(rdm1):.6f} "
                      "(should equal the total electron count)")
                occupations = np.sort(np.linalg.eigvalsh(rdm1))[::-1]
                print(f"Natural orbital occupations (sorted): "
                      f"{np.round(occupations, 6).tolist()}")
                print("  -- occupations near 2 or 0 indicate a "
                      "single-reference-like orbital; occupations near 1 "
                      "(or several clustered together) flag multi-reference "
                      "character / a candidate active space.")
                if args.rdm_output:
                    np.savez(args.rdm_output, rdm1=rdm1, rdm2=rdm2)
                    print(f"Saved rdm1/rdm2 to {args.rdm_output}")

            print("="*70)
            print("\n✓ Calculation completed successfully!")
            print()

        return_code = 0

    except FileNotFoundError as e:
        if rank == 0:
            print(f"\n✗ Error: {e}")
            print("\nPlease check file paths:")
            print(f"  FCIDUMP: {args.fcidump}")
            print(f"  Alpha dets: {args.adetfile}")
        return_code = 1

    except Exception as e:
        if rank == 0:
            print(f"\n✗ Error during calculation: {e}")
            import traceback
            traceback.print_exc()
        return_code = 1

    finally:
        # Synchronize GPU and reset internal state
        # Note: Calls cudaDeviceSynchronize() but NOT cudaDeviceReset() to avoid
        # conflicts with CUDA-aware MPI (UCX). Does not call MPI_Finalize() either.
        sbd.finalize()

    return return_code

if __name__ == "__main__":
    sys.exit(main())
