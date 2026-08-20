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
"""

import argparse
import sys

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
                       help='Path to beta determinants file (optional, uses adetfile if not specified)')
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
                       help='Shuffle determinants (0=no, 1=yes)')
    parser.add_argument('--rdm', '--do_rdm', type=int, default=0, choices=[0, 1], dest='do_rdm',
                       help='Calculate RDM (0=density only, 1=full RDM)')
    
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
    config.do_rdm = args.do_rdm
    config.bit_length = args.bit_length
    config.carryover_type = args.carryover_type
    config.ratio = args.ratio
    config.threshold = args.threshold
    config.adet_comm_size = args.adet_comm_size
    config.bdet_comm_size = args.bdet_comm_size
    config.task_comm_size = args.task_comm_size
    
    if rank == 0:
        print("Configuration:")
        print(f"  Device: {sbd.get_device()}")
        print(f"  Communication: {sbd.get_comm_backend()}")
        print(f"  Method: Davidson")
        print(f"  Max iterations: {config.max_it}")
        print(f"  Tolerance: {config.eps}")
        print(f"  MPI configuration: {args.task_comm_size} × {args.adet_comm_size} × {args.bdet_comm_size} = {size} ranks")
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

        results = sbd.tpb_diag_from_files(
            fcidumpfile=args.fcidump,
            adetfile=args.adetfile,
            sbd_data=config,
            loadname=args.loadname,
            savename=args.savename,
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
