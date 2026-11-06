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

"""Green's function of a CAOP ground state saved by run_caop_diag.py, by block Lanczos.

The Python counterpart of upstream SBD's ``caop-spectrum`` app, with its option names
and its CSV output. Two modes, as upstream's:

* generate: from a ground state saved with ``run_caop_diag.py --savename PREFIX``::

      python run_caop_spectrum.py --hamfile ham.txt --sites 8 --loadname PREFIX \\
          --reference-energy E0 --orbitals 0,4 --channel addition \\
          --extra-detfile addition_basis.txt --save-coefficients add.coeff \\
          --output add.csv --omega-min -6 --omega-max 8 --points 141 --eta 0.1

* reevaluate: the saved coefficients on a new grid or broadening, with no
  Hamiltonian or state::

      python run_caop_spectrum.py --read-coefficients add.coeff --output narrow.csv \\
          --omega-min -6 --omega-max 8 --points 281 --eta 0.05
"""

import argparse
import sys


def _csv(text):
    return [s for s in text.split(',') if s]


def parse_args():
    parser = argparse.ArgumentParser(
        description="CAOP Green's function (upstream caop-spectrum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                        help="'cpu', or 'gpu' (NVHPC Thrust, NVIDIA GPUs only)")
    parser.add_argument('--read-coefficients', default='',
                        help='Reevaluate these saved coefficients; needs only --output and '
                             'the grid')
    parser.add_argument('--hamfile', help='Hamiltonian file the state was solved with')
    parser.add_argument('--loadname', help='Prefix the ground state was saved under')
    parser.add_argument('--wavefunction-shards', type=int, default=1,
                        help='Number of files the state was saved in (one per --b_comm_size '
                             'rank of the ground-state run)')
    parser.add_argument('--sites', type=int, help="Modes, the state's --system_size")
    parser.add_argument('--bit-length', type=int, default=20,
                        help="The state's --bit_length; the files do not record it")
    parser.add_argument('--reference-energy', type=float,
                        help="The ground state's energy E0, as run_caop_diag.py printed it")
    parser.add_argument('--orbitals', type=_csv, default=[],
                        help='Comma-separated modes for c / cdag')
    parser.add_argument('--channel', default='addition', choices=['addition', 'removal'])
    parser.add_argument('--applied-operator-type', default='single-particle',
                        choices=['single-particle', 'general'])
    parser.add_argument('--operator-file', action='append', default=[],
                        help='A CAOP operator file, repeated once per observable, for '
                             '--applied-operator-type general')
    parser.add_argument('--extra-detfile', action='append', default=[],
                        help='Determinants added to the excitation basis, repeatable; '
                             'usually the whole N+1 or N-1 sector')
    parser.add_argument('--steps', type=int, default=100, help='Most block-Lanczos steps')
    parser.add_argument('--rank-atol', type=float, default=1e-13)
    parser.add_argument('--rank-rtol', type=float, default=1e-12)
    parser.add_argument('--hamiltonian-mode', default='stored', choices=['stored', 'on-the-fly'],
                        help="'on-the-fly' is the only mode that uses the GPU")
    parser.add_argument('--h-comm-size', type=int, default=1)
    parser.add_argument('--t-comm-size', type=int, default=1)
    parser.add_argument('--scalar-type', default='real', choices=['real', 'complex'])
    parser.add_argument('--wavefunction-type', default=None, choices=['real', 'complex'],
                        help='Type the state was saved with; default the scalar type')
    parser.add_argument('--save-coefficients', default='',
                        help='Write the coefficients here, for --read-coefficients')
    parser.add_argument('--output', default='', help='CSV of G and A on the grid')
    parser.add_argument('--omega-min', type=float, default=-10.0)
    parser.add_argument('--omega-max', type=float, default=10.0)
    parser.add_argument('--points', type=int, default=201)
    parser.add_argument('--eta', type=float, default=0.1)
    args = parser.parse_args()
    if args.read_coefficients:
        if not args.output:
            parser.error("--read-coefficients needs --output")
        return args
    missing = [f"--{n.replace('_', '-')}" for n in ('hamfile', 'loadname', 'sites',
                                                     'reference_energy')
               if getattr(args, n) is None]
    if missing:
        parser.error(f"generating needs {', '.join(missing)}")
    general = args.applied_operator_type == 'general'
    if general and (args.orbitals or not args.operator_file):
        parser.error("--applied-operator-type general takes --operator-file, not --orbitals")
    if not general and (args.operator_file or not args.orbitals):
        parser.error("single-particle needs --orbitals (and no --operator-file)")
    if not (args.save_coefficients or args.output):
        parser.error("give --save-coefficients, --output, or both")
    return args


def _write_csv(path, omegas, green, labels, general):
    """Upstream's CSV: omega, a, b, then G and A, real and imaginary parts."""
    from sbd.spectrum import spectral_function

    spectral = spectral_function(green)
    names = "operator_a,operator_b" if general else "orbital_a,orbital_b"
    with open(path, "w") as f:
        f.write(f"omega,{names},G_real,G_imag,A_real,A_imag\n")
        for k, w in enumerate(omegas):
            for i, a in enumerate(labels):
                for j, b in enumerate(labels):
                    g, s = complex(green[k, i, j]), complex(spectral[k, i, j])
                    f.write(",".join([f"{float(w):.17g}", str(a), str(b)]
                                     + [f"{x:.17g}" for x in (g.real, g.imag, s.real, s.imag)])
                            + "\n")


def main():
    args = parse_args()
    import numpy as np

    import sbd
    import sbd.spectrum as sp

    sbd.init(device=args.device)
    rank = sbd.get_rank()
    omegas = np.linspace(args.omega_min, args.omega_max, args.points)
    general = args.applied_operator_type == 'general'

    if args.read_coefficients:
        result = sp.green_function_from_coefficients(args.read_coefficients, omegas,
                                                     args.eta)
        n = result["green_function"].shape[1]
        if rank == 0:
            # Label the rows as the generating run did: its orbitals, or for general
            # operators the operator indices 0..n-1.
            general = bool(result["operator_files"])
            labels = list(range(n)) if general else result["orbitals"]
            _write_csv(args.output, omegas, result["green_function"], labels, general)
            print(f"Reevaluated {args.read_coefficients}: {n} x {n}, "
                  f"{len(omegas)} points, eta {args.eta} -> {args.output}")
        return 0

    if rank == 0 and args.device == 'gpu':
        if args.hamiltonian_mode == 'stored':
            print("NOTE: --hamiltonian-mode stored builds and applies the Hamiltonian on "
                  "the CPU only: on --device gpu this run uses no GPU. For the GPU kernel, "
                  "use --hamiltonian-mode on-the-fly.")
        else:
            print("NOTE: --hamiltonian-mode on-the-fly applies the Hamiltonian on the GPU; "
                  "the seed generation, QR and Lanczos vector work stay on the CPU. For a "
                  "small basis, compare against --device cpu with --hamiltonian-mode "
                  "stored.")
    complex_ = args.scalar_type == 'complex'
    wavefunction_complex = (args.wavefunction_type or args.scalar_type) == 'complex'
    orbitals = [int(o) for o in args.orbitals]
    result = sp.green_function(
        args.loadname, args.hamfile, orbitals, args.channel, sites=args.sites,
        bit_length=args.bit_length, reference_energy=args.reference_energy,
        omegas=omegas, eta=args.eta, shards=args.wavefunction_shards,
        extra_basis=args.extra_detfile, max_steps=args.steps,
        h_comm_size=args.h_comm_size, t_comm_size=args.t_comm_size,
        store_hamiltonian=args.hamiltonian_mode == 'stored', complex=complex_,
        wavefunction_complex=wavefunction_complex,
        save_coefficients=args.save_coefficients, operator_files=args.operator_file,
        rank_atol=args.rank_atol, rank_rtol=args.rank_rtol, device=args.device)
    if rank == 0:
        labels = list(range(len(args.operator_file))) if general else orbitals
        if args.output:
            _write_csv(args.output, omegas, result["green_function"], labels, general)
        print(f"Excitation basis: {result['basis_size']}, steps: {result['steps']}, "
              f"stop: {result['stop_reason']}")
        for path in (args.save_coefficients, args.output):
            if path:
                print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
