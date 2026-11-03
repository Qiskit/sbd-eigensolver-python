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

"""Green's functions of a CAOP ground state, by block Lanczos.

The Python counterpart of upstream SBD's
`caop-spectrum <https://github.com/r-ccs-cms/sbd/tree/main/apps/caop-spectrum>`_ app.
The workflow, for a lattice model such as the Hubbard model:

1. Write the model as a CAOP Hamiltonian file, and solve for its ground state with
   :func:`sbd.caop_diag_from_files`, saving it with ``savename``.
2. Call :func:`green_function` with that state, the operator modes to add an electron
   to (``channel="addition"``) or remove one from (``"removal"``), and a frequency
   grid. It applies the creation or annihilation operators to the state and runs block
   Lanczos in the excitation basis.
3. :func:`spectral_function` turns the result into the spectral function, whose peaks
   are the addition or removal energies and whose weights are the spectral weights.

The conventions are upstream's, for addition and removal respectively::

    G+_ab(w) = <psi0| c_a (w + E0 - H + i eta)^-1 c_b^dag |psi0>
    G-_ab(w) = <psi0| c_b^dag (w - E0 + H + i eta)^-1 c_a |psi0>

Add the two channels on a common grid for the fermionic retarded Green's function.

**The excitation basis.** The basis the Lanczos runs in is the union of the
determinants the operators generate from the state and any ``extra_basis`` files.
Upstream warns that the generated determinants alone are usually not enough: neither
mode completes the particle sector by itself, so give ``extra_basis`` the determinants
of the N+1 (addition) or N-1 (removal) sector you want included.
"""

from __future__ import annotations

import os

import numpy as np

import sbd

__all__ = ["green_function", "green_function_from_coefficients", "spectral_function"]


def green_function(loadname, hamfile, orbitals, channel, *, sites, bit_length,
                   reference_energy, omegas, eta=0.05, shards=1, extra_basis=(),
                   max_steps=100, h_comm_size=1, t_comm_size=1, store_hamiltonian=True,
                   complex=False, wavefunction_complex=False, save_coefficients="",
                   operator_files=(), rank_atol=1e-13, rank_rtol=1e-12, device=None):
    """The Green's function of a saved CAOP ground state on a frequency grid.

    Args:
        loadname: The ``savename`` prefix the ground state was saved under, by
            :func:`sbd.caop_diag_from_files` (files ``loadname000000.bin`` ...).
        hamfile: The CAOP Hamiltonian file the state was solved with.
        orbitals: Modes to apply the operators to (0-based CAOP mode indices). Empty
            when ``operator_files`` is given.
        channel: ``"addition"`` (creation operators, the N+1 sector) or
            ``"removal"`` (annihilation operators, the N-1 sector). Ignored for
            general operators, whose response is
            ``<psi0| O_a^dag (w + E0 - H + i eta)^-1 O_b |psi0>``.
        sites: Number of modes, as the state's ``system_size``.
        bit_length: Bits per packed word, as the state was saved with.
        reference_energy: The ground state's energy E0, as the solve returned it.
        omegas: Frequencies to evaluate the Green's function at.
        eta: Broadening.
        shards: Number of files the state was saved in.
        extra_basis: Files of determinants to add to the excitation basis, one
            bitstring of ``sites`` characters per line; see the module notes.
        max_steps: Most block-Lanczos steps.
        h_comm_size, t_comm_size: Helper and task communicator sizes; their product
            must divide the rank count, and the rest of the ranks hold the basis.
        store_hamiltonian: Store H for faster application (upstream's ``stored``
            mode), rather than applying it on the fly.
        complex: Run with complex amplitudes, for a complex Hamiltonian.
        wavefunction_complex: The saved state has complex amplitudes, as one solved
            with ``caop_diag_from_files(..., complex=True)`` does; needs ``complex``.
            Real and complex files are not interchangeable, so give the type the state
            was saved with. A real state is promoted when ``complex`` is set.
        save_coefficients: If given, write upstream's coefficient file here, from
            which :func:`green_function_from_coefficients` re-evaluates the spectrum on
            another grid or broadening without rerunning.
        operator_files: CAOP operator files to apply instead of ``c``/``cdag`` on
            ``orbitals`` (upstream's ``--applied-operator-type general``), one per
            observable, with the Hamiltonian's statistics. They may be non-Hermitian
            or change the particle number.
        rank_atol, rank_rtol: Block-Lanczos deflation tolerances: a column whose norm
            falls below them is dropped. Upstream's defaults.
        device: ``"cpu"`` or ``"gpu"`` (Thrust); there is no OpenMP-offload path.

    Returns:
        dict with ``green_function``, a complex array of shape
        ``(len(omegas), n, n)`` for ``n`` orbitals or operators; ``basis_size``, the
        excitation basis's size; ``steps``; and ``stop_reason``.
    """
    if channel not in ("addition", "removal"):
        raise ValueError(f"channel must be 'addition' or 'removal', got {channel!r}")
    if isinstance(extra_basis, (str, os.PathLike)):
        extra_basis = [extra_basis]
    if isinstance(operator_files, (str, os.PathLike)):
        operator_files = [operator_files]
    backend = sbd._caop_backend(device)
    return backend.caop_spectrum(
        sbd._global_comm, str(loadname), int(shards), str(hamfile),
        [int(o) for o in orbitals], channel == "addition", int(sites), int(bit_length),
        float(reference_energy), [str(f) for f in extra_basis],
        [float(w) for w in np.atleast_1d(omegas)], float(eta), int(max_steps),
        int(h_comm_size), int(t_comm_size), bool(store_hamiltonian), bool(complex),
        bool(wavefunction_complex), str(save_coefficients),
        [str(f) for f in operator_files], float(rank_atol), float(rank_rtol))


def green_function_from_coefficients(path, omegas, eta, *, device=None):
    """The Green's function from coefficients saved by :func:`green_function`.

    Upstream's ``--read-coefficients``: the block-Lanczos coefficients fix the
    response, so a new frequency grid or broadening needs neither the Hamiltonian nor
    the state. Rank-local. It cannot extend the Lanczos run itself.

    Returns:
        dict with ``green_function`` (``(len(omegas), n, n)``), ``basis_size``,
        ``steps``, ``stop_reason`` and ``addition`` (the channel it was saved for).
    """
    backend = sbd._caop_backend(device)
    return backend.caop_spectrum_from_coefficients(
        str(path), [float(w) for w in np.atleast_1d(omegas)], float(eta))


def spectral_function(green):
    """The spectral matrix ``-(G - G^dag) / (2 pi i)`` at each frequency.

    Its diagonal is ``-Im(G_aa) / pi``: peaks at the addition or removal energies,
    with the spectral weights as their areas.
    """
    g = np.asarray(green)
    return -(g - np.conj(np.swapaxes(g, -1, -2))) / (2j * np.pi)
