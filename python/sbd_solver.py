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
SBD solver wrapper compatible with qiskit-addon-sqd interface.

This module provides functions that wrap the SBD (Selected Basis Diagonalization)
library to be compatible with the qiskit-addon-sqd diagonalize_fermionic_hamiltonian
interface, similar to how qiskit-addon-dice-solver works.
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path
from collections.abc import Sequence
from typing import Callable

import numpy as np
from mpi4py import MPI

# Bits packed into each size_t of SBD's ``std::vector<size_t>`` bitstring
# representation. RIKEN's documented default is 20 (see the --bit_length option
# in apps/chemistry_tpb_selected_basis_diagonalization/README.md), which is also
# what run_sbd_diag.py defaults to.
#
# This must stay <= 63. bitadvance() in framework/bit_manipulation.h computes
#     size_t d = (((size_t) 1) << bit_length) - 1;
# so bit_length == 64 shifts a 64-bit size_t by 64, which is undefined behavior;
# in practice the shift count is masked to 0 and the mask collapses to d == 0.
SBD_DEFAULT_BIT_LENGTH = 20

try:
    from pyscf import tools as pyscf_tools
except ImportError:
    pyscf_tools = None

try:
    from qiskit_addon_sqd.fermion import SCIResult, SCIState
except ImportError:
    SCIResult = None
    SCIState = None


def _resolve_backend(device_config=None):
    """Resolve a backend module from a DeviceConfig or the default.

    Reads ``device_config.device`` (string key, e.g. 'cpu', 'gpu',
    'gpu-omp') and routes through ``sbd.get_backend()`` which handles
    aliases ('cuda', 'gpu-thrust', 'gpu-nvhpc-omp', etc.).
    """
    from . import get_backend
    if device_config is not None:
        return get_backend(device_config.device)
    return get_backend()


def solve_sci(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    mpi_comm: MPI.Comm | None = None,
    sbd_config: dict | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
    device_config=None,
    fcidump_path: str | Path | None = None,
    report_carryover: bool | None = None,
) -> SCIResult:
    """
    Diagonalize Hamiltonian in subspace defined by CI strings using SBD.

    Args:
        ci_strings: Pair (strings_a, strings_b) of CI string arrays.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for total spin squared (unused by SBD).
        mpi_comm: MPI communicator. If None, uses MPI.COMM_WORLD.
        sbd_config: Dictionary of SBD configuration parameters.
        temp_dir: Directory for temporary files.
        clean_temp_dir: Whether to delete intermediate files.
        device_config: DeviceConfig object to select CPU/GPU backend.
        fcidump_path: If set, load the FCIDUMP directly from this path on
            every rank and skip the tensor->file round-trip. MUST be on a
            filesystem visible to every rank (shared on multi-node runs).
            When None (default), rank 0 writes a regenerated FCIDUMP into
            ``temp_dir`` and every rank opens that file, which requires
            ``temp_dir`` to be shared for multi-node runs.
        report_carryover: Whether to report the determinants SBD retained as
            :attr:`~qiskit_addon_sqd.fermion.SCIResult.carryover` and return no
            eigenvector, rather than returning an
            :class:`~qiskit_addon_sqd.fermion.SCIState`.

            SBD trims its own subspace whenever ``carryover_type`` is nonzero, so the
            determinants it kept are already the ones a configuration recovery loop
            needs. Reporting them directly avoids writing the eigenvector to disk and
            reading it back: its size is the product of the two spin sector dimensions,
            which is the one quantity a large distributed calculation cannot afford to
            move between processes.

            Defaults to ``None``, meaning report the carryover when ``carryover_type``
            is nonzero and return an eigenvector otherwise. Since ``carryover_type``
            itself defaults to 0, reporting the carryover is opt-in: set it to 1, 2 or 3
            through ``sbd_config``. Pass ``False`` to keep the eigenvector even when SBD
            trims, which is what a caller that wants the amplitudes for its own analysis
            should do. Passing ``True`` with ``carryover_type=0`` raises ``ValueError``,
            since SBD then selects no determinants to report.

            SBD's carryover is selected by weight and returned in descending weight
            order, whereas the eigenvector path labels its amplitudes with the
            canonically ordered, deduplicated lists SBD was given. A caller that needs a
            particular ordering should impose it rather than assume one.

    Returns:
        The diagonalization result as SCIResult.
    """
    if SCIResult is None:
        raise ImportError(
            "qiskit-addon-sqd is required for solve_sci. "
            "Install with: pip install qiskit-addon-sqd"
        )
    backend = _resolve_backend(device_config)

    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    sbd_dir, owns_sbd_dir = _make_sbd_dir(mpi_comm, mpi_rank, temp_dir)

    try:
        fcidump, ecore_offset = _load_or_regenerate_fcidump(
            backend, mpi_rank, mpi_comm, sbd_dir, fcidump_path,
            one_body_tensor, two_body_tensor, norb, nelec,
        )

        return _solve_sci_core(
            ci_strings,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
            report_carryover=report_carryover,
            mpi_comm=mpi_comm,
            mpi_rank=mpi_rank,
            sbd_config=sbd_config,
            sbd_dir=sbd_dir,
            backend=backend,
            fcidump=fcidump,
            device_config=device_config,
            ecore_offset=ecore_offset,
        )
    finally:
        if clean_temp_dir and owns_sbd_dir and mpi_rank == 0:
            shutil.rmtree(sbd_dir, ignore_errors=True)


def _solve_sci_core(
    ci_strings: tuple[np.ndarray, np.ndarray],
    *,
    norb: int,
    nelec: tuple[int, int],
    spin_sq: float | None,
    mpi_comm,
    mpi_rank: int,
    sbd_config: dict | None,
    sbd_dir: Path,
    backend,
    fcidump,
    device_config=None,
    ecore_offset: float = 0.0,
    report_carryover: bool | None = None,
) -> SCIResult:
    """
    Inner diagonalization kernel that operates on a pre-loaded FCIDUMP object.

    Separated from solve_sci so that solve_sci_batch can write and load
    the FCIDUMP only once and reuse it across all batches.
    """
    strings_a, strings_b = ci_strings

    # Build the config first: it carries the effective bit_length (possibly
    # overridden by the caller), and the determinants must be packed with the
    # same value the C++ engine will use to interpret them.
    sbd_data = _create_sbd_config(sbd_config, backend, device_config)

    adet = _ci_strings_to_sbd_dets(strings_a, norb, backend, sbd_data.bit_length)
    bdet = _ci_strings_to_sbd_dets(strings_b, norb, backend, sbd_data.bit_length)

    # SBD trims its own subspace when carryover_type is nonzero, in which case the
    # surviving determinants are what the caller needs and the eigenvector is not. See
    # the report_carryover argument. Note that _create_sbd_config defaults
    # carryover_type to 0, so this is opt-in: a caller wanting SBD's selection must set
    # carryover_type through sbd_config.
    trims_own_subspace = sbd_data.carryover_type != 0
    if report_carryover is None:
        report_carryover = trims_own_subspace
    elif report_carryover and not trims_own_subspace:
        raise ValueError(
            "report_carryover=True requires SBD to select carryover determinants, but "
            "carryover_type is 0, so it selects none. Set carryover_type to 1, 2 or 3 in "
            "sbd_config, or leave report_carryover unset."
        )

    wf_dump_file = sbd_dir / "wavefunction.bin"
    if not report_carryover:
        # Use .bin extension to trigger SBD's fast binary write path
        # (SaveMatrixFormWF in restart.h checks extension: .bin -> raw doubles)
        sbd_data.dump_matrix_form_wf = str(wf_dump_file)

    results = backend.tpb_diag(
        mpi_comm, sbd_data, fcidump, adet, bdet, loadname="", savename=""
    )

    # Rank 0 reads the wavefunction file; Barrier ensures it's flushed.
    mpi_comm.Barrier()

    if mpi_rank != 0:
        # The configuration recovery loop reads results on the control process only, so
        # the other ranks need return nothing but a well-formed object.
        return SCIResult(
            0.0,
            None if report_carryover else _empty_sci_state(norb, nelec),
            orbital_occupancies=(
                np.zeros(norb, dtype=np.float64),
                np.zeros(norb, dtype=np.float64),
            ),
            carryover=(
                (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
                if report_carryover
                else None
            ),
        )

    # --- rank 0 only ---

    # Subtract ECORE so SCIResult.energy is the pure electronic piece,
    # matching the contract callers rely on (they add nuclear_repulsion
    # separately). The regenerate path writes ECORE=0, so offset=0 there;
    # the direct-load path passes whatever ECORE lives in the user file.
    energy = results["energy"] - ecore_offset
    density = np.array(results["density"])
    occupancies_a = density[::2]
    occupancies_b = density[1::2]
    occupancies = (occupancies_a, occupancies_b)

    if report_carryover:
        # The determinants SBD retained are reported directly, so the eigenvector never
        # has to be written to disk and read back. Its size is the product of the two
        # spin sector dimensions, which is what a large calculation cannot afford to move.
        #
        # These come back in descending weight order, not canonical order: CarryOverAdet
        # and CarryOverBdet (upstream chemistry/tpb/rdmat.h) sort by the diagonal reduced
        # density matrix and keep the top entries, and never call sort_bitarray. That is
        # the order a trim wants, and qiskit-addon-sqd applies its own shape constraints
        # to whatever a policy returns, so neither ordering nor uniqueness is assumed
        # here.
        return SCIResult(
            energy,
            None,
            orbital_occupancies=occupancies,
            carryover=(
                _sbd_dets_to_ci_strings(
                    results["carryover_adet"], norb, backend, sbd_data.bit_length
                ),
                _sbd_dets_to_ci_strings(
                    results["carryover_bdet"], norb, backend, sbd_data.bit_length
                ),
            ),
        )

    # Read wavefunction coefficients from the binary dump.
    #
    # SaveMatrixFormWF (upstream chemistry/tpb/restart.h) writes the FULL input
    # subspace -- adet x bdet, row-major, and it self-checks that total -- never
    # the carryover subset, whatever carryover_type is set to. Sizing this read
    # against the carryover counts therefore never matched under the default
    # carryover_type=1 / ratio=0.1, and the previous code answered the mismatch by
    # substituting a uniform array: a correctly-shaped, normalised matrix that
    # looks like a wavefunction and carries no information. That was silent, and
    # it mattered -- qiskit-addon-sqd selects the next iteration's determinants by
    # amplitude magnitude (fermion.py, _carryover_* / weights_a / weights_b), so
    # every iteration after the first was seeded from uniform weights. See #19.
    #
    # The dump is requested unconditionally above, so neither a missing file nor a
    # size mismatch is a situation to paper over: both mean the run did not do what
    # was asked, and a loud failure is the only honest response.
    # Label the amplitudes with the lists that were actually diagonalized, not with
    # the inputs. _ci_strings_to_sbd_dets ends in sort_bitarray, which sorts into
    # SBD's canonical order AND removes duplicates, so adet/bdet can differ from
    # strings_a/strings_b -- and the dump is written in adet/bdet order. Deriving the
    # labels from adet/bdet makes the amplitudes and their labels come from one list
    # by construction, instead of relying on the two orders agreeing.
    solved_strings_a = _sbd_dets_to_ci_strings(adet, norb, backend, sbd_data.bit_length)
    solved_strings_b = _sbd_dets_to_ci_strings(bdet, norb, backend, sbd_data.bit_length)
    n_a = len(solved_strings_a)
    n_b = len(solved_strings_b)
    if not wf_dump_file.exists():
        raise RuntimeError(
            f"SBD wrote no wavefunction dump at {wf_dump_file}. It is requested on "
            "every call, so a missing file means the diagonalization did not "
            "complete as expected."
        )
    flat = np.fromfile(str(wf_dump_file), dtype=np.float64)
    if flat.size != n_a * n_b:
        raise RuntimeError(
            f"wavefunction dump at {wf_dump_file} holds {flat.size} amplitudes, "
            f"expected {n_a * n_b} ({n_a} alpha x {n_b} beta over the "
            "diagonalized subspace). SaveMatrixFormWF writes the full subspace; a "
            "different size means the dump and the subspace have diverged."
        )

    sci_state = SCIState(
        amplitudes=flat.reshape(n_a, n_b),
        ci_strs_a=solved_strings_a,
        ci_strs_b=solved_strings_b,
        norb=norb,
        nelec=nelec,
    )

    rdm1, rdm2 = assemble_rdms(results, norb)

    return SCIResult(energy, sci_state, orbital_occupancies=occupancies, rdm1=rdm1, rdm2=rdm2)


def assemble_rdms(results: dict, norb: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build spin-summed (rdm1, rdm2) from SBD's raw one_p_rdm/two_p_rdm.

    Returns (None, None) when ``do_rdm`` was 0 (SBD leaves these keys as
    empty lists in that case -- see ``sbdiag.h``'s ``do_rdm == 0`` branch,
    which only computes the diagonal density, not the correlation
    functions), matching SCIResult's own ``rdm1``/``rdm2`` default.

    The reshape/transpose below is not a guess: it was verified against
    PySCF's ``make_rdm1``/``make_rdm2`` on a fixed subspace, on all three
    SBD backends (cpu, gpu-thrust, gpu-omp-offload) -- both element-wise on
    the full tensors and via the energy identity
    ``E = einsum("pr,pr->",rdm1,hcore) + 0.5*einsum("prqs,prqs->",rdm2,eri)``
    that ``SCIResult.rdm1``/``rdm2`` are contracted with everywhere else in
    qiskit-addon-sqd (e.g. ``fermion.py``'s own ``solve_fermion``).

    SBD's documented layout (sbd-ext docs/user-guide.md, matching the C++
    reference in apps/chemistry_tpb_selected_basis_diagonalization/main.cc):
        one_p_rdm[s][i + L*j]                 = <c+_{i,s} c_{j,s}>
        two_p_rdm[s+2t][i + L*j + L^2*k + L^3*l] = <c+_{i,s} c+_{j,t} c_{l,t} c_{k,s}>
    A Fortran-order reshape implements those flat-index formulas directly
    (arr_F[i, j] / arr_F[i, j, k, l]); rdm1 needs no further transpose
    (it is symmetric here regardless), and rdm2's spin-summed block sum
    needs axes (0, 2, 1, 3) to land in the "prqs" slot order SCIResult's
    contract expects.
    """
    one_p_rdm = results.get("one_p_rdm")
    two_p_rdm = results.get("two_p_rdm")
    if not one_p_rdm or not two_p_rdm:
        return None, None

    one_p_rdm = np.asarray(one_p_rdm)
    two_p_rdm = np.asarray(two_p_rdm)
    L = norb

    rdm1 = (np.reshape(one_p_rdm[0], (L, L), order="F")
            + np.reshape(one_p_rdm[1], (L, L), order="F"))

    spin_summed = sum(
        np.reshape(two_p_rdm[s], (L, L, L, L), order="F") for s in range(4)
    )
    rdm2 = spin_summed.transpose(0, 2, 1, 3)

    return rdm1, rdm2


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    mpi_comm: MPI.Comm | None = None,
    sbd_config: dict | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
    device_config=None,
    fcidump_path: str | Path | None = None,
    report_carryover: bool | None = None,
) -> list[SCIResult]:
    """
    Diagonalize Hamiltonian in multiple subspaces using SBD.

    The FCIDUMP file is loaded once and reused across all batches.

    Args:
        ci_strings: List of (strings_a, strings_b) pairs.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for total spin squared (unused by SBD).
        mpi_comm: MPI communicator. If None, uses MPI.COMM_WORLD.
        sbd_config: Dictionary of SBD configuration parameters.
        temp_dir: Directory for temporary files.
        clean_temp_dir: Whether to delete intermediate files.
        device_config: DeviceConfig object to select CPU/GPU backend.
        fcidump_path: If set, load the FCIDUMP directly from this path on
            every rank and skip the tensor->file round-trip. MUST be on a
            filesystem visible to every rank (shared on multi-node runs).
            When None (default), rank 0 writes a regenerated FCIDUMP into
            ``temp_dir`` and every rank opens that file, which requires
            ``temp_dir`` to be shared for multi-node runs.
        report_carryover: Whether to report the determinants SBD retained as
            :attr:`~qiskit_addon_sqd.fermion.SCIResult.carryover` and return no
            eigenvector, rather than returning an
            :class:`~qiskit_addon_sqd.fermion.SCIState`.

            SBD trims its own subspace whenever ``carryover_type`` is nonzero, so the
            determinants it kept are already the ones a configuration recovery loop
            needs. Reporting them directly avoids writing the eigenvector to disk and
            reading it back: its size is the product of the two spin sector dimensions,
            which is the one quantity a large distributed calculation cannot afford to
            move between processes.

            Defaults to ``None``, meaning report the carryover when ``carryover_type``
            is nonzero and return an eigenvector otherwise. Since ``carryover_type``
            itself defaults to 0, reporting the carryover is opt-in: set it to 1, 2 or 3
            through ``sbd_config``. Pass ``False`` to keep the eigenvector even when SBD
            trims, which is what a caller that wants the amplitudes for its own analysis
            should do. Passing ``True`` with ``carryover_type=0`` raises ``ValueError``,
            since SBD then selects no determinants to report.

            SBD's carryover is selected by weight and returned in descending weight
            order, whereas the eigenvector path labels its amplitudes with the
            canonically ordered, deduplicated lists SBD was given. A caller that needs a
            particular ordering should impose it rather than assume one.

    Returns:
        List of SCIResult for each batch.
    """
    if not ci_strings:
        return []

    backend = _resolve_backend(device_config)

    if mpi_comm is None:
        mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()

    sbd_dir, owns_sbd_dir = _make_sbd_dir(mpi_comm, mpi_rank, temp_dir)

    try:
        fcidump, ecore_offset = _load_or_regenerate_fcidump(
            backend, mpi_rank, mpi_comm, sbd_dir, fcidump_path,
            one_body_tensor, two_body_tensor, norb, nelec,
        )

        return [
            _solve_sci_core(
                ci_strs,
                norb=norb,
                nelec=nelec,
                spin_sq=spin_sq,
                mpi_comm=mpi_comm,
                mpi_rank=mpi_rank,
                sbd_config=sbd_config,
                sbd_dir=sbd_dir,
                backend=backend,
                fcidump=fcidump,
                device_config=device_config,
                ecore_offset=ecore_offset,
                report_carryover=report_carryover,
            )
            for ci_strs in ci_strings
        ]
    finally:
        if clean_temp_dir and owns_sbd_dir and mpi_rank == 0:
            shutil.rmtree(sbd_dir, ignore_errors=True)


def _make_sbd_dir(mpi_comm, mpi_rank, temp_dir):
    """Create a per-run tempdir on rank 0 and broadcast its path.

    Used to hold the wavefunction.bin written by rank 0 (and the
    regenerated fcidump.txt when ``fcidump_path`` is not provided).
    Returns (path, owns_sbd_dir) where owns_sbd_dir is True on rank 0
    (so the caller knows to rmtree it on exit).

    ``temp_dir`` is created along with any missing parents, so a caller may
    pass a path that does not exist yet (e.g. ``--temp_dir`` pointing at a
    fresh scratch location). Creation happens only on rank 0; if it fails, the
    error is broadcast so that every rank raises together, rather than the
    other ranks deadlocking in the broadcast below while rank 0 unwinds.
    """
    temp_dir = temp_dir or tempfile.gettempdir()
    # (ok, payload): payload is the created directory when ok is True, else a
    # message describing why rank 0 could not create it.
    if mpi_rank == 0:
        try:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            created = tempfile.mkdtemp(prefix="sbd_files_", dir=temp_dir)
            result = (True, str(Path(created)))
        except OSError as exc:
            result = (False, f"{type(exc).__name__}: {exc}")
    else:
        result = None
    ok, payload = mpi_comm.bcast(result, root=0)
    if not ok:
        raise RuntimeError(
            f"rank 0 could not create an SBD temporary directory under "
            f"{temp_dir!r}: {payload}"
        )
    return Path(payload), (mpi_rank == 0)


def _load_or_regenerate_fcidump(
    backend, mpi_rank, mpi_comm, sbd_dir, fcidump_path,
    one_body_tensor, two_body_tensor, norb, nelec,
):
    """Return ``(fcidump, ecore_offset)``.

    ``ecore_offset`` is the ECORE constant baked into whatever FCIDUMP
    SBD is about to load, and is subtracted from the raw energy so that
    the SCIResult contract (``energy`` is the pure electronic piece) is
    preserved in both the regenerate and direct-load paths.

    If ``fcidump_path`` is given, every rank loads that file directly
    (must be on a filesystem visible to every rank). Otherwise rank 0
    regenerates an FCIDUMP with ECORE=0 into ``sbd_dir/fcidump.txt`` and
    every rank opens that file.
    """
    if fcidump_path is not None:
        ecore_offset = _read_fcidump_ecore(fcidump_path)
        return backend.LoadFCIDump(str(fcidump_path)), ecore_offset

    fcidump_path = sbd_dir / "fcidump.txt"
    if mpi_rank == 0:
        pyscf_tools.fcidump.from_integrals(
            str(fcidump_path), one_body_tensor, two_body_tensor, norb, nelec,
        )
    mpi_comm.Barrier()
    return backend.LoadFCIDump(str(fcidump_path)), 0.0


def _read_fcidump_ecore(fcidump_path):
    """Pull the ECORE constant out of an FCIDUMP file.

    ECORE is the line whose four orbital indices are all zero. Return
    0.0 if no such line exists (some FCIDUMPs simply omit it).
    """
    with open(fcidump_path) as f:
        for line in f:
            # FCIDUMP integral lines look like: "<val> i j k l"
            parts = line.split()
            if len(parts) == 5:
                try:
                    val = float(parts[0])
                    i, j, k, l = (int(x) for x in parts[1:])
                except ValueError:
                    continue
                if i == j == k == l == 0:
                    return val
    return 0.0


def _ci_strings_to_sbd_dets(
    ci_strings: np.ndarray, norb: int, backend,
    bit_length: int = SBD_DEFAULT_BIT_LENGTH,
) -> list[list[int]]:
    """Convert CI strings (integers) to SBD determinant format.

    Determinants are sorted in canonical order (matching C++ sort_bitarray)
    which is required by the GPU Correlation kernel (do_rdm=1).
    """
    dets = []
    for ci_str in ci_strings:
        binary_str = format(int(ci_str), f'0{norb}b')
        det = backend.from_string(binary_str, bit_length, norb)
        dets.append(det)
    return backend.sort_bitarray(dets)


def _empty_sci_state(norb: int, nelec: tuple[int, int]) -> SCIState:
    """Build the placeholder state returned by ranks other than the control process."""
    return SCIState(
        amplitudes=np.empty((0, 0), dtype=np.float64),
        ci_strs_a=np.array([], dtype=np.int64),
        ci_strs_b=np.array([], dtype=np.int64),
        norb=norb,
        nelec=nelec,
    )


def _sbd_dets_to_ci_strings(
    dets: list[list[int]], norb: int, backend,
    bit_length: int = SBD_DEFAULT_BIT_LENGTH,
) -> np.ndarray:
    """Convert SBD determinants to CI strings (integers)."""
    ci_strings = []
    for det in dets:
        binary_str = backend.makestring(det, bit_length, norb)
        ci_str = int(binary_str, 2)
        ci_strings.append(ci_str)
    return np.array(ci_strings, dtype=np.int64)


def _create_sbd_config(config_dict: dict | None = None, backend=None, device_config=None):
    """Create SBD configuration object from dictionary."""
    if backend is None:
        backend = _resolve_backend(device_config)

    sbd_data = backend.TPB_SBD()

    # Defaults
    sbd_data.method = 0  # Davidson
    sbd_data.max_it = 100
    sbd_data.max_nb = 50
    sbd_data.eps = 1e-8
    sbd_data.max_time = 3600
    sbd_data.init = 0
    sbd_data.do_shuffle = 0
    sbd_data.do_rdm = 0
    # SBD's carryover is off by default because the default path does not consume it:
    # the SQD loop selects its own determinants from the amplitudes we return, so
    # computing SBD's selection would be pure work -- for carryover_type=2 that includes
    # building singles-extended determinant lists.
    #
    # It is consumed when the caller asks for it. Setting carryover_type to 1, 2 or 3
    # through sbd_config makes _solve_sci_core report SBD's selection as
    # SCIResult.carryover and skip the eigenvector entirely, which is what a large
    # distributed run wants: see the report_carryover argument. Leaving the default at 0
    # keeps that opt-in rather than silently paying for a selection most callers ignore.
    sbd_data.carryover_type = 0
    sbd_data.ratio = 0.1
    sbd_data.threshold = 1e-4
    sbd_data.bit_length = SBD_DEFAULT_BIT_LENGTH

    if config_dict:
        for key, value in config_dict.items():
            if hasattr(sbd_data, key):
                setattr(sbd_data, key, value)

    if device_config is not None:
        device_config.apply(sbd_data)

    return sbd_data


def create_sbd_solver(
    mpi_comm: MPI.Comm | None = None,
    sbd_config: dict | None = None,
    temp_dir: str | Path | None = None,
    clean_temp_dir: bool = True,
    device_config=None,
) -> Callable:
    """
    Create a configured SBD solver function for use with
    diagonalize_fermionic_hamiltonian.

    Example:
        >>> from functools import partial
        >>> sbd_solver = create_sbd_solver(sbd_config={"method": 0, "eps": 1e-10})
        >>> result = diagonalize_fermionic_hamiltonian(
        ...     hcore, eri, bit_array, sci_solver=sbd_solver, ...
        ... )
    """
    from functools import partial

    return partial(
        solve_sci_batch,
        mpi_comm=mpi_comm,
        sbd_config=sbd_config,
        temp_dir=temp_dir,
        clean_temp_dir=clean_temp_dir,
        device_config=device_config,
    )
