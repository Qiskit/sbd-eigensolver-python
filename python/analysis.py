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

"""Analysis of saved SBD wavefunctions: energy variance and Epstein-Nesbet PT2.

A diagonalization run with ``savename`` (``run_sbd_diag.py --savename``,
``run_gdb_diag.py --savename``) leaves the state on disk. This module reads it
back and estimates, for that fixed state :math:`|\\psi\\rangle` with energy
:math:`E` in its subspace :math:`V`,

.. math::

    \\sigma^2 = \\sum_{a \\notin V} |\\langle a|H|\\psi\\rangle|^2, \\qquad
    E_{\\mathrm{PT2}} = \\sum_{a \\notin V}
        \\frac{|\\langle a|H|\\psi\\rangle|^2}{E - H_{aa}}.

For an eigenvector of :math:`H` projected onto :math:`V`, :math:`\\sigma^2` is the
full energy variance :math:`\\langle H^2\\rangle - E^2`; both vanish when
:math:`V` is the whole space. :math:`E + E_{\\mathrm{PT2}}` estimates the exact
energy, and :math:`\\sigma^2 \\to 0` measures convergence.

Two methods:

* ``"stochastic"`` -- upstream SBD's statistical evaluator. It samples parent
  determinants by :math:`|c_i|` and returns a mean and standard error over batches.
  MPI-parallel over the communicator it is given, on CPU cores. The number of ranks
  need not match the run that wrote the files.
* ``"exact"`` -- deterministic, with PySCF, in a single process. Builds
  :math:`H|\\psi\\rangle` over the full CI space, so only for small systems.

Only restricted FCIDUMPs are supported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import comb

import numpy as np

__all__ = ["SavedState", "VarianceResult", "load_state", "state_energy", "variance"]


@dataclass
class SavedState:
    """One rank's share of a saved SBD wavefunction, as full interleaved determinants.

    Attributes:
        det: ``(n, words)`` array of packed determinants, bit ``2p`` alpha orbital
            ``p`` and bit ``2p+1`` beta, in SBD's word packing.
        coef: The ``n`` amplitudes, in SBD's sign convention (not normalized).
        bit_length: Bits per packed word, as the run that wrote the state used.
        kind: ``"tpb"`` or ``"gdb"``: the solver that wrote the files.
        files: The files read, all ranks together.
    """

    det: np.ndarray
    coef: np.ndarray
    bit_length: int
    kind: str
    files: list = field(default_factory=list)


@dataclass
class VarianceResult:
    """Estimated energy variance and Epstein-Nesbet PT2 of a state.

    Attributes:
        variance: External variance :math:`\\sigma^2` (Hartree^2).
        variance_error: Standard error of ``variance`` over batches; 0 for ``exact``,
            NaN for a single stochastic batch.
        pt2: Epstein-Nesbet second-order energy correction (Hartree).
        pt2_error: Standard error of ``pt2``, as for ``variance_error``.
        energy: The reference energy the PT2 denominators use (Hartree).
        method: ``"exact"`` or ``"stochastic"``.
        dimension: Determinants in the state's subspace.
        batches: Per-batch ``variance``/``pt2``/``seconds`` (stochastic only).
        settings: The options the estimate was made with.
    """

    variance: float
    variance_error: float
    pt2: float
    pt2_error: float
    energy: float
    method: str
    dimension: int
    batches: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    @property
    def energy_pt2(self) -> float:
        """The PT2-corrected energy :math:`E + E_{\\mathrm{PT2}}`."""
        return self.energy + self.pt2


# --------------------------------------------------------------------------------
# Reading saved states
# --------------------------------------------------------------------------------


def _shard_files(prefix):
    """The files one ``savename`` run wrote under ``prefix``, in shard order.

    GDB writes ``P000000.bin``, ``P000001.bin``, ...; TPB ``P000000``, ``P000001``, ...
    A prefix holding both kinds, or numbers with a gap, did not come from one run, so
    those are refused rather than guessed at.
    """
    import glob
    import re

    pattern = re.compile(re.escape(os.path.basename(prefix)) + r"(\d{6})(\.bin)?$")
    found = {".bin": {}, "": {}}
    for path in glob.glob(glob.escape(prefix) + "[0-9]" * 6 + "*"):
        match = pattern.match(os.path.basename(path))
        if match:
            found[match.group(2) or ""][int(match.group(1))] = path
    if found[".bin"] and found[""]:
        raise ValueError(
            f"{prefix!r} has both GDB files ({len(found['.bin'])} x {prefix}NNNNNN.bin) and "
            f"TPB files ({len(found[''])} x {prefix}NNNNNN): they are from different runs. "
            f"Keep one run's files per prefix (see examples/analysis/README.md).")
    kind_files = found[".bin"] or found[""]
    if not kind_files:
        raise FileNotFoundError(
            f"no saved state at {prefix!r}: expected {prefix}000000.bin (GDB) or "
            f"{prefix}000000 (TPB) and so on, as written with savename={prefix!r}")
    indices = sorted(kind_files)
    if indices != list(range(len(indices))):
        missing = sorted(set(range(indices[-1] + 1)) - set(indices))
        raise ValueError(
            f"{prefix!r}: shard numbers {indices[0]}..{indices[-1]} have gaps "
            f"(missing {missing[:5]}{'...' if len(missing) > 5 else ''}), so these are not "
            f"one run's files")
    return [kind_files[i] for i in indices]


def _interleave(alpha_words, beta_words, norb, bit_length):
    """Full packed determinants from packed alpha and beta strings, every pairing.

    Bit ``p`` of a half-string moves to bit ``2p`` (alpha) or ``2p + 1`` (beta).
    Each string is spread into its full-determinant bit positions once, and a pairing
    is then the OR of the two; rows are filled one alpha string at a time, so the only
    memory beyond the result is the spread strings.
    """
    alpha = _pack_bits(_spread(_unpack_bits(alpha_words, norb, bit_length), 0), bit_length)
    beta = _pack_bits(_spread(_unpack_bits(beta_words, norb, bit_length), 1), bit_length)
    na, nb = len(alpha), len(beta)
    full = np.empty((na * nb, alpha.shape[1]), dtype=np.uint64)
    for i in range(na):
        np.bitwise_or(alpha[i], beta, out=full[i * nb:(i + 1) * nb])
    return full


def _spread(bits, offset):
    """``(n, norb)`` 0/1 half-strings to ``(n, 2*norb)``, bit p at 2p + offset."""
    out = np.zeros((len(bits), 2 * bits.shape[1]), dtype=np.uint8)
    out[:, offset::2] = bits
    return out


def _unpack_bits(words, nbits, bit_length):
    """``(n, words)`` packed rows to an ``(n, nbits)`` 0/1 array, bit 0 first."""
    words = np.asarray(words, dtype=np.uint64)
    out = np.zeros((len(words), nbits), dtype=np.uint8)
    for p in range(nbits):
        out[:, p] = (words[:, p // bit_length] >> np.uint64(p % bit_length)) & np.uint64(1)
    return out


def _pack_bits(bits, bit_length):
    """``(n, nbits)`` 0/1 array to ``(n, words)`` packed rows, bit_length bits per word."""
    n, nbits = bits.shape
    words = np.zeros((n, -(-nbits // bit_length)), dtype=np.uint64)
    for p in range(nbits):
        words[:, p // bit_length] |= bits[:, p].astype(np.uint64) << np.uint64(p % bit_length)
    return words


def _read_rows(path, offset, dtype, count, width=1):
    """``count`` rows of ``width`` items of ``dtype`` from byte ``offset`` (one read)."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = np.fromfile(handle, dtype=dtype, count=count * width)
    if data.size != count * width:
        raise ValueError(f"{path}: truncated (needed {count} rows at byte {offset})")
    return data.reshape(count, width) if dtype == np.uint64 else data


def _read_gdb_rows(path, words, lo, hi):
    """Rows [lo, hi) of a GDB checkpoint: their determinants and amplitudes."""
    n = int(np.fromfile(path, dtype=np.uint64, count=1)[0])
    det = _read_rows(path, 16 + 8 * lo * words, np.uint64, hi - lo, words)
    coef = _read_rows(path, 16 + 8 * n * words + 8 * lo, np.float64, hi - lo)
    return det, coef


def _read_tpb_rows(path, words, na, nb, lo, hi, norb, bit_length):
    """Amplitudes [lo, hi) of a TPB block (row-major over alpha x beta), as full
    interleaved determinants. Reads only the alpha strings those rows use."""
    a_lo, a_hi = lo // nb, -(-hi // nb)
    alpha = _read_rows(path, 24 + 8 * a_lo * words, np.uint64, a_hi - a_lo, words)
    beta = _read_rows(path, 24 + 8 * na * words, np.uint64, nb, words)
    coef = _read_rows(path, 24 + 8 * (na + nb) * words + 8 * lo, np.float64, hi - lo)
    det = _interleave(alpha, beta, norb, bit_length)
    first = lo - a_lo * nb
    return det[first:first + (hi - lo)], coef


def _looks_like_gdb(path):
    raw = np.fromfile(path, dtype=np.uint64, count=2)
    if raw.size < 2:
        return False
    n, words = int(raw[0]), int(raw[1])
    return 0 < words < 64 and os.path.getsize(path) == 16 + 8 * n * words + 8 * n


def _file_headers(files, kind):
    """``(rows, words, n_alpha, n_beta)`` per file, from headers alone (rank 0 only)."""
    headers = []
    for path in files:
        if kind == "gdb":
            n, words = (int(x) for x in np.fromfile(path, dtype=np.uint64, count=2))
            expected = 16 + 8 * n * words + 8 * n
            headers.append((n, words, 0, 0))
        else:
            na, nb, words = (int(x) for x in np.fromfile(path, dtype=np.uint64, count=3))
            expected = 24 + 8 * (na + nb) * words + 8 * na * nb
            headers.append((na * nb, words, na, nb))
        if os.path.getsize(path) != expected:
            raise ValueError(f"{path}: {os.path.getsize(path)} bytes, but its header implies "
                             f"{expected}; the file is truncated or not an SBD state")
    return headers


def load_state(source, norb, bit_length=None, comm=None):
    """Read a saved SBD wavefunction, spread over the ranks of ``comm``.

    Accepts a ``savename`` prefix (TPB or GDB shard files) or one GDB ``.bin`` file.
    The files must all come from one run. That is checked on every rank together, at
    the cost of one pass over the data, and anything that does not fit raises rather
    than being combined: TPB and GDB files under one prefix, gaps in the shard
    numbers, a size that does not match a file's header, files with different word
    counts, a GDB file not strictly increasing (so with a repeat), and a total norm
    that is not 1 -- SBD saves normalized states, so two runs' files sum to about 2.
    A determinant repeated *across* files is caught by :func:`state_energy` and
    :func:`variance`, which sort the state across ranks anyway.

    The state is spread evenly over the ranks, whatever the number of files, so the
    rank count need not match the run that wrote it.

    Args:
        source: The ``savename`` prefix, or one GDB checkpoint file.
        norb: Spatial orbitals (``NORB`` of the FCIDUMP).
        bit_length: Bits per word the diagonalization used (its config's
            ``bit_length``). Required: the files do not record it.
        comm: MPI communicator; default ``MPI.COMM_WORLD``.

    Returns:
        This rank's :class:`SavedState`.
    """
    from mpi4py import MPI

    comm = comm or MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if bit_length is None:
        raise ValueError(
            "bit_length is required: pass the bit_length the diagonalization used. The "
            "files record words per determinant, which does not determine it.")
    words_needed = -(-2 * norb // bit_length)

    # Rank 0 lists the files and reads their headers; everyone gets the result, or the
    # same error.
    plan = None
    if rank == 0:
        try:
            if os.path.isfile(source) and _looks_like_gdb(source):
                files = [source]
            else:
                files = _shard_files(source)
            kind = "gdb" if files[0].endswith(".bin") else "tpb"
            headers = _file_headers(files, kind)
            words = {h[1] for h in headers}
            if len(words) != 1:
                raise ValueError(f"{source!r}: files disagree on words per determinant "
                                 f"({sorted(words)}), so they are not one run's")
            if kind == "gdb" and words != {words_needed}:
                raise ValueError(f"{files[0]}: {words.pop()} word(s) per determinant, but "
                                 f"NORB={norb} at bit_length={bit_length} implies "
                                 f"{words_needed}; pass the bit_length the run used")
            plan = (files, kind, headers)
        except (OSError, ValueError) as exc:
            plan = exc
    plan = comm.bcast(plan, root=0)
    if isinstance(plan, Exception):
        raise plan
    files, kind, headers = plan

    # Rank r takes rows [r n / P, (r+1) n / P) of the files taken in order, so each rank
    # holds 1/P of the state however many files there are and reads only its own bytes.
    # The rows are in file order, not a global sort order: GDB's default determinant
    # distribution interleaves its shards' ranges.
    rows = np.array([h[0] for h in headers], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(rows)])
    total = int(starts[-1])
    my_lo, my_hi = total * rank // size, total * (rank + 1) // size
    dets, coefs = [], []
    for i, path in enumerate(files):
        lo, hi = max(my_lo, starts[i]) - starts[i], min(my_hi, starts[i + 1]) - starts[i]
        if lo >= hi:
            continue
        words = headers[i][1]
        if kind == "gdb":
            det, coef = _read_gdb_rows(path, words, int(lo), int(hi))
        else:
            det, coef = _read_tpb_rows(path, words, headers[i][2], headers[i][3],
                                       int(lo), int(hi), norb, bit_length)
        dets.append(det)
        coefs.append(coef)
    det = np.concatenate(dets) if dets else np.zeros((0, words_needed), dtype=np.uint64)
    coef = np.concatenate(coefs) if coefs else np.zeros(0)
    if kind == "tpb" and len(det) and det.shape[1] != words_needed:
        problem = f"inconsistent with NORB={norb}, bit_length={bit_length}"
    else:
        # GDB writes each shard sorted with no repeats (its determinant distribution
        # may interleave the shards' ranges, so only within a file is order defined).
        # This rank's rows come from at most a few files, checked file by file.
        problem = "".join(_check_order(d, kind) for d in dets) if kind == "gdb" else ""
    _raise_if_any(comm, problem, source)

    norm2 = comm.allreduce(float(np.dot(coef, coef)))
    if not abs(norm2 - 1.0) < 1e-6:
        raise ValueError(
            f"{source!r}: the amplitudes' norm-squared over all {len(files)} file(s) is "
            f"{norm2:.6f}, not 1. SBD saves normalized states, so these files are not one "
            f"run's (or are not SBD states): keep one run's files per prefix.")
    return SavedState(np.ascontiguousarray(det), np.ascontiguousarray(coef), bit_length, kind,
                      list(files))


def _check_order(det, kind):
    """'' if ``det`` (one rank's GDB files, concatenated) is strictly increasing in SBD's
    order, else the problem. Strictly increasing means no repeats, in one pass."""
    if len(det) < 2:
        return ""
    a, b = det[:-1], det[1:]
    # strictly less, most significant word first, vectorized over rows
    less = np.zeros(len(a), dtype=bool)
    decided = np.zeros(len(a), dtype=bool)
    for w in range(det.shape[1] - 1, -1, -1):
        lt = (a[:, w] < b[:, w]) & ~decided
        gt = (a[:, w] > b[:, w]) & ~decided
        less |= lt
        decided |= lt | gt
    bad = np.flatnonzero(~less)
    if bad.size:
        repeat = bool(np.all(a[bad[0]] == b[bad[0]]))
        return ("a determinant appears twice" if repeat else
                "determinants are out of order") + (
                    ", so the files are not one GDB run's shards")
    return ""


def _raise_if_any(comm, problem, source, force=None):
    """Raise on every rank if any rank found a problem (no rank left in a collective)."""
    message = force or problem
    messages = comm.allgather(message)
    first = next((m for m in messages if m), "")
    if first:
        raise ValueError(f"{source!r}: {first}" if not first.startswith(repr(source)) else first)


# --------------------------------------------------------------------------------
# The estimate
# --------------------------------------------------------------------------------


def _fcidump_header(fcidump):
    header = fcidump.header
    norb = int(header["NORB"])
    nelec = int(header["NELEC"])
    ms2 = int(header.get("MS2", 0))
    n_alpha = (nelec + ms2) // 2
    return norb, (n_alpha, nelec - n_alpha)


def _evaluator_backend(device):
    """A backend module that has the evaluator; it runs on CPU whichever is chosen.

    The Thrust build omits it (upstream includes it only without SBD_THRUST). Prefer a
    module already loaded, so a session that diagonalized on 'gpu-omp' does not load
    _core_cpu next to it, which would demote OpenMP offload to the host for the rest
    of the process. Otherwise the CPU module, which coexists with Thrust.
    """
    from . import available_backends, get_backend, loaded_backends

    if device is not None:
        backend = get_backend(device)
        if not getattr(backend, "__sbd_has_evaluator__", False):
            raise ValueError(
                f"the {device!r} backend has no evaluator (upstream builds it without "
                f"SBD_THRUST only). It runs on CPU in any case: omit device, or use "
                f"'cpu' or 'gpu-omp'.")
        return backend
    loaded = loaded_backends()
    for name in loaded:
        backend = get_backend(name)
        if getattr(backend, "__sbd_has_evaluator__", False):
            return backend
    # Nothing loaded has it. Only ever add the CPU module: it coexists with Thrust,
    # whereas loading the OpenMP-offload module next to another backend is not a
    # combination this package supports.
    if "cpu" in available_backends():
        return get_backend("cpu")
    if not loaded and "gpu-omp" in available_backends():
        return get_backend("gpu-omp")
    raise RuntimeError(
        "method='stochastic' needs SBD's evaluator, which the CPU and OpenMP-offload "
        "backends have and the Thrust backend does not"
        + (f"; this process has loaded {loaded} and no CPU backend is built" if loaded else "")
        + ". Build the CPU backend too (SBD_BUILD_BACKEND including cpu, the default), "
          "run the analysis as a separate job, or use method='exact' for a small system.")


def state_energy(source, fcidump, *, norb=None, bit_length=None, comm=None, device=None):
    """The energy of a saved state, :math:`\\langle\\psi|H|\\psi\\rangle`, computed by SBD.

    Loads the state into its own basis with zero Davidson iterations, so it is the
    state's energy, not a re-optimized one. Use it as the PT2 reference, and to check
    that a state and an FCIDUMP belong together: it should equal the energy the run
    that saved the state reported.

    The state stays distributed over the ranks; a temporary copy of it is written and
    removed afterwards. A determinant found in more than one file raises.

    Args:
        source, norb, bit_length, comm: As for :func:`load_state`.
        fcidump: The FCIDump the state was diagonalized with.
        device: Backend for the evaluation; default the one :func:`sbd.gdb_diag` uses.

    Returns:
        The energy in Hartree (including the FCIDUMP constant), on every rank.
    """
    import shutil
    import tempfile

    from mpi4py import MPI

    from . import GDB_SBD, gdb_diag

    comm = comm or MPI.COMM_WORLD
    norb = norb or _fcidump_header(fcidump)[0]
    state = source if isinstance(source, SavedState) else load_state(source, norb, bit_length, comm)
    det, coef = _sorted_unique(state, comm)
    workdir = None
    try:
        # gdb_diag needs its basis sorted globally across b_comm; write the sorted
        # shards as this run's checkpoint and load them back with zero iterations.
        workdir = comm.bcast(tempfile.mkdtemp(prefix="sbd-energy-")
                             if comm.Get_rank() == 0 else None, root=0)
        prefix = os.path.join(workdir, "state-")
        _write_gdb(f"{prefix}{comm.Get_rank():06d}.bin", det, coef)
        del coef
        comm.Barrier()
        config = GDB_SBD(device)
        config.max_it, config.max_nb, config.bit_length = 0, 2, state.bit_length
        config.b_comm_size = comm.Get_size()
        energy = gdb_diag(fcidump, det, config, loadname=prefix, device=device,
                          determinant_distribution="input")["energy"]
    finally:
        comm.Barrier()
        if workdir and comm.Get_rank() == 0:
            shutil.rmtree(workdir, ignore_errors=True)
    return float(energy)


def _sorted_unique(state, comm):
    """The state sorted globally across ranks, refusing a determinant seen twice.

    After a global sort a repeat is either adjacent on one rank or split across two
    neighbouring ranks' edges, so checking both costs one pass and one exchange of a
    single row per rank -- no gather, no hash table.
    """
    det, coef = _distributed_sort(state.det, state.coef, comm)
    local = bool(len(det) > 1 and np.any(np.all(det[1:] == det[:-1], axis=1)))
    edges = comm.allgather((det[0].copy(), det[-1].copy()) if len(det) else None)
    rank = comm.Get_rank()
    previous = next((e[1] for e in reversed(edges[:rank]) if e is not None), None)
    across = bool(len(det) and previous is not None and np.array_equal(previous, det[0]))
    if comm.allreduce(int(local or across)):
        raise ValueError(
            "the saved state has a determinant in more than one file, so its files are "
            "not one run's: keep one run's files per prefix")
    return det, coef


def _distributed_sort(det, coef, comm):
    """Sort determinants globally in SBD's order across ranks; rank r gets block r.

    A sample sort: each rank sorts its own rows, the ranks agree on P-1 splitters from
    a small regular sample, and one Alltoallv moves each row to its owner. Memory per
    rank stays O(n/P) plus one exchange buffer; nothing is gathered whole.
    """
    size = comm.Get_size()
    # lexsort's last key is primary, so det.T sorts by the last (most significant) word
    order = np.lexsort(det.T) if len(det) else np.zeros(0, dtype=np.intp)
    det, coef = det[order], coef[order]
    words = det.shape[1] if det.ndim == 2 else 1
    if size == 1:
        return np.ascontiguousarray(det), np.ascontiguousarray(coef)
    # regular sample of each rank's sorted rows, then global splitters
    k = min(len(det), 4 * size)
    sample = det[np.linspace(0, len(det) - 1, k).astype(int)] if k else np.zeros((0, words), np.uint64)
    samples = np.concatenate(comm.allgather(sample)) if comm.allreduce(k) else sample
    samples = samples[np.lexsort(samples.T)] if len(samples) else samples
    splitters = samples[np.linspace(0, len(samples) - 1, size + 1).astype(int)[1:-1]] \
        if len(samples) else np.zeros((0, words), np.uint64)
    # destination of each local row: number of splitters strictly below it
    dest = np.zeros(len(det), dtype=np.int64)
    for s in splitters:
        dest += _rows_greater(det, s)
    counts = np.bincount(dest, minlength=size)
    det_out = _alltoallv(comm, det.reshape(-1), counts * words, np.uint64).reshape(-1, words)
    coef_out = _alltoallv(comm, coef, counts, np.float64)
    order = np.lexsort(det_out.T) if len(det_out) else np.zeros(0, dtype=np.intp)
    return np.ascontiguousarray(det_out[order]), np.ascontiguousarray(coef_out[order])


def _rows_greater(det, row):
    """Boolean per row of ``det``: greater than ``row`` in SBD's order (last word first)."""
    greater = np.zeros(len(det), dtype=bool)
    decided = np.zeros(len(det), dtype=bool)
    for w in range(det.shape[1] - 1, -1, -1):
        gt = (det[:, w] > row[w]) & ~decided
        lt = (det[:, w] < row[w]) & ~decided
        greater |= gt
        decided |= gt | lt
    return greater


def _alltoallv(comm, send, counts, dtype):
    """Alltoallv of a contiguous array already grouped by destination rank."""
    from mpi4py import MPI

    send = np.ascontiguousarray(send, dtype=dtype)
    counts = np.asarray(counts, dtype=np.int64)
    recv_counts = np.empty_like(counts)
    comm.Alltoall(counts, recv_counts)
    sdispl = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rdispl = np.concatenate([[0], np.cumsum(recv_counts)[:-1]])
    recv = np.empty(int(recv_counts.sum()), dtype=dtype)
    mpitype = MPI.UINT64_T if dtype == np.uint64 else MPI.DOUBLE
    comm.Alltoallv([send, counts, sdispl, mpitype], [recv, recv_counts, rdispl, mpitype])
    return recv


def _write_gdb(path, det, coef):
    with open(path, "wb") as handle:
        handle.write(np.array([len(det), det.shape[1]], dtype=np.uint64).tobytes())
        handle.write(np.ascontiguousarray(det, dtype=np.uint64).tobytes())
        handle.write(np.ascontiguousarray(coef, dtype=np.float64).tobytes())


def _full_space_size(norb, nelec):
    return comb(norb, nelec[0]) * comb(norb, nelec[1])


def variance(source, fcidump, energy=None, *, method="auto", norb=None, bit_length=None,
             samples=20000, batches=10, seed=1729, cutoff=0.0, comm=None,
             exact_limit=5_000_000, device=None):
    """Energy variance and Epstein-Nesbet PT2 of a saved SBD wavefunction.

    Args:
        source: A ``savename`` prefix or GDB checkpoint (see :func:`load_state`),
            or a :class:`SavedState` already loaded on every rank.
        fcidump: The FCIDump the state was diagonalized with (``sbd.LoadFCIDump``).
        energy: The state's energy (Hartree, including the FCIDUMP constant), the
            PT2 reference. Default: computed from the state by :func:`state_energy`.
            If given, it must be this state's energy, not the exact one. The exact
            method always computes it from the vector, and reports that.
        method: ``"exact"``, ``"stochastic"``, or ``"auto"``: exact when the full CI
            space has at most ``exact_limit`` determinants, otherwise stochastic.
        norb, bit_length: As for :func:`load_state`; ``norb`` defaults to the
            FCIDUMP's ``NORB``.
        samples: Parent draws per stochastic batch.
        batches: Stochastic batches; the error bar is their standard error, so use at
            least a few.
        seed: Sampling seed. Results are reproducible for the same seed, rank count
            and batches, not across rank counts.
        cutoff: Heat-bath screening of parent-child contributions in the stochastic
            method. 0 (the default) is unbiased. A cutoff speeds it up but biases both
            results downward by an amount that grows as the state converges, so check
            any cutoff against a smaller one before trusting it.
        comm: MPI communicator; default ``MPI.COMM_WORLD``. The stochastic method runs
            on all its ranks; the exact one on rank 0, broadcasting the result.
        exact_limit: Largest full CI space ``"auto"`` treats exactly.
        device: Backend for the stochastic method's C++ call. Default: one already
            loaded that has the evaluator, else the CPU one. The evaluator always runs
            on CPU, and the Thrust build does not include it.

    Returns:
        A :class:`VarianceResult`, identical on every rank.
    """
    from mpi4py import MPI

    comm = comm or MPI.COMM_WORLD
    fd_norb, nelec = _fcidump_header(fcidump)
    norb = norb or fd_norb
    if method == "auto":
        method = "exact" if _full_space_size(norb, nelec) <= exact_limit else "stochastic"
    if method not in ("exact", "stochastic"):
        raise ValueError(f"method must be 'auto', 'exact' or 'stochastic', got {method!r}")
    state = source if isinstance(source, SavedState) else load_state(source, norb, bit_length, comm)
    if energy is None and method == "stochastic":
        energy = state_energy(state, fcidump, norb=norb, comm=comm)
    dimension = int(comm.allreduce(len(state.coef)))
    settings = {"method": method, "samples": samples, "batches": batches, "seed": seed,
                "cutoff": cutoff, "ranks": comm.Get_size(), "kind": state.kind,
                "bit_length": state.bit_length}

    if method == "exact":
        result = None
        gathered = comm.gather((state.det, state.coef), root=0)
        if comm.Get_rank() == 0:
            det = np.concatenate([d for d, _ in gathered])
            coef = np.concatenate([c for _, c in gathered])
            var, pt2, e_state = _exact(fcidump, det, coef, norb, nelec, state.bit_length)
            if energy is not None:
                settings["energy_given"] = float(energy)
            result = VarianceResult(var, 0.0, pt2, 0.0, e_state, "exact", dimension,
                                    settings=settings)
        return comm.bcast(result, root=0)

    backend = _evaluator_backend(device)
    out = backend.external_observables(
        comm, fcidump, state.det, state.coef, state.bit_length, float(energy),
        sample_count=samples, batch_count=batches, seed=seed, heatbath_cutoff=cutoff)
    v, p = np.asarray(out["variance"]), np.asarray(out["pt2"])
    k = len(v)
    stderr = (lambda x: float(x.std(ddof=1) / np.sqrt(k))) if k > 1 else (lambda x: float("nan"))
    return VarianceResult(float(v.mean()), stderr(v), float(p.mean()), stderr(p), float(energy),
                          "stochastic", dimension,
                          batches={"variance": v.tolist(), "pt2": p.tolist(),
                                   "seconds": list(out["seconds"])},
                          settings=settings)


def _alpha_first_phase(alpha_bits, beta_bits):
    """(-1)^(sum over occupied beta p of occupied alpha above p), per row (PR #62's rule)."""
    above = np.cumsum(alpha_bits[:, ::-1], axis=1)[:, ::-1] - alpha_bits
    return 1 - 2 * (np.sum(above * beta_bits, axis=1) & 1)


def _exact(fcidump, det, coef, norb, nelec, bit_length):
    """External variance and EN-PT2 by PySCF over the full CI space (one process)."""
    try:
        from pyscf import ao2mo
        from pyscf.fci import cistring, direct_spin1
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError("method='exact' needs PySCF (pip install pyscf)") from exc
    h1, eri, ecore = _integrals(fcidump, norb)
    eri = ao2mo.restore(1, eri, norb)
    bits = _unpack_bits(det, 2 * norb, bit_length).astype(np.int64)
    alpha, beta = bits[:, 0::2], bits[:, 1::2]
    if np.any(alpha.sum(axis=1) != nelec[0]) or np.any(beta.sum(axis=1) != nelec[1]):
        raise ValueError(f"the state has determinants without {nelec} electrons")
    weights = 1 << np.arange(norb, dtype=np.int64)
    ia = cistring.strs2addr(norb, nelec[0], alpha @ weights)
    ib = cistring.strs2addr(norb, nelec[1], beta @ weights)
    ci = np.zeros((cistring.num_strings(norb, nelec[0]), cistring.num_strings(norb, nelec[1])))
    ci[ia, ib] = coef * _alpha_first_phase(alpha, beta)
    norm = np.linalg.norm(ci)
    if norm == 0:
        raise ValueError("the state has zero norm")
    ci /= norm
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)
    hc = direct_spin1.contract_2e(h2e, ci, norb, nelec)
    hdiag = direct_spin1.make_hdiag(h1, eri, norb, nelec).reshape(ci.shape)
    e_el = float(np.vdot(ci, hc))
    outside = np.ones(ci.shape, dtype=bool)
    outside[ia, ib] = False
    hc_out, hd_out = hc[outside], hdiag[outside]
    variance_ = float(np.dot(hc_out, hc_out))
    pt2 = float(np.sum(hc_out ** 2 / (e_el - hd_out)))
    return variance_, pt2, e_el + ecore


def _integrals(fcidump, norb):
    """(h1, eri in 8-fold packed form, ecore) from an sbd FCIDump object."""
    from pyscf import ao2mo

    h1 = np.zeros((norb, norb))
    eri = np.zeros((norb,) * 4)
    ecore = 0.0
    for value, i, j, k, l in fcidump.integrals:
        if max(i, j, k, l) > norb:
            raise ValueError("method='exact' supports restricted FCIDUMPs only")
        if i == j == k == l == 0:
            ecore = value
        elif k == 0 and l == 0:
            h1[i - 1, j - 1] = h1[j - 1, i - 1] = value
        else:
            p, q, r, s = i - 1, j - 1, k - 1, l - 1
            for a, b, c, d in ((p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                               (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)):
                eri[a, b, c, d] = value
    return h1, ao2mo.restore(8, eri, norb), ecore
