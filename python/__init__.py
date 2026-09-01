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
SBD (Selected Basis Diagonalization) Python Bindings

This package provides Python bindings for the SBD library.

Usage:
    import sbd
    results = sbd.tpb_diag_from_files(fcidump, adets, config)

    # Explicit init is optional — auto-initialized on first use
    sbd.init(device='gpu')              # set default device explicitly

Device switching (CPU/GPU) within the same process:
    result_cpu = sbd.tpb_diag(..., device='cpu')
    result_gpu = sbd.tpb_diag(..., device='gpu')
"""

import os
import subprocess

__version__ = "1.6.1"

# ---------------------------------------------------------------------------
# Backend registry — eagerly load all available backends at import time.
# All compiled backends can coexist: separate .so files with separate
# pybind11 namespaces, no global C++ state conflicts.
# ---------------------------------------------------------------------------
_backends = {}

# Why a backend is absent, keyed by device name. A .so that was built but
# cannot be loaded -- missing libmpi at run time, wrong architecture, an
# OpenMP runtime clash -- is otherwise indistinguishable from one that was
# never built, and 'Available: []' with no explanation is a dead end.
_backend_errors = {}

# Device-string aliases. Keys are user-facing strings; values are the
# canonical key in _backends. e.g. 'gpu-omp' resolves to whatever
# OMP-offload backend is built.
_device_aliases = {}


def _try_load(module_name, primary_device, *aliases):
    import os
    from importlib import import_module
    try:
        mod = import_module(f'.{module_name}', package=__name__)
    except Exception as exc:
        # Deliberately broad: a compiled extension can fail with OSError
        # (missing shared library) as readily as ImportError.
        here = os.path.dirname(__file__)
        built = any(n.startswith(module_name + '.') and n.endswith('.so')
                    for n in os.listdir(here)) if os.path.isdir(here) else False
        _backend_errors[primary_device] = (
            f"{'built but failed to load' if built else 'not built'}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False
    _backends[primary_device] = mod
    for a in aliases:
        _device_aliases[a] = primary_device
    return True


_try_load('_core_cpu',             'cpu')
_try_load('_core_gpu_thrust',      'gpu', 'gpu-thrust', 'gpu-nvidia', 'cuda')
_try_load('_core_gpu_omp_offload', 'gpu-omp', 'gpu-omp-offload',
                                   'gpu-nvhpc-omp', 'gpu-nvidia-omp')

# The OMP-offload backend links NVHPC's libnvomp; CPU and Thrust link
# libgomp/libomp. Loading both pulls two OpenMP runtimes into one process,
# and the observed result is not a crash but a silent demotion: offload
# regions quietly run on the host while the device query still reports a
# GPU, so a run looks GPU-accelerated and is not. Since every _core_*.so in
# this directory is loaded eagerly, that happens whenever the three files
# share a directory -- separate environments do not help if they share a
# checkout. Warn loudly rather than let a wrong-looking-right run proceed.
_omp_offload_conflict = (
    'gpu-omp' in _backends and bool({'cpu', 'gpu'} & set(_backends))
)
if _omp_offload_conflict:
    import warnings as _warnings
    _warnings.warn(
        "sbd: incompatible backends loaded together: "
        f"{', '.join(sorted(_backends))}. The OMP-offload backend "
        "(_core_gpu_omp_offload) links a different OpenMP runtime than the "
        "CPU/Thrust backends, and co-loading them silently demotes offload "
        "to the host -- device queries still report a GPU. Keep "
        "_core_gpu_omp_offload.so in a directory of its own (its own "
        "checkout, not merely its own env) and rebuild.",
        RuntimeWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Global session state
# ---------------------------------------------------------------------------
_default_device = None   # set by init(), can be overridden per-call
_comm_backend = None     # 'mpi'
_comm_module = None      # mpi4py.MPI module
_global_comm = None      # MPI communicator
_initialized = False

# Cache GPU detection to avoid repeated subprocess calls
_gpu_check_cache = None


def _gpu_available():
    """Check if GPU is available via nvidia-smi (cached)."""
    global _gpu_check_cache
    if _gpu_check_cache is not None:
        return _gpu_check_cache
    try:
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, timeout=2
        )
        _gpu_check_cache = result.returncode == 0
    except Exception:
        _gpu_check_cache = False
    return _gpu_check_cache


def _resolve_device(device):
    """Resolve 'auto' or an alias to a concrete backend key.

    Auto-resolution prefers Thrust GPU (`'gpu'`) over OMP-offload
    (`'gpu-omp'`) when both are built and a GPU is present, since the
    Thrust path is the long-validated default. In practice OMP-offload
    is built into its own venv/install (different OpenMP runtime, can't
    co-load with Thrust/CPU), so this branch only fires when an OMP-only
    install is in use.
    """
    if device == 'auto':
        if 'gpu' in _backends and _gpu_available():
            return 'gpu'
        if 'gpu-omp' in _backends and _gpu_available():
            return 'gpu-omp'
        return 'cpu'
    return _device_aliases.get(device, device)


def init(device='cpu', comm_backend='mpi'):
    """
    Initialize SBD with MPI and set the default compute device.

    Calling ``init()`` explicitly is **optional** — SBD auto-initializes on
    first use with ``device='cpu'`` and ``comm_backend='mpi'``.  Call it
    explicitly only when you need GPU or want to control startup timing.

    The device can be overridden per-call via the ``device`` parameter on
    ``tpb_diag()``, ``tpb_diag_from_files()``, and ``get_backend()``.

    Args:
        device: Default compute device — 'cpu', 'gpu', 'gpu-omp', or 'auto'.
                Aliases: 'gpu-thrust' / 'gpu-nvidia' / 'cuda' (= 'gpu');
                         'gpu-omp-offload' / 'gpu-nvhpc-omp' / 'gpu-nvidia-omp' (= 'gpu-omp').
        comm_backend: Communication backend — 'mpi'.

    Raises:
        RuntimeError: If MPI is not available or no backends are compiled.
    """
    global _default_device, _comm_backend, _comm_module, _global_comm, _initialized

    if _initialized:
        return  # already initialized — silently no-op

    if not _backends:
        raise RuntimeError(
            "No SBD backends available. Build with:\n"
            "  pip install -e . --no-build-isolation                 (auto: CPU + Thrust GPU)\n"
            "  SBD_BUILD_BACKEND=gpu pip install -e . --no-build-isolation              (Thrust GPU only)\n"
            "  SBD_BUILD_BACKEND=gpu_omp_offload pip install -e . --no-build-isolation  (OMP-offload only)"
        )

    # MPI setup
    if comm_backend == 'mpi':
        try:
            from mpi4py import MPI
            _comm_module = MPI
            _global_comm = MPI.COMM_WORLD
            _comm_backend = 'mpi'
        except ImportError:
            raise RuntimeError(
                "MPI backend requires mpi4py. Install with: pip install mpi4py"
            )
    else:
        raise ValueError(f"Unknown comm_backend: '{comm_backend}'. Supported: 'mpi'")

    # Resolve default device
    resolved = _resolve_device(device)
    if resolved not in _backends:
        available = list(_backends.keys())
        raise RuntimeError(
            f"Device '{resolved}' requested but backend not available. "
            f"Available: {available}"
        )
    _default_device = resolved
    _initialized = True


def finalize():
    """
    Finalize SBD and reset session state.

    Synchronizes GPU (if used) but does NOT call MPI_Finalize — mpi4py
    handles MPI lifecycle automatically.

    After finalize(), init() can be called again.
    """
    global _default_device, _comm_backend, _comm_module, _global_comm, _initialized

    # Synchronize GPU backends
    for name, backend in _backends.items():
        if name == 'gpu' and hasattr(backend, 'cleanup_device'):
            try:
                backend.cleanup_device()
            except Exception:
                pass

    _default_device = None
    _comm_backend = None
    _comm_module = None
    _global_comm = None
    _initialized = False


def is_initialized():
    """Check if SBD has been initialized."""
    return _initialized


# ---------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------

def get_backend(device=None):
    """
    Get the backend module for the given device.

    Auto-initializes SBD if needed. Passing ``device`` overrides the
    default — this is how you switch between CPU and GPU within the
    same process.

    Args:
        device: 'cpu', 'gpu', 'auto', or None (use default).

    Returns:
        The pybind11 backend module (_core_cpu, _core_gpu_thrust, or
        _core_gpu_omp_offload).
    """
    if device is None:
        device = _default_device or 'auto'
    device = _resolve_device(device)
    if device not in _backends:
        available = list(_backends.keys())
        raise RuntimeError(
            f"Backend '{device}' not available. Available: {available}"
        )
    return _backends[device]


def _ensure_initialized():
    """Auto-initialize with defaults if init() hasn't been called yet."""
    if not _initialized:
        init()


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_device():
    """Get the default compute device name."""
    _ensure_initialized()
    return _default_device


def get_comm_backend():
    """Get the communication backend name."""
    _ensure_initialized()
    return _comm_backend


def get_rank():
    """Get MPI rank of current process."""
    _ensure_initialized()
    return _global_comm.Get_rank()


def get_world_size():
    """Get total number of MPI processes."""
    _ensure_initialized()
    return _global_comm.Get_size()


def get_comm():
    """Get the MPI communicator."""
    _ensure_initialized()
    return _global_comm


def barrier():
    """MPI barrier — synchronize all processes."""
    _ensure_initialized()
    _global_comm.Barrier()


# ---------------------------------------------------------------------------
# Wrapper functions — forward to the selected backend
# ---------------------------------------------------------------------------

def TPB_SBD(device=None):
    """Create TPB_SBD configuration object."""
    _ensure_initialized()
    return get_backend(device).TPB_SBD()


def GDB_SBD(device=None):
    """Create GDB_SBD configuration object."""
    _ensure_initialized()
    return get_backend(device).GDB_SBD()


def FCIDump(device=None):
    """Create FCIDump object."""
    _ensure_initialized()
    return get_backend(device).FCIDump()


def LoadFCIDump(filename, device=None):
    """Load FCIDUMP file."""
    _ensure_initialized()
    return get_backend(device).LoadFCIDump(filename)


def LoadAlphaDets(filename, bit_length, total_bit_length, device=None):
    """Load alpha determinants from file."""
    _ensure_initialized()
    return get_backend(device).LoadAlphaDets(filename, bit_length, total_bit_length)


def makestring(config, bit_length, total_bit_length, device=None):
    """Convert determinant to string representation."""
    _ensure_initialized()
    return get_backend(device).makestring(config, bit_length, total_bit_length)


def from_string(s, bit_length, total_bit_length, device=None):
    """Convert binary string to determinant format."""
    _ensure_initialized()
    return get_backend(device).from_string(s, bit_length, total_bit_length)


def sort_bitarray(dets, device=None):
    """Sort determinants into canonical order, removing duplicates.

    Diagonalization requires its determinant lists to be in this order.
    :func:`gdb_diag` sorts its own input, so this reproduces the order it
    works in.
    """
    _ensure_initialized()
    return get_backend(device).sort_bitarray(dets)


def tpb_diag_from_files(fcidumpfile, adetfile, sbd_data,
                        loadname="", savename="", device=None):
    """
    Perform TPB diagonalization from files.

    Args:
        fcidumpfile: Path to FCIDUMP file.
        adetfile: Path to alpha determinants file.
        sbd_data: TPB_SBD configuration object.
        loadname: Path to load initial wavefunction (optional).
        savename: Path to save final wavefunction (optional).
        device: Override device ('cpu', 'gpu', or None for default).

    Returns:
        dict with keys: energy, density, carryover_adet, carryover_bdet,
        one_p_rdm, two_p_rdm.
    """
    _ensure_initialized()
    backend = get_backend(device)
    return backend.tpb_diag_from_files(
        _global_comm, sbd_data, fcidumpfile, adetfile, loadname, savename
    )


def tpb_diag(fcidump, adet, bdet, sbd_data,
             loadname="", savename="", device=None):
    """
    Perform TPB diagonalization with data structures.

    Args:
        fcidump: FCIDump object.
        adet: Alpha determinants.
        bdet: Beta determinants.
        sbd_data: TPB_SBD configuration object.
        loadname: Path to load initial wavefunction (optional).
        savename: Path to save final wavefunction (optional).
        device: Override device ('cpu', 'gpu', or None for default).

    Returns:
        dict with keys: energy, density, carryover_adet, carryover_bdet,
        one_p_rdm, two_p_rdm.
    """
    _ensure_initialized()
    backend = get_backend(device)
    return backend.tpb_diag(
        _global_comm, sbd_data, fcidump, adet, bdet, loadname, savename
    )


def gdb_diag(fcidump, det, sbd_data,
             loadname="", savename="", device=None):
    """
    Perform GDB diagonalization over an explicit list of determinants.

    Unlike TPB, which spans the subspace with the Cartesian product of an alpha
    and a beta determinant list, GDB spans it with the given determinants
    themselves, so an arbitrary sparse subspace can be diagonalized.

    Each determinant is a ``2 * norb``-bit configuration packed into words of
    ``sbd_data.bit_length`` bits, as returned by :func:`from_string`. Bit ``2 * i``
    is the occupation of spin-alpha orbital ``i`` and bit ``2 * i + 1`` that of
    spin-beta orbital ``i``.

    Args:
        fcidump: FCIDump object.
        det: Determinants spanning the subspace. Must be distinct; they are
            sorted into SBD's canonical order internally, which
            :func:`sort_bitarray` reproduces.
        sbd_data: GDB_SBD configuration object. ``b_comm_size`` must be 1.
        loadname: Path to load initial wavefunction (optional).
        savename: Path prefix to save the final wavefunction to (optional). SBD
            writes ``f"{savename}000000.bin"``, holding the determinants in
            canonical order and their amplitudes.
        device: Override device ('cpu', 'gpu', or None for default).

    Returns:
        dict with keys: energy, density, carryover_det, one_p_rdm, two_p_rdm.
    """
    _ensure_initialized()
    backend = get_backend(device)
    return backend.gdb_diag(
        _global_comm, sbd_data, fcidump, det, loadname, savename
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def available_backends():
    """Get list of compiled backends ('cpu', 'gpu', 'gpu-omp')."""
    return list(_backends.keys())


def backend_load_errors():
    """Why each unavailable backend is unavailable.

    Maps device name -> reason, distinguishing "not built" from "built but
    failed to load" (a missing libmpi at run time, a wrong-architecture
    binary, an OpenMP runtime clash). Empty when every backend loaded.
    """
    return dict(_backend_errors)


def has_backend_conflict():
    """True when incompatible backends are loaded in this process.

    The OMP-offload backend cannot share a process with CPU/Thrust: offload
    regions silently run on the host while device queries still report a GPU.
    """
    return _omp_offload_conflict


def print_info():
    """Print SBD information."""
    print("=" * 60)
    print("SBD (Selected Basis Diagonalization) Python Bindings")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Compiled backends: {', '.join(available_backends()) or 'none'}")

    if _initialized:
        print(f"\nCurrent session:")
        print(f"  Default device: {_default_device}")
        print(f"  Communication: {_comm_backend}")
        print(f"  MPI rank: {get_rank()}/{get_world_size()}")
    else:
        print(f"\nNot initialized. Call sbd.init() or any API function to start.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

try:
    from . import sbd_solver
except ImportError:
    sbd_solver = None  # pyscf or qiskit-addon-sqd not installed

__all__ = [
    # Initialization
    'init',
    'finalize',
    'is_initialized',

    # Backend access
    'get_backend',
    'available_backends',
    'backend_load_errors',
    'has_backend_conflict',

    # Query
    'get_device',
    'get_comm_backend',
    'get_rank',
    'get_world_size',
    'get_comm',
    'barrier',

    # Main API
    'TPB_SBD',
    'GDB_SBD',
    'FCIDump',
    'LoadFCIDump',
    'LoadAlphaDets',
    'makestring',
    'from_string',
    'sort_bitarray',
    'tpb_diag_from_files',
    'tpb_diag',
    'gdb_diag',

    # Utilities
    'available_backends',
    'print_info',

    # Sub-modules
    'sbd_solver',

    # Version
    '__version__',
]
