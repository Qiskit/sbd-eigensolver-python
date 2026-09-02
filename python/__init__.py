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
# Backend registry.
#
# Backends are resolved lazily: only the one actually requested gets imported.
# This is not a micro-optimisation. Every _core_*.so here links NVHPC's
# libnvomp, and _core_cpu is built without -mp=gpu; loading it first leaves
# that runtime initialised host-only, after which the OMP-offload backend
# cannot acquire a device. It does not fail -- offload regions quietly run on
# the host while device queries still report a GPU, so the run looks
# accelerated, returns the correct energy, and exits 0. Measured on GB200:
# with _core_cpu co-loaded, OMP_TARGET_OFFLOAD=MANDATORY aborts with "Could
# not run target region"; importing only _core_gpu_omp_offload from the very
# same directory succeeds on the device. Loading one backend per process is
# therefore what makes co-resident .so files safe, and lets a single
# environment serve all three.
# ---------------------------------------------------------------------------

# (module name, canonical device, aliases)
_BACKEND_SPECS = (
    ('_core_cpu',             'cpu',     ()),
    ('_core_gpu_thrust',      'gpu',     ('gpu-thrust', 'gpu-nvidia', 'cuda')),
    ('_core_gpu_omp_offload', 'gpu-omp', ('gpu-omp-offload', 'gpu-nvhpc-omp',
                                          'gpu-nvidia-omp')),
)

_backends = {}          # device -> imported module, populated on first use
_backend_errors = {}    # device -> why it is unusable
_device_aliases = {a: dev for _m, dev, al in _BACKEND_SPECS for a in al}
_usable_cache = None    # list of usable devices; the static scan runs once


def _extension_path(module_name):
    """Path of the built extension for `module_name`, or None."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        names = os.listdir(here)
    except OSError:
        return None
    for n in sorted(names):
        if n.startswith(module_name + '.') and n.endswith(('.so', '.pyd', '.dylib')):
            return os.path.join(here, n)
    return None


def _inspect_extension(path):
    """Statically judge whether `path` can load. Returns (ok, reason).

    Deliberately does NOT import it. Importing a backend runs MPI_Init, which
    on an MPI without PMIx support hangs when the process was not started under
    a launcher -- a diagnostic that hangs is worse than none. `ldd` inspects the
    file without executing its initialisers.

    Catches a missing dependency and an architecture mismatch. It cannot catch
    an ABI/symbol mismatch, which only surfaces on a real dlopen; that is
    reported by _load_backend() at first use.
    """
    import subprocess, sys as _sys
    if _sys.platform == 'darwin':
        return True, None          # otool output does not mark unresolved deps
    try:
        out = subprocess.run(['ldd', path], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:
        return True, None          # no ldd: fall through to the real import
    if 'not a dynamic executable' in out:
        try:
            kind = subprocess.run(['file', '-b', path], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            kind = 'unknown'
        import platform
        return False, (f"architecture mismatch: the file is '{kind}' but this "
                       f"host is {platform.machine()}. Rebuild it here. (Note "
                       f"the loader reports this as 'cannot open shared object "
                       f"file', which is misleading -- the file exists.)")
    missing = [l.split('=>')[0].strip() for l in out.splitlines() if 'not found' in l]
    if missing:
        return False, (f"missing shared librar{'y' if len(missing) == 1 else 'ies'}: "
                       f"{', '.join(missing)}. Put the containing directory on "
                       f"LD_LIBRARY_PATH, or rebuild so RPATH covers it.")
    return True, None


def _scan_backends():
    """Populate the usable-device list and the reasons for the rest.

    Runs once. A backend that was simply not built is recorded but not warned
    about -- building CPU-only is a normal choice. A backend that IS built but
    cannot load is a broken install, so that one warns.
    """
    global _usable_cache
    if _usable_cache is not None:
        return _usable_cache
    import warnings
    usable, broken = [], []
    for module_name, device, _aliases in _BACKEND_SPECS:
        path = _extension_path(module_name)
        if path is None:
            _backend_errors[device] = (
                f"not built (no {module_name}.*.so in the package directory)"
            )
            continue
        ok, reason = _inspect_extension(path)
        if ok:
            usable.append(device)
        else:
            _backend_errors[device] = reason
            broken.append(f"{device}: {reason}")
    _usable_cache = usable
    if broken:
        warnings.warn(
            "sbd: backend(s) present on disk but not usable —\n  "
            + "\n  ".join(broken),
            RuntimeWarning,
            stacklevel=3,
        )
    return usable


def _load_backend(device):
    """Import the backend for `device`, or raise with an actionable message."""
    if device in _backends:
        return _backends[device]
    module_name = next((m for m, d, _a in _BACKEND_SPECS if d == device), None)
    if module_name is None:
        raise RuntimeError(f"Unknown backend device '{device}'.")

    path = _extension_path(module_name)
    if path is None:
        raise RuntimeError(
            f"sbd backend '{device}' is not built.\n"
            f"  expected: {module_name}.*.so in the sbd package directory\n"
            f"  usable  : {_scan_backends() or 'none'}\n"
            f"  fix     : rebuild, e.g. "
            f"SBD_BUILD_BACKEND={'cpu' if device == 'cpu' else 'gpu_omp_offload' if device == 'gpu-omp' else 'gpu'}"
            f" pip install -e . --no-build-isolation"
        )

    from importlib import import_module
    try:
        mod = import_module(f'.{module_name}', package=__name__)
    except Exception as exc:
        # Never substitute a different backend: a silently-swapped device is the
        # same class of lie as a silent host fallback.
        ok, reason = _inspect_extension(path)
        detail = reason or f"{type(exc).__name__}: {exc}"
        _backend_errors[device] = detail
        raise RuntimeError(
            f"sbd backend '{device}' could not be loaded.\n"
            f"  module: {path}\n"
            f"  cause : {type(exc).__name__}: {exc}\n"
            f"  detail: {detail}\n"
            f"  If the cause mentions an undefined symbol, the extension was "
            f"built against a different Python or mpi4py than this "
            f"environment's; rebuild it here."
        ) from exc

    _backends[device] = mod
    _backend_errors.pop(device, None)
    return mod


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
    Thrust path is the long-validated default. Since backends load lazily,
    all three may live in one install: only the resolved one is imported, so
    _core_cpu never poisons the OMP-offload runtime.
    """
    if device == 'auto':
        usable = _scan_backends()
        if 'gpu' in usable and _gpu_available():
            return 'gpu'
        if 'gpu-omp' in usable and _gpu_available():
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

    if not _scan_backends():
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
    usable = _scan_backends()
    if resolved not in usable:
        why = _backend_errors.get(resolved, 'unknown reason')
        raise RuntimeError(
            f"Device '{resolved}' requested but its backend is not usable "
            f"({why}). Usable: {usable or 'none'}"
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
    return _load_backend(device)


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
    """Devices whose extension is built and structurally loadable.

    Each candidate .so is inspected statically -- present on disk, no
    unresolved shared-library dependencies, matching architecture -- without
    importing it. Importing would run MPI_Init, which hangs on an MPI lacking
    PMIx support when the process was not started under a launcher.

    A listed backend is therefore "present and structurally sound", not
    "guaranteed to load": an ABI/symbol mismatch only shows up on a real
    dlopen, and surfaces from get_backend(). Reasons for anything excluded are
    in backend_load_errors(); a backend that is merely not built is recorded
    there but does not warn, since building a subset is normal.
    """
    return list(_scan_backends())


def loaded_backends():
    """Devices actually imported into this process so far.

    Normally one: backends load on first use. More than one means something
    imported them explicitly, which is worth knowing -- co-loading _core_cpu
    with the OMP-offload backend silently demotes offload to the host.
    """
    return list(_backends.keys())


def backend_load_errors():
    """Why each unusable backend is unusable.

    Maps device -> reason, distinguishing "not built" from a missing shared
    library, an architecture mismatch, or a failed import.
    """
    return dict(_backend_errors)


def has_backend_conflict():
    """True when _core_cpu and the OMP-offload backend are both loaded here.

    That combination leaves libnvomp initialised host-only, so offload regions
    run on the host while device queries still report a GPU. Lazy loading
    normally prevents it; this catches a caller that imported both explicitly.
    """
    return 'cpu' in _backends and 'gpu-omp' in _backends


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
    'loaded_backends',
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
