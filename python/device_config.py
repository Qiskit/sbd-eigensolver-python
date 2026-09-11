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
Device configuration helper for SBD Python bindings.

This module provides utilities to easily switch between CPU and GPU execution
without changing user code.
"""

import subprocess


class DeviceConfig:
    """
    Helper class to configure CPU vs GPU execution for SBD calculations.
    
    Usage:
        # Auto-detect (uses GPU if available)
        config = DeviceConfig.auto()
        
        # Force CPU
        config = DeviceConfig.cpu()
        
        # Force GPU with specific settings
        config = DeviceConfig.gpu(max_memory_gb=16)
        
        # Apply to SBD configuration
        sbd_config = sbd.TPB_SBD()
        config.apply(sbd_config)
    """
    
    def __init__(self, device: str = 'cpu',
                 use_precalculated_dets: bool = True,
                 max_memory_gb: int = -1,
                 use_gpu: bool | None = None):
        """
        Initialize device configuration.

        Args:
            device: Backend device key — 'cpu', 'gpu' (NVHPC Thrust,
                NVIDIA-only), 'gpu-omp' (OpenMP target offload, NVIDIA or
                AMD), or any alias known to ``sbd._device_aliases``.
                Default 'cpu'.
            use_precalculated_dets: Use precalculated determinants (GPU only)
            max_memory_gb: Maximum GPU memory in GB (-1 = auto)
            use_gpu: Deprecated boolean. If supplied without ``device``,
                ``True`` maps to 'gpu' (Thrust) and ``False`` to 'cpu'
                for backward compatibility with pre-OMP-offload code.
        """
        if use_gpu is not None and device == 'cpu':
            # Legacy boolean path
            device = 'gpu' if use_gpu else 'cpu'
        self.device = device
        self.use_gpu = device != 'cpu'
        self.use_precalculated_dets = use_precalculated_dets
        self.max_memory_gb = max_memory_gb
    
    @classmethod
    def auto(cls, max_memory_gb: int = -1) -> 'DeviceConfig':
        """
        Auto-detect the best available backend and use it.

        Resolves against the backends that were actually COMPILED, not just the
        hardware that is present. Detecting a GPU and returning 'gpu'
        unconditionally was wrong in two ways: on an AMD host it selected the
        Thrust/CUDA backend, which cannot exist there (upstream wires Thrust to
        nvc++ -cuda), so a machine that reported "GPU detected (HIP)" then failed
        to load a CUDA module; and on a CPU-only build with a GPU present it
        picked a backend that was never built.

        Preference order matches sbd._resolve_device(): Thrust ('gpu') first
        where it exists, since it is the long-validated NVIDIA default and keeps
        more phases on the device, then OpenMP offload ('gpu-omp'), then CPU.

        Args:
            max_memory_gb: Maximum GPU memory in GB (-1 = auto)

        Returns:
            DeviceConfig for the best backend that is both built and runnable
        """
        has_cuda = cls._check_cuda()
        has_hip = cls._check_hip()

        try:
            from . import available_backends
            built = available_backends()
        except Exception:
            built = []

        device = 'cpu'
        if has_cuda or has_hip:
            vendor = 'CUDA' if has_cuda else 'HIP/ROCm'
            for candidate in ('gpu', 'gpu-omp'):
                if candidate in built:
                    device = candidate
                    break
            if device == 'cpu':
                print(f"GPU detected ({vendor}) but no GPU backend is built "
                      f"(available: {built or 'none'}), using CPU")
            else:
                print(f"GPU detected ({vendor}), using GPU acceleration "
                      f"via device={device!r}")
        else:
            print("No GPU detected, using CPU")

        return cls(device=device, max_memory_gb=max_memory_gb)

    @classmethod
    def cpu(cls) -> 'DeviceConfig':
        """Force CPU execution."""
        return cls(device='cpu')

    @classmethod
    def gpu(cls, use_precalculated_dets: bool = True,
            max_memory_gb: int = -1) -> 'DeviceConfig':
        """Force NVHPC Thrust GPU execution. **NVIDIA only.**

        Requires SBD compiled with THRUST (the ``_core_gpu_thrust`` extension,
        i.e. ``SBD_BUILD_BACKEND=gpu``, or the default ``auto``).

        There is no AMD equivalent: upstream SBD wires the Thrust path to
        ``nvc++ -cuda``, so no rocThrust configuration exists to build. On an AMD
        host use :meth:`gpu_omp` instead.
        """
        return cls(device='gpu',
                   use_precalculated_dets=use_precalculated_dets,
                   max_memory_gb=max_memory_gb)

    @classmethod
    def gpu_omp(cls, max_memory_gb: int = -1) -> 'DeviceConfig':
        """Force OpenMP target-offload GPU execution. Works on NVIDIA **and AMD**.

        Requires SBD compiled with the OMP-offload backend (the
        ``_core_gpu_omp_offload`` extension), which the default
        ``SBD_BUILD_BACKEND=auto`` builds whenever a GPU compiler is present;
        narrow it to ``gpu_omp_offload`` to build only this one.

        One module and one device string serve both vendors -- the same source
        and macros, compiled by ``nvc++ -mp=gpu`` with NVHPC's ``libnvomp``, or by
        ``amdclang++ --offload-arch=gfx*`` with LLVM's ``libomp``/``libomptarget``.
        ``sbd.get_backend('gpu-omp').__sbd_offload_target__`` reports which, e.g.
        ``'amdgcn-amd-amdhsa:gfx90a'``.

        It installs alongside the CPU backend (and Thrust, on NVIDIA) -- backends
        are imported lazily, one per process, which is what keeps them apart. The
        one combination to avoid in a single process is this backend together
        with the CPU one: they share an OpenMP runtime (``libnvomp`` on NVIDIA,
        ``libomp`` on AMD), and loading ``_core_cpu`` first leaves it initialised
        host-only, after which offload regions run on the host. See
        :func:`sbd.has_backend_conflict`.
        """
        return cls(device='gpu-omp', max_memory_gb=max_memory_gb)

    @classmethod
    def gpu_nvidia_omp(cls, max_memory_gb: int = -1) -> 'DeviceConfig':
        """Deprecated alias for :meth:`gpu_omp`.

        The old name dates from when SBD shipped a LLVM-with-NVPTX
        offload backend distinct from the nvc++ path. The LLVM backend
        was removed in v1.6 (see tag ``v1.5.0-llvm`` for that history);
        ``gpu_omp`` is the single OpenMP-offload path now.
        """
        import warnings
        warnings.warn(
            "DeviceConfig.gpu_nvidia_omp() is a deprecated alias for "
            "gpu_omp(); use gpu_omp() directly.",
            DeprecationWarning, stacklevel=2,
        )
        return cls.gpu_omp(max_memory_gb=max_memory_gb)
    
    # Cached detection results (None = not yet checked)
    _cuda_cache: bool | None = None
    _hip_cache: bool | None = None

    @classmethod
    def _check_cuda(cls) -> bool:
        """Check if CUDA is available (cached)."""
        if cls._cuda_cache is not None:
            return cls._cuda_cache
        try:
            result = subprocess.run(
                ['nvidia-smi'], capture_output=True, timeout=2
            )
            cls._cuda_cache = result.returncode == 0
        except Exception:
            cls._cuda_cache = False
        return cls._cuda_cache

    @classmethod
    def _check_hip(cls) -> bool:
        """Check if HIP/ROCm is available (cached).

        The timeout is much longer than the CUDA probe's on purpose: rocm-smi is
        a Python program that enumerates devices, measured at 1.1-1.3 s on an
        8-GCD MI250X node, where nvidia-smi answers in tens of milliseconds. The
        2 s used here originally left barely 1.5x of margin and DID flake --
        reporting no AMD GPU on a machine with eight, which then sent
        DeviceConfig.auto() to the CPU backend. The result is cached, so a
        generous timeout costs at most one slow call per process.
        """
        if cls._hip_cache is not None:
            return cls._hip_cache
        try:
            result = subprocess.run(
                ['rocm-smi'], capture_output=True, timeout=30
            )
            cls._hip_cache = result.returncode == 0
        except Exception:
            cls._hip_cache = False
        return cls._hip_cache
    
    def apply(self, sbd_config) -> None:
        """
        Apply device configuration to an SBD TPB_SBD configuration object.

        Args:
            sbd_config: sbd.TPB_SBD configuration object
        """
        # `use_precalculated_dets` and `max_memory_gb_for_determinants`
        # are Thrust-only struct fields (guarded by SBD_THRUST in
        # sbdiag.h). The OMP-offload backend doesn't expose them — that
        # path doesn't have the precalculated-dets cache yet, so there's
        # nothing to wire up.
        if self.device == 'gpu':
            try:
                sbd_config.use_precalculated_dets = self.use_precalculated_dets
                sbd_config.max_memory_gb_for_determinants = self.max_memory_gb
            except AttributeError:
                print("WARNING: Thrust GPU knobs not exposed by this "
                      "backend module — likely a CPU/OMP-offload build "
                      "loaded under device='gpu'. Numerics still run on "
                      "the selected backend.")
    
    def __repr__(self) -> str:
        if self.device == 'cpu':
            return "DeviceConfig(CPU)"
        return (f"DeviceConfig(device={self.device!r}, "
                f"precalc_dets={self.use_precalculated_dets}, "
                f"max_mem={self.max_memory_gb}GB)")


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': DeviceConfig._check_cuda(),
        'hip_available': DeviceConfig._check_hip(),
        'gpu_available': False,
        'gpu_type': None,
        'gpu_count': 0
    }
    
    if info['cuda_available']:
        info['gpu_available'] = True
        info['gpu_type'] = 'CUDA'
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                info['gpu_count'] = len([l for l in result.stdout.split('\n') if l.strip()])
        except Exception:
            pass

    elif info['hip_available']:
        info['gpu_available'] = True
        info['gpu_type'] = 'HIP/ROCm'
        try:
            result = subprocess.run(
                ['rocm-smi', '--showid'],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                # Count DISTINCT GPU indices, not lines mentioning "GPU":
                # --showid prints several lines per device (Device Name, Device
                # ID, Rev, Subsystem ID, GUID), so a line count reported 40 for
                # the 8 GCDs of a 4-card MI250X node.
                import re
                ids = re.findall(r'GPU\[(\d+)\]', result.stdout)
                info['gpu_count'] = len(set(ids))
        except Exception:
            pass
    
    return info


def print_device_info():
    """Print the compiled backends, then the hardware they could run on.

    Backends come first because they are what actually constrains a run: the
    hardware being present says nothing about whether a backend was compiled
    for it. An earlier version printed "CPU Available: Always" unconditionally
    while sbd.available_backends() returned [] -- reported as issue #9.
    """
    info = get_device_info()

    from . import available_backends, backend_load_errors, has_backend_conflict
    backends = available_backends()
    errors = backend_load_errors()

    print("=" * 60)
    print("SBD Device Information")
    print("=" * 60)

    print(f"Compiled backends: {', '.join(backends) if backends else 'NONE'}")
    if not backends:
        print("  Nothing was built, or nothing could be loaded. This install")
        print('  cannot run: solve_sci will raise "Backend not available".')
    for device, reason in sorted(errors.items()):
        print(f"  {device:8} unavailable ({reason})")
    if has_backend_conflict():
        print("  WARNING: OMP-offload is loaded alongside CPU/Thrust. Offload")
        print("           regions will silently run on the host even though the")
        print("           GPU query below succeeds. Give _core_gpu_omp_offload.so")
        print("           a directory of its own and rebuild.")

    print("-" * 60)
    print("Hardware detected (independent of what was compiled):")
    if info['gpu_available']:
        print(f"  GPU: {info['gpu_type']}"
              + (f", count {info['gpu_count']}" if info['gpu_count'] > 0 else ""))
    else:
        print("  GPU: none detected")
    print("  CPU: always present")
    print("=" * 60)
