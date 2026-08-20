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
            device: Backend device key — 'cpu', 'gpu' (NVHPC Thrust),
                'gpu-omp' (nvc++ OpenMP target offload), or any alias
                known to ``sbd._device_aliases``. Default 'cpu'.
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
        Auto-detect GPU availability and use it if available.
        
        Args:
            max_memory_gb: Maximum GPU memory in GB (-1 = auto)
            
        Returns:
            DeviceConfig configured for GPU if available, CPU otherwise
        """
        # Check if CUDA or HIP is available
        has_cuda = cls._check_cuda()
        has_hip = cls._check_hip()
        
        use_gpu = has_cuda or has_hip

        if use_gpu:
            print(f"GPU detected ({'CUDA' if has_cuda else 'HIP'}), using GPU acceleration")
        else:
            print("No GPU detected, using CPU")

        device = 'gpu' if use_gpu else 'cpu'
        return cls(device=device, max_memory_gb=max_memory_gb)

    @classmethod
    def cpu(cls) -> 'DeviceConfig':
        """Force CPU execution."""
        return cls(device='cpu')

    @classmethod
    def gpu(cls, use_precalculated_dets: bool = True,
            max_memory_gb: int = -1) -> 'DeviceConfig':
        """Force NVHPC Thrust GPU execution.

        Requires SBD compiled with THRUST (the ``_core_gpu`` extension,
        i.e. ``SBD_BUILD_BACKEND=gpu`` or ``=both``).
        """
        return cls(device='gpu',
                   use_precalculated_dets=use_precalculated_dets,
                   max_memory_gb=max_memory_gb)

    @classmethod
    def gpu_omp(cls, max_memory_gb: int = -1) -> 'DeviceConfig':
        """Force OpenMP target-offload GPU execution.

        Requires SBD compiled with the OMP-offload backend (the
        ``_core_gpu_omp_offload`` extension, i.e.
        ``SBD_BUILD_BACKEND=gpu_omp_offload``). The backend uses
        ``nvc++ -mp=gpu`` with NVHPC's ``libnvomp`` runtime; it cannot
        coexist in a single Python process with the CPU or Thrust GPU
        backends (different OpenMP runtimes — install separately).
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
        """Check if HIP/ROCm is available (cached)."""
        if cls._hip_cache is not None:
            return cls._hip_cache
        try:
            result = subprocess.run(
                ['rocm-smi'], capture_output=True, timeout=2
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
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                info['gpu_count'] = len([l for l in result.stdout.split('\n')
                                        if 'GPU' in l])
        except Exception:
            pass
    
    return info


def print_device_info():
    """Print information about available compute devices."""
    info = get_device_info()
    
    print("="*60)
    print("SBD Device Information")
    print("="*60)
    
    if info['gpu_available']:
        print(f"✓ GPU Available: {info['gpu_type']}")
        if info['gpu_count'] > 0:
            print(f"  GPU Count: {info['gpu_count']}")
    else:
        print("✗ No GPU detected")
    
    print(f"✓ CPU Available: Always")
    print("="*60)
