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

from setuptools import setup, Extension
import sys
import os
import subprocess
import pybind11


class get_pybind_include(object):
    def __str__(self):
        return pybind11.get_include()


def get_mpi4py_include():
    try:
        import mpi4py
        return mpi4py.get_include()
    except (ImportError, AttributeError):
        import site
        for site_dir in site.getsitepackages():
            mpi4py_inc = os.path.join(site_dir, 'mpi4py', 'include')
            if os.path.exists(mpi4py_inc):
                return mpi4py_inc
        return None


def _mpi_config_from_mpicc():
    """Probe the mpicc compiler wrapper for include/library/link flags.

    Returns (include_dirs, library_dirs, libraries) or None if mpicc is
    absent or does not understand the --showme flags (an OpenMPI-ism;
    MPICH's wrapper does not support them).
    """
    try:
        compile_flags = subprocess.check_output(['mpicc', '--showme:compile'],
                                                universal_newlines=True).strip().split()
        link_flags = subprocess.check_output(['mpicc', '--showme:link'],
                                             universal_newlines=True).strip().split()
    except Exception:
        return None
    include_dirs = [flag[2:] for flag in compile_flags if flag.startswith('-I')]
    library_dirs = [flag[2:] for flag in link_flags if flag.startswith('-L')]
    libraries = [flag[2:] for flag in link_flags if flag.startswith('-l')]
    return include_dirs, library_dirs, libraries


def get_mpi_config():
    # Prefer MPI_HOME, but only trust it if mpi.h actually lives at
    # $MPI_HOME/include. Distros that split MPI into a -devel package
    # (e.g. Fedora's environment-modules sets MPI_HOME=/usr/lib64/openmpi
    # while headers live in /usr/include/openmpi-x86_64) break the naive
    # $MPI_HOME/include assumption, so we fall through to mpicc there.
    mpi_home = os.environ.get('MPI_HOME', None)
    if mpi_home:
        mpi_include = os.path.join(mpi_home, 'include')
        mpi_lib = os.path.join(mpi_home, 'lib')
        if os.path.exists(os.path.join(mpi_include, 'mpi.h')):
            print(f"Using MPI from MPI_HOME: {mpi_home}")
            return [mpi_include], [mpi_lib], ['mpi']
        print(f"Notice: MPI_HOME={mpi_home} set but {mpi_include}/mpi.h "
              "not found; falling back to mpicc detection.")

    mpicc_config = _mpi_config_from_mpicc()
    if mpicc_config is not None:
        print("Using MPI detected from mpicc")
        return mpicc_config

    # Last resort: honor MPI_HOME even without a discoverable mpi.h, so a
    # deliberately-set MPI_HOME on an unusual layout still gets a chance.
    if mpi_home:
        print(f"Warning: using MPI_HOME={mpi_home} despite missing "
              f"{mpi_include}/mpi.h and unusable mpicc.")
        return [os.path.join(mpi_home, 'include')], \
               [os.path.join(mpi_home, 'lib')], ['mpi']

    print("Error: Could not detect MPI. Please set MPI_HOME environment "
          "variable, or ensure mpicc is on PATH.")
    sys.exit(1)


def _resolve_gpu_arch(default='cc90'):
    """Pick the nvc++ -gpu=<arch> value from env, with back-compat alias.

    Both Thrust (_core_gpu_thrust) and OMP-offload (_core_gpu_omp_offload)
    backends compile with nvc++ and take the same -gpu=<arch> flag, so we
    use a single env var.

    Reads SBD_GPU_ARCH (canonical name since v1.6). Falls back to the
    deprecated SBD_GPU_ARCH_NVIDIA (v1.5 and earlier) with a notice so
    existing setup scripts keep working through the transition.
    """
    val = os.environ.get('SBD_GPU_ARCH')
    if val:
        return val
    legacy = os.environ.get('SBD_GPU_ARCH_NVIDIA')
    if legacy:
        print(f"Notice: SBD_GPU_ARCH_NVIDIA={legacy!r} is deprecated since "
              "v1.6 (single SBD_GPU_ARCH covers both Thrust and OMP-offload "
              "since LLVM/clang was removed). Honoring it as a back-compat "
              "alias. Please switch to SBD_GPU_ARCH.")
        return legacy
    return default


def _route_build_through_nvhpc(nvc_path):
    """Configure distutils + sysconfig so a setup() call uses nvc++.

    Called by both the Thrust and OMP-offload extension blocks (both
    compile with nvc++). Idempotent — second call is a no-op.

    Effect: distutils' UnixCCompiler will pick up CC/CXX/LDSHARED from
    os.environ and use them for every Extension in this setup() call.
    Also clears CFLAGS/CXXFLAGS/CPPFLAGS and rewrites sysconfig to drop
    gcc-specific tokens nvc++ rejects (RHEL 9 CPython injects a long
    list — see comment below).

    Co-builds with the CPU extension are safe: nvc++ accepts the CPU
    block's `-fopenmp -O3 -std=c++17` flags (treats -fopenmp as -mp).
    """
    if os.environ.get('_SBD_NVHPC_ROUTING_APPLIED'):
        return
    os.environ['_SBD_NVHPC_ROUTING_APPLIED'] = '1'

    # Respect user-set CC/CXX (e.g. cross-toolchain); otherwise pin nvc++.
    os.environ.setdefault('CC',       nvc_path)
    os.environ.setdefault('CXX',      nvc_path)
    os.environ.setdefault('LDSHARED', f'{nvc_path} -shared')
    os.environ.setdefault('CFLAGS',   '')
    os.environ.setdefault('CXXFLAGS', '')
    os.environ.setdefault('CPPFLAGS', '')

    # RHEL 9 CPython sysconfig injects gcc-specific flags that nvc++
    # rejects (-grecord-gcc-switches, -Wp,-D_FORTIFY_SOURCE=2,
    # -fstack-protector-strong, -fasynchronous-unwind-tables,
    # -fstack-clash-protection, -fcf-protection, -fwrapv) plus a
    # -march=x86-64-v2 default that nvc++ explicitly rejects
    # (requires v3+). distutils pulls these from sysconfig in addition
    # to os.environ.CFLAGS, so blanking the latter alone is not enough
    # — we rewrite the sysconfig dict itself.
    import sysconfig, re as _re
    _cfg = sysconfig.get_config_vars()
    _strip_tokens = (
        '-grecord-gcc-switches',
        '-Wp,-D_FORTIFY_SOURCE=2',
        '-Wp,-D_GLIBCXX_ASSERTIONS',
        '-fstack-protector-strong',
        '-fasynchronous-unwind-tables',
        '-fstack-clash-protection',
        '-fcf-protection',
        '-fwrapv',
        '-Wno-unused-result',
    )
    for _k in list(_cfg.keys()):
        _v = _cfg[_k]
        if not isinstance(_v, str):
            continue
        for _bad in _strip_tokens:
            _v = _v.replace(_bad, '')
        _v = _v.replace('-march=x86-64-v2', '-march=x86-64-v3')
        _cfg[_k] = _re.sub(r' +', ' ', _v).strip()


def find_nvidia_hpc_sdk():
    nvhpc_home = os.environ.get('NVHPC_HOME', None)
    if nvhpc_home:
        nvcxx_path = os.path.join(nvhpc_home, 'bin', 'nvc++')
        if os.path.exists(nvcxx_path):
            print(f"Found NVIDIA HPC SDK at: {nvhpc_home}")
            nvhpc_bin = os.path.join(nvhpc_home, 'bin')
            current_path = os.environ.get('PATH', '')
            if nvhpc_bin not in current_path:
                os.environ['PATH'] = f"{nvhpc_bin}:{current_path}"
            return nvcxx_path, True
        else:
            print(f"Warning: NVHPC_HOME set to {nvhpc_home} but nvc++ not found")
    import shutil
    nvcxx_path = shutil.which('nvc++')
    if nvcxx_path:
        print(f"Found nvc++ in PATH: {nvcxx_path}")
        return nvcxx_path, True
    return None, False


# Get MPI configuration
mpi_includes, mpi_lib_dirs, mpi_libs = get_mpi_config()

# Get mpi4py include path
mpi4py_inc = get_mpi4py_include()
if not mpi4py_inc:
    print("Warning: Could not find mpi4py include path")

# Build include/library directories.
# SBD's C++ headers come from the vendored upstream submodule.
# After cloning the parent repo, run:  git submodule update --init --recursive
SBD_UPSTREAM_INCLUDE = os.path.join('vendor', 'sbd-upstream', 'include')
if not os.path.isdir(SBD_UPSTREAM_INCLUDE):
    print(f"Error: {SBD_UPSTREAM_INCLUDE} not found.")
    print("Run: git submodule update --init --recursive")
    sys.exit(1)
include_dirs = [get_pybind_include(), SBD_UPSTREAM_INCLUDE] + mpi_includes
if mpi4py_inc:
    include_dirs.append(mpi4py_inc)

library_dirs = mpi_lib_dirs.copy()

blas_lib_path = os.environ.get('BLAS_LIB_PATH', None)
if blas_lib_path:
    library_dirs.append(blas_lib_path)
    print(f"Using BLAS from: {blas_lib_path}")
else:
    print("Warning: BLAS_LIB_PATH not set. Assuming BLAS is in system path.")

blas_libs = os.environ.get('BLAS_LIBS', 'openblas').split(',')
print(f"Using BLAS libraries: {blas_libs}")

libraries = mpi_libs + blas_libs

# RPATH so libraries are found at runtime without LD_LIBRARY_PATH
extra_link_args = ['-fopenmp']
for lib_dir in library_dirs:
    extra_link_args.append(f'-Wl,--rpath,{lib_dir}')
print(f"RPATH will be set to: {library_dirs}")

# Detect NVHPC. nvc++ is shared between two GPU backends here:
#   1. _core_gpu_thrust       (Thrust + CUDA path,  nvc++ -cuda)
#   2. _core_gpu_omp_offload  (OpenMP target offload, nvc++ -mp=gpu)
gpu_compiler, has_nvhpc = find_nvidia_hpc_sdk()

# Determine which backends to build.
#   auto                  : cpu + thrust GPU (if nvc++ present)
#   cpu                   : cpu only
#   gpu | gpu_thrust      : thrust GPU only
#   both                  : cpu + thrust
#   gpu_omp_offload       : OpenMP target offload only (nvc++ -mp=gpu)
#
# gpu_omp_offload is built ALONE — it uses a different OpenMP runtime
# (libnvomp) than cpu (libgomp/libomp) and Thrust GPU (CPU OMP via -mp),
# and loading two backends with different OMP runtimes in one Python
# process produces "Another OpenMP runtime library has been detected"
# warnings and can deadlock at first OMP region. Build it into its own
# venv / install dir.
build_backend = os.environ.get('SBD_BUILD_BACKEND', 'auto').lower()

build_cpu = False
build_gpu_thrust = False
build_gpu_omp_offload = False

if build_backend == 'auto':
    build_cpu = True
    build_gpu_thrust = has_nvhpc
    if build_gpu_thrust:
        print("\nAuto-detected nvc++ - will build both CPU and Thrust GPU backends")
    else:
        print("\nnvc++ not found - will build CPU backend only")
elif build_backend == 'cpu':
    build_cpu = True
    print("\nBuilding CPU backend only (SBD_BUILD_BACKEND=cpu)")
elif build_backend in ('gpu', 'gpu_thrust'):
    build_gpu_thrust = True
    print(f"\nBuilding Thrust GPU backend only (SBD_BUILD_BACKEND={build_backend})")
    if not has_nvhpc:
        print("Warning: nvc++ not found, GPU build may fail")
elif build_backend == 'both':
    build_cpu = True
    build_gpu_thrust = True
    print("\nBuilding both CPU and Thrust GPU backends (SBD_BUILD_BACKEND=both)")
    if not has_nvhpc:
        print("Warning: nvc++ not found, GPU build may fail")
elif build_backend == 'gpu_omp_offload':
    # Stand-alone build: this mode only emits _core_gpu_omp_offload.so.
    # See note above on the OpenMP-runtime exclusivity constraint.
    build_gpu_omp_offload = True
    print("\nBuilding GPU OpenMP target-offload backend only "
          "(SBD_BUILD_BACKEND=gpu_omp_offload)")
    if not has_nvhpc:
        print("Error: gpu_omp_offload requires NVHPC_HOME / nvc++.")
        sys.exit(1)
else:
    print(f"Error: Invalid SBD_BUILD_BACKEND='{build_backend}'")
    print("Valid values: auto, cpu, gpu (alias gpu_thrust), both, gpu_omp_offload")
    sys.exit(1)

ext_modules = []

if build_cpu:
    print("\nConfiguring CPU backend (_core_cpu)")
    import platform
    if platform.system() == 'Darwin':
        omp_inc = '/opt/homebrew/opt/libomp/include'
        omp_lib = '/opt/homebrew/opt/libomp/lib'
        openblas_lib = '/opt/homebrew/opt/openblas/lib'
        cpu_compile_args = [
            '-DSBD_TRADMODE',
            '-std=c++17', '-Xpreprocessor', '-fopenmp', '-O3',
            '-Wno-sign-compare', '-Wno-unused-variable', '-fPIC',
            '-DSBD_MODULE_NAME=_core_cpu', f'-I{omp_inc}',
        ]
        cpu_link_args = [f'-L{omp_lib}', f'-L{openblas_lib}', '-lomp']
        cpu_inc = include_dirs + [omp_inc]
        cpu_lib_dirs = library_dirs + [omp_lib, openblas_lib]
        cpu_libs = libraries + ['omp']
    else:
        cpu_compile_args = [
            '-DSBD_TRADMODE',
            '-DOMPI_SKIP_MPICXX',
            '-std=c++17', '-fopenmp', '-O3',
            '-Wno-sign-compare', '-Wno-unused-variable', '-fPIC',
            '-DSBD_MODULE_NAME=_core_cpu',
        ]
        cpu_link_args = extra_link_args
        cpu_inc = include_dirs
        cpu_lib_dirs = library_dirs
        cpu_libs = libraries

    cpu_ext = Extension(
        'sbd._core_cpu',
        ['python/bindings.cpp'],
        include_dirs=cpu_inc,
        libraries=cpu_libs,
        library_dirs=cpu_lib_dirs,
        language='c++',
        extra_compile_args=cpu_compile_args,
        extra_link_args=cpu_link_args,
    )
    ext_modules.append(cpu_ext)


if build_gpu_thrust:
    print("\nConfiguring Thrust GPU backend (_core_gpu_thrust)")
    if not gpu_compiler:
        print("Error: GPU backend requested but nvc++ not found")
        sys.exit(1)
    print(f"Using compiler: {gpu_compiler}")
    # Auto-route the build through nvc++ + sanitize sysconfig flags.
    # No-op if the user already set CC/CXX manually.
    _route_build_through_nvhpc(gpu_compiler)
    gpu_arch = _resolve_gpu_arch(default='cc90')
    print(f"NVHPC -gpu= arch: {gpu_arch} (set SBD_GPU_ARCH to override; "
          "nvc++ accepts cc<XX> and sm_<XX>)")

    gpu_thrust_ext = Extension(
        'sbd._core_gpu_thrust',
        ['python/bindings.cpp'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        extra_compile_args=[
            '-DSBD_THRUST',
            '-DSBD_TRADMODE',
            '-mp',
            '-cuda',
            '-fast',
            '-Minfo=accel',
            '--diag_suppress=declared_but_not_referenced,set_but_not_used',
            '-fmax-errors=0',
            '-fPIC',
            f'-gpu={gpu_arch}',
            '-DSBD_MODULE_NAME=_core_gpu_thrust',
        ],
        # NOTE: -cudalib (no value) makes nvc++ blanket-link every CUDA
        # library NVHPC ships, including math libs SBD never calls
        # (cublasmp, cusolverMp, cutensor, nvblas). On NVHPC 26.3 some of
        # those ship as dangling symlinks (.so name present but versioned
        # target missing), causing the link to fail with "cannot find
        # -lcublasmp" etc. SBD's GPU path only needs the CUDA runtime, so
        # explicitly link -lcudart instead.
        extra_link_args=extra_link_args + ['-mp', '-cuda', '-lcudart'],
    )
    ext_modules.append(gpu_thrust_ext)


if build_gpu_omp_offload:
    print("\nConfiguring GPU OpenMP target-offload backend (_core_gpu_omp_offload)")
    print(f"Using compiler: {gpu_compiler}")
    # Auto-route the build through nvc++ + sanitize sysconfig flags.
    _route_build_through_nvhpc(gpu_compiler)
    offload_arch = _resolve_gpu_arch(default='cc90')
    print(f"NVHPC -gpu= arch: {offload_arch} (set SBD_GPU_ARCH to override)")

    gpu_omp_offload_ext = Extension(
        'sbd._core_gpu_omp_offload',
        ['python/bindings.cpp'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        extra_compile_args=[
            '-O3', '-std=c++17', '-fPIC',
            '-mp=gpu',
            f'-gpu={offload_arch}',
            '-Minfo=mp',
            '-DSBD_TRADMODE',
            '-DUSE_GPU',
            '-DUSE_OMP_OFFLOAD',
            '-DOMPI_SKIP_MPICXX',
            '-DSBD_MODULE_NAME=_core_gpu_omp_offload',
            # Force-include nvc++ shim so __builtin_ffsl / __builtin_popcountl
            # inside #pragma omp declare target lower to portable inlines
            # rather than __blt_pgi_ffsl (host-only NVHPC symbol that nvlink
            # can't resolve from device code).
            '-include', 'python/sbd_nvhpc_compat.h',
        ],
        extra_link_args=extra_link_args + [
            '-mp=gpu',
            f'-gpu={offload_arch}',
        ],
    )
    ext_modules.append(gpu_omp_offload_ext)


# All static metadata (name, version, dependencies, packages, etc.) is
# declared in pyproject.toml. This setup() call only carries the imperative
# ext_modules built above, which cannot be expressed declaratively.
setup(
    ext_modules=ext_modules,
)

print("\nSetup complete!")
if build_cpu:
    print("  - CPU backend:                    sbd._core_cpu")
if build_gpu_thrust:
    print("  - Thrust GPU backend:             sbd._core_gpu_thrust")
if build_gpu_omp_offload:
    print("  - OpenMP-offload GPU backend:     sbd._core_gpu_omp_offload")
print()
