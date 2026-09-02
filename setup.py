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
import re
import sysconfig
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


def _mpi_prefix_from_mpi4py():
    """Derive the MPI install prefix from the library mpi4py is linked against.

    mpi4py is a build requirement, so it is importable here. Deriving the prefix
    from it -- rather than from whichever mpicc happens to be first on PATH --
    guarantees the extensions link the same MPI that mpi4py uses. A mismatch is
    not a build error: it surfaces at run time as an MPI_Init abort (e.g. MPICH
    reporting "unsupported PMI version PMIx" under an Open MPI launcher), which
    is considerably harder to diagnose.

    Returns the prefix, or None when it cannot be determined -- a manylinux
    wheel bundling its own MPI, a static build, or a platform with neither ldd
    nor otool. Callers fall back to MPI_HOME.
    """
    try:
        import mpi4py
    except ImportError:
        return None
    pkg_dir = os.path.dirname(mpi4py.__file__)
    try:
        exts = [n for n in sorted(os.listdir(pkg_dir))
                if n.startswith('MPI.') and n.endswith(('.so', '.dylib', '.pyd'))]
    except OSError:
        return None
    # Some builds ship one extension per MPI flavour (MPI.mpich.*, MPI.openmpi.*)
    # and choose at import time, so linkage cannot tell us which one is in use.
    # Ambiguous: let the caller fall back to MPI_HOME.
    if len(exts) != 1:
        return None
    ext = exts[0]
    probe = ['otool', '-L'] if sys.platform == 'darwin' else ['ldd']
    try:
        out = subprocess.check_output(probe + [os.path.join(pkg_dir, ext)],
                                      universal_newlines=True,
                                      stderr=subprocess.DEVNULL)
    except Exception:
        return None
    # Take the resolved path and normalise it. A conda-installed MPI is reached
    # through a relative path with '..' segments (mpi4py/../../../libmpi.so.12),
    # so matching a literal '/lib/libmpi' misses it entirely; realpath also
    # follows the libmpi.so.12 -> libmpi.so.12.x.y symlink chain.
    match = re.search(r'=>\s*(\S*libmpi\S*)', out) or re.search(r'(\S*libmpi\S*)', out)
    if not match:
        return None
    raw = match.group(1)
    # macOS records @rpath/@loader_path-relative install names; resolving those
    # means walking LC_RPATH, which is not worth it here -- MPI_HOME covers it.
    if not raw.startswith('/'):
        return None
    lib_path = os.path.realpath(raw)
    prefix = os.path.dirname(os.path.dirname(lib_path))
    return prefix if os.path.exists(os.path.join(prefix, 'include', 'mpi.h')) else None


def _building_extensions():
    """True if this invocation will actually compile the C++ extensions.

    Creating an sdist, or generating metadata for one, imports this file but
    never runs a compiler, so a missing MPI or a missing vendored submodule
    must not be fatal there -- otherwise `python -m build --sdist` fails on
    any machine without an MPI toolchain, and the sdist can never be built
    for release. Compilation commands still hard-fail as before.

    Scan all of argv rather than argv[1]: setuptools' build_meta backend
    prepends global options (-q/-v, plus anything from --global-option)
    ahead of the command, so the command's position is not fixed.
    """
    return not {'sdist', 'egg_info'}.intersection(sys.argv[1:])


def get_mpi_config():
    """Resolve MPI include/lib dirs, preferring the MPI that mpi4py uses.

    Order of precedence: MPI_HOME when set (explicit override, and the escape
    hatch for layouts this cannot infer), otherwise the MPI mpi4py is linked
    against. When both are known and disagree, that is a hard error -- linking a
    different MPI than mpi4py aborts at MPI_Init, so failing the build is much
    the cheaper outcome.

    Both MPICH and Open MPI install libmpi, so -lmpi is correct for either.
    """
    derived = _mpi_prefix_from_mpi4py()
    mpi_home = os.environ.get('MPI_HOME') or None

    if mpi_home and derived and \
            os.path.realpath(mpi_home) != os.path.realpath(derived):
        print(f"Error: MPI_HOME={mpi_home} is not the MPI that mpi4py is linked "
              f"against ({derived}).\n"
              "       Building against a different MPI than mpi4py aborts at "
              "MPI_Init rather than at build time.\n"
              "       Either unset MPI_HOME to use mpi4py's MPI, or rebuild "
              "mpi4py against MPI_HOME:\n"
              f"         MPICC={mpi_home}/bin/mpicc pip install --no-binary=mpi4py "
              "--force-reinstall --no-cache-dir mpi4py\n"
              "       Then verify: python -c \"from mpi4py import MPI; "
              "print(MPI.Get_library_version())\"")
        sys.exit(1)

    prefix = mpi_home or derived
    if prefix:
        include_dir = os.path.join(prefix, 'include')
        lib_dir = next((os.path.join(prefix, d) for d in ('lib', 'lib64')
                        if os.path.isdir(os.path.join(prefix, d))),
                       os.path.join(prefix, 'lib'))
        print(f"Using MPI from {'MPI_HOME' if mpi_home else 'mpi4py'}: {prefix}")
        if not os.path.exists(os.path.join(include_dir, 'mpi.h')):
            print(f"Warning: {include_dir}/mpi.h not found; the compile will "
                  "likely fail. Check MPI_HOME points at an MPI *prefix*, not "
                  "its lib or bin directory.")
        return [include_dir], [lib_dir], ['mpi']

    if not _building_extensions():
        print("Notice: Could not detect MPI, but no extension is being "
              "compiled; continuing without MPI flags.")
        return [], [], ['mpi']

    print("Error: Could not determine which MPI to build against.\n"
          "       mpi4py did not reveal an MPI prefix containing include/mpi.h "
          "(a wheel that bundles its own MPI will do that), so set MPI_HOME:\n"
          "         MPI_HOME=/path/to/mpi pip install -e . --no-build-isolation\n"
          "       Better, install mpi4py against that MPI first so the two "
          "cannot diverge:\n"
          "         MPICC=/path/to/mpi/bin/mpicc pip install --no-binary=mpi4py "
          "mpi4py")
    sys.exit(1)


def _resolve_gpu_arch():
    """Return the nvc++ -gpu=<arch> value; required for any GPU backend.

    Both Thrust (_core_gpu_thrust) and OMP-offload (_core_gpu_omp_offload)
    compile with nvc++ and take the same -gpu=<arch> flag, so one env var
    covers both.

    There is deliberately no default. nvc++ happily emits code for a compute
    capability the machine does not have, and a wrong-arch Thrust build returns
    incorrect energies instead of failing -- a silent-wrong-answer mode that is
    far worse than a build error.

    Reads SBD_GPU_ARCH, honoring the deprecated SBD_GPU_ARCH_NVIDIA (v1.5 and
    earlier) with a notice so existing scripts keep working.
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
    print("Error: SBD_GPU_ARCH must be set to build a GPU backend (NVHPC was "
          "found).\n"
          "       No default is assumed: nvc++ will build for the wrong compute "
          "capability without complaint, and a wrong-arch Thrust build returns "
          "incorrect energies rather than failing.\n"
          "       cc80 = A100, cc90 = H100, cc100 = GB200/B200. For example:\n"
          "         SBD_GPU_ARCH=cc100 pip install -e . --no-build-isolation\n"
          "       To build the CPU backend only, unset NVHPC_HOME (and keep "
          "nvc++ off PATH) or set SBD_BUILD_BACKEND=cpu.")
    sys.exit(1)


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
        # conda's Python bakes '-B $CONDA_PREFIX/compiler_compat' into
        # CC/CXX/LDSHARED/LDCXXSHARED. nvc++ rejects -B and hands the
        # path to the linker as an input file, so drop just that flag
        # and keep conda's -L/-rpath entries intact.
        _v = re.sub(r'-B\s*\S*compiler_compat\S*', '', _v)
        _cfg[_k] = re.sub(r' +', ' ', _v).strip()


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
if not os.path.isdir(SBD_UPSTREAM_INCLUDE) and _building_extensions():
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
_rpath_dirs = []
for lib_dir in library_dirs:
    if lib_dir not in _rpath_dirs:          # a dir may appear as both MPI and BLAS
        _rpath_dirs.append(lib_dir)
        extra_link_args.append(f'-Wl,--rpath,{lib_dir}')
print(f"RPATH will be set to: {library_dirs}")

# Runtime search order matters: conda's Python injects -Wl,-rpath,$CONDA_PREFIX/lib
# into LDSHARED/LDCXXSHARED, and setuptools places those flags *before* the ones
# built above. A conda-installed library therefore wins over an explicitly
# requested one -- BLAS_LIB_PATH gets honored at link time and silently ignored at
# run time, which is how you end up running conda's generic OpenBLAS while
# believing you selected a tuned build.
#
# Demote conda: drop its rpath entries (keeping its -L, so link-time discovery of
# conda-provided libraries still works) and re-add the directory last, as a
# fallback behind anything the caller asked for.
_conda_prefix = os.environ.get('CONDA_PREFIX')
if _conda_prefix:
    _conda_lib = os.path.join(_conda_prefix, 'lib')
    _scfg = sysconfig.get_config_vars()
    _rpath_re = re.compile(r'-Wl,-rpath(?:-link)?,' + re.escape(_conda_lib) + r'(?=\s|$)')
    for _key in ('LDSHARED', 'LDCXXSHARED'):
        _val = _scfg.get(_key)
        if isinstance(_val, str):
            _scfg[_key] = re.sub(r' +', ' ', _rpath_re.sub('', _val)).strip()
    if _conda_lib not in _rpath_dirs:   # skip if already requested explicitly
        _rpath_dirs.append(_conda_lib)
        extra_link_args.append(f'-Wl,--rpath,{_conda_lib}')
        print(f"RPATH fallback appended last: {_conda_lib}")

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
# All three may now be installed side by side. They used to be kept apart on
# the theory that they link different OpenMP runtimes; that is not what ldd
# shows -- when NVHPC is present every extension is compiled by nvc++ and all
# three link libnvomp. The real hazard was that _core_cpu, built without
# -mp=gpu, leaves that shared runtime initialised host-only, after which the
# OMP-offload backend cannot acquire a device and silently runs its target
# regions on the host (correct energies, exit 0, GPU still reported). Since the
# Python package now imports backends lazily -- one per process, on first use --
# co-resident .so files no longer interfere, so `auto` builds everything the
# toolchain supports.
build_backend = os.environ.get('SBD_BUILD_BACKEND', 'auto').lower()

build_cpu = False
build_gpu_thrust = False
build_gpu_omp_offload = False

if build_backend in ('auto', 'all'):
    build_cpu = True
    build_gpu_thrust = has_nvhpc
    build_gpu_omp_offload = has_nvhpc
    if has_nvhpc:
        print("\nAuto-detected nvc++ - will build CPU, Thrust GPU and "
              "OMP-offload GPU backends")
    else:
        print("\nnvc++ not found - will build CPU backend only")
    if build_backend == 'all' and not has_nvhpc:
        print("Error: SBD_BUILD_BACKEND=all requires NVHPC_HOME / nvc++.")
        sys.exit(1)
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
    build_gpu_omp_offload = True
    print("\nBuilding GPU OpenMP target-offload backend only "
          "(SBD_BUILD_BACKEND=gpu_omp_offload)")
    if not has_nvhpc:
        print("Error: gpu_omp_offload requires NVHPC_HOME / nvc++.")
        sys.exit(1)
else:
    print(f"Error: Invalid SBD_BUILD_BACKEND='{build_backend}'")
    print("Valid values: auto (= all backends the toolchain supports), all, "
          "cpu, gpu (alias gpu_thrust), both, gpu_omp_offload")
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
    gpu_arch = _resolve_gpu_arch()
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
    offload_arch = _resolve_gpu_arch()
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
