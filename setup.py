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
import platform
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


def _mpi_prefix_from_env_prefix():
    """MPI installed inside the active Python environment.

    Covers conda (`conda install mpich`/`openmpi`) and the PyPI `mpich`/`openmpi`
    wheels, both of which drop mpi.h and libmpi straight into the environment
    prefix. Needed because mpi4py's linkage is not always readable: a conda-forge
    mpi4py ships one extension per MPI flavour (MPI.mpich.*, MPI.openmpi.*) so
    linkage cannot say which is active, and on macOS the install name is
    @rpath-relative. In both cases the prefix answers the question directly.

    This is a heuristic, not authoritative like mpi4py's own linkage, so callers
    let MPI_HOME override it silently rather than treating a difference as a
    conflict.
    """
    for prefix in (sys.prefix, getattr(sys, 'base_prefix', sys.prefix)):
        if prefix and os.path.exists(os.path.join(prefix, 'include', 'mpi.h')):
            lib = os.path.join(prefix, 'lib')
            if os.path.isdir(lib) and any(n.startswith('libmpi') for n in os.listdir(lib)):
                return prefix
    return None


def _mpi_config_from_mpicc():
    """Probe the mpicc wrapper for include/library/link flags.

    Needed because a prefix without $prefix/include/mpi.h is not necessarily the
    wrong MPI: distros that split MPI into a -devel package put the headers
    elsewhere (Debian's Open MPI uses /usr/lib/<triple>/openmpi/include), and
    mpicc is the thing that knows where its own headers live.

    `-show` prints the whole command line and is understood by both Open MPI and
    MPICH -- checked against openmpi 5.0.10 and mpich 5.0.1. Open MPI's
    `--showme:compile` is deliberately not tried: it yields nothing `-show` does
    not, and MPICH's wrapper treats it as a source file and tries to compile it.

    Returns (include_dirs, library_dirs, libraries), or None if mpicc is absent or
    prints no usable flags.
    """
    try:
        tokens = subprocess.check_output(
            ['mpicc', '-show'], universal_newlines=True,
            stderr=subprocess.DEVNULL).split()
    except Exception:
        return None

    inc, lib, libs = [], [], []
    for token in tokens:
        for prefix, acc in (('-I', inc), ('-L', lib), ('-l', libs)):
            if token.startswith(prefix) and token[2:] and token[2:] not in acc:
                acc.append(token[2:])       # MPICH repeats -I/-L; collapse them
                break
    if not (inc or lib):
        return None
    # Open MPI names -lmpi, MPICH -lmpi -lpmpi; take what the wrapper says.
    return inc, lib, libs or ['mpi']


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
    # Only mpi4py's own linkage is authoritative enough to contradict MPI_HOME.
    from_prefix = None if derived else _mpi_prefix_from_env_prefix()
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

    prefix = mpi_home or derived or from_prefix
    if prefix:
        include_dir = os.path.join(prefix, 'include')
        lib_dir = next((os.path.join(prefix, d) for d in ('lib', 'lib64')
                        if os.path.isdir(os.path.join(prefix, d))),
                       os.path.join(prefix, 'lib'))
        source = ('MPI_HOME' if mpi_home else
                  'mpi4py' if derived else 'the environment prefix')
        if os.path.exists(os.path.join(include_dir, 'mpi.h')):
            print(f"Using MPI from {source}: {prefix}")
            return [include_dir], [lib_dir], ['mpi']
        # Same MPI, unusual layout -- ask mpicc before giving up on it.
        print(f"Notice: no mpi.h under {prefix} (from {source}); asking mpicc.")
        mpicc_config = _mpi_config_from_mpicc()
        if mpicc_config is not None:
            print("Using MPI detected from mpicc")
            return mpicc_config
        print(f"Warning: {include_dir}/mpi.h not found and mpicc unusable; "
              "trying the prefix anyway. Check MPI_HOME points at an MPI "
              "*prefix*, not its lib or bin directory.")
        return [include_dir], [lib_dir], ['mpi']

    mpicc_config = _mpi_config_from_mpicc()
    if mpicc_config is not None:
        print("Using MPI detected from mpicc")
        return mpicc_config

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


def _amdgpu_arch(compiler):
    """Ask ROCm which GPU this machine has, e.g. 'gfx90a'. None if it cannot.

    Uses the amdgpu-arch that ships BESIDE the chosen compiler, never one found
    on PATH. ROCm installs versioned trees side by side and also exposes
    /usr/bin/amdgpu-arch via alternatives, so a PATH lookup can easily report
    the arch from a different ROCm than the one doing the compiling. Note that
    amdgpu-arch lives only in lib/llvm/bin, not in the prefix's bin, so resolve
    the compiler symlink before looking next to it.

    Prints one line per GPU, so dedupe. Returns None on a GPU-less build host
    (a container stage, a login node), which the caller turns into a request to
    set SBD_GPU_ARCH explicitly.
    """
    here = os.path.dirname(os.path.realpath(compiler))
    probe = os.path.join(here, 'amdgpu-arch')
    if not os.path.exists(probe):
        return None
    try:
        out = subprocess.check_output([probe], universal_newlines=True,
                                      stderr=subprocess.DEVNULL, timeout=30)
    except Exception:
        return None
    arches = sorted({ln.strip() for ln in out.splitlines() if ln.strip()})
    return ','.join(arches) or None


def _gpu_arch_flags(vendor, arch):
    """Compiler flags that pin the GPU architecture, for compile AND link.

    Returned as a list so an unset arch contributes nothing at all.

    The two vendors spell this differently, and AMD cannot take a comma list:
    nvc++ accepts -gpu=cc80,cc90,cc100 as one flag, while clang wants a repeated
    --offload-arch=. Both MUST be passed at link as well as compile -- see the
    long note on the Thrust extension below for what silently goes missing
    otherwise.
    """
    if not arch:
        return []
    if vendor == 'amd':
        return [f'--offload-arch={a}' for a in arch.split(',') if a]
    return [f'-gpu={arch}']


def _resolve_gpu_arch(vendor='nvidia', compiler=None):
    """Return the GPU architecture to target, or None to let the toolchain pick.

    OPTIONAL BY DESIGN.

    On AMD the arch is auto-DETECTED rather than left implicit: unset, we ask
    ROCm's amdgpu-arch what this machine has (e.g. gfx90a for MI250X, gfx942 for
    MI300X) and pin that. amdclang++ has no useful built-in default the way
    nvc++ does, so if detection fails -- a build host with no GPU -- SBD_GPU_ARCH
    becomes mandatory and the build stops with that message. Several
    architectures can be named at once (gfx90a,gfx942); each becomes its own
    --offload-arch flag. As on NVIDIA there is no JIT fallback, so an
    architecture that is not listed will not run.

    Verify what actually landed rather than trusting the flag:

        llvm-objdump --offloading <the built _core_gpu_omp_offload*.so>

    The NVIDIA contract is unchanged:

    * Set (e.g. ``cc90``, or ``cc80,cc90,cc100`` for a portable
      multi-architecture binary): honored exactly, and passed at BOTH compile
      and link. Passing it at link is not optional -- the device-link step is
      where the final SASS is generated, so a value given only at compile time
      is discarded and you silently get the toolchain's own target instead.
    * Unset: no ``-gpu=`` flag is emitted at all and nvc++ targets the GPU of
      the machine the toolchain was installed on. That is the right default for
      a local build on the machine you will run on, and it keeps a plain
      ``pip install`` working with no environment setup.

    Build a multi-architecture binary whenever the artifact travels -- a
    container image, a shared filesystem, a wheel for a mixed cluster. These
    builds embed no PTX, so an architecture that is not listed has no JIT
    fallback: it simply will not run. ``ccall-major`` covers one target per
    major generation if you would rather not enumerate.

    Verify what actually landed rather than trusting the flag:

        cuobjdump --list-elf <the built _core_gpu_*.so>

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
              "now that the LLVM path is gone). Honoring it as a back-compat "
              "alias. Please switch to SBD_GPU_ARCH.")
        return legacy

    if vendor == 'amd':
        detected = _amdgpu_arch(compiler) if compiler else None
        if detected:
            print(f"Notice: SBD_GPU_ARCH is not set; detected {detected} via "
                  "amdgpu-arch and\n"
                  "        targeting exactly that. If this artifact will run "
                  "anywhere else --\n"
                  "        a container image, a shared filesystem, a mixed-GPU "
                  "cluster -- set\n"
                  "        SBD_GPU_ARCH to every architecture you need, e.g. "
                  "gfx90a,gfx942.\n"
                  "        There is no JIT fallback, so an unlisted "
                  "architecture cannot run.")
            return detected
        # The caller turns this into a hard error. Not a silent pass: unlike
        # nvc++, amdclang++ has no built-in default worth inheriting, so a build
        # with no arch at all produces a module that runs nowhere.
        print("Notice: SBD_GPU_ARCH is not set and amdgpu-arch could not "
              "report an\n"
              "        architecture (normal on a build host with no AMD GPU).")
        return None

    print("Notice: SBD_GPU_ARCH is not set; letting nvc++ target the GPU of the\n"
          "        machine this toolchain was installed on. Fine for a local\n"
          "        build. If this artifact will run anywhere else -- a container\n"
          "        image, a shared filesystem, a mixed-GPU cluster -- set it to\n"
          "        every architecture you need, e.g.\n"
          "          SBD_GPU_ARCH=cc80,cc90,cc100   (A100 / H100 / GB200-B200)\n"
          "          SBD_GPU_ARCH=ccall-major       (one per major generation)\n"
          "        No PTX is embedded, so an unlisted architecture cannot JIT.")
    return None


def _route_build_through_gpu_compiler(gpu_path, vendor='nvidia'):
    """Configure distutils + sysconfig so a setup() call uses the GPU compiler.

    Called by the extension blocks that need it: Thrust and OMP-offload on
    NVIDIA (both nvc++), OMP-offload on AMD (amdclang++). Idempotent — second
    call is a no-op.

    Effect: distutils' UnixCCompiler will pick up CC/CXX/LDSHARED from
    os.environ and use them for every Extension in this setup() call.
    Also clears CFLAGS/CXXFLAGS/CPPFLAGS and rewrites sysconfig to drop
    tokens the chosen compiler rejects (see the two lists below).

    Co-builds with the CPU extension are safe under either vendor: both nvc++
    and amdclang++ accept the CPU block's `-fopenmp -O3 -std=c++17` (nvc++
    treats -fopenmp as -mp; amdclang++ IS clang, so it takes them natively).
    """
    if os.environ.get('_SBD_GPU_ROUTING_APPLIED'):
        return
    os.environ['_SBD_GPU_ROUTING_APPLIED'] = '1'

    # Respect user-set CC/CXX (e.g. cross-toolchain); otherwise pin the GPU one.
    os.environ.setdefault('CC',       gpu_path)
    os.environ.setdefault('CXX',      gpu_path)
    os.environ.setdefault('LDSHARED', f'{gpu_path} -shared')
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
    #
    # amdclang++ needs FAR less scrubbing, because it is clang and accepts the
    # gcc spellings. Measured against ROCm 10.0 / AMD clang 23 with
    # --offload-arch=gfx90a, every token above compiles clean EXCEPT
    # -fcf-protection, which is rejected as "option 'cf-protection=return'
    # cannot be specified on this target" -- the flag is applied to the amdgcn
    # device pass too, and there it is meaningless. -march=x86-64-v2 is fine for
    # clang and is deliberately NOT rewritten to v3 here; that rewrite exists
    # only because nvc++ requires v3+.
    _cfg = sysconfig.get_config_vars()
    if vendor == 'amd':
        _strip_tokens = (
            '-fcf-protection',
        )
    else:
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
        # nvc++ only: it rejects x86-64-v2 and requires v3+. clang accepts v2,
        # so leave it alone there rather than silently raising the CPU baseline
        # of the AMD build above what the caller's Python asked for.
        if vendor != 'amd':
            _v = _v.replace('-march=x86-64-v2', '-march=x86-64-v3')
        # conda's Python bakes '-B $CONDA_PREFIX/compiler_compat' into
        # CC/CXX/LDSHARED/LDCXXSHARED. nvc++ rejects -B and hands the
        # path to the linker as an input file, so drop just that flag
        # and keep conda's -L/-rpath entries intact.
        #
        # Dropped for amdclang++ too, for a different reason: clang accepts -B
        # perfectly well, but that directory holds conda's own (old) `ld`, and
        # the offload link runs through clang-linker-wrapper -> ld.lld. Letting
        # -B redirect the linker there invites a mismatch for no benefit.
        _v = re.sub(r'-B\s*\S*compiler_compat\S*', '', _v)
        _cfg[_k] = re.sub(r' +', ' ', _v).strip()


def _homebrew_prefix():
    """Homebrew's install prefix, or None if brew is not on PATH.

    Not a constant: it is /opt/homebrew on Apple silicon and /usr/local on
    Intel, so either one hardcoded is wrong on the other architecture.
    """
    import shutil
    brew = shutil.which('brew')
    if not brew:
        return None
    try:
        return subprocess.check_output([brew, '--prefix'],
                                       universal_newlines=True).strip()
    except Exception:
        return None


def _resolve_darwin_cxx():
    """(path, version_line) of the C++ compiler this build will actually use.

    distutils takes CC/CXX from the environment, else from sysconfig -- where
    conda records a bare 'clang++' that is resolved through PATH, so a Homebrew
    LLVM silently wins over both Apple clang and a conda toolchain. Knowing
    which one it is decides where OpenMP comes from, so this is resolved before
    the libomp search rather than merely reported afterwards.
    """
    import shutil
    cxx = (os.environ.get('CXX') or sysconfig.get_config_var('CXX')
           or 'clang++').split()[0]
    path = shutil.which(cxx) or cxx
    try:
        version = subprocess.check_output(
            [path, '--version'], universal_newlines=True,
            stderr=subprocess.STDOUT).splitlines()[0]
    except Exception:
        version = '(version unknown)'
    return path, version


def _compiler_openmp(cxx_path):
    """(include_dir, lib_dir) of an OpenMP shipped with cxx_path, or None.

    An LLVM that builds the openmp runtime -- Homebrew's llvm formula, and the
    conda-forge clang packages -- carries a libomp.dylib in its own tree, and
    `-fopenmp` links THAT copy. Adding a second libomp from elsewhere then puts
    two same-named runtimes in one process, which aborts at the first parallel
    region with "OMP: Error #15". So when the compiler brings its own, that is
    the one to build against.

    Do not guess the layout. omp.h is installed into the clang RESOURCE
    directory (lib/clang/<ver>/include), not <prefix>/include -- Homebrew's own
    formula test compiles `#include <omp.h>` with no -I at all -- while
    libomp.dylib does land in <prefix>/lib. Ask the driver for the resource
    directory rather than hardcoding a version number into the path.
    """
    if not cxx_path or not os.path.isabs(cxx_path):
        return None
    # Resolve symlinks first: CC/CXX is commonly Homebrew's opt/ alias
    # (/opt/homebrew/opt/llvm/bin/clang++), while -print-resource-dir answers
    # with the real Cellar path. Comparing or joining the two forms without
    # normalising invites mismatches.
    cxx_real = os.path.realpath(cxx_path)
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(cxx_real)), 'lib')
    have_lib = any(os.path.exists(os.path.join(lib_dir, name))
                   for name in ('libomp.dylib', 'libomp.a'))
    try:
        resource_dir = subprocess.check_output(
            [cxx_real, '-print-resource-dir'], universal_newlines=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception as exc:
        resource_dir = None
        print(f"Notice: {cxx_real} -print-resource-dir failed: {exc!r}")
    inc_dir = os.path.join(resource_dir, 'include') if resource_dir else None
    have_header = bool(inc_dir) and os.path.exists(os.path.join(inc_dir, 'omp.h'))
    # Say what was found either way: a silent None here sends the build to a
    # different OpenMP, which still compiles and still passes its own tests, and
    # only aborts once another OpenMP consumer shares the process.
    print(f"Darwin OpenMP probe: compiler={cxx_real}\n"
          f"                     lib_dir={lib_dir} libomp={have_lib}\n"
          f"                     resource_dir={resource_dir} omp.h={have_header}")
    if not (have_lib and have_header):
        return None
    return inc_dir, lib_dir


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


def find_rocm_toolchain():
    """Locate amdclang++ for the AMD OpenMP target-offload backend.

    Deliberately the same shape as find_nvidia_hpc_sdk(): an explicit env var,
    else PATH. ROCM_HOME is this project's knob, matching NVHPC_HOME, and its
    job is picking a specific ROCm on a node with several installed -- common,
    since ROCm ships side-by-side versioned trees.

    Neither variable is required. A PATH lookup already covers both the
    module-based case (`module add rocm/<ver>` prepends its bin) and a stock
    install (ROCm's packages leave amdclang++ in /usr/bin via alternatives), and
    not every cluster provides modules.

    amdclang++ IS LLVM clang -- ROCm ships it with the amdgcn OpenMP offload
    runtime and matching device libraries already built, so nothing has to be
    compiled from source to get offload working. Unlike the NVHPC branch there
    is no need to touch PATH: amdclang++ finds its device libraries relative to
    its own InstalledDir, so invoking it by absolute path is enough.
    """
    import shutil
    # ROCM_HOME (this project's knob, matching NVHPC_HOME) wins over ROCM_PATH
    # (ROCm's own variable, usually set by a module file).
    for var in ('ROCM_HOME', 'ROCM_PATH'):
        rocm_home = os.environ.get(var) or None
        if not rocm_home:
            continue
        # $ROCM_HOME/bin/amdclang++ is normally a symlink to the second path;
        # older layouts only have lib/llvm/bin.
        for rel in ('bin/amdclang++', 'lib/llvm/bin/amdclang++'):
            cand = os.path.join(rocm_home, rel)
            if os.path.exists(cand):
                print(f"Found ROCm at: {rocm_home} (via {var})")
                return cand, True
        print(f"Warning: {var} set to {rocm_home} but amdclang++ not found")
    amdcxx_path = shutil.which('amdclang++')
    if amdcxx_path:
        print(f"Found amdclang++ in PATH: {amdcxx_path}")
        return amdcxx_path, True
    return None, False


def detect_gpu_toolchain():
    """Pick the GPU toolchain to build with: ('nvidia'|'amd'|None, compiler).

    NVHPC is probed first only because it is the long-established path here; a
    machine with exactly one GPU toolchain installed gets that one either way.
    SBD_GPU_VENDOR forces the choice for the rare host carrying both (a build
    node serving a mixed cluster), where the auto-answer would otherwise be an
    accident of probe order.

    Note the asymmetry in what each vendor can build: nvc++ drives BOTH the
    Thrust and the OpenMP-offload backends, whereas ROCm drives OpenMP offload
    only. Upstream SBD's Thrust path is wired to nvc++ flags (-cuda, -gpu=), so
    there is no rocThrust configuration to build even though some HIP scaffolding
    exists upstream.
    """
    forced = (os.environ.get('SBD_GPU_VENDOR') or '').strip().lower()
    if forced not in ('', 'nvidia', 'amd', 'none'):
        print(f"Error: Invalid SBD_GPU_VENDOR={forced!r}. "
              "Valid values: nvidia, amd, none")
        sys.exit(1)
    if forced == 'none':
        print("SBD_GPU_VENDOR=none - skipping GPU toolchain detection")
        return None, None

    if forced != 'amd':
        nvcxx, ok = find_nvidia_hpc_sdk()
        if ok:
            return 'nvidia', nvcxx
        if forced == 'nvidia':
            print("Error: SBD_GPU_VENDOR=nvidia but nvc++ was not found. "
                  "Set NVHPC_HOME.")
            sys.exit(1)

    if forced != 'nvidia':
        amdcxx, ok = find_rocm_toolchain()
        if ok:
            return 'amd', amdcxx
        if forced == 'amd':
            print("Error: SBD_GPU_VENDOR=amd but amdclang++ was not found. "
                  "Set ROCM_HOME, or put ROCm's bin directory on PATH.")
            sys.exit(1)

    return None, None


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

# RPATH so libraries are found at runtime without LD_LIBRARY_PATH.
#
# `--rpath` is a GNU ld spelling that Apple's linker does not accept; it wants
# the single-dash `-rpath`. GNU ld understands `-rpath` too, so use that form
# on both platforms.
extra_link_args = ['-fopenmp']
_rpath_dirs = []
for lib_dir in library_dirs:
    if lib_dir not in _rpath_dirs:          # a dir may appear as both MPI and BLAS
        _rpath_dirs.append(lib_dir)
        extra_link_args.append(f'-Wl,-rpath,{lib_dir}')
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
        extra_link_args.append(f'-Wl,-rpath,{_conda_lib}')
        print(f"RPATH fallback appended last: {_conda_lib}")

# Detect the GPU toolchain. What it can build depends on the vendor:
#   NVIDIA (nvc++)        1. _core_gpu_thrust       (Thrust + CUDA,  nvc++ -cuda)
#                         2. _core_gpu_omp_offload  (OMP offload,  nvc++ -mp=gpu)
#   AMD (amdclang++)         _core_gpu_omp_offload  (OMP offload, --offload-arch)
#
# The OMP-offload backend is ONE module and ONE device string ('gpu-omp') for
# both vendors: it is the same bindings.cpp with the same USE_GPU +
# USE_OMP_OFFLOAD macros, just a different compiler driving it. A given install
# serves one GPU vendor -- no wheels are published, every install compiles on the
# target machine -- so a vendor-suffixed second device string would buy nothing
# and would undo the deprecation of gpu_nvidia_omp() in favour of gpu_omp().
# Which vendor a build targeted is recorded on the module as
# __sbd_offload_target__ (e.g. 'amdgcn-amd-amdhsa:gfx90a') so it stays
# introspectable.
gpu_vendor, gpu_compiler = detect_gpu_toolchain()
has_gpu_toolchain = gpu_compiler is not None
# Only NVHPC can build the Thrust backend: upstream SBD wires that path to nvc++
# flags (-cuda, -gpu=), with no rocThrust configuration.
has_nvhpc = gpu_vendor == 'nvidia'

# Determine which backends to build.
#   auto                  : cpu, plus both GPU backends when nvc++ is present
#   all                   : same as auto, but an error when nvc++ is missing
#   cpu                   : cpu only
#   gpu | gpu_thrust      : thrust GPU only
#   gpu_omp_offload       : OpenMP target offload only (nvc++ -mp=gpu)
# `both` (cpu + thrust) was removed: it is a strict subset of `auto`, and the
# reason to keep the GPU backends apart went away with lazy loading.
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
    build_gpu_omp_offload = has_gpu_toolchain
    if has_nvhpc:
        print("\nAuto-detected nvc++ - will build CPU, Thrust GPU and "
              "OMP-offload GPU backends")
    elif gpu_vendor == 'amd':
        # No Thrust here, so `auto` yields two backends rather than three.
        print("\nAuto-detected amdclang++ - will build CPU and OMP-offload GPU "
              "backends (Thrust is NVIDIA-only)")
    else:
        print("\nNo GPU compiler found - will build CPU backend only")
    if build_backend == 'all' and not has_gpu_toolchain:
        print("Error: SBD_BUILD_BACKEND=all requires a GPU toolchain "
              "(NVHPC_HOME / nvc++, or ROCM_HOME / amdclang++).")
        sys.exit(1)
elif build_backend == 'cpu':
    build_cpu = True
    print("\nBuilding CPU backend only (SBD_BUILD_BACKEND=cpu)")
elif build_backend in ('gpu', 'gpu_thrust'):
    build_gpu_thrust = True
    print(f"\nBuilding Thrust GPU backend only (SBD_BUILD_BACKEND={build_backend})")
    if gpu_vendor == 'amd':
        # Fail rather than warn: on AMD this is not a maybe-it-links situation,
        # there is no rocThrust configuration to build at all. Silently falling
        # back to CPU under a name that says 'gpu' is exactly the confusion the
        # AMD path is meant to remove.
        print("Error: the Thrust backend is NVIDIA-only (upstream wires it to "
              "nvc++ -cuda).\n"
              "       On AMD use SBD_BUILD_BACKEND=gpu_omp_offload, or leave it "
              "unset for CPU + OMP-offload.")
        sys.exit(1)
    if not has_nvhpc:
        print("Warning: nvc++ not found, GPU build may fail")
elif build_backend == 'gpu_omp_offload':
    build_gpu_omp_offload = True
    print("\nBuilding GPU OpenMP target-offload backend only "
          "(SBD_BUILD_BACKEND=gpu_omp_offload)")
    if not has_gpu_toolchain:
        print("Error: gpu_omp_offload requires a GPU toolchain: NVHPC_HOME / "
              "nvc++, or ROCM_HOME / amdclang++.")
        sys.exit(1)
else:
    print(f"Error: Invalid SBD_BUILD_BACKEND='{build_backend}'")
    print("Valid values: auto (= all backends the toolchain supports), all, "
          "cpu, gpu (alias gpu_thrust), gpu_omp_offload")
    sys.exit(1)

ext_modules = []

if build_cpu:
    print("\nConfiguring CPU backend (_core_cpu)")
    if platform.system() == 'Darwin':
        # Which compiler is used decides where OpenMP may come from, so resolve
        # it first. Printing it is also the difference between a reproducible
        # build and a mystery, since a bare 'clang++' from sysconfig is resolved
        # through PATH.
        _cxx_path, _cxx_ver = _resolve_darwin_cxx()
        print(f"Darwin C++ compiler: {_cxx_path}\n"
              f"                     {_cxx_ver}   (pin it with CC/CXX)")

        # macOS has no system OpenMP, so libomp comes from a package manager.
        #
        # Order matters, and it is about which libomp ends up in the PROCESS,
        # not merely which one satisfies the compile:
        #
        # 1. The compiler's own runtime, when it has one. `-fopenmp` links that
        #    copy no matter what else is on the link line, so naming a second
        #    libomp here is how you get two same-named runtimes in one process
        #    and an "OMP: Error #15" abort at the first parallel region.
        # 2. Otherwise the conda env, whose libraries are the ones actually
        #    LOADED at import time (resolved via the python executable's
        #    @loader_path/../lib).
        # 3. Otherwise Homebrew's standalone libomp, which is what Apple clang
        #    needs, having no OpenMP of its own.
        conda_prefix = os.environ.get('CONDA_PREFIX')
        compiler_omp = _compiler_openmp(_cxx_path)
        if compiler_omp:
            omp_inc, omp_lib = compiler_omp
            # BLAS is a separate question from OpenMP: the compiler tree has no
            # OpenBLAS, so keep taking that from conda or Homebrew.
            if conda_prefix and os.path.isdir(os.path.join(conda_prefix, 'lib')):
                openblas_lib = os.path.join(conda_prefix, 'lib')
            else:
                openblas_lib = os.path.join(_homebrew_prefix() or '/opt/homebrew',
                                            'opt', 'openblas', 'lib')
            print(f"Darwin: libomp from the compiler's own tree\n"
                  f"        headers {omp_inc}\n"
                  f"        library {omp_lib}\n"
                  f"        BLAS    {openblas_lib}")
        elif conda_prefix and os.path.exists(
                os.path.join(conda_prefix, 'include', 'omp.h')):
            omp_inc = os.path.join(conda_prefix, 'include')
            omp_lib = openblas_lib = os.path.join(conda_prefix, 'lib')
            print(f"Darwin: libomp and BLAS from conda env {conda_prefix}")
        else:
            # Ask brew for its prefix rather than assuming: it is /opt/homebrew
            # on Apple silicon and /usr/local on Intel, so either one hardcoded
            # is wrong on the other architecture. Keep /opt/homebrew as the
            # fallback for when brew is not on PATH, so the error below still
            # names a concrete path to look in.
            brew_prefix = _homebrew_prefix() or '/opt/homebrew'
            omp_inc = os.path.join(brew_prefix, 'opt', 'libomp', 'include')
            omp_lib = os.path.join(brew_prefix, 'opt', 'libomp', 'lib')
            openblas_lib = os.path.join(brew_prefix, 'opt', 'openblas', 'lib')
            if not os.path.exists(os.path.join(omp_inc, 'omp.h')):
                # Fail here with the fix, rather than 100 lines later with
                # "'omp.h' file not found" from the middle of a compile.
                print("Error: no OpenMP runtime found on this macOS host.\n"
                      f"       Looked beside {_cxx_path}, in $CONDA_PREFIX/include, "
                      f"and in {omp_inc}.\n"
                      "       Apple clang ships without OpenMP, so install one:\n"
                      "         conda install -c conda-forge llvm-openmp   (preferred)\n"
                      "         brew install libomp")
                sys.exit(1)
            print(f"Darwin: libomp and BLAS from Homebrew at {brew_prefix} "
                  "(no conda libomp found)")

        cpu_compile_args = [
            '-DSBD_TRADMODE',
            '-DOMPI_SKIP_MPICXX',
            '-std=c++17', '-Xpreprocessor', '-fopenmp', '-O3',
            '-Wno-sign-compare', '-Wno-unused-variable', '-fPIC',
            '-DSBD_MODULE_NAME=_core_cpu', f'-I{omp_inc}',
        ]
        cpu_inc = include_dirs + [omp_inc]
        cpu_lib_dirs = library_dirs + [omp_lib, openblas_lib]
        cpu_libs = libraries + ['omp']
        # Not extra_link_args: that carries a bare `-fopenmp`, which Apple
        # clang rejects at link time the same way it does when compiling.
        # Re-derive the rpath entries over the libomp/BLAS directories chosen
        # above, so those dylibs resolve at import time. Apple's linker wants
        # `-rpath`, not GNU ld's `--rpath`.
        cpu_link_args = [f'-L{d}' for d in (omp_lib, openblas_lib)]
        cpu_link_args += [f'-Wl,-rpath,{d}' for d in cpu_lib_dirs]
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
    _route_build_through_gpu_compiler(gpu_compiler, 'nvidia')
    gpu_arch = _resolve_gpu_arch('nvidia', gpu_compiler)
    # Emitted only when the user asked for a specific arch; otherwise omitted
    # entirely so nvc++ picks the build machine's GPU (see _resolve_gpu_arch).
    gpu_arch_flags = _gpu_arch_flags('nvidia', gpu_arch)
    print(f"NVHPC -gpu= arch: {gpu_arch} (set SBD_GPU_ARCH to override; "
          "nvc++ accepts cc<XX> and sm_<XX>)")
    # Stamped for the same reason as the offload backend: so a built module can
    # be asked what it targets. Unambiguously NVIDIA, but the architecture is
    # not, and an arch mismatch is the usual reason a module refuses to run.
    thrust_target = f'cuda:{gpu_arch or "toolchain-default"}'

    # DELIBERATE divergence from upstream, which passes
    #     -gpu=${SBD_GPU_ARCH},mem:unified,interceptdeallocations
    # (CMakeLists sbd_configure_diag, thrust branch). mem:unified puts device
    # allocations in managed memory. Measured on 8x H100, h2o-1em4 (2.38e6
    # determinants), 8 ranks on a 4x2 grid: Davidson went from 0.32 s to
    # 0.75-0.86 s over three runs -- about 2.5x slower -- for bit-identical
    # energies. Its plausible benefit is letting a non-GPU-aware MPI host-copy a
    # device pointer, and SBD_NON_CUDA_AWARE_MPI below covers that directly. So
    # the default stays the separate-memory build we have been validating, rather
    # than a slower one users would need to understand in order to opt out of.
    # It is still reachable without patching this file, since the whole value is
    # passed through:  SBD_GPU_ARCH=cc90,mem:unified,interceptdeallocations
    # Upstream's thrust branch also carries two MPI-safety options. Mirror them
    # as BUILD-TIME env vars of the same name -- they are -D defines compiled into
    # the extension, so they must be set before `pip install` and changing one
    # means rebuilding. Setting them at run time does nothing.
    #   SBD_NON_CUDA_AWARE_MPI=1        host-stage all MPI comm on device memory
    #   SBD_THRUST_SAFE_MPI_ALLREDUCE=1 just the allreduce (a subset of the above)
    # SBD_NON_CUDA_AWARE_MPI is the supported way to build for an MPI that cannot
    # address device memory, which is otherwise a hard requirement.
    # Deliberately NOT mirrored: SBD_USE_NVTX / SBD_USE_NCCL / SBD_USE_CUBLAS
    # (each needs link libraries we do not add) and SBD_COMPLEX (changes the
    # element type, so it is an API change rather than a flag).
    _thrust_opt_defines = []
    for _opt in ('SBD_NON_CUDA_AWARE_MPI', 'SBD_THRUST_SAFE_MPI_ALLREDUCE'):
        if (os.environ.get(_opt) or '').strip().lower() in ('1', 'on', 'true', 'yes'):
            _thrust_opt_defines.append(f'-D{_opt}')
    if _thrust_opt_defines:
        print("Thrust build-time options baked in: "
              + " ".join(d[2:] for d in _thrust_opt_defines))

    thrust_gpu_flags = list(gpu_arch_flags)

    gpu_thrust_ext = Extension(
        'sbd._core_gpu_thrust',
        ['python/bindings.cpp'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        extra_compile_args=[
            '-DSBD_THRUST',
            # NOT -DSBD_TRADMODE: upstream's option is "Traditional (non-Thrust)
            # CPU mode for tpb", default OFF, and its thrust branch does not set
            # it -- only the omp5 branch does. mult.h:17 defines a `mult`
            # overload unconditionally and SBD_TRADMODE merely adds a second one,
            # so this was compiling an overload the Thrust path never calls.
            '-mp',
            '-cuda',
            '-fast',
            '-Minfo=accel',
            '--diag_suppress=declared_but_not_referenced,set_but_not_used',
            '-fmax-errors=0',
            '-fPIC',
            *thrust_gpu_flags,
            *_thrust_opt_defines,
            '-DSBD_MODULE_NAME=_core_gpu_thrust',
            f'-DSBD_OFFLOAD_TARGET="{thrust_target}"',
        ],
        # NOTE: -cudalib (no value) makes nvc++ blanket-link every CUDA
        # library NVHPC ships, including math libs SBD never calls
        # (cublasmp, cusolverMp, cutensor, nvblas). On NVHPC 26.3 some of
        # those ship as dangling symlinks (.so name present but versioned
        # target missing), causing the link to fail with "cannot find
        # -lcublasmp" etc. SBD's GPU path only needs the CUDA runtime, so
        # explicitly link -lcudart instead.
        #
        # -gpu= MUST be repeated here, at LINK time. The compile step above
        # emits device code for every architecture in the list, but the
        # device-link step decides what actually lands in the fatbin, and
        # without -gpu= nvc++ keeps only its own built-in default and silently
        # discards the rest -- no warning, exit 0. The result is a .so that
        # runs only on whatever architecture that default happens to be.
        #
        # Measured on NVHPC 26.1 (aarch64) with SBD_GPU_ARCH=cc80,cc90,cc100:
        #     object after compile        sm_80 sm_90 sm_100
        #     .so linked without -gpu=    sm_100            <- two arches lost
        #     .so linked with    -gpu=    sm_80 sm_90 sm_100
        # The default is compiled into nvc++, not detected from the hardware,
        # so a GPU-less build host (a container stage, say) does not change it.
        #
        # The OMP-offload extension below already passes -gpu= at link, which is
        # why only the Thrust backend was affected: a multi-arch build appeared
        # to succeed and then failed on any GPU other than the build toolchain's
        # default -- observed as a Thrust-only failure on H100 from a fatbin
        # built for cc80,cc90,cc100. These builds embed no PTX, so there is no
        # JIT fallback to mask it.
        extra_link_args=extra_link_args + ['-mp', '-cuda', *thrust_gpu_flags,
                                           '-lcudart'],
    )
    ext_modules.append(gpu_thrust_ext)


if build_gpu_omp_offload:
    print("\nConfiguring GPU OpenMP target-offload backend (_core_gpu_omp_offload)")
    print(f"Using compiler: {gpu_compiler}  (vendor: {gpu_vendor})")
    # Auto-route the build through the GPU compiler + sanitize sysconfig flags.
    _route_build_through_gpu_compiler(gpu_compiler, gpu_vendor)
    offload_arch = _resolve_gpu_arch(gpu_vendor, gpu_compiler)
    offload_arch_flags = _gpu_arch_flags(gpu_vendor, offload_arch)

    if gpu_vendor == 'amd':
        print(f"ROCm offload arch: {offload_arch} "
              "(set SBD_GPU_ARCH to override, e.g. gfx90a or gfx90a,gfx942)")
        if not offload_arch:
            # No baked-in default exists for amdclang++, and an unpinned build
            # would produce a module that cannot run anywhere.
            print("Error: could not detect the AMD GPU architecture and "
                  "SBD_GPU_ARCH is not set.\n"
                  "       Set it explicitly, e.g. SBD_GPU_ARCH=gfx90a "
                  "(MI250X) or gfx942 (MI300X).\n"
                  "       `amdgpu-arch` on a machine with the target GPU "
                  "prints the right value.")
            sys.exit(1)
        offload_target = f'amdgcn-amd-amdhsa:{offload_arch}'
        # -fopenmp-offload-mandatory: refuse to emit a host fallback path at
        # COMPILE time. It pairs with the runtime OMP_TARGET_OFFLOAD=MANDATORY
        # that __init__.py sets before importing this backend; together they
        # make a device-less rank a loud failure rather than a silent host run
        # returning a plausible energy.
        #
        # No sbd_nvhpc_compat.h here: that shim exists because nvc++ lowers
        # __builtin_ffsl inside `declare target` to a host-only symbol. clang
        # lowers those builtins to device intrinsics natively, and the shim is
        # #ifdef __NVCOMPILER anyway, so including it would be a no-op.
        offload_compile_args = [
            '-O3', '-std=c++17', '-fPIC',
            '-fopenmp', '-fopenmp-targets=amdgcn-amd-amdhsa',
            *offload_arch_flags,
            '-fopenmp-offload-mandatory',
            '-DSBD_TRADMODE',
            '-DUSE_GPU',
            '-DUSE_OMP_OFFLOAD',
            '-DOMPI_SKIP_MPICXX',
            '-DSBD_MODULE_NAME=_core_gpu_omp_offload',
            f'-DSBD_OFFLOAD_TARGET="{offload_target}"',
            # Selects the AMD device-visibility variables in bindings.cpp, so
            # the NVIDIA build keeps reading CUDA_VISIBLE_DEVICES first exactly
            # as before.
            '-DSBD_OFFLOAD_VENDOR_AMD',
            # Upstream headers are template-heavy and noisy under clang; these
            # are style warnings in vendored code, not actionable here.
            '-Wno-sign-compare', '-Wno-unused-variable',
        ]
        offload_link_args = extra_link_args + [
            '-fopenmp', *offload_arch_flags,
        ]
    else:
        print(f"NVHPC -gpu= arch: {offload_arch} (set SBD_GPU_ARCH to override)")
        offload_target = f'nvptx64-nvidia-cuda:{offload_arch or "toolchain-default"}'
        offload_compile_args = [
            '-O3', '-std=c++17', '-fPIC',
            '-mp=gpu',
            *offload_arch_flags,
            '-Minfo=mp',
            '-DSBD_TRADMODE',
            '-DUSE_GPU',
            '-DUSE_OMP_OFFLOAD',
            '-DOMPI_SKIP_MPICXX',
            '-DSBD_MODULE_NAME=_core_gpu_omp_offload',
            f'-DSBD_OFFLOAD_TARGET="{offload_target}"',
            # Force-include nvc++ shim so __builtin_ffsl / __builtin_popcountl
            # inside #pragma omp declare target lower to portable inlines
            # rather than __blt_pgi_ffsl (host-only NVHPC symbol that nvlink
            # can't resolve from device code).
            #
            # This is a DELIBERATE divergence from upstream, which instead does
            # -D__builtin_ffsl=__ffsll and therefore has to add -cuda to the
            # omp5 build to expose that CUDA intrinsic. The shim needs no CUDA
            # in an OpenMP-offload translation unit, so -cuda stays off here.
            '-include', 'python/sbd_nvhpc_compat.h',
        ]
        offload_link_args = extra_link_args + [
            '-mp=gpu',
            *offload_arch_flags,
        ]

    gpu_omp_offload_ext = Extension(
        'sbd._core_gpu_omp_offload',
        ['python/bindings.cpp'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language='c++',
        extra_compile_args=offload_compile_args,
        extra_link_args=offload_link_args,
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
    print("  - OpenMP-offload GPU backend:     sbd._core_gpu_omp_offload"
          f"  ({offload_target})")
print()
