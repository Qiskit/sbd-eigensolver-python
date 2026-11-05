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

"""``cpu_subspace`` on the Thrust backend: the subspace vectors in host memory.

Upstream SBD's Thrust builds keep the Davidson (and TPB's Lanczos) subspace vectors in
GPU memory by default; ``cpu_subspace`` keeps them in host memory instead, applying H on
the GPU through two GPU buffers. Upstream documents the energies as agreeing with the
default to within rounding, which is what is checked here, for both solvers. The field
exists on the Thrust backend only, as upstream declares it only under ``SBD_THRUST``.
"""

from __future__ import annotations

import itertools

import pytest

import sbd

# One word per determinant, as the rest of the suite's in-process h2o cases (SBD fixes the
# word count for the life of the process, so a different one here breaks the tests after it).
BIT_LENGTH = 62
ALPHA_LIMIT = 16


def _thrust(backend):
    if not backend.__name__.endswith("_core_gpu_thrust"):
        pytest.skip("cpu_subspace exists on the gpu (Thrust) backend only")


def _alpha(data_dir):
    return (data_dir / "h2o" / "h2o-1em3-alpha.txt").read_text().split()[:ALPHA_LIMIT]


def _interleave(alpha, beta):
    a, b = alpha[::-1], beta[::-1]
    return "".join(a[i] + b[i] for i in range(len(a)))[::-1]


@pytest.mark.parametrize("solver", ["tpb", "gdb"])
def test_cpu_subspace_gives_the_default_energy(backend, data_dir, solver):
    _thrust(backend)
    fcidump = sbd.LoadFCIDump(str(data_dir / "h2o" / "fcidump.txt"))
    norb = int(fcidump.header["NORB"])
    alpha = _alpha(data_dir)

    def solve(cpu_subspace):
        if solver == "tpb":
            config = backend.TPB_SBD()
            config.eps, config.max_it, config.max_nb, config.bit_length = 1e-10, 200, 8, BIT_LENGTH
            config.cpu_subspace = cpu_subspace
            half = backend.sort_bitarray(
                [backend.from_string(s, BIT_LENGTH, norb) for s in alpha])
            return sbd.tpb_diag(fcidump, half, half, config)["energy"]
        config = backend.GDB_SBD()
        config.eps, config.max_it, config.max_nb, config.bit_length = 1e-10, 200, 8, BIT_LENGTH
        config.cpu_subspace = cpu_subspace
        dets = sbd.sort_bitarray_array(sbd.from_strings(
            [_interleave(a, b) for a, b in itertools.product(alpha, alpha)],
            BIT_LENGTH, 2 * norb))
        return sbd.gdb_diag(fcidump, dets, config)["energy"]

    assert solve(True) == pytest.approx(solve(False), abs=1e-9)


@pytest.mark.parametrize("config", ["TPB_SBD", "GDB_SBD"])
def test_cpu_subspace_is_on_the_thrust_backend_only(backend, config):
    has = hasattr(getattr(backend, config)(), "cpu_subspace")
    assert has is backend.__name__.endswith("_core_gpu_thrust")


# --- the drivers' --cpu_subspace ------------------------------------------------------

import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
H2O = REPO / "vendor" / "sbd-upstream" / "data" / "h2o"
DRIVERS = {
    "tpb/run_sbd_diag.py": ["--adetfile", str(H2O / "h2o-1em3-alpha.txt"), "--bit_length", str(BIT_LENGTH)],
    "gdb/run_gdb_diag.py": ["--from-alpha", str(H2O / "h2o-1em3-alpha.txt"),
                            "--alpha-limit", str(ALPHA_LIMIT), "--bit_length", str(BIT_LENGTH)],
}


def _driver(script, flags, threads):
    path = REPO / "examples" / script
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env["OMP_NUM_THREADS"] = threads
    return subprocess.run([sys.executable, path.name, "--device", "gpu", "--fcidump",
                           str(H2O / "fcidump.txt"), "--tolerance", "1e-8", *DRIVERS[script],
                           *flags], cwd=path.parent, capture_output=True, text=True,
                          timeout=600, env=env)


def _energy(stdout):
    return float(re.search(r"Ground state energy:\s*(-?\d+\.\d+)", stdout).group(1))


@pytest.mark.parametrize("script", sorted(DRIVERS))
def test_the_drivers_cpu_subspace_flag(backend, script):
    """--cpu_subspace 1 gives the default energy, and needs more than one thread."""
    _thrust(backend)
    default = _driver(script, [], "2")
    host = _driver(script, ["--cpu_subspace", "1"], "2")
    assert default.returncode == 0 and host.returncode == 0, host.stderr[-2000:]
    assert _energy(host.stdout) == pytest.approx(_energy(default.stdout), abs=1e-8)
    one_thread = _driver(script, ["--cpu_subspace", "1"], "1")
    assert one_thread.returncode != 0
    assert "OMP_NUM_THREADS" in one_thread.stdout + one_thread.stderr
