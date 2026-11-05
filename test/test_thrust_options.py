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

"""The Thrust-only TPB options reach SBD as given, whichever way they are set.

``use_precalculated_dets``, ``max_memory_gb_for_determinants`` and
``thrust_collapse_loops`` exist on the Thrust backend only. They are set through
``sbd_config``; ``DeviceConfig`` used to carry its own copies of the first two and
apply them after ``sbd_config``, which reset whatever the caller had set there, and
with it both SQD drivers' flags. These check the precedence, the deprecated
``DeviceConfig`` route, what the two SQD drivers hand SBD, and that
``thrust_collapse_loops`` does not change the energy.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import warnings

import pytest

import sbd
from sbd import sbd_solver
from sbd.device_config import DeviceConfig

REPO = pathlib.Path(__file__).resolve().parents[1]
FIELDS = ("use_precalculated_dets", "max_memory_gb_for_determinants",
          "thrust_collapse_loops")
NON_DEFAULT = {"use_precalculated_dets": False, "max_memory_gb_for_determinants": 7,
               "thrust_collapse_loops": False}
# One word per determinant, as the rest of the suite's in-process h2o cases.
BIT_LENGTH = 63


def _thrust(backend):
    if not backend.__name__.endswith("_core_gpu_thrust"):
        pytest.skip("these fields exist on the gpu (Thrust) backend only")


def _fields(config):
    return {name: getattr(config, name) for name in FIELDS}


def test_the_fields_are_on_the_thrust_backend_only(backend):
    config = backend.TPB_SBD()
    on_thrust = backend.__name__.endswith("_core_gpu_thrust")
    assert all(hasattr(config, name) is on_thrust for name in FIELDS)


def test_the_defaults_are_upstreams(backend):
    _thrust(backend)
    assert _fields(sbd_solver._create_sbd_config(None, backend, DeviceConfig.gpu())) == {
        "use_precalculated_dets": True, "max_memory_gb_for_determinants": -1,
        "thrust_collapse_loops": True}


@pytest.mark.parametrize("device_config", [None, "gpu()", "gpu-key"])
def test_sbd_config_reaches_sbd(backend, device_config):
    """With or without a DeviceConfig, sbd_config's values are what SBD gets."""
    _thrust(backend)
    device_config = {None: None, "gpu()": DeviceConfig.gpu(),
                     "gpu-key": DeviceConfig(device="gpu")}[device_config]
    config = sbd_solver._create_sbd_config(dict(NON_DEFAULT), backend, device_config)
    assert _fields(config) == NON_DEFAULT


def test_the_deprecated_device_config_arguments_still_apply(backend):
    _thrust(backend)
    with pytest.warns(DeprecationWarning, match="sbd_config"):
        device_config = DeviceConfig.gpu(use_precalculated_dets=False, max_memory_gb=5)
    config = sbd_solver._create_sbd_config(None, backend, device_config)
    assert (config.use_precalculated_dets, config.max_memory_gb_for_determinants) == (False, 5)
    # ... but sbd_config takes precedence over them.
    config = sbd_solver._create_sbd_config(
        {"use_precalculated_dets": True, "max_memory_gb_for_determinants": 9},
        backend, device_config)
    assert (config.use_precalculated_dets, config.max_memory_gb_for_determinants) == (True, 9)


@pytest.mark.parametrize("make", [
    lambda: DeviceConfig(device="gpu", max_memory_gb=5),
    lambda: DeviceConfig.gpu(max_memory_gb=5),
    lambda: DeviceConfig.gpu_omp(max_memory_gb=5),
    lambda: DeviceConfig.gpu_nvidia_omp(max_memory_gb=5),
])
def test_the_deprecation_warning_names_the_callers_line(make):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make()
    deprecations = [w for w in caught if "sbd_config" in str(w.message)]
    assert len(deprecations) == 1 and deprecations[0].filename == __file__


def test_a_device_config_without_them_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DeviceConfig.gpu(), DeviceConfig.gpu_omp(), DeviceConfig.cpu()
        DeviceConfig(device="gpu")


def test_collapse_loops_does_not_change_the_energy(backend, data_dir):
    _thrust(backend)
    fcidump = sbd.LoadFCIDump(str(data_dir / "h2o" / "fcidump.txt"))
    norb = int(fcidump.header["NORB"])
    alpha = (data_dir / "h2o" / "h2o-1em3-alpha.txt").read_text().split()[:16]
    half = backend.sort_bitarray([backend.from_string(s, BIT_LENGTH, norb) for s in alpha])

    def energy(collapse):
        config = backend.TPB_SBD()
        config.eps, config.max_it, config.max_nb, config.bit_length = 1e-10, 200, 8, BIT_LENGTH
        config.use_precalculated_dets, config.thrust_collapse_loops = True, collapse
        return sbd.tpb_diag(fcidump, half, half, config)["energy"]

    assert energy(False) == pytest.approx(energy(True), abs=1e-9)


# --- what the two SQD drivers hand SBD -----------------------------------------------

# Runs a driver with _create_sbd_config wrapped, prints the first config it builds, and
# stops there: what is under test is the config, not the solve.
SPY = r"""
import runpy, sys
from sbd import sbd_solver
fields = ("use_precalculated_dets", "max_memory_gb_for_determinants",
          "thrust_collapse_loops")
real = sbd_solver._create_sbd_config
def spy(*args, **kwargs):
    config = real(*args, **kwargs)
    print("CONFIG " + repr({name: getattr(config, name) for name in fields}), flush=True)
    raise SystemExit(0)
sbd_solver._create_sbd_config = spy
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
"""

DRIVERS = ("run_sqd_sbd.py", "run_sqd_enlarge_subspace_sbd.py")


def _driver_config(script, flags):
    tpb = REPO / "examples" / "tpb"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PMI_", "PMIX_", "OMPI_"))}
    env.setdefault("OMP_NUM_THREADS", "2")
    done = subprocess.run(
        [sys.executable, "-c", SPY, script, "--device", "gpu",
         "--fcidump", str(REPO / "vendor" / "sbd-upstream" / "data" / "h2o" / "fcidump.txt"),
         "--counts", "count_dict_h2o.json", "--max_iterations", "1", *flags],
        cwd=tpb, capture_output=True, text=True, timeout=600, env=env)
    lines = [l for l in done.stdout.splitlines() if l.startswith("CONFIG ")]
    assert lines, f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    return eval(lines[0][len("CONFIG "):])


@pytest.mark.parametrize("script", DRIVERS)
def test_the_sqd_drivers_flags_reach_sbd(backend, script):
    _thrust(backend)
    assert _driver_config(script, []) == {
        "use_precalculated_dets": True, "max_memory_gb_for_determinants": -1,
        "thrust_collapse_loops": True}
    assert _driver_config(script, [
        "--sbd_use_precalculated_dets", "0", "--sbd_max_memory_gb_for_determinants", "7",
        "--sbd_thrust_collapse_loops", "0"]) == NON_DEFAULT
