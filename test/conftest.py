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

"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The reference molecules live in the vendored upstream SBD checkout, which is a git
# submodule. It is not present in an sdist or a fresh clone until the submodule is
# initialized, so tests that need it are skipped rather than failing.
DATA_DIR = Path(__file__).resolve().parents[1] / "vendor" / "sbd-upstream" / "data"

# The curated h2o counts shipped with the examples. Unlike the reference data above this
# is part of the repository proper, so it is always present.
COUNTS_PATH = (
    Path(__file__).resolve().parents[1] / "python" / "examples" / "count_dict_h2o.json"
)


# Slow tests are opt-in through a command-line flag rather than excluded by default,
# so that they are reported as skipped with a reason instead of silently deselected.
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option


# pylint: disable=missing-function-docstring
def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        marker = pytest.mark.skip(reason="skipping slow test, as --run-slow was not provided")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(marker)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to the vendored reference data, skipping the test if it is absent."""
    if not DATA_DIR.is_dir():
        pytest.skip(
            f"reference data not found at {DATA_DIR}; "
            "run 'git submodule update --init --recursive'"
        )
    return DATA_DIR


def pytest_report_header():
    """Record which backends were compiled, and which of them is under test.

    Without this, a run gives no indication of what it actually exercised: setup.py
    builds the GPU backends only when nvc++ is present, so the same command tests CPU
    only on one machine and CPU plus GPU on another. A passing run should say which.
    """
    import sbd

    available = sbd.available_backends()
    requested = os.environ.get("SBD_TEST_DEVICE") or "default"
    return f"sbd {sbd.__version__}: backends built {available}, testing {requested}"


def _requested_device() -> str | None:
    """``SBD_TEST_DEVICE``, or ``None`` to mean whatever the build defaults to.

    A backend named there but not compiled into this build is reported as a skip, since
    which backends exist depends on how the package was built. That a backend is missing
    is a fact about the build; that ``sbd`` itself is missing is a failure, and is left
    to raise.
    """
    import sbd

    device = os.environ.get("SBD_TEST_DEVICE")
    available = sbd.available_backends()
    if device is not None and device not in available:
        pytest.skip(f"backend {device!r} was not built; available: {available}")
    return device


@pytest.fixture(scope="session")
def backend():
    """The SBD backend module to test, for tests calling the extension directly."""
    import sbd

    return sbd.get_backend(_requested_device())


@pytest.fixture(scope="session")
def device_config():
    """The same selection as ``backend``, in the form the solver wrappers take.

    ``solve_sci`` and ``solve_sci_batch`` accept a ``DeviceConfig`` rather than a backend
    module -- they resolve the module themselves from its ``device`` key -- so the two
    fixtures exist to hand each layer the type it expects, off one shared setting.

    ``None`` when nothing is requested is deliberate: it leaves the wrapper's own backend
    resolution in the path, which is what an ordinary caller gets.
    """
    from sbd.device_config import DeviceConfig

    device = _requested_device()
    if device is None:
        return None
    # The generic constructor accepts every device key, including hyphenated ones like
    # 'gpu-omp' that do not correspond to a classmethod name.
    return DeviceConfig(device=device)


@pytest.fixture(scope="session")
def counts_path() -> Path:
    """Path to the curated h2o counts file used by the examples."""
    if not COUNTS_PATH.is_file():
        pytest.skip(f"counts file not found at {COUNTS_PATH}")
    return COUNTS_PATH
