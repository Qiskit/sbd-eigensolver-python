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

"""A missing backend's error names only devices a user can pass.

The package's internal scan also lists the unrestricted builds ('cpu-uhf', 'gpu-uhf'),
which are not devices. Each test stands in a backend scan with the GPU backend missing,
so it runs on any install.
"""

from __future__ import annotations

import pytest

import sbd

_SCAN = ["cpu", "cpu-uhf"]
_WHY = "not built (no _core_gpu_thrust.*.so in the package directory)"


@pytest.fixture
def gpu_missing(monkeypatch):
    monkeypatch.setattr(sbd, "_usable_cache", list(_SCAN))
    monkeypatch.setattr(sbd, "_backend_errors", {"gpu": _WHY})
    monkeypatch.setattr(sbd, "_backends", {})
    monkeypatch.setattr(sbd, "_extension_path", lambda name: None)
    monkeypatch.setattr(sbd, "_initialized", False)
    for name in ("_default_device", "_comm_backend", "_comm_module", "_global_comm"):
        monkeypatch.setattr(sbd, name, getattr(sbd, name))


def _assert_lists_devices_only(message):
    assert "['cpu']" in message
    assert "uhf" not in message


def test_init_with_a_missing_device_lists_only_devices(gpu_missing):
    with pytest.raises(RuntimeError) as err:
        sbd.init(device="gpu")
    message = str(err.value)
    assert "Device 'gpu' was requested, but its backend is not usable" in message
    _assert_lists_devices_only(message)


def test_init_without_a_device_says_the_default_was_used(gpu_missing, monkeypatch):
    monkeypatch.setattr(sbd, "_usable_cache", ["gpu", "gpu-uhf"])
    monkeypatch.setattr(sbd, "_backend_errors", {"cpu": "not built"})
    with pytest.raises(RuntimeError) as err:
        sbd.init()
    message = str(err.value)
    assert "No device was given, and the default 'cpu' backend is not usable" in message
    assert "['gpu']" in message and "uhf" not in message


def test_a_backend_that_is_not_built_lists_only_devices(gpu_missing):
    with pytest.raises(RuntimeError) as err:
        sbd._load_backend("gpu")
    message = str(err.value)
    assert "'gpu' is not built" in message
    _assert_lists_devices_only(message)
