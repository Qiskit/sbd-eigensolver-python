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

"""Check that the build finds a GPU toolchain wherever the environment points.

A missed toolchain is not a loud failure: ``SBD_BUILD_BACKEND=auto`` falls back
to a CPU-only build, pip hides setup.py's output unless ``-v`` is passed, and the
install then succeeds. The user learns about it later, from
``Device 'gpu' requested but its backend is not usable``. So the detection is
worth a test of its own.

The case that motivated this: NVHPC keeps ``nvc++`` in
``<version root>/compilers/bin``, while ``NVHPC_ROOT`` -- set by NVIDIA's own
modulefile -- points at the version root. Probing only ``<root>/bin`` therefore
misses the path NVIDIA hands out.

``setup.py`` runs its detection at import time and cannot be imported without
triggering a build, so the functions under test are extracted from its source.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

SETUP_PY = pathlib.Path(__file__).resolve().parents[1] / "setup.py"

# Module-level names the extracted function closes over.
_CONSTANTS = {"_NVHPC_BIN_RELATIVE", "_NVHPC_VARS"}
_FUNCTIONS = {"find_nvidia_hpc_sdk"}


def _load_from_setup_py():
    """Pull the detection helpers out of setup.py without executing the rest."""
    if not SETUP_PY.is_file():
        pytest.skip(f"setup.py not found at {SETUP_PY}")
    namespace: dict = {"os": os}
    tree = ast.parse(SETUP_PY.read_text())
    for node in tree.body:
        keep = (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in _CONSTANTS for t in node.targets)
        ) or (isinstance(node, ast.FunctionDef) and node.name in _FUNCTIONS)
        if keep:
            exec(compile(ast.Module([node], []), str(SETUP_PY), "exec"), namespace)
    missing = (_CONSTANTS | _FUNCTIONS) - namespace.keys()
    assert not missing, f"setup.py no longer defines {sorted(missing)}"
    return namespace["find_nvidia_hpc_sdk"]


@pytest.fixture
def fake_nvhpc(tmp_path):
    """An NVHPC tree with the real layout: nvc++ under <root>/compilers/bin."""
    root = tmp_path / "Linux_x86_64" / "26.3"
    binary = root / "compilers" / "bin" / "nvc++"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    return root


@pytest.fixture
def clean_env(monkeypatch):
    """No NVHPC_* variables, and a PATH with no nvc++ on it."""
    for name in ("NVHPC_HOME", "NVHPC_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join(("/usr/bin", "/bin")))


@pytest.mark.parametrize(
    "variable, points_at",
    [
        ("NVHPC_HOME", "compilers"),
        ("NVHPC_HOME", "root"),
        ("NVHPC_ROOT", "compilers"),
        ("NVHPC_ROOT", "root"),
    ],
    ids=["home-compilers", "home-root", "root-compilers", "root-root"],
)
def test_finds_nvcxx_from_either_variable_and_layout(
    variable, points_at, fake_nvhpc, clean_env, monkeypatch
):
    """Both variables, and both the version root and its compilers/ subdirectory.

    ``NVHPC_HOME`` pointed at the version root used to fail, and ``NVHPC_ROOT``
    was not consulted at all -- each silently yielding a CPU-only build on a
    machine with a working compiler.
    """
    find = _load_from_setup_py()
    target = fake_nvhpc if points_at == "root" else fake_nvhpc / "compilers"
    monkeypatch.setenv(variable, str(target))

    path, found = find()

    assert found, f"{variable} pointing at the {points_at} was not resolved"
    assert pathlib.Path(path) == fake_nvhpc / "compilers" / "bin" / "nvc++"
    assert os.path.dirname(path) in os.environ["PATH"].split(os.pathsep), (
        "the compiler was found but its directory was not prepended to PATH"
    )


def test_reports_nothing_found_when_there_is_nothing(clean_env):
    """The negative control: no variables, no nvc++ on PATH, no claim of one.

    Without this, a test that only ever asserts success would pass against a
    function that always returned a path.
    """
    find = _load_from_setup_py()
    path, found = find()
    assert not found and path is None


def test_a_wrong_variable_does_not_mask_a_right_one(fake_nvhpc, clean_env, monkeypatch):
    """A stale NVHPC_HOME must not stop NVHPC_ROOT from being used."""
    find = _load_from_setup_py()
    monkeypatch.setenv("NVHPC_HOME", str(fake_nvhpc / "does-not-exist"))
    monkeypatch.setenv("NVHPC_ROOT", str(fake_nvhpc))

    path, found = find()

    assert found, "a bad NVHPC_HOME shadowed a good NVHPC_ROOT"
    assert pathlib.Path(path) == fake_nvhpc / "compilers" / "bin" / "nvc++"
