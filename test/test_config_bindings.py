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

"""Check that config-struct fields reach the solver, not just the Python object.

A ``def_readwrite`` that round-trips proves only that the member pointer compiles.
Whether the value changes anything is a separate question, and the interesting failure
is a field that is bound, stores what you set, and is then ignored.

``seed`` is the case in point: it only matters when ``init == 1`` (random initial
vector), so a test that never sets ``init`` would pass against a binding that does
nothing. The probe is SBD's own ``Davidson iteration 0.0`` line, whose energy is the
Rayleigh quotient of the starting vector -- change the seed and it must move.

Two practical notes. That line is printed by the C++ layer to file descriptor 1, which
``contextlib.redirect_stdout`` does not capture, so these run as subprocesses. And a
random start needs far more Davidson iterations than the default one, so this uses a
small 24-alpha subspace rather than the full 275-alpha reference case -- the same
anchor ``test_gdb_equivalence.py`` uses, which keeps the whole file to a few seconds.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

import sbd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
H2O_DIR = REPO_ROOT / "vendor" / "sbd-upstream" / "data" / "h2o"

# 24 alpha strings -> a 24^2 = 576-determinant product space. TPB and GDB agree on this
# subspace at -76.0588897208 (see test_gdb_equivalence.py), so it is already pinned.
ALPHA_LIMIT = 24
ANCHOR_ENERGY = -76.0588897208

_RANDOM_START = """
import pathlib, sys, sbd
backend = sbd.get_backend("cpu")
config = backend.TPB_SBD()
config.eps, config.max_it, config.bit_length = 1e-10, 200, 64
config.init = 1                       # random initial vector -- what seed affects
config.seed = int(sys.argv[2])
result = sbd.tpb_diag_from_files(sys.argv[1], sys.argv[3], config)
print("FINAL %.10f" % result["energy"])
"""

_cache: dict[int, tuple[str, float]] = {}


@pytest.fixture(scope="module")
def small_alpha_file(tmp_path_factory):
    """The first ALPHA_LIMIT alpha strings, so a random start converges quickly."""
    source = H2O_DIR / "h2o-1em3-alpha.txt"
    if not source.is_file():
        pytest.skip(f"vendored h2o data not found at {source}")
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    path = tmp_path_factory.mktemp("alpha") / "alpha_small.txt"
    path.write_text("\n".join(lines[:ALPHA_LIMIT]) + "\n")
    return path


def _random_start(seed: int, alpha_file: pathlib.Path) -> tuple[str, float]:
    """Run a random-start TPB diagonalization; return (iteration-0 line, energy).

    Cached per seed: the reproducibility check re-uses a seed, and there is no point
    paying for the same run twice.
    """
    if seed in _cache:
        return _cache[seed]
    completed = subprocess.run(
        [sys.executable, "-c", _RANDOM_START,
         str(H2O_DIR / "fcidump.txt"), str(seed), str(alpha_file)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    assert completed.returncode == 0, (
        f"run failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-1000:]}"
    )
    first = re.search(
        r"Davidson iteration 0\.0 \(tol=[0-9.e+-]+\): *(-?\d+\.?\d*)", completed.stdout
    )
    assert first, f"no iteration-0 line to compare:\n{completed.stdout[-2000:]}"
    energy = re.search(r"FINAL (-?\d+\.\d+)", completed.stdout)
    assert energy, f"no final energy:\n{completed.stdout[-2000:]}"
    _cache[seed] = (first.group(0), float(energy.group(1)))
    return _cache[seed]


def test_tpb_seed_round_trips():
    """The attribute exists and stores what it is given."""
    config = sbd.get_backend("cpu").TPB_SBD()
    config.seed = 4242
    assert config.seed == 4242


def test_tpb_seed_changes_the_starting_vector(small_alpha_file):
    """Two seeds must give different start vectors but the same answer.

    The first assertion is what proves the binding reaches ``BasisInitVector``; the
    second is what makes it safe -- wherever the solver starts, it must converge to
    the same eigenvalue.
    """
    first_start, first_energy = _random_start(1729, small_alpha_file)
    other_start, other_energy = _random_start(987654321, small_alpha_file)

    assert first_start != other_start, (
        "different seeds produced an identical starting vector, so seed is bound but "
        f"not reaching the solver: {first_start}"
    )
    assert first_energy == pytest.approx(other_energy, abs=1e-8)
    assert first_energy == pytest.approx(ANCHOR_ENERGY, abs=1e-8)


def test_tpb_seed_is_reproducible(small_alpha_file):
    """The same seed twice must be identical, or the test above proves nothing.

    Without this, two differing start vectors could be run-to-run nondeterminism
    rather than the seed taking effect.
    """
    start, _ = _random_start(1729, small_alpha_file)
    _cache.pop(1729)                      # force a genuinely fresh run
    again, _ = _random_start(1729, small_alpha_file)
    assert start == again
