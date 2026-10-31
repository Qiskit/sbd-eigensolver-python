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

"""``examples/analysis/extrapolate.py``: a known line comes back, and bad input is named."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "examples" / "analysis" / "extrapolate.py"


def _run(*args):
    if not SCRIPT.is_file():
        pytest.skip(f"script not found: {SCRIPT}")
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=60)


def _points(e0, slope, pt2s, err=1e-4):
    return [f"--point={e0 + slope * x},{x},{err}" for x in pt2s]


def test_recovers_a_known_line(tmp_path):
    out = tmp_path / "fit.json"
    done = _run(*_points(-100.0, -1.4, [-0.08, -0.04, -0.02, -0.01]), "--json", str(out))
    assert done.returncode == 0, done.stderr
    fit = json.loads(out.read_text())
    assert fit["energy"] == pytest.approx(-100.0, abs=1e-9)
    assert fit["slope"] == pytest.approx(-1.4, abs=1e-9)
    assert fit["energy_without_largest"] == pytest.approx(-100.0, abs=1e-9)


def test_reads_variance_json_files(tmp_path):
    files = []
    for i, x in enumerate([-0.06, -0.03, -0.015]):
        path = tmp_path / f"r{i}.json"
        path.write_text(json.dumps({"energy": -50.0 - 1.2 * x, "pt2": x, "pt2_error": 1e-4,
                                    "variance": -2 * x, "variance_error": 1e-4,
                                    "dimension": 1000 * (i + 1)}))
        files.append(str(path))
    for axis in ("pt2", "variance"):
        done = _run(*files, "--x", axis, "--json", str(tmp_path / f"{axis}.json"))
        assert done.returncode == 0, done.stderr
        assert json.loads((tmp_path / f"{axis}.json").read_text())["energy"] == \
            pytest.approx(-50.0, abs=1e-9)


def test_two_points_say_there_is_no_error_bar():
    done = _run(*_points(-10.0, -1.0, [-0.1, -0.05]))
    assert done.returncode == 0
    assert "no error estimate" in done.stdout


def test_a_steep_slope_is_flagged():
    done = _run(*_points(-10.0, -3.0, [-0.1, -0.05]))
    assert "steeper than -1" in done.stdout
    assert "before trusting" in done.stdout


def test_a_steep_but_straight_line_is_called_usable():
    done = _run(*_points(-10.0, -3.0, [-0.1, -0.05, -0.02, -0.01]))
    assert "steeper than -1" in done.stdout
    assert "intercept is usable" in done.stdout


def test_a_positive_pt2_is_flagged():
    done = _run(*_points(-10.0, -1.0, [-0.1, -0.05, 0.02]))
    assert "PT2 is positive" in done.stdout


@pytest.mark.parametrize("args, message", [
    (["--point=-1,-0.1"], "at least two"),
    (["--point=-1,-0.1", "--point=-2,-0.1"], "same x"),
    (["--point=-1"], "expected"),
])
def test_bad_input_is_refused(args, message):
    done = _run(*args)
    assert done.returncode != 0
    assert message in done.stderr
