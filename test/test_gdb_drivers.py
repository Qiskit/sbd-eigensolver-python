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

"""Check the GDB example drivers end to end, as a user invokes them.

``test_gdb_equivalence.py`` covers the binding: it calls ``sbd.gdb_diag``
directly and proves the solver right. Nothing covered the layer a user actually
touches -- the argument parsing, the determinant text reading, the sharding
arithmetic, the default file paths -- so a driver could break while every
library test stayed green.

These run the scripts in a **subprocess**, the same device the Fe4S4 case in
``test_gdb_equivalence.py`` uses and for the same reason: ``det_vector``'s row
width is fixed process-wide on first use, so h2o (one word) and Fe4S4 (two)
cannot share a process. Invoking the driver as a subprocess also means the test
exercises ``__main__``, argument defaults included, rather than importing a
function and bypassing them.

The energies asserted here are not new reference values. The small h2o case
reproduces the same -76.0588897208 that ``test_gdb_equivalence.py`` anchors
against TPB, which ties the driver to the solver: if they ever disagree, the
driver's reading or sharding is at fault, not the diagonalization.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_DIR = REPO_ROOT / "examples" / "gdb"
H2O_DIR = REPO_ROOT / "vendor" / "sbd-upstream" / "data" / "h2o"
FE4S4_DIR = (
    REPO_ROOT
    / "vendor"
    / "sbd-upstream"
    / "apps"
    / "chemistry_gdb_selected_basis_diagonalization"
)

# The 24x24 product of the h2o alpha list, interleaved into 576 full
# determinants. Asserted through the binding in test_gdb_equivalence.py; the
# driver must land on the same value from the same input files.
H2O_576_ENERGY = -76.0588897208

# Upstream's shipped GDB subspace: four files of 14,884 determinants over 36
# orbitals. This is what both drivers use when neither --fcidump nor --detfiles
# is given, so a test of the defaults is also a test that those paths resolve.
FE4S4_DIM = 59536
FE4S4_ENERGY = -326.6982518821

# Small enough to keep the guardrail cases near-instant: 12 alphas is a
# 144-determinant subspace, and these never reach a diagonalization anyway.
TINY_ALPHA_LIMIT = 12


def _h2o_args(alpha_limit: int) -> list[str]:
    """Point a driver at the bundled h2o data as an |A|^2 product subspace."""
    return [
        "--fcidump", str(H2O_DIR / "fcidump.txt"),
        "--from-alpha", str(H2O_DIR / "h2o-1em3-alpha.txt"),
        "--alpha-limit", str(alpha_limit),
    ]


def _run(script: str, args: list[str], timeout: int = 900):
    """Run a driver from its own directory, the way the README shows it."""
    if not (DRIVER_DIR / script).is_file():
        pytest.skip(f"driver not found: {DRIVER_DIR / script}")
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=DRIVER_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _require(path: pathlib.Path) -> None:
    if not path.exists():
        pytest.skip(f"vendored upstream data not found at {path} (submodule not checked out?)")


def _assert_ok(completed) -> str:
    assert completed.returncode == 0, (
        "driver exited non-zero:\n"
        f"--- stdout ---\n{completed.stdout[-3000:]}\n"
        f"--- stderr ---\n{completed.stderr[-3000:]}"
    )
    return completed.stdout


def _energy(stdout: str) -> float:
    match = re.search(r"Ground state energy:\s*(-?\d+\.\d+)", stdout)
    assert match, f"no energy line in driver output:\n{stdout[-3000:]}"
    return float(match.group(1))


def _dimension(stdout: str) -> int:
    match = re.search(r"Subspace dimension:\s*(\d+)", stdout)
    assert match, f"no dimension line in driver output:\n{stdout[-3000:]}"
    return int(match.group(1))


# --- run_gdb_diag.py ----------------------------------------------------------


def test_diag_reproduces_the_library_anchor():
    """The driver on h2o must equal what the binding gives on the same subspace."""
    _require(H2O_DIR / "h2o-1em3-alpha.txt")
    stdout = _assert_ok(_run("run_gdb_diag.py", _h2o_args(24)))
    assert _dimension(stdout) == 576
    assert _energy(stdout) == pytest.approx(H2O_576_ENERGY, abs=1e-8)


def test_diag_reports_a_sane_electron_count():
    """The occupation density must sum to the electron count, not merely exist.

    A subspace built with the wrong bit order still diagonalizes and still
    returns a plausible-looking energy; the electron count is what catches it,
    which is why the driver prints the sum rather than the vector alone.
    """
    _require(H2O_DIR / "h2o-1em3-alpha.txt")
    stdout = _assert_ok(_run("run_gdb_diag.py", _h2o_args(24)))
    match = re.search(r"sums to (\d+\.\d+); should equal the electron count (\d+)", stdout)
    assert match, f"no electron-count check in output:\n{stdout[-3000:]}"
    assert float(match.group(1)) == pytest.approx(float(match.group(2)), abs=1e-6)


def test_diag_carryover_returns_parents_with_the_candidates():
    """``carryover_type 2`` must come back larger than the subspace it expanded.

    The expansion starts from ``edet = det`` (gdb/expansion.h:543), so the
    returned list is the next subspace rather than only the additions -- the
    property the heatbath driver's loop depends on.
    """
    _require(H2O_DIR / "h2o-1em3-alpha.txt")
    stdout = _assert_ok(_run("run_gdb_diag.py", [
        *_h2o_args(TINY_ALPHA_LIMIT), "--carryover_type", "2",
        "--heatbath_cutoff", "1e-3",
    ]))
    match = re.search(r"Carryover determinants on this rank:\s*(\d+)", stdout)
    assert match, f"no carryover line in output:\n{stdout[-3000:]}"
    assert int(match.group(1)) > _dimension(stdout)


@pytest.mark.parametrize(
    "extra, expected",
    [
        (["--t_comm_size", "2", "--b_comm_size", "1"], "t_comm_size <= b_comm_size"),
        (["--b_comm_size", "2"], "divide the rank count"),
    ],
    ids=["t-exceeds-b", "decomposition-does-not-divide-ranks"],
)
def test_diag_rejects_an_impossible_decomposition(extra, expected):
    """Both guardrails must fail loudly on one rank rather than segfault.

    ``t > b`` is the dangerous one: upstream does not check it, and MakeHelpers
    then dereferences an empty lookup, so without this guard the failure mode is
    a segfault rather than a message.
    """
    _require(H2O_DIR / "h2o-1em3-alpha.txt")
    completed = _run("run_gdb_diag.py", [*_h2o_args(TINY_ALPHA_LIMIT), *extra])
    assert completed.returncode != 0, f"expected a refusal, got:\n{completed.stdout[-2000:]}"
    combined = completed.stdout + completed.stderr
    assert expected in combined, f"guardrail message missing:\n{combined[-3000:]}"


@pytest.mark.slow
def test_diag_default_files_are_the_fe4s4_subspace():
    """With no input flags at all, the driver must run upstream's Fe4S4 data.

    This is the only test of the default paths, and the README documents them as
    the case every flagless command runs -- so if the vendored layout moves, this
    is what says so.
    """
    _require(FE4S4_DIR / "det0.txt")
    stdout = _assert_ok(_run("run_gdb_diag.py", [], timeout=1800))
    assert _dimension(stdout) == FE4S4_DIM
    assert _energy(stdout) == pytest.approx(FE4S4_ENERGY, abs=1e-8)


# --- run_gdb_heatbath.py ------------------------------------------------------


def test_heatbath_ladder_grows_and_lowers_the_energy(tmp_path):
    """Two rounds on h2o: the subspace must grow and the energy must not rise.

    The energy is variational in the subspace, and each round's subspace
    contains the previous one, so a rise means the expansion dropped parents or
    a round failed to converge. Round 0 diagonalizes the seed untouched, so it
    must equal the same anchor the diagonalization driver reports.
    """
    _require(H2O_DIR / "h2o-1em3-alpha.txt")
    log = tmp_path / "ladder.json"
    _assert_ok(_run("run_gdb_heatbath.py", [
        "--fcidump", str(H2O_DIR / "fcidump.txt"),
        "--subspace-from", "from-alpha",
        "--alpha-file", str(H2O_DIR / "h2o-1em3-alpha.txt"),
        "--alpha-limit", "24",
        "--cutoffs", "1e-3",
        "--max_rounds", "2",
        "--log", str(log),
    ]))

    rounds = json.loads(log.read_text())["rounds"]
    assert len(rounds) >= 2, f"expected at least two rounds, got {rounds}"

    assert rounds[0]["dimension"] == 576
    assert rounds[0]["energy"] == pytest.approx(H2O_576_ENERGY, abs=1e-8)
    assert rounds[0]["delta_energy"] is None, "the seed round has nothing to compare against"

    dims = [entry["dimension"] for entry in rounds]
    energies = [entry["energy"] for entry in rounds]
    assert dims == sorted(dims), f"subspace shrank across rounds: {dims}"
    assert dims[-1] > dims[0], f"expansion added nothing: {dims}"
    for before, after in zip(energies, energies[1:]):
        assert after <= before + 1e-10, f"energy rose across a round: {energies}"


@pytest.mark.parametrize(
    "alpha_flag", ["--from-alpha", "--alpha-file"], ids=["from-alpha", "alpha-file"]
)
def test_both_drivers_accept_either_alpha_flag_spelling(alpha_flag):
    """The two drivers grew different names for the same input; both now take both.

    Asserted on the diagonalization driver, where ``--from-alpha`` is the primary
    name and ``--alpha-file`` the alias added for parity with the heatbath driver.
    """
    _require(H2O_DIR / "h2o-1em3-alpha.txt")
    stdout = _assert_ok(_run("run_gdb_diag.py", [
        "--fcidump", str(H2O_DIR / "fcidump.txt"),
        alpha_flag, str(H2O_DIR / "h2o-1em3-alpha.txt"),
        "--alpha-limit", "24",
    ]))
    assert _dimension(stdout) == 576
    assert _energy(stdout) == pytest.approx(H2O_576_ENERGY, abs=1e-8)


# --- spin-weight validation ---------------------------------------------------

COUNTS_FILE = REPO_ROOT / "examples" / "tpb" / "count_dict_h2o.json"


def _interleave(alpha: str, beta: str) -> str:
    """Bit 2*i alpha orbital i, bit 2*i+1 beta orbital i, counting from the right.

    Spelled out here rather than imported from the driver so the test does not
    agree with the code under test by construction.
    """
    a, b = alpha[::-1], beta[::-1]
    return "".join(a[i] + b[i] for i in range(len(a)))[::-1]


def _counts_as_files(tmp_path):
    """The bundled counts file written both correctly and incorrectly.

    qiskit-addon-sqd emits ``[beta | alpha]`` concatenated; GDB wants the two
    interleaved. The "wrong" file is that raw concatenation, which is the mistake
    a first conversion actually makes.
    """
    counts = json.loads(COUNTS_FILE.read_text())
    norb = len(next(iter(counts))) // 2
    right = sorted({_interleave(k[norb:], k[:norb]) for k in counts})
    wrong = sorted(set(counts))
    good, bad = tmp_path / "right.txt", tmp_path / "wrong.txt"
    good.write_text("\n".join(right) + "\n")
    bad.write_text("\n".join(wrong) + "\n")
    return good, bad


def _heatbath_on(strings_file, tmp_path, extra=()):
    return _run("run_gdb_heatbath.py", [
        "--fcidump", str(H2O_DIR / "fcidump.txt"),
        "--subspace-from", "strings", "--strings-file", str(strings_file),
        "--cutoffs", "1e-3", "--max_rounds", "1",
        "--log", str(tmp_path / "ladder.json"), *extra,
    ])


def test_interleaved_counts_are_accepted(tmp_path):
    """The documented conversion of a counts file must run."""
    _require(COUNTS_FILE)
    _require(H2O_DIR / "fcidump.txt")
    good, _ = _counts_as_files(tmp_path)
    _assert_ok(_heatbath_on(good, tmp_path))
    rounds = json.loads((tmp_path / "ladder.json").read_text())["rounds"]
    assert rounds[0]["dimension"] == 275


def test_concatenated_counts_are_refused_with_a_diagnosis(tmp_path):
    """Feeding [beta | alpha] straight through must be caught before it is solved.

    Without the check this diagonalizes to a plausible-looking energy and only
    aborts later inside the heatbath expansion with std::out_of_range, so the
    failure gave no hint of its cause. The occupation density cannot catch it
    either: permuting bits preserves how many are set.
    """
    _require(COUNTS_FILE)
    _require(H2O_DIR / "fcidump.txt")
    _, bad = _counts_as_files(tmp_path)
    completed = _heatbath_on(bad, tmp_path)
    assert completed.returncode != 0, "mis-ordered determinants were accepted"
    combined = completed.stdout + completed.stderr
    assert "do not have 5 alpha and 5 beta electrons" in combined, combined[-2000:]
    assert "INTERLEAVED" in combined, "the message should name the likely cause"


@pytest.mark.parametrize("bit_length", [20, 30])
def test_the_weight_check_does_not_opt_out_of_other_bit_lengths(tmp_path, bit_length):
    """The refusal holds at every accepted word size, not only the default.

    The check used to return silently unless ``bit_length == 64``, which dropped the
    one validation that nothing downstream replaces. Both values here split h2o's
    48 bits over several words, so the masks are applied per word.
    """
    _require(COUNTS_FILE)
    _require(H2O_DIR / "fcidump.txt")
    _, bad = _counts_as_files(tmp_path)
    completed = _heatbath_on(bad, tmp_path, extra=("--bit_length", str(bit_length)))
    assert completed.returncode != 0, (
        f"mis-ordered determinants were accepted at --bit_length {bit_length}"
    )
    combined = completed.stdout + completed.stderr
    assert "do not have 5 alpha and 5 beta electrons" in combined, combined[-2000:]


def test_interleaved_counts_are_accepted_over_several_words(tmp_path):
    """Per-word masks must not turn a correct file into a false refusal."""
    _require(COUNTS_FILE)
    _require(H2O_DIR / "fcidump.txt")
    good, _ = _counts_as_files(tmp_path)
    _assert_ok(_heatbath_on(good, tmp_path, extra=("--bit_length", "20")))


@pytest.mark.parametrize("script", ["run_gdb_diag.py", "run_gdb_heatbath.py"])
@pytest.mark.parametrize("bit_length", ["64", "31"])
def test_drivers_refuse_an_unsafe_word_size(script, bit_length):
    """64 overflows a shift in SBD; odd sizes break its alpha/beta conversion."""
    completed = _run(script, ["--bit_length", bit_length])
    assert completed.returncode != 0
    assert "--bit_length must be even and between 2 and 62" in completed.stderr


def test_skip_weight_check_bypasses_the_validation(tmp_path):
    """The escape hatch must skip the check, not merely survive it.

    Asserted by the absence of the refusal: the run still fails downstream, since
    the determinants really are malformed, but it fails in SBD rather than here.
    """
    _require(COUNTS_FILE)
    _require(H2O_DIR / "fcidump.txt")
    _, bad = _counts_as_files(tmp_path)
    completed = _heatbath_on(bad, tmp_path, extra=("--skip-weight-check",))
    combined = completed.stdout + completed.stderr
    assert "do not have 5 alpha and 5 beta electrons" not in combined, (
        "--skip-weight-check did not skip the check"
    )
