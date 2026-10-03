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

"""Check which of its two outputs ``solve_sci`` returns, and what each one contains.

``solve_sci`` either reports the determinants SBD selected, as ``SCIResult.carryover``,
or returns the eigenvector as an ``SCIState``. Which one it does is decided by
``report_carryover`` together with SBD's ``carryover_type``, and the two carry different
determinants in different orders. These tests pin that down, since the choice is
invisible in an energy: it is identical either way.
"""

from __future__ import annotations

import numpy as np
import pytest

# SBD packs determinants into words of ``bit_length`` bits and fixes the resulting word
# count for the lifetime of the process, so every test here uses one value. 64 keeps the
# count at 1 for h2o's 24 orbitals. See test_reference_energies.py for the full story.
BIT_LENGTH = 64

# Enough strings that a ratio leaves a proper subset to inspect, few enough to stay fast.
NUM_STRINGS = 40

NORB = 24
NELEC = (5, 5)


@pytest.fixture(name="h2o")
def h2o_fixture(data_dir, backend):
    """The h2o FCIDUMP and a truncated list of its alpha determinants.

    Depends on ``backend`` so that a build without the requested backend is reported as
    a skip before any of these tests tries to diagonalize.
    """
    del backend
    molecule_dir = data_dir / "h2o"
    with open(molecule_dir / "h2o-1em3-alpha.txt", encoding="utf-8") as f:
        strings = sorted({int(line.strip(), 2) for line in f if line.strip()})
    return (
        molecule_dir / "fcidump.txt",
        np.array(strings[:NUM_STRINGS], dtype=np.int64),
    )


def _solve(fcidump, strings, *, carryover_type, report_carryover, ratio=0.1):
    """Diagonalize over ``strings`` in both spin sectors."""
    from sbd.sbd_solver import solve_sci

    norb = NORB
    return solve_sci(
        (strings, strings),
        np.zeros((norb, norb)),
        np.zeros((norb,) * 4),
        norb,
        NELEC,
        sbd_config={
            "carryover_type": carryover_type,
            "ratio": ratio,
            "bit_length": BIT_LENGTH,
        },
        fcidump_path=str(fcidump),
        report_carryover=report_carryover,
    )


@pytest.mark.parametrize(
    "carryover_type,report_carryover,expect_carryover",
    [
        # carryover_type=0 means SBD selects nothing, so there is nothing to report and
        # the eigenvector is the only output. This is the default from
        # _create_sbd_config, which is what makes reporting the carryover opt-in.
        (0, None, False),
        (0, False, False),
        # Nonzero carryover_type: report it by default, but not when asked not to.
        (1, None, True),
        (1, False, False),
        (1, True, True),
    ],
)
def test_which_output_is_returned(
    h2o, carryover_type, report_carryover, expect_carryover
):
    """``report_carryover`` and ``carryover_type`` together decide the output.

    Exactly one of ``sci_state`` and ``carryover`` is populated: a caller can tell which
    path ran by looking at either, and never has to handle both at once.
    """
    fcidump, strings = h2o
    result = _solve(
        fcidump,
        strings,
        carryover_type=carryover_type,
        report_carryover=report_carryover,
    )

    if expect_carryover:
        assert result.carryover is not None
        assert result.sci_state is None
    else:
        assert result.carryover is None
        assert result.sci_state is not None


def test_report_carryover_true_rejects_carryover_type_zero(h2o):
    """Asking for a carryover SBD will not select is an error, not an empty result."""
    fcidump, strings = h2o
    with pytest.raises(ValueError, match="carryover_type is 0"):
        _solve(fcidump, strings, carryover_type=0, report_carryover=True)


def test_carryover_size_follows_the_ratio(h2o):
    """SBD keeps ``ratio`` of the determinants it was given, per spin sector."""
    fcidump, strings = h2o
    ratio = 0.25
    result = _solve(
        fcidump, strings, carryover_type=1, report_carryover=True, ratio=ratio
    )

    expected = int(ratio * len(strings))
    strings_a, strings_b = result.carryover
    assert len(strings_a) == expected
    assert len(strings_b) == expected


def test_eigenvector_spans_the_full_subspace(h2o):
    """With the eigenvector requested, it covers every input determinant.

    SaveMatrixFormWF writes the full ``adet x bdet`` subspace whatever
    ``carryover_type`` is, so a nonzero one must not shrink the ``SCIState``. Sizing that
    read against the carryover counts instead was the bug fixed in #19.
    """
    fcidump, strings = h2o
    result = _solve(fcidump, strings, carryover_type=1, report_carryover=False)

    assert result.sci_state.amplitudes.shape == (len(strings), len(strings))


def test_carryover_is_the_highest_weight_determinants_in_weight_order(h2o):
    """SBD's selection matches ranking the eigenvector by weight, order included.

    Two independent routes to the same answer: SBD's CarryOverAdet sorts by the diagonal
    reduced density matrix, and this ranks the amplitudes it would otherwise have
    returned. Agreement on the set says the selection is the weight-based one it claims
    to be; agreement on the *order* pins the descending-weight ordering that the
    docstrings promise, and that is worth pinning because it differs from every other
    determinant list in the wrapper: CarryOverAdet never calls sort_bitarray, so the
    result is neither canonically ordered nor deduplicated.
    """
    fcidump, strings = h2o
    ratio = 0.25

    reported = _solve(
        fcidump, strings, carryover_type=1, report_carryover=True, ratio=ratio
    ).carryover[0]
    state = _solve(
        fcidump, strings, carryover_type=1, report_carryover=False, ratio=ratio
    ).sci_state

    weights = np.sum(np.abs(state.amplitudes) ** 2, axis=1)
    ranked = state.ci_strs_a[np.argsort(-weights)][: len(reported)]

    assert reported.tolist() == ranked.tolist()
    # Not ascending, i.e. genuinely weight-ordered rather than canonically ordered. A
    # sorted list would pass the comparison above by coincidence if the weights happened
    # to rank the determinants in the same order.
    assert not np.all(np.diff(reported) > 0)


def test_energy_does_not_depend_on_which_output_is_returned(h2o):
    """The choice of output is a reporting decision, not a numerical one."""
    fcidump, strings = h2o
    reported = _solve(fcidump, strings, carryover_type=1, report_carryover=True)
    eigenvector = _solve(fcidump, strings, carryover_type=1, report_carryover=False)

    assert reported.energy == pytest.approx(eigenvector.energy, abs=1e-10)
