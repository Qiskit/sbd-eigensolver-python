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

"""Check SBD against the reference energies published with the upstream test data.

Each molecule directory under ``vendor/sbd-upstream/data`` carries a README with a
table of determinant-selection thresholds and the electronic energy that a
diagonalization over the corresponding determinants should produce. Those tables are
the closest thing to ground truth available here: the determinants are fixed, so the
answer is deterministic and independent of sampling, and the energies were obtained
by filtering a full CI calculation.

The cases are the rows of those tables. Only the cheapest of them run by default;
the rest are marked ``slow`` because they take minutes to hours and, at the far end,
more memory than a workstation has.
"""

from __future__ import annotations

import pytest

import sbd

# (molecule, alpha determinant file, expected electronic energy, is_slow)
#
# Transcribed from the tables in vendor/sbd-upstream/data/<molecule>/README.md. The
# rows beyond the first of each molecule are marked slow: the determinant count grows
# by roughly an order of magnitude per row, and compute time is reported upstream to
# scale as (determinants)**1.23. The largest rows of each table are omitted entirely,
# needing hundreds of gigabytes.
REFERENCE_ENERGIES = [
    # H2O, cc-pvdz, 24 orbitals, 10 electrons. FCI: -76.24377680
    ("h2o", "h2o-1em3-alpha.txt", -76.23594663, False),
    ("h2o", "h2o-1em4-alpha.txt", -76.24295848, True),
    ("h2o", "h2o-1em5-alpha.txt", -76.24373504, True),
    # N2, 6-31g, 18 orbitals, 14 electrons. FCI: -109.04874199
    ("n2", "1em3-alpha.txt", -109.04162110, False),
    ("n2", "3em4-alpha.txt", -109.04697304, True),
    ("n2", "1em4-alpha.txt", -109.04835269, True),
    ("n2", "3em5-alpha.txt", -109.04864315, True),
    ("n2", "1em5-alpha.txt", -109.04871934, True),
]


def _case_id(case) -> str:
    molecule, det_file, _, _ = case
    # e.g. "h2o-1em3": the threshold is the informative part of the file name.
    threshold = det_file.replace(f"{molecule}-", "").replace("-alpha.txt", "")
    return f"{molecule}-{threshold}"


# SBD packs a determinant into words of ``bit_length`` bits, and the resulting word
# count is a process-wide constant: ``det_vector::_elem_size`` is an inline static that
# throws "det_vector: elem_size mismatch" if a later diagonalization needs a different
# one. Two molecules can therefore share a process only if they agree on it.
#
# The word count is ``ceil(2 * norb / bit_length)``, so a ``bit_length`` of 64 keeps it
# at 1 for every reference molecule up to 32 orbitals: h2o (24), n2 (18) and nh3 (29).
# The two larger ones, c2h2 (38) and c4h4 (44), would need 2 words and so cannot share
# a process with these; adding them means a separate module, or forking per test.
#
# ``bit_length`` does not affect the result. Verified across 8, 20, 32, 48 and 64,
# which span word counts 6 down to 1: the h2o energy was identical to ten digits.
BIT_LENGTH = 64


def _diagonalize(backend, fcidump, det_file, **overrides):
    """Diagonalize over the determinants in ``det_file`` and return the energy."""
    sbd_data = backend.TPB_SBD()
    # A tolerance well below the precision the reference energies are quoted to, so
    # that a disagreement means a wrong answer rather than an unconverged one.
    sbd_data.eps = 1e-10
    sbd_data.max_it = 200
    sbd_data.bit_length = BIT_LENGTH
    for name, value in overrides.items():
        setattr(sbd_data, name, value)
    results = sbd.tpb_diag_from_files(str(fcidump), str(det_file), sbd_data)
    return results["energy"]


@pytest.mark.parametrize(
    "molecule,det_file,expected",
    [
        pytest.param(
            molecule,
            det_file,
            expected,
            marks=pytest.mark.slow if is_slow else (),
            id=_case_id((molecule, det_file, expected, is_slow)),
        )
        for molecule, det_file, expected, is_slow in REFERENCE_ENERGIES
    ],
)
def test_reference_energy(data_dir, backend, molecule, det_file, expected):
    """The energy matches the value published for these determinants.

    The reference energies are quoted to eight decimal places, so they are compared to
    that precision rather than to the solver's own convergence tolerance.
    """
    molecule_dir = data_dir / molecule
    energy = _diagonalize(
        backend, molecule_dir / "fcidump.txt", molecule_dir / det_file
    )
    assert energy == pytest.approx(expected, abs=1e-8)


@pytest.mark.mpi
def test_energy_does_not_depend_on_process_count(data_dir, backend):
    """Splitting the determinants across processes does not change the answer.

    Run under ``mpirun``, this diagonalizes the same subspace over however many
    processes were launched and compares against the published energy. A result that
    depends on the process count would mean the distribution itself is wrong, which is
    the failure this guards against; it is also why the comparison is against the
    reference rather than against another run.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    molecule_dir = data_dir / "h2o"
    energy = _diagonalize(
        backend,
        molecule_dir / "fcidump.txt",
        molecule_dir / "h2o-1em3-alpha.txt",
        adet_comm_size=comm.Get_size(),
    )

    # Only rank 0 receives the energy; the others are given a placeholder.
    if comm.Get_rank() == 0:
        assert energy == pytest.approx(-76.23594663, abs=1e-8)
