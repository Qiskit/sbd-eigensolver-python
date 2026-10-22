#!/usr/bin/env python3

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

"""
Extrapolate a series of variational energies to the exact one.

Each state of a selected-CI series -- the rungs of a heat-bath ladder, say -- has a
variational energy E_var, an upper bound, and an estimated missing energy PT2 from
variance.py. As the subspace grows, PT2 goes to zero and E_var to the exact energy,
and over the last few states E_var is close to linear in PT2. The intercept of that
line at PT2 = 0 is the extrapolated energy, the estimate to report.

    for r in rung1 rung2 rung3; do
        mpirun -np 64 python variance.py --fcidump F --state $r- --bit_length 62 \\
            --json $r-var.json
    done
    python extrapolate.py rung1-var.json rung2-var.json rung3-var.json

Arguments are variance.py --json files, in any order. Plain numbers work too, as
``--point E_VAR,PT2[,PT2_ERROR]`` (repeat it), for results from elsewhere.

The fit is weighted. E_var is exact for its state, so the uncertainty is in PT2, the
x-axis; it is carried onto E through the slope, which is iterated once. Two numbers
qualify the intercept:

- its statistical error, from the fit;
- how far it moves when the state with the largest |PT2| -- the least linear -- is
  left out. With few states that is the better guide to the real uncertainty.

It needs at least two states, and three or more for either number to mean much.
"""

import argparse
import json
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrapolate E_var against PT2 (or the variance) to zero",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("files", nargs="*", metavar="JSON",
                        help="variance.py --json outputs, one per state")
    parser.add_argument("--point", action="append", default=[], metavar="E,PT2[,ERR]",
                        help="A state given directly: its E_var, PT2 and optionally "
                             "PT2's error. Repeat for more states")
    parser.add_argument("--x", choices=["pt2", "variance"], default="pt2",
                        help="What to extrapolate against. pt2 is the usual choice; "
                             "variance is a common alternative, and agreement between "
                             "the two is a check")
    parser.add_argument("--json", default="", metavar="FILE",
                        help="Also write the fit and the points to this file")
    return parser.parse_args()


def load(args):
    """The states, as (label, E_var, x, x_error, dimension), sorted by |x| descending."""
    points = []
    for path in args.files:
        with open(path, encoding="utf-8") as handle:
            d = json.load(handle)
        x = d["pt2"] if args.x == "pt2" else d["variance"]
        err = d["pt2_error"] if args.x == "pt2" else d["variance_error"]
        points.append((path, float(d["energy"]), float(x), float(err or 0.0),
                       d.get("dimension")))
    for item in args.point:
        fields = [float(v) for v in item.split(",")]
        if len(fields) not in (2, 3):
            raise ValueError(f"--point {item!r}: expected E_VAR,PT2[,PT2_ERROR]")
        if args.x != "pt2":
            raise ValueError("--point gives PT2 values, so it needs --x pt2")
        points.append((f"--point {item}", fields[0], fields[1],
                       fields[2] if len(fields) == 3 else 0.0, None))
    return sorted(points, key=lambda p: -abs(p[2]))


def fit(e, x, xerr):
    """Weighted straight line e = e0 + slope * x; returns e0, its error, slope, chi2/dof.

    Two points fix the line exactly: there is then no error estimate, and NaN is
    returned for it.
    """
    e, x, xerr = (np.asarray(v, dtype=float) for v in (e, x, xerr))
    n = len(e)
    if n == 2:
        slope = (e[1] - e[0]) / (x[1] - x[0])
        return e[0] - slope * x[0], float("nan"), slope, float("nan")
    # Error bars are on x. Carry them onto e through the slope of an unweighted
    # first pass, then refit; a floor keeps a state with no error bar from
    # taking all the weight.
    slope = np.polyfit(x, e, 1)[0]
    for _ in range(2):
        sigma = np.abs(slope) * xerr
        floor = 1e-3 * (np.max(np.abs(e - e.mean())) or 1.0)
        sigma = np.maximum(sigma, floor)
        w = 1.0 / sigma ** 2
        a = np.vstack([np.ones(n), x]).T
        cov = np.linalg.inv(a.T @ (a * w[:, None]))
        e0, slope = cov @ (a.T @ (w * e))
    residual = e - (e0 + slope * x)
    chi2 = float(np.sum(w * residual ** 2))
    dof = n - 2
    # Scale the covariance by the reduced chi^2 when the points scatter more than
    # their error bars allow, so a poor line gives a wide error, not a narrow one.
    scale = max(1.0, chi2 / dof)
    return float(e0), float(np.sqrt(cov[0, 0] * scale)), float(slope), chi2 / dof


def main():
    args = parse_args()
    try:
        points = load(args)
    except (OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if len(points) < 2:
        print("ERROR: need at least two states to extrapolate", file=sys.stderr)
        return 1
    xs = [p[2] for p in points]
    if len(set(xs)) < len(xs):
        print("ERROR: two states have the same x; give each state once", file=sys.stderr)
        return 1

    labels, e, x, xerr, dims = zip(*points)
    e0, e0_err, slope, chi2 = fit(e, x, xerr)
    drop = None
    if len(points) >= 3:
        drop = fit(e[1:], x[1:], xerr[1:])[0]

    name = "PT2" if args.x == "pt2" else "variance"
    unit = "Ha" if args.x == "pt2" else "Ha^2"
    print("=" * 74)
    print(f"Extrapolation of E_var to {name} = 0, over {len(points)} state(s)")
    print("-" * 74)
    print(f"  {'state':<34} {'dimension':>11} {'E_var (Ha)':>15} {name + ' (' + unit + ')':>14}")
    for label, ev, xv, xe, dim in points:
        shown = label if len(label) <= 34 else "..." + label[-31:]
        dim_s = f"{dim:,}" if isinstance(dim, int) else "-"
        print(f"  {shown:<34} {dim_s:>11} {ev:>15.8f} {xv:>14.6e}")
    print("-" * 74)
    print(f"  Extrapolated energy:  {e0:.8f} Ha"
          + ("" if np.isnan(e0_err) else f"  +- {e0_err:.2e} (fit)"))
    if drop is not None:
        print(f"  Without the largest-{name} state: {drop:.8f} Ha "
              f"(moves {abs(drop - e0) * 1e3:.2f} mHa)")
    print(f"  Slope: {slope:.4f}"
          + ("" if args.x != "pt2" else "  (E_var + PT2 corresponds to a slope of -1)"))
    if not np.isnan(chi2):
        print(f"  chi^2 per degree of freedom: {chi2:.2f}")

    # Warnings: the fit is only as good as the straight line it assumes.
    notes = []
    if len(points) == 2:
        notes.append("two states fix the line exactly, so there is no error estimate; "
                     "add a state")
    elif len(points) == 3:
        notes.append("three states: the 'without the largest' check uses only two")
    if args.x == "pt2" and any(v > 0 for v in x):
        notes.append("a PT2 is positive: Epstein-Nesbet PT2 below the ground state is "
                     "negative, so check that state")
    elif args.x == "pt2" and slope >= 0:
        notes.append("the slope is not negative: E_var should fall as PT2 rises toward "
                     "zero, so these states may not belong to one series")
    if args.x == "pt2" and slope < -1.5:
        notes.append(f"the slope is {slope:.1f}, steeper than -1: PT2 recovers only "
                     f"about {100 / abs(slope):.0f}% of the energy still missing, so "
                     f"E_var + PT2 is well above this estimate; the extrapolation "
                     f"needs states with smaller PT2 to be trusted")
    if abs(x[0]) > 3 * abs(x[-1]) and len(points) >= 3 and drop is not None \
            and abs(drop - e0) > (3 * e0_err if not np.isnan(e0_err) else 0):
        notes.append(f"leaving out the largest-{name} state moves the result by more "
                     f"than 3 error bars: the line is not straight there, so add a "
                     f"smaller-{name} state and drop that one")
    if not np.isnan(chi2) and chi2 > 4:
        notes.append("the points scatter well beyond their error bars; the error has "
                     "been widened to match, but check they come from one series")
    for note in notes:
        print(f"  NOTE: {note}")
    print("=" * 74)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"x": args.x, "energy": e0, "energy_error": e0_err,
                       "energy_without_largest": drop, "slope": slope,
                       "chi2_per_dof": chi2,
                       "points": [{"state": l, "energy": ev, "x": xv, "x_error": xe,
                                   "dimension": d}
                                  for l, ev, xv, xe, d in points]}, handle, indent=2)
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
