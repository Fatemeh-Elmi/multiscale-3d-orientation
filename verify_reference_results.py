"""Verify key manuscript numbers against a fresh run of the final code."""
from __future__ import annotations
from pathlib import Path
import argparse
import math

from experimental_p86 import analyze_experimental_nrrd

EXPECTED = {
    "A": (12.83, 0.162, 0.915, 11.80, 27.4, 22.3, 50.4),
    "B": (12.29, 0.143, 0.923, 11.27, 38.5, 18.6, 42.9),
    "C": (12.96, 0.159, 0.913, 11.71, 27.5, 13.4, 59.1),
    "D": (13.30, 0.179, 0.933, 10.55, 38.3, 19.9, 41.8),
}
FILES = {
    "A": "Cropped volume_2-3low.nrrd",
    "B": "Cropped volume_3_310l.nrrd",
    "C": "Cropped volume_1_1L(1).nrrd",
    "D": "Cropped volume_2_1L.nrrd",
}


def close(actual, expected, decimals):
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return abs(actual - expected) <= tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    args = ap.parse_args()
    ok = True
    for v, fn in FILES.items():
        _, _, _, s, _ = analyze_experimental_nrrd(args.data_dir / fn)
        vals = (
            s["candidate_fraction_percent"],
            s["median_effective_confidence"],
            s["median_cross_scale_stability"],
            s["median_angular_spread_deg"],
            s["best_supported_s0_percent"],
            s["best_supported_s1_percent"],
            s["best_supported_s2_percent"],
        )
        exp = EXPECTED[v]
        checks = [
            close(vals[0], exp[0], 2),
            close(vals[1], exp[1], 3),
            close(vals[2], exp[2], 3),
            close(vals[3], exp[3], 2),
            close(vals[4], exp[4], 1),
            close(vals[5], exp[5], 1),
            close(vals[6], exp[6], 1),
        ]
        passed = all(checks)
        ok &= passed
        print(v, "PASS" if passed else "FAIL", vals)
    if not ok:
        raise SystemExit("One or more manuscript checks failed")
    print("All experimental manuscript checks passed.")


if __name__ == "__main__":
    main()
