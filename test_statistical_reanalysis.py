"""Regression checks for the scientific/statistical revisions; no image files needed.

Run: python test_statistical_reanalysis.py
"""
import csv
from pathlib import Path
import unittest

import numpy as np

from condition_level_statistics import analyze_synthetic_rows, exact_spearman, read_rows


class ConditionCorrelationTests(unittest.TestCase):
    def test_exact_permutations_perfect_spearman(self):
        for n in (4, 6, 7, 8):
            result = exact_spearman(np.arange(n), np.arange(n))
            self.assertAlmostEqual(result["rho"], 1.0)
            self.assertEqual(result["extreme_permutations"], 2)
            import math
            self.assertAlmostEqual(result["p_exact_two_sided"], 2 / math.factorial(n))

    def test_negative_perfect(self):
        r = exact_spearman([1, 2, 3, 4], [4, 3, 2, 1])
        self.assertAlmostEqual(r["rho"], -1.0)
        self.assertAlmostEqual(r["p_exact_two_sided"], 1 / 12)

    def test_inputs(self):
        with self.assertRaises(ValueError):
            exact_spearman([1, 2], [3, 4])
        with self.assertRaises(ValueError):
            exact_spearman([1, 1, 1, 1], [1, 2, 3, 4])

    def test_geometry_collapsing(self):
        def row(family, amp, lam, rep, spread, error, curvature=0):
            obj = {
                "experiment":family,
                "median_angular_spread_deg":spread,
                "weighted_multiscale_median_error_deg":error,
                "replicate":rep,
            }
            if family == "curved_helix":
                obj.update(helix_amplitude_vox=amp, true_curvature_per_vox=curvature)
            elif family == "planar_tortuosity":
                obj.update(amplitude_vox=amp, wavelength_vox=lam)
            else:
                obj.update(bundle_max_deviation_deg=amp)
            return obj
        helix = [row("curved_helix", i, 32, rep, i+.05*rep, i+.2*rep, curvature=i)
                 for i in range(6) for rep in range(2)]
        tort = [row("planar_tortuosity", amp, lam, rep,
                    (lam+amp)/10+.03*rep, (lam+amp)/5+.02*rep)
                for lam in (24,40) for amp in (0,2,4,6) for rep in range(2)]
        bundle = [row("bundle_dispersion", amp, 0, rep, amp+.04*rep, amp+.06*rep)
                  for amp in (0,5,10,20) for rep in range(2)]
        cond, corr = analyze_synthetic_rows(helix,tort+bundle)
        byname={c['analysis']:c for c in corr}
        self.assertEqual(byname['helix_spread_vs_error_6_conditions']['n_conditions'],6)
        self.assertEqual(byname['tortuosity_spread_vs_error_8_settings_descriptive']['n_conditions'],8)
        self.assertEqual(byname['tortuosity_spread_vs_error_7_distinct_geometries']['n_conditions'],7)
        self.assertEqual(byname['bundle_spread_vs_error_4_conditions']['n_conditions'],4)
        self.assertEqual(len(cond),6+8+7+4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
