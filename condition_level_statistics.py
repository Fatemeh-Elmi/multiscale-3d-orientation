"""Reproducible condition-level Spearman analyses for synthetic phantoms.

Each synthetic image has a per-volume MEDIAN angular error and spread. Replicate
volumes are averaged within a prescribed geometric condition BEFORE correlation.
Two-sided exact permutation p-values enumerate all permutations of the
condition-level response ranks (n<=8). Statistical interpretations are limited
to the deliberately prescribed phantom conditions.

For planar tortuosity A=0, wavelength does not change geometry, so the two
straight-reference conditions are additionally merged before the inferential
7-geometry correlation (the eight designed settings are also exported).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import permutations
from math import factorial
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from scipy.stats import rankdata, spearmanr

SPREAD = "median_angular_spread_deg"
ERROR = "weighted_multiscale_median_error_deg"


def exact_spearman(x, y) -> dict:
    """Spearman rho with exact two-sided permutation p for <= 8 finite pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3 or x.size > 8:
        raise ValueError("Exact permutation requires 3 to 8 paired condition values")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite condition metrics")
    xr, yr = rankdata(x), rankdata(y)
    if np.ptp(xr) == 0 or np.ptp(yr) == 0:
        raise ValueError("Spearman correlation is undefined for constant ranks")
    # Normalized ranks ensure a stable, accurate extremeness comparison.
    xc = (xr - xr.mean()) / np.linalg.norm(xr - xr.mean())
    yc = (yr - yr.mean()) / np.linalg.norm(yr - yr.mean())
    rho = float(np.dot(xc, yc))
    n_extreme = 0
    for perm in permutations(range(len(yr))):
        r = float(np.dot(xc, yc[list(perm)]))
        if abs(r) >= abs(rho) - 1e-12:
            n_extreme += 1
    return {
        "rho": float(np.clip(rho, -1, 1)),
        "p_exact_two_sided": n_extreme / factorial(len(yr)),
        "extreme_permutations": n_extreme,
        "total_permutations": factorial(len(yr)),
        "n_conditions": int(len(x)),
    }


def _group(rows: list[dict], label: str, key_fn) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = []
    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        record = {
            "family": label,
            "condition": str(key),
            "n_volumes": len(group),
            "mean_spread_deg": float(np.mean([float(r[SPREAD]) for r in group])),
            "mean_fusion_error_deg": float(np.mean([float(r[ERROR]) for r in group])),
        }
        if label == "helix":
            record["true_curvature_per_vox"] = float(np.mean([float(r["true_curvature_per_vox"]) for r in group]))
        out.append(record)
    return out


def _stats(label: str, conds: list[dict], x_key: str, y_key: str, pervol: list[dict]) -> dict:
    stat = exact_spearman([v[x_key] for v in conds], [v[y_key] for v in conds])
    full_x = "true_curvature_per_vox" if x_key == "true_curvature_per_vox" else SPREAD
    full_y = SPREAD if y_key == "mean_spread_deg" else ERROR
    vol_rho, vol_p = spearmanr([float(v[full_x]) for v in pervol], [float(v[full_y]) for v in pervol])
    return {
        "analysis": label,
        "n_volumes": len(pervol),
        "n_conditions": len(conds),
        "condition_rho": stat["rho"],
        "condition_p_exact_two_sided": stat["p_exact_two_sided"],
        "extreme_permutations": stat["extreme_permutations"],
        "total_permutations": stat["total_permutations"],
        "volume_rho_descriptive": float(vol_rho),
        "volume_p_asymptotic_descriptive": float(vol_p),
    }


def analyze_helix_rows(helix: list[dict]):
    h = _group(helix, "helix", lambda r: float(r["helix_amplitude_vox"]))
    return h, [
        _stats("helix_spread_vs_error_6_conditions", h, "mean_spread_deg", "mean_fusion_error_deg", helix),
        _stats("helix_curvature_vs_spread_6_conditions", h, "true_curvature_per_vox", "mean_spread_deg", helix),
    ]


def analyze_extended_rows(tort: list[dict], bundle: list[dict]):
    t = _group(tort, "tortuosity", lambda r: (float(r["wavelength_vox"]), float(r["amplitude_vox"])))
    b = _group(bundle, "bundle", lambda r: float(r["bundle_max_deviation_deg"]))
    tg = _group(tort, "tortuosity_distinct_geometry",
                lambda r: ("straight", 0.0) if float(r["amplitude_vox"]) == 0.0
                else (float(r["wavelength_vox"]), float(r["amplitude_vox"])))
    return t+b+tg, [
        _stats("tortuosity_spread_vs_error_8_settings_descriptive", t, "mean_spread_deg", "mean_fusion_error_deg", tort),
        _stats("tortuosity_spread_vs_error_7_distinct_geometries", tg, "mean_spread_deg", "mean_fusion_error_deg", tort),
        _stats("bundle_spread_vs_error_4_conditions", b, "mean_spread_deg", "mean_fusion_error_deg", bundle),
    ]


def analyze_synthetic_rows(baseline_rows: list[dict], extended_rows: list[dict]):
    """Return condition means and correlations using actual validation outputs."""
    helix = [r for r in baseline_rows if r.get("experiment") == "curved_helix"]
    tort = [r for r in extended_rows if r.get("experiment") == "planar_tortuosity"]
    bundle = [r for r in extended_rows if r.get("experiment") == "bundle_dispersion"]
    if not (len(helix) == 12 and len(tort) == 16 and len(bundle) == 8):
        raise ValueError(f"Unexpected sample counts: helix={len(helix)}, tort={len(tort)}, bundle={len(bundle)}")

    helix_conditions, helix_stats = analyze_helix_rows(helix)
    extended_conditions, extended_stats = analyze_extended_rows(tort, bundle)
    return helix_conditions + extended_conditions, helix_stats + extended_stats


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def run_condition_analysis(baseline_csv: Path, extended_csv: Path, outdir: Path):
    baseline = read_rows(baseline_csv)
    extended = read_rows(extended_csv)
    conditions, correlations = analyze_synthetic_rows(baseline, extended)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "condition_level_means.csv", conditions)
    write_csv(outdir / "condition_level_correlations.csv", correlations)
    (outdir / "condition_level_statistics.json").write_text(
        json.dumps({"correlations": correlations, "n_distinct_tortuosity_geometries": 7}, indent=2),
        encoding="utf-8")
    return conditions, correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--extended-csv", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args()
    _, correlations = run_condition_analysis(args.baseline_csv, args.extended_csv, args.out)
    for rec in correlations:
        print(f"{rec['analysis']}: n={rec['n_conditions']} "
              f"rho={rec['condition_rho']:.6f} exact_two_sided_p={rec['condition_p_exact_two_sided']:.8g}")


if __name__ == "__main__":
    main()
