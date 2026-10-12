"""Run all seven synthetic phantom families and save manuscript validation outputs."""
from __future__ import annotations
from pathlib import Path
import argparse
import json

from multiscale_orientation_core import MultiscaleOrientationConfig
from condition_level_statistics import run_condition_analysis
from synthetic_validation import SyntheticValidationConfig, run_synthetic_validation
from synthetic_validation_extended import (
    ExtendedSyntheticValidationConfig,
    run_extended_synthetic_validation,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/synthetic"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    orientation_config = MultiscaleOrientationConfig()
    baseline = run_synthetic_validation(
        args.out / "baseline",
        validation_config=SyntheticValidationConfig(),
        orientation_config=orientation_config,
    )
    extended = run_extended_synthetic_validation(
        args.out / "extended",
        validation_config=ExtendedSyntheticValidationConfig(),
        orientation_config=orientation_config,
    )

    condition_means, condition_correlations = run_condition_analysis(
        args.out / "baseline" / "synthetic_validation_per_volume.csv",
        args.out / "extended" / "extended_validation_per_volume.csv",
        args.out,
    )
    total = (
        baseline["summary"]["n_total_accuracy_volumes"]
        + 1  # primary orthogonal crossing challenge
        + extended["summary"]["n_extended_volumes_total"]
    )
    combined = {
        "total_synthetic_volumes_or_challenges": int(total),
        "baseline": baseline["summary"],
        "extended": extended["summary"],
        "condition_level_correlations": condition_correlations,
        "statistical_design_note": (
            "Replicates were averaged per geometry before Spearman correlations. "
            "Two-sided permutation p-values are exact. Tortuosity A=0 geometries "
            "were collapsed from 8 prescribed settings to 7 distinct geometries "
            "for the primary inference."
        ),
    }
    (args.out / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
