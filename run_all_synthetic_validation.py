"""Run all seven synthetic phantom families and save manuscript validation outputs."""
from __future__ import annotations
from pathlib import Path
import argparse
import json

from multiscale_orientation_core import MultiscaleOrientationConfig
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

    total = (
        baseline["summary"]["n_total_accuracy_volumes"]
        + 1  # primary orthogonal crossing challenge
        + extended["summary"]["n_extended_volumes_total"]
    )
    combined = {
        "total_synthetic_volumes_or_challenges": int(total),
        "baseline": baseline["summary"],
        "extended": extended["summary"],
    }
    (args.out / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
