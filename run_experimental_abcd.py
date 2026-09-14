"""Reproduce Table 3 for experimental Volumes A-D using the final P86 workflow."""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import gc
import json

from experimental_p86 import (
    ExperimentalP86Config,
    analyze_experimental_nrrd,
    save_analysis_npz,
    save_config_json,
)
from multiscale_orientation_core import MultiscaleOrientationConfig

DEFAULT_FILES = {
    "A": "Cropped volume_2-3low.nrrd",
    "B": "Cropped volume_3_310l.nrrd",
    "C": "Cropped volume_1_1L(1).nrrd",
    "D": "Cropped volume_2_1L.nrrd",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory containing the four NRRD files")
    ap.add_argument("--out", type=Path, default=Path("outputs/experimental"))
    ap.add_argument("--save-npz", action="store_true", help="Save compact per-volume masked analysis arrays")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ocfg = MultiscaleOrientationConfig()
    ecfg = ExperimentalP86Config()
    save_config_json(args.out / "analysis_config.json", ocfg, ecfg)

    rows: list[dict[str, object]] = []
    for volume, filename in DEFAULT_FILES.items():
        path = args.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Analyzing Volume {volume}: {path.name}", flush=True)
        vol01, mask, result, summary, _header = analyze_experimental_nrrd(path, ocfg, ecfg)
        row = {
            "volume": volume,
            "source_file": filename,
            "candidate_fraction_percent": summary["candidate_fraction_percent"],
            "median_effective_confidence": summary["median_effective_confidence"],
            "median_cross_scale_stability": summary["median_cross_scale_stability"],
            "median_angular_spread_deg": summary["median_angular_spread_deg"],
            "best_supported_s0_percent": summary["best_supported_s0_percent"],
            "best_supported_s1_percent": summary["best_supported_s1_percent"],
            "best_supported_s2_percent": summary["best_supported_s2_percent"],
            "candidate_threshold": summary["candidate_threshold"],
            "candidate_voxels": summary["candidate_voxels"],
        }
        rows.append(row)
        if args.save_npz:
            save_analysis_npz(args.out / f"Volume_{volume}_analysis.npz", vol01.shape, mask, result)
        del vol01, mask, result
        gc.collect()

    fields = list(rows[0].keys())
    with (args.out / "Table3_experimental_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (args.out / "Table3_experimental_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nTable 3 values (rounded as in manuscript):")
    print("Volume  Candidate%  qeff   Stability  Spread(deg)  S0%   S1%   S2%")
    for r in rows:
        print(
            f"{r['volume']:>2}      {r['candidate_fraction_percent']:6.2f}   "
            f"{r['median_effective_confidence']:.3f}   {r['median_cross_scale_stability']:.3f}      "
            f"{r['median_angular_spread_deg']:5.2f}      {r['best_supported_s0_percent']:4.1f}  "
            f"{r['best_supported_s1_percent']:4.1f}  {r['best_supported_s2_percent']:4.1f}"
        )


if __name__ == "__main__":
    main()
