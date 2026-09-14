"""Experimental SR-PhC-microCT analysis using the manuscript's P86 candidate mask.

Candidate selection is intentionally separate from the core estimator:
1. P1-P99 clipping and rescaling to [0,1]
2. Gaussian smoothing (sigma=1 voxel) for mask generation only
3. threshold at the 86th intensity percentile of the smoothed normalized volume
4. remove 3-D connected components smaller than 120 voxels
5. 3-D morphological closing with a radius-1 spherical structuring element
6. 3-D hole filling

The structure tensor is then evaluated on the normalized (not mask-smoothed)
volume and reported only for voxels inside the resulting candidate mask.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from scipy import ndimage as ndi

from multiscale_orientation_core import (
    MultiscaleOrientationConfig,
    analyze_masked_volume,
    normalize_image_percentile,
)
from nrrd_io import read_nrrd


@dataclass(frozen=True)
class ExperimentalP86Config:
    voxel_size_um: float = 3.84
    intensity_clip_low_percentile: float = 1.0
    intensity_clip_high_percentile: float = 99.0
    candidate_smoothing_sigma_vox: float = 1.0
    candidate_percentile: float = 86.0
    candidate_min_component_voxels: int = 120
    candidate_closing_radius_vox: int = 1


def spherical_structure(radius: int) -> np.ndarray:
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    coords = np.indices((2 * radius + 1,) * 3) - radius
    return np.sum(coords * coords, axis=0) <= radius * radius


def build_p86_candidate_mask(
    normalized_volume: np.ndarray,
    config: ExperimentalP86Config | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build the image-derived 3-D candidate mask exactly as described in Methods 2.5."""
    cfg = config or ExperimentalP86Config()
    vol = np.asarray(normalized_volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError("normalized_volume must be 3-D")

    smoothed = ndi.gaussian_filter(vol, sigma=cfg.candidate_smoothing_sigma_vox)
    threshold = float(np.percentile(smoothed[np.isfinite(smoothed)], cfg.candidate_percentile))
    mask = smoothed >= threshold

    labels, _ = ndi.label(mask)
    counts = np.bincount(labels.ravel())
    keep = np.zeros(counts.shape, dtype=bool)
    keep[counts >= cfg.candidate_min_component_voxels] = True
    if keep.size:
        keep[0] = False
    mask = keep[labels]

    mask = ndi.binary_closing(
        mask,
        structure=spherical_structure(cfg.candidate_closing_radius_vox),
    )
    mask = ndi.binary_fill_holes(mask).astype(bool)

    info = {
        "candidate_threshold": threshold,
        "candidate_voxels": int(mask.sum()),
        "candidate_fraction_percent": float(mask.mean() * 100.0),
    }
    return mask, info


def summarize_experimental_result(
    mask: np.ndarray,
    result: dict[str, np.ndarray],
) -> dict[str, float]:
    best = result["best_supported_scale"]
    return {
        "candidate_fraction_percent": float(mask.mean() * 100.0),
        "median_effective_confidence": float(np.nanmedian(result["effective_confidence"])),
        "median_cross_scale_stability": float(np.nanmedian(result["cross_scale_stability"])),
        "median_angular_spread_deg": float(np.nanmedian(result["angular_spread_deg"])),
        "best_supported_s0_percent": float(np.mean(best == 0) * 100.0),
        "best_supported_s1_percent": float(np.mean(best == 1) * 100.0),
        "best_supported_s2_percent": float(np.mean(best == 2) * 100.0),
    }


def analyze_experimental_nrrd(
    nrrd_path: str | Path,
    orientation_config: MultiscaleOrientationConfig | None = None,
    experimental_config: ExperimentalP86Config | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, str]]:
    """Read, normalize, segment by P86, and analyze one experimental NRRD volume."""
    ocfg = orientation_config or MultiscaleOrientationConfig()
    ecfg = experimental_config or ExperimentalP86Config()
    raw, header = read_nrrd(nrrd_path)
    vol01 = normalize_image_percentile(
        raw,
        ecfg.intensity_clip_low_percentile,
        ecfg.intensity_clip_high_percentile,
    )
    mask, mask_info = build_p86_candidate_mask(vol01, ecfg)
    result = analyze_masked_volume(vol01, mask, ocfg)
    summary = {
        **mask_info,
        **summarize_experimental_result(mask, result),
    }
    return vol01, mask, result, summary, header


def save_analysis_npz(
    path: str | Path,
    shape: tuple[int, int, int],
    mask: np.ndarray,
    result: dict[str, np.ndarray],
) -> None:
    """Save compact masked outputs without expanding vectors to dense 3-D arrays."""
    np.savez_compressed(
        path,
        shape=np.asarray(shape, dtype=np.int32),
        mask=np.asarray(mask, dtype=np.uint8),
        flat_indices=result["flat_indices"],
        vectors_by_scale=result["vectors_by_scale"],
        confidence_by_scale=result["confidence_by_scale"],
        fused_orientation=result["fused_orientation"],
        cross_scale_stability=result["cross_scale_stability"],
        effective_confidence=result["effective_confidence"],
        angular_spread_deg=result["angular_spread_deg"],
        best_supported_scale=result["best_supported_scale"],
    )


def save_config_json(
    path: str | Path,
    orientation_config: MultiscaleOrientationConfig,
    experimental_config: ExperimentalP86Config,
) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "orientation_config": asdict(orientation_config),
                "experimental_p86_config": asdict(experimental_config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
