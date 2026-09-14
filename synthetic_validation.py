"""Synthetic ground-truth validation for multiscale 3D orientation analysis.

This module validates the orientation stage independently of candidate
segmentation.  Synthetic 3D tubes provide known ground-truth directions.
Experiments cover:

1. straight fibers under controlled noise and blur,
2. curved/helical fibers with known local tangent direction and curvature,
3. a two-fiber crossing where orientation is intentionally ambiguous.

The validation compares each single structure-tensor scale, equal-weight
multiscale fusion, and confidence-weighted multiscale fusion.  It also tests
whether the reported confidence ranks orientation reliability and whether
cross-scale angular spread exposes scale-dependent bias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from multiscale_orientation_core import (
    MultiscaleOrientationConfig,
    _fuse_axial_directions,
    _tensor_eigensystem_at_mask,
)


@dataclass(frozen=True)
class SyntheticValidationConfig:
    """Reproducible synthetic validation design."""

    shape: tuple[int, int, int] = (64, 64, 64)
    tube_radius_vox: float = 3.0
    tube_length_vox: float = 46.0
    evaluation_end_margin_vox: float = 7.0
    base_blur_sigma_vox: float = 0.45
    signal_amplitude: float = 1.0
    background_level: float = 0.05

    # Straight-fiber experiments.
    noise_sigmas: tuple[float, ...] = (0.0, 0.04, 0.08, 0.12, 0.18)
    blur_sigmas: tuple[float, ...] = (0.0, 0.75, 1.50)
    blur_experiment_noise_sigma: float = 0.08
    straight_replicates: int = 2
    directions: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
    )

    # Curvature experiment: p(t) = (t, A sin(kt), A cos(kt)).
    helix_amplitudes_vox: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    helix_wavelength_vox: float = 32.0
    helix_noise_sigma: float = 0.08
    helix_replicates: int = 2

    # Crossing ambiguity experiment.
    crossing_noise_sigma: float = 0.08
    crossing_intersection_radius_vox: float = 5.0
    crossing_away_inner_radius_vox: float = 10.0
    crossing_away_outer_radius_vox: float = 20.0

    random_seed: int = 20260903
    max_scatter_points: int = 50000


def _unit(v: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        if n <= 0:
            raise ValueError("Direction vector must be nonzero")
        return arr / n
    if arr.ndim == 2 and arr.shape[1] == 3:
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        return np.divide(arr, n, out=np.zeros_like(arr), where=n > 1e-12)
    raise ValueError(f"Expected direction shape (3,) or (N,3), got {arr.shape}")


def make_straight_tube(
    shape: tuple[int, int, int],
    direction: Sequence[float],
    radius_vox: float,
    length_vox: float,
    evaluation_end_margin_vox: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a finite straight 3D tube and constant GT direction."""
    d = _unit(direction)
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    coords = np.indices(shape, dtype=np.float32)
    rel = np.stack([coords[i] - center[i] for i in range(3)], axis=-1)
    t = np.einsum("...i,i->...", rel, d)
    perp = rel - t[..., None] * d
    r = np.linalg.norm(perp, axis=-1)

    half = float(length_vox) / 2.0
    axial_inside = np.abs(t) <= half
    radial = np.exp(-0.5 * (r / max(radius_vox, 1e-6)) ** 2).astype(np.float32)
    image = radial * axial_inside.astype(np.float32)

    eval_half = max(half - float(evaluation_end_margin_vox), radius_vox + 1.0)
    eval_mask = (r <= radius_vox) & (np.abs(t) <= eval_half)
    gt_vectors = np.repeat(d[None, :], int(eval_mask.sum()), axis=0).astype(np.float32)
    return image.astype(np.float32), eval_mask.astype(bool), gt_vectors


def make_helical_tube(
    shape: tuple[int, int, int],
    amplitude_vox: float,
    wavelength_vox: float,
    radius_vox: float,
    length_vox: float,
    evaluation_end_margin_vox: float,
    centerline_samples: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create a 3D helical tube with a known local tangent field.

    Centerline:
        p(t) = (t, A sin(k t), A cos(k t)), k=2*pi/wavelength.

    The curve has constant curvature
        kappa = A*k^2 / (1 + A^2*k^2)
    in inverse voxels.
    """
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    half = float(length_vox) / 2.0
    t = np.linspace(-half, half, centerline_samples, dtype=np.float32)
    k = float(2.0 * np.pi / wavelength_vox)
    a = float(amplitude_vox)

    points = np.column_stack(
        [
            center[0] + t,
            center[1] + a * np.sin(k * t),
            center[2] + a * np.cos(k * t),
        ]
    ).astype(np.float32)
    tangents = np.column_stack(
        [
            np.ones_like(t),
            a * k * np.cos(k * t),
            -a * k * np.sin(k * t),
        ]
    ).astype(np.float32)
    tangents = _unit(tangents)

    grid = np.indices(shape, dtype=np.float32).reshape(3, -1).T
    tree = cKDTree(points)
    distance, nearest = tree.query(grid, k=1)
    distance = distance.reshape(shape).astype(np.float32)
    nearest = nearest.reshape(shape)
    nearest_t = t[nearest]

    # Smooth radial signal around the centerline; truncate far tails.
    image = np.exp(-0.5 * (distance / max(radius_vox, 1e-6)) ** 2).astype(np.float32)
    image[distance > 3.0 * radius_vox] = 0.0

    eval_half = max(half - float(evaluation_end_margin_vox), radius_vox + 1.0)
    eval_mask = (distance <= radius_vox) & (np.abs(nearest_t) <= eval_half)
    gt_vectors = tangents[nearest][eval_mask].astype(np.float32)
    curvature = float((a * k * k) / (1.0 + (a * k) ** 2)) if a > 0 else 0.0
    return image, eval_mask.astype(bool), gt_vectors, curvature


def make_crossing_tubes(
    shape: tuple[int, int, int],
    radius_vox: float,
    length_vox: float,
    evaluation_end_margin_vox: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create two orthogonal crossing tubes and masks for ambiguous/clean regions."""
    a, ma, _ = make_straight_tube(
        shape, (1, 0, 0), radius_vox, length_vox, evaluation_end_margin_vox
    )
    b, mb, _ = make_straight_tube(
        shape, (0, 1, 0), radius_vox, length_vox, evaluation_end_margin_vox
    )
    image = np.maximum(a, b).astype(np.float32)
    union = ma | mb
    coords = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    radial_center = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3)))
    return image, union, radial_center, ma & mb


def degrade_image(
    base: np.ndarray,
    noise_sigma: float,
    blur_sigma: float,
    rng: np.random.Generator,
    base_blur_sigma: float = 0.45,
    background_level: float = 0.05,
    signal_amplitude: float = 1.0,
) -> np.ndarray:
    """Apply controlled PSF blur and additive Gaussian noise."""
    img = background_level + signal_amplitude * base
    total_blur = float(np.sqrt(base_blur_sigma**2 + max(blur_sigma, 0.0) ** 2))
    if total_blur > 0:
        img = ndi.gaussian_filter(img.astype(np.float32), sigma=total_blur)
    if noise_sigma > 0:
        img = img + rng.normal(0.0, noise_sigma, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def axial_error_deg(vectors: np.ndarray, gt_directions: np.ndarray) -> np.ndarray:
    """Axial angular error in degrees, invariant to v ~ -v."""
    v = _unit(vectors)
    gt = _unit(gt_directions)
    if gt.ndim == 1:
        gt = np.repeat(gt[None, :], v.shape[0], axis=0)
    if gt.shape != v.shape:
        raise ValueError(f"GT directions {gt.shape} do not match estimates {v.shape}")
    dots = np.abs(np.sum(v * gt, axis=1))
    return np.degrees(np.arccos(np.clip(dots, 0.0, 1.0))).astype(np.float32)


def _fuse_equal_weight(vectors_by_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight dyadic fusion, returning direction and dyadic agreement."""
    s_count, n, _ = vectors_by_scale.shape
    mats = np.zeros((n, 3, 3), dtype=np.float32)
    for s in range(s_count):
        v = vectors_by_scale[s]
        mats += v[:, :, None] * v[:, None, :]
    mats /= max(s_count, 1)
    evals, evecs = np.linalg.eigh(mats)
    fused = evecs[:, :, 2].astype(np.float32)
    agreement = np.clip((3.0 * evals[:, 2] - 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    return fused, agreement


def _summarize_error(err: np.ndarray) -> dict[str, float]:
    return {
        "mae_deg": float(np.mean(err)),
        "median_error_deg": float(np.median(err)),
        "p90_error_deg": float(np.percentile(err, 90)),
        "within_5deg_percent": float(np.mean(err <= 5.0) * 100.0),
        "within_10deg_percent": float(np.mean(err <= 10.0) * 100.0),
    }


def evaluate_volume(
    vol01: np.ndarray,
    eval_mask: np.ndarray,
    gt_directions: np.ndarray,
    orientation_config: MultiscaleOrientationConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate single-scale and multiscale estimators on one GT mask."""
    vectors_scales: list[np.ndarray] = []
    conf_scales: list[np.ndarray] = []

    for gsig, tsig in orientation_config.scale_pairs_voxels:
        v, q, _line, _strength = _tensor_eigensystem_at_mask(
            vol01,
            eval_mask,
            sigma_gradient=gsig,
            sigma_tensor=tsig,
            strength_percentile_low=orientation_config.strength_percentile_low,
            strength_percentile_high=orientation_config.strength_percentile_high,
            confidence_power_line=orientation_config.confidence_power_line,
            confidence_power_strength=orientation_config.confidence_power_strength,
        )
        vectors_scales.append(v)
        conf_scales.append(q)

    vectors_by_scale = np.stack(vectors_scales, axis=0)
    confidence_by_scale = np.stack(conf_scales, axis=0)
    equal_fused, equal_agreement = _fuse_equal_weight(vectors_by_scale)
    weighted_fused, stability, effective_conf, angular_spread = _fuse_axial_directions(
        vectors_by_scale, confidence_by_scale
    )

    arrays: dict[str, np.ndarray] = {
        "weighted_error_deg": axial_error_deg(weighted_fused, gt_directions),
        "equal_error_deg": axial_error_deg(equal_fused, gt_directions),
        "effective_confidence": effective_conf,
        "stability": stability,
        "angular_spread_deg": angular_spread,
        "equal_agreement": equal_agreement,
        "weighted_vectors": weighted_fused,
        "equal_vectors": equal_fused,
    }
    for s, (v, q) in enumerate(zip(vectors_scales, conf_scales)):
        arrays[f"scale{s}_error_deg"] = axial_error_deg(v, gt_directions)
        arrays[f"scale{s}_confidence"] = q

    row: dict[str, Any] = {}
    method_names = [f"scale{s}" for s in range(len(vectors_scales))] + [
        "equal_multiscale", "weighted_multiscale"
    ]
    for name in method_names:
        err = (
            arrays[f"{name}_error_deg"] if name.startswith("scale")
            else arrays["equal_error_deg"] if name == "equal_multiscale"
            else arrays["weighted_error_deg"]
        )
        for k, val in _summarize_error(err).items():
            row[f"{name}_{k}"] = val

    finite = np.isfinite(effective_conf) & np.isfinite(arrays["weighted_error_deg"])
    if np.sum(finite) >= 3:
        rho, p = spearmanr(effective_conf[finite], arrays["weighted_error_deg"][finite])
        row["confidence_error_spearman_rho"] = float(rho)
        row["confidence_error_spearman_p"] = float(p)
    else:
        row["confidence_error_spearman_rho"] = float("nan")
        row["confidence_error_spearman_p"] = float("nan")

    # Scale disagreement should increase when scale-dependent bias is present.
    finite2 = np.isfinite(angular_spread) & np.isfinite(arrays["weighted_error_deg"])
    if np.sum(finite2) >= 3:
        rho2, p2 = spearmanr(angular_spread[finite2], arrays["weighted_error_deg"][finite2])
        row["spread_error_spearman_rho"] = float(rho2)
        row["spread_error_spearman_p"] = float(p2)
    else:
        row["spread_error_spearman_rho"] = float("nan")
        row["spread_error_spearman_p"] = float("nan")

    for frac in (0.2, 0.5, 1.0):
        n = len(effective_conf)
        take = max(1, int(round(frac * n)))
        order = np.argsort(effective_conf)[::-1][:take]
        row[f"top_{int(frac*100)}pct_conf_median_error_deg"] = float(
            np.median(arrays["weighted_error_deg"][order])
        )

    row["mean_effective_confidence"] = float(np.mean(effective_conf))
    row["median_effective_confidence"] = float(np.median(effective_conf))
    row["mean_stability"] = float(np.mean(stability))
    row["median_stability"] = float(np.median(stability))
    row["mean_angular_spread_deg"] = float(np.nanmean(angular_spread))
    row["median_angular_spread_deg"] = float(np.nanmedian(angular_spread))
    return row, arrays


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    values = sorted({float(r[key]) for r in rows})
    metric_keys = [k for k in rows[0].keys() if k not in {
        "experiment", "direction_index", "replicate", "noise_sigma", "blur_sigma",
        "helix_amplitude_vox", "true_curvature_per_vox"
    } and isinstance(rows[0][k], (int, float, np.floating))]
    out: list[dict[str, Any]] = []
    for val in values:
        sub = [r for r in rows if float(r[key]) == val]
        rec: dict[str, Any] = {key: val, "n_volumes": len(sub)}
        for metric in metric_keys:
            arr = np.asarray([float(r[metric]) for r in sub], dtype=float)
            rec[f"{metric}_mean"] = float(np.nanmean(arr))
            rec[f"{metric}_sd"] = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
        out.append(rec)
    return out


def _plot_straight_accuracy(noise_agg: list[dict[str, Any]], blur_agg: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300, constrained_layout=True)
    methods = [
        ("scale0_median_error_deg_mean", "Single scale S0"),
        ("scale1_median_error_deg_mean", "Single scale S1"),
        ("scale2_median_error_deg_mean", "Single scale S2"),
        ("equal_multiscale_median_error_deg_mean", "Equal multiscale"),
        ("weighted_multiscale_median_error_deg_mean", "Confidence-weighted multiscale"),
    ]
    for metric, label in methods:
        axes[0].plot([r["noise_sigma"] for r in noise_agg], [r[metric] for r in noise_agg], marker="o", label=label)
        axes[1].plot([r["blur_sigma"] for r in blur_agg], [r[metric] for r in blur_agg], marker="o", label=label)
    axes[0].set_xlabel("Gaussian noise sigma")
    axes[0].set_ylabel("Median axial orientation error (deg)")
    axes[0].set_title("Straight fibers: noise sensitivity")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Additional Gaussian blur sigma (voxels)")
    axes[1].set_ylabel("Median axial orientation error (deg)")
    axes[1].set_title("Straight fibers: blur sensitivity")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_curvature(curv_agg: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300, constrained_layout=True)
    x = [r["true_curvature_per_vox"] for r in curv_agg]
    methods = [
        ("scale0_median_error_deg_mean", "Single scale S0"),
        ("scale1_median_error_deg_mean", "Single scale S1"),
        ("scale2_median_error_deg_mean", "Single scale S2"),
        ("weighted_multiscale_median_error_deg_mean", "Confidence-weighted multiscale"),
    ]
    for metric, label in methods:
        axes[0].plot(x, [r[metric] for r in curv_agg], marker="o", label=label)
    axes[0].set_xlabel("True centerline curvature (voxel^-1)")
    axes[0].set_ylabel("Median local tangent error (deg)")
    axes[0].set_title("Scale-dependent bias on curved fibers")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    spread = [r["median_angular_spread_deg_mean"] for r in curv_agg]
    weighted_err = [r["weighted_multiscale_median_error_deg_mean"] for r in curv_agg]
    rho, _p = spearmanr(spread, weighted_err)
    axes[1].plot(spread, weighted_err, marker="o")
    axes[1].set_xlabel("Median cross-scale angular spread (deg)")
    axes[1].set_ylabel("Median weighted-fusion error (deg)")
    axes[1].set_title(f"Angular spread diagnoses scale bias (rho={rho:.3f})")
    axes[1].grid(alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_confidence_validation(conf: np.ndarray, err: np.ndarray, out_path: Path, rng: np.random.Generator, max_points: int) -> None:
    finite = np.isfinite(conf) & np.isfinite(err)
    c, e = conf[finite], err[finite]
    if len(c) > max_points:
        idx = rng.choice(len(c), size=max_points, replace=False)
        cs, es = c[idx], e[idx]
    else:
        cs, es = c, e
    rho, _p = spearmanr(c, e) if len(c) >= 3 else (np.nan, np.nan)

    quantiles = np.quantile(c, np.linspace(0, 1, 11)) if len(c) else np.linspace(0, 1, 11)
    centers, med, q25, q75 = [], [], [], []
    for i in range(10):
        lo, hi = quantiles[i], quantiles[i + 1]
        m = (c >= lo) & (c <= hi) if i == 9 else (c >= lo) & (c < hi)
        if np.any(m):
            centers.append(float(np.median(c[m])))
            med.append(float(np.median(e[m])))
            q25.append(float(np.percentile(e[m], 25)))
            q75.append(float(np.percentile(e[m], 75)))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300, constrained_layout=True)
    axes[0].scatter(cs, es, s=3, alpha=0.10, rasterized=True)
    axes[0].set_xlabel("Effective confidence")
    axes[0].set_ylabel("Orientation error (deg)")
    axes[0].set_ylim(0, 90)
    axes[0].set_title(f"Confidence ranks reliability (rho={rho:.3f})")
    axes[0].grid(alpha=0.2)
    centers = np.asarray(centers); med = np.asarray(med); q25 = np.asarray(q25); q75 = np.asarray(q75)
    axes[1].plot(centers, med, marker="o")
    axes[1].fill_between(centers, q25, q75, alpha=0.20)
    axes[1].set_xlabel("Confidence decile median")
    axes[1].set_ylabel("Orientation error (deg)")
    axes[1].set_title("Error across confidence deciles")
    axes[1].grid(alpha=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_representative(vol: np.ndarray, mask: np.ndarray, arrays: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    idx = np.flatnonzero(mask.ravel())
    err = np.full(vol.shape, np.nan, dtype=np.float32)
    conf = np.full(vol.shape, np.nan, dtype=np.float32)
    spread = np.full(vol.shape, np.nan, dtype=np.float32)
    err.ravel()[idx] = arrays["weighted_error_deg"]
    conf.ravel()[idx] = arrays["effective_confidence"]
    spread.ravel()[idx] = arrays["angular_spread_deg"]
    # MIPs are more robust than a fixed slice for arbitrary 3D orientation.
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8), dpi=300, constrained_layout=True)
    axes[0].imshow(np.max(vol, axis=0), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Synthetic image MIP")
    axes[0].axis("off")
    axes[1].imshow(np.max(mask.astype(float), axis=0), cmap="gray")
    axes[1].set_title("Ground-truth evaluation mask")
    axes[1].axis("off")
    im = axes[2].imshow(np.nanmax(err, axis=0), cmap="magma", vmin=0, vmax=max(10.0, float(np.nanpercentile(err, 95))))
    axes[2].set_title("Weighted orientation error (deg)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    im = axes[3].imshow(np.nanmax(spread, axis=0), cmap="plasma", vmin=0, vmax=max(5.0, float(np.nanpercentile(spread, 95))))
    axes[3].set_title("Cross-scale angular spread (deg)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _crossing_analysis(base: np.ndarray, union: np.ndarray, radial_center: np.ndarray, vol: np.ndarray, ocfg: MultiscaleOrientationConfig, vcfg: SyntheticValidationConfig) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    vectors, confs = [], []
    for gsig, tsig in ocfg.scale_pairs_voxels:
        v, q, _line, _strength = _tensor_eigensystem_at_mask(
            vol, union, gsig, tsig,
            ocfg.strength_percentile_low, ocfg.strength_percentile_high,
            ocfg.confidence_power_line, ocfg.confidence_power_strength,
        )
        vectors.append(v); confs.append(q)
    V = np.stack(vectors); Q = np.stack(confs)
    _fused, stability, eff, spread = _fuse_axial_directions(V, Q)
    flat_idx = np.flatnonzero(union.ravel())
    radial_vals = radial_center.ravel()[flat_idx]
    inter = radial_vals <= vcfg.crossing_intersection_radius_vox
    away = (radial_vals >= vcfg.crossing_away_inner_radius_vox) & (radial_vals <= vcfg.crossing_away_outer_radius_vox)
    summary = {
        "intersection_voxels": int(np.sum(inter)),
        "away_voxels": int(np.sum(away)),
        "intersection_median_confidence": float(np.median(eff[inter])),
        "away_median_confidence": float(np.median(eff[away])),
        "intersection_median_angular_spread_deg": float(np.median(spread[inter])),
        "away_median_angular_spread_deg": float(np.median(spread[away])),
        "intersection_median_cross_scale_stability": float(np.median(stability[inter])),
        "away_median_cross_scale_stability": float(np.median(stability[away])),
    }
    arrays = {
        "flat_idx": flat_idx,
        "effective_confidence": eff,
        "angular_spread_deg": spread,
        "intersection_selector": inter,
        "away_selector": away,
    }
    return summary, arrays


def _plot_crossing(vol: np.ndarray, union: np.ndarray, arrays: dict[str, np.ndarray], summary: dict[str, Any], out_path: Path) -> None:
    conf = np.full(vol.shape, np.nan, dtype=np.float32)
    spread = np.full(vol.shape, np.nan, dtype=np.float32)
    conf.ravel()[arrays["flat_idx"]] = arrays["effective_confidence"]
    spread.ravel()[arrays["flat_idx"]] = arrays["angular_spread_deg"]
    sl = vol.shape[2] // 2
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=300, constrained_layout=True)
    axes[0].imshow(vol[:, :, sl], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Two-fiber crossing")
    axes[0].axis("off")
    axes[1].imshow(union[:, :, sl], cmap="gray")
    axes[1].set_title("Synthetic fiber mask")
    axes[1].axis("off")
    im = axes[2].imshow(conf[:, :, sl], cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Effective confidence")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    im = axes[3].imshow(spread[:, :, sl], cmap="plasma", vmin=0, vmax=max(5.0, float(np.nanpercentile(spread, 95))))
    axes[3].set_title("Cross-scale angular spread")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.suptitle(
        "Ambiguous crossing: median confidence "
        f"{summary['intersection_median_confidence']:.3f} at crossing vs "
        f"{summary['away_median_confidence']:.3f} away",
        fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_synthetic_validation(
    outdir: str | Path,
    validation_config: SyntheticValidationConfig | None = None,
    orientation_config: MultiscaleOrientationConfig | None = None,
) -> dict[str, Any]:
    """Run the complete straight, curved, and crossing validation suite."""
    vcfg = validation_config or SyntheticValidationConfig()
    ocfg = orientation_config or MultiscaleOrientationConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(vcfg.random_seed)

    straight_noise_rows: list[dict[str, Any]] = []
    straight_blur_rows: list[dict[str, Any]] = []
    curved_rows: list[dict[str, Any]] = []
    pooled_straight_conf: list[np.ndarray] = []
    pooled_straight_err: list[np.ndarray] = []
    representative_straight = None
    representative_curved = None

    # 1) Straight fibers: noise sensitivity.
    for d_idx, direction in enumerate(vcfg.directions):
        base, eval_mask, gt = make_straight_tube(
            vcfg.shape, direction, vcfg.tube_radius_vox, vcfg.tube_length_vox,
            vcfg.evaluation_end_margin_vox,
        )
        for noise in vcfg.noise_sigmas:
            for rep in range(vcfg.straight_replicates):
                vol = degrade_image(
                    base, noise, 0.0, rng, vcfg.base_blur_sigma_vox,
                    vcfg.background_level, vcfg.signal_amplitude,
                )
                row, arrays = evaluate_volume(vol, eval_mask, gt, ocfg)
                row.update({"experiment": "straight_noise", "direction_index": d_idx, "noise_sigma": float(noise), "blur_sigma": 0.0, "replicate": rep})
                straight_noise_rows.append(row)
                pooled_straight_conf.append(arrays["effective_confidence"])
                pooled_straight_err.append(arrays["weighted_error_deg"])
                if representative_straight is None and d_idx == 6 and abs(noise - 0.12) < 1e-9 and rep == 0:
                    representative_straight = (vol.copy(), eval_mask.copy(), arrays)

    # 2) Straight fibers: blur sensitivity at moderate noise.
    for d_idx, direction in enumerate(vcfg.directions):
        base, eval_mask, gt = make_straight_tube(
            vcfg.shape, direction, vcfg.tube_radius_vox, vcfg.tube_length_vox,
            vcfg.evaluation_end_margin_vox,
        )
        for blur in vcfg.blur_sigmas:
            for rep in range(vcfg.straight_replicates):
                vol = degrade_image(
                    base, vcfg.blur_experiment_noise_sigma, blur, rng,
                    vcfg.base_blur_sigma_vox, vcfg.background_level, vcfg.signal_amplitude,
                )
                row, arrays = evaluate_volume(vol, eval_mask, gt, ocfg)
                row.update({"experiment": "straight_blur", "direction_index": d_idx, "noise_sigma": float(vcfg.blur_experiment_noise_sigma), "blur_sigma": float(blur), "replicate": rep})
                straight_blur_rows.append(row)
                pooled_straight_conf.append(arrays["effective_confidence"])
                pooled_straight_err.append(arrays["weighted_error_deg"])

    # 3) Curved fibers: local tangent ground truth and true analytic curvature.
    for amp in vcfg.helix_amplitudes_vox:
        base, eval_mask, gt, curvature = make_helical_tube(
            vcfg.shape, amp, vcfg.helix_wavelength_vox, vcfg.tube_radius_vox,
            vcfg.tube_length_vox, vcfg.evaluation_end_margin_vox,
        )
        for rep in range(vcfg.helix_replicates):
            vol = degrade_image(
                base, vcfg.helix_noise_sigma, 0.0, rng,
                vcfg.base_blur_sigma_vox, vcfg.background_level, vcfg.signal_amplitude,
            )
            row, arrays = evaluate_volume(vol, eval_mask, gt, ocfg)
            row.update({
                "experiment": "curved_helix", "helix_amplitude_vox": float(amp),
                "true_curvature_per_vox": float(curvature), "noise_sigma": float(vcfg.helix_noise_sigma),
                "blur_sigma": 0.0, "replicate": rep,
            })
            curved_rows.append(row)
            if representative_curved is None and abs(amp - 4.0) < 1e-9 and rep == 0:
                representative_curved = (vol.copy(), eval_mask.copy(), arrays)

    # 4) Crossing ambiguity challenge.
    crossing_base, crossing_union, radial_center, _overlap = make_crossing_tubes(
        vcfg.shape, vcfg.tube_radius_vox, vcfg.tube_length_vox, vcfg.evaluation_end_margin_vox
    )
    crossing_vol = degrade_image(
        crossing_base, vcfg.crossing_noise_sigma, 0.0, rng,
        vcfg.base_blur_sigma_vox, vcfg.background_level, vcfg.signal_amplitude,
    )
    crossing_summary, crossing_arrays = _crossing_analysis(
        crossing_base, crossing_union, radial_center, crossing_vol, ocfg, vcfg
    )

    # Save per-volume tables.
    all_rows = straight_noise_rows + straight_blur_rows + curved_rows
    _write_csv(outdir / "synthetic_validation_per_volume.csv", all_rows)
    noise_agg = _aggregate(straight_noise_rows, "noise_sigma")
    blur_agg = _aggregate(straight_blur_rows, "blur_sigma")
    curvature_agg = _aggregate(curved_rows, "true_curvature_per_vox")
    _write_csv(outdir / "straight_noise_summary.csv", noise_agg)
    _write_csv(outdir / "straight_blur_summary.csv", blur_agg)
    _write_csv(outdir / "curvature_summary.csv", curvature_agg)
    _write_csv(outdir / "crossing_summary.csv", [crossing_summary])

    # Figures.
    _plot_straight_accuracy(noise_agg, blur_agg, outdir / "Fig1_straight_accuracy_noise_blur.png")
    _plot_curvature(curvature_agg, outdir / "Fig2_curvature_scale_dependence.png")
    straight_conf = np.concatenate(pooled_straight_conf)
    straight_err = np.concatenate(pooled_straight_err)
    _plot_confidence_validation(
        straight_conf, straight_err, outdir / "Fig3_confidence_validity_straight.png",
        rng, vcfg.max_scatter_points,
    )
    _plot_crossing(
        crossing_vol, crossing_union, crossing_arrays, crossing_summary,
        outdir / "Fig4_crossing_ambiguity.png",
    )
    if representative_straight is not None:
        _plot_representative(
            *representative_straight,
            outdir / "Fig5_representative_straight_case.png",
            "Representative straight-fiber ground-truth case",
        )
    if representative_curved is not None:
        _plot_representative(
            *representative_curved,
            outdir / "Fig6_representative_curved_case.png",
            "Representative curved-fiber ground-truth case",
        )

    # Compact global validation summary.
    straight_rows = straight_noise_rows + straight_blur_rows
    weighted = np.asarray([r["weighted_multiscale_median_error_deg"] for r in straight_rows], float)
    equal = np.asarray([r["equal_multiscale_median_error_deg"] for r in straight_rows], float)
    singles = np.vstack([
        np.asarray([r[f"scale{s}_median_error_deg"] for r in straight_rows], float)
        for s in range(len(ocfg.scale_pairs_voxels))
    ])
    best_single = np.min(singles, axis=0)
    middle_single = singles[min(1, singles.shape[0] - 1)]
    rho_conf, p_conf = spearmanr(straight_conf, straight_err)

    curved_spread = np.asarray([r["median_angular_spread_deg"] for r in curved_rows], float)
    curved_error = np.asarray([r["weighted_multiscale_median_error_deg"] for r in curved_rows], float)
    curved_curvature = np.asarray([r["true_curvature_per_vox"] for r in curved_rows], float)
    rho_curv_spread, p_curv_spread = spearmanr(curved_curvature, curved_spread)
    rho_spread_error, p_spread_error = spearmanr(curved_spread, curved_error)

    summary = {
        "n_straight_volumes": len(straight_rows),
        "n_curved_volumes": len(curved_rows),
        "n_total_accuracy_volumes": len(all_rows),
        "straight_weighted_multiscale_median_error_across_volumes_deg": float(np.median(weighted)),
        "straight_equal_multiscale_median_error_across_volumes_deg": float(np.median(equal)),
        "straight_middle_single_scale_median_error_across_volumes_deg": float(np.median(middle_single)),
        "straight_oracle_best_single_scale_median_error_across_volumes_deg": float(np.median(best_single)),
        "straight_weighted_better_than_equal_fraction_percent": float(np.mean(weighted < equal) * 100.0),
        "straight_weighted_better_than_middle_single_fraction_percent": float(np.mean(weighted < middle_single) * 100.0),
        "straight_weighted_within_0p5deg_of_oracle_best_fraction_percent": float(np.mean(weighted <= best_single + 0.5) * 100.0),
        "straight_pooled_confidence_error_spearman_rho": float(rho_conf),
        "straight_pooled_confidence_error_spearman_p": float(p_conf),
        "curvature_vs_angular_spread_spearman_rho": float(rho_curv_spread),
        "curvature_vs_angular_spread_spearman_p": float(p_curv_spread),
        "curved_spread_vs_weighted_error_spearman_rho": float(rho_spread_error),
        "curved_spread_vs_weighted_error_spearman_p": float(p_spread_error),
        **crossing_summary,
    }
    # Confidence-stratified accuracy on straight fibers.
    order = np.argsort(straight_conf)[::-1]
    for frac in (0.10, 0.20, 0.50, 1.00):
        take = max(1, int(round(frac * len(order))))
        summary[f"straight_top_{int(frac*100)}pct_conf_median_error_deg"] = float(
            np.median(straight_err[order[:take]])
        )

    metadata = {
        "validation_design": {
            "straight": "known axial direction; controlled Gaussian noise and blur",
            "curved": "known local tangent field of an analytic helix with known curvature",
            "crossing": "two orthogonal tubes; crossing region intentionally has no unique single-fiber orientation",
            "segmentation_isolation": "ground-truth masks are used so orientation accuracy is not conflated with candidate segmentation error",
        },
        "validation_config": asdict(vcfg),
        "orientation_config": asdict(ocfg),
        "summary": summary,
        "interpretation_note": (
            "The synthetic validation is designed to test orientation accuracy, confidence ranking, "
            "and scale dependence. Confidence is a reliability score, not a calibrated probability. "
            "A method is not claimed to outperform every single tensor scale unless the measured "
            "results support that claim."
        ),
    }
    (outdir / "synthetic_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    readme = f"""Synthetic ground-truth validation of multiscale 3D orientation
================================================================

Validation components
---------------------
1. Straight 3D fibers: {len(straight_rows)} volumes spanning noise and blur.
2. Curved/helical fibers: {len(curved_rows)} volumes with known local tangent and analytic curvature.
3. Orthogonal crossing challenge: tests whether confidence decreases where a unique fiber orientation is not well defined.

Core straight-fiber results
---------------------------
Confidence-weighted multiscale median error across straight test volumes:
  {summary['straight_weighted_multiscale_median_error_across_volumes_deg']:.4f} deg
Equal-weight multiscale:
  {summary['straight_equal_multiscale_median_error_across_volumes_deg']:.4f} deg
Middle single scale:
  {summary['straight_middle_single_scale_median_error_across_volumes_deg']:.4f} deg
Oracle best single scale (uses ground truth; not available in real data):
  {summary['straight_oracle_best_single_scale_median_error_across_volumes_deg']:.4f} deg
Weighted fusion better than equal fusion in:
  {summary['straight_weighted_better_than_equal_fraction_percent']:.2f}% of straight volumes
Weighted fusion better than the middle single scale in:
  {summary['straight_weighted_better_than_middle_single_fraction_percent']:.2f}% of straight volumes
Weighted fusion within 0.5 deg of the oracle best single scale in:
  {summary['straight_weighted_within_0p5deg_of_oracle_best_fraction_percent']:.2f}% of straight volumes

Confidence validity on straight fibers
---------------------------------------
Pooled Spearman rho(confidence, angular error):
  {summary['straight_pooled_confidence_error_spearman_rho']:.4f}
Top 20% confidence median error:
  {summary['straight_top_20pct_conf_median_error_deg']:.4f} deg
All voxels median error:
  {summary['straight_top_100pct_conf_median_error_deg']:.4f} deg

Scale dependence on curved fibers
---------------------------------
Spearman rho(true curvature, cross-scale angular spread):
  {summary['curvature_vs_angular_spread_spearman_rho']:.4f}
Spearman rho(angular spread, weighted-fusion error):
  {summary['curved_spread_vs_weighted_error_spearman_rho']:.4f}

Crossing ambiguity challenge
----------------------------
Median effective confidence at crossing:
  {summary['intersection_median_confidence']:.4f}
Median effective confidence away from crossing:
  {summary['away_median_confidence']:.4f}
Median angular spread at crossing:
  {summary['intersection_median_angular_spread_deg']:.4f} deg
Median angular spread away from crossing:
  {summary['away_median_angular_spread_deg']:.4f} deg

Important interpretation
------------------------
The validation does NOT justify claiming that confidence-weighted fusion is
universally more accurate than every possible single scale. On simple straight
fibers, a sufficiently large single tensor scale can be more accurate because
its larger integration neighborhood suppresses noise without curvature bias.
The multiscale framework remains useful because real fibrous architecture is
not known a priori to be straight or to possess one optimal scale; the curved
experiment explicitly measures this scale dependence, while confidence and
angular-spread outputs expose reliability and disagreement rather than hiding
those effects.
"""
    (outdir / "README_SYNTHETIC_VALIDATION.txt").write_text(readme, encoding="utf-8")
    return metadata
