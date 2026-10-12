"""Extended morphology-focused synthetic validation for multiscale 3D orientation.

This module complements the baseline validation in ``synthetic_validation.py``.
It keeps the validated orientation estimator and its locked multiscale parameters
unchanged, and adds controlled phantom families that probe morphology-dependent
scale behavior:

1. fiber-radius sweep,
2. crossing-angle sweep,
3. planar tortuosity (sinusoidal centerlines), and
4. multi-fiber bundles with controlled angular dispersion.

Ground-truth orientation is used only where a unique local fiber axis is defined.
At crossing centers, where two directions coexist, reliability descriptors are
assessed instead of assigning an artificial single-direction ground truth.
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

from condition_level_statistics import analyze_extended_rows, exact_spearman
from multiscale_orientation_core import (
    MultiscaleOrientationConfig,
    _fuse_axial_directions,
    _tensor_eigensystem_at_mask,
)
from synthetic_validation import (
    _unit,
    axial_error_deg,
    degrade_image,
    evaluate_volume,
    make_straight_tube,
)


@dataclass(frozen=True)
class ExtendedSyntheticValidationConfig:
    """Locked design for the additional morphology-focused phantom tests."""

    shape: tuple[int, int, int] = (64, 64, 64)
    tube_length_vox: float = 46.0
    evaluation_end_margin_vox: float = 7.0
    base_blur_sigma_vox: float = 0.45
    background_level: float = 0.05
    signal_amplitude: float = 1.0
    fixed_noise_sigma: float = 0.08

    # 1) Radius sweep: morphology size relative to fixed tensor scales.
    radius_values_vox: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    radius_directions: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
    )
    radius_replicates: int = 2

    # 2) Crossing angle: controlled directional ambiguity.
    crossing_angles_deg: tuple[float, ...] = (30.0, 45.0, 60.0, 90.0)
    crossing_radius_vox: float = 3.0
    crossing_replicates: int = 2
    crossing_center_radius_vox: float = 5.0
    crossing_away_inner_radius_vox: float = 12.0
    crossing_away_outer_radius_vox: float = 20.0

    # 3) Planar tortuosity: p(t)=(t,A sin(kt),0), with spatially varying curvature.
    tortuosity_amplitudes_vox: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0)
    tortuosity_wavelengths_vox: tuple[float, ...] = (24.0, 40.0)
    tortuosity_radius_vox: float = 3.0
    tortuosity_replicates: int = 2
    centerline_samples: int = 700

    # 4) Multi-fiber bundles with controlled orientation dispersion.
    bundle_max_deviation_deg: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    bundle_radius_vox: float = 2.5
    bundle_replicates: int = 2
    bundle_offsets_yz: tuple[tuple[float, float], ...] = (
        (-10.0, -10.0),
        (-10.0, 10.0),
        (0.0, 0.0),
        (10.0, -10.0),
        (10.0, 10.0),
    )

    random_seed: int = 20260904


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    """Aggregate numeric metrics by one or more exact condition keys."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(row[k] for k in keys)
        groups.setdefault(group_key, []).append(row)
    out: list[dict[str, Any]] = []
    skip = set(keys) | {"experiment", "replicate", "direction_index"}
    for group_key in sorted(groups, key=lambda x: tuple(float(v) if isinstance(v, (int, float)) else str(v) for v in x)):
        sub = groups[group_key]
        rec: dict[str, Any] = {k: v for k, v in zip(keys, group_key)}
        rec["n_volumes"] = len(sub)
        metric_keys = sorted({k for row in sub for k, v in row.items() if k not in skip and isinstance(v, (int, float, np.floating))})
        for metric in metric_keys:
            vals = np.asarray([float(r[metric]) for r in sub if metric in r], dtype=float)
            if vals.size:
                rec[f"{metric}_mean"] = float(np.nanmean(vals))
                rec[f"{metric}_sd"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
        out.append(rec)
    return out


def make_offset_straight_tube(
    shape: tuple[int, int, int],
    direction: Sequence[float],
    radius_vox: float,
    length_vox: float,
    evaluation_end_margin_vox: float,
    offset_yz: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite straight tube centered at an offset in axes 1 and 2."""
    d = _unit(direction)
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    center = center + np.asarray([0.0, offset_yz[0], offset_yz[1]], dtype=np.float32)
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


def make_crossing_tubes_angle(
    shape: tuple[int, int, int],
    angle_deg: float,
    radius_vox: float,
    length_vox: float,
    evaluation_end_margin_vox: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two centered tubes crossing at a prescribed acute angle in the x-y plane."""
    theta = np.deg2rad(float(angle_deg))
    d1 = _unit((1.0, 0.0, 0.0))
    d2 = _unit((np.cos(theta), np.sin(theta), 0.0))
    a, ma, _ = make_straight_tube(shape, d1, radius_vox, length_vox, evaluation_end_margin_vox)
    b, mb, _ = make_straight_tube(shape, d2, radius_vox, length_vox, evaluation_end_margin_vox)
    image = np.maximum(a, b).astype(np.float32)
    union = ma | mb
    coords = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    radial_center = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3))).astype(np.float32)
    return image, union, ma, mb, radial_center, d1, d2


def _analyze_crossing_angle(
    vol: np.ndarray,
    union: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
    radial_center: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    ocfg: MultiscaleOrientationConfig,
    ecfg: ExtendedSyntheticValidationConfig,
) -> dict[str, float]:
    vectors, confs = [], []
    for gsig, tsig in ocfg.scale_pairs_voxels:
        v, q, _line, _strength = _tensor_eigensystem_at_mask(
            vol, union, gsig, tsig,
            ocfg.strength_percentile_low, ocfg.strength_percentile_high,
            ocfg.confidence_power_line, ocfg.confidence_power_strength,
        )
        vectors.append(v)
        confs.append(q)
    V = np.stack(vectors, axis=0)
    Q = np.stack(confs, axis=0)
    fused, stability, eff, spread = _fuse_axial_directions(V, Q)

    flat_idx = np.flatnonzero(union.ravel())
    radial = radial_center.ravel()[flat_idx]
    ma_u = ma.ravel()[flat_idx]
    mb_u = mb.ravel()[flat_idx]
    center_sel = radial <= ecfg.crossing_center_radius_vox
    annulus = (radial >= ecfg.crossing_away_inner_radius_vox) & (radial <= ecfg.crossing_away_outer_radius_vox)
    away1 = annulus & ma_u & ~mb_u
    away2 = annulus & mb_u & ~ma_u
    away = away1 | away2

    e1 = axial_error_deg(fused[away1], d1) if np.any(away1) else np.asarray([], dtype=np.float32)
    e2 = axial_error_deg(fused[away2], d2) if np.any(away2) else np.asarray([], dtype=np.float32)
    away_err = np.concatenate([e1, e2]) if e1.size or e2.size else np.asarray([np.nan])

    return {
        "center_voxels": int(np.sum(center_sel)),
        "away_voxels": int(np.sum(away)),
        "center_median_effective_confidence": float(np.nanmedian(eff[center_sel])),
        "away_median_effective_confidence": float(np.nanmedian(eff[away])),
        "center_median_angular_spread_deg": float(np.nanmedian(spread[center_sel])),
        "away_median_angular_spread_deg": float(np.nanmedian(spread[away])),
        "center_median_stability": float(np.nanmedian(stability[center_sel])),
        "away_median_stability": float(np.nanmedian(stability[away])),
        "away_median_orientation_error_deg": float(np.nanmedian(away_err)),
        "away_p90_orientation_error_deg": float(np.nanpercentile(away_err, 90)),
    }


def make_planar_sinusoidal_tube(
    shape: tuple[int, int, int],
    amplitude_vox: float,
    wavelength_vox: float,
    radius_vox: float,
    length_vox: float,
    evaluation_end_margin_vox: float,
    centerline_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Planar sinusoidal tube with known local tangent and local curvature."""
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    half = float(length_vox) / 2.0
    t = np.linspace(-half, half, centerline_samples, dtype=np.float32)
    a = float(amplitude_vox)
    k = float(2.0 * np.pi / wavelength_vox)

    points = np.column_stack([
        center[0] + t,
        center[1] + a * np.sin(k * t),
        np.full_like(t, center[2]),
    ]).astype(np.float32)
    tangents = np.column_stack([
        np.ones_like(t),
        a * k * np.cos(k * t),
        np.zeros_like(t),
    ]).astype(np.float32)
    tangents = _unit(tangents)
    curvature_centerline = (
        np.abs(a * k * k * np.sin(k * t)) /
        np.power(1.0 + (a * k * np.cos(k * t)) ** 2, 1.5)
    ).astype(np.float32)

    grid = np.indices(shape, dtype=np.float32).reshape(3, -1).T
    tree = cKDTree(points)
    distance, nearest = tree.query(grid, k=1)
    distance = distance.reshape(shape).astype(np.float32)
    nearest = nearest.reshape(shape)
    nearest_t = t[nearest]

    image = np.exp(-0.5 * (distance / max(radius_vox, 1e-6)) ** 2).astype(np.float32)
    image[distance > 3.0 * radius_vox] = 0.0
    eval_half = max(half - float(evaluation_end_margin_vox), radius_vox + 1.0)
    eval_mask = (distance <= radius_vox) & (np.abs(nearest_t) <= eval_half)
    gt_vectors = tangents[nearest][eval_mask].astype(np.float32)
    gt_curvature = curvature_centerline[nearest][eval_mask].astype(np.float32)
    return image, eval_mask.astype(bool), gt_vectors, gt_curvature


def _direction_from_deviation(angle_deg: float, azimuth_deg: float) -> np.ndarray:
    """Direction at a prescribed angular deviation from the +x mean axis."""
    theta = np.deg2rad(float(angle_deg))
    phi = np.deg2rad(float(azimuth_deg))
    return _unit((np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)))


def make_bundle(
    shape: tuple[int, int, int],
    max_deviation_deg: float,
    radius_vox: float,
    length_vox: float,
    evaluation_end_margin_vox: float,
    offsets_yz: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Five separated fibers whose axes have controlled 3-D angular dispersion."""
    # Symmetric deviations from the x-axis; azimuths distribute tilts in 3-D.
    maxd = float(max_deviation_deg)
    deviations = (maxd, 0.5 * maxd, 0.0, 0.5 * maxd, maxd)
    azimuths = (0.0, 90.0, 0.0, 180.0, 270.0)
    if len(offsets_yz) != len(deviations):
        raise ValueError("bundle_offsets_yz must contain five offsets")

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    dirs: list[np.ndarray] = []
    for dev, azi, offset in zip(deviations, azimuths, offsets_yz):
        d = _direction_from_deviation(dev, azi)
        img, mask, _ = make_offset_straight_tube(
            shape, d, radius_vox, length_vox, evaluation_end_margin_vox, offset
        )
        images.append(img)
        masks.append(mask)
        dirs.append(d)

    image = np.maximum.reduce(images).astype(np.float32)
    count = np.sum(np.stack(masks, axis=0), axis=0)
    eval_mask = count == 1
    gt_field = np.zeros(shape + (3,), dtype=np.float32)
    for mask, d in zip(masks, dirs):
        unique = mask & eval_mask
        gt_field[unique] = d
    gt = gt_field[eval_mask]
    # Actual RMS angular dispersion about the intended mean x-axis.
    mean_axis = _unit((1.0, 0.0, 0.0))
    angles = [float(axial_error_deg(np.asarray([d]), mean_axis)[0]) for d in dirs]
    rms_dispersion = float(np.sqrt(np.mean(np.square(angles))))
    return image, eval_mask.astype(bool), gt.astype(np.float32), rms_dispersion


def _plot_radius(summary: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    x = [r["radius_vox"] for r in summary]
    methods = [
        ("scale0_median_error_deg_mean", "S0"),
        ("scale1_median_error_deg_mean", "S1"),
        ("scale2_median_error_deg_mean", "S2"),
        ("weighted_multiscale_median_error_deg_mean", "Weighted fusion"),
    ]
    for metric, label in methods:
        ax.plot(x, [r[metric] for r in summary], marker="o", label=label)
    ax.set_xlabel("Fiber radius (voxels)")
    ax.set_ylabel("Median axial orientation error (deg)")
    ax.set_title("Fiber-size dependence at fixed analysis scales")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_crossing(summary: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=300, constrained_layout=True)
    x = [r["crossing_angle_deg"] for r in summary]
    axes[0].plot(x, [r["center_median_effective_confidence_mean"] for r in summary], marker="o", label="Crossing center")
    axes[0].plot(x, [r["away_median_effective_confidence_mean"] for r in summary], marker="o", label="Away from crossing")
    axes[0].set_xlabel("Crossing angle (deg)")
    axes[0].set_ylabel("Median effective confidence")
    axes[0].set_title("Reliability response to crossing angle")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, [r["center_median_angular_spread_deg_mean"] for r in summary], marker="o", label="Crossing center")
    axes[1].plot(x, [r["away_median_angular_spread_deg_mean"] for r in summary], marker="o", label="Away from crossing")
    axes[1].set_xlabel("Crossing angle (deg)")
    axes[1].set_ylabel("Median cross-scale angular spread (deg)")
    axes[1].set_title("Scale disagreement under directional ambiguity")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_tortuosity(summary: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=300, constrained_layout=True)
    # Use per-condition mean median curvature as the horizontal axis.
    x = [r["median_gt_curvature_per_vox_mean"] for r in summary]
    order = np.argsort(x)
    x_arr = np.asarray(x)[order]
    for metric, label in [
        ("scale0_median_error_deg_mean", "S0"),
        ("scale1_median_error_deg_mean", "S1"),
        ("scale2_median_error_deg_mean", "S2"),
        ("weighted_multiscale_median_error_deg_mean", "Weighted fusion"),
    ]:
        y = np.asarray([r[metric] for r in summary])[order]
        axes[0].plot(x_arr, y, marker="o", label=label)
    axes[0].set_xlabel("Median true local curvature (voxel$^{-1}$)")
    axes[0].set_ylabel("Median local tangent error (deg)")
    axes[0].set_title("Tortuosity-induced scale dependence")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    spread = np.asarray([r["median_angular_spread_deg_mean"] for r in summary])
    error = np.asarray([r["weighted_multiscale_median_error_deg_mean"] for r in summary])
    # Seven distinct geometric cases for significance; the two A=0 settings
    # represent the same straight geometry despite different wavelengths.
    a0 = [r for r in summary if float(r["amplitude_vox"]) == 0.0]
    nonzero = [r for r in summary if float(r["amplitude_vox"]) != 0.0]
    distinct = nonzero + [{
        "median_angular_spread_deg_mean": float(np.mean([r["median_angular_spread_deg_mean"] for r in a0])),
        "weighted_multiscale_median_error_deg_mean": float(np.mean([r["weighted_multiscale_median_error_deg_mean"] for r in a0])),
    }]
    exact = exact_spearman(
        [r["median_angular_spread_deg_mean"] for r in distinct],
        [r["weighted_multiscale_median_error_deg_mean"] for r in distinct],
    )
    rho, p = exact["rho"], exact["p_exact_two_sided"]
    axes[1].scatter(spread, error, s=35)
    axes[1].set_xlabel("Median cross-scale angular spread (deg)")
    axes[1].set_ylabel("Median weighted-fusion error (deg)")
    axes[1].set_title(f"Spread vs fusion error (rho={rho:.3f}, p={p:.3g})")
    axes[1].grid(alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bundle(summary: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=300, constrained_layout=True)
    x = [r["bundle_rms_dispersion_deg"] for r in summary]
    for metric, label in [
        ("scale0_median_error_deg_mean", "S0"),
        ("scale1_median_error_deg_mean", "S1"),
        ("scale2_median_error_deg_mean", "S2"),
        ("weighted_multiscale_median_error_deg_mean", "Weighted fusion"),
    ]:
        axes[0].plot(x, [r[metric] for r in summary], marker="o", label=label)
    axes[0].set_xlabel("Bundle RMS angular dispersion (deg)")
    axes[0].set_ylabel("Median local orientation error (deg)")
    axes[0].set_title("Multi-fiber bundle performance")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, [r["median_angular_spread_deg_mean"] for r in summary], marker="o", label="Angular spread")
    axes[1].plot(x, [r["median_effective_confidence_mean"] for r in summary], marker="o", label="Effective confidence")
    axes[1].set_xlabel("Bundle RMS angular dispersion (deg)")
    axes[1].set_ylabel("Descriptor value")
    axes[1].set_title("Reliability descriptors in heterogeneous bundles")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_extended_synthetic_validation(
    outdir: str | Path,
    validation_config: ExtendedSyntheticValidationConfig | None = None,
    orientation_config: MultiscaleOrientationConfig | None = None,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ecfg = validation_config or ExtendedSyntheticValidationConfig()
    ocfg = orientation_config or MultiscaleOrientationConfig()
    rng = np.random.default_rng(ecfg.random_seed)

    radius_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    tort_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []

    # 1) Fiber-radius sweep.
    for radius in ecfg.radius_values_vox:
        for d_idx, direction in enumerate(ecfg.radius_directions):
            base, mask, gt = make_straight_tube(
                ecfg.shape, direction, radius, ecfg.tube_length_vox,
                ecfg.evaluation_end_margin_vox,
            )
            for rep in range(ecfg.radius_replicates):
                vol = degrade_image(
                    base, ecfg.fixed_noise_sigma, 0.0, rng,
                    ecfg.base_blur_sigma_vox, ecfg.background_level, ecfg.signal_amplitude,
                )
                row, _ = evaluate_volume(vol, mask, gt, ocfg)
                row.update({
                    "experiment": "radius_sweep",
                    "radius_vox": float(radius),
                    "direction_index": d_idx,
                    "replicate": rep,
                    "noise_sigma": float(ecfg.fixed_noise_sigma),
                })
                radius_rows.append(row)

    # 2) Crossing-angle sweep.
    for angle in ecfg.crossing_angles_deg:
        base, union, ma, mb, radial, d1, d2 = make_crossing_tubes_angle(
            ecfg.shape, angle, ecfg.crossing_radius_vox,
            ecfg.tube_length_vox, ecfg.evaluation_end_margin_vox,
        )
        for rep in range(ecfg.crossing_replicates):
            vol = degrade_image(
                base, ecfg.fixed_noise_sigma, 0.0, rng,
                ecfg.base_blur_sigma_vox, ecfg.background_level, ecfg.signal_amplitude,
            )
            row = _analyze_crossing_angle(vol, union, ma, mb, radial, d1, d2, ocfg, ecfg)
            row.update({
                "experiment": "crossing_angle_sweep",
                "crossing_angle_deg": float(angle),
                "replicate": rep,
                "noise_sigma": float(ecfg.fixed_noise_sigma),
            })
            crossing_rows.append(row)

    # 3) Planar tortuosity.
    for wavelength in ecfg.tortuosity_wavelengths_vox:
        for amp in ecfg.tortuosity_amplitudes_vox:
            base, mask, gt, gt_curv = make_planar_sinusoidal_tube(
                ecfg.shape, amp, wavelength, ecfg.tortuosity_radius_vox,
                ecfg.tube_length_vox, ecfg.evaluation_end_margin_vox,
                ecfg.centerline_samples,
            )
            for rep in range(ecfg.tortuosity_replicates):
                vol = degrade_image(
                    base, ecfg.fixed_noise_sigma, 0.0, rng,
                    ecfg.base_blur_sigma_vox, ecfg.background_level, ecfg.signal_amplitude,
                )
                row, _ = evaluate_volume(vol, mask, gt, ocfg)
                row.update({
                    "experiment": "planar_tortuosity",
                    "amplitude_vox": float(amp),
                    "wavelength_vox": float(wavelength),
                    "median_gt_curvature_per_vox": float(np.median(gt_curv)),
                    "p90_gt_curvature_per_vox": float(np.percentile(gt_curv, 90)),
                    "max_gt_curvature_per_vox": float(np.max(gt_curv)),
                    "replicate": rep,
                    "noise_sigma": float(ecfg.fixed_noise_sigma),
                })
                tort_rows.append(row)

    # 4) Multi-fiber bundle dispersion.
    for max_dev in ecfg.bundle_max_deviation_deg:
        base, mask, gt, rms_dispersion = make_bundle(
            ecfg.shape, max_dev, ecfg.bundle_radius_vox,
            ecfg.tube_length_vox, ecfg.evaluation_end_margin_vox,
            ecfg.bundle_offsets_yz,
        )
        for rep in range(ecfg.bundle_replicates):
            vol = degrade_image(
                base, ecfg.fixed_noise_sigma, 0.0, rng,
                ecfg.base_blur_sigma_vox, ecfg.background_level, ecfg.signal_amplitude,
            )
            row, _ = evaluate_volume(vol, mask, gt, ocfg)
            row.update({
                "experiment": "bundle_dispersion",
                "bundle_max_deviation_deg": float(max_dev),
                "bundle_rms_dispersion_deg": float(rms_dispersion),
                "replicate": rep,
                "noise_sigma": float(ecfg.fixed_noise_sigma),
            })
            bundle_rows.append(row)

    radius_summary = _aggregate(radius_rows, ["radius_vox"])
    crossing_summary = _aggregate(crossing_rows, ["crossing_angle_deg"])
    tort_summary = _aggregate(tort_rows, ["wavelength_vox", "amplitude_vox"])
    bundle_summary = _aggregate(bundle_rows, ["bundle_max_deviation_deg", "bundle_rms_dispersion_deg"])

    _write_csv(outdir / "extended_validation_per_volume.csv", radius_rows + crossing_rows + tort_rows + bundle_rows)
    _write_csv(outdir / "radius_sweep_summary.csv", radius_summary)
    _write_csv(outdir / "crossing_angle_summary.csv", crossing_summary)
    _write_csv(outdir / "tortuosity_summary.csv", tort_summary)
    _write_csv(outdir / "bundle_dispersion_summary.csv", bundle_summary)

    _plot_radius(radius_summary, outdir / "FigE1_radius_scale_dependence.png")
    _plot_crossing(crossing_summary, outdir / "FigE2_crossing_angle_reliability.png")
    _plot_tortuosity(tort_summary, outdir / "FigE3_tortuosity_scale_bias.png")
    _plot_bundle(bundle_summary, outdir / "FigE4_bundle_dispersion.png")

    # Global relationships and concise outcomes.
    tort_spread = np.asarray([r["median_angular_spread_deg"] for r in tort_rows], dtype=float)
    tort_error = np.asarray([r["weighted_multiscale_median_error_deg"] for r in tort_rows], dtype=float)
    rho_tort, p_tort = spearmanr(tort_spread, tort_error)
    bundle_spread = np.asarray([r["median_angular_spread_deg"] for r in bundle_rows], dtype=float)
    bundle_error = np.asarray([r["weighted_multiscale_median_error_deg"] for r in bundle_rows], dtype=float)
    rho_bundle, p_bundle = spearmanr(bundle_spread, bundle_error)
    _, extended_stats = analyze_extended_rows(tort_rows, bundle_rows)
    tort_8_stats, tort_7_stats, bundle_condition_stats = extended_stats

    # Best individual scale per tested condition using aggregated median errors.
    def best_scale_from_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            vals = [r[f"scale{s}_median_error_deg_mean"] for s in range(len(ocfg.scale_pairs_voxels))]
            out.append({**{k: v for k, v in r.items() if not k.endswith("_mean") and not k.endswith("_sd")},
                        "best_single_scale": int(np.argmin(vals)),
                        "best_single_scale_error_deg": float(np.min(vals))})
        return out

    radius_best = best_scale_from_summary(radius_summary)
    tort_best = best_scale_from_summary(tort_summary)
    bundle_best = best_scale_from_summary(bundle_summary)

    global_summary = {
        "n_radius_volumes": len(radius_rows),
        "n_crossing_angle_volumes": len(crossing_rows),
        "n_tortuosity_volumes": len(tort_rows),
        "n_bundle_volumes": len(bundle_rows),
        "n_extended_volumes_total": len(radius_rows) + len(crossing_rows) + len(tort_rows) + len(bundle_rows),
        "tortuosity_spread_vs_weighted_error_spearman_rho": float(rho_tort),
        "tortuosity_spread_vs_weighted_error_spearman_p": float(p_tort),
        "bundle_spread_vs_weighted_error_spearman_rho": float(rho_bundle),
        "bundle_spread_vs_weighted_error_spearman_p": float(p_bundle),
        "tortuosity_condition_level_spread_error_8_settings_rho": tort_8_stats["condition_rho"],
        "tortuosity_condition_level_spread_error_8_settings_p_exact": tort_8_stats["condition_p_exact_two_sided"],
        "tortuosity_condition_level_spread_error_7_distinct_rho": tort_7_stats["condition_rho"],
        "tortuosity_condition_level_spread_error_7_distinct_p_exact": tort_7_stats["condition_p_exact_two_sided"],
        "bundle_condition_level_spread_error_rho": bundle_condition_stats["condition_rho"],
        "bundle_condition_level_spread_error_p_exact": bundle_condition_stats["condition_p_exact_two_sided"],
        "radius_best_scales_by_condition": radius_best,
        "tortuosity_best_scales_by_condition": tort_best,
        "bundle_best_scales_by_condition": bundle_best,
    }

    metadata = {
        "purpose": "Extended morphology-focused synthetic validation without changing the locked multiscale estimator.",
        "design": {
            "radius_sweep": "Tests dependence of orientation accuracy on fiber diameter relative to fixed tensor scales.",
            "crossing_angle_sweep": "Tests reliability descriptors as two local directions become more or less separable; no single GT is assigned at the crossing center.",
            "planar_tortuosity": "Tests scale-dependent local-tangent bias on a sinusoidal centerline with spatially varying analytic curvature.",
            "bundle_dispersion": "Tests local orientation estimation in multi-fiber neighborhoods with controlled 3-D angular dispersion.",
        },
        "validation_config": asdict(ecfg),
        "orientation_config": asdict(ocfg),
        "summary": global_summary,
    }
    (outdir / "extended_validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    protocol = "# Extended synthetic validation protocol\n\n"
    protocol += "The existing estimator and its three analysis scales are locked. The new tests vary only phantom morphology.\n\n"
    protocol += "| Phantom family | Conditions | Replicates | Volumes | Main endpoint |\n|---|---:|---:|---:|---|\n"
    protocol += f"| Fiber radius | {len(ecfg.radius_values_vox)} radii x {len(ecfg.radius_directions)} directions | {ecfg.radius_replicates} | {len(radius_rows)} | axial error by scale/fusion |\n"
    protocol += f"| Crossing angle | {len(ecfg.crossing_angles_deg)} angles | {ecfg.crossing_replicates} | {len(crossing_rows)} | confidence/spread at crossing; error away |\n"
    protocol += f"| Planar tortuosity | {len(ecfg.tortuosity_wavelengths_vox)} wavelengths x {len(ecfg.tortuosity_amplitudes_vox)} amplitudes | {ecfg.tortuosity_replicates} | {len(tort_rows)} | local tangent error and scale disagreement |\n"
    protocol += f"| Bundle dispersion | {len(ecfg.bundle_max_deviation_deg)} dispersion levels | {ecfg.bundle_replicates} | {len(bundle_rows)} | local error, confidence, angular spread |\n"
    protocol += f"| **Total new volumes** |  |  | **{global_summary['n_extended_volumes_total']}** |  |\n\n"
    protocol += "All images use additive Gaussian noise sigma=0.08 and baseline Gaussian blur sigma=0.45 voxel unless otherwise stated. Ground-truth masks isolate orientation-estimation performance from candidate-segmentation error.\n"
    (outdir / "EXTENDED_VALIDATION_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    return metadata
