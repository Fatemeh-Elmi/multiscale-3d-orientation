"""Core numerical routines for the final multiscale 3D orientation framework.

This module contains only the orientation estimator used by both synthetic and
experimental analyses in the manuscript. Candidate-region selection for the
experimental data is intentionally kept in ``experimental_p86.py`` so that the
synthetic ground-truth validation remains independent of segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class MultiscaleOrientationConfig:
    """Fixed estimator parameters used throughout the manuscript."""

    scale_pairs_voxels: tuple[tuple[float, float], ...] = (
        (0.75, 1.5),
        (1.25, 3.0),
        (2.0, 5.0),
    )
    strength_percentile_low: float = 5.0
    strength_percentile_high: float = 95.0
    confidence_power_line: float = 0.5
    confidence_power_strength: float = 0.5
    epsilon: float = 1e-12


def normalize_image_percentile(
    volume: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Clip intensities at two percentiles and rescale to [0, 1]."""
    x = np.asarray(volume, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(finite, [low_percentile, high_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _normalize_strength(values: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Robustly normalize structural energy to [0, 1] on analyzed voxels."""
    vals = np.asarray(values, dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals, dtype=np.float32)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vals, dtype=np.float32)
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _tensor_eigensystem_at_mask(
    vol01: np.ndarray,
    mask: np.ndarray,
    sigma_gradient: float,
    sigma_tensor: float,
    strength_percentile_low: float,
    strength_percentile_high: float,
    confidence_power_line: float,
    confidence_power_strength: float,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate 3D axial orientation and confidence on ``mask`` voxels.

    Notes
    -----
    * Gradients are computed after Gaussian smoothing with ``sigma_gradient``.
    * Tensor components are integrated with ``sigma_tensor``.
    * Eigenvalues are ascending. The smallest-eigenvalue eigenvector estimates
      the local line axis for line-like image structure.
    * Strength percentiles are calculated only over the analyzed mask voxels.
    """
    vol01 = np.asarray(vol01, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if vol01.ndim != 3 or mask.shape != vol01.shape:
        raise ValueError("vol01 and mask must be same-shape 3D arrays")
    if not np.any(mask):
        raise ValueError("analysis mask contains no voxels")

    sm = ndi.gaussian_filter(vol01, sigma=float(sigma_gradient))
    g0, g1, g2 = np.gradient(sm)

    j00 = ndi.gaussian_filter(g0 * g0, sigma=float(sigma_tensor))
    j11 = ndi.gaussian_filter(g1 * g1, sigma=float(sigma_tensor))
    j22 = ndi.gaussian_filter(g2 * g2, sigma=float(sigma_tensor))
    j01 = ndi.gaussian_filter(g0 * g1, sigma=float(sigma_tensor))
    j02 = ndi.gaussian_filter(g0 * g2, sigma=float(sigma_tensor))
    j12 = ndi.gaussian_filter(g1 * g2, sigma=float(sigma_tensor))

    idx = np.flatnonzero(mask.ravel())
    mats = np.empty((idx.size, 3, 3), dtype=np.float32)

    mats[:, 0, 0] = j00.ravel()[idx]
    mats[:, 1, 1] = j11.ravel()[idx]
    mats[:, 2, 2] = j22.ravel()[idx]
    mats[:, 0, 1] = mats[:, 1, 0] = j01.ravel()[idx]
    mats[:, 0, 2] = mats[:, 2, 0] = j02.ravel()[idx]
    mats[:, 1, 2] = mats[:, 2, 1] = j12.ravel()[idx]

    evals, evecs = np.linalg.eigh(mats)
    vectors = evecs[:, :, 0].astype(np.float32)

    l0 = evals[:, 0].astype(np.float32)
    l1 = evals[:, 1].astype(np.float32)
    l2 = evals[:, 2].astype(np.float32)
    eps = np.float32(epsilon)

    line_shape = np.clip((l1 - l0) / (l2 + eps), 0.0, 1.0).astype(np.float32)
    structural_energy = (l1 + l2).astype(np.float32)
    strength_norm = _normalize_strength(
        structural_energy,
        strength_percentile_low,
        strength_percentile_high,
    )

    confidence = (
        np.power(line_shape, confidence_power_line)
        * np.power(strength_norm, confidence_power_strength)
    ).astype(np.float32)
    confidence = np.clip(confidence, 0.0, 1.0)
    return vectors, confidence, line_shape, strength_norm


def _fuse_axial_directions(
    vectors_by_scale: np.ndarray,
    confidence_by_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-weighted dyadic fusion of axial directions.

    The dyadic mean is normalized by the sum of confidence weights, so it has
    unit trace wherever at least one weight is positive. This makes the largest
    eigenvalue lie in [1/3, 1] and supports the stability definition used in
    the manuscript.
    """
    vectors_by_scale = np.asarray(vectors_by_scale, dtype=np.float32)
    confidence_by_scale = np.asarray(confidence_by_scale, dtype=np.float32)
    if vectors_by_scale.ndim != 3 or vectors_by_scale.shape[2] != 3:
        raise ValueError("vectors_by_scale must have shape (S, N, 3)")
    if confidence_by_scale.shape != vectors_by_scale.shape[:2]:
        raise ValueError("confidence_by_scale must have shape (S, N)")

    s_count, n, _ = vectors_by_scale.shape
    sum_w = confidence_by_scale.sum(axis=0).astype(np.float32)
    mats = np.zeros((n, 3, 3), dtype=np.float32)

    for s in range(s_count):
        v = vectors_by_scale[s]
        w = confidence_by_scale[s]
        mats += w[:, None, None] * (v[:, :, None] * v[:, None, :])

    valid = sum_w > 1e-8
    mats[valid] /= sum_w[valid, None, None]
    mats[~valid] = np.eye(3, dtype=np.float32) / 3.0

    evals, evecs = np.linalg.eigh(mats)
    fused = evecs[:, :, 2].astype(np.float32)
    lambda_max = evals[:, 2].astype(np.float32)
    stability = np.clip((3.0 * lambda_max - 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

    mean_scale_conf = (sum_w / max(s_count, 1)).astype(np.float32)
    effective_confidence = (mean_scale_conf * stability).astype(np.float32)

    weighted_angle_sum = np.zeros(n, dtype=np.float32)
    for s in range(s_count):
        dots = np.abs(np.sum(vectors_by_scale[s] * fused, axis=1))
        dots = np.clip(dots, 0.0, 1.0)
        angles = np.degrees(np.arccos(dots)).astype(np.float32)
        weighted_angle_sum += confidence_by_scale[s] * angles

    angular_spread = np.divide(
        weighted_angle_sum,
        sum_w,
        out=np.full(n, np.nan, dtype=np.float32),
        where=sum_w > 1e-8,
    )
    return fused, stability, effective_confidence, angular_spread


def analyze_masked_volume(
    vol01: np.ndarray,
    mask: np.ndarray,
    config: MultiscaleOrientationConfig | None = None,
) -> dict[str, np.ndarray]:
    """Run all three scales and multiscale fusion on a supplied analysis mask."""
    cfg = config or MultiscaleOrientationConfig()
    vectors: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    line_shapes: list[np.ndarray] = []
    strengths: list[np.ndarray] = []

    for sigma_g, sigma_t in cfg.scale_pairs_voxels:
        v, q, line, strength = _tensor_eigensystem_at_mask(
            vol01,
            mask,
            sigma_gradient=sigma_g,
            sigma_tensor=sigma_t,
            strength_percentile_low=cfg.strength_percentile_low,
            strength_percentile_high=cfg.strength_percentile_high,
            confidence_power_line=cfg.confidence_power_line,
            confidence_power_strength=cfg.confidence_power_strength,
            epsilon=cfg.epsilon,
        )
        vectors.append(v)
        confidences.append(q)
        line_shapes.append(line)
        strengths.append(strength)

    vectors_by_scale = np.stack(vectors, axis=0)
    confidence_by_scale = np.stack(confidences, axis=0)
    fused, stability, effective_confidence, angular_spread = _fuse_axial_directions(
        vectors_by_scale,
        confidence_by_scale,
    )

    return {
        "flat_indices": np.flatnonzero(np.asarray(mask, dtype=bool).ravel()),
        "vectors_by_scale": vectors_by_scale,
        "confidence_by_scale": confidence_by_scale,
        "line_shape_by_scale": np.stack(line_shapes, axis=0),
        "strength_norm_by_scale": np.stack(strengths, axis=0),
        "fused_orientation": fused,
        "cross_scale_stability": stability,
        "effective_confidence": effective_confidence,
        "angular_spread_deg": angular_spread,
        "best_supported_scale": np.argmax(confidence_by_scale, axis=0).astype(np.int8),
    }
