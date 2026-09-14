"""Generate Figure 9 (Volume A) from the final P86 workflow at 300 dpi."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experimental_p86 import analyze_experimental_nrrd


def masked_scalar(shape, flat_idx, values, fill=np.nan):
    out = np.full(int(np.prod(shape)), fill, dtype=np.float32)
    out[flat_idx] = values
    return out.reshape(shape)


def masked_rgb(shape, flat_idx, vectors):
    """RGB encodes |axis-0|, |axis-1|, |axis-2| inside the analysis mask."""
    rgb = np.zeros((int(np.prod(shape)), 3), dtype=np.float32)
    rgb[flat_idx] = np.clip(np.abs(vectors), 0.0, 1.0)
    return rgb.reshape(shape + (3,))




def orientation_key_image(size: int = 220) -> np.ndarray:
    """Create a compact continuous key for the RGB axial-direction encoding.

    The disk visualizes the same mapping used in the orientation panels:
    RGB = (|n_axis0|, |n_axis1|, |n_axis2|).  The disk parameterization is
    only a legend; antiparallel directions remain equivalent.
    """
    u = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(u, u, indexing="xy")
    rr2 = xx * xx + yy * yy
    inside = rr2 <= 1.0
    zz = np.sqrt(np.clip(1.0 - rr2, 0.0, 1.0))
    rgb = np.zeros((size, size, 4), dtype=np.float32)
    rgb[..., 0] = np.abs(xx)
    rgb[..., 1] = np.abs(yy)
    rgb[..., 2] = zz
    norm = np.linalg.norm(rgb[..., :3], axis=2, keepdims=True)
    rgb[..., :3] = np.divide(rgb[..., :3], norm, out=np.zeros_like(rgb[..., :3]), where=norm > 0)
    rgb[..., 3] = inside.astype(np.float32)
    return rgb

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nrrd", type=Path)
    ap.add_argument("--out", type=Path, default=Path("Figure9_Volume_A_P86_300dpi.png"))
    args = ap.parse_args()

    vol, mask, result, summary, _header = analyze_experimental_nrrd(args.nrrd)
    flat_idx = result["flat_indices"]
    candidate_counts = mask.sum(axis=(1, 2))
    selected = int(np.argmax(candidate_counts))
    sampled = np.array([0, 1, 13, 25, 38, 51, 63, 76, 89, 101, vol.shape[0] - 1], dtype=int)

    conf = result["confidence_by_scale"]
    vec = result["vectors_by_scale"]
    fused = result["fused_orientation"]
    eff = result["effective_confidence"]

    rgb_scales = [masked_rgb(vol.shape, flat_idx, vec[s]) for s in range(3)]
    rgb_fused = masked_rgb(vol.shape, flat_idx, fused)
    conf_maps = [masked_scalar(vol.shape, flat_idx, conf[s]) for s in range(3)]
    eff_map = masked_scalar(vol.shape, flat_idx, eff)

    fig = plt.figure(figsize=(15.5, 10.0), dpi=300, facecolor="white")
    gs = fig.add_gridspec(3, 14, height_ratios=[0.92, 1.05, 1.0], hspace=0.30, wspace=0.42)

    # (a) sampled raw slices across full stack
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_title("(a) Sampled raw slices across axis 0 (all slices analyzed for candidate regions)", loc="left", fontsize=11)
    ax_a.axis("off")
    left, gap = 0.005, 0.008
    width = (0.99 - left - gap * (len(sampled) - 1)) / len(sampled)
    for j, sl in enumerate(sampled):
        x = left + j * (width + gap)
        iax = ax_a.inset_axes([x, 0.02, width, 0.86])
        iax.imshow(vol[sl].T, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        iax.set_title(str(int(sl)), fontsize=8, pad=2)
        iax.axis("off")
        if int(sl) == selected:
            iax.add_patch(Rectangle((0, 0), vol.shape[1]-1, vol.shape[2]-1, fill=False, edgecolor="red", linewidth=2.0))

    # (b) candidate content curve
    ax_b = fig.add_subplot(gs[1, 0:4])
    ax_b.plot(np.arange(vol.shape[0]), candidate_counts, marker="o", ms=2.0, lw=1.2)
    ax_b.scatter([selected], [candidate_counts[selected]], s=28, zorder=5)
    ax_b.annotate(f"Slice {selected}", (selected, candidate_counts[selected]), xytext=(12, -22), textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax_b.set_xlabel("Axis-0 slice index")
    ax_b.set_ylabel("Candidate voxels")
    ax_b.set_title("(b) Slice-wise candidate content", fontsize=10)
    ax_b.grid(alpha=0.2)

    # (c) raw selected slice
    ax_c = fig.add_subplot(gs[1, 4:6])
    ax_c.imshow(vol[selected].T, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_c.set_title(f"(c) Raw candidate-rich slice {selected}", fontsize=10)
    ax_c.axis("off")

    # (d-g) scale and fused orientations
    titles = [
        r"(d) S0 orientation" + "\n" + r"($\sigma_g$=0.75, $\sigma_t$=1.5)",
        r"(e) S1 orientation" + "\n" + r"($\sigma_g$=1.25, $\sigma_t$=3.0)",
        r"(f) S2 orientation" + "\n" + r"($\sigma_g$=2.0, $\sigma_t$=5.0)",
        "(g) Confidence-weighted\nfused orientation",
    ]
    rgbs = [*rgb_scales, rgb_fused]
    for k, (title, rgb) in enumerate(zip(titles, rgbs)):
        start = 6 + 2 * k
        ax = fig.add_subplot(gs[1, start:start+2])
        ax.imshow(rgb[selected].transpose(1, 0, 2), interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # (h-k) confidence panels
    conf_titles = ["(h) S0 confidence", "(i) S1 confidence", "(j) S2 confidence", "(k) Effective confidence"]
    maps = [*conf_maps, eff_map]
    spans = [(0,3), (3,6), (6,9), (9,12)]
    for (start,end), title, arr in zip(spans, conf_titles, maps):
        ax = fig.add_subplot(gs[2, start:end])
        im = ax.imshow(arr[selected].T, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)

    # Continuous orientation color key showing mixed colors (including purple/cyan/yellow).
    ax_key = fig.add_subplot(gs[2, 12:14])
    ax_key.set_title("Orientation color key", fontsize=9)
    ax_key.imshow(orientation_key_image(), interpolation="bilinear", origin="lower")
    ax_key.axis("off")
    ax_key.text(0.50, 1.01, "axis-2 (B)", ha="center", va="bottom", fontsize=7.5, transform=ax_key.transAxes)
    ax_key.text(0.98, 0.49, "axis-0 (R)", ha="right", va="center", fontsize=7.0, transform=ax_key.transAxes)
    ax_key.text(0.50, -0.02, "mixed directions", ha="center", va="top", fontsize=7.0, transform=ax_key.transAxes)
    ax_key.text(0.02, 0.49, "axis-1 (G)", ha="left", va="center", fontsize=7.0, transform=ax_key.transAxes)

    fig.suptitle(
        f"Volume A: full-stack candidate-region analysis; candidate-rich slice {selected} displayed",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.out}")
    print(f"Candidate fraction: {summary['candidate_fraction_percent']:.2f}%")


if __name__ == "__main__":
    main()
