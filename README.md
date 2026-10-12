# Multiscale 3D Fiber Orientation and Reliability Analysis

Reproducible Python source accompanying the manuscript by Fatemeh Elmi.

## Implemented computation

- Grayscale 3D gradient structure tensors at three **fixed** `(sigma_g, sigma_t)` pairs in voxels:
  S0 `(0.75, 1.5)`, S1 `(1.25, 3.0)`, S2 `(2.0, 5.0)`.
- Local axial orientation from the smallest structure-tensor eigenvector.
- Scale-specific structural confidence from tensor line-likeness and P5-P95-normalized transverse strength.
- Sign-invariant, confidence-weighted dyadic fusion.
- Effective confidence, cross-scale stability, weighted angular spread and the **best-supported** scale (highest image-derived confidence, **not** necessarily the lowest-error scale).
- Seven synthetic phantom families (205 volumes/challenges), with geometry-derived ground truth wherever a unique fiber axis exists.
- Experimental P86 candidate-region analysis, without collagen-specific segmentation.

The estimator and synthetic image generation have **not** been changed for this statistical update. Synthetic images are generated/degraded and clipped to `[0,1]`; unlike experimental input, they are **not** subjected to P1-P99 image normalization.

## Setup

Requires Python, NumPy, SciPy, and Matplotlib (see `requirements.txt`).

```bash
python -m pip install -r requirements.txt
python test_statistical_reanalysis.py
```

## Reproduce all synthetic results and the updated statistical analysis

```bash
python run_all_synthetic_validation.py --out outputs/synthetic
```

The output contains (a verified result snapshot is also shipped in `reference_outputs/`):

- `baseline/synthetic_validation_per_volume.csv` (128 straight + 12 helical volumes)
- `extended/extended_validation_per_volume.csv` (64 additional volumes)
- `condition_level_means.csv` (mean of per-volume median metrics within geometric conditions)
- `condition_level_correlations.csv` (two-sided **exact** Spearman permutation tests)
- `condition_level_statistics.json` and `combined_summary.json`
- summary CSVs and illustrative figures from both validation programs

The separate primary orthogonal crossing challenge contributes **one** further experiment, bringing the full total to **205**.

### Correct straight-fiber results (128 volumes)

| Estimator | Median per-volume axial error |
|---|---:|
| S0 | 1.364° |
| S1 | 0.382° |
| **S2** | **0.223°** |
| Equal-weight dyadic fusion | 0.588° |
| Confidence-weighted dyadic fusion | 0.492° |

S2 is the lowest-error individual scale in **113/128** straight-fiber volumes; weighted fusion improves on equal weighting in **114/128 (89.1%)**. A ground-truth-selected per-volume oracle also has median 0.223°, but oracle and S2 are distinct definitions and do not necessarily select the same scale in each volume. Fusion **does not** universally improve accuracy over S2.

### Condition-level Spearman reanalysis

Noise realizations are averaged **within** each geometric setting before condition-level correlation. Exact two-sided permutation tests are used (not the asymptotic p-values from correlations across replicate volumes).

| Comparison | Distinct geometries / conditions | Spearman rho | Exact two-sided p |
|---|---:|---:|---:|
| Helix: spread vs fusion error | 6 | 1.000 | 0.002778 |
| Helix: curvature vs spread | 6 | 0.543 | 0.297222 |
| Tortuosity: spread vs fusion error | **7** | 1.000 | 0.000397 |
| Five-fiber bundle: spread vs fusion error | 4 | 1.000 | **0.083333** |

There are **8 prescribed** wavelength/amplitude settings for tortuosity, but the two `A=0` settings share the **same straight geometry**, so they are pooled for inference. The 8-setting descriptive result (rho 1.000, p 0.0000496) is also exported. The four bundle conditions yield a perfect rank order but **do not** meet the p<0.05 threshold under the exact test. These tests assess ranking across selected synthetic designs, not generalizable or voxel-wise predictive accuracy.

The code also preserves original **per-volume** correlations for descriptive comparison; use the **condition-level** fields above to report the statistical findings in the revised manuscript. The pooled, spatially correlated straight-fiber voxel analysis remains descriptive (Spearman rho about -0.266).

## Experimental P86 workflow

Experimental intensities are clipped at P1/P99 and rescaled to `[0,1]`. The candidate mask is made from a sigma=1-voxel-smoothed **copy** using the P86 threshold, connected-component removal (<120 voxels), radius-1 3D closing, and hole filling. The extra mask smoothing is **not** part of the estimator input; structure tensors operate on the normalized volume.

Use the four previously acquired experimental NRRD volumes and supply their location:

```bash
python run_experimental_abcd.py --data-dir /path/to/nrrds --out outputs/experimental
python verify_reference_results.py --data-dir /path/to/nrrds
```

`verify_reference_results.py` expects these actual filenames by default:

- A: `Cropped volume_2-3low.nrrd`
- B: `Cropped volume_3_310l.nrrd`
- C: `Cropped volume_1_1L(1).nrrd`
- D: `Cropped volume_2_1L.nrrd`

Both experimental commands accept `--file-a`, `--file-b`, `--file-c`, and `--file-d` to override the defaults when your NRRDs have different filenames. The experimental runner exports `Table4_experimental_summary.csv` and `.json`. For Volume A visualization:

```bash
python make_experimental_volume_a_figure.py '/path/to/nrrds/Cropped volume_2-3low.nrrd' --out experimental_Volume_A_P86_300dpi.png
```

Experimental images are **not distributed** in this archive. The previously reported rounded experimental metrics are: A (12.83% candidate, q_eff 0.162, stability 0.915, spread 11.80°); B (12.29%, 0.143, 0.923, 11.27°); C (12.96%, 0.159, 0.913, 11.71°); D (13.30%, 0.179, 0.933, 10.55°). These are not independently revalidated without the NRRD data.

## Interpretation and provenance

The P86 mask identifies image-derived fibrous **candidate** voxels; it is not chemically specific collagen or fibrosis segmentation. Without independent 3D experimental ground truth, real data demonstrate application and sensitivity, not absolute orientation accuracy. The code and report were statistically revised without introducing or re-fitting any estimator parameters. See `STATISTICAL_METHODS.md` and `RELEASE_NOTES.md`.

No license has been selected in this package; the repository owner must decide whether and under which terms to add one before reuse permissions are specified.
