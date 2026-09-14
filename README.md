# Final Python code for the multiscale 3-D orientation manuscript

This repository provides a clean, standalone implementation of the multiscale 3D orientation and reliability analysis framework

## What is implemented

- 3-D gradient structure tensor with three fixed scale pairs:
  - S0 = (sigma_g, sigma_t) = (0.75, 1.5) vox
  - S1 = (1.25, 3.0) vox
  - S2 = (2.0, 5.0) vox
- local axial orientation = eigenvector of the smallest tensor eigenvalue
- line-shape confidence + robust P5-P95 structural-strength normalization
- confidence-weighted dyadic fusion
- cross-scale stability, effective confidence, angular spread
- seven synthetic phantom families with geometry-derived ground-truth masks
- experimental P86 candidate-region analysis for four SR-PhC-microCT volumes
- 300-dpi experimental visualization generator
- reference-result verification utilities

## Experimental P86 workflow

Only the experimental data use the P86 candidate mask. The mask is generated from the normalized volume as follows:

1. clip raw intensity at P1/P99 and rescale to [0,1]
2. Gaussian smooth with sigma = 1 voxel **for mask generation only**
3. threshold at the 86th percentile of the smoothed normalized volume
4. remove 3-D connected components < 120 voxels
5. 3-D morphological closing with a radius-1 spherical element
6. fill holes
7. calculate/report orientation and reliability only within this candidate mask

The multiscale tensor itself is evaluated from the normalized volume using its own sigma_g/sigma_t values; the extra sigma=1 smoothing is not added to the orientation estimator.

## Expected experimental filenames

Place these four NRRDs in one directory:

- A: `experimental_volume_A.nrrd`
- B: `experimental_volume_B.nrrd`
- C: `experimental_volume_C.nrrd`
- D: `experimental_volume_D.nrrd`

## Reproduce experimental summary metrics

```bash
python run_experimental_abcd.py --data-dir /path/to/nrrds --out outputs/experimental
```

Expected rounded values:

| Volume | Candidate % | Median q_eff | Stability | Spread (deg) | Best-supported S0 % | S1 % | S2 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 12.83 | 0.162 | 0.915 | 11.80 | 27.4 | 22.3 | 50.4 |
| B | 12.29 | 0.143 | 0.923 | 11.27 | 38.5 | 18.6 | 42.9 |
| C | 12.96 | 0.159 | 0.913 | 11.71 | 27.5 | 13.4 | 59.1 |
| D | 13.30 | 0.179 | 0.933 | 10.55 | 38.3 | 19.9 | 41.8 |

## Verify experimental summary metrics

```bash
python verify_reference_results.py --data-dir /path/to/nrrds
```

## Reproduce experimental visualization

```bash
python make_experimental_volume_a.py "/path/to/nrrds/Cropped volume_2-3low.nrrd" --out experimental_Volume_A_P86_300dpi.png
```

## Run all synthetic validation

```bash
python run_all_synthetic_validation.py --out outputs/synthetic
```

The synthetic design contains 205 volumes/challenges in total:

- 128 straight-fiber noise/blur volumes
- 12 analytic helical volumes
- 1 primary orthogonal crossing challenge
- 32 radius-sweep volumes
- 8 crossing-angle volumes
- 16 planar-tortuosity volumes
- 8 bundle-dispersion volumes

Baseline RNG seed: `20260903`; extended morphology RNG seed: `20260904`.

Key expected baseline results include:

- weighted fusion straight median = ~0.492 deg
- equal-weight fusion = ~0.588 deg
- S1 = ~0.382 deg
- retrospective oracle = ~0.223 deg
- weighted fusion better than equal weighting in 89.1% of straight volumes
- pooled confidence/error Spearman rho = ~-0.266
- helix spread/error Spearman rho = ~0.993

Key extended results include:

- tortuosity spread/error Spearman rho = ~0.985
- bundle spread/error Spearman rho = ~0.738

## Important interpretation

The P86 mask is an image-derived fibrous candidate selection, not a chemically specific collagen segmentation. The experimental volumes have no independent voxel-wise 3-D orientation ground truth; therefore they demonstrate transfer/reliability behavior, not absolute orientation accuracy.
