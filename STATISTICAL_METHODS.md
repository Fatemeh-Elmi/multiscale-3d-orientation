# Statistical reproducibility details

## Unit of analysis

For each synthetic volume, errors and reliability descriptors are first summarized over the evaluable voxels (typically with the median). The condition-level value is the **arithmetic mean of the per-volume medians** across the two noise realizations within the same prescribed geometry. The two realizations are not treated as two independently sampled anatomical geometries for condition-level correlations.

Spearman rho is then computed from paired condition-level values. For `n<=8`, the two-sided exact p-value is the fraction of all `n!` permutations of the second variable whose absolute Spearman correlation is at least as large as the observed absolute rho. All permutations are enumerated; no large-sample approximation is used. This choice is especially important for `n=4` bundle settings, where perfect ordering still gives `p=2/24=0.08333`.

For planar tortuosity, both A=0, lambda=24 and A=0, lambda=40 represent the same straight geometry. The primary exact test uses seven distinct geometric cases, pooling these four raw volumes into one straight reference. For transparency, the eight separately prescribed parameter settings are also recorded. With the given data, both give rho=1, but the exact p-values differ.

## Results and limitations

| Comparison | n | rho | Exact two-sided p |
|---|---:|---:|---:|
| Helix spread vs fusion error | 6 | 1.000 | 0.00277778 |
| Helix curvature vs spread | 6 | 0.542857 | 0.29722222 |
| Planar tortuosity spread vs fusion error (seven distinct geometries) | 7 | 1.000 | 0.00039683 |
| Planar tortuosity spread vs fusion error (eight designed settings) | 8 | 1.000 | 0.00004960 |
| Bundle spread vs fusion error | 4 | 1.000 | 0.08333333 |

The test treats the deliberately chosen geometries as the ranking units; p-values do not demonstrate generalization to untested geometries or experimental tissue. Correlations between voxel-wise confidence and error are descriptive, because nearby voxels are spatially correlated.

## Commands

```bash
python run_all_synthetic_validation.py --out outputs/synthetic
```

Alternatively, if synthetic per-volume outputs already exist, the statistical analysis can be re-run without reconstructing any images:

```bash
python condition_level_statistics.py \
  --baseline-csv outputs/synthetic/baseline/synthetic_validation_per_volume.csv \
  --extended-csv outputs/synthetic/extended/extended_validation_per_volume.csv \
  --out outputs/synthetic
```

This file does not imply experimental validation or external algorithm benchmarking; neither was performed by this statistical correction.
