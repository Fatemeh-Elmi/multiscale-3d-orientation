# Release 1.0.1 — Statistical and reporting corrections

This is a **source-code and reporting correction** to the supplied 1.0.0 archive, not a new scientific estimator.

- Corrected the straight-fiber summary to report **S2**, rather than implying S1 was the lowest-error single scale, and added best-scale counts.
- Added exact two-sided condition-level Spearman permutation tests. Fixed primary tortuosity inference to use seven physically distinct geometries, and corrected the bundle result to exact p=0.0833.
- Integrated these analyses with the main synthetic-validation runner; standalone script also supports previously generated per-volume CSVs.
- Updated reports, documentation, and labels to make aggregations explicit.
- Preserved structure-tensor, confidence, fusion, phantom generation, RNG seeds, and experimental P86 analysis code.
- The raw experimental NRRDs are not part of the archive; no new four-volume experimental rerun is claimed.
- Re-generated SHA256SUMS for all files in this package.

No license was selected: the code owner must choose a license if intended for public reuse.

Validation status: all 205 synthetic cases regenerated and checked; four exact-permutation regression tests passed. Compact source-data/summary CSV files are included in `reference_outputs/`.
