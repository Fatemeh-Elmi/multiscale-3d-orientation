# Self-check and limitations

- Unit/regression tests: `python test_statistical_reanalysis.py`.
- Source Python syntax: `python -m compileall -q .`.
- To reproduce 205 synthetic volumes and updated exact tests: `python run_all_synthetic_validation.py --out outputs/synthetic`.
- Only the original experimental-data pipeline is supplied, not the raw NRRD images. Values from experimental Table 4 are previously reported values, not newly independently checked by this revision.
- Figure scripts and experimental filenames are documented in `README.md`.
- Original version 1.0.0 had correctly computed fusion and S2 errors but an incomplete/misleading S1/Oracle description and replicate-level statistical tests. Version 1.0.1 explicitly separates volume and geometric-condition statistics.

**Revision verification:** full 205-phantom run completed successfully; 4/4 permutation regression tests passed. Output tables are bundled in `reference_outputs/`.
