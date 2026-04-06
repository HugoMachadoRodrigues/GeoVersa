# Wadoux 2021 Reference

This folder organizes the external reference used to compare GeoVersa against the validation framework of Wadoux et al. (2021).

## Upstream source

- Repository: `https://github.com/AlexandreWadoux/SpatialValidation`
- Local mirror: `external/SpatialValidation`
- Current local commit: recorded in `official_rdata_manifest.csv`

## Important distinction

The Wadoux paper is the methodological reference for the validation framework, but the **upstream code and the paper are not perfectly aligned**:

- In the paper, `r²` is described as the squared **Pearson** correlation coefficient.
- In the upstream code, `external/SpatialValidation/code/Functions_Spat_CV.R` comments `Pearson's correlation squared`, but the actual line computes `cor(..., method = "spearman")^2`.

Because of this, our repository now separates two use cases:

1. **GeoVersa benchmark mode**: uses `Pearson²`, which matches the paper text.
2. **Official-code reference mode**: can still preserve the upstream code behavior when needed for auditability.

## What the official `.Rdata` files contain

The original Wadoux scripts save:

- `res_random_500.Rdata`
- `res_regular_500.Rdata`
- `res_clustered_random_500.Rdata`

Those files store objects such as `res.ME`, `res.RMSE`, `res.r2`, and `res.MEC`.

These are **not absolute map-accuracy tables**. They are the **validation-statistic errors relative to the population metric**, matching the figures in the paper.

## Scripts added in this repository

- `code/run_wadoux_rf_reference.R`
  - Runs the RF reproduction using the official data and protocol structure.
  - Defaults to `Pearson²`.
  - Set `WADOUX_R2_METHOD=spearman` if you want to mimic the upstream code exactly.

- `code/import_wadoux_official_rdata.R`
  - Reads the original `.Rdata` outputs from `external/SpatialValidation/code/`.
  - Writes a tracked manifest and tidy CSV summaries into this folder.

## Files in this folder

- `official_rdata_manifest.csv`: expected upstream `.Rdata` files and their availability status.
- `official_delta_long.csv`: tidy long-format table of imported official deltas, if available.
- `official_delta_summary.csv`: summary table of imported official deltas, if available.

## Current workflow

1. Reproduce or obtain the official Wadoux `.Rdata` outputs.
2. Run `Rscript code/import_wadoux_official_rdata.R`.
3. Use the imported official deltas as the audited reference.
4. Use `Pearson²` in the GeoVersa benchmark for paper-consistent comparisons going forward.
