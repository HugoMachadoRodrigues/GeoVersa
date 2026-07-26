# Changelog

All notable changes to this project should be documented in this file.

The project is currently in research-stage development and follows a `0.x` versioning line.

## [0.3.0] - 2026-07-06 — first CRAN release

The DB-TARF core is published on CRAN as the **GeoVersa** R package:
<https://CRAN.R-project.org/package=GeoVersa>. Install with
`install.packages("GeoVersa")`.

### Added

- R package scaffolding around the DB-TARF core: `DESCRIPTION`, `NAMESPACE`,
  `R/dbtarf.R` (the implementation, moved out of `code/`), `man/`, and a
  `testthat` suite. Exported API: `dbtarf_train_predict()`, returning point
  predictions (`pred_test`), the base-RF predictions (`pred_test_base`),
  design-based conformal interval half-widths (`pi_q90`, `pi_q95`) and
  per-run `diagnostics`.
- Ordinary-kriging and single-hidden-layer MLP residual corrections, a
  spatially-adaptive shrinkage gate, and design-based conformal prediction
  intervals (plain, covariate-shift-weighted, and spatial-block variants).
- Top-k profile ensemble for DB-TARF (`ensemble_top_k`, default 1 in the
  package — a strict no-op — and 3 in the benchmark runner).
- Do-no-harm literal-RF floors for the non-DB-TARF learners (GPBoost, GeoBART,
  DeepKernel, NeuroMatern, ConvKrigingNet), plus a unified model-library
  validation harness (`code/validate_model_library.R`).
- Protocol-expansion runner `code/run_dbtarf_protocols_comparison.R`
  (Population / RandomKFold / SpatialKFold; BLOOCV excluded by cost).

### Changed

- Research code (ConvKrigingNet2D, the torch learners, runners, data and
  results) is excluded from the package build via `.Rbuildignore`; only the
  DB-TARF core ships to CRAN.

### Fixed — CRAN review (Konstanze Lauseker)

- `DESCRIPTION`: dropped the redundant "Tools for" opener and spelled out
  DB-TARF as "Design-Based Targeted Adaptive Residual Forest" on first use.
- Replaced console `cat()` output with `message()` so progress reporting can
  be suppressed.
- Removed all writes to `.GlobalEnv`: the internal `.Random.seed`
  save/restore now uses `withr::local_preserve_seed()` (`withr` added to
  `Imports`), so a user's random-number state is preserved without the
  package touching the global environment.

## [0.3.0-dev] - 2026-06-20

### Added

- **DB-TARF** (Design-Based Targeted Adaptive Residual Forest): `y_hat = RF_selected + lambda * correction`, where the base RF is chosen by a fair profile search (or optional OOB/CV grid tuning) and a cross-fitted, out-of-fold-selected residual correction is added only when it improves OOF RMSE (`R/dbtarf.R`; runner `code/run_dbtarf_wadoux_comparison.R`; validation `docs/dbtarf_wadoux_validation.md`)
- DB-TARF is the first model in this repository to beat the Wadoux (2021) default-RF reference under the `DesignBased` protocol: RMSE 30.75 vs 31.63, winning 27/30 paired runs (mean ΔRMSE = −0.872, paired Wilcoxon p = 1.3e-08, 95% CI [−1.06, −0.68]), while reducing design-based bias to ≈0 (ME −0.04 vs 0.21). Tuning the forest alone is ≈neutral (`RF_tuned` ≈ RF), so the gain is the residual correction. The win generalizes across all three Wadoux sampling designs (`random`, `regular_grid`, `clustered_random`), 27–28/30 each (Wilcoxon p < 1e-7)
- `RF_tuned` transparency model in the benchmark (base RF of the selected profile, no correction) plus a script-generated paired win-rate summary (`dbtarf_winrate_vs_RF.csv`) and RF tuning trace (`dbtarf_rf_tuning_scores.csv`)

### Changed

- fully reproducible benchmark via explicit per-fit `ranger` seeds (ranger >= 0.18 is seed-reproducible across thread counts); `fit_rf_default_wadoux` gained optional `num.threads`/`seed` arguments (defaults preserve prior behaviour) and the default-RF baseline is now seeded
- DB-TARF model selection scores the linear/quadratic/stacking recalibration candidates with the same cross-fit folds as the residual candidates (OOF vs OOF, removing in-sample optimism)

### Fixed

- fair profile search isolates each profile (one degenerate profile/fold no longer aborts an iteration)
- RBF k-means basis: clamp to the number of distinct coordinates with a graceful fallback (avoids a possible `kmeans` error)
- summary aggregation uses `na.action = na.pass` so a single NA metric no longer drops a whole iteration
- DB-TARF auto fold-count is honored unless `DBT_N_FOLDS` is explicitly set; a malformed `DBT_LAMBDA_GRID` now falls back to the default grid
- removed an unused internal function (`.dbt_fit_residual_rf`)

## [0.2.0-dev] - 2026-04-01

### Added

- comprehensive professional README with full architecture documentation for ConvKrigingNet2D
- Mermaid flowcharts covering full model pipeline, PatchEncoder2D stem, and AnisotropicResidualKrigingLayer
- LaTeX mathematical formulation for all model components (encoder fusion, anisotropic distance, attention-weighted residual interpolation, distance-aware gate)
- benchmark results table comparing ConvKrigingNet2D vs RF (Wadoux 2021) across all three validation protocols
- documented distance-aware beta gate (`dist_scale` fix): `effective_β = sigmoid(logit_β) × exp(−min_d_aniso / dist_scale)` — suppresses kriging correction under spatial extrapolation, critical for SpatialKFold robustness
- training protocol diagram with warmup phase, memory bank refresh, LR decay, and early stopping documentation
- algorithm variants table listing all experimental GeoKriging architectures
- shields.io badges, bibtex citation block, and full references section
- updated repo metadata on GitHub: description and 10 topic tags (deep-learning, digital-soil-mapping, kriging, spatial-statistics, pytorch, geostatistics, r-package, pedometrics, remote-sensing, convolutional-neural-network)

### Architecture highlights documented

- **ConvKrigingNet2D**: three-encoder fusion (tabular MLP 256-dim + PatchEncoder2D CNN 256-dim + coordinate MLP 256-dim) → linear fusion → 256-dim embedding z
- **PatchEncoder2D**: 28ch → ConvBlock2D(32) → MaxPool → ConvBlock2D(64) → MaxPool → ConvBlock2D(96) → AdaptiveAvgPool → Linear(128) → Linear(256); each ConvBlock2D uses double Conv3×3 + BN + GELU + residual skip
- **AnisotropicResidualKrigingLayer**: learns ℓ_major, ℓ_minor, θ from gradients; combines anisotropic spatial distance with attention similarity (256→64 projection) via softmax weighting over K=12 neighbours
- **Memory bank**: training embeddings Z_mem and residuals R_mem stored and refreshed every epoch; Z_mem is detached — no gradient flows through the neighbour side
- **dist_scale gate fix**: `dist_scale=1.0` fixed (not learnable) prevents the gate from being optimised away during training, preserving SpatialKFold stability

### Benchmark result

- Design-Based RMSE ≈ 32.8 Mg ha⁻¹ vs RF published 33.43 — ConvKrigingNet2D competitive on the gold-standard unbiased estimator

## [0.1.0-dev] - 2026-04-01

### Added

- initialized repository-level documentation and versioning structure
- documented project scope, tracked assets, and package migration plan
- added a Git-safe ignore policy for local data, generated artifacts, and external repositories

### Included in initial codebase

- `ConvKrigingNet2D` and related benchmark scripts
- Wadoux-style validation runners
- benchmark, plotting, and model-comparison scripts retained from active research workflow
