# Changelog

All notable changes to this project should be documented in this file.

The project is currently in research-stage development and follows a `0.x` versioning line.

## [Unreleased] — 0.4.0-dev (geostatistics as hyperparameters)

The variogram family, its range as a working scale, the lag binning, the
estimator and the anisotropic geometry now live in the profile search and are
selected out-of-fold. Campaign result (30 reps per design, n = 500):

| Design | RF | DB-TARF | wins | Wilcoxon p |
|---|---:|---:|---:|---:|
| random | 32.493 | 31.624 | 28/30 | 6.5e-09 |
| regular_grid | 32.280 | 31.400 | 30/30 | 9.3e-10 |
| clustered_random | 35.778 | 34.538 | 29/30 | 1.9e-09 |

Accuracy is unchanged relative to the fixed recipe (+0.026, +0.021, -0.035 —
inside repetition noise), so the contribution is not an RMSE gain. What the
campaign shows is that the freedom is USED, and used discriminatingly:

| Design | binning/estimator | range scale | anisotropy | conventional recipe kept |
|---|---:|---:|---:|---:|
| random | 9 | 7 | 4 | 10/30 |
| regular_grid | 4 | 5 | 3 | 18/30 |
| clustered_random | 11 | 5 | 0 | 14/30 |

Under clustered sampling anisotropy is refused in every repetition (a clustered
sample cannot support estimating direction) while the binning is displaced most
often (clustering distorts the distance distribution, so a fixed 50% cutoff with
15 lags is a poor fit). The same mechanism discriminates between the
geostatistical parameters the design can support and those it cannot.

**Honest boundary, now stated in the manuscript.** Pooled over 90 repetitions
the selected corrector is a residual random forest 68 times; ordinary kriging
wins outright 3 times and a purely geostatistical interpolator 4 times. The
geostatistics therefore contributes the representation (variogram-scaled radial
basis features, anisotropic geometry) and the regularisation (variogram-scaled
spatial gate), not the final estimator. The manuscript says so explicitly rather
than claiming kriging has returned to the centre of the map.

## [Unreleased] — earlier 0.4.0-dev notes

### Added — anisotropy, admitted on merit

Geometric anisotropy (major-axis angle + minor/major ratio) is estimated from
the residual field and applied to ALL three spatial components: the radial
basis (query points and centres), the kernel residual smoother, and the
ordinary-kriging correction. A ratio of 1 reduces the transform to a pure
rotation, so the isotropic method is recovered exactly — verified bit-for-bit.

The geometry is not imposed: the anisotropic variant enters the fair profile
search as an extra candidate and must beat the isotropic one on the same
out-of-fold criterion (`aniso_search`, default TRUE).

**Estimating the geometry took three attempts; the first two are recorded
because they are instructive failures.** (1) Fitting one variogram per
direction and taking min/max of the fitted ranges is biased by order
statistics — it reports anisotropy for perfectly isotropic fields. (2) Scoring
candidate geometries by variogram fit (SSE) in the transformed space is
invalid — the transform changes the lag binning and pair counts, so the SSEs
are not comparable. (3) The implemented estimator scores each candidate
geometry by k-fold out-of-fold prediction error of a kernel smoother, on the
same held-out points, with a bandwidth set from the median pairwise distance in
the transformed space. Validated on synthetic fields: exact isotropy recovered
for isotropic fields (no false positives), correct angles for anisotropic ones,
errors only ever false negatives.

Result on the Wadoux Amazonian benchmark (30 reps per design, n = 500):

| Design | anisotropy accepted | DB-TARF iso | DB-TARF aniso offered | dRMSE |
|---|---:|---:|---:|---:|
| random | 6/30 | 31.598 | 31.589 | -0.010 |
| regular_grid | 6/30 | 31.379 | 31.378 | -0.001 |
| clustered_random | 0/30 | 34.572 | 34.563 | -0.009 |

Accuracy is unchanged (differences are inside repetition noise), so this
dataset is effectively isotropic at this scale. The informative result is the
clustered row: anisotropy is refused in every repetition, because a clustered
sample carries little information about directional structure and the
out-of-fold criterion declines a geometry the design cannot support.

Written up as Sections 3.1.6 and 5.4 of the manuscript.

### Changed

- Package version 0.3.0 -> 0.4.0.9000. The code now diverges from the CRAN
  0.3.0 release; a CRAN update is planned once the anisotropy work settles.

## [Unreleased — 0.3.0 line]

### Added — robustness across validation estimands

The DB-TARF-vs-RF comparison was repeated under three protocols beyond
`DesignBased`, with the DB-TARF configuration, sample size (n = 500) and seed
held fixed at the canonical settings, 10 repetitions each
(`results/canon_v030_protocols_confirm10`):

| Estimand | n | RF | DB-TARF | ΔRMSE | Wins | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Population (SRS 20 000) | 10 | 32.175 | 31.506 | −0.669 | 10/10 | 0.00098 |
| DesignBased (SRS 500)* | 30 | 32.493 | 31.598 | −0.895 | 28/30 | 6.5e-09 |
| Random 10-fold CV | 10 | 32.240 | 31.115 | −1.125 | 10/10 | 0.00098 |
| Spatial block CV (350 km) | 10 | 34.722 | 33.447 | −1.275 | 10/10 | 0.00098 |

*The DesignBased row is the separate 30-rep campaign, shown for comparison.

DB-TARF wins in all 10 draws under each additional protocol and mean MEC
improves under each. Qualifications, all now stated in the write-ups: only the
Population and DesignBased rows estimate map accuracy (spatial CV is not a valid
accuracy estimator — Wadoux et al. 2021); the two CV protocols are *aligned with
DB-TARF's own selection criteria* (random-fold OOF acceptance, spatial-CV gate
choice) and so favour it rather than stressing it; these are 10 calibration
draws scored three ways, not 30 independent repetitions (within-draw r = 0.59);
and `p = 0.00098` is merely the floor of a one-sided paired Wilcoxon at n = 10.
The deltas order −0.67 → −0.90 → −1.13 → −1.28, but adjacent protocols are not
distinguishable (p = 0.36) and the ordering is not one of spatial stress —
random 10-fold CV is the least spatially demanding. BLOOCV excluded by cost.

Written up as Section 5.3 of the manuscript, a new section in
`docs/dbtarf_wadoux_validation.md`, and a README subsection. Figure 4 (which
had been orphaned and carried superseded numbers) was regenerated from the
canonical result directories by the new `code/figures_protocols.R`, so it can
no longer drift from the tables.

### Changed — canonical v0.3.0 reference campaign

All DB-TARF-vs-RF figures in `README.md`, `docs/dbtarf_wadoux_validation.md`
and the manuscript are now sourced from a **single** campaign, replacing
several earlier runs whose numbers disagreed because they used different
configurations (pre-`ensemble_top_k` vs `top_k = 3`) and therefore different
RNG streams and RF baseline draws.

The canonical campaign runs the exact code published on CRAN (`R/dbtarf.R`)
with runner defaults (`ensemble_top_k = 3`, `n_folds = 5`), `n = 500`,
`DesignBased`, seed 123, and **30 repetitions in each of the three sampling
designs** — removing the uneven repetition counts (30 / 19 / 10) previously
reported as a limitation. Results in
`results/canon_v030_full_{random,regular_grid,clustered_random}_confirm30`:

| Design | RF | RF_tuned | Cubist | DB-TARF | ΔRMSE | Wins | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 32.49 | 32.48 | 33.26 | 31.60 | −0.895 | 28/30 | 6.5e-09 |
| regular_grid | 32.28 | 32.27 | 33.17 | 31.38 | −0.901 | 30/30 | 9.3e-10 |
| clustered_random | 35.78 | 35.60 | 37.16 | 34.57 | −1.205 | 28/30 | 1.8e-08 |

`RF_tuned` ≈ `RF` in every design, so the gain is the residual correction and
not forest tuning. Design-based bias (ME) shrinks toward zero in all three.
These figures supersede the `[0.3.0-dev]` entry below (RMSE 30.75 vs 31.63),
which came from a pre-`ensemble_top_k` configuration.

### Fixed

- Reinstated the optional `Cubist` and `gstat` backends, which had gone
  missing from the local R library. Without them DB-TARF silently lost its
  Cubist and ordinary-kriging residual-correction candidates (and the Cubist
  comparator returned `NA`), so a first pass of the campaign ran against a
  degraded candidate library and was discarded.
- README no longer leads its results section with the 3-repetition deep-backbone
  table; the 30-repetition DB-TARF campaign comes first, and the two
  non-comparable RF baselines are explicitly reconciled.

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
