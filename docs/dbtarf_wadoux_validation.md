# DB-TARF Wadoux DesignBased Validation

DB-TARF (Design-Based Targeted Adaptive Residual Forest) is implemented in
`code/DBTARF.R` and benchmarked with `code/run_dbtarf_wadoux_comparison.R`.

"Design-Based" qualifies the **evaluation regime** the model targets — the
Wadoux et al. (2021) design-based protocol, in which calibration and validation
are independent simple-random samples drawn from the same finite population, so
the population RMSE estimate is design-unbiased. The estimator itself is
model-based.

## Method

DB-TARF predicts

```
y_hat = RF_selected + lambda * correction
```

where `RF_selected` is a random forest chosen by a **fair profile search** and
`correction` is a cross-fitted residual model accepted only when it improves the
out-of-fold (OOF) RMSE.

- **Fair profile search.** The default path evaluates up to 6 fixed RF /
  residual-RF configurations (one of which is the Wadoux *default* RF) on the
  same cross-fit folds and keeps the profile with the best OOF RMSE. Optionally,
  an explicit per-fold grid search (OOB or inner-CV) can be used instead
  (`DBT_RF_TUNE`, `DBT_RF_TUNE_METHOD`).
- **Correction candidates.** Residual RF, residual Cubist, a spatial RBF/kernel
  residual smoother, their ensemble, and RF recalibration / RF+Cubist stacking.
  All candidates — including the linear/quadratic/stacking recalibrators — are
  scored with the **same cross-fit folds** (OOF vs OOF), and the selected
  recalibrator is re-fit on the full calibration set for deployment (standard
  stacked generalization).
- **Reproducibility.** Every `ranger` fit receives an explicit seed; `ranger`
  (>= 0.18) is seed-reproducible across thread counts, so the whole benchmark is
  bit-for-bit reproducible (verified: two independent runs give identical
  output). The Wadoux default-RF baseline is seeded the same way.
- **Correction grid.** `lambda` is n-dependent: `[-0.20, 0.40]` for n < 80,
  `[-0.25, 0.55]` for 80 <= n < 300, and `[-0.25, 0.60]` for n >= 300 (the range
  used for the n = 500 benchmark below).

### What the comparison is, and is not

The headline reference is the **untuned Wadoux default RF** (`ranger` defaults),
exactly as in Wadoux et al. (2021). DB-TARF, by contrast, performs internal model
selection (profile + correction + lambda) on each calibration sample. So
"DB-TARF beats RF" is honestly read as *"our method, with internal selection,
beats the published default-RF baseline."*

To make the source of the gain transparent, the benchmark also reports
**`RF_tuned`** = the base RF of DB-TARF's selected profile, with **no**
correction (it is co-selected with the correction, not an independently
optimised RF). As the table shows, `RF_tuned` ≈ default RF: **RF tuning buys
essentially nothing on this dataset, so DB-TARF's advantage comes from the
residual correction, not from tuning the forest.** RF and Cubist are untuned
default comparators.

## Reproduction Command

Requires the Wadoux (2021) source data — the `AlexandreWadoux/SpatialValidation`
covariate stack and AGB grid (see `docs/wadoux2021-reference/`). That grid is
gitignored (`external/**`, `data/**`), so the table below is regenerable only by
users who obtain the upstream data.

```r
Sys.setenv(DBT_MODELS      = "RF,Cubist,DBTARF",
           DBT_PROTOCOLS   = "DesignBased",
           DBT_N_ITER      = "30",
           DBT_SAMPLE_SIZE = "500",
           DBT_TRAIN_SEED  = "123",
           DBT_RESULTS_DIR = "results/dbtarf_n500_confirm30")
source("code/run_dbtarf_wadoux_comparison.R")
```

## Confirm30 Result

Scenario `random`, protocol `DesignBased`, n = 500, 30 repetitions, seed 123.
Output directory: `results/dbtarf_n500_confirm30`.

| Model | ME mean | RMSE mean | RMSE sd | r2 mean | MEC mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF (Wadoux default) | 0.21 | 31.63 | 1.56 | 0.88 | 0.88 |
| RF_tuned (selected profile, no correction) | 0.25 | 31.61 | 1.62 | 0.88 | 0.88 |
| Cubist (default) | 1.32 | 32.19 | 1.67 | 0.88 | 0.88 |
| **DBTARF** | **-0.04** | **30.75** | 1.65 | **0.89** | **0.89** |

Paired DBTARF vs the Wadoux default RF (from the script-generated
`dbtarf_winrate_vs_RF.csv`):

| Metric | Value |
| --- | ---: |
| DBTARF wins | 27 / 30 |
| Mean delta RMSE | -0.872 |
| Median delta RMSE | -0.890 |
| SD delta RMSE | 0.514 |
| Mean delta ME | -0.244 |
| Mean delta MEC | +0.007 |
| Mean delta r2 | +0.004 |

Significance of the paired RMSE improvement (computed on the 30 paired deltas):

| Test | Statistic | p-value |
| --- | --- | ---: |
| Wilcoxon signed-rank (H1: delta < 0) | V = 6 | 1.3e-08 |
| Paired t-test (H1: delta < 0) | t = -9.29 | 1.7e-10 |
| 95% CI for mean delta RMSE | [-1.064, -0.680] | — |

DB-TARF also drives the design-based bias to essentially zero (ME -0.04 vs RF
0.21). `RF_tuned` ≈ RF confirms the gain is the residual correction, not tuning.

### Honest caveats

- **Selection multiplicity.** DB-TARF chooses among profiles x correction modes x
  lambda on the calibration sample. The reported `selected_oof_rmse` /
  `profile_search_best_oof_rmse` in the diagnostics are therefore *in-search
  selection scores* and are optimistically biased — they are **not** unbiased
  generalisation estimates. The headline RMSE above is **not** affected by this:
  it is computed on an independent SRS validation sample (design-unbiased) that
  never enters tuning or selection.
- **Calibration/validation overlap.** Calibration and validation are independent
  SRS draws from the same finite population (overlap permitted), matching the
  Wadoux DesignBased protocol; all models share the identical validation draw in
  each iteration.
- **Metric precision.** `wadoux_eval` rounds ME/RMSE/r2/MEC to 2 dp (the source
  convention), so the paired deltas and the 27/30 count inherit that precision.
  The deltas here are ~-0.9 (far from 0) with no ties, so the win count is robust
  to rounding.

## Generalization across Wadoux sampling designs

Wadoux et al. (2021) consider three calibration sampling designs. Repeating the
same n = 500, 30-rep, seed-123 protocol for each (DB-TARF vs the untuned Wadoux
default RF):

| Scenario | RF RMSE | RF_tuned RMSE | DBTARF RMSE | DBTARF wins | Mean delta RMSE | 95% CI | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `random` (simple random) | 31.63 | 31.61 | 30.75 | 27/30 | -0.872 | [-1.06, -0.68] | 1.3e-08 |
| `regular_grid` (systematic) | 32.16 | 32.13 | 31.41 | 28/30 | -0.750 | [-0.95, -0.55] | 3.7e-08 |
| `clustered_random` (two-stage) | 35.65 | 35.63 | 34.49 | 28/30 | -1.156 | [-1.45, -0.87] | 4.9e-08 |

DB-TARF wins ~28/30 with a statistically significant RMSE reduction under all
three designs, and `RF_tuned` ≈ RF in every case (tuning the forest is neutral;
the gain is the residual correction). The improvement is largest under the
hardest clustered design (mean delta RMSE -1.16), where DB-TARF also nearly
eliminates the design-based bias (ME 0.06 vs RF -0.73). Reproduce the other two
designs by setting `DBT_SCENARIO=regular_grid` or `DBT_SCENARIO=clustered_random`
(outputs in `results/dbtarf_n500_regular_grid30` and
`results/dbtarf_n500_clustered_random30`).

## Local Evidence Files

Generated outputs, intentionally gitignored through `results/**`:

- `results/dbtarf_n500_confirm30/dbtarf_summary_by_protocol.csv`
- `results/dbtarf_n500_confirm30/dbtarf_winrate_vs_RF.csv`
- `results/dbtarf_n500_confirm30/dbtarf_paired_deltas_vs_RF.csv`
- `results/dbtarf_n500_confirm30/dbtarf_all_results.csv`
- `results/dbtarf_n500_confirm30/dbtarf_diagnostics.csv`
- `results/dbtarf_n500_confirm30/dbtarf_candidate_scores.csv`
- `results/dbtarf_n500_confirm30/dbtarf_fold_diagnostics.csv`
- `results/dbtarf_n500_confirm30/dbtarf_rf_tuning_scores.csv`
- `results/dbtarf_n500_confirm30/dbtarf_config.csv`
- `results/dbtarf_n500_confirm30/dbtarf_checkpoint_status.csv`
