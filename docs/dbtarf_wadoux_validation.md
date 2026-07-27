# DB-TARF Wadoux DesignBased Validation

DB-TARF (Design-Based Targeted Adaptive Residual Forest) is implemented in
`R/dbtarf.R` and benchmarked with `code/run_dbtarf_wadoux_comparison.R`.

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
- **Correction candidates.** Residual RF, residual Cubist, a single-hidden-layer
  MLP, ordinary kriging of the residuals (via `gstat`), a spatial RBF/kernel
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
           DBT_RESULTS_DIR = "results/canon_v030_full_random_confirm30")
source("code/run_dbtarf_wadoux_comparison.R")
```

## Confirm30 Result

Scenario `random`, protocol `DesignBased`, n = 500, 30 repetitions, seed 123.
Output directory: `results/canon_v030_full_random_confirm30`.

This is the **canonical v0.3.0 campaign**: the exact code published on CRAN
(`R/dbtarf.R`) with runner defaults (`ensemble_top_k = 3`, `n_folds = 5`) and
the complete optional-backend set installed (`Cubist`, `gstat`, `sp`, `nnet`),
so the full residual-correction candidate library is available.

| Model | ME mean | RMSE mean | RMSE sd | r2 mean | MEC mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF (Wadoux default) | 1.06 | 32.49 | 1.22 | 0.88 | 0.88 |
| RF_tuned (selected profile, no correction) | 1.14 | 32.48 | 1.25 | 0.88 | 0.88 |
| Cubist (default) | 1.85 | 33.26 | 1.42 | 0.87 | 0.87 |
| **DBTARF** | **0.84** | **31.60** | 1.31 | **0.88** | **0.88** |

Paired DBTARF vs the Wadoux default RF:

| Metric | Value |
| --- | ---: |
| DBTARF wins | 28 / 30 |
| Mean delta RMSE | -0.895 |
| Median delta RMSE | -0.970 |
| SD delta RMSE | 0.497 |
| Mean delta ME | -0.222 |
| Mean delta MEC | +0.007 |
| Mean delta r2 | +0.005 |

Significance of the paired RMSE improvement (computed on the 30 paired deltas):

| Test | Statistic | p-value |
| --- | --- | ---: |
| Wilcoxon signed-rank (H1: delta < 0) | V = 4 | 6.5e-09 |
| Paired t-test (H1: delta < 0) | t = -9.87 | 4.4e-11 |
| 95% CI for mean delta RMSE | [-1.080, -0.710] | — |

DB-TARF also reduces the design-based bias (ME 0.84 vs RF 1.06).
`RF_tuned` ≈ RF confirms the gain is the residual correction, not tuning.

> **Note on superseded figures.** Earlier revisions of this document reported
> RMSE 31.63 → 30.75 with 27/30 wins. Those came from a pre-`ensemble_top_k`
> configuration; because DB-TARF's RNG consumption differs between
> configurations, the RF baseline draw shifts too, which is why *both* columns
> moved. The paired effect size is essentially unchanged (≈ -0.87 then,
> -0.895 now). All figures in this document, the README and the manuscript now
> come from the single canonical campaign described above.

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
  convention), so the paired deltas and the win counts inherit that precision.
  The deltas here are ~-0.9 (far from 0) with no ties, so the win count is robust
  to rounding.

## Generalization across Wadoux sampling designs

Wadoux et al. (2021) consider three calibration sampling designs. Repeating the
same n = 500, 30-rep, seed-123 protocol for each (DB-TARF vs the untuned Wadoux
default RF):

| Scenario | RF RMSE | RF_tuned RMSE | DBTARF RMSE | DBTARF wins | Mean delta RMSE | 95% CI | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `random` (simple random) | 32.49 | 32.48 | 31.60 | 28/30 | -0.895 | [-1.08, -0.71] | 6.5e-09 |
| `regular_grid` (systematic) | 32.28 | 32.27 | 31.38 | 30/30 | -0.901 | [-1.05, -0.75] | 9.3e-10 |
| `clustered_random` (two-stage) | 35.78 | 35.60 | 34.57 | 28/30 | -1.205 | [-1.47, -0.94] | 1.8e-08 |

All three rows use **30 repetitions** (earlier revisions reported 19 and 10 for
the grid and clustered designs). Output directories:
`results/canon_v030_full_{random,regular_grid,clustered_random}_confirm30`.

DB-TARF wins 28-30 of 30 with a statistically significant RMSE reduction under
all three designs, and `RF_tuned` ≈ RF in every case (tuning the forest is
neutral; the gain is the residual correction). The improvement is largest under
the hardest clustered design (mean delta RMSE -1.21), where DB-TARF also cuts
the design-based bias by roughly a third (ME -0.88 vs RF -1.36). Reproduce the
other two designs by setting `DBT_SCENARIO=regular_grid` or
`DBT_SCENARIO=clustered_random` (outputs in
`results/canon_v030_full_regular_grid_confirm30` and
`results/canon_v030_full_clustered_random_confirm30`).

## Robustness across validation estimands

The DesignBased result above scores DB-TARF on the estimand it is named for.
To check that the advantage is not an artifact of that choice, the comparison
was repeated under three further protocols with the DB-TARF configuration,
sample size (n = 500) and seed held fixed, 10 repetitions each
(`code/run_dbtarf_protocols_comparison.R`, output
`results/canon_v030_protocols_confirm10`):

| Estimand | n | RF RMSE | DBTARF RMSE | Mean delta RMSE | 95% CI | Wins | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Population (SRS 20 000) | 10 | 32.175 | 31.506 | -0.669 | [-0.82, -0.51] | 10/10 | 0.00098 |
| DesignBased (SRS 500) | 30 | 32.493 | 31.598 | -0.895 | [-1.08, -0.71] | 28/30 | 6.5e-09 |
| Random 10-fold CV | 10 | 32.240 | 31.115 | -1.125 | [-1.49, -0.76] | 10/10 | 0.00098 |
| Spatial block CV (350 km) | 10 | 34.722 | 33.447 | -1.275 | [-1.68, -0.87] | 10/10 | 0.00098 |

MEC also improves under each: 0.879 -> 0.885 (Population), 0.880 -> 0.888
(random CV), 0.859 -> 0.871 (spatial block CV).

**What this does and does not claim.** Only the Population and DesignBased rows
estimate *map accuracy*; the two cross-validation rows are **not** valid
map-accuracy estimators — that is the central result of Wadoux et al. (2021)
and is not disputed here. They are reported as *model-comparison* estimands
only, to test whether the DB-TARF-vs-RF ranking survives a change of scoring
protocol. It does, in all 10 draws under each protocol.

**The CV protocols FAVOUR DB-TARF; they are not adversarial.** DB-TARF accepts a
correction only if it lowers OOF RMSE on *random* folds (`caret::createFolds`) —
the same construction as the RandomKFold protocol — and its gate length is
chosen by spatial CV, the same construction as SpatialKFold. These are therefore
the least independent of the four checks, and their deltas should be read as an
upper bound. The Population row is the one that carries evidential weight.

Two honest notes:

- `p = 0.00098` is the **smallest value attainable** from a one-sided paired
  Wilcoxon test at n = 10 (2^-10). It certifies that all ten repetitions favour
  DB-TARF and nothing stronger.
- The four deltas order as -0.67 < -0.90 < -1.13 < -1.28, but this is an
  ordering of point estimates only: adjacent protocols are not statistically
  distinguishable (random vs spatial block CV, paired p = 0.36) and the CIs
  overlap heavily. It is also NOT an ordering in spatial stress — random 10-fold
  CV holds out points interspersed among the training points and is the least
  spatially demanding of the four. The defensible statement is that the largest
  gain occurs under spatial block CV.
- These are 10 calibration draws scored three ways, not 30 independent
  repetitions: within a draw the protocol results are dependent (the random- and
  spatial-CV deltas correlate at r = 0.59).
- The CV rows train on strict subsets of the calibration sample (450 points for
  10-fold; data-dependent for spatial blocks), so their absolute RMSE is not
  commensurable with the two design-based rows.
- All four estimands were run under the simple-random calibration design only.
- The DesignBased row comes from a separate campaign
  (`results/canon_v030_full_random_confirm30`, 30 reps, full tuning budget), not
  from the protocols runner.

BLOOCV is deliberately excluded on cost: ~10^3 sequential fits per repetition.
Note that it would be the most adversarial model-comparison estimand available,
being the one least aligned with DB-TARF's internal selection.

Reproduce (the runner is resumable — re-invoke with the same results dir to
continue an interrupted run):

```sh
DBTP_N_ITER=10 DBTP_PROTOCOLS="Population,RandomKFold,SpatialKFold" \
DBTP_SAMPLE_SIZE=500 DBTP_POP_TEST_SIZE=20000 DBTP_KFOLD_K=10 \
DBTP_VAL_DIST_KM=350 DBTP_TRAIN_SEED=123 \
DBTP_N_FOLDS=5 DBTP_RF_TUNE_MAX_GRID=18 DBTP_RESID_RF_TUNE_MAX_GRID=12 \
DBTP_FAIR_PROFILE_MAX=6 DBTP_ENSEMBLE_TOP_K=3 \
DBTP_RESULTS_DIR="results/canon_v030_protocols_confirm10" \
  Rscript code/run_dbtarf_protocols_comparison.R
```

## Local Evidence Files

Generated outputs, intentionally gitignored through `results/**`:

- `results/canon_v030_full_random_confirm30/dbtarf_summary_by_protocol.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_winrate_vs_RF.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_paired_deltas_vs_RF.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_all_results.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_diagnostics.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_candidate_scores.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_fold_diagnostics.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_rf_tuning_scores.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_config.csv`
- `results/canon_v030_full_random_confirm30/dbtarf_checkpoint_status.csv`
