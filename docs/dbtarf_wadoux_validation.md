# DB-TARF Wadoux DesignBased Validation

DB-TARF (Design-Based Targeted Adaptive Residual Forest) is implemented in
`code/DBTARF.R` and benchmarked with
`code/run_dbtarf_wadoux_comparison.R`.

The model keeps the Wadoux-style default RF as the base learner and applies a
cross-fitted residual correction only when the internal out-of-fold calibration
improves over RF. The default correction grid is conservative
(`lambda` from -0.25 to 0.60) to limit over-correction.

## Reproduction Command

```r
Sys.setenv(DBT_MODELS      = "RF,Cubist,DBTARF",
           DBT_PROTOCOLS   = "DesignBased",
           DBT_N_ITER      = "30",
           DBT_SAMPLE_SIZE = "500",
           DBT_RESULTS_DIR = "results/dbtarf_n500_confirm30")
source("code/run_dbtarf_wadoux_comparison.R")
```

## Confirm30 Result

Output directory: `results/dbtarf_n500_confirm30`

| Model | RMSE mean | RMSE sd | r2 mean | MEC mean |
| --- | ---: | ---: | ---: | ---: |
| RF | 32.095 | 1.866 | 0.880 | 0.878 |
| Cubist | 32.849 | 1.939 | 0.873 | 0.872 |
| DBTARF | 31.222 | 1.772 | 0.886 | 0.884 |

Paired DBTARF vs RF:

| Metric | Value |
| --- | ---: |
| Mean delta RMSE | -0.874 |
| Median delta RMSE | -0.780 |
| SD delta RMSE | 0.438 |
| DBTARF wins | 29 / 30 |
| Mean delta MEC | +0.006 |
| Mean delta r2 | +0.006 |

The only RMSE loss was iteration 1 (`delta_RMSE = +0.260`). All subsequent
iterations favored DBTARF on RMSE.

## Local Evidence Files

These files are generated outputs and are intentionally ignored by git through
`results/**`:

- `results/dbtarf_n500_confirm30/dbtarf_summary_by_protocol.csv`
- `results/dbtarf_n500_confirm30/dbtarf_paired_deltas_vs_RF.csv`
- `results/dbtarf_n500_confirm30/dbtarf_diagnostics.csv`
- `results/dbtarf_n500_confirm30/dbtarf_candidate_scores.csv`
- `results/dbtarf_n500_confirm30/dbtarf_fold_diagnostics.csv`
- `results/dbtarf_n500_confirm30/dbtarf_checkpoint_status.csv`

