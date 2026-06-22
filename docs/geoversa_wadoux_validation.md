# GeoVersa vs Wadoux Validation

This repository has three different validation levels. They should not be mixed.

## 1. Functional run check

The algorithm is functional when a benchmark run finishes and writes:

- `wadoux_style_rf_conv_checkpoint_status.csv`
- `wadoux_style_rf_conv_summary_by_protocol.csv`
- `wadoux_style_rf_conv_fold_diagnostics.csv`

For the meta-controller path, the diagnostics must show that
`meta_controller_active` is true for the target model/protocol.

## 2. Wadoux-style RF comparison

The primary model comparison is against RF under the same Wadoux data design:

- scenario: `random`, `regular_grid`, or `clustered_random`
- calibration sample size: `500`
- metrics: `ME`, `RMSE`, `Pearson^2`, `MEC`
- paper-critical protocol: `DesignBased`
- repetition count: at least `30` for a first stable estimate, `500` for paper-level replication

The clean criterion is:

- `DesignBased RMSE` lower than RF;
- `DesignBased MEC` higher than RF;
- `DesignBased Pearson^2` higher than RF;
- `ME` closer to zero than RF.

`RandomKFold` or `SpatialKFold` improvements are not sufficient evidence of map-accuracy improvement.

## 3. Paper-level validation-error comparison

The Wadoux figures are validation-statistic errors relative to the population
metric, not only raw model metrics. For exact comparison with the paper-style
quantity, include `Population` in the protocol set and compute:

```text
protocol_metric - population_metric
```

For those delta metrics, closer to zero is better.

The original official `.Rdata` outputs from Wadoux are tracked under
`docs/wadoux2021-reference/official_rdata_manifest.csv`. When those files are
available, run:

```bash
Rscript code/import_wadoux_official_rdata.R
```

Then compare against `docs/wadoux2021-reference/official_delta_summary.csv`.

## Recommended commands

First reproduce the RF reference:

```bash
env \
  WADOUX_RF_SCENARIO=random \
  WADOUX_RF_PROTOCOLS=Population,DesignBased,RandomKFold,SpatialKFold,BLOOCV \
  WADOUX_RF_SAMPLE_SIZE=500 \
  WADOUX_RF_N_ITER=500 \
  WADOUX_R2_METHOD=pearson \
  WADOUX_RF_RESULTS_DIR=results/wadoux2021_rf_reference_random_500iter_pearson \
  Rscript code/run_wadoux_rf_reference.R
```

Run GeoVersa/meta-controller on the same setup. A shorter first confirmation can
use `WADOUX_N_ITER=30`; use `500` for a paper-level run.

```bash
env \
  WADOUX_SCENARIO=random \
  WADOUX_PROTOCOLS=DesignBased,RandomKFold,SpatialKFold,BLOOCV \
  WADOUX_SAMPLE_SIZE=500 \
  WADOUX_N_ITER=30 \
  WADOUX_MODEL_PROFILE=auto \
  WADOUX_MODELS=RF,ConvKrigingNet2D \
  WADOUX_DEVICE=mps \
  GEOVERSA_AUTO_META_CONTROLLER=1 \
  WADOUX_RESULTS_DIR=results/geoversa_meta_wadoux_random_confirm30 \
  Rscript code/Benchmark_GeoVersaMeta.R
```

For an exact paper-style delta calculation, include `Population`. This can be
much slower for GeoVersa because it predicts the full raster domain.

```bash
env \
  WADOUX_SCENARIO=random \
  WADOUX_PROTOCOLS=Population,DesignBased,RandomKFold,SpatialKFold,BLOOCV \
  WADOUX_INCLUDE_POPULATION=TRUE \
  WADOUX_SAMPLE_SIZE=500 \
  WADOUX_N_ITER=500 \
  WADOUX_MODEL_PROFILE=auto \
  WADOUX_MODELS=RF,ConvKrigingNet2D \
  WADOUX_DEVICE=mps \
  GEOVERSA_AUTO_META_CONTROLLER=1 \
  WADOUX_RESULTS_DIR=results/geoversa_meta_wadoux_random_500iter_population \
  Rscript code/Benchmark_GeoVersaMeta.R
```

Generate the validation report:

```bash
env \
  GEOVERSA_RESULTS_DIR=results/geoversa_meta_wadoux_random_confirm30 \
  WADOUX_RF_REFERENCE_DIR=results/wadoux2021_rf_reference_random_designbased_30iter_pearson_20260406 \
  WADOUX_COMPARE_PROTOCOL=DesignBased \
  WADOUX_COMPARE_MODEL=ConvKrigingNet2D \
  Rscript code/compare_geoversa_to_wadoux_reference.R
```

The comparator writes:

- `geoversa_wadoux_reference_comparison.csv`
- `geoversa_wadoux_acceptance_checks.csv`
- `geoversa_wadoux_validation_report.md`
