# Benchmark data

The analysis reads the public materials of Wadoux et al. (2021). They are not
redistributed here: the repository gitignores `external/`, `data/` and
`results/`, so a fresh clone has the code but not the inputs.

## What to fetch

    git clone https://github.com/AlexandreWadoux/SpatialValidation \
      external/SpatialValidation

That provides `ABG1.tif`, the Amazonian above-ground biomass surface of
Baccini et al. (2012) at 1 km, together with 28 covariate rasters and
`code/polygon.Rdata`. Loaded as a finite population it is 1,310,143 cells.

## Where the code looks

`code/project_paths.R` resolves the data root in this order:

1. `GEOVERSA_DATA`, if set
2. `<repo>/external/SpatialValidation`
3. a sibling `Deep Kriging/external/SpatialValidation`, for machines that
   still carry the pre-rename checkout

So either clone into the default location or point at your own:

    export GEOVERSA_DATA=/somewhere/else/SpatialValidation

The repository root resolves the same way, via `GEOVERSA_ROOT` or by walking up
from the working directory until it finds `DESCRIPTION` alongside `R/`. No
script contains an absolute path.

## Reproducing one repetition

    Rscript -e 'cat(Sys.getenv("GEOVERSA_DATA"), "\n")'   # check the data root
    DBT_SCENARIO=random DBT_PROTOCOLS=DesignBased DBT_MODELS=RF,Cubist,DBTARF \
    DBT_SAMPLE_SIZE=500 DBT_N_ITER=1 DBT_TRAIN_SEED=123 DBT_ENSEMBLE_TOP_K=1 \
    DBT_RESULTS_DIR=results/check_one \
    Rscript code/run_dbtarf_wadoux_comparison.R

One repetition takes roughly twenty minutes, most of it the nested CV+
calibration, which re-runs the whole selection inside each of five folds.
