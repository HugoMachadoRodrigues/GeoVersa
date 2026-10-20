# GeoVersa 0.5.1

The first release archived on Zenodo with the repository public. 0.5.0 was
tagged while the repository was private, and Zenodo, which fetches the release
archive without credentials, could not retrieve it.

No change to the estimator, the intervals or the campaign runners. Every
number in the manuscript was produced at commit `5db0ec3`, which both 0.5.0 and
0.5.1 contain, so the results do not depend on which of the two is used.

* `code/tables_manuscript.R` writes the paired-difference table in four columns
  instead of seven, folding the confidence interval into the mean and reporting
  the Monte Carlo standard error range in its log rather than as a column. The
  footnote marker of the sensitivity table is `\textsuperscript` rather than a
  maths superscript over a zero-width box, which Word rendered as a square.
* `tools/data-provenance-report.sh` states that the repository is public and
  prints the clone command, so the report can be re-derived from a fresh clone.
* The manuscript sources under `Text/ecomod_submission/` carry the final table
  layout, the Word conversion fixes and the cover letter.

# GeoVersa 0.5.0

The release archived alongside the manuscript. Everything below postdates 0.4.0
and changes what the estimator does or what a run records about itself, so the
0.4.0 notes no longer describe the code.

## Prediction intervals

* **CV+ calibration is nested inside the model selection.** 0.4.0 built CV+
  intervals from folds of a model whose profile, corrector and shrinkage had
  already been chosen on the whole calibration sample. The conformity scores
  were therefore scored against a configuration that had seen them, and the
  finite-sample argument does not survive that. `.dbt_cvplus_nested()` re-runs
  the entire selection inside each calibration fold, so a score is produced by
  a model fitted without the point it scores. The fold partition is drawn
  independently of the response.

* **The geostatistical recipe reaches the calibration.** The variogram family,
  practical range, lag binning, estimator and anisotropy setting were read by
  lexical scoping in the interval path rather than passed, so a run that had
  selected a recipe could calibrate against a different one. They are arguments
  now.

* **The map and the interval are one estimator.** `ensemble_top_k` defaults to
  1. The previous default blended the three best profiles for the map while the
  interval came from a single profile, so the reported coverage belonged to a
  model that was not the one being mapped.

## What a run records

* `DBT_VAL_SIZE` separates the validation draw from the calibration draw. Tied
  together, the precision of the measurement moved with the quantity being
  measured, which is enough to make a sample-size sweep look conclusive while
  comparing one experiment with itself.

* `DBT_N_PSU`, `DBT_M_PER_PSU` and `DBT_KMEANS_CENTERS` are read once at the top
  and written to the configuration record, with a `calib_size` field stating the
  calibration size the design will actually produce. Under the clustered design
  that size is the product of the first two and never came from
  `DBT_SAMPLE_SIZE`, which nothing recorded.

* `geo_basis` gates the variogram-derived representation for the ablation, and
  the switch now reaches the model. Plain coordinates survive it deliberately.

* `DBT_DUMP_POINTS`, off by default, writes one row per validation point:
  coordinates, distance to the nearest calibration point, observation,
  prediction and both interval bounds. No other output is finer than one row
  per repetition, which is why the failure could be counted but not located.

* Every campaign directory records the git commit, whether the tree was dirty,
  the R version, a timestamp and the sampling parameters that governed it.

## Portability

* No script carries an absolute path. `code/project_paths.R` resolves the
  repository, the benchmark data and the campaign outputs, and every other
  script bootstraps it.

# GeoVersa 0.4.0

Two defects reported in peer review of the accompanying manuscript are fixed in
this release, and the geostatistical settings that were previously fixed by
convention are now selected from data.

## Prediction intervals

* Interval calibration now uses **CV+** (Barber et al., 2021) rather than the
  earlier scheme, which applied a single out-of-fold quantile to predictions
  from the model refitted on the whole calibration sample. That combination is
  not split conformal and carries no finite-sample guarantee. `.dbt_cvplus()`
  builds each interval from the per-fold predictions and the corresponding
  absolute residuals, so the stated coverage is the one CV+ actually provides.

## Variogram and anisotropy

* **Effective range is now comparable across families.** The fitted range
  parameter was previously carried over unchanged from whichever family was
  selected, but the same nominal value means different things for the
  exponential, spherical, Gaussian and Matern models. `.dbt_practical_range()`
  evaluates the fitted variogram and returns the distance at which it reaches
  95 per cent of the sill, so the families are placed on one scale.

* The variogram family, the lag cutoff and count, and the choice between the
  Matheron and Cressie estimators are now candidates in the out-of-fold search
  rather than analyst settings. A parsimony tolerance keeps the simplest family
  whose fit is within a small margin of the best, which stops Matern winning on
  its extra degree of freedom alone.

* **Geometric anisotropy** is admitted only on out-of-fold merit.
  `.dbt_fit_anisotropy()` scores candidate angles and ratios by the k-fold
  prediction error of a kernel smoother on the transformed coordinates, and
  `.dbt_aniso_transform()` applies the accepted geometry. Scoring by fit alone
  was tried first and rejected: it reported anisotropy on isotropic fields,
  because the transform changes the lag binning and the pair counts.

## Reproducibility

* Global RNG state is preserved with `withr::local_preserve_seed()` instead of
  a manual save and restore of `.Random.seed`, so the package no longer writes
  to the global environment. `withr` is a new import.

# GeoVersa 0.3.0

* First CRAN release. Implements DB-TARF (Design-Based Targeted Adaptive
  Residual Forest) through `dbtarf_train_predict()`.
