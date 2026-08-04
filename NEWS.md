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
