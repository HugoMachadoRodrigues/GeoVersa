## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local macOS, R 4.6.0 -- R CMD check --as-cran

## Notes

DB-TARF augments a random forest with a cross-fitted, out-of-fold-selected
residual correction and design-based conformal prediction intervals. Optional
back-ends (Cubist, gstat, sp, nnet) are in Suggests and are used conditionally
via requireNamespace().
