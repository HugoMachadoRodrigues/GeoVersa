## Resubmission

This is a resubmission. In response to the review by Konstanze Lauseker,
the following changes were made:

* Removed the redundant "Tools for" at the start of the Description field.
* Explained the only acronym in the Description: DB-TARF is now spelled out as
  "Design-Based Targeted Adaptive Residual Forest" on first use.
* Replaced the two cat() console messages in R/dbtarf.R with message(), so the
  progress output can be suppressed (e.g. via suppressMessages()).
* Removed all writes to the .GlobalEnv. The internal save/restore of
  .Random.seed is now delegated to withr::local_preserve_seed() (withr added to
  Imports), so the user's random-number state is preserved without the package
  modifying the global environment.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local macOS, R 4.6.0 -- R CMD check --as-cran

## Notes

DB-TARF (Design-Based Targeted Adaptive Residual Forest) augments a random
forest with a cross-fitted, out-of-fold-selected residual correction and
design-based conformal prediction intervals. Optional
back-ends (Cubist, gstat, sp, nnet) are in Suggests and are used conditionally
via requireNamespace().
