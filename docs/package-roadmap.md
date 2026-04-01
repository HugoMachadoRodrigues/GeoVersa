# Package Roadmap

This project is currently organized as a research repository. The medium-term goal is to turn the stable core into an R package without losing the current benchmark history.

## Current state

- model definitions are stored as standalone scripts in `code/`
- benchmark logic is also script-based
- generated outputs are written directly to `results/` and `figures/`
- there is no exported API yet

## Target package structure

When the codebase stabilizes, the repository should progressively move toward:

- `R/`
  Stable exported and internal functions
- `man/`
  Function documentation
- `tests/testthat/`
  Unit and regression tests
- `inst/`
  Templates, benchmark metadata, and packaged non-code assets
- `vignettes/`
  Reproducible tutorials and workflow notes

## Recommended migration path

1. identify stable utility functions used across multiple scripts
2. move those utilities into package-style functions
3. separate benchmark orchestration from reusable model code
4. add tests for data preparation, metrics, and benchmark split logic
5. add package metadata (`DESCRIPTION`, `NAMESPACE`) only after the API is coherent

## Suggested first candidates for migration

- benchmark-building helpers
- metric functions
- patch extraction and preprocessing helpers
- ConvKrigingNet2D training and prediction helpers
- Wadoux-style validation wrappers once their interfaces stabilize

## What should remain outside the future package

- large benchmark results
- manuscript-only figure scripts
- local scratch analyses
- third-party external repositories
