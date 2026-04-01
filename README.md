# Deep Kriging

`Deep Kriging` is a research repository for hybrid spatial prediction models that combine deep learning, local raster context, and kriging-style residual correction. The current codebase is organized as a reproducible research project first, with a clear migration path toward a future R package.

## Status

- Project phase: active research and benchmarking
- Repository version: `0.1.0-dev`
- Main current model family: `ConvKrigingNet2D`
- Future target: package-oriented refactor with stable APIs, tests, and formal documentation

## Repository goals

- keep the full model-development history in one place
- preserve benchmark scripts used in the paper and thesis workflow
- separate source code from generated artifacts
- make the project safe to version and easier to publish later

## Repository structure

- `code/`
  Research scripts, benchmark runners, model definitions, and helper functions.
- `data/`
  Local data assets used during development. Contents are intentionally not versioned by default.
- `external/`
  External dependencies or third-party repositories used by the project. These are documented but not tracked inside this repository by default.
- `figures/`
  Generated figures and visual outputs for manuscripts or reports. Not versioned by default.
- `results/`
  Benchmark results, intermediate outputs, maps, and experiment artifacts. Not versioned by default.
- `Text/`
  Manuscript binaries or local writing assets. Not versioned by default.
- `docs/`
  Project-level documentation and future package roadmap.

## What is tracked

This repository is intended to track:

- source code in `code/`
- project metadata and repository documentation
- lightweight text-based documentation needed to understand, reproduce, and maintain the project

This repository does **not** track by default:

- heavy benchmark outputs
- generated maps and figures
- local data files
- external cloned repositories
- temporary RStudio or session state

## Getting started

1. Open [SoilFlux.Rproj](C:/Users/rodrigues.h/OneDrive/Deep%20Kriging/SoilFlux.Rproj) from the repository root.
2. Ensure the required R packages are installed in the active R environment.
3. Populate local `data/` and `external/` dependencies as described in the directory README files.
4. Run scripts from `code/` using the repository root as the working directory.

## Current workflow

The project currently follows a script-driven workflow:

- benchmark construction and validation scripts live in `code/`
- model families are defined as standalone `.R` files
- results are written to `results/`
- figures are generated from scripts and written to `figures/`

This structure is preserved for continuity, but the long-term plan is to migrate stable components into package-style directories such as `R/`, `man/`, and `tests/`.

## Versioning policy

The repository uses semantic-style development versions:

- `0.x` = research-stage codebase with active API changes
- `1.0.0` = first stable public release or package release candidate

The current development line is documented in [CHANGELOG.md](C:/Users/rodrigues.h/OneDrive/Deep%20Kriging/CHANGELOG.md).

## Contributing and maintenance

Project conventions and maintenance guidance are described in [CONTRIBUTING.md](C:/Users/rodrigues.h/OneDrive/Deep%20Kriging/CONTRIBUTING.md).

## Package roadmap

The package migration plan is documented in [docs/package-roadmap.md](C:/Users/rodrigues.h/OneDrive/Deep%20Kriging/docs/package-roadmap.md).

## License

No public license has been assigned yet. Until a license is explicitly added, the repository should be treated as private/internal research code.
