# Contributing

This repository is currently a research codebase under active development. The main goal of these guidelines is to preserve reproducibility and avoid mixing source code with generated artifacts.

## Ground rules

- keep source code in `code/`
- do not commit large generated outputs from `results/` or `figures/`
- do not commit local data files from `data/`
- do not vendor external repositories directly into this repository unless explicitly decided

## Before committing

- run the relevant R parse checks for edited scripts
- confirm that paths are repository-root relative when possible
- keep benchmark logic and manuscript-facing scripts clearly named
- prefer text documentation over binary notes or ad hoc scratch files

## Naming conventions

- model definition scripts: `ModelName.R`
- benchmark runners: `run_*.R`
- one-off benchmark summaries should live under `results/`, not in the repository root

## Documentation expectations

When you add a new model family or benchmark runner, document:

- what question it answers
- what benchmark or validation protocol it uses
- what outputs it writes and where
- whether it is exploratory, confirmatory, or manuscript-facing

## Toward a package

As stable components emerge, they should be migrated from `code/` into package-style structure. The roadmap is described in [docs/package-roadmap.md](C:/Users/rodrigues.h/OneDrive/Deep%20Kriging/docs/package-roadmap.md).
