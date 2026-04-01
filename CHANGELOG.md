# Changelog

All notable changes to this project should be documented in this file.

The project is currently in research-stage development and follows a `0.x` versioning line.

## [0.2.0-dev] - 2026-04-01

### Added

- comprehensive professional README with full architecture documentation for ConvKrigingNet2D
- Mermaid flowcharts covering full model pipeline, PatchEncoder2D stem, and AnisotropicResidualKrigingLayer
- LaTeX mathematical formulation for all model components (encoder fusion, anisotropic distance, attention-weighted residual interpolation, distance-aware gate)
- benchmark results table comparing ConvKrigingNet2D vs RF (Wadoux 2021) across all three validation protocols
- documented distance-aware beta gate (`dist_scale` fix): `effective_β = sigmoid(logit_β) × exp(−min_d_aniso / dist_scale)` — suppresses kriging correction under spatial extrapolation, critical for SpatialKFold robustness
- training protocol diagram with warmup phase, memory bank refresh, LR decay, and early stopping documentation
- algorithm variants table listing all experimental GeoKriging architectures
- shields.io badges, bibtex citation block, and full references section
- updated repo metadata on GitHub: description and 10 topic tags (deep-learning, digital-soil-mapping, kriging, spatial-statistics, pytorch, geostatistics, r-package, pedometrics, remote-sensing, convolutional-neural-network)

### Architecture highlights documented

- **ConvKrigingNet2D**: three-encoder fusion (tabular MLP 256-dim + PatchEncoder2D CNN 256-dim + coordinate MLP 256-dim) → linear fusion → 256-dim embedding z
- **PatchEncoder2D**: 28ch → ConvBlock2D(32) → MaxPool → ConvBlock2D(64) → MaxPool → ConvBlock2D(96) → AdaptiveAvgPool → Linear(128) → Linear(256); each ConvBlock2D uses double Conv3×3 + BN + GELU + residual skip
- **AnisotropicResidualKrigingLayer**: learns ℓ_major, ℓ_minor, θ from gradients; combines anisotropic spatial distance with attention similarity (256→64 projection) via softmax weighting over K=12 neighbours
- **Memory bank**: training embeddings Z_mem and residuals R_mem stored and refreshed every epoch; Z_mem is detached — no gradient flows through the neighbour side
- **dist_scale gate fix**: `dist_scale=1.0` fixed (not learnable) prevents the gate from being optimised away during training, preserving SpatialKFold stability

### Benchmark result

- Design-Based RMSE ≈ 32.8 Mg ha⁻¹ vs RF published 33.43 — ConvKrigingNet2D competitive on the gold-standard unbiased estimator

## [0.1.0-dev] - 2026-04-01

### Added

- initialized repository-level documentation and versioning structure
- documented project scope, tracked assets, and package migration plan
- added a Git-safe ignore policy for local data, generated artifacts, and external repositories

### Included in initial codebase

- `ConvKrigingNet2D` and related benchmark scripts
- Wadoux-style validation runners
- benchmark, plotting, and model-comparison scripts retained from active research workflow
