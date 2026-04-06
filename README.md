<p align="center">
  <img src="logo/GeoVersa%20Logo.png" alt="GeoVersa logo" width="280">
</p>

<h1 align="center">GeoVersa</h1>

<p align="center"><em>Deep Learning + Geostatistics for Spatial Prediction</em></p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.15139517"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.15139517.svg"></a>
  <a href="https://cran.r-project.org/"><img alt="CRAN release" src="https://img.shields.io/badge/CRAN-not%20published-lightgrey?logo=r&logoColor=white"></a>
  <a href="https://cran.r-project.org/"><img alt="CRAN downloads" src="https://img.shields.io/badge/CRAN-downloads%20n%2Fa-lightgrey?logo=r&logoColor=white"></a>
  <a href="https://cran.r-project.org/"><img alt="R >= 4.2" src="https://img.shields.io/badge/R-%E2%89%A54.2-276DC3?logo=r&logoColor=white"></a>
  <a href="https://torch.mlverse.org/"><img alt="torch" src="https://img.shields.io/badge/torch-lantern-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

## Overview

**GeoVersa** is the current research codebase for a deep geostatistical model that learns:

- a nonlinear trend from tabular covariates, raster patches, and coordinates;
- a residual memory bank built from the current backbone predictions;
- an anisotropic residual-kriging correction trained end to end with the backbone.

The active benchmark path in this repository is:

- model: `ConvKrigingNet2D`;
- trainer: `train_convkrigingnet2d_auto_one_fold_v5()`;
- auto-config source: `code/ConvKrigingNet2D_Auto_v5.R`;
- evaluation target: Wadoux-style `DesignBased` validation;
- correlation metric: `Pearson^2` by default;
- pure GeoVersa mode: `RF distillation = 0`.

This `README` describes that active path, not older exploratory variants kept in `results/`.

## What The Code Is Doing Today

For each sample at location `s_i`, GeoVersa uses:

- `x_i`: point-level tabular covariates;
- `P_i`: a local raster patch around the point;
- `s_i = (lon_i, lat_i)`: coordinates;
- `y_i`: observed target.

The target is standardized before training:

```text
y_i^(s) = (T(y_i) - mu_y) / sigma_y
```

where `T(.)` is the optional target transform.

### Backbone

The model has three encoders and one fusion block:

```text
e_tab_i   = f_tab(x_i)
e_patch_i = W_patch f_cnn(P_i)
e_coord_i = W_coord f_coord(s_i)

z_i = f_fuse([e_tab_i, e_patch_i, e_coord_i])
yhat_base_i = h(z_i)
```

Implemented components:

- tabular encoder: MLP;
- patch encoder: 2D CNN plus linear projection;
- coordinate encoder: MLP plus linear projection;
- fusion block: `Linear -> GELU -> Dropout -> Linear`;
- scalar head: maps fused latent state to the base prediction.

### Residual Memory Bank

The current backbone is applied to the training set and a residual bank is rebuilt periodically:

```text
r_j = y_j^(s) - yhat_base_j
B = {(z_j, s_j, r_j)} for j in training set
```

The active kriging layer stores latent vectors `z_j`, but the current weight computation uses only spatial distance. There is no latent-similarity term in the active anisotropic kriging weights.

### Anisotropic Residual Kriging

For query point `i` and training neighbour `j`, the code computes rotated anisotropic coordinates:

```text
dx_ij = x_i - x_j
dy_ij = y_i - y_j

u_ij =  cos(theta) dx_ij + sin(theta) dy_ij
v_ij = -sin(theta) dx_ij + cos(theta) dy_ij

d_aniso_ij = sqrt((u_ij / ell_major)^2 + (v_ij / ell_minor)^2 + eps)
```

The residual interpolation weights are:

```text
w_ij = exp(-3 d_aniso_ij) / sum_k exp(-3 d_aniso_ik)
delta_i = sum_j w_ij r_j
```

The final standardized prediction is:

```text
beta = sigmoid(logit_beta)
yhat_i^(s) = yhat_base_i + beta delta_i
```

This is the exact active formulation in `AnisotropicExpCovKrigingLayer_Auto`: pure exponential anisotropic covariance, normalized with `softmax(-3 * distance)`.

## Training Objective

The training procedure has two stages.

### Warmup

Only the backbone is trained:

```text
L_warmup = Huber(y^(s), yhat_base)
```

### Full Training

After warmup, the full model is trained with:

```text
L_full =
    Huber(y^(s), yhat)
  + lambda_base * Huber(y^(s), yhat_base)
  + alpha_ME * (mean(yhat_base) - mean(y^(s)))^2
  + lambda_cov * (sd(yhat_base) / (sd(y^(s)) + eps) - 1)^2
```

Current benchmark constraints:

- `lambda_RF = 0`;
- the residual bank is refreshed during training from the current backbone state;
- validation loss drives LR reduction and early stopping;
- `Pearson^2` is the default `r2` metric in the benchmark outputs.

## Automatic Configuration

The active auto-config logic in `code/ConvKrigingNet2D_Auto_v5.R` derives the main parameters from the training fold, the fitted variogram, and the hardware.

### Spatial Initialization

From the fitted empirical variogram, the code initializes:

- `ell_major`, `ell_minor`, `theta`;
- nugget ratio `r`;
- neighbour count `K`;
- kriging gate prior `beta_init`.

Core rules:

```text
K = clamp(round(n_train * pi * range_major^2 / area), 6, 30)
logit_beta0 = 2 - 6r
beta_init = logit_beta0
```

### Capacity Rules

The current benchmark path uses:

```text
d = clamp(64 * ceil(sqrt(n_train) / 8), 128, 512)
patch_size = clamp(floor(sqrt(n_train)), 8, 31)
patch_dim = clamp(ceil(sqrt(C * H * W)), d / 4, d)
coord_dim = clamp(32 + 24 * (1 - rho_aniso) + 8 * (1 - r), 32, 64)
```

where `rho_aniso = ell_minor / ell_major`.

If cached patches are already available, `patch_size` is inferred from the actual cached tensor shape instead of being recomputed heuristically.

### Regularization And Loss Weights

The current v5 rules are:

```text
lambda_base = max(0.05, 0.10 * r)
alpha_ME    = 0.75 * r
lambda_cov  = 0.025 * (1 - r)
max_warmup  = clamp(round(4 + 16r), 4, 20)
```

This `lambda_base` floor at `0.05` is active because the pure GeoVersa benchmark no longer uses RF distillation, so the backbone needs a minimum direct supervised signal even in strongly spatial folds.

### Optimization Rules

The initial learning rate is estimated with a Polyak-style probe on a preliminary model:

```text
alpha_init = clamp(0.01 * L / ||grad L||^2, 1e-5, 1e-3)
```

The weight decay rule is:

```text
wd = clamp(1e-3 / sqrt((n_params / 1e6) / 5), 1e-4, 1e-2)
```

Batch size is chosen from the smaller of:

- a statistical target of about `n_train / 8`;
- a device-aware memory estimate based on tensor footprint.

After warmup, the code adapts:

- `patience`;
- `lr_patience`;
- `lr_decay`;
- `bank_refresh_every`;

using the observed warmup convergence speed and the variability of warmup loss improvements.

## Wadoux Reference Organization

The repository now separates two different things that should not be mixed:

### 1. GeoVersa Benchmark

`code/run_wadoux_style_rf_conv_comparison.R`

Use this to run GeoVersa itself, optionally alongside local baselines, on Wadoux-style validation splits.

### 2. Wadoux RF Reference Reproduction

`code/run_wadoux_rf_reference.R`

Use this to reproduce the Random Forest reference under the Wadoux validation framework.

Important details tracked in `docs/wadoux2021-reference/`:

- upstream repository: `AlexandreWadoux/SpatialValidation`;
- tracked mirror commit: `ba3ad39bfa8474a09e8ac4cd82a0161649648794`;
- `Pearson^2` is the default in this repository because it matches the paper text;
- the upstream code path can still be audited separately when needed.

The repository also tracks whether the original Wadoux `.Rdata` outputs are present. As currently documented in `docs/wadoux2021-reference/official_rdata_manifest.csv`, the mirrored upstream checkout does not yet contain:

- `res_random_500.Rdata`;
- `res_regular_500.Rdata`;
- `res_clustered_random_500.Rdata`.

## Current Confirmed Benchmark Status

The most relevant confirmed comparison today is:

- scenario: `random`;
- protocol: `DesignBased`;
- sample size: `500`;
- repetitions: `10`;
- `r2`: `Pearson^2`.

### Reference RF Reproduction

Source: `results/wadoux2021_rf_reference_random_designbased_10iter_pearson_20260405/wadoux2021_rf_random_absolute_summary.csv`

| Model | ME | RMSE | Pearson^2 | MEC |
|---|---:|---:|---:|---:|
| Wadoux RF reference | 1.340 | 32.444 | 0.882 | 0.879 |

### GeoVersa Confirmation

Source: `results/geoversa_blw_confirm10_20260406/`

| GeoVersa setting | ME | RMSE | Pearson^2 | MEC |
|---|---:|---:|---:|---:|
| Auto-config with `lambda_base = 0.05` floor | -0.203 | 33.814 | 0.866 | 0.862 |
| Same setup with `lambda_base = 0.20` | -0.257 | 33.852 | 0.865 | 0.862 |

Current takeaway:

- the `0.05` floor is slightly better than `0.20`;
- GeoVersa is now close to the Wadoux RF reference on this benchmark;
- GeoVersa still does not beat the reproduced RF reference on `DesignBased`.

Gap of the current best confirmed GeoVersa run versus RF reference:

- `delta RMSE = +1.370`;
- `delta Pearson^2 = -0.016`;
- `delta MEC = -0.017`.

## How To Run The Current Benchmark

### GeoVersa

```r
Sys.setenv(
  WADOUX_AUTO_V5_SCRIPT = normalizePath(file.path(getwd(), "code", "ConvKrigingNet2D_Auto_v5.R"), mustWork = FALSE),
  WADOUX_AUTO_SCRIPT    = normalizePath(file.path(getwd(), "code", "ConvKrigingNet2D_Auto.R"), mustWork = FALSE),
  WADOUX_MODELS         = "ConvKrigingNet2D",
  WADOUX_PROTOCOLS      = "DesignBased",
  WADOUX_N_ITER         = "10",
  WADOUX_SAMPLE_SIZE    = "500",
  WADOUX_MODEL_PROFILE  = "auto",
  WADOUX_DEVICE         = "mps",
  WADOUX_RESULTS_DIR    = "results/geoversa_run"
)

source("code/run_wadoux_style_rf_conv_comparison.R")
```

Useful ablation overrides exposed by the runner:

- `WADOUX_BASE_LOSS_WEIGHT`
- `WADOUX_K_NEIGHBORS`
- `WADOUX_WEIGHT_DECAY`
- `WADOUX_LR`
- `WADOUX_BATCH_SIZE`
- `WADOUX_PATIENCE`
- `WADOUX_LR_PATIENCE`
- `WADOUX_LR_DECAY`
- `WADOUX_ALPHA_ME`
- `WADOUX_LAMBDA_COV`
- `WADOUX_COORD_DROPOUT`

### Wadoux RF Reference

```r
Sys.setenv(
  WADOUX_RF_SCENARIO    = "random",
  WADOUX_RF_PROTOCOLS   = "Population,DesignBased",
  WADOUX_RF_N_ITER      = "10",
  WADOUX_RF_SAMPLE_SIZE = "500",
  WADOUX_R2_METHOD      = "pearson",
  WADOUX_RF_RESULTS_DIR = "results/wadoux_rf_reference"
)

source("code/run_wadoux_rf_reference.R")
```

### Import Official Wadoux `.Rdata` Outputs

If the upstream project later provides the saved `.Rdata` outputs, import and document them with:

```r
source("code/import_wadoux_official_rdata.R")
```

This writes the manifest and any imported summaries to `docs/wadoux2021-reference/`.

## Repository Layout

```text
code/
  ConvKrigingNet2D_Auto.R
  ConvKrigingNet2D_Auto_v5.R
  run_wadoux_style_rf_conv_comparison.R
  run_wadoux_rf_reference.R
  import_wadoux_official_rdata.R
  wadoux2021_rf_reproduction_helpers.R

docs/
  wadoux2021-reference/

logo/
  GeoVersa Logo.png

results/
  geoversa_blw_confirm10_20260406/
  wadoux2021_rf_reference_random_designbased_10iter_pearson_20260405/
```

## Citation

```bibtex
@software{GeoVersa,
  author = {Rodrigues, Hugo},
  title  = {{GeoVersa}: Deep Learning + Geostatistics for Spatial Prediction},
  year   = {2026},
  doi    = {10.5281/zenodo.15139517},
  url    = {https://github.com/HugoMachadoRodrigues/GeoVersa}
}
```

## Reference

```text
Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S., and Brus, D. J. (2021).
Spatial cross-validation is not the right way to evaluate map accuracy.
Ecological Modelling, 457, 109692.
https://doi.org/10.1016/j.ecolmodel.2021.109692
```
