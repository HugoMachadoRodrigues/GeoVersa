<p align="center">
  <img src="logo/geoversa-logo.png" alt="GeoVersa logo" width="280">
</p>

<h1 align="center">GeoVersa</h1>

<p align="center"><em>Deep Learning + Geostatistics for Spatial Prediction</em></p>

<p align="center">
  <a href="https://cran.r-project.org/"><img alt="R ≥ 4.2" src="https://img.shields.io/badge/R-%E2%89%A54.2-276DC3?logo=r&logoColor=white"></a>
  <a href="https://torch.mlverse.org/"><img alt="torch" src="https://img.shields.io/badge/torch-lantern-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/HugoMachadoRodrigues/GeoVersa"><img alt="Status: Research" src="https://img.shields.io/badge/status-research%20preview-orange"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.pedometrics.org/"><img alt="Pedometrics" src="https://img.shields.io/badge/domain-Pedometrics%20%7C%20DSM-2e8b57"></a>
</p>

<p align="center">
  <a href="https://orcid.org/0000-0002-8070-8126"><img alt="ORCID" src="https://img.shields.io/badge/ORCID-0000--0002--8070--8126-A6CE39?style=flat&logo=orcid&logoColor=white"></a>
  <a href="https://scholar.google.com/citations?user=vu-Ka7wAAAAJ&sortby=pubdate"><img alt="Google Scholar" src="https://img.shields.io/badge/Google%20Scholar-Hugo%20Rodrigues-4285F4?style=flat&logo=googlescholar&logoColor=white"></a>
  <a href="https://www.researchgate.net/profile/Hugo-Rodrigues-12"><img alt="ResearchGate" src="https://img.shields.io/badge/ResearchGate-Hugo%20Rodrigues-00CCBB?style=flat&logo=researchgate&logoColor=white"></a>
  <a href="https://twitter.com/Hugo_MRodrigues"><img alt="X / Twitter" src="https://img.shields.io/badge/X-@Hugo__MRodrigues-000000?style=flat&logo=x&logoColor=white"></a>
</p>

---

## What is GeoVersa?

**GeoVersa** is the current research code for a pedometric model that combines deep learning and geostatistics in a single trainable architecture for spatial prediction in Digital Soil Mapping (DSM).

The current implementation is centered on **ConvKrigingNet2D**, a point-wise predictor that combines:

- A **2D CNN** that reads local raster patches (terrain, remote sensing) around each sample point
- A **tabular MLP** that processes point-level covariates (the SCORPAN factors)
- A **coordinate MLP** that encodes geographic position
- A **differentiable anisotropic residual-kriging layer** that interpolates a learned residual field from nearby training points

In the current code path, the trend model and the residual spatial correction are trained jointly. The benchmarked configuration is the **pure GeoVersa** variant: `RF distillation` is disabled.

---

## Zero User Tuning

**GeoVersa** uses complete automatic configuration. The user provides only the data and the compute device. The main model hyperparameters are derived from the data itself:

| Hyperparameter | Derived from |
|---|---|
| Kriging range (ℓ_maj, ℓ_min), anisotropy angle (θ) | Empirical variogram fitted to training targets |
| Neighbour count K | Training-point density and variogram practical range |
| Network width d, patch embedding dim | √n scaling plus patch geometry |
| Dropout rates | Sample size (larger n → less regularisation needed) |
| Learning rate | Gradient norm statistics (short probe pass) |
| Batch size | Available GPU/MPS/CPU memory |
| Early stopping patience | Loss trajectory during warmup phase |
| Coordinate embedding dimension | Variogram anisotropy and nugget ratio |
| Weight decay | Model parameter count |

Auto-configuration is split across the training pipeline:

1. **Variogram phase**: fit the spatial structure used to initialise anisotropy, neighbour count and kriging weight
2. **Capacity phase**: derive architecture width and dropout from sample size and patch geometry
3. **Optimisation phase**: estimate learning rate, batch size, weight decay, patience and bank refresh from gradients, hardware and warmup dynamics

In the current code path, the training configuration is derived automatically from the data and the available hardware.

---

## Mathematical Formulation

For a sample at location `s_i ∈ R^2`, let:

- `x_i ∈ R^p`: tabular covariates
- `P_i ∈ R^(C x H x W)`: raster patch centered at `s_i`
- `y_i`: target value
- `T(y_i)`: optional target transform used by the trainer; in the current Wadoux benchmark the default is `identity`
- `y_i^(s) = (T(y_i) - μ_y) / σ_y`: standardized target used in training

### Encoders and Fusion

The model computes three embeddings:

```text
e_i^tab   = f_tab(x_i) ∈ R^d
e_i^patch = W_patch f_cnn(P_i) ∈ R^d
e_i^coord = W_coord f_coord(s_i) ∈ R^d
```

These are fused into a latent representation:

```text
z_i = f_fuse([e_i^tab, e_i^patch, e_i^coord])
yhat_i^base = h(z_i)
```

The latent width `d`, patch embedding dimension, coordinate embedding dimension and dropouts are all derived automatically in the current implementation.

### Residual Memory Bank

At each bank refresh, the training set is passed through the current backbone to build:

```text
B = {(z_j, s_j, r_j)} for j = 1, ..., n_train
r_j = y_j^(s) - yhat_j^base
```

So the kriging branch does **not** interpolate raw targets. It interpolates residuals of the current backbone in standardized target space.

### Anisotropic Residual Kriging

For a query `i` and a neighbour `j`, define the coordinate offset:

```text
Δx_ij = x_i - x_j
Δy_ij = y_i - y_j

u_ij = cos(θ) Δx_ij + sin(θ) Δy_ij
v_ij = -sin(θ) Δx_ij + cos(θ) Δy_ij

d_ij^aniso = sqrt((u_ij / l_maj)^2 + (v_ij / l_min)^2 + ε)

q_i = W_q z_i
q_j = W_q z_j
s_ij = <q_i, q_j> / sqrt(d_q)

a_ij = -d_ij^aniso + s_ij
w_ij = exp(a_ij) / Σ_{k in N(i)} exp(a_ik)

δ_i = Σ_{j in N(i)} w_ij r_j
```

### Final Predictor

The current benchmark uses a **global learned kriging gate**:

```text
β = sigmoid(logit_β)
yhat_i^(s) = yhat_i^base + β δ_i
```

The prediction returned to the user is then mapped back to the original target scale by undoing the standardization and any optional target transform.

> **Implementation note**: the core model code supports an optional eval-time distance gate and kriging dropout. In the current pure benchmark path, those mechanisms are not the main formulation being used.

### Training Objective

The warmup phase trains only the backbone:

```text
L_warmup = Huber(y^(s), yhat^base)
```

After warmup, the full model is trained with:

```text
L = Huber(y^(s), yhat^(s))
  + λ_base Huber(y^(s), yhat^base)
  + α_ME (mean(yhat^base) - mean(y^(s)))^2
  + λ_cov (sd(yhat^base) / (sd(y^(s)) + ε) - 1)^2
```

where:

- `λ_base` is `base_loss_weight`
- `α_ME` is the base-prediction mean-error penalty
- `λ_cov` matches the dispersion of the base predictor to the target dispersion

For the current pure benchmark:

- `RF distillation` is removed, so `λ_RF = 0`
- the residual bank is refreshed periodically during training from the current backbone state
- early stopping and LR reduction are driven by validation loss

### What GeoVersa Auto-Configures

The current implementation derives the main parameters as follows:

```text
K = clamp(round(n_train π range_maj^2 / area), 6, 30)

logit_β,0 = 2 - 6r

d = clamp(64 ceil(sqrt(n_train) / 8), 128, 512)

patch_size = clamp(floor(sqrt(n_train)), 8, 31)

patch_dim = clamp(ceil(sqrt(C H W)), d / 4, d)

coord_dim = clamp(32 + 24 (1 - ρ_aniso) + 8 (1 - r), 32, 64)
```

with:

- `r`: nugget-to-sill ratio from the fitted variogram
- `ρ_aniso = range_min / range_maj`

The loss weights are also variogram-derived:

```text
λ_base = 0.10 r
α_ME = 0.75 r
λ_cov = 0.025 (1 - r)
```

The initial learning rate is estimated from a Polyak-style probe on the actual model:

```text
α_Polyak = L / ||∇L||^2
α_init = clamp(0.01 α_Polyak, 1e-5, 1e-3)
```

Batch size is the minimum of:

- a **statistical** target of roughly `n/8` samples per batch
- a **hardware** limit estimated from available memory and per-sample tensor cost

Weight decay scales with parameter count:

```text
wd = clamp(1e-3 / sqrt((n_params / 1e6) / 5), 1e-4, 1e-2)
```

Warmup validation losses then determine:

- LR patience
- early-stopping patience
- LR decay factor
- memory-bank refresh interval

---

## From SCORPAN to DeepSCORPAN

GeoVersa implements the **SCORPAN** framework (McBratney et al., 2003) as a unified neural network. Each soil-forming factor maps to a specialised encoder:

| SCORPAN Factor | Encoder |
|---|---|
| S, C, O, R, P, A (point-level covariates) | Tabular MLP |
| O, R (spatial texture as raster) | 2D CNN PatchEncoder |
| N (geographic position) | Coordinate MLP |
| ε(s) (spatially structured residual) | Differentiable anisotropic residual-kriging layer |

Classical regression-kriging fits these components in separate stages. **DeepSCORPAN trains all of them jointly**, so the trend model is aware of spatial autocorrelation and the kriging layer is aware of covariate structure.

| Aspect | Regression-Kriging | Random Forest | **GeoVersa** |
|---|:---:|:---:|:---:|
| SCORPAN covariates | ✅ linear | ✅ non-linear | ✅ deep non-linear |
| Local raster texture | ❌ | ❌ | ✅ 2D CNN patch |
| Spatial autocorrelation | ✅ fitted variogram | ❌ | ✅ learned residual interpolation |
| Anisotropy | ⚠️ manual | ❌ | ✅ learned θ, ℓ_maj, ℓ_min |
| Joint trend + residual training | ❌ two-stage | ❌ | ✅ end-to-end |
| Zero hyperparameter tuning | ❌ | ❌ | ✅ fully automatic |

---

## Current Benchmark Status

The benchmark runner follows the **Wadoux et al. (2021)** validation framework. In this repository, that means:

- using the same benchmark dataset and protocol family
- treating **Design-Based** validation as the main map-accuracy estimate
- comparing models trained locally under the same runner

The current clean run is:

- model: **GeoVersa / ConvKrigingNet2D**, pure version (`RF distillation` removed)
- baseline: **local RF**, trained on the same sampled calibration sets
- scenario: `random`
- calibration sample size: `n = 500`
- iterations completed: `3`
- verified protocol so far: **DesignBased**

| Model | ME | RMSE | Spearman² | MEC |
|---|:---:|:---:|:---:|:---:|
| RF (local benchmark) | 0.42 | 31.74 | 0.803 | 0.883 |
| GeoVersa (pure) | -0.02 | 35.08 | 0.763 | 0.853 |

At this point, **GeoVersa pure does not outperform the local RF baseline** on the clean Design-Based benchmark.

Source: clean Design-Based benchmark summary generated by this repository under `results/`.

In this benchmark runner, `R²` is the squared **Spearman** correlation, matching the Wadoux-style metric implementation in the repository.

> **Important**: this README no longer reproduces hardcoded numeric summaries attributed to the Wadoux paper. The paper is used here as the **validation-framework reference**; benchmark numbers shown in this README are only the numbers generated by this repository.

`RandomKFold` and `SpatialKFold` remain implemented in the runner, but the clean post-distillation benchmark above was validated only for `DesignBased`.

---

## How to Run

### Requirements

```r
install.packages(c("torch", "ranger", "dplyr", "terra", "sf", "FNN", "caret"))
torch::install_torch()  # one-time setup
```

### Clone

```bash
git clone https://github.com/HugoMachadoRodrigues/GeoVersa.git
cd GeoVersa
```

### Reproduce the benchmark

```r
# Open GeoVersa.Rproj in RStudio, then:
Sys.setenv(
  WADOUX_MODELS      = "RF,ConvKrigingNet2D",
  WADOUX_PROTOCOLS   = "DesignBased,RandomKFold,SpatialKFold",
  WADOUX_N_ITER      = "50",
  WADOUX_SAMPLE_SIZE = "500",
  WADOUX_DEVICE      = "mps",
  WADOUX_RESULTS_DIR = "results/wadoux2021_auto_50iter"
)
source("code/run_wadoux_style_rf_conv_comparison.R")
```

Runs 50 independent iterations of the Wadoux (2021) benchmark with full automatic configuration.
Results saved to `results/wadoux2021_auto_50iter/`.

### Custom run

```r
Sys.setenv(
  WADOUX_MODELS         = "RF,ConvKrigingNet2D",
  WADOUX_PROTOCOLS      = "DesignBased,RandomKFold,SpatialKFold",
  WADOUX_N_ITER         = "50",        # independent iterations
  WADOUX_SAMPLE_SIZE    = "500",
  WADOUX_DEVICE         = "mps",       # "cuda", "mps", or "cpu"
  WADOUX_RESULTS_DIR    = "results/my_run"
)
source("code/run_wadoux_style_rf_conv_comparison.R")
```

---

## Project Structure

```
GeoVersa/
├── code/
│   ├── ConvKrigingNet2D.R                    # Core model architecture (sources utilities below)
│   ├── ConvKrigingNet2D_Auto.R               # Training loop and automatic configuration hooks
│   ├── auto-configuration scripts            # Data-driven parameter derivation utilities
│   ├── benchmark entry scripts               # High-level benchmark launchers
│   ├── run_wadoux_style_rf_conv_comparison.R # Benchmark engine
│   ├── wadoux2021_rf_reproduction_helpers.R  # Data loading, protocols, Wadoux metrics
│   ├── KrigingNet_PointPatchCNN.R            # Patch extraction + memory bank utilities
│   ├── KrigingNet_DualFramework.R            # Context loaders (sim + Wadoux)
│   ├── KrigingNet_WadouxComparison.R         # Foundational utilities (MLP, scalers, losses)
│   ├── figures_wadoux_comparison.R           # Figures from local Wadoux-style benchmark runs
│   ├── figures_paper.R                       # Manuscript figures (requires comparable baseline runs)
│   └── generate_wadoux_maps.R                # Prediction maps
├── external/SpatialValidation/               # Wadoux et al. (2021) original data
├── data/                                     # Local data (gitignored)
├── logo/                                     # GeoVersa visual identity assets
├── results/                                  # Output metrics (gitignored)
├── figures/                                  # Generated figures
└── README.md
```

> **Note on dependency chain**: `ConvKrigingNet2D.R` sources `KrigingNet_PointPatchCNN.R`, which sources `KrigingNet_DualFramework.R`, which sources `KrigingNet_WadouxComparison.R`. All three utility files are required for the model to load. The `load_convkrigingnet2d_env()` runner function handles this chain automatically.

---

## References

- **Wadoux, A.M.J.-C., Heuvelink, G.B.M., de Bruin, S., Brus, D.J.** (2021). Spatial cross-validation is not the right way to evaluate map accuracy. *Ecological Modelling*, 457, 109692. [doi:10.1016/j.ecolmodel.2021.109692](https://doi.org/10.1016/j.ecolmodel.2021.109692)
- **McBratney, A.B., Mendonça Santos, M.L., Minasny, B.** (2003). On digital soil mapping. *Geoderma*, 117(1–2), 3–52. [doi:10.1016/S0016-7061(03)00223-4](https://doi.org/10.1016/S0016-7061(03)00223-4)
- **Matheron, G.** (1963). Principles of geostatistics. *Economic Geology*, 58(8), 1246–1266.
- **He, K., Zhang, X., Ren, S., Sun, J.** (2016). Deep residual learning for image recognition. *CVPR 2016*.

---

## Citation

```bibtex
@software{GeoVersa,
  author = {Rodrigues, Hugo},
  title  = {{GeoVersa}: Deep Learning + Geostatistics for Spatial Prediction},
  year   = {2026},
  url    = {https://github.com/HugoMachadoRodrigues/GeoVersa},
  note   = {Research preview with complete automatic configuration}
}
```

---

<div align="center">

**Built for the Pedometrics community**

*GeoVersa — where deep learning meets geostatistics*

</div>
