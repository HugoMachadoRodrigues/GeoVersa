<div align="center">

# 🌍 GeoVersa

### *Deep Learning + Geostatistics for Spatial Prediction — Fully Automatic*

[![R ≥ 4.2](https://img.shields.io/badge/R-%E2%89%A54.2-276DC3?logo=r&logoColor=white)](https://cran.r-project.org/)
[![torch](https://img.shields.io/badge/torch-lantern-EE4C2C?logo=pytorch&logoColor=white)](https://torch.mlverse.org/)
[![Status: Research](https://img.shields.io/badge/status-research%20preview-orange)](https://github.com/HugoMachadoRodrigues/GeoVersa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pedometrics](https://img.shields.io/badge/domain-Pedometrics%20%7C%20DSM-2e8b57)](https://www.pedometrics.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8070--8126-A6CE39?style=flat&logo=orcid&logoColor=white)](https://orcid.org/0000-0002-8070-8126)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Hugo%20Rodrigues-4285F4?style=flat&logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=vu-Ka7wAAAAJ&sortby=pubdate)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Hugo%20Rodrigues-00CCBB?style=flat&logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Hugo-Rodrigues-12)
[![X / Twitter](https://img.shields.io/badge/X-@Hugo__MRodrigues-000000?style=flat&logo=x&logoColor=white)](https://twitter.com/Hugo_MRodrigues)

</div>

---

## What is GeoVersa?

**GeoVersa** is the current research code for a pedometric model that combines deep learning and geostatistics in a single trainable architecture for spatial prediction in Digital Soil Mapping (DSM).

The current implementation is centered on **ConvKrigingNet2D**, a point-wise predictor that combines:

- A **2D CNN** that reads local raster patches (terrain, remote sensing) around each sample point
- A **tabular MLP** that processes point-level covariates (the SCORPAN factors)
- A **coordinate MLP** that encodes geographic position
- A **differentiable anisotropic residual-kriging layer** that interpolates a learned residual field from nearby training points

In the current code path, the trend model and the residual spatial correction are trained jointly. The benchmarked `v5` configuration is the **pure GeoVersa** variant: `RF distillation` is disabled.

---

## V5: Zero User Tuning

**GeoVersa V5** introduces complete automatic configuration. The user provides only the data and the compute device. All 23 model hyperparameters are derived from the data itself:

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

In the current V5 code path, the training configuration is derived automatically from the data and the available hardware.

---

## Mathematical Formulation

For a sample at location \(\mathbf{s}_i \in \mathbb{R}^2\), let:

- \(\mathbf{x}_i \in \mathbb{R}^p\): tabular covariates
- \(\mathbf{P}_i \in \mathbb{R}^{C \times H \times W}\): raster patch centered at \(\mathbf{s}_i\)
- \(y_i\): target value
- \(T(y_i)\): optional target transform used by the trainer; in the current Wadoux benchmark the default is `identity`
- \(y_i^{(s)} = \dfrac{T(y_i) - \mu_y}{\sigma_y}\): standardized target used in training

### Encoders and Fusion

The model computes three embeddings:

\[
\mathbf{e}^{\text{tab}}_i = f_{\text{tab}}(\mathbf{x}_i) \in \mathbb{R}^d
\]

\[
\mathbf{e}^{\text{patch}}_i = W_{\text{patch}} f_{\text{cnn}}(\mathbf{P}_i) \in \mathbb{R}^d
\]

\[
\mathbf{e}^{\text{coord}}_i = W_{\text{coord}} f_{\text{coord}}(\mathbf{s}_i) \in \mathbb{R}^d
\]

These are fused into a latent representation:

\[
\mathbf{z}_i = f_{\text{fuse}}\left(
\left[
\mathbf{e}^{\text{tab}}_i,\;
\mathbf{e}^{\text{patch}}_i,\;
\mathbf{e}^{\text{coord}}_i
\right]
\right)
\]

The base predictor is:

\[
\hat{y}^{\text{base}}_i = h(\mathbf{z}_i)
\]

The latent width \(d\), patch embedding dimension, coordinate embedding dimension and dropouts are all derived automatically in `v5`.

### Residual Memory Bank

At each bank refresh, the training set is passed through the current backbone to build:

\[
\mathcal{B} = \left\{(\mathbf{z}_j,\; \mathbf{s}_j,\; r_j)\right\}_{j=1}^{n_{\text{train}}}
\]

with residuals

\[
r_j = y_j^{(s)} - \hat{y}^{\text{base}}_j
\]

So the kriging branch does **not** interpolate raw targets. It interpolates residuals of the current backbone in standardized target space.

### Anisotropic Residual Kriging

For a query \(i\) and a neighbour \(j\), define the coordinate offset:

\[
\Delta x_{ij} = x_i - x_j,\qquad
\Delta y_{ij} = y_i - y_j
\]

The model learns a rotation \(\theta\) and two positive spatial scales \(\ell_{\text{maj}}, \ell_{\text{min}}\). The rotated coordinates are:

\[
u_{ij} = \cos\theta \, \Delta x_{ij} + \sin\theta \, \Delta y_{ij}
\]

\[
v_{ij} = -\sin\theta \, \Delta x_{ij} + \cos\theta \, \Delta y_{ij}
\]

The anisotropic distance is:

\[
d^{\text{aniso}}_{ij} =
\sqrt{
\left(\frac{u_{ij}}{\ell_{\text{maj}}}\right)^2 +
\left(\frac{v_{ij}}{\ell_{\text{min}}}\right)^2 + \varepsilon
}
\]

The same latent representation is also projected into a smaller similarity space:

\[
\mathbf{q}_i = W_q \mathbf{z}_i,\qquad
\mathbf{q}_j = W_q \mathbf{z}_j
\]

and the feature-similarity term is:

\[
s_{ij} = \frac{\langle \mathbf{q}_i,\mathbf{q}_j\rangle}{\sqrt{d_q}}
\]

The attention score over neighbours combines geometry and latent similarity:

\[
a_{ij} = - d^{\text{aniso}}_{ij} + s_{ij}
\]

\[
w_{ij} = \frac{\exp(a_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(a_{ik})}
\]

The residual correction is:

\[
\delta_i = \sum_{j \in \mathcal{N}(i)} w_{ij} r_j
\]

### Final Predictor

The current `v5` benchmark uses a **global learned kriging gate**:

\[
\beta = \sigma(\text{logit}_\beta)
\]

and the final prediction in standardized space is:

\[
\hat{y}^{(s)}_i = \hat{y}^{\text{base}}_i + \beta \, \delta_i
\]

The prediction returned to the user is then mapped back to the original target scale by undoing the standardization and any optional target transform.

> **Implementation note**: the core model code supports an optional eval-time distance gate and kriging dropout. In the current pure `v5` benchmark path, those mechanisms are not the main formulation being used.

### Training Objective in V5

The warmup phase trains only the backbone:

\[
\mathcal{L}_{\text{warmup}} =
\operatorname{Huber}\!\left(y^{(s)}, \hat{y}^{\text{base}}\right)
\]

After warmup, the full model is trained with:

\[
\mathcal{L} =
\operatorname{Huber}\!\left(y^{(s)}, \hat{y}^{(s)}\right)
+ \lambda_{\text{base}} \operatorname{Huber}\!\left(y^{(s)}, \hat{y}^{\text{base}}\right)
+ \alpha_{\text{ME}} \left(\overline{\hat{y}^{\text{base}}} - \overline{y^{(s)}}\right)^2
+ \lambda_{\text{cov}} \left(\frac{\operatorname{sd}(\hat{y}^{\text{base}})}{\operatorname{sd}(y^{(s)}) + \varepsilon} - 1\right)^2
\]

where:

- \(\lambda_{\text{base}}\) is `base_loss_weight`
- \(\alpha_{\text{ME}}\) is the base-prediction mean-error penalty
- \(\lambda_{\text{cov}}\) matches the dispersion of the base predictor to the target dispersion

For the current pure `v5` benchmark:

- `RF distillation` is removed, so \(\lambda_{\text{RF}} = 0\)
- the residual bank is refreshed periodically during training from the current backbone state
- early stopping and LR reduction are driven by validation loss

### What V5 Auto-Configures

The current `v5` implementation derives the main parameters as follows:

\[
K = \operatorname{clamp}\!\left(\operatorname{round}\left(
\frac{n_{\text{train}} \pi \, \text{range}_{\text{maj}}^2}{\text{area}}
\right), 6, 30\right)
\]

\[
\text{logit}_{\beta,0} = 2 - 6r
\]

\[
d = \operatorname{clamp}\!\left(64 \cdot \left\lceil \frac{\sqrt{n_{\text{train}}}}{8} \right\rceil, 128, 512\right)
\]

\[
\text{patch\_size} = \operatorname{clamp}\!\left(\lfloor \sqrt{n_{\text{train}}} \rfloor, 8, 31\right)
\]

\[
\text{patch\_dim} =
\operatorname{clamp}\!\left(
\left\lceil \sqrt{C \cdot H \cdot W} \right\rceil,\;
\frac{d}{4},\;
d
\right)
\]

\[
\text{coord\_dim} =
\operatorname{clamp}\!\left(
32 + 24(1-\rho_{\text{aniso}}) + 8(1-r),\;
32,\;
64
\right)
\]

with:

- \(r\): nugget-to-sill ratio from the fitted variogram
- \(\rho_{\text{aniso}} = \dfrac{\text{range}_{\text{min}}}{\text{range}_{\text{maj}}}\)

The loss weights are also variogram-derived:

\[
\lambda_{\text{base}} = 0.10\,r,\qquad
\alpha_{\text{ME}} = 0.75\,r,\qquad
\lambda_{\text{cov}} = 0.025(1-r)
\]

The initial learning rate is estimated from a Polyak-style probe on the actual model:

\[
\alpha_{\text{Polyak}} = \frac{\mathcal{L}}{\|\nabla \mathcal{L}\|^2},
\qquad
\alpha_{\text{init}} = \operatorname{clamp}\!\left(0.01 \,\alpha_{\text{Polyak}}, 10^{-5}, 10^{-3}\right)
\]

Batch size is the minimum of:

- a **statistical** target of roughly \(n/8\) samples per batch
- a **hardware** limit estimated from available memory and per-sample tensor cost

Weight decay scales with parameter count:

\[
\text{wd} =
\operatorname{clamp}\!\left(
\frac{10^{-3}}{\sqrt{(n_{\text{params}}/10^6)/5}},
10^{-4},
10^{-2}
\right)
\]

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
| ε(s) (spatially structured residual) | Differentiable anisotropic kriging layer |

Classical regression-kriging fits these components in separate stages. **DeepSCORPAN trains all of them jointly**, so the trend model is aware of spatial autocorrelation and the kriging layer is aware of covariate structure.

| Aspect | Regression-Kriging | Random Forest | **GeoVersa V5** |
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

- model: **GeoVersa V5 / ConvKrigingNet2D**, pure version (`RF distillation` removed)
- baseline: **local RF**, trained on the same sampled calibration sets
- scenario: `random`
- calibration sample size: `n = 500`
- iterations completed: `3`
- verified protocol so far: **DesignBased**

| Model | ME | RMSE | Spearman² | MEC |
|---|:---:|:---:|:---:|:---:|
| RF (local benchmark) | 0.42 | 31.74 | 0.803 | 0.883 |
| GeoVersa V5 (pure) | -0.02 | 35.08 | 0.763 | 0.853 |

At this point, **GeoVersa V5 pure does not outperform the local RF baseline** on the clean Design-Based benchmark.

Source: `results/codex_v5_pure_designbased_3iter/wadoux_style_rf_conv_summary_by_protocol.csv`

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

### Reproduce the benchmark (entry point)

```r
# Open GeoVersa.Rproj in RStudio, then:
source("code/Benchmark_Auto_v5.R")
```

Runs 50 independent iterations of the Wadoux (2021) benchmark with full V5 auto-configuration.
Results saved to `results/wadoux2021_auto_v5_50iter/`.

### Custom run

```r
Sys.setenv(
  WADOUX_AUTO_V5_SCRIPT = normalizePath("code/ConvKrigingNet2D_Auto_v5.R"),
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
│   ├── ConvKrigingNet2D_Auto.R               # Training loop (V5 trainer + V4 auto-config)
│   ├── ConvKrigingNet2D_Auto_v5.R            # V5 auto-configuration functions
│   ├── Benchmark_Auto_v5.R                   # ⭐ Entry point — run this
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
@software{rodrigues2026geoVersa,
  author = {Rodrigues, Hugo},
  title  = {{GeoVersa}: Deep Learning + Geostatistics for Spatial Prediction},
  year   = {2026},
  url    = {https://github.com/HugoMachadoRodrigues/GeoVersa},
  note   = {Research preview — ConvKrigingNet2D V5 with complete automatic configuration}
}
```

---

<div align="center">

**Built for the Pedometrics community**

*GeoVersa — where deep learning meets geostatistics*

</div>
