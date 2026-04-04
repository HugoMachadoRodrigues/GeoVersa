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

**GeoVersa** is a novel pedometric model that unifies deep learning and geostatistics into a single end-to-end trainable architecture for spatial prediction in Digital Soil Mapping (DSM).

Its core model, **ConvKrigingNet2D**, combines:

- A **2D CNN** that reads local raster patches (terrain, remote sensing) around each sample point
- A **tabular MLP** that processes point-level covariates (the SCORPAN factors)
- A **coordinate MLP** that encodes geographic position
- A **differentiable anisotropic kriging layer** that learns spatial autocorrelation structure — range, anisotropy axes, and rotation — directly from data gradients, jointly with the trend model

The key innovation over classical regression-kriging is that **the trend and the geostatistical residual correction are trained together in a single end-to-end pass**, eliminating the two-stage feedback gap.

---

## V5: Zero User Tuning

**GeoVersa V5** introduces complete automatic configuration. The user provides only the data and the compute device. All 23 model hyperparameters are derived from the data itself:

| Hyperparameter | Derived from |
|---|---|
| Kriging range (ℓ_maj, ℓ_min), anisotropy angle (θ) | Empirical variogram fitted to training targets |
| Neighbour count K | Nugget ratio of the variogram |
| Network width d, patch embedding dim | √n scaling from training sample size |
| Dropout rates | Sample size (larger n → less regularisation needed) |
| Learning rate | Gradient norm statistics (short probe pass) |
| Batch size | Available GPU/MPS/CPU memory |
| Early stopping patience | Loss trajectory during warmup phase |
| Coordinate embedding dimension | Coordinate anisotropy ratio |
| Weight decay | Model parameter count |

Auto-configuration runs in two phases before main training:

1. **Phase 1** — a lightweight probe estimates the learning rate from gradient statistics and the batch size from hardware memory
2. **Phase 2** — coordinate anisotropy and model capacity inform embedding dimensions and weight decay

GeoVersa V5 is fully plug-and-play: no grid search, no manual tuning, no domain-specific parameter knowledge required.

---

## Architecture

The prediction at location **s₀** is:

$$\hat{S}(\mathbf{s}_0) = \underbrace{f_\theta(\mathbf{x}_0,\, \mathbf{P}_{15\times15},\, \mathbf{s}_0)}_{\text{DeepSCORPAN trend}} + \underbrace{\beta^\text{eff}(\mathbf{s}_0) \cdot \delta(\mathbf{s}_0)}_{\text{learned anisotropic kriging}}$$

### Three parallel encoders

| Encoder | Input | Output |
|---|---|---|
| Tabular MLP | Point-level covariates | 256-dim embedding |
| 2D CNN (PatchEncoder2D) | 15×15 raster patch (terrain + RS bands) | 256-dim embedding |
| Coordinate MLP | Geographic coordinates (x, y) | 256-dim embedding |

The three embeddings are concatenated and fused into a shared representation **z**, from which a scalar base prediction **ŷ_base** is produced.

### Anisotropic residual kriging layer

The kriging correction **δ** is computed from the K nearest training neighbours using:

1. **Anisotropic distance** — offsets rotated by learned θ and scaled by learned ℓ_maj / ℓ_min
2. **Feature similarity** — dot product between projected query and neighbour embeddings
3. **Attention weights** — softmax over (−distance + similarity)
4. **Residual interpolation** — weighted sum of stored training residuals rₖ = yₖ − ŷ_base,k

### Distance-aware gate

$$\beta^\text{eff}(\mathbf{s}_0) = \sigma(\text{logit}_\beta) \cdot \exp\!\left(-\frac{d^\min}{\tau}\right)$$

When no training point is nearby the gate suppresses the kriging term and the model falls back to the DeepSCORPAN trend — exactly as classical kriging does beyond the variogram range. This makes GeoVersa robust under spatial extrapolation without any protocol-specific logic.

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
| Spatial autocorrelation | ✅ fitted variogram | ❌ | ✅ learned in-model |
| Anisotropy | ⚠️ manual | ❌ | ✅ learned θ, ℓ_maj, ℓ_min |
| Joint trend + residual training | ❌ two-stage | ❌ | ✅ end-to-end |
| Zero hyperparameter tuning | ❌ | ❌ | ✅ fully automatic |

---

## Benchmark Results

Validation following **Wadoux et al. (2021)**: dataset = above-ground biomass (AGB, Mg ha⁻¹), Amazon basin; N = 500 calibration points, simple random sampling, 3 independent iterations.

| Protocol | RF (Wadoux 2021) | **GeoVersa V5** | Δ RMSE |
|---|:---:|:---:|:---:|
| **Design-Based** ✅ | RMSE 38.81 · R² 0.710 · MEC 0.810 | **RMSE 33.81 · R² 0.770 · MEC 0.863** | **−5.00** |
| Random K-Fold | RMSE 36.54 · R² 0.830 · MEC 0.840 | RMSE 35.40 · R² 0.767 · MEC 0.850 | −1.14 |
| Spatial K-Fold | RMSE 44.25 · R² 0.740 · MEC 0.770 | RMSE 37.63 · R² 0.720 · MEC 0.833 | −6.62 |

**Design-Based is the only statistically valid protocol** (Wadoux et al. 2021): it uses probability sampling and design-based inference to produce unbiased estimates of population-level map accuracy. On this protocol GeoVersa V5 outperforms the RF baseline on all three metrics.

> **Note**: GeoVersa V5 results are preliminary (3 iterations, auto-configured). The RF numbers are from the published paper (500 iterations). A full 50-iteration run is in progress.

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

Runs 3 iterations of the Wadoux (2021) benchmark with full V5 auto-configuration.
Results saved to `results/wadoux2021_auto_v5_quick/`.

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
│   ├── ConvKrigingNet2D.R                    # Core model architecture
│   ├── ConvKrigingNet2D_Auto.R               # Training loop (V5 trainer)
│   ├── ConvKrigingNet2D_Auto_v5.R            # V5 auto-configuration functions
│   ├── Benchmark_Auto_v5.R                   # ⭐ Entry point — run this
│   ├── run_wadoux_style_rf_conv_comparison.R # Benchmark engine
│   ├── wadoux2021_rf_reproduction_helpers.R  # Data loading, protocols, metrics
│   ├── figures_wadoux_comparison.R           # Publication figures
│   ├── figures_paper.R                       # Paper figures
│   └── generate_wadoux_maps.R                # Prediction maps
├── external/SpatialValidation/               # Wadoux et al. (2021) original data
├── data/                                     # Local data (gitignored)
├── results/                                  # Output metrics (gitignored)
├── figures/                                  # Generated figures
└── README.md
```

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
