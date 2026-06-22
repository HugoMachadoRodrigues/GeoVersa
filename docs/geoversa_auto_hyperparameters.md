# GeoVersa Auto Hyperparameters

This document consolidates the automatic hyperparameter rules used by `Auto v5`.
Classes follow the code comments in [ConvKrigingNet2D_Auto_v5.R](/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/Deep%20Kriging/code/ConvKrigingNet2D_Auto_v5.R):

- `A`: variogram-derived or spatial-theory derived
- `B`: sample-size/capacity derived
- `C`: training-curve or optimization derived
- `D`: data-adaptive optimizer or SSL derived
- `Design choice`: fixed guardrail retained intentionally; should be ablated if challenged

| Hyperparameter | Type | Formula / Rule | Inputs | Justification | Reference |
|---|---|---|---|---|---|
| `K_neighbors` | A | `clamp(round(k_mult * (1 + slope*(r-r_ref)) * n*pi*range_major^2/area), 6, 30)` | `n_train`, `vg$range_major`, coordinate extent area, `nugget_ratio` | Neighbor support should scale with point density inside the practical range instead of a blind constant. | Cressie 1993 §2.4 |
| `ell_major_init` | A | `vg$range_major / mean(coord_scaler$scale)` | fitted variogram, coord scaler | Initializes kriging length-scale from the fitted practical range. | Cressie 1993 |
| `ell_minor_init` | A | `vg$range_minor / mean(coord_scaler$scale)` | fitted variogram, coord scaler | Same as above for anisotropic minor axis. | Cressie 1993 |
| `theta_init` | A | `vg$theta_rad` | fitted variogram | Uses empirical anisotropy orientation directly. | Cressie 1993 |
| `beta_init` | A | `2 - 6*r` in logit space | `nugget_ratio` | Low nugget implies kriging should start favored; high nugget implies weaker kriging prior. | Geostatistical prior + logistic gating |
| `coord_dim` | A | `clamp(32 + 24*(1-aniso_ratio) + 8*(1-r), 32, 64)` | `range_minor/range_major`, `nugget_ratio` | Strong anisotropy and strong spatial signal need more coordinate capacity. | Variogram anisotropy prior |
| `linear_blend_center` | A | `clamp(0.10 + 0.25*r, 0.08, 0.40)` | `nugget_ratio` | Local linear correction should be more conservative as nugget increases. | Design choice tied to signal-to-noise |
| `linear_blend_temp` | A | `clamp(4 + 4*(1-r) + 2*(1-aniso_ratio), 3, 12)` | `nugget_ratio`, anisotropy ratio | Stronger structure sharpens the transition between local/global correction. | Design choice tied to signal-to-noise |
| `gate_hidden` | A/B | `clamp(round(coord_dim/4 + 4*(1-r)), 6, 24)` | `coord_dim`, `nugget_ratio` | Gate width increases with coordinate representation size and spatial signal. | Capacity matched to gating task |
| `gate_floor` | A | `clamp(0.15 + 0.35*r, 0.10, 0.70)` | `nugget_ratio` | In noisy fields the gate should not fully collapse. | Design choice, ablate if challenged |
| `nko_gate_floor` | A | `clamp(0.02 + 0.20*r, 0.02, 0.30)` | `nugget_ratio` | Neural kriging residual branch can close more aggressively when structure is strong. | Design choice, ablate if challenged |
| `base_loss_weight` | A | from `.auto_loss_weights_from_nugget(r)` | `nugget_ratio` | Backbone supervision should not vanish when kriging is strong. | Multi-task regularization heuristic |
| `alpha_me` | A | from `.auto_loss_weights_from_nugget(r)` | `nugget_ratio` | Mean-error penalty increases as nugget rises to combat co-training collapse. | Empirical loss shaping |
| `lambda_cov` | A | from `.auto_loss_weights_from_nugget(r)` | `nugget_ratio` | Covariance regularization is stronger when spatial signal exists. | Empirical loss shaping |
| `lambda_spatial` | A | from `.auto_loss_weights_from_nugget(r)` | `nugget_ratio` | Spatial regularization follows residual spatial structure. | Empirical loss shaping |
| `max_warmup_epochs` | A | from `.auto_loss_weights_from_nugget(r)` | `nugget_ratio` | More structured fields warrant slightly longer backbone warmup. | Empirical warmup shaping |
| `d` | B | `64 * ceil(sqrt(n_train)/8)` clamped in code | `n_train` | Embedding width grows sublinearly with sample size. | Bartlett and Mendelson 2002 style capacity scaling |
| `patch_size` | B/A | from `auto_patch_geometry_v5(...)` | `n_train`, variogram range, raster resolution | Uses variogram support when available, with `sqrt(n)` fallback. | Cressie 1993 + sample-size fallback |
| `patch_local_size` | B/A | `clamp(odd(round(0.55*global)), 9, min(21, global))` | `patch_size` | Local branch should cover the inner support of the global patch without collapsing scale separation. | Multiscale design choice |
| `patch_dim` | B | from `.auto_patch_dim_from_cnn(patch_channels, patch_size, d)` | channels, patch size, embedding width | CNN output width should scale with input patch complexity and target embedding width. | Capacity matching |
| `fusion_hidden` | B | `d` | `d` | Fusion head width follows the main embedding width. | Capacity matching |
| `tab_dropout` | B | from `.auto_dropouts_from_n(...)` | `n_train` | More data requires less tabular regularization. | Srivastava et al. 2014 |
| `patch_dropout` | B/A | from `.auto_dropouts_from_n(...)` | `n_train`, `nugget_ratio`, `K_neighbors` | Patch branch dropout responds to sample size plus local structure strength. | Srivastava et al. 2014 + spatial adjustment |
| `coord_dropout` | B | from `.auto_dropouts_from_n(...)` | `n_train` | Coordinate encoder is low-dimensional, so dropout stays light. | Srivastava et al. 2014 |
| `nko_dropout` | B/A | from `.auto_dropouts_from_n(...)` | `n_train`, `nugget_ratio` | Residual kriging branch is regularized more when nugget is high. | Residual-noise adjustment |
| `main_epochs` | B/A | `clamp(round(40 + 2*sqrt(n_train) + 20*(1-r)), 50, 140)` | `n_train`, `nugget_ratio` | Budget grows with dataset size and with usable spatial signal. | Capacity scaling + signal-aware training |
| `batch_size` | B/D | `auto_batch_size_v5(...)` | `n_train`, feature dims, patch size, device | Batch size should match memory footprint and available support. | Throughput/footprint scaling |
| `patience` | C | `max(5, floor(main_epochs / 6))` | `main_epochs` | Early stopping should tolerate about one sixth of the planned budget without improvement. | Training-budget proportional stopping |
| `lr_patience` | C | `max(3, floor(main_epochs / 20))` | `main_epochs` | LR decay should occur before the full early-stop window is exhausted. | Training-budget proportional scheduling |
| `bank_refresh_every` | C | `auto_bank_refresh_from_patience(lr_patience)` | `lr_patience` | Memory-bank refresh cadence follows the LR schedule. | Practical scheduling heuristic |
| `vg_cutoff_frac` | A/B | `clamp((0.35 + 0.10*log10(max(n,50)/50))*mult, 0.35, 0.65)` | `n_train` | Larger datasets allow slightly wider variogram support before unstable long-range bins dominate. | Variogram estimation heuristic |
| `vg_n_lags` | A/B | `clamp(round(0.90*sqrt(n_train)*mult), 12, 24)` | `n_train` | Lag resolution should rise sublinearly with sample size. | Variogram estimation heuristic |
| `target_transform` | A | `log` if `skewness(y_train) > 1` and `y>0`, else `identity` | training target | Right-skewed DSM targets benefit from log-space training. | Standard DSM practice |
| `coord_rff_range_scaled` | A | `range / mean(coord_scaler$scale)` | variogram range, coord scaler | RFF bandwidth follows the empirical spatial scale of the target. | Spectral feature prior |
| `lr` | D | `estimate_initial_lr_polyak(...)` | preliminary model, first-batch gradients | Uses data-dependent Polyak step instead of fixed `1e-4`. | Polyak 1987; Hazan et al. 2014 |
| `wd` | Design choice | fallback `1e-3` unless explicitly overridden | none | Remains a conservative AdamW guardrail; should stay in ablation tables until fully automated. | AdamW default |
| `ssl_mask_ratio` | D | `clamp(0.30 + 0.60*mean_abs_inter_covariate_cor, 0.30, 0.75)` | raster covariate sample | More redundant raster channels support more aggressive MAE masking. | MAE-style masking adapted to DSM covariates |
| `ssl_n_samples` | D | `clamp(round(0.03 * n_valid_cells), 10000, 500000)` | valid raster cells | SSL sample budget should scale with raster support, not a blind constant. | Dataset-size proportional pretraining |
| `ssl_epochs` | D | `clamp(round(20 + sqrt(n_samples / 1000)), 30, 100)` | `n_samples` | SSL budget grows sublinearly with the sampled patch count. | Dataset-size proportional pretraining |
| `DAD-RK weights` | A/B | `mean-normalized clipped kNN spacing weights` | coordinates, covariates, `sqrt(n)` default k | Reweights supervised losses toward underrepresented spatial/environmental support when the scientific target is design-based map accuracy. | Wadoux et al. 2021 design-based validation logic |
| `module meta-controller` | A/B/D | rule-based internal switches with manual env overrides | `n_train`, nugget, anisotropy, density imbalance, covariate redundancy, patch/tab ratio, baseline residual variance | Chooses optional branches only when the dataset has enough signal and support for that branch. | Dataset-conditioned architecture selection |

## Notes

- `GEOVERSA_AUTO_META_CONTROLLER=1` is now the default for Auto v5. Explicit env vars such as `GEOVERSA_AUTO_DAM=1` or `GEOVERSA_AUTO_TAB_QBASIS=0` still override the meta-controller.
- Parameters such as `tab_hidden = c(192L)`, `coord_hidden = max(2*coord_dim, 32)`, `lr_decay = 0.5`, and `min_lr = 1e-5` still contain design choices or safety floors. These should remain explicit in ablations and not be described as “fully derived” without qualification.
- The ablation runner for these rules lives at [run_geoversa_auto_ablation.R](/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/Deep%20Kriging/code/run_geoversa_auto_ablation.R).
- The pure module meta-controller runner lives at [Benchmark_GeoVersaMeta.R](/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/Deep%20Kriging/code/Benchmark_GeoVersaMeta.R).
- The opt-in DAD-RK runner lives at [Benchmark_DADRK.R](/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/Deep%20Kriging/code/Benchmark_DADRK.R).
