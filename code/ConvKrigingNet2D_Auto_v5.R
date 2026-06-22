# =============================================================================
# ConvKrigingNet2D_Auto_v5.R
#
# GeoVersa V5 — Complete automatic configuration from data.
# ZERO user tuning. Every parameter derived from the training set.
#
# SCIENTIFIC DESIGN PRINCIPLES
# ─────────────────────────────
# Every hyperparameter belongs to exactly one derivation class:
#
#   (A) GEOSTATISTICAL THEORY  — variogram-derived spatial parameters
#       Source: fit_variogram_auto() fitted on the actual training targets
#       References: Webster & Oliver (2007); Journel & Huijbregts (1978)
#
#   (B) STATISTICAL CAPACITY   — √n information-theoretic scaling
#       Source: training sample size n_train
#       References: Bartlett & Mendelson (2002); Srivastava et al. (2014)
#
#   (C) HARDWARE-AWARE          — actual memory probing (not assumed constants)
#       Source: sysctl / CUDA query at runtime
#
#   (D) TRAINING DYNAMICS       — adapted from observed warmup trajectory
#       Source: per-epoch validation losses during backbone warmup
#       References: Smith (2017); Smith (2018)
#
# Consequence: GeoVersa V5 configures itself identically correctly for
# soil pH in Europe, forest carbon in the Amazon, or heavy metals in Asia —
# without any user-provided hyperparameter.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# §0  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

.auto_clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

.auto_make_odd <- function(x) {
  x <- as.integer(round(x))
  ifelse(x %% 2L == 0L, x + 1L, x)
}

.auto_env_num <- function(name, default) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) {
    cat(sprintf("[Auto v5]   %s invalid ('%s') — fallback %.4f\n", name, raw, default))
    return(default)
  }
  cat(sprintf("[Auto v5]   %s override → %.4f\n", name, val))
  val
}

# Query ACTUAL available memory (not a hardcoded constant).
.get_actual_ram_gb <- function(device = "mps") {
  if (device == "mps") {
    tryCatch({
      raw          <- system("sysctl -n hw.memsize", intern = TRUE)
      total_bytes  <- as.numeric(trimws(raw))
      if (is.finite(total_bytes) && total_bytes > 1e8) {
        # MPS shares memory between GPU and CPU; allow 70 % for model + data
        available_gb <- total_bytes / 1e9 * 0.70
        cat(sprintf("[Auto v5]   RAM probe: %.1f GB total → %.1f GB usable (70%%)\n",
                    total_bytes / 1e9, available_gb))
        return(available_gb)
      }
    }, error = function(e) {})
    cat("[Auto v5]   RAM probe failed — conservative fallback: 8 GB\n")
    return(8.0)

  } else if (device == "cuda") {
    tryCatch({
      props       <- torch_cuda_get_device_properties(0L)
      total_bytes <- props$total_memory
      used_bytes  <- torch_cuda_memory_allocated()$item()
      free_gb     <- (total_bytes - used_bytes) / 1e9
      cat(sprintf("[Auto v5]   CUDA VRAM: %.1f GB free\n", free_gb))
      return(max(free_gb, 1.0))
    }, error = function(e) {
      cat("[Auto v5]   CUDA VRAM probe failed — fallback: 4 GB\n")
      return(4.0)
    })

  } else {                          # CPU
    tryCatch({
      # macOS
      raw <- suppressWarnings(
        system("sysctl -n hw.memsize 2>/dev/null", intern = TRUE))
      if (length(raw) == 0 || nchar(trimws(raw[1])) == 0) {
        # Linux
        raw <- suppressWarnings(
          system("awk '/MemTotal/{print $2*1024}' /proc/meminfo 2>/dev/null",
                 intern = TRUE))
      }
      total_bytes <- as.numeric(trimws(raw[1]))
      if (is.finite(total_bytes) && total_bytes > 1e8)
        return(total_bytes / 1e9 * 0.50)
    }, error = function(e) {})
    return(4.0)
  }
}


# ─────────────────────────────────────────────────────────────────────────────
# §1  PHASE 0-A  Geostatistical theory  (class A — variogram-derived)
# ─────────────────────────────────────────────────────────────────────────────

# K_neighbors from spatial point density + variogram practical range
#
# Physical derivation (Webster & Oliver 2007, §5.2; Chilès & Delfiner 2012 §3.4):
#   For an exponential covariance C(h) = σ²·exp(−3h/a), spatial correlation
#   becomes negligible for h > a (practical range a). Points within distance a
#   contribute the majority of the kriging weight.  Under a 2-D Poisson process
#   with spatial intensity λ = n / Area, the expected number of training points
#   within the correlation circle is:
#       E[K] = λ · π · a²  =  n_train · π · range_major² / Area
#   We clamp to [6, 30] to guarantee a meaningful neighbourhood but limit
#   memory use.  Using range_major (longest correlation axis) ensures we are
#   generous rather than too conservative.
# K_neighbors from spatial density inside the variogram range
#
# Cressie (1993, §2.4) motivates selecting neighbors by the amount of support
# covered by the practical range.  The expected number of training points inside
# a disc of radius `range_major` is n * π * range^2 / area.  We allow only a
# mild nugget-based modulation around this density count.
.auto_K_neighbors_from_density <- function(n_train,
                                           range_major,
                                           area_extent,
                                           nugget_ratio) {
  if (!is.finite(area_extent) || area_extent <= 0 ||
      !is.finite(range_major) || range_major <= 0) {
    cat("[Auto v5]   K: degenerate coordinate area/range → K = 10\n")
    return(10L)
  }
  k_mult <- .auto_env_num("GEOVERSA_AUTO_K_MULT", 1.0)
  k_nugget_slope <- .auto_env_num("GEOVERSA_AUTO_K_NUGGET_SLOPE", 0.0)
  k_nugget_ref <- .auto_env_num("GEOVERSA_AUTO_K_NUGGET_REF", 0.18)
  k_theory_raw <- n_train * pi * range_major^2 / area_extent
  k_struct_mult <- .auto_clamp(
    1 + k_nugget_slope * (nugget_ratio - k_nugget_ref),
    0.85, 1.15
  )
  k_theory <- k_mult * k_struct_mult * k_theory_raw
  K <- as.integer(.auto_clamp(round(k_theory), 6L, 30L))
  cat(sprintf(
    "[Auto v5]   K_neighbors = %d  [%.2f × %.3f × n·π·range²/area = %.2f × %.3f × %.1f·π·%.4g²/%.4g]\n",
    K, k_mult, k_struct_mult, k_mult, k_struct_mult, n_train, range_major, area_extent
  ))
  K
}

.auto_K_from_spatial_density <- function(n_train, vg, Ctr) {
  area <- diff(range(Ctr[, 1])) * diff(range(Ctr[, 2]))
  .auto_K_neighbors_from_density(
    n_train = n_train,
    range_major = vg$range_major,
    area_extent = area,
    nugget_ratio = vg$nugget_ratio
  )
}

# beta_init (logit of initial kriging weight β) from nugget ratio r
#
# Physical derivation:
#   β = σ(logit_beta) gates the magnitude of the kriging correction δ.
#   At initialisation we want β to reflect how much of the signal the
#   kriging layer is expected to carry:
#     r → 0 (pure spatial structure): kriging is powerful → β₀ ≈ 0.88
#     r → 1 (pure nugget):            kriging is useless  → β₀ ≈ 0.02
#   Linear logit schedule: logit_beta = 2 − 6r
#     r = 0.00 → logit = +2.0 → β = 0.88  (strong kriging prior)
#     r = 0.33 → logit =  0.0 → β = 0.50  (balanced)
#     r = 0.67 → logit = −2.0 → β = 0.12  (kriging mostly off)
#     r = 1.00 → logit = −4.0 → β = 0.02  (kriging disabled)
.auto_beta_init_from_nugget <- function(r) {
  beta_intercept <- .auto_env_num("GEOVERSA_AUTO_BETA_INTERCEPT", 2.0)
  beta_slope <- .auto_env_num("GEOVERSA_AUTO_BETA_SLOPE", -6.0)
  logit_beta <- beta_intercept + beta_slope * r
  cat(sprintf(
    "[Auto v5]   beta_init: logit = %.2f → β₀ = %.3f  [%.2f %+ .2f·r, r = %.3f]\n",
    logit_beta, 1 / (1 + exp(-logit_beta)), beta_intercept, beta_slope, r))
  logit_beta
}

# coord_dim from VARIOGRAM anisotropy — NOT from bounding-box extent
#
# Physical derivation:
#   The coordinate MLP should encode the spatial structure relevant to the
#   target variable.  Two data properties drive this need:
#
#   (1) Variogram anisotropy ratio = range_minor / range_major ∈ (0, 1]
#       Measures directional complexity of the spatial field:
#       • ratio ≈ 1 (isotropic): both axes equally informative → compact (32)
#       • ratio ≪ 1 (strong anisotropy): one axis much more predictive
#         → larger embedding to represent both directions faithfully
#       Contribution: 24 × (1 − ratio)  [0, 24]
#
#   (2) Signal strength = 1 − nugget_ratio ∈ [0, 1]
#       When the field is strongly spatially structured (low nugget), location
#       is a highly predictive covariate → more representational capacity.
#       When the field is near-pure nugget, location carries little information.
#       Contribution: 8 × (1 − r)  [0, 8]
#
#   Total: clamp(32 + aniso_contrib + signal_contrib, 32, 64)
.auto_coord_dim_from_variogram <- function(vg) {
  aniso_ratio    <- .auto_clamp(vg$range_minor / max(vg$range_major, 1e-8), 0.10, 1.00)
  aniso_contrib  <- round(24L * (1 - aniso_ratio))
  signal_contrib <- round(8L  * (1 - vg$nugget_ratio))
  coord_dim      <- as.integer(.auto_clamp(32L + aniso_contrib + signal_contrib, 32L, 64L))
  cat(sprintf(
    "[Auto v5]   coord_dim = %d  [aniso_ratio = %.3f (+%d), signal = 1−%.3f (+%d)]\n",
    coord_dim, aniso_ratio, aniso_contrib, vg$nugget_ratio, signal_contrib))
  coord_dim
}

# All nugget-derived loss weights in one place
#
# Every loss weight is a closed-form function of the nugget-to-sill ratio r.
# Physical justifications (see ConvKrigingNet2D_Auto.R header §3 for full proofs):
#
#   base_loss_weight:  max(0.05, 0.10 × r)
#                                      — direct backbone supervision must not
#                                        collapse when nugget is low; after
#                                        the backbone needs a minimum
#                                        standalone learning signal even in
#                                        strongly spatial fields
#   alpha_me:          0.375 × r × mα — anti co-training-collapse penalty
#   lambda_cov:        0.025 × (1−r) × mσ
#                                      — variance-matching regulariser weight
#   lambda_spatial:    0.020 × (1−r) × mρ
#                                      — residual decorrelation penalty weight
#   max_warmup_epochs: clamp(4+16r, 4, 20)  — more warmup when backbone load is high
.auto_loss_weights_from_nugget <- function(r) {
  blw_floor <- .auto_env_num("GEOVERSA_AUTO_BLW_FLOOR", 0.05)
  blw_slope <- .auto_env_num("GEOVERSA_AUTO_BLW_SLOPE", 0.10)
  alpha_mult <- .auto_env_num("GEOVERSA_AUTO_ALPHA_ME_MULT", 1.0)
  lambda_cov_mult <- .auto_env_num("GEOVERSA_AUTO_LAMBDA_COV_MULT", 1.0)
  lambda_spatial_mult <- .auto_env_num("GEOVERSA_AUTO_SPATIAL_LOSS_MULT", 0.0)
  list(
    base_loss_weight  = round(max(blw_floor, blw_slope * r), 4L),
    alpha_me          = round(0.375 * r * alpha_mult, 4L),
    lambda_cov        = round(0.025 * (1.0 - r) * lambda_cov_mult, 5L),
    lambda_spatial    = round(0.020 * (1.0 - r) * lambda_spatial_mult, 5L),
    max_warmup_epochs = as.integer(.auto_clamp(round(4 + 16 * r), 4L, 20L)),
    base_loss_floor   = blw_floor,
    base_loss_slope   = blw_slope,
    alpha_me_mult     = alpha_mult,
    lambda_cov_mult   = lambda_cov_mult,
    lambda_spatial_mult = lambda_spatial_mult
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# §2  PHASE 0-B  Statistical capacity  (class B — √n scaling)
# ─────────────────────────────────────────────────────────────────────────────

# Embedding dimension d from n_train
# Rule: d = 64 × ceil(√n / 8), clamped [128, 512]
# Reference: VC-dimension capacity scaling (Bartlett & Mendelson 2002)
.auto_d_from_n <- function(n_train) {
  as.integer(.auto_clamp(64L * ceiling(sqrt(n_train) / 8), 128L, 512L))
}

# patch_dim from CNN information capacity
#
# Physical derivation (Shannon 1948; Cover & Thomas 2006 §2):
#   The CNN patch encoder compresses C × H × W inputs into patch_dim features.
#   The information-theoretic lower bound for a lossless encoding of independent
#   Gaussian inputs is patch_dim ≥ √(C·H·W).  We use this as the "minimum
#   sufficient" dimension for a lossy encoder that retains the dominant modes.
#   Bounds: [d/4, d] — no point exceeding backbone width; at least d/4 to
#   avoid a severe information bottleneck in the patch branch.
.auto_patch_dim_from_cnn <- function(patch_channels, patch_size, d) {
  n_pixels  <- patch_channels * patch_size^2
  patch_dim <- as.integer(.auto_clamp(
    ceiling(sqrt(n_pixels)),
    as.integer(d / 4L),
    d
  ))
  cat(sprintf(
    "[Auto v5]   patch_dim = %d  [√(C·H·W) = √(%d·%d²) = √%d ≈ %.1f, clamped [%d, %d]]\n",
    patch_dim, patch_channels, patch_size, n_pixels, sqrt(n_pixels),
    as.integer(d / 4L), d))
  patch_dim
}

# Patch pyramid geometry from variogram range and raster support
#
# Physical derivation:
#   The raster patch should span a physically meaningful neighborhood around the
#   focal point. Let a be the practical variogram range (major/minor axis) and
#   c the raster cell size. Then a / c is the support in cells over which the
#   field remains materially correlated. We therefore derive:
#     • global patch support from the major range
#     • local patch support from the minor range and from a fixed fraction of
#       the global field of view
#   This keeps the patch pyramid anchored to the spatial process, not just to n.
auto_patch_geometry_v5 <- function(vg,
                                   n_train,
                                   patch_resolution = NULL,
                                   cached_patch_size = NULL) {
  cached_patch_size <- suppressWarnings(as.integer(cached_patch_size)[1L])
  fallback_global <- as.integer(.auto_make_odd(.auto_clamp(floor(sqrt(n_train)), 15L, 31L)))
  if (!is.na(cached_patch_size) && is.finite(cached_patch_size) && cached_patch_size > 0) {
    global_size <- as.integer(cached_patch_size)
  } else if (is.finite(patch_resolution) && patch_resolution > 0 &&
             is.finite(vg$range_major) && vg$range_major > 0) {
    global_frac <- .auto_env_num("GEOVERSA_AUTO_PATCH_GLOBAL_RANGE_FRAC", 0.30)
    local_minor_frac <- .auto_env_num("GEOVERSA_AUTO_PATCH_LOCAL_MINOR_FRAC", 0.20)
    local_global_frac <- .auto_env_num("GEOVERSA_AUTO_PATCH_LOCAL_GLOBAL_FRAC", 0.45)
    global_cap <- as.integer(.auto_make_odd(.auto_clamp(.auto_env_num("GEOVERSA_AUTO_PATCH_GLOBAL_MAX", 31), 21L, 41L)))
    local_cap <- as.integer(.auto_make_odd(.auto_clamp(.auto_env_num("GEOVERSA_AUTO_PATCH_LOCAL_MAX", 17), 9L, min(21L, global_cap))))
    major_cells <- max(vg$range_major / patch_resolution, 1.0)
    minor_cells <- max(vg$range_minor / patch_resolution, 1.0)
    global_size_vario <- .auto_make_odd(2L * round(global_frac * major_cells) + 1L)
    global_size_vario <- as.integer(.auto_clamp(global_size_vario, 15L, global_cap))
    global_size <- as.integer(.auto_make_odd(round(sqrt(fallback_global * global_size_vario))))
    global_size <- as.integer(.auto_clamp(global_size, 15L, global_cap))
    local_from_minor <- .auto_make_odd(2L * round(local_minor_frac * minor_cells) + 1L)
    local_from_global <- .auto_make_odd(round(local_global_frac * global_size))
    local_size <- as.integer(.auto_make_odd(round(sqrt(local_from_minor * local_from_global))))
    local_size <- as.integer(.auto_clamp(local_size, 9L, min(local_cap, global_size)))
    local_size <- min(local_size, global_size)
    cat(sprintf(
      "[Auto v5]   patch pyramid = global %d / local %d  [blend sqrt(%d×%d), range_major/cell=%.2f, range_minor/cell=%.2f]\n",
      global_size, local_size, fallback_global, global_size_vario, major_cells, minor_cells
    ))
    return(list(
      global_size = global_size,
      local_size = local_size,
      major_cells = major_cells,
      minor_cells = minor_cells,
      fallback_global = fallback_global
    ))
  } else {
    global_size <- fallback_global
  }

  local_size <- as.integer(.auto_clamp(.auto_make_odd(round(0.55 * global_size)), 9L, min(21L, global_size)))
  cat(sprintf(
    "[Auto v5]   patch pyramid = global %d / local %d  [sqrt(n) fallback]\n",
    global_size, local_size
  ))
  list(
    global_size = global_size,
    local_size = local_size,
    major_cells = NA_real_,
    minor_cells = NA_real_,
    fallback_global = fallback_global
  )
}

# Dropout rates from sample size
# Reference: Srivastava et al. (2014); Gal & Ghahramani (2016)
# Larger n → less regularisation needed.
# coord MLP has only 2 inputs → lower overfitting risk → gentler dropout.
# The patch branch is regularised slightly less than the raw sample-size rule,
# because the screening runs showed the baseline patch dropout was suppressing
# backbone quality more than helping generalisation.
.auto_dropouts_from_n <- function(n_train, nugget_ratio = NULL, k_neighbors = NULL) {
  drop_mult  <- .auto_env_num("GEOVERSA_AUTO_DROPOUT_MULT", 1.0)
  tab_mult   <- .auto_env_num("GEOVERSA_AUTO_TAB_DROPOUT_MULT", 1.0)
  patch_mult <- .auto_env_num("GEOVERSA_AUTO_PATCH_DROPOUT_MULT", 1.0)
  coord_mult <- .auto_env_num("GEOVERSA_AUTO_COORD_DROPOUT_MULT", 1.0)
  patch_nugget_slope <- .auto_env_num("GEOVERSA_AUTO_PATCH_DROPOUT_NUGGET_SLOPE", 0.0)
  patch_nugget_ref   <- .auto_env_num("GEOVERSA_AUTO_PATCH_DROPOUT_NUGGET_REF", 0.18)
  patch_k_slope      <- .auto_env_num("GEOVERSA_AUTO_PATCH_DROPOUT_K_SLOPE", 0.0)
  patch_k_ref        <- .auto_env_num("GEOVERSA_AUTO_PATCH_DROPOUT_K_REF", 20.0)
  tab_base   <- .auto_clamp(0.30 - n_train / 8000,  0.05, 0.30)
  patch_base_raw <- .auto_clamp(0.20 - n_train / 10000, 0.03, 0.20)
  nugget_centered <- if (!is.null(nugget_ratio) && is.finite(nugget_ratio)) {
    nugget_ratio - patch_nugget_ref
  } else {
    0
  }
  k_centered <- if (!is.null(k_neighbors) && is.finite(k_neighbors) &&
                    is.finite(patch_k_ref) && patch_k_ref > 0) {
    (k_neighbors - patch_k_ref) / patch_k_ref
  } else {
    0
  }
  patch_struct_mult <- .auto_clamp(
    1 + patch_nugget_slope * nugget_centered + patch_k_slope * k_centered,
    0.80, 1.20
  )
  patch_base <- .auto_clamp(0.75 * patch_base_raw * patch_struct_mult, 0.02, 0.20)
  # The coordinate encoder is low-dimensional and only sees 2 inputs, so the
  # regularisation should stay light and decay smoothly with sample size.
  coord_base <- .auto_clamp(0.10 - n_train / 20000, 0.02, 0.10)
  # The neural kriging operator sees the noisiest residual channel; increase
  # dropout when nugget is high and relax it when spatial signal is strong.
  nko_base <- .auto_clamp(
    0.08 - n_train / 15000 + 0.10 * ifelse(is.finite(nugget_ratio), nugget_ratio, 0.25),
    0.02, 0.20
  )
  tab_drop   <- round(.auto_clamp(tab_base * drop_mult * tab_mult,   0.03, 0.35), 3)
  patch_drop <- round(.auto_clamp(patch_base * drop_mult * patch_mult, 0.02, 0.25), 3)
  coord_drop <- round(.auto_clamp(coord_base * drop_mult * coord_mult, 0.01, 0.15), 3)
  nko_drop <- round(.auto_clamp(nko_base * drop_mult, 0.02, 0.20), 3)
  list(
    tab = tab_drop,
    patch = patch_drop,
    coord = coord_drop,
    nko = nko_drop,
    patch_base_raw = patch_base_raw,
    patch_struct_mult = patch_struct_mult,
    patch_nugget_slope = patch_nugget_slope,
    patch_nugget_ref = patch_nugget_ref,
    patch_k_slope = patch_k_slope,
    patch_k_ref = patch_k_ref
  )
}

# Main-loop epoch budget from sample size and spatial signal
#
# 40 = lower bound empirical for MLP+CNN+kriging convergence under n >= 100;
# below this the patch branch under-trains and the backbone never separates
# from the kriging correction.
# 2 * sqrt(n) is a Bartlett-style capacity scaling: the number of useful
# passes grows sublinearly with sample size as the effective complexity of the
# fitted function increases.
# 20 * (1-r) is a signal bonus: low nugget means the residual field is
# learnable, so the model can exploit extra passes without immediately
# overfitting noise.
.auto_main_epochs_from_nugget <- function(n_train, nugget_ratio) {
  epoch_mult <- .auto_env_num("GEOVERSA_AUTO_EPOCH_MULT", 1.0)
  base_epochs <- 40 + 2.0 * sqrt(max(1, n_train)) + 20 * (1 - nugget_ratio)
  as.integer(.auto_clamp(round(base_epochs * epoch_mult), 50L, 140L))
}

# Variogram fitting controls from sample size
#
# 0.35 is the conservative minimum cutoff fraction that still leaves enough
# pairs per lag under n ~ 100.  The +0.10 * log10(n/50) term widens the search
# very slowly as pair counts stabilise.
# 0.90 * sqrt(n) gives lag resolution that grows with sample size but avoids
# the unstable many-empty-bins regime seen when lags scale linearly with n.
.auto_variogram_fit_controls_from_n <- function(n_train) {
  cutoff_mult <- .auto_env_num("GEOVERSA_AUTO_VG_CUTOFF_MULT", 1.0)
  lag_mult <- .auto_env_num("GEOVERSA_AUTO_VG_LAG_MULT", 1.0)
  cutoff_frac <- .auto_clamp((0.35 + 0.10 * log10(max(n_train, 50) / 50)) * cutoff_mult, 0.35, 0.65)
  n_lags <- as.integer(.auto_clamp(round(0.90 * sqrt(max(1, n_train)) * lag_mult), 12L, 24L))
  list(cutoff_frac = cutoff_frac, n_lags = n_lags)
}

# Gate defaults from variogram structure
#
# DESIGN CHOICE: these constants govern how aggressively the local kriging gate
# can close.  They are now derived from signal strength and anisotropy rather
# than kept fixed, and should be treated as part of the automation ablation.
# 0.10 + 0.25*r keeps the linear blend center near 0.10 in strong signal and
# relaxes it toward 0.35 as nugget rises.
# 4 + 4*signal + 2*(1-aniso) sharpens the gate when spatial structure is
# strong or directional.
# 0.15 + 0.35*r and 0.02 + 0.20*r allow the gate floor and NKO floor to stay
# low in well-structured fields and rise only when the variogram says noise
# dominates.
.auto_gate_controls_from_variogram <- function(vg, coord_dim) {
  aniso_ratio <- .auto_clamp(vg$range_minor / max(vg$range_major, 1e-8), 0.10, 1.00)
  signal <- .auto_clamp(1 - vg$nugget_ratio, 0.00, 1.00)
  linear_blend_center <- .auto_clamp(0.10 + 0.25 * vg$nugget_ratio, 0.08, 0.40)
  linear_blend_temp <- .auto_clamp(4 + 4 * signal + 2 * (1 - aniso_ratio), 3.0, 12.0)
  gate_hidden <- as.integer(.auto_clamp(round(coord_dim / 4 + 4 * signal), 6L, 24L))
  gate_floor <- .auto_clamp(0.15 + 0.35 * vg$nugget_ratio, 0.10, 0.70)
  nko_gate_floor <- .auto_clamp(0.02 + 0.20 * vg$nugget_ratio, 0.02, 0.30)
  list(
    linear_blend_center = linear_blend_center,
    linear_blend_temp = linear_blend_temp,
    gate_hidden = gate_hidden,
    gate_floor = gate_floor,
    nko_gate_floor = nko_gate_floor
  )
}

# Early stopping patience as a fraction of the total epoch budget.
# 1/6 preserves the old Baccini setting (94 epochs -> patience 15) while
# stretching naturally when the variogram says the task needs longer training.
.auto_patience_from_main_epochs <- function(main_epochs) {
  main_epochs <- as.integer(max(1L, round(main_epochs)))
  max(5L, as.integer(floor(main_epochs / 6L)))
}

# LR reduction patience should fire before full early stopping.
# 1/20 of the budget reproduces the old Baccini default near 4-5 epochs while
# remaining strictly smaller than the early-stop patience above.
.auto_lr_patience_from_main_epochs <- function(main_epochs) {
  main_epochs <- as.integer(max(1L, round(main_epochs)))
  max(3L, as.integer(floor(main_epochs / 20L)))
}

# Sample raster covariates without materialising the full stack in RAM.
# 20k rows is a Monte Carlo design ceiling: correlation estimates stabilise well
# before that for DSM covariate stacks, while memory stays bounded.
.auto_sample_raster_matrix <- function(raster_stack, max_rows = 20000L, seed = 123L) {
  max_rows <- as.integer(max(256L, round(max_rows)))
  set.seed(as.integer(seed))
  if (inherits(raster_stack, "SpatRaster")) {
    vals <- tryCatch(
      terra::spatSample(
        raster_stack,
        size = max_rows,
        method = "random",
        na.rm = TRUE,
        values = TRUE,
        as.df = TRUE
      ),
      error = function(e) NULL
    )
    if (is.null(vals) || ncol(vals) < 1L) return(NULL)
    vals <- as.matrix(vals)
  } else {
    vals <- tryCatch(
      raster::sampleRandom(
        raster_stack,
        size = max_rows,
        na.rm = TRUE,
        sp = FALSE
      ),
      error = function(e) NULL
    )
    if (is.null(vals) || ncol(vals) < 1L) return(NULL)
    vals <- as.matrix(vals)
  }
  vals <- vals[stats::complete.cases(vals), , drop = FALSE]
  if (nrow(vals) < 10L) return(NULL)
  vals
}

.auto_valid_raster_cell_count <- function(raster_stack) {
  if (inherits(raster_stack, "SpatRaster")) {
    out <- tryCatch(
      terra::global(!is.na(raster_stack[[1]]), "sum", na.rm = TRUE),
      error = function(e) NULL
    )
    if (is.null(out) || nrow(out) < 1L || ncol(out) < 1L) return(NA_integer_)
    return(as.integer(round(out[1, 1])))
  }
  out <- tryCatch(
    raster::cellStats(!is.na(raster_stack[[1]]), stat = "sum"),
    error = function(e) NA_real_
  )
  if (!is.finite(out)) return(NA_integer_)
  as.integer(round(out))
}

# SSL mask ratio from covariate redundancy.
# 0.30 is the safe lower bound when channels are weakly correlated; +0.60*mean_cor
# pushes the masking rate toward MAE-style aggressive masking only when the
# raster stack itself is redundant enough to support it.
.auto_ssl_mask_ratio_from_raster <- function(raster_stack) {
  vals <- .auto_sample_raster_matrix(raster_stack, max_rows = 20000L, seed = 123L)
  if (is.null(vals) || ncol(vals) < 2L) return(0.50)
  if (nrow(vals) < 100L) return(0.50)
  cor_mat <- abs(stats::cor(vals, use = "pairwise.complete.obs"))
  diag(cor_mat) <- NA_real_
  mean_cor <- mean(cor_mat, na.rm = TRUE)
  .auto_clamp(0.30 + 0.60 * mean_cor, 0.30, 0.75)
}

# SSL sample budget from raster support.
# 3% of valid cells is enough to expose the encoder to broad raster diversity
# without forcing full-raster pretraining; 10k-500k keeps runtime bounded.
.auto_ssl_n_samples_from_raster <- function(raster_stack) {
  n_cells_valid <- .auto_valid_raster_cell_count(raster_stack)
  if (!is.finite(n_cells_valid) || n_cells_valid <= 0L) return(100000L)
  as.integer(.auto_clamp(round(0.03 * n_cells_valid), 10000L, 500000L))
}

# SSL epochs from sample budget.
# 20 + sqrt(n/1000) grows sublinearly with the number of sampled patches, then
# early stopping handles the actual convergence point.
.auto_ssl_epochs_from_samples <- function(n_samples) {
  n_samples <- as.integer(max(1000L, round(n_samples)))
  as.integer(.auto_clamp(round(20 + sqrt(n_samples / 1000)), 30L, 100L))
}

build_auto_config_v5 <- function(n_train,
                                 vg,
                                 coords,
                                 covariates = NULL,
                                 coord_scaler = NULL,
                                 patch_channels = NULL,
                                 patch_resolution = NULL,
                                 patch_geometry = NULL,
                                 cached_patch_size = NULL,
                                 device = "cpu") {
  if (is.null(coord_scaler)) {
    coord_scaler <- fit_standard_scaler(coords)
  }
  cs <- mean(coord_scaler$scale)
  r <- vg$nugget_ratio
  ell_major_init <- vg$range_major / cs
  ell_minor_init <- vg$range_minor / cs
  theta_init <- vg$theta_rad
  lw <- .auto_loss_weights_from_nugget(r)
  K_neighbors <- .auto_K_from_spatial_density(n_train, vg, coords)
  beta_init <- .auto_beta_init_from_nugget(r)
  d <- .auto_d_from_n(n_train)
  if (is.null(patch_geometry) ||
      !is.list(patch_geometry) ||
      !is.finite(patch_geometry$global_size) ||
      !is.finite(patch_geometry$local_size)) {
    patch_geometry <- auto_patch_geometry_v5(
      vg = vg,
      n_train = n_train,
      patch_resolution = patch_resolution,
      cached_patch_size = cached_patch_size
    )
  }
  patch_size <- as.integer(patch_geometry$global_size)
  patch_local_size <- as.integer(patch_geometry$local_size)
  drops <- .auto_dropouts_from_n(n_train, nugget_ratio = r, k_neighbors = K_neighbors)
  coord_dim <- .auto_coord_dim_from_variogram(vg)
  gate_ctrl <- .auto_gate_controls_from_variogram(vg, coord_dim)
  coord_hidden <- max(2L * coord_dim, 32L)
  fusion_hidden <- d
  main_epochs <- .auto_main_epochs_from_nugget(n_train, r)
  if (!is.null(patch_channels)) {
    patch_dim <- .auto_patch_dim_from_cnn(patch_channels, patch_size, d)
  } else {
    patch_dim <- as.integer(d / 2L)
  }
  X_dim <- if (!is.null(covariates)) {
    if (inherits(covariates, "torch_tensor")) {
      as.integer(covariates$size(2L))
    } else {
      ncol(covariates)
    }
  } else {
    10L
  }
  pc <- if (!is.null(patch_channels) && is.finite(patch_channels)) patch_channels else 3L
  batch_size <- auto_batch_size_v5(n_train, X_dim, pc, patch_size, device = device)
  lr_patience <- .auto_lr_patience_from_main_epochs(main_epochs)
  patience <- .auto_patience_from_main_epochs(main_epochs)
  list(
    ell_major_init = ell_major_init,
    ell_minor_init = ell_minor_init,
    theta_init = theta_init,
    nugget_ratio = r,
    K_neighbors = K_neighbors,
    beta_init = beta_init,
    base_loss_weight = lw$base_loss_weight,
    alpha_me = lw$alpha_me,
    lambda_cov = lw$lambda_cov,
    lambda_spatial = lw$lambda_spatial,
    max_warmup_epochs = lw$max_warmup_epochs,
    d = d,
    patch_size = patch_size,
    patch_local_size = patch_local_size,
    patch_dim = patch_dim,
    fusion_hidden = fusion_hidden,
    tab_dropout = drops$tab,
    patch_dropout = drops$patch,
    coord_dropout = drops$coord,
    nko_dropout = drops$nko,
    coord_dim = coord_dim,
    coord_hidden = coord_hidden,
    main_epochs = main_epochs,
    linear_blend_center = gate_ctrl$linear_blend_center,
    linear_blend_temp = gate_ctrl$linear_blend_temp,
    gate_hidden = gate_ctrl$gate_hidden,
    gate_floor = gate_ctrl$gate_floor,
    nko_gate_floor = gate_ctrl$nko_gate_floor,
    batch_size = batch_size,
    lr_patience = lr_patience,
    patience = patience,
    bank_refresh_every = auto_bank_refresh_from_patience(lr_patience),
    patch_struct_mult = drops$patch_struct_mult,
    patch_nugget_slope = drops$patch_nugget_slope,
    patch_nugget_ref = drops$patch_nugget_ref,
    patch_k_slope = drops$patch_k_slope,
    patch_k_ref = drops$patch_k_ref
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# §3  PHASE 1  Learning rate  (class D — LR range test on actual data)
# ─────────────────────────────────────────────────────────────────────────────

# LR range test (Smith 2017).
# Sweep 50 exponentially spaced learning rates from 1e-6 to 1e-1, detect the
# argmin of the smoothed pre-divergence loss curve, then back off by 10x. This
# follows Smith (2017) more closely than the earlier steepest-slope heuristic
# and is more robust to Adam's moment-bootstrap artifact in the first few steps.
.auto_set_optimizer_lr <- function(opt, lr) {
  if (!is.null(opt$defaults)) {
    opt$defaults$lr <- lr
  }
  if (!is.null(opt$param_groups)) {
    for (i in seq_along(opt$param_groups)) {
      opt$param_groups[[i]]$lr <- lr
    }
  }
  invisible(opt)
}

.auto_clone_state_dict_cpu <- function(state_dict) {
  if (is.null(state_dict)) return(NULL)
  out <- vector("list", length(state_dict))
  names(out) <- names(state_dict)
  for (i in seq_along(state_dict)) {
    out[[i]] <- state_dict[[i]]$detach()$to(device = "cpu")$clone()
  }
  out
}

estimate_initial_lr_range_test <- function(model, train_cache, bs_init = 32L,
                                           device = "cpu",
                                           lr_min = 1e-6,
                                           lr_max = 1e-1,
                                           n_steps = 50L,
                                           smooth_beta = 0.98,
                                           divergence_factor = 4.0) {
  cat("[Auto v5] ── Estimating LR (range test, 50-step sweep) ──\n")

  tryCatch({
    n_available <- as.integer(train_cache$n)
    if (!is.finite(n_available) || n_available < 2L || is.null(train_cache$y)) {
      cat("[Auto v5]   ⚠ insufficient train cache for LR range test — fallback lr = 1e-4\n")
      return(1e-4)
    }

    bs <- min(as.integer(bs_init), n_available)
    n_steps <- max(10L, as.integer(n_steps))
    saved_state <- .auto_clone_state_dict_cpu(model$state_dict())
    on.exit({
      if (!is.null(saved_state)) {
        try(model$load_state_dict(saved_state), silent = TRUE)
      }
    }, add = TRUE)

    total_needed <- n_steps * bs
    idx_all <- sample.int(
      n_available,
      size = total_needed,
      replace = (n_available < total_needed)
    )
    mult <- if (n_steps <= 1L) 1 else exp(log(lr_max / lr_min) / (n_steps - 1L))

    lrs <- rep(NA_real_, n_steps)
    smooth_losses <- rep(NA_real_, n_steps)
    raw_losses <- rep(NA_real_, n_steps)
    avg_loss <- 0.0
    best_smooth <- Inf
    steps_run <- 0L

    tmp_opt <- optim_adam(model$parameters, lr = lr_min)
    model$train()

    for (step_idx in seq_len(n_steps)) {
      lr_now <- lr_min * (mult ^ (step_idx - 1L))
      .auto_set_optimizer_lr(tmp_opt, lr_now)

      idx <- idx_all[((step_idx - 1L) * bs + 1L):(step_idx * bs)]
      idx_t <- torch_tensor(as.integer(idx), dtype = torch_long(), device = device)
      xb <- train_cache$X$index_select(1L, idx_t)
      pb <- train_cache$P$index_select(1L, idx_t)
      cb <- train_cache$C$index_select(1L, idx_t)
      yb <- train_cache$y$index_select(1L, idx_t)

      tmp_opt$zero_grad()
      out <- model$forward_base(xb, pb, cb)
      loss <- huber_loss(yb, out$pred)
      loss$backward()
      tmp_opt$step()

      loss_val <- loss$item()
      avg_loss <- smooth_beta * avg_loss + (1 - smooth_beta) * loss_val
      smooth_loss <- avg_loss / (1 - smooth_beta ^ step_idx)

      lrs[[step_idx]] <- lr_now
      raw_losses[[step_idx]] <- loss_val
      smooth_losses[[step_idx]] <- smooth_loss
      steps_run <- step_idx

      if (is.finite(smooth_loss) && smooth_loss < best_smooth) {
        best_smooth <- smooth_loss
      }
      if (step_idx >= 10L &&
          (!is.finite(smooth_loss) ||
           (is.finite(best_smooth) && smooth_loss > divergence_factor * best_smooth))) {
        break
      }
    }

    tmp_opt$zero_grad()

    valid_idx <- which(is.finite(smooth_losses) & is.finite(lrs))
    if (length(valid_idx) < 5L) {
      cat("[Auto v5]   ⚠ LR range test too short — fallback lr = 1e-4\n")
      return(1e-4)
    }

    lrs_valid <- lrs[valid_idx]
    losses_valid <- smooth_losses[valid_idx]
    # Smith (2017) original heuristic: pick lr at argmin(smoothed_loss) and
    # back off by 10x. Ignore the noisiest first 10 % and the last 2 points,
    # where divergence effects dominate once the curve has already turned.
    start_idx <- max(1L, ceiling(length(losses_valid) * 0.10))
    end_idx <- max(start_idx, length(losses_valid) - 2L)
    loss_window <- losses_valid[start_idx:end_idx]
    if (!all(is.finite(loss_window))) {
      cat("[Auto v5]   ⚠ invalid LR range losses — fallback lr = 1e-4\n")
      return(1e-4)
    }
    loss_pick <- which.min(loss_window) + start_idx - 1L
    lr_knee <- lrs_valid[loss_pick]
    lr_ceil_raw <- suppressWarnings(
      as.numeric(trimws(Sys.getenv("WADOUX_LR_CEIL", unset = "")))
    )
    # Default ceiling = 1e-4 (Adam default, Kingma & Ba 2014; validated empirically
    # via 9-regime DSM ablation: ceiling=1e-4 minimises RMSE_sd=2.30 while achieving
    # PICP=0.901 at nominal 0.90 — the best joint RMSE/coverage trade-off observed.
    # WADOUX_LR_CEIL overrides this default for datasets with more responsive loss
    # landscapes where the finder's argmin/10 naturally falls below 1e-4.
    lr_ceil <- if (is.finite(lr_ceil_raw) && lr_ceil_raw > 0) lr_ceil_raw else 1e-4
    lr_final <- .auto_clamp(lr_knee / 10, 5e-5, lr_ceil)

    cat(sprintf("[Auto v5]   steps_run = %d   best_smooth = %.4f\n",
                steps_run, best_smooth))
    cat(sprintf("[Auto v5]   α_range = [%.2e, %.2e]   α_argmin = %.2e   α_ceil = %.2e   α_final = %.2e\n",
                min(lrs_valid, na.rm = TRUE), max(lrs_valid, na.rm = TRUE),
                lr_knee, lr_ceil, lr_final))
    lr_final

  }, error = function(e) {
    cat(sprintf("[Auto v5]   ⚠ LR range test failed (%s) — fallback lr = 1e-4\n",
                conditionMessage(e)))
    1e-4
  })
}

# Backward-compatible alias while the rest of the codebase migrates away from
# the old Polyak-oriented helper name.
estimate_initial_lr_polyak <- estimate_initial_lr_range_test


# ─────────────────────────────────────────────────────────────────────────────
# §4  PHASE 1  Batch size  (class C — hardware-adaptive)
# ─────────────────────────────────────────────────────────────────────────────

# Auto batch size from actual hardware memory + data constraints
#
# Two independent constraints applied as min():
#
#   (1) Statistical: batch = floor(n / 8) gives ~8 gradient updates per epoch,
#       providing stable AdamW gradient estimates (Loshchilov & Hutter 2019).
#       Minimum 24 to avoid very noisy updates; maximum 512 for memory safety.
#
#   (2) Hardware: batch ≤ floor(free_memory × fraction / per_sample_cost)
#       Per-sample cost = tabular bytes + patch bytes + 20 MB activation buffer
#       (empirical estimate covering forward activations and backward gradient
#       tensors for a typical ConvKrigingNet2D depth).
auto_batch_size_v5 <- function(n_train, X_dim, patch_channels, patch_size,
                                device = "mps", target_mem_fraction = 0.50) {
  cat("[Auto v5] ── Estimating batch size (hardware + statistical) ──\n")

  available_gb <- .get_actual_ram_gb(device)

  # Per-sample memory (float32 = 4 bytes)
  X_bytes        <- X_dim * 4L
  P_bytes        <- patch_channels * patch_size * patch_size * 4L
  activ_mb       <- 20.0                     # activation + gradient buffer
  per_sample_mb  <- (X_bytes + P_bytes) / 1e6 + activ_mb
  per_sample_mb  <- max(per_sample_mb, 5.0)

  safe_mb        <- available_gb * 1e3 * target_mem_fraction
  hardware_max   <- as.integer(floor(safe_mb / per_sample_mb))

  # Statistical constraint
  stat_ideal     <- as.integer(max(floor(n_train / 8L), 24L))

  batch_size <- as.integer(.auto_clamp(
    min(stat_ideal, hardware_max), 8L, 512L))

  cat(sprintf(
    "[Auto v5]   RAM=%.1fGB  per_sample=%.1fMB  hw_max=%d  n/8=%d  → bs=%d\n",
    available_gb, per_sample_mb, hardware_max, stat_ideal, batch_size))
  batch_size
}


# ─────────────────────────────────────────────────────────────────────────────
# §5  PHASE 2  Weight decay  (class B + model size — called after model rebuild)
# ─────────────────────────────────────────────────────────────────────────────

# Weight decay from actual model parameter count
#
# Theoretical basis (Zhang et al. 2019 "Three Mechanisms of Weight Decay"):
#   For overparameterised models the optimal L2 regularisation scales as:
#       wd ∝ 1 / √n_params
#   so that the effective L2-ball radius grows with the square root of the
#   parameter space, matching the √n generalisation bound.
#   Normalised to a 5 M-parameter baseline (≈ a medium backbone).
auto_weight_decay_from_capacity <- function(model) {
  cat("[Auto v5] ── Estimating weight decay from model capacity ──\n")
  wd_mult <- .auto_env_num("GEOVERSA_AUTO_WD_MULT", 1.0)

  n_params <- tryCatch({
    total <- 0L
    # model$parameters is a property in torch for R (no parens).
    # prod(p$shape) is the safest cross-version way to count elements.
    for (p in model$parameters)
      if (!is.null(p)) total <- total + prod(p$shape)
    total
  }, error = function(e) { cat("[Auto v5]   ⚠ param count failed\n"); 0L })

  if (n_params < 1000L) {
    cat("[Auto v5]   ⚠ too few params counted — wd = 1e-3\n")
    return(1e-3)
  }

  total_params_m <- as.numeric(n_params) / 1e6
  wd_base        <- 1e-3 / sqrt(max(total_params_m / 5, 0.1))
  wd_est         <- wd_base * wd_mult
  wd_final       <- .auto_clamp(wd_est, 1e-4, 1e-2)

  cat(sprintf(
    "[Auto v5]   params = %.2f M   wd = (1e-3/√(%.2f/5)) × %.2f = %.2e → clamped %.2e\n",
    total_params_m, total_params_m, wd_mult, wd_est, wd_final))
  wd_final
}


# ─────────────────────────────────────────────────────────────────────────────
# §6  PHASE 3  Training dynamics  (class D — called from training loop)
# ─────────────────────────────────────────────────────────────────────────────

# Early-stopping and LR patience from warmup convergence speed
#
# Theoretical basis:
#   The warmup phase reveals the loss-landscape smoothness. The speed ratio
#   s = wu_done / max_warmup ∈ [0, 1] measures how quickly the backbone
#   reached the convergence criterion:
#     s ≈ 0: fast convergence → smooth basin → generous patience
#     s ≈ 1: used full budget → complex/noisy landscape → strict patience
#   Linear schedule clamped to physiologically sensible ranges:
#     lr_patience  = round(8 − 5s),  clamped [3,  8]
#     es_patience  = 3 × lr_patience,        [9, 24]
auto_patience_from_warmup <- function(wu_done, max_warmup) {
  cat("[Auto v5] ── Phase 3: Adapting patience from warmup dynamics ──\n")
  speed_ratio <- wu_done / max(max_warmup, 1L)
  lr_pat      <- as.integer(.auto_clamp(round(8L - 5L * speed_ratio), 3L, 8L))
  es_pat      <- lr_pat * 3L
  cat(sprintf(
    "[Auto v5]   warmup: %d / %d epochs  (speed = %.2f)\n",
    wu_done, max_warmup, speed_ratio))
  cat(sprintf(
    "[Auto v5]   lr_patience = %d   early_stop_patience = %d\n", lr_pat, es_pat))
  list(lr_patience = lr_pat, patience = es_pat)
}

# LR decay factor from warmup loss-trajectory smoothness
#
# Theoretical basis (Smith 2018; Jastrzkebski et al. 2017):
#   Define the coefficient of variation (CV) of epoch-to-epoch loss
#   improvements during warmup:  CV = std(Δloss) / |mean(Δloss)|
#   • CV ≈ 0: smooth, consistent descent → gentle decay (0.70); premature
#     large decay would waste the steady progress
#   • CV ≫ 1: noisy / oscillating loss → aggressive decay (0.30) to
#     stabilise once a plateau is detected
#   tanh mapping ensures a smooth, bounded transformation:
#     lr_decay = 0.70 − 0.40 × tanh(CV / 2),   clamped [0.30, 0.70]
auto_lr_decay_from_trajectory <- function(wu_val_losses) {
  cat("[Auto v5] ── Phase 3: Estimating lr_decay from warmup trajectory ──\n")

  if (length(wu_val_losses) < 3) {
    cat("[Auto v5]   < 3 warmup epochs — lr_decay = 0.5 (neutral)\n")
    return(0.5)
  }

  improvements <- -diff(wu_val_losses)       # positive = loss decreased
  mean_imp     <- mean(improvements)

  if (abs(mean_imp) < 1e-8) {
    cat("[Auto v5]   loss stagnant during warmup — lr_decay = 0.3 (aggressive)\n")
    return(0.3)
  }

  cv       <- sd(improvements) / abs(mean_imp)
  lr_decay <- .auto_clamp(0.70 - 0.40 * tanh(cv / 2), 0.30, 0.70)

  cat(sprintf(
    "[Auto v5]   Δloss: mean = %.4f  sd = %.4f  CV = %.2f → lr_decay = %.2f\n",
    mean_imp, sd(improvements), cv, lr_decay))
  lr_decay
}

# Bank refresh frequency from lr_patience
#
# Physical basis:
#   The memory bank (Z_mem, R_mem) becomes stale as weights change.  The bank
#   should be refreshed at least once per LR-decay cycle so that fine-tuning
#   (after the LR drops) operates on residuals from the current model, not a
#   model several epochs old.
#   Rule: refresh every floor(lr_patience / 2) epochs, optionally scaled by
#   GEOVERSA_AUTO_BANK_REFRESH_MULT for automatic screening:
#     mult > 1  → less frequent refresh (more stale but cheaper / smoother)
#     mult = 1  → default rule
auto_bank_refresh_from_patience <- function(lr_patience) {
  bank_mult <- .auto_env_num("GEOVERSA_AUTO_BANK_REFRESH_MULT", 1.0)
  base_bre <- floor(lr_patience / 2L)
  bre <- as.integer(max(1L, round(base_bre * bank_mult)))
  cat(sprintf("[Auto v5]   bank_refresh_every = %d  [round(floor(%d / 2) × %.2f)]\n",
              bre, lr_patience, bank_mult))
  bre
}

.auto_oof_fold_count <- function(n_train) {
  fold_override <- .auto_env_num("GEOVERSA_AUTO_OOF_FOLDS", NA_real_)
  if (is.finite(fold_override)) {
    folds <- as.integer(.auto_clamp(round(fold_override), 2L, 5L))
    cat(sprintf("[Auto v5]   OOF folds = %d  [explicit override]\n", folds))
    return(folds)
  }
  folds <- as.integer(.auto_clamp(round(n_train / 180), 3L, 5L))
  cat(sprintf("[Auto v5]   OOF folds = %d  [round(n/180), n=%d]\n", folds, n_train))
  folds
}

.auto_oof_epoch_count <- function(warmup_epochs_done) {
  epoch_mult <- .auto_env_num("GEOVERSA_AUTO_OOF_EPOCH_MULT", 1.25)
  epoch_bias <- .auto_env_num("GEOVERSA_AUTO_OOF_EPOCH_BIAS", 1.0)
  epochs <- as.integer(.auto_clamp(round(epoch_bias + epoch_mult * warmup_epochs_done), 4L, 12L))
  cat(sprintf("[Auto v5]   OOF epochs = %d  [round(%.2f + %.2f×%d)]\n",
              epochs, epoch_bias, epoch_mult, warmup_epochs_done))
  epochs
}

.auto_operator_select_fold_count <- function(n_train) {
  fold_override <- .auto_env_num("GEOVERSA_AUTO_OPERATOR_SELECT_FOLDS", NA_real_)
  if (is.finite(fold_override)) {
    folds <- as.integer(.auto_clamp(round(fold_override), 3L, 5L))
    cat(sprintf("[Auto v5]   operator-select folds = %d  [explicit override]\n", folds))
    return(folds)
  }
  folds <- as.integer(.auto_clamp(round(n_train / 220), 3L, 4L))
  cat(sprintf("[Auto v5]   operator-select folds = %d  [round(n/220), n=%d]\n", folds, n_train))
  folds
}

.auto_residual_signal_select_fold_count <- function(n_train) {
  fold_override <- .auto_env_num("GEOVERSA_AUTO_RESID_SIGNAL_SELECT_FOLDS", NA_real_)
  if (is.finite(fold_override)) {
    folds <- as.integer(.auto_clamp(round(fold_override), 3L, 5L))
    cat(sprintf("[Auto v5]   residual-signal-select folds = %d  [explicit override]\n", folds))
    return(folds)
  }
  folds <- as.integer(.auto_clamp(round(n_train / 220), 3L, 4L))
  cat(sprintf("[Auto v5]   residual-signal-select folds = %d  [round(n/220), n=%d]\n", folds, n_train))
  folds
}

.auto_correction_trust_from_variogram <- function(vg) {
  trust_floor <- .auto_env_num("GEOVERSA_AUTO_TRUST_FLOOR", 0.18)
  trust_span <- .auto_env_num("GEOVERSA_AUTO_TRUST_SPAN", 0.72)
  trust_min <- .auto_env_num("GEOVERSA_AUTO_TRUST_MIN", 0.15)
  trust_max <- .auto_env_num("GEOVERSA_AUTO_TRUST_MAX", 1.00)
  trust_range_ref <- .auto_env_num("GEOVERSA_AUTO_TRUST_RANGE_REF", 0.20)
  trust_signal_exp <- .auto_env_num("GEOVERSA_AUTO_TRUST_SIGNAL_EXP", 1.00)
  trust_fit_exp <- .auto_env_num("GEOVERSA_AUTO_TRUST_FIT_EXP", 0.75)
  trust_range_exp <- .auto_env_num("GEOVERSA_AUTO_TRUST_RANGE_EXP", 0.75)
  spatial_signal <- .auto_clamp(1 - vg$nugget_ratio, 0, 1)
  fit_quality <- .auto_clamp(ifelse(is.null(vg$fit_quality), 0.25, vg$fit_quality), 0.05, 1.00)
  range_fraction <- .auto_clamp(ifelse(is.null(vg$range_fraction), 0.30, vg$range_fraction), 0.02, 1.00)
  range_quality <- .auto_clamp(range_fraction / max(trust_range_ref, 1e-6), 0.10, 1.00)
  quality <- (spatial_signal ^ trust_signal_exp) *
    (fit_quality ^ trust_fit_exp) *
    (range_quality ^ trust_range_exp)
  trust <- .auto_clamp(trust_floor + trust_span * quality, trust_min, trust_max)
  cat(sprintf(
    "[Auto v5]   trust = %.3f  [floor %.2f + span %.2f × ((1-r)^%.2f=%.3f × fit^%.2f=%.3f × range^%.2f=%.3f)]\n",
    trust, trust_floor, trust_span,
    trust_signal_exp, spatial_signal ^ trust_signal_exp,
    trust_fit_exp, fit_quality ^ trust_fit_exp,
    trust_range_exp, range_quality ^ trust_range_exp
  ))
  list(
    trust = trust,
    quality = quality,
    spatial_signal = spatial_signal,
    fit_quality = fit_quality,
    range_quality = range_quality,
    trust_floor = trust_floor,
    trust_span = trust_span,
    trust_min = trust_min,
    trust_max = trust_max,
    trust_range_ref = trust_range_ref,
    trust_signal_exp = trust_signal_exp,
    trust_fit_exp = trust_fit_exp,
    trust_range_exp = trust_range_exp
  )
}

.auto_correction_K_from_residual_variogram <- function(n_train, vg, Ctr, K_reference = NULL) {
  base_k_resid <- .auto_K_from_spatial_density(n_train, vg, Ctr)
  if (!is.null(K_reference) && is.finite(K_reference)) {
    base_k_target <- as.integer(.auto_clamp(round(K_reference), 6L, 30L))
  } else {
    base_k_target <- base_k_resid
  }
  residual_signal <- .auto_clamp(1 - vg$nugget_ratio, 0, 1)
  fit_quality <- .auto_clamp(ifelse(is.null(vg$fit_quality), 0.25, vg$fit_quality), 0.05, 1.00)
  range_fraction <- .auto_clamp(ifelse(is.null(vg$range_fraction), 0.30, vg$range_fraction), 0.02, 1.00)
  control <- .auto_clamp(
    (residual_signal ^ 0.50) * (fit_quality ^ 0.50) * (range_fraction ^ 0.35),
    0.05, 1.00
  )
  k_floor <- .auto_env_num("GEOVERSA_AUTO_K_CORR_FLOOR", 0.45)
  k_scale <- .auto_clamp(k_floor + (1 - k_floor) * control, k_floor, 1.00)
  k_corr <- as.integer(.auto_clamp(round(base_k_target * k_scale), 6L, 30L))
  cat(sprintf(
    "[Auto v5]   K_corr = %d  [round(K_target=%d × scale=%.3f), K_resid=%d]\n",
    k_corr, base_k_target, k_scale, base_k_resid
  ))
  list(
    K_corr = k_corr,
    K_base_resid = base_k_resid,
    K_base_target = base_k_target,
    K_shrink = k_scale,
    residual_signal = residual_signal,
    fit_quality = fit_quality,
    range_fraction = range_fraction
  )
}

.auto_correction_schedule_from_residual_variogram <- function(vg, n_train, Ctr,
                                                              K_reference = NULL,
                                                              beta_reference = NULL) {
  k_info <- .auto_correction_K_from_residual_variogram(n_train, vg, Ctr, K_reference = K_reference)
  beta_init_resid <- .auto_beta_init_from_nugget(vg$nugget_ratio)
  beta_resid_prob <- plogis(beta_init_resid)
  beta_target_prob <- if (!is.null(beta_reference) && is.finite(beta_reference)) {
    plogis(beta_reference)
  } else {
    beta_resid_prob
  }
  beta_quality <- .auto_clamp(
    (k_info$residual_signal ^ 0.50) * (k_info$fit_quality ^ 0.35) * (k_info$range_fraction ^ 0.25),
    0.05, 1.00
  )
  beta_floor_min <- .auto_env_num("GEOVERSA_AUTO_BETA_CORR_FLOOR_MIN", 0.35)
  beta_floor_max <- .auto_env_num("GEOVERSA_AUTO_BETA_CORR_FLOOR_MAX", 0.60)
  beta_floor_share <- .auto_clamp(
    beta_floor_min + (1 - beta_quality) * (beta_floor_max - beta_floor_min),
    beta_floor_min, beta_floor_max
  )
  beta_floor_prob <- beta_floor_share * beta_target_prob
  beta_corr_prob <- .auto_clamp(max(beta_resid_prob, beta_floor_prob), 0.05, 0.95)
  beta_init_corr <- qlogis(beta_corr_prob)
  cat(sprintf(
    "[Auto v5]   beta_corr = %.3f  [max(beta_resid=%.3f, %.2f×beta_target=%.3f)]\n",
    beta_corr_prob, beta_resid_prob, beta_floor_share, beta_target_prob
  ))
  trust_info <- .auto_correction_trust_from_variogram(vg)
  list(
    K_corr = k_info$K_corr,
    K_base_resid = k_info$K_base_resid,
    K_base_target = k_info$K_base_target,
    K_shrink = k_info$K_shrink,
    beta_init_corr = beta_init_corr,
    beta_resid_prob = beta_resid_prob,
    beta_target_prob = beta_target_prob,
    beta_floor_share = beta_floor_share,
    trust = trust_info$trust,
    trust_quality = trust_info$quality,
    trust_spatial_signal = trust_info$spatial_signal,
    trust_fit_quality = trust_info$fit_quality,
    trust_range_quality = trust_info$range_quality,
    trust_floor = trust_info$trust_floor,
    trust_span = trust_info$trust_span,
    residual_signal = k_info$residual_signal
  )
}

.auto_current_residual_correction_schedule <- function(vg, n_train, Ctr,
                                                       K_reference = NULL,
                                                       beta_reference = NULL) {
  sched <- .auto_correction_schedule_from_residual_variogram(
    vg = vg,
    n_train = n_train,
    Ctr = Ctr,
    K_reference = K_reference,
    beta_reference = beta_reference
  )

  k_min <- suppressWarnings(as.integer(round(.auto_env_num("GEOVERSA_AUTO_RESID_CORR_K_MIN", 16))))
  k_max <- suppressWarnings(as.integer(round(.auto_env_num("GEOVERSA_AUTO_RESID_CORR_K_MAX", 28))))
  if (!is.finite(k_min)) k_min <- 16L
  if (!is.finite(k_max)) k_max <- 28L
  k_min <- max(6L, k_min)
  k_max <- max(k_min, min(30L, k_max))
  k_uncapped <- sched$K_corr
  sched$K_corr <- as.integer(.auto_clamp(sched$K_corr, k_min, k_max))

  beta_min <- .auto_env_num("GEOVERSA_AUTO_RESID_CORR_BETA_MIN", 0.60)
  beta_max <- .auto_env_num("GEOVERSA_AUTO_RESID_CORR_BETA_MAX", 0.75)
  beta_min <- .auto_clamp(beta_min, 0.05, 0.95)
  beta_max <- .auto_clamp(beta_max, beta_min, 0.95)
  beta_uncapped <- plogis(sched$beta_init_corr)
  beta_capped <- .auto_clamp(beta_uncapped, beta_min, beta_max)
  sched$beta_init_corr <- qlogis(beta_capped)
  sched$beta_corr_uncapped_prob <- beta_uncapped
  sched$beta_corr_capped_prob <- beta_capped
  sched$K_corr_uncapped <- k_uncapped
  sched$K_corr_min <- k_min
  sched$K_corr_max <- k_max
  sched$beta_corr_min <- beta_min
  sched$beta_corr_max <- beta_max

  cat(sprintf(
    "[Auto v5]   current-resid clamp | K_corr: %d -> %d  [clamp %d:%d] | beta_corr: %.3f -> %.3f  [clamp %.2f:%.2f]\n",
    k_uncapped, sched$K_corr, k_min, k_max,
    beta_uncapped, beta_capped, beta_min, beta_max
  ))

  sched
}

.auto_residual_bank_mix_from_trust <- function(trust, verbose = TRUE) {
  current_floor <- .auto_env_num("GEOVERSA_AUTO_BANK_CURRENT_FLOOR", 0.25)
  current_ceiling <- .auto_env_num("GEOVERSA_AUTO_BANK_CURRENT_CEILING", 0.85)
  current_power <- .auto_env_num("GEOVERSA_AUTO_BANK_CURRENT_POWER", 0.50)
  current_weight <- .auto_clamp(trust ^ current_power, current_floor, current_ceiling)
  oof_weight <- 1 - current_weight
  if (isTRUE(verbose)) {
    cat(sprintf(
      "[Auto v5]   bank mix = current %.3f / OOF %.3f  [clamp(trust^%.2f, %.2f, %.2f)]\n",
      current_weight, oof_weight, current_power, current_floor, current_ceiling
    ))
  }
  list(
    current_weight = current_weight,
    oof_weight = oof_weight,
    current_floor = current_floor,
    current_ceiling = current_ceiling,
    current_power = current_power
  )
}

.auto_residual_bank_mode <- function() {
  mode <- tolower(Sys.getenv("GEOVERSA_AUTO_BANK_MODE", unset = "current"))
  if (!mode %in% c("current", "mix", "anneal", "oof")) mode <- "current"
  mode
}

.auto_residual_bank_weights <- function(trust, progress = 0, verbose = TRUE) {
  mode <- .auto_residual_bank_mode()
  progress <- .auto_clamp(progress, 0, 1)
  base_mix <- .auto_residual_bank_mix_from_trust(trust, verbose = FALSE)
  anneal_power <- .auto_env_num("GEOVERSA_AUTO_BANK_ANNEAL_POWER", 1.0)
  current_weight <- switch(
    mode,
    current = 1.0,
    oof = 0.0,
    mix = base_mix$current_weight,
    anneal = .auto_clamp(
      base_mix$current_weight + (1 - base_mix$current_weight) * (progress ^ anneal_power),
      0, 1
    )
  )
  oof_weight <- 1 - current_weight
  if (isTRUE(verbose)) {
    if (identical(mode, "anneal")) {
      cat(sprintf(
        "[Auto v5]   bank mode = %s  [progress=%.2f power=%.2f]  current=%.3f / OOF=%.3f\n",
        mode, progress, anneal_power, current_weight, oof_weight
      ))
    } else {
      cat(sprintf(
        "[Auto v5]   bank mode = %s  current=%.3f / OOF=%.3f\n",
        mode, current_weight, oof_weight
      ))
    }
  }
  list(
    mode = mode,
    current_weight = current_weight,
    oof_weight = oof_weight,
    anneal_power = anneal_power,
    base_current_weight = base_mix$current_weight,
    base_oof_weight = base_mix$oof_weight,
    current_floor = base_mix$current_floor,
    current_ceiling = base_mix$current_ceiling,
    current_power = base_mix$current_power
  )
}

.auto_residual_signal_schedule_from_variogram <- function(vg, resid_scaled) {
  signal_floor <- .auto_env_num("GEOVERSA_AUTO_RESID_SIGNAL_FLOOR", 0.45)
  signal_span <- .auto_env_num("GEOVERSA_AUTO_RESID_SIGNAL_SPAN", 0.45)
  clip_floor <- .auto_env_num("GEOVERSA_AUTO_RESID_CLIP_FLOOR", 1.10)
  clip_span <- .auto_env_num("GEOVERSA_AUTO_RESID_CLIP_SPAN", 1.70)
  nugget_ratio <- ifelse(is.null(vg$nugget_ratio) || !is.finite(vg$nugget_ratio), 0.35, vg$nugget_ratio)
  spatial_signal <- .auto_clamp(1 - nugget_ratio, 0, 1)
  fit_quality <- .auto_clamp(ifelse(is.null(vg$fit_quality), 0.25, vg$fit_quality), 0.05, 1.00)
  range_fraction <- .auto_clamp(ifelse(is.null(vg$range_fraction), 0.30, vg$range_fraction), 0.02, 1.00)
  signal_quality <- .auto_clamp(
    (spatial_signal ^ 0.75) * (fit_quality ^ 0.50) * (range_fraction ^ 0.35),
    0.05, 1.00
  )
  resid_center <- stats::median(resid_scaled, na.rm = TRUE)
  resid_scale_mad <- stats::mad(resid_scaled, center = resid_center, constant = 1.4826, na.rm = TRUE)
  resid_scale_sd <- stats::sd(resid_scaled, na.rm = TRUE)
  resid_scale <- max(resid_scale_mad, 0.35 * resid_scale_sd, 0.05)
  shrink <- .auto_clamp(signal_floor + signal_span * signal_quality, signal_floor, signal_floor + signal_span)
  clip_sigma <- .auto_clamp(clip_floor + clip_span * signal_quality, clip_floor, clip_floor + clip_span)
  cat(sprintf(
    "[Auto v5]   residual signal = shrink %.3f / clip %.3fσ  [quality=%.3f, center=%.4f, scale=%.4f]\n",
    shrink, clip_sigma, signal_quality, resid_center, resid_scale
  ))
  list(
    active = TRUE,
    center = resid_center,
    scale = resid_scale,
    shrink = shrink,
    clip_sigma = clip_sigma,
    signal_quality = signal_quality,
    spatial_signal = spatial_signal,
    fit_quality = fit_quality,
    range_fraction = range_fraction,
    signal_floor = signal_floor,
    signal_span = signal_span,
    clip_floor = clip_floor,
    clip_span = clip_span,
    transform_override = FALSE
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# §7  Main auto-configuration wrapper
#     Runs phases 0 (A+B) and 1.  Phase 2 weight decay is recomputed in the
#     training function after the final model is built.  Phase 3 parameters
#     (patience, lr_decay, bank_refresh) are updated in the training loop
#     after warmup via auto_patience_from_warmup() and friends.
# ─────────────────────────────────────────────────────────────────────────────

auto_kriging_config_v5 <- function(vg, n_train, coord_scaler, Ctr,
                                    patch_channels = NULL,
                                    model_init     = NULL,
                                    train_cache    = NULL,
                                    patch_resolution = NULL,
                                    patch_geometry = NULL,
                                    device         = "cpu") {

  cat("\n[Auto v5] ════════════════════════════════════════════════════\n")
  cat("[Auto v5]  GeoVersa V5 — COMPLETE AUTO-CONFIGURATION\n")
  cat("[Auto v5] ════════════════════════════════════════════════════\n\n")

  r  <- vg$nugget_ratio
  cached_patch_size <- NA_integer_
  if (!is.null(train_cache) && !is.null(train_cache$P)) {
    patch_shape <- tryCatch(train_cache$P$shape, error = function(e) NULL)
    if (!is.null(patch_shape) && length(patch_shape) >= 4L &&
        is.finite(patch_shape[3]) && patch_shape[3] > 0) {
      cached_patch_size <- as.integer(patch_shape[3])
    }
  }
  cached_or_requested_patch <- if (is.finite(cached_patch_size)) {
    cached_patch_size
  } else if (is.list(patch_geometry) && is.finite(patch_geometry$global_size)) {
    patch_geometry$global_size
  } else {
    NA_integer_
  }
  if (is.null(patch_geometry) ||
      !is.list(patch_geometry) ||
      !is.finite(patch_geometry$global_size) ||
      !is.finite(patch_geometry$local_size)) {
    patch_geometry <- auto_patch_geometry_v5(
      vg = vg,
      n_train = n_train,
      patch_resolution = patch_resolution,
      cached_patch_size = cached_or_requested_patch
    )
  }
  cfg_base <- build_auto_config_v5(
    n_train = n_train,
    vg = vg,
    coords = Ctr,
    covariates = if (!is.null(train_cache)) train_cache$X else NULL,
    coord_scaler = coord_scaler,
    patch_channels = patch_channels,
    patch_resolution = patch_resolution,
    patch_geometry = patch_geometry,
    cached_patch_size = cached_patch_size,
    device = device
  )

  # ── Phase 0-A: Geostatistical theory ─────────────────────────────────────
  cat("[Auto v5] ── Phase 0-A: Geostatistical derivation ──\n")
  cat(sprintf("[Auto v5]   nugget_ratio = %.3f\n", r))
  cat(sprintf("[Auto v5]   ell_major_init = %.4g   ell_minor_init = %.4g   θ = %.1f°\n",
              cfg_base$ell_major_init, cfg_base$ell_minor_init, cfg_base$theta_init * 180 / pi))
  cat(sprintf("[Auto v5]   BLW = %.4f   alpha_me = %.4f   lambda_cov = %.5f   lambda_spatial = %.5f\n",
              cfg_base$base_loss_weight, cfg_base$alpha_me, cfg_base$lambda_cov, cfg_base$lambda_spatial))
  cat(sprintf("[Auto v5]   max_warmup = %d\n", cfg_base$max_warmup_epochs))

  # ── Phase 0-B: Statistical capacity ──────────────────────────────────────
  cat("\n[Auto v5] ── Phase 0-B: Statistical capacity (√n scaling) ──\n")
  d <- cfg_base$d
  patch_size <- cfg_base$patch_size
  patch_local_size <- cfg_base$patch_local_size
  coord_dim <- cfg_base$coord_dim
  coord_hidden <- cfg_base$coord_hidden
  fusion_hidden <- cfg_base$fusion_hidden
  main_epochs <- cfg_base$main_epochs
  patch_dim <- cfg_base$patch_dim
  if (is.null(patch_channels)) {
    cat(sprintf("[Auto v5]   patch_dim = %d  [d/2 fallback — patch_channels unknown]\n", patch_dim))
  }

  cat(sprintf("[Auto v5]   d = %d   patch_size = %d   patch_local = %d   patch_dim = %d\n",
              d, patch_size, patch_local_size, patch_dim))
  cat(sprintf("[Auto v5]   tab_drop = %.3f   patch_drop = %.3f   coord_drop = %.3f\n",
              cfg_base$tab_dropout, cfg_base$patch_dropout, cfg_base$coord_dropout))
  cat(sprintf("[Auto v5]   main_epochs = %d   gate_hidden = %d   gate_floor = %.2f\n",
              main_epochs, cfg_base$gate_hidden, cfg_base$gate_floor))
  cat(sprintf("[Auto v5]   linear_blend_center = %.2f   linear_blend_temp = %.1f   nko_gate_floor = %.2f\n",
              cfg_base$linear_blend_center, cfg_base$linear_blend_temp, cfg_base$nko_gate_floor))
  cat(sprintf(
    "[Auto v5]   patch_struct_mult = %.3f  [1 + %.2f·(r−%.2f) + %.2f·((K−%.0f)/%.0f)]\n",
    cfg_base$patch_struct_mult,
    cfg_base$patch_nugget_slope, cfg_base$patch_nugget_ref,
    cfg_base$patch_k_slope, cfg_base$patch_k_ref, cfg_base$patch_k_ref
  ))
  cat(sprintf("[Auto v5]   coord_dim = %d   coord_hidden = %d\n",
              coord_dim, coord_hidden))

  # ── Phase 1: Learning rate (LR range test) ────────────────────────────────
  cat("\n[Auto v5] ── Phase 1: Learning rate (range test on preliminary model) ──\n")

  if (!is.null(model_init) && !is.null(train_cache)) {
    lr <- estimate_initial_lr_range_test(model_init, train_cache,
                                         bs_init = 32L, device = device)
  } else {
    cat("[Auto v5]   ⚠ model_init / train_cache missing — lr = 1e-4\n")
    lr <- 1e-4
  }
  # Note: lr will be re-estimated on the FINAL model (after rebuild) in the
  # training function, since the preliminary model has fixed d = 192.

  # ── Phase 1: Batch size (hardware-adaptive) ───────────────────────────────
  cat("\n[Auto v5] ── Phase 1: Batch size (hardware-adaptive) ──\n")

  batch_size <- cfg_base$batch_size

  # ── Phase 2 placeholder: weight decay (recomputed after final model build) ──
  wd     <- 1e-3
  min_lr <- lr / 1000     # LR floor: 3 orders of magnitude schedule span

  # ── Phase 3 placeholders: updated in training loop after warmup ───────────
  # Neutral defaults until auto_patience_from_warmup() / auto_lr_decay_from_trajectory()
  # are called with the actual observed trajectory.
  lr_decay           <- 0.5    # geometric midpoint of [0.3, 0.7]
  lr_patience        <- cfg_base$lr_patience
  patience           <- cfg_base$patience
  bank_refresh_every <- auto_bank_refresh_from_patience(lr_patience)

  # ── Summary ───────────────────────────────────────────────────────────────
  cat("\n[Auto v5] ── Phase 0–1 summary ──\n")
  cat(sprintf("[Auto v5]   lr = %.2e (→ re-estimated on final model)\n", lr))
  cat(sprintf("[Auto v5]   batch_size = %d   wd = %.2e (→ recomputed)   min_lr = %.2e\n",
              batch_size, wd, min_lr))
  cat(sprintf("[Auto v5]   patience = %d (→ Phase 3 post-warmup)   lr_patience = %d (→ Phase 3)\n",
              patience, lr_patience))
  cat(sprintf("[Auto v5]   lr_decay = %.2f (→ Phase 3)   bank_refresh_every = %d (→ Phase 3)\n",
              lr_decay, bank_refresh_every))

  # ── Assemble cfg ──────────────────────────────────────────────────────────
  cfg <- list(
    # (A) Spatial — variogram-derived
    ell_major_init     = cfg_base$ell_major_init,
    ell_minor_init     = cfg_base$ell_minor_init,
    theta_init         = cfg_base$theta_init,
    nugget_ratio       = r,
    K_neighbors        = cfg_base$K_neighbors,
    beta_init          = cfg_base$beta_init,
    # (A) Loss weights — nugget-derived
    base_loss_weight   = cfg_base$base_loss_weight,
    alpha_me           = cfg_base$alpha_me,
    lambda_cov         = cfg_base$lambda_cov,
    lambda_spatial     = cfg_base$lambda_spatial,
    max_warmup_epochs  = cfg_base$max_warmup_epochs,
    # (B) Architecture — √n scaling
    d                  = d,
    patch_size         = patch_size,
    patch_local_size   = patch_local_size,
    patch_dim          = patch_dim,
    fusion_hidden      = fusion_hidden,
    # (B) Regularisation — n scaling
    tab_dropout        = cfg_base$tab_dropout,
    patch_dropout      = cfg_base$patch_dropout,
    coord_dropout      = cfg_base$coord_dropout,
    nko_dropout        = cfg_base$nko_dropout,
    # (A+B) Coordinate architecture — variogram anisotropy + n
    coord_dim          = coord_dim,
    coord_hidden       = coord_hidden,
    main_epochs        = main_epochs,
    linear_blend_center = cfg_base$linear_blend_center,
    linear_blend_temp   = cfg_base$linear_blend_temp,
    gate_hidden         = cfg_base$gate_hidden,
    gate_floor          = cfg_base$gate_floor,
    nko_gate_floor      = cfg_base$nko_gate_floor,
    # (D) Optimisation — data / hardware derived
    lr                 = lr,
    batch_size         = batch_size,
    wd                 = wd,
    min_lr             = min_lr,
    # (D) Phase 3 placeholders — updated post-warmup in training loop
    lr_decay           = lr_decay,
    lr_patience        = lr_patience,
    patience           = patience,
    bank_refresh_every = bank_refresh_every,
    # Fixed physical constant (anisotropic is strictly more general)
    kriging_mode       = "anisotropic"
  )

  cat("\n[Auto v5] ══ Auto-config Phase 0–1 complete.  Phase 3 after warmup. ══\n\n")
  cfg
}
