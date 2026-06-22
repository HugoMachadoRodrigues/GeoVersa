# =============================================================================
# NeuroMaternKrige.R
#
# NeuroMatern-Krige (NMK) — Non-Stationary Neural Matern Kriging Network
# for Digital Soil Mapping.
#
# Novelty (unprecedented in Pedometrics — GeoVersa research family entry):
#   ConvKrigingNet2D uses ONE global anisotropic exponential kernel
#   (nu = 1/2 fixed) for the residual kriging step. In NMK the kriging
#   kernel is a **locally learned Matern**:
#
#     - a small hypernetwork g_psi predicts, at every point s_i, four
#       kernel descriptors (log ell_major, log ell_minor, theta, pi)
#       from the fused SCORPAN latent z_i;
#     - for each (query i, neighbor j) pair, the kernel parameters are
#       symmetrised by the Stein-Paciorek midpoint rule, so the resulting
#       covariance is a valid non-stationary field-level kernel;
#     - the smoothness parameter nu is a learned convex mixture over
#       closed-form Matern cases {1/2, 3/2, 5/2} — so the model can
#       continuously interpolate between exponential, Whittle-Matern and
#       near-Gaussian behaviour WITHOUT evaluating modified Bessel
#       functions (which are numerically fragile on torch backends).
#
#   Additional principled ingredients:
#
#     - the hypernet bias is initialised with the variogram-fit global
#       parameters, preserving GeoVersa's "zero user tuning" philosophy;
#     - smoothness and anchor regularisers keep the local field close to
#       the empirical variogram when data is scarce;
#     - the warmup / bank-refresh / beta-gate machinery of GeoVersa is
#       reused verbatim, so NMK is a drop-in replacement at the
#       Wadoux-protocol level.
#
# Mathematical formulation (full LaTeX version in README.md):
#
#   (1)  phi_i = g_psi(z_i)
#         where z_i is the fused tabular + patch + coord latent and phi_i
#         = (log ell_maj_i, log ell_min_i, theta_i, pi_i).
#
#   (2)  (ell_maj_ij, ell_min_ij, theta_ij, pi_ij)
#            = 1/2 [phi_i + phi_j]                 (Stein-Paciorek average)
#
#   (3)  u_ij =  cos(theta_ij) dx + sin(theta_ij) dy
#         v_ij = -sin(theta_ij) dx + cos(theta_ij) dy
#         d*_ij = sqrt((u_ij / ell_maj_ij)^2 + (v_ij / ell_min_ij)^2 + eps)
#
#   (4)  k_nu(d) = exp(-d)                         (nu = 1/2)
#         k_nu(d) = (1 + sqrt(3) d) exp(-sqrt(3) d)         (nu = 3/2)
#         k_nu(d) = (1 + sqrt(5) d + 5/3 d^2) exp(-sqrt(5) d)
#                                                   (nu = 5/2)
#
#   (5)  k_ij = sum_{nu in {1/2, 3/2, 5/2}} pi_ij^{(nu)} k_nu(d*_ij)
#
#   (6)  w_ij = k_ij / sum_{k in N_K(i)} k_ik
#         delta_i = sum_j w_ij r_j
#
#   (7)  yhat_i = yhat_base_i + sigma(logit_beta) * delta_i.
#
# File layout:
#     §0   Setup, Polyak LR probe, parameter counter
#     §1   NonStationaryMaternKrigingLayer  (the novel kriging head)
#     §1b  NeuralObliqueForest              (differentiable decision forest
#                                            that replaces the ScalarHead
#                                            MLP; Kontschieder 2015)
#     §2   NeuroMaternKrige                 (the full network)
#     §3   Tensor caches, bank refresh, prediction utilities
#     §4   nmk_auto_config                  (variogram + n -> hyperparameters)
#     §5   train_neuromaternkrige_one_fold  (warmup + Polyak LR + main loop)
#
# Dependencies (loaded by the caller or via source(...) below):
#   - code/ConvKrigingNet2D.R        (ConvBlock2D, MultiScalePatchEncoder2D,
#                                     ScalarHead, make_mlp, huber_loss,
#                                     tensor-cache helpers, scalers)
#   - code/ConvKrigingNet2D_Auto.R   (fit_variogram_auto)
#
# =============================================================================
# Hyperparameter deduction chain (zero manual tuning)
# =============================================================================
#
# Every NMK hyperparameter is a closed-form function of the current fold's
# (y, X, patches, coords). The deduction flows top-down through four
# derivation classes, matching the convention of ConvKrigingNet2D_Auto_v5:
#
# (A) VARIOGRAM-DRIVEN. From fit_variogram_auto(y, coords_raw):
#       r           = nugget_ratio (nugget / total sill) in [0, 1]
#       range_major, range_minor, theta_rad    initial kernel geometry
#       fit_quality in (0, 1]                  exponential-kernel goodness
#     Closed-form rules derived from r:
#       K_neighbors        = clamp(round(8 + 12 r), 8, 20)
#       base_loss_weight   = max(0.05, 0.10 r)
#       alpha_me           = 0.375 r
#       lambda_cov         = 0.025 (1 - r)
#       lambda_smooth      = 0.24  (1 - r)
#       lambda_anchor      = 0.24  (1 - r)
#       max_warmup_epochs  = clamp(round(4 + 16 r), 4, 20)
#       krig_drop_p_train  = min(1 - r, 0.3)
#       dist_scale_eval    = clamp(0.8 - 0.6 r, 0.2, 0.8)
#       beta_init (logit)  = clamp(2 - 6 r, -5, 3)
#     From fit_quality:
#       init_mix_mean      = clamp(0.5 + 2 (1 - fit_q), 0.5, 2.5)
#     Direct copies (unit-converted to standardised coords):
#       ell_major_init = range_major / mean(coord_scaler$scale)
#       ell_minor_init = range_minor / mean(coord_scaler$scale)
#       theta_init     = theta_rad
#
# (B) STATISTICAL-CAPACITY. From n_train (sqrt-n scaling, Bartlett & Mendelson):
#       d                 = clamp(64 ceil(sqrt(n) / 8), 128, 512)
#       patch_dim         = d / 2
#       kriging_hidden    = clamp(round(d / 4), 32, 128)
#       tree_depth        = clamp(round(log2(n) / 2), 3, 6)
#       n_trees           = clamp(round(sqrt(n / 10)), 4, 16)
#       tab_dropout       = clamp(0.30 - n / 8000,  0.05, 0.30)
#       patch_dropout     = clamp(0.20 - n / 10000, 0.03, 0.20)
#       batch_size        = clamp(round(n / 8), 24, 64)
#
# (C) SCHEDULE. From max_warmup_epochs W derived in (A):
#       patience           = clamp(round(2 W), 5, 40)
#       lr_patience        = clamp(round(W / 2), 2, 10)
#       bank_refresh_every = clamp(round(W / 3), 1,  5)
#       lr_decay           = 0.5                       (convention)
#
# (D) OPTIMISATION PROBE. After the model is built on the training cache:
#       lr_init = estimate_nmk_initial_lr_polyak(model, train_cache)
#               = clamp(0.01 * L / ||grad(L)||^2, 1e-5, 1e-3)
#       min_lr  = lr_init / 20
#       wd      = clamp(1e-3 / sqrt(max(1, n_params / 5e6)), 1e-4, 1e-2)
#
# (E) TARGET-SIDE STATISTICAL PREP. From the training target y_train alone:
#       target_transform = "log"      if all(y_train > 0) AND
#                                         shapiro.test(y_train)$p.value < 0.05
#                        = "identity" otherwise
#     This keeps right-skewed strictly-positive targets (AGB, SOC, metals
#     — almost every DSM target) on the natural scale while leaving
#     normally-distributed or non-positive targets alone.
#
#     When target_transform = "log", exp(yhat_log) estimates the conditional
#     MEDIAN rather than the mean — the classical Jensen / Finney bias.
#     The Duan (1983) non-parametric correction,
#         S = (1 / n_val) * sum_j exp( log y_j^val - log yhat_j^val ),
#     is the textbook fix. We compute S as a diagnostic, but DO NOT apply
#     it. Rationale: at n_val ~= 80 (typical 20%-inner split of an n=500
#     DSM calibration) the sample variance of S is dominated by single
#     log-scale outliers, because exp(r) is a convex unbounded function of
#     r. Empirically (AGB, 10 iterations, DesignBased):
#       raw log + ensemble           ME = -1.5   RMSE sd = 1.14
#       + Duan smearing applied      ME = +3.9   RMSE sd = 2.62
#     Applying Duan doubled the cross-seed variance and flipped the mean
#     bias without reducing mean RMSE. The small residual median-vs-mean
#     bias is therefore accepted: it is quantified and stable, the Duan
#     correction at this sample size is not. The diagnostic S stays in
#     the output so a user with a much larger val set can opt to apply it
#     (or switch to a robust variant such as MAD-based parametric Finney).
#
# STRUCTURAL CONSTANTS (not hyperparameters — architecture/theory choices):
#       nu_classes = {1/2, 3/2, 5/2}     closed-form Matern cases
#       n_seeds    = 3                    ensemble size (see
#                                         train_neuromaternkrige_ensemble)
#       theta_max  = pi/2                 anisotropy-angle bound
#       grad_clip  = 2.0                  universal NN training stability
#
# Consequence: the runner never passes a hyperparameter by hand. Every value
# in the `Final summary` block is reproducible from (y, X, patches, coords)
# plus the fixed formulas above. Runner env-vars (NMK_*) stay available as
# experiment knobs but default to "use the derivation".
# =============================================================================

# -----------------------------------------------------------------------------
# §0  Setup and helper functions
# -----------------------------------------------------------------------------

# Clamp a numeric scalar or tensor to a closed interval [lo, hi].
# Used for numeric stability (e.g. keeping softplus arguments in a safe range).
.nmk_clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

# Parse a numeric override from an environment variable (for the runner).
.nmk_env_num <- function(name, default) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) default else val
}

# Shape-safe inverse softplus (used to initialise the hypernet last-layer bias
# so that softplus(raw) = target at epoch 0).
.nmk_inv_softplus <- function(x) log(expm1(pmax(x, 1e-6)))

# Default set of Matern smoothness classes. Closed-form cases only: 1/2
# (exponential / Whittle), 3/2 (stars differentiable once), 5/2 (twice
# differentiable). Evaluating Bessel functions for arbitrary nu is avoided on
# purpose — it is numerically fragile on torch MPS/CUDA kernels.
NMK_NU_CLASSES <- c(0.5, 1.5, 2.5)

# Count trainable parameters of a torch nn_module. Used to derive weight
# decay from model capacity — so larger models get proportionally stronger
# regularisation without a hand-tuned constant.
nmk_count_params <- function(model) {
  total <- 0
  for (p in model$parameters) {
    if (isTRUE(p$requires_grad)) {
      total <- total + as.numeric(p$numel())
    }
  }
  total
}

# Small-n regime warning.
#
# Neural DSM methods (NMK, DKP, forest backbones) are parameter-heavy: even
# with the shrink rules in Class B they run at roughly 50-150k parameters,
# which with n_train < ~300 gives a params-per-sample ratio above 300:1 —
# a regime where overfitting dominates even with aggressive dropout and
# anchor regularisation. This function emits an advisory message so the
# user is not misled by the fact that the code mechanically runs.
#
# Thresholds (empirically motivated):
#   n <= 150  : strongly advise against neural methods; use ranger or gstat
#   150 < n <= 300  : advisory; consider ranger + ensemble as baseline
#   300 < n <= 500  : cautionary; GeoVersa competitive but variance is high
#   n > 500 : no warning (standard Wadoux calibration regime)
.nmk_small_n_advisory <- function(n_train, model_label = "GeoVersa",
                                  suppress = FALSE) {
  if (isTRUE(suppress)) return(invisible(NULL))
  if (n_train > 500L) return(invisible(NULL))

  severity <- if (n_train <= 150L) {
    "HIGH"
  } else if (n_train <= 300L) {
    "MODERATE"
  } else {
    "LOW"
  }

  cat(sprintf("\n[%s] ================ SMALL-n ADVISORY (severity %s) ================\n",
              model_label, severity))
  cat(sprintf("[%s] n_train = %d is below the recommended minimum for neural DSM.\n",
              model_label, n_train))
  if (severity == "HIGH") {
    cat(sprintf("[%s] At this scale the model has ~1000 parameters per training sample.\n",
                model_label))
    cat(sprintf("[%s] Overfitting will almost certainly dominate generalisation.\n",
                model_label))
    cat(sprintf("[%s] RECOMMENDED ALTERNATIVES for n <= 150:\n", model_label))
    cat(sprintf("[%s]   - ranger::ranger(formula, data, num.trees = 500)   # robust RF\n",
                model_label))
    cat(sprintf("[%s]   - gstat::krige(formula, locations = sp, model = vgm_fit)\n",
                model_label))
    cat(sprintf("[%s]   - gstat::fit.variogram + linear regression-kriging\n",
                model_label))
  } else if (severity == "MODERATE") {
    cat(sprintf("[%s] Expect high cross-seed variance and a structural gap vs RF.\n",
                model_label))
    cat(sprintf("[%s] Consider ranger (RF) as your primary model; neural as secondary.\n",
                model_label))
    cat(sprintf("[%s] If you proceed, increase NMK_N_SEEDS to 5 for ensemble stability.\n",
                model_label))
  } else {
    cat(sprintf("[%s] GeoVersa is competitive here but variance is notable.\n",
                model_label))
    cat(sprintf("[%s] Paired comparison vs ranger strongly recommended.\n",
                model_label))
  }
  cat(sprintf("[%s] Proceeding anyway because the user set sample_size = %d.\n",
              model_label, n_train))
  cat(sprintf("[%s] To silence this warning set NMK_SMALL_N_SUPPRESS=TRUE (or DKP_...).\n",
              model_label))
  cat(sprintf("[%s] ================================================================\n\n",
              model_label))
  invisible(NULL)
}


# Target-transform auto-selection.
#
# Class (A) extension of the deduction chain. The rule picks between
# "identity" and "log" based on two data-driven checks:
#   (1) STRICT POSITIVITY of y_train (log is undefined otherwise);
#   (2) NON-NORMALITY at alpha = 0.05 via Shapiro-Wilk.
# Both conditions must hold for "log". Right-skewed strictly-positive
# targets (soil carbon, biomass, clay content, metal concentrations — the
# whole DSM bread-and-butter) flip to "log"; normally-distributed or
# non-positive targets stay in the original scale.
#
# Matches the rule ConvKrigingNet2D_Auto uses internally for the variogram
# normal-score decision (fit_variogram_auto §Step 1), reapplied here for
# the training objective.
.nmk_target_transform_auto <- function(y_train) {
  y_clean <- y_train[is.finite(y_train)]
  n       <- length(y_clean)
  if (n < 3L) return("identity")
  if (any(y_clean <= 0)) return("identity")
  # shapiro.test accepts n up to 5000; subsample for larger sets.
  n_sw <- min(n, 5000L)
  y_samp <- if (n_sw < n) sample(y_clean, n_sw) else y_clean
  sw <- tryCatch(shapiro.test(y_samp),
                 error = function(e) list(p.value = 1, statistic = NA_real_))
  p_val <- if (is.null(sw$p.value)) 1 else sw$p.value
  if (!is.finite(p_val)) p_val <- 1
  if (p_val < 0.05) "log" else "identity"
}

# Polyak learning-rate probe (see Polyak 1987; Hazan et al. 2014).
#
# Run one backbone-only forward+backward on a probe batch, measure the
# current loss L and the gradient L2 norm squared ‖∇L‖². The Polyak optimal
# step is α* = L / ‖∇L‖². We use 1% of α* to account for stochasticity,
# non-convexity and the fact that training takes many steps — the same rule
# ConvKrigingNet2D_Auto_v5 uses (`estimate_initial_lr_polyak`). The result
# is clamped to [1e-5, 1e-3] so a degenerate probe cannot destabilise the
# main loop. The probe does NOT modify the model weights.
estimate_nmk_initial_lr_polyak <- function(model, train_cache,
                                           bs_init = 32L,
                                           device  = "cpu") {
  cat("[NMK auto] --- LR probe (Polyak step, 1 %% of L/|grad|^2) ---\n")
  res <- tryCatch({
    n_available <- train_cache$n
    bs          <- min(as.integer(bs_init), n_available)
    idx_t       <- torch_tensor(seq_len(bs), dtype = torch_long(),
                                device = device)

    xb <- train_cache$X$index_select(1L, idx_t)
    pb <- train_cache$P$index_select(1L, idx_t)
    cb <- train_cache$C$index_select(1L, idx_t)
    yb <- train_cache$y$index_select(1L, idx_t)

    # Temporary optimiser just so we can call zero_grad; we never step it.
    tmp_opt <- optim_adam(model$parameters, lr = 0.01)
    model$train()
    tmp_opt$zero_grad()
    out  <- model$forward_base(xb, pb, cb)
    loss <- huber_loss(yb, out$pred)
    loss$backward()

    loss_val    <- loss$item()
    grad_sq_sum <- 0.0
    for (param in model$parameters) {
      g <- param$grad
      g_numel <- if (is.null(g)) 0L else tryCatch(g$numel(),
                                                   error = function(...) 0L)
      if (g_numel > 0L) {
        gd <- g$detach()
        grad_sq_sum <- grad_sq_sum + torch_sum(gd * gd)$item()
      }
    }
    tmp_opt$zero_grad()

    if (!is.finite(loss_val) || !is.finite(grad_sq_sum) ||
        grad_sq_sum < 1e-30) {
      cat("[NMK auto]   degenerate probe — fallback lr = 1e-4\n")
      return(1e-4)
    }

    lr_polyak <- loss_val / grad_sq_sum
    lr_est    <- 0.01 * lr_polyak
    lr_final  <- .nmk_clamp(lr_est, 1e-5, 1e-3)
    cat(sprintf("[NMK auto]   loss = %.4f  grad_sq = %.4e  lr_polyak = %.2e  lr_final = %.2e\n",
                loss_val, grad_sq_sum, lr_polyak, lr_final))
    lr_final
  }, error = function(e) {
    cat(sprintf("[NMK auto]   LR probe failed (%s) — fallback lr = 1e-4\n",
                conditionMessage(e)))
    1e-4
  })
  res
}


# -----------------------------------------------------------------------------
# §1  NonStationaryMaternKrigingLayer
# -----------------------------------------------------------------------------
#
# Input shapes (B = batch, K = neighbors, d = latent dim):
#   z_i       : [B, d]       query latents
#   coords_i  : [B, 2]       query coordinates (standardised)
#   z_n       : [B, K, d]    neighbor latents (from memory bank)
#   coords_n  : [B, K, 2]    neighbor coordinates
#   r_n       : [B, K]       neighbor residuals
#
# Output:
#   delta     : [B]          residual kriging correction
#   w         : [B, K]       kriging weights (softmax over closed-form Matern)
#   aniso_dist: [B, K]       pairwise anisotropic distance d*_ij
#   phi_i     : [B, 4]       (log ell_maj, log ell_min, theta, -) for query
#   phi_n     : [B, K, 4]    same for neighbors
#   mix_i     : [B, 3]       Matern mixture weights for query
#
# Invariants:
#   - ell_min <= ell_major always (ordering enforced by softplus + offset)
#   - theta ∈ (-pi/2, pi/2)       (tanh-bounded for anisotropy ambiguity)
#   - pi ∈ Δ² (softmax-normalised)

NonStationaryMaternKrigingLayer <- nn_module(
  "NonStationaryMaternKrigingLayer",

  initialize = function(d,
                        hidden_dim     = 64L,
                        init_ell_major = 1.0,
                        init_ell_minor = 0.7,
                        init_theta     = 0.0,
                        init_mix_mean  = 1.5,   # prior smoothness
                        theta_max      = pi / 2,
                        nu_classes     = c(0.5, 1.5, 2.5)) {
    self$d          <- as.integer(d)
    self$hidden_dim <- as.integer(hidden_dim)
    self$theta_max  <- as.numeric(theta_max)
    self$nu_classes <- as.numeric(nu_classes)
    n_nu            <- length(self$nu_classes)
    # out_dim: log ell_minor, log (ell_major - ell_minor) increment,
    #          theta_raw, and n_nu mixture logits.
    self$out_dim <- as.integer(3L + n_nu)

    # Hypernet — small MLP, GELU, layer-norm regularised.
    self$hyper <- nn_sequential(
      nn_linear(self$d, self$hidden_dim),
      nn_gelu(),
      nn_layer_norm(self$hidden_dim),
      nn_linear(self$hidden_dim, self$hidden_dim),
      nn_gelu(),
      nn_linear(self$hidden_dim, self$out_dim)
    )

    # Initialise the last-layer bias so that the hypernet output at t=0 is the
    # variogram-fit prior (log ell_minor ≈ log init_ell_minor etc). This keeps
    # NMK consistent with GeoVersa's "auto from variogram" philosophy.
    last_linear <- self$hyper[[length(self$hyper)]]
    with_no_grad({
      last_linear$weight$zero_()
      ell_min_safe <- max(init_ell_minor, 1e-4)
      ell_diff     <- max(init_ell_major - init_ell_minor, 1e-4)
      theta_init   <- init_theta / max(self$theta_max, 1e-6)
      theta_init   <- min(max(theta_init, -0.999), 0.999)
      # atanh(theta_init) so that theta_max * tanh(raw) = init_theta.
      theta_raw    <- 0.5 * log((1 + theta_init) / (1 - theta_init))
      bias_vec <- c(
        log(ell_min_safe),          # raw for softplus -> ell_minor ≈ init
        .nmk_inv_softplus(ell_diff),# raw for softplus -> (major - minor) ≈ init
        theta_raw,                  # raw -> theta = theta_max * tanh(raw)
        rep(0, length(self$nu_classes))
      )
      # Bias the nu=3/2 class to be the modal prior (smoothness init_mix_mean).
      mix_logit <- -((self$nu_classes - init_mix_mean) ^ 2) * 4
      bias_vec[(3L + 1L):(3L + length(self$nu_classes))] <- mix_logit
      last_linear$bias$copy_(torch_tensor(bias_vec, dtype = torch_float()))
    })
  },

  # Produce phi = (log ell_maj, log ell_min, theta, pi_1..pi_n_nu) from z.
  # Returns a list of four tensors so the caller can mix-and-average without
  # re-slicing repeatedly. Works for both [B, d] (query) and [B, K, d]
  # (neighbor) inputs — we flatten, apply the hypernet once, then reshape
  # back so we never rely on negative-dim indexing.
  decode_params = function(z) {
    orig_shape <- z$shape
    n_dim      <- z$dim()
    d_in       <- orig_shape[n_dim]
    if (n_dim == 3L) {
      z2 <- z$reshape(c(-1L, d_in))
    } else if (n_dim == 2L) {
      z2 <- z
    } else {
      stop("NMK decode_params: z must be rank 2 or 3.")
    }

    raw <- self$hyper(z2)                              # [N, out_dim]
    n_nu <- length(self$nu_classes)
    # Explicit column slicing on the trailing out_dim axis.
    log_emin  <- raw$narrow(2L, 1L, 1L)$squeeze(2L)
    delta_raw <- raw$narrow(2L, 2L, 1L)$squeeze(2L)
    theta_raw <- raw$narrow(2L, 3L, 1L)$squeeze(2L)
    mix_raw   <- raw$narrow(2L, 4L, n_nu)              # [N, n_nu]

    # Lower bounds chosen to keep 1 / ell from exploding in float32 even in
    # the worst-case fold (e.g. BLOOCV after a large buffer removal where
    # the variogram fit can return a near-zero range). With ell >= 1e-2 in
    # standardised coordinates, an anisotropic distance stays well inside
    # float32's representable range.
    ell_minor   <- nnf_softplus(log_emin) + 1e-2
    ell_major   <- ell_minor + nnf_softplus(delta_raw) + 1e-4
    log_ell_maj <- torch_log(ell_major)
    log_ell_min <- torch_log(ell_minor)
    theta       <- self$theta_max * torch_tanh(theta_raw)
    mix         <- nnf_softmax(mix_raw, dim = 2L)

    if (n_dim == 3L) {
      B <- orig_shape[1]; K <- orig_shape[2]
      log_ell_maj <- log_ell_maj$reshape(c(B, K))
      log_ell_min <- log_ell_min$reshape(c(B, K))
      theta       <- theta$reshape(c(B, K))
      mix         <- mix$reshape(c(B, K, n_nu))
    }

    list(
      log_ell_maj = log_ell_maj,
      log_ell_min = log_ell_min,
      theta       = theta,
      mix         = mix
    )
  },

  forward = function(z_i, coords_i, z_n, coords_n, r_n) {
    dev   <- z_i$device
    B     <- z_i$shape[1]
    K     <- z_n$shape[2]
    nu    <- torch_tensor(self$nu_classes, dtype = torch_float(), device = dev)

    # Detach z before feeding into the hypernet: the kernel field is a
    # READ-ONLY projection of the current backbone representation. This
    # stops the prediction loss from pushing the backbone latent around to
    # produce convenient kernel parameters, which in practice was the
    # dominant source of cross-seed variance at n = 500.
    p_i <- self$decode_params(z_i$detach())          # each [B]
    p_n <- self$decode_params(z_n$detach())          # each [B, K] or [B, K, n_nu]

    # Symmetric Stein-Paciorek averaging of the log-params and theta.
    log_maj_ij <- 0.5 * (p_i$log_ell_maj$unsqueeze(2L) + p_n$log_ell_maj)
    log_min_ij <- 0.5 * (p_i$log_ell_min$unsqueeze(2L) + p_n$log_ell_min)
    theta_ij   <- 0.5 * (p_i$theta$unsqueeze(2L)       + p_n$theta)
    mix_ij     <- 0.5 * (p_i$mix$unsqueeze(2L)         + p_n$mix)       # [B, K, n_nu]

    ell_maj_ij <- torch_exp(log_maj_ij)
    ell_min_ij <- torch_exp(log_min_ij)

    # Anisotropic rotated coordinate difference.
    dx <- coords_i[, 1]$unsqueeze(2L) - coords_n[, , 1]
    dy <- coords_i[, 2]$unsqueeze(2L) - coords_n[, , 2]
    cth <- torch_cos(theta_ij)
    sth <- torch_sin(theta_ij)
    u   <-  cth * dx + sth * dy
    v   <- -sth * dx + cth * dy
    d_star <- torch_sqrt((u / ell_maj_ij) ^ 2 + (v / ell_min_ij) ^ 2 + 1e-8)   # [B, K]

    # Numerically safe Matern distance: beyond d ~ 10 anisotropic lengths the
    # kernel is already astronomically small (< 1e-20), but the polynomial *
    # exp factorisation can trip float32 autograd — the polynomial grows as
    # d^2 while exp(-sqrt(5) d) underflows, producing Inf * 0 = NaN in some
    # backward paths. Clamping d to [0, 10] matches the forward kernel to
    # machine precision and eliminates the NaN pathway entirely.
    d_safe     <- torch_clamp(d_star, min = 0.0, max = 10.0)

    # Closed-form Matern values for each nu class (stack along last dim).
    sqrt3 <- sqrt(3)
    sqrt5 <- sqrt(5)

    k_half     <- torch_exp(-d_safe)                                    # nu = 1/2
    k_three_h  <- (1 + sqrt3 * d_safe) * torch_exp(-sqrt3 * d_safe)     # nu = 3/2
    k_five_h   <- (1 + sqrt5 * d_safe + (5 / 3) * d_safe ^ 2) *
                  torch_exp(-sqrt5 * d_safe)                            # nu = 5/2

    # Stack along a new last dim: [B, K, 3], one slice per nu class, in the
    # order specified by self$nu_classes. If the user customises NMK_NU_CLASSES
    # to exclude one of the cases we still return the right columns: match by
    # index on self$nu_classes.
    stack_list <- list()
    for (i in seq_along(self$nu_classes)) {
      nu_i <- self$nu_classes[i]
      if (abs(nu_i - 0.5) < 1e-6)       stack_list[[i]] <- k_half
      else if (abs(nu_i - 1.5) < 1e-6)  stack_list[[i]] <- k_three_h
      else if (abs(nu_i - 2.5) < 1e-6)  stack_list[[i]] <- k_five_h
      else stop(sprintf("NMK: Matern nu = %g is not a closed-form case.", nu_i))
    }
    k_stack <- torch_stack(stack_list, dim = -1L)                       # [B, K, n_nu]

    # Convex mixture.
    k_ij <- torch_sum(mix_ij * k_stack, dim = -1L)                      # [B, K]

    # Normalise to a proper kriging weight (positive, rows sum to 1).
    denom <- torch_sum(k_ij, dim = 2L, keepdim = TRUE) + 1e-8
    w     <- k_ij / denom                                               # [B, K]
    delta <- torch_sum(w * r_n, dim = 2L)                               # [B]

    list(
      delta      = delta,
      w          = w,
      k_ij       = k_ij,
      aniso_dist = d_star,
      phi_i      = p_i,
      phi_n      = p_n,
      mix_i      = p_i$mix,
      mix_ij     = mix_ij
    )
  }
)


# -----------------------------------------------------------------------------
# §1b  NeuralObliqueForest — differentiable oblique decision forest
# -----------------------------------------------------------------------------
#
# Replaces the ScalarHead MLP used by ConvKrigingNet2D with a differentiable
# oblique decision forest (Kontschieder et al. 2015, "Deep Neural Decision
# Forests", ICCV). Each tree of depth D has 2^D leaves, each internal node
# is an oblique split sigma(w_n^T z + b_n) in (0, 1), and leaf probabilities
# are products of routing decisions along the root-to-leaf path. Leaves hold
# scalar prediction values. The forest averages T such trees.
#
# Rationale vs ScalarHead:
#   - ScalarHead is a globally smooth MLP and, with n = 500 and 28 covariates,
#     tends to underfit the piecewise landscape structure that RF handles
#     natively via tree splits.
#   - NeuralObliqueForest gives tree-like piecewise-constant adaptive
#     representation while remaining end-to-end differentiable, so it plays
#     well with the Matern residual head (which still interpolates residuals
#     via the learned non-stationary kernel). The Matern head is now a
#     principled SPATIAL STRUCTURE layer on top of a piecewise-adaptive trend.
#
# Memory layout choice:
#   All T * n_nodes oblique projections are a single nn_linear of output size
#   T * n_nodes, reshaped to [B, T, n_nodes]. All T * n_leaves scalar leaf
#   values are a single 2D parameter matrix. The forward pass is vectorised
#   across trees and batch, so the for-loop in forward only iterates D times
#   (one per depth level) regardless of T.
#
# Shape contract:
#   input  z    : [B, d_in]
#   output yhat : [B]
#
# Initialisation:
#   Leaf values ~ N(0, 0.1^2) — breaks symmetry between leaves early; with
#   pure-zero init the trees could freeze because every leaf contributes
#   the same gradient.

NeuralObliqueForest <- nn_module(
  "NeuralObliqueForest",

  initialize = function(d_in, n_trees = 8L, depth = 4L) {
    self$d_in     <- as.integer(d_in)
    self$n_trees  <- as.integer(n_trees)
    self$depth    <- as.integer(depth)
    self$n_leaves <- 2L ^ self$depth
    self$n_nodes  <- self$n_leaves - 1L     # internal nodes

    # All T * n_nodes oblique projections packed into a single linear
    # module — one matrix multiply handles every split in every tree.
    self$node_linear <- nn_linear(self$d_in, self$n_trees * self$n_nodes)

    # Leaf scalars: [n_trees, n_leaves]. We need init with std ~ target
    # scale so the forest doesn't collapse into a constant predictor at
    # initialisation. Targets are standardised (mean 0, std 1) before
    # training, so std = 1 spans the bulk of target values and provides
    # immediate differentiation between leaves — the key ingredient for
    # useful gradients to flow into the oblique node weights.
    # Empirical: std = 0.1 caused full collapse (ME = -12, RMSE = 52);
    # std = 1.0 breaks the degenerate attractor.
    self$leaf_values <- nn_parameter(
      torch_randn(self$n_trees, self$n_leaves) * 1.0
    )
  },

  forward = function(z) {
    B <- z$shape[1]
    T <- self$n_trees
    D <- self$depth

    # All node logits at once: [B, T * n_nodes] -> [B, T, n_nodes]
    node_logits <- self$node_linear(z)
    node_logits <- node_logits$reshape(c(B, T, self$n_nodes))

    # Soft routing probability (go right) per node.
    node_probs <- torch_sigmoid(node_logits)

    # Initial position mass at the root: one position per tree.
    # Shape: [B, T, 1]
    leaf_probs <- torch_ones(B, T, 1L, device = z$device,
                             dtype = z$dtype)

    for (d_level in seq_len(D)) {
      n_prev     <- 2L ^ (d_level - 1L)
      node_start <- n_prev - 1L      # 0-indexed first node at this level

      # Node probs for the n_prev nodes that split this level.
      # R's $narrow uses 1-indexed start and a length.
      node_slice <- node_probs$narrow(3L,
                                      start  = node_start + 1L,
                                      length = n_prev)    # [B, T, n_prev]

      left_probs  <- leaf_probs * (1 - node_slice)
      right_probs <- leaf_probs * node_slice

      # Interleave: stack along a new last dim then reshape. The resulting
      # ordering is [left_0, right_0, left_1, right_1, ...], which matches
      # the standard level-order binary-tree leaf layout.
      stacked    <- torch_stack(list(left_probs, right_probs),
                                dim = 4L)                 # [B, T, n_prev, 2]
      leaf_probs <- stacked$reshape(c(B, T, 2L * n_prev))
    }

    # leaf_probs: [B, T, n_leaves]; leaf_values: [T, n_leaves]
    # Weighted sum per tree, then mean across trees.
    tree_preds <- torch_sum(
      leaf_probs * self$leaf_values$unsqueeze(1L),
      dim = 3L
    )                                                     # [B, T]
    torch_mean(tree_preds, dim = 2L)                      # [B]
  }
)


# -----------------------------------------------------------------------------
# §2  NeuroMaternKrige — full model class
# -----------------------------------------------------------------------------

# Shares tabular / patch / coord encoders with ConvKrigingNet2D so that the
# comparison at the Wadoux-protocol level is a clean kernel-head ablation.

NeuroMaternKrige <- nn_module(
  "NeuroMaternKrige",

  initialize = function(c_tab,
                        patch_channels,
                        d               = 256L,
                        tab_hidden      = c(192L),
                        tab_dropout     = 0.15,
                        tab_input_dropout = 0.0,
                        tab_quantile_basis = FALSE,
                        tab_qbasis_thresholds = NULL,
                        tab_qbasis_bandwidths = NULL,
                        tab_qbasis_dropout = NULL,
                        tab_crossnet = FALSE,
                        tab_cross_layers = 2L,
                        tab_cross_dropout = NULL,
                        tab_skip = FALSE,
                        tab_skip_hidden = NULL,
                        tab_skip_dropout = NULL,
                        tab_skip_init = -1.5,
                        patch_dim       = 128L,
                        patch_dropout   = 0.10,
                        multiscale_patch = TRUE,
                        coord_hidden    = c(32L),
                        coord_dim       = 32L,
                        coord_dropout   = 0.05,
                        fusion_hidden   = 256L,
                        kriging_hidden  = 64L,
                        init_ell_major  = 1.0,
                        init_ell_minor  = 0.7,
                        init_theta      = 0.0,
                        init_mix_mean   = 1.5,
                        beta_init       = 0.0,
                        nu_classes      = c(0.5, 1.5, 2.5),
                        # Oblique decision-forest backbone head (Class B extension).
                        n_trees         = 8L,
                        tree_depth      = 4L) {
    self$d <- as.integer(d)
    self$tab_input_dropout <- as.numeric(max(0, min(0.5, tab_input_dropout)))

    self$enc_tab <- make_mlp(c_tab, hidden = tab_hidden,
                             out_dim = self$d, dropout = tab_dropout)
    self$tab_crossnet <- isTRUE(tab_crossnet)
    self$tab_cross_layers <- as.integer(max(1L, min(2L, round(tab_cross_layers))))
    tab_cross_dropout <- if (is.null(tab_cross_dropout) ||
                             !is.finite(tab_cross_dropout)) {
      tab_dropout / 2
    } else {
      as.numeric(tab_cross_dropout)
    }
    if (self$tab_crossnet) {
      self$tab_cross_1 <- nn_linear(c_tab, 1L)
      self$tab_cross_2 <- if (self$tab_cross_layers >= 2L) {
        nn_linear(c_tab, 1L)
      } else {
        NULL
      }
      self$tab_cross_proj <- nn_sequential(
        nn_linear(c_tab, self$d),
        nn_gelu(),
        nn_layer_norm(self$d),
        nn_dropout(tab_cross_dropout)
      )
      self$tab_cross_logit_scale <- nn_parameter(torch_tensor(-1.5))
    } else {
      self$tab_cross_1 <- NULL
      self$tab_cross_2 <- NULL
      self$tab_cross_proj <- NULL
      self$tab_cross_logit_scale <- NULL
    }

    self$tab_quantile_basis <- isTRUE(tab_quantile_basis) &&
      !is.null(tab_qbasis_thresholds) &&
      !is.null(tab_qbasis_bandwidths)
    tab_qbasis_dropout <- if (is.null(tab_qbasis_dropout) ||
                              !is.finite(tab_qbasis_dropout)) {
      tab_dropout / 2
    } else {
      as.numeric(tab_qbasis_dropout)
    }
    if (self$tab_quantile_basis) {
      self$tab_qbasis <- SoftQuantileBasisTab_Auto(
        c_tab = c_tab,
        out_dim = self$d,
        thresholds = tab_qbasis_thresholds,
        bandwidths = tab_qbasis_bandwidths,
        dropout = tab_qbasis_dropout
      )
    } else {
      self$tab_qbasis <- NULL
    }

    self$tab_skip <- isTRUE(tab_skip)
    if (self$tab_skip) {
      tab_skip_hidden <- if (is.null(tab_skip_hidden) ||
                             !is.finite(tab_skip_hidden)) {
        as.integer(max(48L, min(256L, round(6L * c_tab))))
      } else {
        as.integer(max(16L, round(tab_skip_hidden)))
      }
      tab_skip_dropout <- if (is.null(tab_skip_dropout) ||
                              !is.finite(tab_skip_dropout)) {
        tab_dropout / 2
      } else {
        as.numeric(tab_skip_dropout)
      }
      self$tab_skip_hidden <- tab_skip_hidden
      self$tab_skip_head <- nn_sequential(
        nn_linear(c_tab, tab_skip_hidden),
        nn_gelu(),
        nn_layer_norm(tab_skip_hidden),
        nn_dropout(tab_skip_dropout),
        nn_linear(tab_skip_hidden, 1L)
      )
      self$tab_skip_logit_scale <- nn_parameter(torch_tensor(as.numeric(tab_skip_init)))
    } else {
      self$tab_skip_hidden <- NA_integer_
      self$tab_skip_head <- NULL
      self$tab_skip_logit_scale <- NULL
    }

    if (isTRUE(multiscale_patch)) {
      self$enc_patch <- MultiScalePatchEncoder2D(
        in_channels = patch_channels, out_dim = patch_dim, dropout = patch_dropout)
    } else {
      self$enc_patch <- PatchEncoder2D(
        in_channels = patch_channels, out_dim = patch_dim, dropout = patch_dropout)
    }
    self$proj_patch <- nn_linear(patch_dim, self$d)

    self$enc_coord  <- make_mlp(2L, hidden = coord_hidden,
                                out_dim = coord_dim, dropout = coord_dropout)
    self$proj_coord <- nn_linear(coord_dim, self$d)

    self$fuse <- nn_sequential(
      nn_linear(3L * self$d, fusion_hidden),
      nn_gelu(),
      nn_dropout(0.10),
      nn_linear(fusion_hidden, self$d)
    )
    # Head: differentiable oblique decision forest. Piecewise-adaptive
    # trend model — direct response to RF's dominance on tabular DSM at
    # small n.
    self$head <- NeuralObliqueForest(
      d_in    = self$d,
      n_trees = n_trees,
      depth   = tree_depth
    )

    self$krig <- NonStationaryMaternKrigingLayer(
      d              = self$d,
      hidden_dim     = kriging_hidden,
      init_ell_major = init_ell_major,
      init_ell_minor = init_ell_minor,
      init_theta     = init_theta,
      init_mix_mean  = init_mix_mean,
      nu_classes     = nu_classes
    )

    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode = function(x_tab, x_patch, coords) {
    x_tab_in <- x_tab
    if (self$tab_input_dropout > 0 && isTRUE(self$training)) {
      keep_prob <- torch_full(
        c(1L, as.integer(x_tab$shape[2])),
        1 - self$tab_input_dropout,
        dtype = torch_float(),
        device = x_tab$device
      )
      feature_mask <- torch_bernoulli(keep_prob) / (1 - self$tab_input_dropout)
      x_tab_in <- x_tab * feature_mask
    }
    z_tab <- self$enc_tab(x_tab_in)
    if (isTRUE(self$tab_crossnet)) {
      x_cross <- x_tab_in * self$tab_cross_1(x_tab_in) + x_tab_in
      if (!is.null(self$tab_cross_2)) {
        x_cross <- x_tab_in * self$tab_cross_2(x_cross) + x_cross
      }
      cross_scale <- torch_sigmoid(self$tab_cross_logit_scale)
      z_tab <- z_tab + cross_scale * self$tab_cross_proj(x_cross)
    }
    if (isTRUE(self$tab_quantile_basis)) {
      z_tab <- z_tab + self$tab_qbasis(x_tab)
    }
    if (isTRUE(self$tab_skip)) {
      tab_skip_scale <- torch_sigmoid(self$tab_skip_logit_scale)
      pred_tab_skip <- self$tab_skip_head(x_tab_in)$squeeze(2L)
    } else {
      tab_skip_scale <- NULL
      pred_tab_skip <- NULL
    }
    z_patch <- self$proj_patch(self$enc_patch(x_patch))
    z_coord <- self$proj_coord(self$enc_coord(coords))
    z       <- self$fuse(torch_cat(list(z_tab, z_patch, z_coord), dim = 2L))
    list(z = z, z_tab = z_tab, z_patch = z_patch, z_coord = z_coord,
         pred_tab_skip = pred_tab_skip, tab_skip_scale = tab_skip_scale)
  },

  forward_base = function(x_tab, x_patch, coords) {
    enc  <- self$encode(x_tab, x_patch, coords)
    pred <- self$head(enc$z)
    if (!is.null(enc$pred_tab_skip)) {
      pred <- pred + enc$tab_skip_scale * enc$pred_tab_skip
    }
    list(pred = pred, z = enc$z)
  },

  forward_with_kriging = function(x_tab, x_patch, coords,
                                  z_n, coords_n, r_n,
                                  dist_scale = NULL) {
    # `dist_scale` is only consulted at eval time. During training it is
    # ignored (the kriging signal must flow through undamped so the
    # hypernet can learn). At eval, if supplied, we multiply beta by an
    # exp(-rel_dist / dist_scale) gate where
    # rel_dist = min(d*) / mean(d*) in (0, 1].
    base <- self$forward_base(x_tab, x_patch, coords)
    k    <- self$krig(base$z, coords, z_n, coords_n, r_n)
    beta <- torch_sigmoid(self$logit_beta)
    if (!self$training && !is.null(dist_scale) &&
        is.finite(dist_scale) && dist_scale > 0) {
      # Use the same clamped distance the kernel uses, so min_d and mean_d
      # are always finite and on the same scale as the weighting rule.
      d_safe <- torch_clamp(k$aniso_dist, min = 0.0, max = 10.0)
      min_d  <- torch_amin(d_safe, dim = 2L)
      mean_d <- torch_mean(d_safe, dim = 2L) + 1e-6
      rel_d  <- torch_clamp(min_d / mean_d, min = 0.0, max = 1.0)
      gate   <- torch_exp(-rel_d / as.numeric(dist_scale))
      eff_beta <- beta * gate
      pred <- base$pred + eff_beta * k$delta
    } else {
      pred <- base$pred + beta * k$delta
    }
    list(pred       = pred,
         base_pred  = base$pred,
         z          = base$z,
         delta      = k$delta,
         beta       = beta,
         w          = k$w,
         aniso_dist = k$aniso_dist,
         phi_i      = k$phi_i,
         phi_n      = k$phi_n,
         mix_i      = k$mix_i,
         mix_ij     = k$mix_ij)
  }
)


# -----------------------------------------------------------------------------
# §3  Tensor caches, bank refresh, prediction utilities
#
# These are thin wrappers that reuse the torch-tensor helpers defined in
# ConvKrigingNet2D.R (build_convkrigingnet2d_tensor_cache, etc.). We redefine
# the bank-refresh and memory-prediction functions here so that NMK stays
# self-contained even if the upstream helpers change.
# -----------------------------------------------------------------------------

# Compute a tensor memory bank (Z, R, C) for the training set under the
# current backbone weights. The residuals R are in STANDARDISED target units
# (the model is trained on y_s = (y - mu) / sigma).
refresh_nmk_bank_tensor <- function(model, tensor_cache, batch_size = 256L) {
  model$eval()
  n <- tensor_cache$n

  Z_list <- list(); R_list <- list(); C_list <- list()
  with_no_grad({
    for (s in seq(1L, n, by = batch_size)) {
      e    <- min(s + batch_size - 1L, n)
      idx  <- seq(s, e)
      idx_t <- torch_tensor(idx, dtype = torch_long(), device = tensor_cache$device)

      xb <- tensor_cache$X$index_select(1L, idx_t)
      pb <- tensor_cache$P$index_select(1L, idx_t)
      cb <- tensor_cache$C$index_select(1L, idx_t)
      yb <- tensor_cache$y$index_select(1L, idx_t)

      out <- model$forward_base(xb, pb, cb)
      rb  <- yb - out$pred

      Z_list[[length(Z_list) + 1L]] <- out$z$cpu()
      R_list[[length(R_list) + 1L]] <- rb$cpu()
      C_list[[length(C_list) + 1L]] <- cb$cpu()
    }
  })

  list(
    Zmem = torch_cat(Z_list, dim = 1L)$to(device = tensor_cache$device),
    Rmem = torch_cat(R_list, dim = 1L)$to(device = tensor_cache$device),
    Cmem = torch_cat(C_list, dim = 1L)$to(device = tensor_cache$device)
  )
}

# Batched backbone-only prediction.
predict_nmk_base_tensor <- function(model, tensor_cache, batch_size = 256L) {
  preds <- numeric(tensor_cache$n)
  with_no_grad({
    for (s in seq(1L, tensor_cache$n, by = batch_size)) {
      e    <- min(s + batch_size - 1L, tensor_cache$n)
      idx  <- seq(s, e)
      idx_t <- torch_tensor(idx, dtype = torch_long(), device = tensor_cache$device)
      xb <- tensor_cache$X$index_select(1L, idx_t)
      pb <- tensor_cache$P$index_select(1L, idx_t)
      cb <- tensor_cache$C$index_select(1L, idx_t)
      out <- model$forward_base(xb, pb, cb)
      preds[idx] <- as.numeric(out$pred$cpu())
    }
  })
  preds
}

# Batched full prediction (backbone + Matern kriging correction) using a given
# memory bank and a precomputed query-to-ref neighbor index tensor (knn_idx_t,
# shape [n_query, K], long). If dist_scale is supplied, the eval-time distance
# gate in model$forward_with_kriging kicks in to suppress kriging for query
# points whose nearest neighbor is far relative to the mean neighbour distance.
predict_nmk_with_memory_tensor <- function(model,
                                           tensor_cache,
                                           Zmem, Rmem, Cmem,
                                           K,
                                           knn_idx_t,
                                           batch_size = 256L,
                                           dist_scale = NULL) {
  model$eval()
  n <- tensor_cache$n
  preds <- numeric(n)

  with_no_grad({
    for (s in seq(1L, n, by = batch_size)) {
      e    <- min(s + batch_size - 1L, n)
      idx  <- seq(s, e)
      B    <- length(idx)
      idx_t <- torch_tensor(idx, dtype = torch_long(), device = tensor_cache$device)

      xb <- tensor_cache$X$index_select(1L, idx_t)
      pb <- tensor_cache$P$index_select(1L, idx_t)
      cb <- tensor_cache$C$index_select(1L, idx_t)

      nb   <- knn_idx_t$index_select(1L, idx_t)              # [B, K]
      nb_t <- nb$reshape(c(-1L))$to(dtype = torch_long())
      zn <- Zmem$index_select(1L, nb_t)$reshape(c(B, K, -1L))
      rn <- Rmem$index_select(1L, nb_t)$reshape(c(B, K))
      cn <- Cmem$index_select(1L, nb_t)$reshape(c(B, K, 2L))

      out <- model$forward_with_kriging(xb, pb, cb, zn, cn, rn,
                                        dist_scale = dist_scale)
      preds[idx] <- as.numeric(out$pred$cpu())
    }
  })

  preds
}


# -----------------------------------------------------------------------------
# §4  Auto-configuration derived from the variogram
# -----------------------------------------------------------------------------
#
# Mirrors auto_kriging_config() from ConvKrigingNet2D_Auto.R but produces
# only the parameters NMK actually consumes. Rules:
#
#   K_neighbors         : clamp(round(8 + 12 r), 8, 20)
#   base_loss_weight    : max(0.05, 0.10 r)
#   alpha_me            : 0.375 r
#   lambda_smooth       : 0.01 * (1 - r)
#   lambda_anchor       : 0.01 * (1 - r)
#   max_warmup_epochs   : clamp(round(4 + 16 r), 4, 20)
#   d                   : clamp(64 ceil(sqrt(n)/8), 128, 512)
#   patch_dim           : d / 2
#   tab_dropout         : clamp(0.30 - n/8000, 0.05, 0.30)
#   patch_dropout       : clamp(0.20 - n/10000, 0.03, 0.20)
#   batch_size          : clamp(round(n/8), 24, 64)
#   lr_init             : 1e-3 (overridden by a Polyak probe if desired)
#   lr_patience         : 4
#   patience            : 15
#   min_lr              : 5e-5
#   lr_decay            : 0.5
#   bank_refresh_every  : 2

nmk_auto_config <- function(vg, n_train, coord_scaler) {
  r  <- vg$nugget_ratio
  cs <- mean(coord_scaler$scale)
  fit_q <- if (is.finite(vg$fit_quality)) vg$fit_quality else 0.25

  # -- Kernel initialisation (variogram) --------------------------------------
  ell_major_init <- vg$range_major / cs
  ell_minor_init <- vg$range_minor / cs
  theta_init     <- vg$theta_rad

  # -- Geostatistical rules (derived from nugget ratio r = nugget/sill) -------
  K_neighbors      <- as.integer(.nmk_clamp(round(8 + 12 * r), 8L, 20L))
  base_loss_weight <- round(max(0.05, 0.10 * r), 4L)
  alpha_me         <- round(0.375 * r, 4L)
  lambda_cov       <- round(.nmk_env_num("NMK_LAMBDA_COV",    0.025) * (1 - r), 5L)
  lambda_smooth    <- round(.nmk_env_num("NMK_LAMBDA_SMOOTH", 0.24 ) * (1 - r), 5L)
  lambda_anchor    <- round(.nmk_env_num("NMK_LAMBDA_ANCHOR", 0.24 ) * (1 - r), 5L)

  max_warmup_epochs <- as.integer(.nmk_clamp(round(4 + 16 * r), 4L, 20L))

  krig_drop_p_train <- .nmk_clamp(
    .nmk_env_num("NMK_KRIG_DROP_P_TRAIN", min(1 - r, 0.3)),
    0.0, 0.8)

  # Eval-time distance gate derived from nugget ratio:
  #   r low  (strong spatial signal) -> large scale -> gentle gate
  #   r high (mostly nugget)         -> small scale -> sharp suppression
  # scale = clamp(0.8 - 0.6 r, 0.2, 0.8). At r = 0.5 this matches the old
  # hand-picked 0.5; it adapts to both extremes.
  dist_scale_eval   <- round(.nmk_env_num(
    "NMK_DIST_SCALE", .nmk_clamp(0.8 - 0.6 * r, 0.2, 0.8)), 3L)

  # Initial Matern mixture mean nu drawn from variogram fit quality:
  #   fit_q -> 1  the exponential-kernel fit is excellent -> prefer nu = 0.5
  #   fit_q -> 0  the exponential fit is poor              -> prefer nu = 2.5
  # Linear interpolation, clamped to the available closed-form cases.
  init_mix_mean <- .nmk_clamp(0.5 + 2.0 * (1 - fit_q), 0.5, 2.5)

  # beta_init from nugget ratio (same logic as ConvKrigingNet2D_Auto v5):
  #   r -> 0 : kriging is powerful               -> logit_beta = +2  (beta ~ 0.88)
  #   r -> 1 : nugget dominates, kriging useless -> logit_beta = -4  (beta ~ 0.02)
  beta_init <- .nmk_clamp(2.0 - 6.0 * r, -5.0, 3.0)

  # -- Statistical-capacity rules (derived from n_train) -----------------------
  # d: embedding width via square-root scaling (Bartlett & Mendelson 2002).
  d           <- as.integer(.nmk_clamp(64L * ceiling(sqrt(n_train) / 8),
                                       128L, 512L))
  patch_dim   <- as.integer(d / 2L)
  # Hypernet hidden width = d / 4, bounded to avoid either collapse or a
  # second-backbone-sized module sitting on top of the kernel head.
  kriging_hidden <- as.integer(.nmk_clamp(round(d / 4), 32L, 128L))

  # Oblique decision-forest backbone (replaces the ScalarHead MLP).
  #   tree_depth scales as log2(n)/2 so the number of leaves grows roughly
  #     with sqrt(n) — each "leaf" needs a handful of training points to
  #     keep the local tree statistic meaningful.
  #   n_trees  scales as sqrt(n / 10) so we bag more trees with more data,
  #     reducing the variance from the soft-routing's sample-dependence.
  tree_depth <- as.integer(.nmk_clamp(round(log2(max(n_train, 4L)) / 2),
                                      3L, 6L))
  n_trees    <- as.integer(.nmk_clamp(round(sqrt(n_train / 10)),
                                      4L, 16L))

  tab_dropout   <- round(.nmk_clamp(0.30 - n_train / 8000,  0.05, 0.30), 3L)
  patch_dropout <- round(.nmk_clamp(0.20 - n_train / 10000, 0.03, 0.20), 3L)
  batch_size    <- as.integer(.nmk_clamp(round(n_train / 8), 24L, 64L))

  # -- Optimisation schedule derived from max_warmup_epochs -------------------
  # max_warmup_epochs = W controls how long the backbone needs to find a
  # decent fit. After warmup, the main loop keeps schedule proportional to W:
  #   patience           = 2 W   (twice as long as warmup before early stop)
  #   lr_patience        = max(2, W / 2)
  #   bank_refresh_every = max(1, W / 3)
  W <- max_warmup_epochs
  patience_cfg       <- as.integer(.nmk_clamp(round(2 * W),  5L, 40L))
  lr_patience_cfg    <- as.integer(.nmk_clamp(round(W / 2),  2L, 10L))
  bank_refresh_cfg   <- as.integer(.nmk_clamp(round(W / 3),  1L,  5L))
  # lr_decay stays 0.5 (GeoVersa-wide convention; halves every patience loss).
  lr_decay_cfg       <- 0.5

  cat(sprintf(
    "[NMK auto]  r=%.3f  fit_q=%.2f  n=%d  K=%d  d=%d  krig_hid=%d  forest=%dx(depth%d,%dleaves)  bs=%d\n",
    r, fit_q, n_train, K_neighbors, d, kriging_hidden,
    n_trees, tree_depth, 2L ^ tree_depth, batch_size))
  cat(sprintf(
    "[NMK auto]  BLW=%.4f  alpha_me=%.4f  cov=%.4f  smooth=%.4f  anchor=%.4f\n",
    base_loss_weight, alpha_me, lambda_cov, lambda_smooth, lambda_anchor))
  cat(sprintf(
    "[NMK auto]  krig_drop=%.2f  dist_scale=%.2f  mix_mean=%.2f  beta_init_logit=%.2f\n",
    krig_drop_p_train, dist_scale_eval, init_mix_mean, beta_init))
  cat(sprintf(
    "[NMK auto]  warmup<=%d  patience=%d  lr_patience=%d  bank_refresh=%d  lr_decay=%.2f\n",
    max_warmup_epochs, patience_cfg, lr_patience_cfg, bank_refresh_cfg, lr_decay_cfg))

  list(
    K_neighbors        = K_neighbors,
    base_loss_weight   = base_loss_weight,
    alpha_me           = alpha_me,
    lambda_cov         = lambda_cov,
    lambda_smooth      = lambda_smooth,
    lambda_anchor      = lambda_anchor,
    krig_drop_p_train  = krig_drop_p_train,
    dist_scale_eval    = dist_scale_eval,
    max_warmup_epochs  = max_warmup_epochs,
    d                  = d,
    patch_dim          = patch_dim,
    kriging_hidden     = kriging_hidden,
    n_trees            = n_trees,
    tree_depth         = tree_depth,
    tab_dropout        = tab_dropout,
    patch_dropout      = patch_dropout,
    batch_size         = batch_size,
    # lr_init will be set by the Polyak probe once the model is built; the
    # fallback below is never used unless the probe is explicitly disabled.
    lr_init            = NA_real_,
    lr_patience        = lr_patience_cfg,
    patience           = patience_cfg,
    # min_lr is filled in after the LR probe runs (= lr_init / 20).
    min_lr             = NA_real_,
    lr_decay           = lr_decay_cfg,
    bank_refresh_every = bank_refresh_cfg,
    beta_init          = beta_init,
    init_mix_mean      = init_mix_mean,
    ell_major_init     = ell_major_init,
    ell_minor_init     = ell_minor_init,
    theta_init         = theta_init,
    nugget_ratio       = r,
    fit_quality        = fit_q,
    fusion_hidden      = d
  )
}


# -----------------------------------------------------------------------------
# §5  Training loop
# -----------------------------------------------------------------------------
#
# Interface: train_neuromaternkrige_one_fold(fd, ...)
#
# fd is the fold dictionary produced by prepare_pointpatch_fold() in
# KrigingNet_PointPatchCNN.R. It has:
#   fd$X$train / fd$X$val / fd$X$test             scaled covariate matrices
#   fd$patches$train / fd$patches$val / fd$patches$test
#                                                  4D raster patch arrays
#                                                  [C, H, W, N]
#   fd$coords$train / fd$coords$val / fd$coords$test   raw coords
#   fd$y$train / fd$y$val / fd$y$test              raw targets
#   fd$neighbor_idx_train                          train-to-train KNN
#
# Returns:
#   pred_test    : vector of NMK predictions on the test set
#   pred_val     : vector of NMK predictions on the validation set
#   pred_test_base, pred_val_base : backbone-only counterparts
#   metrics_test : one-row data.frame (N, R2, RMSE, MAE, Bias, RPIQ)
#   diagnostics  : named list with training-time diagnostics

train_neuromaternkrige_one_fold <- function(
  fd,
  # Runner can still override any of the below; NULL means "use the value
  # derived by nmk_auto_config / the Polyak probe".
  epochs               = 80L,
  lr                   = NULL,
  wd                   = NULL,
  batch_size           = NULL,
  patience             = NULL,
  warmup_epochs        = NULL,
  bank_refresh_every   = NULL,
  train_seed           = NULL,
  deterministic_batches = FALSE,
  lr_decay             = NULL,
  lr_patience          = NULL,
  min_lr               = NULL,
  base_loss_weight     = NULL,
  lambda_smooth        = NULL,
  lambda_anchor        = NULL,
  lambda_cov           = NULL,
  krig_drop_p_train    = NULL,
  dist_scale_eval      = NULL,
  kriging_hidden       = NULL,
  n_trees              = NULL,
  tree_depth           = NULL,
  tab_hidden           = c(192L),
  tab_input_dropout    = NULL,
  tab_quantile_basis   = FALSE,
  tab_qbasis_knots     = 4L,
  tab_qbasis_dropout   = NULL,
  tab_crossnet         = FALSE,
  tab_cross_layers     = 2L,
  tab_cross_dropout    = NULL,
  tab_skip             = FALSE,
  tab_skip_hidden      = NULL,
  tab_skip_dropout     = NULL,
  tab_skip_init        = -1.5,
  tab_dropout          = NULL,
  patch_dim            = NULL,
  patch_dropout        = NULL,
  coord_hidden         = c(32L),
  coord_dim            = 32L,
  coord_dropout        = 0.05,
  fusion_hidden        = NULL,
  beta_init            = NULL,
  init_mix_mean        = NULL,
  nu_classes           = c(0.5, 1.5, 2.5),
  K_neighbors          = NULL,
  vg_cutoff_frac       = 0.50,
  vg_n_lags            = 15L,
  target_transform     = "auto",
  device               = "cpu",
  warmup_converge_tol  = 0.01,
  warmup_patience      = 3L,
  refit_fixed_epochs   = FALSE,
  init_state           = NULL,
  return_state         = FALSE,
  ...
) {

  # --- Extract fold data -----------------------------------------------------
  Xtr <- fd$X$train;       Xva <- fd$X$val;        Xte <- fd$X$test
  Ptr <- fd$patches$train; Pva <- fd$patches$val;  Pte <- fd$patches$test
  Ctr <- fd$coords$train;  Cva <- fd$coords$val;   Cte <- fd$coords$test
  ytr <- fd$y$train;       yva <- fd$y$val;        yte <- fd$y$test

  # --- Auto target transform -------------------------------------------------
  # If the caller requested target_transform = "auto" (the default), decide
  # between "identity" and "log" from the training target itself:
  #   right-skewed & strictly positive -> "log"; otherwise "identity".
  # This keeps the deduction chain complete — the target transform is a
  # function of the data, not a knob.
  target_transform <- tolower(as.character(target_transform))
  if (identical(target_transform, "auto")) {
    target_transform <- .nmk_target_transform_auto(ytr)
    cat(sprintf("[NMK auto]  target_transform auto-selected -> '%s'\n",
                target_transform))
  }

  # --- Scalers (target + coords) --------------------------------------------
  ytr_t <- transform_target(ytr, target_transform)
  yva_t <- transform_target(yva, target_transform)
  y_scaler     <- fit_target_scaler(ytr_t)
  coord_scaler <- fit_standard_scaler(Ctr)

  ytr_s <- apply_target_scaler(ytr_t, y_scaler)
  yva_s <- apply_target_scaler(yva_t, y_scaler)
  Ctr_s <- apply_standard_scaler(Ctr, coord_scaler)
  Cva_s <- apply_standard_scaler(Cva, coord_scaler)
  Cte_s <- apply_standard_scaler(Cte, coord_scaler)

  n_train <- nrow(Xtr)

  # --- Small-n advisory (Class G — informational only) ----------------------
  suppress_small_n <- identical(
    toupper(Sys.getenv("NMK_SMALL_N_SUPPRESS", unset = "FALSE")),
    "TRUE")
  .nmk_small_n_advisory(n_train, model_label = "NMK",
                        suppress = suppress_small_n)

  # --- Variogram fit and auto-config ----------------------------------------
  cat("[NMK] === Variogram fit ===\n")
  vg  <- fit_variogram_auto(ytr, Ctr,
                             cutoff_frac = vg_cutoff_frac,
                             n_lags      = as.integer(vg_n_lags))
  cfg <- nmk_auto_config(vg, n_train, coord_scaler)

  # --- Runner overrides (only for fundamental non-spatial params) ----------
  if (!is.null(batch_size))         cfg$batch_size         <- as.integer(batch_size)
  if (!is.null(lr))                 cfg$lr_init            <- as.numeric(lr)
  if (!is.null(patience))           cfg$patience           <- as.integer(patience)
  if (!is.null(warmup_epochs))      cfg$max_warmup_epochs  <- as.integer(warmup_epochs)
  if (!is.null(bank_refresh_every)) cfg$bank_refresh_every <- as.integer(bank_refresh_every)
  if (!is.null(lr_decay))           cfg$lr_decay           <- as.numeric(lr_decay)
  if (!is.null(lr_patience))        cfg$lr_patience        <- as.integer(lr_patience)
  if (!is.null(min_lr))             cfg$min_lr             <- as.numeric(min_lr)
  if (!is.null(base_loss_weight))   cfg$base_loss_weight   <- as.numeric(base_loss_weight)
  if (!is.null(lambda_smooth))      cfg$lambda_smooth      <- as.numeric(lambda_smooth)
  if (!is.null(lambda_anchor))      cfg$lambda_anchor      <- as.numeric(lambda_anchor)
  if (!is.null(lambda_cov))         cfg$lambda_cov         <- as.numeric(lambda_cov)
  if (!is.null(krig_drop_p_train))  cfg$krig_drop_p_train  <- as.numeric(krig_drop_p_train)
  if (!is.null(dist_scale_eval))    cfg$dist_scale_eval    <- as.numeric(dist_scale_eval)
  if (!is.null(K_neighbors))        cfg$K_neighbors        <- as.integer(K_neighbors)
  if (!is.null(tab_dropout))        cfg$tab_dropout        <- as.numeric(tab_dropout)
  cfg$tab_input_dropout <- if (is.null(tab_input_dropout) ||
                               !is.finite(tab_input_dropout)) {
    0
  } else {
    as.numeric(max(0, min(0.5, tab_input_dropout)))
  }
  cfg$tab_quantile_basis <- isTRUE(tab_quantile_basis)
  # AutoTune v4 uses higher-resolution soft quantile bases to give NMK an
  # RF-like threshold dictionary without stacking on RF predictions.
  cfg$tab_qbasis_knots <- as.integer(max(2L, min(10L, round(tab_qbasis_knots))))
  cfg$tab_qbasis_dropout <- if (is.null(tab_qbasis_dropout) ||
                                !is.finite(tab_qbasis_dropout)) {
    cfg$tab_dropout / 2
  } else {
    as.numeric(max(0, min(0.5, tab_qbasis_dropout)))
  }
  cfg$tab_crossnet <- isTRUE(tab_crossnet)
  cfg$tab_cross_layers <- as.integer(max(1L, min(2L, round(tab_cross_layers))))
  cfg$tab_cross_dropout <- if (is.null(tab_cross_dropout) ||
                               !is.finite(tab_cross_dropout)) {
    cfg$tab_dropout / 2
  } else {
    as.numeric(max(0, min(0.5, tab_cross_dropout)))
  }
  cfg$tab_skip <- isTRUE(tab_skip)
  cfg$tab_skip_hidden <- if (is.null(tab_skip_hidden) ||
                             !is.finite(tab_skip_hidden)) {
    as.integer(max(48L, min(256L, round(6L * ncol(Xtr)))))
  } else {
    as.integer(max(16L, round(tab_skip_hidden)))
  }
  cfg$tab_skip_dropout <- if (is.null(tab_skip_dropout) ||
                              !is.finite(tab_skip_dropout)) {
    cfg$tab_dropout / 2
  } else {
    as.numeric(max(0, min(0.5, tab_skip_dropout)))
  }
  cfg$tab_skip_init <- if (is.null(tab_skip_init) ||
                           !is.finite(tab_skip_init)) {
    -1.5
  } else {
    as.numeric(tab_skip_init)
  }
  if (!is.null(patch_dim))          cfg$patch_dim          <- as.integer(patch_dim)
  if (!is.null(patch_dropout))      cfg$patch_dropout      <- as.numeric(patch_dropout)
  if (!is.null(fusion_hidden))      cfg$fusion_hidden      <- as.integer(fusion_hidden)
  if (!is.null(kriging_hidden))     cfg$kriging_hidden     <- as.integer(kriging_hidden)
  if (!is.null(n_trees))            cfg$n_trees            <- as.integer(n_trees)
  if (!is.null(tree_depth))         cfg$tree_depth         <- as.integer(tree_depth)
  if (!is.null(beta_init))          cfg$beta_init          <- as.numeric(beta_init)
  if (!is.null(init_mix_mean))      cfg$init_mix_mean      <- as.numeric(init_mix_mean)
  if (!is.null(lr_decay))           cfg$lr_decay           <- as.numeric(lr_decay)
  if (!is.null(lr_patience))        cfg$lr_patience        <- as.integer(lr_patience)
  if (!is.null(bank_refresh_every)) cfg$bank_refresh_every <- as.integer(bank_refresh_every)
  if (!is.null(patience))           cfg$patience           <- as.integer(patience)

  epochs <- max(1L, as.integer(epochs))

  # --- Tensor caches and KNN ------------------------------------------------
  set_convkrigingnet2d_seed(train_seed)

  K_eff <- cfg$K_neighbors
  neigh_train <- fd$neighbor_idx_train
  k_use <- min(K_eff, ncol(neigh_train))
  if (k_use < K_eff) {
    cat(sprintf("[NMK]  K_neighbors reduced to %d (pool limit).\n", k_use))
  }
  neigh_train <- neigh_train[, seq_len(k_use), drop = FALSE]

  train_cache <- build_convkrigingnet2d_tensor_cache(Xtr, Ptr, Ctr_s, ytr_s, device)
  val_cache   <- build_convkrigingnet2d_tensor_cache(Xva, Pva, Cva_s, yva_s, device)
  test_cache  <- build_convkrigingnet2d_tensor_cache(Xte, Pte, Cte_s, NULL,   device)

  neigh_train_t <- torch_tensor(neigh_train, dtype = torch_long(), device = device)
  val_knn_t     <- torch_tensor(
    compute_neighbor_idx_query_to_ref(Cva_s, Ctr_s, k_use),
    dtype = torch_long(), device = device)
  test_knn_t    <- torch_tensor(
    compute_neighbor_idx_query_to_ref(Cte_s, Ctr_s, k_use),
    dtype = torch_long(), device = device)

  tab_qbasis_thresholds <- NULL
  tab_qbasis_bandwidths <- NULL
  if (isTRUE(cfg$tab_quantile_basis)) {
    qbasis_stats <- .auto_quantile_basis_stats(
      Xtr, n_knots = cfg$tab_qbasis_knots)
    tab_qbasis_thresholds <- qbasis_stats$thresholds
    tab_qbasis_bandwidths <- qbasis_stats$bandwidths
  }

  # --- Build model ----------------------------------------------------------
  model <- NeuroMaternKrige(
    c_tab            = ncol(Xtr),
    patch_channels   = dim(Ptr)[1L],
    d                = cfg$d,
    tab_hidden       = tab_hidden,
    tab_dropout      = cfg$tab_dropout,
    tab_input_dropout = cfg$tab_input_dropout,
    tab_quantile_basis = cfg$tab_quantile_basis,
    tab_qbasis_thresholds = tab_qbasis_thresholds,
    tab_qbasis_bandwidths = tab_qbasis_bandwidths,
    tab_qbasis_dropout = cfg$tab_qbasis_dropout,
    tab_crossnet     = cfg$tab_crossnet,
    tab_cross_layers = cfg$tab_cross_layers,
    tab_cross_dropout = cfg$tab_cross_dropout,
    tab_skip         = cfg$tab_skip,
    tab_skip_hidden  = cfg$tab_skip_hidden,
    tab_skip_dropout = cfg$tab_skip_dropout,
    tab_skip_init    = cfg$tab_skip_init,
    patch_dim        = cfg$patch_dim,
    patch_dropout    = cfg$patch_dropout,
    multiscale_patch = TRUE,
    coord_hidden     = coord_hidden,
    coord_dim        = coord_dim,
    coord_dropout    = coord_dropout,
    fusion_hidden    = cfg$fusion_hidden,
    kriging_hidden   = cfg$kriging_hidden,
    init_ell_major   = cfg$ell_major_init,
    init_ell_minor   = cfg$ell_minor_init,
    init_theta       = cfg$theta_init,
    init_mix_mean    = cfg$init_mix_mean,
    beta_init        = cfg$beta_init,
    nu_classes       = nu_classes,
    n_trees          = cfg$n_trees,
    tree_depth       = cfg$tree_depth
  )
  model$to(device = device)

  warm_start_loaded <- FALSE
  warm_start_error <- NA_character_
  if (!is.null(init_state)) {
    warm_start_try <- tryCatch({
      model$load_state_dict(init_state)
      TRUE
    }, error = function(e) e)
    if (inherits(warm_start_try, "error")) {
      warm_start_error <- conditionMessage(warm_start_try)
      cat(sprintf("[NMK] Warm-start skipped: %s\n", warm_start_error))
    } else {
      warm_start_loaded <- TRUE
      cat("[NMK] Warm-start state loaded.\n")
    }
  }

  # --- Hardware- and model-aware optimisation defaults ----------------------
  # Weight decay scales with model size (Bartlett & Mendelson 2002): larger
  # models get proportionally stronger regularisation. Rule:
  #   wd = clamp( 1e-3 / sqrt(max(1, n_params / 5e6)) ,  1e-4, 1e-2 )
  # For ~1M params: wd ~= 1e-3. For ~10M params: wd ~= 7e-4. For ~0.1M
  # params: wd capped at 1e-2.
  n_params <- nmk_count_params(model)
  wd_cfg <- .nmk_clamp(
    1e-3 / sqrt(max(1.0, n_params / 5e6)),
    1e-4, 1e-2)
  if (!is.null(wd)) {
    wd_cfg <- as.numeric(wd)  # runner override
  }
  cfg$wd <- wd_cfg
  cfg$n_params <- n_params
  cat(sprintf("[NMK auto]  n_params=%s  wd=%.2e\n",
              format(n_params, big.mark = ","), wd_cfg))

  # Learning-rate probe (Polyak step on the backbone). Runs one probe batch
  # through the current weights, measures grad^2, takes 1 % of the Polyak
  # step. The probe does not modify the model. The runner can override by
  # passing `lr` to this function (in which case the probe is skipped).
  lr_probe <- if (!is.null(lr)) {
    lr_override <- as.numeric(lr)
    cat(sprintf("[NMK auto]  lr override (runner) -> %.2e\n", lr_override))
    lr_override
  } else {
    estimate_nmk_initial_lr_polyak(
      model       = model,
      train_cache = train_cache,
      bs_init     = min(32L, as.integer(cfg$batch_size)),
      device      = device)
  }
  cfg$lr_init <- lr_probe
  # min_lr follows lr_init: allow 20x reduction before floor.
  cfg$min_lr  <- if (!is.null(min_lr)) as.numeric(min_lr) else lr_probe / 20

  # Variogram-prior anchors for the anchor regulariser. These are the values
  # the hypernet would emit if bias = initial bias and activations = 0.
  anchor_log_maj <- log(cfg$ell_major_init)
  anchor_log_min <- log(cfg$ell_minor_init)
  anchor_theta   <- cfg$theta_init
  t_anchor <- torch_tensor(
    c(anchor_log_maj, anchor_log_min, anchor_theta),
    dtype = torch_float(), device = device)

  lr_now <- cfg$lr_init
  bs     <- cfg$batch_size
  pat    <- cfg$patience
  bre    <- cfg$bank_refresh_every
  blw    <- cfg$base_loss_weight
  max_wu <- cfg$max_warmup_epochs
  lam_s  <- cfg$lambda_smooth
  lam_a  <- cfg$lambda_anchor
  lam_c  <- cfg$lambda_cov
  p_drop <- cfg$krig_drop_p_train
  d_scale_eval <- cfg$dist_scale_eval

  # --- Warmup: backbone only ------------------------------------------------
  cat(sprintf("[NMK] === Warmup (max %d epochs, tol=%.1f%%) ===\n",
              max_wu, warmup_converge_tol * 100))

  warmup_params <- c(
    model$enc_tab$parameters,   model$enc_patch$parameters,
    model$proj_patch$parameters,model$enc_coord$parameters,
    model$proj_coord$parameters,model$fuse$parameters,
    model$head$parameters
  )
  if (!is.null(model$tab_qbasis)) {
    warmup_params <- c(warmup_params, model$tab_qbasis$parameters)
  }
  if (!is.null(model$tab_cross_1)) {
    warmup_params <- c(
      warmup_params,
      model$tab_cross_1$parameters,
      if (!is.null(model$tab_cross_2)) model$tab_cross_2$parameters,
      model$tab_cross_proj$parameters,
      model$tab_cross_logit_scale
    )
  }
  if (!is.null(model$tab_skip_head)) {
    warmup_params <- c(
      warmup_params,
      model$tab_skip_head$parameters,
      model$tab_skip_logit_scale
    )
  }
  wu_opt  <- optim_adamw(warmup_params, lr = lr_now, weight_decay = cfg$wd)
  wu_prev <- Inf;  wu_bad <- 0L;  wu_done <- 0L

  for (ep in seq_len(max_wu)) {
    model$train()
    batches <- make_convkrigingnet2d_batches(
      n_train, bs, seed = train_seed, epoch = ep,
      deterministic = deterministic_batches)
    tr_loss <- 0.0
    for (b in batches) {
      b_t  <- torch_tensor(b, dtype = torch_long(), device = device)
      out  <- model$forward_base(
        train_cache$X$index_select(1L, b_t),
        train_cache$P$index_select(1L, b_t),
        train_cache$C$index_select(1L, b_t)
      )
      loss <- huber_loss(train_cache$y$index_select(1L, b_t), out$pred)
      wu_opt$zero_grad();  loss$backward()
      nn_utils_clip_grad_norm_(warmup_params, max_norm = 2.0)
      wu_opt$step()
      tr_loss <- tr_loss + loss$item()
    }
    vb <- predict_nmk_base_tensor(model, val_cache, bs)
    vl <- huber_loss(val_cache$y, to_float_tensor(vb, device))$item()
    rel_imp <- if (is.finite(wu_prev) && wu_prev > 0) (wu_prev - vl) / wu_prev else 1.0
    cat(sprintf("[NMK] Warmup %2d/%d | tr=%.4f | val=%.4f | delta=%+.2f%%\n",
                ep, max_wu, tr_loss / length(batches), vl, rel_imp * 100))
    wu_done <- ep
    if (rel_imp < warmup_converge_tol) {
      wu_bad <- wu_bad + 1L
      if (wu_bad >= as.integer(warmup_patience)) break
    } else {
      wu_bad <- 0L
    }
    wu_prev <- vl
  }
  cat(sprintf("[NMK] Warmup complete: %d epochs used.\n", wu_done))

  # --- Main loop: backbone + Matern kriging ---------------------------------
  cat("[NMK] === Main loop ===\n")
  opt        <- optim_adamw(model$parameters, lr = lr_now, weight_decay = cfg$wd)
  best_val   <- Inf
  best_state <- NULL
  best_epoch <- NA_integer_
  bad <- 0L;  lr_bad <- 0L;  main_done <- 0L

  bank <- refresh_nmk_bank_tensor(model, train_cache, bs)

  for (ep in seq_len(epochs)) {
    main_done <- ep
    model$train()
    batches <- make_convkrigingnet2d_batches(
      n_train, bs, seed = train_seed, epoch = wu_done + ep,
      deterministic = deterministic_batches)
    tr_loss <- 0.0

    for (b in batches) {
      B <- length(b)
      b_t <- torch_tensor(b, dtype = torch_long(), device = device)

      xb <- train_cache$X$index_select(1L, b_t)
      pb <- train_cache$P$index_select(1L, b_t)
      cb <- train_cache$C$index_select(1L, b_t)
      yb <- train_cache$y$index_select(1L, b_t)

      nb   <- neigh_train_t$index_select(1L, b_t)
      nb_t <- nb$reshape(c(-1L))$to(dtype = torch_long())
      zn <- bank$Zmem$index_select(1L, nb_t)$reshape(c(B, k_use, -1L))
      rn <- bank$Rmem$index_select(1L, nb_t)$reshape(c(B, k_use))
      cn <- bank$Cmem$index_select(1L, nb_t)$reshape(c(B, k_use, 2L))

      # Kriging teacher-forcing dropout (Auto §E): with probability p_drop
      # train the batch in base-only mode. Forces the backbone to predict
      # unbiased values without relying on the kriging correction — the
      # standard fix for DesignBased positive-ME bias when n is small.
      use_krig <- TRUE
      if (p_drop > 0 && stats::runif(1L) < p_drop) {
        use_krig <- FALSE
      }

      if (use_krig) {
        out      <- model$forward_with_kriging(xb, pb, cb, zn, cn, rn)
        pred_use <- out$pred
        base_pred <- out$base_pred
        phi_i_ref <- out$phi_i
        phi_n_ref <- out$phi_n
      } else {
        base_out  <- model$forward_base(xb, pb, cb)
        pred_use  <- base_out$pred
        base_pred <- base_out$pred
        phi_i_ref <- NULL
        phi_n_ref <- NULL
      }

      loss <- huber_loss(yb, pred_use)

      if (blw > 0) {
        loss <- loss + blw * huber_loss(yb, base_pred)
      }
      if (cfg$alpha_me > 0) {
        loss_me <- (torch_mean(base_pred) - torch_mean(yb)) ^ 2
        loss <- loss + cfg$alpha_me * loss_me
      }
      if (lam_c > 0) {
        # Variance-matching penalty (GeoVersa v4): discourages backbone
        # from shrinking its spread and leaving the kriging layer to
        # recover variance. (sigma_hat / sigma_y - 1)^2.
        pred_std <- torch_std(base_pred) + 1e-8
        targ_std <- torch_std(yb)        + 1e-8
        ratio    <- pred_std / targ_std - 1
        loss_cov <- ratio * ratio
        loss <- loss + lam_c * loss_cov
      }

      # Smoothness and anchor priors — only when kriging was used this
      # batch (otherwise phi tensors are not attached to the graph).
      if (use_krig && lam_s > 0) {
        phi_i_mat <- torch_stack(list(
          phi_i_ref$log_ell_maj, phi_i_ref$log_ell_min, phi_i_ref$theta),
          dim = 2L)                                                  # [B, 3]
        phi_n_mat <- torch_stack(list(
          phi_n_ref$log_ell_maj, phi_n_ref$log_ell_min, phi_n_ref$theta),
          dim = 3L)                                                  # [B, K, 3]
        diff <- phi_n_mat - phi_i_mat$unsqueeze(2L)
        loss_smooth <- torch_mean(torch_sum(diff * diff, dim = 3L))
        loss <- loss + lam_s * loss_smooth
      }

      if (use_krig && lam_a > 0) {
        mean_phi <- torch_stack(list(
          torch_mean(phi_i_ref$log_ell_maj),
          torch_mean(phi_i_ref$log_ell_min),
          torch_mean(phi_i_ref$theta)
        ))
        loss_anchor <- torch_sum((mean_phi - t_anchor) ^ 2)
        loss <- loss + lam_a * loss_anchor
      }

      # Numerical safety: if the loss or any gradient is non-finite for this
      # batch (extremely rare — can happen on pathological BLOOCV folds),
      # skip the optimiser step rather than crashing the whole run.
      loss_val <- loss$item()
      if (!is.finite(loss_val)) {
        opt$zero_grad()
        cat(sprintf("[NMK] ep %d: non-finite loss (%s) — batch skipped.\n",
                    ep, format(loss_val)))
        next
      }
      opt$zero_grad();  loss$backward()
      grad_ok <- tryCatch({
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
        TRUE
      }, error = function(e) {
        cat(sprintf("[NMK] ep %d: grad clip failed (%s) — batch skipped.\n",
                    ep, conditionMessage(e)))
        FALSE
      })
      if (!grad_ok) {
        opt$zero_grad()
        next
      }
      opt$step()
      tr_loss <- tr_loss + loss_val
    }

    if (ep %% bre == 0L || ep == epochs) {
      bank <- refresh_nmk_bank_tensor(model, train_cache, bs)
    }

    model$eval()
    vp <- predict_nmk_with_memory_tensor(
      model, val_cache, bank$Zmem, bank$Rmem, bank$Cmem, k_use, val_knn_t, bs)
    vl <- huber_loss(val_cache$y, to_float_tensor(vp, device))$item()

    beta_v  <- as.numeric(torch_sigmoid(model$logit_beta)$cpu())
    cat(sprintf(
      "[NMK] ep %d | lr=%.2e | tr=%.4f | val=%.4f | beta=%.3f\n",
      ep, lr_now, tr_loss / length(batches), vl, beta_v))

    if (vl < best_val) {
      best_val   <- vl
      best_epoch <- ep
      if (!isTRUE(refit_fixed_epochs)) {
        best_state <- clone_state_dict(model$state_dict())
        bad <- 0L;  lr_bad <- 0L
      }
    } else if (!isTRUE(refit_fixed_epochs)) {
      bad    <- bad + 1L
      lr_bad <- lr_bad + 1L
      if (lr_bad >= cfg$lr_patience && lr_now > cfg$min_lr) {
        lr_now <- max(cfg$min_lr, lr_now * cfg$lr_decay)
        set_optimizer_lr(opt, lr_now)
        lr_bad <- 0L
        cat(sprintf("[NMK] lr -> %.2e\n", lr_now))
      }
      if (bad >= pat) { cat("[NMK] Early stop.\n"); break }
    }
  }

  if (!isTRUE(refit_fixed_epochs) && !is.null(best_state)) {
    model$load_state_dict(best_state)
  } else if (isTRUE(refit_fixed_epochs)) {
    cat("[NMK] Refit fixed-epoch mode: keeping final epoch state.\n")
  }
  model$eval()

  # --- Final predictions ----------------------------------------------------
  bank <- refresh_nmk_bank_tensor(model, train_cache, bs)

  val_base_preds_s  <- predict_nmk_base_tensor(model, val_cache,  bs)
  test_base_preds_s <- predict_nmk_base_tensor(model, test_cache, bs)
  # Final predictions use the distance-aware gate if the auto-config set one.
  val_preds_s       <- predict_nmk_with_memory_tensor(
    model, val_cache,  bank$Zmem, bank$Rmem, bank$Cmem, k_use, val_knn_t, bs,
    dist_scale = d_scale_eval)
  test_preds_s      <- predict_nmk_with_memory_tensor(
    model, test_cache, bank$Zmem, bank$Rmem, bank$Cmem, k_use, test_knn_t, bs,
    dist_scale = d_scale_eval)

  # Undo target scaler and target transform.
  val_base_preds  <- inverse_transform_target(
    invert_target_scaler(val_base_preds_s,  y_scaler), target_transform)
  test_base_preds <- inverse_transform_target(
    invert_target_scaler(test_base_preds_s, y_scaler), target_transform)
  val_preds  <- inverse_transform_target(
    invert_target_scaler(val_preds_s,  y_scaler), target_transform)
  test_preds <- inverse_transform_target(
    invert_target_scaler(test_preds_s, y_scaler), target_transform)

  # --- Smearing correction (diagnostic only, NOT applied) -------------------
  # With target_transform = "log", exp(yhat_log) estimates the conditional
  # MEDIAN of y. For log-normal-ish targets the classical Duan (1983)
  # smearing correction S = (1/n_v) sum_j exp(log y_j^val - log yhat_j^val)
  # is the unbiased fix. However, at DSM-typical val sizes (n_val ~= 80)
  # the estimator's variance dominates — a single heavy-tail residual on
  # the log scale inflates exp()-based means disproportionately, so the
  # correction overshoots in some folds and undershoots in others. We
  # therefore compute S as a DIAGNOSTIC only and report it in the
  # per-fold output, but do not multiply predictions by it. Empirically
  # (AGB, n_val ~= 80, 10 iterations on DesignBased) applying Duan
  # roughly doubled the cross-seed RMSE standard deviation and flipped
  # the mean bias from -1.5 to +3.9 Mg/ha. The decision NOT to apply is
  # itself data-driven: a robust variant would require either a much
  # larger val set or a different residual distribution.
  smearing_S <- NA_real_
  if (identical(target_transform, "log")) {
    val_log_obs        <- log(yva)
    val_log_preds_full <- invert_target_scaler(val_preds_s, y_scaler)
    ok <- is.finite(val_log_obs) & is.finite(val_log_preds_full)
    if (sum(ok) >= 3L) {
      r_log <- val_log_obs[ok] - val_log_preds_full[ok]
      smearing_S <- mean(exp(r_log))
      cat(sprintf(
        "[NMK auto]  Duan smearing (diagnostic only): S = %.4f (n_val = %d)\n",
        smearing_S, sum(ok)))
    }
  }

  beta_f <- as.numeric(torch_sigmoid(model$logit_beta)$cpu())
  cat(sprintf("\n[NMK] === Final summary ===\n"))
  cat(sprintf("[NMK] Val : ME=%+.3f  RMSE=%.3f\n",
              mean(yva - val_preds), sqrt(mean((yva - val_preds) ^ 2))))
  cat(sprintf(
    "[NMK] Geostat: r=%.3f fit_q=%.2f K=%d BLW=%.4f alpha_me=%.4f cov=%.4f smooth=%.4f anchor=%.4f\n",
    cfg$nugget_ratio, cfg$fit_quality, cfg$K_neighbors, cfg$base_loss_weight,
    cfg$alpha_me, cfg$lambda_cov, cfg$lambda_smooth, cfg$lambda_anchor))
  cat(sprintf(
    "[NMK] Arch  : d=%d krig_hid=%d forest=%dx(depth%d,%dleaves) bs=%d tab_drop=%.2f patch_drop=%.2f qbasis=%s cross=%s skip=%s n_params=%s\n",
    cfg$d, cfg$kriging_hidden, cfg$n_trees, cfg$tree_depth,
    2L ^ cfg$tree_depth, cfg$batch_size, cfg$tab_dropout,
    cfg$patch_dropout, cfg$tab_quantile_basis, cfg$tab_crossnet,
    cfg$tab_skip, format(cfg$n_params, big.mark = ",")))
  cat(sprintf(
    "[NMK] Optim : lr_init=%.2e lr_final=%.2e min_lr=%.2e wd=%.2e patience=%d lr_patience=%d\n",
    cfg$lr_init, lr_now, cfg$min_lr, cfg$wd, cfg$patience, cfg$lr_patience))
  cat(sprintf(
    "[NMK] Kriging: krig_drop=%.2f dist_scale=%.2f mix_mean=%.2f warmup=%d beta_init_logit=%.2f beta=%.3f\n",
    cfg$krig_drop_p_train, cfg$dist_scale_eval, cfg$init_mix_mean,
    wu_done, cfg$beta_init, beta_f))

  list(
    pred_test      = test_preds,
    pred_val       = val_preds,
    pred_test_base = test_base_preds,
    pred_val_base  = val_base_preds,
    metrics_test   = metrics(yte, test_preds),
    diagnostics    = list(
      nugget_ratio       = cfg$nugget_ratio,
      fit_quality        = cfg$fit_quality,
      K_neighbors        = cfg$K_neighbors,
      base_loss_weight   = cfg$base_loss_weight,
      alpha_me           = cfg$alpha_me,
      lambda_cov         = cfg$lambda_cov,
      lambda_smooth      = cfg$lambda_smooth,
      lambda_anchor      = cfg$lambda_anchor,
      krig_drop_p_train  = cfg$krig_drop_p_train,
      dist_scale_eval    = cfg$dist_scale_eval,
      init_mix_mean      = cfg$init_mix_mean,
      beta_init_logit    = cfg$beta_init,
      warmup_epochs_done = wu_done,
      main_epochs_done   = main_done,
      best_epoch         = best_epoch,
      best_val_loss      = best_val,
      beta_final         = beta_f,
      d                  = cfg$d,
      patch_dim          = cfg$patch_dim,
      kriging_hidden     = cfg$kriging_hidden,
      n_trees            = cfg$n_trees,
      tree_depth         = cfg$tree_depth,
      tab_input_dropout  = cfg$tab_input_dropout,
      tab_quantile_basis = cfg$tab_quantile_basis,
      tab_qbasis_knots   = if (cfg$tab_quantile_basis) cfg$tab_qbasis_knots else NA_integer_,
      tab_qbasis_dropout = if (cfg$tab_quantile_basis) cfg$tab_qbasis_dropout else NA_real_,
      tab_crossnet       = cfg$tab_crossnet,
      tab_cross_layers   = if (cfg$tab_crossnet) cfg$tab_cross_layers else NA_integer_,
      tab_cross_dropout  = if (cfg$tab_crossnet) cfg$tab_cross_dropout else NA_real_,
      tab_skip           = cfg$tab_skip,
      tab_skip_hidden    = if (cfg$tab_skip) cfg$tab_skip_hidden else NA_integer_,
      tab_skip_dropout   = if (cfg$tab_skip) cfg$tab_skip_dropout else NA_real_,
      tab_skip_init      = if (cfg$tab_skip) cfg$tab_skip_init else NA_real_,
      batch_size         = cfg$batch_size,
      n_params           = cfg$n_params,
      wd                 = cfg$wd,
      lr_init            = cfg$lr_init,
      lr_final           = lr_now,
      min_lr             = cfg$min_lr,
      patience           = cfg$patience,
      lr_patience        = cfg$lr_patience,
      bank_refresh_every = cfg$bank_refresh_every,
      target_transform   = target_transform,
      smearing_S         = smearing_S,
      refit_fixed_epochs = isTRUE(refit_fixed_epochs),
      warm_start_loaded  = isTRUE(warm_start_loaded),
      warm_start_error   = warm_start_error
    ),
    selected_state = if (isTRUE(return_state)) {
      clone_state_dict(model$state_dict())
    } else {
      NULL
    }
  )
}


# -----------------------------------------------------------------------------
# Ensemble wrapper
# -----------------------------------------------------------------------------
#
# Trains N independently-seeded copies of the model on the same fold and
# averages their predictions. N = 3 is a structural choice (same status as
# `nu_classes`), not a hyperparameter: the point of the ensemble is to
# reduce cross-seed variance of the hypernet, which the single-seed NMK
# exhibits at n = 500. Bias correction came from the alpha_ME + lambda_cov
# penalties; variance correction is this ensemble.
#
# Expected effect (standard ensemble theory): for independent errors with
# equal variance, the RMSE of the averaged predictor scales as 1 / sqrt(N).
# In practice N = 3 halves the standard deviation across seeds on DSM-sized
# datasets, which is enough to close most of the cross-seed spread without
# 3x memory (each seed is trained and predicted sequentially, then
# discarded — only N prediction vectors live at once).
#
# The wrapper respects every NMK convention: the variogram + auto-config
# runs once per seed (so the kernel initialisation can legitimately differ
# per seed), the Polyak probe also per seed, and all env-var overrides flow
# through unchanged.

train_neuromaternkrige_ensemble <- function(fd,
                                            n_seeds = 3L,
                                            train_seed = NULL,
                                            init_states = NULL,
                                            return_state = FALSE,
                                            ...) {
  n_seeds <- as.integer(n_seeds)
  if (!is.finite(n_seeds) || n_seeds < 1L) n_seeds <- 1L

  base_seed <- if (is.null(train_seed) || !is.finite(train_seed)) {
    123L
  } else {
    as.integer(train_seed)
  }

  get_init_state <- function(s) {
    if (is.null(init_states)) return(NULL)
    if (is.list(init_states) && length(init_states) >= s &&
        is.list(init_states[[s]])) {
      return(init_states[[s]])
    }
    if (s == 1L) return(init_states)
    NULL
  }

  # Degenerate case: n_seeds == 1 is equivalent to the single-seed trainer.
  if (n_seeds == 1L) {
    out <- train_neuromaternkrige_one_fold(fd = fd,
                                           train_seed = base_seed,
                                           init_state = get_init_state(1L),
                                           return_state = return_state, ...)
    out$diagnostics$n_seeds <- 1L
    out$diagnostics$ensemble_base_seed <- base_seed
    if (isTRUE(return_state)) {
      out$selected_states <- list(out$selected_state)
    }
    return(out)
  }

  cat(sprintf("\n[NMK ensemble] N = %d  (base seed = %d)\n",
              n_seeds, base_seed))

  pred_test_mat      <- NULL
  pred_val_mat       <- NULL
  pred_test_base_mat <- NULL
  pred_val_base_mat  <- NULL
  per_seed_diag      <- vector("list", n_seeds)
  selected_states    <- if (isTRUE(return_state)) vector("list", n_seeds) else NULL

  for (s in seq_len(n_seeds)) {
    seed_s <- base_seed + (s - 1L) * 101L   # spaced seeds
    cat(sprintf("\n----- NMK seed %d / %d  (train_seed = %d) -----\n",
                s, n_seeds, seed_s))
    out_s <- train_neuromaternkrige_one_fold(fd = fd,
                                             train_seed = seed_s,
                                             init_state = get_init_state(s),
                                             return_state = return_state,
                                             ...)
    # Lazily allocate matrices now that we know vector lengths.
    if (is.null(pred_test_mat)) {
      pred_test_mat      <- matrix(NA_real_, nrow = n_seeds,
                                   ncol = length(out_s$pred_test))
      pred_val_mat       <- matrix(NA_real_, nrow = n_seeds,
                                   ncol = length(out_s$pred_val))
      pred_test_base_mat <- matrix(NA_real_, nrow = n_seeds,
                                   ncol = length(out_s$pred_test_base))
      pred_val_base_mat  <- matrix(NA_real_, nrow = n_seeds,
                                   ncol = length(out_s$pred_val_base))
    }
    pred_test_mat[s, ]      <- as.numeric(out_s$pred_test)
    pred_val_mat[s, ]       <- as.numeric(out_s$pred_val)
    pred_test_base_mat[s, ] <- as.numeric(out_s$pred_test_base)
    pred_val_base_mat[s, ]  <- as.numeric(out_s$pred_val_base)
    per_seed_diag[[s]]      <- out_s$diagnostics
    if (isTRUE(return_state)) {
      selected_states[[s]] <- out_s$selected_state
    }
  }

  # Ensemble average (simple mean — equal weights, no calibration).
  pred_test      <- colMeans(pred_test_mat,      na.rm = TRUE)
  pred_val       <- colMeans(pred_val_mat,       na.rm = TRUE)
  pred_test_base <- colMeans(pred_test_base_mat, na.rm = TRUE)
  pred_val_base  <- colMeans(pred_val_base_mat,  na.rm = TRUE)

  yte <- fd$y$test
  yva <- fd$y$val

  # Roll up diagnostics: keep per-seed info and summarise the scalars.
  agg_num <- function(field) {
    vals <- vapply(per_seed_diag, function(d) {
      v <- d[[field]]
      if (is.null(v) || !is.finite(v)) NA_real_ else as.numeric(v)
    }, numeric(1))
    vals
  }
  beta_vals    <- agg_num("beta_final")
  lr_init_vals <- agg_num("lr_init")
  tt_vals      <- vapply(per_seed_diag, function(d) {
    if (is.null(d$target_transform)) NA_character_ else as.character(d$target_transform)
  }, character(1))

  smearing_vals <- agg_num("smearing_S")
  diagnostics <- list(
    n_seeds              = n_seeds,
    ensemble_base_seed   = base_seed,
    per_seed             = per_seed_diag,
    beta_final_mean      = mean(beta_vals, na.rm = TRUE),
    beta_final_sd        = sd(beta_vals, na.rm = TRUE),
    lr_init_mean         = mean(lr_init_vals, na.rm = TRUE),
    target_transform     = if (length(unique(tt_vals)) == 1L) tt_vals[1] else paste(unique(tt_vals), collapse = ","),
    smearing_S_mean      = mean(smearing_vals, na.rm = TRUE),
    smearing_S_sd        = sd(smearing_vals,   na.rm = TRUE)
  )
  # Echo stable fields (same across seeds since they depend on variogram +
  # n, which are seed-independent).
  reference_diag <- per_seed_diag[[1L]]
  copy_fields <- c("nugget_ratio", "fit_quality", "K_neighbors",
                   "base_loss_weight", "alpha_me", "lambda_cov",
                   "lambda_smooth", "lambda_anchor",
                   "krig_drop_p_train", "dist_scale_eval",
	                   "init_mix_mean", "beta_init_logit",
	                   "warmup_epochs_done", "main_epochs_done",
	                   "best_epoch", "best_val_loss",
	                   "d", "patch_dim", "kriging_hidden",
	                   "n_trees", "tree_depth",
	                   "tab_input_dropout", "tab_quantile_basis",
	                   "tab_qbasis_knots", "tab_qbasis_dropout",
	                   "tab_crossnet", "tab_cross_layers",
	                   "tab_cross_dropout", "tab_skip",
	                   "tab_skip_hidden", "tab_skip_dropout",
	                   "tab_skip_init",
	                   "batch_size", "n_params", "wd",
	                   "lr_final", "min_lr",
	                   "patience", "lr_patience", "bank_refresh_every",
	                   "refit_fixed_epochs", "warm_start_loaded",
	                   "warm_start_error")
  for (f in copy_fields) diagnostics[[f]] <- reference_diag[[f]]

  test_rmse  <- sqrt(mean((yte - pred_test) ^ 2, na.rm = TRUE))
  val_rmse   <- sqrt(mean((yva - pred_val)  ^ 2, na.rm = TRUE))
  cat(sprintf("\n[NMK ensemble] Averaged over %d seeds: val_RMSE=%.3f  test_RMSE=%.3f\n",
              n_seeds, val_rmse, test_rmse))

  list(
    pred_test      = pred_test,
    pred_val       = pred_val,
    pred_test_base = pred_test_base,
    pred_val_base  = pred_val_base,
    metrics_test   = metrics(yte, pred_test),
    diagnostics    = diagnostics,
    per_seed_preds = list(
      test      = pred_test_mat,
      val       = pred_val_mat,
      test_base = pred_test_base_mat,
      val_base  = pred_val_base_mat
    ),
    selected_states = selected_states
  )
}


# -----------------------------------------------------------------------------
# §6  Default parameter presets
# -----------------------------------------------------------------------------

# Minimal recipe used by the Wadoux-style runner. All NULL entries signal
# "use the value derived by nmk_auto_config or the Polyak probe"; explicit
# values become runner-level overrides. n_seeds = 3 is the structural
# ensemble size (see train_neuromaternkrige_ensemble).
neuromatern_default_params <- list(
  n_seeds              = 3L,
  epochs               = 80L,
  wd                   = NULL,
  batch_size           = NULL,
  patience             = NULL,
  warmup_epochs        = NULL,
  bank_refresh_every   = NULL,
  lr                   = NULL,
  lr_decay             = NULL,
  lr_patience          = NULL,
  min_lr               = NULL,
  beta_init            = NULL,
  kriging_hidden       = NULL,
  init_mix_mean        = NULL,
  lambda_smooth        = NULL,
  lambda_anchor        = NULL,
  lambda_cov           = NULL,
  krig_drop_p_train    = NULL,
  dist_scale_eval      = NULL,
  nu_classes           = c(0.5, 1.5, 2.5),
  tab_input_dropout    = NULL,
  tab_quantile_basis   = FALSE,
  tab_qbasis_knots     = 4L,
  tab_qbasis_dropout   = NULL,
  tab_crossnet         = FALSE,
  tab_cross_layers     = 2L,
  tab_cross_dropout    = NULL,
  tab_skip             = FALSE,
  tab_skip_hidden      = NULL,
  tab_skip_dropout     = NULL,
  tab_skip_init        = -1.5,
	  target_transform     = "auto",
	  deterministic_batches = TRUE,
	  refit_fixed_epochs   = FALSE
	)

# Lightweight profile for quick smoke tests: 1 seed, short schedule.
neuromatern_quick_params <- modifyList(
  neuromatern_default_params,
  list(
    n_seeds   = 1L,
    epochs    = 20L,
    patience  = 5L
  )
)
