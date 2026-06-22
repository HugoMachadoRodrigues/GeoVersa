# =============================================================================
# DeepKernelProcess.R
#
# Deep Kernel Process (DKP) for Digital Soil Mapping.
#
# Novelty (unprecedented in Pedometria as far as the literature review
# shows): a Gaussian Process with a COMPOSITE kernel whose two factors are
# both learned end-to-end, one in feature space (deep kernel learning;
# Wilson 2016) and one in physical space (non-stationary Matern with
# hypernet-emitted parameters):
#
#       K_comb(i, j) = K_feat(h_i, h_j)  *  K_spat(s_i, s_j | h_i, h_j)
#
# with
#   h_i         = Encoder_theta(x_i, Patch_i)                (shared encoder)
#   K_feat(i,j) = exp( -||h_i - h_j||^2 / (2 sigma1^2) )     (learned RBF)
#   K_spat(i,j) = Matern_nu( d*(s_i, s_j; phi_i, phi_j) )    (NS Matern)
#   phi_i       = hypernet(h_i)    (log-ell-major, log-ell-minor, theta, pi)
#
# Predictions are the exact GP posterior mean:
#   yhat_* = mu0 + K_*tr ( K_tr + sigma_n^2 I )^{-1} ( y_tr - mu0 )
#
# Training maximises the log marginal likelihood via Cholesky on the full
# training kernel matrix. With DSM-scale n (500-5000), O(n^3) Cholesky is
# completely practical (tens of ms per step on MPS / CUDA).
#
# This is not the NeuroMaternKrige residual-kriging architecture. It is a
# distinct model: the composite kernel IS the predictor, there is no
# "backbone + correction" split, there is no memory bank — the training
# set IS the memory bank.
#
# Dependencies (must be sourced before this file):
#   - code/ConvKrigingNet2D.R        (encoders, MultiScalePatchEncoder2D,
#                                     make_mlp, tensor-cache helpers,
#                                     scalers, target transform)
#   - code/ConvKrigingNet2D_Auto.R   (fit_variogram_auto)
#   - code/NeuroMaternKrige.R        (.nmk_clamp, .nmk_env_num,
#                                     .nmk_inv_softplus,
#                                     .nmk_target_transform_auto,
#                                     nmk_count_params)
#
# =============================================================================
# Hyperparameter deduction chain — same five-class skeleton as NMK, plus a
# GP-specific (F) class for the GP noise variance.
#
# (A) VARIOGRAM-DRIVEN:
#       K kernel-init params, ell_maj, ell_min, theta (same as NMK);
#       fit_quality -> init_mix_mean;
#       r = nugget_ratio -> noise_init = clamp(r, 0.02, 0.5)   (class F);
#       Matern mix weights as in NMK.
#
# (B) STATISTICAL-CAPACITY. n_train -> d, patch_dim, kriging_hidden, batch_size
#       (batch_size = n_train for full-batch GP training);
#       feature_kernel_init_scale = "median heuristic" (see §5).
#
# (C) SCHEDULE. max_warmup_epochs W -> patience, lr_patience, lr_decay.
#
# (D) OPTIMISATION PROBE. Polyak-style probe on the NEG LOG MARGINAL
#       LIKELIHOOD (not Huber), giving a principled lr_init.
#
# (E) TARGET PREP. Auto log/identity from Shapiro-Wilk + positivity (same
#       helper as NMK). The GP assumes Gaussian residuals on the chosen
#       scale — log transform is especially natural for skewed targets.
#
# (F) NOISE. sigma_n^2 initialised from r; jitter = 1e-4 (stability).
# =============================================================================


# -----------------------------------------------------------------------------
# §0  Setup — reuse helpers from NeuroMaternKrige.R
# -----------------------------------------------------------------------------

# When sourced independently the host env may not expose the NMK helpers
# yet. Guard by testing for them and stop cleanly if missing; in the
# intended loader flow ConvKrigingNet2D_Auto.R + NeuroMaternKrige.R are
# sys.sourced before this file.
if (!exists(".nmk_clamp", mode = "function")) {
  stop("DeepKernelProcess.R requires NeuroMaternKrige.R helpers. Source NMK first.")
}


# -----------------------------------------------------------------------------
# §1  NonStationaryMaternKernelMatrix
# -----------------------------------------------------------------------------
#
# Same hypernet as NMK (decoded (log_ell_maj, log_ell_min, theta, pi)) but
# producing a FULL N1 x N2 kernel matrix, not a K-NN aggregation. Used as
# the spatial factor K_spat of the composite kernel.

NonStationaryMaternKernelMatrix <- nn_module(
  "NonStationaryMaternKernelMatrix",

  initialize = function(d_in,
                        hidden_dim     = 64L,
                        init_ell_major = 1.0,
                        init_ell_minor = 0.7,
                        init_theta     = 0.0,
                        init_mix_mean  = 1.5,
                        theta_max      = pi / 2,
                        nu_classes     = c(0.5, 1.5, 2.5)) {
    self$d_in       <- as.integer(d_in)
    self$hidden_dim <- as.integer(hidden_dim)
    self$theta_max  <- as.numeric(theta_max)
    self$nu_classes <- as.numeric(nu_classes)
    n_nu            <- length(self$nu_classes)
    self$out_dim    <- as.integer(3L + n_nu)

    self$hyper <- nn_sequential(
      nn_linear(self$d_in, self$hidden_dim),
      nn_gelu(),
      nn_layer_norm(self$hidden_dim),
      nn_linear(self$hidden_dim, self$hidden_dim),
      nn_gelu(),
      nn_linear(self$hidden_dim, self$out_dim)
    )

    # Variogram-prior bias initialisation (same derivation as NMK).
    last_linear <- self$hyper[[length(self$hyper)]]
    with_no_grad({
      last_linear$weight$zero_()
      ell_min_safe <- max(init_ell_minor, 1e-4)
      ell_diff     <- max(init_ell_major - init_ell_minor, 1e-4)
      theta_init   <- init_theta / max(self$theta_max, 1e-6)
      theta_init   <- min(max(theta_init, -0.999), 0.999)
      theta_raw    <- 0.5 * log((1 + theta_init) / (1 - theta_init))
      bias_vec <- c(
        log(ell_min_safe),
        .nmk_inv_softplus(ell_diff),
        theta_raw,
        rep(0, length(self$nu_classes))
      )
      mix_logit <- -((self$nu_classes - init_mix_mean) ^ 2) * 4
      bias_vec[(3L + 1L):(3L + length(self$nu_classes))] <- mix_logit
      last_linear$bias$copy_(torch_tensor(bias_vec, dtype = torch_float()))
    })
  },

  # Emit (log_ell_maj, log_ell_min, theta, mix) per sample.
  decode_params = function(h) {
    raw <- self$hyper(h)  # [N, out_dim]
    n_nu <- length(self$nu_classes)

    log_emin  <- raw$narrow(2L, 1L, 1L)$squeeze(2L)
    delta_raw <- raw$narrow(2L, 2L, 1L)$squeeze(2L)
    theta_raw <- raw$narrow(2L, 3L, 1L)$squeeze(2L)
    mix_raw   <- raw$narrow(2L, 4L, n_nu)

    ell_minor   <- nnf_softplus(log_emin) + 1e-4
    ell_major   <- ell_minor + nnf_softplus(delta_raw) + 1e-6
    log_ell_maj <- torch_log(ell_major)
    log_ell_min <- torch_log(ell_minor)
    theta       <- self$theta_max * torch_tanh(theta_raw)
    mix         <- nnf_softmax(mix_raw, dim = 2L)

    list(
      log_ell_maj = log_ell_maj,
      log_ell_min = log_ell_min,
      theta       = theta,
      mix         = mix
    )
  },

  # Full pairwise kernel matrix K2 between (h1, s1) and (h2, s2).
  #   h1: [N1, d_in],  s1: [N1, 2]
  #   h2: [N2, d_in],  s2: [N2, 2]
  # Returns [N1, N2].
  forward = function(h1, s1, h2, s2) {
    p1 <- self$decode_params(h1)   # N1 entries, scalars or [N1, n_nu] for mix
    p2 <- self$decode_params(h2)   # N2 entries

    # Stein-Paciorek midpoint averaging (valid non-stationary kernel).
    log_maj_ij <- 0.5 * (p1$log_ell_maj$unsqueeze(2L) +
                          p2$log_ell_maj$unsqueeze(1L))           # [N1, N2]
    log_min_ij <- 0.5 * (p1$log_ell_min$unsqueeze(2L) +
                          p2$log_ell_min$unsqueeze(1L))
    theta_ij   <- 0.5 * (p1$theta$unsqueeze(2L) +
                          p2$theta$unsqueeze(1L))
    mix_ij     <- 0.5 * (p1$mix$unsqueeze(2L) +
                          p2$mix$unsqueeze(1L))                   # [N1, N2, n_nu]

    ell_maj_ij <- torch_exp(log_maj_ij)
    ell_min_ij <- torch_exp(log_min_ij)

    # Pairwise coordinate differences.
    dx <- s1[, 1]$unsqueeze(2L) - s2[, 1]$unsqueeze(1L)            # [N1, N2]
    dy <- s1[, 2]$unsqueeze(2L) - s2[, 2]$unsqueeze(1L)
    cth <- torch_cos(theta_ij)
    sth <- torch_sin(theta_ij)
    u   <-  cth * dx + sth * dy
    v   <- -sth * dx + cth * dy
    d_star <- torch_sqrt((u / ell_maj_ij) ^ 2 +
                          (v / ell_min_ij) ^ 2 + 1e-8)            # [N1, N2]

    # Closed-form Matern cases.
    sqrt3 <- sqrt(3)
    sqrt5 <- sqrt(5)
    k_half     <- torch_exp(-d_star)
    k_three_h  <- (1 + sqrt3 * d_star) * torch_exp(-sqrt3 * d_star)
    k_five_h   <- (1 + sqrt5 * d_star + (5 / 3) * d_star ^ 2) *
                  torch_exp(-sqrt5 * d_star)

    stack_list <- list()
    for (i in seq_along(self$nu_classes)) {
      nu_i <- self$nu_classes[i]
      if (abs(nu_i - 0.5) < 1e-6)       stack_list[[i]] <- k_half
      else if (abs(nu_i - 1.5) < 1e-6)  stack_list[[i]] <- k_three_h
      else if (abs(nu_i - 2.5) < 1e-6)  stack_list[[i]] <- k_five_h
      else stop(sprintf("DKP: Matern nu = %g not in closed-form set.", nu_i))
    }
    k_stack <- torch_stack(stack_list, dim = -1L)                  # [N1, N2, n_nu]

    # Convex mixture.
    torch_sum(mix_ij * k_stack, dim = -1L)                         # [N1, N2]
  }
)


# -----------------------------------------------------------------------------
# §2  DeepKernelProcessGP — the full model
# -----------------------------------------------------------------------------

DeepKernelProcessGP <- nn_module(
  "DeepKernelProcessGP",

  initialize = function(c_tab,
                        patch_channels,
                        d                = 128L,
                        tab_hidden       = c(192L),
                        tab_dropout      = 0.10,
                        patch_dim        = 64L,
                        patch_dropout    = 0.08,
                        multiscale_patch = TRUE,
                        fusion_hidden    = 128L,
                        matern_hidden    = 64L,
                        init_ell_major   = 1.0,
                        init_ell_minor   = 0.7,
                        init_theta       = 0.0,
                        init_mix_mean    = 1.5,
                        init_log_sigma1  = 0.0,
                        init_log_noise   = -2.0,
                        init_prior_mean  = 0.0,
                        nu_classes       = c(0.5, 1.5, 2.5),
                        jitter           = 1e-3,
                        log_noise_floor  = -8.0) {
    # jitter: constant added to the diagonal of K_noise (floor on the GP
    # "noise" kernel contribution). 1e-3 is conservative for DSM-sized
    # calibrations (n <= 5000) — enough to absorb float32 spectral noise
    # from the Matern kernel without distorting the model significantly.
    # log_noise_floor: lower clamp on the learnable log_noise parameter so
    # the GP cannot collapse to a zero-noise regime (which would make K
    # rank-deficient whenever two calibration points are near-duplicates
    # in covariate + spatial space). -8 => noise >= exp(-8) ~ 3.4e-4.
    self$d <- as.integer(d)
    self$jitter <- as.numeric(jitter)
    self$log_noise_floor <- as.numeric(log_noise_floor)

    self$enc_tab <- make_mlp(c_tab, hidden = tab_hidden,
                             out_dim = self$d, dropout = tab_dropout)

    if (isTRUE(multiscale_patch)) {
      self$enc_patch <- MultiScalePatchEncoder2D(
        in_channels = patch_channels,
        out_dim     = patch_dim,
        dropout     = patch_dropout)
    } else {
      self$enc_patch <- PatchEncoder2D(
        in_channels = patch_channels,
        out_dim     = patch_dim,
        dropout     = patch_dropout)
    }
    self$proj_patch <- nn_linear(patch_dim, self$d)

    # Fuse tabular + patch encodings into a single feature vector h.
    # Coordinates are NOT fused into h — they feed the spatial kernel
    # factor K_spat directly, so K_feat is a pure environmental kernel.
    self$fuse <- nn_sequential(
      nn_linear(2L * self$d, fusion_hidden),
      nn_gelu(),
      nn_dropout(0.10),
      nn_linear(fusion_hidden, self$d)
    )

    # Feature kernel bandwidth (log-parameterised).
    self$log_sigma1 <- nn_parameter(torch_tensor(init_log_sigma1))

    # Spatial non-stationary Matern kernel matrix.
    self$spatial_kernel <- NonStationaryMaternKernelMatrix(
      d_in           = self$d,
      hidden_dim     = matern_hidden,
      init_ell_major = init_ell_major,
      init_ell_minor = init_ell_minor,
      init_theta     = init_theta,
      init_mix_mean  = init_mix_mean,
      nu_classes     = nu_classes
    )

    # Noise variance (log) and prior mean mu_0.
    self$log_noise  <- nn_parameter(torch_tensor(init_log_noise))
    self$prior_mean <- nn_parameter(torch_tensor(init_prior_mean))
  },

  encode = function(x_tab, x_patch) {
    h_tab   <- self$enc_tab(x_tab)
    h_patch <- self$proj_patch(self$enc_patch(x_patch))
    self$fuse(torch_cat(list(h_tab, h_patch), dim = 2L))
  },

  feature_kernel = function(h1, h2) {
    # RBF in feature space: K1(i,j) = exp( -||h_i - h_j||^2 / (2 sigma1^2) ).
    # sigma1 is CLAMPED from below to prevent K_feat from collapsing to a
    # near-delta function (sigma1 -> 0 makes K_feat ~ I, which multiplies
    # K_spat by near-zero off-diagonal and produces a near-rank-1 K that
    # the Cholesky factorisation cannot handle).
    log_sigma1_safe <- torch_clamp(self$log_sigma1, min = -3.0, max = 3.0)
    sigma1_sq <- torch_exp(2 * log_sigma1_safe)
    a2 <- torch_sum(h1 * h1, dim = 2L, keepdim = TRUE)             # [N1, 1]
    b2 <- torch_sum(h2 * h2, dim = 2L, keepdim = TRUE)$t()         # [1, N2]
    ab <- torch_matmul(h1, h2$t())                                 # [N1, N2]
    dist_sq <- a2 + b2 - 2 * ab
    dist_sq <- torch_clamp(dist_sq, min = 0)
    torch_exp(-0.5 * dist_sq / sigma1_sq)
  },

  combined_kernel = function(x1_tab, x1_patch, s1,
                             x2_tab, x2_patch, s2) {
    h1 <- self$encode(x1_tab, x1_patch)
    h2 <- self$encode(x2_tab, x2_patch)
    K1 <- self$feature_kernel(h1, h2)
    K2 <- self$spatial_kernel(h1, s1, h2, s2)
    list(K = K1 * K2, h1 = h1, h2 = h2, K_feat = K1, K_spat = K2)
  },

  # ---- GP marginal likelihood + alpha ---------------------------------------
  # Returns the negative log marginal likelihood on the training set, plus
  # the alpha = K^{-1} (y - mu0) and the Cholesky factor L, both used for
  # downstream prediction.
  #
  # Numerics (two defensive layers for DSM-sized calibrations):
  #   1. The whole Cholesky path runs on CPU in float64 regardless of the
  #      caller device. MPS torch has no aten::_cholesky_solve_helper
  #      (upstream PyTorch #141287), and float32 is too tight for a
  #      500x500 kernel on near-duplicate (x, s) samples. Autograd tracks
  #      gradients across the dtype + device transfer automatically.
  #   2. Progressive jitter retry: if linalg_cholesky fails (K is rank
  #      deficient), we retry with K + j * I for j in an increasing
  #      schedule. This converts "hard" rank-deficient failures into
  #      graceful, logged jitter bumps — far safer than one fixed jitter.
  fit_cholesky = function(x_tab, x_patch, coords, y) {
    n <- y$shape[1]
    comb <- self$combined_kernel(x_tab, x_patch, coords,
                                 x_tab, x_patch, coords)
    K <- comb$K
    gpu_device <- K$device

    # Clamp log_noise from below so the GP cannot learn zero-noise.
    log_noise_safe <- torch_clamp(self$log_noise,
                                   min = self$log_noise_floor)
    noise <- torch_exp(log_noise_safe) + self$jitter
    eye_gpu <- torch_eye(n, dtype = K$dtype, device = gpu_device)
    K_noise <- K + noise * eye_gpu

    # Move to CPU, then cast to float64. The two steps MUST be separate:
    # `.to(device="cpu", dtype=torch_float64())` triggers MPS -> float64
    # conversion first, and MPS backend does not support float64 at all.
    K_noise_cpu <- K_noise$to(device = "cpu")$to(dtype = torch_float64())
    y_c         <- y - self$prior_mean
    y_c_cpu     <- y_c$to(device = "cpu")$to(dtype = torch_float64())
    eye_cpu_d   <- torch_eye(n, dtype = torch_float64(), device = "cpu")

    # Defensive NaN/Inf detection: if the encoder or hypernet diverged the
    # kernel can contain non-finite entries; adding jitter to the diagonal
    # wouldn't save that. Base R on the host-side copy is simpler than
    # wrestling with torch_any / torch_isfinite availability across
    # torch-R versions.
    K_host_probe <- as.numeric(torch_sum(K_noise_cpu)$cpu())
    if (!is.finite(K_host_probe)) {
      stop("[DKP] kernel matrix contains NaN or Inf (non-finite sum). Likely encoder/hypernet divergence.")
    }

    # Progressive jitter retry. 0 = no extra (use K_noise_cpu as is).
    # Schedule extends up to 1e4 to handle pathological cases: even if
    # the off-diagonal structure is weird, adding a large constant to the
    # diagonal ALWAYS produces a PD matrix (trivially diagonal-dominant).
    # If 1e4 is required the model essentially predicts the prior mean
    # for that step, but training continues rather than aborting.
    extra_jitter_schedule <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e4)
    L_cpu <- NULL
    used_extra <- NA_real_
    for (j_extra in extra_jitter_schedule) {
      K_try <- if (j_extra == 0) K_noise_cpu else K_noise_cpu + j_extra * eye_cpu_d
      L_cpu <- tryCatch(linalg_cholesky(K_try), error = function(e) NULL)
      if (!is.null(L_cpu)) {
        used_extra <- j_extra
        break
      }
    }
    if (is.null(L_cpu)) {
      stop(sprintf(
        "[DKP] Cholesky failed at every jitter level up to %.1e. K diagonal range: [%.3e, %.3e]",
        tail(extra_jitter_schedule, 1),
        as.numeric(torch_min(torch_diagonal(K_noise_cpu, dim1 = 1L, dim2 = 2L))$cpu()),
        as.numeric(torch_max(torch_diagonal(K_noise_cpu, dim1 = 1L, dim2 = 2L))$cpu())))
    }
    if (used_extra > 0) {
      cat(sprintf("[DKP]   Cholesky extra jitter used = %.1e\n", used_extra))
    }

    alpha_cpu_d <- torch_cholesky_solve(y_c_cpu$unsqueeze(2L), L_cpu)$squeeze(2L)
    diag_L_d    <- torch_diagonal(L_cpu, offset = 0L, dim1 = 1L, dim2 = 2L)
    log_det_d   <- 2 * torch_sum(torch_log(diag_L_d + 1e-15))

    nll_d <- 0.5 * torch_sum(y_c_cpu * alpha_cpu_d) +
             0.5 * log_det_d +
             0.5 * n * log(2 * pi)

    # Cast outputs back to float32 / caller device. We have to do the
    # dtype cast BEFORE the device move: MPS does not accept float64
    # tensors, so `.to(device="mps", dtype=...)` fails (the cast would
    # otherwise be applied after the device transfer). Autograd still
    # propagates gradients correctly through both ops.
    alpha_cpu_f32 <- alpha_cpu_d$to(dtype = torch_float())
    nll_f32       <- nll_d$to(dtype = torch_float())

    # MEMORY: We deliberately DROP K, L_cpu, K_noise_cpu, h_train from the
    # return list. Downstream callers (predict_mean, dkp_predict_test, the
    # main training loop) only consume `nll` and `alpha`. Holding L (n^2
    # float32 ~88 MB at n=4800) and K (another ~88 MB) across the lifetime
    # of `out` doubled the memory pressure inside each epoch and was the
    # cause of the n=6000 RStudio crash after ~2 hours. With this strip
    # the per-call live footprint drops from ~280 MB to ~30 KB.
    list(nll   = nll_f32,
         alpha = alpha_cpu_f32$to(device = gpu_device))
  },

  # ---- Prediction on test points --------------------------------------------
  # Uses the already-fitted alpha. Returns the posterior mean. The posterior
  # variance is available via torch_cholesky_solve with the transposed k_*
  # matrix (we expose it as a helper too).
  predict_mean = function(fit_out,
                          x_tab_tr, x_patch_tr, s_tr,
                          x_tab_te, x_patch_te, s_te) {
    comb <- self$combined_kernel(x_tab_te, x_patch_te, s_te,
                                 x_tab_tr, x_patch_tr, s_tr)
    # comb$K: [N_te, N_tr]
    y_hat <- self$prior_mean + torch_matmul(comb$K, fit_out$alpha$unsqueeze(2L))$squeeze(2L)
    y_hat
  }
)


# -----------------------------------------------------------------------------
# §3  Prediction / loss utilities
# -----------------------------------------------------------------------------

# Predict the standardised-log-scale mean on test inputs, given a fitted
# model and its training cache. All four tensor caches come from the
# usual build_convkrigingnet2d_tensor_cache helper.
dkp_predict_test <- function(model, train_cache, test_cache) {
  model$eval()
  with_no_grad({
    fit_out <- model$fit_cholesky(train_cache$X, train_cache$P,
                                  train_cache$C, train_cache$y)
    y_hat   <- model$predict_mean(fit_out,
                                  train_cache$X, train_cache$P, train_cache$C,
                                  test_cache$X,  test_cache$P,  test_cache$C)
    as.numeric(y_hat$cpu())
  })
}


# -----------------------------------------------------------------------------
# §3b  Robust Duan smearing correction (Jensen / Finney bias removal)
# -----------------------------------------------------------------------------
#
# When target_transform = "log" the GP predicts the conditional MEDIAN of y;
# exp(yhat_log) systematically under-predicts the conditional MEAN. Duan
# (1983) gives a non-parametric correction:
#
#   S = (1/n_val) * sum_j exp( log y_j^val - log yhat_j^val )
#   yhat_corrected = exp(yhat_log) * S
#
# The estimator's variance is dominated by log-scale outliers because exp()
# amplifies the right tail. We therefore use a TRIMMED mean (default 5% each
# tail) and gate application on a minimum n_val threshold:
#
#   - At n_val >= min_val_n (default 500), S is stable enough to apply.
#   - Below that we report S as a diagnostic only and do not multiply.
#
# Rationale: the sampling variance of mean(exp(r_log)) scales roughly as
# sd(exp(r))/sqrt(n_val). With n_val = 100 this is easily > 5%, producing
# the over-correction we saw in the NMK round-5 experiment. With n_val = 600
# the trimmed mean is within ~0.5% of the true exp(sigma^2/2) for
# well-behaved residual distributions — the correction becomes an unbiased
# variance-reducer rather than a noise injector.
.dkp_duan_smearing_robust <- function(log_residuals,
                                      trim_frac = 0.05,
                                      min_val_n = 500L) {
  r <- log_residuals[is.finite(log_residuals)]
  n <- length(r)

  if (n < as.integer(min_val_n)) {
    return(list(
      S         = 1.0,
      S_raw     = NA_real_,
      S_trimmed = NA_real_,
      applied   = FALSE,
      n_used    = 0L,
      reason    = sprintf("n_val = %d < min_val_n = %d (too noisy to apply safely)",
                          n, as.integer(min_val_n))
    ))
  }

  er <- exp(r)

  # Trimmed mean — robust to the log-scale outliers that inflated the
  # estimator at small n_val.
  trim_frac <- max(0.0, min(0.25, as.numeric(trim_frac)))
  q <- quantile(er, probs = c(trim_frac, 1 - trim_frac),
                names = FALSE, na.rm = TRUE, type = 7L)
  er_trim <- er[er >= q[1] & er <= q[2]]
  S_trim  <- mean(er_trim, na.rm = TRUE)

  # Raw Duan also reported for transparency — if S_trim and S_raw disagree
  # substantially we have heavy-tailed residuals and the trim is doing work.
  S_raw   <- mean(er, na.rm = TRUE)

  if (!is.finite(S_trim) || S_trim <= 0) {
    return(list(
      S         = 1.0,
      S_raw     = S_raw,
      S_trimmed = S_trim,
      applied   = FALSE,
      n_used    = length(er_trim),
      reason    = sprintf("trimmed mean non-finite (raw=%.3f) — fallback to S=1",
                          S_raw)
    ))
  }

  list(
    S         = S_trim,
    S_raw     = S_raw,
    S_trimmed = S_trim,
    applied   = TRUE,
    n_used    = length(er_trim),
    reason    = sprintf("applied trimmed-mean Duan (n_used=%d, trim=%.2f)",
                        length(er_trim), trim_frac)
  )
}


# -----------------------------------------------------------------------------
# §4  dkp_auto_config — variogram + n + probe -> every DKP hyperparameter
# -----------------------------------------------------------------------------

dkp_auto_config <- function(vg, n_train, coord_scaler) {
  r  <- vg$nugget_ratio
  cs <- mean(coord_scaler$scale)
  fit_q <- if (is.finite(vg$fit_quality)) vg$fit_quality else 0.25

  ell_major_init <- vg$range_major / cs
  ell_minor_init <- vg$range_minor / cs
  theta_init     <- vg$theta_rad

  init_mix_mean <- .nmk_clamp(0.5 + 2.0 * (1 - fit_q), 0.5, 2.5)

  # Noise variance init = nugget_ratio. On a standardised target (var=1),
  # sigma_n^2 = r is the natural initialisation — the GP expects this
  # fraction of the variance to be irreducible.
  init_log_noise <- log(.nmk_clamp(r, 0.02, 0.5))

  # Prior mean: zero (target is standardised before training).
  init_prior_mean <- 0.0

  # Statistical capacity — slightly narrower than NMK because GP is
  # full-batch and we don't need the backbone to pull as much of the
  # weight.
  d             <- as.integer(.nmk_clamp(
                     64L * ceiling(sqrt(n_train) / 8), 128L, 512L))
  patch_dim     <- as.integer(d / 2L)
  matern_hidden <- as.integer(.nmk_clamp(round(d / 4), 32L, 128L))

  tab_dropout   <- round(.nmk_clamp(0.15 - n_train / 12000,
                                    0.05, 0.20), 3L)
  patch_dropout <- round(.nmk_clamp(0.10 - n_train / 15000,
                                    0.03, 0.12), 3L)

  # GP trains full-batch. "batch_size" = n_train always.
  batch_size    <- as.integer(n_train)

  # Schedule — DKP typically converges in ~100-200 epochs.
  max_warmup_epochs <- as.integer(.nmk_clamp(round(4 + 16 * r), 4L, 16L))
  W <- max_warmup_epochs
  patience_cfg    <- as.integer(.nmk_clamp(round(3 * W),  10L, 60L))
  lr_patience_cfg <- as.integer(.nmk_clamp(round(W),       3L, 15L))
  lr_decay_cfg    <- 0.5

  # Feature kernel bandwidth sigma1 is initialised by the median-heuristic
  # at fit time (after the encoder has been built), not here — we only
  # store the placeholder log_sigma1 = 0.
  init_log_sigma1 <- 0.0

  cat(sprintf(
    "[DKP auto]  r=%.3f  fit_q=%.2f  n=%d  d=%d  matern_hidden=%d\n",
    r, fit_q, n_train, d, matern_hidden))
  cat(sprintf(
    "[DKP auto]  tab_drop=%.2f  patch_drop=%.2f  noise_init=%.3f  mix_mean=%.2f\n",
    tab_dropout, patch_dropout, exp(init_log_noise), init_mix_mean))
  cat(sprintf(
    "[DKP auto]  max_warmup=%d  patience=%d  lr_patience=%d\n",
    max_warmup_epochs, patience_cfg, lr_patience_cfg))

  list(
    d                 = d,
    patch_dim         = patch_dim,
    matern_hidden     = matern_hidden,
    tab_dropout       = tab_dropout,
    patch_dropout     = patch_dropout,
    batch_size        = batch_size,
    max_warmup_epochs = max_warmup_epochs,
    patience          = patience_cfg,
    lr_patience       = lr_patience_cfg,
    lr_decay          = lr_decay_cfg,
    lr_init           = NA_real_,                # filled by Polyak probe
    min_lr            = NA_real_,
    wd                = NA_real_,
    ell_major_init    = ell_major_init,
    ell_minor_init    = ell_minor_init,
    theta_init        = theta_init,
    init_mix_mean     = init_mix_mean,
    init_log_sigma1   = init_log_sigma1,
    init_log_noise    = init_log_noise,
    init_prior_mean   = init_prior_mean,
    nugget_ratio      = r,
    fit_quality       = fit_q,
    fusion_hidden     = d
  )
}


# -----------------------------------------------------------------------------
# §4b  Polyak LR probe specialised for marginal likelihood
# -----------------------------------------------------------------------------

estimate_dkp_initial_lr_polyak <- function(model, train_cache, device = "cpu") {
  cat("[DKP auto] --- LR probe (Polyak on NLL) ---\n")
  tryCatch({
    tmp_opt <- optim_adam(model$parameters, lr = 0.01)
    model$train()
    tmp_opt$zero_grad()
    out <- model$fit_cholesky(train_cache$X, train_cache$P,
                              train_cache$C, train_cache$y)
    out$nll$backward()

    loss_val    <- out$nll$item()
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
      cat("[DKP auto]   degenerate probe — fallback lr = 1e-3\n")
      return(1e-3)
    }
    lr_polyak <- loss_val / grad_sq_sum
    lr_est    <- 0.01 * lr_polyak
    lr_final  <- .nmk_clamp(lr_est, 1e-5, 1e-2)
    cat(sprintf("[DKP auto]   nll=%.4f  grad_sq=%.4e  lr_polyak=%.2e  lr_final=%.2e\n",
                loss_val, grad_sq_sum, lr_polyak, lr_final))
    lr_final
  }, error = function(e) {
    cat(sprintf("[DKP auto]   LR probe failed (%s) — fallback lr = 1e-3\n",
                conditionMessage(e)))
    1e-3
  })
}


# -----------------------------------------------------------------------------
# §4c  Median-heuristic initialisation for sigma1 (feature-kernel bandwidth)
# -----------------------------------------------------------------------------
#
# After the encoder is built we run one forward pass on the training X and
# pick sigma1 = median(||h_i - h_j||) / sqrt(2). This is the classical
# RBF median heuristic (Scholkopf & Smola 2002) and gives an init where
# the kernel is neither too peaked nor too flat on the real data geometry.

initialise_dkp_sigma1_median <- function(model, train_cache) {
  with_no_grad({
    h <- model$encode(train_cache$X, train_cache$P)
    n <- h$shape[1]
    # Random subsample up to 128 rows to compute the pairwise distance median
    # cheaply.
    m <- min(128L, n)
    if (n > m) {
      idx <- sample.int(n, m)
      idx_t <- torch_tensor(idx, dtype = torch_long(), device = h$device)
      h_sub <- h$index_select(1L, idx_t)
    } else {
      h_sub <- h
    }
    d2 <- torch_sum((h_sub$unsqueeze(2L) - h_sub$unsqueeze(1L)) ^ 2,
                   dim = 3L)
    dist <- torch_sqrt(torch_clamp(d2, min = 1e-12))
    # Flatten and take median.
    flat <- dist$reshape(-1L)
    median_val <- as.numeric(torch_median(flat)$cpu())
    if (!is.finite(median_val) || median_val < 1e-6) {
      cat("[DKP auto]   median heuristic degenerate — keeping log_sigma1 = 0\n")
      return(invisible(NULL))
    }
    sigma1 <- median_val / sqrt(2)
    log_sigma1 <- log(sigma1)
    model$log_sigma1$copy_(torch_tensor(log_sigma1, dtype = model$log_sigma1$dtype,
                                        device    = model$log_sigma1$device))
    cat(sprintf("[DKP auto]   sigma1 (median heuristic) = %.4f (log = %+.3f)\n",
                sigma1, log_sigma1))
  })
  invisible(NULL)
}


# -----------------------------------------------------------------------------
# §5  Training loop — exact-GP marginal likelihood (full batch)
# -----------------------------------------------------------------------------

train_deep_kernel_process_one_fold <- function(
  fd,
  epochs               = 200L,
  lr                   = NULL,
  wd                   = NULL,
  patience             = NULL,
  lr_decay             = NULL,
  lr_patience          = NULL,
  min_lr               = NULL,
  tab_hidden           = c(192L),
  tab_dropout          = NULL,
  patch_dim            = NULL,
  patch_dropout        = NULL,
  fusion_hidden        = NULL,
  matern_hidden        = NULL,
  init_mix_mean        = NULL,
  init_prior_mean      = NULL,
  nu_classes           = c(0.5, 1.5, 2.5),
  vg_cutoff_frac       = 0.50,
  vg_n_lags            = 15L,
  target_transform     = "auto",
  device               = "cpu",
  train_seed           = NULL,
  refit_fixed_epochs   = FALSE,
  duan_apply           = NULL,
  duan_min_val_n       = NULL,
  duan_trim_frac       = NULL,
  ...
) {

  # --- Fold ----------------------------------------------------------------
  Xtr <- fd$X$train;       Xva <- fd$X$val;        Xte <- fd$X$test
  Ptr <- fd$patches$train; Pva <- fd$patches$val;  Pte <- fd$patches$test
  Ctr <- fd$coords$train;  Cva <- fd$coords$val;   Cte <- fd$coords$test
  ytr <- fd$y$train;       yva <- fd$y$val;        yte <- fd$y$test

  # --- Auto target transform ----------------------------------------------
  target_transform <- tolower(as.character(target_transform))
  if (identical(target_transform, "auto")) {
    target_transform <- .nmk_target_transform_auto(ytr)
    cat(sprintf("[DKP auto]  target_transform auto-selected -> '%s'\n",
                target_transform))
  }

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

  # --- Small-n advisory (shared helper from NeuroMaternKrige.R) -----------
  suppress_small_n <- identical(
    toupper(Sys.getenv("DKP_SMALL_N_SUPPRESS",
                       unset = Sys.getenv("NMK_SMALL_N_SUPPRESS",
                                          unset = "FALSE"))),
    "TRUE")
  .nmk_small_n_advisory(n_train, model_label = "DKP",
                        suppress = suppress_small_n)

  # --- Variogram & auto-config --------------------------------------------
  cat("[DKP] === Variogram fit (target scale = chosen transform) ===\n")
  vg  <- fit_variogram_auto(ytr, Ctr,
                             cutoff_frac = vg_cutoff_frac,
                             n_lags      = as.integer(vg_n_lags))
  cfg <- dkp_auto_config(vg, n_train, coord_scaler)

  # Runner overrides.
  if (!is.null(patience))      cfg$patience    <- as.integer(patience)
  if (!is.null(lr_decay))      cfg$lr_decay    <- as.numeric(lr_decay)
  if (!is.null(lr_patience))   cfg$lr_patience <- as.integer(lr_patience)
  if (!is.null(min_lr))        cfg$min_lr      <- as.numeric(min_lr)
  if (!is.null(tab_dropout))   cfg$tab_dropout <- as.numeric(tab_dropout)
  if (!is.null(patch_dim))     cfg$patch_dim   <- as.integer(patch_dim)
  if (!is.null(patch_dropout)) cfg$patch_dropout <- as.numeric(patch_dropout)
  if (!is.null(fusion_hidden)) cfg$fusion_hidden <- as.integer(fusion_hidden)
  if (!is.null(matern_hidden)) cfg$matern_hidden <- as.integer(matern_hidden)
  if (!is.null(init_mix_mean)) cfg$init_mix_mean <- as.numeric(init_mix_mean)
  if (!is.null(init_prior_mean)) cfg$init_prior_mean <- as.numeric(init_prior_mean)

  epochs <- max(1L, as.integer(epochs))

  # --- Tensor caches -------------------------------------------------------
  set_convkrigingnet2d_seed(train_seed)
  train_cache <- build_convkrigingnet2d_tensor_cache(Xtr, Ptr, Ctr_s, ytr_s, device)
  val_cache   <- build_convkrigingnet2d_tensor_cache(Xva, Pva, Cva_s, yva_s, device)
  test_cache  <- build_convkrigingnet2d_tensor_cache(Xte, Pte, Cte_s, NULL,  device)

  # --- Build model ---------------------------------------------------------
  model <- DeepKernelProcessGP(
    c_tab           = ncol(Xtr),
    patch_channels  = dim(Ptr)[1L],
    d               = cfg$d,
    tab_hidden      = tab_hidden,
    tab_dropout     = cfg$tab_dropout,
    patch_dim       = cfg$patch_dim,
    patch_dropout   = cfg$patch_dropout,
    multiscale_patch = TRUE,
    fusion_hidden   = cfg$fusion_hidden,
    matern_hidden   = cfg$matern_hidden,
    init_ell_major  = cfg$ell_major_init,
    init_ell_minor  = cfg$ell_minor_init,
    init_theta      = cfg$theta_init,
    init_mix_mean   = cfg$init_mix_mean,
    init_log_sigma1 = cfg$init_log_sigma1,
    init_log_noise  = cfg$init_log_noise,
    init_prior_mean = cfg$init_prior_mean,
    nu_classes      = nu_classes
  )
  model$to(device = device)

  # Weight decay from param count (same rule as NMK).
  n_params <- nmk_count_params(model)
  wd_cfg <- .nmk_clamp(1e-3 / sqrt(max(1.0, n_params / 5e6)), 1e-4, 1e-2)
  if (!is.null(wd)) wd_cfg <- as.numeric(wd)
  cfg$wd       <- wd_cfg
  cfg$n_params <- n_params
  cat(sprintf("[DKP auto]  n_params=%s  wd=%.2e\n",
              format(n_params, big.mark = ","), wd_cfg))

  # Median-heuristic init for sigma1.
  initialise_dkp_sigma1_median(model, train_cache)

  # Polyak-style LR probe on the actual NLL.
  lr_probe <- if (!is.null(lr)) as.numeric(lr) else {
    estimate_dkp_initial_lr_polyak(model, train_cache, device)
  }
  cfg$lr_init <- lr_probe
  cfg$min_lr  <- if (!is.null(min_lr)) as.numeric(min_lr) else lr_probe / 20
  lr_now <- cfg$lr_init

  # --- Main loop (no mini-batches — GP is global) --------------------------
  cat("[DKP] === Main loop: marginal-likelihood training ===\n")
  opt <- optim_adamw(model$parameters, lr = lr_now, weight_decay = cfg$wd)

  best_val   <- Inf
  best_state <- NULL
  best_epoch <- NA_integer_
  bad <- 0L; lr_bad <- 0L; main_done <- 0L

  skipped_epochs <- 0L
  for (ep in seq_len(epochs)) {
    main_done <- ep

    # Each epoch is a single full-batch Cholesky-backed step plus a
    # validation pass. Wrap both in tryCatch so a transient numerical
    # failure (NaN kernel, Cholesky blow-up) does not abort training —
    # we just skip the step and continue.
    ep_result <- tryCatch({
      model$train()
      opt$zero_grad()
      out <- model$fit_cholesky(train_cache$X, train_cache$P,
                                train_cache$C, train_cache$y)
      loss <- out$nll
      loss_val <- loss$item()
      if (!is.finite(loss_val)) {
        stop(sprintf("non-finite loss = %s", format(loss_val)))
      }
      loss$backward()
      # Tighter grad clip (1.0 vs earlier 2.0) for the GP path: marginal
      # likelihood gradients are sharper than pointwise-Huber gradients
      # and can explode parameters if unclamped.
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      opt$step()

      # MEMORY: explicitly release training-side tensors BEFORE the
      # validation pass. Without this, `out` (which still holds alpha
      # with its post-backward graph reference) coexists with the
      # validation `fit_eval` and roughly doubles the live tensor
      # footprint mid-epoch.
      rm(out, loss)

      # Validation RMSE.
      model$eval()
      val_pred_s <- with_no_grad({
        fit_eval <- model$fit_cholesky(train_cache$X, train_cache$P,
                                       train_cache$C, train_cache$y)
        y_hat <- model$predict_mean(fit_eval,
                                    train_cache$X, train_cache$P, train_cache$C,
                                    val_cache$X,   val_cache$P,   val_cache$C)
        out_vec <- as.numeric(y_hat$cpu())
        rm(fit_eval, y_hat)
        out_vec
      })
      vl <- sqrt(mean((as.numeric(val_cache$y$cpu()) - val_pred_s) ^ 2,
                      na.rm = TRUE))

      list(ok = TRUE, loss = loss_val, val_rmse = vl)
    }, error = function(e) {
      list(ok = FALSE, err = conditionMessage(e))
    })

    # Force R + torch to release the per-epoch allocations. gc() is cheap
    # (<10 ms), and on MPS/CUDA the deallocator runs lazily — without
    # this nudge memory accumulates over hundreds of epochs and triggers
    # the OS-level OOM kill we saw at n=6000 after ~2 hours.
    gc(verbose = FALSE, full = FALSE)

    if (!isTRUE(ep_result$ok)) {
      skipped_epochs <- skipped_epochs + 1L
      cat(sprintf("[DKP] ep %3d SKIPPED (%s). Continuing from last good state.\n",
                  ep, ep_result$err))
      # Rollback to the last best state so we never accumulate corrupt
      # parameter updates from a partial step.
      if (!is.null(best_state)) model$load_state_dict(best_state)
      # Soft lr reduction on each skip as a precaution.
      if (lr_now > cfg$min_lr) {
        lr_now <- max(cfg$min_lr, lr_now * cfg$lr_decay)
        for (pg in opt$param_groups) pg$lr <- lr_now
      }
      next
    }

    loss_val <- ep_result$loss
    vl       <- ep_result$val_rmse

    noise_v  <- as.numeric(torch_exp(model$log_noise)$cpu())
    sigma1_v <- as.numeric(torch_exp(model$log_sigma1)$cpu())
    if (ep <= 10L || ep %% 10L == 0L) {
      cat(sprintf(
        "[DKP] ep %3d | lr=%.2e | nll=%.3f | val_RMSE_s=%.4f | sigma1=%.3f | sigma_n^2=%.4f\n",
        ep, lr_now, loss_val, vl, sigma1_v, noise_v))
    }

    if (vl < best_val) {
      best_val   <- vl
      best_epoch <- ep
      if (!isTRUE(refit_fixed_epochs)) {
        best_state <- clone_state_dict(model$state_dict())
        bad <- 0L; lr_bad <- 0L
      }
    } else if (!isTRUE(refit_fixed_epochs)) {
      bad    <- bad + 1L
      lr_bad <- lr_bad + 1L
      if (lr_bad >= cfg$lr_patience && lr_now > cfg$min_lr) {
        lr_now <- max(cfg$min_lr, lr_now * cfg$lr_decay)
        for (pg in opt$param_groups) pg$lr <- lr_now
        lr_bad <- 0L
        cat(sprintf("[DKP] lr -> %.2e\n", lr_now))
      }
      if (bad >= cfg$patience) {
        cat("[DKP] Early stop.\n")
        break
      }
    }
  }
  if (skipped_epochs > 0L) {
    cat(sprintf("[DKP] %d epoch(s) skipped due to numerical failures.\n",
                skipped_epochs))
  }

  if (!isTRUE(refit_fixed_epochs) && !is.null(best_state)) {
    model$load_state_dict(best_state)
  } else if (isTRUE(refit_fixed_epochs)) {
    cat("[DKP] Refit fixed-epoch mode: keeping final epoch state.\n")
  }
  model$eval()

  # --- Final predictions on val and test (standardised target scale) -----
  val_preds_s <- dkp_predict_test(model, train_cache, val_cache)
  test_preds_s <- dkp_predict_test(model, train_cache, test_cache)

  val_preds  <- inverse_transform_target(
    invert_target_scaler(val_preds_s,  y_scaler), target_transform)
  test_preds <- inverse_transform_target(
    invert_target_scaler(test_preds_s, y_scaler), target_transform)

  # --- Robust Duan smearing correction (when target_transform = "log") -------
  # Gated on n_val >= min_val_n (default 500) — at smaller n_val the
  # estimator's variance exceeds the bias it corrects, as observed in NMK
  # round 5 experiments.
  duan_apply_value <- if (!is.null(duan_apply)) {
    isTRUE(duan_apply)
  } else {
    duan_apply_raw <- Sys.getenv("DKP_DUAN_APPLY", unset = "TRUE")
    toupper(duan_apply_raw) %in% c("TRUE", "1", "YES", "Y")
  }
  duan_min_n <- if (!is.null(duan_min_val_n)) {
    as.integer(duan_min_val_n)
  } else {
    as.integer(.nmk_env_num("DKP_DUAN_MIN_VAL_N", 500))
  }
  duan_trim_frac <- if (!is.null(duan_trim_frac)) {
    as.numeric(duan_trim_frac)
  } else {
    .nmk_env_num("DKP_DUAN_TRIM_FRAC", 0.05)
  }

  smearing_info <- list(S = 1.0, applied = FALSE, reason = "target != log")
  if (duan_apply_value && identical(target_transform, "log")) {
    val_log_obs   <- log(yva)
    val_log_preds <- invert_target_scaler(val_preds_s, y_scaler)
    r_log         <- val_log_obs - val_log_preds
    smearing_info <- .dkp_duan_smearing_robust(
      r_log,
      trim_frac = duan_trim_frac,
      min_val_n = duan_min_n)
  }

  cat(sprintf("[DKP auto]  Duan smearing: %s\n", smearing_info$reason))
  if (isTRUE(smearing_info$applied)) {
    cat(sprintf("[DKP auto]    S_trim = %.5f  (S_raw = %.5f, n_val_used = %d)\n",
                smearing_info$S_trimmed, smearing_info$S_raw,
                smearing_info$n_used))
    # Apply only to the predictive means; the "base" predictions returned
    # by DKP are the prior_mean constant repeated, computed AFTER this
    # block, so they don't need a Duan multiplier (they're not GP
    # posterior means, just the unconditional prior).
    val_preds  <- val_preds  * smearing_info$S
    test_preds <- test_preds * smearing_info$S
  }

  # Backbone-only counterpart: the "base" for DKP is the prior_mean + 0
  # GP correction, which on the original scale is just a constant. Return
  # the prior mean as a diagnostic baseline.
  base_const <- as.numeric(model$prior_mean$cpu())
  base_val   <- rep(inverse_transform_target(
    invert_target_scaler(base_const, y_scaler), target_transform),
    length(yva))
  base_test  <- rep(inverse_transform_target(
    invert_target_scaler(base_const, y_scaler), target_transform),
    nrow(Xte))

  noise_f  <- as.numeric(torch_exp(model$log_noise)$cpu())
  sigma1_f <- as.numeric(torch_exp(model$log_sigma1)$cpu())
  prior_f  <- as.numeric(model$prior_mean$cpu())

  cat(sprintf("\n[DKP] === Final summary ===\n"))
  cat(sprintf("[DKP] Val : ME=%+.3f  RMSE=%.3f  (original scale)\n",
              mean(yva - val_preds), sqrt(mean((yva - val_preds) ^ 2))))
  cat(sprintf(
    "[DKP] Geostat: r=%.3f fit_q=%.2f mix_mean=%.2f sigma1=%.3f sigma_n^2=%.4f prior=%.4f\n",
    cfg$nugget_ratio, cfg$fit_quality, cfg$init_mix_mean,
    sigma1_f, noise_f, prior_f))
  cat(sprintf(
    "[DKP] Arch  : d=%d matern_hid=%d tab_drop=%.2f patch_drop=%.2f n_params=%s\n",
    cfg$d, cfg$matern_hidden, cfg$tab_dropout, cfg$patch_dropout,
    format(cfg$n_params, big.mark = ",")))
  cat(sprintf(
    "[DKP] Optim : lr_init=%.2e lr_final=%.2e min_lr=%.2e wd=%.2e best_epoch=%d / %d\n",
    cfg$lr_init, lr_now, cfg$min_lr, cfg$wd, best_epoch, main_done))
  if (isTRUE(smearing_info$applied)) {
    cat(sprintf(
      "[DKP] Duan  : S=%.5f applied (raw=%.5f, n_val_used=%d, trim=%.2f)\n",
      smearing_info$S, smearing_info$S_raw,
      smearing_info$n_used, duan_trim_frac))
  } else {
    cat(sprintf("[DKP] Duan  : not applied (%s)\n", smearing_info$reason))
  }

  list(
    pred_test      = test_preds,
    pred_val       = val_preds,
    pred_test_base = base_test,
    pred_val_base  = base_val,
    metrics_test   = metrics(yte, test_preds),
    diagnostics    = list(
      nugget_ratio     = cfg$nugget_ratio,
      fit_quality      = cfg$fit_quality,
      init_mix_mean    = cfg$init_mix_mean,
      d                = cfg$d,
      matern_hidden    = cfg$matern_hidden,
      tab_dropout      = cfg$tab_dropout,
      patch_dropout    = cfg$patch_dropout,
      batch_size       = cfg$batch_size,
      n_params         = cfg$n_params,
      wd               = cfg$wd,
      lr_init          = cfg$lr_init,
      lr_final         = lr_now,
      min_lr           = cfg$min_lr,
      patience         = cfg$patience,
      lr_patience      = cfg$lr_patience,
      best_epoch       = best_epoch,
      main_epochs_done = main_done,
      best_val_rmse_s  = best_val,
      sigma1_final     = sigma1_f,
      noise_final      = noise_f,
      prior_mean_final = prior_f,
      target_transform = target_transform,
      duan_applied     = isTRUE(smearing_info$applied),
      duan_S           = smearing_info$S,
      duan_S_raw       = if (is.null(smearing_info$S_raw)) NA_real_ else smearing_info$S_raw,
      duan_n_used      = if (is.null(smearing_info$n_used)) 0L else smearing_info$n_used,
      duan_reason      = smearing_info$reason,
      refit_fixed_epochs = isTRUE(refit_fixed_epochs)
    )
  )
}


# -----------------------------------------------------------------------------
# §6  Default parameter presets
# -----------------------------------------------------------------------------

dkp_default_params <- list(
  epochs               = 200L,
  lr                   = NULL,
  wd                   = NULL,
  patience             = NULL,
  lr_decay             = NULL,
  lr_patience          = NULL,
  min_lr               = NULL,
  tab_hidden           = c(192L),
  tab_dropout          = NULL,
  patch_dim            = NULL,
  patch_dropout        = NULL,
  fusion_hidden        = NULL,
  matern_hidden        = NULL,
  init_mix_mean        = NULL,
  init_prior_mean      = NULL,
  nu_classes           = c(0.5, 1.5, 2.5),
  target_transform     = "auto",
  refit_fixed_epochs   = FALSE,
  duan_apply           = NULL,
  duan_min_val_n       = NULL,
  duan_trim_frac       = NULL
)

dkp_quick_params <- modifyList(dkp_default_params,
                               list(epochs = 40L,
                                    patience = 15L))
