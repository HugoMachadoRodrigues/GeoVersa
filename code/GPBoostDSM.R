# =============================================================================
# GPBoostDSM.R
#
# Gaussian Process Boosting for Digital Soil Mapping.
#
# Built on Sigrist (2022, JMLR) "Gaussian process boosting":
#
#   y_i  =  f_T(x_i)  +  b(s_i)  +  epsilon_i
#
#   f_T  : sum of T gradient-boosted trees on tabular covariates
#   b    : zero-mean Gaussian process over coordinates with Matern kernel
#   eps  ~ N(0, sigma_eps^2)
#
# The trees and the GP are NOT trained sequentially (that would be classical
# regression-kriging). They are jointly optimised in a Newton-Raphson outer
# loop where each boosting iteration uses the GP-projected gradient of the
# negative log marginal likelihood. This means each tree is fit with the
# spatial GP structure already accounted for, and the GP covariance
# parameters are re-estimated at every iteration.
#
# Compared with regression-kriging:
#   RK         :  fit RF -> compute residuals -> fit GP on residuals.
#                 Two stages, no information from one informs the other.
#   GPBoost    :  alternate (i) GP profile-likelihood update of trees
#                 (ii) MLE update of GP covariance, every iteration.
#                 Trees adapt to the GP. GP adapts to the trees.
#
# Novelty in DSM:
#   "GPBoost"   + "digital soil mapping" / "soil"  in Google Scholar:
#       essentially zero published applications. The method has been used in
#       insurance, finance, ecology, but not pedometry. This file is the
#       first GPBoost-DSM application I am aware of.
#
# This file is deliberately self-contained: no torch, no NMK / DKP / GeoBART
# dependency. Mirrors GeoBART.R style for the auto-config + Duan + variogram
# helpers, but uses GPBoost as the engine instead of BART + RBF features.
#
# Hyperparameter deduction chain (zero manual tuning):
#
# (A) VARIOGRAM-DRIVEN.  fit_variogram_auto(y, coords) -> r, rho, fit_q
#     matern_nu    = case_when(fit_q  >= 0.6 -> 2.5,
#                              fit_q  >= 0.3 -> 1.5,
#                              else            -> 0.5)
#     init_range   = rho                    (passed to GP as warm start)
#     # GPBoost optimises range, sigma_GP, sigma_eps via MLE during fitting,
#     # so init_range is just a starting point.
#
# (B) STATISTICAL CAPACITY.  n_train ->
#     learning_rate = clamp(0.05, 0.02, 0.1)               (constant — robust)
#     max_depth     = clamp(round(log2(n)/2 + 1), 4, 8)     (sqrt-n principle)
#     num_leaves    = 2^max_depth - 1                       (LightGBM convention)
#     bagging_frac  = 0.8                                   (constant)
#     feature_frac  = clamp(1 - p / (10*sqrt(n)), 0.5, 0.95) (more features
#                                                            with more data)
#     nrounds_max   = 1500
#     early_stop    = clamp(round(50 + sqrt(n)), 30, 100)   (longer wait when
#                                                            data is plentiful)
#
# (C) TARGET PREP.  Shapiro-Wilk + positivity -> target_transform
#                   robust Duan smearing if log AND n_val >= 500
#
# Total: ~250 lines.
#
# Dependencies:
#   - gpboost   (Sigrist's package on CRAN)
#   - gstat, sp (variogram fit)
# =============================================================================

# ---- Helpers ----------------------------------------------------------------

.gpb_clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

.gpb_env_num <- function(name, default) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  v <- suppressWarnings(as.numeric(raw))
  if (!is.finite(v)) default else v
}

.gpb_rmse <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  if (sum(ok) == 0L) return(Inf)
  sqrt(mean((obs[ok] - pred[ok]) ^ 2))
}

.gpb_rf_predict_internal <- function(X_train, y_train, X_pred, seed = 1L) {
  X_train <- as.data.frame(X_train)
  X_pred <- as.data.frame(X_pred)
  names(X_train) <- paste0("x", seq_len(ncol(X_train)))
  names(X_pred) <- names(X_train)
  df <- cbind(y = as.numeric(y_train), X_train)
  # The in-gate RF baseline MUST be bit-identical to the scored Wadoux RF
  # (fit_rf_default_wadoux), so the OOF do-no-harm gate compares GPBoost
  # against exactly the model it will fall back to. fit_rf_default_wadoux is
  # ranger(formula, data, seed=123) with pure ranger defaults. We mirror it
  # here (formula interface, ranger defaults, fixed seed) instead of the
  # previous hand-set num.trees/mtry/min.node.size/respect.unordered.factors,
  # which produced a *different* RF and broke the >= RF guarantee.
  if (exists("fit_rf_default_wadoux", mode = "function") &&
      exists("predict_rf_default_wadoux", mode = "function")) {
    form_internal <- stats::as.formula(
      paste("y ~", paste(names(X_train), collapse = " + ")))
    fit <- get("fit_rf_default_wadoux")(df, form_internal,
                                        seed = as.integer(seed))
    return(as.numeric(get("predict_rf_default_wadoux")(fit, X_pred)))
  }
  if (requireNamespace("ranger", quietly = TRUE)) {
    fit <- ranger::ranger(
      y ~ .,
      data = df,
      seed = as.integer(seed)
    )
    return(as.numeric(stats::predict(fit, data = X_pred)$predictions))
  }
  Xs <- .gpb_scale_design(X_train)
  Xp <- suppressWarnings(matrix(as.numeric(as.matrix(X_pred)), nrow = nrow(X_pred)))
  if (ncol(Xs) == 0L || ncol(Xp) == 0L) {
    return(rep(mean(y_train, na.rm = TRUE), nrow(X_pred)))
  }
  n_keep <- min(ncol(Xs), ncol(Xp))
  fit <- tryCatch(stats::lm.fit(cbind(1, Xs[, seq_len(n_keep), drop = FALSE]),
                                as.numeric(y_train)),
                  error = function(e) NULL)
  if (is.null(fit) || is.null(fit$coefficients)) {
    return(rep(mean(y_train, na.rm = TRUE), nrow(X_pred)))
  }
  Xp <- scale(Xp[, seq_len(n_keep), drop = FALSE],
              center = attr(Xs, "scaled:center"),
              scale = attr(Xs, "scaled:scale"))
  as.numeric(cbind(1, Xp) %*% fit$coefficients)
}

.gpb_quantile_rmse_values <- function(obs, pred, n_bins = 5L) {
  n_bins <- suppressWarnings(as.integer(n_bins))
  if (!is.finite(n_bins) || n_bins < 2L) {
    return(list(mean = 0, sd = 0, excess = 0, text = NA_character_,
                n_bins = 0L))
  }
  ok <- is.finite(obs) & is.finite(pred)
  obs <- obs[ok]
  pred <- pred[ok]
  if (length(obs) < max(10L, n_bins * 2L)) {
    return(list(mean = 0, sd = 0, excess = 0, text = NA_character_,
                n_bins = 0L))
  }
  probs <- seq(0, 1, length.out = n_bins + 1L)
  breaks <- unique(as.numeric(stats::quantile(obs, probs = probs,
                                              na.rm = TRUE, type = 7)))
  if (length(breaks) < 3L) {
    return(list(mean = 0, sd = 0, excess = 0, text = NA_character_,
                n_bins = 0L))
  }
  bins <- cut(obs, breaks = breaks, include.lowest = TRUE,
              labels = FALSE)
  vals <- vapply(sort(unique(bins[is.finite(bins)])), function(b) {
    .gpb_rmse(obs[bins == b], pred[bins == b])
  }, numeric(1))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0L) {
    return(list(mean = 0, sd = 0, excess = 0, text = NA_character_,
                n_bins = 0L))
  }
  q_mean <- mean(vals)
  q_sd <- if (length(vals) >= 2L) stats::sd(vals) else 0
  global <- .gpb_rmse(obs, pred)
  q_excess <- max(vals, na.rm = TRUE) - global
  if (!is.finite(q_excess) || q_excess < 0) q_excess <- 0
  list(mean = q_mean, sd = q_sd, excess = q_excess,
       text = paste(formatC(vals, digits = 6L, format = "fg"),
                    collapse = "|"),
       n_bins = length(vals))
}

.gpb_preserve_global_rng <- function() {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
  function() {
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }
}

.gpb_num_grid <- function(x, default) {
  if (is.null(x)) return(default)
  out <- sort(unique(as.numeric(x)))
  out[is.finite(out)]
}

.gpb_int_grid <- function(x, default) {
  if (is.null(x)) return(unique(as.integer(round(default))))
  out <- .gpb_num_grid(x, default)
  sort(unique(as.integer(round(out))))
}

.gpb_scale_design <- function(X) {
  X <- suppressWarnings(matrix(as.numeric(as.matrix(X)), nrow = nrow(X)))
  if (ncol(X) == 0L) return(X)
  for (j in seq_len(ncol(X))) {
    z <- X[, j]
    med <- stats::median(z[is.finite(z)], na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    z[!is.finite(z)] <- med
    X[, j] <- z
  }
  sds <- apply(X, 2L, stats::sd, na.rm = TRUE)
  keep <- is.finite(sds) & sds > 0
  if (!any(keep)) return(matrix(numeric(nrow(X) * 0L), nrow = nrow(X)))
  X <- X[, keep, drop = FALSE]
  scale(X)
}

.gpb_linear_r2 <- function(y, X) {
  y <- as.numeric(y)
  ok <- is.finite(y)
  if (sum(ok) < 5L) return(0)
  Xs <- .gpb_scale_design(X[ok, , drop = FALSE])
  if (ncol(Xs) == 0L) return(0)
  yy <- y[ok]
  fit <- tryCatch(stats::lm.fit(cbind(1, Xs), yy), error = function(e) NULL)
  if (is.null(fit) || is.null(fit$fitted.values)) return(0)
  sst <- sum((yy - mean(yy)) ^ 2)
  if (!is.finite(sst) || sst <= 0) return(0)
  sse <- sum((yy - fit$fitted.values) ^ 2)
  .gpb_clamp(1 - sse / sst, 0, 1)
}

.gpb_tabular_residual_signal <- function(X, y) {
  y <- as.numeric(y)
  n <- length(y)
  yc <- y - mean(y, na.rm = TRUE)
  fallback <- list(signal = yc, method = "centered_response",
                   tabular_trend_r2 = 0)
  if (n < 20L || !is.finite(stats::var(y, na.rm = TRUE)) ||
      stats::var(y, na.rm = TRUE) <= 0) {
    return(fallback)
  }
  Xs <- .gpb_scale_design(X)
  if (ncol(Xs) == 0L) return(fallback)
  ok <- is.finite(y) & apply(is.finite(Xs), 1L, all)
  if (sum(ok) < 20L) return(fallback)
  Xok <- Xs[ok, , drop = FALSE]
  yok <- y[ok]
  p <- ncol(Xok)
  Z <- cbind(Intercept = 1, Xok)
  lambda <- .gpb_clamp(p / max(sum(ok), 1L), 1e-4, 10)
  penalty <- diag(c(0, rep(lambda, p)), nrow = p + 1L)
  beta <- tryCatch(
    solve(crossprod(Z) + penalty, crossprod(Z, yok)),
    error = function(e) tryCatch(qr.solve(crossprod(Z) + penalty,
                                          crossprod(Z, yok)),
                                 error = function(e2) NULL)
  )
  if (is.null(beta)) return(fallback)
  pred <- as.numeric(Z %*% beta)
  resid_ok <- yok - pred
  sst <- sum((yok - mean(yok)) ^ 2)
  sse <- sum(resid_ok ^ 2)
  trend_r2 <- if (is.finite(sst) && sst > 0) .gpb_clamp(1 - sse / sst, 0, 1) else 0
  signal <- yc
  signal[ok] <- resid_ok - mean(resid_ok, na.rm = TRUE)
  if (!is.finite(stats::sd(signal, na.rm = TRUE)) ||
      stats::sd(signal, na.rm = TRUE) <= 1e-8) {
    return(fallback)
  }
  list(signal = signal, method = "ridge_tabular_residual",
       tabular_trend_r2 = trend_r2)
}

.gpb_coord_scaler <- function(coords) {
  coords <- as.matrix(coords)
  center <- colMeans(coords, na.rm = TRUE)
  span <- apply(coords, 2L, function(z) diff(range(z, na.rm = TRUE)))
  scale <- stats::median(span[is.finite(span) & span > 0], na.rm = TRUE)
  if (!is.finite(scale) || scale <= 0) scale <- 1
  list(center = center, scale = scale)
}

.gpb_scale_coords <- function(coords, scaler) {
  sweep(as.matrix(coords), 2L, scaler$center, "-") / scaler$scale
}

.gpb_make_init_cov_pars <- function(y_t, cfg, coord_scale,
                                    range_multiplier = 1,
                                    nugget_multiplier = 1) {
  s2 <- stats::var(as.numeric(y_t), na.rm = TRUE)
  if (!is.finite(s2) || s2 <= 0) s2 <- 1
  nug_ratio <- .gpb_clamp(cfg$nugget_ratio * nugget_multiplier, 0.03, 0.95)
  gp_var <- max(s2 * (1 - nug_ratio), 1e-6)
  nugget <- max(s2 * nug_ratio, 1e-6)
  range_scaled <- (cfg$range_init / max(coord_scale, 1e-6)) *
    max(as.numeric(range_multiplier), 1e-6)
  range_scaled <- .gpb_clamp(range_scaled, 1e-4, 100)
  # gpboost order for Gaussian GPModel covariance parameters:
  # error variance, GP marginal variance, GP range.
  c(error_variance = nugget,
    gp_variance = gp_var,
    gp_range = range_scaled)
}

.gpb_extract_mean <- function(pred) {
  if (is.list(pred)) {
    if (!is.null(pred$response_mean)) return(as.numeric(pred$response_mean))
    if (!is.null(pred$mu)) return(as.numeric(pred$mu))
  }
  as.numeric(pred)
}

.gpb_extract_var <- function(pred, n) {
  if (is.list(pred)) {
    if (!is.null(pred$response_var)) return(as.numeric(pred$response_var))
    if (!is.null(pred$var)) return(as.numeric(pred$var))
  }
  rep(NA_real_, n)
}

.gpb_spec_matrix <- function(specs) {
  mat <- do.call(rbind, lapply(specs, function(s) {
    c(mode_spatial = if (identical(s$mode, "spatial")) 1 else 0,
      learning_rate = log10(max(s$learning_rate, 1e-6)),
      max_depth = s$max_depth,
      num_leaves = log2(max(s$num_leaves, 1L)),
      matern_nu = s$matern_nu,
      feature_fraction = s$feature_fraction,
      range_multiplier = log2(max(s$range_multiplier, 1e-6)),
      nugget_multiplier = log2(max(s$nugget_multiplier, 1e-6)),
      min_data_in_leaf = log2(max(s$min_data_in_leaf, 1L)),
      bagging_fraction = s$bagging_fraction,
      lambda_l2 = log10(max(s$lambda_l2, 0) + 1))
  }))
  mat <- as.matrix(mat)
  for (j in seq_len(ncol(mat))) {
    rng <- range(mat[, j], finite = TRUE)
    if (!all(is.finite(rng)) || diff(rng) <= 0) {
      mat[, j] <- 0
    } else {
      mat[, j] <- (mat[, j] - rng[1]) / diff(rng)
    }
  }
  if ("mode_spatial" %in% colnames(mat)) {
    mat[, "mode_spatial"] <- mat[, "mode_spatial"] * 10
  }
  mat
}

.gpb_min_spec_distance <- function(mat, center_idx, i) {
  center_idx <- as.integer(center_idx)
  center_idx <- center_idx[is.finite(center_idx) & center_idx >= 1L &
                             center_idx <= nrow(mat)]
  i <- as.integer(i)
  if (length(center_idx) == 0L || !is.finite(i) || i < 1L || i > nrow(mat)) {
    return(Inf)
  }
  center <- mat[center_idx, , drop = FALSE]
  target <- matrix(mat[i, ], nrow = nrow(center), ncol = ncol(center),
                   byrow = TRUE)
  min(rowSums((center - target) ^ 2))
}

.gpb_select_evenly <- function(specs, n_keep) {
  n_keep <- as.integer(n_keep)
  if (length(specs) <= n_keep) return(specs)
  if (n_keep <= 0L) return(list())
  mat <- .gpb_spec_matrix(specs)
  selected <- 1L
  remaining <- setdiff(seq_along(specs), selected)
  while (length(selected) < n_keep && length(remaining) > 0L) {
    dmin <- vapply(remaining, function(i) {
      .gpb_min_spec_distance(mat, selected, i)
    }, numeric(1))
    next_i <- remaining[which.max(dmin)]
    selected <- c(selected, next_i)
    remaining <- setdiff(remaining, next_i)
  }
  specs[selected]
}

.gpb_select_even_indices <- function(specs, idx, n_keep) {
  idx <- unique(as.integer(idx))
  idx <- idx[is.finite(idx) & idx >= 1L & idx <= length(specs)]
  n_keep <- as.integer(n_keep)
  if (length(idx) <= n_keep) return(idx)
  if (n_keep <= 0L) return(integer())
  selected_specs <- .gpb_select_evenly(specs[idx], n_keep)
  selected_ids <- vapply(selected_specs, function(s) {
    paste(s$mode, s$learning_rate, s$max_depth, s$num_leaves, s$matern_nu,
          s$feature_fraction, s$min_data_in_leaf, s$bagging_fraction,
          s$lambda_l2, s$range_multiplier, s$nugget_multiplier, sep = "|")
  }, character(1))
  all_ids <- vapply(specs, function(s) {
    paste(s$mode, s$learning_rate, s$max_depth, s$num_leaves, s$matern_nu,
          s$feature_fraction, s$min_data_in_leaf, s$bagging_fraction,
          s$lambda_l2, s$range_multiplier, s$nugget_multiplier, sep = "|")
  }, character(1))
  match(selected_ids, all_ids)
}

.gpb_nearest_spec_indices <- function(specs, center_idx, pool_idx,
                                      n_keep, exclude_idx = integer()) {
  center_idx <- unique(as.integer(center_idx))
  pool_idx <- setdiff(unique(as.integer(pool_idx)), as.integer(exclude_idx))
  center_idx <- center_idx[is.finite(center_idx) & center_idx >= 1L &
                             center_idx <= length(specs)]
  pool_idx <- pool_idx[is.finite(pool_idx) & pool_idx >= 1L &
                         pool_idx <= length(specs)]
  n_keep <- as.integer(n_keep)
  if (length(pool_idx) <= n_keep) return(pool_idx)
  if (length(center_idx) == 0L || length(pool_idx) == 0L || n_keep <= 0L) {
    return(integer())
  }
  mat <- .gpb_spec_matrix(specs)
  dmin <- vapply(pool_idx, function(i) {
    .gpb_min_spec_distance(mat, center_idx, i)
  }, numeric(1))
  pool_idx[order(dmin, pool_idx)][seq_len(min(n_keep, length(pool_idx)))]
}

.gpb_make_regression_folds <- function(y, k, seed = NULL) {
  y <- as.numeric(y)
  n <- length(y)
  k <- as.integer(.gpb_clamp(k, 2L, max(2L, n)))
  restore_rng <- .gpb_preserve_global_rng()
  on.exit(restore_rng(), add = TRUE)
  if (!is.null(seed) && is.finite(seed)) set.seed(as.integer(seed))

  fold_id <- rep(NA_integer_, n)
  ok <- is.finite(y)
  if (sum(ok) >= k) {
    probs <- seq(0, 1, length.out = min(6L, sum(ok)) + 1L)
    qs <- unique(as.numeric(stats::quantile(y[ok], probs = probs,
                                            na.rm = TRUE, type = 7L)))
    if (length(qs) >= 3L) {
      bins <- cut(y[ok], breaks = qs, include.lowest = TRUE, labels = FALSE)
    } else {
      bins <- rep(1L, sum(ok))
    }
    ok_idx <- which(ok)
    for (b in sort(unique(bins))) {
      idx <- sample(ok_idx[bins == b])
      fold_id[idx] <- rep(seq_len(k), length.out = length(idx))
    }
  }
  missing <- which(!is.finite(fold_id))
  if (length(missing) > 0L) {
    fold_id[missing] <- rep(seq_len(k), length.out = length(missing))
  }
  empty <- setdiff(seq_len(k), unique(fold_id))
  if (length(empty) > 0L) {
    idx <- sample(seq_len(n), length(empty))
    fold_id[idx] <- empty
  }
  lapply(seq_len(k), function(j) which(fold_id == j))
}

.gpb_build_tune_specs <- function(cfg,
                                  learning_rate = NULL,
                                  max_depth = NULL,
                                  num_leaves = NULL,
                                  matern_nu = NULL,
                                  feature_fraction = NULL,
                                  min_data_in_leaf = NULL,
                                  bagging_fraction = NULL,
                                  lambda_l2 = NULL,
                                  learning_rate_grid = NULL,
                                  max_depth_grid = NULL,
                                  matern_nu_grid = NULL,
                                  feature_fraction_grid = NULL,
                                  min_data_in_leaf_grid = NULL,
                                  bagging_fraction_grid = NULL,
                                  lambda_l2_grid = NULL,
                                  range_init_grid = NULL,
                                  nugget_init_grid = NULL,
                                  fair_tune = TRUE,
                                  max_tune_candidates = Inf,
                                  include_tree = TRUE,
                                  include_spatial = TRUE) {
  base_lr <- if (is.null(learning_rate)) cfg$learning_rate else as.numeric(learning_rate)
  base_depth <- if (is.null(max_depth)) cfg$max_depth else as.integer(max_depth)
  base_nu <- if (is.null(matern_nu)) cfg$matern_nu else as.numeric(matern_nu)
  base_feat <- if (is.null(feature_fraction)) cfg$feature_fraction else as.numeric(feature_fraction)
  base_min_leaf <- if (is.null(min_data_in_leaf)) cfg$min_data_in_leaf else as.integer(min_data_in_leaf)
  base_bag <- if (is.null(bagging_fraction)) cfg$bagging_fraction else as.numeric(bagging_fraction)
  base_l2 <- if (is.null(lambda_l2)) cfg$lambda_l2 else as.numeric(lambda_l2)

  lr_vals <- if (isTRUE(fair_tune) && is.null(learning_rate)) {
    .gpb_num_grid(learning_rate_grid,
                  unique(c(base_lr, 0.02, 0.03, 0.05, 0.08, 0.10)))
  } else base_lr
  depth_vals <- if (isTRUE(fair_tune) && is.null(max_depth)) {
    .gpb_int_grid(max_depth_grid, unique(c(base_depth,
                                           3L, 4L, 5L, 6L, 7L, 8L)))
  } else base_depth
  nu_vals <- if (isTRUE(fair_tune) && is.null(matern_nu)) {
    .gpb_num_grid(matern_nu_grid, unique(c(base_nu, 0.5, 1.5, 2.5)))
  } else .gpb_num_grid(matern_nu_grid, base_nu)
  feat_vals <- if (isTRUE(fair_tune) && is.null(feature_fraction)) {
    .gpb_num_grid(feature_fraction_grid,
                  unique(c(base_feat, 0.60, 0.75, 0.875, 0.95, 1.00)))
  } else .gpb_num_grid(feature_fraction_grid, base_feat)
  min_leaf_vals <- if (isTRUE(fair_tune) && is.null(min_data_in_leaf)) {
    .gpb_int_grid(min_data_in_leaf_grid,
                  unique(c(base_min_leaf, 3L, 5L, 10L, 15L, 20L, 30L,
                           max(5L, as.integer(round(base_min_leaf * 2))))))
  } else .gpb_int_grid(min_data_in_leaf_grid, base_min_leaf)
  bag_vals <- if (isTRUE(fair_tune) && is.null(bagging_fraction)) {
    .gpb_num_grid(bagging_fraction_grid,
                  unique(c(base_bag, 0.70, 0.80, 0.90, 1.00)))
  } else .gpb_num_grid(bagging_fraction_grid, base_bag)
  l2_vals <- if (isTRUE(fair_tune) && is.null(lambda_l2)) {
    .gpb_num_grid(lambda_l2_grid, unique(c(base_l2, 0, 0.1, 1, 5, 10)))
  } else .gpb_num_grid(lambda_l2_grid, base_l2)
  range_vals <- if (isTRUE(fair_tune)) {
    .gpb_num_grid(range_init_grid, c(1, 0.5, 2, 0.25, 4))
  } else 1
  nugget_vals <- if (isTRUE(fair_tune)) {
    .gpb_num_grid(nugget_init_grid, c(1, 0.5, 2))
  } else 1

  mk <- function(mode, lr, depth, nu, feat, min_leaf, bag, l2,
                 range_mult = 1, nugget_mult = 1) {
    leaves <- if (is.null(num_leaves)) as.integer(min(63L, 2L ^ depth - 1L))
              else as.integer(num_leaves)
    list(mode = mode,
         learning_rate = as.numeric(lr),
         max_depth = as.integer(depth),
         num_leaves = leaves,
         matern_nu = as.numeric(nu),
         feature_fraction = as.numeric(feat),
         min_data_in_leaf = as.integer(min_leaf),
         bagging_fraction = as.numeric(bag),
         lambda_l2 = as.numeric(l2),
         range_multiplier = as.numeric(range_mult),
         nugget_multiplier = as.numeric(nugget_mult))
  }

  tree_specs <- list()
  if (isTRUE(include_tree)) {
    for (lr in lr_vals) {
      for (depth in depth_vals) {
        for (feat in feat_vals) {
          for (ml in min_leaf_vals) {
            for (bag in bag_vals) {
              for (l2 in l2_vals) {
                tree_specs[[length(tree_specs) + 1L]] <- mk(
                  "tree", lr, depth, base_nu, feat, ml, bag, l2)
              }
            }
          }
        }
      }
    }
  }
  spatial_specs <- list()
  if (isTRUE(include_spatial)) {
    for (lr in lr_vals) {
      for (depth in depth_vals) {
        for (nu in nu_vals) {
          for (feat in feat_vals) {
            for (ml in min_leaf_vals) {
              for (bag in 1) {
                for (l2 in l2_vals) {
                  for (rm in range_vals) {
                    for (nm in nugget_vals) {
                      spatial_specs[[length(spatial_specs) + 1L]] <- mk(
                        "spatial", lr, depth, nu, feat, ml, bag, l2, rm, nm)
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  specs <- c(tree_specs, spatial_specs)
  ids <- vapply(specs, function(s) {
    paste(s$mode, s$learning_rate, s$max_depth, s$num_leaves, s$matern_nu,
          s$feature_fraction, s$min_data_in_leaf, s$bagging_fraction,
          s$lambda_l2, s$range_multiplier, s$nugget_multiplier, sep = "|")
  }, character(1))
  specs <- specs[!duplicated(ids)]
  attr(specs, "candidate_universe_before_cap") <- length(specs)
  attr(specs, "tree_universe_before_cap") <- length(tree_specs)
  attr(specs, "spatial_universe_before_cap") <- length(spatial_specs)
  if (!is.null(max_tune_candidates) && is.finite(max_tune_candidates) &&
      max_tune_candidates > 0L && length(specs) > max_tune_candidates) {
    cap <- as.integer(max_tune_candidates)
    mode <- vapply(specs, `[[`, character(1), "mode")
    tree_pool <- specs[mode == "tree"]
    spatial_pool <- specs[mode == "spatial"]
    n_tree <- min(length(tree_pool), max(1L, floor(cap * 0.50)))
    if (cap >= 72L && length(tree_pool) <= 36L) n_tree <- length(tree_pool)
    n_spatial <- max(0L, cap - n_tree)
    specs <- c(.gpb_select_evenly(tree_pool, n_tree),
               .gpb_select_evenly(spatial_pool, n_spatial))
  }
  attr(specs, "candidate_universe_after_cap") <- length(specs)
  specs
}

# ---- Target transform auto-selection ---------------------------------------

gpboost_target_transform_auto <- function(y_train) {
  y_clean <- y_train[is.finite(y_train)]
  n       <- length(y_clean)
  if (n < 3L) return("identity")
  if (any(y_clean <= 0)) return("identity")
  n_sw   <- min(n, 5000L)
  y_samp <- if (n_sw < n) {
    restore_rng <- .gpb_preserve_global_rng()
    on.exit(restore_rng(), add = TRUE)
    set.seed(1L)
    sample(y_clean, n_sw)
  } else y_clean
  sw     <- tryCatch(shapiro.test(y_samp),
                     error = function(e) list(p.value = 1))
  p <- if (is.null(sw$p.value)) 1 else sw$p.value
  if (!is.finite(p)) p <- 1
  if (p < 0.05) "log" else "identity"
}

gpboost_apply_target_transform <- function(y, method) {
  if (identical(method, "log")) {
    if (any(y <= 0, na.rm = TRUE)) {
      stop("GPBoostDSM: log transform requires strictly positive y.")
    }
    log(y)
  } else y
}

gpboost_invert_target_transform <- function(y, method) {
  if (identical(method, "log")) exp(y) else y
}

# ---- Robust Duan smearing (gated on n_val) ---------------------------------

gpboost_duan_smearing <- function(log_residuals,
                                  trim_frac = 0.05,
                                  min_val_n = 500L) {
  r <- log_residuals[is.finite(log_residuals)]
  n <- length(r)
  if (n < as.integer(min_val_n)) {
    return(list(S = 1.0, applied = FALSE,
                reason = sprintf("n_val=%d < min=%d", n, min_val_n)))
  }
  er <- exp(r)
  trim_frac <- max(0, min(0.25, as.numeric(trim_frac)))
  q  <- quantile(er, c(trim_frac, 1 - trim_frac), names = FALSE,
                 na.rm = TRUE, type = 7L)
  er_trim <- er[er >= q[1] & er <= q[2]]
  S_trim  <- mean(er_trim, na.rm = TRUE)
  S_raw   <- mean(er,      na.rm = TRUE)
  if (!is.finite(S_trim) || S_trim <= 0) {
    return(list(S = 1.0, S_raw = S_raw, S_trimmed = NA_real_,
                applied = FALSE, n_used = length(er_trim),
                reason = sprintf("S non-finite (raw=%.3f)", S_raw)))
  }
  list(S = S_trim, S_raw = S_raw, S_trimmed = S_trim,
       applied = TRUE, n_used = length(er_trim),
       reason = sprintf("trimmed-mean Duan applied (n=%d, trim=%.2f)",
                        length(er_trim), trim_frac))
}

# ---- Variogram fit (gstat, no torch) ---------------------------------------

gpboost_fit_variogram <- function(y, coords, cutoff_frac = 0.50,
                                  n_lags = 15L) {
  if (!requireNamespace("gstat", quietly = TRUE) ||
      !requireNamespace("sp",    quietly = TRUE)) {
    stop("GPBoostDSM requires gstat + sp. install.packages(c('gstat','sp')).")
  }
  y <- as.numeric(y)
  coords <- as.matrix(coords)
  ok <- is.finite(y) & apply(is.finite(coords), 1L, all)
  y <- y[ok]
  coords <- coords[ok, , drop = FALSE]
  n_eff <- length(y)

  fallback <- function(status, range_major = NA_real_, nugget_ratio = 0.50,
                       fit_quality = 0) {
    if (!is.finite(range_major) || range_major <= 0) range_major <- 1
    list(range_major = range_major, nugget_ratio = nugget_ratio,
         fit_quality = fit_quality, variogram_status = status,
         variogram_n = n_eff, variogram_kappa = NA_real_,
         coord_trend_r2 = NA_real_,
         stationarity_decision = "undetermined",
         spatial_recommended = FALSE)
  }

  if (n_eff < 10L || ncol(coords) < 1L) {
    return(fallback("too_few_points"))
  }

  max_vg_n <- as.integer(.gpb_env_num("GPB_VARIOGRAM_MAX_N", 2000L))
  if (is.finite(max_vg_n) && max_vg_n > 10L && n_eff > max_vg_n) {
    restore_rng <- .gpb_preserve_global_rng()
    on.exit(restore_rng(), add = TRUE)
    set.seed(1L)
    idx <- sample.int(n_eff, max_vg_n)
    y <- y[idx]
    coords <- coords[idx, , drop = FALSE]
    n_eff <- length(y)
  }

  D <- as.matrix(stats::dist(coords))
  dvals <- D[upper.tri(D)]
  dvals <- dvals[is.finite(dvals) & dvals > 0]
  if (length(dvals) < 3L) return(fallback("degenerate_coordinates"))
  max_dist <- max(dvals)
  cutoff <- max_dist * cutoff_frac
  if (!is.finite(cutoff) || cutoff <= 0) return(fallback("invalid_cutoff"))

  coord_scaler <- .gpb_coord_scaler(coords)
  coord_scaled <- .gpb_scale_coords(coords, coord_scaler)
  coord_trend_r2 <- .gpb_linear_r2(y, coord_scaled)
  stationarity_decision <- if (!is.finite(coord_trend_r2)) {
    "undetermined"
  } else if (coord_trend_r2 <= 0.20) {
    "stationary_residual"
  } else if (coord_trend_r2 <= 0.45) {
    "moderate_coordinate_trend"
  } else {
    "nonstationary_coordinate_trend"
  }

  n_sw <- min(length(y), 5000L)
  y_samp <- if (n_sw < length(y)) {
    restore_rng <- .gpb_preserve_global_rng()
    on.exit(restore_rng(), add = TRUE)
    set.seed(1L)
    sample(y, n_sw)
  } else y
  sw_p <- tryCatch(
    shapiro.test(y_samp)$p.value,
    error = function(e) 1
  )
  y_vg <- if (sw_p < 0.05) {
    qx <- (rank(y) - 0.5) / length(y)
    qnorm(.gpb_clamp(qx, 1e-6, 1 - 1e-6))
  } else y

  coord_df <- as.data.frame(coords)
  names(coord_df) <- paste0("c", seq_len(ncol(coord_df)))
  df <- cbind(data.frame(yv = y_vg), coord_df)
  sp::coordinates(df) <- stats::as.formula(
    paste("~", paste(names(coord_df), collapse = "+")))

  vg_emp <- tryCatch(
    gstat::variogram(yv ~ 1, data = df, cutoff = cutoff,
                     width = cutoff / n_lags, cressie = TRUE),
    error = function(e) NULL
  )
  if (is.null(vg_emp) || nrow(vg_emp) < 3L) {
    return(fallback("empirical_variogram_failed",
                    range_major = cutoff * 0.30,
                    nugget_ratio = 0.50))
  }
  sill_tot <- var(y_vg, na.rm = TRUE)
  if (!is.finite(sill_tot) || sill_tot <= 0) {
    return(fallback("zero_variance",
                    range_major = cutoff * 0.30,
                    nugget_ratio = 0.95))
  }
  range_init <- cutoff * 0.30
  fit_one <- function(kappa) {
    fit <- tryCatch(
      suppressWarnings(gstat::fit.variogram(
        vg_emp,
        gstat::vgm(sill_tot * 0.80, "Mat", range_init, sill_tot * 0.20,
                   kappa = kappa),
        fit.method = 7L)),
      error = function(e) NULL
    )
    if (is.null(fit) || isTRUE(attr(fit, "singular")) ||
        any(!is.finite(fit$psill)) || any(fit$psill < 0) ||
        any(!is.finite(fit$range))) {
      return(NULL)
    }
    line <- tryCatch(
      gstat::variogramLine(fit, dist_vector = vg_emp$dist),
      error = function(e) NULL
    )
    if (is.null(line) || !("gamma" %in% names(line))) return(NULL)
    sse <- sum((vg_emp$gamma - line$gamma) ^ 2 * vg_emp$np, na.rm = TRUE)
    list(fit = fit, kappa = kappa, sse = sse)
  }
  fits <- lapply(c(0.5, 1.5, 2.5), fit_one)
  fits <- fits[!vapply(fits, is.null, logical(1))]
  if (length(fits) == 0L) {
    return(fallback("variogram_fit_failed",
                    range_major = range_init,
                    nugget_ratio = 0.50))
  }
  best <- fits[[which.min(vapply(fits, `[[`, numeric(1), "sse"))]]
  vg_fit <- best$fit
  iso_range <- max(vg_fit$range[vg_fit$model != "Nug" &
                                  vg_fit$range > 0], na.rm = TRUE)
  if (!is.finite(iso_range) || iso_range <= 0) iso_range <- range_init
  nugget <- sum(vg_fit$psill[vg_fit$model == "Nug"], na.rm = TRUE)
  psill_sum <- sum(vg_fit$psill, na.rm = TRUE)
  nugget_ratio <- if (is.finite(psill_sum) && psill_sum > 1e-8) {
    .gpb_clamp(nugget / psill_sum, 0.02, 0.95)
  } else 0.50
  sst <- sum((vg_emp$gamma - mean(vg_emp$gamma)) ^ 2 * vg_emp$np,
             na.rm = TRUE)
  fit_q <- .gpb_clamp(1 - best$sse / max(sst, 1e-8), 0, 1)
  spatial_recommended <- is.finite(coord_trend_r2) &&
    coord_trend_r2 <= 0.45 && nugget_ratio < 0.85 && fit_q > 0.02
  list(range_major  = iso_range,
       nugget_ratio = nugget_ratio,
       fit_quality  = fit_q,
       variogram_status = "ok",
       variogram_n = n_eff,
       variogram_kappa = best$kappa,
       coord_trend_r2 = coord_trend_r2,
       stationarity_decision = stationarity_decision,
       spatial_recommended = spatial_recommended)
}

# ---- Auto-config -----------------------------------------------------------

gpboost_auto_config <- function(vg, n_train, n_features) {
  r        <- vg$nugget_ratio
  rho      <- vg$range_major
  fit_q    <- vg$fit_quality

  # Use the best empirical variogram kappa when available; otherwise keep a
  # conservative rough-field default and let inner validation tune nu.
  matern_nu <- if (!is.null(vg$variogram_kappa) &&
                   is.finite(vg$variogram_kappa)) {
    c(0.5, 1.5, 2.5)[which.min(abs(c(0.5, 1.5, 2.5) - vg$variogram_kappa))]
  } else if (fit_q >= 0.60) 2.5 else if (fit_q >= 0.30) 1.5 else 0.5

  # GPBoost capacity rules (sqrt-n style).
  max_depth     <- as.integer(.gpb_clamp(round(log2(max(n_train, 4L)) / 2 + 1),
                                         4L, 8L))
  num_leaves    <- as.integer(min(63L, 2L ^ max_depth - 1L))
  feature_frac  <- .gpb_clamp(1 - n_features / (10 * sqrt(n_train)), 0.5, 0.95)
  bagging_frac  <- 0.8
  min_data_leaf <- as.integer(.gpb_clamp(round(n_train / 100), 5L, 50L))
  lambda_l2     <- 0
  learning_rate <- 0.05
  nrounds_max   <- as.integer(.gpb_env_num("GPB_NROUNDS_MAX", 1500L))
  early_stop    <- as.integer(.gpb_clamp(round(50 + sqrt(n_train)), 30L, 100L))

  list(
    matern_nu        = matern_nu,
    range_init       = rho,
    nugget_ratio     = r,
    fit_quality      = fit_q,
    variogram_status = if (is.null(vg$variogram_status)) NA_character_ else vg$variogram_status,
    variogram_n      = if (is.null(vg$variogram_n)) NA_integer_ else vg$variogram_n,
    variogram_kappa  = if (is.null(vg$variogram_kappa)) NA_real_ else vg$variogram_kappa,
    coord_trend_r2   = if (is.null(vg$coord_trend_r2)) NA_real_ else vg$coord_trend_r2,
    stationarity_decision = if (is.null(vg$stationarity_decision)) {
                              NA_character_
                            } else vg$stationarity_decision,
    spatial_recommended = if (is.null(vg$spatial_recommended)) {
                            NA
                          } else isTRUE(vg$spatial_recommended),
    learning_rate    = learning_rate,
    max_depth        = max_depth,
    num_leaves       = num_leaves,
    feature_fraction = feature_frac,
    bagging_fraction = bagging_frac,
    min_data_in_leaf = min_data_leaf,
    lambda_l2        = lambda_l2,
    nrounds_max      = nrounds_max,
    early_stop       = early_stop,
    duan_min_val_n   = as.integer(.gpb_env_num("GPB_DUAN_MIN_VAL_N", 500L)),
    duan_trim_frac   = .gpb_env_num("GPB_DUAN_TRIM_FRAC", 0.05),
    duan_apply       = toupper(Sys.getenv("GPB_DUAN_APPLY", unset = "TRUE")) %in%
                       c("TRUE", "1", "YES", "Y")
  )
}

# ---- Main training ---------------------------------------------------------

train_gpboost_dsm_one_fold <- function(
  fd,
  target_transform = "auto",
  matern_nu        = NULL,
  learning_rate    = NULL,
  max_depth        = NULL,
  num_leaves       = NULL,
  nrounds_max      = NULL,
  early_stop       = NULL,
  feature_fraction = NULL,
  bagging_fraction = NULL,
  min_data_in_leaf = NULL,
  lambda_l2        = NULL,
  tune_inner_folds = 3L,
  leaves_newton_update       = TRUE,
  use_gp_model_for_validation = FALSE,
  tree_fallback              = TRUE,
  fair_tune                  = TRUE,
  refit_full                 = TRUE,
  learning_rate_grid         = NULL,
  max_depth_grid             = NULL,
  matern_nu_grid             = NULL,
  feature_fraction_grid      = NULL,
  min_data_in_leaf_grid      = NULL,
  bagging_fraction_grid      = NULL,
  lambda_l2_grid             = NULL,
  range_init_grid            = NULL,
  nugget_init_grid           = NULL,
  max_tune_candidates        = 72L,
  autotune_v2                = TRUE,
  stage1_max_candidates      = 96L,
  stage2_top_k               = 12L,
  stage2_max_candidates      = 48L,
  stage2_select_only         = FALSE,
  crash_safe                 = TRUE,
  spatial_policy             = "tree_only",
  rmse_sd_weight             = 0.25,
  quantile_bins              = 5L,
  quantile_excess_weight     = 0.15,
  quantile_sd_weight         = 0.05,
  ensemble_top_k             = 1L,
  ensemble_weighting         = "softmax",
  ensemble_temperature       = 0.75,
  selection_objective        = "rmse",
  policy_blend               = FALSE,
  blend_alpha_grid           = c(0, 0.10, 0.25, 0.50, 0.75, 1),
	  residual_spatial_correction = FALSE,
	  residual_gamma_grid        = c(0, 0.25, 0.50, 0.75, 1),
	  seed_ensemble_n            = 1L,
		  noharm_tree_anchor         = TRUE,
		  noharm_margin_abs          = 0.05,
		  noharm_margin_rel          = 0.0,
		  noharm_tree_only_ensemble  = TRUE,
		  rf_noharm_policy           = TRUE,
		  rf_noharm_margin_abs       = 0.05,
		  rf_noharm_margin_rel       = 0.0,
		  rf_noharm_tree_only_ensemble = TRUE,
		  spatial_filter             = TRUE,
  spatial_filter_max_alpha_no_signal = 0.25,
  spatial_filter_max_gamma_no_signal = 0.25,
  tune_max_seconds           = Inf,
  tune_min_candidates_before_timeout = 12L,
  tune_progress_file         = NULL,
  num_threads                = NULL,
  train_seed       = NULL,
  ...
) {
  if (!requireNamespace("gpboost", quietly = TRUE)) {
    stop("GPBoostDSM requires the gpboost package. install.packages('gpboost').")
  }

  # ---- Fold data ----------------------------------------------------------
  Xtr <- as.matrix(fd$X$train)
  Xva <- as.matrix(fd$X$val)
  Xte <- as.matrix(fd$X$test)
  Ctr <- as.matrix(fd$coords$train)
  Cva <- as.matrix(fd$coords$val)
  Cte <- as.matrix(fd$coords$test)
  ytr <- fd$y$train; yva <- fd$y$val; yte <- fd$y$test
  n_train <- nrow(Xtr)
  n_val   <- nrow(Xva)
  tune_inner_folds <- suppressWarnings(as.integer(tune_inner_folds))
  if (!is.finite(tune_inner_folds)) tune_inner_folds <- 1L
  tune_inner_folds <- max(1L, tune_inner_folds)

  has_full_fd <- !is.null(fd$X$full) && !is.null(fd$coords$full) &&
    !is.null(fd$y$full) && nrow(fd$X$full) > 0L
  Xcal <- if (has_full_fd) as.matrix(fd$X$full) else rbind(Xtr, Xva)
  Ccal <- if (has_full_fd) as.matrix(fd$coords$full) else rbind(Ctr, Cva)
  ycal <- if (has_full_fd) as.numeric(fd$y$full) else c(ytr, yva)
  use_cv_tuning <- tune_inner_folds > 1L &&
    nrow(Xcal) >= max(20L, 2L * tune_inner_folds)
  if (!use_cv_tuning) tune_inner_folds <- 1L

  # ---- Target transform ---------------------------------------------------
  target_transform <- tolower(as.character(target_transform))
  if (identical(target_transform, "auto")) {
    target_transform <- gpboost_target_transform_auto(
      if (use_cv_tuning) ycal else ytr)
    cat(sprintf("[GPBoost-DSM] target_transform auto-selected -> '%s'\n",
                target_transform))
  }
  ytr_t <- gpboost_apply_target_transform(ytr, target_transform)
  yva_t <- gpboost_apply_target_transform(yva, target_transform)
  ycal_t <- gpboost_apply_target_transform(ycal, target_transform)

  # ---- Variogram + auto-config -------------------------------------------
  cat("[GPBoost-DSM] === Variogram fit ===\n")
  Xcfg <- if (use_cv_tuning) Xcal else Xtr
  Ccfg <- if (use_cv_tuning) Ccal else Ctr
  ycfg_t <- if (use_cv_tuning) ycal_t else ytr_t
  vg_signal <- .gpb_tabular_residual_signal(Xcfg, ycfg_t)
  vg  <- gpboost_fit_variogram(vg_signal$signal, Ccfg)
  cfg <- gpboost_auto_config(vg, nrow(Xcfg), ncol(Xcfg))
  cfg$variogram_signal_method <- vg_signal$method
  cfg$tabular_trend_r2 <- vg_signal$tabular_trend_r2

  # Runner overrides ------------------------------------------------------
  if (!is.null(matern_nu))        cfg$matern_nu        <- as.numeric(matern_nu)
  if (!is.null(learning_rate))    cfg$learning_rate    <- as.numeric(learning_rate)
  if (!is.null(max_depth))        cfg$max_depth        <- as.integer(max_depth)
  if (!is.null(num_leaves))       cfg$num_leaves       <- as.integer(num_leaves)
  if (!is.null(nrounds_max))      cfg$nrounds_max      <- as.integer(nrounds_max)
  if (!is.null(early_stop))       cfg$early_stop       <- as.integer(early_stop)
  if (!is.null(feature_fraction)) cfg$feature_fraction <- as.numeric(feature_fraction)
  if (!is.null(bagging_fraction)) cfg$bagging_fraction <- as.numeric(bagging_fraction)
  if (!is.null(min_data_in_leaf)) cfg$min_data_in_leaf <- as.integer(min_data_in_leaf)
  if (!is.null(lambda_l2))        cfg$lambda_l2        <- as.numeric(lambda_l2)
  leaves_newton_update <- isTRUE(leaves_newton_update)
  use_gp_model_for_validation <- isTRUE(use_gp_model_for_validation)
  tree_fallback <- isTRUE(tree_fallback)
  fair_tune <- isTRUE(fair_tune)
  refit_full <- isTRUE(refit_full)
  autotune_v2 <- isTRUE(autotune_v2)
  crash_safe <- isTRUE(crash_safe)
  requested_spatial_policy <- tolower(as.character(spatial_policy)[1L])
  valid_spatial_policies <- c("auto", "tree_only", "mixed", "spatial_only")
  if (!nzchar(requested_spatial_policy) ||
      !(requested_spatial_policy %in% valid_spatial_policies)) {
    requested_spatial_policy <- "tree_only"
  }
  if (!tree_fallback && identical(requested_spatial_policy, "auto")) {
    requested_spatial_policy <- "spatial_only"
  }
  effective_spatial_policy <- requested_spatial_policy
  if (identical(effective_spatial_policy, "auto")) {
    effective_spatial_policy <- if (isTRUE(crash_safe)) "tree_only" else "mixed"
  }
  spatial_policy_override_reason <- NA_character_
  if (isTRUE(crash_safe) &&
      effective_spatial_policy %in% c("mixed", "spatial_only")) {
    spatial_policy_override_reason <- paste(
      "GPB_CRASH_SAFE=TRUE forces tree_only because native spatial",
      "GPBoost candidates can segfault outside R error handling")
    effective_spatial_policy <- "tree_only"
  }
  include_tree_candidates <- effective_spatial_policy %in% c("tree_only", "mixed")
  include_spatial_candidates <- effective_spatial_policy %in% c("mixed", "spatial_only")
  stage1_max_candidates <- as.integer(stage1_max_candidates)
  if (!is.finite(stage1_max_candidates) || stage1_max_candidates < 1L) {
    stage1_max_candidates <- 96L
  }
  stage2_top_k <- as.integer(stage2_top_k)
  if (!is.finite(stage2_top_k) || stage2_top_k < 1L) stage2_top_k <- 12L
  stage2_max_candidates <- as.integer(stage2_max_candidates)
  if (!is.finite(stage2_max_candidates) || stage2_max_candidates < 1L) {
    stage2_max_candidates <- 48L
  }
  stage2_select_only <- isTRUE(stage2_select_only)
  rmse_sd_weight <- as.numeric(rmse_sd_weight)
  if (!is.finite(rmse_sd_weight) || rmse_sd_weight < 0) rmse_sd_weight <- 0
  quantile_bins <- suppressWarnings(as.integer(quantile_bins))
  if (!is.finite(quantile_bins) || quantile_bins < 2L) quantile_bins <- 0L
  quantile_excess_weight <- as.numeric(quantile_excess_weight)
  if (!is.finite(quantile_excess_weight) || quantile_excess_weight < 0) {
    quantile_excess_weight <- 0
  }
  quantile_sd_weight <- as.numeric(quantile_sd_weight)
  if (!is.finite(quantile_sd_weight) || quantile_sd_weight < 0) {
    quantile_sd_weight <- 0
  }
  ensemble_top_k <- suppressWarnings(as.integer(ensemble_top_k))
  if (!is.finite(ensemble_top_k) || ensemble_top_k < 1L) ensemble_top_k <- 1L
  ensemble_weighting <- tolower(as.character(ensemble_weighting)[1L])
  if (!(ensemble_weighting %in% c("softmax", "inverse", "uniform"))) {
    ensemble_weighting <- "softmax"
  }
  ensemble_temperature <- as.numeric(ensemble_temperature)
  if (!is.finite(ensemble_temperature) || ensemble_temperature <= 0) {
    ensemble_temperature <- 0.75
  }
  selection_objective <- tolower(as.character(selection_objective)[1L])
  if (!(selection_objective %in% c("rmse", "paired_rf_delta"))) {
    selection_objective <- "rmse"
  }
  policy_blend <- isTRUE(policy_blend)
  blend_alpha_grid <- sort(unique(as.numeric(blend_alpha_grid)))
  blend_alpha_grid <- blend_alpha_grid[is.finite(blend_alpha_grid)]
  blend_alpha_grid <- .gpb_clamp(blend_alpha_grid, 0, 1)
  if (length(blend_alpha_grid) == 0L) {
    blend_alpha_grid <- c(0, 0.10, 0.25, 0.50, 0.75, 1)
  }
  residual_spatial_correction <- isTRUE(residual_spatial_correction)
  residual_gamma_grid <- sort(unique(as.numeric(residual_gamma_grid)))
  residual_gamma_grid <- residual_gamma_grid[is.finite(residual_gamma_grid)]
  residual_gamma_grid <- .gpb_clamp(residual_gamma_grid, 0, 1)
  if (length(residual_gamma_grid) == 0L) {
    residual_gamma_grid <- c(0, 0.25, 0.50, 0.75, 1)
  }
	  seed_ensemble_n <- suppressWarnings(as.integer(seed_ensemble_n))
	  if (!is.finite(seed_ensemble_n) || seed_ensemble_n < 1L) seed_ensemble_n <- 1L
	  seed_ensemble_n <- min(seed_ensemble_n, 5L)
	  noharm_tree_anchor <- isTRUE(noharm_tree_anchor)
	  noharm_margin_abs <- as.numeric(noharm_margin_abs)
	  if (!is.finite(noharm_margin_abs) || noharm_margin_abs < 0) {
	    noharm_margin_abs <- 0
	  }
	  noharm_margin_rel <- as.numeric(noharm_margin_rel)
		  if (!is.finite(noharm_margin_rel) || noharm_margin_rel < 0) {
		    noharm_margin_rel <- 0
		  }
		  noharm_tree_only_ensemble <- isTRUE(noharm_tree_only_ensemble)
		  rf_noharm_policy <- isTRUE(rf_noharm_policy) &&
		    identical(selection_objective, "paired_rf_delta")
		  rf_noharm_margin_abs <- as.numeric(rf_noharm_margin_abs)
		  if (!is.finite(rf_noharm_margin_abs) || rf_noharm_margin_abs < 0) {
		    rf_noharm_margin_abs <- 0
		  }
		  rf_noharm_margin_rel <- as.numeric(rf_noharm_margin_rel)
		  if (!is.finite(rf_noharm_margin_rel) || rf_noharm_margin_rel < 0) {
		    rf_noharm_margin_rel <- 0
		  }
		  rf_noharm_tree_only_ensemble <- isTRUE(rf_noharm_tree_only_ensemble)
		  spatial_filter <- isTRUE(spatial_filter)
  spatial_filter_max_alpha_no_signal <- as.numeric(spatial_filter_max_alpha_no_signal)
  if (!is.finite(spatial_filter_max_alpha_no_signal)) spatial_filter_max_alpha_no_signal <- 0.25
  spatial_filter_max_alpha_no_signal <- .gpb_clamp(spatial_filter_max_alpha_no_signal, 0, 1)
  spatial_filter_max_gamma_no_signal <- as.numeric(spatial_filter_max_gamma_no_signal)
  if (!is.finite(spatial_filter_max_gamma_no_signal)) spatial_filter_max_gamma_no_signal <- 0.25
  spatial_filter_max_gamma_no_signal <- .gpb_clamp(spatial_filter_max_gamma_no_signal, 0, 1)
  tune_max_seconds <- as.numeric(tune_max_seconds)
  if (!is.finite(tune_max_seconds) || tune_max_seconds <= 0) tune_max_seconds <- Inf
  tune_min_candidates_before_timeout <- as.integer(tune_min_candidates_before_timeout)
  if (!is.finite(tune_min_candidates_before_timeout) ||
      tune_min_candidates_before_timeout < 1L) {
    tune_min_candidates_before_timeout <- 1L
  }
  if (is.null(tune_progress_file) || length(tune_progress_file) == 0L) {
    tune_progress_file <- Sys.getenv("GPB_TUNE_PROGRESS_FILE", unset = "")
  }
  tune_progress_file <- as.character(tune_progress_file)[1L]
  if (!nzchar(tune_progress_file)) tune_progress_file <- NULL
  if (!is.null(num_threads)) {
    num_threads <- suppressWarnings(as.integer(num_threads))
    if (!is.finite(num_threads) || num_threads < 1L) num_threads <- NULL
  }
  num_threads_label <- if (is.null(num_threads)) "auto" else as.character(num_threads)
  max_tune_candidates <- if (is.null(max_tune_candidates) ||
                             !is.finite(max_tune_candidates)) {
    Inf
  } else {
    as.integer(max(1L, max_tune_candidates))
  }

  # Snap matern_nu to a value gpboost actually accepts (Matern only allows
  # 0.5, 1.5, 2.5 directly via cov_fct_shape).
  cfg$matern_nu <- c(0.5, 1.5, 2.5)[which.min(abs(c(0.5, 1.5, 2.5) -
                                                  cfg$matern_nu))]

  cat(sprintf(
    "[GPBoost-DSM] Geostat: r=%.3f  rho=%.3g  fit_q=%.2f  matern_nu=%.1f  stationarity=%s  signal=%s\n",
    cfg$nugget_ratio, cfg$range_init, cfg$fit_quality, cfg$matern_nu,
    cfg$stationarity_decision, cfg$variogram_signal_method))
  coord_scaler <- .gpb_coord_scaler(Ctr)
  Ctr_s <- .gpb_scale_coords(Ctr, coord_scaler)
  Cva_s <- .gpb_scale_coords(Cva, coord_scaler)
  Cte_s <- .gpb_scale_coords(Cte, coord_scaler)

  tune_specs <- .gpb_build_tune_specs(
    cfg = cfg,
    learning_rate = learning_rate,
    max_depth = max_depth,
    num_leaves = num_leaves,
    matern_nu = matern_nu,
    feature_fraction = feature_fraction,
    min_data_in_leaf = min_data_in_leaf,
    bagging_fraction = bagging_fraction,
    lambda_l2 = lambda_l2,
    learning_rate_grid = learning_rate_grid,
    max_depth_grid = max_depth_grid,
    matern_nu_grid = matern_nu_grid,
    feature_fraction_grid = feature_fraction_grid,
    min_data_in_leaf_grid = min_data_in_leaf_grid,
    bagging_fraction_grid = bagging_fraction_grid,
    lambda_l2_grid = lambda_l2_grid,
    range_init_grid = range_init_grid,
    nugget_init_grid = nugget_init_grid,
    fair_tune = fair_tune,
    max_tune_candidates = max_tune_candidates,
    include_tree = include_tree_candidates,
    include_spatial = include_spatial_candidates)
  get_tune_attr <- function(name, default) {
    value <- attr(tune_specs, name, exact = TRUE)
    value <- suppressWarnings(as.numeric(value)[1L])
    if (!is.finite(value)) default else value
  }
  tune_candidate_universe_before_cap <- get_tune_attr(
    "candidate_universe_before_cap", length(tune_specs))
  tune_tree_universe_before_cap <- get_tune_attr(
    "tree_universe_before_cap", NA_real_)
  tune_spatial_universe_before_cap <- get_tune_attr(
    "spatial_universe_before_cap", NA_real_)
  tune_candidate_universe_after_cap <- get_tune_attr(
    "candidate_universe_after_cap", length(tune_specs))
  if (length(tune_specs) == 0L) stop("GPBoostDSM: no tuning candidates.")

  cat(sprintf(
    "[GPBoost-DSM] Boost  : base_lr=%.3f  base_depth=%d  base_leaves=%d  feat_frac=%.2f  bag_frac=%.2f\n",
    cfg$learning_rate, cfg$max_depth, cfg$num_leaves,
    cfg$feature_fraction, cfg$bagging_fraction))
  cat(sprintf(
    "[GPBoost-DSM] Schedule: nrounds<=%d  early_stop=%d  target=%s  newton=%s  gp_val=%s  fallback=%s  fair_tune=%s  refit_full=%s  candidates=%d  tune_folds=%d  threads=%s  autotune_v2=%s  crash_safe=%s  spatial_policy=%s\n",
    cfg$nrounds_max, cfg$early_stop, target_transform,
    leaves_newton_update, use_gp_model_for_validation, tree_fallback,
    fair_tune, refit_full, length(tune_specs), tune_inner_folds,
    num_threads_label, autotune_v2, crash_safe, effective_spatial_policy))
  if (!is.na(spatial_policy_override_reason)) {
    cat(sprintf("[GPBoost-DSM] Spatial policy override: %s\n",
                spatial_policy_override_reason))
  }

  get_best_iter <- function(fit) {
    if (!is.null(fit$best_iter) && is.finite(fit$best_iter) && fit$best_iter > 0L) {
      as.integer(fit$best_iter)
    } else if (!is.null(fit$best_iteration) &&
               is.finite(fit$best_iteration) && fit$best_iteration > 0L) {
      as.integer(fit$best_iteration)
    } else cfg$nrounds_max
  }

  fit_candidate <- function(spec, X_train, y_train_t, C_train_s,
                            X_val = NULL, y_val_t = NULL, C_val_s = NULL,
                            X_pred = NULL, C_pred_s = NULL,
                            cfg_fit = cfg,
                            coord_scale = coord_scaler$scale,
                            nrounds_fit = cfg$nrounds_max,
                            with_validation = TRUE,
                            seed_offset = 0L) {
    params <- list(
      objective        = "regression_l2",
      metric           = "rmse",
      learning_rate    = spec$learning_rate,
      max_depth        = spec$max_depth,
      num_leaves       = spec$num_leaves,
      feature_fraction = spec$feature_fraction,
      bagging_fraction = if (identical(spec$mode, "spatial")) {
                            1.0
                          } else spec$bagging_fraction,
      bagging_freq     = if (!identical(spec$mode, "spatial") &&
                              is.finite(spec$bagging_fraction) &&
                              spec$bagging_fraction < 0.999) 1L else 0L,
      min_data_in_leaf = spec$min_data_in_leaf,
      lambda_l2        = spec$lambda_l2,
      seed             = if (is.null(train_seed)) as.integer(seed_offset)
                         else as.integer(train_seed + seed_offset),
      verbose          = -1
    )
    if (!is.null(num_threads)) params$num_threads <- num_threads
    if (identical(spec$mode, "spatial")) {
      params$leaves_newton_update <- leaves_newton_update
    }
    dtrain <- gpboost::gpb.Dataset(data = X_train, label = as.numeric(y_train_t))
    valids <- list()
    dval <- NULL
    if (isTRUE(with_validation) && !is.null(X_val) && !is.null(y_val_t)) {
      dval <- gpboost::gpb.Dataset(data = X_val, label = as.numeric(y_val_t),
                                   reference = dtrain)
      valids <- list(val = dval)
    }
    gp_model <- NULL
    init_cov_pars <- rep(NA_real_, 3L)
    if (identical(spec$mode, "spatial")) {
      gp_model <- gpboost::GPModel(
        gp_coords     = C_train_s,
        cov_function  = "matern",
        cov_fct_shape = spec$matern_nu,
        likelihood    = "gaussian",
        num_parallel_threads = num_threads,
        seed          = if (is.null(train_seed)) as.integer(seed_offset)
                        else as.integer(train_seed + seed_offset)
      )
      init_cov_pars <- .gpb_make_init_cov_pars(
        y_train_t, cfg_fit, coord_scale,
        range_multiplier = spec$range_multiplier,
        nugget_multiplier = spec$nugget_multiplier)
      gp_model$set_optim_params(params = list(init_cov_pars = init_cov_pars))
      if (isTRUE(with_validation) && isTRUE(use_gp_model_for_validation) &&
          !is.null(C_val_s)) {
        gp_model$set_prediction_data(gp_coords_pred = C_val_s)
      }
    }
    t0 <- Sys.time()
    fit <- tryCatch(
      gpboost::gpb.train(
        params                       = params,
        data                         = dtrain,
        gp_model                     = gp_model,
        nrounds                      = as.integer(max(1L, nrounds_fit)),
        valids                       = valids,
        early_stopping_rounds        = if (isTRUE(with_validation)) cfg_fit$early_stop else NULL,
        use_gp_model_for_validation  = if (isTRUE(with_validation) &&
                                            !is.null(gp_model)) {
                                          use_gp_model_for_validation
                                        } else FALSE,
        verbose                      = -1
      ),
      error = function(e) e
    )
    seconds <- as.numeric(Sys.time() - t0, units = "secs")
    if (inherits(fit, "error")) {
      return(list(status = "failed", error = conditionMessage(fit),
                  seconds = seconds, init_cov_pars = init_cov_pars,
                  best_iter = NA_integer_))
    }
    best_iter <- if (isTRUE(with_validation)) get_best_iter(fit)
                 else as.integer(max(1L, nrounds_fit))
    best_iter <- as.integer(.gpb_clamp(best_iter, 1L, max(1L, nrounds_fit)))
    pred_val_t <- pred_test_t <- pred_val_var <- pred_test_var <- NULL
    predict_at <- function(X, C, iter, with_var = TRUE) {
      if (identical(spec$mode, "spatial")) {
        pred_obj <- predict(fit, data = X, gp_coords_pred = C,
                            predict_var = with_var,
                            num_iteration = iter)
        list(mean = .gpb_extract_mean(pred_obj),
             var = .gpb_extract_var(pred_obj, nrow(X)))
      } else {
        list(mean = as.numeric(predict(fit, data = X, num_iteration = iter)),
             var = rep(NA_real_, nrow(X)))
      }
    }

    if (!is.null(X_val)) {
      pred_val_try <- tryCatch({
        max_iter_scan <- max(best_iter, best_iter + cfg_fit$early_stop)
        max_iter_scan <- min(as.integer(max_iter_scan), as.integer(nrounds_fit))
        iter_grid <- unique(as.integer(pmax(
          1L, pmin(max_iter_scan,
                   c(25L, 50L, 75L, 100L, 150L, 200L, 300L, best_iter)))))
        iter_grid <- sort(iter_grid)
        pred_grid <- lapply(iter_grid, function(it) {
          pr <- predict_at(X_val, C_val_s, it, with_var = TRUE)
          val_rmse <- .gpb_rmse(
            gpboost_invert_target_transform(y_val_t, target_transform),
            gpboost_invert_target_transform(pr$mean, target_transform))
          list(iter = it, rmse = val_rmse, pred = pr)
        })
        rmses <- vapply(pred_grid, `[[`, numeric(1), "rmse")
        j <- which.min(rmses)
        best_iter <- as.integer(pred_grid[[j]]$iter)
        pred_grid[[j]]$pred
      }, error = function(e) e)
      if (inherits(pred_val_try, "error")) {
        return(list(status = "failed",
                    error = paste("prediction failed:", conditionMessage(pred_val_try)),
                    seconds = seconds, init_cov_pars = init_cov_pars,
                    best_iter = best_iter))
      }
      pred_val_t <- pred_val_try$mean
      pred_val_var <- pred_val_try$var
    }
    if (!is.null(X_pred)) {
      pred_test_try <- tryCatch(predict_at(X_pred, C_pred_s, best_iter,
                                           with_var = TRUE),
                                error = function(e) e)
      if (inherits(pred_test_try, "error")) {
        return(list(status = "failed",
                    error = paste("prediction failed:", conditionMessage(pred_test_try)),
                    seconds = seconds, init_cov_pars = init_cov_pars,
                    best_iter = best_iter))
      }
      pred_test_t <- pred_test_try$mean
      pred_test_var <- pred_test_try$var
    }
    list(status = "ok", error = NA_character_, fit = fit, best_iter = best_iter,
         seconds = seconds, pred_val_t = pred_val_t, pred_test_t = pred_test_t,
         pred_val_var = pred_val_var, pred_test_var = pred_test_var,
         init_cov_pars = init_cov_pars)
  }

  fit_residual_gp <- function(res_train, C_train_raw, C_pred_raw,
                              cfg_fit = cfg,
                              seed_offset = 0L,
                              range_multiplier = 1,
                              nugget_multiplier = 1) {
    res_train <- as.numeric(res_train)
    scaler <- .gpb_coord_scaler(C_train_raw)
    C_train_s <- .gpb_scale_coords(C_train_raw, scaler)
    C_pred_s <- .gpb_scale_coords(C_pred_raw, scaler)
    X0 <- matrix(0, nrow = length(res_train), ncol = 1L)
    Xp <- matrix(0, nrow = nrow(C_pred_raw), ncol = 1L)
    params <- list(
      objective = "regression_l2",
      metric = "rmse",
      learning_rate = 0.03,
      max_depth = 1L,
      num_leaves = 2L,
      feature_fraction = 1,
      bagging_fraction = 1,
      bagging_freq = 0L,
      min_data_in_leaf = max(3L, min(20L, floor(length(res_train) / 20L))),
      lambda_l2 = 1,
      seed = if (is.null(train_seed)) as.integer(seed_offset)
             else as.integer(train_seed + seed_offset),
      verbose = -1
    )
    if (!is.null(num_threads)) params$num_threads <- num_threads
    dtrain <- gpboost::gpb.Dataset(data = X0, label = res_train)
    gp_model <- gpboost::GPModel(
      gp_coords = C_train_s,
      cov_function = "matern",
      cov_fct_shape = cfg_fit$matern_nu,
      likelihood = "gaussian",
      num_parallel_threads = num_threads,
      seed = if (is.null(train_seed)) as.integer(seed_offset)
             else as.integer(train_seed + seed_offset)
    )
    init_cov <- .gpb_make_init_cov_pars(
      res_train,
      cfg_fit,
      scaler$scale,
      range_multiplier = range_multiplier,
      nugget_multiplier = nugget_multiplier
    )
    gp_model$set_optim_params(params = list(init_cov_pars = init_cov))
    fit <- tryCatch(
      gpboost::gpb.train(
        params = params,
        data = dtrain,
        gp_model = gp_model,
        nrounds = 20L,
        use_gp_model_for_validation = FALSE,
        verbose = -1
      ),
      error = function(e) e
    )
    if (inherits(fit, "error")) {
      return(list(status = "failed", error = conditionMessage(fit),
                  pred = rep(NA_real_, nrow(C_pred_raw)),
                  init_cov_pars = init_cov))
    }
    pred <- tryCatch(
      predict(fit, data = Xp, gp_coords_pred = C_pred_s, predict_var = TRUE),
      error = function(e) e
    )
    if (inherits(pred, "error")) {
      return(list(status = "failed", error = conditionMessage(pred),
                  pred = rep(NA_real_, nrow(C_pred_raw)),
                  init_cov_pars = init_cov))
    }
    list(status = "ok", error = NA_character_,
         pred = .gpb_extract_mean(pred),
         var = .gpb_extract_var(pred, nrow(C_pred_raw)),
         init_cov_pars = init_cov)
  }

  tune_source <- if (use_cv_tuning) "cv_oof" else "single_holdout"
  cv_folds <- NULL
  cv_fold_data <- NULL
  cv_fold_sizes <- NA_character_
  if (use_cv_tuning) {
    cv_seed <- if (is.null(train_seed)) 7919L else as.integer(train_seed + 7919L)
    cv_folds <- .gpb_make_regression_folds(ycal, tune_inner_folds, seed = cv_seed)
    cv_fold_sizes <- paste(vapply(cv_folds, length, integer(1)), collapse = "|")
    cv_fold_data <- lapply(seq_along(cv_folds), function(j) {
      idx_val <- cv_folds[[j]]
      idx_train <- setdiff(seq_len(nrow(Xcal)), idx_val)
      scaler_j <- .gpb_coord_scaler(Ccal[idx_train, , drop = FALSE])
      sig_j <- .gpb_tabular_residual_signal(
        Xcal[idx_train, , drop = FALSE], ycal_t[idx_train])
      vg_j <- gpboost_fit_variogram(sig_j$signal, Ccal[idx_train, , drop = FALSE])
      cfg_j <- gpboost_auto_config(vg_j, length(idx_train), ncol(Xcal))
      cfg_j$nrounds_max <- cfg$nrounds_max
      cfg_j$early_stop <- cfg$early_stop
      cfg_j$duan_apply <- cfg$duan_apply
      cfg_j$duan_trim_frac <- cfg$duan_trim_frac
      cfg_j$duan_min_val_n <- cfg$duan_min_val_n
      list(
        fold = j,
        idx_train = idx_train,
        idx_val = idx_val,
        cfg = cfg_j,
        coord_scale = scaler_j$scale,
        X_train = Xcal[idx_train, , drop = FALSE],
        y_train_t = ycal_t[idx_train],
        C_train_s = .gpb_scale_coords(Ccal[idx_train, , drop = FALSE], scaler_j),
        X_val = Xcal[idx_val, , drop = FALSE],
        y_val_t = ycal_t[idx_val],
        C_val_s = .gpb_scale_coords(Ccal[idx_val, , drop = FALSE], scaler_j)
      )
    })
    cat(sprintf("[GPBoost-DSM] inner tuning CV (n_cal=%d, folds=%d, fold_sizes=%s)...\n",
                nrow(Xcal), tune_inner_folds, cv_fold_sizes))
  } else {
    cat(sprintf("[GPBoost-DSM] inner tuning holdout (n_train=%d, n_val=%d)...\n",
                n_train, n_val))
  }

  tune_obs <- if (use_cv_tuning) ycal else yva
  rf_tune_pred <- rep(NA_real_, length(tune_obs))
  rf_tune_fold_rmse <- NA_character_
  rf_tune_rmse <- NA_real_
  if (identical(selection_objective, "paired_rf_delta")) {
    if (use_cv_tuning) {
      rf_fold_rmse <- rep(NA_real_, length(cv_fold_data))
      for (j in seq_along(cv_fold_data)) {
        fdj <- cv_fold_data[[j]]
        pred_j <- .gpb_rf_predict_internal(
          X_train = Xcal[fdj$idx_train, , drop = FALSE],
          y_train = ycal[fdj$idx_train],
          X_pred = Xcal[fdj$idx_val, , drop = FALSE],
          # Use the Wadoux default RF seed so the in-gate RF matches the
          # scored RF bit-for-bit on each training subset.
          seed = as.integer(getOption("wadoux_rf_seed", 123L))
        )
        rf_tune_pred[fdj$idx_val] <- pred_j
        rf_fold_rmse[j] <- .gpb_rmse(ycal[fdj$idx_val], pred_j)
      }
      rf_tune_fold_rmse <- paste(ifelse(is.finite(rf_fold_rmse),
                                        sprintf("%.6f", rf_fold_rmse), "NA"),
                                 collapse = "|")
    } else {
      rf_tune_pred <- .gpb_rf_predict_internal(
        X_train = Xtr,
        y_train = ytr,
        X_pred = Xva,
        seed = as.integer(getOption("wadoux_rf_seed", 123L))
      )
      rf_tune_fold_rmse <- sprintf("%.6f", .gpb_rmse(yva, rf_tune_pred))
    }
    rf_tune_rmse <- .gpb_rmse(tune_obs, rf_tune_pred)
    cat(sprintf("[GPBoost-DSM] paired-RF selection baseline RMSE=%.3f\n",
                rf_tune_rmse))
  }

  candidate_label <- function(i, spec) {
    sprintf("%02d_%s_lr%.3f_d%d_l%d_ml%d_b%.2f_l2%.2f_nu%.1f_f%.2f_r%.2f_ng%.2f",
            i, spec$mode, spec$learning_rate, spec$max_depth,
            spec$num_leaves, spec$min_data_in_leaf,
            spec$bagging_fraction, spec$lambda_l2,
            spec$matern_nu, spec$feature_fraction,
            spec$range_multiplier, spec$nugget_multiplier)
  }

  fit_candidate_cv <- function(spec, label, candidate_index) {
    fold_rmse <- rep(NA_real_, length(cv_fold_data))
    fold_best_iter <- rep(NA_integer_, length(cv_fold_data))
    fold_seconds <- rep(NA_real_, length(cv_fold_data))
    oof_t <- rep(NA_real_, nrow(Xcal))
    init_cov_pars <- rep(NA_real_, 3L)
    errors <- character(0)
    for (j in seq_along(cv_fold_data)) {
      fdj <- cv_fold_data[[j]]
      res_j <- fit_candidate(
        spec = spec,
        X_train = fdj$X_train, y_train_t = fdj$y_train_t,
        C_train_s = fdj$C_train_s,
        X_val = fdj$X_val, y_val_t = fdj$y_val_t, C_val_s = fdj$C_val_s,
        X_pred = NULL, C_pred_s = NULL,
        cfg_fit = fdj$cfg,
        coord_scale = fdj$coord_scale,
        nrounds_fit = cfg$nrounds_max,
        with_validation = TRUE,
        seed_offset = as.integer(candidate_index * 1000L + j))
      fold_seconds[j] <- res_j$seconds
      if (all(!is.finite(init_cov_pars)) &&
          !is.null(res_j$init_cov_pars) &&
          any(is.finite(res_j$init_cov_pars))) {
        init_cov_pars <- res_j$init_cov_pars
      }
      if (identical(res_j$status, "ok") && !is.null(res_j$pred_val_t)) {
        oof_t[fdj$idx_val] <- res_j$pred_val_t
        fold_best_iter[j] <- res_j$best_iter
        fold_rmse[j] <- .gpb_rmse(
          ycal[fdj$idx_val],
          gpboost_invert_target_transform(res_j$pred_val_t, target_transform))
      } else {
        errors <- c(errors, sprintf("fold%d: %s", j, res_j$error))
      }
    }
    ok_folds <- is.finite(fold_rmse)
    status <- if (all(ok_folds)) {
      "ok"
    } else if (any(ok_folds)) {
      "partial"
    } else {
      "failed"
    }
    best_iter <- if (any(is.finite(fold_best_iter))) {
      as.integer(stats::median(fold_best_iter[is.finite(fold_best_iter)]))
    } else NA_integer_
    oof_pred <- gpboost_invert_target_transform(oof_t, target_transform)
    val_rmse <- .gpb_rmse(ycal, oof_pred)
    qmet <- .gpb_quantile_rmse_values(ycal, oof_pred, quantile_bins)
    list(status = status,
         error = if (length(errors)) paste(unique(errors), collapse = " | ") else NA_character_,
         seconds = sum(fold_seconds, na.rm = TRUE),
         init_cov_pars = init_cov_pars,
         best_iter = best_iter,
         pred_val_t = oof_t,
         pred_val_var = rep(NA_real_, length(oof_t)),
         pred_test_t = NULL,
         pred_test_var = NULL,
         val_rmse = val_rmse,
         val_quantile_rmse_mean = qmet$mean,
         val_quantile_rmse_sd = qmet$sd,
         val_quantile_rmse_excess = qmet$excess,
         val_quantile_rmse = qmet$text,
         val_quantile_bins = qmet$n_bins,
         cv_rmse_mean = mean(fold_rmse, na.rm = TRUE),
         cv_rmse_sd = stats::sd(fold_rmse, na.rm = TRUE),
         cv_successful_folds = sum(ok_folds),
         cv_failed_folds = sum(!ok_folds),
         cv_best_iter_mean = mean(fold_best_iter, na.rm = TRUE),
         cv_best_iter_sd = stats::sd(fold_best_iter, na.rm = TRUE),
         cv_fold_rmse = paste(ifelse(is.finite(fold_rmse),
                                     sprintf("%.6f", fold_rmse), "NA"),
                              collapse = "|"))
  }

  score_prediction_vector <- function(obs, pred, cv_rmse_sd = NA_real_) {
    val_rmse <- .gpb_rmse(obs, pred)
    qmet <- .gpb_quantile_rmse_values(obs, pred, quantile_bins)
    score <- val_rmse
    delta_vs_rf <- NA_real_
    if (identical(selection_objective, "paired_rf_delta") &&
        is.finite(rf_tune_rmse)) {
      delta_vs_rf <- val_rmse - rf_tune_rmse
      score <- delta_vs_rf
    }
    if (is.finite(cv_rmse_sd)) {
      score <- score + rmse_sd_weight * cv_rmse_sd
    }
    if (is.finite(qmet$excess)) {
      score <- score + quantile_excess_weight * qmet$excess
    }
    if (is.finite(qmet$sd)) {
      score <- score + quantile_sd_weight * qmet$sd
    }
    list(
      val_rmse = val_rmse,
      delta_rmse_vs_rf = delta_vs_rf,
      qmet = qmet,
      score = if (!is.finite(score)) Inf else score
    )
  }

  candidate_score <- function(res) {
    score <- if (identical(selection_objective, "paired_rf_delta") &&
                 is.finite(rf_tune_rmse)) {
      res$val_rmse - rf_tune_rmse
    } else {
      res$val_rmse
    }
    if (is.finite(res$cv_rmse_sd)) {
      score <- score + rmse_sd_weight * res$cv_rmse_sd
    }
    if (is.finite(res$val_quantile_rmse_excess)) {
      score <- score + quantile_excess_weight * res$val_quantile_rmse_excess
    }
    if (is.finite(res$val_quantile_rmse_sd)) {
      score <- score + quantile_sd_weight * res$val_quantile_rmse_sd
    }
    if (!is.finite(score)) Inf else score
  }
  candidates <- list()
  candidate_rows <- list()
  evaluated_idx <- integer()
  tune_started <- Sys.time()
  timed_out <- FALSE
  save_live_progress <- function() {
    if (is.null(tune_progress_file) || length(candidate_rows) == 0L) return(invisible(NULL))
    progress_df <- do.call(rbind, candidate_rows)
    try({
      dir.create(dirname(tune_progress_file), recursive = TRUE, showWarnings = FALSE)
      write.csv(progress_df, tune_progress_file, row.names = FALSE)
    }, silent = TRUE)
    invisible(NULL)
  }
  run_candidate_indices <- function(indices, tune_stage) {
    for (i in indices) {
      if (i %in% evaluated_idx) next
      elapsed_total <- as.numeric(Sys.time() - tune_started, units = "secs")
      if (is.finite(tune_max_seconds) &&
          elapsed_total > tune_max_seconds &&
          length(candidate_rows) >= tune_min_candidates_before_timeout) {
        timed_out <<- TRUE
        cat(sprintf("[GPBoost-DSM] tune time limit reached after %.1f sec; selecting from %d evaluated candidates\n",
                    elapsed_total, length(candidate_rows)))
        break
      }
      spec <- tune_specs[[i]]
      label <- candidate_label(i, spec)
      if (use_cv_tuning) {
        res <- fit_candidate_cv(spec, label, i)
        val_rmse <- res$val_rmse
      } else {
        res <- fit_candidate(
          spec = spec,
          X_train = Xtr, y_train_t = ytr_t, C_train_s = Ctr_s,
          X_val = Xva, y_val_t = yva_t, C_val_s = Cva_s,
          X_pred = Xte, C_pred_s = Cte_s,
          cfg_fit = cfg,
          coord_scale = coord_scaler$scale,
          nrounds_fit = cfg$nrounds_max,
          with_validation = TRUE,
          seed_offset = i)
        val_pred <- if (!is.null(res$pred_val_t)) {
          gpboost_invert_target_transform(res$pred_val_t, target_transform)
        } else rep(NA_real_, n_val)
        val_rmse <- .gpb_rmse(yva, val_pred)
        qmet <- .gpb_quantile_rmse_values(yva, val_pred, quantile_bins)
        res$val_rmse <- val_rmse
        res$val_quantile_rmse_mean <- qmet$mean
        res$val_quantile_rmse_sd <- qmet$sd
        res$val_quantile_rmse_excess <- qmet$excess
        res$val_quantile_rmse <- qmet$text
        res$val_quantile_bins <- qmet$n_bins
        res$cv_rmse_mean <- val_rmse
        res$cv_rmse_sd <- NA_real_
        res$cv_successful_folds <- if (identical(res$status, "ok")) 1L else 0L
        res$cv_failed_folds <- if (identical(res$status, "ok")) 0L else 1L
        res$cv_best_iter_mean <- res$best_iter
        res$cv_best_iter_sd <- NA_real_
        res$cv_fold_rmse <- if (is.finite(val_rmse)) sprintf("%.6f", val_rmse) else "NA"
      }
      score <- candidate_score(res)
      candidate_rows[[length(candidate_rows) + 1L]] <<- data.frame(
        candidate_id = label,
        tune_stage = tune_stage,
        mode = spec$mode,
        status = res$status,
        error = res$error,
        tune_source = tune_source,
        tune_inner_folds = tune_inner_folds,
        crash_safe = crash_safe,
        requested_spatial_policy = requested_spatial_policy,
        spatial_policy = effective_spatial_policy,
        learning_rate = spec$learning_rate,
        max_depth = spec$max_depth,
        num_leaves = spec$num_leaves,
        matern_nu = spec$matern_nu,
        feature_fraction = spec$feature_fraction,
        min_data_in_leaf = spec$min_data_in_leaf,
        bagging_fraction = spec$bagging_fraction,
        lambda_l2 = spec$lambda_l2,
        range_multiplier = spec$range_multiplier,
        nugget_multiplier = spec$nugget_multiplier,
        init_error_variance = res$init_cov_pars[1L],
        init_nugget = res$init_cov_pars[1L],
        init_gp_var = res$init_cov_pars[2L],
        init_range = res$init_cov_pars[3L],
	        best_iter = res$best_iter,
	        val_rmse = val_rmse,
	        rf_inner_rmse = rf_tune_rmse,
	        delta_rmse_vs_rf = if (is.finite(rf_tune_rmse)) val_rmse - rf_tune_rmse else NA_real_,
	        delta_RMSE_vs_RF = if (is.finite(rf_tune_rmse)) val_rmse - rf_tune_rmse else NA_real_,
	        selection_objective = selection_objective,
        rf_inner_fold_rmse = rf_tune_fold_rmse,
        val_quantile_rmse_mean = res$val_quantile_rmse_mean,
        val_quantile_rmse_sd = res$val_quantile_rmse_sd,
        val_quantile_rmse_excess = res$val_quantile_rmse_excess,
        val_quantile_rmse = res$val_quantile_rmse,
        val_quantile_bins = res$val_quantile_bins,
        cv_rmse_mean = res$cv_rmse_mean,
        cv_rmse_sd = res$cv_rmse_sd,
        selection_score = score,
        rmse_sd_weight = rmse_sd_weight,
        quantile_excess_weight = quantile_excess_weight,
        quantile_sd_weight = quantile_sd_weight,
        cv_successful_folds = res$cv_successful_folds,
        cv_failed_folds = res$cv_failed_folds,
        cv_best_iter_mean = res$cv_best_iter_mean,
        cv_best_iter_sd = res$cv_best_iter_sd,
        cv_fold_rmse = res$cv_fold_rmse,
        seconds = res$seconds,
        elapsed_total_seconds = as.numeric(Sys.time() - tune_started,
                                           units = "secs"),
        stringsAsFactors = FALSE
      )
      evaluated_idx <<- c(evaluated_idx, i)
      if (identical(res$status, "ok") && is.finite(val_rmse)) {
        candidates[[label]] <<- c(res, list(spec = spec, label = label,
                                            tune_stage = tune_stage,
                                            selection_score = score,
                                            rf_inner_rmse = rf_tune_rmse,
                                            delta_rmse_vs_rf = if (is.finite(rf_tune_rmse)) {
                                              val_rmse - rf_tune_rmse
                                            } else NA_real_,
                                            val_quantile_rmse_mean =
                                              res$val_quantile_rmse_mean,
                                            val_quantile_rmse_sd =
                                              res$val_quantile_rmse_sd,
                                            val_quantile_rmse_excess =
                                              res$val_quantile_rmse_excess,
                                            val_quantile_rmse =
                                              res$val_quantile_rmse,
                                            val_quantile_bins =
                                              res$val_quantile_bins))
      }
      save_live_progress()
      cat(sprintf("[GPBoost-DSM] candidate=%s  stage=%s  status=%s  val_RMSE=%s  score=%s\n",
                  label, tune_stage, res$status,
                  ifelse(is.finite(val_rmse), sprintf("%.3f", val_rmse), "NA"),
                  ifelse(is.finite(score), sprintf("%.3f", score), "NA")))
    }
  }

  if (isTRUE(autotune_v2) && length(tune_specs) > stage1_max_candidates) {
    mode_vec <- vapply(tune_specs, `[[`, character(1), "mode")
    tree_idx <- which(mode_vec == "tree")
    spatial_idx <- which(mode_vec == "spatial")
    if (length(tree_idx) > 0L && length(spatial_idx) > 0L) {
      n_tree_stage1 <- min(length(tree_idx),
                           max(1L, floor(stage1_max_candidates * 0.40)))
      n_spatial_stage1 <- max(0L, stage1_max_candidates - n_tree_stage1)
    } else if (length(tree_idx) > 0L) {
      n_tree_stage1 <- min(length(tree_idx), stage1_max_candidates)
      n_spatial_stage1 <- 0L
    } else {
      n_tree_stage1 <- 0L
      n_spatial_stage1 <- min(length(spatial_idx), stage1_max_candidates)
    }
    stage1_idx <- unique(c(
      .gpb_select_even_indices(tune_specs, tree_idx, n_tree_stage1),
      .gpb_select_even_indices(tune_specs, spatial_idx, n_spatial_stage1)
    ))
    stage1_idx <- stage1_idx[is.finite(stage1_idx)]
    cat(sprintf("[GPBoost-DSM] AutoTune v2 stage1: evaluating %d/%d diverse candidates\n",
                length(stage1_idx), length(tune_specs)))
    run_candidate_indices(stage1_idx, "stage1_screen")

    if (!isTRUE(timed_out) && length(candidate_rows) > 0L) {
      stage1_table <- do.call(rbind, candidate_rows)
      ok <- is.finite(stage1_table$selection_score) &
        stage1_table$status %in% c("ok", "partial")
      top_rows <- stage1_table[ok, , drop = FALSE]
      top_rows <- top_rows[order(top_rows$selection_score,
                                 top_rows$val_rmse), , drop = FALSE]
      top_rows <- top_rows[seq_len(min(stage2_top_k, nrow(top_rows))), , drop = FALSE]
      top_idx <- as.integer(sub("_.*$", "", top_rows$candidate_id))
      top_mode <- top_rows$mode
      mode_levels <- unique(top_mode[top_mode %in% c("tree", "spatial")])
      stage2_idx <- integer()
      if (length(mode_levels) > 0L) {
        mode_counts <- table(factor(top_mode, levels = mode_levels))
        raw_budget <- as.numeric(mode_counts) / sum(mode_counts) *
          stage2_max_candidates
        mode_budget <- pmax(1L, floor(raw_budget))
        while (sum(mode_budget) > stage2_max_candidates &&
               any(mode_budget > 1L)) {
          j <- which.max(mode_budget)
          mode_budget[j] <- mode_budget[j] - 1L
        }
        while (sum(mode_budget) < stage2_max_candidates) {
          frac <- raw_budget - floor(raw_budget)
          j <- order(frac, raw_budget, decreasing = TRUE)[1L]
          mode_budget[j] <- mode_budget[j] + 1L
        }
        for (m_i in seq_along(mode_levels)) {
          m <- mode_levels[[m_i]]
          center_m <- top_idx[top_mode == m]
          pool_m <- which(mode_vec == m)
          stage2_idx <- c(
            stage2_idx,
            .gpb_nearest_spec_indices(
              tune_specs,
              center_idx = center_m,
              pool_idx = pool_m,
              n_keep = mode_budget[[m_i]],
              exclude_idx = evaluated_idx
            )
          )
        }
      }
      stage2_idx <- unique(c(setdiff(top_idx, evaluated_idx), stage2_idx))
      stage2_idx <- stage2_idx[is.finite(stage2_idx)]
      if (length(stage2_idx) > 0L) {
        cat(sprintf("[GPBoost-DSM] AutoTune v2 stage2: refining %d candidates around top %d\n",
                    length(stage2_idx), nrow(top_rows)))
        run_candidate_indices(stage2_idx, "stage2_refine")
      }
    }
  } else {
    run_candidate_indices(seq_along(tune_specs), "single")
  }

  if (length(candidates) == 0L) stop("GPBoostDSM: no successful candidate fit.")
  candidate_table <- do.call(rbind, candidate_rows)

  make_synthetic_row <- function(label, mode, score_info, stage,
                                 selected_base = NULL,
                                 alpha = NA_real_, gamma = NA_real_,
                                 tree_label = NA_character_,
                                 spatial_label = NA_character_,
                                 residual_status = NA_character_,
                                 residual_spatial_recommended = NA) {
    base_row <- if (!is.null(selected_base)) {
      candidate_table[candidate_table$candidate_id == selected_base, , drop = FALSE][1, , drop = FALSE]
    } else {
      candidate_table[1, , drop = FALSE]
    }
    base_row$candidate_id <- label
    base_row$tune_stage <- stage
    base_row$mode <- mode
    base_row$status <- "ok"
    base_row$error <- NA_character_
	    base_row$val_rmse <- score_info$val_rmse
	    base_row$rf_inner_rmse <- rf_tune_rmse
	    base_row$delta_rmse_vs_rf <- score_info$delta_rmse_vs_rf
	    base_row$delta_RMSE_vs_RF <- score_info$delta_rmse_vs_rf
	    base_row$selection_objective <- selection_objective
    base_row$val_quantile_rmse_mean <- score_info$qmet$mean
    base_row$val_quantile_rmse_sd <- score_info$qmet$sd
    base_row$val_quantile_rmse_excess <- score_info$qmet$excess
    base_row$val_quantile_rmse <- score_info$qmet$text
    base_row$val_quantile_bins <- score_info$qmet$n_bins
    base_row$cv_rmse_mean <- score_info$val_rmse
    base_row$cv_rmse_sd <- NA_real_
    base_row$selection_score <- score_info$score
    base_row$selected <- FALSE
    base_row$ensemble_member <- FALSE
    base_row$policy_blend_alpha <- alpha
    base_row$residual_gamma <- gamma
    base_row$synthetic_tree_candidate <- tree_label
    base_row$synthetic_spatial_candidate <- spatial_label
    base_row$residual_status <- residual_status
    base_row$residual_spatial_recommended <- residual_spatial_recommended
    base_row
  }

	  candidate_table$policy_blend_alpha <- NA_real_
	  candidate_table$residual_gamma <- NA_real_
	  candidate_table$synthetic_tree_candidate <- NA_character_
	  candidate_table$synthetic_spatial_candidate <- NA_character_
	  candidate_table$residual_status <- NA_character_
	  candidate_table$residual_spatial_recommended <- NA
	  candidate_table$selected <- FALSE
	  candidate_table$ensemble_member <- FALSE

  add_synthetic_candidate <- function(label, mode, pred_val_original,
                                      spec, stage, base_name = NULL,
                                      alpha = NA_real_, gamma = NA_real_,
                                      tree_label = NA_character_,
                                      spatial_label = NA_character_,
                                      residual_pred_val = NULL,
                                      residual_status = NA_character_,
                                      residual_spatial_recommended = NA) {
    score_info <- score_prediction_vector(tune_obs, pred_val_original)
    candidates[[label]] <<- list(
      status = "ok",
      error = NA_character_,
      seconds = 0,
      init_cov_pars = rep(NA_real_, 3L),
      best_iter = if (!is.null(base_name) && base_name %in% names(candidates)) {
        candidates[[base_name]]$best_iter
      } else NA_integer_,
      pred_val_t = if (identical(target_transform, "log")) {
        log(pmax(pred_val_original, .Machine$double.eps))
      } else pred_val_original,
      pred_val_var = rep(NA_real_, length(pred_val_original)),
      pred_test_t = NULL,
      pred_test_var = NULL,
      val_rmse = score_info$val_rmse,
      val_quantile_rmse_mean = score_info$qmet$mean,
      val_quantile_rmse_sd = score_info$qmet$sd,
      val_quantile_rmse_excess = score_info$qmet$excess,
      val_quantile_rmse = score_info$qmet$text,
      val_quantile_bins = score_info$qmet$n_bins,
      cv_rmse_mean = score_info$val_rmse,
      cv_rmse_sd = NA_real_,
      cv_successful_folds = tune_inner_folds,
      cv_failed_folds = 0L,
      cv_best_iter_mean = NA_real_,
      cv_best_iter_sd = NA_real_,
      cv_fold_rmse = NA_character_,
      spec = spec,
      label = label,
      tune_stage = stage,
	      selection_score = score_info$score,
	      rf_inner_rmse = rf_tune_rmse,
	      delta_rmse_vs_rf = score_info$delta_rmse_vs_rf,
	      delta_RMSE_vs_RF = score_info$delta_rmse_vs_rf,
	      residual_pred_val = residual_pred_val,
      residual_status = residual_status,
      residual_spatial_recommended = residual_spatial_recommended
    )
    candidate_table <<- rbind(
      candidate_table,
      make_synthetic_row(
        label = label, mode = mode, score_info = score_info, stage = stage,
        selected_base = base_name, alpha = alpha, gamma = gamma,
        tree_label = tree_label, spatial_label = spatial_label,
        residual_status = residual_status,
        residual_spatial_recommended = residual_spatial_recommended
      )
    )
  }

  ok_names <- names(candidates)
  ok_modes <- vapply(candidates, function(x) x$spec$mode, character(1))
  tree_names <- ok_names[ok_modes == "tree"]
  spatial_names <- ok_names[ok_modes == "spatial"]
  best_by_score <- function(nms) {
    if (length(nms) == 0L) return(NA_character_)
    sc <- vapply(candidates[nms], `[[`, numeric(1), "selection_score")
    nms[[which.min(sc)]]
  }
  best_tree_name <- best_by_score(tree_names)
  best_spatial_name <- best_by_score(spatial_names)
  spatial_signal_global <- isTRUE(cfg$spatial_recommended)
  if (isTRUE(policy_blend) && !is.na(best_tree_name) && !is.na(best_spatial_name)) {
    tree_pred <- gpboost_invert_target_transform(
      candidates[[best_tree_name]]$pred_val_t, target_transform)
    spatial_pred <- gpboost_invert_target_transform(
      candidates[[best_spatial_name]]$pred_val_t, target_transform)
    alpha_grid <- blend_alpha_grid
    if (isTRUE(spatial_filter) && !spatial_signal_global) {
      alpha_grid <- alpha_grid[alpha_grid <= spatial_filter_max_alpha_no_signal]
      if (!0 %in% alpha_grid) alpha_grid <- c(0, alpha_grid)
    }
    alpha_grid <- sort(unique(alpha_grid))
    for (a in alpha_grid) {
      lab <- sprintf("BLEND_a%.2f__%s__%s", a, best_tree_name, best_spatial_name)
      spec_blend <- candidates[[best_tree_name]]$spec
      spec_blend$mode <- "blend"
      spec_blend$tree_label <- best_tree_name
      spec_blend$spatial_label <- best_spatial_name
      spec_blend$alpha <- a
      pred_blend <- (1 - a) * tree_pred + a * spatial_pred
      add_synthetic_candidate(
        label = lab,
        mode = "blend",
        pred_val_original = pred_blend,
        spec = spec_blend,
        stage = "policy_blend",
        base_name = best_tree_name,
        alpha = a,
        tree_label = best_tree_name,
        spatial_label = best_spatial_name
      )
    }
  }

  if (isTRUE(residual_spatial_correction) && !is.na(best_tree_name)) {
    tree_spec <- candidates[[best_tree_name]]$spec
    tree_pred <- gpboost_invert_target_transform(
      candidates[[best_tree_name]]$pred_val_t, target_transform)
    resid_pred <- rep(NA_real_, length(tune_obs))
    residual_status <- "ok"
    if (use_cv_tuning) {
      for (j in seq_along(cv_fold_data)) {
        fdj <- cv_fold_data[[j]]
        res_tree_j <- fit_candidate(
          spec = tree_spec,
          X_train = fdj$X_train, y_train_t = fdj$y_train_t,
          C_train_s = fdj$C_train_s,
          X_val = NULL, y_val_t = NULL, C_val_s = NULL,
          X_pred = rbind(fdj$X_train, fdj$X_val),
          C_pred_s = rbind(fdj$C_train_s, fdj$C_val_s),
          cfg_fit = fdj$cfg,
          coord_scale = fdj$coord_scale,
          nrounds_fit = max(1L, candidates[[best_tree_name]]$best_iter),
          with_validation = FALSE,
          seed_offset = as.integer(71000L + j)
        )
        if (!identical(res_tree_j$status, "ok") ||
            is.null(res_tree_j$pred_test_t)) {
          residual_status <- paste("tree_refit_failed_fold", j)
          next
        }
        ntr_j <- length(fdj$idx_train)
        pred_train_j <- gpboost_invert_target_transform(
          res_tree_j$pred_test_t[seq_len(ntr_j)], target_transform)
        resid_train_j <- ycal[fdj$idx_train] - pred_train_j
        res_gp_j <- fit_residual_gp(
          res_train = resid_train_j,
          C_train_raw = Ccal[fdj$idx_train, , drop = FALSE],
          C_pred_raw = Ccal[fdj$idx_val, , drop = FALSE],
          cfg_fit = fdj$cfg,
          seed_offset = as.integer(72000L + j),
          range_multiplier = tree_spec$range_multiplier,
          nugget_multiplier = tree_spec$nugget_multiplier
        )
        if (identical(res_gp_j$status, "ok")) {
          resid_pred[fdj$idx_val] <- res_gp_j$pred
        } else {
          residual_status <- paste("residual_gp_failed_fold", j, res_gp_j$error)
        }
      }
    } else {
      res_tree <- fit_candidate(
        spec = tree_spec,
        X_train = Xtr, y_train_t = ytr_t, C_train_s = Ctr_s,
        X_val = NULL, y_val_t = NULL, C_val_s = NULL,
        X_pred = rbind(Xtr, Xva), C_pred_s = rbind(Ctr_s, Cva_s),
        cfg_fit = cfg,
        coord_scale = coord_scaler$scale,
        nrounds_fit = max(1L, candidates[[best_tree_name]]$best_iter),
        with_validation = FALSE,
        seed_offset = 71000L
      )
      if (identical(res_tree$status, "ok") && !is.null(res_tree$pred_test_t)) {
        pred_train <- gpboost_invert_target_transform(
          res_tree$pred_test_t[seq_len(nrow(Xtr))], target_transform)
        resid_train <- ytr - pred_train
        res_gp <- fit_residual_gp(
          res_train = resid_train,
          C_train_raw = Ctr,
          C_pred_raw = Cva,
          cfg_fit = cfg,
          seed_offset = 72000L,
          range_multiplier = tree_spec$range_multiplier,
          nugget_multiplier = tree_spec$nugget_multiplier
        )
        if (identical(res_gp$status, "ok")) {
          resid_pred <- res_gp$pred
        } else {
          residual_status <- paste("residual_gp_failed", res_gp$error)
        }
      } else {
        residual_status <- "tree_refit_failed"
      }
    }
    resid_signal <- tune_obs - tree_pred
    resid_vg <- tryCatch(gpboost_fit_variogram(resid_signal, if (use_cv_tuning) Ccal else Cva),
                         error = function(e) NULL)
    residual_spatial_recommended <- !is.null(resid_vg) &&
      isTRUE(resid_vg$spatial_recommended)
    gamma_grid <- residual_gamma_grid
    if (isTRUE(spatial_filter) && !residual_spatial_recommended) {
      gamma_grid <- gamma_grid[gamma_grid <= spatial_filter_max_gamma_no_signal]
      if (!0 %in% gamma_grid) gamma_grid <- c(0, gamma_grid)
    }
    gamma_grid <- sort(unique(gamma_grid))
    if (any(is.finite(resid_pred))) {
      resid_pred[!is.finite(resid_pred)] <- 0
      for (g in gamma_grid) {
        lab <- sprintf("RESIDSP_g%.2f__%s", g, best_tree_name)
        spec_resid <- tree_spec
        spec_resid$mode <- "residual_spatial"
        spec_resid$tree_label <- best_tree_name
        spec_resid$gamma <- g
        pred_resid <- tree_pred + g * resid_pred
        add_synthetic_candidate(
          label = lab,
          mode = "residual_spatial",
          pred_val_original = pred_resid,
          spec = spec_resid,
          stage = "residual_spatial",
          base_name = best_tree_name,
          gamma = g,
          tree_label = best_tree_name,
          residual_pred_val = resid_pred,
          residual_status = residual_status,
          residual_spatial_recommended = residual_spatial_recommended
        )
      }
    }
  }

  selection_candidates <- candidates
	  if (isTRUE(autotune_v2) && isTRUE(stage2_select_only) &&
	      any(candidate_table$tune_stage == "stage2_refine" &
	          candidate_table$status == "ok" &
	          is.finite(candidate_table$selection_score))) {
	    stage2_names <- candidate_table$candidate_id[
	      candidate_table$tune_stage == "stage2_refine" &
	        candidate_table$status == "ok" &
	        is.finite(candidate_table$selection_score)]
	    synthetic_names <- candidate_table$candidate_id[
	      candidate_table$tune_stage %in% c("policy_blend", "residual_spatial") &
	        candidate_table$status == "ok" &
	        is.finite(candidate_table$selection_score)]
		    selection_candidates <- candidates[intersect(
		      names(candidates), unique(c(stage2_names, synthetic_names)))]
		  }
		  if ((isTRUE(noharm_tree_anchor) || isTRUE(rf_noharm_policy)) &&
		      !is.na(best_tree_name) &&
		      !(best_tree_name %in% names(selection_candidates)) &&
		      best_tree_name %in% names(candidates)) {
		    selection_candidates[[best_tree_name]] <- candidates[[best_tree_name]]
		  }
		  score_by_candidate <- vapply(selection_candidates, `[[`,
		                               numeric(1), "selection_score")
		  selected_name_pre_noharm <- names(which.min(score_by_candidate))
		  selected_name <- selected_name_pre_noharm
		  selectable_modes <- vapply(selection_candidates, function(x) x$spec$mode,
		                             character(1))
		  tree_selectable <- names(selection_candidates)[selectable_modes == "tree"]
		  noharm_anchor_candidate <- NA_character_
	  noharm_anchor_score <- NA_real_
	  noharm_anchor_val_rmse <- NA_real_
	  noharm_margin <- NA_real_
	  noharm_delta_vs_anchor <- NA_real_
		  noharm_applied <- FALSE
		  noharm_reason <- NA_character_
		  if (isTRUE(noharm_tree_anchor)) {
		    selected_mode_pre <- selection_candidates[[selected_name_pre_noharm]]$spec$mode
	    if (length(tree_selectable) > 0L &&
	        selected_mode_pre %in% c("spatial", "blend", "residual_spatial")) {
	      tree_scores <- score_by_candidate[tree_selectable]
	      noharm_anchor_candidate <- names(which.min(tree_scores))
	      noharm_anchor_score <- as.numeric(tree_scores[[noharm_anchor_candidate]])
	      noharm_anchor_val_rmse <- suppressWarnings(
	        as.numeric(selection_candidates[[noharm_anchor_candidate]]$val_rmse)
	      )
	      if (!is.finite(noharm_anchor_val_rmse)) {
	        noharm_anchor_val_rmse <- abs(noharm_anchor_score)
	      }
	      if (!is.finite(noharm_anchor_val_rmse)) noharm_anchor_val_rmse <- 0
	      noharm_margin <- noharm_margin_abs +
	        noharm_margin_rel * abs(noharm_anchor_val_rmse)
	      noharm_delta_vs_anchor <- noharm_anchor_score -
	        as.numeric(score_by_candidate[[selected_name_pre_noharm]])
	      if (!is.finite(noharm_delta_vs_anchor) ||
	          noharm_delta_vs_anchor < noharm_margin) {
	        selected_name <- noharm_anchor_candidate
	        noharm_applied <- TRUE
	        noharm_reason <- sprintf(
	          "non_tree_gain_%.6f_below_margin_%.6f",
	          ifelse(is.finite(noharm_delta_vs_anchor),
	                 noharm_delta_vs_anchor, NA_real_),
	          noharm_margin
	        )
	      } else {
	        noharm_reason <- sprintf(
	          "non_tree_gain_%.6f_met_margin_%.6f",
	          noharm_delta_vs_anchor, noharm_margin
	        )
	      }
	    } else if (length(tree_selectable) == 0L) {
	      noharm_reason <- "no_tree_anchor_available"
	    } else {
	      noharm_reason <- "preselected_candidate_is_tree"
	    }
		  } else {
		    noharm_reason <- "disabled"
		  }
		  selected_name_pre_rf_noharm <- selected_name
		  rf_noharm_preselected_delta <- suppressWarnings(as.numeric(
		    selection_candidates[[selected_name_pre_rf_noharm]]$delta_rmse_vs_rf))
		  rf_noharm_anchor_candidate <- if (length(tree_selectable) > 0L) {
		    tree_scores <- score_by_candidate[tree_selectable]
		    names(which.min(tree_scores))
		  } else {
		    NA_character_
		  }
		  rf_noharm_anchor_delta <- if (!is.na(rf_noharm_anchor_candidate)) {
		    suppressWarnings(as.numeric(
		      selection_candidates[[rf_noharm_anchor_candidate]]$delta_rmse_vs_rf))
		  } else {
		    NA_real_
		  }
		  rf_noharm_margin <- NA_real_
		  rf_noharm_required_delta <- NA_real_
		  rf_noharm_failed <- FALSE
		  rf_noharm_applied <- FALSE
		  rf_noharm_reason <- if (isTRUE(rf_noharm_policy)) {
		    "not_assessed"
		  } else {
		    "disabled"
		  }
		  if (isTRUE(rf_noharm_policy)) {
		    rf_scale <- suppressWarnings(as.numeric(
		      selection_candidates[[selected_name_pre_rf_noharm]]$rf_inner_rmse))
		    if (!is.finite(rf_scale)) {
		      rf_scale <- suppressWarnings(as.numeric(
		        selection_candidates[[selected_name_pre_rf_noharm]]$val_rmse))
		    }
		    if (!is.finite(rf_scale)) rf_scale <- 0
		    rf_noharm_margin <- rf_noharm_margin_abs +
		      rf_noharm_margin_rel * abs(rf_scale)
		    rf_noharm_required_delta <- -rf_noharm_margin
		    if (is.finite(rf_noharm_preselected_delta)) {
		      if (rf_noharm_preselected_delta <= rf_noharm_required_delta) {
		        rf_noharm_reason <- sprintf(
		          "passed_rf_inner:delta=%.6f<=required=%.6f",
		          rf_noharm_preselected_delta, rf_noharm_required_delta)
		      } else {
		        rf_noharm_failed <- TRUE
		        if (!is.na(rf_noharm_anchor_candidate)) {
		          if (!identical(selected_name, rf_noharm_anchor_candidate)) {
		            selected_name <- rf_noharm_anchor_candidate
		            rf_noharm_applied <- TRUE
		            rf_noharm_reason <- sprintf(
		              "fallback_to_tree_anchor:delta=%.6f>required=%.6f",
		              rf_noharm_preselected_delta, rf_noharm_required_delta)
		          } else {
		            rf_noharm_reason <- sprintf(
		              "selected_tree_still_fails_rf:delta=%.6f>required=%.6f",
		              rf_noharm_preselected_delta, rf_noharm_required_delta)
		          }
		        } else {
		          rf_noharm_reason <- sprintf(
		            "rf_inner_failed_no_tree_anchor:delta=%.6f>required=%.6f",
		            rf_noharm_preselected_delta, rf_noharm_required_delta)
		        }
		      }
		    } else {
		      rf_noharm_reason <- "no_valid_delta_RMSE_vs_RF"
		    }
		  }
		  # ---- HARD do-no-harm: literal Wadoux RF fallback -----------------
		  # The existing rf_noharm_policy reverts to the best TREE-mode GPBoost
		  # candidate (rf_noharm_anchor_candidate). A tree anchor that itself
		  # loses to RF still emits a non-RF prediction, so >= RF is NOT
		  # guaranteed. We add an unconditional final guard: if the FINALLY
		  # selected candidate did not beat the (now bit-identical) Wadoux RF
		  # on leakage-free OOF RMSE by the margin, emit the LITERAL Wadoux RF
		  # prediction (fit_rf_default_wadoux / predict_rf_default_wadoux),
		  # refit on the full calibration set. The OOF lambda blend below can
		  # only shrink further toward this same RF, so the guarantee holds.
		  rf_hard_noharm_applied <- FALSE
		  rf_hard_noharm_reason <- "not_assessed"
		  selected_final_delta_rmse_vs_rf <- suppressWarnings(as.numeric(
		    selection_candidates[[selected_name]]$delta_rmse_vs_rf))
		  rf_hard_required_delta <- if (is.finite(rf_noharm_margin)) {
		    -rf_noharm_margin
		  } else {
		    -(rf_noharm_margin_abs)
		  }
		  emit_literal_rf <- isTRUE(rf_noharm_policy) &&
		    (!is.finite(selected_final_delta_rmse_vs_rf) ||
		     selected_final_delta_rmse_vs_rf > rf_hard_required_delta)
		  selected <- candidates[[selected_name]]
	  selected_spec <- selected$spec
	  selected_fit_mode <- selected_spec$mode
	  fit_mode <- selected_fit_mode
	  best_iter <- selected$best_iter
	  candidate_table$selected <- candidate_table$candidate_id == selected_name
	  candidate_table$noharm_tree_anchor <- noharm_tree_anchor
	  candidate_table$noharm_preselected_candidate <- selected_name_pre_noharm
	  candidate_table$noharm_anchor_candidate <- noharm_anchor_candidate
	  candidate_table$noharm_anchor_score <- noharm_anchor_score
	  candidate_table$noharm_anchor_val_rmse <- noharm_anchor_val_rmse
		  candidate_table$noharm_margin <- noharm_margin
		  candidate_table$noharm_delta_vs_anchor <- noharm_delta_vs_anchor
		  candidate_table$noharm_applied <- noharm_applied
		  candidate_table$noharm_reason <- noharm_reason
		  candidate_table$rf_noharm_policy <- rf_noharm_policy
		  candidate_table$rf_noharm_preselected_candidate <- selected_name_pre_rf_noharm
		  candidate_table$rf_noharm_preselected_delta_rmse_vs_rf <-
		    rf_noharm_preselected_delta
		  candidate_table$rf_noharm_anchor_candidate <- rf_noharm_anchor_candidate
		  candidate_table$rf_noharm_anchor_delta_rmse_vs_rf <- rf_noharm_anchor_delta
		  candidate_table$rf_noharm_margin_abs <- rf_noharm_margin_abs
		  candidate_table$rf_noharm_margin_rel <- rf_noharm_margin_rel
		  candidate_table$rf_noharm_margin <- rf_noharm_margin
		  candidate_table$rf_noharm_required_delta_rmse_vs_rf <-
		    rf_noharm_required_delta
		  candidate_table$rf_noharm_failed <- rf_noharm_failed
		  candidate_table$rf_noharm_applied <- rf_noharm_applied
		  candidate_table$rf_noharm_tree_only_ensemble <-
		    rf_noharm_tree_only_ensemble
		  candidate_table$rf_noharm_reason <- rf_noharm_reason
		  ordered_names <- names(sort(score_by_candidate, decreasing = FALSE))
		  if ((isTRUE(noharm_applied) && isTRUE(noharm_tree_only_ensemble)) ||
		      (isTRUE(rf_noharm_applied) && isTRUE(rf_noharm_tree_only_ensemble))) {
		    tree_ordered <- ordered_names[vapply(selection_candidates[ordered_names],
		                                         function(x) x$spec$mode == "tree",
	                                         logical(1))]
	    ordered_names <- if (length(tree_ordered) > 0L) {
	      unique(c(selected_name, tree_ordered))
	    } else {
	      selected_name
	    }
	  } else if (!identical(ordered_names[[1L]], selected_name)) {
	    ordered_names <- unique(c(selected_name, ordered_names))
	  }
  ensemble_names <- ordered_names[seq_len(min(ensemble_top_k,
                                             length(ordered_names)))]
  ensemble_scores <- score_by_candidate[ensemble_names]
  ensemble_weights <- rep(1 / length(ensemble_names), length(ensemble_names))
  if (length(ensemble_names) > 1L) {
    if (identical(ensemble_weighting, "softmax")) {
      shifted <- ensemble_scores - min(ensemble_scores, na.rm = TRUE)
      ensemble_weights <- exp(-shifted / ensemble_temperature)
    } else if (identical(ensemble_weighting, "inverse")) {
      shifted <- ensemble_scores - min(ensemble_scores, na.rm = TRUE)
      ensemble_weights <- 1 / pmax(shifted + ensemble_temperature, 1e-8)
    }
    ensemble_weights <- ensemble_weights / sum(ensemble_weights)
  }
	  candidate_table$ensemble_member <- candidate_table$candidate_id %in%
	    ensemble_names
	  candidate_table$ensemble_rank <- NA_integer_
	  candidate_table$ensemble_weight <- NA_real_
	  for (e_i in seq_along(ensemble_names)) {
	    ii <- candidate_table$candidate_id == ensemble_names[[e_i]]
	    candidate_table$ensemble_rank[ii] <- e_i
	    candidate_table$ensemble_weight[ii] <- ensemble_weights[[e_i]]
	  }
  if (use_cv_tuning && selected_spec$mode %in% c("tree", "spatial")) {
    selected_prefit <- fit_candidate(
      spec = selected_spec,
      X_train = Xtr, y_train_t = ytr_t, C_train_s = Ctr_s,
      X_val = Xva, y_val_t = yva_t, C_val_s = Cva_s,
      X_pred = Xte, C_pred_s = Cte_s,
      cfg_fit = cfg,
      coord_scale = coord_scaler$scale,
      nrounds_fit = max(1L, best_iter),
      with_validation = FALSE,
      seed_offset = 90000L)
    if (identical(selected_prefit$status, "ok")) {
      selected$pred_test_t <- selected_prefit$pred_test_t
      selected$pred_test_var <- selected_prefit$pred_test_var
    }
  }
  cat(sprintf("[GPBoost-DSM] selected candidate -> %s (best_iter=%d)\n",
              selected_name, best_iter))

  val_preds_t  <- selected$pred_val_t
  test_preds_t <- selected$pred_test_t
  val_var_t    <- selected$pred_val_var
  test_var_t   <- selected$pred_test_var
  val_obs <- if (use_cv_tuning) ycal else yva

  refit_status <- "not_requested"
  refit_error <- NA_character_
  refit_seconds <- NA_real_
  refit_best_iter <- NA_integer_
  ensemble_applied <- FALSE
  ensemble_refit_status <- NA_character_
  ensemble_test_preds <- NULL
  ensemble_val_preds <- NULL
  ensemble_test_var <- NULL
  if (refit_full) {
    Xfull <- if (has_full_fd) fd$X$full else rbind(Xtr, Xva)
    Cfull <- if (has_full_fd) fd$coords$full else rbind(Ctr, Cva)
    yfull <- if (has_full_fd) fd$y$full else c(ytr, yva)
    yfull_t <- gpboost_apply_target_transform(yfull, target_transform)
    scaler_full <- .gpb_coord_scaler(Cfull)
    Cfull_s <- .gpb_scale_coords(Cfull, scaler_full)
    Cte_full_s <- .gpb_scale_coords(Cte, scaler_full)
    vg_signal_full <- .gpb_tabular_residual_signal(Xfull, yfull_t)
    vg_full <- gpboost_fit_variogram(vg_signal_full$signal, Cfull)
    cfg_full_base <- gpboost_auto_config(vg_full, nrow(Xfull), ncol(Xfull))
    cfg_full_base$variogram_signal_method <- vg_signal_full$method
    cfg_full_base$tabular_trend_r2 <- vg_signal_full$tabular_trend_r2
    cfg_full_base$early_stop <- cfg$early_stop
    cfg_full_base$duan_apply <- cfg$duan_apply
    cfg_full_base$duan_trim_frac <- cfg$duan_trim_frac
    cfg_full_base$duan_min_val_n <- cfg$duan_min_val_n
    cfg_for_spec <- function(spec, iter_n) {
      out <- cfg_full_base
      out$learning_rate <- spec$learning_rate
      out$max_depth <- spec$max_depth
      out$num_leaves <- spec$num_leaves
      out$matern_nu <- spec$matern_nu
      out$feature_fraction <- spec$feature_fraction
      out$bagging_fraction <- spec$bagging_fraction
      out$min_data_in_leaf <- spec$min_data_in_leaf
      out$lambda_l2 <- spec$lambda_l2
      out$nrounds_max <- max(1L, iter_n)
      out
    }
    cfg_full <- cfg_for_spec(selected_spec, best_iter)
    to_target_scale <- function(pred_original) {
      if (identical(target_transform, "log")) {
        log(pmax(pred_original, .Machine$double.eps))
      } else pred_original
    }
    refit_candidate_prediction <- function(cand, cand_name, seed_offset) {
      spec <- cand$spec
      mode <- spec$mode
      if (mode %in% c("tree", "spatial")) {
        iter_i <- max(1L, cand$best_iter)
        cfg_i <- cfg_for_spec(spec, iter_i)
        refit_i <- fit_candidate(
          spec = spec,
          X_train = Xfull, y_train_t = yfull_t, C_train_s = Cfull_s,
          X_val = NULL, y_val_t = NULL, C_val_s = NULL,
          X_pred = Xte, C_pred_s = Cte_full_s,
          cfg_fit = cfg_i,
          coord_scale = scaler_full$scale,
          nrounds_fit = iter_i,
          with_validation = FALSE,
          seed_offset = as.integer(seed_offset)
        )
        if (!identical(refit_i$status, "ok") || is.null(refit_i$pred_test_t)) {
          return(list(status = refit_i$status, error = refit_i$error,
                      pred_original = NULL, pred_t = NULL,
                      var = NULL, seconds = refit_i$seconds,
                      best_iter = refit_i$best_iter))
        }
        pred_original <- gpboost_invert_target_transform(
          refit_i$pred_test_t, target_transform)
        return(list(status = "ok", error = NA_character_,
                    pred_original = pred_original,
                    pred_t = to_target_scale(pred_original),
                    var = refit_i$pred_test_var,
                    seconds = refit_i$seconds,
                    best_iter = refit_i$best_iter))
      }
      if (identical(mode, "blend")) {
        tree_label <- spec$tree_label
        spatial_label <- spec$spatial_label
        alpha <- as.numeric(spec$alpha)
        tree_refit <- refit_candidate_prediction(
          candidates[[tree_label]], tree_label, seed_offset + 101L)
        spatial_refit <- refit_candidate_prediction(
          candidates[[spatial_label]], spatial_label, seed_offset + 202L)
        if (!identical(tree_refit$status, "ok") ||
            !identical(spatial_refit$status, "ok")) {
          return(list(status = "failed",
                      error = paste("blend member failed",
                                    tree_refit$error, spatial_refit$error),
                      pred_original = NULL, pred_t = NULL,
                      var = NULL, seconds = sum(c(tree_refit$seconds,
                                                  spatial_refit$seconds),
                                                na.rm = TRUE),
                      best_iter = NA_integer_))
        }
        pred_original <- (1 - alpha) * tree_refit$pred_original +
          alpha * spatial_refit$pred_original
        var <- if (!is.null(tree_refit$var) && !is.null(spatial_refit$var) &&
                   all(is.finite(tree_refit$var)) &&
                   all(is.finite(spatial_refit$var))) {
          (1 - alpha) ^ 2 * tree_refit$var + alpha ^ 2 * spatial_refit$var
        } else rep(NA_real_, length(pred_original))
        return(list(status = "ok", error = NA_character_,
                    pred_original = pred_original,
                    pred_t = to_target_scale(pred_original),
                    var = var,
                    seconds = sum(c(tree_refit$seconds,
                                    spatial_refit$seconds), na.rm = TRUE),
                    best_iter = max(c(tree_refit$best_iter,
                                      spatial_refit$best_iter), na.rm = TRUE)))
      }
      if (identical(mode, "residual_spatial")) {
        tree_label <- spec$tree_label
        gamma <- as.numeric(spec$gamma)
        tree_cand <- candidates[[tree_label]]
        tree_spec <- tree_cand$spec
        iter_i <- max(1L, tree_cand$best_iter)
        cfg_i <- cfg_for_spec(tree_spec, iter_i)
        tree_refit <- fit_candidate(
          spec = tree_spec,
          X_train = Xfull, y_train_t = yfull_t, C_train_s = Cfull_s,
          X_val = NULL, y_val_t = NULL, C_val_s = NULL,
          X_pred = rbind(Xfull, Xte), C_pred_s = rbind(Cfull_s, Cte_full_s),
          cfg_fit = cfg_i,
          coord_scale = scaler_full$scale,
          nrounds_fit = iter_i,
          with_validation = FALSE,
          seed_offset = as.integer(seed_offset + 303L)
        )
        if (!identical(tree_refit$status, "ok") ||
            is.null(tree_refit$pred_test_t)) {
          return(list(status = tree_refit$status, error = tree_refit$error,
                      pred_original = NULL, pred_t = NULL, var = NULL,
                      seconds = tree_refit$seconds, best_iter = iter_i))
        }
        pred_all <- gpboost_invert_target_transform(
          tree_refit$pred_test_t, target_transform)
        pred_full <- pred_all[seq_len(nrow(Xfull))]
        pred_test_tree <- pred_all[-seq_len(nrow(Xfull))]
        resid_full <- yfull - pred_full
        res_gp <- fit_residual_gp(
          res_train = resid_full,
          C_train_raw = Cfull,
          C_pred_raw = Cte,
          cfg_fit = cfg_i,
          seed_offset = as.integer(seed_offset + 404L),
          range_multiplier = tree_spec$range_multiplier,
          nugget_multiplier = tree_spec$nugget_multiplier
        )
        if (!identical(res_gp$status, "ok")) {
          return(list(status = "failed", error = res_gp$error,
                      pred_original = NULL, pred_t = NULL, var = NULL,
                      seconds = tree_refit$seconds, best_iter = iter_i))
        }
        pred_original <- pred_test_tree + gamma * res_gp$pred
        return(list(status = "ok", error = NA_character_,
                    pred_original = pred_original,
                    pred_t = to_target_scale(pred_original),
                    var = res_gp$var,
                    seconds = tree_refit$seconds,
                    best_iter = iter_i))
      }
      list(status = "failed", error = paste("unsupported candidate mode", mode),
           pred_original = NULL, pred_t = NULL, var = NULL,
           seconds = NA_real_, best_iter = NA_integer_)
    }
    cat(sprintf("[GPBoost-DSM] refitting selected candidate on full calibration (n=%d, rounds=%d)...\n",
                nrow(Xfull), max(1L, best_iter)))
    refit <- refit_candidate_prediction(selected, selected_name, 10000L)
    refit_status <- refit$status
    refit_error <- refit$error
    refit_seconds <- refit$seconds
    refit_best_iter <- refit$best_iter
    if (identical(refit$status, "ok") && !is.null(refit$pred_t)) {
      test_preds_t <- refit$pred_t
      test_var_t <- refit$var
    } else {
      cat(sprintf("[GPBoost-DSM] full refit failed; using inner-fit test predictions: %s\n",
                  refit$error))
    }
    if (length(ensemble_names) > 1L) {
      cat(sprintf("[GPBoost-DSM] refitting GPBoost ensemble top_k=%d weighting=%s...\n",
                  length(ensemble_names), ensemble_weighting))
      ens_test <- list()
      ens_var <- list()
      ens_val <- list()
      ens_weights_ok <- numeric(0)
      ens_names_ok <- character(0)
      ens_status <- character(0)
      for (e_i in seq_along(ensemble_names)) {
        nm <- ensemble_names[[e_i]]
        cand_i <- candidates[[nm]]
        seed_preds <- list()
        seed_vars <- list()
        seed_status <- character(0)
        for (s_i in seq_len(seed_ensemble_n)) {
          refit_i <- refit_candidate_prediction(
            cand_i, nm, as.integer(11000L + e_i * 1000L + s_i * 37L))
          seed_status <- c(seed_status, paste0("s", s_i, ":", refit_i$status))
          if (identical(refit_i$status, "ok") && !is.null(refit_i$pred_original)) {
            seed_preds[[length(seed_preds) + 1L]] <- refit_i$pred_original
            seed_vars[[length(seed_vars) + 1L]] <- refit_i$var
          }
        }
        ens_status <- c(ens_status, paste0(nm, ":{",
                                           paste(seed_status, collapse = ","),
                                           "}"))
        if (length(seed_preds) > 0L) {
          pred_i <- Reduce(`+`, seed_preds) / length(seed_preds)
          ens_test[[length(ens_test) + 1L]] <- pred_i
          if (length(seed_vars) == length(seed_preds) &&
              all(vapply(seed_vars, function(z) !is.null(z) && all(is.finite(z)),
                         logical(1)))) {
            ens_var[[length(ens_var) + 1L]] <- Reduce(`+`, seed_vars) / length(seed_vars)
          } else {
            ens_var[[length(ens_var) + 1L]] <- rep(NA_real_, length(pred_i))
          }
          if (!is.null(cand_i$pred_val_t)) {
            ens_val[[length(ens_val) + 1L]] <-
              gpboost_invert_target_transform(cand_i$pred_val_t,
                                              target_transform)
          }
          ens_weights_ok <- c(ens_weights_ok, ensemble_weights[[e_i]])
          ens_names_ok <- c(ens_names_ok, nm)
        }
      }
      ensemble_refit_status <- paste(ens_status, collapse = "|")
      if (length(ens_test) > 1L && sum(ens_weights_ok) > 0) {
        ens_weights_ok <- ens_weights_ok / sum(ens_weights_ok)
        ensemble_test_preds <- Reduce(`+`, Map(`*`, ens_test, ens_weights_ok))
        if (length(ens_val) == length(ens_test)) {
          ensemble_val_preds <- Reduce(`+`, Map(`*`, ens_val, ens_weights_ok))
        }
        if (length(ens_var) == length(ens_test) &&
            all(vapply(ens_var, function(z) all(is.finite(z)), logical(1)))) {
          ensemble_test_var <- Reduce(`+`, Map(`*`, ens_var, ens_weights_ok))
        }
        ensemble_weights <- ens_weights_ok
        ensemble_names <- ens_names_ok
        ensemble_applied <- TRUE
        fit_mode <- "ensemble"
        test_var_t <- if (is.null(ensemble_test_var)) {
          rep(NA_real_, length(ensemble_test_preds))
        } else ensemble_test_var
        cat(sprintf("[GPBoost-DSM] ensemble applied with %d successful members\n",
                    length(ens_test)))
      } else {
        cat("[GPBoost-DSM] ensemble skipped; fewer than two members refit successfully\n")
      }
    }
  }

  # Inverse target transform.
  val_preds  <- if (!is.null(ensemble_val_preds)) {
    ensemble_val_preds
  } else {
    gpboost_invert_target_transform(val_preds_t, target_transform)
  }
  test_preds <- if (!is.null(ensemble_test_preds)) {
    ensemble_test_preds
  } else {
    gpboost_invert_target_transform(test_preds_t, target_transform)
  }

  # ---- Literal Wadoux RF prediction (full calibration) ------------------
  # Always fit the literal Wadoux RF on the full calibration set so we can
  # (a) hard-fall-back to it and (b) OOF-select a continuous shrinkage lambda
  # that blends GPBoost toward it. Column names mirror .gpb_rf_predict_internal.
  rf_lambda_to_rf <- 0
  rf_lambda_grid_used <- NA_character_
  rf_lambda_reason <- "not_assessed"
  Xcal_rf <- as.data.frame(Xcal)
  names(Xcal_rf) <- paste0("x", seq_len(ncol(Xcal_rf)))
  Xte_rf <- as.data.frame(Xte)
  names(Xte_rf) <- names(Xcal_rf)
  rf_full_test_pred <- tryCatch(
    .gpb_rf_predict_internal(
      X_train = Xcal_rf, y_train = ycal, X_pred = Xte_rf,
      seed = as.integer(getOption("wadoux_rf_seed", 123L))),
    error = function(e) rep(NA_real_, nrow(Xte_rf)))

  # OOF-selected lambda in [0,1]: final = (1-lambda)*GPBoost + lambda*RF,
  # chosen on the leakage-free cross-fit OOF set (grid INCLUDES 0 and 1).
  if (isTRUE(rf_noharm_policy) && use_cv_tuning &&
      !is.na(selected_name) &&
      !is.null(selection_candidates[[selected_name]]$oof_t) &&
      any(is.finite(rf_tune_pred))) {
    gpb_oof <- gpboost_invert_target_transform(
      selection_candidates[[selected_name]]$oof_t, target_transform)
    okl <- is.finite(gpb_oof) & is.finite(rf_tune_pred) & is.finite(ycal)
    if (sum(okl) >= 5L) {
      lam_grid <- sort(unique(c(0, 0.10, 0.25, 0.50, 0.75, 1)))
      lam_rmse <- vapply(lam_grid, function(l) {
        .gpb_rmse(ycal[okl],
                  (1 - l) * gpb_oof[okl] + l * rf_tune_pred[okl])
      }, numeric(1))
      rf_lambda_to_rf <- lam_grid[which.min(lam_rmse)]
      rf_lambda_grid_used <- paste(formatC(lam_grid, digits = 4L,
                                           format = "fg"), collapse = "|")
      rf_lambda_reason <- sprintf("oof_selected_lambda=%.2f", rf_lambda_to_rf)
    } else {
      rf_lambda_reason <- "too_few_oof_points"
    }
  } else {
    rf_lambda_reason <- "lambda_disabled_or_no_oof"
  }

  # Apply hard fallback FIRST (forces lambda=1 toward literal RF), then the
  # OOF lambda blend. Both can only move test_preds toward the literal RF.
  if (isTRUE(emit_literal_rf) && all(is.finite(rf_full_test_pred))) {
    rf_lambda_to_rf <- 1
    rf_hard_noharm_applied <- TRUE
    rf_hard_noharm_reason <- sprintf(
      "selected_delta=%.6f>required=%.6f:emit_literal_wadoux_rf",
      selected_final_delta_rmse_vs_rf, rf_hard_required_delta)
    fit_mode <- "literal_wadoux_rf"
  } else if (isTRUE(emit_literal_rf)) {
    rf_hard_noharm_reason <- "would_fallback_but_rf_pred_non_finite"
  } else {
    rf_hard_noharm_reason <- sprintf(
      "selected_delta=%.6f<=required=%.6f:kept_gpboost",
      selected_final_delta_rmse_vs_rf, rf_hard_required_delta)
  }
  if (rf_lambda_to_rf > 0 && all(is.finite(rf_full_test_pred)) &&
      length(rf_full_test_pred) == length(test_preds)) {
    test_preds <- (1 - rf_lambda_to_rf) * test_preds +
      rf_lambda_to_rf * rf_full_test_pred
  }

  # ---- Duan smearing if log + n_val sufficient ---------------------------
  smearing_info <- list(S = 1.0, applied = FALSE, reason = "target != log")
  if (cfg$duan_apply && identical(target_transform, "log")) {
    val_log_obs <- log(val_obs)
    val_log_pred <- if (!is.null(ensemble_val_preds)) {
      log(pmax(ensemble_val_preds, .Machine$double.eps))
    } else val_preds_t
    r_log <- val_log_obs - val_log_pred
    smearing_info <- gpboost_duan_smearing(
      r_log, trim_frac = cfg$duan_trim_frac,
      min_val_n = cfg$duan_min_val_n)
    cat(sprintf("[GPBoost-DSM] Duan: %s\n", smearing_info$reason))
    if (isTRUE(smearing_info$applied)) {
      cat(sprintf("[GPBoost-DSM]   S_trim=%.5f  S_raw=%.5f  n_used=%d\n",
                  smearing_info$S_trimmed, smearing_info$S_raw,
                  smearing_info$n_used))
      val_preds  <- val_preds  * smearing_info$S
      test_preds <- test_preds * smearing_info$S
    }
  }

  # ---- Final summary ----------------------------------------------------
  val_me   <- mean(val_preds - val_obs, na.rm = TRUE)
  val_rmse <- sqrt(mean((val_preds - val_obs) ^ 2, na.rm = TRUE))
  cat("\n[GPBoost-DSM] === Final summary ===\n")
  cat(sprintf("[GPBoost-DSM] Val   : ME=%+.3f  RMSE=%.3f\n",
              val_me, val_rmse))
  cat(sprintf("[GPBoost-DSM] Test  : RMSE=%.3f  mode=%s  refit=%s\n",
              sqrt(mean((yte - test_preds) ^ 2, na.rm = TRUE)),
              fit_mode, refit_status))

  # ---- Return -----------------------------------------------------------
  list(
    pred_test       = test_preds,
    pred_val        = val_preds,
    pred_test_base  = test_preds,                      # GPBoost has no clean
    pred_val_base   = val_preds,                       # base/full split
    pred_test_var   = test_var_t,                      # native uncertainty
    pred_val_var    = val_var_t,
    metrics_test    = data.frame(
      N    = length(yte),
      R2   = if (length(yte) >= 2) cor(yte, test_preds) ^ 2 else NA_real_,
      RMSE = sqrt(mean((yte - test_preds) ^ 2, na.rm = TRUE)),
      MAE  = mean(abs(yte - test_preds), na.rm = TRUE),
      Bias = mean(test_preds - yte, na.rm = TRUE),
      RPIQ = if (length(yte) >= 2)
               IQR(yte) / (sqrt(mean((yte - test_preds) ^ 2)) + 1e-12)
             else NA_real_
    ),
    diagnostics = list(
      target_transform = target_transform,
      n_train          = n_train,
      n_val            = n_val,
      matern_nu        = cfg$matern_nu,
      range_init       = cfg$range_init,
      nugget_ratio     = cfg$nugget_ratio,
      fit_quality      = cfg$fit_quality,
      variogram_status = cfg$variogram_status,
      variogram_n      = cfg$variogram_n,
      variogram_kappa  = cfg$variogram_kappa,
      variogram_signal_method = cfg$variogram_signal_method,
      tabular_trend_r2 = cfg$tabular_trend_r2,
      coord_trend_r2   = cfg$coord_trend_r2,
      stationarity_decision = cfg$stationarity_decision,
      spatial_recommended = cfg$spatial_recommended,
      learning_rate    = cfg$learning_rate,
      max_depth        = cfg$max_depth,
      num_leaves       = cfg$num_leaves,
      feature_fraction = cfg$feature_fraction,
      bagging_fraction = cfg$bagging_fraction,
      min_data_in_leaf = cfg$min_data_in_leaf,
      lambda_l2        = cfg$lambda_l2,
      leaves_newton_update = leaves_newton_update,
      use_gp_model_for_validation = use_gp_model_for_validation,
      tree_fallback     = tree_fallback,
      fair_tune         = fair_tune,
      refit_full        = refit_full,
      autotune_v2       = autotune_v2,
      crash_safe        = crash_safe,
      requested_spatial_policy = requested_spatial_policy,
      spatial_policy    = effective_spatial_policy,
      spatial_policy_override_reason = spatial_policy_override_reason,
      include_tree_candidates = include_tree_candidates,
      include_spatial_candidates = include_spatial_candidates,
      stage1_max_candidates = stage1_max_candidates,
      stage2_top_k      = stage2_top_k,
      stage2_max_candidates = stage2_max_candidates,
      stage2_select_only = stage2_select_only,
      rmse_sd_weight    = rmse_sd_weight,
      quantile_bins     = quantile_bins,
      quantile_excess_weight = quantile_excess_weight,
      quantile_sd_weight = quantile_sd_weight,
      selection_objective = selection_objective,
      rf_inner_rmse = rf_tune_rmse,
      rf_inner_fold_rmse = rf_tune_fold_rmse,
      policy_blend = policy_blend,
      blend_alpha_grid = paste(formatC(blend_alpha_grid, digits = 4L, format = "fg"),
                               collapse = "|"),
      selected_policy_blend_alpha = if (!is.null(selected_spec$alpha)) {
        as.numeric(selected_spec$alpha)
      } else NA_real_,
      residual_spatial_correction = residual_spatial_correction,
      residual_gamma_grid = paste(formatC(residual_gamma_grid, digits = 4L, format = "fg"),
                                  collapse = "|"),
      selected_residual_gamma = if (!is.null(selected_spec$gamma)) {
        as.numeric(selected_spec$gamma)
      } else NA_real_,
      selected_synthetic_tree_candidate = if (!is.null(selected_spec$tree_label)) {
        selected_spec$tree_label
      } else NA_character_,
	      selected_synthetic_spatial_candidate = if (!is.null(selected_spec$spatial_label)) {
	        selected_spec$spatial_label
	      } else NA_character_,
	      selected_residual_status = if (!is.null(selected$residual_status)) {
	        selected$residual_status
	      } else NA_character_,
	      selected_residual_spatial_recommended =
		        if (!is.null(selected$residual_spatial_recommended)) {
		          isTRUE(selected$residual_spatial_recommended)
		        } else NA,
		      seed_ensemble_n = seed_ensemble_n,
	      noharm_tree_anchor = noharm_tree_anchor,
	      noharm_preselected_candidate = selected_name_pre_noharm,
	      noharm_anchor_candidate = noharm_anchor_candidate,
	      noharm_anchor_score = noharm_anchor_score,
	      noharm_anchor_val_rmse = noharm_anchor_val_rmse,
	      noharm_margin_abs = noharm_margin_abs,
	      noharm_margin_rel = noharm_margin_rel,
		      noharm_margin = noharm_margin,
		      noharm_delta_vs_anchor = noharm_delta_vs_anchor,
		      noharm_applied = noharm_applied,
		      noharm_tree_only_ensemble = noharm_tree_only_ensemble,
		      noharm_reason = noharm_reason,
		      rf_noharm_policy = rf_noharm_policy,
		      rf_noharm_preselected_candidate = selected_name_pre_rf_noharm,
		      rf_noharm_preselected_delta_rmse_vs_rf =
		        rf_noharm_preselected_delta,
		      rf_noharm_anchor_candidate = rf_noharm_anchor_candidate,
		      rf_noharm_anchor_delta_rmse_vs_rf = rf_noharm_anchor_delta,
		      rf_noharm_margin_abs = rf_noharm_margin_abs,
		      rf_noharm_margin_rel = rf_noharm_margin_rel,
		      rf_noharm_margin = rf_noharm_margin,
		      rf_noharm_required_delta_rmse_vs_rf = rf_noharm_required_delta,
		      rf_noharm_failed = rf_noharm_failed,
		      rf_noharm_applied = rf_noharm_applied,
		      rf_noharm_tree_only_ensemble = rf_noharm_tree_only_ensemble,
		      rf_noharm_reason = rf_noharm_reason,
		      rf_hard_noharm_applied = rf_hard_noharm_applied,
		      rf_hard_noharm_reason = rf_hard_noharm_reason,
		      rf_hard_required_delta_rmse_vs_rf = rf_hard_required_delta,
		      selected_final_delta_rmse_vs_rf = selected_final_delta_rmse_vs_rf,
		      rf_lambda_to_rf = rf_lambda_to_rf,
		      rf_lambda_grid_used = rf_lambda_grid_used,
		      rf_lambda_reason = rf_lambda_reason,
		      spatial_filter = spatial_filter,
      spatial_filter_max_alpha_no_signal = spatial_filter_max_alpha_no_signal,
      spatial_filter_max_gamma_no_signal = spatial_filter_max_gamma_no_signal,
      tune_timeout_reached = timed_out,
      tune_max_seconds  = tune_max_seconds,
      tune_progress_file = if (is.null(tune_progress_file)) NA_character_ else tune_progress_file,
      tune_candidates_evaluated = nrow(candidate_table),
      tune_candidate_universe = length(tune_specs),
      tune_candidate_universe_before_cap = tune_candidate_universe_before_cap,
      tune_candidate_universe_after_cap = tune_candidate_universe_after_cap,
      tune_tree_universe_before_cap = tune_tree_universe_before_cap,
      tune_spatial_universe_before_cap = tune_spatial_universe_before_cap,
      num_threads       = if (is.null(num_threads)) NA_integer_ else num_threads,
      tune_inner_folds  = tune_inner_folds,
      tune_source       = tune_source,
      tune_n            = if (use_cv_tuning) nrow(Xcal) else n_val,
      tune_fold_sizes   = cv_fold_sizes,
      fit_mode          = fit_mode,
      selected_fit_mode = selected_fit_mode,
      best_iter        = best_iter,
      nrounds_max      = cfg$nrounds_max,
      early_stop       = cfg$early_stop,
	      selected_candidate = selected_name,
	      selected_tune_stage = selected$tune_stage,
	      selected_selection_score = selected$selection_score,
	      selected_delta_rmse_vs_rf = selected$delta_rmse_vs_rf,
	      selected_delta_RMSE_vs_RF = selected$delta_rmse_vs_rf,
	      selected_learning_rate = selected_spec$learning_rate,
      selected_max_depth = selected_spec$max_depth,
      selected_num_leaves = selected_spec$num_leaves,
      selected_matern_nu = selected_spec$matern_nu,
      selected_feature_fraction = selected_spec$feature_fraction,
      selected_min_data_in_leaf = selected_spec$min_data_in_leaf,
      selected_bagging_fraction = selected_spec$bagging_fraction,
      selected_lambda_l2 = selected_spec$lambda_l2,
      selected_range_multiplier = selected_spec$range_multiplier,
      selected_nugget_multiplier = selected_spec$nugget_multiplier,
      selected_cv_rmse_mean = selected$cv_rmse_mean,
      selected_cv_rmse_sd = selected$cv_rmse_sd,
      selected_quantile_rmse_mean = selected$val_quantile_rmse_mean,
      selected_quantile_rmse_sd = selected$val_quantile_rmse_sd,
      selected_quantile_rmse_excess = selected$val_quantile_rmse_excess,
      selected_quantile_rmse = selected$val_quantile_rmse,
      selected_quantile_bins = selected$val_quantile_bins,
      selected_cv_successful_folds = selected$cv_successful_folds,
      selected_cv_failed_folds = selected$cv_failed_folds,
      selected_cv_best_iter_mean = selected$cv_best_iter_mean,
      ensemble_top_k_requested = ensemble_top_k,
      ensemble_applied = isTRUE(ensemble_applied),
      ensemble_size = if (isTRUE(ensemble_applied)) length(ensemble_names) else 1L,
      ensemble_weighting = ensemble_weighting,
      ensemble_temperature = ensemble_temperature,
      ensemble_members = if (isTRUE(ensemble_applied)) {
        paste(ensemble_names, collapse = "|")
      } else selected_name,
      ensemble_weights = if (isTRUE(ensemble_applied)) {
        paste(formatC(ensemble_weights, digits = 6L, format = "fg"),
              collapse = "|")
      } else "1",
      ensemble_refit_status = if (is.na(ensemble_refit_status)) {
        NA_character_
      } else ensemble_refit_status,
      init_cov_pars_order = "error_variance,gp_variance,gp_range",
      coord_scale       = coord_scaler$scale,
      gpboost_seconds  = sum(candidate_table$seconds, na.rm = TRUE) +
                         ifelse(is.finite(refit_seconds), refit_seconds, 0),
      val_rmse_spatial = if (any(candidate_table$mode == "spatial" &
                                  is.finite(candidate_table$val_rmse))) {
                           min(candidate_table$val_rmse[candidate_table$mode == "spatial"],
                               na.rm = TRUE)
                         } else NA_real_,
      val_rmse_tree    = if (any(candidate_table$mode == "tree" &
                                  is.finite(candidate_table$val_rmse))) {
                           min(candidate_table$val_rmse[candidate_table$mode == "tree"],
                               na.rm = TRUE)
                         } else NA_real_,
      refit_status     = refit_status,
      refit_error      = refit_error,
      refit_seconds    = refit_seconds,
      refit_best_iter  = refit_best_iter,
      duan_applied     = isTRUE(smearing_info$applied),
      duan_S           = smearing_info$S,
      duan_S_raw       = if (is.null(smearing_info$S_raw)) NA_real_ else smearing_info$S_raw,
      duan_n_used      = if (is.null(smearing_info$n_used)) 0L else smearing_info$n_used,
      duan_reason      = smearing_info$reason,
      val_me           = val_me,
      val_rmse         = val_rmse
    ),
    candidate_table = candidate_table
  )
}

# ---- Default preset --------------------------------------------------------

gpboost_dsm_default_params <- list(
  # IMPORTANT: target_transform defaults to "identity" for GPBoost-DSM, NOT
  # "auto". Reason: GPBoost is a Gaussian-process boosting model that fits
  # log(y) very tightly when log is auto-selected (because the GP's
  # likelihood is centred Gaussian on log scale). The tight log-scale fit
  # exposes the full Jensen / Finney median-vs-mean bias on inverse:
  # exp(yhat_log) systematically under-predicts the conditional MEAN of y
  # by exp(sigma^2/2) - 1. Empirically (n=500 Wadoux Amazon AGB) this
  # caused ME ~ -15 Mg/ha and RMSE ~50 Mg/ha for GPBoost while the same
  # auto-selector worked acceptably for GeoBART (which has wider posterior
  # uncertainty on log scale and so the bias is partially absorbed).
  # Conditional Duan smearing at n_val < 500 is unstable (NMK round 5
  # demonstrated this). Until we have a robust correction strategy
  # for the small-n_val regime, the safe default is identity. To enable
  # log explicitly, pass `target_transform = "log"` or use the
  # GPB_TARGET_TRANSFORM env var.
  target_transform = "identity",
  matern_nu        = NULL,
  learning_rate    = NULL,
  max_depth        = NULL,
  num_leaves       = NULL,
  nrounds_max      = NULL,
  early_stop       = NULL,
  feature_fraction = NULL,
  bagging_fraction = NULL,
  min_data_in_leaf = NULL,
  lambda_l2        = NULL,
  # Fair-tuned default: choose a compact inner-validation grid, initialise GP
  # covariance parameters from the empirical variogram, and refit the selected
  # candidate on all calibration rows before predicting the design-based test.
  leaves_newton_update       = TRUE,
  use_gp_model_for_validation = FALSE,
  tree_fallback              = TRUE,
  fair_tune                  = TRUE,
  refit_full                 = TRUE,
  tune_inner_folds           = 3L,
  learning_rate_grid         = NULL,
  max_depth_grid             = NULL,
  matern_nu_grid             = NULL,
  feature_fraction_grid      = NULL,
  min_data_in_leaf_grid      = NULL,
  bagging_fraction_grid      = NULL,
  lambda_l2_grid             = NULL,
  range_init_grid            = NULL,
  nugget_init_grid           = NULL,
  max_tune_candidates        = 72L,
  autotune_v2                = TRUE,
  stage1_max_candidates      = 96L,
  stage2_top_k               = 12L,
  stage2_max_candidates      = 48L,
  stage2_select_only         = FALSE,
  # Native spatial GPBoost candidates can segfault the R process before
  # tryCatch can handle the error. The production benchmark therefore defaults
  # to crash-safe tree-only tuning; set crash_safe = FALSE and
  # spatial_policy = "mixed" only for isolated spatial stress tests.
  crash_safe                 = TRUE,
  spatial_policy             = "tree_only",
  rmse_sd_weight             = 0.25,
  quantile_bins              = 5L,
  quantile_excess_weight     = 0.15,
  quantile_sd_weight         = 0.05,
  ensemble_top_k             = 1L,
  ensemble_weighting         = "softmax",
  ensemble_temperature       = 0.75,
  selection_objective        = "rmse",
  policy_blend               = FALSE,
  blend_alpha_grid           = c(0, 0.10, 0.25, 0.50, 0.75, 1),
	  residual_spatial_correction = FALSE,
	  residual_gamma_grid        = c(0, 0.25, 0.50, 0.75, 1),
	  seed_ensemble_n            = 1L,
		  noharm_tree_anchor         = TRUE,
		  noharm_margin_abs          = 0.05,
		  noharm_margin_rel          = 0.0,
		  noharm_tree_only_ensemble  = TRUE,
		  rf_noharm_policy           = TRUE,
		  rf_noharm_margin_abs       = 0.05,
		  rf_noharm_margin_rel       = 0.0,
		  rf_noharm_tree_only_ensemble = TRUE,
		  spatial_filter             = TRUE,
  spatial_filter_max_alpha_no_signal = 0.25,
  spatial_filter_max_gamma_no_signal = 0.25,
  tune_max_seconds           = Inf,
  tune_min_candidates_before_timeout = 12L,
  tune_progress_file         = NULL,
  num_threads                = NULL,
  train_seed       = 123L
)
