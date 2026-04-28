# =============================================================================
# DBTARF.R
#
# DB-TARF: Design-Based Targeted Adaptive Residual Forest.
#
# Prediction form:
#
#   y_hat(s, x) = RF_default(s, x) + lambda * correction(s, x)
#
# The RF component is the Wadoux-style default ranger model. Candidate
# corrections are learned from cross-fitted RF residuals and are accepted only
# when their out-of-fold RMSE beats the RF-only prediction on the calibration
# sample. This makes the model a targeted residual updater rather than a
# replacement for the RF baseline.
#
# Dependencies:
#   - ranger
#   - caret
#   - Cubist (optional residual expert)
#   - gstat, sp (optional variogram-informed RBF features)
# =============================================================================

.dbt_clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

.dbt_rmse <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  if (sum(ok) == 0L) return(Inf)
  sqrt(mean((obs[ok] - pred[ok]) ^ 2))
}

.dbt_predict_rf <- function(model, new_df) {
  as.numeric(predict(model, data = new_df, type = "response")$predictions)
}

.dbt_oob_or_in_sample <- function(model, train_df) {
  p <- suppressWarnings(as.numeric(model$predictions))
  if (length(p) != nrow(train_df) || any(!is.finite(p))) {
    p <- .dbt_predict_rf(model, train_df)
  }
  p
}

.dbt_fit_rf <- function(train_df, response_name, predictor_names) {
  form <- as.formula(
    paste(response_name, "~", paste(predictor_names, collapse = "+"))
  )
  ranger::ranger(formula = form, data = train_df)
}

.dbt_fit_cubist <- function(x, y) {
  if (!requireNamespace("Cubist", quietly = TRUE)) return(NULL)
  tryCatch(
    Cubist::cubist(x = as.data.frame(x), y = as.numeric(y), committees = 10L),
    error = function(e) NULL
  )
}

.dbt_predict_cubist <- function(model, x) {
  if (is.null(model)) return(rep(NA_real_, nrow(x)))
  as.numeric(predict(model, newdata = as.data.frame(x), neighbors = 0L))
}

.dbt_fit_variogram <- function(y, coords, cutoff_frac = 0.50, n_lags = 15L) {
  if (!requireNamespace("gstat", quietly = TRUE) ||
      !requireNamespace("sp", quietly = TRUE)) {
    return(NULL)
  }
  keep <- is.finite(y) & is.finite(coords[, 1]) & is.finite(coords[, 2])
  y <- y[keep]
  coords <- coords[keep, , drop = FALSE]
  if (length(y) < 20L || stats::sd(y, na.rm = TRUE) <= 1e-8) return(NULL)

  D <- as.matrix(stats::dist(coords))
  cutoff <- max(D, na.rm = TRUE) * cutoff_frac
  if (!is.finite(cutoff) || cutoff <= 0) return(NULL)

  df <- data.frame(yv = as.numeric(y), x = coords[, 1], y2 = coords[, 2])
  sp::coordinates(df) <- ~ x + y2
  vg_emp <- tryCatch(
    gstat::variogram(yv ~ 1, data = df, cutoff = cutoff,
                     width = cutoff / n_lags, cressie = TRUE),
    error = function(e) NULL
  )
  if (is.null(vg_emp) || nrow(vg_emp) < 3L) return(NULL)

  sill_tot <- stats::var(y, na.rm = TRUE)
  range_init <- cutoff * 0.30
  vg_fit <- tryCatch(
    suppressWarnings(gstat::fit.variogram(
      vg_emp,
      gstat::vgm(sill_tot * 0.80, "Exp", range_init, sill_tot * 0.20),
      fit.method = 7L
    )),
    error = function(e) NULL
  )
  if (is.null(vg_fit) || !all(is.finite(vg_fit$range))) return(NULL)

  iso_range <- max(vg_fit$range[vg_fit$range > 0], na.rm = TRUE)
  if (!is.finite(iso_range) || iso_range <= 0) iso_range <- range_init
  psill_sum <- sum(vg_fit$psill, na.rm = TRUE)
  nugget_ratio <- if (is.finite(psill_sum) && psill_sum > 1e-8) {
    .dbt_clamp(vg_fit$psill[1L] / psill_sum, 0.02, 0.90)
  } else 0.50

  list(range_major = iso_range, nugget_ratio = nugget_ratio)
}

.dbt_fit_basis <- function(coords, residuals, n_basis = NULL,
                           bandwidth = NULL, seed = NULL) {
  coords <- as.matrix(coords)
  area <- diff(range(coords[, 1], na.rm = TRUE)) *
    diff(range(coords[, 2], na.rm = TRUE))
  vg <- .dbt_fit_variogram(residuals, coords)
  if (is.null(vg)) {
    D <- as.matrix(stats::dist(coords))
    rho <- stats::median(D[upper.tri(D)], na.rm = TRUE)
    if (!is.finite(rho) || rho <= 0) rho <- sqrt(area) / 4
    nugget_ratio <- 0.75
  } else {
    rho <- vg$range_major
    nugget_ratio <- vg$nugget_ratio
  }

  if (is.null(n_basis)) {
    n_basis_theory <- 4 * area / (pi * rho ^ 2) * (1 - nugget_ratio)
    n_basis <- as.integer(.dbt_clamp(round(n_basis_theory), 8L, 48L))
  }
  n_basis <- as.integer(min(max(n_basis, 0L), nrow(coords)))
  if (is.null(bandwidth)) bandwidth <- max(rho / 2, 1e-6)

  if (n_basis <= 0L) {
    return(list(centers = matrix(numeric(0), nrow = 0L, ncol = 2L),
                bandwidth = bandwidth, range_major = rho,
                nugget_ratio = nugget_ratio, n_basis = 0L))
  }

  if (!is.null(seed)) set.seed(as.integer(seed))
  centers <- if (n_basis == 1L) {
    matrix(colMeans(coords), nrow = 1L)
  } else {
    stats::kmeans(coords, centers = n_basis, iter.max = 50L,
                  nstart = 5L)$centers
  }
  list(centers = as.matrix(centers), bandwidth = bandwidth,
       range_major = rho, nugget_ratio = nugget_ratio, n_basis = n_basis)
}

.dbt_make_basis <- function(coords, basis) {
  coords <- as.matrix(coords)
  centers <- as.matrix(basis$centers)
  if (nrow(centers) == 0L) return(matrix(numeric(0), nrow(coords), 0L))
  d2 <- outer(rowSums(coords ^ 2), rowSums(centers ^ 2), `+`) -
    2 * coords %*% t(centers)
  d2[d2 < 0] <- 0
  out <- exp(-d2 / (2 * basis$bandwidth ^ 2))
  colnames(out) <- sprintf("phi_%02d", seq_len(ncol(out)))
  out
}

.dbt_residual_features <- function(df, predictor_names, coord_names,
                                   base_pred, cubist_pred = NULL,
                                   basis = NULL) {
  x <- as.data.frame(df[, predictor_names, drop = FALSE])
  coords <- as.matrix(df[, coord_names, drop = FALSE])
  colnames(coords) <- c("coord_x", "coord_y")
  feat <- cbind(x, as.data.frame(coords), rf_pred = as.numeric(base_pred))
  if (!is.null(cubist_pred) && length(cubist_pred) == nrow(df)) {
    feat$cubist_pred <- as.numeric(cubist_pred)
    feat$cubist_minus_rf <- as.numeric(cubist_pred) - as.numeric(base_pred)
  }
  if (!is.null(basis)) {
    phi <- .dbt_make_basis(coords, basis)
    if (ncol(phi) > 0L) feat <- cbind(feat, as.data.frame(phi))
  }
  feat
}

.dbt_fit_residual_rf <- function(features, residuals) {
  df <- as.data.frame(features)
  df$.resid <- as.numeric(residuals)
  ranger::ranger(.resid ~ ., data = df, num.trees = 300,
                 min.node.size = max(5L, floor(nrow(df) / 80)))
}

.dbt_predict_residual_rf <- function(model, features) {
  as.numeric(predict(model, data = as.data.frame(features))$predictions)
}

.dbt_kernel_residual_predict <- function(coords_train, residuals_train,
                                         coords_new, bandwidth) {
  coords_train <- as.matrix(coords_train)
  coords_new <- as.matrix(coords_new)
  residuals_train <- as.numeric(residuals_train)
  d2 <- outer(rowSums(coords_new ^ 2), rowSums(coords_train ^ 2), `+`) -
    2 * coords_new %*% t(coords_train)
  d2[d2 < 0] <- 0
  w <- exp(-d2 / (2 * max(bandwidth, 1e-6) ^ 2))
  denom <- rowSums(w)
  out <- as.numeric(w %*% residuals_train) / pmax(denom, 1e-12)
  out[!is.finite(out)] <- mean(residuals_train, na.rm = TRUE)
  out
}

.dbt_lambda_grid <- function(lambda_grid = NULL) {
  if (is.null(lambda_grid)) return(seq(-0.25, 0.60, by = 0.05))
  sort(unique(as.numeric(lambda_grid)))
}

.dbt_preserve_global_rng <- function() {
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

train_dbtarf_one_fold <- function(train_df,
                                  test_df,
                                  predictor_names,
                                  response_name,
                                  coord_names = c("x", "y"),
                                  n_folds = 5L,
                                  lambda_grid = NULL,
                                  n_basis = NULL,
                                  bandwidth = NULL,
                                  min_oof_improvement = 0.00,
                                  train_seed = 123L,
                                  ...) {
  if (!requireNamespace("ranger", quietly = TRUE)) {
    stop("DBTARF requires ranger.")
  }
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("DBTARF requires caret.")
  }

  train_df <- as.data.frame(train_df)
  test_df <- as.data.frame(test_df)
  y <- as.numeric(train_df[[response_name]])
  n <- nrow(train_df)
  n_folds <- as.integer(max(2L, min(n_folds, n)))
  lambda_grid <- .dbt_lambda_grid(lambda_grid)

  restore_rng <- .dbt_preserve_global_rng()
  on.exit(restore_rng(), add = TRUE)

  set.seed(as.integer(train_seed))
  folds <- caret::createFolds(y, k = n_folds, list = TRUE,
                              returnTrain = FALSE)

  base_oof <- rep(NA_real_, n)
  cubist_oof <- rep(NA_real_, n)
  corr_oof <- list(
    bias = rep(NA_real_, n),
    cubist_delta = rep(NA_real_, n),
    resid_rf = rep(NA_real_, n),
    resid_cubist = rep(NA_real_, n),
    resid_spatial = rep(NA_real_, n)
  )
  fold_rows <- list()

  cat(sprintf("[DBTARF] cross-fitting residual candidates (n=%d, folds=%d)...\n",
              n, n_folds))
  for (j in seq_along(folds)) {
    id_val <- folds[[j]]
    id_tr <- setdiff(seq_len(n), id_val)
    tr <- train_df[id_tr, , drop = FALSE]
    va <- train_df[id_val, , drop = FALSE]

    rf_j <- .dbt_fit_rf(tr, response_name, predictor_names)
    pred_tr_rf <- .dbt_oob_or_in_sample(rf_j, tr)
    pred_va_rf <- .dbt_predict_rf(rf_j, va)
    base_oof[id_val] <- pred_va_rf

    x_tr <- as.data.frame(tr[, predictor_names, drop = FALSE])
    x_va <- as.data.frame(va[, predictor_names, drop = FALSE])
    cubist_j <- .dbt_fit_cubist(x_tr, tr[[response_name]])
    pred_tr_cb <- .dbt_predict_cubist(cubist_j, x_tr)
    pred_va_cb <- .dbt_predict_cubist(cubist_j, x_va)
    cubist_oof[id_val] <- pred_va_cb

    resid_tr <- as.numeric(tr[[response_name]]) - pred_tr_rf
    bias_j <- mean(resid_tr, na.rm = TRUE)
    corr_oof$bias[id_val] <- bias_j
    corr_oof$cubist_delta[id_val] <- pred_va_cb - pred_va_rf

    coords_tr <- as.matrix(tr[, coord_names, drop = FALSE])
    coords_va <- as.matrix(va[, coord_names, drop = FALSE])
    basis_j <- .dbt_fit_basis(coords_tr, resid_tr, n_basis = n_basis,
                              bandwidth = bandwidth,
                              seed = train_seed + j)
    feat_tr <- .dbt_residual_features(
      tr, predictor_names, coord_names, pred_tr_rf, pred_tr_cb, basis_j)
    feat_va <- .dbt_residual_features(
      va, predictor_names, coord_names, pred_va_rf, pred_va_cb, basis_j)

    resid_rf_j <- tryCatch(.dbt_fit_residual_rf(feat_tr, resid_tr),
                           error = function(e) NULL)
    if (!is.null(resid_rf_j)) {
      corr_oof$resid_rf[id_val] <- .dbt_predict_residual_rf(resid_rf_j, feat_va)
    }

    resid_cb_j <- .dbt_fit_cubist(feat_tr, resid_tr)
    if (!is.null(resid_cb_j)) {
      corr_oof$resid_cubist[id_val] <- .dbt_predict_cubist(resid_cb_j, feat_va)
    }

    corr_oof$resid_spatial[id_val] <- .dbt_kernel_residual_predict(
      coords_tr, resid_tr, coords_va, basis_j$bandwidth)

    fold_rows[[j]] <- data.frame(
      fold = j,
      n_train = length(id_tr),
      n_val = length(id_val),
      rf_rmse = .dbt_rmse(va[[response_name]], pred_va_rf),
      cubist_rmse = .dbt_rmse(va[[response_name]], pred_va_cb),
      residual_sd = stats::sd(resid_tr, na.rm = TRUE),
      n_basis = basis_j$n_basis,
      bandwidth = basis_j$bandwidth,
      range_major = basis_j$range_major,
      nugget_ratio = basis_j$nugget_ratio,
      stringsAsFactors = FALSE
    )
  }

  baseline_rmse <- .dbt_rmse(y, base_oof)
  candidate_rows <- list(data.frame(
    mode = "RF_noharm", lambda = 0, oof_rmse = baseline_rmse,
    delta_rmse = 0, stringsAsFactors = FALSE))

  valid_corr <- corr_oof
  valid_corr$resid_ensemble <- rowMeans(as.data.frame(
    valid_corr[c("resid_rf", "resid_cubist", "resid_spatial")]),
    na.rm = TRUE)
  valid_corr$resid_ensemble[!is.finite(valid_corr$resid_ensemble)] <- 0

  for (nm in names(valid_corr)) {
    corr <- as.numeric(valid_corr[[nm]])
    corr[!is.finite(corr)] <- 0
    for (lam in lambda_grid) {
      rm <- .dbt_rmse(y, base_oof + lam * corr)
      candidate_rows[[length(candidate_rows) + 1L]] <- data.frame(
        mode = nm, lambda = lam, oof_rmse = rm,
        delta_rmse = rm - baseline_rmse, stringsAsFactors = FALSE)
    }
  }

  cal_df <- data.frame(y = y, rf = base_oof, cubist = cubist_oof)
  if (sum(is.finite(cal_df$rf)) >= 20L) {
    lm1 <- stats::lm(y ~ rf, data = cal_df)
    p_lm1 <- as.numeric(stats::predict(lm1, newdata = cal_df))
    candidate_rows[[length(candidate_rows) + 1L]] <- data.frame(
      mode = "linear_cal", lambda = NA_real_, oof_rmse = .dbt_rmse(y, p_lm1),
      delta_rmse = .dbt_rmse(y, p_lm1) - baseline_rmse,
      stringsAsFactors = FALSE)

    lm2 <- stats::lm(y ~ rf + I(rf ^ 2), data = cal_df)
    p_lm2 <- as.numeric(stats::predict(lm2, newdata = cal_df))
    candidate_rows[[length(candidate_rows) + 1L]] <- data.frame(
      mode = "quadratic_cal", lambda = NA_real_, oof_rmse = .dbt_rmse(y, p_lm2),
      delta_rmse = .dbt_rmse(y, p_lm2) - baseline_rmse,
      stringsAsFactors = FALSE)
  }
  if (sum(is.finite(cal_df$rf) & is.finite(cal_df$cubist)) >= 20L) {
    stk <- stats::lm(y ~ rf + cubist, data = cal_df)
    p_stk <- as.numeric(stats::predict(stk, newdata = cal_df))
    candidate_rows[[length(candidate_rows) + 1L]] <- data.frame(
      mode = "stack_rf_cubist", lambda = NA_real_,
      oof_rmse = .dbt_rmse(y, p_stk),
      delta_rmse = .dbt_rmse(y, p_stk) - baseline_rmse,
      stringsAsFactors = FALSE)
  }

  candidates <- do.call(rbind, candidate_rows)
  candidates <- candidates[order(candidates$oof_rmse), , drop = FALSE]
  best <- candidates[1L, , drop = FALSE]
  if (!is.finite(best$oof_rmse) ||
      best$oof_rmse > baseline_rmse - as.numeric(min_oof_improvement)) {
    best <- candidates[candidates$mode == "RF_noharm", , drop = FALSE][1L, ]
  }

  cat(sprintf("[DBTARF] OOF RF RMSE=%.3f | selected=%s lambda=%s OOF RMSE=%.3f\n",
              baseline_rmse, best$mode,
              ifelse(is.na(best$lambda), "NA", sprintf("%.2f", best$lambda)),
              best$oof_rmse))

  rf_full <- .dbt_fit_rf(train_df, response_name, predictor_names)
  base_train_full <- .dbt_oob_or_in_sample(rf_full, train_df)
  base_test <- .dbt_predict_rf(rf_full, test_df)
  x_train <- as.data.frame(train_df[, predictor_names, drop = FALSE])
  x_test <- as.data.frame(test_df[, predictor_names, drop = FALSE])
  cubist_full <- .dbt_fit_cubist(x_train, y)
  cubist_train <- .dbt_predict_cubist(cubist_full, x_train)
  cubist_test <- .dbt_predict_cubist(cubist_full, x_test)

  mode <- as.character(best$mode)
  lambda <- suppressWarnings(as.numeric(best$lambda))
  pred_test <- base_test
  selected_correction_test <- rep(0, length(base_test))

  if (mode == "linear_cal") {
    lm1 <- stats::lm(y ~ rf, data = cal_df)
    pred_test <- as.numeric(stats::predict(lm1, newdata = data.frame(rf = base_test)))
  } else if (mode == "quadratic_cal") {
    lm2 <- stats::lm(y ~ rf + I(rf ^ 2), data = cal_df)
    pred_test <- as.numeric(stats::predict(lm2, newdata = data.frame(rf = base_test)))
  } else if (mode == "stack_rf_cubist") {
    stk <- stats::lm(y ~ rf + cubist, data = cal_df)
    pred_test <- as.numeric(stats::predict(
      stk, newdata = data.frame(rf = base_test, cubist = cubist_test)))
  } else if (mode != "RF_noharm") {
    residual_full <- y - base_train_full
    basis_full <- .dbt_fit_basis(
      as.matrix(train_df[, coord_names, drop = FALSE]), residual_full,
      n_basis = n_basis, bandwidth = bandwidth, seed = train_seed)
    feat_train <- .dbt_residual_features(
      train_df, predictor_names, coord_names, base_train_full,
      cubist_train, basis_full)
    feat_test <- .dbt_residual_features(
      test_df, predictor_names, coord_names, base_test,
      cubist_test, basis_full)

    if (mode == "bias") {
      selected_correction_test <- rep(mean(residual_full, na.rm = TRUE),
                                      length(base_test))
    } else if (mode == "cubist_delta") {
      selected_correction_test <- cubist_test - base_test
    } else if (mode == "resid_rf") {
      mdl <- .dbt_fit_residual_rf(feat_train, residual_full)
      selected_correction_test <- .dbt_predict_residual_rf(mdl, feat_test)
    } else if (mode == "resid_cubist") {
      mdl <- .dbt_fit_cubist(feat_train, residual_full)
      selected_correction_test <- .dbt_predict_cubist(mdl, feat_test)
    } else if (mode == "resid_spatial") {
      selected_correction_test <- .dbt_kernel_residual_predict(
        as.matrix(train_df[, coord_names, drop = FALSE]), residual_full,
        as.matrix(test_df[, coord_names, drop = FALSE]), basis_full$bandwidth)
    } else if (mode == "resid_ensemble") {
      parts <- list()
      mdl_rf <- tryCatch(.dbt_fit_residual_rf(feat_train, residual_full),
                         error = function(e) NULL)
      if (!is.null(mdl_rf)) parts$resid_rf <- .dbt_predict_residual_rf(mdl_rf, feat_test)
      mdl_cb <- .dbt_fit_cubist(feat_train, residual_full)
      if (!is.null(mdl_cb)) parts$resid_cubist <- .dbt_predict_cubist(mdl_cb, feat_test)
      parts$resid_spatial <- .dbt_kernel_residual_predict(
        as.matrix(train_df[, coord_names, drop = FALSE]), residual_full,
        as.matrix(test_df[, coord_names, drop = FALSE]), basis_full$bandwidth)
      selected_correction_test <- rowMeans(as.data.frame(parts), na.rm = TRUE)
      selected_correction_test[!is.finite(selected_correction_test)] <- 0
    }
    selected_correction_test[!is.finite(selected_correction_test)] <- 0
    pred_test <- base_test + lambda * selected_correction_test
  }

  if (any(!is.finite(pred_test))) pred_test[!is.finite(pred_test)] <- base_test[!is.finite(pred_test)]

  test_obs <- if (response_name %in% names(test_df)) test_df[[response_name]] else rep(NA_real_, nrow(test_df))
  list(
    pred_test = as.numeric(pred_test),
    pred_test_base = as.numeric(base_test),
    pred_test_cubist = as.numeric(cubist_test),
    correction_test = as.numeric(selected_correction_test),
    metrics_test = data.frame(
      N = length(test_obs),
      RMSE = .dbt_rmse(test_obs, pred_test),
      RMSE_base = .dbt_rmse(test_obs, base_test),
      stringsAsFactors = FALSE
    ),
    diagnostics = list(
      selected_mode = mode,
      selected_lambda = lambda,
      oof_rf_rmse = baseline_rmse,
      selected_oof_rmse = best$oof_rmse,
      selected_oof_delta = best$oof_rmse - baseline_rmse,
      n_folds = n_folds,
      n_train = n,
      min_oof_improvement = min_oof_improvement
    ),
    candidate_table = candidates,
    fold_diagnostics = do.call(rbind, fold_rows)
  )
}

dbtarf_default_params <- list(
  n_folds = 5L,
  lambda_grid = NULL,
  n_basis = NULL,
  bandwidth = NULL,
  min_oof_improvement = 0.00,
  train_seed = 123L
)
