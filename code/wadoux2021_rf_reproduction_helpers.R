source("./code/geospatial_env_setup.R", local = TRUE)

# Paths resolve through code/project_paths.R rather than being hardcoded to one
# machine. The benchmark data is gitignored, so GEOVERSA_DATA lets a reader
# point at wherever they cloned it; see data/README.md.
local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
external_root_wadoux <- gv_data_root(required = FALSE)
project_root_wadoux <- if (is.na(external_root_wadoux)) gv_root() else
  dirname(dirname(external_root_wadoux))

# Algorithm source code may live in the current working repo even when the DATA
# lives under project_root_wadoux (e.g. after the project folder was renamed
# from "Deep Kriging" to "GeoVersa"). Resolve the code root to wherever the
# algorithm sources actually exist, preferring the current repo.
code_root_wadoux <- local({
  cands <- c(gv_path("code"), file.path(project_root_wadoux, "code"))
  hit <- cands[file.exists(file.path(cands, "ConvKrigingNet2D.R"))]
  if (length(hit) > 0L) hit[1L] else gv_path("code")
})

# ---- Spatial-block CV utilities (shared across models; "#5") ----------------
# Clark-Evans aggregation index R = observed / expected nearest-neighbour
# distance under complete spatial randomness over the bounding box. R < 1 =>
# clustered, ~1 => random, > 1 => dispersed. NA if undefined.
wadoux_clark_evans <- function(coords) {
  coords <- as.matrix(coords)
  ok <- stats::complete.cases(coords)
  coords <- coords[ok, , drop = FALSE]
  n <- nrow(coords)
  if (n < 10L || ncol(coords) < 2L) return(NA_real_)
  d <- as.matrix(stats::dist(coords[, 1:2, drop = FALSE])); diag(d) <- Inf
  obs <- mean(apply(d, 1L, min), na.rm = TRUE)
  rx <- diff(range(coords[, 1], na.rm = TRUE))
  ry <- diff(range(coords[, 2], na.rm = TRUE))
  area <- rx * ry
  if (!is.finite(area) || area <= 0 || !is.finite(obs) || obs <= 0) return(NA_real_)
  obs / (0.5 / sqrt(n / area))
}

# TRUE when the calibration coords are detectably clustered (engage spatial CV).
# Used so spatial-block CV helps under clustered/extrapolative sampling without
# regressing representative (random) sampling, where random folds match the SRS
# test draw.
wadoux_is_clustered <- function(coords, thresh = getOption("wadoux_clark_thresh", 0.80)) {
  ce <- wadoux_clark_evans(coords)
  is.finite(ce) && ce < thresh
}

# k spatial blocks via k-means on scaled coordinates -> list of validation-index
# vectors (same shape as a random k-fold). Returns NULL if coords unusable.
wadoux_spatial_block_folds <- function(coords, k, seed = NULL) {
  coords <- as.matrix(coords); n <- nrow(coords)
  k <- max(2L, min(as.integer(k), n))
  ok <- stats::complete.cases(coords)
  if (sum(ok) < k || ncol(coords) < 1L) return(NULL)
  cs <- scale(coords); cs[!is.finite(cs)] <- 0
  if (!is.null(seed) && is.finite(seed)) {
    if (exists(".Random.seed", envir = .GlobalEnv)) {
      .old_seed <- get(".Random.seed", envir = .GlobalEnv)
      on.exit(assign(".Random.seed", .old_seed, envir = .GlobalEnv), add = TRUE)
    }
    set.seed(as.integer(seed))
  }
  km <- tryCatch(stats::kmeans(cs[ok, , drop = FALSE], centers = k,
                               iter.max = 50L, nstart = 5L)$cluster,
                 error = function(e) NULL)
  if (is.null(km)) return(NULL)
  fid <- rep(NA_integer_, n); fid[which(ok)] <- as.integer(km)
  miss <- which(!is.finite(fid))
  if (length(miss) > 0L) fid[miss] <- rep(seq_len(k), length.out = length(miss))
  folds <- lapply(seq_len(k), function(j) which(fid == j))
  folds <- folds[vapply(folds, length, integer(1)) > 0L]
  if (length(folds) < 2L) NULL else folds
}

# Same spatial blocks but returned as a per-row fold-id vector (for callers that
# use a fold_id vector instead of a list of index vectors). NULL if unusable.
wadoux_spatial_block_fold_ids <- function(coords, k, seed = NULL) {
  folds <- wadoux_spatial_block_folds(coords, k, seed)
  if (is.null(folds)) return(NULL)
  ids <- rep(NA_integer_, nrow(as.matrix(coords)))
  for (j in seq_along(folds)) ids[folds[[j]]] <- j
  ids
}

# ---- #3: per-location extrapolation gate ------------------------------------
# Weight w(s) in [0,1] for blending a spatial model with RF PER TEST LOCATION by
# local calibration support: w ~ 0 where a test point sits in dense calibration
# support (RF / interpolation regime is strong), w ~ 1 where it is far from any
# calibration point (extrapolation regime, favour the spatial process). Distances
# are normalized by a high quantile of the WITHIN-calibration nearest-neighbour
# distance, so it is scale-free. This is a learned-style, support-aware gate, NOT
# fixed-variogram regression kriging.
wadoux_support_gate_weight <- function(test_coords, cal_coords, k = 4,
                                       ref_quantile = 0.90) {
  test_coords <- as.matrix(test_coords)[, 1:2, drop = FALSE]
  cal_coords <- as.matrix(cal_coords)[, 1:2, drop = FALSE]
  nt <- nrow(test_coords)
  if (nrow(cal_coords) < 2L || nt < 1L) return(rep(0, max(nt, 0L)))
  dcal <- as.matrix(stats::dist(cal_coords)); diag(dcal) <- Inf
  cal_nn <- apply(dcal, 1L, min)
  d_ref <- suppressWarnings(stats::quantile(cal_nn, ref_quantile, na.rm = TRUE))
  if (!is.finite(d_ref) || d_ref <= 0) d_ref <- mean(cal_nn[is.finite(cal_nn)])
  if (!is.finite(d_ref) || d_ref <= 0) return(rep(0, nt))
  calt <- t(cal_coords)
  d_nn_test <- vapply(seq_len(nt), function(i) {
    sqrt(min(colSums((calt - test_coords[i, ])^2)))
  }, numeric(1))
  w <- stats::plogis((d_nn_test / d_ref - 1) * k)
  w[!is.finite(w)] <- 0
  pmin(pmax(w, 0), 1)
}

# ---- #6: design-based conformal prediction intervals + coverage --------------
# Split-conformal half-width from OUT-OF-FOLD absolute residuals. Feed the #5
# spatial-block OOF residuals so the interval is calibrated for the clustered /
# extrapolation regime -> coverage (PICP) holds design-based, not just in-sample.
# q = (n+1)*level/n is the finite-sample conformal quantile (Angelopoulos &
# Bates). Any model exposing OOF residuals can be compared to RF/QRF on coverage.
wadoux_conformal_halfwidth <- function(oof_abs_resid, level = 0.90) {
  r <- oof_abs_resid[is.finite(oof_abs_resid)]
  n <- length(r)
  if (n < 5L) return(NA_real_)
  qlev <- min(1, ((n + 1) * level) / n)
  as.numeric(stats::quantile(r, probs = qlev, type = 1L, names = FALSE))
}

# Symmetric conformal intervals around pred_test from OOF abs residuals; reports
# PICP / MPIW / ACE (|PICP-level|) per level on the (design-based SRS) test draw.
wadoux_conformal_coverage <- function(obs_test, pred_test, oof_abs_resid,
                                      levels = c(0.90, 0.95)) {
  obs_test <- as.numeric(obs_test); pred_test <- as.numeric(pred_test)
  keep <- is.finite(obs_test) & is.finite(pred_test)
  out <- list()
  for (lv in levels) {
    hw <- wadoux_conformal_halfwidth(oof_abs_resid, lv)
    suf <- sprintf("%02d", round(lv * 100))
    if (is.finite(hw) && any(keep)) {
      cov <- mean(abs(obs_test[keep] - pred_test[keep]) <= hw)
      out[[paste0("conf_picp_", suf)]] <- cov
      out[[paste0("conf_mpiw_", suf)]] <- 2 * hw
      out[[paste0("conf_ace_", suf)]] <- abs(cov - lv)
    } else {
      out[[paste0("conf_picp_", suf)]] <- NA_real_
      out[[paste0("conf_mpiw_", suf)]] <- NA_real_
      out[[paste0("conf_ace_", suf)]] <- NA_real_
    }
  }
  as.data.frame(out, stringsAsFactors = FALSE)
}

pkgs_wadoux <- c("raster", "ranger", "caret", "dplyr", "sp")
to_install_wadoux <- pkgs_wadoux[!sapply(pkgs_wadoux, requireNamespace, quietly = TRUE)]
if (length(to_install_wadoux) > 0) {
  install.packages(to_install_wadoux, repos = "https://cloud.r-project.org")
}

library(raster)
library(ranger)
library(caret)
library(dplyr)
library(sp)

predList_modelfull_wadoux <- c(
  "AI_glob", "CC_am", "Clay", "Elev", "ETP_Glob", "G_mean", "NIR_mean",
  "OCS", "Prec_am", "Prec_Dm", "Prec_seaso", "Prec_Wm", "R_mean", "Sand",
  "Sha_EVI", "Slope", "Soc", "solRad_m", "SolRad_sd", "SWIR1_mean",
  "SWIR2_mean", "T_am", "T_mdq", "T_mwarmq", "T_seaso", "Terra_PP",
  "Vapor_m", "Vapor_sd"
)
response_name_wadoux <- "ABG1"

load_wadoux_common_data <- function(include_polygon = FALSE) {
  files <- list.files(
    path = file.path(external_root_wadoux, "data"),
    pattern = "\\.tif$",
    full.names = TRUE
  )
  s <- raster::stack(files)
  s[[1]][s[[1]] == 0] <- NA
  s_df <- as.data.frame(s, xy = TRUE, na.rm = TRUE)
  s_df <- na.omit(as.data.frame(s_df))
  s_df$..wadoux_row_id <- seq_len(nrow(s_df))

  out <- list(
    stack = s,
    s_df = s_df
  )

  if (isTRUE(include_polygon)) {
    poly_env <- new.env(parent = emptyenv())
    load(file.path(external_root_wadoux, "code", "polygon.Rdata"), envir = poly_env)
    out$pp <- poly_env$pp
  }

  out
}

wadoux_eval <- function(obs, pred, corr_method = getOption("wadoux_corr_method", "pearson")) {
  corr_method <- match.arg(corr_method, c("pearson", "spearman"))
  me <- round(mean(pred - obs, na.rm = TRUE), digits = 2)
  rmse <- round(sqrt(mean((pred - obs)^2, na.rm = TRUE)), digits = 2)
  r2 <- round((cor(pred, obs, method = corr_method, use = "pairwise.complete.obs")^2), digits = 2)
  sse <- sum((pred - obs)^2, na.rm = TRUE)
  sst <- sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE)
  mec <- round((1 - sse / sst), digits = 2)
  data.frame(ME = me, RMSE = rmse, r2 = r2, MEC = mec)
}

wadoux_eval_gaussian_uq <- function(obs,
                                    pred,
                                    logvar,
                                    levels = c(0.90, 0.95),
                                    reliability_levels = seq(0.10, 0.95, by = 0.05)) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  logvar <- suppressWarnings(as.numeric(logvar))
  keep <- is.finite(obs) & is.finite(pred) & is.finite(logvar)
  if (!any(keep)) {
    return(data.frame(
      uq_available = FALSE,
      gaussian_nll = NA_real_,
      sigma_mean = NA_real_,
      sigma_median = NA_real_,
      reliability_ece = NA_real_,
      picp_90 = NA_real_,
      mpiw_90 = NA_real_,
      ace_90 = NA_real_,
      picp_95 = NA_real_,
      mpiw_95 = NA_real_,
      ace_95 = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  obs <- obs[keep]
  pred <- pred[keep]
  logvar <- pmin(pmax(logvar[keep], -6), 4)
  sigma <- sqrt(pmax(exp(logvar), 1e-8))
  z_abs <- abs(obs - pred)
  gaussian_nll <- mean(0.5 * (log(2 * pi) + logvar + ((obs - pred)^2) / pmax(exp(logvar), 1e-8)), na.rm = TRUE)

  rel_cov <- vapply(reliability_levels, function(level) {
    zq <- stats::qnorm((1 + level) / 2)
    mean(z_abs <= zq * sigma, na.rm = TRUE)
  }, numeric(1))
  reliability_ece <- mean(abs(rel_cov - reliability_levels), na.rm = TRUE)

  out <- data.frame(
    uq_available = TRUE,
    gaussian_nll = gaussian_nll,
    sigma_mean = mean(sigma, na.rm = TRUE),
    sigma_median = stats::median(sigma, na.rm = TRUE),
    reliability_ece = reliability_ece,
    stringsAsFactors = FALSE
  )

  for (level in levels) {
    zq <- stats::qnorm((1 + level) / 2)
    half_width <- zq * sigma
    picp <- mean(z_abs <= half_width, na.rm = TRUE)
    mpiw <- mean(2 * half_width, na.rm = TRUE)
    suffix <- sprintf("%02d", round(level * 100))
    out[[sprintf("picp_%s", suffix)]] <- picp
    out[[sprintf("mpiw_%s", suffix)]] <- mpiw
    out[[sprintf("ace_%s", suffix)]] <- abs(picp - level)
  }

  out
}

.wadoux_empty_transformed_uq_row <- function(levels = c(0.90, 0.95)) {
  out <- data.frame(
    uq_available = FALSE,
    uq_nll = NA_real_,
    reliability_ece = NA_real_,
    stringsAsFactors = FALSE
  )
  for (level in levels) {
    suffix <- sprintf("%02d", round(level * 100))
    out[[sprintf("picp_%s", suffix)]] <- NA_real_
    out[[sprintf("mpiw_%s", suffix)]] <- NA_real_
    out[[sprintf("ace_%s", suffix)]] <- NA_real_
  }
  out
}

wadoux_gaussian_reliability_curve <- function(obs,
                                              pred,
                                              logvar,
                                              levels = seq(0.10, 0.95, by = 0.05)) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  logvar <- suppressWarnings(as.numeric(logvar))
  keep <- is.finite(obs) & is.finite(pred) & is.finite(logvar)
  if (!any(keep)) {
    return(data.frame())
  }

  obs <- obs[keep]
  pred <- pred[keep]
  logvar <- pmin(pmax(logvar[keep], -6), 4)
  sigma <- sqrt(pmax(exp(logvar), 1e-8))
  z_abs <- abs(obs - pred)

  bind_rows(lapply(levels, function(level) {
    zq <- stats::qnorm((1 + level) / 2)
    half_width <- zq * sigma
    empirical <- mean(z_abs <= half_width, na.rm = TRUE)
    data.frame(
      nominal_coverage = level,
      empirical_coverage = empirical,
      abs_calibration_error = abs(empirical - level),
      mean_interval_width = mean(2 * half_width, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
}

.wadoux_inverse_target_scaler <- function(pred_scaled, target_scaler) {
  as.numeric(pred_scaled) * as.numeric(target_scaler$scale) + as.numeric(target_scaler$center)
}

# Split-conformal calibration scores (Angelopoulos & Bates 2023).
# Computes standardised absolute residuals on the calibration (val) set:
#   s_i = |y_i_t - μ_i| / σ_i  (in transformed space)
# These scores replace the Gaussian quantile z_α as follows:
#   q̂_α = quantile(scores, ceil((n+1)·α / n))      # α-th quantile, not (1-α)
# E.g. for α=0.90, n=100: ceil(90.9)/100 = 0.91 → 91st percentile.
# Resulting interval: μ ± q̂_α·σ  →  guaranteed P(Y ∈ interval) ≥ α.
compute_conformal_scores <- function(obs_raw_val,
                                     pred_val_scaled,
                                     pred_val_logvar_scaled,
                                     target_scaler,
                                     target_transform = "identity") {
  obs_raw_val <- as.numeric(obs_raw_val)
  pred_val_scaled <- as.numeric(pred_val_scaled)
  pred_val_logvar_scaled <- suppressWarnings(as.numeric(pred_val_logvar_scaled))
  keep <- is.finite(obs_raw_val) & is.finite(pred_val_scaled) & is.finite(pred_val_logvar_scaled)
  if (sum(keep) < 5L) return(NULL)
  obs_raw_val <- obs_raw_val[keep]
  pred_val_scaled <- pred_val_scaled[keep]
  pred_val_logvar_scaled <- pmin(pmax(pred_val_logvar_scaled[keep], -6), 4)
  mean_t <- .wadoux_inverse_target_scaler(pred_val_scaled, target_scaler)
  sd_t <- abs(as.numeric(target_scaler$scale)) *
    sqrt(pmax(exp(pred_val_logvar_scaled), 1e-8))
  target_transform <- tolower(trimws(as.character(target_transform[1])))
  if (!target_transform %in% c("identity", "log")) target_transform <- "identity"
  if (identical(target_transform, "log")) {
    keep_pos <- obs_raw_val > 0 & is.finite(mean_t) & is.finite(sd_t) & sd_t > 0
    if (sum(keep_pos) < 5L) return(NULL)
    scores <- abs(log(obs_raw_val[keep_pos]) - mean_t[keep_pos]) / sd_t[keep_pos]
  } else {
    keep_fin <- is.finite(mean_t) & is.finite(sd_t) & sd_t > 0
    if (sum(keep_fin) < 5L) return(NULL)
    scores <- abs(obs_raw_val[keep_fin] - mean_t[keep_fin]) / sd_t[keep_fin]
  }
  n <- length(scores)
  q90 <- as.numeric(quantile(scores, min(1, ceiling((n + 1L) * 0.90) / n),
                             type = 1L, na.rm = TRUE))
  cat(sprintf(
    "[Conformal] n_cal=%d  median_s=%.3f  q̂_0.90=%.3f  (z_0.90=%.3f)  ratio=%.3f\n",
    n, median(scores, na.rm = TRUE), q90, qnorm(0.95), q90 / qnorm(0.95)
  ))
  scores
}

wadoux_eval_transformed_gaussian_uq <- function(obs_raw,
                                                pred_scaled,
                                                logvar_scaled,
                                                target_scaler,
                                                target_transform = "identity",
                                                levels = c(0.90, 0.95),
                                                reliability_levels = seq(0.10, 0.95, by = 0.05),
                                                conformal_scores = NULL) {
  obs_raw <- as.numeric(obs_raw)
  pred_scaled <- as.numeric(pred_scaled)
  logvar_scaled <- suppressWarnings(as.numeric(logvar_scaled))
  keep <- is.finite(obs_raw) & is.finite(pred_scaled) & is.finite(logvar_scaled)
  if (!any(keep)) {
    return(.wadoux_empty_transformed_uq_row(levels = levels))
  }

  obs_raw <- obs_raw[keep]
  pred_scaled <- pred_scaled[keep]
  logvar_scaled <- pmin(pmax(logvar_scaled[keep], -6), 4)
  mean_t <- .wadoux_inverse_target_scaler(pred_scaled, target_scaler)
  sd_t <- abs(as.numeric(target_scaler$scale)) * sqrt(pmax(exp(logvar_scaled), 1e-8))
  var_t <- pmax(sd_t^2, 1e-8)

  target_transform <- tolower(trimws(as.character(target_transform[1])))
  if (!target_transform %in% c("identity", "log")) target_transform <- "identity"

  if (identical(target_transform, "log")) {
    keep_pos <- obs_raw > 0 & is.finite(mean_t) & is.finite(sd_t)
    if (!any(keep_pos)) {
      return(.wadoux_empty_transformed_uq_row(levels = levels))
    }
    obs_raw <- obs_raw[keep_pos]
    mean_t <- mean_t[keep_pos]
    sd_t <- sd_t[keep_pos]
    var_t <- var_t[keep_pos]
    obs_t <- log(obs_raw)
    uq_nll <- mean(
      0.5 * (log(2 * pi) + log(var_t) + ((obs_t - mean_t)^2) / var_t) + log(obs_raw),
      na.rm = TRUE
    )
    reliability_curve <- bind_rows(lapply(reliability_levels, function(level) {
      zq <- if (!is.null(conformal_scores) && length(conformal_scores) >= 5L) {
        n_cal <- length(conformal_scores)
        conf_level <- min(1.0, ceiling((n_cal + 1L) * level) / n_cal)
        as.numeric(quantile(conformal_scores, probs = conf_level,
                            type = 1L, na.rm = TRUE))
      } else {
        stats::qnorm((1 + level) / 2)
      }
      lower <- exp(mean_t - zq * sd_t)
      upper <- exp(mean_t + zq * sd_t)
      empirical <- mean(obs_raw >= lower & obs_raw <= upper, na.rm = TRUE)
      data.frame(
        nominal_coverage = level,
        empirical_coverage = empirical,
        abs_calibration_error = abs(empirical - level),
        mean_interval_width = mean(upper - lower, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    }))
  } else {
    obs_t <- obs_raw
    uq_nll <- mean(0.5 * (log(2 * pi) + log(var_t) + ((obs_t - mean_t)^2) / var_t), na.rm = TRUE)
    reliability_curve <- bind_rows(lapply(reliability_levels, function(level) {
      zq <- if (!is.null(conformal_scores) && length(conformal_scores) >= 5L) {
        n_cal <- length(conformal_scores)
        conf_level <- min(1.0, ceiling((n_cal + 1L) * level) / n_cal)
        as.numeric(quantile(conformal_scores, probs = conf_level,
                            type = 1L, na.rm = TRUE))
      } else {
        stats::qnorm((1 + level) / 2)
      }
      lower <- mean_t - zq * sd_t
      upper <- mean_t + zq * sd_t
      empirical <- mean(obs_raw >= lower & obs_raw <= upper, na.rm = TRUE)
      data.frame(
        nominal_coverage = level,
        empirical_coverage = empirical,
        abs_calibration_error = abs(empirical - level),
        mean_interval_width = mean(upper - lower, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    }))
  }

  out <- data.frame(
    uq_available = TRUE,
    uq_nll = uq_nll,
    reliability_ece = mean(reliability_curve$abs_calibration_error, na.rm = TRUE),
    stringsAsFactors = FALSE
  )

  for (level in levels) {
    suffix <- sprintf("%02d", round(level * 100))
    row_i <- reliability_curve[abs(reliability_curve$nominal_coverage - level) < 1e-8, , drop = FALSE]
    out[[sprintf("picp_%s", suffix)]] <- if (nrow(row_i) > 0) row_i$empirical_coverage[1] else NA_real_
    out[[sprintf("mpiw_%s", suffix)]] <- if (nrow(row_i) > 0) row_i$mean_interval_width[1] else NA_real_
    out[[sprintf("ace_%s", suffix)]] <- if (nrow(row_i) > 0) row_i$abs_calibration_error[1] else NA_real_
  }

  out
}

wadoux_transformed_gaussian_reliability_curve <- function(obs_raw,
                                                          pred_scaled,
                                                          logvar_scaled,
                                                          target_scaler,
                                                          target_transform = "identity",
                                                          levels = seq(0.10, 0.95, by = 0.05),
                                                          conformal_scores = NULL) {
  obs_raw <- as.numeric(obs_raw)
  pred_scaled <- as.numeric(pred_scaled)
  logvar_scaled <- suppressWarnings(as.numeric(logvar_scaled))
  keep <- is.finite(obs_raw) & is.finite(pred_scaled) & is.finite(logvar_scaled)
  if (!any(keep)) {
    return(data.frame())
  }
  obs_raw <- obs_raw[keep]
  pred_scaled <- pred_scaled[keep]
  logvar_scaled <- pmin(pmax(logvar_scaled[keep], -6), 4)
  mean_t <- .wadoux_inverse_target_scaler(pred_scaled, target_scaler)
  sd_t <- abs(as.numeric(target_scaler$scale)) * sqrt(pmax(exp(logvar_scaled), 1e-8))
  target_transform <- tolower(trimws(as.character(target_transform[1])))
  if (!target_transform %in% c("identity", "log")) target_transform <- "identity"

  bind_rows(lapply(levels, function(level) {
    zq <- if (!is.null(conformal_scores) && length(conformal_scores) >= 5L) {
      n_cal <- length(conformal_scores)
      conf_level <- min(1.0, ceiling((n_cal + 1L) * level) / n_cal)
      as.numeric(quantile(conformal_scores, probs = conf_level,
                          type = 1L, na.rm = TRUE))
    } else {
      stats::qnorm((1 + level) / 2)
    }
    if (identical(target_transform, "log")) {
      keep_pos <- obs_raw > 0 & is.finite(mean_t) & is.finite(sd_t)
      lower <- exp(mean_t[keep_pos] - zq * sd_t[keep_pos])
      upper <- exp(mean_t[keep_pos] + zq * sd_t[keep_pos])
      empirical <- mean(obs_raw[keep_pos] >= lower & obs_raw[keep_pos] <= upper, na.rm = TRUE)
      width <- mean(upper - lower, na.rm = TRUE)
    } else {
      lower <- mean_t - zq * sd_t
      upper <- mean_t + zq * sd_t
      empirical <- mean(obs_raw >= lower & obs_raw <= upper, na.rm = TRUE)
      width <- mean(upper - lower, na.rm = TRUE)
    }
    data.frame(
      nominal_coverage = level,
      empirical_coverage = empirical,
      abs_calibration_error = abs(empirical - level),
      mean_interval_width = width,
      stringsAsFactors = FALSE
    )
  }))
}

default_rf_qrf_quantile_levels <- function(levels = c(0.90, 0.95),
                                           reliability_levels = seq(0.10, 0.95, by = 0.05)) {
  sort(unique(round(c(
    0.16, 0.50, 0.84,
    (1 - levels) / 2, (1 + levels) / 2,
    (1 - reliability_levels) / 2, (1 + reliability_levels) / 2
  ), 6)))
}

.rf_qrf_pick_quantile <- function(q_matrix, q_levels, level) {
  idx <- which(abs(as.numeric(q_levels) - as.numeric(level)) < 1e-8)
  if (length(idx) < 1L) {
    stop(sprintf("Requested RF quantile %.6f is not available in q_levels.", level))
  }
  as.numeric(q_matrix[, idx[1]])
}

.rf_qrf_interval_bounds <- function(pred_mean,
                                    pred_sd,
                                    q_matrix = NULL,
                                    q_levels = NULL,
                                    level,
                                    conformal_scores = NULL) {
  pred_mean <- as.numeric(pred_mean)
  pred_sd <- pmax(as.numeric(pred_sd), 1e-6)

  if (!is.null(conformal_scores) && length(conformal_scores) >= 5L) {
    n_cal <- length(conformal_scores)
    conf_level <- min(1.0, ceiling((n_cal + 1L) * level) / n_cal)
    qhat <- as.numeric(quantile(conformal_scores, probs = conf_level, type = 1L, na.rm = TRUE))
    return(list(
      lower = pred_mean - qhat * pred_sd,
      upper = pred_mean + qhat * pred_sd
    ))
  }

  if (!is.null(q_matrix) && !is.null(q_levels)) {
    lower_prob <- (1 - level) / 2
    upper_prob <- (1 + level) / 2
    return(list(
      lower = .rf_qrf_pick_quantile(q_matrix, q_levels, lower_prob),
      upper = .rf_qrf_pick_quantile(q_matrix, q_levels, upper_prob)
    ))
  }

  zq <- stats::qnorm((1 + level) / 2)
  list(
    lower = pred_mean - zq * pred_sd,
    upper = pred_mean + zq * pred_sd
  )
}

compute_rf_conformal_scores <- function(obs_cal, pred_cal_mean, pred_cal_sd) {
  obs_cal <- as.numeric(obs_cal)
  pred_cal_mean <- as.numeric(pred_cal_mean)
  pred_cal_sd <- pmax(as.numeric(pred_cal_sd), 1e-6)
  keep <- is.finite(obs_cal) & is.finite(pred_cal_mean) & is.finite(pred_cal_sd) & pred_cal_sd > 0
  if (sum(keep) < 5L) return(NULL)
  scores <- abs(obs_cal[keep] - pred_cal_mean[keep]) / pred_cal_sd[keep]
  n <- length(scores)
  q90 <- as.numeric(quantile(scores, min(1, ceiling((n + 1L) * 0.90) / n),
                             type = 1L, na.rm = TRUE))
  cat(sprintf(
    "[RF conformal] n_cal=%d  median_s=%.3f  q̂_0.90=%.3f  (z_0.90=%.3f)  ratio=%.3f\n",
    n, median(scores, na.rm = TRUE), q90, qnorm(0.95), q90 / qnorm(0.95)
  ))
  scores
}

wadoux_eval_rf_qrf_uq <- function(obs_raw,
                                  pred_mean,
                                  pred_sd,
                                  q_matrix = NULL,
                                  q_levels = NULL,
                                  levels = c(0.90, 0.95),
                                  reliability_levels = seq(0.10, 0.95, by = 0.05),
                                  conformal_scores = NULL) {
  obs_raw <- as.numeric(obs_raw)
  pred_mean <- as.numeric(pred_mean)
  pred_sd <- pmax(as.numeric(pred_sd), 1e-6)
  keep <- is.finite(obs_raw) & is.finite(pred_mean) & is.finite(pred_sd)
  if (!any(keep)) {
    return(.wadoux_empty_transformed_uq_row(levels = levels))
  }

  obs_raw <- obs_raw[keep]
  pred_mean <- pred_mean[keep]
  pred_sd <- pred_sd[keep]
  if (!is.null(q_matrix)) {
    q_matrix <- as.matrix(q_matrix)[keep, , drop = FALSE]
  }
  q_levels <- if (!is.null(q_levels)) as.numeric(q_levels) else NULL

  var_rf <- pmax(pred_sd^2, 1e-8)
  uq_nll <- mean(
    0.5 * (log(2 * pi) + log(var_rf) + ((obs_raw - pred_mean)^2) / var_rf),
    na.rm = TRUE
  )

  reliability_curve <- wadoux_rf_qrf_reliability_curve(
    obs_raw = obs_raw,
    pred_mean = pred_mean,
    pred_sd = pred_sd,
    q_matrix = q_matrix,
    q_levels = q_levels,
    levels = reliability_levels,
    conformal_scores = conformal_scores
  )

  out <- data.frame(
    uq_available = TRUE,
    uq_nll = uq_nll,
    reliability_ece = mean(reliability_curve$abs_calibration_error, na.rm = TRUE),
    stringsAsFactors = FALSE
  )

  for (level in levels) {
    suffix <- sprintf("%02d", round(level * 100))
    row_i <- reliability_curve[abs(reliability_curve$nominal_coverage - level) < 1e-8, , drop = FALSE]
    out[[sprintf("picp_%s", suffix)]] <- if (nrow(row_i) > 0) row_i$empirical_coverage[1] else NA_real_
    out[[sprintf("mpiw_%s", suffix)]] <- if (nrow(row_i) > 0) row_i$mean_interval_width[1] else NA_real_
    out[[sprintf("ace_%s", suffix)]] <- if (nrow(row_i) > 0) row_i$abs_calibration_error[1] else NA_real_
  }

  out
}

wadoux_rf_qrf_reliability_curve <- function(obs_raw,
                                            pred_mean,
                                            pred_sd,
                                            q_matrix = NULL,
                                            q_levels = NULL,
                                            levels = seq(0.10, 0.95, by = 0.05),
                                            conformal_scores = NULL) {
  obs_raw <- as.numeric(obs_raw)
  pred_mean <- as.numeric(pred_mean)
  pred_sd <- pmax(as.numeric(pred_sd), 1e-6)
  keep <- is.finite(obs_raw) & is.finite(pred_mean) & is.finite(pred_sd)
  if (!any(keep)) {
    return(data.frame())
  }

  obs_raw <- obs_raw[keep]
  pred_mean <- pred_mean[keep]
  pred_sd <- pred_sd[keep]
  if (!is.null(q_matrix)) {
    q_matrix <- as.matrix(q_matrix)[keep, , drop = FALSE]
  }
  q_levels <- if (!is.null(q_levels)) as.numeric(q_levels) else NULL

  bind_rows(lapply(levels, function(level) {
    interval <- .rf_qrf_interval_bounds(
      pred_mean = pred_mean,
      pred_sd = pred_sd,
      q_matrix = q_matrix,
      q_levels = q_levels,
      level = level,
      conformal_scores = conformal_scores
    )
    empirical <- mean(obs_raw >= interval$lower & obs_raw <= interval$upper, na.rm = TRUE)
    data.frame(
      nominal_coverage = level,
      empirical_coverage = empirical,
      abs_calibration_error = abs(empirical - level),
      mean_interval_width = mean(interval$upper - interval$lower, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
}

fit_rf_default_wadoux <- function(train_df, form_rf,
                                  seed = getOption("wadoux_rf_seed", 123L)) {
  seed <- suppressWarnings(as.integer(seed))
  if (!is.finite(seed)) seed <- 123L
  ranger::ranger(formula = form_rf, data = train_df, seed = seed)
}

predict_rf_default_wadoux <- function(model, new_df) {
  as.numeric(predict(model, data = new_df, type = "response")$predictions)
}

fit_rf_qrf_wadoux <- function(train_df, form_rf, seed = 123L) {
  ranger::ranger(
    formula = form_rf,
    data = train_df,
    quantreg = TRUE,
    keep.inbag = TRUE,
    seed = as.integer(seed)
  )
}

predict_rf_qrf_uq <- function(model, new_df,
                              levels = default_rf_qrf_quantile_levels()) {
  levels <- sort(unique(as.numeric(levels)))
  pred_obj <- predict(model, data = new_df, type = "quantiles", quantiles = levels)
  q_mat <- pred_obj$predictions
  if (is.null(dim(q_mat))) {
    q_mat <- matrix(q_mat, nrow = nrow(new_df), byrow = TRUE)
  } else {
    q_mat <- as.matrix(q_mat)
  }
  if (nrow(q_mat) != nrow(new_df) && ncol(q_mat) == nrow(new_df) && nrow(q_mat) == length(levels)) {
    q_mat <- t(q_mat)
  }
  if (nrow(q_mat) != nrow(new_df) || ncol(q_mat) != length(levels)) {
    stop(sprintf(
      "Unexpected QRF prediction shape: got %dx%d for %d rows and %d quantiles.",
      nrow(q_mat), ncol(q_mat), nrow(new_df), length(levels)
    ))
  }
  colnames(q_mat) <- sprintf("%.6f", levels)

  q16 <- .rf_qrf_pick_quantile(q_mat, levels, 0.16)
  q50 <- .rf_qrf_pick_quantile(q_mat, levels, 0.50)
  q84 <- .rf_qrf_pick_quantile(q_mat, levels, 0.84)

  list(
    mean = q50,
    sd = pmax((q84 - q16) / 2, 1e-6),
    q_low_90 = .rf_qrf_pick_quantile(q_mat, levels, 0.05),
    q_high_90 = .rf_qrf_pick_quantile(q_mat, levels, 0.95),
    q_low_95 = .rf_qrf_pick_quantile(q_mat, levels, 0.025),
    q_high_95 = .rf_qrf_pick_quantile(q_mat, levels, 0.975),
    q_matrix = q_mat,
    q_levels = levels
  )
}

sample_simple_random_rows_wadoux <- function(df, size) {
  df[sample.int(nrow(df), size = size, replace = FALSE), , drop = FALSE]
}

build_spatial_folds_wadoux <- function(valuetable, val_dist_km) {
  mdist <- dist(valuetable[, c("x", "y")])
  hc <- hclust(mdist, method = "complete")
  cutree(hc, h = val_dist_km * 1000)
}

run_population_protocol_wadoux <- function(valuetable, s_df, form_rf) {
  rf <- fit_rf_default_wadoux(valuetable, form_rf)
  pred <- predict_rf_default_wadoux(rf, s_df)
  wadoux_eval(obs = s_df[[response_name_wadoux]], pred = pred)
}

run_design_based_protocol_wadoux <- function(valuetable, s_df, form_rf, sample_size) {
  val_srs <- sample_simple_random_rows_wadoux(s_df, sample_size)
  rf <- fit_rf_default_wadoux(valuetable, form_rf)
  pred <- predict_rf_default_wadoux(rf, val_srs)
  wadoux_eval(obs = val_srs[[response_name_wadoux]], pred = pred)
}

run_random_kfold_protocol_wadoux <- function(valuetable, form_rf, k = 10) {
  flds <- caret::createFolds(
    valuetable[[response_name_wadoux]],
    k = k,
    list = TRUE,
    returnTrain = FALSE
  )

  pred_list <- vector("list", length(flds))
  obs_list <- vector("list", length(flds))

  for (j in seq_along(flds)) {
    id <- flds[[j]]
    training_data <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]

    rf <- fit_rf_default_wadoux(training_data, form_rf)
    pred_list[[j]] <- predict_rf_default_wadoux(rf, validation_data)
    obs_list[[j]] <- validation_data[[response_name_wadoux]]
  }

  wadoux_eval(
    obs = unlist(obs_list, use.names = FALSE),
    pred = unlist(pred_list, use.names = FALSE)
  )
}

run_spatial_kfold_protocol_wadoux <- function(valuetable, form_rf, val_dist_km) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  fold_ids <- unique(spatial_folds)

  pred_list <- vector("list", length(fold_ids))
  obs_list <- vector("list", length(fold_ids))

  for (j in seq_along(fold_ids)) {
    id <- which(spatial_folds == fold_ids[j])
    training_data <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]

    rf <- fit_rf_default_wadoux(training_data, form_rf)
    pred_list[[j]] <- predict_rf_default_wadoux(rf, validation_data)
    obs_list[[j]] <- validation_data[[response_name_wadoux]]
  }

  wadoux_eval(
    obs = unlist(obs_list, use.names = FALSE),
    pred = unlist(pred_list, use.names = FALSE)
  )
}

point_within_predictor_range_wadoux <- function(train_df, focal_row, predictor_names) {
  train_pred <- train_df[, predictor_names, drop = FALSE]
  focal_pred <- focal_row[, predictor_names, drop = FALSE]
  lower <- apply(train_pred, 2, min, na.rm = TRUE)
  upper <- apply(train_pred, 2, max, na.rm = TRUE)
  all(as.numeric(focal_pred[1, ]) >= lower & as.numeric(focal_pred[1, ]) <= upper)
}

exclude_by_radius_wadoux <- function(df, focal_row, radius_m) {
  dx <- df$x - focal_row$x
  dy <- df$y - focal_row$y
  keep <- sqrt(dx^2 + dy^2) > radius_m
  df[keep, , drop = FALSE]
}

run_bloocv_protocol_wadoux <- function(valuetable,
                                       form_rf,
                                       predictor_names,
                                       val_dist_km,
                                       nb_groups = 10,
                                       nb_test_pixels = 100) {
  r_list <- c(val_dist_km)
  nb_iteration <- nb_groups * nb_test_pixels
  data_res <- vector("list", nb_iteration * length(r_list))
  a <- 0L

  for (j in seq_len(nb_iteration)) {
    point_in_range <- FALSE

    while (!point_in_range) {
      id_focal <- sample.int(nrow(valuetable), size = 1)
      focal_point <- valuetable[id_focal, , drop = FALSE]
      training_tmp <- valuetable[-id_focal, , drop = FALSE]

      ri <- max(r_list) * 1000
      training_tmp <- exclude_by_radius_wadoux(training_tmp, focal_point, ri)

      if (nrow(training_tmp) < 10) {
        next
      }

      if (!point_within_predictor_range_wadoux(training_tmp, focal_point, predictor_names)) {
        next
      }

      nb_training <- nrow(training_tmp)
      point_in_range <- TRUE
    }

    training_tmp_j <- valuetable[-id_focal, , drop = FALSE]

    for (i in seq_along(r_list)) {
      ri <- r_list[i] * 1000
      training_tmp <- exclude_by_radius_wadoux(training_tmp_j, focal_point, ri)

      if (nrow(training_tmp) < nb_training) {
        next
      }

      nb_training_dif <- nrow(training_tmp) - nb_training
      if (nb_training_dif > 0) {
        training_tmp <- training_tmp[
          -sample.int(nrow(training_tmp), size = nb_training_dif),
          ,
          drop = FALSE
        ]
      }

      loss_cells <- nrow(training_tmp_j) - nrow(training_tmp)

      rf <- fit_rf_default_wadoux(training_tmp, form_rf)
      pred_rf <- predict_rf_default_wadoux(rf, focal_point)

      a <- a + 1L
      data_res[[a]] <- data.frame(
        R = r_list[i],
        N_training = nb_training,
        N_cell_lost = loss_cells,
        N_cell_training = nrow(training_tmp),
        AGB = focal_point[[response_name_wadoux]],
        Pred_RF_FULL = pred_rf,
        AGB_calibrationDATA = mean(training_tmp[[response_name_wadoux]], na.rm = TRUE)
      )
    }
  }

  data_res <- bind_rows(data_res)
  x <- unique(data_res$R)
  bloo_metrics <- vector("list", length(x))

  for (i in seq_along(x)) {
    tmp <- data_res[data_res$R == x[i], , drop = FALSE]
    tmp$flds <- caret::createFolds(
      seq_len(nrow(tmp)),
      k = nb_groups,
      list = FALSE,
      returnTrain = FALSE
    )

    r2_rf_list <- c()
    me_rf_list <- c()
    rmse_rf_list <- c()
    mec_rf_list <- c()

    for (j in seq_len(max(tmp$flds))) {
      idx <- which(tmp$flds == j)
      map_quality <- wadoux_eval(obs = tmp$AGB[idx], pred = tmp$Pred_RF_FULL[idx])
      r2_rf_list <- c(r2_rf_list, map_quality$r2)
      me_rf_list <- c(me_rf_list, map_quality$ME)
      rmse_rf_list <- c(rmse_rf_list, map_quality$RMSE)
      mec_rf_list <- c(mec_rf_list, map_quality$MEC)
    }

    bloo_metrics[[i]] <- data.frame(
      ME = mean(me_rf_list, na.rm = TRUE),
      RMSE = mean(rmse_rf_list, na.rm = TRUE),
      r2 = mean(r2_rf_list, na.rm = TRUE),
      MEC = mean(mec_rf_list, na.rm = TRUE)
    )
  }

  bind_rows(bloo_metrics)[1, , drop = FALSE]
}

make_metric_table_wadoux <- function(population_metrics,
                                     design_metrics,
                                     random_metrics,
                                     spatial_metrics,
                                     bloo_metrics,
                                     protocols = c("Population", "DesignBased", "RandomKFold", "SpatialKFold", "BLOOCV")) {
  metric_names <- c("ME", "RMSE", "r2", "MEC")
  out <- data.frame(metric = metric_names, stringsAsFactors = FALSE)
  metrics_by_protocol <- list(
    Population = population_metrics,
    DesignBased = design_metrics,
    RandomKFold = random_metrics,
    SpatialKFold = spatial_metrics,
    BLOOCV = bloo_metrics
  )

  for (protocol in protocols) {
    metrics_df <- metrics_by_protocol[[protocol]]
    if (is.null(metrics_df)) {
      out[[protocol]] <- rep(NA_real_, length(metric_names))
    } else {
      out[[protocol]] <- unname(unlist(metrics_df[1, metric_names]))
    }
  }

  out
}

make_delta_table_wadoux <- function(metric_table) {
  if (!"Population" %in% names(metric_table)) {
    return(data.frame(metric = metric_table$metric, stringsAsFactors = FALSE))
  }

  protocol_cols <- setdiff(names(metric_table), c("metric", "Population"))
  out <- data.frame(metric = metric_table$metric, stringsAsFactors = FALSE)
  for (protocol in protocol_cols) {
    out[[protocol]] <- metric_table[[protocol]] - metric_table$Population
  }
  out
}

save_checkpoint_wadoux <- function(iteration, absolute_rows, delta_rows, results_dir, prefix) {
  absolute_df <- bind_rows(absolute_rows)
  delta_df <- bind_rows(delta_rows)

  write.csv(
    absolute_df,
    file.path(results_dir, sprintf("%s_absolute_metrics.csv", prefix)),
    row.names = FALSE
  )
  write.csv(
    delta_df,
    file.path(results_dir, sprintf("%s_delta_metrics.csv", prefix)),
    row.names = FALSE
  )
  saveRDS(
    list(absolute = absolute_df, delta = delta_df, iteration = iteration),
    file.path(results_dir, sprintf("%s_checkpoint.rds", prefix))
  )
}

finalize_outputs_wadoux <- function(absolute_rows, delta_rows, results_dir, prefix) {
  absolute_df <- bind_rows(absolute_rows)
  delta_df <- bind_rows(delta_rows)
  absolute_protocols <- setdiff(names(absolute_df), c("iteration", "metric"))
  delta_protocols <- setdiff(names(delta_df), c("iteration", "metric"))

  absolute_long <- bind_rows(lapply(seq_len(nrow(absolute_df)), function(i) {
    row <- absolute_df[i, ]
    data.frame(
      iteration = row$iteration,
      metric = row$metric,
      protocol = absolute_protocols,
      value = unname(unlist(row[absolute_protocols]))
    )
  }))

  if (length(delta_protocols) > 0) {
    delta_long <- bind_rows(lapply(seq_len(nrow(delta_df)), function(i) {
      row <- delta_df[i, ]
      data.frame(
        iteration = row$iteration,
        metric = row$metric,
        protocol = delta_protocols,
        delta = unname(unlist(row[delta_protocols]))
      )
    }))
  } else {
    delta_long <- data.frame(
      iteration = integer(),
      metric = character(),
      protocol = character(),
      delta = numeric()
    )
  }

  absolute_summary <- absolute_long %>%
    group_by(metric, protocol) %>%
    summarise(
      mean_value = mean(value, na.rm = TRUE),
      sd_value = sd(value, na.rm = TRUE),
      median_value = median(value, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(metric, protocol)

  if (nrow(delta_long) > 0) {
    delta_summary <- delta_long %>%
      group_by(metric, protocol) %>%
      summarise(
        mean_delta = mean(delta, na.rm = TRUE),
        sd_delta = sd(delta, na.rm = TRUE),
        median_delta = median(delta, na.rm = TRUE),
        q05_delta = unname(quantile(delta, probs = 0.05, na.rm = TRUE)),
        q95_delta = unname(quantile(delta, probs = 0.95, na.rm = TRUE)),
        .groups = "drop"
      ) %>%
      arrange(metric, protocol)
  } else {
    delta_summary <- data.frame(
      metric = character(),
      protocol = character(),
      mean_delta = numeric(),
      sd_delta = numeric(),
      median_delta = numeric(),
      q05_delta = numeric(),
      q95_delta = numeric()
    )
  }

  write.csv(
    absolute_long,
    file.path(results_dir, sprintf("%s_absolute_long.csv", prefix)),
    row.names = FALSE
  )
  write.csv(
    delta_long,
    file.path(results_dir, sprintf("%s_delta_long.csv", prefix)),
    row.names = FALSE
  )
  write.csv(
    absolute_summary,
    file.path(results_dir, sprintf("%s_absolute_summary.csv", prefix)),
    row.names = FALSE
  )
  write.csv(
    delta_summary,
    file.path(results_dir, sprintf("%s_delta_summary.csv", prefix)),
    row.names = FALSE
  )

  list(
    absolute = absolute_df,
    delta = delta_df,
    absolute_long = absolute_long,
    delta_long = delta_long,
    absolute_summary = absolute_summary,
    delta_summary = delta_summary
  )
}

sample_valuetable_regular_grid_wadoux <- function(common_data, sample_size) {
  pt.reg <- sp::spsample(common_data$pp, n = sample_size, type = "regular")
  valuetable <- raster::extract(common_data$stack, pt.reg, sp = TRUE, df = TRUE, na.rm = TRUE)
  valuetable <- na.omit(as.data.frame(valuetable))
  names(valuetable)[names(valuetable) == "x1"] <- "x"
  names(valuetable)[names(valuetable) == "x2"] <- "y"
  valuetable
}

twostage_wadoux <- function(sframe, psu, n, m) {
  units <- sample.int(nrow(sframe), size = n, replace = TRUE)
  mypsusample <- sframe[units, psu]
  ssunits <- NULL
  for (psunit in mypsusample) {
    ssunit <- sample(x = which(sframe[, psu] == psunit), size = m, replace = FALSE)
    ssunits <- c(ssunits, ssunit)
  }
  psudraw <- rep(seq_len(n), each = m)
  data.frame(sframe[ssunits, ], psudraw)
}

sample_valuetable_clustered_random_wadoux <- function(common_data,
                                                      kmeans_centers = 100,
                                                      n_psu = 20,
                                                      m_per_psu = 25) {
  sp.grid <- common_data$s_df[c("x", "y")]
  tkmean <- stats::kmeans(sp.grid, centers = kmeans_centers, nstart = 10)
  sp.grid$psu <- tkmean$cluster
  mysample <- twostage_wadoux(sframe = sp.grid, psu = "psu", n = n_psu, m = m_per_psu)
  sp::coordinates(mysample) <- ~x + y
  valuetable <- raster::extract(common_data$stack, mysample, sp = TRUE, df = TRUE, na.rm = TRUE)
  valuetable <- na.omit(as.data.frame(valuetable))
  names(valuetable)[names(valuetable) == "x1"] <- "x"
  names(valuetable)[names(valuetable) == "x2"] <- "y"
  valuetable
}

run_wadoux_rf_reproduction <- function(common_data,
                                       sample_valuetable_fn,
                                       sample_args,
                                       results_dir,
                                       prefix,
                                       scenario_name,
                                       config_extra = list(),
                                       protocols = c("Population", "DesignBased", "RandomKFold", "SpatialKFold", "BLOOCV"),
                                       sample_size = 500L,
                                       n_iter = 500L,
                                       val_dist_km = 350,
                                       random_k = 10L,
                                       bloo_groups = 10L,
                                       bloo_test_pixels = 100L,
                                       checkpoint_every = 10L) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  form_rf <- as.formula(
    paste(response_name_wadoux, "~", paste(predList_modelfull_wadoux, collapse = "+"))
  )

  config <- c(
    list(
      scenario = scenario_name,
      protocols = paste(protocols, collapse = ","),
      sample_size = sample_size,
      n_iter = n_iter,
      val_dist_km = val_dist_km,
      random_k = random_k,
      bloo_groups = bloo_groups,
      bloo_test_pixels = bloo_test_pixels
    ),
    config_extra
  )
  write.csv(
    as.data.frame(config, stringsAsFactors = FALSE),
    file.path(results_dir, sprintf("%s_config.csv", prefix)),
    row.names = FALSE
  )

  absolute_rows <- vector("list", n_iter)
  delta_rows <- vector("list", n_iter)

  for (sampling in seq_len(n_iter)) {
    cat(sprintf("\n==================================================\n"))
    cat(sprintf("Wadoux RF %s reproduction | iteration %d / %d\n", scenario_name, sampling, n_iter))
    cat(sprintf("==================================================\n"))

    valuetable <- do.call(
      sample_valuetable_fn,
      c(list(common_data = common_data), sample_args)
    )
    valuetable <- na.omit(as.data.frame(valuetable))

    population_metrics <- if ("Population" %in% protocols) {
      run_population_protocol_wadoux(valuetable, common_data$s_df, form_rf)
    } else NULL
    design_metrics <- if ("DesignBased" %in% protocols) {
      run_design_based_protocol_wadoux(valuetable, common_data$s_df, form_rf, sample_size)
    } else NULL
    random_metrics <- if ("RandomKFold" %in% protocols) {
      run_random_kfold_protocol_wadoux(valuetable, form_rf, k = random_k)
    } else NULL
    spatial_metrics <- if ("SpatialKFold" %in% protocols) {
      run_spatial_kfold_protocol_wadoux(valuetable, form_rf, val_dist_km = val_dist_km)
    } else NULL
    bloo_metrics <- if ("BLOOCV" %in% protocols) {
      run_bloocv_protocol_wadoux(
        valuetable = valuetable,
        form_rf = form_rf,
        predictor_names = predList_modelfull_wadoux,
        val_dist_km = val_dist_km,
        nb_groups = bloo_groups,
        nb_test_pixels = bloo_test_pixels
      )
    } else NULL

    metric_table <- make_metric_table_wadoux(
      population_metrics = population_metrics,
      design_metrics = design_metrics,
      random_metrics = random_metrics,
      spatial_metrics = spatial_metrics,
      bloo_metrics = bloo_metrics,
      protocols = protocols
    )
    delta_table <- make_delta_table_wadoux(metric_table)

    absolute_rows[[sampling]] <- mutate(metric_table, iteration = sampling, .before = 1)
    delta_rows[[sampling]] <- mutate(delta_table, iteration = sampling, .before = 1)

    if (sampling %% checkpoint_every == 0L || sampling == n_iter) {
      save_checkpoint_wadoux(sampling, absolute_rows, delta_rows, results_dir, prefix)
    }
  }

  finalize_outputs_wadoux(absolute_rows, delta_rows, results_dir, prefix)
}
