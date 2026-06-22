# =============================================================================
# GeoBART.R
#
# GeoBART — Bayesian Additive Regression Trees with variogram-informed
# spatial basis features for Digital Soil Mapping.
#
# Novelty (unprecedented in Pedometrics, to the extent of our literature
# review): a principled, simple modification of Chipman, George & McCulloch's
# BART (2010) where the spatial component of the input is enriched with a
# Gaussian RBF basis whose number, bandwidth and centres are derived from
# the empirical variogram of the target field.
#
# Mathematical sketch:
#
#   y_i = sum_{t=1}^{T} g_t(z_i; T_t, M_t) + epsilon_i,    epsilon ~ N(0, sigma^2)
#
#   z_i = [ x_i ;            <- p tabular covariates
#           s_i ;            <- 2 raw coordinates
#           phi(s_i) ]       <- K variogram-informed Gaussian basis features
#
#   phi_k(s) = exp( -|| s - c_k ||^2 / (2 h^2) )
#
# with:
#   - K       (number of basis functions) derived from area / range^2 and
#             nugget ratio:    K = clamp( 4 A (1-r) / (pi rho^2) , 16, 80 )
#   - h       (bandwidth)     = rho / 2  (rho = variogram practical range)
#   - c_k     (centres)       = k-means cluster centres of the training
#                               coordinates (centres adapt to point density)
#
# This is mathematically equivalent to giving BART access to a finite-rank
# Karhunen-Loeve approximation of the spatial GP whose covariance matches
# the fitted variogram. BART's tree splits on phi_k(s) act as soft regional
# indicators with scale tied to the data's actual spatial autocorrelation —
# the variogram becomes a structural prior, not a hyperparameter.
#
# The rest of GeoBART is deliberately classical:
#   - target_transform auto-selected via Shapiro-Wilk + positivity
#   - Duan smearing correction conditional on n_val >= 500 (when log)
#   - BART hyperparameters auto-configured by default and optionally selected
#     by an explicit inner-validation candidate search
#
# One fit function plus a fair-tuning wrapper; no torch / GPU machinery.
#
# Dependencies (loaded as needed):
#   - BART       (Sparapani et al., CRAN)
#   - gstat, sp  (variogram fit)
#
# =============================================================================

# ---- Self-contained helpers (avoid torch / NMK source dependency) -----------

.gb_clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

.gb_env_num <- function(name, default) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) default else val
}

.gb_env_num_vec <- function(name, default) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  vals <- suppressWarnings(as.numeric(trimws(unlist(strsplit(raw, ",")))))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0L) default else vals
}

.gb_env_char_vec <- function(name, default) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  vals <- tolower(trimws(unlist(strsplit(raw, ","))))
  vals <- vals[nzchar(vals)]
  if (length(vals) == 0L) default else vals
}

.gb_env_bool <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}

.gb_param <- function(params, name, default = NULL) {
  if (is.null(params) || is.null(params[[name]])) default else params[[name]]
}

.gb_log <- function(verbose, ...) {
  if (isTRUE(verbose)) cat(...)
}

.gb_allowed_spatial_policies <- function() {
  c("coords_basis", "coords_only", "basis_only", "tabular_only",
    "coords_aniso_basis", "aniso_basis_only")
}

.gb_normalize_spatial_policy <- function(policy) {
  policy <- tolower(trimws(as.character(policy)[1L]))
  policy <- gsub("[^a-z0-9]+", "_", policy)
  aliases <- c(
    default = "coords_basis",
    rbf = "coords_basis",
    basis_coords = "coords_basis",
    coords_rbf = "coords_basis",
    no_basis = "coords_only",
    coord_only = "coords_only",
    coords = "coords_only",
    basis = "basis_only",
    rbf_only = "basis_only",
    no_spatial = "tabular_only",
    tabular = "tabular_only",
    x_only = "tabular_only",
    anisotropic = "coords_aniso_basis",
    aniso = "coords_aniso_basis",
    coords_aniso = "coords_aniso_basis",
    aniso_only = "aniso_basis_only"
  )
  if (policy %in% names(aliases)) policy <- aliases[[policy]]
  if (!(policy %in% .gb_allowed_spatial_policies())) "coords_basis" else policy
}

.gb_policy_uses_coords <- function(policy) {
  .gb_normalize_spatial_policy(policy) %in%
    c("coords_basis", "coords_only", "coords_aniso_basis")
}

.gb_policy_uses_basis <- function(policy) {
  .gb_normalize_spatial_policy(policy) %in%
    c("coords_basis", "basis_only", "coords_aniso_basis", "aniso_basis_only")
}

.gb_policy_uses_aniso <- function(policy) {
  .gb_normalize_spatial_policy(policy) %in%
    c("coords_aniso_basis", "aniso_basis_only")
}

.gb_fold_ids <- function(y, k, seed = NULL) {
  n <- length(y)
  k <- as.integer(k)
  if (!is.finite(k) || k < 2L || n < 4L) return(rep(1L, n))
  k <- min(k, floor(n / 2L))
  if (!is.null(seed)) set.seed(as.integer(seed))
  ord <- order(y, runif(n), na.last = TRUE)
  ids <- integer(n)
  blocks <- split(seq_len(n), ceiling(seq_len(n) / k))
  for (b in blocks) {
    ids[b] <- sample(rep(seq_len(k), length.out = length(b)))
  }
  out <- integer(n)
  out[ord] <- ids
  out
}

.gb_fd_pool_train_val <- function(fd) {
  list(
    X = rbind(as.matrix(fd$X$train), as.matrix(fd$X$val)),
    coords = rbind(as.matrix(fd$coords$train), as.matrix(fd$coords$val)),
    y = c(fd$y$train, fd$y$val)
  )
}

.gb_fd_from_pool <- function(pool, train_idx, val_idx) {
  list(
    X = list(
      train = pool$X[train_idx, , drop = FALSE],
      val = pool$X[val_idx, , drop = FALSE],
      test = pool$X[val_idx, , drop = FALSE]
    ),
    coords = list(
      train = pool$coords[train_idx, , drop = FALSE],
      val = pool$coords[val_idx, , drop = FALSE],
      test = pool$coords[val_idx, , drop = FALSE]
    ),
    y = list(
      train = pool$y[train_idx],
      val = pool$y[val_idx],
      test = pool$y[val_idx]
    )
  )
}

# ---- Target transform (mirrors NMK auto-selector but standalone) -----------

geobart_target_transform_auto <- function(y_train) {
  y_clean <- y_train[is.finite(y_train)]
  n       <- length(y_clean)
  if (n < 3L) return("identity")
  if (any(y_clean <= 0)) return("identity")
  n_sw <- min(n, 5000L)
  y_samp <- if (n_sw < n) sample(y_clean, n_sw) else y_clean
  sw <- tryCatch(shapiro.test(y_samp),
                 error = function(e) list(p.value = 1))
  p <- if (is.null(sw$p.value)) 1 else sw$p.value
  if (!is.finite(p)) p <- 1
  if (p < 0.05) "log" else "identity"
}

geobart_apply_target_transform <- function(y, method) {
  if (identical(method, "log")) {
    if (any(y <= 0, na.rm = TRUE)) {
      stop("GeoBART: log transform requires strictly positive y; auto-selector should have caught this.")
    }
    log(y)
  } else {
    y
  }
}

geobart_invert_target_transform <- function(y, method) {
  if (identical(method, "log")) exp(y) else y
}

# ---- Robust Duan smearing (mirrors DKP helper, kept local for autonomy) ----

geobart_duan_smearing <- function(log_residuals,
                                  trim_frac  = 0.05,
                                  min_val_n  = 500L) {
  r <- log_residuals[is.finite(log_residuals)]
  n <- length(r)
  if (n < as.integer(min_val_n)) {
    return(list(S = 1.0, applied = FALSE,
                reason = sprintf("n_val = %d < min_val_n = %d", n, min_val_n)))
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
                reason = sprintf("trimmed mean non-finite (raw=%.3f)", S_raw)))
  }
  list(S = S_trim, S_raw = S_raw, S_trimmed = S_trim,
       applied = TRUE, n_used = length(er_trim),
       reason = sprintf("trimmed-mean Duan applied (n=%d, trim=%.2f)",
                        length(er_trim), trim_frac))
}


# ---- Variogram fit (small, gstat-based, no torch) --------------------------

.geobart_variogram_objects <- function(y, coords, cutoff_frac = 0.50,
                                       n_lags = 15L) {
  if (!requireNamespace("gstat", quietly = TRUE) ||
      !requireNamespace("sp",    quietly = TRUE)) {
    stop("GeoBART: gstat and sp are required for variogram fitting. ",
         "install.packages(c('gstat','sp')).")
  }
  y <- as.numeric(y)
  coords <- as.matrix(coords)
  ok <- is.finite(y) & stats::complete.cases(coords)
  y <- y[ok]
  coords <- coords[ok, , drop = FALSE]
  fallback_summary <- function(status, cutoff = NA_real_, sw_p = NA_real_,
                               transform = "raw", range_init = NA_real_) {
    list(range_major = range_init,
         nugget_ratio = 0.35,
         fit_quality = 0.25,
         area = NA_real_,
         variogram_status = status,
         variogram_transform = transform,
         variogram_shapiro_p = sw_p,
         variogram_cutoff = cutoff,
         variogram_n_lags = as.integer(n_lags),
         variogram_model = NA_character_,
         variogram_nugget = NA_real_,
         variogram_partial_sill = NA_real_,
         variogram_total_sill = NA_real_)
  }
  if (length(y) < 3L || nrow(coords) < 3L) {
    return(list(summary = fallback_summary("too_few_points"),
                empirical = data.frame(), model = data.frame(),
                fitted_line = data.frame()))
  }
  D <- as.matrix(dist(coords))
  cutoff <- max(D, na.rm = TRUE) * cutoff_frac
  if (!is.finite(cutoff) || cutoff <= 0) {
    return(list(summary = fallback_summary("invalid_coordinate_distances"),
                empirical = data.frame(), model = data.frame(),
                fitted_line = data.frame()))
  }

  # Optional NST so the variogram is robust to skew.
  sw_p <- tryCatch(
    shapiro.test(sample(y, min(length(y), 5000L)))$p.value,
    error = function(e) 1
  )
  transform_name <- if (sw_p < 0.05) "normal_score" else "raw"
  y_vg <- if (identical(transform_name, "normal_score")) {
    qx <- (rank(y) - 0.5) / length(y)
    qnorm(.gb_clamp(qx, 1e-6, 1 - 1e-6))
  } else {
    y
  }

  df <- data.frame(yv = y_vg, x = coords[, 1], y2 = coords[, 2])
  sp::coordinates(df) <- ~ x + y2

  vg_emp <- tryCatch(
    gstat::variogram(yv ~ 1, data = df, cutoff = cutoff,
                     width = cutoff / n_lags, cressie = TRUE),
    error = function(e) NULL
  )
  range_init <- cutoff * 0.30
  if (is.null(vg_emp) || nrow(vg_emp) < 3L) {
    return(list(summary = fallback_summary("empirical_variogram_failed",
                                           cutoff = cutoff, sw_p = sw_p,
                                           transform = transform_name,
                                           range_init = range_init),
                empirical = if (is.null(vg_emp)) data.frame() else as.data.frame(vg_emp),
                model = data.frame(), fitted_line = data.frame()))
  }

  sill_tot   <- var(y_vg, na.rm = TRUE)

  vg_fit <- tryCatch(
    suppressWarnings(gstat::fit.variogram(
      vg_emp,
      gstat::vgm(sill_tot * 0.80, "Exp", range_init, sill_tot * 0.20),
      fit.method = 7L
    )),
    error = function(e) NULL
  )

  fitted_line <- data.frame()
  if (!is.null(vg_fit) && all(is.finite(vg_fit$range))) {
    iso_range <- max(vg_fit$range[vg_fit$range > 0], na.rm = TRUE)
    if (!is.finite(iso_range) || iso_range <= 0) iso_range <- range_init
    psill_sum <- sum(vg_fit$psill, na.rm = TRUE)
    nugget_ratio <- if (is.finite(psill_sum) && psill_sum > 1e-8) {
      .gb_clamp(vg_fit$psill[1L] / psill_sum, 0.02, 0.90)
    } else 0.35
    fitted_line <- tryCatch(
      as.data.frame(gstat::variogramLine(vg_fit, maxdist = cutoff,
                                         n = 200L)),
      error = function(e) data.frame()
    )
    variogram_status <- "ok"
  } else {
    iso_range <- range_init
    nugget_ratio <- 0.35
    variogram_status <- "model_fit_failed"
  }

  area <- diff(range(coords[, 1])) * diff(range(coords[, 2]))
  nugget <- if (!is.null(vg_fit) && nrow(vg_fit) > 0L) {
    as.numeric(vg_fit$psill[1L])
  } else NA_real_
  partial_sill <- if (!is.null(vg_fit) && nrow(vg_fit) > 0L) {
    sum(vg_fit$psill[-1L], na.rm = TRUE)
  } else NA_real_
  total_sill <- if (!is.null(vg_fit) && nrow(vg_fit) > 0L) {
    sum(vg_fit$psill, na.rm = TRUE)
  } else NA_real_
  model_name <- if (!is.null(vg_fit) && nrow(vg_fit) >= 2L) {
    as.character(vg_fit$model[2L])
  } else NA_character_

  list(
    summary = list(range_major  = iso_range,
                   nugget_ratio = nugget_ratio,
                   fit_quality  = if (sw_p < 0.05) 0.5 else 0.7,
                   area         = area,
                   variogram_status = variogram_status,
                   variogram_transform = transform_name,
                   variogram_shapiro_p = sw_p,
                   variogram_cutoff = cutoff,
                   variogram_n_lags = as.integer(n_lags),
                   variogram_model = model_name,
                   variogram_nugget = nugget,
                   variogram_partial_sill = partial_sill,
                   variogram_total_sill = total_sill),
    empirical = as.data.frame(vg_emp),
    model = if (is.null(vg_fit)) data.frame() else as.data.frame(vg_fit),
    fitted_line = fitted_line
  )
}

geobart_fit_variogram <- function(y, coords, cutoff_frac = 0.50,
                                  n_lags = 15L) {
  obj <- .geobart_variogram_objects(y, coords, cutoff_frac = cutoff_frac,
                                    n_lags = n_lags)
  obj$summary
}

geobart_save_variogram_diagnostic <- function(y, coords, out_dir, prefix,
                                               title = NULL,
                                               cutoff_frac = 0.50,
                                               n_lags = 15L) {
  obj <- .geobart_variogram_objects(y, coords, cutoff_frac = cutoff_frac,
                                    n_lags = n_lags)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  safe_prefix <- gsub("[^A-Za-z0-9_\\-]+", "_", prefix)
  empirical_path <- file.path(out_dir, paste0(safe_prefix, "_empirical.csv"))
  model_path <- file.path(out_dir, paste0(safe_prefix, "_model.csv"))
  line_path <- file.path(out_dir, paste0(safe_prefix, "_fitted_line.csv"))
  plot_path <- file.path(out_dir, paste0(safe_prefix, ".png"))
  write.csv(obj$empirical, empirical_path, row.names = FALSE)
  write.csv(obj$model, model_path, row.names = FALSE)
  write.csv(obj$fitted_line, line_path, row.names = FALSE)

  if (requireNamespace("ggplot2", quietly = TRUE) && nrow(obj$empirical) > 0L) {
    p <- ggplot2::ggplot(obj$empirical,
                         ggplot2::aes(x = .data$dist, y = .data$gamma)) +
      ggplot2::geom_point(ggplot2::aes(size = .data$np),
                          alpha = 0.75, color = "#2f5d8c") +
      ggplot2::scale_size_continuous(name = "pairs", range = c(1.5, 5)) +
      ggplot2::labs(
        title = if (is.null(title)) "GeoBART empirical semivariogram" else title,
        subtitle = sprintf(
          "status=%s | model=%s | nugget ratio=%.2f | range=%.3g | transform=%s",
          obj$summary$variogram_status,
          ifelse(is.na(obj$summary$variogram_model), "NA",
                 obj$summary$variogram_model),
          obj$summary$nugget_ratio,
          obj$summary$range_major,
          obj$summary$variogram_transform
        ),
        x = "Lag distance",
        y = "Semivariance"
      ) +
      ggplot2::theme_minimal(base_size = 12)
    if (nrow(obj$fitted_line) > 0L) {
      p <- p + ggplot2::geom_line(
        data = obj$fitted_line,
        ggplot2::aes(x = .data$dist, y = .data$gamma),
        inherit.aes = FALSE, linewidth = 0.9, color = "#b13f2a"
      )
    }
    if (is.finite(obj$summary$variogram_total_sill)) {
      p <- p + ggplot2::geom_hline(
        yintercept = obj$summary$variogram_total_sill,
        linetype = "dashed", color = "#5f6368", linewidth = 0.4
      )
    }
    if (is.finite(obj$summary$range_major)) {
      p <- p + ggplot2::geom_vline(
        xintercept = obj$summary$range_major,
        linetype = "dotted", color = "#5f6368", linewidth = 0.4
      )
    }
    ggplot2::ggsave(plot_path, p, width = 7.5, height = 5.2, dpi = 160)
  } else {
    plot_path <- NA_character_
  }

  summary <- as.data.frame(obj$summary, stringsAsFactors = FALSE)
  summary$empirical_csv <- empirical_path
  summary$model_csv <- model_path
  summary$fitted_line_csv <- line_path
  summary$plot_png <- plot_path
  summary
}


# ---- Variogram-informed spatial basis features -----------------------------

# Place K basis centres via k-means on training coords. K-means adapts to
# point density (high-density regions get more centres), which is exactly
# what we want — the spatial prior reflects where the calibration actually
# is, not a uniform tile of the bounding box.
geobart_basis_centers <- function(coords_train, n_basis, seed = NULL) {
  if (!is.null(seed)) set.seed(as.integer(seed))
  coords_train <- as.matrix(coords_train)
  coords_train <- coords_train[stats::complete.cases(coords_train), , drop = FALSE]
  if (nrow(coords_train) == 0L) stop("GeoBART cannot build spatial basis: no valid coordinates.")
  unique_coords <- unique(coords_train)
  max_basis <- if (nrow(unique_coords) <= 2L) 1L else nrow(unique_coords) - 1L
  n_basis <- as.integer(min(n_basis, max_basis))
  if (n_basis <= 1L) return(matrix(colMeans(unique_coords), nrow = 1L))
  km <- stats::kmeans(unique_coords, centers = n_basis, iter.max = 50L,
	                      nstart = 5L)
  km$centers
}

geobart_cap_basis_count <- function(coords_train, n_basis) {
  n_basis <- suppressWarnings(as.integer(n_basis))
  if (!is.finite(n_basis) || n_basis <= 0L) return(0L)
  coords_train <- as.matrix(coords_train)
  coords_train <- coords_train[stats::complete.cases(coords_train), , drop = FALSE]
  if (nrow(coords_train) == 0L) return(0L)
  unique_coords <- unique(coords_train)
  max_basis <- if (nrow(unique_coords) <= 2L) 1L else nrow(unique_coords) - 1L
  as.integer(min(n_basis, max_basis))
}

# Build [N, K] Gaussian RBF feature matrix.
geobart_make_features <- function(coords, centers, bandwidth) {
  d2 <- outer(rowSums(coords ^ 2),
              rowSums(centers ^ 2), `+`) - 2 * coords %*% t(centers)
  d2[d2 < 0] <- 0
  exp(-d2 / (2 * bandwidth ^ 2))
}

geobart_basis_coordinate_system <- function(coords_train, coords_list,
                                            policy = "coords_basis",
                                            anisotropy_ratio = 2) {
  policy <- .gb_normalize_spatial_policy(policy)
  coords_train <- as.matrix(coords_train)
  coords_list <- lapply(coords_list, as.matrix)
  if (!.gb_policy_uses_aniso(policy)) {
    return(coords_list)
  }
  center <- colMeans(coords_train)
  centered <- sweep(coords_train, 2L, center, "-")
  rot <- diag(2L)
  if (nrow(centered) >= 3L && all(is.finite(centered))) {
    pc <- tryCatch(stats::prcomp(centered, center = FALSE, scale. = FALSE),
                   error = function(e) NULL)
    if (!is.null(pc) && all(dim(pc$rotation) == c(2L, 2L))) {
      rot <- pc$rotation
    }
  }
  ratio <- .gb_clamp(as.numeric(anisotropy_ratio), 1, 20)
  lapply(coords_list, function(x) {
    z <- sweep(x, 2L, center, "-") %*% rot
    z[, 2L] <- z[, 2L] * ratio
    z
  })
}

geobart_build_design_matrices <- function(Xtr, Xva, Xte,
                                          Ctr, Cva, Cte,
                                          cfg, seed = NULL) {
  policy <- .gb_normalize_spatial_policy(cfg$spatial_policy)
  raw_coord_names <- c("coord_x", "coord_y")
  colnames(Ctr) <- colnames(Cva) <- colnames(Cte) <- raw_coord_names

  blocks_train <- list(Xtr)
  blocks_val <- list(Xva)
  blocks_test <- list(Xte)
  block_names <- c("tabular")

  if (.gb_policy_uses_coords(policy)) {
    blocks_train[[length(blocks_train) + 1L]] <- Ctr
    blocks_val[[length(blocks_val) + 1L]] <- Cva
    blocks_test[[length(blocks_test) + 1L]] <- Cte
    block_names <- c(block_names, "coords")
  }

  n_basis_effective <- 0L
  if (.gb_policy_uses_basis(policy) && cfg$n_basis > 0L) {
    basis_coords <- geobart_basis_coordinate_system(
      coords_train = Ctr,
      coords_list = list(train = Ctr, val = Cva, test = Cte),
      policy = policy,
      anisotropy_ratio = cfg$anisotropy_ratio
    )
    centers <- geobart_basis_centers(
      basis_coords$train, cfg$n_basis, seed = seed)
    Phi_tr <- geobart_make_features(basis_coords$train, centers, cfg$bandwidth)
    Phi_va <- geobart_make_features(basis_coords$val, centers, cfg$bandwidth)
    Phi_te <- geobart_make_features(basis_coords$test, centers, cfg$bandwidth)
    basis_names <- sprintf("phi_%02d", seq_len(ncol(Phi_tr)))
    colnames(Phi_tr) <- colnames(Phi_va) <- colnames(Phi_te) <- basis_names
    blocks_train[[length(blocks_train) + 1L]] <- Phi_tr
    blocks_val[[length(blocks_val) + 1L]] <- Phi_va
    blocks_test[[length(blocks_test) + 1L]] <- Phi_te
    block_names <- c(block_names, if (.gb_policy_uses_aniso(policy)) {
      "aniso_basis"
    } else {
      "basis"
    })
    n_basis_effective <- ncol(Phi_tr)
  }

  list(
    train = do.call(cbind, blocks_train),
    val = do.call(cbind, blocks_val),
    test = do.call(cbind, blocks_test),
    n_basis_effective = as.integer(n_basis_effective),
    feature_blocks = paste(block_names, collapse = "+")
  )
}


# ---- Auto-config ------------------------------------------------------------
#
# All numeric knobs derive from the variogram + n + area. The fair-tuning
# wrapper expands those values into an explicit candidate grid and selects by
# validation RMSE (or another requested validation metric).

geobart_auto_config <- function(vg, n_train, coords_train) {
  r        <- vg$nugget_ratio
  rho      <- vg$range_major
  area     <- if (is.finite(vg$area)) vg$area else
              (diff(range(coords_train[, 1])) *
               diff(range(coords_train[, 2])))

  # K spatial basis: enough to cover the domain at half-range bandwidth,
  # downscaled by signal strength (1 - nugget_ratio).
  n_basis_theory <- 4 * area / (pi * rho ^ 2) * (1 - r)
  n_basis        <- as.integer(.gb_clamp(round(n_basis_theory), 16L, 80L))

  # Bandwidth h = rho / 2 (standard RBF cover rule of thumb).
  bandwidth <- rho / 2

  # BART defaults — already strong, only modest data dependence helpful.
  # Number of trees grows mildly with n: BART literature suggests 50-200.
  ntree  <- as.integer(.gb_clamp(round(40 + sqrt(n_train)), 50L, 200L))
  ndpost <- as.integer(.gb_env_num("GEOBART_NDPOST", 1000L))
  nskip  <- as.integer(.gb_env_num("GEOBART_NSKIP",   100L))

  list(
    n_basis        = n_basis,
    bandwidth      = bandwidth,
    ntree          = ntree,
    ndpost         = ndpost,
    nskip          = nskip,
    spatial_policy = .gb_normalize_spatial_policy(
      Sys.getenv("GEOBART_SPATIAL_POLICY", unset = "coords_basis")),
    anisotropy_ratio = .gb_env_num("GEOBART_ANISOTROPY_RATIO", 2),
    k              = .gb_env_num("GEOBART_K", 2),
    power          = .gb_env_num("GEOBART_POWER", 2),
    base           = .gb_env_num("GEOBART_BASE", 0.95),
    sigdf          = .gb_env_num("GEOBART_SIGDF", 3),
    sigquant       = .gb_env_num("GEOBART_SIGQUANT", 0.90),
    nugget_ratio   = r,
    range_major    = rho,
    fit_quality    = vg$fit_quality,
    area           = area,
    duan_min_val_n = as.integer(.gb_env_num("GEOBART_DUAN_MIN_VAL_N", 500L)),
    duan_trim_frac = .gb_env_num("GEOBART_DUAN_TRIM_FRAC", 0.05),
    duan_apply     = toupper(Sys.getenv("GEOBART_DUAN_APPLY", unset = "TRUE")) %in%
                     c("TRUE", "1", "YES", "Y")
  )
}

geobart_apply_config_overrides <- function(cfg, params = NULL,
                                           coords_train = NULL) {
  if (!is.null(.gb_param(params, "ntree"))) {
    cfg$ntree <- as.integer(.gb_param(params, "ntree"))
  }
  if (!is.null(.gb_param(params, "ndpost"))) {
    cfg$ndpost <- as.integer(.gb_param(params, "ndpost"))
  }
  if (!is.null(.gb_param(params, "nskip"))) {
    cfg$nskip <- as.integer(.gb_param(params, "nskip"))
  }
  if (!is.null(.gb_param(params, "n_basis"))) {
    cfg$n_basis <- as.integer(.gb_param(params, "n_basis"))
  }
  if (!is.null(.gb_param(params, "bandwidth"))) {
    cfg$bandwidth <- as.numeric(.gb_param(params, "bandwidth"))
  }
  if (!is.null(.gb_param(params, "spatial_policy"))) {
    cfg$spatial_policy <- .gb_normalize_spatial_policy(
      .gb_param(params, "spatial_policy"))
  }
  if (!is.null(.gb_param(params, "anisotropy_ratio"))) {
    cfg$anisotropy_ratio <- as.numeric(.gb_param(params, "anisotropy_ratio"))
  }
  for (nm in c("k", "power", "base", "sigdf", "sigquant")) {
    if (!is.null(.gb_param(params, nm))) {
      cfg[[nm]] <- as.numeric(.gb_param(params, nm))
    }
  }
  if (!is.null(.gb_param(params, "duan_apply"))) {
    cfg$duan_apply <- isTRUE(.gb_param(params, "duan_apply"))
  }
  if (!is.null(.gb_param(params, "duan_min_val_n"))) {
    cfg$duan_min_val_n <- as.integer(.gb_param(params, "duan_min_val_n"))
  }
  if (!is.null(.gb_param(params, "duan_trim_frac"))) {
    cfg$duan_trim_frac <- as.numeric(.gb_param(params, "duan_trim_frac"))
  }

  cfg$ntree <- as.integer(.gb_clamp(cfg$ntree, 1L, 10000L))
  cfg$ndpost <- as.integer(.gb_clamp(cfg$ndpost, 1L, 100000L))
  cfg$nskip <- as.integer(.gb_clamp(cfg$nskip, 0L, 100000L))
  cfg$spatial_policy <- .gb_normalize_spatial_policy(cfg$spatial_policy)
  cfg$anisotropy_ratio <- .gb_clamp(cfg$anisotropy_ratio, 1, 20)
  cfg$k <- .gb_clamp(cfg$k, 0.1, 20)
  cfg$power <- .gb_clamp(cfg$power, 0.1, 20)
  cfg$base <- .gb_clamp(cfg$base, 0.01, 0.999)
  cfg$sigdf <- .gb_clamp(cfg$sigdf, 1, 100)
  cfg$sigquant <- .gb_clamp(cfg$sigquant, 0.01, 0.999)
  if (!is.finite(cfg$bandwidth) || cfg$bandwidth <= 0) {
    coords_train <- as.matrix(coords_train)
    dmax <- if (!is.null(coords_train) && nrow(coords_train) >= 2L) {
      suppressWarnings(max(stats::dist(coords_train), na.rm = TRUE))
    } else {
      NA_real_
    }
    cfg$bandwidth <- if (is.finite(dmax) && dmax > 0) dmax / 4 else 1
  }
  if (!is.null(coords_train)) {
    cfg$n_basis <- geobart_cap_basis_count(coords_train, cfg$n_basis)
  }
  if (!.gb_policy_uses_basis(cfg$spatial_policy)) {
    cfg$n_basis <- 0L
  }
  cfg
}


# ---- Main training function -------------------------------------------------
#
# Interface: train_geobart_one_fold(fd, ...)
#
# fd is the standard fold dictionary used elsewhere in the project. GeoBART
# only needs:
#   fd$X$train / val / test       tabular covariate matrices
#   fd$coords$train / val / test  raw coordinates (n, 2)
#   fd$y$train / val / test       raw targets
# Patches and neighbor indices are ignored — BART does not use them.

train_geobart_one_fold <- function(
  fd,
  target_transform = "auto",
  ntree            = NULL,
  ndpost           = NULL,
  nskip            = NULL,
  n_basis          = NULL,
  bandwidth        = NULL,
  spatial_policy   = NULL,
  anisotropy_ratio = NULL,
  k                = NULL,
  power            = NULL,
  base             = NULL,
  sigdf            = NULL,
  sigquant         = NULL,
  train_seed       = NULL,
  duan_apply       = NULL,
  duan_min_val_n   = NULL,
  duan_trim_frac   = NULL,
  verbose          = TRUE,
  ...
) {
  if (!requireNamespace("BART", quietly = TRUE)) {
    stop("GeoBART requires the BART package. install.packages('BART').")
  }
  # --- RNG hygiene: do not leak our set.seed() into the caller's RNG stream ---
  # The Wadoux runner draws all sampling (calibration draws, k-fold, BLOOCV)
  # from the GLOBAL RNG. Because this function calls set.seed(train_seed) below,
  # without save/restore each GeoBART fit would silently reset the global seed
  # and destroy cross-protocol / cross-iteration independence. Save and restore.
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    .gb_saved_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
    on.exit({
      assign(".Random.seed", .gb_saved_seed, envir = globalenv())
    }, add = TRUE)
  } else {
    on.exit({
      if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
        rm(".Random.seed", envir = globalenv())
      }
    }, add = TRUE)
  }

  # ---- Extract fold data --------------------------------------------------
  Xtr <- as.matrix(fd$X$train)
  Xva <- as.matrix(fd$X$val)
  Xte <- as.matrix(fd$X$test)
  Ctr <- as.matrix(fd$coords$train)
  Cva <- as.matrix(fd$coords$val)
  Cte <- as.matrix(fd$coords$test)
  ytr <- fd$y$train
  yva <- fd$y$val
  yte <- fd$y$test
  n_train <- nrow(Xtr)

  # ---- Target transform ----------------------------------------------------
  if (!is.null(train_seed)) set.seed(as.integer(train_seed))
  target_transform <- tolower(as.character(target_transform))
  if (identical(target_transform, "auto")) {
    target_transform <- geobart_target_transform_auto(ytr)
    .gb_log(verbose, sprintf("[GeoBART] target_transform auto-selected -> '%s'\n",
                             target_transform))
  }
  ytr_t <- geobart_apply_target_transform(ytr, target_transform)
  yva_t <- geobart_apply_target_transform(yva, target_transform)
  # No standardisation: BART's prior is invariant to target scale (it
  # internally rescales). Standardising would just rename the units.

  # ---- Variogram + auto-config ---------------------------------------------
  .gb_log(verbose, "[GeoBART] === Variogram fit ===\n")
  vg  <- geobart_fit_variogram(ytr, Ctr)
  cfg <- geobart_auto_config(vg, n_train, Ctr)
  cfg <- geobart_apply_config_overrides(
    cfg,
    list(
      ntree = ntree, ndpost = ndpost, nskip = nskip,
      n_basis = n_basis, bandwidth = bandwidth,
      spatial_policy = spatial_policy, anisotropy_ratio = anisotropy_ratio,
      k = k, power = power, base = base, sigdf = sigdf, sigquant = sigquant,
      duan_apply = duan_apply, duan_min_val_n = duan_min_val_n,
      duan_trim_frac = duan_trim_frac
    ),
    coords_train = Ctr
  )

  .gb_log(verbose, sprintf(
    "[GeoBART] r=%.3f  rho=%.3g  area=%.3g  policy=%s  n_basis=%d  bandwidth=%.3g  ntree=%d\n",
    cfg$nugget_ratio, cfg$range_major, cfg$area, cfg$spatial_policy,
    cfg$n_basis, cfg$bandwidth, cfg$ntree))
  .gb_log(verbose, sprintf(
    "[GeoBART] target_transform=%s  ndpost=%d  nskip=%d  k=%.2f  base=%.3f  power=%.2f  sigdf=%.1f  sigquant=%.2f\n",
    target_transform, cfg$ndpost, cfg$nskip, cfg$k, cfg$base, cfg$power,
    cfg$sigdf, cfg$sigquant))

  # ---- Spatial basis features ---------------------------------------------
  # spatial_policy controls whether BART sees raw coordinates, RBF basis
  # functions, both, neither, or an anisotropic coordinate system for the basis.
  design <- geobart_build_design_matrices(
    Xtr, Xva, Xte, Ctr, Cva, Cte, cfg, seed = train_seed)
  X_train_full <- design$train
  X_val_full <- design$val
  X_test_full <- design$test

  .gb_log(verbose, sprintf(
    "[GeoBART] design matrix: %d tabular | blocks=%s | basis=%d | total=%d\n",
    ncol(Xtr), design$feature_blocks, design$n_basis_effective,
    ncol(X_train_full)))

  # ---- BART fit -----------------------------------------------------------
  if (!is.null(train_seed)) set.seed(as.integer(train_seed))
  .gb_log(verbose, sprintf("[GeoBART] fitting BART (n_train=%d, ntree=%d, ndpost=%d)...\n",
                           n_train, cfg$ntree, cfg$ndpost))
  t0 <- Sys.time()
  fit <- NULL
  fit_call <- function() {
    wbart_args <- list(
      x.train   = X_train_full,
      y.train   = as.numeric(ytr_t),
      x.test    = rbind(X_val_full, X_test_full),
      ntree     = cfg$ntree,
      ndpost    = cfg$ndpost,
      nskip     = cfg$nskip,
      printevery = 10000L
    )
    supported <- names(formals(BART::wbart))
    for (nm in c("k", "power", "base", "sigdf", "sigquant")) {
      if (nm %in% supported) wbart_args[[nm]] <- cfg[[nm]]
    }
    do.call(BART::wbart, wbart_args)
  }
  if (isTRUE(verbose)) {
    fit <- fit_call()
  } else {
    invisible(utils::capture.output(fit <- fit_call()))
  }
  t1 <- Sys.time()
  .gb_log(verbose, sprintf("[GeoBART] BART fit completed in %.1f sec\n",
                           as.numeric(t1 - t0, units = "secs")))

  n_val  <- nrow(X_val_full)
  n_test <- nrow(X_test_full)
  yhat_test_full <- fit$yhat.test                       # [ndpost, n_val + n_test]
  val_idx  <- 1:n_val
  test_idx <- (n_val + 1):(n_val + n_test)

  val_post  <- yhat_test_full[, val_idx,  drop = FALSE]   # [ndpost, n_val]
  test_post <- yhat_test_full[, test_idx, drop = FALSE]   # [ndpost, n_test]

  # Posterior mean predictions on the transformed scale.
  val_preds_t  <- colMeans(val_post)
  test_preds_t <- colMeans(test_post)

  # Posterior 90% intervals on transformed scale (later inverted).
  val_lo_t  <- apply(val_post,  2, quantile, probs = 0.05, names = FALSE)
  val_hi_t  <- apply(val_post,  2, quantile, probs = 0.95, names = FALSE)
  test_lo_t <- apply(test_post, 2, quantile, probs = 0.05, names = FALSE)
  test_hi_t <- apply(test_post, 2, quantile, probs = 0.95, names = FALSE)

  # Inverse-transform.
  val_preds  <- geobart_invert_target_transform(val_preds_t,  target_transform)
  test_preds <- geobart_invert_target_transform(test_preds_t, target_transform)
  val_lo  <- geobart_invert_target_transform(val_lo_t,  target_transform)
  val_hi  <- geobart_invert_target_transform(val_hi_t,  target_transform)
  test_lo <- geobart_invert_target_transform(test_lo_t, target_transform)
  test_hi <- geobart_invert_target_transform(test_hi_t, target_transform)

  # ---- Duan smearing correction (only if log + n_val >= threshold) ---------
  smearing_info <- if (!identical(target_transform, "log")) {
    list(S = 1.0, applied = FALSE, reason = "target != log")
  } else if (!isTRUE(cfg$duan_apply)) {
    list(S = 1.0, applied = FALSE, reason = "duan_apply = FALSE")
  } else {
    list(S = 1.0, applied = FALSE, reason = "Duan not evaluated")
  }
  if (cfg$duan_apply && identical(target_transform, "log")) {
    val_log_obs <- log(yva)
    r_log <- val_log_obs - val_preds_t
    smearing_info <- geobart_duan_smearing(
      r_log,
      trim_frac = cfg$duan_trim_frac,
      min_val_n = cfg$duan_min_val_n
    )
    .gb_log(verbose, sprintf("[GeoBART] Duan smearing: %s\n", smearing_info$reason))
    if (isTRUE(smearing_info$applied)) {
      .gb_log(verbose, sprintf("[GeoBART]   S_trim=%.5f  S_raw=%.5f  n_used=%d\n",
                               smearing_info$S_trimmed, smearing_info$S_raw,
                               smearing_info$n_used))
      val_preds  <- val_preds  * smearing_info$S
      test_preds <- test_preds * smearing_info$S
      val_lo     <- val_lo     * smearing_info$S
      val_hi     <- val_hi     * smearing_info$S
      test_lo    <- test_lo    * smearing_info$S
      test_hi    <- test_hi    * smearing_info$S
    }
  }

  # ---- Validation diagnostics ---------------------------------------------
  val_me   <- mean(val_preds - yva, na.rm = TRUE)
  val_rmse <- sqrt(mean((val_preds - yva) ^ 2, na.rm = TRUE))
  .gb_log(verbose, sprintf("\n[GeoBART] === Final summary ===\n"))
  .gb_log(verbose, sprintf("[GeoBART] Val : ME=%+.3f  RMSE=%.3f\n", val_me, val_rmse))
  .gb_log(verbose, sprintf(
    "[GeoBART] Geostat: r=%.3f  fit_q=%.2f  rho=%.3g  policy=%s  n_basis=%d  bandwidth=%.3g\n",
    cfg$nugget_ratio, cfg$fit_quality, cfg$range_major,
    cfg$spatial_policy, cfg$n_basis, cfg$bandwidth))
  .gb_log(verbose, sprintf(
    "[GeoBART] BART  : ntree=%d  ndpost=%d  nskip=%d  k=%.2f  base=%.3f  power=%.2f\n",
    cfg$ntree, cfg$ndpost, cfg$nskip, cfg$k, cfg$base, cfg$power))

  # ---- Return -------------------------------------------------------------
  # Match the shape of NMK/DKP outputs so the runners can swap models.
  list(
    pred_test       = test_preds,
    pred_val        = val_preds,
    pred_test_base  = test_preds,           # GeoBART has no separate "base"
    pred_val_base   = val_preds,
    pred_test_lo    = test_lo,
    pred_test_hi    = test_hi,
    pred_val_lo     = val_lo,
    pred_val_hi     = val_hi,
    metrics_test    = data.frame(
      N    = length(yte),
      R2   = if (length(yte) >= 2) cor(yte, test_preds) ^ 2 else NA_real_,
      RMSE = sqrt(mean((yte - test_preds) ^ 2, na.rm = TRUE)),
      MAE  = mean(abs(yte - test_preds), na.rm = TRUE),
      Bias = mean(test_preds - yte, na.rm = TRUE),
      RPIQ = if (length(yte) >= 2) IQR(yte) /
                                   (sqrt(mean((yte - test_preds) ^ 2)) + 1e-12)
             else NA_real_
    ),
    diagnostics = list(
	      target_transform = target_transform,
	      n_train          = n_train,
	      n_basis          = cfg$n_basis,
	      n_basis_effective = design$n_basis_effective,
	      bandwidth        = cfg$bandwidth,
	      spatial_policy   = cfg$spatial_policy,
	      anisotropy_ratio = cfg$anisotropy_ratio,
	      feature_blocks   = design$feature_blocks,
	      ntree            = cfg$ntree,
	      ndpost           = cfg$ndpost,
	      nskip            = cfg$nskip,
	      k                = cfg$k,
	      power            = cfg$power,
	      base             = cfg$base,
	      sigdf            = cfg$sigdf,
	      sigquant         = cfg$sigquant,
      nugget_ratio     = cfg$nugget_ratio,
      range_major      = cfg$range_major,
      fit_quality      = cfg$fit_quality,
      area             = cfg$area,
      duan_apply        = isTRUE(cfg$duan_apply),
      duan_min_val_n    = cfg$duan_min_val_n,
      duan_trim_frac    = cfg$duan_trim_frac,
      duan_applied     = isTRUE(smearing_info$applied),
      duan_S           = smearing_info$S,
      duan_S_raw       = if (is.null(smearing_info$S_raw)) NA_real_ else smearing_info$S_raw,
      duan_n_used      = if (is.null(smearing_info$n_used)) 0L else smearing_info$n_used,
      duan_reason      = smearing_info$reason,
      bart_seconds     = as.numeric(t1 - t0, units = "secs"),
      val_me           = val_me,
      val_rmse         = val_rmse
    )
  )
	}


train_geobart_seed_ensemble <- function(fd, n_seeds = 1L,
                                        train_seed = NULL,
                                        verbose = TRUE,
                                        ...) {
  n_seeds <- suppressWarnings(as.integer(n_seeds))
  if (!is.finite(n_seeds) || n_seeds < 1L) n_seeds <- 1L
  base_seed <- if (is.null(train_seed) || !is.finite(train_seed)) {
    123L
  } else {
    as.integer(train_seed)
  }
  if (n_seeds == 1L) {
    out <- train_geobart_one_fold(
      fd = fd, train_seed = base_seed, verbose = verbose, ...)
    out$diagnostics$seed_ensemble_n <- 1L
    out$diagnostics$seed_ensemble_seeds <- as.character(base_seed)
    return(out)
  }

  outs <- vector("list", n_seeds)
  seeds <- base_seed + (seq_len(n_seeds) - 1L) * 101L
  for (i in seq_len(n_seeds)) {
    .gb_log(verbose, sprintf("[GeoBART ensemble] seed %d/%d (%d)\n",
                             i, n_seeds, seeds[i]))
    outs[[i]] <- train_geobart_one_fold(
      fd = fd, train_seed = seeds[i], verbose = FALSE, ...)
  }
  avg_vec <- function(name) {
    mat <- do.call(rbind, lapply(outs, function(x) as.numeric(x[[name]])))
    colMeans(mat, na.rm = TRUE)
  }
  out <- outs[[1L]]
  for (nm in c("pred_test", "pred_val", "pred_test_base", "pred_val_base",
               "pred_test_lo", "pred_test_hi", "pred_val_lo", "pred_val_hi")) {
    if (!is.null(out[[nm]])) out[[nm]] <- avg_vec(nm)
  }
  yte <- fd$y$test
  yva <- fd$y$val
  out$metrics_test <- data.frame(
    N = length(yte),
    R2 = if (length(yte) >= 2L) stats::cor(yte, out$pred_test) ^ 2 else NA_real_,
    RMSE = sqrt(mean((yte - out$pred_test) ^ 2, na.rm = TRUE)),
    MAE = mean(abs(yte - out$pred_test), na.rm = TRUE),
    Bias = mean(out$pred_test - yte, na.rm = TRUE),
    RPIQ = if (length(yte) >= 2L) {
      stats::IQR(yte) / (sqrt(mean((yte - out$pred_test) ^ 2)) + 1e-12)
    } else {
      NA_real_
    }
  )
  val_rmse <- sqrt(mean((out$pred_val - yva) ^ 2, na.rm = TRUE))
  val_me <- mean(out$pred_val - yva, na.rm = TRUE)
  seed_val_rmse <- vapply(outs, function(x) x$diagnostics$val_rmse, numeric(1))
  seed_test_rmse <- vapply(outs, function(x) x$metrics_test$RMSE, numeric(1))
  seed_bart_seconds <- vapply(outs, function(x) x$diagnostics$bart_seconds,
                              numeric(1))
  out$diagnostics$seed_ensemble_n <- n_seeds
  out$diagnostics$seed_ensemble_seeds <- paste(seeds, collapse = "|")
  out$diagnostics$seed_ensemble_val_rmse_mean <- mean(seed_val_rmse, na.rm = TRUE)
  out$diagnostics$seed_ensemble_val_rmse_sd <- stats::sd(seed_val_rmse, na.rm = TRUE)
  out$diagnostics$seed_ensemble_test_rmse_mean <- mean(seed_test_rmse, na.rm = TRUE)
  out$diagnostics$seed_ensemble_test_rmse_sd <- stats::sd(seed_test_rmse, na.rm = TRUE)
  out$diagnostics$bart_seconds <- sum(seed_bart_seconds, na.rm = TRUE)
  out$diagnostics$val_rmse <- val_rmse
  out$diagnostics$val_me <- val_me
  out
}


# ---- Fair tuning ------------------------------------------------------------

geobart_validation_metrics <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  obs <- obs[ok]
  pred <- pred[ok]
  n <- length(obs)
  if (n == 0L) {
    return(data.frame(N = 0L, ME = NA_real_, RMSE = NA_real_,
                      MAE = NA_real_, r2 = NA_real_, MEC = NA_real_))
  }
  err <- pred - obs
  sse <- sum(err ^ 2)
  sst <- sum((obs - mean(obs)) ^ 2)
  data.frame(
    N = n,
    ME = mean(err),
    RMSE = sqrt(mean(err ^ 2)),
    MAE = mean(abs(err)),
    r2 = if (n >= 2L) suppressWarnings(stats::cor(obs, pred) ^ 2) else NA_real_,
    MEC = if (is.finite(sst) && sst > 0) 1 - sse / sst else NA_real_
  )
}

geobart_metric_score <- function(met, metric_name,
                                 rmse_sd = 0,
                                 me_abs = 0,
                                 rmse_sd_weight = 0,
                                 abs_me_weight = 0) {
  metric_name <- toupper(metric_name)
  base <- switch(metric_name,
                 RMSE = met$RMSE,
                 MAE = met$MAE,
                 MEC = -met$MEC,
                 R2 = -met$r2,
                 met$RMSE)
  if (!is.finite(base)) return(Inf)
  rmse_sd <- if (is.finite(rmse_sd)) rmse_sd else 0
  me_abs <- if (is.finite(me_abs)) me_abs else 0
  base + rmse_sd_weight * rmse_sd + abs_me_weight * me_abs
}

geobart_format_num_vec <- function(x, digits = 6L) {
  x <- as.numeric(x)
  paste(ifelse(is.finite(x), formatC(x, digits = digits, format = "fg"),
               "NA"), collapse = "|")
}

geobart_rf_predict_internal <- function(X_train, y_train, X_pred, seed = 1L) {
  # --- do-no-harm: make the in-gate RF BIT-IDENTICAL to the scored Wadoux RF ---
  # When the Wadoux helpers are in scope (they always are: the runner sources
  # wadoux2021_rf_reproduction_helpers.R before GeoBART.R), delegate to the
  # EXACT fit_rf_default_wadoux / predict_rf_default_wadoux so the paired RF the
  # gate compares against is the same object the benchmark scores. We keep the
  # original predictor column names (predList) and the Wadoux RF seed so the
  # ranger fit is reproducible-identical to the scored baseline.
  if (exists("fit_rf_default_wadoux", mode = "function") &&
      exists("predict_rf_default_wadoux", mode = "function") &&
      exists("response_name_wadoux")) {
    Xtr_df <- as.data.frame(X_train)
    Xpr_df <- as.data.frame(X_pred)
    pred_names <- colnames(Xtr_df)
    if (is.null(pred_names) || any(!nzchar(pred_names))) {
      pred_names <- paste0("x", seq_len(ncol(Xtr_df)))
      names(Xtr_df) <- pred_names
      names(Xpr_df) <- pred_names
    }
    rf_seed <- as.integer(getOption("wadoux_rf_seed", 123L))
    if (!is.finite(rf_seed)) rf_seed <- 123L
    train_df <- cbind(Xtr_df, setNames(list(as.numeric(y_train)),
                                       response_name_wadoux))
    form_rf_internal <- stats::as.formula(
      paste(response_name_wadoux, "~",
            paste(sprintf("`%s`", pred_names), collapse = "+")))
    rf_fit <- tryCatch(
      fit_rf_default_wadoux(train_df, form_rf_internal, seed = rf_seed),
      error = function(e) NULL)
    if (!is.null(rf_fit)) {
      return(as.numeric(predict_rf_default_wadoux(rf_fit, Xpr_df)))
    }
  }
  X_train <- as.data.frame(X_train)
  X_pred <- as.data.frame(X_pred)
  names(X_train) <- paste0("x", seq_len(ncol(X_train)))
  names(X_pred) <- names(X_train)
  for (nm in names(X_train)) {
    tr <- suppressWarnings(as.numeric(X_train[[nm]]))
    pr <- suppressWarnings(as.numeric(X_pred[[nm]]))
    med <- stats::median(tr[is.finite(tr)], na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    tr[!is.finite(tr)] <- med
    pr[!is.finite(pr)] <- med
    X_train[[nm]] <- tr
    X_pred[[nm]] <- pr
  }
  df <- cbind(y = as.numeric(y_train), X_train)
  if (requireNamespace("ranger", quietly = TRUE)) {
    fit <- ranger::ranger(
      y ~ .,
      data = df,
      num.trees = 500,
      mtry = max(1L, floor(sqrt(ncol(X_train)))),
      min.node.size = 5,
      seed = as.integer(seed),
      respect.unordered.factors = "order"
    )
    return(as.numeric(stats::predict(fit, data = X_pred)$predictions))
  }
  Xmat <- as.matrix(X_train)
  Xp <- as.matrix(X_pred)
  sds <- apply(Xmat, 2L, stats::sd, na.rm = TRUE)
  keep <- is.finite(sds) & sds > 0
  if (!any(keep)) return(rep(mean(y_train, na.rm = TRUE), nrow(X_pred)))
  Xsc <- scale(Xmat[, keep, drop = FALSE])
  fit <- tryCatch(
    stats::lm.fit(cbind(1, Xsc), as.numeric(y_train)),
    error = function(e) NULL
  )
  if (is.null(fit) || is.null(fit$coefficients)) {
    return(rep(mean(y_train, na.rm = TRUE), nrow(X_pred)))
  }
  Xpsc <- scale(Xp[, keep, drop = FALSE],
                center = attr(Xsc, "scaled:center"),
                scale = attr(Xsc, "scaled:scale"))
  as.numeric(cbind(1, Xpsc) %*% fit$coefficients)
}

geobart_fold_metric_summary <- function(obs, pred, fold_ids) {
  if (is.null(fold_ids) || length(fold_ids) != length(obs)) {
    fold_ids <- rep(1L, length(obs))
  }
  folds <- sort(unique(fold_ids))
  rmse <- rep(NA_real_, length(folds))
  me <- rep(NA_real_, length(folds))
  for (i in seq_along(folds)) {
    idx <- which(fold_ids == folds[i])
    met <- geobart_validation_metrics(obs[idx], pred[idx])
    rmse[i] <- met$RMSE
    me[i] <- met$ME
  }
  rmse_sd <- if (sum(is.finite(rmse)) >= 2L) stats::sd(rmse, na.rm = TRUE) else 0
  list(
    fold_RMSE = rmse,
    fold_ME = me,
    fold_RMSE_text = geobart_format_num_vec(rmse),
    fold_ME_text = geobart_format_num_vec(me),
    RMSE_sd = rmse_sd
  )
}

geobart_quantile_rmse_values <- function(obs, pred, n_bins = 5L) {
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
    geobart_validation_metrics(obs[bins == b], pred[bins == b])$RMSE
  }, numeric(1))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0L) {
    return(list(mean = 0, sd = 0, excess = 0, text = NA_character_,
                n_bins = 0L))
  }
  global <- geobart_validation_metrics(obs, pred)$RMSE
  excess <- max(vals, na.rm = TRUE) - global
  if (!is.finite(excess) || excess < 0) excess <- 0
  list(mean = mean(vals),
       sd = if (length(vals) >= 2L) stats::sd(vals) else 0,
       excess = excess,
       text = geobart_format_num_vec(vals),
       n_bins = length(vals))
}

geobart_fit_post_calibration <- function(obs, pred, policy,
                                         slope_min = 0.5,
                                         slope_max = 1.5) {
  policy <- tolower(as.character(policy)[1L])
  if (!(policy %in% c("none", "mean_bias", "median_bias", "affine"))) {
    policy <- "none"
  }
  ok <- is.finite(obs) & is.finite(pred)
  obs_ok <- obs[ok]
  pred_ok <- pred[ok]
  n_ok <- length(obs_ok)
  if (policy == "none" || n_ok == 0L) {
    return(list(policy = "none", bias = 0, intercept = 0, slope = 1,
                n = n_ok))
  }
  err <- pred_ok - obs_ok
  if (policy == "mean_bias") {
    bias <- mean(err, na.rm = TRUE)
    if (!is.finite(bias)) bias <- 0
    return(list(policy = policy, bias = bias, intercept = -bias,
                slope = 1, n = n_ok))
  }
  if (policy == "median_bias") {
    bias <- stats::median(err, na.rm = TRUE)
    if (!is.finite(bias)) bias <- 0
    return(list(policy = policy, bias = bias, intercept = -bias,
                slope = 1, n = n_ok))
  }
  if (n_ok < 3L || stats::sd(pred_ok) <= 0) {
    bias <- mean(err, na.rm = TRUE)
    if (!is.finite(bias)) bias <- 0
    return(list(policy = "mean_bias", bias = bias, intercept = -bias,
                slope = 1, n = n_ok))
  }
  fit <- tryCatch(stats::lm(obs_ok ~ pred_ok), error = function(e) NULL)
  if (is.null(fit)) {
    bias <- mean(err, na.rm = TRUE)
    if (!is.finite(bias)) bias <- 0
    return(list(policy = "mean_bias", bias = bias, intercept = -bias,
                slope = 1, n = n_ok))
  }
  cf <- stats::coef(fit)
  slope <- suppressWarnings(as.numeric(cf[2L]))
  if (!is.finite(slope)) slope <- 1
  slope <- .gb_clamp(slope, slope_min, slope_max)
  intercept <- mean(obs_ok, na.rm = TRUE) - slope * mean(pred_ok, na.rm = TRUE)
  if (!is.finite(intercept)) intercept <- 0
  cal_pred <- intercept + slope * pred_ok
  bias <- mean(cal_pred - obs_ok, na.rm = TRUE)
  if (!is.finite(bias)) bias <- 0
  list(policy = "affine", bias = bias, intercept = intercept, slope = slope,
       n = n_ok)
}

geobart_apply_post_calibration <- function(pred, cal) {
  if (is.null(cal) || is.null(cal$policy) || identical(cal$policy, "none")) {
    return(pred)
  }
  as.numeric(cal$intercept) + as.numeric(cal$slope) * as.numeric(pred)
}

geobart_select_post_calibration <- function(obs, pred, fold_ids = NULL,
                                            metric_name = "RMSE",
                                            policies = c("none"),
                                            rmse_sd_weight = 0.25,
                                            abs_me_weight = 0.10,
                                            quantile_bins = 0L,
                                            quantile_excess_weight = 0,
                                            quantile_sd_weight = 0,
                                            slope_min = 0.5,
                                            slope_max = 1.5) {
  policies <- unique(tolower(trimws(as.character(policies))))
  policies <- policies[policies %in% c("none", "mean_bias",
                                       "median_bias", "affine")]
  if (length(policies) == 0L) policies <- "none"
  if (is.null(fold_ids) || length(fold_ids) != length(obs)) {
    fold_ids <- rep(1L, length(obs))
  }
  folds <- sort(unique(fold_ids))
  policy_rank <- setNames(seq_along(c("none", "mean_bias",
                                      "median_bias", "affine")),
                          c("none", "mean_bias", "median_bias", "affine"))
  rows <- vector("list", length(policies))
  best <- NULL
  for (i in seq_along(policies)) {
    pol <- policies[i]
    pred_cal <- rep(NA_real_, length(pred))
    crossfit <- length(folds) >= 2L
    if (isTRUE(crossfit)) {
      for (fold in folds) {
        val_idx <- which(fold_ids == fold)
        train_idx <- which(fold_ids != fold)
        cal_fold <- geobart_fit_post_calibration(
          obs[train_idx], pred[train_idx], pol,
          slope_min = slope_min, slope_max = slope_max)
        pred_cal[val_idx] <- geobart_apply_post_calibration(pred[val_idx],
                                                            cal_fold)
      }
    } else {
      cal_one <- geobart_fit_post_calibration(
        obs, pred, pol, slope_min = slope_min, slope_max = slope_max)
      pred_cal <- geobart_apply_post_calibration(pred, cal_one)
    }
    final_cal <- geobart_fit_post_calibration(
      obs, pred, pol, slope_min = slope_min, slope_max = slope_max)
    met <- geobart_validation_metrics(obs, pred_cal)
    fold_met <- geobart_fold_metric_summary(obs, pred_cal, fold_ids)
    q_met <- geobart_quantile_rmse_values(obs, pred_cal, quantile_bins)
    me_abs <- abs(met$ME)
    score <- geobart_metric_score(
      met, metric_name,
      rmse_sd = fold_met$RMSE_sd,
      me_abs = me_abs,
      rmse_sd_weight = rmse_sd_weight,
      abs_me_weight = abs_me_weight
    )
    score <- score +
      quantile_excess_weight * q_met$excess +
      quantile_sd_weight * q_met$sd
    row <- data.frame(
      policy = final_cal$policy,
      requested_policy = pol,
      crossfit = crossfit,
      n = final_cal$n,
      bias = final_cal$bias,
      intercept = final_cal$intercept,
      slope = final_cal$slope,
      val_ME = met$ME,
      val_RMSE = met$RMSE,
      val_MAE = met$MAE,
      val_r2 = met$r2,
      val_MEC = met$MEC,
      RMSE_sd = fold_met$RMSE_sd,
      ME_abs = me_abs,
      quantile_RMSE_mean = q_met$mean,
      quantile_RMSE_sd = q_met$sd,
      quantile_RMSE_excess = q_met$excess,
      quantile_RMSE = q_met$text,
      quantile_bins = q_met$n_bins,
      quantile_excess_weight = quantile_excess_weight,
      quantile_sd_weight = quantile_sd_weight,
      fold_RMSE = fold_met$fold_RMSE_text,
      fold_ME = fold_met$fold_ME_text,
      selection_score = score,
      policy_rank = policy_rank[[final_cal$policy]],
      stringsAsFactors = FALSE
    )
    rows[[i]] <- row
    if (is.null(best) ||
        isTRUE(order(row$selection_score, row$policy_rank)[1L] == 1L) &&
        (row$selection_score < best$row$selection_score ||
         (isTRUE(all.equal(row$selection_score, best$row$selection_score)) &&
          row$policy_rank < best$row$policy_rank))) {
      best <- list(row = row, calibration = final_cal, pred = pred_cal)
    }
  }
  all_rows <- do.call(rbind, rows)
  all_rows <- all_rows[order(all_rows$selection_score,
                             all_rows$policy_rank), , drop = FALSE]
  best$row <- all_rows[1L, , drop = FALSE]
  best$calibration <- geobart_fit_post_calibration(
    obs, pred, best$row$requested_policy,
    slope_min = slope_min, slope_max = slope_max)
  best$calibration$requested_policy <- best$row$requested_policy
  best$calibration$crossfit <- best$row$crossfit
  best$calibration$selection_score <- best$row$selection_score
  best$calibration$RMSE_sd <- best$row$RMSE_sd
  best$calibration$ME_abs <- best$row$ME_abs
  best$calibration$quantile_RMSE_mean <- best$row$quantile_RMSE_mean
  best$calibration$quantile_RMSE_sd <- best$row$quantile_RMSE_sd
  best$calibration$quantile_RMSE_excess <- best$row$quantile_RMSE_excess
  best$calibration$quantile_RMSE <- best$row$quantile_RMSE
  best$calibration$quantile_bins <- best$row$quantile_bins
  best$calibration$quantile_excess_weight <-
    best$row$quantile_excess_weight
  best$calibration$quantile_sd_weight <- best$row$quantile_sd_weight
  best$calibration$fold_RMSE <- best$row$fold_RMSE
  best$calibration$fold_ME <- best$row$fold_ME
  best$calibration$val_ME <- best$row$val_ME
  best$calibration$val_RMSE <- best$row$val_RMSE
  best$calibration$val_MAE <- best$row$val_MAE
  best$calibration$val_r2 <- best$row$val_r2
  best$calibration$val_MEC <- best$row$val_MEC
  best$pred <- geobart_apply_post_calibration(pred, best$calibration)
  best$all_policies <- all_rows
  best
}

geobart_apply_post_calibration_out <- function(out, cal,
                                               y_val = NULL,
                                               y_test = NULL) {
  if (is.null(out) || is.null(cal)) return(out)
  for (nm in c("pred_val", "pred_test", "pred_val_base", "pred_test_base",
               "pred_val_lo", "pred_val_hi", "pred_test_lo",
               "pred_test_hi")) {
    if (!is.null(out[[nm]])) {
      raw_nm <- paste0(nm, "_raw")
      if (is.null(out[[raw_nm]])) out[[raw_nm]] <- out[[nm]]
      out[[nm]] <- geobart_apply_post_calibration(out[[nm]], cal)
    }
  }
  out$diagnostics$post_calibration_policy <- cal$policy
  out$diagnostics$post_calibration_requested_policy <-
    if (is.null(cal$requested_policy)) cal$policy else cal$requested_policy
  out$diagnostics$post_calibration_crossfit <- isTRUE(cal$crossfit)
  out$diagnostics$post_calibration_n <- cal$n
  out$diagnostics$post_calibration_bias <- cal$bias
  out$diagnostics$post_calibration_intercept <- cal$intercept
  out$diagnostics$post_calibration_slope <- cal$slope
  out$diagnostics$post_calibration_selection_score <- cal$selection_score
  out$diagnostics$post_calibration_RMSE_sd <- cal$RMSE_sd
  out$diagnostics$post_calibration_ME_abs <- cal$ME_abs
  out$diagnostics$post_calibration_quantile_RMSE_mean <-
    cal$quantile_RMSE_mean
  out$diagnostics$post_calibration_quantile_RMSE_sd <-
    cal$quantile_RMSE_sd
  out$diagnostics$post_calibration_quantile_RMSE_excess <-
    cal$quantile_RMSE_excess
  out$diagnostics$post_calibration_quantile_RMSE <- cal$quantile_RMSE
  out$diagnostics$post_calibration_quantile_bins <- cal$quantile_bins
  out$diagnostics$post_calibration_quantile_excess_weight <-
    cal$quantile_excess_weight
  out$diagnostics$post_calibration_quantile_sd_weight <-
    cal$quantile_sd_weight
  out$diagnostics$post_calibration_fold_RMSE <- cal$fold_RMSE
  out$diagnostics$post_calibration_fold_ME <- cal$fold_ME
  if (!is.null(y_val) && !is.null(out$pred_val) &&
      length(y_val) == length(out$pred_val)) {
    met_val <- geobart_validation_metrics(y_val, out$pred_val)
    out$diagnostics$val_me <- met_val$ME
    out$diagnostics$val_rmse <- met_val$RMSE
  }
  if (!is.null(y_test) && !is.null(out$pred_test) &&
      length(y_test) == length(out$pred_test)) {
    out$metrics_test <- data.frame(
      N = length(y_test),
      R2 = if (length(y_test) >= 2L) {
        suppressWarnings(stats::cor(y_test, out$pred_test) ^ 2)
      } else {
        NA_real_
      },
      RMSE = sqrt(mean((y_test - out$pred_test) ^ 2, na.rm = TRUE)),
      MAE = mean(abs(y_test - out$pred_test), na.rm = TRUE),
      Bias = mean(out$pred_test - y_test, na.rm = TRUE),
      RPIQ = if (length(y_test) >= 2L) {
        stats::IQR(y_test) /
          (sqrt(mean((y_test - out$pred_test) ^ 2, na.rm = TRUE)) + 1e-12)
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  }
  out
}

geobart_candidate_grid <- function(fd,
                                   base_params = geobart_default_params,
                                   ntree_values = NULL,
                                   ntree_mults = NULL,
                                   basis_values = NULL,
                                   basis_mults = NULL,
                                   bandwidth_values = NULL,
                                   bandwidth_mults = NULL,
                                   transforms = NULL,
                                   spatial_policies = NULL,
                                   anisotropy_ratios = NULL,
                                   k_values = NULL,
                                   power_values = NULL,
                                   base_values = NULL,
                                   sigdf_values = NULL,
                                   sigquant_values = NULL,
                                   include_no_basis = NULL,
                                   max_candidates = NULL) {
  Ctr <- as.matrix(fd$coords$train)
  ytr <- fd$y$train
  n_train <- nrow(as.matrix(fd$X$train))

  vg <- geobart_fit_variogram(ytr, Ctr)
  cfg <- geobart_auto_config(vg, n_train, Ctr)
  cfg <- geobart_apply_config_overrides(cfg, base_params, coords_train = Ctr)

  if (is.null(basis_mults)) {
    basis_mults <- .gb_env_num_vec("GEOBART_TUNE_BASIS_MULTS",
                                   c(0.5, 1, 1.5, 2, 2.5))
  }
  if (is.null(bandwidth_mults)) {
    bandwidth_mults <- .gb_env_num_vec("GEOBART_TUNE_BANDWIDTH_MULTS",
                                       c(0.25, 0.5, 1, 2, 4))
  }
  if (is.null(transforms)) {
    transforms <- .gb_env_char_vec("GEOBART_TUNE_TRANSFORMS",
                                   c("auto", "identity", "log"))
  }
  transforms <- unique(tolower(transforms))
  transforms <- transforms[transforms %in% c("auto", "identity", "log")]
  if (any(ytr <= 0, na.rm = TRUE)) transforms <- setdiff(transforms, "log")
  if (length(transforms) == 0L) transforms <- "auto"

  if (is.null(spatial_policies)) {
    spatial_policies <- .gb_env_char_vec("GEOBART_TUNE_SPATIAL_POLICIES",
                                         c("coords_basis"))
  }
  spatial_policies <- unique(vapply(spatial_policies,
                                    .gb_normalize_spatial_policy,
                                    character(1)))

  base_seed <- .gb_param(base_params, "train_seed", NULL)
  if (!is.null(base_seed)) set.seed(as.integer(base_seed))
  auto_transform <- geobart_target_transform_auto(ytr)

  include_no_basis <- if (is.null(include_no_basis)) {
    toupper(Sys.getenv("GEOBART_TUNE_INCLUDE_NO_BASIS", unset = "FALSE")) %in%
      c("TRUE", "1", "YES", "Y")
  } else {
    isTRUE(include_no_basis)
  }

  ntree_min <- as.integer(.gb_env_num("GEOBART_TUNE_NTREE_MIN", 50L))
  ntree_max <- as.integer(.gb_env_num("GEOBART_TUNE_NTREE_MAX",
                                      max(300L, cfg$ntree)))
  if (!is.null(ntree_values)) {
    ntree_values <- as.integer(ntree_values)
  } else if (nzchar(Sys.getenv("GEOBART_TUNE_NTREE_VALUES", unset = ""))) {
    ntree_values <- as.integer(.gb_env_num_vec("GEOBART_TUNE_NTREE_VALUES",
                                               c(50, 60, 90, 120, 180)))
  } else if (!is.null(ntree_mults) ||
             nzchar(Sys.getenv("GEOBART_TUNE_NTREE_MULTS", unset = ""))) {
    if (is.null(ntree_mults)) {
      ntree_mults <- .gb_env_num_vec("GEOBART_TUNE_NTREE_MULTS", c(1, 1.5, 2))
    }
    ntree_values <- as.integer(round(cfg$ntree * ntree_mults))
  } else {
    ntree_values <- c(50L, 60L, 90L, 120L, 180L)
  }
  ntree_values <- unique(as.integer(.gb_clamp(ntree_values, ntree_min,
                                              max(ntree_min, ntree_max))))
  ntree_values <- sort(unique(c(cfg$ntree, ntree_values)))

  max_basis_available <- geobart_cap_basis_count(Ctr, nrow(unique(Ctr)))
  basis_max_default <- max(80L, round(cfg$n_basis * 2.5), cfg$n_basis,
                           na.rm = TRUE)
  basis_max <- as.integer(.gb_env_num("GEOBART_TUNE_N_BASIS_MAX",
                                      basis_max_default))
  basis_max <- min(max_basis_available, max(1L, basis_max))
  if (!is.null(basis_values)) {
    n_basis_values <- as.integer(basis_values)
  } else if (nzchar(Sys.getenv("GEOBART_TUNE_N_BASIS_VALUES", unset = ""))) {
    n_basis_values <- as.integer(.gb_env_num_vec("GEOBART_TUNE_N_BASIS_VALUES",
                                                 cfg$n_basis))
  } else {
    n_basis_values <- as.integer(round(cfg$n_basis * basis_mults))
    n_basis_values <- c(n_basis_values, 80L)
  }
  n_basis_values <- n_basis_values[is.finite(n_basis_values)]
  if (length(n_basis_values) == 0L) n_basis_values <- cfg$n_basis
  n_basis_values <- pmax(1L, n_basis_values)
  n_basis_values <- unique(as.integer(.gb_clamp(n_basis_values, 1L, basis_max)))
  n_basis_values <- sort(unique(c(cfg$n_basis, n_basis_values)))
  if (include_no_basis) n_basis_values <- sort(unique(c(0L, n_basis_values)))

  if (!is.null(bandwidth_values)) {
    bandwidth_values <- as.numeric(bandwidth_values)
  } else if (nzchar(Sys.getenv("GEOBART_TUNE_BANDWIDTH_VALUES", unset = ""))) {
    bandwidth_values <- .gb_env_num_vec("GEOBART_TUNE_BANDWIDTH_VALUES",
                                        cfg$bandwidth)
  } else {
    bandwidth_values <- cfg$bandwidth * bandwidth_mults
  }
  bandwidth_values <- bandwidth_values[is.finite(bandwidth_values) &
                                         bandwidth_values > 0]
  if (length(bandwidth_values) == 0L) bandwidth_values <- cfg$bandwidth
  bandwidth_values <- sort(unique(signif(c(cfg$bandwidth, bandwidth_values), 10)))

  if (is.null(anisotropy_ratios)) {
    anisotropy_ratios <- .gb_env_num_vec("GEOBART_TUNE_ANISOTROPY_RATIOS",
                                         cfg$anisotropy_ratio)
  }
  anisotropy_ratios <- sort(unique(.gb_clamp(
    anisotropy_ratios[is.finite(anisotropy_ratios)], 1, 20)))
  if (length(anisotropy_ratios) == 0L) anisotropy_ratios <- cfg$anisotropy_ratio

  read_prior_values <- function(values, env_name, default) {
    if (!is.null(values)) return(as.numeric(values))
    raw <- Sys.getenv(env_name, unset = "")
    if (nzchar(raw)) return(.gb_env_num_vec(env_name, default))
    default
  }
  k_values <- read_prior_values(k_values, "GEOBART_TUNE_K_VALUES", cfg$k)
  power_values <- read_prior_values(power_values, "GEOBART_TUNE_POWER_VALUES", cfg$power)
  base_values <- read_prior_values(base_values, "GEOBART_TUNE_BASE_VALUES", cfg$base)
  sigdf_values <- read_prior_values(sigdf_values, "GEOBART_TUNE_SIGDF_VALUES", cfg$sigdf)
  sigquant_values <- read_prior_values(sigquant_values, "GEOBART_TUNE_SIGQUANT_VALUES", cfg$sigquant)
  k_values <- sort(unique(.gb_clamp(k_values[is.finite(k_values)], 0.1, 20)))
  power_values <- sort(unique(.gb_clamp(power_values[is.finite(power_values)], 0.1, 20)))
  base_values <- sort(unique(.gb_clamp(base_values[is.finite(base_values)], 0.01, 0.999)))
  sigdf_values <- sort(unique(.gb_clamp(sigdf_values[is.finite(sigdf_values)], 1, 100)))
  sigquant_values <- sort(unique(.gb_clamp(sigquant_values[is.finite(sigquant_values)], 0.01, 0.999)))
  if (length(k_values) == 0L) k_values <- cfg$k
  if (length(power_values) == 0L) power_values <- cfg$power
  if (length(base_values) == 0L) base_values <- cfg$base
  if (length(sigdf_values) == 0L) sigdf_values <- cfg$sigdf
  if (length(sigquant_values) == 0L) sigquant_values <- cfg$sigquant

  base_transform <- tolower(as.character(.gb_param(base_params,
                                                   "target_transform", "auto")))
  if (!(base_transform %in% c("auto", "identity", "log"))) {
    base_transform <- "auto"
  }
  if (identical(base_transform, "log") && any(ytr <= 0, na.rm = TRUE)) {
    warning("GeoBART tuning: requested log base transform but y has non-positive values; using identity.")
    base_transform <- "identity"
  }

  base_candidate <- data.frame(
    target_transform_requested = base_transform,
    target_transform_effective = if (identical(base_transform, "auto"))
      auto_transform else base_transform,
	    ntree = as.integer(cfg$ntree),
	    n_basis = as.integer(cfg$n_basis),
	    bandwidth = as.numeric(cfg$bandwidth),
	    spatial_policy = cfg$spatial_policy,
	    anisotropy_ratio = as.numeric(cfg$anisotropy_ratio),
	    k = as.numeric(cfg$k),
	    power = as.numeric(cfg$power),
	    base = as.numeric(cfg$base),
	    sigdf = as.numeric(cfg$sigdf),
	    sigquant = as.numeric(cfg$sigquant),
	    is_auto_base = TRUE,
	    stringsAsFactors = FALSE
	  )
  grid <- expand.grid(
    target_transform_requested = transforms,
	    ntree = ntree_values,
	    n_basis = n_basis_values,
	    bandwidth = bandwidth_values,
	    spatial_policy = spatial_policies,
	    anisotropy_ratio = anisotropy_ratios,
	    k = k_values,
	    power = power_values,
	    base = base_values,
	    sigdf = sigdf_values,
	    sigquant = sigquant_values,
	    KEEP.OUT.ATTRS = FALSE,
	    stringsAsFactors = FALSE
	  )
  grid$target_transform_effective <- ifelse(
    grid$target_transform_requested == "auto",
    auto_transform,
    grid$target_transform_requested
  )
	  grid$is_auto_base <- FALSE
	  grid <- grid[, names(base_candidate), drop = FALSE]
  uses_basis <- vapply(grid$spatial_policy, .gb_policy_uses_basis, logical(1))
  uses_aniso <- vapply(grid$spatial_policy, .gb_policy_uses_aniso, logical(1))
  grid$n_basis[!uses_basis] <- 0L
  grid$bandwidth[!uses_basis] <- NA_real_
  grid$anisotropy_ratio[!uses_aniso] <- 1
	  candidates <- rbind(base_candidate, grid)
	  key_bw <- ifelse(candidates$n_basis == 0L, NA_real_,
	                   signif(candidates$bandwidth, 10))
  key_aniso <- ifelse(vapply(candidates$spatial_policy, .gb_policy_uses_aniso,
                             logical(1)),
                      signif(candidates$anisotropy_ratio, 6), NA_real_)
	  key <- paste(candidates$target_transform_effective, candidates$ntree,
	               candidates$n_basis, key_bw, candidates$spatial_policy,
                 key_aniso, candidates$k, candidates$power, candidates$base,
                 candidates$sigdf, candidates$sigquant, sep = "|")
  candidates <- candidates[!duplicated(key), , drop = FALSE]

  if (is.null(max_candidates)) {
    raw_max <- Sys.getenv("GEOBART_TUNE_MAX_CANDIDATES", unset = "")
    if (nzchar(raw_max)) {
      max_candidates <- suppressWarnings(as.integer(raw_max))
    }
  }
	  if (!is.null(max_candidates) && is.finite(max_candidates) &&
	      max_candidates > 0L && nrow(candidates) > max_candidates) {
	    base_rows <- candidates[candidates$is_auto_base, , drop = FALSE]
	    rest <- candidates[!candidates$is_auto_base, , drop = FALSE]
    bw_dist <- ifelse(is.finite(rest$bandwidth) & is.finite(cfg$bandwidth) &
                        rest$bandwidth > 0 & cfg$bandwidth > 0,
                      abs(log(rest$bandwidth / cfg$bandwidth)), 0)
	    rest$distance_to_base <-
	      abs(log(pmax(rest$ntree, 1) / pmax(cfg$ntree, 1))) +
	      abs(log(pmax(rest$n_basis, 1) / pmax(cfg$n_basis, 1))) +
	      bw_dist +
	      ifelse(rest$target_transform_effective ==
	               if (identical(base_transform, "auto")) auto_transform else base_transform,
	             0, 0.25) +
      ifelse(rest$spatial_policy == cfg$spatial_policy, 0, 0.15)
	    keep_rest <- max(0L, max_candidates - nrow(base_rows))
    if (keep_rest > 0L && nrow(rest) > keep_rest) {
      rest$.row_id <- seq_len(nrow(rest))
      rest$.cap_group <- interaction(rest$spatial_policy,
                                     rest$target_transform_effective,
                                     drop = TRUE)
      groups <- split(rest, rest$.cap_group)
      quota <- max(1L, floor(keep_rest / max(1L, length(groups))))
      selected <- lapply(groups, function(g) {
        g <- g[order(g$distance_to_base, g$.row_id), , drop = FALSE]
        g[seq_len(min(quota, nrow(g))), , drop = FALSE]
      })
      selected <- do.call(rbind, selected)
      if (nrow(selected) > keep_rest) {
        selected <- selected[order(selected$distance_to_base,
                                   selected$.row_id), , drop = FALSE]
        selected <- selected[seq_len(keep_rest), , drop = FALSE]
      }
      remaining <- rest[!(rest$.row_id %in% selected$.row_id), , drop = FALSE]
      remaining <- remaining[order(remaining$distance_to_base,
                                   remaining$.row_id), , drop = FALSE]
      fill_n <- keep_rest - nrow(selected)
      if (fill_n > 0L && nrow(remaining) > 0L) {
        selected <- rbind(
          selected,
          remaining[seq_len(min(fill_n, nrow(remaining))), , drop = FALSE]
        )
      }
      rest <- selected
      rest$.cap_group <- NULL
      rest$.row_id <- NULL
    } else {
      rest <- rest[order(rest$distance_to_base),
                   , drop = FALSE]
      rest <- rest[seq_len(min(keep_rest, nrow(rest))), , drop = FALSE]
    }
    rest$distance_to_base <- NULL
    rest <- rest[, names(base_rows), drop = FALSE]
	    candidates <- rbind(base_rows, rest)
	  }

  candidates$candidate_id <- seq_len(nrow(candidates))
  candidates$target_transform <- candidates$target_transform_effective
	  candidates$candidate_label <- sprintf(
	    "tt=%s;req=%s;pol=%s;aniso=%.2f;ntree=%d;basis=%d;bw=%.6g;k=%.2f;base=%.2f;power=%.2f;sigdf=%.0f;sigq=%.2f",
	    candidates$target_transform_effective,
	    candidates$target_transform_requested,
      candidates$spatial_policy,
      candidates$anisotropy_ratio,
	    candidates$ntree,
	    candidates$n_basis, candidates$bandwidth,
      candidates$k, candidates$base, candidates$power,
      candidates$sigdf, candidates$sigquant
	  )
	  candidates[, c("candidate_id", "candidate_label", "is_auto_base",
	                 "target_transform_requested", "target_transform_effective",
	                 "target_transform", "spatial_policy", "anisotropy_ratio",
	                 "ntree", "n_basis", "bandwidth", "k", "power", "base",
                   "sigdf", "sigquant"),
	             drop = FALSE]
	}

geobart_relabel_candidates <- function(candidates) {
  if (is.null(candidates) || nrow(candidates) == 0L) return(candidates)
  key_bw <- ifelse(candidates$n_basis == 0L, NA_real_,
                   signif(candidates$bandwidth, 10))
  key_aniso <- ifelse(
    vapply(candidates$spatial_policy, .gb_policy_uses_aniso, logical(1)),
    signif(candidates$anisotropy_ratio, 6), NA_real_)
  key <- paste(candidates$target_transform_effective, candidates$ntree,
               candidates$n_basis, key_bw, candidates$spatial_policy,
               key_aniso, candidates$k, candidates$power, candidates$base,
               candidates$sigdf, candidates$sigquant, sep = "|")
  candidates <- candidates[!duplicated(key), , drop = FALSE]
  candidates$candidate_id <- seq_len(nrow(candidates))
  candidates$target_transform <- candidates$target_transform_effective
  candidates$candidate_label <- sprintf(
    "tt=%s;req=%s;pol=%s;aniso=%.2f;ntree=%d;basis=%d;bw=%.6g;k=%.2f;base=%.2f;power=%.2f;sigdf=%.0f;sigq=%.2f",
    candidates$target_transform_effective,
    candidates$target_transform_requested,
    candidates$spatial_policy,
    candidates$anisotropy_ratio,
    candidates$ntree,
    candidates$n_basis,
    candidates$bandwidth,
    candidates$k,
    candidates$base,
    candidates$power,
    candidates$sigdf,
    candidates$sigquant
  )
  candidates
}

geobart_stage2_candidate_grid <- function(stage1_scores,
                                          top_k = 8L,
                                          max_candidates = 240L) {
  if (is.null(stage1_scores) || nrow(stage1_scores) == 0L) return(data.frame())
  ok <- is.finite(stage1_scores$selection_score) &
    (is.na(stage1_scores$error) | stage1_scores$error == "")
  top <- stage1_scores[ok, , drop = FALSE]
  if (nrow(top) == 0L) return(data.frame())
  top$stage1_rank <- rank(top$selection_score, ties.method = "first")
  top <- top[order(top$selection_score, top$candidate_id), , drop = FALSE]
  top <- top[seq_len(min(as.integer(top_k), nrow(top))), , drop = FALSE]

  k_values <- .gb_env_num_vec("GEOBART_TUNE_STAGE2_K_VALUES", c(1, 1.5, 2, 3))
  base_values <- .gb_env_num_vec("GEOBART_TUNE_STAGE2_BASE_VALUES", c(0.85, 0.95, 0.99))
  power_values <- .gb_env_num_vec("GEOBART_TUNE_STAGE2_POWER_VALUES", c(1.5, 2, 2.5))
  sigdf_values <- .gb_env_num_vec("GEOBART_TUNE_STAGE2_SIGDF_VALUES", c(3, 5, 10))
  sigquant_values <- .gb_env_num_vec("GEOBART_TUNE_STAGE2_SIGQUANT_VALUES", c(0.75, 0.90, 0.99))
  ntree_mults <- .gb_env_num_vec("GEOBART_TUNE_STAGE2_NTREE_MULTS", c(0.75, 1, 1.25))
  basis_mults <- .gb_env_num_vec("GEOBART_TUNE_STAGE2_BASIS_MULTS", c(0.75, 1, 1.25))
  bandwidth_mults <- .gb_env_num_vec("GEOBART_TUNE_STAGE2_BANDWIDTH_MULTS",
                                     c(0.5, 0.75, 1, 1.5, 2))

  rows <- vector("list", nrow(top))
  for (i in seq_len(nrow(top))) {
    x <- top[i, , drop = FALSE]
    uses_basis <- .gb_policy_uses_basis(x$spatial_policy)
    ntree_vals <- unique(as.integer(.gb_clamp(
      round(as.numeric(x$ntree) * ntree_mults), 25L, 5000L)))
    basis_vals <- if (uses_basis) {
      unique(as.integer(.gb_clamp(
        round(as.numeric(x$n_basis) * basis_mults), 1L, 500L)))
    } else {
      0L
    }
    bw_vals <- if (uses_basis && is.finite(x$bandwidth)) {
      unique(signif(as.numeric(x$bandwidth) * bandwidth_mults, 10))
    } else {
      NA_real_
    }
    aniso_vals <- if (.gb_policy_uses_aniso(x$spatial_policy)) {
      unique(c(as.numeric(x$anisotropy_ratio),
               .gb_env_num_vec("GEOBART_TUNE_STAGE2_ANISOTROPY_RATIOS",
                               c(1.5, 2, 3))))
    } else {
      1
    }
    rows[[i]] <- expand.grid(
      target_transform_requested = x$target_transform_requested,
      target_transform_effective = x$target_transform_effective,
      target_transform = x$target_transform_effective,
      spatial_policy = x$spatial_policy,
      anisotropy_ratio = aniso_vals,
      ntree = ntree_vals,
      n_basis = basis_vals,
      bandwidth = bw_vals,
      k = k_values,
      power = power_values,
      base = base_values,
      sigdf = sigdf_values,
      sigquant = sigquant_values,
      source_stage1_candidate_id = x$candidate_id,
      source_stage1_rank = x$stage1_rank,
      is_auto_base = FALSE,
      KEEP.OUT.ATTRS = FALSE,
      stringsAsFactors = FALSE
    )
  }
  out <- do.call(rbind, rows)
  out$anisotropy_ratio <- .gb_clamp(as.numeric(out$anisotropy_ratio), 1, 20)
  out$k <- .gb_clamp(as.numeric(out$k), 0.1, 20)
  out$power <- .gb_clamp(as.numeric(out$power), 0.1, 20)
  out$base <- .gb_clamp(as.numeric(out$base), 0.01, 0.999)
  out$sigdf <- .gb_clamp(as.numeric(out$sigdf), 1, 100)
  out$sigquant <- .gb_clamp(as.numeric(out$sigquant), 0.01, 0.999)
  out <- geobart_relabel_candidates(out)

  if (is.finite(max_candidates) && max_candidates > 0L &&
      nrow(out) > max_candidates) {
    out$local_distance <-
      abs(log(pmax(out$ntree, 1) /
                pmax(out$ntree[match(out$source_stage1_candidate_id,
                                      out$source_stage1_candidate_id)], 1)))
    must_keep <- !duplicated(out$source_stage1_candidate_id)
    rest <- out[!must_keep, , drop = FALSE]
    rest <- rest[order(rest$source_stage1_rank, rest$local_distance,
                       rest$candidate_id), , drop = FALSE]
    keep_n <- max(0L, as.integer(max_candidates) - sum(must_keep))
    out <- rbind(out[must_keep, , drop = FALSE],
                 rest[seq_len(min(keep_n, nrow(rest))), , drop = FALSE])
    out$local_distance <- NULL
    out <- geobart_relabel_candidates(out)
  }
  out
}

tune_geobart_one_fold <- function(fd,
	  base_params = geobart_default_params,
	  candidate_grid = NULL,
		                                  selection_metric = "RMSE",
		                                  selection_objective =
		                                    Sys.getenv("GEOBART_SELECTION_OBJECTIVE",
		                                               unset = "paired_rf_delta"),
		                                  max_candidates = NULL,
	                                  inner_folds = 1L,
	                                  num_workers = 1L,
                                    autotune_v2 = .gb_env_bool("GEOBART_AUTOTUNE_V2", FALSE),
                                    autotune_v3 = .gb_env_bool("GEOBART_AUTOTUNE_V3", autotune_v2),
	                                  verbose = TRUE) {
  if (is.null(base_params)) base_params <- list()
  if (is.null(base_params$target_transform)) {
    base_params$target_transform <- "auto"
  }
  inner_folds <- suppressWarnings(as.integer(inner_folds))
  if (!is.finite(inner_folds) || inner_folds < 1L) inner_folds <- 1L
  num_workers <- suppressWarnings(as.integer(num_workers))
  if (!is.finite(num_workers) || num_workers < 1L) num_workers <- 1L
  autotune_v3 <- isTRUE(autotune_v3)
  autotune_v4 <- .gb_env_bool("GEOBART_AUTOTUNE_V4", autotune_v3)
  autotune_v4 <- isTRUE(autotune_v4)
  if (isTRUE(autotune_v4)) autotune_v3 <- TRUE
  if (isTRUE(autotune_v3)) autotune_v2 <- TRUE
  v3_rmse_sd_weight <- .gb_env_num("GEOBART_V3_RMSE_SD_WEIGHT", 0.25)
  if (!is.finite(v3_rmse_sd_weight) || v3_rmse_sd_weight < 0) {
    v3_rmse_sd_weight <- 0
  }
  v3_abs_me_weight <- .gb_env_num("GEOBART_V3_ABS_ME_WEIGHT", 0.10)
  if (!is.finite(v3_abs_me_weight) || v3_abs_me_weight < 0) {
    v3_abs_me_weight <- 0
  }
  v3_calibration_policies <- .gb_env_char_vec(
    "GEOBART_V3_CALIBRATION_POLICIES",
    c("none", "mean_bias", "median_bias", "affine"))
  v3_slope_min <- .gb_env_num("GEOBART_V3_AFFINE_SLOPE_MIN", 0.5)
  v3_slope_max <- .gb_env_num("GEOBART_V3_AFFINE_SLOPE_MAX", 1.5)
  if (!is.finite(v3_slope_min)) v3_slope_min <- 0.5
  if (!is.finite(v3_slope_max)) v3_slope_max <- 1.5
  if (v3_slope_max < v3_slope_min) {
    tmp <- v3_slope_min
    v3_slope_min <- v3_slope_max
    v3_slope_max <- tmp
  }
  v3_stage2_seed_ensemble_n <- as.integer(.gb_env_num(
    "GEOBART_V3_STAGE2_N_SEEDS", 3L))
  if (!is.finite(v3_stage2_seed_ensemble_n) ||
      v3_stage2_seed_ensemble_n < 1L) {
    v3_stage2_seed_ensemble_n <- 1L
  }
  v3_stage2_select_only <- .gb_env_bool("GEOBART_V3_STAGE2_SELECT_ONLY", FALSE)
  v3_log_margin <- .gb_env_num("GEOBART_V3_LOG_MARGIN", 0.50)
  if (!is.finite(v3_log_margin) || v3_log_margin < 0) v3_log_margin <- 0
  v4_quantile_bins <- as.integer(.gb_env_num("GEOBART_V4_QUANTILE_BINS", 5L))
  if (!is.finite(v4_quantile_bins) || v4_quantile_bins < 2L) {
    v4_quantile_bins <- 0L
  }
  v4_quantile_excess_weight <- .gb_env_num(
    "GEOBART_V4_QUANTILE_EXCESS_WEIGHT", 0.15)
  if (!is.finite(v4_quantile_excess_weight) ||
      v4_quantile_excess_weight < 0) {
    v4_quantile_excess_weight <- 0
  }
  v4_quantile_sd_weight <- .gb_env_num("GEOBART_V4_QUANTILE_SD_WEIGHT", 0.05)
  if (!is.finite(v4_quantile_sd_weight) || v4_quantile_sd_weight < 0) {
    v4_quantile_sd_weight <- 0
  }
  staged_full_grid <- .gb_env_bool("GEOBART_TUNE_FULL_GRID", FALSE)
  stage2_full_grid <- .gb_env_bool("GEOBART_V3_STAGE2_FULL_GRID", FALSE)

  pool <- NULL
  grid_fd <- fd
  fold_ids <- NULL
  if (inner_folds > 1L) {
    pool <- .gb_fd_pool_train_val(fd)
    inner_folds <- min(inner_folds, max(1L, floor(length(pool$y) / 2L)))
    grid_fd <- fd
    grid_fd$X$train <- pool$X
    grid_fd$coords$train <- pool$coords
    grid_fd$y$train <- pool$y
    fold_ids <- .gb_fold_ids(
      pool$y, inner_folds,
      seed = .gb_param(base_params, "train_seed", 123L)
    )
    inner_folds <- length(unique(fold_ids))
  }

  cap_fold_basis <- function(candidate_grid) {
    if (is.null(candidate_grid) || nrow(candidate_grid) == 0L) {
      return(candidate_grid)
    }
	  if (inner_folds > 1L) {
	    fold_basis_cap <- vapply(sort(unique(fold_ids)), function(fold) {
	      train_coords <- pool$coords[fold_ids != fold, , drop = FALSE]
	      geobart_cap_basis_count(train_coords, .Machine$integer.max)
    }, integer(1))
    fold_basis_cap <- min(fold_basis_cap, na.rm = TRUE)
    if (is.finite(fold_basis_cap) && fold_basis_cap >= 0L) {
      positive_basis <- candidate_grid$n_basis > 0L
      candidate_grid$n_basis[positive_basis] <- pmin(
        as.integer(candidate_grid$n_basis[positive_basis]),
        as.integer(fold_basis_cap)
      )
	      key_bw <- ifelse(candidate_grid$n_basis == 0L, NA_real_,
	                       signif(candidate_grid$bandwidth, 10))
      key_aniso <- ifelse(
        vapply(candidate_grid$spatial_policy, .gb_policy_uses_aniso, logical(1)),
        signif(candidate_grid$anisotropy_ratio, 6), NA_real_)
	      key <- paste(candidate_grid$target_transform_effective,
	                   candidate_grid$ntree, candidate_grid$n_basis,
	                   key_bw, candidate_grid$spatial_policy, key_aniso,
                     candidate_grid$k, candidate_grid$power,
                     candidate_grid$base, candidate_grid$sigdf,
                     candidate_grid$sigquant, sep = "|")
	      candidate_grid <- candidate_grid[!duplicated(key), , drop = FALSE]
	      candidate_grid$candidate_id <- seq_len(nrow(candidate_grid))
		      candidate_grid$target_transform <- candidate_grid$target_transform_effective
		      candidate_grid$candidate_label <- sprintf(
		        "tt=%s;req=%s;pol=%s;aniso=%.2f;ntree=%d;basis=%d;bw=%.6g;k=%.2f;base=%.2f;power=%.2f;sigdf=%.0f;sigq=%.2f",
		        candidate_grid$target_transform_effective,
		        candidate_grid$target_transform_requested,
          candidate_grid$spatial_policy,
          candidate_grid$anisotropy_ratio,
		        candidate_grid$ntree,
	        candidate_grid$n_basis, candidate_grid$bandwidth,
          candidate_grid$k, candidate_grid$base, candidate_grid$power,
          candidate_grid$sigdf, candidate_grid$sigquant
	      )
	    }
	  }
    candidate_grid
  }

	  if (is.null(candidate_grid)) {
      if (isTRUE(autotune_v2)) {
        stage1_max <- .gb_env_num("GEOBART_TUNE_STAGE1_MAX_CANDIDATES", 240L)
        if (isTRUE(staged_full_grid)) stage1_max <- Inf
        if (!is.finite(stage1_max) || stage1_max <= 0) stage1_max <- Inf
        candidate_grid <- geobart_candidate_grid(
          fd = grid_fd,
          base_params = base_params,
          ntree_values = as.integer(.gb_env_num_vec(
            "GEOBART_TUNE_STAGE1_NTREE_VALUES", c(50, 90, 180))),
          basis_mults = .gb_env_num_vec(
            "GEOBART_TUNE_STAGE1_BASIS_MULTS", c(0.5, 1, 2.5)),
          bandwidth_mults = .gb_env_num_vec(
            "GEOBART_TUNE_STAGE1_BANDWIDTH_MULTS", c(0.5, 1, 2, 4)),
          spatial_policies = .gb_env_char_vec(
            "GEOBART_TUNE_STAGE1_SPATIAL_POLICIES",
            c("coords_basis", "coords_only", "basis_only",
              "tabular_only", "coords_aniso_basis")),
          anisotropy_ratios = .gb_env_num_vec(
            "GEOBART_TUNE_STAGE1_ANISOTROPY_RATIOS", c(1.5, 2.5)),
          k_values = .gb_param(base_params, "k", .gb_env_num("GEOBART_K", 2)),
          power_values = .gb_param(base_params, "power", .gb_env_num("GEOBART_POWER", 2)),
          base_values = .gb_param(base_params, "base", .gb_env_num("GEOBART_BASE", 0.95)),
          sigdf_values = .gb_param(base_params, "sigdf", .gb_env_num("GEOBART_SIGDF", 3)),
          sigquant_values = .gb_param(base_params, "sigquant", .gb_env_num("GEOBART_SIGQUANT", 0.90)),
          max_candidates = stage1_max
        )
      } else {
	      candidate_grid <- geobart_candidate_grid(
	        fd = grid_fd,
	        base_params = base_params,
	        max_candidates = max_candidates
	      )
      }
	  } else {
      autotune_v2 <- FALSE
    }
  candidate_grid <- cap_fold_basis(candidate_grid)
	  if (nrow(candidate_grid) == 0L) {
	    stop("GeoBART tuning produced no candidates.")
	  }

	  metric_name <- toupper(selection_metric)
	  if (!(metric_name %in% c("RMSE", "MAE", "MEC", "R2"))) {
	    stop("Unsupported GeoBART selection metric: ", selection_metric)
	  }
	  selection_objective <- tolower(trimws(as.character(selection_objective)[1L]))
	  if (!(selection_objective %in% c("rmse", "paired_rf_delta"))) {
	    selection_objective <- "rmse"
	  }
	  noharm_policy <- .gb_env_bool(
	    "GEOBART_NOHARM_POLICY",
	    identical(selection_objective, "paired_rf_delta"))
	  noharm_margin_abs <- .gb_env_num("GEOBART_NOHARM_MARGIN_ABS", 0.05)
	  if (!is.finite(noharm_margin_abs) || noharm_margin_abs < 0) {
	    noharm_margin_abs <- 0
	  }
	  noharm_margin_rel <- .gb_env_num("GEOBART_NOHARM_MARGIN_REL", 0.0)
	  if (!is.finite(noharm_margin_rel) || noharm_margin_rel < 0) {
	    noharm_margin_rel <- 0
	  }
		  noharm_restrict_ensemble <- .gb_env_bool(
		    "GEOBART_NOHARM_RESTRICT_ENSEMBLE", TRUE)
		  rf_noharm_policy <- .gb_env_bool(
		    "GEOBART_RF_NOHARM_POLICY",
		    identical(selection_objective, "paired_rf_delta"))
		  rf_noharm_margin_abs <- .gb_env_num(
		    "GEOBART_RF_NOHARM_MARGIN_ABS", 0.05)
		  if (!is.finite(rf_noharm_margin_abs) || rf_noharm_margin_abs < 0) {
		    rf_noharm_margin_abs <- 0
		  }
		  rf_noharm_margin_rel <- .gb_env_num(
		    "GEOBART_RF_NOHARM_MARGIN_REL", 0.0)
		  if (!is.finite(rf_noharm_margin_rel) || rf_noharm_margin_rel < 0) {
		    rf_noharm_margin_rel <- 0
		  }
		  rf_noharm_restrict_ensemble <- .gb_env_bool(
		    "GEOBART_RF_NOHARM_RESTRICT_ENSEMBLE", TRUE)
		  objective_label <- if (identical(selection_objective, "paired_rf_delta")) {
		    "PAIRED_RF_DELTA_RMSE"
		  } else {
	    metric_name
	  }
	  rf_tune_rmse <- NA_real_
	  rf_tune_fold_rmse <- NA_character_
	  if (identical(selection_objective, "paired_rf_delta")) {
	    obs_rf <- if (inner_folds <= 1L) fd$y$val else pool$y
	    rf_pred <- rep(NA_real_, length(obs_rf))
	    if (inner_folds <= 1L) {
	      rf_pred <- geobart_rf_predict_internal(
	        fd$X$train, fd$y$train, fd$X$val,
	        seed = as.integer(.gb_param(base_params, "train_seed", 123L)) + 41000L
	      )
	      rf_tune_fold_rmse <- geobart_format_num_vec(
	        geobart_validation_metrics(fd$y$val, rf_pred)$RMSE)
	    } else {
	      rf_fold_rmse <- rep(NA_real_, length(sort(unique(fold_ids))))
	      fold_counter <- 0L
	      for (fold in sort(unique(fold_ids))) {
	        fold_counter <- fold_counter + 1L
	        val_idx <- which(fold_ids == fold)
	        train_idx <- which(fold_ids != fold)
	        pred_fold <- geobart_rf_predict_internal(
	          pool$X[train_idx, , drop = FALSE],
	          pool$y[train_idx],
	          pool$X[val_idx, , drop = FALSE],
	          seed = as.integer(.gb_param(base_params, "train_seed", 123L)) +
	            41000L + fold_counter
	        )
	        rf_pred[val_idx] <- pred_fold
	        rf_fold_rmse[fold_counter] <- geobart_validation_metrics(
	          pool$y[val_idx], pred_fold)$RMSE
	      }
	      rf_tune_fold_rmse <- geobart_format_num_vec(rf_fold_rmse)
	    }
	    rf_tune_rmse <- geobart_validation_metrics(obs_rf, rf_pred)$RMSE
	    .gb_log(verbose, sprintf(
	      "[GeoBART tuning] paired-RF selection baseline RMSE=%.4f\n",
	      rf_tune_rmse
	    ))
	  }
		  train_arg_names <- setdiff(names(formals(train_geobart_one_fold)),
		                             c("fd", "...", "verbose"))
	  eval_candidate <- function(i, candidate_grid, tune_stage = "single",
                               stage_seed_ensemble_n = 1L) {
	    cand <- candidate_grid[i, , drop = FALSE]
	    .gb_log(verbose, sprintf(
	      "[GeoBART tuning] %s candidate %d/%d: %s\n",
	      tune_stage, i, nrow(candidate_grid), cand$candidate_label
	    ))

    candidate_params <- base_params
    requested_transform <- if ("target_transform_requested" %in% names(cand)) {
      cand$target_transform_requested
    } else {
      cand$target_transform
    }
    effective_transform <- if ("target_transform_effective" %in% names(cand)) {
      cand$target_transform_effective
    } else {
      cand$target_transform
    }

    candidate_params$target_transform <- effective_transform
	    candidate_params$ntree <- as.integer(cand$ntree)
	    candidate_params$n_basis <- as.integer(cand$n_basis)
	    candidate_params$bandwidth <- as.numeric(cand$bandwidth)
	    candidate_params$spatial_policy <- as.character(cand$spatial_policy)
	    candidate_params$anisotropy_ratio <- as.numeric(cand$anisotropy_ratio)
	    candidate_params$k <- as.numeric(cand$k)
	    candidate_params$power <- as.numeric(cand$power)
	    candidate_params$base <- as.numeric(cand$base)
	    candidate_params$sigdf <- as.numeric(cand$sigdf)
	    candidate_params$sigquant <- as.numeric(cand$sigquant)
	    candidate_params$duan_apply <- FALSE
    candidate_params <- candidate_params[
      intersect(names(candidate_params), train_arg_names)
    ]

    t0 <- Sys.time()
    out <- NULL
    preds_val <- NULL
    folds_completed <- 0L
    if (inner_folds <= 1L) {
      seed_use <- .gb_param(candidate_params, "train_seed",
                            .gb_param(base_params, "train_seed", 123L))
      call_params <- candidate_params
      call_params$train_seed <- NULL
      out <- tryCatch(
        do.call(train_geobart_seed_ensemble,
                c(list(fd = fd,
                       n_seeds = stage_seed_ensemble_n,
                       train_seed = seed_use,
                       verbose = FALSE), call_params)),
        error = function(e) e
      )
      if (!inherits(out, "error")) {
        preds_val <- out$pred_val
        folds_completed <- 1L
      }
    } else {
      preds_val <- rep(NA_real_, length(pool$y))
      fold_errors <- character()
      for (fold in sort(unique(fold_ids))) {
        val_idx <- which(fold_ids == fold)
        train_idx <- which(fold_ids != fold)
        cv_fd <- .gb_fd_from_pool(pool, train_idx, val_idx)
        fold_params <- candidate_params
        if (!is.null(fold_params$train_seed)) {
          fold_params$train_seed <- as.integer(fold_params$train_seed) + fold
        }
        seed_use <- .gb_param(fold_params, "train_seed",
                              .gb_param(base_params, "train_seed", 123L))
        fold_params$train_seed <- NULL
        fold_out <- tryCatch(
          do.call(train_geobart_seed_ensemble,
                  c(list(fd = cv_fd,
                         n_seeds = stage_seed_ensemble_n,
                         train_seed = seed_use,
                         verbose = FALSE), fold_params)),
          error = function(e) e
        )
        if (inherits(fold_out, "error")) {
          fold_errors <- c(fold_errors, conditionMessage(fold_out))
          next
        }
        preds_val[val_idx] <- fold_out$pred_val
        folds_completed <- folds_completed + 1L
      }
      if (length(fold_errors) > 0L || any(!is.finite(preds_val))) {
        msg <- paste(unique(fold_errors[nzchar(fold_errors)]), collapse = " | ")
        if (!nzchar(msg)) msg <- "one or more inner-fold predictions were non-finite"
        out <- simpleError(msg)
      }
    }
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")

    if (inherits(out, "error")) {
      row <- data.frame(
	        candidate_id = cand$candidate_id,
	        candidate_label = cand$candidate_label,
	        tune_stage = tune_stage,
	        is_auto_base = cand$is_auto_base,
        target_transform_requested = requested_transform,
        target_transform_effective = effective_transform,
	        ntree = cand$ntree,
	        n_basis = cand$n_basis,
	        n_basis_effective = NA_integer_,
	        bandwidth = cand$bandwidth,
	        spatial_policy = cand$spatial_policy,
	        anisotropy_ratio = cand$anisotropy_ratio,
	        k = cand$k,
	        power = cand$power,
	        base = cand$base,
	        sigdf = cand$sigdf,
	        sigquant = cand$sigquant,
	        ndpost = NA_integer_,
        nskip = NA_integer_,
        duan_apply = FALSE,
        val_N = length(fd$y$val),
        val_ME = NA_real_,
        val_RMSE = NA_real_,
        val_MAE = NA_real_,
        val_r2 = NA_real_,
        val_MEC = NA_real_,
        raw_val_ME = NA_real_,
        raw_val_RMSE = NA_real_,
	        raw_val_MAE = NA_real_,
	        raw_val_r2 = NA_real_,
	        raw_val_MEC = NA_real_,
	        rf_inner_RMSE = rf_tune_rmse,
	        rf_inner_fold_RMSE = rf_tune_fold_rmse,
	        delta_RMSE_vs_RF = NA_real_,
	        RMSE_sd = NA_real_,
        ME_abs = NA_real_,
        fold_RMSE = NA_character_,
        fold_ME = NA_character_,
        quantile_RMSE_mean = NA_real_,
        quantile_RMSE_sd = NA_real_,
        quantile_RMSE_excess = NA_real_,
        quantile_RMSE = NA_character_,
        quantile_bins = NA_integer_,
        post_calibration_policy = NA_character_,
        post_calibration_requested_policy = NA_character_,
        post_calibration_crossfit = NA,
        post_calibration_n = NA_integer_,
        post_calibration_bias = NA_real_,
        post_calibration_intercept = NA_real_,
        post_calibration_slope = NA_real_,
	        selection_objective = selection_objective,
	        selection_metric = objective_label,
	        selection_score = Inf,
        selected = FALSE,
        inner_folds = inner_folds,
        folds_completed = folds_completed,
        autotune_v4 = autotune_v4,
        autotune_v3 = autotune_v3,
        stage_seed_ensemble_n = stage_seed_ensemble_n,
        rmse_sd_weight = if (isTRUE(autotune_v3)) v3_rmse_sd_weight else 0,
        abs_me_weight = if (isTRUE(autotune_v3)) v3_abs_me_weight else 0,
        quantile_excess_weight =
          if (isTRUE(autotune_v4)) v4_quantile_excess_weight else 0,
        quantile_sd_weight =
          if (isTRUE(autotune_v4)) v4_quantile_sd_weight else 0,
        num_workers = num_workers,
        fit_seconds = elapsed,
        error = conditionMessage(out),
        stringsAsFactors = FALSE
      )
      .gb_log(verbose, sprintf("[GeoBART tuning]   failed: %s\n",
                               conditionMessage(out)))
      return(list(row = row, best = NULL))
    }

    obs_eval <- if (inner_folds <= 1L) fd$y$val else pool$y
    folds_eval <- if (inner_folds <= 1L) rep(1L, length(obs_eval)) else fold_ids
    raw_met <- geobart_validation_metrics(obs_eval, preds_val)
    raw_fold_met <- geobart_fold_metric_summary(obs_eval, preds_val, folds_eval)
    post_sel <- NULL
    if (isTRUE(autotune_v3)) {
      post_sel <- geobart_select_post_calibration(
        obs_eval, preds_val, folds_eval,
        metric_name = metric_name,
        policies = v3_calibration_policies,
        rmse_sd_weight = v3_rmse_sd_weight,
        abs_me_weight = v3_abs_me_weight,
        quantile_bins = if (isTRUE(autotune_v4)) v4_quantile_bins else 0L,
        quantile_excess_weight = if (isTRUE(autotune_v4)) {
          v4_quantile_excess_weight
        } else {
          0
        },
        quantile_sd_weight = if (isTRUE(autotune_v4)) {
          v4_quantile_sd_weight
        } else {
          0
        },
        slope_min = v3_slope_min,
        slope_max = v3_slope_max
      )
      met <- data.frame(
        N = raw_met$N,
        ME = post_sel$row$val_ME,
        RMSE = post_sel$row$val_RMSE,
        MAE = post_sel$row$val_MAE,
        r2 = post_sel$row$val_r2,
        MEC = post_sel$row$val_MEC
      )
      fold_met <- list(
        RMSE_sd = post_sel$row$RMSE_sd,
        fold_RMSE_text = post_sel$row$fold_RMSE,
        fold_ME_text = post_sel$row$fold_ME
      )
      score <- post_sel$row$selection_score
    } else {
      met <- raw_met
      fold_met <- raw_fold_met
      score <- geobart_metric_score(met, metric_name)
    }
    q_met <- if (!is.null(post_sel)) {
      list(mean = post_sel$row$quantile_RMSE_mean,
           sd = post_sel$row$quantile_RMSE_sd,
           excess = post_sel$row$quantile_RMSE_excess,
           text = post_sel$row$quantile_RMSE,
           n_bins = post_sel$row$quantile_bins)
    } else {
      geobart_quantile_rmse_values(obs_eval, preds_val,
                                   if (isTRUE(autotune_v4)) {
                                     v4_quantile_bins
                                   } else {
                                     0L
                                   })
    }
	    if (isTRUE(autotune_v4) && is.null(post_sel)) {
	      score <- score +
	        v4_quantile_excess_weight * q_met$excess +
	        v4_quantile_sd_weight * q_met$sd
	    }
	    delta_rmse_vs_rf <- if (is.finite(rf_tune_rmse)) {
	      met$RMSE - rf_tune_rmse
	    } else NA_real_
	    if (identical(selection_objective, "paired_rf_delta") &&
	        is.finite(delta_rmse_vs_rf)) {
	      score <- delta_rmse_vs_rf
	      if (isTRUE(autotune_v3)) {
	        score <- score +
	          v3_rmse_sd_weight * ifelse(is.finite(fold_met$RMSE_sd),
	                                     fold_met$RMSE_sd, 0) +
	          v3_abs_me_weight * abs(met$ME)
	      }
	      if (isTRUE(autotune_v4)) {
	        score <- score +
	          v4_quantile_excess_weight * ifelse(is.finite(q_met$excess),
	                                             q_met$excess, 0) +
	          v4_quantile_sd_weight * ifelse(is.finite(q_met$sd),
	                                         q_met$sd, 0)
	      }
	    }
	    if (!is.finite(score)) score <- Inf
	    complexity <- as.numeric(cand$ntree) *
        max(1, as.numeric(cand$n_basis) +
              ifelse(.gb_policy_uses_coords(cand$spatial_policy), 2, 0))
    row <- data.frame(
	      candidate_id = cand$candidate_id,
	      candidate_label = cand$candidate_label,
	      tune_stage = tune_stage,
	      is_auto_base = cand$is_auto_base,
      target_transform_requested = requested_transform,
      target_transform_effective = effective_transform,
	      ntree = cand$ntree,
	      n_basis = cand$n_basis,
	      n_basis_effective = if (inner_folds <= 1L) {
          out$diagnostics$n_basis_effective
        } else {
          ifelse(.gb_policy_uses_basis(cand$spatial_policy),
                 as.integer(cand$n_basis), 0L)
        },
	      bandwidth = cand$bandwidth,
	      spatial_policy = cand$spatial_policy,
	      anisotropy_ratio = cand$anisotropy_ratio,
	      k = cand$k,
	      power = cand$power,
	      base = cand$base,
	      sigdf = cand$sigdf,
	      sigquant = cand$sigquant,
	      ndpost = if (inner_folds <= 1L) {
        out$diagnostics$ndpost
      } else {
        as.integer(.gb_param(candidate_params, "ndpost",
                             .gb_env_num("GEOBART_NDPOST", 1000L)))
      },
      nskip = if (inner_folds <= 1L) {
        out$diagnostics$nskip
      } else {
        as.integer(.gb_param(candidate_params, "nskip",
                             .gb_env_num("GEOBART_NSKIP", 100L)))
      },
      duan_apply = FALSE,
      val_N = met$N,
      val_ME = met$ME,
      val_RMSE = met$RMSE,
      val_MAE = met$MAE,
      val_r2 = met$r2,
      val_MEC = met$MEC,
      raw_val_ME = raw_met$ME,
      raw_val_RMSE = raw_met$RMSE,
	      raw_val_MAE = raw_met$MAE,
	      raw_val_r2 = raw_met$r2,
	      raw_val_MEC = raw_met$MEC,
	      rf_inner_RMSE = rf_tune_rmse,
	      rf_inner_fold_RMSE = rf_tune_fold_rmse,
	      delta_RMSE_vs_RF = delta_rmse_vs_rf,
	      RMSE_sd = fold_met$RMSE_sd,
      ME_abs = abs(met$ME),
      quantile_RMSE_mean = q_met$mean,
      quantile_RMSE_sd = q_met$sd,
      quantile_RMSE_excess = q_met$excess,
      quantile_RMSE = q_met$text,
      quantile_bins = q_met$n_bins,
      fold_RMSE = fold_met$fold_RMSE_text,
      fold_ME = fold_met$fold_ME_text,
      post_calibration_policy = if (is.null(post_sel)) {
        "none"
      } else {
        post_sel$calibration$policy
      },
      post_calibration_requested_policy = if (is.null(post_sel)) {
        "none"
      } else {
        post_sel$calibration$requested_policy
      },
      post_calibration_crossfit = if (is.null(post_sel)) {
        FALSE
      } else {
        isTRUE(post_sel$calibration$crossfit)
      },
      post_calibration_n = if (is.null(post_sel)) {
        raw_met$N
      } else {
        post_sel$calibration$n
      },
      post_calibration_bias = if (is.null(post_sel)) {
        0
      } else {
        post_sel$calibration$bias
      },
      post_calibration_intercept = if (is.null(post_sel)) {
        0
      } else {
        post_sel$calibration$intercept
      },
      post_calibration_slope = if (is.null(post_sel)) {
        1
      } else {
        post_sel$calibration$slope
      },
	      selection_objective = selection_objective,
	      selection_metric = objective_label,
	      selection_score = score,
      selected = FALSE,
      inner_folds = inner_folds,
      folds_completed = folds_completed,
      autotune_v4 = autotune_v4,
      autotune_v3 = autotune_v3,
      stage_seed_ensemble_n = stage_seed_ensemble_n,
      rmse_sd_weight = if (isTRUE(autotune_v3)) v3_rmse_sd_weight else 0,
      abs_me_weight = if (isTRUE(autotune_v3)) v3_abs_me_weight else 0,
      quantile_excess_weight =
        if (isTRUE(autotune_v4)) v4_quantile_excess_weight else 0,
      quantile_sd_weight =
        if (isTRUE(autotune_v4)) v4_quantile_sd_weight else 0,
      num_workers = num_workers,
      fit_seconds = elapsed,
      error = NA_character_,
      stringsAsFactors = FALSE
    )

    list(
      row = row,
	      best = list(
	        out = out,
	        candidate = cand,
	        params = candidate_params,
	        effective_transform = effective_transform,
	        tune_stage = tune_stage,
	        score = score,
	        complexity = complexity,
          post_calibration = if (is.null(post_sel)) {
            geobart_fit_post_calibration(obs_eval, preds_val, "none")
          } else {
            post_sel$calibration
          },
	          raw_score = geobart_metric_score(raw_met, metric_name),
	          raw_metrics = raw_met,
	          rf_inner_RMSE = rf_tune_rmse,
	          rf_inner_fold_RMSE = rf_tune_fold_rmse,
	          delta_RMSE_vs_RF = delta_rmse_vs_rf,
	          stable_RMSE_sd = fold_met$RMSE_sd,
          stable_ME_abs = abs(met$ME),
          quantile_RMSE_mean = q_met$mean,
          quantile_RMSE_sd = q_met$sd,
          quantile_RMSE_excess = q_met$excess,
          quantile_RMSE = q_met$text,
          quantile_bins = q_met$n_bins,
          fold_RMSE = fold_met$fold_RMSE_text,
          fold_ME = fold_met$fold_ME_text
	      )
    )
  }

	  evaluate_grid <- function(grid, tune_stage, stage_seed_ensemble_n = 1L) {
    grid <- cap_fold_basis(grid)
    if (is.null(grid) || nrow(grid) == 0L) {
      return(list(scores = data.frame(), successful = list()))
    }
    .gb_log(verbose, sprintf(
      "[GeoBART tuning] Evaluating %d %s candidates by validation %s (inner_folds=%d, workers=%d)\n",
      nrow(grid), tune_stage, metric_name, inner_folds, num_workers
    ))
    idx <- seq_len(nrow(grid))
    can_parallel <- num_workers > 1L &&
      .Platform$OS.type != "windows" &&
      requireNamespace("parallel", quietly = TRUE)
    eval_results <- if (isTRUE(can_parallel)) {
      parallel::mclapply(
        idx, eval_candidate,
        candidate_grid = grid,
        tune_stage = tune_stage,
        stage_seed_ensemble_n = stage_seed_ensemble_n,
        mc.cores = min(num_workers, length(idx)),
        mc.preschedule = FALSE,
        mc.set.seed = FALSE
      )
    } else {
      if (num_workers > 1L && .Platform$OS.type == "windows") {
        .gb_log(verbose, "[GeoBART tuning] parallel fork unavailable on Windows; using workers=1\n")
      }
      lapply(idx, eval_candidate, candidate_grid = grid,
             tune_stage = tune_stage,
             stage_seed_ensemble_n = stage_seed_ensemble_n)
    }
    rows <- lapply(eval_results, `[[`, "row")
    scores <- do.call(rbind, rows)
    successful <- eval_results[!vapply(eval_results, function(x) is.null(x$best),
                                       logical(1))]
	    list(scores = scores, successful = successful)
	  }

	  geobart_entry_key <- function(entry) {
	    paste(entry$best$tune_stage,
	          as.integer(entry$best$candidate$candidate_id),
	          sep = "|")
	  }
	  geobart_unique_entries <- function(entries) {
	    entries <- entries[!vapply(entries, is.null, logical(1))]
	    entries <- entries[vapply(entries, function(x) {
	      !is.null(x$best) && is.finite(x$best$score)
	    }, logical(1))]
	    if (length(entries) == 0L) return(list())
	    keys <- vapply(entries, geobart_entry_key, character(1))
	    entries[!duplicated(keys)]
	  }
	  geobart_conservative_entries <- function(entries) {
	    entries <- geobart_unique_entries(entries)
	    if (length(entries) == 0L) return(list())
	    is_auto <- vapply(entries, function(x) {
	      isTRUE(x$best$candidate$is_auto_base[1])
	    }, logical(1))
	    transform <- vapply(entries, function(x) {
	      as.character(x$best$effective_transform)
	    }, character(1))
	    policy <- vapply(entries, function(x) {
	      as.character(x$best$candidate$spatial_policy[1])
	    }, character(1))
	    non_log <- transform %in% c("identity", "auto")
	    tabular <- policy %in% c("tabular_only")
	    tiers <- list(
	      which(is_auto & non_log),
	      which(tabular & non_log),
	      which(is_auto),
	      which(tabular),
	      which(non_log)
	    )
	    for (idx in tiers) {
	      if (length(idx) > 0L) return(entries[idx])
	    }
	    entries
	  }
	  geobart_order_entries <- function(entries) {
	    entries <- geobart_unique_entries(entries)
	    if (length(entries) == 0L) return(integer())
	    order(
	      vapply(entries, function(x) x$best$score, numeric(1)),
	      vapply(entries, function(x) x$best$complexity, numeric(1)),
	      vapply(entries, function(x) as.integer(x$best$candidate$candidate_id),
	             integer(1))
	    )
	  }

	  stage1_name <- if (isTRUE(autotune_v2)) "stage1_screen" else "single"
  stage1 <- evaluate_grid(candidate_grid, stage1_name)
  scores <- stage1$scores
  successful <- stage1$successful
  stage2_successful <- list()
  if (isTRUE(autotune_v2) && length(successful) > 0L) {
    stage2_top_k <- as.integer(.gb_env_num("GEOBART_TUNE_STAGE2_TOP_K", 8L))
    if (!is.finite(stage2_top_k) || stage2_top_k < 1L) stage2_top_k <- 8L
    stage2_max <- .gb_env_num("GEOBART_TUNE_STAGE2_MAX_CANDIDATES", 240L)
    if (isTRUE(stage2_full_grid)) stage2_max <- Inf
    if (!is.finite(stage2_max) || stage2_max <= 0) stage2_max <- Inf
    stage2_grid <- geobart_stage2_candidate_grid(
      stage1$scores,
      top_k = stage2_top_k,
      max_candidates = stage2_max
    )
    if (!is.null(stage2_grid) && nrow(stage2_grid) > 0L) {
      stage2 <- evaluate_grid(
        stage2_grid, "stage2_refine",
        stage_seed_ensemble_n = if (isTRUE(autotune_v3)) {
          v3_stage2_seed_ensemble_n
        } else {
          1L
        })
      if (nrow(stage2$scores) > 0L) {
        scores <- rbind(scores, stage2$scores)
      }
      stage2_successful <- stage2$successful
      successful <- c(successful, stage2_successful)
    }
  }

	  if (length(successful) == 0L) {
	    err_idx <- which(nzchar(as.character(scores$error)))[1L]
    first_error <- if (length(err_idx) && is.finite(err_idx)) {
      scores$error[err_idx]
    } else {
      "no candidate returned finite validation predictions"
    }
    stop("GeoBART tuning failed for all candidates. First error: ",
         first_error)
  }
  selection_pool <- successful
  if (isTRUE(autotune_v3) && isTRUE(v3_stage2_select_only) &&
      length(stage2_successful) > 0L) {
    selection_pool <- stage2_successful
  }
  best_order <- order(
    vapply(selection_pool, function(x) x$best$score, numeric(1)),
    vapply(selection_pool, function(x) x$best$complexity, numeric(1)),
    vapply(selection_pool, function(x) as.integer(x$best$candidate$candidate_id),
           integer(1))
	  )
	  best <- selection_pool[[best_order[1L]]]$best
	  if (isTRUE(autotune_v3) && best$effective_transform == "log" &&
	      v3_log_margin > 0) {
    identity_pool <- selection_pool[vapply(selection_pool, function(x) {
      identical(as.character(x$best$effective_transform), "identity")
    }, logical(1))]
    if (length(identity_pool) > 0L) {
      id_order <- order(
        vapply(identity_pool, function(x) x$best$score, numeric(1)),
        vapply(identity_pool, function(x) x$best$complexity, numeric(1)),
        vapply(identity_pool, function(x) as.integer(x$best$candidate$candidate_id),
               integer(1))
      )
      id_best <- identity_pool[[id_order[1L]]]$best
      if (is.finite(id_best$score) &&
          id_best$score <= best$score + v3_log_margin) {
	        best <- id_best
	      }
	    }
	  }
	  pre_noharm_best <- best
	  anchor_entries <- geobart_conservative_entries(successful)
	  anchor_order <- geobart_order_entries(anchor_entries)
	  anchor_entry <- if (length(anchor_order) > 0L) {
	    anchor_entries[[anchor_order[1L]]]
	  } else {
	    NULL
	  }
	  noharm_anchor_score <- NA_real_
	  noharm_anchor_val_rmse <- NA_real_
	  noharm_anchor_label <- NA_character_
	  noharm_margin <- NA_real_
	  noharm_delta_vs_anchor <- NA_real_
	  noharm_applied <- FALSE
	  noharm_reason <- if (isTRUE(noharm_policy)) {
	    "no_anchor_available"
	  } else {
	    "disabled"
	  }
	  if (isTRUE(noharm_policy) && !is.null(anchor_entry)) {
	    anchor_best <- anchor_entry$best
	    noharm_anchor_score <- as.numeric(anchor_best$score)
	    noharm_anchor_val_rmse <- as.numeric(anchor_best$raw_metrics$RMSE)
	    noharm_anchor_label <- as.character(anchor_best$candidate$candidate_label)
	    anchor_scale <- noharm_anchor_val_rmse
	    if (!is.finite(anchor_scale)) anchor_scale <- abs(noharm_anchor_score)
	    if (!is.finite(anchor_scale)) anchor_scale <- 0
	    noharm_margin <- noharm_margin_abs +
	      noharm_margin_rel * abs(anchor_scale)
	    selected_key <- paste(best$tune_stage,
	                          as.integer(best$candidate$candidate_id),
	                          sep = "|")
	    anchor_key <- geobart_entry_key(anchor_entry)
	    if (!identical(selected_key, anchor_key)) {
	      noharm_delta_vs_anchor <- noharm_anchor_score - as.numeric(best$score)
	      if (!is.finite(noharm_delta_vs_anchor) ||
	          noharm_delta_vs_anchor < noharm_margin) {
	        best <- anchor_best
	        noharm_applied <- TRUE
	        noharm_reason <- sprintf(
	          "fallback_to_anchor:improvement=%.6f<margin=%.6f",
	          ifelse(is.finite(noharm_delta_vs_anchor),
	                 noharm_delta_vs_anchor, NA_real_),
	          noharm_margin)
	      } else {
	        noharm_reason <- sprintf(
	          "aggressive_passed:improvement=%.6f>=margin=%.6f",
	          noharm_delta_vs_anchor, noharm_margin)
	      }
	    } else {
	      noharm_delta_vs_anchor <- 0
	      noharm_reason <- "selected_candidate_is_anchor"
	    }
	  }
		  if (isTRUE(noharm_applied)) {
		    selection_pool <- geobart_unique_entries(c(list(anchor_entry), selection_pool))
		    if (isTRUE(noharm_restrict_ensemble)) {
		      selection_pool <- geobart_unique_entries(
		        c(list(anchor_entry), geobart_conservative_entries(successful)))
		    }
		  }
		  # --- LITERAL-RF fallback entry (do-no-harm hard floor) -------------------
		  # Construct a synthetic selection entry whose held-out prediction IS the
		  # literal Wadoux RF (predict_rf_default_wadoux), so that when no GeoBART
		  # candidate beats RF on OOF the gate can revert to RF itself rather than to
		  # a same-family GeoBART anchor. The RF is fit on fd$X$train (the exact
		  # predList columns) with the Wadoux seed and predicts val+test.
		  geobart_literal_rf_entry <- function() {
		    if (!(exists("fit_rf_default_wadoux", mode = "function") &&
		          exists("predict_rf_default_wadoux", mode = "function") &&
		          exists("response_name_wadoux"))) {
		      return(NULL)
		    }
		    tryCatch({
		      Xtr <- as.data.frame(fd$X$train)
		      Xva <- as.data.frame(fd$X$val)
		      Xte <- as.data.frame(fd$X$test)
		      pred_names <- colnames(Xtr)
		      rf_seed <- as.integer(getOption("wadoux_rf_seed", 123L))
		      if (!is.finite(rf_seed)) rf_seed <- 123L
		      tr_df <- cbind(Xtr, setNames(list(as.numeric(fd$y$train)),
		                                   response_name_wadoux))
		      form_lit <- stats::as.formula(
		        paste(response_name_wadoux, "~",
		              paste(sprintf("`%s`", pred_names), collapse = "+")))
		      rf_fit <- fit_rf_default_wadoux(tr_df, form_lit, seed = rf_seed)
		      pred_val <- as.numeric(predict_rf_default_wadoux(rf_fit, Xva))
		      pred_test <- as.numeric(predict_rf_default_wadoux(rf_fit, Xte))
		      lit_out <- list(
		        pred_test = pred_test, pred_val = pred_val,
		        pred_test_base = pred_test, pred_val_base = pred_val,
		        pred_test_lo = pred_test, pred_test_hi = pred_test,
		        pred_val_lo = pred_val, pred_val_hi = pred_val,
		        metrics_test = data.frame(
		          N = length(fd$y$test),
		          R2 = if (length(fd$y$test) >= 2L)
		            suppressWarnings(stats::cor(fd$y$test, pred_test)^2) else NA_real_,
		          RMSE = sqrt(mean((fd$y$test - pred_test)^2, na.rm = TRUE)),
		          MAE = mean(abs(fd$y$test - pred_test), na.rm = TRUE),
		          Bias = mean(pred_test - fd$y$test, na.rm = TRUE)),
		        diagnostics = list(model = "literal_wadoux_rf",
		                           target_transform = "identity",
		                           spatial_policy = "tabular_only"))
		      rf_val_rmse <- sqrt(mean((fd$y$val - pred_val)^2, na.rm = TRUE))
		      list(best = list(
		        out = lit_out,
		        candidate = data.frame(candidate_id = -1L,
		                               candidate_label = "literal_wadoux_rf",
		                               spatial_policy = "tabular_only",
		                               ntree = NA_integer_, n_basis = 0L,
		                               bandwidth = NA_real_, anisotropy_ratio = NA_real_,
		                               k = NA_real_, power = NA_real_, base = NA_real_,
		                               sigdf = NA_real_, sigquant = NA_real_,
		                               is_auto_base = FALSE,
		                               stringsAsFactors = FALSE),
		        params = list(),
		        tune_stage = "literal_rf",
		        effective_transform = "identity",
		        score = rf_val_rmse,
		        complexity = 0,
		        raw_score = rf_val_rmse,
		        raw_metrics = data.frame(RMSE = rf_val_rmse, ME = NA_real_),
		        rf_inner_RMSE = rf_tune_rmse,
		        rf_inner_fold_RMSE = rf_tune_fold_rmse,
		        delta_RMSE_vs_RF = 0,
		        post_calibration = NULL))
		    }, error = function(e) NULL)
		  }
		  literal_rf_entry <- geobart_literal_rf_entry()

		  pre_rf_noharm_best <- best
		  rf_noharm_preselected_delta <- as.numeric(
		    pre_rf_noharm_best$delta_RMSE_vs_RF)
		  rf_noharm_anchor_delta <- if (!is.null(anchor_entry)) {
		    as.numeric(anchor_entry$best$delta_RMSE_vs_RF)
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
		    rf_scale <- as.numeric(pre_rf_noharm_best$rf_inner_RMSE)
		    if (!is.finite(rf_scale)) rf_scale <- rf_tune_rmse
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
		        # Prefer reverting to the LITERAL Wadoux RF (guaranteed >= RF tie);
		        # fall back to the family anchor only if the literal RF is unavailable.
		        revert_entry <- if (!is.null(literal_rf_entry)) literal_rf_entry else anchor_entry
		        if (!is.null(revert_entry)) {
		          selected_key <- paste(best$tune_stage,
		                                as.integer(best$candidate$candidate_id),
		                                sep = "|")
		          anchor_key <- geobart_entry_key(revert_entry)
		          if (!identical(selected_key, anchor_key)) {
		            best <- revert_entry$best
		            rf_noharm_applied <- TRUE
		            rf_noharm_reason <- sprintf(
		              "fallback_to_%s:delta=%.6f>required=%.6f",
		              if (!is.null(literal_rf_entry)) "literal_rf" else "anchor",
		              rf_noharm_preselected_delta, rf_noharm_required_delta)
		          } else {
		            rf_noharm_reason <- sprintf(
		              "selected_anchor_still_fails_rf:delta=%.6f>required=%.6f",
		              rf_noharm_preselected_delta, rf_noharm_required_delta)
		          }
		          if (isTRUE(rf_noharm_restrict_ensemble)) {
		            selection_pool <- geobart_unique_entries(
		              c(list(revert_entry), geobart_conservative_entries(successful)))
		          } else {
		            selection_pool <- geobart_unique_entries(
		              c(list(revert_entry), selection_pool))
		          }
		        } else if (!is.null(literal_rf_entry)) {
		          # No family anchor, but literal RF is available: still revert to RF
		          # so we are GUARANTEED >= RF even when every candidate failed.
		          best <- literal_rf_entry$best
		          rf_noharm_applied <- TRUE
		          rf_noharm_reason <- sprintf(
		            "fallback_to_literal_rf_no_anchor:delta=%.6f>required=%.6f",
		            rf_noharm_preselected_delta, rf_noharm_required_delta)
		          selection_pool <- geobart_unique_entries(
		            c(list(literal_rf_entry), selection_pool))
		        } else {
		          rf_noharm_reason <- sprintf(
		            "rf_inner_failed_no_anchor:delta=%.6f>required=%.6f",
		            rf_noharm_preselected_delta, rf_noharm_required_delta)
		        }
		      }
		    } else {
		      rf_noharm_reason <- "no_valid_delta_RMSE_vs_RF"
		    }
		  }
			  best_score <- best$score
		  scores$noharm_policy <- noharm_policy
	  scores$noharm_anchor_candidate_id <- if (!is.null(anchor_entry)) {
	    as.integer(anchor_entry$best$candidate$candidate_id)
	  } else {
	    NA_integer_
	  }
	  scores$noharm_anchor_tune_stage <- if (!is.null(anchor_entry)) {
	    as.character(anchor_entry$best$tune_stage)
	  } else {
	    NA_character_
	  }
	  scores$noharm_anchor_candidate_label <- noharm_anchor_label
	  scores$noharm_anchor_score <- noharm_anchor_score
	  scores$noharm_anchor_val_rmse <- noharm_anchor_val_rmse
	  scores$noharm_margin_abs <- noharm_margin_abs
	  scores$noharm_margin_rel <- noharm_margin_rel
	  scores$noharm_margin <- noharm_margin
	  scores$noharm_delta_vs_anchor <- noharm_delta_vs_anchor
		  scores$noharm_applied <- noharm_applied
		  scores$noharm_restrict_ensemble <- noharm_restrict_ensemble
		  scores$noharm_reason <- noharm_reason
		  scores$rf_noharm_policy <- rf_noharm_policy
		  scores$rf_noharm_preselected_candidate_id <-
		    as.integer(pre_rf_noharm_best$candidate$candidate_id)
		  scores$rf_noharm_preselected_tune_stage <-
		    as.character(pre_rf_noharm_best$tune_stage)
		  scores$rf_noharm_preselected_delta_RMSE_vs_RF <-
		    rf_noharm_preselected_delta
		  scores$rf_noharm_anchor_candidate_id <- if (!is.null(anchor_entry)) {
		    as.integer(anchor_entry$best$candidate$candidate_id)
		  } else {
		    NA_integer_
		  }
		  scores$rf_noharm_anchor_tune_stage <- if (!is.null(anchor_entry)) {
		    as.character(anchor_entry$best$tune_stage)
		  } else {
		    NA_character_
		  }
		  scores$rf_noharm_anchor_candidate_label <- noharm_anchor_label
		  scores$rf_noharm_anchor_delta_RMSE_vs_RF <- rf_noharm_anchor_delta
		  scores$rf_noharm_margin_abs <- rf_noharm_margin_abs
		  scores$rf_noharm_margin_rel <- rf_noharm_margin_rel
		  scores$rf_noharm_margin <- rf_noharm_margin
		  scores$rf_noharm_required_delta_RMSE_vs_RF <- rf_noharm_required_delta
		  scores$rf_noharm_failed <- rf_noharm_failed
		  scores$rf_noharm_applied <- rf_noharm_applied
		  scores$rf_noharm_restrict_ensemble <- rf_noharm_restrict_ensemble
		  scores$rf_noharm_reason <- rf_noharm_reason
			  scores$selected <- scores$candidate_id == best$candidate$candidate_id &
	      scores$tune_stage == best$tune_stage
	  best_order <- geobart_order_entries(selection_pool)
	  ranked_best <- lapply(selection_pool[best_order], function(x) x$best)
  best_key <- paste(best$tune_stage, as.integer(best$candidate$candidate_id),
                    sep = "|")
  ranked_keys <- vapply(ranked_best, function(x) {
    paste(x$tune_stage, as.integer(x$candidate$candidate_id), sep = "|")
  }, character(1))
  if (length(ranked_best) > 0L && ranked_keys[1L] != best_key) {
    selected_pos <- match(best_key, ranked_keys)
    if (!is.na(selected_pos)) {
      ranked_best <- c(ranked_best[selected_pos],
                       ranked_best[-selected_pos])
      ranked_keys <- c(ranked_keys[selected_pos],
                       ranked_keys[-selected_pos])
    }
  }
  scores$selection_rank <- NA_integer_
  for (rank_i in seq_along(ranked_best)) {
    rank_key <- paste(ranked_best[[rank_i]]$tune_stage,
                      as.integer(ranked_best[[rank_i]]$candidate$candidate_id),
                      sep = "|")
    idx <- paste(scores$tune_stage, scores$candidate_id, sep = "|") ==
      rank_key
    scores$selection_rank[idx] <- rank_i
  }

  selected_params <- base_params
  selected_params$target_transform <- best$effective_transform
	  selected_params$ntree <- as.integer(best$candidate$ntree)
	  selected_params$ndpost <- .gb_param(best$params, "ndpost",
	                                      .gb_param(base_params, "ndpost", NULL))
	  selected_params$nskip <- .gb_param(best$params, "nskip",
	                                     .gb_param(base_params, "nskip", NULL))
	  selected_params$n_basis <- as.integer(best$candidate$n_basis)
	  selected_params$bandwidth <- as.numeric(best$candidate$bandwidth)
	  selected_params$spatial_policy <- as.character(best$candidate$spatial_policy)
	  selected_params$anisotropy_ratio <- as.numeric(best$candidate$anisotropy_ratio)
	  selected_params$k <- as.numeric(best$candidate$k)
	  selected_params$power <- as.numeric(best$candidate$power)
	  selected_params$base <- as.numeric(best$candidate$base)
	  selected_params$sigdf <- as.numeric(best$candidate$sigdf)
	  selected_params$sigquant <- as.numeric(best$candidate$sigquant)
	  selected_params$duan_apply <- FALSE

  # --- OOF-selected continuous shrinkage/blend toward RF (lambda) --------------
  # Even when the gate keeps a GeoBART candidate, blend its OOF prediction with
  # the paired Wadoux-RF OOF prediction by a convex weight lambda in [0,1],
  # where final = (1-lambda)*GeoBART + lambda*RF. lambda is chosen on the SAME
  # cross-fitted OOF validation targets the gate uses, over a grid that INCLUDES
  # 0 (pure GeoBART) and 1 (pure RF), so the blend can never increase OOF RMSE
  # vs the selected candidate and can only move it toward RF. Controlled by
  # GEOBART_RF_BLEND_LAMBDA_GRID (default "0,0.1,...,1"); set to "0" to disable.
  geobart_oof_rf_lambda <- function() {
    if (!(exists("fit_rf_default_wadoux", mode = "function") &&
          exists("predict_rf_default_wadoux", mode = "function") &&
          exists("response_name_wadoux"))) return(0)
    grid <- .gb_env_num_vec("GEOBART_RF_BLEND_LAMBDA_GRID",
                            seq(0, 1, by = 0.1))
    grid <- sort(unique(pmin(pmax(grid[is.finite(grid)], 0), 1)))
    if (length(grid) == 0L) return(0)
    if (length(grid) == 1L && grid[1L] == 0) return(0)
    gb_val <- best$out$pred_val
    obs_val <- fd$y$val
    if (is.null(gb_val) || length(gb_val) != length(obs_val)) return(0)
    rf_val <- tryCatch({
      Xtr <- as.data.frame(fd$X$train); Xva <- as.data.frame(fd$X$val)
      pn <- colnames(Xtr)
      rseed <- as.integer(getOption("wadoux_rf_seed", 123L))
      if (!is.finite(rseed)) rseed <- 123L
      tdf <- cbind(Xtr, setNames(list(as.numeric(fd$y$train)),
                                 response_name_wadoux))
      frm <- stats::as.formula(paste(response_name_wadoux, "~",
               paste(sprintf("`%s`", pn), collapse = "+")))
      as.numeric(predict_rf_default_wadoux(
        fit_rf_default_wadoux(tdf, frm, seed = rseed), Xva))
    }, error = function(e) NULL)
    if (is.null(rf_val) || length(rf_val) != length(obs_val)) return(0)
    rmse_at <- vapply(grid, function(lam) {
      p <- (1 - lam) * gb_val + lam * rf_val
      sqrt(mean((obs_val - p)^2, na.rm = TRUE))
    }, numeric(1))
    grid[which.min(rmse_at)]
  }
  rf_blend_lambda <- tryCatch(geobart_oof_rf_lambda(), error = function(e) 0)

  selected_out <- best$out
  if (inner_folds > 1L || is.null(selected_out)) {
    final_params <- selected_params[
      intersect(names(selected_params), train_arg_names)
    ]
    selected_out <- do.call(
      train_geobart_one_fold,
      c(list(fd = fd, verbose = FALSE), final_params)
    )
	  } else {
    selected_params$target_transform <- selected_out$diagnostics$target_transform
    selected_params$ntree <- selected_out$diagnostics$ntree
    selected_params$ndpost <- selected_out$diagnostics$ndpost
	    selected_params$nskip <- selected_out$diagnostics$nskip
	    selected_params$n_basis <- selected_out$diagnostics$n_basis
	    selected_params$bandwidth <- selected_out$diagnostics$bandwidth
	    selected_params$spatial_policy <- selected_out$diagnostics$spatial_policy
	    selected_params$anisotropy_ratio <- selected_out$diagnostics$anisotropy_ratio
	    selected_params$k <- selected_out$diagnostics$k
	    selected_params$power <- selected_out$diagnostics$power
	    selected_params$base <- selected_out$diagnostics$base
	    selected_params$sigdf <- selected_out$diagnostics$sigdf
	    selected_params$sigquant <- selected_out$diagnostics$sigquant
	  }

  if (isTRUE(autotune_v3)) {
    selected_out <- geobart_apply_post_calibration_out(
      selected_out, best$post_calibration,
      y_val = fd$y$val, y_test = fd$y$test
    )
  }

  # Apply the OOF-selected RF shrinkage/blend (lambda in [0,1]) to the final
  # held-out predictions. lambda=0 is a no-op; lambda=1 emits the literal RF.
  if (is.finite(rf_blend_lambda) && rf_blend_lambda > 0) {
    rf_blend_pred <- function(Xnew_block) {
      tryCatch({
        Xtr <- as.data.frame(fd$X$train); Xnew <- as.data.frame(Xnew_block)
        pn <- colnames(Xtr)
        rseed <- as.integer(getOption("wadoux_rf_seed", 123L))
        if (!is.finite(rseed)) rseed <- 123L
        tdf <- cbind(Xtr, setNames(list(as.numeric(fd$y$train)),
                                   response_name_wadoux))
        frm <- stats::as.formula(paste(response_name_wadoux, "~",
                 paste(sprintf("`%s`", pn), collapse = "+")))
        as.numeric(predict_rf_default_wadoux(
          fit_rf_default_wadoux(tdf, frm, seed = rseed), Xnew))
      }, error = function(e) NULL)
    }
    rf_test <- rf_blend_pred(fd$X$test)
    rf_val2 <- rf_blend_pred(fd$X$val)
    if (!is.null(rf_test) && !is.null(selected_out$pred_test) &&
        length(rf_test) == length(selected_out$pred_test)) {
      selected_out$pred_test <-
        (1 - rf_blend_lambda) * selected_out$pred_test + rf_blend_lambda * rf_test
    }
    if (!is.null(rf_val2) && !is.null(selected_out$pred_val) &&
        length(rf_val2) == length(selected_out$pred_val)) {
      selected_out$pred_val <-
        (1 - rf_blend_lambda) * selected_out$pred_val + rf_blend_lambda * rf_val2
    }
    selected_out$diagnostics$rf_blend_lambda <- rf_blend_lambda
  } else {
    selected_out$diagnostics$rf_blend_lambda <- 0
  }
	  selected_out$diagnostics$tuning_autotune_v3 <- autotune_v3
	  selected_out$diagnostics$tuning_autotune_v4 <- autotune_v4
	  selected_out$diagnostics$tuning_selection_objective <- selection_objective
	  selected_out$diagnostics$tuning_rf_inner_RMSE <- best$rf_inner_RMSE
	  selected_out$diagnostics$tuning_rf_inner_fold_RMSE <-
	    best$rf_inner_fold_RMSE
		  selected_out$diagnostics$tuning_selected_delta_RMSE_vs_RF <-
		    best$delta_RMSE_vs_RF
		  selected_out$diagnostics$tuning_noharm_policy <- noharm_policy
		  selected_out$diagnostics$tuning_noharm_preselected_candidate_id <-
		    as.integer(pre_noharm_best$candidate$candidate_id)
		  selected_out$diagnostics$tuning_noharm_preselected_tune_stage <-
		    as.character(pre_noharm_best$tune_stage)
		  selected_out$diagnostics$tuning_noharm_anchor_candidate_id <-
		    if (!is.null(anchor_entry)) {
		      as.integer(anchor_entry$best$candidate$candidate_id)
		    } else {
		      NA_integer_
		    }
		  selected_out$diagnostics$tuning_noharm_anchor_tune_stage <-
		    if (!is.null(anchor_entry)) as.character(anchor_entry$best$tune_stage) else NA_character_
		  selected_out$diagnostics$tuning_noharm_anchor_candidate_label <-
		    noharm_anchor_label
		  selected_out$diagnostics$tuning_noharm_anchor_score <-
		    noharm_anchor_score
		  selected_out$diagnostics$tuning_noharm_anchor_val_rmse <-
		    noharm_anchor_val_rmse
		  selected_out$diagnostics$tuning_noharm_margin_abs <- noharm_margin_abs
		  selected_out$diagnostics$tuning_noharm_margin_rel <- noharm_margin_rel
		  selected_out$diagnostics$tuning_noharm_margin <- noharm_margin
		  selected_out$diagnostics$tuning_noharm_delta_vs_anchor <-
		    noharm_delta_vs_anchor
		  selected_out$diagnostics$tuning_noharm_applied <- noharm_applied
			  selected_out$diagnostics$tuning_noharm_restrict_ensemble <-
			    noharm_restrict_ensemble
			  selected_out$diagnostics$tuning_noharm_reason <- noharm_reason
			  selected_out$diagnostics$tuning_rf_noharm_policy <-
			    rf_noharm_policy
			  selected_out$diagnostics$tuning_rf_noharm_preselected_candidate_id <-
			    as.integer(pre_rf_noharm_best$candidate$candidate_id)
			  selected_out$diagnostics$tuning_rf_noharm_preselected_tune_stage <-
			    as.character(pre_rf_noharm_best$tune_stage)
			  selected_out$diagnostics$tuning_rf_noharm_preselected_delta_RMSE_vs_RF <-
			    rf_noharm_preselected_delta
			  selected_out$diagnostics$tuning_rf_noharm_anchor_candidate_id <-
			    if (!is.null(anchor_entry)) {
			      as.integer(anchor_entry$best$candidate$candidate_id)
			    } else {
			      NA_integer_
			    }
			  selected_out$diagnostics$tuning_rf_noharm_anchor_tune_stage <-
			    if (!is.null(anchor_entry)) {
			      as.character(anchor_entry$best$tune_stage)
			    } else {
			      NA_character_
			    }
			  selected_out$diagnostics$tuning_rf_noharm_anchor_candidate_label <-
			    noharm_anchor_label
			  selected_out$diagnostics$tuning_rf_noharm_anchor_delta_RMSE_vs_RF <-
			    rf_noharm_anchor_delta
			  selected_out$diagnostics$tuning_rf_noharm_margin_abs <-
			    rf_noharm_margin_abs
			  selected_out$diagnostics$tuning_rf_noharm_margin_rel <-
			    rf_noharm_margin_rel
			  selected_out$diagnostics$tuning_rf_noharm_margin <-
			    rf_noharm_margin
			  selected_out$diagnostics$tuning_rf_noharm_required_delta_RMSE_vs_RF <-
			    rf_noharm_required_delta
			  selected_out$diagnostics$tuning_rf_noharm_failed <-
			    rf_noharm_failed
			  selected_out$diagnostics$tuning_rf_noharm_applied <-
			    rf_noharm_applied
			  selected_out$diagnostics$tuning_rf_noharm_restrict_ensemble <-
			    rf_noharm_restrict_ensemble
			  selected_out$diagnostics$tuning_rf_noharm_reason <-
			    rf_noharm_reason
			  selected_out$diagnostics$tuning_stage2_select_only <- v3_stage2_select_only
  selected_out$diagnostics$tuning_stage2_seed_ensemble_n <-
    v3_stage2_seed_ensemble_n
  selected_out$diagnostics$tuning_rmse_sd_weight <-
    if (isTRUE(autotune_v3)) v3_rmse_sd_weight else 0
  selected_out$diagnostics$tuning_abs_me_weight <-
    if (isTRUE(autotune_v3)) v3_abs_me_weight else 0
  selected_out$diagnostics$tuning_quantile_bins <-
    if (isTRUE(autotune_v4)) v4_quantile_bins else 0L
  selected_out$diagnostics$tuning_quantile_excess_weight <-
    if (isTRUE(autotune_v4)) v4_quantile_excess_weight else 0
  selected_out$diagnostics$tuning_quantile_sd_weight <-
    if (isTRUE(autotune_v4)) v4_quantile_sd_weight else 0
  selected_out$diagnostics$tuning_selected_raw_score <- best$raw_score
  selected_out$diagnostics$tuning_selected_raw_RMSE <- best$raw_metrics$RMSE
  selected_out$diagnostics$tuning_selected_raw_ME <- best$raw_metrics$ME
  selected_out$diagnostics$tuning_selected_RMSE_sd <- best$stable_RMSE_sd
  selected_out$diagnostics$tuning_selected_ME_abs <- best$stable_ME_abs
  selected_out$diagnostics$tuning_selected_quantile_RMSE_mean <-
    best$quantile_RMSE_mean
  selected_out$diagnostics$tuning_selected_quantile_RMSE_sd <-
    best$quantile_RMSE_sd
  selected_out$diagnostics$tuning_selected_quantile_RMSE_excess <-
    best$quantile_RMSE_excess
  selected_out$diagnostics$tuning_selected_quantile_RMSE <-
    best$quantile_RMSE
  selected_out$diagnostics$tuning_selected_quantile_bins <-
    best$quantile_bins
  selected_out$diagnostics$tuning_selected_fold_RMSE <- best$fold_RMSE
  selected_out$diagnostics$tuning_selected_fold_ME <- best$fold_ME

	  .gb_log(verbose, sprintf(
	    "[GeoBART tuning] selected %s candidate %d: %s | val_%s=%.4f\n",
	    best$tune_stage, best$candidate$candidate_id, best$candidate$candidate_label,
	    metric_name, best_score
	  ))

  list(
    selected_out = selected_out,
    selected_params = selected_params,
    selected_candidate = best$candidate,
    ranked_candidates = ranked_best,
	    post_policy = best$post_calibration,
	    tuning_scores = scores,
	    selection_metric = objective_label
	  )
}


# ---- Default parameter preset (mirrors NMK/DKP convention) -----------------

geobart_default_params <- list(
  target_transform = "auto",
  ntree            = NULL,
  ndpost           = NULL,
	  nskip            = NULL,
	  n_basis          = NULL,
	  bandwidth        = NULL,
	  spatial_policy   = NULL,
	  anisotropy_ratio = NULL,
	  k                = NULL,
	  power            = NULL,
	  base             = NULL,
	  sigdf            = NULL,
	  sigquant         = NULL,
	  train_seed       = 123L,
  duan_apply       = NULL,
  duan_min_val_n   = NULL,
  duan_trim_frac   = NULL
)
