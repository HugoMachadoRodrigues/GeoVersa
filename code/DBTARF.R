# =============================================================================
# DBTARF.R
#
# DB-TARF: Design-Based Targeted Adaptive Residual Forest.
#
# Prediction form:
#
#   y_hat(s, x) = RF_tuned(s, x) + lambda * correction(s, x)
#
# The complete DB-TARF pipeline is tuned by comparing full candidate profiles
# with out-of-fold RMSE. The default RF/residual-RF profile is one candidate in
# the same optimization space, not a separate fallback. Candidate corrections
# are learned from cross-fitted RF residuals and are accepted only when their
# out-of-fold RMSE beats the RF-only prediction on the calibration sample.
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

.dbt_bt <- function(x) {
  paste0("`", gsub("`", "\\\\`", x), "`")
}

.dbt_formula <- function(response_name, predictor_names) {
  stats::as.formula(
    paste(.dbt_bt(response_name), "~",
          paste(vapply(predictor_names, .dbt_bt, character(1)), collapse = "+"))
  )
}

.dbt_auto_folds <- function(n) {
  if (n < 30L) return(3L)
  if (n < 120L) return(4L)
  5L
}

.dbt_lambda_grid <- function(lambda_grid = NULL, n = NULL) {
  if (!is.null(lambda_grid)) return(sort(unique(as.numeric(lambda_grid))))
  if (is.null(n) || !is.finite(n)) n <- 500L
  if (n < 80L) {
    seq(-0.20, 0.40, by = 0.05)
  } else if (n < 300L) {
    seq(-0.25, 0.55, by = 0.05)
  } else {
    seq(-0.25, 0.60, by = 0.05)
  }
}

.dbt_resolve_coord_names <- function(df, coord_names = c("x", "y")) {
  if (is.null(coord_names) || length(coord_names) == 0L) return(character(0))
  if (length(coord_names) == 1L &&
      tolower(as.character(coord_names)) %in% c("auto", "detect", "")) {
    candidates <- list(
      c("x", "y"), c("X", "Y"),
      c("lon", "lat"), c("longitude", "latitude"),
      c("easting", "northing"), c("Easting", "Northing"),
      c("coord_x", "coord_y")
    )
    for (cand in candidates) {
      if (all(cand %in% names(df))) return(cand)
    }
    return(character(0))
  }
  coord_names <- as.character(coord_names)
  if (length(coord_names) != 2L || !all(coord_names %in% names(df))) {
    return(character(0))
  }
  coord_names
}

.dbt_infer_predictors <- function(df, response_name, predictor_names = NULL,
                                  coord_names = character(0)) {
  if (!is.null(predictor_names)) {
    missing <- setdiff(predictor_names, names(df))
    if (length(missing) > 0L) {
      stop("DBTARF predictor columns not found: ", paste(missing, collapse = ", "))
    }
    return(as.character(predictor_names))
  }
  candidates <- setdiff(names(df), c(response_name, coord_names))
  keep <- vapply(df[candidates], function(z) {
    is.numeric(z) || is.integer(z) || is.logical(z) ||
      is.factor(z) || is.character(z)
  }, logical(1))
  out <- candidates[keep]
  if (length(out) == 0L) stop("DBTARF could not infer any usable predictors.")
  out
}

.dbt_prepare_predictors <- function(train_df, test_df, predictor_names) {
  train_df <- as.data.frame(train_df)
  test_df <- as.data.frame(test_df)
  for (nm in predictor_names) {
    tr <- train_df[[nm]]
    te <- test_df[[nm]]
    if (is.numeric(tr) || is.integer(tr)) {
      tr <- as.numeric(tr)
      te <- suppressWarnings(as.numeric(te))
      med <- stats::median(tr[is.finite(tr)], na.rm = TRUE)
      if (!is.finite(med)) med <- 0
      tr[!is.finite(tr)] <- med
      te[!is.finite(te)] <- med
      train_df[[nm]] <- tr
      test_df[[nm]] <- te
    } else if (is.logical(tr)) {
      tr <- ifelse(is.na(tr), "missing", as.character(tr))
      te <- ifelse(is.na(te), "missing", as.character(te))
      lv <- unique(c("FALSE", "TRUE", "missing", tr))
      te[!(te %in% lv)] <- "missing"
      train_df[[nm]] <- factor(tr, levels = lv)
      test_df[[nm]] <- factor(te, levels = lv)
    } else {
      tr <- as.character(tr)
      te <- as.character(te)
      tr[is.na(tr) | !nzchar(tr)] <- "missing"
      te[is.na(te) | !nzchar(te)] <- "missing"
      lv <- unique(c("missing", tr))
      te[!(te %in% lv)] <- "missing"
      train_df[[nm]] <- factor(tr, levels = lv)
      test_df[[nm]] <- factor(te, levels = lv)
    }
  }
  list(train = train_df, test = test_df)
}

.dbt_prepare_coords <- function(train_df, test_df, coord_names) {
  if (length(coord_names) != 2L ||
      !all(coord_names %in% names(train_df)) ||
      !all(coord_names %in% names(test_df))) {
    return(list(train = NULL, test = NULL, use_spatial = FALSE))
  }
  tr <- as.matrix(train_df[, coord_names, drop = FALSE])
  te <- as.matrix(test_df[, coord_names, drop = FALSE])
  storage.mode(tr) <- "numeric"
  storage.mode(te) <- "numeric"
  ok <- is.finite(tr[, 1]) & is.finite(tr[, 2])
  if (sum(ok) < 20L ||
      diff(range(tr[ok, 1], na.rm = TRUE)) <= 0 ||
      diff(range(tr[ok, 2], na.rm = TRUE)) <= 0) {
    return(list(train = NULL, test = NULL, use_spatial = FALSE))
  }
  med <- apply(tr[ok, , drop = FALSE], 2L, stats::median, na.rm = TRUE)
  for (j in 1:2) {
    tr[!is.finite(tr[, j]), j] <- med[j]
    te[!is.finite(te[, j]), j] <- med[j]
  }
  colnames(tr) <- c("coord_x", "coord_y")
  colnames(te) <- c("coord_x", "coord_y")
  list(train = tr, test = te, use_spatial = TRUE)
}

.dbt_sanitize_model_names <- function(train_df, test_df, response_name,
                                      predictor_names, coord_names) {
  vars <- unique(c(response_name, predictor_names, coord_names))
  safe <- paste0(".dbt_v", seq_along(vars))
  names(safe) <- vars
  for (old in vars) {
    new <- safe[[old]]
    names(train_df)[names(train_df) == old] <- new
    if (old %in% names(test_df)) {
      names(test_df)[names(test_df) == old] <- new
    }
  }
  list(
    train = train_df,
    test = test_df,
    response_name = safe[[response_name]],
    predictor_names = unname(safe[predictor_names]),
    coord_names = unname(safe[coord_names])
  )
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

.dbt_as_bool <- function(x) {
  if (is.logical(x)) return(isTRUE(x))
  if (length(x) == 0L || is.null(x)) return(FALSE)
  toupper(as.character(x[1L])) %in% c("TRUE", "1", "YES", "Y")
}

.dbt_numeric_values <- function(x, default, integer = FALSE) {
  if (is.null(x)) x <- default
  out <- suppressWarnings(as.numeric(x))
  out <- out[is.finite(out)]
  if (length(out) == 0L) out <- suppressWarnings(as.numeric(default))
  out <- sort(unique(out))
  if (integer) out <- as.integer(round(out))
  out
}

.dbt_default_mtry_values <- function(p) {
  p <- as.integer(max(1L, p))
  sqrt_p <- max(1L, as.integer(round(sqrt(p))))
  if (p <= 6L) {
    vals <- seq_len(p)
  } else {
    vals <- c(1L, floor(sqrt_p / 2), sqrt_p, ceiling(1.5 * sqrt_p),
              round(p / 3), round(p / 2), p)
  }
  sort(unique(as.integer(.dbt_clamp(vals, 1L, p))))
}

.dbt_default_min_node_values <- function(n, role = "base") {
  n <- as.integer(max(2L, n))
  vals <- if (identical(role, "residual")) {
    c(2L, 3L, 5L, max(5L, floor(n / 80)), 10L, 20L)
  } else {
    c(1L, 3L, 5L, 10L, 20L)
  }
  sort(unique(as.integer(.dbt_clamp(vals, 1L, max(1L, n - 2L)))))
}

.dbt_default_tree_values <- function(n, role = "base") {
  n <- as.integer(max(1L, n))
  if (identical(role, "residual")) {
    if (n < 120L) c(300L, 500L, 1000L) else c(300L, 500L, 1000L, 1500L)
  } else {
    if (n < 120L) c(500L, 1000L, 1500L) else c(500L, 1000L, 1500L, 2000L, 2500L, 3000L)
  }
}

.dbt_rf_default_params <- function(p, n, role = "base") {
  list(
    num.trees = if (identical(role, "residual")) 300L else 500L,
    mtry = max(1L, as.integer(round(sqrt(max(1L, p))))),
    min.node.size = if (identical(role, "residual")) {
      max(5L, floor(max(1L, n) / 80))
    } else 5L,
    sample.fraction = 1.0
  )
}

.dbt_first_numeric <- function(x, default, integer = FALSE) {
  if (is.null(x)) return(default)
  out <- suppressWarnings(as.numeric(x))
  out <- out[is.finite(out)]
  if (length(out) == 0L) return(default)
  if (integer) as.integer(round(out[1L])) else out[1L]
}

.dbt_rf_fixed_grid <- function(p, n, role = "base",
                               num_trees = NULL,
                               mtry = NULL,
                               min_node_size = NULL,
                               sample_fraction = NULL) {
  p <- as.integer(max(1L, p))
  n <- as.integer(max(2L, n))
  defaults <- .dbt_rf_default_params(p, n, role)
  out <- data.frame(
    num.trees = .dbt_first_numeric(num_trees, defaults$num.trees, integer = TRUE),
    mtry = .dbt_first_numeric(mtry, defaults$mtry, integer = TRUE),
    min.node.size = .dbt_first_numeric(
      min_node_size, defaults$min.node.size, integer = TRUE),
    sample.fraction = .dbt_first_numeric(
      sample_fraction, defaults$sample.fraction, integer = FALSE),
    stringsAsFactors = FALSE
  )
  out$num.trees <- as.integer(max(1L, out$num.trees))
  out$mtry <- as.integer(.dbt_clamp(out$mtry, 1L, p))
  out$min.node.size <- as.integer(.dbt_clamp(
    out$min.node.size, 1L, max(1L, n - 2L)))
  out$sample.fraction <- as.numeric(.dbt_clamp(out$sample.fraction, 0.10, 1.00))
  out
}

.dbt_rf_param_grid <- function(p, n, role = "base",
                               num_trees = NULL,
                               mtry = NULL,
                               min_node_size = NULL,
                               sample_fraction = NULL,
                               max_grid = 18L,
                               seed = 1L) {
  p <- as.integer(max(1L, p))
  n <- as.integer(max(2L, n))
  defaults <- .dbt_rf_default_params(p, n, role)
  num_trees <- .dbt_numeric_values(
    num_trees, .dbt_default_tree_values(n, role), integer = TRUE)
  mtry <- .dbt_numeric_values(
    mtry, .dbt_default_mtry_values(p), integer = TRUE)
  min_node_size <- .dbt_numeric_values(
    min_node_size, .dbt_default_min_node_values(n, role), integer = TRUE)
  sample_fraction <- .dbt_numeric_values(
    sample_fraction, c(0.632, 0.80, 1.00), integer = FALSE)

  num_trees <- as.integer(pmax(1L, num_trees))
  mtry <- as.integer(.dbt_clamp(mtry, 1L, p))
  min_node_size <- as.integer(.dbt_clamp(min_node_size, 1L, max(1L, n - 2L)))
  sample_fraction <- .dbt_clamp(sample_fraction, 0.10, 1.00)

  grid <- expand.grid(
    num.trees = num_trees,
    mtry = mtry,
    min.node.size = min_node_size,
    sample.fraction = sample_fraction,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  default_row <- data.frame(
    num.trees = defaults$num.trees,
    mtry = defaults$mtry,
    min.node.size = defaults$min.node.size,
    sample.fraction = defaults$sample.fraction,
    stringsAsFactors = FALSE
  )
  grid <- unique(rbind(default_row, grid))
  grid$num.trees <- as.integer(grid$num.trees)
  grid$mtry <- as.integer(grid$mtry)
  grid$min.node.size <- as.integer(grid$min.node.size)
  grid$sample.fraction <- as.numeric(grid$sample.fraction)

  max_grid <- as.integer(max(1L, max_grid))
  if (nrow(grid) > max_grid) {
    default_key <- do.call(paste, c(default_row, sep = "|"))
    grid_keys <- do.call(paste, c(grid, sep = "|"))
    keep <- which(grid_keys == default_key)
    keep <- if (length(keep) > 0L) keep[1L] else 1L
    anchor_idx <- unique(c(
      keep,
      which(grid$num.trees == max(grid$num.trees) &
              grid$mtry == defaults$mtry &
              grid$min.node.size == defaults$min.node.size)[1L],
      which(grid$mtry == max(grid$mtry) &
              grid$num.trees == defaults$num.trees)[1L],
      which(grid$min.node.size == min(grid$min.node.size) &
              grid$num.trees == defaults$num.trees)[1L],
      which(grid$sample.fraction == min(grid$sample.fraction) &
              grid$mtry == defaults$mtry)[1L]
    ))
    anchor_idx <- anchor_idx[!is.na(anchor_idx) & is.finite(anchor_idx)]
    keep <- unique(c(keep, anchor_idx))
    if (length(keep) > max_grid) keep <- keep[seq_len(max_grid)]
    remaining <- setdiff(seq_len(nrow(grid)), keep)
    if (length(keep) < max_grid && length(remaining) > 0L) {
      set.seed(as.integer(seed))
      keep <- c(keep, sample(remaining, min(length(remaining), max_grid - length(keep))))
    }
    grid <- grid[unique(keep), , drop = FALSE]
  }
  rownames(grid) <- NULL
  grid
}

.dbt_params_from_row <- function(row) {
  list(
    num.trees = as.integer(row[["num.trees"]]),
    mtry = as.integer(row[["mtry"]]),
    min.node.size = as.integer(row[["min.node.size"]]),
    sample.fraction = as.numeric(row[["sample.fraction"]])
  )
}

.dbt_rf_args <- function(params = NULL, seed = NULL) {
  if (is.null(params)) {
    out <- list()
    if (!is.null(seed)) out$seed <- as.integer(seed)
    return(out)
  }
  out <- list(
    num.trees = as.integer(params$num.trees),
    mtry = as.integer(params$mtry),
    min.node.size = as.integer(params$min.node.size),
    sample.fraction = as.numeric(params$sample.fraction)
  )
  if (!is.null(seed)) out$seed <- as.integer(seed)
  out
}

.dbt_fit_rf <- function(train_df, response_name, predictor_names,
                        rf_params = NULL, seed = NULL) {
  form <- .dbt_formula(response_name, predictor_names)
  args <- c(list(formula = form, data = train_df), .dbt_rf_args(rf_params, seed))
  do.call(ranger::ranger, args)
}

.dbt_score_rf_cv <- function(train_df, response_name, predictor_names,
                             rf_params, inner_folds = 3L, seed = 1L) {
  y <- suppressWarnings(as.numeric(train_df[[response_name]]))
  k <- as.integer(max(2L, min(inner_folds, sum(is.finite(y)), nrow(train_df))))
  set.seed(as.integer(seed))
  folds <- caret::createFolds(y, k = k, list = TRUE, returnTrain = FALSE)
  pred <- rep(NA_real_, nrow(train_df))
  for (j in seq_along(folds)) {
    id_val <- folds[[j]]
    id_tr <- setdiff(seq_len(nrow(train_df)), id_val)
    mdl <- .dbt_fit_rf(
      train_df[id_tr, , drop = FALSE], response_name, predictor_names,
      rf_params = rf_params, seed = seed + j)
    pred[id_val] <- .dbt_predict_rf(mdl, train_df[id_val, , drop = FALSE])
  }
  .dbt_rmse(y, pred)
}

.dbt_tune_rf_model <- function(train_df, response_name, predictor_names,
                               role = "base",
                               tune = TRUE,
                               tune_method = "oob",
                               inner_folds = 3L,
                               max_grid = 18L,
                               num_trees = NULL,
                               mtry = NULL,
                               min_node_size = NULL,
                               sample_fraction = NULL,
                               seed = 1L) {
  train_df <- as.data.frame(train_df)
  role_grid <- if (grepl("resid", role)) "residual" else "base"
  tune <- .dbt_as_bool(tune)
  tune_method <- tolower(as.character(tune_method)[1L])
  if (!tune) tune_method <- "none"
  if (!tune_method %in% c("oob", "cv", "none")) tune_method <- "oob"
  grid <- if (identical(tune_method, "none")) {
    .dbt_rf_fixed_grid(
      p = length(predictor_names),
      n = nrow(train_df),
      role = role_grid,
      num_trees = num_trees,
      mtry = mtry,
      min_node_size = min_node_size,
      sample_fraction = sample_fraction)
  } else {
    .dbt_rf_param_grid(
      p = length(predictor_names),
      n = nrow(train_df),
      role = role_grid,
      num_trees = num_trees,
      mtry = mtry,
      min_node_size = min_node_size,
      sample_fraction = sample_fraction,
      max_grid = max_grid,
      seed = seed)
  }

  if (identical(tune_method, "none")) {
    params <- .dbt_params_from_row(grid[1L, , drop = FALSE])
    model <- .dbt_fit_rf(train_df, response_name, predictor_names,
                         rf_params = params, seed = seed)
    score <- .dbt_rmse(
      suppressWarnings(as.numeric(train_df[[response_name]])),
      .dbt_oob_or_in_sample(model, train_df))
    scores <- cbind(
      data.frame(role = role, tune_method = tune_method, candidate_id = 1L,
                 tuning_rmse = score, selected = TRUE,
                 stringsAsFactors = FALSE),
      grid[1L, , drop = FALSE])
    return(list(model = model, params = params, best_rmse = score,
                scores = scores))
  }

  score_rows <- vector("list", nrow(grid))
  models <- vector("list", nrow(grid))
  y <- suppressWarnings(as.numeric(train_df[[response_name]]))
  for (i in seq_len(nrow(grid))) {
    params <- .dbt_params_from_row(grid[i, , drop = FALSE])
    fit <- tryCatch({
      if (identical(tune_method, "cv")) {
        rm <- .dbt_score_rf_cv(
          train_df, response_name, predictor_names, params,
          inner_folds = inner_folds, seed = seed + 1000L + i)
        list(model = NULL, rmse = rm)
      } else {
        mdl <- .dbt_fit_rf(train_df, response_name, predictor_names,
                           rf_params = params, seed = seed + i)
        list(model = mdl, rmse = .dbt_rmse(y, .dbt_oob_or_in_sample(mdl, train_df)))
      }
    }, error = function(e) list(model = NULL, rmse = Inf))
    models[i] <- list(fit$model)
    score_rows[[i]] <- cbind(
      data.frame(role = role, tune_method = tune_method, candidate_id = i,
                 tuning_rmse = fit$rmse, selected = FALSE,
                 stringsAsFactors = FALSE),
      grid[i, , drop = FALSE])
  }
  scores <- do.call(rbind, score_rows)
  ok <- is.finite(scores$tuning_rmse)
  best_i <- if (any(ok)) which.min(scores$tuning_rmse) else 1L
  scores$selected[best_i] <- TRUE
  params <- .dbt_params_from_row(grid[best_i, , drop = FALSE])
  model <- models[[best_i]]
  if (is.null(model) || identical(tune_method, "cv")) {
    model <- .dbt_fit_rf(train_df, response_name, predictor_names,
                         rf_params = params, seed = seed + 99999L)
  }
  list(model = model, params = params,
       best_rmse = scores$tuning_rmse[best_i], scores = scores)
}

.dbt_default_profile_specs <- function(p, n, max_profiles = 6L) {
  p <- as.integer(max(1L, p))
  n <- as.integer(max(2L, n))
  max_profiles <- as.integer(max(1L, max_profiles))
  sqrt_p <- max(1L, as.integer(round(sqrt(p))))
  half_p <- as.integer(.dbt_clamp(round(p / 2), 1L, p))
  third_p <- as.integer(.dbt_clamp(round(p / 3), 1L, p))
  base <- rbind(
    data.frame(profile = "default_rf",
               rf_num_trees = 500L, rf_mtry = sqrt_p,
               rf_min_node_size = 5L, rf_sample_fraction = 1.00,
               resid_rf_num_trees = 300L, resid_rf_mtry = NA_integer_,
               resid_rf_min_node_size = max(5L, floor(n / 80)),
               resid_rf_sample_fraction = 1.00,
               stringsAsFactors = FALSE),
    data.frame(profile = "more_trees",
               rf_num_trees = 1000L, rf_mtry = sqrt_p,
               rf_min_node_size = 5L, rf_sample_fraction = 1.00,
               resid_rf_num_trees = 300L, resid_rf_mtry = NA_integer_,
               resid_rf_min_node_size = max(5L, floor(n / 80)),
               resid_rf_sample_fraction = 1.00,
               stringsAsFactors = FALSE),
    data.frame(profile = "deeper_trees",
               rf_num_trees = 1000L, rf_mtry = sqrt_p,
               rf_min_node_size = 1L, rf_sample_fraction = 1.00,
               resid_rf_num_trees = 500L, resid_rf_mtry = NA_integer_,
               resid_rf_min_node_size = 3L,
               resid_rf_sample_fraction = 1.00,
               stringsAsFactors = FALSE),
    data.frame(profile = "subsample",
               rf_num_trees = 1500L, rf_mtry = sqrt_p,
               rf_min_node_size = 5L, rf_sample_fraction = 0.80,
               resid_rf_num_trees = 500L, resid_rf_mtry = NA_integer_,
               resid_rf_min_node_size = max(3L, floor(n / 120)),
               resid_rf_sample_fraction = 0.80,
               stringsAsFactors = FALSE),
    data.frame(profile = "wide_mtry",
               rf_num_trees = 1000L, rf_mtry = half_p,
               rf_min_node_size = 5L, rf_sample_fraction = 0.80,
               resid_rf_num_trees = 300L, resid_rf_mtry = NA_integer_,
               resid_rf_min_node_size = max(5L, floor(n / 80)),
               resid_rf_sample_fraction = 1.00,
               stringsAsFactors = FALSE),
    data.frame(profile = "wide_deep_subsample",
               rf_num_trees = 2000L, rf_mtry = max(third_p, sqrt_p),
               rf_min_node_size = 1L, rf_sample_fraction = 0.632,
               resid_rf_num_trees = 500L, resid_rf_mtry = NA_integer_,
               resid_rf_min_node_size = 3L,
               resid_rf_sample_fraction = 0.80,
               stringsAsFactors = FALSE)
  )
  base[seq_len(min(nrow(base), max_profiles)), , drop = FALSE]
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
  if (is.null(vg_fit) || !any(is.finite(vg_fit$range))) return(NULL)

  positive_ranges <- vg_fit$range[is.finite(vg_fit$range) & vg_fit$range > 0]
  iso_range <- if (length(positive_ranges) > 0L) max(positive_ranges) else range_init
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
  feat <- cbind(x, rf_pred = as.numeric(base_pred))
  coords <- NULL
  if (length(coord_names) == 2L && all(coord_names %in% names(df))) {
    coords <- as.matrix(df[, coord_names, drop = FALSE])
    storage.mode(coords) <- "numeric"
    colnames(coords) <- c("coord_x", "coord_y")
    feat <- cbind(feat, as.data.frame(coords))
  }
  if (!is.null(cubist_pred) && length(cubist_pred) == nrow(df)) {
    feat$cubist_pred <- as.numeric(cubist_pred)
    feat$cubist_minus_rf <- as.numeric(cubist_pred) - as.numeric(base_pred)
  }
  if (!is.null(basis) && !is.null(coords)) {
    phi <- .dbt_make_basis(coords, basis)
    if (ncol(phi) > 0L) feat <- cbind(feat, as.data.frame(phi))
  }
  feat
}

.dbt_fit_residual_rf <- function(features, residuals, rf_params = NULL,
                                 seed = NULL) {
  df <- as.data.frame(features)
  df$.resid <- as.numeric(residuals)
  args <- c(list(formula = .resid ~ ., data = df),
            .dbt_rf_args(rf_params, seed))
  do.call(ranger::ranger, args)
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

.dbt_train_dbtarf_profile <- function(train_df,
                                      test_df,
                                      response_name,
                                      predictor_names = NULL,
                                      coord_names = "auto",
                                      n_folds = NULL,
                                      lambda_grid = NULL,
                                      n_basis = NULL,
                                      bandwidth = NULL,
                                      min_oof_improvement = 0.00,
                                      train_seed = 123L,
                                      rf_tune = TRUE,
                                      rf_tune_method = "oob",
                                      rf_tune_inner_folds = 3L,
                                      rf_tune_max_grid = 18L,
                                      rf_num_trees = NULL,
                                      rf_mtry = NULL,
                                      rf_min_node_size = NULL,
                                      rf_sample_fraction = NULL,
                                      resid_rf_tune = TRUE,
                                      resid_rf_tune_max_grid = 12L,
                                      resid_rf_num_trees = NULL,
                                      resid_rf_mtry = NULL,
                                      resid_rf_min_node_size = NULL,
                                      resid_rf_sample_fraction = NULL,
                                      ...) {
  if (!requireNamespace("ranger", quietly = TRUE)) {
    stop("DBTARF requires ranger.")
  }
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("DBTARF requires caret.")
  }

  train_df <- as.data.frame(train_df)
  test_df <- as.data.frame(test_df)
  if (!response_name %in% names(train_df)) {
    stop("DBTARF response column not found: ", response_name)
  }

  coord_names <- .dbt_resolve_coord_names(train_df, coord_names)
  predictor_names <- .dbt_infer_predictors(
    train_df, response_name, predictor_names, coord_names)
  original_response_name <- response_name
  original_predictor_names <- predictor_names
  original_coord_names <- coord_names
  missing_test_predictors <- setdiff(predictor_names, names(test_df))
  if (length(missing_test_predictors) > 0L) {
    stop("DBTARF test data is missing predictor columns: ",
         paste(missing_test_predictors, collapse = ", "))
  }
  safe_names <- .dbt_sanitize_model_names(
    train_df, test_df, response_name, predictor_names, coord_names)
  train_df <- safe_names$train
  test_df <- safe_names$test
  response_name <- safe_names$response_name
  predictor_names <- safe_names$predictor_names
  coord_names <- safe_names$coord_names
  prep <- .dbt_prepare_predictors(train_df, test_df, predictor_names)
  train_df <- prep$train
  test_df <- prep$test

  y <- as.numeric(train_df[[response_name]])
  keep_y <- is.finite(y)
  if (sum(keep_y) < 10L) stop("DBTARF requires at least 10 finite response values.")
  if (!all(keep_y)) {
    train_df <- train_df[keep_y, , drop = FALSE]
    y <- y[keep_y]
  }
  coord_pack <- .dbt_prepare_coords(train_df, test_df, coord_names)
  use_spatial <- isTRUE(coord_pack$use_spatial)
  if (use_spatial) {
    train_df[[coord_names[1L]]] <- coord_pack$train[, 1L]
    train_df[[coord_names[2L]]] <- coord_pack$train[, 2L]
    test_df[[coord_names[1L]]] <- coord_pack$test[, 1L]
    test_df[[coord_names[2L]]] <- coord_pack$test[, 2L]
  } else {
    coord_names <- character(0)
  }

  n <- nrow(train_df)
  if (is.null(n_folds)) n_folds <- .dbt_auto_folds(n)
  n_folds <- as.integer(max(2L, min(n_folds, n)))
  lambda_grid <- .dbt_lambda_grid(lambda_grid, n = n)
  rf_tune <- .dbt_as_bool(rf_tune)
  resid_rf_tune <- .dbt_as_bool(resid_rf_tune)
  rf_tune_method <- tolower(as.character(rf_tune_method)[1L])
  if (!rf_tune_method %in% c("oob", "cv", "none")) rf_tune_method <- "oob"
  base_rf_tune_method <- if (rf_tune) rf_tune_method else "none"
  resid_rf_tune_method <- if (resid_rf_tune) rf_tune_method else "none"
  rf_tune_inner_folds <- as.integer(max(2L, rf_tune_inner_folds))
  rf_tune_max_grid <- as.integer(max(1L, rf_tune_max_grid))
  resid_rf_tune_max_grid <- as.integer(max(1L, resid_rf_tune_max_grid))

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
    resid_cubist = rep(NA_real_, n)
  )
  if (use_spatial) corr_oof$resid_spatial <- rep(NA_real_, n)
  fold_rows <- list()
  rf_tune_rows <- list()

  cat(sprintf("[DBTARF] cross-fitting residual candidates (n=%d, folds=%d)...\n",
              n, n_folds))
  for (j in seq_along(folds)) {
    id_val <- folds[[j]]
    id_tr <- setdiff(seq_len(n), id_val)
    tr <- train_df[id_tr, , drop = FALSE]
    va <- train_df[id_val, , drop = FALSE]

    rf_fit_j <- .dbt_tune_rf_model(
      tr, response_name, predictor_names,
      role = sprintf("base_fold_%02d", j),
      tune = rf_tune,
      tune_method = base_rf_tune_method,
      inner_folds = rf_tune_inner_folds,
      max_grid = rf_tune_max_grid,
      num_trees = rf_num_trees,
      mtry = rf_mtry,
      min_node_size = rf_min_node_size,
      sample_fraction = rf_sample_fraction,
      seed = train_seed + 1000L * j)
    rf_tune_rows[[length(rf_tune_rows) + 1L]] <- cbind(
      stage = "crossfit", fold = j, rf_fit_j$scores)
    rf_j <- rf_fit_j$model
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

    basis_j <- NULL
    coords_tr <- coords_va <- NULL
    if (use_spatial) {
      coords_tr <- as.matrix(tr[, coord_names, drop = FALSE])
      coords_va <- as.matrix(va[, coord_names, drop = FALSE])
      basis_j <- .dbt_fit_basis(coords_tr, resid_tr, n_basis = n_basis,
                                bandwidth = bandwidth,
                                seed = train_seed + j)
    }
    feat_tr <- .dbt_residual_features(
      tr, predictor_names, coord_names, pred_tr_rf, pred_tr_cb, basis_j)
    feat_va <- .dbt_residual_features(
      va, predictor_names, coord_names, pred_va_rf, pred_va_cb, basis_j)

    resid_rf_fit_j <- tryCatch(
      .dbt_tune_rf_model(
        cbind(as.data.frame(feat_tr), .resid = as.numeric(resid_tr)),
        ".resid", names(as.data.frame(feat_tr)),
        role = sprintf("resid_fold_%02d", j),
        tune = resid_rf_tune,
        tune_method = resid_rf_tune_method,
        inner_folds = rf_tune_inner_folds,
        max_grid = resid_rf_tune_max_grid,
        num_trees = resid_rf_num_trees,
        mtry = resid_rf_mtry,
        min_node_size = resid_rf_min_node_size,
        sample_fraction = resid_rf_sample_fraction,
        seed = train_seed + 2000L * j),
      error = function(e) NULL)
    if (!is.null(resid_rf_fit_j)) {
      rf_tune_rows[[length(rf_tune_rows) + 1L]] <- cbind(
        stage = "crossfit", fold = j, resid_rf_fit_j$scores)
      corr_oof$resid_rf[id_val] <- .dbt_predict_residual_rf(
        resid_rf_fit_j$model, feat_va)
    }

    resid_cb_j <- .dbt_fit_cubist(feat_tr, resid_tr)
    if (!is.null(resid_cb_j)) {
      corr_oof$resid_cubist[id_val] <- .dbt_predict_cubist(resid_cb_j, feat_va)
    }

    if (use_spatial) {
      corr_oof$resid_spatial[id_val] <- .dbt_kernel_residual_predict(
        coords_tr, resid_tr, coords_va, basis_j$bandwidth)
    }

    fold_rows[[j]] <- data.frame(
      fold = j,
      n_train = length(id_tr),
      n_val = length(id_val),
      rf_rmse = .dbt_rmse(va[[response_name]], pred_va_rf),
      cubist_rmse = .dbt_rmse(va[[response_name]], pred_va_cb),
      residual_sd = stats::sd(resid_tr, na.rm = TRUE),
      rf_num_trees = rf_fit_j$params$num.trees,
      rf_mtry = rf_fit_j$params$mtry,
      rf_min_node_size = rf_fit_j$params$min.node.size,
      rf_sample_fraction = rf_fit_j$params$sample.fraction,
      rf_tuning_rmse = rf_fit_j$best_rmse,
      resid_rf_num_trees = if (is.null(resid_rf_fit_j)) NA_integer_ else resid_rf_fit_j$params$num.trees,
      resid_rf_mtry = if (is.null(resid_rf_fit_j)) NA_integer_ else resid_rf_fit_j$params$mtry,
      resid_rf_min_node_size = if (is.null(resid_rf_fit_j)) NA_integer_ else resid_rf_fit_j$params$min.node.size,
      resid_rf_sample_fraction = if (is.null(resid_rf_fit_j)) NA_real_ else resid_rf_fit_j$params$sample.fraction,
      resid_rf_tuning_rmse = if (is.null(resid_rf_fit_j)) NA_real_ else resid_rf_fit_j$best_rmse,
      n_basis = if (is.null(basis_j)) NA_integer_ else basis_j$n_basis,
      bandwidth = if (is.null(basis_j)) NA_real_ else basis_j$bandwidth,
      range_major = if (is.null(basis_j)) NA_real_ else basis_j$range_major,
      nugget_ratio = if (is.null(basis_j)) NA_real_ else basis_j$nugget_ratio,
      stringsAsFactors = FALSE
    )
  }

  baseline_rmse <- .dbt_rmse(y, base_oof)
  candidate_rows <- list(data.frame(
    mode = "RF_noharm", lambda = 0, oof_rmse = baseline_rmse,
    delta_rmse = 0, stringsAsFactors = FALSE))

  valid_corr <- corr_oof
  ensemble_names <- intersect(c("resid_rf", "resid_cubist", "resid_spatial"),
                              names(valid_corr))
  if (length(ensemble_names) > 0L) {
    valid_corr$resid_ensemble <- rowMeans(as.data.frame(valid_corr[ensemble_names]),
                                          na.rm = TRUE)
    valid_corr$resid_ensemble[!is.finite(valid_corr$resid_ensemble)] <- 0
  }

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

  rf_full_fit <- .dbt_tune_rf_model(
    train_df, response_name, predictor_names,
    role = "base_full",
    tune = rf_tune,
    tune_method = base_rf_tune_method,
    inner_folds = rf_tune_inner_folds,
    max_grid = rf_tune_max_grid,
    num_trees = rf_num_trees,
    mtry = rf_mtry,
    min_node_size = rf_min_node_size,
    sample_fraction = rf_sample_fraction,
    seed = train_seed + 900000L)
  rf_tune_rows[[length(rf_tune_rows) + 1L]] <- cbind(
    stage = "full", fold = NA_integer_, rf_full_fit$scores)
  rf_full <- rf_full_fit$model
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
  resid_rf_full_fit <- NULL

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
    basis_full <- NULL
    if (use_spatial) {
      basis_full <- .dbt_fit_basis(
        as.matrix(train_df[, coord_names, drop = FALSE]), residual_full,
        n_basis = n_basis, bandwidth = bandwidth, seed = train_seed)
    }
    feat_train <- .dbt_residual_features(
      train_df, predictor_names, coord_names, base_train_full,
      cubist_train, basis_full)
    feat_test <- .dbt_residual_features(
      test_df, predictor_names, coord_names, base_test,
      cubist_test, basis_full)
    if (mode %in% c("resid_rf", "resid_ensemble")) {
      resid_rf_full_fit <- tryCatch(
        .dbt_tune_rf_model(
          cbind(as.data.frame(feat_train), .resid = as.numeric(residual_full)),
          ".resid", names(as.data.frame(feat_train)),
          role = "resid_full",
          tune = resid_rf_tune,
          tune_method = resid_rf_tune_method,
          inner_folds = rf_tune_inner_folds,
          max_grid = resid_rf_tune_max_grid,
          num_trees = resid_rf_num_trees,
          mtry = resid_rf_mtry,
          min_node_size = resid_rf_min_node_size,
          sample_fraction = resid_rf_sample_fraction,
          seed = train_seed + 950000L),
        error = function(e) NULL)
      if (!is.null(resid_rf_full_fit)) {
        rf_tune_rows[[length(rf_tune_rows) + 1L]] <- cbind(
          stage = "full", fold = NA_integer_, resid_rf_full_fit$scores)
      }
    }

    if (mode == "bias") {
      selected_correction_test <- rep(mean(residual_full, na.rm = TRUE),
                                      length(base_test))
    } else if (mode == "cubist_delta") {
      selected_correction_test <- cubist_test - base_test
    } else if (mode == "resid_rf") {
      if (!is.null(resid_rf_full_fit)) {
        selected_correction_test <- .dbt_predict_residual_rf(
          resid_rf_full_fit$model, feat_test)
      }
    } else if (mode == "resid_cubist") {
      mdl <- .dbt_fit_cubist(feat_train, residual_full)
      selected_correction_test <- .dbt_predict_cubist(mdl, feat_test)
    } else if (mode == "resid_spatial" && use_spatial) {
      selected_correction_test <- .dbt_kernel_residual_predict(
        as.matrix(train_df[, coord_names, drop = FALSE]), residual_full,
        as.matrix(test_df[, coord_names, drop = FALSE]), basis_full$bandwidth)
    } else if (mode == "resid_ensemble") {
      parts <- list()
      if (!is.null(resid_rf_full_fit)) {
        parts$resid_rf <- .dbt_predict_residual_rf(
          resid_rf_full_fit$model, feat_test)
      }
      mdl_cb <- .dbt_fit_cubist(feat_train, residual_full)
      if (!is.null(mdl_cb)) parts$resid_cubist <- .dbt_predict_cubist(mdl_cb, feat_test)
      if (use_spatial) {
        parts$resid_spatial <- .dbt_kernel_residual_predict(
          as.matrix(train_df[, coord_names, drop = FALSE]), residual_full,
          as.matrix(test_df[, coord_names, drop = FALSE]), basis_full$bandwidth)
      }
      if (length(parts) == 0L) {
        selected_correction_test <- rep(0, length(base_test))
      } else {
        selected_correction_test <- rowMeans(as.data.frame(parts), na.rm = TRUE)
        selected_correction_test[!is.finite(selected_correction_test)] <- 0
      }
    }
    selected_correction_test[!is.finite(selected_correction_test)] <- 0
    pred_test <- base_test + lambda * selected_correction_test
  }

  if (any(!is.finite(pred_test))) pred_test[!is.finite(pred_test)] <- base_test[!is.finite(pred_test)]

  test_obs <- if (response_name %in% names(test_df)) {
    suppressWarnings(as.numeric(test_df[[response_name]]))
  } else rep(NA_real_, nrow(test_df))
  has_test_obs <- any(is.finite(test_obs))
  list(
    pred_test = as.numeric(pred_test),
    pred_test_base = as.numeric(base_test),
    pred_test_cubist = as.numeric(cubist_test),
    correction_test = as.numeric(selected_correction_test),
    metrics_test = data.frame(
      N = length(test_obs),
      RMSE = if (has_test_obs) .dbt_rmse(test_obs, pred_test) else NA_real_,
      RMSE_base = if (has_test_obs) .dbt_rmse(test_obs, base_test) else NA_real_,
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
      response_name = original_response_name,
      predictor_names = paste(original_predictor_names, collapse = ","),
      coord_names = paste(original_coord_names, collapse = ","),
      use_spatial = use_spatial,
      lambda_min = min(lambda_grid, na.rm = TRUE),
      lambda_max = max(lambda_grid, na.rm = TRUE),
      min_oof_improvement = min_oof_improvement,
      rf_tune = rf_tune,
      rf_tune_method = base_rf_tune_method,
      rf_tune_inner_folds = rf_tune_inner_folds,
      rf_tune_max_grid = rf_tune_max_grid,
      resid_rf_tune = resid_rf_tune,
      resid_rf_tune_method = resid_rf_tune_method,
      resid_rf_tune_max_grid = resid_rf_tune_max_grid,
      base_rf_num_trees = rf_full_fit$params$num.trees,
      base_rf_mtry = rf_full_fit$params$mtry,
      base_rf_min_node_size = rf_full_fit$params$min.node.size,
      base_rf_sample_fraction = rf_full_fit$params$sample.fraction,
      base_rf_tuning_rmse = rf_full_fit$best_rmse,
      resid_rf_num_trees = if (is.null(resid_rf_full_fit)) NA_integer_ else resid_rf_full_fit$params$num.trees,
      resid_rf_mtry = if (is.null(resid_rf_full_fit)) NA_integer_ else resid_rf_full_fit$params$mtry,
      resid_rf_min_node_size = if (is.null(resid_rf_full_fit)) NA_integer_ else resid_rf_full_fit$params$min.node.size,
      resid_rf_sample_fraction = if (is.null(resid_rf_full_fit)) NA_real_ else resid_rf_full_fit$params$sample.fraction,
      resid_rf_tuning_rmse = if (is.null(resid_rf_full_fit)) NA_real_ else resid_rf_full_fit$best_rmse
    ),
    candidate_table = candidates,
    fold_diagnostics = do.call(rbind, fold_rows),
    rf_tuning_table = do.call(rbind, rf_tune_rows)
  )
}

train_dbtarf_one_fold <- function(train_df,
                                  test_df,
                                  response_name,
                                  predictor_names = NULL,
                                  coord_names = "auto",
                                  n_folds = NULL,
                                  lambda_grid = NULL,
                                  n_basis = NULL,
                                  bandwidth = NULL,
                                  min_oof_improvement = 0.00,
                                  train_seed = 123L,
                                  rf_tune = TRUE,
                                  rf_tune_method = "oob",
                                  rf_tune_inner_folds = 3L,
                                  rf_tune_max_grid = 18L,
                                  rf_num_trees = NULL,
                                  rf_mtry = NULL,
                                  rf_min_node_size = NULL,
                                  rf_sample_fraction = NULL,
                                  resid_rf_tune = TRUE,
                                  resid_rf_tune_max_grid = 12L,
                                  resid_rf_num_trees = NULL,
                                  resid_rf_mtry = NULL,
                                  resid_rf_min_node_size = NULL,
                                  resid_rf_sample_fraction = NULL,
                                  fair_profile_search = TRUE,
                                  fair_profile_max = 6L,
                                  ensemble_top_k = 1L,
                                  ensemble_weighting = "softmax",
                                  ensemble_temperature = 0.75,
                                  ...) {
  dbt_candidate_weights <- function(scores, weighting, temperature) {
    n <- length(scores)
    if (n == 0L) return(numeric())
    if (n == 1L) return(1)
    scores <- as.numeric(scores)
    finite <- is.finite(scores)
    if (!any(finite)) return(rep(1 / n, n))
    worst <- max(scores[finite], na.rm = TRUE)
    scores[!finite] <- worst + abs(worst) + 1
    weighting <- tolower(trimws(as.character(weighting)[1L]))
    temperature <- suppressWarnings(as.numeric(temperature)[1L])
    if (!is.finite(temperature) || temperature <= 0) temperature <- 1
    if (identical(weighting, "uniform")) {
      w <- rep(1, n)
    } else if (identical(weighting, "inverse")) {
      w <- 1 / pmax(scores - min(scores, na.rm = TRUE) + temperature, 1e-8)
    } else {
      z <- -(scores - min(scores, na.rm = TRUE)) / temperature
      z <- pmax(z, -700)
      w <- exp(z)
    }
    if (!any(is.finite(w)) || sum(w, na.rm = TRUE) <= 0) {
      return(rep(1 / n, n))
    }
    w / sum(w, na.rm = TRUE)
  }

  tag_profile <- function(out, label) {
    out$diagnostics$selected_profile <- label
    out$candidate_table <- cbind(profile = label, out$candidate_table)
    out$fold_diagnostics <- cbind(profile = label, out$fold_diagnostics)
    out$rf_tuning_table <- cbind(profile = label, out$rf_tuning_table)
    out
  }

  run_fixed_profile <- function(spec) {
    out <- .dbt_train_dbtarf_profile(
      train_df = train_df,
      test_df = test_df,
      response_name = response_name,
      predictor_names = predictor_names,
      coord_names = coord_names,
      n_folds = n_folds,
      lambda_grid = lambda_grid,
      n_basis = n_basis,
      bandwidth = bandwidth,
      min_oof_improvement = min_oof_improvement,
      train_seed = train_seed,
      rf_tune = FALSE,
      rf_tune_method = "none",
      rf_tune_inner_folds = rf_tune_inner_folds,
      rf_tune_max_grid = 1L,
      rf_num_trees = spec$rf_num_trees,
      rf_mtry = spec$rf_mtry,
      rf_min_node_size = spec$rf_min_node_size,
      rf_sample_fraction = spec$rf_sample_fraction,
      resid_rf_tune = FALSE,
      resid_rf_tune_max_grid = 1L,
      resid_rf_num_trees = spec$resid_rf_num_trees,
      resid_rf_mtry = spec$resid_rf_mtry,
      resid_rf_min_node_size = spec$resid_rf_min_node_size,
      resid_rf_sample_fraction = spec$resid_rf_sample_fraction,
      ...)
    tag_profile(out, spec$profile)
  }

  run_requested_profile <- function(label) {
    out <- .dbt_train_dbtarf_profile(
      train_df = train_df,
      test_df = test_df,
      response_name = response_name,
      predictor_names = predictor_names,
      coord_names = coord_names,
      n_folds = n_folds,
      lambda_grid = lambda_grid,
      n_basis = n_basis,
      bandwidth = bandwidth,
      min_oof_improvement = min_oof_improvement,
      train_seed = train_seed,
      rf_tune = rf_tune,
      rf_tune_method = rf_tune_method,
      rf_tune_inner_folds = rf_tune_inner_folds,
      rf_tune_max_grid = rf_tune_max_grid,
      rf_num_trees = rf_num_trees,
      rf_mtry = rf_mtry,
      rf_min_node_size = rf_min_node_size,
      rf_sample_fraction = rf_sample_fraction,
      resid_rf_tune = resid_rf_tune,
      resid_rf_tune_max_grid = resid_rf_tune_max_grid,
      resid_rf_num_trees = resid_rf_num_trees,
      resid_rf_mtry = resid_rf_mtry,
      resid_rf_min_node_size = resid_rf_min_node_size,
      resid_rf_sample_fraction = resid_rf_sample_fraction,
      ...)
    tag_profile(out, label)
  }

  requested_label <- if (.dbt_as_bool(rf_tune) || .dbt_as_bool(resid_rf_tune)) {
    "single_tuned"
  } else {
    "default_rf"
  }

  if (.dbt_as_bool(fair_profile_search) &&
      (.dbt_as_bool(rf_tune) || .dbt_as_bool(resid_rf_tune))) {
    train_df_for_specs <- as.data.frame(train_df)
    coord_for_specs <- .dbt_resolve_coord_names(train_df_for_specs, coord_names)
    pred_for_specs <- .dbt_infer_predictors(
      train_df_for_specs, response_name, predictor_names, coord_for_specs)
    profile_specs <- .dbt_default_profile_specs(
      p = length(pred_for_specs),
      n = nrow(train_df_for_specs),
      max_profiles = fair_profile_max)
    profile_out <- vector("list", nrow(profile_specs) + 1L)
    for (i in seq_len(nrow(profile_specs))) {
      profile_out[[i]] <- run_fixed_profile(profile_specs[i, , drop = FALSE])
    }
    profile_out[[length(profile_out)]] <- run_requested_profile(requested_label)
    profile_labels <- c(as.character(profile_specs$profile), requested_label)
    oof <- vapply(profile_out, function(x) {
      suppressWarnings(as.numeric(x$diagnostics$selected_oof_rmse))
    }, numeric(1))
    best_i <- if (any(is.finite(oof))) which.min(oof) else 1L
    ensemble_top_k <- suppressWarnings(as.integer(ensemble_top_k))
    if (!is.finite(ensemble_top_k) || ensemble_top_k < 1L) ensemble_top_k <- 1L
    rank_i <- order(oof, na.last = TRUE)
    rank_i <- rank_i[is.finite(oof[rank_i])]
    if (length(rank_i) == 0L) rank_i <- best_i
    top_i <- rank_i[seq_len(min(ensemble_top_k, length(rank_i)))]
    top_weights <- dbt_candidate_weights(
      oof[top_i], ensemble_weighting, ensemble_temperature)
    selected <- profile_out[[best_i]]
    if (length(top_i) > 1L) {
      weighted_vec <- function(name) {
        mat <- do.call(cbind, lapply(profile_out[top_i], function(x) {
          as.numeric(x[[name]])
        }))
        as.numeric(mat %*% top_weights)
      }
      selected$pred_test <- weighted_vec("pred_test")
      if (!is.null(selected$pred_test_base)) {
        selected$pred_test_base <- weighted_vec("pred_test_base")
      }
      if (!is.null(selected$pred_test_cubist)) {
        selected$pred_test_cubist <- weighted_vec("pred_test_cubist")
      }
      if (!is.null(selected$correction_test)) {
        selected$correction_test <- weighted_vec("correction_test")
      }
      test_obs <- if (response_name %in% names(test_df)) {
        suppressWarnings(as.numeric(test_df[[response_name]]))
      } else {
        rep(NA_real_, nrow(test_df))
      }
      if (any(is.finite(test_obs))) {
        selected$metrics_test$RMSE <- .dbt_rmse(test_obs, selected$pred_test)
        selected$metrics_test$RMSE_base <- .dbt_rmse(
          test_obs, selected$pred_test_base)
      }
    }
    selected$diagnostics$fair_profile_search <- TRUE
    selected$diagnostics$fair_profile_max <- as.integer(fair_profile_max)
    selected$diagnostics$profile_search_n <- length(profile_out)
    selected$diagnostics$profile_search_best_oof_rmse <- oof[best_i]
    selected$diagnostics$default_profile_oof_rmse <- oof[
      match("default_rf", profile_labels)]
    selected$diagnostics$single_tuned_profile_oof_rmse <- oof[
      match("single_tuned", profile_labels)]
    selected$diagnostics$ensemble_top_k_requested <- ensemble_top_k
    selected$diagnostics$ensemble_applied <- length(top_i) > 1L
    selected$diagnostics$ensemble_size <- length(top_i)
    selected$diagnostics$ensemble_weighting <- tolower(
      trimws(as.character(ensemble_weighting)[1L]))
    selected$diagnostics$ensemble_temperature <- suppressWarnings(
      as.numeric(ensemble_temperature)[1L])
    selected$diagnostics$ensemble_profiles <- paste(
      profile_labels[top_i], collapse = "|")
    selected$diagnostics$ensemble_profile_oof_rmse <- paste(
      sprintf("%.8f", oof[top_i]), collapse = "|")
    selected$diagnostics$ensemble_weights <- paste(
      sprintf("%.8f", top_weights), collapse = "|")
    selected$candidate_table <- do.call(
      rbind, lapply(profile_out, `[[`, "candidate_table"))
    selected$candidate_table$ensemble_member <-
      selected$candidate_table$profile %in% profile_labels[top_i]
    selected$candidate_table$ensemble_rank <- match(
      selected$candidate_table$profile, profile_labels[top_i])
    weight_map <- stats::setNames(top_weights, profile_labels[top_i])
    selected$candidate_table$ensemble_weight <- as.numeric(
      weight_map[selected$candidate_table$profile])
    selected$rf_tuning_table <- do.call(rbind, lapply(profile_out, `[[`, "rf_tuning_table"))
    return(selected)
  }

  requested <- run_requested_profile(requested_label)
  requested$diagnostics$fair_profile_search <- FALSE
  requested$diagnostics$fair_profile_max <- as.integer(fair_profile_max)
  requested$diagnostics$profile_search_n <- NA_integer_
  requested$diagnostics$profile_search_best_oof_rmse <-
    requested$diagnostics$selected_oof_rmse
  requested$diagnostics$default_profile_oof_rmse <-
    if (identical(requested_label, "default_rf")) {
      requested$diagnostics$selected_oof_rmse
    } else NA_real_
  requested$diagnostics$ensemble_top_k_requested <- as.integer(ensemble_top_k)
  requested$diagnostics$ensemble_applied <- FALSE
  requested$diagnostics$ensemble_size <- 1L
  requested$diagnostics$ensemble_weighting <- tolower(
    trimws(as.character(ensemble_weighting)[1L]))
  requested$diagnostics$ensemble_temperature <- suppressWarnings(
    as.numeric(ensemble_temperature)[1L])
  requested
}

dbtarf_default_params <- list(
  n_folds = NULL,
  lambda_grid = NULL,
  n_basis = NULL,
  bandwidth = NULL,
  min_oof_improvement = 0.00,
  train_seed = 123L,
  rf_tune = TRUE,
  rf_tune_method = "oob",
  rf_tune_inner_folds = 3L,
  rf_tune_max_grid = 18L,
  rf_num_trees = NULL,
  rf_mtry = NULL,
  rf_min_node_size = NULL,
  rf_sample_fraction = NULL,
  resid_rf_tune = TRUE,
  resid_rf_tune_max_grid = 12L,
  resid_rf_num_trees = NULL,
  resid_rf_mtry = NULL,
  resid_rf_min_node_size = NULL,
  resid_rf_sample_fraction = NULL,
  fair_profile_search = TRUE,
  fair_profile_max = 6L,
  ensemble_top_k = 1L,
  ensemble_weighting = "softmax",
  ensemble_temperature = 0.75
)

dbtarf_train_predict <- function(train_df,
                                 test_df,
                                 response_name,
                                 predictor_names = NULL,
                                 coord_names = "auto",
                                 ...) {
  train_dbtarf_one_fold(
    train_df = train_df,
    test_df = test_df,
    response_name = response_name,
    predictor_names = predictor_names,
    coord_names = coord_names,
    ...)
}
