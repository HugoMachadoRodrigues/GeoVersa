# =============================================================================
# DBTARF.R
#
# DB-TARF: Design-Based Targeted Adaptive Residual Forest.
#
# Prediction form:
#
#   y_hat(s, x) = RF_tuned(s, x) + lambda * w(s) * correction(s, x)
#   w(s) = exp(-d1(s) / ell): spatially-adaptive shrinkage gate (d1 = distance to
#   nearest calibration point, ell = gate_c * residual variogram range; gate_c
#   incl. Inf=off chosen by OOF). Down-weights the correction under extrapolation.
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
#   - nnet (optional single-hidden-layer MLP residual expert)
#   - gstat, sp (variogram-informed RBF features; ordinary kriging of residuals)
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
  if (!is.null(lambda_grid) && length(lambda_grid) > 0L) {
    return(sort(unique(as.numeric(lambda_grid))))
  }
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

# Distance at which a fitted variogram model first reaches `frac` of its sill.
# Applied identically to every family so that ranges are comparable across the
# families the search may select.
.dbt_practical_range <- function(vg_fit, frac = 0.95, n_eval = 2000L) {
  if (!requireNamespace("gstat", quietly = TRUE) || is.null(vg_fit)) return(NA_real_)
  sill <- sum(vg_fit$psill, na.rm = TRUE)
  if (!is.finite(sill) || sill <= 0) return(NA_real_)
  rr <- vg_fit$range[is.finite(vg_fit$range) & vg_fit$range > 0]
  if (length(rr) == 0L) return(NA_real_)
  maxd <- max(rr) * 10
  vl <- tryCatch(gstat::variogramLine(vg_fit, maxdist = maxd, n = n_eval),
                 error = function(e) NULL)
  if (is.null(vl) || nrow(vl) == 0L) return(NA_real_)
  hit <- which(vl$gamma >= frac * sill)
  if (length(hit) == 0L) return(max(vl$dist, na.rm = TRUE))
  as.numeric(vl$dist[hit[1L]])
}

# Fit a residual variogram WITHOUT assuming the model family.
#
# The family is the classical modelling decision that is normally made by eye or
# by habit; here all candidates are fitted to the SAME empirical variogram and
# the one with the lowest weighted SSE is kept. Comparing SSE is legitimate in
# this direction precisely because the binning and the pair counts are identical
# across candidates — which is NOT true when the geometry or the binning changes
# (see .dbt_fit_anisotropy, where SSE comparison is invalid and out-of-fold
# prediction error is used instead).
.dbt_fit_variogram <- function(y, coords, cutoff_frac = 0.50, n_lags = 15L,
                               families = c("Exp", "Sph", "Gau", "Mat"),
                               cressie = TRUE, family_tol = 0.05) {
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
                     width = cutoff / n_lags, cressie = isTRUE(cressie)),
    error = function(e) NULL
  )
  if (is.null(vg_emp) || nrow(vg_emp) < 3L) return(NULL)

  sill_tot <- stats::var(y, na.rm = TRUE)
  range_init <- cutoff * 0.30

  # select the variogram family by weighted SSE on this empirical variogram
  fit_family <- function(fam) {
    f <- tryCatch(suppressWarnings(gstat::fit.variogram(
      vg_emp, gstat::vgm(sill_tot * 0.80, fam, range_init, sill_tot * 0.20),
      fit.method = 7L)), error = function(e) NULL)
    if (is.null(f) || !any(is.finite(f$range))) return(NULL)
    e <- attr(f, "SSErr")
    if (is.null(e) || !is.finite(e)) return(NULL)
    list(fit = f, sse = as.numeric(e), family = fam)
  }
  cand <- Filter(Negate(is.null), lapply(families, fit_family))
  if (length(cand) == 0L) return(NULL)

  # Parsimony among nested families. Raw SSE always favours the most flexible
  # candidate: Matern nests the exponential (kappa = 1/2) and approaches the
  # Gaussian (kappa -> Inf), and gstat fits kappa, so Matern wins by one extra
  # degree of freedom even when the truth is exactly exponential. We therefore
  # keep the SIMPLEST family that is statistically indistinguishable from the
  # best one, where `families` is ordered from least to most flexible.
  sse <- vapply(cand, function(z) z$sse, numeric(1))
  thresh <- min(sse) * (1 + family_tol)
  best <- cand[[which(sse <= thresh)[1L]]]
  vg_fit <- best$fit
  vg_family <- best$family

  # PRACTICAL range, not the raw model parameter.
  #
  # gstat's `range` means different things in different families: for the
  # spherical model it is the distance at which the correlation reaches zero,
  # for the exponential it is the scale parameter a (practical range ~ 3a), for
  # the Gaussian ~ a*sqrt(3), and for Matern it depends on the smoothness. Since
  # the family is now selected from the data, using the raw parameter would put
  # the radial-basis bandwidth and the gate length on scales that are not
  # comparable across repetitions: a nominal range of 100 corresponds to a
  # practical range of roughly 79 (spherical), 167 (Gaussian) or 278
  # (exponential). We therefore evaluate the fitted model numerically and take
  # the distance at which the variogram first reaches 95 per cent of its sill,
  # which is one definition applied identically to every family.
  iso_range <- .dbt_practical_range(vg_fit)
  if (!is.finite(iso_range) || iso_range <= 0) {
    positive_ranges <- vg_fit$range[is.finite(vg_fit$range) & vg_fit$range > 0]
    iso_range <- if (length(positive_ranges) > 0L) max(positive_ranges) else range_init
  }
  if (!is.finite(iso_range) || iso_range <= 0) iso_range <- range_init
  psill_sum <- sum(vg_fit$psill, na.rm = TRUE)
  nugget_ratio <- if (is.finite(psill_sum) && psill_sum > 1e-8) {
    .dbt_clamp(vg_fit$psill[1L] / psill_sum, 0.02, 0.90)
  } else 0.50

  # ---- directional (anisotropic) structure ---------------------------------
  # Empirical variograms along four directions; the ranges of the fitted
  # exponential models give the major/minor axes and the major-axis angle.
  # This is only ever ADVISORY: whether the anisotropy is actually used is
  # decided out-of-fold by the profile search, never assumed here.
  aniso <- .dbt_fit_anisotropy(y, coords, n_lags, sill_tot, iso_range)

  list(range_major  = iso_range,
       nugget_ratio = nugget_ratio,
       family       = vg_family,
       aniso_angle  = aniso$angle,   # degrees clockwise from North (gstat convention)
       aniso_ratio  = aniso$ratio,   # minor / major in (0, 1]; 1 = isotropic
       aniso_major  = aniso$major,
       aniso_minor  = aniso$minor)
}

# Anisotropy estimation by geometry search.
#
# Fitting one variogram per direction and taking min/max of the four ranges is
# badly biased: the maximum of noisy range estimates over-shoots and the minimum
# under-shoots, so min/max reports anisotropy even for a perfectly isotropic
# field. We instead search over candidate geometries: for each (angle, ratio)
# the coordinates are mapped into the space where Euclidean distance IS the
# anisotropic distance, a single omnidirectional exponential variogram is fitted
# there, and the geometry retained is the one under which that single model fits
# best (weighted SSE from gstat). Isotropy (ratio = 1) is one of the candidates
# and wins ties, and an anisotropic geometry is returned only when it improves
# the fit by more than `tol` — so the estimator is parsimonious by construction.
.dbt_fit_anisotropy <- function(y, coords, n_lags, sill_tot, iso_range,
                                angles = seq(0, 157.5, by = 22.5),
                                ratios = c(1, 0.75, 0.55, 0.40, 0.25),
                                tol = 0.03, k = 5L, bw_frac = 0.15) {
  iso <- list(angle = 0, ratio = 1, major = iso_range, minor = iso_range)
  if (!requireNamespace("gstat", quietly = TRUE) ||
      !requireNamespace("sp", quietly = TRUE)) return(iso)

  xy <- as.matrix(coords)
  yv <- as.numeric(y)
  n <- length(yv)
  if (n < 40L) return(iso)

  # Scoring a geometry by fitted-variogram SSE does not work: transforming the
  # coordinates changes the distance distribution, hence the lag binning and the
  # pair counts, so SSE is not comparable across geometries. We therefore score
  # each geometry by k-fold OUT-OF-FOLD prediction error of a Gaussian kernel
  # smoother that uses the corresponding anisotropic distance — the same held-out
  # points for every geometry, so the comparison is like for like. The bandwidth
  # is set as a fraction of the median pairwise distance IN THE TRANSFORMED
  # SPACE, which makes the score invariant to the overall rescaling the
  # transform induces and sensitive only to the SHAPE (angle and ratio).
  set.seed(11L)
  fold <- sample(rep_len(seq_len(k), n))

  cv_rmse <- function(angle, ratio) {
    z <- .dbt_aniso_transform(xy, angle, ratio)
    D <- as.matrix(stats::dist(z))
    med <- stats::median(D[upper.tri(D)], na.rm = TRUE)
    if (!is.finite(med) || med <= 0) return(NA_real_)
    h <- med * bw_frac
    pred <- rep(NA_real_, n)
    for (kk in seq_len(k)) {
      te <- which(fold == kk); tr <- which(fold != kk)
      if (length(tr) < 5L) next
      W <- exp(-(D[te, tr, drop = FALSE] / h)^2)
      sw <- rowSums(W)
      ok <- sw > 1e-12
      pred[te[ok]] <- (W[ok, , drop = FALSE] %*% yv[tr]) / sw[ok]
      pred[te[!ok]] <- mean(yv[tr])
    }
    ok <- is.finite(pred)
    if (sum(ok) < 20L) return(NA_real_)
    sqrt(mean((yv[ok] - pred[ok])^2))
  }

  rmse_iso <- cv_rmse(0, 1)
  if (!is.finite(rmse_iso)) return(iso)

  best <- list(rmse = rmse_iso, angle = 0, ratio = 1)
  for (a in angles) for (r in ratios[ratios < 1]) {
    s <- cv_rmse(a, r)
    if (is.finite(s) && s < best$rmse) best <- list(rmse = s, angle = a, ratio = r)
  }

  # Parsimony: isotropy is the default and is displaced only when an
  # anisotropic geometry predicts held-out values materially better.
  if (best$ratio >= 1 || !is.finite(best$rmse) ||
      best$rmse > rmse_iso * (1 - tol)) return(iso)

  list(angle = as.numeric(best$angle), ratio = as.numeric(best$ratio),
       major = iso_range, minor = iso_range * best$ratio)
}

# Map coordinates into the space where Euclidean distance equals anisotropic
# distance: rotate so the major axis lies on x, then compress the minor axis by
# 1/ratio. With ratio = 1 this is (up to rotation) the identity, so isotropic
# behaviour is recovered exactly.
.dbt_aniso_transform <- function(coords, angle_deg, ratio) {
  coords <- as.matrix(coords)
  if (!is.finite(ratio) || ratio >= 1 || ratio <= 0 || !is.finite(angle_deg)) {
    return(coords)
  }
  # gstat angles run clockwise from North (the +y axis)
  th <- angle_deg * pi / 180
  R <- matrix(c(sin(th), cos(th), -cos(th), sin(th)), nrow = 2L, byrow = TRUE)
  z <- coords %*% t(R)
  z[, 2L] <- z[, 2L] / ratio
  z
}

.dbt_fit_basis <- function(coords, residuals, n_basis = NULL,
                           bandwidth = NULL, seed = NULL, use_aniso = FALSE,
                           bw_mult = 0.5, vg_cutoff = 0.50, vg_lags = 15L,
                           vg_cressie = TRUE) {
  coords <- as.matrix(coords)
  area <- diff(range(coords[, 1], na.rm = TRUE)) *
    diff(range(coords[, 2], na.rm = TRUE))
  vg <- .dbt_fit_variogram(residuals, coords, cutoff_frac = vg_cutoff,
                           n_lags = vg_lags, cressie = vg_cressie)
  if (is.null(vg)) {
    D <- as.matrix(stats::dist(coords))
    rho <- stats::median(D[upper.tri(D)], na.rm = TRUE)
    if (!is.finite(rho) || rho <= 0) rho <- sqrt(area) / 4
    nugget_ratio <- 0.75
  } else {
    rho <- vg$range_major
    nugget_ratio <- vg$nugget_ratio
  }
  # anisotropic geometry (ratio = 1 means the variogram search kept isotropy)
  an_angle <- if (is.null(vg) || is.null(vg$aniso_angle)) 0 else vg$aniso_angle
  an_ratio <- if (is.null(vg) || is.null(vg$aniso_ratio)) 1 else vg$aniso_ratio
  if (!isTRUE(use_aniso)) { an_angle <- 0; an_ratio <- 1 }

  if (is.null(n_basis)) {
    n_basis_theory <- 4 * area / (pi * rho ^ 2) * (1 - nugget_ratio)
    n_basis <- as.integer(.dbt_clamp(round(n_basis_theory), 8L, 48L))
  }
  # kmeans requires centers <= number of DISTINCT rows, not just nrow(coords).
  dup <- duplicated(round(coords, 6))
  n_distinct <- sum(!dup)
  n_basis <- as.integer(min(max(n_basis, 0L), n_distinct))
  # The variogram range enters as a TUNED scale: bandwidth = bw_mult * range,
  # with bw_mult selected out-of-fold by the profile search. bw_mult = 0.5
  # reproduces the previous derived-and-fixed behaviour exactly.
  if (!is.finite(bw_mult) || bw_mult <= 0) bw_mult <- 0.5
  if (is.null(bandwidth)) bandwidth <- max(rho * bw_mult, 1e-6)

  if (n_basis <= 0L) {
    return(list(centers = matrix(numeric(0), nrow = 0L, ncol = 2L),
                bandwidth = bandwidth, range_major = rho,
                nugget_ratio = nugget_ratio, n_basis = 0L,
                aniso_angle = an_angle, aniso_ratio = an_ratio))
  }

  if (!is.null(seed)) set.seed(as.integer(seed))
  centers <- if (n_basis == 1L) {
    matrix(colMeans(coords), nrow = 1L)
  } else {
    km <- tryCatch(
      stats::kmeans(coords, centers = n_basis, iter.max = 50L, nstart = 5L)$centers,
      error = function(e) NULL)
    if (is.null(km)) {
      coords[!dup, , drop = FALSE][seq_len(n_basis), , drop = FALSE]
    } else km
  }
  list(centers = as.matrix(centers), bandwidth = bandwidth,
       range_major = rho, nugget_ratio = nugget_ratio, n_basis = n_basis,
       aniso_angle = an_angle, aniso_ratio = an_ratio)
}

.dbt_make_basis <- function(coords, basis) {
  coords <- as.matrix(coords)
  centers <- as.matrix(basis$centers)
  if (nrow(centers) == 0L) return(matrix(numeric(0), nrow(coords), 0L))
  # Apply the fitted anisotropic geometry to BOTH the query points and the
  # centres, so the radial basis becomes an ellipse elongated along the major
  # axis. With ratio = 1 the transform is a rotation and the basis is exactly
  # the isotropic one, so this is a strict generalisation.
  ar <- if (is.null(basis$aniso_ratio)) 1 else basis$aniso_ratio
  aa <- if (is.null(basis$aniso_angle)) 0 else basis$aniso_angle
  if (is.finite(ar) && ar < 1) {
    coords  <- .dbt_aniso_transform(coords,  aa, ar)
    centers <- .dbt_aniso_transform(centers, aa, ar)
  }
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

.dbt_predict_residual_rf <- function(model, features) {
  as.numeric(predict(model, data = as.data.frame(features))$predictions)
}

# Single-hidden-layer MLP (neural network) residual learner via nnet. Features
# and the residual target are standardised (nnet is scale-sensitive); weight
# decay regularises. Deterministic given the seed (random initial weights).
# Returns NULL if nnet is unavailable or the fit fails, so it is safe to call.
.dbt_fit_mlp <- function(features, residuals, seed = NULL, size = NULL,
                         decay = 1e-3, maxit = 200L) {
  if (!requireNamespace("nnet", quietly = TRUE)) return(NULL)
  X <- suppressWarnings(data.matrix(as.data.frame(features)))
  y <- as.numeric(residuals)
  ok <- is.finite(y) & apply(is.finite(X), 1L, all)
  if (sum(ok) < 20L) return(NULL)
  X <- X[ok, , drop = FALSE]; y <- y[ok]
  ctr <- colMeans(X)
  scl <- apply(X, 2L, stats::sd); scl[!is.finite(scl) | scl < 1e-8] <- 1
  Xs <- scale(X, center = ctr, scale = scl)
  ymu <- mean(y); ysd <- stats::sd(y)
  if (!is.finite(ysd) || ysd < 1e-8) ysd <- 1
  ys <- (y - ymu) / ysd
  if (is.null(size)) size <- max(2L, min(10L, round(sqrt(ncol(Xs))) + 1L))
  if (!is.null(seed)) set.seed(as.integer(seed))
  fit <- tryCatch(
    nnet::nnet(Xs, ys, size = size, decay = decay, maxit = maxit,
               linout = TRUE, trace = FALSE, MaxNWts = 50000L),
    error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  list(fit = fit, ctr = ctr, scl = scl, ymu = ymu, ysd = ysd, cols = colnames(X))
}

.dbt_predict_mlp <- function(model, features) {
  if (is.null(model)) return(NULL)
  X <- suppressWarnings(data.matrix(as.data.frame(features)))
  if (length(setdiff(model$cols, colnames(X))) > 0L) return(NULL)
  X <- X[, model$cols, drop = FALSE]
  X[!is.finite(X)] <- 0
  Xs <- scale(X, center = model$ctr, scale = model$scl)
  p <- suppressWarnings(as.numeric(stats::predict(model$fit, Xs)))
  p * model$ysd + model$ymu
}

.dbt_kernel_residual_predict <- function(coords_train, residuals_train,
                                         coords_new, bandwidth,
                                         chunk = 20000L,
                                         aniso_angle = 0, aniso_ratio = 1) {
  coords_train <- as.matrix(coords_train)
  coords_new <- as.matrix(coords_new)
  # Anisotropic geometry: distances are measured after rotating onto the major
  # axis and compressing the minor one, so the smoothing kernel becomes an
  # ellipse. ratio = 1 leaves the isotropic behaviour untouched.
  if (is.finite(aniso_ratio) && aniso_ratio < 1) {
    coords_train <- .dbt_aniso_transform(coords_train, aniso_angle, aniso_ratio)
    coords_new   <- .dbt_aniso_transform(coords_new,   aniso_angle, aniso_ratio)
  }
  residuals_train <- as.numeric(residuals_train)
  bw2 <- 2 * max(bandwidth, 1e-6) ^ 2
  ct2 <- rowSums(coords_train ^ 2)
  mu <- mean(residuals_train, na.rm = TRUE)
  n_new <- nrow(coords_new)
  out <- numeric(n_new)
  # Chunk over new points so the dense weight matrix stays bounded (lets the
  # Population protocol score the full ~1.3M-cell grid without exhausting memory).
  for (s in seq.int(1L, n_new, by = chunk)) {
    e <- min(s + chunk - 1L, n_new)
    cn <- coords_new[s:e, , drop = FALSE]
    d2 <- outer(rowSums(cn ^ 2), ct2, `+`) - 2 * cn %*% t(coords_train)
    d2[d2 < 0] <- 0
    w <- exp(-d2 / bw2)
    denom <- rowSums(w)
    o <- as.numeric(w %*% residuals_train) / pmax(denom, 1e-12)
    o[!is.finite(o)] <- mu
    out[s:e] <- o
  }
  out
}

# Ordinary kriging of the residuals (regression-kriging correction): fit an
# exponential variogram to the residuals and krige them to new locations via
# gstat. Unlike the kernel smoother, kriging solves the kriging system for BLUP
# weights from the fitted covariance. Deterministic (no RNG). Falls back to the
# kernel smoother (or the residual mean) if gstat/sp are unavailable or the
# variogram fit fails, so it is always safe to call.
.dbt_krige_residual_predict <- function(coords_train, residuals_train, coords_new,
                                        fallback_bandwidth = NULL, chunk = 20000L,
                                        aniso_angle = 0, aniso_ratio = 1) {
  coords_train <- as.matrix(coords_train)
  coords_new <- as.matrix(coords_new)
  # Kriging under geometric anisotropy is ordinary kriging in the transformed
  # space, so we transform once here and let the variogram be fitted there.
  if (is.finite(aniso_ratio) && aniso_ratio < 1) {
    coords_train <- .dbt_aniso_transform(coords_train, aniso_angle, aniso_ratio)
    coords_new   <- .dbt_aniso_transform(coords_new,   aniso_angle, aniso_ratio)
  }
  r <- as.numeric(residuals_train)
  keep <- is.finite(r) & is.finite(coords_train[, 1]) & is.finite(coords_train[, 2])
  r <- r[keep]; coords_train <- coords_train[keep, , drop = FALSE]
  mu <- mean(r, na.rm = TRUE)
  fb <- function() {
    if (is.null(fallback_bandwidth) || !is.finite(fallback_bandwidth)) {
      return(rep(mu, nrow(coords_new)))
    }
    .dbt_kernel_residual_predict(coords_train, r, coords_new, fallback_bandwidth,
                                 chunk = chunk)
  }
  ok_pkg <- requireNamespace("gstat", quietly = TRUE) &&
    requireNamespace("sp", quietly = TRUE)
  if (!ok_pkg || length(r) < 20L || stats::sd(r, na.rm = TRUE) <= 1e-8) return(fb())

  D <- as.matrix(stats::dist(coords_train))
  cutoff <- max(D, na.rm = TRUE) * 0.50
  if (!is.finite(cutoff) || cutoff <= 0) return(fb())
  tr <- data.frame(r = r, x = coords_train[, 1], y2 = coords_train[, 2])
  sp::coordinates(tr) <- ~ x + y2
  vg_emp <- tryCatch(
    gstat::variogram(r ~ 1, data = tr, cutoff = cutoff, width = cutoff / 15,
                     cressie = TRUE), error = function(e) NULL)
  if (is.null(vg_emp) || nrow(vg_emp) < 3L) return(fb())
  sill <- stats::var(r, na.rm = TRUE)
  vg_fit <- tryCatch(suppressWarnings(gstat::fit.variogram(
    vg_emp, gstat::vgm(sill * 0.80, "Exp", cutoff * 0.30, sill * 0.20),
    fit.method = 7L)), error = function(e) NULL)
  if (is.null(vg_fit) || !any(is.finite(vg_fit$range) & vg_fit$range > 0)) return(fb())

  n_new <- nrow(coords_new)
  out <- numeric(n_new)
  for (s in seq.int(1L, n_new, by = chunk)) {
    e <- min(s + chunk - 1L, n_new)
    nd <- data.frame(x = coords_new[s:e, 1], y2 = coords_new[s:e, 2])
    sp::coordinates(nd) <- ~ x + y2
    kk <- tryCatch(suppressWarnings(gstat::krige(
      r ~ 1, tr, nd, model = vg_fit, debug.level = 0)), error = function(e) NULL)
    out[s:e] <- if (is.null(kk)) mu else as.numeric(kk$var1.pred)
  }
  out[!is.finite(out)] <- mu
  out
}

# Distance from each query location to its nearest reference (calibration) point.
# Drives the spatial confidence gate; chunked over queries for the full grid.
.dbt_nn_dist <- function(coords_query, coords_ref, chunk = 20000L) {
  coords_query <- as.matrix(coords_query)
  coords_ref <- as.matrix(coords_ref)
  cr2 <- rowSums(coords_ref ^ 2)
  n_q <- nrow(coords_query)
  out <- numeric(n_q)
  for (s in seq.int(1L, n_q, by = chunk)) {
    e <- min(s + chunk - 1L, n_q)
    cq <- coords_query[s:e, , drop = FALSE]
    d2 <- outer(rowSums(cq ^ 2), cr2, `+`) - 2 * cq %*% t(coords_ref)
    d2[d2 < 0] <- 0
    out[s:e] <- sqrt(apply(d2, 1L, min))
  }
  out
}

# Refit the base RF + a single residual-correction `mode` on a training subset
# and predict a held-out subset. Used by the spatial-CV gate selector; mirrors
# the cross-fit/apply correction logic for one mode.
# CV+ prediction intervals (Barber, Candes, Ramdas and Tibshirani, 2021).
#
# Why this replaces a pooled out-of-fold quantile. Split conformal requires the
# point prediction to come from a model that never saw the calibration set. The
# earlier construction centred the interval on the model refitted on the WHOLE
# calibration sample while calibrating the half-width on residuals pooled from K
# fold models, which is neither split conformal nor covered by its guarantee.
# CV+ instead evaluates, for every calibration point i, the model that excluded
# i's fold AT THE NEW LOCATION, and forms the interval from the n values
#
#     f_hat^(-k(i))(x) +/- R_i ,     R_i = |y_i - f_hat^(-k(i))(x_i)| ,
#
# taking the floor(alpha (n+1)) smallest of the lower set and the
# ceiling((1-alpha)(n+1)) smallest of the upper set. This carries a finite-sample
# marginal coverage guarantee of at least 1 - 2 alpha under exchangeability.
#
# The fold split used here is drawn INDEPENDENTLY of the one used to select the
# mode, shrinkage and gate. Selection and calibration therefore do not reuse the
# same residuals, which removes the second-order adaptivity that a single shared
# split would introduce.
.dbt_cvplus <- function(train_df, test_df, response_name, predictor_names,
                        coord_names, mode, lambda, base_params, n_basis,
                        bandwidth, use_spatial, k = 5L, seed = 1L,
                        alpha = 0.10, max_test = 20000L) {
  n <- nrow(train_df); n_test <- nrow(test_df)
  if (n < 20L || n_test < 1L) return(NULL)
  # the dense n x n_test construction is only affordable for moderate test sets
  if (n_test > max_test) return(NULL)
  lam <- if (is.finite(lambda)) lambda else 0

  fold <- sample(rep_len(seq_len(k), n))
  y <- as.numeric(train_df[[response_name]])
  R <- rep(NA_real_, n)
  P <- matrix(NA_real_, nrow = n, ncol = n_test)

  keep <- intersect(names(train_df), names(test_df))
  for (kk in seq_len(k)) {
    te <- which(fold == kk); tr <- which(fold != kk)
    if (length(tr) < 20L || length(te) < 1L) next
    va <- rbind(train_df[te, keep, drop = FALSE], test_df[, keep, drop = FALSE])
    bc <- tryCatch(.dbt_block_base_corr(
      train_df[tr, , drop = FALSE], va, response_name, predictor_names,
      coord_names, mode, base_params, n_basis, bandwidth, use_spatial,
      seed + kk), error = function(e) NULL)
    if (is.null(bc)) next
    pred <- bc$base + lam * bc$corr
    n_te <- length(te)
    R[te] <- abs(y[te] - pred[seq_len(n_te)])
    P[te, ] <- matrix(pred[-seq_len(n_te)], nrow = n_te, ncol = n_test,
                      byrow = TRUE)
  }

  ok <- is.finite(R) & stats::complete.cases(P)
  m <- sum(ok)
  if (m < 10L) return(NULL)
  lo_i <- max(1L, floor(alpha * (m + 1)))
  hi_i <- min(m, ceiling((1 - alpha) * (m + 1)))
  Lo <- P[ok, , drop = FALSE] - R[ok]
  Hi <- P[ok, , drop = FALSE] + R[ok]
  list(lower = apply(Lo, 2L, function(v) sort(v)[lo_i]),
       upper = apply(Hi, 2L, function(v) sort(v)[hi_i]),
       n_used = m, alpha = alpha)
}


.dbt_block_base_corr <- function(tr, va, response_name, predictor_names, coord_names,
                                 mode, base_params, n_basis, bandwidth,
                                 use_spatial, seed,
                                 use_aniso = FALSE, bw_mult = 0.5,
                                 vg_cutoff = 0.50, vg_lags = 15L,
                                 vg_cressie = TRUE) {
  rf <- .dbt_fit_rf(tr, response_name, predictor_names, rf_params = base_params,
                    seed = seed)
  base_tr <- .dbt_oob_or_in_sample(rf, tr)
  base_va <- .dbt_predict_rf(rf, va)
  resid_tr <- as.numeric(tr[[response_name]]) - base_tr
  if (identical(mode, "bias")) {
    return(list(base = base_va, corr = rep(mean(resid_tr, na.rm = TRUE), nrow(va))))
  }
  x_tr <- as.data.frame(tr[, predictor_names, drop = FALSE])
  x_va <- as.data.frame(va[, predictor_names, drop = FALSE])
  cb <- .dbt_fit_cubist(x_tr, tr[[response_name]])
  cb_tr <- .dbt_predict_cubist(cb, x_tr); cb_va <- .dbt_predict_cubist(cb, x_va)
  if (identical(mode, "cubist_delta")) {
    return(list(base = base_va, corr = cb_va - base_va))
  }
  basis <- NULL; coords_tr <- coords_va <- NULL
  if (use_spatial) {
    coords_tr <- as.matrix(tr[, coord_names, drop = FALSE])
    coords_va <- as.matrix(va[, coord_names, drop = FALSE])
    basis <- .dbt_fit_basis(coords_tr, resid_tr, n_basis = n_basis,
                            bandwidth = bandwidth, use_aniso = use_aniso, bw_mult = bw_mult,
                                                            vg_cutoff = vg_cutoff, vg_lags = vg_lags, vg_cressie = vg_cressie, seed = seed)
  }
  feat_tr <- .dbt_residual_features(tr, predictor_names, coord_names, base_tr, cb_tr, basis)
  feat_va <- .dbt_residual_features(va, predictor_names, coord_names, base_va, cb_va, basis)
  fit_rrf <- function() .dbt_tune_rf_model(
    cbind(as.data.frame(feat_tr), .resid = as.numeric(resid_tr)), ".resid",
    names(as.data.frame(feat_tr)), tune = FALSE, tune_method = "none", seed = seed)
  corr <- switch(mode,
    resid_cubist = .dbt_predict_cubist(.dbt_fit_cubist(feat_tr, resid_tr), feat_va),
    resid_rf = .dbt_predict_residual_rf(fit_rrf()$model, feat_va),
    resid_mlp = { pm <- .dbt_predict_mlp(.dbt_fit_mlp(feat_tr, resid_tr, seed = seed), feat_va)
                  if (is.null(pm)) rep(0, nrow(va)) else pm },
    resid_spatial = if (use_spatial) .dbt_kernel_residual_predict(
      coords_tr, resid_tr, coords_va, basis$bandwidth,
      aniso_angle = basis$aniso_angle, aniso_ratio = basis$aniso_ratio) else rep(0, nrow(va)),
    resid_kriging = if (use_spatial) .dbt_krige_residual_predict(
      coords_tr, resid_tr, coords_va, fallback_bandwidth = basis$bandwidth,
      aniso_angle = basis$aniso_angle, aniso_ratio = basis$aniso_ratio) else rep(0, nrow(va)),
    resid_ensemble = {
      parts <- list(rf = .dbt_predict_residual_rf(fit_rrf()$model, feat_va),
                    cb = .dbt_predict_cubist(.dbt_fit_cubist(feat_tr, resid_tr), feat_va))
      if (use_spatial) parts$sp <- .dbt_kernel_residual_predict(
        coords_tr, resid_tr, coords_va, basis$bandwidth,
        aniso_angle = basis$aniso_angle, aniso_ratio = basis$aniso_ratio)
      rowMeans(as.data.frame(parts), na.rm = TRUE)
    },
    rep(0, nrow(va)))
  corr <- as.numeric(corr); corr[!is.finite(corr)] <- 0
  list(base = base_va, corr = corr)
}

# Select the spatial-confidence gate factor by SPATIAL cross-validation: split the
# calibration sample into k spatial blocks (k-means on coordinates), refit base +
# the selected correction on the other blocks, predict each held-out block (an
# extrapolation regime), and choose the gate length that minimises spatially-
# blocked RMSE. Returns gate_c = Inf (no gate) if nothing beats the ungated fit.
.dbt_select_gate_spatial <- function(train_df, response_name, predictor_names,
                                     coord_names, mode, lambda, base_params,
                                     n_basis, bandwidth, gate_rho,
                                     k_blocks = 4L, seed = 1L) {
  if (!is.finite(gate_rho) || gate_rho <= 0 || !is.finite(lambda)) {
    return(list(gate_c = Inf, gate_length = Inf))
  }
  # Cap trees for the block refits: the gate only needs the length scale, not a
  # full-fidelity forest, so this keeps the spatial-CV selection affordable.
  if (!is.null(base_params)) {
    base_params$num.trees <- as.integer(min(as.integer(base_params$num.trees), 300L))
  }
  coords <- as.matrix(train_df[, coord_names, drop = FALSE])
  n <- nrow(train_df)
  k <- as.integer(max(2L, min(k_blocks, nrow(unique(round(coords, 6))) - 1L)))
  set.seed(as.integer(seed))
  km <- tryCatch(stats::kmeans(coords, centers = k, iter.max = 50L,
                               nstart = 3L)$cluster, error = function(e) NULL)
  if (is.null(km)) return(list(gate_c = Inf, gate_length = Inf))
  y <- as.numeric(train_df[[response_name]])
  base_sp <- rep(NA_real_, n); corr_sp <- rep(NA_real_, n); d1 <- rep(NA_real_, n)
  for (b in sort(unique(km))) {
    te_idx <- which(km == b); tr_idx <- which(km != b)
    if (length(tr_idx) < 20L || length(te_idx) < 1L) next
    bc <- tryCatch(.dbt_block_base_corr(
      train_df[tr_idx, , drop = FALSE], train_df[te_idx, , drop = FALSE],
      response_name, predictor_names, coord_names, mode, base_params,
      n_basis, bandwidth, TRUE, seed + b), error = function(e) NULL)
    if (is.null(bc)) next
    base_sp[te_idx] <- bc$base; corr_sp[te_idx] <- bc$corr
    d1[te_idx] <- .dbt_nn_dist(coords[te_idx, , drop = FALSE],
                               coords[tr_idx, , drop = FALSE])
  }
  ok <- is.finite(base_sp) & is.finite(corr_sp) & is.finite(d1) & is.finite(y)
  if (sum(ok) < 20L) return(list(gate_c = Inf, gate_length = Inf))
  yy <- y[ok]; bb <- base_sp[ok]; cv <- corr_sp[ok]; dd <- d1[ok]
  best_rmse <- .dbt_rmse(yy, bb + lambda * cv)
  gc <- Inf; gl <- Inf
  for (cf in c(3, 2, 1.5, 1, 0.75, 0.5, 0.25, 0.1)) {
    w <- exp(-dd / (cf * gate_rho))
    rmg <- .dbt_rmse(yy, bb + lambda * w * cv)
    if (is.finite(rmg) && rmg < best_rmse - 1e-9) {
      best_rmse <- rmg; gc <- cf; gl <- cf * gate_rho
    }
  }
  list(gate_c = gc, gate_length = gl)
}

# Density-ratio weights w_i ∝ p_pop/p_sample, by classification (calibration vs a
# population reference). Used for design-aware (weighted) conformal calibration so
# prediction-interval coverage is restored under clustered/biased sampling
# (Tibshirani et al. 2019, conformal under covariate shift). RNG preserved.
.dbt_density_ratio_weights <- function(train_df, ref_df, predictor_names, coord_names,
                                       seed = 1L, wmax = 20) {
  n <- nrow(train_df)
  if (!requireNamespace("ranger", quietly = TRUE) || is.null(ref_df) ||
      nrow(ref_df) < n) return(rep(1, n))
  feats <- c(predictor_names, coord_names)
  feats <- feats[feats %in% names(train_df) & feats %in% names(ref_df)]
  if (length(feats) < 2L) return(rep(1, n))
  withr::local_preserve_seed()
  set.seed(as.integer(seed))
  m <- min(5L * n, nrow(ref_df))
  ref_idx <- sample.int(nrow(ref_df), m)
  Xc <- as.data.frame(train_df[, feats, drop = FALSE])
  Xp <- as.data.frame(ref_df[ref_idx, feats, drop = FALSE])
  dat <- rbind(cbind(Xc, .lab = "calib"), cbind(Xp, .lab = "pop"))
  dat$.lab <- factor(dat$.lab, levels = c("pop", "calib"))
  cls <- tryCatch(ranger::ranger(.lab ~ ., data = dat, probability = TRUE,
                                 num.threads = 1L, seed = as.integer(seed)),
                  error = function(e) NULL)
  if (is.null(cls)) return(rep(1, n))
  ph <- tryCatch(predict(cls, data = Xc, num.threads = 1L)$predictions[, "calib"],
                 error = function(e) NULL)
  if (is.null(ph) || any(!is.finite(ph))) return(rep(1, n))
  ph <- pmin(pmax(ph, 1e-4), 1 - 1e-4)
  w <- (1 - ph) / ph
  w[!is.finite(w) | w < 0] <- 0
  if (mean(w) <= 0) return(rep(1, n))
  w <- w / mean(w); w <- pmin(w, wmax); w / mean(w)
}

# Spatial-block conformal calibration residuals: refit base + the selected
# correction on spatial k-means blocks and predict each held-out block (an
# extrapolation regime); the resulting |residuals| are larger than random-fold
# residuals, so their quantile gives intervals that restore design-based coverage
# under clustered sampling. Trees are capped for affordability.
.dbt_spatial_conformal_resid <- function(train_df, response_name, predictor_names,
                                         coord_names, mode, lambda, base_params,
                                         n_basis, bandwidth, k_blocks = 5L, seed = 1L) {
  if (length(coord_names) != 2L) return(NULL)
  coords <- as.matrix(train_df[, coord_names, drop = FALSE])
  n <- nrow(train_df)
  if (!is.null(base_params)) {
    base_params$num.trees <- as.integer(min(as.integer(base_params$num.trees), 300L))
  }
  k <- as.integer(max(2L, min(k_blocks, nrow(unique(round(coords, 6))) - 1L)))
  set.seed(as.integer(seed))
  km <- tryCatch(stats::kmeans(coords, centers = k, iter.max = 50L,
                               nstart = 3L)$cluster, error = function(e) NULL)
  if (is.null(km)) return(NULL)
  y <- as.numeric(train_df[[response_name]])
  res <- rep(NA_real_, n)
  lam <- if (is.finite(lambda)) lambda else 0
  for (b in sort(unique(km))) {
    te <- which(km == b); tr <- which(km != b)
    if (length(tr) < 20L || length(te) < 1L) next
    bc <- tryCatch(.dbt_block_base_corr(
      train_df[tr, , drop = FALSE], train_df[te, , drop = FALSE],
      response_name, predictor_names, coord_names, mode, base_params,
      n_basis, bandwidth, TRUE, seed + b), error = function(e) NULL)
    if (is.null(bc)) next
    res[te] <- y[te] - (bc$base + lam * bc$corr)
  }
  ra <- abs(res)
  ra[is.finite(ra)]
}

# Weighted empirical quantile (weights need not sum to 1).
.dbt_wquantile <- function(v, w, p) {
  ok <- is.finite(v) & is.finite(w) & w >= 0
  if (sum(ok) < 5L || sum(w[ok]) <= 0) return(NA_real_)
  v <- v[ok]; w <- w[ok]; o <- order(v); v <- v[o]; w <- w[o]
  cw <- cumsum(w) / sum(w)
  v[which(cw >= p)[1L]]
}

# Honest out-of-fold predictions for an lm-based recalibration candidate, using
# the SAME folds as the residual candidates so model selection compares OOF to OOF
# (not resubstitution to OOF). The deploy step still re-fits on the full cal_df,
# which is the standard stacked-generalization protocol.
.dbt_oof_calibrate <- function(cal_df, folds, form) {
  pred <- rep(NA_real_, nrow(cal_df))
  for (j in seq_along(folds)) {
    idx <- folds[[j]]
    fit <- tryCatch(stats::lm(form, data = cal_df[-idx, , drop = FALSE]),
                    error = function(e) NULL)
    if (is.null(fit)) next
    pred[idx] <- suppressWarnings(
      as.numeric(stats::predict(fit, newdata = cal_df[idx, , drop = FALSE])))
  }
  pred
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
                                      use_aniso = FALSE,
                                      bw_mult = 0.5,
                                      vg_cutoff = 0.50,
                                      vg_lags = 15L,
                                      vg_cressie = TRUE,
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

  withr::local_preserve_seed()

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
    resid_mlp = rep(NA_real_, n)
  )
  if (use_spatial) {
    corr_oof$resid_spatial <- rep(NA_real_, n)
    corr_oof$resid_kriging <- rep(NA_real_, n)
  }
  fold_rows <- list()
  rf_tune_rows <- list()

  message(sprintf("[DBTARF] cross-fitting residual candidates (n=%d, folds=%d)...",
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
                                bandwidth = bandwidth, use_aniso = use_aniso, bw_mult = bw_mult,
                                vg_cutoff = vg_cutoff, vg_lags = vg_lags, vg_cressie = vg_cressie,
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

    resid_mlp_j <- .dbt_fit_mlp(feat_tr, resid_tr, seed = train_seed + 3000L * j)
    if (!is.null(resid_mlp_j)) {
      pm <- .dbt_predict_mlp(resid_mlp_j, feat_va)
      if (!is.null(pm)) corr_oof$resid_mlp[id_val] <- pm
    }

    if (use_spatial) {
      corr_oof$resid_spatial[id_val] <- .dbt_kernel_residual_predict(
        coords_tr, resid_tr, coords_va, basis_j$bandwidth,
        aniso_angle = basis_j$aniso_angle, aniso_ratio = basis_j$aniso_ratio)
      corr_oof$resid_kriging[id_val] <- .dbt_krige_residual_predict(
        coords_tr, resid_tr, coords_va, fallback_bandwidth = basis_j$bandwidth,
        aniso_angle = basis_j$aniso_angle, aniso_ratio = basis_j$aniso_ratio)
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
    p_lm1 <- .dbt_oof_calibrate(cal_df, folds, y ~ rf)
    candidate_rows[[length(candidate_rows) + 1L]] <- data.frame(
      mode = "linear_cal", lambda = NA_real_, oof_rmse = .dbt_rmse(y, p_lm1),
      delta_rmse = .dbt_rmse(y, p_lm1) - baseline_rmse,
      stringsAsFactors = FALSE)

    p_lm2 <- .dbt_oof_calibrate(cal_df, folds, y ~ rf + I(rf ^ 2))
    candidate_rows[[length(candidate_rows) + 1L]] <- data.frame(
      mode = "quadratic_cal", lambda = NA_real_, oof_rmse = .dbt_rmse(y, p_lm2),
      delta_rmse = .dbt_rmse(y, p_lm2) - baseline_rmse,
      stringsAsFactors = FALSE)
  }
  if (sum(is.finite(cal_df$rf) & is.finite(cal_df$cubist)) >= 20L) {
    p_stk <- .dbt_oof_calibrate(cal_df, folds, y ~ rf + cubist)
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

  # Spatially-adaptive shrinkage gate (novel component): the gate strength is
  # selected by SPATIAL cross-validation in the apply step below, where the base
  # RF configuration is available. Initialised to "off" here.
  additive_modes <- c("bias", "cubist_delta", "resid_rf", "resid_cubist",
                      "resid_mlp", "resid_spatial", "resid_kriging", "resid_ensemble")
  gate_c <- Inf; gate_length <- Inf; gate_rho <- NA_real_
  best_mode_chr <- as.character(best$mode)

  message(sprintf("[DBTARF] OOF RF RMSE=%.3f | selected=%s lambda=%s OOF RMSE=%.3f",
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
        n_basis = n_basis, bandwidth = bandwidth, seed = train_seed,
        use_aniso = use_aniso, bw_mult = bw_mult,
        vg_cutoff = vg_cutoff, vg_lags = vg_lags, vg_cressie = vg_cressie)
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
    } else if (mode == "resid_mlp") {
      mdl <- .dbt_fit_mlp(feat_train, residual_full, seed = train_seed + 970000L)
      pm <- .dbt_predict_mlp(mdl, feat_test)
      if (!is.null(pm)) selected_correction_test <- pm
    } else if (mode == "resid_spatial" && use_spatial) {
      selected_correction_test <- .dbt_kernel_residual_predict(
        as.matrix(train_df[, coord_names, drop = FALSE]), residual_full,
        as.matrix(test_df[, coord_names, drop = FALSE]), basis_full$bandwidth,
        aniso_angle = basis_full$aniso_angle, aniso_ratio = basis_full$aniso_ratio)
    } else if (mode == "resid_kriging" && use_spatial) {
      selected_correction_test <- .dbt_krige_residual_predict(
        as.matrix(train_df[, coord_names, drop = FALSE]), residual_full,
        as.matrix(test_df[, coord_names, drop = FALSE]),
        fallback_bandwidth = basis_full$bandwidth,
        aniso_angle = basis_full$aniso_angle, aniso_ratio = basis_full$aniso_ratio)
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
          as.matrix(test_df[, coord_names, drop = FALSE]), basis_full$bandwidth,
          aniso_angle = basis_full$aniso_angle, aniso_ratio = basis_full$aniso_ratio)
      }
      if (length(parts) == 0L) {
        selected_correction_test <- rep(0, length(base_test))
      } else {
        selected_correction_test <- rowMeans(as.data.frame(parts), na.rm = TRUE)
        selected_correction_test[!is.finite(selected_correction_test)] <- 0
      }
    }
    selected_correction_test[!is.finite(selected_correction_test)] <- 0
    if (use_spatial && length(coord_names) == 2L &&
        best_mode_chr %in% additive_modes && is.finite(lambda)) {
      vg_gate <- .dbt_fit_variogram(
        residual_full, as.matrix(train_df[, coord_names, drop = FALSE]))
      gate_rho <- if (!is.null(vg_gate)) vg_gate$range_major else NA_real_
      if (is.finite(gate_rho) && gate_rho > 0) {
        gsel <- .dbt_select_gate_spatial(
          train_df, response_name, predictor_names, coord_names,
          best_mode_chr, lambda, base_params = rf_full_fit$params,
          n_basis = n_basis, bandwidth = bandwidth, gate_rho = gate_rho,
          k_blocks = 4L, seed = train_seed + 7L)
        gate_c <- gsel$gate_c; gate_length <- gsel$gate_length
      }
    }
    gate_w_test <- rep(1, length(base_test))
    if (is.finite(gate_length) && length(coord_names) == 2L) {
      d_test <- .dbt_nn_dist(as.matrix(test_df[, coord_names, drop = FALSE]),
                             as.matrix(train_df[, coord_names, drop = FALSE]))
      gate_w_test <- exp(-d_test / gate_length)
    }
    pred_test <- base_test + lambda * gate_w_test * selected_correction_test
  }

  if (any(!is.finite(pred_test))) pred_test[!is.finite(pred_test)] <- base_test[!is.finite(pred_test)]

  test_obs <- if (response_name %in% names(test_df)) {
    suppressWarnings(as.numeric(test_df[[response_name]]))
  } else rep(NA_real_, nrow(test_df))
  has_test_obs <- any(is.finite(test_obs))

  # ---- Conformal prediction quantiles from cross-fitted (OOF) residuals -------
  # The OOF prediction of the DEPLOYED configuration gives leakage-free residuals;
  # their absolute conformal quantile yields prediction intervals ŷ(s) ± q with
  # valid marginal coverage, which becomes design-based coverage on a probability
  # sample (extends the Wadoux design-based assessment from RMSE to intervals).
  oof_pred <- base_oof
  if (mode %in% c("linear_cal", "quadratic_cal", "stack_rf_cubist")) {
    oof_pred <- switch(mode,
      linear_cal = .dbt_oof_calibrate(cal_df, folds, y ~ rf),
      quadratic_cal = .dbt_oof_calibrate(cal_df, folds, y ~ rf + I(rf ^ 2)),
      stack_rf_cubist = .dbt_oof_calibrate(cal_df, folds, y ~ rf + cubist))
  } else if (mode != "RF_noharm" && !is.null(valid_corr[[mode]]) &&
             is.finite(lambda)) {
    cc <- as.numeric(valid_corr[[mode]]); cc[!is.finite(cc)] <- 0
    oof_pred <- base_oof + lambda * cc
  }
  resid_oof <- abs(y - oof_pred)
  finite_r <- is.finite(resid_oof)
  .dbt_conf_q <- function(a) {
    rr <- resid_oof[finite_r]
    if (length(rr) < 5L) return(NA_real_)
    k <- min(length(rr), ceiling((length(rr) + 1) * (1 - a)))
    sort(rr)[k]
  }
  conformal_q90 <- .dbt_conf_q(0.10)
  conformal_q95 <- .dbt_conf_q(0.05)
  # Design-aware (weighted) conformal: density-ratio weights (calibration vs the
  # population-reference covariates of test_df) restore design-based coverage under
  # clustered/biased sampling, where plain conformal under-covers.
  dr_w <- .dbt_density_ratio_weights(train_df, test_df, predictor_names,
                                     coord_names, seed = train_seed + 4242L)
  conformal_q90_w <- .dbt_wquantile(resid_oof[finite_r], dr_w[finite_r], 0.90)
  conformal_q95_w <- .dbt_wquantile(resid_oof[finite_r], dr_w[finite_r], 0.95)
  if (!is.finite(conformal_q90_w)) conformal_q90_w <- conformal_q90
  if (!is.finite(conformal_q95_w)) conformal_q95_w <- conformal_q95
  # Spatial-block conformal: calibrate on extrapolation-aware residuals so the
  # interval widens appropriately under clustered sampling (restores coverage).
  conformal_q90_sp <- conformal_q90; conformal_q95_sp <- conformal_q95
  if (use_spatial && (mode %in% additive_modes || mode == "RF_noharm")) {
    rsp <- .dbt_spatial_conformal_resid(train_df, response_name, predictor_names,
             coord_names, mode, lambda, rf_full_fit$params, n_basis, bandwidth,
             k_blocks = 5L, seed = train_seed + 8888L)
    if (!is.null(rsp) && length(rsp) >= 20L) {
      .qsp <- function(a) {
        k <- min(length(rsp), ceiling((length(rsp) + 1) * (1 - a)))
        sort(rsp)[k]
      }
      conformal_q90_sp <- .qsp(0.10); conformal_q95_sp <- .qsp(0.05)
    }
  }

  list(
    pred_test = as.numeric(pred_test),
    pred_test_base = as.numeric(base_test),
    pred_test_cubist = as.numeric(cubist_test),
    correction_test = as.numeric(selected_correction_test),
    pi_q90 = conformal_q90,
    pi_q95 = conformal_q95,
    pi_q90_w = conformal_q90_w,
    pi_q95_w = conformal_q95_w,
    pi_q90_sp = conformal_q90_sp,
    pi_q95_sp = conformal_q95_sp,
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
      gate_c = gate_c,
      gate_length = gate_length,
      gate_rho = gate_rho,
      gate_active = is.finite(gate_length),
      conformal_q90 = conformal_q90,
      conformal_q95 = conformal_q95,
      conformal_q90_w = conformal_q90_w,
      conformal_q95_w = conformal_q95_w,
      conformal_q90_sp = conformal_q90_sp,
      conformal_q95_sp = conformal_q95_sp,
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
                                  aniso_search = TRUE,
                                  bw_search = TRUE,
                                  vg_search = TRUE,
                                  fair_profile_max = 6L,
                                  ensemble_top_k = 1L,
                                  ensemble_weighting = "softmax",
                                  ensemble_temperature = 0.75,
                                  ...) {
  tag_profile <- function(out, label) {
    out$diagnostics$selected_profile <- label
    out$candidate_table <- cbind(profile = label, out$candidate_table)
    out$fold_diagnostics <- cbind(profile = label, out$fold_diagnostics)
    out$rf_tuning_table <- cbind(profile = label, out$rf_tuning_table)
    out
  }

  # Pure deterministic map from per-profile OOF-RMSE scores (lower = better)
  # to normalized blend weights. No RNG; weights depend only on `scores`.
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
      use_aniso = isTRUE(spec$use_aniso),
      bw_mult = if (is.null(spec$bw_mult) || !is.finite(spec$bw_mult)) 0.5 else spec$bw_mult,
      vg_cutoff = if (is.null(spec$vg_cutoff)) 0.50 else spec$vg_cutoff,
      vg_lags = if (is.null(spec$vg_lags)) 15L else as.integer(spec$vg_lags),
      vg_cressie = if (is.null(spec$vg_cressie)) TRUE else isTRUE(spec$vg_cressie),
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
    profile_specs$use_aniso <- FALSE
    profile_specs$bw_mult <- 0.5
    profile_specs$vg_cutoff <- 0.50
    profile_specs$vg_lags <- 15L
    profile_specs$vg_cressie <- TRUE

    # Binning and estimator cannot be compared by variogram SSE (a different
    # cutoff or lag width changes the empirical variogram itself, so the SSEs
    # are computed on different quantities). They therefore enter the profile
    # search and are decided by out-of-fold predictive merit, like everything
    # else in the method.
    if (.dbt_as_bool(vg_search) && length(coord_for_specs) == 2L) {
      vg_variants <- list(
        list(cutoff = 0.35, lags = 15L, cressie = TRUE,  tag = "vgc35"),
        list(cutoff = 0.65, lags = 20L, cressie = TRUE,  tag = "vgc65"),
        list(cutoff = 0.50, lags = 15L, cressie = FALSE, tag = "vgmath"))
      vg_specs <- do.call(rbind, lapply(vg_variants, function(v) {
        z <- profile_specs[seq_len(min(1L, nrow(profile_specs))), , drop = FALSE]
        z$vg_cutoff <- v$cutoff; z$vg_lags <- v$lags; z$vg_cressie <- v$cressie
        z$profile <- paste0(z$profile, "_", v$tag)
        z
      }))
      profile_specs <- rbind(profile_specs, vg_specs)
    }

    # The variogram RANGE enters the search as a tuned scale rather than a
    # derived constant: the top profiles are duplicated with a shorter and a
    # longer radial-basis bandwidth (bw_mult x range), so how far the
    # geostatistical structure is trusted is decided out-of-fold like any other
    # hyperparameter. bw_mult = 0.5 is the previous derived value and is kept as
    # the default arm.
    if (.dbt_as_bool(bw_search) && length(coord_for_specs) == 2L) {
      bw_specs <- do.call(rbind, lapply(c(0.25, 1.0), function(m) {
        z <- profile_specs[seq_len(min(2L, nrow(profile_specs))), , drop = FALSE]
        z$bw_mult <- m
        z$profile <- sprintf("%s_bw%02d", z$profile, round(m * 100))
        z
      }))
      profile_specs <- rbind(profile_specs, bw_specs)
    }

    # Anisotropy is admitted on merit, exactly like every other component: the
    # best isotropic profile is duplicated with the fitted anisotropic geometry
    # and the two compete on the same out-of-fold criterion. If the residual
    # field has no directional structure the variogram search already returns
    # ratio = 1 (the transform is then a pure rotation and the two profiles are
    # equivalent), and if it has but the geometry does not pay off out-of-fold,
    # the isotropic profile simply wins.
    if (.dbt_as_bool(aniso_search) && length(coord_for_specs) == 2L) {
      aniso_specs <- profile_specs[seq_len(min(2L, nrow(profile_specs))), , drop = FALSE]
      aniso_specs$use_aniso <- TRUE
      aniso_specs$profile <- paste0(aniso_specs$profile, "_aniso")
      profile_specs <- rbind(profile_specs, aniso_specs)
    }
    profile_out <- vector("list", nrow(profile_specs))
    for (i in seq_len(nrow(profile_specs))) {
      # Isolate each profile: one degenerate profile/fold must not abort the run.
      profile_out[[i]] <- tryCatch(
        run_fixed_profile(profile_specs[i, , drop = FALSE]),
        error = function(e) NULL)
    }
    ok_profiles <- !vapply(profile_out, is.null, logical(1))
    if (!any(ok_profiles)) {
      stop("DBTARF fair_profile_search: all ", nrow(profile_specs),
           " profiles failed to fit.")
    }
    profile_out <- profile_out[ok_profiles]
    profile_labels <- vapply(profile_out,
      function(x) as.character(x$diagnostics$selected_profile), character(1))
    oof <- vapply(profile_out, function(x) {
      suppressWarnings(as.numeric(x$diagnostics$selected_oof_rmse))
    }, numeric(1))
    best_i <- if (any(is.finite(oof))) which.min(oof) else 1L
    selected <- profile_out[[best_i]]

    # ---- top-k profile ensemble (no-op when ensemble_top_k <= 1) ----
    ens_top_k <- suppressWarnings(as.integer(ensemble_top_k))
    if (length(ens_top_k) != 1L || is.na(ens_top_k) ||
        !is.finite(ens_top_k) || ens_top_k < 1L) ens_top_k <- 1L
    rank_i <- order(oof, na.last = TRUE)
    rank_i <- rank_i[is.finite(oof[rank_i])]
    if (length(rank_i) == 0L) rank_i <- best_i
    top_i <- rank_i[seq_len(min(ens_top_k, length(rank_i)))]
    top_weights <- dbt_candidate_weights(
      oof[top_i], ensemble_weighting, ensemble_temperature)
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
    # Effective (post-normalization) weighting/temperature actually used by
    # dbt_candidate_weights, so the recorded provenance matches the weights.
    eff_weighting <- tolower(trimws(as.character(ensemble_weighting)[1L]))
    if (!(eff_weighting %in% c("uniform", "inverse"))) eff_weighting <- "softmax"
    eff_temperature <- suppressWarnings(as.numeric(ensemble_temperature)[1L])
    if (!is.finite(eff_temperature) || eff_temperature <= 0) eff_temperature <- 1
    selected$diagnostics$ensemble_top_k_requested <- ens_top_k
    selected$diagnostics$ensemble_applied <- length(top_i) > 1L
    selected$diagnostics$ensemble_size <- length(top_i)
    # When blending, pi_* conformal half-widths are inherited from the single
    # best profile and are NOT recalibrated against the blended pred_test
    # (see ?dbtarf_train_predict).
    selected$diagnostics$ensemble_pi_from_best <- length(top_i) > 1L
    selected$diagnostics$ensemble_weighting <- eff_weighting
    selected$diagnostics$ensemble_temperature <- eff_temperature
    selected$diagnostics$ensemble_profiles <- paste(
      profile_labels[top_i], collapse = "|")
    selected$diagnostics$ensemble_profile_oof_rmse <- paste(
      sprintf("%.8f", oof[top_i]), collapse = "|")
    selected$diagnostics$ensemble_weights <- paste(
      sprintf("%.8f", top_weights), collapse = "|")

    selected$diagnostics$fair_profile_search <- TRUE
    selected$diagnostics$fair_profile_max <- as.integer(fair_profile_max)
    selected$diagnostics$profile_search_n <- length(profile_out)
    selected$diagnostics$profile_search_best_oof_rmse <- oof[best_i]
    selected$diagnostics$default_profile_oof_rmse <- {
      di <- match("default_rf", profile_labels)
      if (is.na(di)) NA_real_ else oof[di]
    }
    selected$candidate_table <- do.call(rbind, lapply(profile_out, `[[`, "candidate_table"))
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

  use_aniso_requested <- FALSE
  requested_label <- if (.dbt_as_bool(rf_tune) || .dbt_as_bool(resid_rf_tune)) {
    "single_tuned"
  } else {
    "default_rf"
  }
  requested <- .dbt_train_dbtarf_profile(
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

  requested <- tag_profile(requested, requested_label)
  requested$diagnostics$fair_profile_search <- FALSE
  requested$diagnostics$fair_profile_max <- as.integer(fair_profile_max)
  requested$diagnostics$profile_search_n <- NA_integer_
  requested$diagnostics$profile_search_best_oof_rmse <-
    requested$diagnostics$selected_oof_rmse
  requested$diagnostics$default_profile_oof_rmse <-
    if (identical(requested_label, "default_rf")) {
      requested$diagnostics$selected_oof_rmse
    } else NA_real_

  # Mirror the ensemble diagnostics schema of the search path so both no-op
  # return paths expose the same fields (the single-profile path never blends).
  ens_eff_weighting <- tolower(trimws(as.character(ensemble_weighting)[1L]))
  if (!(ens_eff_weighting %in% c("uniform", "inverse")))
    ens_eff_weighting <- "softmax"
  ens_eff_temperature <- suppressWarnings(as.numeric(ensemble_temperature)[1L])
  if (!is.finite(ens_eff_temperature) || ens_eff_temperature <= 0)
    ens_eff_temperature <- 1
  ens_k_req <- suppressWarnings(as.integer(ensemble_top_k)[1L])
  if (length(ens_k_req) != 1L || is.na(ens_k_req) ||
      !is.finite(ens_k_req) || ens_k_req < 1L) ens_k_req <- 1L
  requested$diagnostics$ensemble_top_k_requested <- ens_k_req
  requested$diagnostics$ensemble_applied <- FALSE
  requested$diagnostics$ensemble_size <- 1L
  requested$diagnostics$ensemble_pi_from_best <- FALSE
  requested$diagnostics$ensemble_weighting <- ens_eff_weighting
  requested$diagnostics$ensemble_temperature <- ens_eff_temperature
  requested$diagnostics$ensemble_profiles <- requested_label
  requested$diagnostics$ensemble_profile_oof_rmse <- sprintf(
    "%.8f", suppressWarnings(as.numeric(requested$diagnostics$selected_oof_rmse)))
  requested$diagnostics$ensemble_weights <- "1.00000000"
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
  aniso_search = TRUE,
  bw_search = TRUE,
  vg_search = TRUE,
  ensemble_top_k = 1L,
  ensemble_weighting = "softmax",
  ensemble_temperature = 0.75
)

#' Fit DB-TARF and predict a test set
#'
#' DB-TARF (Design-Based Targeted Adaptive Residual Forest) fits a random-forest
#' base learner on \code{train_df} and adds a cross-fitted, out-of-fold-selected
#' residual correction, then predicts \code{test_df}. A correction is adopted
#' only when it lowers the out-of-fold RMSE of the calibration sample.
#'
#' @param train_df A data frame with the response, predictors and (optionally)
#'   coordinate columns.
#' @param test_df A data frame with the same predictor (and coordinate) columns
#'   as \code{train_df}.
#' @param response_name Character; name of the response column in \code{train_df}.
#' @param predictor_names Character vector of predictor column names, or
#'   \code{NULL} (default) to infer them from \code{train_df}.
#' @param coord_names Length-2 character vector of coordinate column names, or
#'   \code{"auto"} (default) to detect them.
#' @param ... Further arguments passed to the internal training routine (e.g.
#'   \code{n_folds}, \code{lambda_grid}, \code{rf_tune}, \code{train_seed}), and
#'   the top-k profile-ensemble controls \code{ensemble_top_k} (integer
#'   \eqn{\ge 1}, default \code{1}: the single best profile, i.e. no blending),
#'   \code{ensemble_weighting} (\code{"softmax"}, \code{"inverse"} or
#'   \code{"uniform"}) and \code{ensemble_temperature} (positive, default
#'   \code{0.75}); see \code{dbtarf_default_params}. The ensemble blends the
#'   top-k RF profiles (ranked by out-of-fold RMSE) and only activates when
#'   \code{fair_profile_search} is \code{TRUE}, (\code{rf_tune} or
#'   \code{resid_rf_tune}) is \code{TRUE} and \code{ensemble_top_k > 1}.
#' @return A list with the test predictions (\code{pred_test}), the base-RF
#'   predictions (\code{pred_test_base}), conformal prediction-interval
#'   half-widths (\code{pi_q90}, \code{pi_q95}), per-run \code{diagnostics} and
#'   the \code{candidate_table}. When \code{ensemble_top_k > 1} the
#'   \code{diagnostics} additionally record \code{ensemble_applied},
#'   \code{ensemble_size}, \code{ensemble_weighting}, \code{ensemble_temperature},
#'   \code{ensemble_profiles}, \code{ensemble_profile_oof_rmse} and
#'   \code{ensemble_weights}, and \code{candidate_table} gains
#'   \code{ensemble_member}, \code{ensemble_rank} and \code{ensemble_weight}
#'   columns. Note that when the ensemble is applied the conformal half-widths
#'   (\code{pi_q90}, \code{pi_q95} and the \code{_w}/\code{_sp} variants) are
#'   inherited from the single best (top-ranked OOF) profile and are \emph{not}
#'   recalibrated against the blended \code{pred_test}; the conformal coverage
#'   guarantee therefore pertains to the best single profile, not to the blended
#'   point estimate (\code{diagnostics$ensemble_pi_from_best} flags this).
#' @examples
#' set.seed(1)
#' n <- 120
#' tr <- data.frame(y = rnorm(n), a = rnorm(n), b = rnorm(n))
#' te <- tr[1:15, ]
#' out <- dbtarf_train_predict(tr, te, "y", c("a", "b"),
#'                             coord_names = NULL, rf_tune = FALSE,
#'                             fair_profile_search = FALSE)
#' head(out$pred_test)
#' @importFrom stats predict
#' @export
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
