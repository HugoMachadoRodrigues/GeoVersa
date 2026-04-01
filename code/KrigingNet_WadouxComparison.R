# =============================================================================
# KrigingNet vs baselines under Wadoux-style validation protocols
# - Keeps the model name "KrigingNet" for the neural residual-kriging model
# - Separates the model from the validation protocol
# - Supports random CV, spatial block CV, buffered LOO, and design-based holdout
# =============================================================================

rm(list = ls())
set.seed(123)

# -----------------------------
# 0) Packages
# -----------------------------
# pkgs <- c("torch", "dplyr", "sf", "FNN", "randomForest", "xgboost", "Cubist", "ggplot2")
# to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
# if (length(to_install) > 0) install.packages(to_install)

library(torch)
library(dplyr)
library(sf)
library(FNN)
library(randomForest)
library(xgboost)
library(Cubist)
library(ggplot2)

# -----------------------------
# 1) Load simulation object
# -----------------------------
sim <- readRDS("./data/soilflux_simulation.rds")

# =============================================================================
# 2) Helpers
# =============================================================================

to_float_tensor <- function(x, device = "cpu") {
  torch_tensor(x, dtype = torch_float(), device = device)
}

clone_state_dict <- function(state_dict) {
  lapply(state_dict, function(x) x$clone())
}

metrics <- function(obs, pred) {
  n <- length(obs)
  rmse <- sqrt(mean((obs - pred)^2))
  mae  <- mean(abs(obs - pred))
  bias <- mean(pred - obs)
  r2   <- if (n >= 2) cor(obs, pred)^2 else NA_real_
  rpiq <- if (n >= 2) IQR(obs) / (rmse + 1e-12) else NA_real_
  data.frame(N = n, R2 = r2, RMSE = rmse, MAE = mae, Bias = bias, RPIQ = rpiq)
}

fit_scaler <- function(X, robust = TRUE) {
  if (robust) {
    center <- apply(X, 2, median, na.rm = TRUE)
    scale  <- apply(X, 2, IQR, na.rm = TRUE)
    scale[scale == 0] <- 1
  } else {
    center <- colMeans(X, na.rm = TRUE)
    scale  <- apply(X, 2, sd, na.rm = TRUE)
    scale[scale == 0] <- 1
  }
  list(center = center, scale = scale, robust = robust)
}

apply_scaler <- function(X, scaler) {
  sweep(sweep(X, 2, scaler$center, "-"), 2, scaler$scale, "/")
}

fit_standard_scaler <- function(x) {
  center <- colMeans(as.matrix(x), na.rm = TRUE)
  scale <- apply(as.matrix(x), 2, sd, na.rm = TRUE)
  scale[scale == 0] <- 1
  list(center = center, scale = scale)
}

apply_standard_scaler <- function(x, scaler) {
  x_mat <- as.matrix(x)
  sweep(sweep(x_mat, 2, scaler$center, "-"), 2, scaler$scale, "/")
}

fit_target_scaler <- function(y) {
  center <- mean(y, na.rm = TRUE)
  scale <- sd(y, na.rm = TRUE)
  if (is.na(scale) || scale == 0) scale <- 1
  list(center = center, scale = scale)
}

transform_target <- function(y, method = c("identity", "log")) {
  method <- match.arg(method)
  if (method == "log") {
    if (any(y <= 0, na.rm = TRUE)) {
      stop("log target transform requires strictly positive targets.")
    }
    return(log(y))
  }
  y
}

inverse_transform_target <- function(y, method = c("identity", "log")) {
  method <- match.arg(method)
  if (method == "log") {
    return(exp(y))
  }
  y
}

apply_target_scaler <- function(y, scaler) {
  as.numeric((y - scaler$center) / scaler$scale)
}

invert_target_scaler <- function(y_scaled, scaler) {
  as.numeric(y_scaled * scaler$scale + scaler$center)
}

compute_neighbor_idx_train_only <- function(coords_train, K) {
  if (nrow(coords_train) <= 1) stop("Need at least 2 training points to compute neighbors.")
  k_eff <- min(K, nrow(coords_train) - 1)
  kn <- FNN::get.knn(coords_train, k = k_eff)
  kn$nn.index
}

cdist_safe <- function(A, B) {
  if (exists("torch_cdist", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_cdist(A, B))
  }
  ax <- A[,1]$unsqueeze(2)
  ay <- A[,2]$unsqueeze(2)
  bx <- B[,1]$unsqueeze(1)
  by <- B[,2]$unsqueeze(1)
  torch_sqrt((ax - bx)^2 + (ay - by)^2)
}

extract_topk_indices <- function(topk_out) {
  if (is.list(topk_out) && "indices" %in% names(topk_out)) return(topk_out$indices)
  if (is.list(topk_out) && "index" %in% names(topk_out)) return(topk_out$index)
  if (is.list(topk_out) && "idx" %in% names(topk_out)) return(topk_out$idx)
  if (is.list(topk_out) && length(topk_out) >= 2) return(topk_out[[2]])
  stop("Could not extract indices from torch_topk().")
}

topk_smallest_idx <- function(d, K) {
  out <- try(torch_topk(d, k = K, dim = 2, largest = FALSE), silent = TRUE)
  if (!inherits(out, "try-error")) return(extract_topk_indices(out))

  out2 <- torch_topk(-d, k = K, dim = 2, largest = TRUE)
  extract_topk_indices(out2)
}

reshape_safe <- function(x, shape) {
  if (exists("torch_reshape", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_reshape(x, shape))
  }
  if (exists("torch_view", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_view(x, shape))
  }
  stop("torch_reshape/torch_view not found in this torch version.")
}

flatten_safe <- function(x) {
  if (exists("torch_reshape", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_reshape(x, c(-1)))
  }
  if (exists("torch_view", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_view(x, c(-1)))
  }
  stop("torch_reshape/torch_view not found in this torch version.")
}

make_random_fold_ids <- function(n, n_folds) {
  sample(rep(seq_len(n_folds), length.out = n))
}

make_spatial_block_fold_ids <- function(points_sf, block_size_m, n_folds) {
  bb <- st_bbox(points_sf)
  xy <- st_coordinates(points_sf)
  bx <- floor((xy[,1] - bb["xmin"]) / block_size_m)
  by <- floor((xy[,2] - bb["ymin"]) / block_size_m)
  block_id <- paste(bx, by, sep = "_")

  unique_blocks <- unique(block_id)
  fold_of_block <- sample(rep(seq_len(n_folds), length.out = length(unique_blocks)))
  names(fold_of_block) <- unique_blocks
  as.integer(fold_of_block[block_id])
}

make_inner_validation_idx <- function(points_sf,
                                      train_idx_all,
                                      val_frac = 0.2,
                                      spatial = TRUE,
                                      block_size_m = NULL) {
  if (length(train_idx_all) < 5) {
    return(sample(train_idx_all, size = max(1, floor(length(train_idx_all) * val_frac))))
  }

  if (!spatial || is.null(block_size_m)) {
    n_val <- max(1, floor(length(train_idx_all) * val_frac))
    return(sample(train_idx_all, size = n_val))
  }

  bb <- st_bbox(points_sf[train_idx_all, ])
  xy <- st_coordinates(points_sf[train_idx_all, ])
  bx <- floor((xy[,1] - bb["xmin"]) / block_size_m)
  by <- floor((xy[,2] - bb["ymin"]) / block_size_m)
  block_id <- paste(bx, by, sep = "_")

  blocks <- unique(block_id)
  n_val_blocks <- max(1, floor(length(blocks) * val_frac))
  val_blocks <- sample(blocks, size = n_val_blocks)
  train_idx_all[block_id %in% val_blocks]
}

make_fold_from_split <- function(X,
                                 y,
                                 coords,
                                 points_sf,
                                 split,
                                 cfg) {
  train_idx <- split$train_idx
  val_idx   <- split$val_idx
  test_idx  <- split$test_idx

  X_train <- X[train_idx, , drop = FALSE]
  X_val   <- X[val_idx,   , drop = FALSE]
  X_test  <- X[test_idx,  , drop = FALSE]

  y_train <- y[train_idx]
  y_val   <- y[val_idx]
  y_test  <- y[test_idx]

  coords_train <- coords[train_idx, , drop = FALSE]
  coords_val   <- coords[val_idx,   , drop = FALSE]
  coords_test  <- coords[test_idx,  , drop = FALSE]

  scaler <- fit_scaler(X_train, robust = cfg$use_robust_scaling)
  X_train_s <- apply_scaler(X_train, scaler)
  X_val_s   <- apply_scaler(X_val, scaler)
  X_test_s  <- apply_scaler(X_test, scaler)

  neighbor_idx <- compute_neighbor_idx_train_only(coords_train, cfg$K)

  list(
    idx = list(train = train_idx, val = val_idx, test = test_idx),
    X = list(train = X_train_s, val = X_val_s, test = X_test_s),
    y = list(train = y_train, val = y_val, test = y_test),
    coords = list(train = coords_train, val = coords_val, test = coords_test),
    scaler = scaler,
    neighbor_idx_train = neighbor_idx,
    protocol = split$protocol,
    split_id = split$split_id
  )
}

make_cv_splits <- function(points_sf,
                           protocol = c("random_cv", "spatial_block_cv"),
                           n_folds = 5,
                           block_size_m = 4000,
                           val_frac = 0.2) {
  protocol <- match.arg(protocol)

  fold_id <- switch(
    protocol,
    random_cv = make_random_fold_ids(nrow(points_sf), n_folds),
    spatial_block_cv = make_spatial_block_fold_ids(points_sf, block_size_m, n_folds)
  )

  splits <- vector("list", n_folds)
  for (f in seq_len(n_folds)) {
    train_idx_all <- which(fold_id != f)
    test_idx <- which(fold_id == f)
    val_idx <- make_inner_validation_idx(
      points_sf = points_sf,
      train_idx_all = train_idx_all,
      val_frac = val_frac,
      spatial = protocol == "spatial_block_cv",
      block_size_m = block_size_m
    )
    train_idx <- setdiff(train_idx_all, val_idx)

    splits[[f]] <- list(
      train_idx = train_idx,
      val_idx = val_idx,
      test_idx = test_idx,
      protocol = protocol,
      split_id = f
    )
  }

  splits
}

make_buffered_loo_splits <- function(points_sf,
                                     buffer_radius_m = 3500,
                                     val_frac = 0.2,
                                     max_splits = NULL) {
  coords <- st_coordinates(points_sf)
  n <- nrow(coords)
  use_idx <- seq_len(n)
  if (!is.null(max_splits) && max_splits < n) {
    use_idx <- sort(sample(use_idx, max_splits))
  }

  splits <- vector("list", length(use_idx))
  k <- 1
  for (i in use_idx) {
    d <- sqrt((coords[,1] - coords[i,1])^2 + (coords[,2] - coords[i,2])^2)
    excluded <- which(d <= buffer_radius_m)
    train_idx_all <- setdiff(seq_len(n), excluded)
    test_idx <- i

    if (length(train_idx_all) < 10) next

    val_idx <- make_inner_validation_idx(
      points_sf = points_sf,
      train_idx_all = train_idx_all,
      val_frac = val_frac,
      spatial = FALSE,
      block_size_m = NULL
    )
    train_idx <- setdiff(train_idx_all, val_idx)

    splits[[k]] <- list(
      train_idx = train_idx,
      val_idx = val_idx,
      test_idx = test_idx,
      protocol = "buffered_loo",
      split_id = i
    )
    k <- k + 1
  }

  Filter(Negate(is.null), splits)
}

make_design_based_holdout_splits <- function(points_sf,
                                             n_repeats = 10,
                                             test_size = 100,
                                             val_frac = 0.2) {
  n <- nrow(points_sf)
  splits <- vector("list", n_repeats)

  for (r in seq_len(n_repeats)) {
    test_idx <- sample(seq_len(n), size = min(test_size, n - 5))
    train_idx_all <- setdiff(seq_len(n), test_idx)
    val_idx <- make_inner_validation_idx(
      points_sf = points_sf,
      train_idx_all = train_idx_all,
      val_frac = val_frac,
      spatial = FALSE,
      block_size_m = NULL
    )
    train_idx <- setdiff(train_idx_all, val_idx)

    splits[[r]] <- list(
      train_idx = train_idx,
      val_idx = val_idx,
      test_idx = test_idx,
      protocol = "design_based_holdout",
      split_id = r
    )
  }

  splits
}

build_resampling_plan <- function(sim,
                                  protocol = c("random_cv", "spatial_block_cv", "buffered_loo", "design_based_holdout"),
                                  n_folds = 5,
                                  block_size_m = NULL,
                                  buffer_radius_m = NULL,
                                  val_frac = 0.2,
                                  n_repeats = 10,
                                  test_size = 100,
                                  max_splits = NULL) {
  protocol <- match.arg(protocol)
  points_sf <- sim$points_sf
  cfg <- sim$cfg

  if (is.null(block_size_m)) block_size_m <- cfg$block_size_m
  if (is.null(buffer_radius_m)) buffer_radius_m <- cfg$corr_range_m

  splits <- switch(
    protocol,
    random_cv = make_cv_splits(points_sf, protocol, n_folds, block_size_m, val_frac),
    spatial_block_cv = make_cv_splits(points_sf, protocol, n_folds, block_size_m, val_frac),
    buffered_loo = make_buffered_loo_splits(points_sf, buffer_radius_m, val_frac, max_splits),
    design_based_holdout = make_design_based_holdout_splits(points_sf, n_repeats, test_size, val_frac)
  )

  lapply(splits, function(split) {
    make_fold_from_split(
      X = sim$X,
      y = sim$y,
      coords = sim$coords,
      points_sf = points_sf,
      split = split,
      cfg = cfg
    )
  })
}

# =============================================================================
# 3) Baselines
# =============================================================================

fit_predict_rf <- function(X_train, y_train, X_test, ntree = 500) {
  rf <- randomForest(x = X_train, y = y_train, ntree = ntree)
  as.numeric(predict(rf, X_test))
}

fit_predict_xgb <- function(X_train, y_train, X_val, y_val, X_test,
                            nrounds = 5000, eta = 0.03, max_depth = 6,
                            subsample = 0.8, colsample_bytree = 0.8,
                            min_child_weight = 1, reg_lambda = 1) {
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dval   <- xgb.DMatrix(data = X_val,   label = y_val)
  dtest  <- xgb.DMatrix(data = X_test)

  params <- list(
    objective = "reg:squarederror",
    eta = eta,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    lambda = reg_lambda
  )

  xgbm <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = list(val = dval),
    early_stopping_rounds = 50,
    verbose = 0
  )

  as.numeric(predict(xgbm, dtest))
}

fit_predict_cubist <- function(X_train, y_train, X_test, committees = 50, neighbors = 5) {
  cb <- Cubist::cubist(x = X_train, y = y_train,
                       committees = committees, neighbors = neighbors)
  as.numeric(predict(cb, X_test))
}

# =============================================================================
# 4) KrigingNet model
# =============================================================================

FourierFeatures <- nn_module(
  "FourierFeatures",
  initialize = function(num_freq = 32, max_freq = 10) {
    self$num_freq <- num_freq
    freqs <- torch_logspace(0, log10(max_freq), steps = num_freq)
    self$register_buffer("freqs", freqs)
  },
  forward = function(coords) {
    freq_view <- reshape_safe(self$freqs, c(1, self$num_freq, 1))
    x <- coords$unsqueeze(2) * freq_view
    sinv <- torch_sin(2 * pi * x)
    cosv <- torch_cos(2 * pi * x)
    out <- torch_cat(list(sinv, cosv), dim = 2)
    reshape_safe(out, c(coords$size(1), -1))
  }
)

make_mlp <- function(in_dim, hidden, out_dim, dropout = 0.10) {
  layers <- list()
  prev <- in_dim

  for (h in hidden) {
    layers[[length(layers) + 1]] <- nn_linear(prev, h)
    layers[[length(layers) + 1]] <- nn_gelu()
    layers[[length(layers) + 1]] <- nn_layer_norm(h)
    layers[[length(layers) + 1]] <- nn_dropout(dropout)
    prev <- h
  }

  layers[[length(layers) + 1]] <- nn_linear(prev, out_dim)
  do.call(nn_sequential, layers)
}

ScalarHead <- nn_module(
  "ScalarHead",
  initialize = function(d = 256) {
    self$net <- nn_sequential(
      nn_linear(d, 128), nn_gelu(),
      nn_linear(128, 64), nn_gelu(),
      nn_linear(64, 1)
    )
  },
  forward = function(z) {
    self$net(z)$squeeze(2)
  }
)

ResidualKrigingLayer <- nn_module(
  "ResidualKrigingLayer",
  initialize = function(d = 256, proj_d = 64, init_ell = 1000) {
    self$proj <- nn_linear(d, proj_d, bias = FALSE)
    self$log_ell <- nn_parameter(torch_log(torch_tensor(init_ell)))
    self$scale <- 1 / sqrt(proj_d)
  },
  forward = function(z_i, coords_i, z_n, coords_n, r_n) {
    dist <- torch_norm(coords_i$unsqueeze(2) - coords_n, dim = 3)
    ell  <- nnf_softplus(self$log_ell) + 1e-6

    qi <- self$proj(z_i)
    qn <- self$proj(z_n)
    sim <- torch_sum(qn * qi$unsqueeze(2), dim = 3) * self$scale

    w <- nnf_softmax(-dist / ell + sim, dim = 2)
    delta <- torch_sum(w * r_n, dim = 2)
    list(delta = delta, w = w)
  }
)

KrigingNet <- nn_module(
  "KrigingNet",
  initialize = function(c_tab,
                        d = 256,
                        tab_hidden = c(512),
                        tab_dropout = 0.10,
                        use_tab_skip = FALSE,
                        num_freq = 32,
                        coord_hidden = c(128),
                        coord_dim = 128,
                        coord_dropout = 0.10,
                        spatial_encoder = c("fourier_mlp", "raw_mlp"),
                        beta_mode = c("learned", "fixed_zero"),
                        beta_scope = c("global", "local"),
                        beta_init = -2) {
    spatial_encoder <- match.arg(spatial_encoder)
    beta_mode <- match.arg(beta_mode)
    beta_scope <- match.arg(beta_scope)

    self$d <- d
    self$spatial_encoder <- spatial_encoder
    self$beta_mode <- beta_mode
    self$beta_scope <- beta_scope
    self$use_tab_skip <- use_tab_skip
    self$enc_tab <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = tab_dropout)

    if (spatial_encoder == "fourier_mlp") {
      self$ff <- FourierFeatures(num_freq = num_freq, max_freq = 10)
      coord_in_dim <- 2 * num_freq * 2
    } else {
      self$ff <- NULL
      coord_in_dim <- 2
    }

    self$enc_coord <- make_mlp(
      coord_in_dim,
      hidden = coord_hidden,
      out_dim = coord_dim,
      dropout = coord_dropout
    )
    self$proj_coord <- nn_linear(coord_dim, d)

    self$gate <- nn_sequential(
      nn_linear(d * 2, 128), nn_gelu(),
      nn_linear(128, 2)
    )

    self$head <- ScalarHead(d = d)
    if (use_tab_skip) {
      self$tab_skip <- nn_linear(d, 1)
    } else {
      self$tab_skip <- NULL
    }
    self$krig <- ResidualKrigingLayer(d = d, proj_d = 64, init_ell = 1000)
    if (beta_mode == "learned") {
      if (beta_scope == "global") {
        self$logit_beta <- nn_parameter(torch_tensor(beta_init))
        self$beta_net <- NULL
      } else {
        self$logit_beta <- NULL
        self$beta_net <- nn_sequential(
          nn_linear(d, 64), nn_gelu(),
          nn_linear(64, 1)
        )
      }
    } else {
      self$logit_beta <- NULL
      self$beta_net <- NULL
    }
  },

  encode = function(x_tab, coords) {
    zt <- self$enc_tab(x_tab)

    zf <- if (self$spatial_encoder == "fourier_mlp") self$ff(coords) else coords
    zc <- self$proj_coord(self$enc_coord(zf))

    zcat <- torch_cat(list(zt, zc), dim = 2)
    a <- nnf_softmax(self$gate(zcat), dim = 2)

    z <- zt * a[,1]$unsqueeze(2) + zc * a[,2]$unsqueeze(2)
    list(z = z, z_tab = zt, z_coord = zc, gate = a)
  },

  forward_base = function(x_tab, coords) {
    enc <- self$encode(x_tab, coords)
    pred <- self$head(enc$z)
    if (self$use_tab_skip) {
      pred <- pred + self$tab_skip(enc$z_tab)$squeeze(2)
    }
    list(pred = pred, z = enc$z, z_tab = enc$z_tab, z_coord = enc$z_coord, gate = enc$gate)
  },

  forward_with_kriging = function(x_tab, coords, z_n, coords_n, r_n) {
    base <- self$forward_base(x_tab, coords)
    if (self$beta_mode == "fixed_zero") {
      beta <- torch_zeros(1, device = base$pred$device)
      delta <- torch_zeros_like(base$pred)
      pred_corr <- base$pred
    } else {
      k <- self$krig(base$z, coords, z_n, coords_n, r_n)
      beta <- if (self$beta_scope == "local") {
        torch_sigmoid(self$beta_net(base$z))$squeeze(2)
      } else {
        torch_sigmoid(self$logit_beta)
      }
      delta <- k$delta
      pred_corr <- base$pred + beta * delta
    }
    list(pred = pred_corr, base_pred = base$pred, z = base$z, delta = delta, beta = beta, gate = base$gate)
  }
)

huber_loss <- function(y, pred, delta = 1.0) {
  err <- y - pred
  abs_err <- torch_abs(err)
  delta_t <- torch_full_like(abs_err, delta)
  quadratic <- torch_minimum(abs_err, delta_t)
  linear <- abs_err - quadratic
  torch_mean(0.5 * quadratic^2 + delta * linear)
}

make_batches <- function(n, batch_size = 256) {
  idx <- sample.int(n)
  split(idx, ceiling(seq_along(idx) / batch_size))
}

build_memory_bank <- function(model, X_train, coords_train, y_train, device = "cpu", batch_size = 1024) {
  model$eval()
  n <- nrow(X_train)

  Z_list <- list()
  R_list <- list()
  C_list <- list()

  with_no_grad({
    for (s in seq(1, n, by = batch_size)) {
      e <- min(s + batch_size - 1, n)

      xb <- to_float_tensor(X_train[s:e, , drop = FALSE], device = device)
      cb <- to_float_tensor(coords_train[s:e, , drop = FALSE], device = device)
      yb <- to_float_tensor(y_train[s:e], device = device)

      out <- model$forward_base(xb, cb)
      r <- yb - out$pred

      Z_list[[length(Z_list) + 1]] <- out$z$cpu()
      R_list[[length(R_list) + 1]] <- r$cpu()
      C_list[[length(C_list) + 1]] <- cb$cpu()
    }
  })

  Z <- torch_cat(Z_list, dim = 1)
  R <- torch_cat(R_list, dim = 1)
  C <- torch_cat(C_list, dim = 1)

  list(Z = Z, R = R, C = C)
}

predict_with_memory <- function(model,
                                X_new,
                                coords_new,
                                Zmem,
                                Rmem,
                                Cmem,
                                K,
                                device = "cpu",
                                batch_size = 512) {
  preds <- numeric(nrow(X_new))

  with_no_grad({
    for (s in seq(1, nrow(X_new), by = batch_size)) {
      e <- min(s + batch_size - 1, nrow(X_new))
      B <- e - s + 1

      xb <- to_float_tensor(X_new[s:e, , drop = FALSE], device = device)
      cb <- to_float_tensor(coords_new[s:e, , drop = FALSE], device = device)

      d <- cdist_safe(cb, Cmem)
      knn <- topk_smallest_idx(d, K)
      nb_flat <- flatten_safe(knn)$to(dtype = torch_long())

      zn <- reshape_safe(Zmem$index_select(1, nb_flat), c(B, K, -1))
      rn <- reshape_safe(Rmem$index_select(1, nb_flat), c(B, K))
      cn <- reshape_safe(Cmem$index_select(1, nb_flat), c(B, K, 2))

      out <- model$forward_with_kriging(xb, cb, zn, cn, rn)
      preds[s:e] <- as.numeric(out$pred$cpu())
    }
  })

  preds
}

train_krigingnet_one_fold <- function(fd,
                                      epochs = 200,
                                      lr = 2e-4,
                                      wd = 1e-3,
                                      batch_size = 256,
                                      patience = 20,
                                      d = 256,
                                      tab_hidden = c(512),
                                      tab_dropout = 0.10,
                                      use_tab_skip = FALSE,
                                      num_freq = 32,
                                      coord_hidden = c(128),
                                      coord_dim = 128,
                                      coord_dropout = 0.10,
                                      spatial_encoder = "fourier_mlp",
                                      beta_mode = "learned",
                                      beta_scope = "global",
                                      beta_init = -2,
                                      target_transform = "identity",
                                      K_neighbors = NULL,
                                      device = "cpu") {
  Xtr <- fd$X$train; ytr <- fd$y$train; Ctr <- fd$coords$train
  Xva <- fd$X$val;   yva <- fd$y$val;   Cva <- fd$coords$val
  Xte <- fd$X$test;  yte <- fd$y$test;  Cte <- fd$coords$test

  ytr_t <- transform_target(ytr, target_transform)
  yva_t <- transform_target(yva, target_transform)
  yte_t <- transform_target(yte, target_transform)

  y_scaler <- fit_target_scaler(ytr_t)
  coord_scaler <- fit_standard_scaler(Ctr)

  ytr_s <- apply_target_scaler(ytr_t, y_scaler)
  yva_s <- apply_target_scaler(yva_t, y_scaler)

  Ctr_s <- apply_standard_scaler(Ctr, coord_scaler)
  Cva_s <- apply_standard_scaler(Cva, coord_scaler)
  Cte_s <- apply_standard_scaler(Cte, coord_scaler)

  neigh_train <- fd$neighbor_idx_train
  if (!is.null(K_neighbors)) {
    k_eff <- min(K_neighbors, ncol(neigh_train))
    neigh_train <- neigh_train[, seq_len(k_eff), drop = FALSE]
  }

  model <- KrigingNet(
    c_tab = ncol(Xtr),
    d = d,
    tab_hidden = tab_hidden,
    tab_dropout = tab_dropout,
    use_tab_skip = use_tab_skip,
    num_freq = num_freq,
    coord_hidden = coord_hidden,
    coord_dim = coord_dim,
    coord_dropout = coord_dropout,
    spatial_encoder = spatial_encoder,
    beta_mode = beta_mode,
    beta_scope = beta_scope,
    beta_init = beta_init
  )
  model$to(device = device)

  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)

  best_val <- Inf
  best_state <- NULL
  bad <- 0

  for (ep in seq_len(epochs)) {
    bank <- build_memory_bank(model, Xtr, Ctr_s, ytr_s, device = device, batch_size = 1024)
    Zmem <- bank$Z$to(device = device)
    Rmem <- bank$R$to(device = device)
    Cmem <- bank$C$to(device = device)

    model$train()
    batches <- make_batches(nrow(Xtr), batch_size = batch_size)
    loss_epoch <- 0
    Ktr <- ncol(neigh_train)

    for (b in batches) {
      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr_s[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr_s[b], device = device)

      nb <- neigh_train[b, , drop = FALSE]
      nb_t <- torch_tensor(as.vector(nb), dtype = torch_long(), device = device)

      zn <- reshape_safe(Zmem$index_select(1, nb_t), c(length(b), Ktr, -1))
      rn <- reshape_safe(Rmem$index_select(1, nb_t), c(length(b), Ktr))
      cn <- reshape_safe(Cmem$index_select(1, nb_t), c(length(b), Ktr, 2))

      out <- model$forward_with_kriging(xb, cb, zn, cn, rn)
      loss <- huber_loss(yb, out$pred)

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
      opt$step()

      loss_epoch <- loss_epoch + loss$item()
    }

    model$eval()
    val_pred <- predict_with_memory(
      model = model,
      X_new = Xva,
      coords_new = Cva_s,
      Zmem = Zmem,
      Rmem = Rmem,
      Cmem = Cmem,
      K = Ktr,
      device = device,
      batch_size = 512
    )
    yb <- to_float_tensor(yva_s, device = device)
    pb <- to_float_tensor(val_pred, device = device)
    vloss <- huber_loss(yb, pb)$item()

    if (ep %% 10 == 0) {
      cat(sprintf("[%s:%s] Epoch %d | train_loss=%.4f | val_loss=%.4f\n",
                  fd$protocol, fd$split_id, ep, loss_epoch / length(batches), vloss))
    }

    if (vloss < best_val) {
      best_val <- vloss
      best_state <- clone_state_dict(model$state_dict())
      bad <- 0
    } else {
      bad <- bad + 1
      if (bad >= patience) break
    }
  }

  model$load_state_dict(best_state)
  model$eval()

  bank <- build_memory_bank(model, Xtr, Ctr_s, ytr_s, device = device, batch_size = 1024)
  Zmem <- bank$Z$to(device = device)
  Rmem <- bank$R$to(device = device)
  Cmem <- bank$C$to(device = device)

  K <- ncol(neigh_train)
  preds_scaled <- predict_with_memory(
    model = model,
    X_new = Xte,
    coords_new = Cte_s,
    Zmem = Zmem,
    Rmem = Rmem,
    Cmem = Cmem,
    K = K,
    device = device,
    batch_size = 512
  )
  preds_t <- invert_target_scaler(preds_scaled, y_scaler)
  preds <- inverse_transform_target(preds_t, target_transform)

  list(
    model = model,
    pred_test = preds,
    metrics_test = metrics(yte, preds)
  )
}

make_krigingnet_variants <- function() {
  list(
    KrigingNet = krigingnet_params,
    KrigingNet_NoResidual = modifyList(
      krigingnet_params,
      list(beta_mode = "fixed_zero")
    ),
    KrigingNet_FewerNeighbors = modifyList(
      krigingnet_params,
      list(K_neighbors = 8)
    ),
    KrigingNet_WeakSpatial = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05
      )
    ),
    KrigingNet_WeakSpatial_K12 = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 12
      )
    ),
    KrigingNet_WeakSpatial_K16 = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 16
      )
    ),
    KrigingNet_WeakSpatial_K20 = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 20
      )
    ),
    KrigingNet_WeakSpatial_BetaSmall = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        beta_init = -4
      )
    ),
    KrigingNet_WeakSpatial_K12_BetaSmall = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 12,
        beta_init = -4
      )
    ),
    KrigingNet_WeakSpatial_K16_BetaSmall = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 16,
        beta_init = -4
      )
    ),
    KrigingNet_WeakSpatial_K20_BetaSmall = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 20,
        beta_init = -4
      )
    ),
    KrigingNet_v3 = modifyList(
      krigingnet_params,
      list(
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 12,
        beta_init = -4
      )
    ),
    KrigingNet_v3_TabSmall = modifyList(
      krigingnet_params,
      list(
        tab_hidden = c(256),
        tab_dropout = 0.15,
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 12,
        beta_init = -4
      )
    ),
    KrigingNet_v3_TabSmaller = modifyList(
      krigingnet_params,
      list(
        tab_hidden = c(128),
        tab_dropout = 0.20,
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 12,
        beta_init = -4
      )
    ),
    KrigingNet_v3_TabTwoLayer = modifyList(
      krigingnet_params,
      list(
        tab_hidden = c(256, 128),
        tab_dropout = 0.15,
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 12,
        beta_init = -4
      )
    ),
    KrigingNet_v3_TabSmallDrop = modifyList(
      krigingnet_params,
      list(
        tab_hidden = c(256),
        tab_dropout = 0.25,
        spatial_encoder = "raw_mlp",
        coord_hidden = c(32),
        coord_dim = 32,
        coord_dropout = 0.05,
        K_neighbors = 12,
        beta_init = -4
      )
    ),
    KrigingNet_v4_TabSkip = modifyList(
      krigingnet_params,
      list(
        use_tab_skip = TRUE
      )
    ),
    KrigingNet_v4_LocalBeta = modifyList(
      krigingnet_params,
      list(
        beta_scope = "local"
      )
    ),
    KrigingNet_v4_TabSkip_LocalBeta = modifyList(
      krigingnet_params,
      list(
        use_tab_skip = TRUE,
        beta_scope = "local"
      )
    ),
    KrigingNet_v4_TabSmall_LocalBeta = modifyList(
      krigingnet_params,
      list(
        tab_hidden = c(256),
        tab_dropout = 0.15,
        use_tab_skip = TRUE,
        beta_scope = "local"
      )
    ),
    KrigingNet_v4_TabTwoLayer_LocalBeta = modifyList(
      krigingnet_params,
      list(
        tab_hidden = c(256, 128),
        tab_dropout = 0.15,
        use_tab_skip = TRUE,
        beta_scope = "local"
      )
    ),
    KrigingNet_v3_TabSmall_Log = modifyList(
      krigingnet_params,
      list(
        target_transform = "log"
      )
    ),
    KrigingNet_v4_LocalBeta_Log = modifyList(
      krigingnet_params,
      list(
        beta_scope = "local",
        target_transform = "log"
      )
    )
  )
}

run_krigingnet_ablation <- function(sim,
                                    protocol = c("random_cv", "spatial_block_cv", "buffered_loo", "design_based_holdout"),
                                    variants = make_krigingnet_variants(),
                                    xgb_params = xgb_params,
                                    n_folds = 5,
                                    block_size_m = NULL,
                                    buffer_radius_m = NULL,
                                    val_frac = 0.2,
                                    n_repeats = 10,
                                    test_size = 100,
                                    max_splits = NULL) {
  protocol <- match.arg(protocol)

  folds <- build_resampling_plan(
    sim = sim,
    protocol = protocol,
    n_folds = n_folds,
    block_size_m = block_size_m,
    buffer_radius_m = buffer_radius_m,
    val_frac = val_frac,
    n_repeats = n_repeats,
    test_size = test_size,
    max_splits = max_splits
  )

  results <- list()

  for (variant_name in names(variants)) {
    cat("\n========================================\n")
    cat("ABLATION VARIANT:", variant_name, "\n")
    cat("========================================\n")

    variant_results <- vector("list", length(folds))
    for (i in seq_along(folds)) {
      fd <- folds[[i]]
      kn_out <- do.call(train_krigingnet_one_fold, c(list(fd = fd), variants[[variant_name]]))
      met_kn <- kn_out$metrics_test
      met_kn$model <- variant_name
      variant_results[[i]] <- met_kn %>% mutate(protocol = protocol, split = fd$split_id)
    }

    results[[variant_name]] <- bind_rows(variant_results)
  }

  bind_rows(results)
}

# =============================================================================
# 5) Comparison runner
# =============================================================================

run_krigingnet_comparison <- function(sim,
                                      protocol = c("random_cv", "spatial_block_cv", "buffered_loo", "design_based_holdout"),
                                      rf_ntree = 500,
                                      cubist_committees = 50,
                                      cubist_neighbors = 5,
                                      xgb_params = list(),
                                      krigingnet_params = list(),
                                      n_folds = 5,
                                      block_size_m = NULL,
                                      buffer_radius_m = NULL,
                                      val_frac = 0.2,
                                      n_repeats = 10,
                                      test_size = 100,
                                      max_splits = NULL) {
  protocol <- match.arg(protocol)

  folds <- build_resampling_plan(
    sim = sim,
    protocol = protocol,
    n_folds = n_folds,
    block_size_m = block_size_m,
    buffer_radius_m = buffer_radius_m,
    val_frac = val_frac,
    n_repeats = n_repeats,
    test_size = test_size,
    max_splits = max_splits
  )

  results <- vector("list", length(folds))

  for (i in seq_along(folds)) {
    fd <- folds[[i]]

    cat("\n=============================\n")
    cat("PROTOCOL", protocol, "| SPLIT", fd$split_id, "\n")
    cat("=============================\n")

    pred_rf <- fit_predict_rf(fd$X$train, fd$y$train, fd$X$test, ntree = rf_ntree)
    met_rf <- metrics(fd$y$test, pred_rf); met_rf$model <- "RF"

    pred_cb <- fit_predict_cubist(
      fd$X$train, fd$y$train, fd$X$test,
      committees = cubist_committees, neighbors = cubist_neighbors
    )
    met_cb <- metrics(fd$y$test, pred_cb); met_cb$model <- "Cubist"

    pred_xgb <- do.call(
      fit_predict_xgb,
      c(
        list(
          X_train = fd$X$train, y_train = fd$y$train,
          X_val = fd$X$val, y_val = fd$y$val,
          X_test = fd$X$test
        ),
        xgb_params
      )
    )
    met_xgb <- metrics(fd$y$test, pred_xgb); met_xgb$model <- "XGB"

    kn_out <- do.call(train_krigingnet_one_fold, c(list(fd = fd), krigingnet_params))
    met_kn <- kn_out$metrics_test; met_kn$model <- "KrigingNet"

    df <- bind_rows(met_rf, met_xgb, met_cb, met_kn) %>%
      mutate(protocol = protocol, split = fd$split_id)

    print(df)
    results[[i]] <- df
  }

  bind_rows(results)
}

summarise_comparison <- function(results_df) {
  results_df %>%
    group_by(protocol, model) %>%
    summarise(
      N_mean = mean(N),
      R2_mean = mean(R2), R2_sd = sd(R2),
      RMSE_mean = mean(RMSE), RMSE_sd = sd(RMSE),
      MAE_mean = mean(MAE), MAE_sd = sd(MAE),
      Bias_mean = mean(Bias), Bias_sd = sd(Bias),
      RPIQ_mean = mean(RPIQ), RPIQ_sd = sd(RPIQ),
      .groups = "drop"
    ) %>%
    arrange(protocol, desc(R2_mean))
}

summarise_buffered_loo <- function(results_df) {
  df <- results_df %>% filter(protocol == "buffered_loo")

  if (nrow(df) == 0) {
    stop("No buffered_loo results found.")
  }

  df %>%
    group_by(model) %>%
    summarise(
      splits = n(),
      RMSE_mean = sqrt(mean(RMSE^2)),
      MAE_mean = mean(MAE),
      Bias_mean = mean(Bias),
      .groups = "drop"
    ) %>%
    arrange(RMSE_mean)
}

plot_protocol_comparison <- function(results_df, metric = "RMSE") {
  ggplot(results_df, aes(x = model, y = .data[[metric]], fill = model)) +
    geom_boxplot() +
    facet_wrap(~ protocol, scales = "free_y") +
    theme_minimal() +
    coord_flip() +
    guides(fill = "none") +
    ggtitle(paste("Validation comparison by protocol |", metric))
}

label_validation_goal <- function(protocol) {
  dplyr::case_when(
    protocol %in% c("design_based_holdout", "design_based_validation") ~ "Map accuracy",
    protocol %in% c("spatial_block_cv", "spatial_kfold", "buffered_loo") ~ "Spatial transferability",
    protocol %in% c("random_cv") ~ "Local interpolation performance",
    TRUE ~ "Other"
  )
}

attach_validation_goal <- function(results_df) {
  results_df %>%
    mutate(validation_goal = label_validation_goal(protocol))
}

summarise_by_goal <- function(results_df) {
  attach_validation_goal(results_df) %>%
    group_by(validation_goal, protocol, model) %>%
    summarise(
      N_mean = mean(N, na.rm = TRUE),
      R2_mean = mean(R2, na.rm = TRUE),
      R2_sd = sd(R2, na.rm = TRUE),
      RMSE_mean = mean(RMSE, na.rm = TRUE),
      RMSE_sd = sd(RMSE, na.rm = TRUE),
      MAE_mean = mean(MAE, na.rm = TRUE),
      MAE_sd = sd(MAE, na.rm = TRUE),
      Bias_mean = mean(Bias, na.rm = TRUE),
      Bias_sd = sd(Bias, na.rm = TRUE),
      RPIQ_mean = mean(RPIQ, na.rm = TRUE),
      RPIQ_sd = sd(RPIQ, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(validation_goal, protocol, RMSE_mean)
}

plot_goal_comparison <- function(results_df, metric = "RMSE") {
  attach_validation_goal(results_df) %>%
    ggplot(aes(x = model, y = .data[[metric]], fill = model)) +
    geom_boxplot() +
    facet_wrap(~ validation_goal, scales = "free_y") +
    theme_minimal() +
    coord_flip() +
    guides(fill = "none") +
    ggtitle(paste("Comparison by validation goal |", metric))
}

# =============================================================================
# 6) Defaults and examples
# =============================================================================

xgb_params <- list(
  nrounds = 5000,
  eta = 0.03,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  reg_lambda = 1
)

krigingnet_params <- list(
  epochs = 200,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 256,
  patience = 20,
  d = 256,
  tab_hidden = c(256),
  tab_dropout = 0.15,
  use_tab_skip = FALSE,
  num_freq = 32,
  coord_hidden = c(32),
  coord_dim = 32,
  coord_dropout = 0.05,
  spatial_encoder = "raw_mlp",
  beta_mode = "learned",
  beta_scope = "global",
  beta_init = -4,
  target_transform = "identity",
  K_neighbors = 12,
  device = "cpu"
)

# Example 1: standard random CV
# res_random <- run_krigingnet_comparison(
#   sim,
#   protocol = "random_cv",
#   xgb_params = xgb_params,
#   krigingnet_params = krigingnet_params
# )

# Example 2: spatial block CV
# res_spatial <- run_krigingnet_comparison(
#   sim,
#   protocol = "spatial_block_cv",
#   block_size_m = sim$cfg$block_size_m,
#   xgb_params = xgb_params,
#   krigingnet_params = krigingnet_params
# )

# Example 3: Wadoux-style design-based holdout surrogate
# res_design <- run_krigingnet_comparison(
#   sim,
#   protocol = "design_based_holdout",
#   n_repeats = 10,
#   test_size = 100,
#   xgb_params = xgb_params,
#   krigingnet_params = krigingnet_params
# )

# Example 4: buffered LOO inspired by the spatial validation literature
# res_buffer <- run_krigingnet_comparison(
#   sim,
#   protocol = "buffered_loo",
#   buffer_radius_m = sim$cfg$corr_range_m,
#   max_splits = 50,
#   xgb_params = xgb_params,
#   krigingnet_params = krigingnet_params
# )
#
# summarise_buffered_loo(res_buffer)
