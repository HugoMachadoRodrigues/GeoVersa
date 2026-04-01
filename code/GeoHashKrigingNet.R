rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoHashKrigingNet
# -----------------------------------------------------------------------------
# Lightweight point-based model with:
# - tabular MLP branch
# - multiscale hashed spatial signatures from coordinates
# - conservative fusion of tabular and spatial branches
# - light anisotropic residual kriging-like correction
#
# Plain-language flow:
# 1. Encode covariates with a stable tabular MLP.
# 2. Encode coordinates with multiscale hashed grid signatures.
# 3. Fuse both into a global base prediction.
# 4. Use neighbor residuals for a light anisotropic spatial correction.
# =============================================================================

compute_neighbor_idx_cross <- function(coords_query, coords_ref, K) {
  if (nrow(coords_ref) < 1) stop("Reference coordinates must have at least one row.")
  k_eff <- min(K, nrow(coords_ref))
  FNN::get.knnx(data = coords_ref, query = coords_query, k = k_eff)$nn.index
}

gather_neighbor_array <- function(X_ref, neighbor_idx) {
  B <- nrow(neighbor_idx)
  K <- ncol(neighbor_idx)
  P <- ncol(X_ref)
  out <- array(0, dim = c(B, K, P))
  for (i in seq_len(B)) {
    out[i, , ] <- X_ref[neighbor_idx[i, ], , drop = FALSE]
  }
  out
}

gather_neighbor_matrix <- function(y_ref, neighbor_idx) {
  B <- nrow(neighbor_idx)
  K <- ncol(neighbor_idx)
  out <- matrix(0, nrow = B, ncol = K)
  for (i in seq_len(B)) {
    out[i, ] <- y_ref[neighbor_idx[i, ]]
  }
  out
}

prepare_geohash_fold <- function(context,
                                 calibration_df,
                                 train_idx,
                                 val_idx,
                                 test_idx,
                                 use_robust_scaling = TRUE,
                                 K_neighbors = 12) {
  train_df <- calibration_df[train_idx, , drop = FALSE]
  val_df <- calibration_df[val_idx, , drop = FALSE]
  test_df <- calibration_df[test_idx, , drop = FALSE]

  X_train <- as.matrix(train_df[, context$predictors, drop = FALSE])
  X_val   <- as.matrix(val_df[, context$predictors, drop = FALSE])
  X_test  <- as.matrix(test_df[, context$predictors, drop = FALSE])

  y_train <- train_df[[context$response]]
  y_val   <- val_df[[context$response]]
  y_test  <- test_df[[context$response]]

  coords_train <- as.matrix(train_df[, c("x", "y"), drop = FALSE])
  coords_val   <- as.matrix(val_df[, c("x", "y"), drop = FALSE])
  coords_test  <- as.matrix(test_df[, c("x", "y"), drop = FALSE])

  x_scaler <- fit_scaler(X_train, robust = use_robust_scaling)
  coord_scaler <- fit_standard_scaler(coords_train)

  X_train_s <- apply_scaler(X_train, x_scaler)
  X_val_s   <- apply_scaler(X_val, x_scaler)
  X_test_s  <- apply_scaler(X_test, x_scaler)

  coords_train_s <- apply_standard_scaler(coords_train, coord_scaler)
  coords_val_s   <- apply_standard_scaler(coords_val, coord_scaler)
  coords_test_s  <- apply_standard_scaler(coords_test, coord_scaler)

  k_eff <- min(K_neighbors, nrow(coords_train_s) - 1)
  if (k_eff < 1) stop("Need at least two training points.")

  list(
    X = list(train = X_train_s, val = X_val_s, test = X_test_s),
    y = list(train = y_train, val = y_val, test = y_test),
    coords = list(train = coords_train, val = coords_val, test = coords_test),
    coords_scaled = list(train = coords_train_s, val = coords_val_s, test = coords_test_s),
    neighbor_idx = list(
      train = compute_neighbor_idx_train_only(coords_train_s, k_eff),
      val = compute_neighbor_idx_cross(coords_val_s, coords_train_s, k_eff),
      test = compute_neighbor_idx_cross(coords_test_s, coords_train_s, k_eff)
    )
  )
}

AnisotropicResidualKrigingLayer <- nn_module(
  "AnisotropicResidualKrigingLayer",
  initialize = function(d = 160, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5) {
    self$proj <- nn_linear(d, proj_d, bias = FALSE)
    self$log_ell_major <- nn_parameter(torch_log(torch_tensor(init_ell_major)))
    self$log_ell_minor <- nn_parameter(torch_log(torch_tensor(init_ell_minor)))
    self$theta <- nn_parameter(torch_tensor(0))
    self$scale <- 1 / sqrt(proj_d)
  },
  forward = function(z_i, coords_i, z_n, coords_n, r_n) {
    dx <- coords_i[, 1]$unsqueeze(2) - coords_n[, , 1]
    dy <- coords_i[, 2]$unsqueeze(2) - coords_n[, , 2]

    cth <- torch_cos(self$theta)
    sth <- torch_sin(self$theta)
    u <- cth * dx + sth * dy
    v <- -sth * dx + cth * dy

    ell_major <- nnf_softplus(self$log_ell_major) + 1e-6
    ell_minor <- nnf_softplus(self$log_ell_minor) + 1e-6
    aniso_dist <- torch_sqrt((u / ell_major)^2 + (v / ell_minor)^2 + 1e-8)

    qi <- self$proj(z_i)
    qn <- self$proj(z_n)
    sim <- torch_sum(qn * qi$unsqueeze(2), dim = 3) * self$scale

    w <- nnf_softmax(-aniso_dist + sim, dim = 2)
    delta <- torch_sum(w * r_n, dim = 2)
    list(delta = delta, w = w)
  }
)

GeoHashSpatialEncoder <- nn_module(
  "GeoHashSpatialEncoder",
  initialize = function(out_dim = 160,
                        levels = c(2, 4, 8, 16),
                        hidden = 64,
                        hash_size = 257,
                        dropout = 0.10) {
    self$levels <- as.numeric(levels)
    self$hash_size <- as.numeric(hash_size)
    self$level_nets <- nn_module_list(lapply(seq_along(self$levels), function(i) {
      nn_sequential(
        nn_linear(6, hidden),
        nn_gelu(),
        nn_dropout(dropout),
        nn_linear(hidden, out_dim)
      )
    }))
  },
  forward = function(coords) {
    coords_x <- coords[, 1]
    coords_y <- coords[, 2]
    z_sum <- NULL

    for (i in seq_along(self$levels)) {
      res <- self$levels[[i]]
      gx <- torch_floor((coords_x + 4) * res)
      gy <- torch_floor((coords_y + 4) * res)
      fx <- (coords_x + 4) * res - gx
      fy <- (coords_y + 4) * res - gy

      hash_raw <- torch_remainder(gx * 73856093 + gy * 19349663, self$hash_size)
      hash_phase <- 2 * pi * hash_raw / self$hash_size

      feat <- torch_cat(
        list(
          torch_sin(hash_phase)$unsqueeze(2),
          torch_cos(hash_phase)$unsqueeze(2),
          fx$unsqueeze(2),
          fy$unsqueeze(2),
          (fx^2)$unsqueeze(2),
          (fy^2)$unsqueeze(2)
        ),
        dim = 2
      )

      z_i <- self$level_nets[[i]](feat)
      if (is.null(z_sum)) {
        z_sum <- z_i
      } else {
        z_sum <- z_sum + z_i
      }
    }

    z_sum / sqrt(length(self$levels))
  }
)

GeoHashScalarHead <- nn_module(
  "GeoHashScalarHead",
  initialize = function(d = 160, hidden = 96, dropout = 0.10) {
    self$net <- nn_sequential(
      nn_linear(d, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, 1)
    )
  },
  forward = function(z) {
    self$net(z)
  }
)

GeoHashKrigingNet <- nn_module(
  "GeoHashKrigingNet",
  initialize = function(c_tab,
                        d = 160,
                        tab_hidden = c(192, 96),
                        hash_levels = c(2, 4, 8, 16),
                        hash_hidden = 64,
                        hash_size = 257,
                        fusion_hidden = 160,
                        tab_scale = 1.0,
                        hash_scale = 1.0,
                        fuse_scale = 0.5,
                        dropout = 0.10,
                        beta_init = -4) {
    self$d <- d
    self$tab_scale <- tab_scale
    self$hash_scale <- hash_scale
    self$fuse_scale <- fuse_scale
    self$tab_encoder <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = dropout)
    self$hash_encoder <- GeoHashSpatialEncoder(
      out_dim = d,
      levels = hash_levels,
      hidden = hash_hidden,
      hash_size = hash_size,
      dropout = dropout
    )
    self$fuse_residual <- nn_sequential(
      nn_linear(2 * d, fusion_hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(fusion_hidden, d)
    )
    self$target_head <- GeoHashScalarHead(d = d, hidden = 96, dropout = dropout)
    self$neighbor_head <- GeoHashScalarHead(d = d, hidden = 96, dropout = dropout)
    self$krig <- AnisotropicResidualKrigingLayer(d = d, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_points = function(x, coords) {
    z_tab <- self$tab_encoder(x)
    z_hash <- self$hash_encoder(coords)
    self$tab_scale * z_tab +
      self$hash_scale * z_hash +
      self$fuse_scale * self$fuse_residual(torch_cat(list(z_tab, z_hash), dim = 2))
  },

  encode_neighbors = function(x_neighbors, coords_neighbors) {
    dims <- as.integer(x_neighbors$shape)
    x_flat <- reshape_safe(x_neighbors, c(dims[1] * dims[2], dims[3]))
    c_flat <- reshape_safe(coords_neighbors, c(dims[1] * dims[2], 2))
    z <- self$encode_points(x_flat, c_flat)
    reshape_safe(z, c(dims[1], dims[2], self$d))
  },

  forward = function(x_target, coords_target, x_neighbors, coords_neighbors, y_neighbors, use_residual = TRUE) {
    z_target <- self$encode_points(x_target, coords_target)
    z_neighbors <- self$encode_neighbors(x_neighbors, coords_neighbors)

    base_target <- flatten_safe(self$target_head(z_target))
    dims_neighbors <- as.integer(z_neighbors$shape)
    base_neighbors <- self$neighbor_head(reshape_safe(z_neighbors, c(dims_neighbors[1] * dims_neighbors[2], self$d)))
    base_neighbors <- reshape_safe(flatten_safe(base_neighbors), c(dims_neighbors[1], dims_neighbors[2]))

    if (isTRUE(use_residual)) {
      residual_neighbors <- y_neighbors - base_neighbors
      k <- self$krig(z_target, coords_target, z_neighbors, coords_neighbors, residual_neighbors)
      beta <- torch_sigmoid(self$logit_beta)
      pred <- base_target + beta * k$delta
      delta <- k$delta
    } else {
      beta <- torch_sigmoid(self$logit_beta)
      pred <- base_target
      delta <- torch_zeros_like(base_target)
    }

    list(
      pred = pred,
      base_pred = base_target,
      z = z_target,
      z_neighbors = z_neighbors,
      base_neighbors = base_neighbors,
      delta = delta,
      beta = beta
    )
  }
)

predict_geohash <- function(model,
                            X_query,
                            coords_query,
                            neighbor_idx,
                            X_ref,
                            coords_ref,
                            y_ref,
                            device = "cpu",
                            batch_size = 256) {
  model$eval()
  n <- nrow(X_query)
  out <- numeric(n)

  with_no_grad({
    for (s in seq(1, n, by = batch_size)) {
      e <- min(s + batch_size - 1, n)
      idx <- s:e
      nb <- neighbor_idx[idx, , drop = FALSE]

      xb <- to_float_tensor(X_query[idx, , drop = FALSE], device = device)
      cb <- to_float_tensor(coords_query[idx, , drop = FALSE], device = device)
      xnb <- to_float_tensor(gather_neighbor_array(X_ref, nb), device = device)
      cnb <- to_float_tensor(gather_neighbor_array(coords_ref, nb), device = device)
      ynb <- to_float_tensor(gather_neighbor_matrix(y_ref, nb), device = device)

      pred <- as.numeric(model(xb, cb, xnb, cnb, ynb)$pred$to(device = "cpu"))
      if (length(pred) != length(idx)) {
        stop(sprintf(
          "GeoHashKrigingNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geohash_one_fold <- function(fd,
                                   epochs = 40,
                                   lr = 2e-4,
                                   wd = 1e-3,
                                   batch_size = 96,
                                   patience = 8,
                                   d = 160,
                                   tab_hidden = c(192, 96),
                                   hash_levels = c(2, 4, 8, 16),
                                   hash_hidden = 64,
                                   hash_size = 257,
                                   fusion_hidden = 160,
                                   tab_scale = 1.0,
                                   hash_scale = 1.0,
                                   fuse_scale = 0.5,
                                   dropout = 0.10,
                                   beta_init = -4,
                                   warmup_epochs = 4,
                                   base_loss_weight = 0.35,
                                   target_transform = "identity",
                                   device = "cpu") {
  Xtr <- fd$X$train
  Xva <- fd$X$val
  Xte <- fd$X$test

  Ctr <- fd$coords_scaled$train
  Cva <- fd$coords_scaled$val
  Cte <- fd$coords_scaled$test

  ytr <- fd$y$train
  yva <- fd$y$val
  yte <- fd$y$test

  ytr_t <- transform_target(ytr, target_transform)
  yva_t <- transform_target(yva, target_transform)
  y_scaler <- fit_target_scaler(ytr_t)

  ytr_s <- apply_target_scaler(ytr_t, y_scaler)
  yva_s <- apply_target_scaler(yva_t, y_scaler)

  model <- GeoHashKrigingNet(
    c_tab = ncol(Xtr),
    d = d,
    tab_hidden = tab_hidden,
    hash_levels = hash_levels,
    hash_hidden = hash_hidden,
    hash_size = hash_size,
    fusion_hidden = fusion_hidden,
    tab_scale = tab_scale,
    hash_scale = hash_scale,
    fuse_scale = fuse_scale,
    dropout = dropout,
    beta_init = beta_init
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$tab_encoder$parameters,
      model$hash_encoder$parameters,
      model$fuse_residual$parameters,
      model$target_head$parameters,
      model$neighbor_head$parameters
    )
    warmup_opt <- optim_adamw(warmup_params, lr = lr, weight_decay = wd)

    for (ep in seq_len(warmup_epochs)) {
      model$train()
      batches <- make_batches(nrow(Xtr), batch_size = batch_size)
      train_loss <- 0

      for (batch_id in seq_along(batches)) {
        b <- batches[[batch_id]]
        nb <- fd$neighbor_idx$train[b, , drop = FALSE]

        xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
        cb <- to_float_tensor(Ctr[b, , drop = FALSE], device = device)
        yb <- to_float_tensor(ytr_s[b], device = device)
        xnb <- to_float_tensor(gather_neighbor_array(Xtr, nb), device = device)
        cnb <- to_float_tensor(gather_neighbor_array(Ctr, nb), device = device)
        ynb <- to_float_tensor(gather_neighbor_matrix(ytr_s, nb), device = device)

        out <- model(xb, cb, xnb, cnb, ynb, use_residual = FALSE)
        loss <- huber_loss(yb, out$base_pred)

        warmup_opt$zero_grad()
        loss$backward()
        nn_utils_clip_grad_norm_(warmup_params, max_norm = 2)
        warmup_opt$step()

        train_loss <- train_loss + loss$item()
      }

      cat(sprintf("[GeoHash Warmup] Epoch %d/%d | train_loss=%.4f\n",
                  ep, warmup_epochs, train_loss / length(batches)))
    }
  }

  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)
  best_val <- Inf
  best_state <- NULL
  bad <- 0

  for (ep in seq_len(epochs)) {
    model$train()
    batches <- make_batches(nrow(Xtr), batch_size = batch_size)
    train_loss <- 0

    for (batch_id in seq_along(batches)) {
      b <- batches[[batch_id]]
      nb <- fd$neighbor_idx$train[b, , drop = FALSE]

      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr_s[b], device = device)
      xnb <- to_float_tensor(gather_neighbor_array(Xtr, nb), device = device)
      cnb <- to_float_tensor(gather_neighbor_array(Ctr, nb), device = device)
      ynb <- to_float_tensor(gather_neighbor_matrix(ytr_s, nb), device = device)

      out <- model(xb, cb, xnb, cnb, ynb, use_residual = TRUE)
      loss <- huber_loss(yb, out$pred) + base_loss_weight * huber_loss(yb, out$base_pred)

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[GeoHash] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geohash(
      model = model,
      X_query = Xva,
      coords_query = Cva,
      neighbor_idx = fd$neighbor_idx$val,
      X_ref = Xtr,
      coords_ref = Ctr,
      y_ref = ytr_s,
      device = device,
      batch_size = batch_size
    )
    vloss <- mean((yva_s - val_pred_s)^2)
    cat(sprintf("[GeoHash] Epoch %d complete | train_loss=%.4f | val_mse=%.4f\n",
                ep, train_loss / length(batches), vloss))

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

  preds_scaled <- predict_geohash(
    model = model,
    X_query = Xte,
    coords_query = Cte,
    neighbor_idx = fd$neighbor_idx$test,
    X_ref = Xtr,
    coords_ref = Ctr,
    y_ref = ytr_s,
    device = device,
    batch_size = batch_size
  )

  preds_t <- invert_target_scaler(preds_scaled, y_scaler)
  preds <- inverse_transform_target(preds_t, target_transform)

  list(
    pred_test = preds,
    metrics_test = metrics(yte, preds)
  )
}

geohashkrigingnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 160,
  tab_hidden = c(192, 96),
  hash_levels = c(2, 4, 8, 16),
  hash_hidden = 64,
  hash_size = 257,
  fusion_hidden = 160,
  tab_scale = 1.0,
  hash_scale = 1.0,
  fuse_scale = 0.5,
  dropout = 0.10,
  beta_init = -4,
  warmup_epochs = 4,
  base_loss_weight = 0.35,
  target_transform = "identity",
  device = "cpu"
)

geohashkrigingnet_quick_params <- modifyList(
  geohashkrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    tab_hidden = c(160, 80),
    hash_hidden = 48,
    fusion_hidden = 128,
    warmup_epochs = 3
  )
)

geohashkrigingnet_conservative_quick_params <- modifyList(
  geohashkrigingnet_quick_params,
  list(
    lr = 1e-4,
    hash_scale = 0.35,
    fuse_scale = 0.20,
    warmup_epochs = 6,
    base_loss_weight = 0.45
  )
)

run_geohash_on_fixed_benchmark <- function(benchmark,
                                           context = wadoux_context,
                                           model_params = geohashkrigingnet_params,
                                           K_neighbors = 12) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoHash Fair] split %s | K_neighbors=%d\n",
                sp$split_id, K_neighbors))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geohash_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K_neighbors = K_neighbors
    )

    out <- do.call(train_geohash_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoHashKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geohashkrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                 sample_size = 250,
                                                 sampling = "simple_random",
                                                 n_folds = 5,
                                                 val_dist_km = 350,
                                                 val_frac = 0.2,
                                                 max_splits = 5,
                                                 seed = 123,
                                                 model_params = geohashkrigingnet_quick_params,
                                                 K_neighbors = 12,
                                                 cubist_committees = 50,
                                                 cubist_neighbors = 5,
                                                 results_dir = "results/geohashkrigingnet_vs_cubist",
                                                 save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoHashKrigingNet vs Cubist BENCHMARK\n")
  cat("========================================\n")

  benchmark <- build_pointpatch_fixed_spatial_kfold_benchmark(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed
  )

  manifest <- pointpatch_benchmark_manifest(benchmark)
  if (save_outputs) {
    write.csv(benchmark$calibration_df, file.path(results_dir, "fixed_calibration_sample.csv"), row.names = FALSE)
    write.csv(manifest, file.path(results_dir, "fixed_split_manifest.csv"), row.names = FALSE)
  }

  cubist_res <- run_cubist_on_pointpatch_benchmark(
    benchmark = benchmark,
    context = context,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors
  )

  geo_res <- run_geohash_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params,
    K_neighbors = K_neighbors
  )

  final <- bind_rows(cubist_res, geo_res)
  summary_tbl <- final %>%
    group_by(model) %>%
    summarise(
      RMSE_mean = mean(RMSE, na.rm = TRUE),
      R2_mean = mean(R2, na.rm = TRUE),
      MAE_mean = mean(MAE, na.rm = TRUE),
      Bias_mean = mean(Bias, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(RMSE_mean)

  if (save_outputs) {
    write.csv(cubist_res, file.path(results_dir, "cubist_results.csv"), row.names = FALSE)
    write.csv(geo_res, file.path(results_dir, "geohashkrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geohashkrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geohashkrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geohashkrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                         sample_size = 300,
                                                         sampling = "simple_random",
                                                         n_folds = 10,
                                                         val_dist_km = 350,
                                                         val_frac = 0.2,
                                                         max_splits = 10,
                                                         seed = 123,
                                                         model_params = geohashkrigingnet_params,
                                                         K_neighbors = 12,
                                                         cubist_committees = 50,
                                                         cubist_neighbors = 5,
                                                         results_dir = "results/geohashkrigingnet_vs_cubist_confirmation",
                                                         save_outputs = TRUE) {
  run_geohashkrigingnet_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    model_params = model_params,
    K_neighbors = K_neighbors,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

make_geohash_v2_variants <- function() {
  list(
    GeoHash_Base = list(
      params = geohashkrigingnet_quick_params,
      K_neighbors = 12
    ),
    GeoHash_Conservative = list(
      params = geohashkrigingnet_conservative_quick_params,
      K_neighbors = 12
    )
  )
}

run_geohash_v2_search <- function(context = wadoux_context,
                                  sample_size = 250,
                                  sampling = "simple_random",
                                  n_folds = 5,
                                  val_dist_km = 350,
                                  val_frac = 0.2,
                                  max_splits = 5,
                                  seed = 123,
                                  cubist_committees = 50,
                                  cubist_neighbors = 5,
                                  results_dir = "results/geohashkrigingnet_v2_search",
                                  save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoHash V2 SEARCH\n")
  cat("========================================\n")

  benchmark <- build_pointpatch_fixed_spatial_kfold_benchmark(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed
  )

  manifest <- pointpatch_benchmark_manifest(benchmark)
  if (save_outputs) {
    write.csv(benchmark$calibration_df, file.path(results_dir, "fixed_calibration_sample.csv"), row.names = FALSE)
    write.csv(manifest, file.path(results_dir, "fixed_split_manifest.csv"), row.names = FALSE)
  }

  cubist_res <- run_cubist_on_pointpatch_benchmark(
    benchmark = benchmark,
    context = context,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors
  )

  variants <- make_geohash_v2_variants()
  all_results <- list()

  for (variant_name in names(variants)) {
    spec <- variants[[variant_name]]
    geo_res <- run_geohash_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      model_params = spec$params,
      K_neighbors = spec$K_neighbors
    ) %>%
      mutate(variant = variant_name)

    cubist_variant <- cubist_res %>%
      mutate(variant = variant_name)

    partial <- bind_rows(cubist_variant, geo_res)
    all_results[[variant_name]] <- partial

    if (save_outputs) {
      write.csv(
        partial,
        file.path(results_dir, paste0("partial_", variant_name, ".csv")),
        row.names = FALSE
      )
    }
  }

  final <- bind_rows(all_results)
  summary_tbl <- final %>%
    group_by(variant, model) %>%
    summarise(
      RMSE_mean = mean(RMSE, na.rm = TRUE),
      R2_mean = mean(R2, na.rm = TRUE),
      MAE_mean = mean(MAE, na.rm = TRUE),
      Bias_mean = mean(Bias, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(variant, RMSE_mean)

  if (save_outputs) {
    write.csv(final, file.path(results_dir, "geohash_v2_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geohash_v2_summary.csv"), row.names = FALSE)
  }

  final
}
