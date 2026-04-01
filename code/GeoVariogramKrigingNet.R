rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoVariogramKrigingNet
# -----------------------------------------------------------------------------
# Point-based model with:
# - neural base predictor
# - learned variogram gamma(h)
# - differentiable ordinary kriging layer over residuals
#
# Core idea:
# 1. Predict a base value for the target point.
# 2. Predict base values for observed neighbors.
# 3. Compute neighbor residuals.
# 4. Learn semivariogram values from pairwise lags.
# 5. Solve the ordinary kriging system inside torch.
# 6. Apply the kriging correction to the target base prediction.
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

prepare_geovariogram_fold <- function(context,
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

ParametricVariogramHead <- nn_module(
  "ParametricVariogramHead",
  initialize = function(init_range_major = 1.0,
                        init_range_minor = 0.7,
                        init_sill = 1.0,
                        init_nugget = 0.05) {
    self$log_range_major <- nn_parameter(torch_log(torch_tensor(init_range_major)))
    self$log_range_minor <- nn_parameter(torch_log(torch_tensor(init_range_minor)))
    self$log_sill <- nn_parameter(torch_log(torch_tensor(init_sill)))
    self$log_nugget <- nn_parameter(torch_log(torch_tensor(init_nugget)))
    self$theta <- nn_parameter(torch_tensor(0))
  },
  forward = function(dx, dy, include_nugget = FALSE) {
    cth <- torch_cos(self$theta)
    sth <- torch_sin(self$theta)

    u <- cth * dx + sth * dy
    v <- -sth * dx + cth * dy

    range_major <- nnf_softplus(self$log_range_major) + 1e-6
    range_minor <- nnf_softplus(self$log_range_minor) + 1e-6
    sill <- nnf_softplus(self$log_sill) + 1e-6
    nugget <- nnf_softplus(self$log_nugget) + 1e-6

    h <- torch_sqrt((u / range_major)^2 + (v / range_minor)^2 + 1e-8)
    gamma <- sill * (1 - torch_exp(-h))

    if (isTRUE(include_nugget)) {
      gamma <- gamma + nugget
    }

    gamma
  }
)

GeoVariogramScalarHead <- nn_module(
  "GeoVariogramScalarHead",
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

GeoVariogramKrigingNet <- nn_module(
  "GeoVariogramKrigingNet",
  initialize = function(c_tab,
                        d = 160,
                        base_hidden = c(160, 80),
                        dropout = 0.10,
                        beta_init = -4,
                        init_nugget = 0.05,
                        init_range_major = 1.0,
                        init_range_minor = 0.7,
                        init_sill = 1.0,
                        jitter = 1e-4) {
    self$d <- d
    self$jitter <- jitter

    self$base_encoder <- make_mlp(c_tab + 2, hidden = base_hidden, out_dim = d, dropout = dropout)
    self$target_head <- GeoVariogramScalarHead(d = d, hidden = 96, dropout = dropout)
    self$neighbor_head <- GeoVariogramScalarHead(d = d, hidden = 96, dropout = dropout)
    self$variogram <- ParametricVariogramHead(
      init_range_major = init_range_major,
      init_range_minor = init_range_minor,
      init_sill = init_sill,
      init_nugget = init_nugget
    )
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_points = function(x, coords) {
    self$base_encoder(torch_cat(list(x, coords), dim = 2))
  },

  encode_neighbors = function(x_neighbors, coords_neighbors) {
    dims <- as.integer(x_neighbors$shape)
    x_flat <- reshape_safe(x_neighbors, c(dims[1] * dims[2], dims[3]))
    c_flat <- reshape_safe(coords_neighbors, c(dims[1] * dims[2], 2))
    z <- self$encode_points(x_flat, c_flat)
    reshape_safe(z, c(dims[1], dims[2], self$d))
  },

  ordinary_kriging_delta = function(coords_target, coords_neighbors, residual_neighbors) {
    dims <- as.integer(coords_neighbors$shape)
    B <- dims[1]
    K <- dims[2]

    dx_qn <- coords_target[, 1]$unsqueeze(2) - coords_neighbors[, , 1]
    dy_qn <- coords_target[, 2]$unsqueeze(2) - coords_neighbors[, , 2]
    gamma_qn <- self$variogram(dx_qn, dy_qn, include_nugget = FALSE)

    x_n <- coords_neighbors[, , 1]
    y_n <- coords_neighbors[, , 2]
    dx_nn <- x_n$unsqueeze(3) - x_n$unsqueeze(2)
    dy_nn <- y_n$unsqueeze(3) - y_n$unsqueeze(2)
    gamma_nn <- self$variogram(dx_nn, dy_nn, include_nugget = FALSE)

    eye <- torch_eye(K, dtype = coords_neighbors$dtype, device = coords_neighbors$device)$unsqueeze(1)$expand(c(B, K, K))
    nugget <- nnf_softplus(self$variogram$log_nugget) + self$jitter

    gamma_nn <- gamma_nn * (torch_ones_like(eye) - eye)
    G <- gamma_nn + nugget * eye

    ones_col <- torch_ones(c(B, K, 1), dtype = coords_neighbors$dtype, device = coords_neighbors$device)
    top <- torch_cat(list(G, ones_col), dim = 3)

    ones_row <- torch_ones(c(B, 1, K), dtype = coords_neighbors$dtype, device = coords_neighbors$device)
    zero_corner <- torch_zeros(c(B, 1, 1), dtype = coords_neighbors$dtype, device = coords_neighbors$device)
    bottom <- torch_cat(list(ones_row, zero_corner), dim = 3)

    A <- torch_cat(list(top, bottom), dim = 2)
    rhs <- torch_cat(
      list(
        gamma_qn,
        torch_ones(c(B, 1), dtype = coords_neighbors$dtype, device = coords_neighbors$device)
      ),
      dim = 2
    )

    sol <- linalg_solve(A, rhs$unsqueeze(3))
    weights <- sol[, 1:K, 1]
    torch_sum(weights * residual_neighbors, dim = 2)
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
      delta <- self$ordinary_kriging_delta(coords_target, coords_neighbors, residual_neighbors)
      beta <- torch_sigmoid(self$logit_beta)
      pred <- base_target + beta * delta
    } else {
      delta <- torch_zeros_like(base_target)
      beta <- torch_sigmoid(self$logit_beta)
      pred <- base_target
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

predict_geovariogramkrigingnet <- function(model,
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
          "GeoVariogramKrigingNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geovariogramkrigingnet_one_fold <- function(fd,
                                                  epochs = 40,
                                                  lr = 2e-4,
                                                  wd = 1e-3,
                                                  batch_size = 96,
                                                  patience = 8,
                                                  d = 160,
                                                  base_hidden = c(160, 80),
                                                  dropout = 0.10,
                                                  beta_init = -4,
                                                  init_nugget = 0.05,
                                                  init_range_major = 1.0,
                                                  init_range_minor = 0.7,
                                                  init_sill = 1.0,
                                                  jitter = 1e-4,
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

  model <- GeoVariogramKrigingNet(
    c_tab = ncol(Xtr),
    d = d,
    base_hidden = base_hidden,
    dropout = dropout,
    beta_init = beta_init,
    init_nugget = init_nugget,
    init_range_major = init_range_major,
    init_range_minor = init_range_minor,
    init_sill = init_sill,
    jitter = jitter
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$base_encoder$parameters,
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

      cat(sprintf("[GeoVariogramKrigingNet Warmup] Epoch %d/%d | train_loss=%.4f\n",
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
        cat(sprintf("[GeoVariogramKrigingNet] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geovariogramkrigingnet(
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
    cat(sprintf("[GeoVariogramKrigingNet] Epoch %d complete | train_loss=%.4f | val_mse=%.4f\n",
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

  preds_scaled <- predict_geovariogramkrigingnet(
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

geovariogramkrigingnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 160,
  base_hidden = c(160, 80),
  dropout = 0.10,
  beta_init = -4,
  init_nugget = 0.05,
  init_range_major = 1.0,
  init_range_minor = 0.7,
  init_sill = 1.0,
  jitter = 1e-4,
  warmup_epochs = 4,
  base_loss_weight = 0.35,
  target_transform = "identity",
  device = "cpu"
)

geovariogramkrigingnet_quick_params <- modifyList(
  geovariogramkrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    base_hidden = c(128),
    warmup_epochs = 3
  )
)

run_geovariogramkrigingnet_on_fixed_benchmark <- function(benchmark,
                                                          context = wadoux_context,
                                                          model_params = geovariogramkrigingnet_params,
                                                          K_neighbors = 12) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoVariogramKrigingNet Fair] split %s | K_neighbors=%d\n",
                sp$split_id, K_neighbors))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geovariogram_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K_neighbors = K_neighbors
    )

    out <- do.call(train_geovariogramkrigingnet_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoVariogramKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geovariogramkrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                      sample_size = 250,
                                                      sampling = "simple_random",
                                                      n_folds = 5,
                                                      val_dist_km = 350,
                                                      val_frac = 0.2,
                                                      max_splits = 5,
                                                      seed = 123,
                                                      model_params = geovariogramkrigingnet_quick_params,
                                                      K_neighbors = 12,
                                                      cubist_committees = 50,
                                                      cubist_neighbors = 5,
                                                      results_dir = "results/geovariogramkrigingnet_vs_cubist",
                                                      save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoVariogramKrigingNet vs Cubist BENCHMARK\n")
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

  geo_res <- run_geovariogramkrigingnet_on_fixed_benchmark(
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
    write.csv(geo_res, file.path(results_dir, "geovariogramkrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geovariogramkrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geovariogramkrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geovariogramkrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                              sample_size = 300,
                                                              sampling = "simple_random",
                                                              n_folds = 10,
                                                              val_dist_km = 350,
                                                              val_frac = 0.2,
                                                              max_splits = 10,
                                                              seed = 123,
                                                              model_params = geovariogramkrigingnet_params,
                                                              K_neighbors = 12,
                                                              cubist_committees = 50,
                                                              cubist_neighbors = 5,
                                                              results_dir = "results/geovariogramkrigingnet_vs_cubist_confirmation",
                                                              save_outputs = TRUE) {
  run_geovariogramkrigingnet_vs_cubist_fair(
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

# Example:
# source("code/GeoVariogramKrigingNet.R")
# res_geovar <- run_geovariogramkrigingnet_vs_cubist_fair(
#   context = wadoux_context,
#   sample_size = 250,
#   n_folds = 5,
#   max_splits = 5,
#   model_params = geovariogramkrigingnet_quick_params,
#   K_neighbors = 12
# )
