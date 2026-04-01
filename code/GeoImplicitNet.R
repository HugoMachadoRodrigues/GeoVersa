rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoImplicitNet
# -----------------------------------------------------------------------------
# Point-based implicit neural representation with:
# - strong tabular MLP branch
# - continuous coordinate field via Fourier positional encoding + implicit MLP
# - conservative gated fusion
# - light anisotropic residual kriging-like correction
#
# Plain-language flow:
# 1. Encode covariates into a robust base latent state.
# 2. Encode coordinates as a continuous field with Fourier features.
# 3. Let the field adjust the tabular latent state conservatively.
# 4. Predict target and neighbor bases.
# 5. Use neighbor residuals for a light anisotropic spatial correction.
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

prepare_geoimplicit_fold <- function(context,
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

FourierPositionalEncoding <- nn_module(
  "FourierPositionalEncoding",
  initialize = function(n_frequencies = 16, freq_scale = 2.0) {
    self$n_frequencies <- n_frequencies
    self$B <- nn_parameter(torch_randn(c(n_frequencies, 2)) * freq_scale)
  },
  forward = function(coords) {
    proj <- torch_matmul(coords, self$B$t())
    torch_cat(list(coords, torch_sin(proj), torch_cos(proj)), dim = 2)
  }
)

GeoImplicitField <- nn_module(
  "GeoImplicitField",
  initialize = function(n_frequencies = 16,
                        freq_scale = 2.0,
                        hidden = c(192, 96),
                        out_dim = 160,
                        dropout = 0.10) {
    self$pe <- FourierPositionalEncoding(
      n_frequencies = n_frequencies,
      freq_scale = freq_scale
    )
    in_dim <- 2 + 2 * n_frequencies
    self$field <- make_mlp(
      in_dim,
      hidden = hidden,
      out_dim = out_dim,
      dropout = dropout
    )
  },
  forward = function(coords) {
    self$field(self$pe(coords))
  }
)

GeoImplicitScalarHead <- nn_module(
  "GeoImplicitScalarHead",
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

GeoImplicitNet <- nn_module(
  "GeoImplicitNet",
  initialize = function(c_tab,
                        d = 160,
                        tab_hidden = c(192, 96),
                        n_frequencies = 16,
                        freq_scale = 2.0,
                        field_hidden = c(192, 96),
                        fusion_hidden = 160,
                        dropout = 0.10,
                        field_scale = 1.0,
                        gate_bias_init = 0.0,
                        beta_init = -4) {
    self$d <- d
    self$field_scale <- field_scale
    self$tab_encoder <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = dropout)
    self$field_encoder <- GeoImplicitField(
      n_frequencies = n_frequencies,
      freq_scale = freq_scale,
      hidden = field_hidden,
      out_dim = d,
      dropout = dropout
    )
    self$field_residual <- nn_sequential(
      nn_linear(2 * d, fusion_hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(fusion_hidden, d)
    )
    self$field_gate <- nn_sequential(
      nn_linear(2 * d, d),
      nn_gelu(),
      nn_linear(d, d)
    )
    self$gate_bias <- nn_parameter(torch_full(c(d), gate_bias_init))
    self$target_head <- GeoImplicitScalarHead(d = d, hidden = 96, dropout = dropout)
    self$neighbor_head <- GeoImplicitScalarHead(d = d, hidden = 96, dropout = dropout)
    self$krig <- AnisotropicResidualKrigingLayer(d = d, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_points = function(x, coords) {
    z_tab <- self$tab_encoder(x)
    z_field <- self$field_encoder(coords)
    pair <- torch_cat(list(z_tab, z_field), dim = 2)
    delta_field <- self$field_residual(pair)
    gate <- torch_sigmoid(self$field_gate(pair) + self$gate_bias)
    scaled_delta <- self$field_scale * gate * delta_field
    z <- z_tab + scaled_delta
    list(z = z, z_tab = z_tab, field_delta = scaled_delta)
  },

  encode_neighbors = function(x_neighbors, coords_neighbors) {
    dims <- as.integer(x_neighbors$shape)
    x_flat <- reshape_safe(x_neighbors, c(dims[1] * dims[2], dims[3]))
    c_flat <- reshape_safe(coords_neighbors, c(dims[1] * dims[2], 2))
    enc <- self$encode_points(x_flat, c_flat)
    list(
      z = reshape_safe(enc$z, c(dims[1], dims[2], self$d)),
      z_tab = reshape_safe(enc$z_tab, c(dims[1], dims[2], self$d)),
      field_delta = reshape_safe(enc$field_delta, c(dims[1], dims[2], self$d))
    )
  },

  forward = function(x_target, coords_target, x_neighbors, coords_neighbors, y_neighbors, use_residual = TRUE) {
    enc_target <- self$encode_points(x_target, coords_target)
    enc_neighbors <- self$encode_neighbors(x_neighbors, coords_neighbors)

    z_target <- enc_target$z
    z_neighbors <- enc_neighbors$z

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
      beta = beta,
      field_delta = enc_target$field_delta,
      field_delta_neighbors = enc_neighbors$field_delta
    )
  }
)

predict_geoimplicit <- function(model,
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
          "GeoImplicitNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geoimplicit_one_fold <- function(fd,
                                       epochs = 40,
                                       lr = 2e-4,
                                       wd = 1e-3,
                                       batch_size = 96,
                                       patience = 8,
                                       d = 160,
                                       tab_hidden = c(192, 96),
                                       n_frequencies = 16,
                                       freq_scale = 2.0,
                                       field_hidden = c(192, 96),
                                       fusion_hidden = 160,
                                       dropout = 0.10,
                                       field_scale = 1.0,
                                       gate_bias_init = 0.0,
                                       beta_init = -4,
                                       warmup_epochs = 4,
                                       base_loss_weight = 0.35,
                                       field_penalty_weight = 0.02,
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

  model <- GeoImplicitNet(
    c_tab = ncol(Xtr),
    d = d,
    tab_hidden = tab_hidden,
    n_frequencies = n_frequencies,
    freq_scale = freq_scale,
    field_hidden = field_hidden,
    fusion_hidden = fusion_hidden,
    dropout = dropout,
    field_scale = field_scale,
    gate_bias_init = gate_bias_init,
    beta_init = beta_init
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$tab_encoder$parameters,
      model$field_encoder$parameters,
      model$field_residual$parameters,
      model$field_gate$parameters,
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
        field_penalty <- torch_mean(out$field_delta^2)
        loss <- huber_loss(yb, out$base_pred) + field_penalty_weight * field_penalty

        warmup_opt$zero_grad()
        loss$backward()
        nn_utils_clip_grad_norm_(warmup_params, max_norm = 2)
        warmup_opt$step()

        train_loss <- train_loss + loss$item()
      }

      cat(sprintf("[GeoImplicit Warmup] Epoch %d/%d | train_loss=%.4f\n",
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
      field_penalty <- torch_mean(out$field_delta^2) +
        0.25 * torch_mean(out$field_delta_neighbors^2)
      loss <- huber_loss(yb, out$pred) +
        base_loss_weight * huber_loss(yb, out$base_pred) +
        field_penalty_weight * field_penalty

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[GeoImplicit] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geoimplicit(
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
    cat(sprintf("[GeoImplicit] Epoch %d complete | train_loss=%.4f | val_mse=%.4f\n",
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

  preds_scaled <- predict_geoimplicit(
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

geoimplicitnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 160,
  tab_hidden = c(192, 96),
  n_frequencies = 16,
  freq_scale = 2.0,
  field_hidden = c(192, 96),
  fusion_hidden = 160,
  dropout = 0.10,
  field_scale = 1.0,
  gate_bias_init = 0.0,
  beta_init = -4,
  warmup_epochs = 4,
  base_loss_weight = 0.35,
  field_penalty_weight = 0.02,
  target_transform = "identity",
  device = "cpu"
)

geoimplicitnet_quick_params <- modifyList(
  geoimplicitnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    tab_hidden = c(160, 80),
    n_frequencies = 12,
    field_hidden = c(160, 80),
    fusion_hidden = 128,
    warmup_epochs = 3
  )
)

geoimplicitnet_weakfield_quick_params <- modifyList(
  geoimplicitnet_quick_params,
  list(
    field_scale = 0.25,
    gate_bias_init = -2.5,
    field_penalty_weight = 0.05
  )
)

run_geoimplicit_on_fixed_benchmark <- function(benchmark,
                                               context = wadoux_context,
                                               model_params = geoimplicitnet_params,
                                               K_neighbors = 12) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoImplicit Fair] split %s | K_neighbors=%d\n",
                sp$split_id, K_neighbors))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geoimplicit_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K_neighbors = K_neighbors
    )

    out <- do.call(train_geoimplicit_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoImplicitNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geoimplicitnet_vs_cubist_fair <- function(context = wadoux_context,
                                              sample_size = 250,
                                              sampling = "simple_random",
                                              n_folds = 5,
                                              val_dist_km = 350,
                                              val_frac = 0.2,
                                              max_splits = 5,
                                              seed = 123,
                                              model_params = geoimplicitnet_quick_params,
                                              K_neighbors = 12,
                                              cubist_committees = 50,
                                              cubist_neighbors = 5,
                                              results_dir = "results/geoimplicitnet_vs_cubist",
                                              save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoImplicitNet vs Cubist BENCHMARK\n")
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

  implicit_res <- run_geoimplicit_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params,
    K_neighbors = K_neighbors
  )

  final <- bind_rows(cubist_res, implicit_res)
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
    write.csv(implicit_res, file.path(results_dir, "geoimplicitnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geoimplicitnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geoimplicitnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geoimplicitnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                      sample_size = 300,
                                                      sampling = "simple_random",
                                                      n_folds = 10,
                                                      val_dist_km = 350,
                                                      val_frac = 0.2,
                                                      max_splits = 10,
                                                      seed = 123,
                                                      model_params = geoimplicitnet_params,
                                                      K_neighbors = 12,
                                                      cubist_committees = 50,
                                                      cubist_neighbors = 5,
                                                      results_dir = "results/geoimplicitnet_vs_cubist_confirmation",
                                                      save_outputs = TRUE) {
  run_geoimplicitnet_vs_cubist_fair(
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

make_geoimplicit_weakfield_variants <- function() {
  list(
    GeoImplicitNet_Base = list(params = geoimplicitnet_quick_params),
    GeoImplicitNet_WeakField = list(params = geoimplicitnet_weakfield_quick_params)
  )
}

run_geoimplicitnet_weakfield_search <- function(context = wadoux_context,
                                                sample_size = 250,
                                                sampling = "simple_random",
                                                n_folds = 5,
                                                val_dist_km = 350,
                                                val_frac = 0.2,
                                                max_splits = 5,
                                                seed = 123,
                                                K_neighbors = 12,
                                                cubist_committees = 50,
                                                cubist_neighbors = 5,
                                                variants = make_geoimplicit_weakfield_variants(),
                                                results_dir = "results/geoimplicitnet_weakfield_search",
                                                save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

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

  variant_results <- vector("list", length(variants))
  vnames <- names(variants)

  for (i in seq_along(variants)) {
    v <- variants[[i]]
    vname <- vnames[[i]]

    res <- run_geoimplicit_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      model_params = v$params,
      K_neighbors = K_neighbors
    ) %>%
      mutate(variant = vname)

    cb <- cubist_res %>% mutate(variant = vname)
    variant_results[[i]] <- bind_rows(cb, res)

    if (save_outputs) {
      write.csv(
        variant_results[[i]],
        file.path(results_dir, sprintf("partial_%s.csv", vname)),
        row.names = FALSE
      )
    }
  }

  final <- bind_rows(variant_results)
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
    write.csv(final, file.path(results_dir, "geoimplicitnet_weakfield_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geoimplicitnet_weakfield_summary.csv"), row.names = FALSE)
  }

  final
}
