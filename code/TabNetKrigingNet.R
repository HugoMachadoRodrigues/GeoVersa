rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# TabNetKrigingNet
# -----------------------------------------------------------------------------
# Point-based model with:
# - TabNet-style attentive feature masking for tabular covariates
# - coordinate encoder
# - anisotropic residual kriging-like correction
#
# Plain-language flow:
# 1. Learn which covariates matter at each decision step.
# 2. Build a stable tabular embedding for the target point.
# 3. Combine it with coordinates.
# 4. Predict target and neighbor bases.
# 5. Use neighbor residuals for a light spatial correction.
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

prepare_tabnet_fold <- function(context,
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
  initialize = function(d = 128, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5) {
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

TabNetFeatureBlock <- nn_module(
  "TabNetFeatureBlock",
  initialize = function(c_tab, n_d = 64, n_a = 64, hidden = 128, dropout = 0.10) {
    self$net <- nn_sequential(
      nn_linear(c_tab, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, n_d + n_a)
    )
  },
  forward = function(x) {
    self$net(x)
  }
)

TabNetAttentiveMask <- nn_module(
  "TabNetAttentiveMask",
  initialize = function(n_a = 64, c_tab = 32, hidden = 64, dropout = 0.05) {
    self$net <- nn_sequential(
      nn_linear(n_a, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, c_tab)
    )
  },
  forward = function(a, prior) {
    logits <- self$net(a)
    nnf_softmax(logits + torch_log(prior + 1e-8), dim = 2)
  }
)

TabNetEncoder <- nn_module(
  "TabNetEncoder",
  initialize = function(c_tab,
                        n_steps = 3,
                        n_d = 64,
                        n_a = 64,
                        hidden = 128,
                        gamma = 1.3,
                        dropout = 0.10) {
    self$c_tab <- c_tab
    self$n_steps <- n_steps
    self$n_d <- n_d
    self$n_a <- n_a
    self$gamma <- gamma

    self$input_norm <- nn_layer_norm(normalized_shape = c_tab)
    self$att_init <- nn_sequential(
      nn_linear(c_tab, n_a),
      nn_gelu()
    )

    feat_blocks <- vector("list", n_steps)
    mask_blocks <- vector("list", n_steps)
    for (i in seq_len(n_steps)) {
      feat_blocks[[i]] <- TabNetFeatureBlock(
        c_tab = c_tab,
        n_d = n_d,
        n_a = n_a,
        hidden = hidden,
        dropout = dropout
      )
      mask_blocks[[i]] <- TabNetAttentiveMask(
        n_a = n_a,
        c_tab = c_tab,
        hidden = max(32, n_a),
        dropout = dropout / 2
      )
    }
    self$feat_blocks <- nn_module_list(feat_blocks)
    self$mask_blocks <- nn_module_list(mask_blocks)
  },
  forward = function(x) {
    x_norm <- self$input_norm(x)
    prior <- torch_ones_like(x_norm)
    a <- self$att_init(x_norm)
    decision_sum <- torch_zeros(c(x_norm$shape[1], self$n_d), device = x_norm$device)

    for (i in seq_len(self$n_steps)) {
      mask <- self$mask_blocks[[i]](a, prior)
      feat <- self$feat_blocks[[i]](x_norm * mask)
      d_part <- nnf_gelu(feat[, 1:self$n_d])
      a <- feat[, (self$n_d + 1):(self$n_d + self$n_a)]
      decision_sum <- decision_sum + d_part
      prior <- torch_clamp(prior * (self$gamma - mask), min = 1e-3, max = 5)
    }

    decision_sum
  }
)

TabNetScalarHead <- nn_module(
  "TabNetScalarHead",
  initialize = function(d = 128, hidden = 96, dropout = 0.10) {
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

TabNetKrigingNet <- nn_module(
  "TabNetKrigingNet",
  initialize = function(c_tab,
                        n_steps = 3,
                        n_d = 64,
                        n_a = 64,
                        tab_hidden = 128,
                        coord_dim = 48,
                        d = 128,
                        gamma = 1.3,
                        dropout = 0.10,
                        beta_init = -4) {
    self$d <- d
    self$tab_encoder <- TabNetEncoder(
      c_tab = c_tab,
      n_steps = n_steps,
      n_d = n_d,
      n_a = n_a,
      hidden = tab_hidden,
      gamma = gamma,
      dropout = dropout
    )
    self$coord_encoder <- make_mlp(2, hidden = c(64), out_dim = coord_dim, dropout = dropout)
    self$fuse <- nn_sequential(
      nn_linear(n_d + coord_dim, d),
      nn_gelu(),
      nn_dropout(dropout)
    )
    self$target_head <- TabNetScalarHead(d = d, hidden = 96, dropout = dropout)
    self$neighbor_head <- TabNetScalarHead(d = d, hidden = 96, dropout = dropout)
    self$krig <- AnisotropicResidualKrigingLayer(d = d, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_points = function(x, coords) {
    z_tab <- self$tab_encoder(x)
    z_coord <- self$coord_encoder(coords)
    self$fuse(torch_cat(list(z_tab, z_coord), dim = 2))
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

predict_tabnetkrigingnet <- function(model,
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
          "TabNetKrigingNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_tabnetkrigingnet_one_fold <- function(fd,
                                            epochs = 40,
                                            lr = 2e-4,
                                            wd = 1e-3,
                                            batch_size = 96,
                                            patience = 8,
                                            n_steps = 3,
                                            n_d = 64,
                                            n_a = 64,
                                            tab_hidden = 128,
                                            coord_dim = 48,
                                            d = 128,
                                            gamma = 1.3,
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

  model <- TabNetKrigingNet(
    c_tab = ncol(Xtr),
    n_steps = n_steps,
    n_d = n_d,
    n_a = n_a,
    tab_hidden = tab_hidden,
    coord_dim = coord_dim,
    d = d,
    gamma = gamma,
    dropout = dropout,
    beta_init = beta_init
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$tab_encoder$parameters,
      model$coord_encoder$parameters,
      model$fuse$parameters,
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

      cat(sprintf("[TabNet Warmup] Epoch %d/%d | train_loss=%.4f\n",
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
        cat(sprintf("[TabNet] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_tabnetkrigingnet(
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
    cat(sprintf("[TabNet] Epoch %d complete | train_loss=%.4f | val_mse=%.4f\n",
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

  preds_scaled <- predict_tabnetkrigingnet(
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

tabnetkrigingnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  n_steps = 3,
  n_d = 64,
  n_a = 64,
  tab_hidden = 128,
  coord_dim = 48,
  d = 128,
  gamma = 1.3,
  dropout = 0.10,
  beta_init = -4,
  warmup_epochs = 4,
  base_loss_weight = 0.35,
  target_transform = "identity",
  device = "cpu"
)

tabnetkrigingnet_quick_params <- modifyList(
  tabnetkrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    n_steps = 3,
    n_d = 48,
    n_a = 48,
    tab_hidden = 96,
    coord_dim = 32,
    d = 96,
    warmup_epochs = 3
  )
)

run_tabnet_on_fixed_benchmark <- function(benchmark,
                                          context = wadoux_context,
                                          model_params = tabnetkrigingnet_params,
                                          K_neighbors = 12) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[TabNet Fair] split %s | K_neighbors=%d\n",
                sp$split_id, K_neighbors))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_tabnet_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K_neighbors = K_neighbors
    )

    out <- do.call(train_tabnetkrigingnet_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "TabNetKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_tabnetkrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                sample_size = 250,
                                                sampling = "simple_random",
                                                n_folds = 5,
                                                val_dist_km = 350,
                                                val_frac = 0.2,
                                                max_splits = 5,
                                                seed = 123,
                                                model_params = tabnetkrigingnet_quick_params,
                                                K_neighbors = 12,
                                                cubist_committees = 50,
                                                cubist_neighbors = 5,
                                                results_dir = "results/tabnetkrigingnet_vs_cubist",
                                                save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING TabNetKrigingNet vs Cubist BENCHMARK\n")
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

  tabnet_res <- run_tabnet_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params,
    K_neighbors = K_neighbors
  )

  final <- bind_rows(cubist_res, tabnet_res)
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
    write.csv(tabnet_res, file.path(results_dir, "tabnetkrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "tabnetkrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "tabnetkrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_tabnetkrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                        sample_size = 300,
                                                        sampling = "simple_random",
                                                        n_folds = 10,
                                                        val_dist_km = 350,
                                                        val_frac = 0.2,
                                                        max_splits = 10,
                                                        seed = 123,
                                                        model_params = tabnetkrigingnet_params,
                                                        K_neighbors = 12,
                                                        cubist_committees = 50,
                                                        cubist_neighbors = 5,
                                                        results_dir = "results/tabnetkrigingnet_vs_cubist_confirmation",
                                                        save_outputs = TRUE) {
  run_tabnetkrigingnet_vs_cubist_fair(
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

