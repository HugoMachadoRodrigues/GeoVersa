rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoMoEKrigingNet
# -----------------------------------------------------------------------------
# Spatial mixture-of-experts for point prediction with an explicit kriging-like
# residual correction.
#
# Core idea:
# - each target point belongs to a local pedo-environment regime
# - a gating network decides which expert(s) should dominate at that point
# - the gate uses:
#     * target covariates
#     * target coordinates
#     * pooled neighbor context
# - each expert predicts a base value for the point
# - neighbor residuals are then propagated with an anisotropic kriging-like head
#
# Plain-language architecture:
# 1. Encode the target point.
# 2. Encode K nearest sampled neighbors.
# 3. Pool neighbor embeddings into a local context vector.
# 4. Use a spatial gating network to mix several experts.
# 5. Compute a base prediction for target and neighbors.
# 6. Compute neighbor residuals.
# 7. Apply anisotropic residual correction.
# 8. Final prediction = expert mixture base + residual correction.
#
# Why this is interesting for pedometry:
# - soil-landscape relationships are often locally heterogeneous
# - Cubist already succeeds by partitioning the feature space into local rules
# - GeoMoEKrigingNet is a neural generalization of that idea with a spatial gate
# - the kriging-like layer keeps an explicit geo-statistical interpretation
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

prepare_geomoe_fold <- function(context,
                                calibration_df,
                                train_idx,
                                val_idx,
                                test_idx,
                                use_robust_scaling = TRUE,
                                K = 12) {
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

  k_train <- min(K, nrow(coords_train_s) - 1)
  if (k_train < 1) stop("Need at least two training points.")

  list(
    X = list(train = X_train_s, val = X_val_s, test = X_test_s),
    y = list(train = y_train, val = y_val, test = y_test),
    coords = list(train = coords_train, val = coords_val, test = coords_test),
    coords_scaled = list(train = coords_train_s, val = coords_val_s, test = coords_test_s),
    scalers = list(x = x_scaler, coords = coord_scaler),
    neighbor_idx = list(
      train = compute_neighbor_idx_train_only(coords_train_s, k_train),
      val = compute_neighbor_idx_cross(coords_val_s, coords_train_s, k_train),
      test = compute_neighbor_idx_cross(coords_test_s, coords_train_s, k_train)
    )
  )
}

AnisotropicResidualKrigingLayer <- nn_module(
  "AnisotropicResidualKrigingLayer",
  initialize = function(d = 256, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5) {
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
    list(delta = delta, w = w, aniso_dist = aniso_dist)
  }
)

ExpertScalarHead <- nn_module(
  "ExpertScalarHead",
  initialize = function(d = 256, hidden = 128, dropout = 0.10) {
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

GeoMoEKrigingNet <- nn_module(
  "GeoMoEKrigingNet",
  initialize = function(c_tab,
                        d = 192,
                        n_experts = 4,
                        tab_hidden = c(192, 96),
                        coord_hidden = c(48),
                        coord_dim = 48,
                        context_hidden = c(128),
                        gate_hidden = c(128, 64),
                        expert_hidden = 128,
                        dropout = 0.10,
                        beta_init = -4) {
    self$d <- d
    self$n_experts <- n_experts

    self$enc_tab <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = dropout)
    self$enc_coord <- make_mlp(2, hidden = coord_hidden, out_dim = coord_dim, dropout = dropout)
    self$proj_coord <- nn_linear(coord_dim, d)
    self$context_proj <- make_mlp(d, hidden = context_hidden, out_dim = d, dropout = dropout)

    self$gate_net <- nn_sequential(
      nn_linear(3 * d, gate_hidden[1]),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(gate_hidden[1], gate_hidden[2]),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(gate_hidden[2], n_experts)
    )

    experts <- vector("list", n_experts)
    for (i in seq_len(n_experts)) {
      experts[[i]] <- ExpertScalarHead(d = d, hidden = expert_hidden, dropout = dropout)
    }
    self$experts <- nn_module_list(experts)

    self$fuse <- nn_sequential(
      nn_linear(3 * d, d),
      nn_gelu(),
      nn_dropout(dropout)
    )

    self$krig <- AnisotropicResidualKrigingLayer(d = d, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_points = function(x_tab, coords) {
    z_tab <- self$enc_tab(x_tab)
    z_coord <- self$proj_coord(self$enc_coord(coords))
    list(z_tab = z_tab, z_coord = z_coord)
  },

  pool_context = function(z_neighbors) {
    self$context_proj(torch_mean(z_neighbors, dim = 2))
  },

  gate_from_features = function(z_tab, z_coord, z_ctx) {
    gate_in <- torch_cat(list(z_tab, z_coord, z_ctx), dim = 2)
    nnf_softmax(self$gate_net(gate_in), dim = 2)
  },

  mix_experts = function(z, gates) {
    preds <- lapply(self$experts, function(expert) expert(z))
    pred_mat <- torch_cat(preds, dim = 2)
    torch_sum(pred_mat * gates, dim = 2)
  },

  forward = function(x_target, coords_target, x_neighbors, coords_neighbors, y_neighbors) {
    dims <- as.integer(x_neighbors$shape)
    B <- dims[1]
    K <- dims[2]
    P <- dims[3]

    enc_t <- self$encode_points(x_target, coords_target)

    xnb_flat <- reshape_safe(x_neighbors, c(B * K, P))
    cnb_flat <- reshape_safe(coords_neighbors, c(B * K, 2))
    enc_n_flat <- self$encode_points(xnb_flat, cnb_flat)
    z_tab_n <- reshape_safe(enc_n_flat$z_tab, c(B, K, self$d))
    z_coord_n <- reshape_safe(enc_n_flat$z_coord, c(B, K, self$d))

    z_ctx <- self$pool_context(z_tab_n + z_coord_n)
    z_t <- self$fuse(torch_cat(list(enc_t$z_tab, enc_t$z_coord, z_ctx), dim = 2))

    z_ctx_n <- z_ctx$unsqueeze(2)$expand(c(B, K, self$d))
    z_n <- self$fuse(
      reshape_safe(
        torch_cat(list(z_tab_n, z_coord_n, z_ctx_n), dim = 3),
        c(B * K, 3 * self$d)
      )
    )
    z_n <- reshape_safe(z_n, c(B, K, self$d))

    gate_t <- self$gate_from_features(enc_t$z_tab, enc_t$z_coord, z_ctx)
    base_target <- self$mix_experts(z_t, gate_t)

    gate_n <- self$gate_from_features(
      reshape_safe(z_tab_n, c(B * K, self$d)),
      reshape_safe(z_coord_n, c(B * K, self$d)),
      reshape_safe(z_ctx_n, c(B * K, self$d))
    )
    base_neighbors <- self$mix_experts(
      reshape_safe(z_n, c(B * K, self$d)),
      gate_n
    )
    base_neighbors <- reshape_safe(base_neighbors, c(B, K))

    residual_neighbors <- y_neighbors - base_neighbors
    k <- self$krig(z_t, coords_target, z_n, coords_neighbors, residual_neighbors)
    beta <- torch_sigmoid(self$logit_beta)
    pred <- base_target + beta * k$delta

    list(
      pred = pred,
      base_pred = base_target,
      z = z_t,
      z_neighbors = z_n,
      gate_target = gate_t,
      gate_neighbors = gate_n,
      base_neighbors = base_neighbors,
      delta = k$delta,
      beta = beta
    )
  }
)

gate_entropy_penalty <- function(gates, eps = 1e-6) {
  g <- torch_clamp(gates, min = eps, max = 1 - eps)
  torch_mean(torch_sum(g * torch_log(g), dim = 2))
}

predict_geomoekrigingnet <- function(model,
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
          "GeoMoEKrigingNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geomoekrigingnet_one_fold <- function(fd,
                                            epochs = 40,
                                            lr = 2e-4,
                                            wd = 1e-3,
                                            batch_size = 96,
                                            patience = 8,
                                            d = 192,
                                            n_experts = 4,
                                            tab_hidden = c(192, 96),
                                            coord_hidden = c(48),
                                            coord_dim = 48,
                                            context_hidden = c(128),
                                            gate_hidden = c(128, 64),
                                            expert_hidden = 128,
                                            dropout = 0.10,
                                            beta_init = -4,
                                            warmup_epochs = 6,
                                            base_loss_weight = 0.35,
                                            gate_entropy_weight = 0.01,
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

  model <- GeoMoEKrigingNet(
    c_tab = ncol(Xtr),
    d = d,
    n_experts = n_experts,
    tab_hidden = tab_hidden,
    coord_hidden = coord_hidden,
    coord_dim = coord_dim,
    context_hidden = context_hidden,
    gate_hidden = gate_hidden,
    expert_hidden = expert_hidden,
    dropout = dropout,
    beta_init = beta_init
  )
  model$to(device = device)

  warmup_params <- c(
    model$enc_tab$parameters,
    model$enc_coord$parameters,
    model$proj_coord$parameters,
    model$context_proj$parameters,
    model$fuse$parameters,
    model$experts$parameters
  )

  if (warmup_epochs > 0) {
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

        out <- model(xb, cb, xnb, cnb, ynb)
        uniform_gate <- torch_full_like(out$gate_target, 1 / n_experts)
        base_pred <- model$mix_experts(out$z, uniform_gate)
        loss <- huber_loss(yb, base_pred)

        warmup_opt$zero_grad()
        loss$backward()
        nn_utils_clip_grad_norm_(warmup_params, max_norm = 2)
        warmup_opt$step()

        train_loss <- train_loss + loss$item()
      }

      cat(sprintf("[GeoMoEKrigingNet Warmup] Epoch %d/%d | train_loss=%.4f\n",
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

      out <- model(xb, cb, xnb, cnb, ynb)
      loss <- huber_loss(yb, out$pred) +
        base_loss_weight * huber_loss(yb, out$base_pred) +
        gate_entropy_weight * (gate_entropy_penalty(out$gate_target) + gate_entropy_penalty(out$gate_neighbors))

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[GeoMoEKrigingNet] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geomoekrigingnet(
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
    cat(sprintf("[GeoMoEKrigingNet] Epoch %d complete | train_loss=%.4f | val_mse=%.4f\n",
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

  preds_scaled <- predict_geomoekrigingnet(
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

geomoekrigingnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 192,
  n_experts = 4,
  tab_hidden = c(192, 96),
  coord_hidden = c(48),
  coord_dim = 48,
  context_hidden = c(128),
  gate_hidden = c(128, 64),
  expert_hidden = 128,
  dropout = 0.10,
  beta_init = -4,
  warmup_epochs = 6,
  base_loss_weight = 0.35,
  gate_entropy_weight = 0.01,
  target_transform = "identity",
  device = "cpu"
)

geomoekrigingnet_quick_params <- modifyList(
  geomoekrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    n_experts = 3,
    tab_hidden = c(128),
    coord_hidden = c(32),
    coord_dim = 32,
    context_hidden = c(96),
    gate_hidden = c(96, 48),
    expert_hidden = 96,
    warmup_epochs = 4,
    gate_entropy_weight = 0.02
  )
)

run_geomoekrigingnet_on_fixed_benchmark <- function(benchmark,
                                                    context = wadoux_context,
                                                    model_params = geomoekrigingnet_params,
                                                    K = 12) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoMoEKrigingNet Fair] split %s | K=%d\n", sp$split_id, K))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geomoe_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K = K
    )

    out <- do.call(train_geomoekrigingnet_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoMoEKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geomoekrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                sample_size = 250,
                                                sampling = "simple_random",
                                                n_folds = 5,
                                                val_dist_km = 350,
                                                val_frac = 0.2,
                                                max_splits = 5,
                                                seed = 123,
                                                model_params = geomoekrigingnet_quick_params,
                                                K = 12,
                                                cubist_committees = 50,
                                                cubist_neighbors = 5,
                                                results_dir = "results/geomoekrigingnet_vs_cubist",
                                                save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoMoEKrigingNet vs Cubist BENCHMARK\n")
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

  geo_res <- run_geomoekrigingnet_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params,
    K = K
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
    write.csv(geo_res, file.path(results_dir, "geomoekrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geomoekrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geomoekrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geomoekrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                        sample_size = 300,
                                                        sampling = "simple_random",
                                                        n_folds = 10,
                                                        val_dist_km = 350,
                                                        val_frac = 0.2,
                                                        max_splits = 10,
                                                        seed = 123,
                                                        model_params = geomoekrigingnet_params,
                                                        K = 12,
                                                        cubist_committees = 50,
                                                        cubist_neighbors = 5,
                                                        results_dir = "results/geomoekrigingnet_vs_cubist_confirmation",
                                                        save_outputs = TRUE) {
  run_geomoekrigingnet_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    model_params = model_params,
    K = K,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

# Example:
# source("code/GeoMoEKrigingNet.R")
# res_geomoe <- run_geomoekrigingnet_vs_cubist_fair(
#   context = wadoux_context,
#   sample_size = 250,
#   n_folds = 5,
#   max_splits = 5,
#   model_params = geomoekrigingnet_quick_params,
#   K = 12
# )
# res_geomoe %>%
#   dplyr::group_by(model) %>%
#   dplyr::summarise(
#     RMSE_mean = mean(RMSE, na.rm = TRUE),
#     R2_mean = mean(R2, na.rm = TRUE),
#     MAE_mean = mean(MAE, na.rm = TRUE),
#     Bias_mean = mean(Bias, na.rm = TRUE),
#     .groups = "drop"
#   )
