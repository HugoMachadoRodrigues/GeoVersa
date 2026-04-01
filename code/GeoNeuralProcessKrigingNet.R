rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoNeuralProcessKrigingNet
# -----------------------------------------------------------------------------
# Point-based neural process with:
# - target branch
# - attentive context aggregation over K nearest observed neighbors
# - anisotropic kriging-like residual correction
#
# Plain-language flow:
# 1. Encode the target point from covariates + coordinates.
# 2. Encode observed context neighbors in two ways:
#    - a base encoder for neighbor latent states
#    - a conditional encoder that also sees relative geometry and observed y
# 3. Use target-conditioned attention to summarize the local context set.
# 4. Predict a base value for the target.
# 5. Predict base values for neighbors and form neighbor residuals.
# 6. Apply an anisotropic residual correction to the target prediction.
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

prepare_geonp_fold <- function(context,
                               calibration_df,
                               train_idx,
                               val_idx,
                               test_idx,
                               use_robust_scaling = TRUE,
                               K_context = 12) {
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

  k_eff <- min(K_context, nrow(coords_train_s) - 1)
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

GeoNPScalarHead <- nn_module(
  "GeoNPScalarHead",
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

GeoNeuralProcessKrigingNet <- nn_module(
  "GeoNeuralProcessKrigingNet",
  initialize = function(c_tab,
                        d = 160,
                        target_hidden = c(160, 80),
                        context_hidden = c(128),
                        attn_dim = 64,
                        dropout = 0.10,
                        beta_init = -4) {
    self$d <- d

    self$target_encoder <- make_mlp(c_tab + 2, hidden = target_hidden, out_dim = d, dropout = dropout)
    self$context_base_encoder <- make_mlp(c_tab + 2, hidden = context_hidden, out_dim = d, dropout = dropout)
    self$context_cond_encoder <- make_mlp(c_tab + 6, hidden = context_hidden, out_dim = d, dropout = dropout)

    self$q_proj <- nn_linear(d, attn_dim)
    self$k_proj <- nn_linear(d, attn_dim)
    self$attn_scale <- 1 / sqrt(attn_dim)

    self$fuse <- nn_sequential(
      nn_linear(2 * d, d),
      nn_gelu(),
      nn_dropout(dropout)
    )

    self$target_head <- GeoNPScalarHead(d = d, hidden = 96, dropout = dropout)
    self$neighbor_head <- GeoNPScalarHead(d = d, hidden = 96, dropout = dropout)
    self$krig <- AnisotropicResidualKrigingLayer(d = d, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_context_base = function(x_context, coords_context) {
    feats <- torch_cat(list(x_context, coords_context), dim = 3)
    dims <- as.integer(feats$shape)
    flat <- reshape_safe(feats, c(dims[1] * dims[2], dims[3]))
    z <- self$context_base_encoder(flat)
    reshape_safe(z, c(dims[1], dims[2], self$d))
  },

  encode_context_cond = function(x_context, coords_target, coords_context, y_context) {
    dx <- coords_context[, , 1] - coords_target[, 1]$unsqueeze(2)
    dy <- coords_context[, , 2] - coords_target[, 2]$unsqueeze(2)
    dist <- torch_sqrt(dx^2 + dy^2 + 1e-8)

    feats <- torch_cat(
      list(
        x_context,
        coords_context,
        dx$unsqueeze(3),
        dy$unsqueeze(3),
        dist$unsqueeze(3),
        y_context$unsqueeze(3)
      ),
      dim = 3
    )

    dims <- as.integer(feats$shape)
    flat <- reshape_safe(feats, c(dims[1] * dims[2], dims[3]))
    z <- self$context_cond_encoder(flat)
    reshape_safe(z, c(dims[1], dims[2], self$d))
  },

  aggregate_context = function(z_target, z_context_cond) {
    q <- self$q_proj(z_target)
    k <- self$k_proj(z_context_cond)
    scores <- torch_sum(k * q$unsqueeze(2), dim = 3) * self$attn_scale
    attn <- nnf_softmax(scores, dim = 2)
    z_summary <- torch_sum(z_context_cond * attn$unsqueeze(3), dim = 2)
    list(z_summary = z_summary, attn = attn)
  },

  forward = function(x_target, coords_target, x_context, coords_context, y_context, use_residual = TRUE) {
    z_target <- self$target_encoder(torch_cat(list(x_target, coords_target), dim = 2))
    z_context_base <- self$encode_context_base(x_context, coords_context)
    z_context_cond <- self$encode_context_cond(x_context, coords_target, coords_context, y_context)

    agg <- self$aggregate_context(z_target, z_context_cond)
    z_fused <- self$fuse(torch_cat(list(z_target, agg$z_summary), dim = 2))

    base_target <- flatten_safe(self$target_head(z_fused))
    dims_context <- as.integer(z_context_base$shape)
    base_neighbors <- self$neighbor_head(reshape_safe(z_context_base, c(dims_context[1] * dims_context[2], self$d)))
    base_neighbors <- reshape_safe(flatten_safe(base_neighbors), c(dims_context[1], dims_context[2]))

    if (isTRUE(use_residual)) {
      residual_neighbors <- y_context - base_neighbors
      k <- self$krig(z_fused, coords_target, z_context_base, coords_context, residual_neighbors)
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
      z = z_fused,
      z_context = z_context_base,
      base_neighbors = base_neighbors,
      delta = delta,
      beta = beta,
      attn = agg$attn
    )
  }
)

predict_geoneuralprocesskrigingnet <- function(model,
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
          "GeoNeuralProcessKrigingNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geoneuralprocesskrigingnet_one_fold <- function(fd,
                                                      epochs = 40,
                                                      lr = 2e-4,
                                                      wd = 1e-3,
                                                      batch_size = 96,
                                                      patience = 8,
                                                      d = 160,
                                                      target_hidden = c(160, 80),
                                                      context_hidden = c(128),
                                                      attn_dim = 64,
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

  model <- GeoNeuralProcessKrigingNet(
    c_tab = ncol(Xtr),
    d = d,
    target_hidden = target_hidden,
    context_hidden = context_hidden,
    attn_dim = attn_dim,
    dropout = dropout,
    beta_init = beta_init
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$target_encoder$parameters,
      model$context_base_encoder$parameters,
      model$context_cond_encoder$parameters,
      model$q_proj$parameters,
      model$k_proj$parameters,
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

      cat(sprintf("[GeoNeuralProcess Warmup] Epoch %d/%d | train_loss=%.4f\n",
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
        cat(sprintf("[GeoNeuralProcess] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geoneuralprocesskrigingnet(
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
    cat(sprintf("[GeoNeuralProcess] Epoch %d complete | train_loss=%.4f | val_mse=%.4f\n",
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

  preds_scaled <- predict_geoneuralprocesskrigingnet(
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

geoneuralprocesskrigingnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 160,
  target_hidden = c(160, 80),
  context_hidden = c(128),
  attn_dim = 64,
  dropout = 0.10,
  beta_init = -4,
  warmup_epochs = 4,
  base_loss_weight = 0.35,
  target_transform = "identity",
  device = "cpu"
)

geoneuralprocesskrigingnet_quick_params <- modifyList(
  geoneuralprocesskrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    target_hidden = c(128),
    context_hidden = c(96),
    attn_dim = 48,
    warmup_epochs = 3
  )
)

run_geoneuralprocesskrigingnet_on_fixed_benchmark <- function(benchmark,
                                                              context = wadoux_context,
                                                              model_params = geoneuralprocesskrigingnet_params,
                                                              K_context = 12) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoNeuralProcessKrigingNet Fair] split %s | K_context=%d\n",
                sp$split_id, K_context))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geonp_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K_context = K_context
    )

    out <- do.call(train_geoneuralprocesskrigingnet_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoNeuralProcessKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geoneuralprocesskrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                          sample_size = 250,
                                                          sampling = "simple_random",
                                                          n_folds = 5,
                                                          val_dist_km = 350,
                                                          val_frac = 0.2,
                                                          max_splits = 5,
                                                          seed = 123,
                                                          model_params = geoneuralprocesskrigingnet_quick_params,
                                                          K_context = 12,
                                                          cubist_committees = 50,
                                                          cubist_neighbors = 5,
                                                          results_dir = "results/geoneuralprocesskrigingnet_vs_cubist",
                                                          save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoNeuralProcessKrigingNet vs Cubist BENCHMARK\n")
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

  geo_res <- run_geoneuralprocesskrigingnet_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params,
    K_context = K_context
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
    write.csv(geo_res, file.path(results_dir, "geoneuralprocesskrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geoneuralprocesskrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geoneuralprocesskrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geoneuralprocesskrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                                  sample_size = 300,
                                                                  sampling = "simple_random",
                                                                  n_folds = 10,
                                                                  val_dist_km = 350,
                                                                  val_frac = 0.2,
                                                                  max_splits = 10,
                                                                  seed = 123,
                                                                  model_params = geoneuralprocesskrigingnet_params,
                                                                  K_context = 12,
                                                                  cubist_committees = 50,
                                                                  cubist_neighbors = 5,
                                                                  results_dir = "results/geoneuralprocesskrigingnet_vs_cubist_confirmation",
                                                                  save_outputs = TRUE) {
  run_geoneuralprocesskrigingnet_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    model_params = model_params,
    K_context = K_context,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

# Example:
# source("code/GeoNeuralProcessKrigingNet.R")
# res_geonp <- run_geoneuralprocesskrigingnet_vs_cubist_fair(
#   context = wadoux_context,
#   sample_size = 250,
#   n_folds = 5,
#   max_splits = 5,
#   model_params = geoneuralprocesskrigingnet_quick_params,
#   K_context = 12
# )
