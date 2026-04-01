rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoHyperKrigingNet
# -----------------------------------------------------------------------------
# Point-based geostatistical hypernetwork with:
# - strong tabular + learned spatial-basis context encoder
# - hypernetwork-generated local linear model per point
# - light anisotropic residual kriging-like correction
#
# Plain-language flow:
# 1. Build a context vector from covariates and coordinates.
# 2. Use a hypernetwork to generate a local linear model for each point.
# 3. Predict target and neighbor bases through these local linear models.
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

sample_hyper_centers <- function(coords_train, n_centers = 32, seed = 123) {
  set.seed(seed)
  m_eff <- min(n_centers, nrow(coords_train))
  idx <- sort(sample(seq_len(nrow(coords_train)), size = m_eff))
  coords_train[idx, , drop = FALSE]
}

set_geohyper_seed <- function(seed = 123) {
  seed <- as.integer(seed)
  set.seed(seed)
  torch_manual_seed(seed)
  invisible(seed)
}

make_geohyper_batches <- function(n, batch_size = 256, deterministic = FALSE) {
  if (isTRUE(deterministic)) {
    idx <- seq_len(n)
    return(split(idx, ceiling(seq_along(idx) / batch_size)))
  }
  make_batches(n, batch_size = batch_size)
}

geohyper_local_train_scale <- function(epoch,
                                       epochs,
                                       freeze_epochs = 0,
                                       ramp_epochs = 1) {
  if (epoch <= freeze_epochs) return(0)
  if (ramp_epochs <= 0) return(1)
  scale <- (epoch - freeze_epochs) / ramp_epochs
  max(0, min(1, scale))
}

prepare_geohyper_fold <- function(context,
                                  calibration_df,
                                  train_idx,
                                  val_idx,
                                  test_idx,
                                  use_robust_scaling = TRUE,
                                  K_neighbors = 12,
                                  n_centers = 32,
                                  center_seed = 123) {
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

  init_centers <- sample_hyper_centers(coords_train_s, n_centers = n_centers, seed = center_seed)

  list(
    X = list(train = X_train_s, val = X_val_s, test = X_test_s),
    y = list(train = y_train, val = y_val, test = y_test),
    coords = list(train = coords_train, val = coords_val, test = coords_test),
    coords_scaled = list(train = coords_train_s, val = coords_val_s, test = coords_test_s),
    init_centers = init_centers,
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

LearnedSpatialBasisLayer <- nn_module(
  "LearnedSpatialBasisLayer",
  initialize = function(init_centers,
                        out_dim = 160,
                        hidden = 128,
                        dropout = 0.10,
                        init_range = 1.0) {
    centers_tensor <- torch_tensor(init_centers, dtype = torch_float())
    self$n_centers <- nrow(init_centers)
    self$centers <- nn_parameter(centers_tensor$clone())
    self$log_range <- nn_parameter(torch_log(torch_tensor(init_range)))
    self$head <- nn_sequential(
      nn_linear(self$n_centers + 2, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, out_dim),
      nn_gelu()
    )
  },
  forward = function(coords) {
    dx <- coords[, 1]$unsqueeze(2) - self$centers[, 1]$view(c(1, self$n_centers))
    dy <- coords[, 2]$unsqueeze(2) - self$centers[, 2]$view(c(1, self$n_centers))
    dist2 <- dx^2 + dy^2
    range <- nnf_softplus(self$log_range) + 1e-6
    basis <- torch_exp(-dist2 / (2 * range^2))
    feat <- torch_cat(list(basis, coords), dim = 2)
    self$head(feat)
  }
)

GeoHyperContextEncoder <- nn_module(
  "GeoHyperContextEncoder",
  initialize = function(c_tab,
                        init_centers,
                        d = 160,
                        tab_hidden = c(192, 96),
                        basis_hidden = 128,
                        fusion_hidden = 160,
                        dropout = 0.10) {
    self$d <- d
    self$tab_encoder <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = dropout)
    self$basis_encoder <- LearnedSpatialBasisLayer(
      init_centers = init_centers,
      out_dim = d,
      hidden = basis_hidden,
      dropout = dropout,
      init_range = 1.0
    )
    self$fuse_residual <- nn_sequential(
      nn_linear(2 * d, fusion_hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(fusion_hidden, d)
    )
  },
  forward = function(x, coords) {
    z_tab <- self$tab_encoder(x)
    z_basis <- self$basis_encoder(coords)
    z_tab + self$fuse_residual(torch_cat(list(z_tab, z_basis), dim = 2))
  }
)

GeoHyperLinearHead <- nn_module(
  "GeoHyperLinearHead",
  initialize = function(d_context = 160, c_tab = 32, hidden = 128, coef_scale = 1.0, adjust_scale = 0.25, dropout = 0.10) {
    self$c_tab <- c_tab
    self$coef_scale <- coef_scale
    self$adjust_scale <- adjust_scale
    self$coef_head <- nn_sequential(
      nn_linear(d_context, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, c_tab)
    )
    self$bias_head <- nn_sequential(
      nn_linear(d_context, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, 1)
    )
    self$nonlinear_head <- nn_sequential(
      nn_linear(d_context, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, 1)
    )
  },
  forward = function(h, x) {
    coefs <- self$coef_scale * torch_tanh(self$coef_head(h))
    bias <- flatten_safe(self$bias_head(h))
    delta_nl <- 0.10 * flatten_safe(self$nonlinear_head(h))
    adjustment <- self$adjust_scale * (bias + torch_sum(coefs * x, dim = 2) + delta_nl)
    list(adjustment = adjustment, coefs = coefs, bias = bias, delta_nl = delta_nl)
  }
)

GeoHyperBaseHead <- nn_module(
  "GeoHyperBaseHead",
  initialize = function(d_context = 160, hidden = c(96), dropout = 0.10) {
    self$net <- make_mlp(d_context, hidden = hidden, out_dim = 1, dropout = dropout)
  },
  forward = function(h) {
    flatten_safe(self$net(h))
  }
)

GeoHyperKrigingNet <- nn_module(
  "GeoHyperKrigingNet",
  initialize = function(c_tab,
                        init_centers,
                        d = 160,
                        tab_hidden = c(192, 96),
                        basis_hidden = 128,
                        fusion_hidden = 160,
                        base_hidden = c(96),
                        hyper_hidden = 128,
                        coef_scale = 1.0,
                        adjust_scale = 0.25,
                        local_gate_scale = 1.0,
                        gate_bias_init = -2.0,
                        dropout = 0.10,
                        beta_init = -4) {
    self$d <- d
    self$c_tab <- c_tab
    self$context_encoder <- GeoHyperContextEncoder(
      c_tab = c_tab,
      init_centers = init_centers,
      d = d,
      tab_hidden = tab_hidden,
      basis_hidden = basis_hidden,
      fusion_hidden = fusion_hidden,
      dropout = dropout
    )
    self$base_head <- GeoHyperBaseHead(
      d_context = d,
      hidden = base_hidden,
      dropout = dropout
    )
    self$linear_head <- GeoHyperLinearHead(
      d_context = d,
      c_tab = c_tab,
      hidden = hyper_hidden,
      coef_scale = coef_scale,
      adjust_scale = adjust_scale,
      dropout = dropout
    )
    self$local_gate_head <- nn_linear(d, 1)
    self$local_gate_scale <- local_gate_scale
    self$local_gate_bias <- nn_parameter(torch_tensor(gate_bias_init))
    self$krig <- AnisotropicResidualKrigingLayer(d = d, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_neighbors = function(x_neighbors, coords_neighbors) {
    dims <- as.integer(x_neighbors$shape)
    x_flat <- reshape_safe(x_neighbors, c(dims[1] * dims[2], dims[3]))
    c_flat <- reshape_safe(coords_neighbors, c(dims[1] * dims[2], 2))
    h <- self$context_encoder(x_flat, c_flat)
    reshape_safe(h, c(dims[1], dims[2], self$d))
  },

  predict_from_context = function(h, x, local_scale = 1.0) {
    base <- self$base_head(h)
    local <- self$linear_head(h, x)
    gate <- local_scale * self$local_gate_scale * torch_sigmoid(flatten_safe(self$local_gate_head(h)) + self$local_gate_bias)
    pred <- base + gate * local$adjustment
    c(local, list(base = base, gate = gate, pred = pred))
  },

  forward = function(x_target,
                     coords_target,
                     x_neighbors,
                     coords_neighbors,
                     y_neighbors,
                     use_residual = TRUE,
                     local_scale = 1.0) {
    h_target <- self$context_encoder(x_target, coords_target)
    h_neighbors <- self$encode_neighbors(x_neighbors, coords_neighbors)

    base_target_out <- self$predict_from_context(h_target, x_target, local_scale = local_scale)
    base_target <- flatten_safe(base_target_out$pred)

    dims_neighbors <- as.integer(h_neighbors$shape)
    x_neighbors_flat <- reshape_safe(x_neighbors, c(dims_neighbors[1] * dims_neighbors[2], self$c_tab))
    h_neighbors_flat <- reshape_safe(h_neighbors, c(dims_neighbors[1] * dims_neighbors[2], self$d))
    base_neighbors_out <- self$predict_from_context(h_neighbors_flat, x_neighbors_flat, local_scale = local_scale)
    base_neighbors <- reshape_safe(flatten_safe(base_neighbors_out$pred), c(dims_neighbors[1], dims_neighbors[2]))

    if (isTRUE(use_residual)) {
      residual_neighbors <- y_neighbors - base_neighbors
      k <- self$krig(h_target, coords_target, h_neighbors, coords_neighbors, residual_neighbors)
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
      h = h_target,
      h_neighbors = h_neighbors,
      base_neighbors = base_neighbors,
      delta = delta,
      beta = beta,
      coef_target = base_target_out$coefs,
      coef_neighbors = reshape_safe(base_neighbors_out$coefs, c(dims_neighbors[1], dims_neighbors[2], self$c_tab))
    )
  }
)

predict_geohyper <- function(model,
                             X_query,
                             coords_query,
                             neighbor_idx,
                             X_ref,
                             coords_ref,
                             y_ref,
                             local_scale = 1.0,
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

      pred <- as.numeric(model(xb, cb, xnb, cnb, ynb, local_scale = local_scale)$pred$to(device = "cpu"))
      if (length(pred) != length(idx)) {
        stop(sprintf(
          "GeoHyperKrigingNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geohyper_one_fold <- function(fd,
                                    epochs = 40,
                                    lr = 2e-4,
                                    wd = 1e-3,
                                    batch_size = 96,
                                    patience = 8,
                                    d = 160,
                                    tab_hidden = c(192, 96),
                                    basis_hidden = 128,
                                    fusion_hidden = 160,
                                    base_hidden = c(96),
                                    hyper_hidden = 128,
                                    coef_scale = 1.0,
                                    adjust_scale = 0.25,
                                    local_gate_scale = 1.0,
                                    gate_bias_init = -2.0,
                                    dropout = 0.10,
                                    beta_init = -4,
                                    warmup_epochs = 4,
                                    base_loss_weight = 0.35,
                                    coef_penalty_weight = 0.01,
                                    train_seed = 123,
                                    deterministic_batches = FALSE,
                                    local_freeze_epochs = 0,
                                    local_ramp_epochs = 1,
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

  set_geohyper_seed(train_seed)

  model <- GeoHyperKrigingNet(
    c_tab = ncol(Xtr),
    init_centers = fd$init_centers,
    d = d,
    tab_hidden = tab_hidden,
    basis_hidden = basis_hidden,
    fusion_hidden = fusion_hidden,
    base_hidden = base_hidden,
    hyper_hidden = hyper_hidden,
    coef_scale = coef_scale,
    adjust_scale = adjust_scale,
    local_gate_scale = local_gate_scale,
    gate_bias_init = gate_bias_init,
    dropout = dropout,
    beta_init = beta_init
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$context_encoder$parameters,
      model$base_head$parameters,
      model$linear_head$parameters,
      model$local_gate_head$parameters
    )
    warmup_opt <- optim_adamw(warmup_params, lr = lr, weight_decay = wd)

    for (ep in seq_len(warmup_epochs)) {
      set_geohyper_seed(train_seed + ep - 1)
      model$train()
      batches <- make_geohyper_batches(nrow(Xtr), batch_size = batch_size, deterministic = deterministic_batches)
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

        out <- model(xb, cb, xnb, cnb, ynb, use_residual = FALSE, local_scale = 0)
        coef_penalty <- torch_mean(out$coef_target^2)
        loss <- huber_loss(yb, out$base_pred) + coef_penalty_weight * coef_penalty

        warmup_opt$zero_grad()
        loss$backward()
        nn_utils_clip_grad_norm_(warmup_params, max_norm = 2)
        warmup_opt$step()

        train_loss <- train_loss + loss$item()
      }

      cat(sprintf("[GeoHyper Warmup] Epoch %d/%d | train_loss=%.4f\n",
                  ep, warmup_epochs, train_loss / length(batches)))
    }
  }

  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)
  best_val <- Inf
  best_state <- NULL
  bad <- 0

  for (ep in seq_len(epochs)) {
    set_geohyper_seed(train_seed + warmup_epochs + ep - 1)
    model$train()
    batches <- make_geohyper_batches(nrow(Xtr), batch_size = batch_size, deterministic = deterministic_batches)
    train_loss <- 0
    local_train_scale <- geohyper_local_train_scale(
      epoch = ep,
      epochs = epochs,
      freeze_epochs = local_freeze_epochs,
      ramp_epochs = local_ramp_epochs
    )

    for (batch_id in seq_along(batches)) {
      b <- batches[[batch_id]]
      nb <- fd$neighbor_idx$train[b, , drop = FALSE]

      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr_s[b], device = device)
      xnb <- to_float_tensor(gather_neighbor_array(Xtr, nb), device = device)
      cnb <- to_float_tensor(gather_neighbor_array(Ctr, nb), device = device)
      ynb <- to_float_tensor(gather_neighbor_matrix(ytr_s, nb), device = device)

      out <- model(xb, cb, xnb, cnb, ynb, use_residual = TRUE, local_scale = local_train_scale)
      coef_penalty <- torch_mean(out$coef_target^2) + 0.25 * torch_mean(out$coef_neighbors^2)
      loss <- huber_loss(yb, out$pred) +
        base_loss_weight * huber_loss(yb, out$base_pred) +
        coef_penalty_weight * coef_penalty

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[GeoHyper] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geohyper(
      model = model,
      X_query = Xva,
      coords_query = Cva,
      neighbor_idx = fd$neighbor_idx$val,
      X_ref = Xtr,
      coords_ref = Ctr,
      y_ref = ytr_s,
      local_scale = local_train_scale,
      device = device,
      batch_size = batch_size
    )
    vloss <- mean((yva_s - val_pred_s)^2)
    cat(sprintf("[GeoHyper] Epoch %d complete | local_scale=%.3f | train_loss=%.4f | val_mse=%.4f\n",
                ep, local_train_scale, train_loss / length(batches), vloss))

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

  preds_scaled <- predict_geohyper(
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

geohyperkrigingnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 160,
  tab_hidden = c(192, 96),
  basis_hidden = 128,
  fusion_hidden = 160,
  base_hidden = c(96),
  hyper_hidden = 128,
  coef_scale = 1.0,
  adjust_scale = 0.25,
  local_gate_scale = 1.0,
  gate_bias_init = -2.0,
  dropout = 0.10,
  beta_init = -4,
  warmup_epochs = 4,
  base_loss_weight = 0.35,
  coef_penalty_weight = 0.01,
  train_seed = 123,
  deterministic_batches = FALSE,
  local_freeze_epochs = 0,
  local_ramp_epochs = 1,
  target_transform = "identity",
  device = "cpu"
)

geohyperkrigingnet_quick_params <- modifyList(
  geohyperkrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    tab_hidden = c(160, 80),
    basis_hidden = 96,
    fusion_hidden = 128,
    base_hidden = c(96),
    hyper_hidden = 96,
    warmup_epochs = 3
  )
)

geohyperkrigingnet_tuned_quick_params <- modifyList(
  geohyperkrigingnet_quick_params,
  list(
    adjust_scale = 0.35,
    gate_bias_init = -1.5,
    coef_penalty_weight = 0.0075
  )
)

geohyperkrigingnet_tuned_stronger_quick_params <- modifyList(
  geohyperkrigingnet_quick_params,
  list(
    adjust_scale = 0.45,
    gate_bias_init = -1.25,
    coef_penalty_weight = 0.005
  )
)

geohyperkrigingnet_v3_quick_params <- modifyList(
  geohyperkrigingnet_quick_params,
  list(
    base_hidden = c(192, 96, 48),
    warmup_epochs = 8,
    adjust_scale = 0.12,
    local_gate_scale = 0.35,
    gate_bias_init = -2.75,
    base_loss_weight = 0.45,
    coef_penalty_weight = 0.0125
  )
)

geohyperkrigingnet_v3b_quick_params <- modifyList(
  geohyperkrigingnet_v3_quick_params,
  list(
    adjust_scale = 0.15,
    local_gate_scale = 0.45
  )
)

geohyperkrigingnet_v3_stable_quick_params <- modifyList(
  geohyperkrigingnet_v3_quick_params,
  list(
    lr = 1e-4,
    epochs = 24,
    patience = 7,
    warmup_epochs = 10,
    train_seed = 123
  )
)

geohyperkrigingnet_deterministic_quick_params <- modifyList(
  geohyperkrigingnet_v3_quick_params,
  list(
    lr = 1e-4,
    epochs = 24,
    patience = 7,
    warmup_epochs = 10,
    dropout = 0,
    adjust_scale = 0.10,
    local_gate_scale = 0.25,
    deterministic_batches = TRUE,
    local_freeze_epochs = 12,
    local_ramp_epochs = 8,
    train_seed = 123
  )
)

run_geohyper_on_fixed_benchmark <- function(benchmark,
                                            context = wadoux_context,
                                            model_params = geohyperkrigingnet_params,
                                            K_neighbors = 12,
                                            n_centers = 32) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoHyper Fair] split %s | K_neighbors=%d | centers=%d\n",
                sp$split_id, K_neighbors, n_centers))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geohyper_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K_neighbors = K_neighbors,
      n_centers = n_centers,
      center_seed = benchmark$meta$seed + i
    )

    out <- do.call(train_geohyper_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoHyperKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geohyperkrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                  sample_size = 250,
                                                  sampling = "simple_random",
                                                  n_folds = 5,
                                                  val_dist_km = 350,
                                                  val_frac = 0.2,
                                                  max_splits = 5,
                                                  seed = 123,
                                                  model_params = geohyperkrigingnet_quick_params,
                                                  K_neighbors = 12,
                                                  n_centers = 32,
                                                  cubist_committees = 50,
                                                  cubist_neighbors = 5,
                                                  results_dir = "results/geohyperkrigingnet_vs_cubist",
                                                  save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoHyperKrigingNet vs Cubist BENCHMARK\n")
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

  hyper_res <- run_geohyper_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params,
    K_neighbors = K_neighbors,
    n_centers = n_centers
  )

  final <- bind_rows(cubist_res, hyper_res)
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
    write.csv(hyper_res, file.path(results_dir, "geohyperkrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geohyperkrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geohyperkrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geohyperkrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                          sample_size = 300,
                                                          sampling = "simple_random",
                                                          n_folds = 10,
                                                          val_dist_km = 350,
                                                          val_frac = 0.2,
                                                          max_splits = 10,
                                                          seed = 123,
                                                          model_params = geohyperkrigingnet_params,
                                                          K_neighbors = 12,
                                                          n_centers = 32,
                                                          cubist_committees = 50,
                                                          cubist_neighbors = 5,
                                                          results_dir = "results/geohyperkrigingnet_vs_cubist_confirmation",
                                                          save_outputs = TRUE) {
  run_geohyperkrigingnet_vs_cubist_fair(
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
    n_centers = n_centers,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

make_geohyper_tuning_variants <- function() {
  list(
    GeoHyper_Base = list(params = geohyperkrigingnet_quick_params, K_neighbors = 12),
    GeoHyper_Tuned = list(params = geohyperkrigingnet_tuned_quick_params, K_neighbors = 12),
    GeoHyper_TunedStrong = list(params = geohyperkrigingnet_tuned_stronger_quick_params, K_neighbors = 12),
    GeoHyper_Tuned_K8 = list(params = geohyperkrigingnet_tuned_quick_params, K_neighbors = 8)
  )
}

make_geohyper_v3_variants <- function() {
  list(
    GeoHyper_Base = list(params = geohyperkrigingnet_quick_params, K_neighbors = 12),
    GeoHyper_V3 = list(params = geohyperkrigingnet_v3_quick_params, K_neighbors = 12),
    GeoHyper_V3b = list(params = geohyperkrigingnet_v3b_quick_params, K_neighbors = 12)
  )
}

make_geohyper_stability_variants <- function() {
  list(
    GeoHyper_V3 = list(params = geohyperkrigingnet_v3_quick_params, K_neighbors = 12),
    GeoHyper_V3_Stable = list(params = geohyperkrigingnet_v3_stable_quick_params, K_neighbors = 12)
  )
}

make_geohyper_deterministic_variants <- function() {
  list(
    GeoHyper_V3 = list(params = geohyperkrigingnet_v3_quick_params, K_neighbors = 12),
    GeoHyper_Deterministic = list(params = geohyperkrigingnet_deterministic_quick_params, K_neighbors = 12)
  )
}

run_geohyper_tuning_search <- function(context = wadoux_context,
                                       sample_size = 250,
                                       sampling = "simple_random",
                                       n_folds = 5,
                                       val_dist_km = 350,
                                       val_frac = 0.2,
                                       max_splits = 5,
                                       seed = 123,
                                       n_centers = 32,
                                       cubist_committees = 50,
                                       cubist_neighbors = 5,
                                       variants = make_geohyper_tuning_variants(),
                                       results_dir = "results/geohyper_tuning_search",
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

    res <- run_geohyper_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      model_params = v$params,
      K_neighbors = v$K_neighbors,
      n_centers = n_centers
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
    write.csv(final, file.path(results_dir, "geohyper_tuning_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geohyper_tuning_summary.csv"), row.names = FALSE)
  }

  final
}

run_geohyper_v3_search <- function(context = wadoux_context,
                                   sample_size = 250,
                                   sampling = "simple_random",
                                   n_folds = 5,
                                   val_dist_km = 350,
                                   val_frac = 0.2,
                                   max_splits = 5,
                                   seed = 123,
                                   n_centers = 32,
                                   cubist_committees = 50,
                                   cubist_neighbors = 5,
                                   variants = make_geohyper_v3_variants(),
                                   results_dir = "results/geohyper_v3_search",
                                   save_outputs = TRUE) {
  run_geohyper_tuning_search(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    n_centers = n_centers,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    variants = variants,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

run_geohyper_stability_search <- function(context = wadoux_context,
                                          sample_size = 250,
                                          sampling = "simple_random",
                                          n_folds = 5,
                                          val_dist_km = 350,
                                          val_frac = 0.2,
                                          max_splits = 5,
                                          seed = 123,
                                          n_centers = 32,
                                          cubist_committees = 50,
                                          cubist_neighbors = 5,
                                          train_seeds = c(11, 29, 47),
                                          variants = make_geohyper_stability_variants(),
                                          results_dir = "results/geohyper_stability_search",
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

  variant_results <- list()
  row_id <- 1

  for (vname in names(variants)) {
    v <- variants[[vname]]

    for (train_seed in train_seeds) {
      params_seeded <- modifyList(v$params, list(train_seed = as.integer(train_seed)))

      res <- run_geohyper_on_fixed_benchmark(
        benchmark = benchmark,
        context = context,
        model_params = params_seeded,
        K_neighbors = v$K_neighbors,
        n_centers = n_centers
      ) %>%
        mutate(
          variant = vname,
          train_seed = train_seed
        )

      cb <- cubist_res %>%
        mutate(
          variant = vname,
          train_seed = train_seed
        )

      variant_results[[row_id]] <- bind_rows(cb, res)

      if (save_outputs) {
        write.csv(
          variant_results[[row_id]],
          file.path(results_dir, sprintf("partial_%s_seed%s.csv", vname, train_seed)),
          row.names = FALSE
        )
      }

      row_id <- row_id + 1
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

  summary_by_seed <- final %>%
    group_by(variant, train_seed, model) %>%
    summarise(
      RMSE_mean = mean(RMSE, na.rm = TRUE),
      R2_mean = mean(R2, na.rm = TRUE),
      MAE_mean = mean(MAE, na.rm = TRUE),
      Bias_mean = mean(Bias, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(variant, train_seed, RMSE_mean)

  if (save_outputs) {
    write.csv(final, file.path(results_dir, "geohyper_stability_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geohyper_stability_summary.csv"), row.names = FALSE)
    write.csv(summary_by_seed, file.path(results_dir, "geohyper_stability_summary_by_seed.csv"), row.names = FALSE)
  }

  final
}

run_geohyper_deterministic_search <- function(context = wadoux_context,
                                              sample_size = 250,
                                              sampling = "simple_random",
                                              n_folds = 5,
                                              val_dist_km = 350,
                                              val_frac = 0.2,
                                              max_splits = 5,
                                              seed = 123,
                                              n_centers = 32,
                                              cubist_committees = 50,
                                              cubist_neighbors = 5,
                                              train_seeds = c(11, 29, 47),
                                              variants = make_geohyper_deterministic_variants(),
                                              results_dir = "results/geohyper_deterministic_search",
                                              save_outputs = TRUE) {
  run_geohyper_stability_search(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    n_centers = n_centers,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    train_seeds = train_seeds,
    variants = variants,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}
