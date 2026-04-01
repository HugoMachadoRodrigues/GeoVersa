rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoScoreKrigingNet
# -----------------------------------------------------------------------------
# Point-based score-based spatial model with:
# - strong tabular + learned spatial-basis context encoder
# - light anisotropic residual kriging-like correction
# - scalar denoising score network that refines the final prediction
#
# Plain-language flow:
# 1. Build a stable context vector from covariates and coordinates.
# 2. Predict a base value for the target and for neighbors.
# 3. Apply a light anisotropic residual correction.
# 4. Train a score network to denoise the target under Gaussian perturbations.
# 5. Refine the base prediction via score-based denoising steps.
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

sample_score_centers <- function(coords_train, n_centers = 32, seed = 123) {
  set.seed(seed)
  m_eff <- min(n_centers, nrow(coords_train))
  idx <- sort(sample(seq_len(nrow(coords_train)), size = m_eff))
  coords_train[idx, , drop = FALSE]
}

prepare_geoscore_fold <- function(context,
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

  init_centers <- sample_score_centers(coords_train_s, n_centers = n_centers, seed = center_seed)

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

GeoScoreContextEncoder <- nn_module(
  "GeoScoreContextEncoder",
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

GeoScoreScalarHead <- nn_module(
  "GeoScoreScalarHead",
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

GeoScoreHead <- nn_module(
  "GeoScoreHead",
  initialize = function(d = 160, sigma_hidden = 32, hidden = 128, dropout = 0.10) {
    self$sigma_embed <- nn_sequential(
      nn_linear(1, sigma_hidden),
      nn_gelu(),
      nn_linear(sigma_hidden, sigma_hidden),
      nn_gelu()
    )
    self$net <- nn_sequential(
      nn_linear(d + sigma_hidden + 1, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, hidden),
      nn_gelu(),
      nn_linear(hidden, 1)
    )
  },
  forward = function(h, y_noisy, sigma) {
    sigma_feat <- self$sigma_embed(torch_log(sigma + 1e-6)$unsqueeze(2))
    inp <- torch_cat(list(h, y_noisy$unsqueeze(2), sigma_feat), dim = 2)
    flatten_safe(self$net(inp))
  }
)

GeoScoreKrigingNet <- nn_module(
  "GeoScoreKrigingNet",
  initialize = function(c_tab,
                        init_centers,
                        d = 160,
                        tab_hidden = c(192, 96),
                        basis_hidden = 128,
                        fusion_hidden = 160,
                        score_hidden = 128,
                        dropout = 0.10,
                        beta_init = -4) {
    self$d <- d
    self$context_encoder <- GeoScoreContextEncoder(
      c_tab = c_tab,
      init_centers = init_centers,
      d = d,
      tab_hidden = tab_hidden,
      basis_hidden = basis_hidden,
      fusion_hidden = fusion_hidden,
      dropout = dropout
    )
    self$target_head <- GeoScoreScalarHead(d = d, hidden = 96, dropout = dropout)
    self$neighbor_head <- GeoScoreScalarHead(d = d, hidden = 96, dropout = dropout)
    self$score_head <- GeoScoreHead(d = d, sigma_hidden = 32, hidden = score_hidden, dropout = dropout)
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

  score = function(h, y_noisy, sigma) {
    self$score_head(h, y_noisy, sigma)
  },

  forward = function(x_target, coords_target, x_neighbors, coords_neighbors, y_neighbors, use_residual = TRUE) {
    h_target <- self$context_encoder(x_target, coords_target)
    h_neighbors <- self$encode_neighbors(x_neighbors, coords_neighbors)

    base_target <- flatten_safe(self$target_head(h_target))
    dims_neighbors <- as.integer(h_neighbors$shape)
    base_neighbors <- self$neighbor_head(reshape_safe(h_neighbors, c(dims_neighbors[1] * dims_neighbors[2], self$d)))
    base_neighbors <- reshape_safe(flatten_safe(base_neighbors), c(dims_neighbors[1], dims_neighbors[2]))

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
      beta = beta
    )
  }
)

sample_noise_sigmas <- function(B, device = "cpu", levels = c(0.05, 0.10, 0.20, 0.40)) {
  sig <- sample(levels, size = B, replace = TRUE)
  to_float_tensor(sig, device = device)
}

score_refine_tensor <- function(model,
                                h,
                                y_init,
                                method = "iterative",
                                sigma_schedule = c(0.20, 0.10, 0.05),
                                n_steps_each = 2,
                                step_factor = 0.8,
                                single_step_sigma = 0.10,
                                single_step_factor = 1.0) {
  y <- y_init$clone()
  if (identical(method, "single_step")) {
    sigma <- torch_full_like(y, single_step_sigma)
    sc <- model$score(h, y, sigma)
    y <- y + single_step_factor * (sigma^2) * sc
    return(y)
  }
  for (sigma_val in sigma_schedule) {
    sigma <- torch_full_like(y, sigma_val)
    for (t in seq_len(n_steps_each)) {
      sc <- model$score(h, y, sigma)
      y <- y + step_factor * (sigma^2) * sc
    }
  }
  y
}

predict_geoscore <- function(model,
                             X_query,
                             coords_query,
                             neighbor_idx,
                             X_ref,
                             coords_ref,
                             y_ref,
                             refine_method = "iterative",
                             sigma_schedule = c(0.20, 0.10, 0.05),
                             n_steps_each = 2,
                             step_factor = 0.8,
                             single_step_sigma = 0.10,
                             single_step_factor = 1.0,
                             refine_mix = 1.0,
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

      fwd <- model(xb, cb, xnb, cnb, ynb, use_residual = TRUE)
      pred_refined <- score_refine_tensor(
        model = model,
        h = fwd$h,
        y_init = fwd$pred,
        method = refine_method,
        sigma_schedule = sigma_schedule,
        n_steps_each = n_steps_each,
        step_factor = step_factor,
        single_step_sigma = single_step_sigma,
        single_step_factor = single_step_factor
      )
      pred_refined <- fwd$pred + refine_mix * (pred_refined - fwd$pred)

      pred <- as.numeric(pred_refined$to(device = "cpu"))
      if (length(pred) != length(idx)) {
        stop(sprintf(
          "GeoScoreKrigingNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geoscore_one_fold <- function(fd,
                                    epochs = 40,
                                    lr = 2e-4,
                                    wd = 1e-3,
                                    batch_size = 96,
                                    patience = 8,
                                    d = 160,
                                    tab_hidden = c(192, 96),
                                    basis_hidden = 128,
                                    fusion_hidden = 160,
                                    score_hidden = 128,
                                    dropout = 0.10,
                                    beta_init = -4,
                                    warmup_epochs = 4,
                                    base_loss_weight = 0.35,
                                    score_loss_weight = 0.25,
                                    score_sigma_levels = c(0.05, 0.10, 0.20, 0.40),
                                    refine_method = "iterative",
                                    refine_sigma_schedule = c(0.20, 0.10, 0.05),
                                    refine_steps_each = 2,
                                    refine_step_factor = 0.8,
                                    single_step_sigma = 0.10,
                                    single_step_factor = 1.0,
                                    refine_mix = 1.0,
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

  model <- GeoScoreKrigingNet(
    c_tab = ncol(Xtr),
    init_centers = fd$init_centers,
    d = d,
    tab_hidden = tab_hidden,
    basis_hidden = basis_hidden,
    fusion_hidden = fusion_hidden,
    score_hidden = score_hidden,
    dropout = dropout,
    beta_init = beta_init
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$context_encoder$parameters,
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

      cat(sprintf("[GeoScore Warmup] Epoch %d/%d | train_loss=%.4f\n",
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
      B <- length(b)

      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr_s[b], device = device)
      xnb <- to_float_tensor(gather_neighbor_array(Xtr, nb), device = device)
      cnb <- to_float_tensor(gather_neighbor_array(Ctr, nb), device = device)
      ynb <- to_float_tensor(gather_neighbor_matrix(ytr_s, nb), device = device)

      out <- model(xb, cb, xnb, cnb, ynb, use_residual = TRUE)

      sigma <- sample_noise_sigmas(B, device = device, levels = score_sigma_levels)
      eps <- torch_randn_like(yb)
      y_noisy <- yb + sigma * eps
      score_target <- -(y_noisy - yb) / (sigma^2 + 1e-6)
      score_pred <- model$score(out$h, y_noisy, sigma)

      score_loss <- mean((score_pred - score_target)^2)
      loss <- huber_loss(yb, out$pred) +
        base_loss_weight * huber_loss(yb, out$base_pred) +
        score_loss_weight * score_loss

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[GeoScore] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geoscore(
      model = model,
      X_query = Xva,
      coords_query = Cva,
      neighbor_idx = fd$neighbor_idx$val,
      X_ref = Xtr,
      coords_ref = Ctr,
      y_ref = ytr_s,
      refine_method = refine_method,
      sigma_schedule = refine_sigma_schedule,
      n_steps_each = refine_steps_each,
      step_factor = refine_step_factor,
      single_step_sigma = single_step_sigma,
      single_step_factor = single_step_factor,
      refine_mix = refine_mix,
      device = device,
      batch_size = batch_size
    )
    vloss <- mean((yva_s - val_pred_s)^2)
    cat(sprintf("[GeoScore] Epoch %d complete | train_loss=%.4f | val_mse=%.4f\n",
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

  preds_scaled <- predict_geoscore(
    model = model,
    X_query = Xte,
    coords_query = Cte,
    neighbor_idx = fd$neighbor_idx$test,
    X_ref = Xtr,
    coords_ref = Ctr,
    y_ref = ytr_s,
    refine_method = refine_method,
    sigma_schedule = refine_sigma_schedule,
    n_steps_each = refine_steps_each,
    step_factor = refine_step_factor,
    single_step_sigma = single_step_sigma,
    single_step_factor = single_step_factor,
    refine_mix = refine_mix,
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

geoscorekrigingnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 160,
  tab_hidden = c(192, 96),
  basis_hidden = 128,
  fusion_hidden = 160,
  score_hidden = 128,
  dropout = 0.10,
  beta_init = -4,
  warmup_epochs = 4,
  base_loss_weight = 0.35,
  score_loss_weight = 0.25,
  score_sigma_levels = c(0.05, 0.10, 0.20, 0.40),
  refine_method = "iterative",
  refine_sigma_schedule = c(0.20, 0.10, 0.05),
  refine_steps_each = 2,
  refine_step_factor = 0.8,
  single_step_sigma = 0.10,
  single_step_factor = 1.0,
  refine_mix = 1.0,
  target_transform = "identity",
  device = "cpu"
)

geoscorekrigingnet_quick_params <- modifyList(
  geoscorekrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    tab_hidden = c(160, 80),
    basis_hidden = 96,
    fusion_hidden = 128,
    score_hidden = 96,
    warmup_epochs = 3,
    refine_steps_each = 2
  )
)

geoscorekrigingnet_singlestep_quick_params <- modifyList(
  geoscorekrigingnet_quick_params,
  list(
    refine_method = "single_step",
    single_step_sigma = 0.10,
    single_step_factor = 1.0
  )
)

geoscorekrigingnet_conservative_quick_params <- modifyList(
  geoscorekrigingnet_quick_params,
  list(
    score_loss_weight = 0.15,
    refine_sigma_schedule = c(0.15, 0.08, 0.04),
    refine_steps_each = 2,
    refine_step_factor = 0.6,
    refine_mix = 0.6
  )
)

geoscorekrigingnet_conservative_lowloss_quick_params <- modifyList(
  geoscorekrigingnet_quick_params,
  list(
    score_loss_weight = 0.10,
    refine_sigma_schedule = c(0.12, 0.06, 0.03),
    refine_steps_each = 1,
    refine_step_factor = 0.5,
    refine_mix = 0.5
  )
)

run_geoscore_on_fixed_benchmark <- function(benchmark,
                                            context = wadoux_context,
                                            model_params = geoscorekrigingnet_params,
                                            K_neighbors = 12,
                                            n_centers = 32) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoScore Fair] split %s | K_neighbors=%d | centers=%d\n",
                sp$split_id, K_neighbors, n_centers))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geoscore_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K_neighbors = K_neighbors,
      n_centers = n_centers,
      center_seed = benchmark$meta$seed + i
    )

    out <- do.call(train_geoscore_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoScoreKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geoscorekrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                  sample_size = 250,
                                                  sampling = "simple_random",
                                                  n_folds = 5,
                                                  val_dist_km = 350,
                                                  val_frac = 0.2,
                                                  max_splits = 5,
                                                  seed = 123,
                                                  model_params = geoscorekrigingnet_quick_params,
                                                  K_neighbors = 12,
                                                  n_centers = 32,
                                                  cubist_committees = 50,
                                                  cubist_neighbors = 5,
                                                  results_dir = "results/geoscorekrigingnet_vs_cubist",
                                                  save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoScoreKrigingNet vs Cubist BENCHMARK\n")
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

  score_res <- run_geoscore_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params,
    K_neighbors = K_neighbors,
    n_centers = n_centers
  )

  final <- bind_rows(cubist_res, score_res)
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
    write.csv(score_res, file.path(results_dir, "geoscorekrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geoscorekrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geoscorekrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geoscorekrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                          sample_size = 300,
                                                          sampling = "simple_random",
                                                          n_folds = 10,
                                                          val_dist_km = 350,
                                                          val_frac = 0.2,
                                                          max_splits = 10,
                                                          seed = 123,
                                                          model_params = geoscorekrigingnet_params,
                                                          K_neighbors = 12,
                                                          n_centers = 32,
                                                          cubist_committees = 50,
                                                          cubist_neighbors = 5,
                                                          results_dir = "results/geoscorekrigingnet_vs_cubist_confirmation",
                                                          save_outputs = TRUE) {
  run_geoscorekrigingnet_vs_cubist_fair(
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

make_geoscore_refine_variants <- function() {
  list(
    GeoScoreKrigingNet_Iterative = list(params = geoscorekrigingnet_quick_params),
    GeoScoreKrigingNet_SingleStep = list(params = geoscorekrigingnet_singlestep_quick_params),
    GeoScoreKrigingNet_Conservative = list(params = geoscorekrigingnet_conservative_quick_params),
    GeoScoreKrigingNet_ConservativeLowLoss = list(params = geoscorekrigingnet_conservative_lowloss_quick_params)
  )
}

run_geoscore_refine_search <- function(context = wadoux_context,
                                       sample_size = 250,
                                       sampling = "simple_random",
                                       n_folds = 5,
                                       val_dist_km = 350,
                                       val_frac = 0.2,
                                       max_splits = 5,
                                       seed = 123,
                                       K_neighbors = 12,
                                       n_centers = 32,
                                       cubist_committees = 50,
                                       cubist_neighbors = 5,
                                       variants = make_geoscore_refine_variants(),
                                       results_dir = "results/geoscore_refine_search",
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

    res <- run_geoscore_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      model_params = v$params,
      K_neighbors = K_neighbors,
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
    write.csv(final, file.path(results_dir, "geoscore_refine_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geoscore_refine_summary.csv"), row.names = FALSE)
  }

  final
}
