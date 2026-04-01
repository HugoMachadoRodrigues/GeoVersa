rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoKANKrigingNet
# -----------------------------------------------------------------------------
# Point-based geostatistical model with:
# - KAN-like additive tabular encoder built from per-feature univariate functions
# - learned spatial basis encoder from coordinates
# - fused base prediction
# - light anisotropic residual kriging-like correction
#
# Plain-language flow:
# 1. Each covariate is passed through its own learned 1D function.
# 2. These feature-wise functions are added to build a tabular representation.
# 3. A spatial basis encoder summarizes smooth coordinate effects.
# 4. Both parts are fused into a base prediction.
# 5. Neighbor residuals provide a light spatial correction.
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

sample_kan_centers <- function(coords_train, n_centers = 32, seed = 123) {
  set.seed(seed)
  m_eff <- min(n_centers, nrow(coords_train))
  idx <- sort(sample(seq_len(nrow(coords_train)), size = m_eff))
  coords_train[idx, , drop = FALSE]
}

prepare_geokan_fold <- function(context,
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

  init_centers <- sample_kan_centers(coords_train_s, n_centers = n_centers, seed = center_seed)

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

KANFeatureBranch <- nn_module(
  "KANFeatureBranch",
  initialize = function(n_basis = 8,
                        hidden = 32,
                        out_dim = 160,
                        dropout = 0.10,
                        init_width = 0.75) {
    self$n_basis <- as.integer(n_basis)
    centers <- seq(-2, 2, length.out = self$n_basis)
    self$centers <- nn_parameter(torch_tensor(centers, dtype = torch_float()))
    self$log_widths <- nn_parameter(torch_tensor(rep(log(init_width), self$n_basis), dtype = torch_float()))
    self$net <- nn_sequential(
      nn_linear(self$n_basis + 2, hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden, out_dim)
    )
  },
  forward = function(x_j) {
    x_col <- flatten_safe(x_j)$unsqueeze(2)
    centers <- self$centers$view(c(1, self$n_basis))
    widths <- (nnf_softplus(self$log_widths) + 1e-6)$view(c(1, self$n_basis))
    basis <- torch_exp(-((x_col - centers)^2) / (2 * widths^2))
    feat <- torch_cat(list(x_col, x_col^2, basis), dim = 2)
    self$net(feat)
  }
)

GeoKANAdditiveEncoder <- nn_module(
  "GeoKANAdditiveEncoder",
  initialize = function(c_tab,
                        d = 160,
                        n_basis = 8,
                        branch_hidden = 32,
                        post_hidden = 160,
                        feature_scale = 1.0,
                        post_scale = 0.5,
                        dropout = 0.10) {
    self$n_features <- as.integer(c_tab)
    self$d <- d
    self$feature_scale <- feature_scale
    self$post_scale <- post_scale
    branches <- lapply(seq_len(self$n_features), function(i) {
      KANFeatureBranch(
        n_basis = n_basis,
        hidden = branch_hidden,
        out_dim = d,
        dropout = dropout
      )
    })
    self$branches <- nn_module_list(branches)
    self$post <- nn_sequential(
      nn_linear(d, post_hidden),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(post_hidden, d)
    )
  },
  forward = function(x) {
    acc <- NULL
    for (j in seq_len(self$n_features)) {
      z_j <- self$branches[[j]](x[, j])
      if (is.null(acc)) {
        acc <- z_j
      } else {
        acc <- acc + z_j
      }
    }
    acc <- self$feature_scale * acc / sqrt(self$n_features)
    acc + self$post_scale * self$post(acc)
  }
)

GeoKANScalarHead <- nn_module(
  "GeoKANScalarHead",
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

GeoKANKrigingNet <- nn_module(
  "GeoKANKrigingNet",
  initialize = function(c_tab,
                        init_centers,
                        d = 160,
                        n_basis = 8,
                        kan_hidden = 32,
                        kan_post_hidden = 160,
                        kan_feature_scale = 1.0,
                        kan_post_scale = 0.5,
                        basis_hidden = 128,
                        fusion_hidden = 160,
                        tab_scale = 1.0,
                        basis_scale = 1.0,
                        fuse_scale = 1.0,
                        dropout = 0.10,
                        beta_init = -4) {
    self$d <- d
    self$tab_scale <- tab_scale
    self$basis_scale <- basis_scale
    self$fuse_scale <- fuse_scale
    self$tab_encoder <- GeoKANAdditiveEncoder(
      c_tab = c_tab,
      d = d,
      n_basis = n_basis,
      branch_hidden = kan_hidden,
      post_hidden = kan_post_hidden,
      feature_scale = kan_feature_scale,
      post_scale = kan_post_scale,
      dropout = dropout
    )
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
    self$target_head <- GeoKANScalarHead(d = d, hidden = 96, dropout = dropout)
    self$neighbor_head <- GeoKANScalarHead(d = d, hidden = 96, dropout = dropout)
    self$krig <- AnisotropicResidualKrigingLayer(d = d, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_points = function(x, coords) {
    z_tab <- self$tab_encoder(x)
    z_basis <- self$basis_encoder(coords)
    self$basis_scale * z_basis +
      self$tab_scale * z_tab +
      self$fuse_scale * self$fuse_residual(torch_cat(list(z_tab, z_basis), dim = 2))
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

predict_geokan <- function(model,
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
          "GeoKANKrigingNet prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geokan_one_fold <- function(fd,
                                  epochs = 40,
                                  lr = 2e-4,
                                  wd = 1e-3,
                                  batch_size = 96,
                                  patience = 8,
                                  d = 160,
                                  n_basis = 8,
                                  kan_hidden = 32,
                                  kan_post_hidden = 160,
                                  kan_feature_scale = 1.0,
                                  kan_post_scale = 0.5,
                                  basis_hidden = 128,
                                  fusion_hidden = 160,
                                  tab_scale = 1.0,
                                  basis_scale = 1.0,
                                  fuse_scale = 1.0,
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

  model <- GeoKANKrigingNet(
    c_tab = ncol(Xtr),
    init_centers = fd$init_centers,
    d = d,
    n_basis = n_basis,
    kan_hidden = kan_hidden,
    kan_post_hidden = kan_post_hidden,
    kan_feature_scale = kan_feature_scale,
    kan_post_scale = kan_post_scale,
    basis_hidden = basis_hidden,
    fusion_hidden = fusion_hidden,
    tab_scale = tab_scale,
    basis_scale = basis_scale,
    fuse_scale = fuse_scale,
    dropout = dropout,
    beta_init = beta_init
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$tab_encoder$parameters,
      model$basis_encoder$parameters,
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

      cat(sprintf("[GeoKAN Warmup] Epoch %d/%d | train_loss=%.4f\n",
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
        cat(sprintf("[GeoKAN] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geokan(
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
    cat(sprintf("[GeoKAN] Epoch %d complete | train_loss=%.4f | val_mse=%.4f\n",
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

  preds_scaled <- predict_geokan(
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

geokankrigingnet_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 160,
  n_basis = 8,
  kan_hidden = 32,
  kan_post_hidden = 160,
  kan_feature_scale = 1.0,
  kan_post_scale = 0.5,
  basis_hidden = 128,
  fusion_hidden = 160,
  tab_scale = 1.0,
  basis_scale = 1.0,
  fuse_scale = 1.0,
  dropout = 0.10,
  beta_init = -4,
  warmup_epochs = 4,
  base_loss_weight = 0.35,
  target_transform = "identity",
  device = "cpu"
)

geokankrigingnet_quick_params <- modifyList(
  geokankrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    n_basis = 6,
    kan_hidden = 24,
    kan_post_hidden = 128,
    kan_feature_scale = 1.0,
    kan_post_scale = 0.5,
    basis_hidden = 96,
    fusion_hidden = 128,
    tab_scale = 1.0,
    basis_scale = 1.0,
    fuse_scale = 1.0,
    warmup_epochs = 3
  )
)

geokankrigingnet_conservative_quick_params <- modifyList(
  geokankrigingnet_quick_params,
  list(
    lr = 1e-4,
    n_basis = 4,
    kan_hidden = 16,
    kan_post_hidden = 96,
    kan_feature_scale = 0.50,
    kan_post_scale = 0.15,
    tab_scale = 0.35,
    basis_scale = 1.0,
    fuse_scale = 0.25,
    dropout = 0.05,
    warmup_epochs = 6,
    base_loss_weight = 0.45
  )
)

run_geokan_on_fixed_benchmark <- function(benchmark,
                                          context = wadoux_context,
                                          model_params = geokankrigingnet_params,
                                          K_neighbors = 12,
                                          n_centers = 32) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoKAN Fair] split %s | K_neighbors=%d | centers=%d\n",
                sp$split_id, K_neighbors, n_centers))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geokan_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K_neighbors = K_neighbors,
      n_centers = n_centers,
      center_seed = benchmark$meta$seed + i
    )

    out <- do.call(train_geokan_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoKANKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geokankrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                sample_size = 250,
                                                sampling = "simple_random",
                                                n_folds = 5,
                                                val_dist_km = 350,
                                                val_frac = 0.2,
                                                max_splits = 5,
                                                seed = 123,
                                                model_params = geokankrigingnet_quick_params,
                                                K_neighbors = 12,
                                                n_centers = 32,
                                                cubist_committees = 50,
                                                cubist_neighbors = 5,
                                                results_dir = "results/geokankrigingnet_vs_cubist",
                                                save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoKANKrigingNet vs Cubist BENCHMARK\n")
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

  geo_res <- run_geokan_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params,
    K_neighbors = K_neighbors,
    n_centers = n_centers
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
    write.csv(geo_res, file.path(results_dir, "geokankrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geokankrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geokankrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geokankrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                        sample_size = 300,
                                                        sampling = "simple_random",
                                                        n_folds = 10,
                                                        val_dist_km = 350,
                                                        val_frac = 0.2,
                                                        max_splits = 10,
                                                        seed = 123,
                                                        model_params = geokankrigingnet_params,
                                                        K_neighbors = 12,
                                                        n_centers = 32,
                                                        cubist_committees = 50,
                                                        cubist_neighbors = 5,
                                                        results_dir = "results/geokankrigingnet_vs_cubist_confirmation",
                                                        save_outputs = TRUE) {
  run_geokankrigingnet_vs_cubist_fair(
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

make_geokan_v2_variants <- function() {
  list(
    GeoKAN_Base = list(params = geokankrigingnet_quick_params, K_neighbors = 12),
    GeoKAN_Conservative = list(params = geokankrigingnet_conservative_quick_params, K_neighbors = 12)
  )
}

run_geokan_v2_search <- function(context = wadoux_context,
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
                                 variants = make_geokan_v2_variants(),
                                 results_dir = "results/geokankrigingnet_v2_search",
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

    res <- run_geokan_on_fixed_benchmark(
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
    write.csv(final, file.path(results_dir, "geokan_v2_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geokan_v2_summary.csv"), row.names = FALSE)
  }

  final
}
