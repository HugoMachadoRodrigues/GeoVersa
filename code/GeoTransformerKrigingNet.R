rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoVersa (GeoTransformerKrigingNet backbone)
# -----------------------------------------------------------------------------
# Point-based spatial prediction with explicit geo-statistical structure.
#
# Core idea:
# - each target point is predicted together with its K nearest sampled neighbors
# - the target and its neighbors become a short token sequence
# - a transformer encoder contextualizes these tokens using:
#     * tabular covariates at each point
#     * point coordinates
#     * relative spatial geometry in the attention scores
# - a kriging-like residual head uses observed neighbor residuals to refine the
#   target prediction
#
# Plain-language architecture:
# 1. Build a local neighborhood around the target point.
# 2. Turn target + neighbors into tokens.
# 3. Use relative spatial attention so the target can "look at" nearby points
#    with weights informed by distance and direction.
# 4. Predict a base value for the target and for each neighbor.
# 5. Compute neighbor residuals = observed_neighbor - base_neighbor.
# 6. Apply an anisotropic kriging-like correction from these residuals.
# 7. Final prediction = base prediction + learned residual correction.
#
# Why this differs from what already exists:
# - not a plain tabular transformer
# - not a raster ViT / image transformer
# - not ordinary kriging with a fixed variogram
# - not the PointPatch/CNN line
# - it is a point-neighborhood transformer with explicit relative spatial
#   attention and a separate kriging-like residual correction stage
#
# Pseudocode:
#   neighbors_i = KNN(s_i)
#   tokens_i = [target_i, neighbor_1, ..., neighbor_K]
#   H_i = GeoTransformer(tokens_i, relative_geometry(tokens_i))
#   mu_i = head(H_i[target])
#   mu_neighbors = head(H_i[neighbors])
#   r_neighbors = y_neighbors - mu_neighbors
#   delta_i = kriging_head(H_i[target], H_i[neighbors], coords, r_neighbors)
#   yhat_i = mu_i + beta * delta_i
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

prepare_geotransformer_fold <- function(context,
                                        calibration_df,
                                        train_idx,
                                        val_idx,
                                        test_idx,
                                        use_robust_scaling = TRUE,
                                        K = 24) {
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

GeoRelativeBias <- nn_module(
  "GeoRelativeBias",
  initialize = function(n_heads = 4, hidden_dim = 32) {
    self$n_heads <- n_heads
    self$mlp <- nn_sequential(
      nn_linear(6, hidden_dim),
      nn_gelu(),
      nn_linear(hidden_dim, n_heads)
    )
  },
  forward = function(pair_feat) {
    dims <- as.integer(pair_feat$shape)
    B <- dims[1]
    T <- dims[2]
    flat <- reshape_safe(pair_feat, c(-1, dims[4]))
    out <- self$mlp(flat)
    out <- reshape_safe(out, c(B, T, T, self$n_heads))
    out$permute(c(1, 4, 2, 3))
  }
)

GeoMultiHeadAttention <- nn_module(
  "GeoMultiHeadAttention",
  initialize = function(d_model = 256, n_heads = 4, dropout = 0.10) {
    if (d_model %% n_heads != 0) {
      stop("d_model must be divisible by n_heads.")
    }
    self$d_model <- d_model
    self$n_heads <- n_heads
    self$head_dim <- d_model %/% n_heads
    self$scale <- 1 / sqrt(self$head_dim)

    self$q_proj <- nn_linear(d_model, d_model)
    self$k_proj <- nn_linear(d_model, d_model)
    self$v_proj <- nn_linear(d_model, d_model)
    self$out_proj <- nn_linear(d_model, d_model)
    self$geo_bias <- GeoRelativeBias(n_heads = n_heads, hidden_dim = 32)
    self$drop <- nn_dropout(dropout)
  },
  forward = function(x, pair_feat) {
    dims <- as.integer(x$shape)
    B <- dims[1]
    T <- dims[2]
    H <- self$n_heads
    Dh <- self$head_dim

    q <- reshape_safe(self$q_proj(x), c(B, T, H, Dh))$permute(c(1, 3, 2, 4))
    k <- reshape_safe(self$k_proj(x), c(B, T, H, Dh))$permute(c(1, 3, 2, 4))
    v <- reshape_safe(self$v_proj(x), c(B, T, H, Dh))$permute(c(1, 3, 2, 4))

    scores <- torch_matmul(q, k$transpose(3, 4)) * self$scale
    scores <- scores + self$geo_bias(pair_feat)

    attn <- nnf_softmax(scores, dim = 4)
    attn <- self$drop(attn)

    out <- torch_matmul(attn, v)$permute(c(1, 3, 2, 4))
    out <- reshape_safe(out, c(B, T, self$d_model))
    self$out_proj(out)
  }
)

GeoTransformerBlock <- nn_module(
  "GeoTransformerBlock",
  initialize = function(d_model = 256, n_heads = 4, ff_mult = 2, dropout = 0.10) {
    self$norm1 <- nn_layer_norm(normalized_shape = d_model)
    self$norm2 <- nn_layer_norm(normalized_shape = d_model)
    self$attn <- GeoMultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
    self$ff <- nn_sequential(
      nn_linear(d_model, ff_mult * d_model),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(ff_mult * d_model, d_model)
    )
    self$drop <- nn_dropout(dropout)
  },
  forward = function(x, pair_feat) {
    x <- x + self$drop(self$attn(self$norm1(x), pair_feat))
    x <- x + self$drop(self$ff(self$norm2(x)))
    x
  }
)

GeoTransformerKrigingNet <- nn_module(
  "GeoTransformerKrigingNet",
  initialize = function(c_tab,
                        d = 256,
                        target_hidden = c(256, 128),
                        neighbor_hidden = c(192),
                        token_dropout = 0.10,
                        n_heads = 4,
                        n_layers = 1,
                        ff_mult = 2,
                        beta_init = -4) {
    self$d <- d
    self$target_encoder <- make_mlp(
      c_tab + 2,
      hidden = target_hidden,
      out_dim = d,
      dropout = token_dropout
    )
    self$neighbor_encoder <- make_mlp(
      c_tab + 2,
      hidden = neighbor_hidden,
      out_dim = d,
      dropout = token_dropout
    )
    self$target_role <- nn_parameter(torch_zeros(c(1, 1, d)))
    self$neighbor_role <- nn_parameter(torch_zeros(c(1, 1, d)))
    self$geo_token_encoder <- nn_sequential(
      nn_linear(6, 64),
      nn_gelu(),
      nn_linear(64, d)
    )
    blocks <- vector("list", n_layers)
    for (i in seq_len(n_layers)) {
      blocks[[i]] <- GeoTransformerBlock(
        d_model = d,
        n_heads = n_heads,
        ff_mult = ff_mult,
        dropout = token_dropout
      )
    }
    self$blocks <- blocks
    self$base_head <- ScalarHead(d = d)
    self$krig <- AnisotropicResidualKrigingLayer(d = d, proj_d = 64, init_ell_major = 1, init_ell_minor = 0.5)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_target = function(x_target, coords_target) {
    inp <- torch_cat(list(x_target, coords_target), dim = 2)
    self$target_encoder(inp)
  },

  encode_neighbors = function(x_neighbors, coords_neighbors) {
    dims <- as.integer(x_neighbors$shape)
    B <- dims[1]
    K <- dims[2]
    inp <- torch_cat(list(x_neighbors, coords_neighbors), dim = 3)
    flat <- reshape_safe(inp, c(B * K, dims[3] + 2))
    z <- self$neighbor_encoder(flat)
    reshape_safe(z, c(B, K, self$d))
  },

  build_pair_features = function(coords_seq) {
    x <- coords_seq[, , 1]
    y <- coords_seq[, , 2]
    dx <- x$unsqueeze(3) - x$unsqueeze(2)
    dy <- y$unsqueeze(3) - y$unsqueeze(2)
    dist <- torch_sqrt(dx^2 + dy^2 + 1e-8)
    log_dist <- torch_log(dist + 1)
    ux <- dx / (dist + 1e-6)
    uy <- dy / (dist + 1e-6)
    torch_stack(list(dx, dy, dist, log_dist, ux, uy), dim = 4)
  },

  contextualize = function(x_target, coords_target, x_neighbors, coords_neighbors) {
    z_target_raw <- self$encode_target(x_target, coords_target)
    z_neighbors_raw <- self$encode_neighbors(x_neighbors, coords_neighbors)
    coords_seq <- torch_cat(list(coords_target$unsqueeze(2), coords_neighbors), dim = 2)
    pair_feat <- self$build_pair_features(coords_seq)
    geo_emb <- self$geo_token_encoder(reshape_safe(pair_feat[, 1, 2:dim(pair_feat)[3], ], c(-1, 6)))
    geo_emb <- reshape_safe(geo_emb, c(as.integer(coords_neighbors$shape)[1], as.integer(coords_neighbors$shape)[2], self$d))
    z_target <- z_target_raw$unsqueeze(2) + self$target_role
    z_neighbors <- z_neighbors_raw + self$neighbor_role + geo_emb
    z_seq <- torch_cat(list(z_target, z_neighbors), dim = 2)

    for (blk in self$blocks) {
      z_seq <- blk(z_seq, pair_feat)
    }

    dims <- as.integer(z_seq$shape)

    list(
      z_target_raw = z_target_raw,
      z_neighbors_raw = z_neighbors_raw,
      z_target_corr = z_seq[, 1, ],
      z_neighbors_corr = z_seq[, 2:dims[2], ]
    )
  },

  forward = function(x_target,
                     coords_target,
                     x_neighbors,
                     coords_neighbors,
                     y_neighbors,
                     use_residual = TRUE,
                     residual_scale = 1) {
    ctx <- self$contextualize(x_target, coords_target, x_neighbors, coords_neighbors)

    dims <- as.integer(ctx$z_neighbors_raw$shape)
    B <- dims[1]
    K <- dims[2]

    base_target <- flatten_safe(self$base_head(ctx$z_target_raw$unsqueeze(2)))
    base_neighbors <- self$base_head(
      reshape_safe(ctx$z_neighbors_raw, c(B * K, self$d))$unsqueeze(2)
    )
    base_neighbors <- reshape_safe(flatten_safe(base_neighbors), c(B, K))
    residual_neighbors <- y_neighbors - base_neighbors

    beta <- torch_sigmoid(self$logit_beta)
    if (isTRUE(use_residual)) {
      k <- self$krig(
        z_i = ctx$z_target_corr,
        coords_i = coords_target,
        z_n = ctx$z_neighbors_corr,
        coords_n = coords_neighbors,
        r_n = residual_neighbors
      )
      delta <- k$delta
      pred <- base_target + residual_scale * beta * delta
    } else {
      delta <- torch_zeros_like(base_target)
      pred <- base_target
    }

    list(
      pred = pred,
      base_pred = base_target,
      delta = delta,
      beta = beta,
      base_neighbors = base_neighbors
    )
  }
)

predict_geotransformerkrigingnet <- function(model,
                                             X_query,
                                             coords_query,
                                             neighbor_idx,
                                             X_ref,
                                             coords_ref,
                                             y_ref,
                                             device = "cpu",
                                             batch_size = 256,
                                             prediction = c("full", "base")) {
  prediction <- match.arg(prediction)
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

      out_obj <- model(
        xb, cb, xnb, cnb, ynb,
        use_residual = identical(prediction, "full")
      )
      pred <- as.numeric(out_obj$pred$to(device = "cpu"))
      if (length(pred) != length(idx)) {
        stop(sprintf(
          "GeoVersa prediction length mismatch: expected %d values, got %d.",
          length(idx), length(pred)
        ))
      }
      out[idx] <- pred
    }
  })

  out
}

train_geotransformerkrigingnet_one_fold <- function(fd,
                                                    epochs = 60,
                                                    lr = 2e-4,
                                                    wd = 1e-3,
                                                    batch_size = 96,
                                                    patience = 10,
                                                    d = 256,
                                                    target_hidden = c(256, 128),
                                                    neighbor_hidden = c(192),
                                                    token_dropout = 0.10,
                                                    n_heads = 4,
                                                    n_layers = 1,
                                                    ff_mult = 2,
                                                    beta_init = -4,
                                                    warmup_epochs = 8,
                                                    warmup_lr_mult = 1,
                                                    base_loss_weight = 0.25,
                                                    pred_huber_delta = 1.0,
                                                    base_huber_delta = 1.0,
                                                    residual_start_scale = 0.10,
                                                    residual_ramp_epochs = 10,
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

  model <- GeoTransformerKrigingNet(
    c_tab = ncol(Xtr),
    d = d,
    target_hidden = target_hidden,
    neighbor_hidden = neighbor_hidden,
    token_dropout = token_dropout,
    n_heads = n_heads,
    n_layers = n_layers,
    ff_mult = ff_mult,
    beta_init = beta_init
  )
  model$to(device = device)

  warmup_params <- c(model$target_encoder$parameters, model$base_head$parameters)

  if (warmup_epochs > 0) {
    warmup_opt <- optim_adamw(warmup_params, lr = lr * warmup_lr_mult, weight_decay = wd)

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
        loss <- huber_loss(yb, out$base_pred, delta = base_huber_delta)

        warmup_opt$zero_grad()
        loss$backward()
        nn_utils_clip_grad_norm_(warmup_params, max_norm = 2)
        warmup_opt$step()

        train_loss <- train_loss + loss$item()
      }

      val_base_s <- predict_geotransformerkrigingnet(
        model = model,
        X_query = Xva,
        coords_query = Cva,
        neighbor_idx = fd$neighbor_idx$val,
        X_ref = Xtr,
        coords_ref = Ctr,
        y_ref = ytr_s,
        device = device,
        batch_size = batch_size,
        prediction = "base"
      )
      warmup_vloss <- mean((yva_s - val_base_s)^2)
      cat(sprintf("[GeoVersa Warmup] Epoch %d/%d | train_loss=%.4f | val_base_mse=%.4f\n",
                  ep, warmup_epochs, train_loss / length(batches), warmup_vloss))
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
    residual_scale <- min(1, residual_start_scale + (1 - residual_start_scale) * ((ep - 1) / max(1, residual_ramp_epochs - 1)))

    for (batch_id in seq_along(batches)) {
      b <- batches[[batch_id]]
      nb <- fd$neighbor_idx$train[b, , drop = FALSE]

      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr_s[b], device = device)

      xnb <- to_float_tensor(gather_neighbor_array(Xtr, nb), device = device)
      cnb <- to_float_tensor(gather_neighbor_array(Ctr, nb), device = device)
      ynb <- to_float_tensor(gather_neighbor_matrix(ytr_s, nb), device = device)

      out <- model(
        xb, cb, xnb, cnb, ynb,
        use_residual = TRUE,
        residual_scale = residual_scale
      )
      loss <- huber_loss(yb, out$pred, delta = pred_huber_delta) +
        base_loss_weight * huber_loss(yb, out$base_pred, delta = base_huber_delta)

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[GeoTransformerKrigingNet] Epoch %d | residual_scale=%.2f | batch %d/%d | batch_loss=%.4f\n",
                    ep, residual_scale, batch_id, length(batches), loss$item()))
      }
    }

    val_pred_s <- predict_geotransformerkrigingnet(
      model = model,
      X_query = Xva,
      coords_query = Cva,
      neighbor_idx = fd$neighbor_idx$val,
      X_ref = Xtr,
      coords_ref = Ctr,
      y_ref = ytr_s,
      device = device,
      batch_size = batch_size,
      prediction = "full"
    )
    vloss <- mean((yva_s - val_pred_s)^2)
    cat(sprintf("[GeoTransformerKrigingNet] Epoch %d complete | residual_scale=%.2f | train_loss=%.4f | val_mse=%.4f\n",
                ep, residual_scale, train_loss / length(batches), vloss))

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

  preds_scaled <- predict_geotransformerkrigingnet(
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

geotransformerkrigingnet_params <- list(
  epochs = 60,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 10,
  d = 256,
  target_hidden = c(256, 128),
  neighbor_hidden = c(192),
  token_dropout = 0.10,
  n_heads = 4,
  n_layers = 1,
  ff_mult = 2,
  beta_init = -4,
  warmup_epochs = 8,
  warmup_lr_mult = 1,
  base_loss_weight = 0.25,
  pred_huber_delta = 1.0,
  base_huber_delta = 1.0,
  residual_start_scale = 0.10,
  residual_ramp_epochs = 10,
  target_transform = "identity",
  device = "cpu"
)

geotransformerkrigingnet_quick_params <- modifyList(
  geotransformerkrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    target_hidden = c(128),
    neighbor_hidden = c(96),
    n_heads = 4,
    n_layers = 1,
    beta_init = -5,
    warmup_epochs = 5,
    base_loss_weight = 0.25,
    pred_huber_delta = 1.0,
    base_huber_delta = 1.0,
    residual_start_scale = 0.05,
    residual_ramp_epochs = 8
  )
)

run_geotransformerkrigingnet_on_fixed_benchmark <- function(benchmark,
                                                            context = wadoux_context,
                                                            model_params = geotransformerkrigingnet_params,
                                                            K = 16) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoTransformerKrigingNet Fair] split %s | K=%d\n", sp$split_id, K))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geotransformer_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K = K
    )

    out <- do.call(train_geotransformerkrigingnet_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoTransformerKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geotransformerkrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                        sample_size = 250,
                                                        sampling = "simple_random",
                                                        n_folds = 5,
                                                        val_dist_km = 350,
                                                        val_frac = 0.2,
                                                        max_splits = 5,
                                                        seed = 123,
                                                        model_params = geotransformerkrigingnet_quick_params,
                                                        K = 16,
                                                        cubist_committees = 50,
                                                        cubist_neighbors = 5,
                                                        results_dir = "results/geotransformerkrigingnet_vs_cubist",
                                                        save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoTransformerKrigingNet vs Cubist BENCHMARK\n")
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

  geo_res <- run_geotransformerkrigingnet_on_fixed_benchmark(
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
    write.csv(geo_res, file.path(results_dir, "geotransformerkrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geotransformerkrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geotransformerkrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geotransformerkrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                                sample_size = 300,
                                                                sampling = "simple_random",
                                                                n_folds = 10,
                                                                val_dist_km = 350,
                                                                val_frac = 0.2,
                                                                max_splits = 10,
                                                                seed = 123,
                                                                model_params = geotransformerkrigingnet_params,
                                                                K = 16,
                                                                cubist_committees = 50,
                                                                cubist_neighbors = 5,
                                                                results_dir = "results/geotransformerkrigingnet_vs_cubist_confirmation",
                                                                save_outputs = TRUE) {
  run_geotransformerkrigingnet_vs_cubist_fair(
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

geoversa_params <- geotransformerkrigingnet_params
geoversa_quick_params <- geotransformerkrigingnet_quick_params
geoversa_k12_targetwide_params <- modifyList(
  geoversa_quick_params,
  list(
    target_hidden = c(192, 96)
  )
)
geoversa_k12_targetwide_stable_params <- modifyList(
  geoversa_k12_targetwide_params,
  list(
    base_loss_weight = 0.50
  )
)
geoversa_k12_targetwide_stable_hubertight_params <- modifyList(
  geoversa_k12_targetwide_stable_params,
  list(
    pred_huber_delta = 0.5,
    base_huber_delta = 0.75
  )
)

run_geoversa_on_fixed_benchmark <- function(benchmark,
                                            context = wadoux_context,
                                            model_params = geoversa_params,
                                            K = 16) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoVersa Fair] split %s | K=%d\n", sp$split_id, K))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geotransformer_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K = K
    )

    out <- do.call(train_geotransformerkrigingnet_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoVersa",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geoversa_vs_cubist_fair <- function(context = wadoux_context,
                                        sample_size = 250,
                                        sampling = "simple_random",
                                        n_folds = 5,
                                        val_dist_km = 350,
                                        val_frac = 0.2,
                                        max_splits = 5,
                                        seed = 123,
                                        model_params = geoversa_quick_params,
                                        K = 16,
                                        cubist_committees = 50,
                                        cubist_neighbors = 5,
                                        results_dir = "results/geoversa_vs_cubist",
                                        save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoVersa vs Cubist BENCHMARK\n")
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

  geo_res <- run_geoversa_on_fixed_benchmark(
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
    write.csv(geo_res, file.path(results_dir, "geoversa_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geoversa_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geoversa_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geoversa_vs_cubist_confirmation <- function(context = wadoux_context,
                                                sample_size = 300,
                                                sampling = "simple_random",
                                                n_folds = 10,
                                                val_dist_km = 350,
                                                val_frac = 0.2,
                                                max_splits = 10,
                                                seed = 123,
                                                model_params = geoversa_params,
                                                K = 16,
                                                cubist_committees = 50,
                                                cubist_neighbors = 5,
                                                results_dir = "results/geoversa_vs_cubist_confirmation",
                                                save_outputs = TRUE) {
  run_geoversa_vs_cubist_fair(
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

make_geoversa_k_variants <- function() {
  list(
    GeoVersa_K8 = list(K = 8, model_params = geoversa_quick_params),
    GeoVersa_K12 = list(K = 12, model_params = geoversa_quick_params),
    GeoVersa_K16 = list(K = 16, model_params = geoversa_quick_params)
  )
}

make_geoversa_k12_target_variants <- function() {
  list(
    GeoVersa_K12_Base = list(K = 12, model_params = geoversa_quick_params),
    GeoVersa_K12_TargetWide = list(K = 12, model_params = geoversa_k12_targetwide_params)
  )
}

make_geoversa_k12_stability_variants <- function() {
  list(
    GeoVersa_K12_TargetWide = list(K = 12, model_params = geoversa_k12_targetwide_params),
    GeoVersa_K12_TargetWide_Stable = list(K = 12, model_params = geoversa_k12_targetwide_stable_params)
  )
}

make_geoversa_k12_huber_variants <- function() {
  list(
    GeoVersa_K12_TargetWide_Stable = list(K = 12, model_params = geoversa_k12_targetwide_stable_params),
    GeoVersa_K12_TargetWide_Stable_HuberTight = list(K = 12, model_params = geoversa_k12_targetwide_stable_hubertight_params)
  )
}

make_geoversa_confirmation_variants <- function() {
  make_geoversa_k12_huber_variants()
}

run_geoversa_k_search <- function(context = wadoux_context,
                                  sample_size = 250,
                                  sampling = "simple_random",
                                  variants = make_geoversa_k_variants(),
                                  n_folds = 5,
                                  val_dist_km = 350,
                                  val_frac = 0.2,
                                  max_splits = 5,
                                  seed = 123,
                                  cubist_committees = 50,
                                  cubist_neighbors = 5,
                                  results_dir = "results/geoversa_k_search",
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

  all_results <- list()
  for (variant_name in names(variants)) {
    cat("\n========================================\n")
    cat("GeoVersa K SEARCH:", variant_name, "\n")
    cat("========================================\n")

    cfg <- variants[[variant_name]]
    geo_res <- run_geoversa_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      model_params = cfg$model_params,
      K = cfg$K
    )

    variant_res <- bind_rows(
      cubist_res %>% mutate(variant = variant_name),
      geo_res %>% mutate(variant = variant_name)
    )
    all_results[[variant_name]] <- variant_res

    if (save_outputs) {
      write.csv(
        variant_res,
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
    write.csv(cubist_res, file.path(results_dir, "cubist_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geoversa_k_search_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geoversa_k_search_summary.csv"), row.names = FALSE)
  }

  final
}

run_geoversa_k12_target_search <- function(context = wadoux_context,
                                           sample_size = 250,
                                           sampling = "simple_random",
                                           variants = make_geoversa_k12_target_variants(),
                                           n_folds = 5,
                                           val_dist_km = 350,
                                           val_frac = 0.2,
                                           max_splits = 5,
                                           seed = 123,
                                           cubist_committees = 50,
                                           cubist_neighbors = 5,
                                           results_dir = "results/geoversa_k12_target_search",
                                           save_outputs = TRUE) {
  run_geoversa_k_search(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    variants = variants,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

run_geoversa_k12_stability_search <- function(context = wadoux_context,
                                              sample_size = 250,
                                              sampling = "simple_random",
                                              variants = make_geoversa_k12_stability_variants(),
                                              n_folds = 5,
                                              val_dist_km = 350,
                                              val_frac = 0.2,
                                              max_splits = 5,
                                              seed = 123,
                                              cubist_committees = 50,
                                              cubist_neighbors = 5,
                                              results_dir = "results/geoversa_k12_stability_search",
                                              save_outputs = TRUE) {
  run_geoversa_k_search(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    variants = variants,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

run_geoversa_k12_huber_search <- function(context = wadoux_context,
                                          sample_size = 250,
                                          sampling = "simple_random",
                                          variants = make_geoversa_k12_huber_variants(),
                                          n_folds = 5,
                                          val_dist_km = 350,
                                          val_frac = 0.2,
                                          max_splits = 5,
                                          seed = 123,
                                          cubist_committees = 50,
                                          cubist_neighbors = 5,
                                          results_dir = "results/geoversa_k12_huber_search",
                                          save_outputs = TRUE) {
  run_geoversa_k_search(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    variants = variants,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

run_geoversa_confirmation <- function(context = wadoux_context,
                                      sample_size = 300,
                                      sampling = "simple_random",
                                      variants = make_geoversa_confirmation_variants(),
                                      n_folds = 10,
                                      val_dist_km = 350,
                                      val_frac = 0.2,
                                      max_splits = 10,
                                      seed = 123,
                                      cubist_committees = 50,
                                      cubist_neighbors = 5,
                                      results_dir = "results/geoversa_confirmation",
                                      save_outputs = TRUE) {
  run_geoversa_k_search(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    variants = variants,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

# Example:
# source("code/GeoTransformerKrigingNet.R")
# res_geo_transformer <- run_geotransformerkrigingnet_vs_cubist_fair(
#   context = wadoux_context,
#   sample_size = 250,
#   n_folds = 5,
#   max_splits = 5,
#   model_params = geotransformerkrigingnet_quick_params,
#   K = 16
# )
# res_geo_transformer %>%
#   dplyr::group_by(model) %>%
#   dplyr::summarise(
#     RMSE_mean = mean(RMSE, na.rm = TRUE),
#     R2_mean = mean(R2, na.rm = TRUE),
#     MAE_mean = mean(MAE, na.rm = TRUE),
#     Bias_mean = mean(Bias, na.rm = TRUE),
#     .groups = "drop"
#   )
