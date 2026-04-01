rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)

# =============================================================================
# ConvKrigingNet2D
# - point-based prediction
# - 2D CNN over raster patches around each point
# - tabular + coordinates + patch context
# - residual kriging-like correction over sampled neighbors
# =============================================================================

ConvBlock2D <- nn_module(
  "ConvBlock2D",
  initialize = function(in_channels, out_channels, dropout = 0.05) {
    self$block <- nn_sequential(
      nn_conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
      nn_batch_norm2d(out_channels),
      nn_gelu(),
      nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
      nn_batch_norm2d(out_channels),
      nn_gelu(),
      nn_dropout2d(dropout)
    )
    self$skip <- if (in_channels != out_channels) nn_conv2d(in_channels, out_channels, kernel_size = 1) else NULL
  },
  forward = function(x) {
    residual <- if (is.null(self$skip)) x else self$skip(x)
    self$block(x) + residual
  }
)

PatchEncoder2D <- nn_module(
  "PatchEncoder2D",
  initialize = function(in_channels, out_dim = 128, dropout = 0.10) {
    self$stem <- nn_sequential(
      ConvBlock2D(in_channels, 32, dropout = dropout),
      nn_max_pool2d(kernel_size = 2),
      ConvBlock2D(32, 64, dropout = dropout),
      nn_max_pool2d(kernel_size = 2),
      ConvBlock2D(64, 96, dropout = dropout),
      nn_adaptive_avg_pool2d(output_size = c(1, 1))
    )
    self$head <- nn_sequential(
      nn_flatten(),
      nn_linear(96, out_dim),
      nn_gelu(),
      nn_dropout(dropout)
    )
  },
  forward = function(x_patch) {
    self$head(self$stem(x_patch))
  }
)

AnisotropicResidualKrigingLayer <- nn_module(
  "AnisotropicResidualKrigingLayer",
  initialize = function(d = 256, proj_d = 64, init_ell_major = 1000, init_ell_minor = 500) {
    self$proj <- nn_linear(d, proj_d, bias = FALSE)
    self$log_ell_major <- nn_parameter(torch_log(torch_tensor(init_ell_major)))
    self$log_ell_minor <- nn_parameter(torch_log(torch_tensor(init_ell_minor)))
    self$theta <- nn_parameter(torch_tensor(0))
    self$scale <- 1 / sqrt(proj_d)
  },
  forward = function(z_i, coords_i, z_n, coords_n, r_n) {
    dx <- coords_i[, 1]$unsqueeze(2) - coords_n[, , 1]
    dy <- coords_i[, 2]$unsqueeze(2) - coords_n[, , 2]

    theta <- self$theta
    cth <- torch_cos(theta)
    sth <- torch_sin(theta)

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

ConvKrigingNet2D <- nn_module(
  "ConvKrigingNet2D",
  initialize = function(c_tab,
                        patch_channels,
                        d = 256,
                        tab_hidden = c(192),
                        tab_dropout = 0.15,
                        patch_dim = 128,
                        patch_dropout = 0.10,
                        coord_hidden = c(32),
                        coord_dim = 32,
                        coord_dropout = 0.05,
                        fusion_hidden = 256,
                        kriging_mode = c("isotropic", "anisotropic"),
                        beta_init = -4) {
    kriging_mode <- match.arg(kriging_mode)
    self$kriging_mode <- kriging_mode
    self$enc_tab <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = tab_dropout)
    self$enc_patch <- PatchEncoder2D(
      in_channels = patch_channels,
      out_dim = patch_dim,
      dropout = patch_dropout
    )
    self$proj_patch <- nn_linear(patch_dim, d)
    self$enc_coord <- make_mlp(2, hidden = coord_hidden, out_dim = coord_dim, dropout = coord_dropout)
    self$proj_coord <- nn_linear(coord_dim, d)

    self$fuse <- nn_sequential(
      nn_linear(3 * d, fusion_hidden),
      nn_gelu(),
      nn_dropout(0.10),
      nn_linear(fusion_hidden, d)
    )

    self$head <- ScalarHead(d = d)
    if (kriging_mode == "anisotropic") {
      self$krig <- AnisotropicResidualKrigingLayer(
        d = d,
        proj_d = 64,
        init_ell_major = 1000,
        init_ell_minor = 500
      )
    } else {
      self$krig <- ResidualKrigingLayer(d = d, proj_d = 64, init_ell = 1000)
    }
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode = function(x_tab, x_patch, coords) {
    z_tab <- self$enc_tab(x_tab)
    z_patch <- self$proj_patch(self$enc_patch(x_patch))
    z_coord <- self$proj_coord(self$enc_coord(coords))
    z <- self$fuse(torch_cat(list(z_tab, z_patch, z_coord), dim = 2))
    list(z = z, z_tab = z_tab, z_patch = z_patch, z_coord = z_coord)
  },

  forward_base = function(x_tab, x_patch, coords) {
    enc <- self$encode(x_tab, x_patch, coords)
    pred <- self$head(enc$z)
    list(pred = pred, z = enc$z)
  },

  forward_with_kriging = function(x_tab, x_patch, coords, z_n, coords_n, r_n) {
    base <- self$forward_base(x_tab, x_patch, coords)
    k <- self$krig(base$z, coords, z_n, coords_n, r_n)
    beta <- torch_sigmoid(self$logit_beta)
    pred_corr <- base$pred + beta * k$delta
    list(pred = pred_corr, base_pred = base$pred, z = base$z, delta = k$delta, beta = beta)
  }
)

fit_affine_calibrator <- function(y_true, y_pred, eps = 1e-8) {
  ok <- is.finite(y_true) & is.finite(y_pred)
  y_true <- y_true[ok]
  y_pred <- y_pred[ok]

  if (length(y_true) < 2 || stats::sd(y_pred) < eps) {
    return(list(intercept = 0, slope = 1))
  }

  fit <- tryCatch(
    stats::lm(y_true ~ y_pred),
    error = function(e) NULL
  )

  if (is.null(fit)) {
    return(list(intercept = 0, slope = 1))
  }

  coefs <- stats::coef(fit)
  if (length(coefs) < 2 || any(!is.finite(coefs))) {
    return(list(intercept = 0, slope = 1))
  }

  list(
    intercept = as.numeric(coefs[1]),
    slope = as.numeric(coefs[2])
  )
}

apply_affine_calibrator <- function(pred, calibrator) {
  calibrator$intercept + calibrator$slope * pred
}

set_convkrigingnet2d_seed <- function(seed = NULL) {
  if (is.null(seed)) return(invisible(FALSE))

  set.seed(seed)
  if (exists("torch_manual_seed", mode = "function")) {
    tryCatch(
      torch_manual_seed(as.integer(seed)),
      error = function(e) NULL
    )
  }

  invisible(TRUE)
}

make_convkrigingnet2d_batches <- function(n,
                                          batch_size = 256,
                                          seed = NULL,
                                          epoch = 1,
                                          deterministic = FALSE) {
  if (isTRUE(deterministic) && !is.null(seed)) {
    set.seed(as.integer(seed) + as.integer(epoch) - 1L)
  }

  idx <- sample.int(n)
  split(idx, ceiling(seq_along(idx) / batch_size))
}

set_optimizer_lr <- function(opt, lr) {
  if (!is.null(opt$defaults)) {
    opt$defaults$lr <- lr
  }
  if (!is.null(opt$param_groups)) {
    for (i in seq_along(opt$param_groups)) {
      opt$param_groups[[i]]$lr <- lr
    }
  }
  invisible(opt)
}

refresh_convkrigingnet2d_bank <- function(model,
                                          X_train,
                                          P_train,
                                          coords_train,
                                          y_train,
                                          device = "cpu",
                                          batch_size = 256) {
  bank <- build_memory_bank_pointpatch(
    model = model,
    X_train = X_train,
    P_train = P_train,
    coords_train = coords_train,
    y_train = y_train,
    device = device,
    batch_size = batch_size
  )

  list(
    Zmem = bank$Z$to(device = device),
    Rmem = bank$R$to(device = device),
    Cmem = bank$C$to(device = device)
  )
}

build_convkrigingnet2d_tensor_cache <- function(X,
                                                P,
                                                coords,
                                                y = NULL,
                                                device = "cpu") {
  list(
    X = to_float_tensor(X, device = device),
    P = patches_to_torch(P, device = device),
    C = to_float_tensor(coords, device = device),
    y = if (is.null(y)) NULL else to_float_tensor(y, device = device),
    n = nrow(X),
    device = device
  )
}

build_memory_bank_pointpatch_tensor <- function(model,
                                                tensor_cache,
                                                batch_size = 256) {
  model$eval()
  n <- tensor_cache$n

  Z_list <- list()
  R_list <- list()
  C_list <- list()

  with_no_grad({
    for (s in seq(1, n, by = batch_size)) {
      e <- min(s + batch_size - 1, n)
      idx <- s:e
      idx_t <- torch_tensor(idx, dtype = torch_long(), device = tensor_cache$device)

      xb <- tensor_cache$X$index_select(1, idx_t)
      pb <- tensor_cache$P$index_select(1, idx_t)
      cb <- tensor_cache$C$index_select(1, idx_t)
      yb <- tensor_cache$y$index_select(1, idx_t)

      out <- model$forward_base(xb, pb, cb)
      r <- yb - out$pred

      Z_list[[length(Z_list) + 1]] <- out$z$cpu()
      R_list[[length(R_list) + 1]] <- r$cpu()
      C_list[[length(C_list) + 1]] <- cb$cpu()
    }
  })

  list(
    Z = torch_cat(Z_list, dim = 1),
    R = torch_cat(R_list, dim = 1),
    C = torch_cat(C_list, dim = 1)
  )
}

refresh_convkrigingnet2d_bank_tensor <- function(model,
                                                 tensor_cache,
                                                 batch_size = 256) {
  bank <- build_memory_bank_pointpatch_tensor(
    model = model,
    tensor_cache = tensor_cache,
    batch_size = batch_size
  )

  list(
    Zmem = bank$Z$to(device = tensor_cache$device),
    Rmem = bank$R$to(device = tensor_cache$device),
    Cmem = bank$C$to(device = tensor_cache$device)
  )
}

predict_convkrigingnet2d_base <- function(model,
                                          X_new,
                                          P_new,
                                          coords_new,
                                          device = "cpu",
                                          batch_size = 256) {
  preds <- numeric(nrow(X_new))

  with_no_grad({
    for (s in seq(1, nrow(X_new), by = batch_size)) {
      e <- min(s + batch_size - 1, nrow(X_new))
      idx <- s:e

      xb <- to_float_tensor(X_new[idx, , drop = FALSE], device = device)
      pb <- patches_to_torch(P_new[, , , idx, drop = FALSE], device = device)
      cb <- to_float_tensor(coords_new[idx, , drop = FALSE], device = device)

      out <- model$forward_base(xb, pb, cb)
      preds[idx] <- as.numeric(out$pred$cpu())
    }
  })

  preds
}

predict_convkrigingnet2d_base_tensor <- function(model,
                                                 tensor_cache,
                                                 batch_size = 256) {
  preds <- numeric(tensor_cache$n)

  with_no_grad({
    for (s in seq(1, tensor_cache$n, by = batch_size)) {
      e <- min(s + batch_size - 1, tensor_cache$n)
      idx <- s:e
      idx_t <- torch_tensor(idx, dtype = torch_long(), device = tensor_cache$device)

      xb <- tensor_cache$X$index_select(1, idx_t)
      pb <- tensor_cache$P$index_select(1, idx_t)
      cb <- tensor_cache$C$index_select(1, idx_t)

      out <- model$forward_base(xb, pb, cb)
      preds[idx] <- as.numeric(out$pred$cpu())
    }
  })

  preds
}

compute_neighbor_idx_query_to_ref <- function(coords_query,
                                              coords_ref,
                                              K) {
  if (nrow(coords_query) == 0 || nrow(coords_ref) == 0) {
    return(matrix(integer(0), nrow = nrow(coords_query), ncol = 0))
  }

  K_eff <- min(as.integer(K), nrow(coords_ref))
  query_sq <- rowSums(coords_query^2)
  ref_sq <- rowSums(coords_ref^2)
  d2 <- outer(query_sq, ref_sq, "+") - 2 * tcrossprod(coords_query, coords_ref)
  d2[d2 < 0] <- 0

  knn <- t(apply(d2, 1, function(row) {
    order(row)[seq_len(K_eff)]
  }))
  if (is.null(dim(knn))) {
    knn <- matrix(knn, nrow = 1)
  }
  storage.mode(knn) <- "integer"
  knn
}

predict_with_memory_pointpatch_tensor <- function(model,
                                                  tensor_cache,
                                                  Zmem,
                                                  Rmem,
                                                  Cmem,
                                                  K,
                                                  device = "cpu",
                                                  batch_size = 256,
                                                  knn_idx_t = NULL) {
  preds <- numeric(tensor_cache$n)

  with_no_grad({
    for (s in seq(1, tensor_cache$n, by = batch_size)) {
      e <- min(s + batch_size - 1, tensor_cache$n)
      idx <- s:e
      idx_t <- torch_tensor(idx, dtype = torch_long(), device = tensor_cache$device)
      B <- length(idx)

      xb <- tensor_cache$X$index_select(1, idx_t)
      pb <- tensor_cache$P$index_select(1, idx_t)
      cb <- tensor_cache$C$index_select(1, idx_t)

      if (is.null(knn_idx_t)) {
        d <- cdist_safe(cb, Cmem)
        knn <- topk_smallest_idx(d, K)
        nb_flat <- flatten_safe(knn)$to(dtype = torch_long())
      } else {
        knn <- knn_idx_t$index_select(1, idx_t)
        nb_flat <- flatten_safe(knn)$to(dtype = torch_long())
      }

      zn <- reshape_safe(Zmem$index_select(1, nb_flat), c(B, K, -1))
      rn <- reshape_safe(Rmem$index_select(1, nb_flat), c(B, K))
      cn <- reshape_safe(Cmem$index_select(1, nb_flat), c(B, K, 2))

      out <- model$forward_with_kriging(xb, pb, cb, zn, cn, rn)
      preds[idx] <- as.numeric(out$pred$cpu())
    }
  })

  preds
}

train_convkrigingnet2d_one_fold <- function(fd,
                                            epochs = 60,
                                            lr = 2e-4,
                                            wd = 1e-3,
                                            batch_size = 96,
                                            patience = 10,
                                            warmup_epochs = 0,
                                            bank_refresh_every = 1,
                                            train_seed = NULL,
                                            deterministic_batches = FALSE,
                                            lr_decay = 1.0,
                                            lr_patience = Inf,
                                            min_lr = NULL,
                                            base_loss_weight = 0,
                                            krig_loss_weight = 0,
                                            d = 256,
                                            tab_hidden = c(192),
                                            tab_dropout = 0.15,
                                            patch_dim = 128,
                                            patch_dropout = 0.10,
                                            coord_hidden = c(32),
                                            coord_dim = 32,
                                            coord_dropout = 0.05,
                                            fusion_hidden = 256,
                                            kriging_mode = "isotropic",
                                            beta_init = -4,
                                            target_transform = "identity",
                                            calibrate_method = "none",
                                            K_neighbors = 12,
                                            device = "cpu") {
  Xtr <- fd$X$train
  Xva <- fd$X$val
  Xte <- fd$X$test
  Ptr <- fd$patches$train
  Pva <- fd$patches$val
  Pte <- fd$patches$test
  Ctr <- fd$coords$train
  Cva <- fd$coords$val
  Cte <- fd$coords$test
  ytr <- fd$y$train
  yva <- fd$y$val
  yte <- fd$y$test

  ytr_t <- transform_target(ytr, target_transform)
  yva_t <- transform_target(yva, target_transform)
  y_scaler <- fit_target_scaler(ytr_t)
  coord_scaler <- fit_standard_scaler(Ctr)

  ytr_s <- apply_target_scaler(ytr_t, y_scaler)
  yva_s <- apply_target_scaler(yva_t, y_scaler)
  Ctr_s <- apply_standard_scaler(Ctr, coord_scaler)
  Cva_s <- apply_standard_scaler(Cva, coord_scaler)
  Cte_s <- apply_standard_scaler(Cte, coord_scaler)

  warmup_epochs <- max(0L, as.integer(warmup_epochs))
  bank_refresh_every <- max(1L, as.integer(bank_refresh_every))
  if (is.null(min_lr)) {
    min_lr <- lr
  }
  if (is.finite(lr_patience)) {
    lr_patience <- max(1L, as.integer(lr_patience))
  }

  set_convkrigingnet2d_seed(train_seed)

  neigh_train <- fd$neighbor_idx_train
  if (!is.null(K_neighbors)) {
    k_eff <- min(K_neighbors, ncol(neigh_train))
    neigh_train <- neigh_train[, seq_len(k_eff), drop = FALSE]
  }
  Ktr <- ncol(neigh_train)

  train_cache <- build_convkrigingnet2d_tensor_cache(
    X = Xtr,
    P = Ptr,
    coords = Ctr_s,
    y = ytr_s,
    device = device
  )
  val_cache <- build_convkrigingnet2d_tensor_cache(
    X = Xva,
    P = Pva,
    coords = Cva_s,
    y = yva_s,
    device = device
  )
  test_cache <- build_convkrigingnet2d_tensor_cache(
    X = Xte,
    P = Pte,
    coords = Cte_s,
    y = NULL,
    device = device
  )
  neigh_train_t <- torch_tensor(neigh_train, dtype = torch_long(), device = device)
  val_knn_t <- torch_tensor(
    compute_neighbor_idx_query_to_ref(Cva_s, Ctr_s, Ktr),
    dtype = torch_long(),
    device = device
  )
  test_knn_t <- torch_tensor(
    compute_neighbor_idx_query_to_ref(Cte_s, Ctr_s, Ktr),
    dtype = torch_long(),
    device = device
  )

  model <- ConvKrigingNet2D(
    c_tab = ncol(Xtr),
    patch_channels = dim(Ptr)[1],
    d = d,
    tab_hidden = tab_hidden,
    tab_dropout = tab_dropout,
    patch_dim = patch_dim,
    patch_dropout = patch_dropout,
    coord_hidden = coord_hidden,
    coord_dim = coord_dim,
    coord_dropout = coord_dropout,
    fusion_hidden = fusion_hidden,
    kriging_mode = kriging_mode,
    beta_init = beta_init
  )
  model$to(device = device)

  if (warmup_epochs > 0) {
    warmup_params <- c(
      model$enc_tab$parameters,
      model$enc_patch$parameters,
      model$proj_patch$parameters,
      model$enc_coord$parameters,
      model$proj_coord$parameters,
      model$fuse$parameters,
      model$head$parameters
    )
    warmup_opt <- optim_adamw(warmup_params, lr = lr, weight_decay = wd)

    for (ep in seq_len(warmup_epochs)) {
      model$train()
      batches <- make_convkrigingnet2d_batches(
        n = nrow(Xtr),
        batch_size = batch_size,
        seed = train_seed,
        epoch = ep,
        deterministic = deterministic_batches
      )
      train_loss <- 0

      for (batch_id in seq_along(batches)) {
        b <- batches[[batch_id]]
        b_t <- torch_tensor(b, dtype = torch_long(), device = device)
        xb <- train_cache$X$index_select(1, b_t)
        pb <- train_cache$P$index_select(1, b_t)
        cb <- train_cache$C$index_select(1, b_t)
        yb <- train_cache$y$index_select(1, b_t)

        out <- model$forward_base(xb, pb, cb)
        loss <- huber_loss(yb, out$pred)

        warmup_opt$zero_grad()
        loss$backward()
        nn_utils_clip_grad_norm_(warmup_params, max_norm = 2.0)
        warmup_opt$step()

        train_loss <- train_loss + loss$item()
      }

      val_base_pred <- predict_convkrigingnet2d_base_tensor(
        model = model,
        tensor_cache = val_cache,
        batch_size = batch_size
      )
      warmup_vloss <- huber_loss(
        val_cache$y,
        to_float_tensor(val_base_pred, device = device)
      )$item()

      cat(sprintf(
        "[ConvKrigingNet2D Warmup] Epoch %d/%d | train_loss=%.4f | val_base_loss=%.4f\n",
        ep, warmup_epochs, train_loss / length(batches), warmup_vloss
      ))
    }
  }

  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)
  current_lr <- lr
  best_val <- Inf
  best_state <- NULL
  bad <- 0
  lr_bad <- 0
  bank <- refresh_convkrigingnet2d_bank_tensor(
    model = model,
    tensor_cache = train_cache,
    batch_size = batch_size
  )

  for (ep in seq_len(epochs)) {
    model$train()
    batches <- make_convkrigingnet2d_batches(
      n = nrow(Xtr),
      batch_size = batch_size,
      seed = train_seed,
      epoch = warmup_epochs + ep,
      deterministic = deterministic_batches
    )
    train_loss <- 0

    for (batch_id in seq_along(batches)) {
      b <- batches[[batch_id]]
      b_t <- torch_tensor(b, dtype = torch_long(), device = device)
      xb <- train_cache$X$index_select(1, b_t)
      pb <- train_cache$P$index_select(1, b_t)
      cb <- train_cache$C$index_select(1, b_t)
      yb <- train_cache$y$index_select(1, b_t)

      nb <- neigh_train_t$index_select(1, b_t)
      nb_t <- flatten_safe(nb)$to(dtype = torch_long())
      zn <- reshape_safe(bank$Zmem$index_select(1, nb_t), c(length(b), Ktr, -1))
      rn <- reshape_safe(bank$Rmem$index_select(1, nb_t), c(length(b), Ktr))
      cn <- reshape_safe(bank$Cmem$index_select(1, nb_t), c(length(b), Ktr, 2))

      out <- model$forward_with_kriging(xb, pb, cb, zn, cn, rn)
      loss <- huber_loss(yb, out$pred)
      if (base_loss_weight > 0) {
        loss <- loss + base_loss_weight * huber_loss(yb, out$base_pred)
      }
      # Direct kriging supervision: delta should approximate (y - base_pred).
      # Gives direct gradient signal to kriging parameters (log_ell, theta,
      # proj) which otherwise only receive weak indirect gradients through beta.
      # ideal_delta is detached so it does not pull the base network backward.
      if (krig_loss_weight > 0) {
        ideal_delta <- (yb - out$base_pred$detach())$detach()
        loss <- loss + krig_loss_weight * huber_loss(ideal_delta, out$delta)
      }

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[ConvKrigingNet2D] Epoch %d | lr=%.6f | batch %d/%d | batch_loss=%.4f\n",
                    ep, current_lr, batch_id, length(batches), loss$item()))
      }
    }

    refresh_bank_now <- (ep %% bank_refresh_every == 0) || ep == epochs
    if (refresh_bank_now) {
      cat(sprintf("[ConvKrigingNet2D] Refreshing memory bank after epoch %d...\n", ep))
      bank <- refresh_convkrigingnet2d_bank_tensor(
        model = model,
        tensor_cache = train_cache,
        batch_size = batch_size
      )
    }

    model$eval()
    val_pred <- predict_with_memory_pointpatch_tensor(
      model = model,
      tensor_cache = val_cache,
      Zmem = bank$Zmem,
      Rmem = bank$Rmem,
      Cmem = bank$Cmem,
      K = Ktr,
      device = device,
      batch_size = batch_size,
      knn_idx_t = val_knn_t
    )
    vloss <- huber_loss(
      val_cache$y,
      to_float_tensor(val_pred, device = device)
    )$item()

    cat(sprintf("[ConvKrigingNet2D] Epoch %d complete | lr=%.6f | bank_refresh=%s | train_loss=%.4f | val_loss=%.4f\n",
                ep, current_lr, if (refresh_bank_now) "yes" else "no", train_loss / length(batches), vloss))

    if (vloss < best_val) {
      best_val <- vloss
      best_state <- clone_state_dict(model$state_dict())
      bad <- 0
      lr_bad <- 0
    } else {
      bad <- bad + 1
      lr_bad <- lr_bad + 1
      if (is.finite(lr_patience) && lr_decay < 1 && lr_bad >= lr_patience && current_lr > min_lr) {
        current_lr <- max(min_lr, current_lr * lr_decay)
        set_optimizer_lr(opt, current_lr)
        lr_bad <- 0
        cat(sprintf("[ConvKrigingNet2D] Reducing learning rate to %.6f\n", current_lr))
      }
      if (bad >= patience) break
    }
  }

  model$load_state_dict(best_state)
  model$eval()

  bank <- refresh_convkrigingnet2d_bank_tensor(
    model = model,
    tensor_cache = train_cache,
    batch_size = batch_size
  )
  val_preds_scaled <- predict_with_memory_pointpatch_tensor(
    model = model,
    tensor_cache = val_cache,
    Zmem = bank$Zmem,
    Rmem = bank$Rmem,
    Cmem = bank$Cmem,
    K = ncol(neigh_train),
    device = device,
    batch_size = batch_size,
    knn_idx_t = val_knn_t
  )
  preds_scaled <- predict_with_memory_pointpatch_tensor(
    model = model,
    tensor_cache = test_cache,
    Zmem = bank$Zmem,
    Rmem = bank$Rmem,
    Cmem = bank$Cmem,
    K = ncol(neigh_train),
    device = device,
    batch_size = batch_size,
    knn_idx_t = test_knn_t
  )
  val_preds_t <- invert_target_scaler(val_preds_scaled, y_scaler)
  val_preds <- inverse_transform_target(val_preds_t, target_transform)
  preds_t <- invert_target_scaler(preds_scaled, y_scaler)
  preds <- inverse_transform_target(preds_t, target_transform)

  calibrator <- list(intercept = 0, slope = 1)
  if (identical(calibrate_method, "linear")) {
    calibrator <- fit_affine_calibrator(yva, val_preds)
    preds <- apply_affine_calibrator(preds, calibrator)
  }

  list(
    pred_test = preds,
    pred_val = val_preds,
    calibrator = calibrator,
    metrics_test = metrics(yte, preds)
  )
}

convkriging2d_baseline_params <- list(
  epochs = 60,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 10,
  d = 256,
  tab_hidden = c(192),
  tab_dropout = 0.15,
  patch_dim = 128,
  patch_dropout = 0.10,
  coord_hidden = c(32),
  coord_dim = 32,
  coord_dropout = 0.05,
  fusion_hidden = 256,
  kriging_mode = "isotropic",
  beta_init = -4,
  target_transform = "identity",
  calibrate_method = "none",
  K_neighbors = 12,
  device = "cpu"
)

convkriging2d_params <- convkriging2d_baseline_params

convkriging2d_anisotropic_stable_params <- modifyList(
  convkriging2d_baseline_params,
  list(
    kriging_mode = "anisotropic",
    lr = 1.5e-4,
    warmup_epochs = 10,
    bank_refresh_every = 3,
    deterministic_batches = TRUE,
    lr_decay = 0.6,
    lr_patience = 3,
    min_lr = 5e-5,
    base_loss_weight = 0.10,
    tab_dropout = 0.10,
    patch_dropout = 0.05,
    coord_dropout = 0.02
  )
)

convkriging2d_quick_params <- modifyList(
  convkriging2d_baseline_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    tab_hidden = c(128),
    patch_dim = 64,
    coord_hidden = c(16),
    coord_dim = 16,
    fusion_hidden = 128,
    K_neighbors = 8
  )
)

make_convkriging2d_arch_variants <- function() {
  list(
    ConvKrigingNet2D_Baseline = list(
      patch_size = 15,
      params = convkriging2d_baseline_params
    ),
    ConvKrigingNet2D_Wide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          d = 320,
          patch_dim = 160,
          fusion_hidden = 320
        )
      )
    ),
    ConvKrigingNet2D_TabDeep = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          d = 256,
          tab_hidden = c(256, 128),
          fusion_hidden = 256
        )
      )
    ),
    ConvKrigingNet2D_CoordWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          coord_hidden = c(64, 32),
          coord_dim = 48,
          fusion_hidden = 256
        )
      )
    ),
    ConvKrigingNet2D_FusionWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          fusion_hidden = 384
        )
      )
    )
  )
}

make_convkriging2d_anisotropic_variants <- function() {
  list(
    ConvKrigingNet2D_Anisotropic = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic"
        )
      )
    ),
    ConvKrigingNet2D_Anisotropic_FusionWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          fusion_hidden = 384
        )
      )
    )
  )
}

make_convkriging2d_confirmation_variants <- function() {
  list(
    ConvKrigingNet2D_TabDeep = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          d = 256,
          tab_hidden = c(256, 128),
          fusion_hidden = 256
        )
      )
    )
  )
}

make_convkriging2d_fusionwide_variants <- function() {
  list(
    ConvKrigingNet2D_FusionWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          fusion_hidden = 384
        )
      )
    )
  )
}

make_convkriging2d_anisotropic_confirmation_variants <- function() {
  list(
    ConvKrigingNet2D_Anisotropic = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic"
        )
      )
    )
  )
}

make_convkriging2d_anisotropic_search_variants <- function() {
  list(
    ConvKrigingNet2D_Anisotropic = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic"
        )
      )
    ),
    ConvKrigingNet2D_Anisotropic_CoordWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          coord_hidden = c(64, 32),
          coord_dim = 48
        )
      )
    ),
    ConvKrigingNet2D_Anisotropic_FusionWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          fusion_hidden = 384
        )
      )
    ),
    ConvKrigingNet2D_Anisotropic_CoordFusionWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          coord_hidden = c(64, 32),
          coord_dim = 48,
          fusion_hidden = 384
        )
      )
    ),
    ConvKrigingNet2D_Anisotropic_PatchWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          patch_dim = 160,
          d = 288,
          fusion_hidden = 320
        )
      )
    )
  )
}

make_convkriging2d_anisotropic_finalists <- function() {
  list(
    ConvKrigingNet2D_Anisotropic_PatchWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          patch_dim = 160,
          d = 288,
          fusion_hidden = 320
        )
      )
    ),
    ConvKrigingNet2D_Anisotropic_CoordFusionWide = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          coord_hidden = c(64, 32),
          coord_dim = 48,
          fusion_hidden = 384
        )
      )
    )
  )
}

run_convkriging2d_wadoux_spatial_kfold <- function(context = wadoux_context,
                                                   sample_size = 250,
                                                   n_folds = 5,
                                                   val_dist_km = 350,
                                                   patch_size = 15,
                                                   model_params = convkriging2d_params,
                                                   max_splits = 5,
                                                   seed = 123) {
  benchmark <- build_pointpatch_fixed_spatial_kfold_benchmark(
    context = context,
    sample_size = sample_size,
    sampling = "simple_random",
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = 0.2,
    max_splits = max_splits,
    seed = seed
  )

  results <- vector("list", length(benchmark$splits))
  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[ConvKrigingNet2D] split %s | patch=%d\n", sp$split_id, patch_size))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_pointpatch_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      patch_size = patch_size,
      K = max(24, model_params$K_neighbors)
    )

    out <- do.call(train_convkrigingnet2d_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "ConvKrigingNet2D",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_convkriging2d_on_fixed_benchmark <- function(benchmark,
                                                 context = wadoux_context,
                                                 patch_size = 15,
                                                 model_params = convkriging2d_params) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[ConvKrigingNet2D Fair] split %s | patch=%d\n",
                sp$split_id, patch_size))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_pointpatch_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      patch_size = patch_size,
      K = max(24, model_params$K_neighbors)
    )

    out <- do.call(train_convkrigingnet2d_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "ConvKrigingNet2D",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_convkriging2d_vs_cubist_fair <- function(context = wadoux_context,
                                             sample_size = 250,
                                             sampling = "simple_random",
                                             n_folds = 5,
                                             val_dist_km = 350,
                                             val_frac = 0.2,
                                             max_splits = 5,
                                             seed = 123,
                                             patch_size = 15,
                                             model_params = convkriging2d_quick_params,
                                             cubist_committees = 50,
                                             cubist_neighbors = 5,
                                             results_dir = "results/convkriging2d_vs_cubist",
                                             save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING ConvKrigingNet2D vs Cubist BENCHMARK\n")
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
    write.csv(
      benchmark$calibration_df,
      file.path(results_dir, "fixed_calibration_sample.csv"),
      row.names = FALSE
    )
    write.csv(
      manifest,
      file.path(results_dir, "fixed_split_manifest.csv"),
      row.names = FALSE
    )
  }

  cubist_res <- run_cubist_on_pointpatch_benchmark(
    benchmark = benchmark,
    context = context,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors
  )

  conv_res <- run_convkriging2d_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    patch_size = patch_size,
    model_params = model_params
  )

  final <- bind_rows(cubist_res, conv_res)

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
    write.csv(conv_res, file.path(results_dir, "convkriging2d_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "convkriging2d_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "convkriging2d_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_convkriging2d_vs_cubist_confirmation <- function(context = wadoux_context,
                                                     sample_size = 300,
                                                     sampling = "simple_random",
                                                     n_folds = 10,
                                                     val_dist_km = 350,
                                                     val_frac = 0.2,
                                                     max_splits = 10,
                                                     seed = 123,
                                                     patch_size = 15,
                                                     model_params = convkriging2d_params,
                                                     cubist_committees = 50,
                                                     cubist_neighbors = 5,
                                                     results_dir = "results/convkriging2d_vs_cubist_confirmation",
                                                     save_outputs = TRUE) {
  run_convkriging2d_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    patch_size = patch_size,
    model_params = model_params,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

run_convkriging2d_architecture_variants <- function(context = wadoux_context,
                                                    sample_size = 250,
                                                    sampling = "simple_random",
                                                    variants = make_convkriging2d_arch_variants(),
                                                    n_folds = 5,
                                                    val_dist_km = 350,
                                                    val_frac = 0.2,
                                                    max_splits = 5,
                                                    seed = 123,
                                                    cubist_committees = 50,
                                                    cubist_neighbors = 5,
                                                    results_dir = "results/convkriging2d_architecture",
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
    write.csv(
      benchmark$calibration_df,
      file.path(results_dir, "fixed_calibration_sample.csv"),
      row.names = FALSE
    )
    write.csv(
      manifest,
      file.path(results_dir, "fixed_split_manifest.csv"),
      row.names = FALSE
    )
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
    cat("ARCH VARIANT:", variant_name, "\n")
    cat("========================================\n")

    var_cfg <- variants[[variant_name]]
    conv_res <- run_convkriging2d_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      patch_size = var_cfg$patch_size,
      model_params = var_cfg$params
    )

    variant_res <- bind_rows(
      cubist_res %>% mutate(variant = variant_name),
      conv_res %>% mutate(variant = variant_name)
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
    write.csv(final, file.path(results_dir, "convkriging2d_architecture_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "convkriging2d_architecture_summary.csv"), row.names = FALSE)
  }

  final
}

run_convkriging2d_tabdeep_confirmation <- function(context = wadoux_context,
                                                   sample_size = 300,
                                                   sampling = "simple_random",
                                                   n_folds = 10,
                                                   val_dist_km = 350,
                                                   val_frac = 0.2,
                                                   max_splits = 10,
                                                   seed = 123,
                                                   cubist_committees = 50,
                                                   cubist_neighbors = 5,
                                                   results_dir = "results/convkriging2d_tabdeep_confirmation",
                                                   save_outputs = TRUE) {
  variants <- make_convkriging2d_confirmation_variants()
  var_cfg <- variants[[1]]

  run_convkriging2d_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    patch_size = var_cfg$patch_size,
    model_params = var_cfg$params,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

run_convkriging2d_fusionwide_confirmation <- function(context = wadoux_context,
                                                      sample_size = 300,
                                                      sampling = "simple_random",
                                                      n_folds = 10,
                                                      val_dist_km = 350,
                                                      val_frac = 0.2,
                                                      max_splits = 10,
                                                      seed = 123,
                                                      cubist_committees = 50,
                                                      cubist_neighbors = 5,
                                                      results_dir = "results/convkriging2d_fusionwide_confirmation",
                                                      save_outputs = TRUE) {
  variants <- make_convkriging2d_fusionwide_variants()
  var_cfg <- variants[[1]]

  run_convkriging2d_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    patch_size = var_cfg$patch_size,
    model_params = var_cfg$params,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

run_convkriging2d_anisotropic_confirmation <- function(context = wadoux_context,
                                                       sample_size = 300,
                                                       sampling = "simple_random",
                                                       n_folds = 10,
                                                       val_dist_km = 350,
                                                       val_frac = 0.2,
                                                       max_splits = 10,
                                                       seed = 123,
                                                       cubist_committees = 50,
                                                       cubist_neighbors = 5,
                                                       results_dir = "results/convkriging2d_anisotropic_confirmation",
                                                       save_outputs = TRUE) {
  variants <- make_convkriging2d_anisotropic_confirmation_variants()
  var_cfg <- variants[[1]]

  run_convkriging2d_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    patch_size = var_cfg$patch_size,
    model_params = var_cfg$params,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

make_convkriging2d_anisotropic_stable_variants <- function(train_seed = 123) {
  list(
    ConvKrigingNet2D_Anisotropic = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          train_seed = train_seed
        )
      )
    ),
    ConvKrigingNet2D_Anisotropic_Stable = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_anisotropic_stable_params,
        list(train_seed = train_seed)
      )
    )
  )
}

run_convkriging2d_anisotropic_multiseed_confirmation <- function(context = wadoux_context,
                                                                 sample_size = 300,
                                                                 sampling = "simple_random",
                                                                 n_folds = 10,
                                                                 val_dist_km = 350,
                                                                 val_frac = 0.2,
                                                                 max_splits = 10,
                                                                 train_seeds = c(11, 29, 47),
                                                                 cubist_committees = 50,
                                                                 cubist_neighbors = 5,
                                                                 results_dir = "results/convkriging2d_anisotropic_multiseed_confirmation",
                                                                 save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  all_results <- vector("list", length(train_seeds))

  for (i in seq_along(train_seeds)) {
    train_seed <- train_seeds[[i]]
    cat("\n========================================\n")
    cat(sprintf("ANISOTROPIC MULTISEED | train_seed = %s\n", train_seed))
    cat("========================================\n")

    seed_dir <- file.path(results_dir, paste0("seed_", train_seed))
    dir.create(seed_dir, recursive = TRUE, showWarnings = FALSE)

    seed_res <- run_convkriging2d_anisotropic_confirmation(
      context = context,
      sample_size = sample_size,
      sampling = sampling,
      n_folds = n_folds,
      val_dist_km = val_dist_km,
      val_frac = val_frac,
      max_splits = max_splits,
      seed = train_seed,
      cubist_committees = cubist_committees,
      cubist_neighbors = cubist_neighbors,
      results_dir = seed_dir,
      save_outputs = save_outputs
    ) %>%
      mutate(train_seed = train_seed)

    all_results[[i]] <- seed_res
  }

  final <- bind_rows(all_results)

  summary_by_seed <- final %>%
    group_by(train_seed, model) %>%
    summarise(
      RMSE_mean = mean(RMSE, na.rm = TRUE),
      R2_mean = mean(R2, na.rm = TRUE),
      MAE_mean = mean(MAE, na.rm = TRUE),
      Bias_mean = mean(Bias, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(train_seed, RMSE_mean)

  summary_overall <- final %>%
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
    write.csv(final, file.path(results_dir, "convkriging2d_anisotropic_multiseed_all.csv"), row.names = FALSE)
    write.csv(summary_by_seed, file.path(results_dir, "convkriging2d_anisotropic_multiseed_by_seed.csv"), row.names = FALSE)
    write.csv(summary_overall, file.path(results_dir, "convkriging2d_anisotropic_multiseed_summary.csv"), row.names = FALSE)
  }

  final
}

run_convkriging2d_anisotropic_stable_search <- function(context = wadoux_context,
                                                        sample_size = 250,
                                                        sampling = "simple_random",
                                                        n_folds = 5,
                                                        val_dist_km = 350,
                                                        val_frac = 0.2,
                                                        max_splits = 5,
                                                        seed = 123,
                                                        train_seed = 123,
                                                        cubist_committees = 50,
                                                        cubist_neighbors = 5,
                                                        results_dir = "results/convkriging2d_anisotropic_stable_search",
                                                        save_outputs = TRUE) {
  run_convkriging2d_anisotropic_variants(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    variants = make_convkriging2d_anisotropic_stable_variants(train_seed = train_seed),
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

run_convkriging2d_anisotropic_stable_confirmation <- function(context = wadoux_context,
                                                              sample_size = 300,
                                                              sampling = "simple_random",
                                                              n_folds = 10,
                                                              val_dist_km = 350,
                                                              val_frac = 0.2,
                                                              max_splits = 10,
                                                              seed = 123,
                                                              train_seed = 123,
                                                              cubist_committees = 50,
                                                              cubist_neighbors = 5,
                                                              results_dir = "results/convkriging2d_anisotropic_stable_confirmation",
                                                              save_outputs = TRUE) {
  variants <- make_convkriging2d_anisotropic_stable_variants(train_seed = train_seed)
  var_cfg <- variants[["ConvKrigingNet2D_Anisotropic_Stable"]]

  run_convkriging2d_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    patch_size = var_cfg$patch_size,
    model_params = var_cfg$params,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

run_convkriging2d_anisotropic_stable_multiseed_confirmation <- function(context = wadoux_context,
                                                                        sample_size = 300,
                                                                        sampling = "simple_random",
                                                                        n_folds = 10,
                                                                        val_dist_km = 350,
                                                                        val_frac = 0.2,
                                                                        max_splits = 10,
                                                                        seed = 123,
                                                                        train_seeds = c(11, 29, 47),
                                                                        cubist_committees = 50,
                                                                        cubist_neighbors = 5,
                                                                        results_dir = "results/convkriging2d_anisotropic_stable_multiseed_confirmation",
                                                                        save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  all_results <- vector("list", length(train_seeds))

  for (i in seq_along(train_seeds)) {
    train_seed <- train_seeds[[i]]
    cat("\n========================================\n")
    cat(sprintf("ANISOTROPIC STABLE MULTISEED | train_seed = %s\n", train_seed))
    cat("========================================\n")

    seed_dir <- file.path(results_dir, paste0("train_seed_", train_seed))
    dir.create(seed_dir, recursive = TRUE, showWarnings = FALSE)

    seed_res <- run_convkriging2d_anisotropic_stable_confirmation(
      context = context,
      sample_size = sample_size,
      sampling = sampling,
      n_folds = n_folds,
      val_dist_km = val_dist_km,
      val_frac = val_frac,
      max_splits = max_splits,
      seed = seed,
      train_seed = train_seed,
      cubist_committees = cubist_committees,
      cubist_neighbors = cubist_neighbors,
      results_dir = seed_dir,
      save_outputs = save_outputs
    ) %>%
      mutate(train_seed = train_seed)

    all_results[[i]] <- seed_res
  }

  final <- bind_rows(all_results)

  summary_by_seed <- final %>%
    group_by(train_seed, model) %>%
    summarise(
      RMSE_mean = mean(RMSE, na.rm = TRUE),
      R2_mean = mean(R2, na.rm = TRUE),
      MAE_mean = mean(MAE, na.rm = TRUE),
      Bias_mean = mean(Bias, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(train_seed, RMSE_mean)

  summary_overall <- final %>%
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
    write.csv(final, file.path(results_dir, "convkriging2d_anisotropic_stable_multiseed_all.csv"), row.names = FALSE)
    write.csv(summary_by_seed, file.path(results_dir, "convkriging2d_anisotropic_stable_multiseed_by_seed.csv"), row.names = FALSE)
    write.csv(summary_overall, file.path(results_dir, "convkriging2d_anisotropic_stable_multiseed_summary.csv"), row.names = FALSE)
  }

  final
}

run_convkriging2d_anisotropic_variants <- function(context = wadoux_context,
                                                   sample_size = 250,
                                                   sampling = "simple_random",
                                                   variants = make_convkriging2d_anisotropic_variants(),
                                                   n_folds = 5,
                                                   val_dist_km = 350,
                                                   val_frac = 0.2,
                                                   max_splits = 5,
                                                   seed = 123,
                                                   cubist_committees = 50,
                                                   cubist_neighbors = 5,
                                                   results_dir = "results/convkriging2d_anisotropic",
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
    cat("ANISOTROPIC VARIANT:", variant_name, "\n")
    cat("========================================\n")

    var_cfg <- variants[[variant_name]]
    conv_res <- run_convkriging2d_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      patch_size = var_cfg$patch_size,
      model_params = var_cfg$params
    )

    variant_res <- bind_rows(
      cubist_res %>% mutate(variant = variant_name),
      conv_res %>% mutate(variant = variant_name)
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
    write.csv(final, file.path(results_dir, "convkriging2d_anisotropic_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "convkriging2d_anisotropic_summary.csv"), row.names = FALSE)
  }

  final
}

run_convkriging2d_anisotropic_search <- function(context = wadoux_context,
                                                 sample_size = 250,
                                                 sampling = "simple_random",
                                                 variants = make_convkriging2d_anisotropic_search_variants(),
                                                 n_folds = 5,
                                                 val_dist_km = 350,
                                                 val_frac = 0.2,
                                                 max_splits = 5,
                                                 seed = 123,
                                                 cubist_committees = 50,
                                                 cubist_neighbors = 5,
                                                 results_dir = "results/convkriging2d_anisotropic_search",
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
    write.csv(
      benchmark$calibration_df,
      file.path(results_dir, "fixed_calibration_sample.csv"),
      row.names = FALSE
    )
    write.csv(
      manifest,
      file.path(results_dir, "fixed_split_manifest.csv"),
      row.names = FALSE
    )
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
    cat("ANISOTROPIC SEARCH VARIANT:", variant_name, "\n")
    cat("========================================\n")

    var_cfg <- variants[[variant_name]]
    conv_res <- run_convkriging2d_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      patch_size = var_cfg$patch_size,
      model_params = var_cfg$params
    )

    variant_res <- bind_rows(
      cubist_res %>% mutate(variant = variant_name),
      conv_res %>% mutate(variant = variant_name)
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
    write.csv(final, file.path(results_dir, "convkriging2d_anisotropic_search_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "convkriging2d_anisotropic_search_summary.csv"), row.names = FALSE)
  }

  final
}

run_convkriging2d_anisotropic_finalists_confirmation <- function(context = wadoux_context,
                                                                 sample_size = 300,
                                                                 sampling = "simple_random",
                                                                 variants = make_convkriging2d_anisotropic_finalists(),
                                                                 n_folds = 10,
                                                                 val_dist_km = 350,
                                                                 val_frac = 0.2,
                                                                 max_splits = 10,
                                                                 seed = 123,
                                                                 cubist_committees = 50,
                                                                 cubist_neighbors = 5,
                                                                 results_dir = "results/convkriging2d_anisotropic_finalists_confirmation",
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
    write.csv(
      benchmark$calibration_df,
      file.path(results_dir, "fixed_calibration_sample.csv"),
      row.names = FALSE
    )
    write.csv(
      manifest,
      file.path(results_dir, "fixed_split_manifest.csv"),
      row.names = FALSE
    )
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
    cat("ANISOTROPIC FINALIST:", variant_name, "\n")
    cat("========================================\n")

    var_cfg <- variants[[variant_name]]
    conv_res <- run_convkriging2d_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      patch_size = var_cfg$patch_size,
      model_params = var_cfg$params
    )

    variant_res <- bind_rows(
      cubist_res %>% mutate(variant = variant_name),
      conv_res %>% mutate(variant = variant_name)
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
    write.csv(final, file.path(results_dir, "convkriging2d_anisotropic_finalists_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "convkriging2d_anisotropic_finalists_summary.csv"), row.names = FALSE)
  }

  final
}

run_cubist_convkriging2d_hybrid_on_fixed_benchmark <- function(benchmark,
                                                               context = wadoux_context,
                                                               patch_size = 15,
                                                               model_params = modifyList(
                                                                 convkriging2d_baseline_params,
                                                                 list(
                                                                   kriging_mode = "anisotropic",
                                                                   coord_hidden = c(64, 32),
                                                                   coord_dim = 48,
                                                                   fusion_hidden = 384
                                                                 )
                                                               ),
                                                               cubist_committees = 50,
                                                               cubist_neighbors = 5) {
  results <- vector("list", length(benchmark$splits))
  neighbor_pool_k <- 24
  if (!is.null(model_params$K_neighbors)) {
    neighbor_pool_k <- max(neighbor_pool_k, model_params$K_neighbors)
  }

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[Hybrid] Cubist + ConvKrigingNet2D residual | split %s | patch=%d\n",
                sp$split_id, patch_size))

    train_sub <- sp$train_df[sp$train_idx, , drop = FALSE]
    val_sub <- sp$train_df[sp$val_idx, , drop = FALSE]
    test_df <- sp$test_df

    X_train <- as.matrix(train_sub[, context$predictors, drop = FALSE])
    X_val   <- as.matrix(val_sub[, context$predictors, drop = FALSE])
    X_test  <- as.matrix(test_df[, context$predictors, drop = FALSE])

    y_train <- train_sub[[context$response]]
    y_val   <- val_sub[[context$response]]
    y_test  <- test_df[[context$response]]

    cb_model <- Cubist::cubist(
      x = X_train,
      y = y_train,
      committees = cubist_committees,
      neighbors = cubist_neighbors
    )

    pred_train_cb <- as.numeric(predict(cb_model, X_train))
    pred_val_cb   <- as.numeric(predict(cb_model, X_val))
    pred_test_cb  <- as.numeric(predict(cb_model, X_test))

    train_resid <- train_sub
    val_resid <- val_sub
    test_resid <- test_df

    train_resid[[context$response]] <- y_train - pred_train_cb
    val_resid[[context$response]] <- y_val - pred_val_cb
    test_resid[[context$response]] <- y_test - pred_test_cb

    residual_df <- bind_rows(train_resid, val_resid, test_resid)
    n_train <- nrow(train_sub)
    n_val <- nrow(val_sub)

    fd_conv <- prepare_pointpatch_fold(
      context = context,
      calibration_df = residual_df,
      train_idx = seq_len(n_train),
      val_idx = n_train + seq_len(n_val),
      test_idx = (n_train + n_val + 1):nrow(residual_df),
      patch_size = patch_size,
      K = neighbor_pool_k
    )

    conv_out <- do.call(
      train_convkrigingnet2d_one_fold,
      c(list(fd = fd_conv), model_params)
    )

    pred_hybrid <- pred_test_cb + conv_out$pred_test

    results[[i]] <- metrics(y_test, pred_hybrid) %>%
      mutate(
        model = "CubistConvKrigingNet2D",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

make_cubist_convkriging2d_hybrid_variants <- function() {
  list(
    CubistConvKrigingNet2D_AnisoCoordFusion = list(
      patch_size = 15,
      params = modifyList(
        convkriging2d_baseline_params,
        list(
          kriging_mode = "anisotropic",
          coord_hidden = c(64, 32),
          coord_dim = 48,
          fusion_hidden = 384
        )
      )
    )
  )
}

run_cubist_convkriging2d_hybrid_fair_comparison <- function(context = wadoux_context,
                                                            sample_size = 300,
                                                            sampling = "simple_random",
                                                            variants = make_cubist_convkriging2d_hybrid_variants(),
                                                            n_folds = 10,
                                                            val_dist_km = 350,
                                                            val_frac = 0.2,
                                                            max_splits = 10,
                                                            seed = 123,
                                                            cubist_committees = 50,
                                                            cubist_neighbors = 5,
                                                            results_dir = "results/cubist_convkriging2d_hybrid",
                                                            save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING CUBIST + ConvKrigingNet2D HYBRID BENCHMARK\n")
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
    write.csv(
      benchmark$calibration_df,
      file.path(results_dir, "fixed_calibration_sample.csv"),
      row.names = FALSE
    )
    write.csv(
      manifest,
      file.path(results_dir, "fixed_split_manifest.csv"),
      row.names = FALSE
    )
  }

  cubist_res <- run_cubist_on_pointpatch_benchmark(
    benchmark = benchmark,
    context = context,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors
  )

  if (save_outputs) {
    write.csv(
      cubist_res,
      file.path(results_dir, "cubist_fixed_results.csv"),
      row.names = FALSE
    )
  }

  all_results <- list()
  for (variant_name in names(variants)) {
    cat("\n========================================\n")
    cat("HYBRID VARIANT:", variant_name, "\n")
    cat("========================================\n")

    var_cfg <- variants[[variant_name]]
    hybrid_res <- run_cubist_convkriging2d_hybrid_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      patch_size = var_cfg$patch_size,
      model_params = var_cfg$params,
      cubist_committees = cubist_committees,
      cubist_neighbors = cubist_neighbors
    )

    variant_res <- bind_rows(
      cubist_res %>% mutate(variant = variant_name),
      hybrid_res %>% mutate(variant = variant_name)
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
    write.csv(final, file.path(results_dir, "cubist_convkriging2d_hybrid_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "cubist_convkriging2d_hybrid_summary.csv"), row.names = FALSE)
  }

  final
}

run_cubist_convkriging2d_hybrid_confirmation <- function(context = wadoux_context,
                                                         sample_size = 300,
                                                         sampling = "simple_random",
                                                         n_folds = 10,
                                                         val_dist_km = 350,
                                                         val_frac = 0.2,
                                                         max_splits = 10,
                                                         seed = 123,
                                                         cubist_committees = 50,
                                                         cubist_neighbors = 5,
                                                         results_dir = "results/cubist_convkriging2d_hybrid_confirmation",
                                                         save_outputs = TRUE) {
  run_cubist_convkriging2d_hybrid_fair_comparison(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    variants = make_cubist_convkriging2d_hybrid_variants(),
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
# source("code/ConvKrigingNet2D.R")
# res_conv2d <- run_convkriging2d_wadoux_spatial_kfold(
#   context = wadoux_context,
#   sample_size = 250,
#   n_folds = 5,
#   max_splits = 5,
#   patch_size = 15,
#   model_params = convkriging2d_quick_params
# )
# summarise_comparison(res_conv2d)
# 
# Fair comparison vs Cubist:
# res_conv2d_vs_cubist <- run_convkriging2d_vs_cubist_fair(
#   context = wadoux_context,
#   sample_size = 250,
#   n_folds = 5,
#   max_splits = 5,
#   patch_size = 15,
#   model_params = convkriging2d_quick_params
# )
# res_conv2d_vs_cubist %>%
#   dplyr::group_by(model) %>%
#   dplyr::summarise(
#     RMSE_mean = mean(RMSE, na.rm = TRUE),
#     R2_mean = mean(R2, na.rm = TRUE),
#     MAE_mean = mean(MAE, na.rm = TRUE),
#     .groups = "drop"
#   )
