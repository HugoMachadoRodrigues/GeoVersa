rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)

# =============================================================================
# VAEKrigingNet2D
# - point-based prediction with raster patches
# - variational encoder for local 2D context
# - tabular + coordinates + latent patch embedding
# - residual kriging-like correction over neighbors
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

PatchVAEEncoder2D <- nn_module(
  "PatchVAEEncoder2D",
  initialize = function(in_channels, latent_dim = 64, dropout = 0.10) {
    self$stem <- nn_sequential(
      ConvBlock2D(in_channels, 32, dropout = dropout),
      nn_max_pool2d(kernel_size = 2),
      ConvBlock2D(32, 64, dropout = dropout),
      nn_max_pool2d(kernel_size = 2),
      ConvBlock2D(64, 96, dropout = dropout),
      nn_adaptive_avg_pool2d(output_size = c(1, 1)),
      nn_flatten()
    )
    self$fc_mu <- nn_linear(96, latent_dim)
    self$fc_logvar <- nn_linear(96, latent_dim)
  },
  forward = function(x_patch) {
    h <- self$stem(x_patch)
    list(mu = self$fc_mu(h), logvar = self$fc_logvar(h))
  }
)

PatchVAEDecoder2D <- nn_module(
  "PatchVAEDecoder2D",
  initialize = function(out_channels, latent_dim = 64, patch_size = 15, dropout = 0.10) {
    self$out_channels <- out_channels
    self$patch_size <- patch_size
    self$base_size <- max(4, ceiling(patch_size / 4))
    self$fc <- nn_linear(latent_dim, 96 * self$base_size * self$base_size)
    self$decoder <- nn_sequential(
      nn_conv_transpose2d(96, 64, kernel_size = 4, stride = 2, padding = 1),
      nn_gelu(),
      nn_dropout2d(dropout),
      nn_conv_transpose2d(64, 32, kernel_size = 4, stride = 2, padding = 1),
      nn_gelu(),
      nn_conv2d(32, out_channels, kernel_size = 3, padding = 1)
    )
  },
  forward = function(z) {
    h <- self$fc(z)
    h <- reshape_safe(h, c(z$size(1), 96, self$base_size, self$base_size))
    x_rec <- self$decoder(h)
    if (x_rec$size(3) != self$patch_size || x_rec$size(4) != self$patch_size) {
      x_rec <- nnf_interpolate(
        x_rec,
        size = c(self$patch_size, self$patch_size),
        mode = "bilinear",
        align_corners = FALSE
      )
    }
    x_rec
  }
)

VAEKrigingNet2D <- nn_module(
  "VAEKrigingNet2D",
  initialize = function(c_tab,
                        patch_channels,
                        patch_size = 15,
                        d = 256,
                        tab_hidden = c(192),
                        tab_dropout = 0.15,
                        latent_dim = 64,
                        patch_dropout = 0.10,
                        coord_hidden = c(32),
                        coord_dim = 32,
                        coord_dropout = 0.05,
                        fusion_hidden = 256,
                        beta_init = -4) {
    self$enc_tab <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = tab_dropout)
    self$patch_vae <- PatchVAEEncoder2D(
      in_channels = patch_channels,
      latent_dim = latent_dim,
      dropout = patch_dropout
    )
    self$patch_dec <- PatchVAEDecoder2D(
      out_channels = patch_channels,
      latent_dim = latent_dim,
      patch_size = patch_size,
      dropout = patch_dropout
    )
    self$proj_patch <- nn_linear(latent_dim, d)
    self$enc_coord <- make_mlp(2, hidden = coord_hidden, out_dim = coord_dim, dropout = coord_dropout)
    self$proj_coord <- nn_linear(coord_dim, d)

    self$fuse <- nn_sequential(
      nn_linear(3 * d, fusion_hidden),
      nn_gelu(),
      nn_dropout(0.10),
      nn_linear(fusion_hidden, d)
    )

    self$head <- ScalarHead(d = d)
    self$krig <- ResidualKrigingLayer(d = d, proj_d = 64, init_ell = 1000)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  reparameterize = function(mu, logvar, training = TRUE) {
    if (!training) return(mu)
    std <- torch_exp(0.5 * logvar)
    eps <- torch_randn_like(std)
    mu + eps * std
  },

  encode = function(x_tab, x_patch, coords, training = TRUE) {
    z_tab <- self$enc_tab(x_tab)
    vae <- self$patch_vae(x_patch)
    z_patch_latent <- self$reparameterize(vae$mu, vae$logvar, training = training)
    z_patch <- self$proj_patch(z_patch_latent)
    z_coord <- self$proj_coord(self$enc_coord(coords))
    z <- self$fuse(torch_cat(list(z_tab, z_patch, z_coord), dim = 2))
    x_rec <- self$patch_dec(z_patch_latent)
    list(
      z = z,
      mu = vae$mu,
      logvar = vae$logvar,
      z_patch_latent = z_patch_latent,
      x_rec = x_rec
    )
  },

  forward_base = function(x_tab, x_patch, coords, training = FALSE) {
    enc <- self$encode(x_tab, x_patch, coords, training = training)
    pred <- self$head(enc$z)
    list(
      pred = pred,
      z = enc$z,
      mu = enc$mu,
      logvar = enc$logvar,
      x_rec = enc$x_rec
    )
  },

  forward_with_kriging = function(x_tab, x_patch, coords, z_n, coords_n, r_n, training = FALSE) {
    base <- self$forward_base(x_tab, x_patch, coords, training = training)
    k <- self$krig(base$z, coords, z_n, coords_n, r_n)
    beta <- torch_sigmoid(self$logit_beta)
    pred_corr <- base$pred + beta * k$delta
    list(
      pred = pred_corr,
      base_pred = base$pred,
      z = base$z,
      mu = base$mu,
      logvar = base$logvar,
      x_rec = base$x_rec,
      delta = k$delta,
      beta = beta
    )
  }
)

kl_divergence_loss <- function(mu, logvar) {
  torch_mean(-0.5 * (1 + logvar - mu$pow(2) - torch_exp(logvar))$sum(dim = 2))
}

patch_reconstruction_loss <- function(x_true, x_rec) {
  torch_mean((x_true - x_rec)^2)
}

train_vaekrigingnet2d_one_fold <- function(fd,
                                           epochs = 60,
                                           lr = 2e-4,
                                           wd = 1e-3,
                                           batch_size = 96,
                                           patience = 10,
                                           d = 256,
                                           tab_hidden = c(192),
                                           tab_dropout = 0.15,
                                           latent_dim = 64,
                                           patch_dropout = 0.10,
                                           coord_hidden = c(32),
                                           coord_dim = 32,
                                           coord_dropout = 0.05,
                                           fusion_hidden = 256,
                                           beta_init = -4,
                                           target_transform = "identity",
                                           kl_weight = 0.01,
                                           recon_weight = 0.10,
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

  neigh_train <- fd$neighbor_idx_train
  if (!is.null(K_neighbors)) {
    k_eff <- min(K_neighbors, ncol(neigh_train))
    neigh_train <- neigh_train[, seq_len(k_eff), drop = FALSE]
  }

  model <- VAEKrigingNet2D(
    c_tab = ncol(Xtr),
    patch_channels = dim(Ptr)[1],
    patch_size = dim(Ptr)[2],
    d = d,
    tab_hidden = tab_hidden,
    tab_dropout = tab_dropout,
    latent_dim = latent_dim,
    patch_dropout = patch_dropout,
    coord_hidden = coord_hidden,
    coord_dim = coord_dim,
    coord_dropout = coord_dropout,
    fusion_hidden = fusion_hidden,
    beta_init = beta_init
  )
  model$to(device = device)

  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)
  best_val <- Inf
  best_state <- NULL
  bad <- 0

  for (ep in seq_len(epochs)) {
    cat(sprintf("[VAEKrigingNet2D] Building memory bank for epoch %d...\n", ep))
    bank <- build_memory_bank_pointpatch(model, Xtr, Ptr, Ctr_s, ytr_s, device = device, batch_size = batch_size)
    Zmem <- bank$Z$to(device = device)
    Rmem <- bank$R$to(device = device)
    Cmem <- bank$C$to(device = device)

    model$train()
    batches <- make_batches(nrow(Xtr), batch_size = batch_size)
    Ktr <- ncol(neigh_train)
    train_loss <- 0

    for (batch_id in seq_along(batches)) {
      b <- batches[[batch_id]]
      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      pb <- patches_to_torch(Ptr[, , , b, drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr_s[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr_s[b], device = device)

      nb <- neigh_train[b, , drop = FALSE]
      nb_t <- torch_tensor(as.vector(nb), dtype = torch_long(), device = device)
      zn <- reshape_safe(Zmem$index_select(1, nb_t), c(length(b), Ktr, -1))
      rn <- reshape_safe(Rmem$index_select(1, nb_t), c(length(b), Ktr))
      cn <- reshape_safe(Cmem$index_select(1, nb_t), c(length(b), Ktr, 2))

      out <- model$forward_with_kriging(xb, pb, cb, zn, cn, rn, training = TRUE)
      pred_loss <- huber_loss(yb, out$pred)
      kl_loss <- kl_divergence_loss(out$mu, out$logvar)
      rec_loss <- patch_reconstruction_loss(pb, out$x_rec)
      loss <- pred_loss + kl_weight * kl_loss + recon_weight * rec_loss

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[VAEKrigingNet2D] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    model$eval()
    val_pred <- predict_with_memory_pointpatch(
      model = model,
      X_new = Xva,
      P_new = Pva,
      coords_new = Cva_s,
      Zmem = Zmem,
      Rmem = Rmem,
      Cmem = Cmem,
      K = Ktr,
      device = device,
      batch_size = batch_size
    )
    vloss <- huber_loss(
      to_float_tensor(yva_s, device = device),
      to_float_tensor(val_pred, device = device)
    )$item()

    cat(sprintf("[VAEKrigingNet2D] Epoch %d complete | train_loss=%.4f | val_loss=%.4f\n",
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

  bank <- build_memory_bank_pointpatch(model, Xtr, Ptr, Ctr_s, ytr_s, device = device, batch_size = batch_size)
  preds_scaled <- predict_with_memory_pointpatch(
    model = model,
    X_new = Xte,
    P_new = Pte,
    coords_new = Cte_s,
    Zmem = bank$Z$to(device = device),
    Rmem = bank$R$to(device = device),
    Cmem = bank$C$to(device = device),
    K = ncol(neigh_train),
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

vaekriging2d_params <- list(
  epochs = 60,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 10,
  d = 256,
  tab_hidden = c(192),
  tab_dropout = 0.15,
  latent_dim = 64,
  patch_dropout = 0.10,
  coord_hidden = c(32),
  coord_dim = 32,
  coord_dropout = 0.05,
  fusion_hidden = 256,
  beta_init = -4,
  target_transform = "identity",
  kl_weight = 0.01,
  recon_weight = 0.10,
  K_neighbors = 12,
  device = "cpu"
)

vaekriging2d_quick_params <- modifyList(
  vaekriging2d_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    tab_hidden = c(128),
    latent_dim = 32,
    coord_hidden = c(16),
    coord_dim = 16,
    fusion_hidden = 128,
    K_neighbors = 8
  )
)

run_vaekrigingnet2d_on_fixed_benchmark <- function(benchmark,
                                                   context = wadoux_context,
                                                   patch_size = 15,
                                                   model_params = vaekriging2d_params) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[VAEKrigingNet2D Fair] split %s | patch=%d\n", sp$split_id, patch_size))

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

    out <- do.call(train_vaekrigingnet2d_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "VAEKrigingNet2D",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_vaekrigingnet2d_vs_cubist_fair <- function(context = wadoux_context,
                                               sample_size = 250,
                                               sampling = "simple_random",
                                               n_folds = 5,
                                               val_dist_km = 350,
                                               val_frac = 0.2,
                                               max_splits = 5,
                                               seed = 123,
                                               patch_size = 15,
                                               model_params = vaekriging2d_quick_params,
                                               cubist_committees = 50,
                                               cubist_neighbors = 5,
                                               results_dir = "results/vaekriging2d_vs_cubist",
                                               save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING VAEKrigingNet2D vs Cubist BENCHMARK\n")
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
  vae_res <- run_vaekrigingnet2d_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    patch_size = patch_size,
    model_params = model_params
  )

  final <- bind_rows(cubist_res, vae_res)
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
    write.csv(vae_res, file.path(results_dir, "vaekriging2d_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "vaekriging2d_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "vaekriging2d_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_vaekrigingnet2d_vs_cubist_confirmation <- function(context = wadoux_context,
                                                       sample_size = 300,
                                                       sampling = "simple_random",
                                                       n_folds = 10,
                                                       val_dist_km = 350,
                                                       val_frac = 0.2,
                                                       max_splits = 10,
                                                       seed = 123,
                                                       patch_size = 15,
                                                       model_params = vaekriging2d_params,
                                                       cubist_committees = 50,
                                                       cubist_neighbors = 5,
                                                       results_dir = "results/vaekriging2d_vs_cubist_confirmation",
                                                       save_outputs = TRUE) {
  run_vaekrigingnet2d_vs_cubist_fair(
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

# Example:
# source("code/VAEKrigingNet2D.R")
# res_vae <- run_vaekrigingnet2d_vs_cubist_fair(
#   context = wadoux_context,
#   sample_size = 250,
#   n_folds = 5,
#   max_splits = 5,
#   model_params = vaekriging2d_quick_params
# )
