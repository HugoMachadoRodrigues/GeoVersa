rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")

library(torch)
library(dplyr)
library(FNN)

# =============================================================================
# GeoGraphKrigingNet
# - point-based graph neural network for irregular spatial samples
# - uses a k-NN graph over sampled points
# - edge weights depend on distance and direction
# - keeps a residual kriging-like correction at the end
# =============================================================================

prepare_geograph_fold <- function(context,
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

  neighbor_idx_train <- compute_neighbor_idx_train_only(coords_train_s, K)

  list(
    X = list(train = X_train_s, val = X_val_s, test = X_test_s),
    y = list(train = y_train, val = y_val, test = y_test),
    coords = list(train = coords_train_s, val = coords_val_s, test = coords_test_s),
    x_scaler = x_scaler,
    coord_scaler = coord_scaler,
    neighbor_idx_train = neighbor_idx_train
  )
}

GeoGraphAggregator <- nn_module(
  "GeoGraphAggregator",
  initialize = function(d, edge_hidden = 64) {
    self$edge_mlp <- nn_sequential(
      nn_linear(6, edge_hidden),
      nn_gelu(),
      nn_linear(edge_hidden, 1)
    )
    self$msg_proj <- nn_linear(d, d)
    self$self_proj <- nn_linear(d, d)
  },
  forward = function(z_self, coords_self, z_n, coords_n) {
    self_x <- coords_self[, 1]$unsqueeze(2)
    self_y <- coords_self[, 2]$unsqueeze(2)
    dx <- coords_n[, , 1] - self_x
    dy <- coords_n[, , 2] - self_y
    dist <- torch_sqrt(dx^2 + dy^2 + 1e-8)
    inv_dist <- 1 / (dist + 1e-6)
    angle_cos <- dx / (dist + 1e-6)
    angle_sin <- dy / (dist + 1e-6)
    sim <- (z_n * z_self$unsqueeze(2))$sum(dim = 3)

    edge_feats <- torch_stack(
      list(dist, inv_dist, dx, dy, angle_cos, angle_sin),
      dim = 3
    )
    logits <- self$edge_mlp(edge_feats)$squeeze(3) + 0.1 * sim
    alpha <- nnf_softmax(logits, dim = 2)

    msg <- self$msg_proj(z_n)
    agg <- (alpha$unsqueeze(3) * msg)$sum(dim = 2)
    self_part <- self$self_proj(z_self)
    self_part + agg
  }
)

GeoGraphKrigingNet <- nn_module(
  "GeoGraphKrigingNet",
  initialize = function(c_tab,
                        d = 256,
                        tab_hidden = c(192),
                        tab_dropout = 0.15,
                        coord_hidden = c(64, 32),
                        coord_dim = 64,
                        coord_dropout = 0.05,
                        edge_hidden = 64,
                        fusion_hidden = 256,
                        beta_init = -4) {
    self$enc_tab <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = tab_dropout)
    self$enc_coord <- make_mlp(2, hidden = coord_hidden, out_dim = coord_dim, dropout = coord_dropout)
    self$proj_coord <- nn_linear(coord_dim, d)
    self$graph <- GeoGraphAggregator(d = d, edge_hidden = edge_hidden)
    self$fuse <- nn_sequential(
      nn_linear(2 * d, fusion_hidden),
      nn_gelu(),
      nn_dropout(0.10),
      nn_linear(fusion_hidden, d)
    )
    self$head <- ScalarHead(d = d)
    self$krig <- ResidualKrigingLayer(d = d, proj_d = 64, init_ell = 1000)
    self$logit_beta <- nn_parameter(torch_tensor(beta_init))
  },

  encode_self = function(x_tab, coords) {
    z_tab <- self$enc_tab(x_tab)
    z_coord <- self$proj_coord(self$enc_coord(coords))
    z_tab + z_coord
  },

  forward_base = function(x_tab, coords, z_n0, coords_n) {
    z0 <- self$encode_self(x_tab, coords)
    z_graph <- self$graph(z0, coords, z_n0, coords_n)
    z <- self$fuse(torch_cat(list(z0, z_graph), dim = 2))
    pred <- self$head(z)
    list(pred = pred, z = z, z0 = z0)
  },

  forward_with_kriging = function(x_tab, coords, z_n0, coords_n, z_n, r_n) {
    base <- self$forward_base(x_tab, coords, z_n0, coords_n)
    k <- self$krig(base$z, coords, z_n, coords_n, r_n)
    beta <- torch_sigmoid(self$logit_beta)
    pred_corr <- base$pred + beta * k$delta
    list(pred = pred_corr, base_pred = base$pred, z = base$z, delta = k$delta, beta = beta)
  }
)

build_geograph_self_embeddings <- function(model,
                                           X_train,
                                           coords_train,
                                           device = "cpu",
                                           batch_size = 256) {
  model$eval()
  Z0_list <- list()
  with_no_grad({
    for (s in seq(1, nrow(X_train), by = batch_size)) {
      e <- min(s + batch_size - 1, nrow(X_train))
      idx <- s:e
      xb <- to_float_tensor(X_train[idx, , drop = FALSE], device = device)
      cb <- to_float_tensor(coords_train[idx, , drop = FALSE], device = device)
      z0 <- model$encode_self(xb, cb)
      Z0_list[[length(Z0_list) + 1]] <- z0$cpu()
    }
  })
  torch_cat(Z0_list, dim = 1)
}

build_geograph_memory_bank <- function(model,
                                       X_train,
                                       coords_train,
                                       y_train,
                                       neigh_train,
                                       device = "cpu",
                                       batch_size = 256) {
  model$eval()
  Z0_all <- build_geograph_self_embeddings(
    model = model,
    X_train = X_train,
    coords_train = coords_train,
    device = device,
    batch_size = batch_size
  )

  Z_list <- list()
  R_list <- list()
  C_list <- list()

  with_no_grad({
    for (s in seq(1, nrow(X_train), by = batch_size)) {
      e <- min(s + batch_size - 1, nrow(X_train))
      idx <- s:e
      B <- length(idx)
      xb <- to_float_tensor(X_train[idx, , drop = FALSE], device = device)
      cb <- to_float_tensor(coords_train[idx, , drop = FALSE], device = device)
      yb <- to_float_tensor(y_train[idx], device = device)

      nb <- neigh_train[idx, , drop = FALSE]
      K <- ncol(nb)
      nb_t <- torch_tensor(as.vector(nb), dtype = torch_long(), device = device)
      z_n0 <- reshape_safe(Z0_all$index_select(1, nb_t), c(B, K, -1))$to(device = device)
      c_n <- reshape_safe(
        to_float_tensor(coords_train[as.vector(nb), , drop = FALSE], device = device),
        c(B, K, 2)
      )

      out <- model$forward_base(xb, cb, z_n0, c_n)
      r <- yb - out$pred

      Z_list[[length(Z_list) + 1]] <- out$z$cpu()
      R_list[[length(R_list) + 1]] <- r$cpu()
      C_list[[length(C_list) + 1]] <- cb$cpu()
    }
  })

  list(
    Z0 = Z0_all,
    Z = torch_cat(Z_list, dim = 1),
    R = torch_cat(R_list, dim = 1),
    C = torch_cat(C_list, dim = 1)
  )
}

predict_with_geograph_memory <- function(model,
                                         X_new,
                                         coords_new,
                                         coords_train,
                                         Z0_train,
                                         Zmem,
                                         Rmem,
                                         Cmem,
                                         K,
                                         device = "cpu",
                                         batch_size = 256) {
  preds <- numeric(nrow(X_new))
  coords_train_t <- to_float_tensor(coords_train, device = device)
  Z0_train <- Z0_train$to(device = device)
  Zmem <- Zmem$to(device = device)
  Rmem <- Rmem$to(device = device)
  Cmem <- Cmem$to(device = device)

  with_no_grad({
    for (s in seq(1, nrow(X_new), by = batch_size)) {
      e <- min(s + batch_size - 1, nrow(X_new))
      idx <- s:e
      B <- length(idx)

      xb <- to_float_tensor(X_new[idx, , drop = FALSE], device = device)
      cb <- to_float_tensor(coords_new[idx, , drop = FALSE], device = device)

      d <- cdist_safe(cb, coords_train_t)
      knn <- topk_smallest_idx(d, K)
      nb_flat <- flatten_safe(knn)$to(dtype = torch_long())
      z_n0 <- reshape_safe(Z0_train$index_select(1, nb_flat), c(B, K, -1))
      z_n <- reshape_safe(Zmem$index_select(1, nb_flat), c(B, K, -1))
      r_n <- reshape_safe(Rmem$index_select(1, nb_flat), c(B, K))
      c_n <- reshape_safe(Cmem$index_select(1, nb_flat), c(B, K, 2))

      out <- model$forward_with_kriging(xb, cb, z_n0, c_n, z_n, r_n)
      preds[idx] <- as.numeric(out$pred$cpu())
    }
  })

  preds
}

train_geographkrigingnet_one_fold <- function(fd,
                                              epochs = 60,
                                              lr = 2e-4,
                                              wd = 1e-3,
                                              batch_size = 96,
                                              patience = 10,
                                              d = 256,
                                              tab_hidden = c(192),
                                              tab_dropout = 0.15,
                                              coord_hidden = c(64, 32),
                                              coord_dim = 64,
                                              coord_dropout = 0.05,
                                              edge_hidden = 64,
                                              fusion_hidden = 256,
                                              beta_init = -4,
                                              target_transform = "identity",
                                              K_neighbors = 12,
                                              device = "cpu") {
  Xtr <- fd$X$train
  Xva <- fd$X$val
  Xte <- fd$X$test
  Ctr <- fd$coords$train
  Cva <- fd$coords$val
  Cte <- fd$coords$test
  ytr <- fd$y$train
  yva <- fd$y$val
  yte <- fd$y$test

  ytr_t <- transform_target(ytr, target_transform)
  yva_t <- transform_target(yva, target_transform)
  y_scaler <- fit_target_scaler(ytr_t)

  ytr_s <- apply_target_scaler(ytr_t, y_scaler)
  yva_s <- apply_target_scaler(yva_t, y_scaler)

  neigh_train <- fd$neighbor_idx_train
  if (!is.null(K_neighbors)) {
    k_eff <- min(K_neighbors, ncol(neigh_train))
    neigh_train <- neigh_train[, seq_len(k_eff), drop = FALSE]
  }

  model <- GeoGraphKrigingNet(
    c_tab = ncol(Xtr),
    d = d,
    tab_hidden = tab_hidden,
    tab_dropout = tab_dropout,
    coord_hidden = coord_hidden,
    coord_dim = coord_dim,
    coord_dropout = coord_dropout,
    edge_hidden = edge_hidden,
    fusion_hidden = fusion_hidden,
    beta_init = beta_init
  )
  model$to(device = device)

  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)
  best_val <- Inf
  best_state <- NULL
  bad <- 0

  for (ep in seq_len(epochs)) {
    cat(sprintf("[GeoGraphKrigingNet] Building graph memory for epoch %d...\n", ep))
    bank <- build_geograph_memory_bank(
      model = model,
      X_train = Xtr,
      coords_train = Ctr,
      y_train = ytr_s,
      neigh_train = neigh_train,
      device = device,
      batch_size = batch_size
    )

    model$train()
    batches <- make_batches(nrow(Xtr), batch_size = batch_size)
    Ktr <- ncol(neigh_train)
    train_loss <- 0

    for (batch_id in seq_along(batches)) {
      b <- batches[[batch_id]]
      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr_s[b], device = device)

      nb <- neigh_train[b, , drop = FALSE]
      nb_t <- torch_tensor(as.vector(nb), dtype = torch_long(), device = device)
      z_n0 <- reshape_safe(bank$Z0$index_select(1, nb_t), c(length(b), Ktr, -1))$to(device = device)
      z_n  <- reshape_safe(bank$Z$index_select(1, nb_t), c(length(b), Ktr, -1))$to(device = device)
      r_n  <- reshape_safe(bank$R$index_select(1, nb_t), c(length(b), Ktr))$to(device = device)
      c_n  <- reshape_safe(bank$C$index_select(1, nb_t), c(length(b), Ktr, 2))$to(device = device)

      out <- model$forward_with_kriging(xb, cb, z_n0, c_n, z_n, r_n)
      loss <- huber_loss(yb, out$pred)

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[GeoGraphKrigingNet] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
                    ep, batch_id, length(batches), loss$item()))
      }
    }

    model$eval()
    val_pred <- predict_with_geograph_memory(
      model = model,
      X_new = Xva,
      coords_new = Cva,
      coords_train = Ctr,
      Z0_train = bank$Z0,
      Zmem = bank$Z,
      Rmem = bank$R,
      Cmem = bank$C,
      K = Ktr,
      device = device,
      batch_size = batch_size
    )
    vloss <- huber_loss(
      to_float_tensor(yva_s, device = device),
      to_float_tensor(val_pred, device = device)
    )$item()

    cat(sprintf("[GeoGraphKrigingNet] Epoch %d complete | train_loss=%.4f | val_loss=%.4f\n",
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

  bank <- build_geograph_memory_bank(
    model = model,
    X_train = Xtr,
    coords_train = Ctr,
    y_train = ytr_s,
    neigh_train = neigh_train,
    device = device,
    batch_size = batch_size
  )
  preds_scaled <- predict_with_geograph_memory(
    model = model,
    X_new = Xte,
    coords_new = Cte,
    coords_train = Ctr,
    Z0_train = bank$Z0,
    Zmem = bank$Z,
    Rmem = bank$R,
    Cmem = bank$C,
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

geographkrigingnet_params <- list(
  epochs = 60,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 10,
  d = 256,
  tab_hidden = c(192),
  tab_dropout = 0.15,
  coord_hidden = c(64, 32),
  coord_dim = 64,
  coord_dropout = 0.05,
  edge_hidden = 64,
  fusion_hidden = 256,
  beta_init = -4,
  target_transform = "identity",
  K_neighbors = 12,
  device = "cpu"
)

geographkrigingnet_quick_params <- modifyList(
  geographkrigingnet_params,
  list(
    epochs = 20,
    batch_size = 64,
    patience = 5,
    d = 128,
    tab_hidden = c(128),
    coord_hidden = c(32, 16),
    coord_dim = 32,
    edge_hidden = 32,
    fusion_hidden = 128,
    K_neighbors = 8
  )
)

run_geographkrigingnet_on_fixed_benchmark <- function(benchmark,
                                                      context = wadoux_context,
                                                      model_params = geographkrigingnet_params) {
  results <- vector("list", length(benchmark$splits))
  k_pool <- max(24, model_params$K_neighbors)

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[GeoGraphKrigingNet Fair] split %s\n", sp$split_id))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd <- prepare_geograph_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      K = k_pool
    )

    out <- do.call(train_geographkrigingnet_one_fold, c(list(fd = fd), model_params))
    results[[i]] <- out$metrics_test %>%
      mutate(
        model = "GeoGraphKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_geographkrigingnet_vs_cubist_fair <- function(context = wadoux_context,
                                                  sample_size = 250,
                                                  sampling = "simple_random",
                                                  n_folds = 5,
                                                  val_dist_km = 350,
                                                  val_frac = 0.2,
                                                  max_splits = 5,
                                                  seed = 123,
                                                  model_params = geographkrigingnet_quick_params,
                                                  cubist_committees = 50,
                                                  cubist_neighbors = 5,
                                                  results_dir = "results/geographkrigingnet_vs_cubist",
                                                  save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING GeoGraphKrigingNet vs Cubist BENCHMARK\n")
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
  graph_res <- run_geographkrigingnet_on_fixed_benchmark(
    benchmark = benchmark,
    context = context,
    model_params = model_params
  )

  final <- bind_rows(cubist_res, graph_res)
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
    write.csv(graph_res, file.path(results_dir, "geographkrigingnet_results.csv"), row.names = FALSE)
    write.csv(final, file.path(results_dir, "geographkrigingnet_vs_cubist_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "geographkrigingnet_vs_cubist_summary.csv"), row.names = FALSE)
  }

  final
}

run_geographkrigingnet_vs_cubist_confirmation <- function(context = wadoux_context,
                                                          sample_size = 300,
                                                          sampling = "simple_random",
                                                          n_folds = 10,
                                                          val_dist_km = 350,
                                                          val_frac = 0.2,
                                                          max_splits = 10,
                                                          seed = 123,
                                                          model_params = geographkrigingnet_params,
                                                          cubist_committees = 50,
                                                          cubist_neighbors = 5,
                                                          results_dir = "results/geographkrigingnet_vs_cubist_confirmation",
                                                          save_outputs = TRUE) {
  run_geographkrigingnet_vs_cubist_fair(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    val_frac = val_frac,
    max_splits = max_splits,
    seed = seed,
    model_params = model_params,
    cubist_committees = cubist_committees,
    cubist_neighbors = cubist_neighbors,
    results_dir = results_dir,
    save_outputs = save_outputs
  )
}

# Example:
# source("code/GeoGraphKrigingNet.R")
# res_geo <- run_geographkrigingnet_vs_cubist_fair(
#   context = wadoux_context,
#   sample_size = 250,
#   n_folds = 5,
#   max_splits = 5,
#   model_params = geographkrigingnet_quick_params
# )
# res_geo %>%
#   dplyr::group_by(model) %>%
#   dplyr::summarise(
#     RMSE_mean = mean(RMSE, na.rm = TRUE),
#     R2_mean = mean(R2, na.rm = TRUE),
#     MAE_mean = mean(MAE, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   dplyr::arrange(RMSE_mean)
