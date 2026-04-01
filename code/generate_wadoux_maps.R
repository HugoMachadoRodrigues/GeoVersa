rm(list = ls())
set.seed(123)

source("./code/KrigingNet_DualFramework.R")

library(terra)
library(dplyr)
library(ggplot2)

prepare_empirical_map_data <- function(context,
                                       sample_size = 500,
                                       sampling = c("simple_random", "systematic"),
                                       val_frac = 0.2,
                                       use_robust_scaling = TRUE,
                                       K = 24) {
  sampling <- match.arg(sampling)
  calibration_df <- sample_empirical_calibration(
    context,
    sample_size = sample_size,
    sampling = sampling
  )

  n_train <- nrow(calibration_df)
  val_size <- max(1, floor(n_train * val_frac))
  val_idx <- sample(seq_len(n_train), size = val_size)
  train_idx <- setdiff(seq_len(n_train), val_idx)

  train_df <- calibration_df[train_idx, , drop = FALSE]
  val_df <- calibration_df[val_idx, , drop = FALSE]

  X_train <- as.matrix(train_df[, context$predictors, drop = FALSE])
  X_val   <- as.matrix(val_df[, context$predictors, drop = FALSE])
  y_train <- train_df[[context$response]]
  y_val   <- val_df[[context$response]]
  coords_train <- as.matrix(train_df[, c("x", "y"), drop = FALSE])
  coords_val   <- as.matrix(val_df[, c("x", "y"), drop = FALSE])

  scaler <- fit_scaler(X_train, robust = use_robust_scaling)
  X_train_s <- apply_scaler(X_train, scaler)
  X_val_s   <- apply_scaler(X_val, scaler)

  list(
    calibration_df = calibration_df,
    train_df = train_df,
    val_df = val_df,
    X = list(train = X_train_s, val = X_val_s),
    y = list(train = y_train, val = y_val),
    coords = list(train = coords_train, val = coords_val),
    scaler = scaler,
    neighbor_idx_train = compute_neighbor_idx_train_only(coords_train, K)
  )
}

fit_krigingnet_map_model <- function(fd,
                                     epochs = 200,
                                     lr = 2e-4,
                                     wd = 1e-3,
                                     batch_size = 256,
                                     patience = 20,
                                     d = 256,
                                     tab_hidden = c(256),
                                     tab_dropout = 0.15,
                                     use_tab_skip = FALSE,
                                     num_freq = 32,
                                     coord_hidden = c(32),
                                     coord_dim = 32,
                                     coord_dropout = 0.05,
                                     spatial_encoder = "raw_mlp",
                                     beta_mode = "learned",
                                     beta_scope = "global",
                                     beta_init = -4,
                                     target_transform = "identity",
                                     K_neighbors = 12,
                                     device = "cpu") {
  Xtr <- fd$X$train
  ytr <- fd$y$train
  Ctr <- fd$coords$train
  Xva <- fd$X$val
  yva <- fd$y$val
  Cva <- fd$coords$val

  ytr_t <- transform_target(ytr, target_transform)
  yva_t <- transform_target(yva, target_transform)

  y_scaler <- fit_target_scaler(ytr_t)
  coord_scaler <- fit_standard_scaler(Ctr)

  ytr_s <- apply_target_scaler(ytr_t, y_scaler)
  yva_s <- apply_target_scaler(yva_t, y_scaler)
  Ctr_s <- apply_standard_scaler(Ctr, coord_scaler)
  Cva_s <- apply_standard_scaler(Cva, coord_scaler)

  neigh_train <- fd$neighbor_idx_train
  if (!is.null(K_neighbors)) {
    k_eff <- min(K_neighbors, ncol(neigh_train))
    neigh_train <- neigh_train[, seq_len(k_eff), drop = FALSE]
  }

  model <- KrigingNet(
    c_tab = ncol(Xtr),
    d = d,
    tab_hidden = tab_hidden,
    tab_dropout = tab_dropout,
    use_tab_skip = use_tab_skip,
    num_freq = num_freq,
    coord_hidden = coord_hidden,
    coord_dim = coord_dim,
    coord_dropout = coord_dropout,
    spatial_encoder = spatial_encoder,
    beta_mode = beta_mode,
    beta_scope = beta_scope,
    beta_init = beta_init
  )
  model$to(device = device)

  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)
  best_val <- Inf
  best_state <- NULL
  bad <- 0

  for (ep in seq_len(epochs)) {
    bank <- build_memory_bank(model, Xtr, Ctr_s, ytr_s, device = device, batch_size = 1024)
    Zmem <- bank$Z$to(device = device)
    Rmem <- bank$R$to(device = device)
    Cmem <- bank$C$to(device = device)

    model$train()
    batches <- make_batches(nrow(Xtr), batch_size = batch_size)
    Ktr <- ncol(neigh_train)

    for (b in batches) {
      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr_s[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr_s[b], device = device)

      nb <- neigh_train[b, , drop = FALSE]
      nb_t <- torch_tensor(as.vector(nb), dtype = torch_long(), device = device)

      zn <- reshape_safe(Zmem$index_select(1, nb_t), c(length(b), Ktr, -1))
      rn <- reshape_safe(Rmem$index_select(1, nb_t), c(length(b), Ktr))
      cn <- reshape_safe(Cmem$index_select(1, nb_t), c(length(b), Ktr, 2))

      out <- model$forward_with_kriging(xb, cb, zn, cn, rn)
      loss <- huber_loss(yb, out$pred)

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
      opt$step()
    }

    model$eval()
    val_pred <- predict_with_memory(
      model = model,
      X_new = Xva,
      coords_new = Cva_s,
      Zmem = Zmem,
      Rmem = Rmem,
      Cmem = Cmem,
      K = Ktr,
      device = device,
      batch_size = 512
    )
    yb <- to_float_tensor(yva_s, device = device)
    pb <- to_float_tensor(val_pred, device = device)
    vloss <- huber_loss(yb, pb)$item()

    if (ep %% 10 == 0) {
      cat(sprintf("[map] Epoch %d | val_loss=%.4f\n", ep, vloss))
    }

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

  bank <- build_memory_bank(model, Xtr, Ctr_s, ytr_s, device = device, batch_size = 1024)

  list(
    model = model,
    x_scaler = fd$scaler,
    coord_scaler = coord_scaler,
    y_scaler = y_scaler,
    target_transform = target_transform,
    bank = list(
      Z = bank$Z$to(device = device),
      R = bank$R$to(device = device),
      C = bank$C$to(device = device)
    ),
    K = ncol(neigh_train),
    device = device
  )
}

predict_krigingnet_map <- function(bundle, context_df, predictor_names, batch_size = 4096) {
  X_raw <- as.matrix(context_df[, predictor_names, drop = FALSE])
  X_s <- apply_scaler(X_raw, bundle$x_scaler)
  coords_raw <- as.matrix(context_df[, c("x", "y"), drop = FALSE])
  coords_s <- apply_standard_scaler(coords_raw, bundle$coord_scaler)

  pred_scaled <- predict_with_memory(
    model = bundle$model,
    X_new = X_s,
    coords_new = coords_s,
    Zmem = bundle$bank$Z,
    Rmem = bundle$bank$R,
    Cmem = bundle$bank$C,
    K = bundle$K,
    device = bundle$device,
    batch_size = batch_size
  )

  pred_t <- invert_target_scaler(pred_scaled, bundle$y_scaler)
  inverse_transform_target(pred_t, bundle$target_transform)
}

fit_cubist_map_model <- function(fd, committees = 50, neighbors = 5) {
  model <- Cubist::cubist(
    x = fd$X$train,
    y = fd$y$train,
    committees = committees,
    neighbors = neighbors
  )
  list(model = model, x_scaler = fd$scaler)
}

predict_cubist_map <- function(bundle, context_df, predictor_names) {
  X_raw <- as.matrix(context_df[, predictor_names, drop = FALSE])
  X_s <- apply_scaler(X_raw, bundle$x_scaler)
  as.numeric(predict(bundle$model, X_s))
}

make_prediction_raster <- function(template_rast, context_df, values, name) {
  out <- template_rast
  terra::values(out) <- NA_real_
  cell_ids <- terra::cellFromXY(out, as.matrix(context_df[, c("x", "y"), drop = FALSE]))
  terra::values(out)[cell_ids] <- values
  names(out) <- name
  out
}

save_map_png <- function(df_long, out_path) {
  p <- ggplot(df_long, aes(x = x, y = y, fill = value)) +
    geom_raster() +
    scale_fill_viridis_c(option = "C", na.value = "white") +
    coord_equal() +
    facet_wrap(~ model, ncol = 2) +
    theme_minimal() +
    labs(x = NULL, y = NULL, fill = "AGB", title = "Wadoux empirical maps: Cubist vs KrigingNet")

  ggsave(out_path, p, width = 12, height = 8, dpi = 300)
}

generate_wadoux_maps <- function(context = wadoux_context,
                                 sample_size = 500,
                                 sampling = "simple_random",
                                 results_dir = "results/maps",
                                 cubist_committees = 50,
                                 cubist_neighbors = 5,
                                 krigingnet_params = krigingnet_params) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  fd <- prepare_empirical_map_data(
    context = context,
    sample_size = sample_size,
    sampling = sampling
  )

  cat("Training Cubist map model...\n")
  cubist_bundle <- fit_cubist_map_model(
    fd,
    committees = cubist_committees,
    neighbors = cubist_neighbors
  )

  cat("Training KrigingNet map model...\n")
  krigingnet_bundle <- do.call(
    fit_krigingnet_map_model,
    c(list(fd = fd), krigingnet_params)
  )

  domain_df <- context$data
  cubist_pred <- predict_cubist_map(cubist_bundle, domain_df, context$predictors)
  krigingnet_pred <- predict_krigingnet_map(krigingnet_bundle, domain_df, context$predictors)

  pred_df <- domain_df %>%
    transmute(
      x = x,
      y = y,
      observed = .data[[context$response]],
      Cubist = cubist_pred,
      KrigingNet = krigingnet_pred
    )

  template <- context$stack[[context$response]]
  rast_obs <- make_prediction_raster(template, domain_df, pred_df$observed, "Observed")
  rast_cubist <- make_prediction_raster(template, domain_df, pred_df$Cubist, "Cubist")
  rast_kn <- make_prediction_raster(template, domain_df, pred_df$KrigingNet, "KrigingNet")
  rast_diff <- make_prediction_raster(template, domain_df, pred_df$KrigingNet - pred_df$Cubist, "KrigingNet_minus_Cubist")

  terra::writeRaster(rast_obs, file.path(results_dir, "wadoux_observed.tif"), overwrite = TRUE)
  terra::writeRaster(rast_cubist, file.path(results_dir, "wadoux_cubist_map.tif"), overwrite = TRUE)
  terra::writeRaster(rast_kn, file.path(results_dir, "wadoux_krigingnet_map.tif"), overwrite = TRUE)
  terra::writeRaster(rast_diff, file.path(results_dir, "wadoux_krigingnet_minus_cubist.tif"), overwrite = TRUE)

  write.csv(pred_df, file.path(results_dir, "wadoux_map_predictions.csv"), row.names = FALSE)

  plot_df <- bind_rows(
    pred_df %>% transmute(x, y, model = "Observed", value = observed),
    pred_df %>% transmute(x, y, model = "Cubist", value = Cubist),
    pred_df %>% transmute(x, y, model = "KrigingNet", value = KrigingNet),
    pred_df %>% transmute(x, y, model = "KrigingNet - Cubist", value = KrigingNet - Cubist)
  )

  save_map_png(plot_df, file.path(results_dir, "wadoux_map_comparison.png"))

  list(
    predictions = pred_df,
    rasters = list(
      observed = rast_obs,
      cubist = rast_cubist,
      krigingnet = rast_kn,
      difference = rast_diff
    )
  )
}

# Example:
# maps <- generate_wadoux_maps()
