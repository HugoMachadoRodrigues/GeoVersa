rm(list = ls())
set.seed(123)

source("./code/KrigingNet_DualFramework.R")

library(torch)
library(terra)
library(dplyr)

.pointpatch_raster_array_cache <- new.env(parent = emptyenv())

# =============================================================================
# PointPatchKrigingNet
# - keeps the "point prediction" logic
# - uses raster patches around each point as local spatial context
# - combines tabular covariates + patch CNN embedding + residual kriging layer
# =============================================================================

extract_single_patch <- function(r_stack,
                                 x,
                                 y,
                                 patch_size = 15,
                                 fill_value = NA_real_) {
  if (patch_size %% 2 != 1) stop("patch_size must be odd.")

  center_cell <- terra::cellFromXY(r_stack[[1]], matrix(c(x, y), ncol = 2))
  if (is.na(center_cell)) {
    n_channels <- terra::nlyr(r_stack)
    return(array(fill_value, dim = c(n_channels, patch_size, patch_size)))
  }

  rc <- terra::rowColFromCell(r_stack[[1]], center_cell)
  half <- floor(patch_size / 2)
  rows <- seq.int(rc[1] - half, rc[1] + half)
  cols <- seq.int(rc[2] - half, rc[2] + half)

  patch <- array(fill_value, dim = c(terra::nlyr(r_stack), patch_size, patch_size))

  valid_rows <- rows >= 1 & rows <= terra::nrow(r_stack)
  valid_cols <- cols >= 1 & cols <= terra::ncol(r_stack)
  if (!any(valid_rows) || !any(valid_cols)) return(patch)

  for (i in seq_along(rows)) {
    if (!valid_rows[i]) next
    for (j in seq_along(cols)) {
      if (!valid_cols[j]) next
      cell_id <- terra::cellFromRowCol(r_stack[[1]], rows[i], cols[j])
      vals <- terra::extract(r_stack, cell_id)
      if ("ID" %in% names(vals)) {
        vals <- vals[, setdiff(names(vals), "ID"), drop = FALSE]
      }
      patch[, i, j] <- as.numeric(vals[1, ])
    }
  }
  patch
}

make_pointpatch_raster_cache_key <- function(r_stack) {
  paste(
    terra::nrow(r_stack),
    terra::ncol(r_stack),
    terra::nlyr(r_stack),
    paste(names(r_stack), collapse = "|"),
    sep = "::"
  )
}

get_pointpatch_raster_array_cache <- function(r_stack) {
  key <- make_pointpatch_raster_cache_key(r_stack)
  if (!exists(key, envir = .pointpatch_raster_array_cache, inherits = FALSE)) {
    cat(sprintf("Building in-memory raster array cache for %d layers...\n", terra::nlyr(r_stack)))
    arr <- terra::as.array(r_stack)
    assign(
      key,
      list(
        array = arr,
        nrow = terra::nrow(r_stack),
        ncol = terra::ncol(r_stack),
        nlyr = terra::nlyr(r_stack)
      ),
      envir = .pointpatch_raster_array_cache
    )
  }
  get(key, envir = .pointpatch_raster_array_cache, inherits = FALSE)
}

extract_point_patches <- function(r_stack,
                                  coords,
                                  patch_size = 15,
                                  fill_value = NA_real_,
                                  label = "patches") {
  cat(sprintf("Extracting %s for %d points (patch_size=%d)...\n",
              label, nrow(coords), patch_size))

  cache <- get_pointpatch_raster_array_cache(r_stack)
  cells <- terra::cellFromXY(r_stack[[1]], coords)
  rowcol <- matrix(NA_integer_, nrow = nrow(coords), ncol = 2)
  valid_cells <- !is.na(cells)
  if (any(valid_cells)) {
    rowcol[valid_cells, ] <- terra::rowColFromCell(r_stack[[1]], cells[valid_cells])
  }

  half <- floor(patch_size / 2)
  patches <- lapply(seq_len(nrow(coords)), function(i) {
    if (i %% 50 == 0 || i == nrow(coords)) {
      cat(sprintf("  %s: %d/%d\n", label, i, nrow(coords)))
    }

    if (is.na(rowcol[i, 1]) || is.na(rowcol[i, 2])) {
      return(array(fill_value, dim = c(cache$nlyr, patch_size, patch_size)))
    }

    rows <- seq.int(rowcol[i, 1] - half, rowcol[i, 1] + half)
    cols <- seq.int(rowcol[i, 2] - half, rowcol[i, 2] + half)
    patch <- array(fill_value, dim = c(cache$nlyr, patch_size, patch_size))

    valid_rows <- rows >= 1 & rows <= cache$nrow
    valid_cols <- cols >= 1 & cols <= cache$ncol
    if (!any(valid_rows) || !any(valid_cols)) {
      return(patch)
    }

    sub_arr <- cache$array[rows[valid_rows], cols[valid_cols], , drop = FALSE]
    patch[, valid_rows, valid_cols] <- aperm(sub_arr, c(3, 1, 2))
    patch
  })
  simplify2array(patches)
}

normalize_patch_array <- function(patches, eps = 1e-6) {
  dims <- dim(patches)
  if (length(dims) != 4) stop("Expected patch array with dims c(C, H, W, N).")

  means <- apply(patches, 1, mean, na.rm = TRUE)
  sds <- apply(patches, 1, sd, na.rm = TRUE)
  sds[is.na(sds) | sds < eps] <- 1

  out <- patches
  for (ch in seq_len(dims[1])) {
    out[ch, , , ] <- (patches[ch, , , ] - means[ch]) / sds[ch]
  }

  out[is.na(out)] <- 0
  list(patches = out, mean = means, sd = sds)
}

apply_patch_scaler <- function(patches, scaler) {
  dims <- dim(patches)
  out <- patches
  for (ch in seq_len(dims[1])) {
    out[ch, , , ] <- (patches[ch, , , ] - scaler$mean[ch]) / scaler$sd[ch]
  }
  out[is.na(out)] <- 0
  out
}

patches_to_torch <- function(patches, device = "cpu") {
  dims <- dim(patches)
  arr <- aperm(patches, c(4, 1, 2, 3))
  torch_tensor(arr, dtype = torch_float(), device = device)
}

build_pointpatch_patch_cache <- function(context,
                                         calibration_df,
                                         patch_size = 15,
                                         fill_value = NA_real_,
                                         label = "calibration patches") {
  coords <- as.matrix(calibration_df[, c("x", "y"), drop = FALSE])
  patch_layers <- context$stack[[context$predictors]]
  patches_raw <- extract_point_patches(
    patch_layers,
    coords,
    patch_size = patch_size,
    fill_value = fill_value,
    label = label
  )

  list(
    patches_raw = patches_raw,
    patch_size = patch_size,
    n = nrow(calibration_df)
  )
}

prepare_pointpatch_fold <- function(context,
                                    calibration_df,
                                    train_idx,
                                    val_idx,
                                    test_idx,
                                    patch_size = 15,
                                    use_robust_scaling = TRUE,
                                    K = 24,
                                    patch_cache = NULL) {
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
  X_train_s <- apply_scaler(X_train, x_scaler)
  X_val_s   <- apply_scaler(X_val, x_scaler)
  X_test_s  <- apply_scaler(X_test, x_scaler)

  if (!is.null(patch_cache)) {
    if (is.null(patch_cache$patches_raw) || is.null(patch_cache$n) || is.null(patch_cache$patch_size)) {
      stop("patch_cache must contain patches_raw, n, and patch_size.")
    }
    if (!identical(as.integer(patch_cache$n), as.integer(nrow(calibration_df)))) {
      stop("patch_cache$n does not match nrow(calibration_df).")
    }
    if (!identical(as.integer(patch_cache$patch_size), as.integer(patch_size))) {
      stop("patch_cache$patch_size does not match patch_size.")
    }

    p_train <- patch_cache$patches_raw[, , , train_idx, drop = FALSE]
    p_val   <- patch_cache$patches_raw[, , , val_idx, drop = FALSE]
    p_test  <- patch_cache$patches_raw[, , , test_idx, drop = FALSE]
  } else {
    patch_layers <- context$stack[[context$predictors]]
    p_train <- extract_point_patches(patch_layers, coords_train, patch_size = patch_size, label = "train patches")
    p_val   <- extract_point_patches(patch_layers, coords_val, patch_size = patch_size, label = "val patches")
    p_test  <- extract_point_patches(patch_layers, coords_test, patch_size = patch_size, label = "test patches")
  }

  cat("Normalizing patch channels...\n")

  patch_scaler <- normalize_patch_array(p_train)

  list(
    X = list(train = X_train_s, val = X_val_s, test = X_test_s),
    y = list(train = y_train, val = y_val, test = y_test),
    coords = list(train = coords_train, val = coords_val, test = coords_test),
    patches = list(
      train = patch_scaler$patches,
      val = apply_patch_scaler(p_val, patch_scaler),
      test = apply_patch_scaler(p_test, patch_scaler)
    ),
    x_scaler = x_scaler,
    patch_scaler = patch_scaler,
    neighbor_idx_train = compute_neighbor_idx_train_only(coords_train, K)
  )
}

PatchCNNEncoder <- nn_module(
  "PatchCNNEncoder",
  initialize = function(in_channels, out_dim = 128, dropout = 0.10) {
    self$conv <- nn_sequential(
      nn_conv2d(in_channels, 32, kernel_size = 3, padding = 1),
      nn_gelu(),
      nn_max_pool2d(kernel_size = 2),
      nn_conv2d(32, 64, kernel_size = 3, padding = 1),
      nn_gelu(),
      nn_max_pool2d(kernel_size = 2),
      nn_adaptive_avg_pool2d(output_size = c(1, 1))
    )
    self$head <- nn_sequential(
      nn_flatten(),
      nn_linear(64, out_dim),
      nn_gelu(),
      nn_dropout(dropout)
    )
  },
  forward = function(x_patch) {
    self$head(self$conv(x_patch))
  }
)

PointPatchKrigingNet <- nn_module(
  "PointPatchKrigingNet",
  initialize = function(c_tab,
                        patch_channels,
                        d = 256,
                        tab_hidden = c(256),
                        tab_dropout = 0.15,
                        patch_dim = 128,
                        patch_dropout = 0.10,
                        coord_hidden = c(32),
                        coord_dim = 32,
                        coord_dropout = 0.05,
                        beta_init = -4) {
    self$enc_tab <- make_mlp(c_tab, hidden = tab_hidden, out_dim = d, dropout = tab_dropout)
    self$enc_patch <- PatchCNNEncoder(
      in_channels = patch_channels,
      out_dim = patch_dim,
      dropout = patch_dropout
    )
    self$proj_patch <- nn_linear(patch_dim, d)
    self$enc_coord <- make_mlp(2, hidden = coord_hidden, out_dim = coord_dim, dropout = coord_dropout)
    self$proj_coord <- nn_linear(coord_dim, d)

    self$fuse <- nn_sequential(
      nn_linear(3 * d, 256), nn_gelu(),
      nn_linear(256, d)
    )

    self$head <- ScalarHead(d = d)
    self$krig <- ResidualKrigingLayer(d = d, proj_d = 64, init_ell = 1000)
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

build_memory_bank_pointpatch <- function(model,
                                         X_train,
                                         P_train,
                                         coords_train,
                                         y_train,
                                         device = "cpu",
                                         batch_size = 256) {
  model$eval()
  n <- nrow(X_train)

  Z_list <- list()
  R_list <- list()
  C_list <- list()

  with_no_grad({
    for (s in seq(1, n, by = batch_size)) {
      e <- min(s + batch_size - 1, n)
      idx <- s:e

      xb <- to_float_tensor(X_train[idx, , drop = FALSE], device = device)
      pb <- patches_to_torch(P_train[, , , idx, drop = FALSE], device = device)
      cb <- to_float_tensor(coords_train[idx, , drop = FALSE], device = device)
      yb <- to_float_tensor(y_train[idx], device = device)

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

predict_with_memory_pointpatch <- function(model,
                                           X_new,
                                           P_new,
                                           coords_new,
                                           Zmem,
                                           Rmem,
                                           Cmem,
                                           K,
                                           device = "cpu",
                                           batch_size = 256) {
  preds <- numeric(nrow(X_new))

  with_no_grad({
    for (s in seq(1, nrow(X_new), by = batch_size)) {
      e <- min(s + batch_size - 1, nrow(X_new))
      idx <- s:e
      B <- length(idx)

      xb <- to_float_tensor(X_new[idx, , drop = FALSE], device = device)
      pb <- patches_to_torch(P_new[, , , idx, drop = FALSE], device = device)
      cb <- to_float_tensor(coords_new[idx, , drop = FALSE], device = device)

      d <- cdist_safe(cb, Cmem)
      knn <- topk_smallest_idx(d, K)
      nb_flat <- flatten_safe(knn)$to(dtype = torch_long())

      zn <- reshape_safe(Zmem$index_select(1, nb_flat), c(B, K, -1))
      rn <- reshape_safe(Rmem$index_select(1, nb_flat), c(B, K))
      cn <- reshape_safe(Cmem$index_select(1, nb_flat), c(B, K, 2))

      out <- model$forward_with_kriging(xb, pb, cb, zn, cn, rn)
      preds[idx] <- as.numeric(out$pred$cpu())
    }
  })

  preds
}

train_pointpatch_krigingnet_one_fold <- function(fd,
                                                 epochs = 100,
                                                 lr = 2e-4,
                                                 wd = 1e-3,
                                                 batch_size = 128,
                                                 patience = 15,
                                                 d = 256,
                                                 tab_hidden = c(256),
                                                 tab_dropout = 0.15,
                                                 patch_dim = 128,
                                                 patch_dropout = 0.10,
                                                 coord_hidden = c(32),
                                                 coord_dim = 32,
                                                 coord_dropout = 0.05,
                                                 beta_init = -4,
                                                 target_transform = "identity",
                                                 K_neighbors = NULL,
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

  model <- PointPatchKrigingNet(
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
    beta_init = beta_init
  )
  model$to(device = device)

  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)
  best_val <- Inf
  best_state <- NULL
  bad <- 0

  for (ep in seq_len(epochs)) {
    cat(sprintf("[PointPatch] Building memory bank for epoch %d...\n", ep))
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

      out <- model$forward_with_kriging(xb, pb, cb, zn, cn, rn)
      loss <- huber_loss(yb, out$pred)

      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
      opt$step()

      train_loss <- train_loss + loss$item()

      if (batch_id %% 10 == 0 || batch_id == length(batches)) {
        cat(sprintf("[PointPatch] Epoch %d | batch %d/%d | batch_loss=%.4f\n",
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

    cat(sprintf("[PointPatch] Epoch %d complete | train_loss=%.4f | val_loss=%.4f\n",
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

run_pointpatch_wadoux_spatial_kfold <- function(context = wadoux_context,
                                                sample_size = 500,
                                                n_folds = 10,
                                                val_dist_km = 350,
                                                patch_size = 15,
                                                pointpatch_params = list(),
                                                cubist_committees = 50,
                                                cubist_neighbors = 5,
                                                krigingnet_params_local = krigingnet_params,
                                                max_splits = NULL,
                                                include_krigingnet = TRUE) {
  calibration_df <- sample_empirical_calibration(
    context,
    sample_size = sample_size,
    sampling = "simple_random"
  )

  splits <- build_empirical_protocol_splits(
    calibration_df = calibration_df,
    context = context,
    protocol = "spatial_kfold",
    n_folds = n_folds,
    val_dist_km = val_dist_km
  )

  if (!is.null(max_splits) && max_splits < length(splits)) {
    splits <- splits[seq_len(max_splits)]
  }

  results <- vector("list", length(splits))

  for (i in seq_along(splits)) {
    sp <- splits[[i]]
    cat(sprintf("\nPreparing split %s...\n", sp$split_id))
    train_n <- nrow(sp$train)
    val_size <- max(1, floor(train_n * 0.2))
    val_idx <- sample(seq_len(train_n), size = val_size)
    train_idx <- setdiff(seq_len(train_n), val_idx)
    test_idx <- seq_len(nrow(sp$test))

    train_val_df <- bind_rows(sp$train, sp$test)
    train_rows <- seq_len(nrow(sp$train))
    test_rows <- (nrow(sp$train) + 1):nrow(train_val_df)

    cat("Preparing PointPatch fold data...\n")
    fd_pp <- prepare_pointpatch_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[train_idx],
      val_idx = train_rows[val_idx],
      test_idx = test_rows,
      patch_size = patch_size,
      K = 24
    )

    cat("Preparing baseline/KrigingNet fold data...\n")
    fd_kn <- prepare_empirical_split(
      train_df = sp$train,
      test_df = sp$test,
      predictor_names = context$predictors,
      response_name = context$response,
      use_robust_scaling = TRUE,
      K = 24
    )
    fd_kn$protocol <- "spatial_kfold"
    fd_kn$split_id <- sp$split_id

    cat("\n=============================\n")
    cat("POINT PATCH WADOUX | spatial_kfold | SPLIT", sp$split_id, "\n")
    cat("=============================\n")

    cat("Fitting Cubist...\n")
    pred_cb <- fit_predict_cubist(
      fd_kn$X$train,
      fd_kn$y$train,
      fd_kn$X$test,
      committees = cubist_committees,
      neighbors = cubist_neighbors
    )
    met_cb <- metrics(fd_kn$y$test, pred_cb) %>% mutate(model = "Cubist")

    cat("Fitting PointPatchKrigingNet...\n")
    pp_out <- do.call(train_pointpatch_krigingnet_one_fold, c(list(fd = fd_pp), pointpatch_params))
    met_pp <- pp_out$metrics_test %>% mutate(model = "PointPatchKrigingNet")

    split_res <- bind_rows(met_cb, met_pp)
    if (include_krigingnet) {
      cat("Fitting KrigingNet...\n")
      kn_out <- do.call(train_krigingnet_one_fold, c(list(fd = fd_kn), krigingnet_params_local))
      met_kn <- kn_out$metrics_test %>% mutate(model = "KrigingNet")
      split_res <- bind_rows(split_res, met_kn)
    }

    results[[i]] <- split_res %>%
      mutate(protocol = "spatial_kfold", split = sp$split_id)
  }

  bind_rows(results)
}

pointpatch_params <- list(
  epochs = 80,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 128,
  patience = 12,
  d = 256,
  tab_hidden = c(256),
  tab_dropout = 0.15,
  patch_dim = 128,
  patch_dropout = 0.10,
  coord_hidden = c(32),
  coord_dim = 32,
  coord_dropout = 0.05,
  beta_init = -4,
  target_transform = "identity",
  K_neighbors = 12,
  device = "cpu"
)

pointpatch_quick_params <- list(
  epochs = 20,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 64,
  patience = 5,
  d = 128,
  tab_hidden = c(128),
  tab_dropout = 0.15,
  patch_dim = 64,
  patch_dropout = 0.10,
  coord_hidden = c(16),
  coord_dim = 16,
  coord_dropout = 0.05,
  beta_init = -4,
  target_transform = "identity",
  K_neighbors = 8,
  device = "cpu"
)

pointpatch_tuned_params <- list(
  epochs = 40,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 96,
  patience = 8,
  d = 192,
  tab_hidden = c(192),
  tab_dropout = 0.15,
  patch_dim = 96,
  patch_dropout = 0.10,
  coord_hidden = c(24),
  coord_dim = 24,
  coord_dropout = 0.05,
  beta_init = -4,
  target_transform = "identity",
  K_neighbors = 10,
  device = "cpu"
)

make_pointpatch_variants <- function() {
  list(
    PointPatch_Quick = list(
      patch_size = 9,
      params = pointpatch_quick_params
    ),
    PointPatch_Tuned_P15 = list(
      patch_size = 15,
      params = pointpatch_tuned_params
    ),
    PointPatch_Tuned_P21 = list(
      patch_size = 21,
      params = pointpatch_tuned_params
    ),
    PointPatch_Tuned_P15_Wide = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          d = 256,
          tab_hidden = c(256),
          patch_dim = 128,
          coord_hidden = c(32),
          coord_dim = 32,
          K_neighbors = 12
        )
      )
    ),
    PointPatch_Tuned_P21_Wide = list(
      patch_size = 21,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          d = 256,
          tab_hidden = c(256),
          patch_dim = 128,
          coord_hidden = c(32),
          coord_dim = 32,
          K_neighbors = 12
        )
      )
    )
  )
}

make_pointpatch_p15_variants <- function() {
  list(
    PointPatch_P15_Base = list(
      patch_size = 15,
      params = pointpatch_tuned_params
    ),
    PointPatch_P15_MoreEpochs = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          epochs = 60,
          patience = 12
        )
      )
    ),
    PointPatch_P15_PatchDim128 = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          patch_dim = 128
        )
      )
    ),
    PointPatch_P15_K8 = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          K_neighbors = 8
        )
      )
    ),
    PointPatch_P15_K12 = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          K_neighbors = 12
        )
      )
    )
  )
}

run_pointpatch_wadoux_spatial_kfold_variants <- function(context = wadoux_context,
                                                         sample_size = 250,
                                                         variants = make_pointpatch_variants(),
                                                         n_folds = 10,
                                                         val_dist_km = 350,
                                                         cubist_committees = 50,
                                                         cubist_neighbors = 5,
                                                         krigingnet_params_local = krigingnet_params,
                                                         max_splits = NULL,
                                                         results_dir = "results/pointpatch_variants",
                                                         save_partial = TRUE,
                                                         include_krigingnet = TRUE) {
  all_results <- list()
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  for (variant_name in names(variants)) {
    cat("\n========================================\n")
    cat("POINTPATCH VARIANT:", variant_name, "\n")
    cat("========================================\n")

    var_cfg <- variants[[variant_name]]
    res <- run_pointpatch_wadoux_spatial_kfold(
      context = context,
      sample_size = sample_size,
      n_folds = n_folds,
      val_dist_km = val_dist_km,
      patch_size = var_cfg$patch_size,
      pointpatch_params = var_cfg$params,
      cubist_committees = cubist_committees,
      cubist_neighbors = cubist_neighbors,
      krigingnet_params_local = krigingnet_params_local,
      max_splits = max_splits,
      include_krigingnet = include_krigingnet
    ) %>%
      mutate(variant = variant_name)

    all_results[[variant_name]] <- res

    if (save_partial) {
      out_csv <- file.path(results_dir, paste0("partial_", variant_name, ".csv"))
      write.csv(res, out_csv, row.names = FALSE)
      cat(sprintf("Saved partial results: %s\n", out_csv))
    }
  }

  final <- bind_rows(all_results)
  if (save_partial) {
    write.csv(final, file.path(results_dir, "pointpatch_variants_all.csv"), row.names = FALSE)
  }
  final
}

pointpatch_benchmark_manifest <- function(benchmark) {
  bind_rows(lapply(benchmark$splits, function(sp) {
    data.frame(
      split = sp$split_id,
      train_n = nrow(sp$train_df),
      subtrain_n = length(sp$train_idx),
      val_n = length(sp$val_idx),
      test_n = nrow(sp$test_df),
      stringsAsFactors = FALSE
    )
  }))
}

build_pointpatch_fixed_spatial_kfold_benchmark <- function(context = wadoux_context,
                                                           sample_size = 150,
                                                           sampling = "simple_random",
                                                           n_folds = 5,
                                                           val_dist_km = 350,
                                                           val_frac = 0.2,
                                                           max_splits = NULL,
                                                           seed = 123) {
  set.seed(seed)
  calibration_df <- sample_empirical_calibration(
    context,
    sample_size = sample_size,
    sampling = sampling
  )

  splits <- build_empirical_protocol_splits(
    calibration_df = calibration_df,
    context = context,
    protocol = "spatial_kfold",
    n_folds = n_folds,
    val_dist_km = val_dist_km
  )

  if (!is.null(max_splits) && max_splits < length(splits)) {
    splits <- splits[seq_len(max_splits)]
  }

  split_plan <- lapply(seq_along(splits), function(i) {
    sp <- splits[[i]]
    train_n <- nrow(sp$train)
    val_size <- max(1, floor(train_n * val_frac))

    set.seed(seed + i)
    val_idx <- sample(seq_len(train_n), size = val_size)
    train_idx <- setdiff(seq_len(train_n), val_idx)

    list(
      split_id = sp$split_id,
      train_df = sp$train,
      test_df = sp$test,
      train_idx = train_idx,
      val_idx = val_idx
    )
  })

  structure(
    list(
      calibration_df = calibration_df,
      splits = split_plan,
      meta = list(
        sample_size = sample_size,
        sampling = sampling,
        n_folds = n_folds,
        val_dist_km = val_dist_km,
        val_frac = val_frac,
        max_splits = max_splits,
        seed = seed
      )
    ),
    class = "pointpatch_fixed_benchmark"
  )
}

run_cubist_on_pointpatch_benchmark <- function(benchmark,
                                               context = wadoux_context,
                                               cubist_committees = 50,
                                               cubist_neighbors = 5) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[Fair Comparison] Cubist | split %s\n", sp$split_id))

    subtrain_df <- sp$train_df[sp$train_idx, , drop = FALSE]

    fd_kn <- prepare_empirical_split(
      train_df = subtrain_df,
      test_df = sp$test_df,
      predictor_names = context$predictors,
      response_name = context$response,
      use_robust_scaling = TRUE,
      K = 24
    )

    pred_cb <- fit_predict_cubist(
      fd_kn$X$train,
      fd_kn$y$train,
      fd_kn$X$test,
      committees = cubist_committees,
      neighbors = cubist_neighbors
    )

    results[[i]] <- metrics(fd_kn$y$test, pred_cb) %>%
      mutate(
        model = "Cubist",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

run_pointpatch_on_fixed_benchmark <- function(benchmark,
                                              context = wadoux_context,
                                              patch_size = 15,
                                              pointpatch_params = pointpatch_tuned_params) {
  results <- vector("list", length(benchmark$splits))
  neighbor_pool_k <- 24
  if (!is.null(pointpatch_params$K_neighbors)) {
    neighbor_pool_k <- max(neighbor_pool_k, pointpatch_params$K_neighbors)
  }

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[Fair Comparison] PointPatch | split %s | patch=%d\n",
                sp$split_id, patch_size))

    train_val_df <- bind_rows(sp$train_df, sp$test_df)
    train_rows <- seq_len(nrow(sp$train_df))
    test_rows <- (nrow(sp$train_df) + 1):nrow(train_val_df)

    fd_pp <- prepare_pointpatch_fold(
      context = context,
      calibration_df = train_val_df,
      train_idx = train_rows[sp$train_idx],
      val_idx = train_rows[sp$val_idx],
      test_idx = test_rows,
      patch_size = patch_size,
      K = neighbor_pool_k
    )

    pp_out <- do.call(
      train_pointpatch_krigingnet_one_fold,
      c(list(fd = fd_pp), pointpatch_params)
    )

    results[[i]] <- pp_out$metrics_test %>%
      mutate(
        model = "PointPatchKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

make_pointpatch_fair_variants <- function() {
  list(
    PointPatch_P15_MoreEpochs = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          epochs = 60,
          patience = 12
        )
      )
    ),
    PointPatch_P15_PatchDim128 = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          patch_dim = 128
        )
      )
    )
  )
}

make_pointpatch_confirmation_variants <- function() {
  list(
    PointPatch_P15_MoreEpochs = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          epochs = 60,
          patience = 12
        )
      )
    )
  )
}

run_pointpatch_fair_comparison <- function(context = wadoux_context,
                                           sample_size = 150,
                                           sampling = "simple_random",
                                           variants = make_pointpatch_fair_variants(),
                                           n_folds = 5,
                                           val_dist_km = 350,
                                           val_frac = 0.2,
                                           max_splits = 5,
                                           seed = 123,
                                           cubist_committees = 50,
                                           cubist_neighbors = 5,
                                           results_dir = "results/pointpatch_fair_comparison",
                                           save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING FAIR COMPARISON BENCHMARK\n")
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
    cat("FAIR VARIANT:", variant_name, "\n")
    cat("========================================\n")

    var_cfg <- variants[[variant_name]]
    pp_res <- run_pointpatch_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      patch_size = var_cfg$patch_size,
      pointpatch_params = var_cfg$params
    )

    variant_res <- bind_rows(
      cubist_res %>% mutate(variant = variant_name),
      pp_res %>% mutate(variant = variant_name)
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
    write.csv(final, file.path(results_dir, "pointpatch_fair_comparison_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "pointpatch_fair_comparison_summary.csv"), row.names = FALSE)
  }

  final
}

run_pointpatch_confirmation <- function(context = wadoux_context,
                                        sample_size = 300,
                                        sampling = "simple_random",
                                        n_folds = 10,
                                        val_dist_km = 350,
                                        val_frac = 0.2,
                                        max_splits = 10,
                                        seed = 123,
                                        cubist_committees = 50,
                                        cubist_neighbors = 5,
                                        results_dir = "results/pointpatch_confirmation",
                                        save_outputs = TRUE) {
  run_pointpatch_fair_comparison(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    variants = make_pointpatch_confirmation_variants(),
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

run_cubist_pointpatch_hybrid_on_fixed_benchmark <- function(benchmark,
                                                            context = wadoux_context,
                                                            patch_size = 15,
                                                            pointpatch_params = pointpatch_tuned_params,
                                                            cubist_committees = 50,
                                                            cubist_neighbors = 5) {
  results <- vector("list", length(benchmark$splits))
  neighbor_pool_k <- 24
  if (!is.null(pointpatch_params$K_neighbors)) {
    neighbor_pool_k <- max(neighbor_pool_k, pointpatch_params$K_neighbors)
  }

  for (i in seq_along(benchmark$splits)) {
    sp <- benchmark$splits[[i]]
    cat(sprintf("\n[Hybrid] Cubist + PointPatch residual | split %s | patch=%d\n",
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

    fd_pp <- prepare_pointpatch_fold(
      context = context,
      calibration_df = residual_df,
      train_idx = seq_len(n_train),
      val_idx = n_train + seq_len(n_val),
      test_idx = (n_train + n_val + 1):nrow(residual_df),
      patch_size = patch_size,
      K = neighbor_pool_k
    )

    pp_out <- do.call(
      train_pointpatch_krigingnet_one_fold,
      c(list(fd = fd_pp), pointpatch_params)
    )

    pred_hybrid <- pred_test_cb + pp_out$pred_test

    results[[i]] <- metrics(y_test, pred_hybrid) %>%
      mutate(
        model = "CubistPointPatchKrigingNet",
        protocol = "spatial_kfold",
        split = sp$split_id
      )
  }

  bind_rows(results)
}

make_cubist_pointpatch_hybrid_variants <- function() {
  list(
    CubistPointPatch_P15_MoreEpochs = list(
      patch_size = 15,
      params = modifyList(
        pointpatch_tuned_params,
        list(
          epochs = 60,
          patience = 12
        )
      )
    )
  )
}

run_cubist_pointpatch_hybrid_fair_comparison <- function(context = wadoux_context,
                                                         sample_size = 300,
                                                         sampling = "simple_random",
                                                         variants = make_cubist_pointpatch_hybrid_variants(),
                                                         n_folds = 10,
                                                         val_dist_km = 350,
                                                         val_frac = 0.2,
                                                         max_splits = 10,
                                                         seed = 123,
                                                         cubist_committees = 50,
                                                         cubist_neighbors = 5,
                                                         results_dir = "results/cubist_pointpatch_hybrid",
                                                         save_outputs = TRUE) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  cat("\n========================================\n")
  cat("BUILDING HYBRID FAIR COMPARISON BENCHMARK\n")
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
    hybrid_res <- run_cubist_pointpatch_hybrid_on_fixed_benchmark(
      benchmark = benchmark,
      context = context,
      patch_size = var_cfg$patch_size,
      pointpatch_params = var_cfg$params,
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
    write.csv(final, file.path(results_dir, "cubist_pointpatch_hybrid_all.csv"), row.names = FALSE)
    write.csv(summary_tbl, file.path(results_dir, "cubist_pointpatch_hybrid_summary.csv"), row.names = FALSE)
  }

  final
}

run_cubist_pointpatch_hybrid_confirmation <- function(context = wadoux_context,
                                                      sample_size = 300,
                                                      sampling = "simple_random",
                                                      n_folds = 10,
                                                      val_dist_km = 350,
                                                      val_frac = 0.2,
                                                      max_splits = 10,
                                                      seed = 123,
                                                      cubist_committees = 50,
                                                      cubist_neighbors = 5,
                                                      results_dir = "results/cubist_pointpatch_hybrid_confirmation",
                                                      save_outputs = TRUE) {
  run_cubist_pointpatch_hybrid_fair_comparison(
    context = context,
    sample_size = sample_size,
    sampling = sampling,
    variants = make_cubist_pointpatch_hybrid_variants(),
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

# -----------------------------------------------------------------------------
# Architecture in plain language
# -----------------------------------------------------------------------------
# 1. For each sampled point s, extract:
#    - tabular covariates X(s)
#    - coordinates (x, y)
#    - a small raster patch around s
# 2. Encode X(s) with an MLP.
# 3. Encode the raster patch with a CNN.
# 4. Encode coordinates with a small MLP.
# 5. Fuse the three embeddings into one latent representation.
# 6. Predict the base signal at the point.
# 7. Apply a residual kriging-like correction using nearby training points.
# 8. Final prediction = base prediction + learned spatial residual correction.

# -----------------------------------------------------------------------------
# Pseudocode
# -----------------------------------------------------------------------------
# for each point s_i:
#   x_i <- tabular covariates at s_i
#   p_i <- raster patch centered at s_i
#   c_i <- coordinates of s_i
#
#   z_tab   = MLP_tab(x_i)
#   z_patch = CNN_patch(p_i)
#   z_coord = MLP_coord(c_i)
#   z_i     = Fuse(z_tab, z_patch, z_coord)
#   mu_i    = Head(z_i)
#
#   neighbors = nearest training points around s_i
#   residual_correction =
#       sum_j w_ij(z_i, z_j, d_ij) * residual_j
#
#   yhat_i = mu_i + beta * residual_correction

# Example usage:
# source("code/KrigingNet_PointPatchCNN.R")
# res_pp <- run_pointpatch_wadoux_spatial_kfold(
#   context = wadoux_context,
#   sample_size = 500,
#   patch_size = 15,
#   pointpatch_params = pointpatch_params
# )
# summarise_comparison(res_pp)
#
# Quick test:
# res_pp_quick <- run_pointpatch_wadoux_spatial_kfold(
#   context = wadoux_context,
#   sample_size = 250,
#   patch_size = 9,
#   pointpatch_params = pointpatch_quick_params
# )
# summarise_comparison(res_pp_quick)
#
# Tuning round:
# res_pp_tuned <- run_pointpatch_wadoux_spatial_kfold_variants(
#   context = wadoux_context,
#   sample_size = 250
# )
# res_pp_tuned %>%
#   group_by(variant, model) %>%
#   summarise(
#     RMSE_mean = mean(RMSE, na.rm = TRUE),
#     R2_mean = mean(R2, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
#   arrange(variant, RMSE_mean)
