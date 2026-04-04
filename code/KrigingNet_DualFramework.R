# =============================================================================
# KrigingNet on:
# 1) simulated DSM data
# 2) the empirical Wadoux et al. SpatialValidation framework
# =============================================================================

rm(list = ls())
set.seed(123)

source("./code/KrigingNet_WadouxComparison.R")

library(terra)
library(sf)
library(dplyr)

# =============================================================================
# 1) Loaders
# =============================================================================

load_simulated_context <- function(sim_path = "./data/soilflux_simulation.rds") {
  readRDS(sim_path)
}

load_wadoux_context <- function(repo_dir = "./external/SpatialValidation") {
  data_dir <- file.path(repo_dir, "data")
  tif_files <- list.files(data_dir, pattern = "\\.tif$", full.names = TRUE)
  if (length(tif_files) == 0) stop("No .tif files found in Wadoux data directory.")

  s <- terra::rast(tif_files)
  names(s) <- tools::file_path_sans_ext(basename(tif_files))

  if (!"ABG1" %in% names(s)) stop("ABG1.tif not found in Wadoux dataset.")

  # Match the original preprocessing: remove water bodies / zero biomass cells.
  s[["ABG1"]][s[["ABG1"]] == 0] <- NA

  s_df <- as.data.frame(s, xy = TRUE, na.rm = TRUE)

  predictor_names <- setdiff(names(s_df), c("x", "y", "ABG1"))
  points_sf <- st_as_sf(s_df, coords = c("x", "y"), crs = terra::crs(s), remove = FALSE)

  list(
    stack = s,
    data = s_df,
    points_sf = points_sf,
    response = "ABG1",
    predictors = predictor_names,
    val_dist_km = 350,
    repo_dir = repo_dir
  )
}

# =============================================================================
# 2) Empirical Wadoux-style protocols
# =============================================================================

prepare_empirical_split <- function(train_df,
                                    test_df,
                                    predictor_names,
                                    response_name = "ABG1",
                                    use_robust_scaling = TRUE,
                                    K = 24,
                                    val_frac = 0.2) {
  n_train <- nrow(train_df)
  if (n_train < 10) stop("Training set too small.")

  val_size <- max(1, floor(n_train * val_frac))
  val_idx_rel <- sample(seq_len(n_train), size = val_size)
  train_idx_rel <- setdiff(seq_len(n_train), val_idx_rel)

  train_core <- train_df[train_idx_rel, , drop = FALSE]
  val_df <- train_df[val_idx_rel, , drop = FALSE]

  X_train <- as.matrix(train_core[, predictor_names, drop = FALSE])
  X_val   <- as.matrix(val_df[, predictor_names, drop = FALSE])
  X_test  <- as.matrix(test_df[, predictor_names, drop = FALSE])

  y_train <- train_core[[response_name]]
  y_val   <- val_df[[response_name]]
  y_test  <- test_df[[response_name]]

  coords_train <- as.matrix(train_core[, c("x", "y"), drop = FALSE])
  coords_val   <- as.matrix(val_df[, c("x", "y"), drop = FALSE])
  coords_test  <- as.matrix(test_df[, c("x", "y"), drop = FALSE])

  scaler <- fit_scaler(X_train, robust = use_robust_scaling)
  X_train_s <- apply_scaler(X_train, scaler)
  X_val_s   <- apply_scaler(X_val, scaler)
  X_test_s  <- apply_scaler(X_test, scaler)

  neighbor_idx <- compute_neighbor_idx_train_only(coords_train, K)

  list(
    X = list(train = X_train_s, val = X_val_s, test = X_test_s),
    y = list(train = y_train, val = y_val, test = y_test),
    coords = list(train = coords_train, val = coords_val, test = coords_test),
    scaler = scaler,
    neighbor_idx_train = neighbor_idx
  )
}

sample_empirical_calibration <- function(context,
                                         sample_size = 500,
                                         sampling = c("simple_random", "systematic")) {
  sampling <- match.arg(sampling)

  if (sampling == "simple_random") {
    idx <- sample(seq_len(nrow(context$data)), size = sample_size)
    return(context$data[idx, , drop = FALSE])
  }

  pts <- terra::spatSample(context$stack[[context$response]],
                           size = sample_size,
                           method = "regular",
                           as.points = TRUE,
                           na.rm = TRUE)
  sampled <- terra::extract(context$stack, pts, xy = TRUE)
  sampled <- as.data.frame(sampled) %>% dplyr::select(-ID) %>% na.omit()
  sampled
}

build_empirical_protocol_splits <- function(calibration_df,
                                            context,
                                            protocol = c("random_cv", "spatial_kfold", "buffered_loo", "design_based_validation"),
                                            n_folds = 10,
                                            val_dist_km = 350,
                                            design_validation_size = 500,
                                            max_splits = NULL) {
  protocol <- match.arg(protocol)

  if (protocol == "design_based_validation") {
    val_df <- terra::spatSample(context$stack,
                                size = design_validation_size,
                                method = "random",
                                na.rm = TRUE,
                                as.points = FALSE,
                                values = TRUE,
                                xy = TRUE) %>%
      as.data.frame() %>%
      na.omit()

    return(list(list(
      train = calibration_df,
      test = val_df,
      protocol = protocol,
      split_id = 1
    )))
  }

  if (protocol == "random_cv") {
    fold_id <- sample(rep(seq_len(n_folds), length.out = nrow(calibration_df)))
    return(lapply(seq_len(n_folds), function(f) {
      list(
        train = calibration_df[fold_id != f, , drop = FALSE],
        test = calibration_df[fold_id == f, , drop = FALSE],
        protocol = protocol,
        split_id = f
      )
    }))
  }

  coords <- as.matrix(calibration_df[, c("x", "y"), drop = FALSE])

  if (protocol == "spatial_kfold") {
    mdist <- dist(coords)
    hc <- hclust(mdist, method = "complete")
    cluster_id <- cutree(hc, h = val_dist_km * 1000)
    groups <- sort(unique(cluster_id))

    return(lapply(seq_along(groups), function(i) {
      g <- groups[i]
      list(
        train = calibration_df[cluster_id != g, , drop = FALSE],
        test = calibration_df[cluster_id == g, , drop = FALSE],
        protocol = protocol,
        split_id = g
      )
    }))
  }

  if (protocol == "buffered_loo") {
    use_idx <- seq_len(nrow(calibration_df))
    if (!is.null(max_splits) && max_splits < length(use_idx)) {
      use_idx <- sort(sample(use_idx, max_splits))
    }

    splits <- lapply(use_idx, function(i) {
      d <- sqrt((coords[,1] - coords[i,1])^2 + (coords[,2] - coords[i,2])^2)
      train_idx <- which(d > val_dist_km * 1000)
      if (length(train_idx) < 10) return(NULL)

      list(
        train = calibration_df[train_idx, , drop = FALSE],
        test = calibration_df[i, , drop = FALSE],
        protocol = protocol,
        split_id = i
      )
    })

    return(Filter(Negate(is.null), splits))
  }

  stop("Unsupported empirical protocol.")
}

run_empirical_protocol <- function(context,
                                   sampling = c("simple_random", "systematic"),
                                   protocol = c("random_cv", "spatial_kfold", "buffered_loo", "design_based_validation"),
                                   sample_size = 500,
                                   n_folds = 10,
                                   val_dist_km = 350,
                                   design_validation_size = 500,
                                   max_splits = NULL,
                                   rf_ntree = 500,
                                   cubist_committees = 50,
                                   cubist_neighbors = 5,
                                   xgb_params = list(),
                                   krigingnet_params = list(),
                                   use_robust_scaling = TRUE,
                                   K = 24) {
  sampling <- match.arg(sampling)
  protocol <- match.arg(protocol)

  calibration_df <- sample_empirical_calibration(context, sample_size = sample_size, sampling = sampling)
  splits <- build_empirical_protocol_splits(
    calibration_df = calibration_df,
    context = context,
    protocol = protocol,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    design_validation_size = design_validation_size,
    max_splits = max_splits
  )

  results <- vector("list", length(splits))

  for (i in seq_along(splits)) {
    sp <- splits[[i]]
    prep <- prepare_empirical_split(
      train_df = sp$train,
      test_df = sp$test,
      predictor_names = context$predictors,
      response_name = context$response,
      use_robust_scaling = use_robust_scaling,
      K = K
    )
    prep$protocol <- protocol
    prep$split_id <- sp$split_id

    cat("\n=============================\n")
    cat("EMPIRICAL", sampling, "|", protocol, "| SPLIT", sp$split_id, "\n")
    cat("=============================\n")

    pred_rf <- fit_predict_rf(prep$X$train, prep$y$train, prep$X$test, ntree = rf_ntree)
    met_rf <- metrics(prep$y$test, pred_rf); met_rf$model <- "RF"

    pred_cb <- fit_predict_cubist(prep$X$train, prep$y$train, prep$X$test,
                                  committees = cubist_committees, neighbors = cubist_neighbors)
    met_cb <- metrics(prep$y$test, pred_cb); met_cb$model <- "Cubist"

    pred_xgb <- do.call(
      fit_predict_xgb,
      c(
        list(
          X_train = prep$X$train, y_train = prep$y$train,
          X_val = prep$X$val, y_val = prep$y$val,
          X_test = prep$X$test
        ),
        xgb_params
      )
    )
    met_xgb <- metrics(prep$y$test, pred_xgb); met_xgb$model <- "XGB"

    kn_out <- do.call(train_krigingnet_one_fold, c(list(fd = prep), krigingnet_params))
    met_kn <- kn_out$metrics_test; met_kn$model <- "KrigingNet"

    df <- bind_rows(met_rf, met_xgb, met_cb, met_kn) %>%
      mutate(dataset = "wadoux_empirical", sampling = sampling, protocol = protocol, split = sp$split_id)

    print(df)
    results[[i]] <- df
  }

  bind_rows(results)
}

run_empirical_krigingnet_ablation <- function(context,
                                              sampling = c("simple_random", "systematic"),
                                              protocol = c("random_cv", "spatial_kfold", "buffered_loo", "design_based_validation"),
                                              sample_size = 500,
                                              n_folds = 10,
                                              val_dist_km = 350,
                                              design_validation_size = 500,
                                              max_splits = NULL,
                                              variants = make_krigingnet_variants(),
                                              use_robust_scaling = TRUE,
                                              K = 24) {
  sampling <- match.arg(sampling)
  protocol <- match.arg(protocol)

  calibration_df <- sample_empirical_calibration(context, sample_size = sample_size, sampling = sampling)
  splits <- build_empirical_protocol_splits(
    calibration_df = calibration_df,
    context = context,
    protocol = protocol,
    n_folds = n_folds,
    val_dist_km = val_dist_km,
    design_validation_size = design_validation_size,
    max_splits = max_splits
  )

  results <- list()

  for (variant_name in names(variants)) {
    cat("\n========================================\n")
    cat("EMPIRICAL ABLATION:", variant_name, "|", sampling, "|", protocol, "\n")
    cat("========================================\n")

    variant_results <- vector("list", length(splits))
    for (i in seq_along(splits)) {
      sp <- splits[[i]]
      prep <- prepare_empirical_split(
        train_df = sp$train,
        test_df = sp$test,
        predictor_names = context$predictors,
        response_name = context$response,
        use_robust_scaling = use_robust_scaling,
        K = K
      )
      prep$protocol <- protocol
      prep$split_id <- sp$split_id

      kn_out <- do.call(train_krigingnet_one_fold, c(list(fd = prep), variants[[variant_name]]))
      met_kn <- kn_out$metrics_test
      met_kn$model <- variant_name
      variant_results[[i]] <- met_kn %>%
        mutate(dataset = "wadoux_empirical", sampling = sampling, protocol = protocol, split = sp$split_id)
    }

    results[[variant_name]] <- bind_rows(variant_results)
  }

  bind_rows(results)
}

# =============================================================================
# 3) Convenience wrappers
# =============================================================================

run_simulated_suite <- function(sim,
                                xgb_params = xgb_params,
                                krigingnet_params = krigingnet_params) {
  bind_rows(
    run_krigingnet_comparison(sim, protocol = "random_cv",
                              xgb_params = xgb_params, krigingnet_params = krigingnet_params) %>%
      mutate(dataset = "simulated"),
    run_krigingnet_comparison(sim, protocol = "spatial_block_cv",
                              xgb_params = xgb_params, krigingnet_params = krigingnet_params) %>%
      mutate(dataset = "simulated"),
    run_krigingnet_comparison(sim, protocol = "design_based_holdout",
                              xgb_params = xgb_params, krigingnet_params = krigingnet_params) %>%
      mutate(dataset = "simulated")
  )
}

run_wadoux_suite <- function(context,
                             xgb_params = xgb_params,
                             krigingnet_params = krigingnet_params,
                             sample_size = 500,
                             max_buffer_splits = 50) {
  bind_rows(
    run_empirical_protocol(context, sampling = "simple_random", protocol = "random_cv",
                           sample_size = sample_size,
                           xgb_params = xgb_params, krigingnet_params = krigingnet_params),
    run_empirical_protocol(context, sampling = "simple_random", protocol = "spatial_kfold",
                           sample_size = sample_size,
                           xgb_params = xgb_params, krigingnet_params = krigingnet_params),
    run_empirical_protocol(context, sampling = "simple_random", protocol = "design_based_validation",
                           sample_size = sample_size,
                           xgb_params = xgb_params, krigingnet_params = krigingnet_params),
    run_empirical_protocol(context, sampling = "simple_random", protocol = "buffered_loo",
                           sample_size = sample_size,
                           max_splits = max_buffer_splits,
                           xgb_params = xgb_params, krigingnet_params = krigingnet_params)
  )
}

summarise_dual_framework <- function(sim_results = NULL, empirical_results = NULL) {
  bind_rows(sim_results, empirical_results) %>%
    attach_validation_goal() %>%
    group_by(dataset, validation_goal, protocol, model) %>%
    summarise(
      N_mean = mean(N, na.rm = TRUE),
      R2_mean = mean(R2, na.rm = TRUE),
      RMSE_mean = mean(RMSE, na.rm = TRUE),
      MAE_mean = mean(MAE, na.rm = TRUE),
      Bias_mean = mean(Bias, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(dataset, validation_goal, RMSE_mean)
}

plot_dual_framework_goals <- function(sim_results = NULL, empirical_results = NULL, metric = "RMSE") {
  bind_rows(sim_results, empirical_results) %>%
    attach_validation_goal() %>%
    ggplot(aes(x = model, y = .data[[metric]], fill = model)) +
    geom_boxplot() +
    facet_grid(dataset ~ validation_goal, scales = "free_y") +
    theme_minimal() +
    coord_flip() +
    guides(fill = "none") +
    ggtitle(paste("KrigingNet framework comparison |", metric))
}

# =============================================================================
# 4) Defaults and examples
# =============================================================================

sim_context <- load_simulated_context()
wadoux_context <- load_wadoux_context()

# Example: one empirical protocol on the exact Wadoux dataset
# res_wadoux_random <- run_empirical_protocol(
#   wadoux_context,
#   sampling = "simple_random",
#   protocol = "random_cv",
#   sample_size = 500,
#   xgb_params = xgb_params,
#   krigingnet_params = krigingnet_params
# )

# Example: full Wadoux-style suite
# res_wadoux_all <- run_wadoux_suite(
#   wadoux_context,
#   sample_size = 500,
#   max_buffer_splits = 50,
#   xgb_params = xgb_params,
#   krigingnet_params = krigingnet_params
# )

# Example: simulated + empirical summaries
# res_sim_all <- run_simulated_suite(sim_context, xgb_params, krigingnet_params)
# bind_rows(res_sim_all, res_wadoux_all) %>%
#   group_by(dataset, protocol, model) %>%
#   summarise(
#     RMSE_mean = mean(RMSE, na.rm = TRUE),
#     MAE_mean = mean(MAE, na.rm = TRUE),
#     R2_mean = mean(R2, na.rm = TRUE),
#     .groups = "drop"
#   )
