rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")

# =============================================================================
# run_wadoux_style_rf_conv_comparison.R
#
# Fair head-to-head benchmark between:
#   - RF (ranger defaults; Wadoux-style baseline)
#   - ConvKrigingNet2D (anisotropic variant by default)
#
# Goals:
#   - same sampled calibration sets
#   - same outer folds / test points
#   - same final metrics (Wadoux metrics: ME, RMSE, Spearman^2, MEC)
#   - same inner train/validation split inside each outer training set
#
# This script is the comparison benchmark. Exact RF-only reproduction of Wadoux
# remains separate in the dedicated reproduction runners.
#
# Default protocols in the comparison benchmark:
#   - DesignBased
#   - RandomKFold
#   - SpatialKFold
#   - BLOOCV
#
# Optional:
#   - Population can be enabled, but is disabled by default because full-map
#     ConvKrigingNet2D prediction over ~1.3M cells is extremely expensive.
#
# Environment overrides:
#   WADOUX_SCENARIO           random | regular_grid | clustered_random
#   WADOUX_PROTOCOLS          comma-separated list
#   WADOUX_INCLUDE_POPULATION TRUE/FALSE
#   WADOUX_N_ITER
#   WADOUX_SAMPLE_SIZE
#   WADOUX_VAL_FRAC
#   WADOUX_VAL_DIST_KM
#   WADOUX_RANDOM_K
#   WADOUX_BLOO_GROUPS
#   WADOUX_BLOO_TEST_PIXELS
#   WADOUX_CLUSTER_CENTERS
#   WADOUX_CLUSTER_N_PSU
#   WADOUX_CLUSTER_M_PER_PSU
#   WADOUX_MODEL_PROFILE      quick | full
#   WADOUX_TRAIN_SEED
#   WADOUX_DEVICE
#   WADOUX_RESULTS_DIR
# =============================================================================

parse_bool_env <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}

parse_protocols_env <- function(default = c("DesignBased", "RandomKFold", "SpatialKFold", "BLOOCV")) {
  raw <- Sys.getenv("WADOUX_PROTOCOLS", unset = paste(default, collapse = ","))
  vals <- trimws(unlist(strsplit(raw, ",")))
  vals[vals != ""]
}

parse_models_env <- function(default = c("RF", "ConvKrigingNet2D")) {
  raw <- Sys.getenv("WADOUX_MODELS", unset = paste(default, collapse = ","))
  vals <- trimws(unlist(strsplit(raw, ",")))
  vals <- vals[vals != ""]
  allowed <- c("RF", "ConvKrigingNet2D")
  invalid <- setdiff(vals, allowed)
  if (length(invalid) > 0) {
    stop("Unsupported model(s) in WADOUX_MODELS: ", paste(invalid, collapse = ", "))
  }
  unique(vals)
}

bind_patch_arrays_4d <- function(arrays) {
  arrays <- arrays[!vapply(arrays, is.null, logical(1))]
  if (length(arrays) == 0) {
    return(NULL)
  }
  if (length(arrays) == 1) {
    return(arrays[[1]])
  }

  ref_dim <- dim(arrays[[1]])
  if (length(ref_dim) != 4) {
    stop("Expected 4D patch arrays.")
  }
  for (i in 2:length(arrays)) {
    d <- dim(arrays[[i]])
    if (length(d) != 4 || !all(d[1:3] == ref_dim[1:3])) {
      stop("All patch arrays must agree in channels, height, and width.")
    }
  }

  n_total <- sum(vapply(arrays, function(a) dim(a)[4], numeric(1)))
  out <- array(NA_real_, dim = c(ref_dim[1], ref_dim[2], ref_dim[3], n_total))
  pos <- 1L
  for (arr in arrays) {
    n_i <- dim(arr)[4]
    out[, , , pos:(pos + n_i - 1L)] <- arr
    pos <- pos + n_i
  }
  out
}

load_convkrigingnet2d_env <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop(
      "The 'torch' package is not available in this R library. ",
      "Run this benchmark from the same R environment used to train ConvKrigingNet2D, ",
      "or install torch for this R installation before executing this script."
    )
  }

  loader_parent <- new.env(parent = globalenv())
  env <- new.env(parent = loader_parent)

  loader_parent$source <- function(file, ...) {
    is_absolute <- grepl("^(?:[A-Za-z]:[\\\\/]|/)", file)
    resolved <- if (is_absolute) {
      file
    } else {
      file.path(project_root_wadoux, sub("^\\./", "", file))
    }
    sys.source(resolved, envir = env)
    invisible(NULL)
  }

  sys.source(file.path(project_root_wadoux, "code", "ConvKrigingNet2D.R"), envir = env)
  env
}

make_inner_val_split_wadoux <- function(train_df, val_frac = 0.2, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n_train <- nrow(train_df)
  if (n_train < 10) stop("Outer training set too small for fair comparison.")
  val_size <- max(1L, floor(n_train * val_frac))
  val_idx <- sample.int(n_train, size = val_size, replace = FALSE)
  train_idx <- setdiff(seq_len(n_train), val_idx)
  list(
    train = train_df[train_idx, , drop = FALSE],
    val = train_df[val_idx, , drop = FALSE]
  )
}

get_conv_params_profile <- function(conv_env, profile = c("quick", "full", "n500"), train_seed = 123L, device = "cpu") {
  profile <- match.arg(profile)
  if (profile == "quick") {
    params <- modifyList(
      conv_env$convkriging2d_quick_params,
      list(
        kriging_mode = "anisotropic",
        train_seed = train_seed,
        deterministic_batches = TRUE,
        device = device
      )
    )
  } else if (profile == "n500") {
    # Targeted fixes for n=500 Wadoux benchmark.
    # Base architecture unchanged from multiseed confirmation (baseline_params).
    # Four targeted tweaks for n=500:
    #   1. warmup_epochs=8  : train backbone first; bank residuals are meaningful
    #                         before the kriging layer starts learning ell/theta
    #   2. base_loss_weight=0.05 : small auxiliary backbone loss; ensures the DNN
    #                         gets direct gradient signal even when kriging dominates
    #   3. lr_decay=0.5, lr_patience=4, min_lr=5e-5 : reduce LR when stuck
    #                         instead of stopping; patience=15 gives more room
    #   4. batch_size=48    : ~400 train pts / 48 = 8 batches/epoch (vs 4 with 96)
    #                         doubles gradient steps per epoch
    params <- modifyList(
      conv_env$convkriging2d_baseline_params,
      list(
        kriging_mode       = "anisotropic",
        train_seed         = train_seed,
        deterministic_batches = TRUE,
        device             = device,
        warmup_epochs      = 8L,
        # beta_init=0 -> sigmoid(0)=0.5: model uses kriging from the start.
        # Safe because warmup already gave the base model 8 epochs.
        # Default beta_init=-4 -> sigmoid(-4)=0.018 was effectively killing kriging.
        # beta_init=0 -> sigmoid(0)=0.5: model uses kriging from the start.
        # krig_loss_weight removed: direct delta supervision introduced +5 Mg/ha
        # bias in DesignBased because it memorizes training residual direction.
        beta_init          = 0,
        base_loss_weight   = 0.05,
        batch_size         = 48L,
        lr_decay           = 0.5,
        lr_patience        = 4L,
        min_lr             = 5e-5,
        patience           = 15L,
        epochs             = 80L
        # dist_scale = NULL, krig_dropout = 0 (both disabled — defaults).
        #
        # Distance-aware gate tested across 4 variants (all 1-iteration):
        #   baseline (no gate):              DesignBased RMSE=32.8, ME=0,   Spatial=37.0
        #   gate=1.0 train+eval:             DesignBased RMSE=35.5, ME=3.1, Spatial=38.3
        #   gate=0.5 eval-only rel:          DesignBased RMSE=35.7, ME=5.3, Spatial=38.3
        #   krig_dropout=0.3 + gate=0.5:    DesignBased RMSE=35.8, ME=3.2, Spatial=39.0
        #
        # All gate variants degraded DesignBased and did not improve SpatialKFold.
        # Root cause: base + kriging are co-trained; the base learns to rely on
        # β×δ to correct its positive bias. Any eval-time suppression of kriging
        # exposes this bias (ME > 0 → higher RMSE). krig_dropout at 0.3 is
        # insufficient to break the dependency; would need ~0.7+ which would
        # prevent the kriging layer from learning. Requires separate architectural
        # treatment (future work).
        #
        # Historical note: earlier exploratory runs reported DesignBased
        # RMSE near the Wadoux RF baseline. Treat the CSV outputs written by
        # this runner as the source of truth for any current claim.
      )
    )
  } else {
    # "full" uses the same params as the multiseed confirmation:
    # convkriging2d_baseline_params + kriging_mode="anisotropic".
    # convkriging2d_anisotropic_stable_params was tested and performed worse.
    params <- modifyList(
      conv_env$convkriging2d_baseline_params,
      list(
        kriging_mode = "anisotropic",
        train_seed = train_seed,
        deterministic_batches = TRUE,
        device = device
      )
    )
  }
  params
}

compare_rf_conv_on_explicit_split <- function(train_core,
                                              val_df,
                                              test_df,
                                              models = c("RF", "ConvKrigingNet2D"),
                                              calibration_patch_cache = NULL,
                                              conv_env,
                                              conv_params,
                                              patch_size = 15,
                                              context = NULL) {
  out <- list(pred_rf = NULL, pred_conv = NULL)

  rf_model <- NULL   # kept in scope for the benchmark baseline only

  if ("RF" %in% models) {
    form_rf <- as.formula(
      paste(response_name_wadoux, "~", paste(predList_modelfull_wadoux, collapse = "+"))
    )
    rf_model <- fit_rf_default_wadoux(train_core, form_rf)
    out$pred_rf <- predict_rf_default_wadoux(rf_model, test_df)
  }

  if ("ConvKrigingNet2D" %in% models) {
    if (is.null(conv_env) || is.null(conv_params)) {
      stop("ConvKrigingNet2D requested but conv_env/conv_params were not provided.")
    }
    if (is.null(context)) {
      context <- conv_env$wadoux_context
    }

    combined_df <- bind_rows(train_core, val_df, test_df)
    n_train <- nrow(train_core)
    n_val <- nrow(val_df)
    n_test <- nrow(test_df)
    patch_cache_combined <- NULL

    cache_col <- "..patch_cache_row_id"
    train_val_cache_ok <- !is.null(calibration_patch_cache) &&
      all(cache_col %in% names(train_core), cache_col %in% names(val_df)) &&
      !anyNA(train_core[[cache_col]]) &&
      !anyNA(val_df[[cache_col]])
    test_cache_ok <- !is.null(calibration_patch_cache) &&
      cache_col %in% names(test_df) &&
      !anyNA(test_df[[cache_col]])

    if (train_val_cache_ok) {
      train_cache_idx <- as.integer(train_core[[cache_col]])
      val_cache_idx <- as.integer(val_df[[cache_col]])
      patch_arrays <- list(
        calibration_patch_cache$patches_raw[, , , train_cache_idx, drop = FALSE],
        calibration_patch_cache$patches_raw[, , , val_cache_idx, drop = FALSE]
      )

      if (test_cache_ok) {
        test_cache_idx <- as.integer(test_df[[cache_col]])
        patch_arrays[[3]] <- calibration_patch_cache$patches_raw[, , , test_cache_idx, drop = FALSE]
      } else {
        external_test_cache <- conv_env$build_pointpatch_patch_cache(
          context = context,
          calibration_df = test_df,
          patch_size = patch_size,
          label = "test patches"
        )
        patch_arrays[[3]] <- external_test_cache$patches_raw
      }

      patch_cache_combined <- list(
        patches_raw = bind_patch_arrays_4d(patch_arrays),
        patch_size = patch_size,
        n = nrow(combined_df)
      )
    }

    neighbor_pool_k <- 24L
    if (!is.null(conv_params$K_neighbors)) {
      neighbor_pool_k <- max(neighbor_pool_k, as.integer(conv_params$K_neighbors))
    }

    fd_conv <- conv_env$prepare_pointpatch_fold(
      context = context,
      calibration_df = combined_df,
      train_idx = seq_len(n_train),
      val_idx = n_train + seq_len(n_val),
      test_idx = n_train + n_val + seq_len(n_test),
      patch_size = patch_size,
      K = neighbor_pool_k,
      patch_cache = patch_cache_combined
    )

    conv_out <- do.call(
      conv_env$train_convkrigingnet2d_one_fold,
      c(list(fd = fd_conv), conv_params)
    )

    out$pred_conv <- conv_out$pred_test
  }

  out
}

make_metric_rows <- function(obs, preds, models, protocol) {
  rows <- list()

  if ("RF" %in% models && !is.null(preds$pred_rf)) {
    rows[[length(rows) + 1L]] <- wadoux_eval(obs = obs, pred = preds$pred_rf) %>%
      mutate(model = "RF")
  }

  if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
    rows[[length(rows) + 1L]] <- wadoux_eval(obs = obs, pred = preds$pred_conv) %>%
      mutate(model = "ConvKrigingNet2D")
  }

  bind_rows(rows) %>% mutate(protocol = protocol)
}

evaluate_design_based_protocol <- function(valuetable,
                                           common_data,
                                           models,
                                           calibration_patch_cache,
                                           conv_env,
                                           conv_params,
                                           sample_size,
                                           val_frac,
                                           iter_seed,
                                           patch_size = 15) {
  inner <- make_inner_val_split_wadoux(valuetable, val_frac = val_frac, seed = iter_seed + 101L)
  val_srs <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  preds <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    test_df = val_srs,
    models = models,
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
    patch_size = patch_size
  )

  obs <- val_srs[[response_name_wadoux]]
  make_metric_rows(obs = obs, preds = preds, models = models, protocol = "DesignBased")
}

evaluate_random_kfold_protocol <- function(valuetable,
                                           models,
                                           calibration_patch_cache,
                                           conv_env,
                                           conv_params,
                                           random_k,
                                           val_frac,
                                           iter_seed,
                                           patch_size = 15) {
  flds <- caret::createFolds(
    valuetable[[response_name_wadoux]],
    k = random_k,
    list = TRUE,
    returnTrain = FALSE
  )

  pred_rf_all <- c()
  pred_conv_all <- c()
  obs_all <- c()

  for (j in seq_along(flds)) {
    id <- flds[[j]]
    outer_train <- valuetable[-id, , drop = FALSE]
    outer_test <- valuetable[id, , drop = FALSE]
    inner <- make_inner_val_split_wadoux(
      outer_train,
      val_frac = val_frac,
      seed = iter_seed * 1000L + j
    )

    preds <- compare_rf_conv_on_explicit_split(
      train_core = inner$train,
      val_df = inner$val,
      test_df = outer_test,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      patch_size = patch_size
    )

    obs_all <- c(obs_all, outer_test[[response_name_wadoux]])
    if ("RF" %in% models && !is.null(preds$pred_rf)) {
      pred_rf_all <- c(pred_rf_all, preds$pred_rf)
    }
    if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
      pred_conv_all <- c(pred_conv_all, preds$pred_conv)
    }
  }

  make_metric_rows(
    obs = obs_all,
    preds = list(pred_rf = pred_rf_all, pred_conv = pred_conv_all),
    models = models,
    protocol = "RandomKFold"
  )
}

evaluate_spatial_kfold_protocol <- function(valuetable,
                                            models,
                                            calibration_patch_cache,
                                            conv_env,
                                            conv_params,
                                            val_dist_km,
                                            val_frac,
                                            iter_seed,
                                            patch_size = 15) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  groups <- sort(unique(spatial_folds))

  pred_rf_all <- c()
  pred_conv_all <- c()
  obs_all <- c()

  for (j in seq_along(groups)) {
    g <- groups[j]
    outer_train <- valuetable[spatial_folds != g, , drop = FALSE]
    outer_test <- valuetable[spatial_folds == g, , drop = FALSE]
    inner <- make_inner_val_split_wadoux(
      outer_train,
      val_frac = val_frac,
      seed = iter_seed * 1000L + j
    )

    preds <- compare_rf_conv_on_explicit_split(
      train_core = inner$train,
      val_df = inner$val,
      test_df = outer_test,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      patch_size = patch_size
    )

    obs_all <- c(obs_all, outer_test[[response_name_wadoux]])
    if ("RF" %in% models && !is.null(preds$pred_rf)) {
      pred_rf_all <- c(pred_rf_all, preds$pred_rf)
    }
    if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
      pred_conv_all <- c(pred_conv_all, preds$pred_conv)
    }
  }

  make_metric_rows(
    obs = obs_all,
    preds = list(pred_rf = pred_rf_all, pred_conv = pred_conv_all),
    models = models,
    protocol = "SpatialKFold"
  )
}

evaluate_bloocv_protocol <- function(valuetable,
                                     models,
                                     calibration_patch_cache,
                                     conv_env,
                                     conv_params,
                                     val_dist_km,
                                     val_frac,
                                     iter_seed,
                                     nb_groups = 10,
                                     nb_test_pixels = 100,
                                     patch_size = 15) {
  # BLOOCV trains one model per test pixel. With ConvKrigingNet2D this is very
  # expensive: nb_groups * nb_test_pixels complete neural network trainings.
  # Hard-cap nb_test_pixels at 10 when ConvKrigingNet2D is included to prevent
  # multi-hour runs. For RF-only benchmarks the cap does not apply.
  if ("ConvKrigingNet2D" %in% models && nb_test_pixels > 10) {
    cap <- 10L
    cat(sprintf(
      "[BLOOCV] ConvKrigingNet2D detected: capping nb_test_pixels %d -> %d to avoid %d+ model trainings.\n",
      nb_test_pixels, cap, nb_groups * nb_test_pixels
    ))
    nb_test_pixels <- cap
  }
  nb_iteration <- nb_groups * nb_test_pixels
  rows <- vector("list", nb_iteration)

  for (j in seq_len(nb_iteration)) {
    point_in_range <- FALSE

    while (!point_in_range) {
      id_focal <- sample.int(nrow(valuetable), size = 1)
      focal_point <- valuetable[id_focal, , drop = FALSE]
      training_tmp <- valuetable[-id_focal, , drop = FALSE]
      training_tmp <- exclude_by_radius_wadoux(training_tmp, focal_point, val_dist_km * 1000)

      if (nrow(training_tmp) < 10) next
      if (!point_within_predictor_range_wadoux(training_tmp, focal_point, predList_modelfull_wadoux)) next

      point_in_range <- TRUE
      outer_train <- training_tmp
      outer_test <- focal_point
    }

    inner <- make_inner_val_split_wadoux(
      outer_train,
      val_frac = val_frac,
      seed = iter_seed * 100000L + j
    )

    preds <- compare_rf_conv_on_explicit_split(
      train_core = inner$train,
      val_df = inner$val,
      test_df = outer_test,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      patch_size = patch_size
    )

    rows[[j]] <- data.frame(
      task = j,
      obs = outer_test[[response_name_wadoux]],
      pred_rf = if ("RF" %in% models) preds$pred_rf else NA_real_,
      pred_conv = if ("ConvKrigingNet2D" %in% models) preds$pred_conv else NA_real_
    )
  }

  tmp <- bind_rows(rows)
  tmp$group <- caret::createFolds(seq_len(nrow(tmp)), k = nb_groups, list = FALSE, returnTrain = FALSE)

  rf_group_metrics <- vector("list", max(tmp$group))
  conv_group_metrics <- vector("list", max(tmp$group))

  for (g in seq_len(max(tmp$group))) {
    idx <- which(tmp$group == g)
    if ("RF" %in% models) {
      rf_group_metrics[[g]] <- wadoux_eval(obs = tmp$obs[idx], pred = tmp$pred_rf[idx])
    }
    if ("ConvKrigingNet2D" %in% models) {
      conv_group_metrics[[g]] <- wadoux_eval(obs = tmp$obs[idx], pred = tmp$pred_conv[idx])
    }
  }

  rows_out <- list()
  if ("RF" %in% models) {
    rows_out[[length(rows_out) + 1L]] <- bind_rows(rf_group_metrics) %>%
      summarise(
        ME = mean(ME, na.rm = TRUE),
        RMSE = mean(RMSE, na.rm = TRUE),
        r2 = mean(r2, na.rm = TRUE),
        MEC = mean(MEC, na.rm = TRUE)
      ) %>%
      mutate(model = "RF")
  }
  if ("ConvKrigingNet2D" %in% models) {
    rows_out[[length(rows_out) + 1L]] <- bind_rows(conv_group_metrics) %>%
      summarise(
        ME = mean(ME, na.rm = TRUE),
        RMSE = mean(RMSE, na.rm = TRUE),
        r2 = mean(r2, na.rm = TRUE),
        MEC = mean(MEC, na.rm = TRUE)
      ) %>%
      mutate(model = "ConvKrigingNet2D")
  }

  bind_rows(rows_out) %>% mutate(protocol = "BLOOCV")
}

evaluate_population_protocol_optional <- function(valuetable,
                                                  common_data,
                                                  models,
                                                  calibration_patch_cache,
                                                  conv_env,
                                                  conv_params,
                                                  val_frac,
                                                  iter_seed,
                                                  patch_size = 15) {
  inner <- make_inner_val_split_wadoux(valuetable, val_frac = val_frac, seed = iter_seed + 101L)
  preds <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    test_df = common_data$s_df,
    models = models,
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
    patch_size = patch_size
  )
  obs <- common_data$s_df[[response_name_wadoux]]
  make_metric_rows(obs = obs, preds = preds, models = models, protocol = "Population")
}

scenario <- Sys.getenv("WADOUX_SCENARIO", unset = "random")
protocols <- parse_protocols_env()
models <- parse_models_env()
include_population <- parse_bool_env("WADOUX_INCLUDE_POPULATION", default = FALSE)
if (include_population && !"Population" %in% protocols) {
  protocols <- c("Population", protocols)
}

sample_size <- as.integer(Sys.getenv("WADOUX_SAMPLE_SIZE", unset = "500"))
n_iter <- as.integer(Sys.getenv("WADOUX_N_ITER", unset = "500"))
val_frac <- as.numeric(Sys.getenv("WADOUX_VAL_FRAC", unset = "0.2"))
val_dist_km <- as.numeric(Sys.getenv("WADOUX_VAL_DIST_KM", unset = "350"))
random_k <- as.integer(Sys.getenv("WADOUX_RANDOM_K", unset = "10"))
bloo_groups <- as.integer(Sys.getenv("WADOUX_BLOO_GROUPS", unset = "10"))
bloo_test_pixels <- as.integer(Sys.getenv("WADOUX_BLOO_TEST_PIXELS", unset = "100"))
kmeans_centers <- as.integer(Sys.getenv("WADOUX_CLUSTER_CENTERS", unset = "100"))
n_psu <- as.integer(Sys.getenv("WADOUX_CLUSTER_N_PSU", unset = "20"))
m_per_psu <- as.integer(Sys.getenv("WADOUX_CLUSTER_M_PER_PSU", unset = "25"))
model_profile <- Sys.getenv("WADOUX_MODEL_PROFILE", unset = "quick")
train_seed <- as.integer(Sys.getenv("WADOUX_TRAIN_SEED", unset = "123"))
device <- Sys.getenv("WADOUX_DEVICE", unset = "cpu")

if (scenario == "clustered_random") {
  sample_size <- n_psu * m_per_psu
}

results_dir <- Sys.getenv(
  "WADOUX_RESULTS_DIR",
  unset = file.path(project_root_wadoux, "results", sprintf("wadoux_style_rf_conv_%s_comparison", scenario))
)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

need_polygon <- identical(scenario, "regular_grid")
cat(sprintf("Loading common data for scenario: %s\n", scenario))
common_data <- load_wadoux_common_data(include_polygon = need_polygon)

if ("Population" %in% protocols && "ConvKrigingNet2D" %in% models) {
  cat(sprintf(
    "Warning: Population protocol with ConvKrigingNet2D will score the full raster domain (%d locations) and may be extremely slow.\n",
    nrow(common_data$s_df)
  ))
}

conv_env <- NULL
conv_params <- NULL
if ("ConvKrigingNet2D" %in% models) {
  cat("Loading ConvKrigingNet2D environment...\n")
  conv_env <- load_convkrigingnet2d_env()
  override_path <- Sys.getenv("WADOUX_TWOSTAGE_SCRIPT", unset = "")
  if (nchar(override_path) > 0 && file.exists(override_path)) {
    sys.source(override_path, envir = conv_env)
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_twostage_one_fold
    cat("[TwoStage] Overriding train function with two-stage trainer\n")
  }
  # ExpCov hook — replaces kriging layer with pure exponential covariance
  #               takes priority over TwoStage
  expcov_path <- Sys.getenv("WADOUX_EXPCOV_SCRIPT", unset = "")
  if (nchar(expcov_path) > 0 && file.exists(expcov_path)) {
    sys.source(expcov_path, envir = conv_env)
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_expcov_one_fold
    cat("[ExpCov] Overriding train function with ExpCov trainer\n")
  }
  # Auto hook — fully self-configuring model; takes priority over ExpCov/TwoStage
  # v5 hook — takes priority over v4 Auto
  auto_v5_path <- Sys.getenv("WADOUX_AUTO_V5_SCRIPT", unset = "")
  if (nchar(auto_v5_path) > 0 && file.exists(auto_v5_path)) {
    sys.source(auto_v5_path, envir = conv_env)
    # Also load v4 base (v5 builds on top)
    auto_path <- Sys.getenv("WADOUX_AUTO_SCRIPT", unset = "")
    if (nchar(auto_path) > 0 && file.exists(auto_path)) {
      sys.source(auto_path, envir = conv_env)
    }
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_auto_one_fold_v5
    cat("[Auto v5] Overriding train function with COMPLETE AUTO-CONFIG v5\n")
  } else {
    # Fall back to v4 Auto
    auto_path <- Sys.getenv("WADOUX_AUTO_SCRIPT", unset = "")
    if (nchar(auto_path) > 0 && file.exists(auto_path)) {
      sys.source(auto_path, envir = conv_env)
      conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_auto_one_fold
      cat("[Auto v4] Overriding train function with Auto (self-configuring) trainer\n")
    }
  }
  # DeepRK hook — takes priority over TwoStage, ExpCov and Auto if all are set
  deeprk_path <- Sys.getenv("WADOUX_DEEPRK_SCRIPT", unset = "")
  if (nchar(deeprk_path) > 0 && file.exists(deeprk_path)) {
    sys.source(deeprk_path, envir = conv_env)
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_deeprk_one_fold
    cat("[DeepRK] Overriding train function with DeepRK trainer\n")
  }
  conv_params <- get_conv_params_profile(
    conv_env = conv_env,
    profile = model_profile,
    train_seed = train_seed,
    device = device
  )
  # DeepRK calibration override: allows Benchmark_DeepRK.R to inject
  # calibrate_method = "linear" without touching conv_params directly.
  deeprk_calibrate <- Sys.getenv("WADOUX_DEEPRK_CALIBRATE", unset = "")
  if (nchar(deeprk_calibrate) > 0 && nchar(deeprk_path) > 0) {
    conv_params$calibrate_method <- deeprk_calibrate
    cat(sprintf("[DeepRK] calibrate_method overridden → '%s'\n", deeprk_calibrate))
  }
  # Ablation override: WADOUX_WARMUP_EPOCHS overrides warmup_epochs in conv_params.
  ablation_warmup <- Sys.getenv("WADOUX_WARMUP_EPOCHS", unset = "")
  if (nchar(ablation_warmup) > 0) {
    conv_params$warmup_epochs <- as.integer(ablation_warmup)
    cat(sprintf("[Ablation] warmup_epochs overridden → %dL\n", conv_params$warmup_epochs))
  }
  # Ablation override: WADOUX_BASE_LOSS_WEIGHT
  ablation_blw <- Sys.getenv("WADOUX_BASE_LOSS_WEIGHT", unset = "")
  if (nchar(ablation_blw) > 0) {
    conv_params$base_loss_weight <- as.numeric(ablation_blw)
    cat(sprintf("[Ablation] base_loss_weight overridden → %.4f\n", conv_params$base_loss_weight))
  }
  # Ablation override: WADOUX_LR
  ablation_lr <- Sys.getenv("WADOUX_LR", unset = "")
  if (nchar(ablation_lr) > 0) {
    conv_params$lr <- as.numeric(ablation_lr)
    cat(sprintf("[Ablation] lr overridden → %.2e\n", conv_params$lr))
  }
  # Ablation override: WADOUX_BANK_REFRESH_EVERY
  ablation_bre <- Sys.getenv("WADOUX_BANK_REFRESH_EVERY", unset = "")
  if (nchar(ablation_bre) > 0) {
    conv_params$bank_refresh_every <- as.integer(ablation_bre)
    cat(sprintf("[Ablation] bank_refresh_every overridden → %dL\n", conv_params$bank_refresh_every))
  }
  # Ablation override: architecture dimensions and dropouts
  ablation_patch_dim   <- Sys.getenv("WADOUX_PATCH_DIM",    unset = "")
  ablation_d           <- Sys.getenv("WADOUX_D",            unset = "")
  ablation_tab_drop    <- Sys.getenv("WADOUX_TAB_DROPOUT",  unset = "")
  ablation_patch_drop  <- Sys.getenv("WADOUX_PATCH_DROPOUT",unset = "")
  ablation_coord_dim   <- Sys.getenv("WADOUX_COORD_DIM",    unset = "")
  if (nchar(ablation_patch_dim)  > 0) { conv_params$patch_dim    <- as.integer(ablation_patch_dim);  cat(sprintf("[Ablation] patch_dim    → %d\n",    conv_params$patch_dim))    }
  if (nchar(ablation_d)          > 0) { conv_params$d             <- as.integer(ablation_d);          cat(sprintf("[Ablation] d           → %d\n",    conv_params$d))            }
  if (nchar(ablation_tab_drop)   > 0) { conv_params$tab_dropout   <- as.numeric(ablation_tab_drop);   cat(sprintf("[Ablation] tab_dropout → %.2f\n", conv_params$tab_dropout))   }
  if (nchar(ablation_patch_drop) > 0) { conv_params$patch_dropout <- as.numeric(ablation_patch_drop); cat(sprintf("[Ablation] patch_drop  → %.2f\n", conv_params$patch_dropout)) }
  if (nchar(ablation_coord_dim)  > 0) { conv_params$coord_dim     <- as.integer(ablation_coord_dim);  cat(sprintf("[Ablation] coord_dim   → %d\n",    conv_params$coord_dim))    }
}
# v4 DYNAMIC patch_size: derived from sample_size using auto_kriging_config rule
# patch_size = min(max(8, floor(√n)), 31)
# Ablation override: WADOUX_PATCH_SIZE can still override if explicitly set.
patch_size_ablation <- Sys.getenv("WADOUX_PATCH_SIZE", unset = "")
if (nchar(patch_size_ablation) > 0) {
  # Explicit ablation override takes priority
  patch_size <- as.integer(patch_size_ablation)
  cat(sprintf("[Ablation] patch_size overridden → %dL\n", patch_size))
} else {
  # v4 AUTO: dynamic patch_size from sample_size
  patch_size <- as.integer(min(max(8L, floor(sqrt(sample_size))), 31L))
  cat(sprintf("[Auto v4] patch_size dynamically set → %d [rule: min(max(8,⌊√n⌋), 31)]\n", patch_size))
}

config <- data.frame(
  scenario = scenario,
  models = paste(models, collapse = ","),
  protocols = paste(protocols, collapse = ","),
  include_population = include_population,
  sample_size = sample_size,
  n_iter = n_iter,
  val_frac = val_frac,
  val_dist_km = val_dist_km,
  random_k = random_k,
  bloo_groups = bloo_groups,
  bloo_test_pixels = bloo_test_pixels,
  kmeans_centers = kmeans_centers,
  n_psu = n_psu,
  m_per_psu = m_per_psu,
  model_profile = model_profile,
  train_seed = train_seed,
  device = device,
  stringsAsFactors = FALSE
)
write.csv(config, file.path(results_dir, "wadoux_style_rf_conv_config.csv"), row.names = FALSE)

sample_calibration <- function() {
  if (scenario == "random") {
    out <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  } else if (scenario == "regular_grid") {
    out <- sample_valuetable_regular_grid_wadoux(common_data = common_data, sample_size = sample_size)
  } else if (scenario == "clustered_random") {
    out <- sample_valuetable_clustered_random_wadoux(
      common_data = common_data,
      kmeans_centers = kmeans_centers,
      n_psu = n_psu,
      m_per_psu = m_per_psu
    )
  } else {
    stop("Unsupported scenario: ", scenario)
  }
  out$..patch_cache_row_id <- seq_len(nrow(out))
  out
}

all_results <- vector("list", n_iter)

for (iter in seq_len(n_iter)) {
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("Wadoux-style RF vs ConvKrigingNet2D | scenario=%s | iteration %d / %d\n", scenario, iter, n_iter))
  cat(sprintf("==================================================\n"))

  valuetable <- na.omit(as.data.frame(sample_calibration()))
  valuetable$..patch_cache_row_id <- seq_len(nrow(valuetable))
  calibration_patch_cache <- NULL
  if ("ConvKrigingNet2D" %in% models) {
    calibration_patch_cache <- conv_env$build_pointpatch_patch_cache(
      context = conv_env$wadoux_context,
      calibration_df = valuetable,
      patch_size = patch_size,
      label = sprintf("scenario=%s iter=%d calibration patches", scenario, iter)
    )
  }
  iter_results <- list()

  if ("Population" %in% protocols) {
    cat("\n--- Population ---\n")
    iter_results[["Population"]] <- evaluate_population_protocol_optional(
      valuetable = valuetable,
      common_data = common_data,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      val_frac = val_frac,
      iter_seed = train_seed + iter,
      patch_size = patch_size
    )
  }

  if ("DesignBased" %in% protocols) {
    cat("\n--- DesignBased ---\n")
    iter_results[["DesignBased"]] <- evaluate_design_based_protocol(
      valuetable = valuetable,
      common_data = common_data,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      sample_size = sample_size,
      val_frac = val_frac,
      iter_seed = train_seed + iter,
      patch_size = patch_size
    )
  }

  if ("RandomKFold" %in% protocols) {
    cat("\n--- RandomKFold ---\n")
    iter_results[["RandomKFold"]] <- evaluate_random_kfold_protocol(
      valuetable = valuetable,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      random_k = random_k,
      val_frac = val_frac,
      iter_seed = train_seed + iter,
      patch_size = patch_size
    )
  }

  if ("SpatialKFold" %in% protocols) {
    cat("\n--- SpatialKFold ---\n")
    iter_results[["SpatialKFold"]] <- evaluate_spatial_kfold_protocol(
      valuetable = valuetable,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      val_dist_km = val_dist_km,
      val_frac = val_frac,
      iter_seed = train_seed + iter,
      patch_size = patch_size
    )
  }

  if ("BLOOCV" %in% protocols) {
    cat("\n--- BLOOCV ---\n")
    iter_results[["BLOOCV"]] <- evaluate_bloocv_protocol(
      valuetable = valuetable,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      val_dist_km = val_dist_km,
      val_frac = val_frac,
      iter_seed = train_seed + iter,
      nb_groups = bloo_groups,
      nb_test_pixels = bloo_test_pixels,
      patch_size = patch_size
    )
  }

  all_results[[iter]] <- bind_rows(iter_results) %>%
    mutate(
      iteration = iter,
      scenario = scenario,
      model_profile = model_profile,
      .before = 1
    )

  write.csv(
    bind_rows(all_results),
    file.path(results_dir, "wadoux_style_rf_conv_all_results.csv"),
    row.names = FALSE
  )
}

final_results <- bind_rows(all_results)

summary_by_protocol <- final_results %>%
  group_by(scenario, protocol, model) %>%
  summarise(
    ME_mean = mean(ME, na.rm = TRUE),
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    r2_mean = mean(r2, na.rm = TRUE),
    MEC_mean = mean(MEC, na.rm = TRUE),
    ME_sd = sd(ME, na.rm = TRUE),
    RMSE_sd = sd(RMSE, na.rm = TRUE),
    r2_sd = sd(r2, na.rm = TRUE),
    MEC_sd = sd(MEC, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(protocol, RMSE_mean)

write.csv(final_results, file.path(results_dir, "wadoux_style_rf_conv_all_results.csv"), row.names = FALSE)
write.csv(summary_by_protocol, file.path(results_dir, "wadoux_style_rf_conv_summary_by_protocol.csv"), row.names = FALSE)

cat("\n=== Wadoux-style RF vs ConvKrigingNet2D comparison complete ===\n")
print(summary_by_protocol)
cat(sprintf("\nResults written to: %s\n", results_dir))
