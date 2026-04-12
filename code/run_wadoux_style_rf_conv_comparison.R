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
#   - same final metrics (Wadoux metrics: ME, RMSE, Pearson^2, MEC)
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
#   WADOUX_REFIT_SELECTION_SPLITS
#   WADOUX_RESULTS_DIR
# =============================================================================

parse_bool_env <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}

parse_num_env <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) default else val
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

rename_model_rows <- function(df, model_label) {
  if (is.null(df) || nrow(df) == 0L) {
    return(data.frame())
  }
  df$model <- model_label
  df
}

make_metric_rows_named <- function(obs, pred_map, protocol) {
  rows <- list()

  for (model_label in names(pred_map)) {
    pred <- pred_map[[model_label]]
    if (is.null(pred) || length(pred) == 0L) next
    rows[[length(rows) + 1L]] <- wadoux_eval(obs = obs, pred = pred) %>%
      mutate(model = model_label)
  }

  bind_rows(rows) %>% mutate(protocol = protocol)
}

make_paired_delta_rows <- function(metric_rows, protocol) {
  required_models <- c("RF_400", "GeoVersa_400", "RF_500", "GeoVersa_500")
  present_models <- unique(metric_rows$model)
  if (!all(required_models %in% present_models)) {
    return(data.frame())
  }

  metric_lookup <- split(metric_rows, metric_rows$model)
  comparisons <- list(
    list(lhs = "GeoVersa_400", rhs = "RF_400"),
    list(lhs = "GeoVersa_500", rhs = "RF_500"),
    list(lhs = "GeoVersa_500", rhs = "GeoVersa_400"),
    list(lhs = "RF_500", rhs = "RF_400")
  )

  rows <- lapply(comparisons, function(comp) {
    lhs <- metric_lookup[[comp$lhs]][1, , drop = FALSE]
    rhs <- metric_lookup[[comp$rhs]][1, , drop = FALSE]
    data.frame(
      protocol = protocol,
      comparison = sprintf("%s_vs_%s", comp$lhs, comp$rhs),
      lhs_model = comp$lhs,
      rhs_model = comp$rhs,
      lhs_ME = lhs$ME,
      lhs_RMSE = lhs$RMSE,
      lhs_r2 = lhs$r2,
      lhs_MEC = lhs$MEC,
      rhs_ME = rhs$ME,
      rhs_RMSE = rhs$RMSE,
      rhs_r2 = rhs$r2,
      rhs_MEC = rhs$MEC,
      delta_ME = lhs$ME - rhs$ME,
      delta_RMSE = lhs$RMSE - rhs$RMSE,
      delta_r2 = lhs$r2 - rhs$r2,
      delta_MEC = lhs$MEC - rhs$MEC,
      lhs_better_rmse = lhs$RMSE < rhs$RMSE,
      lhs_better_r2 = lhs$r2 > rhs$r2,
      lhs_better_mec = lhs$MEC > rhs$MEC,
      stringsAsFactors = FALSE
    )
  })

  bind_rows(rows)
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

singleton_df <- function(df) {
  if (nrow(df) < 1L) {
    stop("singleton_df requires at least one row.")
  }
  df[1L, , drop = FALSE]
}

coerce_single_diag_row <- function(diag_df, stage_label) {
  if (is.null(diag_df)) {
    stop(stage_label, " did not return diagnostics.")
  }
  if (!is.data.frame(diag_df)) {
    diag_df <- as.data.frame(diag_df, stringsAsFactors = FALSE)
  }
  if (nrow(diag_df) < 1L) {
    stop(stage_label, " returned an empty diagnostics table.")
  }
  if (nrow(diag_df) > 1L) {
    diag_unique <- unique(diag_df)
    if (nrow(diag_unique) == 1L) {
      cat(sprintf("[GeoVersa refit] %s returned %d duplicated diagnostics rows; collapsing to one.\n",
                  stage_label, nrow(diag_df)))
      diag_df <- diag_unique
    } else {
      cat(sprintf("[GeoVersa refit] WARN: %s returned %d diagnostics rows; using the first row for refit scheduling.\n",
                  stage_label, nrow(diag_df)))
      diag_df <- diag_df[1L, , drop = FALSE]
    }
  }
  diag_df[1L, , drop = FALSE]
}

choose_smoothed_refit_main_epochs <- function(selection_diag, n_refit_train) {
  best_main <- as.integer(selection_diag$best_epoch_main[1])
  main_done <- as.integer(selection_diag$main_epochs_done[1])
  lr_patience <- as.integer(selection_diag$lr_patience[1])
  n_select_train <- as.integer(selection_diag$n_train[1])
  plateau_median <- suppressWarnings(as.integer(selection_diag$plateau_median_epoch_main[1]))
  plateau_last <- suppressWarnings(as.integer(selection_diag$plateau_last_epoch_main[1]))

  if (!is.finite(best_main) || best_main < 1L) {
    best_main <- main_done
  }
  if (!is.finite(main_done) || main_done < best_main) {
    main_done <- best_main
  }
  if (!is.finite(lr_patience) || lr_patience < 0L) {
    lr_patience <- 0L
  }
  if (!is.finite(n_select_train) || n_select_train < 1L) {
    n_select_train <- n_refit_train
  }
  if (!is.finite(plateau_median) || plateau_median < best_main) {
    plateau_median <- best_main
  }
  if (!is.finite(plateau_last) || plateau_last < plateau_median) {
    plateau_last <- plateau_median
  }

  epoch_floor <- as.integer(round(parse_num_env("WADOUX_REFIT_MAIN_EPOCH_FLOOR", 5)))
  trail_frac <- parse_num_env("WADOUX_REFIT_MAIN_TRAIL_FRAC", 0.60)
  lr_buffer_frac <- parse_num_env("WADOUX_REFIT_MAIN_LR_BUFFER_FRAC", 0.50)
  plateau_blend <- parse_num_env("WADOUX_REFIT_MAIN_PLATEAU_BLEND", 0.75)
  plateau_blend <- max(0.0, min(1.0, plateau_blend))

  size_scale <- sqrt(max(1, n_refit_train) / max(1, n_select_train))
  plateau_anchor <- as.numeric(best_main) + plateau_blend * (as.numeric(plateau_last) - as.numeric(best_main))
  scaled_best <- plateau_anchor * size_scale
  buffered_best <- scaled_best + max(1, ceiling(lr_patience * lr_buffer_frac))
  trailing_anchor <- trail_frac * main_done

  refit_main_epochs <- as.integer(round(min(
    main_done,
    max(epoch_floor, buffered_best, trailing_anchor, plateau_median)
  )))

  list(
    refit_main_epochs = refit_main_epochs,
    raw_best_epoch_main = best_main,
    plateau_median_epoch_main = plateau_median,
    plateau_last_epoch_main = plateau_last,
    plateau_anchor_epoch = plateau_anchor,
    scaled_best_epoch_main = scaled_best,
    buffered_best_epoch_main = buffered_best,
    trailing_anchor_epoch = trailing_anchor,
    refit_epoch_floor = epoch_floor,
    refit_size_scale = size_scale,
    refit_n_select_train = n_select_train,
    refit_n_full_train = n_refit_train
  )
}

choose_adaptive_warmstart_policy <- function(selection_diag,
                                             selection_val_rmse_gain,
                                             base_epoch_mult,
                                             base_lr_mult,
                                             scheduled_main_epochs) {
  best_main <- suppressWarnings(as.integer(selection_diag$best_epoch_main[1]))
  main_done <- suppressWarnings(as.integer(selection_diag$main_epochs_done[1]))
  plateau_median <- suppressWarnings(as.integer(selection_diag$plateau_median_epoch_main[1]))
  plateau_last <- suppressWarnings(as.integer(selection_diag$plateau_last_epoch_main[1]))
  nugget_ratio <- suppressWarnings(as.numeric(selection_diag$nugget_ratio[1]))

  if (!is.finite(best_main) || best_main < 1L) {
    best_main <- 1L
  }
  if (!is.finite(main_done) || main_done < best_main) {
    main_done <- best_main
  }
  if (!is.finite(plateau_median) || plateau_median < 1L) {
    plateau_median <- best_main
  }
  if (!is.finite(plateau_last) || plateau_last < plateau_median) {
    plateau_last <- plateau_median
  }
  if (!is.finite(nugget_ratio)) {
    nugget_ratio <- 0.25
  }
  if (!is.finite(selection_val_rmse_gain)) {
    selection_val_rmse_gain <- 0
  }

  policy <- list(
    regime = "baseline",
    epoch_mult = base_epoch_mult,
    lr_mult = base_lr_mult,
    min_main_epochs = NA_integer_
  )

  short_plateau <- best_main <= max(8L, floor(plateau_median * 0.75))
  undertrained_backbone <- (nugget_ratio <= 0.26) &&
    (selection_val_rmse_gain >= 0.35) && (
      short_plateau ||
        scheduled_main_epochs <= max(8L, plateau_median - 2L) ||
        (main_done >= 15L && plateau_last <= max(6L, floor(plateau_median * 0.75)))
    )

  conservative_noisy <- (nugget_ratio >= 0.28) &&
    (selection_val_rmse_gain <= 0.35)

  if (undertrained_backbone) {
    policy$regime <- "backbone_recovery"
    policy$epoch_mult <- max(base_epoch_mult, 0.70)
    policy$lr_mult <- min(base_lr_mult, 0.20)
    policy$min_main_epochs <- max(
      8L,
      min(main_done, max(best_main, plateau_median, plateau_last, 8L))
    )
  } else if (conservative_noisy) {
    policy$regime <- "noisy_conservative"
    policy$epoch_mult <- min(base_epoch_mult, 0.40)
    policy$lr_mult <- min(base_lr_mult, 0.15)
    policy$min_main_epochs <- max(
      5L,
      min(main_done, max(5L, floor(plateau_median * 0.80)))
    )
  }

  policy
}

aggregate_refit_schedule <- function(selection_diags, n_refit_train) {
  if (!is.list(selection_diags) || length(selection_diags) < 1L) {
    stop("aggregate_refit_schedule requires at least one selection diagnostics row.")
  }

  schedules <- lapply(selection_diags, function(diag_row) {
    choose_smoothed_refit_main_epochs(
      selection_diag = diag_row,
      n_refit_train = n_refit_train
    )
  })

  int_median <- function(x, floor = 1L) {
    val <- as.integer(round(stats::median(as.numeric(x), na.rm = TRUE)))
    if (!is.finite(val) || val < floor) floor else val
  }
  num_median <- function(x, fallback) {
    val <- stats::median(as.numeric(x), na.rm = TRUE)
    if (!is.finite(val)) fallback else as.numeric(val)
  }

  main_epochs <- int_median(vapply(schedules, `[[`, numeric(1), "refit_main_epochs"), floor = 1L)
  warmup_epochs <- int_median(vapply(selection_diags, function(x) x$warmup_epochs_done[1], numeric(1)), floor = 1L)
  lr_patience <- int_median(vapply(selection_diags, function(x) x$lr_patience[1], numeric(1)), floor = 1L)
  patience <- int_median(vapply(selection_diags, function(x) x$patience[1], numeric(1)), floor = 1L)
  bank_refresh_every <- int_median(vapply(selection_diags, function(x) x$bank_refresh_every[1], numeric(1)), floor = 1L)
  lr_decay <- num_median(vapply(selection_diags, function(x) x$lr_decay[1], numeric(1)), fallback = 0.5)

  per_split <- data.frame(
    split_id = seq_along(selection_diags),
    warmup_epochs_done = vapply(selection_diags, function(x) x$warmup_epochs_done[1], numeric(1)),
    patience = vapply(selection_diags, function(x) x$patience[1], numeric(1)),
    lr_patience = vapply(selection_diags, function(x) x$lr_patience[1], numeric(1)),
    lr_decay = vapply(selection_diags, function(x) x$lr_decay[1], numeric(1)),
    bank_refresh_every = vapply(selection_diags, function(x) x$bank_refresh_every[1], numeric(1)),
    raw_best_epoch_main = vapply(schedules, `[[`, numeric(1), "raw_best_epoch_main"),
    plateau_median_epoch_main = vapply(schedules, `[[`, numeric(1), "plateau_median_epoch_main"),
    plateau_last_epoch_main = vapply(schedules, `[[`, numeric(1), "plateau_last_epoch_main"),
    plateau_anchor_epoch = vapply(schedules, `[[`, numeric(1), "plateau_anchor_epoch"),
    scaled_best_epoch_main = vapply(schedules, `[[`, numeric(1), "scaled_best_epoch_main"),
    buffered_best_epoch_main = vapply(schedules, `[[`, numeric(1), "buffered_best_epoch_main"),
    trailing_anchor_epoch = vapply(schedules, `[[`, numeric(1), "trailing_anchor_epoch"),
    refit_selected_main_epochs = vapply(schedules, `[[`, numeric(1), "refit_main_epochs"),
    stringsAsFactors = FALSE
  )

  list(
    refit_main_epochs = main_epochs,
    refit_warmup_epochs = warmup_epochs,
    refit_lr_patience = lr_patience,
    refit_patience = max(main_epochs, patience),
    refit_lr_decay = lr_decay,
    refit_bank_refresh_every = bank_refresh_every,
    n_selection_splits = length(selection_diags),
    per_split = per_split,
    raw_best_epoch_main = stats::median(per_split$raw_best_epoch_main, na.rm = TRUE),
    plateau_median_epoch_main = stats::median(per_split$plateau_median_epoch_main, na.rm = TRUE),
    plateau_last_epoch_main = stats::median(per_split$plateau_last_epoch_main, na.rm = TRUE),
    plateau_anchor_epoch = stats::median(per_split$plateau_anchor_epoch, na.rm = TRUE),
    scaled_best_epoch_main = stats::median(per_split$scaled_best_epoch_main, na.rm = TRUE),
    buffered_best_epoch_main = stats::median(per_split$buffered_best_epoch_main, na.rm = TRUE),
    trailing_anchor_epoch = stats::median(per_split$trailing_anchor_epoch, na.rm = TRUE),
    refit_epoch_floor = stats::median(vapply(schedules, `[[`, numeric(1), "refit_epoch_floor"), na.rm = TRUE),
    refit_size_scale = stats::median(vapply(schedules, `[[`, numeric(1), "refit_size_scale"), na.rm = TRUE)
  )
}

get_conv_params_profile <- function(conv_env, profile = c("quick", "full", "n500", "auto"), train_seed = 123L, device = "cpu") {
  profile <- match.arg(profile)
  if (profile == "auto") {
    params <- list(
      kriging_mode = "anisotropic",
      train_seed = train_seed,
      deterministic_batches = TRUE,
      device = device,
      epochs = 80L
    )
  } else if (profile == "quick") {
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
                                              context = NULL,
                                              refit_full_calibration = FALSE,
                                              selection_seed = NULL) {
  prepare_conv_fd_on_explicit_split <- function(train_df, val_df, test_df, params) {
    combined_df <- bind_rows(train_df, val_df, test_df)
    n_train <- nrow(train_df)
    n_val <- nrow(val_df)
    n_test <- nrow(test_df)
    patch_cache_combined <- NULL

    cache_col <- "..patch_cache_row_id"
    train_val_cache_ok <- !is.null(calibration_patch_cache) &&
      all(cache_col %in% names(train_df), cache_col %in% names(val_df)) &&
      !anyNA(train_df[[cache_col]]) &&
      !anyNA(val_df[[cache_col]])
    test_cache_ok <- !is.null(calibration_patch_cache) &&
      cache_col %in% names(test_df) &&
      !anyNA(test_df[[cache_col]])

    if (train_val_cache_ok) {
      train_cache_idx <- as.integer(train_df[[cache_col]])
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

    neighbor_pool_k <- suppressWarnings(as.integer(Sys.getenv("WADOUX_NEIGHBOR_POOL_K", unset = "30")))
    if (!is.finite(neighbor_pool_k) || neighbor_pool_k < 6L) {
      neighbor_pool_k <- 30L
    }
    if (!is.null(params$K_neighbors)) {
      neighbor_pool_k <- max(neighbor_pool_k, as.integer(params$K_neighbors))
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

    list(fd = fd_conv, neighbor_pool_k = neighbor_pool_k)
  }

  run_conv_on_explicit_split <- function(train_df, val_df, test_df, params) {
    prep <- prepare_conv_fd_on_explicit_split(
      train_df = train_df,
      val_df = val_df,
      test_df = test_df,
      params = params
    )

    fd_use <- prep$fd
    conv_out <- do.call(
      conv_env$train_convkrigingnet2d_one_fold,
      c(list(fd = fd_use), params)
    )
    conv_out
  }

  extract_warmstart_shape_overrides <- function(diag_row) {
    if (is.null(diag_row) || nrow(diag_row) == 0L) {
      return(list())
    }
    overrides <- list(
      d = as.integer(diag_row$d[1]),
      patch_dim = as.integer(diag_row$patch_dim[1]),
      coord_dim = as.integer(diag_row$coord_dim[1]),
      tab_dropout = as.numeric(diag_row$tab_dropout[1]),
      patch_dropout = as.numeric(diag_row$patch_dropout[1]),
      coord_dropout = as.numeric(diag_row$coord_dropout[1])
    )
    overrides[vapply(overrides, function(x) length(x) == 1L && is.finite(x) && x > 0, logical(1))]
  }

  out <- list(
    pred_rf = NULL,
    pred_conv = NULL,
    pred_conv_base = NULL,
    pred_conv_val = NULL,
    pred_conv_val_base = NULL,
    conv_diagnostics = NULL
  )

  rf_model <- NULL   # kept in scope for the benchmark baseline only

  if ("RF" %in% models) {
    form_rf <- as.formula(
      paste(response_name_wadoux, "~", paste(predList_modelfull_wadoux, collapse = "+"))
    )
    rf_train_df <- if (refit_full_calibration) bind_rows(train_core, val_df) else train_core
    rf_model <- fit_rf_default_wadoux(rf_train_df, form_rf)
    out$pred_rf <- predict_rf_default_wadoux(rf_model, test_df)
  }

  if ("ConvKrigingNet2D" %in% models) {
    if (is.null(conv_env) || is.null(conv_params)) {
      stop("ConvKrigingNet2D requested but conv_env/conv_params were not provided.")
    }
    if (is.null(context)) {
      context <- conv_env$wadoux_context
    }
    if (!refit_full_calibration) {
      conv_out <- run_conv_on_explicit_split(
        train_df = train_core,
        val_df = val_df,
        test_df = test_df,
        params = conv_params
      )

      out$pred_conv <- conv_out$pred_test
      out$pred_conv_base <- conv_out$pred_test_base
      out$pred_conv_val <- conv_out$pred_val
      out$pred_conv_val_base <- conv_out$pred_val_base
      out$conv_diagnostics <- conv_out$diagnostics
    } else {
      calibration_full <- bind_rows(train_core, val_df)
      selection_val_frac <- min(0.5, max(0.05, nrow(val_df) / max(1L, nrow(calibration_full))))
      selection_split_count <- as.integer(round(parse_num_env("WADOUX_REFIT_SELECTION_SPLITS", 3)))
      refit_warmstart <- parse_bool_env("WADOUX_REFIT_WARMSTART", default = TRUE)
      refit_warmstart_lr_mult <- parse_num_env("WADOUX_REFIT_WARMSTART_LR_MULT", 0.25)
      refit_warmstart_lr_mult <- max(0.05, min(1.0, refit_warmstart_lr_mult))
      refit_warmstart_epoch_mult <- parse_num_env("WADOUX_REFIT_WARMSTART_EPOCH_MULT", 0.50)
      refit_warmstart_epoch_mult <- max(0.20, min(1.0, refit_warmstart_epoch_mult))
      refit_consistency_backbone_recovery <- parse_bool_env("WADOUX_REFIT_CONSISTENCY_BACKBONE_RECOVERY", default = FALSE)
      refit_consistency_weight <- parse_num_env("WADOUX_REFIT_CONSISTENCY_WEIGHT", NA_real_)
      refit_consistency_tab_noise <- parse_num_env("WADOUX_REFIT_CONSISTENCY_TAB_NOISE", 0.02)
      refit_consistency_patch_noise <- parse_num_env("WADOUX_REFIT_CONSISTENCY_PATCH_NOISE", 0.01)
      refit_consistency_coord_noise <- parse_num_env("WADOUX_REFIT_CONSISTENCY_COORD_NOISE", 0.01)
      refit_anchor_backbone_recovery <- parse_bool_env("WADOUX_REFIT_ANCHOR_BACKBONE_RECOVERY", default = TRUE)
      refit_anchor_weight <- parse_num_env("WADOUX_REFIT_ANCHOR_WEIGHT", NA_real_)
      refit_predavg_backbone_recovery <- parse_bool_env("WADOUX_REFIT_PREDAVG_BACKBONE_RECOVERY", default = FALSE)
      refit_predavg_topk <- as.integer(round(parse_num_env("WADOUX_REFIT_PREDAVG_TOPK", 0)))
      if (!is.finite(refit_predavg_topk) || refit_predavg_topk < 1L) {
        refit_predavg_topk <- NA_integer_
      }
      refit_predavg_rel_tol <- parse_num_env("WADOUX_REFIT_PREDAVG_REL_TOL", 0.03)
      refit_predavg_rel_tol <- max(0.0, min(0.20, refit_predavg_rel_tol))
      refit_ckptavg_backbone_recovery <- parse_bool_env("WADOUX_REFIT_CKPTAVG_BACKBONE_RECOVERY", default = TRUE)
      refit_ckptavg_topk <- as.integer(round(parse_num_env("WADOUX_REFIT_CKPTAVG_TOPK", 0)))
      if (!is.finite(refit_ckptavg_topk) || refit_ckptavg_topk < 1L) {
        refit_ckptavg_topk <- NA_integer_
      }
      refit_ckptavg_rel_tol <- parse_num_env("WADOUX_REFIT_CKPTAVG_REL_TOL", 0.03)
      refit_ckptavg_rel_tol <- max(0.0, min(0.20, refit_ckptavg_rel_tol))
      refit_ema_backbone_recovery <- parse_bool_env("WADOUX_REFIT_EMA_BACKBONE_RECOVERY", default = TRUE)
      refit_ema_decay <- parse_num_env("WADOUX_REFIT_EMA_DECAY", 0.65)
      refit_ema_decay <- max(0.05, min(0.95, refit_ema_decay))
      refit_krig_only <- parse_bool_env("WADOUX_REFIT_KRIG_ONLY", default = FALSE)
      refit_krig_only_epoch_mult <- parse_num_env("WADOUX_REFIT_KRIG_ONLY_EPOCH_MULT", 0.50)
      refit_krig_only_epoch_mult <- max(0.20, min(1.50, refit_krig_only_epoch_mult))
      refit_krig_only_lr_mult <- parse_num_env("WADOUX_REFIT_KRIG_ONLY_LR_MULT", 1.0)
      refit_krig_only_lr_mult <- max(0.10, min(2.0, refit_krig_only_lr_mult))
      if (!is.finite(selection_split_count) || selection_split_count < 1L) {
        selection_split_count <- 1L
      }
      selection_seed_base <- if (is.null(selection_seed) || !is.finite(selection_seed)) {
        12345L
      } else {
        as.integer(selection_seed)
      }

      selection_runs <- vector("list", selection_split_count)
      selection_test_df <- if (nrow(val_df) > 0L) singleton_df(val_df) else singleton_df(train_core)
      selection_out <- run_conv_on_explicit_split(
        train_df = train_core,
        val_df = val_df,
        test_df = selection_test_df,
        params = modifyList(
          conv_params,
          list(return_state = refit_warmstart)
        )
      )
      selection_runs[[1L]] <- list(
        split_id = 1L,
        source = "provided",
        out = selection_out,
        val_obs = val_df[[response_name_wadoux]],
        diag = coerce_single_diag_row(
          selection_out$diagnostics,
          stage_label = "GeoVersa selection stage"
        )
      )

      if (selection_split_count > 1L) {
        for (sid in 2:selection_split_count) {
          extra_split <- make_inner_val_split_wadoux(
            calibration_full,
            val_frac = selection_val_frac,
            seed = selection_seed_base + 1000L + sid
          )
          extra_test_df <- if (nrow(extra_split$val) > 0L) singleton_df(extra_split$val) else singleton_df(extra_split$train)
          extra_out <- run_conv_on_explicit_split(
            train_df = extra_split$train,
            val_df = extra_split$val,
            test_df = extra_test_df,
            params = modifyList(
              conv_params,
              list(return_state = FALSE)
            )
          )
          selection_runs[[sid]] <- list(
            split_id = sid,
            source = "resampled_full_calibration",
            out = extra_out,
            val_obs = extra_split$val[[response_name_wadoux]],
            diag = coerce_single_diag_row(
              extra_out$diagnostics,
              stage_label = sprintf("GeoVersa selection stage split %d", sid)
            )
          )
        }
      }

      selection_diags <- lapply(selection_runs, `[[`, "diag")
      selection_val_rmse_gains <- vapply(
        selection_runs,
        function(x) {
          if (is.null(x$out$pred_val) || is.null(x$out$pred_val_base) || is.null(x$val_obs)) {
            return(NA_real_)
          }
          base_rmse <- wadoux_eval(obs = x$val_obs, pred = x$out$pred_val_base)$RMSE
          full_rmse <- wadoux_eval(obs = x$val_obs, pred = x$out$pred_val)$RMSE
          as.numeric(base_rmse - full_rmse)
        },
        numeric(1)
      )
      selection_val_rmse_gain_median <- suppressWarnings(stats::median(selection_val_rmse_gains, na.rm = TRUE))
      if (!is.finite(selection_val_rmse_gain_median)) {
        selection_val_rmse_gain_median <- NA_real_
      }
      refit_schedule <- aggregate_refit_schedule(
        selection_diags = selection_diags,
        n_refit_train = nrow(calibration_full)
      )
      refit_warmup_epochs <- refit_schedule$refit_warmup_epochs
      refit_main_epochs <- refit_schedule$refit_main_epochs
      refit_patience <- refit_schedule$refit_patience
      refit_lr_patience <- refit_schedule$refit_lr_patience
      refit_lr_decay <- refit_schedule$refit_lr_decay
      refit_bank_refresh <- refit_schedule$refit_bank_refresh_every
      refit_warmstart_used <- FALSE
      refit_init_state <- NULL
      refit_warmstart_lr <- NA_real_
      refit_warmstart_shape_source <- "full_calibration_auto"
      refit_consistency_active <- FALSE
      refit_anchor_active <- FALSE
      refit_predavg_active <- FALSE
      refit_ckptavg_active <- FALSE
      refit_ema_active <- FALSE
      refit_ema_start_epoch <- NA_integer_
      refit_krig_only_epochs <- 0L
      refit_krig_only_lr <- NA_real_

      if (refit_warmstart && !is.null(selection_out$selected_state)) {
        selection_anchor_diag <- selection_runs[[1L]]$diag
        adaptive_warmstart <- choose_adaptive_warmstart_policy(
          selection_diag = selection_anchor_diag,
          selection_val_rmse_gain = selection_val_rmse_gain_median,
          base_epoch_mult = refit_warmstart_epoch_mult,
          base_lr_mult = refit_warmstart_lr_mult,
          scheduled_main_epochs = refit_main_epochs
        )
        selection_lr_init <- suppressWarnings(as.numeric(selection_anchor_diag$lr_init[1]))
        selection_lr_final <- suppressWarnings(as.numeric(selection_anchor_diag$lr_final[1]))
        warmstart_lr_candidate <- selection_lr_init * refit_warmstart_lr_mult
        if (!is.finite(warmstart_lr_candidate) || warmstart_lr_candidate <= 0) {
          warmstart_lr_candidate <- selection_lr_final
        } else if (is.finite(selection_lr_final) && selection_lr_final > 0) {
          warmstart_lr_candidate <- max(warmstart_lr_candidate, selection_lr_final)
        }

        refit_shape_overrides <- extract_warmstart_shape_overrides(selection_anchor_diag)
        refit_warmup_epochs <- 0L
        refit_main_epochs <- max(3L, as.integer(round(refit_main_epochs * refit_warmstart_epoch_mult)))
        refit_init_state <- selection_out$selected_state
        refit_warmstart_lr <- warmstart_lr_candidate
        refit_warmstart_shape_source <- "selection_model"
        refit_warmstart_used <- TRUE
        refit_consistency_active <- refit_consistency_backbone_recovery &&
          identical(adaptive_warmstart$regime, "backbone_recovery")
        refit_predavg_active <- !refit_consistency_active && refit_predavg_backbone_recovery &&
          identical(adaptive_warmstart$regime, "backbone_recovery")
        refit_anchor_active <- !refit_consistency_active && !refit_predavg_active && refit_anchor_backbone_recovery &&
          identical(adaptive_warmstart$regime, "backbone_recovery")
        refit_ckptavg_active <- !refit_consistency_active && !refit_predavg_active && !refit_anchor_active &&
          refit_ckptavg_backbone_recovery &&
          identical(adaptive_warmstart$regime, "backbone_recovery")
        refit_ema_active <- !refit_consistency_active && !refit_predavg_active && !refit_anchor_active && !refit_ckptavg_active &&
          refit_ema_backbone_recovery &&
          identical(adaptive_warmstart$regime, "backbone_recovery")
        if (refit_ema_active) {
          refit_ema_start_epoch <- max(2L, as.integer(round(refit_main_epochs * 0.50)))
        }
        if (refit_krig_only) {
          refit_krig_only_epochs <- max(2L, as.integer(round(refit_main_epochs * refit_krig_only_epoch_mult)))
          refit_krig_only_lr <- refit_warmstart_lr * refit_krig_only_lr_mult
        }
      } else {
        refit_shape_overrides <- list()
        adaptive_warmstart <- list(
          regime = "disabled",
          epoch_mult = refit_warmstart_epoch_mult,
          lr_mult = refit_warmstart_lr_mult,
          min_main_epochs = NA_integer_
        )
      }

      cat(sprintf(
        "[GeoVersa refit] Selection aggregation: splits=%d | warmup=%d | main=%d | lr_patience=%d | lr_decay=%.3f | bank_refresh=%d\n",
        refit_schedule$n_selection_splits,
        refit_warmup_epochs,
        refit_main_epochs,
        refit_lr_patience,
        refit_lr_decay,
        refit_bank_refresh
      ))
      if (refit_warmstart_used) {
        cat(sprintf(
          "[GeoVersa refit] Warm-start active: lr=%.2e | epoch_mult=%.2f | regime_signal=%s | consistency=%s | predavg=%s | anchor=%s | ckptavg=%s | EMA=%s | shape_source=%s | krig_only=%s (%d ep @ %.2e)\n",
          refit_warmstart_lr,
          refit_warmstart_epoch_mult,
          adaptive_warmstart$regime,
          if (refit_consistency_active) {
            sprintf(
              "on(w=%s,tab=%.3f,patch=%.3f,coord=%.3f)",
              if (is.finite(refit_consistency_weight) && refit_consistency_weight > 0) sprintf("%.3f", refit_consistency_weight) else "auto",
              refit_consistency_tab_noise,
              refit_consistency_patch_noise,
              refit_consistency_coord_noise
            )
          } else "off",
          if (refit_predavg_active) {
            sprintf("on(topk=%s,tol=%.2f)", if (is.na(refit_predavg_topk)) "auto" else as.character(refit_predavg_topk), refit_predavg_rel_tol)
          } else "off",
          if (refit_anchor_active) {
            if (is.finite(refit_anchor_weight) && refit_anchor_weight > 0) sprintf("on(%.3f)", refit_anchor_weight) else "on(auto)"
          } else "off",
          if (refit_ckptavg_active) "on" else "off",
          if (refit_ema_active) sprintf("on(start=%d,decay=%.2f)", refit_ema_start_epoch, refit_ema_decay) else "off",
          refit_warmstart_shape_source,
          if (refit_krig_only && refit_krig_only_epochs > 0L) "on" else "off",
          refit_krig_only_epochs,
          if (is.finite(refit_krig_only_lr)) refit_krig_only_lr else NA_real_
        ))
      }

      refit_params <- modifyList(
        conv_params,
        list(
          warmup_epochs = refit_warmup_epochs,
          epochs = refit_main_epochs,
          patience = refit_patience,
          lr_patience = refit_lr_patience,
          lr_decay = refit_lr_decay,
          bank_refresh_every = refit_bank_refresh,
          refit_fixed_warmup_epochs = refit_warmup_epochs,
          refit_fixed_main_epochs = refit_main_epochs,
          refit_use_final_state = TRUE,
          init_state = refit_init_state,
          lr = if (refit_warmstart_used) refit_warmstart_lr else conv_params$lr,
          min_lr = if (refit_warmstart_used && is.finite(refit_warmstart_lr)) refit_warmstart_lr / 1000 else conv_params$min_lr,
          refit_consistency_active = refit_consistency_active,
          refit_consistency_weight = refit_consistency_weight,
          refit_consistency_tab_noise = refit_consistency_tab_noise,
          refit_consistency_patch_noise = refit_consistency_patch_noise,
          refit_consistency_coord_noise = refit_consistency_coord_noise,
          refit_anchor_active = refit_anchor_active,
          refit_anchor_weight = refit_anchor_weight,
          refit_predavg_active = refit_predavg_active,
          refit_predavg_topk = refit_predavg_topk,
          refit_predavg_rel_tol = refit_predavg_rel_tol,
          refit_ckptavg_active = refit_ckptavg_active,
          refit_ckptavg_topk = refit_ckptavg_topk,
          refit_ckptavg_rel_tol = refit_ckptavg_rel_tol,
          refit_ema_active = refit_ema_active,
          refit_ema_decay = refit_ema_decay,
          refit_ema_start_epoch = refit_ema_start_epoch,
          refit_krig_only_epochs = refit_krig_only_epochs,
          refit_krig_only_lr = refit_krig_only_lr
        )
      )
      refit_params <- modifyList(refit_params, refit_shape_overrides)

      refit_out <- run_conv_on_explicit_split(
        train_df = calibration_full,
        val_df = singleton_df(calibration_full),
        test_df = test_df,
        params = refit_params
      )

      out$pred_conv <- refit_out$pred_test
      out$pred_conv_base <- refit_out$pred_test_base
      out$pred_conv_val <- selection_out$pred_val
      out$pred_conv_val_base <- selection_out$pred_val_base
      out$conv_diagnostics <- cbind(
        refit_out$diagnostics,
        data.frame(
          refit_full_calibration = TRUE,
          refit_calibration_n = nrow(calibration_full),
          selection_train_n = nrow(train_core),
          selection_val_n = nrow(val_df),
          selection_splits_used = refit_schedule$n_selection_splits,
          selection_canonical_best_epoch_main = as.integer(selection_runs[[1L]]$diag$best_epoch_main[1]),
          selection_canonical_best_epoch_total = as.integer(selection_runs[[1L]]$diag$best_epoch_total[1]),
          selection_warmup_epochs_done = as.integer(round(stats::median(vapply(selection_diags, function(x) x$warmup_epochs_done[1], numeric(1)), na.rm = TRUE))),
          selection_main_epochs_done = as.integer(round(stats::median(vapply(selection_diags, function(x) x$main_epochs_done[1], numeric(1)), na.rm = TRUE))),
          selection_best_epoch_main = as.integer(round(stats::median(vapply(selection_diags, function(x) x$best_epoch_main[1], numeric(1)), na.rm = TRUE))),
          selection_best_epoch_total = as.integer(round(stats::median(vapply(selection_diags, function(x) x$best_epoch_total[1], numeric(1)), na.rm = TRUE))),
          selection_patience = refit_patience,
          selection_lr_patience = refit_lr_patience,
          selection_lr_decay = refit_lr_decay,
          selection_bank_refresh_every = refit_bank_refresh,
          selection_best_val_huber = as.numeric(stats::median(vapply(selection_diags, function(x) x$best_val_huber[1], numeric(1)), na.rm = TRUE)),
          selection_val_RMSE_gain = selection_val_rmse_gain_median,
          refit_epoch_rule = if (refit_schedule$n_selection_splits > 1L) "smoothed_multi_split_median" else "smoothed_single_split",
          refit_raw_best_epoch_main = refit_schedule$raw_best_epoch_main,
          refit_plateau_median_epoch_main = refit_schedule$plateau_median_epoch_main,
          refit_plateau_last_epoch_main = refit_schedule$plateau_last_epoch_main,
          refit_plateau_anchor_epoch = refit_schedule$plateau_anchor_epoch,
          refit_scaled_best_epoch_main = refit_schedule$scaled_best_epoch_main,
          refit_buffered_best_epoch_main = refit_schedule$buffered_best_epoch_main,
          refit_trailing_anchor_epoch = refit_schedule$trailing_anchor_epoch,
          refit_epoch_floor = refit_schedule$refit_epoch_floor,
          refit_size_scale = refit_schedule$refit_size_scale,
          refit_selected_main_epochs = refit_main_epochs,
          refit_selected_warmup_epochs = refit_warmup_epochs,
          refit_selected_lr_patience = refit_lr_patience,
          refit_selected_lr_decay = refit_lr_decay,
          refit_selected_bank_refresh = refit_bank_refresh,
          refit_warmstart = refit_warmstart_used,
          refit_warmstart_lr = refit_warmstart_lr,
          refit_warmstart_lr_mult = if (refit_warmstart_used) refit_warmstart_lr_mult else NA_real_,
          refit_warmstart_epoch_mult = if (refit_warmstart_used) refit_warmstart_epoch_mult else NA_real_,
          refit_warmstart_shape_source = refit_warmstart_shape_source,
          refit_warmstart_regime = adaptive_warmstart$regime,
          refit_warmstart_min_main_epochs = if (identical(adaptive_warmstart$regime, "backbone_recovery")) max(8L, refit_main_epochs) else NA_integer_,
          refit_warmstart_schedule_mode = "plateau_base",
          refit_consistency_requested = refit_consistency_active,
          refit_consistency_requested_weight = if (refit_consistency_active && is.finite(refit_consistency_weight) && refit_consistency_weight > 0) refit_consistency_weight else NA_real_,
          refit_consistency_requested_tab_noise = if (refit_consistency_active) refit_consistency_tab_noise else NA_real_,
          refit_consistency_requested_patch_noise = if (refit_consistency_active) refit_consistency_patch_noise else NA_real_,
          refit_consistency_requested_coord_noise = if (refit_consistency_active) refit_consistency_coord_noise else NA_real_,
          refit_anchor_requested = refit_anchor_active,
          refit_anchor_requested_weight = if (refit_anchor_active && is.finite(refit_anchor_weight) && refit_anchor_weight > 0) refit_anchor_weight else NA_real_,
          refit_predavg_requested = refit_predavg_active,
          refit_predavg_requested_topk = if (refit_predavg_active) refit_predavg_topk else NA_integer_,
          refit_predavg_requested_rel_tol = if (refit_predavg_active) refit_predavg_rel_tol else NA_real_,
          refit_ckptavg_requested = refit_ckptavg_active,
          refit_ckptavg_requested_topk = if (refit_ckptavg_active) refit_ckptavg_topk else NA_integer_,
          refit_ckptavg_requested_rel_tol = if (refit_ckptavg_active) refit_ckptavg_rel_tol else NA_real_,
          refit_ema_requested = refit_ema_active,
          refit_ema_requested_decay = if (refit_ema_active) refit_ema_decay else NA_real_,
          refit_ema_requested_start_epoch = if (refit_ema_active) refit_ema_start_epoch else NA_integer_,
          refit_krig_only = refit_krig_only && refit_krig_only_epochs > 0L,
          refit_krig_only_epochs = refit_krig_only_epochs,
          refit_krig_only_lr = refit_krig_only_lr,
          selection_refit_main_epochs_sd = stats::sd(refit_schedule$per_split$refit_selected_main_epochs),
          stringsAsFactors = FALSE
        )
      )
    }
  }

  out
}

empty_conv_diag_df <- function() {
  data.frame()
}

make_conv_diagnostic_row <- function(preds,
                                     val_obs,
                                     test_obs,
                                     protocol,
                                     split_id = 1L) {
  if (is.null(preds$conv_diagnostics) ||
      is.null(preds$pred_conv) ||
      is.null(preds$pred_conv_base) ||
      is.null(preds$pred_conv_val) ||
      is.null(preds$pred_conv_val_base)) {
    return(empty_conv_diag_df())
  }

  val_base <- wadoux_eval(obs = val_obs, pred = preds$pred_conv_val_base)
  val_full <- wadoux_eval(obs = val_obs, pred = preds$pred_conv_val)
  test_base <- wadoux_eval(obs = test_obs, pred = preds$pred_conv_base)
  test_full <- wadoux_eval(obs = test_obs, pred = preds$pred_conv)

  cbind(
    data.frame(
      protocol = protocol,
      split_id = split_id,
      model = "ConvKrigingNet2D",
      stringsAsFactors = FALSE
    ),
    preds$conv_diagnostics,
    setNames(val_base, paste0("val_", names(val_base), "_base")),
    setNames(val_full, paste0("val_", names(val_full), "_full")),
    setNames(test_base, paste0("test_", names(test_base), "_base")),
    setNames(test_full, paste0("test_", names(test_full), "_full")),
    data.frame(
      val_RMSE_gain = val_base$RMSE - val_full$RMSE,
      val_r2_gain = val_full$r2 - val_base$r2,
      val_MEC_gain = val_full$MEC - val_base$MEC,
      test_RMSE_gain = test_base$RMSE - test_full$RMSE,
      test_r2_gain = test_full$r2 - test_base$r2,
      test_MEC_gain = test_full$MEC - test_base$MEC,
      stringsAsFactors = FALSE
    )
  )
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
                                           patch_size = 15,
                                           refit_full_calibration = FALSE) {
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
    patch_size = patch_size,
    refit_full_calibration = refit_full_calibration,
    selection_seed = iter_seed + 101L
  )

  obs <- val_srs[[response_name_wadoux]]
  list(
    metrics = make_metric_rows(obs = obs, preds = preds, models = models, protocol = "DesignBased"),
    diagnostics = make_conv_diagnostic_row(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = obs,
      protocol = "DesignBased",
      split_id = 1L
    )
  )
}

evaluate_design_based_protocol_paired <- function(valuetable,
                                                  common_data,
                                                  calibration_patch_cache,
                                                  conv_env,
                                                  conv_params,
                                                  sample_size,
                                                  val_frac,
                                                  iter_seed,
                                                  patch_size = 15) {
  inner <- make_inner_val_split_wadoux(valuetable, val_frac = val_frac, seed = iter_seed + 101L)
  val_srs <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)

  preds_400 <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    test_df = val_srs,
    models = c("RF", "ConvKrigingNet2D"),
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
    patch_size = patch_size,
    refit_full_calibration = FALSE,
    selection_seed = iter_seed + 101L
  )

  preds_500 <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    test_df = val_srs,
    models = c("RF", "ConvKrigingNet2D"),
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
    patch_size = patch_size,
    refit_full_calibration = TRUE,
    selection_seed = iter_seed + 101L
  )

  obs <- val_srs[[response_name_wadoux]]
  metrics <- make_metric_rows_named(
    obs = obs,
    pred_map = list(
      RF_400 = preds_400$pred_rf,
      GeoVersa_400 = preds_400$pred_conv,
      RF_500 = preds_500$pred_rf,
      GeoVersa_500 = preds_500$pred_conv
    ),
    protocol = "DesignBased"
  )

  diagnostics <- bind_rows(
    rename_model_rows(
      make_conv_diagnostic_row(
        preds = preds_400,
        val_obs = inner$val[[response_name_wadoux]],
        test_obs = obs,
        protocol = "DesignBased",
        split_id = 1L
      ),
      "GeoVersa_400"
    ),
    rename_model_rows(
      make_conv_diagnostic_row(
        preds = preds_500,
        val_obs = inner$val[[response_name_wadoux]],
        test_obs = obs,
        protocol = "DesignBased",
        split_id = 1L
      ),
      "GeoVersa_500"
    )
  )

  paired_deltas <- make_paired_delta_rows(metrics, protocol = "DesignBased")
  paired_meta <- data.frame(
    protocol = "DesignBased",
    calibration_n = nrow(valuetable),
    train_core_n = nrow(inner$train),
    val_n = nrow(inner$val),
    test_n = nrow(val_srs),
    stringsAsFactors = FALSE
  )

  list(
    metrics = metrics,
    diagnostics = diagnostics,
    paired_deltas = paired_deltas,
    paired_meta = paired_meta
  )
}

evaluate_random_kfold_protocol <- function(valuetable,
                                           models,
                                           calibration_patch_cache,
                                           conv_env,
                                           conv_params,
                                           random_k,
                                           val_frac,
                                           iter_seed,
                                           patch_size = 15,
                                           refit_full_calibration = FALSE) {
  flds <- caret::createFolds(
    valuetable[[response_name_wadoux]],
    k = random_k,
    list = TRUE,
    returnTrain = FALSE
  )

  pred_rf_all <- c()
  pred_conv_all <- c()
  obs_all <- c()
  diag_rows <- vector("list", length(flds))

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
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration,
      selection_seed = iter_seed * 1000L + j
    )

    obs_all <- c(obs_all, outer_test[[response_name_wadoux]])
    if ("RF" %in% models && !is.null(preds$pred_rf)) {
      pred_rf_all <- c(pred_rf_all, preds$pred_rf)
    }
    if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
      pred_conv_all <- c(pred_conv_all, preds$pred_conv)
    }
    diag_rows[[j]] <- make_conv_diagnostic_row(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = outer_test[[response_name_wadoux]],
      protocol = "RandomKFold",
      split_id = j
    )
  }

  list(
    metrics = make_metric_rows(
      obs = obs_all,
      preds = list(pred_rf = pred_rf_all, pred_conv = pred_conv_all),
      models = models,
      protocol = "RandomKFold"
    ),
    diagnostics = bind_rows(diag_rows)
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
                                            patch_size = 15,
                                            refit_full_calibration = FALSE) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  groups <- sort(unique(spatial_folds))

  pred_rf_all <- c()
  pred_conv_all <- c()
  obs_all <- c()
  diag_rows <- vector("list", length(groups))

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
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration,
      selection_seed = iter_seed * 1000L + j
    )

    obs_all <- c(obs_all, outer_test[[response_name_wadoux]])
    if ("RF" %in% models && !is.null(preds$pred_rf)) {
      pred_rf_all <- c(pred_rf_all, preds$pred_rf)
    }
    if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
      pred_conv_all <- c(pred_conv_all, preds$pred_conv)
    }
    diag_rows[[j]] <- make_conv_diagnostic_row(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = outer_test[[response_name_wadoux]],
      protocol = "SpatialKFold",
      split_id = j
    )
  }

  list(
    metrics = make_metric_rows(
      obs = obs_all,
      preds = list(pred_rf = pred_rf_all, pred_conv = pred_conv_all),
      models = models,
      protocol = "SpatialKFold"
    ),
    diagnostics = bind_rows(diag_rows)
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
                                     patch_size = 15,
                                     refit_full_calibration = FALSE) {
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
  diag_rows <- vector("list", nb_iteration)

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
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration,
      selection_seed = iter_seed * 100000L + j
    )

    rows[[j]] <- data.frame(
      task = j,
      obs = outer_test[[response_name_wadoux]],
      pred_rf = if ("RF" %in% models) preds$pred_rf else NA_real_,
      pred_conv = if ("ConvKrigingNet2D" %in% models) preds$pred_conv else NA_real_
    )
    diag_rows[[j]] <- make_conv_diagnostic_row(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = outer_test[[response_name_wadoux]],
      protocol = "BLOOCV",
      split_id = j
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

  list(
    metrics = bind_rows(rows_out) %>% mutate(protocol = "BLOOCV"),
    diagnostics = bind_rows(diag_rows)
  )
}

evaluate_population_protocol_optional <- function(valuetable,
                                                  common_data,
                                                  models,
                                                  calibration_patch_cache,
                                                  conv_env,
                                                  conv_params,
                                                  val_frac,
                                                  iter_seed,
                                                  patch_size = 15,
                                                  refit_full_calibration = FALSE) {
  inner <- make_inner_val_split_wadoux(valuetable, val_frac = val_frac, seed = iter_seed + 101L)
  preds <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    test_df = common_data$s_df,
    models = models,
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
    patch_size = patch_size,
    refit_full_calibration = refit_full_calibration,
    selection_seed = iter_seed + 101L
  )
  obs <- common_data$s_df[[response_name_wadoux]]
  list(
    metrics = make_metric_rows(obs = obs, preds = preds, models = models, protocol = "Population"),
    diagnostics = make_conv_diagnostic_row(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = obs,
      protocol = "Population",
      split_id = 1L
    )
  )
}

scenario <- Sys.getenv("WADOUX_SCENARIO", unset = "random")
protocols <- parse_protocols_env()
models <- parse_models_env()
include_population <- parse_bool_env("WADOUX_INCLUDE_POPULATION", default = FALSE)
refit_full_calibration <- parse_bool_env("WADOUX_REFIT_FULL_CALIBRATION", default = FALSE)
paired_final_benchmark <- parse_bool_env("WADOUX_PAIRED_FINAL_BENCHMARK", default = FALSE)
if (include_population && !"Population" %in% protocols) {
  protocols <- c("Population", protocols)
}
if (paired_final_benchmark) {
  if (!all(c("RF", "ConvKrigingNet2D") %in% models)) {
    stop("WADOUX_PAIRED_FINAL_BENCHMARK requires both RF and ConvKrigingNet2D in WADOUX_MODELS.")
  }
  if (!"DesignBased" %in% protocols) {
    stop("WADOUX_PAIRED_FINAL_BENCHMARK currently supports DesignBased and requires WADOUX_PROTOCOLS to include DesignBased.")
  }
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
refit_selection_splits <- as.integer(round(parse_num_env("WADOUX_REFIT_SELECTION_SPLITS", 3)))

if ((refit_full_calibration || paired_final_benchmark) && !identical(model_profile, "auto")) {
  stop("WADOUX_REFIT_FULL_CALIBRATION and WADOUX_PAIRED_FINAL_BENCHMARK require WADOUX_MODEL_PROFILE=auto because the refit path needs return_state from Auto v5.")
}

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
  default_auto_v5_path <- file.path(project_root_wadoux, "code", "ConvKrigingNet2D_Auto_v5.R")
  default_auto_path <- file.path(project_root_wadoux, "code", "ConvKrigingNet2D_Auto.R")
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
  if (nchar(auto_v5_path) == 0 && identical(model_profile, "auto") && file.exists(default_auto_v5_path)) {
    auto_v5_path <- default_auto_v5_path
  }
  if (nchar(auto_v5_path) > 0 && file.exists(auto_v5_path)) {
    sys.source(auto_v5_path, envir = conv_env)
    # Also load v4 base (v5 builds on top)
    auto_path <- Sys.getenv("WADOUX_AUTO_SCRIPT", unset = "")
    if (nchar(auto_path) == 0 && file.exists(default_auto_path)) {
      auto_path <- default_auto_path
    }
    if (nchar(auto_path) > 0 && file.exists(auto_path)) {
      sys.source(auto_path, envir = conv_env)
    }
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_auto_one_fold_v5
    cat("[Auto v5] Overriding train function with COMPLETE AUTO-CONFIG v5\n")
  } else {
    # Fall back to v4 Auto
    auto_path <- Sys.getenv("WADOUX_AUTO_SCRIPT", unset = "")
    if (nchar(auto_path) == 0 && identical(model_profile, "auto") && file.exists(default_auto_path)) {
      auto_path <- default_auto_path
    }
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
  # Ablation override: WADOUX_ALPHA_ME
  ablation_alpha_me <- Sys.getenv("WADOUX_ALPHA_ME", unset = "")
  if (nchar(ablation_alpha_me) > 0) {
    conv_params$alpha_me <- as.numeric(ablation_alpha_me)
    cat(sprintf("[Ablation] alpha_me overridden → %.4f\n", conv_params$alpha_me))
  }
  # Ablation override: WADOUX_LAMBDA_COV
  ablation_lambda_cov <- Sys.getenv("WADOUX_LAMBDA_COV", unset = "")
  if (nchar(ablation_lambda_cov) > 0) {
    conv_params$lambda_cov <- as.numeric(ablation_lambda_cov)
    cat(sprintf("[Ablation] lambda_cov overridden → %.5f\n", conv_params$lambda_cov))
  }
  # Ablation override: WADOUX_LR
  ablation_lr <- Sys.getenv("WADOUX_LR", unset = "")
  if (nchar(ablation_lr) > 0) {
    conv_params$lr <- as.numeric(ablation_lr)
    cat(sprintf("[Ablation] lr overridden → %.2e\n", conv_params$lr))
  }
  # Ablation override: WADOUX_MIN_LR
  ablation_min_lr <- Sys.getenv("WADOUX_MIN_LR", unset = "")
  if (nchar(ablation_min_lr) > 0) {
    conv_params$min_lr <- as.numeric(ablation_min_lr)
    cat(sprintf("[Ablation] min_lr overridden → %.2e\n", conv_params$min_lr))
  }
  # Ablation override: WADOUX_WEIGHT_DECAY
  ablation_wd <- Sys.getenv("WADOUX_WEIGHT_DECAY", unset = "")
  if (nchar(ablation_wd) > 0) {
    conv_params$wd <- as.numeric(ablation_wd)
    cat(sprintf("[Ablation] wd overridden → %.2e\n", conv_params$wd))
  }
  # Ablation override: WADOUX_BATCH_SIZE
  ablation_bs <- Sys.getenv("WADOUX_BATCH_SIZE", unset = "")
  if (nchar(ablation_bs) > 0) {
    conv_params$batch_size <- as.integer(ablation_bs)
    cat(sprintf("[Ablation] batch_size overridden → %dL\n", conv_params$batch_size))
  }
  # Ablation override: WADOUX_BANK_REFRESH_EVERY
  ablation_bre <- Sys.getenv("WADOUX_BANK_REFRESH_EVERY", unset = "")
  if (nchar(ablation_bre) > 0) {
    conv_params$bank_refresh_every <- as.integer(ablation_bre)
    cat(sprintf("[Ablation] bank_refresh_every overridden → %dL\n", conv_params$bank_refresh_every))
  }
  # Ablation override: WADOUX_PATIENCE
  ablation_pat <- Sys.getenv("WADOUX_PATIENCE", unset = "")
  if (nchar(ablation_pat) > 0) {
    conv_params$patience <- as.integer(ablation_pat)
    cat(sprintf("[Ablation] patience overridden → %dL\n", conv_params$patience))
  }
  # Ablation override: WADOUX_LR_PATIENCE
  ablation_lr_pat <- Sys.getenv("WADOUX_LR_PATIENCE", unset = "")
  if (nchar(ablation_lr_pat) > 0) {
    conv_params$lr_patience <- as.integer(ablation_lr_pat)
    cat(sprintf("[Ablation] lr_patience overridden → %dL\n", conv_params$lr_patience))
  }
  # Ablation override: WADOUX_LR_DECAY
  ablation_lr_decay <- Sys.getenv("WADOUX_LR_DECAY", unset = "")
  if (nchar(ablation_lr_decay) > 0) {
    conv_params$lr_decay <- as.numeric(ablation_lr_decay)
    cat(sprintf("[Ablation] lr_decay overridden → %.2f\n", conv_params$lr_decay))
  }
  # Ablation override: WADOUX_K_NEIGHBORS
  ablation_k_neighbors <- Sys.getenv("WADOUX_K_NEIGHBORS", unset = "")
  if (nchar(ablation_k_neighbors) > 0) {
    conv_params$K_neighbors <- as.integer(ablation_k_neighbors)
    cat(sprintf("[Ablation] K_neighbors overridden → %dL\n", conv_params$K_neighbors))
  }
  # Ablation override: WADOUX_BETA_INIT
  ablation_beta_init <- Sys.getenv("WADOUX_BETA_INIT", unset = "")
  if (nchar(ablation_beta_init) > 0) {
    conv_params$beta_init <- as.numeric(ablation_beta_init)
    cat(sprintf("[Ablation] beta_init overridden → %.4f\n", conv_params$beta_init))
  }
  # Ablation override: architecture dimensions and dropouts
  ablation_patch_dim   <- Sys.getenv("WADOUX_PATCH_DIM",    unset = "")
  ablation_d           <- Sys.getenv("WADOUX_D",            unset = "")
  ablation_tab_drop    <- Sys.getenv("WADOUX_TAB_DROPOUT",  unset = "")
  ablation_patch_drop  <- Sys.getenv("WADOUX_PATCH_DROPOUT",unset = "")
  ablation_coord_dim   <- Sys.getenv("WADOUX_COORD_DIM",    unset = "")
  ablation_coord_drop  <- Sys.getenv("WADOUX_COORD_DROPOUT",unset = "")
  if (nchar(ablation_patch_dim)  > 0) { conv_params$patch_dim    <- as.integer(ablation_patch_dim);  cat(sprintf("[Ablation] patch_dim    → %d\n",    conv_params$patch_dim))    }
  if (nchar(ablation_d)          > 0) { conv_params$d             <- as.integer(ablation_d);          cat(sprintf("[Ablation] d           → %d\n",    conv_params$d))            }
  if (nchar(ablation_tab_drop)   > 0) { conv_params$tab_dropout   <- as.numeric(ablation_tab_drop);   cat(sprintf("[Ablation] tab_dropout → %.2f\n", conv_params$tab_dropout))   }
  if (nchar(ablation_patch_drop) > 0) { conv_params$patch_dropout <- as.numeric(ablation_patch_drop); cat(sprintf("[Ablation] patch_drop  → %.2f\n", conv_params$patch_dropout)) }
  if (nchar(ablation_coord_dim)  > 0) { conv_params$coord_dim     <- as.integer(ablation_coord_dim);  cat(sprintf("[Ablation] coord_dim   → %d\n",    conv_params$coord_dim))    }
  if (nchar(ablation_coord_drop) > 0) { conv_params$coord_dropout <- as.numeric(ablation_coord_drop); cat(sprintf("[Ablation] coord_drop  → %.2f\n", conv_params$coord_dropout)) }
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
  refit_full_calibration = refit_full_calibration,
  paired_final_benchmark = paired_final_benchmark,
  sample_size = sample_size,
  n_iter = n_iter,
  val_frac = val_frac,
  val_dist_km = val_dist_km,
  random_k = random_k,
  refit_selection_splits = refit_selection_splits,
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
all_diagnostics <- vector("list", n_iter)
all_paired_deltas <- vector("list", n_iter)
all_paired_meta <- vector("list", n_iter)

summarise_results_by_protocol <- function(results_df) {
  if (nrow(results_df) == 0) {
    return(data.frame())
  }
  results_df %>%
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
}

summarise_diagnostics <- function(diagnostics_df) {
  if (nrow(diagnostics_df) == 0) {
    return(data.frame())
  }
  diagnostics_df %>%
    group_by(scenario, protocol, model) %>%
    summarise(
      val_RMSE_base_mean = mean(val_RMSE_base, na.rm = TRUE),
      val_RMSE_full_mean = mean(val_RMSE_full, na.rm = TRUE),
      val_RMSE_gain_mean = mean(val_RMSE_gain, na.rm = TRUE),
      test_RMSE_base_mean = mean(test_RMSE_base, na.rm = TRUE),
      test_RMSE_full_mean = mean(test_RMSE_full, na.rm = TRUE),
      test_RMSE_gain_mean = mean(test_RMSE_gain, na.rm = TRUE),
      val_r2_gain_mean = mean(val_r2_gain, na.rm = TRUE),
      test_r2_gain_mean = mean(test_r2_gain, na.rm = TRUE),
      val_MEC_gain_mean = mean(val_MEC_gain, na.rm = TRUE),
      test_MEC_gain_mean = mean(test_MEC_gain, na.rm = TRUE),
      beta_final_mean = mean(beta_final, na.rm = TRUE),
      beta_final_sd = sd(beta_final, na.rm = TRUE),
      K_neighbors_mean = mean(K_neighbors, na.rm = TRUE),
      nugget_ratio_mean = mean(nugget_ratio, na.rm = TRUE),
      mean_abs_delta_scaled_test_mean = mean(mean_abs_delta_scaled_test, na.rm = TRUE),
      mean_abs_delta_scaled_test_sd = sd(mean_abs_delta_scaled_test, na.rm = TRUE),
      warmup_epochs_done_mean = mean(warmup_epochs_done, na.rm = TRUE),
      best_epoch_total_mean = mean(best_epoch_total, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(protocol)
}

summarise_paired_deltas <- function(paired_deltas_df) {
  if (nrow(paired_deltas_df) == 0) {
    return(data.frame())
  }
  paired_deltas_df %>%
    group_by(scenario, protocol, comparison, lhs_model, rhs_model) %>%
    summarise(
      delta_ME_mean = mean(delta_ME, na.rm = TRUE),
      delta_RMSE_mean = mean(delta_RMSE, na.rm = TRUE),
      delta_r2_mean = mean(delta_r2, na.rm = TRUE),
      delta_MEC_mean = mean(delta_MEC, na.rm = TRUE),
      delta_ME_sd = sd(delta_ME, na.rm = TRUE),
      delta_RMSE_sd = sd(delta_RMSE, na.rm = TRUE),
      delta_r2_sd = sd(delta_r2, na.rm = TRUE),
      delta_MEC_sd = sd(delta_MEC, na.rm = TRUE),
      lhs_better_rmse_rate = mean(lhs_better_rmse, na.rm = TRUE),
      lhs_better_r2_rate = mean(lhs_better_r2, na.rm = TRUE),
      lhs_better_mec_rate = mean(lhs_better_mec, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(protocol, comparison)
}

write_iteration_checkpoint_outputs <- function(results_dir,
                                             all_results,
                                             all_diagnostics,
                                             all_paired_deltas,
                                             all_paired_meta,
                                             completed_iteration,
                                             total_iterations) {
  final_results <- bind_rows(all_results)
  final_diagnostics <- bind_rows(all_diagnostics)
  final_paired_deltas <- bind_rows(all_paired_deltas)
  final_paired_meta <- bind_rows(all_paired_meta)

  summary_by_protocol <- summarise_results_by_protocol(final_results)
  diagnostic_summary <- summarise_diagnostics(final_diagnostics)
  paired_delta_summary <- summarise_paired_deltas(final_paired_deltas)

  unlink(file.path(results_dir, c(
    "wadoux_style_rf_conv_fold_diagnostics.csv",
    "wadoux_style_rf_conv_fold_diagnostic_summary.csv",
    "wadoux_style_rf_conv_paired_deltas.csv",
    "wadoux_style_rf_conv_paired_meta.csv",
    "wadoux_style_rf_conv_paired_delta_summary.csv"
  )), force = TRUE)

  write.csv(final_results, file.path(results_dir, "wadoux_style_rf_conv_all_results.csv"), row.names = FALSE)
  write.csv(summary_by_protocol, file.path(results_dir, "wadoux_style_rf_conv_summary_by_protocol.csv"), row.names = FALSE)
  if (nrow(final_diagnostics) > 0) {
    write.csv(final_diagnostics, file.path(results_dir, "wadoux_style_rf_conv_fold_diagnostics.csv"), row.names = FALSE)
    write.csv(diagnostic_summary, file.path(results_dir, "wadoux_style_rf_conv_fold_diagnostic_summary.csv"), row.names = FALSE)
  }
  if (nrow(final_paired_deltas) > 0) {
    write.csv(final_paired_deltas, file.path(results_dir, "wadoux_style_rf_conv_paired_deltas.csv"), row.names = FALSE)
    write.csv(final_paired_meta, file.path(results_dir, "wadoux_style_rf_conv_paired_meta.csv"), row.names = FALSE)
    write.csv(paired_delta_summary, file.path(results_dir, "wadoux_style_rf_conv_paired_delta_summary.csv"), row.names = FALSE)
  }
  write.csv(
    data.frame(
      completed_iteration = completed_iteration,
      total_iterations = total_iterations,
      completed_at = format(Sys.time(), tz = "", usetz = TRUE),
      stringsAsFactors = FALSE
    ),
    file.path(results_dir, "wadoux_style_rf_conv_checkpoint_status.csv"),
    row.names = FALSE
  )
}

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
  iter_diagnostics <- list()

  if ("Population" %in% protocols) {
    cat("\n--- Population ---\n")
    eval_out <- evaluate_population_protocol_optional(
      valuetable = valuetable,
      common_data = common_data,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      val_frac = val_frac,
      iter_seed = train_seed + iter,
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration
    )
    iter_results[["Population"]] <- eval_out$metrics
    iter_diagnostics[["Population"]] <- eval_out$diagnostics
  }

  if ("DesignBased" %in% protocols) {
    cat("\n--- DesignBased ---\n")
    eval_out <- if (paired_final_benchmark) {
      evaluate_design_based_protocol_paired(
        valuetable = valuetable,
        common_data = common_data,
        calibration_patch_cache = calibration_patch_cache,
        conv_env = conv_env,
        conv_params = conv_params,
        sample_size = sample_size,
        val_frac = val_frac,
        iter_seed = train_seed + iter,
        patch_size = patch_size
      )
    } else {
      evaluate_design_based_protocol(
        valuetable = valuetable,
        common_data = common_data,
        models = models,
        calibration_patch_cache = calibration_patch_cache,
        conv_env = conv_env,
        conv_params = conv_params,
        sample_size = sample_size,
        val_frac = val_frac,
        iter_seed = train_seed + iter,
        patch_size = patch_size,
        refit_full_calibration = refit_full_calibration
      )
    }
    iter_results[["DesignBased"]] <- eval_out$metrics
    iter_diagnostics[["DesignBased"]] <- eval_out$diagnostics
    all_paired_deltas[[iter]] <- if (!is.null(eval_out$paired_deltas) && nrow(eval_out$paired_deltas) > 0) {
      eval_out$paired_deltas %>%
        mutate(
          iteration = iter,
          scenario = scenario,
          model_profile = model_profile,
          .before = 1
        )
    } else {
      data.frame()
    }
    all_paired_meta[[iter]] <- if (!is.null(eval_out$paired_meta) && nrow(eval_out$paired_meta) > 0) {
      eval_out$paired_meta %>%
        mutate(
          iteration = iter,
          scenario = scenario,
          model_profile = model_profile,
          .before = 1
        )
    } else {
      data.frame()
    }
  }

  if ("RandomKFold" %in% protocols) {
    cat("\n--- RandomKFold ---\n")
    eval_out <- evaluate_random_kfold_protocol(
      valuetable = valuetable,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      random_k = random_k,
      val_frac = val_frac,
      iter_seed = train_seed + iter,
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration
    )
    iter_results[["RandomKFold"]] <- eval_out$metrics
    iter_diagnostics[["RandomKFold"]] <- eval_out$diagnostics
  }

  if ("SpatialKFold" %in% protocols) {
    cat("\n--- SpatialKFold ---\n")
    eval_out <- evaluate_spatial_kfold_protocol(
      valuetable = valuetable,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      val_dist_km = val_dist_km,
      val_frac = val_frac,
      iter_seed = train_seed + iter,
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration
    )
    iter_results[["SpatialKFold"]] <- eval_out$metrics
    iter_diagnostics[["SpatialKFold"]] <- eval_out$diagnostics
  }

  if ("BLOOCV" %in% protocols) {
    cat("\n--- BLOOCV ---\n")
    eval_out <- evaluate_bloocv_protocol(
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
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration
    )
    iter_results[["BLOOCV"]] <- eval_out$metrics
    iter_diagnostics[["BLOOCV"]] <- eval_out$diagnostics
  }

  all_results[[iter]] <- bind_rows(iter_results) %>%
    mutate(
      iteration = iter,
      scenario = scenario,
      model_profile = model_profile,
      .before = 1
    )

  iter_diag_df <- bind_rows(iter_diagnostics)
  if (nrow(iter_diag_df) > 0) {
    all_diagnostics[[iter]] <- iter_diag_df %>%
      mutate(
        iteration = iter,
        scenario = scenario,
        model_profile = model_profile,
        .before = 1
      )
  } else {
    all_diagnostics[[iter]] <- data.frame()
  }

  write_iteration_checkpoint_outputs(
    results_dir = results_dir,
    all_results = all_results,
    all_diagnostics = all_diagnostics,
    all_paired_deltas = all_paired_deltas,
    all_paired_meta = all_paired_meta,
    completed_iteration = iter,
    total_iterations = n_iter
  )
}

final_results <- bind_rows(all_results)
final_diagnostics <- bind_rows(all_diagnostics)
final_paired_deltas <- bind_rows(all_paired_deltas)
final_paired_meta <- bind_rows(all_paired_meta)

summary_by_protocol <- summarise_results_by_protocol(final_results)
diagnostic_summary <- summarise_diagnostics(final_diagnostics)
paired_delta_summary <- summarise_paired_deltas(final_paired_deltas)

write_iteration_checkpoint_outputs(
  results_dir = results_dir,
  all_results = all_results,
  all_diagnostics = all_diagnostics,
  all_paired_deltas = all_paired_deltas,
  all_paired_meta = all_paired_meta,
  completed_iteration = n_iter,
  total_iterations = n_iter
)

cat("\n=== Wadoux-style RF vs ConvKrigingNet2D comparison complete ===\n")
print(summary_by_protocol)
if (nrow(diagnostic_summary) > 0) {
  cat("\n=== GeoVersa fold diagnostic summary ===\n")
  print(diagnostic_summary)
}
if (nrow(paired_delta_summary) > 0) {
  cat("\n=== Paired delta summary ===\n")
  print(paired_delta_summary)
}
cat(sprintf("\nResults written to: %s\n", results_dir))
