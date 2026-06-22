# =============================================================================
# run_gpboost_wadoux_comparison.R
#
# Head-to-head benchmark under Wadoux (2021) protocols:
#
#   - RF        (ranger defaults — Wadoux baseline)
#   - Cubist    (rule-based, classical DSM)
#   - GPBoost   (Sigrist 2022 GP boosting, defined in code/GPBoostDSM.R)
#
# Mirrors run_geobart_wadoux_comparison.R structure (no torch, no patches).
#
# Environment overrides:
#   GPB_SCENARIO         random | regular_grid | clustered_random
#   GPB_PROTOCOLS        comma list of protocols
#   GPB_MODELS           comma list of {RF, Cubist, GPBoost}
#   GPB_N_ITER           default 5
#   GPB_SAMPLE_SIZE      default 500
#   GPB_VAL_FRAC         default 0.20
#   GPB_VAL_DIST_KM      default 350
#   GPB_RANDOM_K         default 10
#   GPB_BLOO_GROUPS      default 10
#   GPB_BLOO_TEST_PIXELS default 100
#   GPB_BLOO_GPB_PIXELS  cap GPBoost retrains in BLOOCV
#   GPB_TRAIN_SEED       default 123
#   GPB_RESULTS_DIR      default results/gpboost_run
#   GPB_NROUNDS_MAX, GPB_LEARNING_RATE, GPB_MAX_DEPTH, GPB_NUM_LEAVES,
#   GPB_FEATURE_FRACTION, GPB_BAGGING_FRACTION, GPB_MIN_DATA_IN_LEAF,
#   GPB_LAMBDA_L2, GPB_MATERN_NU
#   GPB_TARGET_TRANSFORM identity (default) | auto | log
#   GPB_LEAVES_NEWTON default TRUE
#   GPB_USE_GP_VAL    default FALSE
#   GPB_TREE_FALLBACK default TRUE
#   GPB_FAIR_TUNE     default TRUE
#   GPB_REFIT_FULL    default TRUE
#   GPB_TUNE_INNER_FOLDS default 3; set 1 for legacy single holdout
#   GPB_TUNE_LR_GRID, GPB_TUNE_DEPTH_GRID, GPB_TUNE_MATERN_GRID
#   GPB_TUNE_FEATURE_GRID, GPB_TUNE_MIN_LEAF_GRID,
#   GPB_TUNE_BAGGING_GRID, GPB_TUNE_L2_GRID,
#   GPB_RANGE_INIT_GRID, GPB_NUGGET_INIT_GRID
#   GPB_MAX_TUNE_CANDIDATES default 72 when GPB_TUNE_FULL_GRID=FALSE
#   GPB_TUNE_FULL_GRID default FALSE; set TRUE only to expand the candidate
#                      universe. AutoTune v2 still stages evaluation unless
#                      GPB_AUTOTUNE_V2=FALSE.
#   GPB_AUTOTUNE_V2 default TRUE; staged screen + refine search
#   GPB_STAGE1_MAX_CANDIDATES, GPB_STAGE2_TOP_K, GPB_STAGE2_MAX_CANDIDATES
#   GPB_CRASH_SAFE default TRUE; prevents native spatial GPBoost segfaults
#   GPB_SPATIAL_POLICY tree_only (default) | mixed | spatial_only | auto
#   GPB_SELECTION_OBJECTIVE rmse | paired_rf_delta
#   GPB_POLICY_BLEND TRUE/FALSE; GPB_BLEND_ALPHA_GRID
#   GPB_RESIDUAL_SPATIAL_CORRECTION TRUE/FALSE; GPB_RESIDUAL_GAMMA_GRID
#   GPB_SEED_ENSEMBLE_N refit seeds per ensemble member
#   GPB_SPATIAL_FILTER TRUE/FALSE
#   GPB_TUNE_MAX_SECONDS optional wall-clock cap per GPBoost fit
#   GPB_DUAN_*           passed through to GPBoostDSM.R
#   GPB_CHECKPOINT_EVERY default 1
# =============================================================================

rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")
source("./code/GPBoostDSM.R")

suppressPackageStartupMessages({
  if (requireNamespace("Cubist",  quietly = TRUE)) library(Cubist)
  if (requireNamespace("ranger",  quietly = TRUE)) library(ranger)
  if (requireNamespace("gpboost", quietly = TRUE)) library(gpboost)
})

if (!requireNamespace("gpboost", quietly = TRUE)) {
  stop("Install gpboost first: install.packages('gpboost').")
}

# ---- Env-var parsing --------------------------------------------------------

parse_bool_gpb <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}
parse_num_gpb <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  v   <- suppressWarnings(as.numeric(raw))
  if (!is.finite(v)) default else v
}
parse_list_gpb <- function(name, default) {
  raw <- Sys.getenv(name, unset = paste(default, collapse = ","))
  v   <- trimws(unlist(strsplit(raw, ",")))
  v[v != ""]
}
parse_num_list_gpb <- function(name, default = NULL) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  v <- suppressWarnings(as.numeric(trimws(unlist(strsplit(raw, ",")))))
  v[is.finite(v)]
}
raw_env_gpb <- function(name, default = "") {
  Sys.getenv(name, unset = default)
}

# ---- Config -----------------------------------------------------------------

scenario        <- Sys.getenv("GPB_SCENARIO", unset = "random")
protocols       <- parse_list_gpb("GPB_PROTOCOLS",
                                   c("DesignBased", "RandomKFold",
                                     "SpatialKFold", "BLOOCV"))
models          <- parse_list_gpb("GPB_MODELS",
                                   c("RF", "Cubist", "GPBoost"))
allowed_models  <- c("RF", "Cubist", "GPBoost")
invalid         <- setdiff(models, allowed_models)
if (length(invalid) > 0L) {
  stop("Unsupported GPB_MODELS entries: ", paste(invalid, collapse = ", "))
}
include_population <- "Population" %in% protocols

n_iter          <- as.integer(parse_num_gpb("GPB_N_ITER",          5L))
sample_size     <- as.integer(parse_num_gpb("GPB_SAMPLE_SIZE",     500L))
val_frac        <-             parse_num_gpb("GPB_VAL_FRAC",       0.20)
val_dist_km     <-             parse_num_gpb("GPB_VAL_DIST_KM",    350)
random_k        <- as.integer(parse_num_gpb("GPB_RANDOM_K",        10L))
bloo_groups     <- as.integer(parse_num_gpb("GPB_BLOO_GROUPS",     10L))
bloo_pixels     <- as.integer(parse_num_gpb("GPB_BLOO_TEST_PIXELS", 100L))
train_seed      <- as.integer(parse_num_gpb("GPB_TRAIN_SEED",      123L))
iter_offset     <- as.integer(parse_num_gpb("GPB_ITER_OFFSET",      0L))
results_dir     <- Sys.getenv("GPB_RESULTS_DIR",
                               unset = "results/gpboost_run")
checkpoint_every <- as.integer(parse_num_gpb("GPB_CHECKPOINT_EVERY", 1L))
gpb_tune_full_grid <- parse_bool_gpb("GPB_TUNE_FULL_GRID", FALSE)
gpb_save_prediction_audit <- parse_bool_gpb("GPB_SAVE_PREDICTION_AUDIT", TRUE)

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
options(wadoux_corr_method = "pearson")

gpb_progress_file <- file.path(results_dir, "gpboost_tuning_progress_live.csv")
if (parse_bool_gpb("GPB_ALLOW_EXTERNAL_PROGRESS_FILE", FALSE)) {
  progress_env <- Sys.getenv("GPB_TUNE_PROGRESS_FILE", unset = "")
  if (nzchar(progress_env)) gpb_progress_file <- progress_env
}
Sys.setenv(GPB_TUNE_PROGRESS_FILE = gpb_progress_file)

# GPBoost params. Drop NULL overrides so the default preset remains active.
gpboost_overrides <- list(
  train_seed      = train_seed,
  nrounds_max     = if (nzchar(Sys.getenv("GPB_NROUNDS_MAX", "")))
                      as.integer(parse_num_gpb("GPB_NROUNDS_MAX", NA)) else NULL,
  learning_rate   = if (nzchar(Sys.getenv("GPB_LEARNING_RATE", "")))
                      parse_num_gpb("GPB_LEARNING_RATE", NA) else NULL,
  max_depth       = if (nzchar(Sys.getenv("GPB_MAX_DEPTH", "")))
                      as.integer(parse_num_gpb("GPB_MAX_DEPTH", NA)) else NULL,
  num_leaves      = if (nzchar(Sys.getenv("GPB_NUM_LEAVES", "")))
                      as.integer(parse_num_gpb("GPB_NUM_LEAVES", NA)) else NULL,
  feature_fraction = if (nzchar(Sys.getenv("GPB_FEATURE_FRACTION", "")))
                      parse_num_gpb("GPB_FEATURE_FRACTION", NA) else NULL,
  bagging_fraction = if (nzchar(Sys.getenv("GPB_BAGGING_FRACTION", "")))
                      parse_num_gpb("GPB_BAGGING_FRACTION", NA) else NULL,
  min_data_in_leaf = if (nzchar(Sys.getenv("GPB_MIN_DATA_IN_LEAF", "")))
                      as.integer(parse_num_gpb("GPB_MIN_DATA_IN_LEAF", NA)) else NULL,
  lambda_l2        = if (nzchar(Sys.getenv("GPB_LAMBDA_L2", "")))
                      parse_num_gpb("GPB_LAMBDA_L2", NA) else NULL,
  matern_nu       = if (nzchar(Sys.getenv("GPB_MATERN_NU", "")))
                      parse_num_gpb("GPB_MATERN_NU", NA) else NULL,
  early_stop      = if (nzchar(Sys.getenv("GPB_EARLY_STOP", "")))
                      as.integer(parse_num_gpb("GPB_EARLY_STOP", NA)) else NULL,
  leaves_newton_update = if (nzchar(Sys.getenv("GPB_LEAVES_NEWTON", "")))
                           parse_bool_gpb("GPB_LEAVES_NEWTON", TRUE) else NULL,
  use_gp_model_for_validation = if (nzchar(Sys.getenv("GPB_USE_GP_VAL", "")))
                                  parse_bool_gpb("GPB_USE_GP_VAL", TRUE) else NULL,
  tree_fallback = if (nzchar(Sys.getenv("GPB_TREE_FALLBACK", "")))
                    parse_bool_gpb("GPB_TREE_FALLBACK", TRUE) else NULL,
  fair_tune = if (nzchar(Sys.getenv("GPB_FAIR_TUNE", "")))
                parse_bool_gpb("GPB_FAIR_TUNE", TRUE) else NULL,
  refit_full = if (nzchar(Sys.getenv("GPB_REFIT_FULL", "")))
                 parse_bool_gpb("GPB_REFIT_FULL", TRUE) else NULL,
  num_threads = if (nzchar(Sys.getenv("GPB_NUM_THREADS", "")))
                  as.integer(parse_num_gpb("GPB_NUM_THREADS", NA)) else NULL,
  tune_inner_folds = as.integer(parse_num_gpb("GPB_TUNE_INNER_FOLDS", 3L)),
  learning_rate_grid = parse_num_list_gpb("GPB_TUNE_LR_GRID", NULL),
  max_depth_grid = parse_num_list_gpb("GPB_TUNE_DEPTH_GRID", NULL),
  matern_nu_grid = parse_num_list_gpb("GPB_TUNE_MATERN_GRID", NULL),
  feature_fraction_grid = parse_num_list_gpb("GPB_TUNE_FEATURE_GRID", NULL),
  min_data_in_leaf_grid = parse_num_list_gpb("GPB_TUNE_MIN_LEAF_GRID", NULL),
  bagging_fraction_grid = parse_num_list_gpb("GPB_TUNE_BAGGING_GRID", NULL),
  lambda_l2_grid = parse_num_list_gpb("GPB_TUNE_L2_GRID", NULL),
  range_init_grid = parse_num_list_gpb("GPB_RANGE_INIT_GRID", NULL),
  nugget_init_grid = parse_num_list_gpb("GPB_NUGGET_INIT_GRID", NULL),
  max_tune_candidates = if (isTRUE(gpb_tune_full_grid)) {
                          Inf
                        } else {
                          as.integer(parse_num_gpb("GPB_MAX_TUNE_CANDIDATES", 72L))
                        },
  autotune_v2 = parse_bool_gpb("GPB_AUTOTUNE_V2", TRUE),
  stage1_max_candidates = as.integer(parse_num_gpb(
    "GPB_STAGE1_MAX_CANDIDATES", 96L)),
  stage2_top_k = as.integer(parse_num_gpb("GPB_STAGE2_TOP_K", 12L)),
  stage2_max_candidates = as.integer(parse_num_gpb(
    "GPB_STAGE2_MAX_CANDIDATES", 48L)),
  stage2_select_only = parse_bool_gpb("GPB_STAGE2_SELECT_ONLY", FALSE),
  crash_safe = parse_bool_gpb("GPB_CRASH_SAFE", TRUE),
  spatial_policy = Sys.getenv("GPB_SPATIAL_POLICY", unset = "tree_only"),
  rmse_sd_weight = parse_num_gpb("GPB_RMSE_SD_WEIGHT", 0.25),
  quantile_bins = as.integer(parse_num_gpb("GPB_QUANTILE_BINS", 5L)),
  quantile_excess_weight = parse_num_gpb("GPB_QUANTILE_EXCESS_WEIGHT", 0.15),
  quantile_sd_weight = parse_num_gpb("GPB_QUANTILE_SD_WEIGHT", 0.05),
  ensemble_top_k = as.integer(parse_num_gpb("GPB_ENSEMBLE_TOP_K", 1L)),
  ensemble_weighting = Sys.getenv("GPB_ENSEMBLE_WEIGHTING", unset = "softmax"),
  ensemble_temperature = parse_num_gpb("GPB_ENSEMBLE_TEMPERATURE", 0.75),
	  selection_objective = Sys.getenv("GPB_SELECTION_OBJECTIVE", unset = "paired_rf_delta"),
  policy_blend = parse_bool_gpb("GPB_POLICY_BLEND", FALSE),
  blend_alpha_grid = parse_num_list_gpb("GPB_BLEND_ALPHA_GRID",
                                        c(0, 0.10, 0.25, 0.50, 0.75, 1)),
  residual_spatial_correction = parse_bool_gpb(
    "GPB_RESIDUAL_SPATIAL_CORRECTION", FALSE),
	  residual_gamma_grid = parse_num_list_gpb("GPB_RESIDUAL_GAMMA_GRID",
	                                           c(0, 0.25, 0.50, 0.75, 1)),
	  seed_ensemble_n = as.integer(parse_num_gpb("GPB_SEED_ENSEMBLE_N", 1L)),
	  noharm_tree_anchor = parse_bool_gpb("GPB_NOHARM_TREE_ANCHOR", TRUE),
	  noharm_margin_abs = parse_num_gpb("GPB_NOHARM_MARGIN_ABS", 0.05),
	  noharm_margin_rel = parse_num_gpb("GPB_NOHARM_MARGIN_REL", 0.0),
	  noharm_tree_only_ensemble = parse_bool_gpb(
	    "GPB_NOHARM_TREE_ONLY_ENSEMBLE", TRUE),
	  rf_noharm_policy = parse_bool_gpb("GPB_RF_NOHARM_POLICY", TRUE),
	  rf_noharm_margin_abs = parse_num_gpb("GPB_RF_NOHARM_MARGIN_ABS", 0.05),
	  rf_noharm_margin_rel = parse_num_gpb("GPB_RF_NOHARM_MARGIN_REL", 0.0),
	  rf_noharm_tree_only_ensemble = parse_bool_gpb(
	    "GPB_RF_NOHARM_TREE_ONLY_ENSEMBLE", TRUE),
	  spatial_filter = parse_bool_gpb("GPB_SPATIAL_FILTER", TRUE),
  spatial_filter_max_alpha_no_signal = parse_num_gpb(
    "GPB_SPATIAL_FILTER_MAX_ALPHA_NO_SIGNAL", 0.25),
  spatial_filter_max_gamma_no_signal = parse_num_gpb(
    "GPB_SPATIAL_FILTER_MAX_GAMMA_NO_SIGNAL", 0.25),
  tune_max_seconds = parse_num_gpb("GPB_TUNE_MAX_SECONDS", Inf),
  tune_min_candidates_before_timeout = as.integer(parse_num_gpb(
    "GPB_TUNE_MIN_CANDIDATES_BEFORE_TIMEOUT", 12L)),
  tune_progress_file = gpb_progress_file
)
gpboost_overrides <- gpboost_overrides[!vapply(gpboost_overrides, is.null, logical(1))]
gpboost_params <- modifyList(gpboost_dsm_default_params, gpboost_overrides)
tt_override <- Sys.getenv("GPB_TARGET_TRANSFORM", unset = "")
if (nzchar(tt_override)) {
  v <- tolower(trimws(tt_override))
  if (v %in% c("auto", "identity", "log")) gpboost_params$target_transform <- v
}

cat(sprintf(
  "[GPBoost runner] scenario=%s  protocols=%s  models=%s  n_iter=%d  n=%d  autotune_v2=%s  stage1=%d  stage2=%d  crash_safe=%s  spatial_policy=%s  objective=%s  progress=%s\n",
  scenario, paste(protocols, collapse = ","),
  paste(models, collapse = ","), n_iter, sample_size,
  isTRUE(gpboost_params$autotune_v2),
  gpboost_params$stage1_max_candidates,
  gpboost_params$stage2_max_candidates,
  isTRUE(gpboost_params$crash_safe),
  gpboost_params$spatial_policy,
  gpboost_params$selection_objective,
  gpboost_params$tune_progress_file))

# ---- Data -------------------------------------------------------------------

need_polygon <- scenario %in% c("regular_grid", "clustered_random")
common_data  <- load_wadoux_common_data(include_polygon = need_polygon)
predList     <- predList_modelfull_wadoux
form_rf      <- as.formula(
  paste(response_name_wadoux, "~", paste(predList, collapse = "+"))
)

# ---- Sampling helpers (mirror NMK/DKP/GeoBART) ------------------------------

sample_calibration_gpb <- function(seed = NULL) {
  restore_rng <- .gpb_preserve_global_rng()
  on.exit(restore_rng(), add = TRUE)
  if (!is.null(seed) && is.finite(seed)) set.seed(as.integer(seed))
  if (scenario == "random") {
    sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  } else if (scenario == "regular_grid") {
    sample_valuetable_regular_grid_wadoux(
      common_data = common_data, sample_size = sample_size)
  } else if (scenario == "clustered_random") {
    sample_valuetable_clustered_random_wadoux(
      common_data    = common_data,
      kmeans_centers = as.integer(parse_num_gpb("GPB_KMEANS_CENTERS", 100L)),
      n_psu          = as.integer(parse_num_gpb("GPB_N_PSU",          20L)),
      m_per_psu      = as.integer(parse_num_gpb("GPB_M_PER_PSU",      25L)))
  } else stop("Unsupported GPB_SCENARIO: ", scenario)
}

make_inner_val_split_gpb <- function(train_df, val_frac = 0.2, seed = NULL) {
  restore_rng <- .gpb_preserve_global_rng()
  on.exit(restore_rng(), add = TRUE)
  if (!is.null(seed)) set.seed(seed)
  n_train <- nrow(train_df)
  if (n_train < 10L) stop("Outer training set too small for GPBoost.")
  val_size  <- max(2L, floor(n_train * val_frac))
  val_size  <- min(val_size, n_train - 2L)
  val_idx   <- sample.int(n_train, size = val_size, replace = FALSE)
  train_idx <- setdiff(seq_len(n_train), val_idx)
  list(train = train_df[train_idx, , drop = FALSE],
       val   = train_df[val_idx,   , drop = FALSE])
}

prepare_gpboost_fold <- function(train_df, val_df, test_df,
                                 full_train_df = NULL) {
  full_x <- full_coords <- full_y <- NULL
  if (!is.null(full_train_df)) {
    full_x <- as.matrix(full_train_df[, predList, drop = FALSE])
    full_coords <- as.matrix(full_train_df[, c("x", "y"), drop = FALSE])
    full_y <- full_train_df[[response_name_wadoux]]
  }
  list(
    X = list(
      train = as.matrix(train_df[, predList,    drop = FALSE]),
      val   = as.matrix(val_df  [, predList,    drop = FALSE]),
      test  = as.matrix(test_df [, predList,    drop = FALSE]),
      full  = full_x
    ),
    coords = list(
      train = as.matrix(train_df[, c("x", "y"), drop = FALSE]),
      val   = as.matrix(val_df  [, c("x", "y"), drop = FALSE]),
      test  = as.matrix(test_df [, c("x", "y"), drop = FALSE]),
      full  = full_coords
    ),
    y = list(
      train = train_df[[response_name_wadoux]],
      val   = val_df  [[response_name_wadoux]],
      test  = test_df [[response_name_wadoux]],
      full  = full_y
    )
  )
}

run_gpboost_on_split <- function(train_core, val_df, test_df,
                                 full_train_df = NULL) {
  fd <- prepare_gpboost_fold(train_core, val_df, test_df,
                             full_train_df = full_train_df)
  do.call(train_gpboost_dsm_one_fold, c(list(fd = fd), gpboost_params))
}

diagnostics_row_gpb <- function(out, iter, protocol) {
  if (is.null(out$diagnostics)) return(data.frame())
  cbind(iteration = iter, protocol = protocol,
        as.data.frame(out$diagnostics, stringsAsFactors = FALSE))
}

row_id_gpb <- function(df) {
  if ("..wadoux_row_id" %in% names(df)) {
    df$..wadoux_row_id
  } else {
    seq_len(nrow(df))
  }
}

coord_key_gpb <- function(df) {
  if (!all(c("x", "y") %in% names(df))) {
    return(rep(NA_character_, nrow(df)))
  }
  paste0(round(as.numeric(df$x), 6), "|", round(as.numeric(df$y), 6))
}

sample_design_test_gpb <- function(calibration_full, size, seed = NULL) {
  restore_rng <- .gpb_preserve_global_rng()
  on.exit(restore_rng(), add = TRUE)
  if (!is.null(seed) && is.finite(seed)) set.seed(as.integer(seed))
  pool <- common_data$s_df
  keep <- rep(TRUE, nrow(pool))

  if ("..wadoux_row_id" %in% names(pool) &&
      "..wadoux_row_id" %in% names(calibration_full)) {
    keep <- keep & !(pool$..wadoux_row_id %in% calibration_full$..wadoux_row_id)
  }

  pool_xy <- coord_key_gpb(pool)
  cal_xy <- coord_key_gpb(calibration_full)
  if (any(!is.na(pool_xy)) && any(!is.na(cal_xy))) {
    keep <- keep & !(pool_xy %in% cal_xy)
  }

  eligible <- pool[keep, , drop = FALSE]
  if (nrow(eligible) < size) {
    stop(sprintf(
      "DesignBased test pool has only %d eligible rows after excluding calibration rows; need %d.",
      nrow(eligible), size
    ))
  }
  sample_simple_random_rows_wadoux(eligible, size)
}

prediction_audit_designbased_gpb <- function(test_df, pred_map, iter,
                                             diag = NULL) {
  if (!isTRUE(gpb_save_prediction_audit)) return(data.frame())
  obs <- test_df[[response_name_wadoux]]
  out <- data.frame(
    iteration = iter,
    scenario = scenario,
    protocol = "DesignBased",
    split_role = "design_test",
    wadoux_row_id = row_id_gpb(test_df),
    x = test_df$x,
    y = test_df$y,
    obs = obs,
    stringsAsFactors = FALSE
  )
  for (nm in names(pred_map)) {
    pred <- as.numeric(pred_map[[nm]])
    out[[paste0("pred_", nm)]] <- pred
    out[[paste0("resid_", nm)]] <- pred - obs
  }
  if (!is.null(diag)) {
    fields <- c(
      "target_transform", "fair_tune", "refit_full", "refit_status",
      "autotune_v2", "crash_safe", "requested_spatial_policy",
      "spatial_policy", "stage1_max_candidates", "stage2_top_k",
      "stage2_max_candidates", "stage2_select_only",
      "rmse_sd_weight", "quantile_bins", "quantile_excess_weight",
      "quantile_sd_weight", "selection_objective", "rf_inner_rmse",
      "policy_blend", "blend_alpha_grid", "selected_policy_blend_alpha",
      "residual_spatial_correction", "residual_gamma_grid",
	      "selected_residual_gamma", "selected_synthetic_tree_candidate",
	      "selected_synthetic_spatial_candidate", "selected_residual_status",
	      "selected_residual_spatial_recommended", "seed_ensemble_n",
		      "noharm_tree_anchor", "noharm_preselected_candidate",
		      "noharm_anchor_candidate", "noharm_anchor_score",
		      "noharm_anchor_val_rmse", "noharm_margin_abs",
		      "noharm_margin_rel", "noharm_margin", "noharm_delta_vs_anchor",
		      "noharm_applied", "noharm_tree_only_ensemble", "noharm_reason",
		      "rf_noharm_policy", "rf_noharm_preselected_candidate",
		      "rf_noharm_preselected_delta_rmse_vs_rf",
		      "rf_noharm_anchor_candidate",
		      "rf_noharm_anchor_delta_rmse_vs_rf",
		      "rf_noharm_margin_abs", "rf_noharm_margin_rel",
		      "rf_noharm_margin", "rf_noharm_required_delta_rmse_vs_rf",
		      "rf_noharm_failed", "rf_noharm_applied",
		      "rf_noharm_tree_only_ensemble", "rf_noharm_reason",
		      "spatial_filter", "tune_candidates_evaluated",
	      "tune_candidate_universe", "tune_source", "tune_inner_folds",
	      "tune_fold_sizes", "selected_candidate", "selected_tune_stage",
	      "selected_selection_score", "selected_delta_rmse_vs_rf",
	      "selected_delta_RMSE_vs_RF",
	      "selected_learning_rate",
      "selected_max_depth", "selected_num_leaves", "selected_matern_nu",
      "selected_feature_fraction", "selected_min_data_in_leaf",
      "selected_bagging_fraction", "selected_lambda_l2",
      "selected_cv_rmse_mean", "selected_cv_rmse_sd",
      "selected_quantile_rmse_mean", "selected_quantile_rmse_sd",
      "selected_quantile_rmse_excess", "selected_quantile_rmse",
      "selected_quantile_bins", "best_iter", "refit_best_iter",
      "duan_applied", "duan_reason", "val_me", "val_rmse"
    )
    for (f in fields) {
      val <- diag[[f]]
      if (is.null(val) || length(val) != 1L) val <- NA
      out[[paste("GPBoost", f, sep = "_")]] <- val
    }
  }
  out
}

split_audit_designbased_gpb <- function(calibration_full, inner, test_df, iter) {
  if (!isTRUE(gpb_save_prediction_audit)) return(data.frame())
  mk <- function(df, role) {
    data.frame(
      iteration = iter,
      scenario = scenario,
      protocol = "DesignBased",
      split_role = role,
      wadoux_row_id = row_id_gpb(df),
      x = df$x,
      y = df$y,
      obs = df[[response_name_wadoux]],
      stringsAsFactors = FALSE
    )
  }
  rbind(
    mk(calibration_full, "calibration_full"),
    mk(inner$train, "inner_train"),
    mk(inner$val, "inner_val"),
    mk(test_df, "design_test")
  )
}

# ---- Cubist wrapper ---------------------------------------------------------

fit_predict_cubist_gpb <- function(X_train, y_train, X_test,
                                   committees = 10L, neighbors = 0L) {
  if (!requireNamespace("Cubist", quietly = TRUE)) {
    return(rep(NA_real_, nrow(X_test)))
  }
  cb <- Cubist::cubist(x = as.data.frame(X_train), y = y_train,
                       committees = committees)
  as.numeric(predict(cb, newdata = as.data.frame(X_test),
                     neighbors = neighbors))
}

# ---- Metrics bookkeeping ----------------------------------------------------

emit_metric_rows_gpb <- function(obs, pred_map, protocol, iter, scenario) {
  rows <- list()
  for (m in names(pred_map)) {
    p <- pred_map[[m]]
    if (is.null(p) || length(p) == 0L) next
    e <- wadoux_eval(obs = obs, pred = p)
    rows[[length(rows) + 1L]] <- data.frame(
      iteration = iter, scenario = scenario, protocol = protocol,
      model = m, ME = e$ME, RMSE = e$RMSE, r2 = e$r2, MEC = e$MEC,
      stringsAsFactors = FALSE)
  }
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}

# ---- Protocol evaluators ----------------------------------------------------

eval_designbased_gpb <- function(valuetable, iter) {
  val_srs  <- sample_design_test_gpb(
    valuetable, sample_size,
    seed = train_seed + iter * 100000L + 17L)
  obs      <- val_srs[[response_name_wadoux]]
  inner <- make_inner_val_split_gpb(valuetable, val_frac = val_frac,
                                    seed = train_seed + iter)
  calibration_full <- valuetable
  pred_map <- list()
  diag_rows <- list()
  candidate_rows <- list()
  gpboost_diag <- NULL

  if ("RF" %in% models) {
    rf <- fit_rf_default_wadoux(calibration_full, form_rf)
    pred_map$RF <- predict_rf_default_wadoux(rf, val_srs)
  }
  if ("Cubist" %in% models) {
    pred_map$Cubist <- fit_predict_cubist_gpb(
      X_train = calibration_full[, predList, drop = FALSE],
      y_train = calibration_full[[response_name_wadoux]],
      X_test  = val_srs[, predList, drop = FALSE]
    )
  }
  if ("GPBoost" %in% models) {
    out <- run_gpboost_on_split(inner$train, inner$val, val_srs,
                                full_train_df = calibration_full)
    pred_map$GPBoost <- out$pred_test
    gpboost_diag <- out$diagnostics
    diag_rows[[1L]] <- diagnostics_row_gpb(out, iter, "DesignBased")
    if (!is.null(out$candidate_table) && nrow(out$candidate_table) > 0L) {
      candidate_rows[[1L]] <- cbind(iteration = iter,
                                    protocol = "DesignBased",
                                    out$candidate_table)
    }
  }
  audit_rows <- prediction_audit_designbased_gpb(
    val_srs, pred_map, iter, diag = gpboost_diag)
  split_rows <- split_audit_designbased_gpb(
    calibration_full, inner, val_srs, iter)
  list(metrics = emit_metric_rows_gpb(obs, pred_map,
                                      "DesignBased", iter, scenario),
       diagnostics = if (length(diag_rows)) do.call(rbind, diag_rows) else data.frame(),
       candidates = if (length(candidate_rows)) do.call(rbind, candidate_rows) else data.frame(),
       prediction_audit = audit_rows,
       split_audit = split_rows)
}

eval_randomkfold_gpb <- function(valuetable, iter) {
  flds <- caret::createFolds(valuetable[[response_name_wadoux]],
                             k = random_k, list = TRUE, returnTrain = FALSE)
  preds_rf  <- rep(NA_real_, nrow(valuetable))
  preds_cb  <- rep(NA_real_, nrow(valuetable))
  preds_gpb <- rep(NA_real_, nrow(valuetable))
  for (j in seq_along(flds)) {
    id  <- flds[[j]]
    trn <- valuetable[-id, , drop = FALSE]
    val <- valuetable[id,  , drop = FALSE]
    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(trn, form_rf)
      preds_rf[id] <- predict_rf_default_wadoux(rf, val)
    }
    if ("Cubist" %in% models) {
      preds_cb[id] <- fit_predict_cubist_gpb(
        X_train = trn[, predList, drop = FALSE],
        y_train = trn[[response_name_wadoux]],
        X_test  = val[, predList, drop = FALSE])
    }
    if ("GPBoost" %in% models) {
      inner <- make_inner_val_split_gpb(trn, val_frac = val_frac,
                                        seed = train_seed + iter * 1000L + j)
      out <- run_gpboost_on_split(inner$train, inner$val, val,
                                  full_train_df = trn)
      preds_gpb[id] <- out$pred_test
    }
  }
  pred_map <- list()
  if ("RF"      %in% models) pred_map$RF      <- preds_rf
  if ("Cubist"  %in% models) pred_map$Cubist  <- preds_cb
  if ("GPBoost" %in% models) pred_map$GPBoost <- preds_gpb
  list(metrics = emit_metric_rows_gpb(valuetable[[response_name_wadoux]],
                                      pred_map, "RandomKFold", iter, scenario))
}

eval_spatialkfold_gpb <- function(valuetable, iter) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  fold_ids      <- unique(spatial_folds)
  preds_rf  <- rep(NA_real_, nrow(valuetable))
  preds_cb  <- rep(NA_real_, nrow(valuetable))
  preds_gpb <- rep(NA_real_, nrow(valuetable))
  for (j in seq_along(fold_ids)) {
    id  <- which(spatial_folds == fold_ids[j])
    trn <- valuetable[-id, , drop = FALSE]
    val <- valuetable[id,  , drop = FALSE]
    if (nrow(trn) < 10L) next
    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(trn, form_rf)
      preds_rf[id] <- predict_rf_default_wadoux(rf, val)
    }
    if ("Cubist" %in% models) {
      preds_cb[id] <- fit_predict_cubist_gpb(
        X_train = trn[, predList, drop = FALSE],
        y_train = trn[[response_name_wadoux]],
        X_test  = val[, predList, drop = FALSE])
    }
    if ("GPBoost" %in% models) {
      inner <- make_inner_val_split_gpb(
        trn, val_frac = val_frac,
        seed = train_seed + iter * 1000L + fold_ids[j])
      out <- run_gpboost_on_split(inner$train, inner$val, val,
                                  full_train_df = trn)
      preds_gpb[id] <- out$pred_test
    }
  }
  pred_map <- list()
  if ("RF"      %in% models) pred_map$RF      <- preds_rf
  if ("Cubist"  %in% models) pred_map$Cubist  <- preds_cb
  if ("GPBoost" %in% models) pred_map$GPBoost <- preds_gpb
  list(metrics = emit_metric_rows_gpb(valuetable[[response_name_wadoux]],
                                      pred_map, "SpatialKFold", iter, scenario))
}

eval_bloocv_gpb <- function(valuetable, iter) {
  nb_iteration <- bloo_groups * bloo_pixels
  gpb_cap_raw  <- Sys.getenv("GPB_BLOO_GPB_PIXELS", unset = "")
  gpb_cap <- if (nzchar(gpb_cap_raw)) suppressWarnings(as.integer(gpb_cap_raw))
             else as.integer(min(nb_iteration, bloo_groups * 10L))
  if (!is.finite(gpb_cap) || gpb_cap < 1L) gpb_cap <- nb_iteration
  set.seed(train_seed + iter)
  ri <- val_dist_km * 1000
  gpb_used <- 0L
  pred_rows <- list()
  for (j in seq_len(nb_iteration)) {
    point_in_range <- FALSE
    id_focal <- NA_integer_
    while (!isTRUE(point_in_range)) {
      id_focal     <- sample.int(nrow(valuetable), size = 1L)
      focal_point  <- valuetable[id_focal, , drop = FALSE]
      training_tmp <- valuetable[-id_focal, , drop = FALSE]
      training_tmp <- exclude_by_radius_wadoux(training_tmp, focal_point, ri)
      if (nrow(training_tmp) < 10L) next
      if (!point_within_predictor_range_wadoux(training_tmp, focal_point,
                                               predList)) next
      point_in_range <- TRUE
    }
    focal_point   <- valuetable[id_focal, , drop = FALSE]
    training_full <- valuetable[-id_focal, , drop = FALSE]
    training_full <- exclude_by_radius_wadoux(training_full, focal_point, ri)
    pred_rf <- NA_real_; pred_cb <- NA_real_; pred_gpb <- NA_real_
    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(training_full, form_rf)
      pred_rf <- predict_rf_default_wadoux(rf, focal_point)
    }
    if ("Cubist" %in% models) {
      pred_cb <- fit_predict_cubist_gpb(
        X_train = training_full[, predList, drop = FALSE],
        y_train = training_full[[response_name_wadoux]],
        X_test  = focal_point [, predList, drop = FALSE])
    }
    if ("GPBoost" %in% models && gpb_used < gpb_cap) {
      inner <- make_inner_val_split_gpb(training_full, val_frac = val_frac,
                                        seed = train_seed + iter * 1000L + j)
      out <- run_gpboost_on_split(inner$train, inner$val, focal_point,
                                  full_train_df = training_full)
      pred_gpb <- out$pred_test[1]
      gpb_used <- gpb_used + 1L
    }
    pred_rows[[j]] <- data.frame(
      focal_id = id_focal,
      obs      = focal_point[[response_name_wadoux]],
      pred_rf  = pred_rf,
      pred_cb  = pred_cb,
      pred_gpb = pred_gpb,
      stringsAsFactors = FALSE)
  }
  pred_df <- do.call(rbind, pred_rows)
  pred_df$group <- caret::createFolds(seq_len(nrow(pred_df)), k = bloo_groups,
                                      list = FALSE, returnTrain = FALSE)
  group_ids <- sort(unique(pred_df$group))
  emit_one <- function(label, col) {
    if (!(label %in% models)) return(NULL)
    per_group <- lapply(group_ids, function(g) {
      sub <- pred_df[pred_df$group == g, , drop = FALSE]
      ok  <- is.finite(sub[[col]]) & is.finite(sub$obs)
      if (sum(ok) < 2L) return(NULL)
      wadoux_eval(obs = sub$obs[ok], pred = sub[[col]][ok])
    })
    ok <- per_group[!vapply(per_group, is.null, logical(1))]
    if (length(ok) == 0L) return(NULL)
    stk <- do.call(rbind, ok)
    data.frame(iteration = iter, scenario = scenario, protocol = "BLOOCV",
               model = label,
               ME = mean(stk$ME, na.rm = TRUE),
               RMSE = mean(stk$RMSE, na.rm = TRUE),
               r2 = mean(stk$r2, na.rm = TRUE),
               MEC = mean(stk$MEC, na.rm = TRUE),
               stringsAsFactors = FALSE)
  }
  rows <- list(emit_one("RF",      "pred_rf"),
               emit_one("Cubist",  "pred_cb"),
               emit_one("GPBoost", "pred_gpb"))
  rows <- rows[!vapply(rows, is.null, logical(1))]
  list(metrics = if (length(rows) == 0L) data.frame() else do.call(rbind, rows))
}

# ---- Save helpers -----------------------------------------------------------

save_gpboost_checkpoint <- function(iter, metric_rows,
                                    diagnostic_rows = list(),
                                    candidate_rows = list(),
                                    prediction_audit_rows = list(),
                                    split_audit_rows = list()) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  write.csv(df, file.path(results_dir, "gpboost_all_results.csv"),
            row.names = FALSE)
  diag_df <- do.call(rbind, diagnostic_rows[!vapply(diagnostic_rows, is.null, logical(1))])
  if (!is.null(diag_df) && nrow(diag_df) > 0L) {
    write.csv(diag_df, file.path(results_dir, "gpboost_diagnostics.csv"),
              row.names = FALSE)
  }
  cand_df <- do.call(rbind, candidate_rows[!vapply(candidate_rows, is.null, logical(1))])
  if (!is.null(cand_df) && nrow(cand_df) > 0L) {
    write.csv(cand_df, file.path(results_dir, "gpboost_candidate_scores.csv"),
              row.names = FALSE)
  }
  pred_audit_df <- do.call(rbind, prediction_audit_rows[
    !vapply(prediction_audit_rows, is.null, logical(1))])
  if (!is.null(pred_audit_df) && nrow(pred_audit_df) > 0L) {
    write.csv(pred_audit_df,
              file.path(results_dir, "gpboost_prediction_audit.csv"),
              row.names = FALSE)
  }
  split_audit_df <- do.call(rbind, split_audit_rows[
    !vapply(split_audit_rows, is.null, logical(1))])
  if (!is.null(split_audit_df) && nrow(split_audit_df) > 0L) {
    write.csv(split_audit_df,
              file.path(results_dir, "gpboost_split_audit.csv"),
              row.names = FALSE)
  }
  if ("RF" %in% models && "GPBoost" %in% models) {
    key_cols <- c("iteration", "scenario", "protocol")
    rf_df <- df[df$model == "RF", , drop = FALSE]
    gp_df <- df[df$model == "GPBoost", , drop = FALSE]
    if (nrow(rf_df) > 0L && nrow(gp_df) > 0L) {
      merged <- merge(
        rf_df[, c(key_cols, "ME", "RMSE", "r2", "MEC")],
        gp_df[, c(key_cols, "ME", "RMSE", "r2", "MEC")],
        by = key_cols, suffixes = c("_RF", "_GPB"))
      merged$delta_ME   <- merged$ME_GPB   - merged$ME_RF
      merged$delta_RMSE <- merged$RMSE_GPB - merged$RMSE_RF
      merged$delta_r2   <- merged$r2_GPB   - merged$r2_RF
      merged$delta_MEC  <- merged$MEC_GPB  - merged$MEC_RF
      write.csv(merged, file.path(results_dir,
                                  "gpboost_paired_deltas_vs_RF.csv"),
                row.names = FALSE)
    }
  }
  write.csv(
    data.frame(completed_iteration = iter, total_iterations = n_iter,
               scenario = scenario, protocols = paste(protocols, collapse = ","),
               models = paste(models, collapse = ","),
               completed_at = format(Sys.time(), tz = "", usetz = TRUE),
               stringsAsFactors = FALSE),
    file.path(results_dir, "gpboost_checkpoint_status.csv"), row.names = FALSE)
}

summarise_and_save_gpb <- function(metric_rows) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  by_protocol <- aggregate(
    cbind(ME, RMSE, r2, MEC) ~ scenario + protocol + model,
    data = df,
    FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                        sd   = sd(x,   na.rm = TRUE)))
  stats_df <- do.call(cbind, lapply(c("ME", "RMSE", "r2", "MEC"),
                                    function(m) {
    mat <- by_protocol[[m]]
    setNames(as.data.frame(mat), c(paste0(m, "_mean"), paste0(m, "_sd")))
  }))
  out <- cbind(by_protocol[, c("scenario", "protocol", "model")], stats_df)
  write.csv(out, file.path(results_dir, "gpboost_summary_by_protocol.csv"),
            row.names = FALSE)
}

# ---- Main loop --------------------------------------------------------------

write.csv(
  data.frame(scenario = scenario, protocols = paste(protocols, collapse = ","),
             models = paste(models, collapse = ","),
             sample_size = sample_size, n_iter = n_iter, val_frac = val_frac,
             val_dist_km = val_dist_km, random_k = random_k,
             bloo_groups = bloo_groups, bloo_test_pixels = bloo_pixels,
	             train_seed = train_seed,
	             iter_offset = iter_offset,
	             fair_tune = isTRUE(gpboost_params$fair_tune),
	             save_prediction_audit = isTRUE(gpb_save_prediction_audit),
	             autotune_v2 = isTRUE(gpboost_params$autotune_v2),
	             crash_safe = isTRUE(gpboost_params$crash_safe),
	             spatial_policy = gpboost_params$spatial_policy,
	             tune_full_grid = gpb_tune_full_grid,
		             refit_full = isTRUE(gpboost_params$refit_full),
		             num_threads = if (is.null(gpboost_params$num_threads)) {
		               NA_integer_
		             } else {
		               gpboost_params$num_threads
		             },
			             tune_inner_folds = gpboost_params$tune_inner_folds,
			             selection_objective = gpboost_params$selection_objective,
			             rf_noharm_policy = isTRUE(gpboost_params$rf_noharm_policy),
			             rf_noharm_margin_abs = gpboost_params$rf_noharm_margin_abs,
			             rf_noharm_margin_rel = gpboost_params$rf_noharm_margin_rel,
			             rf_noharm_tree_only_ensemble =
			               isTRUE(gpboost_params$rf_noharm_tree_only_ensemble),
		             target_transform = gpboost_params$target_transform,
	             init_cov_pars_order = "error_variance,gp_variance,gp_range",
	             use_gp_model_for_validation = isTRUE(gpboost_params$use_gp_model_for_validation),
	             max_tune_candidates =
	               if (is.finite(gpboost_params$max_tune_candidates)) {
	                 gpboost_params$max_tune_candidates
	               } else NA_integer_,
	             GPB_TUNE_FULL_GRID = raw_env_gpb("GPB_TUNE_FULL_GRID", "FALSE"),
	             GPB_AUTOTUNE_V2 = raw_env_gpb("GPB_AUTOTUNE_V2", "TRUE"),
	             GPB_STAGE1_MAX_CANDIDATES = raw_env_gpb(
	               "GPB_STAGE1_MAX_CANDIDATES", "96"),
	             GPB_STAGE2_TOP_K = raw_env_gpb("GPB_STAGE2_TOP_K", "12"),
	             GPB_STAGE2_MAX_CANDIDATES = raw_env_gpb(
	               "GPB_STAGE2_MAX_CANDIDATES", "48"),
	             GPB_STAGE2_SELECT_ONLY = raw_env_gpb(
	               "GPB_STAGE2_SELECT_ONLY", "FALSE"),
	             GPB_CRASH_SAFE = raw_env_gpb("GPB_CRASH_SAFE", "TRUE"),
	             GPB_SPATIAL_POLICY = raw_env_gpb("GPB_SPATIAL_POLICY", "tree_only"),
	             GPB_RMSE_SD_WEIGHT = raw_env_gpb("GPB_RMSE_SD_WEIGHT", "0.25"),
	             GPB_QUANTILE_BINS = raw_env_gpb("GPB_QUANTILE_BINS", "5"),
	             GPB_QUANTILE_EXCESS_WEIGHT = raw_env_gpb(
	               "GPB_QUANTILE_EXCESS_WEIGHT", "0.15"),
	             GPB_QUANTILE_SD_WEIGHT = raw_env_gpb(
	               "GPB_QUANTILE_SD_WEIGHT", "0.05"),
	             GPB_ENSEMBLE_TOP_K = raw_env_gpb(
	               "GPB_ENSEMBLE_TOP_K", "1"),
	             GPB_ENSEMBLE_WEIGHTING = raw_env_gpb(
	               "GPB_ENSEMBLE_WEIGHTING", "softmax"),
	             GPB_ENSEMBLE_TEMPERATURE = raw_env_gpb(
	               "GPB_ENSEMBLE_TEMPERATURE", "0.75"),
		             GPB_SELECTION_OBJECTIVE = raw_env_gpb(
		               "GPB_SELECTION_OBJECTIVE", "paired_rf_delta"),
	             GPB_POLICY_BLEND = raw_env_gpb("GPB_POLICY_BLEND", "FALSE"),
	             GPB_BLEND_ALPHA_GRID = raw_env_gpb(
	               "GPB_BLEND_ALPHA_GRID", "0,0.1,0.25,0.5,0.75,1"),
	             GPB_RESIDUAL_SPATIAL_CORRECTION = raw_env_gpb(
	               "GPB_RESIDUAL_SPATIAL_CORRECTION", "FALSE"),
		             GPB_RESIDUAL_GAMMA_GRID = raw_env_gpb(
		               "GPB_RESIDUAL_GAMMA_GRID", "0,0.25,0.5,0.75,1"),
		             GPB_SEED_ENSEMBLE_N = raw_env_gpb("GPB_SEED_ENSEMBLE_N", "1"),
		             GPB_NOHARM_TREE_ANCHOR = raw_env_gpb(
		               "GPB_NOHARM_TREE_ANCHOR", "TRUE"),
		             GPB_NOHARM_MARGIN_ABS = raw_env_gpb(
		               "GPB_NOHARM_MARGIN_ABS", "0.05"),
		             GPB_NOHARM_MARGIN_REL = raw_env_gpb(
		               "GPB_NOHARM_MARGIN_REL", "0"),
			             GPB_NOHARM_TREE_ONLY_ENSEMBLE = raw_env_gpb(
			               "GPB_NOHARM_TREE_ONLY_ENSEMBLE", "TRUE"),
			             GPB_RF_NOHARM_POLICY = raw_env_gpb(
			               "GPB_RF_NOHARM_POLICY", "TRUE"),
			             GPB_RF_NOHARM_MARGIN_ABS = raw_env_gpb(
			               "GPB_RF_NOHARM_MARGIN_ABS", "0.05"),
			             GPB_RF_NOHARM_MARGIN_REL = raw_env_gpb(
			               "GPB_RF_NOHARM_MARGIN_REL", "0"),
			             GPB_RF_NOHARM_TREE_ONLY_ENSEMBLE = raw_env_gpb(
			               "GPB_RF_NOHARM_TREE_ONLY_ENSEMBLE", "TRUE"),
			             GPB_SPATIAL_FILTER = raw_env_gpb("GPB_SPATIAL_FILTER", "TRUE"),
	             GPB_SPATIAL_FILTER_MAX_ALPHA_NO_SIGNAL = raw_env_gpb(
	               "GPB_SPATIAL_FILTER_MAX_ALPHA_NO_SIGNAL", "0.25"),
	             GPB_SPATIAL_FILTER_MAX_GAMMA_NO_SIGNAL = raw_env_gpb(
	               "GPB_SPATIAL_FILTER_MAX_GAMMA_NO_SIGNAL", "0.25"),
	             GPB_SAVE_PREDICTION_AUDIT = raw_env_gpb(
	               "GPB_SAVE_PREDICTION_AUDIT", "TRUE"),
	             GPB_TUNE_MAX_SECONDS = raw_env_gpb("GPB_TUNE_MAX_SECONDS", "Inf"),
	             GPB_TUNE_PROGRESS_FILE = raw_env_gpb(
	               "GPB_TUNE_PROGRESS_FILE",
	               file.path(results_dir, "gpboost_tuning_progress_live.csv")),
		             GPB_MAX_TUNE_CANDIDATES = raw_env_gpb("GPB_MAX_TUNE_CANDIDATES", ""),
		             GPB_NUM_THREADS = raw_env_gpb("GPB_NUM_THREADS", ""),
		             GPB_TUNE_INNER_FOLDS = raw_env_gpb("GPB_TUNE_INNER_FOLDS", "3"),
	             GPB_TUNE_LR_GRID = raw_env_gpb("GPB_TUNE_LR_GRID", ""),
	             GPB_TUNE_DEPTH_GRID = raw_env_gpb("GPB_TUNE_DEPTH_GRID", ""),
	             GPB_TUNE_MATERN_GRID = raw_env_gpb("GPB_TUNE_MATERN_GRID", ""),
	             GPB_TUNE_FEATURE_GRID = raw_env_gpb("GPB_TUNE_FEATURE_GRID", ""),
	             GPB_TUNE_MIN_LEAF_GRID = raw_env_gpb("GPB_TUNE_MIN_LEAF_GRID", ""),
	             GPB_TUNE_BAGGING_GRID = raw_env_gpb("GPB_TUNE_BAGGING_GRID", ""),
	             GPB_TUNE_L2_GRID = raw_env_gpb("GPB_TUNE_L2_GRID", ""),
	             GPB_RANGE_INIT_GRID = raw_env_gpb("GPB_RANGE_INIT_GRID", ""),
	             GPB_NUGGET_INIT_GRID = raw_env_gpb("GPB_NUGGET_INIT_GRID", ""),
	             stringsAsFactors = FALSE),
  file.path(results_dir, "gpboost_config.csv"), row.names = FALSE)

all_metric_rows <- list()
all_diagnostic_rows <- list()
all_candidate_rows <- list()
all_prediction_audit_rows <- list()
all_split_audit_rows <- list()
for (iter in seq_len(n_iter)) {
  global_iter <- iter + iter_offset
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("GPBoost benchmark | scenario=%s | iter %d / %d\n",
              scenario, global_iter, iter_offset + n_iter))
  cat(sprintf("==================================================\n"))

  valuetable <- na.omit(as.data.frame(sample_calibration_gpb(
    seed = train_seed + global_iter * 100000L + 1L)))

  iter_rows <- list()
  if ("DesignBased"  %in% protocols) {
    cat("\n--- DesignBased ---\n")
    out_design <- eval_designbased_gpb(valuetable, global_iter)
    iter_rows[["DesignBased"]] <- out_design$metrics
    if (!is.null(out_design$diagnostics) && nrow(out_design$diagnostics) > 0L) {
      all_diagnostic_rows[[length(all_diagnostic_rows) + 1L]] <- out_design$diagnostics
    }
    if (!is.null(out_design$candidates) && nrow(out_design$candidates) > 0L) {
      all_candidate_rows[[length(all_candidate_rows) + 1L]] <- out_design$candidates
    }
    if (!is.null(out_design$prediction_audit) &&
        nrow(out_design$prediction_audit) > 0L) {
      all_prediction_audit_rows[[length(all_prediction_audit_rows) + 1L]] <-
        out_design$prediction_audit
    }
    if (!is.null(out_design$split_audit) &&
        nrow(out_design$split_audit) > 0L) {
      all_split_audit_rows[[length(all_split_audit_rows) + 1L]] <-
        out_design$split_audit
    }
  }
  if ("RandomKFold"  %in% protocols) {
    cat("\n--- RandomKFold ---\n")
    iter_rows[["RandomKFold"]]  <- eval_randomkfold_gpb(valuetable, global_iter)$metrics
  }
  if ("SpatialKFold" %in% protocols) {
    cat("\n--- SpatialKFold ---\n")
    iter_rows[["SpatialKFold"]] <- eval_spatialkfold_gpb(valuetable, global_iter)$metrics
  }
  if ("BLOOCV"       %in% protocols) {
    cat("\n--- BLOOCV ---\n")
    iter_rows[["BLOOCV"]]       <- eval_bloocv_gpb(valuetable, global_iter)$metrics
  }

  iter_df <- do.call(rbind, iter_rows[!vapply(iter_rows, is.null, logical(1))])
  if (!is.null(iter_df) && nrow(iter_df) > 0L) {
    print(iter_df)
    all_metric_rows[[iter]] <- iter_df
  }

  if (iter %% checkpoint_every == 0L || iter == n_iter) {
    save_gpboost_checkpoint(iter, all_metric_rows,
                            all_diagnostic_rows, all_candidate_rows,
                            all_prediction_audit_rows,
                            all_split_audit_rows)
    summarise_and_save_gpb(all_metric_rows)
  }
}

cat("\n[GPBoost runner] Done. Results written to: ", results_dir, "\n",
    sep = "")
