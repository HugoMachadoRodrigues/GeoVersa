# =============================================================================
# run_deepkernel_wadoux_comparison.R
#
# Head-to-head benchmark: RF vs DeepKernelProcessGP (DKP) under the Wadoux
# 2021 validation protocols. DKP is a standalone Gaussian-Process predictor
# with a composite kernel (learned feature RBF x non-stationary Matern),
# defined in code/DeepKernelProcess.R. The runner MIRRORS the NMK runner
# except for the model it calls — so the comparison against RF stays
# apples-to-apples: same scenarios, same protocols, same inner val split,
# same metric convention.
#
# Environment overrides:
#   DKP_SCENARIO            random | regular_grid | clustered_random
#   DKP_PROTOCOLS           comma list of DesignBased|RandomKFold|
#                                         SpatialKFold|BLOOCV|Population
#   DKP_MODELS              comma list of RF|DeepKernelProcess
#   DKP_N_ITER              number of calibration samples (default 10)
#   DKP_SAMPLE_SIZE         calibration size per iter (default 500)
#   DKP_VAL_FRAC            inner val fraction (default 0.20)
#   DKP_VAL_DIST_KM         spatial CV / BLOOCV exclusion radius (km)
#   DKP_RANDOM_K            random k-fold splits (default 10)
#   DKP_BLOO_GROUPS         BLOOCV groups (default 10)
#   DKP_BLOO_TEST_PIXELS    BLOOCV focal pixels per group (default 100)
#   DKP_BLOO_NMK_PIXELS     cap on DKP retrains inside BLOOCV
#   DKP_DEVICE              cpu | mps | cuda
#   DKP_TRAIN_SEED          RNG seed (default 123)
#   DKP_RESULTS_DIR         output dir (default results/dkp_run)
#   DKP_PROFILE             default | quick
#   DKP_EPOCHS, DKP_PATIENCE, DKP_LR, DKP_WEIGHT_DECAY,
#   DKP_LAMBDA_MIX_MEAN, DKP_MATERN_HIDDEN, DKP_TARGET_TRANSFORM,
#   DKP_CHECKPOINT_EVERY
#   DKP_DUAN_APPLY        TRUE (default) | FALSE to disable smearing
#                         (only matters when target_transform = "log")
#   DKP_DUAN_MIN_VAL_N    minimum n_val required to apply Duan (default 500)
#   DKP_DUAN_TRIM_FRAC    trimmed-mean fraction per tail (default 0.05)
#   DKP_TUNE              TRUE by default for DesignBased DKP
#   DKP_TUNE_TARGET_TRANSFORMS comma list from {auto,identity,log}
#   DKP_TUNE_MAX_CANDIDATES default 18
#   DKP_TUNE_INNER_FOLDS default 3
#   DKP_SELECTION_OBJECTIVE rmse | paired_rf_delta (default paired_rf_delta)
#   DKP_NOHARM_POLICY, DKP_NOHARM_MARGIN_ABS, DKP_NOHARM_MARGIN_REL
#   DKP_NOHARM_RESTRICT_ENSEMBLE keep fallback ensembles conservative
#   DKP_REFIT_FULL_CALIBRATION TRUE by default
#   DKP_ENSEMBLE_TOP_K, DKP_ENSEMBLE_WEIGHTING, DKP_ENSEMBLE_TEMPERATURE
#   DKP_SAVE_PREDICTION_AUDIT TRUE writes split/prediction audit CSVs
# =============================================================================

rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")

parse_bool_env_dkp <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}
parse_num_env_dkp <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) default else val
}
parse_list_env_dkp <- function(name, default) {
  raw <- Sys.getenv(name, unset = paste(default, collapse = ","))
  vals <- trimws(unlist(strsplit(raw, ",")))
  vals[vals != ""]
}
raw_env_dkp <- function(name, default = "") {
  Sys.getenv(name, unset = default)
}

scenario    <- Sys.getenv("DKP_SCENARIO", unset = "random")
protocols   <- parse_list_env_dkp("DKP_PROTOCOLS",
                                  c("DesignBased", "RandomKFold",
                                    "SpatialKFold", "BLOOCV"))
models      <- parse_list_env_dkp("DKP_MODELS",
                                  c("RF", "DeepKernelProcess"))
allowed_models <- c("RF", "DeepKernelProcess")
invalid_models <- setdiff(models, allowed_models)
if (length(invalid_models) > 0) {
  stop("Unsupported DKP_MODELS entries: ", paste(invalid_models, collapse = ", "))
}
include_population <- "Population" %in% protocols

n_iter        <- as.integer(parse_num_env_dkp("DKP_N_ITER",       10L))
sample_size   <- as.integer(parse_num_env_dkp("DKP_SAMPLE_SIZE",  500L))
val_frac      <-             parse_num_env_dkp("DKP_VAL_FRAC",    0.20)
val_dist_km   <-             parse_num_env_dkp("DKP_VAL_DIST_KM", 350)
random_k      <- as.integer(parse_num_env_dkp("DKP_RANDOM_K",     10L))
bloo_groups   <- as.integer(parse_num_env_dkp("DKP_BLOO_GROUPS",  10L))
bloo_pixels   <- as.integer(parse_num_env_dkp("DKP_BLOO_TEST_PIXELS", 100L))
device        <- Sys.getenv("DKP_DEVICE", unset = "cpu")
train_seed    <- as.integer(parse_num_env_dkp("DKP_TRAIN_SEED",   123L))
results_dir   <- Sys.getenv("DKP_RESULTS_DIR", unset = "results/dkp_run")
profile       <- Sys.getenv("DKP_PROFILE", unset = "default")
checkpoint_every <- as.integer(parse_num_env_dkp("DKP_CHECKPOINT_EVERY", 1L))
dkp_tune <- parse_bool_env_dkp("DKP_TUNE", TRUE)
dkp_tune_max_candidates <- as.integer(parse_num_env_dkp("DKP_TUNE_MAX_CANDIDATES", 18L))
dkp_refit_full_calibration <- parse_bool_env_dkp("DKP_REFIT_FULL_CALIBRATION", TRUE)
dkp_tune_inner_folds <- as.integer(parse_num_env_dkp("DKP_TUNE_INNER_FOLDS", 3L))
if (!is.finite(dkp_tune_inner_folds) || dkp_tune_inner_folds < 1L) {
  dkp_tune_inner_folds <- 1L
}
dkp_ensemble_top_k <- as.integer(parse_num_env_dkp("DKP_ENSEMBLE_TOP_K", 3L))
if (!is.finite(dkp_ensemble_top_k) || dkp_ensemble_top_k < 1L) {
  dkp_ensemble_top_k <- 1L
}
dkp_ensemble_weighting <- tolower(trimws(Sys.getenv(
  "DKP_ENSEMBLE_WEIGHTING", unset = "softmax")))
if (!(dkp_ensemble_weighting %in% c("softmax", "inverse", "uniform"))) {
  stop("Unsupported DKP_ENSEMBLE_WEIGHTING: ", dkp_ensemble_weighting)
}
dkp_ensemble_temperature <- parse_num_env_dkp("DKP_ENSEMBLE_TEMPERATURE", 0.75)
if (!is.finite(dkp_ensemble_temperature) || dkp_ensemble_temperature <= 0) {
  dkp_ensemble_temperature <- 0.75
}
dkp_rmse_sd_weight <- parse_num_env_dkp("DKP_RMSE_SD_WEIGHT", 0.25)
if (!is.finite(dkp_rmse_sd_weight) || dkp_rmse_sd_weight < 0) {
  dkp_rmse_sd_weight <- 0
}
dkp_selection_objective <- tolower(trimws(Sys.getenv(
  "DKP_SELECTION_OBJECTIVE", unset = "paired_rf_delta")))
if (!(dkp_selection_objective %in% c("rmse", "paired_rf_delta"))) {
  dkp_selection_objective <- "rmse"
}
dkp_noharm_policy <- parse_bool_env_dkp(
  "DKP_NOHARM_POLICY",
  identical(dkp_selection_objective, "paired_rf_delta"))
dkp_noharm_margin_abs <- parse_num_env_dkp("DKP_NOHARM_MARGIN_ABS", 0.05)
if (!is.finite(dkp_noharm_margin_abs) || dkp_noharm_margin_abs < 0) {
  dkp_noharm_margin_abs <- 0
}
dkp_noharm_margin_rel <- parse_num_env_dkp("DKP_NOHARM_MARGIN_REL", 0.0)
if (!is.finite(dkp_noharm_margin_rel) || dkp_noharm_margin_rel < 0) {
  dkp_noharm_margin_rel <- 0
}
dkp_noharm_restrict_ensemble <- parse_bool_env_dkp(
  "DKP_NOHARM_RESTRICT_ENSEMBLE", TRUE)
dkp_rf_noharm_policy <- parse_bool_env_dkp(
  "DKP_RF_NOHARM_POLICY",
  identical(dkp_selection_objective, "paired_rf_delta"))
dkp_rf_noharm_margin_abs <- parse_num_env_dkp("DKP_RF_NOHARM_MARGIN_ABS", 0.05)
if (!is.finite(dkp_rf_noharm_margin_abs) || dkp_rf_noharm_margin_abs < 0) {
  dkp_rf_noharm_margin_abs <- 0
}
dkp_rf_noharm_margin_rel <- parse_num_env_dkp("DKP_RF_NOHARM_MARGIN_REL", 0.0)
if (!is.finite(dkp_rf_noharm_margin_rel) || dkp_rf_noharm_margin_rel < 0) {
  dkp_rf_noharm_margin_rel <- 0
}
dkp_rf_noharm_restrict_ensemble <- parse_bool_env_dkp(
  "DKP_RF_NOHARM_RESTRICT_ENSEMBLE", TRUE)
dkp_save_prediction_audit <- parse_bool_env_dkp("DKP_SAVE_PREDICTION_AUDIT", TRUE)
# Pin the Wadoux RF seed for the WHOLE run so every fit_rf_default_wadoux call
# (in-gate RF, scored RF_500, the literal-RF floor below) is bit-identical.
options(wadoux_rf_seed = train_seed)
# OOF lambda grid: continuous shrinkage of the DKP prediction toward the
# literal Wadoux RF prediction. 0 == pure RF (guaranteed == RF), 1 == pure DKP.
# Grid INCLUDES 0 so the floor is always reachable on leakage-free OOF data.
dkp_rf_blend_lambda_grid <- {
  raw <- Sys.getenv("DKP_RF_BLEND_LAMBDA_GRID", unset = "0,0.25,0.5,0.75,1")
  v <- suppressWarnings(as.numeric(trimws(unlist(strsplit(raw, ",")))))
  v <- v[is.finite(v) & v >= 0 & v <= 1]
  if (length(v) == 0L) v <- c(0, 0.25, 0.5, 0.75, 1)
  sort(unique(c(0, v)))
}
dkp_tune_target_transforms <- tolower(parse_list_env_dkp(
  "DKP_TUNE_TARGET_TRANSFORMS", c("auto", "identity", "log")))
dkp_tune_target_transforms <- unique(dkp_tune_target_transforms[
  dkp_tune_target_transforms %in% c("auto", "identity", "log")])
if (length(dkp_tune_target_transforms) == 0L) {
  dkp_tune_target_transforms <- c("auto", "identity")
}

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
options(wadoux_corr_method = "pearson")

# -----------------------------------------------------------------------------
# Load DKP env: GeoVersa helpers + NMK helpers + DKP
# -----------------------------------------------------------------------------

load_dkp_env <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("torch R package not available — install torch before running DKP.")
  }
  loader_parent <- new.env(parent = globalenv())
  env <- new.env(parent = loader_parent)
  loader_parent$source <- function(file, ...) {
    is_absolute <- grepl("^(?:[A-Za-z]:[\\\\/]|/)", file)
    resolved <- if (is_absolute) file else {
      file.path(dirname(code_root_wadoux), sub("^\\./", "", file))
    }
    sys.source(resolved, envir = env)
    invisible(NULL)
  }
  sys.source(file.path(code_root_wadoux, "ConvKrigingNet2D.R"),
             envir = env)
  sys.source(file.path(code_root_wadoux, "ConvKrigingNet2D_Auto.R"),
             envir = env)
  sys.source(file.path(code_root_wadoux, "NeuroMaternKrige.R"),
             envir = env)
  sys.source(file.path(code_root_wadoux, "DeepKernelProcess.R"),
             envir = env)
  env
}

dkp_env <- load_dkp_env()

get_dkp_params <- function(profile = c("default", "quick"),
                           train_seed = 123L, device = "cpu") {
  profile <- match.arg(profile)
  base <- if (profile == "quick") dkp_env$dkp_quick_params else dkp_env$dkp_default_params
  modifyList(base, list(train_seed = train_seed, device = device))
}

dkp_params <- get_dkp_params(profile = profile,
                             train_seed = train_seed,
                             device = device)

.apply_param_override_num_dkp <- function(params, env_name, field, as_int = FALSE) {
  raw <- Sys.getenv(env_name, unset = "")
  if (!nzchar(raw)) return(params)
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) return(params)
  params[[field]] <- if (as_int) as.integer(val) else val
  cat(sprintf("[DKP runner] override %s -> %s\n", field, format(params[[field]])))
  params
}
dkp_params <- .apply_param_override_num_dkp(dkp_params, "DKP_EPOCHS",     "epochs",     as_int = TRUE)
dkp_params <- .apply_param_override_num_dkp(dkp_params, "DKP_PATIENCE",   "patience",   as_int = TRUE)
dkp_params <- .apply_param_override_num_dkp(dkp_params, "DKP_LR",         "lr")
dkp_params <- .apply_param_override_num_dkp(dkp_params, "DKP_WEIGHT_DECAY", "wd")
dkp_params <- .apply_param_override_num_dkp(dkp_params, "DKP_LAMBDA_MIX_MEAN", "init_mix_mean")
dkp_params <- .apply_param_override_num_dkp(dkp_params, "DKP_MATERN_HIDDEN", "matern_hidden", as_int = TRUE)

.apply_target_transform_override_dkp <- function(params) {
  raw <- Sys.getenv("DKP_TARGET_TRANSFORM", unset = "")
  if (!nzchar(raw)) return(params)
  val <- tolower(trimws(raw))
  if (!val %in% c("auto", "identity", "log")) {
    cat(sprintf("[DKP runner] DKP_TARGET_TRANSFORM = '%s' invalid — ignored\n", raw))
    return(params)
  }
  params$target_transform <- val
  cat(sprintf("[DKP runner] override target_transform -> %s\n", val))
  params
}
dkp_params <- .apply_target_transform_override_dkp(dkp_params)

patch_size <- {
  override <- Sys.getenv("DKP_PATCH_SIZE", unset = "")
  if (nzchar(override)) as.integer(override)
  else as.integer(min(max(8L, floor(sqrt(sample_size))), 31L))
}

cat(sprintf(
  "[DKP runner] scenario=%s  protocols=%s  models=%s  n_iter=%d  n=%d  patch=%d  device=%s  tune_folds=%d  objective=%s  noharm=%s  rf_noharm=%s  ensemble_top_k=%d  weighting=%s\n",
  scenario, paste(protocols, collapse = ","),
  paste(models, collapse = ","),
  n_iter, sample_size, patch_size, device,
  dkp_tune_inner_folds, dkp_selection_objective, dkp_noharm_policy,
  dkp_rf_noharm_policy, dkp_ensemble_top_k, dkp_ensemble_weighting))

need_polygon <- scenario %in% c("regular_grid", "clustered_random")
common_data <- load_wadoux_common_data(include_polygon = need_polygon)
predList <- predList_modelfull_wadoux
form_rf  <- as.formula(
  paste(response_name_wadoux, "~", paste(predList, collapse = "+"))
)

# -----------------------------------------------------------------------------
# Sampling + fold prep
# -----------------------------------------------------------------------------

sample_calibration_dkp <- function() {
  if (scenario == "random") {
    out <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  } else if (scenario == "regular_grid") {
    out <- sample_valuetable_regular_grid_wadoux(
      common_data = common_data, sample_size = sample_size)
  } else if (scenario == "clustered_random") {
    out <- sample_valuetable_clustered_random_wadoux(
      common_data   = common_data,
      kmeans_centers = as.integer(parse_num_env_dkp("DKP_KMEANS_CENTERS", 100L)),
      n_psu          = as.integer(parse_num_env_dkp("DKP_N_PSU",          20L)),
      m_per_psu      = as.integer(parse_num_env_dkp("DKP_M_PER_PSU",      25L))
    )
  } else {
    stop("Unsupported DKP_SCENARIO: ", scenario)
  }
  out$..patch_cache_row_id <- seq_len(nrow(out))
  out
}

make_inner_val_split_dkp <- function(train_df, val_frac = 0.2, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n_train <- nrow(train_df)
  if (n_train < 10) stop("Outer training set too small for DKP.")
  val_size <- max(1L, floor(n_train * val_frac))
  val_idx <- sample.int(n_train, size = val_size, replace = FALSE)
  train_idx <- setdiff(seq_len(n_train), val_idx)
  list(train = train_df[train_idx, , drop = FALSE],
       val   = train_df[val_idx,   , drop = FALSE])
}

row_id_dkp <- function(df) {
  if ("..wadoux_row_id" %in% names(df)) {
    df$..wadoux_row_id
  } else {
    seq_len(nrow(df))
  }
}

coord_key_dkp <- function(df) {
  if (!all(c("x", "y") %in% names(df))) {
    return(rep(NA_character_, nrow(df)))
  }
  paste0(round(as.numeric(df$x), 6), "|", round(as.numeric(df$y), 6))
}

sample_design_test_dkp <- function(calibration_full, size) {
  pool <- common_data$s_df
  keep <- rep(TRUE, nrow(pool))
  if ("..wadoux_row_id" %in% names(pool) &&
      "..wadoux_row_id" %in% names(calibration_full)) {
    keep <- keep & !(pool$..wadoux_row_id %in% calibration_full$..wadoux_row_id)
  }
  pool_xy <- coord_key_dkp(pool)
  cal_xy <- coord_key_dkp(calibration_full)
  if (any(!is.na(pool_xy)) && any(!is.na(cal_xy))) {
    keep <- keep & !(pool_xy %in% cal_xy)
  }
  eligible <- pool[keep, , drop = FALSE]
  if (nrow(eligible) < size) {
    stop(sprintf(
      "DesignBased DKP test pool has only %d eligible rows after excluding calibration rows; need %d.",
      nrow(eligible), size
    ))
  }
  sample_simple_random_rows_wadoux(eligible, size)
}

bind_patch_arrays_4d_dkp <- function(arrays) {
  arrays <- arrays[!vapply(arrays, is.null, logical(1))]
  if (length(arrays) == 0L) return(NULL)
  if (length(arrays) == 1L) return(arrays[[1]])
  ref_dim <- dim(arrays[[1]])
  if (length(ref_dim) != 4L) stop("Expected 4D patch arrays.")
  for (i in 2:length(arrays)) {
    d <- dim(arrays[[i]])
    if (length(d) != 4L || !all(d[1:3] == ref_dim[1:3])) {
      stop("All patch arrays must agree in channels, height and width.")
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

prepare_dkp_fold <- function(train_df, val_df, test_df,
                             calibration_patch_cache,
                             context, patch_size) {
  combined_df <- rbind(train_df, val_df, test_df)
  n_train <- nrow(train_df); n_val <- nrow(val_df); n_test <- nrow(test_df)

  cache_col <- "..patch_cache_row_id"
  train_val_ok <- !is.null(calibration_patch_cache) &&
    all(cache_col %in% names(train_df), cache_col %in% names(val_df)) &&
    !anyNA(train_df[[cache_col]]) && !anyNA(val_df[[cache_col]])
  test_ok <- !is.null(calibration_patch_cache) &&
    cache_col %in% names(test_df) && !anyNA(test_df[[cache_col]])

  combined_cache <- NULL
  if (train_val_ok) {
    train_cache_idx <- as.integer(train_df[[cache_col]])
    val_cache_idx   <- as.integer(val_df[[cache_col]])
    parts <- list(
      calibration_patch_cache$patches_raw[, , , train_cache_idx, drop = FALSE],
      calibration_patch_cache$patches_raw[, , , val_cache_idx,   drop = FALSE]
    )
    if (test_ok) {
      test_cache_idx <- as.integer(test_df[[cache_col]])
      parts[[3]] <- calibration_patch_cache$patches_raw[, , , test_cache_idx, drop = FALSE]
    } else {
      external_cache <- dkp_env$build_pointpatch_patch_cache(
        context        = context,
        calibration_df = test_df,
        patch_size     = patch_size,
        label          = "test patches"
      )
      parts[[3]] <- external_cache$patches_raw
    }
    combined_cache <- list(
      patches_raw = bind_patch_arrays_4d_dkp(parts),
      patch_size  = patch_size,
      n           = nrow(combined_df)
    )
  }

  # The K-NN pool in prepare_pointpatch_fold is unused by DKP but the helper
  # expects a K. Use a small placeholder — DKP ignores it.
  dkp_env$prepare_pointpatch_fold(
    context        = context,
    calibration_df = combined_df,
    train_idx      = seq_len(n_train),
    val_idx        = n_train + seq_len(n_val),
    test_idx       = n_train + n_val + seq_len(n_test),
    patch_size     = patch_size,
    K              = 10L,
    patch_cache    = combined_cache
  )
}

run_dkp_on_split <- function(train_core, val_df, test_df,
	                             calibration_patch_cache,
                             params = dkp_params) {
  fd <- prepare_dkp_fold(
    train_df = train_core, val_df = val_df, test_df = test_df,
    calibration_patch_cache = calibration_patch_cache,
    context = dkp_env$wadoux_context,
    patch_size = patch_size
  )
  do.call(
    dkp_env$train_deep_kernel_process_one_fold,
    c(list(fd = fd), params)
  )
}

dkp_refit_monitor_df <- function(train_df) {
  if (nrow(train_df) == 0L) stop("DKP refit monitor requires training rows.")
  train_df[1L, , drop = FALSE]
}

dkp_refit_epochs_from_diag <- function(diag, default_epochs) {
  vals <- numeric()
  if (!is.null(diag$best_epoch)) {
    vals <- c(vals, suppressWarnings(as.numeric(diag$best_epoch)))
  }
  vals <- vals[is.finite(vals) & vals > 0]
  if (length(vals) == 0L && !is.null(diag$main_epochs_done)) {
    vals <- suppressWarnings(as.numeric(diag$main_epochs_done))
    vals <- vals[is.finite(vals) & vals > 0]
  }
  if (length(vals) == 0L) vals <- suppressWarnings(as.numeric(default_epochs))
  vals <- vals[is.finite(vals) & vals > 0]
  if (length(vals) == 0L) vals <- 1
  as.integer(max(1L, round(stats::median(vals, na.rm = TRUE))))
}

dkp_refit_params_from_selection <- function(selection_diag, base_params = dkp_params) {
  params <- base_params
  refit_epochs <- dkp_refit_epochs_from_diag(
    selection_diag,
    default_epochs = if (is.null(base_params$epochs)) 200L else base_params$epochs
  )
  params$epochs <- refit_epochs
  params$patience <- refit_epochs + 1L
  params$lr_patience <- refit_epochs + 1L
  params$refit_fixed_epochs <- TRUE
  params$duan_apply <- FALSE
  params
}

dkp_candidate_profiles <- function() {
  data.frame(
    profile = c("auto_current", "compact_kernel", "balanced_kernel",
                "wide_kernel", "regularized", "low_mix", "high_mix"),
    matern_hidden = c(NA_integer_, 32L, 64L, 96L, 64L, 64L, 96L),
    init_mix_mean = c(NA_real_, 0.15, 0.35, 0.45, 0.35, 0.05, 0.60),
    tab_dropout = c(NA_real_, 0.05, 0.08, 0.08, 0.15, 0.08, 0.10),
    patch_dropout = c(NA_real_, 0.05, 0.08, 0.08, 0.10, 0.08, 0.10),
    stringsAsFactors = FALSE
  )
}

dkp_tune_grid <- function(train_df) {
  transforms <- dkp_tune_target_transforms
  y <- train_df[[response_name_wadoux]]
  if (!all(is.finite(y) & y > 0)) {
    transforms <- setdiff(transforms, "log")
  }
  if (length(transforms) == 0L) transforms <- "identity"
  profiles <- dkp_candidate_profiles()
  rows <- list()
  for (tf in transforms) {
    for (i in seq_len(nrow(profiles))) {
      p <- profiles[i, , drop = FALSE]
      rows[[length(rows) + 1L]] <- data.frame(
        candidate_id = length(rows) + 1L,
        candidate_label = paste(tf, p$profile, sep = "__"),
        target_transform = tf,
        profile = p$profile,
        matern_hidden = p$matern_hidden,
        init_mix_mean = p$init_mix_mean,
        tab_dropout = p$tab_dropout,
        patch_dropout = p$patch_dropout,
        stringsAsFactors = FALSE
      )
    }
  }
  grid <- do.call(rbind, rows)
  key_cols <- c("target_transform", "matern_hidden", "init_mix_mean",
                "tab_dropout", "patch_dropout")
  keys <- apply(grid[, key_cols, drop = FALSE], 1, paste, collapse = "|")
  grid <- grid[!duplicated(keys), , drop = FALSE]
  if (is.finite(dkp_tune_max_candidates) && nrow(grid) > dkp_tune_max_candidates) {
    grid <- grid[seq_len(dkp_tune_max_candidates), , drop = FALSE]
  }
  grid$candidate_id <- seq_len(nrow(grid))
  grid
}

dkp_params_from_candidate <- function(candidate, candidate_seed) {
  params <- dkp_params
  set_num <- function(field) {
    val <- suppressWarnings(as.numeric(candidate[[field]]))
    if (length(val) == 1L && is.finite(val)) params[[field]] <<- val
  }
  set_int <- function(field) {
    val <- suppressWarnings(as.integer(candidate[[field]]))
    if (length(val) == 1L && is.finite(val)) params[[field]] <<- val
  }
  params$target_transform <- as.character(candidate$target_transform)
  set_int("matern_hidden")
  set_num("init_mix_mean")
  set_num("tab_dropout")
  set_num("patch_dropout")
  params$train_seed <- as.integer(candidate_seed)
  params
}

dkp_candidate_weights <- function(scores,
                                  weighting = dkp_ensemble_weighting,
                                  temperature = dkp_ensemble_temperature) {
  n <- length(scores)
  if (n == 0L) return(numeric())
  if (n == 1L) return(1)
  scores <- as.numeric(scores)
  finite <- is.finite(scores)
  if (!any(finite)) return(rep(1 / n, n))
  worst <- max(scores[finite], na.rm = TRUE)
  scores[!finite] <- worst + abs(worst) + 1
  if (identical(weighting, "uniform")) {
    w <- rep(1, n)
  } else if (identical(weighting, "inverse")) {
    gap <- scores - min(scores, na.rm = TRUE) + temperature
    w <- 1 / pmax(gap, 1e-8)
  } else {
    z <- -(scores - min(scores, na.rm = TRUE)) / temperature
    z <- pmax(z, -700)
    w <- exp(z)
  }
  if (!any(is.finite(w)) || sum(w, na.rm = TRUE) <= 0) {
    return(rep(1 / n, n))
  }
  w / sum(w, na.rm = TRUE)
}

dkp_metric_score <- function(rmse, rmse_sd = NA_real_) {
  rmse + dkp_rmse_sd_weight * ifelse(is.finite(rmse_sd), rmse_sd, 0)
}

dkp_empty_rf_inner <- function(status = "not_available", error = NA_character_) {
  list(
    pred = NULL,
    metrics = list(ME = NA_real_, RMSE = NA_real_, r2 = NA_real_, MEC = NA_real_),
    fold_RMSE = NA_real_,
    status = status,
    error = error
  )
}

dkp_inner_rf_baseline <- function(train_core, val_df, pool = NULL,
                                  fold_ids = NULL, inner_folds = 1L) {
  out <- tryCatch({
    if (is.finite(inner_folds) && inner_folds > 1L &&
        !is.null(pool) && !is.null(fold_ids) &&
        length(fold_ids) == nrow(pool)) {
      preds <- rep(NA_real_, nrow(pool))
      fold_rmse <- numeric()
      for (fold in sort(unique(fold_ids))) {
        val_idx <- which(fold_ids == fold)
        tr_idx <- which(fold_ids != fold)
        rf <- fit_rf_default_wadoux(pool[tr_idx, , drop = FALSE], form_rf)
        preds[val_idx] <- predict_rf_default_wadoux(
          rf, pool[val_idx, , drop = FALSE])
        m_fold <- wadoux_eval(
          obs = pool[[response_name_wadoux]][val_idx],
          pred = preds[val_idx])
        fold_rmse <- c(fold_rmse, m_fold$RMSE)
      }
      m <- wadoux_eval(obs = pool[[response_name_wadoux]], pred = preds)
      list(pred = preds, metrics = m, fold_RMSE = fold_rmse,
           status = "ok", error = NA_character_)
    } else {
      rf <- fit_rf_default_wadoux(train_core, form_rf)
      preds <- predict_rf_default_wadoux(rf, val_df)
      m <- wadoux_eval(obs = val_df[[response_name_wadoux]], pred = preds)
      list(pred = preds, metrics = m, fold_RMSE = m$RMSE,
           status = "ok", error = NA_character_)
    }
  }, error = function(e) {
    dkp_empty_rf_inner("error", conditionMessage(e))
  })
  if (is.null(out$metrics)) {
    out$metrics <- dkp_empty_rf_inner()$metrics
  }
  out
}

dkp_selection_score <- function(val_rmse, rmse_sd, rf_inner_rmse) {
  base <- dkp_metric_score(val_rmse, rmse_sd)
  if (identical(dkp_selection_objective, "paired_rf_delta") &&
      is.finite(val_rmse) && is.finite(rf_inner_rmse)) {
    return((val_rmse - rf_inner_rmse) +
             dkp_rmse_sd_weight * ifelse(is.finite(rmse_sd), rmse_sd, 0))
  }
  base
}

dkp_noharm_anchor_indices <- function(score_df, ok) {
  profile <- as.character(score_df$profile)
  transform <- as.character(score_df$target_transform)
  conservative_profile <- profile %in% c("auto_current", "compact_kernel")
  non_log <- transform %in% c("auto", "identity")
  tiers <- list(
    which(ok & profile == "auto_current" & non_log),
    which(ok & conservative_profile & non_log),
    which(ok & conservative_profile),
    which(ok & non_log)
  )
  for (idx in tiers) {
    if (length(idx) > 0L) return(idx)
  }
  which(ok)
}

dkp_eval_candidate_cv <- function(cand, train_core, val_df, test_df,
                                  calibration_patch_cache, iter, i,
                                  protocol, model_label,
                                  inner_folds = NULL, fold_ids = NULL,
                                  rf_inner = NULL) {
  params_i <- dkp_params_from_candidate(
    cand, candidate_seed = train_seed + 10000L * iter + i)
  pool <- rbind(train_core, val_df)
  n_pool <- nrow(pool)
  if (is.null(inner_folds)) {
    inner_folds <- min(dkp_tune_inner_folds, max(1L, floor(n_pool / 2L)))
  }
  fold_rmse <- numeric()
  fold_me <- numeric()
  out_i <- NULL
  error_msg <- NA_character_
  status <- "ok"
  started <- Sys.time()
  if (inner_folds <= 1L) {
    out_i <- tryCatch(
      run_dkp_on_split(
        train_core = train_core, val_df = val_df, test_df = test_df,
        calibration_patch_cache = calibration_patch_cache,
        params = params_i),
      error = function(e) e
    )
    if (inherits(out_i, "error")) {
      status <- "error"
      error_msg <- conditionMessage(out_i)
    } else {
      m <- wadoux_eval(obs = val_df[[response_name_wadoux]],
                       pred = out_i$pred_val)
      fold_rmse <- m$RMSE
      fold_me <- m$ME
    }
  } else {
    if (is.null(fold_ids) || length(fold_ids) != n_pool) {
      set.seed(train_seed + 810000L + iter * 1000L)
      fold_ids <- caret::createFolds(
        pool[[response_name_wadoux]], k = inner_folds,
        list = FALSE, returnTrain = FALSE)
    }
    preds <- rep(NA_real_, n_pool)
    for (fold in sort(unique(fold_ids))) {
      val_idx <- which(fold_ids == fold)
      tr_idx <- which(fold_ids != fold)
      fold_params <- params_i
      fold_params$train_seed <- as.integer(params_i$train_seed) + fold
      fold_out <- tryCatch(
        run_dkp_on_split(
          train_core = pool[tr_idx, , drop = FALSE],
          val_df = pool[val_idx, , drop = FALSE],
          test_df = test_df,
          calibration_patch_cache = calibration_patch_cache,
          params = fold_params),
        error = function(e) e
      )
      if (inherits(fold_out, "error")) {
        status <- "error"
        error_msg <- conditionMessage(fold_out)
        break
      }
      preds[val_idx] <- fold_out$pred_val
      m_fold <- wadoux_eval(obs = pool[[response_name_wadoux]][val_idx],
                            pred = fold_out$pred_val)
      fold_rmse <- c(fold_rmse, m_fold$RMSE)
      fold_me <- c(fold_me, m_fold$ME)
    }
    if (identical(status, "ok")) {
      if (any(!is.finite(preds))) {
        status <- "error"
        error_msg <- "non-finite inner-CV predictions"
      } else {
        m <- wadoux_eval(obs = pool[[response_name_wadoux]], pred = preds)
      }
    }
  }
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  if (identical(status, "ok") && exists("m", inherits = FALSE)) {
    val_rmse <- m$RMSE
    val_me <- m$ME
    val_r2 <- m$r2
    val_mec <- m$MEC
  } else {
    val_rmse <- val_me <- val_r2 <- val_mec <- NA_real_
  }
  rmse_sd <- if (length(fold_rmse) > 1L) stats::sd(fold_rmse, na.rm = TRUE) else NA_real_
  rf_metrics <- if (!is.null(rf_inner) && !is.null(rf_inner$metrics)) {
    rf_inner$metrics
  } else {
    dkp_empty_rf_inner()$metrics
  }
  rf_inner_rmse <- suppressWarnings(as.numeric(rf_metrics$RMSE))
  rf_inner_me <- suppressWarnings(as.numeric(rf_metrics$ME))
  rf_inner_r2 <- suppressWarnings(as.numeric(rf_metrics$r2))
  rf_inner_mec <- suppressWarnings(as.numeric(rf_metrics$MEC))
  delta_rmse_vs_rf <- if (is.finite(val_rmse) && is.finite(rf_inner_rmse)) {
    val_rmse - rf_inner_rmse
  } else {
    NA_real_
  }
  rmse_selection_score <- if (identical(status, "ok")) {
    dkp_metric_score(val_rmse, rmse_sd)
  } else {
    Inf
  }
  row_i <- cbind(
    data.frame(
      iteration = iter, scenario = scenario, protocol = protocol,
      model = model_label, selected = FALSE, ensemble_member = FALSE,
      ensemble_rank = NA_integer_, ensemble_weight = NA_real_,
      selection_rank = NA_integer_, status = status,
      elapsed_seconds = elapsed, stringsAsFactors = FALSE),
    cand
  )
  row_i$error <- error_msg
  row_i$val_ME <- val_me
  row_i$val_RMSE <- val_rmse
  row_i$val_r2 <- val_r2
  row_i$val_MEC <- val_mec
  row_i$RMSE_sd <- rmse_sd
  row_i$fold_RMSE <- paste(sprintf("%.6f", fold_rmse), collapse = "|")
  row_i$fold_ME <- paste(sprintf("%.6f", fold_me), collapse = "|")
  row_i$inner_folds <- inner_folds
  row_i$selection_objective <- dkp_selection_objective
  row_i$rf_inner_status <- if (!is.null(rf_inner$status)) rf_inner$status else NA_character_
  row_i$rf_inner_error <- if (!is.null(rf_inner$error)) rf_inner$error else NA_character_
  row_i$rf_inner_ME <- rf_inner_me
  row_i$rf_inner_RMSE <- rf_inner_rmse
  row_i$rf_inner_r2 <- rf_inner_r2
  row_i$rf_inner_MEC <- rf_inner_mec
  row_i$rf_inner_fold_RMSE <- if (!is.null(rf_inner$fold_RMSE)) {
    paste(sprintf("%.6f", as.numeric(rf_inner$fold_RMSE)), collapse = "|")
  } else {
    NA_character_
  }
  row_i$delta_RMSE_vs_RF <- delta_rmse_vs_rf
  row_i$rmse_selection_score <- rmse_selection_score
  row_i$selection_score <- if (identical(status, "ok")) {
    dkp_selection_score(val_rmse, rmse_sd, rf_inner_rmse)
  } else {
    Inf
  }
  list(row = row_i, params = params_i, out = out_i)
}

tune_dkp_on_split <- function(train_core, val_df, test_df,
                              calibration_patch_cache,
                              iter, protocol, model_label) {
  grid <- dkp_tune_grid(train_core)
  pool <- rbind(train_core, val_df)
  inner_folds <- min(dkp_tune_inner_folds, max(1L, floor(nrow(pool) / 2L)))
  fold_ids <- NULL
  if (inner_folds > 1L) {
    set.seed(train_seed + 810000L + iter * 1000L)
    fold_ids <- caret::createFolds(
      pool[[response_name_wadoux]], k = inner_folds,
      list = FALSE, returnTrain = FALSE)
  }
  # Persist the inner OOF fold map so the literal-RF blend lambda is selected
  # on EXACTLY the same leakage-free OOF folds the gate scored DKP on, and so
  # the in-gate RF (dkp_inner_rf_baseline below) is the same RF used for blend.
  attr_fold_ids <- fold_ids
  rf_inner <- dkp_inner_rf_baseline(
    train_core = train_core, val_df = val_df, pool = pool,
    fold_ids = fold_ids, inner_folds = inner_folds)
  evals <- vector("list", nrow(grid))
  for (i in seq_len(nrow(grid))) {
    cand <- grid[i, , drop = FALSE]
    cat(sprintf("[DKP tune] iter=%d %s candidate %d/%d: %s\n",
                iter, model_label, i, nrow(grid), cand$candidate_label))
    evals[[i]] <- dkp_eval_candidate_cv(
      cand = cand, train_core = train_core, val_df = val_df,
      test_df = test_df, calibration_patch_cache = calibration_patch_cache,
      iter = iter, i = i, protocol = protocol, model_label = model_label,
      inner_folds = inner_folds, fold_ids = fold_ids, rf_inner = rf_inner)
  }
  score_df <- bind_rows_fill_dkp(lapply(evals, `[[`, "row"))
  ok <- is.finite(score_df$selection_score)
  if (!any(ok)) {
    stop("DKP tuning failed: no candidate completed successfully.")
  }
  ord <- order(score_df$selection_score, score_df$val_RMSE,
               score_df$candidate_id, na.last = TRUE)
  ord <- ord[ok[ord]]
  score_df$selection_rank[ord] <- seq_along(ord)
  pre_best_i <- ord[1L]
  best_i <- pre_best_i
  anchor_idx <- dkp_noharm_anchor_indices(score_df, ok)
  anchor_i <- if (length(anchor_idx) > 0L) {
    anchor_ord <- order(score_df$selection_score[anchor_idx],
                        score_df$val_RMSE[anchor_idx],
                        score_df$candidate_id[anchor_idx],
                        na.last = TRUE)
    anchor_idx[anchor_ord[1L]]
  } else {
    NA_integer_
  }
  noharm_anchor_score <- NA_real_
  noharm_anchor_val_rmse <- NA_real_
  noharm_margin <- NA_real_
  noharm_delta_vs_anchor <- NA_real_
  noharm_applied <- FALSE
  noharm_reason <- if (isTRUE(dkp_noharm_policy)) {
    "no_anchor_available"
  } else {
    "disabled"
  }
	  if (isTRUE(dkp_noharm_policy) && is.finite(anchor_i)) {
	    noharm_anchor_score <- as.numeric(score_df$selection_score[anchor_i])
    noharm_anchor_val_rmse <- as.numeric(score_df$val_RMSE[anchor_i])
    anchor_scale <- noharm_anchor_val_rmse
    if (!is.finite(anchor_scale)) anchor_scale <- abs(noharm_anchor_score)
    if (!is.finite(anchor_scale)) anchor_scale <- 0
    noharm_margin <- dkp_noharm_margin_abs +
      dkp_noharm_margin_rel * abs(anchor_scale)
    if (!identical(pre_best_i, anchor_i)) {
      noharm_delta_vs_anchor <- noharm_anchor_score -
        as.numeric(score_df$selection_score[pre_best_i])
      if (!is.finite(noharm_delta_vs_anchor) ||
          noharm_delta_vs_anchor < noharm_margin) {
        best_i <- anchor_i
        noharm_applied <- TRUE
        noharm_reason <- sprintf(
          "fallback_to_anchor:improvement=%.6f<margin=%.6f",
          ifelse(is.finite(noharm_delta_vs_anchor),
                 noharm_delta_vs_anchor, NA_real_),
          noharm_margin)
      } else {
        noharm_reason <- sprintf(
          "aggressive_passed:improvement=%.6f>=margin=%.6f",
          noharm_delta_vs_anchor, noharm_margin)
      }
    } else {
      noharm_delta_vs_anchor <- 0
	      noharm_reason <- "selected_candidate_is_anchor"
	    }
	  }
	  pre_rf_noharm_i <- best_i
	  rf_noharm_preselected_delta <- as.numeric(
	    score_df$delta_RMSE_vs_RF[pre_rf_noharm_i])
	  rf_noharm_anchor_delta <- if (is.finite(anchor_i)) {
	    as.numeric(score_df$delta_RMSE_vs_RF[anchor_i])
	  } else {
	    NA_real_
	  }
	  rf_noharm_margin <- NA_real_
	  rf_noharm_required_delta <- NA_real_
	  rf_noharm_failed <- FALSE
	  rf_noharm_applied <- FALSE
	  rf_noharm_reason <- if (isTRUE(dkp_rf_noharm_policy)) {
	    "not_assessed"
	  } else {
	    "disabled"
	  }
	  if (isTRUE(dkp_rf_noharm_policy)) {
	    rf_scale <- as.numeric(score_df$rf_inner_RMSE[pre_rf_noharm_i])
	    if (!is.finite(rf_scale)) rf_scale <- 0
	    rf_noharm_margin <- dkp_rf_noharm_margin_abs +
	      dkp_rf_noharm_margin_rel * abs(rf_scale)
	    rf_noharm_required_delta <- -rf_noharm_margin
	    if (is.finite(rf_noharm_preselected_delta)) {
	      if (rf_noharm_preselected_delta <= rf_noharm_required_delta) {
	        rf_noharm_reason <- sprintf(
	          "passed_rf_inner:delta=%.6f<=required=%.6f",
	          rf_noharm_preselected_delta, rf_noharm_required_delta)
	      } else {
	        rf_noharm_failed <- TRUE
	        if (is.finite(anchor_i)) {
	          if (!identical(best_i, anchor_i)) {
	            best_i <- anchor_i
	            rf_noharm_applied <- TRUE
	            rf_noharm_reason <- sprintf(
	              "fallback_to_anchor:delta=%.6f>required=%.6f",
	              rf_noharm_preselected_delta, rf_noharm_required_delta)
	          } else {
	            rf_noharm_reason <- sprintf(
	              "selected_anchor_still_fails_rf:delta=%.6f>required=%.6f",
	              rf_noharm_preselected_delta, rf_noharm_required_delta)
	          }
	        } else {
	          rf_noharm_reason <- sprintf(
	            "rf_inner_failed_no_anchor:delta=%.6f>required=%.6f",
	            rf_noharm_preselected_delta, rf_noharm_required_delta)
	        }
	      }
	    } else {
	      rf_noharm_reason <- "no_valid_delta_RMSE_vs_RF"
	    }
	  }
	  score_df$noharm_policy <- dkp_noharm_policy
  score_df$noharm_anchor_candidate_id <- if (is.finite(anchor_i)) {
    score_df$candidate_id[anchor_i]
  } else {
    NA_integer_
  }
  score_df$noharm_anchor_candidate_label <- if (is.finite(anchor_i)) {
    as.character(score_df$candidate_label[anchor_i])
  } else {
    NA_character_
  }
  score_df$noharm_anchor_score <- noharm_anchor_score
  score_df$noharm_anchor_val_rmse <- noharm_anchor_val_rmse
  score_df$noharm_margin_abs <- dkp_noharm_margin_abs
  score_df$noharm_margin_rel <- dkp_noharm_margin_rel
  score_df$noharm_margin <- noharm_margin
  score_df$noharm_delta_vs_anchor <- noharm_delta_vs_anchor
	  score_df$noharm_applied <- noharm_applied
	  score_df$noharm_restrict_ensemble <- dkp_noharm_restrict_ensemble
	  score_df$noharm_reason <- noharm_reason
	  score_df$rf_noharm_policy <- dkp_rf_noharm_policy
	  score_df$rf_noharm_preselected_candidate_id <-
	    score_df$candidate_id[pre_rf_noharm_i]
	  score_df$rf_noharm_preselected_candidate_label <-
	    score_df$candidate_label[pre_rf_noharm_i]
	  score_df$rf_noharm_preselected_delta_RMSE_vs_RF <-
	    rf_noharm_preselected_delta
	  score_df$rf_noharm_anchor_candidate_id <- if (is.finite(anchor_i)) {
	    score_df$candidate_id[anchor_i]
	  } else {
	    NA_integer_
	  }
	  score_df$rf_noharm_anchor_candidate_label <- if (is.finite(anchor_i)) {
	    as.character(score_df$candidate_label[anchor_i])
	  } else {
	    NA_character_
	  }
	  score_df$rf_noharm_anchor_delta_RMSE_vs_RF <- rf_noharm_anchor_delta
	  score_df$rf_noharm_margin_abs <- dkp_rf_noharm_margin_abs
	  score_df$rf_noharm_margin_rel <- dkp_rf_noharm_margin_rel
	  score_df$rf_noharm_margin <- rf_noharm_margin
	  score_df$rf_noharm_required_delta_RMSE_vs_RF <- rf_noharm_required_delta
	  score_df$rf_noharm_failed <- rf_noharm_failed
	  score_df$rf_noharm_applied <- rf_noharm_applied
	  score_df$rf_noharm_restrict_ensemble <- dkp_rf_noharm_restrict_ensemble
	  score_df$rf_noharm_reason <- rf_noharm_reason
	  restrict_ensemble_to_anchor <- (
	    isTRUE(noharm_applied) && isTRUE(dkp_noharm_restrict_ensemble)
	  ) || (
	    isTRUE(rf_noharm_applied) && isTRUE(dkp_rf_noharm_restrict_ensemble)
	  )
	  ensemble_ord <- if (isTRUE(restrict_ensemble_to_anchor) &&
	                      length(anchor_idx) > 0L) {
    anchor_idx[order(score_df$selection_score[anchor_idx],
                     score_df$val_RMSE[anchor_idx],
                     score_df$candidate_id[anchor_idx],
                     na.last = TRUE)]
  } else {
    ord
  }
  top_idx <- unique(c(best_i, ensemble_ord))
  top_idx <- top_idx[seq_len(min(dkp_ensemble_top_k, length(top_idx)))]
  weights <- dkp_candidate_weights(score_df$selection_score[top_idx])
  score_df$ensemble_member[top_idx] <- TRUE
  score_df$ensemble_rank[top_idx] <- seq_along(top_idx)
  score_df$ensemble_weight[top_idx] <- weights
  score_df$selected[best_i] <- TRUE
  top_candidates <- vector("list", length(top_idx))
  for (j in seq_along(top_idx)) {
    row_j <- score_df[top_idx[j], , drop = FALSE]
    params_j <- evals[[top_idx[j]]]$params
    if (is.null(evals[[top_idx[j]]]$out) ||
        inherits(evals[[top_idx[j]]]$out, "error") ||
        dkp_tune_inner_folds > 1L) {
      params_j$train_seed <- as.integer(params_j$train_seed) + 90000L + j
      out_j <- run_dkp_on_split(
        train_core = train_core, val_df = val_df, test_df = test_df,
        calibration_patch_cache = calibration_patch_cache,
        params = params_j)
    } else {
      out_j <- evals[[top_idx[j]]]$out
    }
    top_candidates[[j]] <- list(row = row_j, params = params_j, out = out_j)
  }
  best_out <- top_candidates[[1L]]$out
  best_params <- top_candidates[[1L]]$params
  if (!is.null(best_out$diagnostics)) {
    best_out$diagnostics$tune_selected_candidate_id <- score_df$candidate_id[best_i]
    best_out$diagnostics$tune_selected_candidate_label <-
      score_df$candidate_label[best_i]
    best_out$diagnostics$tune_selected_val_rmse <- score_df$val_RMSE[best_i]
    best_out$diagnostics$tune_selected_score <- score_df$selection_score[best_i]
    best_out$diagnostics$tune_selection_objective <- dkp_selection_objective
    best_out$diagnostics$tune_selected_rf_inner_RMSE <-
      score_df$rf_inner_RMSE[best_i]
    best_out$diagnostics$tune_selected_delta_RMSE_vs_RF <-
      score_df$delta_RMSE_vs_RF[best_i]
    best_out$diagnostics$tune_selected_rmse_selection_score <-
      score_df$rmse_selection_score[best_i]
    best_out$diagnostics$tune_candidate_n <- nrow(score_df)
    best_out$diagnostics$tune_inner_folds <- inner_folds
    best_out$diagnostics$tune_rmse_sd_weight <- dkp_rmse_sd_weight
    best_out$diagnostics$noharm_policy <- dkp_noharm_policy
    best_out$diagnostics$noharm_anchor_candidate_id <-
      score_df$noharm_anchor_candidate_id[best_i]
    best_out$diagnostics$noharm_anchor_candidate_label <-
      score_df$noharm_anchor_candidate_label[best_i]
    best_out$diagnostics$noharm_anchor_score <- noharm_anchor_score
    best_out$diagnostics$noharm_anchor_val_rmse <- noharm_anchor_val_rmse
    best_out$diagnostics$noharm_margin_abs <- dkp_noharm_margin_abs
    best_out$diagnostics$noharm_margin_rel <- dkp_noharm_margin_rel
    best_out$diagnostics$noharm_margin <- noharm_margin
    best_out$diagnostics$noharm_delta_vs_anchor <- noharm_delta_vs_anchor
	    best_out$diagnostics$noharm_applied <- noharm_applied
	    best_out$diagnostics$noharm_restrict_ensemble <- dkp_noharm_restrict_ensemble
	    best_out$diagnostics$noharm_reason <- noharm_reason
	    best_out$diagnostics$rf_noharm_policy <- dkp_rf_noharm_policy
	    best_out$diagnostics$rf_noharm_preselected_candidate_id <-
	      score_df$candidate_id[pre_rf_noharm_i]
	    best_out$diagnostics$rf_noharm_preselected_candidate_label <-
	      score_df$candidate_label[pre_rf_noharm_i]
	    best_out$diagnostics$rf_noharm_preselected_delta_RMSE_vs_RF <-
	      rf_noharm_preselected_delta
	    best_out$diagnostics$rf_noharm_anchor_candidate_id <-
	      if (is.finite(anchor_i)) score_df$candidate_id[anchor_i] else NA_integer_
	    best_out$diagnostics$rf_noharm_anchor_candidate_label <-
	      if (is.finite(anchor_i)) as.character(score_df$candidate_label[anchor_i]) else NA_character_
	    best_out$diagnostics$rf_noharm_anchor_delta_RMSE_vs_RF <-
	      rf_noharm_anchor_delta
	    best_out$diagnostics$rf_noharm_margin_abs <- dkp_rf_noharm_margin_abs
	    best_out$diagnostics$rf_noharm_margin_rel <- dkp_rf_noharm_margin_rel
	    best_out$diagnostics$rf_noharm_margin <- rf_noharm_margin
	    best_out$diagnostics$rf_noharm_required_delta_RMSE_vs_RF <-
	      rf_noharm_required_delta
	    best_out$diagnostics$rf_noharm_failed <- rf_noharm_failed
	    best_out$diagnostics$rf_noharm_applied <- rf_noharm_applied
	    best_out$diagnostics$rf_noharm_restrict_ensemble <-
	      dkp_rf_noharm_restrict_ensemble
	    best_out$diagnostics$rf_noharm_reason <- rf_noharm_reason
	  }
  list(out = best_out, params = best_params, tuning = score_df,
       top_candidates = top_candidates,
       oof_pool = pool, oof_fold_ids = attr_fold_ids,
       oof_inner_folds = inner_folds)
}

dkp_apply_carried_smearing <- function(out, selection_out, y_test = NULL) {# -----------------------------------------------------------------------------
# Literal-RF floor + OOF-selected continuous blend lambda.
#
# Given a DKP prediction vector and an inner OOF setup, fit the LITERAL Wadoux
# RF (fit_rf_default_wadoux / predict_rf_default_wadoux) on the SAME inner OOF
# folds, choose lambda in dkp_rf_blend_lambda_grid (incl 0) that minimises
# leakage-free OOF RMSE of  yhat = lambda*DKP + (1-lambda)*RF , then return the
# RF model fit on the FULL calibration set plus the chosen lambda. When DKP does
# not beat RF on OOF, the grid picks lambda=0 and the emitted prediction becomes
# the LITERAL Wadoux RF prediction (do-no-harm floor, guaranteed >= RF up to OOF
# noise; lambda is selected ONLY on leakage-free OOF data).
# -----------------------------------------------------------------------------
dkp_oof_rf_blend_lambda <- function(pool, fold_ids, dkp_oof_pred) {
  # pool: data.frame with response; fold_ids: integer fold per pool row (OOF);
  # dkp_oof_pred: DKP OOF predictions aligned to pool rows (same as gate used).
  y <- as.numeric(pool[[response_name_wadoux]])
  rf_oof <- rep(NA_real_, nrow(pool))
  for (fold in sort(unique(fold_ids))) {
    val_idx <- which(fold_ids == fold)
    tr_idx  <- which(fold_ids != fold)
    rf <- fit_rf_default_wadoux(pool[tr_idx, , drop = FALSE], form_rf)
    rf_oof[val_idx] <- predict_rf_default_wadoux(rf, pool[val_idx, , drop = FALSE])
  }
  ok <- is.finite(y) & is.finite(rf_oof) & is.finite(dkp_oof_pred)
  best_lambda <- 0; best_rmse <- Inf
  for (lam in dkp_rf_blend_lambda_grid) {
    blend <- lam * dkp_oof_pred[ok] + (1 - lam) * rf_oof[ok]
    rmse  <- sqrt(mean((y[ok] - blend) ^ 2, na.rm = TRUE))
    # strict '<' so ties (notably DKP-not-better) keep the SMALLEST lambda,
    # i.e. the grid is ordered ascending and lambda=0 (pure RF) wins ties.
    if (is.finite(rmse) && rmse < best_rmse - 1e-12) {
      best_rmse <- rmse; best_lambda <- lam
    }
  }
  list(lambda = best_lambda, oof_rmse = best_rmse,
       rf_oof = rf_oof, n_used = sum(ok))
}

dkp_apply_rf_floor <- function(out, calibration_full, test_df, blend) {
  # Emit  lambda*DKP + (1-lambda)*literal_RF  on the test set, where the literal
  # RF is fit_rf_default_wadoux on the FULL calibration set (== scored RF_500).
  lam <- suppressWarnings(as.numeric(blend$lambda))
  if (!is.finite(lam)) lam <- 0
  rf_full <- fit_rf_default_wadoux(calibration_full, form_rf)
  rf_test <- predict_rf_default_wadoux(rf_full, test_df)
  if (!is.null(out$pred_test) && length(out$pred_test) == length(rf_test)) {
    out$pred_test <- lam * as.numeric(out$pred_test) + (1 - lam) * rf_test
  } else if (is.null(out$pred_test)) {
    out$pred_test <- rf_test
  }
  out$diagnostics$rf_blend_lambda        <- lam
  out$diagnostics$rf_blend_oof_rmse       <- blend$oof_rmse
  out$diagnostics$rf_blend_n_used         <- blend$n_used
  out$diagnostics$rf_floor_is_literal_rf  <- isTRUE(lam == 0)
  out$diagnostics$rf_floor_reverts_to     <- "literal_wadoux_rf"
  y_test <- test_df[[response_name_wadoux]]
  if (length(y_test) == length(out$pred_test)) {
    out$metrics_test <- data.frame(
      N = length(y_test),
      RMSE = sqrt(mean((y_test - out$pred_test) ^ 2, na.rm = TRUE)),
      Bias = mean(out$pred_test - y_test, na.rm = TRUE),
      stringsAsFactors = FALSE)
  }
  out
}


  diag <- selection_out$diagnostics
  applied <- !is.null(diag$duan_applied) && isTRUE(diag$duan_applied)
  S <- suppressWarnings(as.numeric(diag$duan_S))
  if (!isTRUE(applied) || !is.finite(S) || S <= 0) {
    out$diagnostics$duan_carried_from_inner_val <- FALSE
    return(out)
  }
  for (nm in c("pred_test", "pred_val")) {
    if (!is.null(out[[nm]])) out[[nm]] <- out[[nm]] * S
  }
  out$diagnostics$duan_carried_from_inner_val <- TRUE
  out$diagnostics$duan_applied <- TRUE
  out$diagnostics$duan_S <- S
  out$diagnostics$duan_reason <- paste(
    "carried_inner_val:", diag$duan_reason)
  if (!is.null(y_test) && length(y_test) == length(out$pred_test)) {
    out$metrics_test <- data.frame(
      N = length(y_test),
      RMSE = sqrt(mean((y_test - out$pred_test) ^ 2, na.rm = TRUE)),
      Bias = mean(out$pred_test - y_test, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  out
}

refit_dkp_topk_full_calibration <- function(tuned, calibration_full, test_df,
                                            calibration_patch_cache) {
  candidates <- tuned$top_candidates
  if (is.null(candidates) || length(candidates) == 0L) {
    candidates <- list(list(
      row = data.frame(candidate_id = tuned$out$diagnostics$tune_selected_candidate_id,
                       candidate_label = tuned$out$diagnostics$tune_selected_candidate_label,
                       selection_score = tuned$out$diagnostics$tune_selected_score,
                       stringsAsFactors = FALSE),
      params = tuned$params,
      out = tuned$out
    ))
  }
  candidates <- candidates[seq_len(min(dkp_ensemble_top_k, length(candidates)))]
  scores <- vapply(candidates, function(x) {
    s <- suppressWarnings(as.numeric(x$row$selection_score[1L]))
    if (is.finite(s)) s else Inf
  }, numeric(1))
  requested_weights <- dkp_candidate_weights(scores)
  member_results <- lapply(seq_along(candidates), function(i) {
    tryCatch({
      cand <- candidates[[i]]
      refit_monitor <- dkp_refit_monitor_df(calibration_full)
      refit_params <- dkp_refit_params_from_selection(
        cand$out$diagnostics, base_params = cand$params)
      if (!is.null(refit_params$train_seed)) {
        refit_params$train_seed <- as.integer(refit_params$train_seed) +
          700000L + 1009L * i
      }
      out <- run_dkp_on_split(
        calibration_full, refit_monitor, test_df,
        calibration_patch_cache, params = refit_params)
      out <- dkp_apply_carried_smearing(
        out, cand$out, y_test = test_df[[response_name_wadoux]])
      out$diagnostics$tune_selected_candidate_id <-
        tuned$out$diagnostics$tune_selected_candidate_id
      out$diagnostics$tune_selected_candidate_label <-
        tuned$out$diagnostics$tune_selected_candidate_label
      out$diagnostics$tune_selected_val_rmse <-
        tuned$out$diagnostics$tune_selected_val_rmse
      out$diagnostics$tune_selected_score <-
        tuned$out$diagnostics$tune_selected_score
      out$diagnostics$tune_candidate_n <- tuned$out$diagnostics$tune_candidate_n
      out$diagnostics$tune_inner_folds <- dkp_tune_inner_folds
      out$diagnostics$tune_rmse_sd_weight <- dkp_rmse_sd_weight
      for (f in c(
        "tune_selection_objective", "tune_selected_rf_inner_RMSE",
        "tune_selected_delta_RMSE_vs_RF",
	        "tune_selected_rmse_selection_score", "noharm_policy",
	        "noharm_anchor_candidate_id", "noharm_anchor_candidate_label",
	        "noharm_anchor_score", "noharm_anchor_val_rmse",
	        "noharm_margin_abs", "noharm_margin_rel", "noharm_margin",
	        "noharm_delta_vs_anchor", "noharm_applied",
	        "noharm_restrict_ensemble", "noharm_reason",
	        "rf_noharm_policy", "rf_noharm_preselected_candidate_id",
	        "rf_noharm_preselected_candidate_label",
	        "rf_noharm_preselected_delta_RMSE_vs_RF",
	        "rf_noharm_anchor_candidate_id",
	        "rf_noharm_anchor_candidate_label",
	        "rf_noharm_anchor_delta_RMSE_vs_RF",
	        "rf_noharm_margin_abs", "rf_noharm_margin_rel",
	        "rf_noharm_margin", "rf_noharm_required_delta_RMSE_vs_RF",
	        "rf_noharm_failed", "rf_noharm_applied",
	        "rf_noharm_restrict_ensemble", "rf_noharm_reason"
	      )) {
        if (!is.null(tuned$out$diagnostics[[f]])) {
          out$diagnostics[[f]] <- tuned$out$diagnostics[[f]]
        }
      }
      list(status = "ok", out = out,
           candidate_id = cand$row$candidate_id[1L],
           candidate_label = as.character(cand$row$candidate_label[1L]),
           score = scores[i])
    }, error = function(e) {
      cand <- candidates[[i]]
      list(status = paste("failed:", conditionMessage(e)), out = NULL,
           candidate_id = cand$row$candidate_id[1L],
           candidate_label = as.character(cand$row$candidate_label[1L]),
           score = scores[i])
    })
  })
  ok <- vapply(member_results, function(x) identical(x$status, "ok"),
               logical(1))
  if (!any(ok)) {
    out <- tuned$out
    out$diagnostics$refit_full_calibration <- FALSE
    out$diagnostics$refit_status <- paste(
      "failed:", paste(vapply(member_results, `[[`, character(1), "status"),
                       collapse = " | "))
    return(out)
  }
  member_ok <- member_results[ok]
  weights <- requested_weights[ok]
  weights <- weights / sum(weights, na.rm = TRUE)
  out <- member_ok[[1L]]$out
  weighted <- function(name) {
    mat <- do.call(cbind, lapply(member_ok, function(x) as.numeric(x$out[[name]])))
    as.numeric(mat %*% weights)
  }
  for (nm in c("pred_test", "pred_val", "pred_test_base", "pred_val_base")) {
    if (!is.null(out[[nm]]) && length(member_ok) > 1L) {
      out[[nm]] <- weighted(nm)
    }
  }
  y_test <- test_df[[response_name_wadoux]]
  if (length(y_test) == length(out$pred_test)) {
    out$metrics_test <- data.frame(
      N = length(y_test),
      RMSE = sqrt(mean((y_test - out$pred_test) ^ 2, na.rm = TRUE)),
      Bias = mean(out$pred_test - y_test, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  out$diagnostics$refit_full_calibration <- TRUE
  out$diagnostics$refit_monitor_overlap <- TRUE
  out$diagnostics$refit_status <- if (all(ok)) "ok" else "partial_ok"
  out$diagnostics$refit_mode <- if (length(member_ok) > 1L) {
    "topk_candidate_ensemble"
  } else {
    "single_candidate_refit"
  }
  out$diagnostics$refit_ensemble_top_k_requested <- dkp_ensemble_top_k
  out$diagnostics$refit_ensemble_applied <- length(member_ok) > 1L
  out$diagnostics$refit_ensemble_size <- length(member_ok)
  out$diagnostics$refit_ensemble_weighting <- dkp_ensemble_weighting
  out$diagnostics$refit_ensemble_temperature <- dkp_ensemble_temperature
  out$diagnostics$refit_ensemble_member_ids <- paste(
    vapply(member_ok, `[[`, integer(1), "candidate_id"), collapse = "|")
  out$diagnostics$refit_ensemble_members <- paste(
    vapply(member_ok, `[[`, character(1), "candidate_label"),
    collapse = " || ")
  out$diagnostics$refit_ensemble_scores <- paste(
    sprintf("%.8f", vapply(member_ok, `[[`, numeric(1), "score")),
    collapse = "|")
  out$diagnostics$refit_ensemble_weights <- paste(
    sprintf("%.8f", weights), collapse = "|")
  out$diagnostics$refit_ensemble_status <- paste(
    vapply(member_results, `[[`, character(1), "status"), collapse = " || ")
  out$diagnostics$refit_train_n <- nrow(calibration_full)
  out$diagnostics$refit_monitor_n <- 1L
  out
}

# -----------------------------------------------------------------------------
# Metrics bookkeeping
# -----------------------------------------------------------------------------

emit_metric_rows_dkp <- function(obs, pred_map, protocol, iter, scenario) {
  rows <- list()
  for (model_label in names(pred_map)) {
    pred <- pred_map[[model_label]]
    if (is.null(pred) || length(pred) == 0L) next
    m <- wadoux_eval(obs = obs, pred = pred)
    rows[[length(rows) + 1L]] <- data.frame(
      iteration = iter, scenario = scenario, protocol = protocol,
      model = model_label,
      ME = m$ME, RMSE = m$RMSE, r2 = m$r2, MEC = m$MEC,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}

bind_rows_fill_dkp <- function(rows) {
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) return(data.frame())
  all_cols <- unique(unlist(lapply(rows, names), use.names = FALSE))
  aligned <- lapply(rows, function(x) {
    missing <- setdiff(all_cols, names(x))
    for (col in missing) x[[col]] <- NA
    x[, all_cols, drop = FALSE]
  })
  do.call(rbind, aligned)
}

scalar_diag_row_dkp <- function(diag, iter, protocol, model, stage,
                                train_n, val_n, test_n,
                                refit_full_calibration = FALSE,
                                refit_monitor_overlap = FALSE) {
  row <- data.frame(
    iteration = iter, scenario = scenario, protocol = protocol,
    model = model, stage = stage,
    train_n = train_n, val_n = val_n, test_n = test_n,
    refit_full_calibration = refit_full_calibration,
    refit_monitor_overlap = refit_monitor_overlap,
    stringsAsFactors = FALSE
  )
  if (is.null(diag) || length(diag) == 0L) return(row)
  keep <- vapply(diag, function(x) {
    !is.list(x) && length(x) == 1L
  }, logical(1))
  if (!any(keep)) return(row)
  vals <- as.data.frame(diag[keep], stringsAsFactors = FALSE)
  cbind(row, vals)
}

paired_delta_rows_dkp <- function(metric_df) {
  if (is.null(metric_df) || nrow(metric_df) == 0L) return(data.frame())
  key_cols <- c("iteration", "scenario", "protocol")
  pairs <- list(
    c("DeepKernelProcess_400", "RF_400"),
    c("DeepKernelProcess_500", "RF_500"),
    c("DeepKernelProcess_500", "DeepKernelProcess_400"),
    c("RF_500", "RF_400")
  )
  groups <- split(metric_df, interaction(metric_df[, key_cols], drop = TRUE))
  rows <- list()
  for (grp in groups) {
    for (pair in pairs) {
      candidate <- grp[grp$model == pair[1], , drop = FALSE]
      reference <- grp[grp$model == pair[2], , drop = FALSE]
      if (nrow(candidate) == 0L || nrow(reference) == 0L) next
      candidate <- candidate[1, , drop = FALSE]
      reference <- reference[1, , drop = FALSE]
      rows[[length(rows) + 1L]] <- data.frame(
        iteration = candidate$iteration,
        scenario = candidate$scenario,
        protocol = candidate$protocol,
        candidate_model = pair[1],
        reference_model = pair[2],
        delta_ME = candidate$ME - reference$ME,
        delta_RMSE = candidate$RMSE - reference$RMSE,
        delta_r2 = candidate$r2 - reference$r2,
        delta_MEC = candidate$MEC - reference$MEC,
        stringsAsFactors = FALSE
      )
    }
  }
  bind_rows_fill_dkp(rows)
}

prediction_audit_designbased_dkp <- function(test_df, pred_map, iter,
                                             diag_400 = NULL,
                                             diag_500 = NULL) {
  if (!isTRUE(dkp_save_prediction_audit)) return(data.frame())
  obs <- test_df[[response_name_wadoux]]
  out <- data.frame(
    iteration = iter,
    scenario = scenario,
    protocol = "DesignBased",
    split_role = "design_test",
    wadoux_row_id = row_id_dkp(test_df),
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
  add_diag <- function(prefix, diag) {
    if (is.null(diag)) return()
    fields <- c(
      "tune_selected_candidate_id", "tune_selected_candidate_label",
      "tune_selected_val_rmse", "tune_selected_score",
      "tune_selection_objective", "tune_selected_rf_inner_RMSE",
      "tune_selected_delta_RMSE_vs_RF",
      "tune_selected_rmse_selection_score",
      "tune_candidate_n", "tune_inner_folds", "tune_rmse_sd_weight",
      "noharm_policy", "noharm_anchor_candidate_id",
      "noharm_anchor_candidate_label", "noharm_anchor_score",
	      "noharm_anchor_val_rmse", "noharm_margin_abs",
	      "noharm_margin_rel", "noharm_margin",
	      "noharm_delta_vs_anchor", "noharm_applied",
	      "noharm_restrict_ensemble", "noharm_reason",
	      "rf_noharm_policy", "rf_noharm_preselected_candidate_id",
	      "rf_noharm_preselected_candidate_label",
	      "rf_noharm_preselected_delta_RMSE_vs_RF",
	      "rf_noharm_anchor_candidate_id",
	      "rf_noharm_anchor_candidate_label",
	      "rf_noharm_anchor_delta_RMSE_vs_RF",
	      "rf_noharm_margin_abs", "rf_noharm_margin_rel",
	      "rf_noharm_margin", "rf_noharm_required_delta_RMSE_vs_RF",
	      "rf_noharm_failed", "rf_noharm_applied",
	      "rf_noharm_restrict_ensemble", "rf_noharm_reason",
	      "target_transform", "matern_hidden", "init_mix_mean",
      "tab_dropout", "patch_dropout", "best_epoch", "main_epochs_done",
      "duan_applied", "duan_carried_from_inner_val", "duan_S",
      "duan_reason", "refit_status", "refit_mode",
      "refit_ensemble_top_k_requested", "refit_ensemble_applied",
      "refit_ensemble_size", "refit_ensemble_weighting",
      "refit_ensemble_temperature", "refit_ensemble_members",
      "refit_ensemble_member_ids", "refit_ensemble_scores",
      "refit_ensemble_weights", "refit_ensemble_status"
    )
    for (f in fields) {
      val <- diag[[f]]
      if (is.null(val) || length(val) != 1L) val <- NA
      out[[paste(prefix, f, sep = "_")]] <<- val
    }
  }
  add_diag("DeepKernelProcess_400", diag_400)
  add_diag("DeepKernelProcess_500", diag_500)
  out
}

split_audit_designbased_dkp <- function(calibration_full, inner, test_df, iter) {
  if (!isTRUE(dkp_save_prediction_audit)) return(data.frame())
  mk <- function(df, role) {
    data.frame(
      iteration = iter,
      scenario = scenario,
      protocol = "DesignBased",
      split_role = role,
      wadoux_row_id = row_id_dkp(df),
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

# -----------------------------------------------------------------------------
# Protocol evaluators (same shape as NMK runner)
# -----------------------------------------------------------------------------

eval_designbased_dkp <- function(valuetable, iter, calibration_patch_cache) {
  val_srs <- sample_design_test_dkp(valuetable, sample_size)
  obs     <- val_srs[[response_name_wadoux]]
  inner <- make_inner_val_split_dkp(valuetable, val_frac = val_frac,
                                    seed = train_seed + iter)
  calibration_full <- valuetable
  test_df <- val_srs
  test_df$..patch_cache_row_id <- rep(NA_integer_, nrow(test_df))

  pred_map <- list()
  diag_rows <- list()
  tune_rows <- list()
  if ("RF" %in% models) {
    rf_400 <- fit_rf_default_wadoux(inner$train, form_rf)
    pred_map$RF_400 <- predict_rf_default_wadoux(rf_400, val_srs)

    rf_500 <- fit_rf_default_wadoux(calibration_full, form_rf)
    pred_map$RF_500 <- predict_rf_default_wadoux(rf_500, val_srs)
  }
  if ("DeepKernelProcess" %in% models) {
    selected_params <- dkp_params
    if (isTRUE(dkp_tune)) {
      tuned <- tune_dkp_on_split(
        train_core = inner$train, val_df = inner$val, test_df = test_df,
        calibration_patch_cache = calibration_patch_cache,
        iter = iter, protocol = "DesignBased",
        model_label = "DeepKernelProcess_400")
      out_400 <- tuned$out
      selected_params <- tuned$params
      tune_rows[[length(tune_rows) + 1L]] <- tuned$tuning
    } else {
      out_400 <- run_dkp_on_split(inner$train, inner$val, test_df,
                                  calibration_patch_cache)
    }
    pred_map$DeepKernelProcess_400 <- out_400$pred_test
    diag_rows[[length(diag_rows) + 1L]] <- scalar_diag_row_dkp(
      out_400$diagnostics, iter, "DesignBased", "DeepKernelProcess_400",
      "inner_selection", nrow(inner$train), nrow(inner$val), nrow(test_df)
    )

    if (isTRUE(dkp_refit_full_calibration)) {
      if (exists("tuned", inherits = FALSE)) {
        out_500 <- refit_dkp_topk_full_calibration(
          tuned, calibration_full, test_df, calibration_patch_cache)
      } else {
        refit_monitor <- dkp_refit_monitor_df(calibration_full)
        refit_params <- dkp_refit_params_from_selection(
          out_400$diagnostics, base_params = selected_params)
        out_500 <- run_dkp_on_split(calibration_full, refit_monitor, test_df,
                                    calibration_patch_cache,
                                    params = refit_params)
        out_500$diagnostics$refit_full_calibration <- TRUE
        out_500$diagnostics$refit_status <- "ok"
        out_500$diagnostics$refit_mode <- "single_candidate_refit"
        out_500$diagnostics$refit_train_n <- nrow(calibration_full)
        out_500$diagnostics$refit_monitor_n <- nrow(refit_monitor)
      }
      # DO-NO-HARM FLOOR: select a continuous blend lambda toward the LITERAL
      # Wadoux RF on leakage-free inner OOF data (grid incl 0), then emit
      # lambda*DKP + (1-lambda)*literal_RF. lambda=0 => emitted == literal
      # Wadoux RF (guaranteed >= RF up to OOF noise). Only runs when the gate
      # produced an OOF fold map (tuned path); otherwise emit DKP unchanged.
      if (exists("tuned", inherits = FALSE) &&
          !is.null(tuned$oof_pool) && !is.null(tuned$oof_fold_ids) &&
          length(tuned$oof_fold_ids) == nrow(tuned$oof_pool)) {
        dkp_oof_pred <- predict_rf_default_wadoux  # placeholder, replaced below
        # DKP OOF predictions on the SAME inner folds the gate used:
        dkp_oof_pred <- rep(NA_real_, nrow(tuned$oof_pool))
        for (fold in sort(unique(tuned$oof_fold_ids))) {
          v_idx <- which(tuned$oof_fold_ids == fold)
          t_idx <- which(tuned$oof_fold_ids != fold)
          fold_params <- selected_params
          fold_params$train_seed <- as.integer(selected_params$train_seed) + fold
          fo <- tryCatch(run_dkp_on_split(
            train_core = tuned$oof_pool[t_idx, , drop = FALSE],
            val_df     = tuned$oof_pool[v_idx, , drop = FALSE],
            test_df    = tuned$oof_pool[v_idx, , drop = FALSE],
            calibration_patch_cache = calibration_patch_cache,
            params = fold_params), error = function(e) NULL)
          if (!is.null(fo)) dkp_oof_pred[v_idx] <- fo$pred_test
        }
        blend <- dkp_oof_rf_blend_lambda(
          pool = tuned$oof_pool, fold_ids = tuned$oof_fold_ids,
          dkp_oof_pred = dkp_oof_pred)
        out_500 <- dkp_apply_rf_floor(out_500, calibration_full, test_df, blend)
      }
      pred_map$DeepKernelProcess_500 <- out_500$pred_test
      diag_rows[[length(diag_rows) + 1L]] <- scalar_diag_row_dkp(
        out_500$diagnostics, iter, "DesignBased", "DeepKernelProcess_500",
        "full_calibration_refit", nrow(calibration_full),
        out_500$diagnostics$refit_monitor_n,
        nrow(test_df), refit_full_calibration = TRUE,
        refit_monitor_overlap = TRUE
      )
    }
  }
  metrics <- emit_metric_rows_dkp(obs, pred_map, "DesignBased", iter, scenario)
  diag_400 <- if (exists("out_400", inherits = FALSE)) out_400$diagnostics else NULL
  diag_500 <- if (exists("out_500", inherits = FALSE)) out_500$diagnostics else NULL
  audit_rows <- prediction_audit_designbased_dkp(
    test_df, pred_map, iter, diag_400 = diag_400, diag_500 = diag_500)
  split_rows <- split_audit_designbased_dkp(calibration_full, inner, test_df, iter)
  list(metrics = metrics, diagnostics = bind_rows_fill_dkp(diag_rows),
       paired_deltas = paired_delta_rows_dkp(metrics),
       tuning = bind_rows_fill_dkp(tune_rows),
       prediction_audit = audit_rows,
       split_audit = split_rows)
}

eval_randomkfold_dkp <- function(valuetable, iter, calibration_patch_cache) {
  flds <- caret::createFolds(valuetable[[response_name_wadoux]],
                             k = random_k, list = TRUE, returnTrain = FALSE)
  preds_rf  <- rep(NA_real_, nrow(valuetable))
  preds_dkp <- rep(NA_real_, nrow(valuetable))
  for (j in seq_along(flds)) {
    id <- flds[[j]]
    training_data   <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]
    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(training_data, form_rf)
      preds_rf[id] <- predict_rf_default_wadoux(rf, validation_data)
    }
    if ("DeepKernelProcess" %in% models) {
      inner <- make_inner_val_split_dkp(training_data, val_frac = val_frac,
                                        seed = train_seed + iter * 1000L + j)
      out <- run_dkp_on_split(inner$train, inner$val, validation_data,
                              calibration_patch_cache)
      preds_dkp[id] <- out$pred_test
    }
  }
  pred_map <- list()
  if ("RF" %in% models) pred_map$RF <- preds_rf
  if ("DeepKernelProcess" %in% models) pred_map$DeepKernelProcess <- preds_dkp
  list(metrics = emit_metric_rows_dkp(valuetable[[response_name_wadoux]],
                                      pred_map, "RandomKFold", iter, scenario))
}

eval_spatialkfold_dkp <- function(valuetable, iter, calibration_patch_cache) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  fold_ids <- unique(spatial_folds)
  preds_rf  <- rep(NA_real_, nrow(valuetable))
  preds_dkp <- rep(NA_real_, nrow(valuetable))
  for (j in seq_along(fold_ids)) {
    id <- which(spatial_folds == fold_ids[j])
    training_data   <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]
    if (nrow(training_data) < 10L) next
    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(training_data, form_rf)
      preds_rf[id] <- predict_rf_default_wadoux(rf, validation_data)
    }
    if ("DeepKernelProcess" %in% models) {
      inner <- make_inner_val_split_dkp(training_data, val_frac = val_frac,
                                        seed = train_seed + iter * 1000L + fold_ids[j])
      out <- run_dkp_on_split(inner$train, inner$val, validation_data,
                              calibration_patch_cache)
      preds_dkp[id] <- out$pred_test
    }
  }
  pred_map <- list()
  if ("RF" %in% models) pred_map$RF <- preds_rf
  if ("DeepKernelProcess" %in% models) pred_map$DeepKernelProcess <- preds_dkp
  list(metrics = emit_metric_rows_dkp(valuetable[[response_name_wadoux]],
                                      pred_map, "SpatialKFold", iter, scenario))
}

eval_bloocv_dkp <- function(valuetable, iter, calibration_patch_cache) {
  nb_iteration <- bloo_groups * bloo_pixels
  dkp_cap_raw <- Sys.getenv("DKP_BLOO_DKP_PIXELS", unset = "")
  if (!nzchar(dkp_cap_raw)) {
    dkp_cap_raw <- Sys.getenv("DKP_BLOO_NMK_PIXELS", unset = "")
  }
  dkp_cap <- if (nzchar(dkp_cap_raw)) suppressWarnings(as.integer(dkp_cap_raw))
             else as.integer(min(nb_iteration, bloo_groups * 10L))
  if (!is.finite(dkp_cap) || dkp_cap < 1L) dkp_cap <- nb_iteration

  set.seed(train_seed + iter)
  ri <- val_dist_km * 1000
  dkp_used <- 0L
  pred_rows <- list()
  for (j in seq_len(nb_iteration)) {
    point_in_range <- FALSE
    id_focal <- NA_integer_
    while (!isTRUE(point_in_range)) {
      id_focal <- sample.int(nrow(valuetable), size = 1L)
      focal_point  <- valuetable[id_focal, , drop = FALSE]
      training_tmp <- valuetable[-id_focal, , drop = FALSE]
      training_tmp <- exclude_by_radius_wadoux(training_tmp, focal_point, ri)
      if (nrow(training_tmp) < 10L) next
      if (!point_within_predictor_range_wadoux(training_tmp, focal_point, predList)) next
      point_in_range <- TRUE
    }
    focal_point   <- valuetable[id_focal, , drop = FALSE]
    training_full <- valuetable[-id_focal, , drop = FALSE]
    training_full <- exclude_by_radius_wadoux(training_full, focal_point, ri)

    pred_rf  <- NA_real_;  pred_dkp <- NA_real_
    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(training_full, form_rf)
      pred_rf <- predict_rf_default_wadoux(rf, focal_point)
    }
    if ("DeepKernelProcess" %in% models && dkp_used < dkp_cap) {
      inner <- make_inner_val_split_dkp(training_full, val_frac = val_frac,
                                        seed = train_seed + iter * 1000L + j)
      out <- run_dkp_on_split(inner$train, inner$val, focal_point,
                              calibration_patch_cache)
      pred_dkp <- out$pred_test[1]
      dkp_used <- dkp_used + 1L
    }
    pred_rows[[j]] <- data.frame(
      focal_id = id_focal,
      obs      = focal_point[[response_name_wadoux]],
      pred_rf  = pred_rf,
      pred_dkp = pred_dkp,
      stringsAsFactors = FALSE
    )
  }
  pred_df <- do.call(rbind, pred_rows)
  pred_df$group <- caret::createFolds(seq_len(nrow(pred_df)), k = bloo_groups,
                                      list = FALSE, returnTrain = FALSE)
  agg <- function(df_sub, col) {
    idx <- is.finite(df_sub[[col]]) & is.finite(df_sub$obs)
    if (sum(idx) < 2L) return(NULL)
    wadoux_eval(obs = df_sub$obs[idx], pred = df_sub[[col]][idx])
  }
  group_ids <- sort(unique(pred_df$group))
  rows <- list()
  emit_one <- function(mlabel, col) {
    if (!(mlabel %in% models)) return(NULL)
    per_group <- lapply(group_ids, function(g) {
      agg(pred_df[pred_df$group == g, , drop = FALSE], col)
    })
    ok <- per_group[!vapply(per_group, is.null, logical(1))]
    if (length(ok) == 0L) return(NULL)
    stacked <- do.call(rbind, ok)
    data.frame(
      iteration = iter, scenario = scenario, protocol = "BLOOCV",
      model = mlabel,
      ME = mean(stacked$ME, na.rm = TRUE),
      RMSE = mean(stacked$RMSE, na.rm = TRUE),
      r2 = mean(stacked$r2, na.rm = TRUE),
      MEC = mean(stacked$MEC, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  rows <- list(emit_one("RF", "pred_rf"),
               emit_one("DeepKernelProcess", "pred_dkp"))
  rows <- rows[!vapply(rows, is.null, logical(1))]
  list(metrics = if (length(rows) == 0L) data.frame() else do.call(rbind, rows))
}

# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------

save_dkp_checkpoint <- function(iter, metric_rows, diagnostic_rows = list(),
                                tuning_rows = list(),
                                prediction_audit_rows = list(),
                                split_audit_rows = list()) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  write.csv(df, file.path(results_dir, "dkp_all_results.csv"), row.names = FALSE)

  deltas <- paired_delta_rows_dkp(df)
  if (nrow(deltas) > 0L) {
    write.csv(deltas, file.path(results_dir, "dkp_paired_deltas.csv"),
              row.names = FALSE)
  }

  diag_df <- bind_rows_fill_dkp(diagnostic_rows)
  if (nrow(diag_df) > 0L) {
    write.csv(diag_df, file.path(results_dir, "dkp_fold_diagnostics.csv"),
              row.names = FALSE)
  }

  tune_df <- bind_rows_fill_dkp(tuning_rows)
  if (nrow(tune_df) > 0L) {
    write.csv(tune_df, file.path(results_dir, "dkp_tuning_scores.csv"),
              row.names = FALSE)
  }

  audit_df <- bind_rows_fill_dkp(prediction_audit_rows)
  if (nrow(audit_df) > 0L) {
    write.csv(audit_df, file.path(results_dir, "dkp_prediction_audit.csv"),
              row.names = FALSE)
  }

  split_df <- bind_rows_fill_dkp(split_audit_rows)
  if (nrow(split_df) > 0L) {
    write.csv(split_df, file.path(results_dir, "dkp_split_audit.csv"),
              row.names = FALSE)
  }

  write.csv(
    data.frame(
      completed_iteration = iter, total_iterations = n_iter,
      scenario = scenario,
      protocols = paste(protocols, collapse = ","),
      models = paste(models, collapse = ","),
      completed_at = format(Sys.time(), tz = "", usetz = TRUE),
      stringsAsFactors = FALSE),
    file.path(results_dir, "dkp_checkpoint_status.csv"), row.names = FALSE)
}

summarise_and_save_dkp <- function(metric_rows) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  by_protocol <- aggregate(
    cbind(ME, RMSE, r2, MEC) ~ scenario + protocol + model,
    data = df,
    FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                        sd   = sd(x,   na.rm = TRUE))
  )
  stats_df <- do.call(cbind, lapply(c("ME", "RMSE", "r2", "MEC"), function(m) {
    mat <- by_protocol[[m]]
    setNames(as.data.frame(mat), c(paste0(m, "_mean"), paste0(m, "_sd")))
  }))
  out <- cbind(by_protocol[, c("scenario", "protocol", "model")], stats_df)
  write.csv(out, file.path(results_dir, "dkp_summary_by_protocol.csv"),
            row.names = FALSE)
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

write.csv(
  data.frame(
    scenario = scenario, protocols = paste(protocols, collapse = ","),
    models = paste(models, collapse = ","),
    include_population = include_population, sample_size = sample_size,
    n_iter = n_iter, val_frac = val_frac, val_dist_km = val_dist_km,
    random_k = random_k, bloo_groups = bloo_groups,
    bloo_test_pixels = bloo_pixels, patch_size = patch_size,
    device = device, profile = profile, train_seed = train_seed,
    fair_tune = dkp_tune,
    refit_full_calibration = dkp_refit_full_calibration,
    tune_max_candidates = dkp_tune_max_candidates,
	    tune_inner_folds = dkp_tune_inner_folds,
	    selection_objective = dkp_selection_objective,
		    noharm_policy = dkp_noharm_policy,
		    noharm_margin_abs = dkp_noharm_margin_abs,
		    noharm_margin_rel = dkp_noharm_margin_rel,
		    noharm_restrict_ensemble = dkp_noharm_restrict_ensemble,
		    rf_noharm_policy = dkp_rf_noharm_policy,
		    rf_noharm_margin_abs = dkp_rf_noharm_margin_abs,
		    rf_noharm_margin_rel = dkp_rf_noharm_margin_rel,
		    rf_noharm_restrict_ensemble = dkp_rf_noharm_restrict_ensemble,
		    ensemble_top_k = dkp_ensemble_top_k,
    ensemble_weighting = dkp_ensemble_weighting,
    ensemble_temperature = dkp_ensemble_temperature,
    rmse_sd_weight = dkp_rmse_sd_weight,
    save_prediction_audit = dkp_save_prediction_audit,
    DKP_TUNE = raw_env_dkp("DKP_TUNE", "TRUE"),
    DKP_TUNE_TARGET_TRANSFORMS = raw_env_dkp(
      "DKP_TUNE_TARGET_TRANSFORMS", "auto,identity,log"),
    DKP_TUNE_MAX_CANDIDATES = raw_env_dkp("DKP_TUNE_MAX_CANDIDATES", "18"),
	    DKP_TUNE_INNER_FOLDS = raw_env_dkp("DKP_TUNE_INNER_FOLDS", "3"),
	    DKP_SELECTION_OBJECTIVE = raw_env_dkp(
	      "DKP_SELECTION_OBJECTIVE", "paired_rf_delta"),
	    DKP_NOHARM_POLICY = raw_env_dkp("DKP_NOHARM_POLICY", "TRUE"),
	    DKP_NOHARM_MARGIN_ABS = raw_env_dkp("DKP_NOHARM_MARGIN_ABS", "0.05"),
	    DKP_NOHARM_MARGIN_REL = raw_env_dkp("DKP_NOHARM_MARGIN_REL", "0"),
		    DKP_NOHARM_RESTRICT_ENSEMBLE = raw_env_dkp(
		      "DKP_NOHARM_RESTRICT_ENSEMBLE", "TRUE"),
		    DKP_RF_NOHARM_POLICY = raw_env_dkp("DKP_RF_NOHARM_POLICY", "TRUE"),
		    DKP_RF_NOHARM_MARGIN_ABS = raw_env_dkp(
		      "DKP_RF_NOHARM_MARGIN_ABS", "0.05"),
		    DKP_RF_NOHARM_MARGIN_REL = raw_env_dkp(
		      "DKP_RF_NOHARM_MARGIN_REL", "0"),
		    DKP_RF_NOHARM_RESTRICT_ENSEMBLE = raw_env_dkp(
		      "DKP_RF_NOHARM_RESTRICT_ENSEMBLE", "TRUE"),
		    DKP_REFIT_FULL_CALIBRATION = raw_env_dkp("DKP_REFIT_FULL_CALIBRATION", "TRUE"),
    DKP_ENSEMBLE_TOP_K = raw_env_dkp("DKP_ENSEMBLE_TOP_K", "3"),
    DKP_ENSEMBLE_WEIGHTING = raw_env_dkp("DKP_ENSEMBLE_WEIGHTING", "softmax"),
    DKP_ENSEMBLE_TEMPERATURE = raw_env_dkp("DKP_ENSEMBLE_TEMPERATURE", "0.75"),
    DKP_RMSE_SD_WEIGHT = raw_env_dkp("DKP_RMSE_SD_WEIGHT", "0.25"),
    DKP_SAVE_PREDICTION_AUDIT = raw_env_dkp("DKP_SAVE_PREDICTION_AUDIT", "TRUE"),
    stringsAsFactors = FALSE),
  file.path(results_dir, "dkp_config.csv"), row.names = FALSE)

all_metric_rows <- list()
all_diag_rows <- list()
all_tune_rows <- list()
all_prediction_audit_rows <- list()
all_split_audit_rows <- list()
for (iter in seq_len(n_iter)) {
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("DKP vs RF | scenario=%s | iteration %d / %d\n",
              scenario, iter, n_iter))
  cat(sprintf("==================================================\n"))

  valuetable <- na.omit(as.data.frame(sample_calibration_dkp()))
  valuetable$..patch_cache_row_id <- seq_len(nrow(valuetable))

  calibration_patch_cache <- NULL
  if ("DeepKernelProcess" %in% models) {
    calibration_patch_cache <- dkp_env$build_pointpatch_patch_cache(
      context        = dkp_env$wadoux_context,
      calibration_df = valuetable,
      patch_size     = patch_size,
      label          = sprintf("scenario=%s iter=%d patches", scenario, iter)
    )
  }

  iter_rows <- list()
  iter_diag <- list()
  iter_tune <- list()
  iter_prediction_audit <- list()
  iter_split_audit <- list()
  if ("DesignBased"  %in% protocols) {
    cat("\n--- DesignBased ---\n")
    out <- eval_designbased_dkp(valuetable, iter, calibration_patch_cache)
    iter_rows[["DesignBased"]] <- out$metrics
    iter_diag[["DesignBased"]] <- out$diagnostics
    iter_tune[["DesignBased"]] <- out$tuning
    iter_prediction_audit[["DesignBased"]] <- out$prediction_audit
    iter_split_audit[["DesignBased"]] <- out$split_audit
  }
  if ("RandomKFold"  %in% protocols) {
    cat("\n--- RandomKFold ---\n")
    iter_rows[["RandomKFold"]]  <- eval_randomkfold_dkp(valuetable, iter, calibration_patch_cache)$metrics
  }
  if ("SpatialKFold" %in% protocols) {
    cat("\n--- SpatialKFold ---\n")
    iter_rows[["SpatialKFold"]] <- eval_spatialkfold_dkp(valuetable, iter, calibration_patch_cache)$metrics
  }
  if ("BLOOCV"       %in% protocols) {
    cat("\n--- BLOOCV ---\n")
    iter_rows[["BLOOCV"]]       <- eval_bloocv_dkp(valuetable, iter, calibration_patch_cache)$metrics
  }

  iter_df <- do.call(rbind, iter_rows[!vapply(iter_rows, is.null, logical(1))])
  if (!is.null(iter_df) && nrow(iter_df) > 0L) {
    print(iter_df)
    all_metric_rows[[iter]] <- iter_df
  }
  iter_diag_df <- bind_rows_fill_dkp(iter_diag)
  if (nrow(iter_diag_df) > 0L) {
    all_diag_rows[[iter]] <- iter_diag_df
  }
  iter_tune_df <- bind_rows_fill_dkp(iter_tune)
  if (nrow(iter_tune_df) > 0L) {
    all_tune_rows[[iter]] <- iter_tune_df
  }
  iter_prediction_audit_df <- bind_rows_fill_dkp(iter_prediction_audit)
  if (nrow(iter_prediction_audit_df) > 0L) {
    all_prediction_audit_rows[[iter]] <- iter_prediction_audit_df
  }
  iter_split_audit_df <- bind_rows_fill_dkp(iter_split_audit)
  if (nrow(iter_split_audit_df) > 0L) {
    all_split_audit_rows[[iter]] <- iter_split_audit_df
  }

  if (iter %% checkpoint_every == 0L || iter == n_iter) {
    save_dkp_checkpoint(iter, all_metric_rows, all_diag_rows, all_tune_rows,
                        all_prediction_audit_rows, all_split_audit_rows)
    summarise_and_save_dkp(all_metric_rows)
  }
}

cat("\n[DKP runner] Done. Results written to: ", results_dir, "\n", sep = "")
