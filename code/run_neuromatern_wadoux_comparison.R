# =============================================================================
# run_neuromatern_wadoux_comparison.R
#
# Head-to-head benchmark between:
#   - RF (ranger defaults; Wadoux-style baseline)
#   - NeuroMaternKrige (NMK — the non-stationary neural Matern kriging net
#                       defined in code/NeuroMaternKrige.R)
#
# NMK runs STANDALONE (the regression-kriging over RF mode that once lived
# here was removed: it is classical Odeh 1995 / Hengl 2007 and would defeat
# the purpose of evaluating the novel non-stationary Matern head as a
# primary predictor. NMK has to stand on its own feet.)
#
# Design mirrors run_wadoux_style_rf_conv_comparison.R but is deliberately
# slimmer. Shared guarantees:
#
#   * same calibration sample per iteration (simple-random, regular-grid or
#     clustered-random, matching Wadoux 2021);
#   * same outer folds per protocol (DesignBased, RandomKFold, SpatialKFold,
#     BLOOCV; optional Population);
#   * same Wadoux metrics (ME, RMSE, Pearson^2, MEC);
#   * same inner train/validation split inside each outer training set.
#
# Environment overrides (all optional):
#   NMK_SCENARIO            random | regular_grid | clustered_random
#   NMK_PROTOCOLS           comma list of {DesignBased,RandomKFold,
#                                         SpatialKFold,BLOOCV,Population}
#   NMK_MODELS              comma list of {RF,NeuroMaternKrige}
#   NMK_N_ITER              number of calibration samples
#   NMK_SAMPLE_SIZE         calibration size per iteration
#   NMK_VAL_FRAC            inner validation fraction for NMK
#   NMK_VAL_DIST_KM         spatial CV / BLOOCV exclusion radius (km)
#   NMK_RANDOM_K            k of random K-fold
#   NMK_BLOO_GROUPS         BLOOCV groups
#   NMK_BLOO_TEST_PIXELS    BLOOCV test pixels per group
#   NMK_DEVICE              cpu | mps | cuda
#   NMK_TRAIN_SEED          seed (rng + torch)
#   NMK_RESULTS_DIR         where to write CSVs
#   NMK_PROFILE             default | quick
#   NMK_EPOCHS              max main-loop epochs
#   NMK_PATIENCE, NMK_LR, NMK_BATCH_SIZE, NMK_WEIGHT_DECAY,
#   NMK_LAMBDA_SMOOTH, NMK_LAMBDA_ANCHOR, NMK_BETA_INIT,
#   NMK_KRIGING_HIDDEN, NMK_INIT_MIX_MEAN, NMK_K_NEIGHBORS,
#   NMK_N_SEEDS             structural ensemble size, default 3
#                           (set 1 to disable ensemble for debugging)
#   NMK_TARGET_TRANSFORM    auto (default, Shapiro-Wilk on y_train) |
#                           identity | log
#   NMK_TUNE                TRUE by default for DesignBased NMK
#   NMK_AUTOTUNE_V2         TRUE (default) expands the candidate policies
#                           beyond the conservative NMK profile grid
#   NMK_AUTOTUNE_V4         TRUE (default) adds RF-capacity neural forest
#                           profiles: more trees, deeper soft trees and
#                           higher-resolution quantile bases, still no RF stack
#   NMK_TUNE_TARGET_TRANSFORMS comma list from {auto,identity,log}
#   NMK_TUNE_FULL_GRID      TRUE (default) evaluates the full deduplicated grid
#   NMK_TUNE_MAX_CANDIDATES only used when NMK_TUNE_FULL_GRID=FALSE
#   NMK_TUNE_INNER_FOLDS    inner CV folds for candidate scoring (default 3)
#   NMK_SELECTION_OBJECTIVE rmse | paired_rf_delta (default paired_rf_delta)
#   NMK_NOHARM_POLICY, NMK_NOHARM_MARGIN_ABS, NMK_NOHARM_MARGIN_REL
#   NMK_NOHARM_RESTRICT_ENSEMBLE keep fallback ensembles conservative
#   NMK_NUM_WORKERS         requested workers; NMK keeps torch tuning in-process
#                           because forked torch candidate parallelism is unsafe
#   NMK_TUNE_N_SEEDS        default 1 for candidate scoring; refit restores
#                           NMK_N_SEEDS unless NMK_REFIT_N_SEEDS is set
#   NMK_REFIT_FULL_CALIBRATION TRUE by default
#   NMK_SAVE_VARIOGRAMS     TRUE (default) writes semivariogram CSV/PNG diagnostics
#   NMK_SAVE_PREDICTION_AUDIT TRUE (default) writes split/prediction audits
#   NMK_BLOO_NMK_PIXELS     cap on NMK retrains inside BLOOCV
#   NMK_CHECKPOINT_EVERY
# =============================================================================

rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")

# -----------------------------------------------------------------------------
# Environment-variable parsing
# -----------------------------------------------------------------------------

parse_bool_env_nmk <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}

parse_num_env_nmk <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) default else val
}

parse_list_env_nmk <- function(name, default) {
  raw <- Sys.getenv(name, unset = paste(default, collapse = ","))
  vals <- trimws(unlist(strsplit(raw, ",")))
  vals[vals != ""]
}

parse_num_list_env_nmk <- function(name, default = NULL) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  vals <- suppressWarnings(as.numeric(trimws(unlist(strsplit(raw, ",")))))
  vals[is.finite(vals)]
}

parse_int_list_env_nmk <- function(name, default = NULL) {
  vals <- parse_num_list_env_nmk(name, default)
  if (is.null(vals)) return(NULL)
  as.integer(vals)
}

raw_env_nmk <- function(name, default = "") {
  Sys.getenv(name, unset = default)
}

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

scenario    <- Sys.getenv("NMK_SCENARIO", unset = "random")
protocols   <- parse_list_env_nmk("NMK_PROTOCOLS",
                                  c("DesignBased", "RandomKFold",
                                    "SpatialKFold", "BLOOCV"))
models      <- parse_list_env_nmk("NMK_MODELS",
                                  c("RF", "NeuroMaternKrige"))
allowed_models <- c("RF", "NeuroMaternKrige")
invalid_models <- setdiff(models, allowed_models)
if (length(invalid_models) > 0) {
  stop("Unsupported NMK_MODELS entries: ", paste(invalid_models, collapse = ", "))
}

include_population <- "Population" %in% protocols

n_iter        <- as.integer(parse_num_env_nmk("NMK_N_ITER",       10L))
sample_size   <- as.integer(parse_num_env_nmk("NMK_SAMPLE_SIZE",  500L))
val_frac      <-             parse_num_env_nmk("NMK_VAL_FRAC",    0.20)
val_dist_km   <-             parse_num_env_nmk("NMK_VAL_DIST_KM", 350)
random_k      <- as.integer(parse_num_env_nmk("NMK_RANDOM_K",     10L))
bloo_groups   <- as.integer(parse_num_env_nmk("NMK_BLOO_GROUPS",  10L))
bloo_pixels   <- as.integer(parse_num_env_nmk("NMK_BLOO_TEST_PIXELS", 100L))
device        <- Sys.getenv("NMK_DEVICE", unset = "cpu")
train_seed    <- as.integer(parse_num_env_nmk("NMK_TRAIN_SEED",   123L))
results_dir   <- Sys.getenv("NMK_RESULTS_DIR", unset = "results/neuromatern_run")
profile       <- Sys.getenv("NMK_PROFILE", unset = "default")
checkpoint_every <- as.integer(parse_num_env_nmk("NMK_CHECKPOINT_EVERY", 1L))
nmk_tune <- parse_bool_env_nmk("NMK_TUNE", TRUE)
nmk_autotune_v2 <- parse_bool_env_nmk("NMK_AUTOTUNE_V2", TRUE)
nmk_autotune_v3 <- parse_bool_env_nmk("NMK_AUTOTUNE_V3", TRUE)
nmk_autotune_v4 <- parse_bool_env_nmk("NMK_AUTOTUNE_V4", TRUE)
if (isTRUE(nmk_autotune_v3) && !isTRUE(nmk_autotune_v2)) {
  nmk_autotune_v2 <- TRUE
}
nmk_autotune_backbone <- parse_bool_env_nmk("NMK_AUTOTUNE_BACKBONE", TRUE)
if (isTRUE(nmk_autotune_v4)) {
  nmk_autotune_v3 <- TRUE
  nmk_autotune_v2 <- TRUE
  nmk_autotune_backbone <- TRUE
}
nmk_tune_full_grid <- parse_bool_env_nmk("NMK_TUNE_FULL_GRID", TRUE)
nmk_tune_max_candidates <- as.integer(parse_num_env_nmk("NMK_TUNE_MAX_CANDIDATES", 28L))
nmk_tune_inner_folds <- as.integer(parse_num_env_nmk("NMK_TUNE_INNER_FOLDS", 3L))
if (!is.finite(nmk_tune_inner_folds) || nmk_tune_inner_folds < 1L) {
  nmk_tune_inner_folds <- 1L
}
nmk_num_workers <- as.integer(parse_num_env_nmk("NMK_NUM_WORKERS", 1L))
if (!is.finite(nmk_num_workers) || nmk_num_workers < 1L) {
  nmk_num_workers <- 1L
}
nmk_tune_n_seeds <- as.integer(parse_num_env_nmk("NMK_TUNE_N_SEEDS", 1L))
nmk_refit_full_calibration <- parse_bool_env_nmk("NMK_REFIT_FULL_CALIBRATION", TRUE)
nmk_refit_warm_start <- parse_bool_env_nmk("NMK_REFIT_WARM_START", TRUE)
nmk_refit_warm_start_epoch_mult <- parse_num_env_nmk(
  "NMK_REFIT_WARM_START_EPOCH_MULT", 0.35)
if (!is.finite(nmk_refit_warm_start_epoch_mult) ||
    nmk_refit_warm_start_epoch_mult <= 0) {
  nmk_refit_warm_start_epoch_mult <- 0.35
}
nmk_refit_warm_start_min_epochs <- as.integer(parse_num_env_nmk(
  "NMK_REFIT_WARM_START_MIN_EPOCHS", 3L))
if (!is.finite(nmk_refit_warm_start_min_epochs) ||
    nmk_refit_warm_start_min_epochs < 1L) {
  nmk_refit_warm_start_min_epochs <- 3L
}
nmk_refit_warm_start_max_epochs <- as.integer(parse_num_env_nmk(
  "NMK_REFIT_WARM_START_MAX_EPOCHS", 15L))
if (!is.finite(nmk_refit_warm_start_max_epochs) ||
    nmk_refit_warm_start_max_epochs < nmk_refit_warm_start_min_epochs) {
  nmk_refit_warm_start_max_epochs <- max(15L, nmk_refit_warm_start_min_epochs)
}
nmk_save_variograms <- parse_bool_env_nmk("NMK_SAVE_VARIOGRAMS", TRUE)
nmk_save_prediction_audit <- parse_bool_env_nmk("NMK_SAVE_PREDICTION_AUDIT", TRUE)
nmk_v3_alpha_grid <- parse_num_list_env_nmk(
  "NMK_V3_ALPHA_GRID", c(0, 0.25, 0.50, 0.75, 1.00))
nmk_v3_alpha_grid <- sort(unique(pmax(0, pmin(1, nmk_v3_alpha_grid))))
if (length(nmk_v3_alpha_grid) == 0L) {
  nmk_v3_alpha_grid <- c(0, 0.25, 0.50, 0.75, 1.00)
}
nmk_v3_calibration_policies <- tolower(parse_list_env_nmk(
  "NMK_V3_CALIBRATION_POLICIES",
  c("none", "median_bias", "mean_bias", "affine")))
nmk_v3_calibration_policies <- unique(nmk_v3_calibration_policies[
  nmk_v3_calibration_policies %in% c("none", "median_bias", "mean_bias", "affine")])
if (length(nmk_v3_calibration_policies) == 0L) {
  nmk_v3_calibration_policies <- "none"
}
nmk_v3_stage2_top_k <- as.integer(parse_num_env_nmk("NMK_V3_STAGE2_TOP_K", 5L))
if (!is.finite(nmk_v3_stage2_top_k) || nmk_v3_stage2_top_k < 0L) {
  nmk_v3_stage2_top_k <- 0L
}
nmk_v3_stage2_n_seeds <- as.integer(parse_num_env_nmk("NMK_V3_STAGE2_N_SEEDS", 3L))
if (!is.finite(nmk_v3_stage2_n_seeds) || nmk_v3_stage2_n_seeds < 1L) {
  nmk_v3_stage2_n_seeds <- 1L
}
nmk_ensemble_top_k <- as.integer(parse_num_env_nmk(
  "NMK_ENSEMBLE_TOP_K", if (isTRUE(nmk_autotune_v4)) 5L else 1L))
if (!is.finite(nmk_ensemble_top_k) || nmk_ensemble_top_k < 1L) {
  nmk_ensemble_top_k <- 1L
}
nmk_ensemble_weighting <- tolower(Sys.getenv(
  "NMK_ENSEMBLE_WEIGHTING", unset = "softmax"))
if (!nmk_ensemble_weighting %in% c("softmax", "inverse", "uniform")) {
  nmk_ensemble_weighting <- "softmax"
}
nmk_ensemble_temperature <- parse_num_env_nmk("NMK_ENSEMBLE_TEMPERATURE", 0.75)
if (!is.finite(nmk_ensemble_temperature) || nmk_ensemble_temperature <= 0) {
  nmk_ensemble_temperature <- 0.75
}
nmk_v3_rmse_sd_weight <- parse_num_env_nmk("NMK_V3_RMSE_SD_WEIGHT", 0.25)
nmk_v3_abs_me_weight <- parse_num_env_nmk("NMK_V3_ABS_ME_WEIGHT", 0.10)
nmk_v3_log_margin <- parse_num_env_nmk("NMK_V3_LOG_MARGIN", 0.50)
nmk_v3_affine_slope_min <- parse_num_env_nmk("NMK_V3_AFFINE_SLOPE_MIN", 0.50)
nmk_v3_affine_slope_max <- parse_num_env_nmk("NMK_V3_AFFINE_SLOPE_MAX", 1.50)
nmk_v4_quantile_bins <- as.integer(parse_num_env_nmk("NMK_V4_QUANTILE_BINS", 5L))
if (!is.finite(nmk_v4_quantile_bins) || nmk_v4_quantile_bins < 2L) {
  nmk_v4_quantile_bins <- 5L
}
nmk_v4_quantile_excess_weight <- parse_num_env_nmk(
  "NMK_V4_QUANTILE_EXCESS_WEIGHT", if (isTRUE(nmk_autotune_v4)) 0.15 else 0)
if (!is.finite(nmk_v4_quantile_excess_weight) ||
    nmk_v4_quantile_excess_weight < 0) {
  nmk_v4_quantile_excess_weight <- 0
}
nmk_v4_quantile_sd_weight <- parse_num_env_nmk(
  "NMK_V4_QUANTILE_SD_WEIGHT", if (isTRUE(nmk_autotune_v4)) 0.05 else 0)
if (!is.finite(nmk_v4_quantile_sd_weight) || nmk_v4_quantile_sd_weight < 0) {
  nmk_v4_quantile_sd_weight <- 0
}
nmk_selection_objective <- tolower(trimws(Sys.getenv(
  "NMK_SELECTION_OBJECTIVE", unset = "paired_rf_delta")))
if (!(nmk_selection_objective %in% c("rmse", "paired_rf_delta"))) {
  nmk_selection_objective <- "rmse"
}
nmk_noharm_policy <- parse_bool_env_nmk(
  "NMK_NOHARM_POLICY",
  identical(nmk_selection_objective, "paired_rf_delta"))
nmk_noharm_margin_abs <- parse_num_env_nmk("NMK_NOHARM_MARGIN_ABS", 0.05)
if (!is.finite(nmk_noharm_margin_abs) || nmk_noharm_margin_abs < 0) {
  nmk_noharm_margin_abs <- 0
}
nmk_noharm_margin_rel <- parse_num_env_nmk("NMK_NOHARM_MARGIN_REL", 0.0)
if (!is.finite(nmk_noharm_margin_rel) || nmk_noharm_margin_rel < 0) {
  nmk_noharm_margin_rel <- 0
}
nmk_noharm_restrict_ensemble <- parse_bool_env_nmk(
  "NMK_NOHARM_RESTRICT_ENSEMBLE", TRUE)
nmk_rf_noharm_policy <- parse_bool_env_nmk(
  "NMK_RF_NOHARM_POLICY",
  identical(nmk_selection_objective, "paired_rf_delta"))
nmk_rf_noharm_margin_abs <- parse_num_env_nmk("NMK_RF_NOHARM_MARGIN_ABS", 0.05)
if (!is.finite(nmk_rf_noharm_margin_abs) || nmk_rf_noharm_margin_abs < 0) {
  nmk_rf_noharm_margin_abs <- 0
}
nmk_rf_noharm_margin_rel <- parse_num_env_nmk("NMK_RF_NOHARM_MARGIN_REL", 0.0)
if (!is.finite(nmk_rf_noharm_margin_rel) || nmk_rf_noharm_margin_rel < 0) {
  nmk_rf_noharm_margin_rel <- 0
}
nmk_rf_noharm_restrict_ensemble <- parse_bool_env_nmk(
  "NMK_RF_NOHARM_RESTRICT_ENSEMBLE", TRUE)
nmk_tune_target_transforms <- tolower(parse_list_env_nmk(
  "NMK_TUNE_TARGET_TRANSFORMS", c("auto", "identity", "log")))
nmk_tune_target_transforms <- unique(nmk_tune_target_transforms[
  nmk_tune_target_transforms %in% c("auto", "identity", "log")])
if (length(nmk_tune_target_transforms) == 0L) {
  nmk_tune_target_transforms <- c("auto", "identity")
}

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# Corr method used by wadoux_eval (Pearson² matches the paper text).
options(wadoux_corr_method = "pearson")

# -----------------------------------------------------------------------------
# Load GeoVersa environment (for encoders, tensor caches, patch cache and
# prepare_pointpatch_fold) and then layer NMK on top of it.
# -----------------------------------------------------------------------------

load_nmk_env <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("torch R package not available — install torch before running NMK.")
  }

  loader_parent <- new.env(parent = globalenv())
  env <- new.env(parent = loader_parent)

  loader_parent$source <- function(file, ...) {
    is_absolute <- grepl("^(?:[A-Za-z]:[\\\\/]|/)", file)
    resolved <- if (is_absolute) {
      file
    } else {
      file.path(dirname(code_root_wadoux), sub("^\\./", "", file))
    }
    sys.source(resolved, envir = env)
    invisible(NULL)
  }

  # Layer order matters: ConvKrigingNet2D.R sources KrigingNet_PointPatchCNN.R
  # which in turn sources KrigingNet_WadouxComparison.R. That defines make_mlp,
  # ScalarHead, huber_loss, transform_target, the tensor caches and the fold
  # preparation used by NMK.
  sys.source(file.path(code_root_wadoux, "ConvKrigingNet2D.R"),
             envir = env)
  # ConvKrigingNet2D_Auto.R provides fit_variogram_auto().
  sys.source(file.path(code_root_wadoux, "ConvKrigingNet2D_Auto.R"),
             envir = env)
  # Finally layer NMK. It depends on the helpers above.
  sys.source(file.path(code_root_wadoux, "NeuroMaternKrige.R"),
             envir = env)

  env
}

nmk_env <- load_nmk_env()

# -----------------------------------------------------------------------------
# Per-profile NMK parameter dictionary
# -----------------------------------------------------------------------------

get_nmk_params <- function(profile = c("default", "quick"),
                           train_seed = 123L,
                           device = "cpu") {
  profile <- match.arg(profile)
  base <- if (profile == "quick") {
    nmk_env$neuromatern_quick_params
  } else {
    nmk_env$neuromatern_default_params
  }
  modifyList(
    base,
    list(
      train_seed = train_seed,
      device = device,
      deterministic_batches = TRUE
    )
  )
}

nmk_params <- get_nmk_params(profile = profile,
                             train_seed = train_seed,
                             device = device)

# Environment-variable overrides for NMK hyperparameters.
.apply_param_override_num <- function(params, env_name, field, as_int = FALSE) {
  raw <- Sys.getenv(env_name, unset = "")
  if (!nzchar(raw)) return(params)
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) return(params)
  params[[field]] <- if (as_int) as.integer(val) else val
  cat(sprintf("[NMK runner] override %s -> %s\n", field,
              format(params[[field]])))
  params
}

.apply_param_override_bool <- function(params, env_name, field) {
  raw <- Sys.getenv(env_name, unset = "")
  if (!nzchar(raw)) return(params)
  val <- toupper(trimws(raw)) %in% c("TRUE", "1", "YES", "Y")
  params[[field]] <- val
  cat(sprintf("[NMK runner] override %s -> %s\n", field, val))
  params
}

nmk_params <- .apply_param_override_num(nmk_params, "NMK_EPOCHS",         "epochs",           as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_PATIENCE",       "patience",         as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_LR",             "lr")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_BATCH_SIZE",     "batch_size",       as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_WEIGHT_DECAY",   "wd")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_LAMBDA_SMOOTH",  "lambda_smooth")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_LAMBDA_ANCHOR",  "lambda_anchor")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_BETA_INIT",      "beta_init")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_KRIGING_HIDDEN", "kriging_hidden",   as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_INIT_MIX_MEAN",  "init_mix_mean")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_K_NEIGHBORS",    "K_neighbors",      as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_BASE_LOSS_WEIGHT", "base_loss_weight")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_LAMBDA_COV",        "lambda_cov")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_KRIG_DROP_P_TRAIN", "krig_drop_p_train")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_DIST_SCALE",        "dist_scale_eval")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_N_SEEDS",           "n_seeds",  as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TAB_INPUT_DROPOUT", "tab_input_dropout")
nmk_params <- .apply_param_override_bool(nmk_params, "NMK_TAB_QUANTILE_BASIS", "tab_quantile_basis")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TAB_QBASIS_KNOTS", "tab_qbasis_knots", as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TAB_QBASIS_DROPOUT", "tab_qbasis_dropout")
nmk_params <- .apply_param_override_bool(nmk_params, "NMK_TAB_CROSSNET", "tab_crossnet")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TAB_CROSS_LAYERS", "tab_cross_layers", as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TAB_CROSS_DROPOUT", "tab_cross_dropout")
nmk_params <- .apply_param_override_bool(nmk_params, "NMK_TAB_SKIP", "tab_skip")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TAB_SKIP_HIDDEN", "tab_skip_hidden", as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TAB_SKIP_DROPOUT", "tab_skip_dropout")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TAB_SKIP_INIT", "tab_skip_init")
nmk_params <- .apply_param_override_num(nmk_params, "NMK_N_TREES", "n_trees", as_int = TRUE)
nmk_params <- .apply_param_override_num(nmk_params, "NMK_TREE_DEPTH", "tree_depth", as_int = TRUE)

# Target transform override: "auto" (default), "log", or "identity".
.apply_target_transform_override <- function(params) {
  raw <- Sys.getenv("NMK_TARGET_TRANSFORM", unset = "")
  if (!nzchar(raw)) return(params)
  val <- tolower(trimws(raw))
  if (!val %in% c("auto", "identity", "log")) {
    cat(sprintf("[NMK runner] NMK_TARGET_TRANSFORM = '%s' invalid (auto|identity|log) — ignored\n", raw))
    return(params)
  }
  params$target_transform <- val
  cat(sprintf("[NMK runner] override target_transform -> %s\n", val))
  params
}
nmk_params <- .apply_target_transform_override(nmk_params)

# -----------------------------------------------------------------------------
# Patch size (follows GeoVersa auto rule: min(max(8, ⌊√n⌋), 31), overridable).
# -----------------------------------------------------------------------------

patch_size <- {
  override <- Sys.getenv("NMK_PATCH_SIZE", unset = "")
  if (nzchar(override)) {
    as.integer(override)
  } else {
    as.integer(min(max(8L, floor(sqrt(sample_size))), 31L))
  }
}

# -----------------------------------------------------------------------------
# Wadoux data
# -----------------------------------------------------------------------------

cat(sprintf(
  "[NMK runner] scenario=%s  protocols=%s  models=%s  n_iter=%d  n=%d  patch=%d  device=%s  tune=%s  objective=%s  noharm=%s  rf_noharm=%s  autotune_v2=%s  autotune_v3=%s  autotune_v4=%s  backbone=%s  full_grid=%s  inner_folds=%d  workers=%d  refit_full=%s  refit_warm=%s  topk=%d  variograms=%s  audit=%s\n",
  scenario, paste(protocols, collapse = ","),
  paste(models, collapse = ","),
  n_iter, sample_size, patch_size, device,
  nmk_tune, nmk_selection_objective, nmk_noharm_policy,
  nmk_rf_noharm_policy,
  nmk_autotune_v2, nmk_autotune_v3, nmk_autotune_v4,
  nmk_autotune_backbone,
  nmk_tune_full_grid, nmk_tune_inner_folds,
  nmk_num_workers, nmk_refit_full_calibration, nmk_refit_warm_start,
  nmk_ensemble_top_k, nmk_save_variograms, nmk_save_prediction_audit))

need_polygon <- scenario %in% c("regular_grid", "clustered_random")
common_data <- load_wadoux_common_data(include_polygon = need_polygon)

predList <- predList_modelfull_wadoux
form_rf  <- as.formula(
  paste(response_name_wadoux, "~", paste(predList, collapse = "+"))
)

# -----------------------------------------------------------------------------
# Helpers: sampling
# -----------------------------------------------------------------------------

sample_calibration_nmk <- function() {
  if (scenario == "random") {
    out <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  } else if (scenario == "regular_grid") {
    out <- sample_valuetable_regular_grid_wadoux(
      common_data = common_data, sample_size = sample_size)
  } else if (scenario == "clustered_random") {
    out <- sample_valuetable_clustered_random_wadoux(
      common_data   = common_data,
      kmeans_centers = as.integer(parse_num_env_nmk("NMK_KMEANS_CENTERS", 100L)),
      n_psu          = as.integer(parse_num_env_nmk("NMK_N_PSU",          20L)),
      m_per_psu      = as.integer(parse_num_env_nmk("NMK_M_PER_PSU",      25L))
    )
  } else {
    stop("Unsupported NMK_SCENARIO: ", scenario)
  }
  out$..patch_cache_row_id <- seq_len(nrow(out))
  out
}

# -----------------------------------------------------------------------------
# Helpers: inner train/val split + fold-data preparation for NMK
# -----------------------------------------------------------------------------

make_inner_val_split_nmk <- function(train_df, val_frac = 0.2, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n_train <- nrow(train_df)
  if (n_train < 10) stop("Outer training set too small for NMK fair split.")
  val_size <- max(1L, floor(n_train * val_frac))
  val_idx <- sample.int(n_train, size = val_size, replace = FALSE)
  train_idx <- setdiff(seq_len(n_train), val_idx)
  list(
    train = train_df[train_idx, , drop = FALSE],
    val   = train_df[val_idx,   , drop = FALSE]
  )
}

nmk_target_transform_auto_runner <- function(y_train) {
  fn <- nmk_env$.nmk_target_transform_auto
  if (is.function(fn)) return(fn(y_train))
  y_clean <- y_train[is.finite(y_train)]
  if (length(y_clean) < 3L || any(y_clean <= 0)) return("identity")
  sw <- tryCatch(stats::shapiro.test(
    sample(y_clean, min(length(y_clean), 5000L)))$p.value,
    error = function(e) 1)
  if (is.finite(sw) && sw < 0.05) "log" else "identity"
}

make_regression_folds_nmk <- function(y, k, seed = NULL) {
  y <- as.numeric(y)
  n <- length(y)
  k <- as.integer(max(2L, min(k, floor(n / 2L))))
  if (!is.null(seed)) set.seed(as.integer(seed))
  fold_id <- rep(NA_integer_, n)
  ok <- is.finite(y)
  if (sum(ok) >= k) {
    qs <- unique(as.numeric(stats::quantile(
      y[ok], probs = seq(0, 1, length.out = min(6L, sum(ok)) + 1L),
      na.rm = TRUE, type = 7L)))
    bins <- if (length(qs) >= 3L) {
      cut(y[ok], breaks = qs, include.lowest = TRUE, labels = FALSE)
    } else {
      rep(1L, sum(ok))
    }
    ok_idx <- which(ok)
    for (b in sort(unique(bins))) {
      idx <- sample(ok_idx[bins == b])
      fold_id[idx] <- rep(seq_len(k), length.out = length(idx))
    }
  }
  missing <- which(!is.finite(fold_id))
  if (length(missing) > 0L) {
    fold_id[missing] <- rep(seq_len(k), length.out = length(missing))
  }
  empty <- setdiff(seq_len(k), unique(fold_id))
  if (length(empty) > 0L) {
    idx <- sample(seq_len(n), length(empty))
    fold_id[idx] <- empty
  }
  fold_id
}

bind_patch_arrays_4d_nmk <- function(arrays) {
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

prepare_nmk_fold <- function(train_df, val_df, test_df,
                             calibration_patch_cache,
                             context,
                             patch_size,
                             neighbor_pool_k = NULL) {
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
      external_cache <- nmk_env$build_pointpatch_patch_cache(
        context        = context,
        calibration_df = test_df,
        patch_size     = patch_size,
        label          = "test patches"
      )
      parts[[3]] <- external_cache$patches_raw
    }
    combined_cache <- list(
      patches_raw = bind_patch_arrays_4d_nmk(parts),
      patch_size  = patch_size,
      n           = nrow(combined_df)
    )
  }

  neighbor_pool_k <- if (is.null(neighbor_pool_k)) 30L else as.integer(neighbor_pool_k)
  if (!is.finite(neighbor_pool_k) || neighbor_pool_k < 1L) {
    neighbor_pool_k <- 30L
  }

  nmk_env$prepare_pointpatch_fold(
    context        = context,
    calibration_df = combined_df,
    train_idx      = seq_len(n_train),
    val_idx        = n_train + seq_len(n_val),
    test_idx       = n_train + n_val + seq_len(n_test),
    patch_size     = patch_size,
    K              = neighbor_pool_k,
    patch_cache    = combined_cache
  )
}

run_nmk_on_split <- function(train_core, val_df, test_df,
	                             calibration_patch_cache,
                             params = nmk_params) {
  fd <- prepare_nmk_fold(
    train_df = train_core, val_df = val_df, test_df = test_df,
    calibration_patch_cache = calibration_patch_cache,
    context = nmk_env$wadoux_context,
    patch_size = patch_size,
    neighbor_pool_k = if (!is.null(params$K_neighbors)) {
      max(30L, as.integer(params$K_neighbors))
    } else {
      30L
    }
  )
  # NMK runs as a primary predictor. The ensemble wrapper degrades to the
  # single-seed trainer when n_seeds = 1.
  do.call(
    nmk_env$train_neuromaternkrige_ensemble,
    c(list(fd = fd), params)
  )
}

nmk_refit_monitor_df <- function(train_df) {
  if (nrow(train_df) == 0L) stop("NMK refit monitor requires training rows.")
  train_df[1L, , drop = FALSE]
}

nmk_refit_epochs_from_diag <- function(diag, default_epochs) {
  vals <- numeric()
  if (!is.null(diag$best_epoch)) {
    vals <- c(vals, suppressWarnings(as.numeric(diag$best_epoch)))
  }
  if (!is.null(diag$per_seed) && is.list(diag$per_seed)) {
    vals <- c(vals, vapply(diag$per_seed, function(d) {
      suppressWarnings(as.numeric(d$best_epoch))
    }, numeric(1)))
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

nmk_refit_params_from_selection <- function(selection_diag, base_params = nmk_params) {
  params <- base_params
  refit_epochs <- nmk_refit_epochs_from_diag(
    selection_diag,
    default_epochs = if (is.null(base_params$epochs)) 80L else base_params$epochs
  )
  if (isTRUE(nmk_refit_warm_start)) {
    refit_epochs <- as.integer(round(refit_epochs * nmk_refit_warm_start_epoch_mult))
    refit_epochs <- as.integer(max(
      nmk_refit_warm_start_min_epochs,
      min(nmk_refit_warm_start_max_epochs, refit_epochs)
    ))
    params$warmup_epochs <- 0L
  }
  params$epochs <- refit_epochs
  params$patience <- refit_epochs + 1L
  params$lr_patience <- refit_epochs + 1L
  params$refit_fixed_epochs <- TRUE
  raw_refit_n_seeds <- Sys.getenv("NMK_REFIT_N_SEEDS", unset = "")
  if (nzchar(raw_refit_n_seeds)) {
    refit_n_seeds <- suppressWarnings(as.integer(raw_refit_n_seeds))
    if (is.finite(refit_n_seeds) && refit_n_seeds > 0L) {
      params$n_seeds <- refit_n_seeds
    }
  } else if (!is.null(nmk_params$n_seeds)) {
    params$n_seeds <- nmk_params$n_seeds
  }
  params
}

nmk_clip_k <- function(k, n) {
  as.integer(max(2L, min(as.integer(k), max(2L, n - 1L))))
}

nmk_add_backbone_defaults <- function(df) {
  n <- nrow(df)
  add <- function(name, value) {
    if (!name %in% names(df)) df[[name]] <<- rep(value, n)
  }
  add("tab_quantile_basis", FALSE)
  add("tab_qbasis_knots", NA_integer_)
  add("tab_qbasis_dropout", NA_real_)
  add("tab_crossnet", FALSE)
  add("tab_cross_layers", NA_integer_)
  add("tab_cross_dropout", NA_real_)
  add("tab_input_dropout", NA_real_)
  add("tab_skip", FALSE)
  add("tab_skip_hidden", NA_integer_)
  add("tab_skip_dropout", NA_real_)
  add("tab_skip_init", NA_real_)
  add("n_trees", NA_integer_)
  add("tree_depth", NA_integer_)
  add("fusion_hidden", NA_integer_)
  add("patch_dim", NA_integer_)
  df
}

nmk_candidate_profiles <- function(train_df) {
  n <- nrow(train_df)
  sqrt_n <- sqrt(max(1L, n))
  k_low <- nmk_clip_k(min(40L, max(8L, round(1.0 * sqrt_n))), n)
  k_mid <- nmk_clip_k(min(48L, max(12L, round(1.5 * sqrt_n))), n)
  k_high <- nmk_clip_k(min(64L, max(16L, round(2.0 * sqrt_n))), n)
  base <- data.frame(
    profile = c("auto_current", "compact_spatial", "balanced_spatial",
                "wide_spatial", "regularized", "no_krig_dropout",
                "low_mix"),
    K_neighbors = c(NA_integer_, k_low, k_mid, k_high, k_mid, k_mid, k_mid),
    kriging_hidden = c(NA_integer_, 32L, 64L, 96L, 64L, 64L, 64L),
    krig_drop_p_train = c(NA_real_, 0.10, 0.10, 0.05, 0.25, 0.00, 0.10),
    init_mix_mean = c(NA_real_, 0.20, 0.35, 0.45, 0.35, 0.35, 0.05),
    tab_dropout = c(NA_real_, 0.05, 0.08, 0.08, 0.15, 0.05, 0.08),
    patch_dropout = c(NA_real_, 0.05, 0.08, 0.08, 0.10, 0.05, 0.08),
    stringsAsFactors = FALSE
  )
  base$lr <- NA_real_
  base$wd <- NA_real_
  base$batch_size <- NA_integer_
  base$lambda_cov <- NA_real_
  base$lambda_smooth <- NA_real_
  base$lambda_anchor <- NA_real_
  base$base_loss_weight <- NA_real_
  base$beta_init <- NA_real_
  base$dist_scale_eval <- NA_real_
  base <- nmk_add_backbone_defaults(base)

  if (!isTRUE(nmk_autotune_v2)) {
    return(base)
  }

  extra <- data.frame(
    profile = c("short_range_strong", "long_range_smooth",
                "high_regularization", "low_regularization",
                "tabular_stable", "kriging_aggressive",
                "bias_cov_control"),
    K_neighbors = c(k_low, k_high, k_mid, k_mid, k_low, k_high, k_mid),
    kriging_hidden = c(64L, 96L, 64L, 64L, 32L, 96L, 64L),
    krig_drop_p_train = c(0.05, 0.05, 0.25, 0.05, 0.20, 0.00, 0.15),
    init_mix_mean = c(0.20, 0.45, 0.35, 0.35, 0.35, 0.20, 0.35),
    tab_dropout = c(0.05, 0.08, 0.18, 0.03, 0.02, 0.05, 0.10),
    patch_dropout = c(0.08, 0.08, 0.12, 0.03, 0.20, 0.05, 0.10),
    lr = c(NA_real_, NA_real_, 7.5e-4, 1.5e-3, NA_real_, 1.0e-3, NA_real_),
    wd = c(1e-5, 1e-5, 1e-4, 1e-6, 5e-5, 1e-5, 5e-5),
    batch_size = c(NA_integer_, NA_integer_, NA_integer_, NA_integer_,
                   NA_integer_, NA_integer_, NA_integer_),
    lambda_cov = c(NA_real_, NA_real_, 0.040, 0.010, 0.020, 0.015, 0.055),
    lambda_smooth = c(0.120, 0.180, 0.360, 0.080, 0.180, 0.080, 0.240),
    lambda_anchor = c(0.120, 0.180, 0.360, 0.080, 0.180, 0.080, 0.240),
    base_loss_weight = c(NA_real_, NA_real_, NA_real_, NA_real_,
                         1.000, 0.250, 0.750),
    beta_init = c(NA_real_, NA_real_, -2.000, -1.000, -3.000, -0.500, -2.000),
    dist_scale_eval = c(0.450, 0.950, NA_real_, NA_real_, 0.600, 0.900, 0.700),
    stringsAsFactors = FALSE
  )
  extra <- nmk_add_backbone_defaults(extra)

  if (!isTRUE(nmk_autotune_backbone)) {
    return(rbind(base, extra))
  }

  p <- length(intersect(predList_modelfull_wadoux, names(train_df)))
  if (!is.finite(p) || p < 1L) {
    p <- max(1L, ncol(train_df) - length(c(response_name_wadoux, "x", "y")))
  }
  skip_hidden <- as.integer(max(64L, min(256L, round(6L * p))))
  backbone <- data.frame(
    profile = c("tab_skip_lowvar", "tab_qbasis_trend",
                "tab_qbasis_cross", "tab_qbasis_forest_deep",
                "tab_cross_forest", "tab_qbasis_patch_lean",
                "tab_qbasis_skip_regularized"),
    K_neighbors = c(k_mid, k_low, k_mid, k_mid, k_high, k_low, k_mid),
    kriging_hidden = c(32L, 32L, 64L, 64L, 96L, 32L, 64L),
    krig_drop_p_train = c(0.25, 0.20, 0.15, 0.10, 0.10, 0.25, 0.20),
    init_mix_mean = c(0.25, 0.25, 0.35, 0.35, 0.45, 0.25, 0.35),
    tab_dropout = c(0.03, 0.04, 0.05, 0.05, 0.06, 0.03, 0.10),
    patch_dropout = c(0.15, 0.12, 0.10, 0.08, 0.08, 0.25, 0.12),
    lr = c(NA_real_, NA_real_, NA_real_, 7.5e-4, NA_real_, NA_real_, 7.5e-4),
    wd = c(5e-5, 5e-5, 5e-5, 1e-4, 5e-5, 5e-5, 1e-4),
    batch_size = c(NA_integer_, NA_integer_, NA_integer_, NA_integer_,
                   NA_integer_, NA_integer_, NA_integer_),
    lambda_cov = c(0.040, 0.035, 0.035, 0.040, 0.030, 0.040, 0.050),
    lambda_smooth = c(0.240, 0.180, 0.180, 0.240, 0.160, 0.220, 0.300),
    lambda_anchor = c(0.240, 0.180, 0.180, 0.240, 0.160, 0.220, 0.300),
    base_loss_weight = c(1.000, 1.000, 0.900, 0.900, 0.750, 1.000, 1.000),
    beta_init = c(-2.500, -2.500, -2.000, -2.000, -1.500, -3.000, -2.500),
    dist_scale_eval = c(0.550, 0.600, 0.650, 0.700, 0.800, 0.550, 0.600),
    tab_quantile_basis = c(FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE),
    tab_qbasis_knots = c(NA_integer_, 4L, 4L, 5L, NA_integer_, 5L, 6L),
    tab_qbasis_dropout = c(NA_real_, 0.03, 0.03, 0.05, NA_real_, 0.03, 0.08),
    tab_crossnet = c(FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE),
    tab_cross_layers = c(NA_integer_, NA_integer_, 2L, NA_integer_, 2L,
                         NA_integer_, 1L),
    tab_cross_dropout = c(NA_real_, NA_real_, 0.03, NA_real_, 0.04,
                          NA_real_, 0.08),
    tab_input_dropout = c(0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.05),
    tab_skip = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE),
    tab_skip_hidden = rep(skip_hidden, 7L),
    tab_skip_dropout = c(0.02, 0.02, 0.02, 0.03, 0.03, 0.02, 0.05),
    tab_skip_init = c(-1.00, -1.25, -1.25, -1.50, -1.50, -1.00, -1.75),
    n_trees = c(6L, 8L, 8L, 12L, 12L, 8L, 10L),
    tree_depth = c(3L, 4L, 4L, 4L, 4L, 3L, 4L),
    fusion_hidden = rep(NA_integer_, 7L),
    patch_dim = rep(NA_integer_, 7L),
    stringsAsFactors = FALSE
  )
  backbone <- nmk_add_backbone_defaults(backbone)
  if (!isTRUE(nmk_autotune_v4)) {
    return(rbind(base, extra, backbone))
  }

  skip_hidden_v4 <- as.integer(max(192L, min(512L, round(10L * p))))
  skip_hidden_v4_big <- as.integer(max(256L, min(768L, round(14L * p))))
  capacity <- data.frame(
    profile = c("rf_capacity_forest", "rf_capacity_qbasis",
                "rf_capacity_qbasis_cross", "rf_capacity_deep_forest",
                "rf_capacity_skip_only", "rf_capacity_spatial_resid",
                "rf_capacity_regularized", "rf_capacity_patch_suppressed"),
    K_neighbors = c(k_mid, k_low, k_mid, k_mid, k_low, k_high, k_mid, k_low),
    kriging_hidden = c(96L, 64L, 96L, 128L, 64L, 128L, 96L, 64L),
    krig_drop_p_train = c(0.15, 0.20, 0.15, 0.10, 0.25, 0.05, 0.20, 0.30),
    init_mix_mean = c(0.25, 0.20, 0.30, 0.35, 0.15, 0.45, 0.25, 0.10),
    tab_dropout = c(0.05, 0.04, 0.05, 0.08, 0.03, 0.06, 0.10, 0.03),
    patch_dropout = c(0.18, 0.22, 0.18, 0.12, 0.25, 0.08, 0.18, 0.35),
    lr = c(5.0e-4, 5.0e-4, 5.0e-4, 4.0e-4,
           7.5e-4, 5.0e-4, 4.0e-4, 7.5e-4),
    wd = c(2e-4, 1e-4, 2e-4, 5e-4, 1e-4, 2e-4, 5e-4, 1e-4),
    batch_size = c(48L, 48L, 48L, 40L, 56L, 48L, 40L, 56L),
    lambda_cov = c(0.040, 0.035, 0.040, 0.050, 0.035, 0.030, 0.060, 0.035),
    lambda_smooth = c(0.220, 0.200, 0.220, 0.300, 0.180, 0.160, 0.350, 0.180),
    lambda_anchor = c(0.220, 0.200, 0.220, 0.300, 0.180, 0.160, 0.350, 0.180),
    base_loss_weight = c(1.000, 1.000, 0.950, 0.900, 1.000, 0.750, 0.950, 1.000),
    beta_init = c(-2.500, -3.000, -2.500, -2.000,
                  -3.500, -1.500, -2.500, -3.500),
    dist_scale_eval = c(0.600, 0.550, 0.650, 0.700, 0.500, 0.850, 0.600, 0.500),
    tab_quantile_basis = c(TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE),
    tab_qbasis_knots = c(8L, 10L, 8L, 10L, NA_integer_, 8L, 10L, 10L),
    tab_qbasis_dropout = c(0.04, 0.04, 0.04, 0.06, NA_real_, 0.04, 0.08, 0.04),
    tab_crossnet = c(FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE, FALSE),
    tab_cross_layers = c(NA_integer_, NA_integer_, 2L, 2L,
                         NA_integer_, NA_integer_, 2L, NA_integer_),
    tab_cross_dropout = c(NA_real_, NA_real_, 0.04, 0.06,
                          NA_real_, NA_real_, 0.08, NA_real_),
    tab_input_dropout = c(0.03, 0.02, 0.03, 0.04, 0.02, 0.03, 0.05, 0.02),
    tab_skip = rep(TRUE, 8L),
    tab_skip_hidden = c(skip_hidden_v4, skip_hidden_v4_big, skip_hidden_v4_big,
                        skip_hidden_v4_big, skip_hidden_v4_big, skip_hidden_v4,
                        skip_hidden_v4_big, skip_hidden_v4_big),
    tab_skip_dropout = c(0.03, 0.02, 0.03, 0.04, 0.02, 0.03, 0.05, 0.02),
    tab_skip_init = c(-1.25, -1.00, -1.25, -1.50, -0.75, -1.50, -1.75, -0.75),
    n_trees = c(24L, 32L, 32L, 48L, 32L, 24L, 48L, 32L),
    tree_depth = c(5L, 5L, 5L, 5L, 6L, 5L, 5L, 6L),
    fusion_hidden = c(384L, 384L, 512L, 512L, 384L, 384L, 512L, 384L),
    patch_dim = c(96L, 64L, 96L, 128L, 64L, 128L, 96L, 48L),
    stringsAsFactors = FALSE
  )
  capacity <- nmk_add_backbone_defaults(capacity)
  rbind(base, extra, backbone, capacity)
}

nmk_tune_grid <- function(train_df) {
  transforms_requested <- nmk_tune_target_transforms
  y <- train_df[[response_name_wadoux]]
  if (!all(is.finite(y) & y > 0)) {
    transforms_requested <- setdiff(transforms_requested, "log")
  }
  if (length(transforms_requested) == 0L) transforms_requested <- "identity"
  auto_transform <- nmk_target_transform_auto_runner(y)
  profiles <- nmk_candidate_profiles(train_df)
  rows <- list()
  for (tf_req in transforms_requested) {
    tf_eff <- if (identical(tf_req, "auto")) auto_transform else tf_req
    if (identical(tf_eff, "log") && !all(is.finite(y) & y > 0)) next
    for (i in seq_len(nrow(profiles))) {
      p <- profiles[i, , drop = FALSE]
      rows[[length(rows) + 1L]] <- data.frame(
        candidate_id = length(rows) + 1L,
        candidate_label = paste(tf_eff, p$profile, sep = "__"),
        target_transform_requested = tf_req,
        target_transform_effective = tf_eff,
        target_transform = tf_eff,
        profile = p$profile,
        is_auto_base = identical(tf_req, "auto") &&
          identical(p$profile, "auto_current"),
        K_neighbors = p$K_neighbors,
        kriging_hidden = p$kriging_hidden,
        krig_drop_p_train = p$krig_drop_p_train,
        init_mix_mean = p$init_mix_mean,
        tab_dropout = p$tab_dropout,
        patch_dropout = p$patch_dropout,
        lr = p$lr,
        wd = p$wd,
        batch_size = p$batch_size,
        lambda_cov = p$lambda_cov,
        lambda_smooth = p$lambda_smooth,
        lambda_anchor = p$lambda_anchor,
        base_loss_weight = p$base_loss_weight,
        beta_init = p$beta_init,
        dist_scale_eval = p$dist_scale_eval,
        tab_quantile_basis = p$tab_quantile_basis,
        tab_qbasis_knots = p$tab_qbasis_knots,
        tab_qbasis_dropout = p$tab_qbasis_dropout,
        tab_crossnet = p$tab_crossnet,
        tab_cross_layers = p$tab_cross_layers,
        tab_cross_dropout = p$tab_cross_dropout,
        tab_input_dropout = p$tab_input_dropout,
        tab_skip = p$tab_skip,
        tab_skip_hidden = p$tab_skip_hidden,
        tab_skip_dropout = p$tab_skip_dropout,
        tab_skip_init = p$tab_skip_init,
        n_trees = p$n_trees,
        tree_depth = p$tree_depth,
        fusion_hidden = p$fusion_hidden,
        patch_dim = p$patch_dim,
        stringsAsFactors = FALSE
      )
    }
  }
  grid <- do.call(rbind, rows)
  key_cols <- c("target_transform_effective", "K_neighbors", "kriging_hidden",
                "krig_drop_p_train", "init_mix_mean", "tab_dropout",
                "patch_dropout", "lr", "wd", "batch_size", "lambda_cov",
                "lambda_smooth", "lambda_anchor", "base_loss_weight",
                "beta_init", "dist_scale_eval", "tab_quantile_basis",
                "tab_qbasis_knots", "tab_qbasis_dropout", "tab_crossnet",
                "tab_cross_layers", "tab_cross_dropout", "tab_input_dropout",
                "tab_skip", "tab_skip_hidden", "tab_skip_dropout",
                "tab_skip_init", "n_trees", "tree_depth", "fusion_hidden",
                "patch_dim")
  keys <- apply(grid[, key_cols, drop = FALSE], 1, paste, collapse = "|")
  grid <- grid[!duplicated(keys), , drop = FALSE]
  if (!isTRUE(nmk_tune_full_grid) &&
      is.finite(nmk_tune_max_candidates) &&
      nrow(grid) > nmk_tune_max_candidates) {
    base_rows <- grid[grid$is_auto_base, , drop = FALSE]
    rest <- grid[!grid$is_auto_base, , drop = FALSE]
    keep_rest <- max(0L, nmk_tune_max_candidates - nrow(base_rows))
    if (nrow(rest) > keep_rest) {
      keep_idx <- unique(as.integer(round(seq(1, nrow(rest),
                                             length.out = keep_rest))))
      rest <- rest[keep_idx, , drop = FALSE]
    }
    grid <- rbind(base_rows, rest)
  }
  grid$candidate_id <- seq_len(nrow(grid))
  grid$candidate_label <- paste(grid$target_transform_effective,
                                grid$profile, sep = "__")
  grid
}

nmk_params_from_candidate <- function(candidate, candidate_seed) {
  params <- nmk_params
  set_num <- function(field) {
    val <- suppressWarnings(as.numeric(candidate[[field]]))
    if (length(val) == 1L && is.finite(val)) params[[field]] <<- val
  }
  set_int <- function(field) {
    val <- suppressWarnings(as.integer(candidate[[field]]))
    if (length(val) == 1L && is.finite(val)) params[[field]] <<- val
  }
  set_bool <- function(field) {
    val <- candidate[[field]]
    if (length(val) == 1L && !is.na(val)) {
      params[[field]] <<- isTRUE(val) ||
        toupper(as.character(val)) %in% c("TRUE", "1", "YES", "Y")
    }
  }
  params$target_transform <- as.character(candidate$target_transform_effective)
  set_int("K_neighbors")
  set_int("kriging_hidden")
  set_num("krig_drop_p_train")
  set_num("init_mix_mean")
  set_num("tab_dropout")
  set_num("patch_dropout")
  set_num("lr")
  set_num("wd")
  set_int("batch_size")
  set_num("lambda_cov")
  set_num("lambda_smooth")
  set_num("lambda_anchor")
  set_num("base_loss_weight")
  set_num("beta_init")
  set_num("dist_scale_eval")
  set_bool("tab_quantile_basis")
  set_int("tab_qbasis_knots")
  set_num("tab_qbasis_dropout")
  set_bool("tab_crossnet")
  set_int("tab_cross_layers")
  set_num("tab_cross_dropout")
  set_num("tab_input_dropout")
  set_bool("tab_skip")
  set_int("tab_skip_hidden")
  set_num("tab_skip_dropout")
  set_num("tab_skip_init")
  set_int("n_trees")
  set_int("tree_depth")
  set_int("fusion_hidden")
  set_int("patch_dim")
  if (is.finite(nmk_tune_n_seeds) && nmk_tune_n_seeds > 0L) {
    params$n_seeds <- as.integer(nmk_tune_n_seeds)
  }
  params$train_seed <- as.integer(candidate_seed)
  params
}

nmk_metric_values <- function(obs, pred) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  ok <- is.finite(obs) & is.finite(pred)
  if (sum(ok) < 1L) {
    return(list(ME = NA_real_, RMSE = Inf, r2 = NA_real_, MEC = NA_real_))
  }
  obs_ok <- obs[ok]
  pred_ok <- pred[ok]
  err <- pred_ok - obs_ok
  me <- mean(err)
  rmse <- sqrt(mean(err^2))
  r2 <- if (length(obs_ok) >= 2L &&
            stats::sd(obs_ok) > 0 &&
            stats::sd(pred_ok) > 0) {
    suppressWarnings(stats::cor(obs_ok, pred_ok)^2)
  } else {
    NA_real_
  }
  denom <- sum((obs_ok - mean(obs_ok))^2)
  mec <- if (is.finite(denom) && denom > 0) {
    1 - sum(err^2) / denom
  } else {
    NA_real_
  }
  list(ME = me, RMSE = rmse, r2 = r2, MEC = mec)
}

nmk_fold_rmse_values <- function(obs, pred, fold_id = NULL) {
  if (is.null(fold_id) || length(fold_id) != length(obs)) return(NA_real_)
  folds <- sort(unique(fold_id[is.finite(fold_id)]))
  vals <- vapply(folds, function(f) {
    idx <- which(fold_id == f)
    nmk_metric_values(obs[idx], pred[idx])$RMSE
  }, numeric(1))
  vals[is.finite(vals)]
}

nmk_quantile_rmse_values <- function(obs, pred, n_bins = 5L) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  ok <- is.finite(obs) & is.finite(pred)
  obs <- obs[ok]
  pred <- pred[ok]
  if (length(obs) < 5L) return(numeric())
  n_bins <- as.integer(max(2L, min(10L, n_bins)))
  breaks <- unique(as.numeric(stats::quantile(
    obs, probs = seq(0, 1, length.out = n_bins + 1L),
    na.rm = TRUE, type = 7L)))
  if (length(breaks) < 3L) return(numeric())
  bins <- cut(obs, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  vals <- vapply(sort(unique(bins)), function(b) {
    idx <- which(bins == b)
    if (length(idx) < 2L) return(NA_real_)
    sqrt(mean((obs[idx] - pred[idx]) ^ 2, na.rm = TRUE))
  }, numeric(1))
  vals[is.finite(vals)]
}

nmk_format_fold_rmse <- function(vals) {
  vals <- as.numeric(vals)
  if (length(vals) == 0L || all(!is.finite(vals))) return("NA")
  paste(ifelse(is.finite(vals), sprintf("%.6f", vals), "NA"), collapse = "|")
}

nmk_empty_rf_inner <- function(status = "not_available", error = NA_character_) {
  list(
    pred = NULL,
    metrics = list(ME = NA_real_, RMSE = NA_real_, r2 = NA_real_, MEC = NA_real_),
    fold_RMSE = NA_real_,
    status = status,
    error = error
  )
}

# NOTE (do-no-harm alignment): the inner RF below is used ONLY for candidate
# SELECTION inside tuning. It is intentionally NOT relied upon as the final
# do-no-harm certificate, because it is trained on train_core / tune_pool and
# is therefore NOT bit-identical to the scored RF_500 = fit_rf_default_wadoux(
# calibration_full). The binding do-no-harm guarantee is enforced AFTER refit
# by nmk_select_rf_shrinkage_lambda(), whose RF leg reuses the literal scored
# RF. fit_rf_default_wadoux uses a fixed seed (getOption('wadoux_rf_seed',123))
# so both the inner and the final RF are deterministic given identical data.
nmk_inner_rf_baseline <- function(train_core, val_df, tune_pool = NULL,
                                  fold_ids = NULL, use_cv = FALSE) {
  out <- tryCatch({
    if (isTRUE(use_cv) && !is.null(tune_pool) && !is.null(fold_ids) &&
        length(fold_ids) == nrow(tune_pool)) {
      preds <- rep(NA_real_, nrow(tune_pool))
      fold_rmse <- numeric()
      for (fold in sort(unique(fold_ids))) {
        val_idx <- which(fold_ids == fold)
        train_idx <- which(fold_ids != fold)
        rf <- fit_rf_default_wadoux(tune_pool[train_idx, , drop = FALSE], form_rf)
        preds[val_idx] <- predict_rf_default_wadoux(
          rf, tune_pool[val_idx, , drop = FALSE])
        fold_rmse <- c(
          fold_rmse,
          nmk_metric_values(
            tune_pool[[response_name_wadoux]][val_idx],
            preds[val_idx])$RMSE)
      }
      list(
        pred = preds,
        metrics = nmk_metric_values(tune_pool[[response_name_wadoux]], preds),
        fold_RMSE = fold_rmse,
        status = "ok",
        error = NA_character_
      )
    } else {
      rf <- fit_rf_default_wadoux(train_core, form_rf)
      preds <- predict_rf_default_wadoux(rf, val_df)
      list(
        pred = preds,
        metrics = nmk_metric_values(val_df[[response_name_wadoux]], preds),
        fold_RMSE = nmk_metric_values(val_df[[response_name_wadoux]], preds)$RMSE,
        status = "ok",
        error = NA_character_
      )
    }
  }, error = function(e) {
    nmk_empty_rf_inner("error", conditionMessage(e))
  })
  if (is.null(out$metrics)) {
    out$metrics <- nmk_empty_rf_inner()$metrics
  }
  out
}

nmk_blend_values <- function(pred_krig, pred_base = NULL, alpha = 1) {
  pred_krig <- as.numeric(pred_krig)
  if (is.null(pred_base) || length(pred_base) != length(pred_krig)) {
    pred_base <- pred_krig
  }
  pred_base <- as.numeric(pred_base)
  pred_base[!is.finite(pred_base) & is.finite(pred_krig)] <-
    pred_krig[!is.finite(pred_base) & is.finite(pred_krig)]
  pred_krig[!is.finite(pred_krig) & is.finite(pred_base)] <-
    pred_base[!is.finite(pred_krig) & is.finite(pred_base)]
  alpha <- pmax(0, pmin(1, as.numeric(alpha)[1]))
  alpha * pred_krig + (1 - alpha) * pred_base
}

nmk_fit_calibration_policy <- function(obs, pred, policy) {
  policy_req <- tolower(as.character(policy)[1])
  if (!policy_req %in% c("none", "median_bias", "mean_bias", "affine")) {
    policy_req <- "none"
  }
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  ok <- is.finite(obs) & is.finite(pred)
  obs_ok <- obs[ok]
  pred_ok <- pred[ok]
  out <- list(
    calibration_policy_requested = policy_req,
    calibration_policy = policy_req,
    bias = 0,
    intercept = 0,
    slope = 1
  )
  if (policy_req == "none" || length(obs_ok) < 3L) {
    out$calibration_policy <- "none"
    return(out)
  }
  resid <- obs_ok - pred_ok
  if (policy_req == "median_bias") {
    out$bias <- stats::median(resid, na.rm = TRUE)
    return(out)
  }
  if (policy_req == "mean_bias") {
    out$bias <- mean(resid, na.rm = TRUE)
    return(out)
  }
  if (length(obs_ok) < 20L ||
      !is.finite(stats::sd(pred_ok)) ||
      stats::sd(pred_ok) <= 1e-8) {
    out$calibration_policy <- "mean_bias"
    out$bias <- mean(resid, na.rm = TRUE)
    return(out)
  }
  fit <- tryCatch(
    stats::lm(obs_ok ~ pred_ok),
    error = function(e) NULL
  )
  if (is.null(fit) || length(stats::coef(fit)) < 2L) {
    out$calibration_policy <- "mean_bias"
    out$bias <- mean(resid, na.rm = TRUE)
    return(out)
  }
  co <- stats::coef(fit)
  intercept <- as.numeric(co[[1L]])
  slope <- as.numeric(co[[2L]])
  if (!is.finite(intercept) || !is.finite(slope)) {
    out$calibration_policy <- "mean_bias"
    out$bias <- mean(resid, na.rm = TRUE)
    return(out)
  }
  slope <- pmax(nmk_v3_affine_slope_min,
                pmin(nmk_v3_affine_slope_max, slope))
  out$intercept <- intercept
  out$slope <- slope
  out
}

nmk_apply_calibration_policy <- function(pred, cal) {
  pred <- as.numeric(pred)
  if (is.null(cal) || length(cal) == 0L) return(pred)
  policy <- as.character(cal$calibration_policy)[1]
  if (identical(policy, "affine")) {
    return(as.numeric(cal$intercept) + as.numeric(cal$slope) * pred)
  }
  pred + as.numeric(cal$bias)
}

nmk_select_postprocess_policy <- function(obs, pred_krig, pred_base = NULL,
                                          fold_id = NULL,
                                          rf_inner = NULL) {
  alpha_grid <- if (isTRUE(nmk_autotune_v3)) nmk_v3_alpha_grid else 1
  cal_grid <- if (isTRUE(nmk_autotune_v3)) nmk_v3_calibration_policies else "none"
  obs <- as.numeric(obs)
  pred_krig <- as.numeric(pred_krig)
  if (is.null(pred_base) || length(pred_base) != length(pred_krig)) {
    pred_base <- pred_krig
  }
  pred_base <- as.numeric(pred_base)
  can_crossfit <- !is.null(fold_id) &&
    length(fold_id) == length(obs) &&
    length(unique(fold_id[is.finite(fold_id)])) > 1L
  rf_metrics <- if (!is.null(rf_inner) && !is.null(rf_inner$metrics)) {
    rf_inner$metrics
  } else {
    nmk_empty_rf_inner()$metrics
  }
  rf_inner_rmse <- suppressWarnings(as.numeric(rf_metrics$RMSE))
  rf_inner_me <- suppressWarnings(as.numeric(rf_metrics$ME))
  rf_inner_r2 <- suppressWarnings(as.numeric(rf_metrics$r2))
  rf_inner_mec <- suppressWarnings(as.numeric(rf_metrics$MEC))
  rf_fold_rmse <- if (!is.null(rf_inner) && !is.null(rf_inner$fold_RMSE)) {
    suppressWarnings(as.numeric(rf_inner$fold_RMSE))
  } else {
    NA_real_
  }
  rf_rmse_sd <- if (length(rf_fold_rmse) > 1L) {
    stats::sd(rf_fold_rmse, na.rm = TRUE)
  } else {
    0
  }
  if (!is.finite(rf_rmse_sd)) rf_rmse_sd <- 0
  rows <- list()
  best <- NULL
  for (alpha in alpha_grid) {
    blended <- nmk_blend_values(pred_krig, pred_base, alpha)
    for (cal_policy in cal_grid) {
      final_cal <- nmk_fit_calibration_policy(obs, blended, cal_policy)
      pred_scored <- if (can_crossfit && !identical(cal_policy, "none")) {
        pred_cf <- rep(NA_real_, length(obs))
        for (fold in sort(unique(fold_id))) {
          score_idx <- which(fold_id == fold)
          fit_idx <- which(fold_id != fold)
          cal_fold <- nmk_fit_calibration_policy(
            obs[fit_idx], blended[fit_idx], cal_policy)
          pred_cf[score_idx] <- nmk_apply_calibration_policy(
            blended[score_idx], cal_fold)
        }
        pred_cf
      } else {
        nmk_apply_calibration_policy(blended, final_cal)
      }
      metrics <- nmk_metric_values(obs, pred_scored)
      fold_rmse <- nmk_fold_rmse_values(obs, pred_scored, fold_id)
      rmse_sd <- if (length(fold_rmse) > 1L) {
        stats::sd(fold_rmse, na.rm = TRUE)
      } else {
        0
      }
      if (!is.finite(rmse_sd)) rmse_sd <- 0
      q_rmse <- nmk_quantile_rmse_values(
        obs, pred_scored, n_bins = nmk_v4_quantile_bins)
      q_rmse_mean <- if (length(q_rmse) > 0L) {
        mean(q_rmse, na.rm = TRUE)
      } else {
        metrics$RMSE
      }
      q_rmse_sd <- if (length(q_rmse) > 1L) {
        stats::sd(q_rmse, na.rm = TRUE)
      } else {
        0
      }
      if (!is.finite(q_rmse_mean)) q_rmse_mean <- metrics$RMSE
      if (!is.finite(q_rmse_sd)) q_rmse_sd <- 0
      q_rmse_excess <- max(0, q_rmse_mean - metrics$RMSE)
      penalty <- nmk_v3_rmse_sd_weight * rmse_sd +
        nmk_v3_abs_me_weight * abs(metrics$ME) +
        nmk_v4_quantile_excess_weight * q_rmse_excess +
        nmk_v4_quantile_sd_weight * q_rmse_sd
      rmse_selection_score <- metrics$RMSE + penalty
      delta_rmse_vs_rf <- if (is.finite(metrics$RMSE) &&
                              is.finite(rf_inner_rmse)) {
        metrics$RMSE - rf_inner_rmse
      } else {
        NA_real_
      }
      selection_score <- if (identical(nmk_selection_objective,
                                       "paired_rf_delta") &&
                             is.finite(delta_rmse_vs_rf)) {
        delta_rmse_vs_rf + penalty
      } else {
        rmse_selection_score
      }
      row <- data.frame(
        post_alpha = alpha,
        post_calibration_requested = cal_policy,
        post_calibration_policy = final_cal$calibration_policy,
        post_bias_correction = final_cal$bias,
        post_affine_intercept = final_cal$intercept,
        post_affine_slope = final_cal$slope,
        post_crossfit_calibration = can_crossfit && !identical(cal_policy, "none"),
        post_ME = metrics$ME,
        post_RMSE = metrics$RMSE,
        post_r2 = metrics$r2,
        post_MEC = metrics$MEC,
        post_RMSE_sd = rmse_sd,
        post_fold_RMSE = nmk_format_fold_rmse(fold_rmse),
        post_quantile_RMSE_mean = q_rmse_mean,
        post_quantile_RMSE_sd = q_rmse_sd,
        post_quantile_RMSE_excess = q_rmse_excess,
        post_quantile_RMSE = nmk_format_fold_rmse(q_rmse),
        selection_objective = nmk_selection_objective,
        rf_inner_ME = rf_inner_me,
        rf_inner_RMSE = rf_inner_rmse,
        rf_inner_r2 = rf_inner_r2,
        rf_inner_MEC = rf_inner_mec,
        rf_inner_RMSE_sd = rf_rmse_sd,
        rf_inner_fold_RMSE = nmk_format_fold_rmse(rf_fold_rmse),
        delta_RMSE_vs_RF = delta_rmse_vs_rf,
        rmse_selection_score = rmse_selection_score,
        selection_score = selection_score,
        stringsAsFactors = FALSE
      )
      rows[[length(rows) + 1L]] <- row
      if (is.null(best) || selection_score < best$selection_score) {
        best <- c(
          as.list(row[1, , drop = FALSE]),
          list(
            alpha = alpha,
            calibration = final_cal,
            selection_score = selection_score,
            pred_val = pred_scored
          )
        )
      }
    }
  }
	  if (is.null(best)) {
	    final_cal <- nmk_fit_calibration_policy(obs, pred_krig, "none")
	    metrics <- nmk_metric_values(obs, pred_krig)
	    delta_default <- if (is.finite(metrics$RMSE) &&
	                         is.finite(rf_inner_rmse)) {
	      metrics$RMSE - rf_inner_rmse
	    } else {
	      NA_real_
	    }
	    default_selection_score <- if (identical(nmk_selection_objective,
	                                             "paired_rf_delta") &&
	                                   is.finite(delta_default)) {
	      delta_default
	    } else {
	      metrics$RMSE
	    }
	    best <- list(
      post_alpha = 1,
      post_calibration_requested = "none",
      post_calibration_policy = "none",
      post_bias_correction = 0,
      post_affine_intercept = 0,
      post_affine_slope = 1,
      post_crossfit_calibration = FALSE,
      post_ME = metrics$ME,
      post_RMSE = metrics$RMSE,
      post_r2 = metrics$r2,
      post_MEC = metrics$MEC,
      post_RMSE_sd = 0,
      post_fold_RMSE = "NA",
      post_quantile_RMSE_mean = metrics$RMSE,
      post_quantile_RMSE_sd = 0,
      post_quantile_RMSE_excess = 0,
      post_quantile_RMSE = "NA",
      selection_objective = nmk_selection_objective,
      rf_inner_ME = rf_inner_me,
      rf_inner_RMSE = rf_inner_rmse,
      rf_inner_r2 = rf_inner_r2,
      rf_inner_MEC = rf_inner_mec,
      rf_inner_RMSE_sd = rf_rmse_sd,
      rf_inner_fold_RMSE = nmk_format_fold_rmse(rf_fold_rmse),
	      delta_RMSE_vs_RF = delta_default,
	      rmse_selection_score = metrics$RMSE,
	      selection_score = default_selection_score,
      alpha = 1,
      calibration = final_cal,
      pred_val = pred_krig
    )
  }
  best$post_grid <- bind_rows_fill_nmk(rows)
  best
}

nmk_apply_postprocess_values <- function(pred_krig, pred_base = NULL, policy = NULL) {
  if (is.null(policy) || length(policy) == 0L) return(as.numeric(pred_krig))
  blended <- nmk_blend_values(pred_krig, pred_base, policy$alpha)
  nmk_apply_calibration_policy(blended, policy$calibration)
}

nmk_attach_postprocess_diag <- function(diag, policy, tune_stage = NA_character_) {
  if (is.null(diag)) diag <- list()
  if (is.null(policy) || length(policy) == 0L) return(diag)
  diag$autotune_v3 <- isTRUE(nmk_autotune_v3)
  diag$autotune_v4 <- isTRUE(nmk_autotune_v4)
  diag$tune_stage <- tune_stage
  diag$post_alpha <- as.numeric(policy$alpha)
  diag$post_calibration_requested <- as.character(policy$post_calibration_requested)
  diag$post_calibration_policy <- as.character(policy$post_calibration_policy)
  diag$post_bias_correction <- as.numeric(policy$post_bias_correction)
  diag$post_affine_intercept <- as.numeric(policy$post_affine_intercept)
  diag$post_affine_slope <- as.numeric(policy$post_affine_slope)
  diag$post_crossfit_calibration <- isTRUE(policy$post_crossfit_calibration)
  diag$post_selection_score <- as.numeric(policy$selection_score)
  diag$post_val_rmse <- as.numeric(policy$post_RMSE)
  diag$post_val_me <- as.numeric(policy$post_ME)
  diag$post_rmse_sd <- as.numeric(policy$post_RMSE_sd)
  diag$post_fold_RMSE <- as.character(policy$post_fold_RMSE)
  diag$post_quantile_RMSE_mean <- as.numeric(policy$post_quantile_RMSE_mean)
  diag$post_quantile_RMSE_sd <- as.numeric(policy$post_quantile_RMSE_sd)
  diag$post_quantile_RMSE_excess <- as.numeric(policy$post_quantile_RMSE_excess)
  diag$post_quantile_RMSE <- as.character(policy$post_quantile_RMSE)
  diag$post_selection_objective <- as.character(policy$selection_objective)
  diag$post_rf_inner_ME <- as.numeric(policy$rf_inner_ME)
  diag$post_rf_inner_RMSE <- as.numeric(policy$rf_inner_RMSE)
  diag$post_rf_inner_r2 <- as.numeric(policy$rf_inner_r2)
  diag$post_rf_inner_MEC <- as.numeric(policy$rf_inner_MEC)
  diag$post_rf_inner_RMSE_sd <- as.numeric(policy$rf_inner_RMSE_sd)
  diag$post_rf_inner_fold_RMSE <- as.character(policy$rf_inner_fold_RMSE)
  diag$post_delta_RMSE_vs_RF <- as.numeric(policy$delta_RMSE_vs_RF)
  diag$post_rmse_selection_score <- as.numeric(policy$rmse_selection_score)
  diag$post_alpha_grid <- paste(nmk_v3_alpha_grid, collapse = "|")
  diag$post_calibration_grid <- paste(nmk_v3_calibration_policies, collapse = "|")
  diag$post_quantile_bins <- nmk_v4_quantile_bins
  diag$post_quantile_excess_weight <- nmk_v4_quantile_excess_weight
  diag$post_quantile_sd_weight <- nmk_v4_quantile_sd_weight
  diag
}

nmk_apply_postprocess_out <- function(out, policy = NULL,
                                      tune_stage = NA_character_) {
  if (is.null(out) || is.null(policy) || length(policy) == 0L) return(out)
  out$pred_val_raw <- out$pred_val
  out$pred_test_raw <- out$pred_test
  out$pred_val <- nmk_apply_postprocess_values(
    out$pred_val, out$pred_val_base, policy)
  out$pred_test <- nmk_apply_postprocess_values(
    out$pred_test, out$pred_test_base, policy)
  out$diagnostics <- nmk_attach_postprocess_diag(
    out$diagnostics, policy, tune_stage = tune_stage)
  out
}

nmk_eval_key <- function(x) {
  paste(as.character(x$candidate_id), as.character(x$tune_stage), sep = "::")
}

nmk_candidate_label <- function(score_df, candidate_id, tune_stage) {
  idx <- which(score_df$candidate_id == candidate_id &
                 score_df$tune_stage == tune_stage)
  if (length(idx) == 0L) return(as.character(candidate_id))
  as.character(score_df$candidate_label[idx[1L]])
}

nmk_rank_ensemble_candidates <- function(best_eval, candidate_pool) {
  pool <- c(list(best_eval), candidate_pool)
  ok <- vapply(pool, function(x) {
    !is.null(x) && is.finite(x$selection_score)
  }, logical(1))
  pool <- pool[ok]
  if (length(pool) == 0L) return(list())
  keys <- vapply(pool, nmk_eval_key, character(1))
  pool <- pool[!duplicated(keys)]
  ord <- order(
    vapply(pool, `[[`, numeric(1), "selection_score"),
    vapply(pool, `[[`, numeric(1), "rmse"),
    vapply(pool, `[[`, character(1), "tune_stage")
  )
  pool[ord]
}

nmk_unique_candidate_evals <- function(pool) {
  pool <- pool[vapply(pool, function(x) {
    !is.null(x) && is.finite(x$selection_score)
  }, logical(1))]
  if (length(pool) == 0L) return(list())
  keys <- vapply(pool, nmk_eval_key, character(1))
  pool[!duplicated(keys)]
}

nmk_conservative_candidate_pool <- function(pool, grid) {
  pool <- nmk_unique_candidate_evals(pool)
  if (length(pool) == 0L) return(list())
  meta <- lapply(pool, function(x) {
    row <- grid[grid$candidate_id == x$candidate_id, , drop = FALSE]
    if (nrow(row) == 0L) row <- data.frame()
    row[1, , drop = FALSE]
  })
  profile <- vapply(meta, function(x) {
    if ("profile" %in% names(x)) as.character(x$profile[1]) else ""
  }, character(1))
  transform <- vapply(meta, function(x) {
    if ("target_transform_effective" %in% names(x)) {
      as.character(x$target_transform_effective[1])
    } else {
      ""
    }
  }, character(1))
  is_auto_base <- vapply(meta, function(x) {
    if ("is_auto_base" %in% names(x)) isTRUE(x$is_auto_base[1]) else FALSE
  }, logical(1))
  conservative_profile <- profile %in% c(
    "auto_current", "compact_spatial", "balanced_spatial",
    "tab_skip_lowvar", "tab_qbasis_trend", "rf_capacity_skip_only",
    "rf_capacity_patch_suppressed")
  non_log <- transform %in% c("identity", "auto")
  tiers <- list(
    which(is_auto_base & non_log),
    which(profile == "auto_current" & non_log),
    which(conservative_profile & non_log),
    which(conservative_profile),
    which(non_log)
  )
  for (idx in tiers) {
    if (length(idx) > 0L) return(pool[idx])
  }
  pool
}

nmk_best_candidate_eval <- function(pool) {
  pool <- nmk_unique_candidate_evals(pool)
  if (length(pool) == 0L) return(NULL)
  ord <- order(
    vapply(pool, `[[`, numeric(1), "selection_score"),
    vapply(pool, `[[`, numeric(1), "rmse"),
    vapply(pool, `[[`, character(1), "tune_stage")
  )
  pool[[ord[1L]]]
}

nmk_ensemble_weights <- function(scores) {
  scores <- as.numeric(scores)
  if (length(scores) == 0L) return(numeric())
  finite <- is.finite(scores)
  if (!any(finite)) return(rep(1 / length(scores), length(scores)))
  scores[!finite] <- max(scores[finite], na.rm = TRUE) + 1e6
  if (length(scores) == 1L || identical(nmk_ensemble_weighting, "uniform")) {
    return(rep(1 / length(scores), length(scores)))
  }
  shifted <- scores - min(scores, na.rm = TRUE)
  if (identical(nmk_ensemble_weighting, "inverse")) {
    w <- 1 / pmax(shifted + nmk_ensemble_temperature, 1e-8)
  } else {
    w <- exp(-shifted / nmk_ensemble_temperature)
  }
  if (!all(is.finite(w)) || sum(w) <= 0) {
    return(rep(1 / length(scores), length(scores)))
  }
  w / sum(w)
}

nmk_get_warm_states <- function(out) {
  if (is.null(out)) return(NULL)
  if (!is.null(out$selected_states)) return(out$selected_states)
  if (!is.null(out$selected_state)) return(list(out$selected_state))
  NULL
}

nmk_refit_candidate_full <- function(candidate_eval, candidate_label,
                                     selected_inner_out = NULL,
                                     train_core, val_df, test_df,
                                     calibration_full, refit_monitor,
                                     calibration_patch_cache,
                                     iter, member_index) {
  params <- candidate_eval$params
  policy <- candidate_eval$policy
  inner_out <- selected_inner_out
  cid_int <- suppressWarnings(as.integer(candidate_eval$candidate_id))
  if (!is.finite(cid_int)) cid_int <- as.integer(member_index)
  if (is.null(inner_out)) {
    params_inner <- params
    params_inner$train_seed <- as.integer(
      train_seed + 910000L * iter + 1000L * member_index +
        cid_int)
    if (isTRUE(nmk_refit_warm_start)) {
      params_inner$return_state <- TRUE
    }
    cat(sprintf(
      "[NMK ensemble] fitting inner state for member %d: %s\n",
      member_index, candidate_label))
    inner_out <- tryCatch(
      run_nmk_on_split(
        train_core = train_core, val_df = val_df, test_df = test_df,
        calibration_patch_cache = calibration_patch_cache,
        params = params_inner),
      error = function(e) e
    )
    if (inherits(inner_out, "error")) {
      return(list(
        status = "inner_error",
        error = conditionMessage(inner_out),
        candidate_id = candidate_eval$candidate_id,
        candidate_label = candidate_label,
        tune_stage = candidate_eval$tune_stage
      ))
    }
    inner_out <- nmk_apply_postprocess_out(
      inner_out, policy, tune_stage = candidate_eval$tune_stage)
  }

  refit_params <- nmk_refit_params_from_selection(
    inner_out$diagnostics, base_params = params)
  refit_params$train_seed <- as.integer(
    train_seed + 930000L * iter + 1000L * member_index +
      cid_int)
  warm_states <- nmk_get_warm_states(inner_out)
  warm_start_applied <- isTRUE(nmk_refit_warm_start) && !is.null(warm_states)
  if (isTRUE(warm_start_applied)) {
    refit_params$init_states <- warm_states
    refit_params$return_state <- FALSE
  }

  cat(sprintf(
    "[NMK ensemble] refitting full-calibration member %d: %s\n",
    member_index, candidate_label))
  out <- tryCatch(
    run_nmk_on_split(
      train_core = calibration_full, val_df = refit_monitor,
      test_df = test_df, calibration_patch_cache = calibration_patch_cache,
      params = refit_params),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    return(list(
      status = "refit_error",
      error = conditionMessage(out),
      candidate_id = candidate_eval$candidate_id,
      candidate_label = candidate_label,
      tune_stage = candidate_eval$tune_stage,
      warm_start_applied = warm_start_applied
    ))
  }
  out <- nmk_apply_postprocess_out(
    out, policy, tune_stage = candidate_eval$tune_stage)
  list(
    status = "ok",
    error = NA_character_,
    candidate_id = candidate_eval$candidate_id,
    candidate_label = candidate_label,
    tune_stage = candidate_eval$tune_stage,
    score = candidate_eval$selection_score,
    pred_test = out$pred_test,
    out = out,
    warm_start_applied = warm_start_applied
  )
}

tune_nmk_on_split <- function(train_core, val_df, test_df,
                              calibration_patch_cache,
                              iter, protocol, model_label) {
  use_cv_tuning <- nmk_tune_inner_folds > 1L
  tune_pool <- if (use_cv_tuning) {
    rbind(train_core, val_df)
  } else {
    NULL
  }
  if (use_cv_tuning && nrow(tune_pool) < max(20L, 2L * nmk_tune_inner_folds)) {
    use_cv_tuning <- FALSE
  }
  fold_ids <- NULL
  tune_source <- if (use_cv_tuning) "cv_oof" else "holdout"
  tune_n <- if (use_cv_tuning) nrow(tune_pool) else nrow(val_df)
  tune_fold_sizes <- NA_character_
  if (use_cv_tuning) {
    # #5: spatial-block CV for the selection OOF, auto-engaged only when the
    # calibration pool is clustered (Clark-Evans); random folds otherwise.
    fold_ids <- NULL
    nmk_tp_coords <- tryCatch(
      as.matrix(tune_pool[, c("x", "y"), drop = FALSE]), error = function(e) NULL)
    if (!is.null(nmk_tp_coords) && wadoux_is_clustered(nmk_tp_coords)) {
      fold_ids <- wadoux_spatial_block_fold_ids(
        nmk_tp_coords, nmk_tune_inner_folds, seed = train_seed + 91000L + iter)
    }
    if (is.null(fold_ids)) {
      fold_ids <- make_regression_folds_nmk(
        tune_pool[[response_name_wadoux]],
        nmk_tune_inner_folds,
        seed = train_seed + 91000L + iter)
    }
    tune_fold_sizes <- paste(as.integer(table(fold_ids)), collapse = "|")
  }
  rf_inner <- nmk_inner_rf_baseline(
    train_core = train_core, val_df = val_df, tune_pool = tune_pool,
    fold_ids = fold_ids, use_cv = use_cv_tuning)

  grid <- nmk_tune_grid(if (use_cv_tuning) tune_pool else train_core)
  can_parallel <- FALSE
  if (nmk_num_workers > 1L) {
    cat("[NMK tune] NMK_NUM_WORKERS requested, but NMK torch tuning runs ",
        "sequentially to avoid fork-related torch crashes.\n", sep = "")
  }

  eval_candidate <- function(i, tune_stage = "stage1_screen",
                             seed_offset = 10000L,
                             n_seeds_override = NULL) {
    cand <- grid[i, , drop = FALSE]
    cat(sprintf(
      "[NMK tune] iter=%d %s %s candidate %d/%d: %s | source=%s\n",
      iter, model_label, tune_stage, i, nrow(grid), cand$candidate_label,
      tune_source))
    params_i <- nmk_params_from_candidate(
      cand,
      candidate_seed = train_seed + seed_offset * iter + i)
    if (!is.null(n_seeds_override) &&
        is.finite(n_seeds_override) &&
        n_seeds_override > 0L) {
      params_i$n_seeds <- as.integer(n_seeds_override)
    }
    started <- Sys.time()
    out_i <- NULL
    preds_val_krig <- NULL
    preds_val_base <- NULL
    obs_val <- NULL
    fold_rmse <- NA_character_
    post_policy <- NULL
    folds_completed <- 0L
    if (use_cv_tuning) {
      preds_val_krig <- rep(NA_real_, nrow(tune_pool))
      preds_val_base <- rep(NA_real_, nrow(tune_pool))
      fold_levels <- sort(unique(fold_ids))
      fold_scores <- setNames(rep(NA_real_, length(fold_levels)),
                              as.character(fold_levels))
      fold_errors <- character()
      for (fold in fold_levels) {
        val_idx <- which(fold_ids == fold)
        train_idx <- which(fold_ids != fold)
        params_fold <- params_i
        params_fold$train_seed <- as.integer(params_i$train_seed) + as.integer(fold)
        fold_out <- tryCatch(
          run_nmk_on_split(
            train_core = tune_pool[train_idx, , drop = FALSE],
            val_df = tune_pool[val_idx, , drop = FALSE],
            test_df = tune_pool[val_idx, , drop = FALSE],
            calibration_patch_cache = calibration_patch_cache,
            params = params_fold),
          error = function(e) e
        )
        if (inherits(fold_out, "error")) {
          fold_errors <- c(fold_errors, conditionMessage(fold_out))
          next
        }
        preds_val_krig[val_idx] <- fold_out$pred_val
        preds_val_base[val_idx] <- if (!is.null(fold_out$pred_val_base)) {
          fold_out$pred_val_base
        } else {
          fold_out$pred_val
        }
        m_fold <- wadoux_eval(
          obs = tune_pool[[response_name_wadoux]][val_idx],
          pred = fold_out$pred_val)
        fold_scores[as.character(fold)] <- m_fold$RMSE
        folds_completed <- folds_completed + 1L
      }
      fold_rmse <- paste(ifelse(is.finite(fold_scores),
                                sprintf("%.6f", fold_scores), "NA"),
                         collapse = "|")
      obs_val <- tune_pool[[response_name_wadoux]]
      if (length(fold_errors) > 0L || any(!is.finite(preds_val_krig))) {
        msg <- paste(unique(fold_errors[nzchar(fold_errors)]), collapse = " | ")
        if (!nzchar(msg)) msg <- "one or more inner-fold predictions were non-finite"
        out_i <- simpleError(msg)
      }
    } else {
      out_i <- tryCatch(
        run_nmk_on_split(
          train_core = train_core, val_df = val_df, test_df = test_df,
          calibration_patch_cache = calibration_patch_cache,
          params = params_i),
        error = function(e) e
      )
      if (!inherits(out_i, "error")) {
        preds_val_krig <- out_i$pred_val
        preds_val_base <- if (!is.null(out_i$pred_val_base)) {
          out_i$pred_val_base
        } else {
          out_i$pred_val
        }
        obs_val <- val_df[[response_name_wadoux]]
        folds_completed <- 1L
        m_fold <- wadoux_eval(obs = obs_val, pred = preds_val_krig)
        fold_rmse <- if (is.finite(m_fold$RMSE)) {
          sprintf("%.6f", m_fold$RMSE)
        } else {
          "NA"
        }
      }
    }
    elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    row_i <- cbind(
      data.frame(
        iteration = iter, scenario = scenario, protocol = protocol,
        model = model_label, selected = FALSE, status = "ok",
        tune_source = tune_source,
        tune_inner_folds = if (use_cv_tuning) nmk_tune_inner_folds else 1L,
        tune_n = tune_n,
        tune_fold_sizes = tune_fold_sizes,
        tune_num_workers = 1L,
        tune_num_workers_requested = nmk_num_workers,
        tune_autotune_v4 = nmk_autotune_v4,
        tune_stage = tune_stage,
        tune_stage2_top_k = if (isTRUE(nmk_autotune_v3)) {
          nmk_v3_stage2_top_k
        } else {
          0L
        },
        tune_stage2_n_seeds = if (identical(tune_stage, "stage2_refine")) {
          as.integer(params_i$n_seeds)
        } else {
          NA_integer_
        },
        folds_completed = folds_completed,
        raw_fold_RMSE = fold_rmse,
        elapsed_seconds = elapsed, stringsAsFactors = FALSE),
      cand
    )
    if (inherits(out_i, "error")) {
      row_i$status <- "error"
      row_i$error <- conditionMessage(out_i)
      row_i$val_ME <- NA_real_
      row_i$val_RMSE <- NA_real_
      row_i$val_r2 <- NA_real_
      row_i$val_MEC <- NA_real_
      row_i$raw_val_ME <- NA_real_
      row_i$raw_val_RMSE <- NA_real_
      row_i$raw_val_r2 <- NA_real_
      row_i$raw_val_MEC <- NA_real_
      row_i$post_alpha <- NA_real_
      row_i$post_calibration_requested <- NA_character_
      row_i$post_calibration_policy <- NA_character_
      row_i$post_bias_correction <- NA_real_
      row_i$post_affine_intercept <- NA_real_
      row_i$post_affine_slope <- NA_real_
	      row_i$post_crossfit_calibration <- NA
	      row_i$post_RMSE_sd <- NA_real_
	      row_i$post_fold_RMSE <- NA_character_
	      row_i$selection_objective <- nmk_selection_objective
	      row_i$rf_inner_status <- rf_inner$status
	      row_i$rf_inner_error <- rf_inner$error
	      row_i$rf_inner_ME <- suppressWarnings(as.numeric(rf_inner$metrics$ME))
	      row_i$rf_inner_RMSE <- suppressWarnings(as.numeric(rf_inner$metrics$RMSE))
	      row_i$rf_inner_r2 <- suppressWarnings(as.numeric(rf_inner$metrics$r2))
	      row_i$rf_inner_MEC <- suppressWarnings(as.numeric(rf_inner$metrics$MEC))
	      row_i$rf_inner_RMSE_sd <- NA_real_
	      row_i$rf_inner_fold_RMSE <- if (!is.null(rf_inner$fold_RMSE)) {
	        nmk_format_fold_rmse(rf_inner$fold_RMSE)
	      } else {
	        NA_character_
	      }
	      row_i$delta_RMSE_vs_RF <- NA_real_
	      row_i$rmse_selection_score <- NA_real_
	      row_i$selection_score <- Inf
	    } else {
	      raw_m <- nmk_metric_values(obs = obs_val, pred = preds_val_krig)
      post_policy <- nmk_select_postprocess_policy(
	        obs = obs_val,
	        pred_krig = preds_val_krig,
	        pred_base = preds_val_base,
	        fold_id = if (use_cv_tuning) fold_ids else NULL,
	        rf_inner = rf_inner
	      )
      row_i$error <- NA_character_
      row_i$raw_val_ME <- raw_m$ME
      row_i$raw_val_RMSE <- raw_m$RMSE
      row_i$raw_val_r2 <- raw_m$r2
      row_i$raw_val_MEC <- raw_m$MEC
      row_i$val_ME <- post_policy$post_ME
      row_i$val_RMSE <- post_policy$post_RMSE
      row_i$val_r2 <- post_policy$post_r2
      row_i$val_MEC <- post_policy$post_MEC
      row_i$post_alpha <- post_policy$post_alpha
      row_i$post_calibration_requested <- post_policy$post_calibration_requested
      row_i$post_calibration_policy <- post_policy$post_calibration_policy
      row_i$post_bias_correction <- post_policy$post_bias_correction
      row_i$post_affine_intercept <- post_policy$post_affine_intercept
      row_i$post_affine_slope <- post_policy$post_affine_slope
	      row_i$post_crossfit_calibration <- post_policy$post_crossfit_calibration
	      row_i$post_RMSE_sd <- post_policy$post_RMSE_sd
	      row_i$post_fold_RMSE <- post_policy$post_fold_RMSE
	      row_i$selection_objective <- post_policy$selection_objective
	      row_i$rf_inner_status <- rf_inner$status
	      row_i$rf_inner_error <- rf_inner$error
	      row_i$rf_inner_ME <- post_policy$rf_inner_ME
	      row_i$rf_inner_RMSE <- post_policy$rf_inner_RMSE
	      row_i$rf_inner_r2 <- post_policy$rf_inner_r2
	      row_i$rf_inner_MEC <- post_policy$rf_inner_MEC
	      row_i$rf_inner_RMSE_sd <- post_policy$rf_inner_RMSE_sd
	      row_i$rf_inner_fold_RMSE <- post_policy$rf_inner_fold_RMSE
	      row_i$delta_RMSE_vs_RF <- post_policy$delta_RMSE_vs_RF
	      row_i$rmse_selection_score <- post_policy$rmse_selection_score
	      row_i$selection_score <- post_policy$selection_score
	    }
    list(
      row = row_i,
      candidate_index = i,
      candidate_id = cand$candidate_id,
      tune_stage = tune_stage,
      selection_score = if (inherits(out_i, "error")) {
        Inf
      } else {
        post_policy$selection_score
      },
      rmse = if (inherits(out_i, "error")) Inf else post_policy$post_RMSE,
      out = if (!use_cv_tuning && !inherits(out_i, "error")) out_i else NULL,
      params = params_i,
      policy = post_policy
    )
  }

  idx <- seq_len(nrow(grid))
  eval_results <- if (isTRUE(can_parallel)) {
    parallel::mclapply(
      idx, eval_candidate,
      mc.cores = min(nmk_num_workers, length(idx)),
      mc.preschedule = FALSE,
      mc.set.seed = FALSE)
  } else {
    lapply(idx, eval_candidate)
  }

  stage1_results <- eval_results
  score_df <- bind_rows_fill_nmk(lapply(stage1_results, `[[`, "row"))
  stage1_successful <- stage1_results[vapply(stage1_results, function(x) {
    is.finite(x$selection_score)
  }, logical(1))]
  if (length(stage1_successful) == 0L) {
    stop("NMK tuning failed: no candidate completed successfully.")
  }

  stage2_results <- list()
  if (isTRUE(nmk_autotune_v3) && nmk_v3_stage2_top_k > 0L) {
    stage1_order <- order(vapply(stage1_successful, `[[`, numeric(1),
                                 "selection_score"))
    top <- stage1_successful[stage1_order]
    top <- top[seq_len(min(length(top), nmk_v3_stage2_top_k))]
    top_idx <- unique(vapply(top, `[[`, integer(1), "candidate_index"))
    stage2_results <- lapply(top_idx, eval_candidate,
                             tune_stage = "stage2_refine",
                             seed_offset = 50000L,
                             n_seeds_override = nmk_v3_stage2_n_seeds)
    if (length(stage2_results) > 0L) {
      score_df <- bind_rows_fill_nmk(c(
        list(score_df),
        lapply(stage2_results, `[[`, "row")
      ))
    }
  }

  candidate_pool <- if (length(stage2_results) > 0L) {
    stage2_successful <- stage2_results[vapply(stage2_results, function(x) {
      is.finite(x$selection_score)
    }, logical(1))]
    if (length(stage2_successful) > 0L) stage2_successful else stage1_successful
  } else {
    stage1_successful
  }

  best_eval <- candidate_pool[[which.min(vapply(candidate_pool, `[[`, numeric(1),
                                                "selection_score"))]]
	  if (isTRUE(nmk_autotune_v3)) {
	    transforms <- vapply(candidate_pool, function(x) {
      as.character(grid$target_transform_effective[
        grid$candidate_id == x$candidate_id][1])
    }, character(1))
    identity_pool <- candidate_pool[transforms == "identity"]
    if (length(identity_pool) == 0L) {
      stage1_transforms <- vapply(stage1_successful, function(x) {
        as.character(grid$target_transform_effective[
          grid$candidate_id == x$candidate_id][1])
      }, character(1))
      identity_pool <- stage1_successful[stage1_transforms == "identity"]
    }
    best_transform <- as.character(grid$target_transform_effective[
      grid$candidate_id == best_eval$candidate_id][1])
    if (identical(best_transform, "log") && length(identity_pool) > 0L) {
      best_identity <- identity_pool[[which.min(vapply(
        identity_pool, `[[`, numeric(1), "selection_score"))]]
      if (best_eval$selection_score >
          best_identity$selection_score - nmk_v3_log_margin) {
        best_eval <- best_identity
	      }
	    }
	  }

	  pre_noharm_eval <- best_eval
	  noharm_anchor_pool <- nmk_conservative_candidate_pool(
	    c(candidate_pool, stage1_successful), grid)
	  noharm_anchor_eval <- nmk_best_candidate_eval(noharm_anchor_pool)
	  noharm_anchor_score <- NA_real_
	  noharm_anchor_rmse <- NA_real_
	  noharm_anchor_label <- NA_character_
	  noharm_margin <- NA_real_
	  noharm_delta_vs_anchor <- NA_real_
	  noharm_applied <- FALSE
	  noharm_reason <- if (isTRUE(nmk_noharm_policy)) {
	    "no_anchor_available"
	  } else {
	    "disabled"
	  }
	  if (isTRUE(nmk_noharm_policy) && !is.null(noharm_anchor_eval)) {
	    noharm_anchor_score <- as.numeric(noharm_anchor_eval$selection_score)
	    noharm_anchor_rmse <- as.numeric(noharm_anchor_eval$rmse)
	    noharm_anchor_label <- nmk_candidate_label(
	      score_df, noharm_anchor_eval$candidate_id,
	      noharm_anchor_eval$tune_stage)
	    anchor_scale <- noharm_anchor_rmse
	    if (!is.finite(anchor_scale)) anchor_scale <- abs(noharm_anchor_score)
	    if (!is.finite(anchor_scale)) anchor_scale <- 0
	    noharm_margin <- nmk_noharm_margin_abs +
	      nmk_noharm_margin_rel * abs(anchor_scale)
	    if (!identical(nmk_eval_key(best_eval), nmk_eval_key(noharm_anchor_eval))) {
	      noharm_delta_vs_anchor <- noharm_anchor_score -
	        as.numeric(best_eval$selection_score)
	      if (!is.finite(noharm_delta_vs_anchor) ||
	          noharm_delta_vs_anchor < noharm_margin) {
	        best_eval <- noharm_anchor_eval
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
		  if (isTRUE(noharm_applied)) {
		    candidate_pool <- nmk_unique_candidate_evals(
		      c(list(best_eval), candidate_pool, stage1_successful))
		    if (isTRUE(nmk_noharm_restrict_ensemble)) {
		      candidate_pool <- nmk_conservative_candidate_pool(candidate_pool, grid)
		      candidate_pool <- nmk_unique_candidate_evals(
		        c(list(best_eval), candidate_pool))
		    }
		  }
		  pre_rf_noharm_eval <- best_eval
		  rf_noharm_preselected_delta <- as.numeric(
		    pre_rf_noharm_eval$policy$delta_RMSE_vs_RF)
		  rf_noharm_anchor_delta <- if (!is.null(noharm_anchor_eval)) {
		    as.numeric(noharm_anchor_eval$policy$delta_RMSE_vs_RF)
		  } else {
		    NA_real_
		  }
		  rf_noharm_margin <- NA_real_
		  rf_noharm_required_delta <- NA_real_
		  rf_noharm_failed <- FALSE
		  rf_noharm_applied <- FALSE
		  rf_noharm_reason <- if (isTRUE(nmk_rf_noharm_policy)) {
		    "not_assessed"
		  } else {
		    "disabled"
		  }
		  if (isTRUE(nmk_rf_noharm_policy)) {
		    rf_scale <- as.numeric(pre_rf_noharm_eval$policy$rf_inner_RMSE)
		    if (!is.finite(rf_scale)) rf_scale <- 0
		    rf_noharm_margin <- nmk_rf_noharm_margin_abs +
		      nmk_rf_noharm_margin_rel * abs(rf_scale)
		    rf_noharm_required_delta <- -rf_noharm_margin
		    if (is.finite(rf_noharm_preselected_delta)) {
		      if (rf_noharm_preselected_delta <= rf_noharm_required_delta) {
		        rf_noharm_reason <- sprintf(
		          "passed_rf_inner:delta=%.6f<=required=%.6f",
		          rf_noharm_preselected_delta, rf_noharm_required_delta)
		      } else {
		        rf_noharm_failed <- TRUE
		        if (!is.null(noharm_anchor_eval)) {
		          if (!identical(nmk_eval_key(best_eval),
		                         nmk_eval_key(noharm_anchor_eval))) {
		            best_eval <- noharm_anchor_eval
		            rf_noharm_applied <- TRUE
		            rf_noharm_reason <- sprintf(
		              "fallback_to_anchor:delta=%.6f>required=%.6f",
		              rf_noharm_preselected_delta, rf_noharm_required_delta)
		          } else {
		            rf_noharm_reason <- sprintf(
		              "selected_anchor_still_fails_rf:delta=%.6f>required=%.6f",
		              rf_noharm_preselected_delta, rf_noharm_required_delta)
		          }
		          if (isTRUE(nmk_rf_noharm_restrict_ensemble)) {
		            candidate_pool <- nmk_conservative_candidate_pool(
		              candidate_pool, grid)
		            candidate_pool <- nmk_unique_candidate_evals(
		              c(list(best_eval), candidate_pool))
		          } else {
		            candidate_pool <- nmk_unique_candidate_evals(
		              c(list(best_eval), candidate_pool))
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

		  best_score <- best_eval$selection_score
  best_rmse <- best_eval$rmse
  best_out <- best_eval$out
  best_params <- best_eval$params
  best_i <- best_eval$candidate_id
  best_stage <- best_eval$tune_stage
  best_policy <- best_eval$policy
  ensemble_candidates <- nmk_rank_ensemble_candidates(best_eval, candidate_pool)
  if (length(ensemble_candidates) > 0L) {
    ensemble_candidates <- ensemble_candidates[seq_len(min(
      nmk_ensemble_top_k, length(ensemble_candidates)))]
  }
  ensemble_scores <- if (length(ensemble_candidates) > 0L) {
    vapply(ensemble_candidates, `[[`, numeric(1), "selection_score")
  } else {
    numeric()
  }
  ensemble_weights <- nmk_ensemble_weights(ensemble_scores)
  ensemble_labels <- if (length(ensemble_candidates) > 0L) {
    vapply(ensemble_candidates, function(x) {
      nmk_candidate_label(score_df, x$candidate_id, x$tune_stage)
    }, character(1))
  } else {
    character()
  }
  score_df$selected <- FALSE
  score_df$selected[
    score_df$candidate_id == best_i &
      score_df$tune_stage == best_stage
  ] <- TRUE
  score_df$ensemble_member <- FALSE
  score_df$ensemble_weight <- NA_real_
	  if (length(ensemble_candidates) > 0L) {
	    for (e_i in seq_along(ensemble_candidates)) {
      e <- ensemble_candidates[[e_i]]
      idx <- which(score_df$candidate_id == e$candidate_id &
                     score_df$tune_stage == e$tune_stage)
      if (length(idx) > 0L) {
        score_df$ensemble_member[idx] <- TRUE
        score_df$ensemble_weight[idx] <- ensemble_weights[[e_i]]
	      }
	    }
	  }
	  score_df$noharm_policy <- nmk_noharm_policy
	  score_df$noharm_anchor_candidate_id <- if (!is.null(noharm_anchor_eval)) {
	    noharm_anchor_eval$candidate_id
	  } else {
	    NA_integer_
	  }
	  score_df$noharm_anchor_tune_stage <- if (!is.null(noharm_anchor_eval)) {
	    noharm_anchor_eval$tune_stage
	  } else {
	    NA_character_
	  }
	  score_df$noharm_anchor_candidate_label <- noharm_anchor_label
	  score_df$noharm_anchor_score <- noharm_anchor_score
	  score_df$noharm_anchor_rmse <- noharm_anchor_rmse
	  score_df$noharm_margin_abs <- nmk_noharm_margin_abs
	  score_df$noharm_margin_rel <- nmk_noharm_margin_rel
	  score_df$noharm_margin <- noharm_margin
	  score_df$noharm_delta_vs_anchor <- noharm_delta_vs_anchor
		  score_df$noharm_applied <- noharm_applied
		  score_df$noharm_restrict_ensemble <- nmk_noharm_restrict_ensemble
		  score_df$noharm_reason <- noharm_reason
		  score_df$rf_noharm_policy <- nmk_rf_noharm_policy
		  score_df$rf_noharm_preselected_candidate_id <-
		    pre_rf_noharm_eval$candidate_id
		  score_df$rf_noharm_preselected_tune_stage <-
		    pre_rf_noharm_eval$tune_stage
		  score_df$rf_noharm_preselected_delta_RMSE_vs_RF <-
		    rf_noharm_preselected_delta
		  score_df$rf_noharm_anchor_candidate_id <-
		    if (!is.null(noharm_anchor_eval)) noharm_anchor_eval$candidate_id else NA_integer_
		  score_df$rf_noharm_anchor_tune_stage <-
		    if (!is.null(noharm_anchor_eval)) noharm_anchor_eval$tune_stage else NA_character_
		  score_df$rf_noharm_anchor_candidate_label <- noharm_anchor_label
		  score_df$rf_noharm_anchor_delta_RMSE_vs_RF <- rf_noharm_anchor_delta
		  score_df$rf_noharm_margin_abs <- nmk_rf_noharm_margin_abs
		  score_df$rf_noharm_margin_rel <- nmk_rf_noharm_margin_rel
		  score_df$rf_noharm_margin <- rf_noharm_margin
		  score_df$rf_noharm_required_delta_RMSE_vs_RF <-
		    rf_noharm_required_delta
		  score_df$rf_noharm_failed <- rf_noharm_failed
		  score_df$rf_noharm_applied <- rf_noharm_applied
		  score_df$rf_noharm_restrict_ensemble <- nmk_rf_noharm_restrict_ensemble
		  score_df$rf_noharm_reason <- rf_noharm_reason
	  if (is.null(best_out)) {
    best_params$train_seed <- as.integer(train_seed + 90000L * iter + best_i)
    if (isTRUE(nmk_refit_warm_start)) {
      best_params$return_state <- TRUE
    }
    best_out <- run_nmk_on_split(
      train_core = train_core, val_df = val_df, test_df = test_df,
      calibration_patch_cache = calibration_patch_cache,
      params = best_params)
  } else if (isTRUE(nmk_refit_warm_start) &&
             is.null(best_out$selected_states) &&
             is.null(best_out$selected_state)) {
    best_params$train_seed <- as.integer(train_seed + 90000L * iter + best_i)
    best_params$return_state <- TRUE
    best_out <- run_nmk_on_split(
      train_core = train_core, val_df = val_df, test_df = test_df,
      calibration_patch_cache = calibration_patch_cache,
      params = best_params)
  }
  best_out <- nmk_apply_postprocess_out(
    best_out, best_policy, tune_stage = best_stage)
  if (!is.null(best_out$diagnostics)) {
    best_out$diagnostics$tune_selected_candidate_id <- best_i
    best_out$diagnostics$tune_selected_candidate_label <-
      score_df$candidate_label[
        score_df$candidate_id == best_i &
          score_df$tune_stage == best_stage][1]
	    best_out$diagnostics$tune_selected_val_rmse <- best_rmse
	    best_out$diagnostics$tune_selected_score <- best_score
	    best_out$diagnostics$tune_selection_objective <- nmk_selection_objective
	    best_out$diagnostics$tune_selected_rf_inner_RMSE <-
	      as.numeric(best_policy$rf_inner_RMSE)
	    best_out$diagnostics$tune_selected_delta_RMSE_vs_RF <-
	      as.numeric(best_policy$delta_RMSE_vs_RF)
	    best_out$diagnostics$tune_selected_rmse_selection_score <-
	      as.numeric(best_policy$rmse_selection_score)
	    best_out$diagnostics$tune_candidate_n <- nrow(score_df)
    best_out$diagnostics$tune_unique_candidate_n <- nrow(grid)
    best_out$diagnostics$tune_stage <- best_stage
    best_out$diagnostics$tune_stage2_top_k <- nmk_v3_stage2_top_k
    best_out$diagnostics$tune_stage2_n_seeds <- nmk_v3_stage2_n_seeds
    best_out$diagnostics$tune_source <- tune_source
    best_out$diagnostics$tune_inner_folds <- if (use_cv_tuning) {
      nmk_tune_inner_folds
    } else {
      1L
    }
    best_out$diagnostics$tune_n <- tune_n
    best_out$diagnostics$tune_fold_sizes <- tune_fold_sizes
    best_out$diagnostics$tune_full_grid <- isTRUE(nmk_tune_full_grid)
    best_out$diagnostics$tune_num_workers <- 1L
    best_out$diagnostics$tune_num_workers_requested <- nmk_num_workers
    best_out$diagnostics$autotune_v4 <- isTRUE(nmk_autotune_v4)
    best_out$diagnostics$tune_n_seeds <- nmk_tune_n_seeds
    best_out$diagnostics$target_transform_requested <-
      score_df$target_transform_requested[
        score_df$candidate_id == best_i &
          score_df$tune_stage == best_stage][1]
	    best_out$diagnostics$target_transform_effective <-
	      score_df$target_transform_effective[
	        score_df$candidate_id == best_i &
	          score_df$tune_stage == best_stage][1]
	    best_out$diagnostics$noharm_policy <- nmk_noharm_policy
	    best_out$diagnostics$noharm_preselected_candidate_id <-
	      pre_noharm_eval$candidate_id
	    best_out$diagnostics$noharm_preselected_tune_stage <-
	      pre_noharm_eval$tune_stage
	    best_out$diagnostics$noharm_anchor_candidate_id <-
	      if (!is.null(noharm_anchor_eval)) noharm_anchor_eval$candidate_id else NA_integer_
	    best_out$diagnostics$noharm_anchor_tune_stage <-
	      if (!is.null(noharm_anchor_eval)) noharm_anchor_eval$tune_stage else NA_character_
	    best_out$diagnostics$noharm_anchor_candidate_label <- noharm_anchor_label
	    best_out$diagnostics$noharm_anchor_score <- noharm_anchor_score
	    best_out$diagnostics$noharm_anchor_rmse <- noharm_anchor_rmse
	    best_out$diagnostics$noharm_margin_abs <- nmk_noharm_margin_abs
	    best_out$diagnostics$noharm_margin_rel <- nmk_noharm_margin_rel
	    best_out$diagnostics$noharm_margin <- noharm_margin
	    best_out$diagnostics$noharm_delta_vs_anchor <- noharm_delta_vs_anchor
		    best_out$diagnostics$noharm_applied <- noharm_applied
		    best_out$diagnostics$noharm_restrict_ensemble <- nmk_noharm_restrict_ensemble
		    best_out$diagnostics$noharm_reason <- noharm_reason
		    best_out$diagnostics$rf_noharm_policy <- nmk_rf_noharm_policy
		    best_out$diagnostics$rf_noharm_preselected_candidate_id <-
		      pre_rf_noharm_eval$candidate_id
		    best_out$diagnostics$rf_noharm_preselected_tune_stage <-
		      pre_rf_noharm_eval$tune_stage
		    best_out$diagnostics$rf_noharm_preselected_delta_RMSE_vs_RF <-
		      rf_noharm_preselected_delta
		    best_out$diagnostics$rf_noharm_anchor_candidate_id <-
		      if (!is.null(noharm_anchor_eval)) noharm_anchor_eval$candidate_id else NA_integer_
		    best_out$diagnostics$rf_noharm_anchor_tune_stage <-
		      if (!is.null(noharm_anchor_eval)) noharm_anchor_eval$tune_stage else NA_character_
		    best_out$diagnostics$rf_noharm_anchor_candidate_label <-
		      noharm_anchor_label
		    best_out$diagnostics$rf_noharm_anchor_delta_RMSE_vs_RF <-
		      rf_noharm_anchor_delta
		    best_out$diagnostics$rf_noharm_margin_abs <- nmk_rf_noharm_margin_abs
		    best_out$diagnostics$rf_noharm_margin_rel <- nmk_rf_noharm_margin_rel
		    best_out$diagnostics$rf_noharm_margin <- rf_noharm_margin
		    best_out$diagnostics$rf_noharm_required_delta_RMSE_vs_RF <-
		      rf_noharm_required_delta
		    best_out$diagnostics$rf_noharm_failed <- rf_noharm_failed
		    best_out$diagnostics$rf_noharm_applied <- rf_noharm_applied
		    best_out$diagnostics$rf_noharm_restrict_ensemble <-
		      nmk_rf_noharm_restrict_ensemble
		    best_out$diagnostics$rf_noharm_reason <- rf_noharm_reason
		  }
  list(
    out = best_out,
    params = best_params,
    tuning = score_df,
    policy = best_policy,
    ensemble = list(
      candidates = ensemble_candidates,
      weights = ensemble_weights,
      scores = ensemble_scores,
      labels = ensemble_labels
    )
  )
}

# -----------------------------------------------------------------------------
# Metrics bookkeeping
# -----------------------------------------------------------------------------

emit_metric_rows <- function(obs, pred_map, protocol, iter, scenario) {
  rows <- list()
  for (model_label in names(pred_map)) {
    pred <- pred_map[[model_label]]
    if (is.null(pred) || length(pred) == 0L) next
    m <- wadoux_eval(obs = obs, pred = pred)
    rows[[length(rows) + 1L]] <- data.frame(
      iteration = iter,
      scenario  = scenario,
      protocol  = protocol,
      model     = model_label,
      ME        = m$ME,
      RMSE      = m$RMSE,
      r2        = m$r2,
      MEC       = m$MEC,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}

bind_rows_fill_nmk <- function(rows) {
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

nmk_variogram_objects <- function(y, coords, cutoff_frac = 0.50,
                                  n_lags = 15L) {
  if (!requireNamespace("gstat", quietly = TRUE) ||
      !requireNamespace("sp", quietly = TRUE)) {
    stop("NMK variogram diagnostics require packages gstat and sp.")
  }
  y <- as.numeric(y)
  coords <- as.matrix(coords)
  ok <- is.finite(y) & stats::complete.cases(coords)
  y <- y[ok]
  coords <- coords[ok, , drop = FALSE]
  fallback <- function(status, cutoff = NA_real_, transform = NA_character_,
                       sw_p = NA_real_, range_init = NA_real_) {
    list(
      summary = list(
        range_major = range_init,
        nugget_ratio = 0.35,
        fit_quality = 0.25,
        area = NA_real_,
        variogram_status = status,
        variogram_transform = transform,
        variogram_shapiro_p = sw_p,
        variogram_cutoff = cutoff,
        variogram_n_lags = as.integer(n_lags),
        variogram_model = NA_character_,
        variogram_nugget = NA_real_,
        variogram_partial_sill = NA_real_,
        variogram_total_sill = NA_real_
      ),
      empirical = data.frame(),
      model = data.frame(),
      fitted_line = data.frame()
    )
  }
  if (length(y) < 3L || nrow(coords) < 3L) return(fallback("too_few_points"))
  D <- as.matrix(stats::dist(coords))
  cutoff <- max(D, na.rm = TRUE) * cutoff_frac
  if (!is.finite(cutoff) || cutoff <= 0) {
    return(fallback("invalid_coordinate_distances"))
  }
  sw_p <- tryCatch(stats::shapiro.test(
    sample(y, min(length(y), 5000L)))$p.value,
    error = function(e) 1)
  transform_name <- if (is.finite(sw_p) && sw_p < 0.05) {
    "normal_score"
  } else {
    "raw"
  }
  y_vg <- if (identical(transform_name, "normal_score")) {
    qx <- (rank(y) - 0.5) / length(y)
    stats::qnorm(pmax(1e-6, pmin(1 - 1e-6, qx)))
  } else {
    y
  }
  df <- data.frame(yv = y_vg, x = coords[, 1], y2 = coords[, 2])
  sp::coordinates(df) <- ~ x + y2
  range_init <- cutoff * 0.30
  vg_emp <- tryCatch(
    gstat::variogram(yv ~ 1, data = df, cutoff = cutoff,
                     width = cutoff / n_lags, cressie = TRUE),
    error = function(e) NULL
  )
  if (is.null(vg_emp) || nrow(vg_emp) < 3L) {
    out <- fallback("empirical_variogram_failed", cutoff = cutoff,
                    transform = transform_name, sw_p = sw_p,
                    range_init = range_init)
    out$empirical <- if (is.null(vg_emp)) data.frame() else as.data.frame(vg_emp)
    return(out)
  }
  sill_tot <- stats::var(y_vg, na.rm = TRUE)
  vg_fit <- tryCatch(
    suppressWarnings(gstat::fit.variogram(
      vg_emp,
      gstat::vgm(sill_tot * 0.80, "Exp", range_init, sill_tot * 0.20),
      fit.method = 7L
    )),
    error = function(e) NULL
  )
  fitted_line <- data.frame()
  if (!is.null(vg_fit) && all(is.finite(vg_fit$range))) {
    iso_range <- max(vg_fit$range[vg_fit$range > 0], na.rm = TRUE)
    if (!is.finite(iso_range) || iso_range <= 0) iso_range <- range_init
    psill_sum <- sum(vg_fit$psill, na.rm = TRUE)
    nugget_ratio <- if (is.finite(psill_sum) && psill_sum > 1e-8) {
      pmax(0.02, pmin(0.90, vg_fit$psill[1L] / psill_sum))
    } else {
      0.35
    }
    fitted_line <- tryCatch(
      as.data.frame(gstat::variogramLine(vg_fit, maxdist = cutoff, n = 200L)),
      error = function(e) data.frame()
    )
    status <- "ok"
  } else {
    iso_range <- range_init
    nugget_ratio <- 0.35
    status <- "model_fit_failed"
  }
  area <- diff(range(coords[, 1])) * diff(range(coords[, 2]))
  nugget <- if (!is.null(vg_fit) && nrow(vg_fit) > 0L) {
    as.numeric(vg_fit$psill[1L])
  } else NA_real_
  partial_sill <- if (!is.null(vg_fit) && nrow(vg_fit) > 0L) {
    sum(vg_fit$psill[-1L], na.rm = TRUE)
  } else NA_real_
  total_sill <- if (!is.null(vg_fit) && nrow(vg_fit) > 0L) {
    sum(vg_fit$psill, na.rm = TRUE)
  } else NA_real_
  model_name <- if (!is.null(vg_fit) && nrow(vg_fit) >= 2L) {
    as.character(vg_fit$model[2L])
  } else NA_character_
  list(
    summary = list(
      range_major = iso_range,
      nugget_ratio = nugget_ratio,
      fit_quality = if (is.finite(sw_p) && sw_p < 0.05) 0.5 else 0.7,
      area = area,
      variogram_status = status,
      variogram_transform = transform_name,
      variogram_shapiro_p = sw_p,
      variogram_cutoff = cutoff,
      variogram_n_lags = as.integer(n_lags),
      variogram_model = model_name,
      variogram_nugget = nugget,
      variogram_partial_sill = partial_sill,
      variogram_total_sill = total_sill
    ),
    empirical = as.data.frame(vg_emp),
    model = if (is.null(vg_fit)) data.frame() else as.data.frame(vg_fit),
    fitted_line = fitted_line
  )
}

nmk_save_variogram_diagnostic <- function(train_df, iter, protocol, model,
                                          stage) {
  if (!isTRUE(nmk_save_variograms)) return(data.frame())
  out_dir <- file.path(results_dir, "nmk_semivariograms")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  prefix <- gsub("[^A-Za-z0-9_\\-]+", "_",
                 sprintf("iter%03d_%s_%s_%s", iter, protocol, model, stage))
  obj <- tryCatch(
    nmk_variogram_objects(
      y = train_df[[response_name_wadoux]],
      coords = as.matrix(train_df[, c("x", "y"), drop = FALSE])),
    error = function(e) {
      list(
        summary = list(
          range_major = NA_real_, nugget_ratio = NA_real_,
          fit_quality = NA_real_, area = NA_real_,
          variogram_status = paste("failed:", conditionMessage(e)),
          variogram_transform = NA_character_,
          variogram_shapiro_p = NA_real_,
          variogram_cutoff = NA_real_,
          variogram_n_lags = NA_integer_,
          variogram_model = NA_character_,
          variogram_nugget = NA_real_,
          variogram_partial_sill = NA_real_,
          variogram_total_sill = NA_real_),
        empirical = data.frame(), model = data.frame(),
        fitted_line = data.frame())
    }
  )
  empirical_path <- file.path(out_dir, paste0(prefix, "_empirical.csv"))
  model_path <- file.path(out_dir, paste0(prefix, "_model.csv"))
  line_path <- file.path(out_dir, paste0(prefix, "_fitted_line.csv"))
  plot_path <- file.path(out_dir, paste0(prefix, ".png"))
  write.csv(obj$empirical, empirical_path, row.names = FALSE)
  write.csv(obj$model, model_path, row.names = FALSE)
  write.csv(obj$fitted_line, line_path, row.names = FALSE)
  if (requireNamespace("ggplot2", quietly = TRUE) && nrow(obj$empirical) > 0L) {
    p <- ggplot2::ggplot(
      obj$empirical,
      ggplot2::aes(x = .data$dist, y = .data$gamma)
    ) +
      ggplot2::geom_point(ggplot2::aes(size = .data$np),
                          alpha = 0.75, color = "#2f5d8c") +
      ggplot2::scale_size_continuous(name = "pairs", range = c(1.5, 5)) +
      ggplot2::labs(
        title = sprintf("NMK semivariogram | iter %d | %s | %s",
                        iter, model, stage),
        subtitle = sprintf(
          "status=%s | model=%s | nugget ratio=%.2f | range=%.3g | transform=%s",
          obj$summary$variogram_status,
          ifelse(is.na(obj$summary$variogram_model), "NA",
                 obj$summary$variogram_model),
          obj$summary$nugget_ratio,
          obj$summary$range_major,
          obj$summary$variogram_transform
        ),
        x = "Lag distance",
        y = "Semivariance"
      ) +
      ggplot2::theme_minimal(base_size = 12)
    if (nrow(obj$fitted_line) > 0L) {
      p <- p + ggplot2::geom_line(
        data = obj$fitted_line,
        ggplot2::aes(x = .data$dist, y = .data$gamma),
        inherit.aes = FALSE, linewidth = 0.9, color = "#b13f2a"
      )
    }
    if (is.finite(obj$summary$variogram_total_sill)) {
      p <- p + ggplot2::geom_hline(
        yintercept = obj$summary$variogram_total_sill,
        linetype = "dashed", color = "#5f6368", linewidth = 0.4
      )
    }
    if (is.finite(obj$summary$range_major)) {
      p <- p + ggplot2::geom_vline(
        xintercept = obj$summary$range_major,
        linetype = "dotted", color = "#5f6368", linewidth = 0.4
      )
    }
    ggplot2::ggsave(plot_path, p, width = 7.5, height = 5.2, dpi = 160)
  } else {
    plot_path <- NA_character_
  }
  summary <- as.data.frame(obj$summary, stringsAsFactors = FALSE)
  summary$empirical_csv <- empirical_path
  summary$model_csv <- model_path
  summary$fitted_line_csv <- line_path
  summary$plot_png <- plot_path
  cbind(
    data.frame(iteration = iter, scenario = scenario, protocol = protocol,
               model = model, stage = stage, train_n = nrow(train_df),
               stringsAsFactors = FALSE),
    summary
  )
}

scalar_diag_row_nmk <- function(diag, iter, protocol, model, stage,
                                train_n, val_n, test_n,
                                refit_full_calibration = FALSE,
                                refit_monitor_overlap = FALSE) {
  row <- data.frame(
    iteration = iter,
    scenario = scenario,
    protocol = protocol,
    model = model,
    stage = stage,
    train_n = train_n,
    val_n = val_n,
    test_n = test_n,
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

paired_delta_rows_nmk <- function(metric_df) {
  if (is.null(metric_df) || nrow(metric_df) == 0L) return(data.frame())
  key_cols <- c("iteration", "scenario", "protocol")
  pairs <- list(
    c("NeuroMaternKrige_400", "RF_400"),
    c("NeuroMaternKrige_500", "RF_500"),
    c("NeuroMaternKrige_500", "NeuroMaternKrige_400"),
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
  bind_rows_fill_nmk(rows)
}

row_id_nmk <- function(df) {
  if ("..wadoux_row_id" %in% names(df)) {
    df$..wadoux_row_id
  } else {
    seq_len(nrow(df))
  }
}

coord_key_nmk <- function(df) {
  if (!all(c("x", "y") %in% names(df))) {
    return(rep(NA_character_, nrow(df)))
  }
  paste0(round(as.numeric(df$x), 6), "|", round(as.numeric(df$y), 6))
}

sample_design_test_nmk <- function(calibration_full, size) {
  pool <- common_data$s_df
  keep <- rep(TRUE, nrow(pool))

  if ("..wadoux_row_id" %in% names(pool) &&
      "..wadoux_row_id" %in% names(calibration_full)) {
    keep <- keep & !(pool$..wadoux_row_id %in% calibration_full$..wadoux_row_id)
  }

  pool_xy <- coord_key_nmk(pool)
  cal_xy <- coord_key_nmk(calibration_full)
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
  out <- sample_simple_random_rows_wadoux(eligible, size)
  out$..patch_cache_row_id <- rep(NA_integer_, nrow(out))
  out
}

prediction_audit_designbased_nmk <- function(test_df, pred_map, iter,
                                             diag_400 = NULL,
                                             diag_500 = NULL) {
  if (!isTRUE(nmk_save_prediction_audit)) return(data.frame())
  obs <- test_df[[response_name_wadoux]]
  out <- data.frame(
    iteration = iter,
    scenario = scenario,
    protocol = "DesignBased",
    split_role = "design_test",
    wadoux_row_id = row_id_nmk(test_df),
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
	      "tune_source", "tune_inner_folds", "tune_stage",
      "tune_stage2_top_k", "tune_stage2_n_seeds",
      "target_transform_requested", "target_transform_effective",
      "target_transform", "K_neighbors", "kriging_hidden",
      "krig_drop_p_train", "init_mix_mean", "tab_dropout",
      "patch_dropout", "lambda_cov", "lambda_smooth", "lambda_anchor",
      "base_loss_weight", "beta_init_logit", "dist_scale_eval",
      "tab_input_dropout", "tab_quantile_basis", "tab_qbasis_knots",
      "tab_qbasis_dropout", "tab_crossnet", "tab_cross_layers",
      "tab_cross_dropout", "tab_skip", "tab_skip_hidden",
      "tab_skip_dropout", "tab_skip_init", "n_trees", "tree_depth",
      "n_seeds", "refit_fixed_epochs", "autotune_v3", "autotune_v4",
      "refit_full_calibration", "refit_warm_start", "refit_mode",
      "refit_monitor_role", "warm_start_loaded", "warm_start_error",
      "refit_status", "refit_ensemble_top_k_requested",
      "refit_ensemble_applied", "refit_ensemble_size",
      "refit_ensemble_weighting", "refit_ensemble_temperature",
	      "refit_ensemble_members", "refit_ensemble_weights",
	      "refit_ensemble_status",
	      "noharm_policy", "noharm_preselected_candidate_id",
	      "noharm_preselected_tune_stage", "noharm_anchor_candidate_id",
	      "noharm_anchor_tune_stage", "noharm_anchor_candidate_label",
		      "noharm_anchor_score", "noharm_anchor_rmse",
		      "noharm_margin_abs", "noharm_margin_rel", "noharm_margin",
		      "noharm_delta_vs_anchor", "noharm_applied",
		      "noharm_restrict_ensemble", "noharm_reason",
		      "rf_noharm_policy",
		      "rf_noharm_preselected_candidate_id",
		      "rf_noharm_preselected_tune_stage",
		      "rf_noharm_preselected_delta_RMSE_vs_RF",
		      "rf_noharm_anchor_candidate_id",
		      "rf_noharm_anchor_tune_stage",
		      "rf_noharm_anchor_candidate_label",
		      "rf_noharm_anchor_delta_RMSE_vs_RF",
		      "rf_noharm_margin_abs", "rf_noharm_margin_rel",
		      "rf_noharm_margin", "rf_noharm_required_delta_RMSE_vs_RF",
		      "rf_noharm_failed", "rf_noharm_applied",
		      "rf_noharm_restrict_ensemble", "rf_noharm_reason",
		      "post_alpha", "post_calibration_requested",
      "post_calibration_policy", "post_bias_correction",
      "post_affine_intercept", "post_affine_slope",
      "post_crossfit_calibration", "post_selection_score",
      "post_val_rmse", "post_val_me", "post_rmse_sd",
	      "post_fold_RMSE", "post_quantile_RMSE_mean",
	      "post_quantile_RMSE_sd", "post_quantile_RMSE_excess",
	      "post_quantile_RMSE", "post_selection_objective",
	      "post_rf_inner_ME", "post_rf_inner_RMSE",
	      "post_rf_inner_r2", "post_rf_inner_MEC",
	      "post_rf_inner_RMSE_sd", "post_rf_inner_fold_RMSE",
	      "post_delta_RMSE_vs_RF", "post_rmse_selection_score",
	      "post_quantile_bins",
      "post_quantile_excess_weight", "post_quantile_sd_weight",
      "post_alpha_grid", "post_calibration_grid"
    )
    for (f in fields) {
      val <- diag[[f]]
      if (is.null(val) || length(val) != 1L) val <- NA
      out[[paste(prefix, f, sep = "_")]] <<- val
    }
  }
  add_diag("NeuroMaternKrige_400", diag_400)
  add_diag("NeuroMaternKrige_500", diag_500)
  out
}

split_audit_designbased_nmk <- function(calibration_full, inner, test_df, iter) {
  if (!isTRUE(nmk_save_prediction_audit)) return(data.frame())
  mk <- function(df, role) {
    data.frame(
      iteration = iter,
      scenario = scenario,
      protocol = "DesignBased",
      split_role = role,
      wadoux_row_id = row_id_nmk(df),
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
# FINAL do-no-harm gate against the LITERAL Wadoux default RF.
#
# The inner gate in tune_nmk_on_split only ever reverts to a conservative NMK
# profile, and decides on an inner RF that is NOT bit-identical to the scored
# RF. This final gate closes that gap: on the FULL calibration set it computes
# leakage-free OOF RMSE for (a) the emitted NMK prediction surrogate and (b)
# the LITERAL Wadoux RF (fit_rf_default_wadoux / predict_rf_default_wadoux),
# then selects a continuous shrinkage lambda in [0,1] (grid including 0 = pure
# literal RF) that minimises OOF RMSE of  lambda*NMK + (1-lambda)*RF_literal.
# The emitted test prediction is the same convex blend evaluated at the
# already-fit, bit-identical scored RF. lambda = 0 GUARANTEES the emitted
# prediction is the literal Wadoux RF, so NMK can never do worse than RF on
# the OOF criterion (ties at worst).
#
# Controlled by NMK_RF_FINAL_GATE (default = nmk_rf_noharm_policy). The lambda
# grid is NMK_RF_FINAL_LAMBDA_GRID (default 0,0.25,0.5,0.75,1). OOF folds are
# NMK_RF_FINAL_OOF_FOLDS (default 5). The required OOF improvement margin to
# move lambda above 0 reuses the rf_noharm margins.
nmk_rf_final_gate_enabled <- parse_bool_env_nmk(
  "NMK_RF_FINAL_GATE", nmk_rf_noharm_policy)
nmk_rf_final_lambda_grid <- parse_num_list_env_nmk(
  "NMK_RF_FINAL_LAMBDA_GRID", c(0, 0.25, 0.50, 0.75, 1.00))
nmk_rf_final_lambda_grid <- sort(unique(pmax(0, pmin(1, nmk_rf_final_lambda_grid))))
if (length(nmk_rf_final_lambda_grid) == 0L) {
  nmk_rf_final_lambda_grid <- c(0, 0.25, 0.50, 0.75, 1.00)
}
if (!any(abs(nmk_rf_final_lambda_grid - 0) < 1e-9)) {
  nmk_rf_final_lambda_grid <- sort(unique(c(0, nmk_rf_final_lambda_grid)))
}
nmk_rf_final_oof_folds <- as.integer(parse_num_env_nmk("NMK_RF_FINAL_OOF_FOLDS", 5L))
if (!is.finite(nmk_rf_final_oof_folds) || nmk_rf_final_oof_folds < 2L) {
  nmk_rf_final_oof_folds <- 5L
}

# Select the do-no-harm shrinkage lambda toward the literal RF on OOF data.
# nmk_oof_fn(train_df, score_df) must return NMK predictions for the rows of
# score_df, trained only on train_df (leakage-free). Returns a list with the
# chosen lambda and the OOF RMSE table.
nmk_select_rf_shrinkage_lambda <- function(calibration_full, nmk_oof_fn) {
  obs <- as.numeric(calibration_full[[response_name_wadoux]])
  n <- length(obs)
  # #5: spatial-block OOF folds when the calibration is clustered.
  fold_id <- NULL
  nmk_cal_coords <- tryCatch(
    as.matrix(calibration_full[, c("x", "y"), drop = FALSE]), error = function(e) NULL)
  if (!is.null(nmk_cal_coords) && wadoux_is_clustered(nmk_cal_coords)) {
    fold_id <- wadoux_spatial_block_fold_ids(
      nmk_cal_coords, nmk_rf_final_oof_folds, seed = train_seed + 73000L)
  }
  if (is.null(fold_id)) fold_id <- make_regression_folds_nmk(
    obs, nmk_rf_final_oof_folds, seed = train_seed + 73000L)
  pred_rf <- rep(NA_real_, n)
  pred_nmk <- rep(NA_real_, n)
  for (fold in sort(unique(fold_id))) {
    score_idx <- which(fold_id == fold)
    fit_idx <- which(fold_id != fold)
    if (length(fit_idx) < 5L || length(score_idx) < 1L) next
    rf_f <- tryCatch(
      fit_rf_default_wadoux(calibration_full[fit_idx, , drop = FALSE], form_rf),
      error = function(e) NULL)
    if (!is.null(rf_f)) {
      pred_rf[score_idx] <- predict_rf_default_wadoux(
        rf_f, calibration_full[score_idx, , drop = FALSE])
    }
    nmk_f <- tryCatch(
      nmk_oof_fn(calibration_full[fit_idx, , drop = FALSE],
                 calibration_full[score_idx, , drop = FALSE]),
      error = function(e) NULL)
    if (!is.null(nmk_f) && length(nmk_f) == length(score_idx)) {
      pred_nmk[score_idx] <- as.numeric(nmk_f)
    }
  }
  ok <- is.finite(obs) & is.finite(pred_rf) & is.finite(pred_nmk)
  rf_oof_rmse <- if (any(ok)) sqrt(mean((pred_rf[ok] - obs[ok])^2)) else Inf
  rows <- list()
  best_lambda <- 0
  best_rmse <- Inf
  for (lam in nmk_rf_final_lambda_grid) {
    blend <- lam * pred_nmk + (1 - lam) * pred_rf
    okb <- is.finite(obs) & is.finite(blend)
    rmse <- if (any(okb)) sqrt(mean((blend[okb] - obs[okb])^2)) else Inf
    rows[[length(rows) + 1L]] <- data.frame(
      lambda = lam, oof_rmse = rmse, stringsAsFactors = FALSE)
    # Strictly-better rule: lambda only rises above the running best when it
    # beats it. lambda = 0 (pure RF) is evaluated first, so ties keep RF.
    if (is.finite(rmse) && rmse < best_rmse - 1e-12) {
      best_rmse <- rmse
      best_lambda <- lam
    }
  }
  # Require the chosen lambda to beat pure-RF OOF by the rf_noharm margin;
  # otherwise fall back to lambda = 0 (literal RF) — guaranteed do-no-harm.
  rf_scale <- if (is.finite(rf_oof_rmse)) rf_oof_rmse else 0
  required_margin <- nmk_rf_noharm_margin_abs +
    nmk_rf_noharm_margin_rel * abs(rf_scale)
  if (best_lambda > 0 &&
      (!is.finite(rf_oof_rmse) ||
       rf_oof_rmse - best_rmse < required_margin)) {
    best_lambda <- 0
    best_rmse <- rf_oof_rmse
  }
  list(
    lambda = best_lambda,
    nmk_oof_rmse = {
      okn <- is.finite(obs) & is.finite(pred_nmk)
      if (any(okn)) sqrt(mean((pred_nmk[okn] - obs[okn])^2)) else Inf
    },
    rf_oof_rmse = rf_oof_rmse,
    blend_oof_rmse = best_rmse,
    required_margin = required_margin,
    grid = do.call(rbind, rows)
  )
}

# -----------------------------------------------------------------------------
# Protocol evaluators
#
# Each returns a data.frame with (iteration, scenario, protocol, model,
# ME, RMSE, r2, MEC) for all requested models, plus a list of diagnostics.
# -----------------------------------------------------------------------------

# ---- DesignBased: independent SRS validation sample from the full domain ----
eval_designbased <- function(valuetable, iter, calibration_patch_cache) {
  val_srs <- sample_design_test_nmk(valuetable, sample_size)
  obs     <- val_srs[[response_name_wadoux]]
  inner <- make_inner_val_split_nmk(valuetable, val_frac = val_frac,
                                    seed = train_seed + iter)
  calibration_full <- valuetable
  test_df <- val_srs

  pred_map <- list()
  diag_rows <- list()
  tune_rows <- list()
  vario_rows <- list()

  if ("NeuroMaternKrige" %in% models && isTRUE(nmk_save_variograms)) {
    vario_rows[[length(vario_rows) + 1L]] <- nmk_save_variogram_diagnostic(
      inner$train, iter, "DesignBased", "NeuroMaternKrige_400",
      "inner_selection")
    vario_rows[[length(vario_rows) + 1L]] <- nmk_save_variogram_diagnostic(
      calibration_full, iter, "DesignBased", "NeuroMaternKrige_500",
      "full_calibration_refit")
  }

  if ("RF" %in% models) {
    rf_400 <- fit_rf_default_wadoux(inner$train, form_rf)
    pred_map$RF_400 <- predict_rf_default_wadoux(rf_400, val_srs)

    rf_500 <- fit_rf_default_wadoux(calibration_full, form_rf)
    pred_map$RF_500 <- predict_rf_default_wadoux(rf_500, val_srs)
  }

  if ("NeuroMaternKrige" %in% models) {
   # do-no-harm: if NMK tuning/fit fails entirely (e.g. clustered coords ->
   # singular neural-kriging fit -> "no candidate completed"), revert
   # NeuroMaternKrige to the literal Wadoux RF instead of crashing the run.
   nmk_failed_revert_rf <- FALSE
   tryCatch({
    selected_params <- nmk_params
    selected_policy <- NULL
    selected_tune_stage <- NA_character_
    if (isTRUE(nmk_tune)) {
      tuned <- tune_nmk_on_split(
        train_core = inner$train, val_df = inner$val, test_df = test_df,
        calibration_patch_cache = calibration_patch_cache,
        iter = iter, protocol = "DesignBased",
        model_label = "NeuroMaternKrige_400"
      )
      out_400 <- tuned$out
      selected_params <- tuned$params
      selected_policy <- tuned$policy
      if (!is.null(out_400$diagnostics$tune_stage)) {
        selected_tune_stage <- out_400$diagnostics$tune_stage
      }
      tune_rows[[length(tune_rows) + 1L]] <- tuned$tuning
    } else {
      out_400 <- run_nmk_on_split(
        train_core = inner$train, val_df = inner$val, test_df = test_df,
        calibration_patch_cache = calibration_patch_cache
      )
    }
    pred_map$NeuroMaternKrige_400 <- out_400$pred_test
    diag_rows[[length(diag_rows) + 1L]] <- scalar_diag_row_nmk(
      out_400$diagnostics, iter, "DesignBased", "NeuroMaternKrige_400",
      "inner_selection", nrow(inner$train), nrow(inner$val), nrow(test_df)
    )

    if (isTRUE(nmk_refit_full_calibration)) {
      refit_monitor <- nmk_refit_monitor_df(calibration_full)
      ensemble_info <- if (exists("tuned", inherits = FALSE) &&
                           !is.null(tuned$ensemble) &&
                           length(tuned$ensemble$candidates) > 0L) {
        tuned$ensemble
      } else {
        list(
          candidates = list(list(
            candidate_id = 1L,
            tune_stage = selected_tune_stage,
            selection_score = 0,
            rmse = 0,
            params = selected_params,
            policy = selected_policy
          )),
          weights = 1,
          scores = 0,
          labels = "selected"
        )
      }
      member_candidates <- ensemble_info$candidates
      member_labels <- ensemble_info$labels
      if (length(member_labels) != length(member_candidates)) {
        member_labels <- vapply(member_candidates, function(x) {
          paste0("candidate_", x$candidate_id)
        }, character(1))
      }
      requested_weights <- as.numeric(ensemble_info$weights)
      if (length(requested_weights) != length(member_candidates) ||
          !all(is.finite(requested_weights)) ||
          sum(requested_weights) <= 0) {
        requested_weights <- rep(1 / length(member_candidates),
                                 length(member_candidates))
      }
      requested_weights <- requested_weights / sum(requested_weights)

      selected_id <- out_400$diagnostics$tune_selected_candidate_id
      selected_stage <- out_400$diagnostics$tune_stage
      member_results <- vector("list", length(member_candidates))
      for (m_i in seq_along(member_candidates)) {
        cand_i <- member_candidates[[m_i]]
        use_selected_inner <- !is.null(selected_id) &&
          !is.na(selected_id) &&
          identical(as.character(cand_i$candidate_id), as.character(selected_id)) &&
          identical(as.character(cand_i$tune_stage), as.character(selected_stage))
        member_results[[m_i]] <- nmk_refit_candidate_full(
          candidate_eval = cand_i,
          candidate_label = member_labels[[m_i]],
          selected_inner_out = if (use_selected_inner) out_400 else NULL,
          train_core = inner$train,
          val_df = inner$val,
          test_df = test_df,
          calibration_full = calibration_full,
          refit_monitor = refit_monitor,
          calibration_patch_cache = calibration_patch_cache,
          iter = iter,
          member_index = m_i
        )
      }

      member_status <- vapply(member_results, function(x) {
        paste0(x$candidate_label, ":", x$status)
      }, character(1))
      ok_idx <- which(vapply(member_results, function(x) {
        identical(x$status, "ok") &&
          !is.null(x$pred_test) &&
          all(is.finite(x$pred_test))
      }, logical(1)))

      ensemble_applied <- FALSE
      ensemble_weights_ok <- numeric()
      ensemble_members_ok <- character()
      if (length(ok_idx) > 0L) {
        successful <- member_results[ok_idx]
        ensemble_weights_ok <- requested_weights[ok_idx]
        ensemble_weights_ok <- ensemble_weights_ok / sum(ensemble_weights_ok)
        ensemble_members_ok <- vapply(successful, `[[`, character(1),
                                      "candidate_label")
        pred_list <- lapply(successful, `[[`, "pred_test")
        ensemble_pred <- Reduce(`+`, Map(`*`, pred_list, ensemble_weights_ok))
        selected_ok <- ok_idx[which(vapply(member_results[ok_idx], function(x) {
          identical(as.character(x$candidate_id), as.character(selected_id)) &&
            identical(as.character(x$tune_stage), as.character(selected_stage))
        }, logical(1)))]
        base_result <- if (length(selected_ok) > 0L) {
          member_results[[selected_ok[[1L]]]]
        } else {
          successful[[1L]]
        }
        out_500 <- base_result$out
        out_500$pred_test <- ensemble_pred
        ensemble_applied <- length(successful) > 1L
      } else {
        out_500 <- out_400
        out_500$pred_test <- out_400$pred_test
      }

      tune_meta <- c(
        "tune_selected_candidate_id",
        "tune_selected_candidate_label",
	        "tune_selected_val_rmse",
	        "tune_selected_score",
	        "tune_selection_objective",
	        "tune_selected_rf_inner_RMSE",
	        "tune_selected_delta_RMSE_vs_RF",
	        "tune_selected_rmse_selection_score",
	        "tune_candidate_n",
        "tune_unique_candidate_n",
        "tune_stage",
        "tune_stage2_top_k",
        "tune_stage2_n_seeds",
        "tune_source",
        "tune_inner_folds",
        "tune_n",
        "tune_fold_sizes",
        "tune_full_grid",
        "tune_num_workers",
        "tune_num_workers_requested",
        "tune_n_seeds",
        "target_transform_requested",
        "target_transform_effective",
        "tab_input_dropout",
        "tab_quantile_basis",
        "tab_qbasis_knots",
        "tab_qbasis_dropout",
        "tab_crossnet",
        "tab_cross_layers",
        "tab_cross_dropout",
        "tab_skip",
        "tab_skip_hidden",
        "tab_skip_dropout",
        "tab_skip_init",
        "n_trees",
        "tree_depth",
	        "autotune_v3",
	        "autotune_v4",
	        "noharm_policy",
	        "noharm_preselected_candidate_id",
	        "noharm_preselected_tune_stage",
	        "noharm_anchor_candidate_id",
	        "noharm_anchor_tune_stage",
	        "noharm_anchor_candidate_label",
	        "noharm_anchor_score",
	        "noharm_anchor_rmse",
		        "noharm_margin_abs",
		        "noharm_margin_rel",
		        "noharm_margin",
		        "noharm_delta_vs_anchor",
		        "noharm_applied",
		        "noharm_restrict_ensemble",
		        "noharm_reason",
		        "rf_noharm_policy",
		        "rf_noharm_preselected_candidate_id",
		        "rf_noharm_preselected_tune_stage",
		        "rf_noharm_preselected_delta_RMSE_vs_RF",
		        "rf_noharm_anchor_candidate_id",
		        "rf_noharm_anchor_tune_stage",
		        "rf_noharm_anchor_candidate_label",
		        "rf_noharm_anchor_delta_RMSE_vs_RF",
		        "rf_noharm_margin_abs",
		        "rf_noharm_margin_rel",
		        "rf_noharm_margin",
		        "rf_noharm_required_delta_RMSE_vs_RF",
		        "rf_noharm_failed",
		        "rf_noharm_applied",
		        "rf_noharm_restrict_ensemble",
		        "rf_noharm_reason",
		        "post_alpha",
        "post_calibration_requested",
        "post_calibration_policy",
        "post_bias_correction",
        "post_affine_intercept",
        "post_affine_slope",
        "post_crossfit_calibration",
        "post_selection_score",
        "post_val_rmse",
        "post_val_me",
        "post_rmse_sd",
        "post_fold_RMSE",
        "post_quantile_RMSE_mean",
        "post_quantile_RMSE_sd",
	        "post_quantile_RMSE_excess",
	        "post_quantile_RMSE",
	        "post_selection_objective",
	        "post_rf_inner_ME",
	        "post_rf_inner_RMSE",
	        "post_rf_inner_r2",
	        "post_rf_inner_MEC",
	        "post_rf_inner_RMSE_sd",
	        "post_rf_inner_fold_RMSE",
	        "post_delta_RMSE_vs_RF",
	        "post_rmse_selection_score",
	        "post_quantile_bins",
        "post_quantile_excess_weight",
        "post_quantile_sd_weight",
        "post_alpha_grid",
        "post_calibration_grid"
      )
      for (nm in tune_meta) {
        if (!is.null(out_400$diagnostics[[nm]])) {
          out_500$diagnostics[[nm]] <- out_400$diagnostics[[nm]]
        }
      }
      out_500$diagnostics$refit_full_calibration <- TRUE
      out_500$diagnostics$refit_monitor_overlap <- TRUE
      out_500$diagnostics$refit_warm_start <- any(vapply(member_results, function(x) {
        isTRUE(x$warm_start_applied)
      }, logical(1)))
      out_500$diagnostics$refit_mode <- if (isTRUE(ensemble_applied)) {
        "topk_candidate_ensemble"
      } else if (length(ok_idx) == 0L) {
        "inner_fallback"
      } else if (isTRUE(out_500$diagnostics$warm_start_loaded)) {
        "warm_start_continue"
      } else {
        "cold_start_fixed_epoch"
      }
      out_500$diagnostics$refit_monitor_role <- "metric_only_fixed_epoch"
      out_500$diagnostics$refit_train_n <- nrow(calibration_full)
      out_500$diagnostics$refit_monitor_n <- nrow(refit_monitor)
      out_500$diagnostics$refit_status <- if (length(ok_idx) == 0L) {
        "failed_inner_fallback"
      } else if (length(ok_idx) == length(member_results)) {
        "ok"
      } else {
        "partial"
      }
      out_500$diagnostics$refit_ensemble_top_k_requested <- nmk_ensemble_top_k
      out_500$diagnostics$refit_ensemble_applied <- isTRUE(ensemble_applied)
      out_500$diagnostics$refit_ensemble_size <- if (length(ok_idx) > 0L) {
        length(ok_idx)
      } else {
        0L
      }
      out_500$diagnostics$refit_ensemble_weighting <- nmk_ensemble_weighting
      out_500$diagnostics$refit_ensemble_temperature <- nmk_ensemble_temperature
      out_500$diagnostics$refit_ensemble_members <- if (length(ensemble_members_ok) > 0L) {
        paste(ensemble_members_ok, collapse = "|")
      } else {
        NA_character_
      }
      out_500$diagnostics$refit_ensemble_weights <- if (length(ensemble_weights_ok) > 0L) {
        paste(formatC(ensemble_weights_ok, digits = 6L, format = "fg"),
              collapse = "|")
      } else {
        NA_character_
      }
      out_500$diagnostics$refit_ensemble_status <- paste(member_status, collapse = "|")
      # FINAL do-no-harm gate vs the LITERAL Wadoux RF. The scored RF_500 above
      # is fit_rf_default_wadoux(calibration_full); we reuse the SAME literal
      # RF prediction here so the gate's RF leg is bit-identical to the scored
      # RF. lambda is chosen on leakage-free OOF; lambda = 0 emits literal RF.
      nmk_pred_test_pre_gate <- out_500$pred_test
      rf_final_lambda <- 1
      rf_final_gate_info <- NULL
      if (isTRUE(nmk_rf_final_gate_enabled) && ("RF" %in% models) &&
          !is.null(pred_map$RF_500) &&
          length(pred_map$RF_500) == length(out_500$pred_test)) {
        # OOF NMK surrogate: a single-seed NMK fit (fast) trained leakage-free
        # on each OOF fold of the full calibration set, scored on the held-out
        # fold. This estimates how the emitted NMK generalises relative to the
        # literal RF on the same folds.
        nmk_oof_fn <- function(tr_df, sc_df) {
          p_in <- selected_params
          p_in$n_seeds <- 1L
          inner_split <- make_inner_val_split_nmk(
            tr_df, val_frac = val_frac, seed = train_seed + 74000L + iter)
          o <- run_nmk_on_split(
            train_core = inner_split$train, val_df = inner_split$val,
            test_df = sc_df,
            calibration_patch_cache = calibration_patch_cache,
            params = p_in)
          nmk_apply_postprocess_values(
            o$pred_test, o$pred_test_base, selected_policy)
        }
        rf_final_gate_info <- tryCatch(
          nmk_select_rf_shrinkage_lambda(calibration_full, nmk_oof_fn),
          error = function(e) NULL)
        if (!is.null(rf_final_gate_info) &&
            is.finite(rf_final_gate_info$lambda)) {
          rf_final_lambda <- rf_final_gate_info$lambda
        } else {
          # If the gate could not run, be conservative: emit literal RF.
          rf_final_lambda <- 0
        }
        out_500$pred_test <- rf_final_lambda * nmk_pred_test_pre_gate +
          (1 - rf_final_lambda) * as.numeric(pred_map$RF_500)
        out_500$diagnostics$rf_final_gate_enabled <- TRUE
        out_500$diagnostics$rf_final_lambda <- rf_final_lambda
        out_500$diagnostics$rf_final_reverted_to_literal_rf <-
          isTRUE(abs(rf_final_lambda) < 1e-9)
        if (!is.null(rf_final_gate_info)) {
          out_500$diagnostics$rf_final_nmk_oof_RMSE <- rf_final_gate_info$nmk_oof_rmse
          out_500$diagnostics$rf_final_rf_oof_RMSE <- rf_final_gate_info$rf_oof_rmse
          out_500$diagnostics$rf_final_blend_oof_RMSE <- rf_final_gate_info$blend_oof_rmse
          out_500$diagnostics$rf_final_required_margin <- rf_final_gate_info$required_margin
        }
      } else {
        out_500$diagnostics$rf_final_gate_enabled <- FALSE
        out_500$diagnostics$rf_final_lambda <- NA_real_
        out_500$diagnostics$rf_final_reverted_to_literal_rf <- FALSE
      }
      pred_map$NeuroMaternKrige_500 <- out_500$pred_test
      diag_rows[[length(diag_rows) + 1L]] <- scalar_diag_row_nmk(
        out_500$diagnostics, iter, "DesignBased", "NeuroMaternKrige_500",
        "full_calibration_refit", nrow(calibration_full), nrow(refit_monitor),
        nrow(test_df), refit_full_calibration = TRUE,
        refit_monitor_overlap = TRUE
      )
    }
   }, error = function(e) {
     cat(sprintf("[NMK] do-no-harm: tuning/fit failed (%s); reverting NeuroMaternKrige to literal RF.\n",
                 conditionMessage(e)))
     nmk_failed_revert_rf <<- TRUE
   })
   if (isTRUE(nmk_failed_revert_rf)) {
     pred_map$NeuroMaternKrige_400 <- pred_map$RF_400
     pred_map$NeuroMaternKrige_500 <- pred_map$RF_500
   }
  }

  metrics <- emit_metric_rows(obs, pred_map, "DesignBased", iter, scenario)
  diag_400 <- if (exists("out_400", inherits = FALSE)) out_400$diagnostics else NULL
  diag_500 <- if (exists("out_500", inherits = FALSE)) out_500$diagnostics else NULL
  audit_rows <- prediction_audit_designbased_nmk(
    test_df, pred_map, iter, diag_400 = diag_400, diag_500 = diag_500)
  split_rows <- split_audit_designbased_nmk(
    calibration_full, inner, test_df, iter)
  list(
    metrics = metrics,
    diagnostics = bind_rows_fill_nmk(diag_rows),
    paired_deltas = paired_delta_rows_nmk(metrics),
    tuning = bind_rows_fill_nmk(tune_rows),
    variograms = bind_rows_fill_nmk(vario_rows),
    prediction_audit = audit_rows,
    split_audit = split_rows
  )
}

# ---- Population: predict every cell of the full domain ----------------------
eval_population <- function(valuetable, iter, calibration_patch_cache) {
  obs <- common_data$s_df[[response_name_wadoux]]

  pred_map <- list()
  diag_map <- list()

  if ("RF" %in% models) {
    rf <- fit_rf_default_wadoux(valuetable, form_rf)
    pred_map$RF <- predict_rf_default_wadoux(rf, common_data$s_df)
  }

  if ("NeuroMaternKrige" %in% models) {
    inner <- make_inner_val_split_nmk(valuetable, val_frac = val_frac,
                                      seed = train_seed + iter)
    test_df <- common_data$s_df
    test_df$..patch_cache_row_id <- rep(NA_integer_, nrow(test_df))
    out <- run_nmk_on_split(
      train_core = inner$train, val_df = inner$val, test_df = test_df,
      calibration_patch_cache = calibration_patch_cache
    )
    pred_map$NeuroMaternKrige <- out$pred_test
    diag_map$NeuroMaternKrige <- out$diagnostics
  }

  list(
    metrics = emit_metric_rows(obs, pred_map, "Population", iter, scenario),
    diagnostics = diag_map
  )
}

# ---- Random K-fold CV over the calibration sample --------------------------
eval_randomkfold <- function(valuetable, iter, calibration_patch_cache) {
  flds <- caret::createFolds(valuetable[[response_name_wadoux]],
                             k = random_k, list = TRUE, returnTrain = FALSE)
  preds_rf <- rep(NA_real_, nrow(valuetable))
  preds_nmk <- rep(NA_real_, nrow(valuetable))
  diag_list <- list()

  for (j in seq_along(flds)) {
    id <- flds[[j]]
    training_data <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]

    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(training_data, form_rf)
      preds_rf[id] <- predict_rf_default_wadoux(rf, validation_data)
    }

    if ("NeuroMaternKrige" %in% models) {
      inner <- make_inner_val_split_nmk(
        training_data, val_frac = val_frac,
        seed = train_seed + iter * 1000L + j)
      out <- run_nmk_on_split(
        train_core = inner$train, val_df = inner$val,
        test_df    = validation_data,
        calibration_patch_cache = calibration_patch_cache
      )
      preds_nmk[id] <- out$pred_test
      diag_list[[length(diag_list) + 1L]] <- out$diagnostics
    }
  }

  pred_map <- list()
  if ("RF" %in% models) pred_map$RF <- preds_rf
  if ("NeuroMaternKrige" %in% models) pred_map$NeuroMaternKrige <- preds_nmk

  list(
    metrics = emit_metric_rows(valuetable[[response_name_wadoux]], pred_map,
                               "RandomKFold", iter, scenario),
    diagnostics = list(NeuroMaternKrige = diag_list)
  )
}

# ---- Spatial K-fold CV: hierarchical clustering on (x, y) ------------------
eval_spatialkfold <- function(valuetable, iter, calibration_patch_cache) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  fold_ids <- unique(spatial_folds)
  preds_rf  <- rep(NA_real_, nrow(valuetable))
  preds_nmk <- rep(NA_real_, nrow(valuetable))
  diag_list <- list()

  for (j in seq_along(fold_ids)) {
    id <- which(spatial_folds == fold_ids[j])
    training_data <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]
    if (nrow(training_data) < 10L) next

    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(training_data, form_rf)
      preds_rf[id] <- predict_rf_default_wadoux(rf, validation_data)
    }

    if ("NeuroMaternKrige" %in% models) {
      inner <- make_inner_val_split_nmk(
        training_data, val_frac = val_frac,
        seed = train_seed + iter * 1000L + fold_ids[j])
      out <- run_nmk_on_split(
        train_core = inner$train, val_df = inner$val,
        test_df    = validation_data,
        calibration_patch_cache = calibration_patch_cache
      )
      preds_nmk[id] <- out$pred_test
      diag_list[[length(diag_list) + 1L]] <- out$diagnostics
    }
  }

  pred_map <- list()
  if ("RF" %in% models) pred_map$RF <- preds_rf
  if ("NeuroMaternKrige" %in% models) pred_map$NeuroMaternKrige <- preds_nmk

  list(
    metrics = emit_metric_rows(valuetable[[response_name_wadoux]], pred_map,
                               "SpatialKFold", iter, scenario),
    diagnostics = list(NeuroMaternKrige = diag_list)
  )
}

# ---- Buffered Leave-One-Out CV ---------------------------------------------
#
# For each focal test pixel we exclude a val_dist_km radius around it from the
# training set, refit, predict the focal value. Mirrors the official Wadoux
# BLOOCV with a performance guard: NMK retrains once per focal, which is
# extremely expensive — so by default we use only a fraction of the pixels
# unless NMK_BLOO_NMK_PIXELS is provided. RF still evaluates the full BLOOCV
# as in the paper.
eval_bloocv <- function(valuetable, iter, calibration_patch_cache) {
  pred_rows <- list()
  a <- 0L
  nb_iteration <- bloo_groups * bloo_pixels
  nmk_cap_raw <- Sys.getenv("NMK_BLOO_NMK_PIXELS", unset = "")
  nmk_cap <- if (nzchar(nmk_cap_raw)) {
    suppressWarnings(as.integer(nmk_cap_raw))
  } else {
    as.integer(min(nb_iteration, bloo_groups * 10L))   # 10 per group by default
  }
  if (!is.finite(nmk_cap) || nmk_cap < 1L) nmk_cap <- nb_iteration

  # Precompute a random focal ordering so NMK_cap selects the first nmk_cap
  # valid focals deterministically w.r.t. train_seed + iter.
  set.seed(train_seed + iter)
  ri <- val_dist_km * 1000
  diag_list <- list()
  nmk_used <- 0L

  for (j in seq_len(nb_iteration)) {
    point_in_range <- FALSE
    id_focal <- NA_integer_
    while (!isTRUE(point_in_range)) {
      id_focal <- sample.int(nrow(valuetable), size = 1L)
      focal_point <- valuetable[id_focal, , drop = FALSE]
      training_tmp <- valuetable[-id_focal, , drop = FALSE]
      training_tmp <- exclude_by_radius_wadoux(training_tmp, focal_point, ri)
      if (nrow(training_tmp) < 10L) next
      if (!point_within_predictor_range_wadoux(training_tmp, focal_point, predList)) next
      point_in_range <- TRUE
    }
    focal_point <- valuetable[id_focal, , drop = FALSE]
    training_full <- valuetable[-id_focal, , drop = FALSE]
    training_full <- exclude_by_radius_wadoux(training_full, focal_point, ri)

    pred_rf <- NA_real_
    pred_nmk <- NA_real_
    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(training_full, form_rf)
      pred_rf <- predict_rf_default_wadoux(rf, focal_point)
    }

    if ("NeuroMaternKrige" %in% models && nmk_used < nmk_cap) {
      inner <- make_inner_val_split_nmk(
        training_full, val_frac = val_frac,
        seed = train_seed + iter * 1000L + j)
      out <- run_nmk_on_split(
        train_core = inner$train, val_df = inner$val, test_df = focal_point,
        calibration_patch_cache = calibration_patch_cache
      )
      pred_nmk <- out$pred_test[1]
      diag_list[[length(diag_list) + 1L]] <- out$diagnostics
      nmk_used <- nmk_used + 1L
    }

    pred_rows[[j]] <- data.frame(
      focal_id = id_focal,
      obs      = focal_point[[response_name_wadoux]],
      pred_rf  = pred_rf,
      pred_nmk = pred_nmk,
      stringsAsFactors = FALSE
    )
  }

  pred_df <- do.call(rbind, pred_rows)

  # Mirror the official Wadoux grouping: assign focal predictions to
  # bloo_groups subgroups, compute metrics per group, average.
  pred_df$group <- caret::createFolds(
    seq_len(nrow(pred_df)), k = bloo_groups,
    list = FALSE, returnTrain = FALSE)

  rows <- list()
  group_ids <- sort(unique(pred_df$group))

  aggregate_group_metrics <- function(df_sub, model_col) {
    idx <- is.finite(df_sub[[model_col]]) & is.finite(df_sub$obs)
    if (sum(idx) < 2L) return(NULL)
    wadoux_eval(obs = df_sub$obs[idx], pred = df_sub[[model_col]][idx])
  }

  if ("RF" %in% models) {
    per_group <- lapply(group_ids, function(g) {
      sub <- pred_df[pred_df$group == g, , drop = FALSE]
      aggregate_group_metrics(sub, "pred_rf")
    })
    ok <- per_group[!vapply(per_group, is.null, logical(1))]
    if (length(ok) > 0L) {
      stacked <- do.call(rbind, ok)
      rows[[length(rows) + 1L]] <- data.frame(
        iteration = iter,  scenario = scenario,  protocol = "BLOOCV",
        model = "RF",
        ME    = mean(stacked$ME,   na.rm = TRUE),
        RMSE  = mean(stacked$RMSE, na.rm = TRUE),
        r2    = mean(stacked$r2,   na.rm = TRUE),
        MEC   = mean(stacked$MEC,  na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    }
  }

  if ("NeuroMaternKrige" %in% models) {
    per_group <- lapply(group_ids, function(g) {
      sub <- pred_df[pred_df$group == g, , drop = FALSE]
      aggregate_group_metrics(sub, "pred_nmk")
    })
    ok <- per_group[!vapply(per_group, is.null, logical(1))]
    if (length(ok) > 0L) {
      stacked <- do.call(rbind, ok)
      rows[[length(rows) + 1L]] <- data.frame(
        iteration = iter,  scenario = scenario,  protocol = "BLOOCV",
        model = "NeuroMaternKrige",
        ME    = mean(stacked$ME,   na.rm = TRUE),
        RMSE  = mean(stacked$RMSE, na.rm = TRUE),
        r2    = mean(stacked$r2,   na.rm = TRUE),
        MEC   = mean(stacked$MEC,  na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    }
  }

  list(
    metrics = if (length(rows) == 0L) data.frame() else do.call(rbind, rows),
    diagnostics = list(NeuroMaternKrige = diag_list)
  )
}

# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------

save_nmk_checkpoint <- function(iter, metric_rows, diagnostic_rows = list(),
                                tuning_rows = list(),
                                variogram_rows = list(),
                                prediction_audit_rows = list(),
                                split_audit_rows = list()) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  write.csv(df, file.path(results_dir, "nmk_all_results.csv"),
            row.names = FALSE)

  deltas <- paired_delta_rows_nmk(df)
  if (nrow(deltas) > 0L) {
    write.csv(deltas, file.path(results_dir, "nmk_paired_deltas.csv"),
              row.names = FALSE)
  }

  diag_df <- bind_rows_fill_nmk(diagnostic_rows)
  if (nrow(diag_df) > 0L) {
    write.csv(diag_df, file.path(results_dir, "nmk_fold_diagnostics.csv"),
              row.names = FALSE)
  }

  tune_df <- bind_rows_fill_nmk(tuning_rows)
  if (nrow(tune_df) > 0L) {
    write.csv(tune_df, file.path(results_dir, "nmk_tuning_scores.csv"),
              row.names = FALSE)
  }

  vario_df <- bind_rows_fill_nmk(variogram_rows)
  if (nrow(vario_df) > 0L) {
    write.csv(vario_df, file.path(results_dir, "nmk_variogram_summary.csv"),
              row.names = FALSE)
  }

  audit_df <- bind_rows_fill_nmk(prediction_audit_rows)
  if (nrow(audit_df) > 0L) {
    write.csv(audit_df, file.path(results_dir, "nmk_prediction_audit.csv"),
              row.names = FALSE)
  }

  split_df <- bind_rows_fill_nmk(split_audit_rows)
  if (nrow(split_df) > 0L) {
    write.csv(split_df, file.path(results_dir, "nmk_split_audit.csv"),
              row.names = FALSE)
  }

  write.csv(
    data.frame(
      completed_iteration = iter,
      total_iterations = n_iter,
      scenario = scenario,
      protocols = paste(protocols, collapse = ","),
      models = paste(models, collapse = ","),
      completed_at = format(Sys.time(), tz = "", usetz = TRUE),
      stringsAsFactors = FALSE
    ),
    file.path(results_dir, "nmk_checkpoint_status.csv"),
    row.names = FALSE
  )
}

summarise_and_save <- function(metric_rows) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) {
    return(invisible(NULL))
  }
  by_protocol <- aggregate(
    cbind(ME, RMSE, r2, MEC) ~ scenario + protocol + model,
    data = df,
    FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                        sd   = sd(x,   na.rm = TRUE))
  )
  # Flatten for CSV friendliness.
  stats_df <- do.call(cbind, lapply(c("ME", "RMSE", "r2", "MEC"), function(m) {
    mat <- by_protocol[[m]]
    setNames(as.data.frame(mat), c(paste0(m, "_mean"), paste0(m, "_sd")))
  }))
  out <- cbind(by_protocol[, c("scenario", "protocol", "model")], stats_df)
  write.csv(out, file.path(results_dir, "nmk_summary_by_protocol.csv"),
            row.names = FALSE)
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

all_metric_rows <- list()
all_diag_rows <- list()
all_tune_rows <- list()
all_vario_rows <- list()
all_prediction_audit_rows <- list()
all_split_audit_rows <- list()

write.csv(
  data.frame(
    scenario = scenario,
    protocols = paste(protocols, collapse = ","),
    models = paste(models, collapse = ","),
    include_population = include_population,
    sample_size = sample_size,
    n_iter = n_iter,
    val_frac = val_frac,
    val_dist_km = val_dist_km,
    random_k = random_k,
    bloo_groups = bloo_groups,
    bloo_test_pixels = bloo_pixels,
    patch_size = patch_size,
    device = device,
    profile = profile,
    train_seed = train_seed,
	    fair_tune = nmk_tune,
	    autotune_v2 = nmk_autotune_v2,
	    autotune_v3 = nmk_autotune_v3,
	    autotune_v4 = nmk_autotune_v4,
	    autotune_backbone = nmk_autotune_backbone,
	    refit_full_calibration = nmk_refit_full_calibration,
	    refit_warm_start = nmk_refit_warm_start,
	    refit_warm_start_epoch_mult = nmk_refit_warm_start_epoch_mult,
	    refit_warm_start_min_epochs = nmk_refit_warm_start_min_epochs,
	    refit_warm_start_max_epochs = nmk_refit_warm_start_max_epochs,
	    tune_full_grid = nmk_tune_full_grid,
	    tune_max_candidates = nmk_tune_max_candidates,
	    tune_inner_folds = nmk_tune_inner_folds,
	    tune_num_workers = 1L,
	    tune_num_workers_requested = nmk_num_workers,
	    tune_n_seeds = nmk_tune_n_seeds,
	    v3_stage2_top_k = nmk_v3_stage2_top_k,
	    v3_stage2_n_seeds = nmk_v3_stage2_n_seeds,
	    ensemble_top_k = nmk_ensemble_top_k,
	    ensemble_weighting = nmk_ensemble_weighting,
	    ensemble_temperature = nmk_ensemble_temperature,
	    v3_alpha_grid = paste(nmk_v3_alpha_grid, collapse = ","),
	    v3_calibration_policies = paste(nmk_v3_calibration_policies, collapse = ","),
	    v3_rmse_sd_weight = nmk_v3_rmse_sd_weight,
	    v3_abs_me_weight = nmk_v3_abs_me_weight,
		    v3_log_margin = nmk_v3_log_margin,
		    v3_affine_slope_min = nmk_v3_affine_slope_min,
		    v3_affine_slope_max = nmk_v3_affine_slope_max,
			    selection_objective = nmk_selection_objective,
			    noharm_policy = nmk_noharm_policy,
			    noharm_margin_abs = nmk_noharm_margin_abs,
			    noharm_margin_rel = nmk_noharm_margin_rel,
			    noharm_restrict_ensemble = nmk_noharm_restrict_ensemble,
			    rf_noharm_policy = nmk_rf_noharm_policy,
			    rf_noharm_margin_abs = nmk_rf_noharm_margin_abs,
			    rf_noharm_margin_rel = nmk_rf_noharm_margin_rel,
			    rf_noharm_restrict_ensemble = nmk_rf_noharm_restrict_ensemble,
			    v4_quantile_bins = nmk_v4_quantile_bins,
	    v4_quantile_excess_weight = nmk_v4_quantile_excess_weight,
	    v4_quantile_sd_weight = nmk_v4_quantile_sd_weight,
	    save_variograms = nmk_save_variograms,
	    save_prediction_audit = nmk_save_prediction_audit,
	    NMK_TUNE = raw_env_nmk("NMK_TUNE", "TRUE"),
	    NMK_AUTOTUNE_V2 = raw_env_nmk("NMK_AUTOTUNE_V2", "TRUE"),
	    NMK_AUTOTUNE_V3 = raw_env_nmk("NMK_AUTOTUNE_V3", "TRUE"),
	    NMK_AUTOTUNE_V4 = raw_env_nmk("NMK_AUTOTUNE_V4", "TRUE"),
	    NMK_AUTOTUNE_BACKBONE = raw_env_nmk("NMK_AUTOTUNE_BACKBONE", "TRUE"),
	    NMK_TUNE_FULL_GRID = raw_env_nmk("NMK_TUNE_FULL_GRID", "TRUE"),
	    NMK_TUNE_TARGET_TRANSFORMS = raw_env_nmk(
	      "NMK_TUNE_TARGET_TRANSFORMS", "auto,identity,log"),
	    NMK_TUNE_MAX_CANDIDATES = raw_env_nmk("NMK_TUNE_MAX_CANDIDATES", "28"),
	    NMK_TUNE_INNER_FOLDS = raw_env_nmk("NMK_TUNE_INNER_FOLDS", "3"),
	    NMK_NUM_WORKERS = raw_env_nmk("NMK_NUM_WORKERS", "1"),
	    NMK_TUNE_N_SEEDS = raw_env_nmk("NMK_TUNE_N_SEEDS", "1"),
	    NMK_V3_STAGE2_TOP_K = raw_env_nmk("NMK_V3_STAGE2_TOP_K", "5"),
	    NMK_V3_STAGE2_N_SEEDS = raw_env_nmk("NMK_V3_STAGE2_N_SEEDS", "3"),
	    NMK_ENSEMBLE_TOP_K = raw_env_nmk("NMK_ENSEMBLE_TOP_K", "5"),
	    NMK_ENSEMBLE_WEIGHTING = raw_env_nmk("NMK_ENSEMBLE_WEIGHTING", "softmax"),
	    NMK_ENSEMBLE_TEMPERATURE = raw_env_nmk("NMK_ENSEMBLE_TEMPERATURE", "0.75"),
	    NMK_V3_ALPHA_GRID = raw_env_nmk("NMK_V3_ALPHA_GRID", "0,0.25,0.5,0.75,1"),
	    NMK_V3_CALIBRATION_POLICIES = raw_env_nmk(
	      "NMK_V3_CALIBRATION_POLICIES", "none,median_bias,mean_bias,affine"),
	    NMK_V3_RMSE_SD_WEIGHT = raw_env_nmk("NMK_V3_RMSE_SD_WEIGHT", "0.25"),
	    NMK_V3_ABS_ME_WEIGHT = raw_env_nmk("NMK_V3_ABS_ME_WEIGHT", "0.10"),
	    NMK_V3_LOG_MARGIN = raw_env_nmk("NMK_V3_LOG_MARGIN", "0.50"),
		    NMK_V3_AFFINE_SLOPE_MIN = raw_env_nmk("NMK_V3_AFFINE_SLOPE_MIN", "0.50"),
		    NMK_V3_AFFINE_SLOPE_MAX = raw_env_nmk("NMK_V3_AFFINE_SLOPE_MAX", "1.50"),
		    NMK_SELECTION_OBJECTIVE = raw_env_nmk(
		      "NMK_SELECTION_OBJECTIVE", "paired_rf_delta"),
		    NMK_NOHARM_POLICY = raw_env_nmk("NMK_NOHARM_POLICY", "TRUE"),
		    NMK_NOHARM_MARGIN_ABS = raw_env_nmk("NMK_NOHARM_MARGIN_ABS", "0.05"),
		    NMK_NOHARM_MARGIN_REL = raw_env_nmk("NMK_NOHARM_MARGIN_REL", "0"),
			    NMK_NOHARM_RESTRICT_ENSEMBLE = raw_env_nmk(
			      "NMK_NOHARM_RESTRICT_ENSEMBLE", "TRUE"),
			    NMK_RF_NOHARM_POLICY = raw_env_nmk("NMK_RF_NOHARM_POLICY", "TRUE"),
			    NMK_RF_NOHARM_MARGIN_ABS = raw_env_nmk(
			      "NMK_RF_NOHARM_MARGIN_ABS", "0.05"),
			    NMK_RF_NOHARM_MARGIN_REL = raw_env_nmk(
			      "NMK_RF_NOHARM_MARGIN_REL", "0"),
			    NMK_RF_NOHARM_RESTRICT_ENSEMBLE = raw_env_nmk(
			      "NMK_RF_NOHARM_RESTRICT_ENSEMBLE", "TRUE"),
			    NMK_V4_QUANTILE_BINS = raw_env_nmk("NMK_V4_QUANTILE_BINS", "5"),
	    NMK_V4_QUANTILE_EXCESS_WEIGHT = raw_env_nmk(
	      "NMK_V4_QUANTILE_EXCESS_WEIGHT", "0.15"),
	    NMK_V4_QUANTILE_SD_WEIGHT = raw_env_nmk("NMK_V4_QUANTILE_SD_WEIGHT", "0.05"),
	    NMK_TAB_QUANTILE_BASIS = raw_env_nmk("NMK_TAB_QUANTILE_BASIS", ""),
	    NMK_TAB_QBASIS_KNOTS = raw_env_nmk("NMK_TAB_QBASIS_KNOTS", ""),
	    NMK_TAB_CROSSNET = raw_env_nmk("NMK_TAB_CROSSNET", ""),
	    NMK_TAB_CROSS_LAYERS = raw_env_nmk("NMK_TAB_CROSS_LAYERS", ""),
	    NMK_TAB_INPUT_DROPOUT = raw_env_nmk("NMK_TAB_INPUT_DROPOUT", ""),
	    NMK_TAB_SKIP = raw_env_nmk("NMK_TAB_SKIP", ""),
	    NMK_TAB_SKIP_HIDDEN = raw_env_nmk("NMK_TAB_SKIP_HIDDEN", ""),
	    NMK_N_TREES = raw_env_nmk("NMK_N_TREES", ""),
	    NMK_TREE_DEPTH = raw_env_nmk("NMK_TREE_DEPTH", ""),
	    NMK_REFIT_FULL_CALIBRATION = raw_env_nmk("NMK_REFIT_FULL_CALIBRATION", "TRUE"),
	    NMK_REFIT_WARM_START = raw_env_nmk("NMK_REFIT_WARM_START", "TRUE"),
	    NMK_REFIT_WARM_START_EPOCH_MULT = raw_env_nmk(
	      "NMK_REFIT_WARM_START_EPOCH_MULT", "0.35"),
	    NMK_REFIT_WARM_START_MIN_EPOCHS = raw_env_nmk(
	      "NMK_REFIT_WARM_START_MIN_EPOCHS", "3"),
	    NMK_REFIT_WARM_START_MAX_EPOCHS = raw_env_nmk(
	      "NMK_REFIT_WARM_START_MAX_EPOCHS", "15"),
	    NMK_REFIT_N_SEEDS = raw_env_nmk("NMK_REFIT_N_SEEDS", ""),
	    NMK_SAVE_VARIOGRAMS = raw_env_nmk("NMK_SAVE_VARIOGRAMS", "TRUE"),
	    NMK_SAVE_PREDICTION_AUDIT = raw_env_nmk("NMK_SAVE_PREDICTION_AUDIT", "TRUE"),
	    stringsAsFactors = FALSE
  ),
  file.path(results_dir, "nmk_config.csv"),
  row.names = FALSE
)

for (iter in seq_len(n_iter)) {
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("NMK vs RF | scenario=%s | iteration %d / %d\n",
              scenario, iter, n_iter))
  cat(sprintf("==================================================\n"))

  valuetable <- na.omit(as.data.frame(sample_calibration_nmk()))
  valuetable$..patch_cache_row_id <- seq_len(nrow(valuetable))

  calibration_patch_cache <- NULL
  if ("NeuroMaternKrige" %in% models) {
    calibration_patch_cache <- nmk_env$build_pointpatch_patch_cache(
      context        = nmk_env$wadoux_context,
      calibration_df = valuetable,
      patch_size     = patch_size,
      label          = sprintf("scenario=%s iter=%d calibration patches",
                               scenario, iter)
    )
  }

  iter_rows <- list()
  iter_diag <- list()
  iter_tune <- list()
  iter_vario <- list()
  iter_prediction_audit <- list()
  iter_split_audit <- list()

  if ("Population" %in% protocols) {
    cat("\n--- Population ---\n")
    out <- eval_population(valuetable, iter, calibration_patch_cache)
    iter_rows[["Population"]] <- out$metrics
  }

  if ("DesignBased" %in% protocols) {
    cat("\n--- DesignBased ---\n")
    out <- eval_designbased(valuetable, iter, calibration_patch_cache)
    iter_rows[["DesignBased"]] <- out$metrics
    iter_diag[["DesignBased"]] <- out$diagnostics
    iter_tune[["DesignBased"]] <- out$tuning
    iter_vario[["DesignBased"]] <- out$variograms
    iter_prediction_audit[["DesignBased"]] <- out$prediction_audit
    iter_split_audit[["DesignBased"]] <- out$split_audit
  }

  if ("RandomKFold" %in% protocols) {
    cat("\n--- RandomKFold ---\n")
    out <- eval_randomkfold(valuetable, iter, calibration_patch_cache)
    iter_rows[["RandomKFold"]] <- out$metrics
  }

  if ("SpatialKFold" %in% protocols) {
    cat("\n--- SpatialKFold ---\n")
    out <- eval_spatialkfold(valuetable, iter, calibration_patch_cache)
    iter_rows[["SpatialKFold"]] <- out$metrics
  }

  if ("BLOOCV" %in% protocols) {
    cat("\n--- BLOOCV ---\n")
    out <- eval_bloocv(valuetable, iter, calibration_patch_cache)
    iter_rows[["BLOOCV"]] <- out$metrics
  }

  iter_df <- do.call(rbind, iter_rows[!vapply(iter_rows, is.null, logical(1))])
  if (!is.null(iter_df) && nrow(iter_df) > 0L) {
    print(iter_df)
    all_metric_rows[[iter]] <- iter_df
  }
  iter_diag_df <- bind_rows_fill_nmk(iter_diag)
  if (nrow(iter_diag_df) > 0L) {
    all_diag_rows[[iter]] <- iter_diag_df
  }
  iter_tune_df <- bind_rows_fill_nmk(iter_tune)
  if (nrow(iter_tune_df) > 0L) {
    all_tune_rows[[iter]] <- iter_tune_df
  }
  iter_vario_df <- bind_rows_fill_nmk(iter_vario)
  if (nrow(iter_vario_df) > 0L) {
    all_vario_rows[[iter]] <- iter_vario_df
  }
  iter_prediction_audit_df <- bind_rows_fill_nmk(iter_prediction_audit)
  if (nrow(iter_prediction_audit_df) > 0L) {
    all_prediction_audit_rows[[iter]] <- iter_prediction_audit_df
  }
  iter_split_audit_df <- bind_rows_fill_nmk(iter_split_audit)
  if (nrow(iter_split_audit_df) > 0L) {
    all_split_audit_rows[[iter]] <- iter_split_audit_df
  }

  if (iter %% checkpoint_every == 0L || iter == n_iter) {
    save_nmk_checkpoint(iter, all_metric_rows, all_diag_rows, all_tune_rows,
                        all_vario_rows, all_prediction_audit_rows,
                        all_split_audit_rows)
    summarise_and_save(all_metric_rows)
  }
}

cat("\n[NMK runner] Done.  Results written to: ", results_dir, "\n", sep = "")
