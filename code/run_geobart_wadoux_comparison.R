# =============================================================================
# run_geobart_wadoux_comparison.R
#
# Head-to-head benchmark under the Wadoux (2021) protocols:
#
#   - RF       (ranger defaults — Wadoux baseline)
#   - Cubist   (rule-based regression, classical DSM)
#   - GeoBART  (Bayesian Additive Regression Trees with variogram-informed
#              spatial RBF basis, defined in code/GeoBART.R)
#
# Mirrors run_neuromatern_wadoux_comparison.R / run_deepkernel_wadoux_comparison.R
# in scenario, sampling, protocol, and metric conventions, but is much
# slimmer because GeoBART has no patches, no MPS, no GPU, no torch — it is
# a self-contained tabular Bayesian model.
#
# Environment overrides:
#   GEOBART_SCENARIO         random | regular_grid | clustered_random
#   GEOBART_PROTOCOLS        comma list of {DesignBased, RandomKFold,
#                                            SpatialKFold, BLOOCV, Population}
#   GEOBART_MODELS           comma list of {RF, Cubist, GeoBART}
#   GEOBART_N_ITER           number of calibration samples (default 5)
#   GEOBART_SAMPLE_SIZE      calibration size per iteration (default 500)
#   GEOBART_VAL_FRAC         inner validation fraction (default 0.20)
#   GEOBART_VAL_DIST_KM      spatial CV / BLOOCV exclusion radius km
#   GEOBART_RANDOM_K         random k-fold splits (default 10)
#   GEOBART_BLOO_GROUPS      BLOOCV groups (default 10)
#   GEOBART_BLOO_TEST_PIXELS BLOOCV focal pixels per group (default 100)
#   GEOBART_BLOO_GB_PIXELS   cap GeoBART retrains in BLOOCV (default 50/group)
#   GEOBART_TRAIN_SEED       RNG seed (default 123)
#   GEOBART_RESULTS_DIR      output dir (default results/geobart_run)
#   GEOBART_NDPOST, GEOBART_NSKIP, GEOBART_NTREE
#   GEOBART_DUAN_*           see GeoBART.R
#   GEOBART_TARGET_TRANSFORM auto (default) | identity | log
#   GEOBART_TUNE             TRUE (default) | FALSE
#   GEOBART_TUNE_METRIC      RMSE (default) | MAE | MEC | R2
#   GEOBART_AUTOTUNE_V2      TRUE enables staged tuning + policy/prior search
#   GEOBART_AUTOTUNE_V3      TRUE adds fold-stability scoring, post-calibration
#                            policy search, and seed ensembles in stage 2
#   GEOBART_AUTOTUNE_V4      TRUE adds target-quantile robust scoring
#   GEOBART_TUNE_FULL_GRID   TRUE (default) removes the stage-1 cap; stage 2
#                            stays capped unless GEOBART_V3_STAGE2_FULL_GRID=TRUE
#   GEOBART_TUNE_INNER_FOLDS inner CV folds for candidate scoring (default 3)
#   GEOBART_REFIT_FULL_CALIBRATION TRUE (default) refits selected params on all
#                                  calibration rows before held-out prediction
#   GEOBART_REFIT_N_SEEDS    seed ensemble size for full-calibration refit
#   GEOBART_NUM_WORKERS      parallel candidate workers for tuning
#   GEOBART_SAVE_VARIOGRAMS  TRUE (default) writes semivariogram CSV/PNG diagnostics
#   GEOBART_SAVE_PREDICTION_AUDIT TRUE writes per-point held-out predictions
#   GEOBART_TUNE_*           candidate grid controls; see GeoBART.R
#   GEOBART_CHECKPOINT_EVERY
# =============================================================================

rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")
source("./code/GeoBART.R")

# Cubist + ranger are in CRAN; loaded by helpers, but we re-load defensively.
suppressPackageStartupMessages({
  if (requireNamespace("Cubist", quietly = TRUE)) library(Cubist)
  if (requireNamespace("ranger", quietly = TRUE)) library(ranger)
  if (requireNamespace("BART",   quietly = TRUE)) library(BART)
})

if (!requireNamespace("BART", quietly = TRUE)) {
  stop("Install the BART package first: install.packages('BART').")
}

# ---- Env-var parsing --------------------------------------------------------

parse_bool_gb <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}
parse_num_gb <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  v   <- suppressWarnings(as.numeric(raw))
  if (!is.finite(v)) default else v
}
parse_optional_num_gb <- function(name) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(NULL)
  v <- suppressWarnings(as.numeric(raw))
  if (!is.finite(v)) NULL else v
}
raw_env_gb <- function(name, default = "") {
  Sys.getenv(name, unset = default)
}
parse_list_gb <- function(name, default) {
  raw <- Sys.getenv(name, unset = paste(default, collapse = ","))
  v   <- trimws(unlist(strsplit(raw, ",")))
  v[v != ""]
}

# ---- Config -----------------------------------------------------------------

scenario        <- Sys.getenv("GEOBART_SCENARIO", unset = "random")
protocols       <- parse_list_gb("GEOBART_PROTOCOLS",
                                  c("DesignBased", "RandomKFold",
                                    "SpatialKFold", "BLOOCV"))
models          <- parse_list_gb("GEOBART_MODELS",
                                  c("RF", "Cubist", "GeoBART"))
allowed_models  <- c("RF", "Cubist", "GeoBART")
invalid         <- setdiff(models, allowed_models)
if (length(invalid) > 0L) {
  stop("Unsupported GEOBART_MODELS entries: ", paste(invalid, collapse = ", "))
}
include_population <- "Population" %in% protocols

n_iter          <- as.integer(parse_num_gb("GEOBART_N_ITER",          5L))
sample_size     <- as.integer(parse_num_gb("GEOBART_SAMPLE_SIZE",     500L))
val_frac        <-             parse_num_gb("GEOBART_VAL_FRAC",       0.20)
val_dist_km     <-             parse_num_gb("GEOBART_VAL_DIST_KM",    350)
random_k        <- as.integer(parse_num_gb("GEOBART_RANDOM_K",        10L))
bloo_groups     <- as.integer(parse_num_gb("GEOBART_BLOO_GROUPS",     10L))
bloo_pixels     <- as.integer(parse_num_gb("GEOBART_BLOO_TEST_PIXELS", 100L))
train_seed      <- as.integer(parse_num_gb("GEOBART_TRAIN_SEED",      123L))
results_dir     <- Sys.getenv("GEOBART_RESULTS_DIR",
                               unset = "results/geobart_run")
checkpoint_every <- as.integer(parse_num_gb("GEOBART_CHECKPOINT_EVERY", 1L))
geobart_tune     <- parse_bool_gb("GEOBART_TUNE", TRUE)
geobart_tune_metric <- toupper(Sys.getenv("GEOBART_TUNE_METRIC",
                                           unset = "RMSE"))
geobart_selection_objective <- tolower(trimws(Sys.getenv(
  "GEOBART_SELECTION_OBJECTIVE", unset = "paired_rf_delta")))
if (!(geobart_selection_objective %in% c("rmse", "paired_rf_delta"))) {
  geobart_selection_objective <- "rmse"
}
geobart_noharm_policy <- parse_bool_gb(
  "GEOBART_NOHARM_POLICY",
  identical(geobart_selection_objective, "paired_rf_delta"))
geobart_noharm_margin_abs <- parse_num_gb("GEOBART_NOHARM_MARGIN_ABS", 0.05)
if (!is.finite(geobart_noharm_margin_abs) || geobart_noharm_margin_abs < 0) {
  geobart_noharm_margin_abs <- 0
}
geobart_noharm_margin_rel <- parse_num_gb("GEOBART_NOHARM_MARGIN_REL", 0.0)
if (!is.finite(geobart_noharm_margin_rel) || geobart_noharm_margin_rel < 0) {
  geobart_noharm_margin_rel <- 0
}
geobart_noharm_restrict_ensemble <- parse_bool_gb(
  "GEOBART_NOHARM_RESTRICT_ENSEMBLE", TRUE)
geobart_rf_noharm_policy <- parse_bool_gb(
  "GEOBART_RF_NOHARM_POLICY",
  identical(geobart_selection_objective, "paired_rf_delta"))
geobart_rf_noharm_margin_abs <- parse_num_gb(
  "GEOBART_RF_NOHARM_MARGIN_ABS", 0.05)
if (!is.finite(geobart_rf_noharm_margin_abs) ||
    geobart_rf_noharm_margin_abs < 0) {
  geobart_rf_noharm_margin_abs <- 0
}
geobart_rf_noharm_margin_rel <- parse_num_gb(
  "GEOBART_RF_NOHARM_MARGIN_REL", 0.0)
if (!is.finite(geobart_rf_noharm_margin_rel) ||
    geobart_rf_noharm_margin_rel < 0) {
  geobart_rf_noharm_margin_rel <- 0
}
geobart_rf_noharm_restrict_ensemble <- parse_bool_gb(
  "GEOBART_RF_NOHARM_RESTRICT_ENSEMBLE", TRUE)
geobart_autotune_v2 <- parse_bool_gb("GEOBART_AUTOTUNE_V2", TRUE)
geobart_autotune_v3 <- parse_bool_gb("GEOBART_AUTOTUNE_V3",
                                      geobart_autotune_v2)
geobart_autotune_v4 <- parse_bool_gb("GEOBART_AUTOTUNE_V4",
                                      geobart_autotune_v3)
if (isTRUE(geobart_autotune_v4)) geobart_autotune_v3 <- TRUE
if (isTRUE(geobart_autotune_v3)) geobart_autotune_v2 <- TRUE
geobart_tune_full_grid <- parse_bool_gb("GEOBART_TUNE_FULL_GRID", TRUE)
geobart_tune_max_candidates <- if (isTRUE(geobart_tune_full_grid)) {
  Inf
} else {
  parse_optional_num_gb("GEOBART_TUNE_MAX_CANDIDATES")
}
geobart_tune_inner_folds <- as.integer(parse_num_gb(
  "GEOBART_TUNE_INNER_FOLDS", 3L))
if (!is.finite(geobart_tune_inner_folds) || geobart_tune_inner_folds < 1L) {
  geobart_tune_inner_folds <- 1L
}
geobart_refit_full_calibration <- parse_bool_gb(
  "GEOBART_REFIT_FULL_CALIBRATION", TRUE)
geobart_refit_n_seeds <- as.integer(parse_num_gb("GEOBART_REFIT_N_SEEDS", 5L))
if (!is.finite(geobart_refit_n_seeds) || geobart_refit_n_seeds < 1L) {
  geobart_refit_n_seeds <- 1L
}
geobart_ensemble_top_k <- as.integer(parse_num_gb(
  "GEOBART_ENSEMBLE_TOP_K", if (isTRUE(geobart_autotune_v4)) 5L else 1L))
if (!is.finite(geobart_ensemble_top_k) || geobart_ensemble_top_k < 1L) {
  geobart_ensemble_top_k <- 1L
}
geobart_ensemble_weighting <- tolower(trimws(Sys.getenv(
  "GEOBART_ENSEMBLE_WEIGHTING", unset = "softmax")))
if (!(geobart_ensemble_weighting %in% c("softmax", "inverse", "uniform"))) {
  stop("Unsupported GEOBART_ENSEMBLE_WEIGHTING: ",
       geobart_ensemble_weighting,
       ". Use softmax, inverse, or uniform.")
}
geobart_ensemble_temperature <- parse_num_gb(
  "GEOBART_ENSEMBLE_TEMPERATURE", 0.75)
if (!is.finite(geobart_ensemble_temperature) ||
    geobart_ensemble_temperature <= 0) {
  geobart_ensemble_temperature <- 0.75
}
geobart_num_workers <- as.integer(parse_num_gb("GEOBART_NUM_WORKERS", 1L))
if (!is.finite(geobart_num_workers) || geobart_num_workers < 1L) {
  geobart_num_workers <- 1L
}
geobart_save_variograms <- parse_bool_gb("GEOBART_SAVE_VARIOGRAMS", TRUE)
geobart_save_prediction_audit <- parse_bool_gb(
  "GEOBART_SAVE_PREDICTION_AUDIT", TRUE)

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
options(wadoux_corr_method = "pearson")

# GeoBART parameters (most NULL = derive automatically)
geobart_params <- modifyList(
  geobart_default_params,
  list(
    train_seed = train_seed,
    ntree      = if (nzchar(Sys.getenv("GEOBART_NTREE", "")))
                   as.integer(parse_num_gb("GEOBART_NTREE",  NA)) else NULL,
    ndpost     = if (nzchar(Sys.getenv("GEOBART_NDPOST", "")))
                   as.integer(parse_num_gb("GEOBART_NDPOST", NA)) else NULL,
    nskip      = if (nzchar(Sys.getenv("GEOBART_NSKIP", "")))
                   as.integer(parse_num_gb("GEOBART_NSKIP",  NA)) else NULL,
	    n_basis    = if (nzchar(Sys.getenv("GEOBART_N_BASIS", "")))
	                   as.integer(parse_num_gb("GEOBART_N_BASIS", NA)) else NULL,
	    bandwidth  = if (nzchar(Sys.getenv("GEOBART_BANDWIDTH", "")))
	                   parse_num_gb("GEOBART_BANDWIDTH", NA) else NULL,
	    spatial_policy = if (nzchar(Sys.getenv("GEOBART_SPATIAL_POLICY", "")))
	                   Sys.getenv("GEOBART_SPATIAL_POLICY") else NULL,
	    anisotropy_ratio = if (nzchar(Sys.getenv("GEOBART_ANISOTROPY_RATIO", "")))
	                   parse_num_gb("GEOBART_ANISOTROPY_RATIO", NA) else NULL,
	    k          = if (nzchar(Sys.getenv("GEOBART_K", "")))
	                   parse_num_gb("GEOBART_K", NA) else NULL,
	    power      = if (nzchar(Sys.getenv("GEOBART_POWER", "")))
	                   parse_num_gb("GEOBART_POWER", NA) else NULL,
	    base       = if (nzchar(Sys.getenv("GEOBART_BASE", "")))
	                   parse_num_gb("GEOBART_BASE", NA) else NULL,
	    sigdf      = if (nzchar(Sys.getenv("GEOBART_SIGDF", "")))
	                   parse_num_gb("GEOBART_SIGDF", NA) else NULL,
	    sigquant   = if (nzchar(Sys.getenv("GEOBART_SIGQUANT", "")))
	                   parse_num_gb("GEOBART_SIGQUANT", NA) else NULL,
	    duan_apply = if (nzchar(Sys.getenv("GEOBART_DUAN_APPLY", "")))
	                   parse_bool_gb("GEOBART_DUAN_APPLY", TRUE) else NULL,
    duan_min_val_n = if (nzchar(Sys.getenv("GEOBART_DUAN_MIN_VAL_N", "")))
                   as.integer(parse_num_gb("GEOBART_DUAN_MIN_VAL_N", NA)) else NULL,
    duan_trim_frac = if (nzchar(Sys.getenv("GEOBART_DUAN_TRIM_FRAC", "")))
                   parse_num_gb("GEOBART_DUAN_TRIM_FRAC", NA) else NULL
  )
)
tt_override <- Sys.getenv("GEOBART_TARGET_TRANSFORM", unset = "")
if (nzchar(tt_override)) {
  v <- tolower(trimws(tt_override))
  if (v %in% c("auto", "identity", "log")) geobart_params$target_transform <- v
}

cat(sprintf(
  "[GeoBART runner] scenario=%s  protocols=%s  models=%s  n_iter=%d  n=%d  tune=%s  objective=%s  noharm=%s  rf_noharm=%s  autotune_v2=%s  autotune_v3=%s  autotune_v4=%s  full_grid=%s  inner_folds=%d  refit_full=%s  refit_seeds=%d  ensemble_top_k=%d  ensemble_weighting=%s  ensemble_temp=%.3f  workers=%d  variograms=%s  audit=%s\n",
  scenario, paste(protocols, collapse = ","),
  paste(models, collapse = ","), n_iter, sample_size, geobart_tune,
  geobart_selection_objective, geobart_noharm_policy,
  geobart_rf_noharm_policy,
  geobart_autotune_v2, geobart_autotune_v3, geobart_autotune_v4,
  geobart_tune_full_grid,
  geobart_tune_inner_folds,
  geobart_refit_full_calibration, geobart_refit_n_seeds,
  geobart_ensemble_top_k, geobart_ensemble_weighting,
  geobart_ensemble_temperature,
  geobart_num_workers, geobart_save_variograms,
  geobart_save_prediction_audit))

# ---- Data -------------------------------------------------------------------

need_polygon <- scenario %in% c("regular_grid", "clustered_random")
common_data <- load_wadoux_common_data(include_polygon = need_polygon)
predList    <- predList_modelfull_wadoux
form_rf     <- as.formula(
  paste(response_name_wadoux, "~", paste(predList, collapse = "+"))
)

# ---- Sampling helpers (mirror NMK/DKP runners) ------------------------------

sample_calibration_gb <- function() {
  if (scenario == "random") {
    sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  } else if (scenario == "regular_grid") {
    sample_valuetable_regular_grid_wadoux(
      common_data = common_data, sample_size = sample_size)
  } else if (scenario == "clustered_random") {
    sample_valuetable_clustered_random_wadoux(
      common_data    = common_data,
      kmeans_centers = as.integer(parse_num_gb("GEOBART_KMEANS_CENTERS", 100L)),
      n_psu          = as.integer(parse_num_gb("GEOBART_N_PSU",          20L)),
      m_per_psu      = as.integer(parse_num_gb("GEOBART_M_PER_PSU",      25L)))
  } else stop("Unsupported GEOBART_SCENARIO: ", scenario)
}

make_inner_val_split_gb <- function(train_df, val_frac = 0.2, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n_train <- nrow(train_df)
  if (n_train < 10L) stop("Outer training set too small for GeoBART.")
  val_size  <- max(1L, floor(n_train * val_frac))
  val_idx   <- sample.int(n_train, size = val_size, replace = FALSE)
  train_idx <- setdiff(seq_len(n_train), val_idx)
  list(train = train_df[train_idx, , drop = FALSE],
       val   = train_df[val_idx,   , drop = FALSE])
}

# Build GeoBART fold from train/val/test data frames.
prepare_geobart_fold <- function(train_df, val_df, test_df) {
  list(
    X = list(
      train = as.matrix(train_df[, predList,    drop = FALSE]),
      val   = as.matrix(val_df  [, predList,    drop = FALSE]),
      test  = as.matrix(test_df [, predList,    drop = FALSE])
    ),
    coords = list(
      train = as.matrix(train_df[, c("x", "y"), drop = FALSE]),
      val   = as.matrix(val_df  [, c("x", "y"), drop = FALSE]),
      test  = as.matrix(test_df [, c("x", "y"), drop = FALSE])
    ),
    y = list(
      train = train_df[[response_name_wadoux]],
      val   = val_df  [[response_name_wadoux]],
      test  = test_df [[response_name_wadoux]]
    )
  )
}

run_geobart_on_split <- function(train_core, val_df, test_df,
                                 params = geobart_params,
                                 verbose = TRUE) {
  fd <- prepare_geobart_fold(train_core, val_df, test_df)
  do.call(train_geobart_one_fold,
          c(list(fd = fd, verbose = verbose), params))
}

tune_geobart_on_split <- function(train_core, val_df, test_df,
                                  params = geobart_params,
                                  verbose = TRUE) {
  fd <- prepare_geobart_fold(train_core, val_df, test_df)
  tune_geobart_one_fold(
	    fd = fd,
		    base_params = params,
		    selection_metric = geobart_tune_metric,
		    selection_objective = geobart_selection_objective,
		    max_candidates = geobart_tune_max_candidates,
	    inner_folds = geobart_tune_inner_folds,
	    num_workers = geobart_num_workers,
	    autotune_v2 = geobart_autotune_v2,
	    autotune_v3 = geobart_autotune_v3,
	    verbose = verbose
	  )
}

geobart_candidate_key_gb <- function(candidate_best) {
  if (is.null(candidate_best)) return(NA_character_)
  cand <- candidate_best$candidate
  candidate_id <- if (!is.null(cand) && "candidate_id" %in% names(cand)) {
    as.integer(cand$candidate_id[1L])
  } else {
    NA_integer_
  }
  stage <- if (!is.null(candidate_best$tune_stage)) {
    as.character(candidate_best$tune_stage[1L])
  } else {
    NA_character_
  }
  paste(stage, candidate_id, sep = "|")
}

geobart_candidate_label_gb <- function(candidate_best) {
  if (is.null(candidate_best)) return(NA_character_)
  cand <- candidate_best$candidate
  if (!is.null(cand) && "candidate_label" %in% names(cand)) {
    return(as.character(cand$candidate_label[1L]))
  }
  geobart_candidate_key_gb(candidate_best)
}

geobart_candidate_stage_gb <- function(candidate_best) {
  if (is.null(candidate_best) || is.null(candidate_best$tune_stage)) {
    return(NA_character_)
  }
  as.character(candidate_best$tune_stage[1L])
}

geobart_candidate_id_gb <- function(candidate_best) {
  cand <- candidate_best$candidate
  if (!is.null(cand) && "candidate_id" %in% names(cand)) {
    return(as.integer(cand$candidate_id[1L]))
  }
  NA_integer_
}

geobart_candidate_score_gb <- function(candidate_best) {
  score <- candidate_best$score
  if (is.null(score) || length(score) == 0L || !is.finite(score[1L])) {
    return(Inf)
  }
  as.numeric(score[1L])
}

geobart_candidate_pool_gb <- function(selection,
                                      top_k = geobart_ensemble_top_k) {
  pool <- selection$ranked_candidates
  if (is.null(pool) || length(pool) == 0L) {
    if (is.null(selection$params) || is.null(selection$out)) return(list())
    pool <- list(list(
      out = selection$out,
      candidate = selection$selected_candidate,
      params = selection$params,
      effective_transform =
        selection$out$diagnostics$target_transform_effective,
      tune_stage = selection$out$diagnostics$tuning_selected_stage,
      score = selection$out$diagnostics$tuning_selected_score,
      complexity = NA_real_,
      post_calibration = selection$post_policy
    ))
  }
  keep <- vapply(pool, function(x) {
    !is.null(x) && !is.null(x$params) && length(x$params) > 0L
  }, logical(1))
  pool <- pool[keep]
  if (length(pool) == 0L) return(list())
  keys <- vapply(pool, geobart_candidate_key_gb, character(1))
  pool <- pool[!duplicated(keys)]
  top_k <- as.integer(top_k)
  if (!is.finite(top_k) || top_k < 1L) top_k <- 1L
  pool[seq_len(min(top_k, length(pool)))]
}

geobart_ensemble_weights_gb <- function(scores,
                                        weighting = geobart_ensemble_weighting,
                                        temperature =
                                          geobart_ensemble_temperature) {
  n <- length(scores)
  if (n == 0L) return(numeric())
  if (n == 1L) return(1)
  scores <- as.numeric(scores)
  finite <- is.finite(scores)
  if (!any(finite)) return(rep(1 / n, n))
  worst <- max(scores[finite], na.rm = TRUE)
  scores[!finite] <- worst + abs(worst) + 1
  weighting <- tolower(as.character(weighting)[1L])
  temperature <- as.numeric(temperature)[1L]
  if (!is.finite(temperature) || temperature <= 0) temperature <- 1
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

geobart_mark_ensemble_tuning_rows_gb <- function(tuning_df, ranked_candidates) {
  if (is.null(tuning_df) || nrow(tuning_df) == 0L) return(tuning_df)
  tuning_df$ensemble_member <- FALSE
  tuning_df$ensemble_rank <- NA_integer_
  tuning_df$ensemble_weight <- NA_real_
  pool <- geobart_candidate_pool_gb(
    list(ranked_candidates = ranked_candidates,
         out = NULL, selected_candidate = NULL,
         params = NULL, post_policy = NULL),
    top_k = geobart_ensemble_top_k
  )
  if (length(pool) == 0L) return(tuning_df)
  scores <- vapply(pool, geobart_candidate_score_gb, numeric(1))
  weights <- geobart_ensemble_weights_gb(scores)
  for (i in seq_along(pool)) {
    idx <- tuning_df$tune_stage == geobart_candidate_stage_gb(pool[[i]]) &
      tuning_df$candidate_id == geobart_candidate_id_gb(pool[[i]])
    tuning_df$ensemble_member[idx] <- TRUE
    tuning_df$ensemble_rank[idx] <- i
    tuning_df$ensemble_weight[idx] <- weights[i]
  }
  tuning_df
}

fit_or_tune_geobart_on_split <- function(train_core, val_df, test_df,
                                         iter, protocol, model, stage,
                                         params = geobart_params,
                                         verbose = TRUE) {
  if (isTRUE(geobart_tune)) {
    tuned <- tune_geobart_on_split(train_core, val_df, test_df,
                                   params = params, verbose = verbose)
    selected_idx <- which(as.logical(tuned$tuning_scores$selected))
    selected_score <- NA_real_
	    selected_label <- NA_character_
	    selected_requested <- NA_character_
	    selected_effective <- NA_character_
	    selected_stage <- NA_character_
	    selected_policy <- NA_character_
	    selected_ntree <- NA_integer_
	    selected_n_basis <- NA_integer_
	    selected_bandwidth <- NA_real_
	    selected_k <- NA_real_
	    selected_base <- NA_real_
	    selected_power <- NA_real_
	    selected_sigdf <- NA_real_
	    selected_sigquant <- NA_real_
	    selected_raw_score <- NA_real_
	    selected_raw_rmse <- NA_real_
	    selected_raw_me <- NA_real_
		    selected_rmse_sd <- NA_real_
		    selected_me_abs <- NA_real_
		    selected_rf_inner_rmse <- NA_real_
		    selected_delta_rmse_vs_rf <- NA_real_
		    selected_rf_inner_fold_rmse <- NA_character_
		    selected_fold_rmse <- NA_character_
	    selected_fold_me <- NA_character_
	    selected_post_policy <- NA_character_
	    selected_post_requested <- NA_character_
	    selected_post_crossfit <- NA
	    selected_post_bias <- NA_real_
	    selected_post_intercept <- NA_real_
	    selected_post_slope <- NA_real_
	    if (length(selected_idx) > 0L) {
	      selected_row <- tuned$tuning_scores[selected_idx[1L], , drop = FALSE]
	      selected_score <- selected_row$selection_score
	      selected_label <- selected_row$candidate_label
	      selected_requested <- selected_row$target_transform_requested
	      selected_effective <- selected_row$target_transform_effective
	      selected_stage <- selected_row$tune_stage
	      selected_policy <- selected_row$spatial_policy
	      selected_ntree <- selected_row$ntree
	      selected_n_basis <- selected_row$n_basis
	      selected_bandwidth <- selected_row$bandwidth
	      selected_k <- selected_row$k
	      selected_base <- selected_row$base
	      selected_power <- selected_row$power
	      selected_sigdf <- selected_row$sigdf
	      selected_sigquant <- selected_row$sigquant
	      if ("raw_val_RMSE" %in% names(selected_row)) {
	        selected_raw_score <- geobart_metric_score(
	          data.frame(RMSE = selected_row$raw_val_RMSE,
	                     MAE = selected_row$raw_val_MAE,
	                     MEC = selected_row$raw_val_MEC,
	                     r2 = selected_row$raw_val_r2),
	          geobart_tune_metric)
	        selected_raw_rmse <- selected_row$raw_val_RMSE
	        selected_raw_me <- selected_row$raw_val_ME
	      }
		      if ("RMSE_sd" %in% names(selected_row)) selected_rmse_sd <- selected_row$RMSE_sd
		      if ("ME_abs" %in% names(selected_row)) selected_me_abs <- selected_row$ME_abs
		      if ("rf_inner_RMSE" %in% names(selected_row)) {
		        selected_rf_inner_rmse <- selected_row$rf_inner_RMSE
		      }
		      if ("delta_RMSE_vs_RF" %in% names(selected_row)) {
		        selected_delta_rmse_vs_rf <- selected_row$delta_RMSE_vs_RF
		      }
		      if ("rf_inner_fold_RMSE" %in% names(selected_row)) {
		        selected_rf_inner_fold_rmse <- selected_row$rf_inner_fold_RMSE
		      }
		      if ("fold_RMSE" %in% names(selected_row)) selected_fold_rmse <- selected_row$fold_RMSE
	      if ("fold_ME" %in% names(selected_row)) selected_fold_me <- selected_row$fold_ME
	      if ("post_calibration_policy" %in% names(selected_row)) {
	        selected_post_policy <- selected_row$post_calibration_policy
	        selected_post_requested <- selected_row$post_calibration_requested_policy
	        selected_post_crossfit <- selected_row$post_calibration_crossfit
	        selected_post_bias <- selected_row$post_calibration_bias
	        selected_post_intercept <- selected_row$post_calibration_intercept
	        selected_post_slope <- selected_row$post_calibration_slope
	      }
	    }
	    tuned$selected_out$diagnostics$tuning_candidates <- nrow(tuned$tuning_scores)
	    tuned$selected_out$diagnostics$tuning_inner_folds <- geobart_tune_inner_folds
	    tuned$selected_out$diagnostics$tuning_num_workers <- geobart_num_workers
	    tuned$selected_out$diagnostics$tuning_autotune_v2 <- geobart_autotune_v2
	    tuned$selected_out$diagnostics$tuning_autotune_v3 <- geobart_autotune_v3
	    tuned$selected_out$diagnostics$tuning_autotune_v4 <- geobart_autotune_v4
	    tuned$selected_out$diagnostics$tuning_metric <- tuned$selection_metric
	    tuned$selected_out$diagnostics$tuning_selected_score <- selected_score
	    tuned$selected_out$diagnostics$tuning_selected_raw_score <- selected_raw_score
	    tuned$selected_out$diagnostics$tuning_selected_raw_RMSE <- selected_raw_rmse
	    tuned$selected_out$diagnostics$tuning_selected_raw_ME <- selected_raw_me
		    tuned$selected_out$diagnostics$tuning_selected_RMSE_sd <- selected_rmse_sd
		    tuned$selected_out$diagnostics$tuning_selected_ME_abs <- selected_me_abs
		    tuned$selected_out$diagnostics$tuning_selection_objective <-
		      geobart_selection_objective
		    tuned$selected_out$diagnostics$tuning_selected_rf_inner_RMSE <-
		      selected_rf_inner_rmse
		    tuned$selected_out$diagnostics$tuning_selected_rf_inner_fold_RMSE <-
		      selected_rf_inner_fold_rmse
		    tuned$selected_out$diagnostics$tuning_selected_delta_RMSE_vs_RF <-
		      selected_delta_rmse_vs_rf
		    tuned$selected_out$diagnostics$tuning_selected_fold_RMSE <- selected_fold_rmse
	    tuned$selected_out$diagnostics$tuning_selected_fold_ME <- selected_fold_me
	    tuned$selected_out$diagnostics$tuning_selected_label <- selected_label
	    tuned$selected_out$diagnostics$tuning_selected_stage <- selected_stage
	    tuned$selected_out$diagnostics$tuning_selected_spatial_policy <- selected_policy
	    tuned$selected_out$diagnostics$tuning_selected_ntree <- selected_ntree
	    tuned$selected_out$diagnostics$tuning_selected_n_basis <- selected_n_basis
	    tuned$selected_out$diagnostics$tuning_selected_bandwidth <- selected_bandwidth
	    tuned$selected_out$diagnostics$tuning_selected_k <- selected_k
	    tuned$selected_out$diagnostics$tuning_selected_base <- selected_base
	    tuned$selected_out$diagnostics$tuning_selected_power <- selected_power
	    tuned$selected_out$diagnostics$tuning_selected_sigdf <- selected_sigdf
	    tuned$selected_out$diagnostics$tuning_selected_sigquant <- selected_sigquant
	    tuned$selected_out$diagnostics$tuning_selected_post_calibration_policy <-
	      selected_post_policy
	    tuned$selected_out$diagnostics$tuning_selected_post_calibration_requested_policy <-
	      selected_post_requested
	    tuned$selected_out$diagnostics$tuning_selected_post_calibration_crossfit <-
	      selected_post_crossfit
	    tuned$selected_out$diagnostics$tuning_selected_post_calibration_bias <-
	      selected_post_bias
	    tuned$selected_out$diagnostics$tuning_selected_post_calibration_intercept <-
	      selected_post_intercept
	    tuned$selected_out$diagnostics$tuning_selected_post_calibration_slope <-
	      selected_post_slope
	    tuned$selected_out$diagnostics$target_transform_requested <- selected_requested
	    tuned$selected_out$diagnostics$target_transform_effective <- selected_effective
    tuning <- annotate_tuning_rows_gb(
      tuned$tuning_scores, iter, protocol, model, stage,
      nrow(train_core), nrow(val_df), nrow(test_df)
    )
    tuning <- geobart_mark_ensemble_tuning_rows_gb(
      tuning, tuned$ranked_candidates)
    return(list(out = tuned$selected_out,
                params = tuned$selected_params,
                selected_candidate = tuned$selected_candidate,
                ranked_candidates = tuned$ranked_candidates,
                post_policy = tuned$post_policy,
                tuning = tuning))
  }
  out <- run_geobart_on_split(train_core, val_df, test_df,
                              params = params, verbose = verbose)
  list(out = out, params = params, selected_candidate = NULL,
       ranked_candidates = list(), post_policy = NULL, tuning = data.frame())
}

geobart_inner_duan_smearing <- function(selection_out, val_df,
                                        params = geobart_params) {
  duan_requested <- if (is.null(params$duan_apply)) TRUE else isTRUE(params$duan_apply)
  target_effective <- selection_out$diagnostics$target_transform_effective
	  if (is.null(target_effective) || length(target_effective) == 0L ||
	      is.na(target_effective[1L])) {
	    target_effective <- selection_out$diagnostics$target_transform
	  }
	  if (is.null(target_effective) || length(target_effective) == 0L ||
	      is.na(target_effective[1L])) {
	    target_effective <- params$target_transform
	  }
  target_effective <- tolower(as.character(target_effective)[1L])
  if (!isTRUE(duan_requested) || !identical(target_effective, "log")) {
    return(list(S = 1.0, applied = FALSE, reason = "inner Duan not requested"))
  }
  y_val <- val_df[[response_name_wadoux]]
  pred_val <- if (!is.null(selection_out$pred_val_raw)) {
    selection_out$pred_val_raw
  } else {
    selection_out$pred_val
  }
  ok <- is.finite(y_val) & is.finite(pred_val) & y_val > 0 & pred_val > 0
  if (!any(ok)) {
    return(list(S = 1.0, applied = FALSE, reason = "no positive inner-val pairs"))
  }
  min_val_n <- if (is.null(params$duan_min_val_n)) 500L else as.integer(params$duan_min_val_n)
  trim_frac <- if (is.null(params$duan_trim_frac)) 0.05 else as.numeric(params$duan_trim_frac)
  geobart_duan_smearing(
    log(y_val[ok]) - log(pred_val[ok]),
    trim_frac = trim_frac,
    min_val_n = min_val_n
  )
}

geobart_apply_carried_smearing <- function(out, smearing_info, y_test = NULL) {
  if (!isTRUE(smearing_info$applied)) {
    out$diagnostics$duan_carried_from_inner_val <- FALSE
    out$diagnostics$duan_carried_reason <- smearing_info$reason
    return(out)
  }
  for (nm in c("pred_test", "pred_test_lo", "pred_test_hi")) {
    if (!is.null(out[[nm]])) out[[nm]] <- out[[nm]] * smearing_info$S
  }
  out$diagnostics$duan_carried_from_inner_val <- TRUE
  out$diagnostics$duan_applied <- TRUE
  out$diagnostics$duan_S <- smearing_info$S
  out$diagnostics$duan_S_raw <- if (is.null(smearing_info$S_raw)) NA_real_ else smearing_info$S_raw
  out$diagnostics$duan_n_used <- if (is.null(smearing_info$n_used)) 0L else smearing_info$n_used
  out$diagnostics$duan_reason <- paste("carried_inner_val:", smearing_info$reason)
  if (!is.null(y_test) && !is.null(out$pred_test) &&
      length(y_test) == length(out$pred_test)) {
    out$metrics_test <- data.frame(
      N = length(y_test),
      R2 = if (length(y_test) >= 2) stats::cor(y_test, out$pred_test) ^ 2 else NA_real_,
      RMSE = sqrt(mean((y_test - out$pred_test) ^ 2, na.rm = TRUE)),
      MAE = mean(abs(y_test - out$pred_test), na.rm = TRUE),
      Bias = mean(out$pred_test - y_test, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  out
}

geobart_refit_monitor_gb <- function(train_df) {
  if (nrow(train_df) == 0L) stop("GeoBART refit monitor requires training rows.")
  train_df[1L, , drop = FALSE]
}

geobart_refit_carry_diag_names_gb <- function() {
  c(
    "tuning_candidates", "tuning_inner_folds", "tuning_num_workers",
    "tuning_autotune_v2", "tuning_autotune_v3", "tuning_autotune_v4",
    "tuning_metric",
    "tuning_selected_score", "tuning_selected_raw_score",
	    "tuning_selected_raw_RMSE", "tuning_selected_raw_ME",
	    "tuning_selected_RMSE_sd", "tuning_selected_ME_abs",
	    "tuning_selection_objective", "tuning_selected_rf_inner_RMSE",
		    "tuning_selected_rf_inner_fold_RMSE",
		    "tuning_selected_delta_RMSE_vs_RF",
		    "tuning_noharm_policy",
		    "tuning_noharm_preselected_candidate_id",
		    "tuning_noharm_preselected_tune_stage",
		    "tuning_noharm_anchor_candidate_id",
		    "tuning_noharm_anchor_tune_stage",
		    "tuning_noharm_anchor_candidate_label",
		    "tuning_noharm_anchor_score",
		    "tuning_noharm_anchor_val_rmse",
		    "tuning_noharm_margin_abs",
		    "tuning_noharm_margin_rel",
		    "tuning_noharm_margin",
		    "tuning_noharm_delta_vs_anchor",
			    "tuning_noharm_applied",
			    "tuning_noharm_restrict_ensemble",
			    "tuning_noharm_reason",
			    "tuning_rf_noharm_policy",
			    "tuning_rf_noharm_preselected_candidate_id",
			    "tuning_rf_noharm_preselected_tune_stage",
			    "tuning_rf_noharm_preselected_delta_RMSE_vs_RF",
			    "tuning_rf_noharm_anchor_candidate_id",
			    "tuning_rf_noharm_anchor_tune_stage",
			    "tuning_rf_noharm_anchor_candidate_label",
			    "tuning_rf_noharm_anchor_delta_RMSE_vs_RF",
			    "tuning_rf_noharm_margin_abs",
			    "tuning_rf_noharm_margin_rel",
			    "tuning_rf_noharm_margin",
			    "tuning_rf_noharm_required_delta_RMSE_vs_RF",
			    "tuning_rf_noharm_failed",
			    "tuning_rf_noharm_applied",
			    "tuning_rf_noharm_restrict_ensemble",
			    "tuning_rf_noharm_reason",
			    "tuning_selected_quantile_RMSE_mean",
    "tuning_selected_quantile_RMSE_sd",
    "tuning_selected_quantile_RMSE_excess",
    "tuning_selected_quantile_RMSE",
    "tuning_selected_quantile_bins",
    "tuning_selected_fold_RMSE", "tuning_selected_fold_ME",
    "tuning_selected_label", "tuning_selected_stage",
    "tuning_selected_spatial_policy", "tuning_selected_ntree",
    "tuning_selected_n_basis", "tuning_selected_bandwidth",
    "tuning_selected_k", "tuning_selected_base", "tuning_selected_power",
    "tuning_selected_sigdf", "tuning_selected_sigquant",
    "tuning_selected_post_calibration_policy",
    "tuning_selected_post_calibration_requested_policy",
    "tuning_selected_post_calibration_crossfit",
    "tuning_selected_post_calibration_bias",
    "tuning_selected_post_calibration_intercept",
    "tuning_selected_post_calibration_slope",
    "tuning_stage2_select_only", "tuning_stage2_seed_ensemble_n",
    "tuning_rmse_sd_weight", "tuning_abs_me_weight",
    "tuning_quantile_bins", "tuning_quantile_excess_weight",
    "tuning_quantile_sd_weight",
    "target_transform_requested", "target_transform_effective"
  )
}

geobart_candidate_effective_transform_gb <- function(candidate_best,
                                                    params = geobart_params) {
  tt <- candidate_best$effective_transform
  if (is.null(tt) || length(tt) == 0L || is.na(tt[1L])) {
    if (!is.null(candidate_best$params$target_transform)) {
      tt <- candidate_best$params$target_transform
    }
  }
  if (is.null(tt) || length(tt) == 0L || is.na(tt[1L])) {
    tt <- params$target_transform
  }
  tolower(as.character(tt[1L]))
}

geobart_candidate_inner_smearing_gb <- function(candidate_best, selection,
                                                inner_train_df, inner_val_df,
                                                test_df,
                                                params = geobart_params,
                                                member_index = 1L,
                                                verbose = TRUE) {
  duan_requested <- if (is.null(params$duan_apply)) TRUE else isTRUE(params$duan_apply)
  target_effective <- geobart_candidate_effective_transform_gb(
    candidate_best, params = params)
  if (!isTRUE(duan_requested) || !identical(target_effective, "log")) {
    return(list(S = 1.0, applied = FALSE,
                reason = "inner Duan not requested"))
  }

  inner_out <- candidate_best$out
  if (is.null(inner_out) || is.null(inner_out$pred_val)) {
    if (is.null(inner_train_df) || nrow(inner_train_df) == 0L) {
      return(list(S = 1.0, applied = FALSE,
                  reason = "missing inner train rows for candidate Duan"))
    }
    inner_params <- candidate_best$params
    if (is.null(inner_params) || length(inner_params) == 0L) {
      return(list(S = 1.0, applied = FALSE,
                  reason = "missing candidate params for candidate Duan"))
    }
    inner_params$duan_apply <- FALSE
    inner_args <- inner_params[
      intersect(names(inner_params),
                setdiff(names(formals(train_geobart_one_fold)),
                        c("fd", "...", "verbose")))
    ]
    inner_seed <- if (is.null(inner_args$train_seed) ||
                      !is.finite(inner_args$train_seed)) {
      train_seed
    } else {
      as.integer(inner_args$train_seed)
    }
    candidate_id <- geobart_candidate_id_gb(candidate_best)
    if (is.finite(candidate_id)) {
      inner_seed <- inner_seed + 37L * candidate_id
    }
    inner_seed <- inner_seed + 1009L * as.integer(member_index)
    inner_args$train_seed <- NULL
    fd_inner <- prepare_geobart_fold(inner_train_df, inner_val_df, test_df)
    inner_out <- do.call(
      train_geobart_one_fold,
      c(list(fd = fd_inner, train_seed = inner_seed, verbose = FALSE),
        inner_args)
    )
  }
  geobart_inner_duan_smearing(inner_out, inner_val_df, params = params)
}

geobart_refit_one_candidate_gb <- function(candidate_best, selection,
                                           calibration_full, test_df,
                                           inner_train_df, inner_val_df,
                                           params = geobart_params,
                                           member_index = 1L,
                                           verbose = TRUE) {
  tryCatch({
    refit_params <- candidate_best$params
    if (is.null(refit_params) || length(refit_params) == 0L) {
      refit_params <- selection$params
    }
    inner_smearing <- geobart_candidate_inner_smearing_gb(
      candidate_best, selection, inner_train_df, inner_val_df, test_df,
      params = params, member_index = member_index, verbose = verbose)
    refit_params$duan_apply <- FALSE
    monitor_val <- geobart_refit_monitor_gb(calibration_full)
    fd_refit <- prepare_geobart_fold(calibration_full, monitor_val, test_df)
    refit_args <- refit_params[
      intersect(names(refit_params),
                setdiff(names(formals(train_geobart_one_fold)),
                        c("fd", "...", "verbose")))
    ]
    refit_seed <- if (is.null(refit_args$train_seed) ||
                      !is.finite(refit_args$train_seed)) {
      train_seed
    } else {
      as.integer(refit_args$train_seed)
    }
    candidate_id <- geobart_candidate_id_gb(candidate_best)
    if (is.finite(candidate_id)) {
      refit_seed <- refit_seed + 53L * candidate_id
    }
    refit_seed <- refit_seed + 2003L * as.integer(member_index)
    refit_args$train_seed <- NULL
    out <- do.call(
      train_geobart_seed_ensemble,
      c(list(fd = fd_refit,
             n_seeds = geobart_refit_n_seeds,
             train_seed = refit_seed,
             verbose = verbose),
        refit_args)
    )
    out <- geobart_apply_carried_smearing(
      out, inner_smearing, y_test = test_df[[response_name_wadoux]])
    if (isTRUE(geobart_autotune_v3) &&
        !is.null(candidate_best$post_calibration)) {
      out <- geobart_apply_post_calibration_out(
        out, candidate_best$post_calibration,
        y_val = fd_refit$y$val,
        y_test = test_df[[response_name_wadoux]]
      )
      out$diagnostics$post_calibration_carried_from_inner <- TRUE
    } else {
      out$diagnostics$post_calibration_carried_from_inner <- FALSE
    }
    if (!is.null(selection$out$diagnostics)) {
      for (nm in intersect(geobart_refit_carry_diag_names_gb(),
                           names(selection$out$diagnostics))) {
        out$diagnostics[[nm]] <- selection$out$diagnostics[[nm]]
      }
    }
    out$diagnostics$refit_full_calibration <- TRUE
    out$diagnostics$refit_monitor_overlap <- TRUE
    out$diagnostics$refit_seed_ensemble_n <- geobart_refit_n_seeds
    out$diagnostics$refit_train_n <- nrow(calibration_full)
    out$diagnostics$refit_monitor_n <- nrow(monitor_val)
    out$diagnostics$refit_member_index <- as.integer(member_index)
    out$diagnostics$refit_member_candidate_id <- candidate_id
    out$diagnostics$refit_member_candidate_label <-
      geobart_candidate_label_gb(candidate_best)
    out$diagnostics$refit_member_tune_stage <-
      geobart_candidate_stage_gb(candidate_best)
    out$diagnostics$refit_member_selection_score <-
      geobart_candidate_score_gb(candidate_best)
    list(status = "ok", out = out, candidate = candidate_best,
         score = geobart_candidate_score_gb(candidate_best),
         label = geobart_candidate_label_gb(candidate_best),
         stage = geobart_candidate_stage_gb(candidate_best),
         candidate_id = candidate_id,
         post_policy = out$diagnostics$post_calibration_policy)
  }, error = function(e) {
    list(status = paste("failed:", conditionMessage(e)),
         out = NULL, candidate = candidate_best,
         score = geobart_candidate_score_gb(candidate_best),
         label = geobart_candidate_label_gb(candidate_best),
         stage = geobart_candidate_stage_gb(candidate_best),
         candidate_id = geobart_candidate_id_gb(candidate_best),
         post_policy = NA_character_)
  })
}

geobart_weighted_member_vector_gb <- function(member_results, weights, name) {
  vals <- lapply(member_results, function(x) x$out[[name]])
  ok <- vapply(vals, function(x) !is.null(x) && length(x) == length(vals[[1L]]),
               logical(1))
  if (!all(ok)) return(member_results[[1L]]$out[[name]])
  mat <- do.call(cbind, lapply(vals, as.numeric))
  as.numeric(mat %*% as.numeric(weights))
}

geobart_refit_full_calibration_gb <- function(selection, calibration_full,
                                              test_df, inner_val_df,
                                              inner_train_df = NULL,
                                              params = geobart_params,
                                              verbose = TRUE) {
  candidates <- geobart_candidate_pool_gb(selection, geobart_ensemble_top_k)
  if (length(candidates) == 0L) {
    out <- selection$out
    out$diagnostics$refit_full_calibration <- FALSE
    out$diagnostics$refit_status <- "failed:no_candidate_pool"
    return(out)
  }
  scores <- vapply(candidates, geobart_candidate_score_gb, numeric(1))
  requested_weights <- geobart_ensemble_weights_gb(scores)
  member_results <- lapply(seq_along(candidates), function(i) {
    geobart_refit_one_candidate_gb(
      candidates[[i]], selection, calibration_full, test_df,
      inner_train_df, inner_val_df,
      params = params, member_index = i, verbose = verbose
    )
  })
  ok <- vapply(member_results, function(x) identical(x$status, "ok"),
               logical(1))
  if (!any(ok)) {
    out <- selection$out
    out$diagnostics$refit_full_calibration <- FALSE
    out$diagnostics$refit_status <- paste(
      "failed:", paste(vapply(member_results, `[[`, character(1), "status"),
                       collapse = " | "))
    return(out)
  }
  member_results_ok <- member_results[ok]
  weights <- requested_weights[ok]
  weights <- weights / sum(weights, na.rm = TRUE)
  out <- member_results_ok[[1L]]$out
  if (length(member_results_ok) > 1L) {
    for (nm in c("pred_test", "pred_val", "pred_test_base",
                 "pred_val_base", "pred_test_lo", "pred_test_hi",
                 "pred_val_lo", "pred_val_hi")) {
      if (!is.null(out[[nm]])) {
        out[[nm]] <- geobart_weighted_member_vector_gb(
          member_results_ok, weights, nm)
      }
    }
  }
  y_test <- test_df[[response_name_wadoux]]
  if (!is.null(y_test) && !is.null(out$pred_test) &&
      length(y_test) == length(out$pred_test)) {
    out$metrics_test <- data.frame(
      N = length(y_test),
      R2 = if (length(y_test) >= 2L) {
        suppressWarnings(stats::cor(y_test, out$pred_test) ^ 2)
      } else {
        NA_real_
      },
      RMSE = sqrt(mean((y_test - out$pred_test) ^ 2, na.rm = TRUE)),
      MAE = mean(abs(y_test - out$pred_test), na.rm = TRUE),
      Bias = mean(out$pred_test - y_test, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  status_vec <- vapply(member_results, `[[`, character(1), "status")
  label_vec <- vapply(member_results_ok, `[[`, character(1), "label")
  stage_vec <- vapply(member_results_ok, `[[`, character(1), "stage")
  id_vec <- vapply(member_results_ok, `[[`, integer(1), "candidate_id")
  policy_vec <- vapply(member_results_ok, function(x) {
    if (is.null(x$post_policy) || length(x$post_policy) == 0L ||
        is.na(x$post_policy[1L])) {
      NA_character_
    } else {
      as.character(x$post_policy[1L])
    }
  }, character(1))
  out$diagnostics$refit_full_calibration <- TRUE
  out$diagnostics$refit_monitor_overlap <- TRUE
  out$diagnostics$refit_status <- if (all(ok)) "ok" else "partial_ok"
  out$diagnostics$refit_mode <- if (length(member_results_ok) > 1L) {
    "topk_candidate_ensemble"
  } else {
    "single_candidate_refit"
  }
  out$diagnostics$refit_ensemble_top_k_requested <- geobart_ensemble_top_k
  out$diagnostics$refit_ensemble_applied <- length(member_results_ok) > 1L
  out$diagnostics$refit_ensemble_size <- length(member_results_ok)
  out$diagnostics$refit_ensemble_weighting <- geobart_ensemble_weighting
  out$diagnostics$refit_ensemble_temperature <- geobart_ensemble_temperature
  out$diagnostics$refit_ensemble_members <- paste(label_vec, collapse = " || ")
  out$diagnostics$refit_ensemble_member_ids <- paste(id_vec, collapse = "|")
  out$diagnostics$refit_ensemble_member_stages <- paste(stage_vec, collapse = "|")
  out$diagnostics$refit_ensemble_scores <- paste(
    sprintf("%.8f", vapply(member_results_ok, `[[`, numeric(1), "score")),
    collapse = "|")
  out$diagnostics$refit_ensemble_weights <- paste(
    sprintf("%.8f", weights), collapse = "|")
  out$diagnostics$refit_ensemble_status <- paste(status_vec, collapse = " || ")
  out$diagnostics$refit_ensemble_post_policies <- paste(policy_vec, collapse = "|")
  out$diagnostics$refit_seed_ensemble_n <- geobart_refit_n_seeds
  out$diagnostics$refit_train_n <- nrow(calibration_full)
  out$diagnostics$refit_monitor_n <- 1L
  if (length(member_results_ok) > 1L) {
    out$diagnostics$post_calibration_policy <- "candidate_specific_ensemble"
    out$diagnostics$post_calibration_carried_from_inner <- TRUE
  }
  out
}

geobart_save_variogram_gb <- function(train_df, iter, protocol, model, stage) {
  if (!isTRUE(geobart_save_variograms)) return(data.frame())
  out_dir <- file.path(results_dir, "geobart_semivariograms")
  prefix <- sprintf("iter%03d_%s_%s_%s", iter, protocol, model, stage)
  title <- sprintf("GeoBART semivariogram | iter %d | %s | %s",
                   iter, model, stage)
  res <- tryCatch(
    geobart_save_variogram_diagnostic(
      y = train_df[[response_name_wadoux]],
      coords = as.matrix(train_df[, c("x", "y"), drop = FALSE]),
      out_dir = out_dir,
      prefix = prefix,
      title = title
    ),
    error = function(e) {
      data.frame(
        range_major = NA_real_,
        nugget_ratio = NA_real_,
        fit_quality = NA_real_,
        area = NA_real_,
        variogram_status = paste("failed:", conditionMessage(e)),
        variogram_transform = NA_character_,
        variogram_shapiro_p = NA_real_,
        variogram_cutoff = NA_real_,
        variogram_n_lags = NA_integer_,
        variogram_model = NA_character_,
        variogram_nugget = NA_real_,
        variogram_partial_sill = NA_real_,
        variogram_total_sill = NA_real_,
        empirical_csv = NA_character_,
        model_csv = NA_character_,
        fitted_line_csv = NA_character_,
        plot_png = NA_character_,
        stringsAsFactors = FALSE
      )
    }
  )
  cbind(
    data.frame(iteration = iter, scenario = scenario, protocol = protocol,
               model = model, stage = stage, train_n = nrow(train_df),
               stringsAsFactors = FALSE),
    res
  )
}

# ---- Cubist wrapper ---------------------------------------------------------

fit_predict_cubist_gb <- function(X_train, y_train, X_test,
                                  committees = 10L, neighbors = 0L) {
  if (!requireNamespace("Cubist", quietly = TRUE)) {
    return(rep(NA_real_, nrow(X_test)))
  }
  cb <- Cubist::cubist(x = as.data.frame(X_train), y = y_train,
                       committees = committees)
  preds <- predict(cb, newdata = as.data.frame(X_test),
                   neighbors = neighbors)
  as.numeric(preds)
}

# ---- Metrics bookkeeping ----------------------------------------------------

emit_metric_rows_gb <- function(obs, pred_map, protocol, iter, scenario) {
  rows <- list()
  for (m in names(pred_map)) {
    p <- pred_map[[m]]
    if (is.null(p) || length(p) == 0L) next
    e <- wadoux_eval(obs = obs, pred = p)
    rows[[length(rows) + 1L]] <- data.frame(
      iteration = iter, scenario = scenario, protocol = protocol,
      model = m, ME = e$ME, RMSE = e$RMSE, r2 = e$r2, MEC = e$MEC,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}

bind_rows_fill_gb <- function(rows) {
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

scalar_diag_row_gb <- function(diag, iter, protocol, model, stage,
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

annotate_tuning_rows_gb <- function(tuning_df, iter, protocol, model, stage,
                                    train_n, val_n, test_n) {
  if (is.null(tuning_df) || nrow(tuning_df) == 0L) return(data.frame())
  prefix <- data.frame(
    iteration = iter, scenario = scenario, protocol = protocol,
    model = model, stage = stage,
    train_n = train_n, val_n = val_n, test_n = test_n,
    stringsAsFactors = FALSE
  )
  cbind(prefix[rep(1L, nrow(tuning_df)), , drop = FALSE], tuning_df)
}

paired_delta_rows_gb <- function(metric_df) {
  if (is.null(metric_df) || nrow(metric_df) == 0L) return(data.frame())
  key_cols <- c("iteration", "scenario", "protocol")
  pairs <- list(
    c("GeoBART_400", "RF_400"),
    c("GeoBART_500", "RF_500"),
    c("Cubist_400", "RF_400"),
    c("Cubist_500", "RF_500"),
    c("GeoBART_500", "GeoBART_400"),
    c("RF_500", "RF_400")
  )
  split_key <- interaction(metric_df[, key_cols], drop = TRUE)
  groups <- split(metric_df, split_key)
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
  bind_rows_fill_gb(rows)
}

row_id_gb <- function(df) {
  if ("..wadoux_row_id" %in% names(df)) {
    df$..wadoux_row_id
  } else {
    seq_len(nrow(df))
  }
}

coord_key_gb <- function(df) {
  if (!all(c("x", "y") %in% names(df))) {
    return(rep(NA_character_, nrow(df)))
  }
  paste0(round(as.numeric(df$x), 6), "|", round(as.numeric(df$y), 6))
}

sample_design_test_gb <- function(calibration_full, size) {
  pool <- common_data$s_df
  keep <- rep(TRUE, nrow(pool))

  if ("..wadoux_row_id" %in% names(pool) &&
      "..wadoux_row_id" %in% names(calibration_full)) {
    keep <- keep & !(pool$..wadoux_row_id %in% calibration_full$..wadoux_row_id)
  }

  pool_xy <- coord_key_gb(pool)
  cal_xy  <- coord_key_gb(calibration_full)
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

prediction_audit_designbased_gb <- function(test_df, pred_map, iter,
                                            diag_400 = NULL,
                                            diag_500 = NULL) {
  if (!isTRUE(geobart_save_prediction_audit)) return(data.frame())
  obs <- test_df[[response_name_wadoux]]
  out <- data.frame(
    iteration = iter,
    scenario = scenario,
    protocol = "DesignBased",
    split_role = "design_test",
    wadoux_row_id = row_id_gb(test_df),
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
    fields <- c("tuning_selected_label", "tuning_selected_stage",
                "tuning_autotune_v3", "tuning_autotune_v4",
                "tuning_selected_score",
	                "tuning_selected_raw_score", "tuning_selected_raw_RMSE",
	                "tuning_selected_raw_ME", "tuning_selected_RMSE_sd",
	                "tuning_selected_ME_abs", "tuning_selection_objective",
		                "tuning_selected_rf_inner_RMSE",
		                "tuning_selected_rf_inner_fold_RMSE",
		                "tuning_selected_delta_RMSE_vs_RF",
		                "tuning_noharm_policy",
		                "tuning_noharm_preselected_candidate_id",
		                "tuning_noharm_preselected_tune_stage",
		                "tuning_noharm_anchor_candidate_id",
		                "tuning_noharm_anchor_tune_stage",
		                "tuning_noharm_anchor_candidate_label",
		                "tuning_noharm_anchor_score",
		                "tuning_noharm_anchor_val_rmse",
		                "tuning_noharm_margin_abs",
		                "tuning_noharm_margin_rel",
		                "tuning_noharm_margin",
		                "tuning_noharm_delta_vs_anchor",
			                "tuning_noharm_applied",
			                "tuning_noharm_restrict_ensemble",
			                "tuning_noharm_reason",
			                "tuning_rf_noharm_policy",
			                "tuning_rf_noharm_preselected_candidate_id",
			                "tuning_rf_noharm_preselected_tune_stage",
			                "tuning_rf_noharm_preselected_delta_RMSE_vs_RF",
			                "tuning_rf_noharm_anchor_candidate_id",
			                "tuning_rf_noharm_anchor_tune_stage",
			                "tuning_rf_noharm_anchor_candidate_label",
			                "tuning_rf_noharm_anchor_delta_RMSE_vs_RF",
			                "tuning_rf_noharm_margin_abs",
			                "tuning_rf_noharm_margin_rel",
			                "tuning_rf_noharm_margin",
			                "tuning_rf_noharm_required_delta_RMSE_vs_RF",
			                "tuning_rf_noharm_failed",
			                "tuning_rf_noharm_applied",
			                "tuning_rf_noharm_restrict_ensemble",
			                "tuning_rf_noharm_reason",
			                "tuning_selected_fold_RMSE",
                "tuning_selected_fold_ME",
                "tuning_selected_quantile_RMSE_mean",
                "tuning_selected_quantile_RMSE_sd",
                "tuning_selected_quantile_RMSE_excess",
                "tuning_selected_quantile_RMSE",
                "tuning_selected_quantile_bins",
                "tuning_selected_post_calibration_policy",
                "tuning_selected_post_calibration_requested_policy",
                "tuning_selected_post_calibration_crossfit",
                "tuning_selected_post_calibration_bias",
                "tuning_selected_post_calibration_intercept",
                "tuning_selected_post_calibration_slope",
                "post_calibration_policy",
                "post_calibration_carried_from_inner",
                "post_calibration_bias", "post_calibration_intercept",
                "post_calibration_slope",
                "post_calibration_quantile_RMSE_mean",
                "post_calibration_quantile_RMSE_sd",
                "post_calibration_quantile_RMSE_excess",
                "post_calibration_quantile_RMSE",
                "post_calibration_quantile_bins",
                "tuning_stage2_select_only",
                "tuning_stage2_seed_ensemble_n",
                "refit_status", "refit_mode",
                "refit_ensemble_top_k_requested",
                "refit_ensemble_applied", "refit_ensemble_size",
                "refit_ensemble_weighting",
                "refit_ensemble_temperature",
                "refit_ensemble_members",
                "refit_ensemble_member_ids",
                "refit_ensemble_member_stages",
                "refit_ensemble_scores",
                "refit_ensemble_weights",
                "refit_ensemble_status",
                "refit_ensemble_post_policies",
                "target_transform_effective", "spatial_policy", "n_basis",
                "bandwidth", "ntree", "k", "base", "power", "sigdf",
                "sigquant", "seed_ensemble_n", "seed_ensemble_seeds")
    for (f in fields) {
      val <- diag[[f]]
      if (is.null(val) || length(val) != 1L) val <- NA
      out[[paste(prefix, f, sep = "_")]] <<- val
    }
  }
  add_diag("GeoBART_400", diag_400)
  add_diag("GeoBART_500", diag_500)
  out
}

split_audit_designbased_gb <- function(calibration_full, inner, test_df, iter) {
  if (!isTRUE(geobart_save_prediction_audit)) return(data.frame())
  mk <- function(df, role) {
    data.frame(
      iteration = iter,
      scenario = scenario,
      protocol = "DesignBased",
      split_role = role,
      wadoux_row_id = row_id_gb(df),
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

# ---- Protocol evaluators ----------------------------------------------------

eval_designbased_gb <- function(valuetable, iter) {
  val_srs <- sample_design_test_gb(valuetable, sample_size)
  obs     <- val_srs[[response_name_wadoux]]
  inner <- make_inner_val_split_gb(valuetable, val_frac = val_frac,
                                   seed = train_seed + iter)
  calibration_full <- valuetable
  pred_map <- list()
  diag_rows <- list()
  tune_rows <- list()
  vario_rows <- list()

  if ("GeoBART" %in% models && isTRUE(geobart_save_variograms)) {
    vario_rows[[length(vario_rows) + 1L]] <- geobart_save_variogram_gb(
      inner$train, iter, "DesignBased", "GeoBART_400", "inner_selection"
    )
    vario_rows[[length(vario_rows) + 1L]] <- geobart_save_variogram_gb(
      calibration_full, iter, "DesignBased", "GeoBART_500",
      "full_calibration_refit"
    )
  }

  if ("RF" %in% models) {
    rf_400 <- fit_rf_default_wadoux(inner$train, form_rf)
    pred_map$RF_400 <- predict_rf_default_wadoux(rf_400, val_srs)

    rf_500 <- fit_rf_default_wadoux(calibration_full, form_rf)
    pred_map$RF_500 <- predict_rf_default_wadoux(rf_500, val_srs)
  }
  if ("Cubist" %in% models) {
    pred_map$Cubist_400 <- fit_predict_cubist_gb(
      X_train = inner$train[, predList, drop = FALSE],
      y_train = inner$train[[response_name_wadoux]],
      X_test  = val_srs[, predList, drop = FALSE]
    )
    pred_map$Cubist_500 <- fit_predict_cubist_gb(
      X_train = calibration_full[, predList, drop = FALSE],
      y_train = calibration_full[[response_name_wadoux]],
      X_test  = val_srs[, predList, drop = FALSE]
    )
  }
  if ("GeoBART" %in% models) {
    gb_400 <- fit_or_tune_geobart_on_split(
      inner$train, inner$val, val_srs,
      iter = iter, protocol = "DesignBased",
      model = "GeoBART_400", stage = "inner_selection"
    )
    out_400 <- gb_400$out
    pred_map$GeoBART_400 <- out_400$pred_test
    if (nrow(gb_400$tuning) > 0L) {
      tune_rows[[length(tune_rows) + 1L]] <- gb_400$tuning
    }
    diag_rows[[length(diag_rows) + 1L]] <- scalar_diag_row_gb(
      out_400$diagnostics, iter, "DesignBased", "GeoBART_400",
      "inner_selection", nrow(inner$train), nrow(inner$val), nrow(val_srs)
    )

    if (isTRUE(geobart_refit_full_calibration)) {
      out_500 <- geobart_refit_full_calibration_gb(
        gb_400, calibration_full, val_srs, inner$val,
        inner_train_df = inner$train,
        params = geobart_params, verbose = TRUE
      )
      pred_map$GeoBART_500 <- out_500$pred_test
      diag_rows[[length(diag_rows) + 1L]] <- scalar_diag_row_gb(
        out_500$diagnostics, iter, "DesignBased", "GeoBART_500",
        "full_calibration_refit", nrow(calibration_full),
        out_500$diagnostics$refit_monitor_n,
        nrow(val_srs), refit_full_calibration = TRUE,
        refit_monitor_overlap = TRUE
      )
    }
  }
  metrics <- emit_metric_rows_gb(obs, pred_map, "DesignBased", iter, scenario)
  diag_400 <- if (exists("out_400", inherits = FALSE)) out_400$diagnostics else NULL
  diag_500 <- if (exists("out_500", inherits = FALSE)) out_500$diagnostics else NULL
  audit_rows <- prediction_audit_designbased_gb(
    val_srs, pred_map, iter, diag_400 = diag_400, diag_500 = diag_500)
  split_rows <- split_audit_designbased_gb(calibration_full, inner, val_srs, iter)
  list(metrics = metrics, diagnostics = bind_rows_fill_gb(diag_rows),
       tuning = bind_rows_fill_gb(tune_rows),
       variograms = bind_rows_fill_gb(vario_rows),
       paired_deltas = paired_delta_rows_gb(metrics),
       prediction_audit = audit_rows,
       split_audit = split_rows)
}

eval_randomkfold_gb <- function(valuetable, iter) {
  flds <- caret::createFolds(valuetable[[response_name_wadoux]],
                             k = random_k, list = TRUE, returnTrain = FALSE)
  preds_rf  <- rep(NA_real_, nrow(valuetable))
  preds_cb  <- rep(NA_real_, nrow(valuetable))
  preds_gb  <- rep(NA_real_, nrow(valuetable))
  tune_rows <- list()
  for (j in seq_along(flds)) {
    id   <- flds[[j]]
    trn  <- valuetable[-id, , drop = FALSE]
    val  <- valuetable[id,  , drop = FALSE]
    if ("RF" %in% models) {
      rf <- fit_rf_default_wadoux(trn, form_rf)
      preds_rf[id] <- predict_rf_default_wadoux(rf, val)
    }
    if ("Cubist" %in% models) {
      preds_cb[id] <- fit_predict_cubist_gb(
        X_train = trn[, predList, drop = FALSE],
        y_train = trn[[response_name_wadoux]],
        X_test  = val[, predList, drop = FALSE]
      )
    }
    if ("GeoBART" %in% models) {
      inner <- make_inner_val_split_gb(trn, val_frac = val_frac,
                                       seed = train_seed + iter * 1000L + j)
	      gb <- fit_or_tune_geobart_on_split(
	        inner$train, inner$val, val,
	        iter = iter, protocol = "RandomKFold",
	        model = sprintf("GeoBART_fold_%02d", j),
	        stage = "fold_inner_selection",
	        verbose = TRUE
	      )
	      out_use <- if (isTRUE(geobart_refit_full_calibration)) {
	        geobart_refit_full_calibration_gb(
	          gb, trn, val, inner$val,
	          inner_train_df = inner$train,
	          params = geobart_params, verbose = TRUE
	        )
	      } else {
	        gb$out
	      }
	      preds_gb[id] <- out_use$pred_test
      if (nrow(gb$tuning) > 0L) {
        tune_rows[[length(tune_rows) + 1L]] <- gb$tuning
      }
    }
  }
  pred_map <- list()
  if ("RF"      %in% models) pred_map$RF      <- preds_rf
  if ("Cubist"  %in% models) pred_map$Cubist  <- preds_cb
  if ("GeoBART" %in% models) pred_map$GeoBART <- preds_gb
  list(metrics = emit_metric_rows_gb(valuetable[[response_name_wadoux]],
                                     pred_map, "RandomKFold", iter, scenario),
       tuning = bind_rows_fill_gb(tune_rows))
}

eval_spatialkfold_gb <- function(valuetable, iter) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  fold_ids <- unique(spatial_folds)
  preds_rf  <- rep(NA_real_, nrow(valuetable))
  preds_cb  <- rep(NA_real_, nrow(valuetable))
  preds_gb  <- rep(NA_real_, nrow(valuetable))
  tune_rows <- list()
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
      preds_cb[id] <- fit_predict_cubist_gb(
        X_train = trn[, predList, drop = FALSE],
        y_train = trn[[response_name_wadoux]],
        X_test  = val[, predList, drop = FALSE]
      )
    }
    if ("GeoBART" %in% models) {
      inner <- make_inner_val_split_gb(
        trn, val_frac = val_frac,
        seed = train_seed + iter * 1000L + fold_ids[j])
	      gb <- fit_or_tune_geobart_on_split(
	        inner$train, inner$val, val,
	        iter = iter, protocol = "SpatialKFold",
	        model = sprintf("GeoBART_fold_%s", fold_ids[j]),
	        stage = "fold_inner_selection",
	        verbose = TRUE
	      )
	      out_use <- if (isTRUE(geobart_refit_full_calibration)) {
	        geobart_refit_full_calibration_gb(
	          gb, trn, val, inner$val,
	          inner_train_df = inner$train,
	          params = geobart_params, verbose = TRUE
	        )
	      } else {
	        gb$out
	      }
	      preds_gb[id] <- out_use$pred_test
      if (nrow(gb$tuning) > 0L) {
        tune_rows[[length(tune_rows) + 1L]] <- gb$tuning
      }
    }
  }
  pred_map <- list()
  if ("RF"      %in% models) pred_map$RF      <- preds_rf
  if ("Cubist"  %in% models) pred_map$Cubist  <- preds_cb
  if ("GeoBART" %in% models) pred_map$GeoBART <- preds_gb
  list(metrics = emit_metric_rows_gb(valuetable[[response_name_wadoux]],
                                     pred_map, "SpatialKFold", iter, scenario),
       tuning = bind_rows_fill_gb(tune_rows))
}

eval_bloocv_gb <- function(valuetable, iter) {
  nb_iteration <- bloo_groups * bloo_pixels
  gb_cap_raw <- Sys.getenv("GEOBART_BLOO_GB_PIXELS", unset = "")
  gb_cap <- if (nzchar(gb_cap_raw)) suppressWarnings(as.integer(gb_cap_raw))
            else as.integer(min(nb_iteration, bloo_groups * 10L))
  if (!is.finite(gb_cap) || gb_cap < 1L) gb_cap <- nb_iteration

  set.seed(train_seed + iter)
  ri <- val_dist_km * 1000
  gb_used <- 0L
  pred_rows <- list()
  tune_rows <- list()
  for (j in seq_len(nb_iteration)) {
    point_in_range <- FALSE
    id_focal <- NA_integer_
    while (!isTRUE(point_in_range)) {
      id_focal      <- sample.int(nrow(valuetable), size = 1L)
      focal_point   <- valuetable[id_focal, , drop = FALSE]
      training_tmp  <- valuetable[-id_focal, , drop = FALSE]
      training_tmp  <- exclude_by_radius_wadoux(training_tmp, focal_point, ri)
      if (nrow(training_tmp) < 10L) next
      if (!point_within_predictor_range_wadoux(training_tmp, focal_point,
                                               predList)) next
      point_in_range <- TRUE
    }
    focal_point   <- valuetable[id_focal, , drop = FALSE]
    training_full <- valuetable[-id_focal, , drop = FALSE]
    training_full <- exclude_by_radius_wadoux(training_full, focal_point, ri)

    pred_rf <- NA_real_; pred_cb <- NA_real_; pred_gb <- NA_real_
    if ("RF" %in% models) {
      rf      <- fit_rf_default_wadoux(training_full, form_rf)
      pred_rf <- predict_rf_default_wadoux(rf, focal_point)
    }
    if ("Cubist" %in% models) {
      pred_cb <- fit_predict_cubist_gb(
        X_train = training_full[, predList, drop = FALSE],
        y_train = training_full[[response_name_wadoux]],
        X_test  = focal_point [, predList, drop = FALSE]
      )
    }
    if ("GeoBART" %in% models && gb_used < gb_cap) {
      inner <- make_inner_val_split_gb(training_full, val_frac = val_frac,
                                       seed = train_seed + iter * 1000L + j)
	      gb <- fit_or_tune_geobart_on_split(
	        inner$train, inner$val, focal_point,
	        iter = iter, protocol = "BLOOCV",
	        model = sprintf("GeoBART_focal_%04d", j),
	        stage = "focal_inner_selection",
	        verbose = TRUE
	      )
	      out_use <- if (isTRUE(geobart_refit_full_calibration)) {
	        geobart_refit_full_calibration_gb(
	          gb, training_full, focal_point, inner$val,
	          inner_train_df = inner$train,
	          params = geobart_params, verbose = TRUE
	        )
	      } else {
	        gb$out
	      }
	      pred_gb <- out_use$pred_test[1]
      if (nrow(gb$tuning) > 0L) {
        tune_rows[[length(tune_rows) + 1L]] <- gb$tuning
      }
      gb_used <- gb_used + 1L
    }
    pred_rows[[j]] <- data.frame(
      focal_id = id_focal,
      obs      = focal_point[[response_name_wadoux]],
      pred_rf  = pred_rf,
      pred_cb  = pred_cb,
      pred_gb  = pred_gb,
      stringsAsFactors = FALSE
    )
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
    data.frame(
      iteration = iter, scenario = scenario, protocol = "BLOOCV",
      model = label,
      ME = mean(stk$ME, na.rm = TRUE),
      RMSE = mean(stk$RMSE, na.rm = TRUE),
      r2 = mean(stk$r2, na.rm = TRUE),
      MEC = mean(stk$MEC, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
  rows <- list(emit_one("RF", "pred_rf"),
               emit_one("Cubist", "pred_cb"),
               emit_one("GeoBART", "pred_gb"))
  rows <- rows[!vapply(rows, is.null, logical(1))]
  list(metrics = if (length(rows) == 0L) data.frame() else do.call(rbind, rows),
       tuning = bind_rows_fill_gb(tune_rows))
}

# ---- Save helpers -----------------------------------------------------------

save_geobart_checkpoint <- function(iter, metric_rows, diagnostic_rows = list(),
                                    tuning_rows = list(),
                                    variogram_rows = list(),
                                    prediction_audit_rows = list(),
                                    split_audit_rows = list()) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  write.csv(df, file.path(results_dir, "geobart_all_results.csv"),
            row.names = FALSE)

  deltas <- paired_delta_rows_gb(df)
  if (nrow(deltas) > 0L) {
    write.csv(deltas, file.path(results_dir, "geobart_paired_deltas_vs_RF.csv"),
              row.names = FALSE)
  }

  diag_df <- bind_rows_fill_gb(diagnostic_rows)
  if (nrow(diag_df) > 0L) {
    write.csv(diag_df, file.path(results_dir, "geobart_fold_diagnostics.csv"),
              row.names = FALSE)
  }

  tune_df <- bind_rows_fill_gb(tuning_rows)
  if (nrow(tune_df) > 0L) {
    write.csv(tune_df, file.path(results_dir, "geobart_tuning_scores.csv"),
              row.names = FALSE)
  }

	  vario_df <- bind_rows_fill_gb(variogram_rows)
	  if (nrow(vario_df) > 0L) {
	    write.csv(vario_df, file.path(results_dir, "geobart_variogram_summary.csv"),
	              row.names = FALSE)
	  }

  audit_df <- bind_rows_fill_gb(prediction_audit_rows)
  if (nrow(audit_df) > 0L) {
    write.csv(audit_df, file.path(results_dir, "geobart_prediction_audit.csv"),
              row.names = FALSE)
  }

  split_df <- bind_rows_fill_gb(split_audit_rows)
  if (nrow(split_df) > 0L) {
    write.csv(split_df, file.path(results_dir, "geobart_split_audit.csv"),
              row.names = FALSE)
  }

  write.csv(
    data.frame(
      completed_iteration = iter, total_iterations = n_iter,
      scenario = scenario, protocols = paste(protocols, collapse = ","),
      models = paste(models, collapse = ","),
      completed_at = format(Sys.time(), tz = "", usetz = TRUE),
      stringsAsFactors = FALSE),
    file.path(results_dir, "geobart_checkpoint_status.csv"),
    row.names = FALSE)
}

summarise_and_save_gb <- function(metric_rows) {
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
  write.csv(out, file.path(results_dir, "geobart_summary_by_protocol.csv"),
            row.names = FALSE)
}

# ---- Main loop --------------------------------------------------------------

write.csv(
  data.frame(
    scenario = scenario, protocols = paste(protocols, collapse = ","),
    models = paste(models, collapse = ","),
    sample_size = sample_size, n_iter = n_iter, val_frac = val_frac,
    val_dist_km = val_dist_km, random_k = random_k,
    bloo_groups = bloo_groups, bloo_test_pixels = bloo_pixels,
    train_seed = train_seed,
		    geobart_tune = geobart_tune,
			    geobart_tune_metric = geobart_tune_metric,
			    geobart_selection_objective = geobart_selection_objective,
			    geobart_noharm_policy = geobart_noharm_policy,
			    geobart_noharm_margin_abs = geobart_noharm_margin_abs,
				    geobart_noharm_margin_rel = geobart_noharm_margin_rel,
				    geobart_noharm_restrict_ensemble =
				      geobart_noharm_restrict_ensemble,
				    geobart_rf_noharm_policy = geobart_rf_noharm_policy,
				    geobart_rf_noharm_margin_abs =
				      geobart_rf_noharm_margin_abs,
				    geobart_rf_noharm_margin_rel =
				      geobart_rf_noharm_margin_rel,
				    geobart_rf_noharm_restrict_ensemble =
				      geobart_rf_noharm_restrict_ensemble,
				    geobart_autotune_v2 = geobart_autotune_v2,
	    geobart_autotune_v3 = geobart_autotune_v3,
	    geobart_autotune_v4 = geobart_autotune_v4,
	    geobart_tune_full_grid = geobart_tune_full_grid,
	    geobart_tune_max_candidates =
	      if (is.null(geobart_tune_max_candidates) ||
	          !is.finite(geobart_tune_max_candidates)) NA_integer_
	      else as.integer(geobart_tune_max_candidates),
	    geobart_tune_inner_folds = geobart_tune_inner_folds,
	    geobart_refit_full_calibration = geobart_refit_full_calibration,
	    geobart_refit_n_seeds = geobart_refit_n_seeds,
	    geobart_ensemble_top_k = geobart_ensemble_top_k,
	    geobart_ensemble_weighting = geobart_ensemble_weighting,
	    geobart_ensemble_temperature = geobart_ensemble_temperature,
	    geobart_num_workers = geobart_num_workers,
	    geobart_save_variograms = geobart_save_variograms,
	    geobart_save_prediction_audit = geobart_save_prediction_audit,
		    GEOBART_AUTOTUNE_V2 = raw_env_gb("GEOBART_AUTOTUNE_V2", "TRUE"),
			    GEOBART_SELECTION_OBJECTIVE = raw_env_gb(
			      "GEOBART_SELECTION_OBJECTIVE", "paired_rf_delta"),
			    GEOBART_NOHARM_POLICY = raw_env_gb(
			      "GEOBART_NOHARM_POLICY", "TRUE"),
			    GEOBART_NOHARM_MARGIN_ABS = raw_env_gb(
			      "GEOBART_NOHARM_MARGIN_ABS", "0.05"),
			    GEOBART_NOHARM_MARGIN_REL = raw_env_gb(
			      "GEOBART_NOHARM_MARGIN_REL", "0"),
				    GEOBART_NOHARM_RESTRICT_ENSEMBLE = raw_env_gb(
				      "GEOBART_NOHARM_RESTRICT_ENSEMBLE", "TRUE"),
				    GEOBART_RF_NOHARM_POLICY = raw_env_gb(
				      "GEOBART_RF_NOHARM_POLICY", "TRUE"),
				    GEOBART_RF_NOHARM_MARGIN_ABS = raw_env_gb(
				      "GEOBART_RF_NOHARM_MARGIN_ABS", "0.05"),
				    GEOBART_RF_NOHARM_MARGIN_REL = raw_env_gb(
				      "GEOBART_RF_NOHARM_MARGIN_REL", "0"),
				    GEOBART_RF_NOHARM_RESTRICT_ENSEMBLE = raw_env_gb(
				      "GEOBART_RF_NOHARM_RESTRICT_ENSEMBLE", "TRUE"),
				    GEOBART_AUTOTUNE_V3 = raw_env_gb("GEOBART_AUTOTUNE_V3", "TRUE"),
	    GEOBART_AUTOTUNE_V4 = raw_env_gb("GEOBART_AUTOTUNE_V4", "TRUE"),
	    GEOBART_V3_RMSE_SD_WEIGHT = raw_env_gb(
	      "GEOBART_V3_RMSE_SD_WEIGHT", "0.25"),
	    GEOBART_V3_ABS_ME_WEIGHT = raw_env_gb(
	      "GEOBART_V3_ABS_ME_WEIGHT", "0.10"),
	    GEOBART_V3_CALIBRATION_POLICIES = raw_env_gb(
	      "GEOBART_V3_CALIBRATION_POLICIES",
	      "none,mean_bias,median_bias,affine"),
	    GEOBART_V3_AFFINE_SLOPE_MIN = raw_env_gb(
	      "GEOBART_V3_AFFINE_SLOPE_MIN", "0.5"),
	    GEOBART_V3_AFFINE_SLOPE_MAX = raw_env_gb(
	      "GEOBART_V3_AFFINE_SLOPE_MAX", "1.5"),
	    GEOBART_V3_STAGE2_N_SEEDS = raw_env_gb(
	      "GEOBART_V3_STAGE2_N_SEEDS", "3"),
	    GEOBART_V3_STAGE2_SELECT_ONLY = raw_env_gb(
	      "GEOBART_V3_STAGE2_SELECT_ONLY", "FALSE"),
	    GEOBART_V3_LOG_MARGIN = raw_env_gb("GEOBART_V3_LOG_MARGIN", "0.50"),
	    GEOBART_V4_QUANTILE_BINS = raw_env_gb(
	      "GEOBART_V4_QUANTILE_BINS", "5"),
	    GEOBART_V4_QUANTILE_EXCESS_WEIGHT = raw_env_gb(
	      "GEOBART_V4_QUANTILE_EXCESS_WEIGHT", "0.15"),
	    GEOBART_V4_QUANTILE_SD_WEIGHT = raw_env_gb(
	      "GEOBART_V4_QUANTILE_SD_WEIGHT", "0.05"),
	    GEOBART_V3_STAGE2_FULL_GRID = raw_env_gb(
	      "GEOBART_V3_STAGE2_FULL_GRID", "FALSE"),
	    GEOBART_REFIT_N_SEEDS = raw_env_gb("GEOBART_REFIT_N_SEEDS", "5"),
	    GEOBART_ENSEMBLE_TOP_K = raw_env_gb(
	      "GEOBART_ENSEMBLE_TOP_K",
	      if (isTRUE(geobart_autotune_v4)) "5" else "1"),
	    GEOBART_ENSEMBLE_WEIGHTING = raw_env_gb(
	      "GEOBART_ENSEMBLE_WEIGHTING", "softmax"),
	    GEOBART_ENSEMBLE_TEMPERATURE = raw_env_gb(
	      "GEOBART_ENSEMBLE_TEMPERATURE", "0.75"),
	    GEOBART_SAVE_PREDICTION_AUDIT = raw_env_gb(
	      "GEOBART_SAVE_PREDICTION_AUDIT", "TRUE"),
	    GEOBART_TUNE_STAGE1_MAX_CANDIDATES = raw_env_gb(
	      "GEOBART_TUNE_STAGE1_MAX_CANDIDATES", "240"),
	    GEOBART_TUNE_STAGE2_TOP_K = raw_env_gb("GEOBART_TUNE_STAGE2_TOP_K", "8"),
	    GEOBART_TUNE_STAGE2_MAX_CANDIDATES = raw_env_gb(
	      "GEOBART_TUNE_STAGE2_MAX_CANDIDATES", "240"),
	    GEOBART_TUNE_STAGE1_SPATIAL_POLICIES = raw_env_gb(
	      "GEOBART_TUNE_STAGE1_SPATIAL_POLICIES",
	      "coords_basis,coords_only,basis_only,tabular_only,coords_aniso_basis"),
	    GEOBART_TUNE_STAGE2_K_VALUES = raw_env_gb(
	      "GEOBART_TUNE_STAGE2_K_VALUES", "1,1.5,2,3"),
	    GEOBART_TUNE_STAGE2_BASE_VALUES = raw_env_gb(
	      "GEOBART_TUNE_STAGE2_BASE_VALUES", "0.85,0.95,0.99"),
	    GEOBART_TUNE_STAGE2_POWER_VALUES = raw_env_gb(
	      "GEOBART_TUNE_STAGE2_POWER_VALUES", "1.5,2,2.5"),
	    GEOBART_TUNE_STAGE2_SIGDF_VALUES = raw_env_gb(
	      "GEOBART_TUNE_STAGE2_SIGDF_VALUES", "3,5,10"),
	    GEOBART_TUNE_STAGE2_SIGQUANT_VALUES = raw_env_gb(
	      "GEOBART_TUNE_STAGE2_SIGQUANT_VALUES", "0.75,0.90,0.99"),
	    GEOBART_TUNE_NTREE_VALUES = raw_env_gb(
      "GEOBART_TUNE_NTREE_VALUES", "50,60,90,120,180"),
    GEOBART_TUNE_NTREE_MULTS = raw_env_gb("GEOBART_TUNE_NTREE_MULTS", ""),
    GEOBART_TUNE_NTREE_MIN = raw_env_gb("GEOBART_TUNE_NTREE_MIN", "50"),
    GEOBART_TUNE_NTREE_MAX = raw_env_gb("GEOBART_TUNE_NTREE_MAX", "300"),
    GEOBART_TUNE_N_BASIS_VALUES = raw_env_gb(
      "GEOBART_TUNE_N_BASIS_VALUES", ""),
    GEOBART_TUNE_BASIS_MULTS = raw_env_gb(
      "GEOBART_TUNE_BASIS_MULTS", "0.5,1,1.5,2,2.5"),
    GEOBART_TUNE_N_BASIS_MAX = raw_env_gb(
      "GEOBART_TUNE_N_BASIS_MAX", "max(80,round(auto*2.5))"),
    GEOBART_TUNE_BANDWIDTH_VALUES = raw_env_gb(
      "GEOBART_TUNE_BANDWIDTH_VALUES", ""),
    GEOBART_TUNE_BANDWIDTH_MULTS = raw_env_gb(
      "GEOBART_TUNE_BANDWIDTH_MULTS", "0.25,0.5,1,2,4"),
    GEOBART_TUNE_TRANSFORMS = raw_env_gb(
      "GEOBART_TUNE_TRANSFORMS", "auto,identity,log"),
    GEOBART_TUNE_INCLUDE_NO_BASIS = raw_env_gb(
      "GEOBART_TUNE_INCLUDE_NO_BASIS", "FALSE"),
    GEOBART_TUNE_FULL_GRID = raw_env_gb("GEOBART_TUNE_FULL_GRID", "TRUE"),
    GEOBART_TUNE_MAX_CANDIDATES = raw_env_gb(
      "GEOBART_TUNE_MAX_CANDIDATES", ""),
	    GEOBART_TUNE_INNER_FOLDS = raw_env_gb("GEOBART_TUNE_INNER_FOLDS", "3"),
	    GEOBART_REFIT_FULL_CALIBRATION = raw_env_gb(
	      "GEOBART_REFIT_FULL_CALIBRATION", "TRUE"),
	    GEOBART_NUM_WORKERS = raw_env_gb("GEOBART_NUM_WORKERS", "1"),
	    GEOBART_SAVE_VARIOGRAMS = raw_env_gb("GEOBART_SAVE_VARIOGRAMS", "TRUE"),
	    stringsAsFactors = FALSE),
  file.path(results_dir, "geobart_config.csv"),
  row.names = FALSE)

all_metric_rows <- list()
all_diag_rows <- list()
all_tune_rows <- list()
all_vario_rows <- list()
all_prediction_audit_rows <- list()
all_split_audit_rows <- list()
for (iter in seq_len(n_iter)) {
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("GeoBART benchmark | scenario=%s | iter %d / %d\n",
              scenario, iter, n_iter))
  cat(sprintf("==================================================\n"))

  valuetable <- na.omit(as.data.frame(sample_calibration_gb()))

  iter_rows <- list()
	  iter_diag <- list()
	  iter_tune <- list()
	  iter_vario <- list()
	  iter_prediction_audit <- list()
	  iter_split_audit <- list()
	  if ("DesignBased"  %in% protocols) {
	    cat("\n--- DesignBased ---\n")
	    out <- eval_designbased_gb(valuetable, iter)
	    iter_rows[["DesignBased"]] <- out$metrics
	    iter_diag[["DesignBased"]] <- out$diagnostics
	    iter_tune[["DesignBased"]] <- out$tuning
	    iter_vario[["DesignBased"]] <- out$variograms
	    iter_prediction_audit[["DesignBased"]] <- out$prediction_audit
	    iter_split_audit[["DesignBased"]] <- out$split_audit
	  }
  if ("RandomKFold"  %in% protocols) {
    cat("\n--- RandomKFold ---\n")
    out <- eval_randomkfold_gb(valuetable, iter)
    iter_rows[["RandomKFold"]] <- out$metrics
    iter_tune[["RandomKFold"]] <- out$tuning
  }
  if ("SpatialKFold" %in% protocols) {
    cat("\n--- SpatialKFold ---\n")
    out <- eval_spatialkfold_gb(valuetable, iter)
    iter_rows[["SpatialKFold"]] <- out$metrics
    iter_tune[["SpatialKFold"]] <- out$tuning
  }
  if ("BLOOCV"       %in% protocols) {
    cat("\n--- BLOOCV ---\n")
    out <- eval_bloocv_gb(valuetable, iter)
    iter_rows[["BLOOCV"]] <- out$metrics
    iter_tune[["BLOOCV"]] <- out$tuning
  }

  iter_df <- do.call(rbind, iter_rows[!vapply(iter_rows, is.null, logical(1))])
  if (!is.null(iter_df) && nrow(iter_df) > 0L) {
    print(iter_df)
    all_metric_rows[[iter]] <- iter_df
  }
  iter_diag_df <- bind_rows_fill_gb(iter_diag)
  if (nrow(iter_diag_df) > 0L) {
    all_diag_rows[[iter]] <- iter_diag_df
  }
  iter_tune_df <- bind_rows_fill_gb(iter_tune)
  if (nrow(iter_tune_df) > 0L) {
    all_tune_rows[[iter]] <- iter_tune_df
  }
	  iter_vario_df <- bind_rows_fill_gb(iter_vario)
	  if (nrow(iter_vario_df) > 0L) {
	    all_vario_rows[[iter]] <- iter_vario_df
	  }
	  iter_prediction_audit_df <- bind_rows_fill_gb(iter_prediction_audit)
	  if (nrow(iter_prediction_audit_df) > 0L) {
	    all_prediction_audit_rows[[iter]] <- iter_prediction_audit_df
	  }
	  iter_split_audit_df <- bind_rows_fill_gb(iter_split_audit)
	  if (nrow(iter_split_audit_df) > 0L) {
	    all_split_audit_rows[[iter]] <- iter_split_audit_df
	  }

	  if (iter %% checkpoint_every == 0L || iter == n_iter) {
	    save_geobart_checkpoint(iter, all_metric_rows, all_diag_rows,
	                            all_tune_rows, all_vario_rows,
	                            all_prediction_audit_rows,
	                            all_split_audit_rows)
    summarise_and_save_gb(all_metric_rows)
  }
}

cat("\n[GeoBART runner] Done. Results written to: ", results_dir, "\n",
    sep = "")
