# =============================================================================
# run_dbtarf_wadoux_comparison.R
#
# Head-to-head benchmark under Wadoux (2021) DesignBased protocol:
#
#   - RF       (ranger defaults, Wadoux baseline)
#   - Cubist   (optional classical DSM comparator)
#   - DBTARF   (Design-Based Targeted Adaptive Residual Forest)
#
# Environment overrides:
#   DBT_SCENARIO            random | regular_grid | clustered_random
#   DBT_PROTOCOLS           comma list of {Population, DesignBased, SpatialKFold}
#   DBT_VAL_DIST_KM         SpatialKFold exclusion distance (km), default 350
#   DBT_FOLD_CORES          fork-parallel SpatialKFold folds, default 1 (serial)
#   DBT_MODELS              comma list of {RF, Cubist, DBTARF}
#   DBT_N_ITER              default 5
#   DBT_SAMPLE_SIZE         default 500
#   DBT_N_FOLDS             DBTARF internal cross-fitting folds, default 5
#   DBT_LAMBDA_GRID         comma list, e.g. "-0.25,0,0.25,0.5"
#   DBT_MIN_OOF_IMPROVEMENT default 0
#   DBT_RF_TUNE             TRUE/FALSE, default TRUE
#   DBT_RF_TUNE_METHOD      oob | cv | none, default oob
#   DBT_RF_TUNE_INNER_FOLDS default 3, used when DBT_RF_TUNE_METHOD=cv
#   DBT_RF_TUNE_MAX_GRID    max RF base candidates, default 18
#   DBT_RF_NUM_TREES        optional comma list, e.g. "500,1000,2000,3000"
#   DBT_RF_MTRY             optional comma list
#   DBT_RF_MIN_NODE_SIZE    optional comma list
#   DBT_RF_SAMPLE_FRACTION  optional comma list, e.g. "0.632,0.8,1"
#   DBT_RESID_RF_TUNE       TRUE/FALSE, default TRUE
#   DBT_RESID_RF_TUNE_MAX_GRID default 12
#   DBT_RESID_RF_NUM_TREES       optional comma list
#   DBT_RESID_RF_MTRY            optional comma list
#   DBT_RESID_RF_MIN_NODE_SIZE   optional comma list
#   DBT_RESID_RF_SAMPLE_FRACTION optional comma list
#   DBT_FAIR_PROFILE_SEARCH      TRUE/FALSE, default TRUE
#   DBT_GEO_BASIS                variogram-derived basis features, default TRUE
#   DBT_VG_SEARCH                select variogram binning/estimator, default TRUE
#   DBT_BW_SEARCH                select the range multiplier, default TRUE
#   DBT_ANISO_SEARCH             offer anisotropic geometry, default TRUE
#   DBT_FAIR_PROFILE_MAX         default 6
#   DBT_ENSEMBLE_TOP_K           top-k profile ensemble, default 3 (1 = off)
#   DBT_ENSEMBLE_WEIGHTING       softmax | inverse | uniform, default softmax
#   DBT_ENSEMBLE_TEMPERATURE     default 0.75
#   DBT_TRAIN_SEED          default 123
#   DBT_RESULTS_DIR         default results/dbtarf_run
#   DBT_CHECKPOINT_EVERY    default 1
# =============================================================================

rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")
source("./R/dbtarf.R")  # DB-TARF lives in the package R/ directory

suppressPackageStartupMessages({
  if (requireNamespace("Cubist", quietly = TRUE)) library(Cubist)
  if (requireNamespace("ranger", quietly = TRUE)) library(ranger)
})

# ---- Env-var parsing --------------------------------------------------------

parse_bool_dbt <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}
parse_num_dbt <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  v <- suppressWarnings(as.numeric(raw))
  if (!is.finite(v)) default else v
}
parse_list_dbt <- function(name, default) {
  raw <- Sys.getenv(name, unset = paste(default, collapse = ","))
  v <- trimws(unlist(strsplit(raw, ",")))
  v[v != ""]
}
parse_num_list_dbt <- function(name, default = NULL) {
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) return(default)
  vals <- suppressWarnings(as.numeric(trimws(unlist(strsplit(raw, ",")))))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0L) default else vals
}

# ---- Config -----------------------------------------------------------------

scenario <- Sys.getenv("DBT_SCENARIO", unset = "random")
protocols <- parse_list_dbt("DBT_PROTOCOLS", c("DesignBased"))
allowed_protocols <- c("Population", "DesignBased", "SpatialKFold")
unsupported_protocols <- setdiff(protocols, allowed_protocols)
if (length(unsupported_protocols) > 0L) {
  stop("DBTARF runner supports {Population, DesignBased, SpatialKFold}. Unsupported: ",
       paste(unsupported_protocols, collapse = ", "))
}

models <- parse_list_dbt("DBT_MODELS", c("RF", "Cubist", "DBTARF"))
allowed_models <- c("RF", "Cubist", "DBTARF")
invalid <- setdiff(models, allowed_models)
if (length(invalid) > 0L) {
  stop("Unsupported DBT_MODELS entries: ", paste(invalid, collapse = ", "))
}

n_iter <- as.integer(parse_num_dbt("DBT_N_ITER", 5L))
sample_size <- as.integer(parse_num_dbt("DBT_SAMPLE_SIZE", 500L))
train_seed <- as.integer(parse_num_dbt("DBT_TRAIN_SEED", 123L))
results_dir <- Sys.getenv("DBT_RESULTS_DIR", unset = "results/dbtarf_run")
checkpoint_every <- as.integer(parse_num_dbt("DBT_CHECKPOINT_EVERY", 1L))
# SpatialKFold exclusion distance (km) for hclust/cutree spatial folds; the
# Wadoux (2021) case study uses ~350 km (slightly above the AGB variogram range).
val_dist_km <- parse_num_dbt("DBT_VAL_DIST_KM", 350)
# Optional fork-parallel evaluation of SpatialKFold folds (each fold is seeded
# independently, so results are identical to serial). Default 1 = serial.
fold_cores <- as.integer(parse_num_dbt("DBT_FOLD_CORES", 1L))
# Population protocol: score a fixed reference subset of the grid instead of all
# ~1.3M cells. 0 = full grid (exact, slow). A large subset (e.g. 200000) is a
# high-precision Monte-Carlo estimate of the population RMSE; the SAME cells are
# used for every model and iteration, so the paired comparison is unaffected.
pop_sample <- as.integer(parse_num_dbt("DBT_POP_SAMPLE", 0L))

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
options(wadoux_corr_method = "pearson")

# Only override the cross-fitting fold count when DBT_N_FOLDS is explicitly set;
# otherwise leave it NULL so DBTARF's data-driven auto-folds choice applies.
n_folds_env <- Sys.getenv("DBT_N_FOLDS", unset = "")
# Top-k profile-ensemble weighting scheme (validated up front).
dbt_ensemble_weighting <- tolower(trimws(Sys.getenv(
  "DBT_ENSEMBLE_WEIGHTING", unset = "softmax")))
if (!(dbt_ensemble_weighting %in% c("softmax", "inverse", "uniform"))) {
  stop("Unsupported DBT_ENSEMBLE_WEIGHTING: ", dbt_ensemble_weighting)
}
dbtarf_params <- modifyList(
  dbtarf_default_params,
  list(
    train_seed = train_seed,
    n_folds = if (nzchar(n_folds_env))
                as.integer(parse_num_dbt("DBT_N_FOLDS", 5L)) else NULL,
    lambda_grid = parse_num_list_dbt("DBT_LAMBDA_GRID", NULL),
    min_oof_improvement = parse_num_dbt("DBT_MIN_OOF_IMPROVEMENT", 0),
    rf_tune = parse_bool_dbt("DBT_RF_TUNE", TRUE),
    rf_tune_method = Sys.getenv("DBT_RF_TUNE_METHOD", unset = "oob"),
    rf_tune_inner_folds = as.integer(parse_num_dbt("DBT_RF_TUNE_INNER_FOLDS", 3L)),
    rf_tune_max_grid = as.integer(parse_num_dbt("DBT_RF_TUNE_MAX_GRID", 18L)),
    rf_num_trees = parse_num_list_dbt("DBT_RF_NUM_TREES", NULL),
    rf_mtry = parse_num_list_dbt("DBT_RF_MTRY", NULL),
    rf_min_node_size = parse_num_list_dbt("DBT_RF_MIN_NODE_SIZE", NULL),
    rf_sample_fraction = parse_num_list_dbt("DBT_RF_SAMPLE_FRACTION", NULL),
    resid_rf_tune = parse_bool_dbt("DBT_RESID_RF_TUNE", TRUE),
    resid_rf_tune_max_grid = as.integer(parse_num_dbt("DBT_RESID_RF_TUNE_MAX_GRID", 12L)),
    resid_rf_num_trees = parse_num_list_dbt("DBT_RESID_RF_NUM_TREES", NULL),
    resid_rf_mtry = parse_num_list_dbt("DBT_RESID_RF_MTRY", NULL),
    resid_rf_min_node_size = parse_num_list_dbt("DBT_RESID_RF_MIN_NODE_SIZE", NULL),
    resid_rf_sample_fraction = parse_num_list_dbt("DBT_RESID_RF_SAMPLE_FRACTION", NULL),
    fair_profile_search = parse_bool_dbt("DBT_FAIR_PROFILE_SEARCH", TRUE),
    vg_search = parse_bool_dbt("DBT_VG_SEARCH", TRUE),
    bw_search = parse_bool_dbt("DBT_BW_SEARCH", TRUE),
    aniso_search = parse_bool_dbt("DBT_ANISO_SEARCH", TRUE),
    fair_profile_max = as.integer(parse_num_dbt("DBT_FAIR_PROFILE_MAX", 6L)),
    geo_basis = parse_bool_dbt("DBT_GEO_BASIS", TRUE),
    ensemble_top_k = as.integer(parse_num_dbt("DBT_ENSEMBLE_TOP_K", 3L)),
    ensemble_weighting = dbt_ensemble_weighting,
    ensemble_temperature = parse_num_dbt("DBT_ENSEMBLE_TEMPERATURE", 0.75)
  )
)
if (!is.finite(dbtarf_params$ensemble_top_k) ||
    dbtarf_params$ensemble_top_k < 1L) {
  dbtarf_params$ensemble_top_k <- 1L
}
if (!is.finite(dbtarf_params$ensemble_temperature) ||
    dbtarf_params$ensemble_temperature <= 0) {
  dbtarf_params$ensemble_temperature <- 0.75
}
n_folds_report <- if (is.null(dbtarf_params$n_folds)) "auto" else as.character(dbtarf_params$n_folds)

cat(sprintf(
  "[DBTARF runner] scenario=%s  protocols=%s  models=%s  n_iter=%d  n=%d  ensemble_top_k=%d weighting=%s\n",
  scenario, paste(protocols, collapse = ","),
  paste(models, collapse = ","), n_iter, sample_size,
  dbtarf_params$ensemble_top_k, dbtarf_params$ensemble_weighting))

# ---- Data -------------------------------------------------------------------

need_polygon <- scenario %in% c("regular_grid", "clustered_random")
common_data <- load_wadoux_common_data(include_polygon = need_polygon)
predList <- predList_modelfull_wadoux
form_rf <- as.formula(
  paste(response_name_wadoux, "~", paste(predList, collapse = "+"))
)

# Fixed reference set the Population protocol scores against (full grid by
# default). Drawn with a dedicated seed and the main RNG stream restored, so
# calibration sampling in the loop is identical whether or not subsampling is on.
pop_eval_df <- common_data$s_df
if ("Population" %in% protocols && pop_sample > 0L &&
    pop_sample < nrow(common_data$s_df)) {
  .pop_old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) {
    get(".Random.seed", envir = .GlobalEnv)
  } else NULL
  set.seed(20240620L)
  pop_eval_df <- common_data$s_df[
    sample.int(nrow(common_data$s_df), pop_sample), , drop = FALSE]
  if (!is.null(.pop_old_seed)) {
    assign(".Random.seed", .pop_old_seed, envir = .GlobalEnv)
  }
  cat(sprintf("[DBTARF runner] Population scored on a fixed %d-cell reference subset (of %d)\n",
              pop_sample, nrow(common_data$s_df)))
}

# ---- Sampling ---------------------------------------------------------------

sample_calibration_dbt <- function() {
  if (scenario == "random") {
    sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  } else if (scenario == "regular_grid") {
    sample_valuetable_regular_grid_wadoux(
      common_data = common_data, sample_size = sample_size)
  } else if (scenario == "clustered_random") {
    sample_valuetable_clustered_random_wadoux(
      common_data = common_data,
      kmeans_centers = as.integer(parse_num_dbt("DBT_KMEANS_CENTERS", 100L)),
      n_psu = as.integer(parse_num_dbt("DBT_N_PSU", 20L)),
      m_per_psu = as.integer(parse_num_dbt("DBT_M_PER_PSU", 25L)))
  } else {
    stop("Unsupported DBT_SCENARIO: ", scenario)
  }
}

fit_predict_cubist_dbt <- function(X_train, y_train, X_test,
                                   committees = 10L, neighbors = 0L) {
  if (!requireNamespace("Cubist", quietly = TRUE)) {
    return(rep(NA_real_, nrow(X_test)))
  }
  cb <- Cubist::cubist(x = as.data.frame(X_train), y = y_train,
                       committees = committees)
  as.numeric(predict(cb, newdata = as.data.frame(X_test),
                     neighbors = neighbors))
}

# ---- Uncertainty (design-based prediction intervals) ------------------------

# Quantile-regression-forest 90% interval (Meinshausen QRF), the uncertainty
# baseline for the DB-TARF conformal interval.
fit_rf_quantile_interval <- function(train_df, test_df, alpha = 0.10, seed = 1L) {
  if (!requireNamespace("ranger", quietly = TRUE)) return(NULL)
  m <- tryCatch(ranger::ranger(form_rf, data = train_df, quantreg = TRUE,
                               num.threads = 1L, seed = as.integer(seed)),
                error = function(e) NULL)
  if (is.null(m)) return(NULL)
  p <- tryCatch(predict(m, data = test_df, type = "quantiles",
                        quantiles = c(alpha / 2, 1 - alpha / 2),
                        num.threads = 1L)$predictions, error = function(e) NULL)
  if (is.null(p)) return(NULL)
  list(lwr = as.numeric(p[, 1]), upr = as.numeric(p[, 2]))
}

# Design-based interval assessment on the probability validation sample:
# PICP90 = coverage (design-unbiased estimate of population coverage),
# MPIW90 = mean interval width. Compares DB-TARF conformal vs QRF at nominal 0.90.
eval_uncertainty_dbt <- function(obs, dbt_out, pred_dbt, valuetable, test_df,
                                 iter, protocol, seed) {
  rows <- list()
  wrow <- function(model, lwr, upr) {
    ok <- is.finite(lwr) & is.finite(upr) & is.finite(obs)
    if (sum(ok) == 0L) return(NULL)
    data.frame(iteration = iter, scenario = scenario, protocol = protocol,
      model = model,
      PICP90 = mean(obs[ok] >= lwr[ok] & obs[ok] <= upr[ok]),
      MPIW90 = mean(upr[ok] - lwr[ok]), stringsAsFactors = FALSE)
  }
  if (!is.null(dbt_out) && !is.null(dbt_out$pi_q90) &&
      is.finite(dbt_out$pi_q90) && !is.null(pred_dbt)) {
    q <- dbt_out$pi_q90
    # The plain interval is CV+ (Barber et al. 2021), which is what the paper
    # claims and what carries a finite-sample bound. The pooled out-of-fold
    # half-width centred on the full-sample refit is kept alongside it, under
    # its own name, because the two are worth comparing and the older result
    # should stay visible rather than disappear.
    if (!is.null(dbt_out$pi_lo90) && !is.null(dbt_out$pi_hi90) &&
        length(dbt_out$pi_lo90) == length(pred_dbt)) {
      rows[[length(rows) + 1L]] <- wrow("DBTARF", dbt_out$pi_lo90, dbt_out$pi_hi90)
      rows[[length(rows) + 1L]] <- wrow("DBTARF_oofq", pred_dbt - q, pred_dbt + q)
    } else {
      rows[[length(rows) + 1L]] <- wrow("DBTARF", pred_dbt - q, pred_dbt + q)
    }
    # Design-aware (density-ratio weighted) conformal: restores coverage under
    # clustered sampling where plain conformal under-covers.
    if (!is.null(dbt_out$pi_q90_w) && is.finite(dbt_out$pi_q90_w)) {
      qw <- dbt_out$pi_q90_w
      rows[[length(rows) + 1L]] <- wrow("DBTARF_dbcal", pred_dbt - qw, pred_dbt + qw)
    }
    # Spatial-block-calibrated conformal (extrapolation-aware).
    if (!is.null(dbt_out$pi_q90_sp) && is.finite(dbt_out$pi_q90_sp)) {
      qs <- dbt_out$pi_q90_sp
      rows[[length(rows) + 1L]] <- wrow("DBTARF_spcal", pred_dbt - qs, pred_dbt + qs)
    }
  }
  qi <- fit_rf_quantile_interval(valuetable, test_df, alpha = 0.10, seed = seed)
  if (!is.null(qi)) rows[[length(rows) + 1L]] <- wrow("RF_quantile", qi$lwr, qi$upr)
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows)) do.call(rbind, rows) else data.frame()
}

# ---- Metrics ----------------------------------------------------------------

emit_metric_rows_dbt <- function(obs, pred_map, protocol, iter, scenario) {
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

# Fit RF / Cubist / DBTARF on train_df and predict test_df (shared by every
# protocol so the comparison is identical across Population/DesignBased/SpatialKFold).
fit_predict_all_dbt <- function(train_df, test_df, iter, seed_off = 0L,
                                capture_diag = FALSE) {
  pred_map <- list()
  out <- NULL
  if ("RF" %in% models) {
    # seed makes the Wadoux default-RF baseline reproducible (ranger >= 0.18 is
    # seed-reproducible across thread counts); threads left at default for speed.
    rf <- fit_rf_default_wadoux(train_df, form_rf,
                                seed = train_seed + iter + seed_off)
    pred_map$RF <- predict_rf_default_wadoux(rf, test_df)
  }
  if ("Cubist" %in% models) {
    pred_map$Cubist <- fit_predict_cubist_dbt(
      X_train = train_df[, predList, drop = FALSE],
      y_train = train_df[[response_name_wadoux]],
      X_test = test_df[, predList, drop = FALSE])
  }
  if ("DBTARF" %in% models) {
    params_iter <- dbtarf_params
    params_iter$train_seed <- train_seed + iter + seed_off
    out <- do.call(train_dbtarf_one_fold, c(list(
      train_df = train_df,
      test_df = test_df,
      predictor_names = predList,
      response_name = response_name_wadoux,
      coord_names = c("x", "y")
    ), params_iter))
    pred_map$DBTARF <- out$pred_test
    # Transparency: DBTARF's selected-profile tuned base RF with NO correction.
    # Lets the benchmark show default-RF vs tuned-RF vs DBTARF, so the gain
    # attributable to the residual correction (beyond RF tuning) is visible.
    if (!is.null(out$pred_test_base)) {
      pred_map$RF_tuned <- as.numeric(out$pred_test_base)
    }
  }
  list(pred_map = pred_map, dbtarf = if (capture_diag) out else NULL)
}

# DBTARF diagnostic capture (only for single-train protocols). Tags by protocol.
collect_dbtarf_diag <- function(out, protocol, iter) {
  if (is.null(out)) {
    return(list(diagnostics = data.frame(), candidates = data.frame(),
                fold_diagnostics = data.frame(), rf_tuning = data.frame()))
  }
  list(
    diagnostics = cbind(protocol = protocol, iteration = iter,
                        as.data.frame(out$diagnostics, stringsAsFactors = FALSE)),
    candidates = cbind(protocol = protocol, iteration = iter, out$candidate_table),
    fold_diagnostics = cbind(protocol = protocol, iteration = iter, out$fold_diagnostics),
    rf_tuning = cbind(protocol = protocol, iteration = iter, out$rf_tuning_table)
  )
}

# Evaluate one Wadoux protocol for one iteration, returning protocol-tagged rows.
#   - DesignBased:  train on calibration, score an independent SRS validation draw
#   - Population:   train on calibration, score the FULL population grid (truth)
#   - SpatialKFold: spatial hclust/cutree folds on the calibration sample, pooled
eval_protocol_dbt <- function(protocol, valuetable, iter) {
  if (identical(protocol, "DesignBased")) {
    val_srs <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
    fp <- fit_predict_all_dbt(valuetable, val_srs, iter, capture_diag = TRUE)
    unc <- eval_uncertainty_dbt(val_srs[[response_name_wadoux]], fp$dbtarf,
             fp$pred_map$DBTARF, valuetable, val_srs, iter, protocol,
             seed = train_seed + iter)
    return(c(list(metrics = emit_metric_rows_dbt(
      val_srs[[response_name_wadoux]], fp$pred_map, protocol, iter, scenario),
      uncertainty = unc),
      collect_dbtarf_diag(fp$dbtarf, protocol, iter)))
  }
  if (identical(protocol, "Population")) {
    fp <- fit_predict_all_dbt(valuetable, pop_eval_df, iter, capture_diag = TRUE)
    unc <- eval_uncertainty_dbt(pop_eval_df[[response_name_wadoux]], fp$dbtarf,
             fp$pred_map$DBTARF, valuetable, pop_eval_df, iter, protocol,
             seed = train_seed + iter)
    return(c(list(metrics = emit_metric_rows_dbt(
      pop_eval_df[[response_name_wadoux]], fp$pred_map, protocol, iter, scenario),
      uncertainty = unc),
      collect_dbtarf_diag(fp$dbtarf, protocol, iter)))
  }
  if (identical(protocol, "SpatialKFold")) {
    spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
    groups <- sort(unique(spatial_folds))
    one_fold <- function(jg) {
      g <- groups[jg]
      tr <- valuetable[spatial_folds != g, , drop = FALSE]
      te <- valuetable[spatial_folds == g, , drop = FALSE]
      fp <- fit_predict_all_dbt(tr, te, iter, seed_off = jg * 7919L)
      list(obs = te[[response_name_wadoux]], pred = fp$pred_map)
    }
    fold_res <- if (fold_cores > 1L) {
      parallel::mclapply(seq_along(groups), one_fold,
                         mc.cores = fold_cores, mc.preschedule = FALSE)
    } else {
      lapply(seq_along(groups), one_fold)
    }
    obs_all <- numeric(0)
    acc <- list()
    for (fr in fold_res) {
      if (is.null(fr) || inherits(fr, "try-error")) next
      obs_all <- c(obs_all, fr$obs)
      for (m in names(fr$pred)) acc[[m]] <- c(acc[[m]], fr$pred[[m]])
    }
    cat(sprintf("[DBTARF] SpatialKFold: %d spatial folds (val_dist=%g km, cores=%d)\n",
                length(groups), val_dist_km, fold_cores))
    return(c(list(metrics = emit_metric_rows_dbt(obs_all, acc, protocol, iter, scenario),
      uncertainty = data.frame()),
      collect_dbtarf_diag(NULL, protocol, iter)))
  }
  stop("Unsupported protocol: ", protocol)
}

# ---- Save helpers -----------------------------------------------------------

save_dbt_checkpoint <- function(iter, metric_rows, diagnostic_rows,
                                candidate_rows, fold_rows, tuning_rows) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  write.csv(df, file.path(results_dir, "dbtarf_all_results.csv"),
            row.names = FALSE)

  diag_df <- do.call(rbind, diagnostic_rows[!vapply(diagnostic_rows, is.null, logical(1))])
  if (!is.null(diag_df) && nrow(diag_df) > 0L) {
    write.csv(diag_df, file.path(results_dir, "dbtarf_diagnostics.csv"),
              row.names = FALSE)
  }
  cand_df <- do.call(rbind, candidate_rows[!vapply(candidate_rows, is.null, logical(1))])
  if (!is.null(cand_df) && nrow(cand_df) > 0L) {
    write.csv(cand_df, file.path(results_dir, "dbtarf_candidate_scores.csv"),
              row.names = FALSE)
  }
  fold_df <- do.call(rbind, fold_rows[!vapply(fold_rows, is.null, logical(1))])
  if (!is.null(fold_df) && nrow(fold_df) > 0L) {
    write.csv(fold_df, file.path(results_dir, "dbtarf_fold_diagnostics.csv"),
              row.names = FALSE)
  }
  tune_df <- do.call(rbind, tuning_rows[!vapply(tuning_rows, is.null, logical(1))])
  if (!is.null(tune_df) && nrow(tune_df) > 0L) {
    write.csv(tune_df, file.path(results_dir, "dbtarf_rf_tuning_scores.csv"),
              row.names = FALSE)
  }

  if ("RF" %in% models && "DBTARF" %in% models) {
    key_cols <- c("iteration", "scenario", "protocol")
    rf_df <- df[df$model == "RF", , drop = FALSE]
    dbt_df <- df[df$model == "DBTARF", , drop = FALSE]
    if (nrow(rf_df) > 0L && nrow(dbt_df) > 0L) {
      merged <- merge(
        rf_df[, c(key_cols, "ME", "RMSE", "r2", "MEC")],
        dbt_df[, c(key_cols, "ME", "RMSE", "r2", "MEC")],
        by = key_cols, suffixes = c("_RF", "_DBTARF"))
      merged$delta_ME <- merged$ME_DBTARF - merged$ME_RF
      merged$delta_RMSE <- merged$RMSE_DBTARF - merged$RMSE_RF
      merged$delta_r2 <- merged$r2_DBTARF - merged$r2_RF
      merged$delta_MEC <- merged$MEC_DBTARF - merged$MEC_RF
      write.csv(merged, file.path(results_dir,
                                  "dbtarf_paired_deltas_vs_RF.csv"),
                row.names = FALSE)

      # Script-generated win-rate summary, per protocol (so the headline "k/n
      # wins" claim is reproducible from the run, not hand-counted).
      winrate <- do.call(rbind, lapply(split(merged, merged$protocol), function(mp) {
        d <- mp$delta_RMSE
        data.frame(
          protocol = mp$protocol[1L],
          n_iter = length(d),
          dbtarf_wins = sum(d < 0, na.rm = TRUE),
          ties = sum(d == 0, na.rm = TRUE),
          rf_wins = sum(d > 0, na.rm = TRUE),
          mean_delta_RMSE = mean(d, na.rm = TRUE),
          median_delta_RMSE = stats::median(d, na.rm = TRUE),
          sd_delta_RMSE = stats::sd(d, na.rm = TRUE),
          mean_delta_ME = mean(mp$delta_ME, na.rm = TRUE),
          mean_delta_r2 = mean(mp$delta_r2, na.rm = TRUE),
          mean_delta_MEC = mean(mp$delta_MEC, na.rm = TRUE),
          stringsAsFactors = FALSE)
      }))
      write.csv(winrate, file.path(results_dir,
                                   "dbtarf_winrate_vs_RF.csv"),
                row.names = FALSE)
    }
  }

  write.csv(
    data.frame(completed_iteration = iter, total_iterations = n_iter,
               scenario = scenario, protocols = paste(protocols, collapse = ","),
               models = paste(models, collapse = ","),
               completed_at = format(Sys.time(), tz = "", usetz = TRUE),
               stringsAsFactors = FALSE),
    file.path(results_dir, "dbtarf_checkpoint_status.csv"), row.names = FALSE)
}

summarise_and_save_dbt <- function(metric_rows) {
  df <- do.call(rbind, metric_rows[!vapply(metric_rows, is.null, logical(1))])
  if (is.null(df) || nrow(df) == 0L) return(invisible(NULL))
  by_protocol <- aggregate(
    cbind(ME, RMSE, r2, MEC) ~ scenario + protocol + model,
    data = df,
    FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                        sd = sd(x, na.rm = TRUE)),
    na.action = na.pass)
  stats_df <- do.call(cbind, lapply(c("ME", "RMSE", "r2", "MEC"),
                                    function(m) {
    mat <- by_protocol[[m]]
    setNames(as.data.frame(mat), c(paste0(m, "_mean"), paste0(m, "_sd")))
  }))
  out <- cbind(by_protocol[, c("scenario", "protocol", "model")], stats_df)
  write.csv(out, file.path(results_dir, "dbtarf_summary_by_protocol.csv"),
            row.names = FALSE)
}

# ---- Main loop --------------------------------------------------------------

write.csv(
  data.frame(scenario = scenario, protocols = paste(protocols, collapse = ","),
             models = paste(models, collapse = ","),
             sample_size = sample_size, n_iter = n_iter,
             n_folds = n_folds_report,
             min_oof_improvement = dbtarf_params$min_oof_improvement,
             rf_tune = dbtarf_params$rf_tune,
             rf_tune_method = dbtarf_params$rf_tune_method,
             rf_tune_inner_folds = dbtarf_params$rf_tune_inner_folds,
             rf_tune_max_grid = dbtarf_params$rf_tune_max_grid,
             rf_num_trees = paste(dbtarf_params$rf_num_trees, collapse = ","),
             rf_mtry = paste(dbtarf_params$rf_mtry, collapse = ","),
             rf_min_node_size = paste(dbtarf_params$rf_min_node_size, collapse = ","),
             rf_sample_fraction = paste(dbtarf_params$rf_sample_fraction, collapse = ","),
             resid_rf_tune = dbtarf_params$resid_rf_tune,
             resid_rf_tune_method = ifelse(dbtarf_params$resid_rf_tune,
                                           dbtarf_params$rf_tune_method, "none"),
             resid_rf_tune_max_grid = dbtarf_params$resid_rf_tune_max_grid,
             resid_rf_num_trees = paste(dbtarf_params$resid_rf_num_trees, collapse = ","),
             resid_rf_mtry = paste(dbtarf_params$resid_rf_mtry, collapse = ","),
             resid_rf_min_node_size = paste(dbtarf_params$resid_rf_min_node_size, collapse = ","),
             resid_rf_sample_fraction = paste(dbtarf_params$resid_rf_sample_fraction, collapse = ","),
             fair_profile_search = dbtarf_params$fair_profile_search,
             fair_profile_max = dbtarf_params$fair_profile_max,
             val_dist_km = val_dist_km, fold_cores = fold_cores,
             pop_sample = pop_sample,
             train_seed = train_seed, stringsAsFactors = FALSE),
  file.path(results_dir, "dbtarf_config.csv"), row.names = FALSE)

all_metric_rows <- list()
all_diagnostic_rows <- list()
all_candidate_rows <- list()
all_fold_rows <- list()
all_tuning_rows <- list()
all_unc_rows <- list()

for (iter in seq_len(n_iter)) {
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("DBTARF benchmark | scenario=%s | iter %d / %d\n",
              scenario, iter, n_iter))
  cat(sprintf("==================================================\n"))

  valuetable <- na.omit(as.data.frame(sample_calibration_dbt()))

  metric_parts <- list(); diag_parts <- list(); cand_parts <- list()
  fold_parts <- list(); tune_parts <- list(); unc_parts <- list()
  for (protocol in protocols) {
    cat(sprintf("--- protocol: %s ---\n", protocol))
    po <- eval_protocol_dbt(protocol, valuetable, iter)
    if (!is.null(po$metrics) && nrow(po$metrics) > 0L) {
      print(po$metrics)
      metric_parts[[length(metric_parts) + 1L]] <- po$metrics
    }
    if (!is.null(po$uncertainty) && nrow(po$uncertainty) > 0L) {
      unc_parts[[length(unc_parts) + 1L]] <- po$uncertainty
    }
    if (!is.null(po$diagnostics) && nrow(po$diagnostics) > 0L) {
      diag_parts[[length(diag_parts) + 1L]] <- po$diagnostics
    }
    if (!is.null(po$candidates) && nrow(po$candidates) > 0L) {
      cand_parts[[length(cand_parts) + 1L]] <- po$candidates
    }
    if (!is.null(po$fold_diagnostics) && nrow(po$fold_diagnostics) > 0L) {
      fold_parts[[length(fold_parts) + 1L]] <- po$fold_diagnostics
    }
    if (!is.null(po$rf_tuning) && nrow(po$rf_tuning) > 0L) {
      tune_parts[[length(tune_parts) + 1L]] <- po$rf_tuning
    }
  }
  if (length(metric_parts)) all_metric_rows[[iter]] <- do.call(rbind, metric_parts)
  if (length(diag_parts)) all_diagnostic_rows[[iter]] <- do.call(rbind, diag_parts)
  if (length(cand_parts)) all_candidate_rows[[iter]] <- do.call(rbind, cand_parts)
  if (length(fold_parts)) all_fold_rows[[iter]] <- do.call(rbind, fold_parts)
  if (length(tune_parts)) all_tuning_rows[[iter]] <- do.call(rbind, tune_parts)
  if (length(unc_parts)) all_unc_rows[[iter]] <- do.call(rbind, unc_parts)

  if (iter %% checkpoint_every == 0L || iter == n_iter) {
    save_dbt_checkpoint(iter, all_metric_rows, all_diagnostic_rows,
                        all_candidate_rows, all_fold_rows, all_tuning_rows)
    summarise_and_save_dbt(all_metric_rows)
    unc_df <- do.call(rbind, all_unc_rows[!vapply(all_unc_rows, is.null, logical(1))])
    if (!is.null(unc_df) && nrow(unc_df) > 0L) {
      write.csv(unc_df, file.path(results_dir, "dbtarf_uncertainty.csv"),
                row.names = FALSE)
      usum <- aggregate(cbind(PICP90, MPIW90) ~ scenario + protocol + model,
                        data = unc_df, na.action = na.pass,
                        FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                            sd = sd(x, na.rm = TRUE)))
      ustat <- do.call(cbind, lapply(c("PICP90", "MPIW90"), function(m) {
        mat <- usum[[m]]
        setNames(as.data.frame(mat), c(paste0(m, "_mean"), paste0(m, "_sd")))
      }))
      write.csv(cbind(usum[, c("scenario", "protocol", "model")], ustat),
                file.path(results_dir, "dbtarf_uncertainty_summary.csv"),
                row.names = FALSE)
    }
  }
}

cat("\n[DBTARF runner] Done. Results written to: ", results_dir, "\n",
    sep = "")
