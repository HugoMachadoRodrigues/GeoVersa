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
#   DBT_PROTOCOLS           comma list, currently {DesignBased}
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
#   DBT_FAIR_PROFILE_MAX         default 6
#   DBT_TRAIN_SEED          default 123
#   DBT_RESULTS_DIR         default results/dbtarf_run
#   DBT_CHECKPOINT_EVERY    default 1
# =============================================================================

rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")
source("./code/DBTARF.R")

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
unsupported_protocols <- setdiff(protocols, "DesignBased")
if (length(unsupported_protocols) > 0L) {
  stop("DBTARF runner currently supports DesignBased only. Unsupported: ",
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

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
options(wadoux_corr_method = "pearson")

# Only override the cross-fitting fold count when DBT_N_FOLDS is explicitly set;
# otherwise leave it NULL so DBTARF's data-driven auto-folds choice applies.
n_folds_env <- Sys.getenv("DBT_N_FOLDS", unset = "")
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
    fair_profile_max = as.integer(parse_num_dbt("DBT_FAIR_PROFILE_MAX", 6L))
  )
)
n_folds_report <- if (is.null(dbtarf_params$n_folds)) "auto" else as.character(dbtarf_params$n_folds)

cat(sprintf(
  "[DBTARF runner] scenario=%s  protocols=%s  models=%s  n_iter=%d  n=%d\n",
  scenario, paste(protocols, collapse = ","),
  paste(models, collapse = ","), n_iter, sample_size))

# ---- Data -------------------------------------------------------------------

need_polygon <- scenario %in% c("regular_grid", "clustered_random")
common_data <- load_wadoux_common_data(include_polygon = need_polygon)
predList <- predList_modelfull_wadoux
form_rf <- as.formula(
  paste(response_name_wadoux, "~", paste(predList, collapse = "+"))
)

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

eval_designbased_dbt <- function(valuetable, iter) {
  val_srs <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  obs <- val_srs[[response_name_wadoux]]
  pred_map <- list()
  diag_rows <- list()
  candidate_rows <- list()
  fold_rows <- list()
  tuning_rows <- list()

  if ("RF" %in% models) {
    # seed makes the Wadoux default-RF baseline reproducible (ranger >= 0.18 is
    # seed-reproducible across thread counts); threads left at default for speed.
    rf <- fit_rf_default_wadoux(valuetable, form_rf, seed = train_seed + iter)
    pred_map$RF <- predict_rf_default_wadoux(rf, val_srs)
  }
  if ("Cubist" %in% models) {
    pred_map$Cubist <- fit_predict_cubist_dbt(
      X_train = valuetable[, predList, drop = FALSE],
      y_train = valuetable[[response_name_wadoux]],
      X_test = val_srs[, predList, drop = FALSE])
  }
  if ("DBTARF" %in% models) {
    params_iter <- dbtarf_params
    params_iter$train_seed <- train_seed + iter
    out <- do.call(train_dbtarf_one_fold, c(list(
      train_df = valuetable,
      test_df = val_srs,
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
    diag_rows[[1L]] <- cbind(
      iteration = iter,
      as.data.frame(out$diagnostics, stringsAsFactors = FALSE))
    candidate_rows[[1L]] <- cbind(iteration = iter, out$candidate_table)
    fold_rows[[1L]] <- cbind(iteration = iter, out$fold_diagnostics)
    tuning_rows[[1L]] <- cbind(iteration = iter, out$rf_tuning_table)
  }

  list(
    metrics = emit_metric_rows_dbt(obs, pred_map, "DesignBased", iter, scenario),
    diagnostics = if (length(diag_rows)) do.call(rbind, diag_rows) else data.frame(),
    candidates = if (length(candidate_rows)) do.call(rbind, candidate_rows) else data.frame(),
    fold_diagnostics = if (length(fold_rows)) do.call(rbind, fold_rows) else data.frame(),
    rf_tuning = if (length(tuning_rows)) do.call(rbind, tuning_rows) else data.frame()
  )
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

      # Script-generated win-rate summary (so the headline "k/n wins" claim is
      # reproducible from the run, not hand-counted).
      d <- merged$delta_RMSE
      winrate <- data.frame(
        n_iter = length(d),
        dbtarf_wins = sum(d < 0, na.rm = TRUE),
        ties = sum(d == 0, na.rm = TRUE),
        rf_wins = sum(d > 0, na.rm = TRUE),
        mean_delta_RMSE = mean(d, na.rm = TRUE),
        median_delta_RMSE = stats::median(d, na.rm = TRUE),
        sd_delta_RMSE = stats::sd(d, na.rm = TRUE),
        mean_delta_ME = mean(merged$delta_ME, na.rm = TRUE),
        mean_delta_r2 = mean(merged$delta_r2, na.rm = TRUE),
        mean_delta_MEC = mean(merged$delta_MEC, na.rm = TRUE),
        stringsAsFactors = FALSE)
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
             train_seed = train_seed, stringsAsFactors = FALSE),
  file.path(results_dir, "dbtarf_config.csv"), row.names = FALSE)

all_metric_rows <- list()
all_diagnostic_rows <- list()
all_candidate_rows <- list()
all_fold_rows <- list()
all_tuning_rows <- list()

for (iter in seq_len(n_iter)) {
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("DBTARF benchmark | scenario=%s | iter %d / %d\n",
              scenario, iter, n_iter))
  cat(sprintf("==================================================\n"))

  valuetable <- na.omit(as.data.frame(sample_calibration_dbt()))
  iter_out <- eval_designbased_dbt(valuetable, iter)
  iter_df <- iter_out$metrics
  if (!is.null(iter_df) && nrow(iter_df) > 0L) {
    print(iter_df)
    all_metric_rows[[iter]] <- iter_df
  }
  if (!is.null(iter_out$diagnostics) && nrow(iter_out$diagnostics) > 0L) {
    all_diagnostic_rows[[iter]] <- iter_out$diagnostics
  }
  if (!is.null(iter_out$candidates) && nrow(iter_out$candidates) > 0L) {
    all_candidate_rows[[iter]] <- iter_out$candidates
  }
  if (!is.null(iter_out$fold_diagnostics) && nrow(iter_out$fold_diagnostics) > 0L) {
    all_fold_rows[[iter]] <- iter_out$fold_diagnostics
  }
  if (!is.null(iter_out$rf_tuning) && nrow(iter_out$rf_tuning) > 0L) {
    all_tuning_rows[[iter]] <- iter_out$rf_tuning
  }

  if (iter %% checkpoint_every == 0L || iter == n_iter) {
    save_dbt_checkpoint(iter, all_metric_rows, all_diagnostic_rows,
                        all_candidate_rows, all_fold_rows, all_tuning_rows)
    summarise_and_save_dbt(all_metric_rows)
  }
}

cat("\n[DBTARF runner] Done. Results written to: ", results_dir, "\n",
    sep = "")
