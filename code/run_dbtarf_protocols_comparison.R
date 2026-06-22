# =============================================================================
# run_dbtarf_protocols_comparison.R
#
# Extends the DB-TARF vs RF head-to-head BEYOND the DesignBased protocol.
#
# The shipped DB-TARF runner (run_dbtarf_wadoux_comparison.R) supports only the
# DesignBased estimand. This script evaluates DB-TARF against the tuned RF
# baseline under the remaining Wadoux (2021) validation protocols, with BOTH
# models scored on IDENTICAL splits each iteration (paired comparison):
#
#   - Population    : train on the calibration sample, predict a large
#                     independent SRS of the population (map-accuracy gold
#                     standard alongside DesignBased).
#   - RandomKFold   : k random folds over the calibration sample (OOF).
#   - SpatialKFold  : spatially-blocked folds over the calibration sample (OOF).
#
# BLOOCV is intentionally NOT run here: it requires nb_groups * nb_test_pixels
# (>=100, default 1000) sequential model fits, and at DB-TARF's per-fit cost
# (cross-fitting + RF tuning) that is hours-to-days. The repo itself hard-caps
# BLOOCV for expensive models (ConvKrigingNet2D). BLOOCV also probes short-range
# interpolation rather than the design-based map-accuracy claim, so it is
# tangential to the DB-TARF result. See README note.
#
# To keep the k-fold protocols tractable, DB-TARF runs here at a configurable
# SCREENING tuning budget (smaller RF grid, fewer cross-fit folds, fewer fair
# profiles) than the full-budget 30-iter DesignBased gold run. The DesignBased
# headline stands at full budget; this script asks the narrower question of
# whether the advantage TRANSFERS across estimands.
#
# Environment overrides:
#   DBTP_N_ITER              repetitions, default 5
#   DBTP_SAMPLE_SIZE         calibration sample size, default 500
#   DBTP_POP_TEST_SIZE       Population test SRS size, default 20000
#   DBTP_KFOLD_K             RandomKFold folds, default 10
#   DBTP_VAL_DIST_KM         SpatialKFold block distance (km), default 350
#   DBTP_PROTOCOLS           comma list, default Population,RandomKFold,SpatialKFold
#   DBTP_TRAIN_SEED          base seed, default 123
#   DBTP_RESULTS_DIR         default results/dbtarf_protocols_run
#   DBTP_RF_TUNE_MAX_GRID    DB-TARF base RF grid, default 8
#   DBTP_RESID_RF_TUNE_MAX_GRID  DB-TARF resid RF grid, default 6
#   DBTP_FAIR_PROFILE_MAX    DB-TARF fair-profile count, default 3
#   DBTP_N_FOLDS             DB-TARF internal cross-fit folds, default 4
#   DBTP_MIN_SPATIAL_TRAIN   skip spatial folds whose training set < this, default 50
# =============================================================================

rm(list = ls())

source("./code/wadoux2021_rf_reproduction_helpers.R")
source("./code/DBTARF.R")

suppressPackageStartupMessages({
  if (requireNamespace("Cubist", quietly = TRUE)) library(Cubist)
  if (requireNamespace("ranger", quietly = TRUE)) library(ranger)
  library(caret)
})

# ---- Config -----------------------------------------------------------------

getnum <- function(name, default) {
  v <- suppressWarnings(as.numeric(Sys.getenv(name, unset = as.character(default))))
  if (!is.finite(v)) default else v
}

n_iter        <- as.integer(getnum("DBTP_N_ITER", 5L))
sample_size   <- as.integer(getnum("DBTP_SAMPLE_SIZE", 500L))
pop_test_size <- as.integer(getnum("DBTP_POP_TEST_SIZE", 20000L))
kfold_k       <- as.integer(getnum("DBTP_KFOLD_K", 10L))
val_dist_km   <- getnum("DBTP_VAL_DIST_KM", 350)
train_seed0   <- as.integer(getnum("DBTP_TRAIN_SEED", 123L))
results_dir   <- Sys.getenv("DBTP_RESULTS_DIR", unset = "results/dbtarf_protocols_run")
min_sp_train  <- as.integer(getnum("DBTP_MIN_SPATIAL_TRAIN", 50L))

protocols <- trimws(unlist(strsplit(
  Sys.getenv("DBTP_PROTOCOLS",
             unset = "Population,RandomKFold,SpatialKFold"), ",")))
protocols <- protocols[protocols != ""]

dbtarf_screen_params <- modifyList(dbtarf_default_params, list(
  n_folds                = as.integer(getnum("DBTP_N_FOLDS", 4L)),
  rf_tune_max_grid       = as.integer(getnum("DBTP_RF_TUNE_MAX_GRID", 8L)),
  resid_rf_tune_max_grid = as.integer(getnum("DBTP_RESID_RF_TUNE_MAX_GRID", 6L)),
  fair_profile_max       = as.integer(getnum("DBTP_FAIR_PROFILE_MAX", 3L))
))

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
options(wadoux_corr_method = "pearson")

common_data <- load_wadoux_common_data()
predList <- predList_modelfull_wadoux
form_rf <- as.formula(
  paste(response_name_wadoux, "~", paste(predList, collapse = "+")))

cat(sprintf(
  "[DBTARF-protocols] n_iter=%d  n_cal=%d  pop_test=%d  k=%d  val_dist_km=%g\n",
  n_iter, sample_size, pop_test_size, kfold_k, val_dist_km))
cat(sprintf("[DBTARF-protocols] protocols=%s\n", paste(protocols, collapse = ",")))
cat(sprintf(
  "[DBTARF-protocols] DB-TARF screening budget: n_folds=%d rf_grid=%d resid_grid=%d fair_max=%d\n",
  dbtarf_screen_params$n_folds, dbtarf_screen_params$rf_tune_max_grid,
  dbtarf_screen_params$resid_rf_tune_max_grid, dbtarf_screen_params$fair_profile_max))

# ---- Model wrappers (paired: both see the same split) -----------------------

rf_fit_predict <- function(train_df, test_df) {
  rf <- fit_rf_default_wadoux(train_df, form_rf)
  predict_rf_default_wadoux(rf, test_df)
}

dbtarf_fit_predict <- function(train_df, test_df, seed) {
  out <- tryCatch(
    do.call(dbtarf_train_predict, c(list(
      train_df = train_df,
      test_df = test_df,
      response_name = response_name_wadoux,
      predictor_names = predList,
      coord_names = c("x", "y")),
      modifyList(dbtarf_screen_params, list(train_seed = seed)))),
    error = function(e) {
      cat(sprintf("  [DBTARF fit error] %s\n", conditionMessage(e)))
      NULL
    })
  if (is.null(out)) return(rep(NA_real_, nrow(test_df)))
  as.numeric(out$pred_test)
}

score_pair <- function(obs, pred_rf, pred_db, protocol, iter) {
  e_rf <- wadoux_eval(obs = obs, pred = pred_rf)
  e_db <- wadoux_eval(obs = obs, pred = pred_db)
  data.frame(
    iteration = iter, protocol = protocol,
    ME_RF = e_rf$ME, RMSE_RF = e_rf$RMSE, r2_RF = e_rf$r2, MEC_RF = e_rf$MEC,
    ME_DBTARF = e_db$ME, RMSE_DBTARF = e_db$RMSE, r2_DBTARF = e_db$r2,
    MEC_DBTARF = e_db$MEC,
    delta_RMSE = e_db$RMSE - e_rf$RMSE,
    delta_MEC = e_db$MEC - e_rf$MEC,
    stringsAsFactors = FALSE)
}

# ---- Per-protocol evaluators (identical splits for RF and DB-TARF) ----------

eval_population <- function(cal, iter, seed) {
  pool <- common_data$s_df
  cal_xy <- paste0(round(cal$x, 6), "|", round(cal$y, 6))
  pool_xy <- paste0(round(pool$x, 6), "|", round(pool$y, 6))
  eligible <- pool[!(pool_xy %in% cal_xy), , drop = FALSE]
  test_df <- sample_simple_random_rows_wadoux(
    eligible, min(pop_test_size, nrow(eligible)))
  obs <- test_df[[response_name_wadoux]]
  score_pair(obs,
             rf_fit_predict(cal, test_df),
             dbtarf_fit_predict(cal, test_df, seed),
             "Population", iter)
}

eval_kfold <- function(cal, iter, seed, spatial = FALSE) {
  if (spatial) {
    folds <- build_spatial_folds_wadoux(cal, val_dist_km)
    fold_ids <- unique(folds)
    proto <- "SpatialKFold"
    idx_of <- function(j) which(folds == fold_ids[j])
    nfold <- length(fold_ids)
  } else {
    flds <- caret::createFolds(cal[[response_name_wadoux]], k = kfold_k,
                               list = TRUE, returnTrain = FALSE)
    proto <- "RandomKFold"
    idx_of <- function(j) flds[[j]]
    nfold <- length(flds)
  }
  obs <- c(); p_rf <- c(); p_db <- c()
  for (j in seq_len(nfold)) {
    id <- idx_of(j)
    tr <- cal[-id, , drop = FALSE]
    va <- cal[id, , drop = FALSE]
    if (nrow(tr) < min_sp_train || nrow(va) < 1L) next
    obs  <- c(obs, va[[response_name_wadoux]])
    p_rf <- c(p_rf, rf_fit_predict(tr, va))
    p_db <- c(p_db, dbtarf_fit_predict(tr, va, seed + j))
    cat(sprintf("  [%s] fold %d/%d done (n_tr=%d n_va=%d)\n",
                proto, j, nfold, nrow(tr), nrow(va)))
  }
  score_pair(obs, p_rf, p_db, proto, iter)
}

# ---- Main loop --------------------------------------------------------------

all_rows <- list()
for (it in seq_len(n_iter)) {
  seed_it <- train_seed0 + (it - 1L) * 1000L
  set.seed(seed_it)
  cal <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  cat(sprintf("\n===== iteration %d/%d (seed=%d, n_cal=%d) =====\n",
              it, n_iter, seed_it, nrow(cal)))

  if ("Population" %in% protocols) {
    cat("--- Population ---\n")
    all_rows[[length(all_rows) + 1L]] <- eval_population(cal, it, seed_it)
  }
  if ("RandomKFold" %in% protocols) {
    cat("--- RandomKFold ---\n")
    all_rows[[length(all_rows) + 1L]] <- eval_kfold(cal, it, seed_it, spatial = FALSE)
  }
  if ("SpatialKFold" %in% protocols) {
    cat("--- SpatialKFold ---\n")
    all_rows[[length(all_rows) + 1L]] <- eval_kfold(cal, it, seed_it, spatial = TRUE)
  }

  # checkpoint after every iteration
  ckpt <- do.call(rbind, all_rows)
  write.csv(ckpt, file.path(results_dir, "dbtarf_protocols_all_results.csv"),
            row.names = FALSE)
}

results <- do.call(rbind, all_rows)
write.csv(results, file.path(results_dir, "dbtarf_protocols_all_results.csv"),
          row.names = FALSE)

# ---- Paired summary per protocol --------------------------------------------

summ <- do.call(rbind, lapply(split(results, results$protocol), function(d) {
  delta <- d$delta_RMSE
  wins <- sum(delta < 0, na.rm = TRUE)
  losses <- sum(delta > 0, na.rm = TRUE)
  p_t <- tryCatch(
    if (sum(is.finite(delta)) >= 2 && sd(delta, na.rm = TRUE) > 0)
      t.test(d$RMSE_DBTARF, d$RMSE_RF, paired = TRUE,
             alternative = "less")$p.value else NA_real_,
    error = function(e) NA_real_)
  data.frame(
    protocol = d$protocol[1], n_iter = nrow(d),
    RMSE_RF = round(mean(d$RMSE_RF, na.rm = TRUE), 3),
    RMSE_DBTARF = round(mean(d$RMSE_DBTARF, na.rm = TRUE), 3),
    mean_delta_RMSE = round(mean(delta, na.rm = TRUE), 3),
    MEC_RF = round(mean(d$MEC_RF, na.rm = TRUE), 3),
    MEC_DBTARF = round(mean(d$MEC_DBTARF, na.rm = TRUE), 3),
    wins_DBTARF = wins, losses_DBTARF = losses,
    paired_t_p_less = round(p_t, 4),
    stringsAsFactors = FALSE)
}))
write.csv(summ, file.path(results_dir, "dbtarf_protocols_summary.csv"),
          row.names = FALSE)

cat("\n========== DB-TARF vs RF across protocols ==========\n")
print(summ, row.names = FALSE)
cat(sprintf("\n[DBTARF-protocols] Done. Results in: %s\n", results_dir))
