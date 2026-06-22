# =============================================================================
# run_gpboost_spatial_batch_external.R
#
# RStudio-safe launcher for spatial GPBoost benchmarks.
#
# Native spatial GPBoost can crash the host R session. This launcher does not
# load gpboost. It runs each benchmark iteration in a fresh Rscript child
# process, then combines the per-iteration CSV outputs in GPB_RESULTS_DIR.
# =============================================================================

set_default_env <- function(name, value) {
  if (!nzchar(Sys.getenv(name, unset = ""))) {
    do.call(Sys.setenv, stats::setNames(list(value), name))
  }
}

parse_bool_batch <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}

parse_int_batch <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  out <- suppressWarnings(as.integer(raw))
  if (!is.finite(out)) as.integer(default) else out
}

set_default_env("PROJ_LIB", "/opt/homebrew/share/proj")
set_default_env("GDAL_DATA", "/opt/homebrew/share/gdal")
set_default_env("GPB_MODELS", "RF,GPBoost")
set_default_env("GPB_PROTOCOLS", "DesignBased")
set_default_env("GPB_N_ITER", "3")
set_default_env("GPB_SAMPLE_SIZE", "500")
set_default_env("GPB_RESULTS_DIR", "results/gpboost_spatial_batch_external_confirm3")
set_default_env("GPB_SAVE_PREDICTION_AUDIT", "TRUE")
set_default_env("GPB_FAIR_TUNE", "TRUE")
set_default_env("GPB_REFIT_FULL", "TRUE")
set_default_env("GPB_TUNE_INNER_FOLDS", "3")
set_default_env("GPB_AUTOTUNE_V2", "TRUE")
set_default_env("GPB_CRASH_SAFE", "FALSE")
set_default_env("GPB_SPATIAL_POLICY", "mixed")
set_default_env("GPB_TREE_FALLBACK", "TRUE")
set_default_env("GPB_USE_GP_VAL", "FALSE")
set_default_env("GPB_NUM_THREADS", "1")
set_default_env("GPB_CHECKPOINT_EVERY", "1")
set_default_env("GPB_TUNE_FULL_GRID", "FALSE")
set_default_env("GPB_MAX_TUNE_CANDIDATES", "72")
set_default_env("GPB_STAGE1_MAX_CANDIDATES", "24")
set_default_env("GPB_STAGE2_TOP_K", "6")
set_default_env("GPB_STAGE2_MAX_CANDIDATES", "12")
set_default_env("GPB_STAGE2_SELECT_ONLY", "FALSE")
set_default_env("GPB_RMSE_SD_WEIGHT", "0.25")
set_default_env("GPB_QUANTILE_BINS", "5")
set_default_env("GPB_QUANTILE_EXCESS_WEIGHT", "0.15")
set_default_env("GPB_QUANTILE_SD_WEIGHT", "0.05")
set_default_env("GPB_ENSEMBLE_TOP_K", "5")
set_default_env("GPB_ENSEMBLE_WEIGHTING", "softmax")
set_default_env("GPB_ENSEMBLE_TEMPERATURE", "0.75")
set_default_env("GPB_NROUNDS_MAX", "400")
set_default_env("GPB_EARLY_STOP", "50")
set_default_env("GPB_TUNE_LR_GRID", "0.02,0.03,0.05,0.08")
set_default_env("GPB_TUNE_DEPTH_GRID", "3,4,5,6")
set_default_env("GPB_TUNE_FEATURE_GRID", "0.60,0.75,0.875,1.00")
set_default_env("GPB_TUNE_MIN_LEAF_GRID", "3,5,10,20")
set_default_env("GPB_TUNE_BAGGING_GRID", "0.70,0.85,1.00")
set_default_env("GPB_TUNE_L2_GRID", "0,0.1,1,5")
set_default_env("GPB_TUNE_MATERN_GRID", "0.5,1.5,2.5")
set_default_env("GPB_RANGE_INIT_GRID", "0.5,1,2")
set_default_env("GPB_NUGGET_INIT_GRID", "0.5,1,2")

if (!requireNamespace("processx", quietly = TRUE)) {
  stop("Package 'processx' is required for the GPBoost batch launcher.")
}

base_results_dir <- Sys.getenv("GPB_RESULTS_DIR")
n_iter_total <- parse_int_batch("GPB_N_ITER", 3L)
base_seed <- parse_int_batch("GPB_TRAIN_SEED", 123L)
retries <- parse_int_batch("GPB_BATCH_RETRIES", 1L)
continue_on_crash <- parse_bool_batch("GPB_BATCH_CONTINUE_ON_CRASH", TRUE)
poll_seconds <- parse_int_batch("GPB_BATCH_POLL_SECONDS", 20L)

dir.create(base_results_dir, recursive = TRUE, showWarnings = FALSE)
status_file <- file.path(base_results_dir, "gpboost_batch_status.csv")

status_rows <- list()
success_rows <- list()

append_status <- function(row) {
  status_rows[[length(status_rows) + 1L]] <<- row
  write.csv(do.call(rbind, status_rows), status_file, row.names = FALSE)
}

run_child_iteration <- function(iter, attempt) {
  child_dir <- file.path(
    base_results_dir,
    sprintf("iter_%03d_attempt_%02d", iter, attempt)
  )
  dir.create(child_dir, recursive = TRUE, showWarnings = FALSE)
  log_file <- file.path(child_dir, "gpboost_child.log")
  child_env <- c(
    GPB_N_ITER = "1",
    GPB_ITER_OFFSET = as.character(iter - 1L),
    GPB_TRAIN_SEED = as.character(base_seed),
    GPB_RESULTS_DIR = child_dir,
    GPB_TUNE_PROGRESS_FILE = file.path(child_dir, "gpboost_tuning_progress_live.csv"),
    GPB_ALLOW_EXTERNAL_PROGRESS_FILE = "FALSE",
    GPB_CHECKPOINT_EVERY = "1"
  )
  old_env <- Sys.getenv(names(child_env), unset = NA_character_)
  names(old_env) <- names(child_env)
  on.exit({
    for (nm in names(old_env)) {
      if (is.na(old_env[[nm]])) {
        Sys.unsetenv(nm)
      } else {
        do.call(Sys.setenv, stats::setNames(list(old_env[[nm]]), nm))
      }
    }
  }, add = TRUE)
  do.call(Sys.setenv, as.list(child_env))

  proc <- processx::process$new(
    command = file.path(R.home("bin"), "Rscript"),
    args = "code/run_gpboost_wadoux_comparison.R",
    wd = getwd(),
    stdout = log_file,
    stderr = log_file,
    supervise = TRUE,
    cleanup = FALSE
  )
  progress_file <- child_env[["GPB_TUNE_PROGRESS_FILE"]]
  last_progress_n <- 0L
  while (proc$is_alive()) {
    Sys.sleep(max(1L, poll_seconds))
    if (file.exists(progress_file)) {
      progress <- tryCatch(read.csv(progress_file, stringsAsFactors = FALSE),
                           error = function(e) NULL)
      if (!is.null(progress) && nrow(progress) > last_progress_n) {
        last_progress_n <- nrow(progress)
        cols <- intersect(c("candidate_id", "mode", "tune_stage", "status",
                            "selection_score", "val_rmse",
                            "elapsed_total_seconds"),
                          names(progress))
        cat(sprintf("[GPBoost batch launcher]   progress rows: %d\n",
                    nrow(progress)))
        print(utils::tail(progress[, cols, drop = FALSE], 3L))
      }
    } else {
      cat("[GPBoost batch launcher]   waiting for tuning progress...\n")
    }
  }
  proc$wait()
  exit_status <- proc$get_exit_status()
  result_file <- file.path(child_dir, "gpboost_all_results.csv")
  ok <- identical(exit_status, 0L) && file.exists(result_file)
  row <- data.frame(
    iteration = iter,
    attempt = attempt,
    child_dir = child_dir,
    log_file = log_file,
    exit_status = if (is.null(exit_status)) NA_integer_ else exit_status,
    success = ok,
    completed_at = format(Sys.time(), tz = "", usetz = TRUE),
    stringsAsFactors = FALSE
  )
  append_status(row)
  row
}

cat(sprintf("[GPBoost batch launcher] Results: %s\n", base_results_dir))
cat(sprintf("[GPBoost batch launcher] Iterations: %d  retries: %d\n",
            n_iter_total, retries))

for (iter in seq_len(n_iter_total)) {
  cat(sprintf("\n[GPBoost batch launcher] Iteration %d/%d\n",
              iter, n_iter_total))
  iter_success <- NULL
  for (attempt in seq_len(retries + 1L)) {
    cat(sprintf("[GPBoost batch launcher]   attempt %d/%d\n",
                attempt, retries + 1L))
    row <- run_child_iteration(iter, attempt)
    if (isTRUE(row$success[[1L]])) {
      iter_success <- row
      break
    }
    cat(sprintf("[GPBoost batch launcher]   child failed, exit_status=%s\n",
                row$exit_status[[1L]]))
  }
  if (is.null(iter_success)) {
    msg <- sprintf("GPBoost child failed for iteration %d after %d attempt(s).",
                   iter, retries + 1L)
    if (continue_on_crash) {
      warning(msg, immediate. = TRUE)
    } else {
      stop(msg)
    }
  } else {
    success_rows[[length(success_rows) + 1L]] <- iter_success
  }
}

read_child_csv <- function(child_dir, file, iteration) {
  path <- file.path(child_dir, file)
  if (!file.exists(path)) return(NULL)
  df <- read.csv(path, stringsAsFactors = FALSE)
  if ("iteration" %in% names(df)) df$iteration <- iteration
  df
}

combine_child_csv <- function(file) {
  pieces <- lapply(success_rows, function(row) {
    read_child_csv(row$child_dir[[1L]], file, row$iteration[[1L]])
  })
  pieces <- pieces[!vapply(pieces, is.null, logical(1))]
  if (length(pieces) == 0L) return(NULL)
  out <- do.call(rbind, pieces)
  write.csv(out, file.path(base_results_dir, file), row.names = FALSE)
  out
}

combined_results <- combine_child_csv("gpboost_all_results.csv")
combine_child_csv("gpboost_paired_deltas_vs_RF.csv")
combine_child_csv("gpboost_diagnostics.csv")
combine_child_csv("gpboost_candidate_scores.csv")
combine_child_csv("gpboost_prediction_audit.csv")
combine_child_csv("gpboost_split_audit.csv")

config_pieces <- lapply(success_rows, function(row) {
  cfg <- read_child_csv(row$child_dir[[1L]], "gpboost_config.csv",
                        row$iteration[[1L]])
  if (is.null(cfg)) return(NULL)
  cfg$batch_iteration <- row$iteration[[1L]]
  cfg$batch_child_dir <- row$child_dir[[1L]]
  cfg
})
config_pieces <- config_pieces[!vapply(config_pieces, is.null, logical(1))]
if (length(config_pieces) > 0L) {
  write.csv(do.call(rbind, config_pieces),
            file.path(base_results_dir, "gpboost_config.csv"),
            row.names = FALSE)
}

if (!is.null(combined_results) && nrow(combined_results) > 0L) {
  by_protocol <- aggregate(
    cbind(ME, RMSE, r2, MEC) ~ scenario + protocol + model,
    data = combined_results,
    FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                        sd = stats::sd(x, na.rm = TRUE))
  )
  stats_df <- do.call(cbind, lapply(c("ME", "RMSE", "r2", "MEC"),
                                    function(m) {
    mat <- by_protocol[[m]]
    setNames(as.data.frame(mat), c(paste0(m, "_mean"), paste0(m, "_sd")))
  }))
  summary_df <- cbind(by_protocol[, c("scenario", "protocol", "model")],
                      stats_df)
  write.csv(summary_df,
            file.path(base_results_dir, "gpboost_summary_by_protocol.csv"),
            row.names = FALSE)
}

write.csv(
  data.frame(
    completed_successful_iterations = length(success_rows),
    total_iterations = n_iter_total,
    failed_iterations = paste(
      setdiff(seq_len(n_iter_total),
              vapply(success_rows, `[[`, integer(1), "iteration")),
      collapse = "|"
    ),
    completed_at = format(Sys.time(), tz = "", usetz = TRUE),
    stringsAsFactors = FALSE
  ),
  file.path(base_results_dir, "gpboost_checkpoint_status.csv"),
  row.names = FALSE
)

cat("\n[GPBoost batch launcher] Done.\n")
cat(sprintf("[GPBoost batch launcher] Combined results: %s\n",
            base_results_dir))
cat("[GPBoost batch launcher] Status file:\n")
cat(sprintf("  %s\n", status_file))
