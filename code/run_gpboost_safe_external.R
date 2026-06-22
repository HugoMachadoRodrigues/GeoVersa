# =============================================================================
# run_gpboost_safe_external.R
#
# RStudio-safe launcher for the GPBoost benchmark.
#
# GPBoost can crash the host R session through native C++/threading failures.
# This launcher intentionally does not load gpboost. It starts the real runner
# in a separate Rscript process and writes logs/progress to GPB_RESULTS_DIR.
# If the child process crashes, RStudio stays alive.
# =============================================================================

set_default_env <- function(name, value) {
  if (!nzchar(Sys.getenv(name, unset = ""))) {
    do.call(Sys.setenv, stats::setNames(list(value), name))
  }
}

set_default_env("PROJ_LIB", "/opt/homebrew/share/proj")
set_default_env("GDAL_DATA", "/opt/homebrew/share/gdal")
set_default_env("GPB_MODELS", "RF,GPBoost")
set_default_env("GPB_PROTOCOLS", "DesignBased")
set_default_env("GPB_N_ITER", "3")
set_default_env("GPB_SAMPLE_SIZE", "500")
set_default_env("GPB_RESULTS_DIR", "results/gpboost_safe_external_confirm3")

# Safe defaults: expand the candidate universe, but evaluate it by staged
# AutoTune; avoid multithreaded native GPBoost calls inside each candidate.
set_default_env("GPB_FAIR_TUNE", "TRUE")
set_default_env("GPB_REFIT_FULL", "TRUE")
set_default_env("GPB_TUNE_INNER_FOLDS", "3")
set_default_env("GPB_TUNE_FULL_GRID", "TRUE")
set_default_env("GPB_AUTOTUNE_V2", "TRUE")
set_default_env("GPB_CRASH_SAFE", "TRUE")
set_default_env("GPB_SPATIAL_POLICY", "tree_only")
set_default_env("GPB_STAGE1_MAX_CANDIDATES", "96")
set_default_env("GPB_STAGE2_TOP_K", "12")
set_default_env("GPB_STAGE2_MAX_CANDIDATES", "48")
set_default_env("GPB_STAGE2_SELECT_ONLY", "FALSE")
set_default_env("GPB_RMSE_SD_WEIGHT", "0.25")
set_default_env("GPB_QUANTILE_BINS", "5")
set_default_env("GPB_QUANTILE_EXCESS_WEIGHT", "0.15")
set_default_env("GPB_QUANTILE_SD_WEIGHT", "0.05")
set_default_env("GPB_SAVE_PREDICTION_AUDIT", "TRUE")
set_default_env("GPB_NROUNDS_MAX", "600")
set_default_env("GPB_EARLY_STOP", "60")
set_default_env("GPB_NUM_THREADS", "1")
set_default_env("GPB_TUNE_MAX_SECONDS", "2400")
set_default_env("GPB_TUNE_MIN_CANDIDATES_BEFORE_TIMEOUT", "48")
set_default_env("GPB_CHECKPOINT_EVERY", "1")
set_default_env("GPB_TUNE_LR_GRID", "0.02,0.03,0.05,0.08,0.10")
set_default_env("GPB_TUNE_DEPTH_GRID", "3,4,5,6,7,8")
set_default_env("GPB_TUNE_FEATURE_GRID", "0.60,0.75,0.875,0.95,1.00")
set_default_env("GPB_TUNE_MIN_LEAF_GRID", "3,5,10,15,20,30")
set_default_env("GPB_TUNE_BAGGING_GRID", "0.70,0.80,0.90,1.00")
set_default_env("GPB_TUNE_L2_GRID", "0,0.1,1,5,10")
set_default_env("GPB_SELECTION_OBJECTIVE", "rmse")
set_default_env("GPB_POLICY_BLEND", "FALSE")
set_default_env("GPB_BLEND_ALPHA_GRID", "0,0.1,0.25,0.5,0.75,1")
set_default_env("GPB_RESIDUAL_SPATIAL_CORRECTION", "FALSE")
set_default_env("GPB_RESIDUAL_GAMMA_GRID", "0,0.25,0.5,0.75,1")
set_default_env("GPB_SEED_ENSEMBLE_N", "1")
set_default_env("GPB_SPATIAL_FILTER", "TRUE")

# Force the benchmark-critical settings. GPBoost runs in a child R process, but
# it still inherits stale environment values from the parent RStudio session.
force_defaults <- toupper(Sys.getenv("GPB_SAFE_FORCE_DEFAULTS", unset = "TRUE")) %in%
  c("TRUE", "1", "YES", "Y")
if (isTRUE(force_defaults)) {
  Sys.setenv(
    GPB_FAIR_TUNE = "TRUE",
    GPB_REFIT_FULL = "TRUE",
    GPB_TUNE_INNER_FOLDS = "3",
    GPB_TUNE_FULL_GRID = "TRUE",
    GPB_AUTOTUNE_V2 = "TRUE",
    GPB_CRASH_SAFE = "TRUE",
    GPB_SPATIAL_POLICY = "tree_only",
    GPB_STAGE1_MAX_CANDIDATES = "96",
    GPB_STAGE2_TOP_K = "12",
    GPB_STAGE2_MAX_CANDIDATES = "48",
    GPB_STAGE2_SELECT_ONLY = "FALSE",
    GPB_RMSE_SD_WEIGHT = "0.25",
    GPB_QUANTILE_BINS = "5",
    GPB_QUANTILE_EXCESS_WEIGHT = "0.15",
    GPB_QUANTILE_SD_WEIGHT = "0.05",
    GPB_SAVE_PREDICTION_AUDIT = "TRUE",
    GPB_NROUNDS_MAX = "600",
    GPB_EARLY_STOP = "60",
    GPB_NUM_THREADS = "1",
    GPB_TUNE_MAX_SECONDS = "2400",
    GPB_TUNE_MIN_CANDIDATES_BEFORE_TIMEOUT = "48",
    GPB_TREE_FALLBACK = "TRUE",
    GPB_USE_GP_VAL = "FALSE",
    GPB_TUNE_LR_GRID = "0.02,0.03,0.05,0.08,0.10",
    GPB_TUNE_DEPTH_GRID = "3,4,5,6,7,8",
    GPB_TUNE_FEATURE_GRID = "0.60,0.75,0.875,0.95,1.00",
    GPB_TUNE_MIN_LEAF_GRID = "3,5,10,15,20,30",
    GPB_TUNE_BAGGING_GRID = "0.70,0.80,0.90,1.00",
    GPB_TUNE_L2_GRID = "0,0.1,1,5,10"
  )
}

results_dir <- Sys.getenv("GPB_RESULTS_DIR")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
set_default_env(
  "GPB_TUNE_PROGRESS_FILE",
  file.path(results_dir, "gpboost_tuning_progress_live.csv")
)
Sys.setenv(GPB_TUNE_PROGRESS_FILE = file.path(
  results_dir, "gpboost_tuning_progress_live.csv"
))

pid_file <- file.path(results_dir, "gpboost_safe_external.pid")
pid_is_alive <- function(pid) {
  pid <- suppressWarnings(as.integer(pid))
  if (!is.finite(pid) || pid <= 0L) return(FALSE)
  identical(system2("kill", c("-0", as.character(pid)),
                    stdout = FALSE, stderr = FALSE), 0L)
}

make_gpboost_status_helpers <- function(results_dir, log_file) {
  read_csv_safe <- function(path) {
    if (!file.exists(path)) return(NULL)
    tryCatch(utils::read.csv(path, stringsAsFactors = FALSE),
             error = function(e) NULL)
  }

  is_alive <- function() {
    pid <- if (exists("GPBOOST_SAFE_PID", envir = .GlobalEnv,
                      inherits = FALSE)) {
      get("GPBOOST_SAFE_PID", envir = .GlobalEnv)
    } else {
      NA_integer_
    }
    pid <- suppressWarnings(as.integer(pid))
    if (!is.finite(pid) || pid <= 0L) return(FALSE)
    identical(system2("kill", c("-0", as.character(pid)),
                      stdout = FALSE, stderr = FALSE), 0L)
  }

  status <- function(n = 30L) {
    n <- suppressWarnings(as.integer(n))
    if (!is.finite(n) || n < 1L) n <- 30L
    pid <- if (exists("GPBOOST_SAFE_PID", envir = .GlobalEnv,
                      inherits = FALSE)) {
      get("GPBOOST_SAFE_PID", envir = .GlobalEnv)
    } else {
      NA_integer_
    }
    log_path <- if (exists("GPBOOST_SAFE_LOG", envir = .GlobalEnv,
                           inherits = FALSE)) {
      get("GPBOOST_SAFE_LOG", envir = .GlobalEnv)
    } else {
      log_file
    }
    progress_path <- file.path(results_dir, "gpboost_tuning_progress_live.csv")
    cat(sprintf("[GPBoost safe status] PID: %s\n", pid))
    cat(sprintf("[GPBoost safe status] alive: %s\n", is_alive()))
    cat(sprintf("[GPBoost safe status] results: %s\n", results_dir))
    cat(sprintf("[GPBoost safe status] log: %s\n", log_path))
    if (file.exists(log_path)) {
      cat("\n--- log tail ---\n")
      print(utils::tail(readLines(log_path, warn = FALSE), n))
    } else {
      cat("\n--- log tail ---\nlog file not found yet\n")
    }
    if (file.exists(progress_path)) {
      cat("\n--- tuning progress tail ---\n")
      print(utils::tail(utils::read.csv(progress_path), min(10L, n)))
    } else {
      cat("\n--- tuning progress tail ---\nprogress file not found yet\n")
    }
    invisible(is_alive())
  }

  live <- function(n_log = 12L) {
    n_log <- suppressWarnings(as.integer(n_log))
    if (!is.finite(n_log) || n_log < 1L) n_log <- 12L
    pid <- if (exists("GPBOOST_SAFE_PID", envir = .GlobalEnv,
                      inherits = FALSE)) {
      get("GPBOOST_SAFE_PID", envir = .GlobalEnv)
    } else {
      NA_integer_
    }
    log_path <- if (exists("GPBOOST_SAFE_LOG", envir = .GlobalEnv,
                           inherits = FALSE)) {
      get("GPBOOST_SAFE_LOG", envir = .GlobalEnv)
    } else {
      log_file
    }
    progress_path <- file.path(results_dir, "gpboost_tuning_progress_live.csv")
    diag_path <- file.path(results_dir, "gpboost_diagnostics.csv")
    summary_path <- file.path(results_dir, "gpboost_summary_by_protocol.csv")
    delta_path <- file.path(results_dir, "gpboost_paired_deltas_vs_RF.csv")

    cat("\n================ GPBoost live check ================\n")
    cat(sprintf("alive       : %s\n", is_alive()))
    cat(sprintf("pid         : %s\n", pid))
    cat(sprintf("results_dir : %s\n", results_dir))
    cat(sprintf("log         : %s\n", log_path))

    progress <- read_csv_safe(progress_path)
    if (!is.null(progress) && nrow(progress) > 0L) {
      cat("\n-- tuning progress --\n")
      cat(sprintf("candidates evaluated: %d\n", nrow(progress)))
      if ("mode" %in% names(progress)) {
        mode_txt <- paste(names(table(progress$mode)), as.integer(table(progress$mode)),
                          sep = "=", collapse = " | ")
        cat(sprintf("modes               : %s\n", mode_txt))
      }
      if ("tune_stage" %in% names(progress)) {
        stage_txt <- paste(names(table(progress$tune_stage)),
                           as.integer(table(progress$tune_stage)),
                           sep = "=", collapse = " | ")
        cat(sprintf("stages              : %s\n", stage_txt))
      }
      last <- progress[nrow(progress), , drop = FALSE]
      last_cols <- intersect(c("candidate_id", "mode", "tune_stage", "status",
                               "val_rmse", "rf_inner_rmse",
                               "delta_rmse_vs_rf", "selection_score"),
                             names(last))
      if (length(last_cols) > 0L) {
        cat("last candidate      :\n")
        print(last[, last_cols, drop = FALSE], row.names = FALSE)
      }
      if ("selection_score" %in% names(progress)) {
        sc <- suppressWarnings(as.numeric(progress$selection_score))
        ok <- is.finite(sc)
        if (any(ok)) {
          best <- progress[which.min(ifelse(ok, sc, Inf)), , drop = FALSE]
          best_cols <- intersect(c("candidate_id", "mode", "tune_stage",
                                   "val_rmse", "rf_inner_rmse",
                                   "delta_rmse_vs_rf", "selection_score"),
                                 names(best))
          cat("best so far         :\n")
          print(best[, best_cols, drop = FALSE], row.names = FALSE)
        }
      }
    } else {
      cat("\n-- tuning progress --\n")
      cat("progress file not written yet. This is normal in the first seconds.\n")
    }

    diag <- read_csv_safe(diag_path)
    if (!is.null(diag) && nrow(diag) > 0L) {
      cat("\n-- completed GPBoost iterations --\n")
      cat(sprintf("completed iterations: %d\n", nrow(diag)))
      diag_cols <- intersect(c("iteration", "fit_mode", "selected_fit_mode",
                               "selection_objective", "selected_candidate",
                               "selected_policy_blend_alpha",
                               "selected_residual_gamma",
                               "ensemble_applied", "ensemble_size",
                               "seed_ensemble_n", "refit_status",
                               "selected_delta_rmse_vs_rf"),
                             names(diag))
      print(utils::tail(diag[, diag_cols, drop = FALSE], 5L),
            row.names = FALSE)
    } else {
      cat("\n-- completed GPBoost iterations --\n")
      cat("none yet.\n")
    }

    summary <- read_csv_safe(summary_path)
    if (!is.null(summary) && nrow(summary) > 0L) {
      cat("\n-- current summary --\n")
      print(summary, row.names = FALSE)
    }

    delta <- read_csv_safe(delta_path)
    if (!is.null(delta) && nrow(delta) > 0L &&
        "delta_RMSE" %in% names(delta)) {
      d <- suppressWarnings(as.numeric(delta$delta_RMSE))
      cat("\n-- paired delta vs RF --\n")
      cat(sprintf("mean(delta_RMSE)    : %.4f  (negative favors GPBoost)\n",
                  mean(d, na.rm = TRUE)))
      cat(sprintf("GPBoost win rate    : %.3f\n", mean(d < 0, na.rm = TRUE)))
    }

    if (file.exists(log_path)) {
      lines <- readLines(log_path, warn = FALSE)
      useful <- grep("GPBoost|candidate=|Final summary|Done|Error|error|failed",
                     lines, value = TRUE)
      cat("\n-- useful log tail --\n")
      print(utils::tail(useful, n_log))
    }
    cat("====================================================\n")
    invisible(is_alive())
  }

  watch <- function(interval = 30L, max_checks = Inf) {
    interval <- suppressWarnings(as.numeric(interval))
    if (!is.finite(interval) || interval < 1) interval <- 30
    max_checks <- suppressWarnings(as.integer(max_checks))
    if (!is.finite(max_checks) || max_checks < 1L) max_checks <- .Machine$integer.max
    for (i in seq_len(max_checks)) {
      cat(sprintf("\n[GPBoost watch] check %d | %s\n",
                  i, format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
      alive_now <- live()
      if (!isTRUE(alive_now)) break
      Sys.sleep(interval)
    }
    invisible(NULL)
  }

  assign("GPBOOST_SAFE_IS_ALIVE", is_alive, envir = .GlobalEnv)
  assign("GPBOOST_SAFE_STATUS", status, envir = .GlobalEnv)
  assign("GPBOOST_SAFE_LIVE", live, envir = .GlobalEnv)
  assign("GPBOOST_SAFE_WATCH", watch, envir = .GlobalEnv)
}

log_file <- file.path(
  results_dir,
  paste0("gpboost_safe_external_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log")
)

if (!requireNamespace("processx", quietly = TRUE)) {
  stop("Package 'processx' is required for the safe external launcher.")
}

cmd <- file.path(R.home("bin"), "Rscript")
args <- c("code/run_gpboost_wadoux_comparison.R")
existing_pid <- if (file.exists(pid_file)) {
  suppressWarnings(as.integer(readLines(pid_file, warn = FALSE)[1L]))
} else {
  NA_integer_
}

if (pid_is_alive(existing_pid)) {
  latest_log <- tail(sort(list.files(results_dir, pattern = "^gpboost_safe_external_.*\\.log$",
                                     full.names = TRUE)), 1L)
  assign("GPBOOST_SAFE_PROCESS", NULL, envir = .GlobalEnv)
  assign("GPBOOST_SAFE_PID", existing_pid, envir = .GlobalEnv)
  assign("GPBOOST_SAFE_LOG", if (length(latest_log)) latest_log else log_file,
         envir = .GlobalEnv)
  assign("GPBOOST_SAFE_RESULTS_DIR", results_dir, envir = .GlobalEnv)
  make_gpboost_status_helpers(
    results_dir,
    if (length(latest_log)) latest_log else log_file
  )
  cat("[GPBoost safe launcher] Existing external GPBoost process is already running.\n")
  cat(sprintf("[GPBoost safe launcher] PID: %s\n", existing_pid))
  cat(sprintf("[GPBoost safe launcher] Results: %s\n", results_dir))
  cat("[GPBoost safe launcher] Not starting a duplicate process.\n")
} else {
  proc <- processx::process$new(
    command = cmd,
    args = args,
    wd = getwd(),
    stdout = log_file,
    stderr = log_file,
    supervise = TRUE,
    cleanup = FALSE
  )
  writeLines(as.character(proc$get_pid()), pid_file)
  assign("GPBOOST_SAFE_PROCESS", proc, envir = .GlobalEnv)
  assign("GPBOOST_SAFE_PID", proc$get_pid(), envir = .GlobalEnv)
  assign("GPBOOST_SAFE_LOG", log_file, envir = .GlobalEnv)
  assign("GPBOOST_SAFE_RESULTS_DIR", results_dir, envir = .GlobalEnv)
  make_gpboost_status_helpers(results_dir, log_file)

  cat("[GPBoost safe launcher] Started external Rscript process.\n")
  cat(sprintf("[GPBoost safe launcher] PID: %s\n", proc$get_pid()))
  cat(sprintf("[GPBoost safe launcher] Log: %s\n", log_file))
  cat(sprintf("[GPBoost safe launcher] Results: %s\n", results_dir))
}

cat("[GPBoost safe launcher] Check status with:\n")
cat("  GPBOOST_SAFE_IS_ALIVE()\n")
cat("  GPBOOST_SAFE_LIVE()\n")
cat("  GPBOOST_SAFE_WATCH(interval = 30)\n")
cat("  GPBOOST_SAFE_STATUS(30)\n")
cat("  tail(readLines(GPBOOST_SAFE_LOG), 30)\n")
cat(sprintf("  tail(read.csv('%s'), 10)\n", Sys.getenv("GPB_TUNE_PROGRESS_FILE")))
