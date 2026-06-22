# =============================================================================
# run_geobart_safe_external.R
#
# RStudio-safe launcher for long GeoBART benchmarks.
#
# The actual benchmark runs in a separate Rscript process. This keeps the
# RStudio console free while still writing checkpoint, audit, tuning, variogram,
# and log files into GEOBART_RESULTS_DIR.
# =============================================================================

set_default_env <- function(name, value) {
  if (!nzchar(Sys.getenv(name, unset = ""))) {
    do.call(Sys.setenv, stats::setNames(list(value), name))
  }
}

set_default_env("PROJ_LIB", "/opt/homebrew/share/proj")
set_default_env("GDAL_DATA", "/opt/homebrew/share/gdal")
set_default_env("GEOBART_MODELS", "RF,GeoBART")
set_default_env("GEOBART_PROTOCOLS", "DesignBased")
set_default_env("GEOBART_N_ITER", "30")
set_default_env("GEOBART_SAMPLE_SIZE", "500")
set_default_env(
  "GEOBART_RESULTS_DIR",
  "results/geobart_autotune_v3_confirm30_fix_selection"
)

set_default_env("GEOBART_TUNE", "TRUE")
set_default_env("GEOBART_AUTOTUNE_V2", "TRUE")
set_default_env("GEOBART_AUTOTUNE_V3", "TRUE")
set_default_env("GEOBART_TUNE_FULL_GRID", "TRUE")
set_default_env("GEOBART_TUNE_INNER_FOLDS", "3")
set_default_env("GEOBART_TUNE_STAGE2_TOP_K", "10")
set_default_env("GEOBART_TUNE_STAGE2_MAX_CANDIDATES", "240")
set_default_env("GEOBART_V3_STAGE2_FULL_GRID", "FALSE")
set_default_env("GEOBART_V3_STAGE2_N_SEEDS", "3")
set_default_env("GEOBART_V3_STAGE2_SELECT_ONLY", "FALSE")
set_default_env("GEOBART_V3_RMSE_SD_WEIGHT", "0.25")
set_default_env("GEOBART_V3_ABS_ME_WEIGHT", "0.10")
set_default_env(
  "GEOBART_V3_CALIBRATION_POLICIES",
  "none,mean_bias,median_bias,affine"
)
set_default_env("GEOBART_REFIT_FULL_CALIBRATION", "TRUE")
set_default_env("GEOBART_REFIT_N_SEEDS", "5")
set_default_env("GEOBART_NUM_WORKERS", "4")
set_default_env("GEOBART_SAVE_VARIOGRAMS", "TRUE")
set_default_env("GEOBART_SAVE_PREDICTION_AUDIT", "TRUE")
set_default_env("GEOBART_CHECKPOINT_EVERY", "1")

# These critical settings must override stale values left in the parent RStudio
# session. set_default_env() is intentionally gentle, but here stale values would
# change the tuning policy and invalidate the benchmark.
Sys.setenv(
  GEOBART_V3_STAGE2_SELECT_ONLY = "FALSE",
  GEOBART_V3_STAGE2_FULL_GRID = "FALSE",
  GEOBART_REFIT_FULL_CALIBRATION = "TRUE",
  GEOBART_SAVE_PREDICTION_AUDIT = "TRUE",
  GEOBART_SAVE_VARIOGRAMS = "TRUE"
)

results_dir <- Sys.getenv("GEOBART_RESULTS_DIR")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

pid_file <- file.path(results_dir, "geobart_safe_external.pid")
pid_is_alive <- function(pid) {
  pid <- suppressWarnings(as.integer(pid))
  if (!is.finite(pid) || pid <= 0L) return(FALSE)
  identical(system2("kill", c("-0", as.character(pid)),
                    stdout = FALSE, stderr = FALSE), 0L)
}

log_file <- file.path(
  results_dir,
  paste0("geobart_safe_external_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log")
)

if (!requireNamespace("processx", quietly = TRUE)) {
  stop("Package 'processx' is required for the safe external launcher.")
}

cmd <- file.path(R.home("bin"), "Rscript")
args <- c("code/run_geobart_wadoux_comparison.R")
existing_pid <- if (file.exists(pid_file)) {
  suppressWarnings(as.integer(readLines(pid_file, warn = FALSE)[1L]))
} else {
  NA_integer_
}

if (pid_is_alive(existing_pid)) {
  latest_log <- tail(sort(list.files(
    results_dir,
    pattern = "^geobart_safe_external_.*\\.log$",
    full.names = TRUE
  )), 1L)
  assign("GEOBART_SAFE_PROCESS", NULL, envir = .GlobalEnv)
  assign("GEOBART_SAFE_PID", existing_pid, envir = .GlobalEnv)
  assign("GEOBART_SAFE_LOG", if (length(latest_log)) latest_log else log_file,
         envir = .GlobalEnv)
  assign("GEOBART_SAFE_RESULTS_DIR", results_dir, envir = .GlobalEnv)
  cat("[GeoBART safe launcher] Existing external GeoBART process is already running.\n")
  cat(sprintf("[GeoBART safe launcher] PID: %s\n", existing_pid))
  cat(sprintf("[GeoBART safe launcher] Results: %s\n", results_dir))
  cat("[GeoBART safe launcher] Not starting a duplicate process.\n")
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
  assign("GEOBART_SAFE_PROCESS", proc, envir = .GlobalEnv)
  assign("GEOBART_SAFE_PID", proc$get_pid(), envir = .GlobalEnv)
  assign("GEOBART_SAFE_LOG", log_file, envir = .GlobalEnv)
  assign("GEOBART_SAFE_RESULTS_DIR", results_dir, envir = .GlobalEnv)

  cat("[GeoBART safe launcher] Started external Rscript process.\n")
  cat(sprintf("[GeoBART safe launcher] PID: %s\n", proc$get_pid()))
  cat(sprintf("[GeoBART safe launcher] Log: %s\n", log_file))
  cat(sprintf("[GeoBART safe launcher] Results: %s\n", results_dir))
}

cat("[GeoBART safe launcher] Check status with:\n")
cat("  if (!is.null(GEOBART_SAFE_PROCESS)) GEOBART_SAFE_PROCESS$is_alive()\n")
cat("  tail(readLines(GEOBART_SAFE_LOG), 30)\n")
cat("  read.csv(file.path(GEOBART_SAFE_RESULTS_DIR, 'geobart_checkpoint_status.csv'))\n")
cat("  tail(read.csv(file.path(GEOBART_SAFE_RESULTS_DIR, 'geobart_tuning_scores.csv')), 10)\n")
