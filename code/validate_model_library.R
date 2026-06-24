# =============================================================================
# validate_model_library.R
#
# ONE honest, standardized design-based benchmark for the whole GeoVersa model
# library vs the Wadoux (2021) default RF -- the same validation logic that
# proved DB-TARF and GPBoost, applied uniformly to every model.
#
# For each model x sampling design (random / clustered_random), it runs the
# model's own runner (paired vs RF, same calibration/validation draws), then
# aggregates the per-iteration paired ΔRMSE into: n, mean ΔRMSE (model - RF),
# win-rate (model strictly better), paired Wilcoxon signed-rank p, and an HONEST
# verdict (beats_rf / ties_rf / loses_rf / run_failed). Writes one comparison
# table: results/model_library_benchmark.csv.
#
# Run from the repository that carries the gitignored data/ (the primary repo):
#   LIB_N_ITER=10 LIB_SCENARIOS=random,clustered_random Rscript code/validate_model_library.R
# Set LIB_RUN=0 to skip running and only re-aggregate existing results/lib_* dirs.
# =============================================================================

n_iter <- as.integer(Sys.getenv("LIB_N_ITER", unset = "10"))
if (!is.finite(n_iter) || n_iter < 1L) n_iter <- 10L
scenarios <- trimws(unlist(strsplit(
  Sys.getenv("LIB_SCENARIOS", unset = "random,clustered_random"), ",")))
scenarios <- scenarios[nzchar(scenarios)]
seed0 <- as.integer(Sys.getenv("LIB_TRAIN_SEED", unset = "123"))
do_run <- !identical(Sys.getenv("LIB_RUN", unset = "1"), "0")
# Torch device for the deep models (cpu | mps | cuda) and a per-model wall-clock
# timeout so a slow/stuck model is killed and marked, never blocking the campaign.
lib_device <- Sys.getenv("LIB_DEVICE", unset = "cpu")
timeout_sec <- as.integer(Sys.getenv("LIB_TIMEOUT_SEC", unset = "2400"))
if (!is.finite(timeout_sec) || timeout_sec < 60L) timeout_sec <- 2400L
dir.create("results", showWarnings = FALSE)

# --- model registry: runner + env + the (model, RF) labels in all_results.csv -
# `extra` keeps each model's run modest/crash-safe; tune as needed.
models <- list(
  list(key = "DBTARF", runner = "code/run_dbtarf_wadoux_comparison.R",
       prefix = "DBT", models_val = "RF,DBTARF",
       model_label = "DBTARF", rf_label = "RF",
       extra = c()),
  list(key = "GPBoost", runner = "code/run_gpboost_wadoux_comparison.R",
       prefix = "GPB", models_val = "RF,GPBoost",
       model_label = "GPBoost", rf_label = "RF",
       extra = c()),
  list(key = "GeoBART", runner = "code/run_geobart_wadoux_comparison.R",
       prefix = "GEOBART", models_val = "RF,GeoBART",
       model_label = "GeoBART_500", rf_label = "RF_500",
       extra = c(GEOBART_TUNE_FULL_GRID = "FALSE",
                 GEOBART_TUNE_MAX_CANDIDATES = "12",
                 GEOBART_REFIT_N_SEEDS = "1",
                 GEOBART_AUTOTUNE_V2 = "FALSE", GEOBART_AUTOTUNE_V3 = "FALSE",
                 GEOBART_AUTOTUNE_V4 = "FALSE")),
  list(key = "DeepKernel", runner = "code/run_deepkernel_wadoux_comparison.R",
       prefix = "DKP", models_val = "RF,DeepKernelProcess",
       model_label = "DeepKernelProcess_500", rf_label = "RF_500",
       # reduced tuning so the exact-GP stays tractable on CPU
       extra = c(DKP_DEVICE = lib_device, DKP_TUNE_MAX_CANDIDATES = "4",
                 DKP_TUNE_INNER_FOLDS = "1")),
  list(key = "NeuroMatern", runner = "code/run_neuromatern_wadoux_comparison.R",
       prefix = "NMK", models_val = "RF,NeuroMaternKrige",
       model_label = "NeuroMaternKrige_500", rf_label = "RF_500",
       # NMK's default is the full autotune grid (V2/V3/V4) x inner folds, which
       # is intractable on CPU; cut it to a small grid + single holdout.
       extra = c(NMK_DEVICE = lib_device, NMK_TUNE_FULL_GRID = "FALSE",
                 NMK_TUNE_MAX_CANDIDATES = "4", NMK_AUTOTUNE_V2 = "FALSE",
                 NMK_AUTOTUNE_V3 = "FALSE", NMK_AUTOTUNE_V4 = "FALSE",
                 NMK_TUNE_INNER_FOLDS = "1")),
  list(key = "ConvKrigingNet", runner = "code/Benchmark_GeoVersaMeta.R",
       prefix = "WADOUX", models_val = "RF,ConvKrigingNet2D",
       model_label = "GeoVersa_500", rf_label = "RF_500",
       # paired final benchmark; few selection splits to bound the deep-net cost.
       extra = c(WADOUX_DEVICE = lib_device, WADOUX_INCLUDE_POPULATION = "FALSE",
                 WADOUX_PAIRED_FINAL_BENCHMARK = "TRUE",
                 WADOUX_REFIT_FULL_CALIBRATION = "TRUE",
                 WADOUX_RF_REFIT_FULL_CALIBRATION = "TRUE",
                 WADOUX_REFIT_SELECTION_SPLITS = "2"))
)

# Optional subset: LIB_MODELS=DeepKernel,NeuroMatern runs only those.
lib_only <- trimws(unlist(strsplit(Sys.getenv("LIB_MODELS", unset = ""), ",")))
lib_only <- lib_only[nzchar(lib_only)]
if (length(lib_only) > 0L) models <- Filter(function(m) m$key %in% lib_only, models)

read_all_results <- function(dir) {
  f <- list.files(dir, pattern = "all_results.*\\.csv$", full.names = TRUE)
  if (length(f) == 0L) return(NULL)
  tryCatch(utils::read.csv(f[[1]], stringsAsFactors = FALSE),
           error = function(e) NULL)
}

# pair model vs RF per (iteration, scenario) and compute honest stats
summarise_model <- function(df, model_label, rf_label, scen) {
  if (is.null(df) || !all(c("iteration", "model", "RMSE", "scenario") %in% names(df)))
    return(NULL)
  d <- df[df$scenario == scen, , drop = FALSE]
  mm <- d[d$model == model_label, c("iteration", "RMSE")]
  rr <- d[d$model == rf_label, c("iteration", "RMSE")]
  if (nrow(mm) == 0L || nrow(rr) == 0L) return(NULL)
  m <- merge(mm, rr, by = "iteration", suffixes = c("_model", "_rf"))
  if (nrow(m) == 0L) return(NULL)
  delta <- m$RMSE_model - m$RMSE_rf            # < 0 => model better than RF
  n <- length(delta)
  wins <- sum(delta < -1e-9); ties <- sum(abs(delta) <= 1e-9)
  p <- if (n >= 5L && stats::sd(delta) > 0)
    suppressWarnings(stats::wilcox.test(m$RMSE_model, m$RMSE_rf,
                                        paired = TRUE)$p.value) else NA_real_
  mean_d <- mean(delta)
  verdict <- if (is.finite(p) && p < 0.05 && mean_d < 0) "beats_rf" else
             if (is.finite(p) && p < 0.05 && mean_d > 0) "loses_rf" else "ties_rf"
  data.frame(scenario = scen, n = n, mean_dRMSE = round(mean_d, 3),
             win_rate = sprintf("%d/%d", wins, n), ties = ties,
             wilcoxon_p = signif(p, 3), verdict = verdict,
             stringsAsFactors = FALSE)
}

rows <- list()
for (mc in models) {
  for (scen in scenarios) {
    rdir <- file.path("results", paste0("lib_", tolower(mc$key), "_", scen))
    status <- "ok"
    if (do_run) {
      unlink(rdir, recursive = TRUE); dir.create(rdir, recursive = TRUE, showWarnings = FALSE)
      env <- c(
        stats::setNames(mc$models_val, paste0(mc$prefix, "_MODELS")),
        stats::setNames("DesignBased", paste0(mc$prefix, "_PROTOCOLS")),
        stats::setNames(scen, paste0(mc$prefix, "_SCENARIO")),
        stats::setNames(as.character(n_iter), paste0(mc$prefix, "_N_ITER")),
        stats::setNames("500", paste0(mc$prefix, "_SAMPLE_SIZE")),
        stats::setNames(as.character(seed0), paste0(mc$prefix, "_TRAIN_SEED")),
        stats::setNames(rdir, paste0(mc$prefix, "_RESULTS_DIR")),
        mc$extra)
      cat(sprintf("[lib] running %-12s scenario=%-16s n_iter=%d ...\n",
                  mc$key, scen, n_iter))
      do.call(Sys.setenv, as.list(stats::setNames(as.character(env), names(env))))
      logp <- file.path(rdir, "run.log")
      res <- tryCatch(
        processx::run("Rscript", mc$runner, timeout = timeout_sec,
                      error_on_status = FALSE, stdout = logp, stderr = "2>&1"),
        error = function(e) {
          if (inherits(e, "system_command_timeout_error") ||
              grepl("timed?.?out|timeout", conditionMessage(e), ignore.case = TRUE))
            list(timeout = TRUE)
          else list(timeout = FALSE, status = 999L, msg = conditionMessage(e))
        })
      if (isTRUE(res$timeout)) {
        status <- sprintf("timed_out(%ds)", timeout_sec)
      } else if (!identical(as.integer(res$status), 0L)) {
        status <- sprintf("run_failed(rc=%s)", res$status)
      }
    }
    s <- summarise_model(read_all_results(rdir), mc$model_label, mc$rf_label, scen)
    if (is.null(s)) {
      rows[[length(rows) + 1L]] <- data.frame(
        model = mc$key, scenario = scen, n = 0L, mean_dRMSE = NA_real_,
        win_rate = NA_character_, ties = NA_integer_, wilcoxon_p = NA_real_,
        verdict = if (status == "ok") "no_results" else status,
        stringsAsFactors = FALSE)
    } else {
      s <- cbind(model = mc$key, s)
      if (status != "ok") s$verdict <- status
      rows[[length(rows) + 1L]] <- s
    }
    # incremental write: keep results/model_library_benchmark.csv current after
    # every model x scenario, so an interrupted/overnight/crashing run never
    # leaves a stale table and the completed cells are never lost.
    utils::write.csv(do.call(rbind, rows),
                     "results/model_library_benchmark.csv", row.names = FALSE)
    cat(sprintf("[lib] done %-12s %-16s -> %s\n",
                mc$key, scen, rows[[length(rows)]]$verdict[1]))
  }
}

tbl <- do.call(rbind, rows)
out <- "results/model_library_benchmark.csv"
utils::write.csv(tbl, out, row.names = FALSE)
cat("\n===== GeoVersa model library vs Wadoux RF (DesignBased) =====\n")
print(tbl, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", out))
cat("verdict: beats_rf = mean ΔRMSE<0 & Wilcoxon p<0.05; loses_rf = >0 & p<0.05; else ties_rf (honest, no cherry-picking).\n")
