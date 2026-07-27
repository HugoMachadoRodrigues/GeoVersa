# =============================================================================
# maps_compute.R  —  STEP 1 of 2 (expensive)
#
# Fits the Wadoux default RF and DB-TARF on ONE calibration sample and predicts
# both over the whole 1,310,143-cell Amazonian grid, so the manuscript can show
# the actual maps rather than only aggregate statistics.
#
# Writes results/maps_v030/map_predictions.rds with, per grid cell:
#   x, y, truth (ABG1), rf, dbtarf, err_rf, err_dbtarf
# plus the calibration sample used, so the plotting step (maps_plot.R) can be
# re-run cheaply while iterating on the figure design.
#
# Env:
#   MAP_SEED        calibration seed, default 123
#   MAP_N           calibration sample size, default 500
#   MAP_AGG         aggregate the grid by this integer factor before predicting
#                   (1 = native 1 km resolution, default 1). Use 2-4 if memory
#                   is tight: the kernel/kriging correction paths cost
#                   O(n_grid x n_cal) distances.
#   MAP_CHUNK       prediction chunk size for the RF baseline, default 200000
# =============================================================================

root <- "/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/GeoVersa"
setwd(root)

suppressPackageStartupMessages({
  source("./code/wadoux2021_rf_reproduction_helpers.R")
  source("./R/dbtarf.R")
  library(ranger)
})

getnum <- function(k, d) { v <- suppressWarnings(as.numeric(Sys.getenv(k, unset = as.character(d)))); if (is.finite(v)) v else d }
seed  <- as.integer(getnum("MAP_SEED", 123))
n_cal <- as.integer(getnum("MAP_N", 500))
agg   <- as.integer(getnum("MAP_AGG", 1))
chunk <- as.integer(getnum("MAP_CHUNK", 200000))

out_dir <- "results/maps_v030"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

options(wadoux_corr_method = "pearson")
common <- load_wadoux_common_data()
predList <- predList_modelfull_wadoux
form_rf <- as.formula(paste(response_name_wadoux, "~", paste(predList, collapse = "+")))

grid <- common$s_df
message(sprintf("[maps] full grid: %d cells", nrow(grid)))

# optional thinning (keeps every agg-th cell in x and y) --------------------
if (agg > 1L) {
  ux <- sort(unique(grid$x)); uy <- sort(unique(grid$y))
  keep_x <- ux[seq(1, length(ux), by = agg)]
  keep_y <- uy[seq(1, length(uy), by = agg)]
  grid <- grid[grid$x %in% keep_x & grid$y %in% keep_y, , drop = FALSE]
  message(sprintf("[maps] thinned by %d -> %d cells", agg, nrow(grid)))
}

# ---- calibration sample -----------------------------------------------------

set.seed(seed)
cal <- sample_simple_random_rows_wadoux(common$s_df, n_cal)
message(sprintf("[maps] calibration sample: n = %d (seed %d)", nrow(cal), seed))

# ---- RF baseline ------------------------------------------------------------

message("[maps] fitting the Wadoux default RF ...")
rf <- fit_rf_default_wadoux(cal, form_rf)

message("[maps] predicting RF over the grid ...")
pred_rf <- numeric(nrow(grid))
idx <- split(seq_len(nrow(grid)), ceiling(seq_len(nrow(grid)) / chunk))
for (i in seq_along(idx)) {
  pred_rf[idx[[i]]] <- predict_rf_default_wadoux(rf, grid[idx[[i]], , drop = FALSE])
  message(sprintf("  RF chunk %d/%d", i, length(idx)))
}

# ---- DB-TARF ----------------------------------------------------------------
# One call, full grid as test_df: the correction is selected once on the
# calibration sample, so the map is a single coherent model (chunking here
# would refit and select per chunk, which is not the same estimator).

message("[maps] fitting DB-TARF and predicting over the grid (single call) ...")
t0 <- Sys.time()
db <- dbtarf_train_predict(
  train_df        = cal,
  test_df         = grid,
  response_name   = response_name_wadoux,
  predictor_names = predList,
  coord_names     = c("x", "y"),
  n_folds         = 5L,
  rf_tune_max_grid = 18L,
  resid_rf_tune_max_grid = 12L,
  fair_profile_max = 6L,
  ensemble_top_k  = 3L,
  train_seed      = seed)
message(sprintf("[maps] DB-TARF done in %.1f min | selected mode = %s, lambda = %s",
                as.numeric(difftime(Sys.time(), t0, units = "mins")),
                db$diagnostics$selected_mode[1],
                format(db$diagnostics$selected_lambda[1])))

# ---- assemble ---------------------------------------------------------------

truth <- grid[[response_name_wadoux]]
res <- data.frame(
  x         = grid$x,
  y         = grid$y,
  truth     = truth,
  rf        = pred_rf,
  dbtarf    = as.numeric(db$pred_test),
  stringsAsFactors = FALSE)
res$err_rf     <- res$rf     - res$truth
res$err_dbtarf <- res$dbtarf - res$truth

saveRDS(list(
  maps = res,
  cal  = cal[, c("x", "y", response_name_wadoux)],
  meta = list(seed = seed, n_cal = n_cal, agg = agg,
              selected_mode = db$diagnostics$selected_mode[1],
              selected_lambda = db$diagnostics$selected_lambda[1],
              pi_q90 = db$pi_q90)
), file.path(out_dir, "map_predictions.rds"))

# ---- console summary --------------------------------------------------------

rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))
cat("\n========== whole-grid (population) performance ==========\n")
cat(sprintf("  RF      : RMSE %.3f   ME %+.3f\n", rmse(res$err_rf), mean(res$err_rf, na.rm = TRUE)))
cat(sprintf("  DB-TARF : RMSE %.3f   ME %+.3f\n", rmse(res$err_dbtarf), mean(res$err_dbtarf, na.rm = TRUE)))
cat(sprintf("  delta RMSE = %+.3f\n", rmse(res$err_dbtarf) - rmse(res$err_rf)))
cat(sprintf("\nwrote %s\n", file.path(out_dir, "map_predictions.rds")))
