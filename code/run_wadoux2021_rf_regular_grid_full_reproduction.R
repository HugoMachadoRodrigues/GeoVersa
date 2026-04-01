rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")

# =============================================================================
# run_wadoux2021_rf_regular_grid_full_reproduction.R
#
# Full RF reproduction of the regular-grid sampling arm from Wadoux et al.
# (2021), using:
#   - regular sampling over polygon `pp`
#   - RF = ranger(formula, data) with defaults
#   - Population, DesignBased, RandomKFold, SpatialKFold, BLOOCV
#   - 500 repetitions by default
#
# Environment overrides:
#   WADOUX_N_ITER
#   WADOUX_SAMPLE_SIZE
#   WADOUX_VAL_DIST_KM
#   WADOUX_RANDOM_K
#   WADOUX_BLOO_GROUPS
#   WADOUX_BLOO_TEST_PIXELS
#   WADOUX_RESULTS_DIR
# =============================================================================

SAMPLE_SIZE <- as.integer(Sys.getenv("WADOUX_SAMPLE_SIZE", unset = "500"))
N_ITER <- as.integer(Sys.getenv("WADOUX_N_ITER", unset = "500"))
VAL_DIST_KM <- as.numeric(Sys.getenv("WADOUX_VAL_DIST_KM", unset = "350"))
RANDOM_K <- as.integer(Sys.getenv("WADOUX_RANDOM_K", unset = "10"))
BLOO_GROUPS <- as.integer(Sys.getenv("WADOUX_BLOO_GROUPS", unset = "10"))
BLOO_TEST_PIXELS <- as.integer(Sys.getenv("WADOUX_BLOO_TEST_PIXELS", unset = "100"))
RESULTS_DIR <- Sys.getenv(
  "WADOUX_RESULTS_DIR",
  unset = file.path(project_root_wadoux, "results", "wadoux2021_rf_regular_grid_full_reproduction")
)

cat("Loading SpatialValidation raster stack and polygon for regular-grid reproduction...\n")
common_data <- load_wadoux_common_data(include_polygon = TRUE)

out <- run_wadoux_rf_reproduction(
  common_data = common_data,
  sample_valuetable_fn = sample_valuetable_regular_grid_wadoux,
  sample_args = list(sample_size = SAMPLE_SIZE),
  results_dir = RESULTS_DIR,
  prefix = "wadoux2021_rf_regular_grid",
  scenario_name = "regular_grid",
  config_extra = list(
    sampling_design = "regular_grid"
  ),
  sample_size = SAMPLE_SIZE,
  n_iter = N_ITER,
  val_dist_km = VAL_DIST_KM,
  random_k = RANDOM_K,
  bloo_groups = BLOO_GROUPS,
  bloo_test_pixels = BLOO_TEST_PIXELS,
  checkpoint_every = 10L
)

cat("\n=== Full Wadoux-style RF regular-grid reproduction complete ===\n")
print(out$delta_summary)
cat(sprintf("\nResults written to: %s\n", RESULTS_DIR))
