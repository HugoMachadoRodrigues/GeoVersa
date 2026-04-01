rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")

# =============================================================================
# run_wadoux2021_rf_clustered_random_full_reproduction.R
#
# Full RF reproduction of the clustered-random sampling arm from Wadoux et al.
# (2021), using:
#   - k-means PSUs over x/y coordinates
#   - two-stage clustered sampling with defaults: 100 centers, n=20, m=25
#   - RF = ranger(formula, data) with defaults
#   - Population, DesignBased, RandomKFold, SpatialKFold, BLOOCV
#   - 500 repetitions by default
#
# Environment overrides:
#   WADOUX_N_ITER
#   WADOUX_VAL_DIST_KM
#   WADOUX_RANDOM_K
#   WADOUX_BLOO_GROUPS
#   WADOUX_BLOO_TEST_PIXELS
#   WADOUX_CLUSTER_CENTERS
#   WADOUX_CLUSTER_N_PSU
#   WADOUX_CLUSTER_M_PER_PSU
#   WADOUX_RESULTS_DIR
# =============================================================================

N_ITER <- as.integer(Sys.getenv("WADOUX_N_ITER", unset = "500"))
VAL_DIST_KM <- as.numeric(Sys.getenv("WADOUX_VAL_DIST_KM", unset = "350"))
RANDOM_K <- as.integer(Sys.getenv("WADOUX_RANDOM_K", unset = "10"))
BLOO_GROUPS <- as.integer(Sys.getenv("WADOUX_BLOO_GROUPS", unset = "10"))
BLOO_TEST_PIXELS <- as.integer(Sys.getenv("WADOUX_BLOO_TEST_PIXELS", unset = "100"))
KMEANS_CENTERS <- as.integer(Sys.getenv("WADOUX_CLUSTER_CENTERS", unset = "100"))
N_PSU <- as.integer(Sys.getenv("WADOUX_CLUSTER_N_PSU", unset = "20"))
M_PER_PSU <- as.integer(Sys.getenv("WADOUX_CLUSTER_M_PER_PSU", unset = "25"))
SAMPLE_SIZE <- N_PSU * M_PER_PSU
RESULTS_DIR <- Sys.getenv(
  "WADOUX_RESULTS_DIR",
  unset = file.path(project_root_wadoux, "results", "wadoux2021_rf_clustered_random_full_reproduction")
)

cat("Loading SpatialValidation raster stack for clustered-random reproduction...\n")
common_data <- load_wadoux_common_data(include_polygon = FALSE)

out <- run_wadoux_rf_reproduction(
  common_data = common_data,
  sample_valuetable_fn = sample_valuetable_clustered_random_wadoux,
  sample_args = list(
    kmeans_centers = KMEANS_CENTERS,
    n_psu = N_PSU,
    m_per_psu = M_PER_PSU
  ),
  results_dir = RESULTS_DIR,
  prefix = "wadoux2021_rf_clustered_random",
  scenario_name = "clustered_random",
  config_extra = list(
    sampling_design = "clustered_random",
    kmeans_centers = KMEANS_CENTERS,
    n_psu = N_PSU,
    m_per_psu = M_PER_PSU
  ),
  sample_size = SAMPLE_SIZE,
  n_iter = N_ITER,
  val_dist_km = VAL_DIST_KM,
  random_k = RANDOM_K,
  bloo_groups = BLOO_GROUPS,
  bloo_test_pixels = BLOO_TEST_PIXELS,
  checkpoint_every = 10L
)

cat("\n=== Full Wadoux-style RF clustered-random reproduction complete ===\n")
print(out$delta_summary)
cat(sprintf("\nResults written to: %s\n", RESULTS_DIR))
