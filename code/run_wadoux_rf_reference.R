rm(list = ls())

source("./code/wadoux2021_rf_reproduction_helpers.R")

parse_int_env_wadoux <- function(name, default) {
  as.integer(Sys.getenv(name, unset = as.character(default)))
}

parse_num_env_wadoux <- function(name, default) {
  as.numeric(Sys.getenv(name, unset = as.character(default)))
}

parse_protocols_env_wadoux <- function(name,
                                       default = c("Population", "DesignBased", "RandomKFold", "SpatialKFold", "BLOOCV")) {
  raw <- Sys.getenv(name, unset = paste(default, collapse = ","))
  vals <- trimws(unlist(strsplit(raw, ",")))
  vals <- vals[nzchar(vals)]
  allowed <- c("Population", "DesignBased", "RandomKFold", "SpatialKFold", "BLOOCV")
  invalid <- setdiff(vals, allowed)
  if (length(invalid) > 0) {
    stop("Unsupported protocol(s) in ", name, ": ", paste(invalid, collapse = ", "))
  }
  unique(vals)
}

scenario <- Sys.getenv("WADOUX_RF_SCENARIO", unset = "random")
r2_method <- tolower(Sys.getenv("WADOUX_R2_METHOD", unset = "pearson"))
r2_method <- match.arg(r2_method, c("pearson", "spearman"))
options(wadoux_corr_method = r2_method)
protocols <- parse_protocols_env_wadoux("WADOUX_RF_PROTOCOLS")

sample_size <- parse_int_env_wadoux("WADOUX_RF_SAMPLE_SIZE", 500L)
n_iter <- parse_int_env_wadoux("WADOUX_RF_N_ITER", 500L)
val_dist_km <- parse_num_env_wadoux("WADOUX_RF_VAL_DIST_KM", 350)
random_k <- parse_int_env_wadoux("WADOUX_RF_RANDOM_K", 10L)
bloo_groups <- parse_int_env_wadoux("WADOUX_RF_BLOO_GROUPS", 10L)
bloo_test_pixels <- parse_int_env_wadoux("WADOUX_RF_BLOO_TEST_PIXELS", 100L)
kmeans_centers <- parse_int_env_wadoux("WADOUX_RF_CLUSTER_CENTERS", 100L)
n_psu <- parse_int_env_wadoux("WADOUX_RF_N_PSU", 20L)
m_per_psu <- parse_int_env_wadoux("WADOUX_RF_M_PER_PSU", 25L)
checkpoint_every <- parse_int_env_wadoux("WADOUX_RF_CHECKPOINT_EVERY", 10L)

if (identical(scenario, "clustered_random")) {
  sample_size <- n_psu * m_per_psu
}

results_dir <- Sys.getenv(
  "WADOUX_RF_RESULTS_DIR",
  unset = file.path(
    project_root_wadoux,
    "results",
    sprintf("wadoux2021_rf_reference_%s_%s", scenario, r2_method)
  )
)

upstream_commit <- tryCatch(
  system2("git", c("-C", file.path("external", "SpatialValidation"), "rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE)[1],
  error = function(e) NA_character_
)
upstream_commit <- trimws(upstream_commit)
if (!nzchar(upstream_commit)) {
  upstream_commit <- NA_character_
}

need_polygon <- identical(scenario, "regular_grid")
cat(sprintf("Loading Wadoux common data for scenario: %s\n", scenario))
common_data <- load_wadoux_common_data(include_polygon = need_polygon)

sample_fn <- switch(
  scenario,
  random = function(common_data, sample_size) {
    sample_simple_random_rows_wadoux(common_data$s_df, size = sample_size)
  },
  regular_grid = sample_valuetable_regular_grid_wadoux,
  clustered_random = sample_valuetable_clustered_random_wadoux,
  stop("Unsupported WADOUX_RF_SCENARIO: ", scenario)
)

sample_args <- switch(
  scenario,
  random = list(sample_size = sample_size),
  regular_grid = list(sample_size = sample_size),
  clustered_random = list(
    kmeans_centers = kmeans_centers,
    n_psu = n_psu,
    m_per_psu = m_per_psu
  )
)

prefix <- sprintf("wadoux2021_rf_%s", scenario)

run_wadoux_rf_reproduction(
  common_data = common_data,
  sample_valuetable_fn = sample_fn,
  sample_args = sample_args,
  results_dir = results_dir,
  prefix = prefix,
  scenario_name = scenario,
  config_extra = list(
    r2_method = r2_method,
    upstream_repo = "https://github.com/AlexandreWadoux/SpatialValidation",
    upstream_commit = upstream_commit
  ),
  protocols = protocols,
  sample_size = sample_size,
  n_iter = n_iter,
  val_dist_km = val_dist_km,
  random_k = random_k,
  bloo_groups = bloo_groups,
  bloo_test_pixels = bloo_test_pixels,
  checkpoint_every = checkpoint_every
)

cat("\n=== Wadoux RF reference reproduction complete ===\n")
cat(sprintf("Results written to: %s\n", results_dir))
