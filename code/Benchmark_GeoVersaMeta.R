# =============================================================================
# Benchmark_GeoVersaMeta.R
#
# Runs GeoVersa Auto v5 with the internal module meta-controller enabled.
# The controller decides optional branches from dataset signals; explicit
# GEOVERSA_AUTO_* environment variables still override its decisions.
# =============================================================================

set_if_unset <- function(...) {
  vals <- list(...)
  for (nm in names(vals)) {
    if (!nzchar(Sys.getenv(nm, unset = ""))) {
      do.call(Sys.setenv, stats::setNames(list(as.character(vals[[nm]])), nm))
    }
  }
}

set_if_unset(
  WADOUX_AUTO_V5_SCRIPT = normalizePath(
    file.path(getwd(), "code", "ConvKrigingNet2D_Auto_v5.R"), mustWork = FALSE),
  WADOUX_AUTO_SCRIPT = normalizePath(
    file.path(getwd(), "code", "ConvKrigingNet2D_Auto.R"), mustWork = FALSE),
  WADOUX_MODELS = "RF,ConvKrigingNet2D",
  WADOUX_SCENARIO = "random",
  WADOUX_PROTOCOLS = "DesignBased,RandomKFold",
  WADOUX_INCLUDE_POPULATION = "FALSE",
  WADOUX_N_ITER = "3",
  WADOUX_SAMPLE_SIZE = "500",
  WADOUX_MODEL_PROFILE = "auto",
  WADOUX_TRAIN_SEED = "123",
  WADOUX_DEVICE = "mps",
  WADOUX_RESULTS_DIR = "results/geoversa_meta_designbased_3iter",
  GEOVERSA_AUTO_META_CONTROLLER = "1"
)

source("code/run_wadoux_style_rf_conv_comparison.R")
