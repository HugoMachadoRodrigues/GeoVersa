# =============================================================================
# Benchmark_DADRK.R
#
# DAD-RK: Design-Aware Density-Reweighted Kriging.
#
# The algorithm keeps the GeoVersa Auto v5 architecture and changes the
# objective: supervised losses are reweighted by a hybrid spatial/environmental
# density proxy so clustered calibration regions cannot dominate a model whose
# target claim is DesignBased map accuracy.
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
	  WADOUX_PROTOCOLS = "DesignBased",
	  WADOUX_INCLUDE_POPULATION = "FALSE",
	  WADOUX_N_ITER = "3",
	  WADOUX_SAMPLE_SIZE = "500",
	  WADOUX_MODEL_PROFILE = "auto",
	  WADOUX_TRAIN_SEED = "123",
	  WADOUX_DEVICE = "cpu",
	  WADOUX_RESULTS_DIR = "results/geoversa_dadrk_paired_designbased_3iter",
	  WADOUX_PAIRED_FINAL_BENCHMARK = "TRUE",
	  WADOUX_REFIT_FULL_CALIBRATION = "TRUE",
	  WADOUX_RF_REFIT_FULL_CALIBRATION = "TRUE",
	  WADOUX_REFIT_SELECTION_SPLITS = "3",
	  WADOUX_RF_QRF = "FALSE",
	  WADOUX_CONFORMAL_THIRD_SPLIT = "FALSE",
	  GEOVERSA_AUTO_META_CONTROLLER = "1",
	  GEOVERSA_AUTO_TAB_QBASIS = "1",
  GEOVERSA_AUTO_DESIGN_WEIGHTS = "1",
  GEOVERSA_AUTO_DESIGN_WEIGHT_MODE = "hybrid",
  GEOVERSA_AUTO_DESIGN_WEIGHT_POWER = "1.0",
  GEOVERSA_AUTO_DESIGN_WEIGHT_ENV_SHARE = "0.5",
  GEOVERSA_AUTO_DESIGN_WEIGHT_CLIP_MIN = "0.35",
  GEOVERSA_AUTO_DESIGN_WEIGHT_CLIP_MAX = "2.75"
)

source("code/run_wadoux_style_rf_conv_comparison.R")
