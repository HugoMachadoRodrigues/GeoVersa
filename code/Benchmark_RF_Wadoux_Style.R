Sys.setenv(
  # RF only — same protocol as ConvKrigingNet2D benchmark for fair comparison.
  # Wadoux (2021) published RF with 500 iterations; we use 3 here to match
  # the ConvKrigingNet2D benchmark and show the comparison is fair.
  WADOUX_MODELS = "RF",
  WADOUX_SCENARIO = "random",
  # Same 3 protocols as ConvKrigingNet2D benchmark.
  # BLOOCV excluded for consistency (excluded from ConvKrigingNet2D due to cost).
  # Population excluded: RF prediction over 1.3M cells is fast but we disabled
  # it for ConvKrigingNet2D, so excluding here keeps the comparison symmetric.
  WADOUX_PROTOCOLS = "DesignBased,RandomKFold,SpatialKFold",
  WADOUX_INCLUDE_POPULATION = "FALSE",
  # Same 3 iterations and same sample size as ConvKrigingNet2D benchmark.
  WADOUX_N_ITER = "3",
  WADOUX_SAMPLE_SIZE = "500",
  WADOUX_MODEL_PROFILE = "full",  # irrelevant for RF, but kept for consistency
  WADOUX_TRAIN_SEED = "123",      # same seed -> same 3 calibration samples
  WADOUX_DEVICE = "cpu",
  WADOUX_RESULTS_DIR = "results/wadoux2021_rf_random_validation"
)
source("code/run_wadoux_style_rf_conv_comparison.R")
