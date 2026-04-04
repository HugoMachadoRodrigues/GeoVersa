# =============================================================================
# Benchmark_Auto_v5.R
#
# Complete automatic configuration benchmark.
# NO HYPERPARAMETER TUNING. ALL parameters derived from data.
#
# v5 Features:
#   ✅ Learning rate from gradient statistics
#   ✅ Batch size from GPU memory  
#   ✅ Early stopping from training dynamics
#   ✅ Coordinate embedding from anisotropy
#   ✅ Weight decay from model capacity
#
# Comparison: RF vs ConvKrigingNet2D_Auto_v5
# Protocols: DesignBased, RandomKFold, SpatialKFold
# N_ITER: 50 (paper-quality run)
# =============================================================================

Sys.setenv(
  WADOUX_AUTO_V5_SCRIPT = normalizePath(
    file.path(getwd(), "code", "ConvKrigingNet2D_Auto_v5.R"), mustWork = FALSE),
  WADOUX_AUTO_SCRIPT = normalizePath(
    file.path(getwd(), "code", "ConvKrigingNet2D_Auto.R"), mustWork = FALSE),
  WADOUX_MODELS            = "RF,ConvKrigingNet2D",
  WADOUX_SCENARIO          = "random",
  WADOUX_PROTOCOLS         = "DesignBased,RandomKFold,SpatialKFold",
  WADOUX_INCLUDE_POPULATION = "FALSE",
  WADOUX_N_ITER            = "50",       # Paper-quality: 50 independent iterations
  WADOUX_SAMPLE_SIZE       = "500",
  WADOUX_MODEL_PROFILE     = "n500",
  WADOUX_TRAIN_SEED        = "123",
  WADOUX_DEVICE            = "mps",
  WADOUX_RESULTS_DIR       = "results/wadoux2021_auto_v5_50iter"
)

source("code/run_wadoux_style_rf_conv_comparison.R")

