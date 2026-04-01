Sys.setenv(
  # RF is excluded: Wadoux already published RF results for 500 iterations.
  # Goal is to show ConvKrigingNet2D is competitive using the SAME protocol.
  WADOUX_MODELS = "ConvKrigingNet2D",
  WADOUX_SCENARIO = "random",
  # Replicates Wadoux (2021): Population is the true reference; each protocol's
  # metric is reported as (protocol - Population), same as Wadoux's paper.
  # BLOOCV excluded: requires 1000 complete model trainings per iteration —
  # feasible for RF but ~50h per iteration for ConvKrigingNet2D.
  # This limitation is noted explicitly in the paper.
  WADOUX_PROTOCOLS = "DesignBased,RandomKFold,SpatialKFold",
  # Population disabled: requires assembling a 4D patch array of ~66 GB
  # (1.31M points x 28 channels x 15x15 pixels) — exceeds available RAM.
  # Noted as a limitation in the paper alongside BLOOCV.
  WADOUX_INCLUDE_POPULATION = "FALSE",
  # 3 iterations (vs 500 in Wadoux) — sufficient to show stability.
  # Paper reports mean +/- sd across 3 draws alongside Wadoux's published RF.
  WADOUX_N_ITER = "1",
  WADOUX_SAMPLE_SIZE = "500",
  # "n500" profile: same architecture as multiseed confirmation +
  # 4 targeted fixes for n=500: warmup(8), base_loss(0.05), batch(48), lr_decay(0.5)
  WADOUX_MODEL_PROFILE = "n500",
  WADOUX_TRAIN_SEED = "123",
  WADOUX_DEVICE = "cpu",
  WADOUX_RESULTS_DIR = "results/wadoux2021_conv_random_validation_n500_krig2"
)
source("code/run_wadoux_style_rf_conv_comparison.R")
