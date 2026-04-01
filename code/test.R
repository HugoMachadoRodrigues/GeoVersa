source("code/KrigingNet_DualFramework.R")

variants <- make_krigingnet_variants()[c(
  "KrigingNet_v3_TabSmall",
  "KrigingNet_v3_TabSmall_Log",
  "KrigingNet_v4_LocalBeta_Log"
)]

ablation_wadoux <- run_empirical_krigingnet_ablation(
  wadoux_context,
  sampling = "simple_random",
  protocol = "spatial_kfold",
  sample_size = 500,
  variants = variants
)

summarise_comparison(ablation_wadoux)
