source("code/ConvKrigingNet2D.R")

res_conv2d_aniso_multiseed <- run_convkriging2d_anisotropic_multiseed_confirmation(
  context = wadoux_context,
  sample_size = 300,
  n_folds = 10,
  max_splits = 10,
  train_seeds = c(11, 29, 47),
  results_dir = "results/convkriging2d_anisotropic_multiseed_confirmation",
  save_outputs = TRUE
)

res_conv2d_aniso_multiseed %>%
  dplyr::group_by(train_seed, model) %>%
  dplyr::summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    R2_mean = mean(R2, na.rm = TRUE),
    MAE_mean = mean(MAE, na.rm = TRUE),
    Bias_mean = mean(Bias, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(train_seed, RMSE_mean)

res_conv2d_aniso_multiseed %>%
  dplyr::group_by(model) %>%
  dplyr::summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    R2_mean = mean(R2, na.rm = TRUE),
    MAE_mean = mean(MAE, na.rm = TRUE),
    Bias_mean = mean(Bias, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(RMSE_mean)

res_conv2d_aniso_multiseed %>%
  dplyr::group_by(train_seed, model) %>%
  dplyr::summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    R2_mean = mean(R2, na.rm = TRUE),
    MAE_mean = mean(MAE, na.rm = TRUE),
    Bias_mean = mean(Bias, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(train_seed, RMSE_mean)

res_conv2d_aniso_multiseed %>%
  dplyr::group_by(train_seed, model) %>%
  dplyr::summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    MAE_mean = mean(MAE, na.rm = TRUE),
    R2_mean = mean(R2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(train_seed, RMSE_mean)


