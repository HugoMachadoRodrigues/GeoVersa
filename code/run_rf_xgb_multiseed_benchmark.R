# =============================================================================
# run_rf_xgb_multiseed_benchmark.R
#
# Roda RF e XGB no MESMO benchmark exato que foi usado para ConvKrigingNet2D e
# Cubist na confirmação multiseed (seeds 11, 29, 47).
#
# Como funciona:
#   - Para cada train_seed, reconstrói o benchmark com seed=train_seed
#     (idêntico ao que run_convkriging2d_anisotropic_multiseed_confirmation fez)
#   - Treina RF e XGB em cada split (subtrain → test)
#   - Salva rf_results.csv e xgb_results.csv nos diretórios seed_XX/
#   - Ao final, gera arquivos consolidados na raiz do diretório de resultados
#
# Saída:
#   results/convkriging2d_anisotropic_multiseed_confirmation/
#     rf_xgb_multiseed_all.csv      — todos os folds × seeds × modelos
#     rf_xgb_multiseed_by_seed.csv  — média por seed × modelo
#     rf_xgb_multiseed_summary.csv  — média geral por modelo
#     seed_11/rf_results.csv
#     seed_11/xgb_results.csv
#     seed_29/rf_results.csv
#     seed_29/xgb_results.csv
#     seed_47/rf_results.csv
#     seed_47/xgb_results.csv
# =============================================================================

rm(list = ls())
set.seed(123)

source("./code/KrigingNet_PointPatchCNN.R")   # carrega wadoux_context e helpers

# Pacotes
pkgs <- c("ranger", "xgboost", "dplyr")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install, repos = "https://cloud.r-project.org")

library(ranger)
library(xgboost)
library(dplyr)

# =============================================================================
# Parâmetros — DEVEM ser idênticos ao run_convkriging2d_anisotropic_multiseed_confirmation
# =============================================================================
SAMPLE_SIZE  <- 300
N_FOLDS      <- 10
VAL_DIST_KM  <- 350
VAL_FRAC     <- 0.2
MAX_SPLITS   <- 10
TRAIN_SEEDS  <- c(11, 29, 47)
RESULTS_DIR  <- "results/convkriging2d_anisotropic_multiseed_confirmation"

# Hiperparâmetros dos modelos
XGB_NROUNDS  <- 500
XGB_PARAMS   <- list(
  objective        = "reg:squarederror",
  eta              = 0.05,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  verbosity        = 0
)

# =============================================================================
# Helpers
# =============================================================================

metrics <- function(obs, pred) {
  n    <- length(obs)
  rmse <- sqrt(mean((obs - pred)^2))
  mae  <- mean(abs(obs - pred))
  bias <- mean(pred - obs)
  r2   <- if (n >= 2) cor(obs, pred)^2 else NA_real_
  rpiq <- if (n >= 2) IQR(obs) / (rmse + 1e-12) else NA_real_
  data.frame(N = n, R2 = r2, RMSE = rmse, MAE = mae, Bias = bias, RPIQ = rpiq)
}

run_rf_on_benchmark <- function(benchmark, context) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp         <- benchmark$splits[[i]]
    subtrain   <- sp$train_df[sp$train_idx, , drop = FALSE]
    test_df    <- sp$test_df

    y_te  <- test_df[[context$response]]
    form_rf <- reformulate(context$predictors, response = context$response)

    set.seed(42)
    rf_mod  <- ranger::ranger(formula = form_rf, data = subtrain)
    pred_rf <- as.numeric(predict(rf_mod, data = test_df)$predictions)

    results[[i]] <- metrics(y_te, pred_rf) %>%
      mutate(
        model = "RF",
        baseline_spec = "ranger_defaults_wadoux_style",
        protocol = "spatial_kfold",
        split = sp$split_id
      )

    cat(sprintf("[RF] split %s — RMSE=%.2f  R2=%.3f\n",
                sp$split_id, results[[i]]$RMSE, results[[i]]$R2))
  }

  bind_rows(results)
}

run_xgb_on_benchmark <- function(benchmark, context,
                                  nrounds = 500, params = XGB_PARAMS) {
  results <- vector("list", length(benchmark$splits))

  for (i in seq_along(benchmark$splits)) {
    sp       <- benchmark$splits[[i]]
    subtrain <- sp$train_df[sp$train_idx, , drop = FALSE]
    test_df  <- sp$test_df

    X_tr <- as.matrix(subtrain[, context$predictors, drop = FALSE])
    y_tr <- subtrain[[context$response]]
    X_te <- as.matrix(test_df[, context$predictors, drop = FALSE])
    y_te <- test_df[[context$response]]

    scaler   <- fit_scaler(X_tr, robust = TRUE)
    X_tr_s   <- apply_scaler(X_tr, scaler)
    X_te_s   <- apply_scaler(X_te, scaler)

    dtrain   <- xgb.DMatrix(data = X_tr_s, label = y_tr)
    dtest    <- xgb.DMatrix(data = X_te_s)

    set.seed(42)
    xgb_mod  <- xgb.train(params = params, data = dtrain,
                           nrounds = nrounds, verbose = 0)
    pred_xgb <- predict(xgb_mod, dtest)

    results[[i]] <- metrics(y_te, pred_xgb) %>%
      mutate(
        model = "XGB",
        baseline_spec = "additional_baseline",
        protocol = "spatial_kfold",
        split = sp$split_id
      )

    cat(sprintf("[XGB] split %s — RMSE=%.2f  R2=%.3f\n",
                sp$split_id, results[[i]]$RMSE, results[[i]]$R2))
  }

  bind_rows(results)
}

# =============================================================================
# Loop principal: um benchmark por seed
# =============================================================================

all_results <- vector("list", length(TRAIN_SEEDS))

for (i in seq_along(TRAIN_SEEDS)) {
  ts       <- TRAIN_SEEDS[[i]]
  seed_dir <- file.path(RESULTS_DIR, paste0("seed_", ts))
  dir.create(seed_dir, recursive = TRUE, showWarnings = FALSE)

  cat(sprintf("\n========================================\n"))
  cat(sprintf("RF + XGB  |  train_seed = %d\n", ts))
  cat(sprintf("========================================\n"))

  # Reconstrói o benchmark EXATO desse seed
  # (mesmo seed=ts que run_convkriging2d_anisotropic_confirmation usou)
  benchmark <- build_pointpatch_fixed_spatial_kfold_benchmark(
    context    = wadoux_context,
    sample_size = SAMPLE_SIZE,
    sampling   = "simple_random",
    n_folds    = N_FOLDS,
    val_dist_km = VAL_DIST_KM,
    val_frac   = VAL_FRAC,
    max_splits = MAX_SPLITS,
    seed       = ts
  )

  # RF
  cat("\n--- Random Forest (ranger defaults; Wadoux-style) ---\n")
  rf_res  <- run_rf_on_benchmark(benchmark, wadoux_context)

  # XGB
  cat("\n--- XGBoost ---\n")
  xgb_res <- run_xgb_on_benchmark(benchmark, wadoux_context,
                                   nrounds = XGB_NROUNDS, params = XGB_PARAMS)

  # Salva nos diretórios seed_XX/
  write.csv(rf_res,  file.path(seed_dir, "rf_results.csv"),  row.names = FALSE)
  write.csv(xgb_res, file.path(seed_dir, "xgb_results.csv"), row.names = FALSE)

  all_results[[i]] <- bind_rows(
    rf_res  %>% mutate(train_seed = ts),
    xgb_res %>% mutate(train_seed = ts)
  )

  cat(sprintf("\nSeed %d done.\n", ts))
}

# =============================================================================
# Consolida e salva arquivos agregados
# =============================================================================

final_rf_xgb <- bind_rows(all_results)

summary_by_seed <- final_rf_xgb %>%
  group_by(train_seed, model, baseline_spec) %>%
  summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    R2_mean   = mean(R2,   na.rm = TRUE),
    MAE_mean  = mean(MAE,  na.rm = TRUE),
    Bias_mean = mean(Bias, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(train_seed, RMSE_mean)

summary_overall <- final_rf_xgb %>%
  group_by(model, baseline_spec) %>%
  summarise(
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    R2_mean   = mean(R2,   na.rm = TRUE),
    MAE_mean  = mean(MAE,  na.rm = TRUE),
    Bias_mean = mean(Bias, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(RMSE_mean)

write.csv(final_rf_xgb,   file.path(RESULTS_DIR, "rf_xgb_multiseed_all.csv"),     row.names = FALSE)
write.csv(summary_by_seed, file.path(RESULTS_DIR, "rf_xgb_multiseed_by_seed.csv"), row.names = FALSE)
write.csv(summary_overall, file.path(RESULTS_DIR, "rf_xgb_multiseed_summary.csv"), row.names = FALSE)

cat("\n=== RF + XGB multiseed benchmark concluido ===\n")
print(summary_overall)
cat(sprintf("\nResultados em: %s\n", RESULTS_DIR))
