rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
})

project_root <- normalizePath(getwd(), mustWork = TRUE)
benchmark_script <- file.path(project_root, "code", "run_wadoux_style_rf_conv_comparison.R")
auto_script <- normalizePath(file.path(project_root, "code", "ConvKrigingNet2D_Auto.R"), mustWork = FALSE)
auto_v5_script <- normalizePath(file.path(project_root, "code", "ConvKrigingNet2D_Auto_v5.R"), mustWork = FALSE)

results_root <- Sys.getenv(
  "GEOVERSA_ABLATION_RESULTS_DIR",
  unset = file.path(project_root, "results", sprintf("geoversa_variogram_ablation_%s", format(Sys.Date(), "%Y%m%d")))
)
dir.create(results_root, recursive = TRUE, showWarnings = FALSE)

reference_dir <- Sys.getenv(
  "GEOVERSA_ABLATION_REFERENCE_DIR",
  unset = file.path(project_root, "results", "geoversa2_leaderensemble_confirm10_full_20260415")
)

scenario <- Sys.getenv("GEOVERSA_ABLATION_SCENARIO", unset = "random")
protocols <- Sys.getenv("GEOVERSA_ABLATION_PROTOCOLS", unset = "DesignBased")
sample_size <- as.integer(Sys.getenv("GEOVERSA_ABLATION_SAMPLE_SIZE", unset = "500"))
n_iter <- as.integer(Sys.getenv("GEOVERSA_ABLATION_N_ITER", unset = "10"))
device <- Sys.getenv("GEOVERSA_ABLATION_DEVICE", unset = Sys.getenv("WADOUX_DEVICE", unset = "cpu"))
model_profile <- Sys.getenv("GEOVERSA_ABLATION_PROFILE", unset = "auto")
train_seed <- as.integer(Sys.getenv("GEOVERSA_ABLATION_TRAIN_SEED", unset = "123"))
use_leader_ensemble <- tolower(Sys.getenv("GEOVERSA_ABLATION_USE_LEADER_ENSEMBLE", unset = "1")) %in% c("1", "true", "yes")
leader_members <- as.integer(Sys.getenv("GEOVERSA_ABLATION_LEADER_MEMBERS", unset = "3"))
random_variants <- as.integer(Sys.getenv("GEOVERSA_ABLATION_RANDOM_VARIANTS", unset = "5"))
rf_refit_full_calibration <- tolower(Sys.getenv("GEOVERSA_ABLATION_RF_REFIT_FULL_CALIBRATION", unset = "1")) %in% c("1", "true", "yes")
refit_full_calibration <- tolower(Sys.getenv("GEOVERSA_ABLATION_REFIT_FULL_CALIBRATION", unset = "0")) %in% c("1", "true", "yes")
use_ssl <- tolower(Sys.getenv("GEOVERSA_ABLATION_USE_SSL", unset = "0")) %in% c("1", "true", "yes")
ssl_patch_state <- trimws(Sys.getenv("GEOVERSA_ABLATION_SSL_PATCH_STATE", unset = Sys.getenv("GEOVERSA_SSL_PATCH_STATE", unset = "")))
fixed_patch_global_raw <- trimws(Sys.getenv("GEOVERSA_ABLATION_FIXED_PATCH_GLOBAL", unset = Sys.getenv("GEOVERSA_PATCH_GLOBAL_SIZE", unset = "")))
fixed_patch_local_raw <- trimws(Sys.getenv("GEOVERSA_ABLATION_FIXED_PATCH_LOCAL", unset = Sys.getenv("GEOVERSA_PATCH_LOCAL_SIZE", unset = "")))
target_model <- Sys.getenv(
  "GEOVERSA_ABLATION_TARGET_MODEL",
  unset = if (use_leader_ensemble) "GeoVersa2-LeaderEnsemble" else "ConvKrigingNet2D"
)

parse_bool_local <- function(x, default = FALSE) {
  if (!nzchar(x)) return(default)
  tolower(trimws(x)) %in% c("1", "true", "yes", "on")
}

with_env_overrides <- function(overrides, expr) {
  keys <- names(overrides)
  old <- Sys.getenv(keys, unset = NA_character_)
  on.exit({
    for (i in seq_along(keys)) {
      key <- keys[[i]]
      old_val <- old[[i]]
      if (is.na(old_val)) {
        Sys.unsetenv(key)
      } else {
        do.call(Sys.setenv, setNames(list(old_val), key))
      }
    }
  }, add = TRUE)
  if (length(overrides) > 0) {
    do.call(Sys.setenv, as.list(overrides))
  }
  force(expr)
}

odd_int <- function(x, lo = 3L, hi = Inf) {
  x <- as.integer(round(x))
  x <- max(lo, min(as.integer(hi), x))
  if (x %% 2L == 0L) x <- x - 1L
  max(lo, x)
}

default_local_from_global <- function(global_size) {
  local_size <- odd_int(round(global_size * 0.63), lo = 9L, hi = global_size)
  min(local_size, odd_int(global_size, lo = 9L, hi = global_size))
}

read_csv_required <- function(path) {
  if (!file.exists(path)) stop("Missing required file: ", path)
  read.csv(path, stringsAsFactors = FALSE)
}

reference_diag_path <- file.path(reference_dir, "wadoux_style_rf_conv_fold_diagnostics.csv")
reference_cfg_path <- file.path(reference_dir, "wadoux_style_rf_conv_config.csv")
reference_diag <- read_csv_required(reference_diag_path)
reference_cfg <- if (file.exists(reference_cfg_path)) read.csv(reference_cfg_path, stringsAsFactors = FALSE) else data.frame()
reference_diag_model <- reference_diag %>% filter(model == target_model)
if (nrow(reference_diag_model) < 1L) {
  stop("Target model not found in reference diagnostics: ", target_model)
}

fixed_patch_global <- suppressWarnings(as.integer(round(as.numeric(fixed_patch_global_raw))))
fixed_patch_local <- suppressWarnings(as.integer(round(as.numeric(fixed_patch_local_raw))))
if (!is.finite(fixed_patch_global) || fixed_patch_global <= 0L) {
  if (nrow(reference_cfg) > 0L && "fixed_patch_global_size" %in% names(reference_cfg)) {
    fixed_patch_global <- suppressWarnings(as.integer(reference_cfg$fixed_patch_global_size[1]))
  }
}
if (!is.finite(fixed_patch_global) || fixed_patch_global <= 0L) {
  fixed_patch_global <- odd_int(stats::median(reference_diag_model$patch_size, na.rm = TRUE), lo = 9L, hi = 31L)
}
if (!is.finite(fixed_patch_local) || fixed_patch_local <= 0L) {
  if (nrow(reference_cfg) > 0L && "fixed_patch_local_size" %in% names(reference_cfg)) {
    fixed_patch_local <- suppressWarnings(as.integer(reference_cfg$fixed_patch_local_size[1]))
  }
}
if (!is.finite(fixed_patch_local) || fixed_patch_local <= 0L) {
  fixed_patch_local <- default_local_from_global(fixed_patch_global)
}

diag_num <- function(col, default = NA_real_, fun = stats::median) {
  if (!col %in% names(reference_diag_model)) return(default)
  vals <- suppressWarnings(as.numeric(reference_diag_model[[col]]))
  vals <- vals[is.finite(vals)]
  if (length(vals) < 1L) return(default)
  as.numeric(fun(vals, na.rm = TRUE))
}

reference_nugget_mean <- diag_num("nugget_ratio", default = 0.25)
fixed_config <- list(
  warmup_epochs = as.integer(max(4L, round(diag_num("warmup_epochs_done", default = 7)))),
  base_loss_weight = as.numeric(max(0.05, 0.10 * reference_nugget_mean)),
  alpha_me = as.numeric(diag_num("alpha_me", default = 0.10)),
  lambda_cov = as.numeric(diag_num("lambda_cov", default = 0.02)),
  lr = as.numeric(diag_num("lr_init", default = 1e-3)),
  min_lr = as.numeric(diag_num("min_lr", default = 1e-6)),
  wd = as.numeric(diag_num("wd", default = 3e-3)),
  batch_size = as.integer(max(24L, round(diag_num("batch_size", default = 50)))),
  bank_refresh_every = 1L,
  patience = as.integer(max(6L, round(diag_num("patience", default = 9)))),
  lr_patience = as.integer(max(2L, round(diag_num("lr_patience", default = 3)))),
  lr_decay = as.numeric(diag_num("lr_decay", default = 0.45)),
  K_neighbors = as.integer(max(8L, round(diag_num("K_neighbors", default = 25)))),
  beta_init = as.numeric(diag_num("beta_init", default = 0.5)),
  patch_dim = as.integer(max(48L, round(diag_num("patch_dim", default = 128)))),
  d = as.integer(max(128L, round(diag_num("d", default = 192)))),
  tab_dropout = as.numeric(diag_num("tab_dropout", default = 0.25)),
  patch_dropout = as.numeric(diag_num("patch_dropout", default = 0.12)),
  coord_dim = as.integer(max(16L, round(diag_num("coord_dim", default = 45)))),
  coord_dropout = as.numeric(diag_num("coord_dropout", default = 0.08)),
  patch_size = fixed_patch_global,
  patch_local_size = fixed_patch_local
)

fixed_manifest <- data.frame(
  name = names(fixed_config),
  value = unlist(fixed_config, use.names = FALSE),
  stringsAsFactors = FALSE
)
write.csv(fixed_manifest, file.path(results_root, "geoversa_variogram_ablation_fixed_config.csv"), row.names = FALSE)

make_common_env <- function(run_dir) {
  env <- c(
    WADOUX_AUTO_V5_SCRIPT = auto_v5_script,
    WADOUX_AUTO_SCRIPT = auto_script,
    WADOUX_MODELS = "RF,ConvKrigingNet2D",
    WADOUX_PROTOCOLS = protocols,
    WADOUX_SCENARIO = scenario,
    WADOUX_N_ITER = as.character(n_iter),
    WADOUX_SAMPLE_SIZE = as.character(sample_size),
    WADOUX_MODEL_PROFILE = model_profile,
    WADOUX_DEVICE = device,
    WADOUX_TRAIN_SEED = as.character(train_seed),
    WADOUX_RESULTS_DIR = run_dir,
    WADOUX_REFIT_FULL_CALIBRATION = if (refit_full_calibration) "TRUE" else "FALSE",
    WADOUX_RF_REFIT_FULL_CALIBRATION = if (rf_refit_full_calibration) "TRUE" else "FALSE",
    WADOUX_PAIRED_FINAL_BENCHMARK = "FALSE",
    GEOVERSA2_LEADER_ENSEMBLE_BENCHMARK = if (use_leader_ensemble) "1" else "0",
    GEOVERSA2_LEADER_ENSEMBLE_MEMBERS = as.character(leader_members),
    GEOVERSA_AUTO_PATCH_SSL = "0"
  )
  if (use_ssl) {
    if (!nzchar(ssl_patch_state)) stop("GEOVERSA_ABLATION_USE_SSL=1 requires GEOVERSA_ABLATION_SSL_PATCH_STATE or GEOVERSA_SSL_PATCH_STATE.")
    env <- c(
      env,
      GEOVERSA_SSL_PATCH_STATE = ssl_patch_state,
      GEOVERSA_SSL_PATCH_WARMUP_LR_MULT = "0.1",
      GEOVERSA_PATCH_GLOBAL_SIZE = as.character(fixed_patch_global),
      GEOVERSA_PATCH_LOCAL_SIZE = as.character(fixed_patch_local)
    )
  }
  env
}

fixed_env_from_config <- function(cfg) {
  env <- c(
    WADOUX_WARMUP_EPOCHS = as.character(cfg$warmup_epochs),
    WADOUX_BASE_LOSS_WEIGHT = format(cfg$base_loss_weight, scientific = FALSE, trim = TRUE),
    WADOUX_ALPHA_ME = format(cfg$alpha_me, scientific = FALSE, trim = TRUE),
    WADOUX_LAMBDA_COV = format(cfg$lambda_cov, scientific = FALSE, trim = TRUE),
    WADOUX_LR = format(cfg$lr, scientific = TRUE, digits = 6),
    WADOUX_MIN_LR = format(cfg$min_lr, scientific = TRUE, digits = 6),
    WADOUX_WEIGHT_DECAY = format(cfg$wd, scientific = TRUE, digits = 6),
    WADOUX_BATCH_SIZE = as.character(cfg$batch_size),
    WADOUX_BANK_REFRESH_EVERY = as.character(cfg$bank_refresh_every),
    WADOUX_PATIENCE = as.character(cfg$patience),
    WADOUX_LR_PATIENCE = as.character(cfg$lr_patience),
    WADOUX_LR_DECAY = format(cfg$lr_decay, scientific = FALSE, trim = TRUE),
    WADOUX_K_NEIGHBORS = as.character(cfg$K_neighbors),
    WADOUX_BETA_INIT = format(cfg$beta_init, scientific = FALSE, trim = TRUE),
    WADOUX_PATCH_DIM = as.character(cfg$patch_dim),
    WADOUX_D = as.character(cfg$d),
    WADOUX_TAB_DROPOUT = format(cfg$tab_dropout, scientific = FALSE, trim = TRUE),
    WADOUX_PATCH_DROPOUT = format(cfg$patch_dropout, scientific = FALSE, trim = TRUE),
    WADOUX_COORD_DIM = as.character(cfg$coord_dim),
    WADOUX_COORD_DROPOUT = format(cfg$coord_dropout, scientific = FALSE, trim = TRUE)
  )
  if (!use_ssl) {
    env <- c(
      env,
      GEOVERSA_PATCH_GLOBAL_SIZE = as.character(cfg$patch_size),
      GEOVERSA_PATCH_LOCAL_SIZE = as.character(cfg$patch_local_size)
    )
  }
  env
}

sample_random_config <- function(idx, base_cfg, seed_base = train_seed) {
  set.seed(seed_base + idx * 1009L)
  if (use_ssl) {
    patch_size <- fixed_patch_global
    local_choices <- fixed_patch_local
  } else {
    patch_size_choices <- seq(15L, 31L, by = 2L)
    patch_size <- sample(patch_size_choices, size = 1L)
    local_max <- max(9L, patch_size - 2L)
    local_choices <- seq(9L, local_max, by = 2L)
    local_choices <- local_choices[local_choices <= patch_size]
  }
  list(
    warmup_epochs = sample(4L:12L, size = 1L),
    base_loss_weight = round(runif(1L, 0.05, 0.15), 4),
    alpha_me = round(runif(1L, 0.00, 0.20), 4),
    lambda_cov = round(runif(1L, 0.00, 0.04), 5),
    lr = 10^(runif(1L, log10(5e-4), log10(2e-3))),
    min_lr = 10^(runif(1L, log10(5e-7), log10(5e-6))),
    wd = 10^(runif(1L, log10(8e-4), log10(6e-3))),
    batch_size = sample(c(24L, 32L, 40L, 50L, 64L), size = 1L),
    bank_refresh_every = sample(1L:3L, size = 1L),
    patience = sample(6L:14L, size = 1L),
    lr_patience = sample(2L:5L, size = 1L),
    lr_decay = round(runif(1L, 0.35, 0.65), 3),
    K_neighbors = sample(8L:36L, size = 1L),
    beta_init = round(runif(1L, -0.5, 1.5), 4),
    patch_dim = sample(unique(as.integer(c(96L, 117L, 128L, 143L, 160L, 192L))), size = 1L),
    d = sample(c(128L, 192L, 256L), size = 1L),
    tab_dropout = round(runif(1L, 0.10, 0.30), 3),
    patch_dropout = round(runif(1L, 0.05, 0.20), 3),
    coord_dim = sample(c(32L, 40L, 45L, 46L, 48L), size = 1L),
    coord_dropout = round(runif(1L, 0.05, 0.12), 3),
    patch_size = patch_size,
    patch_local_size = if (length(local_choices) > 1L) sample(local_choices, size = 1L) else as.integer(local_choices[1])
  )
}

regimes <- c("variogram_auto", "fixed_config", sprintf("random_config_%02d", seq_len(random_variants)))

build_regime_env <- function(regime) {
  if (identical(regime, "variogram_auto")) {
    return(character())
  }
  if (identical(regime, "fixed_config")) {
    return(fixed_env_from_config(fixed_config))
  }
  if (startsWith(regime, "random_config_")) {
    idx <- as.integer(sub("random_config_", "", regime))
    cfg <- sample_random_config(idx, fixed_config)
    return(fixed_env_from_config(cfg))
  }
  stop("Unsupported regime: ", regime)
}

run_regime <- function(regime) {
  run_dir <- file.path(results_root, regime)
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  env_values <- c(make_common_env(run_dir), build_regime_env(regime))
  env_manifest <- data.frame(
    key = names(env_values),
    value = unname(env_values),
    stringsAsFactors = FALSE
  )
  write.csv(env_manifest, file.path(run_dir, "geoversa_variogram_ablation_env_manifest.csv"), row.names = FALSE)

  cat("\n==================================================\n")
  cat(sprintf("GeoVersa variogram ablation | %s\n", regime))
  cat("==================================================\n")

  status <- with_env_overrides(env_values, {
    system2("Rscript", c("code/run_wadoux_style_rf_conv_comparison.R"))
  })
  if (!identical(status, 0L)) {
    stop("Benchmark failed for regime ", regime, " with exit status ", status)
  }

  summary_path <- file.path(run_dir, "wadoux_style_rf_conv_summary_by_protocol.csv")
  all_results_path <- file.path(run_dir, "wadoux_style_rf_conv_all_results.csv")
  if (!file.exists(summary_path) || !file.exists(all_results_path)) {
    stop("Missing benchmark outputs for regime ", regime)
  }

  summary_df <- read.csv(summary_path, stringsAsFactors = FALSE) %>%
    mutate(regime = regime, .before = 1)
  all_results_df <- read.csv(all_results_path, stringsAsFactors = FALSE) %>%
    mutate(regime = regime, .before = 1)
  list(summary = summary_df, all_results = all_results_df)
}

run_outputs <- lapply(regimes, run_regime)
summary_all <- bind_rows(lapply(run_outputs, `[[`, "summary"))
all_results_all <- bind_rows(lapply(run_outputs, `[[`, "all_results"))

rf_model_label <- if (rf_refit_full_calibration) "RF_500" else "RF_400"
protocol_focus <- strsplit(protocols, ",", fixed = TRUE)[[1L]][1L]
protocol_focus <- trimws(protocol_focus)

extract_regime_table <- function(regime_name) {
  sum_reg <- summary_all %>% filter(regime == regime_name, protocol == protocol_focus)
  geo_row <- sum_reg %>% filter(model == target_model)
  rf_row <- sum_reg %>% filter(model == rf_model_label)
  if (nrow(geo_row) < 1L || nrow(rf_row) < 1L) return(NULL)
  data.frame(
    regime = regime_name,
    protocol = protocol_focus,
    geoversa_model = target_model,
    rf_model = rf_model_label,
    geoversa_rmse_mean = as.numeric(geo_row$RMSE_mean[1]),
    rf_rmse_mean = as.numeric(rf_row$RMSE_mean[1]),
    delta_vs_rf = as.numeric(geo_row$RMSE_mean[1]) - as.numeric(rf_row$RMSE_mean[1]),
    geoversa_me_mean = as.numeric(geo_row$ME_mean[1]),
    rf_me_mean = as.numeric(rf_row$ME_mean[1]),
    geoversa_rmse_sd = as.numeric(geo_row$RMSE_sd[1]),
    rf_rmse_sd = as.numeric(rf_row$RMSE_sd[1]),
    stringsAsFactors = FALSE
  )
}

ablation_rows <- bind_rows(lapply(regimes, extract_regime_table))

random_rows <- ablation_rows %>% filter(startsWith(regime, "random_config_"))
if (nrow(random_rows) > 0) {
  random_agg <- random_rows %>%
    summarise(
      regime = "random_config",
      protocol = first(protocol),
      geoversa_model = first(geoversa_model),
      rf_model = first(rf_model),
      geoversa_rmse_mean = mean(geoversa_rmse_mean, na.rm = TRUE),
      rf_rmse_mean = mean(rf_rmse_mean, na.rm = TRUE),
      delta_vs_rf = mean(delta_vs_rf, na.rm = TRUE),
      geoversa_me_mean = mean(geoversa_me_mean, na.rm = TRUE),
      rf_me_mean = mean(rf_me_mean, na.rm = TRUE),
      geoversa_rmse_sd = mean(geoversa_rmse_sd, na.rm = TRUE),
      rf_rmse_sd = mean(rf_rmse_sd, na.rm = TRUE),
      n_random_variants = n(),
      random_best_rmse = min(geoversa_rmse_mean, na.rm = TRUE),
      random_worst_rmse = max(geoversa_rmse_mean, na.rm = TRUE),
      .groups = "drop"
    )
} else {
  random_agg <- data.frame()
}

table_core <- ablation_rows %>%
  filter(regime %in% c("variogram_auto", "fixed_config")) %>%
  mutate(n_random_variants = NA_integer_, random_best_rmse = NA_real_, random_worst_rmse = NA_real_)
final_table <- bind_rows(table_core, random_agg) %>%
  mutate(
    regime = factor(regime, levels = c("variogram_auto", "fixed_config", "random_config"))
  ) %>%
  arrange(regime)

write.csv(summary_all, file.path(results_root, "geoversa_variogram_ablation_all_summaries.csv"), row.names = FALSE)
write.csv(all_results_all, file.path(results_root, "geoversa_variogram_ablation_all_results.csv"), row.names = FALSE)
write.csv(ablation_rows, file.path(results_root, "geoversa_variogram_ablation_variant_table.csv"), row.names = FALSE)
write.csv(final_table, file.path(results_root, "geoversa_variogram_ablation_table.csv"), row.names = FALSE)

cat("\n=== GeoVersa variogram ablation complete ===\n")
if (nrow(final_table) > 0) {
  print(final_table)
}
