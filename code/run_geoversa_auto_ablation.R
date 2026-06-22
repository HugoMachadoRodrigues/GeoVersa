rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
})

project_root <- normalizePath(getwd(), mustWork = TRUE)
benchmark_script <- file.path(project_root, "code", "run_wadoux_style_rf_conv_comparison.R")
ssl_pretrain_script <- file.path(project_root, "code", "geoversa_ssl_pretrain.R")

results_root <- Sys.getenv(
  "GEOVERSA_AUTO_ABLATION_RESULTS_DIR",
  unset = file.path(project_root, "results", sprintf("geoversa_auto_ablation_%s", format(Sys.Date(), "%Y%m%d")))
)
dir.create(results_root, recursive = TRUE, showWarnings = FALSE)

scenario <- Sys.getenv("GEOVERSA_AUTO_ABLATION_SCENARIO", unset = "random")
protocols <- Sys.getenv("GEOVERSA_AUTO_ABLATION_PROTOCOLS", unset = "DesignBased")
sample_size <- as.integer(Sys.getenv("GEOVERSA_AUTO_ABLATION_SAMPLE_SIZE", unset = "500"))
n_iter <- as.integer(Sys.getenv("GEOVERSA_AUTO_ABLATION_N_ITER", unset = "3"))
device <- Sys.getenv("GEOVERSA_AUTO_ABLATION_DEVICE", unset = Sys.getenv("WADOUX_DEVICE", unset = "cpu"))
model_profile <- Sys.getenv("GEOVERSA_AUTO_ABLATION_PROFILE", unset = "auto")
train_seed <- as.integer(Sys.getenv("GEOVERSA_AUTO_ABLATION_TRAIN_SEED", unset = "123"))
rf_refit_full_calibration <- tolower(Sys.getenv("GEOVERSA_AUTO_ABLATION_RF_REFIT_FULL_CALIBRATION", unset = "1")) %in% c("1", "true", "yes")
refit_full_calibration <- tolower(Sys.getenv("GEOVERSA_AUTO_ABLATION_REFIT_FULL_CALIBRATION", unset = "0")) %in% c("1", "true", "yes")
use_leader_ensemble <- tolower(Sys.getenv("GEOVERSA_AUTO_ABLATION_USE_LEADER_ENSEMBLE", unset = "0")) %in% c("1", "true", "yes")
leader_members <- as.integer(Sys.getenv("GEOVERSA_AUTO_ABLATION_LEADER_MEMBERS", unset = "3"))
use_ssl <- tolower(Sys.getenv("GEOVERSA_AUTO_ABLATION_USE_SSL", unset = "1")) %in% c("1", "true", "yes")
target_model <- if (use_leader_ensemble) "GeoVersa2-LeaderEnsemble" else "ConvKrigingNet2D"

parse_int_env <- function(name, default = NA_integer_) {
  raw <- trimws(Sys.getenv(name, unset = ""))
  if (!nzchar(raw)) return(default)
  val <- suppressWarnings(as.integer(round(as.numeric(raw))))
  if (!is.finite(val)) default else val
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

resolve_patch_geometry <- function() {
  fixed_global <- parse_int_env("GEOVERSA_AUTO_ABLATION_PATCH_GLOBAL", default = NA_integer_)
  fixed_local <- parse_int_env("GEOVERSA_AUTO_ABLATION_PATCH_LOCAL", default = NA_integer_)
  if (is.finite(fixed_global) && fixed_global > 0L) {
    if (!is.finite(fixed_local) || fixed_local <= 0L) {
      fixed_local <- max(9L, min(fixed_global, as.integer(round(0.63 * fixed_global))))
      if (fixed_local %% 2L == 0L) fixed_local <- fixed_local - 1L
    }
    return(list(global = fixed_global, local = fixed_local, source = "env"))
  }

  canonical_summary <- file.path(project_root, "results", "ssl_pretrain", "geoversa_ssl_pretrain_summary.csv")
  if (file.exists(canonical_summary)) {
    meta <- read.csv(canonical_summary, stringsAsFactors = FALSE)
    if (nrow(meta) > 0L) {
      g <- suppressWarnings(as.integer(meta$ssl_patch_global_size[1]))
      l <- suppressWarnings(as.integer(meta$ssl_patch_local_size[1]))
      if (is.finite(g) && g > 0L && is.finite(l) && l > 0L) {
        return(list(global = g, local = l, source = "canonical_ssl"))
      }
    }
  }

  list(global = 27L, local = 17L, source = "default_27_17")
}

patch_geom <- resolve_patch_geometry()

prepare_ssl_artifact <- function(kind) {
  stopifnot(kind %in% c("fullauto", "fixedmask"))
  artifact_dir <- file.path(results_root, "ssl_artifacts", kind)
  state_path <- file.path(artifact_dir, "ssl_patch_encoder_state.pt")
  summary_path <- file.path(artifact_dir, "geoversa_ssl_pretrain_summary.csv")
  if (file.exists(state_path) && file.exists(summary_path)) {
    return(list(path = state_path, summary = summary_path, dir = artifact_dir, reused = TRUE))
  }
  dir.create(artifact_dir, recursive = TRUE, showWarnings = FALSE)
  ssl_env <- c(
    GEOVERSA_SSL_RESULTS_DIR = artifact_dir,
    GEOVERSA_SSL_DEVICE = device,
    WADOUX_SAMPLE_SIZE = as.character(sample_size),
    GEOVERSA_PATCH_GLOBAL_SIZE = as.character(patch_geom$global),
    GEOVERSA_PATCH_LOCAL_SIZE = as.character(patch_geom$local),
    GEOVERSA_SSL_SEED = as.character(train_seed + if (identical(kind, "fullauto")) 9100L else 9200L)
  )
  if (identical(kind, "fixedmask")) {
    ssl_env <- c(ssl_env, GEOVERSA_SSL_MASK_RATIO = "0.50")
  }
  status <- with_env_overrides(ssl_env, {
    system2("Rscript", c("code/geoversa_ssl_pretrain.R"))
  })
  if (!identical(status, 0L)) {
    stop("SSL pretrain failed for artifact kind ", kind, " with exit status ", status)
  }
  if (!file.exists(state_path) || !file.exists(summary_path)) {
    stop("Missing SSL artifact outputs for kind ", kind)
  }
  list(path = state_path, summary = summary_path, dir = artifact_dir, reused = FALSE)
}

make_common_env <- function(run_dir, ssl_artifact = NULL) {
  env <- c(
    WADOUX_AUTO_V5_SCRIPT = file.path(project_root, "code", "ConvKrigingNet2D_Auto_v5.R"),
    WADOUX_AUTO_SCRIPT = file.path(project_root, "code", "ConvKrigingNet2D_Auto.R"),
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
  if (!is.null(ssl_artifact)) {
    env <- c(
      env,
      GEOVERSA_SSL_PATCH_STATE = ssl_artifact$path,
      GEOVERSA_SSL_PATCH_WARMUP_LR_MULT = "0.10",
      GEOVERSA_PATCH_GLOBAL_SIZE = as.character(patch_geom$global),
      GEOVERSA_PATCH_LOCAL_SIZE = as.character(patch_geom$local)
    )
  }
  env
}

regime_definitions <- list(
  FullAuto = list(
    ssl_kind = if (use_ssl) "fullauto" else NULL,
    env = c()
  ),
  `Baccini-tuned` = list(
    ssl_kind = if (use_ssl) "fixedmask" else NULL,
    env = c(
      WADOUX_MAIN_EPOCHS = "80",
      WADOUX_COORD_DIM = "32",
      WADOUX_COORD_DROPOUT = "0.05",
      GEOVERSA_AUTO_NKO_DROPOUT_OVERRIDE = "0.05",
      WADOUX_PATIENCE = "15",
      WADOUX_LR_PATIENCE = "4",
      WADOUX_LR = "1e-4",
      WADOUX_WEIGHT_DECAY = "1e-3"
    )
  ),
  No_coord_dim_derivation = list(
    ssl_kind = if (use_ssl) "fullauto" else NULL,
    env = c(WADOUX_COORD_DIM = "32")
  ),
  No_dropout_derivation = list(
    ssl_kind = if (use_ssl) "fullauto" else NULL,
    env = c(
      WADOUX_TAB_DROPOUT = "0.25",
      WADOUX_PATCH_DROPOUT = "0.16",
      WADOUX_COORD_DROPOUT = "0.05",
      GEOVERSA_AUTO_NKO_DROPOUT_OVERRIDE = "0.05"
    )
  ),
  No_ssl_mask_derivation = list(
    ssl_kind = if (use_ssl) "fixedmask" else NULL,
    env = c()
  ),
  No_beta_init = list(
    ssl_kind = if (use_ssl) "fullauto" else NULL,
    env = c(WADOUX_BETA_INIT = "0.0")
  ),
  FullAuto_fixed_lr = list(
    ssl_kind = if (use_ssl) "fullauto" else NULL,
    env = c(WADOUX_LR = "1e-4")
  ),
  FullAuto_ceil_2e4 = list(
    ssl_kind = if (use_ssl) "fullauto" else NULL,
    env = c(WADOUX_LR_CEIL = "2e-4")
  ),
  FullAuto_ceil_1e3 = list(
    ssl_kind = if (use_ssl) "fullauto" else NULL,
    env = c(WADOUX_LR_CEIL = "1e-3")
  )
)

if (!use_ssl) {
  regime_definitions$No_ssl_mask_derivation$env <- c(regime_definitions$No_ssl_mask_derivation$env, GEOVERSA_AUTO_ABLATION_NOTE = "ssl_disabled")
}

requested_configs_raw <- trimws(Sys.getenv("GEOVERSA_AUTO_ABLATION_CONFIGS", unset = ""))
if (nzchar(requested_configs_raw)) {
  requested_configs <- trimws(strsplit(requested_configs_raw, ",", fixed = TRUE)[[1L]])
  requested_configs <- requested_configs[nzchar(requested_configs)]
  unknown_configs <- setdiff(requested_configs, names(regime_definitions))
  if (length(unknown_configs) > 0L) {
    stop("Unknown GEOVERSA_AUTO_ABLATION_CONFIGS entries: ", paste(unknown_configs, collapse = ", "))
  }
  regime_definitions <- regime_definitions[requested_configs]
}

run_regime <- function(regime_name, regime_spec) {
  run_dir <- file.path(results_root, regime_name)
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
  ssl_artifact <- if (!is.null(regime_spec$ssl_kind)) prepare_ssl_artifact(regime_spec$ssl_kind) else NULL
  env_values <- c(make_common_env(run_dir, ssl_artifact = ssl_artifact), regime_spec$env)
  env_manifest <- data.frame(
    key = names(env_values),
    value = unname(env_values),
    stringsAsFactors = FALSE
  )
  if (!is.null(ssl_artifact)) {
    env_manifest <- bind_rows(
      env_manifest,
      data.frame(
        key = c("ssl_artifact_dir", "ssl_artifact_kind", "ssl_artifact_reused"),
        value = c(ssl_artifact$dir, regime_spec$ssl_kind, as.character(ssl_artifact$reused)),
        stringsAsFactors = FALSE
      )
    )
  }
  write.csv(env_manifest, file.path(run_dir, "geoversa_auto_ablation_env_manifest.csv"), row.names = FALSE)

  cat("\n==================================================\n")
  cat(sprintf("GeoVersa auto ablation | %s\n", regime_name))
  cat("==================================================\n")

  status <- with_env_overrides(env_values, {
    system2("Rscript", c("code/run_wadoux_style_rf_conv_comparison.R"))
  })
  if (!identical(status, 0L)) {
    stop("Benchmark failed for regime ", regime_name, " with exit status ", status)
  }

  summary_path <- file.path(run_dir, "wadoux_style_rf_conv_summary_by_protocol.csv")
  uq_summary_path <- file.path(run_dir, "wadoux_style_rf_conv_uq_summary_by_protocol.csv")
  if (!file.exists(summary_path) || !file.exists(uq_summary_path)) {
    stop("Missing benchmark outputs for regime ", regime_name)
  }

  list(
    summary = read.csv(summary_path, stringsAsFactors = FALSE) %>% mutate(config = regime_name, .before = 1),
    uq_summary = read.csv(uq_summary_path, stringsAsFactors = FALSE) %>% mutate(config = regime_name, .before = 1)
  )
}

run_outputs <- Map(run_regime, names(regime_definitions), regime_definitions)
summary_all <- bind_rows(lapply(run_outputs, `[[`, "summary"))
uq_summary_all <- bind_rows(lapply(run_outputs, `[[`, "uq_summary"))

rf_model_label <- if (rf_refit_full_calibration) "RF_500" else "RF_400"
protocol_focus <- trimws(strsplit(protocols, ",", fixed = TRUE)[[1L]][1L])

extract_row <- function(config_name) {
  sum_reg <- summary_all %>% filter(config == config_name, protocol == protocol_focus)
  uq_reg <- uq_summary_all %>% filter(config == config_name, protocol == protocol_focus)
  geo_row <- sum_reg %>% filter(model == target_model)
  rf_row <- sum_reg %>% filter(model == rf_model_label)
  uq_row <- uq_reg %>% filter(model == target_model)
  if (nrow(geo_row) < 1L || nrow(rf_row) < 1L) return(NULL)
  data.frame(
    config = config_name,
    protocol = protocol_focus,
    geoversa_model = target_model,
    rf_model = rf_model_label,
    rmse_mean = as.numeric(geo_row$RMSE_mean[1]),
    rmse_sd = as.numeric(geo_row$RMSE_sd[1]),
    rf_rmse_mean = as.numeric(rf_row$RMSE_mean[1]),
    delta_vs_rf = as.numeric(geo_row$RMSE_mean[1]) - as.numeric(rf_row$RMSE_mean[1]),
    picp_90 = if (nrow(uq_row) > 0L) as.numeric(uq_row$picp_90_mean[1]) else NA_real_,
    ece = if (nrow(uq_row) > 0L) as.numeric(uq_row$reliability_ece_mean[1]) else NA_real_,
    uq_nll = if (nrow(uq_row) > 0L) as.numeric(uq_row$uq_nll_mean[1]) else NA_real_,
    uq_available_rate = if (nrow(uq_row) > 0L) as.numeric(uq_row$uq_available_rate[1]) else NA_real_,
    stringsAsFactors = FALSE
  )
}

final_table <- bind_rows(lapply(names(regime_definitions), extract_row))

write.csv(summary_all, file.path(results_root, "geoversa_auto_ablation_all_summaries.csv"), row.names = FALSE)
write.csv(uq_summary_all, file.path(results_root, "geoversa_auto_ablation_all_uq_summaries.csv"), row.names = FALSE)
write.csv(final_table, file.path(results_root, "geoversa_auto_ablation_table.csv"), row.names = FALSE)

cat("\n=== GeoVersa automation ablation complete ===\n")
if (nrow(final_table) > 0) {
  print(final_table)
}
