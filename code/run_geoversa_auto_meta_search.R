rm(list = ls())

library(dplyr)

project_root <- normalizePath(getwd(), mustWork = TRUE)
benchmark_script <- file.path(project_root, "code", "run_wadoux_style_rf_conv_comparison.R")
auto_script <- normalizePath(file.path(project_root, "code", "ConvKrigingNet2D_Auto.R"), mustWork = FALSE)
auto_v5_script <- normalizePath(file.path(project_root, "code", "ConvKrigingNet2D_Auto_v5.R"), mustWork = FALSE)

screen_scenario <- Sys.getenv("GEOVERSA_META_SCENARIO", unset = "random")
screen_protocols <- Sys.getenv("GEOVERSA_META_PROTOCOLS", unset = "DesignBased")
screen_sample_size <- as.integer(Sys.getenv("GEOVERSA_META_SAMPLE_SIZE", unset = "500"))
screen_n_iter <- as.integer(Sys.getenv("GEOVERSA_META_N_ITER", unset = "3"))
screen_device <- Sys.getenv("GEOVERSA_META_DEVICE", unset = Sys.getenv("WADOUX_DEVICE", unset = "cpu"))
screen_profile <- Sys.getenv("GEOVERSA_META_PROFILE", unset = "auto")
screen_focus <- tolower(Sys.getenv("GEOVERSA_META_FOCUS", unset = "best_non_oof"))
screen_use_oof <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_META_USE_OOF", unset = if (identical(screen_focus, "best_non_oof")) "0" else "1")))
screen_use_oof <- is.finite(screen_use_oof) && screen_use_oof > 0
meta_seed <- as.integer(Sys.getenv("GEOVERSA_META_SEED", unset = "20260408"))
candidate_count <- as.integer(Sys.getenv("GEOVERSA_META_CANDIDATES", unset = "6"))
candidate_count <- max(2L, candidate_count)
seed_csv_path <- Sys.getenv("GEOVERSA_META_SEED_CSV", unset = "")
seed_topk <- as.integer(Sys.getenv("GEOVERSA_META_SEED_TOPK", unset = "0"))
if (!is.finite(seed_topk)) seed_topk <- 0L
seed_topk <- max(0L, seed_topk)
jitter_per_seed <- as.integer(Sys.getenv("GEOVERSA_META_JITTER_PER_SEED", unset = "1"))
if (!is.finite(jitter_per_seed)) jitter_per_seed <- 1L
jitter_per_seed <- max(0L, jitter_per_seed)
results_root <- Sys.getenv(
  "GEOVERSA_META_RESULTS_DIR",
  unset = file.path(project_root, "results", sprintf("geoversa_auto_meta_search_%s", format(Sys.Date(), "%Y%m%d")))
)
dir.create(results_root, recursive = TRUE, showWarnings = FALSE)

rmse_sd_weight <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_META_RMSE_SD_WEIGHT", unset = "0.20")))
if (!is.finite(rmse_sd_weight)) rmse_sd_weight <- 0.20
gain_penalty_weight <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_META_GAIN_PENALTY", unset = "2.00")))
if (!is.finite(gain_penalty_weight)) gain_penalty_weight <- 2.00

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

leader_backbone_env <- function(use_oof = FALSE) {
  c(
    GEOVERSA_AUTO_PATCH_PYRAMID = "0",
    GEOVERSA_AUTO_PATCH_CONTEXT = "0",
    GEOVERSA_AUTO_TAB_CROSSNET = "1",
    GEOVERSA_AUTO_TAB_QBASIS = "1",
    GEOVERSA_AUTO_TAB_QBASIS_KNOTS = "4",
    GEOVERSA_AUTO_TAB_TREE = "0",
    GEOVERSA_AUTO_TAB_RS = "0",
    GEOVERSA_AUTO_CROSS_TAB_PATCH = "0",
    GEOVERSA_AUTO_PEDOMOE = "0",
    GEOVERSA_AUTO_NKO = "0",
    GEOVERSA_AUTO_DAM = "0",
    GEOVERSA_AUTO_VRM = "0",
    GEOVERSA_AUTO_CVRM = "0",
    GEOVERSA_AUTO_OOF_BANK = if (use_oof) "1" else "0",
    GEOVERSA_AUTO_TRUST_FROM_OOF = if (use_oof) "1" else "0",
    GEOVERSA_AUTO_BANK_MODE = if (use_oof) "mix" else "current"
  )
}

make_candidate <- function(name,
                           blw_floor = 0.05,
                           blw_slope = 0.10,
                           alpha_me_mult = 1.0,
                           lambda_cov_mult = 1.0,
                           beta_intercept = 2.0,
                           beta_slope = -6.0,
                           k_mult = 1.0,
                           patch_dropout_mult = 1.0,
                           wd_mult = 1.0,
                           bank_refresh_mult = 1.0,
                           trust_floor = 0.10,
                           trust_span = 0.90,
                           bank_current_floor = 0.25,
                           bank_current_ceiling = 0.85,
                           bank_current_power = 0.50,
                           use_oof = screen_use_oof) {
  list(
    name = name,
    blw_floor = blw_floor,
    blw_slope = blw_slope,
    alpha_me_mult = alpha_me_mult,
    lambda_cov_mult = lambda_cov_mult,
    beta_intercept = beta_intercept,
    beta_slope = beta_slope,
    k_mult = k_mult,
    patch_dropout_mult = patch_dropout_mult,
    wd_mult = wd_mult,
    bank_refresh_mult = bank_refresh_mult,
    trust_floor = trust_floor,
    trust_span = trust_span,
    bank_current_floor = bank_current_floor,
    bank_current_ceiling = bank_current_ceiling,
    bank_current_power = bank_current_power,
    use_oof = use_oof
  )
}

sample_candidate <- function(i) {
  if (identical(screen_focus, "best_non_oof")) {
    return(make_candidate(
      name = sprintf("candidate_%02d", i),
      blw_floor = runif(1, 0.048, 0.055),
      blw_slope = runif(1, 0.095, 0.112),
      alpha_me_mult = runif(1, 0.88, 1.08),
      lambda_cov_mult = runif(1, 0.75, 1.10),
      beta_intercept = runif(1, 1.85, 2.15),
      beta_slope = runif(1, -6.20, -5.60),
      k_mult = runif(1, 0.92, 1.05),
      patch_dropout_mult = runif(1, 0.92, 1.05),
      wd_mult = runif(1, 0.85, 1.10),
      bank_refresh_mult = runif(1, 1.0, 1.75),
      trust_floor = 0.18,
      trust_span = 0.72,
      bank_current_floor = 0.25,
      bank_current_ceiling = 0.85,
      bank_current_power = 0.50,
      use_oof = FALSE
    ))
  }
  make_candidate(
    name = sprintf("candidate_%02d", i),
    blw_floor = runif(1, 0.045, 0.060),
    blw_slope = runif(1, 0.085, 0.120),
    alpha_me_mult = runif(1, 0.80, 1.20),
    lambda_cov_mult = runif(1, 0.60, 1.40),
    beta_intercept = runif(1, 1.6, 2.4),
    beta_slope = runif(1, -6.8, -5.2),
    k_mult = runif(1, 0.85, 1.10),
    patch_dropout_mult = runif(1, 0.90, 1.10),
    wd_mult = runif(1, 0.75, 1.25),
    bank_refresh_mult = runif(1, 0.80, 2.00),
    trust_floor = runif(1, 0.05, 0.20),
    trust_span = runif(1, 0.70, 1.00),
    bank_current_floor = runif(1, 0.20, 0.45),
    bank_current_ceiling = runif(1, 0.75, 0.90),
    bank_current_power = runif(1, 0.35, 0.80),
    use_oof = screen_use_oof
  )
}

coerce_seed_candidate <- function(row, idx = 1L) {
  get_num <- function(nm, default) {
    val <- suppressWarnings(as.numeric(row[[nm]][1]))
    if (!is.finite(val)) default else val
  }
  base_name <- if ("variant" %in% names(row)) {
    as.character(row$variant[1])
  } else if ("name" %in% names(row)) {
    as.character(row$name[1])
  } else {
    sprintf("seed_%02d", idx)
  }
  make_candidate(
    name = sprintf("seed_%02d_%s", idx, gsub("[^A-Za-z0-9_]+", "_", base_name)),
    blw_floor = get_num("auto_blw_floor", 0.05),
    blw_slope = get_num("auto_blw_slope", 0.10),
    alpha_me_mult = get_num("auto_alpha_me_mult", 1.0),
    lambda_cov_mult = get_num("auto_lambda_cov_mult", 1.0),
    beta_intercept = get_num("auto_beta_intercept", 2.0),
    beta_slope = get_num("auto_beta_slope", -6.0),
    k_mult = get_num("auto_k_mult", 1.0),
    patch_dropout_mult = get_num("auto_patch_dropout_mult", 1.0),
    wd_mult = get_num("auto_wd_mult", 1.0),
    bank_refresh_mult = get_num("auto_bank_refresh_mult", 1.0),
    trust_floor = get_num("auto_trust_floor", 0.18),
    trust_span = get_num("auto_trust_span", 0.72),
    bank_current_floor = get_num("auto_bank_current_floor", 0.25),
    bank_current_ceiling = get_num("auto_bank_current_ceiling", 0.85),
    bank_current_power = get_num("auto_bank_current_power", 0.50),
    use_oof = identical(toupper(as.character(row$auto_use_oof[1])), "TRUE")
  )
}

jitter_seed_candidate <- function(seed_cand, idx = 1L) {
  jitter_mul <- function(x, lo, hi, frac) {
    . <- x * stats::runif(1, 1 - frac, 1 + frac)
    max(lo, min(hi, .))
  }
  jitter_add <- function(x, lo, hi, delta) {
    . <- x + stats::runif(1, -delta, delta)
    max(lo, min(hi, .))
  }
  make_candidate(
    name = sprintf("%s_j%02d", seed_cand$name, idx),
    blw_floor = jitter_add(seed_cand$blw_floor, 0.045, 0.060, 0.004),
    blw_slope = jitter_add(seed_cand$blw_slope, 0.085, 0.120, 0.010),
    alpha_me_mult = jitter_mul(seed_cand$alpha_me_mult, 0.80, 1.20, 0.10),
    lambda_cov_mult = jitter_mul(seed_cand$lambda_cov_mult, 0.60, 1.40, 0.15),
    beta_intercept = jitter_add(seed_cand$beta_intercept, 1.6, 2.4, 0.20),
    beta_slope = jitter_add(seed_cand$beta_slope, -6.8, -5.2, 0.25),
    k_mult = jitter_mul(seed_cand$k_mult, 0.85, 1.10, 0.08),
    patch_dropout_mult = jitter_mul(seed_cand$patch_dropout_mult, 0.90, 1.10, 0.06),
    wd_mult = jitter_mul(seed_cand$wd_mult, 0.75, 1.25, 0.12),
    bank_refresh_mult = jitter_mul(seed_cand$bank_refresh_mult, 0.80, 2.00, 0.20),
    trust_floor = jitter_add(seed_cand$trust_floor, 0.05, 0.20, 0.02),
    trust_span = jitter_add(seed_cand$trust_span, 0.70, 1.00, 0.05),
    bank_current_floor = jitter_add(seed_cand$bank_current_floor, 0.20, 0.45, 0.05),
    bank_current_ceiling = jitter_add(seed_cand$bank_current_ceiling, 0.75, 0.90, 0.04),
    bank_current_power = jitter_add(seed_cand$bank_current_power, 0.35, 0.80, 0.10),
    use_oof = seed_cand$use_oof
  )
}

set.seed(meta_seed)
if (nzchar(seed_csv_path) && file.exists(seed_csv_path) && seed_topk > 0L) {
  seed_df <- read.csv(seed_csv_path, stringsAsFactors = FALSE)
  if ("objective" %in% names(seed_df)) {
    seed_df <- seed_df[order(seed_df$objective, seed_df$RMSE_mean), , drop = FALSE]
  } else if ("RMSE_mean" %in% names(seed_df)) {
    seed_df <- seed_df[order(seed_df$RMSE_mean), , drop = FALSE]
  }
  seed_df <- head(seed_df, seed_topk)
  seed_candidates <- lapply(seq_len(nrow(seed_df)), function(i) {
    coerce_seed_candidate(seed_df[i, , drop = FALSE], idx = i)
  })
  jittered <- unlist(lapply(seq_along(seed_candidates), function(i) {
    if (jitter_per_seed < 1L) return(list())
    lapply(seq_len(jitter_per_seed), function(j) jitter_seed_candidate(seed_candidates[[i]], idx = j))
  }), recursive = FALSE)
  candidates <- c(seed_candidates, jittered)
} else {
  candidates <- c(
    list(make_candidate(name = "baseline")),
    lapply(seq_len(candidate_count - 1L), sample_candidate)
  )
}

manifest <- bind_rows(lapply(candidates, as.data.frame))
write.csv(manifest, file.path(results_root, "geoversa_auto_meta_manifest.csv"), row.names = FALSE)

score_rows <- vector("list", length(candidates))
diag_rows <- vector("list", length(candidates))

for (i in seq_along(candidates)) {
  cand <- candidates[[i]]
  variant_dir <- file.path(results_root, cand$name)
  dir.create(variant_dir, recursive = TRUE, showWarnings = FALSE)

  env_values <- c(
    WADOUX_AUTO_V5_SCRIPT = auto_v5_script,
    WADOUX_AUTO_SCRIPT = auto_script,
    WADOUX_MODELS = "ConvKrigingNet2D",
    WADOUX_PROTOCOLS = screen_protocols,
    WADOUX_SCENARIO = screen_scenario,
    WADOUX_N_ITER = as.character(screen_n_iter),
    WADOUX_SAMPLE_SIZE = as.character(screen_sample_size),
    WADOUX_MODEL_PROFILE = screen_profile,
    WADOUX_DEVICE = screen_device,
    WADOUX_RESULTS_DIR = variant_dir,
    WADOUX_REFIT_FULL_CALIBRATION = "FALSE",
    WADOUX_PAIRED_FINAL_BENCHMARK = "FALSE",
    leader_backbone_env(use_oof = cand$use_oof),
    GEOVERSA_AUTO_BLW_FLOOR = format(cand$blw_floor, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BLW_SLOPE = format(cand$blw_slope, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_ALPHA_ME_MULT = format(cand$alpha_me_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_LAMBDA_COV_MULT = format(cand$lambda_cov_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BETA_INTERCEPT = format(cand$beta_intercept, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BETA_SLOPE = format(cand$beta_slope, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_K_MULT = format(cand$k_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_PATCH_DROPOUT_MULT = format(cand$patch_dropout_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_WD_MULT = format(cand$wd_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BANK_REFRESH_MULT = format(cand$bank_refresh_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_TRUST_FLOOR = format(cand$trust_floor, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_TRUST_SPAN = format(cand$trust_span, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BANK_CURRENT_FLOOR = format(cand$bank_current_floor, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BANK_CURRENT_CEILING = format(cand$bank_current_ceiling, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BANK_CURRENT_POWER = format(cand$bank_current_power, scientific = FALSE, trim = TRUE)
  )

  cat("\n==================================================\n")
  cat(sprintf("GeoVersa auto meta-search | candidate %d/%d | %s\n",
              i, length(candidates), cand$name))
  cat("==================================================\n")

  with_env_overrides(env_values, {
    run_env <- new.env(parent = globalenv())
    sys.source(benchmark_script, envir = run_env)
  })

  summary_path <- file.path(variant_dir, "wadoux_style_rf_conv_summary_by_protocol.csv")
  diag_path <- file.path(variant_dir, "wadoux_style_rf_conv_fold_diagnostic_summary.csv")

  summary_df <- read.csv(summary_path, stringsAsFactors = FALSE) %>%
    filter(protocol == "DesignBased", model == "ConvKrigingNet2D")
  diag_df <- read.csv(diag_path, stringsAsFactors = FALSE) %>%
    filter(protocol == "DesignBased", model == "ConvKrigingNet2D")

  rmse_sd_value <- summary_df$RMSE_sd[1]
  if (!is.finite(rmse_sd_value)) rmse_sd_value <- 0
  gain_shortfall <- diag_df$test_RMSE_gain_mean[1]
  if (!is.finite(gain_shortfall)) gain_shortfall <- 0
  gain_shortfall <- max(0, -gain_shortfall)
  objective <- summary_df$RMSE_mean[1] +
    rmse_sd_weight * rmse_sd_value +
    gain_penalty_weight * gain_shortfall

  score_rows[[i]] <- cbind(
    as.data.frame(cand, stringsAsFactors = FALSE),
    data.frame(
      RMSE_mean = summary_df$RMSE_mean[1],
      RMSE_sd = rmse_sd_value,
      r2_mean = summary_df$r2_mean[1],
      MEC_mean = summary_df$MEC_mean[1],
      ME_mean = summary_df$ME_mean[1],
      test_RMSE_gain_mean = diag_df$test_RMSE_gain_mean[1],
      objective = objective,
      stringsAsFactors = FALSE
    )
  )
  diag_rows[[i]] <- cbind(
    data.frame(candidate = cand$name, stringsAsFactors = FALSE),
    diag_df
  )
}

score_table <- bind_rows(score_rows) %>%
  arrange(objective, RMSE_mean)
diag_table <- bind_rows(diag_rows)

write.csv(score_table, file.path(results_root, "geoversa_auto_meta_scores.csv"), row.names = FALSE)
write.csv(diag_table, file.path(results_root, "geoversa_auto_meta_diagnostics.csv"), row.names = FALSE)

cat("\n=== GeoVersa auto meta-search complete ===\n")
print(score_table)
cat(sprintf("\nResults written to: %s\n", results_root))
