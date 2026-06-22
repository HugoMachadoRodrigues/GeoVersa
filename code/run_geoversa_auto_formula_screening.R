rm(list = ls())

library(dplyr)

project_root <- normalizePath(getwd(), mustWork = TRUE)
benchmark_script <- file.path(project_root, "code", "run_wadoux_style_rf_conv_comparison.R")
auto_script <- normalizePath(file.path(project_root, "code", "ConvKrigingNet2D_Auto.R"), mustWork = FALSE)
auto_v5_script <- normalizePath(file.path(project_root, "code", "ConvKrigingNet2D_Auto_v5.R"), mustWork = FALSE)

screen_scenario <- Sys.getenv("GEOVERSA_SCREEN_SCENARIO", unset = "random")
screen_protocols <- Sys.getenv("GEOVERSA_SCREEN_PROTOCOLS", unset = "DesignBased")
screen_sample_size <- as.integer(Sys.getenv("GEOVERSA_SCREEN_SAMPLE_SIZE", unset = "500"))
screen_n_iter <- as.integer(Sys.getenv("GEOVERSA_SCREEN_N_ITER", unset = "3"))
screen_device <- Sys.getenv("GEOVERSA_SCREEN_DEVICE", unset = Sys.getenv("WADOUX_DEVICE", unset = "cpu"))
screen_profile <- Sys.getenv("GEOVERSA_SCREEN_PROFILE", unset = "auto")
screen_variant_set <- tolower(Sys.getenv("GEOVERSA_SCREEN_VARIANT_SET", unset = "metarules_quick"))
screen_use_oof <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_SCREEN_USE_OOF", unset = "0")))
screen_use_oof <- is.finite(screen_use_oof) && screen_use_oof > 0
results_root <- Sys.getenv(
  "GEOVERSA_SCREEN_RESULTS_DIR",
  unset = file.path(project_root, "results", sprintf("geoversa_auto_formula_screening_%s", format(Sys.Date(), "%Y%m%d")))
)
dir.create(results_root, recursive = TRUE, showWarnings = FALSE)

rmse_sd_weight <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_SCREEN_RMSE_SD_WEIGHT", unset = "0.20")))
if (!is.finite(rmse_sd_weight)) rmse_sd_weight <- 0.20
gain_penalty_weight <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_SCREEN_GAIN_PENALTY", unset = "2.00")))
if (!is.finite(gain_penalty_weight)) gain_penalty_weight <- 2.00

reference_summary_path <- Sys.getenv(
  "GEOVERSA_SCREEN_REFERENCE_SUMMARY",
  unset = file.path(
    project_root,
    "results",
    "wadoux2021_rf_reference_random_designbased_10iter_pearson_20260405",
    "wadoux2021_rf_random_absolute_summary.csv"
  )
)

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

make_variant <- function(name,
                         blw_floor = 0.05,
                         blw_slope = 0.10,
                         alpha_me_mult = 1.0,
                         lambda_cov_mult = 1.0,
                         beta_intercept = 2.0,
                         beta_slope = -6.0,
                         k_mult = 1.0,
                         k_nugget_slope = 0.0,
                         k_nugget_ref = 0.18,
                         bank_refresh_mult = 1.0,
                         wd_mult = 1.0,
                         dropout_mult = 1.0,
                         tab_dropout_mult = 1.0,
                         patch_dropout_mult = 1.0,
                         coord_dropout_mult = 1.0,
                         patch_dropout_nugget_slope = 0.0,
                         patch_dropout_nugget_ref = 0.18,
                         patch_dropout_k_slope = 0.0,
                         patch_dropout_k_ref = 20.0,
                         trust_floor = 0.18,
                         trust_span = 0.72,
                         bank_current_floor = 0.25,
                         bank_current_ceiling = 0.85,
                         bank_current_power = 0.50,
                         use_oof = screen_use_oof,
                         overrides = character()) {
  env <- c(
    leader_backbone_env(use_oof = use_oof),
    GEOVERSA_AUTO_BLW_FLOOR = format(blw_floor, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BLW_SLOPE = format(blw_slope, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_ALPHA_ME_MULT = format(alpha_me_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_LAMBDA_COV_MULT = format(lambda_cov_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BETA_INTERCEPT = format(beta_intercept, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BETA_SLOPE = format(beta_slope, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_K_MULT = format(k_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_K_NUGGET_SLOPE = format(k_nugget_slope, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_K_NUGGET_REF = format(k_nugget_ref, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BANK_REFRESH_MULT = format(bank_refresh_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_WD_MULT = format(wd_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_DROPOUT_MULT = format(dropout_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_TAB_DROPOUT_MULT = format(tab_dropout_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_PATCH_DROPOUT_MULT = format(patch_dropout_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_COORD_DROPOUT_MULT = format(coord_dropout_mult, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_PATCH_DROPOUT_NUGGET_SLOPE = format(patch_dropout_nugget_slope, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_PATCH_DROPOUT_NUGGET_REF = format(patch_dropout_nugget_ref, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_PATCH_DROPOUT_K_SLOPE = format(patch_dropout_k_slope, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_PATCH_DROPOUT_K_REF = format(patch_dropout_k_ref, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_TRUST_FLOOR = format(trust_floor, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_TRUST_SPAN = format(trust_span, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BANK_CURRENT_FLOOR = format(bank_current_floor, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BANK_CURRENT_CEILING = format(bank_current_ceiling, scientific = FALSE, trim = TRUE),
    GEOVERSA_AUTO_BANK_CURRENT_POWER = format(bank_current_power, scientific = FALSE, trim = TRUE)
  )
  if (length(overrides) > 0) {
    env[names(overrides)] <- overrides
  }
  list(
    name = name,
    env = env,
    auto_blw_floor = blw_floor,
    auto_blw_slope = blw_slope,
    auto_alpha_me_mult = alpha_me_mult,
    auto_lambda_cov_mult = lambda_cov_mult,
    auto_beta_intercept = beta_intercept,
    auto_beta_slope = beta_slope,
    auto_k_mult = k_mult,
    auto_k_nugget_slope = k_nugget_slope,
    auto_k_nugget_ref = k_nugget_ref,
    auto_bank_refresh_mult = bank_refresh_mult,
    auto_wd_mult = wd_mult,
    auto_dropout_mult = dropout_mult,
    auto_tab_dropout_mult = tab_dropout_mult,
    auto_patch_dropout_mult = patch_dropout_mult,
    auto_coord_dropout_mult = coord_dropout_mult,
    auto_patch_dropout_nugget_slope = patch_dropout_nugget_slope,
    auto_patch_dropout_nugget_ref = patch_dropout_nugget_ref,
    auto_patch_dropout_k_slope = patch_dropout_k_slope,
    auto_patch_dropout_k_ref = patch_dropout_k_ref,
    auto_trust_floor = trust_floor,
    auto_trust_span = trust_span,
    auto_bank_current_floor = bank_current_floor,
    auto_bank_current_ceiling = bank_current_ceiling,
    auto_bank_current_power = bank_current_power,
    auto_use_oof = use_oof
  )
}

variants_quick <- list(
  make_variant("baseline"),
  make_variant("blw_floor_0_06", blw_floor = 0.06),
  make_variant("blw_slope_0_12", blw_slope = 0.12),
  make_variant("beta_gentler", beta_intercept = 1.5, beta_slope = -5.0),
  make_variant("k_mult_0_85", k_mult = 0.85),
  make_variant("blw_floor_0_06_k_mult_0_85", blw_floor = 0.06, k_mult = 0.85),
  make_variant("k_mult_1_15", k_mult = 1.15)
)

variants_full <- c(
  variants_quick,
  list(
    make_variant("beta_stronger", beta_intercept = 2.5, beta_slope = -7.0),
    make_variant("blw_floor_0_06_slope_0_12", blw_floor = 0.06, blw_slope = 0.12),
    make_variant("k_mult_0_70", k_mult = 0.70),
    make_variant("k_mult_1_30", k_mult = 1.30)
  )
)

variants_beta_bank <- list(
  make_variant("baseline"),
  make_variant("beta_gentler", beta_intercept = 1.5, beta_slope = -5.0),
  make_variant("beta_stronger", beta_intercept = 2.5, beta_slope = -7.0),
  make_variant("bank_refresh_x2", bank_refresh_mult = 2.0),
  make_variant("bank_refresh_x3", bank_refresh_mult = 3.0)
)

variants_alpha_cov <- list(
  make_variant("baseline"),
  make_variant("alpha_me_x0_5", alpha_me_mult = 0.5),
  make_variant("alpha_me_x1_5", alpha_me_mult = 1.5),
  make_variant("lambda_cov_x0", lambda_cov_mult = 0.0),
  make_variant("lambda_cov_x2", lambda_cov_mult = 2.0)
)

variants_interaction_abk <- list(
  make_variant("baseline"),
  make_variant("a0_85_k0_85_beta_gentler", alpha_me_mult = 0.85, k_mult = 0.85, beta_intercept = 1.5, beta_slope = -5.0),
  make_variant("a0_85_k0_85_beta_stronger", alpha_me_mult = 0.85, k_mult = 0.85, beta_intercept = 2.5, beta_slope = -7.0),
  make_variant("a0_85_k1_15_beta_gentler", alpha_me_mult = 0.85, k_mult = 1.15, beta_intercept = 1.5, beta_slope = -5.0),
  make_variant("a0_85_k1_15_beta_stronger", alpha_me_mult = 0.85, k_mult = 1.15, beta_intercept = 2.5, beta_slope = -7.0),
  make_variant("a1_15_k0_85_beta_gentler", alpha_me_mult = 1.15, k_mult = 0.85, beta_intercept = 1.5, beta_slope = -5.0),
  make_variant("a1_15_k0_85_beta_stronger", alpha_me_mult = 1.15, k_mult = 0.85, beta_intercept = 2.5, beta_slope = -7.0),
  make_variant("a1_15_k1_15_beta_gentler", alpha_me_mult = 1.15, k_mult = 1.15, beta_intercept = 1.5, beta_slope = -5.0),
  make_variant("a1_15_k1_15_beta_stronger", alpha_me_mult = 1.15, k_mult = 1.15, beta_intercept = 2.5, beta_slope = -7.0)
)

variants_cov_beta <- list(
  make_variant("baseline"),
  make_variant("beta_gentler_cov_x0_5", beta_intercept = 1.5, beta_slope = -5.0, lambda_cov_mult = 0.5),
  make_variant("beta_gentler_cov_x1_5", beta_intercept = 1.5, beta_slope = -5.0, lambda_cov_mult = 1.5),
  make_variant("beta_stronger_cov_x0_5", beta_intercept = 2.5, beta_slope = -7.0, lambda_cov_mult = 0.5),
  make_variant("beta_stronger_cov_x1_0", beta_intercept = 2.5, beta_slope = -7.0, lambda_cov_mult = 1.0),
  make_variant("beta_stronger_cov_x1_5", beta_intercept = 2.5, beta_slope = -7.0, lambda_cov_mult = 1.5),
  make_variant("beta_stronger_cov_x2_0", beta_intercept = 2.5, beta_slope = -7.0, lambda_cov_mult = 2.0)
)

variants_reg_fine <- list(
  make_variant("baseline"),
  make_variant("wd_x0_5", wd_mult = 0.5),
  make_variant("wd_x1_5", wd_mult = 1.5),
  make_variant("drop_x0_75", dropout_mult = 0.75),
  make_variant("drop_x1_25", dropout_mult = 1.25),
  make_variant("wd_x0_5_drop_x0_75", wd_mult = 0.5, dropout_mult = 0.75),
  make_variant("wd_x1_5_drop_x1_25", wd_mult = 1.5, dropout_mult = 1.25)
)

variants_followup_fine <- list(
  make_variant("baseline"),
  make_variant("drop_x0_75", dropout_mult = 0.75),
  make_variant("drop_x1_25", dropout_mult = 1.25),
  make_variant("beta_stronger_cov_x1_0", beta_intercept = 2.5, beta_slope = -7.0, lambda_cov_mult = 1.0)
)

variants_drop_split <- list(
  make_variant("baseline"),
  make_variant("tab_drop_x0_75", tab_dropout_mult = 0.75),
  make_variant("tab_drop_x1_25", tab_dropout_mult = 1.25),
  make_variant("patch_drop_x0_75", patch_dropout_mult = 0.75),
  make_variant("patch_drop_x1_25", patch_dropout_mult = 1.25),
  make_variant("coord_drop_x0_75", coord_dropout_mult = 0.75),
  make_variant("coord_drop_x1_25", coord_dropout_mult = 1.25)
)

variants_patch_wd_fine <- list(
  make_variant("baseline"),
  make_variant("patch_x0_90", patch_dropout_mult = 0.90),
  make_variant("patch_x1_10", patch_dropout_mult = 1.10),
  make_variant("wd_x0_80", wd_mult = 0.80),
  make_variant("wd_x1_20", wd_mult = 1.20),
  make_variant("patch_x0_90_wd_x0_80", patch_dropout_mult = 0.90, wd_mult = 0.80),
  make_variant("patch_x0_90_wd_x1_20", patch_dropout_mult = 0.90, wd_mult = 1.20),
  make_variant("patch_x1_10_wd_x0_80", patch_dropout_mult = 1.10, wd_mult = 0.80),
  make_variant("patch_x1_10_wd_x1_20", patch_dropout_mult = 1.10, wd_mult = 1.20)
)

variants_patch_structural <- list(
  make_variant("baseline"),
  make_variant("patch_nugget_up", patch_dropout_nugget_slope = 0.80),
  make_variant("patch_nugget_down", patch_dropout_nugget_slope = -0.80),
  make_variant("patch_k_up", patch_dropout_k_slope = 0.20),
  make_variant("patch_k_down", patch_dropout_k_slope = -0.20),
  make_variant("patch_nugget_k_up", patch_dropout_nugget_slope = 0.80, patch_dropout_k_slope = 0.20)
)

variants_patch_backbone <- list(
  make_variant("baseline"),
  make_variant("blw_floor_0_045", blw_floor = 0.045),
  make_variant("blw_floor_0_060", blw_floor = 0.060),
  make_variant("patch_x0_90_blw_0_045", blw_floor = 0.045, patch_dropout_mult = 0.90),
  make_variant("patch_x0_90_blw_0_060", blw_floor = 0.060, patch_dropout_mult = 0.90)
)

variants_k_structural <- list(
  make_variant("baseline"),
  make_variant("k_nugget_up", k_nugget_slope = 2.0),
  make_variant("k_nugget_down", k_nugget_slope = -2.0)
)

variants_metarules_quick <- list(
  make_variant("baseline"),
  make_variant("beta_gentler", beta_intercept = 1.7, beta_slope = -5.5),
  make_variant("beta_stronger", beta_intercept = 2.3, beta_slope = -6.5),
  make_variant("k_mult_0_92", k_mult = 0.92),
  make_variant("k_mult_1_05", k_mult = 1.05),
  make_variant("patch_drop_0_95", patch_dropout_mult = 0.95),
  make_variant("wd_x0_90", wd_mult = 0.90),
  make_variant("blw_floor_0_048", blw_floor = 0.048),
  make_variant("blw_floor_0_055", blw_floor = 0.055),
  make_variant("alpha_me_0_92", alpha_me_mult = 0.92),
  make_variant("alpha_me_1_08", alpha_me_mult = 1.08),
  make_variant("lambda_cov_0_75", lambda_cov_mult = 0.75)
)

variants_metarules_full <- c(
  variants_metarules_quick,
  list(
    make_variant("blw_slope_0_095", blw_slope = 0.095),
    make_variant("blw_slope_0_110", blw_slope = 0.110),
    make_variant("wd_x1_10", wd_mult = 1.10),
    make_variant("patch_drop_1_05", patch_dropout_mult = 1.05),
    make_variant("bank_refresh_x1_5", bank_refresh_mult = 1.5),
    make_variant("bank_refresh_x2_0", bank_refresh_mult = 2.0),
    make_variant("beta_gentler_k_0_92", beta_intercept = 1.7, beta_slope = -5.5, k_mult = 0.92),
    make_variant("beta_stronger_k_1_05", beta_intercept = 2.3, beta_slope = -6.5, k_mult = 1.05)
  )
)

variants <- switch(
  screen_variant_set,
  quick = variants_quick,
  full = variants_full,
  beta_bank = variants_beta_bank,
  alpha_cov = variants_alpha_cov,
  interaction_abk = variants_interaction_abk,
  cov_beta = variants_cov_beta,
  reg_fine = variants_reg_fine,
  followup_fine = variants_followup_fine,
  drop_split = variants_drop_split,
  patch_wd_fine = variants_patch_wd_fine,
  patch_structural = variants_patch_structural,
  patch_backbone = variants_patch_backbone,
  k_structural = variants_k_structural,
  metarules_quick = variants_metarules_quick,
  metarules_full = variants_metarules_full,
  stop("Unsupported GEOVERSA_SCREEN_VARIANT_SET: ", screen_variant_set)
)

variant_filter_raw <- Sys.getenv("GEOVERSA_SCREEN_VARIANTS", unset = "")
if (nzchar(variant_filter_raw)) {
  variant_filter <- trimws(unlist(strsplit(variant_filter_raw, ",")))
  variant_filter <- variant_filter[nzchar(variant_filter)]
  variants <- Filter(function(v) v$name %in% variant_filter, variants)
  if (length(variants) == 0) {
    stop("GEOVERSA_SCREEN_VARIANTS did not match any defined variants.")
  }
}

reference_design <- NULL
if (file.exists(reference_summary_path)) {
  ref_df <- read.csv(reference_summary_path, stringsAsFactors = FALSE)
  ref_design <- ref_df %>% filter(protocol == "DesignBased")
  if (nrow(ref_design) > 0) {
    reference_design <- setNames(ref_design$mean_value, ref_design$metric)
  }
}

manifest <- bind_rows(lapply(variants, function(v) {
  data.frame(
    variant = v$name,
    auto_use_oof = v$auto_use_oof,
    auto_blw_floor = v$auto_blw_floor,
    auto_blw_slope = v$auto_blw_slope,
    auto_alpha_me_mult = v$auto_alpha_me_mult,
    auto_lambda_cov_mult = v$auto_lambda_cov_mult,
    auto_beta_intercept = v$auto_beta_intercept,
    auto_beta_slope = v$auto_beta_slope,
    auto_k_mult = v$auto_k_mult,
    auto_k_nugget_slope = v$auto_k_nugget_slope,
    auto_k_nugget_ref = v$auto_k_nugget_ref,
    auto_bank_refresh_mult = v$auto_bank_refresh_mult,
    auto_wd_mult = v$auto_wd_mult,
    auto_dropout_mult = v$auto_dropout_mult,
    auto_tab_dropout_mult = v$auto_tab_dropout_mult,
    auto_patch_dropout_mult = v$auto_patch_dropout_mult,
    auto_coord_dropout_mult = v$auto_coord_dropout_mult,
    auto_patch_dropout_nugget_slope = v$auto_patch_dropout_nugget_slope,
    auto_patch_dropout_nugget_ref = v$auto_patch_dropout_nugget_ref,
    auto_patch_dropout_k_slope = v$auto_patch_dropout_k_slope,
    auto_patch_dropout_k_ref = v$auto_patch_dropout_k_ref,
    auto_trust_floor = v$auto_trust_floor,
    auto_trust_span = v$auto_trust_span,
    auto_bank_current_floor = v$auto_bank_current_floor,
    auto_bank_current_ceiling = v$auto_bank_current_ceiling,
    auto_bank_current_power = v$auto_bank_current_power,
    stringsAsFactors = FALSE
  )
}))
write.csv(manifest, file.path(results_root, "geoversa_formula_variant_manifest.csv"), row.names = FALSE)

summary_rows <- vector("list", length(variants))
protocol_rows <- vector("list", length(variants))
diag_rows <- vector("list", length(variants))

for (i in seq_along(variants)) {
  variant_def <- variants[[i]]
  variant_dir <- file.path(results_root, variant_def$name)
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
    WADOUX_PAIRED_FINAL_BENCHMARK = "FALSE"
  )
  env_values[names(variant_def$env)] <- variant_def$env

  cat("\n==================================================\n")
  cat(sprintf("GeoVersa auto-formula screening | variant %d/%d | %s\n",
              i, length(variants), variant_def$name))
  cat("==================================================\n")

  with_env_overrides(env_values, {
    run_env <- new.env(parent = globalenv())
    sys.source(benchmark_script, envir = run_env)
  })

  summary_path <- file.path(variant_dir, "wadoux_style_rf_conv_summary_by_protocol.csv")
  diag_summary_path <- file.path(variant_dir, "wadoux_style_rf_conv_fold_diagnostic_summary.csv")

  variant_protocol_summary <- read.csv(summary_path, stringsAsFactors = FALSE) %>%
    mutate(
      variant = variant_def$name,
      auto_use_oof = variant_def$auto_use_oof,
      auto_blw_floor = variant_def$auto_blw_floor,
      auto_blw_slope = variant_def$auto_blw_slope,
      auto_alpha_me_mult = variant_def$auto_alpha_me_mult,
      auto_lambda_cov_mult = variant_def$auto_lambda_cov_mult,
      auto_beta_intercept = variant_def$auto_beta_intercept,
      auto_beta_slope = variant_def$auto_beta_slope,
      auto_k_mult = variant_def$auto_k_mult,
      auto_k_nugget_slope = variant_def$auto_k_nugget_slope,
      auto_k_nugget_ref = variant_def$auto_k_nugget_ref,
      auto_bank_refresh_mult = variant_def$auto_bank_refresh_mult,
      auto_wd_mult = variant_def$auto_wd_mult,
      auto_dropout_mult = variant_def$auto_dropout_mult,
      auto_tab_dropout_mult = variant_def$auto_tab_dropout_mult,
      auto_patch_dropout_mult = variant_def$auto_patch_dropout_mult,
      auto_coord_dropout_mult = variant_def$auto_coord_dropout_mult,
      auto_patch_dropout_nugget_slope = variant_def$auto_patch_dropout_nugget_slope,
      auto_patch_dropout_nugget_ref = variant_def$auto_patch_dropout_nugget_ref,
      auto_patch_dropout_k_slope = variant_def$auto_patch_dropout_k_slope,
      auto_patch_dropout_k_ref = variant_def$auto_patch_dropout_k_ref,
      auto_trust_floor = variant_def$auto_trust_floor,
      auto_trust_span = variant_def$auto_trust_span,
      auto_bank_current_floor = variant_def$auto_bank_current_floor,
      auto_bank_current_ceiling = variant_def$auto_bank_current_ceiling,
      auto_bank_current_power = variant_def$auto_bank_current_power,
      .before = 1
    )
  protocol_rows[[i]] <- variant_protocol_summary

  variant_diag_summary <- if (file.exists(diag_summary_path)) {
    read.csv(diag_summary_path, stringsAsFactors = FALSE) %>%
      mutate(
        variant = variant_def$name,
        auto_use_oof = variant_def$auto_use_oof,
        auto_blw_floor = variant_def$auto_blw_floor,
        auto_blw_slope = variant_def$auto_blw_slope,
        auto_alpha_me_mult = variant_def$auto_alpha_me_mult,
        auto_lambda_cov_mult = variant_def$auto_lambda_cov_mult,
        auto_beta_intercept = variant_def$auto_beta_intercept,
        auto_beta_slope = variant_def$auto_beta_slope,
        auto_k_mult = variant_def$auto_k_mult,
        auto_k_nugget_slope = variant_def$auto_k_nugget_slope,
        auto_k_nugget_ref = variant_def$auto_k_nugget_ref,
        auto_bank_refresh_mult = variant_def$auto_bank_refresh_mult,
        auto_wd_mult = variant_def$auto_wd_mult,
        auto_dropout_mult = variant_def$auto_dropout_mult,
        auto_tab_dropout_mult = variant_def$auto_tab_dropout_mult,
        auto_patch_dropout_mult = variant_def$auto_patch_dropout_mult,
        auto_coord_dropout_mult = variant_def$auto_coord_dropout_mult,
        auto_patch_dropout_nugget_slope = variant_def$auto_patch_dropout_nugget_slope,
        auto_patch_dropout_nugget_ref = variant_def$auto_patch_dropout_nugget_ref,
        auto_patch_dropout_k_slope = variant_def$auto_patch_dropout_k_slope,
        auto_patch_dropout_k_ref = variant_def$auto_patch_dropout_k_ref,
        auto_trust_floor = variant_def$auto_trust_floor,
        auto_trust_span = variant_def$auto_trust_span,
        auto_bank_current_floor = variant_def$auto_bank_current_floor,
        auto_bank_current_ceiling = variant_def$auto_bank_current_ceiling,
        auto_bank_current_power = variant_def$auto_bank_current_power,
        .before = 1
      )
  } else {
    data.frame()
  }
  diag_rows[[i]] <- variant_diag_summary

  design_row <- variant_protocol_summary %>%
    filter(protocol == "DesignBased", model == "ConvKrigingNet2D")
  if (nrow(design_row) == 0) next

  design_diag_row <- variant_diag_summary %>%
    filter(protocol == "DesignBased", model == "ConvKrigingNet2D")

  row <- data.frame(
    variant = variant_def$name,
    auto_use_oof = variant_def$auto_use_oof,
    auto_blw_floor = variant_def$auto_blw_floor,
    auto_blw_slope = variant_def$auto_blw_slope,
    auto_alpha_me_mult = variant_def$auto_alpha_me_mult,
    auto_lambda_cov_mult = variant_def$auto_lambda_cov_mult,
    auto_beta_intercept = variant_def$auto_beta_intercept,
    auto_beta_slope = variant_def$auto_beta_slope,
    auto_k_mult = variant_def$auto_k_mult,
    auto_k_nugget_slope = variant_def$auto_k_nugget_slope,
    auto_k_nugget_ref = variant_def$auto_k_nugget_ref,
    auto_bank_refresh_mult = variant_def$auto_bank_refresh_mult,
    auto_wd_mult = variant_def$auto_wd_mult,
    auto_dropout_mult = variant_def$auto_dropout_mult,
    auto_tab_dropout_mult = variant_def$auto_tab_dropout_mult,
    auto_patch_dropout_mult = variant_def$auto_patch_dropout_mult,
    auto_coord_dropout_mult = variant_def$auto_coord_dropout_mult,
    auto_patch_dropout_nugget_slope = variant_def$auto_patch_dropout_nugget_slope,
    auto_patch_dropout_nugget_ref = variant_def$auto_patch_dropout_nugget_ref,
    auto_patch_dropout_k_slope = variant_def$auto_patch_dropout_k_slope,
    auto_patch_dropout_k_ref = variant_def$auto_patch_dropout_k_ref,
    auto_trust_floor = variant_def$auto_trust_floor,
    auto_trust_span = variant_def$auto_trust_span,
    auto_bank_current_floor = variant_def$auto_bank_current_floor,
    auto_bank_current_ceiling = variant_def$auto_bank_current_ceiling,
    auto_bank_current_power = variant_def$auto_bank_current_power,
    scenario = design_row$scenario[1],
    protocol = design_row$protocol[1],
    n_iter = screen_n_iter,
    RMSE_mean = design_row$RMSE_mean[1],
    r2_mean = design_row$r2_mean[1],
    MEC_mean = design_row$MEC_mean[1],
    ME_mean = design_row$ME_mean[1],
    stringsAsFactors = FALSE
  )

  if (nrow(design_diag_row) > 0) {
    row$test_RMSE_base_mean <- design_diag_row$test_RMSE_base_mean[1]
    row$test_RMSE_full_mean <- design_diag_row$test_RMSE_full_mean[1]
    row$test_RMSE_gain_mean <- design_diag_row$test_RMSE_gain_mean[1]
    row$test_r2_gain_mean <- design_diag_row$test_r2_gain_mean[1]
    row$test_MEC_gain_mean <- design_diag_row$test_MEC_gain_mean[1]
    row$beta_final_mean <- design_diag_row$beta_final_mean[1]
    row$K_neighbors_mean <- design_diag_row$K_neighbors_mean[1]
    row$nugget_ratio_mean <- design_diag_row$nugget_ratio_mean[1]
    row$mean_abs_delta_scaled_test_mean <- design_diag_row$mean_abs_delta_scaled_test_mean[1]
    row$warmup_epochs_done_mean <- design_diag_row$warmup_epochs_done_mean[1]
  }

  if (!is.null(reference_design)) {
    row$rf_reference_RMSE <- unname(reference_design["RMSE"])
    row$rf_reference_r2 <- unname(reference_design["r2"])
    row$rf_reference_MEC <- unname(reference_design["MEC"])
    row$delta_RMSE_vs_rf_reference <- row$RMSE_mean - row$rf_reference_RMSE
    row$delta_r2_vs_rf_reference <- row$r2_mean - row$rf_reference_r2
    row$delta_MEC_vs_rf_reference <- row$MEC_mean - row$rf_reference_MEC
  }

  rmse_sd_value <- design_row$RMSE_sd[1]
  if (!is.finite(rmse_sd_value)) rmse_sd_value <- 0
  gain_shortfall <- if (!is.null(row$test_RMSE_gain_mean) && is.finite(row$test_RMSE_gain_mean)) {
    max(0, -row$test_RMSE_gain_mean)
  } else {
    0
  }
  row$objective <- row$RMSE_mean + rmse_sd_weight * rmse_sd_value + gain_penalty_weight * gain_shortfall

  summary_rows[[i]] <- row
}

screening_summary <- bind_rows(summary_rows) %>% arrange(objective, RMSE_mean)
protocol_summary <- bind_rows(protocol_rows)
diagnostic_summary <- bind_rows(diag_rows)

write.csv(screening_summary, file.path(results_root, "geoversa_formula_screening_summary.csv"), row.names = FALSE)
write.csv(protocol_summary, file.path(results_root, "geoversa_formula_screening_protocol_summary.csv"), row.names = FALSE)
if (nrow(diagnostic_summary) > 0) {
  write.csv(diagnostic_summary, file.path(results_root, "geoversa_formula_screening_diagnostic_summary.csv"), row.names = FALSE)
}

cat("\n=== GeoVersa auto-formula screening complete ===\n")
if (nrow(screening_summary) > 0) {
  print(screening_summary)
}
cat(sprintf("\nScreening results written to: %s\n", results_root))
