rm(list = ls())
set.seed(123)

source("./code/wadoux2021_rf_reproduction_helpers.R")

# =============================================================================
# run_wadoux_style_rf_conv_comparison.R
#
# Fair head-to-head benchmark between:
#   - RF (ranger defaults; Wadoux-style baseline)
#   - ConvKrigingNet2D (anisotropic variant by default)
#
# Goals:
#   - same sampled calibration sets
#   - same outer folds / test points
#   - same final metrics (Wadoux metrics: ME, RMSE, Pearson^2, MEC)
#   - same inner train/validation split inside each outer training set
#
# This script is the comparison benchmark. Exact RF-only reproduction of Wadoux
# remains separate in the dedicated reproduction runners.
#
# Default protocols in the comparison benchmark:
#   - DesignBased
#   - RandomKFold
#   - SpatialKFold
#   - BLOOCV
#
# Optional:
#   - Population can be enabled, but is disabled by default because full-map
#     ConvKrigingNet2D prediction over ~1.3M cells is extremely expensive.
#
# Environment overrides:
#   WADOUX_SCENARIO           random | regular_grid | clustered_random
#   WADOUX_PROTOCOLS          comma-separated list
#   WADOUX_INCLUDE_POPULATION TRUE/FALSE
#   WADOUX_N_ITER
#   WADOUX_SAMPLE_SIZE
#   WADOUX_VAL_FRAC
#   WADOUX_VAL_DIST_KM
#   WADOUX_RANDOM_K
#   WADOUX_BLOO_GROUPS
#   WADOUX_BLOO_TEST_PIXELS
#   WADOUX_CLUSTER_CENTERS
#   WADOUX_CLUSTER_N_PSU
#   WADOUX_CLUSTER_M_PER_PSU
#   WADOUX_MODEL_PROFILE      quick | full
#   WADOUX_TRAIN_SEED
#   WADOUX_DEVICE
#   WADOUX_RF_REFIT_FULL_CALIBRATION
#   WADOUX_REFIT_SELECTION_SPLITS
#   WADOUX_RESULTS_DIR
# =============================================================================

parse_bool_env <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}

parse_num_env <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) default else val
}

parse_protocols_env <- function(default = c("DesignBased", "RandomKFold", "SpatialKFold", "BLOOCV")) {
  raw <- Sys.getenv("WADOUX_PROTOCOLS", unset = paste(default, collapse = ","))
  vals <- trimws(unlist(strsplit(raw, ",")))
  vals[vals != ""]
}

parse_models_env <- function(default = c("RF", "ConvKrigingNet2D")) {
  raw <- Sys.getenv("WADOUX_MODELS", unset = paste(default, collapse = ","))
  vals <- trimws(unlist(strsplit(raw, ",")))
  vals <- vals[vals != ""]
  allowed <- c("RF", "ConvKrigingNet2D")
  invalid <- setdiff(vals, allowed)
  if (length(invalid) > 0) {
    stop("Unsupported model(s) in WADOUX_MODELS: ", paste(invalid, collapse = ", "))
  }
  unique(vals)
}

env_string <- function(x) {
  if (is.null(x) || length(x) == 0L) return("")
  x <- x[nzchar(names(x)) & !is.na(x)]
  if (length(x) == 0L) return("")
  paste(sprintf("%s=%s", names(x), as.character(x)), collapse = ";")
}

merge_env_overrides <- function(...) {
  vals <- unlist(list(...), recursive = FALSE, use.names = TRUE)
  if (is.null(vals) || length(vals) == 0L) return(character())
  val_names <- names(vals)
  vals <- as.character(vals)
  names(vals) <- val_names
  vals <- vals[nzchar(names(vals)) & !is.na(vals) & nzchar(vals)]
  vals[!duplicated(names(vals), fromLast = TRUE)]
}

make_geoversa_policy <- function(label, env = character(), complexity = 0L,
                                 description = "") {
  env_chr <- as.character(env)
  names(env_chr) <- names(env)
  list(
    label = label,
    env = env_chr,
    complexity = as.integer(complexity),
    description = description
  )
}

make_geoversa_policy_grid <- function(policy_set = "default",
                                      max_candidates = Inf) {
  policies <- list(
    make_geoversa_policy(
      "auto_current",
      character(),
      complexity = 0L,
      description = "Current Auto v5/meta-controller policy; mandatory conservative candidate."
    ),
    make_geoversa_policy(
      "qbasis_no_patch",
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "0",
        GEOVERSA_AUTO_PATCH_CONTEXT = "0",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = "4",
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = "0",
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0"
      ),
      complexity = 1L,
      description = "Strong tabular q-basis with patches/context disabled; tests whether 2D patches hurt DesignBased."
    ),
    make_geoversa_policy(
      "qbasis_patch_lean",
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "0",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = "4",
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = "0",
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0"
      ),
      complexity = 2L,
      description = "Patch pyramid retained, context/VRM disabled; lean spatial feature policy."
    ),
    make_geoversa_policy(
      "spatial_conservative",
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "1",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = "4",
        GEOVERSA_AUTO_K_MULTISCALE = "0",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_LOCAL_GATE_LEARNED = "0",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0",
        GEOVERSA_AUTO_BETA_SIGMA_GATE = "1"
      ),
      complexity = 2L,
      description = "Patch/context on, but single-scale kriging and no VRM/DAM; conservative residual correction."
    ),
    make_geoversa_policy(
      "spatial_regularized",
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "1",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = "4",
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = "0.05",
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "1",
        GEOVERSA_AUTO_VRM_WEIGHT = "0.03",
        GEOVERSA_AUTO_VRM_RESID_WEIGHT = "0.00",
        GEOVERSA_AUTO_DAM = "0",
        GEOVERSA_AUTO_BETA_SIGMA_GATE = "1"
      ),
      complexity = 3L,
      description = "Full spatial policy with lighter VRM regularization and mild tabular dropout."
    )
  )

  if (tolower(policy_set) %in% c("expanded", "full")) {
    policies <- c(
      policies,
      list(
        make_geoversa_policy(
          "tabtree_qbasis_no_patch",
          c(
            GEOVERSA_AUTO_PATCH_PYRAMID = "0",
            GEOVERSA_AUTO_PATCH_CONTEXT = "0",
            GEOVERSA_AUTO_TAB_QBASIS = "1",
            GEOVERSA_AUTO_TAB_QBASIS_KNOTS = "4",
            GEOVERSA_AUTO_TAB_TREE = "1",
            GEOVERSA_AUTO_TAB_TREE_NTREES = "6",
            GEOVERSA_AUTO_TAB_TREE_DEPTH = "2",
            GEOVERSA_AUTO_K_MULTISCALE = "1",
            GEOVERSA_AUTO_LOCAL_GATE = "1",
            GEOVERSA_AUTO_VRM = "0",
            GEOVERSA_AUTO_DAM = "0"
          ),
          complexity = 3L,
          description = "Tabular q-basis plus shallow tree features, without image patches."
        )
      )
    )
  }

  key <- vapply(policies, function(p) env_string(p$env), character(1))
  policies <- policies[!duplicated(key)]
  for (i in seq_along(policies)) {
    policies[[i]]$candidate_id <- i
  }
  if (is.finite(max_candidates) && max_candidates > 0L &&
      length(policies) > max_candidates) {
    policies <- policies[seq_len(max_candidates)]
  }
  policies
}

make_geoversa_data_profile <- function(calibration_df,
                                       response_name,
                                       predictor_names) {
  y <- suppressWarnings(as.numeric(calibration_df[[response_name]]))
  y <- y[is.finite(y)]
  n <- nrow(calibration_df)
  p <- length(intersect(predictor_names, names(calibration_df)))
  skew <- NA_real_
  if (length(y) >= 3L) {
    y_sd <- stats::sd(y)
    if (is.finite(y_sd) && y_sd > 0) {
      skew <- mean(((y - mean(y)) / y_sd)^3)
    }
  }
  list(
    n = as.integer(n),
    p = as.integer(p),
    response_skewness = skew,
    p_over_n = if (n > 0L) p / n else NA_real_
  )
}

make_geoversa_autotune_v2_grid <- function(policy_set = "default",
                                           max_candidates = Inf,
                                           data_profile = list()) {
  n <- suppressWarnings(as.integer(data_profile$n))
  p <- suppressWarnings(as.integer(data_profile$p))
  if (!is.finite(n) || n < 1L) n <- 500L
  if (!is.finite(p) || p < 1L) p <- 20L

  knot_mid <- if (n < 300L) 3L else 4L
  knot_hi <- if (n >= 700L) 8L else if (n >= 350L) 6L else 4L
  drop_mild <- if (p >= 20L) "0.05" else "0.02"
  drop_hi <- if (p >= 35L) "0.12" else if (p >= 20L) "0.10" else "0.05"

  policies <- list(
    make_geoversa_policy(
      "auto_current",
      character(),
      complexity = 0L,
      description = "Auto v5/meta-controller candidate kept as the mandatory reference policy."
    ),
    make_geoversa_policy(
      sprintf("qbasis_no_patch_k%d", knot_mid),
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "0",
        GEOVERSA_AUTO_PATCH_CONTEXT = "0",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = "0",
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0"
      ),
      complexity = 1L,
      description = "Tabular q-basis without patch features; strong low-variance DesignBased candidate."
    ),
    make_geoversa_policy(
      sprintf("qbasis_no_patch_k%d_drop%s", knot_hi, gsub("\\.", "p", drop_mild)),
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "0",
        GEOVERSA_AUTO_PATCH_CONTEXT = "0",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_hi),
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = drop_mild,
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0"
      ),
      complexity = 2L,
      description = "Higher q-basis resolution with mild tabular regularization."
    ),
    make_geoversa_policy(
      "qbasis_no_patch_kmult075",
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "0",
        GEOVERSA_AUTO_PATCH_CONTEXT = "0",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = "0",
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_K_MULT = "0.75",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0"
      ),
      complexity = 2L,
      description = "Tabular q-basis with a smaller kriging neighborhood scale."
    ),
    make_geoversa_policy(
      "qbasis_no_patch_kmult125",
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "0",
        GEOVERSA_AUTO_PATCH_CONTEXT = "0",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = "0",
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_K_MULT = "1.25",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0"
      ),
      complexity = 2L,
      description = "Tabular q-basis with a larger kriging neighborhood scale."
    ),
    make_geoversa_policy(
      sprintf("qbasis_patch_lean_k%d", knot_mid),
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "0",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = "0",
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0"
      ),
      complexity = 2L,
      description = "Lean patch pyramid plus q-basis, without context or VRM."
    ),
    make_geoversa_policy(
      "spatial_conservative",
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "1",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
        GEOVERSA_AUTO_K_MULTISCALE = "0",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_LOCAL_GATE_LEARNED = "0",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0",
        GEOVERSA_AUTO_BETA_SIGMA_GATE = "1"
      ),
      complexity = 2L,
      description = "Patch/context enabled with single-scale kriging and no VRM/DAM."
    ),
    make_geoversa_policy(
      "spatial_conservative_kmult075",
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "1",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
        GEOVERSA_AUTO_K_MULTISCALE = "0",
        GEOVERSA_AUTO_K_MULT = "0.75",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_LOCAL_GATE_LEARNED = "0",
        GEOVERSA_AUTO_VRM = "0",
        GEOVERSA_AUTO_DAM = "0",
        GEOVERSA_AUTO_BETA_SIGMA_GATE = "1"
      ),
      complexity = 3L,
      description = "Spatial conservative policy with smaller kriging neighborhood."
    ),
    make_geoversa_policy(
      sprintf("spatial_regularized_vrm001_drop%s", gsub("\\.", "p", drop_mild)),
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "1",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = drop_mild,
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "1",
        GEOVERSA_AUTO_VRM_WEIGHT = "0.01",
        GEOVERSA_AUTO_VRM_RESID_WEIGHT = "0.00",
        GEOVERSA_AUTO_DAM = "0",
        GEOVERSA_AUTO_BETA_SIGMA_GATE = "1"
      ),
      complexity = 3L,
      description = "Spatial regularized policy with very light VRM."
    ),
    make_geoversa_policy(
      sprintf("spatial_regularized_vrm003_drop%s", gsub("\\.", "p", drop_mild)),
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "1",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = drop_mild,
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "1",
        GEOVERSA_AUTO_VRM_WEIGHT = "0.03",
        GEOVERSA_AUTO_VRM_RESID_WEIGHT = "0.00",
        GEOVERSA_AUTO_DAM = "0",
        GEOVERSA_AUTO_BETA_SIGMA_GATE = "1"
      ),
      complexity = 3L,
      description = "Spatial regularized policy with light VRM."
    ),
    make_geoversa_policy(
      sprintf("spatial_regularized_vrm006_drop%s", gsub("\\.", "p", drop_hi)),
      c(
        GEOVERSA_AUTO_PATCH_PYRAMID = "1",
        GEOVERSA_AUTO_PATCH_CONTEXT = "1",
        GEOVERSA_AUTO_TAB_QBASIS = "1",
        GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_hi),
        GEOVERSA_AUTO_TAB_INPUT_DROPOUT = drop_hi,
        GEOVERSA_AUTO_K_MULTISCALE = "1",
        GEOVERSA_AUTO_LOCAL_GATE = "1",
        GEOVERSA_AUTO_VRM = "1",
        GEOVERSA_AUTO_VRM_WEIGHT = "0.06",
        GEOVERSA_AUTO_VRM_RESID_WEIGHT = "0.00",
        GEOVERSA_AUTO_DAM = "0",
        GEOVERSA_AUTO_BETA_SIGMA_GATE = "1"
      ),
      complexity = 4L,
      description = "Higher-capacity spatial policy with stronger regularization."
    )
  )

  if (tolower(policy_set) %in% c("expanded", "full", "v2_full", "autotune_v2_full")) {
    policies <- c(
      policies,
      list(
        make_geoversa_policy(
          "tabtree_qbasis_no_patch",
          c(
            GEOVERSA_AUTO_PATCH_PYRAMID = "0",
            GEOVERSA_AUTO_PATCH_CONTEXT = "0",
            GEOVERSA_AUTO_TAB_QBASIS = "1",
            GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
            GEOVERSA_AUTO_TAB_TREE = "1",
            GEOVERSA_AUTO_TAB_TREE_NTREES = "6",
            GEOVERSA_AUTO_TAB_TREE_DEPTH = "2",
            GEOVERSA_AUTO_TAB_INPUT_DROPOUT = drop_mild,
            GEOVERSA_AUTO_K_MULTISCALE = "1",
            GEOVERSA_AUTO_LOCAL_GATE = "1",
            GEOVERSA_AUTO_VRM = "0",
            GEOVERSA_AUTO_DAM = "0"
          ),
          complexity = 4L,
          description = "Tabular q-basis plus shallow tree features, without image patches."
        ),
        make_geoversa_policy(
          "design_weight_mild",
          c(
            GEOVERSA_AUTO_PATCH_PYRAMID = "1",
            GEOVERSA_AUTO_PATCH_CONTEXT = "1",
            GEOVERSA_AUTO_TAB_QBASIS = "1",
            GEOVERSA_AUTO_TAB_QBASIS_KNOTS = as.character(knot_mid),
            GEOVERSA_AUTO_DESIGN_WEIGHTS = "1",
            GEOVERSA_AUTO_DESIGN_WEIGHT_MODE = "hybrid",
            GEOVERSA_AUTO_DESIGN_WEIGHT_POWER = "0.75",
            GEOVERSA_AUTO_DESIGN_WEIGHT_ENV_SHARE = "0.50",
            GEOVERSA_AUTO_DESIGN_WEIGHT_CLIP_MIN = "0.50",
            GEOVERSA_AUTO_DESIGN_WEIGHT_CLIP_MAX = "2.00",
            GEOVERSA_AUTO_K_MULTISCALE = "0",
            GEOVERSA_AUTO_VRM = "0",
            GEOVERSA_AUTO_DAM = "0"
          ),
          complexity = 4L,
          description = "Mild design-adaptive weighting for uneven calibration designs."
        )
      )
    )
  }

  key <- vapply(policies, function(p) env_string(p$env), character(1))
  policies <- policies[!duplicated(key)]
  for (i in seq_along(policies)) {
    policies[[i]]$candidate_id <- i
  }
  if (is.finite(max_candidates) && max_candidates > 0L &&
      length(policies) > max_candidates) {
    policies <- policies[seq_len(max_candidates)]
  }
  policies
}

clamp01 <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  if (!is.finite(x)) return(NA_real_)
  max(0, min(1, x))
}

fit_blend_alpha <- function(obs, base_pred, full_pred) {
  obs <- as.numeric(obs)
  base_pred <- as.numeric(base_pred)
  full_pred <- as.numeric(full_pred)
  ok <- is.finite(obs) & is.finite(base_pred) & is.finite(full_pred)
  if (sum(ok) < 3L) return(NA_real_)
  delta <- full_pred[ok] - base_pred[ok]
  denom <- sum(delta^2)
  if (!is.finite(denom) || denom <= 1e-12) return(NA_real_)
  clamp01(sum((obs[ok] - base_pred[ok]) * delta) / denom)
}

blend_predictions <- function(base_pred, full_pred, alpha) {
  alpha <- clamp01(alpha)
  if (!is.finite(alpha)) alpha <- 1
  as.numeric(base_pred) + alpha * (as.numeric(full_pred) - as.numeric(base_pred))
}

geoversa_autotune_equiv_tol <- function(score,
                                        abs_tol = 0.10,
                                        rel_tol = 0.005) {
  score <- suppressWarnings(as.numeric(score)[1])
  abs_tol <- suppressWarnings(as.numeric(abs_tol)[1])
  rel_tol <- suppressWarnings(as.numeric(rel_tol)[1])
  if (!is.finite(abs_tol) || abs_tol < 0) abs_tol <- 0
  if (!is.finite(rel_tol) || rel_tol < 0) rel_tol <- 0
  if (!is.finite(score)) score <- 0
  max(abs_tol, rel_tol * abs(score))
}

geoversa_operator_safety_rank <- function(label) {
  label <- as.character(label)
  ranks <- c(
    base = 0,
    "blend_0.25" = 1,
    "blend_0.50" = 2,
    "blend_0.75" = 3,
    full = 4,
    blend_nnls = 5
  )
  out <- unname(ranks[label])
  out[is.na(out)] <- 99
  as.integer(out)
}

geoversa_select_with_equivalence <- function(score,
                                             complexity,
                                             equiv_abs = 0,
                                             equiv_rel = 0,
                                             safety_rank = NULL,
                                             instability = NULL,
                                             gain = NULL,
                                             candidate_idx = seq_along(score)) {
  score <- suppressWarnings(as.numeric(score))
  complexity <- suppressWarnings(as.numeric(complexity))
  n <- length(score)
  if (length(complexity) != n) complexity <- rep(Inf, n)
  if (is.null(safety_rank) || length(safety_rank) != n) safety_rank <- rep(Inf, n)
  if (is.null(instability) || length(instability) != n) instability <- rep(Inf, n)
  if (is.null(gain) || length(gain) != n) gain <- rep(-Inf, n)
  candidate_idx <- as.integer(candidate_idx)
  if (length(candidate_idx) != n) candidate_idx <- seq_len(n)

  finite <- is.finite(score)
  if (!any(finite)) {
    return(list(
      selected_pos = 1L,
      best_score = Inf,
      equiv_tol = geoversa_autotune_equiv_tol(Inf, equiv_abs, equiv_rel),
      practically_equivalent = rep(FALSE, n),
      practical_delta = rep(Inf, n),
      ordered_pos = seq_len(n)
    ))
  }

  best_score <- min(score[finite], na.rm = TRUE)
  equiv_tol <- geoversa_autotune_equiv_tol(best_score, equiv_abs, equiv_rel)
  practical_delta <- score - best_score
  practically_equivalent <- finite & practical_delta <= equiv_tol
  eligible <- which(practically_equivalent)
  ord <- order(
    complexity[eligible],
    safety_rank[eligible],
    instability[eligible],
    -gain[eligible],
    score[eligible],
    candidate_idx[eligible],
    na.last = TRUE
  )
  non_eligible <- which(finite & !practically_equivalent)
  non_ord <- if (length(non_eligible) > 0L) {
    order(
      score[non_eligible],
      complexity[non_eligible],
      safety_rank[non_eligible],
      instability[non_eligible],
      -gain[non_eligible],
      candidate_idx[non_eligible],
      na.last = TRUE
    )
  } else {
    integer(0)
  }
  failed <- which(!finite)
  ordered_pos <- c(eligible[ord], non_eligible[non_ord], failed)

  list(
    selected_pos = eligible[ord[[1L]]],
    best_score = best_score,
    equiv_tol = equiv_tol,
    practically_equivalent = practically_equivalent,
    practical_delta = practical_delta,
    ordered_pos = ordered_pos
  )
}

evaluate_geoversa_operator_candidates <- function(runs,
                                                  operator_set = "default",
                                                  sd_weight = 0.15,
                                                  equiv_abs = 0,
	                                                  equiv_rel = 0,
	                                                  selection_objective = "rmse",
	                                                  noharm_policy = TRUE,
	                                                  noharm_margin_abs = 0.05,
	                                                  noharm_margin_rel = 0.0,
	                                                  rf_noharm_policy = TRUE,
	                                                  rf_noharm_margin_abs = 0.05,
	                                                  rf_noharm_margin_rel = 0.0) {
  if (!is.list(runs) || length(runs) < 1L) {
    stop("Operator tuning requires at least one selection run.")
  }
  fixed_alphas <- c(0, 0.25, 0.5, 0.75, 1)
  operator_labels <- c("base", "blend_0.25", "blend_0.50", "blend_0.75", "full")
  if (tolower(operator_set) %in% c("minimal", "safe")) {
    fixed_alphas <- c(0, 0.5, 1)
    operator_labels <- c("base", "blend_0.50", "full")
  }
  candidate_defs <- data.frame(
    operator_label = operator_labels,
    alpha_fixed = fixed_alphas,
    complexity = seq_along(fixed_alphas) - 1L,
    stringsAsFactors = FALSE
  )
  if (!tolower(operator_set) %in% c("fixed_only", "minimal_fixed")) {
    candidate_defs <- rbind(
      candidate_defs,
      data.frame(
        operator_label = "blend_nnls",
        alpha_fixed = NA_real_,
        complexity = max(candidate_defs$complexity) + 1L,
        stringsAsFactors = FALSE
      )
    )
  }

  rows <- lapply(seq_len(nrow(candidate_defs)), function(i) {
    op <- candidate_defs[i, , drop = FALSE]
    split_rows <- lapply(runs, function(x) {
      obs <- x$val_obs
      base_pred <- x$out$pred_val_base
      full_pred <- x$out$pred_val
      alpha <- if (is.finite(op$alpha_fixed)) {
        op$alpha_fixed
      } else {
        fit_blend_alpha(obs, base_pred, full_pred)
      }
      if (!is.finite(alpha)) alpha <- 1
      pred <- blend_predictions(base_pred, full_pred, alpha)
	      base_rmse <- wadoux_eval(obs = obs, pred = base_pred)$RMSE
	      full_rmse <- wadoux_eval(obs = obs, pred = full_pred)$RMSE
	      rmse <- wadoux_eval(obs = obs, pred = pred)$RMSE
	      rf_rmse <- if (!is.null(x$rf_pred_val)) {
	        wadoux_eval(obs = obs, pred = x$rf_pred_val)$RMSE
	      } else NA_real_
	      data.frame(
	        split_id = x$split_id,
	        operator_alpha = alpha,
	        val_RMSE = rmse,
	        val_RMSE_base = base_rmse,
	        val_RMSE_full = full_rmse,
	        val_RMSE_rf = rf_rmse,
	        delta_RMSE_vs_RF = if (is.finite(rf_rmse)) rmse - rf_rmse else NA_real_,
	        stringsAsFactors = FALSE
	      )
	    })
	    split_df <- bind_rows(split_rows)
	    rmse_sd <- stats::sd(split_df$val_RMSE, na.rm = TRUE)
	    if (!is.finite(rmse_sd)) rmse_sd <- 0
	    delta_sd <- stats::sd(split_df$delta_RMSE_vs_RF, na.rm = TRUE)
	    if (!is.finite(delta_sd)) delta_sd <- 0
	    selection_objective <- tolower(as.character(selection_objective)[1L])
	    paired_active <- identical(selection_objective, "paired_rf_delta") &&
	      any(is.finite(split_df$delta_RMSE_vs_RF))
	    selection_score <- if (isTRUE(paired_active)) {
	      stats::median(split_df$delta_RMSE_vs_RF, na.rm = TRUE) +
	        sd_weight * delta_sd
	    } else {
	      stats::median(split_df$val_RMSE, na.rm = TRUE) + sd_weight * rmse_sd
	    }
	    data.frame(
	      operator_label = op$operator_label,
	      operator_alpha = stats::median(split_df$operator_alpha, na.rm = TRUE),
	      operator_complexity = op$complexity,
	      selection_metric = if (isTRUE(paired_active)) {
	        "delta_RMSE_vs_RF_robust_operator"
	      } else {
	        "RMSE_robust_operator"
	      },
	      selection_score = selection_score,
	      val_RMSE_median = stats::median(split_df$val_RMSE, na.rm = TRUE),
	      val_RMSE_mean = mean(split_df$val_RMSE, na.rm = TRUE),
	      val_RMSE_sd = rmse_sd,
	      val_RMSE_base_median = stats::median(split_df$val_RMSE_base, na.rm = TRUE),
	      val_RMSE_full_median = stats::median(split_df$val_RMSE_full, na.rm = TRUE),
	      val_RMSE_rf_median = stats::median(split_df$val_RMSE_rf, na.rm = TRUE),
	      delta_RMSE_vs_RF_median = stats::median(split_df$delta_RMSE_vs_RF, na.rm = TRUE),
	      delta_RMSE_vs_RF_sd = delta_sd,
	      val_RMSE_gain_vs_base_median = stats::median(split_df$val_RMSE_base - split_df$val_RMSE, na.rm = TRUE),
	      inner_splits = length(runs),
      selected = FALSE,
      stringsAsFactors = FALSE
    )
  })

  score_df <- bind_rows(rows)
  score_df$selection_score[!is.finite(score_df$selection_score)] <- Inf
  score_df$operator_safety_rank <- geoversa_operator_safety_rank(score_df$operator_label)
  selection <- geoversa_select_with_equivalence(
    score = score_df$selection_score,
    complexity = score_df$operator_complexity,
    equiv_abs = equiv_abs,
    equiv_rel = equiv_rel,
    safety_rank = score_df$operator_safety_rank,
    instability = score_df$val_RMSE_sd,
    gain = score_df$val_RMSE_gain_vs_base_median
	  )
	  selected_idx <- selection$selected_pos
	  noharm_applied <- FALSE
	  noharm_reason <- if (isTRUE(noharm_policy)) "eligible" else "disabled"
	  noharm_anchor_score <- NA_real_
	  noharm_anchor_label <- NA_character_
	  noharm_delta_vs_anchor <- NA_real_
	  noharm_margin <- NA_real_
	  base_idx <- which(score_df$operator_label == "base")[1L]
		  if (isTRUE(noharm_policy) && length(base_idx) == 1L &&
	      !is.na(base_idx) &&
	      is.finite(score_df$selection_score[base_idx])) {
	    noharm_anchor_score <- score_df$selection_score[base_idx]
	    noharm_anchor_label <- score_df$operator_label[base_idx]
	    anchor_rmse <- score_df$val_RMSE_median[base_idx]
	    if (!is.finite(anchor_rmse)) anchor_rmse <- abs(noharm_anchor_score)
	    if (!is.finite(anchor_rmse)) anchor_rmse <- 0
	    noharm_margin <- noharm_margin_abs +
	      noharm_margin_rel * abs(anchor_rmse)
	    if (!identical(score_df$operator_label[selected_idx], "base")) {
	      noharm_delta_vs_anchor <- noharm_anchor_score -
	        score_df$selection_score[selected_idx]
	      if (!is.finite(noharm_delta_vs_anchor) ||
	          noharm_delta_vs_anchor < noharm_margin) {
	        selected_idx <- base_idx
	        noharm_applied <- TRUE
	        noharm_reason <- sprintf(
	          "operator_gain_%.6f_below_margin_%.6f",
	          ifelse(is.finite(noharm_delta_vs_anchor),
	                 noharm_delta_vs_anchor, NA_real_),
	          noharm_margin
	        )
	      } else {
	        noharm_reason <- sprintf(
	          "operator_gain_%.6f_met_margin_%.6f",
	          noharm_delta_vs_anchor, noharm_margin
	        )
	      }
	    } else {
	      noharm_delta_vs_anchor <- 0
	      noharm_reason <- "selected_operator_is_base"
	    }
		  } else if (isTRUE(noharm_policy)) {
		    noharm_reason <- "no_base_anchor_available"
		  }
		  rf_noharm_policy <- isTRUE(rf_noharm_policy) &&
		    identical(tolower(as.character(selection_objective)[1L]),
		              "paired_rf_delta")
		  rf_noharm_preselected_idx <- selected_idx
		  rf_noharm_preselected_score <- score_df$selection_score[selected_idx]
		  rf_noharm_preselected_delta <-
		    score_df$delta_RMSE_vs_RF_median[selected_idx]
		  rf_noharm_anchor_score <- if (length(base_idx) == 1L && !is.na(base_idx)) {
		    score_df$selection_score[base_idx]
		  } else {
		    NA_real_
		  }
		  rf_noharm_anchor_delta <- if (length(base_idx) == 1L && !is.na(base_idx)) {
		    score_df$delta_RMSE_vs_RF_median[base_idx]
		  } else {
		    NA_real_
		  }
		  rf_noharm_margin <- NA_real_
		  rf_noharm_required_score <- NA_real_
		  rf_noharm_failed <- FALSE
		  rf_noharm_applied <- FALSE
		  rf_noharm_reason <- if (isTRUE(rf_noharm_policy)) {
		    "not_assessed"
		  } else {
		    "disabled"
		  }
		  if (isTRUE(rf_noharm_policy)) {
		    rf_scale <- score_df$val_RMSE_rf_median[selected_idx]
		    if (!is.finite(rf_scale)) rf_scale <- score_df$val_RMSE_median[selected_idx]
		    if (!is.finite(rf_scale)) rf_scale <- 0
		    rf_noharm_margin <- rf_noharm_margin_abs +
		      rf_noharm_margin_rel * abs(rf_scale)
		    rf_noharm_required_score <- -rf_noharm_margin
		    if (is.finite(rf_noharm_preselected_score)) {
		      if (rf_noharm_preselected_score <= rf_noharm_required_score) {
		        rf_noharm_reason <- sprintf(
		          "passed_rf_inner:score=%.6f<=required=%.6f",
		          rf_noharm_preselected_score, rf_noharm_required_score)
		      } else {
		        rf_noharm_failed <- TRUE
		        if (length(base_idx) == 1L && !is.na(base_idx) &&
		            is.finite(score_df$selection_score[base_idx])) {
		          if (!identical(selected_idx, base_idx)) {
		            selected_idx <- base_idx
		            rf_noharm_applied <- TRUE
		            rf_noharm_reason <- sprintf(
		              "fallback_to_base_operator:score=%.6f>required=%.6f",
		              rf_noharm_preselected_score, rf_noharm_required_score)
		          } else {
		            rf_noharm_reason <- sprintf(
		              "selected_base_still_fails_rf:score=%.6f>required=%.6f",
		              rf_noharm_preselected_score, rf_noharm_required_score)
		          }
		        } else {
		          rf_noharm_reason <- sprintf(
		            "rf_inner_failed_no_base_operator:score=%.6f>required=%.6f",
		            rf_noharm_preselected_score, rf_noharm_required_score)
		        }
		      }
		    } else {
		      rf_noharm_reason <- "no_valid_delta_RMSE_vs_RF"
		    }
		  }
		  score_df$selection_equiv_abs <- equiv_abs
	  score_df$selection_equiv_rel <- equiv_rel
	  score_df$selection_equiv_tol <- selection$equiv_tol
	  score_df$selection_practical_delta <- selection$practical_delta
	  score_df$selection_practically_equivalent <- selection$practically_equivalent
	  score_df$noharm_policy <- noharm_policy
	  score_df$noharm_anchor_operator <- noharm_anchor_label
	  score_df$noharm_anchor_score <- noharm_anchor_score
	  score_df$noharm_margin_abs <- noharm_margin_abs
	  score_df$noharm_margin_rel <- noharm_margin_rel
	  score_df$noharm_margin <- noharm_margin
		  score_df$noharm_delta_vs_anchor <- noharm_delta_vs_anchor
		  score_df$noharm_applied <- noharm_applied
		  score_df$noharm_reason <- noharm_reason
		  score_df$rf_noharm_policy <- rf_noharm_policy
		  score_df$rf_noharm_preselected_operator <-
		    score_df$operator_label[rf_noharm_preselected_idx]
		  score_df$rf_noharm_preselected_score <- rf_noharm_preselected_score
		  score_df$rf_noharm_preselected_delta_RMSE_vs_RF <-
		    rf_noharm_preselected_delta
		  score_df$rf_noharm_anchor_operator <-
		    if (length(base_idx) == 1L && !is.na(base_idx)) {
		      score_df$operator_label[base_idx]
		    } else NA_character_
		  score_df$rf_noharm_anchor_score <- rf_noharm_anchor_score
		  score_df$rf_noharm_anchor_delta_RMSE_vs_RF <- rf_noharm_anchor_delta
		  score_df$rf_noharm_margin_abs <- rf_noharm_margin_abs
		  score_df$rf_noharm_margin_rel <- rf_noharm_margin_rel
		  score_df$rf_noharm_margin <- rf_noharm_margin
		  score_df$rf_noharm_required_score <- rf_noharm_required_score
		  score_df$rf_noharm_failed <- rf_noharm_failed
		  score_df$rf_noharm_applied <- rf_noharm_applied
		  score_df$rf_noharm_reason <- rf_noharm_reason
		  score_df$selected[selected_idx] <- TRUE
  list(
    rows = score_df,
    selected = score_df[selected_idx, , drop = FALSE]
  )
}

rename_model_rows <- function(df, model_label) {
  if (is.null(df) || nrow(df) == 0L) {
    return(data.frame())
  }
  df$model <- model_label
  df
}

resolve_rf_model_label <- function(rf_refit_full_calibration,
                                   conformal_third_split = FALSE) {
  if (isTRUE(conformal_third_split)) {
    if (isTRUE(rf_refit_full_calibration)) "RF_train_plus_val" else "RF_train_core"
  } else {
    if (isTRUE(rf_refit_full_calibration)) "RF_500" else "RF_400"
  }
}

make_na_uq_row <- function(model_label) {
  data.frame(
    model = model_label,
    uq_available = FALSE,
    uq_nll = NA_real_,
    reliability_ece = NA_real_,
    picp_90 = NA_real_,
    mpiw_90 = NA_real_,
    ace_90 = NA_real_,
    picp_95 = NA_real_,
    mpiw_95 = NA_real_,
    ace_95 = NA_real_,
    stringsAsFactors = FALSE
  )
}

make_model_uq_eval <- function(obs,
                               pred_scaled,
                               logvar_scaled,
                               target_scaler,
                               target_transform,
                               model_label,
                               protocol,
                               split_id = 1L,
                               conformal_scores = NULL) {
  if (is.null(pred_scaled) ||
      is.null(logvar_scaled) ||
      is.null(target_scaler) ||
      !is.list(target_scaler) ||
      !all(c("center", "scale") %in% names(target_scaler))) {
    return(list(
      metrics = make_na_uq_row(model_label) %>% mutate(protocol = protocol),
      reliability = data.frame()
    ))
  }

  uq_row <- wadoux_eval_transformed_gaussian_uq(
    obs_raw = obs,
    pred_scaled = pred_scaled,
    logvar_scaled = logvar_scaled,
    target_scaler = target_scaler,
    target_transform = target_transform,
    conformal_scores = conformal_scores
  ) %>%
    mutate(model = model_label, protocol = protocol, .before = 1)

  rel_curve <- wadoux_transformed_gaussian_reliability_curve(
    obs_raw = obs,
    pred_scaled = pred_scaled,
    logvar_scaled = logvar_scaled,
    target_scaler = target_scaler,
    target_transform = target_transform,
    conformal_scores = conformal_scores
  )
  if (nrow(rel_curve) > 0L) {
    rel_curve <- rel_curve %>%
      mutate(model = model_label, protocol = protocol, split_id = split_id, .before = 1)
  }

  list(metrics = uq_row, reliability = rel_curve)
}

make_rf_uq_eval <- function(obs,
                            pred_mean,
                            pred_sd,
                            q_matrix,
                            q_levels,
                            model_label,
                            protocol,
                            split_id = 1L,
                            conformal_scores = NULL) {
  if (is.null(pred_mean) || is.null(pred_sd)) {
    return(list(
      metrics = make_na_uq_row(model_label) %>% mutate(protocol = protocol),
      reliability = data.frame()
    ))
  }

  uq_row <- wadoux_eval_rf_qrf_uq(
    obs_raw = obs,
    pred_mean = pred_mean,
    pred_sd = pred_sd,
    q_matrix = q_matrix,
    q_levels = q_levels,
    conformal_scores = conformal_scores
  ) %>%
    mutate(model = model_label, protocol = protocol, .before = 1)

  rel_curve <- wadoux_rf_qrf_reliability_curve(
    obs_raw = obs,
    pred_mean = pred_mean,
    pred_sd = pred_sd,
    q_matrix = q_matrix,
    q_levels = q_levels,
    conformal_scores = conformal_scores
  )
  if (nrow(rel_curve) > 0L) {
    rel_curve <- rel_curve %>%
      mutate(model = model_label, protocol = protocol, split_id = split_id, .before = 1)
  }

  list(metrics = uq_row, reliability = rel_curve)
}

make_uq_rows <- function(obs, preds, models, protocol, split_id = 1L,
                         val_obs = NULL,
                         calibration_obs = NULL) {
  rows <- list()
  rel_rows <- list()
  geoversa2_label <- if (!is.null(preds$geoversa2_model_label)) preds$geoversa2_model_label else "GeoVersa2-HyperLocal"
  rf_model_label <- if (!is.null(preds$rf_model_label)) preds$rf_model_label else "RF"
  rf_qrf_active <- tolower(trimws(
    Sys.getenv("WADOUX_RF_QRF", unset = "0")
  )) %in% c("1", "true", "yes")

  # Split-conformal calibration (opt-in via GEOVERSA_CONFORMAL_CALIBRATION=1).
  # Uses val set residuals to replace z_α with a data-driven conformal quantile,
  # guaranteeing marginal coverage P(Y ∈ interval) ≥ 1-α by construction.
  use_conformal <- tolower(trimws(
    Sys.getenv("GEOVERSA_CONFORMAL_CALIBRATION", unset = "0")
  )) %in% c("1", "true", "yes")

  conv_conformal_scores <- NULL
  conformal_obs <- if (!is.null(calibration_obs) &&
                       !is.null(preds$pred_conv_cal_scaled) &&
                       !is.null(preds$pred_conv_cal_logvar_scaled)) {
    calibration_obs
  } else {
    val_obs
  }
  conformal_pred_scaled <- if (!is.null(calibration_obs) &&
                               !is.null(preds$pred_conv_cal_scaled) &&
                               !is.null(preds$pred_conv_cal_logvar_scaled)) {
    preds$pred_conv_cal_scaled
  } else {
    preds$pred_conv_val_scaled
  }
  conformal_pred_logvar_scaled <- if (!is.null(calibration_obs) &&
                                      !is.null(preds$pred_conv_cal_scaled) &&
                                      !is.null(preds$pred_conv_cal_logvar_scaled)) {
    preds$pred_conv_cal_logvar_scaled
  } else {
    preds$pred_conv_val_logvar_scaled
  }

  if (use_conformal &&
      !is.null(conformal_obs) &&
      !is.null(conformal_pred_scaled) &&
      !is.null(conformal_pred_logvar_scaled) &&
      !is.null(preds$pred_conv_target_scaler)) {
    conv_conformal_scores <- tryCatch(
      compute_conformal_scores(
        obs_raw_val = conformal_obs,
        pred_val_scaled = conformal_pred_scaled,
        pred_val_logvar_scaled = conformal_pred_logvar_scaled,
        target_scaler = preds$pred_conv_target_scaler,
        target_transform = preds$pred_conv_target_transform
      ),
      error = function(e) {
        cat(sprintf("[Conformal] ⚠ score computation failed: %s\n",
                    conditionMessage(e)))
        NULL
      }
    )
  }

  rf_conformal_scores <- NULL
  rf_conformal_obs <- if (!is.null(calibration_obs) &&
                          !is.null(preds$pred_rf_cal_mean) &&
                          !is.null(preds$pred_rf_cal_sd)) {
    calibration_obs
  } else {
    val_obs
  }
  rf_conformal_mean <- if (!is.null(calibration_obs) &&
                           !is.null(preds$pred_rf_cal_mean) &&
                           !is.null(preds$pred_rf_cal_sd)) {
    preds$pred_rf_cal_mean
  } else {
    preds$pred_rf_val_mean
  }
  rf_conformal_sd <- if (!is.null(calibration_obs) &&
                         !is.null(preds$pred_rf_cal_mean) &&
                         !is.null(preds$pred_rf_cal_sd)) {
    preds$pred_rf_cal_sd
  } else {
    preds$pred_rf_val_sd
  }
  if (rf_qrf_active &&
      use_conformal &&
      !is.null(rf_conformal_obs) &&
      !is.null(rf_conformal_mean) &&
      !is.null(rf_conformal_sd)) {
    rf_conformal_scores <- tryCatch(
      compute_rf_conformal_scores(
        obs_cal = rf_conformal_obs,
        pred_cal_mean = rf_conformal_mean,
        pred_cal_sd = rf_conformal_sd
      ),
      error = function(e) {
        cat(sprintf("[RF conformal] ⚠ score computation failed: %s\n",
                    conditionMessage(e)))
        NULL
      }
    )
  }

  if ("RF" %in% models && !is.null(preds$pred_rf)) {
    if (rf_qrf_active &&
        !is.null(preds$pred_rf_mean) &&
        !is.null(preds$pred_rf_sd)) {
      rf_uq <- make_rf_uq_eval(
        obs = obs,
        pred_mean = preds$pred_rf_mean,
        pred_sd = preds$pred_rf_sd,
        q_matrix = preds$pred_rf_quantiles,
        q_levels = preds$pred_rf_quantile_levels,
        model_label = rf_model_label,
        protocol = protocol,
        split_id = split_id,
        conformal_scores = rf_conformal_scores
      )
      rows[[length(rows) + 1L]] <- rf_uq$metrics
      rel_rows[[length(rel_rows) + 1L]] <- rf_uq$reliability
    } else {
      rows[[length(rows) + 1L]] <- make_na_uq_row(rf_model_label) %>% mutate(protocol = protocol)
    }
  }
  if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
    conv_uq <- make_model_uq_eval(
      obs = obs,
      pred_scaled = preds$pred_conv_test_scaled,
      logvar_scaled = preds$pred_conv_test_logvar_scaled,
      target_scaler = preds$pred_conv_target_scaler,
      target_transform = preds$pred_conv_target_transform,
      model_label = "ConvKrigingNet2D",
      protocol = protocol,
      split_id = split_id,
      conformal_scores = conv_conformal_scores
    )
    rows[[length(rows) + 1L]] <- conv_uq$metrics
    rel_rows[[length(rel_rows) + 1L]] <- conv_uq$reliability
  }
  if (!is.null(preds$pred_geoversa2)) {
    geoversa2_uq <- make_model_uq_eval(
      obs = obs,
      pred_scaled = preds$pred_geoversa2_test_scaled,
      logvar_scaled = preds$pred_geoversa2_test_logvar_scaled,
      target_scaler = preds$pred_geoversa2_target_scaler,
      target_transform = preds$pred_geoversa2_target_transform,
      model_label = geoversa2_label,
      protocol = protocol,
      split_id = split_id
    )
    rows[[length(rows) + 1L]] <- geoversa2_uq$metrics
    rel_rows[[length(rel_rows) + 1L]] <- geoversa2_uq$reliability
  }

  list(
    metrics = bind_rows(rows),
    reliability = bind_rows(rel_rows)
  )
}

summarise_uq_results_by_protocol <- function(uq_df) {
  if (nrow(uq_df) == 0) {
    return(data.frame())
  }
  uq_df %>%
    group_by(scenario, protocol, model) %>%
    summarise(
      uq_available_rate = mean(uq_available, na.rm = TRUE),
      uq_nll_mean = mean(uq_nll, na.rm = TRUE),
      uq_nll_sd = sd(uq_nll, na.rm = TRUE),
      reliability_ece_mean = mean(reliability_ece, na.rm = TRUE),
      reliability_ece_sd = sd(reliability_ece, na.rm = TRUE),
      picp_90_mean = mean(picp_90, na.rm = TRUE),
      mpiw_90_mean = mean(mpiw_90, na.rm = TRUE),
      ace_90_mean = mean(ace_90, na.rm = TRUE),
      picp_95_mean = mean(picp_95, na.rm = TRUE),
      mpiw_95_mean = mean(mpiw_95, na.rm = TRUE),
      ace_95_mean = mean(ace_95, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(protocol, model)
}

summarise_reliability_by_protocol <- function(reliability_df) {
  if (nrow(reliability_df) == 0) {
    return(data.frame())
  }
  reliability_df %>%
    group_by(scenario, protocol, model, nominal_coverage) %>%
    summarise(
      empirical_coverage_mean = mean(empirical_coverage, na.rm = TRUE),
      empirical_coverage_sd = sd(empirical_coverage, na.rm = TRUE),
      abs_calibration_error_mean = mean(abs_calibration_error, na.rm = TRUE),
      mean_interval_width_mean = mean(mean_interval_width, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(protocol, model, nominal_coverage)
}

combine_normal_members_scaled <- function(mean_mat, logvar_mat, weights = NULL) {
  if (is.null(mean_mat) || !is.matrix(mean_mat) || nrow(mean_mat) < 1L) {
    return(list(mean = NULL, logvar = NULL))
  }
  if (is.null(weights)) {
    weights <- rep(1 / nrow(mean_mat), nrow(mean_mat))
  }
  weights <- as.numeric(weights)
  weights[!is.finite(weights) | weights < 0] <- 0
  if (sum(weights) <= 0) {
    weights <- rep(1 / nrow(mean_mat), nrow(mean_mat))
  } else {
    weights <- weights / sum(weights)
  }
  mean_mix <- as.numeric(colSums(mean_mat * weights))
  if (is.null(logvar_mat) ||
      !is.matrix(logvar_mat) ||
      !all(dim(logvar_mat) == dim(mean_mat)) ||
      any(!is.finite(logvar_mat))) {
    return(list(mean = mean_mix, logvar = rep(NA_real_, length(mean_mix))))
  }
  second_moment <- colSums((exp(logvar_mat) + mean_mat^2) * weights)
  var_mix <- pmax(as.numeric(second_moment - mean_mix^2), 1e-8)
  list(mean = mean_mix, logvar = log(var_mix))
}

extract_shared_uq_meta <- function(member_out) {
  member_out <- member_out[!vapply(member_out, is.null, logical(1))]
  if (length(member_out) < 1L) {
    return(NULL)
  }
  ref_scaler <- member_out[[1]]$target_scaler
  ref_transform <- member_out[[1]]$target_transform
  if (is.null(ref_scaler) ||
      !is.list(ref_scaler) ||
      !all(c("center", "scale") %in% names(ref_scaler)) ||
      length(ref_transform) < 1L) {
    return(NULL)
  }
  same_meta <- vapply(member_out, function(x) {
    scaler <- x$target_scaler
    transform <- x$target_transform
    !is.null(scaler) &&
      is.list(scaler) &&
      all(c("center", "scale") %in% names(scaler)) &&
      identical(as.character(transform[1]), as.character(ref_transform[1])) &&
      isTRUE(all.equal(as.numeric(scaler$center), as.numeric(ref_scaler$center), tolerance = 1e-8)) &&
      isTRUE(all.equal(as.numeric(scaler$scale), as.numeric(ref_scaler$scale), tolerance = 1e-8))
  }, logical(1))
  if (!all(same_meta)) {
    return(NULL)
  }
  list(target_scaler = ref_scaler, target_transform = ref_transform)
}

make_metric_rows_named <- function(obs, pred_map, protocol) {
  rows <- list()

  for (model_label in names(pred_map)) {
    pred <- pred_map[[model_label]]
    if (is.null(pred) || length(pred) == 0L) next
    rows[[length(rows) + 1L]] <- wadoux_eval(obs = obs, pred = pred) %>%
      mutate(model = model_label)
  }

  bind_rows(rows) %>% mutate(protocol = protocol)
}

make_paired_delta_rows <- function(metric_rows, protocol) {
  required_models <- c("RF_400", "GeoVersa_400", "RF_500", "GeoVersa_500")
  present_models <- unique(metric_rows$model)
  if (!all(required_models %in% present_models)) {
    return(data.frame())
  }

  metric_lookup <- split(metric_rows, metric_rows$model)
  comparisons <- list(
    list(lhs = "GeoVersa_400", rhs = "RF_400"),
    list(lhs = "GeoVersa_500", rhs = "RF_500"),
    list(lhs = "GeoVersa_500", rhs = "GeoVersa_400"),
    list(lhs = "RF_500", rhs = "RF_400")
  )

  rows <- lapply(comparisons, function(comp) {
    lhs <- metric_lookup[[comp$lhs]][1, , drop = FALSE]
    rhs <- metric_lookup[[comp$rhs]][1, , drop = FALSE]
    data.frame(
      protocol = protocol,
      comparison = sprintf("%s_vs_%s", comp$lhs, comp$rhs),
      lhs_model = comp$lhs,
      rhs_model = comp$rhs,
      lhs_ME = lhs$ME,
      lhs_RMSE = lhs$RMSE,
      lhs_r2 = lhs$r2,
      lhs_MEC = lhs$MEC,
      rhs_ME = rhs$ME,
      rhs_RMSE = rhs$RMSE,
      rhs_r2 = rhs$r2,
      rhs_MEC = rhs$MEC,
      delta_ME = lhs$ME - rhs$ME,
      delta_RMSE = lhs$RMSE - rhs$RMSE,
      delta_r2 = lhs$r2 - rhs$r2,
      delta_MEC = lhs$MEC - rhs$MEC,
      lhs_better_rmse = lhs$RMSE < rhs$RMSE,
      lhs_better_r2 = lhs$r2 > rhs$r2,
      lhs_better_mec = lhs$MEC > rhs$MEC,
      stringsAsFactors = FALSE
    )
  })

  bind_rows(rows)
}

bind_patch_arrays_4d <- function(arrays) {
  arrays <- arrays[!vapply(arrays, is.null, logical(1))]
  if (length(arrays) == 0) {
    return(NULL)
  }
  if (length(arrays) == 1) {
    return(arrays[[1]])
  }

  ref_dim <- dim(arrays[[1]])
  if (length(ref_dim) != 4) {
    stop("Expected 4D patch arrays.")
  }
  for (i in 2:length(arrays)) {
    d <- dim(arrays[[i]])
    if (length(d) != 4 || !all(d[1:3] == ref_dim[1:3])) {
      stop("All patch arrays must agree in channels, height, and width.")
    }
  }

  n_total <- sum(vapply(arrays, function(a) dim(a)[4], numeric(1)))
  out <- array(NA_real_, dim = c(ref_dim[1], ref_dim[2], ref_dim[3], n_total))
  pos <- 1L
  for (arr in arrays) {
    n_i <- dim(arr)[4]
    out[, , , pos:(pos + n_i - 1L)] <- arr
    pos <- pos + n_i
  }
  out
}

load_convkrigingnet2d_env <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop(
      "The 'torch' package is not available in this R library. ",
      "Run this benchmark from the same R environment used to train ConvKrigingNet2D, ",
      "or install torch for this R installation before executing this script."
    )
  }

  loader_parent <- new.env(parent = globalenv())
  env <- new.env(parent = loader_parent)

  loader_parent$source <- function(file, ...) {
    is_absolute <- grepl("^(?:[A-Za-z]:[\\\\/]|/)", file)
    resolved <- if (is_absolute) {
      file
    } else {
      file.path(project_root_wadoux, sub("^\\./", "", file))
    }
    sys.source(resolved, envir = env)
    invisible(NULL)
  }

  sys.source(file.path(project_root_wadoux, "code", "ConvKrigingNet2D.R"), envir = env)
  env
}

make_inner_val_split_wadoux <- function(train_df, val_frac = 0.2, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n_train <- nrow(train_df)
  if (n_train < 10) stop("Outer training set too small for fair comparison.")
  val_size <- max(1L, floor(n_train * val_frac))
  val_idx <- sample.int(n_train, size = val_size, replace = FALSE)
  train_idx <- setdiff(seq_len(n_train), val_idx)
  list(
    train = train_df[train_idx, , drop = FALSE],
    val = train_df[val_idx, , drop = FALSE]
  )
}

make_inner_val_cal_split_wadoux <- function(train_df,
                                            val_frac = 0.2,
                                            cal_frac = val_frac,
                                            seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n_train <- nrow(train_df)
  if (n_train < 15L) stop("Outer training set too small for three-way comparison.")
  val_size <- max(1L, floor(n_train * val_frac))
  cal_size <- max(1L, floor(n_train * cal_frac))
  if ((val_size + cal_size) >= n_train) {
    stop("Three-way split invalid: val_size + cal_size must leave at least one training row.")
  }
  perm <- sample.int(n_train, size = n_train, replace = FALSE)
  val_idx <- perm[seq_len(val_size)]
  cal_idx <- perm[val_size + seq_len(cal_size)]
  train_idx <- setdiff(seq_len(n_train), c(val_idx, cal_idx))
  list(
    train = train_df[train_idx, , drop = FALSE],
    val = train_df[val_idx, , drop = FALSE],
    cal = train_df[cal_idx, , drop = FALSE]
  )
}

make_inner_protocol_split_wadoux <- function(train_df,
                                             val_frac = 0.2,
                                             conformal_third_split = FALSE,
                                             conformal_cal_frac = val_frac,
                                             seed = NULL) {
  if (isTRUE(conformal_third_split)) {
    inner <- make_inner_val_cal_split_wadoux(
      train_df = train_df,
      val_frac = val_frac,
      cal_frac = conformal_cal_frac,
      seed = seed
    )
  } else {
    inner <- make_inner_val_split_wadoux(train_df, val_frac = val_frac, seed = seed)
    inner$cal <- NULL
  }
  inner
}

singleton_df <- function(df) {
  if (nrow(df) < 1L) {
    stop("singleton_df requires at least one row.")
  }
  df[1L, , drop = FALSE]
}

coerce_single_diag_row <- function(diag_df, stage_label) {
  if (is.null(diag_df)) {
    stop(stage_label, " did not return diagnostics.")
  }
  if (!is.data.frame(diag_df)) {
    diag_df <- as.data.frame(diag_df, stringsAsFactors = FALSE)
  }
  if (nrow(diag_df) < 1L) {
    stop(stage_label, " returned an empty diagnostics table.")
  }
  if (nrow(diag_df) > 1L) {
    diag_unique <- unique(diag_df)
    if (nrow(diag_unique) == 1L) {
      cat(sprintf("[GeoVersa refit] %s returned %d duplicated diagnostics rows; collapsing to one.\n",
                  stage_label, nrow(diag_df)))
      diag_df <- diag_unique
    } else {
      cat(sprintf("[GeoVersa refit] WARN: %s returned %d diagnostics rows; using the first row for refit scheduling.\n",
                  stage_label, nrow(diag_df)))
      diag_df <- diag_df[1L, , drop = FALSE]
    }
  }
  diag_df[1L, , drop = FALSE]
}

bind_diag_columns_unique <- function(lhs, rhs, rhs_suffix = "_runner") {
  if (is.null(lhs) || nrow(lhs) == 0L) return(rhs)
  if (is.null(rhs) || nrow(rhs) == 0L) return(lhs)
  lhs_names <- names(lhs)
  rhs_names <- names(rhs)
  overlap <- intersect(lhs_names, rhs_names)
  if (length(overlap) > 0L) {
    for (nm in overlap) {
      idx <- which(rhs_names == nm)
      base_nm <- paste0(nm, rhs_suffix)
      rhs_names[idx] <- make.unique(rep(base_nm, length(idx)), sep = "_")
    }
    names(rhs) <- rhs_names
  }
  cbind(lhs, rhs)
}

choose_smoothed_refit_main_epochs <- function(selection_diag, n_refit_train) {
  best_main <- as.integer(selection_diag$best_epoch_main[1])
  main_done <- as.integer(selection_diag$main_epochs_done[1])
  lr_patience <- as.integer(selection_diag$lr_patience[1])
  n_select_train <- as.integer(selection_diag$n_train[1])
  plateau_median <- suppressWarnings(as.integer(selection_diag$plateau_median_epoch_main[1]))
  plateau_last <- suppressWarnings(as.integer(selection_diag$plateau_last_epoch_main[1]))

  if (!is.finite(best_main) || best_main < 1L) {
    best_main <- main_done
  }
  if (!is.finite(main_done) || main_done < best_main) {
    main_done <- best_main
  }
  if (!is.finite(lr_patience) || lr_patience < 0L) {
    lr_patience <- 0L
  }
  if (!is.finite(n_select_train) || n_select_train < 1L) {
    n_select_train <- n_refit_train
  }
  if (!is.finite(plateau_median) || plateau_median < best_main) {
    plateau_median <- best_main
  }
  if (!is.finite(plateau_last) || plateau_last < plateau_median) {
    plateau_last <- plateau_median
  }

  epoch_floor <- as.integer(round(parse_num_env("WADOUX_REFIT_MAIN_EPOCH_FLOOR", 8)))
  trail_frac <- parse_num_env("WADOUX_REFIT_MAIN_TRAIL_FRAC", 0.60)
  lr_buffer_frac <- parse_num_env("WADOUX_REFIT_MAIN_LR_BUFFER_FRAC", 0.50)
  plateau_blend <- parse_num_env("WADOUX_REFIT_MAIN_PLATEAU_BLEND", 0.75)
  selection_floor_frac <- parse_num_env("WADOUX_REFIT_MAIN_SELECTION_FLOOR_FRAC", 0.60)
  plateau_blend <- max(0.0, min(1.0, plateau_blend))
  selection_floor_frac <- max(0.0, min(1.0, selection_floor_frac))

  size_scale <- sqrt(max(1, n_refit_train) / max(1, n_select_train))
  plateau_anchor <- as.numeric(best_main) + plateau_blend * (as.numeric(plateau_last) - as.numeric(best_main))
  scaled_best <- plateau_anchor * size_scale
  buffered_best <- scaled_best + max(1, ceiling(lr_patience * lr_buffer_frac))
  trailing_anchor <- trail_frac * main_done
  adaptive_epoch_floor <- max(epoch_floor, ceiling(main_done * selection_floor_frac))

  refit_main_epochs <- as.integer(round(min(
    main_done,
    max(adaptive_epoch_floor, buffered_best, trailing_anchor, plateau_median)
  )))

  list(
    refit_main_epochs = refit_main_epochs,
    raw_best_epoch_main = best_main,
    plateau_median_epoch_main = plateau_median,
    plateau_last_epoch_main = plateau_last,
    plateau_anchor_epoch = plateau_anchor,
    scaled_best_epoch_main = scaled_best,
    buffered_best_epoch_main = buffered_best,
    trailing_anchor_epoch = trailing_anchor,
    refit_epoch_floor = adaptive_epoch_floor,
    refit_epoch_floor_requested = epoch_floor,
    refit_selection_floor_frac = selection_floor_frac,
    refit_size_scale = size_scale,
    refit_n_select_train = n_select_train,
    refit_n_full_train = n_refit_train
  )
}

choose_adaptive_warmstart_policy <- function(selection_diag,
                                             selection_val_rmse_gain,
                                             base_epoch_mult,
                                             base_lr_mult,
                                             scheduled_main_epochs) {
  best_main <- suppressWarnings(as.integer(selection_diag$best_epoch_main[1]))
  main_done <- suppressWarnings(as.integer(selection_diag$main_epochs_done[1]))
  lr_patience <- suppressWarnings(as.integer(selection_diag$lr_patience[1]))
  plateau_median <- suppressWarnings(as.integer(selection_diag$plateau_median_epoch_main[1]))
  plateau_last <- suppressWarnings(as.integer(selection_diag$plateau_last_epoch_main[1]))
  nugget_ratio <- suppressWarnings(as.numeric(selection_diag$nugget_ratio[1]))

  if (!is.finite(best_main) || best_main < 1L) {
    best_main <- 1L
  }
  if (!is.finite(main_done) || main_done < best_main) {
    main_done <- best_main
  }
  if (!is.finite(lr_patience) || lr_patience < 0L) {
    lr_patience <- 0L
  }
  if (!is.finite(plateau_median) || plateau_median < 1L) {
    plateau_median <- best_main
  }
  if (!is.finite(plateau_last) || plateau_last < plateau_median) {
    plateau_last <- plateau_median
  }
  if (!is.finite(nugget_ratio)) {
    nugget_ratio <- 0.25
  }
  if (!is.finite(selection_val_rmse_gain)) {
    selection_val_rmse_gain <- 0
  }

  policy <- list(
    regime = "baseline",
    epoch_mult = base_epoch_mult,
    lr_mult = base_lr_mult,
    min_main_epochs = NA_integer_
  )
  noisy_epoch_mult <- parse_num_env("WADOUX_REFIT_NOISY_EPOCH_MULT", 0.65)
  noisy_lr_mult <- parse_num_env("WADOUX_REFIT_NOISY_LR_MULT", 0.20)
  noisy_min_main <- as.integer(round(parse_num_env("WADOUX_REFIT_NOISY_MIN_MAIN_EPOCHS", 8)))
  noisy_min_frac <- parse_num_env("WADOUX_REFIT_NOISY_MIN_MAIN_FRAC", 0.60)
  if (!is.finite(noisy_epoch_mult) || noisy_epoch_mult <= 0) noisy_epoch_mult <- 0.65
  if (!is.finite(noisy_lr_mult) || noisy_lr_mult <= 0) noisy_lr_mult <- 0.20
  if (!is.finite(noisy_min_main) || noisy_min_main < 1L) noisy_min_main <- 8L
  if (!is.finite(noisy_min_frac) || noisy_min_frac < 0) noisy_min_frac <- 0.60
  noisy_epoch_mult <- max(0.20, min(1.0, noisy_epoch_mult))
  noisy_lr_mult <- max(0.05, min(1.0, noisy_lr_mult))
  noisy_min_frac <- max(0.0, min(1.0, noisy_min_frac))

  short_plateau <- best_main <= max(8L, floor(plateau_median * 0.75))
  undertrained_backbone <- (nugget_ratio <= 0.26) &&
    (selection_val_rmse_gain >= 0.35) && (
      short_plateau ||
        scheduled_main_epochs <= max(8L, plateau_median - 2L) ||
        (main_done >= 15L && plateau_last <= max(6L, floor(plateau_median * 0.75)))
    )

  conservative_noisy <- (nugget_ratio >= 0.28) &&
    (selection_val_rmse_gain <= 0.35)

  if (undertrained_backbone) {
    policy$regime <- "backbone_recovery"
    policy$epoch_mult <- max(base_epoch_mult, 0.70)
    policy$lr_mult <- min(base_lr_mult, 0.20)
    policy$min_main_epochs <- max(
      8L,
      min(main_done, max(best_main, plateau_median, plateau_last, 8L))
    )
  } else if (conservative_noisy) {
    policy$regime <- "noisy_conservative"
    policy$epoch_mult <- max(base_epoch_mult, noisy_epoch_mult)
    policy$lr_mult <- min(base_lr_mult, noisy_lr_mult)
    policy$min_main_epochs <- max(
      noisy_min_main,
      min(
        main_done,
        max(
          noisy_min_main,
          ceiling(main_done * noisy_min_frac),
          floor(plateau_median * 0.90),
          best_main + max(1L, lr_patience)
        )
      )
    )
  }

  policy
}

aggregate_refit_schedule <- function(selection_diags, n_refit_train) {
  if (!is.list(selection_diags) || length(selection_diags) < 1L) {
    stop("aggregate_refit_schedule requires at least one selection diagnostics row.")
  }

  schedules <- lapply(selection_diags, function(diag_row) {
    choose_smoothed_refit_main_epochs(
      selection_diag = diag_row,
      n_refit_train = n_refit_train
    )
  })

  int_median <- function(x, floor = 1L) {
    val <- as.integer(round(stats::median(as.numeric(x), na.rm = TRUE)))
    if (!is.finite(val) || val < floor) floor else val
  }
  num_median <- function(x, fallback) {
    val <- stats::median(as.numeric(x), na.rm = TRUE)
    if (!is.finite(val)) fallback else as.numeric(val)
  }

  main_epochs <- int_median(vapply(schedules, `[[`, numeric(1), "refit_main_epochs"), floor = 1L)
  warmup_epochs <- int_median(vapply(selection_diags, function(x) x$warmup_epochs_done[1], numeric(1)), floor = 1L)
  lr_patience <- int_median(vapply(selection_diags, function(x) x$lr_patience[1], numeric(1)), floor = 1L)
  patience <- int_median(vapply(selection_diags, function(x) x$patience[1], numeric(1)), floor = 1L)
  bank_refresh_every <- int_median(vapply(selection_diags, function(x) x$bank_refresh_every[1], numeric(1)), floor = 1L)
  lr_decay <- num_median(vapply(selection_diags, function(x) x$lr_decay[1], numeric(1)), fallback = 0.5)

  per_split <- data.frame(
    split_id = seq_along(selection_diags),
    warmup_epochs_done = vapply(selection_diags, function(x) x$warmup_epochs_done[1], numeric(1)),
    patience = vapply(selection_diags, function(x) x$patience[1], numeric(1)),
    lr_patience = vapply(selection_diags, function(x) x$lr_patience[1], numeric(1)),
    lr_decay = vapply(selection_diags, function(x) x$lr_decay[1], numeric(1)),
    bank_refresh_every = vapply(selection_diags, function(x) x$bank_refresh_every[1], numeric(1)),
    raw_best_epoch_main = vapply(schedules, `[[`, numeric(1), "raw_best_epoch_main"),
    plateau_median_epoch_main = vapply(schedules, `[[`, numeric(1), "plateau_median_epoch_main"),
    plateau_last_epoch_main = vapply(schedules, `[[`, numeric(1), "plateau_last_epoch_main"),
    plateau_anchor_epoch = vapply(schedules, `[[`, numeric(1), "plateau_anchor_epoch"),
    scaled_best_epoch_main = vapply(schedules, `[[`, numeric(1), "scaled_best_epoch_main"),
    buffered_best_epoch_main = vapply(schedules, `[[`, numeric(1), "buffered_best_epoch_main"),
    trailing_anchor_epoch = vapply(schedules, `[[`, numeric(1), "trailing_anchor_epoch"),
    refit_epoch_floor = vapply(schedules, `[[`, numeric(1), "refit_epoch_floor"),
    refit_epoch_floor_requested = vapply(schedules, `[[`, numeric(1), "refit_epoch_floor_requested"),
    refit_selection_floor_frac = vapply(schedules, `[[`, numeric(1), "refit_selection_floor_frac"),
    refit_selected_main_epochs = vapply(schedules, `[[`, numeric(1), "refit_main_epochs"),
    stringsAsFactors = FALSE
  )

  list(
    refit_main_epochs = main_epochs,
    refit_warmup_epochs = warmup_epochs,
    refit_lr_patience = lr_patience,
    refit_patience = max(main_epochs, patience),
    refit_lr_decay = lr_decay,
    refit_bank_refresh_every = bank_refresh_every,
    n_selection_splits = length(selection_diags),
    per_split = per_split,
    raw_best_epoch_main = stats::median(per_split$raw_best_epoch_main, na.rm = TRUE),
    plateau_median_epoch_main = stats::median(per_split$plateau_median_epoch_main, na.rm = TRUE),
    plateau_last_epoch_main = stats::median(per_split$plateau_last_epoch_main, na.rm = TRUE),
    plateau_anchor_epoch = stats::median(per_split$plateau_anchor_epoch, na.rm = TRUE),
    scaled_best_epoch_main = stats::median(per_split$scaled_best_epoch_main, na.rm = TRUE),
    buffered_best_epoch_main = stats::median(per_split$buffered_best_epoch_main, na.rm = TRUE),
    trailing_anchor_epoch = stats::median(per_split$trailing_anchor_epoch, na.rm = TRUE),
    refit_epoch_floor = stats::median(vapply(schedules, `[[`, numeric(1), "refit_epoch_floor"), na.rm = TRUE),
    refit_epoch_floor_requested = stats::median(vapply(schedules, `[[`, numeric(1), "refit_epoch_floor_requested"), na.rm = TRUE),
    refit_selection_floor_frac = stats::median(vapply(schedules, `[[`, numeric(1), "refit_selection_floor_frac"), na.rm = TRUE),
    refit_size_scale = stats::median(vapply(schedules, `[[`, numeric(1), "refit_size_scale"), na.rm = TRUE)
  )
}

get_conv_params_profile <- function(conv_env, profile = c("quick", "full", "n500", "auto"), train_seed = 123L, device = "cpu") {
  profile <- match.arg(profile)
  if (profile == "auto") {
    params <- list(
      kriging_mode = "anisotropic",
      train_seed = train_seed,
      deterministic_batches = TRUE,
      device = device,
      epochs = 80L
    )
  } else if (profile == "quick") {
    params <- modifyList(
      conv_env$convkriging2d_quick_params,
      list(
        kriging_mode = "anisotropic",
        train_seed = train_seed,
        deterministic_batches = TRUE,
        device = device
      )
    )
  } else if (profile == "n500") {
    # Targeted fixes for n=500 Wadoux benchmark.
    # Base architecture unchanged from multiseed confirmation (baseline_params).
    # Four targeted tweaks for n=500:
    #   1. warmup_epochs=8  : train backbone first; bank residuals are meaningful
    #                         before the kriging layer starts learning ell/theta
    #   2. base_loss_weight=0.05 : small auxiliary backbone loss; ensures the DNN
    #                         gets direct gradient signal even when kriging dominates
    #   3. lr_decay=0.5, lr_patience=4, min_lr=5e-5 : reduce LR when stuck
    #                         instead of stopping; patience=15 gives more room
    #   4. batch_size=48    : ~400 train pts / 48 = 8 batches/epoch (vs 4 with 96)
    #                         doubles gradient steps per epoch
    params <- modifyList(
      conv_env$convkriging2d_baseline_params,
      list(
        kriging_mode       = "anisotropic",
        train_seed         = train_seed,
        deterministic_batches = TRUE,
        device             = device,
        warmup_epochs      = 8L,
        # beta_init=0 -> sigmoid(0)=0.5: model uses kriging from the start.
        # Safe because warmup already gave the base model 8 epochs.
        # Default beta_init=-4 -> sigmoid(-4)=0.018 was effectively killing kriging.
        # beta_init=0 -> sigmoid(0)=0.5: model uses kriging from the start.
        # krig_loss_weight removed: direct delta supervision introduced +5 Mg/ha
        # bias in DesignBased because it memorizes training residual direction.
        beta_init          = 0,
        base_loss_weight   = 0.05,
        batch_size         = 48L,
        lr_decay           = 0.5,
        lr_patience        = 4L,
        min_lr             = 5e-5,
        patience           = 15L,
        epochs             = 80L
        # dist_scale = NULL, krig_dropout = 0 (both disabled — defaults).
        #
        # Distance-aware gate tested across 4 variants (all 1-iteration):
        #   baseline (no gate):              DesignBased RMSE=32.8, ME=0,   Spatial=37.0
        #   gate=1.0 train+eval:             DesignBased RMSE=35.5, ME=3.1, Spatial=38.3
        #   gate=0.5 eval-only rel:          DesignBased RMSE=35.7, ME=5.3, Spatial=38.3
        #   krig_dropout=0.3 + gate=0.5:    DesignBased RMSE=35.8, ME=3.2, Spatial=39.0
        #
        # All gate variants degraded DesignBased and did not improve SpatialKFold.
        # Root cause: base + kriging are co-trained; the base learns to rely on
        # β×δ to correct its positive bias. Any eval-time suppression of kriging
        # exposes this bias (ME > 0 → higher RMSE). krig_dropout at 0.3 is
        # insufficient to break the dependency; would need ~0.7+ which would
        # prevent the kriging layer from learning. Requires separate architectural
        # treatment (future work).
        #
        # Historical note: earlier exploratory runs reported DesignBased
        # RMSE near the Wadoux RF baseline. Treat the CSV outputs written by
        # this runner as the source of truth for any current claim.
      )
    )
  } else {
    # "full" uses the same params as the multiseed confirmation:
    # convkriging2d_baseline_params + kriging_mode="anisotropic".
    # convkriging2d_anisotropic_stable_params was tested and performed worse.
    params <- modifyList(
      conv_env$convkriging2d_baseline_params,
      list(
        kriging_mode = "anisotropic",
        train_seed = train_seed,
        deterministic_batches = TRUE,
        device = device
      )
    )
  }
  params
}

compare_rf_conv_on_explicit_split <- function(train_core,
                                              val_df,
                                              cal_df = NULL,
                                              test_df,
                                              models = c("RF", "ConvKrigingNet2D"),
                                              calibration_patch_cache = NULL,
                                              conv_env,
                                              conv_params,
                                              patch_size = 15,
                                              context = NULL,
                                              refit_full_calibration = FALSE,
                                              rf_refit_full_calibration = refit_full_calibration,
                                              full_calibration_df = NULL,
                                              selection_seed = NULL,
                                              policy_iteration = NA_integer_,
                                              policy_protocol = NA_character_,
                                              policy_model = NA_character_) {
  has_conformal_cal_split <- !is.null(cal_df) && is.data.frame(cal_df) && nrow(cal_df) > 0L
  n_cal_rows <- if (has_conformal_cal_split) nrow(cal_df) else 0L
  n_test_rows <- nrow(test_df)
  eval_test_df <- if (has_conformal_cal_split) bind_rows(cal_df, test_df) else test_df

  split_eval_outputs <- function(model_out) {
    if (!has_conformal_cal_split || is.null(model_out)) {
      return(list(
        pred_test = model_out$pred_test,
        pred_test_base = model_out$pred_test_base,
        pred_test_scaled = model_out$pred_test_scaled,
        pred_test_logvar_scaled = model_out$pred_test_logvar_scaled,
        pred_cal_scaled = NULL,
        pred_cal_logvar_scaled = NULL
      ))
    }
    total_n <- n_cal_rows + n_test_rows
    split_vec <- function(x, take = c("test", "cal")) {
      take <- match.arg(take)
      if (is.null(x)) return(NULL)
      x <- as.numeric(x)
      if (length(x) != total_n) {
        stop(sprintf(
          "Expected %d predictions for conformal split (cal=%d, test=%d) but got %d.",
          total_n, n_cal_rows, n_test_rows, length(x)
        ))
      }
      if (identical(take, "cal")) {
        x[seq_len(n_cal_rows)]
      } else {
        x[n_cal_rows + seq_len(n_test_rows)]
      }
    }
    list(
      pred_test = split_vec(model_out$pred_test, "test"),
      pred_test_base = split_vec(model_out$pred_test_base, "test"),
      pred_test_scaled = split_vec(model_out$pred_test_scaled, "test"),
      pred_test_logvar_scaled = split_vec(model_out$pred_test_logvar_scaled, "test"),
      pred_cal_scaled = split_vec(model_out$pred_test_scaled, "cal"),
      pred_cal_logvar_scaled = split_vec(model_out$pred_test_logvar_scaled, "cal")
    )
  }

  split_rf_eval_outputs <- function(rf_uq_out) {
    if (!has_conformal_cal_split || is.null(rf_uq_out)) {
      return(list(
        pred_test_mean = rf_uq_out$mean,
        pred_test_sd = rf_uq_out$sd,
        pred_test_q = rf_uq_out$q_matrix,
        pred_cal_mean = NULL,
        pred_cal_sd = NULL
      ))
    }
    total_n <- n_cal_rows + n_test_rows
    split_vec <- function(x, take = c("test", "cal")) {
      take <- match.arg(take)
      if (is.null(x)) return(NULL)
      x <- as.numeric(x)
      if (length(x) != total_n) {
        stop(sprintf(
          "Expected %d RF-UQ values for conformal split (cal=%d, test=%d) but got %d.",
          total_n, n_cal_rows, n_test_rows, length(x)
        ))
      }
      if (identical(take, "cal")) {
        x[seq_len(n_cal_rows)]
      } else {
        x[n_cal_rows + seq_len(n_test_rows)]
      }
    }
    split_mat <- function(x, take = c("test", "cal")) {
      take <- match.arg(take)
      if (is.null(x)) return(NULL)
      x <- as.matrix(x)
      if (nrow(x) != total_n) {
        stop(sprintf(
          "Expected %d RF-QRF rows for conformal split (cal=%d, test=%d) but got %d.",
          total_n, n_cal_rows, n_test_rows, nrow(x)
        ))
      }
      if (identical(take, "cal")) {
        x[seq_len(n_cal_rows), , drop = FALSE]
      } else {
        x[n_cal_rows + seq_len(n_test_rows), , drop = FALSE]
      }
    }
    list(
      pred_test_mean = split_vec(rf_uq_out$mean, "test"),
      pred_test_sd = split_vec(rf_uq_out$sd, "test"),
      pred_test_q = split_mat(rf_uq_out$q_matrix, "test"),
      pred_cal_mean = split_vec(rf_uq_out$mean, "cal"),
      pred_cal_sd = split_vec(rf_uq_out$sd, "cal")
    )
  }

  build_canonical_preprocess_bundle <- function(calibration_df) {
    if (!isTRUE(refit_full_calibration) ||
        is.null(conv_env) ||
        !is.function(conv_env$fit_standard_scaler) ||
        nrow(calibration_df) < 2L) {
      return(NULL)
    }
    qbasis_knots <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_AUTO_TAB_QBASIS_KNOTS", unset = "4")))
    if (!is.finite(qbasis_knots)) qbasis_knots <- 4
    qbasis_knots <- as.integer(max(2L, min(6L, round(qbasis_knots))))
    bundle <- list(
      source_n = as.integer(nrow(calibration_df)),
      coord_scaler = conv_env$fit_standard_scaler(as.matrix(calibration_df[, c("x", "y"), drop = FALSE])),
      tab_qbasis_thresholds = NULL,
      tab_qbasis_bandwidths = NULL,
      tab_qbasis_by_knots = list()
    )
    if (!is.null(conv_env$.auto_quantile_basis_stats) &&
        is.function(conv_env$.auto_quantile_basis_stats)) {
      qbasis_x <- as.matrix(calibration_df[, predList_modelfull_wadoux, drop = FALSE])
      qbasis_knots_grid <- sort(unique(as.integer(c(2:6, qbasis_knots))))
      for (k in qbasis_knots_grid) {
        qbasis_stats <- conv_env$.auto_quantile_basis_stats(qbasis_x, n_knots = k)
        bundle$tab_qbasis_by_knots[[as.character(k)]] <- list(
          thresholds = qbasis_stats$thresholds,
          bandwidths = qbasis_stats$bandwidths
        )
      }
      qbasis_default <- bundle$tab_qbasis_by_knots[[as.character(qbasis_knots)]]
      if (!is.null(qbasis_default)) {
        bundle$tab_qbasis_thresholds <- qbasis_default$thresholds
        bundle$tab_qbasis_bandwidths <- qbasis_default$bandwidths
      }
    }
    bundle
  }

  compute_target_scaler_for_training_df <- function(train_df) {
    if (is.null(conv_env) ||
        is.null(conv_env$transform_target) ||
        !is.function(conv_env$transform_target) ||
        is.null(conv_env$fit_target_scaler) ||
        !is.function(conv_env$fit_target_scaler)) {
      return(NULL)
    }
    target_transform <- if (!is.null(conv_params$target_transform)) conv_params$target_transform else "identity"
    y_train_t <- conv_env$transform_target(train_df[[response_name_wadoux]], target_transform)
    conv_env$fit_target_scaler(y_train_t)
  }

  transport_warmstart_head_state <- function(state_dict, selection_scaler, refit_scaler) {
    if (is.null(state_dict) || is.null(selection_scaler) || is.null(refit_scaler)) {
      return(list(
        state = state_dict,
        active = FALSE,
        ratio = NA_real_,
        offset = NA_real_,
        head_count = 0L,
        logvar_offset = NA_real_,
        logvar_count = 0L
      ))
    }
    sel_scale <- suppressWarnings(as.numeric(selection_scaler$scale))
    ref_scale <- suppressWarnings(as.numeric(refit_scaler$scale))
    sel_center <- suppressWarnings(as.numeric(selection_scaler$center))
    ref_center <- suppressWarnings(as.numeric(refit_scaler$center))
    if (!is.finite(sel_scale) || !is.finite(ref_scale) || ref_scale == 0 ||
        !is.finite(sel_center) || !is.finite(ref_center)) {
      return(list(
        state = state_dict,
        active = FALSE,
        ratio = NA_real_,
        offset = NA_real_,
        head_count = 0L,
        logvar_offset = NA_real_,
        logvar_count = 0L
      ))
    }
    ratio <- as.numeric(sel_scale / ref_scale)
    offset <- as.numeric((sel_center - ref_center) / ref_scale)
    state_out <- if (!is.null(conv_env$clone_state_dict) && is.function(conv_env$clone_state_dict)) {
      conv_env$clone_state_dict(state_dict)
    } else {
      lapply(state_dict, function(x) x$clone())
    }
    mean_bias_keys <- grep(
      "^(head\\.net|head\\.mean_net|tab_rs_head_[0-9]+\\.net)\\.4\\.bias$",
      names(state_out),
      value = TRUE
    )
    head_count <- 0L
    for (bias_key in mean_bias_keys) {
      weight_key <- sub("\\.bias$", ".weight", bias_key)
      if (!weight_key %in% names(state_out)) next
      state_out[[weight_key]] <- state_out[[weight_key]] * ratio
      state_out[[bias_key]] <- state_out[[bias_key]] * ratio + offset
      head_count <- head_count + 1L
    }
    logvar_offset <- 2 * log(abs(ratio))
    logvar_count <- 0L
    logvar_bias_keys <- grep("^head\\.logvar_net\\.4\\.bias$", names(state_out), value = TRUE)
    if (length(logvar_bias_keys) < 1L && "head.logvar_bias" %in% names(state_out)) {
      logvar_bias_keys <- "head.logvar_bias"
    }
    for (bias_key in logvar_bias_keys) {
      state_out[[bias_key]] <- state_out[[bias_key]] + logvar_offset
      logvar_count <- logvar_count + 1L
    }
    list(
      state = state_out,
      active = head_count > 0L && (abs(ratio - 1) > 1e-12 || abs(offset) > 1e-12),
      ratio = ratio,
      offset = offset,
      head_count = head_count,
      logvar_offset = logvar_offset,
      logvar_count = logvar_count
    )
  }

  resolve_auto_patch_geometry <- function(train_df, val_df, default_patch_size) {
    patch_pyramid_active <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_AUTO_PATCH_PYRAMID", unset = "1")))
    patch_pyramid_active <- is.finite(patch_pyramid_active) && patch_pyramid_active > 0
    patch_layers <- tryCatch(context$stack[[context$predictors]], error = function(e) NULL)
    patch_resolution <- tryCatch({
      res_xy <- terra::res(patch_layers[[1]])
      mean(abs(as.numeric(res_xy)), na.rm = TRUE)
    }, error = function(e) NA_real_)
    patch_plan <- list(
      global_size = as.integer(default_patch_size),
      local_size = NA_integer_,
      patch_resolution = NA_real_
    )
    forced_patch_geometry <- resolve_fixed_patch_geometry(default_patch_size = default_patch_size)
    if (!is.null(forced_patch_geometry)) {
      patch_plan$global_size <- as.integer(forced_patch_geometry$global_size)
      patch_plan$local_size <- suppressWarnings(as.integer(forced_patch_geometry$local_size))
      patch_plan$patch_resolution <- as.numeric(patch_resolution)
      cat(sprintf(
        "[Auto patch] fixed geometry -> global=%d local=%s [source=%s]\n",
        patch_plan$global_size,
        if (is.finite(patch_plan$local_size)) as.character(patch_plan$local_size) else "NA",
        forced_patch_geometry$source
      ))
      return(patch_plan)
    }
    if (!patch_pyramid_active) {
      patch_plan$local_size <- as.integer(default_patch_size)
      patch_plan$patch_resolution <- as.numeric(patch_resolution)
      cat(sprintf(
        "[Auto patch] pyramid disabled -> global=%d local=%d [leader support]\n",
        patch_plan$global_size,
        patch_plan$local_size
      ))
      return(patch_plan)
    }

    if (is.null(conv_env) || is.null(context) ||
        is.null(conv_env$auto_patch_geometry_v5) ||
        !is.function(conv_env$auto_patch_geometry_v5) ||
        is.null(conv_env$fit_variogram_auto) ||
        !is.function(conv_env$fit_variogram_auto)) {
      return(patch_plan)
    }

    calibration_df <- if (isTRUE(refit_full_calibration)) {
      if (!is.null(full_calibration_df)) full_calibration_df else bind_rows(train_df, val_df)
    } else {
      train_df
    }
    if (nrow(calibration_df) < 25L) {
      return(patch_plan)
    }
    coords_cal <- as.matrix(calibration_df[, c("x", "y"), drop = FALSE])
    vg_fit_ctrl <- if (!is.null(conv_env$.auto_variogram_fit_controls_from_n) &&
                       is.function(conv_env$.auto_variogram_fit_controls_from_n)) {
      conv_env$.auto_variogram_fit_controls_from_n(nrow(calibration_df))
    } else {
      list(
        cutoff_frac = if (!is.null(conv_params$vg_cutoff_frac)) conv_params$vg_cutoff_frac else 0.50,
        n_lags = if (!is.null(conv_params$vg_n_lags)) as.integer(conv_params$vg_n_lags) else 15L
      )
    }
    vg_cutoff_use <- suppressWarnings(as.numeric(vg_fit_ctrl$cutoff_frac))
    if (!is.finite(vg_cutoff_use)) vg_cutoff_use <- if (!is.null(conv_params$vg_cutoff_frac)) conv_params$vg_cutoff_frac else 0.50
    vg_n_lags_use <- suppressWarnings(as.integer(round(vg_fit_ctrl$n_lags)))
    if (!is.finite(vg_n_lags_use)) vg_n_lags_use <- if (!is.null(conv_params$vg_n_lags)) as.integer(conv_params$vg_n_lags) else 15L
    vg_patch <- tryCatch(
      conv_env$fit_variogram_auto(
        calibration_df[[response_name_wadoux]],
        coords_cal,
        cutoff_frac = vg_cutoff_use,
        n_lags = vg_n_lags_use
      ),
      error = function(e) NULL
    )
    if (is.null(vg_patch)) {
      return(patch_plan)
    }

    geom <- tryCatch(
      conv_env$auto_patch_geometry_v5(
        vg = vg_patch,
        n_train = nrow(calibration_df),
        patch_resolution = patch_resolution
      ),
      error = function(e) NULL
    )
    if (is.null(geom)) {
      return(patch_plan)
    }

    patch_plan$global_size <- as.integer(geom$global_size)
    patch_plan$local_size <- as.integer(geom$local_size)
    patch_plan$patch_resolution <- as.numeric(patch_resolution)
    cat(sprintf(
      "[Auto patch] resolved global=%d local=%d (resolution=%.4f)\n",
      patch_plan$global_size,
      patch_plan$local_size,
      patch_plan$patch_resolution
    ))
    patch_plan
  }

  prepare_conv_fd_on_explicit_split <- function(train_df, val_df, test_df, params) {
    combined_df <- bind_rows(train_df, val_df, test_df)
    n_train <- nrow(train_df)
    n_val <- nrow(val_df)
    n_test <- nrow(test_df)
    patch_cache_combined <- NULL
    patch_plan <- resolve_auto_patch_geometry(train_df, val_df, default_patch_size = patch_size)
    patch_size_use <- as.integer(patch_plan$global_size)

    cache_col <- "..patch_cache_row_id"
    train_val_cache_ok <- !is.null(calibration_patch_cache) &&
      identical(as.integer(calibration_patch_cache$patch_size), patch_size_use) &&
      all(cache_col %in% names(train_df), cache_col %in% names(val_df)) &&
      !anyNA(train_df[[cache_col]]) &&
      !anyNA(val_df[[cache_col]])
    test_cache_ok <- !is.null(calibration_patch_cache) &&
      identical(as.integer(calibration_patch_cache$patch_size), patch_size_use) &&
      cache_col %in% names(test_df) &&
      !anyNA(test_df[[cache_col]])

    if (train_val_cache_ok) {
      train_cache_idx <- as.integer(train_df[[cache_col]])
      val_cache_idx <- as.integer(val_df[[cache_col]])
      patch_arrays <- list(
        calibration_patch_cache$patches_raw[, , , train_cache_idx, drop = FALSE],
        calibration_patch_cache$patches_raw[, , , val_cache_idx, drop = FALSE]
      )

      if (test_cache_ok) {
        test_cache_idx <- as.integer(test_df[[cache_col]])
        patch_arrays[[3]] <- calibration_patch_cache$patches_raw[, , , test_cache_idx, drop = FALSE]
      } else {
        external_test_cache <- conv_env$build_pointpatch_patch_cache(
          context = context,
          calibration_df = test_df,
          patch_size = patch_size_use,
          label = "test patches"
        )
        patch_arrays[[3]] <- external_test_cache$patches_raw
      }

      patch_cache_combined <- list(
        patches_raw = bind_patch_arrays_4d(patch_arrays),
        patch_size = patch_size_use,
        n = nrow(combined_df),
        resolution_x = calibration_patch_cache$resolution_x,
        resolution_y = calibration_patch_cache$resolution_y,
        resolution_mean = calibration_patch_cache$resolution_mean,
        patch_local_size = patch_plan$local_size
      )
    }

    neighbor_pool_k <- suppressWarnings(as.integer(Sys.getenv("WADOUX_NEIGHBOR_POOL_K", unset = "30")))
    if (!is.finite(neighbor_pool_k) || neighbor_pool_k < 6L) {
      neighbor_pool_k <- 30L
    }
    if (!is.null(params$K_neighbors)) {
      neighbor_pool_k <- max(neighbor_pool_k, as.integer(params$K_neighbors))
    }

    fd_conv <- conv_env$prepare_pointpatch_fold(
      context = context,
      calibration_df = combined_df,
      train_idx = seq_len(n_train),
      val_idx = n_train + seq_len(n_val),
      test_idx = n_train + n_val + seq_len(n_test),
      patch_size = patch_size_use,
      K = neighbor_pool_k,
      patch_cache = patch_cache_combined
    )
    if (is.null(fd_conv$patch_meta)) fd_conv$patch_meta <- list()
    fd_conv$patch_meta$patch_local_size <- patch_plan$local_size
    fd_conv$patch_meta$patch_size <- patch_size_use
    fd_conv$patch_meta$resolution_mean <- patch_plan$patch_resolution

    list(fd = fd_conv, neighbor_pool_k = neighbor_pool_k, patch_plan = patch_plan)
  }

  run_conv_on_explicit_split <- function(train_df, val_df, test_df, params) {
    prep <- prepare_conv_fd_on_explicit_split(
      train_df = train_df,
      val_df = val_df,
      test_df = test_df,
      params = params
    )

    fd_use <- prep$fd
    params_use <- params
    params_use$patch_resolution <- prep$patch_plan$patch_resolution
    params_use$patch_global_size <- prep$patch_plan$global_size
    params_use$patch_local_size <- prep$patch_plan$local_size
    trainer_fn <- conv_env$train_convkrigingnet2d_one_fold
    trainer_formals <- names(formals(trainer_fn))
    trainer_has_dots <- "..." %in% trainer_formals
    if (!trainer_has_dots) {
      params_use <- params_use[intersect(names(params_use), trainer_formals)]
    }
    conv_out <- do.call(
      trainer_fn,
      c(list(fd = fd_use), params_use)
    )
    conv_out
  }

  run_conv_on_prepared_split <- function(prep, params) {
    fd_use <- prep$fd
    params_use <- params
    params_use$patch_resolution <- prep$patch_plan$patch_resolution
    params_use$patch_global_size <- prep$patch_plan$global_size
    params_use$patch_local_size <- prep$patch_plan$local_size
    trainer_fn <- conv_env$train_convkrigingnet2d_one_fold
    trainer_formals <- names(formals(trainer_fn))
    trainer_has_dots <- "..." %in% trainer_formals
    if (!trainer_has_dots) {
      params_use <- params_use[intersect(names(params_use), trainer_formals)]
    }
    conv_out <- do.call(
      trainer_fn,
      c(list(fd = fd_use), params_use)
    )
    conv_out
  }

  run_conv_on_prepared_split_with_env <- function(prep, params, env_overrides = list()) {
    env_names <- names(env_overrides)
    old_env <- setNames(vector("list", length(env_names)), env_names)
    for (nm in env_names) {
      old_env[[nm]] <- Sys.getenv(nm, unset = NA_character_)
    }
    on.exit({
      for (nm in env_names) {
        old_val <- old_env[[nm]]
        if (length(old_val) < 1L || is.na(old_val)) {
          Sys.unsetenv(nm)
        } else {
          do.call(Sys.setenv, setNames(list(old_val), nm))
        }
      }
    }, add = TRUE)
    if (length(env_overrides) > 0L) {
      do.call(Sys.setenv, env_overrides)
    }
    run_conv_on_prepared_split(prep, params)
  }

  median_nn_distance <- function(coords) {
    coords <- as.matrix(coords)
    if (nrow(coords) < 2L) return(NA_real_)
    dmat <- as.matrix(stats::dist(coords))
    diag(dmat) <- Inf
    nn <- apply(dmat, 1L, min, na.rm = TRUE)
    nn <- nn[is.finite(nn) & nn > 0]
    if (length(nn) < 1L) return(NA_real_)
    stats::median(nn, na.rm = TRUE)
  }

  build_decluster_bootstrap_plan <- function(train_df,
                                             cell_mult = 1.0) {
    coords <- as.matrix(train_df[, c("x", "y"), drop = FALSE])
    n <- nrow(coords)
    nn_med <- median_nn_distance(coords)
    if (!is.finite(nn_med) || nn_med <= 0) {
      x_span <- diff(range(coords[, 1], na.rm = TRUE))
      y_span <- diff(range(coords[, 2], na.rm = TRUE))
      area <- max(1e-6, x_span * y_span)
      nn_med <- sqrt(area / max(1L, n))
    }
    cell_size <- as.numeric(max(1e-6, nn_med * max(0.25, cell_mult)))
    origin_x <- min(coords[, 1], na.rm = TRUE)
    origin_y <- min(coords[, 2], na.rm = TRUE)
    gx <- floor((coords[, 1] - origin_x) / cell_size)
    gy <- floor((coords[, 2] - origin_y) / cell_size)
    keys <- paste(gx, gy, sep = ":")
    counts_tab <- table(keys)
    counts <- as.numeric(counts_tab[keys])
    inv_counts <- 1 / pmax(1, counts)
    probs <- inv_counts / sum(inv_counts)
    list(
      probs = probs,
      cell_size = cell_size,
      median_nn = nn_med,
      n_cells = length(counts_tab),
      max_cell_count = max(counts),
      mean_cell_count = mean(as.numeric(counts_tab))
    )
  }

  sample_decluster_bootstrap_train <- function(train_df,
                                               plan,
                                               seed,
                                               size = nrow(train_df)) {
    set.seed(as.integer(seed))
    idx <- sample.int(
      n = nrow(train_df),
      size = as.integer(size),
      replace = TRUE,
      prob = plan$probs
    )
    sampled_df <- train_df[idx, , drop = FALSE]
    unique_fraction <- length(unique(idx)) / max(1L, length(idx))
    list(
      train_df = sampled_df,
      idx = idx,
      unique_fraction = unique_fraction,
      duplicate_fraction = 1 - unique_fraction
    )
  }

  compute_prediction_diversity_stats <- function(pred_matrix) {
    if (is.null(pred_matrix) || nrow(pred_matrix) < 2L || ncol(pred_matrix) < 2L) {
      return(list(
        pred_sd_mean = NA_real_,
        pred_sd_median = NA_real_,
        corr_mean = NA_real_,
        corr_sd = NA_real_
      ))
    }
    point_sd <- apply(pred_matrix, 2L, stats::sd)
    cor_mat <- suppressWarnings(stats::cor(t(pred_matrix)))
    cor_vals <- cor_mat[upper.tri(cor_mat)]
    cor_vals <- cor_vals[is.finite(cor_vals)]
    list(
      pred_sd_mean = mean(point_sd, na.rm = TRUE),
      pred_sd_median = stats::median(point_sd, na.rm = TRUE),
      corr_mean = if (length(cor_vals) > 0L) mean(cor_vals, na.rm = TRUE) else NA_real_,
      corr_sd = if (length(cor_vals) > 1L) stats::sd(cor_vals, na.rm = TRUE) else NA_real_
    )
  }

  fit_isotonic_calibrator <- function(pred, obs) {
    keep <- is.finite(pred) & is.finite(obs)
    pred <- as.numeric(pred[keep])
    obs <- as.numeric(obs[keep])
    if (length(pred) < 8L || length(unique(pred)) < 4L) {
      return(NULL)
    }
    ord <- order(pred, obs)
    iso <- stats::isoreg(pred[ord], obs[ord])
    x_iso <- as.numeric(iso$x)
    y_iso <- as.numeric(iso$yf)
    if (length(x_iso) < 2L || length(y_iso) < 2L) {
      return(NULL)
    }
    split_idx <- split(seq_along(x_iso), x_iso)
    x_unique <- as.numeric(names(split_idx))
    y_unique <- vapply(split_idx, function(idx) mean(y_iso[idx], na.rm = TRUE), numeric(1))
    if (length(x_unique) < 2L || any(!is.finite(x_unique)) || any(!is.finite(y_unique))) {
      return(NULL)
    }
    y_unique <- cummax(y_unique)
    fn <- stats::approxfun(x = x_unique, y = y_unique, method = "linear", rule = 2, ties = "ordered")
    list(
      x = x_unique,
      y = y_unique,
      apply = function(new_pred) as.numeric(fn(new_pred))
    )
  }

  apply_isotonic_calibrator <- function(pred, calibrator) {
    if (is.null(calibrator)) return(pred)
    calibrator$apply(pred)
  }

  load_external_patch_ssl_meta_only <- local({
    cached_meta_path <- NULL
    cached_meta <- NULL
    function() {
      state_path <- trimws(Sys.getenv("GEOVERSA_SSL_PATCH_STATE", unset = ""))
      if (!nzchar(state_path)) {
        return(NULL)
      }
      state_path <- path.expand(state_path)
      meta_path <- file.path(dirname(state_path), "geoversa_ssl_pretrain_summary.csv")
      if (identical(cached_meta_path, meta_path)) {
        return(cached_meta)
      }
      meta <- if (file.exists(meta_path)) {
        tryCatch(read.csv(meta_path, stringsAsFactors = FALSE), error = function(e) NULL)
      } else {
        NULL
      }
      if (!is.null(meta) && is.data.frame(meta) && nrow(meta) > 0L) {
        meta <- meta[1L, , drop = FALSE]
      } else {
        meta <- NULL
      }
      cached_meta_path <<- meta_path
      cached_meta <<- meta
      meta
    }
  })

  resolve_fixed_patch_geometry <- function(default_patch_size = NA_integer_) {
    global_raw <- trimws(Sys.getenv(
      "GEOVERSA_PATCH_GLOBAL_SIZE",
      unset = Sys.getenv("GEOVERSA_SSL_PATCH_GLOBAL_SIZE", unset = "")
    ))
    local_raw <- trimws(Sys.getenv(
      "GEOVERSA_PATCH_LOCAL_SIZE",
      unset = Sys.getenv("GEOVERSA_SSL_PATCH_LOCAL_SIZE", unset = "")
    ))
    source <- NULL
    global_size <- suppressWarnings(as.integer(round(as.numeric(global_raw))))
    local_size <- suppressWarnings(as.integer(round(as.numeric(local_raw))))
    observed_multiscale <- NA

    if (nzchar(global_raw)) {
      if (!is.finite(global_size) || global_size <= 0L) {
        stop("GEOVERSA_PATCH_GLOBAL_SIZE must be a positive integer.")
      }
      source <- "env"
      if (nzchar(local_raw) && (!is.finite(local_size) || local_size <= 0L)) {
        stop("GEOVERSA_PATCH_LOCAL_SIZE must be a positive integer when provided.")
      }
    } else {
      meta <- load_external_patch_ssl_meta_only()
      if (!is.null(meta)) {
        source <- "external_ssl"
        global_size <- if ("ssl_patch_global_size" %in% names(meta)) {
          suppressWarnings(as.integer(meta$ssl_patch_global_size[1]))
        } else {
          suppressWarnings(as.integer(meta$ssl_patch_size[1]))
        }
        local_size <- suppressWarnings(as.integer(meta$ssl_patch_local_size[1]))
        observed_multiscale <- if ("ssl_multiscale" %in% names(meta)) {
          isTRUE(meta$ssl_multiscale[1])
        } else if ("ssl_encoder_type" %in% names(meta)) {
          identical(as.character(meta$ssl_encoder_type[1]), "multiscale")
        } else {
          FALSE
        }
        if (!is.finite(global_size) || global_size <= 0L) {
          stop("External SSL patch metadata is missing a valid global patch size.")
        }
        if (isTRUE(observed_multiscale) && (!is.finite(local_size) || local_size <= 0L)) {
          stop("External SSL patch metadata indicates multiscale encoder but is missing a valid local patch size.")
        }
      }
    }

    if (is.null(source)) {
      return(NULL)
    }

    if (!is.finite(global_size) || global_size <= 0L) {
      global_size <- suppressWarnings(as.integer(default_patch_size))
    }
    global_size <- as.integer(max(3L, global_size))

    if (is.finite(local_size) && local_size > 0L) {
      local_size <- as.integer(max(3L, min(global_size, local_size)))
      if (local_size %% 2L == 0L) local_size <- local_size - 1L
      if (local_size < 3L) local_size <- NA_integer_
    } else if (identical(source, "env")) {
      patch_pyramid_active <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_AUTO_PATCH_PYRAMID", unset = "1")))
      patch_pyramid_active <- is.finite(patch_pyramid_active) && patch_pyramid_active > 0
      if (patch_pyramid_active) {
        local_size <- as.integer(max(9L, min(global_size, ifelse(global_size %% 2L == 0L, global_size - 1L, global_size))))
      } else {
        local_size <- NA_integer_
      }
    } else if (!isTRUE(observed_multiscale)) {
      local_size <- NA_integer_
    }

    list(
      global_size = global_size,
      local_size = local_size,
      source = source
    )
  }

  load_external_patch_ssl_artifact <- function() {
    state_path <- trimws(Sys.getenv("GEOVERSA_SSL_PATCH_STATE", unset = ""))
    if (!nzchar(state_path)) {
      return(NULL)
    }
    state_path <- path.expand(state_path)
    if (!file.exists(state_path)) {
      stop("GEOVERSA_SSL_PATCH_STATE does not exist: ", state_path)
    }
    state_dict <- torch::torch_load(state_path)
    meta <- load_external_patch_ssl_meta_only()
    list(
      state_path = state_path,
      state_dict = state_dict,
      meta = meta
    )
  }

  validate_external_patch_ssl_artifact <- function(ssl_artifact, prep) {
    if (is.null(ssl_artifact)) {
      return(invisible(NULL))
    }
    meta <- ssl_artifact$meta
    if (is.null(meta) || !is.data.frame(meta) || nrow(meta) < 1L) {
      stop("External SSL patch state is missing geoversa_ssl_pretrain_summary.csv metadata; regenerate the artifact before using GEOVERSA_SSL_PATCH_STATE.")
    }
    meta <- meta[1L, , drop = FALSE]
    expected_global <- as.integer(prep$patch_plan$global_size)
    expected_local <- suppressWarnings(as.integer(prep$patch_plan$local_size))
    expected_multiscale <- is.finite(expected_local) && expected_local > 0L
    trainer_multiscale <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_AUTO_PATCH_PYRAMID", unset = "1")))
    trainer_multiscale <- is.finite(trainer_multiscale) && trainer_multiscale > 0
    observed_multiscale <- isTRUE(meta$ssl_multiscale[1])
    observed_encoder_type <- if ("ssl_encoder_type" %in% names(meta)) as.character(meta$ssl_encoder_type[1]) else if (observed_multiscale) "multiscale" else "single_scale"
    observed_global <- if ("ssl_patch_global_size" %in% names(meta)) suppressWarnings(as.integer(meta$ssl_patch_global_size[1])) else suppressWarnings(as.integer(meta$ssl_patch_size[1]))
    observed_local <- suppressWarnings(as.integer(meta$ssl_patch_local_size[1]))
    observed_channels <- suppressWarnings(as.integer(meta$ssl_patch_channels[1]))
    observed_patch_dim <- suppressWarnings(as.integer(meta$ssl_patch_dim[1]))
    expected_channels <- as.integer(dim(prep$fd$patches$train)[1L])
    expected_d <- if (!is.null(conv_env$.auto_d_from_n) && is.function(conv_env$.auto_d_from_n)) {
      conv_env$.auto_d_from_n(nrow(prep$fd$X$train))
    } else {
      192L
    }
    expected_patch_dim <- if (!is.null(conv_env$.auto_patch_dim_from_cnn) &&
                              is.function(conv_env$.auto_patch_dim_from_cnn)) {
      conv_env$.auto_patch_dim_from_cnn(expected_channels, expected_global, expected_d)
    } else {
      NA_integer_
    }

    problems <- character(0)
    if (!is.finite(observed_global) || observed_global != expected_global) {
      problems <- c(problems, sprintf("global patch size mismatch (ssl=%s, benchmark=%s)", observed_global, expected_global))
    }
    if (expected_multiscale != observed_multiscale) {
      problems <- c(problems, sprintf("encoder pyramid mismatch (ssl=%s/%s, benchmark=%s)", observed_encoder_type, observed_multiscale, expected_multiscale))
    }
    if (trainer_multiscale != observed_multiscale) {
      problems <- c(problems, sprintf("trainer patch pyramid mismatch (ssl=%s/%s, trainer=%s)", observed_encoder_type, observed_multiscale, trainer_multiscale))
    }
    if (expected_multiscale && (!is.finite(observed_local) || observed_local != expected_local)) {
      problems <- c(problems, sprintf("local patch size mismatch (ssl=%s, benchmark=%s)", observed_local, expected_local))
    }
    if (is.finite(observed_channels) && observed_channels != expected_channels) {
      problems <- c(problems, sprintf("patch channel mismatch (ssl=%s, benchmark=%s)", observed_channels, expected_channels))
    }
    if (is.finite(observed_patch_dim) && is.finite(expected_patch_dim) && observed_patch_dim != expected_patch_dim) {
      problems <- c(problems, sprintf("patch_dim mismatch (ssl=%s, benchmark=%s)", observed_patch_dim, expected_patch_dim))
    }
    if (length(problems) > 0L) {
      stop(
        "External SSL patch state is incompatible with the current LeaderEnsemble benchmark: ",
        paste(problems, collapse = "; "),
        ". Regenerate the SSL artifact with matching geometry before rerunning."
      )
    }
    invisible(NULL)
  }

  aggregate_committee_diagnostics <- function(diag_list,
                                              weights,
                                              committee_size,
                                              committee_selected_count,
                                              weighting_mode,
                                              tau,
                                              topk,
                                              member_val_rmse,
                                              member_test_rmse) {
    if (!is.list(diag_list) || length(diag_list) < 1L) {
      return(data.frame())
    }
    diag_clean <- lapply(diag_list, function(x) {
      if (is.null(x)) return(data.frame())
      if (!is.data.frame(x)) x <- as.data.frame(x, stringsAsFactors = FALSE)
      x[1L, , drop = FALSE]
    })
    template <- diag_clean[[1L]]
    out <- template
    numeric_cols <- names(template)[vapply(template, is.numeric, logical(1))]
    for (nm in numeric_cols) {
      vals <- vapply(diag_clean, function(x) {
        v <- suppressWarnings(as.numeric(x[[nm]][1]))
        if (!is.finite(v)) NA_real_ else v
      }, numeric(1))
      if (all(!is.finite(vals))) next
      out[[nm]] <- stats::weighted.mean(vals, weights, na.rm = TRUE)
    }
    out$committee_active <- TRUE
    out$committee_size <- as.integer(committee_size)
    out$committee_selected_count <- as.integer(committee_selected_count)
    out$committee_weighting <- as.character(weighting_mode)
    out$committee_tau <- as.numeric(tau)
    out$committee_topk <- as.integer(topk)
    out$committee_val_rmse_mean <- mean(member_val_rmse, na.rm = TRUE)
    out$committee_val_rmse_sd <- if (length(member_val_rmse) > 1L) stats::sd(member_val_rmse, na.rm = TRUE) else NA_real_
    out$committee_val_rmse_best <- min(member_val_rmse, na.rm = TRUE)
    out$committee_test_rmse_mean <- mean(member_test_rmse, na.rm = TRUE)
    out$committee_test_rmse_sd <- if (length(member_test_rmse) > 1L) stats::sd(member_test_rmse, na.rm = TRUE) else NA_real_
    out$committee_weight_max <- max(weights, na.rm = TRUE)
    out$committee_weight_min <- min(weights, na.rm = TRUE)
    out
  }

  run_conv_committee_on_explicit_split <- function(train_df, val_df, test_df, params) {
    prep <- prepare_conv_fd_on_explicit_split(
      train_df = train_df,
      val_df = val_df,
      test_df = test_df,
      params = params
    )

    committee_size <- as.integer(round(parse_num_env("GEOVERSA_AUTO_COMMITTEE_SIZE", 5)))
    committee_size <- max(2L, min(9L, committee_size))
    weighting_mode <- Sys.getenv("GEOVERSA_AUTO_COMMITTEE_WEIGHTING", unset = "val_rmse_softmax")
    weighting_mode <- trimws(tolower(weighting_mode))
    if (!(weighting_mode %in% c("uniform", "val_rmse_softmax", "topk_uniform"))) {
      weighting_mode <- "val_rmse_softmax"
    }
    tau <- parse_num_env("GEOVERSA_AUTO_COMMITTEE_TAU", 8)
    tau <- max(0.1, min(50, tau))
    topk <- as.integer(round(parse_num_env("GEOVERSA_AUTO_COMMITTEE_TOPK", max(2L, ceiling(0.6 * committee_size)))))
    if (!is.finite(topk)) topk <- max(2L, ceiling(0.6 * committee_size))
    topk <- max(2L, min(committee_size, topk))
    seed_stride <- as.integer(round(parse_num_env("GEOVERSA_AUTO_COMMITTEE_SEED_STRIDE", 1009)))
    if (!is.finite(seed_stride) || seed_stride < 1L) seed_stride <- 1009L

    member_out <- vector("list", committee_size)
    member_val_rmse <- rep(NA_real_, committee_size)
    member_test_rmse <- rep(NA_real_, committee_size)
    pred_test_scaled_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_scaled_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_logvar_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_logvar_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    base_seed <- if (!is.null(params$train_seed) && is.finite(params$train_seed)) as.integer(params$train_seed) else 123L

    cat(sprintf("[GeoVersa committee] training %d members | weighting=%s | tau=%.2f | topk=%d\n",
                committee_size, weighting_mode, tau, topk))

    for (m in seq_len(committee_size)) {
      params_m <- params
      params_m$train_seed <- as.integer(base_seed + (m - 1L) * seed_stride)
      cat(sprintf("[GeoVersa committee] member %d/%d | train_seed=%d\n",
                  m, committee_size, params_m$train_seed))
      member_out[[m]] <- run_conv_on_prepared_split(prep, params_m)
      member_val_rmse[m] <- wadoux_eval(obs = val_df[[response_name_wadoux]], pred = member_out[[m]]$pred_val)$RMSE
      member_test_rmse[m] <- wadoux_eval(obs = test_df[[response_name_wadoux]], pred = member_out[[m]]$pred_test)$RMSE
      pred_val_scaled_mat[m, ] <- member_out[[m]]$pred_val_scaled
      pred_test_scaled_mat[m, ] <- member_out[[m]]$pred_test_scaled
      pred_val_logvar_mat[m, ] <- member_out[[m]]$pred_val_logvar_scaled
      pred_test_logvar_mat[m, ] <- member_out[[m]]$pred_test_logvar_scaled
      cat(sprintf("[GeoVersa committee] member %d | val_RMSE=%.3f | test_RMSE=%.3f\n",
                  m, member_val_rmse[m], member_test_rmse[m]))
    }

    ranking <- order(member_val_rmse, decreasing = FALSE, na.last = TRUE)
    selected_idx <- if (identical(weighting_mode, "topk_uniform")) {
      ranking[seq_len(min(topk, length(ranking)))]
    } else {
      seq_len(committee_size)
    }
    weights <- rep(0, committee_size)
    if (identical(weighting_mode, "uniform")) {
      weights[] <- 1 / committee_size
    } else if (identical(weighting_mode, "topk_uniform")) {
      weights[selected_idx] <- 1 / length(selected_idx)
    } else {
      score <- exp(-tau * (member_val_rmse - min(member_val_rmse, na.rm = TRUE)))
      weights <- score / sum(score)
      selected_idx <- seq_len(committee_size)
    }

    weighted_average <- function(key) {
      Reduce(`+`, lapply(seq_along(member_out), function(i) weights[i] * member_out[[i]][[key]]))
    }
    uq_meta <- extract_shared_uq_meta(member_out)
    uq_test <- combine_normal_members_scaled(pred_test_scaled_mat, pred_test_logvar_mat, weights = weights)
    uq_val <- combine_normal_members_scaled(pred_val_scaled_mat, pred_val_logvar_mat, weights = weights)

    agg <- list(
      pred_test = weighted_average("pred_test"),
      pred_test_base = weighted_average("pred_test_base"),
      pred_val = weighted_average("pred_val"),
      pred_val_base = weighted_average("pred_val_base"),
      pred_test_scaled = uq_test$mean,
      pred_test_logvar_scaled = uq_test$logvar,
      pred_val_scaled = uq_val$mean,
      pred_val_logvar_scaled = uq_val$logvar,
      target_scaler = if (!is.null(uq_meta)) uq_meta$target_scaler else NULL,
      target_transform = if (!is.null(uq_meta)) uq_meta$target_transform else NA_character_,
      diagnostics = aggregate_committee_diagnostics(
        diag_list = lapply(member_out, `[[`, "diagnostics"),
        weights = weights,
        committee_size = committee_size,
        committee_selected_count = length(selected_idx),
        weighting_mode = weighting_mode,
        tau = tau,
        topk = topk,
        member_val_rmse = member_val_rmse,
        member_test_rmse = member_test_rmse
      ),
      committee_member_val_rmse = member_val_rmse,
      committee_member_test_rmse = member_test_rmse,
      committee_weights = weights,
      committee_member_count = committee_size
    )
    agg
  }

  run_geoversa2_dbhe_on_explicit_split <- function(train_df,
                                                   val_df,
                                                   test_df,
                                                   params,
                                                   leader_out) {
    committee_size <- as.integer(round(parse_num_env("GEOVERSA2_DBHE_HEADS", 5)))
    committee_size <- max(2L, min(9L, committee_size))
    seed_stride <- as.integer(round(parse_num_env("GEOVERSA2_DBHE_SEED_STRIDE", 1009)))
    if (!is.finite(seed_stride) || seed_stride < 1L) seed_stride <- 1009L
    cell_mult <- parse_num_env("GEOVERSA2_DBHE_DECLUSTER_CELL_MULT", 1.0)
    cell_mult <- max(0.25, min(4.0, cell_mult))
    train_size <- as.integer(round(parse_num_env("GEOVERSA2_DBHE_BOOTSTRAP_SIZE", nrow(train_df))))
    train_size <- max(16L, train_size)
    weighting_mode <- "uniform"
    weights <- rep(1 / committee_size, committee_size)
    base_seed <- if (!is.null(params$train_seed) && is.finite(params$train_seed)) as.integer(params$train_seed) else 123L

    decluster_plan <- build_decluster_bootstrap_plan(train_df, cell_mult = cell_mult)
    member_out <- vector("list", committee_size)
    member_val_rmse <- rep(NA_real_, committee_size)
    member_test_rmse <- rep(NA_real_, committee_size)
    member_unique_fraction <- rep(NA_real_, committee_size)
    pred_test_scaled_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_scaled_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_logvar_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_logvar_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_base_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_base_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))

    cat(sprintf(
      "[GeoVersa2-DBHE] training %d heads | bootstrap=declustered | cell_size=%.3f | median_nn=%.3f\n",
      committee_size,
      decluster_plan$cell_size,
      decluster_plan$median_nn
    ))

    for (m in seq_len(committee_size)) {
      seed_m <- as.integer(base_seed + (m - 1L) * seed_stride)
      boot <- sample_decluster_bootstrap_train(
        train_df = train_df,
        plan = decluster_plan,
        seed = seed_m,
        size = train_size
      )
      member_unique_fraction[m] <- boot$unique_fraction
      prep_m <- prepare_conv_fd_on_explicit_split(
        train_df = boot$train_df,
        val_df = val_df,
        test_df = test_df,
        params = params
      )
      params_m <- modifyList(
        params,
        list(
          return_state = FALSE,
          init_state = NULL,
          hyperlocal_trunk_state = leader_out$selected_state,
          train_seed = seed_m
        )
      )
      cat(sprintf(
        "[GeoVersa2-DBHE] head %d/%d | seed=%d | unique=%.3f\n",
        m, committee_size, seed_m, member_unique_fraction[m]
      ))
      member_out[[m]] <- run_conv_on_prepared_split_with_env(
        prep_m,
        params_m,
        env_overrides = list(
          GEOVERSA2_HYPERLOCAL = "1",
          GEOVERSA2_HYPERLOCAL_FAITHFUL = "1",
          GEOVERSA2_HYPERLOCAL_WARMUP_BYPASS = "1",
          GEOVERSA2_HYPERLOCAL_FREEZE_TRUNK = "1",
          GEOVERSA2_HYPERLOCAL_TRAIN_SCOPE = "fuse_head_krig"
        )
      )
      member_val_rmse[m] <- wadoux_eval(obs = val_df[[response_name_wadoux]], pred = member_out[[m]]$pred_val)$RMSE
      member_test_rmse[m] <- wadoux_eval(obs = test_df[[response_name_wadoux]], pred = member_out[[m]]$pred_test)$RMSE
      pred_val_scaled_mat[m, ] <- member_out[[m]]$pred_val_scaled
      pred_test_scaled_mat[m, ] <- member_out[[m]]$pred_test_scaled
      pred_val_logvar_mat[m, ] <- member_out[[m]]$pred_val_logvar_scaled
      pred_test_logvar_mat[m, ] <- member_out[[m]]$pred_test_logvar_scaled
      pred_val_mat[m, ] <- member_out[[m]]$pred_val
      pred_test_mat[m, ] <- member_out[[m]]$pred_test
      pred_val_base_mat[m, ] <- member_out[[m]]$pred_val_base
      pred_test_base_mat[m, ] <- member_out[[m]]$pred_test_base
      cat(sprintf(
        "[GeoVersa2-DBHE] head %d | val_RMSE=%.3f | test_RMSE=%.3f\n",
        m, member_val_rmse[m], member_test_rmse[m]
      ))
    }

    diversity <- compute_prediction_diversity_stats(pred_test_mat)
    diagnostics <- aggregate_committee_diagnostics(
      diag_list = lapply(member_out, `[[`, "diagnostics"),
      weights = weights,
      committee_size = committee_size,
      committee_selected_count = committee_size,
      weighting_mode = weighting_mode,
      tau = NA_real_,
      topk = committee_size,
      member_val_rmse = member_val_rmse,
      member_test_rmse = member_test_rmse
    )
    diagnostics <- bind_diag_columns_unique(
      diagnostics,
      data.frame(
        dbhe_active = TRUE,
        dbhe_heads = committee_size,
        dbhe_weighting = weighting_mode,
        dbhe_decluster_cell_size = decluster_plan$cell_size,
        dbhe_decluster_median_nn = decluster_plan$median_nn,
        dbhe_decluster_n_cells = decluster_plan$n_cells,
        dbhe_decluster_max_cell_count = decluster_plan$max_cell_count,
        dbhe_head_unique_fraction_mean = mean(member_unique_fraction, na.rm = TRUE),
        dbhe_head_unique_fraction_sd = if (length(member_unique_fraction) > 1L) stats::sd(member_unique_fraction, na.rm = TRUE) else NA_real_,
        dbhe_test_pred_sd_mean = diversity$pred_sd_mean,
        dbhe_test_pred_sd_median = diversity$pred_sd_median,
        dbhe_head_corr_mean = diversity$corr_mean,
        dbhe_head_corr_sd = diversity$corr_sd,
        stringsAsFactors = FALSE
      ),
      rhs_suffix = "_dbhe"
    )
    uq_meta <- extract_shared_uq_meta(member_out)
    uq_test <- combine_normal_members_scaled(pred_test_scaled_mat, pred_test_logvar_mat, weights = weights)
    uq_val <- combine_normal_members_scaled(pred_val_scaled_mat, pred_val_logvar_mat, weights = weights)

    list(
      pred_test = as.numeric(colMeans(pred_test_mat)),
      pred_test_base = as.numeric(colMeans(pred_test_base_mat)),
      pred_val = as.numeric(colMeans(pred_val_mat)),
      pred_val_base = as.numeric(colMeans(pred_val_base_mat)),
      pred_test_scaled = uq_test$mean,
      pred_test_logvar_scaled = uq_test$logvar,
      pred_val_scaled = uq_val$mean,
      pred_val_logvar_scaled = uq_val$logvar,
      target_scaler = if (!is.null(uq_meta)) uq_meta$target_scaler else NULL,
      target_transform = if (!is.null(uq_meta)) uq_meta$target_transform else NA_character_,
      diagnostics = diagnostics,
      member_val_rmse = member_val_rmse,
      member_test_rmse = member_test_rmse,
      member_unique_fraction = member_unique_fraction
    )
  }

  run_geoversa2_trunk_ensemble_on_explicit_split <- function(train_df,
                                                             val_df,
                                                             test_df,
                                                             params) {
    committee_size <- as.integer(round(parse_num_env("GEOVERSA2_TRUNK_ENSEMBLE_MEMBERS", 5)))
    committee_size <- max(2L, min(9L, committee_size))
    seed_stride <- as.integer(round(parse_num_env("GEOVERSA2_TRUNK_ENSEMBLE_SEED_STRIDE", 1009)))
    if (!is.finite(seed_stride) || seed_stride < 1L) seed_stride <- 1009L
    weighting_mode <- "uniform"
    weights <- rep(1 / committee_size, committee_size)
    base_seed <- if (!is.null(params$train_seed) && is.finite(params$train_seed)) as.integer(params$train_seed) else 123L
    ssl_artifact <- load_external_patch_ssl_artifact()
    ssl_warmup_lr_mult <- parse_num_env("GEOVERSA_SSL_PATCH_WARMUP_LR_MULT", 0.10)
    if (!is.finite(ssl_warmup_lr_mult) || ssl_warmup_lr_mult <= 0) ssl_warmup_lr_mult <- 0.10

    prep <- prepare_conv_fd_on_explicit_split(
      train_df = train_df,
      val_df = val_df,
      test_df = test_df,
      params = params
    )
    validate_external_patch_ssl_artifact(ssl_artifact, prep)

    member_out <- vector("list", committee_size)
    member_val_rmse <- rep(NA_real_, committee_size)
    member_test_rmse <- rep(NA_real_, committee_size)
    leader_val_rmse <- rep(NA_real_, committee_size)
    leader_test_rmse <- rep(NA_real_, committee_size)
    pred_test_scaled_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_scaled_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_logvar_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_logvar_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_base_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_base_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))

    cat(sprintf(
      "[GeoVersa2-TrunkEnsemble] training %d members | trunk_diversity=independent_stageA | weighting=%s\n",
      committee_size,
      weighting_mode
    ))

    for (m in seq_len(committee_size)) {
      seed_m <- as.integer(base_seed + (m - 1L) * seed_stride)
      leader_params_m <- modifyList(
        params,
        list(
          return_state = TRUE,
          train_seed = seed_m,
          patch_ssl_state = if (!is.null(ssl_artifact)) ssl_artifact$state_dict else NULL,
          patch_ssl_meta = if (!is.null(ssl_artifact)) ssl_artifact$meta else NULL,
          patch_ssl_freeze_warmup = FALSE,
          patch_ssl_warmup_lr_mult = if (!is.null(ssl_artifact)) ssl_warmup_lr_mult else NULL
        )
      )
      cat(sprintf(
        "[GeoVersa2-TrunkEnsemble] member %d/%d | leader_seed=%d\n",
        m, committee_size, seed_m
      ))
      leader_out <- run_conv_on_prepared_split_with_env(
        prep,
        leader_params_m,
        env_overrides = list(
          GEOVERSA2_HYPERLOCAL = "0",
          GEOVERSA2_HYPERLOCAL_FAITHFUL = "0",
          GEOVERSA2_HYPERLOCAL_WARMUP_BYPASS = "0",
          GEOVERSA2_HYPERLOCAL_FREEZE_TRUNK = "0",
          GEOVERSA2_HYPERLOCAL_TRAIN_SCOPE = "head_only"
        )
      )
      if (is.null(leader_out$selected_state)) {
        stop("GeoVersa2 trunk ensemble requires leader return_state=TRUE, but no selected_state was returned.")
      }
      leader_val_rmse[m] <- wadoux_eval(obs = val_df[[response_name_wadoux]], pred = leader_out$pred_val)$RMSE
      leader_test_rmse[m] <- wadoux_eval(obs = test_df[[response_name_wadoux]], pred = leader_out$pred_test)$RMSE

      member_params_m <- modifyList(
        params,
        list(
          return_state = FALSE,
          init_state = NULL,
          hyperlocal_trunk_state = leader_out$selected_state,
          train_seed = seed_m,
          patch_ssl_state = if (!is.null(ssl_artifact)) ssl_artifact$state_dict else NULL,
          patch_ssl_meta = if (!is.null(ssl_artifact)) ssl_artifact$meta else NULL,
          patch_ssl_freeze_warmup = FALSE,
          patch_ssl_warmup_lr_mult = if (!is.null(ssl_artifact)) ssl_warmup_lr_mult else NULL
        )
      )
      member_out[[m]] <- run_conv_on_prepared_split_with_env(
        prep,
        member_params_m,
        env_overrides = list(
          GEOVERSA2_HYPERLOCAL = "1",
          GEOVERSA2_HYPERLOCAL_FAITHFUL = "1",
          GEOVERSA2_HYPERLOCAL_WARMUP_BYPASS = "1",
          GEOVERSA2_HYPERLOCAL_FREEZE_TRUNK = "1",
          GEOVERSA2_HYPERLOCAL_TRAIN_SCOPE = "fuse_head_krig"
        )
      )
      member_val_rmse[m] <- wadoux_eval(obs = val_df[[response_name_wadoux]], pred = member_out[[m]]$pred_val)$RMSE
      member_test_rmse[m] <- wadoux_eval(obs = test_df[[response_name_wadoux]], pred = member_out[[m]]$pred_test)$RMSE
      pred_val_scaled_mat[m, ] <- member_out[[m]]$pred_val_scaled
      pred_test_scaled_mat[m, ] <- member_out[[m]]$pred_test_scaled
      pred_val_logvar_mat[m, ] <- member_out[[m]]$pred_val_logvar_scaled
      pred_test_logvar_mat[m, ] <- member_out[[m]]$pred_test_logvar_scaled
      pred_val_mat[m, ] <- member_out[[m]]$pred_val
      pred_test_mat[m, ] <- member_out[[m]]$pred_test
      pred_val_base_mat[m, ] <- member_out[[m]]$pred_val_base
      pred_test_base_mat[m, ] <- member_out[[m]]$pred_test_base
      cat(sprintf(
        "[GeoVersa2-TrunkEnsemble] member %d | leader_test_RMSE=%.3f | faithful_val_RMSE=%.3f | faithful_test_RMSE=%.3f\n",
        m, leader_test_rmse[m], member_val_rmse[m], member_test_rmse[m]
      ))
    }

    diversity <- compute_prediction_diversity_stats(pred_test_mat)
    diagnostics <- aggregate_committee_diagnostics(
      diag_list = lapply(member_out, `[[`, "diagnostics"),
      weights = weights,
      committee_size = committee_size,
      committee_selected_count = committee_size,
      weighting_mode = weighting_mode,
      tau = NA_real_,
      topk = committee_size,
      member_val_rmse = member_val_rmse,
      member_test_rmse = member_test_rmse
    )
    diagnostics <- bind_diag_columns_unique(
      diagnostics,
      data.frame(
        trunk_ensemble_active = TRUE,
        trunk_ensemble_members = committee_size,
        trunk_ensemble_weighting = weighting_mode,
        trunk_ensemble_leader_val_rmse_mean = mean(leader_val_rmse, na.rm = TRUE),
        trunk_ensemble_leader_val_rmse_sd = if (length(leader_val_rmse) > 1L) stats::sd(leader_val_rmse, na.rm = TRUE) else NA_real_,
        trunk_ensemble_leader_test_rmse_mean = mean(leader_test_rmse, na.rm = TRUE),
        trunk_ensemble_leader_test_rmse_sd = if (length(leader_test_rmse) > 1L) stats::sd(leader_test_rmse, na.rm = TRUE) else NA_real_,
        trunk_ensemble_test_pred_sd_mean = diversity$pred_sd_mean,
        trunk_ensemble_test_pred_sd_median = diversity$pred_sd_median,
        trunk_ensemble_member_corr_mean = diversity$corr_mean,
        trunk_ensemble_member_corr_sd = diversity$corr_sd,
        stringsAsFactors = FALSE
      ),
      rhs_suffix = "_trunkens"
    )
    uq_meta <- extract_shared_uq_meta(member_out)
    uq_test <- combine_normal_members_scaled(pred_test_scaled_mat, pred_test_logvar_mat, weights = weights)
    uq_val <- combine_normal_members_scaled(pred_val_scaled_mat, pred_val_logvar_mat, weights = weights)

    list(
      pred_test = as.numeric(colMeans(pred_test_mat)),
      pred_test_base = as.numeric(colMeans(pred_test_base_mat)),
      pred_val = as.numeric(colMeans(pred_val_mat)),
      pred_val_base = as.numeric(colMeans(pred_val_base_mat)),
      pred_test_scaled = uq_test$mean,
      pred_test_logvar_scaled = uq_test$logvar,
      pred_val_scaled = uq_val$mean,
      pred_val_logvar_scaled = uq_val$logvar,
      target_scaler = if (!is.null(uq_meta)) uq_meta$target_scaler else NULL,
      target_transform = if (!is.null(uq_meta)) uq_meta$target_transform else NA_character_,
      diagnostics = diagnostics,
      member_val_rmse = member_val_rmse,
      member_test_rmse = member_test_rmse,
      leader_val_rmse = leader_val_rmse,
      leader_test_rmse = leader_test_rmse
    )
  }

  run_geoversa2_leader_ensemble_on_explicit_split <- function(train_df,
                                                              val_df,
                                                              test_df,
                                                              params) {
    committee_size <- as.integer(round(parse_num_env(
      "GEOVERSA2_LEADER_ENSEMBLE_MEMBERS",
      parse_num_env("GEOVERSA2_TRUNK_ENSEMBLE_MEMBERS", 5)
    )))
    committee_size <- max(2L, min(9L, committee_size))
    seed_stride <- as.integer(round(parse_num_env(
      "GEOVERSA2_LEADER_ENSEMBLE_SEED_STRIDE",
      parse_num_env("GEOVERSA2_TRUNK_ENSEMBLE_SEED_STRIDE", 1009)
    )))
    if (!is.finite(seed_stride) || seed_stride < 1L) seed_stride <- 1009L
    weighting_mode <- "uniform"
    weights <- rep(1 / committee_size, committee_size)
    isotonic_active <- parse_bool_env("GEOVERSA2_LEADER_ENSEMBLE_ISOTONIC", default = FALSE)
    base_seed <- if (!is.null(params$train_seed) && is.finite(params$train_seed)) as.integer(params$train_seed) else 123L
    ssl_artifact <- load_external_patch_ssl_artifact()
    ssl_warmup_lr_mult <- parse_num_env("GEOVERSA_SSL_PATCH_WARMUP_LR_MULT", 0.10)
    if (!is.finite(ssl_warmup_lr_mult) || ssl_warmup_lr_mult <= 0) ssl_warmup_lr_mult <- 0.10

    prep <- prepare_conv_fd_on_explicit_split(
      train_df = train_df,
      val_df = val_df,
      test_df = test_df,
      params = params
    )
    validate_external_patch_ssl_artifact(ssl_artifact, prep)

    member_out <- vector("list", committee_size)
    member_val_rmse <- rep(NA_real_, committee_size)
    member_test_rmse <- rep(NA_real_, committee_size)
    pred_test_scaled_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_scaled_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_logvar_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_logvar_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))
    pred_test_base_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(test_df))
    pred_val_base_mat <- matrix(NA_real_, nrow = committee_size, ncol = nrow(val_df))

    cat(sprintf(
      "[GeoVersa2-LeaderEnsemble] training %d leaders | stageB=off | weighting=%s | ssl=%s\n",
      committee_size,
      weighting_mode,
      if (!is.null(ssl_artifact)) basename(ssl_artifact$state_path) else "off"
    ))

    for (m in seq_len(committee_size)) {
      seed_m <- as.integer(base_seed + (m - 1L) * seed_stride)
      params_m <- modifyList(
        params,
        list(
          return_state = FALSE,
          init_state = NULL,
          hyperlocal_trunk_state = NULL,
          train_seed = seed_m,
          patch_ssl_state = if (!is.null(ssl_artifact)) ssl_artifact$state_dict else NULL,
          patch_ssl_meta = if (!is.null(ssl_artifact)) ssl_artifact$meta else NULL,
          patch_ssl_freeze_warmup = FALSE,
          patch_ssl_warmup_lr_mult = if (!is.null(ssl_artifact)) ssl_warmup_lr_mult else NULL
        )
      )
      cat(sprintf(
        "[GeoVersa2-LeaderEnsemble] member %d/%d | leader_seed=%d | ssl=%s\n",
        m, committee_size, seed_m,
        if (!is.null(ssl_artifact)) "on" else "off"
      ))
      member_out[[m]] <- run_conv_on_prepared_split_with_env(
        prep,
        params_m,
        env_overrides = list(
          GEOVERSA_AUTO_PATCH_SSL = "0",
          GEOVERSA2_HYPERLOCAL = "0",
          GEOVERSA2_HYPERLOCAL_FAITHFUL = "0",
          GEOVERSA2_HYPERLOCAL_WARMUP_BYPASS = "0",
          GEOVERSA2_HYPERLOCAL_FREEZE_TRUNK = "0",
          GEOVERSA2_HYPERLOCAL_TRAIN_SCOPE = "head_only"
        )
      )
      member_val_rmse[m] <- wadoux_eval(obs = val_df[[response_name_wadoux]], pred = member_out[[m]]$pred_val)$RMSE
      member_test_rmse[m] <- wadoux_eval(obs = test_df[[response_name_wadoux]], pred = member_out[[m]]$pred_test)$RMSE
      pred_val_scaled_mat[m, ] <- member_out[[m]]$pred_val_scaled
      pred_test_scaled_mat[m, ] <- member_out[[m]]$pred_test_scaled
      pred_val_logvar_mat[m, ] <- member_out[[m]]$pred_val_logvar_scaled
      pred_test_logvar_mat[m, ] <- member_out[[m]]$pred_test_logvar_scaled
      pred_val_mat[m, ] <- member_out[[m]]$pred_val
      pred_test_mat[m, ] <- member_out[[m]]$pred_test
      pred_val_base_mat[m, ] <- member_out[[m]]$pred_val_base
      pred_test_base_mat[m, ] <- member_out[[m]]$pred_test_base
      cat(sprintf(
        "[GeoVersa2-LeaderEnsemble] member %d | val_RMSE=%.3f | test_RMSE=%.3f\n",
        m, member_val_rmse[m], member_test_rmse[m]
      ))
    }

    diversity <- compute_prediction_diversity_stats(pred_test_mat)
    diagnostics <- aggregate_committee_diagnostics(
      diag_list = lapply(member_out, `[[`, "diagnostics"),
      weights = weights,
      committee_size = committee_size,
      committee_selected_count = committee_size,
      weighting_mode = weighting_mode,
      tau = NA_real_,
      topk = committee_size,
      member_val_rmse = member_val_rmse,
      member_test_rmse = member_test_rmse
    )
    diagnostics <- bind_diag_columns_unique(
      diagnostics,
      data.frame(
        leader_ensemble_active = TRUE,
        leader_ensemble_members = committee_size,
        leader_ensemble_weighting = weighting_mode,
        leader_ensemble_isotonic_active = isotonic_active,
        leader_ensemble_ssl_active = !is.null(ssl_artifact),
        leader_ensemble_ssl_warmup_lr_mult = if (!is.null(ssl_artifact)) ssl_warmup_lr_mult else NA_real_,
        leader_ensemble_test_pred_sd_mean = diversity$pred_sd_mean,
        leader_ensemble_test_pred_sd_median = diversity$pred_sd_median,
        leader_ensemble_member_corr_mean = diversity$corr_mean,
        leader_ensemble_member_corr_sd = diversity$corr_sd,
        stringsAsFactors = FALSE
      ),
      rhs_suffix = "_leaderens"
    )

    pred_test <- as.numeric(colMeans(pred_test_mat))
    pred_test_base <- as.numeric(colMeans(pred_test_base_mat))
    pred_val <- as.numeric(colMeans(pred_val_mat))
    pred_val_base <- as.numeric(colMeans(pred_val_base_mat))
    uq_meta <- extract_shared_uq_meta(member_out)
    uq_test <- combine_normal_members_scaled(pred_test_scaled_mat, pred_test_logvar_mat, weights = weights)
    uq_val <- combine_normal_members_scaled(pred_val_scaled_mat, pred_val_logvar_mat, weights = weights)
    pred_test_scaled <- uq_test$mean
    pred_test_logvar_scaled <- uq_test$logvar
    pred_val_scaled <- uq_val$mean
    pred_val_logvar_scaled <- uq_val$logvar

    isotonic_calibrator <- NULL
    val_rmse_raw <- wadoux_eval(obs = val_df[[response_name_wadoux]], pred = pred_val)$RMSE
    test_rmse_raw <- wadoux_eval(obs = test_df[[response_name_wadoux]], pred = pred_test)$RMSE
    val_rmse_iso <- NA_real_
    test_rmse_iso <- NA_real_
    if (isotonic_active) {
      isotonic_calibrator <- fit_isotonic_calibrator(
        pred = pred_val,
        obs = val_df[[response_name_wadoux]]
      )
      if (!is.null(isotonic_calibrator)) {
        pred_val <- apply_isotonic_calibrator(pred_val, isotonic_calibrator)
        pred_test <- apply_isotonic_calibrator(pred_test, isotonic_calibrator)
        val_rmse_iso <- wadoux_eval(obs = val_df[[response_name_wadoux]], pred = pred_val)$RMSE
        test_rmse_iso <- wadoux_eval(obs = test_df[[response_name_wadoux]], pred = pred_test)$RMSE
      }
    }
    diagnostics <- bind_diag_columns_unique(
      diagnostics,
      data.frame(
        leader_ensemble_isotonic_fitted = !is.null(isotonic_calibrator),
        leader_ensemble_val_rmse_raw = val_rmse_raw,
        leader_ensemble_test_rmse_raw = test_rmse_raw,
        leader_ensemble_val_rmse_isotonic = val_rmse_iso,
        leader_ensemble_test_rmse_isotonic = test_rmse_iso,
        stringsAsFactors = FALSE
      ),
      rhs_suffix = "_leaderiso"
    )

    list(
      pred_test = pred_test,
      pred_test_base = pred_test_base,
      pred_val = pred_val,
      pred_val_base = pred_val_base,
      diagnostics = diagnostics,
      member_val_rmse = member_val_rmse,
      member_test_rmse = member_test_rmse,
      baseline_out = member_out[[1]],
      isotonic_active = isotonic_active,
      isotonic_fitted = !is.null(isotonic_calibrator),
      pred_test_scaled = if (!is.null(isotonic_calibrator)) rep(NA_real_, nrow(test_df)) else pred_test_scaled,
      pred_test_logvar_scaled = if (!is.null(isotonic_calibrator)) rep(NA_real_, nrow(test_df)) else pred_test_logvar_scaled,
      pred_val_scaled = if (!is.null(isotonic_calibrator)) rep(NA_real_, nrow(val_df)) else pred_val_scaled,
      pred_val_logvar_scaled = if (!is.null(isotonic_calibrator)) rep(NA_real_, nrow(val_df)) else pred_val_logvar_scaled,
      target_scaler = if (!is.null(isotonic_calibrator) || is.null(uq_meta)) NULL else uq_meta$target_scaler,
      target_transform = if (!is.null(isotonic_calibrator) || is.null(uq_meta)) NA_character_ else uq_meta$target_transform
    )
  }

	  extract_warmstart_shape_overrides <- function(diag_row) {
	    if (is.null(diag_row) || nrow(diag_row) == 0L) {
	      return(list())
	    }
	    overrides <- list(
      d = as.integer(diag_row$d[1]),
      patch_dim = as.integer(diag_row$patch_dim[1]),
      coord_dim = as.integer(diag_row$coord_dim[1]),
      tab_dropout = as.numeric(diag_row$tab_dropout[1]),
      patch_dropout = as.numeric(diag_row$patch_dropout[1]),
      coord_dropout = as.numeric(diag_row$coord_dropout[1])
	    )
	    overrides[vapply(overrides, function(x) length(x) == 1L && is.finite(x) && x > 0, logical(1))]
	  }

	  diag_scalar <- function(diag_row, name) {
	    if (is.null(diag_row) || nrow(diag_row) == 0L || !name %in% names(diag_row)) {
	      return(NULL)
	    }
	    diag_row[[name]][1]
	  }

	  diag_bool_env <- function(diag_row, name) {
	    x <- diag_scalar(diag_row, name)
	    if (is.null(x) || length(x) == 0L || is.na(x)) {
	      return(NULL)
	    }
	    if (is.logical(x)) {
	      return(if (isTRUE(x)) "1" else "0")
	    }
	    if (is.numeric(x) || is.integer(x)) {
	      if (!is.finite(x)) return(NULL)
	      return(if (as.numeric(x) > 0) "1" else "0")
	    }
	    x_chr <- tolower(trimws(as.character(x)))
	    if (!x_chr %in% c("true", "false", "1", "0", "yes", "no", "y", "n")) {
	      return(NULL)
	    }
	    if (x_chr %in% c("true", "1", "yes", "y")) "1" else "0"
	  }

		  diag_num_env <- function(diag_row, name) {
		    x <- suppressWarnings(as.numeric(diag_scalar(diag_row, name)))
		    if (length(x) == 0L || !is.finite(x)) {
		      return(NULL)
		    }
		    format(x, scientific = FALSE, trim = TRUE, digits = 15)
		  }

		  diag_char_env <- function(diag_row, name, allowed = NULL) {
		    x <- diag_scalar(diag_row, name)
		    if (is.null(x) || length(x) == 0L || is.na(x)) {
		      return(NULL)
		    }
		    x <- trimws(as.character(x[1]))
		    if (!nzchar(x)) {
		      return(NULL)
		    }
		    if (!is.null(allowed) && !tolower(x) %in% tolower(allowed)) {
		      return(NULL)
		    }
		    x
		  }

	  with_temp_env <- function(values, code) {
	    values <- values[!is.na(values) & nzchar(names(values)) & nzchar(values)]
	    if (length(values) == 0L) {
	      return(force(code))
	    }
	    env_names <- names(values)
	    old_values <- Sys.getenv(env_names, unset = NA_character_)
	    on.exit({
	      for (i in seq_along(env_names)) {
	        if (is.na(old_values[[i]])) {
	          Sys.unsetenv(env_names[[i]])
	        } else {
	          do.call(Sys.setenv, stats::setNames(list(old_values[[i]]), env_names[[i]]))
	        }
	      }
	    }, add = TRUE)
	    unset_idx <- values == "__UNSET__"
	    if (any(unset_idx)) {
	      Sys.unsetenv(env_names[unset_idx])
	    }
	    set_values <- values[!unset_idx]
	    if (length(set_values) > 0L) {
	      do.call(Sys.setenv, as.list(set_values))
	    }
	    force(code)
	  }

	  extract_warmstart_env_overrides <- function(diag_row) {
	    if (is.null(diag_row) || nrow(diag_row) == 0L) {
	      return(character())
	    }
	    bool_map <- c(
	      GEOVERSA_AUTO_PATCH_PYRAMID = "patch_pyramid_active",
	      GEOVERSA_AUTO_PATCH_CONTEXT = "patch_context_active",
	      GEOVERSA_AUTO_TAB_QBASIS = "tab_qbasis_active",
	      GEOVERSA_AUTO_TAB_CROSSNET = "tab_crossnet_active",
	      GEOVERSA_AUTO_TAB_TREE = "tab_tree_active",
	      GEOVERSA_AUTO_TAB_RS = "tab_rs_active",
	      GEOVERSA_AUTO_CROSS_TAB_PATCH = "cross_tab_patch_active",
	      GEOVERSA_AUTO_COORD_RFF = "coord_rff_active",
	      GEOVERSA_AUTO_HETERO_NLL = "hetero_active",
	      GEOVERSA_AUTO_BETA_SIGMA_GATE = "beta_uncertainty_active",
	      GEOVERSA_AUTO_LOCAL_GATE = "local_gate_active",
	      GEOVERSA_AUTO_LOCAL_GATE_LEARNED = "learnable_gate_active",
	      GEOVERSA_AUTO_LOCAL_LINEAR = "local_linear_active",
	      GEOVERSA_AUTO_K_MULTISCALE = "krig_multiscale",
	      GEOVERSA_AUTO_VRM = "vrm_active",
	      GEOVERSA_AUTO_CVRM = "cvrm_active",
		      GEOVERSA_AUTO_DAM = "dam_active",
		      GEOVERSA_AUTO_DESIGN_WEIGHTS = "dad_rk_active",
		      GEOVERSA_AUTO_PEDOMOE = "pedomoe_active",
		      GEOVERSA_AUTO_NKO = "nko_active",
	      GEOVERSA2_HYPERLOCAL = "hyperlocal_active",
	      GEOVERSA2_HYPERLOCAL_FREEZE_TRUNK = "hyperlocal_freeze_trunk",
	      GEOVERSA2_HYPERLOCAL_WARMUP_BYPASS = "hyperlocal_warmup_bypass",
	      GEOVERSA2_HYPERLOCAL_FAITHFUL = "hyperlocal_faithful_mode"
	    )
	    num_map <- c(
	      GEOVERSA_AUTO_TAB_QBASIS_KNOTS = "tab_qbasis_knots",
	      GEOVERSA_AUTO_TAB_CROSS_LAYERS = "tab_cross_layers",
	      GEOVERSA_AUTO_TAB_TREE_NTREES = "tab_tree_num_trees",
	      GEOVERSA_AUTO_TAB_TREE_DEPTH = "tab_tree_depth",
	      GEOVERSA_AUTO_TAB_RS_HEADS = "tab_rs_num_heads",
	      GEOVERSA_AUTO_TAB_INPUT_DROPOUT = "tab_input_dropout",
	      GEOVERSA_AUTO_COORD_RFF_K = "coord_rff_num_freq",
	      GEOVERSA_AUTO_LINEAR_BLEND_CENTER = "linear_blend_center",
	      GEOVERSA_AUTO_LINEAR_BLEND_TEMP = "linear_blend_temp",
	      GEOVERSA_AUTO_LINEAR_RIDGE = "linear_ridge",
	      GEOVERSA_AUTO_LINEAR_DELTA_CLIP = "linear_delta_clip",
	      GEOVERSA_AUTO_GATE_ESS_WEIGHT = "gate_ess_weight",
	      GEOVERSA_AUTO_GATE_DIST_WEIGHT = "gate_dist_weight",
	      GEOVERSA_AUTO_GATE_SIGNAL_WEIGHT = "gate_signal_weight",
	      GEOVERSA_AUTO_GATE_HIDDEN = "gate_hidden",
	      GEOVERSA_AUTO_GATE_FLOOR = "gate_floor",
	      GEOVERSA_AUTO_K_SMALL_FRAC = "krig_multiscale_small_frac",
	      GEOVERSA_AUTO_K_MIX_TEMP = "krig_multiscale_temp",
	      GEOVERSA_AUTO_VRM_TOPK = "vrm_topk",
	      GEOVERSA_AUTO_VRM_ALPHA = "vrm_alpha",
	      GEOVERSA_AUTO_VRM_WEIGHT = "vrm_weight",
	      GEOVERSA_AUTO_VRM_RESID_WEIGHT = "vrm_resid_weight",
	      GEOVERSA_AUTO_VRM_RANK_DECAY = "vrm_rank_decay",
		      GEOVERSA_AUTO_CVRM_TAU_MULT = "cvrm_tau_mult",
		      GEOVERSA_AUTO_CVRM_DELTA_WEIGHT = "cvrm_delta_weight",
		      GEOVERSA_AUTO_DESIGN_WEIGHT_K = "dad_rk_k",
		      GEOVERSA_AUTO_DESIGN_WEIGHT_POWER = "dad_rk_power",
		      GEOVERSA_AUTO_DESIGN_WEIGHT_ENV_SHARE = "dad_rk_env_share",
		      GEOVERSA_AUTO_DESIGN_WEIGHT_CLIP_MIN = "dad_rk_clip_min",
		      GEOVERSA_AUTO_DESIGN_WEIGHT_CLIP_MAX = "dad_rk_clip_max",
		      GEOVERSA_AUTO_PEDOMOE_EXPERTS = "pedomoe_experts",
	      GEOVERSA_AUTO_PEDOMOE_GATE_HIDDEN = "pedomoe_gate_hidden",
	      GEOVERSA_AUTO_PEDOMOE_ENTROPY_TARGET = "pedomoe_entropy_target",
	      GEOVERSA_AUTO_NKO_HIDDEN = "nko_hidden",
	      GEOVERSA_AUTO_NKO_DROPOUT_OVERRIDE = "nko_dropout",
	      GEOVERSA_AUTO_NKO_GATE_FLOOR = "nko_gate_floor",
	      GEOVERSA2_HYPERLOCAL_BASIS_DIM = "hyperlocal_basis_dim",
	      GEOVERSA2_HYPERLOCAL_DROPOUT = "hyperlocal_dropout",
	      GEOVERSA2_HYPERLOCAL_SCALE_INIT = "hyperlocal_scale_init"
	    )
	    vals <- character()
	    for (env_name in names(bool_map)) {
	      val <- diag_bool_env(diag_row, bool_map[[env_name]])
	      if (!is.null(val)) vals[[env_name]] <- val
	    }
		    for (env_name in names(num_map)) {
		      val <- diag_num_env(diag_row, num_map[[env_name]])
		      if (!is.null(val)) vals[[env_name]] <- val
		    }
		    dad_mode <- diag_char_env(
		      diag_row, "dad_rk_mode",
		      allowed = c("hybrid", "spatial", "environmental", "off")
		    )
		    if (!is.null(dad_mode) && !identical(tolower(dad_mode), "off")) {
		      vals[["GEOVERSA_AUTO_DESIGN_WEIGHT_MODE"]] <- tolower(dad_mode)
		    }
		    hyperlocal_hidden <- suppressWarnings(as.numeric(diag_scalar(diag_row, "hyperlocal_hidden_dim")))
	    model_dim <- suppressWarnings(as.numeric(diag_scalar(diag_row, "d")))
	    if (is.finite(hyperlocal_hidden) && is.finite(model_dim) && model_dim > 0) {
	      vals[["GEOVERSA2_HYPERLOCAL_HIDDEN_FRAC"]] <- format(
	        max(0.20, min(1.00, hyperlocal_hidden / model_dim)),
	        scientific = FALSE,
	        trim = TRUE,
	        digits = 15
	      )
	    }
	    vals
	  }

	  out <- list(
    pred_rf = NULL,
    pred_rf_mean = NULL,
    pred_rf_sd = NULL,
    pred_rf_quantiles = NULL,
    pred_rf_quantile_levels = NULL,
    pred_rf_cal_mean = NULL,
    pred_rf_cal_sd = NULL,
    pred_rf_val_mean = NULL,
    pred_rf_val_sd = NULL,
    rf_model_label = resolve_rf_model_label(rf_refit_full_calibration, has_conformal_cal_split),
    pred_conv = NULL,
    pred_conv_base = NULL,
    pred_conv_val = NULL,
    pred_conv_val_base = NULL,
    pred_conv_test_scaled = NULL,
    pred_conv_test_logvar_scaled = NULL,
    pred_conv_cal_scaled = NULL,
    pred_conv_cal_logvar_scaled = NULL,
    pred_conv_val_scaled = NULL,
    pred_conv_val_logvar_scaled = NULL,
    pred_conv_target_scaler = NULL,
    pred_conv_target_transform = NA_character_,
    conv_diagnostics = NULL,
    pred_geoversa2 = NULL,
    pred_geoversa2_base = NULL,
    pred_geoversa2_val = NULL,
    pred_geoversa2_val_base = NULL,
    pred_geoversa2_test_scaled = NULL,
    pred_geoversa2_test_logvar_scaled = NULL,
    pred_geoversa2_target_scaler = NULL,
    pred_geoversa2_target_transform = NA_character_,
    geoversa2_diagnostics = NULL,
    geoversa2_model_label = "GeoVersa2-HyperLocal"
  )

  rf_model <- NULL   # kept in scope for the benchmark baseline only

  if ("RF" %in% models) {
    form_rf <- as.formula(
      paste(response_name_wadoux, "~", paste(predList_modelfull_wadoux, collapse = "+"))
    )
    rf_train_df <- if (rf_refit_full_calibration) {
      if (!is.null(full_calibration_df)) full_calibration_df else bind_rows(train_core, val_df)
    } else {
      train_core
    }
    rf_seed <- if (!is.null(selection_seed) && is.finite(selection_seed)) as.integer(selection_seed) else 123L
    rf_qrf_active <- parse_bool_env("WADOUX_RF_QRF", default = FALSE)
    if (rf_qrf_active) {
      rf_model <- fit_rf_qrf_wadoux(rf_train_df, form_rf, seed = rf_seed)
      rf_uq_eval <- predict_rf_qrf_uq(rf_model, eval_test_df)
      rf_preds <- split_rf_eval_outputs(rf_uq_eval)
      out$pred_rf <- rf_preds$pred_test_mean
      out$pred_rf_mean <- rf_preds$pred_test_mean
      out$pred_rf_sd <- rf_preds$pred_test_sd
      out$pred_rf_quantiles <- rf_preds$pred_test_q
      out$pred_rf_quantile_levels <- rf_uq_eval$q_levels
      out$pred_rf_cal_mean <- rf_preds$pred_cal_mean
      out$pred_rf_cal_sd <- rf_preds$pred_cal_sd
      if (!rf_refit_full_calibration && !has_conformal_cal_split) {
        rf_val_uq <- predict_rf_qrf_uq(rf_model, val_df, levels = rf_uq_eval$q_levels)
        out$pred_rf_val_mean <- rf_val_uq$mean
        out$pred_rf_val_sd <- rf_val_uq$sd
      }
    } else {
      rf_model <- fit_rf_default_wadoux(rf_train_df, form_rf)
      out$pred_rf <- predict_rf_default_wadoux(rf_model, test_df)
    }
  }

  if ("ConvKrigingNet2D" %in% models) {
    if (is.null(conv_env) || is.null(conv_params)) {
      stop("ConvKrigingNet2D requested but conv_env/conv_params were not provided.")
    }
    if (is.null(context)) {
      context <- conv_env$wadoux_context
    }
    committee_active <- identical(model_profile, "auto") &&
      parse_bool_env("GEOVERSA_AUTO_COMMITTEE", default = FALSE)
    geoversa2_hyperlocal_benchmark <- identical(model_profile, "auto") &&
      parse_bool_env("GEOVERSA2_HYPERLOCAL_BENCHMARK", default = FALSE)
    geoversa2_dbhe_benchmark <- identical(model_profile, "auto") &&
      parse_bool_env("GEOVERSA2_DBHE_BENCHMARK", default = FALSE)
    geoversa2_trunk_ensemble_benchmark <- identical(model_profile, "auto") &&
      parse_bool_env("GEOVERSA2_TRUNK_ENSEMBLE_BENCHMARK", default = FALSE)
    geoversa2_leader_ensemble_benchmark <- identical(model_profile, "auto") &&
      parse_bool_env("GEOVERSA2_LEADER_ENSEMBLE_BENCHMARK", default = FALSE)
    if (sum(c(
      geoversa2_dbhe_benchmark,
      geoversa2_trunk_ensemble_benchmark,
      geoversa2_leader_ensemble_benchmark
    )) > 1L) {
      stop("Only one GeoVersa2 ensemble benchmark can be active at a time.")
    }
    if (geoversa2_dbhe_benchmark) {
      geoversa2_hyperlocal_benchmark <- TRUE
    }
    if (geoversa2_trunk_ensemble_benchmark) {
      geoversa2_hyperlocal_benchmark <- TRUE
    }
    if (geoversa2_hyperlocal_benchmark &&
        !parse_bool_env("GEOVERSA2_HYPERLOCAL", default = FALSE)) {
      stop("GEOVERSA2_HYPERLOCAL_BENCHMARK=TRUE requires GEOVERSA2_HYPERLOCAL=1.")
    }
    if (geoversa2_hyperlocal_benchmark &&
        !parse_bool_env("GEOVERSA2_HYPERLOCAL_FAITHFUL", default = FALSE)) {
      stop("GEOVERSA2_HYPERLOCAL_BENCHMARK=TRUE requires GEOVERSA2_HYPERLOCAL_FAITHFUL=1.")
    }
    if (!refit_full_calibration) {
      if (committee_active && (geoversa2_hyperlocal_benchmark || geoversa2_leader_ensemble_benchmark)) {
        cat("[GeoVersa2 benchmark] committee disabled because a GeoVersa2 benchmark is active.\n")
        committee_active <- FALSE
      }
      conv_out <- if (geoversa2_leader_ensemble_benchmark) {
        ensemble_out <- run_geoversa2_leader_ensemble_on_explicit_split(
          train_df = train_core,
          val_df = val_df,
          test_df = eval_test_df,
          params = conv_params
        )
        ensemble_preds <- split_eval_outputs(ensemble_out)
        out$geoversa2_model_label <- if (isTRUE(ensemble_out$isotonic_active)) {
          "GeoVersa2-LeaderEnsemble+Isotonic"
        } else {
          "GeoVersa2-LeaderEnsemble"
        }
        out$pred_geoversa2 <- ensemble_preds$pred_test
        out$pred_geoversa2_base <- ensemble_preds$pred_test_base
        out$pred_geoversa2_val <- ensemble_out$pred_val
        out$pred_geoversa2_val_base <- ensemble_out$pred_val_base
        out$pred_geoversa2_test_scaled <- ensemble_preds$pred_test_scaled
        out$pred_geoversa2_test_logvar_scaled <- ensemble_preds$pred_test_logvar_scaled
        out$pred_geoversa2_target_scaler <- ensemble_out$target_scaler
        out$pred_geoversa2_target_transform <- ensemble_out$target_transform
        out$geoversa2_diagnostics <- bind_diag_columns_unique(
          ensemble_out$diagnostics,
          data.frame(
            geoversa2_hyperlocal_benchmark = FALSE,
            geoversa2_dbhe_benchmark = FALSE,
            geoversa2_trunk_ensemble_benchmark = FALSE,
            geoversa2_leader_ensemble_benchmark = TRUE,
            geoversa2_trunk_source = "independent_stageA_per_member",
            geoversa2_leader_state_available = FALSE,
            geoversa2_leader_val_rmse = ensemble_out$member_val_rmse[1],
            geoversa2_leader_test_rmse = ensemble_out$member_test_rmse[1],
            stringsAsFactors = FALSE
          ),
          rhs_suffix = "_meta"
        )
        ensemble_out$baseline_out
      } else if (geoversa2_hyperlocal_benchmark) {
        prep <- prepare_conv_fd_on_explicit_split(
          train_df = train_core,
          val_df = val_df,
          test_df = eval_test_df,
          params = conv_params
        )
        leader_params <- modifyList(conv_params, list(return_state = TRUE))
        leader_out <- run_conv_on_prepared_split_with_env(
          prep,
          leader_params,
          env_overrides = list(
            GEOVERSA2_HYPERLOCAL = "0",
            GEOVERSA2_HYPERLOCAL_FAITHFUL = "0",
            GEOVERSA2_HYPERLOCAL_WARMUP_BYPASS = "0",
            GEOVERSA2_HYPERLOCAL_FREEZE_TRUNK = "0",
            GEOVERSA2_HYPERLOCAL_TRAIN_SCOPE = "head_only"
          )
        )
        if (is.null(leader_out$selected_state)) {
          stop("GeoVersa2 faithful benchmark requires leader return_state=TRUE, but no selected_state was returned.")
        }
        hyper_out <- if (geoversa2_dbhe_benchmark) {
          out$geoversa2_model_label <- "GeoVersa2-DBHE"
          run_geoversa2_dbhe_on_explicit_split(
            train_df = train_core,
            val_df = val_df,
            test_df = test_df,
            params = conv_params,
            leader_out = leader_out
          )
        } else if (geoversa2_trunk_ensemble_benchmark) {
          out$geoversa2_model_label <- "GeoVersa2-TrunkEnsemble"
          run_geoversa2_trunk_ensemble_on_explicit_split(
            train_df = train_core,
            val_df = val_df,
            test_df = test_df,
            params = conv_params
          )
        } else {
          out$geoversa2_model_label <- "GeoVersa2-HyperLocal"
          hyper_params <- modifyList(
            conv_params,
            list(
              return_state = FALSE,
              init_state = NULL,
              hyperlocal_trunk_state = leader_out$selected_state
            )
          )
          run_conv_on_prepared_split_with_env(
            prep,
            hyper_params,
            env_overrides = list(
              GEOVERSA2_HYPERLOCAL = "1",
              GEOVERSA2_HYPERLOCAL_FAITHFUL = "1",
              GEOVERSA2_HYPERLOCAL_WARMUP_BYPASS = "1",
              GEOVERSA2_HYPERLOCAL_FREEZE_TRUNK = "1",
              GEOVERSA2_HYPERLOCAL_TRAIN_SCOPE = "head_only"
            )
          )
        }
        hyper_preds <- split_eval_outputs(hyper_out)
        out$pred_geoversa2 <- hyper_preds$pred_test
        out$pred_geoversa2_base <- hyper_preds$pred_test_base
        out$pred_geoversa2_val <- hyper_out$pred_val
        out$pred_geoversa2_val_base <- hyper_out$pred_val_base
        out$pred_geoversa2_test_scaled <- hyper_preds$pred_test_scaled
        out$pred_geoversa2_test_logvar_scaled <- hyper_preds$pred_test_logvar_scaled
        out$pred_geoversa2_target_scaler <- hyper_out$target_scaler
        out$pred_geoversa2_target_transform <- hyper_out$target_transform
        out$geoversa2_diagnostics <- bind_diag_columns_unique(
          hyper_out$diagnostics,
          data.frame(
            geoversa2_hyperlocal_benchmark = !(geoversa2_dbhe_benchmark || geoversa2_trunk_ensemble_benchmark || geoversa2_leader_ensemble_benchmark),
            geoversa2_dbhe_benchmark = geoversa2_dbhe_benchmark,
            geoversa2_trunk_ensemble_benchmark = geoversa2_trunk_ensemble_benchmark,
            geoversa2_leader_ensemble_benchmark = geoversa2_leader_ensemble_benchmark,
            geoversa2_trunk_source = if (geoversa2_trunk_ensemble_benchmark) "independent_stageA_per_member" else "leader_same_fold",
            geoversa2_leader_state_available = TRUE,
            geoversa2_leader_val_rmse = wadoux_eval(obs = val_df[[response_name_wadoux]], pred = leader_out$pred_val)$RMSE,
            geoversa2_leader_test_rmse = wadoux_eval(obs = test_df[[response_name_wadoux]], pred = leader_out$pred_test)$RMSE,
            stringsAsFactors = FALSE
          ),
          rhs_suffix = "_meta"
        )
        leader_out
      } else if (committee_active) {
        run_conv_committee_on_explicit_split(
          train_df = train_core,
          val_df = val_df,
          test_df = eval_test_df,
          params = conv_params
        )
      } else {
        run_conv_on_explicit_split(
          train_df = train_core,
          val_df = val_df,
          test_df = eval_test_df,
          params = conv_params
        )
      }

      conv_preds <- split_eval_outputs(conv_out)
      out$pred_conv <- conv_preds$pred_test
      out$pred_conv_base <- conv_preds$pred_test_base
      out$pred_conv_val <- conv_out$pred_val
      out$pred_conv_val_base <- conv_out$pred_val_base
      out$pred_conv_test_scaled <- conv_preds$pred_test_scaled
      out$pred_conv_test_logvar_scaled <- conv_preds$pred_test_logvar_scaled
      out$pred_conv_cal_scaled <- conv_preds$pred_cal_scaled
      out$pred_conv_cal_logvar_scaled <- conv_preds$pred_cal_logvar_scaled
      # Val UQ for split-conformal calibration (val not in training, so exchange-
      # ability holds; NULL when isotonic calibrator is active, which disables conformal)
      out$pred_conv_val_scaled <- conv_out$pred_val_scaled
      out$pred_conv_val_logvar_scaled <- conv_out$pred_val_logvar_scaled
      out$pred_conv_target_scaler <- conv_out$target_scaler
      out$pred_conv_target_transform <- conv_out$target_transform
      out$conv_diagnostics <- conv_out$diagnostics
    } else {
      if (committee_active) {
        cat("[GeoVersa committee] refit_full_calibration=TRUE detected; committee disabled for this path.\n")
      }
      calibration_full <- if (!is.null(full_calibration_df)) full_calibration_df else bind_rows(train_core, val_df)
      canonical_preprocess_bundle <- build_canonical_preprocess_bundle(calibration_full)
      canonical_preprocess_active <- !is.null(canonical_preprocess_bundle)
      selection_target_scaler <- compute_target_scaler_for_training_df(train_core)
      refit_target_scaler <- compute_target_scaler_for_training_df(calibration_full)
      head_transport_active <- FALSE
      head_transport_ratio <- NA_real_
      head_transport_offset <- NA_real_
      head_transport_head_count <- 0L
      head_transport_logvar_offset <- NA_real_
      head_transport_logvar_count <- 0L
      selection_val_frac <- min(0.5, max(0.05, nrow(val_df) / max(1L, nrow(calibration_full))))
      selection_split_count <- as.integer(round(parse_num_env("WADOUX_REFIT_SELECTION_SPLITS", 3)))
      refit_warmstart <- parse_bool_env("WADOUX_REFIT_WARMSTART", default = TRUE)
      refit_warmstart_lr_mult <- parse_num_env("WADOUX_REFIT_WARMSTART_LR_MULT", 0.25)
      refit_warmstart_lr_mult <- max(0.05, min(1.0, refit_warmstart_lr_mult))
      refit_warmstart_epoch_mult <- parse_num_env("WADOUX_REFIT_WARMSTART_EPOCH_MULT", 0.50)
      refit_warmstart_epoch_mult <- max(0.20, min(1.0, refit_warmstart_epoch_mult))
      refit_warmstart_lr_mult_requested <- refit_warmstart_lr_mult
      refit_warmstart_epoch_mult_requested <- refit_warmstart_epoch_mult
      refit_warmstart_lr_mult_applied <- refit_warmstart_lr_mult
      refit_warmstart_epoch_mult_applied <- refit_warmstart_epoch_mult
      refit_consistency_backbone_recovery <- parse_bool_env("WADOUX_REFIT_CONSISTENCY_BACKBONE_RECOVERY", default = FALSE)
      refit_consistency_weight <- parse_num_env("WADOUX_REFIT_CONSISTENCY_WEIGHT", NA_real_)
      refit_consistency_tab_noise <- parse_num_env("WADOUX_REFIT_CONSISTENCY_TAB_NOISE", 0.02)
      refit_consistency_patch_noise <- parse_num_env("WADOUX_REFIT_CONSISTENCY_PATCH_NOISE", 0.01)
      refit_consistency_coord_noise <- parse_num_env("WADOUX_REFIT_CONSISTENCY_COORD_NOISE", 0.01)
      refit_anchor_backbone_recovery <- parse_bool_env("WADOUX_REFIT_ANCHOR_BACKBONE_RECOVERY", default = TRUE)
      refit_anchor_noisy_conservative <- parse_bool_env("WADOUX_REFIT_ANCHOR_NOISY_CONSERVATIVE", default = TRUE)
      refit_anchor_weight <- parse_num_env("WADOUX_REFIT_ANCHOR_WEIGHT", NA_real_)
      refit_predavg_backbone_recovery <- parse_bool_env("WADOUX_REFIT_PREDAVG_BACKBONE_RECOVERY", default = FALSE)
      refit_predavg_topk <- as.integer(round(parse_num_env("WADOUX_REFIT_PREDAVG_TOPK", 0)))
      if (!is.finite(refit_predavg_topk) || refit_predavg_topk < 1L) {
        refit_predavg_topk <- NA_integer_
      }
      refit_predavg_rel_tol <- parse_num_env("WADOUX_REFIT_PREDAVG_REL_TOL", 0.03)
      refit_predavg_rel_tol <- max(0.0, min(0.20, refit_predavg_rel_tol))
      refit_ckptavg_backbone_recovery <- parse_bool_env("WADOUX_REFIT_CKPTAVG_BACKBONE_RECOVERY", default = TRUE)
      refit_ckptavg_topk <- as.integer(round(parse_num_env("WADOUX_REFIT_CKPTAVG_TOPK", 0)))
      if (!is.finite(refit_ckptavg_topk) || refit_ckptavg_topk < 1L) {
        refit_ckptavg_topk <- NA_integer_
      }
      refit_ckptavg_rel_tol <- parse_num_env("WADOUX_REFIT_CKPTAVG_REL_TOL", 0.03)
      refit_ckptavg_rel_tol <- max(0.0, min(0.20, refit_ckptavg_rel_tol))
      refit_ema_backbone_recovery <- parse_bool_env("WADOUX_REFIT_EMA_BACKBONE_RECOVERY", default = TRUE)
      refit_ema_decay <- parse_num_env("WADOUX_REFIT_EMA_DECAY", 0.65)
      refit_ema_decay <- max(0.05, min(0.95, refit_ema_decay))
      refit_krig_only <- parse_bool_env("WADOUX_REFIT_KRIG_ONLY", default = FALSE)
      refit_krig_only_epoch_mult <- parse_num_env("WADOUX_REFIT_KRIG_ONLY_EPOCH_MULT", 0.50)
      refit_krig_only_epoch_mult <- max(0.20, min(1.50, refit_krig_only_epoch_mult))
      refit_krig_only_lr_mult <- parse_num_env("WADOUX_REFIT_KRIG_ONLY_LR_MULT", 1.0)
      refit_krig_only_lr_mult <- max(0.10, min(2.0, refit_krig_only_lr_mult))
      if (!is.finite(selection_split_count) || selection_split_count < 1L) {
        selection_split_count <- 1L
      }
      selection_seed_base <- if (is.null(selection_seed) || !is.finite(selection_seed)) {
        12345L
      } else {
        as.integer(selection_seed)
      }

	      selection_split_defs <- vector("list", selection_split_count)
	      selection_test_df <- if (nrow(val_df) > 0L) singleton_df(val_df) else singleton_df(train_core)
	      selection_split_defs[[1L]] <- list(
	        split_id = 1L,
	        source = "provided",
	        train = train_core,
	        val = val_df,
	        test = selection_test_df
	      )
	      if (selection_split_count > 1L) {
	        for (sid in 2:selection_split_count) {
	          extra_split <- make_inner_val_split_wadoux(
	            calibration_full,
	            val_frac = selection_val_frac,
	            seed = selection_seed_base + 1000L + sid
	          )
	          extra_test_df <- if (nrow(extra_split$val) > 0L) singleton_df(extra_split$val) else singleton_df(extra_split$train)
	          selection_split_defs[[sid]] <- list(
	            split_id = sid,
	            source = "resampled_full_calibration",
	            train = extra_split$train,
	            val = extra_split$val,
	            test = extra_test_df
	          )
		        }
		      }

		      selection_rf_baselines <- vector("list", length(selection_split_defs))
		      if (identical(geoversa_selection_objective, "paired_rf_delta")) {
		        for (sid in seq_along(selection_split_defs)) {
		          def <- selection_split_defs[[sid]]
		          rf_fit_i <- tryCatch(
		            fit_rf_default_wadoux(def$train, form_rf),
		            error = function(e) e
		          )
		          if (inherits(rf_fit_i, "error")) {
		            selection_rf_baselines[[sid]] <- list(
		              pred_val = NULL,
		              val_rmse = NA_real_,
		              error = conditionMessage(rf_fit_i)
		            )
		          } else {
		            pred_i <- tryCatch(
		              predict_rf_default_wadoux(rf_fit_i, def$val),
		              error = function(e) e
		            )
		            if (inherits(pred_i, "error")) {
		              selection_rf_baselines[[sid]] <- list(
		                pred_val = NULL,
		                val_rmse = NA_real_,
		                error = conditionMessage(pred_i)
		              )
		            } else {
		              pred_i <- as.numeric(pred_i)
		              selection_rf_baselines[[sid]] <- list(
		                pred_val = pred_i,
		                val_rmse = wadoux_eval(
		                  obs = def$val[[response_name_wadoux]],
		                  pred = pred_i
		                )$RMSE,
		                error = NA_character_
		              )
		            }
		          }
		        }
		        rf_rmse_text <- paste(vapply(selection_rf_baselines, function(x) {
		          if (!is.null(x) && is.finite(x$val_rmse)) {
		            sprintf("%.4f", x$val_rmse)
		          } else {
		            "NA"
		          }
		        }, character(1)), collapse = "|")
		        cat(sprintf(
		          "[GeoVersa AutoTune] paired-RF selection baseline RMSE by split: %s\n",
		          rf_rmse_text
		        ))
		      }

		      selection_params_base <- conv_params
	      if (canonical_preprocess_active) {
	        selection_params_base$canonical_preprocess_bundle <- canonical_preprocess_bundle
	      }

	      run_selection_runs_for_policy <- function(policy,
	                                               split_ids,
	                                               return_state_first = refit_warmstart) {
	        runs <- vector("list", length(split_ids))
	        for (ii in seq_along(split_ids)) {
	          sid <- split_ids[[ii]]
	          def <- selection_split_defs[[sid]]
	          params_i <- modifyList(
	            selection_params_base,
	            list(return_state = isTRUE(return_state_first) && sid == 1L)
	          )
	          out_i <- with_temp_env(
	            policy$env,
	            run_conv_on_explicit_split(
	              train_df = def$train,
	              val_df = def$val,
	              test_df = def$test,
	              params = params_i
	            )
	          )
	          runs[[ii]] <- list(
	            split_id = sid,
		            source = def$source,
		            out = out_i,
		            val_obs = def$val[[response_name_wadoux]],
		            rf_pred_val = if (!is.null(selection_rf_baselines[[sid]])) {
		              selection_rf_baselines[[sid]]$pred_val
		            } else NULL,
		            rf_val_rmse = if (!is.null(selection_rf_baselines[[sid]])) {
		              selection_rf_baselines[[sid]]$val_rmse
		            } else NA_real_,
		            rf_error = if (!is.null(selection_rf_baselines[[sid]])) {
		              selection_rf_baselines[[sid]]$error
		            } else NA_character_,
		            diag = coerce_single_diag_row(
	              out_i$diagnostics,
	              stage_label = sprintf("GeoVersa policy %s split %d", policy$label, sid)
	            )
	          )
	        }
	        runs
	      }

	      score_selection_runs <- function(runs) {
	        per_split <- lapply(runs, function(x) {
	          base_rmse <- if (!is.null(x$out$pred_val_base)) {
	            wadoux_eval(obs = x$val_obs, pred = x$out$pred_val_base)$RMSE
	          } else {
	            NA_real_
	          }
	          full_rmse <- if (!is.null(x$out$pred_val)) {
	            wadoux_eval(obs = x$val_obs, pred = x$out$pred_val)$RMSE
	          } else {
	            NA_real_
	          }
		          data.frame(
		            split_id = x$split_id,
		            source = x$source,
		            val_RMSE_base = base_rmse,
		            val_RMSE_full = full_rmse,
		            val_RMSE_rf = if (!is.null(x$rf_val_rmse)) {
		              x$rf_val_rmse
		            } else NA_real_,
		            delta_RMSE_full_vs_RF = if (!is.null(x$rf_val_rmse) &&
		                                        is.finite(x$rf_val_rmse)) {
		              full_rmse - x$rf_val_rmse
		            } else NA_real_,
		            delta_RMSE_base_vs_RF = if (!is.null(x$rf_val_rmse) &&
		                                        is.finite(x$rf_val_rmse)) {
		              base_rmse - x$rf_val_rmse
		            } else NA_real_,
		            val_RMSE_gain = base_rmse - full_rmse,
	            best_epoch_main = suppressWarnings(as.numeric(x$diag$best_epoch_main[1])),
	            main_epochs_done = suppressWarnings(as.numeric(x$diag$main_epochs_done[1])),
	            stringsAsFactors = FALSE
	          )
	        })
		        df <- bind_rows(per_split)
		        paired_active <- identical(geoversa_selection_objective,
		                                   "paired_rf_delta") &&
		          any(is.finite(df$delta_RMSE_full_vs_RF))
		        rmse_sd <- stats::sd(df$val_RMSE_full, na.rm = TRUE)
		        if (!is.finite(rmse_sd)) rmse_sd <- 0
		        delta_sd <- stats::sd(df$delta_RMSE_full_vs_RF, na.rm = TRUE)
		        if (!is.finite(delta_sd)) delta_sd <- 0
		        score_main <- if (isTRUE(paired_active)) {
		          stats::median(df$delta_RMSE_full_vs_RF, na.rm = TRUE) +
		            autotune_sd_weight * delta_sd
		        } else {
		          stats::median(df$val_RMSE_full, na.rm = TRUE)
		        }
		        list(
		          per_split = df,
		          score = score_main,
		          val_RMSE_mean = mean(df$val_RMSE_full, na.rm = TRUE),
		          val_RMSE_sd = rmse_sd,
		          val_RMSE_base_median = stats::median(df$val_RMSE_base, na.rm = TRUE),
		          val_RMSE_full_median = stats::median(df$val_RMSE_full, na.rm = TRUE),
		          val_RMSE_rf_median = stats::median(df$val_RMSE_rf, na.rm = TRUE),
		          delta_RMSE_full_vs_RF_median = stats::median(df$delta_RMSE_full_vs_RF, na.rm = TRUE),
		          delta_RMSE_full_vs_RF_sd = delta_sd,
		          delta_RMSE_base_vs_RF_median = stats::median(df$delta_RMSE_base_vs_RF, na.rm = TRUE),
		          val_RMSE_gain_median = stats::median(df$val_RMSE_gain, na.rm = TRUE),
	          best_epoch_main_median = stats::median(df$best_epoch_main, na.rm = TRUE),
	          main_epochs_done_median = stats::median(df$main_epochs_done, na.rm = TRUE)
	        )
	      }

	      autotune_version_raw <- trimws(tolower(Sys.getenv(
	        "WADOUX_GEOVERSA_AUTOTUNE_VERSION",
	        unset = Sys.getenv("WADOUX_GEOVERSA_AUTOTUNE", unset = "")
	      )))
	      policy_tune_requested <- parse_bool_env("WADOUX_GEOVERSA_POLICY_TUNE", default = FALSE)
	      policy_tune_active <- identical(model_profile, "auto") &&
	        (policy_tune_requested || nzchar(autotune_version_raw))
	      autotune_version <- if (nzchar(autotune_version_raw)) {
	        autotune_version_raw
	      } else if (policy_tune_active) {
	        "v2"
	      } else {
	        "off"
	      }
	      autotune_v2_active <- policy_tune_active &&
	        autotune_version %in% c("2", "v2", "autotune_v2", "geoversa_autotune_v2")
	      policy_set <- Sys.getenv("WADOUX_GEOVERSA_POLICY_SET", unset = "default")
	      policy_max_candidates <- parse_num_env("WADOUX_GEOVERSA_POLICY_MAX_CANDIDATES", Inf)
	      policy_inner_default <- if (autotune_v2_active) min(selection_split_count, 2L) else 1L
	      policy_inner_raw <- trimws(Sys.getenv("WADOUX_GEOVERSA_POLICY_INNER_SPLITS", unset = ""))
	      policy_inner_splits <- if (nzchar(policy_inner_raw)) {
	        as.integer(round(suppressWarnings(as.numeric(policy_inner_raw))))
	      } else {
	        policy_inner_default
	      }
	      if (!is.finite(policy_inner_splits) || policy_inner_splits < 1L) {
	        policy_inner_splits <- policy_inner_default
	      }
	      policy_inner_splits <- min(selection_split_count, policy_inner_splits)
	      autotune_topk <- as.integer(round(parse_num_env("WADOUX_GEOVERSA_AUTOTUNE_TOPK", 5L)))
	      if (!is.finite(autotune_topk) || autotune_topk < 1L) autotune_topk <- 1L
	      refit_policy_ensemble_top_k <- as.integer(round(parse_num_env("WADOUX_GEOVERSA_REFIT_POLICY_ENSEMBLE_TOP_K", 1L)))
	      if (!is.finite(refit_policy_ensemble_top_k) || refit_policy_ensemble_top_k < 1L) {
	        refit_policy_ensemble_top_k <- 1L
	      }
	      refit_policy_ensemble_top_k <- min(refit_policy_ensemble_top_k, 5L)
	      refit_policy_ensemble_weighting <- tolower(trimws(Sys.getenv(
	        "WADOUX_GEOVERSA_REFIT_POLICY_ENSEMBLE_WEIGHTING", unset = "softmax")))
	      if (!refit_policy_ensemble_weighting %in% c("softmax", "inverse", "uniform")) {
	        refit_policy_ensemble_weighting <- "softmax"
	      }
	      refit_policy_ensemble_temperature <- parse_num_env(
	        "WADOUX_GEOVERSA_REFIT_POLICY_ENSEMBLE_TEMPERATURE", 0.75)
	      if (!is.finite(refit_policy_ensemble_temperature) ||
	          refit_policy_ensemble_temperature <= 0) {
	        refit_policy_ensemble_temperature <- 0.75
	      }
	      refit_policy_ensemble_indices <- integer()
	      refit_policy_ensemble_scores <- numeric()
	      refit_policy_ensemble_weights <- numeric()
	      geoversa_refit_policy_weights <- function(scores) {
	        n <- length(scores)
	        if (n == 0L) return(numeric())
	        if (n == 1L) return(1)
	        scores <- as.numeric(scores)
	        finite <- is.finite(scores)
	        if (!any(finite)) return(rep(1 / n, n))
	        worst <- max(scores[finite], na.rm = TRUE)
	        scores[!finite] <- worst + abs(worst) + 1
	        if (identical(refit_policy_ensemble_weighting, "uniform")) {
	          w <- rep(1, n)
	        } else if (identical(refit_policy_ensemble_weighting, "inverse")) {
	          w <- 1 / pmax(scores - min(scores, na.rm = TRUE) + refit_policy_ensemble_temperature, 1e-8)
	        } else {
	          z <- -(scores - min(scores, na.rm = TRUE)) / refit_policy_ensemble_temperature
	          z <- pmax(z, -700)
	          w <- exp(z)
	        }
	        if (!any(is.finite(w)) || sum(w, na.rm = TRUE) <= 0) {
	          return(rep(1 / n, n))
	        }
	        w / sum(w, na.rm = TRUE)
	      }
		      autotune_racing <- parse_bool_env("WADOUX_GEOVERSA_AUTOTUNE_RACING", default = FALSE)
		      autotune_operator_set <- Sys.getenv("WADOUX_GEOVERSA_AUTOTUNE_OPERATOR_SET", unset = "default")
		      autotune_sd_weight <- parse_num_env("WADOUX_GEOVERSA_AUTOTUNE_SD_WEIGHT", 0.15)
		      if (!is.finite(autotune_sd_weight) || autotune_sd_weight < 0) autotune_sd_weight <- 0.15
		      autotune_equiv_abs <- parse_num_env("WADOUX_GEOVERSA_AUTOTUNE_EQUIV_ABS", 0)
		      if (!is.finite(autotune_equiv_abs) || autotune_equiv_abs < 0) autotune_equiv_abs <- 0
		      autotune_equiv_rel <- parse_num_env("WADOUX_GEOVERSA_AUTOTUNE_EQUIV_REL", 0)
		      if (!is.finite(autotune_equiv_rel) || autotune_equiv_rel < 0) autotune_equiv_rel <- 0
		      autotune_data_profile <- make_geoversa_data_profile(
		        calibration_full,
		        response_name = response_name_wadoux,
		        predictor_names = predList_modelfull_wadoux
	      )
	      selected_policy <- make_geoversa_policy_grid("default", max_candidates = 1L)[[1L]]
	      selected_policy_score <- NA_real_
	      selected_policy_score_margin <- NA_real_
	      selected_policy_eval_splits <- 0L
	      selected_operator_label <- "full"
	      selected_operator_alpha <- 1
	      selected_operator_score <- NA_real_
	      selected_operator_val_rmse <- NA_real_
	      selected_operator_gain <- NA_real_
	      noharm_policy_anchor_idx <- NA_integer_
	      noharm_policy_anchor_score <- NA_real_
		      noharm_policy_anchor_label <- NA_character_
		      noharm_policy_margin <- NA_real_
		      noharm_policy_delta_vs_anchor <- NA_real_
		      noharm_policy_applied <- FALSE
		      noharm_policy_reason <- if (isTRUE(geoversa_noharm_policy)) {
		        "policy_tune_inactive"
	      } else {
	        "disabled"
	      }
	      rf_noharm_policy_preselected_idx <- NA_integer_
	      rf_noharm_policy_preselected_score <- NA_real_
	      rf_noharm_policy_preselected_delta <- NA_real_
	      rf_noharm_policy_anchor_idx <- NA_integer_
	      rf_noharm_policy_anchor_score <- NA_real_
	      rf_noharm_policy_anchor_delta <- NA_real_
	      rf_noharm_policy_margin <- NA_real_
	      rf_noharm_policy_required_score <- NA_real_
	      rf_noharm_policy_failed <- FALSE
	      rf_noharm_policy_applied <- FALSE
	      rf_noharm_policy_reason <- if (isTRUE(geoversa_rf_noharm_policy)) {
	        "policy_tune_inactive"
	      } else {
	        "disabled"
	      }
	      policy_rows <- data.frame()

	      make_policy_result <- function(policy, runs) {
	        score <- score_selection_runs(runs)
	        if (!is.finite(score$score)) score$score <- Inf
	        operator <- NULL
	        if (autotune_v2_active) {
		          operator <- evaluate_geoversa_operator_candidates(
		            runs,
			          operator_set = autotune_operator_set,
			          sd_weight = autotune_sd_weight,
			          equiv_abs = autotune_equiv_abs,
			          equiv_rel = autotune_equiv_rel,
				          selection_objective = geoversa_selection_objective,
				          noharm_policy = geoversa_noharm_policy,
				          noharm_margin_abs = geoversa_noharm_margin_abs,
				          noharm_margin_rel = geoversa_noharm_margin_rel,
				          rf_noharm_policy = geoversa_rf_noharm_policy,
				          rf_noharm_margin_abs = geoversa_rf_noharm_margin_abs,
				          rf_noharm_margin_rel = geoversa_rf_noharm_margin_rel
				          )
	          score$policy_score <- operator$selected$selection_score[1]
	          score$operator_label <- operator$selected$operator_label[1]
	          score$operator_alpha <- operator$selected$operator_alpha[1]
	          score$operator_val_RMSE_median <- operator$selected$val_RMSE_median[1]
	          score$operator_gain_median <- operator$selected$val_RMSE_gain_vs_base_median[1]
	          score$operator_delta_RMSE_vs_RF_median <-
	            operator$selected$delta_RMSE_vs_RF_median[1]
	          score$operator_rf_noharm_policy <- operator$selected$rf_noharm_policy[1]
	          score$operator_rf_noharm_failed <- operator$selected$rf_noharm_failed[1]
	          score$operator_rf_noharm_applied <- operator$selected$rf_noharm_applied[1]
	          score$operator_rf_noharm_reason <- operator$selected$rf_noharm_reason[1]
	        } else {
		          score$policy_score <- score$score
		          score$operator_label <- "full"
		          score$operator_alpha <- 1
		          score$operator_val_RMSE_median <- score$val_RMSE_full_median
		          score$operator_gain_median <- score$val_RMSE_gain_median
		          score$operator_delta_RMSE_vs_RF_median <-
		            score$delta_RMSE_full_vs_RF_median
		          score$operator_rf_noharm_policy <- FALSE
		          score$operator_rf_noharm_failed <- FALSE
		          score$operator_rf_noharm_applied <- FALSE
		          score$operator_rf_noharm_reason <- "operator_tune_inactive"
	        }
	        list(policy = policy, runs = runs, score = score, operator = operator)
	      }

	      policy_result_row <- function(policy, result = NULL, error = NA_character_,
	                                    promoted = FALSE, selected = FALSE) {
	        has_error <- is.null(result) || inherits(result, "error") || !is.na(error)
	        if (has_error) {
	          return(data.frame(
	            iteration = policy_iteration,
	            scenario = scenario,
	            protocol = policy_protocol,
	            model = if (is.na(policy_model)) "GeoVersa_500" else policy_model,
	            autotune_version = if (policy_tune_active) autotune_version else "off",
	            candidate_id = policy$candidate_id,
	            candidate_label = policy$label,
	            policy_description = policy$description,
		            policy_env = env_string(policy$env),
		            policy_complexity = policy$complexity,
		            selection_objective = geoversa_selection_objective,
		            selection_metric = if (identical(geoversa_selection_objective, "paired_rf_delta")) {
		              "delta_RMSE_vs_RF_robust_policy_operator"
		            } else if (autotune_v2_active) "RMSE_robust_policy_operator" else "RMSE",
		            selection_score = Inf,
		            val_RMSE_mean = NA_real_,
		            val_RMSE_sd = NA_real_,
		            val_RMSE_base_median = NA_real_,
		            val_RMSE_full_median = NA_real_,
		            val_RMSE_rf_median = NA_real_,
		            delta_RMSE_full_vs_RF_median = NA_real_,
		            delta_RMSE_full_vs_RF_sd = NA_real_,
		            delta_RMSE_base_vs_RF_median = NA_real_,
		            val_RMSE_gain_median = NA_real_,
	            best_epoch_main_median = NA_real_,
	            main_epochs_done_median = NA_real_,
	            operator_selected_label = NA_character_,
	            operator_selected_alpha = NA_real_,
	            operator_selection_score = NA_real_,
	            operator_val_RMSE_median = NA_real_,
	            operator_gain_vs_base_median = NA_real_,
	            operator_delta_RMSE_vs_RF_median = NA_real_,
	            operator_rf_noharm_policy = NA,
	            operator_rf_noharm_failed = NA,
	            operator_rf_noharm_applied = NA,
	            operator_rf_noharm_reason = NA_character_,
	            inner_splits = policy_inner_splits,
	            inner_splits_evaluated = 0L,
	            selection_splits_total = selection_split_count,
	            autotune_promoted = promoted,
	            selected = selected,
	            error = error,
	            stringsAsFactors = FALSE
	          ))
	        }
	        score <- result$score
	        data.frame(
	          iteration = policy_iteration,
	          scenario = scenario,
	          protocol = policy_protocol,
	          model = if (is.na(policy_model)) "GeoVersa_500" else policy_model,
	          autotune_version = if (policy_tune_active) autotune_version else "off",
	          candidate_id = policy$candidate_id,
	          candidate_label = policy$label,
	          policy_description = policy$description,
		          policy_env = env_string(policy$env),
		          policy_complexity = policy$complexity,
		          selection_objective = geoversa_selection_objective,
		          selection_metric = if (identical(geoversa_selection_objective, "paired_rf_delta")) {
		            "delta_RMSE_vs_RF_robust_policy_operator"
		          } else if (autotune_v2_active) "RMSE_robust_policy_operator" else "RMSE",
		          selection_score = score$policy_score,
		          val_RMSE_mean = score$val_RMSE_mean,
		          val_RMSE_sd = score$val_RMSE_sd,
		          val_RMSE_base_median = score$val_RMSE_base_median,
		          val_RMSE_full_median = score$val_RMSE_full_median,
		          val_RMSE_rf_median = score$val_RMSE_rf_median,
		          delta_RMSE_full_vs_RF_median = score$delta_RMSE_full_vs_RF_median,
		          delta_RMSE_full_vs_RF_sd = score$delta_RMSE_full_vs_RF_sd,
		          delta_RMSE_base_vs_RF_median = score$delta_RMSE_base_vs_RF_median,
		          val_RMSE_gain_median = score$val_RMSE_gain_median,
	          best_epoch_main_median = score$best_epoch_main_median,
	          main_epochs_done_median = score$main_epochs_done_median,
	          operator_selected_label = score$operator_label,
	          operator_selected_alpha = score$operator_alpha,
	          operator_selection_score = score$policy_score,
	          operator_val_RMSE_median = score$operator_val_RMSE_median,
	          operator_gain_vs_base_median = score$operator_gain_median,
	          operator_delta_RMSE_vs_RF_median =
	            score$operator_delta_RMSE_vs_RF_median,
	          operator_rf_noharm_policy = score$operator_rf_noharm_policy,
	          operator_rf_noharm_failed = score$operator_rf_noharm_failed,
	          operator_rf_noharm_applied = score$operator_rf_noharm_applied,
	          operator_rf_noharm_reason = score$operator_rf_noharm_reason,
	          inner_splits = policy_inner_splits,
	          inner_splits_evaluated = length(result$runs),
	          selection_splits_total = selection_split_count,
	          autotune_promoted = promoted,
	          selected = selected,
	          error = NA_character_,
	          stringsAsFactors = FALSE
	        )
	      }

	      if (policy_tune_active) {
	        policy_grid <- if (autotune_v2_active) {
	          make_geoversa_autotune_v2_grid(
	            policy_set = policy_set,
	            max_candidates = policy_max_candidates,
	            data_profile = autotune_data_profile
	          )
	        } else {
	          make_geoversa_policy_grid(
	            policy_set = policy_set,
	            max_candidates = policy_max_candidates
	          )
	        }
		        cat(sprintf(
		          "[GeoVersa AutoTune %s] Evaluating %d policies (inner_splits=%d, set=%s, operator_set=%s, equiv_abs=%.4f, equiv_rel=%.4f)\n",
		          autotune_version,
		          length(policy_grid), policy_inner_splits, policy_set, autotune_operator_set,
		          autotune_equiv_abs, autotune_equiv_rel
		        ))
	        candidate_results <- vector("list", length(policy_grid))
	        candidate_errors <- rep(NA_character_, length(policy_grid))
	        promoted <- rep(FALSE, length(policy_grid))
	        first_error <- NA_character_
	        stage1_split_ids <- 1L
	        for (pi in seq_along(policy_grid)) {
	          policy <- policy_grid[[pi]]
	          cat(sprintf(
	            "[GeoVersa AutoTune %s] candidate %d/%d: %s | %s\n",
	            autotune_version, pi, length(policy_grid), policy$label, env_string(policy$env)
	          ))
	          candidate <- tryCatch({
	            runs <- run_selection_runs_for_policy(
	              policy = policy,
	              split_ids = stage1_split_ids,
	              return_state_first = refit_warmstart
	            )
	            make_policy_result(policy, runs)
	          }, error = function(e) e)
	          if (inherits(candidate, "error")) {
	            msg <- conditionMessage(candidate)
	            candidate_errors[[pi]] <- msg
	            if (is.na(first_error)) first_error <- msg
	            cat(sprintf("[GeoVersa AutoTune %s]   failed: %s\n", autotune_version, msg))
	          } else {
	            candidate_results[[pi]] <- candidate
	          }
	        }

	        valid_idx <- which(vapply(candidate_results, Negate(is.null), logical(1)))
	        if (length(valid_idx) == 0L) {
	          stop("GeoVersa AutoTune failed for all candidates. First error: ", first_error)
	        }
	        stage2_active <- autotune_v2_active && autotune_racing &&
	          policy_inner_splits > 1L && length(valid_idx) > 1L
		        final_pool <- valid_idx
		        if (stage2_active) {
		          stage1_scores <- vapply(valid_idx, function(i) candidate_results[[i]]$score$policy_score, numeric(1))
		          stage1_selection <- geoversa_select_with_equivalence(
		            score = stage1_scores,
		            complexity = vapply(valid_idx, function(i) policy_grid[[i]]$complexity, numeric(1)),
		            equiv_abs = autotune_equiv_abs,
		            equiv_rel = autotune_equiv_rel,
		            safety_rank = geoversa_operator_safety_rank(vapply(
		              valid_idx,
		              function(i) candidate_results[[i]]$score$operator_label,
		              character(1)
		            )),
		            instability = vapply(valid_idx, function(i) candidate_results[[i]]$score$val_RMSE_sd, numeric(1)),
		            gain = vapply(valid_idx, function(i) candidate_results[[i]]$score$operator_gain_median, numeric(1)),
		            candidate_idx = valid_idx
		          )
		          stage1_complexity <- vapply(valid_idx, function(i) policy_grid[[i]]$complexity, numeric(1))
		          stage1_operator_rank <- geoversa_operator_safety_rank(vapply(
		            valid_idx,
		            function(i) candidate_results[[i]]$score$operator_label,
		            character(1)
		          ))
		          stage1_sd <- vapply(valid_idx, function(i) candidate_results[[i]]$score$val_RMSE_sd, numeric(1))
		          stage1_gain <- vapply(valid_idx, function(i) candidate_results[[i]]$score$operator_gain_median, numeric(1))
		          raw_ord <- valid_idx[order(
		            stage1_scores,
		            stage1_complexity,
		            stage1_operator_rank,
		            stage1_sd,
		            -stage1_gain,
		            valid_idx,
		            na.last = TRUE
		          )]
		          raw_top_idx <- raw_ord[seq_len(min(length(raw_ord), autotune_topk))]
		          equiv_ord <- valid_idx[stage1_selection$ordered_pos]
		          equiv_top_idx <- equiv_ord[seq_len(min(length(equiv_ord), autotune_topk))]
		          top_idx <- unique(c(raw_top_idx, equiv_top_idx))
		          promoted[top_idx] <- TRUE
		          final_pool <- top_idx
		          cat(sprintf(
		            "[GeoVersa AutoTune v2] racing promotion: %s | raw_top=%s | equiv_top=%s | stage1_equiv_tol=%.4f\n",
		            paste(vapply(policy_grid[top_idx], `[[`, character(1), "label"), collapse = ", "),
		            paste(vapply(policy_grid[raw_top_idx], `[[`, character(1), "label"), collapse = ", "),
		            paste(vapply(policy_grid[equiv_top_idx], `[[`, character(1), "label"), collapse = ", "),
		            stage1_selection$equiv_tol
		          ))
	          extra_ids <- seq.int(2L, policy_inner_splits)
	          for (pi in top_idx) {
	            policy <- policy_grid[[pi]]
	            candidate <- tryCatch({
	              extra_runs <- run_selection_runs_for_policy(
	                policy = policy,
	                split_ids = extra_ids,
	                return_state_first = FALSE
	              )
	              runs <- c(candidate_results[[pi]]$runs, extra_runs)
	              make_policy_result(policy, runs)
	            }, error = function(e) e)
	            if (inherits(candidate, "error")) {
	              msg <- conditionMessage(candidate)
	              candidate_errors[[pi]] <- msg
	              if (is.na(first_error)) first_error <- msg
	              cat(sprintf("[GeoVersa AutoTune v2]   promoted candidate failed: %s | %s\n", policy$label, msg))
	            } else {
	              candidate_results[[pi]] <- candidate
	            }
	          }
	          final_pool <- final_pool[vapply(final_pool, function(i) !is.null(candidate_results[[i]]), logical(1))]
	          if (length(final_pool) == 0L) {
	            stop("GeoVersa AutoTune v2 failed for all promoted candidates. First error: ", first_error)
	          }
	        } else if (autotune_v2_active && policy_inner_splits > 1L) {
	          promoted[valid_idx] <- TRUE
	          final_pool <- valid_idx
	          cat(sprintf(
	            "[GeoVersa AutoTune v2] full inner evaluation: %s | inner_splits=%d | racing=off\n",
	            paste(vapply(policy_grid[valid_idx], `[[`, character(1), "label"), collapse = ", "),
	            policy_inner_splits
	          ))
	          extra_ids <- seq.int(2L, policy_inner_splits)
	          for (pi in valid_idx) {
	            policy <- policy_grid[[pi]]
	            candidate <- tryCatch({
	              extra_runs <- run_selection_runs_for_policy(
	                policy = policy,
	                split_ids = extra_ids,
	                return_state_first = FALSE
	              )
	              runs <- c(candidate_results[[pi]]$runs, extra_runs)
	              make_policy_result(policy, runs)
	            }, error = function(e) e)
	            if (inherits(candidate, "error")) {
	              msg <- conditionMessage(candidate)
	              candidate_errors[[pi]] <- msg
	              if (is.na(first_error)) first_error <- msg
	              cat(sprintf("[GeoVersa AutoTune v2]   full-eval candidate failed: %s | %s\n", policy$label, msg))
	            } else {
	              candidate_results[[pi]] <- candidate
	            }
	          }
	          final_pool <- final_pool[vapply(final_pool, function(i) !is.null(candidate_results[[i]]), logical(1))]
	          if (length(final_pool) == 0L) {
	            stop("GeoVersa AutoTune v2 failed for all full-eval candidates. First error: ", first_error)
	          }
		        } else {
		          promoted[valid_idx] <- TRUE
		        }

			        noharm_policy_anchor_idx <- NA_integer_
			        noharm_policy_anchor_score <- NA_real_
			        noharm_policy_anchor_label <- NA_character_
			        noharm_policy_margin <- NA_real_
			        noharm_policy_delta_vs_anchor <- NA_real_
			        noharm_policy_applied <- FALSE
			        noharm_policy_reason <- if (isTRUE(geoversa_noharm_policy)) {
			          "eligible"
			        } else {
			          "disabled"
			        }
			        rf_noharm_policy_preselected_idx <- NA_integer_
			        rf_noharm_policy_preselected_score <- NA_real_
			        rf_noharm_policy_preselected_delta <- NA_real_
			        rf_noharm_policy_anchor_idx <- NA_integer_
			        rf_noharm_policy_anchor_score <- NA_real_
			        rf_noharm_policy_anchor_delta <- NA_real_
			        rf_noharm_policy_margin <- NA_real_
			        rf_noharm_policy_required_score <- NA_real_
			        rf_noharm_policy_failed <- FALSE
			        rf_noharm_policy_applied <- FALSE
			        rf_noharm_policy_reason <- if (isTRUE(geoversa_rf_noharm_policy)) {
			          "eligible"
			        } else {
			          "disabled"
			        }
			        if ((isTRUE(geoversa_noharm_policy) ||
			             isTRUE(geoversa_rf_noharm_policy)) &&
			            length(valid_idx) > 0L) {
			          valid_complexity <- vapply(valid_idx, function(i) {
			            policy_grid[[i]]$complexity
			          }, numeric(1))
			          noharm_policy_anchor_idx <- valid_idx[order(
			            valid_complexity, valid_idx, na.last = TRUE)[1L]]
			          if (length(noharm_policy_anchor_idx) == 1L &&
			              is.finite(noharm_policy_anchor_idx) &&
			              !(noharm_policy_anchor_idx %in% final_pool)) {
			            policy <- policy_grid[[noharm_policy_anchor_idx]]
			            existing_runs <- candidate_results[[noharm_policy_anchor_idx]]$runs
			            existing_ids <- vapply(existing_runs, `[[`, integer(1), "split_id")
			            missing_ids <- setdiff(seq_len(policy_inner_splits), existing_ids)
			            if (length(missing_ids) > 0L) {
			              anchor_candidate <- tryCatch({
			                extra_runs <- run_selection_runs_for_policy(
			                  policy = policy,
			                  split_ids = missing_ids,
			                  return_state_first = FALSE
			                )
			                make_policy_result(policy, c(existing_runs, extra_runs))
			              }, error = function(e) e)
			              if (inherits(anchor_candidate, "error")) {
			                candidate_errors[[noharm_policy_anchor_idx]] <- conditionMessage(anchor_candidate)
			                noharm_policy_reason <- paste(
			                  "anchor_full_eval_failed",
			                  conditionMessage(anchor_candidate),
			                  sep = ": "
			                )
			              } else {
			                candidate_results[[noharm_policy_anchor_idx]] <- anchor_candidate
			                promoted[[noharm_policy_anchor_idx]] <- TRUE
			                final_pool <- unique(c(noharm_policy_anchor_idx, final_pool))
			              }
			            } else {
			              final_pool <- unique(c(noharm_policy_anchor_idx, final_pool))
			            }
			          }
			        }

			        final_pool_scores <- vapply(final_pool, function(i) candidate_results[[i]]$score$policy_score, numeric(1))
			        final_pool_complexity <- vapply(final_pool, function(i) policy_grid[[i]]$complexity, numeric(1))
		        final_pool_operator_rank <- geoversa_operator_safety_rank(vapply(
		          final_pool,
		          function(i) candidate_results[[i]]$score$operator_label,
		          character(1)
		        ))
		        final_pool_sd <- vapply(final_pool, function(i) candidate_results[[i]]$score$val_RMSE_sd, numeric(1))
		        final_pool_gain <- vapply(final_pool, function(i) candidate_results[[i]]$score$operator_gain_median, numeric(1))
		        policy_selection <- geoversa_select_with_equivalence(
		          score = final_pool_scores,
		          complexity = final_pool_complexity,
		          equiv_abs = autotune_equiv_abs,
		          equiv_rel = autotune_equiv_rel,
		          safety_rank = final_pool_operator_rank,
		          instability = final_pool_sd,
		          gain = final_pool_gain,
		          candidate_idx = final_pool
		        )
			        best_idx <- final_pool[[policy_selection$selected_pos]]
			        best_policy_result <- candidate_results[[best_idx]]
			        if (isTRUE(geoversa_noharm_policy) &&
			            is.finite(noharm_policy_anchor_idx) &&
			            noharm_policy_anchor_idx %in% final_pool &&
			            !identical(best_idx, noharm_policy_anchor_idx)) {
			          anchor_result <- candidate_results[[noharm_policy_anchor_idx]]
			          noharm_policy_anchor_score <- anchor_result$score$policy_score
			          noharm_policy_anchor_label <- anchor_result$policy$label
			          anchor_rmse <- anchor_result$score$operator_val_RMSE_median
			          if (!is.finite(anchor_rmse)) {
			            anchor_rmse <- abs(noharm_policy_anchor_score)
			          }
			          if (!is.finite(anchor_rmse)) anchor_rmse <- 0
			          noharm_policy_margin <- geoversa_noharm_margin_abs +
			            geoversa_noharm_margin_rel * abs(anchor_rmse)
			          noharm_policy_delta_vs_anchor <- noharm_policy_anchor_score -
			            best_policy_result$score$policy_score
			          if (!is.finite(noharm_policy_delta_vs_anchor) ||
			              noharm_policy_delta_vs_anchor < noharm_policy_margin) {
			            best_idx <- noharm_policy_anchor_idx
			            best_policy_result <- anchor_result
			            noharm_policy_applied <- TRUE
			            noharm_policy_reason <- sprintf(
			              "policy_gain_%.6f_below_margin_%.6f",
			              ifelse(is.finite(noharm_policy_delta_vs_anchor),
			                     noharm_policy_delta_vs_anchor, NA_real_),
			              noharm_policy_margin
			            )
			          } else {
			            noharm_policy_reason <- sprintf(
			              "policy_gain_%.6f_met_margin_%.6f",
			              noharm_policy_delta_vs_anchor,
			              noharm_policy_margin
			            )
			          }
			        } else if (isTRUE(geoversa_noharm_policy) &&
			                   identical(best_idx, noharm_policy_anchor_idx)) {
			          noharm_policy_anchor_score <- best_policy_result$score$policy_score
			          noharm_policy_anchor_label <- best_policy_result$policy$label
		          noharm_policy_delta_vs_anchor <- 0
		          noharm_policy_margin <- geoversa_noharm_margin_abs
		          noharm_policy_reason <- "selected_policy_is_anchor"
		        }
		        if (isTRUE(geoversa_rf_noharm_policy)) {
		          rf_noharm_policy_preselected_idx <- best_idx
		          rf_noharm_policy_preselected_score <-
		            best_policy_result$score$policy_score
		          rf_noharm_policy_preselected_delta <-
		            best_policy_result$score$operator_delta_RMSE_vs_RF_median
		          if (!is.finite(rf_noharm_policy_preselected_delta)) {
		            rf_noharm_policy_preselected_delta <-
		              best_policy_result$score$delta_RMSE_full_vs_RF_median
		          }
		          rf_noharm_policy_anchor_idx <- noharm_policy_anchor_idx
		          if (is.finite(rf_noharm_policy_anchor_idx) &&
		              rf_noharm_policy_anchor_idx %in% valid_idx &&
		              !is.null(candidate_results[[rf_noharm_policy_anchor_idx]])) {
		            anchor_result <- candidate_results[[rf_noharm_policy_anchor_idx]]
		            rf_noharm_policy_anchor_score <- anchor_result$score$policy_score
		            rf_noharm_policy_anchor_delta <-
		              anchor_result$score$operator_delta_RMSE_vs_RF_median
		            if (!is.finite(rf_noharm_policy_anchor_delta)) {
		              rf_noharm_policy_anchor_delta <-
		                anchor_result$score$delta_RMSE_full_vs_RF_median
		            }
		          }
		          rf_scale <- best_policy_result$score$val_RMSE_rf_median
		          if (!is.finite(rf_scale)) {
		            rf_scale <- best_policy_result$score$operator_val_RMSE_median
		          }
		          if (!is.finite(rf_scale)) rf_scale <- 0
		          rf_noharm_policy_margin <- geoversa_rf_noharm_margin_abs +
		            geoversa_rf_noharm_margin_rel * abs(rf_scale)
		          rf_noharm_policy_required_score <- -rf_noharm_policy_margin
		          if (is.finite(rf_noharm_policy_preselected_score)) {
		            if (rf_noharm_policy_preselected_score <=
		                rf_noharm_policy_required_score) {
		              rf_noharm_policy_reason <- sprintf(
		                "passed_rf_inner:score=%.6f<=required=%.6f",
		                rf_noharm_policy_preselected_score,
		                rf_noharm_policy_required_score
		              )
		            } else {
		              rf_noharm_policy_failed <- TRUE
		              if (is.finite(rf_noharm_policy_anchor_idx) &&
		                  rf_noharm_policy_anchor_idx %in% valid_idx &&
		                  !is.null(candidate_results[[rf_noharm_policy_anchor_idx]])) {
		                if (!identical(best_idx, rf_noharm_policy_anchor_idx)) {
		                  best_idx <- rf_noharm_policy_anchor_idx
		                  best_policy_result <- candidate_results[[best_idx]]
		                  rf_noharm_policy_applied <- TRUE
		                  rf_noharm_policy_reason <- sprintf(
		                    "fallback_to_anchor_policy:score=%.6f>required=%.6f",
		                    rf_noharm_policy_preselected_score,
		                    rf_noharm_policy_required_score
		                  )
		                } else {
		                  rf_noharm_policy_reason <- sprintf(
		                    "selected_anchor_still_fails_rf:score=%.6f>required=%.6f",
		                    rf_noharm_policy_preselected_score,
		                    rf_noharm_policy_required_score
		                  )
		                }
		              } else {
		                rf_noharm_policy_reason <- sprintf(
		                  "rf_inner_failed_no_anchor_policy:score=%.6f>required=%.6f",
		                  rf_noharm_policy_preselected_score,
		                  rf_noharm_policy_required_score
		                )
		              }
		            }
		          } else {
		            rf_noharm_policy_reason <- "no_valid_delta_RMSE_vs_RF"
		          }
		        }
		        selected_policy <- best_policy_result$policy
		        selected_policy_score <- best_policy_result$score$policy_score
	        selected_policy_eval_splits <- length(best_policy_result$runs)
	        row_list <- lapply(seq_along(policy_grid), function(pi) {
	          policy_result_row(
	            policy_grid[[pi]],
	            result = candidate_results[[pi]],
	            error = candidate_errors[[pi]],
	            promoted = promoted[[pi]],
	            selected = identical(pi, best_idx)
	          )
	        })
		        policy_rows <- bind_rows(row_list)
			        if (nrow(policy_rows) > 0L) {
			          policy_rows$selection_equiv_abs <- autotune_equiv_abs
			          policy_rows$selection_equiv_rel <- autotune_equiv_rel
			          policy_rows$selection_equiv_tol <- policy_selection$equiv_tol
			          policy_rows$selection_final_pool <- FALSE
			          policy_rows$selection_practical_delta <- NA_real_
			          policy_rows$selection_practically_equivalent <- FALSE
			          final_pool_ids <- vapply(policy_grid[final_pool], `[[`, numeric(1), "candidate_id")
			          final_pool_rows <- match(final_pool_ids, policy_rows$candidate_id)
			          final_pool_keep <- !is.na(final_pool_rows)
			          final_pool_rows <- final_pool_rows[final_pool_keep]
			          policy_rows$selection_final_pool[final_pool_rows] <- TRUE
				          policy_rows$selection_practical_delta[final_pool_rows] <- policy_selection$practical_delta[final_pool_keep]
				          policy_rows$selection_practically_equivalent[final_pool_rows] <- policy_selection$practically_equivalent[final_pool_keep]
				          policy_rows$noharm_policy <- geoversa_noharm_policy
				          policy_rows$noharm_policy_anchor_candidate_id <-
				            if (is.finite(noharm_policy_anchor_idx)) {
				              policy_grid[[noharm_policy_anchor_idx]]$candidate_id
				            } else NA_real_
				          policy_rows$noharm_policy_anchor_label <- noharm_policy_anchor_label
				          policy_rows$noharm_policy_anchor_score <- noharm_policy_anchor_score
				          policy_rows$noharm_policy_margin_abs <- geoversa_noharm_margin_abs
				          policy_rows$noharm_policy_margin_rel <- geoversa_noharm_margin_rel
				          policy_rows$noharm_policy_margin <- noharm_policy_margin
				          policy_rows$noharm_policy_delta_vs_anchor <-
				            noharm_policy_delta_vs_anchor
				          policy_rows$noharm_policy_applied <- noharm_policy_applied
				          policy_rows$noharm_policy_reason <- noharm_policy_reason
				          policy_rows$rf_noharm_policy <- geoversa_rf_noharm_policy
				          policy_rows$rf_noharm_policy_preselected_candidate_id <-
				            if (is.finite(rf_noharm_policy_preselected_idx)) {
				              policy_grid[[rf_noharm_policy_preselected_idx]]$candidate_id
				            } else NA_real_
				          policy_rows$rf_noharm_policy_preselected_label <-
				            if (is.finite(rf_noharm_policy_preselected_idx)) {
				              policy_grid[[rf_noharm_policy_preselected_idx]]$label
				            } else NA_character_
				          policy_rows$rf_noharm_policy_preselected_score <-
				            rf_noharm_policy_preselected_score
				          policy_rows$rf_noharm_policy_preselected_delta_RMSE_vs_RF <-
				            rf_noharm_policy_preselected_delta
				          policy_rows$rf_noharm_policy_anchor_candidate_id <-
				            if (is.finite(rf_noharm_policy_anchor_idx)) {
				              policy_grid[[rf_noharm_policy_anchor_idx]]$candidate_id
				            } else NA_real_
				          policy_rows$rf_noharm_policy_anchor_label <-
				            if (is.finite(rf_noharm_policy_anchor_idx)) {
				              policy_grid[[rf_noharm_policy_anchor_idx]]$label
				            } else NA_character_
				          policy_rows$rf_noharm_policy_anchor_score <-
				            rf_noharm_policy_anchor_score
				          policy_rows$rf_noharm_policy_anchor_delta_RMSE_vs_RF <-
				            rf_noharm_policy_anchor_delta
				          policy_rows$rf_noharm_policy_margin_abs <-
				            geoversa_rf_noharm_margin_abs
				          policy_rows$rf_noharm_policy_margin_rel <-
				            geoversa_rf_noharm_margin_rel
				          policy_rows$rf_noharm_policy_margin <- rf_noharm_policy_margin
				          policy_rows$rf_noharm_policy_required_score <-
				            rf_noharm_policy_required_score
				          policy_rows$rf_noharm_policy_failed <-
				            rf_noharm_policy_failed
				          policy_rows$rf_noharm_policy_applied <-
				            rf_noharm_policy_applied
				          policy_rows$rf_noharm_policy_restrict_ensemble <-
				            geoversa_rf_noharm_restrict_ensemble
				          policy_rows$rf_noharm_policy_reason <-
				            rf_noharm_policy_reason
				        }
		        if (nrow(policy_rows) > 0L) {
		          policy_rows$refit_ensemble_member <- FALSE
		          policy_rows$refit_ensemble_rank <- NA_integer_
		          policy_rows$refit_ensemble_weight <- NA_real_
			        }
			        if (refit_policy_ensemble_top_k > 1L) {
			          final_pool_ranked <- if (isTRUE(noharm_policy_applied) ||
			                                   (isTRUE(rf_noharm_policy_applied) &&
			                                    isTRUE(geoversa_rf_noharm_restrict_ensemble))) {
			            best_idx
			          } else {
			            final_pool[order(
			              final_pool_scores,
			              final_pool_complexity,
			              final_pool_operator_rank,
			              final_pool_sd,
			              -final_pool_gain,
			              final_pool,
			              na.last = TRUE
			            )]
			          }
		          refit_policy_ensemble_indices <- unique(c(
		            best_idx,
		            final_pool_ranked[seq_len(min(length(final_pool_ranked), refit_policy_ensemble_top_k))]
		          ))
		          refit_policy_ensemble_indices <- refit_policy_ensemble_indices[
		            vapply(refit_policy_ensemble_indices, function(i) !is.null(candidate_results[[i]]), logical(1))
		          ]
		          refit_policy_ensemble_indices <- refit_policy_ensemble_indices[
		            seq_len(min(length(refit_policy_ensemble_indices), refit_policy_ensemble_top_k))
		          ]
		          refit_policy_ensemble_scores <- vapply(
		            refit_policy_ensemble_indices,
		            function(i) candidate_results[[i]]$score$policy_score,
		            numeric(1)
		          )
		          refit_policy_ensemble_weights <- geoversa_refit_policy_weights(
		            refit_policy_ensemble_scores)
		          if (nrow(policy_rows) > 0L && length(refit_policy_ensemble_indices) > 0L) {
		            ensemble_ids <- vapply(
		              policy_grid[refit_policy_ensemble_indices], `[[`, numeric(1), "candidate_id")
		            ensemble_rows <- match(ensemble_ids, policy_rows$candidate_id)
		            keep_rows <- !is.na(ensemble_rows)
		            policy_rows$refit_ensemble_member[ensemble_rows[keep_rows]] <- TRUE
		            policy_rows$refit_ensemble_rank[ensemble_rows[keep_rows]] <- seq_along(ensemble_rows)[keep_rows]
		            policy_rows$refit_ensemble_weight[ensemble_rows[keep_rows]] <-
		              refit_policy_ensemble_weights[keep_rows]
		          }
		        }
		        finite_scores <- sort(policy_rows$selection_score[
		          is.finite(policy_rows$selection_score) & policy_rows$autotune_promoted
		        ])
	        if (length(finite_scores) >= 2L) {
	          selected_policy_score_margin <- finite_scores[[2L]] - finite_scores[[1L]]
	        }
		        cat(sprintf(
		          "[GeoVersa AutoTune %s] selected candidate %d: %s | score=%.4f | margin=%.4f | equiv_tol=%.4f | operator=%s α=%.3f\n",
		          autotune_version,
		          selected_policy$candidate_id,
		          selected_policy$label,
		          selected_policy_score,
		          selected_policy_score_margin,
		          policy_selection$equiv_tol,
		          best_policy_result$score$operator_label,
		          best_policy_result$score$operator_alpha
		        ))
	        selection_runs <- best_policy_result$runs
	        if (length(selection_runs) < selection_split_count) {
	          extra_ids <- seq.int(length(selection_runs) + 1L, selection_split_count)
	          extra_runs <- run_selection_runs_for_policy(
	            policy = selected_policy,
	            split_ids = extra_ids,
	            return_state_first = FALSE
	          )
	          selection_runs <- c(selection_runs, extra_runs)
	        }
	      } else {
	        selection_runs <- run_selection_runs_for_policy(
	          policy = selected_policy,
	          split_ids = seq_len(selection_split_count),
	          return_state_first = refit_warmstart
	        )
	      }

	      final_operator <- if (autotune_v2_active) {
		        evaluate_geoversa_operator_candidates(
		          selection_runs,
			          operator_set = autotune_operator_set,
			          sd_weight = autotune_sd_weight,
			          equiv_abs = autotune_equiv_abs,
			          equiv_rel = autotune_equiv_rel,
				          selection_objective = geoversa_selection_objective,
				          noharm_policy = geoversa_noharm_policy,
				          noharm_margin_abs = geoversa_noharm_margin_abs,
				          noharm_margin_rel = geoversa_noharm_margin_rel,
				          rf_noharm_policy = geoversa_rf_noharm_policy,
				          rf_noharm_margin_abs = geoversa_rf_noharm_margin_abs,
				          rf_noharm_margin_rel = geoversa_rf_noharm_margin_rel
				        )
		      } else {
		        evaluate_geoversa_operator_candidates(
		          selection_runs,
			          operator_set = "fixed_only",
			          sd_weight = 0,
			          equiv_abs = 0,
			          equiv_rel = 0,
				          selection_objective = geoversa_selection_objective,
				          noharm_policy = geoversa_noharm_policy,
				          noharm_margin_abs = geoversa_noharm_margin_abs,
				          noharm_margin_rel = geoversa_noharm_margin_rel,
				          rf_noharm_policy = geoversa_rf_noharm_policy,
				          rf_noharm_margin_abs = geoversa_rf_noharm_margin_abs,
				          rf_noharm_margin_rel = geoversa_rf_noharm_margin_rel
				        )
	      }
	      if (!autotune_v2_active) {
	        full_idx <- which(final_operator$rows$operator_label == "full")
	        if (length(full_idx) == 1L) {
	          final_operator$rows$selected <- FALSE
	          final_operator$rows$selected[full_idx] <- TRUE
	          final_operator$selected <- final_operator$rows[full_idx, , drop = FALSE]
	        }
	      }
	      selected_operator_label <- final_operator$selected$operator_label[1]
	      selected_operator_alpha <- final_operator$selected$operator_alpha[1]
	      selected_operator_score <- final_operator$selected$selection_score[1]
	      selected_operator_val_rmse <- final_operator$selected$val_RMSE_median[1]
	      selected_operator_gain <- final_operator$selected$val_RMSE_gain_vs_base_median[1]
	      if (autotune_v2_active) {
	        selected_policy_score <- selected_operator_score
	      }
	      if (autotune_v2_active && nrow(policy_rows) > 0L) {
	        policy_rows$selected <- policy_rows$candidate_id == selected_policy$candidate_id
	        selected_row <- which(policy_rows$selected)
	        if (length(selected_row) == 1L) {
	          policy_rows$operator_selected_label[selected_row] <- selected_operator_label
	          policy_rows$operator_selected_alpha[selected_row] <- selected_operator_alpha
	          policy_rows$operator_selection_score[selected_row] <- selected_operator_score
	          policy_rows$operator_val_RMSE_median[selected_row] <- selected_operator_val_rmse
	          policy_rows$operator_gain_vs_base_median[selected_row] <- selected_operator_gain
	          policy_rows$operator_delta_RMSE_vs_RF_median[selected_row] <-
	            final_operator$selected$delta_RMSE_vs_RF_median[1]
	          policy_rows$operator_rf_noharm_policy[selected_row] <-
	            final_operator$selected$rf_noharm_policy[1]
	          policy_rows$operator_rf_noharm_failed[selected_row] <-
	            final_operator$selected$rf_noharm_failed[1]
	          policy_rows$operator_rf_noharm_applied[selected_row] <-
	            final_operator$selected$rf_noharm_applied[1]
	          policy_rows$operator_rf_noharm_reason[selected_row] <-
	            final_operator$selected$rf_noharm_reason[1]
	          policy_rows$selection_score[selected_row] <- selected_operator_score
	        }
	      }
	      if (policy_tune_active && exists(".geoversa_policy_tuning_rows", inherits = TRUE)) {
	        .geoversa_policy_tuning_rows[[length(.geoversa_policy_tuning_rows) + 1L]] <<- policy_rows
	      }
	      if (autotune_v2_active && exists(".geoversa_operator_tuning_rows", inherits = TRUE)) {
	        op_rows <- final_operator$rows
	        op_rows <- cbind(
	          data.frame(
	            iteration = policy_iteration,
	            scenario = scenario,
	            protocol = policy_protocol,
	            model = if (is.na(policy_model)) "GeoVersa_500" else policy_model,
	            autotune_version = autotune_version,
	            candidate_id = selected_policy$candidate_id,
	            candidate_label = selected_policy$label,
	            stringsAsFactors = FALSE
	          ),
	          op_rows
	        )
	        .geoversa_operator_tuning_rows[[length(.geoversa_operator_tuning_rows) + 1L]] <<- op_rows
	      }

	      selection_out <- selection_runs[[1L]]$out
	      selected_policy_env <- selected_policy$env

		      selection_diags <- lapply(selection_runs, `[[`, "diag")
	      selection_val_rmse_gains <- vapply(
	        selection_runs,
	        function(x) {
	          if (is.null(x$out$pred_val) || is.null(x$out$pred_val_base) || is.null(x$val_obs)) {
	            return(NA_real_)
	          }
	          base_rmse <- wadoux_eval(obs = x$val_obs, pred = x$out$pred_val_base)$RMSE
	          operator_pred <- blend_predictions(
	            x$out$pred_val_base,
	            x$out$pred_val,
	            selected_operator_alpha
	          )
	          operator_rmse <- wadoux_eval(obs = x$val_obs, pred = operator_pred)$RMSE
	          as.numeric(base_rmse - operator_rmse)
	        },
	        numeric(1)
	      )
      selection_val_rmse_gain_median <- suppressWarnings(stats::median(selection_val_rmse_gains, na.rm = TRUE))
      if (!is.finite(selection_val_rmse_gain_median)) {
        selection_val_rmse_gain_median <- NA_real_
      }
	      refit_schedule <- aggregate_refit_schedule(
	        selection_diags = selection_diags,
	        n_refit_train = nrow(calibration_full)
	      )
	      refit_warmup_epochs <- refit_schedule$refit_warmup_epochs
	      refit_main_epochs <- refit_schedule$refit_main_epochs
	      refit_patience <- refit_schedule$refit_patience
	      refit_lr_patience <- refit_schedule$refit_lr_patience
	      refit_lr_decay <- refit_schedule$refit_lr_decay
	      refit_bank_refresh <- refit_schedule$refit_bank_refresh_every
	      cold_refit_warmup_epochs <- refit_warmup_epochs
	      cold_refit_main_epochs <- refit_main_epochs
	      cold_refit_patience <- refit_patience
	      cold_refit_lr_patience <- refit_lr_patience
	      cold_refit_lr_decay <- refit_lr_decay
	      cold_refit_bank_refresh <- refit_bank_refresh
	      refit_warmstart_used <- FALSE
	      refit_init_state <- NULL
	      refit_warmstart_lr <- NA_real_
	      refit_warmstart_shape_source <- "full_calibration_auto"
	      refit_state_mismatch_fallback <- FALSE
	      refit_state_mismatch_message <- NA_character_
	      refit_consistency_active <- FALSE
      refit_env_overrides <- selected_policy_env
      refit_architecture_lock_active <- FALSE
      refit_architecture_lock_n <- 0L
      refit_architecture_lock_keys <- NA_character_
      refit_anchor_active <- FALSE
      refit_predavg_active <- FALSE
      refit_ckptavg_active <- FALSE
      refit_ema_active <- FALSE
      refit_ema_start_epoch <- NA_integer_
      refit_krig_only_epochs <- 0L
      refit_krig_only_lr <- NA_real_

      if (refit_warmstart && !is.null(selection_out$selected_state)) {
        selection_anchor_diag <- selection_runs[[1L]]$diag
        adaptive_warmstart <- choose_adaptive_warmstart_policy(
          selection_diag = selection_anchor_diag,
          selection_val_rmse_gain = selection_val_rmse_gain_median,
          base_epoch_mult = refit_warmstart_epoch_mult,
          base_lr_mult = refit_warmstart_lr_mult,
          scheduled_main_epochs = refit_main_epochs
        )
        refit_warmstart_epoch_mult_applied <- suppressWarnings(as.numeric(adaptive_warmstart$epoch_mult))
        if (!is.finite(refit_warmstart_epoch_mult_applied) || refit_warmstart_epoch_mult_applied <= 0) {
          refit_warmstart_epoch_mult_applied <- refit_warmstart_epoch_mult
        }
        refit_warmstart_epoch_mult_applied <- max(0.20, min(1.0, refit_warmstart_epoch_mult_applied))
        refit_warmstart_lr_mult_applied <- suppressWarnings(as.numeric(adaptive_warmstart$lr_mult))
        if (!is.finite(refit_warmstart_lr_mult_applied) || refit_warmstart_lr_mult_applied <= 0) {
          refit_warmstart_lr_mult_applied <- refit_warmstart_lr_mult
        }
        refit_warmstart_lr_mult_applied <- max(0.05, min(1.0, refit_warmstart_lr_mult_applied))
        selection_lr_init <- suppressWarnings(as.numeric(selection_anchor_diag$lr_init[1]))
        selection_lr_final <- suppressWarnings(as.numeric(selection_anchor_diag$lr_final[1]))
        warmstart_lr_candidate <- selection_lr_init * refit_warmstart_lr_mult_applied
        if (!is.finite(warmstart_lr_candidate) || warmstart_lr_candidate <= 0) {
          warmstart_lr_candidate <- selection_lr_final
        } else if (is.finite(selection_lr_final) && selection_lr_final > 0) {
          warmstart_lr_candidate <- max(warmstart_lr_candidate, selection_lr_final)
        }

	        refit_shape_overrides <- extract_warmstart_shape_overrides(selection_anchor_diag)
	        refit_env_overrides <- merge_env_overrides(
	          selected_policy_env,
	          extract_warmstart_env_overrides(selection_anchor_diag)
	        )
        refit_architecture_lock_active <- length(refit_env_overrides) > 0L
        refit_architecture_lock_n <- length(refit_env_overrides)
        refit_architecture_lock_keys <- if (length(refit_env_overrides) > 0L) {
          paste(names(refit_env_overrides), collapse = ",")
        } else {
          NA_character_
        }
        refit_warmup_epochs <- 0L
        refit_main_epochs <- max(3L, as.integer(round(refit_main_epochs * refit_warmstart_epoch_mult_applied)))
        adaptive_min_main <- suppressWarnings(as.integer(adaptive_warmstart$min_main_epochs))
        if (is.finite(adaptive_min_main) && adaptive_min_main > 0L) {
          refit_main_epochs <- max(refit_main_epochs, adaptive_min_main)
        }
        transported_state <- transport_warmstart_head_state(
          state_dict = selection_out$selected_state,
          selection_scaler = selection_target_scaler,
          refit_scaler = refit_target_scaler
        )
        refit_init_state <- transported_state$state
        head_transport_active <- transported_state$active
        head_transport_ratio <- transported_state$ratio
        head_transport_offset <- transported_state$offset
        head_transport_head_count <- transported_state$head_count
        head_transport_logvar_offset <- transported_state$logvar_offset
        head_transport_logvar_count <- transported_state$logvar_count
        refit_warmstart_lr <- warmstart_lr_candidate
        refit_warmstart_shape_source <- "selection_model"
        refit_warmstart_used <- TRUE
        refit_regime_backbone_recovery <- identical(adaptive_warmstart$regime, "backbone_recovery")
        refit_regime_noisy_conservative <- identical(adaptive_warmstart$regime, "noisy_conservative")
        refit_consistency_active <- refit_consistency_backbone_recovery &&
          refit_regime_backbone_recovery
        refit_predavg_active <- !refit_consistency_active && refit_predavg_backbone_recovery &&
          refit_regime_backbone_recovery
        refit_anchor_active <- !refit_consistency_active && !refit_predavg_active && (
          (refit_anchor_backbone_recovery && refit_regime_backbone_recovery) ||
            (refit_anchor_noisy_conservative && refit_regime_noisy_conservative)
        )
        refit_ckptavg_active <- !refit_consistency_active && !refit_predavg_active && !refit_anchor_active &&
          refit_ckptavg_backbone_recovery &&
          refit_regime_backbone_recovery
        refit_ema_active <- !refit_consistency_active && !refit_predavg_active && !refit_anchor_active && !refit_ckptavg_active &&
          refit_ema_backbone_recovery &&
          refit_regime_backbone_recovery
        if (refit_ema_active) {
          refit_ema_start_epoch <- max(2L, as.integer(round(refit_main_epochs * 0.50)))
        }
        if (refit_krig_only) {
          refit_krig_only_epochs <- max(2L, as.integer(round(refit_main_epochs * refit_krig_only_epoch_mult)))
          refit_krig_only_lr <- refit_warmstart_lr * refit_krig_only_lr_mult
        }
      } else {
        refit_shape_overrides <- list()
        refit_env_overrides <- selected_policy_env
        adaptive_warmstart <- list(
          regime = "disabled",
          epoch_mult = refit_warmstart_epoch_mult,
          lr_mult = refit_warmstart_lr_mult,
          min_main_epochs = NA_integer_
        )
      }
      refit_architecture_lock_active <- length(refit_env_overrides) > 0L
      refit_architecture_lock_n <- length(refit_env_overrides)
      refit_architecture_lock_keys <- if (length(refit_env_overrides) > 0L) {
        paste(names(refit_env_overrides), collapse = ",")
      } else {
        NA_character_
      }

      cat(sprintf(
        "[GeoVersa refit] Selection aggregation: splits=%d | warmup=%d | main=%d | lr_patience=%d | lr_decay=%.3f | bank_refresh=%d\n",
        refit_schedule$n_selection_splits,
        refit_warmup_epochs,
        refit_main_epochs,
        refit_lr_patience,
        refit_lr_decay,
        refit_bank_refresh
      ))
      if (refit_warmstart_used) {
        cat(sprintf(
          "[GeoVersa refit] Warm-start active: lr=%.2e | epoch_mult=%.2f | regime_signal=%s | consistency=%s | predavg=%s | anchor=%s | ckptavg=%s | EMA=%s | shape_source=%s | head_transport=%s | krig_only=%s (%d ep @ %.2e)\n",
          refit_warmstart_lr,
          refit_warmstart_epoch_mult_applied,
          adaptive_warmstart$regime,
          if (refit_consistency_active) {
            sprintf(
              "on(w=%s,tab=%.3f,patch=%.3f,coord=%.3f)",
              if (is.finite(refit_consistency_weight) && refit_consistency_weight > 0) sprintf("%.3f", refit_consistency_weight) else "auto",
              refit_consistency_tab_noise,
              refit_consistency_patch_noise,
              refit_consistency_coord_noise
            )
          } else "off",
          if (refit_predavg_active) {
            sprintf("on(topk=%s,tol=%.2f)", if (is.na(refit_predavg_topk)) "auto" else as.character(refit_predavg_topk), refit_predavg_rel_tol)
          } else "off",
          if (refit_anchor_active) {
            if (is.finite(refit_anchor_weight) && refit_anchor_weight > 0) sprintf("on(%.3f)", refit_anchor_weight) else "on(auto)"
          } else "off",
          if (refit_ckptavg_active) "on" else "off",
          if (refit_ema_active) sprintf("on(start=%d,decay=%.2f)", refit_ema_start_epoch, refit_ema_decay) else "off",
          refit_warmstart_shape_source,
	          if (head_transport_active) sprintf("on(r=%.4f,off=%.4f,n=%d,logvar_n=%d)", head_transport_ratio, head_transport_offset, head_transport_head_count, head_transport_logvar_count) else "off",
          if (refit_krig_only && refit_krig_only_epochs > 0L) "on" else "off",
          refit_krig_only_epochs,
          if (is.finite(refit_krig_only_lr)) refit_krig_only_lr else NA_real_
        ))
        cat(sprintf(
          "[GeoVersa refit] Architecture lock from selection: %s (%d env overrides)\n",
          if (refit_architecture_lock_active) "on" else "off",
          refit_architecture_lock_n
        ))
      }

      refit_params <- modifyList(
        conv_params,
        list(
          warmup_epochs = refit_warmup_epochs,
          epochs = refit_main_epochs,
          patience = refit_patience,
          lr_patience = refit_lr_patience,
          lr_decay = refit_lr_decay,
          bank_refresh_every = refit_bank_refresh,
          refit_fixed_warmup_epochs = refit_warmup_epochs,
          refit_fixed_main_epochs = refit_main_epochs,
          refit_use_final_state = TRUE,
          init_state = refit_init_state,
          lr = if (refit_warmstart_used) refit_warmstart_lr else conv_params$lr,
          min_lr = if (refit_warmstart_used && is.finite(refit_warmstart_lr)) refit_warmstart_lr / 1000 else conv_params$min_lr,
          refit_consistency_active = refit_consistency_active,
          refit_consistency_weight = refit_consistency_weight,
          refit_consistency_tab_noise = refit_consistency_tab_noise,
          refit_consistency_patch_noise = refit_consistency_patch_noise,
          refit_consistency_coord_noise = refit_consistency_coord_noise,
          refit_anchor_active = refit_anchor_active,
          refit_anchor_weight = refit_anchor_weight,
          refit_predavg_active = refit_predavg_active,
          refit_predavg_topk = refit_predavg_topk,
          refit_predavg_rel_tol = refit_predavg_rel_tol,
          refit_ckptavg_active = refit_ckptavg_active,
          refit_ckptavg_topk = refit_ckptavg_topk,
          refit_ckptavg_rel_tol = refit_ckptavg_rel_tol,
          refit_ema_active = refit_ema_active,
          refit_ema_decay = refit_ema_decay,
          refit_ema_start_epoch = refit_ema_start_epoch,
          refit_krig_only_epochs = refit_krig_only_epochs,
          refit_krig_only_lr = refit_krig_only_lr,
          canonical_preprocess_bundle = canonical_preprocess_bundle
        )
      )
      refit_params <- modifyList(refit_params, refit_shape_overrides)

	      run_refit_once <- function(params) {
	        run_conv_on_explicit_split(
	          train_df = calibration_full,
	          val_df = singleton_df(calibration_full),
	          test_df = eval_test_df,
	          params = params
	        )
	      }
	      refit_out <- tryCatch(
	        with_temp_env(refit_env_overrides, run_refit_once(refit_params)),
	        error = function(e) {
	          msg <- conditionMessage(e)
	          state_load_problem <- grepl(
	            "state_dict|load_state_dict|Could not find|size mismatch|tensor size|shape",
	            msg,
	            ignore.case = TRUE
	          )
	          if (!refit_warmstart_used || !state_load_problem) {
	            stop(e)
	          }
	          cat(sprintf(
	            "[GeoVersa refit] Warm-start state incompatible with full-calibration architecture; retrying cold-start refit. Cause: %s\n",
	            msg
	          ))
	          refit_state_mismatch_fallback <<- TRUE
	          refit_state_mismatch_message <<- msg
		          refit_warmstart_used <<- FALSE
		          refit_architecture_lock_active <<- length(refit_env_overrides) > 0L
		          refit_architecture_lock_n <<- length(refit_env_overrides)
		          refit_architecture_lock_keys <<- if (length(refit_env_overrides) > 0L) {
		            paste(names(refit_env_overrides), collapse = ",")
		          } else {
		            NA_character_
		          }
		          refit_init_state <<- NULL
	          refit_warmstart_lr <<- NA_real_
	          refit_warmstart_shape_source <<- "full_calibration_auto_after_state_mismatch"
		          head_transport_active <<- FALSE
		          head_transport_ratio <<- NA_real_
		          head_transport_offset <<- NA_real_
		          head_transport_head_count <<- 0L
		          head_transport_logvar_offset <<- NA_real_
		          head_transport_logvar_count <<- 0L
	          refit_consistency_active <<- FALSE
	          refit_anchor_active <<- FALSE
	          refit_predavg_active <<- FALSE
	          refit_ckptavg_active <<- FALSE
	          refit_ema_active <<- FALSE
	          refit_ema_start_epoch <<- NA_integer_
	          refit_krig_only_epochs <<- 0L
	          refit_krig_only_lr <<- NA_real_
	          refit_warmup_epochs <<- cold_refit_warmup_epochs
	          refit_main_epochs <<- cold_refit_main_epochs
	          refit_patience <<- cold_refit_patience
	          refit_lr_patience <<- cold_refit_lr_patience
	          refit_lr_decay <<- cold_refit_lr_decay
	          refit_bank_refresh <<- cold_refit_bank_refresh
	          adaptive_warmstart$regime <<- "state_mismatch_fallback"
	          adaptive_warmstart$min_main_epochs <<- NA_integer_

	          cold_refit_params <- modifyList(
	            conv_params,
	            list(
	              warmup_epochs = cold_refit_warmup_epochs,
	              epochs = cold_refit_main_epochs,
	              patience = cold_refit_patience,
	              lr_patience = cold_refit_lr_patience,
	              lr_decay = cold_refit_lr_decay,
	              bank_refresh_every = cold_refit_bank_refresh,
	              refit_fixed_warmup_epochs = cold_refit_warmup_epochs,
	              refit_fixed_main_epochs = cold_refit_main_epochs,
	              refit_use_final_state = TRUE,
	              init_state = NULL,
	              lr = conv_params$lr,
	              min_lr = conv_params$min_lr,
	              refit_consistency_active = FALSE,
	              refit_anchor_active = FALSE,
	              refit_predavg_active = FALSE,
	              refit_ckptavg_active = FALSE,
	              refit_ema_active = FALSE,
	              refit_krig_only_epochs = 0L,
	              refit_krig_only_lr = NA_real_,
	              canonical_preprocess_bundle = canonical_preprocess_bundle
	            )
	          )
		          with_temp_env(refit_env_overrides, run_refit_once(cold_refit_params))
		        }
		      )

	      refit_preds <- split_eval_outputs(refit_out)
	      operator_alpha_safe <- clamp01(selected_operator_alpha)
	      if (!is.finite(operator_alpha_safe)) operator_alpha_safe <- 1
	      operator_is_full <- abs(operator_alpha_safe - 1) <= 1e-8
	      operator_apply <- function(base_pred, full_pred) {
	        if (is.null(full_pred)) return(NULL)
	        if (is.null(base_pred)) return(as.numeric(full_pred))
	        blend_predictions(base_pred, full_pred, operator_alpha_safe)
	      }
	      out$pred_conv_full <- refit_preds$pred_test
	      out$pred_conv_val_full <- selection_out$pred_val
	      out$pred_conv <- operator_apply(refit_preds$pred_test_base, refit_preds$pred_test)
	      out$pred_conv_base <- refit_preds$pred_test_base
	      out$pred_conv_val <- operator_apply(selection_out$pred_val_base, selection_out$pred_val)
	      out$pred_conv_val_base <- selection_out$pred_val_base
	      out$pred_conv_test_scaled <- if (operator_is_full) refit_preds$pred_test_scaled else NULL
	      out$pred_conv_test_logvar_scaled <- if (operator_is_full) refit_preds$pred_test_logvar_scaled else NULL
	      out$pred_conv_cal_scaled <- if (operator_is_full) refit_preds$pred_cal_scaled else NULL
	      out$pred_conv_cal_logvar_scaled <- if (operator_is_full) refit_preds$pred_cal_logvar_scaled else NULL
      out$pred_conv_target_scaler <- refit_out$target_scaler
      out$pred_conv_target_transform <- refit_out$target_transform
      out$conv_diagnostics <- bind_diag_columns_unique(
        refit_out$diagnostics,
        data.frame(
          refit_full_calibration = TRUE,
          refit_calibration_n = nrow(calibration_full),
          selection_train_n = nrow(train_core),
	          selection_val_n = nrow(val_df),
	          canonical_preprocess_bundle_active = canonical_preprocess_active,
	          canonical_preprocess_source_n = if (canonical_preprocess_active) canonical_preprocess_bundle$source_n else NA_integer_,
	          geoversa_policy_tune_active = policy_tune_active,
	          geoversa_policy_set = if (policy_tune_active) policy_set else "off",
		          geoversa_policy_inner_splits = if (policy_tune_active) policy_inner_splits else 0L,
		          geoversa_selection_objective = geoversa_selection_objective,
		          geoversa_noharm_policy = geoversa_noharm_policy,
		          geoversa_noharm_margin_abs = geoversa_noharm_margin_abs,
		          geoversa_noharm_margin_rel = geoversa_noharm_margin_rel,
		          geoversa_noharm_policy_anchor_id =
		            if (is.finite(noharm_policy_anchor_idx)) {
		              policy_grid[[noharm_policy_anchor_idx]]$candidate_id
		            } else NA_real_,
		          geoversa_noharm_policy_anchor_label = noharm_policy_anchor_label,
		          geoversa_noharm_policy_anchor_score = noharm_policy_anchor_score,
		          geoversa_noharm_policy_margin = noharm_policy_margin,
		          geoversa_noharm_policy_delta_vs_anchor =
		            noharm_policy_delta_vs_anchor,
		          geoversa_noharm_policy_applied = noharm_policy_applied,
		          geoversa_noharm_policy_reason = noharm_policy_reason,
		          geoversa_rf_noharm_policy = geoversa_rf_noharm_policy,
		          geoversa_rf_noharm_margin_abs = geoversa_rf_noharm_margin_abs,
		          geoversa_rf_noharm_margin_rel = geoversa_rf_noharm_margin_rel,
		          geoversa_rf_noharm_restrict_ensemble =
		            geoversa_rf_noharm_restrict_ensemble,
		          geoversa_rf_noharm_policy_preselected_id =
		            if (is.finite(rf_noharm_policy_preselected_idx)) {
		              policy_grid[[rf_noharm_policy_preselected_idx]]$candidate_id
		            } else NA_real_,
		          geoversa_rf_noharm_policy_preselected_label =
		            if (is.finite(rf_noharm_policy_preselected_idx)) {
		              policy_grid[[rf_noharm_policy_preselected_idx]]$label
		            } else NA_character_,
		          geoversa_rf_noharm_policy_preselected_score =
		            rf_noharm_policy_preselected_score,
		          geoversa_rf_noharm_policy_preselected_delta_RMSE_vs_RF =
		            rf_noharm_policy_preselected_delta,
		          geoversa_rf_noharm_policy_anchor_id =
		            if (is.finite(rf_noharm_policy_anchor_idx)) {
		              policy_grid[[rf_noharm_policy_anchor_idx]]$candidate_id
		            } else NA_real_,
		          geoversa_rf_noharm_policy_anchor_label =
		            if (is.finite(rf_noharm_policy_anchor_idx)) {
		              policy_grid[[rf_noharm_policy_anchor_idx]]$label
		            } else NA_character_,
		          geoversa_rf_noharm_policy_anchor_score =
		            rf_noharm_policy_anchor_score,
		          geoversa_rf_noharm_policy_anchor_delta_RMSE_vs_RF =
		            rf_noharm_policy_anchor_delta,
		          geoversa_rf_noharm_policy_margin = rf_noharm_policy_margin,
		          geoversa_rf_noharm_policy_required_score =
		            rf_noharm_policy_required_score,
		          geoversa_rf_noharm_policy_failed =
		            rf_noharm_policy_failed,
		          geoversa_rf_noharm_policy_applied =
		            rf_noharm_policy_applied,
		          geoversa_rf_noharm_policy_reason =
		            rf_noharm_policy_reason,
		          geoversa_policy_selected_id = selected_policy$candidate_id,
	          geoversa_policy_selected_label = selected_policy$label,
	          geoversa_policy_selected_env = env_string(selected_policy_env),
		          geoversa_policy_selected_score = selected_policy_score,
		          geoversa_policy_selected_margin = selected_policy_score_margin,
		          geoversa_policy_selected_eval_splits = selected_policy_eval_splits,
			          geoversa_autotune_version = if (policy_tune_active) autotune_version else "off",
		          geoversa_autotune_v2_active = autotune_v2_active,
			          geoversa_autotune_topk = if (autotune_v2_active) autotune_topk else NA_integer_,
			          geoversa_autotune_racing = if (autotune_v2_active) autotune_racing else NA,
			          geoversa_autotune_operator_set = if (autotune_v2_active) autotune_operator_set else "fixed_full",
			          geoversa_autotune_sd_weight = if (autotune_v2_active) autotune_sd_weight else NA_real_,
			          geoversa_autotune_equiv_abs = if (autotune_v2_active) autotune_equiv_abs else NA_real_,
			          geoversa_autotune_equiv_rel = if (autotune_v2_active) autotune_equiv_rel else NA_real_,
			          geoversa_operator_selected_label = selected_operator_label,
		          geoversa_operator_selected_alpha = operator_alpha_safe,
		          geoversa_operator_selection_score = selected_operator_score,
		          geoversa_operator_val_RMSE_median = selected_operator_val_rmse,
		          geoversa_operator_gain_vs_base_median = selected_operator_gain,
		          geoversa_operator_delta_RMSE_vs_RF_median =
		            final_operator$selected$delta_RMSE_vs_RF_median[1],
		          geoversa_operator_rf_noharm_policy =
		            final_operator$selected$rf_noharm_policy[1],
		          geoversa_operator_rf_noharm_failed =
		            final_operator$selected$rf_noharm_failed[1],
		          geoversa_operator_rf_noharm_applied =
		            final_operator$selected$rf_noharm_applied[1],
		          geoversa_operator_rf_noharm_reason =
		            final_operator$selected$rf_noharm_reason[1],
		          selection_splits_used = refit_schedule$n_selection_splits,
          selection_canonical_best_epoch_main = as.integer(selection_runs[[1L]]$diag$best_epoch_main[1]),
          selection_canonical_best_epoch_total = as.integer(selection_runs[[1L]]$diag$best_epoch_total[1]),
          selection_warmup_epochs_done = as.integer(round(stats::median(vapply(selection_diags, function(x) x$warmup_epochs_done[1], numeric(1)), na.rm = TRUE))),
          selection_main_epochs_done = as.integer(round(stats::median(vapply(selection_diags, function(x) x$main_epochs_done[1], numeric(1)), na.rm = TRUE))),
          selection_best_epoch_main = as.integer(round(stats::median(vapply(selection_diags, function(x) x$best_epoch_main[1], numeric(1)), na.rm = TRUE))),
          selection_best_epoch_total = as.integer(round(stats::median(vapply(selection_diags, function(x) x$best_epoch_total[1], numeric(1)), na.rm = TRUE))),
          selection_patience = refit_patience,
          selection_lr_patience = refit_lr_patience,
          selection_lr_decay = refit_lr_decay,
          selection_bank_refresh_every = refit_bank_refresh,
          selection_best_val_huber = as.numeric(stats::median(vapply(selection_diags, function(x) x$best_val_huber[1], numeric(1)), na.rm = TRUE)),
          selection_val_RMSE_gain = selection_val_rmse_gain_median,
          refit_epoch_rule = if (refit_schedule$n_selection_splits > 1L) "smoothed_multi_split_median" else "smoothed_single_split",
          refit_raw_best_epoch_main = refit_schedule$raw_best_epoch_main,
          refit_plateau_median_epoch_main = refit_schedule$plateau_median_epoch_main,
          refit_plateau_last_epoch_main = refit_schedule$plateau_last_epoch_main,
          refit_plateau_anchor_epoch = refit_schedule$plateau_anchor_epoch,
          refit_scaled_best_epoch_main = refit_schedule$scaled_best_epoch_main,
	          refit_buffered_best_epoch_main = refit_schedule$buffered_best_epoch_main,
	          refit_trailing_anchor_epoch = refit_schedule$trailing_anchor_epoch,
	          refit_epoch_floor = refit_schedule$refit_epoch_floor,
	          refit_epoch_floor_requested = refit_schedule$refit_epoch_floor_requested,
	          refit_selection_floor_frac = refit_schedule$refit_selection_floor_frac,
	          refit_size_scale = refit_schedule$refit_size_scale,
	          refit_scheduled_main_epochs_pre_warmstart = refit_schedule$refit_main_epochs,
	          refit_selected_main_epochs = refit_main_epochs,
          refit_selected_warmup_epochs = refit_warmup_epochs,
          refit_selected_lr_patience = refit_lr_patience,
          refit_selected_lr_decay = refit_lr_decay,
          refit_selected_bank_refresh = refit_bank_refresh,
          refit_warmstart = refit_warmstart_used,
          refit_architecture_lock_active = refit_architecture_lock_active,
          refit_architecture_lock_n = refit_architecture_lock_n,
          refit_architecture_lock_keys = refit_architecture_lock_keys,
          refit_warmstart_lr = refit_warmstart_lr,
          refit_warmstart_lr_mult = if (refit_warmstart_used) refit_warmstart_lr_mult_applied else NA_real_,
          refit_warmstart_epoch_mult = if (refit_warmstart_used) refit_warmstart_epoch_mult_applied else NA_real_,
          refit_warmstart_lr_mult_requested = if (refit_warmstart_used) refit_warmstart_lr_mult_requested else NA_real_,
          refit_warmstart_epoch_mult_requested = if (refit_warmstart_used) refit_warmstart_epoch_mult_requested else NA_real_,
          refit_warmstart_shape_source = refit_warmstart_shape_source,
          refit_warmstart_head_transport_active = head_transport_active,
          refit_warmstart_head_transport_ratio = head_transport_ratio,
		          refit_warmstart_head_transport_offset = head_transport_offset,
		          refit_warmstart_head_transport_head_count = head_transport_head_count,
		          refit_warmstart_head_transport_logvar_offset = head_transport_logvar_offset,
		          refit_warmstart_head_transport_logvar_count = head_transport_logvar_count,
	          refit_state_mismatch_fallback = refit_state_mismatch_fallback,
	          refit_state_mismatch_message = refit_state_mismatch_message,
	          selection_target_center = if (!is.null(selection_target_scaler)) selection_target_scaler$center else NA_real_,
          selection_target_scale = if (!is.null(selection_target_scaler)) selection_target_scaler$scale else NA_real_,
          refit_target_center = if (!is.null(refit_target_scaler)) refit_target_scaler$center else NA_real_,
          refit_target_scale = if (!is.null(refit_target_scaler)) refit_target_scaler$scale else NA_real_,
          refit_warmstart_regime = adaptive_warmstart$regime,
          refit_warmstart_min_main_epochs = if (is.finite(suppressWarnings(as.integer(adaptive_warmstart$min_main_epochs)))) {
            suppressWarnings(as.integer(adaptive_warmstart$min_main_epochs))
          } else {
            NA_integer_
          },
          refit_warmstart_schedule_mode = "plateau_base",
          refit_consistency_requested = refit_consistency_active,
          refit_consistency_requested_weight = if (refit_consistency_active && is.finite(refit_consistency_weight) && refit_consistency_weight > 0) refit_consistency_weight else NA_real_,
          refit_consistency_requested_tab_noise = if (refit_consistency_active) refit_consistency_tab_noise else NA_real_,
          refit_consistency_requested_patch_noise = if (refit_consistency_active) refit_consistency_patch_noise else NA_real_,
          refit_consistency_requested_coord_noise = if (refit_consistency_active) refit_consistency_coord_noise else NA_real_,
          refit_anchor_requested = refit_anchor_active,
          refit_anchor_requested_weight = if (refit_anchor_active && is.finite(refit_anchor_weight) && refit_anchor_weight > 0) refit_anchor_weight else NA_real_,
          refit_predavg_requested = refit_predavg_active,
          refit_predavg_requested_topk = if (refit_predavg_active) refit_predavg_topk else NA_integer_,
          refit_predavg_requested_rel_tol = if (refit_predavg_active) refit_predavg_rel_tol else NA_real_,
          refit_ckptavg_requested = refit_ckptavg_active,
          refit_ckptavg_requested_topk = if (refit_ckptavg_active) refit_ckptavg_topk else NA_integer_,
          refit_ckptavg_requested_rel_tol = if (refit_ckptavg_active) refit_ckptavg_rel_tol else NA_real_,
          refit_ema_requested = refit_ema_active,
          refit_ema_requested_decay = if (refit_ema_active) refit_ema_decay else NA_real_,
          refit_ema_requested_start_epoch = if (refit_ema_active) refit_ema_start_epoch else NA_integer_,
          refit_krig_only = refit_krig_only && refit_krig_only_epochs > 0L,
          refit_krig_only_epochs = refit_krig_only_epochs,
          refit_krig_only_lr = refit_krig_only_lr,
          selection_refit_main_epochs_sd = stats::sd(refit_schedule$per_split$refit_selected_main_epochs),
          stringsAsFactors = FALSE
        ),
        rhs_suffix = "_meta"
      )
      if (policy_tune_active &&
          length(refit_policy_ensemble_indices) > 1L &&
          refit_policy_ensemble_top_k > 1L) {
        member_pred <- list()
        member_pred[[1L]] <- list(
          candidate_idx = selected_policy$candidate_id,
          candidate_label = selected_policy$label,
          status = "ok",
          score = selected_policy_score,
          weight = NA_real_,
          operator_alpha = operator_alpha_safe,
          pred_test = out$pred_conv,
          pred_test_base = out$pred_conv_base,
          pred_test_full = out$pred_conv_full
        )
        selected_refit_idx <- match(
          selected_policy$candidate_id,
          vapply(policy_grid[refit_policy_ensemble_indices], `[[`, numeric(1), "candidate_id")
        )
        for (ii in seq_along(refit_policy_ensemble_indices)) {
          candidate_idx <- refit_policy_ensemble_indices[[ii]]
          candidate_result <- candidate_results[[candidate_idx]]
          candidate_policy <- candidate_result$policy
          if (identical(candidate_policy$candidate_id, selected_policy$candidate_id)) {
            next
          }
          candidate_alpha <- suppressWarnings(as.numeric(candidate_result$score$operator_alpha))
          if (!is.finite(candidate_alpha)) candidate_alpha <- 1
          candidate_alpha <- clamp01(candidate_alpha)
          member_status <- "ok"
          member_out <- tryCatch({
            nested_env <- merge_env_overrides(
              candidate_policy$env,
              c(
                WADOUX_GEOVERSA_POLICY_TUNE = "FALSE",
                WADOUX_GEOVERSA_AUTOTUNE = "__UNSET__",
                WADOUX_GEOVERSA_AUTOTUNE_VERSION = "__UNSET__",
                WADOUX_GEOVERSA_REFIT_POLICY_ENSEMBLE_TOP_K = "1"
              )
            )
            with_temp_env(
              nested_env,
              compare_rf_conv_on_explicit_split(
                train_core = train_core,
                val_df = val_df,
                cal_df = cal_df,
                test_df = test_df,
                models = "ConvKrigingNet2D",
                calibration_patch_cache = calibration_patch_cache,
                conv_env = conv_env,
                conv_params = conv_params,
                patch_size = patch_size,
                refit_full_calibration = TRUE,
                rf_refit_full_calibration = FALSE,
                full_calibration_df = calibration_full,
                selection_seed = selection_seed_base + 71000L + ii,
                policy_iteration = policy_iteration,
                policy_protocol = policy_protocol,
                policy_model = paste0(if (is.na(policy_model)) "GeoVersa_500" else policy_model, "_policy_ensemble")
              )
            )
          }, error = function(e) {
            member_status <<- conditionMessage(e)
            NULL
          })
          if (is.null(member_out) || is.null(member_out$pred_conv)) {
            cat(sprintf(
              "[GeoVersa refit policy ensemble] candidate %s failed: %s\n",
              candidate_policy$label, member_status
            ))
            next
          }
          pred_full <- if (!is.null(member_out$pred_conv_full)) {
            member_out$pred_conv_full
          } else {
            member_out$pred_conv
          }
          pred_base <- member_out$pred_conv_base
          pred_blend <- if (!is.null(pred_base)) {
            blend_predictions(pred_base, pred_full, candidate_alpha)
          } else {
            as.numeric(pred_full)
          }
          member_pred[[length(member_pred) + 1L]] <- list(
            candidate_idx = candidate_policy$candidate_id,
            candidate_label = candidate_policy$label,
            status = "ok",
            score = suppressWarnings(as.numeric(candidate_result$score$policy_score)),
            weight = NA_real_,
            operator_alpha = candidate_alpha,
            pred_test = pred_blend,
            pred_test_base = pred_base,
            pred_test_full = pred_full
          )
        }
        member_ids <- vapply(member_pred, `[[`, numeric(1), "candidate_idx")
        target_order <- vapply(
          policy_grid[refit_policy_ensemble_indices], `[[`, numeric(1), "candidate_id"
        )
        member_order <- match(target_order, member_ids)
        member_order <- member_order[!is.na(member_order)]
        member_pred <- member_pred[member_order]
        if (length(member_pred) > 1L) {
          member_scores <- vapply(member_pred, `[[`, numeric(1), "score")
          member_weights <- geoversa_refit_policy_weights(member_scores)
          weighted_member <- function(name) {
            mat <- do.call(cbind, lapply(member_pred, function(x) {
              as.numeric(x[[name]])
            }))
            as.numeric(mat %*% member_weights)
          }
          out$pred_conv <- weighted_member("pred_test")
          if (all(vapply(member_pred, function(x) !is.null(x$pred_test_base), logical(1)))) {
            out$pred_conv_base <- weighted_member("pred_test_base")
          }
          if (all(vapply(member_pred, function(x) !is.null(x$pred_test_full), logical(1)))) {
            out$pred_conv_full <- weighted_member("pred_test_full")
          }
          out$pred_conv_test_scaled <- NULL
          out$pred_conv_test_logvar_scaled <- NULL
          out$pred_conv_cal_scaled <- NULL
          out$pred_conv_cal_logvar_scaled <- NULL
          out$conv_diagnostics <- bind_diag_columns_unique(
            out$conv_diagnostics,
            data.frame(
              geoversa_refit_policy_ensemble_requested_top_k = refit_policy_ensemble_top_k,
              geoversa_refit_policy_ensemble_applied = TRUE,
              geoversa_refit_policy_ensemble_size = length(member_pred),
              geoversa_refit_policy_ensemble_weighting = refit_policy_ensemble_weighting,
              geoversa_refit_policy_ensemble_temperature = refit_policy_ensemble_temperature,
              geoversa_refit_policy_ensemble_candidate_ids = paste(member_ids[member_order], collapse = "|"),
              geoversa_refit_policy_ensemble_candidate_labels = paste(
                vapply(member_pred, `[[`, character(1), "candidate_label"), collapse = "|"),
              geoversa_refit_policy_ensemble_scores = paste(
                sprintf("%.8f", member_scores), collapse = "|"),
              geoversa_refit_policy_ensemble_weights = paste(
                sprintf("%.8f", member_weights), collapse = "|"),
              geoversa_refit_policy_ensemble_operator_alpha = paste(
                sprintf("%.8f", vapply(member_pred, `[[`, numeric(1), "operator_alpha")),
                collapse = "|"),
              stringsAsFactors = FALSE
            ),
            rhs_suffix = "_policy_ensemble"
          )
          cat(sprintf(
            "[GeoVersa refit policy ensemble] applied top-k=%d | members=%s | weights=%s\n",
            length(member_pred),
            paste(vapply(member_pred, `[[`, character(1), "candidate_label"), collapse = ","),
            paste(sprintf("%.3f", member_weights), collapse = ",")
          ))
        }
      } else if (policy_tune_active) {
        out$conv_diagnostics <- bind_diag_columns_unique(
          out$conv_diagnostics,
          data.frame(
            geoversa_refit_policy_ensemble_requested_top_k = refit_policy_ensemble_top_k,
            geoversa_refit_policy_ensemble_applied = FALSE,
            geoversa_refit_policy_ensemble_size = 1L,
            geoversa_refit_policy_ensemble_weighting = refit_policy_ensemble_weighting,
            geoversa_refit_policy_ensemble_temperature = refit_policy_ensemble_temperature,
            stringsAsFactors = FALSE
          ),
          rhs_suffix = "_policy_ensemble"
        )
      }
    }
  }

  out
}

empty_conv_diag_df <- function() {
  data.frame()
}

make_conv_diagnostic_row <- function(preds,
                                     val_obs,
                                     test_obs,
                                     protocol,
                                     split_id = 1L,
                                     model_label = "ConvKrigingNet2D",
                                     pred_key = "pred_conv",
                                     pred_base_key = "pred_conv_base",
                                     pred_val_key = "pred_conv_val",
                                     pred_val_base_key = "pred_conv_val_base",
                                     diag_key = "conv_diagnostics") {
  diag_df <- preds[[diag_key]]
  pred_test <- preds[[pred_key]]
  pred_test_base <- preds[[pred_base_key]]
  pred_val <- preds[[pred_val_key]]
  pred_val_base <- preds[[pred_val_base_key]]

  if (is.null(diag_df) ||
      is.null(pred_test) ||
      is.null(pred_test_base) ||
      is.null(pred_val) ||
      is.null(pred_val_base)) {
    return(empty_conv_diag_df())
  }

  val_base <- wadoux_eval(obs = val_obs, pred = pred_val_base)
  val_full <- wadoux_eval(obs = val_obs, pred = pred_val)
  test_base <- wadoux_eval(obs = test_obs, pred = pred_test_base)
  test_full <- wadoux_eval(obs = test_obs, pred = pred_test)

  cbind(
    data.frame(
      protocol = protocol,
      split_id = split_id,
      model = model_label,
      stringsAsFactors = FALSE
    ),
    diag_df,
    setNames(val_base, paste0("val_", names(val_base), "_base")),
    setNames(val_full, paste0("val_", names(val_full), "_full")),
    setNames(test_base, paste0("test_", names(test_base), "_base")),
    setNames(test_full, paste0("test_", names(test_full), "_full")),
    data.frame(
      val_RMSE_gain = val_base$RMSE - val_full$RMSE,
      val_r2_gain = val_full$r2 - val_base$r2,
      val_MEC_gain = val_full$MEC - val_base$MEC,
      test_RMSE_gain = test_base$RMSE - test_full$RMSE,
      test_r2_gain = test_full$r2 - test_base$r2,
      test_MEC_gain = test_full$MEC - test_base$MEC,
      stringsAsFactors = FALSE
    )
  )
}

make_all_conv_diagnostic_rows <- function(preds,
                                          val_obs,
                                          test_obs,
                                          protocol,
                                          split_id = 1L) {
  geoversa2_label <- if (!is.null(preds$geoversa2_model_label)) preds$geoversa2_model_label else "GeoVersa2-HyperLocal"
  rows <- list()
  rows[[length(rows) + 1L]] <- make_conv_diagnostic_row(
    preds = preds,
    val_obs = val_obs,
    test_obs = test_obs,
    protocol = protocol,
    split_id = split_id,
    model_label = "ConvKrigingNet2D",
    pred_key = "pred_conv",
    pred_base_key = "pred_conv_base",
    pred_val_key = "pred_conv_val",
    pred_val_base_key = "pred_conv_val_base",
    diag_key = "conv_diagnostics"
  )
  rows[[length(rows) + 1L]] <- make_conv_diagnostic_row(
    preds = preds,
    val_obs = val_obs,
    test_obs = test_obs,
    protocol = protocol,
    split_id = split_id,
    model_label = geoversa2_label,
    pred_key = "pred_geoversa2",
    pred_base_key = "pred_geoversa2_base",
    pred_val_key = "pred_geoversa2_val",
    pred_val_base_key = "pred_geoversa2_val_base",
    diag_key = "geoversa2_diagnostics"
  )
  bind_rows(rows)
}

make_metric_rows <- function(obs, preds, models, protocol) {
  rows <- list()
  geoversa2_label <- if (!is.null(preds$geoversa2_model_label)) preds$geoversa2_model_label else "GeoVersa2-HyperLocal"
  rf_model_label <- if (!is.null(preds$rf_model_label)) preds$rf_model_label else "RF"

  if ("RF" %in% models && !is.null(preds$pred_rf)) {
    rows[[length(rows) + 1L]] <- wadoux_eval(obs = obs, pred = preds$pred_rf) %>%
      mutate(model = rf_model_label)
  }

  if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
    rows[[length(rows) + 1L]] <- wadoux_eval(obs = obs, pred = preds$pred_conv) %>%
      mutate(model = "ConvKrigingNet2D")
  }
  if (!is.null(preds$pred_geoversa2)) {
    rows[[length(rows) + 1L]] <- wadoux_eval(obs = obs, pred = preds$pred_geoversa2) %>%
      mutate(model = geoversa2_label)
  }

  bind_rows(rows) %>% mutate(protocol = protocol)
}

row_id_conv <- function(df) {
  if ("..wadoux_row_id" %in% names(df)) {
    df$..wadoux_row_id
  } else {
    seq_len(nrow(df))
  }
}

coord_key_conv <- function(df) {
  if (!all(c("x", "y") %in% names(df))) {
    return(rep(NA_character_, nrow(df)))
  }
  paste0(round(as.numeric(df$x), 6), "|", round(as.numeric(df$y), 6))
}

sample_design_test_conv <- function(common_data, calibration_full, size) {
  pool <- common_data$s_df
  keep <- rep(TRUE, nrow(pool))
  if ("..wadoux_row_id" %in% names(pool) &&
      "..wadoux_row_id" %in% names(calibration_full)) {
    keep <- keep & !(pool$..wadoux_row_id %in% calibration_full$..wadoux_row_id)
  }
  pool_xy <- coord_key_conv(pool)
  cal_xy <- coord_key_conv(calibration_full)
  if (any(!is.na(pool_xy)) && any(!is.na(cal_xy))) {
    keep <- keep & !(pool_xy %in% cal_xy)
  }
  eligible <- pool[keep, , drop = FALSE]
  if (nrow(eligible) < size) {
    stop(sprintf(
      "DesignBased GeoVersa test pool has only %d eligible rows after excluding calibration rows; need %d.",
      nrow(eligible), size
    ))
  }
  sample_simple_random_rows_wadoux(eligible, size)
}

make_prediction_audit_conv <- function(test_df, pred_map, iter,
                                       protocol = "DesignBased") {
  obs <- test_df[[response_name_wadoux]]
  out <- data.frame(
    iteration = iter,
    scenario = scenario,
    protocol = protocol,
    split_role = "design_test",
    wadoux_row_id = row_id_conv(test_df),
    x = test_df$x,
    y = test_df$y,
    obs = obs,
    stringsAsFactors = FALSE
  )
  for (nm in names(pred_map)) {
    pred <- as.numeric(pred_map[[nm]])
    out[[paste0("pred_", nm)]] <- pred
    out[[paste0("resid_", nm)]] <- pred - obs
  }
  out
}

make_split_audit_conv <- function(calibration_full, inner, test_df, iter,
                                  protocol = "DesignBased") {
  mk <- function(df, role) {
    if (is.null(df) || nrow(df) == 0L) return(data.frame())
    data.frame(
      iteration = iter,
      scenario = scenario,
      protocol = protocol,
      split_role = role,
      wadoux_row_id = row_id_conv(df),
      x = df$x,
      y = df$y,
      obs = df[[response_name_wadoux]],
      stringsAsFactors = FALSE
    )
  }
  bind_rows(
    mk(calibration_full, "calibration_full"),
    mk(inner$train, "inner_train"),
    mk(inner$val, "inner_val"),
    mk(inner$cal, "inner_cal"),
    mk(test_df, "design_test")
  )
}

evaluate_design_based_protocol <- function(valuetable,
                                           common_data,
                                           models,
                                           calibration_patch_cache,
                                           conv_env,
                                           conv_params,
                                           sample_size,
                                           val_frac,
                                           conformal_third_split = FALSE,
	                                           conformal_cal_frac = val_frac,
	                                           iter_seed,
	                                           iter_index = NA_integer_,
	                                           patch_size = 15,
	                                           refit_full_calibration = FALSE,
	                                           rf_refit_full_calibration = refit_full_calibration) {
  val_srs <- sample_design_test_conv(common_data, valuetable, sample_size)
  inner <- make_inner_protocol_split_wadoux(
    valuetable,
    val_frac = val_frac,
    conformal_third_split = conformal_third_split,
    conformal_cal_frac = conformal_cal_frac,
    seed = iter_seed + 101L
  )
  preds <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    cal_df = inner$cal,
    test_df = val_srs,
    models = models,
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
    patch_size = patch_size,
	    refit_full_calibration = refit_full_calibration,
	    rf_refit_full_calibration = rf_refit_full_calibration,
	    full_calibration_df = valuetable,
	    selection_seed = iter_seed + 101L,
	    policy_iteration = iter_index,
	    policy_protocol = "DesignBased",
	    policy_model = if (refit_full_calibration) "GeoVersa_500" else "GeoVersa_400"
	  )

	  obs <- val_srs[[response_name_wadoux]]
  audit_pred_map <- list()
  if ("RF" %in% models && !is.null(preds$pred_rf)) {
    audit_pred_map[[if (!is.null(preds$rf_model_label)) preds$rf_model_label else "RF"]] <- preds$pred_rf
  }
  if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
    audit_pred_map$ConvKrigingNet2D <- preds$pred_conv
  }
  if (!is.null(preds$pred_geoversa2)) {
    audit_pred_map[[if (!is.null(preds$geoversa2_model_label)) preds$geoversa2_model_label else "GeoVersa2-HyperLocal"]] <- preds$pred_geoversa2
  }
	  uq_out <- make_uq_rows(
    obs = obs,
    preds = preds,
    models = models,
    protocol = "DesignBased",
    split_id = 1L,
    val_obs = inner$val[[response_name_wadoux]],
    calibration_obs = if (!is.null(inner$cal)) inner$cal[[response_name_wadoux]] else NULL
  )
  list(
    metrics = make_metric_rows(obs = obs, preds = preds, models = models, protocol = "DesignBased"),
    uq_metrics = uq_out$metrics,
    reliability = uq_out$reliability,
	    diagnostics = make_all_conv_diagnostic_rows(
	      preds = preds,
	      val_obs = inner$val[[response_name_wadoux]],
	      test_obs = obs,
	      protocol = "DesignBased",
	      split_id = 1L
	    ),
    prediction_audit = make_prediction_audit_conv(
      val_srs, audit_pred_map, iter = iter_index, protocol = "DesignBased"),
    split_audit = make_split_audit_conv(
      valuetable, inner, val_srs, iter = iter_index, protocol = "DesignBased")
	  )
	}

evaluate_design_based_protocol_paired <- function(valuetable,
                                                  common_data,
                                                  calibration_patch_cache,
                                                  conv_env,
                                                  conv_params,
                                                  sample_size,
                                                  val_frac,
                                                  conformal_third_split = FALSE,
	                                                  conformal_cal_frac = val_frac,
	                                                  iter_seed,
		                                                  iter_index = NA_integer_,
		                                                  patch_size = 15) {
  val_srs <- sample_design_test_conv(common_data, valuetable, sample_size)
  inner <- make_inner_protocol_split_wadoux(
    valuetable,
    val_frac = val_frac,
    conformal_third_split = conformal_third_split,
    conformal_cal_frac = conformal_cal_frac,
    seed = iter_seed + 101L
  )

  preds_400 <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    cal_df = inner$cal,
    test_df = val_srs,
    models = c("RF", "ConvKrigingNet2D"),
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
	    patch_size = patch_size,
	    refit_full_calibration = FALSE,
	    full_calibration_df = valuetable,
	    selection_seed = iter_seed + 101L,
	    policy_iteration = iter_index,
	    policy_protocol = "DesignBased",
	    policy_model = if (conformal_third_split) "GeoVersa_train_core" else "GeoVersa_400"
	  )

  preds_500 <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    cal_df = inner$cal,
    test_df = val_srs,
    models = c("RF", "ConvKrigingNet2D"),
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
	    patch_size = patch_size,
	    refit_full_calibration = TRUE,
	    full_calibration_df = valuetable,
	    selection_seed = iter_seed + 101L,
	    policy_iteration = iter_index,
	    policy_protocol = "DesignBased",
	    policy_model = if (conformal_third_split) "GeoVersa_train_plus_val" else "GeoVersa_500"
	  )

  obs <- val_srs[[response_name_wadoux]]
  pred_map <- list(
    preds_400$pred_rf,
    preds_400$pred_conv,
    preds_500$pred_rf,
    preds_500$pred_conv
  )
  names(pred_map) <- c(
    resolve_rf_model_label(FALSE, conformal_third_split),
    if (conformal_third_split) "GeoVersa_train_core" else "GeoVersa_400",
    resolve_rf_model_label(TRUE, conformal_third_split),
    if (conformal_third_split) "GeoVersa_train_plus_val" else "GeoVersa_500"
  )
	  metrics <- make_metric_rows_named(
    obs = obs,
    pred_map = pred_map,
    protocol = "DesignBased"
	  )
  prediction_audit <- make_prediction_audit_conv(
    val_srs, pred_map, iter = iter_index, protocol = "DesignBased")
  split_audit <- make_split_audit_conv(
    valuetable, inner, val_srs, iter = iter_index, protocol = "DesignBased")

  diagnostics <- bind_rows(
    rename_model_rows(
      make_all_conv_diagnostic_rows(
        preds = preds_400,
        val_obs = inner$val[[response_name_wadoux]],
        test_obs = obs,
        protocol = "DesignBased",
        split_id = 1L
      ),
      if (conformal_third_split) "GeoVersa_train_core" else "GeoVersa_400"
    ),
    rename_model_rows(
      make_all_conv_diagnostic_rows(
        preds = preds_500,
        val_obs = inner$val[[response_name_wadoux]],
        test_obs = obs,
        protocol = "DesignBased",
        split_id = 1L
      ),
      if (conformal_third_split) "GeoVersa_train_plus_val" else "GeoVersa_500"
    )
  )

  paired_deltas <- make_paired_delta_rows(metrics, protocol = "DesignBased")
  paired_meta <- data.frame(
    protocol = "DesignBased",
    calibration_n = nrow(valuetable),
    train_core_n = nrow(inner$train),
    val_n = nrow(inner$val),
    cal_n = if (!is.null(inner$cal)) nrow(inner$cal) else 0L,
    test_n = nrow(val_srs),
    stringsAsFactors = FALSE
  )

  list(
    metrics = metrics,
    uq_metrics = data.frame(),
    reliability = data.frame(),
	    diagnostics = diagnostics,
	    paired_deltas = paired_deltas,
	    paired_meta = paired_meta,
    prediction_audit = prediction_audit,
    split_audit = split_audit
	  )
	}

evaluate_random_kfold_protocol <- function(valuetable,
                                           models,
                                           calibration_patch_cache,
                                           conv_env,
                                           conv_params,
                                           random_k,
                                           val_frac,
                                           conformal_third_split = FALSE,
                                           conformal_cal_frac = val_frac,
                                           iter_seed,
                                           patch_size = 15,
                                           refit_full_calibration = FALSE,
                                           rf_refit_full_calibration = refit_full_calibration) {
  flds <- caret::createFolds(
    valuetable[[response_name_wadoux]],
    k = random_k,
    list = TRUE,
    returnTrain = FALSE
  )

  pred_rf_all <- c()
  pred_conv_all <- c()
  pred_geoversa2_all <- c()
  obs_all <- c()
  diag_rows <- vector("list", length(flds))
  uq_rows <- vector("list", length(flds))
  reliability_rows <- vector("list", length(flds))

  for (j in seq_along(flds)) {
    id <- flds[[j]]
    outer_train <- valuetable[-id, , drop = FALSE]
    outer_test <- valuetable[id, , drop = FALSE]
    inner <- make_inner_protocol_split_wadoux(
      outer_train,
      val_frac = val_frac,
      conformal_third_split = conformal_third_split,
      conformal_cal_frac = conformal_cal_frac,
      seed = iter_seed * 1000L + j
    )

    preds <- compare_rf_conv_on_explicit_split(
      train_core = inner$train,
      val_df = inner$val,
      cal_df = inner$cal,
      test_df = outer_test,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration,
      rf_refit_full_calibration = rf_refit_full_calibration,
      full_calibration_df = outer_train,
      selection_seed = iter_seed * 1000L + j
    )

    obs_all <- c(obs_all, outer_test[[response_name_wadoux]])
    if ("RF" %in% models && !is.null(preds$pred_rf)) {
      pred_rf_all <- c(pred_rf_all, preds$pred_rf)
    }
    if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
      pred_conv_all <- c(pred_conv_all, preds$pred_conv)
    }
    if (!is.null(preds$pred_geoversa2)) {
      pred_geoversa2_all <- c(pred_geoversa2_all, preds$pred_geoversa2)
    }
    uq_out <- make_uq_rows(
      obs = outer_test[[response_name_wadoux]],
      preds = preds,
      models = models,
      protocol = "RandomKFold",
      split_id = j,
      val_obs = inner$val[[response_name_wadoux]],
      calibration_obs = if (!is.null(inner$cal)) inner$cal[[response_name_wadoux]] else NULL
    )
    uq_rows[[j]] <- uq_out$metrics
    reliability_rows[[j]] <- uq_out$reliability
    diag_rows[[j]] <- make_all_conv_diagnostic_rows(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = outer_test[[response_name_wadoux]],
      protocol = "RandomKFold",
      split_id = j
    )
  }

  list(
    metrics = make_metric_rows(
      obs = obs_all,
      preds = list(
        pred_rf = pred_rf_all,
        rf_model_label = resolve_rf_model_label(rf_refit_full_calibration, conformal_third_split),
        pred_conv = pred_conv_all,
        pred_geoversa2 = pred_geoversa2_all
      ),
      models = models,
      protocol = "RandomKFold"
    ),
    uq_metrics = bind_rows(uq_rows),
    reliability = bind_rows(reliability_rows),
    diagnostics = bind_rows(diag_rows)
  )
}

evaluate_spatial_kfold_protocol <- function(valuetable,
                                            models,
                                            calibration_patch_cache,
                                            conv_env,
                                            conv_params,
                                            val_dist_km,
                                            val_frac,
                                            conformal_third_split = FALSE,
                                            conformal_cal_frac = val_frac,
                                            iter_seed,
                                            patch_size = 15,
                                            refit_full_calibration = FALSE,
                                            rf_refit_full_calibration = refit_full_calibration) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  groups <- sort(unique(spatial_folds))

  pred_rf_all <- c()
  pred_conv_all <- c()
  pred_geoversa2_all <- c()
  obs_all <- c()
  diag_rows <- vector("list", length(groups))
  uq_rows <- vector("list", length(groups))
  reliability_rows <- vector("list", length(groups))

  for (j in seq_along(groups)) {
    g <- groups[j]
    outer_train <- valuetable[spatial_folds != g, , drop = FALSE]
    outer_test <- valuetable[spatial_folds == g, , drop = FALSE]
    inner <- make_inner_protocol_split_wadoux(
      outer_train,
      val_frac = val_frac,
      conformal_third_split = conformal_third_split,
      conformal_cal_frac = conformal_cal_frac,
      seed = iter_seed * 1000L + j
    )

    preds <- compare_rf_conv_on_explicit_split(
      train_core = inner$train,
      val_df = inner$val,
      cal_df = inner$cal,
      test_df = outer_test,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration,
      rf_refit_full_calibration = rf_refit_full_calibration,
      full_calibration_df = outer_train,
      selection_seed = iter_seed * 1000L + j
    )

    obs_all <- c(obs_all, outer_test[[response_name_wadoux]])
    if ("RF" %in% models && !is.null(preds$pred_rf)) {
      pred_rf_all <- c(pred_rf_all, preds$pred_rf)
    }
    if ("ConvKrigingNet2D" %in% models && !is.null(preds$pred_conv)) {
      pred_conv_all <- c(pred_conv_all, preds$pred_conv)
    }
    if (!is.null(preds$pred_geoversa2)) {
      pred_geoversa2_all <- c(pred_geoversa2_all, preds$pred_geoversa2)
    }
    uq_out <- make_uq_rows(
      obs = outer_test[[response_name_wadoux]],
      preds = preds,
      models = models,
      protocol = "SpatialKFold",
      split_id = j,
      val_obs = inner$val[[response_name_wadoux]],
      calibration_obs = if (!is.null(inner$cal)) inner$cal[[response_name_wadoux]] else NULL
    )
    uq_rows[[j]] <- uq_out$metrics
    reliability_rows[[j]] <- uq_out$reliability
    diag_rows[[j]] <- make_all_conv_diagnostic_rows(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = outer_test[[response_name_wadoux]],
      protocol = "SpatialKFold",
      split_id = j
    )
  }

  list(
    metrics = make_metric_rows(
      obs = obs_all,
      preds = list(
        pred_rf = pred_rf_all,
        rf_model_label = resolve_rf_model_label(rf_refit_full_calibration, conformal_third_split),
        pred_conv = pred_conv_all,
        pred_geoversa2 = pred_geoversa2_all
      ),
      models = models,
      protocol = "SpatialKFold"
    ),
    uq_metrics = bind_rows(uq_rows),
    reliability = bind_rows(reliability_rows),
    diagnostics = bind_rows(diag_rows)
  )
}

evaluate_bloocv_protocol <- function(valuetable,
                                     models,
                                     calibration_patch_cache,
                                     conv_env,
                                     conv_params,
                                     val_dist_km,
                                     val_frac,
                                     conformal_third_split = FALSE,
                                     conformal_cal_frac = val_frac,
                                     iter_seed,
                                     nb_groups = 10,
                                     nb_test_pixels = 100,
                                     patch_size = 15,
                                     refit_full_calibration = FALSE,
                                     rf_refit_full_calibration = refit_full_calibration) {
  # BLOOCV trains one model per test pixel. With ConvKrigingNet2D this is very
  # expensive: nb_groups * nb_test_pixels complete neural network trainings.
  # Hard-cap nb_test_pixels at 10 when ConvKrigingNet2D is included to prevent
  # multi-hour runs. For RF-only benchmarks the cap does not apply.
  if ("ConvKrigingNet2D" %in% models && nb_test_pixels > 10) {
    cap <- 10L
    cat(sprintf(
      "[BLOOCV] ConvKrigingNet2D detected: capping nb_test_pixels %d -> %d to avoid %d+ model trainings.\n",
      nb_test_pixels, cap, nb_groups * nb_test_pixels
    ))
    nb_test_pixels <- cap
  }
  nb_iteration <- nb_groups * nb_test_pixels
  rows <- vector("list", nb_iteration)
  diag_rows <- vector("list", nb_iteration)
  uq_rows <- vector("list", nb_iteration)
  reliability_rows <- vector("list", nb_iteration)

  for (j in seq_len(nb_iteration)) {
    point_in_range <- FALSE

    while (!point_in_range) {
      id_focal <- sample.int(nrow(valuetable), size = 1)
      focal_point <- valuetable[id_focal, , drop = FALSE]
      training_tmp <- valuetable[-id_focal, , drop = FALSE]
      training_tmp <- exclude_by_radius_wadoux(training_tmp, focal_point, val_dist_km * 1000)

      if (nrow(training_tmp) < 10) next
      if (!point_within_predictor_range_wadoux(training_tmp, focal_point, predList_modelfull_wadoux)) next

      point_in_range <- TRUE
      outer_train <- training_tmp
      outer_test <- focal_point
    }

    inner <- make_inner_protocol_split_wadoux(
      outer_train,
      val_frac = val_frac,
      conformal_third_split = conformal_third_split,
      conformal_cal_frac = conformal_cal_frac,
      seed = iter_seed * 100000L + j
    )

    preds <- compare_rf_conv_on_explicit_split(
      train_core = inner$train,
      val_df = inner$val,
      cal_df = inner$cal,
      test_df = outer_test,
      models = models,
      calibration_patch_cache = calibration_patch_cache,
      conv_env = conv_env,
      conv_params = conv_params,
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration,
      rf_refit_full_calibration = rf_refit_full_calibration,
      full_calibration_df = outer_train,
      selection_seed = iter_seed * 100000L + j
    )
    if (!is.null(preds$geoversa2_model_label) && nzchar(preds$geoversa2_model_label[1])) {
      geoversa2_label <- preds$geoversa2_model_label[1]
    }

    rows[[j]] <- data.frame(
      task = j,
      obs = outer_test[[response_name_wadoux]],
      pred_rf = if ("RF" %in% models) preds$pred_rf else NA_real_,
      pred_conv = if ("ConvKrigingNet2D" %in% models) preds$pred_conv else NA_real_,
      pred_geoversa2 = if (!is.null(preds$pred_geoversa2)) preds$pred_geoversa2 else NA_real_
    )
    uq_out <- make_uq_rows(
      obs = outer_test[[response_name_wadoux]],
      preds = preds,
      models = models,
      protocol = "BLOOCV",
      split_id = j,
      val_obs = inner$val[[response_name_wadoux]],
      calibration_obs = if (!is.null(inner$cal)) inner$cal[[response_name_wadoux]] else NULL
    )
    uq_rows[[j]] <- uq_out$metrics
    reliability_rows[[j]] <- uq_out$reliability
    diag_rows[[j]] <- make_all_conv_diagnostic_rows(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = outer_test[[response_name_wadoux]],
      protocol = "BLOOCV",
      split_id = j
    )
  }

  tmp <- bind_rows(rows)
  tmp$group <- caret::createFolds(seq_len(nrow(tmp)), k = nb_groups, list = FALSE, returnTrain = FALSE)

  rf_group_metrics <- vector("list", max(tmp$group))
  conv_group_metrics <- vector("list", max(tmp$group))
  geoversa2_group_metrics <- vector("list", max(tmp$group))
  geoversa2_label <- "GeoVersa2-HyperLocal"

  for (g in seq_len(max(tmp$group))) {
    idx <- which(tmp$group == g)
    if ("RF" %in% models) {
      rf_group_metrics[[g]] <- wadoux_eval(obs = tmp$obs[idx], pred = tmp$pred_rf[idx])
    }
    if ("ConvKrigingNet2D" %in% models) {
      conv_group_metrics[[g]] <- wadoux_eval(obs = tmp$obs[idx], pred = tmp$pred_conv[idx])
    }
    if (all(!is.na(tmp$pred_geoversa2[idx]))) {
      geoversa2_group_metrics[[g]] <- wadoux_eval(obs = tmp$obs[idx], pred = tmp$pred_geoversa2[idx])
    }
  }

  rows_out <- list()
  if ("RF" %in% models) {
    rows_out[[length(rows_out) + 1L]] <- bind_rows(rf_group_metrics) %>%
      summarise(
        ME = mean(ME, na.rm = TRUE),
        RMSE = mean(RMSE, na.rm = TRUE),
        r2 = mean(r2, na.rm = TRUE),
        MEC = mean(MEC, na.rm = TRUE)
      ) %>%
      mutate(model = resolve_rf_model_label(rf_refit_full_calibration, conformal_third_split))
  }
  if ("ConvKrigingNet2D" %in% models) {
    rows_out[[length(rows_out) + 1L]] <- bind_rows(conv_group_metrics) %>%
      summarise(
        ME = mean(ME, na.rm = TRUE),
        RMSE = mean(RMSE, na.rm = TRUE),
        r2 = mean(r2, na.rm = TRUE),
        MEC = mean(MEC, na.rm = TRUE)
      ) %>%
      mutate(model = "ConvKrigingNet2D")
  }
  if (length(geoversa2_group_metrics) > 0L && any(vapply(geoversa2_group_metrics, Negate(is.null), logical(1)))) {
    rows_out[[length(rows_out) + 1L]] <- bind_rows(geoversa2_group_metrics) %>%
      summarise(
        ME = mean(ME, na.rm = TRUE),
        RMSE = mean(RMSE, na.rm = TRUE),
        r2 = mean(r2, na.rm = TRUE),
        MEC = mean(MEC, na.rm = TRUE)
      ) %>%
      mutate(model = geoversa2_label)
  }

  list(
    metrics = bind_rows(rows_out) %>% mutate(protocol = "BLOOCV"),
    uq_metrics = bind_rows(uq_rows),
    reliability = bind_rows(reliability_rows),
    diagnostics = bind_rows(diag_rows)
  )
}

evaluate_population_protocol_optional <- function(valuetable,
                                                  common_data,
                                                  models,
                                                  calibration_patch_cache,
                                                  conv_env,
                                                  conv_params,
                                                  val_frac,
                                                  conformal_third_split = FALSE,
                                                  conformal_cal_frac = val_frac,
                                                  iter_seed,
                                                  patch_size = 15,
                                                  refit_full_calibration = FALSE,
                                                  rf_refit_full_calibration = refit_full_calibration) {
  inner <- make_inner_protocol_split_wadoux(
    valuetable,
    val_frac = val_frac,
    conformal_third_split = conformal_third_split,
    conformal_cal_frac = conformal_cal_frac,
    seed = iter_seed + 101L
  )
  preds <- compare_rf_conv_on_explicit_split(
    train_core = inner$train,
    val_df = inner$val,
    cal_df = inner$cal,
    test_df = common_data$s_df,
    models = models,
    calibration_patch_cache = calibration_patch_cache,
    conv_env = conv_env,
    conv_params = conv_params,
    patch_size = patch_size,
    refit_full_calibration = refit_full_calibration,
    rf_refit_full_calibration = rf_refit_full_calibration,
    full_calibration_df = valuetable,
    selection_seed = iter_seed + 101L
  )
  obs <- common_data$s_df[[response_name_wadoux]]
  uq_out <- make_uq_rows(
    obs = obs,
    preds = preds,
    models = models,
    protocol = "Population",
    split_id = 1L,
    val_obs = inner$val[[response_name_wadoux]],
    calibration_obs = if (!is.null(inner$cal)) inner$cal[[response_name_wadoux]] else NULL
  )
  list(
    metrics = make_metric_rows(obs = obs, preds = preds, models = models, protocol = "Population"),
    uq_metrics = uq_out$metrics,
    reliability = uq_out$reliability,
    diagnostics = make_all_conv_diagnostic_rows(
      preds = preds,
      val_obs = inner$val[[response_name_wadoux]],
      test_obs = obs,
      protocol = "Population",
      split_id = 1L
    )
  )
}

scenario <- Sys.getenv("WADOUX_SCENARIO", unset = "random")
protocols <- parse_protocols_env()
models <- parse_models_env()
include_population <- parse_bool_env("WADOUX_INCLUDE_POPULATION", default = FALSE)
refit_full_calibration <- parse_bool_env("WADOUX_REFIT_FULL_CALIBRATION", default = FALSE)
rf_refit_full_calibration <- parse_bool_env("WADOUX_RF_REFIT_FULL_CALIBRATION", default = refit_full_calibration)
geoversa_autotune_version_raw <- trimws(tolower(Sys.getenv(
  "WADOUX_GEOVERSA_AUTOTUNE_VERSION",
  unset = Sys.getenv("WADOUX_GEOVERSA_AUTOTUNE", unset = "")
)))
geoversa_policy_tune_requested <- parse_bool_env("WADOUX_GEOVERSA_POLICY_TUNE", default = FALSE)
geoversa_policy_tune <- geoversa_policy_tune_requested || nzchar(geoversa_autotune_version_raw)
paired_final_default <- all(c("RF", "ConvKrigingNet2D") %in% models) &&
  "DesignBased" %in% protocols &&
  (isTRUE(refit_full_calibration) || isTRUE(geoversa_policy_tune))
paired_final_benchmark <- parse_bool_env(
  "WADOUX_PAIRED_FINAL_BENCHMARK",
  default = paired_final_default
)
geoversa_autotune_version <- if (nzchar(geoversa_autotune_version_raw)) {
  geoversa_autotune_version_raw
} else if (geoversa_policy_tune) {
  "v2"
} else {
  "off"
}
geoversa_autotune_v2 <- geoversa_policy_tune &&
  geoversa_autotune_version %in% c("2", "v2", "autotune_v2", "geoversa_autotune_v2")
geoversa_policy_set <- Sys.getenv("WADOUX_GEOVERSA_POLICY_SET", unset = "default")
geoversa_policy_inner_splits_raw <- trimws(Sys.getenv("WADOUX_GEOVERSA_POLICY_INNER_SPLITS", unset = ""))
geoversa_policy_inner_splits <- if (nzchar(geoversa_policy_inner_splits_raw)) {
  as.integer(round(suppressWarnings(as.numeric(geoversa_policy_inner_splits_raw))))
} else {
  NA_integer_
}
geoversa_policy_max_candidates <- parse_num_env("WADOUX_GEOVERSA_POLICY_MAX_CANDIDATES", Inf)
geoversa_autotune_topk <- as.integer(round(parse_num_env("WADOUX_GEOVERSA_AUTOTUNE_TOPK", 5L)))
geoversa_autotune_racing <- parse_bool_env("WADOUX_GEOVERSA_AUTOTUNE_RACING", default = FALSE)
geoversa_autotune_operator_set <- Sys.getenv("WADOUX_GEOVERSA_AUTOTUNE_OPERATOR_SET", unset = "default")
geoversa_autotune_sd_weight <- parse_num_env("WADOUX_GEOVERSA_AUTOTUNE_SD_WEIGHT", 0.15)
geoversa_autotune_equiv_abs <- parse_num_env("WADOUX_GEOVERSA_AUTOTUNE_EQUIV_ABS", 0)
if (!is.finite(geoversa_autotune_equiv_abs) || geoversa_autotune_equiv_abs < 0) geoversa_autotune_equiv_abs <- 0
geoversa_autotune_equiv_rel <- parse_num_env("WADOUX_GEOVERSA_AUTOTUNE_EQUIV_REL", 0)
if (!is.finite(geoversa_autotune_equiv_rel) || geoversa_autotune_equiv_rel < 0) geoversa_autotune_equiv_rel <- 0
geoversa_selection_objective <- tolower(trimws(Sys.getenv(
  "WADOUX_GEOVERSA_SELECTION_OBJECTIVE", unset = "paired_rf_delta")))
if (!(geoversa_selection_objective %in% c("rmse", "paired_rf_delta"))) {
  geoversa_selection_objective <- "rmse"
}
geoversa_noharm_policy <- parse_bool_env(
  "WADOUX_GEOVERSA_NOHARM_POLICY",
  default = identical(geoversa_selection_objective, "paired_rf_delta")
)
geoversa_noharm_margin_abs <- parse_num_env(
  "WADOUX_GEOVERSA_NOHARM_MARGIN_ABS", 0.05)
if (!is.finite(geoversa_noharm_margin_abs) ||
    geoversa_noharm_margin_abs < 0) {
  geoversa_noharm_margin_abs <- 0
}
geoversa_noharm_margin_rel <- parse_num_env(
  "WADOUX_GEOVERSA_NOHARM_MARGIN_REL", 0.0)
if (!is.finite(geoversa_noharm_margin_rel) ||
    geoversa_noharm_margin_rel < 0) {
  geoversa_noharm_margin_rel <- 0
}
geoversa_rf_noharm_policy <- parse_bool_env(
  "WADOUX_GEOVERSA_RF_NOHARM_POLICY",
  default = identical(geoversa_selection_objective, "paired_rf_delta")
)
geoversa_rf_noharm_margin_abs <- parse_num_env(
  "WADOUX_GEOVERSA_RF_NOHARM_MARGIN_ABS", 0.05)
if (!is.finite(geoversa_rf_noharm_margin_abs) ||
    geoversa_rf_noharm_margin_abs < 0) {
  geoversa_rf_noharm_margin_abs <- 0
}
geoversa_rf_noharm_margin_rel <- parse_num_env(
  "WADOUX_GEOVERSA_RF_NOHARM_MARGIN_REL", 0.0)
if (!is.finite(geoversa_rf_noharm_margin_rel) ||
    geoversa_rf_noharm_margin_rel < 0) {
  geoversa_rf_noharm_margin_rel <- 0
}
geoversa_rf_noharm_restrict_ensemble <- parse_bool_env(
  "WADOUX_GEOVERSA_RF_NOHARM_RESTRICT_ENSEMBLE",
  default = TRUE
)
if (include_population && !"Population" %in% protocols) {
  protocols <- c("Population", protocols)
}
if (paired_final_benchmark) {
  if (!all(c("RF", "ConvKrigingNet2D") %in% models)) {
    stop("WADOUX_PAIRED_FINAL_BENCHMARK requires both RF and ConvKrigingNet2D in WADOUX_MODELS.")
  }
  if (!"DesignBased" %in% protocols) {
    stop("WADOUX_PAIRED_FINAL_BENCHMARK currently supports DesignBased and requires WADOUX_PROTOCOLS to include DesignBased.")
  }
}

sample_size <- as.integer(Sys.getenv("WADOUX_SAMPLE_SIZE", unset = "500"))
n_iter <- as.integer(Sys.getenv("WADOUX_N_ITER", unset = "500"))
val_frac <- as.numeric(Sys.getenv("WADOUX_VAL_FRAC", unset = "0.2"))
conformal_third_split <- parse_bool_env("WADOUX_CONFORMAL_THIRD_SPLIT", default = FALSE)
conformal_cal_frac <- parse_num_env("WADOUX_CONFORMAL_CAL_FRAC", val_frac)
rf_qrf_active <- parse_bool_env("WADOUX_RF_QRF", default = FALSE)
val_dist_km <- as.numeric(Sys.getenv("WADOUX_VAL_DIST_KM", unset = "350"))
random_k <- as.integer(Sys.getenv("WADOUX_RANDOM_K", unset = "10"))
bloo_groups <- as.integer(Sys.getenv("WADOUX_BLOO_GROUPS", unset = "10"))
bloo_test_pixels <- as.integer(Sys.getenv("WADOUX_BLOO_TEST_PIXELS", unset = "100"))
kmeans_centers <- as.integer(Sys.getenv("WADOUX_CLUSTER_CENTERS", unset = "100"))
n_psu <- as.integer(Sys.getenv("WADOUX_CLUSTER_N_PSU", unset = "20"))
m_per_psu <- as.integer(Sys.getenv("WADOUX_CLUSTER_M_PER_PSU", unset = "25"))
model_profile <- Sys.getenv("WADOUX_MODEL_PROFILE", unset = "quick")
train_seed <- as.integer(Sys.getenv("WADOUX_TRAIN_SEED", unset = "123"))
device <- Sys.getenv("WADOUX_DEVICE", unset = "cpu")
refit_selection_splits <- as.integer(round(parse_num_env("WADOUX_REFIT_SELECTION_SPLITS", 3)))
if (geoversa_policy_tune) {
  geoversa_policy_inner_default <- if (geoversa_autotune_v2) {
    min(refit_selection_splits, 2L)
  } else {
    1L
  }
  if (!is.finite(geoversa_policy_inner_splits) || geoversa_policy_inner_splits < 1L) {
    geoversa_policy_inner_splits <- geoversa_policy_inner_default
  }
  geoversa_policy_inner_splits <- min(refit_selection_splits, geoversa_policy_inner_splits)
}

if ((refit_full_calibration || paired_final_benchmark) && !identical(model_profile, "auto")) {
  stop("WADOUX_REFIT_FULL_CALIBRATION and WADOUX_PAIRED_FINAL_BENCHMARK require WADOUX_MODEL_PROFILE=auto because the refit path needs return_state from Auto v5.")
}
if (isTRUE(conformal_third_split)) {
  if (!is.finite(conformal_cal_frac) || conformal_cal_frac <= 0 || conformal_cal_frac >= 1) {
    stop("WADOUX_CONFORMAL_CAL_FRAC must be in (0,1) when WADOUX_CONFORMAL_THIRD_SPLIT=TRUE.")
  }
  if (!is.finite(val_frac) || val_frac <= 0 || val_frac >= 1) {
    stop("WADOUX_VAL_FRAC must be in (0,1) when WADOUX_CONFORMAL_THIRD_SPLIT=TRUE.")
  }
  if ((val_frac + conformal_cal_frac) >= 1) {
    stop("WADOUX_VAL_FRAC + WADOUX_CONFORMAL_CAL_FRAC must be < 1 for a valid three-way split.")
  }
}

if (scenario == "clustered_random") {
  sample_size <- n_psu * m_per_psu
}

results_dir <- Sys.getenv(
  "WADOUX_RESULTS_DIR",
  unset = file.path(project_root_wadoux, "results", sprintf("wadoux_style_rf_conv_%s_comparison", scenario))
)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

need_polygon <- identical(scenario, "regular_grid")
cat(sprintf("Loading common data for scenario: %s\n", scenario))
common_data <- load_wadoux_common_data(include_polygon = need_polygon)

if ("Population" %in% protocols && "ConvKrigingNet2D" %in% models) {
  cat(sprintf(
    "Warning: Population protocol with ConvKrigingNet2D will score the full raster domain (%d locations) and may be extremely slow.\n",
    nrow(common_data$s_df)
  ))
}

conv_env <- NULL
conv_params <- NULL
if ("ConvKrigingNet2D" %in% models) {
  cat("Loading ConvKrigingNet2D environment...\n")
  conv_env <- load_convkrigingnet2d_env()
  default_auto_v5_path <- file.path(project_root_wadoux, "code", "ConvKrigingNet2D_Auto_v5.R")
  default_auto_path <- file.path(project_root_wadoux, "code", "ConvKrigingNet2D_Auto.R")
  override_path <- Sys.getenv("WADOUX_TWOSTAGE_SCRIPT", unset = "")
  if (nchar(override_path) > 0 && file.exists(override_path)) {
    sys.source(override_path, envir = conv_env)
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_twostage_one_fold
    cat("[TwoStage] Overriding train function with two-stage trainer\n")
  }
  # ExpCov hook — replaces kriging layer with pure exponential covariance
  #               takes priority over TwoStage
  expcov_path <- Sys.getenv("WADOUX_EXPCOV_SCRIPT", unset = "")
  if (nchar(expcov_path) > 0 && file.exists(expcov_path)) {
    sys.source(expcov_path, envir = conv_env)
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_expcov_one_fold
    cat("[ExpCov] Overriding train function with ExpCov trainer\n")
  }
  # Auto hook — fully self-configuring model; takes priority over ExpCov/TwoStage
  # v5 hook — takes priority over v4 Auto
  auto_v5_path <- Sys.getenv("WADOUX_AUTO_V5_SCRIPT", unset = "")
  if (nchar(auto_v5_path) == 0 && identical(model_profile, "auto") && file.exists(default_auto_v5_path)) {
    auto_v5_path <- default_auto_v5_path
  }
  if (nchar(auto_v5_path) > 0 && file.exists(auto_v5_path)) {
    sys.source(auto_v5_path, envir = conv_env)
    # Also load v4 base (v5 builds on top)
    auto_path <- Sys.getenv("WADOUX_AUTO_SCRIPT", unset = "")
    if (nchar(auto_path) == 0 && file.exists(default_auto_path)) {
      auto_path <- default_auto_path
    }
    if (nchar(auto_path) > 0 && file.exists(auto_path)) {
      sys.source(auto_path, envir = conv_env)
    }
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_auto_one_fold_v5
    cat("[Auto v5] Overriding train function with COMPLETE AUTO-CONFIG v5\n")
  } else {
    # Fall back to v4 Auto
    auto_path <- Sys.getenv("WADOUX_AUTO_SCRIPT", unset = "")
    if (nchar(auto_path) == 0 && identical(model_profile, "auto") && file.exists(default_auto_path)) {
      auto_path <- default_auto_path
    }
    if (nchar(auto_path) > 0 && file.exists(auto_path)) {
      sys.source(auto_path, envir = conv_env)
      conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_auto_one_fold
      cat("[Auto v4] Overriding train function with Auto (self-configuring) trainer\n")
    }
  }
  # DeepRK hook — takes priority over TwoStage, ExpCov and Auto if all are set
  deeprk_path <- Sys.getenv("WADOUX_DEEPRK_SCRIPT", unset = "")
  if (nchar(deeprk_path) > 0 && file.exists(deeprk_path)) {
    sys.source(deeprk_path, envir = conv_env)
    conv_env$train_convkrigingnet2d_one_fold <- conv_env$train_convkrigingnet2d_deeprk_one_fold
    cat("[DeepRK] Overriding train function with DeepRK trainer\n")
  }
  conv_params <- get_conv_params_profile(
    conv_env = conv_env,
    profile = model_profile,
    train_seed = train_seed,
    device = device
  )
  # DeepRK calibration override: allows Benchmark_DeepRK.R to inject
  # calibrate_method = "linear" without touching conv_params directly.
  deeprk_calibrate <- Sys.getenv("WADOUX_DEEPRK_CALIBRATE", unset = "")
  if (nchar(deeprk_calibrate) > 0 && nchar(deeprk_path) > 0) {
    conv_params$calibrate_method <- deeprk_calibrate
    cat(sprintf("[DeepRK] calibrate_method overridden → '%s'\n", deeprk_calibrate))
  }
  # Ablation override: WADOUX_WARMUP_EPOCHS overrides warmup_epochs in conv_params.
  ablation_warmup <- Sys.getenv("WADOUX_WARMUP_EPOCHS", unset = "")
  if (nchar(ablation_warmup) > 0) {
    conv_params$warmup_epochs <- as.integer(ablation_warmup)
    cat(sprintf("[Ablation] warmup_epochs overridden → %dL\n", conv_params$warmup_epochs))
  }
  # Ablation override: WADOUX_MAIN_EPOCHS overrides the main epoch budget.
  ablation_main_epochs <- Sys.getenv("WADOUX_MAIN_EPOCHS", unset = "")
  if (nchar(ablation_main_epochs) > 0) {
    conv_params$epochs <- as.integer(ablation_main_epochs)
    cat(sprintf("[Ablation] main epochs overridden → %dL\n", conv_params$epochs))
  }
  # Ablation override: WADOUX_BASE_LOSS_WEIGHT
  ablation_blw <- Sys.getenv("WADOUX_BASE_LOSS_WEIGHT", unset = "")
  if (nchar(ablation_blw) > 0) {
    conv_params$base_loss_weight <- as.numeric(ablation_blw)
    cat(sprintf("[Ablation] base_loss_weight overridden → %.4f\n", conv_params$base_loss_weight))
  }
  # Ablation override: WADOUX_ALPHA_ME
  ablation_alpha_me <- Sys.getenv("WADOUX_ALPHA_ME", unset = "")
  if (nchar(ablation_alpha_me) > 0) {
    conv_params$alpha_me <- as.numeric(ablation_alpha_me)
    cat(sprintf("[Ablation] alpha_me overridden → %.4f\n", conv_params$alpha_me))
  }
  # Ablation override: WADOUX_LAMBDA_COV
  ablation_lambda_cov <- Sys.getenv("WADOUX_LAMBDA_COV", unset = "")
  if (nchar(ablation_lambda_cov) > 0) {
    conv_params$lambda_cov <- as.numeric(ablation_lambda_cov)
    cat(sprintf("[Ablation] lambda_cov overridden → %.5f\n", conv_params$lambda_cov))
  }
  # Ablation override: WADOUX_LR
  ablation_lr <- Sys.getenv("WADOUX_LR", unset = "")
  if (nchar(ablation_lr) > 0) {
    conv_params$lr <- as.numeric(ablation_lr)
    cat(sprintf("[Ablation] lr overridden → %.2e\n", conv_params$lr))
  }
  # Ablation override: WADOUX_MIN_LR
  ablation_min_lr <- Sys.getenv("WADOUX_MIN_LR", unset = "")
  if (nchar(ablation_min_lr) > 0) {
    conv_params$min_lr <- as.numeric(ablation_min_lr)
    cat(sprintf("[Ablation] min_lr overridden → %.2e\n", conv_params$min_lr))
  }
  # Ablation trace: WADOUX_LR_CEIL — applied inside the LR finder via Sys.getenv().
  ablation_lr_ceil <- Sys.getenv("WADOUX_LR_CEIL", unset = "")
  if (nchar(ablation_lr_ceil) > 0) {
    cat(sprintf("[Ablation] WADOUX_LR_CEIL active → %.2e (applied inside LR finder)\n",
                as.numeric(ablation_lr_ceil)))
  }
  # Ablation override: WADOUX_WEIGHT_DECAY
  ablation_wd <- Sys.getenv("WADOUX_WEIGHT_DECAY", unset = "")
  if (nchar(ablation_wd) > 0) {
    conv_params$wd <- as.numeric(ablation_wd)
    cat(sprintf("[Ablation] wd overridden → %.2e\n", conv_params$wd))
  }
  # Ablation override: WADOUX_BATCH_SIZE
  ablation_bs <- Sys.getenv("WADOUX_BATCH_SIZE", unset = "")
  if (nchar(ablation_bs) > 0) {
    conv_params$batch_size <- as.integer(ablation_bs)
    cat(sprintf("[Ablation] batch_size overridden → %dL\n", conv_params$batch_size))
  }
  # Ablation override: WADOUX_BANK_REFRESH_EVERY
  ablation_bre <- Sys.getenv("WADOUX_BANK_REFRESH_EVERY", unset = "")
  if (nchar(ablation_bre) > 0) {
    conv_params$bank_refresh_every <- as.integer(ablation_bre)
    cat(sprintf("[Ablation] bank_refresh_every overridden → %dL\n", conv_params$bank_refresh_every))
  }
  # Ablation override: WADOUX_PATIENCE
  ablation_pat <- Sys.getenv("WADOUX_PATIENCE", unset = "")
  if (nchar(ablation_pat) > 0) {
    conv_params$patience <- as.integer(ablation_pat)
    cat(sprintf("[Ablation] patience overridden → %dL\n", conv_params$patience))
  }
  # Ablation override: WADOUX_LR_PATIENCE
  ablation_lr_pat <- Sys.getenv("WADOUX_LR_PATIENCE", unset = "")
  if (nchar(ablation_lr_pat) > 0) {
    conv_params$lr_patience <- as.integer(ablation_lr_pat)
    cat(sprintf("[Ablation] lr_patience overridden → %dL\n", conv_params$lr_patience))
  }
  # Ablation override: WADOUX_LR_DECAY
  ablation_lr_decay <- Sys.getenv("WADOUX_LR_DECAY", unset = "")
  if (nchar(ablation_lr_decay) > 0) {
    conv_params$lr_decay <- as.numeric(ablation_lr_decay)
    cat(sprintf("[Ablation] lr_decay overridden → %.2f\n", conv_params$lr_decay))
  }
  # Ablation override: WADOUX_K_NEIGHBORS
  ablation_k_neighbors <- Sys.getenv("WADOUX_K_NEIGHBORS", unset = "")
  if (nchar(ablation_k_neighbors) > 0) {
    conv_params$K_neighbors <- as.integer(ablation_k_neighbors)
    cat(sprintf("[Ablation] K_neighbors overridden → %dL\n", conv_params$K_neighbors))
  }
  # Ablation override: WADOUX_BETA_INIT
  ablation_beta_init <- Sys.getenv("WADOUX_BETA_INIT", unset = "")
  if (nchar(ablation_beta_init) > 0) {
    conv_params$beta_init <- as.numeric(ablation_beta_init)
    cat(sprintf("[Ablation] beta_init overridden → %.4f\n", conv_params$beta_init))
  }
  # Ablation override: architecture dimensions and dropouts
  ablation_patch_dim   <- Sys.getenv("WADOUX_PATCH_DIM",    unset = "")
  ablation_d           <- Sys.getenv("WADOUX_D",            unset = "")
  ablation_tab_drop    <- Sys.getenv("WADOUX_TAB_DROPOUT",  unset = "")
  ablation_patch_drop  <- Sys.getenv("WADOUX_PATCH_DROPOUT",unset = "")
  ablation_coord_dim   <- Sys.getenv("WADOUX_COORD_DIM",    unset = "")
  ablation_coord_drop  <- Sys.getenv("WADOUX_COORD_DROPOUT",unset = "")
  if (nchar(ablation_patch_dim)  > 0) { conv_params$patch_dim    <- as.integer(ablation_patch_dim);  cat(sprintf("[Ablation] patch_dim    → %d\n",    conv_params$patch_dim))    }
  if (nchar(ablation_d)          > 0) { conv_params$d             <- as.integer(ablation_d);          cat(sprintf("[Ablation] d           → %d\n",    conv_params$d))            }
  if (nchar(ablation_tab_drop)   > 0) { conv_params$tab_dropout   <- as.numeric(ablation_tab_drop);   cat(sprintf("[Ablation] tab_dropout → %.2f\n", conv_params$tab_dropout))   }
  if (nchar(ablation_patch_drop) > 0) { conv_params$patch_dropout <- as.numeric(ablation_patch_drop); cat(sprintf("[Ablation] patch_drop  → %.2f\n", conv_params$patch_dropout)) }
  if (nchar(ablation_coord_dim)  > 0) { conv_params$coord_dim     <- as.integer(ablation_coord_dim);  cat(sprintf("[Ablation] coord_dim   → %d\n",    conv_params$coord_dim))    }
  if (nchar(ablation_coord_drop) > 0) { conv_params$coord_dropout <- as.numeric(ablation_coord_drop); cat(sprintf("[Ablation] coord_drop  → %.2f\n", conv_params$coord_dropout)) }
}
# v4 DYNAMIC patch_size: derived from sample_size using auto_kriging_config rule
# patch_size = min(max(8, floor(√n)), 31)
# Ablation override: WADOUX_PATCH_SIZE can still override if explicitly set.
patch_size_ablation <- Sys.getenv("WADOUX_PATCH_SIZE", unset = "")
if (nchar(patch_size_ablation) > 0) {
  # Explicit ablation override takes priority
  patch_size <- as.integer(patch_size_ablation)
  cat(sprintf("[Ablation] patch_size overridden → %dL\n", patch_size))
} else {
  # v4 AUTO: dynamic patch_size from sample_size
  patch_size <- as.integer(min(max(8L, floor(sqrt(sample_size))), 31L))
  cat(sprintf("[Auto v4] patch_size dynamically set → %d [rule: min(max(8,⌊√n⌋), 31)]\n", patch_size))
}

patch_ssl_active_runner <- identical(model_profile, "auto") &&
  parse_bool_env("GEOVERSA_AUTO_PATCH_SSL", default = FALSE) &&
  "ConvKrigingNet2D" %in% models
patch_ssl_freeze_warmup_runner <- patch_ssl_active_runner &&
  parse_bool_env("GEOVERSA_AUTO_PATCH_SSL_FREEZE_WARMUP", default = TRUE)
patch_ssl_artifact <- NULL
patch_ssl_meta <- NULL

if (patch_ssl_active_runner) {
  if (is.null(conv_env$pretrain_patch_encoder_ssl_auto) ||
      !is.function(conv_env$pretrain_patch_encoder_ssl_auto)) {
    stop("GEOVERSA_AUTO_PATCH_SSL=1 requires pretrain_patch_encoder_ssl_auto() in ConvKrigingNet2D_Auto.R.")
  }
  patch_pyramid_active_runner <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_AUTO_PATCH_PYRAMID", unset = "1")))
  patch_pyramid_active_runner <- is.finite(patch_pyramid_active_runner) && patch_pyramid_active_runner > 0
  patch_layers_ssl <- conv_env$wadoux_context$stack[[conv_env$wadoux_context$predictors]]
  patch_channels_ssl <- terra::nlyr(patch_layers_ssl)
  d_ssl <- if (!is.null(conv_env$.auto_d_from_n) && is.function(conv_env$.auto_d_from_n)) {
    conv_env$.auto_d_from_n(sample_size)
  } else {
    192L
  }
  patch_dim_ssl <- if (!is.null(conv_env$.auto_patch_dim_from_cnn) && is.function(conv_env$.auto_patch_dim_from_cnn)) {
    conv_env$.auto_patch_dim_from_cnn(patch_channels_ssl, patch_size, d_ssl)
  } else {
    as.integer(max(48L, min(d_ssl, ceiling(sqrt(patch_channels_ssl * patch_size^2)))))
  }
  patch_local_ssl <- if (patch_pyramid_active_runner) {
    as.integer(max(9L, min(patch_size, ifelse(patch_size %% 2L == 0L, patch_size - 1L, patch_size))))
  } else {
    NULL
  }
  patch_ssl_samples_env <- trimws(Sys.getenv("GEOVERSA_AUTO_PATCH_SSL_SAMPLES", unset = ""))
  patch_ssl_epochs_env <- trimws(Sys.getenv("GEOVERSA_AUTO_PATCH_SSL_EPOCHS", unset = ""))
  patch_ssl_batch_env <- trimws(Sys.getenv("GEOVERSA_AUTO_PATCH_SSL_BATCH_SIZE", unset = ""))
  patch_ssl_samples <- if (nzchar(patch_ssl_samples_env)) {
    suppressWarnings(as.integer(round(as.numeric(patch_ssl_samples_env))))
  } else if (!is.null(conv_env$.auto_ssl_n_samples_from_raster) &&
             is.function(conv_env$.auto_ssl_n_samples_from_raster)) {
    conv_env$.auto_ssl_n_samples_from_raster(patch_layers_ssl)
  } else {
    4096L
  }
  patch_ssl_epochs <- if (nzchar(patch_ssl_epochs_env)) {
    suppressWarnings(as.integer(round(as.numeric(patch_ssl_epochs_env))))
  } else if (!is.null(conv_env$.auto_ssl_epochs_from_samples) &&
             is.function(conv_env$.auto_ssl_epochs_from_samples)) {
    conv_env$.auto_ssl_epochs_from_samples(patch_ssl_samples)
  } else {
    8L
  }
  patch_ssl_batch <- if (nzchar(patch_ssl_batch_env)) {
    suppressWarnings(as.integer(round(as.numeric(patch_ssl_batch_env))))
  } else {
    as.integer(min(256L, max(64L, floor(patch_ssl_samples / 32L))))
  }
  patch_ssl_seed <- as.integer(train_seed + 70000L)
  patch_ssl_artifact <- conv_env$pretrain_patch_encoder_ssl_auto(
    context = conv_env$wadoux_context,
    patch_size = patch_size,
    patch_dim = patch_dim_ssl,
    multiscale_patch = patch_pyramid_active_runner,
    patch_local_size = patch_local_ssl,
    n_samples = patch_ssl_samples,
    epochs = patch_ssl_epochs,
    batch_size = patch_ssl_batch,
    seed = patch_ssl_seed,
    device = device
  )
  patch_ssl_meta <- patch_ssl_artifact$meta
  conv_params$patch_ssl_state <- patch_ssl_artifact$enc_patch_state
  conv_params$patch_ssl_meta <- patch_ssl_meta
  conv_params$patch_ssl_freeze_warmup <- patch_ssl_freeze_warmup_runner
  if (!is.null(patch_ssl_meta) && nrow(patch_ssl_meta) > 0L) {
    write.csv(
      patch_ssl_meta,
      file.path(results_dir, "geoversa_patch_ssl_pretrain_summary.csv"),
      row.names = FALSE
    )
  }
  if (!is.null(patch_ssl_artifact$history) && nrow(patch_ssl_artifact$history) > 0L) {
    write.csv(
      patch_ssl_artifact$history,
      file.path(results_dir, "geoversa_patch_ssl_pretrain_history.csv"),
      row.names = FALSE
    )
  }
}

fixed_patch_global_cfg <- NA_integer_
fixed_patch_local_cfg <- NA_integer_
fixed_patch_source_cfg <- ""
external_ssl_patch_state_cfg <- trimws(Sys.getenv("GEOVERSA_SSL_PATCH_STATE", unset = ""))
external_ssl_patch_active_cfg <- nzchar(external_ssl_patch_state_cfg)
fixed_patch_global_raw <- trimws(Sys.getenv(
  "GEOVERSA_PATCH_GLOBAL_SIZE",
  unset = Sys.getenv("GEOVERSA_SSL_PATCH_GLOBAL_SIZE", unset = "")
))
fixed_patch_local_raw <- trimws(Sys.getenv(
  "GEOVERSA_PATCH_LOCAL_SIZE",
  unset = Sys.getenv("GEOVERSA_SSL_PATCH_LOCAL_SIZE", unset = "")
))
if (nzchar(fixed_patch_global_raw)) {
  fixed_patch_global_cfg <- suppressWarnings(as.integer(round(as.numeric(fixed_patch_global_raw))))
  fixed_patch_local_cfg <- suppressWarnings(as.integer(round(as.numeric(fixed_patch_local_raw))))
  fixed_patch_source_cfg <- "env"
} else {
  if (external_ssl_patch_active_cfg) {
    meta_path_cfg <- file.path(dirname(path.expand(external_ssl_patch_state_cfg)), "geoversa_ssl_pretrain_summary.csv")
    if (file.exists(meta_path_cfg)) {
      meta_cfg <- tryCatch(read.csv(meta_path_cfg, stringsAsFactors = FALSE), error = function(e) NULL)
      if (!is.null(meta_cfg) && is.data.frame(meta_cfg) && nrow(meta_cfg) > 0L) {
        meta_cfg <- meta_cfg[1L, , drop = FALSE]
        fixed_patch_global_cfg <- if ("ssl_patch_global_size" %in% names(meta_cfg)) {
          suppressWarnings(as.integer(meta_cfg$ssl_patch_global_size[1]))
        } else {
          suppressWarnings(as.integer(meta_cfg$ssl_patch_size[1]))
        }
        fixed_patch_local_cfg <- suppressWarnings(as.integer(meta_cfg$ssl_patch_local_size[1]))
        fixed_patch_source_cfg <- "external_ssl"
      }
    }
  }
}

config <- data.frame(
  scenario = scenario,
  models = paste(models, collapse = ","),
  protocols = paste(protocols, collapse = ","),
  include_population = include_population,
	  refit_full_calibration = refit_full_calibration,
	  geoversa_train_regime = if (refit_full_calibration) {
	    if (conformal_third_split) "train_plus_val_refit_with_calibration_holdout" else "500_full_refit"
	  } else {
	    if (conformal_third_split) "three_way_inner_split" else "400_plus_100_inner_validation"
		  },
		  geoversa_policy_tune = geoversa_policy_tune,
		  geoversa_autotune_version = geoversa_autotune_version,
		  geoversa_autotune_v2 = geoversa_autotune_v2,
			  geoversa_autotune_topk = geoversa_autotune_topk,
			  geoversa_autotune_racing = geoversa_autotune_racing,
			  geoversa_autotune_operator_set = geoversa_autotune_operator_set,
			  geoversa_autotune_sd_weight = geoversa_autotune_sd_weight,
			  geoversa_autotune_equiv_abs = geoversa_autotune_equiv_abs,
			  geoversa_autotune_equiv_rel = geoversa_autotune_equiv_rel,
			  geoversa_selection_objective = geoversa_selection_objective,
			  geoversa_noharm_policy = geoversa_noharm_policy,
			  geoversa_noharm_margin_abs = geoversa_noharm_margin_abs,
			  geoversa_noharm_margin_rel = geoversa_noharm_margin_rel,
			  geoversa_rf_noharm_policy = geoversa_rf_noharm_policy,
			  geoversa_rf_noharm_margin_abs = geoversa_rf_noharm_margin_abs,
			  geoversa_rf_noharm_margin_rel = geoversa_rf_noharm_margin_rel,
			  geoversa_rf_noharm_restrict_ensemble =
			    geoversa_rf_noharm_restrict_ensemble,
			  geoversa_policy_set = geoversa_policy_set,
		  geoversa_policy_inner_splits = if (is.finite(geoversa_policy_inner_splits)) geoversa_policy_inner_splits else NA_integer_,
			  geoversa_policy_max_candidates = if (is.finite(geoversa_policy_max_candidates)) {
			    as.character(geoversa_policy_max_candidates)
			  } else {
			    "Inf"
			  },
		  geoversa_policy_candidate_labels = paste(
		    vapply(
		      if (isTRUE(geoversa_autotune_v2)) {
		        make_geoversa_autotune_v2_grid(
		          geoversa_policy_set,
		          geoversa_policy_max_candidates,
		          data_profile = list(n = sample_size, p = length(predList_modelfull_wadoux))
		        )
		      } else {
		        make_geoversa_policy_grid(geoversa_policy_set, geoversa_policy_max_candidates)
		      },
		      `[[`,
		      character(1),
		      "label"
	    ),
	    collapse = ","
	  ),
	  geoversa_conformal_calibration = parse_bool_env("GEOVERSA_CONFORMAL_CALIBRATION", default = FALSE),
  conformal_third_split = conformal_third_split,
  conformal_cal_frac = if (conformal_third_split) conformal_cal_frac else 0,
  rf_qrf_active = rf_qrf_active,
  rf_refit_full_calibration = rf_refit_full_calibration,
  rf_model_label = resolve_rf_model_label(rf_refit_full_calibration, conformal_third_split),
  rf_train_regime = if (rf_refit_full_calibration) {
    if (conformal_third_split) "train_plus_val_with_calibration_holdout" else "500_full_calibration"
  } else {
    if (conformal_third_split) "train_core_with_calibration_holdout" else "400_train_core"
  },
  paired_final_benchmark = paired_final_benchmark,
  sample_size = sample_size,
  n_iter = n_iter,
  val_frac = val_frac,
	  val_dist_km = val_dist_km,
	  random_k = random_k,
	  refit_selection_splits = refit_selection_splits,
	  refit_main_epoch_floor = parse_num_env("WADOUX_REFIT_MAIN_EPOCH_FLOOR", 8),
	  refit_main_selection_floor_frac = parse_num_env("WADOUX_REFIT_MAIN_SELECTION_FLOOR_FRAC", 0.60),
	  refit_noisy_epoch_mult = parse_num_env("WADOUX_REFIT_NOISY_EPOCH_MULT", 0.65),
	  refit_noisy_lr_mult = parse_num_env("WADOUX_REFIT_NOISY_LR_MULT", 0.20),
	  refit_noisy_min_main_epochs = as.integer(round(parse_num_env("WADOUX_REFIT_NOISY_MIN_MAIN_EPOCHS", 8))),
	  refit_noisy_min_main_frac = parse_num_env("WADOUX_REFIT_NOISY_MIN_MAIN_FRAC", 0.60),
	  refit_anchor_noisy_conservative = parse_bool_env("WADOUX_REFIT_ANCHOR_NOISY_CONSERVATIVE", default = TRUE),
	  bloo_groups = bloo_groups,
  bloo_test_pixels = bloo_test_pixels,
  kmeans_centers = kmeans_centers,
  n_psu = n_psu,
  m_per_psu = m_per_psu,
  model_profile = model_profile,
  train_seed = train_seed,
  device = device,
  patch_ssl_active = patch_ssl_active_runner || external_ssl_patch_active_cfg,
  patch_ssl_external_active = external_ssl_patch_active_cfg,
  patch_ssl_freeze_warmup = patch_ssl_freeze_warmup_runner,
  patch_ssl_patch_size = if (patch_ssl_active_runner) patch_size else fixed_patch_global_cfg,
  fixed_patch_global_size = fixed_patch_global_cfg,
  fixed_patch_local_size = fixed_patch_local_cfg,
  fixed_patch_geometry_source = fixed_patch_source_cfg,
  geoversa_meta_controller = parse_bool_env("GEOVERSA_AUTO_META_CONTROLLER", default = TRUE),
  dad_rk_active = parse_bool_env("GEOVERSA_AUTO_DESIGN_WEIGHTS", default = FALSE),
  dad_rk_mode = Sys.getenv("GEOVERSA_AUTO_DESIGN_WEIGHT_MODE", unset = "hybrid"),
  dad_rk_k = parse_num_env("GEOVERSA_AUTO_DESIGN_WEIGHT_K", NA_real_),
  dad_rk_power = parse_num_env("GEOVERSA_AUTO_DESIGN_WEIGHT_POWER", 1.0),
  dad_rk_env_share = parse_num_env("GEOVERSA_AUTO_DESIGN_WEIGHT_ENV_SHARE", 0.5),
  dad_rk_clip_min = parse_num_env("GEOVERSA_AUTO_DESIGN_WEIGHT_CLIP_MIN", 0.35),
  dad_rk_clip_max = parse_num_env("GEOVERSA_AUTO_DESIGN_WEIGHT_CLIP_MAX", 2.75),
  external_ssl_patch_state = external_ssl_patch_state_cfg,
  external_ssl_patch_warmup_lr_mult = parse_num_env("GEOVERSA_SSL_PATCH_WARMUP_LR_MULT", NA_real_),
  stringsAsFactors = FALSE
)
write.csv(config, file.path(results_dir, "wadoux_style_rf_conv_config.csv"), row.names = FALSE)

sample_calibration <- function() {
  if (scenario == "random") {
    out <- sample_simple_random_rows_wadoux(common_data$s_df, sample_size)
  } else if (scenario == "regular_grid") {
    out <- sample_valuetable_regular_grid_wadoux(common_data = common_data, sample_size = sample_size)
  } else if (scenario == "clustered_random") {
    out <- sample_valuetable_clustered_random_wadoux(
      common_data = common_data,
      kmeans_centers = kmeans_centers,
      n_psu = n_psu,
      m_per_psu = m_per_psu
    )
  } else {
    stop("Unsupported scenario: ", scenario)
  }
  out$..patch_cache_row_id <- seq_len(nrow(out))
  out
}

all_results <- vector("list", n_iter)
all_diagnostics <- vector("list", n_iter)
all_uq_results <- vector("list", n_iter)
all_reliability <- vector("list", n_iter)
all_paired_deltas <- vector("list", n_iter)
all_paired_meta <- vector("list", n_iter)
all_prediction_audit <- vector("list", n_iter)
all_split_audit <- vector("list", n_iter)
.geoversa_policy_tuning_rows <- list()
.geoversa_operator_tuning_rows <- list()

summarise_results_by_protocol <- function(results_df) {
  if (nrow(results_df) == 0) {
    return(data.frame())
  }
  results_df %>%
    group_by(scenario, protocol, model) %>%
    summarise(
      ME_mean = mean(ME, na.rm = TRUE),
      RMSE_mean = mean(RMSE, na.rm = TRUE),
      r2_mean = mean(r2, na.rm = TRUE),
      MEC_mean = mean(MEC, na.rm = TRUE),
      ME_sd = sd(ME, na.rm = TRUE),
      RMSE_sd = sd(RMSE, na.rm = TRUE),
      r2_sd = sd(r2, na.rm = TRUE),
      MEC_sd = sd(MEC, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(protocol, RMSE_mean)
}

summarise_diagnostics <- function(diagnostics_df) {
  if (nrow(diagnostics_df) == 0) {
    return(data.frame())
  }
  diagnostics_df %>%
    group_by(scenario, protocol, model) %>%
    summarise(
      val_RMSE_base_mean = mean(val_RMSE_base, na.rm = TRUE),
      val_RMSE_full_mean = mean(val_RMSE_full, na.rm = TRUE),
      val_RMSE_gain_mean = mean(val_RMSE_gain, na.rm = TRUE),
      test_RMSE_base_mean = mean(test_RMSE_base, na.rm = TRUE),
      test_RMSE_full_mean = mean(test_RMSE_full, na.rm = TRUE),
      test_RMSE_gain_mean = mean(test_RMSE_gain, na.rm = TRUE),
      val_r2_gain_mean = mean(val_r2_gain, na.rm = TRUE),
      test_r2_gain_mean = mean(test_r2_gain, na.rm = TRUE),
      val_MEC_gain_mean = mean(val_MEC_gain, na.rm = TRUE),
      test_MEC_gain_mean = mean(test_MEC_gain, na.rm = TRUE),
      beta_final_mean = mean(beta_final, na.rm = TRUE),
      beta_final_sd = sd(beta_final, na.rm = TRUE),
      K_neighbors_mean = mean(K_neighbors, na.rm = TRUE),
      nugget_ratio_mean = mean(nugget_ratio, na.rm = TRUE),
      mean_abs_delta_scaled_test_mean = mean(mean_abs_delta_scaled_test, na.rm = TRUE),
      mean_abs_delta_scaled_test_sd = sd(mean_abs_delta_scaled_test, na.rm = TRUE),
      warmup_epochs_done_mean = mean(warmup_epochs_done, na.rm = TRUE),
      best_epoch_total_mean = mean(best_epoch_total, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(protocol)
}

summarise_paired_deltas <- function(paired_deltas_df) {
  if (nrow(paired_deltas_df) == 0) {
    return(data.frame())
  }
  paired_deltas_df %>%
    group_by(scenario, protocol, comparison, lhs_model, rhs_model) %>%
    summarise(
      delta_ME_mean = mean(delta_ME, na.rm = TRUE),
      delta_RMSE_mean = mean(delta_RMSE, na.rm = TRUE),
      delta_r2_mean = mean(delta_r2, na.rm = TRUE),
      delta_MEC_mean = mean(delta_MEC, na.rm = TRUE),
      delta_ME_sd = sd(delta_ME, na.rm = TRUE),
      delta_RMSE_sd = sd(delta_RMSE, na.rm = TRUE),
      delta_r2_sd = sd(delta_r2, na.rm = TRUE),
      delta_MEC_sd = sd(delta_MEC, na.rm = TRUE),
      lhs_better_rmse_rate = mean(lhs_better_rmse, na.rm = TRUE),
      lhs_better_r2_rate = mean(lhs_better_r2, na.rm = TRUE),
      lhs_better_mec_rate = mean(lhs_better_mec, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(protocol, comparison)
}

write_iteration_checkpoint_outputs <- function(results_dir,
                                             all_results,
                                             all_diagnostics,
                                             all_uq_results,
                                             all_reliability,
                                             all_paired_deltas,
                                             all_paired_meta,
                                             all_prediction_audit,
                                             all_split_audit,
                                             completed_iteration,
                                             total_iterations) {
  final_results <- bind_rows(all_results)
  final_diagnostics <- bind_rows(all_diagnostics)
  final_uq_results <- bind_rows(all_uq_results)
	  final_reliability <- bind_rows(all_reliability)
	  final_paired_deltas <- bind_rows(all_paired_deltas)
	  final_paired_meta <- bind_rows(all_paired_meta)
	  final_prediction_audit <- bind_rows(all_prediction_audit)
	  final_split_audit <- bind_rows(all_split_audit)
	  final_policy_tuning <- if (exists(".geoversa_policy_tuning_rows", inherits = TRUE)) {
	    bind_rows(.geoversa_policy_tuning_rows)
	  } else {
	    data.frame()
	  }
	  final_operator_tuning <- if (exists(".geoversa_operator_tuning_rows", inherits = TRUE)) {
	    bind_rows(.geoversa_operator_tuning_rows)
	  } else {
	    data.frame()
	  }

  summary_by_protocol <- summarise_results_by_protocol(final_results)
  diagnostic_summary <- summarise_diagnostics(final_diagnostics)
  uq_summary <- summarise_uq_results_by_protocol(final_uq_results)
  reliability_summary <- summarise_reliability_by_protocol(final_reliability)
  paired_delta_summary <- summarise_paired_deltas(final_paired_deltas)

  unlink(file.path(results_dir, c(
    "wadoux_style_rf_conv_fold_diagnostics.csv",
    "wadoux_style_rf_conv_fold_diagnostic_summary.csv",
    "wadoux_style_rf_conv_uq_all_results.csv",
    "wadoux_style_rf_conv_uq_summary_by_protocol.csv",
    "wadoux_style_rf_conv_reliability.csv",
	    "wadoux_style_rf_conv_reliability_summary.csv",
	    "wadoux_style_rf_conv_paired_deltas.csv",
	    "wadoux_style_rf_conv_paired_meta.csv",
	    "wadoux_style_rf_conv_paired_delta_summary.csv",
	    "wadoux_style_rf_conv_prediction_audit.csv",
	    "wadoux_style_rf_conv_split_audit.csv",
	    "wadoux_style_rf_conv_policy_tuning_scores.csv",
	    "wadoux_style_rf_conv_operator_tuning_scores.csv"
	  )), force = TRUE)

  write.csv(final_results, file.path(results_dir, "wadoux_style_rf_conv_all_results.csv"), row.names = FALSE)
  write.csv(summary_by_protocol, file.path(results_dir, "wadoux_style_rf_conv_summary_by_protocol.csv"), row.names = FALSE)
  if (nrow(final_diagnostics) > 0) {
    write.csv(final_diagnostics, file.path(results_dir, "wadoux_style_rf_conv_fold_diagnostics.csv"), row.names = FALSE)
    write.csv(diagnostic_summary, file.path(results_dir, "wadoux_style_rf_conv_fold_diagnostic_summary.csv"), row.names = FALSE)
  }
  if (nrow(final_uq_results) > 0) {
    write.csv(final_uq_results, file.path(results_dir, "wadoux_style_rf_conv_uq_all_results.csv"), row.names = FALSE)
    write.csv(uq_summary, file.path(results_dir, "wadoux_style_rf_conv_uq_summary_by_protocol.csv"), row.names = FALSE)
  }
  if (nrow(final_reliability) > 0) {
    write.csv(final_reliability, file.path(results_dir, "wadoux_style_rf_conv_reliability.csv"), row.names = FALSE)
    write.csv(reliability_summary, file.path(results_dir, "wadoux_style_rf_conv_reliability_summary.csv"), row.names = FALSE)
  }
	  if (nrow(final_paired_deltas) > 0) {
	    write.csv(final_paired_deltas, file.path(results_dir, "wadoux_style_rf_conv_paired_deltas.csv"), row.names = FALSE)
	    write.csv(final_paired_meta, file.path(results_dir, "wadoux_style_rf_conv_paired_meta.csv"), row.names = FALSE)
	    write.csv(paired_delta_summary, file.path(results_dir, "wadoux_style_rf_conv_paired_delta_summary.csv"), row.names = FALSE)
	  }
	  if (nrow(final_prediction_audit) > 0) {
	    write.csv(final_prediction_audit, file.path(results_dir, "wadoux_style_rf_conv_prediction_audit.csv"), row.names = FALSE)
	  }
	  if (nrow(final_split_audit) > 0) {
	    write.csv(final_split_audit, file.path(results_dir, "wadoux_style_rf_conv_split_audit.csv"), row.names = FALSE)
	  }
	  if (nrow(final_policy_tuning) > 0) {
	    write.csv(final_policy_tuning, file.path(results_dir, "wadoux_style_rf_conv_policy_tuning_scores.csv"), row.names = FALSE)
	  }
	  if (nrow(final_operator_tuning) > 0) {
	    write.csv(final_operator_tuning, file.path(results_dir, "wadoux_style_rf_conv_operator_tuning_scores.csv"), row.names = FALSE)
	  }
  write.csv(
    data.frame(
      completed_iteration = completed_iteration,
      total_iterations = total_iterations,
      completed_at = format(Sys.time(), tz = "", usetz = TRUE),
      stringsAsFactors = FALSE
    ),
    file.path(results_dir, "wadoux_style_rf_conv_checkpoint_status.csv"),
    row.names = FALSE
  )
}

for (iter in seq_len(n_iter)) {
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("Wadoux-style RF vs ConvKrigingNet2D | scenario=%s | iteration %d / %d\n", scenario, iter, n_iter))
  cat(sprintf("==================================================\n"))

  valuetable <- na.omit(as.data.frame(sample_calibration()))
  valuetable$..patch_cache_row_id <- seq_len(nrow(valuetable))
  calibration_patch_cache <- NULL
  patch_pyramid_active_runner <- suppressWarnings(as.numeric(Sys.getenv("GEOVERSA_AUTO_PATCH_PYRAMID", unset = "1")))
  patch_pyramid_active_runner <- is.finite(patch_pyramid_active_runner) && patch_pyramid_active_runner > 0
  skip_prebuilt_auto_cache <- identical(model_profile, "auto") &&
    patch_pyramid_active_runner &&
    !is.null(conv_env$auto_patch_geometry_v5) &&
    is.function(conv_env$auto_patch_geometry_v5)
  if ("ConvKrigingNet2D" %in% models && !skip_prebuilt_auto_cache) {
    calibration_patch_cache <- conv_env$build_pointpatch_patch_cache(
      context = conv_env$wadoux_context,
      calibration_df = valuetable,
      patch_size = patch_size,
      label = sprintf("scenario=%s iter=%d calibration patches", scenario, iter)
    )
  } else if ("ConvKrigingNet2D" %in% models && skip_prebuilt_auto_cache) {
    cat("[Auto patch] deferring calibration patch cache until split-specific geometry is resolved.\n")
  }
  iter_results <- list()
  iter_diagnostics <- list()
  iter_uq_results <- list()
  iter_reliability <- list()
  iter_prediction_audit <- list()
  iter_split_audit <- list()

  if ("Population" %in% protocols) {
    cat("\n--- Population ---\n")
      eval_out <- evaluate_population_protocol_optional(
        valuetable = valuetable,
        common_data = common_data,
        models = models,
        calibration_patch_cache = calibration_patch_cache,
        conv_env = conv_env,
        conv_params = conv_params,
        val_frac = val_frac,
        conformal_third_split = conformal_third_split,
        conformal_cal_frac = conformal_cal_frac,
        iter_seed = train_seed + iter,
        patch_size = patch_size,
        refit_full_calibration = refit_full_calibration,
        rf_refit_full_calibration = rf_refit_full_calibration
      )
    iter_results[["Population"]] <- eval_out$metrics
    iter_diagnostics[["Population"]] <- eval_out$diagnostics
    iter_uq_results[["Population"]] <- eval_out$uq_metrics
    iter_reliability[["Population"]] <- eval_out$reliability
  }

  if ("DesignBased" %in% protocols) {
    cat("\n--- DesignBased ---\n")
    eval_out <- if (paired_final_benchmark) {
        evaluate_design_based_protocol_paired(
          valuetable = valuetable,
          common_data = common_data,
          calibration_patch_cache = calibration_patch_cache,
          conv_env = conv_env,
          conv_params = conv_params,
          sample_size = sample_size,
          val_frac = val_frac,
	          conformal_third_split = conformal_third_split,
	          conformal_cal_frac = conformal_cal_frac,
	          iter_seed = train_seed + iter,
	          iter_index = iter,
	          patch_size = patch_size
	        )
      } else {
        evaluate_design_based_protocol(
        valuetable = valuetable,
        common_data = common_data,
        models = models,
        calibration_patch_cache = calibration_patch_cache,
        conv_env = conv_env,
          conv_params = conv_params,
          sample_size = sample_size,
          val_frac = val_frac,
	          conformal_third_split = conformal_third_split,
	          conformal_cal_frac = conformal_cal_frac,
	          iter_seed = train_seed + iter,
	          iter_index = iter,
	          patch_size = patch_size,
	          refit_full_calibration = refit_full_calibration,
          rf_refit_full_calibration = rf_refit_full_calibration
        )
    }
    iter_results[["DesignBased"]] <- eval_out$metrics
    iter_diagnostics[["DesignBased"]] <- eval_out$diagnostics
    iter_uq_results[["DesignBased"]] <- eval_out$uq_metrics
    iter_reliability[["DesignBased"]] <- eval_out$reliability
    iter_prediction_audit[["DesignBased"]] <- eval_out$prediction_audit
    iter_split_audit[["DesignBased"]] <- eval_out$split_audit
    all_paired_deltas[[iter]] <- if (!is.null(eval_out$paired_deltas) && nrow(eval_out$paired_deltas) > 0) {
      eval_out$paired_deltas %>%
        mutate(
          iteration = iter,
          scenario = scenario,
          model_profile = model_profile,
          .before = 1
        )
    } else {
      data.frame()
    }
    all_paired_meta[[iter]] <- if (!is.null(eval_out$paired_meta) && nrow(eval_out$paired_meta) > 0) {
      eval_out$paired_meta %>%
        mutate(
          iteration = iter,
          scenario = scenario,
          model_profile = model_profile,
          .before = 1
        )
    } else {
      data.frame()
    }
  }
  all_prediction_audit[[iter]] <- if (length(iter_prediction_audit) > 0) {
    bind_rows(iter_prediction_audit) %>%
      mutate(
        model_profile = model_profile,
        .after = scenario
      )
  } else {
    data.frame()
  }
  all_split_audit[[iter]] <- if (length(iter_split_audit) > 0) {
    bind_rows(iter_split_audit) %>%
      mutate(
        model_profile = model_profile,
        .after = scenario
      )
  } else {
    data.frame()
  }

  if ("RandomKFold" %in% protocols) {
    cat("\n--- RandomKFold ---\n")
      eval_out <- evaluate_random_kfold_protocol(
        valuetable = valuetable,
        models = models,
        calibration_patch_cache = calibration_patch_cache,
        conv_env = conv_env,
        conv_params = conv_params,
        random_k = random_k,
        val_frac = val_frac,
        conformal_third_split = conformal_third_split,
        conformal_cal_frac = conformal_cal_frac,
        iter_seed = train_seed + iter,
        patch_size = patch_size,
        refit_full_calibration = refit_full_calibration,
        rf_refit_full_calibration = rf_refit_full_calibration
      )
    iter_results[["RandomKFold"]] <- eval_out$metrics
    iter_diagnostics[["RandomKFold"]] <- eval_out$diagnostics
    iter_uq_results[["RandomKFold"]] <- eval_out$uq_metrics
    iter_reliability[["RandomKFold"]] <- eval_out$reliability
  }

  if ("SpatialKFold" %in% protocols) {
    cat("\n--- SpatialKFold ---\n")
      eval_out <- evaluate_spatial_kfold_protocol(
        valuetable = valuetable,
        models = models,
        calibration_patch_cache = calibration_patch_cache,
        conv_env = conv_env,
        conv_params = conv_params,
        val_dist_km = val_dist_km,
        val_frac = val_frac,
        conformal_third_split = conformal_third_split,
        conformal_cal_frac = conformal_cal_frac,
        iter_seed = train_seed + iter,
        patch_size = patch_size,
        refit_full_calibration = refit_full_calibration,
        rf_refit_full_calibration = rf_refit_full_calibration
      )
    iter_results[["SpatialKFold"]] <- eval_out$metrics
    iter_diagnostics[["SpatialKFold"]] <- eval_out$diagnostics
    iter_uq_results[["SpatialKFold"]] <- eval_out$uq_metrics
    iter_reliability[["SpatialKFold"]] <- eval_out$reliability
  }

  if ("BLOOCV" %in% protocols) {
    cat("\n--- BLOOCV ---\n")
      eval_out <- evaluate_bloocv_protocol(
        valuetable = valuetable,
        models = models,
        calibration_patch_cache = calibration_patch_cache,
        conv_env = conv_env,
        conv_params = conv_params,
        val_dist_km = val_dist_km,
        val_frac = val_frac,
        conformal_third_split = conformal_third_split,
        conformal_cal_frac = conformal_cal_frac,
        iter_seed = train_seed + iter,
        nb_groups = bloo_groups,
        nb_test_pixels = bloo_test_pixels,
      patch_size = patch_size,
      refit_full_calibration = refit_full_calibration,
      rf_refit_full_calibration = rf_refit_full_calibration
    )
    iter_results[["BLOOCV"]] <- eval_out$metrics
    iter_diagnostics[["BLOOCV"]] <- eval_out$diagnostics
    iter_uq_results[["BLOOCV"]] <- eval_out$uq_metrics
    iter_reliability[["BLOOCV"]] <- eval_out$reliability
  }

  all_results[[iter]] <- bind_rows(iter_results) %>%
    mutate(
      iteration = iter,
      scenario = scenario,
      model_profile = model_profile,
      .before = 1
    )

  iter_diag_df <- bind_rows(iter_diagnostics)
  if (nrow(iter_diag_df) > 0) {
    all_diagnostics[[iter]] <- iter_diag_df %>%
      mutate(
        iteration = iter,
        scenario = scenario,
        model_profile = model_profile,
        .before = 1
      )
  } else {
    all_diagnostics[[iter]] <- data.frame()
  }
  iter_uq_df <- bind_rows(iter_uq_results)
  if (nrow(iter_uq_df) > 0) {
    all_uq_results[[iter]] <- iter_uq_df %>%
      mutate(
        iteration = iter,
        scenario = scenario,
        model_profile = model_profile,
        .before = 1
      )
  } else {
    all_uq_results[[iter]] <- data.frame()
  }
  iter_reliability_df <- bind_rows(iter_reliability)
  if (nrow(iter_reliability_df) > 0) {
    all_reliability[[iter]] <- iter_reliability_df %>%
      mutate(
        iteration = iter,
        scenario = scenario,
        model_profile = model_profile,
        .before = 1
      )
  } else {
    all_reliability[[iter]] <- data.frame()
  }

  write_iteration_checkpoint_outputs(
    results_dir = results_dir,
    all_results = all_results,
    all_diagnostics = all_diagnostics,
    all_uq_results = all_uq_results,
    all_reliability = all_reliability,
    all_paired_deltas = all_paired_deltas,
    all_paired_meta = all_paired_meta,
    all_prediction_audit = all_prediction_audit,
    all_split_audit = all_split_audit,
    completed_iteration = iter,
    total_iterations = n_iter
  )
}

final_results <- bind_rows(all_results)
final_diagnostics <- bind_rows(all_diagnostics)
final_uq_results <- bind_rows(all_uq_results)
final_reliability <- bind_rows(all_reliability)
final_paired_deltas <- bind_rows(all_paired_deltas)
final_paired_meta <- bind_rows(all_paired_meta)

summary_by_protocol <- summarise_results_by_protocol(final_results)
diagnostic_summary <- summarise_diagnostics(final_diagnostics)
uq_summary <- summarise_uq_results_by_protocol(final_uq_results)
reliability_summary <- summarise_reliability_by_protocol(final_reliability)
paired_delta_summary <- summarise_paired_deltas(final_paired_deltas)

write_iteration_checkpoint_outputs(
  results_dir = results_dir,
  all_results = all_results,
  all_diagnostics = all_diagnostics,
  all_uq_results = all_uq_results,
  all_reliability = all_reliability,
  all_paired_deltas = all_paired_deltas,
  all_paired_meta = all_paired_meta,
  all_prediction_audit = all_prediction_audit,
  all_split_audit = all_split_audit,
  completed_iteration = n_iter,
  total_iterations = n_iter
)

cat("\n=== Wadoux-style RF vs ConvKrigingNet2D comparison complete ===\n")
print(summary_by_protocol)
if (nrow(diagnostic_summary) > 0) {
  cat("\n=== GeoVersa fold diagnostic summary ===\n")
  print(diagnostic_summary)
}
if (nrow(uq_summary) > 0) {
  cat("\n=== GeoVersa UQ summary ===\n")
  print(uq_summary)
}
if (nrow(paired_delta_summary) > 0) {
  cat("\n=== Paired delta summary ===\n")
  print(paired_delta_summary)
}
cat(sprintf("\nResults written to: %s\n", results_dir))
