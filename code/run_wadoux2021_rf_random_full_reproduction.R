rm(list = ls())
set.seed(123)

# =============================================================================
# run_wadoux2021_rf_random_full_reproduction.R
#
# Purpose:
#   Reproduce the RF experiment from Wadoux et al. (2021), following the
#   simple-random sampling script `external/SpatialValidation/code/Spat_CV_random.R`.
#
# What is reproduced here:
#   - response: ABG1
#   - predictors: same 28-covariate set used in the original script
#   - calibration sample: simple random sample of size 500
#   - RF model: ranger with default settings
#   - protocols inside the same iteration loop:
#       * Population
#       * DesignBased
#       * RandomKFold
#       * SpatialKFold
#       * BLOOCV
#   - number of repetitions: 500 by default
#
# Notes:
#   - This script is intentionally separate from the neural benchmark.
#   - The buffered LOO exclusion is implemented from Euclidean distances on the
#     projected x/y coordinates of the SpatialValidation data, which is
#     equivalent to the circular exclusion used in the original script.
#   - Defaults can be overridden via environment variables for smoke tests:
#       WADOUX_N_ITER
#       WADOUX_SAMPLE_SIZE
#       WADOUX_VAL_DIST_KM
#       WADOUX_RANDOM_K
#       WADOUX_BLOO_GROUPS
#       WADOUX_BLOO_TEST_PIXELS
#       WADOUX_RESULTS_DIR
# =============================================================================

project_root <- "C:/Users/rodrigues.h/OneDrive/Deep Kriging"
external_root <- file.path(project_root, "external", "SpatialValidation")
results_dir <- Sys.getenv(
  "WADOUX_RESULTS_DIR",
  unset = file.path(project_root, "results", "wadoux2021_rf_random_full_reproduction")
)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

pkgs <- c("raster", "ranger", "caret", "dplyr")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

library(raster)
library(ranger)
library(caret)
library(dplyr)

SAMPLE_SIZE <- as.integer(Sys.getenv("WADOUX_SAMPLE_SIZE", unset = "500"))
N_ITER <- as.integer(Sys.getenv("WADOUX_N_ITER", unset = "500"))
VAL_DIST_KM <- as.numeric(Sys.getenv("WADOUX_VAL_DIST_KM", unset = "350"))
RANDOM_K <- as.integer(Sys.getenv("WADOUX_RANDOM_K", unset = "10"))
BLOO_GROUPS <- as.integer(Sys.getenv("WADOUX_BLOO_GROUPS", unset = "10"))
BLOO_TEST_PIXELS <- as.integer(Sys.getenv("WADOUX_BLOO_TEST_PIXELS", unset = "100"))
CHECKPOINT_EVERY <- 10L

predList_modelfull <- c(
  "AI_glob", "CC_am", "Clay", "Elev", "ETP_Glob", "G_mean", "NIR_mean",
  "OCS", "Prec_am", "Prec_Dm", "Prec_seaso", "Prec_Wm", "R_mean", "Sand",
  "Sha_EVI", "Slope", "Soc", "solRad_m", "SolRad_sd", "SWIR1_mean",
  "SWIR2_mean", "T_am", "T_mdq", "T_mwarmq", "T_seaso", "Terra_PP",
  "Vapor_m", "Vapor_sd"
)
response_name <- "ABG1"

wadoux_eval <- function(obs, pred) {
  me <- round(mean(pred - obs, na.rm = TRUE), digits = 2)
  rmse <- round(sqrt(mean((pred - obs)^2, na.rm = TRUE)), digits = 2)
  r2 <- round((cor(pred, obs, method = "spearman", use = "pairwise.complete.obs")^2), digits = 2)
  sse <- sum((pred - obs)^2, na.rm = TRUE)
  sst <- sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE)
  mec <- round((1 - sse / sst), digits = 2)
  data.frame(ME = me, RMSE = rmse, r2 = r2, MEC = mec)
}

fit_rf_default <- function(train_df, form_rf) {
  ranger::ranger(formula = form_rf, data = train_df)
}

predict_rf_default <- function(model, new_df) {
  as.numeric(predict(model, data = new_df, type = "response")$predictions)
}

sample_simple_random_rows <- function(df, size) {
  df[sample.int(nrow(df), size = size, replace = FALSE), , drop = FALSE]
}

build_spatial_folds <- function(valuetable, val_dist_km) {
  mdist <- dist(valuetable[, c("x", "y")])
  hc <- hclust(mdist, method = "complete")
  cutree(hc, h = val_dist_km * 1000)
}

run_population_protocol <- function(valuetable, s_df, form_rf) {
  rf <- fit_rf_default(valuetable, form_rf)
  pred <- predict_rf_default(rf, s_df)
  wadoux_eval(obs = s_df[[response_name]], pred = pred)
}

run_design_based_protocol <- function(valuetable, s_df, form_rf) {
  val_srs <- sample_simple_random_rows(s_df, SAMPLE_SIZE)
  rf <- fit_rf_default(valuetable, form_rf)
  pred <- predict_rf_default(rf, val_srs)
  wadoux_eval(obs = val_srs[[response_name]], pred = pred)
}

run_random_kfold_protocol <- function(valuetable, form_rf, k = 10) {
  flds <- caret::createFolds(
    valuetable[[response_name]],
    k = k,
    list = TRUE,
    returnTrain = FALSE
  )

  pred_list <- vector("list", length(flds))
  obs_list <- vector("list", length(flds))

  for (j in seq_along(flds)) {
    id <- flds[[j]]
    training_data <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]

    rf <- fit_rf_default(training_data, form_rf)
    pred_list[[j]] <- predict_rf_default(rf, validation_data)
    obs_list[[j]] <- validation_data[[response_name]]
  }

  wadoux_eval(
    obs = unlist(obs_list, use.names = FALSE),
    pred = unlist(pred_list, use.names = FALSE)
  )
}

run_spatial_kfold_protocol <- function(valuetable, form_rf, val_dist_km) {
  spatial_folds <- build_spatial_folds(valuetable, val_dist_km)
  fold_ids <- unique(spatial_folds)

  pred_list <- vector("list", length(fold_ids))
  obs_list <- vector("list", length(fold_ids))

  for (j in seq_along(fold_ids)) {
    id <- which(spatial_folds == fold_ids[j])
    training_data <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]

    rf <- fit_rf_default(training_data, form_rf)
    pred_list[[j]] <- predict_rf_default(rf, validation_data)
    obs_list[[j]] <- validation_data[[response_name]]
  }

  wadoux_eval(
    obs = unlist(obs_list, use.names = FALSE),
    pred = unlist(pred_list, use.names = FALSE)
  )
}

point_within_predictor_range <- function(train_df, focal_row, predictor_names) {
  train_pred <- train_df[, predictor_names, drop = FALSE]
  focal_pred <- focal_row[, predictor_names, drop = FALSE]
  lower <- apply(train_pred, 2, min, na.rm = TRUE)
  upper <- apply(train_pred, 2, max, na.rm = TRUE)
  all(as.numeric(focal_pred[1, ]) >= lower & as.numeric(focal_pred[1, ]) <= upper)
}

exclude_by_radius <- function(df, focal_row, radius_m) {
  dx <- df$x - focal_row$x
  dy <- df$y - focal_row$y
  keep <- sqrt(dx^2 + dy^2) > radius_m
  df[keep, , drop = FALSE]
}

run_bloocv_protocol <- function(valuetable,
                                form_rf,
                                predictor_names,
                                val_dist_km,
                                nb_groups = 10,
                                nb_test_pixels = 100) {
  r_list <- c(val_dist_km)
  nb_iteration <- nb_groups * nb_test_pixels
  data_res <- vector("list", nb_iteration * length(r_list))
  a <- 0L

  for (j in seq_len(nb_iteration)) {
    point_in_range <- FALSE

    while (!point_in_range) {
      id_focal <- sample.int(nrow(valuetable), size = 1)
      focal_point <- valuetable[id_focal, , drop = FALSE]
      training_tmp <- valuetable[-id_focal, , drop = FALSE]

      ri <- max(r_list) * 1000
      training_tmp <- exclude_by_radius(training_tmp, focal_point, ri)

      if (nrow(training_tmp) < 10) {
        next
      }

      if (!point_within_predictor_range(training_tmp, focal_point, predictor_names)) {
        next
      }

      nb_training <- nrow(training_tmp)
      point_in_range <- TRUE
    }

    training_tmp_j <- valuetable[-id_focal, , drop = FALSE]

    for (i in seq_along(r_list)) {
      ri <- r_list[i] * 1000
      training_tmp <- exclude_by_radius(training_tmp_j, focal_point, ri)

      if (nrow(training_tmp) < nb_training) {
        next
      }

      nb_training_dif <- nrow(training_tmp) - nb_training
      if (nb_training_dif > 0) {
        training_tmp <- training_tmp[
          -sample.int(nrow(training_tmp), size = nb_training_dif),
          ,
          drop = FALSE
        ]
      }

      loss_cells <- nrow(training_tmp_j) - nrow(training_tmp)

      rf <- fit_rf_default(training_tmp, form_rf)
      pred_rf <- predict_rf_default(rf, focal_point)

      a <- a + 1L
      data_res[[a]] <- data.frame(
        R = r_list[i],
        N_training = nb_training,
        N_cell_lost = loss_cells,
        N_cell_training = nrow(training_tmp),
        AGB = focal_point[[response_name]],
        Pred_RF_FULL = pred_rf,
        AGB_calibrationDATA = mean(training_tmp[[response_name]], na.rm = TRUE)
      )
    }
  }

  data_res <- bind_rows(data_res)

  x <- unique(data_res$R)
  bloo_metrics <- vector("list", length(x))

  for (i in seq_along(x)) {
    tmp <- data_res[data_res$R == x[i], , drop = FALSE]
    tmp$flds <- caret::createFolds(
      seq_len(nrow(tmp)),
      k = nb_groups,
      list = FALSE,
      returnTrain = FALSE
    )

    rmse_null_list <- c()
    r2_rf_list <- c()
    me_rf_list <- c()
    rmse_rf_list <- c()
    mec_rf_list <- c()

    for (j in seq_len(max(tmp$flds))) {
      idx <- which(tmp$flds == j)
      rmse_null_list <- c(
        rmse_null_list,
        wadoux_eval(obs = tmp$AGB_calibrationDATA[idx], pred = tmp$AGB[idx])$RMSE
      )

      map_quality <- wadoux_eval(obs = tmp$AGB[idx], pred = tmp$Pred_RF_FULL[idx])
      r2_rf_list <- c(r2_rf_list, map_quality$r2)
      me_rf_list <- c(me_rf_list, map_quality$ME)
      rmse_rf_list <- c(rmse_rf_list, map_quality$RMSE)
      mec_rf_list <- c(mec_rf_list, map_quality$MEC)
    }

    bloo_metrics[[i]] <- data.frame(
      ME = mean(me_rf_list, na.rm = TRUE),
      RMSE = mean(rmse_rf_list, na.rm = TRUE),
      r2 = mean(r2_rf_list, na.rm = TRUE),
      MEC = mean(mec_rf_list, na.rm = TRUE)
    )
  }

  bind_rows(bloo_metrics)[1, , drop = FALSE]
}

make_metric_table <- function(population_metrics,
                              design_metrics,
                              random_metrics,
                              spatial_metrics,
                              bloo_metrics) {
  metric_names <- c("ME", "RMSE", "r2", "MEC")
  data.frame(
    metric = metric_names,
    Population = unname(unlist(population_metrics[1, metric_names])),
    DesignBased = unname(unlist(design_metrics[1, metric_names])),
    RandomKFold = unname(unlist(random_metrics[1, metric_names])),
    SpatialKFold = unname(unlist(spatial_metrics[1, metric_names])),
    BLOOCV = unname(unlist(bloo_metrics[1, metric_names]))
  )
}

make_delta_table <- function(metric_table) {
  data.frame(
    metric = metric_table$metric,
    DesignBased = metric_table$DesignBased - metric_table$Population,
    RandomKFold = metric_table$RandomKFold - metric_table$Population,
    SpatialKFold = metric_table$SpatialKFold - metric_table$Population,
    BLOOCV = metric_table$BLOOCV - metric_table$Population
  )
}

save_checkpoint <- function(iteration, absolute_rows, delta_rows) {
  absolute_df <- bind_rows(absolute_rows)
  delta_df <- bind_rows(delta_rows)

  write.csv(
    absolute_df,
    file.path(results_dir, "wadoux2021_rf_random_absolute_metrics.csv"),
    row.names = FALSE
  )
  write.csv(
    delta_df,
    file.path(results_dir, "wadoux2021_rf_random_delta_metrics.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(absolute = absolute_df, delta = delta_df, iteration = iteration),
    file.path(results_dir, "wadoux2021_rf_random_checkpoint.rds")
  )
}

cat("Loading SpatialValidation raster stack...\n")
files <- list.files(path = file.path(external_root, "data"), pattern = "\\.tif$", full.names = TRUE)
s <- raster::stack(files)
s[[1]][s[[1]] == 0] <- NA
s_df <- as.data.frame(s, xy = TRUE, na.rm = TRUE)
s_df <- na.omit(as.data.frame(s_df))

form_rf <- as.formula(paste(response_name, "~", paste(predList_modelfull, collapse = "+")))

absolute_rows <- vector("list", N_ITER)
delta_rows <- vector("list", N_ITER)

config <- data.frame(
  sample_size = SAMPLE_SIZE,
  n_iter = N_ITER,
  val_dist_km = VAL_DIST_KM,
  random_k = RANDOM_K,
  bloo_groups = BLOO_GROUPS,
  bloo_test_pixels = BLOO_TEST_PIXELS,
  stringsAsFactors = FALSE
)
write.csv(config, file.path(results_dir, "wadoux2021_rf_random_config.csv"), row.names = FALSE)

for (sampling in seq_len(N_ITER)) {
  cat(sprintf("\n==================================================\n"))
  cat(sprintf("Wadoux RF random reproduction | iteration %d / %d\n", sampling, N_ITER))
  cat(sprintf("==================================================\n"))

  valuetable <- sample_simple_random_rows(s_df, SAMPLE_SIZE)
  valuetable <- na.omit(as.data.frame(valuetable))

  population_metrics <- run_population_protocol(valuetable, s_df, form_rf)
  design_metrics <- run_design_based_protocol(valuetable, s_df, form_rf)
  random_metrics <- run_random_kfold_protocol(valuetable, form_rf, k = RANDOM_K)
  spatial_metrics <- run_spatial_kfold_protocol(valuetable, form_rf, val_dist_km = VAL_DIST_KM)
  bloo_metrics <- run_bloocv_protocol(
    valuetable = valuetable,
    form_rf = form_rf,
    predictor_names = predList_modelfull,
    val_dist_km = VAL_DIST_KM,
    nb_groups = BLOO_GROUPS,
    nb_test_pixels = BLOO_TEST_PIXELS
  )

  metric_table <- make_metric_table(
    population_metrics = population_metrics,
    design_metrics = design_metrics,
    random_metrics = random_metrics,
    spatial_metrics = spatial_metrics,
    bloo_metrics = bloo_metrics
  )
  delta_table <- make_delta_table(metric_table)

  absolute_rows[[sampling]] <- mutate(metric_table, iteration = sampling, .before = 1)
  delta_rows[[sampling]] <- mutate(delta_table, iteration = sampling, .before = 1)

  if (sampling %% CHECKPOINT_EVERY == 0L || sampling == N_ITER) {
    save_checkpoint(sampling, absolute_rows, delta_rows)
  }
}

absolute_df <- bind_rows(absolute_rows)
delta_df <- bind_rows(delta_rows)

absolute_long <- bind_rows(lapply(seq_len(nrow(absolute_df)), function(i) {
  row <- absolute_df[i, ]
  data.frame(
    iteration = row$iteration,
    metric = row$metric,
    protocol = c("Population", "DesignBased", "RandomKFold", "SpatialKFold", "BLOOCV"),
    value = c(row$Population, row$DesignBased, row$RandomKFold, row$SpatialKFold, row$BLOOCV)
  )
}))

delta_long <- bind_rows(lapply(seq_len(nrow(delta_df)), function(i) {
  row <- delta_df[i, ]
  data.frame(
    iteration = row$iteration,
    metric = row$metric,
    protocol = c("DesignBased", "RandomKFold", "SpatialKFold", "BLOOCV"),
    delta = c(row$DesignBased, row$RandomKFold, row$SpatialKFold, row$BLOOCV)
  )
}))

absolute_summary <- absolute_long %>%
  group_by(metric, protocol) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    sd_value = sd(value, na.rm = TRUE),
    median_value = median(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(metric, protocol)

delta_summary <- delta_long %>%
  group_by(metric, protocol) %>%
  summarise(
    mean_delta = mean(delta, na.rm = TRUE),
    sd_delta = sd(delta, na.rm = TRUE),
    median_delta = median(delta, na.rm = TRUE),
    q05_delta = unname(quantile(delta, probs = 0.05, na.rm = TRUE)),
    q95_delta = unname(quantile(delta, probs = 0.95, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  arrange(metric, protocol)

write.csv(absolute_long, file.path(results_dir, "wadoux2021_rf_random_absolute_long.csv"), row.names = FALSE)
write.csv(delta_long, file.path(results_dir, "wadoux2021_rf_random_delta_long.csv"), row.names = FALSE)
write.csv(absolute_summary, file.path(results_dir, "wadoux2021_rf_random_absolute_summary.csv"), row.names = FALSE)
write.csv(delta_summary, file.path(results_dir, "wadoux2021_rf_random_delta_summary.csv"), row.names = FALSE)

cat("\n=== Full Wadoux-style RF random reproduction complete ===\n")
print(delta_summary)
cat(sprintf("\nResults written to: %s\n", results_dir))
