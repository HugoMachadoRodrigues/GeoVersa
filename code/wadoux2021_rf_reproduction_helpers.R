project_root_wadoux <- if (
  file.exists("C:/Users/rodrigues.h/OneDrive/Deep Kriging")
) {
  "C:/Users/rodrigues.h/OneDrive/Deep Kriging"
} else {
  normalizePath(
    file.path(Sys.getenv("HOME"),
              "Library/CloudStorage/OneDrive-Personal/Deep Kriging"),
    mustWork = FALSE
  )
}
external_root_wadoux <- file.path(project_root_wadoux, "external", "SpatialValidation")

pkgs_wadoux <- c("raster", "ranger", "caret", "dplyr", "sp")
to_install_wadoux <- pkgs_wadoux[!sapply(pkgs_wadoux, requireNamespace, quietly = TRUE)]
if (length(to_install_wadoux) > 0) {
  install.packages(to_install_wadoux, repos = "https://cloud.r-project.org")
}

library(raster)
library(ranger)
library(caret)
library(dplyr)
library(sp)

predList_modelfull_wadoux <- c(
  "AI_glob", "CC_am", "Clay", "Elev", "ETP_Glob", "G_mean", "NIR_mean",
  "OCS", "Prec_am", "Prec_Dm", "Prec_seaso", "Prec_Wm", "R_mean", "Sand",
  "Sha_EVI", "Slope", "Soc", "solRad_m", "SolRad_sd", "SWIR1_mean",
  "SWIR2_mean", "T_am", "T_mdq", "T_mwarmq", "T_seaso", "Terra_PP",
  "Vapor_m", "Vapor_sd"
)
response_name_wadoux <- "ABG1"

load_wadoux_common_data <- function(include_polygon = FALSE) {
  files <- list.files(
    path = file.path(external_root_wadoux, "data"),
    pattern = "\\.tif$",
    full.names = TRUE
  )
  s <- raster::stack(files)
  s[[1]][s[[1]] == 0] <- NA
  s_df <- as.data.frame(s, xy = TRUE, na.rm = TRUE)
  s_df <- na.omit(as.data.frame(s_df))

  out <- list(
    stack = s,
    s_df = s_df
  )

  if (isTRUE(include_polygon)) {
    poly_env <- new.env(parent = emptyenv())
    load(file.path(external_root_wadoux, "code", "polygon.Rdata"), envir = poly_env)
    out$pp <- poly_env$pp
  }

  out
}

wadoux_eval <- function(obs, pred) {
  me <- round(mean(pred - obs, na.rm = TRUE), digits = 2)
  rmse <- round(sqrt(mean((pred - obs)^2, na.rm = TRUE)), digits = 2)
  r2 <- round((cor(pred, obs, method = "spearman", use = "pairwise.complete.obs")^2), digits = 2)
  sse <- sum((pred - obs)^2, na.rm = TRUE)
  sst <- sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE)
  mec <- round((1 - sse / sst), digits = 2)
  data.frame(ME = me, RMSE = rmse, r2 = r2, MEC = mec)
}

fit_rf_default_wadoux <- function(train_df, form_rf) {
  ranger::ranger(formula = form_rf, data = train_df)
}

predict_rf_default_wadoux <- function(model, new_df) {
  as.numeric(predict(model, data = new_df, type = "response")$predictions)
}

sample_simple_random_rows_wadoux <- function(df, size) {
  df[sample.int(nrow(df), size = size, replace = FALSE), , drop = FALSE]
}

build_spatial_folds_wadoux <- function(valuetable, val_dist_km) {
  mdist <- dist(valuetable[, c("x", "y")])
  hc <- hclust(mdist, method = "complete")
  cutree(hc, h = val_dist_km * 1000)
}

run_population_protocol_wadoux <- function(valuetable, s_df, form_rf) {
  rf <- fit_rf_default_wadoux(valuetable, form_rf)
  pred <- predict_rf_default_wadoux(rf, s_df)
  wadoux_eval(obs = s_df[[response_name_wadoux]], pred = pred)
}

run_design_based_protocol_wadoux <- function(valuetable, s_df, form_rf, sample_size) {
  val_srs <- sample_simple_random_rows_wadoux(s_df, sample_size)
  rf <- fit_rf_default_wadoux(valuetable, form_rf)
  pred <- predict_rf_default_wadoux(rf, val_srs)
  wadoux_eval(obs = val_srs[[response_name_wadoux]], pred = pred)
}

run_random_kfold_protocol_wadoux <- function(valuetable, form_rf, k = 10) {
  flds <- caret::createFolds(
    valuetable[[response_name_wadoux]],
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

    rf <- fit_rf_default_wadoux(training_data, form_rf)
    pred_list[[j]] <- predict_rf_default_wadoux(rf, validation_data)
    obs_list[[j]] <- validation_data[[response_name_wadoux]]
  }

  wadoux_eval(
    obs = unlist(obs_list, use.names = FALSE),
    pred = unlist(pred_list, use.names = FALSE)
  )
}

run_spatial_kfold_protocol_wadoux <- function(valuetable, form_rf, val_dist_km) {
  spatial_folds <- build_spatial_folds_wadoux(valuetable, val_dist_km)
  fold_ids <- unique(spatial_folds)

  pred_list <- vector("list", length(fold_ids))
  obs_list <- vector("list", length(fold_ids))

  for (j in seq_along(fold_ids)) {
    id <- which(spatial_folds == fold_ids[j])
    training_data <- valuetable[-id, , drop = FALSE]
    validation_data <- valuetable[id, , drop = FALSE]

    rf <- fit_rf_default_wadoux(training_data, form_rf)
    pred_list[[j]] <- predict_rf_default_wadoux(rf, validation_data)
    obs_list[[j]] <- validation_data[[response_name_wadoux]]
  }

  wadoux_eval(
    obs = unlist(obs_list, use.names = FALSE),
    pred = unlist(pred_list, use.names = FALSE)
  )
}

point_within_predictor_range_wadoux <- function(train_df, focal_row, predictor_names) {
  train_pred <- train_df[, predictor_names, drop = FALSE]
  focal_pred <- focal_row[, predictor_names, drop = FALSE]
  lower <- apply(train_pred, 2, min, na.rm = TRUE)
  upper <- apply(train_pred, 2, max, na.rm = TRUE)
  all(as.numeric(focal_pred[1, ]) >= lower & as.numeric(focal_pred[1, ]) <= upper)
}

exclude_by_radius_wadoux <- function(df, focal_row, radius_m) {
  dx <- df$x - focal_row$x
  dy <- df$y - focal_row$y
  keep <- sqrt(dx^2 + dy^2) > radius_m
  df[keep, , drop = FALSE]
}

run_bloocv_protocol_wadoux <- function(valuetable,
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
      training_tmp <- exclude_by_radius_wadoux(training_tmp, focal_point, ri)

      if (nrow(training_tmp) < 10) {
        next
      }

      if (!point_within_predictor_range_wadoux(training_tmp, focal_point, predictor_names)) {
        next
      }

      nb_training <- nrow(training_tmp)
      point_in_range <- TRUE
    }

    training_tmp_j <- valuetable[-id_focal, , drop = FALSE]

    for (i in seq_along(r_list)) {
      ri <- r_list[i] * 1000
      training_tmp <- exclude_by_radius_wadoux(training_tmp_j, focal_point, ri)

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

      rf <- fit_rf_default_wadoux(training_tmp, form_rf)
      pred_rf <- predict_rf_default_wadoux(rf, focal_point)

      a <- a + 1L
      data_res[[a]] <- data.frame(
        R = r_list[i],
        N_training = nb_training,
        N_cell_lost = loss_cells,
        N_cell_training = nrow(training_tmp),
        AGB = focal_point[[response_name_wadoux]],
        Pred_RF_FULL = pred_rf,
        AGB_calibrationDATA = mean(training_tmp[[response_name_wadoux]], na.rm = TRUE)
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

    r2_rf_list <- c()
    me_rf_list <- c()
    rmse_rf_list <- c()
    mec_rf_list <- c()

    for (j in seq_len(max(tmp$flds))) {
      idx <- which(tmp$flds == j)
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

make_metric_table_wadoux <- function(population_metrics,
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

make_delta_table_wadoux <- function(metric_table) {
  data.frame(
    metric = metric_table$metric,
    DesignBased = metric_table$DesignBased - metric_table$Population,
    RandomKFold = metric_table$RandomKFold - metric_table$Population,
    SpatialKFold = metric_table$SpatialKFold - metric_table$Population,
    BLOOCV = metric_table$BLOOCV - metric_table$Population
  )
}

save_checkpoint_wadoux <- function(iteration, absolute_rows, delta_rows, results_dir, prefix) {
  absolute_df <- bind_rows(absolute_rows)
  delta_df <- bind_rows(delta_rows)

  write.csv(
    absolute_df,
    file.path(results_dir, sprintf("%s_absolute_metrics.csv", prefix)),
    row.names = FALSE
  )
  write.csv(
    delta_df,
    file.path(results_dir, sprintf("%s_delta_metrics.csv", prefix)),
    row.names = FALSE
  )
  saveRDS(
    list(absolute = absolute_df, delta = delta_df, iteration = iteration),
    file.path(results_dir, sprintf("%s_checkpoint.rds", prefix))
  )
}

finalize_outputs_wadoux <- function(absolute_rows, delta_rows, results_dir, prefix) {
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

  write.csv(
    absolute_long,
    file.path(results_dir, sprintf("%s_absolute_long.csv", prefix)),
    row.names = FALSE
  )
  write.csv(
    delta_long,
    file.path(results_dir, sprintf("%s_delta_long.csv", prefix)),
    row.names = FALSE
  )
  write.csv(
    absolute_summary,
    file.path(results_dir, sprintf("%s_absolute_summary.csv", prefix)),
    row.names = FALSE
  )
  write.csv(
    delta_summary,
    file.path(results_dir, sprintf("%s_delta_summary.csv", prefix)),
    row.names = FALSE
  )

  list(
    absolute = absolute_df,
    delta = delta_df,
    absolute_long = absolute_long,
    delta_long = delta_long,
    absolute_summary = absolute_summary,
    delta_summary = delta_summary
  )
}

sample_valuetable_regular_grid_wadoux <- function(common_data, sample_size) {
  pt.reg <- sp::spsample(common_data$pp, n = sample_size, type = "regular")
  valuetable <- raster::extract(common_data$stack, pt.reg, sp = TRUE, df = TRUE, na.rm = TRUE)
  valuetable <- na.omit(as.data.frame(valuetable))
  names(valuetable)[names(valuetable) == "x1"] <- "x"
  names(valuetable)[names(valuetable) == "x2"] <- "y"
  valuetable
}

twostage_wadoux <- function(sframe, psu, n, m) {
  units <- sample.int(nrow(sframe), size = n, replace = TRUE)
  mypsusample <- sframe[units, psu]
  ssunits <- NULL
  for (psunit in mypsusample) {
    ssunit <- sample(x = which(sframe[, psu] == psunit), size = m, replace = FALSE)
    ssunits <- c(ssunits, ssunit)
  }
  psudraw <- rep(seq_len(n), each = m)
  data.frame(sframe[ssunits, ], psudraw)
}

sample_valuetable_clustered_random_wadoux <- function(common_data,
                                                      kmeans_centers = 100,
                                                      n_psu = 20,
                                                      m_per_psu = 25) {
  sp.grid <- common_data$s_df[c("x", "y")]
  tkmean <- stats::kmeans(sp.grid, centers = kmeans_centers, nstart = 10)
  sp.grid$psu <- tkmean$cluster
  mysample <- twostage_wadoux(sframe = sp.grid, psu = "psu", n = n_psu, m = m_per_psu)
  sp::coordinates(mysample) <- ~x + y
  valuetable <- raster::extract(common_data$stack, mysample, sp = TRUE, df = TRUE, na.rm = TRUE)
  valuetable <- na.omit(as.data.frame(valuetable))
  names(valuetable)[names(valuetable) == "x1"] <- "x"
  names(valuetable)[names(valuetable) == "x2"] <- "y"
  valuetable
}

run_wadoux_rf_reproduction <- function(common_data,
                                       sample_valuetable_fn,
                                       sample_args,
                                       results_dir,
                                       prefix,
                                       scenario_name,
                                       config_extra = list(),
                                       sample_size = 500L,
                                       n_iter = 500L,
                                       val_dist_km = 350,
                                       random_k = 10L,
                                       bloo_groups = 10L,
                                       bloo_test_pixels = 100L,
                                       checkpoint_every = 10L) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  form_rf <- as.formula(
    paste(response_name_wadoux, "~", paste(predList_modelfull_wadoux, collapse = "+"))
  )

  config <- c(
    list(
      scenario = scenario_name,
      sample_size = sample_size,
      n_iter = n_iter,
      val_dist_km = val_dist_km,
      random_k = random_k,
      bloo_groups = bloo_groups,
      bloo_test_pixels = bloo_test_pixels
    ),
    config_extra
  )
  write.csv(
    as.data.frame(config, stringsAsFactors = FALSE),
    file.path(results_dir, sprintf("%s_config.csv", prefix)),
    row.names = FALSE
  )

  absolute_rows <- vector("list", n_iter)
  delta_rows <- vector("list", n_iter)

  for (sampling in seq_len(n_iter)) {
    cat(sprintf("\n==================================================\n"))
    cat(sprintf("Wadoux RF %s reproduction | iteration %d / %d\n", scenario_name, sampling, n_iter))
    cat(sprintf("==================================================\n"))

    valuetable <- do.call(
      sample_valuetable_fn,
      c(list(common_data = common_data), sample_args)
    )
    valuetable <- na.omit(as.data.frame(valuetable))

    population_metrics <- run_population_protocol_wadoux(valuetable, common_data$s_df, form_rf)
    design_metrics <- run_design_based_protocol_wadoux(valuetable, common_data$s_df, form_rf, sample_size)
    random_metrics <- run_random_kfold_protocol_wadoux(valuetable, form_rf, k = random_k)
    spatial_metrics <- run_spatial_kfold_protocol_wadoux(valuetable, form_rf, val_dist_km = val_dist_km)
    bloo_metrics <- run_bloocv_protocol_wadoux(
      valuetable = valuetable,
      form_rf = form_rf,
      predictor_names = predList_modelfull_wadoux,
      val_dist_km = val_dist_km,
      nb_groups = bloo_groups,
      nb_test_pixels = bloo_test_pixels
    )

    metric_table <- make_metric_table_wadoux(
      population_metrics = population_metrics,
      design_metrics = design_metrics,
      random_metrics = random_metrics,
      spatial_metrics = spatial_metrics,
      bloo_metrics = bloo_metrics
    )
    delta_table <- make_delta_table_wadoux(metric_table)

    absolute_rows[[sampling]] <- mutate(metric_table, iteration = sampling, .before = 1)
    delta_rows[[sampling]] <- mutate(delta_table, iteration = sampling, .before = 1)

    if (sampling %% checkpoint_every == 0L || sampling == n_iter) {
      save_checkpoint_wadoux(sampling, absolute_rows, delta_rows, results_dir, prefix)
    }
  }

  finalize_outputs_wadoux(absolute_rows, delta_rows, results_dir, prefix)
}
