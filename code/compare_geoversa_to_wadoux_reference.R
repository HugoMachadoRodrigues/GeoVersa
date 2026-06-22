rm(list = ls())

# Compare a GeoVersa benchmark run against the Wadoux-style RF reference.
#
# Inputs are configured through environment variables:
#   GEOVERSA_RESULTS_DIR          GeoVersa run directory containing
#                                 wadoux_style_rf_conv_summary_by_protocol.csv
#   WADOUX_RF_REFERENCE_DIR       RF reference directory containing
#                                 wadoux2021_*_absolute_summary.csv
#   WADOUX_OFFICIAL_DELTA_SUMMARY Optional official imported Wadoux delta summary
#   WADOUX_COMPARE_PROTOCOL       Protocol to score, default DesignBased
#   WADOUX_COMPARE_MODEL          GeoVersa model name, default ConvKrigingNet2D
#   WADOUX_COMPARE_OUT_DIR        Output directory, default GEOVERSA_RESULTS_DIR

env_or <- function(name, default) {
  value <- Sys.getenv(name, unset = default)
  if (!nzchar(value)) default else value
}

path_or <- function(name, default) {
  normalizePath(env_or(name, default), mustWork = FALSE)
}

project_root <- normalizePath(getwd(), mustWork = FALSE)

geoversa_dir <- path_or(
  "GEOVERSA_RESULTS_DIR",
  file.path(project_root, "results", "codex_geoversa_meta_smoketest_1iter")
)
rf_reference_dir <- path_or(
  "WADOUX_RF_REFERENCE_DIR",
  file.path(project_root, "results", "wadoux2021_rf_reference_random_designbased_30iter_pearson_20260406")
)
official_delta_summary <- path_or(
  "WADOUX_OFFICIAL_DELTA_SUMMARY",
  file.path(project_root, "docs", "wadoux2021-reference", "official_delta_summary.csv")
)
protocol_target <- env_or("WADOUX_COMPARE_PROTOCOL", "DesignBased")
model_target <- env_or("WADOUX_COMPARE_MODEL", "ConvKrigingNet2D")
out_dir <- path_or("WADOUX_COMPARE_OUT_DIR", geoversa_dir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

summary_path <- file.path(geoversa_dir, "wadoux_style_rf_conv_summary_by_protocol.csv")
status_path <- file.path(geoversa_dir, "wadoux_style_rf_conv_checkpoint_status.csv")
config_path <- file.path(geoversa_dir, "wadoux_style_rf_conv_config.csv")
diag_path <- file.path(geoversa_dir, "wadoux_style_rf_conv_fold_diagnostics.csv")

read_csv_required <- function(path, label) {
  if (!file.exists(path)) {
    stop(sprintf("%s not found: %s", label, path), call. = FALSE)
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

safe_read_csv <- function(path) {
  if (!file.exists(path)) {
    return(NULL)
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

first_or_na <- function(x) {
  if (length(x) == 0L) NA else x[[1]]
}

numeric_or_na <- function(x) {
  value <- suppressWarnings(as.numeric(first_or_na(x)))
  if (length(value) == 0L || !is.finite(value)) NA_real_ else value
}

finite_mean <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

finite_sd <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) <= 1L) NA_real_ else stats::sd(x)
}

value_col <- function(df, col) {
  if (col %in% names(df)) numeric_or_na(df[[col]]) else NA_real_
}

metric_cols <- c(
  ME = "ME_mean",
  RMSE = "RMSE_mean",
  r2 = "r2_mean",
  MEC = "MEC_mean"
)

metric_sd_cols <- c(
  ME = "ME_sd",
  RMSE = "RMSE_sd",
  r2 = "r2_sd",
  MEC = "MEC_sd"
)

better_absolute_metric <- function(metric, value, reference) {
  if (!is.finite(value) || !is.finite(reference)) {
    return(NA)
  }
  if (identical(metric, "RMSE")) {
    return(value <= reference)
  }
  if (identical(metric, "ME")) {
    return(abs(value) <= abs(reference))
  }
  if (metric %in% c("r2", "MEC")) {
    return(value >= reference)
  }
  NA
}

closer_to_zero <- function(value, reference) {
  if (!is.finite(value) || !is.finite(reference)) {
    return(NA)
  }
  abs(value) <= abs(reference)
}

make_empty_comparison <- function() {
  data.frame(
    comparison_type = character(),
    scenario = character(),
    protocol = character(),
    model = character(),
    reference_label = character(),
    metric = character(),
    geoversa_value = numeric(),
    reference_value = numeric(),
    delta_geoversa_minus_reference = numeric(),
    better_than_reference = logical(),
    stringsAsFactors = FALSE
  )
}

append_comparison <- function(rows,
                              comparison_type,
                              scenario,
                              protocol,
                              model,
                              reference_label,
                              metric,
                              geoversa_value,
                              reference_value,
                              better_than_reference) {
  rows[[length(rows) + 1L]] <- data.frame(
    comparison_type = comparison_type,
    scenario = scenario,
    protocol = protocol,
    model = model,
    reference_label = reference_label,
    metric = metric,
    geoversa_value = geoversa_value,
    reference_value = reference_value,
    delta_geoversa_minus_reference = geoversa_value - reference_value,
    better_than_reference = better_than_reference,
    stringsAsFactors = FALSE
  )
  rows
}

summarise_run_status <- function(status_df, config_df, diag_df, target_row) {
  completed_iteration <- NA_real_
  total_iterations <- NA_real_
  run_complete <- NA
  if (!is.null(status_df) && nrow(status_df) > 0L) {
    completed_iteration <- numeric_or_na(status_df$completed_iteration)
    total_iterations <- numeric_or_na(status_df$total_iterations)
    if (is.finite(completed_iteration) && is.finite(total_iterations)) {
      run_complete <- completed_iteration >= total_iterations
    }
  }

  sample_size <- if (!is.null(config_df) && "sample_size" %in% names(config_df)) {
    numeric_or_na(config_df$sample_size)
  } else {
    NA_real_
  }
  n_iter <- if (!is.null(config_df) && "n_iter" %in% names(config_df)) {
    numeric_or_na(config_df$n_iter)
  } else {
    total_iterations
  }
  geoversa_meta_controller <- if (!is.null(config_df) && "geoversa_meta_controller" %in% names(config_df)) {
    as.character(first_or_na(config_df$geoversa_meta_controller))
  } else {
    NA_character_
  }

  meta_active_rate <- NA_real_
  dad_rk_active_rate <- NA_real_
  if (!is.null(diag_df) && nrow(diag_df) > 0L) {
    diag_target <- diag_df[diag_df$protocol == protocol_target & diag_df$model == model_target, , drop = FALSE]
    if (nrow(diag_target) > 0L && "meta_controller_active" %in% names(diag_target)) {
      meta_active_rate <- mean(toupper(as.character(diag_target$meta_controller_active)) %in% c("TRUE", "1"), na.rm = TRUE)
    }
    if (nrow(diag_target) > 0L && "dad_rk_active" %in% names(diag_target)) {
      dad_rk_active_rate <- mean(toupper(as.character(diag_target$dad_rk_active)) %in% c("TRUE", "1"), na.rm = TRUE)
    }
  }

  metrics_finite <- all(vapply(metric_cols, function(col) {
    col %in% names(target_row) && is.finite(numeric_or_na(target_row[[col]]))
  }, logical(1)))

  data.frame(
    check = c(
      "geoversa_summary_present",
      "geoversa_checkpoint_complete",
      "designbased_metrics_finite",
      "sample_size_500",
      "iteration_count_ge_30",
      "iteration_count_500_paper_level",
      "meta_controller_active"
    ),
    status = c(
      "PASS",
      if (isTRUE(run_complete)) "PASS" else if (isFALSE(run_complete)) "FAIL" else "WARN",
      if (isTRUE(metrics_finite)) "PASS" else "FAIL",
      if (is.finite(sample_size) && sample_size == 500) "PASS" else "WARN",
      if (is.finite(n_iter) && n_iter >= 30) "PASS" else "WARN",
      if (is.finite(n_iter) && n_iter >= 500) "PASS" else "WARN",
      if (is.finite(meta_active_rate) && meta_active_rate > 0) "PASS" else if (identical(geoversa_meta_controller, "TRUE")) "PASS" else "WARN"
    ),
    detail = c(
      summary_path,
      sprintf("completed=%s total=%s", completed_iteration, total_iterations),
      paste(names(metric_cols), collapse = ", "),
      sprintf("sample_size=%s", sample_size),
      sprintf("n_iter=%s", n_iter),
      sprintf("n_iter=%s", n_iter),
      sprintf("config=%s diagnostic_rate=%s dad_rk_rate=%s",
              geoversa_meta_controller, meta_active_rate, dad_rk_active_rate)
    ),
    stringsAsFactors = FALSE
  )
}

geoversa_summary <- read_csv_required(summary_path, "GeoVersa summary")
target_rows <- geoversa_summary[
  geoversa_summary$protocol == protocol_target &
    geoversa_summary$model == model_target,
  ,
  drop = FALSE
]
if (nrow(target_rows) == 0L) {
  stop(sprintf(
    "No GeoVersa row found for protocol=%s model=%s in %s",
    protocol_target, model_target, summary_path
  ), call. = FALSE)
}
target_row <- target_rows[1L, , drop = FALSE]
scenario_target <- as.character(first_or_na(target_row$scenario))

status_df <- safe_read_csv(status_path)
config_df <- safe_read_csv(config_path)
diag_df <- safe_read_csv(diag_path)
acceptance <- summarise_run_status(status_df, config_df, diag_df, target_row)

comparison_rows <- list()

# 1) Same-run RF comparison, when RF was included in WADOUX_MODELS.
same_run_rf <- geoversa_summary[
  geoversa_summary$protocol == protocol_target &
    grepl("^RF", geoversa_summary$model),
  ,
  drop = FALSE
]
if (nrow(same_run_rf) > 0L) {
  same_run_rf <- same_run_rf[order(same_run_rf$model), , drop = FALSE][1L, , drop = FALSE]
  for (metric in names(metric_cols)) {
    geoversa_value <- value_col(target_row, metric_cols[[metric]])
    reference_value <- value_col(same_run_rf, metric_cols[[metric]])
    comparison_rows <- append_comparison(
      comparison_rows,
      comparison_type = "absolute_metric_same_run_rf",
      scenario = scenario_target,
      protocol = protocol_target,
      model = model_target,
      reference_label = as.character(first_or_na(same_run_rf$model)),
      metric = metric,
      geoversa_value = geoversa_value,
      reference_value = reference_value,
      better_than_reference = better_absolute_metric(metric, geoversa_value, reference_value)
    )
  }
}

# 2) External RF reproduction reference comparison.
rf_reference_files <- list.files(
  rf_reference_dir,
  pattern = "_absolute_summary[.]csv$",
  full.names = TRUE
)
if (length(rf_reference_files) > 0L) {
  rf_reference_path <- rf_reference_files[[1L]]
  rf_reference <- read.csv(rf_reference_path, stringsAsFactors = FALSE, check.names = FALSE)
  rf_reference_target <- rf_reference[rf_reference$protocol == protocol_target, , drop = FALSE]
  if (nrow(rf_reference_target) > 0L) {
    for (metric in names(metric_cols)) {
      ref_row <- rf_reference_target[rf_reference_target$metric == metric, , drop = FALSE]
      if (nrow(ref_row) == 0L) {
        next
      }
      geoversa_value <- value_col(target_row, metric_cols[[metric]])
      reference_value <- numeric_or_na(ref_row$mean_value)
      comparison_rows <- append_comparison(
        comparison_rows,
        comparison_type = "absolute_metric_rf_reference",
        scenario = scenario_target,
        protocol = protocol_target,
        model = model_target,
        reference_label = basename(rf_reference_path),
        metric = metric,
        geoversa_value = geoversa_value,
        reference_value = reference_value,
        better_than_reference = better_absolute_metric(metric, geoversa_value, reference_value)
      )
    }
  }
}

# 3) Paper-style validation error: protocol estimate minus Population estimate.
population_rows <- geoversa_summary[
  geoversa_summary$protocol == "Population" &
    geoversa_summary$model == model_target,
  ,
  drop = FALSE
]
if (nrow(population_rows) > 0L) {
  population_row <- population_rows[1L, , drop = FALSE]
  geoversa_delta_rows <- list()
  for (metric in names(metric_cols)) {
    geoversa_value <- value_col(target_row, metric_cols[[metric]])
    population_value <- value_col(population_row, metric_cols[[metric]])
    delta_value <- geoversa_value - population_value
    geoversa_delta_rows[[length(geoversa_delta_rows) + 1L]] <- data.frame(
      scenario = scenario_target,
      protocol = protocol_target,
      model = model_target,
      metric = metric,
      protocol_value = geoversa_value,
      population_value = population_value,
      validation_error_vs_population = delta_value,
      stringsAsFactors = FALSE
    )
  }
  geoversa_delta_df <- do.call(rbind, geoversa_delta_rows)
  write.csv(
    geoversa_delta_df,
    file.path(out_dir, "geoversa_wadoux_population_delta.csv"),
    row.names = FALSE
  )

  same_run_rf_pop <- geoversa_summary[
    geoversa_summary$protocol == "Population" &
      grepl("^RF", geoversa_summary$model),
    ,
    drop = FALSE
  ]
  if (nrow(same_run_rf) > 0L && nrow(same_run_rf_pop) > 0L) {
    same_run_rf_pop <- same_run_rf_pop[
      same_run_rf_pop$model == as.character(first_or_na(same_run_rf$model)),
      ,
      drop = FALSE
    ]
    if (nrow(same_run_rf_pop) > 0L) {
      same_run_rf_pop <- same_run_rf_pop[1L, , drop = FALSE]
      for (metric in names(metric_cols)) {
        geo_delta <- (
          value_col(target_row, metric_cols[[metric]]) -
            value_col(population_row, metric_cols[[metric]])
        )
        rf_delta <- (
          value_col(same_run_rf, metric_cols[[metric]]) -
            value_col(same_run_rf_pop, metric_cols[[metric]])
        )
        comparison_rows <- append_comparison(
          comparison_rows,
          comparison_type = "paper_delta_same_run_rf",
          scenario = scenario_target,
          protocol = protocol_target,
          model = model_target,
          reference_label = paste0(as.character(first_or_na(same_run_rf$model)), "_delta_vs_population"),
          metric = metric,
          geoversa_value = geo_delta,
          reference_value = rf_delta,
          better_than_reference = closer_to_zero(geo_delta, rf_delta)
        )
      }
    }
  }

  if (file.exists(official_delta_summary)) {
    official_delta <- read.csv(official_delta_summary, stringsAsFactors = FALSE, check.names = FALSE)
    if (all(c("scenario", "metric", "protocol", "mean_value") %in% names(official_delta))) {
      official_target <- official_delta[
        official_delta$scenario == scenario_target &
          official_delta$protocol == protocol_target,
        ,
        drop = FALSE
      ]
      for (metric in names(metric_cols)) {
        off_row <- official_target[official_target$metric == metric, , drop = FALSE]
        geo_row <- geoversa_delta_df[geoversa_delta_df$metric == metric, , drop = FALSE]
        if (nrow(off_row) == 0L || nrow(geo_row) == 0L) {
          next
        }
        geoversa_value <- numeric_or_na(geo_row$validation_error_vs_population)
        reference_value <- numeric_or_na(off_row$mean_value)
        comparison_rows <- append_comparison(
          comparison_rows,
          comparison_type = "paper_delta_official_wadoux",
          scenario = scenario_target,
          protocol = protocol_target,
          model = model_target,
          reference_label = basename(official_delta_summary),
          metric = metric,
          geoversa_value = geoversa_value,
          reference_value = reference_value,
          better_than_reference = closer_to_zero(geoversa_value, reference_value)
        )
      }
    }
  }
}

comparison <- if (length(comparison_rows) == 0L) {
  make_empty_comparison()
} else {
  do.call(rbind, comparison_rows)
}

comparison_path <- file.path(out_dir, "geoversa_wadoux_reference_comparison.csv")
acceptance_path <- file.path(out_dir, "geoversa_wadoux_acceptance_checks.csv")
report_path <- file.path(out_dir, "geoversa_wadoux_validation_report.md")

write.csv(comparison, comparison_path, row.names = FALSE)
write.csv(acceptance, acceptance_path, row.names = FALSE)

metric_line <- function(metric) {
  col <- metric_cols[[metric]]
  sd_col <- metric_sd_cols[[metric]]
  value <- value_col(target_row, col)
  sd_value <- value_col(target_row, sd_col)
  if (is.finite(sd_value)) {
    sprintf("- %s: %.4f +/- %.4f", metric, value, sd_value)
  } else {
    sprintf("- %s: %.4f", metric, value)
  }
}

comparison_lines <- if (nrow(comparison) > 0L) {
  apply(comparison, 1L, function(row) {
    better <- as.character(row[["better_than_reference"]])
    sprintf(
      "- %s | %s | %s: GeoVersa %.4f vs ref %.4f | delta %.4f | better=%s",
      row[["comparison_type"]],
      row[["reference_label"]],
      row[["metric"]],
      as.numeric(row[["geoversa_value"]]),
      as.numeric(row[["reference_value"]]),
      as.numeric(row[["delta_geoversa_minus_reference"]]),
      better
    )
  })
} else {
  "- No reference comparison rows were produced. Include RF in the same run or provide a RF reference directory."
}

acceptance_lines <- apply(acceptance, 1L, function(row) {
  sprintf("- %s: %s | %s", row[["check"]], row[["status"]], row[["detail"]])
})

official_status <- if (file.exists(official_delta_summary)) {
  sprintf("available: %s", official_delta_summary)
} else {
  sprintf("not available: %s", official_delta_summary)
}

report <- c(
  "# GeoVersa Wadoux Validation Report",
  "",
  sprintf("- GeoVersa results: `%s`", geoversa_dir),
  sprintf("- RF reference: `%s`", rf_reference_dir),
  sprintf("- Official Wadoux delta summary: `%s`", official_status),
  sprintf("- Target protocol: `%s`", protocol_target),
  sprintf("- Target model: `%s`", model_target),
  sprintf("- Scenario: `%s`", scenario_target),
  "",
  "## Target Metrics",
  vapply(names(metric_cols), metric_line, character(1)),
  "",
  "## Acceptance Checks",
  acceptance_lines,
  "",
  "## Comparisons",
  comparison_lines,
  "",
  "## Interpretation",
  "- Functional check: the run must complete, produce finite Wadoux metrics, and log the meta-controller diagnostics.",
  "- Wadoux-style model check: on DesignBased, RMSE should be lower than RF and MEC/Pearson2 should be higher than RF under the same sample design.",
  "- Paper-level validation-error check: when Population is included, compare each protocol minus Population; smaller absolute validation error is better.",
  "- Exact paper audit requires the official `.Rdata` outputs imported through `code/import_wadoux_official_rdata.R`.",
  "",
  "## Outputs",
  sprintf("- `%s`", comparison_path),
  sprintf("- `%s`", acceptance_path)
)

writeLines(report, report_path)

cat("GeoVersa Wadoux validation comparison written.\n")
cat(sprintf("Comparison: %s\n", comparison_path))
cat(sprintf("Acceptance: %s\n", acceptance_path))
cat(sprintf("Report: %s\n", report_path))
