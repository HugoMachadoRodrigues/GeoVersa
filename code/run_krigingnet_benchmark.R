# =============================================================================
# run_krigingnet_benchmark.R
# One-stop benchmark runner for KrigingNet on:
# - simulated data
# - Wadoux empirical framework
# Results are rounded to 2 decimals.
# =============================================================================

rm(list = ls())
set.seed(123)

source("code/KrigingNet_DualFramework.R")

round_numeric_df <- function(df, digits = 2) {
  num_cols <- vapply(df, is.numeric, logical(1))
  df[num_cols] <- lapply(df[num_cols], round, digits = digits)
  df
}

save_plot_png <- function(plot_obj, path, width = 10, height = 6, dpi = 300) {
  ggplot2::ggsave(
    filename = path,
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi
  )
}

save_benchmark_outputs <- function(bench, results_dir = "results") {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  if (!is.null(bench$simulated)) {
    write.csv(bench$simulated$results,
              file.path(results_dir, "simulated_results.csv"),
              row.names = FALSE)
    write.csv(bench$simulated$summary_protocol,
              file.path(results_dir, "simulated_summary_protocol.csv"),
              row.names = FALSE)
    write.csv(bench$simulated$summary_goal,
              file.path(results_dir, "simulated_summary_goal.csv"),
              row.names = FALSE)

    save_plot_png(
      plot_protocol_comparison(bench$simulated$results, metric = "RMSE"),
      file.path(results_dir, "simulated_protocol_rmse.png")
    )
    save_plot_png(
      plot_goal_comparison(bench$simulated$results, metric = "RMSE"),
      file.path(results_dir, "simulated_goal_rmse.png")
    )
  }

  if (!is.null(bench$wadoux)) {
    write.csv(bench$wadoux$results,
              file.path(results_dir, "wadoux_results.csv"),
              row.names = FALSE)
    write.csv(bench$wadoux$summary_protocol,
              file.path(results_dir, "wadoux_summary_protocol.csv"),
              row.names = FALSE)
    write.csv(bench$wadoux$summary_goal,
              file.path(results_dir, "wadoux_summary_goal.csv"),
              row.names = FALSE)
    write.csv(bench$wadoux$summary_buffer,
              file.path(results_dir, "wadoux_summary_buffered_loo.csv"),
              row.names = FALSE)

    save_plot_png(
      plot_protocol_comparison(bench$wadoux$results, metric = "RMSE"),
      file.path(results_dir, "wadoux_protocol_rmse.png")
    )
    save_plot_png(
      plot_goal_comparison(bench$wadoux$results, metric = "RMSE"),
      file.path(results_dir, "wadoux_goal_rmse.png")
    )
  }

  write.csv(bench$combined_results,
            file.path(results_dir, "combined_results.csv"),
            row.names = FALSE)
  write.csv(bench$combined_summary,
            file.path(results_dir, "combined_summary.csv"),
            row.names = FALSE)

  save_plot_png(
    plot_dual_framework_goals(
      sim_results = if (!is.null(bench$simulated)) bench$simulated$results else NULL,
      empirical_results = if (!is.null(bench$wadoux)) bench$wadoux$results else NULL,
      metric = "RMSE"
    ),
    file.path(results_dir, "combined_goal_rmse.png"),
    width = 12,
    height = 8
  )
}

run_simulated_benchmark <- function(sim = sim_context,
                                    xgb_params = xgb_params,
                                    krigingnet_params = krigingnet_params) {
  cat("\n========================================\n")
  cat("SIMULATED BENCHMARK\n")
  cat("========================================\n")

  res_random <- run_krigingnet_comparison(
    sim,
    protocol = "random_cv",
    xgb_params = xgb_params,
    krigingnet_params = krigingnet_params
  ) %>% mutate(dataset = "simulated")

  res_spatial <- run_krigingnet_comparison(
    sim,
    protocol = "spatial_block_cv",
    xgb_params = xgb_params,
    krigingnet_params = krigingnet_params
  ) %>% mutate(dataset = "simulated")

  res_design <- run_krigingnet_comparison(
    sim,
    protocol = "design_based_holdout",
    xgb_params = xgb_params,
    krigingnet_params = krigingnet_params
  ) %>% mutate(dataset = "simulated")

  results <- bind_rows(res_random, res_spatial, res_design)

  summary_protocol <- summarise_comparison(results) %>%
    round_numeric_df(2)

  summary_goal <- summarise_by_goal(results) %>%
    round_numeric_df(2)

  list(
    results = results,
    summary_protocol = summary_protocol,
    summary_goal = summary_goal
  )
}

run_wadoux_benchmark <- function(context = wadoux_context,
                                 sample_size = 500,
                                 max_buffer_splits = 50,
                                 xgb_params = xgb_params,
                                 krigingnet_params = krigingnet_params) {
  cat("\n========================================\n")
  cat("WADOUX EMPIRICAL BENCHMARK\n")
  cat("========================================\n")

  res_random <- run_empirical_protocol(
    context,
    sampling = "simple_random",
    protocol = "random_cv",
    sample_size = sample_size,
    xgb_params = xgb_params,
    krigingnet_params = krigingnet_params
  )

  res_spatial <- run_empirical_protocol(
    context,
    sampling = "simple_random",
    protocol = "spatial_kfold",
    sample_size = sample_size,
    xgb_params = xgb_params,
    krigingnet_params = krigingnet_params
  )

  res_design <- run_empirical_protocol(
    context,
    sampling = "simple_random",
    protocol = "design_based_validation",
    sample_size = sample_size,
    xgb_params = xgb_params,
    krigingnet_params = krigingnet_params
  )

  res_buffer <- run_empirical_protocol(
    context,
    sampling = "simple_random",
    protocol = "buffered_loo",
    sample_size = sample_size,
    max_splits = max_buffer_splits,
    xgb_params = xgb_params,
    krigingnet_params = krigingnet_params
  )

  results <- bind_rows(res_random, res_spatial, res_design, res_buffer)

  summary_protocol <- summarise_comparison(results) %>%
    round_numeric_df(2)

  summary_goal <- summarise_by_goal(results) %>%
    round_numeric_df(2)

  summary_buffer <- summarise_buffered_loo(results) %>%
    round_numeric_df(2)

  list(
    results = results,
    summary_protocol = summary_protocol,
    summary_goal = summary_goal,
    summary_buffer = summary_buffer
  )
}

run_full_krigingnet_benchmark <- function(run_simulated = TRUE,
                                          run_wadoux = TRUE,
                                          sample_size = 500,
                                          max_buffer_splits = 50,
                                          results_dir = "results",
                                          auto_save = TRUE,
                                          xgb_params = xgb_params,
                                          krigingnet_params = krigingnet_params) {
  out <- list()

  if (run_simulated) {
    out$simulated <- run_simulated_benchmark(
      sim = sim_context,
      xgb_params = xgb_params,
      krigingnet_params = krigingnet_params
    )
  }

  if (run_wadoux) {
    out$wadoux <- run_wadoux_benchmark(
      context = wadoux_context,
      sample_size = sample_size,
      max_buffer_splits = max_buffer_splits,
      xgb_params = xgb_params,
      krigingnet_params = krigingnet_params
    )
  }

  combined_results <- bind_rows(
    if (!is.null(out$simulated)) out$simulated$results else NULL,
    if (!is.null(out$wadoux)) out$wadoux$results else NULL
  )

  out$combined_summary <- summarise_dual_framework(
    sim_results = if (!is.null(out$simulated)) out$simulated$results else NULL,
    empirical_results = if (!is.null(out$wadoux)) out$wadoux$results else NULL
  ) %>%
    round_numeric_df(2)

  out$combined_results <- combined_results

  if (auto_save) {
    save_benchmark_outputs(out, results_dir = results_dir)
  }

  out
}

print_benchmark_report <- function(bench) {
  if (!is.null(bench$simulated)) {
    cat("\n\n===== SIMULATED | SUMMARY BY GOAL =====\n")
    print(bench$simulated$summary_goal)
  }

  if (!is.null(bench$wadoux)) {
    cat("\n\n===== WADOUX | SUMMARY BY GOAL =====\n")
    print(bench$wadoux$summary_goal)

    cat("\n\n===== WADOUX | BUFFERED LOO SUMMARY =====\n")
    print(bench$wadoux$summary_buffer)
  }

  cat("\n\n===== COMBINED | SUMMARY =====\n")
  print(bench$combined_summary)
}

# -----------------------------------------------------------------------------
# Recommended usage:
#
# source("code/run_krigingnet_benchmark.R")
#
# bench <- run_full_krigingnet_benchmark(
#   run_simulated = TRUE,
#   run_wadoux = TRUE,
#   sample_size = 500,
#   max_buffer_splits = 50,
#   results_dir = "results",
#   auto_save = TRUE,
#   xgb_params = xgb_params,
#   krigingnet_params = krigingnet_params
# )
#
# print_benchmark_report(bench)
# -----------------------------------------------------------------------------
