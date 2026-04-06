rm(list = ls())

project_root <- normalizePath(getwd(), mustWork = TRUE)
external_root <- file.path(project_root, "external", "SpatialValidation")
external_root_rel <- file.path("external", "SpatialValidation")
official_code_dir <- file.path(external_root, "code")
docs_root <- file.path(project_root, "docs", "wadoux2021-reference")
dir.create(docs_root, recursive = TRUE, showWarnings = FALSE)

safe_git_stdout <- function(args) {
  out <- tryCatch(
    system2("git", args, stdout = TRUE, stderr = FALSE),
    error = function(e) character()
  )
  out <- trimws(out)
  out[nzchar(out)]
}

normalize_protocol_wadoux <- function(x) {
  x[x == "RandonKFold"] <- "RandomKFold"
  x
}

stack_official_metric <- function(df, scenario, metric, source_file, upstream_commit) {
  protocol_cols <- setdiff(names(df), "Stat")
  rows <- lapply(protocol_cols, function(protocol) {
    data.frame(
      scenario = scenario,
      iteration = seq_len(nrow(df)),
      metric = metric,
      protocol = normalize_protocol_wadoux(protocol),
      delta_value = df[[protocol]],
      source_file = basename(source_file),
      upstream_commit = upstream_commit,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

import_official_rdata <- function(file, scenario, upstream_commit) {
  env <- new.env(parent = emptyenv())
  load(file, envir = env)

  metric_map <- list(
    ME = "res.ME",
    RMSE = "res.RMSE",
    r2 = "res.r2",
    MEC = "res.MEC"
  )

  rows <- lapply(names(metric_map), function(metric_name) {
    object_name <- metric_map[[metric_name]]
    if (!exists(object_name, envir = env, inherits = FALSE)) {
      stop(sprintf("Object '%s' not found in %s", object_name, file))
    }
    stack_official_metric(
      df = get(object_name, envir = env, inherits = FALSE),
      scenario = scenario,
      metric = metric_name,
      source_file = file,
      upstream_commit = upstream_commit
    )
  })

  do.call(rbind, rows)
}

upstream_commit <- safe_git_stdout(c("-C", external_root_rel, "rev-parse", "HEAD"))
upstream_commit <- if (length(upstream_commit) > 0) upstream_commit[1] else NA_character_

expected <- data.frame(
  scenario = c("random", "regular_grid", "clustered_random"),
  expected_rdata = c("res_random_500.Rdata", "res_regular_500.Rdata", "res_clustered_random_500.Rdata"),
  stringsAsFactors = FALSE
)
expected$full_path <- file.path(official_code_dir, expected$expected_rdata)
expected$exists <- file.exists(expected$full_path)
expected$upstream_repo <- "https://github.com/AlexandreWadoux/SpatialValidation"
expected$upstream_commit <- upstream_commit

write.csv(
  expected,
  file.path(docs_root, "official_rdata_manifest.csv"),
  row.names = FALSE
)

if (!any(expected$exists)) {
  cat("No official Wadoux .Rdata outputs were found.\n")
  cat(sprintf("Manifest written to: %s\n", file.path(docs_root, "official_rdata_manifest.csv")))
  quit(save = "no", status = 0)
}

delta_rows <- lapply(seq_len(nrow(expected)), function(i) {
  if (!expected$exists[i]) {
    return(NULL)
  }
  import_official_rdata(
    file = expected$full_path[i],
    scenario = expected$scenario[i],
    upstream_commit = expected$upstream_commit[i]
  )
})
delta_rows <- delta_rows[!vapply(delta_rows, is.null, logical(1))]
delta_long <- do.call(rbind, delta_rows)

split_key <- interaction(delta_long$scenario, delta_long$metric, delta_long$protocol, drop = TRUE)
delta_summary_list <- lapply(split(delta_long, split_key), function(df) {
  data.frame(
    scenario = df$scenario[1],
    metric = df$metric[1],
    protocol = df$protocol[1],
    n_iter = nrow(df),
    mean_delta = mean(df$delta_value, na.rm = TRUE),
    sd_delta = sd(df$delta_value, na.rm = TRUE),
    median_delta = median(df$delta_value, na.rm = TRUE),
    q05_delta = unname(stats::quantile(df$delta_value, probs = 0.05, na.rm = TRUE)),
    q95_delta = unname(stats::quantile(df$delta_value, probs = 0.95, na.rm = TRUE)),
    upstream_commit = df$upstream_commit[1],
    stringsAsFactors = FALSE
  )
})
delta_summary <- do.call(rbind, delta_summary_list)
delta_summary <- delta_summary[order(delta_summary$scenario, delta_summary$metric, delta_summary$protocol), ]

write.csv(
  delta_long,
  file.path(docs_root, "official_delta_long.csv"),
  row.names = FALSE
)
write.csv(
  delta_summary,
  file.path(docs_root, "official_delta_summary.csv"),
  row.names = FALSE
)

cat("Imported official Wadoux .Rdata outputs.\n")
cat(sprintf("Manifest: %s\n", file.path(docs_root, "official_rdata_manifest.csv")))
cat(sprintf("Long table: %s\n", file.path(docs_root, "official_delta_long.csv")))
cat(sprintf("Summary: %s\n", file.path(docs_root, "official_delta_summary.csv")))
