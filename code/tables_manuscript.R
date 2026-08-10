# =============================================================================
# tables_manuscript.R
#
# Generates every results table in the manuscript directly from the result CSVs,
# as LaTeX, so that no number in the paper can drift from the run that produced
# it. Writes Text/ecomod_submission/tables/*.tex, which manuscript.tex \input's.
#
# Every table the manuscript shows is generated here. Two of them used to be
# typed by hand into the .tex and had gone stale against the re-run campaigns,
# which is exactly the failure this file exists to prevent.
#
# Reporting follows the review: means are accompanied by standard deviations,
# Monte Carlo standard errors, medians, interquartile ranges and confidence
# intervals for the mean paired difference. Wilcoxon p values are still given
# but are no longer the headline, since with computational repetitions the
# effect size and its variability carry the information.
#
# Usage:  Rscript code/tables_manuscript.R
# =============================================================================

# Portable root resolution; see code/project_paths.R. Works whether the script
# is run from the repository root or from inside code/.
local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
setwd(gv_root())
out <- "Text/ecomod_submission/tables"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

DESIGNS <- c(random = "Simple random",
             regular_grid = "Systematic",
             clustered_random = "Clustered random")
dir_for <- function(sc) gv_results(sprintf("final_%s_confirm30", sc))

sm <- function(sc) read.csv(file.path(dir_for(sc), "dbtarf_summary_by_protocol.csv"))
pd <- function(sc) read.csv(file.path(dir_for(sc), "dbtarf_paired_deltas_vs_RF.csv"))
dg <- function(sc) read.csv(file.path(dir_for(sc), "dbtarf_diagnostics.csv"))
val <- function(s, m, col) { v <- s[[col]][s$model == m]; if (length(v)) v[1] else NA_real_ }

fmt <- function(x, d = 3) formatC(x, format = "f", digits = d)

# ---- Table: design-based accuracy, with dispersion --------------------------

lines <- c(
  "\\begin{tabular}{llrrrr}", "\\toprule",
  "Design & Model & ME & RMSE & SD(RMSE) & MEC \\\\", "\\midrule")
for (sc in names(DESIGNS)) {
  s <- sm(sc)
  models <- c("RF", "RF_tuned", "Cubist", "DBTARF")
  labels <- c("RF (default)", "RF-tuned", "Cubist", "TARF")
  for (j in seq_along(models)) {
    m <- models[j]
    lab <- DESIGNS[[sc]]
    bold <- m == "DBTARF"
    wrap <- function(z) if (bold) sprintf("\\textbf{%s}", z) else z
    lines <- c(lines, sprintf("%s & %s & %s & %s & %s & %s \\\\",
      lab, wrap(labels[j]),
      wrap(sprintf("%+.3f", val(s, m, "ME_mean"))),
      wrap(fmt(val(s, m, "RMSE_mean"))),
      wrap(fmt(val(s, m, "RMSE_sd"))),
      wrap(fmt(val(s, m, "MEC_mean")))))
  }
  lines <- c(lines, if (sc == tail(names(DESIGNS), 1)) "\\bottomrule" else "\\midrule")
}
lines <- c(lines, "\\end{tabular}")
writeLines(lines, file.path(out, "accuracy.tex"))

# ---- Table: paired differences, the uncertainty the review asked for --------

lines <- c(
  "\\begin{tabular}{lrrrrrr}", "\\toprule",
  paste("Design & Mean $\\Delta$ & SD & MCSE & 95\\% CI &",
        "Median [IQR] & Wins \\\\"), "\\midrule")
for (sc in names(DESIGNS)) {
  d <- pd(sc); dl <- d$RMSE_DBTARF - d$RMSE_RF
  tt <- stats::t.test(dl)
  lines <- c(lines, sprintf(
    "%s & $%+.3f$ & %.2f & %.3f & $[%.2f, %.2f]$ & $%+.2f$ $[%.2f, %.2f]$ & %d/%d \\\\",
    DESIGNS[[sc]], mean(dl), stats::sd(dl), stats::sd(dl) / sqrt(length(dl)),
    tt$conf.int[1], tt$conf.int[2], stats::median(dl),
    stats::quantile(dl, .25), stats::quantile(dl, .75),
    sum(dl < 0), sum(dl != 0)))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}")
writeLines(lines, file.path(out, "paired.tex"))

# ---- Table: which geostatistical choices the search makes -------------------

lines <- c("\\begin{tabular}{lrrrr}", "\\toprule",
  "Design & Binning or estimator & Range scale & Anisotropy & Recipe kept \\\\",
  "\\midrule")
for (sc in names(DESIGNS)) {
  p <- dg(sc)$selected_profile
  lines <- c(lines, sprintf("%s & %d & %d & %d & %d \\\\", DESIGNS[[sc]],
    sum(grepl("vgc|vgmath", p)), sum(grepl("_bw", p)),
    sum(grepl("aniso", p)), sum(!grepl("vgc|vgmath|_bw|aniso", p))))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}")
writeLines(lines, file.path(out, "axes.tex"))

# ---- Table: selected corrector across all repetitions -----------------------

modes <- unlist(lapply(names(DESIGNS), function(sc) dg(sc)$selected_mode))
tb <- sort(table(modes), decreasing = TRUE)
pretty <- c(resid_rf = "Residual random forest",
            resid_ensemble = "Ensemble of residual learners",
            resid_kriging = "Ordinary kriging of residuals",
            resid_spatial = "Kernel (radial basis) smoother",
            resid_cubist = "Residual Cubist",
            resid_mlp = "Single-hidden-layer network",
            cubist_delta = "Cubist delta",
            linear_cal = "Linear recalibration",
            quadratic_cal = "Quadratic recalibration",
            stack_rf_cubist = "RF and Cubist stack",
            bias = "Bias correction",
            RF_noharm = "No correction (base forest)")
lines <- c("\\begin{tabular}{lr}", "\\toprule",
           "Selected corrector & Count \\\\", "\\midrule")
for (nm in names(tb)) {
  lab <- if (!is.na(pretty[nm])) pretty[[nm]] else nm
  lines <- c(lines, sprintf("%s & %d \\\\", lab, as.integer(tb[[nm]])))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}")
writeLines(lines, file.path(out, "modes.tex"))

# ---- Table: ablation, if the fixed-recipe run is available ------------------

# ---- Table: ablation ---------------------------------------------------------
# Decomposes the gain under simple random calibration. Two contrasts, and they
# answer different questions: whether the variogram-derived representation helps
# at all (geo_basis FALSE), and whether tuning its parameters helps on top of
# that (fixed recipe versus selected). Only arms whose config records a commit
# are admitted, so an arm produced by superseded code cannot sit beside the rest.

has_prov <- function(d) {
  f <- file.path(d, "dbtarf_config.csv")
  file.exists(f) && "git_commit" %in% names(read.csv(f))
}
per_iter <- function(d, model = "DBTARF") {
  x <- read.csv(file.path(d, "dbtarf_all_results.csv"))
  x <- x[x$model == model, ]
  stats::setNames(x$RMSE, x$iteration)
}

main <- dir_for("random")
sr <- sm("random")
rf_mean <- val(sr, "RF", "RMSE_mean")
a_main <- per_iter(main)
rf_iter <- per_iter(main, "RF")

rows <- list(
  c("Default random forest", fmt(rf_mean), "n/a", "n/a"),
  c("Tuned base forest, no correction", fmt(val(sr, "RF_tuned", "RMSE_mean")), "n/a", "n/a"))

add_arm <- function(rows, label, dir) {
  if (!dir.exists(dir) || !has_prov(dir)) {
    message("[tables] arm omitted, no provenance: ", dir); return(rows)
  }
  b <- per_iter(dir)
  it <- intersect(names(a_main), names(b))
  d_rf <- b[it] - rf_iter[it]
  append(rows, list(c(label, fmt(mean(b[it])),
                      sprintf("%+.3f", mean(d_rf)),
                      sprintf("%d/%d", sum(d_rf < 0), length(it)))))
}
rows <- add_arm(rows, "Correction, no variogram-derived features",
                gv_results("abl_nogeo_random"))
rows <- add_arm(rows, "Correction, fixed variogram recipe",
                "results/abl_fixedrecipe_random")

d_sel <- a_main - rf_iter[names(a_main)]
rows <- append(rows, list(c("Correction, selected geostatistical hyperparameters",
                            fmt(mean(a_main)), sprintf("%+.3f", mean(d_sel)),
                            sprintf("%d/%d", sum(d_sel < 0), length(a_main)))))

lines <- c("\\begin{tabular}{lrrr}", "\\toprule",
           "Configuration & RMSE & $\\Delta$ vs RF & Wins \\\\", "\\midrule")
for (r in rows) lines <- c(lines, sprintf("%s & %s & %s & %s \\\\", r[1], r[2], r[3], r[4]))
lines <- c(lines, "\\bottomrule", "\\end{tabular}")
writeLines(lines, file.path(out, "ablation.tex"))
message("[tables] ablation table written")

# ---- Table: simulations -----------------------------------------------------

if (file.exists(gv_results("simulations", "S1_anisotropy.csv"))) {
  S1 <- read.csv(gv_results("simulations", "S1_anisotropy.csv"))
  S2 <- read.csv(gv_results("simulations", "S2_design.csv"))
  a1 <- aggregate(declared_aniso ~ true_ratio + snr, S1, mean)
  lines <- c("\\begin{tabular}{rrr}", "\\toprule",
             "True ratio $\\varrho$ & Signal-to-noise & Anisotropy declared \\\\",
             "\\midrule")
  a1 <- a1[order(a1$snr, -a1$true_ratio), ]
  for (i in seq_len(nrow(a1))) lines <- c(lines, sprintf("%.2f & %d & %.2f \\\\",
    a1$true_ratio[i], a1$snr[i], a1$declared_aniso[i]))
  lines <- c(lines, "\\bottomrule", "\\end{tabular}")
  writeLines(lines, file.path(out, "sim_s1.tex"))

  a2 <- aggregate(declared_aniso ~ design + true_ratio, S2, mean)
  lines <- c("\\begin{tabular}{lrr}", "\\toprule",
             "Sampling design & True ratio $\\varrho$ & Anisotropy declared \\\\",
             "\\midrule")
  for (i in seq_len(nrow(a2))) lines <- c(lines, sprintf("%s & %.2f & %.3f \\\\",
    ifelse(a2$design[i] == "random", "Simple random", "Clustered"),
    a2$true_ratio[i], a2$declared_aniso[i]))
  lines <- c(lines, "\\bottomrule", "\\end{tabular}")
  writeLines(lines, file.path(out, "sim_s2.tex"))
  message("[tables] simulation tables written")
}

cat(sprintf("\nwrote %s\n", out))
print(list.files(out))

# ---- Table: design-based interval coverage ----------------------------------
# This table was hand-typed into the manuscript and drifted: it still carried
# clustered widths of 111 and 114 from a campaign superseded by the CV+ fix,
# while the data give 94 and 98. Generating it closes that door.

uq <- function(sc) read.csv(file.path(dir_for(sc), "dbtarf_uncertainty_summary.csv"))
pick <- function(u, m, col) {
  v <- u[[col]][u$model == m]
  if (length(v)) v[1] else NA_real_
}

rows <- list(
  list("Simple random", "CV+",                       "random",           "DBTARF",       FALSE),
  list("Simple random",                   "Pooled out-of-fold quantile", "random",          "DBTARF_oofq",  FALSE),
  list("Simple random",                   "Quantile regression forest", "random",           "RF_quantile",  FALSE),
  list("MID",                             NA, NA, NA, NA),
  list("Systematic",                      "CV+",                        "regular_grid",     "DBTARF",       FALSE),
  list("MID",                             NA, NA, NA, NA),
  list("Clustered random", "CV+",                     "clustered_random", "DBTARF",       FALSE),
  list("Clustered random",                "Pooled out-of-fold quantile", "clustered_random","DBTARF_oofq",  FALSE),
  list("Clustered random",                "Covariate-shift weighted",   "clustered_random", "DBTARF_dbcal", FALSE),
  list("Clustered random",                "Spatial-block conformal",    "clustered_random", "DBTARF_spcal", TRUE)
)

lines <- c("\\begin{tabular}{llrr}", "\\toprule",
           "Design & Interval & PICP & MPIW \\\\", "\\midrule")
for (r in rows) {
  if (identical(r[[1]], "MID")) { lines <- c(lines, "\\midrule"); next }
  u <- uq(r[[3]])
  picp <- pick(u, r[[4]], "PICP90_mean"); mpiw <- pick(u, r[[4]], "MPIW90_mean")
  b <- isTRUE(r[[5]])
  wrap <- function(z) if (b) sprintf("\\textbf{%s}", z) else z
  lines <- c(lines, sprintf("%s & %s & %s & %s \\\\", r[[1]], wrap(r[[2]]),
                            wrap(sprintf("%.3f", picp)), wrap(sprintf("%.0f", mpiw))))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}")
writeLines(lines, file.path(out, "coverage.tex"))
message("[tables] coverage table written")

# ---- Table: comparison across estimands -------------------------------------
# The design-based row must agree with the accuracy and ablation tables, so it
# is read from the same rev1 campaign rather than from the protocols run.

pr <- read.csv(gv_results("canon_v030_protocols_confirm10", "dbtarf_protocols_summary.csv"))
prow <- function(tag) pr[pr$protocol == tag, ][1, ]
dr <- pd("random"); dld <- dr$RMSE_DBTARF - dr$RMSE_RF
sr <- sm("random")

fmt_row <- function(label, n, rf, db, d, w) sprintf(
  "%s & %d & %.2f & %.2f & %+.3f & %s \\\\", label, n, rf, db, d, w)

p_pop <- prow("Population"); p_rkf <- prow("RandomKFold"); p_skf <- prow("SpatialKFold")
lines <- c("\\begin{tabular}{lrrrrr}", "\\toprule",
  "Estimand & Reps & RF & TARF & $\\Delta$RMSE & Wins \\\\", "\\midrule",
  "\\multicolumn{6}{l}{\\emph{Design-based map accuracy, independent probability samples}} \\\\",
  fmt_row("Large independent SRS (20{,}000)", p_pop$n_iter, p_pop$RMSE_RF, p_pop$RMSE_DBTARF,
          p_pop$mean_delta_RMSE, sprintf("%d/%d", p_pop$wins_DBTARF, p_pop$n_iter)),
  fmt_row("Design-based (SRS 500)", length(dld), val(sr, "RF", "RMSE_mean"),
          val(sr, "DBTARF", "RMSE_mean"), mean(dld),
          sprintf("%d/%d", sum(dld < 0), length(dld))),
  "\\midrule",
  "\\multicolumn{6}{l}{\\emph{Model comparison only, not map accuracy}} \\\\",
  fmt_row("Random 10-fold CV", p_rkf$n_iter, p_rkf$RMSE_RF, p_rkf$RMSE_DBTARF,
          p_rkf$mean_delta_RMSE, sprintf("%d/%d", p_rkf$wins_DBTARF, p_rkf$n_iter)),
  fmt_row("Spatial block CV (350 km)", p_skf$n_iter, p_skf$RMSE_RF, p_skf$RMSE_DBTARF,
          p_skf$mean_delta_RMSE, sprintf("%d/%d", p_skf$wins_DBTARF, p_skf$n_iter)),
  "\\bottomrule", "\\end{tabular}")
writeLines(lines, file.path(out, "estimands.tex"))
message("[tables] estimands table written")
