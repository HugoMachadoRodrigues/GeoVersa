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
        "Median [IQR] & W / L / T \\\\"), "\\midrule")
for (sc in names(DESIGNS)) {
  d <- pd(sc); dl <- d$RMSE_DBTARF - d$RMSE_RF
  tt <- stats::t.test(dl)
  lines <- c(lines, sprintf(
    "%s & $%+.3f$ & %.2f & %.3f & $[%.2f, %.2f]$ & $%+.2f$ $[%.2f, %.2f]$ & %d / %d / %d \\\\",
    DESIGNS[[sc]], mean(dl), stats::sd(dl), stats::sd(dl) / sqrt(length(dl)),
    tt$conf.int[1], tt$conf.int[2], stats::median(dl),
    stats::quantile(dl, .25), stats::quantile(dl, .75),
    # Wins, losses and ties, not wins over a denominator that silently drops
    # the ties. The systematic design has one exact tie, and printing it as
    # 29/29 invites the reader to think that design ran fewer repetitions.
    sum(dl < 0), sum(dl > 0), sum(dl == 0)))
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
                gv_results("abl_fixedrecipe_random"))

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

# Every interval is shown under every design. An earlier version of this table
# printed the quantile regression forest only under simple random calibration,
# where it over-covers at 0.937 and reads as the loose baseline the text calls
# it, and dropped it under clustered calibration, where it holds 0.928 with a
# narrower interval than the spatial-block repair. Whatever the reason that
# happened, the effect was to show a comparator only where it lost.
rows <- list(
  list("Simple random",    "CV+",                      "random",           "DBTARF",       FALSE),
  list("Simple random",    "Pooled OOF quantile",      "random",           "DBTARF_oofq",  FALSE),
  list("Simple random",    "Spatial-block",            "random",           "DBTARF_spcal", FALSE),
  list("Simple random",    "Quantile RF",              "random",           "RF_quantile",  FALSE),
  list("MID", NA, NA, NA, NA),
  list("Systematic",       "CV+",                      "regular_grid",     "DBTARF",       FALSE),
  list("Systematic",       "Spatial-block",            "regular_grid",     "DBTARF_spcal", FALSE),
  list("Systematic",       "Quantile RF",              "regular_grid",     "RF_quantile",  FALSE),
  list("MID", NA, NA, NA, NA),
  list("Clustered random", "CV+",                      "clustered_random", "DBTARF",       FALSE),
  list("Clustered random", "Pooled OOF quantile",      "clustered_random", "DBTARF_oofq",  FALSE),
  list("Clustered random", "Covariate-shift weighted", "clustered_random", "DBTARF_dbcal", FALSE),
  list("Clustered random", "Spatial-block",            "clustered_random", "DBTARF_spcal", TRUE),
  list("Clustered random", "Quantile RF",              "clustered_random", "RF_quantile",  FALSE)
)

# Review asked for dispersion, not just means: a mean of 0.861 could hide wide
# heterogeneity, and the reader cannot tell from a point estimate whether the
# shortfall is systematic or an artefact of a few repetitions.
picp_stats <- function(sc, model) {
  x <- read.csv(file.path(dir_for(sc), "dbtarf_uncertainty.csv"))
  v <- x$PICP90[x$model == model]
  n <- length(v); mu <- mean(v); sd <- stats::sd(v); se <- sd / sqrt(n)
  list(mu = mu, sd = sd, lo = mu - 1.96 * se, hi = mu + 1.96 * se,
       below = sum(v < 0.90), n = n)
}

lines <- c("\\begin{tabular}{llrrrrr}", "\\toprule",
           paste("Design & Interval & PICP & SD & 95\\% CI & MPIW &",
                 "Below 0.90 \\\\"), "\\midrule")
for (r in rows) {
  if (identical(r[[1]], "MID")) { lines <- c(lines, "\\midrule"); next }
  u <- uq(r[[3]]); st <- picp_stats(r[[3]], r[[4]])
  mpiw <- pick(u, r[[4]], "MPIW90_mean")
  b <- isTRUE(r[[5]])
  wrap <- function(z) if (b) sprintf("\\textbf{%s}", z) else z
  lines <- c(lines, sprintf("%s & %s & %s & %s & %s & %s & %s \\\\",
    r[[1]], wrap(r[[2]]),
    # leading zeros dropped: these are bounded by 1 and the column is tight
    wrap(sub("^0", "", sprintf("%.3f", st$mu))),
    wrap(sub("^0", "", sprintf("%.3f", st$sd))),
    wrap(sprintf("[%s,%s]", sub("^0", "", sprintf("%.3f", st$lo)),
                            sub("^0", "", sprintf("%.3f", st$hi)))),
    wrap(sprintf("%.0f", mpiw)),
    wrap(sprintf("%d/%d", st$below, st$n))))
}
lines <- c(lines, "\\bottomrule", "\\end{tabular}")
writeLines(lines, file.path(out, "coverage.tex"))
message("[tables] coverage table written")

# ---- Table: sample-size sensitivity -----------------------------------------
# Does the clustered shortfall shrink when the calibration sample grows? Three
# arms, because "a bigger sample" is ambiguous under a clustered design:
#
#   Simple random          the control; the design whose coverage was never in
#                          question
#   Clustered, denser      20 primary units at every size, more points inside
#                          each. Identical spatial footprint, so this isolates
#                          sample size and tests the mechanism the paper claims
#   Clustered, wider       25 points per unit at every size, more units. The
#                          footprint grows with n, so any repair is partly a
#                          coverage effect and not a sample-size effect
#
# The n = 500 midpoint is 20 x 25 and belongs to both clustered arms.
#
# Sizes are included only when a campaign holds the full 30 repetitions, records
# a git commit, and trained on the size the row claims. Campaigns predating the
# DBT_VAL_SIZE split are still comparable: those commits left R/dbtarf.R and the
# estimator path in the runner byte-identical, and every campaign here scores a
# validation draw of 500.

ARMS <- list(
  list(key = "random", label = "Simple random", sc = "random",
       dirs = c(`200` = "rand_n200", `500` = NA, `800` = "rand_n800")),
  list(key = "dens", label = "Clustered, dense", sc = "clustered_random",
       dirs = c(`200` = "clus_dens_n200", `500` = "clus_mid_n500", `800` = "clus_dens_n800")),
  list(key = "spread", label = "Clustered, wide", sc = "clustered_random",
       dirs = c(`200` = "clus_spread_n200", `500` = "clus_mid_n500", `800` = "clus_spread_n800"))
)

# The clustered midpoint is clus_mid_n500 rather than the older campaign it
# reproduces. Both are 20 x 25, but only the newer one says so: the older
# predates the commit that began recording n_psu and m_per_psu, so its geometry
# was an inference from the sampler defaults. The re-run settles it, and settles
# it strongly, reproducing all 30 repetitions bit for bit on PICP, MPIW and
# RMSE. A different geometry could not have done that.
#
# It is one campaign appearing in two arms, which is what a shared midpoint
# means. The arms do not cross there; they meet at a node they have in common,
# and the table marks it.
size_dir <- function(arm, n) {
  d <- arm$dirs[[as.character(n)]]
  if (is.na(d)) dir_for(arm$sc) else gv_results(d)
}

# A campaign is admitted only if it has provenance, the full 30 repetitions, and
# a calibration sample that is actually the size the row claims.
#
# That last check exists because it caught a real error. DBT_SAMPLE_SIZE governs
# the calibration draw under the random and systematic designs, but the
# clustered design ignores it and takes its size from DBT_N_PSU times
# DBT_M_PER_PSU. Requesting n = 200 clustered therefore shrank only the
# validation sample and left calibration at 500, and the resulting row was not a
# sample-size comparison at all: it was the same experiment measured twice. The
# mean interval width gave it away, 98.9 against 97.7 where the random design
# moved 114.7 against 103.3, but nothing in the pipeline objected. Now n_train
# is read back from the diagnostics and has to agree.
size_ready <- function(arm, n, want = 30L) {
  d <- size_dir(arm, n)
  cfg <- file.path(d, "dbtarf_config.csv")
  unc <- file.path(d, "dbtarf_uncertainty.csv")
  dia <- file.path(d, "dbtarf_diagnostics.csv")
  if (!file.exists(cfg) || !file.exists(unc) || !file.exists(dia)) return(FALSE)
  k <- read.csv(cfg)
  if (!"git_commit" %in% names(k) || !nzchar(as.character(k$git_commit[1]))) return(FALSE)
  x <- read.csv(unc)
  if (sum(x$model == "DBTARF") < want) return(FALSE)

  # The calibration sample must be the size the row claims. This check is here
  # because a campaign once ran to completion at the wrong size without a single
  # complaint: DBT_SAMPLE_SIZE governs the calibration draw under the random and
  # systematic designs, but the clustered design takes its size from DBT_N_PSU
  # times DBT_M_PER_PSU, so a sweep over the former left clustered calibration
  # at 500 and shrank only the validation draw. The resulting row was one
  # experiment compared with itself. Mean interval width was the tell, 98.9
  # against 97.7 where the random design moved 114.7 against 103.3, and nothing
  # in the pipeline objected. Now n_train has to agree.
  g <- read.csv(dia)
  nt <- unique(g$n_train)
  if (length(nt) != 1L || nt != n) {
    warning(sprintf(
      "sample-size row dropped: %s asked for n = %d but trained on %s",
      basename(d), n, paste(nt, collapse = "/")), call. = FALSE)
    return(FALSE)
  }

  # Coverage must also be measured at one precision across the table, or the
  # standard deviations are not comparable between rows. Campaigns written
  # before DBT_VAL_SIZE existed have no such column and validated at
  # sample_size, which for those campaigns was 500.
  vs <- if ("val_size" %in% names(k)) k$val_size[1] else k$sample_size[1]
  if (!identical(as.integer(vs), 500L)) {
    warning(sprintf("sample-size row dropped: %s validated on %s, not 500",
                    basename(d), vs), call. = FALSE)
    return(FALSE)
  }
  TRUE
}

SIZES <- c(200L, 500L, 800L)
kmc <- 100L   # kmeans clusters the primary units are drawn from, DBT_KMEANS_CENTERS
sens <- list()
for (arm in ARMS) {
  for (n in SIZES) {
    if (!size_ready(arm, n)) next
    d <- size_dir(arm, n)
    x <- read.csv(file.path(d, "dbtarf_uncertainty.csv"))
    a <- read.csv(file.path(d, "dbtarf_all_results.csv"))
    v <- x$PICP90[x$model == "DBTARF"]
    se <- stats::sd(v) / sqrt(length(v))
    blk <- x$PICP90[x$model == "DBTARF_spcal"]
    g <- read.csv(file.path(d, "dbtarf_diagnostics.csv"))
    k <- read.csv(file.path(d, "dbtarf_config.csv"))
    # Geometry, read from the provenance rather than from the directory name.
    # Primary units are drawn with replacement, so the nominal count overstates
    # the number of distinct clusters; the realised expectation is reported
    # alongside it because the argument is about spatial spread and the nominal
    # figure is not the spread the sample actually has.
    geom <- if (identical(arm$sc, "clustered_random") && "n_psu" %in% names(k)) {
      npsu <- k$n_psu[1]; mper <- k$m_per_psu[1]
      distinct <- kmc * (1 - (1 - 1 / kmc)^npsu)
      sprintf("%d/%d (%.1f)", npsu, mper, distinct)
    } else "SRS"
    # What the interval was calibrated on versus what it is applied to. The
    # conformal quantile summarises the out-of-fold error of the calibration
    # sample faithfully in every cell; the design changes the error over the
    # domain, and their ratio is what coverage tracks.
    sens[[length(sens) + 1L]] <- list(
      design = arm$label, arm = arm$key, n = n, geom = geom,
      rmse = mean(a$RMSE[a$model == "DBTARF"]),
      mec  = mean(a$MEC[a$model == "DBTARF"]),
      gap = mean(a$RMSE[a$model == "DBTARF"]) / mean(g$selected_oof_rmse),
      mu = mean(v), sd = stats::sd(v),
      lo = mean(v) - 1.96 * se, hi = mean(v) + 1.96 * se,
      below = sum(v < 0.90), reps = length(v),
      mpiw = mean(x$MPIW90[x$model == "DBTARF"]),
      blk = if (length(blk)) mean(blk) else NA_real_)
  }
}

# The table exists to say something about clustering. A run of the control arm
# alone would fill it with rows nobody asked about, so it is written only once a
# clustered arm has at least two admissible sizes to compare.
per_arm <- table(vapply(sens, function(r) r$arm, character(1)))
clustered_ok <- any(per_arm[intersect(names(per_arm), c("dens", "spread"))] >= 2L)

if (clustered_ok) {
  z <- function(x, d = 3) sub("^0", "", sprintf(paste0("%.", d, "f"), x))
  lines <- c("\\begin{tabular}{llrrrrrrrr}", "\\toprule",
             paste("Design & Units/pts & $n$ & RMSE & Gap & PICP & SD &",
                   "95\\% CI & Below & Block \\\\"), "\\midrule")
  prev <- NULL
  for (r in sens) {
    if (!is.null(prev) && !identical(prev, r$design)) lines <- c(lines, "\\midrule")
    prev <- r$design
    # The design label is repeated rather than left blank on continuation rows:
    # a blank leading cell survives LaTeX but reads as missing data once a
    # typesetter rebuilds the table from the Word file. The midpoint row is
    # marked, since it is one campaign shown in two arms rather than two runs.
    mark <- if (identical(r$n, 500L) && r$arm %in% c("dens", "spread")) "\\rlap{$^{a}$}" else ""
    lines <- c(lines, sprintf("%s & %s%s & %d & %.2f & %s & %s & %s & [%s,%s] & %d/%d & %s \\\\",
      r$design, r$geom, mark, r$n, r$rmse, sprintf("%.2f", r$gap),
      z(r$mu), z(r$sd), z(r$lo), z(r$hi),
      r$below, r$reps, if (is.na(r$blk)) "--" else z(r$blk)))
  }
  lines <- c(lines, "\\bottomrule", "\\end{tabular}")
  writeLines(lines, file.path(out, "sizes.tex"))
  message(sprintf("[tables] sample-size table written, %d rows across %s",
                  length(sens),
                  paste(sprintf("%s(%s)", names(per_arm), per_arm), collapse = " ")))
} else {
  # Deleting rather than merely skipping. A previous version of this table was
  # written from campaigns that a later check disqualified, and skipping left
  # the disqualified file on disk where the manuscript went on including it.
  # A table that cannot be regenerated must not survive the attempt.
  stale <- file.path(out, "sizes.tex")
  if (file.exists(stale)) {
    file.remove(stale)
    message("[tables] stale sizes.tex removed; the manuscript must not \\input it")
  }
  message("[tables] sample-size table skipped: fewer than two admissible sizes per design")
}

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
