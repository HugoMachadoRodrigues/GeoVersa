# =============================================================================
# figures_distance.R
#
# Where the interval fails, as a function of distance from the calibration
# sample. The manuscript argues that clustered calibration breaks conformal
# coverage through spatial extrapolation; every other figure shows the failure
# aggregated over the map, and this one shows it against the variable the
# mechanism names.
#
# Three panels:
#   a  exposure. What fraction of the domain lies farther than delta from any
#      calibration point, per design. This is why the designs differ at all,
#      and it is estimated design-unbiasedly because validation is an SRS from
#      the same frame in every repetition.
#   b  the two ingredients. Conditional RMSE and the conditional mean half-width
#      the interval actually issued. The error responds to distance; the
#      half-width does not.
#   c  coverage. The cumulative estimator P(covered | distance >= delta), which
#      has no bin, bandwidth or basis, with equal-count decile points over it.
#
# What this figure must not be read as saying, and what the caption therefore
# avoids:
#
#   * Not "coverage falls because the error exceeds the half-width". Conditional
#     on distance, that statement and "coverage is below nominal" are the same
#     event, so it explains nothing. An earlier draft of the manuscript made
#     exactly that mistake in prose and it was retracted. Panel b is here
#     because one curve responds to distance and the other does not, which is a
#     fact about the interval rather than a restatement of coverage.
#   * Nothing is plotted as |obs - pred| over half-width, and no crossing
#     distance is annotated. The CV+ interval is not centred on pred: its bounds
#     come from leave-fold-out predictions while pred is the full-data refit,
#     and 31 per cent of points sit more than a tenth of a half-width off centre.
#     A ratio of the two would be comparing quantities that do not share a
#     centre.
#   * Not a causal claim. Within one design, distance from the calibration
#     sample and geographic location are the same variable. Distance indexes the
#     failure. The causal argument is the stationary-field simulation, where the
#     field has no structure for location to stand in for.
#   * Not a covariate-space claim. The dump carries coordinates and no
#     covariates, so this is geometric distance and says nothing directly about
#     support in feature space.
#
# Points within a repetition share one fitted model and one conformal
# calibration, so they are not independent. Every interval here resamples whole
# repetitions, never points.
#
# Usage:  Rscript code/figures_distance.R
# =============================================================================

local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
setwd(gv_root())
source("code/figure_theme.R")

out_png <- "Text/ecomod_submission/figures/fig8_distance.png"
out_csv <- gv_results("fig8_curves.csv")
B <- 1000L                      # bootstrap replicates, resampling repetitions
set.seed(20260814L)

DESIGNS <- list(
  list(dir = "pts_random",    lab = "Simple random",        col = COL_SRS),
  list(dir = "pts_clus_mid",  lab = "Clustered 20 \u00d7 25", col = COL_CLUS),
  list(dir = "pts_clus_wide", lab = "Clustered 8 \u00d7 25",  col = COL_CLUSW)
)

# ---- load, with the assertions the dump cannot make for itself --------------
# design comes from the directory, never from the scenario column: both
# clustered campaigns record scenario == "clustered_random" and would collapse
# into one.
load_one <- function(d) {
  f <- gv_results(d$dir, "dbtarf_points.csv")
  if (!file.exists(f)) return(NULL)
  x <- read.csv(f)
  reps <- unique(x$iteration)
  if (length(reps) < 10L) {
    message(sprintf("[fig8] %s has %d repetitions, skipping", d$dir, length(reps)))
    return(NULL)
  }
  stopifnot(all(is.finite(x$dist_nn)), all(is.finite(x$obs)),
            all(is.finite(x$pred)), all(is.finite(x$lo)), all(is.finite(x$hi)),
            all(x$hi > x$lo))
  # metres, not kilometres: a unit slip here would silently rescale the whole
  # figure, and the domain is a 928 by 1642 km rectangle
  stopifnot(max(x$dist_nn) > 1e4, max(x$dist_nn) < 2e6)
  n_per <- table(x$iteration)
  stopifnot(length(unique(n_per)) == 1L)
  data.frame(design = d$lab,
             rep = as.integer(factor(x$iteration)),
             d_km = pmax(x$dist_nn, 500) / 1000,   # half a 1 km cell
             cover = as.integer(x$obs >= x$lo & x$obs <= x$hi),
             err = x$obs - x$pred,
             w = (x$hi - x$lo) / 2,
             stringsAsFactors = FALSE)
}

dat <- do.call(rbind, Filter(Negate(is.null), lapply(DESIGNS, load_one)))
if (is.null(dat) || !length(unique(dat$design))) {
  stop("no point dumps with 10 repetitions found; run code/run_point_dump.sh")
}
present <- vapply(DESIGNS, function(d) d$lab %in% dat$design, logical(1))
DES <- DESIGNS[present]
LEV <- vapply(DES, function(d) d$lab, character(1))
PAL <- stats::setNames(vapply(DES, function(d) d$col, character(1)), LEV)
dat$design <- factor(dat$design, levels = LEV)
message(sprintf("[fig8] %d designs, %d points, %d repetitions each",
                length(LEV), nrow(dat), length(unique(dat$rep))))

# ---- the estimators, each a function of a data frame so the bootstrap can
# ---- call them on resampled repetitions without any code duplicated ---------

GRID <- exp(seq(log(2), log(400), length.out = 60))   # km, log-spaced

# Cumulative coverage beyond delta. No bin, no bandwidth, no basis. This is the
# estimator the text quotes; the decile points and the panel-b curves exist to
# make the shape legible.
cov_beyond <- function(z, grid = GRID) {
  vapply(grid, function(g) {
    s <- z$cover[z$d_km >= g]
    if (length(s) < 100L) NA_real_ else mean(s)
  }, numeric(1))
}
frac_beyond <- function(z, grid = GRID) {
  vapply(grid, function(g) mean(z$d_km >= g), numeric(1))
}

# Local behaviour in bands shared by every design. Equal-count bins within a
# design would put each design's points at different distances and the curves
# could not be read against each other, which is the comparison that matters:
# whether two designs that differ in their marginal coverage still cover the
# same at the same distance.
BANDS <- c(0, 10, 25, 50, 100, 200, 400, Inf)
band_stats <- function(z) {
  b <- cut(z$d_km, BANDS, labels = FALSE, right = FALSE)
  do.call(rbind, lapply(seq_len(length(BANDS) - 1L), function(i) {
    s <- z[which(b == i), , drop = FALSE]
    if (nrow(s) < 60L) return(NULL)
    data.frame(band = i, d_km = stats::median(s$d_km),
               rmse = sqrt(mean(s$err^2)),
               w = mean(s$w),
               cover = mean(s$cover))
  }))
}
# Vector of band coverages on a fixed length, so the bootstrap can stack them
band_cover <- function(z) {
  b <- cut(z$d_km, BANDS, labels = FALSE, right = FALSE)
  vapply(seq_len(length(BANDS) - 1L), function(i) {
    s <- z$cover[which(b == i)]
    if (length(s) < 60L) NA_real_ else mean(s)
  }, numeric(1))
}

# Bootstrap over repetitions. A repetition drawn twice contributes its rows
# twice, which is what makes the interval reflect between-repetition variation
# rather than between-point variation.
boot_stat <- function(z, fn, B. = B) {
  reps <- unique(z$rep)
  R <- length(reps)
  idx <- split(seq_len(nrow(z)), z$rep)
  out <- replicate(B., {
    take <- sample(reps, R, replace = TRUE)
    fn(z[unlist(idx[as.character(take)], use.names = FALSE), , drop = FALSE])
  })
  if (is.null(dim(out))) out <- matrix(out, nrow = 1L)
  data.frame(lo = apply(out, 1L, stats::quantile, 0.025, na.rm = TRUE),
             hi = apply(out, 1L, stats::quantile, 0.975, na.rm = TRUE))
}

# ---- panel data -------------------------------------------------------------
curves <- do.call(rbind, lapply(LEV, function(L) {
  z <- dat[dat$design == L, , drop = FALSE]
  ci_c <- boot_stat(z, cov_beyond)
  ci_f <- boot_stat(z, frac_beyond)
  data.frame(design = L, d_km = GRID,
             cov = cov_beyond(z), cov_lo = ci_c$lo, cov_hi = ci_c$hi,
             frac = frac_beyond(z), frac_lo = ci_f$lo, frac_hi = ci_f$hi,
             stringsAsFactors = FALSE)
}))
curves$design <- factor(curves$design, levels = LEV)

dec <- do.call(rbind, lapply(LEV, function(L) {
  z <- dat[dat$design == L, , drop = FALSE]
  d <- band_stats(z)
  ci <- boot_stat(z, band_cover)
  d <- cbind(d, cov_lo = ci$lo[d$band], cov_hi = ci$hi[d$band])
  d$design <- L; d
}))
dec$design <- factor(dec$design, levels = LEV)

# ---- the headline test, and it is not the figure ----------------------------
# One logistic slope per repetition, then a t test on the R slopes. The
# within-repetition standard errors are deliberately discarded: only the spread
# between repetitions is used, so nothing here assumes points are independent.
slopes <- do.call(rbind, lapply(LEV, function(L) {
  z <- dat[dat$design == L, , drop = FALSE]
  b <- vapply(unique(z$rep), function(r) {
    s <- z[z$rep == r, , drop = FALSE]
    if (length(unique(s$cover)) < 2L) return(NA_real_)
    unname(stats::coef(stats::glm(I(1 - cover) ~ log2(d_km), family = "binomial",
                                  data = s))[2])
  }, numeric(1))
  b <- b[is.finite(b)]
  tt <- stats::t.test(b)
  data.frame(design = L, R = length(b), beta = mean(b),
             lo = tt$conf.int[1], hi = tt$conf.int[2], p = tt$p.value,
             stringsAsFactors = FALSE)
}))
message("[fig8] log-odds of a miss per doubling of distance, t test over repetitions:")
for (i in seq_len(nrow(slopes))) {
  message(sprintf("  %-24s beta %+0.3f  95%% CI [%+0.3f, %+0.3f]  p %.4f  (R = %d)",
                  slopes$design[i], slopes$beta[i], slopes$lo[i], slopes$hi[i],
                  slopes$p[i], slopes$R[i]))
}

# ---- plot -------------------------------------------------------------------
XLIM <- range(GRID)
xsc <- function(title = NULL) scale_x_log10(
  breaks = c(2, 5, 10, 25, 50, 100, 200, 400),
  labels = c("2", "5", "10", "25", "50", "100", "200", "400"),
  limits = XLIM, name = title)
med <- vapply(LEV, function(L) stats::median(dat$d_km[dat$design == L]), 0)
far <- vapply(LEV, function(L) mean(dat$d_km[dat$design == L] > 100), 0)

# a: exposure, directly labelled at each design's median distance
lab_a <- data.frame(design = factor(LEV, levels = LEV), d_km = med,
                    frac = 0.5,
                    txt = sprintf("%.0f km", med),
                    hj = c(1.25, 0.5, 0.5)[seq_along(LEV)],
                    vj = c(0.5, 2.0, -1.0)[seq_along(LEV)])
p_a <- ggplot(curves, aes(d_km, frac, colour = design, fill = design)) +
  geom_ribbon(aes(ymin = frac_lo, ymax = frac_hi), alpha = 0.18, colour = NA) +
  geom_line(linewidth = 0.9) +
  geom_point(data = lab_a, size = 2.2, shape = 21, fill = "white", stroke = 1) +

  scale_colour_manual(values = PAL) + scale_fill_manual(values = PAL) +
  xsc("Distance (km)") +
  scale_y_continuous(limits = c(0, 1), labels = function(b) sprintf("%.0f%%", 100 * b)) +
  labs(title = "a  Distance to the sample",
       subtitle = sprintf("Share of the domain farther than a given\ndistance; circles: medians, %s km",
                          paste(sprintf("%.0f", med), collapse = ", ")),
       y = "Share of domain") +
  theme_paper() + guides(colour = "none", fill = "none")

# b: error against the half-width the interval issued, eight-unit design
# annotated because it spans the widest range of distance
w8 <- dec[dec$design == LEV[length(LEV)], ]
rise_e <- 100 * (w8$rmse[nrow(w8)] / w8$rmse[1] - 1)
rise_w <- 100 * (w8$w[nrow(w8)] / w8$w[1] - 1)
message(sprintf("[fig8] 8x25: error +%.1f%%, half-width +%.1f%%", rise_e, rise_w))
dec_long <- rbind(
  data.frame(dec[c("design", "d_km")], quantity = "error", value = dec$rmse),
  data.frame(dec[c("design", "d_km")], quantity = "width", value = dec$w))
p_b <- ggplot(dec_long, aes(d_km, value, colour = design,
                            linetype = quantity, group = interaction(design, quantity))) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.5, show.legend = FALSE) +
  annotate("text", x = 2.2, y = 59, label = "Half-width issued (dashed)",
           hjust = 0, vjust = 1, size = pt2mm(PT_SMALL), family = FONT,
           colour = INK, fontface = "bold") +
  annotate("text", x = 2.2, y = 21, label = "Realised RMSE (solid)",
           hjust = 0, vjust = 0, size = pt2mm(PT_SMALL), family = FONT,
           colour = INK, fontface = "bold") +
  annotate("text", x = 2.2, y = 44, hjust = 0, vjust = 0.5,
           label = sprintf("8 \u00d7 25 design: error +%.0f%%,\nhalf-width +%.0f%%",
                           rise_e, rise_w),
           size = pt2mm(PT_SMALL - 1), family = FONT, colour = COL_CLUSW,
           fontface = "bold", lineheight = 0.95) +
  scale_colour_manual(values = PAL) +
  scale_linetype_manual(values = c(error = "solid", width = "22")) +
  xsc("Distance (km)") +
  scale_y_continuous(limits = c(20, 59), breaks = seq(20, 55, 5)) +
  labs(title = "b  Error grows, width does not",
       subtitle = "Mean in each distance band, pooled over\nten repetitions",
       y = expression("Mg ha"^-1)) +
  theme_paper() + guides(colour = "none", linetype = "none")

# c: coverage within shared distance bands
p_c <- ggplot(dec, aes(d_km, cover, colour = design)) +
  geom_hline(yintercept = 0.90, colour = INK, linewidth = 0.5, linetype = "22") +
  annotate("text", x = 2.1, y = 0.905, label = "nominal 0.90", hjust = 0,
           vjust = 0, size = pt2mm(PT_SMALL), family = FONT, colour = INK,
           fontface = "bold") +
  geom_errorbar(aes(ymin = cov_lo, ymax = cov_hi), width = 0.05,
                linewidth = 0.6) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2, shape = 21, fill = "white", stroke = 1) +
  scale_colour_manual(values = PAL) +
  xsc("Distance to the nearest calibration point (km)") +
  scale_y_continuous(limits = c(0.70, 1.0), breaks = seq(0.7, 1, 0.05),
                     labels = c(".70", ".75", ".80", ".85", ".90", ".95", "1")) +
  labs(title = "c  At the same distance, the designs cover the same",
       subtitle = paste0("Coverage of the CV+ interval within each distance band; bars are 95% intervals from a\n",
                         "bootstrap over whole repetitions. What differs between designs is how much of the map\n",
                         "lies at each distance (a), not how the interval behaves there."),
       y = "Coverage") +
  theme_paper() +
  theme(legend.position = "bottom") +
  guides(colour = guide_legend(override.aes = list(linewidth = 0.9)))

fig <- ((p_a | p_b) / p_c) +
  plot_layout(heights = c(1, 1), guides = "collect") +
  plot_annotation(
    title = "Where the interval fails: distance from the calibration sample",
    subtitle = paste0("Calibration designs: simple random, n = 500; clustered, 20 units \u00d7 25 points, n = 500;\n",
                      "clustered, 8 \u00d7 25, n = 200. Ten repetitions each, scored on 500 validation points apiece."),
    theme = theme_paper()) &
  theme(legend.position = "bottom")

save_fig(fig, out_png, height_mm = 148)
write.csv(curves, out_csv, row.names = FALSE)
cat(sprintf("wrote %s\nwrote %s\n", out_png, out_csv))
