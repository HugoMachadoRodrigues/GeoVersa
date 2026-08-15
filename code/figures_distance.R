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
suppressPackageStartupMessages({ library(ggplot2); library(patchwork) })

out_png <- "Text/ecomod_submission/figures/fig8_distance.png"
out_csv <- gv_results("fig8_curves.csv")
B <- 1000L                      # bootstrap replicates, resampling repetitions
set.seed(20260814L)

DESIGNS <- list(
  list(dir = "pts_random",    lab = "Simple random, 500",        col = "#668B5A"),
  list(dir = "pts_clus_mid",  lab = "Clustered 20/25, 500",      col = "#BE9058"),
  list(dir = "pts_clus_wide", lab = "Clustered 8/25, 200",       col = "#8C5A24")
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
theme_d <- function() {
  theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 11),
          legend.position = "bottom", legend.title = element_blank())
}
XLIM <- range(GRID)
xsc <- function() scale_x_log10(
  breaks = c(2, 5, 10, 25, 50, 100, 200, 400),
  labels = c("2", "5", "10", "25", "50", "100", "200", "400"),
  limits = XLIM)

# Only the line layers carry the design legend. Ribbons and points repeat it,
# which is how the first draft acquired three identical entries.
p_a <- ggplot(curves, aes(d_km, frac, colour = design, fill = design)) +
  geom_ribbon(aes(ymin = frac_lo, ymax = frac_hi), alpha = 0.18, colour = NA,
              show.legend = FALSE) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = PAL) + scale_fill_manual(values = PAL) +
  xsc() + scale_y_continuous(limits = c(0, 1)) +
  labs(title = expression(bold("a  How much of the map lies beyond ") * bold(delta)),
       x = NULL, y = "fraction of domain") +
  theme_d() + guides(colour = "none", fill = "none")

# Long form so the two quantities are a mapped aesthetic with a legend, rather
# than two anonymous line types the reader has to guess at.
dec_long <- rbind(
  data.frame(dec[c("design", "d_km")], quantity = "realised error (RMSE)", value = dec$rmse),
  data.frame(dec[c("design", "d_km")], quantity = "interval half-width",   value = dec$w))
dec_long$quantity <- factor(dec_long$quantity,
                            levels = c("realised error (RMSE)", "interval half-width"))

p_b <- ggplot(dec_long, aes(d_km, value, colour = design, linetype = quantity)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.2, show.legend = FALSE) +
  scale_colour_manual(values = PAL) +
  scale_linetype_manual(values = c("solid", "22")) +
  xsc() + expand_limits(y = 0) +
  labs(title = "b  The error responds to distance and the interval does not",
       x = NULL, y = expression("Mg ha"^-1)) +
  theme_d() +
  # colour legend suppressed here: panel b maps linetype as well, so its colour
  # key carries a different glyph and patchwork will not merge it with the other
  # two, producing the same three designs twice
  guides(colour = "none",
         linetype = guide_legend(order = 2, override.aes = list(colour = "grey30")))

# Local coverage is the primary here. The cumulative curve answers a different
# question, "everything beyond delta", and because each design integrates over
# its own distance distribution the three cumulative curves cannot coincide even
# when the designs behave identically at matched distance. Showing it as the
# heavy line hid exactly the comparison the panel exists to make.
p_c <- ggplot(dec, aes(d_km, cover, colour = design, fill = design)) +
  geom_line(data = curves, aes(y = cov), linewidth = 0.4, alpha = 0.45,
            show.legend = FALSE) +
  geom_errorbar(aes(ymin = cov_lo, ymax = cov_hi), width = 0.04,
                linewidth = 0.5, show.legend = FALSE) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.6, shape = 21, fill = "white", stroke = 0.8,
             show.legend = FALSE) +
  geom_hline(yintercept = 0.90, linetype = "dashed", colour = "#B03A2E",
             linewidth = 0.7) +
  annotate("text", x = 2.2, y = 0.907, label = "nominal 0.90", hjust = 0,
           vjust = 0, size = 3, colour = "#B03A2E", fontface = "bold") +
  annotate("text", x = 2.2, y = 0.625,
           label = "circles: coverage within a distance band, 95% CI over repetitions\nfaint line: coverage of everything beyond delta",
           hjust = 0, vjust = 0, size = 2.7, colour = "grey35", lineheight = 1.1) +
  scale_colour_manual(values = PAL) + scale_fill_manual(values = PAL) +
  xsc() + coord_cartesian(ylim = c(0.60, 1.0)) +
  labs(title = "c  At the same distance, the designs cover the same",
       x = "Distance to the nearest calibration point (km)",
       y = "coverage") +
  theme_d() +
  # the only design key in the composition: panel a suppresses its own and
  # panel b maps linetype as well, so neither merges cleanly with this one
  guides(fill = "none", colour = guide_legend(order = 1))

fig <- (p_a / p_b / p_c) + plot_layout(guides = "collect", heights = c(0.85, 1, 1.25)) &
  theme(legend.position = "bottom", legend.box = "vertical",
        legend.margin = margin(0, 0, 0, 0))

# Kept wider than tall. At width = textwidth an aspect ratio above about 1.1
# makes the float taller than the text block, and LaTeX defers it to the end
# of the document; this one was landing after the bibliography.
ggsave(out_png, fig, width = 9.2, height = 8.4, dpi = 300, bg = "white")
write.csv(curves, out_csv, row.names = FALSE)
cat(sprintf("wrote %s\nwrote %s\n", out_png, out_csv))
