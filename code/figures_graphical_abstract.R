# =============================================================================
# figures_graphical_abstract.R
#
# The graphical abstract required by Ecological Modelling. The journal asks for
# 531 x 1328 pixels (h x w) or proportionally more, readable at 5 x 13 cm, so
# this renders at 3x: 1593 x 3984.
#
# It carries one message. An earlier version carried a different one, three bars
# showing that clustered calibration under-covers and a spatial-block
# calibration repairs it, which was true and is still in the paper. It was
# replaced because the paper's claim became more specific: the shortfall is not
# a property of clustering as such but of how far the map lies from the sample
# that calibrated it, and designs that differ in their marginal coverage cover
# almost identically at matched distance. That is a curve, not three bars.
#
# Every number is read from the campaign outputs. The bar version was hand-typed
# once and drifted from the campaign that produced it.
#
# Usage:  Rscript code/figures_graphical_abstract.R
# =============================================================================

local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
setwd(gv_root())
out_png <- "Text/ecomod_submission/figures/graphical_abstract.png"

DES <- list(
  list(dir = "pts_random",    lab = "simple random",     col = "#668B5A", pch = 21),
  list(dir = "pts_clus_mid",  lab = "clustered 20 x 25", col = "#BE9058", pch = 22),
  list(dir = "pts_clus_wide", lab = "clustered 8 x 25",  col = "#8C5A24", pch = 24)
)
BANDS <- c(0, 10, 25, 50, 100, 200, 400, Inf)

read_one <- function(d) {
  f <- gv_results(d$dir, "dbtarf_points.csv")
  if (!file.exists(f)) stop("missing point dump: ", f, "\nrun code/run_point_dump.sh")
  x <- read.csv(f)
  km <- pmax(x$dist_nn, 500) / 1000
  cov <- as.integer(x$obs >= x$lo & x$obs <= x$hi)
  b <- cut(km, BANDS, labels = FALSE, right = FALSE)
  tab <- do.call(rbind, lapply(seq_len(length(BANDS) - 1L), function(i) {
    k <- which(b == i)
    if (length(k) < 60L) return(NULL)
    data.frame(d = stats::median(km[k]), p = mean(cov[k]))
  }))
  list(tab = tab, med = stats::median(km), cov = mean(cov))
}
D <- lapply(DES, read_one)

# The repair, over every cell of the sensitivity table, so the claim in the
# strapline is the range the paper reports rather than one campaign's number.
blk <- unlist(lapply(c("clus_dens_n200", "clus_mid_n500", "clus_dens_n800",
                       "clus_spread_n200", "clus_spread_n800",
                       "final_random_confirm30", "rand_n200", "rand_n800"),
  function(d) {
    u <- read.csv(gv_results(d, "dbtarf_uncertainty.csv"))
    mean(u$PICP90[u$model == "DBTARF_spcal"])
  }))

W <- 3984; H <- 1593
grDevices::png(out_png, width = W, height = H, res = 300)
op <- par(mar = c(0, 0, 0, 0), bg = "white")
plot(NA, xlim = c(0, 100), ylim = c(0, 100), axes = FALSE, xlab = "", ylab = "",
     xaxs = "i", yaxs = "i")

# --- left: the claim ---------------------------------------------------------
text(2, 88, "Interval coverage is a", adj = 0, cex = 1.3, font = 2, col = "#1B2E20")
text(2, 78, "design-based quantity", adj = 0, cex = 1.3, font = 2, col = "#1B2E20")

text(2, 63, expression(hat(P)[HT] == sum(frac(1, pi[k]) * I(y[k] %in% PI[k]),
                                         k %in% S, "") / N),
     adj = 0, cex = 1.05, col = "#3F6B48")

text(2, 47, "Estimate it on an independent", adj = 0, cex = 0.9, col = "#4A4A4A")
text(2, 40, "probability sample, exactly as", adj = 0, cex = 0.9, col = "#4A4A4A")
text(2, 33, "map accuracy is estimated.", adj = 0, cex = 0.9, col = "#4A4A4A")

text(2, 18, "Amazon above-ground biomass,", adj = 0, cex = 0.78, col = "#7A7A7A")
text(2, 11, "1.3 M cells, nominal 0.90", adj = 0, cex = 0.78, col = "#7A7A7A")

segments(33, 6, 33, 92, col = "#D8D2C4", lwd = 2)

# --- right: coverage against distance ----------------------------------------
xl <- 44; xr <- 88; yb <- 30; yt <- 76
lx <- c(log10(3), log10(500))
sx <- function(d) xl + (log10(d) - lx[1]) / diff(lx) * (xr - xl)
ry <- c(0.72, 0.98)
sy <- function(p) yb + (p - ry[1]) / diff(ry) * (yt - yb)

text(xl - 1, 90, "The map is covered where the sample is, and not where it is not.",
     adj = 0, cex = 0.95, font = 2, col = "#1B2E20")
text(xl - 1, 83.5, "Three designs, one curve: what differs is how far the map lies from the sample.",
     adj = 0, cex = 0.78, col = "#5A5A5A")

# nominal level first, so the curves sit on top of it
segments(xl - 1, sy(0.90), xr + 1, sy(0.90), col = "#B03A2E", lwd = 2, lty = 2)
text(xr + 1.5, sy(0.90), "nominal\n0.90", adj = 0, cex = 0.72, col = "#B03A2E", font = 2)

# axes
segments(xl - 1, yb, xr + 1, yb, col = "#3A3A3A", lwd = 1.2)
for (d in c(3, 10, 30, 100, 300)) {
  segments(sx(d), yb, sx(d), yb - 1.2, col = "#3A3A3A", lwd = 1.2)
  text(sx(d), yb - 4.5, d, cex = 0.72, col = "#3A3A3A")
}
text((xl + xr) / 2, yb - 11, "km to the nearest calibration point",
     cex = 0.8, col = "#3A3A3A")
segments(xl - 1, yb, xl - 1, yt, col = "#3A3A3A", lwd = 1.2)
for (p in c(0.75, 0.85, 0.95)) {
  segments(xl - 1, sy(p), xl - 2.2, sy(p), col = "#3A3A3A", lwd = 1.2)
  text(xl - 3.2, sy(p), sprintf("%.2f", p), adj = 1, cex = 0.72,
       col = "#3A3A3A")
}

for (i in seq_along(D)) {
  t <- D[[i]]$tab
  k <- t$d >= 3 & t$d <= 500
  lines(sx(t$d[k]), sy(t$p[k]), col = DES[[i]]$col, lwd = 2.4)
  points(sx(t$d[k]), sy(t$p[k]), pch = DES[[i]]$pch, bg = "white",
         col = DES[[i]]$col, cex = 0.75, lwd = 1.8)
}

# legend, with each design's median distance, which is the quantity that
# actually separates them
ly <- 71
for (i in seq_along(D)) {
  yy <- ly - (i - 1) * 5
  segments(xl + 1.5, sy(0.79) - (i - 1) * 2.2, xl + 5, sy(0.79) - (i - 1) * 2.2,
           col = DES[[i]]$col, lwd = 2.4)
  points(mean(c(xl + 1.5, xl + 5)), sy(0.79) - (i - 1) * 2.2, pch = DES[[i]]$pch,
         bg = "white", col = DES[[i]]$col, cex = 0.75, lwd = 1.8)
  text(xl + 6, sy(0.79) - (i - 1) * 2.2,
       sprintf("%s: median %.0f km, coverage %.3f",
               DES[[i]]$lab, D[[i]]$med, D[[i]]$cov),
       adj = 0, cex = 0.68, col = "#3A3A3A")
}

text(xl - 1, 9,
     sprintf("A spatial-block calibration holds %.2f to %.2f across every design, size and geometry tested.",
             min(blk), max(blk)),
     adj = 0, cex = 0.72, col = "#26492F", font = 2)

par(op)
grDevices::dev.off()
cat(sprintf("wrote %s (%d x %d px)\n", out_png, W, H))
