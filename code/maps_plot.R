# =============================================================================
# maps_plot.R  —  STEP 2 of 2 (cheap; safe to re-run while tuning the design)
#
# Reads results/maps_v030/map_predictions.rds (written by maps_compute.R) and
# draws Figure 2: one calibration sample, the corrected map against the
# default forest, scored over every cell of the benchmark.
#
# The figure is built around this paper's question rather than around the
# usual observed / predicted / error triptych, which is the layout of the
# reference study and says nothing a reader could not get from the RMSE alone:
#   a  the reference surface with the calibration sample on it, so the reader
#      sees how sparse 500 points are over 1.3 M cells;
#   b  where the correction changes the error, as the difference in RMSE in
#      10 km blocks, so improvement and damage are both visible;
#   c  the error against distance to the nearest calibration point, computed
#      over the whole population rather than a validation sample, which is the
#      population counterpart of Section 7.5;
#   d  the gain in each of those distance bands, with the share of the map each
#      band holds, so the overall difference can be read as a sum.
#
# Usage:  Rscript code/maps_plot.R
# =============================================================================

local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
setwd(gv_root())
source("code/figure_theme.R")
suppressPackageStartupMessages(library(scales))

in_rds  <- "results/maps_v030/map_predictions.rds"
out_png <- "Text/ecomod_submission/figures/fig7_maps.png"
stopifnot(file.exists(in_rds))
obj  <- readRDS(in_rds)
d    <- obj$maps
cal  <- obj$cal
meta <- obj$meta
N    <- nrow(d)
rmse <- function(e) sqrt(mean(e^2))

pop_rf   <- rmse(d$err_rf)
pop_tarf <- rmse(d$err_dbtarf)
message(sprintf("[maps] %d cells | mode %s, lambda %s", N, meta$selected_mode,
                format(meta$selected_lambda)))
message(sprintf("  RF   RMSE %.3f   ME %+.3f", pop_rf, mean(d$err_rf)))
message(sprintf("  TARF RMSE %.3f   ME %+.3f", pop_tarf, mean(d$err_dbtarf)))

# ---- distance of every cell to the nearest calibration point ---------------
d$dist_km <- FNN::get.knnx(as.matrix(cal[, c("x", "y")]),
                           as.matrix(d[, c("x", "y")]), k = 1)$nn.dist[, 1] / 1000

# Coordinates are metres in an equal-area projection; show kilometres from the
# south-west corner so the axes say how large the domain is.
x0 <- min(d$x); y0 <- min(d$y)
d$xk <- (d$x - x0) / 1000; d$yk <- (d$y - y0) / 1000
cal$xk <- (cal$x - x0) / 1000; cal$yk <- (cal$y - y0) / 1000

theme_map <- function() {
  theme_paper() +
    theme(axis.title = element_blank(), axis.text = element_blank(),
          axis.line.x = element_blank(), axis.ticks.x = element_blank(),
          panel.grid.major = element_blank(),
          legend.position = "bottom",
          legend.key.width = unit(9, "mm"), legend.key.height = unit(2, "mm"),
          legend.title = element_text(size = PT_SMALL, colour = INK2,
                                      vjust = 1),
          legend.text = element_text(size = PT_SMALL - 1))
}
scalebar <- function() {
  # 500 km bar in the empty north-east corner of the rectangle
  list(annotate("segment", x = 1080, xend = 1580, y = 880, yend = 880,
                linewidth = 0.8, colour = INK),
       annotate("text", x = 1330, y = 860, label = "500 km", vjust = 1,
                size = pt2mm(PT_SMALL - 1), family = FONT, colour = INK))
}

# ---- a: reference surface and the calibration sample ------------------------
p_a <- ggplot(d, aes(xk, yk, fill = truth)) +
  geom_raster() +
  geom_point(data = cal, aes(xk, yk), inherit.aes = FALSE, shape = 21,
             fill = "white", colour = INK, size = 0.55, stroke = 0.25) +
  scalebar() +
  coord_equal(expand = FALSE) +
  scale_fill_gradientn(colours = SEQ_GREEN, limits = c(0, 400),
                       oob = squish, breaks = c(0, 100, 200, 300, 400),
                       name = expression("Biomass (Mg ha"^-1 * ")"),
                       guide = guide_colourbar(title.position = "top")) +
  labs(title = "a  Reference biomass and sample",
       subtitle = sprintf("%s cells at 1 km; dots: the %d\ncalibration points, simple random",
                         format(N, big.mark = ","), nrow(cal))) +
  theme_map()

# ---- b: where the correction changes the error ------------------------------
BLK <- 10                                             # km
d$bx <- floor(d$xk / BLK); d$by <- floor(d$yk / BLK)
blk <- stats::aggregate(cbind(e2r = err_rf^2, e2t = err_dbtarf^2) ~ bx + by,
                        data = d, FUN = mean)
blk$delta <- sqrt(blk$e2t) - sqrt(blk$e2r)
blk$xk <- (blk$bx + 0.5) * BLK; blk$yk <- (blk$by + 0.5) * BLK
share_better <- mean(blk$delta < 0)
lim_b <- 6
p_b <- ggplot(blk, aes(xk, yk, fill = delta)) +
  geom_tile(width = BLK, height = BLK) +
  scalebar() +
  coord_equal(expand = FALSE, xlim = range(d$xk), ylim = range(d$yk)) +
  scale_fill_gradient2(low = DIV_LO, mid = DIV_MID, high = DIV_HI, midpoint = 0,
                       limits = c(-lim_b, lim_b), oob = squish,
                       breaks = c(-6, -3, 0, 3, 6),
                       labels = c("-6", "-3", "0", "+3", "+6"),
                       name = expression("TARF " - " RF (Mg ha"^-1 * ")"),
                       guide = guide_colourbar(title.position = "top")) +
  labs(title = "b  Where TARF helps and hurts",
       subtitle = sprintf("RMSE change in 10 km blocks; blue in\n%.0f%% of %s blocks, red elsewhere",
                         100 * share_better, format(nrow(blk), big.mark = ","))) +
  theme_map()

# ---- c and d: error against distance, over the whole population -------------
BANDS <- c(0, 10, 20, 30, 40, 60, Inf)
LAB   <- c("<10", "<20", "<30", "<40", "<60", "\u226560")
d$band <- cut(d$dist_km, BANDS, labels = LAB, right = FALSE)
bd <- do.call(rbind, lapply(split(d, d$band), function(z) data.frame(
  band = z$band[1], n = nrow(z), med = stats::median(z$dist_km),
  rf = rmse(z$err_rf), tarf = rmse(z$err_dbtarf))))
bd$share <- bd$n / N
bd$delta <- bd$tarf - bd$rf
# Bars span their band, so the axis is distance itself; the open band is
# drawn to 100 km, beyond the farthest cell (102 km is the maximum).
bd$lo <- head(BANDS, -1); bd$hi <- pmin(tail(BANDS, -1), 100)
print(bd, row.names = FALSE)

long <- rbind(data.frame(x = bd$med, v = bd$rf,   m = "Default random forest"),
              data.frame(x = bd$med, v = bd$tarf, m = "TARF"))
k <- nrow(bd)
p_c <- ggplot(long, aes(x, v, colour = m)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  annotate("text", x = bd$med[k] + 3, y = bd$rf[k] + 0.15,
           label = sprintf("RF %.2f", pop_rf), hjust = 0,
           vjust = 0, size = pt2mm(PT_SMALL), family = FONT, colour = INK2,
           lineheight = 0.95) +
  annotate("text", x = bd$med[k] + 3, y = bd$tarf[k] - 0.15,
           label = sprintf("TARF %.2f", pop_tarf), hjust = 0,
           vjust = 1, size = pt2mm(PT_SMALL), family = FONT, colour = COL_TARF,
           fontface = "bold", lineheight = 0.95) +
  scale_colour_manual(values = c("Default random forest" = COL_RF,
                                 "TARF" = COL_TARF), guide = "none") +
  scale_x_continuous(limits = c(0, 90), breaks = seq(0, 80, 20),
                     expand = expansion(add = c(1, 0))) +
  scale_y_continuous(limits = c(28, 34), breaks = 28:34) +
  coord_cartesian(clip = "off") +
  labs(title = "c  Error grows with distance",
       subtitle = "Population RMSE in distance bands, all\n1.31 M cells; labels: overall RMSE",
       x = "Distance to nearest calibration point (km)",
       y = expression("RMSE (Mg ha"^-1 * ")")) +
  theme_paper() + theme(plot.margin = margin(4, 16, 4, 4))

p_d <- ggplot(bd) +
  geom_hline(yintercept = 0, colour = AXIS, linewidth = 0.4) +
  geom_col(aes(x = seq_len(k), y = delta), fill = DIV_LO, width = 0.72) +
  geom_text(aes(x = seq_len(k), y = delta - 0.04,
                label = sprintf("%.0f%%", 100 * share)),
            vjust = 1, size = pt2mm(PT_SMALL - 1), family = FONT, colour = INK2) +
  scale_x_continuous(breaks = seq_len(k), labels = LAB,
                     expand = expansion(add = 0.5)) +
  scale_y_continuous(limits = c(-0.9, 0.02), breaks = seq(-0.8, 0, 0.2)) +
  labs(title = "d  TARF gains at every distance",
       subtitle = "RMSE change, TARF \u2212 RF, in each band;\nlabel: share of the map in the band",
       x = "Distance band, upper limit (km)",
       y = expression(Delta * "RMSE (Mg ha"^-1 * ")")) +
  theme_paper() +
  theme(panel.grid.major.x = element_blank(),
        axis.text.x = element_text(size = PT_SMALL - 1))

fig <- (p_a | p_b) / (p_c | p_d) + plot_layout(heights = c(1, 1.15))

save_fig(fig, out_png, height_mm = 140)
