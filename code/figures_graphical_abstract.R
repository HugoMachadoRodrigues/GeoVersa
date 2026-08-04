# =============================================================================
# figures_graphical_abstract.R
#
# Builds the graphical abstract required by Ecological Modelling. The journal asks for 531 x 1328 pixels (h x w) or proportionally
# more, readable at 5 x 13 cm, so this renders at 3x that: 1593 x 3984.
#
# The panel carries the paper's single message and nothing else: coverage is a
# design-based quantity, a clustered sample breaks it, and a spatial-block
# calibration repairs it. Every number matches the generated coverage table,
# which is built from results/rev1_*_confirm30/dbtarf_uncertainty_summary.csv.
#
# Usage:  Rscript code/figures_graphical_abstract.R
# =============================================================================

root <- "/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/GeoVersa"
setwd(root)
out_png <- "Text/ecomod_submission/figures/graphical_abstract.png"

# --- the measured numbers, read from the same campaign as the coverage table --
bars <- data.frame(
  label = c("Simple random\ncalibration",
            "Clustered\ncalibration",
            "Clustered +\nspatial-block\ncalibration"),
  picp  = c(0.895, 0.849, 0.946),
  # brown = broken, green = honest
  fill  = c("#668B5A", "#8C5A24", "#26492F"),
  stringsAsFactors = FALSE
)
nominal <- 0.90

W <- 3984; H <- 1593                     # 3x the journal minimum, same aspect
grDevices::png(out_png, width = W, height = H, res = 300)
op <- par(mar = c(0, 0, 0, 0), bg = "white")
plot(NA, xlim = c(0, 100), ylim = c(0, 100), axes = FALSE, xlab = "", ylab = "",
     xaxs = "i", yaxs = "i")

# --- left panel: the claim ---------------------------------------------------
text(2, 88, "Prediction-interval coverage is", adj = 0, cex = 1.35, font = 2,
     col = "#1B2E20")
text(2, 76, "a design-based quantity", adj = 0, cex = 1.35, font = 2,
     col = "#1B2E20")

text(2, 60, expression(hat(P)[HT] == sum(frac(1, pi[k]) * I(y[k] %in% PI[k]),
                                         k %in% S, "") / N),
     adj = 0, cex = 1.15, col = "#3F6B48")

text(2, 44, "Estimate it from an independent", adj = 0, cex = 0.95, col = "#4A4A4A")
text(2, 37, "probability sample, exactly as", adj = 0, cex = 0.95, col = "#4A4A4A")
text(2, 30, "map accuracy is estimated.", adj = 0, cex = 0.95, col = "#4A4A4A")

text(2, 15, "Amazon above-ground biomass,", adj = 0, cex = 0.82, col = "#7A7A7A")
text(2,  8, "1.3 M cells, n = 500, nominal 0.90", adj = 0, cex = 0.82, col = "#7A7A7A")

# thin rule between the two halves
segments(37, 6, 37, 92, col = "#D8D2C4", lwd = 2)

# --- right panel: the measurement --------------------------------------------
x0 <- 44; bw <- 11; gap <- 4.5
y_base <- 22; y_top <- 82
rng <- c(0.78, 0.98)
sc <- function(p) y_base + (p - rng[1]) / diff(rng) * (y_top - y_base)
y_nom <- sc(nominal)
x_end <- x0 + 3 * bw + 2 * gap

# Bars first, then the nominal line on top of them, so the reference level is
# never hidden behind a bar that happens to exceed it.
for (i in seq_len(nrow(bars))) {
  xl <- x0 + (i - 1) * (bw + gap)
  rect(xl, y_base, xl + bw, sc(bars$picp[i]), col = bars$fill[i],
       border = NA)
  text(xl + bw / 2, sc(bars$picp[i]) + 3.5, sprintf("%.3f", bars$picp[i]),
       cex = 1.15, font = 2, col = bars$fill[i])
  text(xl + bw / 2, y_base - 5, bars$label[i], cex = 0.8, col = "#3A3A3A")
}
segments(x0 - 3, y_nom, x_end + 1, y_nom, col = "#B03A2E", lwd = 2.4, lty = 2)
text(x_end + 2.5, y_nom, "nominal\n0.90", adj = 0, cex = 0.85,
     col = "#B03A2E", font = 2)
segments(x0 - 3, y_base, x_end, y_base, col = "#3A3A3A",
         lwd = 1.4)
text(x0 - 5, (y_base + y_top) / 2, "coverage (PICP)", srt = 90, cex = 0.92,
     col = "#3A3A3A")

text(x0, 93, "Clustering breaks it. A spatial-block calibration repairs it.",
     adj = 0, cex = 1.0, font = 2, col = "#1B2E20")

par(op)
grDevices::dev.off()
cat(sprintf("wrote %s (%d x %d px)\n", out_png, W, H))
