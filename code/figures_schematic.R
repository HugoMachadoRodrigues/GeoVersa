# =============================================================================
# figures_schematic.R
#
# Regenerates Text/dbtarf_article/figures/fig1_schematic.png — the DB-TARF
# pipeline diagram — so it stays in step with R/dbtarf.R.
#
# The correction library drawn here is the one actually implemented in
# R/dbtarf.R (verified against the mode strings in that file):
#   residual learners : resid_rf, resid_cubist, resid_mlp,
#                       resid_spatial (RBF/kernel), resid_kriging (ordinary
#                       kriging via gstat), resid_ensemble
#   delta / recalib.  : bias, cubist_delta, linear_cal, quadratic_cal,
#                       stack_rf_cubist
#   baseline          : RF_noharm (lambda = 0, do-no-harm)
# plus the spatially-adaptive shrinkage gate w(s) = exp(-d1(s) / ell).
#
# Usage:  Rscript code/figures_schematic.R
# =============================================================================

root <- "/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/GeoVersa"
setwd(root)
out_png <- "Text/dbtarf_article/figures/fig1_schematic.png"

grDevices::png(out_png, width = 2200, height = 1650, res = 170)
op <- par(mar = c(0.4, 0.4, 0.4, 0.4))
plot(NA, xlim = c(0, 100), ylim = c(0, 100), axes = FALSE, xlab = "", ylab = "",
     xaxs = "i", yaxs = "i")

box_col <- c(step = "#E8E8E8", cross = "#CFEBD8", sel = "#CFE0F5",
             refit = "#E8E8E8", pred = "#FBF0C4", eval = "#F7D6D6")
edge <- "#2C3E50"

# helper: rounded-ish box with centred multi-line text
draw_box <- function(ymid, h, lines, fill, cex_first = 1.20, cex_rest = 1.04,
                     x0 = 6, x1 = 94) {
  rect(x0, ymid - h / 2, x1, ymid + h / 2, col = fill, border = edge, lwd = 2.1)
  n <- length(lines)
  # first line bold/larger, remainder smaller
  ytop <- ymid + h / 2
  gap <- h / (n + 1)
  for (i in seq_len(n)) {
    text(50, ytop - i * gap, lines[i],
         cex = if (i == 1) cex_first else cex_rest,
         font = if (i == 1) 2 else 1)
  }
}

arrow_down <- function(y_from, y_to) {
  arrows(50, y_from, 50, y_to, length = 0.12, lwd = 2.4, col = edge)
}

# ---- boxes ------------------------------------------------------------------

draw_box(94, 8, c(
  "Calibration sample  S = { (y_i , x_i , s_i) },  n points",
  "the map is scored later on an INDEPENDENT probability sample"),
  box_col["step"])

arrow_down(90, 87.2)

draw_box(76.5, 20, c(
  "K-fold CROSS-FITTING on S   (out-of-fold, leakage-free)",
  "per fold: fit the RF profiles  ->  OOF base prediction  f-hat^(-k)",
  "build residual candidates from the OOF residuals:",
  "residual RF  ·  residual Cubist  ·  MLP  ·  ordinary kriging (gstat)",
  "RBF / kernel smoother  ·  ensemble of the above",
  "+ delta & recalibration: bias, Cubist-delta, linear, quadratic, RF+Cubist stack"),
  box_col["cross"])

arrow_down(66.5, 63.7)

draw_box(56.5, 13, c(
  "MODEL SELECTION by out-of-fold RMSE",
  "pick profile p*, correction mode c*, shrinkage lambda*, gate length ell",
  "accept a correction ONLY if it beats the RF-only OOF baseline (else lambda = 0)"),
  box_col["sel"])

arrow_down(50, 47.2)

draw_box(42, 8, c(
  "REFIT on the full calibration sample S",
  "RF_selected (profile p*)  and  the residual model (mode c*)"),
  box_col["refit"])

arrow_down(38, 35.2)

draw_box(27.5, 13, c(
  "PREDICTION over targets s",
  "y-hat(s)  =  RF_selected(s)  +  lambda*  ·  w(s)  ·  correction(s)",
  "w(s) = exp( - d1(s) / ell ) : spatially-adaptive gate, d1 = distance to nearest",
  "calibration point; it shrinks the correction under spatial extrapolation"),
  box_col["pred"])

arrow_down(21, 18.2)

draw_box(11.5, 12, c(
  "DESIGN-BASED ASSESSMENT on an independent probability sample",
  "point accuracy   : ME, RMSE, r^2, MEC        (Wadoux et al., 2021)",
  "interval accuracy: PICP, MPIW                (this paper's extension)"),
  box_col["eval"])

text(50, 2.2, "DB-TARF: Design-Based Targeted Adaptive Residual Forest",
     cex = 1.38, font = 2, col = "#1B5E20")

par(op)
grDevices::dev.off()
cat(sprintf("wrote %s\n", out_png))
