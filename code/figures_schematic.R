# =============================================================================
# figures_schematic.R
#
# Figure 1: how the map and its interval are built from one calibration sample
# and judged on another.
#
# Drawn as two lanes because the paper's argument has two objects that share an
# input and differ in everything else: the point map comes from the
# configuration selected on all of S and refitted there; the interval comes
# from fold models in which the whole selection was re-run (Section 5), so it
# is not a band around the map. Both are then scored on an independent
# probability sample, which is the only step that carries the design-based
# argument of Section 2.
#
# The correction library named here is the one implemented in R/dbtarf.R
# (mode strings resid_rf, resid_cubist, resid_mlp, resid_kriging,
# resid_spatial, resid_ensemble, and the recalibrations), with lambda = 0 the
# do-no-harm candidate.
#
# Usage:  Rscript code/figures_schematic.R
# =============================================================================

local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
setwd(gv_root())
source("code/figure_theme.R")
out_png <- "Text/ecomod_submission/figures/fig1_schematic.png"

W <- FIG_W; H <- 141                      # mm; the plot is drawn in mm
ACC_MAP <- COL_TARF                       # the point map
ACC_INT <- "#008300"                      # the interval
ACC_VAL <- INK                            # the independent assessment

# A box: white card, hairline border, coloured accent bar on the left, a bold
# heading and body lines. Body lines may be plotmath when parse = TRUE.
box <- function(x0, x1, y1, h, acc, head, body, parse = FALSE, fill = "white",
                gap = if (parse) 5.4 else 4.4) {
  y0 <- y1 - h
  n <- length(body)
  list(
    annotate("rect", xmin = x0, xmax = x1, ymin = y0, ymax = y1, fill = fill,
             colour = AXIS, linewidth = 0.35),
    annotate("rect", xmin = x0, xmax = x0 + 1.4, ymin = y0, ymax = y1,
             fill = acc, colour = NA),
    annotate("text", x = x0 + 3.5, y = y1 - 2.2, label = head, hjust = 0,
             vjust = 1, size = pt2mm(PT_BASE), family = FONT, fontface = "bold",
             colour = INK),
    if (n) annotate("text", x = x0 + 3.5, y = y1 - 7.6 - (seq_len(n) - 1) * gap,
                    label = body, hjust = 0, vjust = 1, parse = parse,
                    size = pt2mm(PT_SMALL), family = FONT, colour = INK2)
  )
}
arrow_down <- function(x, y0, y1, col = INK2) {
  annotate("segment", x = x, xend = x, y = y0, yend = y1, colour = col,
           linewidth = 0.5,
           arrow = grid::arrow(length = grid::unit(1.8, "mm"), type = "closed"))
}

L0 <- 2;  L1 <- 67.3                      # left lane
R0 <- 70.3; R1 <- W - 2                   # right lane
mid <- function(a, b) (a + b) / 2

p <- ggplot() +
  coord_cartesian(xlim = c(0, W), ylim = c(0, H), expand = FALSE) +
  theme_void() +
  theme(plot.margin = margin(0, 0, 0, 0)) +

  # ---- the input ------------------------------------------------------------
  box(L0, R1, H - 2, 15, INK2, "Calibration sample S, n points",
      c("Drawn under any design: simple random, systematic or clustered. It fits the model",
        "and builds the interval; it is never used to judge either.")) +
  arrow_down(mid(L0, L1), H - 17, H - 24, ACC_MAP) +
  arrow_down(mid(R0, R1), H - 17, H - 24, ACC_INT) +

  # ---- lane headers ---------------------------------------------------------
  annotate("text", x = L0, y = H - 26, label = "THE POINT MAP", hjust = 0,
           vjust = 1, size = pt2mm(PT_SMALL), family = FONT, fontface = "bold",
           colour = ACC_MAP) +
  annotate("text", x = R0, y = H - 26, label = "THE PREDICTION INTERVAL",
           hjust = 0, vjust = 1, size = pt2mm(PT_SMALL), family = FONT,
           fontface = "bold", colour = ACC_INT) +

  # ---- left lane: TARF ------------------------------------------------------
  box(L0, L1, H - 31, 28, ACC_MAP, "1  Select out of fold",
      c("K-fold cross-fitting scores every candidate:",
        "6 forest profiles × correction library",
        "× shrinkage λ × gate × variogram recipe.",
        "λ = 0 (no correction) is always a candidate.")) +
  arrow_down(mid(L0, L1), H - 59, H - 63, ACC_MAP) +
  box(L0, L1, H - 63, 14, ACC_MAP, "2  Refit the winner on all of S",
      "base forest and selected residual corrector") +
  arrow_down(mid(L0, L1), H - 77, H - 81, ACC_MAP) +
  box(L0, L1, H - 81, 23, ACC_MAP, "3  Predict every cell",
      c("hat(y)(s) == hat(f)(s) + lambda^'*' * w(s) * hat(g)(s)",
        "w(s) == exp(-d[1](s) / \u2113)",
        "d[1]*': distance to nearest calibration point'"),
      parse = TRUE) +

  # ---- right lane: CV+ ------------------------------------------------------
  box(R0, R1, H - 31, 28, ACC_INT, "1  Re-run the selection per fold",
      c("For each of K folds, the whole of step 1 is",
        "repeated without that fold, so no residual",
        "comes from a model that saw its point or",
        "chose its configuration on it.")) +
  arrow_down(mid(R0, R1), H - 59, H - 63, ACC_INT) +
  box(R0, R1, H - 63, 14, ACC_INT, "2  Conformity scores",
      "r[i] == '|' * y[i] - hat(f)^{(-k)}*(x[i]) * '|'", parse = TRUE) +
  arrow_down(mid(R0, R1), H - 77, H - 81, ACC_INT) +
  box(R0, R1, H - 81, 23, ACC_INT, "3  CV+ bounds for every cell",
      c("Quantiles of the n fold-wise bounds",
        "Not a band around the map of step 3",
        "Guarantee ≥ 1 − 2α under exchangeability")) +

  # ---- the assessment -------------------------------------------------------
  arrow_down(mid(L0, L1), H - 104, H - 111, ACC_MAP) +
  arrow_down(mid(R0, R1), H - 104, H - 111, ACC_INT) +
  box(L0, R1, H - 111, 27, ACC_VAL,
      "Judged on an independent probability sample (the validation design)",
      c("Drawn with known inclusion probabilities, independently of S. Each cell contributes",
        "its squared error and whether its interval held the true value, and the Horvitz-Thompson",
        "estimator makes RMSE approximately, and coverage (PICP) exactly, design-unbiased.",
        "Nothing about exchangeability or autocorrelation is assumed at this step."),
      fill = PANEL)

save_fig(p, out_png, height_mm = H)
