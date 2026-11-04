# =============================================================================
# figures_uncertainty.R
#
# Figure 4: design-based interval coverage and width, one dot per repetition.
#
# It reads exactly the files tables_manuscript.R reads and shows exactly the
# rows Table 7 shows, so the figure and the table cannot disagree. The first
# version of this figure had no generator and drifted from the table; the
# second used boxplots, which summarise thirty repetitions into five numbers
# and hide what the text says matters: under clustering the whole distribution
# moves down and roughly doubles its spread.
#
# Colour is the calibration design, as in every figure of the paper.
#
# Usage:  Rscript code/figures_uncertainty.R
# =============================================================================

local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
setwd(gv_root())
source("code/figure_theme.R")

out_png <- "Text/ecomod_submission/figures/fig6_uncertainty.png"
dir_for <- function(sc) gv_results(sprintf("final_%s_confirm30", sc))

DESIGNS <- c(random = "Simple random", regular_grid = "Systematic",
             clustered_random = "Clustered, 20 × 25")
DCOL <- stats::setNames(c(COL_SRS, COL_SYS, COL_CLUS), unname(DESIGNS))
METHODS <- c(DBTARF = "CV+ (deployed)", DBTARF_oofq = "Pooled OOF quantile",
             DBTARF_dbcal = "Covariate-shift weighted",
             DBTARF_spcal = "Spatial-block calibration",
             RF_quantile = "Quantile regression forest")
# The rows of Table 7, design by design.
SHOWN <- list(random = c("DBTARF", "DBTARF_oofq", "DBTARF_spcal", "RF_quantile"),
              regular_grid = c("DBTARF", "DBTARF_spcal", "RF_quantile"),
              clustered_random = names(METHODS))

d <- do.call(rbind, lapply(names(DESIGNS), function(sc) {
  u <- read.csv(file.path(dir_for(sc), "dbtarf_uncertainty.csv"))
  u <- u[u$model %in% SHOWN[[sc]], ]
  data.frame(design = DESIGNS[[sc]], model = u$model, picp = u$PICP90,
             mpiw = u$MPIW90, stringsAsFactors = FALSE)
}))
s <- do.call(rbind, lapply(split(d, list(d$design, d$model), drop = TRUE),
  function(z) {
    tt <- stats::t.test(z$picp)
    data.frame(design = z$design[1], model = z$model[1], n = nrow(z),
               picp = mean(z$picp), lo = tt$conf.int[1], hi = tt$conf.int[2],
               mpiw = mean(z$mpiw), below = sum(z$picp < 0.90))
  }))
print(s[order(s$model, s$design), ], row.names = FALSE, digits = 3)

# Rows: one band per method, the designs stacked inside it.
MROW <- stats::setNames(rev(seq_along(METHODS)), names(METHODS))
DOFF <- stats::setNames(c(0.24, 0, -0.24), unname(DESIGNS))
d$y <- MROW[d$model] + DOFF[d$design]
s$y <- MROW[s$model] + DOFF[s$design]
set.seed(11)
d$yj <- d$y + stats::runif(nrow(d), -0.06, 0.06)
d$design <- factor(d$design, levels = unname(DESIGNS))
s$design <- factor(s$design, levels = unname(DESIGNS))
bands <- data.frame(y = MROW, alt = seq_along(MROW) %% 2 == 0)

yscale <- scale_y_continuous(breaks = unname(MROW), labels = unname(METHODS),
                             expand = expansion(add = 0))
YL <- c(0.55, length(METHODS) + 0.75)
band_layer <- geom_rect(data = bands[bands$alt, ], aes(ymin = y - 0.45,
                        ymax = y + 0.45), xmin = -Inf, xmax = Inf,
                        fill = PANEL, inherit.aes = FALSE)

pick <- function(m, dz) s[s$model == m & s$design == dz, ]
clu  <- DESIGNS[["clustered_random"]]
cvc  <- pick("DBTARF", clu); spc <- pick("DBTARF_spcal", clu)

p_a <- ggplot() +
  band_layer +
  geom_vline(xintercept = 0.90, colour = INK, linewidth = 0.5, linetype = "22") +
  annotate("text", x = 0.903, y = YL[2] - 0.05, label = "nominal 0.90",
           hjust = 0, vjust = 1, size = pt2mm(PT_SMALL), family = FONT,
           colour = INK, fontface = "bold") +
  geom_point(data = d, aes(picp, yj, colour = design), size = 0.9, alpha = 0.4,
             shape = 16) +
  geom_errorbar(data = s, aes(xmin = lo, xmax = hi, y = y, colour = design),
                width = 0, linewidth = 1.2, orientation = "y") +
  geom_point(data = s, aes(picp, y, fill = design), shape = 21, size = 2.4,
             colour = "white", stroke = 0.6) +
  annotate("text", x = 0.928, y = cvc$y,
           label = sprintf("%d/30 below", cvc$below), hjust = 0,
           size = pt2mm(PT_SMALL - 1), family = FONT, colour = COL_CLUS,
           fontface = "bold") +
  scale_colour_manual(values = DCOL, aesthetics = c("colour", "fill")) +
  guides(colour = guide_legend(override.aes = list(size = 2.4, alpha = 1,
                                                   linewidth = 0)),
         fill = "none") +
  scale_x_continuous(breaks = seq(0.75, 1, 0.05),
                     labels = c(".75", ".80", ".85", ".90", ".95", "1")) +
  yscale +
  coord_cartesian(xlim = c(0.74, 1.0), ylim = YL, expand = FALSE, clip = "off") +
  labs(title = "a  Coverage (PICP)", x = "Share of validation points inside their interval",
       y = NULL) +
  theme_paper() +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = PT_SMALL, colour = INK, hjust = 0))

p_b <- ggplot() +
  band_layer +
  geom_point(data = d, aes(mpiw, yj, colour = design), size = 0.9, alpha = 0.4,
             shape = 16) +
  geom_point(data = s, aes(mpiw, y, fill = design), shape = 21, size = 2.4,
             colour = "white", stroke = 0.6) +
  annotate("text", x = 103, y = spc$y,
           label = sprintf("+%.0f%%\nvs CV+", 100 * (spc$mpiw / cvc$mpiw - 1)),
           hjust = 1, size = pt2mm(PT_SMALL - 1), family = FONT,
           colour = COL_CLUS, fontface = "bold", lineheight = 0.9) +
  scale_colour_manual(values = DCOL, aesthetics = c("colour", "fill"),
                      guide = "none") +
  scale_x_continuous(breaks = seq(0, 200, 50)) +
  yscale +
  coord_cartesian(xlim = c(0, 230), ylim = YL, expand = FALSE) +
  labs(title = "b  Width (MPIW)", x = expression("Mg ha"^-1), y = NULL) +
  theme_paper() +
  theme(panel.grid.major.y = element_blank(), axis.text.y = element_blank())

fig <- (p_a | p_b) +
  plot_layout(widths = c(1.65, 1), guides = "collect") +
  plot_annotation(
    title = "Prediction intervals judged by an independent probability sample",
    subtitle = paste0("Nominal 0.90 intervals; 30 repetitions per design, each scored on its own simple random\n",
                      "sample of 500 cells. Small dots: repetitions; large dot and bar: mean and 95% CI."),
    theme = theme_paper()) &
  theme(legend.position = "bottom", legend.justification = "center",
        legend.text = element_text(size = PT_SMALL))

save_fig(fig, out_png, height_mm = 118)
