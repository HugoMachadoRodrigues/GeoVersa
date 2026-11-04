# =============================================================================
# figures_protocols.R
#
# Figure 3: the paired difference in RMSE between TARF and the default forest
# under four estimands, one dot per repetition.
#
# An earlier version drew two bars per estimand. Bars on an RMSE axis starting
# at zero spend most of their height on the 30 Mg/ha both models share, show
# the difference only as a sliver, and hide the spread between repetitions,
# which is what tells a reader whether ten draws are enough. Plotting the
# paired difference itself puts the quantity being compared on the axis.
#
# Inputs:
#   final_random_confirm30           design-based campaign, 30 repetitions
#   canon_v030_protocols_confirm10   the 20,000-cell sample and the two
#                                    cross-validation protocols, 10 repetitions
#
# Usage:  Rscript code/figures_protocols.R
# =============================================================================

local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
setwd(gv_root())
source("code/figure_theme.R")

out_png <- "Text/ecomod_submission/figures/fig4_protocols.png"

# ---- design-based campaign (30 reps) ----------------------------------------
db_dir <- gv_results("final_random_confirm30")
db_cfg <- read.csv(file.path(db_dir, "dbtarf_config.csv"), stringsAsFactors = FALSE)
if (!"git_commit" %in% names(db_cfg) || !nzchar(as.character(db_cfg$git_commit[1]))) {
  stop("campaign without provenance: ", db_dir)
}
pd <- read.csv(file.path(db_dir, "dbtarf_paired_deltas_vs_RF.csv"),
               stringsAsFactors = FALSE)
pd <- pd[pd$protocol == "DesignBased", ]

# ---- the other three estimands (10 reps) ------------------------------------
pr <- read.csv(gv_results("canon_v030_protocols_confirm10",
                          "dbtarf_protocols_all_results.csv"),
               stringsAsFactors = FALSE)

ROWS <- c(Population   = "Independent SRS,\n20,000 cells",
          DesignBased  = "Independent SRS,\n500 cells",
          RandomKFold  = "Random 10-fold\ncross-validation",
          SpatialKFold = "Spatial block\ncross-validation")
GROUP <- c(Population = "map", DesignBased = "map",
           RandomKFold = "cv", SpatialKFold = "cv")

# Per-repetition deltas recomputed from the two RMSE columns rather than read
# from a rounded delta column.
pts <- rbind(
  data.frame(est = "DesignBased", delta = pd$RMSE_DBTARF - pd$RMSE_RF),
  do.call(rbind, lapply(c("Population", "RandomKFold", "SpatialKFold"),
    function(p) { z <- pr[pr$protocol == p, ]
      data.frame(est = p, delta = z$RMSE_DBTARF - z$RMSE_RF) })))
pts$row <- factor(ROWS[pts$est], levels = rev(unname(ROWS)))
pts$grp <- GROUP[pts$est]

smry <- do.call(rbind, lapply(names(ROWS), function(e) {
  x <- pts$delta[pts$est == e]
  tt <- stats::t.test(x)
  data.frame(est = e, row = ROWS[[e]], grp = GROUP[[e]], n = length(x),
             mean = mean(x), lo = tt$conf.int[1], hi = tt$conf.int[2],
             wins = sum(x < 0))
}))
smry$row <- factor(smry$row, levels = levels(pts$row))
print(smry[, c("est", "n", "mean", "lo", "hi", "wins")], row.names = FALSE)

COLS <- c(map = COL_TARF, cv = MUTED)
# Row positions with a gap between the two groups, so each group can carry a
# header without touching the dots of the row beneath it.
YPOS <- c(SpatialKFold = 1, RandomKFold = 2, DesignBased = 3.55, Population = 4.55)
set.seed(7)
pts$y  <- YPOS[pts$est] + stats::runif(nrow(pts), -0.17, 0.17)
smry$y <- YPOS[smry$est]
umin <- function(x) gsub("-", "\u2212", x, fixed = TRUE)

XL <- c(-2.6, 0.55)
p <- ggplot() +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 3.1, ymax = 5.0, fill = PANEL) +
  annotate("text", x = XL[1], y = 5.12, hjust = 0, vjust = 0,
           label = "Map accuracy: independent probability samples",
           size = pt2mm(PT_SMALL), family = FONT, fontface = "bold",
           colour = COL_TARF) +
  annotate("text", x = XL[1], y = 2.4, hjust = 0, vjust = 0,
           label = "Model comparison only: not map accuracy",
           size = pt2mm(PT_SMALL), family = FONT, fontface = "bold",
           colour = INK2) +
  geom_vline(xintercept = 0, colour = INK2, linewidth = 0.4) +
  annotate("text", x = 0.04, y = 0.45, hjust = 0, vjust = 0,
           label = "no difference", size = pt2mm(PT_SMALL - 1),
           family = FONT, colour = INK2) +
  annotate("text", x = -0.06, y = 0.45, hjust = 1, vjust = 0,
           label = "\u2190 TARF more accurate", size = pt2mm(PT_SMALL - 1),
           family = FONT, colour = COL_TARF, fontface = "bold") +
  geom_point(data = pts, aes(delta, y, colour = grp), size = 1.5, alpha = 0.55,
             shape = 16) +
  geom_errorbar(data = smry, aes(xmin = lo, xmax = hi, y = y), width = 0,
                linewidth = 1.1, colour = INK, orientation = "y") +
  geom_point(data = smry, aes(mean, y), shape = 23, size = 3, fill = "white",
             colour = INK, stroke = 0.9) +
  # summary column outside the panel, so it never sits on a dot
  annotate("text", x = 0.72, y = 5.12, hjust = 0, vjust = 0, label = "Mean,\nwins",
           size = pt2mm(PT_SMALL), family = FONT, fontface = "bold", colour = INK) +
  geom_text(data = smry, aes(x = 0.72, y = y,
            label = umin(sprintf("%+.2f\n%d of %d", mean, wins, n))),
            hjust = 0, size = pt2mm(PT_SMALL), family = FONT, colour = INK,
            lineheight = 0.95) +
  scale_colour_manual(values = COLS, guide = "none") +
  scale_y_continuous(breaks = unname(YPOS[names(ROWS)]), labels = unname(ROWS)) +
  scale_x_continuous(breaks = seq(-2.5, 0.5, 0.5),
                     labels = function(b) ifelse(b == 0, "0", umin(sprintf("%+.1f", b)))) +
  labs(title = "TARF against the default forest under four estimands",
       subtitle = paste0("Paired RMSE difference per repetition (dots), with mean and 95%\n",
                         "confidence interval (diamond, bar). Simple random calibration, n = 500.\n",
                         "Cross-validation trains on subsets of the sample, so its rows are not\n",
                         "on the same footing as the two map-accuracy estimates."),
       x = expression("Difference in RMSE, TARF " - " RF (Mg ha"^-1 * ")"),
       y = NULL) +
  coord_cartesian(xlim = XL, ylim = c(0.4, 5.45), expand = FALSE, clip = "off") +
  theme_paper() +
  theme(panel.grid.major.y = element_blank(),
        plot.margin = margin(4, 52, 4, 4),
        axis.text.y = element_text(size = PT_SMALL, colour = INK, hjust = 0,
                                   lineheight = 0.95))

save_fig(p, out_png, height_mm = 104)
