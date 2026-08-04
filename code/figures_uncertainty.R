# =============================================================================
# figures_uncertainty.R
#
# The interval-assessment figure. This script did not exist, which is how the
# figure came to disagree with the coverage table: the table was regenerated
# after the campaigns were re-run and the figure, having no generator, was not.
# It now reads exactly the files tables_manuscript.R reads, so the two cannot
# drift apart again.
#
# Usage:  Rscript code/figures_uncertainty.R
# =============================================================================

root <- "/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/GeoVersa"
setwd(root)
suppressPackageStartupMessages({ library(ggplot2); library(patchwork) })

out_png <- "Text/ecomod_submission/figures/fig6_uncertainty.png"
dir_for <- function(sc) sprintf("results/rev1_%s_confirm30", sc)

DESIGNS <- c(random = "Simple random", regular_grid = "Systematic",
             clustered_random = "Clustered")
LABELS  <- c(DBTARF = "Conformal", DBTARF_dbcal = "Covariate-shift weighted",
             DBTARF_spcal = "Spatial-block", RF_quantile = "Quantile RF")
PAL <- c("Conformal" = "#668B5A", "Covariate-shift weighted" = "#BE9058",
         "Spatial-block" = "#26492F", "Quantile RF" = "#8C5A24")

d <- do.call(rbind, lapply(names(DESIGNS), function(sc) {
  u <- read.csv(file.path(dir_for(sc), "dbtarf_uncertainty_summary.csv"))
  u <- u[u$model %in% names(LABELS), ]
  data.frame(design = factor(DESIGNS[[sc]], levels = unname(DESIGNS)),
             interval = factor(LABELS[u$model], levels = unname(LABELS)),
             picp = u$PICP90_mean, mpiw = u$MPIW90_mean,
             stringsAsFactors = FALSE)
}))

message(sprintf("[uq] %d rows from %s", nrow(d), paste(dir_for(names(DESIGNS)), collapse = ", ")))

theme_uq <- function() {
  theme_minimal(base_size = 12) +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 12),
          axis.text.x = element_text(size = 9),
          legend.position = "bottom", legend.title = element_blank())
}

p_cov <- ggplot(d, aes(design, picp, fill = interval)) +
  geom_col(position = position_dodge(0.8), width = 0.72) +
  geom_hline(yintercept = 0.90, linetype = "dashed", colour = "#B03A2E",
             linewidth = 0.7) +
  geom_text(aes(label = sprintf("%.3f", picp)),
            position = position_dodge(0.8), vjust = -0.5, size = 2.7) +
  annotate("text", x = 0.5, y = 0.988, label = "dashed line: nominal 0.90",
           hjust = 0, size = 3, colour = "#B03A2E", fontface = "bold") +
  scale_fill_manual(values = PAL) +
  coord_cartesian(ylim = c(0.78, 1.0)) +
  labs(title = "a  Coverage", x = NULL, y = "PICP") +
  theme_uq()

p_wid <- ggplot(d, aes(design, mpiw, fill = interval)) +
  geom_col(position = position_dodge(0.8), width = 0.72) +
  geom_text(aes(label = sprintf("%.0f", mpiw)),
            position = position_dodge(0.8), vjust = -0.5, size = 2.7) +
  scale_fill_manual(values = PAL) +
  coord_cartesian(ylim = c(0, max(d$mpiw) * 1.15)) +
  labs(title = "b  Mean interval width", x = NULL,
       y = expression("MPIW (Mg ha"^-1 * ")")) +
  theme_uq()

fig <- (p_cov | p_wid) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(out_png, fig, width = 10.4, height = 4.6, dpi = 300, bg = "white")
cat(sprintf("wrote %s\n", out_png))
print(d[order(d$design, d$interval), ], row.names = FALSE)
