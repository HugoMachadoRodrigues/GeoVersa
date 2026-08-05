# =============================================================================
# maps_plot.R  —  STEP 2 of 2 (cheap; safe to re-run while tuning the design)
#
# Reads results/maps_v030/map_predictions.rds (written by maps_compute.R) and
# produces the manuscript map figures.
#
# Design rules enforced here:
#   * the three value maps (truth, RF, DB-TARF) share ONE common colour scale,
#     so the panels are visually comparable;
#   * the two error maps share ONE common, SYMMETRIC diverging scale centred on
#     zero, so over- and under-prediction are equally legible and the two models
#     are directly comparable;
#   * earth-toned, nature-like palettes throughout (sand -> olive -> forest for
#     biomass; brown <-> green for signed error), colour-blind checked by using
#     perceptually ordered ramps.
#
# Usage:  Rscript code/maps_plot.R
# =============================================================================

# Portable root resolution; see code/project_paths.R. Works whether the script
# is run from the repository root or from inside code/.
local({
  cands <- c("code/project_paths.R", "project_paths.R", "../code/project_paths.R")
  hit <- cands[file.exists(cands)]
  if (!length(hit)) stop("project_paths.R not found; run from the repo root")
  source(hit[1L])
})
setwd(gv_root())

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(scales)
})

in_rds  <- "results/maps_v030/map_predictions.rds"
fig_dir <- "Text/ecomod_submission/figures"
stopifnot(file.exists(in_rds))
obj <- readRDS(in_rds)
d   <- obj$maps
cal <- obj$cal
meta <- obj$meta

message(sprintf("[plot] %d cells | selected mode = %s, lambda = %s",
                nrow(d), meta$selected_mode, format(meta$selected_lambda)))

# ---- palettes ---------------------------------------------------------------
# Sequential, biomass: bare sand -> dry grass -> olive -> canopy -> deep forest.
pal_biomass <- c("#F2EAD6", "#DCCFA4", "#BDBC7E", "#94A96A", "#668B5A",
                 "#3F6B48", "#26492F", "#16301F")

# Diverging, signed error: eroded earth (under) <-> living canopy (over),
# with a warm neutral centre rather than pure white.
pal_error <- c("#5A3210", "#8C5A24", "#BE9058", "#E3D3B4", "#F6F1E4",
               "#C3D3B4", "#8DAF83", "#4F7E58", "#26492F")

# ---- shared scales ----------------------------------------------------------

val_lim <- range(c(d$truth, d$rf, d$dbtarf), na.rm = TRUE)
message(sprintf("[plot] shared value scale: %.1f - %.1f Mg/ha", val_lim[1], val_lim[2]))

# robust symmetric limit for the error panels (99th percentile of |error|),
# with out-of-bounds values squished to the extremes rather than dropped
err_all <- c(d$err_rf, d$err_dbtarf)
err_lim <- stats::quantile(abs(err_all), 0.99, na.rm = TRUE)
err_lim <- c(-err_lim, err_lim)
message(sprintf("[plot] shared error scale: %+.1f / %+.1f Mg/ha (99th pct)",
                err_lim[1], err_lim[2]))

# ---- theme ------------------------------------------------------------------

theme_map <- function() {
  theme_void(base_size = 14) +
    theme(
      plot.title    = element_text(face = "bold", size = 14, hjust = 0,
                                   margin = margin(b = 2)),
      plot.subtitle = element_text(size = 11, colour = "grey30", hjust = 0,
                                   margin = margin(b = 4)),
      legend.position  = "right",
      legend.key.width = unit(0.32, "cm"),
      legend.key.height= unit(1.15, "cm"),
      legend.title     = element_text(size = 11),
      legend.text      = element_text(size = 10),
      plot.margin      = margin(4, 6, 4, 6)
    )
}

map_panel <- function(fill, title, subtitle, diverging = FALSE) {
  p <- ggplot(d, aes(x = x, y = y, fill = .data[[fill]])) +
    geom_raster() +
    coord_equal(expand = FALSE) +
    labs(title = title, subtitle = subtitle) +
    theme_map()
  if (diverging) {
    p + scale_fill_gradientn(
      colours = pal_error, limits = err_lim, oob = scales::squish,
      name = expression(atop("error", "(Mg ha"^-1 * ")")))
  } else {
    p + scale_fill_gradientn(
      colours = pal_biomass, limits = val_lim, oob = scales::squish,
      name = expression(atop("AGB", "(Mg ha"^-1 * ")")))
  }
}

rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))

# ---- Figure: value maps + error maps ----------------------------------------

p_truth <- map_panel("truth", "a  Observed AGB",
                     "Baccini et al. (2012), 1 km") +
  geom_point(data = cal, aes(x = x, y = y), inherit.aes = FALSE,
             colour = "#B03A2E", fill = NA, shape = 21, size = 0.45, stroke = 0.25)
p_truth <- p_truth + labs(subtitle = sprintf(
  "Baccini et al. (2012), 1 km; red = %d calibration points", nrow(cal)))

p_rf <- map_panel("rf", "b  RF (Wadoux default)",
                  sprintf("population RMSE %.2f Mg ha⁻¹", rmse(d$err_rf)))
p_db <- map_panel("dbtarf", "c  DB-TARF",
                  sprintf("population RMSE %.2f Mg ha⁻¹", rmse(d$err_dbtarf)))

p_erf <- map_panel("err_rf", "d  RF error", "prediction - truth", diverging = TRUE)
p_edb <- map_panel("err_dbtarf", "e  DB-TARF error", "prediction - truth", diverging = TRUE)

fig <- (p_truth | p_rf | p_db) / (plot_spacer() | p_erf | p_edb) +
  plot_layout(guides = "collect", heights = c(1, 1)) &
  theme(legend.position = "right")

out <- file.path(fig_dir, "fig7_maps.png")
ggsave(out, fig, width = 13.2, height = 8.0, dpi = 220, bg = "white")
message(sprintf("[plot] wrote %s", out))

# ---- console: population-level numbers for the caption ----------------------

cat("\n========== whole-grid (population) numbers ==========\n")
cat(sprintf("  RF      RMSE %.3f   ME %+.3f\n", rmse(d$err_rf), mean(d$err_rf, na.rm = TRUE)))
cat(sprintf("  DB-TARF RMSE %.3f   ME %+.3f\n", rmse(d$err_dbtarf), mean(d$err_dbtarf, na.rm = TRUE)))
cat(sprintf("  delta RMSE %+.3f\n", rmse(d$err_dbtarf) - rmse(d$err_rf)))
cat(sprintf("  shared value scale : %.1f - %.1f\n", val_lim[1], val_lim[2]))
cat(sprintf("  shared error scale : %+.1f / %+.1f\n", err_lim[1], err_lim[2]))
