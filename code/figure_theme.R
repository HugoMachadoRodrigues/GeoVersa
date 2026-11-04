# =============================================================================
# figure_theme.R
#
# One visual system for every manuscript figure, sourced by each generator.
#
# Sizes are set at the size the figure is printed, not at an arbitrary pixel
# canvas. The manuscript's text block is 390 pt (137.6 mm) wide and every figure
# is included at \textwidth, so a figure saved at FIG_W mm wide prints at 1:1
# and a 10 pt label here is a 10 pt label on the page. The journal's
# double-column width is 174 mm, so in print the text grows by about a quarter
# rather than shrinking. The previous figures were drawn on canvases of 9 to 13
# inches and scaled down to the text block, which is why their labels came out
# at 6 to 7 pt.
#
# Colour carries one meaning across the whole paper:
#   * calibration design  -> categorical hue (simple random blue, systematic
#     aqua, clustered orange; the narrower clustered footprint is a darker step
#     of the same orange, since it is the same design with fewer units)
#   * method              -> TARF violet, the default forest grey
#   * magnitude           -> one green ramp for biomass
#   * sign                -> blue/red diverging around a grey midpoint
# The design palette passes the colour-vision-deficiency checks for all pairs
# (worst Delta E 9.2); aqua is below 3:1 against white, so every design is also
# named in a direct label or a legend, never by colour alone.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

FIG_W  <- 137.6                 # mm, \textwidth
FONT   <- "Helvetica"

# ---- ink and chrome ---------------------------------------------------------
INK    <- "#0b0b0b"
INK2   <- "#52514e"
MUTED  <- "#898781"
COL_GRID <- "#e1e0d9"
AXIS   <- "#c3c2b7"
PANEL  <- "#f6f5f1"             # quiet band behind annotations

# ---- categorical: calibration designs ---------------------------------------
COL_SRS   <- "#2a78d6"
COL_SYS   <- "#1baf7a"
COL_CLUS  <- "#eb6834"
COL_CLUSW <- "#9c3a12"

# ---- methods ----------------------------------------------------------------
COL_RF    <- "#9a988f"
COL_TARF  <- "#6250d6"

# ---- diverging: blue (better / below) <-> red (worse / above) ---------------
DIV_LO  <- "#2a78d6"
DIV_MID <- "#f0efec"
DIV_HI  <- "#d03b3b"

# ---- sequential: biomass ----------------------------------------------------
SEQ_GREEN <- c("#f4f1e4", "#d7e3b5", "#a9cc86", "#6faa5c", "#3d8545",
               "#1d6337", "#0b4026")

# Text sizes in points, at print size.
PT_BASE  <- 10
PT_SMALL <- 9
PT_TITLE <- 11
pt2mm <- function(pt) pt / ggplot2::.pt   # geom_text sizes are in mm

theme_paper <- function(base = PT_BASE) {
  theme_minimal(base_size = base, base_family = FONT) +
    theme(
      text              = element_text(colour = INK),
      plot.title        = element_text(face = "bold", size = PT_TITLE,
                                       margin = margin(0, 0, 2, 0)),
      plot.subtitle     = element_text(colour = INK2, size = PT_SMALL,
                                       lineheight = 1.1,
                                       margin = margin(0, 0, 6, 0)),
      plot.title.position = "plot",
      plot.caption      = element_text(colour = INK2, size = PT_SMALL - 1,
                                       hjust = 0, lineheight = 1.1),
      plot.caption.position = "plot",
      axis.title        = element_text(size = base, colour = INK2),
      axis.text         = element_text(size = PT_SMALL, colour = INK2),
      axis.line.x       = element_line(colour = AXIS, linewidth = 0.4),
      axis.ticks.x      = element_line(colour = AXIS, linewidth = 0.4),
      axis.ticks.length = unit(2, "pt"),
      panel.grid.major  = element_line(colour = COL_GRID, linewidth = 0.3),
      panel.grid.minor  = element_blank(),
      strip.text        = element_text(face = "bold", size = base,
                                       hjust = 0, colour = INK),
      legend.title      = element_blank(),
      legend.text       = element_text(size = PT_SMALL, colour = INK),
      legend.key.size   = unit(10, "pt"),
      legend.position   = "bottom",
      legend.margin     = margin(0, 0, 0, 0),
      plot.margin       = margin(4, 6, 4, 4)
    )
}

# Panel tag in the corner, the way the captions refer to panels.
tag_theme <- theme(plot.tag = element_text(face = "bold", size = PT_TITLE + 1,
                                           family = FONT, colour = INK),
                   plot.tag.position = c(0, 1))

save_fig <- function(plot, file, height_mm, width_mm = FIG_W, dpi = 600) {
  ggplot2::ggsave(file, plot, device = ragg::agg_png, width = width_mm,
                  height = height_mm, units = "mm", dpi = dpi, bg = "white")
  message(sprintf("[fig] wrote %s (%.0f x %.0f mm, %d dpi)", file, width_mm,
                  height_mm, dpi))
}
