# =============================================================================
# figures_graphical_abstract.R
#
# The graphical abstract. Ecological Modelling asks for 531 x 1328 pixels
# (h x w) or proportionally more, readable at 5 x 13 cm. It is drawn at
# exactly 130 x 52 mm, so the text sizes below are the sizes a reader sees at
# that scale, and saved at 600 dpi (3071 x 1228 px).
#
# One message: coverage is a design-based quantity, and once it is measured
# that way the shortfall turns out to be indexed by distance from the
# calibration sample, with designs that differ in marginal coverage covering
# almost identically at matched distance. Every number is read from the
# campaign outputs.
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
source("code/figure_theme.R")
out_png <- "Text/ecomod_submission/figures/graphical_abstract.png"

DES <- list(
  list(dir = "pts_random",    lab = "Random",            col = COL_SRS),
  list(dir = "pts_clus_mid",  lab = "Clustered 20 × 25", col = COL_CLUS),
  list(dir = "pts_clus_wide", lab = "Clustered 8 × 25",  col = COL_CLUSW)
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
  tab$design <- d$lab
  list(tab = tab, cov = mean(cov), lab = d$lab)
}
D <- lapply(DES, read_one)
curves <- do.call(rbind, lapply(D, `[[`, "tab"))
LEV <- vapply(DES, `[[`, "", "lab")
PAL <- stats::setNames(vapply(DES, `[[`, "", "col"), LEV)
curves$design <- factor(curves$design, levels = LEV)

# The repair's range over every cell of the sensitivity table.
blk <- unlist(lapply(c("clus_dens_n200", "clus_mid_n500", "clus_dens_n800",
                       "clus_spread_n200", "clus_spread_n800",
                       "final_random_confirm30", "rand_n200", "rand_n800"),
  function(d) {
    u <- read.csv(gv_results(d, "dbtarf_uncertainty.csv"))
    mean(u$PICP90[u$model == "DBTARF_spcal"])
  }))

S <- 8.5                                           # pt, readable at 13 cm
# ---- left: the claim -------------------------------------------------------
eq <- "widehat(PICP) == frac(1, N) * sum(frac(1, pi[k]) * 1 * group('{', y[k] %in% PI[k], '}'), k %in% S, '')"
p_l <- ggplot() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  annotate("text", x = 0, y = 0.97, hjust = 0, vjust = 1,
           label = "Interval coverage is a\ndesign-based quantity",
           size = pt2mm(S + 2.5), family = FONT, fontface = "bold",
           colour = INK, lineheight = 0.95) +
  annotate("text", x = 0, y = 0.555, hjust = 0, label = eq, parse = TRUE,
           size = pt2mm(S), family = FONT, colour = COL_TARF) +
  annotate("text", x = 0, y = 0.40, hjust = 0, vjust = 1,
           label = "Estimate it on an independent\nprobability sample, exactly as\nmap accuracy is estimated.",
           size = pt2mm(S), family = FONT, colour = INK2, lineheight = 1.0) +
  annotate("text", x = 0, y = 0.02, hjust = 0, vjust = 0,
           label = "Amazon biomass, 1.31 M cells, nominal 0.90",
           size = pt2mm(S - 1.5), family = FONT, colour = MUTED) +
  theme_void()

# ---- right: coverage against distance --------------------------------------
lab_end <- do.call(rbind, lapply(split(curves, curves$design), function(z)
  z[which.max(z$d), ]))
lab_end$txt <- sprintf("%s: %.3f", lab_end$design,
                       vapply(D, `[[`, 0, "cov")[match(lab_end$design, LEV)])
p_r <- ggplot(curves, aes(d, p, colour = design)) +
  geom_hline(yintercept = 0.90, colour = INK, linewidth = 0.45, linetype = "22") +
  annotate("text", x = 480, y = 0.905, label = "nominal 0.90", hjust = 1,
           vjust = 0, size = pt2mm(S - 1), family = FONT, colour = INK,
           fontface = "bold") +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5, shape = 21, fill = "white", stroke = 0.8) +
  scale_colour_manual(values = PAL) +
  scale_x_log10(breaks = c(3, 10, 30, 100, 300), limits = c(3, 500),
                name = "km to the nearest calibration point") +
  scale_y_continuous(limits = c(0.74, 0.98), breaks = c(0.75, 0.85, 0.95),
                     labels = c(".75", ".85", ".95"), name = "Coverage") +
  labs(title = "Covered where the sample is, not where it is not",
       subtitle = sprintf("Three calibration designs, one curve. A spatial-block\ncalibration holds %.2f–%.2f in every design tested.",
                          min(blk), max(blk))) +
  theme_paper(base = S) +
  theme(plot.title = element_text(size = S + 1, face = "bold"),
        plot.subtitle = element_text(size = S - 0.5),
        axis.text = element_text(size = S - 1),
        axis.title = element_text(size = S - 0.5),
        legend.position = "bottom",
        legend.text = element_text(size = S - 1),
        legend.key.size = unit(7, "pt"), legend.justification = "right",
        plot.margin = margin(2, 2, 2, 2))

fig <- (wrap_elements(full = p_l) | p_r) + plot_layout(widths = c(0.8, 1.2))
save_fig(fig, out_png, height_mm = 52, width_mm = 130)
