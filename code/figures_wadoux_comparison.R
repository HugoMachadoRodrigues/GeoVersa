# =============================================================================
# figures_wadoux_comparison.R
#
# Narrative: ConvKrigingNet2D vs RF using the Wadoux (2021) validation framework.
#
# This script compares only locally generated benchmark runs executed under the
# same protocol implementation. No hardcoded "published paper" metric table is
# used here.
#
# Protocols: DesignBased, RandomKFold, SpatialKFold
# (BLOOCV: 1000 model trainings/iter — computationally infeasible for deep learning)
# (Population: ~66 GB RAM required for patch array — hardware limitation)
# =============================================================================

rm(list = ls())

pkgs <- c("ggplot2", "dplyr", "tidyr", "scales", "patchwork")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install, repos = "https://cloud.r-project.org")

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)

# =============================================================================
# Paths
# =============================================================================
dir_conv <- "results/wadoux2021_conv_random_validation_n500"
dir_rf   <- "results/wadoux2021_rf_random_validation"
dir_fig  <- "figures/wadoux_comparison"
dir.create(dir_fig, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Load our benchmark runs
# =============================================================================
conv_file <- file.path(dir_conv, "wadoux_style_rf_conv_all_results.csv")
rf_file   <- file.path(dir_rf,   "wadoux_style_rf_conv_all_results.csv")

if (!file.exists(conv_file)) stop("ConvKrigingNet2D results not found. Run Benchmark_Unified_ConvKriging2DNet_RF.R first.")
if (!file.exists(rf_file))   stop("RF results not found. Run Benchmark_RF_Wadoux_Style.R first.")

conv_raw <- read.csv(conv_file, stringsAsFactors = FALSE)
rf_raw   <- read.csv(rf_file,   stringsAsFactors = FALSE)

# Summarise our runs: mean +/- sd across iterations
our_summary <- bind_rows(conv_raw, rf_raw) %>%
  group_by(protocol, model) %>%
  summarise(
    ME_mean   = mean(ME,   na.rm = TRUE),
    RMSE_mean = mean(RMSE, na.rm = TRUE),
    r2_mean   = mean(r2,   na.rm = TRUE),
    MEC_mean  = mean(MEC,  na.rm = TRUE),
    ME_sd     = sd(ME,     na.rm = TRUE),
    RMSE_sd   = sd(RMSE,   na.rm = TRUE),
    r2_sd     = sd(r2,     na.rm = TRUE),
    MEC_sd    = sd(MEC,    na.rm = TRUE),
    n_iter    = n(),
    .groups   = "drop"
  ) %>%
  mutate(
    source     = "Our benchmark\n(n=500, 3 iter)",
    data_label = case_when(
      model == "ConvKrigingNet2D" ~ "ConvKrigingNet2D\n(this study)",
      model == "RF"               ~ "RF\n(our run, n=500, 3 iter)"
    )
  )

# =============================================================================
# Combine
# =============================================================================
combined <- our_summary %>%
  select(protocol, model, source, data_label,
         ME_mean, RMSE_mean, r2_mean, MEC_mean,
         ME_sd, RMSE_sd, r2_sd, MEC_sd, n_iter)

protocol_labels <- c(
  DesignBased  = "Design-based\n(unbiased estimator)",
  RandomKFold  = "Random k-fold CV\n(slightly optimistic)",
  SpatialKFold = "Spatial k-fold CV\n(systematically pessimistic)"
)

combined <- combined %>%
  mutate(
    protocol_label = factor(
      recode(protocol, !!!protocol_labels),
      levels = unname(protocol_labels)
    )
  )

# Factor levels: ConvKrigingNet2D first
model_levels <- c(
  "ConvKrigingNet2D\n(this study)",
  "RF\n(our run, n=500, 3 iter)"
)
combined$data_label <- factor(combined$data_label, levels = model_levels)

# =============================================================================
# Colours and shapes
# =============================================================================
palette <- c(
  "ConvKrigingNet2D\n(this study)"            = "#2166AC",
  "RF\n(our run, n=500, 3 iter)"              = "#4DAC26"
)
shape_map <- c(
  "ConvKrigingNet2D\n(this study)"            = 16,
  "RF\n(our run, n=500, 3 iter)"              = 17
)
linetype_map <- c(
  "ConvKrigingNet2D\n(this study)"            = "solid",
  "RF\n(our run, n=500, 3 iter)"              = "solid"
)

# =============================================================================
# Theme
# =============================================================================
theme_paper <- function(base = 11) {
  theme_bw(base_size = base) +
    theme(
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(colour = "grey92"),
      strip.background  = element_rect(fill = "grey95", colour = "grey70"),
      strip.text        = element_text(face = "bold", size = base - 1),
      legend.position   = "bottom",
      legend.title      = element_blank(),
      legend.key.width  = unit(1.8, "lines"),
      legend.text       = element_text(size = base - 2),
      axis.text         = element_text(size = base - 2),
      axis.title        = element_text(size = base - 1),
      plot.title        = element_text(face = "bold", size = base + 1),
      plot.subtitle     = element_text(size = base - 2, colour = "grey35"),
      plot.caption      = element_text(size = base - 3, colour = "grey50", hjust = 0)
    )
}

save_fig <- function(p, name, w = 8, h = 5) {
  ggsave(file.path(dir_fig, paste0(name, ".pdf")), p, width = w, height = h)
  ggsave(file.path(dir_fig, paste0(name, ".png")), p, width = w, height = h,
         dpi = 300, bg = "white")
  cat(sprintf("  Saved: figures/wadoux_comparison/%s.pdf/png\n", name))
}

# =============================================================================
# FIGURE 1 (MAIN) — RMSE across protocols
# Key message: compare the local ConvKrigingNet2D and RF runs under the same
# protocol implementation.
# =============================================================================

# Annotation: arrow/label for design-based equivalence
design_data <- combined %>% filter(protocol == "DesignBased")

p_rmse_main <- ggplot(combined,
  aes(x = protocol_label, y = RMSE_mean,
      colour = data_label, shape = data_label,
      linetype = data_label, group = data_label)) +
  # Connecting lines between protocols
  geom_line(linewidth = 0.7, position = position_dodge(width = 0.4)) +
  # Error bars (our runs only)
  geom_errorbar(
    aes(ymin = RMSE_mean - RMSE_sd, ymax = RMSE_mean + RMSE_sd),
    width = 0.12, na.rm = TRUE, linewidth = 0.6,
    position = position_dodge(width = 0.4)
  ) +
  geom_point(size = 3.8, position = position_dodge(width = 0.4)) +
  # Label values on points
  geom_text(
    aes(label = sprintf("%.1f", RMSE_mean)),
    vjust = -1.0, size = 3.0, fontface = "bold", show.legend = FALSE,
    position = position_dodge(width = 0.4)
  ) +
  scale_colour_manual(values = palette, drop = FALSE) +
  scale_shape_manual(values = shape_map, drop = FALSE) +
  scale_linetype_manual(values = linetype_map, drop = FALSE) +
  scale_y_continuous(
    limits = c(28, 44),
    breaks = seq(28, 44, by = 2),
    name = "RMSE (Mg/ha)"
  ) +
  labs(
    x = NULL,
    title = "Map accuracy of ConvKrigingNet2D evaluated with Wadoux et al. (2021) protocols",
    subtitle = paste0(
      "Local benchmark only: ConvKrigingNet2D vs RF under the same protocol implementation\n",
      "Design-based is the map-accuracy protocol emphasised by Wadoux et al. (2021)"
    ),
    caption = paste0(
      "Error bars: mean +/- 1 SD across 3 independent sampling iterations\n",
      "BLOOCV excluded: requires 1,000 model trainings/iteration (computationally infeasible for deep learning)"
    )
  ) +
  theme_paper()

save_fig(p_rmse_main, "fig1_rmse_main", w = 10, h = 6)

# =============================================================================
# FIGURE 2 — r2 (Pearson squared) across protocols
# =============================================================================
p_r2 <- ggplot(combined,
  aes(x = protocol_label, y = r2_mean,
      colour = data_label, shape = data_label,
      linetype = data_label, group = data_label)) +
  geom_line(linewidth = 0.7, position = position_dodge(width = 0.4)) +
  geom_errorbar(
    aes(ymin = r2_mean - r2_sd, ymax = r2_mean + r2_sd),
    width = 0.12, na.rm = TRUE, linewidth = 0.6,
    position = position_dodge(width = 0.4)
  ) +
  geom_point(size = 3.8, position = position_dodge(width = 0.4)) +
  geom_text(
    aes(label = sprintf("%.2f", r2_mean)),
    vjust = -1.0, size = 3.0, fontface = "bold", show.legend = FALSE,
    position = position_dodge(width = 0.4)
  ) +
  scale_colour_manual(values = palette, drop = FALSE) +
  scale_shape_manual(values = shape_map, drop = FALSE) +
  scale_linetype_manual(values = linetype_map, drop = FALSE) +
  scale_y_continuous(
    limits = c(0.60, 0.95),
    breaks = seq(0.60, 0.95, by = 0.05),
    name = expression(r^2~"(Pearson)")
  ) +
  labs(
    x = NULL,
    title = expression(r^2~"by validation protocol"),
    subtitle = "Local benchmark only",
    caption = "Error bars: +/- 1 SD"
  ) +
  theme_paper()

save_fig(p_r2, "fig2_r2", w = 10, h = 6)

# =============================================================================
# FIGURE 3 — ME across protocols
# =============================================================================
p_me <- ggplot(combined,
  aes(x = protocol_label, y = ME_mean,
      colour = data_label, shape = data_label,
      linetype = data_label, group = data_label)) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey40", linewidth = 0.7) +
  geom_line(linewidth = 0.7, position = position_dodge(width = 0.4)) +
  geom_errorbar(
    aes(ymin = ME_mean - ME_sd, ymax = ME_mean + ME_sd),
    width = 0.12, na.rm = TRUE, linewidth = 0.6,
    position = position_dodge(width = 0.4)
  ) +
  geom_point(size = 3.8, position = position_dodge(width = 0.4)) +
  geom_text(
    aes(label = sprintf("%.1f", ME_mean)),
    vjust = -1.0, size = 3.0, fontface = "bold", show.legend = FALSE,
    position = position_dodge(width = 0.4)
  ) +
  scale_colour_manual(values = palette, drop = FALSE) +
  scale_shape_manual(values = shape_map, drop = FALSE) +
  scale_linetype_manual(values = linetype_map, drop = FALSE) +
  labs(
    x = NULL, y = "Mean Error (Mg/ha)",
    title = "Mean Error by validation protocol",
    subtitle = "Values near zero indicate unbiased predictions",
    caption = "Dotted line: ME = 0 (no bias) | Error bars: +/- 1 SD"
  ) +
  theme_paper()

save_fig(p_me, "fig3_me", w = 10, h = 6)

# =============================================================================
# FIGURE 4 — MEC across protocols (our runs only)
# =============================================================================
mec_data <- combined %>% filter(!is.na(MEC_mean))

if (nrow(mec_data) > 0) {
  p_mec <- ggplot(mec_data,
    aes(x = protocol_label, y = MEC_mean,
        colour = data_label, shape = data_label,
        linetype = data_label, group = data_label)) +
    geom_line(linewidth = 0.7, position = position_dodge(width = 0.4)) +
    geom_errorbar(
      aes(ymin = MEC_mean - MEC_sd, ymax = MEC_mean + MEC_sd),
      width = 0.12, na.rm = TRUE, linewidth = 0.6,
      position = position_dodge(width = 0.4)
    ) +
    geom_point(size = 3.8, position = position_dodge(width = 0.4)) +
    geom_text(
      aes(label = sprintf("%.2f", MEC_mean)),
      vjust = -1.0, size = 3.0, fontface = "bold", show.legend = FALSE,
      position = position_dodge(width = 0.4)
    ) +
    scale_colour_manual(values = palette, drop = FALSE) +
    scale_shape_manual(values = shape_map, drop = FALSE) +
    scale_linetype_manual(values = linetype_map, drop = FALSE) +
    scale_y_continuous(limits = c(0.78, 0.92), breaks = seq(0.78, 0.92, by = 0.02)) +
    labs(
      x = NULL, y = "Model Efficiency Coefficient (MEC)",
      title = "MEC by validation protocol",
      subtitle = "Local benchmark only",
      caption = "MEC = 1 - SSE/SST; closer to 1 = better | Error bars: +/- 1 SD"
    ) +
    theme_paper()
  save_fig(p_mec, "fig4_mec", w = 10, h = 6)
}

# =============================================================================
# FIGURE 5 (COMPOSITE) — Main paper figure: RMSE + r2 side by side
# =============================================================================
p_composite <- (
  p_rmse_main + theme(legend.position = "none",
                      plot.title   = element_text(size = 10, face = "bold"),
                      plot.subtitle = element_text(size = 8)) |
  p_r2        + theme(legend.position = "none",
                      plot.title   = element_text(size = 10, face = "bold"),
                      plot.subtitle = element_text(size = 8))
) /
  guide_area() +
  plot_layout(heights = c(1, 0.12), guides = "collect") +
  plot_annotation(
    title    = "ConvKrigingNet2D map accuracy: Wadoux et al. (2021) validation framework",
    subtitle = paste0(
      "Local benchmark only: ConvKrigingNet2D and RF evaluated under the same protocol implementation\n",
      "Design-based remains the primary protocol for map-accuracy interpretation"
    ),
    caption  = paste0(
      "Random sampling scenario | n = 500 calibration points | Amazon basin above-ground biomass\n",
      "Error bars: mean +/- 1 SD (3 iterations)"
    ),
    theme = theme(
      plot.title    = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 9, colour = "grey35"),
      plot.caption  = element_text(size = 8, colour = "grey50", hjust = 0)
    )
  )

save_fig(p_composite, "fig5_composite_rmse_r2", w = 14, h = 7)

# =============================================================================
# FIGURE 6 — Bar chart: Design-based only
# =============================================================================
design_only <- combined %>%
  filter(protocol == "DesignBased") %>%
  mutate(label_short = case_when(
    grepl("ConvKrigingNet2D", data_label) ~ "ConvKrigingNet2D\n(this study)",
    grepl("our run", data_label)          ~ "RF\n(our run)"
  )) %>%
  mutate(label_short = factor(label_short, levels = c(
    "ConvKrigingNet2D\n(this study)", "RF\n(our run)"
  )))

p_design_bar <- ggplot(design_only,
  aes(x = label_short, y = RMSE_mean, fill = data_label)) +
  geom_col(width = 0.55, colour = "white") +
  geom_errorbar(
    aes(ymin = RMSE_mean - RMSE_sd, ymax = RMSE_mean + RMSE_sd),
    width = 0.18, na.rm = TRUE, linewidth = 0.8, colour = "grey30"
  ) +
  geom_text(
    aes(label = sprintf("%.1f\nMg/ha", RMSE_mean)),
    vjust = -0.4, size = 3.8, fontface = "bold"
  ) +
  scale_fill_manual(values = palette, guide = "none") +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.20)),
    breaks = seq(0, 40, by = 5),
    name   = "RMSE (Mg/ha)"
  ) +
  labs(
    x = NULL,
    title = "Design-based validation RMSE (Wadoux et al. 2021 framework)",
    subtitle = paste0(
      "Local benchmark only\n",
      "Compare ConvKrigingNet2D and RF on the design-based protocol"
    ),
    caption = paste0(
      "Design-based validation: probability sample of n = 500, random sampling scenario\n",
      "Error bars: +/- 1 SD across 3 iterations"
    )
  ) +
  theme_paper() +
  theme(legend.position = "none")

save_fig(p_design_bar, "fig6_design_based_bar", w = 7, h = 6)

# =============================================================================
# Export comparison table
# =============================================================================
table_out <- combined %>%
  select(protocol_label, data_label,
         ME_mean, ME_sd, RMSE_mean, RMSE_sd,
         r2_mean, r2_sd, MEC_mean, MEC_sd, n_iter) %>%
  arrange(protocol_label, data_label) %>%
  rename(
    Protocol = protocol_label,
    Model    = data_label,
    ME       = ME_mean,   ME_SD   = ME_sd,
    RMSE     = RMSE_mean, RMSE_SD = RMSE_sd,
    r2       = r2_mean,   r2_SD   = r2_sd,
    MEC      = MEC_mean,  MEC_SD  = MEC_sd,
    N_iter   = n_iter
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

write.csv(table_out, file.path(dir_fig, "comparison_table.csv"), row.names = FALSE)

# =============================================================================
# Console summary
# =============================================================================
cat("\n=======================================================\n")
cat("COMPARISON SUMMARY — Wadoux (2021) framework\n")
cat("=======================================================\n\n")
cat("KEY RESULT (Design-based = unbiased estimator):\n")
design_console <- combined %>%
  filter(protocol == "DesignBased") %>%
  select(data_label, RMSE_mean, RMSE_sd, r2_mean, r2_sd, MEC_mean) %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))
print(as.data.frame(design_console), row.names = FALSE)

cat("\nFULL TABLE (all protocols):\n")
print(as.data.frame(table_out %>% select(Protocol, Model, RMSE, RMSE_SD, r2, r2_SD)), row.names = FALSE)

cat(sprintf("\nFigures saved to: figures/wadoux_comparison/\n"))
cat("\nNARRATIVE:\n")
conv_db <- combined %>% filter(protocol == "DesignBased", grepl("ConvKriging", data_label)) %>% pull(RMSE_mean)
rf_db   <- combined %>% filter(protocol == "DesignBased", grepl("our run", data_label))      %>% pull(RMSE_mean)
conv_spk <- combined %>% filter(protocol == "SpatialKFold", grepl("ConvKriging", data_label)) %>% pull(RMSE_mean)
cat(sprintf(
  "ConvKrigingNet2D Design-based RMSE = %.1f Mg/ha\nRF (our run) Design-based RMSE     = %.1f Mg/ha\nConvKrigingNet2D SpatialKFold RMSE   = %.1f Mg/ha\n",
  conv_db, rf_db, conv_spk
))
