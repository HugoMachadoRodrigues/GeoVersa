# =============================================================================
# figures_paper.R
# Figuras científicas para o paper ConvKrigingNet2D
#
# Gera todos os painéis em:
#   figures/fig1_reference_benchmark.pdf/.png
#   figures/fig2_convkrig_vs_cubist_summary.pdf/.png
#   figures/fig3_seedwise.pdf/.png
#   figures/fig4_scatter_per_fold.pdf/.png
#   figures/fig5_violin_rmse.pdf/.png
#   figures/fig6_delta_per_seed.pdf/.png
#
# Dependências: ggplot2, dplyr, patchwork, scales, ggtext
# =============================================================================

rm(list = ls())

# Instala pacotes se necessário
pkgs <- c("ggplot2", "dplyr", "patchwork", "scales", "ggtext", "tidyr")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install, repos = "https://cloud.r-project.org")

library(ggplot2)
library(dplyr)
library(patchwork)
library(scales)
library(ggtext)
library(tidyr)

# -----------------------------------------------------------------------------
# 0) Configurações gerais
# -----------------------------------------------------------------------------

RESULTS <- "results"
OUT_DIR <- "figures"
dir.create(OUT_DIR, showWarnings = FALSE)

# Paleta de cores (colorblind-friendly, revista científica)
PALETTE <- c(
  "ConvKrigingNet2D" = "#2166AC",
  "Cubist"           = "#D6604D",
  "RF"               = "#4DAC26",
  "XGB"              = "#8073AC",
  "KrigingNet"       = "#878787"
)

SHAPES <- c(
  "ConvKrigingNet2D" = 19,
  "Cubist"           = 17,
  "RF"               = 15,
  "XGB"              = 18,
  "KrigingNet"       = 4
)

# Tema base para publicação
theme_paper <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor    = element_blank(),
      panel.grid.major.x  = element_blank(),
      strip.background    = element_rect(fill = "grey92", colour = "grey70"),
      strip.text          = element_text(face = "bold", size = base_size),
      axis.title          = element_text(size = base_size),
      axis.text           = element_text(size = base_size - 1, colour = "grey20"),
      legend.title        = element_text(face = "bold", size = base_size),
      legend.text         = element_text(size = base_size - 1),
      legend.position     = "bottom",
      legend.key.size     = unit(0.45, "cm"),
      plot.title          = element_text(face = "bold", size = base_size + 1),
      plot.subtitle       = element_text(size = base_size - 1, colour = "grey40"),
      plot.caption        = element_text(size = base_size - 2, colour = "grey50",
                                         hjust = 0)
    )
}

# Função utilitária: salva PDF + PNG de alta resolução
save_fig <- function(p, name, width = 7, height = 5, dpi = 300) {
  pdf_path <- file.path(OUT_DIR, paste0(name, ".pdf"))
  png_path <- file.path(OUT_DIR, paste0(name, ".png"))
  ggsave(pdf_path, plot = p, width = width, height = height, device = "pdf")
  ggsave(png_path, plot = p, width = width, height = height, dpi = dpi,
         device = "png", bg = "white")
  message("Saved: ", pdf_path, " | ", png_path)
}

# Rótulos de métricas
metric_labs <- c(
  RMSE = "RMSE",
  MAE  = "MAE",
  R2   = expression(R^2),
  Bias = "Bias"
)

# =============================================================================
# 1) Carrega dados
# =============================================================================

MS_DIR <- file.path(RESULTS, "convkriging2d_anisotropic_multiseed_confirmation")

# --- 1a) Benchmark comparável: RF + XGB no mesmo setup multiseed
#         Gerado por run_rf_xgb_multiseed_benchmark.R
#         Historical fallback removed: wadoux_summary_goal.csv mixed
#         unverified reference values with a different setup.
rf_xgb_all_path <- file.path(MS_DIR, "rf_xgb_multiseed_all.csv")
rf_xgb_available <- file.exists(rf_xgb_all_path)

if (rf_xgb_available) {
  message("Usando RF/XGB comparaveis (mesmo benchmark multiseed).")
  rf_xgb_all <- read.csv(rf_xgb_all_path, check.names = FALSE) %>%
    mutate(train_seed = factor(train_seed))

  rf_xgb_summary <- read.csv(file.path(MS_DIR, "rf_xgb_multiseed_summary.csv"),
                              check.names = FALSE)

  # Junta com Cubist do mesmo benchmark (seed_XX/cubist_results.csv)
  cubist_seeds <- lapply(c(11, 29, 47), function(s) {
    p <- file.path(MS_DIR, paste0("seed_", s), "cubist_results.csv")
    if (file.exists(p)) read.csv(p, check.names = FALSE) %>% mutate(train_seed = s)
    else NULL
  })
  cubist_all_per_seed <- bind_rows(Filter(Negate(is.null), cubist_seeds)) %>%
    mutate(train_seed = factor(train_seed))

  # Resumo Cubist comparavel
  cubist_summary_comp <- cubist_all_per_seed %>%
    group_by(model) %>%
    summarise(RMSE_mean = mean(RMSE, na.rm = TRUE),
              R2_mean   = mean(R2,   na.rm = TRUE),
              MAE_mean  = mean(MAE,  na.rm = TRUE),
              Bias_mean = mean(Bias, na.rm = TRUE),
              .groups = "drop")

  # Resumo de TODOS os baselines comparaveis (RF, XGB, Cubist)
  baselines_summary <- bind_rows(rf_xgb_summary, cubist_summary_comp) %>%
    arrange(RMSE_mean)

  fig1_subtitle <- paste0(
    "Same fixed benchmark as ConvKrigingNet2D -- ABG1 (Wadoux), n = 300, ",
    "10 outer spatial folds, seeds 11/29/47"
  )
  fig1_caption  <- "RF, XGB and Cubist trained on identical subtrain splits; averaged over 3 seeds"

} else {
  message("AVISO: rf_xgb_multiseed_all.csv nao encontrado.")
  message("  -> Execute code/run_rf_xgb_multiseed_benchmark.R primeiro.")
  stop(
    "Resultados comparaveis de RF/XGB nao encontrados. ",
    "O fallback 'wadoux_summary_goal.csv' foi removido porque misturava ",
    "valores nao verificados e um setup diferente. ",
    "Execute code/run_rf_xgb_multiseed_benchmark.R antes de gerar estas figuras."
  )
}

# Garante ordem dos modelos (pior para melhor em RMSE, de cima para baixo)
model_order_fig1 <- baselines_summary %>%
  arrange(desc(RMSE_mean)) %>%
  pull(model)

baselines_summary <- baselines_summary %>%
  mutate(model = factor(model, levels = model_order_fig1))

# --- 1b) Multiseed confirmation: resultados por fold
ms_all_path <- file.path(MS_DIR, "convkriging2d_anisotropic_multiseed_all.csv")
ms_all <- read.csv(ms_all_path, check.names = FALSE) %>%
  mutate(
    model = factor(model, levels = c("ConvKrigingNet2D", "Cubist")),
    train_seed = factor(train_seed)
  )

# --- 1c) Multiseed: resumo geral
ms_summary <- read.csv(file.path(MS_DIR, "convkriging2d_anisotropic_multiseed_summary.csv"),
                       check.names = FALSE) %>%
  mutate(model = factor(model, levels = c("ConvKrigingNet2D", "Cubist")))

# --- 1d) Multiseed: por seed
ms_seed <- read.csv(file.path(MS_DIR, "convkriging2d_anisotropic_multiseed_by_seed.csv"),
                    check.names = FALSE) %>%
  mutate(
    model = factor(model, levels = c("ConvKrigingNet2D", "Cubist")),
    train_seed = factor(train_seed)
  )

# =============================================================================
# 2) FIGURA 1 — Baselines comparaveis no mesmo benchmark (RF / XGB / Cubist)
#               + ConvKrigingNet2D sobreposto
# =============================================================================

# Adiciona ConvKrigingNet2D ao painel de comparação
conv_summary_row <- ms_summary %>%
  filter(model == "ConvKrigingNet2D") %>%
  select(model, RMSE_mean, R2_mean, MAE_mean, Bias_mean)

fig1_data <- bind_rows(baselines_summary, conv_summary_row) %>%
  mutate(
    is_conv = model == "ConvKrigingNet2D",
    model   = factor(model, levels = c(
      as.character(model_order_fig1), "ConvKrigingNet2D"
    ))
  )

fig1_long <- fig1_data %>%
  pivot_longer(cols = c(RMSE_mean, MAE_mean, R2_mean),
               names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
    RMSE_mean = "RMSE", MAE_mean = "MAE", R2_mean = "R2"
  ),
  metric = factor(metric, levels = c("RMSE", "MAE", "R2"),
                  labels = c("RMSE", "MAE", "R2")))

fig1 <- ggplot(fig1_long,
               aes(x = model, y = value, fill = model, alpha = is_conv)) +
  geom_col(width = 0.65, colour = "white", linewidth = 0.3) +
  geom_text(aes(label = round(value, 2)),
            hjust = -0.15, size = 3.0, colour = "grey20") +
  coord_flip(clip = "off") +
  scale_fill_manual(values = PALETTE, guide = "none") +
  scale_alpha_manual(values = c("FALSE" = 0.75, "TRUE" = 1.0), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22))) +
  facet_wrap(~ metric, scales = "free_x", nrow = 1) +
  labs(
    title    = "All models on the same spatial k-fold benchmark",
    subtitle = fig1_subtitle,
    x        = NULL,
    y        = NULL,
    caption  = fig1_caption
  ) +
  theme_paper(base_size = 10) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey88")
  )

save_fig(fig1, "fig1_all_models_comparable", width = 9, height = 4.5)

# =============================================================================
# 3) FIGURA 2 — Comparação geral ConvKrigingNet2D vs Cubist (resumo multiseed)
# =============================================================================

ms_summary_long <- ms_summary %>%
  pivot_longer(cols = c(RMSE_mean, MAE_mean, R2_mean, Bias_mean),
               names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
    RMSE_mean = "RMSE", MAE_mean = "MAE",
    R2_mean   = "R²",   Bias_mean = "Bias"
  ),
  metric = factor(metric, levels = c("RMSE", "MAE", "R²", "Bias")))

# Rótulos de posição do texto (acima/abaixo de zero)
ms_summary_long <- ms_summary_long %>%
  mutate(label_y = ifelse(value >= 0, value + max(abs(value)) * 0.03,
                          value - max(abs(value)) * 0.03))

fig2 <- ggplot(ms_summary_long,
               aes(x = model, y = value, fill = model)) +
  geom_col(width = 0.55, colour = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", value),
                vjust = ifelse(value >= 0, -0.4, 1.3)),
            size = 3.0, colour = "grey20") +
  scale_fill_manual(values = PALETTE, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0.1, 0.15))) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  labs(
    title    = "ConvKrigingNet2D vs Cubist -- corrected fixed benchmark",
    subtitle = "Mean over 10 outer spatial folds x 3 training seeds (seeds 11, 29, 47)",
    x        = NULL,
    y        = NULL,
    caption  = "Lower RMSE/MAE/Bias = better; higher R² = better"
  ) +
  theme_paper(base_size = 10) +
  theme(axis.text.x = element_text(angle = 0))

save_fig(fig2, "fig2_summary_comparison", width = 9, height = 4.5)

# =============================================================================
# 4) FIGURA 3 — Comparação por seed (dot plot conectado)
# =============================================================================

ms_seed_long <- ms_seed %>%
  pivot_longer(cols = c(RMSE_mean, MAE_mean, R2_mean),
               names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
    RMSE_mean = "RMSE", MAE_mean = "MAE", R2_mean = "R²"
  ),
  metric = factor(metric, levels = c("RMSE", "MAE", "R²")))

fig3 <- ggplot(ms_seed_long,
               aes(x = train_seed, y = value,
                   colour = model, shape = model, group = model)) +
  geom_line(linewidth = 0.7, alpha = 0.7, linetype = "dashed") +
  geom_point(size = 3.5) +
  geom_text(aes(label = sprintf("%.1f", value)),
            vjust = -0.85, size = 2.7, show.legend = FALSE) +
  scale_colour_manual(values = PALETTE, name = "Model") +
  scale_shape_manual(values = SHAPES, name = "Model") +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  labs(
    title    = "Seed-wise corrected benchmark results",
    subtitle = "Each point = mean over 10 outer spatial folds for the given training seed",
    x        = "Training seed",
    y        = NULL,
    caption  = "ConvKrigingNet2D achieves lower RMSE and higher R² than Cubist across all seeds"
  ) +
  theme_paper(base_size = 10)

save_fig(fig3, "fig3_seedwise", width = 9, height = 4.5)

# =============================================================================
# 5) FIGURA 4 — Scatter por fold: RMSE ConvKrigingNet2D vs Cubist
# =============================================================================
# Cada ponto = 1 outer fold × 1 seed.  Pontos abaixo da diagonal = CKN2D wins.

scatter_df <- ms_all %>%
  select(split, train_seed, model, RMSE) %>%
  pivot_wider(names_from = model, values_from = RMSE) %>%
  rename(RMSE_Conv = ConvKrigingNet2D, RMSE_Cubist = Cubist)

# Limite comum dos eixos (quadrado)
lim_max <- max(c(scatter_df$RMSE_Conv, scatter_df$RMSE_Cubist), na.rm = TRUE) * 1.05
lim_min <- min(c(scatter_df$RMSE_Conv, scatter_df$RMSE_Cubist), na.rm = TRUE) * 0.95

fig4 <- ggplot(scatter_df,
               aes(x = RMSE_Cubist, y = RMSE_Conv, colour = train_seed)) +
  # Diagonal de paridade
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = "grey55", linewidth = 0.8) +
  # Anotações das zonas
  annotate("text", x = lim_max * 0.95, y = lim_max * 0.87,
           label = "ConvKrigingNet2D\nbetter", hjust = 1, vjust = 0,
           size = 3.0, colour = "#2166AC", fontface = "italic") +
  annotate("text", x = lim_min * 1.12, y = lim_min * 1.25,
           label = "Cubist\nbetter", hjust = 0, vjust = 1,
           size = 3.0, colour = "#D6604D", fontface = "italic") +
  geom_point(size = 2.8, alpha = 0.85) +
  scale_colour_manual(
    values = c("11" = "#1B7837", "29" = "#762A83", "47" = "#E08214"),
    name   = "Training\nseed"
  ) +
  coord_fixed(xlim = c(lim_min, lim_max), ylim = c(lim_min, lim_max)) +
  scale_x_continuous(breaks = pretty_breaks(5)) +
  scale_y_continuous(breaks = pretty_breaks(5)) +
  labs(
    title    = "Fold-level RMSE: ConvKrigingNet2D vs Cubist",
    subtitle = "Each point = one outer spatial fold. Points below diagonal favour ConvKrigingNet2D",
    x        = "Cubist RMSE",
    y        = "ConvKrigingNet2D RMSE",
    caption  = "30 observations total (10 folds × 3 seeds)"
  ) +
  theme_paper(base_size = 10) +
  theme(legend.position = "right",
        panel.grid.major = element_line(colour = "grey90"))

save_fig(fig4, "fig4_scatter_per_fold", width = 6, height = 6)

# =============================================================================
# 6) FIGURA 5 — Distribuição de RMSE por modelo (violin + jitter)
# =============================================================================

fig5 <- ggplot(ms_all,
               aes(x = model, y = RMSE, fill = model, colour = model)) +
  geom_violin(alpha = 0.25, linewidth = 0.6, width = 0.7, trim = FALSE) +
  geom_jitter(aes(shape = train_seed),
              width = 0.12, size = 2.2, alpha = 0.80) +
  stat_summary(fun = mean, geom = "crossbar",
               width = 0.45, fatten = 2.5,
               colour = "grey20", linewidth = 0.7) +
  scale_fill_manual(values   = PALETTE, guide = "none") +
  scale_colour_manual(values = PALETTE, guide = "none") +
  scale_shape_manual(
    values = c("11" = 15, "29" = 17, "47" = 18),
    name   = "Training\nseed"
  ) +
  labs(
    title    = "Distribution of fold-level RMSE across seeds",
    subtitle = "Violin + individual folds (jittered). Horizontal bar = mean",
    x        = NULL,
    y        = "RMSE",
    caption  = "30 fold-level estimates per model (10 folds × 3 seeds)"
  ) +
  theme_paper(base_size = 10) +
  theme(legend.position = "right")

save_fig(fig5, "fig5_violin_rmse", width = 6, height = 5)

# =============================================================================
# 7) FIGURA 6 — Delta (ΔRMSE, ΔMAE, ΔR²) por seed (ConvKrigingNet2D − Cubist)
# =============================================================================
# Δ negativo em RMSE/MAE = bom para ConvKrigingNet2D
# Δ positivo em R²       = bom para ConvKrigingNet2D

delta_df <- ms_seed %>%
  select(train_seed, model, RMSE_mean, MAE_mean, R2_mean) %>%
  pivot_wider(names_from = model,
              values_from = c(RMSE_mean, MAE_mean, R2_mean)) %>%
  mutate(
    delta_RMSE = RMSE_mean_ConvKrigingNet2D - RMSE_mean_Cubist,
    delta_MAE  = MAE_mean_ConvKrigingNet2D  - MAE_mean_Cubist,
    delta_R2   = R2_mean_ConvKrigingNet2D   - R2_mean_Cubist
  ) %>%
  select(train_seed, delta_RMSE, delta_MAE, delta_R2) %>%
  pivot_longer(cols = starts_with("delta_"),
               names_to = "metric", values_to = "delta") %>%
  mutate(
    metric  = recode(metric,
                     delta_RMSE = "RMSE difference",
                     delta_MAE  = "MAE difference",
                     delta_R2   = "R2 difference"),
    metric  = factor(metric, levels = c("RMSE difference", "MAE difference", "R2 difference")),
    better  = case_when(
      metric %in% c("RMSE difference", "MAE difference") & delta < 0 ~ TRUE,
      metric == "R2 difference"                          & delta > 0 ~ TRUE,
      TRUE ~ FALSE
    ),
    # Anotação de direção
    direction = ifelse(better, "ConvKrigingNet2D\nbetter", "Cubist\nbetter")
  )

fig6 <- ggplot(delta_df,
               aes(x = train_seed, y = delta,
                   fill = better, colour = better)) +
  geom_col(width = 0.55, linewidth = 0.3, colour = "white") +
  geom_hline(yintercept = 0, linewidth = 0.6, colour = "grey30") +
  geom_text(aes(label = sprintf("%+.2f", delta),
                vjust = ifelse(delta >= 0, -0.4, 1.3)),
            size = 3.0, colour = "grey20") +
  scale_fill_manual(
    values = c("TRUE" = "#2166AC", "FALSE" = "#D6604D"),
    labels = c("TRUE" = "ConvKrigingNet2D better", "FALSE" = "Cubist better"),
    name   = NULL
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.18))) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  labs(
    title    = "Relative performance: ConvKrigingNet2D minus Cubist",
    subtitle = "Negative dRMSE/dMAE and positive dR2 indicate advantage of ConvKrigingNet2D",
    x        = "Training seed",
    y        = "Difference (ConvKrigingNet2D - Cubist)",
    caption  = "Values averaged over 10 outer spatial folds"
  ) +
  theme_paper(base_size = 10) +
  theme(legend.position = "bottom")

save_fig(fig6, "fig6_delta_per_seed", width = 9, height = 4.5)

# =============================================================================
# 8) FIGURA 7 — Painel combinado para o paper (Fig 2 + Fig 3 sobrepostos)
#    Útil para uma única figura de 2 painéis no manuscrito
# =============================================================================

fig7 <- (fig2 / fig3) +
  plot_annotation(
    title = "ConvKrigingNet2D vs Cubist -- corrected fixed benchmark",
    tag_levels = "A",
    theme = theme(
      plot.title = element_text(face = "bold", size = 12)
    )
  )

save_fig(fig7, "fig7_combined_summary_seed", width = 10, height = 9)

# =============================================================================
# 9) FIGURA 8 — Painel combinado para o paper (Fig 4 + Fig 6)
# =============================================================================

fig8 <- (fig4 | fig6) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(plot.title = element_text(face = "bold", size = 12))
  )

save_fig(fig8, "fig8_combined_scatter_delta", width = 13, height = 6)

# =============================================================================
# 10) FIGURA SUPLEMENTAR — R² por fold e seed (heatmap)
# =============================================================================

heatmap_df <- ms_all %>%
  mutate(split = factor(split, levels = sort(unique(split))))

fig_supp_r2 <- ggplot(heatmap_df,
                      aes(x = split, y = train_seed, fill = R2)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", R2)), size = 2.5, colour = "grey10") +
  scale_fill_gradient2(low = "#D6604D", mid = "white", high = "#2166AC",
                       midpoint = 0.65, limits = c(0, 1),
                       name = expression(R^2)) +
  facet_wrap(~ model, nrow = 1) +
  labs(
    title    = "Fold-level R² by training seed",
    subtitle = "Each cell = one outer spatial fold × training seed",
    x        = "Outer fold",
    y        = "Training seed"
  ) +
  theme_paper(base_size = 10) +
  theme(
    panel.grid   = element_blank(),
    legend.position = "right"
  )

save_fig(fig_supp_r2, "figS1_heatmap_r2_per_fold", width = 10, height = 4)

fig_supp_rmse <- ggplot(heatmap_df,
                        aes(x = split, y = train_seed, fill = RMSE)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.0f", RMSE)), size = 2.5, colour = "grey10") +
  scale_fill_gradient(low = "#2166AC", high = "#D6604D",
                      name = "RMSE") +
  facet_wrap(~ model, nrow = 1) +
  labs(
    title    = "Fold-level RMSE by training seed",
    subtitle = "Each cell = one outer spatial fold × training seed",
    x        = "Outer fold",
    y        = "Training seed"
  ) +
  theme_paper(base_size = 10) +
  theme(panel.grid = element_blank(), legend.position = "right")

save_fig(fig_supp_rmse, "figS2_heatmap_rmse_per_fold", width = 10, height = 4)

# =============================================================================
# Resumo final
# =============================================================================

message("\n=== Figuras geradas em: ", OUT_DIR, " ===")
figs <- list.files(OUT_DIR, pattern = "\\.(pdf|png)$")
for (f in sort(figs)) message("  ", f)
message("Total: ", length(figs), " arquivos")
