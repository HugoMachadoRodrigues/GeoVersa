# =============================================================================
# figures_protocols.R
#
# Regenerates Text/dbtarf_article/figures/fig4_protocols.png from the canonical
# v0.3.0 result directories, so the figure cannot drift from the tables again.
#
# Reads:
#   results/canon_v030_full_random_confirm30/      (DesignBased, 30 reps)
#   results/canon_v030_protocols_confirm10/        (Population / RandomKFold /
#                                                   SpatialKFold, 10 reps each)
#
# Usage:  Rscript code/figures_protocols.R
# =============================================================================

root <- "/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/GeoVersa"
setwd(root)

out_png <- "Text/dbtarf_article/figures/fig4_protocols.png"

# ---- DesignBased (random design, 30 reps) -----------------------------------

db <- read.csv("results/canon_v030_full_random_confirm30/dbtarf_summary_by_protocol.csv",
               stringsAsFactors = FALSE)
get_db <- function(m) db$RMSE_mean[db$model == m][1]

# ---- Population / RandomKFold / SpatialKFold (10 reps each) -----------------

pr <- read.csv("results/canon_v030_protocols_confirm10/dbtarf_protocols_all_results.csv",
               stringsAsFactors = FALSE)
agg <- function(proto, col) mean(pr[[col]][pr$protocol == proto], na.rm = TRUE)

estimands <- c("Population", "DesignBased", "RandomKFold", "SpatialKFold")
n_reps    <- c(10, 30, 10, 10)

rf_rmse <- c(agg("Population", "RMSE_RF"),
             get_db("RF"),
             agg("RandomKFold", "RMSE_RF"),
             agg("SpatialKFold", "RMSE_RF"))

db_rmse <- c(agg("Population", "RMSE_DBTARF"),
             get_db("DBTARF"),
             agg("RandomKFold", "RMSE_DBTARF"),
             agg("SpatialKFold", "RMSE_DBTARF"))

M <- rbind(RF = rf_rmse, DBTARF = db_rmse)
colnames(M) <- estimands

# ---- Plot -------------------------------------------------------------------

grDevices::png(out_png, width = 2200, height = 1250, res = 200)
op <- par(mar = c(6.2, 5.5, 4.2, 1.5))

cols <- c("grey55", "#1B7837")
labs <- sprintf("%s\n(n = %d)\n%s = %+.2f",
                estimands, n_reps, "Δ", db_rmse - rf_rmse)

bp <- barplot(M, beside = TRUE, col = cols, border = "black",
              ylim = c(0, 44), yaxs = "i", axisnames = FALSE,
              ylab = expression("Estimated RMSE  (Mg ha"^-1 * ")"),
              cex.lab = 1.2, cex.axis = 1.05,
              main = "DB-TARF vs the Wadoux default RF across four validation estimands",
              cex.main = 1.15)

text(bp, M + 1.2, sprintf("%.1f", M), cex = 1.0)

# grouped x labels, with the paired delta folded in
mtext(labs, side = 1, line = 2.6, at = colMeans(bp), cex = 1.0)

# framing: only the first two estimands are map accuracy
mtext(expression(bold("map accuracy")), side = 3, line = -0.4,
      at = mean(colMeans(bp)[1:2]), cex = 0.95, col = "#1B5E20")
mtext("model comparison only (not map accuracy)", side = 3, line = -0.4,
      at = mean(colMeans(bp)[3:4]), cex = 0.95, col = "grey30")

legend("topleft", inset = c(0.01, 0.02),
       legend = c("RF (Wadoux default)", "DB-TARF"),
       fill = cols, border = "black", bty = "n", cex = 1.1, horiz = TRUE)

par(op)
grDevices::dev.off()

cat(sprintf("wrote %s\n", out_png))
print(round(M, 3))
cat("\ndeltas:\n")
print(round(db_rmse - rf_rmse, 3))
