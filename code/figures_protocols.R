# =============================================================================
# figures_protocols.R
#
# Regenerates Text/ecomod_submission/figures/fig4_protocols.png from the canonical
# v0.3.0 result directories, so the figure cannot drift from the tables again.
#
# Reads:
#   results/rev1_random_confirm30/                 (DesignBased, 30 reps; the
#                                                  same campaign the estimand
#                                                  table reads, so the two agree)
#   results/canon_v030_protocols_confirm10/        (Population / RandomKFold /
#                                                   SpatialKFold, 10 reps each)
#
# Usage:  Rscript code/figures_protocols.R
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

out_png <- "Text/ecomod_submission/figures/fig4_protocols.png"

# ---- DesignBased (random design, 30 reps) -----------------------------------

# Read through gv_results and from the campaign the tables read, not through a
# relative path into the repository's own results directory. Both mattered: the
# hardcoded path bypassed the resolver that exists so that no script carries a
# machine-specific location, and rev1_random_confirm30 is a superseded campaign
# that predates the provenance discipline, records no git_commit, and disagrees
# with the current one. The figure showed a design-based gain of 0.876 while
# Table 6 showed 0.849, from what a reader would take to be one experiment.
db_dir <- gv_results("final_random_confirm30")
stopifnot(dir.exists(db_dir))
db_cfg <- read.csv(file.path(db_dir, "dbtarf_config.csv"), stringsAsFactors = FALSE)
if (!"git_commit" %in% names(db_cfg) || !nzchar(as.character(db_cfg$git_commit[1]))) {
  stop("campaign without provenance: ", db_dir)
}
db <- read.csv(file.path(db_dir, "dbtarf_summary_by_protocol.csv"),
               stringsAsFactors = FALSE)
get_db <- function(m) db$RMSE_mean[db$model == m][1]

# ---- Population / RandomKFold / SpatialKFold (10 reps each) -----------------

pr <- read.csv(gv_results("canon_v030_protocols_confirm10",
                          "dbtarf_protocols_all_results.csv"),
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
