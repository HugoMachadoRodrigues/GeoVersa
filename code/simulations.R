# =============================================================================
# simulations.R
#
# Controlled simulations backing three claims that the manuscript would
# otherwise assert without evidence:
#
#   S1. Anisotropy recovery. The geometry estimator returns isotropy for
#       isotropic fields (no false positives) and the correct major-axis angle
#       for anisotropic ones. Run over a grid of anisotropy ratios, angles and
#       signal-to-noise ratios.
#
#   S2. Why clustering refuses anisotropy. The manuscript observed 0 of 30
#       acceptances under clustered sampling and explained it as the design
#       being unable to support estimating direction. That is a hypothesis, not
#       a result. Here we hold the TRUE anisotropy fixed and vary only the
#       sampling design, so any change in recovery is attributable to the design.
#
#   S3. Conformal coverage. Empirical coverage of the CV+ interval against its
#       1-2alpha guarantee, under representative and clustered calibration.
#
# Usage:  Rscript code/simulations.R
# Output: results/simulations/{S1_anisotropy,S2_design,S3_coverage}.csv
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
suppressWarnings(suppressPackageStartupMessages({
  library(ranger); library(caret); library(gstat); library(sp)
}))
source("R/dbtarf.R")

out_dir <- "results/simulations"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- field simulator --------------------------------------------------------
# Gaussian field with exponential covariance on an anisotropic geometry.
sim_field <- function(n, ratio, angle, rng = 25, snr = 4, seed = 1,
                      design = c("random", "clustered"), extent = 100,
                      calib_n = NULL) {
  design <- match.arg(design)
  set.seed(seed)
  # calib_n, when given, splits the draw: the first calib_n points follow the
  # named design and the remainder are uniform over the domain, standing in for
  # the target population the map has to cover.
  n_design <- if (is.null(calib_n)) n else min(calib_n, n)
  place <- function(m) {
    if (design == "random" || m == 0L) {
      cbind(runif(m, 0, extent), runif(m, 0, extent))
    } else {
      n_psu <- max(2L, round(m / 25))
      ctr <- cbind(runif(n_psu, 0, extent), runif(n_psu, 0, extent))
      idx <- sample(seq_len(n_psu), m, replace = TRUE)
      cbind(pmin(pmax(ctr[idx, 1] + rnorm(m, 0, extent * 0.03), 0), extent),
            pmin(pmax(ctr[idx, 2] + rnorm(m, 0, extent * 0.03), 0), extent))
    }
  }
  xy <- rbind(place(n_design),
              if (n > n_design)
                cbind(runif(n - n_design, 0, extent),
                      runif(n - n_design, 0, extent)) else NULL)
  z <- .dbt_aniso_transform(xy, angle, ratio)
  S <- exp(-as.matrix(dist(z)) / rng)
  L <- chol(S + diag(1e-6, n))
  sig <- as.numeric(t(L) %*% rnorm(n))
  sig <- sig / stats::sd(sig)
  list(xy = xy, y = sig + rnorm(n, 0, 1 / sqrt(snr)))
}

ang_err <- function(est, truth) {
  d <- abs(((est - truth + 90) %% 180) - 90)  # circular, period 180 degrees
  d
}

# ---- S1: anisotropy recovery under simple random sampling -------------------

message("[sim] S1 anisotropy recovery ...")
S1 <- list()
grid1 <- expand.grid(ratio = c(1.00, 0.60, 0.35, 0.20),
                     angle = c(0, 45, 90, 135),
                     snr   = c(2, 8),
                     rep   = 1:6)
for (i in seq_len(nrow(grid1))) {
  g <- grid1[i, ]
  d <- sim_field(500, g$ratio, g$angle, snr = g$snr, seed = 1000 + i)
  vg <- tryCatch(.dbt_fit_variogram(d$y, d$xy), error = function(e) NULL)
  S1[[i]] <- data.frame(
    true_ratio = g$ratio, true_angle = g$angle, snr = g$snr, rep = g$rep,
    est_ratio = if (is.null(vg)) NA_real_ else vg$aniso_ratio,
    est_angle = if (is.null(vg)) NA_real_ else vg$aniso_angle,
    family    = if (is.null(vg)) NA_character_ else vg$family,
    stringsAsFactors = FALSE)
}
S1 <- do.call(rbind, S1)
S1$declared_aniso <- is.finite(S1$est_ratio) & S1$est_ratio < 1
S1$angle_error <- ifelse(S1$declared_aniso & S1$true_ratio < 1,
                         ang_err(S1$est_angle, S1$true_angle), NA_real_)
write.csv(S1, file.path(out_dir, "S1_anisotropy.csv"), row.names = FALSE)

cat("\n== S1: anisotropy recovery (simple random, n = 500) ==\n")
s1 <- aggregate(cbind(declared_aniso, angle_error) ~ true_ratio + snr, S1,
                function(v) mean(v, na.rm = TRUE), na.action = na.pass)
print(s1, row.names = FALSE)
fp <- mean(S1$declared_aniso[S1$true_ratio == 1])
cat(sprintf("\n  false-positive rate on isotropic fields: %.3f\n", fp))

# ---- S2: does the sampling design drive the refusal? ------------------------

message("[sim] S2 design effect ...")
S2 <- list(); k <- 0L
for (design in c("random", "clustered")) {
  for (ratio in c(1.00, 0.35)) {
    for (rep in 1:12) {
      k <- k + 1L
      d <- sim_field(500, ratio, 45, snr = 8, seed = 5000 + k, design = design)
      vg <- tryCatch(.dbt_fit_variogram(d$y, d$xy), error = function(e) NULL)
      S2[[k]] <- data.frame(
        design = design, true_ratio = ratio, rep = rep,
        est_ratio = if (is.null(vg)) NA_real_ else vg$aniso_ratio,
        est_angle = if (is.null(vg)) NA_real_ else vg$aniso_angle,
        stringsAsFactors = FALSE)
    }
  }
}
S2 <- do.call(rbind, S2)
S2$declared_aniso <- is.finite(S2$est_ratio) & S2$est_ratio < 1
write.csv(S2, file.path(out_dir, "S2_design.csv"), row.names = FALSE)

cat("\n== S2: same truth, different sampling design ==\n")
print(aggregate(declared_aniso ~ design + true_ratio, S2, mean), row.names = FALSE)

# ---- S3: CV+ coverage -------------------------------------------------------

message("[sim] S3 CV+ coverage ...")
# One field realisation is generated over n + nt locations and then SPLIT into
# calibration and test sets. This matters: drawing the test responses from an
# independent realisation would destroy exchangeability by construction and
# would test the simulator rather than the interval.
S3 <- list(); k <- 0L
for (design in c("random", "clustered")) {
  for (rep in 1:10) {
    k <- k + 1L
    n <- 300; nt <- 400
    # The earlier version drew n + nt points under the design and split them at
    # random. Under the clustered design that made the test set a second sample
    # of the same clustered cloud, so calibration and test were exchangeable by
    # construction and the run could not detect the failure it was meant to
    # probe. Audit of 2026-08-05 caught this.
    #
    # The target population is now uniform over the domain in both arms, and only
    # the calibration design changes. That is the contrast the manuscript argues
    # about: a clustered calibration sample predicting a domain it does not cover.
    f <- sim_field(n + nt, 0.5, 45, snr = 4, seed = 9000 + k, design = design,
                   calib_n = n)
    dat <- data.frame(y = f$y, a = rnorm(n + nt), b = rnorm(n + nt),
                      x = f$xy[, 1], yy = f$xy[, 2])
    tr <- dat[seq_len(n), ]; te <- dat[(n + 1):(n + nt), ]
    bp <- .dbt_rf_default_params(p = 2, n = n, role = "base")
    cv <- tryCatch(.dbt_cvplus(tr, te, "y", c("a", "b"), c("x", "yy"),
                               "resid_rf", 0.5, bp, NULL, NULL, TRUE,
                               k = 5L, seed = k, alpha = 0.10),
                   error = function(e) NULL)
    if (is.null(cv)) next
    S3[[k]] <- data.frame(
      design = design, rep = rep,
      coverage = mean(te$y >= cv$lower & te$y <= cv$upper),
      width = mean(cv$upper - cv$lower), stringsAsFactors = FALSE)
  }
}
S3 <- do.call(rbind, S3)
write.csv(S3, file.path(out_dir, "S3_coverage.csv"), row.names = FALSE)

cat("\n== S3: CV+ empirical coverage (nominal 0.90, guarantee >= 0.80) ==\n")
print(aggregate(cbind(coverage, width) ~ design, S3, mean), row.names = FALSE)

cat(sprintf("\nwrote %s\n", out_dir))
