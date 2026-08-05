# =============================================================================
# project_paths.R
#
# One place that answers "where am I?" so no other script has to hardcode a
# path. Every analysis script sources this and then uses gv_root(), gv_path()
# and gv_data_root() instead of an absolute string.
#
# Reviewers cannot run a script that says
# /Users/<someone>/Library/CloudStorage/OneDrive-Personal/GeoVersa, and the
# 2026-08-05 audit flagged nine such scripts plus a data root that still
# pointed at a folder named "Deep Kriging" from before the project was renamed.
#
# Resolution order for the repository root:
#   1. the GEOVERSA_ROOT environment variable, if set
#   2. the nearest ancestor of the working directory holding DESCRIPTION and R/
#   3. the working directory, as a last resort
#
# Resolution order for the benchmark data, which is gitignored and must be
# fetched separately (see data/README.md):
#   1. the GEOVERSA_DATA environment variable, if set
#   2. <root>/external/SpatialValidation
#   3. a sibling "Deep Kriging" checkout, for machines that still have one
# =============================================================================

gv_root <- local({
  cached <- NULL
  function() {
    if (!is.null(cached)) return(cached)
    env <- Sys.getenv("GEOVERSA_ROOT", unset = "")
    if (nzchar(env) && dir.exists(env)) {
      cached <<- normalizePath(env, mustWork = FALSE)
      return(cached)
    }
    d <- normalizePath(getwd(), mustWork = FALSE)
    repeat {
      if (file.exists(file.path(d, "DESCRIPTION")) && dir.exists(file.path(d, "R"))) {
        cached <<- d
        return(cached)
      }
      up <- dirname(d)
      if (identical(up, d)) break
      d <- up
    }
    warning("gv_root(): no DESCRIPTION found above ", getwd(),
            "; falling back to the working directory")
    cached <<- normalizePath(getwd(), mustWork = FALSE)
    cached
  }
})

# Path relative to the repository root, created on demand when `create = TRUE`.
gv_path <- function(..., create = FALSE) {
  p <- file.path(gv_root(), ...)
  if (isTRUE(create)) dir.create(p, recursive = TRUE, showWarnings = FALSE)
  p
}

# Root of the Wadoux (2021) benchmark materials. Errors early and legibly
# rather than letting a downstream read fail with a confusing message.
gv_data_root <- function(required = TRUE) {
  env <- Sys.getenv("GEOVERSA_DATA", unset = "")
  cands <- c(
    if (nzchar(env)) env,
    gv_path("external", "SpatialValidation"),
    file.path(dirname(gv_root()), "Deep Kriging", "external", "SpatialValidation")
  )
  hit <- cands[dir.exists(cands)]
  if (length(hit) > 0L) return(normalizePath(hit[1L], mustWork = FALSE))
  if (!required) return(NA_character_)
  stop("Benchmark data not found. Set GEOVERSA_DATA, or place the clone of\n",
       "github.com/AlexandreWadoux/SpatialValidation at\n  ",
       gv_path("external", "SpatialValidation"), "\n",
       "See data/README.md for the acquisition steps.", call. = FALSE)
}

invisible(NULL)
