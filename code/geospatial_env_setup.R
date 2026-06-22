configure_geospatial_env <- function() {
  set_env_if_missing_or_invalid <- function(var, candidates) {
    current <- Sys.getenv(var, unset = "")
    if (nzchar(current) && dir.exists(current)) return(invisible(current))

    candidates <- unique(candidates[nzchar(candidates)])
    hit <- candidates[dir.exists(candidates)][1]
    if (!is.na(hit)) {
      args <- stats::setNames(as.list(hit), var)
      do.call(Sys.setenv, args)
    }
    invisible(Sys.getenv(var, unset = ""))
  }

  homebrew_prefix <- Sys.getenv("HOMEBREW_PREFIX", unset = "")
  set_env_if_missing_or_invalid(
    "PROJ_LIB",
    c(file.path(homebrew_prefix, "share", "proj"),
      "/opt/homebrew/share/proj",
      "/usr/local/share/proj")
  )
  set_env_if_missing_or_invalid(
    "GDAL_DATA",
    c(file.path(homebrew_prefix, "share", "gdal"),
      "/opt/homebrew/share/gdal",
      "/usr/local/share/gdal")
  )
}

configure_geospatial_env()
