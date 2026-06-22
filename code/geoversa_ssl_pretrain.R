project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source(file.path(project_root, "code", "geospatial_env_setup.R"), local = TRUE)

library(torch)
library(terra)
library(dplyr)
library(sf)

parse_bool_env_ssl <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = if (default) "TRUE" else "FALSE")
  toupper(raw) %in% c("TRUE", "1", "YES", "Y")
}

parse_num_env_ssl <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  val <- suppressWarnings(as.numeric(raw))
  if (!is.finite(val)) default else val
}

load_conv_auto_env_ssl <- function(project_root) {
  loader_parent <- new.env(parent = globalenv())
  env <- new.env(parent = loader_parent)
  loader_parent$source <- function(file, ...) {
    is_absolute <- grepl("^(?:[A-Za-z]:[\\\\/]|/)", file)
    resolved <- if (is_absolute) file else file.path(project_root, sub("^\\./", "", file))
    sys.source(resolved, envir = env)
    invisible(NULL)
  }
  sys.source(file.path(project_root, "code", "ConvKrigingNet2D.R"), envir = env)
  sys.source(file.path(project_root, "code", "ConvKrigingNet2D_Auto_v5.R"), envir = env)
  sys.source(file.path(project_root, "code", "ConvKrigingNet2D_Auto.R"), envir = env)
  env
}

load_wadoux_context_ssl <- function(project_root) {
  repo_dir <- file.path(project_root, "external", "SpatialValidation")
  data_dir <- file.path(repo_dir, "data")
  tif_files <- list.files(data_dir, pattern = "\\.tif$", full.names = TRUE)
  if (length(tif_files) == 0L) {
    stop("No .tif files found in Wadoux data directory: ", data_dir)
  }
  s <- terra::rast(tif_files)
  names(s) <- tools::file_path_sans_ext(basename(tif_files))
  if (!"ABG1" %in% names(s)) {
    stop("ABG1.tif not found in Wadoux dataset.")
  }
  s[["ABG1"]][s[["ABG1"]] == 0] <- NA
  s_df <- as.data.frame(s, xy = TRUE, na.rm = TRUE)
  predictor_names <- setdiff(names(s_df), c("x", "y", "ABG1"))
  points_sf <- sf::st_as_sf(s_df, coords = c("x", "y"), crs = terra::crs(s), remove = FALSE)
  list(
    stack = s,
    data = s_df,
    points_sf = points_sf,
    response = "ABG1",
    predictors = predictor_names,
    val_dist_km = 350,
    repo_dir = repo_dir
  )
}

conv_env <- load_conv_auto_env_ssl(project_root)

device <- Sys.getenv("GEOVERSA_SSL_DEVICE", unset = Sys.getenv("WADOUX_DEVICE", unset = "cpu"))
results_dir <- Sys.getenv("GEOVERSA_SSL_RESULTS_DIR", unset = file.path(project_root, "results", "ssl_pretrain"))
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

sample_size <- as.integer(round(parse_num_env_ssl("WADOUX_SAMPLE_SIZE", 500)))
sample_size <- max(100L, sample_size)
patch_size_default <- as.integer(min(max(8L, floor(sqrt(sample_size))), 31L))
patch_global_raw <- trimws(Sys.getenv(
  "GEOVERSA_PATCH_GLOBAL_SIZE",
  unset = Sys.getenv("GEOVERSA_SSL_PATCH_GLOBAL_SIZE", unset = "")
))
patch_local_raw <- trimws(Sys.getenv(
  "GEOVERSA_PATCH_LOCAL_SIZE",
  unset = Sys.getenv("GEOVERSA_SSL_PATCH_LOCAL_SIZE", unset = "")
))
patch_size <- if (nzchar(patch_global_raw)) {
  as.integer(round(suppressWarnings(as.numeric(patch_global_raw))))
} else {
  as.integer(round(parse_num_env_ssl("GEOVERSA_SSL_PATCH_SIZE", patch_size_default)))
}
patch_size <- max(3L, patch_size)
n_samples_env <- trimws(Sys.getenv("GEOVERSA_SSL_SAMPLES", unset = ""))
epochs_env <- trimws(Sys.getenv("GEOVERSA_SSL_EPOCHS", unset = ""))
batch_size <- as.integer(round(parse_num_env_ssl("GEOVERSA_SSL_BATCH_SIZE", 256)))
cache_chunk_size <- as.integer(round(parse_num_env_ssl("GEOVERSA_SSL_CACHE_CHUNK_SIZE", 1024)))
lr <- parse_num_env_ssl("GEOVERSA_SSL_LR", 1e-3)
wd <- parse_num_env_ssl("GEOVERSA_SSL_WD", 1e-4)
mask_ratio_env <- trimws(Sys.getenv("GEOVERSA_SSL_MASK_RATIO", unset = ""))
seed <- as.integer(round(parse_num_env_ssl("GEOVERSA_SSL_SEED", 12345)))
multiscale_patch <- if (nzchar(patch_local_raw)) {
  TRUE
} else {
  parse_bool_env_ssl(
    "GEOVERSA_SSL_MULTISCALE",
    default = parse_bool_env_ssl(
      "GEOVERSA_AUTO_PATCH_PYRAMID",
      default = identical(Sys.getenv("WADOUX_MODEL_PROFILE", unset = "auto"), "auto")
    )
  )
}

context <- load_wadoux_context_ssl(project_root)
if (is.null(context) || is.null(context$stack) || is.null(context$predictors)) {
  stop("Failed to load Wadoux context for SSL pretraining.")
}

patch_layers <- context$stack[[context$predictors]]
patch_channels <- terra::nlyr(patch_layers)
n_samples <- if (nzchar(n_samples_env)) {
  as.integer(round(parse_num_env_ssl("GEOVERSA_SSL_SAMPLES", 100000)))
} else if (!is.null(conv_env$.auto_ssl_n_samples_from_raster) && is.function(conv_env$.auto_ssl_n_samples_from_raster)) {
  conv_env$.auto_ssl_n_samples_from_raster(patch_layers)
} else {
  100000L
}
epochs <- if (nzchar(epochs_env)) {
  as.integer(round(parse_num_env_ssl("GEOVERSA_SSL_EPOCHS", 50)))
} else if (!is.null(conv_env$.auto_ssl_epochs_from_samples) && is.function(conv_env$.auto_ssl_epochs_from_samples)) {
  conv_env$.auto_ssl_epochs_from_samples(n_samples)
} else {
  50L
}
mask_ratio <- if (nzchar(mask_ratio_env)) {
  parse_num_env_ssl("GEOVERSA_SSL_MASK_RATIO", 0.50)
} else if (!is.null(conv_env$.auto_ssl_mask_ratio_from_raster) && is.function(conv_env$.auto_ssl_mask_ratio_from_raster)) {
  conv_env$.auto_ssl_mask_ratio_from_raster(patch_layers)
} else {
  0.50
}
d_ssl <- if (!is.null(conv_env$.auto_d_from_n) && is.function(conv_env$.auto_d_from_n)) {
  conv_env$.auto_d_from_n(sample_size)
} else {
  192L
}
patch_dim <- if (!is.null(conv_env$.auto_patch_dim_from_cnn) && is.function(conv_env$.auto_patch_dim_from_cnn)) {
  conv_env$.auto_patch_dim_from_cnn(patch_channels, patch_size, d_ssl)
} else {
  as.integer(max(48L, min(d_ssl, ceiling(sqrt(patch_channels * patch_size^2)))))
}
patch_local_size <- if (isTRUE(multiscale_patch)) {
  patch_local_env <- patch_local_raw
  if (nzchar(trimws(patch_local_env))) {
    local_size <- as.integer(max(9L, round(suppressWarnings(as.numeric(patch_local_env)))))
    if (isTRUE(local_size %% 2L == 0L)) local_size <- local_size - 1L
    local_size
  } else {
    as.integer(max(9L, min(patch_size, ifelse(patch_size %% 2L == 0L, patch_size - 1L, patch_size))))
  }
} else {
  NULL
}

cat(sprintf(
  "[GeoVersa SSL] Starting pretrain | device=%s | n=%d | epochs=%d | batch=%d | patch=%d | patch_dim=%d | multiscale=%s\n",
  device, n_samples, epochs, batch_size, patch_size, patch_dim, if (isTRUE(multiscale_patch)) "ON" else "OFF"
))

artifact <- conv_env$pretrain_patch_encoder_mae_auto(
  context = context,
  patch_size = patch_size,
  patch_dim = patch_dim,
  multiscale_patch = multiscale_patch,
  patch_local_size = patch_local_size,
  n_samples = n_samples,
  epochs = epochs,
  batch_size = batch_size,
  cache_chunk_size = cache_chunk_size,
  seed = seed,
  device = device,
  lr = lr,
  wd = wd,
  mask_ratio = mask_ratio
)

state_path <- file.path(results_dir, "ssl_patch_encoder_state.pt")
summary_path <- file.path(results_dir, "geoversa_ssl_pretrain_summary.csv")
history_path <- file.path(results_dir, "geoversa_ssl_pretrain_history.csv")

torch::torch_save(artifact$enc_patch_state, state_path)
write.csv(artifact$meta, summary_path, row.names = FALSE)
write.csv(artifact$history, history_path, row.names = FALSE)

cat(sprintf("[GeoVersa SSL] Saved encoder state: %s\n", state_path))
cat(sprintf("[GeoVersa SSL] Saved summary: %s\n", summary_path))
cat(sprintf("[GeoVersa SSL] Saved history: %s\n", history_path))
