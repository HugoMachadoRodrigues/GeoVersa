# =============================================================================
# SoilFLUX-KrigingNet - SIMULAÇÃO DSM COMPLETA (R)
# Cria rasters, pontos, alvo (y), blocked CV, normalização por fold,
# e vizinhos K (neighbor_idx) sem vazamento (train-only).
# =============================================================================

rm(list = ls())
set.seed(123)

# -----------------------------
# 0) Pacotes (instala se faltar)
# -----------------------------
# pkgs <- c("terra", "sf", "dplyr", "FNN")
# to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
# if (length(to_install) > 0) install.packages(to_install)

library(terra)
library(sf)
library(dplyr)
library(FNN)

# =============================================================================
# 1) PARÂMETROS DA SIMULAÇÃO (AJUSTE AQUI)
# =============================================================================
cfg <- list(
  # domínio espacial (metros) - área retangular
  width_m  = 20000,   # 20 km
  height_m = 20000,   # 20 km
  
  # raster
  res_m   = 30,       # resolução 30 m
  n_cov   = 12,       # número de covariáveis raster simuladas (C_tab)
  nodata_frac = 0.05, # fração de pixels sem dado
  
  # pontos (pedons)
  n_points = 500,
  
  # geração do alvo y
  beta_strength = 1.5,   # força dos efeitos das covariáveis
  trend_strength = 0.8,  # força da tendência espacial suave (macro)
  nugget_sd = 0.3,       # ruído independente
  corr_sd   = 0.8,       # amplitude do ruído espacialmente correlacionado
  corr_range_m = 2500,   # alcance (range) do ruído correlacionado (m)
  
  # blocked CV
  n_folds = 5,
  block_size_m = 4000,   # tamanho do bloco p/ CV (m) (ex.: 4 km)
  
  # vizinhos K (para o bloco Neural Kriging)
  K = 24,
  
  # normalização robusta
  use_robust_scaling = TRUE
)

# =============================================================================
# 2) FUNÇÕES AUXILIARES
# =============================================================================

# 2.1) Robust scaling (median/IQR) ou z-score (mean/sd)
fit_scaler <- function(X, robust = TRUE) {
  if (robust) {
    center <- apply(X, 2, median, na.rm = TRUE)
    scale  <- apply(X, 2, IQR, na.rm = TRUE)
    scale[scale == 0] <- 1
  } else {
    center <- colMeans(X, na.rm = TRUE)
    scale  <- apply(X, 2, sd, na.rm = TRUE)
    scale[scale == 0] <- 1
  }
  list(center = center, scale = scale, robust = robust)
}

apply_scaler <- function(X, scaler) {
  sweep(sweep(X, 2, scaler$center, "-"), 2, scaler$scale, "/")
}

# 2.2) Cria folds blocked CV por grid (sem pacote extra)
make_block_folds <- function(points_sf, block_size_m, n_folds) {
  # points_sf em CRS métrico (aqui usamos sistema planar local)
  bb <- st_bbox(points_sf)
  
  # ID do bloco por coordenada
  xy <- st_coordinates(points_sf)
  bx <- floor((xy[,1] - bb["xmin"]) / block_size_m)
  by <- floor((xy[,2] - bb["ymin"]) / block_size_m)
  block_id <- paste(bx, by, sep = "_")
  
  # atribui fold por bloco (random mas consistente)
  unique_blocks <- unique(block_id)
  fold_of_block <- sample(rep(1:n_folds, length.out = length(unique_blocks)))
  names(fold_of_block) <- unique_blocks
  
  fold_id <- fold_of_block[block_id]
  fold_id <- as.integer(fold_id)
  
  fold_id
}

# 2.3) Split train/val dentro do "treino" por blocos (val = subset de blocos do treino)
make_train_val_split_by_blocks <- function(points_sf, fold_id, test_fold, val_frac = 0.2, block_size_m = 4000) {
  # pontos de treino = todos exceto test_fold
  train_idx_all <- which(fold_id != test_fold)
  test_idx      <- which(fold_id == test_fold)
  
  # refaz bloco_id só para os pontos de treino (para selecionar val por bloco)
  bb <- st_bbox(points_sf[train_idx_all, ])
  xy <- st_coordinates(points_sf[train_idx_all, ])
  bx <- floor((xy[,1] - bb["xmin"]) / block_size_m)
  by <- floor((xy[,2] - bb["ymin"]) / block_size_m)
  block_id <- paste(bx, by, sep = "_")
  
  blocks <- unique(block_id)
  n_val_blocks <- max(1, floor(length(blocks) * val_frac))
  val_blocks <- sample(blocks, size = n_val_blocks)
  
  val_rel <- which(block_id %in% val_blocks)
  val_idx <- train_idx_all[val_rel]
  
  train_idx <- setdiff(train_idx_all, val_idx)
  
  list(train_idx = train_idx, val_idx = val_idx, test_idx = test_idx)
}

# 2.4) KNN espacial (somente em coords) dentro do treino
compute_neighbor_idx_train_only <- function(coords_train, K) {
  # coords_train: matrix (Ntrain, 2)
  # retorna matriz (Ntrain, K) com índices em [1..Ntrain]
  kn <- FNN::get.knn(coords_train, k = K)
  kn$nn.index
}

# =============================================================================
# 3) CRIAR RASTER STACK (COVARIÁVEIS) + NODATA
# =============================================================================

# CRS métrico simples (plano local). Para simulação é suficiente.
crs_m <- "EPSG:3857"

ncol <- ceiling(cfg$width_m / cfg$res_m)
nrow <- ceiling(cfg$height_m / cfg$res_m)

r0 <- rast(ncols = ncol, nrows = nrow,
           xmin = 0, xmax = cfg$width_m,
           ymin = 0, ymax = cfg$height_m,
           crs = crs_m)

# grade de coordenadas para gerar campos suaves
xy <- crds(r0, df = TRUE)  # data.frame com x,y por célula

# Função para criar um campo suave (mistura de senos/cossenos + ruído filtrado)
smooth_field <- function(x, y, freq1, freq2, noise_sd = 0.2) {
  z <- sin(2*pi*x/freq1) + cos(2*pi*y/freq2) + 0.5*sin(2*pi*(x+y)/(freq1+freq2))
  z <- z + rnorm(length(z), 0, noise_sd)
  as.numeric(scale(z))
}

cov_list <- vector("list", cfg$n_cov)
for (j in 1:cfg$n_cov) {
  freq1 <- sample(seq(1500, 8000, by = 500), 1)
  freq2 <- sample(seq(1500, 8000, by = 500), 1)
  vals <- smooth_field(xy$x, xy$y, freq1, freq2, noise_sd = 0.25)
  cov_list[[j]] <- setValues(r0, vals)
  names(cov_list[[j]]) <- paste0("cov_", sprintf("%02d", j))
}

cov_stack <- rast(cov_list)

# Cria uma máscara NoData (fração nodata_frac aleatória)
nodata_mask <- setValues(r0, runif(ncell(r0)) > cfg$nodata_frac)
cov_stack <- mask(cov_stack, nodata_mask, maskvalues = 0)  # onde mask=0 vira NA

plot(cov_stack)

# =============================================================================
# 4) AMOSTRAR PONTOS E EXTRAIR COVARIÁVEIS
# =============================================================================

# Amostra pontos em células válidas (não-NA)
valid_cells <- which(!is.na(values(nodata_mask)))
if (length(valid_cells) < cfg$n_points) stop("Poucas células válidas para amostrar pontos.")

sample_cells <- sample(valid_cells, cfg$n_points)
pts_xy <- xyFromCell(r0, sample_cells)

points_sf <- st_as_sf(data.frame(id = 1:cfg$n_points, x = pts_xy[,1], y = pts_xy[,2]),
                      coords = c("x","y"), crs = crs_m)

# Extrai covariáveis no pixel do ponto
X <- terra::extract(cov_stack, vect(points_sf)) %>%
  as.data.frame()

# extract retorna coluna ID (primeira) + covs
X <- X %>% dplyr::select(-1)

# Garantia: sem NA (se algum ponto caiu em NA, remova)
na_rows <- which(!complete.cases(X))
if (length(na_rows) > 0) {
  points_sf <- points_sf[-na_rows, ]
  X <- X[complete.cases(X), , drop = FALSE]
  message("Removi pontos com NA após extração: ", length(na_rows))
}

N <- nrow(X)

# =============================================================================
# 5) GERAR ALVO y (propriedade do solo) COM ESTRUTURA DSM REALISTA
# =============================================================================

# 5.1) Efeito das covariáveis (linear + alguns termos não-lineares leves)
X_mat <- as.matrix(X)
p <- ncol(X_mat)

# betas simulados
beta <- rnorm(p, 0, 1)
beta <- beta / sqrt(sum(beta^2)) * cfg$beta_strength

signal_cov <- as.numeric(X_mat %*% beta)

# 5.2) Tendência espacial suave (macro)
coords <- st_coordinates(points_sf)
trend <- cfg$trend_strength * as.numeric(scale(
  sin(2*pi*coords[,1]/12000) + cos(2*pi*coords[,2]/15000) + 0.3*sin(2*pi*(coords[,1]+coords[,2])/20000)
))

# 5.3) Ruído espacialmente correlacionado (aproximação: base radial por "anchors")
# Método rápido e bom para simulação:
# - cria M "pontos âncora" e soma kernels gaussianos
M <- 60
anchor_cells <- sample(valid_cells, M)
anchor_xy <- xyFromCell(r0, anchor_cells)
anchor_w  <- rnorm(M, 0, 1)

# kernel gaussiano
gauss_kernel <- function(d, range_m) exp(-(d^2) / (2 * (range_m^2)))

corr_component <- rep(0, N)
for (m in 1:M) {
  d <- sqrt((coords[,1] - anchor_xy[m,1])^2 + (coords[,2] - anchor_xy[m,2])^2)
  corr_component <- corr_component + anchor_w[m] * gauss_kernel(d, cfg$corr_range_m)
}
corr_component <- cfg$corr_sd * as.numeric(scale(corr_component))

# 5.4) Nugget (ruído iid)
nugget <- rnorm(N, 0, cfg$nugget_sd)

# 5.5) y final
y <- signal_cov + trend + corr_component + nugget

# Anexa ao sf
points_sf$y <- y

# =============================================================================
# 6) BLOCKED SPATIAL CV (fold_id) + SPLIT TRAIN/VAL/TEST
# =============================================================================

fold_id <- make_block_folds(points_sf, cfg$block_size_m, cfg$n_folds)
points_sf$fold_id <- fold_id

folds <- vector("list", cfg$n_folds)

for (f in 1:cfg$n_folds) {
  split <- make_train_val_split_by_blocks(
    points_sf = points_sf,
    fold_id   = fold_id,
    test_fold = f,
    val_frac  = 0.2,
    block_size_m = cfg$block_size_m
  )
  
  # dados do fold
  train_idx <- split$train_idx
  val_idx   <- split$val_idx
  test_idx  <- split$test_idx
  
  X_train <- X_mat[train_idx, , drop = FALSE]
  X_val   <- X_mat[val_idx,   , drop = FALSE]
  X_test  <- X_mat[test_idx,  , drop = FALSE]
  
  y_train <- y[train_idx]
  y_val   <- y[val_idx]
  y_test  <- y[test_idx]
  
  coords_train <- coords[train_idx, , drop = FALSE]
  coords_val   <- coords[val_idx,   , drop = FALSE]
  coords_test  <- coords[test_idx,  , drop = FALSE]
  
  # scaler APENAS no treino
  scaler <- fit_scaler(X_train, robust = cfg$use_robust_scaling)
  X_train_s <- apply_scaler(X_train, scaler)
  X_val_s   <- apply_scaler(X_val, scaler)
  X_test_s  <- apply_scaler(X_test, scaler)
  
  # neighbor_idx APENAS dentro do treino:
  # neighbor_idx é indexado de 1..Ntrain (não indices globais)
  neighbor_idx <- compute_neighbor_idx_train_only(coords_train, cfg$K)
  
  folds[[f]] <- list(
    fold = f,
    idx = list(train = train_idx, val = val_idx, test = test_idx),
    X = list(train = X_train_s, val = X_val_s, test = X_test_s),
    y = list(train = y_train,   val = y_val,   test = y_test),
    coords = list(train = coords_train, val = coords_val, test = coords_test),
    scaler = scaler,
    neighbor_idx_train = neighbor_idx
  )
}

# =============================================================================
# 7) (OPCIONAL) OBJETOS PARA "TILES" (PRÉ-TREINO / INFERÊNCIA)
# =============================================================================
# Aqui criamos uma grade de tiles (índices de células) para treinar auto-supervisionado
# ou para inferência em blocos sem estourar RAM.
make_tiles <- function(r, tile_ncol = 256, tile_nrow = 256, stride = 128) {
  stopifnot(inherits(r, "SpatRaster"))
  stopifnot(tile_ncol >= 1, tile_nrow >= 1, stride >= 1)
  
  nR <- nrow(r)
  nC <- ncol(r)
  rx <- res(r)[1]
  ry <- res(r)[2]
  
  tiles <- vector("list", length = 0)
  k <- 1
  
  for (row0 in seq(1, nR, by = stride)) {
    for (col0 in seq(1, nC, by = stride)) {
      
      row1 <- min(row0 + tile_nrow - 1, nR)
      col1 <- min(col0 + tile_ncol - 1, nC)
      
      # centros das células (colunas/linhas)
      x0c <- terra::xFromCol(r, col0)
      x1c <- terra::xFromCol(r, col1)
      y0c <- terra::yFromRow(r, row0)
      y1c <- terra::yFromRow(r, row1)
      
      # converter centro -> borda do pixel (extent real)
      xmin <- min(x0c, x1c) - rx / 2
      xmax <- max(x0c, x1c) + rx / 2
      ymin <- min(y0c, y1c) - ry / 2
      ymax <- max(y0c, y1c) + ry / 2
      
      e <- terra::ext(xmin, xmax, ymin, ymax)
      
      tiles[[k]] <- list(
        row0 = row0, row1 = row1,
        col0 = col0, col1 = col1,
        extent = e
      )
      k <- k + 1
    }
  }
  
  tiles
}

tiles <- make_tiles(cov_stack[[1]], tile_ncol = 256, tile_nrow = 256, stride = 128)
length(tiles)
tiles[[1]]$extent


# =============================================================================
# 8) EMPACOTAR TUDO EM UM OBJETO "sim" (PRONTO PARA EXPORT/USO)
# =============================================================================

sim <- list(
  cfg = cfg,
  rasters = cov_stack,
  nodata_mask = nodata_mask,
  points_sf = points_sf,
  X = X_mat,
  y = y,
  coords = coords,
  fold_id = fold_id,
  folds = folds,
  tiles = tiles
)

# Salva em disco (opcional)
saveRDS(sim, file = "./data/soilflux_simulation.rds")

# =============================================================================
# 9) SANITY CHECKS (BÁSICOS)
# =============================================================================
cat("\n--- SANITY CHECKS ---\n")
cat("N pontos:", nrow(sim$points_sf), "\n")
cat("N covariáveis:", nlyr(sim$rasters), "\n")
cat("Folds:", cfg$n_folds, "\n")

# checa vazamento: neighbor_idx sempre dentro do treino
for (f in 1:cfg$n_folds) {
  Ntr <- nrow(sim$folds[[f]]$X$train)
  stopifnot(all(sim$folds[[f]]$neighbor_idx_train >= 1))
  stopifnot(all(sim$folds[[f]]$neighbor_idx_train <= Ntr))
}
cat("OK: neighbor_idx_train dentro do treino em todos os folds.\n")

cat("Objeto final: sim (também salvo em soilflux_simulation.rds)\n")

metrics <- function(obs, pred) {
  rmse <- sqrt(mean((obs - pred)^2))
  r2   <- cor(obs, pred)^2
  rpiq <- IQR(obs) / rmse
  data.frame(R2 = r2, RMSE = rmse, RPIQ = rpiq)
}

library(randomForest)

res <- list()

for (f in 1:cfg$n_folds) {
  
  fold <- sim$folds[[f]]
  
  rf <- randomForest(
    x = fold$X$train,
    y = fold$y$train,
    ntree = 300
  )
  
  pred_test <- predict(rf, fold$X$test)
  
  res[[f]] <- metrics(fold$y$test, pred_test)
}

do.call(rbind, res) |> round(2)

library(FNN)

res_corr <- list()

for (f in 1:cfg$n_folds) {
  
  fold <- sim$folds[[f]]
  
  rf <- randomForest(
    x = fold$X$train,
    y = fold$y$train,
    ntree = 300
  )
  
  # predições
  pred_train <- predict(rf, fold$X$train)
  pred_test  <- predict(rf, fold$X$test)
  
  # resíduos no treino
  r_train <- fold$y$train - pred_train
  
  # KNN do teste em relação ao treino
  kn <- get.knnx(
    data = fold$coords$train,
    query = fold$coords$test,
    k = cfg$K
  )
  
  # correção espacial simples
  delta <- rowMeans(matrix(
    r_train[kn$nn.index],
    nrow = nrow(kn$nn.index)
  ))
  
  pred_corr <- pred_test + delta
  
  res_corr[[f]] <- metrics(fold$y$test, pred_corr)
}

do.call(rbind, res_corr) |> round(2)

# =============================================================================
# FIGURA: ganho espacial (erro antes/depois) no TESTE - por fold
# Usa sim (soilflux_simulation.rds) + RandomForest + correção KNN nos resíduos
# =============================================================================

library(sf)
library(dplyr)
library(FNN)
library(randomForest)
library(ggplot2)

sim <- readRDS("./data/soilflux_simulation.rds")

# Métricas rápidas (opcional, para checar números)
metrics <- function(obs, pred) {
  rmse <- sqrt(mean((obs - pred)^2))
  r2   <- cor(obs, pred)^2
  rpiq <- IQR(obs) / rmse
  data.frame(R2 = r2, RMSE = rmse, RPIQ = rpiq)
}

plot_gain_for_fold <- function(sim, fold = 1, K = NULL, ntree = 300) {
  
  if (is.null(K)) K <- sim$cfg$K
  fd <- sim$folds[[fold]]
  
  # ---- treina baseline RF
  rf <- randomForest(x = fd$X$train, y = fd$y$train, ntree = ntree)
  
  pred_train <- predict(rf, fd$X$train)
  pred_test  <- predict(rf, fd$X$test)
  
  # ---- resíduos no treino
  r_train <- fd$y$train - pred_train
  
  # ---- KNN do teste em relação ao treino (espacial)
  kn <- FNN::get.knnx(data = fd$coords$train, query = fd$coords$test, k = K)
  
  # correção simples: média dos resíduos dos K vizinhos
  delta <- rowMeans(matrix(r_train[kn$nn.index], nrow = nrow(kn$nn.index)))
  
  pred_test_corr <- pred_test + delta
  
  # ---- tabela com erros nos pontos de teste
  test_idx <- fd$idx$test
  pts_test <- sim$points_sf[test_idx, ]  # sf com geometria
  df <- pts_test %>%
    st_as_sf() %>%
    mutate(
      y_obs = fd$y$test,
      pred_base = pred_test,
      pred_corr = pred_test_corr,
      err_base = y_obs - pred_base,
      err_corr = y_obs - pred_corr,
      abs_base = abs(err_base),
      abs_corr = abs(err_corr),
      gain_abs = abs_base - abs_corr
    )
  
  # ---- imprime métricas (para você ver na hora)
  cat("\nFold:", fold, "\n")
  cat("Baseline RF:\n"); print(metrics(df$y_obs, df$pred_base))
  cat("RF + corr(KNN-residual):\n"); print(metrics(df$y_obs, df$pred_corr))
  
  # ---- 3 mapas (pontos)
  p1 <- ggplot(df) +
    geom_sf(aes(color = abs_base), size = 2) +
    scale_color_viridis_c() +
    ggtitle(paste0("TEST | Error | - Baseline RF (fold ", fold, ")")) +
    theme_minimal()
  
  p2 <- ggplot(df) +
    geom_sf(aes(color = abs_corr), size = 2) +
    scale_color_viridis_c() +
    ggtitle(paste0("TEST | Error | - RF + Spatial Residual Correction (fold ", fold, ")")) +
    theme_minimal()
  
  p3 <- ggplot(df) +
    geom_sf(aes(color = gain_abs), size = 2) +
    scale_color_gradient2(midpoint = 0) +
    ggtitle(paste0("GAIN = |err_base| - |err_corr| (positive = improved) (fold ", fold, ")")) +
    theme_minimal()
  
  list(df = df, p_base = p1, p_corr = p2, p_gain = p3)
}

# ---- rode para um fold (ex.: 1)
out <- plot_gain_for_fold(sim, fold = 1, ntree = 300)

print(out$p_base)
print(out$p_corr)
print(out$p_gain)

