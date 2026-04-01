# =============================================================================
# SoilFLUX-KrigingNet (R torch) vs RF + XGB + Cubist
# - Usa objeto sim (soilflux_simulation.rds) já criado
# - Blocked CV (folds dentro de sim$folds)
# - Métricas no TEST: R2, RMSE, RPIQ
# =============================================================================

rm(list = ls())
set.seed(123)

# -----------------------------
# 0) Pacotes
# -----------------------------
# pkgs <- c("torch", "dplyr", "ggplot2", "randomForest", "xgboost", "Cubist")
# to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
# if (length(to_install) > 0) install.packages(to_install)

# install.packages("torch")
# torch::install_torch(type = "cpu")

library(torch)
library(dplyr)
library(ggplot2)
library(randomForest)
library(xgboost)
library(Cubist)

# -----------------------------
# 1) Carrega sim
# -----------------------------
sim <- readRDS("../data/soilflux_simulation.rds")

to_float_tensor <- function(x, device = "cpu") {
  torch_tensor(x, dtype = torch_float(), device = device)
}

clone_state_dict <- function(state_dict) {
  lapply(state_dict, function(x) x$clone())
}

# =============================================================================
# 2) Métricas
# =============================================================================
metrics <- function(obs, pred) {
  rmse <- sqrt(mean((obs - pred)^2))
  r2   <- cor(obs, pred)^2
  rpiq <- IQR(obs) / (rmse + 1e-12)
  data.frame(R2 = r2, RMSE = rmse, RPIQ = rpiq)
}


# --- cdist robusto (se torch_cdist não existir, faz na mão)
cdist_safe <- function(A, B) {
  # A: (B,2), B: (N,2)
  if (exists("torch_cdist", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_cdist(A, B))
  }
  # fallback: ||a-b|| = sqrt( (ax-bx)^2 + (ay-by)^2 )
  ax <- A[,1]$unsqueeze(2)  # (B,1)
  ay <- A[,2]$unsqueeze(2)
  bx <- B[,1]$unsqueeze(1)  # (1,N)
  by <- B[,2]$unsqueeze(1)
  torch_sqrt((ax - bx)^2 + (ay - by)^2)
}



# --- pega indices do retorno de torch_topk em qualquer versão
extract_topk_indices <- function(topk_out) {
  # Em algumas versões: list(values=..., indices=...)
  if (is.list(topk_out) && "indices" %in% names(topk_out)) return(topk_out$indices)
  
  # Em outras: list(values=..., index=...) ou list(values=..., idx=...)
  if (is.list(topk_out) && "index" %in% names(topk_out)) return(topk_out$index)
  if (is.list(topk_out) && "idx" %in% names(topk_out))   return(topk_out$idx)
  
  # Em várias versões: retorno é lista posicional: [[1]]=values, [[2]]=indices
  if (is.list(topk_out) && length(topk_out) >= 2) return(topk_out[[2]])
  
  stop("Não consegui extrair índices do torch_topk(). Rode str(topk_out) para inspecionar.")
}


# --- topk robusto para "menores distâncias"
topk_smallest_idx <- function(d, K) {
  out <- try(torch_topk(d, k = K, dim = 2, largest = FALSE), silent = TRUE)
  if (!inherits(out, "try-error")) return(extract_topk_indices(out))
  
  out2 <- torch_topk(-d, k = K, dim = 2, largest = TRUE)
  extract_topk_indices(out2)
}

# --- reshape robusto (evita $view / $reshape que mudam por versão)
reshape_safe <- function(x, shape) {
  # tenta torch_reshape (mais comum)
  if (exists("torch_reshape", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_reshape(x, shape))
  }
  # fallback: às vezes existe torch_view
  if (exists("torch_view", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_view(x, shape))
  }
  stop("Não encontrei torch_reshape/torch_view nesta versão do torch. Atualize {torch}.")
}

# --- flatten super-robusto (sem depender de torch_flatten)
flatten_safe <- function(x) {
  # reshape para vetor (é o mais compatível)
  if (exists("torch_reshape", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_reshape(x, c(-1)))
  }
  # fallback raro
  if (exists("torch_view", where = asNamespace("torch"), inherits = FALSE)) {
    return(torch_view(x, c(-1)))
  }
  stop("Não encontrei torch_reshape/torch_view nesta versão do torch.")
}

# =============================================================================
# 3) Modelos baselines (RF, XGB, Cubist)
# =============================================================================
fit_predict_rf <- function(X_train, y_train, X_test, ntree = 500) {
  rf <- randomForest(x = X_train, y = y_train, ntree = ntree)
  as.numeric(predict(rf, X_test))
}

fit_predict_xgb <- function(X_train, y_train, X_val, y_val, X_test,
                            nrounds = 5000, eta = 0.03, max_depth = 6,
                            subsample = 0.8, colsample_bytree = 0.8,
                            min_child_weight = 1, reg_lambda = 1) {
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dval   <- xgb.DMatrix(data = X_val,   label = y_val)
  dtest  <- xgb.DMatrix(data = X_test)
  
  params <- list(
    objective = "reg:squarederror",
    eta = eta,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    lambda = reg_lambda
  )
  
  xgbm <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = list(val = dval),
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  as.numeric(predict(xgbm, dtest))
}

fit_predict_cubist <- function(X_train, y_train, X_test, committees = 50, neighbors = 5) {
  cb <- Cubist::cubist(x = X_train, y = y_train,
                       committees = committees, neighbors = neighbors)
  as.numeric(predict(cb, X_test))
}

# =============================================================================
# 4) SoilFLUX-KrigingNet mínimo em {torch} (R)
#    - Tabular encoder + Coord encoder (Fourier) + gated fusion
#    - Quantile head q10,q50,q90 (com ordem garantida)
#    - Neural kriging correction com pesos aprendidos
# =============================================================================

# ---- Fourier features
FourierFeatures <- nn_module(
  "FourierFeatures",
  initialize = function(num_freq = 32, max_freq = 10) {
    self$num_freq <- num_freq
    freqs <- torch_logspace(0, log10(max_freq), steps = num_freq)
    self$register_buffer("freqs", freqs)
  },
  forward = function(coords) {
    # coords: (B,2)
    # x: (B,F,2)
    freq_view <- reshape_safe(self$freqs, c(1, self$num_freq, 1))
    x <- coords$unsqueeze(2) * freq_view
    sinv <- torch_sin(2 * pi * x)
    cosv <- torch_cos(2 * pi * x)
    
    # concatena na dimensão F (dim=2) => (B,2F,2)
    out <- torch_cat(list(sinv, cosv), dim = 2)
    reshape_safe(out, c(coords$size(1), -1))  # (B, 2F*2)
  }
)
# ---- MLP helper
make_mlp <- function(in_dim, hidden, out_dim, dropout = 0.10) {
  layers <- list()
  prev <- in_dim
  
  for (h in hidden) {
    layers[[length(layers) + 1]] <- nn_linear(prev, h)
    layers[[length(layers) + 1]] <- nn_gelu()
    layers[[length(layers) + 1]] <- nn_layer_norm(h)
    layers[[length(layers) + 1]] <- nn_dropout(dropout)
    prev <- h
  }
  
  layers[[length(layers) + 1]] <- nn_linear(prev, out_dim)
  
  do.call(nn_sequential, layers)
}

# ---- Quantile head com ordem garantida
QuantileHead <- nn_module(
  "QuantileHead",
  initialize = function(d = 256) {
    self$net <- nn_sequential(
      nn_linear(d, 128), nn_gelu(),
      nn_linear(128, 64), nn_gelu(),
      nn_linear(64, 3)
    )
  },
  forward = function(z) {
    raw <- self$net(z)               # (B,3)
    q50 <- raw[,2]                   # coluna 2
    d1  <- nnf_softplus(raw[,1])     # >=0
    d2  <- nnf_softplus(raw[,3])     # >=0
    q10 <- q50 - d1
    q90 <- q50 + d2
    list(q10 = q10, q50 = q50, q90 = q90)
  }
)

# ---- Neural Kriging
NeuralKriging <- nn_module(
  "NeuralKriging",
  initialize = function(d = 256, proj_d = 64, init_ell = 1000) {
    self$proj <- nn_linear(d, proj_d, bias = FALSE)
    self$log_ell <- nn_parameter(torch_log(torch_tensor(init_ell)))
    self$scale <- 1 / sqrt(proj_d)
  },
  forward = function(z_i, coords_i, z_n, coords_n, r_n) {
    # z_i: (B,d)
    # coords_i: (B,2)
    # z_n: (B,K,d)
    # coords_n: (B,K,2)
    # r_n: (B,K)
    
    dist <- torch_norm(coords_i$unsqueeze(2) - coords_n, dim = 3)  # (B,K)
    ell  <- nnf_softplus(self$log_ell) + 1e-6
    
    qi <- self$proj(z_i)                 # (B,proj_d)
    qn <- self$proj(z_n)                 # (B,K,proj_d)
    sim <- torch_sum(qn * qi$unsqueeze(2), dim = 3) * self$scale  # (B,K)
    
    w <- nnf_softmax(-dist / ell + sim, dim = 2) # (B,K)
    delta <- torch_sum(w * r_n, dim = 2)         # (B)
    list(delta = delta, w = w)
  }
)

# ---- SoilFluxKrigingNet mínimo
SoilFluxKrigingNetMin <- nn_module(
  "SoilFluxKrigingNetMin",
  initialize = function(c_tab, d = 256, num_freq = 32) {
    self$d <- d
    self$enc_tab <- make_mlp(c_tab, hidden = c(512), out_dim = d, dropout = 0.10)
    
    self$ff <- FourierFeatures(num_freq = num_freq, max_freq = 10)
    self$enc_coord <- make_mlp(2 * num_freq * 2, hidden = c(128), out_dim = 128, dropout = 0.10)
    self$proj_coord <- nn_linear(128, d)
    
    # gate: 2 inputs (tab + coord)
    self$gate <- nn_sequential(
      nn_linear(d * 2, 128), nn_gelu(),
      nn_linear(128, 2)
    )
    
    self$head <- QuantileHead(d = d)
    self$krig <- NeuralKriging(d = d, proj_d = 64, init_ell = 1000)
  },
  
  encode = function(x_tab, coords) {
    zt <- self$enc_tab(x_tab)   # (B,d)
    zf <- self$ff(coords)       # (B, 2F*2)
    zc <- self$proj_coord(self$enc_coord(zf))  # (B,d)
    
    zcat <- torch_cat(list(zt, zc), dim = 2)   # (B,2d)
    a <- nnf_softmax(self$gate(zcat), dim = 2) # (B,2)
    
    z <- zt * a[,1]$unsqueeze(2) + zc * a[,2]$unsqueeze(2)
    list(z = z, gate = a)
  },
  
  forward_base = function(x_tab, coords) {
    enc <- self$encode(x_tab, coords)
    q <- self$head(enc$z)
    list(q = q, z = enc$z, gate = enc$gate)
  },
  
  forward_with_kriging = function(x_tab, coords, z_n, coords_n, r_n) {
    base <- self$forward_base(x_tab, coords)
    k <- self$krig(base$z, coords, z_n, coords_n, r_n)
    q <- base$q
    q_corr <- list(
      q10 = q$q10 + k$delta,
      q50 = q$q50 + k$delta,
      q90 = q$q90 + k$delta
    )
    list(q = q_corr, z = base$z, delta = k$delta, gate = base$gate)
  }
)

# ---- pinball loss
pinball_loss <- function(y, q, tau) {
  e <- y - q
  torch_mean(torch_maximum(tau * e, (tau - 1) * e))
}

quantile_loss <- function(y, q10, q50, q90) {
  pinball_loss(y, q10, 0.1) + pinball_loss(y, q50, 0.5) + pinball_loss(y, q90, 0.9)
}

# =============================================================================
# 5) Treino SoilFLUX em R torch (memory bank por epoch)
# =============================================================================

# helper: cria batches de índices
make_batches <- function(n, batch_size = 256) {
  idx <- sample.int(n)
  split(idx, ceiling(seq_along(idx) / batch_size))
}

# monta memory bank (Z, R, C) usando forward_base no treino
build_memory_bank <- function(model, X_train, coords_train, y_train, device = "cpu", batch_size = 1024) {
  model$eval()
  n <- nrow(X_train)
  
  Z_list <- list()
  R_list <- list()
  C_list <- list()
  
  with_no_grad({
    for (s in seq(1, n, by = batch_size)) {
      e <- min(s + batch_size - 1, n)
      
      xb <- to_float_tensor(X_train[s:e, , drop = FALSE], device = device)
      cb <- to_float_tensor(coords_train[s:e, , drop = FALSE], device = device)
      yb <- to_float_tensor(y_train[s:e], device = device)
      
      out <- model$forward_base(xb, cb)
      q50 <- out$q$q50
      r <- yb - q50
      
      Z_list[[length(Z_list) + 1]] <- out$z$cpu()
      R_list[[length(R_list) + 1]] <- r$cpu()
      C_list[[length(C_list) + 1]] <- cb$cpu()
    }
  })
  
  Z <- torch_cat(Z_list, dim = 1)  # (N,d)
  R <- torch_cat(R_list, dim = 1)  # (N)
  C <- torch_cat(C_list, dim = 1)  # (N,2)
  
  list(Z = Z, R = R, C = C)
}

# Treina 1 fold
train_soilflux_one_fold <- function(fd,
                                    epochs = 200,
                                    lr = 2e-4,
                                    wd = 1e-3,
                                    batch_size = 256,
                                    patience = 20,
                                    device = NULL) {
  
  if (is.null(device)) device <- "cpu"
  
  # -----------------------------
  # Data
  # -----------------------------
  Xtr <- fd$X$train; ytr <- fd$y$train; Ctr <- fd$coords$train
  Xva <- fd$X$val;   yva <- fd$y$val;   Cva <- fd$coords$val
  Xte <- fd$X$test;  yte <- fd$y$test;  Cte <- fd$coords$test
  
  neigh_train <- fd$neighbor_idx_train   # (Ntrain,K) 1-based
  
  # -----------------------------
  # Model + optimizer
  # -----------------------------
  c_tab <- ncol(Xtr)
  model <- SoilFluxKrigingNetMin(c_tab = c_tab, d = 256, num_freq = 32)
  model$to(device = device)
  
  opt <- optim_adamw(model$parameters, lr = lr, weight_decay = wd)
  
  best_val <- Inf
  best_state <- NULL
  bad <- 0
  
  # -----------------------------
  # Train loop
  # -----------------------------
  for (ep in 1:epochs) {
    
    # 1) memory bank do treino (Zmem, Rmem, Cmem)
    bank <- build_memory_bank(model, Xtr, Ctr, ytr, device = device, batch_size = 1024)
    Zmem <- bank$Z$to(device = device)   # (Ntrain,d)
    Rmem <- bank$R$to(device = device)   # (Ntrain)
    Cmem <- bank$C$to(device = device)   # (Ntrain,2)
    
    model$train()
    batches <- make_batches(nrow(Xtr), batch_size = batch_size)
    loss_epoch <- 0
    
    Ktr <- ncol(neigh_train)
    
    for (b in batches) {
      
      xb <- to_float_tensor(Xtr[b, , drop = FALSE], device = device)
      cb <- to_float_tensor(Ctr[b, , drop = FALSE], device = device)
      yb <- to_float_tensor(ytr[b], device = device)
      
      # vizinhos já pré-computados no treino (sem KNN aqui!)
      nb <- neigh_train[b, , drop = FALSE]      # (B,K) indices 1..Ntrain
      nb_t <- torch_tensor(as.vector(nb), dtype = torch_long(), device = device)
      
      # gather neighbors from memory bank -> reshape robusto
      zn <- reshape_safe(Zmem$index_select(1, nb_t), c(length(b), Ktr, -1))
      rn <- reshape_safe(Rmem$index_select(1, nb_t), c(length(b), Ktr))
      cn <- reshape_safe(Cmem$index_select(1, nb_t), c(length(b), Ktr, 2))
      
      out <- model$forward_with_kriging(xb, cb, zn, cn, rn)
      
      loss <- quantile_loss(yb, out$q$q10, out$q$q50, out$q$q90)
      
      opt$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 2.0)
      opt$step()
      
      loss_epoch <- loss_epoch + loss$item()
    }
    
    # 2) validação: forward_base (sem kriging) para early stopping
    model$eval()
    with_no_grad({
      xb <- to_float_tensor(Xva, device = device)
      cb <- to_float_tensor(Cva, device = device)
      yb <- to_float_tensor(yva, device = device)
      
      outv <- model$forward_base(xb, cb)
      vloss <- quantile_loss(yb, outv$q$q10, outv$q$q50, outv$q$q90)$item()
    })
    
    if (ep %% 10 == 0) {
      cat(sprintf("Epoch %d | train_loss=%.4f | val_loss=%.4f\n",
                  ep, loss_epoch / length(batches), vloss))
    }
    
    if (vloss < best_val) {
      best_val <- vloss
      best_state <- clone_state_dict(model$state_dict())
      bad <- 0
    } else {
      bad <- bad + 1
      if (bad >= patience) break
    }
  }
  
  # -----------------------------
  # Load best + TEST inference
  # -----------------------------
  model$load_state_dict(best_state)
  model$eval()
  
  # memory bank final (treino)
  bank <- build_memory_bank(model, Xtr, Ctr, ytr, device = device, batch_size = 1024)
  Zmem <- bank$Z$to(device = device)
  Rmem <- bank$R$to(device = device)
  Cmem <- bank$C$to(device = device)
  
  K <- ncol(neigh_train)
  preds <- numeric(nrow(Xte))
  
  with_no_grad({
    for (s in seq(1, nrow(Xte), by = 512)) {
      e <- min(s + 512 - 1, nrow(Xte))
      B <- e - s + 1
      
      xb <- to_float_tensor(Xte[s:e, , drop = FALSE], device = device)
      cb <- to_float_tensor(Cte[s:e, , drop = FALSE], device = device)
      
      # KNN espacial do teste contra coords de treino
      d   <- cdist_safe(cb, Cmem)
      knn <- topk_smallest_idx(d, K)
      
      if (is.null(knn)) stop("knn é NULL: topk_smallest_idx() não está retornando índices.")
      
      nb_flat <- flatten_safe(knn)
      if (is.null(nb_flat)) stop("nb_flat é NULL: flatten_safe() falhou.")
      nb_flat <- nb_flat$to(dtype = torch_long())
      
      zn <- reshape_safe(Zmem$index_select(1, nb_flat), c(B, K, -1))
      rn <- reshape_safe(Rmem$index_select(1, nb_flat), c(B, K))
      cn <- reshape_safe(Cmem$index_select(1, nb_flat), c(B, K, 2))
      
      out <- model$forward_with_kriging(xb, cb, zn, cn, rn)
      preds[s:e] <- as.numeric(out$q$q50$cpu())
    }
  })
  
  list(
    model = model,
    pred_test = preds,
    metrics_test = metrics(yte, preds),
    device = device
  )
}

# =============================================================================
# 6) Runner: roda todos os folds e compara tudo
# =============================================================================
run_all_folds <- function(sim,
                          rf_ntree = 500,
                          cubist_committees = 50,
                          cubist_neighbors = 5,
                          xgb_params = list(),
                          soilflux_params = list()) {
  
  nfold <- sim$cfg$n_folds
  results <- list()
  
  for (f in 1:nfold) {
    cat("\n=============================\n")
    cat("FOLD", f, "\n")
    cat("=============================\n")
    
    fd <- sim$folds[[f]]
    
    # ---- baselines
    pred_rf  <- fit_predict_rf(fd$X$train, fd$y$train, fd$X$test, ntree = rf_ntree)
    met_rf   <- metrics(fd$y$test, pred_rf);  met_rf$model <- "RF"
    
    pred_cb  <- fit_predict_cubist(fd$X$train, fd$y$train, fd$X$test,
                                   committees = cubist_committees, neighbors = cubist_neighbors)
    met_cb   <- metrics(fd$y$test, pred_cb);  met_cb$model <- "Cubist"
    
    # XGB usa validação (fd$X$val) para early stopping
    pred_xgb <- do.call(fit_predict_xgb, c(
      list(X_train = fd$X$train, y_train = fd$y$train,
           X_val = fd$X$val, y_val = fd$y$val,
           X_test = fd$X$test),
      xgb_params
    ))
    met_xgb  <- metrics(fd$y$test, pred_xgb); met_xgb$model <- "XGB"
    
    # ---- SoilFLUX (torch)
    sf_out <- do.call(train_soilflux_one_fold, c(list(fd = fd), soilflux_params))
    met_sf <- sf_out$metrics_test; met_sf$model <- "SoilFLUX-KrigingNet"
    
    # empacota
    df <- bind_rows(met_rf, met_xgb, met_cb, met_sf) %>%
      mutate(fold = f)
    
    print(df)
    
    results[[f]] <- df
  }
  
  bind_rows(results)
}

# =============================================================================
# 7) Rodar tudo (com defaults bons)
# =============================================================================

xgb_params <- list(
  nrounds = 5000,
  eta = 0.03,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  reg_lambda = 1
)

soilflux_params <- list(
  epochs = 200,
  lr = 2e-4,
  wd = 1e-3,
  batch_size = 256,
  patience = 20,
  device = "cpu"  # auto: cuda se disponível
)

fd <- sim$folds[[1]]
sf_out <- train_soilflux_one_fold(fd, epochs = 50, device = "cpu")
sf_out$metrics_test

all_res <- run_all_folds(sim,
                         rf_ntree = 500,
                         cubist_committees = 50,
                         cubist_neighbors = 5,
                         xgb_params = xgb_params,
                         soilflux_params = soilflux_params)

# resumo (média por modelo)
summary_res <- all_res %>%
  group_by(model) %>%
  summarise(
    R2_mean = mean(R2), R2_sd = sd(R2),
    RMSE_mean = mean(RMSE), RMSE_sd = sd(RMSE),
    RPIQ_mean = mean(RPIQ), RPIQ_sd = sd(RPIQ),
    .groups = "drop"
  ) %>%
  arrange(desc(R2_mean))

cat("\n\n===== SUMMARY (mean ± sd across folds) =====\n")
print(summary_res)

# salva resultados
# write.csv(all_res, "results_by_fold.csv", row.names = FALSE)
# write.csv(summary_res, "results_summary.csv", row.names = FALSE)

# =============================================================================
# 8) Gráfico rápido (opcional)
# =============================================================================
p <- ggplot(all_res, aes(x = model, y = R2)) +
  geom_boxplot() +
  theme_minimal() +
  coord_flip() +
  ggtitle("Blocked CV: R² distribution by model")

print(p)
