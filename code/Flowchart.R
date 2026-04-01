# install.packages("DiagrammeR")
library(DiagrammeR)

grViz("
digraph soilflux {
  graph [rankdir=LR, splines=true, nodesep=0.35, ranksep=0.5]
  node  [shape=box, style=rounded, fontsize=10]

  inputs [label='Inputs\\n\\n• Covariate stack (rasters)\\n• Sample points: (x_i,y_i), y_i']
  cv     [label='Spatial blocked CV\\n\\nTrain / Val / Test\\n(no spatial leakage)', style='rounded,dashed']
  neigh  [label='Neighbor graph (train)\\n\\nK nearest neighbors\\nwithin TRAIN only']

  tab    [label='Tabular encoder f_θ(x)\\n(MLP)\\n\\nLearn global soil–covariate signal']
  coord  [label='Coordinate encoder\\nFourier features φ(s)\\n→ MLP → z_coord']
  gate   [label='Gated fusion\\n\\nz = α·z_tab + (1−α)·z_coord\\n(learned gate)', style='rounded,dashed']

  qhead  [label='Quantile head\\n\\nPredict q10, q50, q90\\n(ordered quantiles)']
  resid  [label='Residuals (TRAIN)\\n\\nr_i = y_i − q50_i\\nstored in memory bank']
  nk     [label='Neural kriging correction (Δ)\\n\\nΔ(s_i)= Σ_j w_ij · r_j\\nw_ij = softmax( −d/ℓ + sim(z_i,z_j) )\\n(distance + latent similarity)', style='rounded,dashed']

  out    [label='Final prediction\\n\\nqτ,final = qτ + Δ\\n(τ ∈ {0.1,0.5,0.9})\\n\\nOutputs:\\n• mean/median map\\n• uncertainty (IQR, PI)']

  invA   [label='INOVAÇÃO A\\nDecomposição explícita:\\nGlobal (covariáveis) + Local (resíduo espacial)', style='rounded,dashed', fontsize=9]
  invB   [label='INOVAÇÃO B\\nKriging neural contextual (não estacionário):\\npesos = distância + similaridade latente', style='rounded,dashed', fontsize=9]
  invC   [label='INOVAÇÃO C\\nValidação como stress-test\\nde generalização espacial', style='rounded,dashed', fontsize=9]

  inputs -> tab
  inputs -> coord
  cv -> neigh
  tab -> qhead
  coord -> gate
  tab -> gate
  gate -> nk
  qhead -> resid
  neigh -> resid
  resid -> nk
  nk -> out

  gate -> invA [style=dashed, arrowhead=none]
  nk   -> invB [style=dashed, arrowhead=none]
  cv   -> invC [style=dashed, arrowhead=none]
}
")
