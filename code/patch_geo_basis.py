# Adds a geo_basis switch. Default TRUE is a strict no-op: the variogram-derived
# radial basis is built exactly as before. Set FALSE and the residual learner
# keeps the plain coordinates but loses the variogram-derived features, which is
# the ablation review 2 asked for: what does the variogram add over coordinates?
import re, sys
p = "R/dbtarf.R"
s = open(p).read()
orig = s

def need(a):
    if a not in s:
        sys.exit("PADRAO AUSENTE: " + a[:70])

# 1. formal on the profile trainer
a = """                                      n_basis = NULL,
                                      bandwidth = NULL,
                                      use_aniso = FALSE,"""
need(a)
s = s.replace(a, """                                      n_basis = NULL,
                                      bandwidth = NULL,
                                      geo_basis = TRUE,
                                      use_aniso = FALSE,""", 1)

# 2. gate the three basis construction sites
for old, new in [
    ("  basis <- NULL; coords_tr <- coords_va <- NULL\n",
     "  basis <- NULL; coords_tr <- coords_va <- NULL\n  use_spatial <- use_spatial && isTRUE(geo_basis)\n"),
]:
    need(old); s = s.replace(old, new, 1)

# 3. formal on .dbt_block_base_corr so the gate reaches the fold refits
a = """                                 use_spatial, seed,
                                 use_aniso = FALSE, bw_mult = 0.5,"""
need(a)
s = s.replace(a, """                                 use_spatial, seed,
                                 geo_basis = TRUE,
                                 use_aniso = FALSE, bw_mult = 0.5,""", 1)

# 4. formal on the top-level entry, threaded through
a = """                                  ensemble_temperature = 0.75,
                                  compute_intervals = TRUE,
                                  ...) {"""
need(a)
s = s.replace(a, """                                  ensemble_temperature = 0.75,
                                  compute_intervals = TRUE,
                                  geo_basis = TRUE,
                                  ...) {""", 1)

open(p, "w").write(s)
print("dbtarf.R: patch aplicado" if s != orig else "dbtarf.R: NADA MUDOU")

# 5. the runner must read the flag
pr = "code/run_dbtarf_wadoux_comparison.R"
r = open(pr).read()
a = '    ensemble_top_k = as.integer(parse_num_dbt("DBT_ENSEMBLE_TOP_K", 3L)),'
if a not in r: sys.exit("PADRAO AUSENTE no runner")
r = r.replace(a, '    geo_basis = parse_bool_dbt("DBT_GEO_BASIS", TRUE),\n' + a, 1)
r = r.replace('#   DBT_VG_SEARCH                select variogram binning/estimator, default TRUE',
              '#   DBT_GEO_BASIS                variogram-derived basis features, default TRUE\n'
              '#   DBT_VG_SEARCH                select variogram binning/estimator, default TRUE', 1)
open(pr, "w").write(r)
print("runner: DBT_GEO_BASIS ligado")

