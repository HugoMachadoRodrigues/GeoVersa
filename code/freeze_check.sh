#!/bin/zsh
# Pre-flight for a long campaign: proves the frozen code parses, checks clean,
# resolves paths portably, reproduces the committed artefacts, honours the
# geo_basis switch in both states, and leaves nothing uncommitted.
cd "$(dirname "$0")/.."
fail=0
ok()   { printf "  %-50s ok\n" "$1"; }
bad()  { printf "  %-50s FALHA\n" "$1"; fail=1; }
try()  { if ( eval "$2" ) >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

echo "########## VERIFICACAO DO CODIGO CONGELADO ##########"
echo "--- 1. parse ---"
bad_parse=0
for f in R/*.R code/*.R; do
  Rscript -e "invisible(parse('$f'))" >/dev/null 2>&1 || { bad "parse $f"; bad_parse=1; }
done
[ $bad_parse -eq 0 ] && ok "todos os arquivos R parseiam"

echo "--- 2. caminhos absolutos ---"
n=$(grep -l 'OneDrive-Personal' code/*.R R/*.R 2>/dev/null | grep -v project_paths.R | wc -l | tr -d ' ')
[ "$n" = "0" ] && ok "nenhum caminho absoluto fora de project_paths.R" || bad "$n script(s) com caminho absoluto"

echo "--- 3. resolvedor ---"
try "gv_root() da raiz"        'Rscript -e "source(\"code/project_paths.R\"); stopifnot(dir.exists(gv_root()))"'
try "gv_root() de code/"       'cd code && Rscript -e "source(\"project_paths.R\"); stopifnot(dir.exists(gv_root()))"'
try "gv_data_root() acha dados" 'Rscript -e "source(\"code/project_paths.R\"); stopifnot(dir.exists(gv_data_root()))"'

echo "--- 4. R CMD check ---"
rm -rf /tmp/gvchk; mkdir -p /tmp/gvchk
if R CMD build --no-build-vignettes . >/tmp/gvchk/build.log 2>&1; then ok "build"; else bad "build"; fi
TB=$(ls -t GeoVersa_*.tar.gz 2>/dev/null | head -1)
if [ -n "$TB" ]; then
  R CMD check --no-manual --no-vignettes -o /tmp/gvchk "$TB" >/tmp/gvchk/check.log 2>&1
  e=$(grep -c '^\* .*ERROR' /tmp/gvchk/check.log); w=$(grep -c '^\* .*WARNING' /tmp/gvchk/check.log)
  [ "$e" = "0" ] && [ "$w" = "0" ] && ok "check limpo (0 ERROR, 0 WARNING)" || bad "check: $e ERROR, $w WARNING"
  rm -f "$TB"
fi

echo "--- 5. geradores reproduzem ---"
before=$(md5 -q Text/ecomod_submission/tables/coverage.tex 2>/dev/null)
[ -n "$before" ] || bad "coverage.tex ilegivel antes do teste"
try "tables_manuscript.R" 'Rscript code/tables_manuscript.R'
after=$(md5 -q Text/ecomod_submission/tables/coverage.tex 2>/dev/null)
[ -n "$after" ] && [ "$before" = "$after" ] && ok "tabelas byte a byte" || bad "tabelas mudaram ou ilegiveis"
for g in figures_uncertainty figures_schematic figures_protocols figures_graphical_abstract figures_distance; do
  try "$g.R" "Rscript code/$g.R"
done

echo "--- 6. geo_basis nos dois estados ---"
for gb in TRUE FALSE; do
  rm -rf results/gbchk_$gb
  DBT_SCENARIO=random DBT_PROTOCOLS=DesignBased DBT_MODELS=DBTARF DBT_N_ITER=1 \
  DBT_SAMPLE_SIZE=150 DBT_TRAIN_SEED=7 DBT_ENSEMBLE_TOP_K=1 DBT_GEO_BASIS=$gb \
  DBT_RESULTS_DIR=results/gbchk_$gb Rscript code/run_dbtarf_wadoux_comparison.R >/dev/null 2>&1
done
Rscript -e '
g <- function(d) { p <- file.path("results", d, "dbtarf_all_results.csv")
  if (!file.exists(p)) return(NA); x <- read.csv(p); x$RMSE[x$model == "DBTARF"][1] }
a <- g("gbchk_TRUE"); b <- g("gbchk_FALSE")
cat(sprintf("  TRUE %.4f | FALSE %.4f -> %s\n", a, b,
            if (!is.na(a) && !is.na(b) && a != b) "switch ativo" else "SEM EFEITO"))
quit(status = if (!is.na(a) && !is.na(b) && a != b) 0 else 1)' && ok "geo_basis altera o modelo" || bad "geo_basis sem efeito"

echo "--- 7. no-op com geo_basis=TRUE (contra a campanha preservada) ---"
rm -rf results/noop_final
DBT_SCENARIO=random DBT_PROTOCOLS=DesignBased DBT_MODELS=RF,Cubist,DBTARF \
DBT_SAMPLE_SIZE=500 DBT_N_ITER=1 DBT_TRAIN_SEED=123 DBT_ENSEMBLE_TOP_K=1 \
DBT_FOLD_CORES=6 DBT_RESULTS_DIR=results/noop_final \
Rscript code/run_dbtarf_wadoux_comparison.R >/dev/null 2>&1
Rscript -e '
rd <- function(d, f) { p <- file.path("results", d, f)
  if (!file.exists(p)) stop("ausente: ", p); x <- read.csv(p); x[x$iteration == 1, ] }
a <- rd("noop_final","dbtarf_all_results.csv"); b <- rd("nested_random_confirm30","dbtarf_all_results.csv")
ua <- rd("noop_final","dbtarf_uncertainty.csv");  ub <- rd("nested_random_confirm30","dbtarf_uncertainty.csv")
same <- TRUE
for (m in intersect(a$model, b$model)) {
  d <- abs(a$RMSE[a$model==m] - b$RMSE[b$model==m])
  if (!isTRUE(d < 1e-9)) { cat("  RMSE difere:", m, "\n"); same <- FALSE } }
for (m in intersect(ua$model, ub$model)) {
  d <- abs(ua$MPIW90[ua$model==m] - ub$MPIW90[ub$model==m])
  if (!isTRUE(d < 1e-9)) { cat("  MPIW difere:", m, "\n"); same <- FALSE } }
quit(status = if (same) 0 else 1)' && ok "reproduz a campanha preservada" || bad "diverge da campanha preservada"

echo "--- 8. git ---"
[ -z "$(git status --porcelain | grep -v '^?? results/')" ] && ok "sem alteracoes pendentes" || bad "ha alteracoes nao commitadas"

echo
echo "########## $([ $fail -eq 0 ] && echo 'APTO A DISPARAR' || echo 'NAO DISPARAR') ##########"
exit $fail
