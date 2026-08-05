#!/bin/zsh
# Waits for the nested campaign to finish, then applies the geo_basis patch and
# runs the secondary experiments. Nothing touches R/dbtarf.R while the main
# campaign is still opening fresh R processes against it.
cd "/Users/rodrigues.h/Library/CloudStorage/OneDrive-Personal/GeoVersa"

echo "[fila] aguardando a campanha principal ($(date '+%m-%d %H:%M'))"
while pgrep -f 'run_dbtarf_wadoux_comparison' >/dev/null 2>&1; do sleep 120; done
echo "[fila] principal encerrada ($(date '+%m-%d %H:%M'))"

for SC in random regular_grid clustered_random; do
  n=$(tail -n +2 results/nested_${SC}_confirm30/dbtarf_all_results.csv 2>/dev/null | wc -l | tr -d ' ')
  echo "[fila]   nested_${SC}: ${n:-0} linhas"
done

echo "[fila] aplicando o patch geo_basis"
python3 /tmp/geo_basis_patch.py || { echo "[fila] PATCH FALHOU, abortando"; exit 1; }
Rscript -e 'source("R/dbtarf.R"); cat("[fila] sintaxe ok\n")' || { echo "[fila] SINTAXE QUEBRADA, abortando"; exit 1; }

echo "[fila] fumaca: geo_basis=TRUE deve reproduzir a campanha principal"
DBT_SCENARIO=random DBT_PROTOCOLS=DesignBased DBT_MODELS=DBTARF DBT_N_ITER=1 \
DBT_SAMPLE_SIZE=500 DBT_TRAIN_SEED=123 DBT_ENSEMBLE_TOP_K=1 DBT_FOLD_CORES=6 \
DBT_RESULTS_DIR=results/geo_noop_check Rscript code/run_dbtarf_wadoux_comparison.R >/dev/null 2>&1
python3 - <<'PYX'
import csv, os
def rmse(d):
    p=f"results/{d}/dbtarf_all_results.csv"
    if not os.path.exists(p): return None
    for r in csv.DictReader(open(p)):
        if r.get("model")=="DBTARF" and r.get("iteration")=="1": return round(float(r["RMSE"]),6)
a,b = rmse("geo_noop_check"), rmse("nested_random_confirm30")
print(f"[fila] no-op check: patch={a} principal={b} -> {'IDENTICO' if a==b else 'DIVERGIU'}")
PYX

echo "[fila] 1/5 ablacao sem base geoestatistica"
DBT_SCENARIO=random DBT_PROTOCOLS=DesignBased DBT_MODELS=RF,DBTARF \
DBT_SAMPLE_SIZE=500 DBT_N_ITER=30 DBT_TRAIN_SEED=123 DBT_FOLD_CORES=6 \
DBT_ENSEMBLE_TOP_K=1 DBT_GEO_BASIS=FALSE \
DBT_RESULTS_DIR=results/abl_nogeo_random Rscript code/run_dbtarf_wadoux_comparison.R 2>&1 | tail -2

i=2
for N in 200 800; do
  for SC in random clustered_random; do
    echo "[fila] ${i}/5 n=${N} ${SC} ($(date '+%m-%d %H:%M'))"
    DBT_SCENARIO=$SC DBT_PROTOCOLS=DesignBased DBT_MODELS=RF,Cubist,DBTARF \
    DBT_SAMPLE_SIZE=$N DBT_N_ITER=30 DBT_TRAIN_SEED=123 DBT_FOLD_CORES=6 \
    DBT_ENSEMBLE_TOP_K=1 DBT_RESULTS_DIR=results/size_n${N}_${SC} \
    Rscript code/run_dbtarf_wadoux_comparison.R 2>&1 | tail -2
    i=$((i+1))
  done
done
echo "########## SECUNDARIOS CONCLUIDOS $(date '+%m-%d %H:%M') ##########"
