#!/bin/zsh
# The full campaign set behind the revised manuscript. Results are written
# outside the synced folder and linked back in, so a day of checkpointing does
# not fight the file provider.
#
# Order matters only in that the three main designs come first: they are what
# the manuscript's tables need, and if anything has to be cut it should be the
# secondaries.
cd "$(dirname "$0")/.."
EXT="$HOME/geoversa_results"
COMMIT=$(git rev-parse --short HEAD)

run() {                       # run <dir> <extra env assignments...>
  local dir="$1"; shift
  # A directory left over from an aborted attempt would be written into rather
  # than replaced, mixing two campaigns under one name. Anything already there
  # is set aside before the run starts, never silently reused.
  if [ -d "$EXT/$dir" ] && [ -n "$(ls -A "$EXT/$dir" 2>/dev/null)" ]; then
    mv "$EXT/$dir" "$EXT/STALE_${dir}_$(date '+%Y%m%d_%H%M%S')"
    echo "  (dados anteriores de $dir movidos para STALE_*)"
  fi
  mkdir -p "$EXT/$dir"
  ln -sfn "$EXT/$dir" "results/$dir"
  echo "===== $dir  $(date '+%m-%d %H:%M') ====="
  env "$@" DBT_RESULTS_DIR="results/$dir" \
    Rscript code/run_dbtarf_wadoux_comparison.R 2>&1 | grep -E 'Done|Error' | tail -2
  local n=$(tail -n +2 "results/$dir/dbtarf_all_results.csv" 2>/dev/null | wc -l | tr -d ' ')
  echo "----- $dir concluido $(date '+%m-%d %H:%M'), ${n:-0} linhas -----"
}

BASE=(DBT_PROTOCOLS=DesignBased DBT_MODELS=RF,Cubist,DBTARF
      DBT_N_ITER=30 DBT_TRAIN_SEED=123 DBT_FOLD_CORES=8)

echo "########## CAMPANHAS | commit $COMMIT | $(date '+%m-%d %H:%M') ##########"

# --- principais: o que as tabelas do manuscrito consomem
for SC in random regular_grid clustered_random; do
  run "final_${SC}_confirm30" "${BASE[@]}" DBT_SCENARIO=$SC DBT_SAMPLE_SIZE=500
done

# --- ablacao: o que o variograma acrescenta sobre coordenadas simples
run "abl_nogeo_random" "${BASE[@]}" DBT_SCENARIO=random DBT_SAMPLE_SIZE=500 \
    DBT_GEO_BASIS=FALSE

# --- sensibilidade ao tamanho amostral, nos dois desenhos que carregam a tese
for N in 200 800; do
  for SC in random clustered_random; do
    run "size_n${N}_${SC}" "${BASE[@]}" DBT_SCENARIO=$SC DBT_SAMPLE_SIZE=$N
  done
done

echo
echo "########## TODAS CONCLUIDAS $(date '+%m-%d %H:%M') ##########"
for d in final_random_confirm30 final_regular_grid_confirm30 final_clustered_random_confirm30 \
         abl_nogeo_random size_n200_random size_n200_clustered_random \
         size_n800_random size_n800_clustered_random; do
  n=$(tail -n +2 "results/$d/dbtarf_all_results.csv" 2>/dev/null | wc -l | tr -d ' ')
  printf "  %-36s %s linhas\n" "$d" "${n:-0}"
done
