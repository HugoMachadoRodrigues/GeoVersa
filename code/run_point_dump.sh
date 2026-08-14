#!/bin/zsh
# =============================================================================
# run_point_dump.sh
#
# Campaigns for the error-against-distance figure. They differ from the others
# only in DBT_DUMP_POINTS, which makes eval_uncertainty_dbt write one row per
# validation point: its coordinates, its distance to the nearest calibration
# point, the observation, the prediction and both interval bounds.
#
# Ten repetitions rather than thirty. The figure bins over points, and ten
# repetitions already give five thousand of them per design; the repetition
# count matters here only because points within a repetition share one fitted
# model and are therefore not independent, which the figure's uncertainty
# bands have to respect.
#
# Three designs spanning the transfer gap measured in Table 5:
#   random 500      gap 1.01, the design whose intervals are valid
#   clustered 20/25 gap 1.16, the design of Wadoux et al.
#   clustered 8/25  gap 1.40, the worst geometry in the sweep
#
# Usage:  zsh code/run_point_dump.sh
# =============================================================================

set -u
cd "${0:a:h}/.." || exit 1
EXT="$HOME/geoversa_results"
LOG="$EXT/point_dump.log"
WANT=10

say() { print -r -- "$@" | tee -a "$LOG"; return 0 }

done_already() {
  local f="$EXT/$1/dbtarf_points.csv"
  [[ -f "$f" ]] || return 1
  local k
  k=$(tail -n +2 "$f" | cut -d, -f1 | sort -u | wc -l | tr -d ' ')
  [[ "$k" -ge "$WANT" ]]
}

run() {
  local dir="$1" sc="$2"; shift 2
  if done_already "$dir"; then say "== $dir ja completo, pulando"; return 0; fi
  [[ -d "$EXT/$dir" ]] && mv "$EXT/$dir" "$EXT/STALE_${dir}_$(date '+%Y%m%d_%H%M%S')"
  mkdir -p "$EXT/$dir"
  say "===== $dir  $(date '+%m-%d %H:%M') ====="
  env "$@" \
    DBT_SCENARIO="$sc" DBT_PROTOCOLS=DesignBased DBT_MODELS=DBTARF \
    DBT_VAL_SIZE=500 DBT_N_ITER=$WANT DBT_TRAIN_SEED=123 DBT_FOLD_CORES=8 \
    DBT_ENSEMBLE_TOP_K=1 DBT_DUMP_POINTS=1 DBT_RESULTS_DIR="$EXT/$dir" \
    Rscript code/run_dbtarf_wadoux_comparison.R 2>&1 | grep -E 'Done|Error' | tail -2
  say "----- $dir terminou $(date '+%m-%d %H:%M'), $(tail -n +2 "$EXT/$dir/dbtarf_points.csv" 2>/dev/null | wc -l | tr -d ' ') pontos -----"
}

say ""
say "######## despejo iniciado $(date '+%m-%d %H:%M'), commit $(git rev-parse --short HEAD) ########"

run pts_clus_mid  clustered_random DBT_SAMPLE_SIZE=500 DBT_N_PSU=20 DBT_M_PER_PSU=25
run pts_clus_wide clustered_random DBT_SAMPLE_SIZE=200 DBT_N_PSU=8  DBT_M_PER_PSU=25
run pts_random    random           DBT_SAMPLE_SIZE=500

say "######## despejo concluido $(date '+%m-%d %H:%M') ########"
