#!/bin/zsh
# =============================================================================
# run_size_sweep.sh
#
# Sample-size sensitivity for the coverage result. Six campaigns, 30 repetitions
# each, ordered so the decisive ones finish first.
#
# The clustered design does not take its calibration size from DBT_SAMPLE_SIZE.
# It takes it from DBT_N_PSU times DBT_M_PER_PSU. An earlier sweep set only
# DBT_SAMPLE_SIZE, which under that design changes nothing but the validation
# draw, and so compared one experiment with itself at two measurement
# precisions. Every campaign here therefore states its calibration size through
# the knob that actually governs it, and asserts afterwards that the recorded
# n_train came out as asked.
#
# Two clustered arms, because "a bigger sample" means two different things:
#
#   density  fixed 20 primary units, more points inside each. The spatial
#            footprint is identical at every size, so this isolates sample size
#            and tests the paper's mechanism directly: more conformity scores
#            drawn from the same places should not describe errors elsewhere.
#
#   spread   fixed 25 points per unit, more units. This is how a field campaign
#            usually grows with budget, and the footprint grows with it, so
#            coverage may improve partly by covering more of the domain. It
#            answers the sampling reviewer's question rather than the mechanism
#            one, and the two must not be reported as the same experiment.
#
# The n = 500 midpoint is shared: 20 x 25 is both arms at once, already run as
# final_clustered_random_confirm30.
#
# Validation is held at 500 everywhere via DBT_VAL_SIZE, so the precision of the
# measurement no longer moves with the quantity being measured.
#
# Usage:  zsh code/run_size_sweep.sh
# =============================================================================

set -u
cd "${0:a:h}/.." || exit 1
EXT="$HOME/geoversa_results"
mkdir -p "$EXT"

WANT_ROWS=120          # 30 iterations x 4 model rows
LOG="$EXT/size_sweep.log"

say() { print -r -- "$@" | tee -a "$LOG"; return 0 }

# A campaign counts as done only with the full row count AND a recorded commit.
# Row count alone would accept a run whose provenance was never written.
is_complete() {
  local d="$EXT/$1" f="$EXT/$1/dbtarf_all_results.csv" c="$EXT/$1/dbtarf_config.csv"
  [[ -f "$f" && -f "$c" ]] || return 1
  local n
  n=$(tail -n +2 "$f" 2>/dev/null | wc -l | tr -d ' ')
  [[ "$n" -ge "$WANT_ROWS" ]] || return 1
  grep -q 'git_commit' "$c" || return 1
  return 0
}

# Post-condition. The whole reason this file exists is that a campaign can run
# to completion with a calibration sample that is not the size it was asked for.
assert_calib() {
  local dir="$1" want="$2"
  local got
  got=$(python3 - "$EXT/$dir/dbtarf_diagnostics.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
sizes = sorted({int(r["n_train"]) for r in rows})
print(",".join(str(s) for s in sizes))
PY
)
  if [[ "$got" != "$want" ]]; then
    say "  !! $dir pediu calibracao de $want e treinou com $got  -- CAMPANHA INVALIDA"
    return 1
  fi
  say "  ok  $dir treinou com $got, como pedido"
  return 0
}

# run <dir> <scenario> <calib> <extra env assignments...>
run() {
  local dir="$1" sc="$2" calib="$3"; shift 3
  if is_complete "$dir"; then
    say "== $dir ja completo, pulando"
    return 0
  fi
  if [[ -d "$EXT/$dir" ]]; then
    mv "$EXT/$dir" "$EXT/STALE_${dir}_$(date '+%Y%m%d_%H%M%S')"
    say "   (parcial anterior movido para STALE_*)"
  fi
  mkdir -p "$EXT/$dir"
  say "===== $dir  calib=$calib  $(date '+%m-%d %H:%M') ====="
  env "$@" \
    DBT_SCENARIO="$sc" DBT_PROTOCOLS=DesignBased DBT_MODELS=RF,Cubist,DBTARF \
    DBT_VAL_SIZE=500 DBT_N_ITER=30 DBT_TRAIN_SEED=123 DBT_FOLD_CORES=8 \
    DBT_ENSEMBLE_TOP_K=1 DBT_RESULTS_DIR="$EXT/$dir" \
    Rscript code/run_dbtarf_wadoux_comparison.R 2>&1 | grep -E 'Done|Error' | tail -2
  say "----- $dir terminou $(date '+%m-%d %H:%M'), $(tail -n +2 "$EXT/$dir/dbtarf_all_results.csv" 2>/dev/null | wc -l | tr -d ' ') linhas -----"
  assert_calib "$dir" "$calib"
}

say ""
say "######## varredura iniciada $(date '+%m-%d %H:%M'), commit $(git rev-parse --short HEAD) ########"

# --- decisive first: the clustered contrast at the cheap size ----------------
run clus_dens_n200   clustered_random 200 DBT_SAMPLE_SIZE=200 DBT_N_PSU=20 DBT_M_PER_PSU=10
run clus_spread_n200 clustered_random 200 DBT_SAMPLE_SIZE=200 DBT_N_PSU=8  DBT_M_PER_PSU=25

# --- then the expensive size -------------------------------------------------
run clus_dens_n800   clustered_random 800 DBT_SAMPLE_SIZE=800 DBT_N_PSU=20 DBT_M_PER_PSU=40
run clus_spread_n800 clustered_random 800 DBT_SAMPLE_SIZE=800 DBT_N_PSU=32 DBT_M_PER_PSU=25

# --- the shared midpoint, re-run for provenance ------------------------------
# 20 x 25 = 500 already exists as final_clustered_random_confirm30 and is the
# hinge of both arms. That campaign predates the commit that started recording
# n_psu and m_per_psu, so its geometry is inferred from the sampler defaults
# rather than written down. Every other cell in the table states its geometry;
# the one both arms pivot on should not be the exception, and inferring a
# parameter from a default is exactly what produced the retraction this sweep
# exists to repair.
run clus_mid_n500 clustered_random 500 DBT_SAMPLE_SIZE=500 DBT_N_PSU=20 DBT_M_PER_PSU=25

# --- the representative control, last: it was never the design in question ---
# Re-run at n = 200 rather than reusing size_n200_random, whose validation draw
# was 200 and so is not measured at the same precision as everything else here.
run rand_n200 random 200 DBT_SAMPLE_SIZE=200
run rand_n800 random 800 DBT_SAMPLE_SIZE=800

say ""
say "######## varredura concluida $(date '+%m-%d %H:%M') ########"
Rscript code/tables_manuscript.R 2>&1 | grep -iE 'sample-size|dropped|stale' | tee -a "$LOG"
