#!/usr/bin/env bash
# =============================================================================
# Submit the v0.3 "re-run every w_p / ΔΣ / Σ_y / w_θ calculation" campaign on
# dahu.  Each family is one OAR array job over the matching param file, sized
# per oarsub/README.md.  Family C (Comparat+2025 X-ray MAP presets) keeps its
# five dedicated fit_comparat2025_*.sh scripts.
#
# Usage:
#   ./oarsub/submit_campaign.sh <PROJECT> <family>
#     family ∈ { benchmarks_map | benchmarks_mcmc | production | forecasts
#                | comparat2025 | full_joint | all }
#   ./oarsub/submit_campaign.sh <PROJECT> benchmarks_map --devel   # smoke (1 line, 30 min)
#
# Prereqs on dahu: repo pulled at the v0.3 commit; env in run_job.sh edited for
# your account; data staged (see README).  Smoke-test with --devel first.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT="${1:?usage: submit_campaign.sh <PROJECT> <family> [--devel]}"
FAMILY="${2:?family ∈ benchmarks_map|benchmarks_mcmc|production|forecasts|comparat2025|full_joint|all}"
DEVEL="${3:-}"

WRAP="./oarsub/run_job.sh"
declare -A LFLAG=(
  [benchmarks_map]="/nodes=1/core=8,walltime=02:00:00"
  [benchmarks_mcmc]="/nodes=1/core=8,walltime=08:00:00"
  [production]="/nodes=1/core=16,walltime=24:00:00"
  [forecasts]="/nodes=1/core=16,walltime=24:00:00"
)
declare -A PARAM=(
  [benchmarks_map]="oarsub/params/benchmarks_map.txt"
  [benchmarks_mcmc]="oarsub/params/benchmarks_mcmc.txt"
  [production]="oarsub/params/production_mcmc.txt"
  [forecasts]="oarsub/params/forecasts.txt"
)

submit_array () {
  local fam="$1" extra="${2:-}"
  local pf="${PARAM[$fam]}" res="${LFLAG[$fam]}"
  if [[ -n "$DEVEL" ]]; then
    # devel: single line, short walltime — quick end-to-end smoke of the family.
    local first; first="$(grep -vE '^\s*#|^\s*$' "$pf" | head -1)"
    echo "[devel] $fam: $first"
    oarsub --project "$PROJECT" -t devel -l "/nodes=1/core=8,walltime=00:30:00" \
      -S "$WRAP $first"
  else
    echo "[submit] $fam  array over $pf  -l $res  $extra"
    oarsub --project "$PROJECT" -l "$res" $extra \
      --array-param-file "$pf" -S "$WRAP"
  fi
}

case "$FAMILY" in
  benchmarks_map|benchmarks_mcmc|forecasts) submit_array "$FAMILY" ;;
  production) submit_array production "-t besteffort -t idempotent" ;;   # resumable chains
  comparat2025)
    for s in gas-shape gas-temp gas-full agn-occ agn-lum; do
      echo "[submit] comparat2025 $s"; oarsub --project "$PROJECT" -S "./oarsub/fit_comparat2025_${s//-/_}.sh"
    done ;;
  full_joint)
    oarsub --project "$PROJECT" -S ./oarsub/fit_bgs_full_joint_fixedzm15_mcmc.sh
    oarsub --project "$PROJECT" -S ./oarsub/fit_bgs_full_joint_allparams_mcmc.sh ;;
  all)
    "$0" "$PROJECT" benchmarks_map;  "$0" "$PROJECT" benchmarks_mcmc
    "$0" "$PROJECT" production;       "$0" "$PROJECT" forecasts
    "$0" "$PROJECT" comparat2025;     "$0" "$PROJECT" full_joint ;;
  *) echo "unknown family: $FAMILY" >&2; exit 2 ;;
esac
echo "monitor:  oarstat -u \$USER   |   logs in oarsub/logs/"
