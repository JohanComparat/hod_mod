#!/usr/bin/env bash
# =============================================================================
# Submit one campaign family on dahu.  Each array family is one OAR array job
# over the matching param file, sized per oarsub/README.md.  Family C
# (Comparat+2025 X-ray MAP presets) and the two full-joint fits keep their
# dedicated fit_*.sh scripts.
#
# Usage:
#   ./oarsub/submit_campaign.sh <PROJECT> <family>
#     family in { benchmarks_map | benchmarks_mcmc | production | forecasts
#                 | comparat2025 | comparat2025_ecf | full_joint | all }
#   ./oarsub/submit_campaign.sh <PROJECT> benchmarks_map --devel   # smoke (1 line, 30 min)
#
#   VTAG=v0.4 ./oarsub/submit_campaign.sh your-oar-project all
#
# Prereqs on dahu: repo pulled at the campaign commit; env in run_job.sh edited
# for your account; data staged (see README).  Smoke-test with --devel first.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROJECT="${1:?usage: submit_campaign.sh <PROJECT> <family> [--devel]}"
FAMILY="${2:?family in benchmarks_map|benchmarks_mcmc|production|forecasts|comparat2025|comparat2025_ecf|full_joint|all}"
DEVEL="${3:-}"

# VTAG selects the campaign (tree + pinned P(k) backend). It is resolved HERE, in
# the submitting shell, and forwarded as an ARGUMENT: OAR does not propagate the
# environment to the node, so exporting it would be silently dropped.
VTAG="${VTAG:-v0.4}"
echo "[submit] VTAG=$VTAG"
WRAP="./oarsub/run_job.sh --vtag ${VTAG}"

# Log stem.  The #OAR directives inside run_job.sh and the standalone fit_*.sh
# are literal text -- they cannot interpolate VTAG -- so every one of them wrote
# the same v03_rerun / c2025_<preset> stem regardless of campaign, and a second
# campaign's logs landed indistinguishably on top of the first's.  Command-line
# --name/--stdout/--stderr override the in-script directives, so the stem is
# built here instead.  Dots are stripped: v0.31 -> v031.
TAGSAFE="${VTAG//./}"

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

# --name / --stdout / --stderr for a given stem.  Emitted as words so callers can
# splice it into an oarsub invocation.
log_flags () {
  local stem="$1"
  printf -- '--name hodmod_%s --stdout oarsub/logs/%%jobid%%.%s.out --stderr oarsub/logs/%%jobid%%.%s.err' \
    "$stem" "$stem" "$stem"
}

submit_array () {
  local fam="$1" extra="${2:-}"
  local pf="${PARAM[$fam]}" res="${LFLAG[$fam]}"
  if [[ -n "$DEVEL" ]]; then
    # devel: single line, short walltime — quick end-to-end smoke of the family.
    local first; first="$(grep -vE '^\s*#|^\s*$' "$pf" | head -1)"
    echo "[devel] $fam: $first"
    oarsub --project "$PROJECT" -t devel -l "/nodes=1/core=8,walltime=00:30:00" \
      $(log_flags "${TAGSAFE}_${fam}_devel") \
      -S "$WRAP $first"
  else
    echo "[submit] $fam  array over $pf  -l $res  $extra"
    oarsub --project "$PROJECT" -l "$res" $extra \
      $(log_flags "${TAGSAFE}_${fam}") \
      --array-param-file "$pf" -S "$WRAP"
  fi
}

# The standalone families (C and full_joint) take their resources from their own
# #OAR -l directives rather than from LFLAG, so --devel has to be applied here
# too -- otherwise `all --devel` sent twelve multi-hour jobs to the real queue
# while the four array families were correctly smoke-testing.  In devel mode a
# family submits ONE job, matching what submit_array does with a param file.
DEVEL_L="-t devel -l /nodes=1/core=8,walltime=00:30:00"

# Family C.  $1 = "" (plain) or "ecf" (fold the physical ECF chain in, 0.4.0).
# The five scripts carry their own #OAR -l (16 cores, 4-18 h depending on
# preset); only the log stem is overridden here.  MAP/L-BFGS-B and NOT
# resumable, so no besteffort.
submit_comparat2025 () {
  local variant="${1:-}" suffix="" arg="" presets=(gas-shape gas-temp gas-full agn-occ agn-lum)
  if [ -n "$variant" ]; then suffix="_${variant}"; arg=" ${variant}"; fi
  # A 30-min devel slot cannot finish any of these MAPs, and is not meant to:
  # it exercises the env, the data staging and the ZM15 reference guard, which
  # is what failed twice in the v0.3 campaign.
  [[ -n "$DEVEL" ]] && presets=(gas-shape)
  for s in "${presets[@]}"; do
    echo "[${DEVEL:+devel}${DEVEL:-submit}] comparat2025${suffix} $s"
    # shellcheck disable=SC2086
    oarsub --project "$PROJECT" ${DEVEL:+$DEVEL_L} \
      $(log_flags "c2025_${TAGSAFE}_${s//-/_}${suffix}${DEVEL:+_devel}") \
      -S "./oarsub/fit_comparat2025_${s//-/_}.sh ${VTAG}${arg}"
  done
}

case "$FAMILY" in
  benchmarks_map|forecasts) submit_array "$FAMILY" ;;
  # benchmarks_mcmc: the 8 h wall is what stranded seven ZM15 chains in the v0.3
  # campaign.  Those samplers are HDF-backed now, and run_benchmark refuses to
  # discard an *unfinished* chain even under --force-mcmc, so a besteffort
  # restart resumes rather than restarts.
  benchmarks_mcmc) submit_array benchmarks_mcmc "-t besteffort -t idempotent" ;;
  production) submit_array production "-t besteffort -t idempotent" ;;   # resumable chains
  comparat2025)     submit_comparat2025 ;;
  comparat2025_ecf) submit_comparat2025 ecf ;;
  full_joint)
    # shellcheck disable=SC2086
    oarsub --project "$PROJECT" ${DEVEL:+$DEVEL_L} \
      $(log_flags "fulljoint_${TAGSAFE}_fixedzm15${DEVEL:+_devel}") \
      -S "./oarsub/fit_bgs_full_joint_fixedzm15_mcmc.sh ${VTAG}"
    # devel smokes one job per family; allparams is the 48 h one, so the
    # 6 h fixedzm15 above is the representative.
    if [[ -z "$DEVEL" ]]; then
      oarsub --project "$PROJECT" \
        $(log_flags "fulljoint_${TAGSAFE}_allparams") \
        -S "./oarsub/fit_bgs_full_joint_allparams_mcmc.sh ${VTAG}"
    fi ;;
  all)
    # $DEVEL is forwarded: without it `all --devel` silently submitted the whole
    # campaign to the real queue.
    for f in benchmarks_map benchmarks_mcmc production forecasts \
             comparat2025 comparat2025_ecf full_joint; do
      VTAG="$VTAG" "$0" "$PROJECT" "$f" $DEVEL
    done ;;
  *) echo "unknown family: $FAMILY" >&2; exit 2 ;;
esac
echo "monitor:  oarstat -u \$USER   |   logs in oarsub/logs/"
