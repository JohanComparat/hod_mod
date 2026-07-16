#!/usr/bin/env bash
# =============================================================================
# Campaign preflight / completion audit.
#
# Answers "has the campaign landed, and can I run collect_and_plot.sh yet?".
# collect_and_plot.sh runs under `set -euo pipefail` and calls plotters with an
# explicit --out-dir, so a single missing out-dir aborts it half-way through
# (leaving docs/_images/ partially refreshed).  Run this first.
#
#   HOD_MOD_RESULTS=/home/comparat/data/hod_mod_results ./oarsub/campaign_status.sh
#   VTAG=v0.31 HOD_MOD_RESULTS=~/data/hod_mod_results_v0.31 ./oarsub/campaign_status.sh
#
# Exit 0 = every required output present; 1 = something missing (details above).
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RES="${HOD_MOD_RESULTS:?set HOD_MOD_RESULTS}"
VTAG="${VTAG:-v0.31}"   # keep in sync with _campaign_env.sh / submit_campaign.sh

miss=0
ok()   { printf "  \033[32m OK \033[0m %-52s %s\n" "$1" "$2"; }
bad()  { printf "  \033[31mMISS\033[0m %-52s %s\n" "$1" "$2"; miss=$((miss+1)); }

# A directory "exists" for our purposes only if it has files in it — an empty
# dir left by a crashed job is worse than an absent one (the plotters would
# read it and produce empty/stale figures).
check_dir() {
    local d="$1" what="$2" n
    if [ -d "$RES/$d" ]; then
        n=$(find "$RES/$d" -type f 2>/dev/null | wc -l)
        if [ "$n" -gt 0 ]; then ok "$d" "$n files, newest $(find "$RES/$d" -type f -printf '%TY-%Tm-%Td %TH:%TM\n' 2>/dev/null | sort -r | head -1)"
        else bad "$d" "EMPTY dir (crashed job?)"; fi
    else
        bad "$d" "absent — $what"
    fi
}

echo "== campaign status  VTAG=$VTAG  RESULTS=$RES"
echo
echo "-- production galaxy joint fits (param file: production_mcmc.txt) --"
check_dir "bgs_zm15_joint_wp_ngal_${VTAG}"      "production_mcmc family not finished/synced"
check_dir "bgs_full_joint_fixedzm15_${VTAG}"    "full_joint family not finished/synced"
check_dir "bgs_full_joint_allparams_${VTAG}"    "full_joint family not finished/synced"
check_dir "bgs_comparat2025_${VTAG}"            "production_mcmc family not finished/synced"
check_dir "bgs_zm15_thresh_joint_${VTAG}"       "production_mcmc family not finished/synced"

echo
echo "-- Comparat+2025 X-ray w_theta presets (Family C, 5 dedicated scripts) --"
for p in gas-shape gas-temp gas-full agn-occ agn-lum; do
    check_dir "fits/comparat2025_fixedZM15_${p}_${VTAG}" "fit_comparat2025_${p}.sh not finished/synced"
done

echo
echo "-- literature benchmarks (benchmarks_map/mcmc: NO --out-dir, written in place) --"
check_dir "benchmarks" "benchmarks families not finished/synced"
if [ -d "$RES/benchmarks" ]; then
    printf "       %s model subdirs, %s PNGs\n" \
        "$(find "$RES/benchmarks" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)" \
        "$(find "$RES/benchmarks" -name '*.png' 2>/dev/null | wc -l)"
fi

echo
echo "-- forecasts (forecasts.txt: written in place via results_root()) --"
for d in sensitivity_fisher tier2_forecast tier3_forecast stage4_forecast; do
    check_dir "$d" "forecasts family not finished/synced"
done

echo
if [ "$miss" -eq 0 ]; then
    echo "== ALL REQUIRED OUTPUTS PRESENT — safe to run:"
    echo "   HOD_MOD_RESULTS=$RES ./oarsub/collect_and_plot.sh"
    exit 0
else
    echo "== $miss required output(s) MISSING — collect_and_plot.sh would abort part-way."
    echo "   Wait for the jobs to finish, then re-sync:  ./oarsub/pull_results.sh"
    exit 1
fi
