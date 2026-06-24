#!/usr/bin/env bash
# run_wp_survey.sh — complete WP-only quickfit survey for BGS M* > 10^10 M_sun
#
# Grid: 6 HOD models × 2 profiles × 5 scale cuts = 60 MAP fits
# Physics: IA + baryon fraction + off-centering (standard set)
# Output: results/bgs_multiprobe/mstar10.0_wp_{model}_{profile}_rp{rpmin}/
#
# Usage:
#   bash scripts/fitting/bgs_ls10/run_wp_survey.sh            # sequential
#   bash scripts/fitting/bgs_ls10/run_wp_survey.sh --parallel # 4 jobs in parallel

set -euo pipefail

SCRIPT="hod_mod/scripts/fitting/bgs_ls10/fit_bgs_multiprobe.py"
MSTAR=10.0
PROBES="wp"
COMMON="--mstar ${MSTAR} --probes ${PROBES} --use-ia --use-baryon-fraction --use-offcentering --map-only"

MODELS=(more2015 zheng2007 aum zu_mandelbaum15 vanuitert16 zacharegkas25)
PROFILES=(nfw einasto)
RP_MINS=(0.30 0.05 0.04 0.02 0.01)

PARALLEL=false
[[ "${1:-}" == "--parallel" ]] && PARALLEL=true

run() {
    local model=$1 profile=$2 rp_min=$3
    echo "--- $(date '+%H:%M:%S')  ${model}  ${profile}  rp_min=${rp_min} ---"
    python "${SCRIPT}" ${COMMON} \
        --hod-model  "${model}" \
        --profile    "${profile}" \
        --rp-min-wp  "${rp_min}" \
        && echo "    OK" \
        || echo "    FAILED (${model} ${profile} rp${rp_min})"
}

if $PARALLEL; then
    # Run up to 4 jobs simultaneously
    for model in "${MODELS[@]}"; do
        for profile in "${PROFILES[@]}"; do
            for rp_min in "${RP_MINS[@]}"; do
                run "${model}" "${profile}" "${rp_min}" &
                # Throttle to 4 concurrent jobs
                while [[ $(jobs -r | wc -l) -ge 4 ]]; do sleep 5; done
            done
        done
    done
    wait
else
    for model in "${MODELS[@]}"; do
        for profile in "${PROFILES[@]}"; do
            for rp_min in "${RP_MINS[@]}"; do
                run "${model}" "${profile}" "${rp_min}"
            done
        done
    done
fi

echo ""
echo "=== Survey complete $(date) ==="
echo ""

# Print summary table
echo "model                    profile   rp_min   chi2      ndof  chi2/ndof"
echo "----------------------------------------------------------------------"
for model in "${MODELS[@]}"; do
    for profile in "${PROFILES[@]}"; do
        for rp_min in "${RP_MINS[@]}"; do
            rp_tag=$(python3 -c "print(f'rp{int(round(${rp_min}*1000)):03d}')")
            jfile="results/bgs_multiprobe/mstar${MSTAR}_${PROBES}_${model}_${profile}_${rp_tag}/map_result.json"
            if [[ -f "$jfile" ]]; then
                python3 - "$jfile" "$model" "$profile" "$rp_min" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
m, p, rp = sys.argv[2], sys.argv[3], sys.argv[4]
chi2, ndof = r["chi2"], r["ndof"]
print(f"{m:24s} {p:9s} {rp:8s} {chi2:9.2f} {ndof:5d}  {chi2/ndof:8.3f}")
PY
            else
                printf "%-24s %-9s %-8s  MISSING\n" "$model" "$profile" "$rp_min"
            fi
        done
    done
done
