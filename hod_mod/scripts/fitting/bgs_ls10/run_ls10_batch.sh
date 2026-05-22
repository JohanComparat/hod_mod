#!/usr/bin/env bash
# Run More+2015 HOD fits on all available LS10 stellar mass bins.
#
# Each bin is fit sequentially with MAP + MCMC (emcee).
# Results are written to results/bgs_ls10/mstar{XX}/.
#
# Usage:
#   bash scripts/fitting/bgs_ls10/run_ls10_batch.sh [--vary-cosmo] [--map-only]
#
# Pass any extra flags through to fit_ls10_more2015.py, e.g.:
#   bash run_ls10_batch.sh --vary-cosmo --n-steps 3000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_SCRIPT="${SCRIPT_DIR}/fit_ls10_more2015.py"
EXTRA_ARGS="$@"

# LS10 mass bins with sys-comb wp files
MSTAR_BINS=(9.0 9.5 10.0 11.0 11.25 11.5)

echo "=== LS10 More+2015 batch fitting ==="
echo "Bins: ${MSTAR_BINS[*]}"
echo "Extra args: ${EXTRA_ARGS:-none}"
echo ""

for mstar in "${MSTAR_BINS[@]}"; do
    echo "--------------------------------------"
    echo " Fitting  log10(M*/M_sun) > ${mstar}"
    echo "--------------------------------------"
    python "${FIT_SCRIPT}" --mstar "${mstar}" ${EXTRA_ARGS} \
        2>&1 | tee "results/bgs_ls10/mstar${mstar}_fit.log" || true
    echo ""
done

echo "=== Batch complete ==="
