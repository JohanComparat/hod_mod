#!/usr/bin/env bash
# Run More+2015 HOD fits on all Uchuu mock stellar mass bins.
#
# Cosmological parameters are varied with Planck 2018 3σ Gaussian prior by
# default.  Add --wide-cosmo to use wide uniform priors instead.
#
# Usage:
#   bash scripts/fitting/mocks/run_mocks_batch.sh [--wide-cosmo] [--map-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_SCRIPT="${SCRIPT_DIR}/fit_mocks_more2015.py"
EXTRA_ARGS="$@"

MSTAR_BINS=(9.29 9.78 10.24 10.45 10.65 10.84 11.03 11.22 11.39)

echo "=== Mock More+2015 batch fitting ==="
echo "Bins: ${MSTAR_BINS[*]}"
echo "Extra args: ${EXTRA_ARGS:-none}"
echo ""

mkdir -p results/mocks

for mstar in "${MSTAR_BINS[@]}"; do
    echo "--------------------------------------"
    echo " Fitting  log10(M*/M_sun) > ${mstar}"
    echo "--------------------------------------"
    python "${FIT_SCRIPT}" --mstar "${mstar}" ${EXTRA_ARGS} \
        2>&1 | tee "results/mocks/mstar${mstar}_fit.log" || true
    echo ""
done

echo "=== Batch complete ==="
