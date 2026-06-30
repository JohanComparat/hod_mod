#!/usr/bin/env bash
# Run Tests A1-A5 (wp-only) and B1-B4 (ESD-only) progressive MAP fits.
# Must be run from the repository root:
#   bash hod_mod/scripts/fitting/bgs_ls10/run_tests_ab.sh
set -e

PYTHON="/home/comparat/mamba/envs/hod_mod/bin/python3"
RUNNER="hod_mod/scripts/fitting/run_fit.py"
CONF="configs/fitting"

run_fit() {
    local cfg="$1"
    echo "========================================"
    echo "Running: $cfg"
    echo "========================================"
    t0=$SECONDS
    $PYTHON $RUNNER "$CONF/$cfg" --map-only
    echo "  --> done in $((SECONDS - t0)) s"
    echo ""
}

# ── Test A: wp-only, progressive complexity ──────────────────────────────────
run_fit TestA1_wp_rp500.yml           # A1: 5-param, rp > 0.5 Mpc/h
run_fit TestA2_wp_rp300.yml           # A2: 5-param, rp > 0.3 Mpc/h
run_fit TestA3_wp_rp300_inc.yml       # A3: 7-param (+inc), rp > 0.3 Mpc/h
run_fit TestA4_wp_rp100_offcen_inc.yml  # A4: 9-param (+offcen+inc), rp > 0.1 Mpc/h
run_fit TestA5_wp_rp050_offcen_inc.yml  # A5: 9-param (+offcen+inc), rp > 0.05 Mpc/h

# ── Test B: ESD-only, progressive complexity ──────────────────────────────────
run_fit TestB1_esd_rp1500.yml                  # B1: 5-param, R > 1.5 Mpc/h
run_fit TestB2_esd_rp1500_ia.yml               # B2: 6-param (+IA), R > 1.5 Mpc/h
run_fit TestB3_esd_rp500_ia_stellar.yml        # B3: 7-param (+IA+stellar), R > 0.5 Mpc/h
run_fit TestB4_esd_rp300_ia_stellar_offcen.yml # B4: 9-param (+IA+stellar+offcen), R > 0.3 Mpc/h

echo "========================================"
echo "All tests complete."
echo "========================================"
