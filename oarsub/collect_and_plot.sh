#!/usr/bin/env bash
# =============================================================================
# Phase 4 — after the dahu campaign lands, regenerate every affected figure into
# docs/_images/ from the fresh v0.3 results.  Run on the machine that has the
# repo + $HOD_MOD_RESULTS (workstation, after syncing the dahu out-dirs back).
#
#   HOD_MOD_RESULTS=/home/comparat/data/hod_mod_results ./oarsub/collect_and_plot.sh
#
# Idempotent: re-runs safely.  Numbers in the .rst prose (χ²/dof, best-fit
# params, the AUM agreement line) are updated by hand afterwards — this script
# only refreshes the PNGs.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python}"
# VTAG selects which campaign's out-dirs to collect (v0.3 = Hankel/CAMB,
# v0.31 = + CosmoPower P(k)).  HOD_MOD_RESULTS, if already set, still wins.
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
RES="${HOD_MOD_RESULTS:?set HOD_MOD_RESULTS}"
IMG="docs/_images"

echo "== collecting VTAG=$VTAG from $RES into $IMG"
# Preflight: this script runs under `set -e` and the plotters below take an
# explicit --out-dir, so one missing dir aborts the run half-way and leaves
# docs/_images/ partially refreshed (some figures new, some stale — the worst
# outcome, because git diff then looks like a legitimate partial update).
if ! VTAG="$VTAG" HOD_MOD_RESULTS="$RES" ./oarsub/campaign_status.sh >/dev/null 2>&1; then
    echo "!! campaign incomplete — refusing to half-refresh docs/_images/."
    echo "   Details:  VTAG=$VTAG HOD_MOD_RESULTS=$RES ./oarsub/campaign_status.sh"
    echo "   Override (collect what exists):  ALLOW_PARTIAL=1 $0"
    [ "${ALLOW_PARTIAL:-0}" = "1" ] || exit 1
    echo "   ALLOW_PARTIAL=1 set — continuing; missing sections are skipped."
fi

echo "== A. literature benchmarks: flatten results/benchmarks/<dir>/*.png -> $IMG/benchmarks__<dir>__*"
# run_benchmark --plot already wrote per-model PNGs under $RES/benchmarks/<subdir>/.
# The doc naming is exactly that path with '/'->'__'.
find "$RES/benchmarks" -type f -name '*.png' -print0 | while IFS= read -r -d '' f; do
    rel="${f#"$RES"/}"              # e.g. benchmarks/zumandelbaum2015_sdss/benchmark_..._wp.png
    dest="$IMG/${rel//\//__}"       # -> benchmarks__zumandelbaum2015_sdss__benchmark_..._wp.png
    cp -f "$f" "$dest"
done

echo "== B. production galaxy joint fits (point plotters at the _${VTAG} out-dirs)"
# has_files <dir> — a plotter pointed at an empty/absent dir either crashes or
# silently emits an empty figure; skip instead.
has_files() { [ -d "$1" ] && [ -n "$(find "$1" -type f -print -quit 2>/dev/null)" ]; }

if has_files "$RES/bgs_zm15_joint_wp_ngal_${VTAG}"; then
    $PY -m hod_mod.scripts.fitting.bgs_ls10.plot_bgs_zm15_joint_posterior \
        --out-dir "$RES/bgs_zm15_joint_wp_ngal_${VTAG}"
else echo "   .. skip bgs_zm15_joint_wp_ngal_${VTAG} (no output)"; fi

if has_files "$RES/bgs_full_joint_fixedzm15_${VTAG}"; then
    $PY -m hod_mod.scripts.fitting.plot_bgs_full_joint \
        --out-dir "$RES/bgs_full_joint_fixedzm15_${VTAG}" --docs
else echo "   .. skip bgs_full_joint_fixedzm15_${VTAG} (no output)"; fi

#   bgs_comparat2025 (fit_joint_lsdr10) figures are written by the fit itself
#   (--no-plot was NOT passed); copy them if the fit put them in the run dir:
find "$RES/bgs_comparat2025_${VTAG}" -maxdepth 1 -name '*.png' -exec cp -f {} "$IMG/" \; 2>/dev/null || true

echo "== C. Comparat+2025 X-ray w_θ presets — figures written by each fit into its _${VTAG} run dir"
for p in gas-shape gas-temp gas-full agn-occ agn-lum; do
    find "$RES/fits/comparat2025_fixedZM15_${p}_${VTAG}" -name '*.png' -exec cp -f {} "$IMG/" \; 2>/dev/null || true
done

echo "== D. forecasts + benchmark MAP+MCMC"
JAX_ENABLE_X64=1 $PY -m hod_mod.scripts.forecasts.make_sensitivity_figures || true
JAX_ENABLE_X64=1 $PY -m hod_mod.scripts.forecasts.make_tier2_figures || true
JAX_ENABLE_X64=1 $PY -m hod_mod.scripts.fitting.plot_benchmark_fit --label benchmark_map_mcmc || true
#   run_stage4_forecast already copies stage4_forecast.png into docs/_images.

echo "== done.  Review with:  git status $IMG   &&   git diff --stat docs/"
echo "   Then update the χ²/dof, best-fit params and the AUM agreement line in the .rst pages."
