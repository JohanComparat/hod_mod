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
# Family D (forecasts) is sourced separately.  oarsub_campaign.rst records that
# the HOD_MOD_PK_BACKEND pin reaches Families A-C only: the forecast stack builds
# ForwardModel, which picks its P(k) correction from a driver flag and defaults
# to the emulator because real CAMB is not JAX-traceable.  So a v0.3 and a v0.31
# forecast are the same physics (their tier sigmas agree to 3-4%), and the only
# question is which campaign actually finished one -- for tier2/3/4 that is
# v0.31, while v0.3 still holds the pre-campaign 2026-07-02..04 leftovers.
# Reading Family D from where it ran is therefore not mixing campaigns; pinning
# it to $RES for symmetry's sake would just keep publishing stale numbers.
RES_FAMILY_D="${RES_FAMILY_D:-${RES}}"
IMG="${IMG:-docs/_images}"   # overridable so a collection can be rehearsed off-repo

echo "== collecting VTAG=$VTAG from $RES into $IMG"

# docs/_images holds ONE campaign's figures.  The two campaigns do not produce
# byte-identical file *sets* (v0.31 is behind v0.3 on several benchmarks), so
# collecting one on top of the other leaves a silent mix: the overlapping names
# are overwritten and the rest keep the previous campaign's numbers, under doc
# prose that claims a single P(k) backend.  Stamp what is in there and refuse
# to switch without saying so out loud.
STAMP="$IMG/.campaign_vtag"
prev_vtag="$(cat "$STAMP" 2>/dev/null || true)"
if [ -n "$prev_vtag" ] && [ "$prev_vtag" != "$VTAG" ] && [ "${FORCE_VTAG_SWITCH:-0}" != "1" ]; then
    echo "!! $IMG currently holds VTAG=$prev_vtag figures; you asked for $VTAG."
    echo "   Collecting on top would mix the two campaigns.  To switch deliberately:"
    echo "     FORCE_VTAG_SWITCH=1 VTAG=$VTAG HOD_MOD_RESULTS=$RES $0"
    echo "   (and check 'git status $IMG' afterwards — figures the new campaign"
    echo "    has not produced yet stay at their $prev_vtag values)."
    exit 1
fi

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
# Guarded: with ALLOW_PARTIAL=1 the tree may have no benchmarks/ at all, and a
# failing `find` on the left of a pipe kills the whole script under pipefail —
# which would contradict the "missing sections are skipped" promise above.
if [ ! -d "$RES/benchmarks" ]; then
    echo "   .. skip (no $RES/benchmarks)"
else
find "$RES/benchmarks" -type f -name '*.png' -print0 | while IFS= read -r -d '' f; do
    rel="${f#"$RES"/}"              # e.g. benchmarks/zumandelbaum2015_sdss/benchmark_..._wp.png
    dest="$IMG/${rel//\//__}"       # -> benchmarks__zumandelbaum2015_sdss__benchmark_..._wp.png
    cp -f "$f" "$dest"
done
fi

echo "== B. production galaxy joint fits (point plotters at the _${VTAG} out-dirs)"
# has_chain <dir> — every posterior plotter below opens exactly one file,
# flatchain.npz, and the fits write it only once the sampler has reached its
# full step budget.  A walltime-killed MCMC leaves chain.h5 and the MAP figures
# behind, so "the directory has files in it" was true for four fits that had no
# flatchain.npz, and the first plotter died with FileNotFoundError.  Test the
# file the plotter opens, and say which step the chain stopped at.
has_chain() {
    [ -f "$1/flatchain.npz" ] && return 0
    if [ -f "$1/chain.h5" ]; then
        echo "   .. skip $(basename "$1"): chain.h5 present but no flatchain.npz"
        echo "      (MCMC unfinished — resubmit the job to resume; see campaign_status.sh)"
    else
        echo "   .. skip $(basename "$1") (no output)"
    fi
    return 1
}

if has_chain "$RES/bgs_zm15_joint_wp_ngal_${VTAG}"; then
    $PY -m hod_mod.scripts.fitting.bgs_ls10.plot_bgs_zm15_joint_posterior \
        --out-dir "$RES/bgs_zm15_joint_wp_ngal_${VTAG}"
fi

if has_chain "$RES/bgs_full_joint_fixedzm15_${VTAG}"; then
    $PY -m hod_mod.scripts.fitting.plot_bgs_full_joint \
        --out-dir "$RES/bgs_full_joint_fixedzm15_${VTAG}" --docs
fi

# --free-zm15 switches the plotter to the bgs_full_joint_allparams__* figure
# prefix, so this cannot clobber the fixedzm15 figures above.
if has_chain "$RES/bgs_full_joint_allparams_${VTAG}"; then
    $PY -m hod_mod.scripts.fitting.plot_bgs_full_joint --free-zm15 \
        --out-dir "$RES/bgs_full_joint_allparams_${VTAG}" --docs
fi

#   bgs_comparat2025 (fit_joint_lsdr10) figures are written by the fit itself
#   (--no-plot was NOT passed); copy them if the fit put them in the run dir.
#   Note it saves S1_bestfit.pdf / S1_corner.pdf, so the PNG glob matches
#   nothing today — the docs page's bgs_comparat2025__* figures come from the
#   older per-config runs in the base tree, not from this campaign line.
find "$RES/bgs_comparat2025_${VTAG}" -maxdepth 1 -name '*.png' -exec cp -f {} "$IMG/" \; 2>/dev/null || true

echo "== C. Comparat+2025 X-ray w_θ presets — figures written by each fit into its _${VTAG} run dir"
for p in gas-shape gas-temp gas-full agn-occ agn-lum; do
    find "$RES/fits/comparat2025_fixedZM15_${p}_${VTAG}" -name '*.png' -exec cp -f {} "$IMG/" \; 2>/dev/null || true
done

echo "== D. forecasts + benchmark MAP+MCMC"
# NB: the writers in this section resolve docs/_images THEMSELVES (their own
# _docs_images() / _save()), so unlike sections A-C they ignore $IMG.  An
# IMG=/tmp/... rehearsal still overwrites the real figures.
#
# make_sensitivity_figures takes no arguments — it resolves
# results_root()/sensitivity_fisher/fisher_study.npz itself.  It is slow (jax +
# ForwardModel import alone is minutes) and it silently RECOMPUTES the Jacobian
# whenever the npz's param_names no longer match forward_jax.PARAM_NAMES, which
# is the case today (the pressure sector was re-parametrised after the run).
# Recomputing would produce HEAD-derived figures under a v0.3 label, so it stays
# off until run_sensitivity_study is re-run; see docs/sensitivity_fisher.rst.
if [ "${REFRESH_SENSITIVITY:-0}" = "1" ]; then
    JAX_ENABLE_X64=1 $PY -m hod_mod.scripts.forecasts.make_sensitivity_figures || true
else
    echo "   .. skip sensitivity figures (REFRESH_SENSITIVITY=1 to force — reads a"
    echo "      fisher_study.npz whose param_names no longer match the code)"
fi

# make_tier2_figures takes a POSITIONAL npz (ap.add_argument("npz")).  Called
# bare — as this line used to be — it exits 2 on argparse and `|| true` swallowed
# it, which is why docs/_images/tier{2,3,4}_forecast__*.png had never once been
# refreshed by this script.  One script serves all three tiers: it switches the
# figure prefix off the npz basename.  Guarded on SINCE so a run that only
# refreshed its cache/ cannot restamp 2026-07 science with today's date.
for t in 2 3 4; do
    npz="$RES_FAMILY_D/tier${t}_forecast/tier${t}_forecast_nb6.npz"
    if [ ! -f "$npz" ]; then
        echo "   .. skip tier${t} figures (no $(basename "$npz") — run never finished)"; continue
    fi
    if [ -n "${SINCE:-}" ] && [ "$(date -r "$npz" +%s)" -lt "$(date -d "$SINCE" +%s)" ]; then
        echo "   .. skip tier${t} figures ($(basename "$npz") predates $SINCE)"; continue
    fi
    extra=""
    [ "$t" = 2 ] && [ -f "$RES_FAMILY_D/tier2_forecast/tier2_forecast_nb1.npz" ] \
        && extra="--compare $RES_FAMILY_D/tier2_forecast/tier2_forecast_nb1.npz"
    JAX_ENABLE_X64=1 $PY -m hod_mod.scripts.forecasts.make_tier2_figures "$npz" $extra || true
done

# Same freshness gate: fits/benchmark/ last produced output on 2026-07-12, four
# days before the campaign opened, so regenerating its figures now would stamp a
# pre-campaign run with today's date on a page that claims to be v0.3.
_bm="$RES/fits/benchmark/benchmark_map_mcmc_mcmc_summary.json"
if [ -n "${SINCE:-}" ] && [ -f "$_bm" ] \
   && [ "$(date -r "$_bm" +%s)" -lt "$(date -d "$SINCE" +%s)" ]; then
    echo "   .. skip benchmark_map_mcmc figures (run predates $SINCE)"
else
    JAX_ENABLE_X64=1 $PY -m hod_mod.scripts.fitting.plot_benchmark_fit --label benchmark_map_mcmc || true
fi
#   run_stage4_forecast copies stage4_forecast.png into docs/_images — but it ran
#   on dahu, so that copy landed in dahu's checkout.  Copy the pulled one here.
cp -f "$RES_FAMILY_D/stage4_forecast/stage4_forecast.png" "$IMG/" 2>/dev/null || true

{ printf '%s\n' "$VTAG"
  [ "$RES_FAMILY_D" != "$RES" ] && printf '# family-D from: %s\n' "$RES_FAMILY_D"
} > "$STAMP"
echo "== done.  Review with:  git status $IMG   &&   git diff --stat docs/"
echo "   Then update the χ²/dof, best-fit params and the AUM agreement line in the .rst pages."
