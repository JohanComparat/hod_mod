# shellcheck shell=bash
# =============================================================================
# Shared campaign-version resolution.  Sourced by run_job.sh (array jobs) and by
# each standalone oarsub script (Family C, full_joint), so every submission path
# agrees on one definition instead of hard-coding the tag in 8 places.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
#
# Sets: VTAG, HOD_MOD_RESULTS (exported).
#
# Why a per-campaign results tree, and not just a per-run --out-dir suffix:
# only the 3 production lines carry a suffix.  The ~47 benchmark/forecast lines
# have NO --out-dir and write in place via hod_mod.paths.results_root(), so two
# campaigns sharing one root means the second silently overwrites the first's
# benchmarks — destroying the very comparison the campaign exists to make.
#
#   VTAG=v0.3   -> ~/data/hod_mod_results         Hankel fix, CAMB P(k)
#   VTAG=v0.31  -> ~/data/hod_mod_results_v0.31   Hankel fix + CosmoPower P(k)
#
# v0.3 ran before the cluster was updated, so it used CAMB.  The v0.3 -> v0.31
# difference therefore isolates the P(k) swap alone (~+2.5% in P(k) amplitude,
# ~+1.3% in sigma8), with the Hankel fix common to both.  Override per run:
#   VTAG=v0.31 oarsub ... -S ./oarsub/fit_comparat2025_gas_shape.sh
# =============================================================================

VTAG="${VTAG:-v0.31}"

# --- read-data root ----------------------------------------------------------
# Must be exported by EVERY submission path, not just run_job.sh: without it
# hod_mod.paths.data_root() silently falls back to the in-repo hod_mod/data/,
# which has no zenodo/ or xray_bands/.  That is precisely how the Family C
# presets failed a second time -- once past the ZM15 guard they died on
#   FileNotFoundError: Zenodo data not found:
#     <repo>/hod_mod/data/zenodo/LSDR10_GALxEVT/.../..._GALxEVT_wtheta.fits
# because the 5 standalone scripts never exported it (run_job.sh and the
# full_joint scripts did).  Same failure mode as the unexported HOD_MOD_RESULTS.
export HOD_MOD_DATA_DIR="${HOD_MOD_DATA_DIR:-${HOME}/data/hod_mod_data}"
export HOD_MOD_SUMSTAT="${HOD_MOD_SUMSTAT:-${HOME}/software/sum_stat/data}"

# Fail loudly rather than let data_root() fall back to the repo and surface as a
# confusing "file not found" deep inside a fit.
if [ ! -d "${HOD_MOD_DATA_DIR}" ]; then
    echo "FATAL: HOD_MOD_DATA_DIR does not exist: ${HOD_MOD_DATA_DIR}" >&2
    echo "       hod_mod.paths.data_root() would silently fall back to the" >&2
    echo "       in-repo hod_mod/data/, which lacks zenodo/ and xray_bands/." >&2
    exit 2
fi

# Pin the linear P(k) to the backend that campaign was RUN with, so a re-run
# lands on the same physics as its siblings.  The default moved CAMB ->
# CosmoPower emulator, which shifts P(k) ~+2.5% in amplitude (~+1.3% sigma8);
# v0.3 ran entirely on CAMB (the cluster was updated afterwards), so a v0.3
# re-run on today's code must be pinned back or it is not comparable to the
# v0.3 results it is meant to complete.
#
# This is derived from VTAG rather than left to the caller because OAR does not
# propagate the submitting shell's environment to the node — an
# `HOD_MOD_PK_BACKEND=camb oarsub ...` would be silently dropped.
if [ -z "${HOD_MOD_PK_BACKEND:-}" ]; then
    case "${VTAG}" in
        v0.3) export HOD_MOD_PK_BACKEND=camb ;;        # Hankel fix, CAMB P(k)
        *)    export HOD_MOD_PK_BACKEND=cosmopower ;;  # v0.31+: emulator
    esac
else
    export HOD_MOD_PK_BACKEND
fi

# The base (un-versioned) tree: holds fixed *reference inputs* produced by
# earlier work, as opposed to this campaign's outputs.
HOD_MOD_RESULTS_BASE="${HOD_MOD_RESULTS_BASE:-${HOME}/data/hod_mod_results}"
if [ "${VTAG}" = "v0.3" ]; then
    export HOD_MOD_RESULTS="${HOD_MOD_RESULTS:-${HOD_MOD_RESULTS_BASE}}"
else
    export HOD_MOD_RESULTS="${HOD_MOD_RESULTS:-${HOD_MOD_RESULTS_BASE}_${VTAG}}"
fi

# ZM15 MAP reference for `fit_comparat2025 --fix-zm15`.
#
# This is an INPUT, not a campaign output — a June 2026 ZM15 joint-fit MAP that
# every Family C preset holds fixed.  It must therefore come from the BASE tree,
# never from ${HOD_MOD_RESULTS}: a versioned campaign tree starts empty.
#
# Why this is passed explicitly.  fit_comparat2025 computes its default as
# results_root()/bgs_zm15_joint_wp_ngal/map_result.json *at import time*, so a
# bare `--fix-zm15` silently follows whatever results_root() resolves to.  In the
# v0.3 campaign HOD_MOD_RESULTS was never exported by the Family C scripts, so
# results_root() fell back to platformdirs (~/.local/share/hod_mod/results) and
# all 5 presets died with "--fix-zm15 json not found" after creating their empty
# out-dirs (jobs 289462-289466).  Passing the path explicitly makes the
# dependency visible and immune to both bugs.
export ZM15_JSON="${ZM15_JSON:-${HOD_MOD_RESULTS_BASE}/bgs_zm15_joint_wp_ngal/map_result.json}"
export VTAG HOD_MOD_RESULTS_BASE

# --- seed the versioned tree with fixed reference inputs ---------------------
# A versioned tree starts EMPTY, but several consumers resolve *reference*
# inputs through results_root() — which now points at that empty tree:
#
#   fitting/full_joint.py::load_zm15_median()
#       results_root()/bgs_zm15_joint_wp_ngal/posterior_summary.json
#   agn/duty_cycle.py::_ZM15_MAP_JSON            (module level, at import)
#       results_root()/bgs_zm15_joint_wp_ngal/map_result.json
#   scripts/fitting/fit_comparat2025.py::_DEFAULT_ZM15_JSON   (ditto)
#
# These are June-2026 products, not campaign outputs, so they must resolve to
# the base tree.  Symlinking them into the versioned tree fixes every consumer
# at once without patching results_root() call sites (the alternative — passing
# an explicit path everywhere — only works where a CLI flag exists; duty_cycle
# and full_joint have none).  Symlink, not copy: one 6.7 MB source of truth.
REFERENCE_INPUTS="${REFERENCE_INPUTS:-bgs_zm15_joint_wp_ngal}"
if [ "${HOD_MOD_RESULTS}" != "${HOD_MOD_RESULTS_BASE}" ]; then
    mkdir -p "${HOD_MOD_RESULTS}"
    for _ref in ${REFERENCE_INPUTS}; do
        if [ -e "${HOD_MOD_RESULTS_BASE}/${_ref}" ] && [ ! -e "${HOD_MOD_RESULTS}/${_ref}" ]; then
            ln -sfn "${HOD_MOD_RESULTS_BASE}/${_ref}" "${HOD_MOD_RESULTS}/${_ref}"
            echo "[campaign_env] seeded reference input: ${HOD_MOD_RESULTS}/${_ref} -> base"
        fi
    done
fi

# Fail before the job burns a slot: in v0.3 the missing reference surfaced only
# after the fit had started and created its out-dir, which is why 5 empty
# `fits/comparat2025_fixedZM15_*_v0.3/` dirs were left behind (an empty dir is
# worse than none — collect_and_plot.sh would read it and emit stale figures).
campaign_require_zm15_json() {
    if [ ! -f "${ZM15_JSON}" ]; then
        echo "FATAL: ZM15 reference not found: ${ZM15_JSON}" >&2
        echo "       It is an input from the June-2026 ZM15 joint fit, not a" >&2
        echo "       campaign product, so it lives in the BASE results tree:" >&2
        echo "         HOD_MOD_RESULTS_BASE=${HOD_MOD_RESULTS_BASE}" >&2
        echo "       Stage it on the cluster, or point ZM15_JSON at it." >&2
        exit 2
    fi
}
