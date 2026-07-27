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
PY="${PY:-python}"      # only used to read the emcee HDF backend's step counter

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

# MCMC progress out of an emcee HDF backend: "<steps done>/<steps allocated>".
# emcee grows the `chain` dataset to the full requested length up front, so
# shape[0] is the target and `iteration` is where the sampler actually got to.
_chain_progress() {
    [ -f "$1" ] || return 0
    "$PY" - "$1" 2>/dev/null <<'PYEOF'
import sys
try:
    import h5py
    with h5py.File(sys.argv[1], "r") as f:
        g = f[list(f.keys())[0]]
        print("%d/%d steps" % (int(g.attrs["iteration"]), g["chain"].shape[0]))
except Exception:
    pass
PYEOF
}

# check_fit <dir> <required artifact> <what>
#
# A non-empty directory is NOT the question this script is being asked.  Each
# production fit writes its MAP figures and a per-step chain.h5 as it goes, but
# the artifact the plotters actually read (flatchain.npz) is written once, only
# after the sampler reaches its full step budget.  A walltime-killed MCMC
# therefore leaves a directory full of files and no flatchain.npz — which
# check_dir called OK, and collect_and_plot.sh then died on:
#   FileNotFoundError: .../bgs_zm15_joint_wp_ngal_v0.3/flatchain.npz
# Check the artifact the consumer opens, and report how far the chain got.
check_fit() {
    local d="$1" art="$2" what="$3" p
    if [ ! -d "$RES/$d" ]; then bad "$d" "absent — $what"; return; fi
    if [ -f "$RES/$d/$art" ]; then
        ok "$d" "$art, $(find "$RES/$d/$art" -printf '%TY-%Tm-%Td %TH:%TM\n' 2>/dev/null)"
    else
        # chain.h5 for the ZM15/full-joint fits; <sample>_backend.h5 for
        # fit_joint_lsdr10, which names its checkpoint after the sample.
        local ck="$RES/$d/chain.h5"
        [ -f "$ck" ] || ck="$(find "$RES/$d" -maxdepth 1 -name '*_backend.h5' -print -quit 2>/dev/null)"
        p="$(_chain_progress "$ck")"
        if [ -n "$p" ]; then
            bad "$d" "no $art — MCMC unfinished at $p (resubmit to resume)"
        else
            bad "$d" "no $art — $what"
        fi
    fi
}

echo "== campaign status  VTAG=$VTAG  RESULTS=$RES"
echo
echo "-- production galaxy joint fits (param file: production_mcmc.txt) --"
check_fit "bgs_zm15_joint_wp_ngal_${VTAG}"   flatchain.npz "production_mcmc family not finished/synced"
check_fit "bgs_full_joint_fixedzm15_${VTAG}" flatchain.npz "full_joint family not finished/synced"
check_fit "bgs_full_joint_allparams_${VTAG}" flatchain.npz "full_joint family not finished/synced"
check_fit "bgs_zm15_thresh_joint_${VTAG}"    flatchain.npz "production_mcmc family not finished/synced"
# fit_joint_lsdr10 names its products after the sample, and its MCMC is the one
# that used to keep the whole chain in memory until the end (see run_mcmc).
check_fit "bgs_comparat2025_${VTAG}"         S1_chain.h5   "production_mcmc family not finished/synced"

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
    # Those two totals say nothing about coverage: for VTAG=v0.3 the campaign
    # tree IS the base tree, so it also holds June figures and legacy version0/
    # dirs, and a family that never ran still counts as "241 files".  Resolve
    # every requested --model through the run_benchmark registry to its config's
    # output.dir and check the artifacts there instead.  SINCE=YYYY-MM-DD also
    # flags results that predate the campaign (i.e. were never re-run).
    bench_out="$("$PY" - "$PWD" "$RES" "${SINCE:-}" 2>/dev/null <<'PYEOF'
import os, re, sys
REPO, RES, SINCE = sys.argv[1], sys.argv[2], sys.argv[3]
import yaml
sys.path.insert(0, REPO)
from hod_mod.scripts.benchmarks.run_benchmark import BENCHMARK_REGISTRY

cut = 0.0
if SINCE:
    import datetime
    cut = datetime.datetime.fromisoformat(SINCE).timestamp()

models = []
for pf in ("oarsub/params/benchmarks_map.txt", "oarsub/params/benchmarks_mcmc.txt"):
    p = os.path.join(REPO, pf)
    if not os.path.exists(p):
        continue
    for line in open(p):
        s = line.strip()
        if s and not s.startswith("#"):
            m = re.search(r"--model\s+(\S+)", s)
            if m:
                models.append((m.group(1), "mcmc" if "--mcmc" in s else "map"))

bad = 0
for name, kind in models:
    entry = BENCHMARK_REGISTRY.get(name)
    if entry is None:
        print(f"       ?? {name:40s} not in the run_benchmark registry"); bad += 1; continue
    cfg = yaml.safe_load(open(os.path.join(REPO, entry["config"])))
    rel = re.sub(r"^results/", "", cfg.get("output", {}).get("dir", "")).rstrip("/")
    d = os.path.join(RES, rel)
    res = os.path.join(d, "benchmark_result.json")
    pngs = len([f for f in os.listdir(d) if f.endswith(".png")]) if os.path.isdir(d) else 0
    if not os.path.exists(res):
        print(f"       MISS {name:40s} [{kind}] no benchmark_result.json ({pngs} PNGs)"); bad += 1
    elif pngs == 0:
        print(f"       MISS {name:40s} [{kind}] result but no figures"); bad += 1
    elif cut and os.path.getmtime(res) < cut:
        import datetime as _dt
        when = _dt.datetime.fromtimestamp(os.path.getmtime(res)).strftime("%Y-%m-%d")
        print(f"       OLD  {name:40s} [{kind}] last run {when} (predates {SINCE})"); bad += 1
print(f"       {len(models) - bad}/{len(models)} requested models complete")
sys.exit(1 if bad else 0)
PYEOF
    )"; bench_rc=$?
    [ -n "$bench_out" ] && printf '%s\n' "$bench_out"
    if [ "$bench_rc" -ne 0 ] && [ -n "$bench_out" ]; then
        miss=$((miss+1))
    elif [ -z "$bench_out" ]; then
        printf "       (per-model check unavailable: %s could not import the registry)\n" "$PY"
    fi
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
    echo "== $miss required output(s) MISSING — collect_and_plot.sh would skip those sections."
    echo "   Finish/resume the jobs, then re-sync:  ./oarsub/pull_results.sh --go $VTAG"
    echo "   Or refresh only what IS complete:"
    echo "     VTAG=$VTAG HOD_MOD_RESULTS=$RES ALLOW_PARTIAL=1 ./oarsub/collect_and_plot.sh"
    exit 1
fi
