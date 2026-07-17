#!/usr/bin/env bash
# =============================================================================
# Pull campaign results back from dahu, one VTAG at a time.
#
# Each campaign lives in its OWN results tree (see oarsub/_campaign_env.sh), so
# a pull is per-version.  The root convention is mirrored from that file:
#
#   v0.3   -> <base>            Hankel fix, CAMB P(k)
#   v0.31  -> <base>_v0.31      Hankel fix + CosmoPower P(k)
#
# By default BOTH are pulled, in order, so the v0.3 -> v0.31 comparison (which
# isolates the P(k) swap) can be made locally.
#
# Safe by default: `rsync --dry-run` prints the file list.  Add --go to transfer.
# Config is env-overridable — set DAHU_HOST / BASTION to match ~/.ssh/config.
#
#   ./oarsub/pull_results.sh                 # dry-run, both versions
#   ./oarsub/pull_results.sh --go            # transfer, both versions
#   ./oarsub/pull_results.sh --go v0.31      # transfer, only v0.31
#   VTAGS="v0.3 v0.31" ./oarsub/pull_results.sh --go
# =============================================================================
set -uo pipefail

# --- config (override via env) ----------------------------------------------
DAHU_USER="${DAHU_USER:-your-cluster-login}"                     # login on dahu
DAHU_HOST="${DAHU_HOST:-dahu.ciment}"                         # ssh alias or FQDN
# BASE roots (un-versioned); the per-VTAG roots are derived from these exactly
# as _campaign_env.sh derives them on the cluster.
DAHU_RESULTS_BASE="${DAHU_RESULTS_BASE:-/home/${DAHU_USER}/data/hod_mod_results}"
LOCAL_RESULTS_BASE="${LOCAL_RESULTS_BASE:-/home/comparat/data/hod_mod_results}"
# Extra ProxyJump. EMPTY BY DEFAULT — ~/.ssh/config already routes the cluster:
#
#   Host *.ciment
#     ProxyCommand ssh -q your-cluster-login@access-gricad... "nc -w 60 `basename %h .ciment` %p"
#
# That ProxyCommand strips the `.ciment` suffix, so the bastion is asked for the
# name it knows (`dahu`).  Adding `-J <bastion>` here *overrides* it and hands the
# bastion the raw `dahu.ciment` instead, which it cannot resolve:
#   channel 0: open failed: connect failed: Name or service not known
# Only set BASTION on a host whose ssh config does NOT already route to dahu.
BASTION="${BASTION:-}"

# --- args --------------------------------------------------------------------
DRY="--dry-run --stats"
ARGS=()
for a in "$@"; do
    case "$a" in
        --go) DRY="" ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) ARGS+=("$a") ;;
    esac
done
# positional VTAGs win over $VTAGS, which wins over the both-versions default.
if [ "${#ARGS[@]}" -gt 0 ]; then VTAGS=("${ARGS[@]}"); else read -r -a VTAGS <<<"${VTAGS:-v0.3 v0.31}"; fi

# Mirror of _campaign_env.sh: v0.3 keeps the un-versioned root, everything else
# gets its own `<base>_<vtag>` tree.
root_for() {  # $1=vtag  $2=base
    if [ "$1" = "v0.3" ]; then printf '%s' "$2"; else printf '%s_%s' "$2" "$1"; fi
}

# Campaign outputs under a results root.  %V% is replaced by the VTAG: the
# version-suffixed dirs are the 3 production lines + the 2 full_joint runs; the
# rest (benchmarks/, tier*/, fits/) are written in place by jobs that take no
# --out-dir, which is exactly why each campaign needs its own tree.
SUBDIR_TMPL=(
  benchmarks                          # Family A (run_benchmark --plot)
  fits                                # Family C comparat2025_*_%V% + Family D fits/benchmark
  bgs_zm15_joint_wp_ngal_%V%          # Family B
  bgs_zm15_thresh_joint_%V%           # Family B
  bgs_comparat2025_%V%                # Family B
  bgs_full_joint_fixedzm15_%V%        # Family B full-joint
  bgs_full_joint_allparams_%V%        # Family B full-joint
  tier2_forecast tier3_forecast tier4_forecast stage4_forecast sensitivity_fisher   # Family D
)

SSH_E=(); [[ -n "$BASTION" ]] && SSH_E=(-e "ssh -J ${BASTION}")
[[ -n "$DRY" ]] && echo "mode: DRY-RUN (pass --go to transfer)" || echo "mode: TRANSFER"
echo "versions: ${VTAGS[*]}"
echo

rc_all=0
for vtag in "${VTAGS[@]}"; do
    remote_root="$(root_for "$vtag" "$DAHU_RESULTS_BASE")"
    local_root="$(root_for "$vtag" "$LOCAL_RESULTS_BASE")"

    # One explicit source per subdir.  NOT a `host:{a,b,c}` brace list: bash does
    # brace expansion *before* variable expansion, so a list built into a variable
    # is never expanded — locally or remotely — and rsync receives the literal
    # string `{a,b,c}` as a single filename ("link_stat ... failed").  rsync
    # accepts multiple sources sharing one remote host over a single connection.
    srcs=()
    for s in "${SUBDIR_TMPL[@]}"; do
        srcs+=("${DAHU_USER}@${DAHU_HOST}:${remote_root}/${s//%V%/$vtag}")
    done

    echo "=============================================================="
    echo "VTAG $vtag"
    echo "  src : ${DAHU_USER}@${DAHU_HOST}:${remote_root}/"
    echo "        {$(printf '%s ' "${SUBDIR_TMPL[@]//%V%/$vtag}")}"
    echo "  dst : ${local_root}/"
    mkdir -p "$local_root"

    # Do NOT abort the loop when a tree is incomplete: a campaign still running
    # (or one whose family failed) simply lacks some of these dirs, and rsync
    # exits 23 ("partial transfer due to error") for each missing source.  That
    # must not stop the *other* version from being pulled.
    # shellcheck disable=SC2086
    rsync -avhz --progress $DRY "${SSH_E[@]}" "${srcs[@]}" "${local_root}/"
    rc=$?
    case "$rc" in
        0)  echo "  -> OK" ;;
        23) echo "  -> partial (rc=23): some dirs absent on dahu — expected while the campaign is still running or a family failed" ;;
        24) echo "  -> partial (rc=24): some files vanished mid-transfer (jobs still writing)" ;;
        *)  echo "  -> FAILED (rc=$rc)"; rc_all=$rc ;;
    esac
    echo
done

if [[ -n "$DRY" ]]; then
    echo "Preview only. Re-run with --go to transfer."
else
    echo "Done. Next, per version:"
    for vtag in "${VTAGS[@]}"; do
        lr="$(root_for "$vtag" "$LOCAL_RESULTS_BASE")"
        echo "  VTAG=$vtag HOD_MOD_RESULTS=$lr ./oarsub/campaign_status.sh"
        echo "  VTAG=$vtag HOD_MOD_RESULTS=$lr PY=/home/comparat/mamba/envs/hod_mod/bin/python ./oarsub/collect_and_plot.sh"
    done
fi
exit "$rc_all"
