#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR generic array-job wrapper for the v0.3 re-run campaign.
#
# One wrapper for the whole "re-run every w_p / ΔΣ / Σ_y / w_θ calculation after
# the v0.3 Hankel fix" campaign.  Submit as an OAR *array* job with a param file
# whose every line is one full `python -m ...` command:
#
#   oarsub --project <PROJECT> \
#          -l /nodes=1/core=8,walltime=02:00:00 \
#          --array-param-file oarsub/params/benchmarks_map.txt \
#          -S ./oarsub/run_job.sh
#
# OAR passes the selected param line as the script's arguments ($@).  This
# wrapper sets up the environment, cd's into the repo, and `eval`s the line so
# that the shell variables defined below (DATA_BGS, RESULTS, ...) expand — that
# keeps the param files portable between the workstation (/home/comparat) and
# dahu (/home/your-cluster-login).  Resources (cores / walltime) come from the `-l` flag
# on the command line so one wrapper serves every family; see
# oarsub/submit_campaign.sh for the per-family sizing.
#
# Smoke-test any single line on the dev partition first, e.g.:
#   sed -n '1p' oarsub/params/benchmarks_map.txt        # inspect the line
#   oarsub -t devel -l /nodes=1/core=8,walltime=00:30:00 \
#          -p "network_address='<a devel node>'" -S ...   # or run the line locally
# =============================================================================

#OAR -n hodmod_v03_rerun
#OAR --project PROJECTNAME
#OAR -l /nodes=1/core=8,walltime=02:00:00
#OAR --stdout oarsub/logs/%jobid%.v03_rerun.out
#OAR --stderr oarsub/logs/%jobid%.v03_rerun.err
# --- resumable MCMC jobs (production_mcmc.txt) can be auto-restarted: the emcee
#     HDF backend is idempotent, so uncomment for CiGri / besteffort submission.
# #OAR -t besteffort
# #OAR -t idempotent

set -euo pipefail

# --- user configuration (edit for your GRICAD account) ----------------------
REPO="${HOME}/software/hod_mod"                        # repo location on the cluster
CONDA_ENV="hod_mod"                                    # conda/mamba env name

# --- campaign version (VTAG + HOD_MOD_RESULTS) ------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
export HOD_MOD_DATA_DIR="${HOD_MOD_DATA_DIR:-${HOME}/data/hod_mod_data}"
export HOD_MOD_SUMSTAT="${HOD_MOD_SUMSTAT:-${HOME}/software/sum_stat/data}"
# convenience shortcuts referenced by the param files ------------------------
DATA_BGS="${HOD_MOD_SUMSTAT}/BGS_Mstar10_massbins"     # ZM15 M*>10 mass-bin joint HDF5
RESULTS="${HOD_MOD_RESULTS}"                            # out-dir root (v0.3 suffix per line)

# --- environment ------------------------------------------------------------
export MAMBA_EXE="${MAMBA_EXE:-/home/your-cluster-login/miniforge3/bin/mamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/your-cluster-login/miniforge3}"
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="$MAMBA_EXE"
fi
unset __mamba_setup
mamba activate "${CONDA_ENV}"

# emcee / scipy MAP are serial; let JAX-XLA + BLAS use the allocated cores for
# the per-step halo-model linear algebra.  x64 is set globally so the forecast
# multiprocessing workers (--jobs N, spawned) inherit it — the known spawn-x64
# gotcha — and so the differentiable fits get reliable gradients.
NCORES="${OAR_RES_NB_CORES:-8}"
export OMP_NUM_THREADS="${NCORES}"
export OPENBLAS_NUM_THREADS="${NCORES}"
export MKL_NUM_THREADS="${NCORES}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

cd "${REPO}"
mkdir -p oarsub/logs "${HOD_MOD_RESULTS}"

CMD="$*"
echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  array=${OAR_ARRAY_INDEX:-0}  cores=${NCORES}  start=$(date -Is)"
echo "vtag=${VTAG}  results=${HOD_MOD_RESULTS}"
# Record which linear P(k) actually produced these numbers: the default moved
# from CAMB to the CosmoPower-JAX emulator, which shifts P(k) by ~+2.5% in
# amplitude (~+1.3% in sigma8), so "which backend ran" is provenance, not
# trivia.  Under `set -e` this also fails the job immediately (rather than deep
# inside the fit) if cosmopower-jax is missing from the cluster env.
echo "pk_backend=$(python -c 'from hod_mod.core.power_spectrum import default_pk_linear; print(type(default_pk_linear()).__name__)')"
echo "cmd> ${CMD}"

# The param line references $DATA_BGS / $RESULTS (defined above), so eval it.
eval "${CMD}"

echo "done=$(date -Is)  cmd> ${CMD}"
