#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR job: BGS S1 full-model joint fit — Step 3 (ALL params free), MCMC.
#
# ZM15 (13, Gaussian priors from the mass-bin posterior) + ESD point mass (1) +
# X-ray band relations (8) + Powell AGN (5) = 27 free params.  Unlike Step 2, the
# galaxy sector is recomputed every eval (ZM15 is free), so this is the heavier run
# — use a long walltime.  Resumable (emcee HDF backend), so besteffort is safe.
#
# Prereqs on dahu: bash oarsub/rsync_data_to_dahu.sh  +  git pull.
#
# Submit:  oarsub --project your-oar-project -S ./oarsub/fit_bgs_full_joint_allparams_mcmc.sh
# =============================================================================

#OAR --name bgs_full_joint_allparams
#OAR --project your-oar-project
#OAR -l /nodes=1/core=16,walltime=48:00:00
#OAR --stdout oarsub/logs/%jobid%.bgs_full_joint_allparams.out
#OAR --stderr oarsub/logs/%jobid%.bgs_full_joint_allparams.err
#OAR -t besteffort
#OAR -t idempotent

set -euo pipefail

REPO="${HOME}/software/hod_mod"
CONDA_ENV="hod_mod"

export HOD_MOD_DATA_DIR="${HOME}/data/hod_mod_data"
# defines VTAG + HOD_MOD_RESULTS; must precede OUT_DIR, which uses both
# Optional campaign tag as $1: OAR does NOT propagate the submitting shell's
# environment to the node, so `VTAG=v0.3 oarsub -S ./script.sh` would silently
# run as the default.  Pass it as an argument instead:
#   oarsub --project P -S "./oarsub/fit_bgs_full_joint_allparams_mcmc.sh v0.3"
VTAG="${1:-${VTAG:-v0.31}}"
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
OUT_DIR="${HOD_MOD_RESULTS}/bgs_full_joint_allparams_${VTAG}"

export MAMBA_EXE='/home/your-cluster-login/miniforge3/bin/mamba'
export MAMBA_ROOT_PREFIX='/home/your-cluster-login/miniforge3'
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__mamba_setup"; else alias mamba="$MAMBA_EXE"; fi
unset __mamba_setup
mamba activate "${CONDA_ENV}"

NCORES="${OAR_RES_NB_CORES:-8}"
export OMP_NUM_THREADS="${NCORES}" OPENBLAS_NUM_THREADS="${NCORES}" MKL_NUM_THREADS="${NCORES}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_PLATFORMS=cpu

cd "${REPO}"
mkdir -p oarsub/logs "${OUT_DIR}"
echo "host=$(hostname) job=${OAR_JOB_ID:-local} cores=${NCORES} start=$(date -Is)"

# Skip the MAP if it is already done: if map_result.json exists in OUT_DIR run
# MCMC only (resumes chain.h5); otherwise run the MAP first (--mode both) to seed
# the walkers.  So a first submit does MAP+MCMC and every idempotent/besteffort
# re-submit continues from the chain without ever redoing the MAP.
if [ -f "${OUT_DIR}/map_result.json" ]; then
    MODE=mcmc; echo "[job] map_result.json present -> MCMC only (resume)"
else
    MODE=both; echo "[job] no map_result.json -> MAP then MCMC"
fi
python -m hod_mod.scripts.fitting.fit_bgs_full_joint \
    --free-zm15 --mode "${MODE}" \
    --n-walkers 64 --n-burnin 1000 --n-steps 3000 \
    --out-dir "${OUT_DIR}"

echo "done=$(date -Is)"
