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
# Submit:  oarsub --project pr-orphans -S ./oarsub/fit_bgs_full_joint_allparams_mcmc.sh
# =============================================================================

#OAR -n bgs_full_joint_allparams
#OAR --project pr-orphans
#OAR -l /nodes=1/core=16,walltime=48:00:00
#OAR --stdout oarsub/logs/%jobid%.bgs_full_joint_allparams.out
#OAR --stderr oarsub/logs/%jobid%.bgs_full_joint_allparams.err
#OAR -t besteffort
#OAR -t idempotent

set -euo pipefail

REPO="${HOME}/software/hod_mod"
CONDA_ENV="hod_mod"
OUT_DIR="${HOME}/data/hod_mod_results/bgs_full_joint_allparams"

export HOD_MOD_DATA_DIR="${HOME}/data/hod_mod_data"
export HOD_MOD_RESULTS="${HOME}/data/hod_mod_results"

export MAMBA_EXE='/home/comparaj/miniforge3/bin/mamba'
export MAMBA_ROOT_PREFIX='/home/comparaj/miniforge3'
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

# MCMC only — resumes from ${OUT_DIR}/chain.h5 (emcee HDF backend) and seeds a
# fresh chain from ${OUT_DIR}/map_result.json if present.  No MAP re-run, so an
# idempotent/besteffort re-submit continues exactly where it left off.  (For a
# cold start, run the MAP once locally with --mode map and rsync map_result.json.)
python -m hod_mod.scripts.fitting.fit_bgs_full_joint \
    --free-zm15 --mode mcmc \
    --n-walkers 64 --n-burnin 1000 --n-steps 3000 \
    --out-dir "${OUT_DIR}"

echo "done=$(date -Is)"
