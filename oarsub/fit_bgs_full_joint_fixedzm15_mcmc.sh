#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR job: BGS S1 full-model joint fit — Step 2 (FIXED ZM15), MCMC.
#
# Galaxies (ZM15 held at the mass-bin posterior median) + hot gas (transfer-grid
# band model) + AGN (Powell XLF/bias).  Free params: X-ray band relations (8) +
# Powell AGN (5) + ESD point mass (1).  The galaxy sector is precomputed once, so
# each likelihood eval is ~0.01 s (fast MCMC).
#
# Resumable: emcee HDF backend (chain.h5) flushed every step; re-submit to continue.
#
# Prereqs on dahu (run once): bash oarsub/rsync_data_to_dahu.sh   (stages the data +
# the X-ray transfer cache + the ZM15 posterior), and `git pull` the repo.
#
# Submit:  oarsub --project your-oar-project -S ./oarsub/fit_bgs_full_joint_fixedzm15_mcmc.sh
# Devel:   oarsub -t devel -S ./oarsub/fit_bgs_full_joint_fixedzm15_mcmc.sh
# =============================================================================

#OAR -n bgs_full_joint_fixedzm15
#OAR --project your-oar-project
#OAR -l /nodes=1/core=16,walltime=06:00:00
#OAR --stdout oarsub/logs/%jobid%.bgs_full_joint_fixedzm15.out
#OAR --stderr oarsub/logs/%jobid%.bgs_full_joint_fixedzm15.err
#OAR -t besteffort
#OAR -t idempotent

set -euo pipefail

REPO="${HOME}/software/hod_mod"
CONDA_ENV="hod_mod"
OUT_DIR="${HOME}/data/hod_mod_results/bgs_full_joint_fixedzm15"

# --- data / results roots (the COMPLETE data root includes xray_bands/) ------
export HOD_MOD_DATA_DIR="${HOME}/data/hod_mod_data"
export HOD_MOD_RESULTS="${HOME}/data/hod_mod_results"

# --- environment ------------------------------------------------------------
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

# MCMC only — resumes from ${OUT_DIR}/chain.h5 (emcee HDF backend) and seeds a
# fresh chain from ${OUT_DIR}/map_result.json if present.  No MAP re-run, so an
# idempotent/besteffort re-submit continues exactly where it left off.
python -m hod_mod.scripts.fitting.fit_bgs_full_joint \
    --mode mcmc \
    --n-walkers 48 --n-burnin 500 --n-steps 2000 \
    --out-dir "${OUT_DIR}"

echo "done=$(date -Is)"
