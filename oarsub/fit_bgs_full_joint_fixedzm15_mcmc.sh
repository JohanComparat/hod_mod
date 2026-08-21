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

#OAR --name bgs_full_joint_fixedzm15
#OAR --project your-oar-project
#OAR -l /nodes=1/core=16,walltime=06:00:00
#OAR --stdout oarsub/logs/%jobid%.bgs_full_joint_fixedzm15.out
#OAR --stderr oarsub/logs/%jobid%.bgs_full_joint_fixedzm15.err
#OAR -t besteffort
#OAR -t idempotent

set -euo pipefail

REPO="${HOME}/software/hod_mod"
CONDA_ENV="hod_mod"

# --- data / results roots (the COMPLETE data root includes xray_bands/) ------
export HOD_MOD_DATA_DIR="${HOME}/data/hod_mod_data"
# defines VTAG + HOD_MOD_RESULTS; must precede OUT_DIR, which uses both
# Optional campaign tag as $1: OAR does NOT propagate the submitting shell's
# environment to the node, so `VTAG=v0.3 oarsub -S ./script.sh` would silently
# run as the default.  Pass it as an argument instead:
#   oarsub --project P -S "./oarsub/fit_bgs_full_joint_fixedzm15_mcmc.sh v0.3"
VTAG="${1:-${VTAG:-v0.31}}"
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
OUT_DIR="${HOD_MOD_RESULTS}/bgs_full_joint_fixedzm15_${VTAG}"

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

# Skip the MAP if it is already done: if map_result.json exists in OUT_DIR run
# MCMC only (resumes chain.h5); otherwise run the MAP first (--mode both) to seed
# the walkers.  So a first submit does MAP+MCMC and every idempotent/besteffort
# re-submit continues from the chain without ever redoing the MAP.
if [ -f "${OUT_DIR}/map_result.json" ]; then
    MODE=mcmc; echo "[job] map_result.json present -> MCMC only (resume)"
else
    MODE=both; echo "[job] no map_result.json -> MAP then MCMC"
fi
# Data selection: only ESD HSC on 2<rp<8, XLF z=0.1 with LX>41, AGN bias from
# Comparat23+Krumpe15 only; keep the broad + 15 narrow X-ray bands.  The AGN
# photon index agn_gamma is free (9th band param) so the continuum -- not hot gas
# -- absorbs the flat band spectrum; with the tight kt prior below, the posterior
# kT-M then tracks the cluster scaling.  ndim = 15.
# NOTE: changing the likelihood needs a fresh OUT_DIR (rm map_result.json chain.h5).
python -m hod_mod.scripts.fitting.fit_bgs_full_joint \
    --mode "${MODE}" \
    --esd-surveys HSC --esd-rp-max 8.0 \
    --xlf-z 0.1 --xlf-lx-min 41.0 \
    --agn-bias-refs Comparat23 Krumpe15 \
    --n-walkers 48 --n-burnin 500 --n-steps 2000 \
    --out-dir "${OUT_DIR}"

echo "done=$(date -Is)"
