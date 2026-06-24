#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR job: BGS x LS10 — Zu & Mandelbaum 2015 joint wp + n_gal, MCMC.
#
# Cluster: DAHU (CPU platform).  This is a CPU-only workload — the likelihood is
# JAX-on-CPU + (cached) CAMB and the emcee sampler is serial, so a GPU node
# (bigfoot) would give no benefit.  See oarsub/README.md for the rationale and
# the recommended core/walltime sizing.
#
# Resumable: the emcee HDF backend (chain.h5) is flushed after EVERY step and
# burn-in+production form one continuous chain, so if the job is killed by the
# walltime you just re-submit this same script and it continues from where it
# stopped (see JointZM15.sample).
#
# Submit:   oarsub --project <PROJECT> -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh
#   (or set the --project directive below and run:  oarsub -S ./...sh)
# Quick test on the dev partition (<=30 min):
#           oarsub -t devel -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh
# =============================================================================

#OAR -n bgs_zm15_joint_mcmc
#OAR --project PROJECTNAME
#OAR -l /nodes=1/core=8,walltime=04:00:00
#OAR --stdout oarsub/logs/%jobid%.bgs_zm15_joint_mcmc.out
#OAR --stderr oarsub/logs/%jobid%.bgs_zm15_joint_mcmc.err
# --- auto-resubmit alternative (CiGri/besteffort): the resumable chain makes
#     this job idempotent, so it can be safely restarted by the scheduler.
# #OAR -t besteffort
# #OAR -t idempotent

set -euo pipefail

# --- user configuration (edit for your GRICAD account) ----------------------
REPO="${HOME}/software/hod_mod"                       # repo location on the cluster
DATA_DIR="${HOME}/software/sum_stat/data/BGS_Mstar10_massbins"
CONDA_ENV="hod_mod"                                   # conda/mamba env name
OUT_DIR="results/bgs_zm15_joint_wp_ngal"              # repo-relative

# --- environment ------------------------------------------------------------
# GRICAD provides conda via /applis; activate the project env.
source /applis/environments/conda.sh
conda activate "${CONDA_ENV}"

# emcee is serial; let JAX/XLA + BLAS use the allocated cores for the per-step
# halo-model linear algebra.
NCORES="${OAR_RES_NB_CORES:-8}"
export OMP_NUM_THREADS="${NCORES}"
export OPENBLAS_NUM_THREADS="${NCORES}"
export MKL_NUM_THREADS="${NCORES}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_PLATFORMS=cpu

cd "${REPO}"
mkdir -p oarsub/logs "${OUT_DIR}"

echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  start=$(date -Is)"

# Seeds the walkers from the MAP best fit (map_result.json in OUT_DIR) when
# present — run the MAP stage first, or change --mode to 'both' for a single job.
python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint \
    --data-dir "${DATA_DIR}" \
    --rp-min 0.5 --rp-max 20 \
    --surveys \
    --mode mcmc \
    --n-walkers 32 --n-burnin 500 --n-steps 2000 \
    --out-dir "${OUT_DIR}"

echo "done=$(date -Is)"
