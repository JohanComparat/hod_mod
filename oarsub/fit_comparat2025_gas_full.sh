#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR job: Comparat+2025 fixed-ZM15 X-ray fit — S1, MAP, gas-full.
#
# Cluster: DAHU (CPU).  fit_comparat2025 is JAX-on-CPU; a GPU node gives no
# benefit.  This is a MAP (scipy L-BFGS-B) fit and is **NOT resumable** (unlike
# the MCMC jobs), so the walltime must cover the whole optimisation.
#
# gas-full frees every DPM gas parameter (density + pressure + metallicity); the
# full DPM gas stack is rebuilt per likelihood evaluation (~25-40 s/eval after the
# ~120 s first JAX trace), so a 14-parameter MAP runs a few hours — walltime 18 h
# leaves margin (the w_θ-degenerate log10_ne_03 + amplitude can slow convergence).
# See oarsub/README.md.  Density, pressure and metallicity params all reshape w_θ.
#
# Submit:  oarsub --project pr-orphans -S ./oarsub/fit_comparat2025_gas_full.sh
# Smoke:   oarsub -t devel -S ./oarsub/fit_comparat2025_gas_full.sh    (<=30 min)
# =============================================================================

#OAR -n c2025_gas_full
#OAR --project pr-orphans
#OAR -l /nodes=1/core=16,walltime=18:00:00
#OAR --stdout oarsub/logs/%jobid%.c2025_gas_full.out
#OAR --stderr oarsub/logs/%jobid%.c2025_gas_full.err

set -euo pipefail

# --- user configuration (edit for your GRICAD account) ----------------------
REPO="${HOME}/software/hod_mod"                       # repo location on the cluster
CONDA_ENV="hod_mod"                                   # conda/mamba env name
OUT_DIR="results/fits/comparat2025_fixedZM15_gas-full"   # repo-relative

# --- environment ------------------------------------------------------------
# GRICAD provides conda via /applis; activate the project env.
export MAMBA_EXE='/home/comparaj/miniforge3/bin/mamba';
export MAMBA_ROOT_PREFIX='/home/comparaj/miniforge3';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
mamba activate "${CONDA_ENV}"

# Single serial L-BFGS-B process; let JAX/XLA + BLAS use the allocated cores for
# the per-eval halo-model linear algebra.
NCORES="${OAR_RES_NB_CORES:-8}"
export OMP_NUM_THREADS="${NCORES}"
export OPENBLAS_NUM_THREADS="${NCORES}"
export MKL_NUM_THREADS="${NCORES}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_PLATFORMS=cpu

cd "${REPO}"
mkdir -p oarsub/logs "${OUT_DIR}"

echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  start=$(date -Is)"

python -m hod_mod.scripts.fitting.fit_comparat2025 \
    --sample S1 --fix-zm15 --mode map \
    --free-params gas-full --agn-model hod \
    --out-dir "${OUT_DIR}"

echo "done=$(date -Is)"
