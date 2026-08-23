#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR job: Comparat+2025 fixed-ZM15 X-ray fit — S1, MAP, agn-occ.
#
# Cluster: DAHU (CPU).  fit_comparat2025 is JAX-on-CPU; a GPU node gives no
# benefit.  This is a MAP (scipy L-BFGS-B) fit and is **NOT resumable** (unlike
# the MCMC jobs), so the walltime must cover the whole optimisation.
#
# agn-occ frees the HOD-AGN occupation (f_inc, log10mmin, sigma_logm, alpha);
# each rebuilds the HODAgnModel abundance match + a fresh angular_cl_gX trace
# (~120 s/eval), so the MAP takes several hours.  Requires --agn-model hod.
# See oarsub/README.md.
#
# Submit:  oarsub --project your-oar-project -S ./oarsub/fit_comparat2025_agn_occ.sh
# Smoke:   oarsub -t devel -S ./oarsub/fit_comparat2025_agn_occ.sh     (<=30 min)
# =============================================================================

#OAR --name c2025_agn_occ
#OAR --project your-oar-project
#OAR -l /nodes=1/core=16,walltime=10:00:00
#OAR --stdout oarsub/logs/%jobid%.c2025_agn_occ.out
#OAR --stderr oarsub/logs/%jobid%.c2025_agn_occ.err

set -euo pipefail

# --- user configuration (edit for your GRICAD account) ----------------------
REPO="${HOME}/software/hod_mod"                       # repo location on the cluster
CONDA_ENV="hod_mod"                                   # conda/mamba env name
# Optional campaign tag as $1: OAR does NOT propagate the submitting shell's
# environment to the node, so `VTAG=v0.3 oarsub -S ./script.sh` would silently
# run as the default.  Pass it as an argument instead:
#   oarsub --project P -S "./oarsub/fit_comparat2025_agn_occ.sh v0.3"
VTAG="${1:-${VTAG:-v0.4}}"

# Optional second positional arg `ecf` (0.4.0): fold the validated eROSITA TM0
# flux->count ECF into the cross-power and anchor the sample-independent
# geometry on S1, so log10_A_gas / log10_A_AGN come out as O(1) residuals
# instead of absorbing the whole ~1e8 model->counts chain.  Written to a
# separate _ecf out-dir so the ECF and non-ECF fits live side by side and stay
# directly comparable.  Passed unquoted below: fixed literal, no word splitting
# to worry about, and an empty array under `set -u` is not portable.
ECF_FLAG=""; ECF_SUFFIX=""
if [ "${2:-}" = "ecf" ]; then ECF_FLAG="--ecf"; ECF_SUFFIX="_ecf"; fi
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
campaign_require_zm15_json
OUT_DIR="${HOD_MOD_RESULTS}/fits/comparat2025_fixedZM15_agn-occ${ECF_SUFFIX}_${VTAG}"

# --- environment ------------------------------------------------------------
# GRICAD provides conda via /applis; activate the project env.
export MAMBA_EXE='/home/your-cluster-login/miniforge3/bin/mamba';
export MAMBA_ROOT_PREFIX='/home/your-cluster-login/miniforge3';
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
    --sample S1 --fix-zm15 "${ZM15_JSON}" --mode map \
    --free-params agn-occ --agn-model hod ${ECF_FLAG} \
    --out-dir "${OUT_DIR}"

echo "done=$(date -Is)"
