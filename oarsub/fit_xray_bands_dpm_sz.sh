#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR job: native-DPM X-ray bands x tSZ joint fit — S1..S7, MCMC.
#
# Cluster: DAHU (CPU).  Everything here is JAX-on-CPU; a GPU node gives no
# benefit (see reference_cpu_gpu_benchmark).
#
# What this fits.  The band model is re-based onto the NATIVE DPM gas
# parameters (Oppenheimer+2025):
#
#     n_e(r,M,z) = n_e,0.3 f_n(x) E(z)^gamma_n M_12^beta_n
#     P (r,M,z) = P_0.3   f_P(x) E(z)^gamma_P M_12^beta_P
#     T (r,M,z) = P/n_e
#
# so ONE gas model drives both legs: (P_0.3, beta_P) set the pressure the tSZ
# Sigma_y integrates AND the X-ray temperature T = P/n_e; (n_e,0.3, beta_n) set
# the density the X-ray emission measure integrates AND the same temperature.
# Vary once, both observables move.  That coupling is the point of the job.
#
# Priors: the GAS.py LX-M / kT-M priors are back-propagated onto the native
# parameters with JAX autodiff (hod_mod.fitting.dpm_priors), giving a FULL-
# COVARIANCE Gaussian (correlations 0.71-0.95 — e.g. kT constrains the RATIO
# P_0.3/n_e,0.3, so those two must move together).  Widened x1.5.
#
# Expect tension, not a clean fit.  The X-ray bands pull the gas HOT (the
# long-standing "gas runs hot" result: X-ray-only, log10_p03 rails its bound),
# while the DPM pressure calibrated to X-ray UNDER-predicts the observed Sigma_y
# by ~30%.  Both now act on the same P_0.3.  Read the posterior with that in mind.
#
# Cost.  Three caches are built on first run and reused after:
#   * dpm_j_grid.npz        ~90 s   (shared by all samples)
#   * <S>_transfer.npz      ~20 min per sample  (X-ray band transfer)
#   * <S>_sz_sumstat_transfer.npz  ~3-5 min per sample  (Sigma_y kernel)
# So a cold 7-sample run spends ~3 h in precompute before the MCMC starts.
# The MCMC itself is cheap per eval (both legs are precomputed kernels + an
# analytic weight): no FT, no APEC call, per-eval work is O(matrix-vector).
#
# REQUIRES float64: the emission integral carries r_cm^2 ~ 1e49, which silently
# overflows float32 to inf.  JAX_ENABLE_X64=1 below is not optional.
#
# Submit:  oarsub --project your-oar-project -S ./oarsub/fit_xray_bands_dpm_sz.sh
# Smoke:   oarsub -t devel -S "./oarsub/fit_xray_bands_dpm_sz.sh v0.31 S1"   (<=30 min)
# =============================================================================

#OAR --name xband_dpm_sz
#OAR --project your-oar-project
#OAR -l /nodes=1/core=16,walltime=24:00:00
#OAR --stdout oarsub/logs/%jobid%.xband_dpm_sz.out
#OAR --stderr oarsub/logs/%jobid%.xband_dpm_sz.err

set -euo pipefail

# --- user configuration (edit for your GRICAD account) ----------------------
# REPO is derived from THIS SCRIPT's location, not hard-coded to
# ~/software/hod_mod, so the job runs the checkout it was submitted from.
#
# Why that matters: every job does `cd $REPO && python -m hod_mod...`, and
# `python -m` puts the CWD first on sys.path — so the running branch of that
# working tree IS the code being fitted.  Checking this branch out in the shared
# ~/software/hod_mod would therefore change the code under any campaign job that
# starts (or, for besteffort, RESTARTS) afterwards — full_joint in particular
# imports fit_xray_joint_bands, which this branch edits.  Run it from a
# `git worktree` instead and the campaign is untouched:
#
#   git worktree add ~/software/hod_mod_dpmsz feature/dpm-native-xray-sz
#   cd ~/software/hod_mod_dpmsz
#   oarsub --project your-oar-project -S "./oarsub/fit_xray_bands_dpm_sz.sh v0.31"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="hod_mod"
# OAR does NOT propagate the submitting shell's environment to the node, so pass
# the campaign tag and the sample list as ARGUMENTS, not env vars:
#   oarsub --project P -S "./oarsub/fit_xray_bands_dpm_sz.sh v0.31 S1 S4 S7"
VTAG="${1:-${VTAG:-v0.4}}"
shift || true
SAMPLES=("${@:-S1 S2 S3 S4 S5 S6 S7}")
if [ "$#" -eq 0 ]; then SAMPLES=(S1 S2 S3 S4 S5 S6 S7); fi

source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
# The band transfer needs the ZM15 MAP: _precompute calls B._build_hod_params()
# and builds a DutyCycleAGNModel, whose _ZM15_MAP_JSON resolves through
# results_root() AT IMPORT.  A versioned tree starts empty, so _campaign_env.sh
# symlinks the reference in — guard anyway rather than discover it 20 min into
# the transfer build, having already burned the slot.
campaign_require_zm15_json
OUT_DIR="${HOD_MOD_RESULTS}/fits/xray_bands_dpm_sz_${VTAG}"

# --- environment ------------------------------------------------------------
export MAMBA_EXE='/home/your-cluster-login/miniforge3/bin/mamba';
export MAMBA_ROOT_PREFIX='/home/your-cluster-login/miniforge3';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="$MAMBA_EXE"
fi
unset __mamba_setup
mamba activate "${CONDA_ENV}"

NCORES="${OAR_RES_NB_CORES:-16}"
export OMP_NUM_THREADS="${NCORES}"
export OPENBLAS_NUM_THREADS="${NCORES}"
export MKL_NUM_THREADS="${NCORES}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export JAX_PLATFORMS=cpu
# NOT optional — see the header: r_cm^2 ~ 1e49 overflows float32 to inf and the
# emission integral silently returns nan/inf.
export JAX_ENABLE_X64=1

# The Sigma_y stack lives in the sum_stat repo (BGS_SZ/); _campaign_env.sh
# defaults HOD_MOD_SUMSTAT, but fail loudly rather than fit X-ray-only by
# accident if it is missing.
if [ ! -d "${HOD_MOD_SUMSTAT}/BGS_SZ" ]; then
    echo "FATAL: no BGS_SZ stack under HOD_MOD_SUMSTAT=${HOD_MOD_SUMSTAT}" >&2
    echo "       The SZ leg would be silently skipped and this would be an" >&2
    echo "       X-ray-only fit — i.e. not the job you meant to submit." >&2
    exit 2
fi

cd "${REPO}"
mkdir -p oarsub/logs "${OUT_DIR}"

echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  start=$(date -Is)"
echo "vtag=${VTAG}  samples=${SAMPLES[*]}"
echo "results=${HOD_MOD_RESULTS}  sumstat=${HOD_MOD_SUMSTAT}"
git rev-parse HEAD | sed 's/^/commit=/'

python -m hod_mod.scripts.fitting.fit_xray_joint_bands \
    --samples "${SAMPLES[@]}" \
    --sz --sz-source sumstat \
    --candidate baseline \
    --mcmc --nwalkers 64 --nsteps 8000 --nburn 2000

echo "done=$(date -Is)"
