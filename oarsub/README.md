# OAR / GRICAD submission scripts

HPC job scripts for the GRICAD clusters (OAR resource manager). Each `*.sh` is a
self-contained job: it sets up the environment, `cd`s into the repo, and runs
one `hod_mod` fitting/analysis command. The `#OAR` directives at the top declare
the job name, **project (mandatory)**, resources and log files.

Docs: <https://gricad-doc.univ-grenoble-alpes.fr/hpc/joblaunch/job_management/>
and <https://gricad-doc.univ-grenoble-alpes.fr/hpc/description/>.

## Which machine?

| Cluster | Hardware | Use for |
| --- | --- | --- |
| **Dahu** | CPU nodes (Intel Xeon, ~32 cores/node), OmniPath 100 Gb | **CPU-heavy workloads — use this** |
| Bigfoot | V100 / A100 GPU nodes | deep-learning / GPU workloads |
| Luke | heterogeneous / visualization | specialised needs |

**These fits run on Dahu (CPU).** The likelihood is JAX-on-CPU + (cached) CAMB
and the `emcee` sampler is **serial** (no multiprocessing pool), so a GPU node
gives no benefit — the per-step arrays are tiny and a single walker loop cannot
saturate a GPU. We therefore request a **few CPU cores on one Dahu node**
(`/nodes=1/core=8`); those cores are used by JAX/XLA + BLAS *within* each
likelihood evaluation. More cores give little extra speed-up unless the sampler
is changed to evaluate walkers in parallel with a process pool (possible future
improvement — would let 32 walkers use up to 32 cores).

## Submitting

`--project` is mandatory on GRICAD. Either edit the `#OAR --project PROJECTNAME`
line in the script, or pass it on the command line:

on bigfoot:

oarsub -T
OAR_API_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiY29tcGFyYWoiLCJleHAiOjE3ODI5MDk5NTIsImRhdGUiOiIyMDI2LTA2LTI0IDEyOjQ1OjUyIn0.2IFh654D7OFXs6cP6vbG-6j4FAcZmyGb1dYKbgGojsQ

gridclusters
8: dahu
9: bigfoot
11: kraken-cpu
12: kraken-gpu

gridtoken -i 8 -t "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiY29tcGFyYWoiLCJleHAiOjE3ODI5MDk5NTIsImRhdGUiOiIyMDI2LTA2LTI0IDEyOjQ1OjUyIn0.2IFh654D7OFXs6cP6vbG-6j4FAcZmyGb1dYKbgGojsQ"
New token registered.

OAR_API_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiY29tcGFyYWoiLCJleHAiOjE3ODI5MTg2MjMsImRhdGUiOiIyMDI2LTA2LTI0IDE1OjEwOjIzIn0.abax6Y6K8J-1YFFbDClvvCmFiqwCpMYnGrdlynvVTi0

gridtoken -i 8 -t "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiY29tcGFyYWoiLCJleHAiOjE3ODI5MTg2MjMsImRhdGUiOiIyMDI2LTA2LTI0IDE1OjEwOjIzIn0.abax6Y6K8J-1YFFbDClvvCmFiqwCpMYnGrdlynvVTi0"

```bash
oarsub --project pr-orphans -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh
```

Quick test on the dev partition (≤ 30 min), then the real run:

```bash
oarsub -t devel -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh     # smoke test
oarsub --project pr-orphans -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh
```

Monitor / manage:

```bash
oarstat -u $USER
oarstat -fj <jobid>
oardel <jobid>
tail -f oarsub/logs/<jobid>.bgs_zm15_joint_mcmc.out
```

Before first submission, edit the config block at the top of the script:
`REPO`, `DATA_DIR`, `CONDA_ENV`, and the `--project` directive. Logs land in
`oarsub/logs/`.

## Walltime and restart (resumable chains)

The script sets a short `walltime=04:00:00` (re-submit to continue — see below).
The MCMC is **checkpointed every step**:
`JointZM15.sample` writes burn-in + production as one continuous `emcee` HDF
backend (`<out-dir>/chain.h5`), flushed after every iteration. If the job is
killed by the walltime, **just re-submit the same script** — it reads
`chain.h5`, sees how many steps survived, and runs only the remainder. Burn-in
is discarded only at read-out (`flatchain.npz`). So the chosen walltime is not a
correctness constraint, only a convenience.

For hands-off auto-resubmission, uncomment the `#OAR -t besteffort` /
`#OAR -t idempotent` directives: the resumable chain makes the job idempotent,
so CiGri/OAR can restart it automatically after a kill. A manual loop works too:

```bash
until grep -q "done=" oarsub/logs/*.bgs_zm15_joint_mcmc.out 2>/dev/null; do
    oarsub --project pr-orphans -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh
    sleep <until-this-job-ends>
done
```

## Scripts

| Script | Cluster | What it does |
| --- | --- | --- |
| `fit_bgs_zm15_joint_mcmc.sh` | Dahu (CPU) | ZM15 joint `wp + n_gal` MCMC (M\* > 10¹⁰ bins), `rp ∈ [0.5, 20]`, 32 walkers × (500 burn-in + 2000 steps). Resumable. |
| `fit_comparat2025_gas_shape.sh` | Dahu (CPU) | Comparat+2025 fixed-ZM15 **MAP**, S1, `--free-params gas-shape` (gas density α-slopes). |
| `fit_comparat2025_gas_temp.sh`  | Dahu (CPU) | …`gas-temp` (gas density α-slopes + pressure α_out, P_0.3, γ). |
| `fit_comparat2025_gas_full.sh`  | Dahu (CPU) | …`gas-full` (all DPM gas params: density + pressure + metallicity). |
| `fit_comparat2025_agn_occ.sh`   | Dahu (CPU) | …`agn-occ`, `--agn-model hod` (HOD-AGN occupation). |
| `fit_comparat2025_agn_lum.sh`   | Dahu (CPU) | …`agn-lum`, `--agn-model ham` (luminosity overrides; degenerate). |


DONE MAP, MCMC ONGOING : 
```bash
oarsub --project pr-orphans -S ./oarsub/fit_comparat2025_gas_shape.sh
oarsub --project pr-orphans -S ./oarsub/fit_comparat2025_agn_lum.sh
oarsub --project pr-orphans -S ./oarsub/fit_comparat2025_agn_occ.sh
```


The 5 scripts (Dahu / CPU)
Script	preset / model	cores	walltime
fit_comparat2025_agn_lum.sh	agn-lum / ham	16	4h
fit_comparat2025_gas_shape.sh	gas-shape / hod	16	6h
fit_comparat2025_gas_temp.sh	gas-temp / hod	16	10h
fit_comparat2025_agn_occ.sh	agn-occ / hod	16	10h
fit_comparat2025_gas_full.sh	gas-full / hod	16	18h
Each reuses the existing mamba bootstrap + thread/XLA env, runs --sample S1 --fix-zm15 --mode map, writes to its own results/fits/comparat2025_fixedZM15_<preset>/, and is chmod +x. oarsub/README.md updated with the table, sizing, and caveats.

Optimal configuration (decided + measured)
16 cores / 1 Dahu node: the MAP optimiser is a single serial L-BFGS-B process — cores only feed XLA/BLAS within each eval (modest arrays), so 16 (half a node, low fragmentation) is the sweet spot.
Walltimes set from a measured per-eval cost (~18s/eval after a ~120s first JIT trace, not the 110s worst case I first feared), with margin since MAP is not resumable (unlike the MCMC jobs — so no besteffort/idempotent). Max Dahu walltime is 48h; all jobs fit comfortably.
Two issues I found and handled during calibration
The full-APEC gas path produces NaN. Activating pressure+metallicity+cooling makes the entire C_ℓ^{gX} non-finite (a downstream normalization bug — both emissivity functions are individually finite). That would crash the gas jobs at startup. I switched the gas presets to the robust density-only path (emissivity_uk ∝ n_e², what amps/all use). Verified: gas-shape is finite and α_out reshapes w_θ by 100×. Consequence: the density-profile params (α_out/α_in/α_tr, n_e,0.3, β) are live; the pressure/temperature/metallicity params of gas-temp/gas-full are inert until the full-APEC normalization is fixed (documented in the README). I can fix that path as a follow-up if you want gas-temp/gas-full to exercise the gas thermodynamics.

Walltimes revised down (10/16/40h → 6/10/18h) once the real ~18s/eval cost was measured.

Two heads-ups: I set --project pr-orphans (from your README) — edit if wrong; and the README still contains live OAR_API_TOKEN JWTs committed in git (lines ~36–49) that you should rotate and scrub from history.

Submit with oarsub --project pr-orphans -S ./oarsub/fit_comparat2025_<preset>.sh (smoke-test first with -t devel).


### `fit_comparat2025_*.sh` — fixed-ZM15 X-ray MAP presets

Five MAP fits of the Comparat+2025 model to the S1 (M\* > 10¹⁰) galaxy × eROSITA
`w_θ` cross-correlation, with the ZM15 galaxy connection held fixed
(`--fix-zm15`, from `results/bgs_zm15_joint_wp_ngal/map_result.json`). Each frees a
different gas/AGN `--free-params` preset and writes to its own
`results/fits/comparat2025_fixedZM15_<preset>/` (`S1_map.json` + figures).

**These are MAP (`scipy` L-BFGS-B) fits — NOT resumable** (no checkpoint, unlike the
MCMC chain). The walltime must cover the whole optimisation; do **not** use
`besteffort`/`idempotent` (a kill loses all progress). Sizing comes from the per-eval
cost, which is dominated by `angular_cl_gX`:

- `agn-lum` has no profile/AGN rebuild → the first JAX trace (~90 s) then ~2 s/eval →
  **minutes** total (walltime 4 h, generous).
- `gas-*` rebuild the full DPM gas stack (density + pressure + metallicity, full-APEC
  emissivity) every eval and `agn-occ` rebuilds the HOD-AGN abundance match —
  ~25–40 s/eval after a ~120 s first trace (+ a one-time ~10 s APEC cooling-table build),
  so MAP runs **a few hours**: walltimes 6 h (`gas-shape`), 10 h (`gas-temp`, `agn-occ`),
  18 h (`gas-full`; 14 params → extra margin). All well under the Dahu 48 h cap.

All use `/nodes=1/core=16`: the optimiser is a single serial process and the cores only
feed JAX/XLA + BLAS within each likelihood eval (modest arrays), so 16 (half a Dahu node)
is the sweet spot — more gives little.

> **Caveat — degenerate parameters.** The likelihood is `w_θ`-only, so amplitude-degenerate
> parameters sit as flat directions: the `agn-lum` luminosity params (`scatter_lx`,
> `log10_A_kcorr`, `log10_A_dc`) and the gas **normalisation** `log10_ne_03`. Everything
> else moves the fit — the gas **shape** params (`alpha_out`/`alpha_in`/`alpha_tr`), the
> **pressure/temperature/metallicity** params (`alpha_out_pressure`, `log10_P_03`, `Z_0`,
> via the full-APEC emissivity, now fixed — they change `w_θ` by tens of %),
> `beta_gas`/`beta_pressure`, and the `agn-occ` occupation. Smoke-test with `-t devel` first.

### `fit_bgs_zm15_joint_mcmc.sh`

Run the **MAP stage first** so the walkers seed from the best fit
(`map_result.json` in the out-dir):

```bash
python -m hod_mod.scripts.fitting.bgs_ls10.fit_bgs_zm15_joint \
    --data-dir ~/software/sum_stat/data/BGS_Mstar10_massbins \
    --rp-min 0.5 --rp-max 20 \
    --surveys --mode map --out-dir results/bgs_zm15_joint_wp_ngal
```

then submit the MCMC job (same `--out-dir`). To do MAP + MCMC in a single job,
change `--mode mcmc` to `--mode both` in the script.
