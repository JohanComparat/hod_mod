# OAR / GRICAD submission scripts

HPC job scripts for the GRICAD clusters (OAR resource manager). Each `*.sh` is a
self-contained job: it sets up the environment, `cd`s into the repo, and runs
one `hod_mod` fitting/analysis command. The `#OAR` directives at the top declare
the job name, **project (mandatory)**, resources and log files.

Docs: <https://gricad-doc.univ-grenoble-alpes.fr/hpc/joblaunch/job_management/>
and <https://gricad-doc.univ-grenoble-alpes.fr/hpc/description/>.

## v0.3 re-run campaign (array jobs)

The v0.3.0 Hankel fix in `_pk_to_xi` moves every real-space observable
(`w_p` ≤ 19 %, `ΔΣ` ≤ 20 %, `Σ_y` ~16 %, `w_θ` cross), so **every fit and figure
built at ≤ 0.2.3 must be re-run.** Instead of one script per job, the campaign
uses **OAR array jobs**: a single generic wrapper `run_job.sh` plus one param
file per family (`params/*.txt`), each line a full `python -m ...` command that
`run_job.sh` `eval`s (so `$DATA_BGS` / `$RESULTS` expand portably). All fits
write to `*_v0.3` out-dirs, leaving the 0.2.3 results for a before/after diff.

**Precondition:** the cluster repo must be at the v0.3 commit (`git pull`), or the
jobs re-run the *old* transform. Edit the config block in `run_job.sh` for your
account first.

| Family | Param file / script | Sizing | Notes |
| --- | --- | --- | --- |
| A. benchmarks (MAP) | `params/benchmarks_map.txt` | `core=8, 2 h` | Guo18/19, Leauthaud/vanUitert/ZM15 ΔΣ, all Lange25 |
| A. benchmarks (MCMC) | `params/benchmarks_mcmc.txt` | `core=8, 8 h` | More2015 ×3, Kravtsov, Zheng, ZM15 + 7 multisample bins |
| B. production joint fits | `params/production_mcmc.txt` | `core=16, 24 h`, resumable | bgs_zm15_joint, thresh, comparat2025 |
| C. Comparat2025 X-ray | `fit_comparat2025_*.sh` (×5) | 4–18 h | unchanged, re-pointed to `_v0.3` |
| D. forecasts | `params/forecasts.txt` | `core=16, 24 h` | tier2/3/4, stage4, sensitivity, benchmark_map_mcmc |
| B. full-joint | `fit_bgs_full_joint_*_mcmc.sh` (×2) | see script | re-pointed to `_v0.3` |

Submit a family (smoke-test with `--devel` first) via the helper:

```bash
./oarsub/submit_campaign.sh your-oar-project benchmarks_map --devel   # 1-line smoke
./oarsub/submit_campaign.sh your-oar-project benchmarks_map
./oarsub/submit_campaign.sh your-oar-project all                      # everything
```

or by hand:

```bash
oarsub --project your-oar-project -l /nodes=1/core=8,walltime=02:00:00 \
       --array-param-file oarsub/params/benchmarks_map.txt -S ./oarsub/run_job.sh
```

When the chains land, sync the `_v0.3` out-dirs back, audit with
`./oarsub/campaign_status.sh` (per-fit and per-benchmark-model, so a
walltime-killed chain or a family that never ran is named rather than counted as
"OK, N files"; `SINCE=YYYY-MM-DD` also flags results predating the campaign),
then regenerate every figure with `./oarsub/collect_and_plot.sh` — which refuses
to run while anything is missing unless you pass `ALLOW_PARTIAL=1`, and refuses
to collect a different `VTAG` on top of the last one unless you pass
`FORCE_VTAG_SWITCH=1` (`docs/_images/.campaign_vtag` records which campaign the
figures come from). Then update the χ²/dof, best-fit params and the
AUM-agreement line in the affected `docs/*.rst` pages.

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

oarsub -T # gives you a token

gridclusters
8: dahu
9: bigfoot
11: kraken-cpu
12: kraken-gpu

gridtoken -i 8 -t "<TOKEN>"

```bash
oarsub --project your-oar-project -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh
```

Quick test on the dev partition (≤ 30 min), then the real run:

```bash
oarsub -t devel -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh     # smoke test
oarsub --project your-oar-project -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh
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

## OAR dialect and node heterogeneity

Two things this GRICAD build does differently, both learned the hard way:

* **`--name`, not `-n`.**  This `oarsub` rejects the short form; every script
  under `oarsub/` uses `#OAR --name`.
* **`[ANTIFRAG] resources may be heterogeneous.**  A job can land on any node
  in the default queue, and they are not the same machine.  Wall-clock is
  therefore **not comparable between jobs** unless you pin the CPU model:
  prefix the resource string with `/cpumodel=1/`, e.g.

  ```bash
  oarsub -l /cpumodel=1/nodes=1/core=8,walltime=24:00:00 ...
  ```

  Concretely: the full-joint MAP re-run of 2026-08-21 landed on `dahu-fat4`, a
  16-core Gold 6244 fat node, so its wall-clock cannot be compared against the
  8 h 30 m measured for the same fit on `dahu189`.  Pin the model whenever the
  timing is the measurement; leave it unpinned when throughput is.

## Walltime and restart (resumable chains)

The script sets a short `walltime=04:00:00` (re-submit to continue — see below).
The MCMC is **checkpointed every step**:
`JointZM15.sample` writes burn-in + production as one continuous `emcee` HDF
backend (`<out-dir>/chain.h5`), flushed after every iteration. If the job is
killed by the walltime, **just re-submit the same script** — it reads
`chain.h5`, sees how many steps survived, and runs only the remainder. Burn-in
is discarded only at read-out (`flatchain.npz`). So the chosen walltime is not a
correctness constraint, only a convenience.

`--mode both` re-runs are cheap on restart: if `map_result.json` already exists
the MAP optimisation and its figure set are **skipped** and the job goes straight
to resuming the chain (`--force-map` to redo it, `--plot-only` for just the
figures). Without that skip every besteffort restart spent its first hours
re-fitting the same deterministic Powell MAP before touching the sampler, which
is how the 2026-07 campaign reached only ~300/2500 steps in ten days.

For hands-off auto-resubmission, uncomment the `#OAR -t besteffort` /
`#OAR -t idempotent` directives: the resumable chain makes the job idempotent,
so CiGri/OAR can restart it automatically after a kill. A manual loop works too:

```bash
until grep -q "done=" oarsub/logs/*.bgs_zm15_joint_mcmc.out 2>/dev/null; do
    oarsub --project your-oar-project -S ./oarsub/fit_bgs_zm15_joint_mcmc.sh
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


Sizing of the five presets (Dahu / CPU, `--sample S1 --fix-zm15 --mode map`,
each writing to its own `results/fits/comparat2025_fixedZM15_<preset>/`):

| Script | preset / model | cores | walltime |
| --- | --- | --- | --- |
| `fit_comparat2025_agn_lum.sh` | agn-lum / ham | 16 | 4 h |
| `fit_comparat2025_gas_shape.sh` | gas-shape / hod | 16 | 6 h |
| `fit_comparat2025_gas_temp.sh` | gas-temp / hod | 16 | 10 h |
| `fit_comparat2025_agn_occ.sh` | agn-occ / hod | 16 | 10 h |
| `fit_comparat2025_gas_full.sh` | gas-full / hod | 16 | 18 h |

Rationale (measured): the MAP optimiser is a single serial L-BFGS-B process —
cores only feed XLA/BLAS within each eval, so 16 (half a Dahu node, low
fragmentation) is the sweet spot. Walltimes follow from ~18 s/eval after a
~120 s first JIT trace, with margin because MAP is **not resumable** (no
`besteffort`/`idempotent`). Max Dahu walltime is 48 h; all jobs fit.

Historical note: during the first calibration **the full-APEC gas path
produced NaN** (float32 underflow of Λ~1e-24 inside `C_ℓ^{gX}`), and the gas
presets were temporarily pinned to the density-only path. That is **fixed**
(float32-safe `safe_log` floor + the `emissivity_full_uk / Λ_ref`
renormalisation + a non-finite C_ℓ sanitizer in `fit_comparat2025`): the
`gas-temp`/`gas-full` presets now run the full DPM stack and their
pressure/temperature/metallicity parameters are live (they move `w_θ` by tens
of %). Guarded by the `cross_gX_full` regression test in
`tests/test_cross_spectra.py`.

> **Security note.** Early campaign commits briefly carried live
> `OAR_API_TOKEN` JWTs in this file. The token strings were scrubbed from the
> entire git history on 2026-07-17 (`git filter-repo --replace-text`), but any
> token that was ever pushed must be treated as compromised — rotate it on the
> GRICAD side if it has not already expired.

Submit with `oarsub --project your-oar-project -S ./oarsub/fit_comparat2025_<preset>.sh`
(smoke-test first with `-t devel`).


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
