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
