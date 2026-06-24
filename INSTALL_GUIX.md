# Installing `hod_mod` with GNU Guix (no conda)

This is a conda-free, reproducible way to build a development/runtime environment for
`hod_mod` using [GNU Guix](https://guix.gnu.org/manual/1.5.0/fr/guix.fr.html).

## How it works

`hod_mod` is pure Python (Python ≥ 3.11, no compiled extensions of its own), but it
depends on `jax`/`jaxlib`, `camb`, `colossus` and `AletheiaCosmo`, which are **not**
packaged by Guix, and on numpy/scipy/h5py, whose Guix ABIs can be transiently
incoherent on the rolling `master` channel (e.g. mid numpy 1.x → 2.x migration).

So rather than relying on Guix's Python libraries, this setup uses Guix to provide a
**hermetic, reproducible Python interpreter + C/Fortran toolchain + the few shared
libraries that PyPI "manylinux" wheels load** (zlib, libstdc++, libgfortran). The
Python libraries themselves are installed with `pip` into a virtualenv. `pip` resolves
a self-consistent set of wheels, and the work happens inside a Guix **container** so the
Guix Python's loader can find the wheels' libraries via `LD_LIBRARY_PATH` without
clobbering glibc.

Two files in the repo root drive this:

- [`manifest.scm`](manifest.scm) — the declarative list of Guix packages.
- [`channels.scm`](channels.scm) — pins the Guix revision for reproducibility.

## Prerequisites

A working Guix installation (`guix --version`). On a foreign distro, follow the
[binary install instructions](https://guix.gnu.org/manual/1.5.0/fr/guix.fr.html#Installation-binaire).

If you did a manual binary install, make sure these one-time steps were done (the
`guix-install.sh` script normally does them, but some setups miss them):

```bash
# Build users/group (needed by the daemon):
sudo groupadd --system guixbuild
for i in $(seq -w 1 10); do
  sudo useradd -g guixbuild -G guixbuild -d /var/empty -s /usr/sbin/nologin \
       -c "Guix build user $i" --system "guixbuilder$i";
done
sudo systemctl restart guix-daemon

# Authorize substitute servers so prebuilt binaries are downloaded
# (otherwise everything compiles from source):
KD=/var/guix/profiles/per-user/root/current-guix/share/guix
sudo guix archive --authorize < "$KD/ci.guix.gnu.org.pub"
sudo guix archive --authorize < "$KD/bordeaux.guix.gnu.org.pub"
```

## Step-by-step

```bash
# 0. (Optional, reproducibility) Pin channels.scm to YOUR current, known-good Guix:
#    guix describe -f channels > channels.scm

# 1. Enter a Guix container with network access.
#    Reproducible variant: prefix with  guix time-machine -C channels.scm --
guix shell --container --network -m manifest.scm

# --- everything below runs INSIDE the container ---

# 2. Create a virtualenv (lives in the repo, reused across sessions).
python -m venv .venv-guix
source .venv-guix/bin/activate

# 3. Let the Guix Python loader find the wheels' shared libraries.
export LD_LIBRARY_PATH="$GUIX_ENVIRONMENT/lib"

# 4. Install the Python dependencies from PyPI (wheels).
python -m pip install --upgrade pip
pip install numpy scipy astropy matplotlib h5py \
            jax jaxlib colossus AletheiaCosmo camb
pip install emcee pyyaml pandas   # the hod_mod[fitting] extra (needed for hod_mod.fitting)
pip install pytest pytest-cov sphinx sphinx-rtd-theme numpydoc   # the dev extra

# 5. Install hod_mod itself (editable, deps already satisfied above).
pip install --no-build-isolation --no-deps -e .

# 6. Verify.
JAX_PLATFORMS=cpu python -c "import hod_mod; print(hod_mod.__file__)"
JAX_PLATFORMS=cpu python -m pytest tests/ -q
```

## Re-entering the environment later

Re-enter the **same** container, re-activate the venv, and re-export the library path:

```bash
guix shell --container --network -m manifest.scm   # same manifest as before!
source .venv-guix/bin/activate
export LD_LIBRARY_PATH="$GUIX_ENVIRONMENT/lib"
```

## Notes and gotchas

- **The venv is bound to the exact Guix profile.** `.venv-guix/` records the absolute
  store path of the Guix Python. If you change `manifest.scm` (or the pinned channel),
  that path changes and the venv breaks — delete `.venv-guix/` and recreate it.
- **`LD_LIBRARY_PATH="$GUIX_ENVIRONMENT/lib"` is required** every session: it points the
  Guix Python's loader at libz/libstdc++/libgfortran for the manylinux wheels. It
  contains no glibc, so it does not clobber the C library. Do **not** set this outside a
  container — there it would be forced onto system binaries and crash them.
- **Use `--container`.** Inside it, all binaries come from the Guix profile, which is why
  the `LD_LIBRARY_PATH` trick is safe. `--network` is needed so `pip` can reach PyPI.
- **CPU-only JAX:** set `JAX_PLATFORMS=cpu`. The PyPI `jaxlib` wheel is the CPU build;
  install the CUDA variant instead if you need GPU.
- `.venv-guix/` is git-ignored; delete it to rebuild from scratch.

## Alternative: Guix-native packages

If you prefer Guix to provide numpy/scipy/etc. declaratively (no pip for those), pin
`channels.scm` to a Guix commit where the Python stack is coherent and add
`python-numpy`, `python-scipy`, … to `manifest.scm`. You will still need pip for
`jax`/`jaxlib`, `camb`, `colossus` and `AletheiaCosmo`, which Guix does not package.
The container + `LD_LIBRARY_PATH` recipe above avoids having to find such a commit.
