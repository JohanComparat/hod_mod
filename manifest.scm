;; manifest.scm — GNU Guix environment for hod_mod (no conda).
;;
;; This manifest is meant to be used inside a Guix *container* with FHS-free
;; LD_LIBRARY_PATH wiring (see INSTALL_GUIX.md), e.g.:
;;
;;     guix shell --container --network -m manifest.scm
;;
;; or, pinned for reproducibility (see channels.scm):
;;
;;     guix time-machine -C channels.scm -- shell --container --network -m manifest.scm
;;
;; Strategy: Guix provides a hermetic, reproducible Python interpreter, the
;; C/Fortran toolchain, and the small set of shared libraries that PyPI
;; "manylinux" wheels load at runtime (zlib, libstdc++ via gcc-toolchain,
;; libgfortran via gfortran-toolchain). The Python libraries themselves are
;; installed with pip into a virtualenv. This deliberately avoids relying on
;; Guix's python-numpy/-scipy/-h5py, whose ABIs can be transiently incoherent on
;; the rolling `master` channel (e.g. during a numpy 1.x -> 2.x migration), and
;; also covers deps Guix does not package at all (jax/jaxlib, camb, colossus,
;; AletheiaCosmo). pip resolves a self-consistent set of wheels instead.
;;
;; The container's binaries are all from this profile, so pointing
;; LD_LIBRARY_PATH at "$GUIX_ENVIRONMENT/lib" lets the Guix Python's loader find
;; libz/libstdc++/libgfortran for the wheels without clobbering glibc.
;;
;; If a package name does not resolve on your channel, find the right one with:
;;
;;     guix search <name>

(specifications->manifest
 (list
  ;; --- shell utilities (needed inside --container, where only profile
  ;;     packages exist) ---
  "bash"
  "coreutils"
  "grep"
  "sed"
  "which"
  "findutils"

  ;; --- Python interpreter & packaging tools ---
  ;; python-wrapper provides the bare `python` command (the `python` package
  ;; only ships `python3`); it propagates the interpreter, so >= 3.11.
  "python-wrapper"
  "python-pip"
  "python-setuptools"
  "python-wheel"
  "python-virtualenv"

  ;; --- toolchain + runtime libs for manylinux wheels / source builds ---
  "gcc-toolchain"        ; libstdc++, libgcc_s, C compiler
  "gfortran-toolchain"   ; libgfortran (scipy), Fortran compiler (camb)
  "make"
  "pkg-config"
  "zlib"))               ; libz.so.1 — required by the numpy wheel
