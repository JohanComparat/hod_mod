;; channels.scm — pin GNU Guix for reproducible hod_mod environments.
;;
;; This file pins the exact Guix revision used to resolve `manifest.scm`, so the
;; same package versions are obtained on any machine and at any date.
;;
;; To generate a channel file pinned to YOUR current, known-good Guix, run:
;;
;;     guix describe -f channels > channels.scm
;;
;; Then use it with the time machine (see INSTALL_GUIX.md):
;;
;;     guix time-machine -C channels.scm -- shell -m manifest.scm
;;
;; The commit below is a placeholder: replace it with a recent commit hash from
;; `guix describe` (or regenerate the whole file as shown above).

(list (channel
       (name 'guix)
       (url "https://git.guix.gnu.org/guix.git")
       (branch "master")
       ;; Pinned to a known-good revision. Regenerate for your machine with:
       ;;   guix describe -f channels > channels.scm
       (commit "230aa373f315f247852ee07dff34146e9b480aec")))
