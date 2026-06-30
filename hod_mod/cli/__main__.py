"""``hod-mod`` / ``python -m hod_mod`` command dispatcher.

Thin delegation layer: each subcommand re-runs the corresponding script module as
``python -m <module>`` with the forwarded arguments (via :func:`runpy.run_module`),
so the scripts keep their own argument parsers and behaviour unchanged.
"""
from __future__ import annotations

import runpy
import sys

# subcommand -> target module run as __main__
COMMANDS: dict[str, str] = {
    "fit":        "hod_mod.scripts.fitting.run_fit",
    "fit-cross":  "hod_mod.scripts.fitting.fit_comparat2025",
    "fit-joint":  "hod_mod.scripts.fitting.fit_joint_lsdr10",
    "benchmark":  "hod_mod.scripts.benchmarks.run_benchmark",
    "predict":    "hod_mod.scripts.direct_prediction_gal_gas_agn",
}

# `hod-mod validate <name>` -> target module
VALIDATE: dict[str, str] = {
    "sz-xray":      "hod_mod.scripts.validate_sz_xray",
    "gas-profiles": "hod_mod.scripts.validate_gas_profiles",
    "comparat2025": "hod_mod.scripts.validate_comparat2025",
    "arnaud2010":   "hod_mod.scripts.validate_arnaud2010",
    "amodeo2021":   "hod_mod.scripts.validate_amodeo2021",
    "pandey2025":   "hod_mod.scripts.validate_pandey2025",
    "oppenheimer2025": "hod_mod.scripts.validate_oppenheimer2025",
    "bnl":          "hod_mod.scripts.validate_bnl",
}

_HELP = """\
hod-mod — HOD galaxy clustering, lensing and X-ray/tSZ cross-correlations

usage: hod-mod <command> [options]

commands:
  fit         config-driven MAP/MCMC fit (wp / ESD / joint)
  fit-cross   galaxy x AGN/gas soft-X-ray cross-correlation fit
  fit-joint   joint wp + ESD fit (LS DR10)
  benchmark   run a validation benchmark (e.g. --model more2015_logM11_12)
  predict     forward-model prediction (galaxy + gas + AGN)
  validate    validation figures: {targets}

Run 'hod-mod <command> --help' for that command's options.
""".format(targets=" ".join(sorted(VALIDATE)))


def _run(module: str, prog: str, args: list[str]) -> None:
    sys.argv = [prog, *args]
    runpy.run_module(module, run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_HELP)
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "validate":
        if not rest or rest[0] in ("-h", "--help"):
            print("usage: hod-mod validate <target> [options]\n\ntargets:\n  "
                  + "\n  ".join(sorted(VALIDATE)))
            return 0
        target = rest[0]
        if target not in VALIDATE:
            print(f"hod-mod: unknown validate target {target!r}. "
                  f"Choose from: {', '.join(sorted(VALIDATE))}", file=sys.stderr)
            return 2
        _run(VALIDATE[target], f"hod-mod validate {target}", rest[1:])
        return 0

    if cmd in COMMANDS:
        _run(COMMANDS[cmd], f"hod-mod {cmd}", rest)
        return 0

    print(f"hod-mod: unknown command {cmd!r}. Run 'hod-mod --help'.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
