"""Unified command-line interface for hod_mod.

A single front door (``hod-mod`` console script, or ``python -m hod_mod``) that
dispatches to the package's runnable entry points, grouped by observable pipeline:

* ``hod-mod fit``        — config-driven MAP/MCMC fit (wp / ESD / joint)
* ``hod-mod fit-cross``  — galaxy × AGN/gas X-ray cross-correlation fit
* ``hod-mod benchmark``  — run a validation benchmark from the registry
* ``hod-mod predict``    — forward-model prediction (galaxy + gas + AGN)
* ``hod-mod validate``   — gas / X-ray / tSZ validation figures

Each subcommand forwards its remaining arguments to the underlying script, so
``hod-mod <cmd> --help`` shows that command's own options.
"""

from .__main__ import main

__all__ = ["main"]
