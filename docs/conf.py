"""Sphinx configuration for hod_mod documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'hod_mod'
author = "Johan Comparat"
copyright = "2025, Johan Comparat"
release = '0.0.1'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = True

# Safety net for the ReadTheDocs build: if a heavy runtime dependency fails to
# import at doc-build time, mock it so autodoc can still import the modules.
# Only optional / non-JAX backends are mocked — JAX is left real so that
# jax.jit-decorated functions keep their signatures in the API docs.
autodoc_mock_imports = [
    "camb",
    "colossus",
    "AletheiaCosmo",
    "CEmulator",
    "aemulusnu_hmf",
    "soxs",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store",
    # Benchmark pages removed from navigation (keep files on disk but don't build)
    "benchmarks.rst",
    "benchmark_lange2025.rst",
    "benchmark_comparat2025.rst",
    "benchmark_zheng2007.rst",
    "benchmark_guo2018.rst",
    "benchmark_guo2019.rst",
    "benchmark_kravtsov2004.rst",
    "benchmark_leauthaud2012.rst",
    "benchmark_vanutert2016.rst",
    "benchmark_zumandelbaum2015.rst",
    "benchmark_zacharegkas2025.rst",
    "benchmarks_deltasigma.rst",
    "benchmarks_joint.rst",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}

rst_prolog = """
.. warning::

   This documentation is under construction. Content may be incomplete or subject to change.

"""

mathjax3_config = {
    "tex": {
        "macros": {
            "Msun": r"M_\odot",
        }
    }
}
