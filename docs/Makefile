# Use the active Python's Sphinx (robust: no dependency on the sphinx-build
# console script being on PATH). Activate the env first, e.g. `mamba activate
# hod_mod`, or override:  make html PYTHON=/path/to/python
PYTHON      ?= python
SPHINXBUILD ?= $(PYTHON) -m sphinx
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean serve

help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR)

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Build finished. Open _build/html/index.html"

clean:
	rm -rf $(BUILDDIR)

serve: html
	cd $(BUILDDIR)/html && python -m http.server 8080
