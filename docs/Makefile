SPHINXBUILD = sphinx-build
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
