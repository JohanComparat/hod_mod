"""Coverage for the Zenodo/pooch data fetcher (data_io.registry).

Exercises the offline resolution paths (local mirror, registry lookup, error
guards) without any network download.
"""
import pytest

from hod_mod.data_io.registry import (
    fetch, list_registry, zenodo_key, PLACEHOLDER_DOI,
)


def test_registry_lists_shipped_entries():
    r = list_registry()
    assert isinstance(r, dict) and len(r) >= 1
    assert "S1_mcmc_summary.json" in r


def test_zenodo_key_flattens_paths():
    # Zenodo file keys cannot contain '/', so nested paths flatten to '__';
    # already-flat names (the seed) pass through unchanged.
    assert zenodo_key("results/a/b/flatchain.npz") == "results__a__b__flatchain.npz"
    assert zenodo_key("S1_mcmc_summary.json") == "S1_mcmc_summary.json"


def test_fetch_local_mirror_returns_file(tmp_path, monkeypatch):
    """When $HOD_MOD_DATA_DIR holds the file, fetch returns it directly (no download)."""
    name = "S1_mcmc_summary.json"
    (tmp_path / name).write_text("{}")
    monkeypatch.setenv("HOD_MOD_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("HOD_MOD_DATA_BASEURL", raising=False)
    path = fetch(name)
    assert str(path) == str(tmp_path / name) and path.exists()


def test_fetch_unknown_name_raises_keyerror(tmp_path, monkeypatch):
    monkeypatch.setenv("HOD_MOD_DATA_DIR", str(tmp_path))   # mirror set but file absent
    with pytest.raises(KeyError):
        fetch("definitely_not_in_registry.bin")


def test_fetch_unpublished_doi_raises_runtimeerror(monkeypatch):
    """With the placeholder DOI, no local mirror and no base-url override, fetch
    must refuse to download rather than hit an unpublished record."""
    monkeypatch.delenv("HOD_MOD_DATA_DIR", raising=False)
    monkeypatch.delenv("HOD_MOD_DATA_BASEURL", raising=False)
    monkeypatch.setenv("HOD_MOD_DATA_DOI", PLACEHOLDER_DOI)
    with pytest.raises(RuntimeError):
        fetch("S1_mcmc_summary.json")
