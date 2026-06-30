"""Tests for the unified ``hod-mod`` command-line interface (hod_mod.cli)."""
import importlib.util
import sys

import pytest

from hod_mod.cli.__main__ import main, COMMANDS, VALIDATE


def test_no_args_prints_help_returns_0(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "hod-mod" in out and "commands:" in out


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_flag_returns_0(flag, capsys):
    assert main([flag]) == 0
    assert "fit" in capsys.readouterr().out


def test_unknown_command_returns_2(capsys):
    assert main(["definitely-not-a-command"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_validate_no_target_lists_targets_returns_0(capsys):
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    for target in VALIDATE:
        assert target in out


def test_validate_unknown_target_returns_2(capsys):
    assert main(["validate", "nope"]) == 2
    assert "unknown validate target" in capsys.readouterr().err


@pytest.mark.parametrize("module", sorted(set(COMMANDS.values()) | set(VALIDATE.values())))
def test_all_target_modules_are_importable(module):
    """Every CLI subcommand must point at a real, importable module."""
    assert importlib.util.find_spec(module) is not None, module


def test_fit_delegates_to_run_fit(monkeypatch):
    calls = {}
    monkeypatch.setattr("hod_mod.cli.__main__.runpy.run_module",
                        lambda mod, run_name: calls.update(mod=mod, argv=list(sys.argv),
                                                            run_name=run_name))
    assert main(["fit", "cfg.yml", "--map-only"]) == 0
    assert calls["mod"] == "hod_mod.scripts.fitting.run_fit"
    assert calls["run_name"] == "__main__"
    assert calls["argv"] == ["hod-mod fit", "cfg.yml", "--map-only"]


def test_validate_delegates_with_forwarded_args(monkeypatch):
    calls = {}
    monkeypatch.setattr("hod_mod.cli.__main__.runpy.run_module",
                        lambda mod, run_name: calls.update(mod=mod, argv=list(sys.argv)))
    assert main(["validate", "sz-xray", "--out", "x.png"]) == 0
    assert calls["mod"] == "hod_mod.scripts.validate_sz_xray"
    assert calls["argv"] == ["hod-mod validate sz-xray", "--out", "x.png"]
