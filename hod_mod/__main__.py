"""Enable ``python -m hod_mod <command>`` (delegates to the unified CLI)."""
from hod_mod.cli.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
