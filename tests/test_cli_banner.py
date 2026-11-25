import json
import subprocess
import sys

from asof import TAGLINE, __version__


def test_cli_help_shows_ascii_banner_and_tagline_without_polluting_json():
    help_res = subprocess.run([sys.executable, "-m", "asof", "--help"], capture_output=True, text=True, check=True)
    assert "/___/" not in help_res.stdout or "/ ___/" in help_res.stdout
    assert "___" in help_res.stdout
    assert TAGLINE in help_res.stdout

    ver_res = subprocess.run([sys.executable, "-m", "asof", "--version"], capture_output=True, text=True, check=True)
    assert __version__ in ver_res.stdout
    assert TAGLINE in ver_res.stdout

    # Bare `asof` renders the Option A startup dashboard (banner + status + quick start, NOT raw Binance JSON).
    bare = subprocess.run([sys.executable, "-m", "asof"], capture_output=True, text=True, check=True)
    assert "Workspace Status" in bare.stdout
    assert "Quick Start" in bare.stdout
    assert "asof config init" in bare.stdout
    assert '"observations"' not in bare.stdout

    # Normal JSON commands must remain 100% pure JSON on stdout (never prefixed by ASCII art).
    replay = subprocess.run(
        [sys.executable, "-m", "asof", "replay", "--dataset", "nse", "--as-of", "2025-01-08T12:00:00Z"],
        capture_output=True, text=True, check=True,
    )
    parsed = json.loads(replay.stdout)
    assert parsed["dataset"]["instrument"] == "XNAI:DEMO-001"
