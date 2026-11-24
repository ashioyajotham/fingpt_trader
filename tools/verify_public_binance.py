"""Verify the public Binance REST kline fixture without account credentials or third-party SDKs."""
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fingpt_research.data import FIXTURES


def verify():
    root = FIXTURES / "binance"
    url = json.loads((root / "manifest.json").read_text())["source"]
    with urlopen(url, timeout=30) as response:
        data = response.read()
    if json.loads(data) != json.loads((root / "klines.json").read_text()):
        raise ValueError("historical public response differs from the frozen fixture")
    print(json.dumps({
        "venue": "Binance spot",
        "rest_candles": len(json.loads(data)),
        "sha256": hashlib.sha256(data).hexdigest(),
    }), flush=True)


if __name__ == "__main__":
    verify()
