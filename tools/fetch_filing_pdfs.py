"""Retrieve the originals listed in a frozen filing manifest after verifying their hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def fetch(row, root):
    name = row["pdf"]
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("expected a local PDF filename")
    target = root / name
    if target.exists():
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["pdf_sha256"]:
            raise ValueError(f"existing file differs from manifest: {name}")
        return
    if urlsplit(row["source"]).scheme != "https":
        raise ValueError("expected a public HTTPS issuer source")
    req = Request(row["source"], headers={"User-Agent": "FinGPT-Research/0.2"})
    with urlopen(req, timeout=45) as response, NamedTemporaryFile(dir=root, delete=False) as stream:
        temp = Path(stream.name)
        try:
            digest = hashlib.sha256()
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
            if digest.hexdigest() != row["pdf_sha256"]:
                raise ValueError(f"download differs from frozen manifest: {name}")
            stream.close()
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for row in json.loads(args.manifest.read_text())["filings"]:
        fetch(row, args.manifest.parent)
