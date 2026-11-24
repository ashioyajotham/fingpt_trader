"""Run rules and local FinGPT on the same frozen candidates for each verified PDF."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fingpt_research.evidence import LocalFinGPT
from fingpt_research.filings import extract_filing, prepare_filing


def compare_filings(manifest_path, out, gguf):
    rows = json.loads(manifest_path.read_text())["filings"]
    ids = [row["id"] for row in rows]
    if not ids or len(ids) != len(set(ids)) or any(
            not key or Path(key).name != key or key in {".", ".."} for key in ids):
        raise ValueError("expected distinct filing IDs suitable for report filenames")
    if out.exists():
        raise ValueError(f"choose a new run directory; {out} already exists")
    filings = []
    for row in rows:
        prepared = prepare_filing(manifest_path.parent / row["pdf"], row["symbol"], row["source"])
        if (prepared["pdf_sha256"] != row["pdf_sha256"] or
                prepared["coverage"]["page_count"] != row["page_count"]):
            raise ValueError(f"frozen PDF mismatch for {row['id']}")
        filings.append(prepared)
    methods = {"rules": None, "local": LocalFinGPT(gguf, structured=True)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as scratch:
        run = Path(scratch) / "reports"
        for name, model in methods.items():
            dest = run / name
            dest.mkdir(parents=True)
            for key, prepared in zip(ids, filings):
                result = extract_filing(prepared, model)
                (dest / (key + ".json")).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        # Publish only complete, paired runs. A second run cannot replace a populated output directory.
        run.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    args = parser.parse_args()
    compare_filings(args.manifest, args.out, args.gguf)
