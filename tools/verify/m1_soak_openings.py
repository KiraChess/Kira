#!/usr/bin/env python3
"""Select the first 250 lines from the seeded 500-line HCE opening set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    lines = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 500 or len(set(lines)) != 500:
        raise SystemExit(f"expected 500 unique opening lines, found {len(lines)} ({len(set(lines))} unique)")
    selected = lines[:250]
    if len(set(selected)) != 250:
        raise SystemExit("selected opening subset contains duplicates")
    with args.output.open("x", encoding="utf-8") as output:
        output.write("\n".join(selected) + "\n")
    metadata = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "selection": "first 250 lines of the seeded 500-position random-play set",
        "input_lines": len(lines),
        "selected_lines": len(selected),
        "unique_selected": len(set(selected)),
        "games_with_repeat": 500,
        "color_reversal": "each selected opening played twice with colors swapped",
    }
    with args.metadata.open("x", encoding="utf-8") as output:
        json.dump(metadata, output, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
