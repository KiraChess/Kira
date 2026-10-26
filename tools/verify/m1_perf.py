#!/usr/bin/env python3
"""Measure a candidate's UCI node rate over repeated fixed-time searches."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from m1 import UCIProcess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--movetime-ms", type=int, default=1500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.movetime_ms <= 0 or args.repeats <= 0:
        parser.error("movetime and repeats must be positive")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = [{
        "kind": "run",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": str(Path(args.engine).resolve()),
        "command": f"position startpos; go movetime {args.movetime_ms}",
        "hash_mb": 16,
        "threads": 1,
        "repeats": args.repeats,
    }]
    nps_samples: list[int] = []
    for trial in range(1, args.repeats + 1):
        engine = UCIProcess(args.engine, f"M1 performance trial {trial}")
        try:
            engine.setoption("Hash", 16)
            engine.setoption("Threads", 1)
            engine.new_game()
            engine.send("position startpos")
            result = engine.search(f"go movetime {args.movetime_ms}", max(10.0, args.movetime_ms / 1000 + 10), stage="M1 node-rate measurement")
            final_info = result.parsed_info[-1] if result.parsed_info else {}
            nodes = int(final_info.get("nodes", 0))
            reported_nps = int(final_info.get("nps", 0))
            wall_nps = round(nodes / result.elapsed_seconds) if result.elapsed_seconds else 0
            if nodes <= 0 or reported_nps <= 0:
                raise RuntimeError(f"trial {trial} did not report nodes and nps: {result.output_lines}")
            nps_samples.append(reported_nps)
            rows.append({
                "kind": "trial",
                "trial": trial,
                "depth": final_info.get("depth"),
                "nodes": nodes,
                "uci_nps": reported_nps,
                "wall_nps": wall_nps,
                "elapsed_seconds": round(result.elapsed_seconds, 4),
                "bestmove": result.move,
            })
        finally:
            engine.close()
    rows.append({
        "kind": "summary",
        "median_uci_nps": round(statistics.median(nps_samples)),
        "min_uci_nps": min(nps_samples),
        "max_uci_nps": max(nps_samples),
    })
    with args.artifact.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(rows[-1], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
