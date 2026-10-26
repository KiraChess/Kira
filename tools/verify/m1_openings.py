#!/usr/bin/env python3
"""Generate balanced random-play EPD openings using Kira-HCE only.

This script intentionally uses only the candidate's UCI legal-move divide,
board dump, and depth-one Kira-HCE score. No external chess package or engine
is used. The output EPD is compatible with Fastchess's EPD opening reader.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from m1 import START_FEN, UCIProcess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--max-attempts", type=int, default=20_000)
    parser.add_argument("--score-limit", type=int, default=150)
    parser.add_argument("--epd", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    args = parser.parse_args()
    if args.target < 1 or args.plies < 1 or args.max_attempts < args.target:
        parser.error("target and plies must be positive; max-attempts must be >= target")

    args.epd.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    epd_file = args.epd.open("x", encoding="utf-8")
    jsonl_file = args.jsonl.open("x", encoding="utf-8")

    def log(row: dict) -> None:
        jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")
        jsonl_file.flush()

    engine = None
    rng = random.Random(args.seed)
    seen: set[str] = set()
    accepted = attempts = rejected_score = rejected_duplicate = rejected_terminal = 0
    try:
        engine = UCIProcess(args.engine, "Kira-HCE openings")
        engine.setoption("Hash", 16)
        engine.setoption("Threads", 1)
        engine.ready()
        log({
            "kind": "opening_generation", "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "engine": str(Path(args.engine).resolve()), "seed": args.seed,
            "target": args.target, "plies": args.plies,
            "score_limit_cp": args.score_limit, "max_attempts": args.max_attempts,
            "python": sys.version.split()[0], "move_source": "candidate go perft 1",
            "score_source": "candidate go depth 1 HCE search",
        })

        while accepted < args.target and attempts < args.max_attempts:
            attempts += 1
            engine.new_game()
            fen = START_FEN
            moves: list[str] = []
            terminal = False
            for ply in range(args.plies):
                position = f"position fen {fen}"
                legal = engine.legal_moves(position, stage=f"attempt {attempts}, ply {ply + 1}")
                if not legal:
                    terminal = True
                    break
                move = rng.choice(legal)
                moves.append(move)
                fen = engine.fen(
                    f"position fen {fen} moves {move}",
                    stage=f"attempt {attempts}, ply {ply + 1} FEN update",
                )
            if terminal:
                rejected_terminal += 1
                log({"kind": "opening_attempt", "attempt": attempts, "status": "terminal_before_target_plies", "moves": moves})
                continue

            position = f"position fen {fen}"
            engine.send(position)
            result = engine.search("go depth 1", 15, stage=f"opening attempt {attempts} score")
            if not result.parsed_info:
                raise RuntimeError(f"attempt {attempts}: Kira returned no info line")
            score_line = result.parsed_info[-1]
            if score_line.get("score_type") != "cp":
                rejected_score += 1
                log({"kind": "opening_attempt", "attempt": attempts, "status": "mate_score", "moves": moves, "fen": fen})
                continue
            score = int(score_line["score"])
            key = " ".join(fen.split()[:4])
            if key in seen:
                rejected_duplicate += 1
                log({"kind": "opening_attempt", "attempt": attempts, "status": "duplicate", "score_cp": score, "moves": moves, "fen": fen})
                continue
            if abs(score) > args.score_limit:
                rejected_score += 1
                log({"kind": "opening_attempt", "attempt": attempts, "status": "outside_score_window", "score_cp": score, "moves": moves, "fen": fen})
                continue

            seen.add(key)
            epd_file.write(key + "\n")
            epd_file.flush()
            accepted += 1
            log({
                "kind": "opening_attempt", "attempt": attempts,
                "status": "accepted", "opening_id": accepted,
                "score_cp_stm": score, "moves": moves, "epd": key,
            })

        summary = {
            "kind": "opening_generation_summary", "seed": args.seed,
            "target": args.target, "accepted": accepted, "attempts": attempts,
            "rejected_score_or_mate": rejected_score,
            "rejected_duplicate": rejected_duplicate,
            "rejected_terminal": rejected_terminal,
            "complete": accepted == args.target,
        }
        log(summary)
        return 0 if accepted == args.target else 1
    except Exception as exc:
        log({"kind": "failure", "error": str(exc)})
        print(f"opening generator failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if engine is not None:
            engine.close()
        epd_file.close()
        jsonl_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
