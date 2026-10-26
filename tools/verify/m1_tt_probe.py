#!/usr/bin/env python3
"""Black-box regressions for transposition scores across search contexts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from m1 import UCIProcess


def depth_score(result) -> int:
    scores = [row["score"] for row in result.parsed_info if row.get("depth") == 2 and row.get("score_type") == "cp"]
    if not scores:
        raise RuntimeError(f"no depth-2 centipawn score in UCI output: {result.output_lines}")
    return int(scores[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)

    near_fifty = "4k3/8/8/8/8/8/8/3QK3 w - - 98 1"
    clock_zero = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    p_fen = "k1r3n1/1nB5/1BB5/8/8/3q4/8/1N4NK w - - 0 1"
    p_moves = "g1f3 g8f6 b1c3 f6g8 c3b1 g8f6 f3g1 f6g8"
    q_fen = "k1r3n1/2B5/1BB5/2n5/8/3q4/8/1N4NK b - - 7 4"
    rows: list[dict] = [{
        "kind": "run", "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": str(Path(args.engine).resolve()),
        "tt_context": "halfmove clock and repetition-history signature included in TT key",
    }]

    same = UCIProcess(args.engine, "same-process TT context probe")
    try:
        scores = []
        for fen in (near_fifty, clock_zero):
            same.send("position fen " + fen)
            result = same.search("go depth 2", 15, stage="same-process TT context regression")
            score = depth_score(result)
            scores.append(score)
            rows.append({"kind": "same_process", "fen": fen, "score_cp": score, "bestmove": result.move})
    finally:
        same.close()

    fresh = UCIProcess(args.engine, "fresh-process TT context probe")
    try:
        fresh.send("position fen " + clock_zero)
        result = fresh.search("go depth 2", 15, stage="fresh-process TT context regression")
        fresh_score = depth_score(result)
        rows.append({"kind": "fresh_process", "fen": clock_zero, "score_cp": fresh_score, "bestmove": result.move})
    finally:
        fresh.close()

    same_history = UCIProcess(args.engine, "same-process repetition-history probe")
    try:
        same_history.send(f"position fen {p_fen} moves {p_moves}")
        prime = same_history.search("go depth 3", 20, stage="prime repetition-dependent TT score")
        prime_score = depth_score_at(prime, 3)
        same_history.send("position fen " + q_fen)
        reused_q = same_history.search("go depth 4", 20, stage="repetition-context TT reuse regression")
        reused_q_score = depth_score_at(reused_q, 4)
        rows.append({
            "kind": "repetition_prime",
            "fen": p_fen,
            "moves": p_moves.split(),
            "score_cp": prime_score,
            "bestmove": prime.move,
        })
        rows.append({"kind": "repetition_same_process", "fen": q_fen, "score_cp": reused_q_score, "bestmove": reused_q.move})
    finally:
        same_history.close()

    fresh_q = UCIProcess(args.engine, "fresh-process repetition-history probe")
    try:
        fresh_q.send("position fen " + q_fen)
        result_q = fresh_q.search("go depth 4", 20, stage="fresh repetition-context TT score")
        fresh_q_score = depth_score_at(result_q, 4)
        rows.append({"kind": "repetition_fresh_process", "fen": q_fen, "score_cp": fresh_q_score, "bestmove": result_q.move})
    finally:
        fresh_q.close()

    clock_passed = scores[1] == fresh_score and scores[0] != scores[1] and scores[1] > 800
    repetition_passed = prime_score == 0 and reused_q_score == fresh_q_score and reused_q_score == 670
    passed = clock_passed and repetition_passed
    rows.append({
        "kind": "summary", "clock98_score_cp": scores[0],
        "same_process_clock0_score_cp": scores[1], "fresh_clock0_score_cp": fresh_score,
        "clock_context_passed": clock_passed,
        "repetition_prime_score_cp": prime_score,
        "same_process_repetition_q_score_cp": reused_q_score,
        "fresh_repetition_q_score_cp": fresh_q_score,
        "repetition_context_passed": repetition_passed,
        "passed": passed,
    })
    with args.artifact.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    return 0 if passed else 1


def depth_score_at(result, depth: int) -> int:
    scores = [row["score"] for row in result.parsed_info if row.get("depth") == depth and row.get("score_type") == "cp"]
    if not scores:
        raise RuntimeError(f"no depth-{depth} centipawn score in UCI output: {result.output_lines}")
    return int(scores[-1])


if __name__ == "__main__":
    raise SystemExit(main())
