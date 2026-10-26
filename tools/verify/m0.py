#!/usr/bin/env python3
"""Independent M0 perft and differential-fuzz driver for a UCI chess engine.

Uses only Python's standard library. The candidate supplies legal random moves
and resulting FENs for fuzz walks; Stockfish is used only for perft counts. This
script does not import or inspect engine code.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


KNOWN = [
    {
        "name": "startpos",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "depths": {5: 4_865_609, 6: 119_060_324},
    },
    {
        "name": "kiwipete",
        "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "depths": {4: 4_085_603, 5: 193_690_690},
    },
    {
        "name": "position-3",
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "depths": {5: 674_624, 6: 11_030_083},
    },
    {
        "name": "position-4",
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "depths": {4: 422_333, 5: 15_833_292},
    },
    {
        "name": "position-5",
        "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "depths": {4: 2_103_487, 5: 89_941_194},
    },
    {
        "name": "position-6",
        "fen": "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "depths": {4: 3_894_594, 5: 164_075_551},
    },
]

NODE_RE = re.compile(r"^(?:Nodes searched:|info string perft depth \d+ nodes)\s*(\d+)\s*$")
DIVIDE_RE = re.compile(r"^info string perft divide ([a-h][1-8][a-h][1-8][qrbn]?)\s+(\d+)\s*$")


class UCI:
    def __init__(self, command: str, label: str):
        self.command = command
        self.label = label
        self.proc = subprocess.Popen(
            [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        self.send("uci")
        self.until(lambda s: s == "uciok", 20)
        self.send("isready")
        self.until(lambda s: s == "readyok", 20)

    def _read_output(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def send(self, command: str) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError(f"{self.label} exited with {self.proc.returncode}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def until(self, predicate, timeout: float = 30) -> list[str]:
        deadline = time.monotonic() + timeout
        found: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                raise RuntimeError(f"{self.label} closed stdout unexpectedly")
            found.append(line)
            if predicate(line):
                return found
        tail = "\n".join(found[-12:])
        raise TimeoutError(f"Timed out waiting for {self.label}; recent output:\n{tail}")

    def perft(self, fen: str, depth: int, timeout: float = 600) -> tuple[int, list[str]]:
        self.send(f"position fen {fen}")
        self.send(f"go perft {depth}")
        # UCI's go perft extension has no single standard completion marker.
        # Kira prints a conventional `Nodes searched:` total and then an
        # additional `info string perft depth ... nodes ...` total.  Returning
        # on the first marker leaves the latter queued, where it can be
        # mistaken for the next perft command's result.  Wait for one total,
        # then use isready as a command-order fence to drain all output already
        # produced by the synchronous perft command before returning.
        out = self.until(lambda s: NODE_RE.match(s) is not None, timeout)
        self.send("isready")
        out.extend(self.until(lambda s: s == "readyok", timeout))
        totals = [int(match.group(1)) for line in out if (match := NODE_RE.match(line))]
        if not totals:
            raise AssertionError("perft terminator not parsed")
        if len(set(totals)) != 1:
            raise RuntimeError(f"{self.label} emitted conflicting perft totals at depth {depth}: {totals}")
        return totals[-1], out

    def divide_moves(self, fen: str) -> list[str]:
        _, out = self.perft(fen, 1, 30)
        return [m.group(1) for line in out if (m := DIVIDE_RE.match(line))]

    def fen_after(self, fen: str, move: str) -> str:
        self.send(f"position fen {fen} moves {move}")
        self.send("d")
        out = self.until(lambda s: s.startswith("Fen:"), 30)
        for line in out:
            if line.startswith("Fen:"):
                return line.partition(":")[2].strip()
        raise RuntimeError(f"{self.label} did not provide a FEN after {move}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=5)


def run_known(stockfish: UCI, candidate: UCI, output: list[dict]) -> bool:
    all_ok = True
    for case in KNOWN:
        for depth, expected in case["depths"].items():
            sf_nodes, _ = stockfish.perft(case["fen"], depth)
            row = {
                "kind": "known", "name": case["name"], "depth": depth,
                "expected": expected, "stockfish": sf_nodes,
            }
            if sf_nodes != expected:
                row["status"] = "oracle_mismatch"
                all_ok = False
            else:
                engine_nodes, _ = candidate.perft(case["fen"], depth)
                row["engine"] = engine_nodes
                row["status"] = "pass" if engine_nodes == expected else "mismatch"
                all_ok &= engine_nodes == expected
            output.append(row)
            print(json.dumps(row), flush=True)
    return all_ok


def run_fuzz(
    stockfish: UCI, candidate: UCI, positions: int, depth4_count: int,
    seed: int, output: list[dict],
) -> bool:
    rng = random.Random(seed)
    seen: set[str] = set()
    fen = KNOWN[0]["fen"]
    plies = 0
    position_index = 0
    mismatches = 0
    while position_index < positions:
        # Reset after terminal positions and long walks to keep the sample broad.
        moves = candidate.divide_moves(fen)
        if not moves or plies >= 200:
            fen = KNOWN[0]["fen"]
            plies = 0
            moves = candidate.divide_moves(fen)
        move = rng.choice(moves)
        fen = candidate.fen_after(fen, move)
        plies += 1
        key = " ".join(fen.split()[:4])
        if key in seen:
            continue
        seen.add(key)

        sf3, _ = stockfish.perft(fen, 3)
        engine3, _ = candidate.perft(fen, 3)
        row = {
            "kind": "fuzz", "index": position_index, "seed": seed,
            "fen": fen, "depth": 3, "stockfish": sf3, "engine": engine3,
        }
        ok = sf3 == engine3
        if position_index < depth4_count:
            sf4, _ = stockfish.perft(fen, 4)
            engine4, _ = candidate.perft(fen, 4)
            row.update({"depth4_stockfish": sf4, "depth4_engine": engine4})
            ok &= sf4 == engine4
        row["status"] = "pass" if ok else "mismatch"
        if not ok:
            mismatches += 1
        output.append(row)
        if position_index % 100 == 0 or not ok:
            print(json.dumps(row), flush=True)
        position_index += 1
    summary = {
        "kind": "fuzz_summary", "seed": seed, "positions": positions,
        "depth4_positions": min(depth4_count, positions),
        "unique_positions": len(seen), "mismatches": mismatches,
        "status": "pass" if mismatches == 0 else "mismatch",
    }
    output.append(summary)
    print(json.dumps(summary), flush=True)
    return mismatches == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="path to candidate UCI executable")
    parser.add_argument("--stockfish", default="stockfish", help="path to Stockfish executable")
    parser.add_argument("--known-only", action="store_true", help="run the exact known perft suite")
    parser.add_argument("--fuzz", action="store_true", help="run random legal-play differential fuzzing")
    parser.add_argument("--positions", type=int, default=100_000)
    parser.add_argument("--depth4-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--artifact", type=Path, help="write JSONL evidence to this path")
    args = parser.parse_args()

    run_fuzz_requested = args.fuzz
    run_known_requested = args.known_only or not run_fuzz_requested
    if args.positions < 0 or args.depth4_count < 0 or args.depth4_count > args.positions:
        parser.error("require 0 <= --depth4-count <= --positions")

    rows: list[dict] = []
    stockfish = candidate = None
    try:
        stockfish = UCI(args.stockfish, "Stockfish")
        print(f"Stockfish command: {args.stockfish}", flush=True)
        candidate = UCI(args.engine, "candidate")
        passed = True
        if run_known_requested:
            passed &= run_known(stockfish, candidate, rows)
        if run_fuzz_requested:
            passed &= run_fuzz(stockfish, candidate, args.positions, args.depth4_count, args.seed, rows)
        if args.artifact:
            args.artifact.parent.mkdir(parents=True, exist_ok=True)
            with args.artifact.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        print("M0 RESULT: " + ("PASS" if passed else "FAIL"), flush=True)
        return 0 if passed else 1
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"M0 HARNESS ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if candidate is not None:
            candidate.close()
        if stockfish is not None:
            stockfish.close()


if __name__ == "__main__":
    raise SystemExit(main())
