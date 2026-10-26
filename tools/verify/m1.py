#!/usr/bin/env python3
"""Black-box M1 UCI checks and deterministic games against a random mover.

Only Python's standard library is used. Stockfish checks protocol-probe moves
as a perft oracle. During random-mover games Kira supplies its own legal-move
lists and FEN diagnostics; candidate engine source is never read. The harness
sends the complete move history in each standard UCI `position` command so
the engine can apply repetition rules correctly.
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
from dataclasses import dataclass
from pathlib import Path


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_AFTER_E4_E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
MOVE_LINE = re.compile(r"^([a-h][1-8][a-h][1-8][qrbn]?):\s*(\d+)\s*$")
BESTMOVE_LINE = re.compile(r"^bestmove\s+(\S+)(?:\s+ponder\s+\S+)?\s*$")
FEN_LINE = re.compile(r"^Fen:\s*(.+?)\s*$")
INTEGER_INFO = ("depth", "seldepth", "nodes", "nps", "time", "multipv")


class HarnessError(RuntimeError):
    pass


class UCIProcess:
    """A line-oriented UCI process with explicit timeouts."""

    def __init__(self, executable: str, label: str, handshake_timeout: float = 20):
        self.executable = executable
        self.label = label
        self.proc = subprocess.Popen(
            [executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.output: queue.Queue[str | None] = queue.Queue()
        self.last_fen: str | None = None
        self.last_checkers: str | None = None
        self.reader = threading.Thread(target=self._read_lines, daemon=True)
        self.reader.start()
        self.send("uci")
        self.handshake_output = self.read_until(lambda s: s == "uciok", handshake_timeout)
        self.options = self._parse_options(self.handshake_output)
        self.send("isready")
        self.read_until(lambda s: s == "readyok", handshake_timeout)

    def _read_lines(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.output.put(line.rstrip("\r\n"))
        self.output.put(None)

    @staticmethod
    def _parse_options(lines: list[str]) -> dict[str, str]:
        options: dict[str, str] = {}
        for line in lines:
            if line.startswith("option name "):
                rest = line[len("option name "):]
                name, marker, _ = rest.partition(" type ")
                if marker:
                    options[name.casefold()] = name
        return options

    def send(self, command: str) -> None:
        if self.proc.poll() is not None:
            raise HarnessError(f"{self.label} exited with status {self.proc.returncode}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_one(self, timeout: float) -> str:
        try:
            line = self.output.get(timeout=max(0.001, timeout))
        except queue.Empty as exc:
            raise TimeoutError(f"{self.label} did not respond within {timeout:.2f}s") from exc
        if line is None:
            raise HarnessError(f"{self.label} closed stdout unexpectedly")
        return line

    def read_until(self, predicate, timeout: float) -> list[str]:
        deadline = time.monotonic() + timeout
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for {self.label}; recent output: {lines[-10:]}"
                )
            line = self.read_one(remaining)
            lines.append(line)
            if predicate(line):
                return lines

    def ready(self, timeout: float = 10) -> None:
        self.send("isready")
        self.read_until(lambda s: s == "readyok", timeout)

    def setoption(self, name: str, value: int | str) -> None:
        if name.casefold() not in self.options:
            raise HarnessError(f"{self.label} did not advertise UCI option {name!r}")
        self.send(f"setoption name {self.options[name.casefold()]} value {value}")

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.ready()

    def legal_moves(
        self, position: str, timeout: float = 30, stage: str | None = None,
    ) -> list[str]:
        """Read a perft-1 divide list; supports standard coordinate UCI moves."""
        self.send(position)
        self.send("go perft 1")
        try:
            lines = self.read_until(lambda s: s.startswith("Nodes searched:"), timeout)
        except TimeoutError as exc:
            detail = f" during {stage}" if stage else ""
            raise TimeoutError(f"{self.label} timed out awaiting `go perft 1` result{detail}") from exc
        return [m.group(1) for line in lines if (m := MOVE_LINE.match(line))]

    def fen(
        self, position: str, timeout: float = 30, stage: str | None = None,
    ) -> str:
        self.send(position)
        self.send("d")
        try:
            lines = self.read_until(lambda s: s.startswith("Checkers:"), timeout)
        except TimeoutError as exc:
            detail = f" during {stage}" if stage else ""
            raise TimeoutError(f"{self.label} timed out awaiting `d` board dump{detail}") from exc
        checker_lines = [line for line in lines if line.startswith("Checkers:")]
        if not checker_lines:
            raise HarnessError(f"{self.label} `d` output did not include Checkers:")
        self.last_checkers = checker_lines[-1].partition(":")[2].strip()
        for line in lines:
            match = FEN_LINE.match(line)
            if match:
                self.last_fen = match.group(1)
                return self.last_fen
        raise HarnessError(f"{self.label} `d` output did not include a FEN line")

    def is_in_check(
        self, position: str, timeout: float = 30, stage: str | None = None,
    ) -> bool:
        if self.last_fen is not None and position == f"position fen {self.last_fen}":
            return bool(self.last_checkers)
        self.send(position)
        self.send("d")
        try:
            lines = self.read_until(lambda s: s.startswith("Checkers:"), timeout)
        except TimeoutError as exc:
            detail = f" during {stage}" if stage else ""
            raise TimeoutError(f"{self.label} timed out awaiting `d` check status{detail}") from exc
        checker_lines = [line for line in lines if line.startswith("Checkers:")]
        if not checker_lines:
            raise HarnessError(f"{self.label} `d` output did not include Checkers:")
        self.last_checkers = checker_lines[-1].partition(":")[2].strip()
        return bool(self.last_checkers)

    def search(
        self, go_command: str, timeout: float, stage: str | None = None,
    ) -> "SearchResult":
        start = time.monotonic()
        self.send(go_command)
        try:
            lines = self.read_until(lambda s: BESTMOVE_LINE.match(s) is not None, timeout)
        except TimeoutError as exc:
            detail = f" during {stage}" if stage else ""
            raise TimeoutError(f"{self.label} timed out waiting for {go_command!r}{detail}") from exc
        match = next(BESTMOVE_LINE.match(line) for line in reversed(lines) if BESTMOVE_LINE.match(line))
        return SearchResult(
            move=match.group(1).lower(),
            elapsed_seconds=time.monotonic() - start,
            info_lines=[line for line in lines if line.startswith("info ")],
            parsed_info=parse_info(lines),
            output_lines=lines,
        )

    def infinite_then_stop(self, timeout: float, stop_delay: float) -> "SearchResult":
        start = time.monotonic()
        self.send("go infinite")
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        # Wait until the engine demonstrates that it started searching.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.label} produced no info for go infinite")
            line = self.read_one(remaining)
            lines.append(line)
            if BESTMOVE_LINE.match(line):
                raise HarnessError(f"{self.label} ended go infinite before stop")
            if line.startswith("info "):
                break
        if stop_delay > 0:
            time.sleep(stop_delay)
        self.send("stop")
        remaining = max(0.001, deadline - time.monotonic())
        lines.extend(self.read_until(lambda s: BESTMOVE_LINE.match(s) is not None, remaining))
        match = next(BESTMOVE_LINE.match(line) for line in reversed(lines) if BESTMOVE_LINE.match(line))
        return SearchResult(
            move=match.group(1).lower(),
            elapsed_seconds=time.monotonic() - start,
            info_lines=[line for line in lines if line.startswith("info ")],
            parsed_info=parse_info(lines),
            output_lines=lines,
        )

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=5)


@dataclass
class SearchResult:
    move: str
    elapsed_seconds: float
    info_lines: list[str]
    parsed_info: list[dict]
    output_lines: list[str]


def parse_info(lines: list[str]) -> list[dict]:
    """Extract the common numeric UCI info fields without assuming field order."""
    parsed: list[dict] = []
    for line in lines:
        if not line.startswith("info "):
            continue
        record: dict[str, int | str] = {"raw": line}
        tokens = line.split()
        for key in INTEGER_INFO:
            try:
                index = tokens.index(key)
                record[key] = int(tokens[index + 1])
            except (ValueError, IndexError):
                pass
        if "score" in tokens:
            try:
                idx = tokens.index("score")
                record["score_type"] = tokens[idx + 1]
                record["score"] = int(tokens[idx + 2])
            except (ValueError, IndexError):
                pass
        if "pv" in tokens:
            idx = tokens.index("pv")
            record["pv"] = tokens[idx + 1:]
        if any(key in record for key in ("depth", "nodes", "pv")):
            parsed.append(record)
    return parsed


def position_command(root: str, moves: list[str]) -> str:
    suffix = " moves " + " ".join(moves) if moves else ""
    return root + suffix


class StockfishOracle:
    """Use Stockfish's perft divide and debug board dump as black-box oracles."""

    def __init__(self, executable: str):
        self.uci = UCIProcess(executable, "Stockfish")

    def legal_moves(self, position: str) -> list[str]:
        return self.uci.legal_moves(position)

    def fen(self, position: str) -> str:
        return self.uci.fen(position)

    def is_in_check(self, position: str) -> bool:
        return self.uci.is_in_check(position)

    def close(self) -> None:
        self.uci.close()


class JsonlWriter:
    def __init__(self, path: Path | None):
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("x", encoding="utf-8")

    def write(self, row: dict, echo: bool = True) -> None:
        line = json.dumps(row, sort_keys=True)
        if echo:
            print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def require_parseable_info(result: SearchResult, label: str) -> None:
    if not result.parsed_info:
        raise HarnessError(f"{label} returned no parseable info line")


def check_bestmove(
    oracle: StockfishOracle, position: str, result: SearchResult, label: str,
) -> None:
    legal = oracle.legal_moves(position)
    if result.move not in legal:
        raise HarnessError(
            f"{label} returned illegal bestmove {result.move!r}; legal moves: {legal}"
        )


def protocol_probe(
    engine: UCIProcess, oracle: StockfishOracle, search_timeout: float,
    stop_delay: float, writer: JsonlWriter,
) -> list[dict]:
    required = ("hash", "threads")
    missing = [name for name in required if name not in engine.options]
    if missing:
        raise HarnessError(f"candidate did not advertise required options: {', '.join(missing)}")

    engine.setoption("Hash", 16)
    engine.setoption("Threads", 1)
    engine.ready()
    engine.new_game()

    probes: list[dict] = []

    def run_probe(name: str, position: str, go: str, timeout: float) -> None:
        engine.send(position)
        result = engine.search(go, timeout)
        require_parseable_info(result, name)
        check_bestmove(oracle, position, result, name)
        record = {
            "kind": "protocol_probe", "name": name, "go": go,
            "bestmove": result.move,
            "elapsed_seconds": round(result.elapsed_seconds, 4),
            "info_line_count": len(result.info_lines),
            "parsed_info": result.parsed_info,
            "status": "pass",
        }
        probes.append(record)
        writer.write(record)

    startpos = "position startpos"
    run_probe("startpos-depth", startpos, "go depth 2", search_timeout)

    fen_moves = "position fen " + FEN_AFTER_E4_E5 + " moves g1f3 b8c6"
    run_probe("fen-with-moves-nodes", fen_moves, "go nodes 1000", search_timeout)

    run_probe("movetime", startpos, "go movetime 100", max(search_timeout, 5))
    run_probe(
        "clock-fields", startpos,
        "go wtime 5000 btime 5000 winc 50 binc 50", max(search_timeout, 10),
    )

    # An explicit stop is needed because `go infinite` has no natural end.
    engine.send(startpos)
    result = engine.infinite_then_stop(max(search_timeout, 10), stop_delay)
    require_parseable_info(result, "infinite-stop")
    check_bestmove(oracle, startpos, result, "infinite-stop")
    record = {
        "kind": "protocol_probe", "name": "infinite-stop", "go": "go infinite; stop",
        "bestmove": result.move,
        "elapsed_seconds": round(result.elapsed_seconds, 4),
        "info_line_count": len(result.info_lines),
        "parsed_info": result.parsed_info,
        "status": "pass",
    }
    probes.append(record)
    writer.write(record)

    engine.ready()
    return probes


def insufficient_material(fen: str) -> bool:
    """Conservative automatic dead-position check for simple minor-only endings."""
    board = fen.split()[0]
    pieces = [c for c in board if c.isalpha() and c.lower() != "k"]
    if any(piece.lower() in ("p", "r", "q") for piece in pieces):
        return False
    if not pieces:
        return True
    if len(pieces) == 1 and pieces[0].lower() in ("b", "n"):
        return True
    if all(piece.lower() == "b" for piece in pieces):
        # Bishops confined to same-colour squares cannot produce mate.
        square_colours: list[int] = []
        rank = 7
        file_index = 0
        for char in board:
            if char == "/":
                rank -= 1
                file_index = 0
            elif char.isdigit():
                file_index += int(char)
            elif char.lower() == "b":
                square_colours.append((file_index + rank) & 1)
                file_index += 1
            else:
                file_index += 1
        return bool(square_colours) and len(set(square_colours)) == 1
    return False


@dataclass
class GameResult:
    outcome: str
    reason: str
    plies: int
    moves: list[str]


def play_random_game(
    engine: UCIProcess, rng: random.Random, writer: JsonlWriter,
    candidate_white: bool, depth: int, search_timeout: float, max_plies: int,
    game_id: int,
) -> GameResult:
    current_fen = START_FEN
    moves: list[str] = []
    repetition: dict[str, int] = {" ".join(current_fen.split()[:4]): 1}
    engine.new_game()

    def candidate_won(winner_white: bool) -> bool:
        return winner_white == candidate_white

    while len(moves) < max_plies:
        # Send the game history so search has the positions needed for
        # threefold detection. This is the standard UCI client contract.
        position = "position startpos" + (f" moves {' '.join(moves)}" if moves else "")
        stage = f"game {game_id}, ply {len(moves) + 1}"
        # M1's random opponent is supplied by Kira's own perft divide. Stockfish
        # is intentionally not consulted during this 100-game evaluation.
        legal = engine.legal_moves(position, stage=stage)
        if not legal:
            if engine.is_in_check(position, stage=stage):
                side_to_move_white = current_fen.split()[1] == "w"
                winner_white = not side_to_move_white
                outcome = "win" if candidate_won(winner_white) else "loss"
                return GameResult(outcome, "checkmate", len(moves), moves)
            return GameResult("draw", "stalemate", len(moves), moves)

        side_to_move_white = len(moves) % 2 == 0
        is_candidate_turn = side_to_move_white == candidate_white
        if is_candidate_turn:
            engine.send(position)
            result = engine.search(f"go depth {depth}", search_timeout, stage=stage)
            require_parseable_info(result, stage)
            move = result.move
            if move not in legal:
                raise HarnessError(
                    f"game {game_id}, ply {len(moves) + 1}: illegal bestmove {move!r}; "
                    f"legal moves are {legal}; history: {' '.join(moves)}"
                )
        else:
            # random.choice gives every legal UCI move equal probability.
            move = rng.choice(legal)

        moves.append(move)
        update_position = "position startpos moves " + " ".join(moves)
        current_fen = engine.fen(update_position, stage=stage + " FEN update")
        fields = current_fen.split()
        writer.write({
            "kind": "random_ply", "game": game_id, "ply": len(moves),
            "candidate_color": "white" if candidate_white else "black",
            "move": move, "fen": current_fen,
        }, echo=(len(moves) == 1 or len(moves) % 10 == 0))
        if len(fields) >= 6:
            state_key = " ".join(fields[:4])
            repetition[state_key] = repetition.get(state_key, 0) + 1
            if repetition[state_key] >= 3:
                return GameResult("draw", "threefold-repetition", len(moves), moves)
            try:
                if int(fields[4]) >= 100:
                    return GameResult("draw", "fifty-move-rule", len(moves), moves)
            except ValueError:
                pass
        if insufficient_material(current_fen):
            return GameResult("draw", "insufficient-material", len(moves), moves)

    return GameResult("draw", "max-plies", len(moves), moves)


def random_evaluation(
    engine: UCIProcess, writer: JsonlWriter,
    games: int, seed: int, depth: int, search_timeout: float, max_plies: int,
) -> dict:
    rng = random.Random(seed)
    wins = draws = losses = 0
    total_plies = 0
    for game_id in range(1, games + 1):
        candidate_white = game_id % 2 == 1
        result = play_random_game(
            engine, rng, writer, candidate_white, depth, search_timeout,
            max_plies, game_id,
        )
        wins += result.outcome == "win"
        draws += result.outcome == "draw"
        losses += result.outcome == "loss"
        total_plies += result.plies
        row = {
            "kind": "random_game", "game": game_id,
            "seed": seed, "candidate_color": "white" if candidate_white else "black",
            "depth": depth, "outcome": result.outcome, "reason": result.reason,
            "plies": result.plies, "moves": " ".join(result.moves),
        }
        writer.write(row)

    passed_gate = games == 100 and wins == 100 and draws == 0 and losses == 0
    summary = {
        "kind": "random_evaluation_summary", "seed": seed, "games": games,
        "depth": depth, "wins": wins, "draws": draws, "losses": losses,
        "mean_plies": round(total_plies / games, 2) if games else None,
        "passed_m1_random_gate": passed_gate,
        "status": "pass" if passed_gate else "incomplete_or_failed",
    }
    writer.write(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="path to candidate UCI executable")
    parser.add_argument("--stockfish", default="stockfish", help="path to Stockfish executable")
    parser.add_argument("--protocol-only", action="store_true", help="skip the 100-game random evaluation")
    parser.add_argument("--random-games", type=int, default=100, help="number of random-mover games (M1 gate is exactly 100)")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--random-depth", type=int, default=4, help="fixed candidate search depth in random games")
    parser.add_argument("--search-timeout", type=float, default=15.0, help="maximum seconds for each candidate search")
    parser.add_argument("--stop-delay", type=float, default=0.1, help="seconds to continue go infinite after first info line")
    parser.add_argument("--max-plies", type=int, default=300, help="cap each random game; capped games score as draws")
    parser.add_argument("--artifact", type=Path, help="new JSONL evidence path; existing files are not overwritten")
    args = parser.parse_args()

    if args.random_games < 0 or args.random_depth < 1 or args.search_timeout <= 0 or args.max_plies < 1:
        parser.error("random-games must be >= 0, random-depth >= 1, timeout > 0 and max-plies >= 1")

    writer = JsonlWriter(args.artifact)
    candidate = oracle = None
    passed = False
    try:
        writer.write({
            "kind": "run", "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": sys.argv, "engine": args.engine,
            "stockfish": args.stockfish, "seed": args.seed,
            "random_games_requested": 0 if args.protocol_only else args.random_games,
            "random_depth": args.random_depth, "search_timeout": args.search_timeout,
            "max_plies": args.max_plies,
        })
        oracle = StockfishOracle(args.stockfish)
        candidate = UCIProcess(args.engine, "candidate")
        writer.write({
            "kind": "candidate_uci_identity",
            "identity": [line for line in candidate.handshake_output if line.startswith("id ")],
            "advertised_options": list(candidate.options.values()),
        })
        probes = protocol_probe(
            candidate, oracle, args.search_timeout, args.stop_delay, writer,
        )
        summary = None
        if not args.protocol_only:
            summary = random_evaluation(
                candidate, writer, args.random_games, args.seed,
                args.random_depth, args.search_timeout, args.max_plies,
            )
        probes_passed = bool(probes)
        passed = probes_passed and (args.protocol_only or bool(summary and summary["passed_m1_random_gate"]))
        writer.write({
            "kind": "m1_summary", "protocol_probes": len(probes),
            "random_evaluation_ran": not args.protocol_only,
            "passed_m1_protocol_probes": probes_passed,
            "status": "protocol_only_pass" if args.protocol_only and probes_passed else ("pass" if passed else "incomplete_or_failed"),
        })
        return 0 if passed else 1
    except (OSError, HarnessError, TimeoutError) as exc:
        writer.write({"kind": "failure", "error": str(exc), "status": "fail"})
        print(f"M1 HARNESS FAILURE: {exc}", file=sys.stderr)
        return 2
    finally:
        if candidate is not None:
            candidate.close()
        if oracle is not None:
            oracle.close()
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
