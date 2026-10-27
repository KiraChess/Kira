#!/usr/bin/env python3
"""Package a native build and exercise it from an unrelated working directory."""
import argparse
import hashlib
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
NET_NAME = "kira1a-retrained.nnue"
NET_URL = "https://github.com/KiraChess/Kira/releases/download/kira-1-alpha/kira1a-9ae5897d.nnue"
NET_SHA = "9ae5897d19b5a3e283c89b9267e2a4c28381dc1fde5aba2513d6c7d52994f756"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoke(executable, cwd, expected, override=None):
    proc = subprocess.Popen([str(executable)], cwd=cwd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    lines = queue.Queue()
    def read_output():
        for line in proc.stdout:
            lines.put(line.rstrip())
        lines.put(None)
    threading.Thread(target=read_output, daemon=True).start()
    seen = []
    def send(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
    def until(predicate):
        while True:
            try:
                line = lines.get(timeout=20)
            except queue.Empty:
                raise RuntimeError(f"UCI timeout; output: {seen}") from None
            if line is None:
                raise RuntimeError(f"Engine exited early; output: {seen}")
            seen.append(line)
            print(line, flush=True)
            if predicate(line):
                return
    try:
        send("uci")
        until(lambda line: line == "uciok")
        if not any(expected in line for line in seen):
            raise RuntimeError(f"Missing startup diagnostic {expected!r}: {seen}")
        if not any(line.startswith("id name Kira 1 Alpha") for line in seen):
            raise RuntimeError("Incorrect engine identity")
        send("isready")
        until(lambda line: line == "readyok")
        if override is not None:
            send(f"setoption name EvalFile value {override}")
            until(lambda line: line.startswith("info string EvalFile "))
            if "EvalFile loaded" not in seen[-1] or "NNUE is active" not in seen[-1]:
                raise RuntimeError(f"Network override failed: {seen[-1]}")
        send("ucinewgame")
        send("position startpos")
        send("go depth 2")
        until(lambda line: line.startswith("bestmove "))
        if not re.fullmatch(r"bestmove [a-h][1-8][a-h][1-8][qrbn]?(?: ponder [a-h][1-8][a-h][1-8][qrbn]?)?", seen[-1]):
            raise RuntimeError(f"Invalid bestmove for starting position: {seen[-1]}")
        send("quit")
        proc.wait(timeout=20)
        if proc.returncode != 0:
            raise RuntimeError(f"Engine exited with status {proc.returncode}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=["windows-x64", "macos-arm64", "linux-x64"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--network", type=Path, help="Use a local copy of the frozen network instead of downloading it")
    args = parser.parse_args()
    exe_name = "kira1a.exe" if args.platform == "windows-x64" else "kira1a"
    binary = ROOT / "target" / args.target / "release" / exe_name
    if not binary.is_file():
        raise SystemExit(f"Missing built executable: {binary}")
    out = ROOT / "dist" / "release"
    out.mkdir(parents=True, exist_ok=True)
    name = f"kira-1-alpha-v1.0.2-{args.platform}"
    with tempfile.TemporaryDirectory(prefix="kira package ") as temp:
        temp = Path(temp)
        bundle = temp / name
        bundle.mkdir()
        shutil.copy2(binary, bundle / exe_name)
        shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
        shutil.copy2(ROOT / "tools" / "release" / "INSTALL.txt", bundle / "INSTALL.txt")
        if args.network:
            net_bytes = args.network.read_bytes()
        else:
            with urllib.request.urlopen(NET_URL, timeout=60) as response:
                net_bytes = response.read()
        (bundle / NET_NAME).write_bytes(net_bytes)
        if sha256(bundle / NET_NAME) != NET_SHA:
            raise RuntimeError("Released network SHA-256 does not match the frozen Alpha network")
        archive = out / (name + ".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(bundle.iterdir()):
                z.write(file, Path(name) / file.name)
        extracted = temp / "extracted package"
        extracted.mkdir()
        if args.platform in ("macos-arm64", "linux-x64"):
            # Use the system extractor to verify that the ZIP preserves executable mode.
            subprocess.run(["unzip", "-q", str(archive), "-d", str(extracted)], check=True)
        else:
            env = os.environ.copy()
            env["KIRA_ARCHIVE"] = str(archive)
            env["KIRA_EXTRACTED"] = str(extracted)
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "Expand-Archive -LiteralPath $env:KIRA_ARCHIVE -DestinationPath $env:KIRA_EXTRACTED"],
                           env=env, check=True)
        installed = extracted / name
        if sha256(installed / NET_NAME) != NET_SHA:
            raise RuntimeError("Extracted network SHA-256 mismatch")
        unrelated = temp / "unrelated working directory"
        unrelated.mkdir()
        override = unrelated / "network override with spaces.nnue"
        override.write_bytes(net_bytes)
        smoke(installed / exe_name, unrelated, "NNUE is active", override)
        (installed / NET_NAME).unlink()
        smoke(installed / exe_name, unrelated, "HCE is active", override)
        (installed / NET_NAME).write_bytes(b"corrupt network")
        smoke(installed / exe_name, unrelated, "HCE is active")
        (out / (name + ".sha256")).write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
        print(f"Created {archive} (SHA-256 {sha256(archive)})")


if __name__ == "__main__":
    main()
