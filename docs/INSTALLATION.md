# Installing Kira 1 Alpha

## Supported release

The provided executable is **macOS arm64 for Apple Silicon**, validated on **macOS 27.2**. The Mach-O header declares a minimum OS of 11.0, but older macOS versions have not been verified. No Intel, Linux, or Windows binary is provided. The executable links only to macOS's system library.

Download `kira-1-alpha-macos-arm64.tar.gz` and `SHA256SUMS` from [Kira 1 Alpha](https://github.com/KiraChess/Kira/releases/tag/kira-1-alpha).

```sh
shasum -a 256 kira-1-alpha-macos-arm64.tar.gz
# Compare the result with that file's entry in SHA256SUMS.
tar -xzf kira-1-alpha-macos-arm64.tar.gz
cd kira-1-alpha-macos-arm64
chmod +x kira1a
./kira1a
```

The archive includes the executable, released NNUE, this guide, MIT license, and internal checksums. The executable is unsigned and not notarized. If macOS blocks it, review the downloaded release and checksum before following macOS's approval prompt.

## UCI chess GUI

Add the extracted `kira1a` executable as a UCI engine. Configure:

| Option | Value |
|---|---|
| EvalFile | Absolute path to the extracted `kira1a-retrained.nnue` |
| Eval | NNUE |
| Threads | 1 (the only supported value) |
| Hash | 16 MB by default; the ladder used 64 MB |

Set `EvalFile` before selecting `NNUE`, save the options in your GUI, and check for `info string EvalFile loaded ...` in its engine log. A successful `isready` alone does not prove that a network loaded.

The engine starts in **HCE** mode. Its built-in default `EvalFile` names a legacy Gen1 network that is not shipped in the runtime archive. Explicit configuration is required to use this release's NNUE. If the net fails to load, Kira reports the failure and may continue with HCE or the previously selected configuration.

## Terminal session

Run `./kira1a`, then enter these commands interactively. Replace the example path with the actual absolute path on your Mac. UCI paths are entered without shell quotes, even if they contain spaces.

```text
uci
setoption name EvalFile value /absolute/path/kira-1-alpha-macos-arm64/kira1a-retrained.nnue
setoption name Eval value NNUE
setoption name Hash value 64
setoption name Threads value 1
isready
ucinewgame
position startpos
go movetime 1000
```

Wait for `bestmove ...`, then enter `quit`. `uci` should print `id name Kira 1 Alpha`, `id author whosnds`, and `uciok`; `isready` should print `readyok`. The engine also accepts clock controls, increments, depth and node limits, `go infinite`, and `stop`.

Do not send `quit` immediately after `go` in a shell pipe when expecting a completed search: `quit` stops the active search.

## Checksums

- Engine SHA-256: `34b12d73f2342e2b0a7cfac76231a176e960e2d6a929278179de051dbb6cf17c`
- NNUE SHA-256: `9ae5897d19b5a3e283c89b9267e2a4c28381dc1fde5aba2513d6c7d52994f756`

The release-level `SHA256SUMS` covers each downloadable asset except itself. The runtime archive has a separate internal checksum file.
