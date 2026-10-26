# Installing Kira 1 Alpha v1.0.1

Kira is a UCI chess engine. You need a chess GUI that supports UCI engines to play on a graphical board.

## Download and install

Download the appropriate ZIP and `SHA256SUMS` from the [releases page](https://github.com/KiraChess/Kira/releases):

- **Apple Silicon Mac:** `kira-1-alpha-v1.0.1-macos-arm64.zip`
- **Windows x64:** `kira-1-alpha-v1.0.1-windows-x64.zip`

Compare the ZIP's SHA-256 checksum with its entry in `SHA256SUMS`, then extract the entire ZIP. On macOS, use `shasum -a 256 path/to/download.zip`; on Windows PowerShell, use `Get-FileHash path\to\download.zip -Algorithm SHA256`.

In your chess GUI, add the extracted `kira1a` (macOS) or `kira1a.exe` (Windows) as a UCI engine. Keep `kira1a-retrained.nnue` in the same folder as the executable. Kira looks for it relative to the executable, regardless of the GUI's working directory, and selects NNUE when it loads successfully. No Rust installation or manual network configuration is required.

The original macOS Alpha was validated on Apple Silicon with macOS 27.2; older macOS versions have not been verified. The macOS executable is unsigned and not notarized. If macOS blocks it, check the download and checksum before following the system approval prompt. See the release notes for Windows verification details.

## Network and UCI options

Check the GUI's engine log for `info string NNUE loaded ...; NNUE is active` after startup. If the bundled network is missing, corrupt or incompatible, Kira reports `info string NNUE unavailable ...; HCE is active` and runs using its handcrafted evaluation instead. `readyok` alone does not confirm that NNUE loaded. If this happens, restore the network beside the executable and restart the engine, or supply a valid network with **EvalFile**.

| Option | Use |
|---|---|
| EvalFile | Path to a compatible NNUE file; overrides the bundled default. |
| Eval | Select `NNUE` or `HCE`. |
| Threads | 1 (the only supported value). |
| Hash | 16 MB by default; the historical ladder used 64 MB. |

To override the network, set `EvalFile` in your GUI and check the engine log for the loading result. If needed, select `Eval = NNUE` after loading. Consult your GUI's documentation for where to configure UCI options.

## Terminal session

Launch the extracted executable and enter the following commands interactively:

```text
uci
isready
ucinewgame
position startpos
go movetime 1000
```

Wait for `bestmove ...`, then enter `quit`. The engine identifies itself and responds with `uciok` and `readyok`. To use a different network, enter `setoption name EvalFile value /absolute/path/to/network.nnue`, followed if needed by `setoption name Eval value NNUE`. UCI paths are entered without shell quotes, even if they contain spaces. On Windows, use the corresponding Windows path.

Do not send `quit` immediately after `go` in a shell pipe when expecting a completed search: `quit` stops the active search.
