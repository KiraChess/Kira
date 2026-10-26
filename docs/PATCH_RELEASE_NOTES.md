# Kira 1 Alpha v1.0.1

This usability and compatibility hotfix makes Kira easier to install. Extract the ZIP for your platform and add `kira1a` (macOS) or `kira1a.exe` (Windows) to a UCI chess GUI. Keep the included `kira1a-retrained.nnue` beside the executable: Kira loads it automatically and uses NNUE when it is available.

If the network cannot load, Kira reports the failure and active HCE fallback in its UCI log. `EvalFile` remains available for overriding the network path. See [installation instructions](https://github.com/KiraChess/Kira/blob/kira-1-alpha-v1.0.1/docs/INSTALLATION.md).

Downloads:

- `kira-1-alpha-v1.0.1-macos-arm64.zip` — Apple Silicon macOS.
- `kira-1-alpha-v1.0.1-windows-x64.zip` — Windows x64; see the published release for verification details.
- `SHA256SUMS` — checksums for the release files.

The original frozen Alpha candidate's internal Stockfish-UCI_Elo anchored estimate was **2194.6 Elo (95% CI 2149.2–2240.0)**. This is historical calibration evidence, not a new measurement of v1.0.1 or an official CCRL rating. Possible score-adjudication effects on 207 ladder games remain unquantified; see [measurement notes](https://github.com/KiraChess/Kira/blob/kira-1-alpha-v1.0.1/docs/MEASUREMENTS.md). The network weights are unchanged and the original match campaigns were not repeated.

The original [Kira 1 Alpha release](https://github.com/KiraChess/Kira/releases/tag/kira-1-alpha) and its artifacts remain available.
