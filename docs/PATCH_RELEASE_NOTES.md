# Kira 1 Alpha v1.0.2

Kira now has a **Linux x64 build**, alongside the Apple Silicon macOS and Windows x64 builds. This release also cleans up the repository and installation docs to make them easier to read.

Download the ZIP for your platform, extract it, and add `kira1a` (macOS or Linux) or `kira1a.exe` (Windows) to a UCI chess GUI. Keep `kira1a-retrained.nnue` next to the executable so Kira can load it automatically. On Linux, run `chmod +x kira1a` if necessary. Check `SHA256SUMS` to verify your download. See the [installation guide](https://github.com/KiraChess/Kira/blob/kira-1-alpha-v1.0.2/docs/INSTALLATION.md).

If the network cannot load, Kira reports the failure in the UCI log and falls back to handcrafted evaluation (HCE). You can override the network path with the `EvalFile` UCI option.

The macOS build is unsigned and not notarized. The Windows build passed a UCI runtime smoke test on Windows Server 2022 for v1.0.1, but has not been tested in an interactive chess GUI.

The original Alpha's **2194.6 Elo estimate (95% CI 2149.2–2240.0)** is an internal Stockfish-UCI_Elo anchored estimate, not an official CCRL rating or a new measurement of this patch. Possible score-adjudication effects on **207 ladder games** remain unquantified; see the [measurement notes](https://github.com/KiraChess/Kira/blob/kira-1-alpha-v1.0.2/docs/MEASUREMENTS.md). The NNUE weights are unchanged.

Earlier [releases](https://github.com/KiraChess/Kira/releases) remain available.
