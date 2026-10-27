# Kira

**Kira 1 Alpha** is a Rust chess engine with a small NNUE trained on Kira's own games. It's a solo hobby project by **whosnds**, released by **KiraChess**.

[Download the latest release](https://github.com/KiraChess/Kira/releases/tag/kira-1-alpha-v1.0.2) · [Installation and UCI](docs/INSTALLATION.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [MIT license](LICENSE)

## Get started

Kira is a UCI engine, so you'll need a compatible chess GUI for a graphical board. Download a ZIP from the [releases page](https://github.com/KiraChess/Kira/releases), extract it, and add the executable to your GUI:

- **macOS Apple Silicon:** select `kira1a` in the extracted folder. The original Alpha was validated on macOS 27.2; v1.0.1 was also smoke tested on macOS 14. The binary is unsigned and not notarized, so macOS may ask you to approve it.
- **Windows x64:** select `kira1a.exe` in the extracted folder. See the release notes for Windows verification details.
- **Linux x64:** select `kira1a` in the extracted folder. If needed, make it executable with `chmod +x kira1a`.

Keep `kira1a-retrained.nnue` beside the executable. Kira loads it automatically; no Rust installation or manual network setup is needed. If loading fails, Kira reports it in the UCI log and falls back to handcrafted evaluation (HCE). You can override the path with **EvalFile** and choose **Eval = NNUE** or **HCE**. See [installation instructions](docs/INSTALLATION.md) for details.

## Measured strength

The original frozen Alpha candidate's **internal Stockfish-UCI_Elo anchored estimate is 2194.6 Elo**, with a **95% confidence interval of 2149.2–2240.0**. This comes from 300 games at **60+0.6**, 100 each against Stockfish 19 UCI_Elo 1900/2000/2100, 64 MB hash and one thread per engine, using reversed-color opening pairs.

**This is not an official CCRL, FIDE, or Lichess rating.** No independently rated reference engine was available locally. The interval describes sampling uncertainty; it does not account for calibration or adjudication bias.

**PV/adjudication caveat:** fastchess recorded 5,289 Stockfish-named PV/bestmove warnings in the ladder. Reviewers matched the warning-named moves to the recorded Stockfish moves, but possible score-adjudication effects on **207 ladder games** remain unquantified. The 1,000-game soak also had 12,085 such warnings and 419 score-adjudicated Stockfish games. The warnings are retained as an unresolved limitation of the measurement.

| Existing verification | Recorded result |
|---|---|
| Move generation | 12 required perft vectors; 100,000 depth-3 differential positions, including 10,000 at depth 4; zero mismatches |
| Promotion | SPRT H1 accepted against Gen1 NNUE and Kira-HCE |
| Stability soak | 1,000 games; 838 wins, 32 draws, 130 losses; no recorded technical failures caused by Kira |
| Quantized parity | 100,000 positions; mean absolute difference 0.05168 cp, maximum 1 cp |
| Original release packaging | Creator-metadata rebuild; search/evaluation logic and NNUE unchanged |

The soak is a stability/matchup check. Its score is not an Elo estimate. Release checksums, a concise measurement report, and the recorded evaluation games are available with the release. [Measurement notes](docs/MEASUREMENTS.md) describe scope and limitations.

## Build from source

```sh
git clone https://github.com/KiraChess/Kira.git
cd Kira
git checkout kira-1-alpha-v1.0.2
cargo build --release --locked
./target/release/kira1a
```

The validated toolchain was **Rust/Cargo 1.98.1**. The engine has no external Cargo dependencies. Place the released `kira1a-retrained.nnue` next to the built executable, or set its path with the UCI `EvalFile` option. See the release notes for platform verification details.

The historical measurements apply to the original Alpha candidate, not a new measurement of v1.0.2. The hotfixes did not retrain the network or repeat the recorded match campaigns.

## Engine and training

Kira implements legal move generation, FEN handling, Zobrist hashing, make/unmake, PVS alpha-beta search, iterative deepening, quiescence search, transposition tables, killer/history move ordering, repetition and fifty-move handling, UCI time controls, and HCE/NNUE evaluation.

The network uses 768 piece-square inputs, two 128-neuron perspectives, clipped ReLU, side-to-move-first output, and quantized inference. Training used Kira's own record format and self-play data with Bullet's generic optimizer and Metal backend. Stockfish was used for perft checks and match opposition, never to label training data. Sliding attacks use direct ray scans.

See [score and feature conventions](docs/CONVENTIONS.md), [training pipeline](tools/train/README.md), and [reproducibility](docs/REPRODUCIBILITY.md).

## License

Kira's original source, documentation, and released NNUE weights are licensed under **MIT**. Optional third-party development tools retain their own licenses; see [third-party notes](docs/THIRD_PARTY.md).
