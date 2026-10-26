# Reproducibility

The release includes source code, the trained NNUE, SHA-256 checksums, a compact measurement report, and recorded evaluation games.

## Build

```sh
git clone https://github.com/KiraChess/Kira.git
cd Kira
git checkout kira-1-alpha
cargo build --release --locked
```

Rust/Cargo 1.98.1 was used for the validated Apple Silicon macOS build. The engine has no external Cargo dependencies. Obtain the released NNUE separately, set its absolute EvalFile path, and select Eval = NNUE as described in [installation](INSTALLATION.md).

## Release identity

`release-manifest.json` and `SHA256SUMS` identify the distributed files. The public materials contain the engine source, build instructions, and release measurement data. The executable differs from the evaluated Alpha only in author metadata; search/evaluation logic and NNUE weights are unchanged. The existing evaluation results were retained without repeating the match campaigns.

Checksums authenticate file identity; they do not independently prove the strength estimate. Exact binary reproduction can depend on the compiler, linker, platform, and toolchain. Ordinary source builds are supported using the commands above; byte identity across other environments is not promised.

## Evaluation record

The recorded ladder used Stockfish 19 UCI_Elo 1900/2000/2100 at 60+0.6, 64 MB hash, one thread per engine, 100 games per anchor, and color-reversed opening pairs. The optional evaluation-games archive contains the ladder, promotion SPRT, and stability-soak PGNs. It excludes obsolete experiments and internal review notes. See [measurement notes](MEASUREMENTS.md) for the estimator and PV/adjudication caveat.

## Training

The NNUE was trained on Kira-generated self-play using the [training pipeline](../tools/train/README.md). The generation datasets and optimizer checkpoints are not distributed. No claim of deterministic retraining is made. The released net retains SHA-256 `9ae5897d19b5a3e283c89b9267e2a4c28381dc1fde5aba2513d6c7d52994f756`.
