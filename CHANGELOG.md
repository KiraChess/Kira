# Changelog

## Kira 1 Alpha v1.0.1

- Load the bundled NNUE automatically from the executable's directory and use it by default when available.
- Report network loading failures and the active HCE fallback in the UCI log. Retain the `EvalFile` override and the `Eval` option.
- Package the engine and network together for Apple Silicon macOS and Windows x64, with installation instructions and checksums. See the release notes for Windows verification details.
- Preserve the original Alpha's network weights and historical measurements. The earlier 2194.6 internal estimate is not a new measurement of this patch.

## Kira 1 Alpha

- Initial Alpha release. See the [original release](https://github.com/KiraChess/Kira/releases/tag/kira-1-alpha) and [measurement notes](docs/MEASUREMENTS.md).
