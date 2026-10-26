# Verification tools

The Rust tests cover board/hash restoration, NNUE accumulation, known-weight inference, data generation, and record validation. Python tools provide external UCI, perft, opening, and ladder-analysis utilities.

```sh
cargo test --release --locked
python3 tools/verify/m0.py --help
python3 tools/verify/m1.py --help
```

Perft comparison uses an external Stockfish executable only as a move-count oracle. Some corpus tests require locally generated datasets. The published Alpha measurement and its limitations are described in [measurement notes](../../docs/MEASUREMENTS.md).
