# Kira training pipeline

Kira generates its own search-score and game-result labels. The Rust adapter, training-record codec, feature mapping, exporter, and reference inference live in this repository. Bullet supplies the generic optimizer and Metal backend. External chess evaluations and datasets are not used for labels.

## Components

- `src/record.rs`: Kira-owned 32-byte training record and sparse feature mapping.
- `src/main.rs`: corpus conversion, deduplication, game-based split, and deterministic interleaving.
- `driver/src/main.rs` and `run_driver.py`: isolated training driver with pinned upstream source and dependency checks.
- `src/bin/export_nnue.rs`: quantized network export.
- `src/bin/verify_nnue.rs`: float/integer inference comparison.

## Data and target

Scores are centipawns from the side-to-move's view; results are 0, 0.5, or 1 from the same view. The target is:

```text
0.7 * sigmoid(score / 400) + 0.3 * result
```

Exactly 2% of complete game identities are held out for validation. Deduplication of exact side-to-move-first feature inputs precedes the split, so identical model inputs cannot cross train and validation. When combining generations, the newer source takes precedence. Training rows are deterministically interleaved across games.

The trainer's piece order and engine order differ for bishop/rook rows; the exporter handles that mapping. The input layer has 768 features per perspective, 128 hidden neurons per perspective, clipped ReLU, and a side-to-move-first output. The output scale is 400.

## Upstream trainer

The recorded Bullet source revision is `579b78b37a694cacd6e9ede4706386c15c34837c`. `patches/bullet-training-only.patch` removes unused chess-format loaders. The driver validates its sanitized source manifest and locked dependency graph and uses Kira's own records/features. Bullet's MIT notice is in [the third-party licenses](../../docs/licenses/Bullet-MIT.txt).

Run the adapter after generating a Kira corpus with the data generator. The input must be a `dataset-{SHA256}.k1d` file beside its manifest.

```sh
cargo run --release --manifest-path tools/train/Cargo.toml -- \
  --input /path/to/dataset-SHA256.k1d \
  --out /path/to/new-training-data \
  --bullet-root /path/to/pinned-bullet-checkout \
  --seed 0x4b4952414d345631
python3 tools/train/run_driver.py --help
```

The converter validates input provenance and refuses existing output directories. The trainer uses AdamW, batches of 16,384, a decaying learning rate, and validation-based early stopping. The Alpha training seed was `0x00004b494d345452`.

Large self-play datasets and optimizer checkpoints are not included in the release. Reproducing training requires generating suitable data and obtaining the pinned upstream trainer; identical weights across retraining runs are not promised.
