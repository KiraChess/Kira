use kira1a::{
    Board,
    datagen,
    eval::{Evaluator, nnue::NnueNetwork, nnue_quantized::{QuantizedNnueNetwork, sha256_hex}},
};
use kira_m4_adapter::{
    bullet_feature_to_kira, nnue_feature_key, NnueFeatureKey, NNUE_FEATURE_KEY_SCHEMA,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

const HIDDEN: usize = 128;
const SAMPLE_COUNT: usize = 100_000;
const DEFAULT_SEED: u64 = 0x4b49_5241_4d34_5632; // "KIRAM4V2"
const MAX_MAE_CP: f64 = 2.0;
const MAX_ABS_ERROR_CP: u32 = 10;
const MAX_START_ABS_CP: i32 = 30;
const MIN_QUEEN_UP_CP: i32 = 600;
const TRAINER_FORMAT: &str = "kira-m4-bullet-trainer-v1";
const SANITIZED_TRAINER_FORMAT: &str = "kira-m4-kira-record-trainer-v5-sanitized-locked-source-tree";
const BULLET_PATCH_SHA256: &str = "ecf8ce81486d5e2d3cfbd881bbd053e8b3682b8de4acf24a6824310cb29b5141";
const BULLET_COMMIT: &str = "579b78b37a694cacd6e9ede4706386c15c34837c";
const PINNED_GEN2_NNUE_SHA256: &str = "842e3870cdb02aa762b7d1f4ab0ec441c9df9b6de911395ba759ee1840cefca9";

fn main() {
    if let Err(error) = run_cli() {
        eprintln!("kira-m4-verify-nnue: {error}");
        std::process::exit(2);
    }
}

fn run_cli() -> Result<(), String> {
    let mut dataset = None;
    let mut gen2_dataset = None;
    let mut gen1_dataset = None;
    let mut run_dir = None;
    let mut network_path = None;
    let mut seed = DEFAULT_SEED;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                println!("Usage: kira-m4-verify-nnue --dataset APPROVED_M3.k1d --run-dir TRAINING_RUN --network MODEL.nnue [--seed U64]");
                println!("   or: kira-m4-verify-nnue --gen2-dataset APPROVED_GEN2.k1d --gen1-dataset APPROVED_GEN1.k1d --run-dir TRAINING_RUN --network MODEL.nnue [--seed U64]");
                println!("Evaluates {SAMPLE_COUNT} deterministic distinct dataset positions; default seed 0x{DEFAULT_SEED:016x}.");
                return Ok(());
            }
            "--dataset" => dataset = Some(PathBuf::from(next_arg(&mut args, "--dataset")?)),
            "--gen2-dataset" => gen2_dataset = Some(PathBuf::from(next_arg(&mut args, "--gen2-dataset")?)),
            "--gen1-dataset" => gen1_dataset = Some(PathBuf::from(next_arg(&mut args, "--gen1-dataset")?)),
            "--run-dir" => run_dir = Some(PathBuf::from(next_arg(&mut args, "--run-dir")?)),
            "--network" => network_path = Some(PathBuf::from(next_arg(&mut args, "--network")?)),
            "--seed" => seed = parse_seed(&next_arg(&mut args, "--seed")?)?,
            other => return Err(format!("unknown argument {other}; use --help")),
        }
    }
    let run_dir = run_dir.ok_or_else(|| "--run-dir is required".to_string())?;
    let network_path = network_path.ok_or_else(|| "--network is required".to_string())?;
    match (dataset, gen2_dataset, gen1_dataset) {
        (Some(dataset), None, None) => verify(Some(&dataset), None, None, &run_dir, &network_path, seed),
        (None, Some(gen2), Some(gen1)) => verify(None, Some(&gen2), Some(&gen1), &run_dir, &network_path, seed),
        _ => Err("provide either --dataset alone or both --gen2-dataset and --gen1-dataset".into()),
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_seed(value: &str) -> Result<u64, String> {
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|error| format!("invalid hexadecimal seed: {error}"))
    } else {
        value.parse::<u64>().map_err(|error| format!("invalid decimal seed: {error}"))
    }
}

fn verify(
    dataset_path: Option<&Path>,
    gen2_dataset_path: Option<&Path>,
    gen1_dataset_path: Option<&Path>,
    run_dir: &Path,
    network_path: &Path,
    seed: u64,
) -> Result<(), String> {
    let network_bytes = fs::read(network_path).map_err(io_error)?;
    let network_sha = sha256_hex(&network_bytes);
    let quantized = QuantizedNnueNetwork::<HIDDEN>::load(network_path)?;

    let run_manifest_bytes = fs::read(run_dir.join("run_manifest.txt")).map_err(io_error)?;
    let run_manifest_sha = sha256_hex(&run_manifest_bytes);
    if quantized.metadata().run_manifest_sha256 != parse_sha256(&run_manifest_sha, "run manifest SHA-256")? {
        return Err("NNUE run-manifest SHA-256 does not match the supplied training run".into());
    }
    let run_manifest = parse_key_values(
        std::str::from_utf8(&run_manifest_bytes).map_err(|error| format!("run manifest is not UTF-8: {error}"))?,
        "run manifest",
    )?;
    validate_run_manifest(&run_manifest)?;

    let best = parse_best_epoch(&fs::read_to_string(run_dir.join("losses.csv")).map_err(io_error)?)?;
    let checkpoint_dir = run_dir.join(format!("checkpoint-{:04}", best.epoch));
    let checkpoint_manifest = parse_key_values(
        &fs::read_to_string(checkpoint_dir.join("manifest.txt")).map_err(io_error)?,
        "checkpoint manifest",
    )?;
    validate_checkpoint_manifest(&checkpoint_manifest, &run_manifest, best)?;
    let weight_bytes = fs::read(checkpoint_dir.join("weights.bin")).map_err(io_error)?;
    let weights_sha = sha256_hex(&weight_bytes);
    if quantized.metadata().source_weights_sha256 != parse_sha256(&weights_sha, "weights SHA-256")? {
        return Err("NNUE source-weights SHA-256 does not match the selected best checkpoint".into());
    }
    let float_network = build_float_network(&weight_bytes)?;

    let probes = float_sanity_probes(&float_network)?;
    println!("float_probe_start_white_cp={}", probes.start_white);
    println!("float_probe_start_black_cp={}", probes.start_black);
    println!("float_probe_queen_up_cp={}", probes.queen_up);
    println!("float_probe_queen_mirror_cp={}", probes.queen_mirror);
    println!("float_probe_pass={}", probes.passed);
    println!("sample_seed=0x{seed:016x}");
    println!("sample_target={SAMPLE_COUNT}");
    let (positions, sample_hash, sample_hash_label) = match run_manifest.get("dataset_mode").map(String::as_str) {
        Some("multi-source-v2") => {
            let gen2 = gen2_dataset_path.ok_or_else(|| "M6 run requires --gen2-dataset and --gen1-dataset".to_string())?;
            let gen1 = gen1_dataset_path.ok_or_else(|| "M6 run requires --gen2-dataset and --gen1-dataset".to_string())?;
            verify_m6_source_provenance(gen2, gen1, &run_manifest)?;
            println!("sampling_gen2_dataset={} sha256_expected={}", gen2.display(), run_manifest["source_gen2_dataset_sha256"]);
            println!("sampling_gen1_dataset={} sha256_expected={}", gen1.display(), run_manifest["source_gen1_dataset_sha256"]);
            let sample = sample_multi_dataset(gen2, gen1, seed, SAMPLE_COUNT)?;
            if sample.gen2_summary.sha256 != run_manifest["source_gen2_dataset_sha256"]
                || sample.gen1_summary.sha256 != run_manifest["source_gen1_dataset_sha256"]
            {
                return Err("source artifact SHA-256s disagree with the M6 run manifest".into());
            }
            let expected_kept = parse_u64_value(&run_manifest, "kept_records")?;
            let expected_raw = parse_u64_value(&run_manifest, "raw_records")?;
            let expected_dropped = parse_u64_value(&run_manifest, "dropped_duplicate_records")?;
            let expected_cross = parse_u64_value(&run_manifest, "cross_source_duplicate_records")?;
            let expected_within = parse_u64_value(&run_manifest, "within_source_duplicate_records")?;
            if sample.kept_positions != expected_kept
                || sample.gen2_summary.unique_positions + sample.gen1_summary.unique_positions != expected_raw
                || sample.dropped_duplicates != expected_dropped
                || sample.cross_duplicates != expected_cross
                || sample.within_duplicates != expected_within
                || sample.gen2_dropped != parse_u64_value(&run_manifest, "source_gen2_dropped_duplicate_records")?
                || sample.gen1_dropped != parse_u64_value(&run_manifest, "source_gen1_dropped_duplicate_records")?
                || sample.gen1_within != parse_u64_value(&run_manifest, "source_gen1_within_source_duplicate_records")?
                || sample.cross_duplicates != parse_u64_value(&run_manifest, "source_gen1_cross_source_duplicate_records")?
            {
                return Err("NNUE feature-key dedup counts disagree with the M6 run manifest".into());
            }
            println!("source_gen2_sha256={}", sample.gen2_summary.sha256);
            println!("source_gen1_sha256={}", sample.gen1_summary.sha256);
            println!("source_raw_positions={}", sample.gen2_summary.unique_positions + sample.gen1_summary.unique_positions);
            println!("source_cross_duplicates={}", sample.cross_duplicates);
            println!("source_within_duplicates={}", sample.within_duplicates);
            println!("source_kept_positions={}", sample.kept_positions);
            (sample.positions, sample.sample_hash, "sample_nnue_feature_key_sequence_sha256")
        }
        Some("single-source-v1") | None => {
            let dataset = dataset_path.ok_or_else(|| "M4 run requires --dataset".to_string())?;
            if gen2_dataset_path.is_some() || gen1_dataset_path.is_some() {
                return Err("single-source training run does not accept gen1/gen2 dataset arguments".into());
            }
            println!("sampling_dataset={} sha256_expected={}", dataset.display(), run_manifest["m3_dataset_sha256"]);
            let (positions, source_summary, sample_hash) = sample_dataset(dataset, seed, SAMPLE_COUNT)?;
            if source_summary.sha256 != run_manifest["m3_dataset_sha256"] {
                return Err(format!(
                    "dataset SHA-256 {} disagrees with run manifest {}",
                    source_summary.sha256, run_manifest["m3_dataset_sha256"]
                ));
            }
            println!("dataset_sha256={}", source_summary.sha256);
            println!("dataset_candidate_rows={}", source_summary.stored_candidate_rows);
            (positions, sample_hash, "sample_zobrist_sequence_sha256")
        }
        Some(other) => return Err(format!("unsupported training dataset mode {other}")),
    };
    if positions.len() != SAMPLE_COUNT {
        return Err(format!("dataset supplied only {} distinct positions; need {SAMPLE_COUNT}", positions.len()));
    }

    let mut total_error = 0u64;
    let mut max_error = 0u32;
    for (index, position) in positions.iter().enumerate() {
        let float_score = float_network.evaluate(&position.board);
        let int_score = quantized.evaluate(&position.board);
        let error = float_score.abs_diff(int_score);
        total_error += u64::from(error);
        max_error = max_error.max(error);
        if index > 0 && index % 25_000 == 0 {
            eprintln!("evaluated_positions={index}");
        }
    }
    let mae = total_error as f64 / positions.len() as f64;
    let parity_pass = mae <= MAX_MAE_CP && max_error <= MAX_ABS_ERROR_CP;
    println!("sample_count={}", positions.len());
    println!("{sample_hash_label}={sample_hash}");
    println!("selected_best_epoch={}", best.epoch);
    println!("selected_checkpoint_weights_sha256={weights_sha}");
    println!("network_sha256={network_sha}");
    println!("parity_mae_cp={mae:.8}");
    println!("parity_max_abs_error_cp={max_error}");
    println!("parity_thresholds=mae<={MAX_MAE_CP:.1};max_abs<={MAX_ABS_ERROR_CP}");
    println!("parity_pass={parity_pass}");
    let overall_pass = parity_pass && probes.passed;
    println!("overall_pass={overall_pass}");
    if overall_pass { Ok(()) } else { Err("float sanity or float/int parity exceeded a required threshold".into()) }
}

#[derive(Clone, Copy, Debug)]
struct FloatProbes {
    start_white: i32,
    start_black: i32,
    queen_up: i32,
    queen_mirror: i32,
    passed: bool,
}

fn float_sanity_probes(network: &NnueNetwork<HIDDEN>) -> Result<FloatProbes, String> {
    let start_white = Board::startpos();
    let start_black = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")?;
    let queen_up = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")?;
    let queen_mirror = Board::from_fen("3qk3/8/8/8/8/8/8/4K3 b - - 0 1")?;
    let probes = FloatProbes {
        start_white: network.evaluate(&start_white),
        start_black: network.evaluate(&start_black),
        queen_up: network.evaluate(&queen_up),
        queen_mirror: network.evaluate(&queen_mirror),
        passed: true,
    };
    let passed = probes.start_white.abs() <= MAX_START_ABS_CP
        && probes.start_black.abs() <= MAX_START_ABS_CP
        && probes.queen_up > MIN_QUEEN_UP_CP
        && probes.queen_up == probes.queen_mirror;
    Ok(FloatProbes { passed, ..probes })
}

#[derive(Clone, Debug)]
struct Position {
    zobrist: u64,
    board: Board,
}

struct Reservoir<T> {
    capacity: usize,
    seen: u64,
    rng: SplitMix64,
    items: Vec<T>,
}

impl<T> Reservoir<T> {
    fn new(capacity: usize, seed: u64) -> Result<Self, String> {
        if capacity == 0 { return Err("reservoir capacity must be positive".into()); }
        Ok(Self { capacity, seen: 0, rng: SplitMix64::new(seed), items: Vec::with_capacity(capacity) })
    }

    fn push(&mut self, item: T) -> Result<(), String> {
        self.seen = self.seen.checked_add(1).ok_or_else(|| "candidate row count overflow".to_string())?;
        if self.items.len() < self.capacity {
            self.items.push(item);
            return Ok(());
        }
        let slot = self.rng.below(self.seen)?;
        if slot < self.capacity as u64 {
            self.items[slot as usize] = item;
        }
        Ok(())
    }
}

struct SplitMix64 { state: u64 }

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn below(&mut self, bound: u64) -> Result<u64, String> {
        if bound == 0 { return Err("reservoir random bound must be positive".into()); }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold { return Ok(value % bound); }
        }
    }
}

fn sample_dataset(path: &Path, seed: u64, capacity: usize) -> Result<(Vec<Position>, datagen::ArtifactSummary, String), String> {
    let mut reservoir = Reservoir::new(capacity, seed)?;
    let mut sample_error = None;
    let mut processed_games = 0u64;
    let summary = datagen::inspect_artifact(path, |game| {
        processed_games += 1;
        if processed_games % 5_000 == 0 {
            eprintln!("audited_games={processed_games} reservoir_positions={}", reservoir.items.len());
        }
        if sample_error.is_some() { return; }
        for candidate in &game.candidates {
            let board = match Board::from_fen(&candidate.fen) {
                Ok(board) => board,
                Err(error) => {
                    sample_error = Some(format!("candidate FEN failed to parse after artifact audit: {error}"));
                    return;
                }
            };
            if board.hash() != candidate.zobrist {
                sample_error = Some(format!("candidate FEN hash mismatch at ply {}", candidate.ply));
                return;
            }
            if let Err(error) = reservoir.push(Position { zobrist: candidate.zobrist, board }) {
                sample_error = Some(error);
                return;
            }
        }
    })?;
    if let Some(error) = sample_error { return Err(error); }
    if reservoir.seen != summary.stored_candidate_rows {
        return Err(format!("reservoir saw {} candidates but artifact summary reports {}", reservoir.seen, summary.stored_candidate_rows));
    }
    let mut unique = HashSet::with_capacity(reservoir.items.len());
    let mut sample_hash_bytes = Vec::with_capacity(reservoir.items.len() * 8);
    for position in &reservoir.items {
        if !unique.insert(position.zobrist) { return Err("reservoir selected a duplicate source position".into()); }
        if position.board.hash() != position.zobrist { return Err("reservoir position key does not match decoded board".into()); }
        sample_hash_bytes.extend_from_slice(&position.zobrist.to_le_bytes());
    }
    let sample_hash = sha256_hex(&sample_hash_bytes);
    Ok((reservoir.items, summary, sample_hash))
}

struct MultiDatasetSample {
    positions: Vec<Position>,
    gen2_summary: datagen::ArtifactSummary,
    gen1_summary: datagen::ArtifactSummary,
    kept_positions: u64,
    dropped_duplicates: u64,
    gen2_dropped: u64,
    gen1_dropped: u64,
    gen1_within: u64,
    within_duplicates: u64,
    cross_duplicates: u64,
    sample_hash: String,
}

#[derive(Clone, Debug)]
struct FeaturePosition {
    position: Position,
    key: NnueFeatureKey,
}

fn sample_multi_dataset(
    gen2_path: &Path,
    gen1_path: &Path,
    seed: u64,
    capacity: usize,
) -> Result<MultiDatasetSample, String> {
    let mut reservoir = Reservoir::new(capacity, seed)?;
    let mut feature_owners = HashMap::<NnueFeatureKey, bool>::new();
    let mut gen2_dropped = 0u64;
    let mut gen1_within = 0u64;
    let mut cross_duplicates = 0u64;
    let mut callback_error = None;
    let gen2_summary = datagen::inspect_artifact(gen2_path, |game| {
        if callback_error.is_some() { return; }
        for candidate in &game.candidates {
            let key = match nnue_feature_key(candidate) {
                Ok(key) => key,
                Err(error) => { callback_error = Some(format!("gen2 NNUE feature key failed: {error}")); return; }
            };
            if feature_owners.contains_key(&key) {
                gen2_dropped += 1;
                continue;
            }
            feature_owners.insert(key.clone(), true);
            let board = match Board::from_fen(&candidate.fen) {
                Ok(board) => board,
                Err(error) => { callback_error = Some(format!("gen2 candidate FEN failed to parse: {error}")); return; }
            };
            if board.hash() != candidate.zobrist {
                callback_error = Some(format!("gen2 candidate FEN hash mismatch at ply {}", candidate.ply));
                return;
            }
            if let Err(error) = reservoir.push(FeaturePosition {
                position: Position { zobrist: candidate.zobrist, board },
                key,
            }) {
                callback_error = Some(error);
                return;
            }
        }
    })?;
    if gen2_summary.kind != "merged-dataset" { return Err("gen2 parity input is not a merged dataset".into()); }
    if let Some(error) = callback_error.take() { return Err(error); }

    let gen1_summary = datagen::inspect_artifact(gen1_path, |game| {
        if callback_error.is_some() { return; }
        for candidate in &game.candidates {
            let key = match nnue_feature_key(candidate) {
                Ok(key) => key,
                Err(error) => { callback_error = Some(format!("gen1 NNUE feature key failed: {error}")); return; }
            };
            match feature_owners.get(&key).copied() {
                Some(true) => { cross_duplicates += 1; continue; }
                Some(false) => { gen1_within += 1; continue; }
                None => { feature_owners.insert(key.clone(), false); }
            }
            let board = match Board::from_fen(&candidate.fen) {
                Ok(board) => board,
                Err(error) => { callback_error = Some(format!("gen1 candidate FEN failed to parse: {error}")); return; }
            };
            if board.hash() != candidate.zobrist {
                callback_error = Some(format!("gen1 candidate FEN hash mismatch at ply {}", candidate.ply));
                return;
            }
            if let Err(error) = reservoir.push(FeaturePosition {
                position: Position { zobrist: candidate.zobrist, board },
                key,
            }) {
                callback_error = Some(error);
                return;
            }
        }
    })?;
    if gen1_summary.kind != "merged-dataset" { return Err("gen1 parity input is not a merged dataset".into()); }
    if let Some(error) = callback_error { return Err(error); }
    let raw_positions = gen2_summary.unique_positions + gen1_summary.unique_positions;
    let within_duplicates = gen2_dropped + gen1_within;
    let dropped_duplicates = within_duplicates + cross_duplicates;
    if gen2_summary.unique_positions < gen2_dropped
        || gen1_summary.unique_positions < gen1_within + cross_duplicates
        || reservoir.seen != raw_positions - dropped_duplicates
        || feature_owners.len() as u64 != reservoir.seen
    {
        return Err("multi-source NNUE feature-key or duplicate totals disagree".into());
    }
    let mut unique_sample = HashSet::with_capacity(reservoir.items.len());
    let mut sample_hash_bytes = Vec::with_capacity(reservoir.items.len() * 81);
    for sampled in &reservoir.items {
        if !unique_sample.insert(sampled.key.clone()) || sampled.position.board.hash() != sampled.position.zobrist {
            return Err("multi-source reservoir contains a duplicate NNUE feature key or invalid Zobrist position".into());
        }
        sample_hash_bytes.extend_from_slice(&sampled.key.to_bytes());
    }
    Ok(MultiDatasetSample {
        positions: reservoir.items.into_iter().map(|sampled| sampled.position).collect(),
        gen2_summary,
        gen1_summary,
        kept_positions: feature_owners.len() as u64,
        dropped_duplicates,
        gen2_dropped,
        gen1_dropped: gen1_within + cross_duplicates,
        gen1_within,
        within_duplicates,
        cross_duplicates,
        sample_hash: sha256_hex(&sample_hash_bytes),
    })
}

fn verify_m6_source_provenance(
    gen2_path: &Path,
    gen1_path: &Path,
    run: &BTreeMap<String, String>,
) -> Result<(), String> {
    let data_dir = PathBuf::from(run.get("adapter_data_dir").ok_or_else(|| "M6 run manifest is missing adapter_data_dir".to_string())?);
    let adapter_manifest_path = data_dir.join("manifest.txt");
    let adapter_bytes = fs::read(&adapter_manifest_path).map_err(io_error)?;
    if sha256_hex(&adapter_bytes) != run["adapter_manifest_sha256"] {
        return Err("M6 adapter manifest SHA-256 disagrees with run manifest".into());
    }
    let adapter = parse_key_values(
        std::str::from_utf8(&adapter_bytes).map_err(|error| format!("M6 adapter manifest is not UTF-8: {error}"))?,
        "M6 adapter manifest",
    )?;
    expect(&adapter, "adapter_format", "kira-m6-multisource-record-v3")?;
    expect(&adapter, "source_precedence", "gen2>gen1;first-occurrence-within-source;duplicate_key=NNUE-active-feature-key-v1")?;
    expect(&adapter, "nnue_feature_key_schema", NNUE_FEATURE_KEY_SCHEMA)?;
    expect(&adapter, "nnue_feature_key_encoding", "sparse-sorted-10-bit-indices;stm-perspective-first;max-32-pieces-per-perspective")?;
    expect(&adapter, "source_pair_sha256", &run["source_pair_sha256"])?;
    if run["m3_dataset_sha256"] != run["source_pair_sha256"] {
        return Err("M6 run dataset identity is not bound to source_pair_sha256".into());
    }
    let sources_path = data_dir.join("sources.tsv");
    let sources_bytes = fs::read(&sources_path).map_err(io_error)?;
    if sha256_hex(&sources_bytes) != adapter["source_pair_sha256"] {
        return Err("M6 sources.tsv SHA-256 disagrees with adapter manifest".into());
    }
    for source in ["gen2", "gen1"] {
        for (run_suffix, adapter_suffix) in [
            ("dataset_sha256", "sha256"),
            ("config_sha256", "config_sha256"),
            ("source_tree_sha256", "source_tree_sha256"),
            ("manifest_sha256", "manifest_sha256"),
            ("engine_sha256", "engine_sha256"),
            ("evaluator", "evaluator"),
        ] {
            let run_key = format!("source_{source}_{run_suffix}");
            let adapter_key = format!("source_{source}_{adapter_suffix}");
            expect(run, &run_key, &adapter[&adapter_key])?;
        }
    }
    expect(&adapter, "source_gen2_search_evaluator", "QuantizedNnueNetwork<128>")?;
    expect(&adapter, "source_gen2_nnue_sha256", PINNED_GEN2_NNUE_SHA256)?;
    let mut raw_source_counts = BTreeMap::new();
    for (source, path) in [("gen2", gen2_path), ("gen1", gen1_path)] {
        let summary = datagen::inspect_artifact(path, |_| {})?;
        if summary.kind != "merged-dataset" { return Err(format!("{source} parity input is not a merged dataset")); }
        expect(&adapter, &format!("source_{source}_raw_records"), &summary.unique_positions.to_string())?;
        raw_source_counts.insert(source, summary.unique_positions);
        expect(run, &format!("source_{source}_dataset_sha256"), &summary.sha256)?;
        expect(&adapter, &format!("source_{source}_sha256"), &summary.sha256)?;
        expect(run, &format!("source_{source}_config_sha256"), &summary.config_sha256)?;
        expect(&adapter, &format!("source_{source}_config_sha256"), &summary.config_sha256)?;
        let expected_name = format!("dataset-{}.k1d", summary.sha256);
        if path.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str()) {
            return Err(format!("{source} parity input is not content-addressed as {expected_name}"));
        }
        let manifest_path = path.parent().unwrap_or(Path::new(".")).join("manifest.txt");
        let manifest_bytes = fs::read(&manifest_path).map_err(io_error)?;
        let manifest_sha = sha256_hex(&manifest_bytes);
        expect(run, &format!("source_{source}_manifest_sha256"), &manifest_sha)?;
        expect(&adapter, &format!("source_{source}_manifest_sha256"), &manifest_sha)?;
        let source_manifest = parse_key_values(
            std::str::from_utf8(&manifest_bytes).map_err(|error| format!("{source} source manifest is not UTF-8: {error}"))?,
            "source manifest",
        )?;
        expect(&source_manifest, "format_version", "2")?;
        expect(&source_manifest, "record_version", "2")?;
        expect(&source_manifest, "config_sha256", &summary.config_sha256)?;
        let source_tree_sha = source_manifest.get("source_sha256")
            .ok_or_else(|| format!("{source} source manifest is missing source_sha256"))?;
        let engine_sha = source_manifest.get("engine_sha256")
            .ok_or_else(|| format!("{source} source manifest is missing engine_sha256"))?;
        let evaluator = source_manifest.get("evaluator")
            .ok_or_else(|| format!("{source} source manifest is missing evaluator"))?;
        expect(run, &format!("source_{source}_source_tree_sha256"), source_tree_sha)?;
        expect(run, &format!("source_{source}_engine_sha256"), engine_sha)?;
        expect(run, &format!("source_{source}_evaluator"), evaluator)?;
        if source == "gen2" {
            expect(&source_manifest, "evaluator", "Kira-HCE")?;
            expect(&source_manifest, "search_evaluator", "QuantizedNnueNetwork<128>")?;
            expect(&source_manifest, "nnue_sha256", PINNED_GEN2_NNUE_SHA256)?;
        } else if source_manifest.get("evaluator").map(String::as_str) != Some("Kira-HCE")
            || source_manifest.contains_key("search_evaluator")
            || source_manifest.contains_key("nnue_sha256")
        {
            return Err("gen1 source manifest does not carry the pinned HCE label provenance".into());
        }
        let revision_fields = ["source_revision", "source_commit", "source_tag"];
        let revisions = revision_fields.iter().filter_map(|key| source_manifest.get(*key)).collect::<Vec<_>>();
        if revisions.windows(2).any(|pair| pair[0] != pair[1]) {
            return Err(format!("{source} source manifest revision fields disagree"));
        }
        let actual_revision = revisions.first().map(|value| value.as_str()).unwrap_or("not-recorded-in-m3-manifest");
        expect(run, &format!("source_{source}_revision"), actual_revision)?;
        if let Some(dataset_sha) = source_manifest.get("dataset_sha256") {
            expect(&source_manifest, "dataset_sha256", &summary.sha256)?;
            if dataset_sha != &summary.sha256 { return Err(format!("{source} source manifest dataset SHA mismatch")); }
        }
    }
    let gen2_raw = parse_u64_value(&adapter, "source_gen2_raw_records")?;
    let gen2_kept = parse_u64_value(&adapter, "source_gen2_kept_records")?;
    let gen2_dropped = parse_u64_value(&adapter, "source_gen2_dropped_duplicate_records")?;
    let gen2_within = parse_u64_value(&adapter, "source_gen2_within_source_duplicate_records")?;
    let gen2_cross = parse_u64_value(&adapter, "source_gen2_cross_source_duplicate_records")?;
    let gen1_raw = parse_u64_value(&adapter, "source_gen1_raw_records")?;
    let gen1_kept = parse_u64_value(&adapter, "source_gen1_kept_records")?;
    let gen1_dropped = parse_u64_value(&adapter, "source_gen1_dropped_duplicate_records")?;
    let gen1_within = parse_u64_value(&adapter, "source_gen1_within_source_duplicate_records")?;
    let gen1_cross = parse_u64_value(&adapter, "source_gen1_cross_source_duplicate_records")?;
    let dropped = parse_u64_value(&adapter, "dropped_duplicate_records")?;
    let within = parse_u64_value(&adapter, "within_source_duplicate_records")?;
    let cross = parse_u64_value(&adapter, "cross_source_duplicate_records")?;
    if raw_source_counts.get("gen2") != Some(&gen2_raw)
        || raw_source_counts.get("gen1") != Some(&gen1_raw)
        || gen2_raw != gen2_kept + gen2_dropped || gen2_within != gen2_dropped
        || gen1_raw != gen1_kept + gen1_dropped
        || gen1_dropped != gen1_within + gen1_cross || gen2_cross != 0 || gen1_cross != cross
        || dropped != gen2_dropped + gen1_dropped
        || within != gen2_within + gen1_within
        || dropped != within + cross
        || parse_u64_value(&adapter, "raw_records")? != gen2_raw + gen1_raw
        || parse_u64_value(&adapter, "kept_records")? != gen2_kept + gen1_kept
        || parse_u64_value(run, "raw_records")? != gen2_raw + gen1_raw
        || parse_u64_value(run, "kept_records")? != gen2_kept + gen1_kept
        || parse_u64_value(run, "dropped_duplicate_records")? != dropped
        || parse_u64_value(run, "within_source_duplicate_records")? != within
        || parse_u64_value(run, "cross_source_duplicate_records")? != cross
    {
        return Err("M6 source raw/kept/dropped counts do not reconcile with verified artifacts".into());
    }
    for key in [
        "source_gen2_raw_records", "source_gen2_kept_records", "source_gen2_dropped_duplicate_records",
        "source_gen2_within_source_duplicate_records", "source_gen1_raw_records", "source_gen1_kept_records",
        "source_gen1_dropped_duplicate_records", "source_gen1_within_source_duplicate_records",
        "source_gen2_cross_source_duplicate_records", "source_gen1_cross_source_duplicate_records",
        "source_gen2_zero_row_games", "source_gen1_zero_row_games",
        "zero_row_games",
    ] {
        expect(run, key, &adapter[key])?;
    }
    Ok(())
}

fn validate_run_manifest(values: &BTreeMap<String, String>) -> Result<(), String> {
    validate_trainer_format(values)?;
    expect(values, "bullet_commit", BULLET_COMMIT)?;
    expect(values, "architecture", "Chess768;dual-perspective;768x128-crelu;256x1-sigmoid")?;
    expect(values, "optimizer", "AdamW;decay=0.01;beta1=0.9;beta2=0.999;clip=[-1.98,1.98]")?;
    expect(values, "score_lambda", "0.7;bullet_constant_wdl=0.3;scale=400")?;
    expect(values, "batch_size", "16384")?;
    for key in ["m3_dataset_sha256", "adapter_manifest_sha256", "m3_source_tree_sha256", "m3_source_manifest_sha256", "train_sha256", "validation_sha256"] {
        parse_sha256(values.get(key).ok_or_else(|| format!("run manifest is missing {key}"))?, key)?;
    }
    Ok(())
}

fn validate_trainer_format(values: &BTreeMap<String, String>) -> Result<(), String> {
    match values.get("trainer_format").map(String::as_str) {
        Some(TRAINER_FORMAT) => Ok(()),
        Some(SANITIZED_TRAINER_FORMAT) => {
            expect(values, "bullet_patch_sha256", BULLET_PATCH_SHA256)?;
            let source_tree = values.get("bullet_source_tree_sha256")
                .ok_or_else(|| "run manifest is missing bullet_source_tree_sha256".to_string())?;
            let source_manifest = values.get("bullet_source_manifest_sha256")
                .ok_or_else(|| "run manifest is missing bullet_source_manifest_sha256".to_string())?;
            parse_sha256(source_tree, "bullet_source_tree_sha256")?;
            parse_sha256(source_manifest, "bullet_source_manifest_sha256")?;
            if source_tree != source_manifest {
                return Err("sanitized Bullet source tree and source manifest hashes differ".into());
            }
            parse_sha256(
                values.get("bullet_dependency_tree_sha256")
                    .ok_or_else(|| "run manifest is missing bullet_dependency_tree_sha256".to_string())?,
                "bullet_dependency_tree_sha256",
            )?;
            if values.get("training_record_source_path").is_none() {
                return Err("run manifest is missing training_record_source_path".into());
            }
            parse_sha256(
                values.get("training_record_source_sha256")
                    .ok_or_else(|| "run manifest is missing training_record_source_sha256".to_string())?,
                "training_record_source_sha256",
            )?;
            Ok(())
        }
        Some(found) => Err(format!("unsupported trainer_format {found}")),
        None => Err("run manifest is missing trainer_format".into()),
    }
}

#[derive(Clone, Copy, Debug)]
struct BestEpoch {
    epoch: usize,
    learning_rate: f64,
    train_mse: f64,
    validation_mse: f64,
}

#[derive(Clone, Copy, Debug)]
struct LossRow {
    learning_rate: f64,
    train_mse: f64,
    validation_mse: f64,
    best_epoch: usize,
}

fn parse_best_epoch(text: &str) -> Result<BestEpoch, String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| "losses.csv is empty".to_string())?;
    if header != "epoch,learning_rate,train_mse,validation_mse,validation_records,best_epoch,stale_epochs" {
        return Err("losses.csv header mismatch".into());
    }
    let mut footer = None;
    let mut rows = BTreeMap::<usize, LossRow>::new();
    for line in lines {
        if let Some(footer_text) = line.strip_prefix("# best_epoch=") {
            if footer.is_some() { return Err("losses.csv has multiple best-epoch footers".into()); }
            let (epoch, mse) = footer_text.split_once(" best_validation_mse=")
                .ok_or_else(|| "losses.csv best footer is malformed".to_string())?;
            let epoch = epoch.parse::<usize>().map_err(|error| format!("invalid best epoch: {error}"))?;
            let mse = mse.parse::<f64>().map_err(|error| format!("invalid footer validation MSE: {error}"))?;
            if !mse.is_finite() { return Err("best validation MSE must be finite".into()); }
            footer = Some((epoch, mse));
        } else if !line.is_empty() && !line.starts_with('#') {
            let fields: Vec<_> = line.split(',').collect();
            if fields.len() != 7 { return Err("losses.csv has a malformed row".into()); }
            let epoch = fields[0].parse::<usize>().map_err(|error| format!("invalid epoch: {error}"))?;
            let learning_rate = fields[1].parse::<f64>().map_err(|error| format!("invalid learning rate: {error}"))?;
            let train_mse = fields[2].parse::<f64>().map_err(|error| format!("invalid train MSE: {error}"))?;
            let validation_mse = fields[3].parse::<f64>().map_err(|error| format!("invalid validation MSE: {error}"))?;
            let validation_records = fields[4].parse::<usize>().map_err(|error| format!("invalid validation record count: {error}"))?;
            let row_best = fields[5].parse::<usize>().map_err(|error| format!("invalid row best epoch: {error}"))?;
            let _stale_epochs = fields[6].parse::<usize>().map_err(|error| format!("invalid stale epoch count: {error}"))?;
            if epoch == 0 || !learning_rate.is_finite() || learning_rate <= 0.0
                || !train_mse.is_finite() || train_mse < 0.0
                || !validation_mse.is_finite() || validation_mse < 0.0
                || validation_records == 0 || row_best == 0 || row_best > epoch
                || rows.insert(epoch, LossRow { learning_rate, train_mse, validation_mse, best_epoch: row_best }).is_some()
            {
                return Err("losses.csv contains invalid or duplicate epoch rows".into());
            }
        }
    }
    let (footer_epoch, footer_mse) = footer.ok_or_else(|| "losses.csv has no best-epoch footer".to_string())?;
    let (&minimum_epoch, minimum_row) = rows.iter()
        .min_by(|(left_epoch, left), (right_epoch, right)| {
            left.validation_mse.total_cmp(&right.validation_mse).then(left_epoch.cmp(right_epoch))
        })
        .ok_or_else(|| "losses.csv contains no epoch rows".to_string())?;
    if footer_epoch != minimum_epoch {
        return Err(format!("best footer selects epoch {footer_epoch}, global minimum is epoch {minimum_epoch}"));
    }
    let selected = rows.get(&footer_epoch).copied()
        .ok_or_else(|| "best epoch has no matching losses row".to_string())?;
    if (minimum_row.validation_mse - footer_mse).abs() > 5.1e-11 || selected.best_epoch != footer_epoch {
        return Err("best-epoch footer disagrees with selected losses row".into());
    }
    Ok(BestEpoch { epoch: footer_epoch, learning_rate: selected.learning_rate, train_mse: selected.train_mse, validation_mse: selected.validation_mse })
}

fn validate_checkpoint_manifest(
    checkpoint: &BTreeMap<String, String>,
    run: &BTreeMap<String, String>,
    best: BestEpoch,
) -> Result<(), String> {
    expect(checkpoint, "epoch", &best.epoch.to_string())?;
    let checkpoint_learning_rate = parse_finite(checkpoint.get("learning_rate").ok_or_else(|| "checkpoint manifest is missing learning_rate".to_string())?, "checkpoint learning rate")?;
    let checkpoint_train_mse = parse_finite(checkpoint.get("train_mse").ok_or_else(|| "checkpoint manifest is missing train_mse".to_string())?, "checkpoint train MSE")?;
    let checkpoint_validation_mse = parse_finite(checkpoint.get("validation_mse").ok_or_else(|| "checkpoint manifest is missing validation_mse".to_string())?, "checkpoint validation MSE")?;
    if (checkpoint_learning_rate - best.learning_rate).abs() > 5.1e-11
        || (checkpoint_train_mse - best.train_mse).abs() > 5.1e-11
        || (checkpoint_validation_mse - best.validation_mse).abs() > 5.1e-11
    {
        return Err("checkpoint learning rate or losses disagree with selected CSV row".into());
    }
    for key in ["train_sha256", "validation_sha256"] {
        expect(checkpoint, key, run.get(key).ok_or_else(|| format!("run manifest is missing {key}"))?)?;
    }
    Ok(())
}

fn build_float_network(bytes: &[u8]) -> Result<NnueNetwork<HIDDEN>, String> {
    let tensors = parse_weights(bytes)?;
    build_float_network_from_tensors(&tensors)
}

fn build_float_network_from_tensors(tensors: &BTreeMap<String, Vec<f32>>) -> Result<NnueNetwork<HIDDEN>, String> {
    let l0b = &tensors["l0b"];
    let l0w = &tensors["l0w"];
    let l1b = &tensors["l1b"];
    let l1w = &tensors["l1w"];
    let mut feature_weights = vec![0.0f32; 768 * HIDDEN];
    for bullet_feature in 0..768 {
        let kira_feature = bullet_feature_to_kira(bullet_feature)
            .ok_or_else(|| format!("cannot map Bullet feature {bullet_feature}"))?;
        for hidden in 0..HIDDEN {
            // Bullet serializes l0w's (hidden, input) tensor column-major.
            feature_weights[kira_feature * HIDDEN + hidden] = l0w[bullet_feature * HIDDEN + hidden];
        }
    }
    let output_weights: [[f32; HIDDEN]; 2] = [
        l1w[..HIDDEN].to_vec().try_into().map_err(|_| "STM output head size mismatch")?,
        l1w[HIDDEN..].to_vec().try_into().map_err(|_| "NSTM output head size mismatch")?,
    ];
    NnueNetwork::from_parts(
        feature_weights,
        l0b.clone().try_into().map_err(|_| "hidden bias size mismatch")?,
        output_weights,
        l1b[0],
        400.0,
    )
}

fn parse_weights(bytes: &[u8]) -> Result<BTreeMap<String, Vec<f32>>, String> {
    const EXPECTED: [(&str, usize); 4] = [
        ("l0b", HIDDEN),
        ("l0w", HIDDEN * 768),
        ("l1b", 1),
        ("l1w", 2 * HIDDEN),
    ];
    if usize::BITS != 64 { return Err("Bullet weights.bin requires a 64-bit host".into()); }
    let mut offset = 0usize;
    let mut tensors = BTreeMap::new();
    let mut order = Vec::new();
    while offset < bytes.len() {
        let newline = bytes[offset..].iter().position(|&byte| byte == b'\n')
            .ok_or_else(|| "weights.bin tensor name has no newline".to_string())? + offset;
        if newline == offset { return Err("weights.bin contains an empty tensor name".into()); }
        let name = std::str::from_utf8(&bytes[offset..newline])
            .map_err(|error| format!("weights.bin tensor name is invalid UTF-8: {error}"))?.to_owned();
        if !name.is_ascii() { return Err("weights.bin tensor names must be ASCII".into()); }
        offset = newline + 1;
        let len_end = offset.checked_add(8).ok_or_else(|| "weights.bin offset overflow".to_string())?;
        if len_end > bytes.len() { return Err("weights.bin tensor count is truncated".into()); }
        let count = usize::try_from(u64::from_le_bytes(bytes[offset..len_end].try_into().unwrap()))
            .map_err(|_| "weights.bin tensor count exceeds usize")?;
        let expected = EXPECTED.iter().find(|(expected, _)| *expected == name)
            .map(|(_, count)| *count).ok_or_else(|| format!("weights.bin has unexpected tensor {name}"))?;
        if count != expected { return Err(format!("tensor {name} must have {expected} floats, found {count}")); }
        offset = len_end;
        let byte_count = count.checked_mul(4).ok_or_else(|| "weights.bin byte count overflow".to_string())?;
        let end = offset.checked_add(byte_count).ok_or_else(|| "weights.bin offset overflow".to_string())?;
        if end > bytes.len() { return Err(format!("tensor {name} is truncated")); }
        let values = bytes[offset..end].chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap())).collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) { return Err(format!("tensor {name} has non-finite values")); }
        if tensors.insert(name.clone(), values).is_some() { return Err(format!("weights.bin duplicates tensor {name}")); }
        order.push(name);
        offset = end;
    }
    let expected_order: Vec<_> = EXPECTED.iter().map(|(name, _)| (*name).to_string()).collect();
    if order != expected_order { return Err(format!("weights.bin tensor order mismatch: {order:?}")); }
    Ok(tensors)
}

fn parse_key_values(text: &str, label: &str) -> Result<BTreeMap<String, String>, String> {
    let mut values = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') { continue; }
        let (key, value) = line.split_once('=').ok_or_else(|| format!("{label} line {} is malformed", index + 1))?;
        if key.is_empty() || value.is_empty() || values.insert(key.to_string(), value.to_string()).is_some() {
            return Err(format!("{label} has an empty or duplicate key at line {}", index + 1));
        }
    }
    Ok(values)
}

fn expect(values: &BTreeMap<String, String>, key: &str, expected: &str) -> Result<(), String> {
    let found = values.get(key).ok_or_else(|| format!("manifest is missing {key}"))?;
    if found != expected { return Err(format!("manifest {key} mismatch: expected {expected}, found {found}")); }
    Ok(())
}

fn parse_u64_value(values: &BTreeMap<String, String>, key: &str) -> Result<u64, String> {
    values.get(key)
        .ok_or_else(|| format!("manifest is missing {key}"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid {key}: {error}"))
}

fn parse_finite(value: &str, label: &str) -> Result<f64, String> {
    let parsed = value.parse::<f64>().map_err(|error| format!("invalid {label}: {error}"))?;
    if !parsed.is_finite() {
        return Err(format!("{label} must be finite"));
    }
    Ok(parsed)
}

fn parse_sha256(value: &str, label: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} must contain 64 hex digits"));
    }
    let mut result = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        result[index] = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16)
            .map_err(|error| format!("invalid {label}: {error}"))?;
    }
    Ok(result)
}

fn io_error(error: std::io::Error) -> String { error.to_string() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_sampling_is_seeded_distinct_and_reproducible() {
        fn sample(seed: u64) -> Vec<u64> {
            let mut reservoir = Reservoir::new(100, seed).unwrap();
            for value in 0..10_000u64 { reservoir.push(value).unwrap(); }
            assert_eq!(reservoir.seen, 10_000);
            reservoir.items
        }
        let first = sample(DEFAULT_SEED);
        let second = sample(DEFAULT_SEED);
        let other = sample(DEFAULT_SEED + 1);
        assert_eq!(first, second);
        assert_ne!(first, other);
        assert_eq!(first.iter().copied().collect::<HashSet<_>>().len(), first.len());
        assert!(first.iter().all(|&value| value < 10_000));
    }

    #[test]
    fn default_seed_is_stable() {
        let mut first = SplitMix64::new(DEFAULT_SEED);
        let mut second = SplitMix64::new(DEFAULT_SEED);
        assert_eq!((0..10).map(|_| first.next()).collect::<Vec<_>>(), (0..10).map(|_| second.next()).collect::<Vec<_>>());
    }

    #[test]
    fn accepts_sanitized_trainer_format_only_with_pinned_source_provenance() {
        let mut values = BTreeMap::from([
            ("trainer_format".to_string(), SANITIZED_TRAINER_FORMAT.to_string()),
            ("bullet_patch_sha256".to_string(), BULLET_PATCH_SHA256.to_string()),
            ("bullet_source_tree_sha256".to_string(), "a".repeat(64)),
            ("bullet_source_manifest_sha256".to_string(), "a".repeat(64)),
            ("bullet_dependency_tree_sha256".to_string(), "b".repeat(64)),
            ("training_record_source_path".to_string(), "/source/tools/train/src/record.rs".to_string()),
            ("training_record_source_sha256".to_string(), "c".repeat(64)),
        ]);
        validate_trainer_format(&values).unwrap();

        values.insert("bullet_source_manifest_sha256".to_string(), "c".repeat(64));
        assert!(validate_trainer_format(&values).unwrap_err().contains("hashes differ"));
        values.insert("bullet_source_manifest_sha256".to_string(), "a".repeat(64));
        values.insert("bullet_patch_sha256".to_string(), "0".repeat(64));
        assert!(validate_trainer_format(&values).unwrap_err().contains("bullet_patch_sha256 mismatch"));
    }

    #[test]
    fn best_epoch_must_be_global_minimum_and_metrics_are_returned() {
        let losses = concat!(
            "epoch,learning_rate,train_mse,validation_mse,validation_records,best_epoch,stale_epochs\n",
            "1,1.0e-3,0.20,0.30,20,1,0\n",
            "2,9.0e-4,0.10,0.10,20,2,0\n",
            "3,8.0e-4,0.09,0.15,20,2,1\n",
            "# best_epoch=2 best_validation_mse=0.1000000000\n",
        );
        let best = parse_best_epoch(losses).unwrap();
        assert_eq!(best.epoch, 2);
        assert_eq!(best.learning_rate, 9.0e-4);
        assert_eq!(best.train_mse, 0.10);
        assert_eq!(best.validation_mse, 0.10);
        let wrong_footer = losses.replace("# best_epoch=2 best_validation_mse=0.1000000000", "# best_epoch=1 best_validation_mse=0.3000000000");
        assert!(parse_best_epoch(&wrong_footer).unwrap_err().contains("global minimum"));
    }

    #[test]
    fn checkpoint_metrics_must_match_the_selected_loss_row() {
        let run = BTreeMap::from([
            ("train_sha256".to_string(), "a".repeat(64)),
            ("validation_sha256".to_string(), "b".repeat(64)),
        ]);
        let checkpoint = BTreeMap::from([
            ("epoch".to_string(), "6".to_string()),
            ("learning_rate".to_string(), "7.9248292604e-4".to_string()),
            ("train_mse".to_string(), "0.013175333091".to_string()),
            ("validation_mse".to_string(), "0.014813416472".to_string()),
            ("train_sha256".to_string(), "a".repeat(64)),
            ("validation_sha256".to_string(), "b".repeat(64)),
        ]);
        let best = BestEpoch {
            epoch: 6,
            learning_rate: 7.9248292604e-4,
            train_mse: 0.0131753331,
            validation_mse: 0.0148134165,
        };
        validate_checkpoint_manifest(&checkpoint, &run, best).unwrap();
        let mut bad = checkpoint;
        bad.insert("validation_mse".into(), "0.2".into());
        assert!(validate_checkpoint_manifest(&bad, &run, best).unwrap_err().contains("disagree"));
    }

    #[test]
    fn float_checkpoint_l0_reads_bullet_column_major_feature_rows() {
        let mut tensors = BTreeMap::new();
        tensors.insert("l0b".into(), vec![0.0; HIDDEN]);
        let mut l0w = vec![0.0; HIDDEN * 768];
        let bullet_bishop = 2 * 64 + 7;
        l0w[bullet_bishop * HIDDEN + 9] = 0.375;
        tensors.insert("l0w".into(), l0w);
        tensors.insert("l1b".into(), vec![0.0]);
        tensors.insert("l1w".into(), vec![0.0; 2 * HIDDEN]);

        let network = build_float_network_from_tensors(&tensors).unwrap();
        let kira_bishop = 3 * 64 + 7;
        assert_eq!(network.feature_weights()[kira_bishop * HIDDEN + 9], 0.375);
        assert_eq!(network.feature_weights()[9 * HIDDEN + bullet_bishop], 0.0);
    }
}
