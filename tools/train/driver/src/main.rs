use bullet_compiler::{ir::NodeId, tensor::TValue};
use bullet_gpu::{
    buffer::Buffer,
    function::Function,
    runtime::{Device, Gpu, Stream},
};
use bullet_lib::{game::inputs::SparseInputType, nn::ExecutionContext};
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, TensorMap},
    optimiser::adam::{AdamW, AdamWParams},
    reader::{FixedSizeData, FixedSizeDataReader, ReadMapLoader},
    run::{self, HostPool, Step, TrainingSteps, train},
};
#[path = "../../src/record.rs"]
mod record;
use record::{TrainingRecord, decode_training_record, encode_training_record};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
};

const PINNED_BULLET_COMMIT: &str = "579b78b37a694cacd6e9ede4706386c15c34837c";
const PINNED_BULLET_PATCH_SHA256: &str = "ecf8ce81486d5e2d3cfbd881bbd053e8b3682b8de4acf24a6824310cb29b5141";
const RECORD_BYTES: u64 = 32;
const BATCH_SIZE: usize = 16_384;
const HIDDEN: usize = 128;
const SEED_DEFAULT: u64 = 0x4b49_5241_4d34_5452;
const SCORE_LAMBDA: f32 = 0.7;
const BULLET_WDL: f32 = 0.3;
const SCALE: f32 = 400.0;
const INITIAL_LR: f32 = 1.0e-3;
const FINAL_LR: f32 = 1.0e-5;
const MIN_IMPROVEMENT: f64 = 1.0e-7;
const MULTI_SPLIT_DOMAIN: u64 = 0x4b31_4144_4154_4132;
const GEN2_SPLIT_TAG: u64 = 0x4745_4e32_0000_0002;
const GEN1_SPLIT_TAG: u64 = 0x4745_4e31_0000_0001;
const PINNED_GEN2_NNUE_SHA256: &str = "842e3870cdb02aa762b7d1f4ab0ec441c9df9b6de911395ba759ee1840cefca9";
const NNUE_FEATURE_KEY_SCHEMA: &str = "two-perspective-active-feature-set-v1;768-each;stm-first;feature_index-v1;ignore-castling-ep-clocks";

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
struct TrainRecord(TrainingRecord);

// Safety: TrainingRecord is a repr(C), 32-byte struct containing only integer
// fields; the transparent wrapper has the same valid bit patterns and layout.
unsafe impl FixedSizeData for TrainRecord {}

#[derive(Clone, Copy, Debug, Default)]
struct KiraChess768;

impl SparseInputType for KiraChess768 {
    type RequiredDataType = TrainingRecord;

    fn num_inputs(&self) -> usize { 768 }

    fn max_active(&self) -> usize { 32 }

    fn map_features<F: FnMut(usize, usize)>(&self, record: &TrainingRecord, visit: F) {
        record.map_chess768(visit);
    }

    fn shorthand(&self) -> String { "768".to_string() }

    fn description(&self) -> String { "Kira-owned dual-perspective chess inputs".to_string() }
}

#[derive(Clone, Debug)]
struct Config {
    data_dir: PathBuf,
    output_dir: PathBuf,
    max_epochs: usize,
    patience: usize,
    seed: u64,
    loader_threads: u8,
    validation_batch_size: usize,
    validate_only: bool,
}

#[derive(Clone, Debug)]
struct DataInfo {
    dataset_mode: String,
    train_records: usize,
    validation_records: usize,
    train_games: usize,
    validation_games: usize,
    train_sha256: String,
    validation_sha256: String,
    adapter_manifest_sha256: String,
    m3_dataset_sha256: String,
    m3_config_sha256: String,
    m3_source_tree_sha256: String,
    m3_source_manifest_sha256: String,
    m3_source_revision: String,
    source_provenance: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
struct RuntimeInfo {
    bullet_root: String,
    bullet_commit: String,
    bullet_lock_sha256: String,
    bullet_patch_sha256: String,
    bullet_source_tree_sha256: String,
    bullet_source_manifest_sha256: String,
    bullet_dependency_tree_sha256: String,
    driver_source_path: String,
    driver_source_sha256: String,
    record_source_path: String,
    record_source_sha256: String,
    runner_path: String,
    runner_sha256: String,
    runtime_manifest_sha256: String,
    runtime_lock_sha256: String,
}

fn has_help_flag() -> bool {
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => return true,
            "--data-dir"
            | "--out"
            | "--epochs"
            | "--patience"
            | "--seed"
            | "--threads"
            | "--validation-batch-size" => {
                let _ = args.next();
            }
            _ => {}
        }
    }
    false
}

fn main() {
    if has_help_flag() {
        println!("Usage: kira-m4-trainer --data-dir ADAPTER_OUTPUT --out NEW_DIRECTORY [--epochs N] [--patience N] [--seed U64] [--threads N] [--validate-only]");
        println!("Requires approved M3 format-v2 adapter output. Batch size is fixed at {BATCH_SIZE}.");
        return;
    }
    if let Err(error) = run() {
        eprintln!("kira-m4-trainer: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let config = parse_args()?;
    let data = validate_dataset(&config.data_dir)?;
    let runtime = collect_runtime_info()?;
    if config.validate_only {
        print_runtime_info(&runtime);
    }
    let model = build_model();
    let device = Device::<ExecutionContext>::new(0).map_err(|error| format!("create Bullet device: {error}"))?;
    let weights = ModelWeights::new(&model, config.seed);
    let mut evaluator = BatchEvaluator::new(&model, device.clone(), config.validation_batch_size)?;
    let mut optimiser = AdamW::new(
        model,
        weights,
        device.clone(),
        AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -1.98, max_weight: 1.98 },
    )
    .map_err(|error| format!("create Bullet AdamW optimiser: {error}"))?;
    let mapper = build_mapper();
    let validation_mse = evaluate_validation(
        &config.data_dir.join("validation.bullet"),
        data.validation_records,
        &mapper,
        &mut evaluator,
        optimiser.weights(),
        &device,
        &config,
    )?;

    if config.validate_only {
        println!("validation_records={}", data.validation_records);
        println!("validation_mse={validation_mse:.10}");
        println!("float_model=deterministic_initialization_seed_0x{:016x}", config.seed);
        return Ok(());
    }

    if config.output_dir.exists() {
        return Err(format!("refusing to overwrite output directory {}", config.output_dir.display()));
    }
    fs::create_dir(&config.output_dir).map_err(|error| format!("create output directory: {error}"))?;
    write_run_manifest(&config, &data, &runtime, &config.output_dir)?;
    let losses_path = config.output_dir.join("losses.csv");
    let mut losses = BufWriter::new(File::create(&losses_path).map_err(io_error)?);
    writeln!(losses, "epoch,learning_rate,train_mse,validation_mse,validation_records,best_epoch,stale_epochs")
        .map_err(io_error)?;

    let steps = training_epoch_steps(data.train_records)?;
    let batches_per_epoch = steps.batches_per_superbatch;
    let reader_path = config.data_dir.join("train.bullet");
    let reader_path_string = reader_path
        .to_str()
        .ok_or_else(|| "training shard path is not valid UTF-8".to_string())?
        .to_owned();
    let reader = FixedSizeDataReader::<TrainRecord>::new(&[reader_path_string.as_str()]);
    let mut best = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut stale_epochs = 0usize;

    for epoch in 1..=config.max_epochs {
        let lr = epoch_learning_rate(epoch, config.max_epochs);
        let schedule = run::TrainingSchedule {
            steps,
            log_rate: batches_per_epoch.max(1),
            lr_schedule: Box::new(move |_| lr),
            track_grad_norms: false,
        };
        let dataloader = ReadMapLoader::new(reader.clone(), mapper.clone(), config.loader_threads);
        let mut train_loss = 0.0f64;
        let mut observed_batches = 0usize;
        train(
            &mut optimiser,
            schedule,
            dataloader,
            |_, _, batch_loss| {
                train_loss += f64::from(batch_loss);
                observed_batches += 1;
            },
            |_, _| {},
        )
        .map_err(|error| format!("Bullet training failed on epoch {epoch}: {error:?}"))?;
        if observed_batches != batches_per_epoch {
            return Err(format!(
                "Bullet returned {observed_batches} train batches for epoch {epoch}; expected {batches_per_epoch}"
            ));
        }
        train_loss /= observed_batches as f64;
        let validation_mse = evaluate_validation(
            &config.data_dir.join("validation.bullet"),
            data.validation_records,
            &mapper,
            &mut evaluator,
            optimiser.weights(),
            &device,
            &config,
        )?;

        let improved = !best.is_finite() || validation_mse + MIN_IMPROVEMENT < best;
        if improved {
            best = validation_mse;
            best_epoch = epoch;
            stale_epochs = 0;
        } else {
            stale_epochs += 1;
        }

        let epoch_dir = config.output_dir.join(format!("checkpoint-{epoch:04}"));
        fs::create_dir(&epoch_dir).map_err(io_error)?;
        let epoch_dir_string = epoch_dir
            .to_str()
            .ok_or_else(|| "checkpoint path is not valid UTF-8".to_string())?;
        optimiser
            .write_to_checkpoint(epoch_dir_string)
            .map_err(|error| format!("write epoch {epoch} checkpoint: {error}"))?;
        write_checkpoint_manifest(&epoch_dir.join("manifest.txt"), epoch, lr, train_loss, validation_mse, &data)?;

        writeln!(
            losses,
            "{epoch},{lr:.10e},{train_loss:.10},{validation_mse:.10},{},{best_epoch},{stale_epochs}",
            data.validation_records
        )
        .map_err(io_error)?;
        losses.flush().map_err(io_error)?;
        losses.get_ref().sync_all().map_err(io_error)?;
        println!(
            "epoch={epoch} lr={lr:.6e} train_mse={train_loss:.10} validation_mse={validation_mse:.10} best_epoch={best_epoch} stale={stale_epochs}"
        );

        if stale_epochs >= config.patience {
            println!("early_stop=validation_plateau patience={}", config.patience);
            break;
        }
    }

    writeln!(losses, "# best_epoch={best_epoch} best_validation_mse={best:.10}").map_err(io_error)?;
    losses.flush().map_err(io_error)?;
    losses.get_ref().sync_all().map_err(io_error)?;
    println!("best_epoch={best_epoch}");
    println!("best_validation_mse={best:.10}");
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut data_dir = None;
    let mut output_dir = None;
    let mut max_epochs = 100usize;
    let mut patience = 5usize;
    let mut seed = SEED_DEFAULT;
    let mut loader_threads = 4u8;
    let mut validation_batch_size = BATCH_SIZE;
    let mut validate_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, name: &str| {
            args.next().ok_or_else(|| format!("{name} requires a value"))
        };
        match flag.as_str() {
            "--data-dir" => data_dir = Some(PathBuf::from(value(&mut args, "--data-dir")?)),
            "--out" => {
                let path = value(&mut args, "--out")?;
                if path.is_empty() || path == "--help" || path == "-h" || path.starts_with("--") {
                    return Err("--out value cannot be empty, help, or another option".into());
                }
                output_dir = Some(PathBuf::from(path));
            }
            "--epochs" => max_epochs = parse_positive(&value(&mut args, "--epochs")?, "--epochs")?,
            "--patience" => patience = parse_positive(&value(&mut args, "--patience")?, "--patience")?,
            "--seed" => seed = parse_seed(&value(&mut args, "--seed")?)?,
            "--threads" => {
                let parsed = parse_positive(&value(&mut args, "--threads")?, "--threads")?;
                loader_threads = u8::try_from(parsed).map_err(|_| "--threads must be in 1..=255")?;
            }
            "--validation-batch-size" => {
                validation_batch_size = parse_positive(&value(&mut args, "--validation-batch-size")?, "--validation-batch-size")?;
            }
            "--validate-only" => validate_only = true,
            other => return Err(format!("unknown argument {other}; use --help")),
        }
    }
    let data_dir = data_dir.ok_or_else(|| "--data-dir is required".to_string())?;
    let output_dir = output_dir.unwrap_or_else(|| PathBuf::from("m4-training-run"));
    if max_epochs < 2 {
        return Err("--epochs must be at least 2 for a decaying learning-rate schedule".into());
    }
    if !validate_only && patience == 0 {
        return Err("--patience must be positive".into());
    }
    if validation_batch_size == 0 {
        return Err("validation batch size must be positive".into());
    }
    Ok(Config { data_dir, output_dir, max_epochs, patience, seed, loader_threads, validation_batch_size, validate_only })
}

fn parse_positive(value: &str, flag: &str) -> Result<usize, String> {
    let parsed = value.parse::<usize>().map_err(|error| format!("invalid {flag}: {error}"))?;
    if parsed == 0 { Err(format!("{flag} must be positive")) } else { Ok(parsed) }
}

fn parse_seed(value: &str) -> Result<u64, String> {
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|error| format!("invalid hexadecimal seed: {error}"))
    } else {
        value.parse::<u64>().map_err(|error| format!("invalid decimal seed: {error}"))
    }
}

fn build_model() -> ModelDefinition {
    let inputs = ModelInputs::default()
        .add_sparse("stm", (768, 1), 32)
        .add_sparse("nstm", (768, 1), 32)
        .add_dense("targets", (1, 1));
    let mut definition = ModelDefinition::build(
        &inputs,
        |builder, ((stm, nstm), targets)| {
            let l0 = builder.new_affine("l0", 768, HIDDEN);
            let l1 = builder.new_affine("l1", 2 * HIDDEN, 1);
            let stm_hidden = l0.forward(stm).crelu();
            let nstm_hidden = l0.forward(nstm).crelu();
            let logits = l1.forward(stm_hidden.concat(nstm_hidden));
            let loss = logits.sigmoid().squared_error(targets).reduce_sum_batch();
            (Some(loss), vec![("output".to_string(), logits)])
        },
    );
    definition.debug = false;
    definition
}

fn build_mapper() -> ModelInputsMapper<TrainRecord> {
    let inputs = ModelInputs::default()
        .add_sparse("stm", (768, 1), 32)
        .add_sparse("nstm", (768, 1), 32)
        .add_dense("targets", (1, 1));
    ModelInputsMapper::build(&inputs, |record: &TrainRecord, _step, ((stm, nstm), target)| {
        stm.fill(-1);
        nstm.fill(-1);
        let mut count = 0usize;
        KiraChess768.map_features(&record.0, |stm_feature, nstm_feature| {
            assert!(count < 32, "Kira record has more than 32 pieces");
            stm[count] = stm_feature as i32;
            nstm[count] = nstm_feature as i32;
            count += 1;
        });
        target[0] = blended_target(&record.0);
    })
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn blended_target(board: &TrainingRecord) -> f32 {
    let score_probability = sigmoid(f32::from(board.score) / SCALE);
    let result = f32::from(board.result) / 2.0;
    SCORE_LAMBDA * score_probability + BULLET_WDL * result
}

fn epoch_learning_rate(epoch: usize, max_epochs: usize) -> f32 {
    let t = (epoch - 1) as f32 / (max_epochs - 1) as f32;
    INITIAL_LR * (FINAL_LR / INITIAL_LR).powf(t)
}

fn training_epoch_steps(train_records: usize) -> Result<TrainingSteps, String> {
    if train_records == 0 {
        return Err("training shard contains no records".into());
    }
    Ok(TrainingSteps {
        batch_size: BATCH_SIZE,
        batches_per_superbatch: train_records.div_ceil(BATCH_SIZE),
        start_superbatch: 1,
        end_superbatch: 1,
    })
}

fn validate_dataset(data_dir: &Path) -> Result<DataInfo, String> {
    if !data_dir.is_dir() {
        return Err(format!("adapter data directory does not exist: {}", data_dir.display()));
    }
    let manifest = read_key_values(&data_dir.join("manifest.txt"))?;
    match manifest.get("adapter_format").map(String::as_str) {
        Some("kira-m4-record-v2") => validate_dataset_m4(data_dir),
        Some("kira-m6-multisource-record-v3") => validate_dataset_m6(data_dir, &manifest),
        Some(other) => Err(format!("unsupported adapter format {other}")),
        None => Err("adapter manifest is missing adapter_format".into()),
    }
}

fn validate_dataset_m4(data_dir: &Path) -> Result<DataInfo, String> {
    if !data_dir.is_dir() {
        return Err(format!("adapter data directory does not exist: {}", data_dir.display()));
    }
    let manifest = read_key_values(&data_dir.join("manifest.txt"))?;
    expect(&manifest, "adapter_format", "kira-m4-record-v2")?;
    expect(&manifest, "source_format", "K1ADATAS-v2")?;
    expect(&manifest, "bullet_git_commit", PINNED_BULLET_COMMIT)?;
    expect(&manifest, "record_format", "KiraPackedTrainingRecord-v1;size=32;little-endian")?;
    expect(&manifest, "kira_score_lambda", "0.7")?;
    expect(&manifest, "bullet_constant_wdl", "0.3")?;
    expect(&manifest, "eval_scale", "400")?;
    if !manifest.get("source_sha256").is_some_and(|value| is_sha256(value)) {
        return Err("adapter manifest has no valid source_sha256".into());
    }
    let train_path = data_dir.join("train.bullet");
    let validation_path = data_dir.join("validation.bullet");
    let train_len = file_len(&train_path)?;
    let validation_len = file_len(&validation_path)?;
    if train_len == 0 || validation_len == 0 || train_len % RECORD_BYTES != 0 || validation_len % RECORD_BYTES != 0 {
        return Err("train and validation shards must be non-empty multiples of 32 bytes".into());
    }
    let train_sha = sha256(&train_path)?;
    let validation_sha = sha256(&validation_path)?;
    expect(&manifest, "train_sha256", &train_sha)?;
    expect(&manifest, "validation_sha256", &validation_sha)?;
    for (sidecar, sha_key) in [
        ("train_game_ids.tsv", "train_game_ids_sha256"),
        ("validation_game_ids.tsv", "validation_game_ids_sha256"),
    ] {
        expect(&manifest, sha_key, &sha256(&data_dir.join(sidecar))?)?;
    }
    let train_records = usize::try_from(train_len / RECORD_BYTES).map_err(|_| "train record count exceeds usize")?;
    let validation_records = usize::try_from(validation_len / RECORD_BYTES).map_err(|_| "validation record count exceeds usize")?;
    if parse_usize(&manifest, "train_records")? != train_records || parse_usize(&manifest, "validation_records")? != validation_records {
        return Err("adapter manifest record counts disagree with shard sizes".into());
    }
    let train_games = validate_sidecar(&data_dir.join("train_game_ids.tsv"), train_records)?;
    let validation_games = validate_sidecar(&data_dir.join("validation_game_ids.tsv"), validation_records)?;
    let train_ids = sidecar_game_ids(&data_dir.join("train_game_ids.tsv"))?;
    let validation_ids = sidecar_game_ids(&data_dir.join("validation_game_ids.tsv"))?;
    if !train_ids.is_disjoint(&validation_ids) {
        return Err("a game ID appears in both train and validation sidecars".into());
    }
    if train_games != parse_usize(&manifest, "train_games")? || validation_games != parse_usize(&manifest, "validation_games")? {
        return Err("adapter manifest game counts disagree with game-ID sidecars".into());
    }
    let source_games = parse_usize(&manifest, "source_games")?;
    if train_games + validation_games != source_games || validation_games != source_games * 2 / 100 {
        return Err("game sidecars do not match the adapter's exact floor(2%) split".into());
    }
    expect(&manifest, "validation_percent", "2")?;
    let m3_dataset_sha256 = manifest.get("source_sha256").cloned().unwrap();
    let m3_config_sha256 = manifest.get("source_config_sha256").ok_or_else(|| "adapter manifest is missing source_config_sha256".to_string())?.clone();
    let m3_source_tree_sha256 = manifest.get("kira_source_tree_sha256").ok_or_else(|| "adapter manifest is missing Kira source tree hash".to_string())?.clone();
    let m3_source_manifest_sha256 = manifest.get("m3_source_manifest_sha256").ok_or_else(|| "adapter manifest is missing M3 manifest hash".to_string())?.clone();
    for (name, hash) in [
        ("source_config_sha256", m3_config_sha256.as_str()),
        ("kira_source_tree_sha256", m3_source_tree_sha256.as_str()),
        ("m3_source_manifest_sha256", m3_source_manifest_sha256.as_str()),
    ] {
        if !is_sha256(hash) { return Err(format!("adapter manifest field {name} is not a SHA-256")); }
    }
    let m3_source_revision = manifest.get("kira_source_revision").cloned().ok_or_else(|| "adapter manifest is missing Kira source revision field".to_string())?;
    Ok(DataInfo {
        dataset_mode: "single-source-v1".into(),
        train_records,
        validation_records,
        train_games,
        validation_games,
        train_sha256: train_sha,
        validation_sha256: validation_sha,
        adapter_manifest_sha256: sha256(&data_dir.join("manifest.txt"))?,
        m3_dataset_sha256,
        m3_config_sha256,
        m3_source_tree_sha256,
        m3_source_manifest_sha256,
        m3_source_revision,
        source_provenance: BTreeMap::new(),
    })
}

fn validate_dataset_m6(data_dir: &Path, manifest: &BTreeMap<String, String>) -> Result<DataInfo, String> {
    expect(manifest, "adapter_format", "kira-m6-multisource-record-v3")?;
    expect(manifest, "source_format", "K1ADATAS-v2")?;
    expect(manifest, "source_precedence", "gen2>gen1;first-occurrence-within-source;duplicate_key=NNUE-active-feature-key-v1")?;
    expect(manifest, "nnue_feature_key_schema", NNUE_FEATURE_KEY_SCHEMA)?;
    expect(manifest, "nnue_feature_key_encoding", "sparse-sorted-10-bit-indices;stm-perspective-first;max-32-pieces-per-perspective")?;
    expect(manifest, "bullet_git_commit", PINNED_BULLET_COMMIT)?;
    expect(manifest, "record_format", "KiraPackedTrainingRecord-v1;size=32;little-endian")?;
    expect(manifest, "kira_score_lambda", "0.7")?;
    expect(manifest, "bullet_constant_wdl", "0.3")?;
    expect(manifest, "eval_scale", "400")?;
    expect(manifest, "source_gen2_search_evaluator", "QuantizedNnueNetwork<128>")?;
    expect(manifest, "source_gen2_nnue_sha256", PINNED_GEN2_NNUE_SHA256)?;
    expect(manifest, "source_gen2_evaluator", "Kira-HCE")?;
    expect(manifest, "source_gen1_evaluator", "Kira-HCE")?;
    for key in [
        "source_gen2_sha256", "source_gen1_sha256",
        "source_gen2_config_sha256", "source_gen1_config_sha256",
        "source_gen2_source_tree_sha256", "source_gen1_source_tree_sha256",
        "source_gen2_manifest_sha256", "source_gen1_manifest_sha256",
        "source_gen2_engine_sha256", "source_gen1_engine_sha256",
        "source_pair_sha256", "train_sha256", "validation_sha256",
        "train_game_ids_sha256", "validation_game_ids_sha256",
    ] {
        let value = manifest.get(key).ok_or_else(|| format!("M6 adapter manifest is missing {key}"))?;
        if !is_sha256(value) { return Err(format!("M6 adapter manifest field {key} is not a SHA-256")); }
    }
    for source in ["gen2", "gen1"] {
        let revision = manifest.get(&format!("source_{source}_revision"))
            .ok_or_else(|| format!("M6 adapter manifest is missing source_{source}_revision"))?;
        if revision.is_empty() || revision.contains(['\t', '\n', '\r']) {
            return Err(format!("M6 source {source} revision is malformed"));
        }
    }
    if manifest.get("source_gen1_search_evaluator").is_some()
        || manifest.get("source_gen1_nnue_sha256").is_some()
    {
        return Err("M6 gen1 source unexpectedly carries an NNUE search evaluator".into());
    }

    let sources_path = data_dir.join("sources.tsv");
    expect(manifest, "source_pair_sha256", &sha256(&sources_path)?)?;
    validate_m6_sources_tsv(&sources_path, manifest)?;

    let train_path = data_dir.join("train.bullet");
    let validation_path = data_dir.join("validation.bullet");
    let train_len = file_len(&train_path)?;
    let validation_len = file_len(&validation_path)?;
    if train_len == 0 || validation_len == 0 || train_len % RECORD_BYTES != 0 || validation_len % RECORD_BYTES != 0 {
        return Err("M6 train and validation shards must be non-empty multiples of 32 bytes".into());
    }
    let train_records = usize::try_from(train_len / RECORD_BYTES).map_err(|_| "train record count exceeds usize")?;
    let validation_records = usize::try_from(validation_len / RECORD_BYTES).map_err(|_| "validation record count exceeds usize")?;
    expect(manifest, "train_sha256", &sha256(&train_path)?)?;
    expect(manifest, "validation_sha256", &sha256(&validation_path)?)?;
    expect(manifest, "train_game_ids_sha256", &sha256(&data_dir.join("train_game_ids.tsv"))?)?;
    expect(manifest, "validation_game_ids_sha256", &sha256(&data_dir.join("validation_game_ids.tsv"))?)?;
    if parse_usize(manifest, "train_records")? != train_records
        || parse_usize(manifest, "validation_records")? != validation_records
    {
        return Err("M6 adapter manifest record counts disagree with shard sizes".into());
    }

    let (train_games, train_ids, train_zero_games) = validate_multi_sidecar(&data_dir.join("train_game_ids.tsv"), train_records, true)?;
    let (validation_games, validation_ids, validation_zero_games) = validate_multi_sidecar(&data_dir.join("validation_game_ids.tsv"), validation_records, false)?;
    if !train_ids.is_disjoint(&validation_ids) {
        return Err("a composite source/game identity appears in both M6 train and validation sidecars".into());
    }
    if train_games != parse_usize(manifest, "train_games")? || validation_games != parse_usize(manifest, "validation_games")? {
        return Err("M6 adapter manifest game counts disagree with composite game sidecars".into());
    }
    let all_ids: BTreeSet<_> = train_ids.union(&validation_ids).cloned().collect();
    let source_games = parse_usize(manifest, "source_games")?;
    let component_games = parse_usize(manifest, "source_gen2_games")?
        .checked_add(parse_usize(manifest, "source_gen1_games")?)
        .ok_or_else(|| "M6 source game count overflow".to_string())?;
    if all_ids.len() != source_games || train_games + validation_games != source_games || component_games != source_games {
        return Err("M6 composite game sidecars disagree with source game count".into());
    }
    for (source, manifest_key) in [("gen2", "source_gen2_games"), ("gen1", "source_gen1_games")] {
        let sidecar_count = all_ids.iter().filter(|identity| identity.0 == source).count();
        if sidecar_count != parse_usize(manifest, manifest_key)? {
            return Err(format!("M6 per-source game count for {source} disagrees with composite sidecars"));
        }
    }
    let zero_gen2 = train_zero_games.get("gen2").copied().unwrap_or(0) + validation_zero_games.get("gen2").copied().unwrap_or(0);
    let zero_gen1 = train_zero_games.get("gen1").copied().unwrap_or(0) + validation_zero_games.get("gen1").copied().unwrap_or(0);
    if zero_gen2 != parse_usize(manifest, "source_gen2_zero_row_games")?
        || zero_gen1 != parse_usize(manifest, "source_gen1_zero_row_games")?
        || zero_gen1 + zero_gen2 != parse_usize(manifest, "zero_row_games")?
    {
        return Err("M6 zero-row composite identities disagree with sidecar and manifest counts".into());
    }
    expect(manifest, "validation_percent", "2")?;
    let expected_validation_count = source_games * 2 / 100;
    if validation_games != expected_validation_count {
        return Err("M6 sidecars do not match exact floor(2%) whole-game holdout".into());
    }
    let seed_text = manifest.get("split_seed").ok_or_else(|| "M6 adapter manifest is missing split_seed".to_string())?;
    let seed = if let Some(hex) = seed_text.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|error| format!("invalid M6 split seed: {error}"))?
    } else {
        seed_text.parse().map_err(|error| format!("invalid M6 split seed: {error}"))?
    };
    let ranked_validation = m6_validation_ids(&all_ids, seed)?;
    if ranked_validation != validation_ids {
        return Err("M6 game sidecars disagree with the manifest's composite-ID split algorithm".into());
    }

    let gen2_raw = parse_usize(manifest, "source_gen2_raw_records")?;
    let gen2_kept = parse_usize(manifest, "source_gen2_kept_records")?;
    let gen2_dropped = parse_usize(manifest, "source_gen2_dropped_duplicate_records")?;
    let gen2_within = parse_usize(manifest, "source_gen2_within_source_duplicate_records")?;
    let gen1_raw = parse_usize(manifest, "source_gen1_raw_records")?;
    let gen1_kept = parse_usize(manifest, "source_gen1_kept_records")?;
    let gen1_dropped = parse_usize(manifest, "source_gen1_dropped_duplicate_records")?;
    let gen1_within = parse_usize(manifest, "source_gen1_within_source_duplicate_records")?;
    let gen2_cross = parse_usize(manifest, "source_gen2_cross_source_duplicate_records")?;
    let gen1_cross = parse_usize(manifest, "source_gen1_cross_source_duplicate_records")?;
    let raw = parse_usize(manifest, "raw_records")?;
    let kept = parse_usize(manifest, "kept_records")?;
    let dropped = parse_usize(manifest, "dropped_duplicate_records")?;
    let within = parse_usize(manifest, "within_source_duplicate_records")?;
    let duplicates = parse_usize(manifest, "cross_source_duplicate_records")?;
    if gen2_cross != 0 || gen1_cross != duplicates {
        return Err("M6 per-source cross-source duplicate counts disagree with total cross-source duplicates".into());
    }
    if gen2_raw != gen2_kept + gen2_dropped || gen2_within != gen2_dropped
        || gen1_raw != gen1_kept + gen1_dropped || gen1_within + gen1_cross != gen1_dropped
        || raw != gen2_raw + gen1_raw || kept != gen2_kept + gen1_kept
        || dropped != gen2_dropped + gen1_dropped || within != gen2_within + gen1_within
        || kept != train_records + validation_records
    {
        return Err("M6 raw/kept/dropped source counts do not reconcile with shards".into());
    }

    let source_pair_sha = manifest.get("source_pair_sha256").cloned().unwrap();
    Ok(DataInfo {
        dataset_mode: "multi-source-v2".into(),
        train_records,
        validation_records,
        train_games,
        validation_games,
        train_sha256: sha256(&train_path)?,
        validation_sha256: sha256(&validation_path)?,
        adapter_manifest_sha256: sha256(&data_dir.join("manifest.txt"))?,
        m3_dataset_sha256: source_pair_sha,
        m3_config_sha256: manifest["source_gen2_config_sha256"].clone(),
        m3_source_tree_sha256: manifest["source_gen2_source_tree_sha256"].clone(),
        m3_source_manifest_sha256: manifest["source_gen2_manifest_sha256"].clone(),
        m3_source_revision: manifest["source_gen2_revision"].clone(),
        source_provenance: [
            ("source_gen2_dataset_sha256", "source_gen2_sha256"),
            ("source_gen1_dataset_sha256", "source_gen1_sha256"),
            ("source_gen2_config_sha256", "source_gen2_config_sha256"),
            ("source_gen1_config_sha256", "source_gen1_config_sha256"),
            ("source_gen2_source_tree_sha256", "source_gen2_source_tree_sha256"),
            ("source_gen1_source_tree_sha256", "source_gen1_source_tree_sha256"),
            ("source_gen2_manifest_sha256", "source_gen2_manifest_sha256"),
            ("source_gen1_manifest_sha256", "source_gen1_manifest_sha256"),
            ("source_gen2_engine_sha256", "source_gen2_engine_sha256"),
            ("source_gen1_engine_sha256", "source_gen1_engine_sha256"),
            ("source_gen2_evaluator", "source_gen2_evaluator"),
            ("source_gen1_evaluator", "source_gen1_evaluator"),
            ("source_gen2_search_evaluator", "source_gen2_search_evaluator"),
            ("source_gen2_nnue_sha256", "source_gen2_nnue_sha256"),
            ("source_gen2_raw_records", "source_gen2_raw_records"),
            ("source_gen2_kept_records", "source_gen2_kept_records"),
            ("source_gen2_dropped_duplicate_records", "source_gen2_dropped_duplicate_records"),
            ("source_gen2_within_source_duplicate_records", "source_gen2_within_source_duplicate_records"),
            ("source_gen2_cross_source_duplicate_records", "source_gen2_cross_source_duplicate_records"),
            ("source_gen1_raw_records", "source_gen1_raw_records"),
            ("source_gen1_kept_records", "source_gen1_kept_records"),
            ("source_gen1_dropped_duplicate_records", "source_gen1_dropped_duplicate_records"),
            ("source_gen1_within_source_duplicate_records", "source_gen1_within_source_duplicate_records"),
            ("source_gen1_cross_source_duplicate_records", "source_gen1_cross_source_duplicate_records"),
            ("source_gen2_revision", "source_gen2_revision"),
            ("source_gen1_revision", "source_gen1_revision"),
            ("source_gen2_zero_row_games", "source_gen2_zero_row_games"),
            ("source_gen1_zero_row_games", "source_gen1_zero_row_games"),
            ("zero_row_games", "zero_row_games"),
            ("source_pair_sha256", "source_pair_sha256"),
            ("raw_records", "raw_records"),
            ("kept_records", "kept_records"),
            ("dropped_duplicate_records", "dropped_duplicate_records"),
            ("within_source_duplicate_records", "within_source_duplicate_records"),
            ("cross_source_duplicate_records", "cross_source_duplicate_records"),
        ].into_iter().map(|(run_key, adapter_key)| (run_key.to_string(), manifest[adapter_key].clone())).collect(),
    })
}

fn validate_m6_sources_tsv(path: &Path, manifest: &BTreeMap<String, String>) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(io_error)?;
    let mut lines = text.lines();
    if lines.next() != Some("source\tartifact_sha256\tconfig_sha256\tsource_tree_sha256\tmanifest_sha256\tengine_sha256\tevaluator\tsearch_evaluator\tnnue_sha256") {
        return Err("M6 sources.tsv has an invalid header".into());
    }
    for source in ["gen2", "gen1"] {
        let line = lines.next().ok_or_else(|| "M6 sources.tsv is missing a source row".to_string())?;
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 9 || fields[0] != source {
            return Err("M6 sources.tsv rows are malformed or out of order".into());
        }
        for (index, suffix) in [
            "sha256", "config_sha256", "source_tree_sha256", "manifest_sha256", "engine_sha256",
            "evaluator", "search_evaluator", "nnue_sha256",
        ].iter().enumerate() {
            let key = format!("source_{source}_{suffix}");
            let expected = manifest.get(&key).map(String::as_str).unwrap_or("not-applicable");
            if fields[index + 1] != expected {
                return Err(format!("M6 sources.tsv field {key} disagrees with manifest"));
            }
        }
    }
    if lines.next().is_some() { return Err("M6 sources.tsv contains unexpected rows".into()); }
    Ok(())
}

fn validate_multi_sidecar(path: &Path, expected_records: usize, training: bool) -> Result<(usize, BTreeSet<(String, u64)>, BTreeMap<String, usize>), String> {
    let file = File::open(path).map_err(io_error)?;
    let mut lines = BufReader::new(file).lines();
    let header = if training {
        "source\tgame_id\tsegment_index\tfirst_record\trecord_count"
    } else {
        "source\tgame_id\tsegment_index\tfirst_record\trecord_count"
    };
    if lines.next().transpose().map_err(io_error)?.as_deref() != Some(header) {
        return Err(format!("{} has an invalid M6 composite sidecar header", path.display()));
    }
    let mut identities = BTreeSet::new();
    let mut seen_segments = BTreeSet::new();
    let mut per_identity = BTreeMap::<(String, u64), Vec<usize>>::new();
    let mut per_identity_rows = BTreeMap::<(String, u64), usize>::new();
    let mut ranges = Vec::<(usize, usize)>::new();
    let mut sum = 0usize;
    for line in lines {
        let line = line.map_err(io_error)?;
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 5 || !matches!(fields[0], "gen1" | "gen2") {
            return Err(format!("{} has a malformed M6 composite sidecar row", path.display()));
        }
        let identity = (fields[0].to_string(), fields[1].parse::<u64>().map_err(|error| format!("invalid composite game ID: {error}"))?);
        let segment = fields[2].parse::<usize>().map_err(|error| format!("invalid segment index: {error}"))?;
        let first = fields[3].parse::<usize>().map_err(|error| format!("invalid first record: {error}"))?;
        let count = fields[4].parse::<usize>().map_err(|error| format!("invalid record count: {error}"))?;
        if first.checked_add(count).is_none_or(|end| end > expected_records) {
            return Err(format!("{} sidecar row exceeds shard bounds", path.display()));
        }
        if !seen_segments.insert((identity.clone(), segment)) {
            return Err(format!("{} repeats a composite game segment", path.display()));
        }
        identities.insert(identity.clone());
        per_identity.entry(identity).or_default().push(segment);
        let identity_rows = per_identity_rows.entry((fields[0].to_string(), fields[1].parse::<u64>().map_err(|error| format!("invalid composite game ID: {error}"))?)).or_default();
        *identity_rows = identity_rows.checked_add(count).ok_or_else(|| "M6 per-game record count overflow".to_string())?;
        ranges.push((first, count));
        sum = sum.checked_add(count).ok_or_else(|| "M6 sidecar record count overflow".to_string())?;
    }
    for segments in per_identity.values_mut() {
        segments.sort_unstable();
        if segments.iter().enumerate().any(|(expected, actual)| expected != *actual) {
            return Err(format!("{} composite game segments are not contiguous from zero", path.display()));
        }
        if !training && segments.len() != 1 {
            return Err("M6 validation games must have one segment each".into());
        }
    }
    if sum != expected_records {
        return Err(format!("{} sidecar sums to {sum} records; shard contains {expected_records}", path.display()));
    }
    ranges.sort_unstable();
    let mut covered = 0usize;
    for (first, count) in ranges {
        if count == 0 { continue; }
        if first != covered { return Err(format!("{} has a gap or overlap in record offsets", path.display())); }
        covered = covered.checked_add(count).ok_or_else(|| "M6 sidecar range overflow".to_string())?;
    }
    if covered != expected_records { return Err(format!("{} does not cover every record exactly once", path.display())); }
    let mut zero_games_by_source = BTreeMap::<String, usize>::new();
    for ((source, _), record_count) in per_identity_rows {
        if record_count == 0 {
            *zero_games_by_source.entry(source).or_default() += 1;
        }
    }
    Ok((identities.len(), identities, zero_games_by_source))
}

fn m6_validation_ids(ids: &BTreeSet<(String, u64)>, seed: u64) -> Result<BTreeSet<(String, u64)>, String> {
    let validation_count = ids.len() * 2 / 100;
    if validation_count == 0 { return Err("M6 source pair is too small for an exact 2% game holdout".into()); }
    let mut ranked = ids.iter().cloned().map(|identity| {
        let (tag, source_order) = if identity.0 == "gen2" { (GEN2_SPLIT_TAG, 0u8) } else { (GEN1_SPLIT_TAG, 1u8) };
        (splitmix_finalizer(seed ^ MULTI_SPLIT_DOMAIN ^ tag ^ identity.1), source_order, identity.1, identity)
    }).collect::<Vec<_>>();
    ranked.sort_unstable();
    Ok(ranked.into_iter().take(validation_count).map(|(_, _, _, identity)| identity).collect())
}

fn splitmix_finalizer(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn validate_sidecar(path: &Path, expected_records: usize) -> Result<usize, String> {
    let file = File::open(path).map_err(io_error)?;
    let mut lines = BufReader::new(file).lines();
    if lines.next().transpose().map_err(io_error)?.as_deref() != Some("game_id\tsegment_index\tfirst_record\trecord_count") {
        return Err(format!("{} has an invalid game-ID sidecar header", path.display()));
    }
    let mut ids = BTreeSet::new();
    let mut sum = 0usize;
    for line in lines {
        let line = line.map_err(io_error)?;
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 4 {
            return Err(format!("{} has a malformed sidecar row", path.display()));
        }
        let id = fields[0].parse::<u64>().map_err(|error| format!("invalid game ID: {error}"))?;
        let _segment = fields[1].parse::<usize>().map_err(|error| format!("invalid segment index: {error}"))?;
        let _first = fields[2].parse::<usize>().map_err(|error| format!("invalid first record: {error}"))?;
        let count = fields[3].parse::<usize>().map_err(|error| format!("invalid record count: {error}"))?;
        ids.insert(id);
        sum = sum.checked_add(count).ok_or_else(|| "sidecar record count overflow".to_string())?;
    }
    if sum != expected_records {
        return Err(format!("{} sidecar sums to {sum} records; shard contains {expected_records}", path.display()));
    }
    Ok(ids.len())
}

fn sidecar_game_ids(path: &Path) -> Result<BTreeSet<u64>, String> {
    let text = fs::read_to_string(path).map_err(io_error)?;
    let mut ids = BTreeSet::new();
    for (row_number, line) in text.lines().enumerate().skip(1) {
        let id = line.split('\t').next().ok_or_else(|| format!("malformed sidecar row {}", row_number + 1))?;
        ids.insert(id.parse::<u64>().map_err(|error| format!("invalid sidecar game ID: {error}"))?);
    }
    Ok(ids)
}

fn evaluate_validation(
    path: &Path,
    expected_records: usize,
    mapper: &ModelInputsMapper<TrainRecord>,
    evaluator: &mut BatchEvaluator<ExecutionContext>,
    weights: &TensorMap<ExecutionContext>,
    device: &Arc<Device<ExecutionContext>>,
    config: &Config,
) -> Result<f64, String> {
    let file = File::open(path).map_err(io_error)?;
    let mut reader = BufReader::new(file);
    let pool = HostPool::new(device.clone());
    let mut records_seen = 0usize;
    let mut squared_error = 0.0f64;
    let mut step = Step::default();
    let mut batch = vec![TrainRecord::default(); config.validation_batch_size];
    loop {
        let remaining = expected_records.saturating_sub(records_seen);
        if remaining == 0 {
            break;
        }
        let count = remaining.min(config.validation_batch_size);
        let byte_count = count.checked_mul(RECORD_BYTES as usize).ok_or_else(|| "validation batch size overflow".to_string())?;
        let mut bytes = vec![0u8; byte_count];
        reader.read_exact(&mut bytes).map_err(io_error)?;
        for (slot, chunk) in batch[..count].iter_mut().zip(bytes.chunks_exact(RECORD_BYTES as usize)) {
            slot.0 = decode_record(chunk)?;
            validate_record(&slot.0)?;
        }
        if count < config.validation_batch_size {
            let last = batch[count - 1];
            batch[count..].fill(last);
        }
        let host = mapper.map(&pool, &batch, step, config.loader_threads).map_err(|error| format!("map validation batch: {error}"))?;
        let tensors = host.to_device(device).map_err(|error| format!("copy validation batch: {error}"))?;
        let output = evaluator
            .evaluate(&tensors, weights)
            .map_err(|error| format!("evaluate validation batch: {error}"))?;
        let logits = output.to_host().map_err(|error| format!("copy validation logits: {error}"))?;
        let TValue::F32(logits) = logits else { return Err("Bullet validation output is not f32".into()) };
        if logits.len() < count {
            return Err(format!("validation evaluator returned {} outputs for a {count}-record batch", logits.len()));
        }
        for (record, logit) in batch[..count].iter().zip(logits.iter()) {
            if !logit.is_finite() {
                return Err("validation model produced a non-finite logit".into());
            }
            let error = f64::from(sigmoid(*logit) - blended_target(&record.0));
            squared_error += error * error;
        }
        records_seen += count;
        step.step();
    }
    if records_seen != expected_records {
        return Err(format!("validation visited {records_seen} records; expected {expected_records}"));
    }
    Ok(squared_error / expected_records as f64)
}

fn decode_record(bytes: &[u8]) -> Result<TrainingRecord, String> {
    decode_training_record(bytes)
}

fn encode_record(board: &TrainingRecord) -> [u8; RECORD_BYTES as usize] {
    encode_training_record(board)
}

fn validate_record(board: &TrainingRecord) -> Result<(), String> {
    if board.result_idx() > 2 {
        return Err(format!("invalid side-to-move result code {}", board.result));
    }
    if !(-2_000..=2_000).contains(&board.score) {
        return Err(format!("score {} exceeds M3's ±2000 cp filter", board.score));
    }
    let mut count = 0;
    let mut invalid_piece = false;
    board.for_each_piece(|piece, square| {
        if square >= 64 || piece > 13 || piece & 7 > 5 {
            invalid_piece = true;
        }
        count += 1;
    });
    if invalid_piece { return Err("record contains a piece encoding outside Kira Chess768".into()); }
    if count == 0 || count > 32 {
        return Err(format!("record has invalid occupied-piece count {count}"));
    }
    Ok(())
}

struct BatchEvaluator<G: Gpu> {
    function: Function<G>,
    stream: Arc<Stream<G>>,
    buffers: BTreeMap<NodeId, Arc<Buffer<G>>>,
    input_ids: BTreeMap<String, NodeId>,
    weight_ids: BTreeMap<String, NodeId>,
    outputs: BTreeMap<String, Arc<Buffer<G>>>,
}

impl<G: Gpu> BatchEvaluator<G> {
    fn new(definition: &ModelDefinition, device: Arc<Device<G>>, batch_size: usize) -> Result<Self, String> {
        let forward = definition.lower_forward(batch_size).map_err(|error| format!("lower validation graph: {error}"))?;
        let map = forward.map();
        let mut buffers = BTreeMap::new();
        let mut input_ids = BTreeMap::new();
        let mut weight_ids = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        for (id, name) in definition.ir().inputs() {
            let Some(tensor_id) = map.get(id) else { continue };
            input_ids.insert(name.clone(), *tensor_id);
        }
        for (id, (name, _)) in definition.ir().weights() {
            let tensor_id = *map.get(id).ok_or_else(|| format!("lowered graph omitted weight {name}"))?;
            weight_ids.insert(name.clone(), tensor_id);
        }
        for (id, name) in definition.outputs() {
            let tensor_id = *map.get(id).ok_or_else(|| format!("lowered graph omitted output {name}"))?;
            let ty = forward.ir().get_node(tensor_id).map_err(|error| format!("find output tensor: {error}"))?.ty();
            let output = Buffer::zeroed(&device, ty.dtype(), ty.size().get()).map_err(|error| format!("allocate validation output: {error:?}"))?;
            buffers.insert(tensor_id, output.clone());
            outputs.insert(name.clone(), output);
        }
        let mut function = Function::new(device.clone(), forward.ir().clone()).map_err(|error| format!("compile validation forward graph: {error}"))?;
        function.prealloc().map_err(|error| format!("allocate validation graph: {error:?}"))?;
        let stream = device.new_stream().map_err(|error| format!("create validation stream: {error:?}"))?;
        Ok(Self { function, stream, buffers, input_ids, weight_ids, outputs })
    }

    fn evaluate(
        &mut self,
        tensors: &BTreeMap<String, Arc<Buffer<G>>>,
        weights: &TensorMap<G>,
    ) -> Result<Arc<Buffer<G>>, String> {
        let mut buffers = self.buffers.clone();
        for (name, tensor_id) in &self.input_ids {
            if let Some(value) = tensors.get(name) {
                buffers.insert(*tensor_id, value.clone());
            }
        }
        for (name, tensor_id) in &self.weight_ids {
            let value = weights.get(name).ok_or_else(|| format!("optimizer has no model weight {name}"))?;
            buffers.insert(*tensor_id, value.clone());
        }
        self.function
            .execute(self.stream.clone(), &buffers)
            .map_err(|error| format!("execute validation graph: {error:?}"))?
            .value()
            .map_err(|error| format!("synchronize validation graph: {error:?}"))?;
        self.outputs.get("output").cloned().ok_or_else(|| "model has no output tensor".into())
    }
}

fn collect_runtime_info() -> Result<RuntimeInfo, String> {
    let expected = |key: &str| {
        std::env::var(key).map_err(|_| format!("runtime provenance environment variable {key} is missing"))
    };
    let bullet_root = PathBuf::from(expected("KIRA_M4_BULLET_ROOT")?)
        .canonicalize()
        .map_err(io_error)?;
    let bullet_commit = expected("KIRA_M4_BULLET_COMMIT")?;
    if bullet_commit != PINNED_BULLET_COMMIT {
        return Err(format!("Bullet source commit mismatch: expected {PINNED_BULLET_COMMIT}, found {bullet_commit}"));
    }
    let expected_patch_sha256 = expected("KIRA_M4_BULLET_PATCH_SHA256")?;
    if expected_patch_sha256 != PINNED_BULLET_PATCH_SHA256 {
        return Err(format!("Bullet trainer-only patch mismatch: expected {PINNED_BULLET_PATCH_SHA256}, found {expected_patch_sha256}"));
    }
    let source_manifest_path = PathBuf::from(expected("KIRA_M4_BULLET_SOURCE_MANIFEST")?).canonicalize().map_err(io_error)?;
    let bullet_source_manifest_sha256 = expected("KIRA_M4_BULLET_SOURCE_MANIFEST_SHA256")?;
    let bullet_source_tree_sha256 = expected("KIRA_M4_BULLET_SOURCE_TREE_SHA256")?;
    if !is_sha256(&bullet_source_manifest_sha256)
        || bullet_source_manifest_sha256 != bullet_source_tree_sha256
    {
        return Err("sanitized Bullet source-tree/manifest SHA-256 is malformed or inconsistent".into());
    }
    verify_bullet_source_manifest(&bullet_root, &source_manifest_path, &bullet_source_tree_sha256)?;
    let dependency_tree_path = PathBuf::from(expected("KIRA_M4_BULLET_DEPENDENCY_TREE")?).canonicalize().map_err(io_error)?;
    let bullet_dependency_tree_sha256 = expected("KIRA_M4_BULLET_DEPENDENCY_TREE_SHA256")?;
    let dependency_tree_text = fs::read_to_string(&dependency_tree_path).map_err(io_error)?;
    if dependency_tree_text.contains("sfbinpack")
        || dependency_tree_text.contains("montyformat")
        || dependency_tree_text.contains("viriformat")
        || dependency_tree_text.contains("bulletformat")
    {
        return Err("locked sanitized Bullet dependency tree contains a forbidden chess-record library".into());
    }
    for removed in [
        "crates/bullet_lib/src/value.rs",
        "crates/bullet_lib/src/value",
    ] {
        if bullet_root.join(removed).exists() {
            return Err(format!("forbidden Bullet value/loader source remains in sanitized runtime: {removed}"));
        }
    }

    let bullet_lock = bullet_root.join("Cargo.lock");
    let driver_source_path = PathBuf::from(file!()).canonicalize().map_err(io_error)?;
    let record_source_path = driver_source_path
        .parent().and_then(Path::parent).and_then(Path::parent)
        .ok_or_else(|| "trainer source path has no repository root ancestor".to_string())?
        .join("src/record.rs").canonicalize().map_err(io_error)?;
    let runner_path = PathBuf::from(expected("KIRA_M4_RUNNER_PATH")?).canonicalize().map_err(io_error)?;
    let runtime_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let runtime_lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let values = [
        ("Bullet Cargo.lock", bullet_lock.as_path(), expected("KIRA_M4_BULLET_LOCK_SHA256")?),
        ("Bullet dependency tree", dependency_tree_path.as_path(), bullet_dependency_tree_sha256),
        ("trainer source", driver_source_path.as_path(), expected("KIRA_M4_DRIVER_SOURCE_SHA256")?),
        ("training record source", record_source_path.as_path(), expected("KIRA_M4_RECORD_SOURCE_SHA256")?),
        ("run_driver.py", runner_path.as_path(), expected("KIRA_M4_RUNNER_SHA256")?),
        ("runtime Cargo.toml", runtime_manifest.as_path(), expected("KIRA_M4_RUNTIME_MANIFEST_SHA256")?),
        ("runtime Cargo.lock", runtime_lock.as_path(), expected("KIRA_M4_RUNTIME_LOCK_SHA256")?),
    ];
    let mut hashes = BTreeMap::new();
    for (name, path, claimed) in values {
        if !is_sha256(&claimed) {
            return Err(format!("{name} provenance SHA-256 is malformed"));
        }
        let actual = sha256(path)?;
        if actual != claimed {
            return Err(format!("{name} changed between build and execution: expected {claimed}, found {actual}"));
        }
        hashes.insert(name, actual);
    }
    Ok(RuntimeInfo {
        bullet_root: bullet_root.display().to_string(),
        bullet_commit,
        bullet_lock_sha256: hashes["Bullet Cargo.lock"].clone(),
        bullet_patch_sha256: expected_patch_sha256,
        bullet_source_tree_sha256,
        bullet_source_manifest_sha256,
        bullet_dependency_tree_sha256: hashes["Bullet dependency tree"].clone(),
        driver_source_path: driver_source_path.display().to_string(),
        driver_source_sha256: hashes["trainer source"].clone(),
        record_source_path: record_source_path.display().to_string(),
        record_source_sha256: hashes["training record source"].clone(),
        runner_path: runner_path.display().to_string(),
        runner_sha256: hashes["run_driver.py"].clone(),
        runtime_manifest_sha256: hashes["runtime Cargo.toml"].clone(),
        runtime_lock_sha256: hashes["runtime Cargo.lock"].clone(),
    })
}

fn verify_bullet_source_manifest(root: &Path, manifest_path: &Path, expected_sha256: &str) -> Result<(), String> {
    let bytes = fs::read(manifest_path).map_err(io_error)?;
    let actual_sha256 = sha256_bytes(&bytes)?;
    if actual_sha256 != expected_sha256 {
        return Err(format!("sanitized Bullet source manifest changed: expected {expected_sha256}, found {actual_sha256}"));
    }
    let manifest = String::from_utf8(bytes).map_err(|error| format!("Bullet source manifest is not UTF-8: {error}"))?;
    let mut listed_paths = Vec::new();
    for line in manifest.lines() {
        let (claimed_sha256, relative) = line
            .split_once('\t')
            .ok_or_else(|| "malformed sanitized Bullet source-manifest row".to_string())?;
        if !is_sha256(claimed_sha256) || relative.is_empty() {
            return Err("malformed sanitized Bullet source-manifest entry".into());
        }
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path.components().any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            return Err(format!("unsafe path in sanitized Bullet source manifest: {relative}"));
        }
        let path = root.join(relative_path);
        let metadata = fs::symlink_metadata(&path).map_err(io_error)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!("sanitized Bullet source is missing or is not a regular file: {relative}"));
        }
        let actual = sha256(&path)?;
        if actual != claimed_sha256 {
            return Err(format!("sanitized Bullet source changed: {relative}"));
        }
        listed_paths.push(relative.to_owned());
    }
    if listed_paths.windows(2).any(|window| window[0] >= window[1]) {
        return Err("sanitized Bullet source manifest paths are not strictly sorted".into());
    }
    let mut actual_paths = Vec::new();
    collect_regular_source_paths(root, root, &mut actual_paths)?;
    actual_paths.sort();
    if actual_paths != listed_paths {
        return Err("sanitized Bullet source file set differs from the pinned source manifest".into());
    }
    Ok(())
}

fn collect_regular_source_paths(root: &Path, directory: &Path, paths: &mut Vec<String>) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(io_error)?;
        if metadata.file_type().is_symlink() {
            return Err(format!("symlink is not permitted in sanitized Bullet source: {}", path.display()));
        }
        if metadata.is_dir() {
            collect_regular_source_paths(root, &path, paths)?;
        } else if metadata.is_file() {
            let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
            let relative = relative.to_str().ok_or_else(|| "non-UTF-8 path in sanitized Bullet source".to_string())?;
            paths.push(relative.replace('\\', "/"));
        } else {
            return Err(format!("non-regular file in sanitized Bullet source: {}", path.display()));
        }
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> Result<String, String> {
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(io_error)?;
    child.stdin.as_mut().ok_or("shasum stdin is unavailable")?.write_all(bytes).map_err(io_error)?;
    drop(child.stdin.take());
    let output = child.wait_with_output().map_err(io_error)?;
    if !output.status.success() { return Err(format!("shasum failed: {}", String::from_utf8_lossy(&output.stderr))); }
    let stdout = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    let digest = stdout.split_whitespace().next().unwrap_or_default().to_ascii_lowercase();
    if !is_sha256(&digest) { return Err("shasum returned malformed patch SHA-256".into()); }
    Ok(digest)
}

fn print_runtime_info(runtime: &RuntimeInfo) {
    println!("bullet_root={}", runtime.bullet_root);
    println!("bullet_commit={}", runtime.bullet_commit);
    println!("bullet_cargo_lock_sha256={}", runtime.bullet_lock_sha256);
    println!("bullet_patch_sha256={}", runtime.bullet_patch_sha256);
    println!("bullet_source_tree_sha256={}", runtime.bullet_source_tree_sha256);
    println!("bullet_source_manifest_sha256={}", runtime.bullet_source_manifest_sha256);
    println!("bullet_dependency_tree_sha256={}", runtime.bullet_dependency_tree_sha256);
    println!("trainer_source_path={}", runtime.driver_source_path);
    println!("trainer_source_sha256={}", runtime.driver_source_sha256);
    println!("training_record_source_path={}", runtime.record_source_path);
    println!("training_record_source_sha256={}", runtime.record_source_sha256);
    println!("run_driver_path={}", runtime.runner_path);
    println!("run_driver_sha256={}", runtime.runner_sha256);
    println!("runtime_cargo_manifest_sha256={}", runtime.runtime_manifest_sha256);
    println!("runtime_cargo_lock_sha256={}", runtime.runtime_lock_sha256);
}

fn write_run_manifest(config: &Config, data: &DataInfo, runtime: &RuntimeInfo, output: &Path) -> Result<(), String> {
    let mut file = BufWriter::new(File::create(output.join("run_manifest.txt")).map_err(io_error)?);
    writeln!(file, "trainer_format=kira-m4-kira-record-trainer-v5-sanitized-locked-source-tree").map_err(io_error)?;
    writeln!(file, "bullet_root={}", runtime.bullet_root).map_err(io_error)?;
    writeln!(file, "bullet_commit={}", runtime.bullet_commit).map_err(io_error)?;
    writeln!(file, "bullet_cargo_lock_sha256={}", runtime.bullet_lock_sha256).map_err(io_error)?;
    writeln!(file, "bullet_patch_sha256={}", runtime.bullet_patch_sha256).map_err(io_error)?;
    writeln!(file, "bullet_source_tree_sha256={}", runtime.bullet_source_tree_sha256).map_err(io_error)?;
    writeln!(file, "bullet_source_manifest_sha256={}", runtime.bullet_source_manifest_sha256).map_err(io_error)?;
    writeln!(file, "bullet_dependency_tree_sha256={}", runtime.bullet_dependency_tree_sha256).map_err(io_error)?;
    writeln!(file, "trainer_source_path={}", runtime.driver_source_path).map_err(io_error)?;
    writeln!(file, "trainer_source_sha256={}", runtime.driver_source_sha256).map_err(io_error)?;
    writeln!(file, "training_record_source_path={}", runtime.record_source_path).map_err(io_error)?;
    writeln!(file, "training_record_source_sha256={}", runtime.record_source_sha256).map_err(io_error)?;
    writeln!(file, "run_driver_path={}", runtime.runner_path).map_err(io_error)?;
    writeln!(file, "run_driver_sha256={}", runtime.runner_sha256).map_err(io_error)?;
    writeln!(file, "runtime_cargo_manifest_sha256={}", runtime.runtime_manifest_sha256).map_err(io_error)?;
    writeln!(file, "runtime_cargo_lock_sha256={}", runtime.runtime_lock_sha256).map_err(io_error)?;
    writeln!(file, "adapter_data_dir={}", config.data_dir.display()).map_err(io_error)?;
    writeln!(file, "dataset_mode={}", data.dataset_mode).map_err(io_error)?;
    writeln!(file, "adapter_manifest_sha256={}", data.adapter_manifest_sha256).map_err(io_error)?;
    writeln!(file, "m3_dataset_sha256={}", data.m3_dataset_sha256).map_err(io_error)?;
    writeln!(file, "m3_config_sha256={}", data.m3_config_sha256).map_err(io_error)?;
    writeln!(file, "m3_source_tree_sha256={}", data.m3_source_tree_sha256).map_err(io_error)?;
    writeln!(file, "m3_source_manifest_sha256={}", data.m3_source_manifest_sha256).map_err(io_error)?;
    writeln!(file, "m3_source_revision={}", data.m3_source_revision).map_err(io_error)?;
    for (key, value) in &data.source_provenance {
        writeln!(file, "{key}={value}").map_err(io_error)?;
    }
    writeln!(file, "train_sha256={}", data.train_sha256).map_err(io_error)?;
    writeln!(file, "validation_sha256={}", data.validation_sha256).map_err(io_error)?;
    writeln!(file, "train_games={}", data.train_games).map_err(io_error)?;
    writeln!(file, "validation_games={}", data.validation_games).map_err(io_error)?;
    writeln!(file, "architecture=Chess768;dual-perspective;768x128-crelu;256x1-sigmoid").map_err(io_error)?;
    writeln!(file, "optimizer=AdamW;decay=0.01;beta1=0.9;beta2=0.999;clip=[-1.98,1.98]").map_err(io_error)?;
    writeln!(file, "batch_size={BATCH_SIZE}").map_err(io_error)?;
    writeln!(file, "seed=0x{:016x}", config.seed).map_err(io_error)?;
    writeln!(file, "score_lambda={SCORE_LAMBDA};bullet_constant_wdl={BULLET_WDL};scale={SCALE}").map_err(io_error)?;
    writeln!(file, "learning_rate=exponential;initial={INITIAL_LR};final={FINAL_LR};max_epochs={}", config.max_epochs).map_err(io_error)?;
    writeln!(file, "early_stopping=validation_mse;patience={};min_improvement={MIN_IMPROVEMENT}", config.patience).map_err(io_error)?;
    file.flush().map_err(io_error)?;
    file.get_ref().sync_all().map_err(io_error)
}

fn write_checkpoint_manifest(path: &Path, epoch: usize, lr: f32, train_mse: f64, validation_mse: f64, data: &DataInfo) -> Result<(), String> {
    let mut file = BufWriter::new(File::create(path).map_err(io_error)?);
    writeln!(file, "epoch={epoch}").map_err(io_error)?;
    writeln!(file, "learning_rate={lr:.10e}").map_err(io_error)?;
    writeln!(file, "train_mse={train_mse:.12}").map_err(io_error)?;
    writeln!(file, "validation_mse={validation_mse:.12}").map_err(io_error)?;
    writeln!(file, "train_sha256={}", data.train_sha256).map_err(io_error)?;
    writeln!(file, "validation_sha256={}", data.validation_sha256).map_err(io_error)?;
    file.flush().map_err(io_error)?;
    file.get_ref().sync_all().map_err(io_error)
}

fn read_key_values(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let text = fs::read_to_string(path).map_err(io_error)?;
    let mut values = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') { continue }
        let Some((key, value)) = line.split_once('=') else { return Err(format!("{} line {} is malformed", path.display(), index + 1)) };
        if values.insert(key.to_string(), value.to_string()).is_some() {
            return Err(format!("{} has duplicate key {key}", path.display()));
        }
    }
    Ok(values)
}

fn expect(values: &BTreeMap<String, String>, key: &str, expected: &str) -> Result<(), String> {
    let found = values.get(key).ok_or_else(|| format!("adapter manifest is missing {key}"))?;
    if found != expected { return Err(format!("adapter manifest {key} mismatch: expected {expected}, found {found}")); }
    Ok(())
}

fn parse_usize(values: &BTreeMap<String, String>, key: &str) -> Result<usize, String> {
    values.get(key).ok_or_else(|| format!("adapter manifest is missing {key}"))?.parse().map_err(|error| format!("invalid {key}: {error}"))
}

fn file_len(path: &Path) -> Result<u64, String> {
    fs::metadata(path).map(|metadata| metadata.len()).map_err(io_error)
}

fn sha256(path: &Path) -> Result<String, String> {
    let output = Command::new("shasum").args(["-a", "256", "--"]).arg(path).output().map_err(io_error)?;
    if !output.status.success() { return Err(format!("shasum failed: {}", String::from_utf8_lossy(&output.stderr))); }
    let stdout = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    let digest = stdout.split_whitespace().next().unwrap_or_default().to_ascii_lowercase();
    if !is_sha256(&digest) { return Err("shasum returned malformed SHA-256".into()); }
    Ok(digest)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn io_error(error: std::io::Error) -> String { error.to_string() }

#[cfg(test)]
mod tests {
    use super::*;

    fn record(score: i16, result: u8) -> TrainingRecord {
        TrainingRecord { occ: 0, pcs: [0; 16], score, result, ksq: 0, opp_ksq: 0, extra: [0; 3] }
    }

    #[test]
    fn target_semantics_keep_score_lambda_and_bullet_wdl_distinct() {
        let board = record(400, 2);
        let expected = 0.7 * sigmoid(1.0) + 0.3;
        assert!((blended_target(&board) - expected).abs() < 1.0e-7);
        assert_eq!(epoch_learning_rate(1, 100), INITIAL_LR);
        assert!((epoch_learning_rate(100, 100) - FINAL_LR).abs() < 1.0e-10);
    }

    #[test]
    fn an_epoch_is_one_superbatch_with_a_fixed_16384_batch_size() {
        let steps = training_epoch_steps(16_385).unwrap();
        assert_eq!(steps.batch_size, 16_384);
        assert_eq!(steps.batches_per_superbatch, 2);
        assert_eq!(steps.start_superbatch, 1);
        assert_eq!(steps.end_superbatch, 1);
        assert!(training_epoch_steps(0).is_err());
    }

    #[test]
    fn model_build_lowers_a_single_raw_logit_for_each_position() {
        let model = build_model();
        assert_eq!(model.outputs().len(), 1);
        let (output, name) = &model.outputs()[0];
        assert_eq!(name, "output");
        assert_eq!(model.ir().weights().len(), 4, "l0/l1 each have weights and bias");

        let forward = model.lower_forward(7).expect("model forward graph should lower");
        let output_id = *forward.map().get(output).expect("raw-logit output remains mapped");
        let output_type = forward.ir().get_node(output_id).expect("lowered output node").ty();
        assert_eq!(output_type.dtype(), bullet_compiler::tensor::DType::F32);
        assert_eq!(output_type.size().get(), 7, "one scalar logit for each of seven rows");
    }

    #[test]
    fn run_manifest_records_adapter_m3_and_runtime_provenance() {
        let output = std::env::temp_dir().join(format!("kira-m4-run-manifest-{}", std::process::id()));
        if output.exists() { fs::remove_dir_all(&output).unwrap(); }
        fs::create_dir(&output).unwrap();
        let digest = |byte| char::from(byte).to_string().repeat(64);
        let data = DataInfo {
            dataset_mode: "single-source-v1".into(),
            train_records: 49,
            validation_records: 1,
            train_games: 49,
            validation_games: 1,
            train_sha256: digest(b'a'),
            validation_sha256: digest(b'b'),
            adapter_manifest_sha256: digest(b'c'),
            m3_dataset_sha256: digest(b'd'),
            m3_config_sha256: digest(b'e'),
            m3_source_tree_sha256: digest(b'f'),
            m3_source_manifest_sha256: digest(b'1'),
            m3_source_revision: "fixture-source-tag".into(),
            source_provenance: BTreeMap::new(),
        };
        let runtime = RuntimeInfo {
            bullet_root: "/pinned/bullet".into(),
            bullet_commit: PINNED_BULLET_COMMIT.into(),
            bullet_lock_sha256: digest(b'2'),
            bullet_patch_sha256: PINNED_BULLET_PATCH_SHA256.into(),
            bullet_source_tree_sha256: digest(b'7'),
            bullet_source_manifest_sha256: digest(b'7'),
            bullet_dependency_tree_sha256: digest(b'8'),
            driver_source_path: "/workspace/driver/src/main.rs".into(),
            driver_source_sha256: digest(b'3'),
            record_source_path: "/workspace/src/record.rs".into(),
            record_source_sha256: digest(b'3'),
            runner_path: "/workspace/run_driver.py".into(),
            runner_sha256: digest(b'4'),
            runtime_manifest_sha256: digest(b'5'),
            runtime_lock_sha256: digest(b'6'),
        };
        let config = Config {
            data_dir: PathBuf::from("fixture-data"),
            output_dir: output.clone(),
            max_epochs: 2,
            patience: 1,
            seed: SEED_DEFAULT,
            loader_threads: 1,
            validation_batch_size: 1,
            validate_only: false,
        };
        write_run_manifest(&config, &data, &runtime, &output).unwrap();
        let manifest = fs::read_to_string(output.join("run_manifest.txt")).unwrap();
        for expected in [
            format!("adapter_manifest_sha256={}", data.adapter_manifest_sha256),
            format!("m3_dataset_sha256={}", data.m3_dataset_sha256),
            format!("m3_source_tree_sha256={}", data.m3_source_tree_sha256),
            format!("m3_source_revision={}", data.m3_source_revision),
            format!("train_sha256={}", data.train_sha256),
            format!("validation_sha256={}", data.validation_sha256),
            format!("bullet_commit={}", runtime.bullet_commit),
            format!("bullet_cargo_lock_sha256={}", runtime.bullet_lock_sha256),
            format!("bullet_patch_sha256={}", runtime.bullet_patch_sha256),
            format!("bullet_source_tree_sha256={}", runtime.bullet_source_tree_sha256),
            format!("bullet_source_manifest_sha256={}", runtime.bullet_source_manifest_sha256),
            format!("bullet_dependency_tree_sha256={}", runtime.bullet_dependency_tree_sha256),
            format!("trainer_source_sha256={}", runtime.driver_source_sha256),
            format!("training_record_source_sha256={}", runtime.record_source_sha256),
            format!("run_driver_sha256={}", runtime.runner_sha256),
            format!("runtime_cargo_manifest_sha256={}", runtime.runtime_manifest_sha256),
            format!("runtime_cargo_lock_sha256={}", runtime.runtime_lock_sha256),
        ] {
            assert!(manifest.lines().any(|line| line == expected), "missing {expected}");
        }
        fs::remove_dir_all(output).unwrap();
    }

    #[test]
    fn piece_row_mapping_uses_kira_chess768_order_and_rank_flip() {
        // White K at e1, Black K at e8, encoded as Kira's K piece index 5.
        let bbs = [1 << 4, 1 << 60, 0, 0, 0, 0, 0, (1 << 4) | (1 << 60)];
        let board = TrainingRecord::from_bitboards(bbs, false, 0, 1).unwrap();
        let mut mapped = Vec::new();
        KiraChess768.map_features(&board, |stm, nstm| mapped.push((stm, nstm)));
        assert!(mapped.contains(&(5 * 64 + 4, 384 + 5 * 64 + (4 ^ 56))));
        assert!(mapped.contains(&(384 + 5 * 64 + 60, 5 * 64 + (60 ^ 56))));
    }

    #[test]
    fn kira_record_codec_is_explicit_little_endian() {
        let bytes = [
            4, 3, 2, 1, 8, 7, 6, 5, 0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 0x34, 0x12, 2, 4, 60, 0xaa, 0xbb, 0xcc,
        ];
        let board = decode_record(&bytes).unwrap();
        assert_eq!(board.occ, 0x0506_0708_0102_0304);
        assert_eq!(board.score, 0x1234);
        assert_eq!(board.result, 2);
        assert_eq!(board.extra, [0xaa, 0xbb, 0xcc]);
    }

    #[test]
    fn validation_plateau_minimum_improvement_is_pinned() {
        let mut best = f64::INFINITY;
        let mut stale = 0;
        for value in [0.9, 0.900_000_01, 0.899_999_8] {
            if !best.is_finite() || value + MIN_IMPROVEMENT < best { best = value; stale = 0; } else { stale += 1; }
        }
        assert_eq!(best, 0.899_999_8);
        assert_eq!(stale, 0);
    }

    #[test]
    fn synthetic_adapter_artifact_passes_provenance_hash_and_game_split_gates() {
        let fixture = std::env::temp_dir().join(format!("kira-m4-driver-fixture-{}", std::process::id()));
        if fixture.exists() { fs::remove_dir_all(&fixture).unwrap(); }
        fs::create_dir(&fixture).unwrap();
        let board = TrainingRecord {
            occ: (1u64 << 4) | (1u64 << 60),
            pcs: [0xd5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            score: 0,
            result: 1,
            ksq: 4,
            opp_ksq: 60,
            extra: [0; 3],
        };
        let mut train = File::create(fixture.join("train.bullet")).unwrap();
        for _ in 0..49 { train.write_all(&encode_record(&board)).unwrap(); }
        train.sync_all().unwrap();
        let mut validation = File::create(fixture.join("validation.bullet")).unwrap();
        validation.write_all(&encode_record(&board)).unwrap();
        validation.sync_all().unwrap();
        let mut train_ids = BufWriter::new(File::create(fixture.join("train_game_ids.tsv")).unwrap());
        writeln!(train_ids, "game_id\tsegment_index\tfirst_record\trecord_count").unwrap();
        for id in 0..49 { writeln!(train_ids, "{id}\t0\t{id}\t1").unwrap(); }
        train_ids.flush().unwrap();
        train_ids.get_ref().sync_all().unwrap();
        let mut validation_ids = BufWriter::new(File::create(fixture.join("validation_game_ids.tsv")).unwrap());
        writeln!(validation_ids, "game_id\tsegment_index\tfirst_record\trecord_count").unwrap();
        writeln!(validation_ids, "1000\t0\t0\t1").unwrap();
        validation_ids.flush().unwrap();
        validation_ids.get_ref().sync_all().unwrap();
        let manifest = format!(
            "adapter_format=kira-m4-record-v2\nsource_format=K1ADATAS-v2\nbullet_git_commit={PINNED_BULLET_COMMIT}\nrecord_format=KiraPackedTrainingRecord-v1;size=32;little-endian\nkira_score_lambda=0.7\nbullet_constant_wdl=0.3\neval_scale=400\nsource_sha256={}\nsource_config_sha256={}\nkira_source_tree_sha256={}\nm3_source_manifest_sha256={}\nkira_source_revision=test-fixture\ntrain_sha256={}\nvalidation_sha256={}\ntrain_game_ids_sha256={}\nvalidation_game_ids_sha256={}\ntrain_records=49\nvalidation_records=1\ntrain_games=49\nvalidation_games=1\nsource_games=50\nvalidation_percent=2\n",
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            sha256(&fixture.join("train.bullet")).unwrap(),
            sha256(&fixture.join("validation.bullet")).unwrap(),
            sha256(&fixture.join("train_game_ids.tsv")).unwrap(),
            sha256(&fixture.join("validation_game_ids.tsv")).unwrap(),
        );
        fs::write(fixture.join("manifest.txt"), manifest).unwrap();

        let accepted = validate_dataset(&fixture).unwrap();
        assert_eq!(accepted.train_records, 49);
        assert_eq!(accepted.validation_records, 1);
        assert_eq!(accepted.train_games, 49);
        assert_eq!(accepted.validation_games, 1);
        assert_eq!(accepted.m3_source_revision, "test-fixture");

        let bad_hash = format!("{}\n", fs::read_to_string(fixture.join("manifest.txt")).unwrap()).replace(
            &accepted.validation_sha256,
            &"f".repeat(64),
        );
        fs::write(fixture.join("manifest.txt"), bad_hash).unwrap();
        assert!(validate_dataset(&fixture).is_err());
        fs::remove_dir_all(fixture).unwrap();
    }

    #[test]
    fn m6_adapter_manifest_accepts_composite_ids_and_rejects_wrong_gen2_network_pin() {
        let fixture = std::env::temp_dir().join(format!("kira-m6-driver-fixture-{}", std::process::id()));
        if fixture.exists() { fs::remove_dir_all(&fixture).unwrap(); }
        fs::create_dir(&fixture).unwrap();
        let board = TrainingRecord {
            occ: (1u64 << 4) | (1u64 << 60),
            pcs: [0xd5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            score: 0, result: 1, ksq: 4, opp_ksq: 60, extra: [0; 3],
        };
        let identities = (0..50u64)
            .flat_map(|game_id| [("gen2".to_string(), game_id), ("gen1".to_string(), game_id)])
            .collect::<BTreeSet<_>>();
        let validation = m6_validation_ids(&identities, 0x4b4952414d345631).unwrap();
        assert_eq!(identities.len(), 100, "overlapping game IDs remain composite identities");
        assert_eq!(validation.len(), 2);
        let train_ids = identities.difference(&validation).cloned().collect::<BTreeSet<_>>();

        let empty_identity = train_ids.iter().find(|(source, _)| source == "gen1").unwrap().clone();
        let mut train_file = File::create(fixture.join("train.bullet")).unwrap();
        let mut train_sidecar = BufWriter::new(File::create(fixture.join("train_game_ids.tsv")).unwrap());
        writeln!(train_sidecar, "source\tgame_id\tsegment_index\tfirst_record\trecord_count").unwrap();
        let mut train_record_count = 0usize;
        for (source, game_id) in &train_ids {
            let is_empty = (source, game_id) == (&empty_identity.0, &empty_identity.1);
            let count = if is_empty { 0 } else { 1 };
            let first_record = train_record_count;
            if count == 1 {
                train_file.write_all(&encode_record(&board)).unwrap();
                train_record_count += 1;
            }
            writeln!(train_sidecar, "{source}\t{game_id}\t0\t{first_record}\t{count}").unwrap();
        }
        train_file.sync_all().unwrap();
        train_sidecar.flush().unwrap();
        train_sidecar.get_ref().sync_all().unwrap();
        let mut validation_file = File::create(fixture.join("validation.bullet")).unwrap();
        let mut validation_sidecar = BufWriter::new(File::create(fixture.join("validation_game_ids.tsv")).unwrap());
        writeln!(validation_sidecar, "source\tgame_id\tsegment_index\tfirst_record\trecord_count").unwrap();
        for (index, (source, game_id)) in validation.iter().enumerate() {
            validation_file.write_all(&encode_record(&board)).unwrap();
            writeln!(validation_sidecar, "{source}\t{game_id}\t0\t{index}\t1").unwrap();
        }
        validation_file.sync_all().unwrap();
        validation_sidecar.flush().unwrap();
        validation_sidecar.get_ref().sync_all().unwrap();

        let digest = |ch: char| ch.to_string().repeat(64);
        let sources_tsv = format!(
            "source\tartifact_sha256\tconfig_sha256\tsource_tree_sha256\tmanifest_sha256\tengine_sha256\tevaluator\tsearch_evaluator\tnnue_sha256\n\
             gen2\t{}\t{}\t{}\t{}\t{}\tKira-HCE\tQuantizedNnueNetwork<128>\t{}\n\
             gen1\t{}\t{}\t{}\t{}\t{}\tKira-HCE\tnot-applicable\tnot-applicable\n",
            digest('1'), digest('2'), digest('3'), digest('4'), digest('5'), PINNED_GEN2_NNUE_SHA256,
            digest('6'), digest('7'), digest('8'), digest('9'), digest('a'),
        );
        fs::write(fixture.join("sources.tsv"), &sources_tsv).unwrap();
        let source_pair_sha = sha256(&fixture.join("sources.tsv")).unwrap();
        let manifest = format!(
            "adapter_format=kira-m6-multisource-record-v3\nsource_format=K1ADATAS-v2\nsource_precedence=gen2>gen1;first-occurrence-within-source;duplicate_key=NNUE-active-feature-key-v1\nnnue_feature_key_schema={}\nnnue_feature_key_encoding=sparse-sorted-10-bit-indices;stm-perspective-first;max-32-pieces-per-perspective\nbullet_git_commit={PINNED_BULLET_COMMIT}\nrecord_format=KiraPackedTrainingRecord-v1;size=32;little-endian\nkira_score_lambda=0.7\nbullet_constant_wdl=0.3\neval_scale=400\nsource_gen2_sha256={}\nsource_gen1_sha256={}\nsource_pair_sha256={source_pair_sha}\nsource_gen2_config_sha256={}\nsource_gen1_config_sha256={}\nsource_gen2_source_tree_sha256={}\nsource_gen1_source_tree_sha256={}\nsource_gen2_manifest_sha256={}\nsource_gen1_manifest_sha256={}\nsource_gen2_engine_sha256={}\nsource_gen1_engine_sha256={}\nsource_gen2_evaluator=Kira-HCE\nsource_gen1_evaluator=Kira-HCE\nsource_gen2_search_evaluator=QuantizedNnueNetwork<128>\nsource_gen2_nnue_sha256={PINNED_GEN2_NNUE_SHA256}\nsource_gen2_revision=gen2-fixture\nsource_gen1_revision=gen1-fixture\nsource_gen2_games=50\nsource_gen1_games=50\nsource_games=100\nsource_gen2_zero_row_games=0\nsource_gen1_zero_row_games=1\nzero_row_games=1\nsource_gen2_raw_records=50\nsource_gen2_kept_records=50\nsource_gen2_dropped_duplicate_records=0\nsource_gen2_within_source_duplicate_records=0\nsource_gen2_cross_source_duplicate_records=0\nsource_gen1_raw_records=50\nsource_gen1_kept_records=49\nsource_gen1_dropped_duplicate_records=1\nsource_gen1_within_source_duplicate_records=0\nsource_gen1_cross_source_duplicate_records=1\nraw_records=100\nkept_records=99\ndropped_duplicate_records=1\nwithin_source_duplicate_records=0\ncross_source_duplicate_records=1\nsplit_seed=0x4b4952414d345631\nvalidation_percent=2\ntrain_games=98\nvalidation_games=2\ntrain_records={}\nvalidation_records=2\ntrain_sha256={}\nvalidation_sha256={}\ntrain_game_ids_sha256={}\nvalidation_game_ids_sha256={}\n",
            NNUE_FEATURE_KEY_SCHEMA,
            digest('1'), digest('6'), digest('2'), digest('7'), digest('3'), digest('8'), digest('4'), digest('9'), digest('5'), digest('a'),
            train_record_count,
            sha256(&fixture.join("train.bullet")).unwrap(),
            sha256(&fixture.join("validation.bullet")).unwrap(),
            sha256(&fixture.join("train_game_ids.tsv")).unwrap(),
            sha256(&fixture.join("validation_game_ids.tsv")).unwrap(),
        );
        fs::write(fixture.join("manifest.txt"), &manifest).unwrap();

        let accepted = validate_dataset(&fixture).unwrap();
        assert_eq!(accepted.dataset_mode, "multi-source-v2");
        assert_eq!(accepted.train_games, 98);
        assert_eq!(accepted.validation_games, 2);
        assert_eq!(accepted.train_records, 97);
        assert_eq!(accepted.validation_records, 2);
        assert_eq!(accepted.m3_dataset_sha256, source_pair_sha);
        let train_sidecar = fs::read_to_string(fixture.join("train_game_ids.tsv")).unwrap();
        assert!(train_sidecar.lines().any(|line| {
            let fields = line.split('\t').collect::<Vec<_>>();
            fields.len() == 5 && fields[0] == empty_identity.0 && fields[1] == empty_identity.1.to_string() && fields[4] == "0"
        }));

        let mismatched_source_games = manifest
            .replace("source_gen2_games=50", "source_gen2_games=49")
            .replace("source_gen1_games=50", "source_gen1_games=51");
        fs::write(fixture.join("manifest.txt"), mismatched_source_games).unwrap();
        assert!(validate_dataset(&fixture).unwrap_err().contains("per-source game count"));

        let mismatched_gen1_cross = manifest.replace("source_gen1_cross_source_duplicate_records=1", "source_gen1_cross_source_duplicate_records=0");
        fs::write(fixture.join("manifest.txt"), mismatched_gen1_cross).unwrap();
        assert!(validate_dataset(&fixture).unwrap_err().contains("per-source cross-source duplicate counts"));

        let unexpected_gen2_cross = manifest.replace("source_gen2_cross_source_duplicate_records=0", "source_gen2_cross_source_duplicate_records=1");
        fs::write(fixture.join("manifest.txt"), unexpected_gen2_cross).unwrap();
        assert!(validate_dataset(&fixture).unwrap_err().contains("per-source cross-source duplicate counts"));

        fs::write(
            fixture.join("manifest.txt"),
            manifest.replace(PINNED_GEN2_NNUE_SHA256, &digest('d')),
        ).unwrap();
        assert!(validate_dataset(&fixture).unwrap_err().contains("source_gen2_nnue_sha256"));
        fs::remove_dir_all(fixture).unwrap();
    }
}
