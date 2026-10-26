use kira1a::eval::nnue_quantized::{
    HEADER_BYTES, QuantizedNnueMetadata, QuantizedNnueNetwork, sha256_hex,
};
use kira_m4_adapter::bullet_feature_to_kira;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const HIDDEN: usize = 128;
const BULLET_COMMIT: &str = "579b78b37a694cacd6e9ede4706386c15c34837c";
const TRAINER_FORMAT: &str = "kira-m4-bullet-trainer-v1";
const SANITIZED_TRAINER_FORMAT: &str = "kira-m4-kira-record-trainer-v5-sanitized-locked-source-tree";
const BULLET_PATCH_SHA256: &str = "ecf8ce81486d5e2d3cfbd881bbd053e8b3682b8de4acf24a6824310cb29b5141";
const ARCHITECTURE: &str = "Chess768;dual-perspective;768x128-crelu;256x1-sigmoid";
const OPTIMIZER: &str = "AdamW;decay=0.01;beta1=0.9;beta2=0.999;clip=[-1.98,1.98]";
const SCORE_CONFIG: &str = "0.7;bullet_constant_wdl=0.3;scale=400";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn main() {
    if let Err(error) = run_cli() {
        eprintln!("kira-m4-export-nnue: {error}");
        std::process::exit(2);
    }
}

fn run_cli() -> Result<(), String> {
    let mut run_dir = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                println!("Usage: kira-m4-export-nnue --run-dir TRAINING_RUN --out NEW_FILE.nnue");
                return Ok(());
            }
            "--run-dir" => run_dir = Some(PathBuf::from(next_arg(&mut args, "--run-dir")?)),
            "--out" => output = Some(PathBuf::from(next_arg(&mut args, "--out")?)),
            other => return Err(format!("unknown argument {other}; use --help")),
        }
    }
    let run_dir = run_dir.ok_or_else(|| "--run-dir is required".to_string())?;
    let output = output.ok_or_else(|| "--out is required".to_string())?;
    let result = export_best_checkpoint(&run_dir, &output)?;
    println!("output={}", output.display());
    println!("best_epoch={}", result.epoch);
    println!("best_validation_mse={:.10}", result.validation_mse);
    println!("weights_sha256={}", result.weights_sha256);
    println!("run_manifest_sha256={}", result.run_manifest_sha256);
    println!("payload_sha256={}", result.payload_sha256);
    println!("q_scale=16384 score_scale=400 optimizer_clip=[-1.98,1.98]");
    println!("clamped_feature_weights={}", result.report.feature_weights_clamped);
    println!("clamped_hidden_biases={}", result.report.hidden_bias_clamped);
    println!("clamped_output_weights={}", result.report.output_weights_clamped);
    println!("clamped_output_biases={}", result.report.output_bias_clamped);
    println!("clamped_total={}", result.report.clamped_total());
    println!("optimizer_clip_violations={}", result.report.optimizer_clip_violations);
    Ok(())
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} requires a value"))
}

#[derive(Debug)]
struct ExportResult {
    epoch: usize,
    validation_mse: f64,
    weights_sha256: String,
    run_manifest_sha256: String,
    payload_sha256: String,
    report: kira1a::eval::nnue_quantized::QuantizationReport,
}

fn export_best_checkpoint(run_dir: &Path, output: &Path) -> Result<ExportResult, String> {
    if output.exists() {
        return Err(format!("refusing to overwrite existing output {}", output.display()));
    }
    let run_manifest_path = run_dir.join("run_manifest.txt");
    let run_manifest_bytes = fs::read(&run_manifest_path).map_err(io_error)?;
    let run_manifest_text = std::str::from_utf8(&run_manifest_bytes)
        .map_err(|error| format!("run manifest is not UTF-8: {error}"))?;
    let run_manifest = parse_key_values(run_manifest_text, "run manifest")?;
    validate_run_manifest(&run_manifest)?;
    let run_manifest_sha256 = sha256_hex(&run_manifest_bytes);

    let losses_path = run_dir.join("losses.csv");
    let losses_text = fs::read_to_string(&losses_path).map_err(io_error)?;
    let best = parse_best_epoch(&losses_text)?;
    let epoch_dir = run_dir.join(format!("checkpoint-{:04}", best.epoch));
    let checkpoint_manifest = parse_key_values(
        &fs::read_to_string(epoch_dir.join("manifest.txt")).map_err(io_error)?,
        "checkpoint manifest",
    )?;
    validate_checkpoint_manifest(&checkpoint_manifest, &run_manifest, &best)?;

    let weights_path = epoch_dir.join("weights.bin");
    let weights_bytes = fs::read(&weights_path).map_err(io_error)?;
    let weights_sha256 = sha256_hex(&weights_bytes);
    let tensors = parse_weights(&weights_bytes)?;
    let converted = convert_tensors(&tensors)?;
    let network = QuantizedNnueNetwork::<HIDDEN>::from_float_parts(
        &converted.feature_weights,
        &converted.hidden_bias,
        &converted.output_weights,
        converted.output_bias,
        400.0,
        QuantizedNnueMetadata {
            source_weights_sha256: parse_sha256(&weights_sha256, "weights SHA-256")?,
            run_manifest_sha256: parse_sha256(&run_manifest_sha256, "run manifest SHA-256")?,
        },
    )?;
    let report = network.quantization_report();
    let bytes = network.to_bytes()?;
    let payload_sha256 = sha256_hex(&bytes[HEADER_BYTES..]);
    atomic_write_new(output, &bytes)?;

    Ok(ExportResult {
        epoch: best.epoch,
        validation_mse: best.validation_mse,
        weights_sha256,
        run_manifest_sha256,
        payload_sha256,
        report,
    })
}

fn validate_run_manifest(values: &BTreeMap<String, String>) -> Result<(), String> {
    validate_trainer_format(values)?;
    expect(values, "bullet_commit", BULLET_COMMIT)?;
    expect(values, "architecture", ARCHITECTURE)?;
    expect(values, "optimizer", OPTIMIZER)?;
    expect(values, "score_lambda", SCORE_CONFIG)?;
    expect(values, "batch_size", "16384")?;
    for key in [
        "bullet_cargo_lock_sha256",
        "trainer_source_sha256",
        "training_record_source_sha256",
        "run_driver_sha256",
        "runtime_cargo_manifest_sha256",
        "runtime_cargo_lock_sha256",
        "adapter_manifest_sha256",
        "m3_dataset_sha256",
        "m3_config_sha256",
        "m3_source_tree_sha256",
        "m3_source_manifest_sha256",
        "train_sha256",
        "validation_sha256",
    ] {
        parse_sha256(values.get(key).ok_or_else(|| format!("run manifest is missing {key}"))?, key)?;
    }
    for key in ["bullet_root", "trainer_source_path", "training_record_source_path", "run_driver_path", "m3_source_revision", "seed"] {
        if values.get(key).is_none_or(|value| value.is_empty()) {
            return Err(format!("run manifest is missing {key}"));
        }
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
struct EpochRow {
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
    let mut rows = BTreeMap::<usize, EpochRow>::new();
    let mut footer = None;
    for line in lines {
        if line.is_empty() { continue; }
        if let Some(rest) = line.strip_prefix("# best_epoch=") {
            if footer.is_some() { return Err("losses.csv has multiple best-epoch footers".into()); }
            let (epoch, mse) = rest.split_once(" best_validation_mse=")
                .ok_or_else(|| "losses.csv best-epoch footer is malformed".to_string())?;
            let epoch = epoch.parse::<usize>().map_err(|error| format!("invalid best epoch: {error}"))?;
            let mse = parse_finite(mse, "best validation MSE")?;
            footer = Some(BestEpoch { epoch, learning_rate: 0.0, train_mse: 0.0, validation_mse: mse });
            continue;
        }
        if line.starts_with('#') { return Err("losses.csv contains an unknown comment/footer".into()); }
        let fields: Vec<_> = line.split(',').collect();
        if fields.len() != 7 { return Err("losses.csv row must have seven fields".into()); }
        let epoch = fields[0].parse::<usize>().map_err(|error| format!("invalid epoch: {error}"))?;
        let learning_rate = parse_finite(fields[1], "learning rate")?;
        let train_mse = parse_finite(fields[2], "train MSE")?;
        let validation_mse = parse_finite(fields[3], "validation MSE")?;
        let validation_records = fields[4].parse::<usize>().map_err(|error| format!("invalid validation record count: {error}"))?;
        let selected_epoch = fields[5].parse::<usize>().map_err(|error| format!("invalid row best epoch: {error}"))?;
        let _stale_epochs = fields[6].parse::<usize>().map_err(|error| format!("invalid stale epoch count: {error}"))?;
        if epoch == 0 || learning_rate <= 0.0 || train_mse < 0.0 || validation_mse < 0.0 || validation_records == 0
            || selected_epoch == 0 || selected_epoch > epoch
            || rows.insert(epoch, EpochRow { learning_rate, train_mse, validation_mse, best_epoch: selected_epoch }).is_some()
        {
            return Err("losses.csv has invalid or duplicate epoch rows".into());
        }
    }
    let best = footer.ok_or_else(|| "losses.csv is missing its best-epoch footer".to_string())?;
    let selected = rows.get(&best.epoch).copied()
        .ok_or_else(|| "best epoch has no losses.csv row".to_string())?;
    if selected.best_epoch != best.epoch || (selected.validation_mse - best.validation_mse).abs() > 5.1e-11 {
        return Err("losses.csv best footer does not match its selected epoch row".into());
    }
    for (&epoch, row) in &rows {
        if epoch != best.epoch && row.validation_mse + 1.0e-7 < best.validation_mse {
            return Err("losses.csv footer does not select the best validation checkpoint".into());
        }
    }
    Ok(BestEpoch { epoch: best.epoch, learning_rate: selected.learning_rate, train_mse: selected.train_mse, validation_mse: best.validation_mse })
}

fn validate_checkpoint_manifest(
    checkpoint: &BTreeMap<String, String>,
    run: &BTreeMap<String, String>,
    best: &BestEpoch,
) -> Result<(), String> {
    expect(checkpoint, "epoch", &best.epoch.to_string())?;
    let learning_rate = parse_finite(checkpoint.get("learning_rate").ok_or_else(|| "checkpoint manifest is missing learning_rate".to_string())?, "checkpoint learning rate")?;
    if (learning_rate - best.learning_rate).abs() > 5.1e-11 {
        return Err("checkpoint manifest learning rate disagrees with losses.csv".into());
    }
    let train_mse = parse_finite(checkpoint.get("train_mse").ok_or_else(|| "checkpoint manifest is missing train_mse".to_string())?, "checkpoint train MSE")?;
    if (train_mse - best.train_mse).abs() > 5.1e-11 {
        return Err("checkpoint manifest train MSE disagrees with losses.csv".into());
    }
    let mse = parse_finite(checkpoint.get("validation_mse").ok_or_else(|| "checkpoint manifest is missing validation_mse".to_string())?, "checkpoint validation MSE")?;
    if (mse - best.validation_mse).abs() > 5.1e-11 {
        return Err("checkpoint manifest validation MSE disagrees with losses.csv".into());
    }
    for key in ["train_sha256", "validation_sha256"] {
        expect(checkpoint, key, run.get(key).ok_or_else(|| format!("run manifest is missing {key}"))?)?;
    }
    Ok(())
}

fn parse_key_values(text: &str, description: &str) -> Result<BTreeMap<String, String>, String> {
    let mut values = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') { continue; }
        let (key, value) = line.split_once('=').ok_or_else(|| format!("{description} line {} is malformed", index + 1))?;
        if key.is_empty() || value.is_empty() || values.insert(key.to_string(), value.to_string()).is_some() {
            return Err(format!("{description} has an empty or duplicate key at line {}", index + 1));
        }
    }
    Ok(values)
}

fn expect(values: &BTreeMap<String, String>, key: &str, expected: &str) -> Result<(), String> {
    let found = values.get(key).ok_or_else(|| format!("manifest is missing {key}"))?;
    if found != expected { return Err(format!("manifest {key} mismatch: expected {expected}, found {found}")); }
    Ok(())
}

fn parse_finite(value: &str, label: &str) -> Result<f64, String> {
    let parsed = value.parse::<f64>().map_err(|error| format!("invalid {label}: {error}"))?;
    if !parsed.is_finite() { return Err(format!("{label} must be finite")); }
    Ok(parsed)
}

fn parse_sha256(value: &str, label: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} must be 64 hexadecimal digits"));
    }
    let mut result = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        result[index] = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16)
            .map_err(|error| format!("invalid {label}: {error}"))?;
    }
    Ok(result)
}

fn parse_weights(bytes: &[u8]) -> Result<BTreeMap<String, Vec<f32>>, String> {
    const EXPECTED: [(&str, usize); 4] = [
        ("l0b", HIDDEN),
        ("l0w", HIDDEN * 768),
        ("l1b", 1),
        ("l1w", 2 * HIDDEN),
    ];
    if usize::BITS != 64 {
        return Err("Bullet weights.bin uses 64-bit usize lengths; exporter host must be 64-bit".into());
    }
    let mut offset = 0usize;
    let mut parsed = BTreeMap::new();
    let mut order = Vec::new();
    while offset < bytes.len() {
        let newline = bytes[offset..].iter().position(|&byte| byte == b'\n')
            .ok_or_else(|| "weights.bin has an unterminated tensor name".to_string())? + offset;
        if newline == offset { return Err("weights.bin contains an empty tensor name".into()); }
        let name = std::str::from_utf8(&bytes[offset..newline])
            .map_err(|error| format!("weights.bin tensor name is not UTF-8: {error}"))?.to_owned();
        if !name.is_ascii() { return Err("weights.bin tensor names must be ASCII".into()); }
        offset = newline + 1;
        let len_end = offset.checked_add(8).ok_or_else(|| "weights.bin offset overflow".to_string())?;
        if len_end > bytes.len() { return Err("weights.bin has a truncated tensor length".into()); }
        let count = u64::from_le_bytes(bytes[offset..len_end].try_into().unwrap());
        let count = usize::try_from(count).map_err(|_| "weights.bin tensor length exceeds usize")?;
        offset = len_end;
        let expected = EXPECTED.iter().find(|(candidate, _)| *candidate == name).map(|(_, count)| *count)
            .ok_or_else(|| format!("weights.bin contains unexpected tensor {name}"))?;
        if count != expected { return Err(format!("weights.bin tensor {name} must have {expected} floats, found {count}")); }
        let byte_count = count.checked_mul(4).ok_or_else(|| "weights.bin tensor byte count overflow".to_string())?;
        let end = offset.checked_add(byte_count).ok_or_else(|| "weights.bin offset overflow".to_string())?;
        if end > bytes.len() { return Err(format!("weights.bin tensor {name} is truncated")); }
        let values = bytes[offset..end].chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("weights.bin tensor {name} contains non-finite values"));
        }
        if parsed.insert(name.clone(), values).is_some() { return Err(format!("weights.bin duplicates tensor {name}")); }
        order.push(name);
        offset = end;
    }
    let expected_order: Vec<_> = EXPECTED.iter().map(|(name, _)| (*name).to_string()).collect();
    if order != expected_order { return Err(format!("weights.bin tensor order must be {expected_order:?}, found {order:?}")); }
    Ok(parsed)
}

struct FloatNetworkParts {
    feature_weights: Vec<f32>,
    hidden_bias: Vec<f32>,
    output_weights: Vec<f32>,
    output_bias: f32,
}

fn convert_tensors(tensors: &BTreeMap<String, Vec<f32>>) -> Result<FloatNetworkParts, String> {
    let l0b = tensors.get("l0b").ok_or_else(|| "missing l0b".to_string())?;
    let l0w = tensors.get("l0w").ok_or_else(|| "missing l0w".to_string())?;
    let l1b = tensors.get("l1b").ok_or_else(|| "missing l1b".to_string())?;
    let l1w = tensors.get("l1w").ok_or_else(|| "missing l1w".to_string())?;
    let mut feature_weights = vec![0.0; 768 * HIDDEN];
    for bullet_feature in 0..768 {
        let kira_feature = bullet_feature_to_kira(bullet_feature)
            .ok_or_else(|| format!("cannot map Bullet feature {bullet_feature}"))?;
        for hidden in 0..HIDDEN {
            // Bullet stores l0w as a (hidden, input) matrix in column-major
            // order, so each input feature's 128 hidden weights are adjacent.
            feature_weights[kira_feature * HIDDEN + hidden] = l0w[bullet_feature * HIDDEN + hidden];
        }
    }
    Ok(FloatNetworkParts {
        feature_weights,
        hidden_bias: l0b.clone(),
        output_weights: l1w.clone(), // Bullet concat_sides writes STM then NSTM.
        output_bias: l1b[0],
    })
}

fn atomic_write_new(output: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = output.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));
    if !parent.is_dir() { return Err(format!("output parent directory does not exist: {}", parent.display())); }
    if output.exists() { return Err(format!("refusing to overwrite existing output {}", output.display())); }
    let filename = output.file_name().ok_or_else(|| "output path has no filename".to_string())?.to_string_lossy();
    let (temporary, mut file) = loop {
        let serial = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{filename}.tmp-{}-{serial}", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(file) => break (candidate, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("create temporary export: {error}")),
        }
    };
    let result = (|| {
        file.write_all(bytes).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        drop(file);
        fs::hard_link(&temporary, output).map_err(|error| format!("publish NNUE without overwrite: {error}"))?;
        fs::remove_file(&temporary).map_err(io_error)?;
        File::open(parent).and_then(|directory| directory.sync_all()).map_err(io_error)?;
        Ok(())
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

fn io_error(error: std::io::Error) -> String { error.to_string() }

#[cfg(test)]
mod tests {
    use super::*;
    use kira1a::eval::nnue_quantized::QuantizedNnueNetwork;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let serial = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("kira-export-test-{}-{nanos}-{serial}", std::process::id()));
        fs::create_dir(&path).unwrap();
        path
    }

    fn fixture_weights() -> Vec<u8> {
        let mut tensors = BTreeMap::<String, Vec<f32>>::new();
        let mut l0b = vec![0.0; HIDDEN];
        l0b[0] = 0.125;
        let mut l0w = vec![0.0; HIDDEN * 768];
        // Bullet bishop row 2/square 7 becomes Kira bishop row 3/square 7.
        // Bullet's raw (hidden, feature) matrix is column-major/feature-major.
        l0w[(2 * 64 + 7) * HIDDEN] = 0.25;
        // Bullet rook row 3/square 11 becomes Kira rook row 2/square 11.
        l0w[(3 * 64 + 11) * HIDDEN + 1] = -0.5;
        let mut l1w = vec![0.0; 2 * HIDDEN];
        l1w[0] = 0.75;
        l1w[HIDDEN] = -0.625;
        tensors.insert("l0b".to_string(), l0b);
        tensors.insert("l0w".to_string(), l0w);
        tensors.insert("l1b".to_string(), vec![0.0625]);
        tensors.insert("l1w".to_string(), l1w);
        let mut bytes = Vec::new();
        for (name, values) in tensors {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(b'\n');
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values { bytes.extend_from_slice(&value.to_le_bytes()); }
        }
        bytes
    }

    fn run_manifest() -> String {
        let mut result = String::new();
        for (key, value) in [
            ("trainer_format", TRAINER_FORMAT),
            ("bullet_root", "/synthetic/pinned-bullet"),
            ("bullet_commit", BULLET_COMMIT),
            ("bullet_cargo_lock_sha256", &"a".repeat(64)),
            ("trainer_source_path", "driver/src/main.rs"),
            ("trainer_source_sha256", &"b".repeat(64)),
            ("run_driver_path", "run_driver.py"),
            ("run_driver_sha256", &"c".repeat(64)),
            ("runtime_cargo_manifest_sha256", &"d".repeat(64)),
            ("runtime_cargo_lock_sha256", &"e".repeat(64)),
            ("adapter_manifest_sha256", &"f".repeat(64)),
            ("m3_dataset_sha256", &"1".repeat(64)),
            ("m3_config_sha256", &"2".repeat(64)),
            ("m3_source_tree_sha256", &"3".repeat(64)),
            ("m3_source_manifest_sha256", &"4".repeat(64)),
            ("m3_source_revision", "synthetic-fixture"),
            ("train_sha256", &"5".repeat(64)),
            ("validation_sha256", &"6".repeat(64)),
            ("architecture", ARCHITECTURE),
            ("optimizer", OPTIMIZER),
            ("batch_size", "16384"),
            ("seed", "0x1234"),
            ("score_lambda", SCORE_CONFIG),
        ] {
            result.push_str(key);
            result.push('=');
            result.push_str(value);
            result.push('\n');
        }
        result
    }

    fn create_run(path: &Path) {
        fs::create_dir(path).unwrap();
        fs::write(path.join("run_manifest.txt"), run_manifest()).unwrap();
        fs::write(
            path.join("losses.csv"),
            concat!(
                "epoch,learning_rate,train_mse,validation_mse,validation_records,best_epoch,stale_epochs\n",
                "1,1.0e-3,0.10,0.20,100,1,0\n",
                "2,1.0e-4,0.09,0.15,100,2,0\n",
                "# best_epoch=2 best_validation_mse=0.1500000000\n",
            ),
        ).unwrap();
        for (epoch, lr, train_mse, mse) in [
            (1, "1.0000000000e-3", "0.100000000000", "0.200000000000"),
            (2, "1.0000000000e-4", "0.090000000000", "0.150000000000"),
        ] {
            let checkpoint = path.join(format!("checkpoint-{epoch:04}"));
            fs::create_dir(&checkpoint).unwrap();
            fs::write(checkpoint.join("manifest.txt"), format!(
                "epoch={epoch}\nlearning_rate={lr}\ntrain_mse={train_mse}\nvalidation_mse={mse}\ntrain_sha256={}\nvalidation_sha256={}\n",
                "5".repeat(64), "6".repeat(64),
            )).unwrap();
            fs::write(checkpoint.join("weights.bin"), fixture_weights()).unwrap();
        }
    }

    #[test]
    fn selects_best_epoch_converts_feature_rows_and_preserves_heads_and_hashes() {
        let root = temp_dir();
        let run = root.join("run");
        let output = root.join("model.nnue");
        create_run(&run);
        let result = export_best_checkpoint(&run, &output).unwrap();
        assert_eq!(result.epoch, 2);
        let network = QuantizedNnueNetwork::<HIDDEN>::load(&output).unwrap();
        assert_eq!(network.feature_weights()[(3 * 64 + 7) * HIDDEN], 4096);
        assert_eq!(network.feature_weights()[((2 * 64 + 11) * HIDDEN) + 1], -8192);
        assert_eq!(network.hidden_bias()[0], 2048);
        assert_eq!(network.output_weights()[0][0], 12288);
        assert_eq!(network.output_weights()[1][0], -10240);
        assert_eq!(network.output_bias(), 1024);
        assert_eq!(network.metadata().source_weights_sha256, parse_sha256(&sha256_hex(&fixture_weights()), "fixture").unwrap());
        assert_eq!(result.report.clamped_total(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accepts_sanitized_trainer_format_only_with_pinned_source_provenance() {
        let mut values = parse_key_values(&run_manifest(), "test run manifest").unwrap();
        values.insert("trainer_format".into(), SANITIZED_TRAINER_FORMAT.into());
        values.insert("bullet_patch_sha256".into(), BULLET_PATCH_SHA256.into());
        values.insert("bullet_source_tree_sha256".into(), "a".repeat(64));
        values.insert("bullet_source_manifest_sha256".into(), "a".repeat(64));
        values.insert("bullet_dependency_tree_sha256".into(), "b".repeat(64));
        validate_trainer_format(&values).unwrap();

        values.insert("bullet_source_manifest_sha256".into(), "c".repeat(64));
        assert!(validate_trainer_format(&values).unwrap_err().contains("hashes differ"));
        values.insert("bullet_source_manifest_sha256".into(), "a".repeat(64));
        values.insert("bullet_patch_sha256".into(), "0".repeat(64));
        assert!(validate_trainer_format(&values).unwrap_err().contains("bullet_patch_sha256 mismatch"));
    }

    #[test]
    fn refuses_invalid_best_epoch_manifest_and_existing_output() {
        let root = temp_dir();
        let run = root.join("run");
        let output = root.join("model.nnue");
        create_run(&run);
        fs::write(run.join("losses.csv"), "epoch,learning_rate,train_mse,validation_mse,validation_records,best_epoch,stale_epochs\n# best_epoch=3 best_validation_mse=0.1\n").unwrap();
        assert!(export_best_checkpoint(&run, &output).unwrap_err().contains("no losses.csv row"));
        fs::write(run.join("losses.csv"), concat!(
            "epoch,learning_rate,train_mse,validation_mse,validation_records,best_epoch,stale_epochs\n",
            "1,1e-3,0.1,0.2,100,1,0\n",
            "2,1e-4,0.09,0.15,100,2,0\n",
            "# best_epoch=2 best_validation_mse=0.1500000000\n",
        )).unwrap();
        fs::write(&output, b"do not overwrite").unwrap();
        assert!(export_best_checkpoint(&run, &output).unwrap_err().contains("refusing to overwrite"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn weights_parser_rejects_nonfinite_duplicate_truncated_and_extra_tensors() {
        let valid = fixture_weights();
        let mut nonfinite = valid.clone();
        let name_end = nonfinite.iter().position(|&byte| byte == b'\n').unwrap();
        let first = name_end + 1 + 8;
        nonfinite[first..first + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(parse_weights(&nonfinite).unwrap_err().contains("non-finite"));
        assert!(parse_weights(&valid[..valid.len() - 1]).is_err());
        let mut extra = valid.clone();
        extra.extend_from_slice(b"unexpected\n");
        extra.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse_weights(&extra).unwrap_err().contains("unexpected tensor"));
        let mut duplicate = valid.clone();
        duplicate.extend_from_slice(b"l0b\n");
        duplicate.extend_from_slice(&(HIDDEN as u64).to_le_bytes());
        duplicate.extend_from_slice(&vec![0u8; HIDDEN * 4]);
        assert!(parse_weights(&duplicate).unwrap_err().contains("duplicates tensor"));
    }
}
