use kira1a::datagen::{self, DatagenConfig};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("kira1a-datagen: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let process_args: Vec<String> = std::env::args().skip(1).collect();
    if process_args.first().is_some_and(|arg| arg == "--inspect") {
        if process_args.len() != 2 {
            return Err("usage: kira1a-datagen --inspect ARTIFACT".into());
        }
        let summary = datagen::inspect_artifact(PathBuf::from(&process_args[1]).as_path(), |_| {})?;
        println!("kind={}", summary.kind);
        println!("sha256={}", summary.sha256);
        println!("config_sha256={}", summary.config_sha256);
        println!("games={}", summary.games);
        println!("eligible_pre_dedup={}", summary.eligible_pre_dedup);
        println!("stored_candidate_rows={}", summary.stored_candidate_rows);
        println!("search_events={}", summary.search_events);
        println!("search_evaluator={}", summary.search_evaluator);
        if let Some(sha256) = &summary.nnue_sha256 { println!("nnue_sha256={sha256}"); }
        println!("duplicates={}", summary.duplicates);
        println!("unique_positions={}", summary.unique_positions);
        if let Some(worker) = summary.worker { println!("worker={worker}"); }
        if let Some(nodes) = summary.node_limit { println!("node_limit={nodes}"); }
        return Ok(());
    }

    let internal_worker = process_args.first().is_some_and(|arg| arg == "--_worker-internal");
    let visible_args = if internal_worker { &process_args[1..] } else { &process_args[..] };

    let mut output = None;
    let mut games = None;
    let mut base_seed = None;
    let mut start_game_id = 0u64;
    let mut nodes = 5_000u64;
    let mut hash_mb = 16usize;
    let mut workers = std::thread::available_parallelism().map_or(1, usize::from);
    let mut chunk_games = 32usize;
    let mut worker_index = None;
    let mut worker_spool = None;
    let mut expected_engine_sha256 = None;
    let mut expected_source_sha256 = None;
    let mut expected_config_sha256 = None;
    let mut expected_evaluator_kind = None;
    let mut expected_nnue_sha256 = None;
    let mut eval_file = None;

    let mut args = visible_args.iter().cloned();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--out" => output = Some(PathBuf::from(take_value(&mut args, "--out")?)),
            "--games" => games = Some(parse_u64(&take_value(&mut args, "--games")?, "--games")?),
            "--base-seed" => base_seed = Some(parse_u64(&take_value(&mut args, "--base-seed")?, "--base-seed")?),
            "--start-id" => start_game_id = parse_u64(&take_value(&mut args, "--start-id")?, "--start-id")?,
            "--nodes" => nodes = parse_u64(&take_value(&mut args, "--nodes")?, "--nodes")?,
            "--eval-file" => eval_file = Some(PathBuf::from(take_value(&mut args, "--eval-file")?)),
            "--hash-mb" => hash_mb = parse_usize(&take_value(&mut args, "--hash-mb")?, "--hash-mb")?,
            "--workers" => workers = parse_usize(&take_value(&mut args, "--workers")?, "--workers")?,
            "--chunk-games" => chunk_games = parse_usize(&take_value(&mut args, "--chunk-games")?, "--chunk-games")?,
            "--_worker-index" if internal_worker => worker_index = Some(parse_usize(&take_value(&mut args, "--_worker-index")?, "worker index")?),
            "--_worker-spool" if internal_worker => worker_spool = Some(PathBuf::from(take_value(&mut args, "--_worker-spool")?)),
            "--_expected-engine-sha256" if internal_worker => expected_engine_sha256 = Some(take_value(&mut args, "--_expected-engine-sha256")?),
            "--_expected-source-sha256" if internal_worker => expected_source_sha256 = Some(take_value(&mut args, "--_expected-source-sha256")?),
            "--_expected-config-sha256" if internal_worker => expected_config_sha256 = Some(take_value(&mut args, "--_expected-config-sha256")?),
            "--_expected-evaluator-kind" if internal_worker => expected_evaluator_kind = Some(take_value(&mut args, "--_expected-evaluator-kind")?),
            "--_expected-nnue-sha256" if internal_worker => expected_nnue_sha256 = Some(take_value(&mut args, "--_expected-nnue-sha256")?),
            other => return Err(format!("unknown argument {other}; pass --help for usage")),
        }
    }

    let using_nnue = eval_file.is_some();
    let config = DatagenConfig {
        output: output.ok_or_else(|| "--out is required".to_string())?,
        start_game_id,
        games: games.ok_or_else(|| "--games is required".to_string())?,
        base_seed: base_seed.ok_or_else(|| "--base-seed is required".to_string())?,
        nodes,
        hash_mb,
        workers,
        chunk_games,
        eval_file,
    };
    if internal_worker {
        return datagen::run_worker_child(
            config,
            worker_index.ok_or_else(|| "worker subprocess lacks --_worker-index".to_string())?,
            worker_spool.as_deref().ok_or_else(|| "worker subprocess lacks --_worker-spool".to_string())?,
            expected_engine_sha256.as_deref().ok_or_else(|| "worker subprocess lacks expected engine hash".to_string())?,
            expected_source_sha256.as_deref().ok_or_else(|| "worker subprocess lacks expected source hash".to_string())?,
            expected_config_sha256.as_deref().ok_or_else(|| "worker subprocess lacks expected config hash".to_string())?,
            expected_evaluator_kind.as_deref().ok_or_else(|| "worker subprocess lacks expected evaluator kind".to_string())?,
            expected_nnue_sha256.as_deref().ok_or_else(|| "worker subprocess lacks expected NNUE hash".to_string())?,
        );
    }
    let summary = datagen::run(config)?;
    println!("games={}", summary.games);
    println!("eligible_pre_dedup={}", summary.eligible_pre_dedup);
    println!("duplicate_candidates={}", summary.duplicate_candidates);
    println!("unique_positions={}", summary.unique_positions);
    println!("duplicate_rate={:.12}", if summary.eligible_pre_dedup == 0 {
        0.0
    } else {
        summary.duplicate_candidates as f64 / summary.eligible_pre_dedup as f64
    });
    println!("dataset={}", summary.dataset.display());
    println!("dataset_sha256={}", summary.dataset_sha256);
    println!("search_evaluator={}", if using_nnue { "QuantizedNnueNetwork<128>" } else { "Kira-HCE" });
    Ok(())
}

fn take_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{name} requires a value"))
}

fn parse_u64(text: &str, name: &str) -> Result<u64, String> {
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|_| format!("invalid {name} value {text}"))
    } else {
        text.parse().map_err(|_| format!("invalid {name} value {text}"))
    }
}

fn parse_usize(text: &str, name: &str) -> Result<usize, String> {
    text.parse().map_err(|_| format!("invalid {name} value {text}"))
}

fn print_help() {
    println!("Kira 1 Alpha self-play datagen\n\nUsage:\n  kira1a-datagen --out DIR --games N --base-seed U64 [options]\n  kira1a-datagen --inspect ARTIFACT\n\nOptions:\n  --start-id N       First game ID (default: 0)\n  --nodes N          Search node limit per move (default: 5000)\n  --eval-file PATH   Quantized NNUE search evaluator (128 hidden units); opening gate remains HCE\n  --hash-mb N        Hash MB per worker (default: 16)\n  --workers N        Fixed worker count (default: logical CPUs)\n  --chunk-games N    Whole games per raw chunk (default: 32)\n  --help             Show this help\n\nIf DIR contains a matching manifest, committed chunks are verified and\nuncommitted game IDs are resumed. Settings, evaluator, and executable/source hashes must match.\nExact source/executable pins intentionally make cross-source-cut resume impossible;\nproduction gen2 should use a fresh output directory. --inspect requires the\nartifact's adjacent manifest.\nTest-only recovery failpoints (use one worker/chunk): set\nKIRA1A_DATAGEN_TEST_FAILPOINT to before-temp-write, during-temp-write,\nafter-temp-fsync, after-chunk-rename, after-manifest-commit, or\nbefore-spool-create-worker-N (N is the worker index).");
}
