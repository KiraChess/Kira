use kira_m4_adapter::{
    NNUE_FEATURE_KEY_SCHEMA, split_artifact, split_generation_artifacts, DEFAULT_SPLIT_SEED,
};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("kira-m4-adapter: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut input = None;
    let mut gen2_input = None;
    let mut gen1_input = None;
    let mut output = None;
    let mut bullet_root = None;
    let mut seed = DEFAULT_SPLIT_SEED;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                println!("Usage: kira-m4-adapter --input MERGED_V2.k1d --out NEW_DIRECTORY --bullet-root BULLET_CHECKOUT [--seed U64]");
                println!("   or: kira-m4-adapter --gen2 GEN2_V2.k1d --gen1 GEN1_V2.k1d --out NEW_DIRECTORY --bullet-root BULLET_CHECKOUT [--seed U64]");
                println!("--input is the unchanged M4 single-source path. The gen2/gen1 pair uses the M6 global split and gen2-preferred exact NNUE feature-key deduplication.");
                println!("M6 feature key schema: {NNUE_FEATURE_KEY_SCHEMA}");
                println!("Default split seed: 0x{DEFAULT_SPLIT_SEED:016x}");
                println!("--bullet-root may be omitted when KIRA_M4_BULLET_ROOT is set.");
                return Ok(());
            }
            "--input" => input = Some(PathBuf::from(next_value(&mut args, "--input")?)),
            "--gen2" => gen2_input = Some(PathBuf::from(next_value(&mut args, "--gen2")?)),
            "--gen1" => gen1_input = Some(PathBuf::from(next_value(&mut args, "--gen1")?)),
            "--out" => output = Some(PathBuf::from(next_value(&mut args, "--out")?)),
            "--bullet-root" => bullet_root = Some(PathBuf::from(next_value(&mut args, "--bullet-root")?)),
            "--seed" => seed = parse_seed(&next_value(&mut args, "--seed")?)?,
            other => return Err(format!("unknown argument {other}; use --help")),
        }
    }
    let output = output.ok_or_else(|| "--out is required".to_string())?;
    let bullet_root = bullet_root
        .or_else(|| std::env::var_os("KIRA_M4_BULLET_ROOT").map(PathBuf::from))
        .ok_or_else(|| "--bullet-root is required unless KIRA_M4_BULLET_ROOT is set".to_string())?;
    match (input, gen2_input, gen1_input) {
        (Some(input), None, None) => {
            let summary = split_artifact(&input, &output, seed, &bullet_root)?;
            println!("source_sha256={}", summary.source_sha256);
            println!("output={}", summary.output.display());
            println!("train_games={}", summary.train_games);
            println!("validation_games={}", summary.validation_games);
            println!("train_records={}", summary.train_records);
            println!("validation_records={}", summary.validation_records);
        }
        (None, Some(gen2), Some(gen1)) => {
            let summary = split_generation_artifacts(&gen2, &gen1, &output, seed, &bullet_root)?;
            println!("source_gen2_sha256={}", summary.gen2_sha256);
            println!("source_gen1_sha256={}", summary.gen1_sha256);
            println!("output={}", summary.output.display());
            println!("raw_records={}", summary.raw_records);
            println!("kept_records={}", summary.kept_records);
            println!("dropped_duplicate_records={}", summary.dropped_duplicate_records);
            println!("source_gen2_raw_records={}", summary.gen2_raw_records);
            println!("source_gen2_kept_records={}", summary.gen2_kept_records);
            println!("source_gen2_dropped_duplicate_records={}", summary.gen2_dropped_duplicate_records);
            println!("source_gen2_cross_source_duplicate_records={}", summary.gen2_cross_source_duplicate_records);
            println!("source_gen1_raw_records={}", summary.gen1_raw_records);
            println!("source_gen1_kept_records={}", summary.gen1_kept_records);
            println!("source_gen1_dropped_duplicate_records={}", summary.gen1_dropped_duplicate_records);
            println!("within_source_duplicate_records={}", summary.within_source_duplicate_records);
            println!("cross_source_duplicate_records={}", summary.cross_source_duplicate_records);
            println!("source_gen2_zero_row_games={}", summary.gen2_zero_row_games);
            println!("source_gen1_zero_row_games={}", summary.gen1_zero_row_games);
            println!("zero_row_games={}", summary.zero_row_games);
            println!("train_games={}", summary.train_games);
            println!("validation_games={}", summary.validation_games);
            println!("train_records={}", summary.train_records);
            println!("validation_records={}", summary.validation_records);
        }
        _ => return Err("provide either --input alone or both --gen2 and --gen1".into()),
    }
    Ok(())
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_seed(value: &str) -> Result<u64, String> {
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|error| format!("invalid hexadecimal seed: {error}"))
    } else {
        value.parse::<u64>().map_err(|error| format!("invalid decimal seed: {error}"))
    }
}
