//! Deterministic, independent 1,000-record audit over a bounded scratch set.
//! Set `M3_AUDIT_DATASET` and `M3_AUDIT_REPORT` when invoking this test.

use kira1a::datagen::{self, DecodedCandidate, DecodedGame};
use kira1a::Board;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::PathBuf;

const SAMPLE_SIZE: usize = 1_000;
const AUDIT_SEED: u64 = 0x4d33_4155_4449_545f;

#[derive(Clone)]
struct Row {
    game_id: u64,
    candidate: DecodedCandidate,
    game_plies: usize,
}

fn stm_label(result_white: u8, white_to_move: bool) -> u8 {
    match (result_white, white_to_move) {
        (0, true) | (2, false) => 0,
        (1, _) => 1,
        (2, true) | (0, false) => 2,
        (other, _) => panic!("invalid final result code {other}"),
    }
}

fn xorshift64star(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn uniform_below(state: &mut u64, bound: usize) -> usize {
    assert!(bound > 0);
    let bound = bound as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let word = xorshift64star(state);
        if word >= threshold {
            return (word % bound) as usize;
        }
    }
}

fn position_count(fen: &str) -> usize {
    fen.split_whitespace()
        .next()
        .expect("FEN placement missing")
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .count()
}

fn score_bucket(score: i32) -> &'static str {
    match score {
        i32::MIN..=-1_001 => "<-1000",
        -1_000..=-501 => "-1000..-501",
        -500..=-101 => "-500..-101",
        -100..=100 => "-100..100",
        101..=500 => "101..500",
        501..=1_000 => "501..1000",
        _ => "1001..2000",
    }
}

#[test]
fn audit_exactly_1000_final_records_without_replacement() {
    let dataset = PathBuf::from(std::env::var("M3_AUDIT_DATASET").expect("set M3_AUDIT_DATASET"));
    let report_path = PathBuf::from(std::env::var("M3_AUDIT_REPORT").expect("set M3_AUDIT_REPORT"));
    let mut games = Vec::<DecodedGame>::new();
    let summary = datagen::inspect_artifact(&dataset, |game| games.push(game.clone()))
        .unwrap_or_else(|e| panic!("failed to decode scratch dataset {}: {e}", dataset.display()));
    assert_eq!(summary.kind, "merged-dataset");
    assert!(summary.unique_positions >= SAMPLE_SIZE as u64,
        "only {} unique rows; cannot sample {SAMPLE_SIZE}", summary.unique_positions);
    assert_eq!(summary.stored_candidate_rows, summary.unique_positions);

    let mut rows = Vec::<Row>::new();
    let mut hash_set = HashSet::new();
    for game in &games {
        let mut board = Board::from_fen(&game.start_fen)
            .unwrap_or_else(|e| panic!("invalid start FEN for game {}: {e}", game.game_id));
        assert_eq!(board.to_fen(), game.start_fen);
        let mut states = vec![board.clone()];
        for (ply, mv) in game.moves.iter().enumerate() {
            board.play_uci(mv).unwrap_or_else(|e| {
                panic!("illegal history move {mv} at ply {ply}, game {}: {e}", game.game_id)
            });
            states.push(board.clone());
        }
        for candidate in &game.candidates {
            assert!(candidate.ply as usize <= game.moves.len());
            let state = &states[candidate.ply as usize];
            assert_eq!(state.to_fen(), candidate.fen, "FEN round-trip/replay mismatch");
            assert_eq!(state.hash(), candidate.zobrist, "replayed Zobrist mismatch");
            assert_eq!(state.white_to_move, candidate.white_to_move, "replayed STM mismatch");
            assert_eq!(candidate.result_stm, stm_label(game.result_white, candidate.white_to_move));
            assert!(hash_set.insert(candidate.zobrist), "merged dataset contains a duplicate Zobrist");
            rows.push(Row {
                game_id: game.game_id,
                candidate: candidate.clone(),
                game_plies: game.moves.len(),
            });
        }
    }
    assert_eq!(rows.len() as u64, summary.unique_positions);
    assert_eq!(hash_set.len(), rows.len());
    rows.sort_by_key(|row| (row.game_id, row.candidate.ply, row.candidate.zobrist));

    // Verifier-owned xorshift64* stream and rejection sampling; partial
    // Fisher-Yates gives exactly 1,000 distinct rows from the whole dataset.
    let mut state = AUDIT_SEED;
    let mut indexes: Vec<usize> = (0..rows.len()).collect();
    for i in 0..SAMPLE_SIZE {
        let j = i + uniform_below(&mut state, rows.len() - i);
        indexes.swap(i, j);
    }
    let sample_indexes = &indexes[..SAMPLE_SIZE];
    let mut sampled_identity = HashSet::new();
    let mut results = [0u64; 3];
    let mut side_results = [[0u64; 3]; 2];
    let mut score_hist = BTreeMap::<&'static str, u64>::new();
    let mut length_hist = BTreeMap::<&'static str, u64>::new();
    let mut piece_hist = BTreeMap::<usize, u64>::new();
    let mut xs = Vec::<f64>::with_capacity(SAMPLE_SIZE);
    let mut ys = Vec::<f64>::with_capacity(SAMPLE_SIZE);
    let mut selected = Vec::<String>::with_capacity(SAMPLE_SIZE);

    for &index in sample_indexes {
        let row = &rows[index];
        let candidate = &row.candidate;
        assert!(sampled_identity.insert((row.game_id, candidate.ply, candidate.zobrist)));
        let result = candidate.result_stm as usize;
        assert!(result <= 2);
        results[result] += 1;
        side_results[usize::from(!candidate.white_to_move)][result] += 1;
        *score_hist.entry(score_bucket(candidate.score_stm)).or_default() += 1;
        *piece_hist.entry(position_count(&candidate.fen)).or_default() += 1;
        let length_bucket = match row.game_plies {
            0..=79 => "0..79",
            80..=119 => "80..119",
            120..=159 => "120..159",
            _ => ">=160",
        };
        *length_hist.entry(length_bucket).or_default() += 1;
        xs.push(candidate.score_stm as f64);
        ys.push(match result { 0 => 0.0, 1 => 0.5, 2 => 1.0, _ => unreachable!() });
        selected.push(format!("{},{},{:016x}", row.game_id, candidate.ply, candidate.zobrist));
    }
    assert_eq!(sampled_identity.len(), SAMPLE_SIZE);

    let mean_x = xs.iter().sum::<f64>() / SAMPLE_SIZE as f64;
    let mean_y = ys.iter().sum::<f64>() / SAMPLE_SIZE as f64;
    let covariance = xs.iter().zip(&ys).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum::<f64>();
    let variance_x = xs.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    let variance_y = ys.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    let correlation = if variance_x == 0.0 || variance_y == 0.0 {
        None
    } else {
        Some(covariance / (variance_x * variance_y).sqrt())
    };

    let mut report = format!(
        "M3 verifier-only scratch sample audit\nsource_artifact={}\nartifact_sha256={}\nconfig_sha256={}\nkind={}\ngames={}\neligible_pre_dedup={}\nduplicates={}\nunique_positions={}\nsample_seed=0x{AUDIT_SEED:016x}\nsampling=verifier-owned xorshift64*; rejection-sampled partial Fisher-Yates; no replacement\nsample_size={}\nresult_stm_counts=loss:{},draw:{},win:{}\nresult_by_stm=white:{:?};black:{:?}\nscore_result_pearson_r={}\nscore_histogram={:?}\ngame_length_ply_histogram={:?}\npiece_count_histogram={:?}\n",
        dataset.display(), summary.sha256, summary.config_sha256, summary.kind,
        summary.games, summary.eligible_pre_dedup, summary.duplicates,
        summary.unique_positions, SAMPLE_SIZE, results[0], results[1], results[2],
        side_results[0], side_results[1],
        correlation.map_or_else(|| "undefined (zero variance)".to_owned(), |r| format!("{r:.8}")),
        score_hist, length_hist, piece_hist,
    );
    report.push_str("sample_rows=game_id,ply,zobrist_hex\n");
    for row in selected {
        report.push_str(&row);
        report.push('\n');
    }
    fs::write(&report_path, report).unwrap_or_else(|e| {
        panic!("could not write audit report {}: {e}", report_path.display())
    });
    println!("M3_AUDIT_REPORT={}", report_path.display());
    println!("M3_AUDIT_CORRELATION={}", correlation.map_or_else(|| "undefined".to_owned(), |r| format!("{r:.8}")));
}
