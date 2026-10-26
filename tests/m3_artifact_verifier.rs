//! Black-box M3 artifact and recovery checks.
//!
//! The datagen CLI is invoked as a separate process so its child-worker path,
//! manifest ownership, and failpoints are exercised exactly as shipped. All
//! generated files stay beneath `/private/tmp`; no production dataset is used.

use kira1a::datagen::{self, BestMoveClass, DecodedCandidate, DecodedGame, FilterReason};
use kira1a::{Board, Move};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SCRATCH_PARENT: &str = "/private/tmp";
const CORE_NODES: u64 = 8;
const CORE_GAMES: u64 = 4;
const CORE_START_ID: u64 = 101;
const CORE_SEED: u64 = 0x4d33_5645_5249_4659;

fn unique_root(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();
    Path::new(SCRATCH_PARENT).join(format!(
        "kira1a-m3-verifier-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn datagen_binary() -> &'static str {
    env!("CARGO_BIN_EXE_kira1a-datagen")
}

fn invoke(
    output_dir: &Path,
    games: u64,
    start_id: u64,
    base_seed: u64,
    nodes: u64,
    workers: usize,
    chunk_games: usize,
    failpoint: Option<&str>,
) -> Output {
    let mut command = Command::new(datagen_binary());
    command
        .arg("--out")
        .arg(output_dir)
        .arg("--games")
        .arg(games.to_string())
        .arg("--start-id")
        .arg(start_id.to_string())
        .arg("--base-seed")
        .arg(base_seed.to_string())
        .arg("--nodes")
        .arg(nodes.to_string())
        .arg("--hash-mb")
        .arg("1")
        .arg("--workers")
        .arg(workers.to_string())
        .arg("--chunk-games")
        .arg(chunk_games.to_string());
    if let Some(value) = failpoint {
        command.env("KIRA1A_DATAGEN_TEST_FAILPOINT", value);
    } else {
        command.env_remove("KIRA1A_DATAGEN_TEST_FAILPOINT");
    }
    command.output().expect("failed to start datagen CLI")
}

fn stdout_text(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("CLI stdout must be UTF-8")
}

fn stderr_text(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("CLI stderr must be UTF-8")
}

fn key_values(text: &str) -> HashMap<String, String> {
    text.lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect()
}

fn sha256(path: &Path) -> String {
    let output = Command::new("shasum")
        .arg("-a")
        .arg("256")
        .arg(path)
        .output()
        .expect("shasum must be installed locally");
    assert!(output.status.success(), "shasum failed: {}", stderr_text(&output));
    stdout_text(&output)
        .split_whitespace()
        .next()
        .expect("shasum output missing digest")
        .to_owned()
}

fn inspect(path: &Path) -> (datagen::ArtifactSummary, Vec<DecodedGame>) {
    let mut games = Vec::new();
    let summary = datagen::inspect_artifact(path, |game| games.push(game.clone()))
        .unwrap_or_else(|e| panic!("inspect_artifact failed for {}: {e}", path.display()));
    (summary, games)
}

fn replay_states(game: &DecodedGame) -> Vec<Board> {
    let mut board = Board::from_fen(&game.start_fen)
        .unwrap_or_else(|e| panic!("invalid recorded start FEN for game {}: {e}", game.game_id));
    assert_eq!(board.to_fen(), game.start_fen, "start FEN was not lossless");
    let mut states = vec![board.clone()];
    for (ply, mv) in game.moves.iter().enumerate() {
        board.play_uci(mv).unwrap_or_else(|e| {
            panic!("illegal recorded move {} at ply {ply} in game {}: {e}", mv, game.game_id)
        });
        states.push(board.clone());
    }
    states
}

fn stm_label(result_white: u8, white_to_move: bool) -> u8 {
    match (result_white, white_to_move) {
        (0, true) | (2, false) => 0,
        (1, _) => 1,
        (2, true) | (0, false) => 2,
        (other, _) => panic!("invalid final result code {other}"),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PositionTuple {
    game_id: u64,
    ply: u32,
    zobrist: u64,
    fen: String,
    white_to_move: bool,
    score_stm: i32,
    result_stm: u8,
    class: BestMoveClass,
}

fn position_tuple(game_id: u64, candidate: &DecodedCandidate) -> PositionTuple {
    PositionTuple {
        game_id,
        ply: candidate.ply,
        zobrist: candidate.zobrist,
        fen: candidate.fen.clone(),
        white_to_move: candidate.white_to_move,
        score_stm: candidate.score_stm,
        result_stm: candidate.result_stm,
        class: candidate.best_move_class,
    }
}

fn audit_game(game: &DecodedGame, node_limit: u64, raw: bool) -> (Vec<PositionTuple>, Vec<bool>) {
    let states = replay_states(game);
    assert!(game.moves.len() >= 8, "generation must retain the eight-ply opening");
    assert!((-300..=300).contains(&game.accepted_opening_hce),
        "accepted HCE {} lies outside [-300,+300]", game.accepted_opening_hce);
    assert!(!game.moves.is_empty());

    let mut seen_colors = Vec::new();
    let mut kept_events = BTreeMap::<u32, &datagen::DecodedSearchEvent>::new();
    let mut streaks = datagen::ScoreStreaks::default();
    let mut adjudicated_end = None;
    let history_hashes: Vec<u64> = states.iter().map(Board::hash).collect();
    for event in &game.search_events {
        let board = states.get(event.ply as usize).unwrap_or_else(|| {
            panic!("event ply {} exceeds game {} history", event.ply, game.game_id)
        });
        assert_eq!(board.to_fen(), event.fen, "event FEN mismatch at game {} ply {}", game.game_id, event.ply);
        assert_eq!(board.hash(), event.zobrist, "event Zobrist mismatch at game {} ply {}", game.game_id, event.ply);
        assert_eq!(board.white_to_move, event.white_to_move, "event STM mismatch");
        assert!(event.ply >= 8, "opening plies must not be searched for candidates");
        assert_eq!(game.moves.get(event.ply as usize), Some(&event.best_move_uci),
            "game history move differs from traced bestmove at ply {}", event.ply);
        assert_eq!(event.node_limit, node_limit, "search must receive pinned node limit");
        assert_eq!(event.depth_limit, None, "depth limit must not replace node limit");
        assert_eq!(event.time_limit_ms, None, "time limit must not replace node limit");
        assert!(event.search_nodes > 0, "search event should record completed nodes");
        seen_colors.push(event.white_to_move);
        let mv = Move::from_uci(&event.best_move_uci).unwrap_or_else(|| {
            panic!("invalid bestmove UCI {}", event.best_move_uci)
        });
        assert_eq!(datagen::classify_best_move(board, mv), event.best_move_class);
        let mut after_best = board.clone();
        after_best.play_uci(&event.best_move_uci).unwrap_or_else(|e| {
            panic!("illegal trace bestmove {}: {e}", event.best_move_uci)
        });
        let expected_reason = datagen::candidate_filter_reason(
            board.is_in_check(board.white_to_move),
            event.best_move_class,
            event.score_stm,
        );
        assert_eq!(event.filter_reason, expected_reason, "raw filter reason mismatch");
        if event.filter_reason == FilterReason::Kept {
            assert!(kept_events.insert(event.ply, event).is_none(), "multiple searches recorded at one game ply");
        }
        if raw {
            let pending_win = datagen::observe_score(
                &mut streaks,
                event.ply,
                event.score_stm,
                event.white_to_move,
            );
            let completed_plies = event.ply + 1;
            let mut after = states[completed_plies as usize].clone();
            let outcome = datagen::adjudicate_after_move(
                &mut after,
                &history_hashes[..=completed_plies as usize],
                pending_win,
                &streaks,
                completed_plies,
            );
            if let Some(outcome) = outcome {
                assert_eq!(completed_plies as usize, game.moves.len(), "game continued past an adjudicated terminal");
                adjudicated_end = Some(outcome);
            }
        }
    }

    let mut rows = Vec::new();
    for candidate in &game.candidates {
        let board = states.get(candidate.ply as usize).unwrap_or_else(|| {
            panic!("candidate ply {} exceeds game {} history", candidate.ply, game.game_id)
        });
        assert!(candidate.ply >= 8, "candidate emitted before eight opening plies");
        assert_eq!(board.to_fen(), candidate.fen, "candidate FEN mismatch");
        assert_eq!(board.white_to_move, candidate.white_to_move, "candidate STM mismatch");
        assert_eq!(board.hash(), candidate.zobrist, "candidate hash mismatch");
        assert_eq!(candidate.result_stm, stm_label(game.result_white, candidate.white_to_move));
        if raw {
            let event = kept_events.get(&candidate.ply).expect("candidate has no kept raw search event");
            assert_eq!(event.fen, candidate.fen);
            assert_eq!(event.score_stm, candidate.score_stm);
            assert_eq!(event.best_move_class, candidate.best_move_class);
            assert_eq!(event.zobrist, candidate.zobrist);
        }
        rows.push(position_tuple(game.game_id, candidate));
    }
    if raw {
        assert_eq!(game.eligible_count_pre_dedup, game.candidates.len() as u64);
        assert_eq!(game.candidates.len(), kept_events.len(), "raw candidate rows must include every kept event");
        let outcome = adjudicated_end.unwrap_or_else(|| {
            let mut final_board = states.last().unwrap().clone();
            let natural = datagen::classify_natural_terminal(&mut final_board, &history_hashes)
                .expect("raw game ended without natural or score adjudication");
            datagen::PostMoveOutcome::Natural(natural)
        });
        assert_recorded_outcome(game, outcome);
    }
    (rows, seen_colors)
}

fn assert_recorded_outcome(game: &DecodedGame, outcome: datagen::PostMoveOutcome) {
    use datagen::{GameEndReason, NaturalReason, NaturalResult, PostMoveOutcome, ScoreDecision};
    match outcome {
        PostMoveOutcome::Natural(natural) => {
            let result_white = match natural.result {
                NaturalResult::BlackWin => 0,
                NaturalResult::Draw => 1,
                NaturalResult::WhiteWin => 2,
            };
            let reason = match natural.reason {
                NaturalReason::Checkmate => GameEndReason::Checkmate,
                NaturalReason::Stalemate => GameEndReason::Stalemate,
                NaturalReason::Threefold => GameEndReason::Threefold,
                NaturalReason::FiftyMove => GameEndReason::FiftyMove,
                NaturalReason::BasicMaterial => GameEndReason::BasicMaterial,
            };
            assert_eq!(game.result_white, result_white);
            assert_eq!(game.end_reason, reason);
        }
        PostMoveOutcome::ScoreWin(ScoreDecision::WhiteWin) => {
            assert_eq!(game.result_white, 2);
            assert_eq!(game.end_reason, GameEndReason::ScoreWin);
        }
        PostMoveOutcome::ScoreWin(ScoreDecision::BlackWin) => {
            assert_eq!(game.result_white, 0);
            assert_eq!(game.end_reason, GameEndReason::ScoreWin);
        }
        PostMoveOutcome::ScoreDraw => {
            assert_eq!(game.result_white, 1);
            assert_eq!(game.end_reason, GameEndReason::ScoreDraw);
        }
    }
}

fn chunk_files(output_dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(output_dir.join("chunks"))
        .expect("chunk directory missing")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "k1c"))
        .collect();
    paths.sort();
    paths
}

fn file_snapshot(root: &Path) -> BTreeMap<String, (u64, String)> {
    fn walk(root: &Path, here: &Path, out: &mut BTreeMap<String, (u64, String)>) {
        for entry in fs::read_dir(here).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let relative = path.strip_prefix(root).unwrap().to_string_lossy().into_owned();
                out.insert(relative, (fs::metadata(&path).unwrap().len(), sha256(&path)));
            }
        }
    }
    let mut result = BTreeMap::new();
    walk(root, root, &mut result);
    result
}

fn worker_pid_from_chunk(path: &Path) -> u32 {
    let stem = path.file_stem().unwrap().to_string_lossy();
    let after_pid = stem.split_once("-p").expect("chunk name lacks worker PID").1;
    after_pid.split('-').next().unwrap().parse().expect("invalid chunk worker PID")
}

fn process_is_alive(pid: u32) -> Result<bool, String> {
    // `ps` can be denied by the desktop sandbox even for our own child
    // process. A signal-0 probe checks this exact PID without sending a signal.
    let output = Command::new("/bin/kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map_err(|error| format!("could not execute /bin/kill: {error}"))?;
    if output.status.success() {
        return Ok(true);
    }
    let stderr = stderr_text(&output);
    if stderr.to_ascii_lowercase().contains("no such process") {
        Ok(false)
    } else {
        Err(format!("signal-0 probe was inconclusive: {stderr}"))
    }
}

fn assert_success(output: &Output, context: &str) -> HashMap<String, String> {
    assert!(output.status.success(), "{context} failed: stdout={} stderr={}", stdout_text(output), stderr_text(output));
    key_values(&stdout_text(output))
}

#[test]
fn process_workers_trace_replay_dedup_resume_and_settings_guard() {
    let root = unique_root("process-artifacts");
    fs::create_dir_all(&root).unwrap();

    let first = invoke(&root, CORE_GAMES, CORE_START_ID, CORE_SEED, CORE_NODES, 2, 1, None);
    let first_values = assert_success(&first, "two-worker generation");
    assert_eq!(first_values["games"].parse::<u64>().unwrap(), CORE_GAMES);
    let raw_total = first_values["eligible_pre_dedup"].parse::<u64>().unwrap();
    let reported_duplicates = first_values["duplicate_candidates"].parse::<u64>().unwrap();
    let reported_unique = first_values["unique_positions"].parse::<u64>().unwrap();
    let dataset = PathBuf::from(&first_values["dataset"]);
    let cli_dataset_sha = &first_values["dataset_sha256"];
    assert_eq!(sha256(&dataset), *cli_dataset_sha, "dataset file SHA mismatch");

    let mut raw_games = Vec::new();
    let mut raw_positions = Vec::new();
    let mut raw_colors = Vec::new();
    let mut worker_games = BTreeMap::<u64, Vec<u64>>::new();
    let mut worker_pids = BTreeMap::<u64, Vec<u32>>::new();
    let chunks = chunk_files(&root);
    assert_eq!(chunks.len(), CORE_GAMES as usize, "chunk_games=1 should produce one committed chunk per game");
    for chunk in &chunks {
        let (summary, games) = inspect(chunk);
        assert_eq!(summary.kind, "raw-chunk");
        assert_eq!(summary.node_limit, Some(CORE_NODES));
        assert_eq!(games.len(), 1);
        let worker = summary.worker.expect("raw chunk worker index missing") as u64;
        worker_pids.entry(worker).or_default().push(worker_pid_from_chunk(chunk));
        for game in games {
            assert_eq!(game.seed, datagen::derive_game_seed(CORE_SEED, game.game_id));
            assert_eq!(worker, game.game_id % 2, "game assigned to wrong process worker");
            worker_games.entry(worker).or_default().push(game.game_id);
            let (rows, colors) = audit_game(&game, CORE_NODES, true);
            raw_positions.extend(rows);
            raw_colors.extend(colors);
            raw_games.push(game);
        }
    }
    raw_games.sort_by_key(|game| game.game_id);
    assert_eq!(raw_games.iter().map(|g| g.game_id).collect::<Vec<_>>(),
        (CORE_START_ID..CORE_START_ID + CORE_GAMES).collect::<Vec<_>>());
    assert!(raw_colors.contains(&true), "trace did not include White searches");
    assert!(raw_colors.contains(&false), "trace did not include Black searches");
    assert_eq!(raw_positions.len() as u64, raw_total);
    for worker in 0..2 {
        let ids = worker_games.get(&worker).expect("worker has no games");
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(*ids, sorted, "worker {worker} game IDs were not consumed in ascending order");
        let pids = worker_pids.get(&worker).expect("worker has no child PID evidence");
        assert!(pids.windows(2).all(|pair| pair[0] == pair[1]), "one fixed child process should own each worker shard");
    }
    assert_ne!(worker_pids[&0][0], worker_pids[&1][0], "worker shards must use distinct OS processes");

    let manifest = fs::read_to_string(&manifest_path_for(&root)).unwrap();
    let manifest_fields = key_values(&manifest);
    assert_eq!(manifest_fields["record_version"], "2");
    assert_eq!(manifest_fields["base_seed"], format!("{CORE_SEED:x}"));
    assert_eq!(manifest_fields["start_game_id"], CORE_START_ID.to_string());
    assert_eq!(manifest_fields["end_game_id_exclusive"], (CORE_START_ID + CORE_GAMES).to_string());
    assert_eq!(manifest_fields["node_limit"], CORE_NODES.to_string());
    assert_eq!(manifest_fields["worker_count"], "2");
    assert_eq!(manifest_fields["hash_mb_per_worker"], "1");
    assert_eq!(manifest_fields["engine_sha256"], sha256(Path::new(datagen_binary())));
    assert!(manifest_fields["search_limit_policy"].contains("depth None; deadline None"));

    let mut first_by_hash = BTreeMap::<u64, PositionTuple>::new();
    raw_positions.sort_by_key(|row| (row.game_id, row.ply));
    for row in raw_positions {
        first_by_hash.entry(row.zobrist).or_insert(row);
    }
    let expected_duplicates = raw_total - first_by_hash.len() as u64;
    assert_eq!(reported_duplicates, expected_duplicates);
    assert_eq!(reported_unique, first_by_hash.len() as u64);

    let (final_summary, final_games) = inspect(&dataset);
    assert_eq!(final_summary.kind, "merged-dataset");
    assert_eq!(final_summary.sha256, *cli_dataset_sha);
    assert_eq!(final_summary.games, CORE_GAMES);
    assert_eq!(final_summary.eligible_pre_dedup, raw_total);
    assert_eq!(final_summary.duplicates, expected_duplicates);
    assert_eq!(final_summary.unique_positions, first_by_hash.len() as u64);
    assert_eq!(final_summary.stored_candidate_rows, first_by_hash.len() as u64);

    let mut actual_final = Vec::new();
    let mut final_colors = Vec::new();
    for game in &final_games {
        let (rows, colors) = audit_game(game, CORE_NODES, false);
        actual_final.extend(rows);
        final_colors.extend(colors);
    }
    assert!(final_colors.contains(&true));
    assert!(final_colors.contains(&false));
    let mut expected_final: Vec<_> = first_by_hash.into_values().collect();
    expected_final.sort_by_key(|row| (row.game_id, row.ply));
    actual_final.sort_by_key(|row| (row.game_id, row.ply));
    assert_eq!(actual_final, expected_final, "global first-by-(game,ply) dedup differs from raw reconstruction");

    let inspect_cli = Command::new(datagen_binary()).arg("--inspect").arg(&dataset)
        .output().expect("failed to run CLI inspector");
    let inspect_values = assert_success(&inspect_cli, "CLI merged-dataset inspect");
    assert_eq!(inspect_values["sha256"], *cli_dataset_sha);
    assert_eq!(inspect_values["games"].parse::<u64>().unwrap(), CORE_GAMES);
    assert_eq!(inspect_values["eligible_pre_dedup"].parse::<u64>().unwrap(), raw_total);
    assert_eq!(inspect_values["duplicates"].parse::<u64>().unwrap(), expected_duplicates);

    let manifest_path = root.join("manifest.txt");
    let commits_path = root.join("commits.tsv");
    let manifest_before = fs::read(&manifest_path).expect("manifest missing");
    let commits_before = fs::read(&commits_path).expect("commit log missing");
    let chunk_hashes_before: Vec<_> = chunks.iter().map(|path| sha256(path)).collect();
    let second = invoke(&root, CORE_GAMES, CORE_START_ID, CORE_SEED, CORE_NODES, 2, 1, None);
    let second_values = assert_success(&second, "idempotent resume");
    assert_eq!(second_values["dataset_sha256"], *cli_dataset_sha);
    assert_eq!(fs::read(&manifest_path).unwrap(), manifest_before, "matching resume changed manifest");
    assert_eq!(fs::read(&commits_path).unwrap(), commits_before, "matching resume appended commits");
    assert_eq!(chunk_files(&root).iter().map(|path| sha256(path)).collect::<Vec<_>>(), chunk_hashes_before);

    let mismatch = invoke(&root, CORE_GAMES, CORE_START_ID, CORE_SEED, CORE_NODES + 1, 2, 1, None);
    assert!(!mismatch.status.success(), "resume with changed node limit should be rejected");
    assert_eq!(fs::read(&manifest_path).unwrap(), manifest_before, "mismatched resume changed manifest");
    assert_eq!(fs::read(&commits_path).unwrap(), commits_before, "mismatched resume changed commits");
    assert_eq!(sha256(&dataset), *cli_dataset_sha);

    println!("M3_PROCESS_ARTIFACT_ROOT={}", root.display());
    println!("M3_PROCESS_DATASET={}", dataset.display());
    println!("M3_PROCESS_GAMES={CORE_GAMES} ELIGIBLE={raw_total} DUPLICATES={expected_duplicates} UNIQUE={reported_unique}");
}

fn manifest_path_for(root: &Path) -> PathBuf {
    root.join("manifest.txt")
}

#[test]
fn failpoint_recovery_ignores_uncommitted_files_and_commits_once() {
    let failpoints = [
        "before-temp-write",
        "during-temp-write",
        "after-temp-fsync",
        "after-chunk-rename",
        "after-manifest-commit",
    ];
    let root = unique_root("failpoints");
    fs::create_dir_all(&root).unwrap();
    let mut recovery_rows = Vec::new();

    for failpoint in failpoints {
        let output_dir = root.join(failpoint);
        let failed = invoke(&output_dir, 1, 7, 0x6d33_6661_696c, 1, 1, 1, Some(failpoint));
        assert!(!failed.status.success(), "failpoint {failpoint} unexpectedly succeeded");
        let failure_text = format!("{}{}", stdout_text(&failed), stderr_text(&failed));
        assert!(failure_text.contains(failpoint), "failure log did not identify failpoint {failpoint}: {failure_text}");

        fs::create_dir_all(output_dir.join("chunks")).unwrap();
        let abandoned_temp = output_dir.join("chunks/verifier-uncommitted.tmp");
        fs::write(&abandoned_temp, b"uncommitted verifier scratch\n").unwrap();
        let orphan_count_before = chunk_files(&output_dir).len();
        if failpoint == "after-chunk-rename" {
            assert!(orphan_count_before > 0, "after-rename failpoint should leave an orphan raw chunk");
            let (orphan_summary, orphan_games) = inspect(&chunk_files(&output_dir)[0]);
            assert_eq!(orphan_summary.kind, "raw-chunk");
            assert_eq!(orphan_games.len(), 1);
        }

        let commits_path = output_dir.join("commits.tsv");
        let committed_before_retry = fs::read(&commits_path).unwrap_or_default();
        let chunks_before_retry = chunk_files(&output_dir);
        let retry = invoke(&output_dir, 1, 7, 0x6d33_6661_696c, 1, 1, 1, None);
        let retry_values = assert_success(&retry, &format!("retry after {failpoint}"));
        assert_eq!(retry_values["games"].parse::<u64>().unwrap(), 1);
        assert_eq!(retry_values["eligible_pre_dedup"].parse::<u64>().unwrap(), retry_values["unique_positions"].parse::<u64>().unwrap() + retry_values["duplicate_candidates"].parse::<u64>().unwrap());
        let dataset = PathBuf::from(&retry_values["dataset"]);
        let (summary, games) = inspect(&dataset);
        assert_eq!(summary.games, 1);
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].game_id, 7);
        assert_eq!(games[0].seed, datagen::derive_game_seed(0x6d33_6661_696c, 7));
        audit_game(&games[0], 1, false);
        assert!(abandoned_temp.exists(), "the verifier's orphan temp should remain ignored, not be consumed");

        if failpoint == "after-manifest-commit" {
            assert!(!committed_before_retry.is_empty(), "post-commit failpoint should leave a committed manifest row");
            assert_eq!(chunk_files(&output_dir), chunks_before_retry, "post-commit retry should not rewrite committed chunks");
            assert_eq!(fs::read(&commits_path).unwrap(), committed_before_retry, "post-commit retry duplicated the commit row");
        } else {
            assert!(fs::read(&commits_path).unwrap().len() >= committed_before_retry.len());
        }
        recovery_rows.push(format!("{failpoint}: recovered one game, sha256={}", summary.sha256));
    }

    let report = root.join("failpoint-recovery-summary.txt");
    fs::write(&report, format!("root={}\n{}\n", root.display(), recovery_rows.join("\n"))).unwrap();
    println!("M3_FAILPOINT_REPORT={}", report.display());
}

#[test]
fn worker_launch_failure_reaps_started_child_before_return_and_resumes() {
    let output_dir = unique_root("launch-failure");
    fs::create_dir_all(&output_dir).unwrap();
    let failed = invoke(&output_dir, 2, 100, 0x4d33_4c41_554e_4348, 1, 2, 1,
        Some("before-spool-create-worker-1"));
    assert!(!failed.status.success(), "launch failpoint should abort the coordinator");
    let text = format!("{}{}", stdout_text(&failed), stderr_text(&failed));
    assert!(text.contains("before-spool-create-worker-1"), "launch failure reason missing: {text}");

    let partial_chunks = chunk_files(&output_dir);
    assert_eq!(partial_chunks.len(), 1, "worker 0 should commit its completed game before return");
    let (partial_summary, partial_games) = inspect(&partial_chunks[0]);
    assert_eq!(partial_summary.worker, Some(0));
    assert_eq!(partial_games.len(), 1);
    assert_eq!(partial_games[0].game_id, 100);
    let child_pid = worker_pid_from_chunk(&partial_chunks[0]);
    match process_is_alive(child_pid) {
        Ok(false) => println!("PROCESS_REAP_CHECK=pid {child_pid} absent after coordinator return"),
        Ok(true) => panic!("worker child {child_pid} is still alive after coordinator returned"),
        Err(reason) => println!("PROCESS_REAP_CHECK_UNAVAILABLE=pid {child_pid}: {reason}"),
    }
    let partial_commits = fs::read_to_string(output_dir.join("commits.tsv")).unwrap();
    assert!(partial_commits.lines().any(|line| line.split('\t').any(|field| field == "100")));
    assert!(!partial_commits.lines().any(|line| line.split('\t').any(|field| field == "101")));
    assert!(!fs::read_dir(&output_dir).unwrap().any(|entry| {
        entry.unwrap().file_name().to_string_lossy().starts_with(".worker-spool-")
    }), "worker spool should have been imported and removed");

    let immediately_after_return = file_snapshot(&output_dir);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(file_snapshot(&output_dir), immediately_after_return,
        "output changed after failed coordinator returned; a worker may still be writing");

    let resumed = invoke(&output_dir, 2, 100, 0x4d33_4c41_554e_4348, 1, 2, 1, None);
    let values = assert_success(&resumed, "resume after worker launch failure");
    assert_eq!(values["games"].parse::<u64>().unwrap(), 2);
    let dataset = PathBuf::from(&values["dataset"]);
    let (summary, games) = inspect(&dataset);
    assert_eq!(summary.games, 2);
    assert_eq!(games.iter().map(|game| game.game_id).collect::<Vec<_>>(), vec![100, 101]);
    assert_eq!(games.iter().find(|game| game.game_id == 100).unwrap().seed,
        datagen::derive_game_seed(0x4d33_4c41_554e_4348, 100));
    assert_eq!(games.iter().find(|game| game.game_id == 101).unwrap().seed,
        datagen::derive_game_seed(0x4d33_4c41_554e_4348, 101));
    let final_snapshot = file_snapshot(&output_dir);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(file_snapshot(&output_dir), final_snapshot,
        "output changed after successful resume returned; a worker may still be writing");

    let report_path = output_dir.join("launch-failure-recovery-summary.txt");
    fs::write(&report_path, format!(
        "failpoint=before-spool-create-worker-1\npartial_game=100\npartial_worker=0\nreaped_child_pid={child_pid}\npartial_commit_rows={}\nresumed_games={}\nfinal_sha256={}\noutput_stable_after_return=true\n",
        partial_commits.lines().count(), summary.games, summary.sha256,
    )).unwrap();
    println!("M3_LAUNCH_FAILURE_REPORT={}", report_path.display());
}

#[test]
fn signed_promotion_bestmoves_inspect_and_parallel_runs_match_records() {
    const GAMES: u64 = 8;
    const START_ID: u64 = 3_000;
    const BASE_SEED: u64 = 0x4d33_5245_5045_4154;
    const NODES: u64 = 5_000;

    struct Run {
        workers: usize,
        root: PathBuf,
        dataset: PathBuf,
        dataset_sha: String,
        config_sha: String,
        games: u64,
        eligible: u64,
        duplicates: u64,
        unique: u64,
        promotion_events: usize,
        histories: Vec<(u64, u64, i32, String, Vec<String>, u8)>,
        rows: Vec<PositionTuple>,
    }

    let mut runs = Vec::new();
    for workers in [1usize, 4usize] {
        let root = unique_root(&format!("promotion-probe-w{workers}"));
        fs::create_dir_all(&root).unwrap();
        let output = invoke(&root, GAMES, START_ID, BASE_SEED, NODES, workers, 1, None);
        let values = assert_success(&output, &format!("promotion probe with {workers} workers"));
        let dataset = PathBuf::from(&values["dataset"]);
        let dataset_sha = sha256(&dataset);
        assert_eq!(dataset_sha, values["dataset_sha256"]);

        let chunks = chunk_files(&root);
        assert_eq!(chunks.len(), GAMES as usize);
        let mut raw_games = Vec::new();
        let mut promotion_events = 0usize;
        for chunk in &chunks {
            let (chunk_summary, games) = inspect(chunk);
            assert_eq!(chunk_summary.kind, "raw-chunk");
            assert_eq!(chunk_summary.node_limit, Some(NODES));
            assert_eq!(games.len(), 1);
            for game in games {
                assert_eq!(chunk_summary.worker, Some((game.game_id % workers as u64) as u32));
                assert_eq!(game.seed, datagen::derive_game_seed(BASE_SEED, game.game_id));
                promotion_events += game.search_events.iter()
                    .filter(|event| event.best_move_uci.len() == 5)
                    .count();
                let _ = audit_game(&game, NODES, true);
                raw_games.push(game);
            }
        }
        raw_games.sort_by_key(|game| game.game_id);
        assert_eq!(raw_games.iter().map(|game| game.game_id).collect::<Vec<_>>(),
            (START_ID..START_ID + GAMES).collect::<Vec<_>>());
        assert!(promotion_events > 0,
            "the pinned seed/range must exercise at least one promotion bestmove; otherwise this does not cover promotion parsing");
        assert!(raw_games[0].search_events.iter().any(|event|
            event.ply == 102 && event.white_to_move && event.best_move_uci == "a7a8q"),
            "game 3000 must retain the known White a7a8q raw event at ply 102");

        let (summary, merged_games) = inspect(&dataset);
        assert_eq!(summary.kind, "merged-dataset");
        assert_eq!(summary.games, GAMES);
        assert_eq!(summary.node_limit, Some(NODES));
        assert_eq!(summary.stored_candidate_rows, summary.unique_positions);
        assert_eq!(summary.sha256, dataset_sha);
        let mut histories = Vec::new();
        let mut rows = Vec::new();
        for game in &merged_games {
            let (game_rows, _) = audit_game(game, NODES, false);
            rows.extend(game_rows);
            histories.push((game.game_id, game.seed, game.accepted_opening_hce,
                game.start_fen.clone(), game.moves.clone(), game.result_white));
        }
        rows.sort_by_key(|row| (row.game_id, row.ply));
        histories.sort_by_key(|row| row.0);
        assert_eq!(rows.len() as u64, summary.unique_positions);

        runs.push(Run {
            workers,
            root,
            dataset,
            dataset_sha,
            config_sha: summary.config_sha256,
            games: summary.games,
            eligible: summary.eligible_pre_dedup,
            duplicates: summary.duplicates,
            unique: summary.unique_positions,
            promotion_events,
            histories,
            rows,
        });
    }

    let bytes_one = fs::read(&runs[0].dataset).unwrap();
    let bytes_four = fs::read(&runs[1].dataset).unwrap();
    let bytes_equal = bytes_one == bytes_four;
    let first_difference = bytes_one.iter().zip(&bytes_four)
        .position(|(left, right)| left != right)
        .unwrap_or(bytes_one.len().min(bytes_four.len()));
    const DATASET_CONFIG_HASH_START: usize = 10; // 8-byte magic + 2-byte format version
    const DATASET_CONFIG_HASH_END: usize = DATASET_CONFIG_HASH_START + 32;
    assert_eq!(&bytes_one[..DATASET_CONFIG_HASH_START], &bytes_four[..DATASET_CONFIG_HASH_START]);
    assert_eq!(&bytes_one[DATASET_CONFIG_HASH_END..], &bytes_four[DATASET_CONFIG_HASH_END..],
        "merged data differs outside the pinned config-hash header field");
    let mut normalized_one = bytes_one.clone();
    let mut normalized_four = bytes_four.clone();
    normalized_one[DATASET_CONFIG_HASH_START..DATASET_CONFIG_HASH_END].fill(0);
    normalized_four[DATASET_CONFIG_HASH_START..DATASET_CONFIG_HASH_END].fill(0);
    assert_eq!(normalized_one, normalized_four,
        "normalized dataset bytes must match after zeroing only the 32-byte config-hash header");
    assert_ne!(runs[0].config_sha, runs[1].config_sha,
        "worker count is expected to be pinned by the dataset config hash");
    let normalized_one_path = runs[0].root.join("normalized-dataset.k1d");
    let normalized_four_path = runs[1].root.join("normalized-dataset.k1d");
    fs::write(&normalized_one_path, &normalized_one).unwrap();
    fs::write(&normalized_four_path, &normalized_four).unwrap();
    let normalized_sha_one = sha256(&normalized_one_path);
    let normalized_sha_four = sha256(&normalized_four_path);
    assert_eq!(normalized_sha_one, normalized_sha_four);
    assert_eq!(runs[0].games, runs[1].games);
    assert_eq!(runs[0].eligible, runs[1].eligible);
    assert_eq!(runs[0].duplicates, runs[1].duplicates);
    assert_eq!(runs[0].unique, runs[1].unique);
    assert_eq!(runs[0].histories, runs[1].histories,
        "worker count changed deterministic game histories/results");
    assert_eq!(runs[0].rows, runs[1].rows,
        "worker count changed merged candidate rows");

    println!("M3_PROMOTION_PROBE=games:{} start_id:{} base_seed:{} nodes:{}", GAMES, START_ID, BASE_SEED, NODES);
    for run in &runs {
        println!("M3_PROMOTION_RUN=workers:{} root:{} dataset:{} sha256:{} config_sha256:{} games:{} eligible:{} duplicates:{} unique:{} promotion_events:{}",
            run.workers, run.root.display(), run.dataset.display(), run.dataset_sha, run.config_sha,
            run.games, run.eligible, run.duplicates, run.unique, run.promotion_events);
    }
    println!("M3_PROMOTION_DATASET_BYTES_EQUAL={bytes_equal} first_difference={first_difference} lengths={}/{}",
        bytes_one.len(), bytes_four.len());
    println!("M3_PROMOTION_NORMALIZED_BYTES_EQUAL=true config_hash_byte_range={}..{} normalized_sha256={}",
        DATASET_CONFIG_HASH_START, DATASET_CONFIG_HASH_END, normalized_sha_one);
}

