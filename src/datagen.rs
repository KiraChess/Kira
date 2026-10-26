//! Deterministic, resumable Kira self-play data generation.
//!
//! The on-disk formats in this module are deliberately versioned and use
//! length-prefixed game and position records. Raw worker chunks retain every
//! eligible position before global Zobrist deduplication, as well as the
//! complete legal game history needed by independent auditors.

use crate::eval::nnue_quantized::QuantizedNnueNetwork;
use crate::eval::{Evaluator, Hce};
use crate::search::{MATE_SCORE, SearchLimits, Searcher};
use crate::{Board, Move, START_FEN};
use std::collections::{HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Child, Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const FORMAT_VERSION: u16 = 2;
const CHUNK_MAGIC: &[u8; 8] = b"K1ACHUNK";
const DATASET_MAGIC: &[u8; 8] = b"K1ADATAS";
const OPENING_PROTOCOL: &str = "startpos; exactly 8 legal uniform plies; SplitMix64-v1 rejection modulo; retry opening until abs(HCE)<=300cp; RNG continues between retries";
const DEDUP_POLICY: &str = "Zobrist-u64-only; globally first by (game_id,ply); denominator is post-filter/pre-dedup candidates";
const WIN_STREAK_POLICY: &str = "5 consecutive plies; convert STM cp to White-relative (white_to_move ? score : -score); White >= +1500 or Black <= -1500 inclusive; adjudicate after move then natural-terminal precedence";
const PRNG_VERSION: &str = "SplitMix64-v1; game_seed=finalizer(base_seed XOR game_id); next=finalizer(state += 0x9e3779b97f4a7c15)";

/// Literal v6 design fixtures in side-to-move pairs: expected result of the
/// selected basic insufficient-material predicate.
pub const MATERIAL_RULE_FIXTURES: [(&str, bool); 24] = [
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", true),
    ("4k3/8/8/8/8/8/8/4K3 b - - 0 1", true),
    ("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1", true),
    ("4k3/8/8/8/8/8/8/2B1K3 b - - 0 1", true),
    ("2b1k3/8/8/8/8/8/8/4K3 w - - 0 1", true),
    ("2b1k3/8/8/8/8/8/8/4K3 b - - 0 1", true),
    ("4k3/8/8/8/8/8/8/2N1K3 w - - 0 1", true),
    ("4k3/8/8/8/8/8/8/2N1K3 b - - 0 1", true),
    ("2n1k3/8/8/8/8/8/8/4K3 w - - 0 1", true),
    ("2n1k3/8/8/8/8/8/8/4K3 b - - 0 1", true),
    ("1b2k3/8/8/8/8/8/8/2B1K3 w - - 0 1", false),
    ("1b2k3/8/8/8/8/8/8/2B1K3 b - - 0 1", false),
    ("2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1", false),
    ("2b1k3/8/8/8/8/8/8/2B1K3 b - - 0 1", false),
    ("2n1k3/8/8/8/8/8/8/2N1K3 w - - 0 1", false),
    ("2n1k3/8/8/8/8/8/8/2N1K3 b - - 0 1", false),
    ("4k3/8/8/8/8/8/NN6/4K3 w - - 0 1", false),
    ("4k3/8/8/8/8/8/NN6/4K3 b - - 0 1", false),
    ("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", false),
    ("4k3/8/8/8/8/8/P7/4K3 b - - 0 1", false),
    ("4k3/8/8/8/8/8/8/3RK3 w - - 0 1", false),
    ("4k3/8/8/8/8/8/8/3RK3 b - - 0 1", false),
    ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", false),
    ("4k3/8/8/8/8/8/8/3QK3 b - - 0 1", false),
];

pub const CHECKMATE_RULE50_PRECEDENCE_FEN: &str = "7k/6Q1/6K1/8/8/8/8/8 b - - 100 51";
pub const STALEMATE_MATERIAL_PRECEDENCE_FEN: &str = "7k/5K2/6B1/8/8/8/8/8 b - - 0 1";
const WIN_STREAK_PLIES: u8 = 5;
const DRAW_STREAK_PLY_START: u32 = 120;
const DRAW_STREAK_PLY_END: u32 = 159;
const DRAW_STREAK_REQUIRED: u8 = 40;
const MATE_THRESHOLD: i32 = MATE_SCORE - 256;

/// Side from White's point of view, used by the fixed-side win streak rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WinSide {
    White,
    Black,
}

/// Result of observing a single search score under M3 adjudication rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreDecision {
    WhiteWin,
    BlackWin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NaturalReason { Checkmate, Stalemate, Threefold, FiftyMove, BasicMaterial }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NaturalResult { WhiteWin, BlackWin, Draw }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NaturalOutcome { pub result: NaturalResult, pub reason: NaturalReason }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PostMoveOutcome {
    Natural(NaturalOutcome),
    ScoreWin(ScoreDecision),
    ScoreDraw,
}

/// Minimal state for the signed-off five-ply win and move-60-to-80 draw rules.
/// This pure helper is public so the boundary fixtures can exercise p=120,
/// p=159, p=160, inclusive ±1500 scores, and STM-to-White conversion directly.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScoreStreaks {
    winner: Option<WinSide>,
    winner_plies: u8,
    draw_plies: u8,
    last_draw_ply: Option<u32>,
}

/// Convert one side-to-move search score to a stable White-relative score.
pub fn white_relative_score(score_stm: i32, white_to_move: bool) -> i32 {
    if white_to_move { score_stm } else { -score_stm }
}

/// Observe the score before the move at `completed_plies`. A returned win is
/// pending until the caller makes that move and checks natural terminal rules.
pub fn observe_score(
    streaks: &mut ScoreStreaks,
    completed_plies: u32,
    score_stm: i32,
    white_to_move: bool,
) -> Option<ScoreDecision> {
    let white_score = white_relative_score(score_stm, white_to_move);
    let mate_score = score_stm >= MATE_THRESHOLD || score_stm <= -MATE_THRESHOLD;
    let current_winner = if !mate_score && white_score >= 1_500 {
        Some(WinSide::White)
    } else if !mate_score && white_score <= -1_500 {
        Some(WinSide::Black)
    } else {
        None
    };
    if let Some(winner) = current_winner {
        if streaks.winner == Some(winner) {
            streaks.winner_plies = streaks.winner_plies.saturating_add(1);
        } else {
            streaks.winner = Some(winner);
            streaks.winner_plies = 1;
        }
    } else {
        streaks.winner = None;
        streaks.winner_plies = 0;
    }

    if (DRAW_STREAK_PLY_START..=DRAW_STREAK_PLY_END).contains(&completed_plies) {
        if completed_plies == DRAW_STREAK_PLY_START
            || streaks.last_draw_ply != Some(completed_plies - 1)
        {
            streaks.draw_plies = 0;
        }
        if score_stm.abs() <= 5 {
            streaks.draw_plies = streaks.draw_plies.saturating_add(1);
        } else {
            streaks.draw_plies = 0;
        }
        streaks.last_draw_ply = Some(completed_plies);
    }

    if streaks.winner_plies >= WIN_STREAK_PLIES {
        Some(match streaks.winner {
            Some(WinSide::White) => ScoreDecision::WhiteWin,
            Some(WinSide::Black) => ScoreDecision::BlackWin,
            None => unreachable!("nonzero win streak has a winner"),
        })
    } else {
        None
    }
}

/// The ±5 cp streak can adjudicate only after Black completes move 80 (160
/// plies), after the caller has checked mate, stalemate, and natural draws.
pub fn score_draw_ready_after_move(streaks: &ScoreStreaks, completed_plies: u32) -> bool {
    completed_plies == 160
        && streaks.last_draw_ply == Some(DRAW_STREAK_PLY_END)
        && streaks.draw_plies >= DRAW_STREAK_REQUIRED
}

/// Opening acceptance uses the inclusive HCE interval [-300,+300].
pub fn opening_score_accepted(score_stm: i32) -> bool { (-300..=300).contains(&score_stm) }

/// Opening gating and its stored audit label remain HCE even when search
/// positions use an NNUE evaluator.
pub fn opening_hce_score(board: &Board) -> i32 { Hce.evaluate(board) }

/// Derive a game seed using the signed-off SplitMix64 finalizer.
pub fn derive_game_seed(base_seed: u64, game_id: u64) -> u64 {
    splitmix_finalizer(base_seed ^ game_id)
}

/// Return the pinned worker owner for a game ID.
pub fn worker_for_game(game_id: u64, worker_count: usize) -> Result<usize, String> {
    if worker_count == 0 { return Err("worker count must be positive".into()); }
    Ok((game_id % worker_count as u64) as usize)
}

/// Expose deterministic SplitMix64 words for verifier known-answer fixtures.
pub struct SplitMix64 { state: u64 }

impl SplitMix64 {
    pub fn seeded(game_seed: u64) -> Self { Self { state: game_seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        splitmix_finalizer(self.state)
    }
}

/// Use rejection-sampled modulo over supplied words so a verifier can force
/// the retry boundary deterministically.
pub fn uniform_index_from_words(len: usize, words: impl IntoIterator<Item = u64>) -> Result<usize, String> {
    if len == 0 { return Err("cannot select from an empty legal move list".into()); }
    let bound = len as u64;
    let threshold = bound.wrapping_neg() % bound;
    for value in words {
        if value >= threshold { return Ok((value % bound) as usize); }
    }
    Err("raw RNG word stream ended before an accepted sample".into())
}

/// The user-selected basic material subset, without full FIDE dead-position logic.
pub fn has_basic_insufficient_material(board: &Board) -> bool { insufficient_material_basic(board) }

/// Classify natural terminals in the signed-off order: legal moves/checkmate,
/// stalemate, repetition, rule 50, then the selected basic material subset.
pub fn classify_natural_terminal(board: &mut Board, history: &[u64]) -> Option<NaturalOutcome> {
    natural_terminal(board, history).map(|(result, reason)| NaturalOutcome {
        result: match result {
            ResultWhite::WhiteWin => NaturalResult::WhiteWin,
            ResultWhite::BlackWin => NaturalResult::BlackWin,
            ResultWhite::Draw => NaturalResult::Draw,
        },
        reason: match reason {
            EndReason::Checkmate => NaturalReason::Checkmate,
            EndReason::Stalemate => NaturalReason::Stalemate,
            EndReason::Threefold => NaturalReason::Threefold,
            EndReason::FiftyMove => NaturalReason::FiftyMove,
            EndReason::BasicMaterial => NaturalReason::BasicMaterial,
            EndReason::ScoreWin | EndReason::ScoreDraw => unreachable!("natural classifier returns no score endings"),
        },
    })
}

/// Resolve a just-completed move with the production precedence: natural
/// terminals first, then a pending five-ply score win, then the p=160 score
/// draw. The completed board and its hash must already be in `history`.
pub fn adjudicate_after_move(
    board: &mut Board,
    history: &[u64],
    pending_score_win: Option<ScoreDecision>,
    streaks: &ScoreStreaks,
    completed_plies: u32,
) -> Option<PostMoveOutcome> {
    if let Some(natural) = classify_natural_terminal(board, history) {
        return Some(PostMoveOutcome::Natural(natural));
    }
    if let Some(decision) = pending_score_win {
        return Some(PostMoveOutcome::ScoreWin(decision));
    }
    if score_draw_ready_after_move(streaks, completed_plies) {
        return Some(PostMoveOutcome::ScoreDraw);
    }
    None
}

#[derive(Clone, Debug)]
pub struct DatagenConfig {
    pub output: PathBuf,
    pub start_game_id: u64,
    pub games: u64,
    pub base_seed: u64,
    pub nodes: u64,
    pub hash_mb: usize,
    pub workers: usize,
    pub chunk_games: usize,
    /// Optional quantized NNUE used only for fixed-node search scores/labels.
    /// The eight-ply opening acceptance gate always remains HCE.
    pub eval_file: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct DatagenSummary {
    pub dataset: PathBuf,
    pub dataset_sha256: String,
    pub games: u64,
    pub eligible_pre_dedup: u64,
    pub duplicate_candidates: u64,
    pub unique_positions: u64,
}

#[derive(Clone, Debug)]
pub struct DecodedCandidate {
    pub fen: String,
    pub ply: u32,
    pub white_to_move: bool,
    pub score_stm: i32,
    /// Integer encoding: 0=loss, 1=draw, 2=win, from side-to-move view.
    pub result_stm: u8,
    pub best_move_class: BestMoveClass,
    pub zobrist: u64,
}

#[derive(Clone, Debug)]
pub struct DecodedSearchEvent {
    pub fen: String,
    pub ply: u32,
    pub white_to_move: bool,
    pub score_stm: i32,
    pub search_call_index: u32,
    pub node_limit: u64,
    /// Search API depth limit; M3 records must contain `None`.
    pub depth_limit: Option<u8>,
    /// Search API deadline in milliseconds; M3 records must contain `None`.
    pub time_limit_ms: Option<u64>,
    pub search_nodes: u64,
    pub search_depth: u8,
    pub best_move_uci: String,
    pub best_move_class: BestMoveClass,
    pub filter_reason: FilterReason,
    pub zobrist: u64,
}

#[derive(Clone, Debug)]
pub struct DecodedGame {
    pub game_id: u64,
    pub seed: u64,
    pub start_fen: String,
    pub opening_attempts: u32,
    pub accepted_opening_hce: i32,
    pub moves: Vec<String>,
    /// Integer encoding: 0=Black win, 1=draw, 2=White win.
    pub result_white: u8,
    pub end_reason: GameEndReason,
    pub eligible_count_pre_dedup: u64,
    pub candidates: Vec<DecodedCandidate>,
    pub search_events: Vec<DecodedSearchEvent>,
}

#[derive(Clone, Debug)]
pub struct ArtifactSummary {
    pub kind: String,
    pub sha256: String,
    pub config_sha256: String,
    pub games: u64,
    pub eligible_pre_dedup: u64,
    pub stored_candidate_rows: u64,
    pub search_events: u64,
    pub duplicates: u64,
    pub unique_positions: u64,
    pub worker: Option<u32>,
    pub node_limit: Option<u64>,
    pub search_evaluator: String,
    pub nnue_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum ResultWhite {
    BlackWin = 0,
    Draw = 1,
    WhiteWin = 2,
}

impl ResultWhite {
    fn stm_label(self, white_to_move: bool) -> u8 {
        match (self, white_to_move) {
            (Self::Draw, _) => 1,
            (Self::WhiteWin, true) | (Self::BlackWin, false) => 2,
            _ => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GameEndReason {
    Checkmate = 0,
    Stalemate = 1,
    Threefold = 2,
    FiftyMove = 3,
    BasicMaterial = 4,
    ScoreWin = 5,
    ScoreDraw = 6,
}

type EndReason = GameEndReason;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum BestMoveClass {
    Quiet = 0,
    Capture = 1,
    Promotion = 2,
    CapturePromotion = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FilterReason {
    Kept = 0,
    InCheck = 1,
    BestMoveCapture = 2,
    BestMovePromotion = 3,
    MateScore = 4,
    ScoreOutsideRange = 5,
}

#[derive(Clone, Debug)]
struct SearchEvent {
    fen: String,
    ply: u32,
    white_to_move: bool,
    score_stm: i32,
    search_call_index: u32,
    node_limit: u64,
    depth_limit: Option<u8>,
    time_limit_ms: Option<u64>,
    search_nodes: u64,
    search_depth: u8,
    best_move_uci: String,
    best_move_class: BestMoveClass,
    filter_reason: FilterReason,
    zobrist: u64,
}

#[derive(Clone, Debug)]
struct PositionRecord {
    fen: String,
    ply: u32,
    white_to_move: bool,
    score_stm: i32,
    result_stm: u8,
    best_move_class: BestMoveClass,
    zobrist: u64,
}

#[derive(Clone, Debug)]
struct GameRecord {
    game_id: u64,
    seed: u64,
    start_fen: String,
    opening_attempts: u32,
    accepted_opening_hce: i32,
    moves: Vec<String>,
    result_white: ResultWhite,
    end_reason: EndReason,
    eligible_count: u64,
    positions: Vec<PositionRecord>,
    events: Vec<SearchEvent>,
}

#[derive(Clone, Debug)]
struct GenerationSettings {
    start_game_id: u64,
    end_game_id_exclusive: u64,
    base_seed: u64,
    nodes: u64,
    hash_mb: usize,
    workers: usize,
    chunk_games: usize,
    engine_sha256: String,
    source_sha256: String,
    config_sha256: String,
    search_evaluator: SearchEvaluatorPin,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SearchEvaluatorPin {
    kind: SearchEvaluatorKind,
    nnue_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchEvaluatorKind {
    Hce,
    QuantizedNnue128,
}

impl SearchEvaluatorKind {
    fn manifest_name(self) -> &'static str {
        match self {
            Self::Hce => "Kira-HCE",
            Self::QuantizedNnue128 => "QuantizedNnueNetwork<128>",
        }
    }
}

#[derive(Debug)]
enum SearchEvaluator {
    Hce(Hce),
    Nnue(QuantizedNnueNetwork<128>),
}

impl Evaluator for SearchEvaluator {
    fn evaluate(&self, board: &Board) -> i32 {
        match self {
            Self::Hce(evaluator) => evaluator.evaluate(board),
            Self::Nnue(evaluator) => evaluator.evaluate(board),
        }
    }
}

#[derive(Clone, Debug)]
struct ChunkCommit {
    filename: String,
    worker: usize,
    ids: Vec<u64>,
    seeds: Vec<u64>,
    candidate_counts: Vec<u64>,
    sha256: String,
    engine_sha256: String,
    source_sha256: String,
    config_sha256: String,
}

struct LockGuard(PathBuf);

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Run or resume a fixed generation, then globally deduplicate all committed
/// candidate chunks and write an immutable final dataset artifact.
pub fn run(config: DatagenConfig) -> Result<DatagenSummary, String> {
    validate_config(&config)?;
    let end_game_id_exclusive = config
        .start_game_id
        .checked_add(config.games)
        .ok_or_else(|| "game ID range overflows u64".to_string())?;
    fs::create_dir_all(&config.output).map_err(io_error)?;
    fs::create_dir_all(config.output.join("chunks")).map_err(io_error)?;
    let lock_path = config.output.join("writer.lock");
    let _lock = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .map_err(|e| {
            if e.kind() == io::ErrorKind::AlreadyExists {
                format!("generation lock already exists at {}; remove it only after confirming no writer is active", lock_path.display())
            } else {
                io_error(e)
            }
        })?;
    let _lock_guard = LockGuard(lock_path);

    // Exact source/executable pins intentionally prohibit cross-source-cut
    // resume; each production generation (including gen2) uses a fresh output.
    let settings = generation_settings(&config, end_game_id_exclusive)?;

    let manifest_path = config.output.join("manifest.txt");
    let commits_path = config.output.join("commits.tsv");
    initialize_or_check_manifest(&manifest_path, &commits_path, &settings)?;
    repair_unterminated_commit_tail(&commits_path)?;
    let commits = load_and_verify_commits(&config.output, &settings)?;
    let mut committed_ids = HashSet::new();
    for commit in &commits {
        for &id in &commit.ids {
            if !committed_ids.insert(id) {
                return Err(format!("game ID {id} appears in multiple committed chunks"));
            }
        }
    }

    run_workers_as_processes(&config, &settings, &committed_ids, &commits_path)?;
    verify_search_evaluator_unchanged(&config, &settings.search_evaluator)?;

    // Reload commits because workers appended them during this invocation.
    repair_unterminated_commit_tail(&commits_path)?;
    let commits = load_and_verify_commits(&config.output, &settings)?;
    let mut all_ids = HashSet::new();
    for commit in &commits {
        for &id in &commit.ids {
            if !all_ids.insert(id) {
                return Err(format!("game ID {id} appears in multiple committed chunks"));
            }
        }
    }
    if all_ids.len() as u64 != config.games
        || (config.start_game_id..end_game_id_exclusive).any(|id| !all_ids.contains(&id))
    {
        return Err(format!(
            "generation incomplete: {} of {} game IDs are committed",
            all_ids.len(), config.games
        ));
    }

    merge_committed_chunks(&config.output, &settings, &commits, config.games)
}

fn generation_settings(config: &DatagenConfig, end_game_id_exclusive: u64) -> Result<GenerationSettings, String> {
    let search_evaluator = search_evaluator_pin(config)?;
    Ok(GenerationSettings {
        start_game_id: config.start_game_id,
        end_game_id_exclusive,
        base_seed: config.base_seed,
        nodes: config.nodes,
        hash_mb: config.hash_mb,
        workers: config.workers,
        chunk_games: config.chunk_games,
        engine_sha256: sha256_file(&std::env::current_exe().map_err(io_error)?)?,
        source_sha256: source_tree_sha256()?,
        config_sha256: configuration_sha256(config, end_game_id_exclusive, &search_evaluator),
        search_evaluator,
    })
}

fn search_evaluator_pin(config: &DatagenConfig) -> Result<SearchEvaluatorPin, String> {
    match &config.eval_file {
        None => Ok(SearchEvaluatorPin {
            kind: SearchEvaluatorKind::Hce,
            nnue_sha256: None,
        }),
        Some(path) => {
            let bytes = fs::read(path).map_err(|error| {
                format!("could not read NNUE evaluator {}: {}", path.display(), io_error(error))
            })?;
            QuantizedNnueNetwork::<128>::from_bytes(&bytes)
                .map_err(|error| format!("invalid NNUE evaluator {}: {error}", path.display()))?;
            Ok(SearchEvaluatorPin {
                kind: SearchEvaluatorKind::QuantizedNnue128,
                nnue_sha256: Some(sha256_hex(&bytes)),
            })
        }
    }
}

fn load_search_evaluator(config: &DatagenConfig, pin: &SearchEvaluatorPin) -> Result<SearchEvaluator, String> {
    match (pin.kind, &config.eval_file, &pin.nnue_sha256) {
        (SearchEvaluatorKind::Hce, None, None) => Ok(SearchEvaluator::Hce(Hce)),
        (SearchEvaluatorKind::QuantizedNnue128, Some(path), Some(expected_sha256)) => {
            let bytes = fs::read(path).map_err(|error| {
                format!("could not read NNUE evaluator {}: {}", path.display(), io_error(error))
            })?;
            let actual_sha256 = sha256_hex(&bytes);
            if &actual_sha256 != expected_sha256 {
                return Err(format!(
                    "NNUE evaluator SHA-256 changed: expected {expected_sha256}, found {actual_sha256}"
                ));
            }
            let network = QuantizedNnueNetwork::<128>::from_bytes(&bytes)
                .map_err(|error| format!("invalid NNUE evaluator {}: {error}", path.display()))?;
            Ok(SearchEvaluator::Nnue(network))
        }
        _ => Err("search evaluator kind, NNUE path, and pinned SHA-256 disagree".into()),
    }
}

fn verify_search_evaluator_unchanged(config: &DatagenConfig, pin: &SearchEvaluatorPin) -> Result<(), String> {
    match (pin.kind, &config.eval_file, &pin.nnue_sha256) {
        (SearchEvaluatorKind::Hce, None, None) => Ok(()),
        (SearchEvaluatorKind::QuantizedNnue128, Some(path), Some(expected_sha256)) => {
            let actual_sha256 = sha256_file(path)?;
            if &actual_sha256 != expected_sha256 {
                return Err(format!(
                    "NNUE evaluator SHA-256 changed: expected {expected_sha256}, found {actual_sha256}"
                ));
            }
            Ok(())
        }
        _ => Err("search evaluator kind, NNUE path, and pinned SHA-256 disagree".into()),
    }
}

struct RunningWorker {
    worker: usize,
    child: Child,
    spool: PathBuf,
}

/// Hidden OS-child entry point. The parent has pinned the manifest and owns
/// commits.tsv; this process writes complete chunks and syncs its private
/// worker spool after each chunk.
pub fn run_worker_child(
    config: DatagenConfig,
    worker: usize,
    spool: &Path,
    expected_engine_sha256: &str,
    expected_source_sha256: &str,
    expected_config_sha256: &str,
    expected_evaluator_kind: &str,
    expected_nnue_sha256: &str,
) -> Result<(), String> {
    validate_config(&config)?;
    if worker >= config.workers {
        return Err(format!("worker index {worker} is outside configured worker count {}", config.workers));
    }
    if !spool.is_file() {
        return Err(format!("worker commit spool was not created by parent: {}", spool.display()));
    }
    let end = config.start_game_id.checked_add(config.games)
        .ok_or_else(|| "game ID range overflows u64".to_string())?;
    let settings = generation_settings(&config, end)?;
    if settings.engine_sha256 != expected_engine_sha256
        || settings.source_sha256 != expected_source_sha256
        || settings.config_sha256 != expected_config_sha256
        || settings.search_evaluator.kind.manifest_name() != expected_evaluator_kind
        || settings.search_evaluator.nnue_sha256.as_deref().unwrap_or("none") != expected_nnue_sha256
    {
        return Err("worker executable/source/config/evaluator pins differ from parent".into());
    }
    if fs::read_to_string(config.output.join("manifest.txt")).map_err(io_error)? != manifest_text(&settings) {
        return Err("worker manifest differs from pinned generation settings".into());
    }

    let committed_ids = read_committed_ids_for_worker(&config.output.join("commits.tsv"), worker, &settings)?;
    let ids: Vec<u64> = (settings.start_game_id..settings.end_game_id_exclusive)
        .filter(|id| id % settings.workers as u64 == worker as u64 && !committed_ids.contains(id))
        .collect();
    let mut searcher = Searcher::new(config.hash_mb);
    let search_evaluator = load_search_evaluator(&config, &settings.search_evaluator)?;
    let stop = AtomicBool::new(false);
    let mut queue: VecDeque<u64> = ids.into();
    while !queue.is_empty() {
        let mut games = Vec::new();
        for _ in 0..config.chunk_games {
            let Some(id) = queue.pop_front() else { break };
            let seed = game_seed(config.base_seed, id);
            verify_search_evaluator_unchanged(&config, &settings.search_evaluator)?;
            let game = play_game(id, seed, &config, &mut searcher, &search_evaluator, &stop)?;
            verify_search_evaluator_unchanged(&config, &settings.search_evaluator)?;
            games.push(game);
        }
        let commit = write_chunk(worker, &games, &config, &settings)?;
        let path = config.output.join("chunks").join(&commit.filename);
        verify_chunk_contents(&path, &commit, &settings)?;
        append_worker_spool(spool, &commit)?;
    }
    Ok(())
}

fn read_committed_ids_for_worker(
    path: &Path,
    worker: usize,
    settings: &GenerationSettings,
) -> Result<HashSet<u64>, String> {
    let text = fs::read_to_string(path).map_err(io_error)?;
    let mut ids = HashSet::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.is_empty() { continue; }
        let commit = parse_commit(line).map_err(|error| format!("commits.tsv line {}: {error}", line_no + 1))?;
        if commit.engine_sha256 != settings.engine_sha256
            || commit.source_sha256 != settings.source_sha256
            || commit.config_sha256 != settings.config_sha256
            || commit.worker >= settings.workers
        {
            return Err(format!("existing chunk {} has mismatched worker pins", commit.filename));
        }
        if commit.worker == worker {
            for id in commit.ids {
                if id < settings.start_game_id || id >= settings.end_game_id_exclusive
                    || id % settings.workers as u64 != worker as u64 || !ids.insert(id)
                {
                    return Err(format!("invalid or duplicated committed game ID {id} for worker {worker}"));
                }
            }
        }
    }
    Ok(ids)
}

fn append_worker_spool(path: &Path, commit: &ChunkCommit) -> Result<(), String> {
    let mut file = OpenOptions::new().append(true).open(path).map_err(io_error)?;
    file.write_all(commit_line(commit).as_bytes()).map_err(io_error)?;
    file.sync_all().map_err(io_error)
}

fn run_workers_as_processes(
    config: &DatagenConfig,
    settings: &GenerationSettings,
    committed_ids: &HashSet<u64>,
    commits_path: &Path,
) -> Result<(), String> {
    let executable = std::env::current_exe().map_err(io_error)?;
    let mut running = Vec::new();
    let mut launch_failure = None;
    for worker in 0..config.workers {
        let has_work = (settings.start_game_id..settings.end_game_id_exclusive)
            .any(|id| id % settings.workers as u64 == worker as u64 && !committed_ids.contains(&id));
        if !has_work { continue; }
        if let Err(error) = trigger_test_failpoint(&format!("before-spool-create-worker-{worker}")) {
            launch_failure = Some(error);
            break;
        }
        let spool = config.output.join(format!(
            ".worker-spool-w{worker:04}-p{}-{:032x}.tsv", process::id(), unique_nonce()
        ));
        let spool_file = match OpenOptions::new().write(true).create_new(true).open(&spool) {
            Ok(file) => file,
            Err(error) => {
                launch_failure = Some(format!("could not create spool for worker {worker}: {}", io_error(error)));
                break;
            }
        };
        if let Err(error) = spool_file.sync_all() {
            drop(spool_file);
            let cleanup = fs::remove_file(&spool).err().map(io_error);
            launch_failure = Some(match cleanup {
                Some(cleanup) => format!("could not sync spool for worker {worker}: {}; also could not remove spool: {cleanup}", io_error(error)),
                None => format!("could not sync spool for worker {worker}: {}", io_error(error)),
            });
            break;
        }
        drop(spool_file);
        let mut args = vec![
            "--_worker-internal".to_string(),
            "--out".to_string(), config.output.to_string_lossy().into_owned(),
            "--games".to_string(), config.games.to_string(),
            "--base-seed".to_string(), config.base_seed.to_string(),
            "--start-id".to_string(), config.start_game_id.to_string(),
            "--nodes".to_string(), config.nodes.to_string(),
            "--hash-mb".to_string(), config.hash_mb.to_string(),
            "--workers".to_string(), config.workers.to_string(),
            "--chunk-games".to_string(), config.chunk_games.to_string(),
            "--_worker-index".to_string(), worker.to_string(),
            "--_worker-spool".to_string(), spool.to_string_lossy().into_owned(),
            "--_expected-engine-sha256".to_string(), settings.engine_sha256.clone(),
            "--_expected-source-sha256".to_string(), settings.source_sha256.clone(),
            "--_expected-config-sha256".to_string(), settings.config_sha256.clone(),
            "--_expected-evaluator-kind".to_string(), settings.search_evaluator.kind.manifest_name().to_string(),
            "--_expected-nnue-sha256".to_string(), settings.search_evaluator.nnue_sha256.clone().unwrap_or_else(|| "none".to_string()),
        ];
        if let Some(path) = &config.eval_file {
            args.push("--eval-file".to_string());
            args.push(path.to_string_lossy().into_owned());
        }
        match Command::new(&executable).args(args).stdout(Stdio::inherit()).stderr(Stdio::inherit()).spawn() {
            Ok(child) => running.push(RunningWorker { worker, child, spool }),
            Err(error) => {
                let cleanup = fs::remove_file(&spool).err().map(io_error);
                launch_failure = Some(match cleanup {
                    Some(cleanup) => format!("could not spawn worker {worker}: {error}; also could not remove spool: {cleanup}"),
                    None => format!("could not spawn worker {worker}: {error}"),
                });
                break;
            }
        }
    }

    let mut worker_failures = Vec::new();
    for process in &mut running {
        let (status, wait_error) = wait_for_worker_to_reap(process);
        if let Some(error) = wait_error {
            worker_failures.push(error);
        }
        if !status.success() {
            worker_failures.push(format!("worker {} exited with {status}", process.worker));
        }
    }

    // The coordinator is the sole manifest writer. Child processes append
    // only to their own synced spool files, avoiding process-local lock bugs.
    let coordinator_lock = Mutex::new(());
    let mut newly_committed_ids = HashSet::new();
    for process in &running {
        if !process.spool.exists() { continue; }
        repair_unterminated_commit_tail(&process.spool)?;
        let contents = fs::read_to_string(&process.spool).map_err(io_error)?;
        for (line_no, line) in contents.lines().enumerate() {
            if line.is_empty() { continue; }
            let commit = parse_commit(line).map_err(|error| format!("worker {} spool line {}: {error}", process.worker, line_no + 1))?;
            verify_pending_worker_commit(&commit, process.worker, config, settings, committed_ids, &newly_committed_ids)?;
            let path = config.output.join("chunks").join(&commit.filename);
            if sha256_file(&path)? != commit.sha256 {
                return Err(format!("worker {} produced a chunk with a SHA mismatch", process.worker));
            }
            verify_chunk_contents(&path, &commit, settings)?;
            for &id in &commit.ids { newly_committed_ids.insert(id); }
            append_commit(commits_path, &commit, &coordinator_lock)?;
            trigger_test_failpoint("after-manifest-commit")?;
        }
        fs::remove_file(&process.spool).map_err(io_error)?;
    }

    if let Some(error) = launch_failure { return Err(error); }
    if !worker_failures.is_empty() { return Err(worker_failures.join("; ")); }
    Ok(())
}

/// Do not release the coordinator lock while any spawned child might still be
/// running. A failed wait triggers termination and retries until the OS has
/// reaped the child; this also covers unusual wait errors during launch-failure
/// cleanup.
fn wait_for_worker_to_reap(process: &mut RunningWorker) -> (std::process::ExitStatus, Option<String>) {
    match process.child.wait() {
        Ok(status) => (status, None),
        Err(wait_error) => {
            let diagnostic = match process.child.kill() {
                Ok(()) => format!("could not wait for worker {}: {wait_error}; termination requested; child reaped after wait recovery", process.worker),
                Err(kill_error) => format!("could not wait for worker {}: {wait_error}; termination request failed ({kill_error}); child reaped after wait recovery", process.worker),
            };
            loop {
                match process.child.wait() {
                    Ok(status) => return (status, Some(diagnostic)),
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            }
        }
    }
}

fn verify_pending_worker_commit(
    commit: &ChunkCommit,
    expected_worker: usize,
    config: &DatagenConfig,
    settings: &GenerationSettings,
    already_committed: &HashSet<u64>,
    newly_committed: &HashSet<u64>,
) -> Result<(), String> {
    if commit.worker != expected_worker
        || commit.engine_sha256 != settings.engine_sha256
        || commit.source_sha256 != settings.source_sha256
        || commit.config_sha256 != settings.config_sha256
        || commit.ids.is_empty()
        || commit.ids.len() != commit.seeds.len()
        || commit.ids.len() != commit.candidate_counts.len()
        || commit.ids.len() > config.chunk_games
        || commit.ids.windows(2).any(|pair| pair[0] >= pair[1])
        || commit.filename.is_empty()
        || commit.filename.contains('/')
        || commit.filename.contains('\\')
        || commit.filename == "."
        || commit.filename == ".."
    {
        return Err(format!("worker {expected_worker} produced invalid chunk commit metadata"));
    }
    for ((&id, &seed), _) in commit.ids.iter().zip(&commit.seeds).zip(&commit.candidate_counts) {
        if id < settings.start_game_id || id >= settings.end_game_id_exclusive
            || id % settings.workers as u64 != expected_worker as u64
            || seed != game_seed(settings.base_seed, id)
            || already_committed.contains(&id)
            || newly_committed.contains(&id)
        {
            return Err(format!("worker {expected_worker} produced invalid/duplicate game ID {id}"));
        }
    }
    Ok(())
}

/// Named recovery seams remain inert unless explicitly requested by tests.
fn trigger_test_failpoint(name: &str) -> Result<(), String> {
    if std::env::var("KIRA1A_DATAGEN_TEST_FAILPOINT").ok().as_deref() == Some(name) {
        Err(format!("injected datagen test failure at {name}"))
    } else {
        Ok(())
    }
}

fn validate_config(config: &DatagenConfig) -> Result<(), String> {
    if config.games == 0 {
        return Err("--games must be greater than zero".into());
    }
    if config.nodes == 0 {
        return Err("--nodes must be greater than zero".into());
    }
    if config.hash_mb == 0 {
        return Err("--hash-mb must be greater than zero".into());
    }
    if config.workers == 0 || config.workers > u32::MAX as usize {
        return Err("--workers must be in 1..=u32::MAX".into());
    }
    if config.chunk_games == 0 || config.chunk_games > u32::MAX as usize {
        return Err("--chunk-games must be in 1..=u32::MAX".into());
    }
    config
        .start_game_id
        .checked_add(config.games)
        .ok_or_else(|| "game ID range overflows u64".to_string())?;
    Ok(())
}

fn manifest_text(settings: &GenerationSettings) -> String {
    let evaluator_metadata = match settings.search_evaluator.kind {
        // Keep the original M3 manifest byte-for-byte stable so an HCE run can
        // resume from an existing M3 manifest and artifact set unchanged.
        SearchEvaluatorKind::Hce => "evaluator=Kira-HCE\n".to_string(),
        SearchEvaluatorKind::QuantizedNnue128 => format!(
            "evaluator=Kira-HCE\nsearch_evaluator={}\nnnue_sha256={}\n",
            settings.search_evaluator.kind.manifest_name(),
            settings.search_evaluator.nnue_sha256.as_deref().expect("NNUE pin has SHA-256"),
        ),
    };
    format!(
        "format_version={FORMAT_VERSION}\nrecord_version={FORMAT_VERSION}\n{evaluator_metadata}search=Searcher-fixed-nodes\nsearch_limit_policy=nodes Some node_limit; depth None; deadline None; Searcher::clear per game\nprng_version={}\nbase_seed={:016x}\nstart_game_id={}\nend_game_id_exclusive={}\nnode_limit={}\nhash_mb_per_worker={}\nworker_count={}\nchunk_games={}\nengine_sha256={}\nsource_sha256={}\nconfig_sha256={}\nopening_protocol={}\nwin_streak_policy={}\ndedup_policy={}\n",
        PRNG_VERSION,
        settings.base_seed,
        settings.start_game_id,
        settings.end_game_id_exclusive,
        settings.nodes,
        settings.hash_mb,
        settings.workers,
        settings.chunk_games,
        settings.engine_sha256,
        settings.source_sha256,
        settings.config_sha256,
        OPENING_PROTOCOL,
        WIN_STREAK_POLICY,
        DEDUP_POLICY,
    )
}

fn initialize_or_check_manifest(
    manifest_path: &Path,
    commits_path: &Path,
    settings: &GenerationSettings,
) -> Result<(), String> {
    let expected = manifest_text(settings);
    match fs::read_to_string(manifest_path) {
        Ok(actual) if actual != expected => {
            return Err(format!(
                "resume settings differ from pinned manifest {}",
                manifest_path.display()
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if commits_path.exists() {
                return Err("commits.tsv exists but manifest.txt is missing".into());
            }
            let temp = manifest_path.with_extension(format!("tmp-{}", process::id()));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)
                .map_err(io_error)?;
            file.write_all(expected.as_bytes()).map_err(io_error)?;
            file.sync_all().map_err(io_error)?;
            fs::rename(&temp, manifest_path).map_err(io_error)?;
        }
        Err(error) => return Err(io_error(error)),
    }
    if !commits_path.exists() {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(commits_path)
            .map_err(io_error)?
            .sync_all()
            .map_err(io_error)?;
    }
    Ok(())
}

fn repair_unterminated_commit_tail(path: &Path) -> Result<(), String> {
    let mut file = OpenOptions::new().read(true).write(true).open(path).map_err(io_error)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(io_error)?;
    if !bytes.is_empty() && bytes.last() != Some(&b'\n') {
        let last_newline = bytes.iter().rposition(|&byte| byte == b'\n').map_or(0, |i| i + 1);
        file.set_len(last_newline as u64).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
    }
    Ok(())
}

fn load_and_verify_commits(
    root: &Path,
    settings: &GenerationSettings,
) -> Result<Vec<ChunkCommit>, String> {
    let bytes = fs::read(root.join("commits.tsv")).map_err(io_error)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| "commits.tsv is not UTF-8".to_string())?;
    let mut commits = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let commit = parse_commit(line).map_err(|error| format!("commits.tsv line {}: {error}", line_no + 1))?;
        if commit.filename.is_empty()
            || commit.filename.contains('/')
            || commit.filename.contains('\\')
            || commit.filename == "."
            || commit.filename == ".."
        {
            return Err(format!("commits.tsv line {} has an unsafe chunk filename", line_no + 1));
        }
        if commit.engine_sha256 != settings.engine_sha256
            || commit.source_sha256 != settings.source_sha256
            || commit.config_sha256 != settings.config_sha256
        {
            return Err(format!("committed chunk {} has mismatched pinned hashes", commit.filename));
        }
        if commit.worker >= settings.workers {
            return Err(format!("chunk {} has out-of-range worker index", commit.filename));
        }
        let path = root.join("chunks").join(&commit.filename);
        if !path.is_file() {
            return Err(format!("committed chunk is missing: {}", path.display()));
        }
        let actual_sha = sha256_file(&path)?;
        if actual_sha != commit.sha256 {
            return Err(format!("committed chunk hash mismatch: {}", path.display()));
        }
        if commit.ids.is_empty()
            || commit.ids.len() != commit.seeds.len()
            || commit.ids.len() != commit.candidate_counts.len()
        {
            return Err(format!("chunk {} has inconsistent game metadata", commit.filename));
        }
        if commit.ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!("chunk {} game IDs are not ascending", commit.filename));
        }
        for ((&id, &seed), _) in commit
            .ids
            .iter()
            .zip(&commit.seeds)
            .zip(&commit.candidate_counts)
        {
            if id < settings.start_game_id || id >= settings.end_game_id_exclusive {
                return Err(format!("chunk {} contains out-of-range game ID {id}", commit.filename));
            }
            if id % settings.workers as u64 != commit.worker as u64 {
                return Err(format!("game {id} is assigned to the wrong worker"));
            }
            if seed != game_seed(settings.base_seed, id) {
                return Err(format!("game {id} has an incorrect derived seed"));
            }
        }
        verify_chunk_contents(&path, &commit, settings)?;
        commits.push(commit);
    }
    Ok(commits)
}

fn parse_commit(line: &str) -> Result<ChunkCommit, String> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() != 9 {
        return Err(format!("expected 9 tab-separated fields, found {}", fields.len()));
    }
    let worker = fields[1].parse().map_err(|_| "invalid worker".to_string())?;
    let ids = parse_csv(fields[2], |x| x.parse::<u64>().map_err(|_| "invalid game ID".to_string()))?;
    let seeds = parse_csv(fields[3], parse_hex_u64)?;
    let candidate_counts = parse_csv(fields[4], |x| x.parse::<u64>().map_err(|_| "invalid candidate count".to_string()))?;
    for hash in &fields[5..9] {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("invalid SHA-256 field".into());
        }
    }
    Ok(ChunkCommit {
        filename: fields[0].to_string(),
        worker,
        ids,
        seeds,
        candidate_counts,
        sha256: fields[5].to_string(),
        engine_sha256: fields[6].to_string(),
        source_sha256: fields[7].to_string(),
        config_sha256: fields[8].to_string(),
    })
}

fn parse_csv<T>(text: &str, mut parse: impl FnMut(&str) -> Result<T, String>) -> Result<Vec<T>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    text.split(',').map(&mut parse).collect()
}

fn parse_hex_u64(text: &str) -> Result<u64, String> {
    u64::from_str_radix(text, 16).map_err(|_| "invalid hex seed".into())
}

fn commit_line(commit: &ChunkCommit) -> String {
    let ids = commit.ids.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
    let seeds = commit.seeds.iter().map(|x| format!("{x:016x}")).collect::<Vec<_>>().join(",");
    let counts = commit.candidate_counts.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        commit.filename,
        commit.worker,
        ids,
        seeds,
        counts,
        commit.sha256,
        commit.engine_sha256,
        commit.source_sha256,
        commit.config_sha256,
    )
}

fn append_commit(path: &Path, commit: &ChunkCommit, lock: &Mutex<()>) -> Result<(), String> {
    let _guard = lock.lock().map_err(|_| "manifest commit mutex poisoned".to_string())?;
    let mut file = OpenOptions::new().append(true).open(path).map_err(io_error)?;
    file.write_all(commit_line(commit).as_bytes()).map_err(io_error)?;
    file.sync_all().map_err(io_error)
}

fn write_chunk(
    worker: usize,
    games: &[GameRecord],
    config: &DatagenConfig,
    settings: &GenerationSettings,
) -> Result<ChunkCommit, String> {
    if games.is_empty() { return Err("cannot write an empty raw chunk".into()); }
    let ids: Vec<u64> = games.iter().map(|g| g.game_id).collect();
    let seeds: Vec<u64> = games.iter().map(|g| g.seed).collect();
    let candidate_counts: Vec<u64> = games.iter().map(|g| g.eligible_count).collect();
    let unique = unique_nonce();
    let filename = format!(
        "w{worker:04}-g{}-{}-p{}-{unique:032x}.k1c",
        ids[0], ids[ids.len() - 1], process::id()
    );
    let chunks_dir = config.output.join("chunks");
    let final_path = chunks_dir.join(&filename);
    if final_path.exists() {
        return Err(format!("refusing to overwrite existing chunk {}", final_path.display()));
    }
    let temp_path = chunks_dir.join(format!(".{filename}.tmp"));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .map_err(io_error)?;
    let mut writer = BufWriter::new(file);
    trigger_test_failpoint("before-temp-write")?;
    writer.write_all(CHUNK_MAGIC).map_err(io_error)?;
    write_u16(&mut writer, FORMAT_VERSION).map_err(io_error)?;
    write_u32(&mut writer, worker as u32).map_err(io_error)?;
    write_hash(&mut writer, &settings.config_sha256).map_err(io_error)?;
    trigger_test_failpoint("during-temp-write")?;
    for game in games {
        write_game_frame(&mut writer, game).map_err(io_error)?;
    }
    writer.flush().map_err(io_error)?;
    writer.get_ref().sync_all().map_err(io_error)?;
    drop(writer);
    trigger_test_failpoint("after-temp-fsync")?;
    let sha256 = sha256_file(&temp_path)?;
    fs::rename(&temp_path, &final_path).map_err(io_error)?;
    sync_directory(&chunks_dir)?;
    trigger_test_failpoint("after-chunk-rename")?;

    Ok(ChunkCommit {
        filename,
        worker,
        ids,
        seeds,
        candidate_counts,
        sha256,
        engine_sha256: settings.engine_sha256.clone(),
        source_sha256: settings.source_sha256.clone(),
        config_sha256: settings.config_sha256.clone(),
    })
}

fn play_game(
    game_id: u64,
    seed: u64,
    config: &DatagenConfig,
    searcher: &mut Searcher,
    search_evaluator: &SearchEvaluator,
    stop: &AtomicBool,
) -> Result<GameRecord, String> {
    // The game is a pure fixed-node unit: no hash/history/killer state leaks
    // from an earlier game assigned to this worker.
    searcher.clear();
    let mut rng = SplitMix64::seeded(seed);
    let mut opening_attempts = 0u32;
    let (mut board, mut moves, accepted_opening_hce) = loop {
        opening_attempts = opening_attempts.saturating_add(1);
        let mut candidate = Board::startpos();
        let mut opening_moves = Vec::with_capacity(8);
        let mut complete = true;
        for _ in 0..8 {
            let legal = candidate.legal_moves();
            if legal.is_empty() {
                complete = false;
                break;
            }
            let mv = legal[uniform_index(&mut rng, legal.len())];
            opening_moves.push(mv.uci());
            candidate.make_move(mv);
        }
        if !complete {
            continue;
        }
        let opening_hce = opening_hce_score(&candidate);
        if !opening_score_accepted(opening_hce) {
            continue;
        }
        break (candidate, opening_moves, opening_hce);
    };

    let mut history = vec![Board::startpos().hash()];
    let mut replay = Board::startpos();
    for text in &moves {
        replay.play_uci(text)?;
        history.push(replay.hash());
    }
    if replay.to_fen() != board.to_fen() || replay.hash() != board.hash() {
        return Err(format!("opening replay diverged for game {game_id}"));
    }
    let mut candidates = Vec::new();
    let mut events = Vec::new();
    let mut streaks = ScoreStreaks::default();
    let (result_white, end_reason) = loop {
        if let Some(result) = natural_terminal(&mut board, &history) {
            break result;
        }
        let ply = u32::try_from(moves.len()).map_err(|_| format!("game {game_id} exceeds u32 plies"))?;
        let searched = searcher.search(
            &board,
            &history,
            search_evaluator,
            SearchLimits { depth: None, nodes: Some(config.nodes), deadline: None },
            stop,
            |_| {},
        );
        if searched.nodes > config.nodes {
            return Err(format!("search exceeded fixed node limit in game {game_id} at ply {ply}: {} > {}", searched.nodes, config.nodes));
        }
        let Some(best_move) = searched.best_move else {
            return Err(format!("search returned no move in nonterminal game {game_id} at ply {ply}"));
        };
        let class = classify_best_move(&board, best_move);
        let in_check = board.is_in_check(board.white_to_move);
        let reason = candidate_filter_reason(in_check, class, searched.score);
        events.push(SearchEvent {
            fen: board.to_fen(),
            ply,
            white_to_move: board.white_to_move,
            score_stm: searched.score,
            search_call_index: u32::try_from(events.len()).map_err(|_| format!("game {game_id} exceeds u32 search calls"))?,
            node_limit: config.nodes,
            depth_limit: None,
            time_limit_ms: None,
            search_nodes: searched.nodes,
            search_depth: searched.depth,
            best_move_uci: best_move.uci(),
            best_move_class: class,
            filter_reason: reason,
            zobrist: board.hash(),
        });
        if reason == FilterReason::Kept {
            candidates.push(PositionRecord {
                fen: board.to_fen(),
                ply,
                white_to_move: board.white_to_move,
                score_stm: searched.score,
                result_stm: 1, // Filled from the final game result after adjudication.
                best_move_class: class,
                zobrist: board.hash(),
            });
        }

        let score_decision = observe_score(&mut streaks, ply, searched.score, board.white_to_move);

        // Search scores label the current position. Play its best move before
        // adjudicating a fifth-ply win or a completed move-80 score draw.
        moves.push(best_move.uci());
        board.make_move(best_move);
        history.push(board.hash());

        // At p=159 the fortieth in-band score has been observed, but Black
        // must still complete move 80. Resolve p=160 after natural terminals.
        if let Some(outcome) = adjudicate_after_move(
            &mut board,
            &history,
            score_decision,
            &streaks,
            moves.len() as u32,
        ) {
            break to_internal_outcome(outcome);
        }
    };

    for position in &mut candidates {
        position.result_stm = result_white.stm_label(position.white_to_move);
    }
    Ok(GameRecord {
        game_id,
        seed,
        start_fen: START_FEN.to_string(),
        opening_attempts,
        accepted_opening_hce,
        moves,
        result_white,
        end_reason,
        eligible_count: candidates.len() as u64,
        positions: candidates,
        events,
    })
}

fn natural_terminal(board: &mut Board, history: &[u64]) -> Option<(ResultWhite, EndReason)> {
    let legal = board.legal_moves();
    if legal.is_empty() {
        if board.is_in_check(board.white_to_move) {
            return Some((
                if board.white_to_move { ResultWhite::BlackWin } else { ResultWhite::WhiteWin },
                EndReason::Checkmate,
            ));
        }
        return Some((ResultWhite::Draw, EndReason::Stalemate));
    }
    if history.iter().filter(|&&hash| hash == board.hash()).count() >= 3 {
        return Some((ResultWhite::Draw, EndReason::Threefold));
    }
    if board.halfmove >= 100 {
        return Some((ResultWhite::Draw, EndReason::FiftyMove));
    }
    if insufficient_material_basic(board) {
        return Some((ResultWhite::Draw, EndReason::BasicMaterial));
    }
    None
}

fn to_internal_outcome(outcome: PostMoveOutcome) -> (ResultWhite, EndReason) {
    match outcome {
        PostMoveOutcome::Natural(NaturalOutcome { result, reason }) => (
            match result {
                NaturalResult::WhiteWin => ResultWhite::WhiteWin,
                NaturalResult::BlackWin => ResultWhite::BlackWin,
                NaturalResult::Draw => ResultWhite::Draw,
            },
            match reason {
                NaturalReason::Checkmate => EndReason::Checkmate,
                NaturalReason::Stalemate => EndReason::Stalemate,
                NaturalReason::Threefold => EndReason::Threefold,
                NaturalReason::FiftyMove => EndReason::FiftyMove,
                NaturalReason::BasicMaterial => EndReason::BasicMaterial,
            },
        ),
        PostMoveOutcome::ScoreWin(ScoreDecision::WhiteWin) => (ResultWhite::WhiteWin, EndReason::ScoreWin),
        PostMoveOutcome::ScoreWin(ScoreDecision::BlackWin) => (ResultWhite::BlackWin, EndReason::ScoreWin),
        PostMoveOutcome::ScoreDraw => (ResultWhite::Draw, EndReason::ScoreDraw),
    }
}

pub fn candidate_filter_reason(in_check: bool, class: BestMoveClass, score: i32) -> FilterReason {
    if in_check {
        FilterReason::InCheck
    } else if matches!(class, BestMoveClass::Capture | BestMoveClass::CapturePromotion) {
        FilterReason::BestMoveCapture
    } else if matches!(class, BestMoveClass::Promotion | BestMoveClass::CapturePromotion) {
        FilterReason::BestMovePromotion
    } else if score >= MATE_THRESHOLD || score <= -MATE_THRESHOLD {
        FilterReason::MateScore
    } else if score > 2_000 || score < -2_000 {
        FilterReason::ScoreOutsideRange
    } else {
        FilterReason::Kept
    }
}

pub fn classify_best_move(board: &Board, mv: Move) -> BestMoveClass {
    let mover = board.squares[mv.from as usize];
    let capture = board.squares[mv.to as usize] != 0
        || (mover.unsigned_abs() == 1 && mv.from % 8 != mv.to % 8 && board.ep == Some(mv.to));
    match (capture, mv.promotion != 0) {
        (false, false) => BestMoveClass::Quiet,
        (true, false) => BestMoveClass::Capture,
        (false, true) => BestMoveClass::Promotion,
        (true, true) => BestMoveClass::CapturePromotion,
    }
}

fn insufficient_material_basic(board: &Board) -> bool {
    let mut non_king = 0usize;
    let mut sole_kind = 0i8;
    for &piece in &board.squares {
        if piece == 0 || piece.unsigned_abs() == 6 {
            continue;
        }
        non_king += 1;
        sole_kind = piece.unsigned_abs() as i8;
        if non_king > 1 {
            return false;
        }
    }
    non_king == 0 || (non_king == 1 && matches!(sole_kind, 2 | 4))
}

fn game_seed(base_seed: u64, game_id: u64) -> u64 {
    splitmix_finalizer(base_seed ^ game_id)
}

fn splitmix_finalizer(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn uniform_index(rng: &mut SplitMix64, len: usize) -> usize {
    let words = std::iter::from_fn(|| Some(rng.next_u64()));
    uniform_index_from_words(len, words).expect("legal move list is nonempty")
}

fn verify_chunk_contents(path: &Path, commit: &ChunkCommit, settings: &GenerationSettings) -> Result<(), String> {
    let mut reader = ChunkReader::open(path, settings)?;
    if reader.worker != commit.worker {
        return Err(format!("chunk worker header disagrees with commit: {}", path.display()));
    }
    let mut ids = Vec::new();
    let mut seeds = Vec::new();
    let mut counts = Vec::new();
    while let Some(game) = reader.next_game()? {
        validate_game_history(&game, true, Some(settings.nodes), &settings.search_evaluator)?;
        ids.push(game.game_id);
        seeds.push(game.seed);
        counts.push(game.eligible_count);
    }
    if ids != commit.ids || seeds != commit.seeds || counts != commit.candidate_counts {
        return Err(format!("chunk metadata disagrees with manifest commit: {}", path.display()));
    }
    Ok(())
}

fn validate_game_history(
    game: &GameRecord,
    raw_chunk: bool,
    expected_node_limit: Option<u64>,
    search_evaluator: &SearchEvaluatorPin,
) -> Result<(), String> {
    if game.start_fen != START_FEN {
        return Err(format!("game {} has unexpected start FEN", game.game_id));
    }
    if game.moves.len() < 8 {
        return Err(format!("game {} has fewer than eight opening plies", game.game_id));
    }
    if game.opening_attempts == 0 || game.accepted_opening_hce.abs() > 300 {
        return Err(format!("game {} has invalid accepted-opening metadata", game.game_id));
    }
    let mut opening = Board::from_fen(&game.start_fen).map_err(|e| format!("game {} start FEN: {e}", game.game_id))?;
    for text in game.moves.iter().take(8) {
        opening.play_uci(text).map_err(|e| format!("game {} illegal opening: {e}", game.game_id))?;
    }
    if opening_hce_score(&opening) != game.accepted_opening_hce {
        return Err(format!("game {} accepted-opening HCE mismatch", game.game_id));
    }

    let mut event_board = Board::from_fen(&game.start_fen).map_err(|e| format!("game {} start FEN: {e}", game.game_id))?;
    let mut event_ply = 0usize;
    let mut kept_events = 0usize;
    let mut kept_event_rows = Vec::new();
    let pinned_event_node_limit = expected_node_limit.or_else(|| game.events.first().map(|event| event.node_limit));
    for (event_index, event) in game.events.iter().enumerate() {
        let expected_ply = 8 + event_index as u32;
        if event.ply != expected_ply || (event.ply as usize) >= game.moves.len() || (event.ply as usize) < event_ply {
            return Err(format!("game {} has invalid search-event ply {}", game.game_id, event.ply));
        }
        if event.search_call_index != event_index as u32
            || event.node_limit == 0
            || event.node_limit != pinned_event_node_limit.unwrap_or(event.node_limit)
            || event.depth_limit.is_some()
            || event.time_limit_ms.is_some()
            || event.search_nodes > event.node_limit
        {
            return Err(format!("game {} search call {} violates pinned fixed-node accounting", game.game_id, event_index));
        }
        while event_ply < event.ply as usize {
            event_board.play_uci(&game.moves[event_ply]).map_err(|e| format!("game {} illegal event history move: {e}", game.game_id))?;
            event_ply += 1;
        }
        if event_board.to_fen() != event.fen
            || event_board.hash() != event.zobrist
            || event_board.white_to_move != event.white_to_move
        {
            return Err(format!("game {} raw search-event replay mismatch at ply {}", game.game_id, event.ply));
        }
        let parsed = Move::from_uci(&event.best_move_uci)
            .ok_or_else(|| format!("game {} has malformed raw bestmove", game.game_id))?;
        let legal = event_board.legal_moves();
        // UCI promotion text does not encode piece color. `parse_uci_move`
        // therefore normalizes promotion pieces to a negative value, while
        // legal moves carry the mover's signed piece code. Compare the UCI
        // identity (including promotion kind) rather than the signed struct.
        let best = legal.iter().copied().find(|mv| {
            mv.from == parsed.from
                && mv.to == parsed.to
                && mv.promotion.abs() == parsed.promotion.abs()
        })
            .ok_or_else(|| format!("game {} raw bestmove is illegal at ply {}", game.game_id, event.ply))?;
        let class = classify_best_move(&event_board, best);
        if class != event.best_move_class
            || candidate_filter_reason(event_board.is_in_check(event_board.white_to_move), class, event.score_stm) != event.filter_reason
        {
            return Err(format!("game {} raw filter trace mismatch at ply {}", game.game_id, event.ply));
        }
        kept_events += usize::from(event.filter_reason == FilterReason::Kept);
        if event.filter_reason == FilterReason::Kept { kept_event_rows.push(event); }
    }
    if (raw_chunk && kept_events != game.positions.len()) || kept_events < game.positions.len() {
        return Err(format!("game {} kept-event/candidate count mismatch", game.game_id));
    }
    for position in &game.positions {
        let event = kept_event_rows.iter().find(|event| event.ply == position.ply && event.zobrist == position.zobrist)
            .ok_or_else(|| format!("game {} candidate has no kept raw event", game.game_id))?;
        if event.fen != position.fen
            || event.white_to_move != position.white_to_move
            || event.score_stm != position.score_stm
            || event.best_move_class != position.best_move_class
        { return Err(format!("game {} candidate row differs from raw event", game.game_id)); }
    }

    let mut board = Board::from_fen(&game.start_fen).map_err(|e| format!("game {} start FEN: {e}", game.game_id))?;
    let mut by_ply = 0usize;
    for position in &game.positions {
        if position.ply as usize >= game.moves.len() {
            return Err(format!("game {} candidate ply {} is past game end", game.game_id, position.ply));
        }
        if (position.ply as usize) < by_ply {
            return Err(format!("game {} candidate plies are not ascending", game.game_id));
        }
        while by_ply < position.ply as usize {
            board.play_uci(&game.moves[by_ply]).map_err(|e| format!("game {} illegal history move {}: {e}", game.game_id, by_ply))?;
            by_ply += 1;
        }
        let decoded = Board::from_fen(&position.fen).map_err(|e| format!("game {} candidate FEN: {e}", game.game_id))?;
        if decoded.to_fen() != position.fen || decoded.hash() != position.zobrist {
            return Err(format!("game {} candidate FEN/hash failed canonical audit at ply {}", game.game_id, position.ply));
        }
        if board.to_fen() != position.fen
            || board.hash() != position.zobrist
            || board.white_to_move != position.white_to_move
        {
            return Err(format!("game {} replay mismatch at ply {}", game.game_id, position.ply));
        }
        if position.result_stm != game.result_white.stm_label(position.white_to_move) {
            return Err(format!("game {} has inconsistent result perspective", game.game_id));
        }
    }
    for text in game.moves.iter().skip(by_ply) {
        board.play_uci(text).map_err(|e| format!("game {} illegal trailing history: {e}", game.game_id))?;
    }
    if (raw_chunk && game.eligible_count != game.positions.len() as u64)
        || (!raw_chunk && game.eligible_count < game.positions.len() as u64)
    {
        return Err(format!("game {} candidate count mismatch", game.game_id));
    }
    validate_adjudication(game, search_evaluator)?;
    Ok(())
}

fn validate_adjudication(game: &GameRecord, search_evaluator: &SearchEvaluatorPin) -> Result<(), String> {
    // Search scores remain centipawn, side-to-move values for either source;
    // evaluator identity pins their provenance while all M3 streak and natural
    // terminal rules remain independent of whether HCE or NNUE produced them.
    if (search_evaluator.kind == SearchEvaluatorKind::Hce) != search_evaluator.nnue_sha256.is_none() {
        return Err("invalid pinned search evaluator metadata during replay".into());
    }
    let mut board = Board::from_fen(&game.start_fen).map_err(io_error)?;
    let mut history = vec![board.hash()];
    for text in game.moves.iter().take(8) {
        board.play_uci(text).map_err(io_error)?;
        history.push(board.hash());
    }
    let mut resolved = natural_terminal(&mut board, &history);
    let mut streaks = ScoreStreaks::default();
    let mut processed_events = 0usize;
    if resolved.is_none() {
        for event in &game.events {
            if event.ply as usize >= game.moves.len() || game.moves[event.ply as usize] != event.best_move_uci {
                return Err(format!("game {} history move differs from raw bestmove", game.game_id));
            }
            let score_decision = observe_score(&mut streaks, event.ply, event.score_stm, event.white_to_move);
            board.play_uci(&event.best_move_uci).map_err(io_error)?;
            history.push(board.hash());
            processed_events += 1;
            resolved = adjudicate_after_move(
                &mut board,
                &history,
                score_decision,
                &streaks,
                event.ply + 1,
            ).map(to_internal_outcome);
            if resolved.is_some() { break; }
        }
    }
    if processed_events != game.events.len() || game.moves.len() != 8 + processed_events {
        return Err(format!("game {} contains moves/events after its adjudication", game.game_id));
    }
    let Some((result, reason)) = resolved else {
        return Err(format!("game {} has no reproducible terminal/adjudication", game.game_id));
    };
    if result != game.result_white || reason != game.end_reason {
        return Err(format!("game {} stored result/reason differs from replayed adjudication", game.game_id));
    }
    Ok(())
}

struct ChunkReader {
    reader: BufReader<File>,
    worker: usize,
    finished: bool,
}

impl ChunkReader {
    fn open(path: &Path, settings: &GenerationSettings) -> Result<Self, String> {
        let mut reader = BufReader::new(File::open(path).map_err(io_error)?);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).map_err(io_error)?;
        if &magic != CHUNK_MAGIC {
            return Err(format!("bad chunk magic in {}", path.display()));
        }
        let version = read_u16(&mut reader).map_err(io_error)?;
        if version != FORMAT_VERSION {
            return Err(format!("unsupported chunk version {version}"));
        }
        let worker = read_u32(&mut reader).map_err(io_error)? as usize;
        let hash = read_hash(&mut reader).map_err(io_error)?;
        if hash != settings.config_sha256 {
            return Err(format!("chunk config hash mismatch in {}", path.display()));
        }
        Ok(Self { reader, worker, finished: false })
    }

    fn next_game(&mut self) -> Result<Option<GameRecord>, String> {
        if self.finished {
            return Ok(None);
        }
        let mut length_bytes = [0u8; 8];
        match self.reader.read(&mut length_bytes[..1]) {
            Ok(0) => {
                self.finished = true;
                return Ok(None);
            }
            Ok(1) => {}
            Ok(_) => unreachable!(),
            Err(error) => return Err(io_error(error)),
        }
        self.reader.read_exact(&mut length_bytes[1..]).map_err(io_error)?;
        let length = u64::from_le_bytes(length_bytes);
        if length > usize::MAX as u64 || length > 256 * 1024 * 1024 {
            return Err("game record exceeds 256 MiB safety bound".into());
        }
        let mut payload = vec![0u8; length as usize];
        self.reader.read_exact(&mut payload).map_err(io_error)?;
        decode_game(&payload).map(Some)
    }
}

struct WorkerStream {
    chunks: VecDeque<ChunkCommit>,
    current: Option<ChunkReader>,
    current_game: Option<GameRecord>,
    root: PathBuf,
    settings: GenerationSettings,
    last_id: Option<u64>,
}

impl WorkerStream {
    fn new(root: &Path, settings: &GenerationSettings, mut chunks: Vec<ChunkCommit>) -> Result<Self, String> {
        chunks.sort_by_key(|chunk| chunk.ids.first().copied().unwrap_or(u64::MAX));
        let mut stream = Self {
            chunks: chunks.into(),
            current: None,
            current_game: None,
            root: root.to_path_buf(),
            settings: settings.clone(),
            last_id: None,
        };
        stream.advance()?;
        Ok(stream)
    }

    fn advance(&mut self) -> Result<(), String> {
        loop {
            if let Some(reader) = &mut self.current {
                if let Some(game) = reader.next_game()? {
                    if let Some(last) = self.last_id {
                        if game.game_id <= last {
                            return Err(format!("worker chunks are not increasing at game {}", game.game_id));
                        }
                    }
                    self.last_id = Some(game.game_id);
                    validate_game_history(&game, true, Some(self.settings.nodes), &self.settings.search_evaluator)?;
                    self.current_game = Some(game);
                    return Ok(());
                }
                self.current = None;
            }
            let Some(commit) = self.chunks.pop_front() else {
                self.current_game = None;
                return Ok(())
            };
            let path = self.root.join("chunks").join(commit.filename);
            self.current = Some(ChunkReader::open(&path, &self.settings)?);
        }
    }
}

fn merge_committed_chunks(
    root: &Path,
    settings: &GenerationSettings,
    commits: &[ChunkCommit],
    total_games: u64,
) -> Result<DatagenSummary, String> {
    let mut streams = Vec::new();
    for worker in 0..settings.workers {
        let chunks = commits.iter().filter(|chunk| chunk.worker == worker).cloned().collect();
        streams.push(WorkerStream::new(root, settings, chunks)?);
    }

    let temp_path = root.join(format!(".dataset-tmp-{}-{:032x}", process::id(), unique_nonce()));
    let file = OpenOptions::new().write(true).create_new(true).open(&temp_path).map_err(io_error)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(DATASET_MAGIC).map_err(io_error)?;
    write_u16(&mut writer, FORMAT_VERSION).map_err(io_error)?;
    write_hash(&mut writer, &settings.config_sha256).map_err(io_error)?;
    let totals_offset = 8 + 2 + 32;
    for _ in 0..4 {
        write_u64(&mut writer, 0).map_err(io_error)?;
    }

    let mut seen = HashSet::<u64>::new();
    let (mut eligible, mut duplicates, mut unique, mut merged_games) = (0u64, 0u64, 0u64, 0u64);
    loop {
        let selected = streams
            .iter()
            .enumerate()
            .filter_map(|(index, stream)| stream.current_game.as_ref().map(|game| (index, game.game_id)))
            .min_by_key(|&(_, game_id)| game_id)
            .map(|(index, _)| index);
        let Some(index) = selected else { break };
        let mut game = streams[index].current_game.take().expect("selected current game");
        if game.game_id < settings.start_game_id || game.game_id >= settings.end_game_id_exclusive {
            return Err(format!("merged game ID {} is outside configured range", game.game_id));
        }
        let mut retained = Vec::new();
        for position in game.positions.drain(..) {
            eligible = eligible.checked_add(1).ok_or_else(|| "candidate counter overflow".to_string())?;
            if seen.insert(position.zobrist) {
                unique += 1;
                retained.push(position);
            } else {
                duplicates += 1;
            }
        }
        game.positions = retained;
        write_game_frame(&mut writer, &game).map_err(io_error)?;
        merged_games += 1;
        streams[index].advance()?;
    }
    if merged_games != total_games {
        return Err(format!("merger consumed {merged_games} of {total_games} committed games"));
    }
    writer.flush().map_err(io_error)?;
    let file = writer.get_mut();
    file.seek(SeekFrom::Start(totals_offset as u64)).map_err(io_error)?;
    for count in [merged_games, eligible, duplicates, unique] {
        write_u64(file, count).map_err(io_error)?;
    }
    file.flush().map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    drop(writer);

    let dataset_sha256 = sha256_file(&temp_path)?;
    let final_path = root.join(format!("dataset-{dataset_sha256}.k1d"));
    if final_path.exists() {
        let existing_sha = sha256_file(&final_path)?;
        if existing_sha != dataset_sha256 {
            return Err(format!("refusing to overwrite mismatched dataset {}", final_path.display()));
        }
        fs::remove_file(&temp_path).map_err(io_error)?;
    } else {
        fs::rename(&temp_path, &final_path).map_err(io_error)?;
        sync_directory(root)?;
    }
    Ok(DatagenSummary {
        dataset: final_path,
        dataset_sha256,
        games: merged_games,
        eligible_pre_dedup: eligible,
        duplicate_candidates: duplicates,
        unique_positions: unique,
    })
}

fn write_game_frame(writer: &mut impl Write, game: &GameRecord) -> io::Result<()> {
    let mut payload = Vec::new();
    payload.push(1); // Complete-game block.
    write_u64(&mut payload, game.game_id)?;
    write_u64(&mut payload, game.seed)?;
    write_string(&mut payload, &game.start_fen)?;
    write_u32(&mut payload, game.opening_attempts)?;
    write_i32(&mut payload, game.accepted_opening_hce)?;
    payload.push(game.result_white as u8);
    payload.push(game.end_reason as u8);
    write_u64(&mut payload, game.eligible_count)?;
    write_u32(&mut payload, u32::try_from(game.moves.len()).map_err(|_| io::Error::other("too many moves"))?)?;
    for text in &game.moves {
        write_string(&mut payload, text)?;
    }
    write_u32(&mut payload, u32::try_from(game.events.len()).map_err(|_| io::Error::other("too many search events"))?)?;
    for event in &game.events {
        let mut record = Vec::new();
        record.push(1); // Raw search-event record.
        write_string(&mut record, &event.fen)?;
        write_u32(&mut record, event.ply)?;
        record.push(u8::from(!event.white_to_move));
        write_i32(&mut record, event.score_stm)?;
        write_u32(&mut record, event.search_call_index)?;
        write_u64(&mut record, event.node_limit)?;
        write_option_u8(&mut record, event.depth_limit)?;
        write_option_u64(&mut record, event.time_limit_ms)?;
        write_u64(&mut record, event.search_nodes)?;
        record.push(event.search_depth);
        write_string(&mut record, &event.best_move_uci)?;
        record.push(event.best_move_class as u8);
        record.push(event.filter_reason as u8);
        write_u64(&mut record, event.zobrist)?;
        write_u32(&mut payload, u32::try_from(record.len()).map_err(|_| io::Error::other("search event too long"))?)?;
        payload.extend_from_slice(&record);
    }
    write_u32(&mut payload, u32::try_from(game.positions.len()).map_err(|_| io::Error::other("too many records"))?)?;
    for position in &game.positions {
        let mut record = Vec::new();
        record.push(1); // Candidate-position record type.
        write_string(&mut record, &position.fen)?;
        write_u32(&mut record, position.ply)?;
        record.push(u8::from(!position.white_to_move));
        write_i32(&mut record, position.score_stm)?;
        record.push(position.result_stm);
        record.push(position.best_move_class as u8);
        write_u64(&mut record, position.zobrist)?;
        write_u32(&mut payload, u32::try_from(record.len()).map_err(|_| io::Error::other("position record too long"))?)?;
        payload.extend_from_slice(&record);
    }
    write_u64(writer, payload.len() as u64)?;
    writer.write_all(&payload)
}

fn decode_game(payload: &[u8]) -> Result<GameRecord, String> {
    let mut cursor = io::Cursor::new(payload);
    if read_u8(&mut cursor).map_err(io_error)? != 1 {
        return Err("unknown game block type".into());
    }
    let game_id = read_u64(&mut cursor).map_err(io_error)?;
    let seed = read_u64(&mut cursor).map_err(io_error)?;
    let start_fen = read_string(&mut cursor).map_err(io_error)?;
    let opening_attempts = read_u32(&mut cursor).map_err(io_error)?;
    let accepted_opening_hce = read_i32(&mut cursor).map_err(io_error)?;
    let result_white = match read_u8(&mut cursor).map_err(io_error)? {
        0 => ResultWhite::BlackWin,
        1 => ResultWhite::Draw,
        2 => ResultWhite::WhiteWin,
        _ => return Err("unknown game result code".into()),
    };
    let end_reason = match read_u8(&mut cursor).map_err(io_error)? {
        0 => EndReason::Checkmate,
        1 => EndReason::Stalemate,
        2 => EndReason::Threefold,
        3 => EndReason::FiftyMove,
        4 => EndReason::BasicMaterial,
        5 => EndReason::ScoreWin,
        6 => EndReason::ScoreDraw,
        _ => return Err("unknown end reason code".into()),
    };
    let eligible_count = read_u64(&mut cursor).map_err(io_error)?;
    let move_count = read_u32(&mut cursor).map_err(io_error)? as usize;
    if move_count > 1_000_000 {
        return Err("game history exceeds one million plies".into());
    }
    let mut moves = Vec::with_capacity(move_count);
    for _ in 0..move_count {
        moves.push(read_string(&mut cursor).map_err(io_error)?);
    }
    let event_count = read_u32(&mut cursor).map_err(io_error)? as usize;
    if event_count > 1_000_000 {
        return Err("search event count exceeds one million".into());
    }
    let mut events = Vec::with_capacity(event_count);
    for _ in 0..event_count {
        let length = read_u32(&mut cursor).map_err(io_error)? as usize;
        if length > 8192 {
            return Err("search event exceeds safety bound".into());
        }
        let mut bytes = vec![0; length];
        cursor.read_exact(&mut bytes).map_err(io_error)?;
        events.push(decode_event(&bytes)?);
    }
    let position_count = read_u32(&mut cursor).map_err(io_error)? as usize;
    if position_count > 10_000_000 {
        return Err("candidate count exceeds safety bound".into());
    }
    let mut positions = Vec::with_capacity(position_count);
    for _ in 0..position_count {
        let length = read_u32(&mut cursor).map_err(io_error)? as usize;
        if length > 4096 {
            return Err("position record exceeds safety bound".into());
        }
        let mut bytes = vec![0; length];
        cursor.read_exact(&mut bytes).map_err(io_error)?;
        positions.push(decode_position(&bytes)?);
    }
    if cursor.position() != payload.len() as u64 {
        return Err("trailing bytes in game block".into());
    }
    Ok(GameRecord {
        game_id,
        seed,
        start_fen,
        opening_attempts,
        accepted_opening_hce,
        moves,
        result_white,
        end_reason,
        eligible_count,
        positions,
        events,
    })
}

fn decode_event(payload: &[u8]) -> Result<SearchEvent, String> {
    let mut cursor = io::Cursor::new(payload);
    if read_u8(&mut cursor).map_err(io_error)? != 1 {
        return Err("unknown search event type".into());
    }
    let fen = read_string(&mut cursor).map_err(io_error)?;
    let ply = read_u32(&mut cursor).map_err(io_error)?;
    let side = read_u8(&mut cursor).map_err(io_error)?;
    if side > 1 { return Err("invalid search-event side-to-move code".into()); }
    let score_stm = read_i32(&mut cursor).map_err(io_error)?;
    let search_call_index = read_u32(&mut cursor).map_err(io_error)?;
    let node_limit = read_u64(&mut cursor).map_err(io_error)?;
    let depth_limit = read_option_u8(&mut cursor).map_err(io_error)?;
    let time_limit_ms = read_option_u64(&mut cursor).map_err(io_error)?;
    let search_nodes = read_u64(&mut cursor).map_err(io_error)?;
    let search_depth = read_u8(&mut cursor).map_err(io_error)?;
    let best_move_uci = read_string(&mut cursor).map_err(io_error)?;
    let best_move_class = decode_best_move_class(read_u8(&mut cursor).map_err(io_error)?)?;
    let filter_reason = match read_u8(&mut cursor).map_err(io_error)? {
        0 => FilterReason::Kept,
        1 => FilterReason::InCheck,
        2 => FilterReason::BestMoveCapture,
        3 => FilterReason::BestMovePromotion,
        4 => FilterReason::MateScore,
        5 => FilterReason::ScoreOutsideRange,
        _ => return Err("unknown filter reason".into()),
    };
    let zobrist = read_u64(&mut cursor).map_err(io_error)?;
    if cursor.position() != payload.len() as u64 { return Err("trailing bytes in search event".into()); }
    Ok(SearchEvent {
        fen,
        ply,
        white_to_move: side == 0,
        score_stm,
        search_call_index,
        node_limit,
        depth_limit,
        time_limit_ms,
        search_nodes,
        search_depth,
        best_move_uci,
        best_move_class,
        filter_reason,
        zobrist,
    })
}

fn decode_best_move_class(code: u8) -> Result<BestMoveClass, String> {
    match code {
        0 => Ok(BestMoveClass::Quiet),
        1 => Ok(BestMoveClass::Capture),
        2 => Ok(BestMoveClass::Promotion),
        3 => Ok(BestMoveClass::CapturePromotion),
        _ => Err("unknown best-move classification".into()),
    }
}

fn decode_position(payload: &[u8]) -> Result<PositionRecord, String> {
    let mut cursor = io::Cursor::new(payload);
    if read_u8(&mut cursor).map_err(io_error)? != 1 {
        return Err("unknown candidate record type".into());
    }
    let fen = read_string(&mut cursor).map_err(io_error)?;
    let ply = read_u32(&mut cursor).map_err(io_error)?;
    let side = read_u8(&mut cursor).map_err(io_error)?;
    if side > 1 {
        return Err("invalid side-to-move code".into());
    }
    let score_stm = read_i32(&mut cursor).map_err(io_error)?;
    let result_stm = read_u8(&mut cursor).map_err(io_error)?;
    if result_stm > 2 {
        return Err("invalid STM result code".into());
    }
    let best_move_class = decode_best_move_class(read_u8(&mut cursor).map_err(io_error)?)?;
    let zobrist = read_u64(&mut cursor).map_err(io_error)?;
    if cursor.position() != payload.len() as u64 {
        return Err("trailing bytes in candidate record".into());
    }
    Ok(PositionRecord {
        fen,
        ply,
        white_to_move: side == 0,
        score_stm,
        result_stm,
        best_move_class,
        zobrist,
    })
}

/// Decode and replay-audit one raw worker chunk or merged dataset. The
/// callback receives complete game histories plus candidate rows and raw
/// bestmove/filter traces without loading the whole generation into memory.
pub fn inspect_artifact(
    path: &Path,
    mut visit_game: impl FnMut(&DecodedGame),
) -> Result<ArtifactSummary, String> {
    let mut reader = BufReader::new(File::open(path).map_err(io_error)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(io_error)?;
    let version = read_u16(&mut reader).map_err(io_error)?;
    if version != FORMAT_VERSION {
        return Err(format!("unsupported artifact version {version}"));
    }
    let (config_sha256, kind, worker, raw_chunk, header_totals) = if &magic == CHUNK_MAGIC {
        let worker = read_u32(&mut reader).map_err(io_error)?;
        let config_sha256 = read_hash(&mut reader).map_err(io_error)?;
        (config_sha256, "raw-chunk".to_string(), Some(worker), true, None)
    } else if &magic == DATASET_MAGIC {
        let config_sha256 = read_hash(&mut reader).map_err(io_error)?;
        let totals = [
            read_u64(&mut reader).map_err(io_error)?,
            read_u64(&mut reader).map_err(io_error)?,
            read_u64(&mut reader).map_err(io_error)?,
            read_u64(&mut reader).map_err(io_error)?,
        ];
        (config_sha256, "merged-dataset".to_string(), None, false, Some(totals))
    } else {
        return Err(format!("unrecognized artifact magic in {}", path.display()));
    };
    let search_evaluator = artifact_search_evaluator(path, &config_sha256)?;

    let mut games = 0u64;
    let mut eligible = 0u64;
    let mut stored_candidates = 0u64;
    let mut events = 0u64;
    let mut artifact_node_limit = None;
    let mut previous_game_id = None;
    let mut merged_hashes = HashSet::new();
    loop {
        let Some(game) = read_game_frame(&mut reader)? else { break };
        validate_game_history(&game, raw_chunk, None, &search_evaluator)?;
        if let Some(previous) = previous_game_id {
            if game.game_id <= previous {
                return Err(format!("artifact game IDs are not globally ascending: {} after {previous}", game.game_id));
            }
        }
        previous_game_id = Some(game.game_id);
        if !raw_chunk {
            for candidate in &game.positions {
                if !merged_hashes.insert(candidate.zobrist) {
                    return Err(format!("merged dataset repeats Zobrist key {:016x}", candidate.zobrist));
                }
            }
        }
        for event in &game.events {
            match artifact_node_limit {
                Some(limit) if limit != event.node_limit => {
                    return Err(format!(
                        "artifact mixes fixed node limits {limit} and {}",
                        event.node_limit
                    ));
                }
                None => artifact_node_limit = Some(event.node_limit),
                _ => {}
            }
        }
        games += 1;
        eligible = eligible.checked_add(game.eligible_count).ok_or_else(|| "eligible count overflow".to_string())?;
        stored_candidates = stored_candidates.checked_add(game.positions.len() as u64).ok_or_else(|| "stored count overflow".to_string())?;
        events = events.checked_add(game.events.len() as u64).ok_or_else(|| "event count overflow".to_string())?;
        let decoded = to_decoded_game(game);
        visit_game(&decoded);
    }
    let (duplicates, unique_positions) = if let Some([expected_games, expected_eligible, expected_duplicates, expected_unique]) = header_totals {
        if games != expected_games || eligible != expected_eligible || stored_candidates != expected_unique {
            return Err("merged dataset header totals disagree with decoded records".into());
        }
        if expected_eligible.checked_sub(expected_duplicates) != Some(expected_unique) {
            return Err("merged dataset duplicate/unique totals are inconsistent".into());
        }
        (expected_duplicates, expected_unique)
    } else {
        if eligible != stored_candidates {
            return Err("raw chunk candidate count disagrees with stored rows".into());
        }
        (0, stored_candidates)
    };
    Ok(ArtifactSummary {
        kind,
        sha256: sha256_file(path)?,
        config_sha256,
        games,
        eligible_pre_dedup: eligible,
        stored_candidate_rows: stored_candidates,
        search_events: events,
        duplicates,
        unique_positions,
        worker,
        node_limit: artifact_node_limit,
        search_evaluator: search_evaluator.kind.manifest_name().to_string(),
        nnue_sha256: search_evaluator.nnue_sha256,
    })
}

fn artifact_search_evaluator(path: &Path, artifact_config_sha256: &str) -> Result<SearchEvaluatorPin, String> {
    let adjacent_manifest = path.parent().and_then(|parent| {
        let direct = parent.join("manifest.txt");
        if direct.is_file() {
            Some(direct)
        } else {
            parent.parent().map(|root| root.join("manifest.txt")).filter(|candidate| candidate.is_file())
        }
    });
    let Some(manifest_path) = adjacent_manifest else {
        return Err(format!(
            "cannot identify search evaluator for detached artifact {}; adjacent manifest.txt is required",
            path.display()
        ));
    };
    let manifest = fs::read_to_string(&manifest_path).map_err(io_error)?;
    let fields: std::collections::HashMap<&str, &str> = manifest
        .lines()
        .filter_map(|line| line.split_once('='))
        .collect();
    if fields.get("config_sha256").copied() != Some(artifact_config_sha256) {
        return Err(format!(
            "artifact config SHA-256 does not match adjacent manifest {}",
            manifest_path.display()
        ));
    }
    match fields.get("search_evaluator").copied() {
        None if fields.get("evaluator").copied() == Some("Kira-HCE") => {
            if fields.contains_key("nnue_sha256") {
                return Err("HCE manifest unexpectedly contains an NNUE SHA-256".into());
            }
            Ok(SearchEvaluatorPin { kind: SearchEvaluatorKind::Hce, nnue_sha256: None })
        }
        Some("QuantizedNnueNetwork<128>") => {
            let sha = fields.get("nnue_sha256").copied()
                .ok_or_else(|| "NNUE manifest lacks nnue_sha256".to_string())?;
            if sha.len() != 64 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err("NNUE manifest contains an invalid nnue_sha256".into());
            }
            Ok(SearchEvaluatorPin {
                kind: SearchEvaluatorKind::QuantizedNnue128,
                nnue_sha256: Some(sha.to_ascii_lowercase()),
            })
        }
        Some(other) => Err(format!("unsupported search evaluator in manifest: {other}")),
        None => Err("manifest does not identify a supported search evaluator".into()),
    }
}

fn read_game_frame(reader: &mut impl Read) -> Result<Option<GameRecord>, String> {
    let mut length_bytes = [0u8; 8];
    match reader.read(&mut length_bytes[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => {}
        Ok(_) => unreachable!(),
        Err(error) => return Err(io_error(error)),
    }
    reader.read_exact(&mut length_bytes[1..]).map_err(io_error)?;
    let length = u64::from_le_bytes(length_bytes);
    if length > usize::MAX as u64 || length > 256 * 1024 * 1024 {
        return Err("game record exceeds 256 MiB safety bound".into());
    }
    let mut payload = vec![0u8; length as usize];
    reader.read_exact(&mut payload).map_err(io_error)?;
    decode_game(&payload).map(Some)
}

fn to_decoded_game(game: GameRecord) -> DecodedGame {
    DecodedGame {
        game_id: game.game_id,
        seed: game.seed,
        start_fen: game.start_fen,
        opening_attempts: game.opening_attempts,
        accepted_opening_hce: game.accepted_opening_hce,
        moves: game.moves,
        result_white: game.result_white as u8,
        end_reason: game.end_reason,
        eligible_count_pre_dedup: game.eligible_count,
        candidates: game.positions.into_iter().map(|position| DecodedCandidate {
            fen: position.fen,
            ply: position.ply,
            white_to_move: position.white_to_move,
            score_stm: position.score_stm,
            result_stm: position.result_stm,
            best_move_class: position.best_move_class,
            zobrist: position.zobrist,
        }).collect(),
        search_events: game.events.into_iter().map(|event| DecodedSearchEvent {
            fen: event.fen,
            ply: event.ply,
            white_to_move: event.white_to_move,
            score_stm: event.score_stm,
            search_call_index: event.search_call_index,
            node_limit: event.node_limit,
            depth_limit: event.depth_limit,
            time_limit_ms: event.time_limit_ms,
            search_nodes: event.search_nodes,
            search_depth: event.search_depth,
            best_move_uci: event.best_move_uci,
            best_move_class: event.best_move_class,
            filter_reason: event.filter_reason,
            zobrist: event.zobrist,
        }).collect(),
    }
}

fn configuration_sha256(config: &DatagenConfig, end: u64, evaluator: &SearchEvaluatorPin) -> String {
    let evaluator_metadata = match evaluator.kind {
        // This exact legacy text preserves M3's config hash and artifact IDs.
        SearchEvaluatorKind::Hce => "evaluator=Kira-HCE\n".to_string(),
        SearchEvaluatorKind::QuantizedNnue128 => format!(
            "evaluator=Kira-HCE\nsearch_evaluator={}\nnnue_sha256={}\n",
            evaluator.kind.manifest_name(),
            evaluator.nnue_sha256.as_deref().expect("NNUE pin has SHA-256"),
        ),
    };
    let text = format!(
        "record_version={FORMAT_VERSION}\n{evaluator_metadata}search=Searcher-fixed-nodes\nsearch_limit_policy=nodes Some node_limit; depth None; deadline None; Searcher::clear per game\nprng_version={}\nbase_seed={:016x}\nstart={}\nend={}\nnodes={}\nhash_mb={}\nworkers={}\nchunk_games={}\nopening_protocol={}\nwin_streak_policy={}\ndedup_policy={}\n",
        PRNG_VERSION,
        config.base_seed,
        config.start_game_id,
        end,
        config.nodes,
        config.hash_mb,
        config.workers,
        config.chunk_games,
        OPENING_PROTOCOL,
        WIN_STREAK_POLICY,
        DEDUP_POLICY,
    );
    sha256_hex(text.as_bytes())
}

fn source_tree_sha256() -> Result<String, String> {
    // Resolve the source root at runtime so release binaries remain byte-for-
    // byte reproducible across checkout paths. Embedding CARGO_MANIFEST_DIR
    // would make otherwise identical binaries hash differently by location.
    let executable = std::env::current_exe().map_err(io_error)?;
    let root = executable
        .ancestors()
        .find(|candidate| {
            candidate.join("Cargo.toml").is_file()
                && candidate.join("src").join("datagen.rs").is_file()
        })
        .map(Path::to_path_buf)
        .or_else(|| {
            std::env::current_dir().ok().filter(|candidate| {
                candidate.join("Cargo.toml").is_file()
                    && candidate.join("src").join("datagen.rs").is_file()
            })
        })
        .ok_or_else(|| "unable to locate project source root from executable or current directory".to_string())?;
    let mut files = Vec::new();
    collect_rust_files(&root.join("src"), &mut files)?;
    for name in ["Cargo.toml", "Cargo.lock"] {
        let path = root.join(name);
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(&root).map_err(|e| e.to_string())?.to_string_lossy();
        let bytes = fs::read(&path).map_err(io_error)?;
        hash.update(&(relative.len() as u64).to_le_bytes());
        hash.update(relative.as_bytes());
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
    }
    Ok(hash.finish_hex())
}

fn collect_rust_files(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, output)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
    Ok(())
}

fn unique_nonce() -> u128 {
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    ((process::id() as u128) << 96) ^ time
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path).and_then(|file| file.sync_all()).map_err(io_error)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(io_error)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(io_error)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finish_hex())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finish_hex()
}

struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length_bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buffer: [0; 64],
            buffered: 0,
            length_bytes: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.length_bytes = self.length_bytes.wrapping_add(input.len() as u64);
        if self.buffered != 0 {
            let take = (64 - self.buffered).min(input.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&input[..take]);
            self.buffered += take;
            input = &input[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while input.len() >= 64 {
            let block: &[u8; 64] = input[..64].try_into().expect("block length");
            self.compress(block);
            input = &input[64..];
        }
        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffered = input.len();
        }
    }

    fn finish_hex(mut self) -> String {
        let bit_len = self.length_bytes.wrapping_mul(8);
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
            self.buffered = 0;
        }
        self.buffer[self.buffered..56].fill(0);
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        self.state.iter().map(|word| format!("{word:08x}")).collect()
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
        ];
        let mut words = [0u32; 64];
        for (i, slot) in words.iter_mut().take(16).enumerate() {
            *slot = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().expect("word"));
        }
        for i in 16..64 {
            let x = words[i - 15];
            let y = words[i - 2];
            let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
            let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
            words[i] = words[i - 16].wrapping_add(s0).wrapping_add(words[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ (!e & g);
            let t1 = h.wrapping_add(s1).wrapping_add(choice).wrapping_add(K[i]).wrapping_add(words[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            h = g; g = f; f = e; e = d.wrapping_add(t1);
            d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (state, value) in self.state.iter_mut().zip([a,b,c,d,e,f,g,h]) {
            *state = state.wrapping_add(value);
        }
    }
}

fn write_hash(writer: &mut impl Write, hex: &str) -> io::Result<()> {
    if hex.len() != 64 {
        return Err(io::Error::other("bad SHA-256 length"));
    }
    for pair in hex.as_bytes().chunks_exact(2) {
        let text = std::str::from_utf8(pair).map_err(io::Error::other)?;
        writer.write_all(&[u8::from_str_radix(text, 16).map_err(io::Error::other)?])?;
    }
    Ok(())
}

fn read_hash(reader: &mut impl Read) -> io::Result<String> {
    let mut bytes = [0u8; 32];
    reader.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn write_string(writer: &mut impl Write, text: &str) -> io::Result<()> {
    let bytes = text.as_bytes();
    write_u32(writer, u32::try_from(bytes.len()).map_err(|_| io::Error::other("string too long"))?)?;
    writer.write_all(bytes)
}

fn read_string(reader: &mut impl Read) -> io::Result<String> {
    let length = read_u32(reader)? as usize;
    if length > 4096 {
        return Err(io::Error::other("string exceeds 4096 byte bound"));
    }
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(io::Error::other)
}

fn write_u16(writer: &mut impl Write, value: u16) -> io::Result<()> { writer.write_all(&value.to_le_bytes()) }
fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> { writer.write_all(&value.to_le_bytes()) }
fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> { writer.write_all(&value.to_le_bytes()) }
fn write_i32(writer: &mut impl Write, value: i32) -> io::Result<()> { writer.write_all(&value.to_le_bytes()) }
fn write_option_u8(writer: &mut impl Write, value: Option<u8>) -> io::Result<()> {
    match value { Some(value) => { writer.write_all(&[1, value]) }, None => writer.write_all(&[0]) }
}
fn write_option_u64(writer: &mut impl Write, value: Option<u64>) -> io::Result<()> {
    match value {
        Some(value) => { writer.write_all(&[1])?; write_u64(writer, value) }
        None => writer.write_all(&[0]),
    }
}
fn read_u8(reader: &mut impl Read) -> io::Result<u8> { let mut b=[0]; reader.read_exact(&mut b)?; Ok(b[0]) }
fn read_u16(reader: &mut impl Read) -> io::Result<u16> { let mut b=[0;2]; reader.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
fn read_u32(reader: &mut impl Read) -> io::Result<u32> { let mut b=[0;4]; reader.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn read_u64(reader: &mut impl Read) -> io::Result<u64> { let mut b=[0;8]; reader.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn read_i32(reader: &mut impl Read) -> io::Result<i32> { let mut b=[0;4]; reader.read_exact(&mut b)?; Ok(i32::from_le_bytes(b)) }
fn read_option_u8(reader: &mut impl Read) -> io::Result<Option<u8>> {
    match read_u8(reader)? {
        0 => Ok(None),
        1 => Ok(Some(read_u8(reader)?)),
        _ => Err(io::Error::other("invalid Option<u8> tag")),
    }
}
fn read_option_u64(reader: &mut impl Read) -> io::Result<Option<u64>> {
    match read_u8(reader)? {
        0 => Ok(None),
        1 => Ok(Some(read_u64(reader)?)),
        _ => Err(io::Error::other("invalid Option<u64> tag")),
    }
}

fn io_error(error: impl std::fmt::Display) -> String { error.to_string() }

#[cfg(test)]
mod evaluator_selection_tests {
    use super::*;
    use crate::eval::nnue::FEATURE_COUNT;

    fn test_config(output: PathBuf, eval_file: Option<PathBuf>) -> DatagenConfig {
        DatagenConfig {
            output,
            start_game_id: 3,
            games: 2,
            base_seed: 7,
            nodes: 11,
            hash_mb: 16,
            workers: 1,
            chunk_games: 32,
            eval_file,
        }
    }

    fn test_network(output_weight: f32) -> QuantizedNnueNetwork<128> {
        let hidden_bias = vec![1.0; 128];
        let mut output_weights = vec![0.0; 256];
        output_weights[..128].fill(output_weight);
        QuantizedNnueNetwork::from_float_parts(
            &vec![0.0; FEATURE_COUNT * 128],
            &hidden_bias,
            &output_weights,
            0.0,
            400.0,
            Default::default(),
        ).unwrap()
    }

    fn scratch_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("kira1a-datagen-{label}-{}-{nonce}", process::id()))
    }

    #[test]
    fn default_hce_config_hash_and_manifest_remain_m3_compatible() {
        let config = test_config(PathBuf::from("unused"), None);
        let pin = search_evaluator_pin(&config).unwrap();
        assert_eq!(pin, SearchEvaluatorPin { kind: SearchEvaluatorKind::Hce, nnue_sha256: None });
        let end = config.start_game_id + config.games;
        let legacy_text = format!(
            "record_version={FORMAT_VERSION}\nevaluator=Kira-HCE\nsearch=Searcher-fixed-nodes\nsearch_limit_policy=nodes Some node_limit; depth None; deadline None; Searcher::clear per game\nprng_version={PRNG_VERSION}\nbase_seed={:016x}\nstart={}\nend={}\nnodes={}\nhash_mb={}\nworkers={}\nchunk_games={}\nopening_protocol={OPENING_PROTOCOL}\nwin_streak_policy={WIN_STREAK_POLICY}\ndedup_policy={DEDUP_POLICY}\n",
            config.base_seed, config.start_game_id, end, config.nodes, config.hash_mb,
            config.workers, config.chunk_games,
        );
        assert_eq!(configuration_sha256(&config, end, &pin), sha256_hex(legacy_text.as_bytes()));
        let settings = GenerationSettings {
            start_game_id: 0,
            end_game_id_exclusive: 1,
            base_seed: 0,
            nodes: 1,
            hash_mb: 1,
            workers: 1,
            chunk_games: 1,
            engine_sha256: "e".repeat(64),
            source_sha256: "s".repeat(64),
            config_sha256: "c".repeat(64),
            search_evaluator: pin,
        };
        let manifest = manifest_text(&settings);
        assert!(manifest.contains("evaluator=Kira-HCE\nsearch=Searcher-fixed-nodes\n"));
        assert!(!manifest.contains("search_evaluator="));
        assert!(!manifest.contains("nnue_sha256="));
    }

    #[test]
    fn nnue_file_selects_search_labels_and_is_hash_pinned() {
        let dir = scratch_dir("nnue-pin");
        fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("labels.nnue");
        fs::write(&model_path, test_network(0.01).to_bytes().unwrap()).unwrap();
        let config = test_config(dir.clone(), Some(model_path.clone()));
        let pin = search_evaluator_pin(&config).unwrap();
        assert_eq!(pin.kind, SearchEvaluatorKind::QuantizedNnue128);
        assert_eq!(pin.nnue_sha256.as_deref(), Some(sha256_file(&model_path).unwrap().as_str()));

        let loaded = load_search_evaluator(&config, &pin).unwrap();
        assert!(matches!(loaded, SearchEvaluator::Nnue(_)));
        let board = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let search_score = loaded.evaluate(&board);
        assert_ne!(search_score, Hce.evaluate(&board));

        // The opening gate stays HCE even though the selected search network
        // gives a different score for that same position.
        let opening_hce = opening_hce_score(&board);
        assert!(!opening_score_accepted(opening_hce));
        assert_ne!(search_score, opening_hce);

        let old_config_sha = configuration_sha256(&config, config.start_game_id + config.games, &pin);
        let replacement = test_network(0.02).to_bytes().unwrap();
        fs::write(&model_path, replacement).unwrap();
        let changed = search_evaluator_pin(&config).unwrap();
        assert_ne!(pin.nnue_sha256, changed.nnue_sha256);
        assert_ne!(old_config_sha, configuration_sha256(&config, config.start_game_id + config.games, &changed));
        let error = verify_search_evaluator_unchanged(&config, &pin).unwrap_err();
        assert!(error.contains("NNUE evaluator SHA-256 changed"));
        assert!(load_search_evaluator(&config, &pin).unwrap_err().contains("NNUE evaluator SHA-256 changed"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn artifact_replay_selects_manifest_evaluator_metadata() {
        let dir = scratch_dir("manifest-read");
        fs::create_dir_all(dir.join("chunks")).unwrap();
        let config_sha = "a".repeat(64);
        let nnue_sha = "b".repeat(64);
        let manifest = format!(
            "evaluator=Kira-HCE\nsearch_evaluator=QuantizedNnueNetwork<128>\nnnue_sha256={nnue_sha}\nconfig_sha256={config_sha}\n"
        );
        fs::write(dir.join("manifest.txt"), manifest).unwrap();
        let artifact = dir.join("chunks").join("chunk.bin");
        fs::write(&artifact, []).unwrap();
        let pin = artifact_search_evaluator(&artifact, &config_sha).unwrap();
        assert_eq!(pin.kind, SearchEvaluatorKind::QuantizedNnue128);
        assert_eq!(pin.nnue_sha256.as_deref(), Some(nnue_sha.as_str()));
        assert!(artifact_search_evaluator(&artifact, &"c".repeat(64)).is_err());
        fs::remove_file(dir.join("manifest.txt")).unwrap();
        let detached_error = artifact_search_evaluator(&artifact, &config_sha).unwrap_err();
        assert!(detached_error.contains("adjacent manifest.txt is required"));
        fs::remove_dir_all(dir).unwrap();
    }
}
