use kira1a::{
    Board,
    datagen::{self, BestMoveClass, DecodedCandidate},
    eval::nnue::feature_index,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

pub mod record;
pub use record::TrainingRecord;

pub const DEFAULT_SPLIT_SEED: u64 = 0x4b49_5241_4d34_5631;
pub const VALIDATION_PERCENT: usize = 2;
pub const KIRA_SCORE_LAMBDA: f32 = 0.7;
pub const BULLET_CONSTANT_WDL: f32 = 1.0 - KIRA_SCORE_LAMBDA;
pub const EVAL_SCALE: f32 = 400.0;
pub const TRAINING_RECORD_BYTES: usize = record::TRAINING_RECORD_BYTES;
pub const TRAIN_INTERLEAVE_RECORDS: u64 = 8;
pub const PINNED_BULLET_COMMIT: &str = "579b78b37a694cacd6e9ede4706386c15c34837c";
pub const PINNED_GEN2_NNUE_SHA256: &str = "842e3870cdb02aa762b7d1f4ab0ec441c9df9b6de911395ba759ee1840cefca9";
pub const NNUE_FEATURE_KEY_SCHEMA: &str = "two-perspective-active-feature-set-v1;768-each;stm-first;feature_index-v1;ignore-castling-ep-clocks";
const TRAIN_ORDER_DOMAIN: u64 = 0x5452_4149_4e5f_5631;
const MULTI_SPLIT_DOMAIN: u64 = 0x4b31_4144_4154_4132;
const MULTI_TRAIN_ORDER_DOMAIN: u64 = 0x4d55_4c54_495f_5631;
const NNUE_FEATURES_PER_PERSPECTIVE: usize = 768;
const MAX_BOARD_PIECES: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitSummary {
    pub output: PathBuf,
    pub source_sha256: String,
    pub train_games: u64,
    pub validation_games: u64,
    pub train_records: u64,
    pub validation_records: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiSourceSplitSummary {
    pub output: PathBuf,
    pub gen2_sha256: String,
    pub gen1_sha256: String,
    pub raw_records: u64,
    pub kept_records: u64,
    pub dropped_duplicate_records: u64,
    pub gen2_raw_records: u64,
    pub gen2_kept_records: u64,
    pub gen2_dropped_duplicate_records: u64,
    pub gen2_cross_source_duplicate_records: u64,
    pub gen1_raw_records: u64,
    pub gen1_kept_records: u64,
    pub gen1_dropped_duplicate_records: u64,
    pub within_source_duplicate_records: u64,
    pub cross_source_duplicate_records: u64,
    pub gen2_zero_row_games: u64,
    pub gen1_zero_row_games: u64,
    pub zero_row_games: u64,
    pub train_games: u64,
    pub validation_games: u64,
    pub train_records: u64,
    pub validation_records: u64,
}

/// Exact sparse representation of the two 768-wide active NNUE feature rows.
/// Each perspective contains at most 32 sorted feature indices, packed at 10
/// bits each. The first row is side-to-move, followed by the other side.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NnueFeatureKey {
    piece_count: u8,
    packed_indices: [u8; 80],
}

impl NnueFeatureKey {
    /// Stable bytes for provenance and deterministic parity sample hashing.
    pub fn to_bytes(&self) -> [u8; 81] {
        let mut bytes = [0u8; 81];
        bytes[0] = self.piece_count;
        bytes[1..].copy_from_slice(&self.packed_indices);
        bytes
    }
}

/// Compute the exact two-perspective NNUE input identity for a source row.
/// Castling, en-passant, and move counters do not participate in the key.
pub fn nnue_feature_key(candidate: &DecodedCandidate) -> Result<NnueFeatureKey, String> {
    nnue_feature_key_from_fen(&candidate.fen, candidate.white_to_move)
}

/// Compute a canonical active-feature key from a replay-audited FEN and STM.
pub fn nnue_feature_key_from_fen(fen: &str, white_to_move: bool) -> Result<NnueFeatureKey, String> {
    let board = Board::from_fen(fen).map_err(|error| format!("candidate FEN failed to parse for NNUE key: {error}"))?;
    if board.white_to_move != white_to_move {
        return Err("candidate FEN active color disagrees with format-v2 STM flag".into());
    }
    let mut fields = fen.split_whitespace();
    let placement = fields.next().ok_or_else(|| "candidate FEN has no placement field".to_string())?;
    let active_color = fields.next().ok_or_else(|| "candidate FEN has no active-color field".to_string())?;
    if active_color != if white_to_move { "w" } else { "b" } {
        return Err("candidate FEN active color disagrees with format-v2 STM flag".into());
    }

    let mut stm_features = [u16::MAX; MAX_BOARD_PIECES];
    let mut other_features = [u16::MAX; MAX_BOARD_PIECES];
    let mut piece_count = 0usize;
    let ranks: Vec<_> = placement.split('/').collect();
    if ranks.len() != 8 {
        return Err("candidate FEN placement does not have eight ranks".into());
    }
    for (fen_rank, row) in ranks.iter().enumerate() {
        let mut file = 0usize;
        for ch in row.chars() {
            if let Some(empty) = ch.to_digit(10) {
                if !(1..=8).contains(&empty) {
                    return Err("candidate FEN contains an invalid empty-square run".into());
                }
                file = file.checked_add(empty as usize).ok_or_else(|| "candidate FEN rank overflow".to_string())?;
                if file > 8 {
                    return Err("candidate FEN rank exceeds eight files".into());
                }
                continue;
            }
            if file >= 8 || piece_count >= MAX_BOARD_PIECES {
                return Err("candidate FEN has too many pieces or files".into());
            }
            let piece = match ch {
                'P' => 1,
                'N' => 2,
                'B' => 3,
                'R' => 4,
                'Q' => 5,
                'K' => 6,
                'p' => -1,
                'n' => -2,
                'b' => -3,
                'r' => -4,
                'q' => -5,
                'k' => -6,
                _ => return Err(format!("candidate FEN contains invalid piece {ch:?}")),
            };
            let square = ((7 - fen_rank) * 8 + file) as u8;
            let stm_feature = feature_index(piece, square, white_to_move)
                .ok_or_else(|| "candidate FEN produced an invalid side-to-move feature".to_string())?;
            let other_feature = feature_index(piece, square, !white_to_move)
                .ok_or_else(|| "candidate FEN produced an invalid other-side feature".to_string())?;
            if stm_feature >= NNUE_FEATURES_PER_PERSPECTIVE || other_feature >= NNUE_FEATURES_PER_PERSPECTIVE {
                return Err("NNUE feature index exceeds the pinned 768-feature map".into());
            }
            stm_features[piece_count] = stm_feature as u16;
            other_features[piece_count] = other_feature as u16;
            piece_count += 1;
            file += 1;
        }
        if file != 8 {
            return Err("candidate FEN rank does not describe eight files".into());
        }
    }
    stm_features[..piece_count].sort_unstable();
    other_features[..piece_count].sort_unstable();

    let mut packed_indices = [0u8; 80];
    for (index, feature) in stm_features[..piece_count].iter().chain(other_features[..piece_count].iter()).enumerate() {
        pack_ten_bit_value(&mut packed_indices, index * 10, *feature);
    }
    Ok(NnueFeatureKey { piece_count: piece_count as u8, packed_indices })
}

fn pack_ten_bit_value(output: &mut [u8; 80], bit_offset: usize, value: u16) {
    for bit in 0..10 {
        if value & (1 << bit) != 0 {
            let target = bit_offset + bit;
            output[target / 8] |= 1 << (target % 8);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceProvenance {
    manifest_sha256: String,
    source_tree_sha256: String,
    source_revision: String,
}

/// Split the sorted unique IDs by deterministic rank, with exactly floor(2%)
/// of all games held out. The SplitMix64 finalizer and collision tie-break are
/// pinned by tests and documented in README.md.
pub fn validation_game_ids(game_ids: &[u64], seed: u64) -> Result<BTreeSet<u64>, String> {
    if game_ids.is_empty() {
        return Err("cannot split an empty game-ID list".into());
    }
    let mut sorted_ids = game_ids.to_vec();
    sorted_ids.sort_unstable();
    if sorted_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("duplicate game ID in split input".into());
    }
    let validation_count = sorted_ids.len() * VALIDATION_PERCENT / 100;
    if validation_count == 0 {
        return Err(format!(
            "{} games yield no validation games at {VALIDATION_PERCENT}% (minimum 50)",
            sorted_ids.len()
        ));
    }
    let mut ranked: Vec<(u64, u64)> = sorted_ids
        .iter()
        .copied()
        .map(|id| (splitmix_finalizer(seed ^ id), id))
        .collect();
    ranked.sort_unstable();
    Ok(ranked
        .into_iter()
        .take(validation_count)
        .map(|(_, id)| id)
        .collect())
}

fn splitmix_finalizer(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Convert a Bullet Chess768 row index to the corresponding Kira feature row.
/// Bullet kind indices are P,N,B,R,Q,K; Kira's are P,N,R,B,Q,K.
pub fn bullet_feature_to_kira(feature: usize) -> Option<usize> {
    if feature >= 768 {
        return None;
    }
    let relative_color_base = feature / 384 * 384;
    let within_color = feature % 384;
    let bullet_kind = within_color / 64;
    let square = within_color % 64;
    let kira_kind = match bullet_kind {
        0 => 0, // pawn
        1 => 1, // knight
        2 => 3, // bishop: Kira kind index 3
        3 => 2, // rook: Kira kind index 2
        4 => 4, // queen
        5 => 5, // king
        _ => unreachable!(),
    };
    Some(relative_color_base + kira_kind * 64 + square)
}

pub fn bullet_constant_wdl_for_kira_lambda(score_lambda: f32) -> Result<f32, String> {
    if !score_lambda.is_finite() || !(0.0..=1.0).contains(&score_lambda) {
        return Err("Kira score lambda must be finite and in [0, 1]".into());
    }
    Ok(1.0 - score_lambda)
}

/// Build Kira's bitboards from a replay-audited Kira FEN. This parses only
/// placement and active color; Kira's format-v2 decoder validates the full FEN,
/// legal history, STM, and Zobrist correspondence before invoking the visitor.
fn fen_bitboards(fen: &str, expected_white_to_move: bool) -> Result<[u64; 8], String> {
    let fields: Vec<&str> = fen.split_whitespace().collect();
    if fields.len() < 4 {
        return Err("candidate FEN has fewer than four fields".into());
    }
    let white_to_move = match fields[1] {
        "w" => true,
        "b" => false,
        _ => return Err("candidate FEN has invalid active color".into()),
    };
    if white_to_move != expected_white_to_move {
        return Err("candidate FEN active color disagrees with format-v2 STM flag".into());
    }
    let ranks: Vec<&str> = fields[0].split('/').collect();
    if ranks.len() != 8 {
        return Err("candidate FEN placement does not have eight ranks".into());
    }
    let mut bitboards = [0u64; 8];
    let mut occupied = 0u64;
    let mut king_counts = [0u8; 2];
    let mut piece_count = 0u8;
    for (fen_rank, row) in ranks.iter().enumerate() {
        let mut file = 0usize;
        for ch in row.chars() {
            if let Some(empty) = ch.to_digit(10) {
                if !(1..=8).contains(&empty) {
                    return Err("candidate FEN contains invalid empty-square run".into());
                }
                file += empty as usize;
                if file > 8 {
                    return Err("candidate FEN rank exceeds eight files".into());
                }
                continue;
            }
            let (color, piece_index, is_king) = match ch {
                'P' => (0, 2, false),
                'N' => (0, 3, false),
                'B' => (0, 4, false),
                'R' => (0, 5, false),
                'Q' => (0, 6, false),
                'K' => (0, 7, true),
                'p' => (1, 2, false),
                'n' => (1, 3, false),
                'b' => (1, 4, false),
                'r' => (1, 5, false),
                'q' => (1, 6, false),
                'k' => (1, 7, true),
                _ => return Err(format!("candidate FEN contains invalid piece {ch:?}")),
            };
            if file >= 8 {
                return Err("candidate FEN rank exceeds eight files".into());
            }
            let square = (7 - fen_rank) * 8 + file;
            let bit = 1u64 << square;
            if occupied & bit != 0 {
                return Err("candidate FEN places multiple pieces on a square".into());
            }
            occupied |= bit;
            bitboards[color] |= bit;
            bitboards[piece_index] |= bit;
            if is_king {
                king_counts[color] += 1;
            }
            piece_count += 1;
            file += 1;
        }
        if file != 8 {
            return Err("candidate FEN rank does not describe eight files".into());
        }
    }
    if king_counts != [1, 1] {
        return Err("candidate FEN must contain exactly one king per color".into());
    }
    if piece_count > 32 {
        return Err("candidate FEN has more than 32 pieces".into());
    }
    Ok(bitboards)
}

fn result_stm_from_white(result_white: u8, white_to_move: bool) -> Result<u8, String> {
    if result_white > 2 {
        return Err(format!("invalid White-relative result code {result_white}"));
    }
    Ok(if white_to_move {
        result_white
    } else {
        2 - result_white
    })
}

pub fn convert_candidate(
    game_result_white: u8,
    candidate: &DecodedCandidate,
) -> Result<TrainingRecord, String> {
    if candidate.best_move_class != BestMoveClass::Quiet {
        return Err("format-v2 training candidate has a non-quiet best move".into());
    }
    if !(-2_000..=2_000).contains(&candidate.score_stm) {
        return Err(format!(
            "format-v2 training score {} is outside the M3 ±2000 cp filter",
            candidate.score_stm
        ));
    }
    let expected_result_stm = result_stm_from_white(game_result_white, candidate.white_to_move)?;
    if candidate.result_stm != expected_result_stm {
        return Err(format!(
            "game result implies STM result {expected_result_stm}, candidate stores {}",
            candidate.result_stm
        ));
    }
    let bitboards = fen_bitboards(&candidate.fen, candidate.white_to_move)?;
    let score_stm = i16::try_from(candidate.score_stm)
        .map_err(|_| "side-to-move score does not fit the record's i16 field".to_string())?;
    let board = TrainingRecord::from_bitboards(
        bitboards,
        !candidate.white_to_move,
        score_stm,
        candidate.result_stm,
    )?;
    if board.score != candidate.score_stm as i16 || board.result != candidate.result_stm {
        return Err("Kira record failed to preserve STM-relative score/result".into());
    }
    Ok(board)
}

/// Stable, documented 32-byte little-endian encoding of Kira's training row.
pub fn encode_training_record(board: &TrainingRecord) -> Result<[u8; TRAINING_RECORD_BYTES], String> {
    ensure_training_record_layout()?;
    Ok(record::encode_training_record(board))
}

pub fn decode_training_record(bytes: &[u8]) -> Result<TrainingRecord, String> {
    ensure_training_record_layout()?;
    record::decode_training_record(bytes)
}

/// Reject hosts where Bullet's fixed-size reader would interpret the local
/// little-endian training rows differently.
pub fn ensure_training_record_layout() -> Result<(), String> {
    if !cfg!(target_endian = "little") {
        return Err("Kira training record adapter requires a little-endian host".into());
    }
    if std::mem::size_of::<TrainingRecord>() != TRAINING_RECORD_BYTES {
        return Err(format!(
            "Kira training record changed size: expected {TRAINING_RECORD_BYTES}, found {}",
            std::mem::size_of::<TrainingRecord>()
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GameRows {
    game_id: u64,
    first_record: u64,
    record_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputSegment {
    game_id: u64,
    segment_index: u64,
    source_first_record: u64,
    first_record: u64,
    record_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum MultiSource {
    Gen2,
    Gen1,
}

impl MultiSource {
    fn label(self) -> &'static str {
        match self {
            Self::Gen2 => "gen2",
            Self::Gen1 => "gen1",
        }
    }

    fn split_tag(self) -> u64 {
        match self {
            Self::Gen2 => 0x4745_4e32_0000_0002,
            Self::Gen1 => 0x4745_4e31_0000_0001,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeatureDuplicateKind {
    Keep,
    WithinSource,
    CrossSource,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FeatureDedupCounts {
    raw_records: [u64; 2],
    kept_records: [u64; 2],
    within_source_dropped: [u64; 2],
    cross_source_dropped: u64,
}

impl FeatureDedupCounts {
    fn source_index(source: MultiSource) -> usize {
        usize::from(source == MultiSource::Gen1)
    }

    fn observe(&mut self, source: MultiSource, kind: FeatureDuplicateKind) -> Result<(), String> {
        let index = Self::source_index(source);
        self.raw_records[index] = self.raw_records[index]
            .checked_add(1)
            .ok_or_else(|| "M6 source raw record count overflow".to_string())?;
        match kind {
            FeatureDuplicateKind::Keep => {
                self.kept_records[index] = self.kept_records[index]
                    .checked_add(1)
                    .ok_or_else(|| "M6 source kept record count overflow".to_string())?;
            }
            FeatureDuplicateKind::WithinSource => {
                self.within_source_dropped[index] = self.within_source_dropped[index]
                    .checked_add(1)
                    .ok_or_else(|| "M6 within-source duplicate count overflow".to_string())?;
            }
            FeatureDuplicateKind::CrossSource => {
                self.cross_source_dropped = self.cross_source_dropped
                    .checked_add(1)
                    .ok_or_else(|| "M6 cross-source duplicate count overflow".to_string())?;
            }
        }
        Ok(())
    }

    fn source_dropped(&self, source: MultiSource) -> u64 {
        let index = Self::source_index(source);
        self.within_source_dropped[index]
            + if source == MultiSource::Gen1 { self.cross_source_dropped } else { 0 }
    }

    fn total_raw(&self) -> u64 {
        self.raw_records.iter().sum()
    }

    fn total_kept(&self) -> u64 {
        self.kept_records.iter().sum()
    }

    fn total_dropped(&self) -> u64 {
        self.within_source_dropped.iter().sum::<u64>() + self.cross_source_dropped
    }

    fn total_within_source_dropped(&self) -> u64 {
        self.within_source_dropped.iter().sum()
    }
}

fn claim_feature_key(
    source: MultiSource,
    key: NnueFeatureKey,
    owners: &mut HashMap<NnueFeatureKey, MultiSource>,
) -> FeatureDuplicateKind {
    use std::collections::hash_map::Entry;
    match owners.entry(key) {
        Entry::Vacant(entry) => {
            entry.insert(source);
            FeatureDuplicateKind::Keep
        }
        Entry::Occupied(entry) if source == MultiSource::Gen1 && *entry.get() == MultiSource::Gen2 => {
            FeatureDuplicateKind::CrossSource
        }
        Entry::Occupied(_) => FeatureDuplicateKind::WithinSource,
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CompositeGameId {
    source: MultiSource,
    game_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MultiGameRows {
    identity: CompositeGameId,
    first_record: u64,
    record_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MultiOutputSegment {
    identity: CompositeGameId,
    segment_index: u64,
    source_first_record: u64,
    first_record: u64,
    record_count: u64,
}

#[derive(Clone, Debug)]
struct MultiSourceProvenance {
    source: SourceProvenance,
    engine_sha256: String,
    evaluator: String,
    search_evaluator: Option<String>,
    nnue_sha256: Option<String>,
}

/// Deterministically split composite `(source, game_id)` identities. Game IDs
/// can overlap across generations; the source tag is part of the rank input
/// and of the tie-break key, so those games remain independent units.
fn validation_composite_game_ids(
    game_ids: &[CompositeGameId],
    seed: u64,
) -> Result<BTreeSet<CompositeGameId>, String> {
    if game_ids.is_empty() {
        return Err("cannot split an empty composite game-ID list".into());
    }
    let mut sorted_ids = game_ids.to_vec();
    sorted_ids.sort_unstable();
    if sorted_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("duplicate composite game identity in split input".into());
    }
    let validation_count = sorted_ids.len() * VALIDATION_PERCENT / 100;
    if validation_count == 0 {
        return Err(format!(
            "{} composite games yield no validation games at {VALIDATION_PERCENT}% (minimum 50)",
            sorted_ids.len()
        ));
    }
    let mut ranked: Vec<(u64, CompositeGameId)> = sorted_ids
        .iter()
        .copied()
        .map(|identity| {
            (
                splitmix_finalizer(
                    seed ^ MULTI_SPLIT_DOMAIN ^ identity.source.split_tag() ^ identity.game_id,
                ),
                identity,
            )
        })
        .collect();
    ranked.sort_unstable();
    Ok(ranked
        .into_iter()
        .take(validation_count)
        .map(|(_, identity)| identity)
        .collect())
}

fn multi_training_interleave_segments(
    games: &[MultiGameRows],
    seed: u64,
    chunk_records: u64,
) -> Result<Vec<MultiOutputSegment>, String> {
    if chunk_records == 0 {
        return Err("training interleave chunk size must be positive".into());
    }
    let mut ordered = games.to_vec();
    ordered.sort_unstable_by_key(|game| {
        (
            splitmix_finalizer(
                seed ^ MULTI_TRAIN_ORDER_DOMAIN
                    ^ game.identity.source.split_tag()
                    ^ game.identity.game_id,
            ),
            game.identity,
        )
    });
    let mut seen = BTreeSet::new();
    if ordered.iter().any(|game| !seen.insert(game.identity)) {
        return Err("duplicate composite game identity in training interleave input".into());
    }

    let mut segments = Vec::new();
    let mut output_first = 0u64;
    let mut chunk_index = 0u64;
    loop {
        let mut any = false;
        for game in &ordered {
            let source_offset = chunk_index
                .checked_mul(chunk_records)
                .ok_or_else(|| "training interleave source offset overflow".to_string())?;
            if source_offset < game.record_count {
                any = true;
                let count = (game.record_count - source_offset).min(chunk_records);
                segments.push(MultiOutputSegment {
                    identity: game.identity,
                    segment_index: chunk_index,
                    source_first_record: game
                        .first_record
                        .checked_add(source_offset)
                        .ok_or_else(|| "training interleave source row offset overflow".to_string())?,
                    first_record: output_first,
                    record_count: count,
                });
                output_first = output_first
                    .checked_add(count)
                    .ok_or_else(|| "training interleave output row count overflow".to_string())?;
            }
        }
        if !any {
            break;
        }
        chunk_index = chunk_index
            .checked_add(1)
            .ok_or_else(|| "training interleave chunk index overflow".to_string())?;
    }
    for game in ordered.iter().filter(|game| game.record_count == 0) {
        segments.push(MultiOutputSegment {
            identity: game.identity,
            segment_index: 0,
            source_first_record: game.first_record,
            first_record: output_first,
            record_count: 0,
        });
    }
    Ok(segments)
}

/// Build an auditable deterministic round-robin schedule over fixed-size game
/// row ranges. Within each game, candidate order is retained. Game order is
/// keyed by the pinned seed/domain; successive chunks are interleaved until
/// every training row is scheduled.
fn training_interleave_segments(
    games: &[GameRows],
    seed: u64,
    chunk_records: u64,
) -> Result<Vec<OutputSegment>, String> {
    if chunk_records == 0 {
        return Err("training interleave chunk size must be positive".into());
    }
    let mut ordered = games.to_vec();
    ordered.sort_unstable_by_key(|game| {
        (splitmix_finalizer(seed ^ TRAIN_ORDER_DOMAIN ^ game.game_id), game.game_id)
    });
    if ordered.windows(2).any(|pair| pair[0].game_id == pair[1].game_id) {
        return Err("duplicate game ID in training interleave input".into());
    }

    let mut segments = Vec::new();
    let mut output_first = 0u64;
    let mut chunk_index = 0u64;
    loop {
        let mut any = false;
        for game in &ordered {
            let source_offset = chunk_index
                .checked_mul(chunk_records)
                .ok_or_else(|| "training interleave source offset overflow".to_string())?;
            if source_offset < game.record_count {
                any = true;
                let count = (game.record_count - source_offset).min(chunk_records);
                let source_first_record = game
                    .first_record
                    .checked_add(source_offset)
                    .ok_or_else(|| "training interleave source row offset overflow".to_string())?;
                segments.push(OutputSegment {
                    game_id: game.game_id,
                    segment_index: chunk_index,
                    source_first_record,
                    first_record: output_first,
                    record_count: count,
                });
                output_first = output_first
                    .checked_add(count)
                    .ok_or_else(|| "training interleave output row count overflow".to_string())?;
            }
        }
        if !any {
            break;
        }
        chunk_index = chunk_index
            .checked_add(1)
            .ok_or_else(|| "training interleave chunk index overflow".to_string())?;
    }

    // Preserve empty games in the audit sidecar, even though they produce no
    // record segment. Their first_record is the end offset of the output.
    for game in ordered.iter().filter(|game| game.record_count == 0) {
        segments.push(OutputSegment {
            game_id: game.game_id,
            segment_index: 0,
            source_first_record: game.first_record,
            first_record: output_first,
            record_count: 0,
        });
    }
    Ok(segments)
}

fn write_interleaved_training_file(
    source_path: &Path,
    output_writer: &mut BufWriter<File>,
    ids_writer: &mut BufWriter<File>,
    segments: &[OutputSegment],
) -> Result<u64, String> {
    let mut source = BufReader::new(File::open(source_path).map_err(io_error)?);
    let mut output_records = 0u64;
    for segment in segments {
        if segment.first_record != output_records {
            return Err("training interleave segment output offsets are not contiguous".into());
        }
        if segment.record_count > 0 {
            let byte_offset = segment
                .source_first_record
                .checked_mul(TRAINING_RECORD_BYTES as u64)
                .ok_or_else(|| "training interleave byte offset overflow".to_string())?;
            source.seek(SeekFrom::Start(byte_offset)).map_err(io_error)?;
            let byte_count = usize::try_from(segment.record_count)
                .ok()
                .and_then(|count| count.checked_mul(TRAINING_RECORD_BYTES))
                .ok_or_else(|| "training interleave segment size overflow".to_string())?;
            let mut bytes = vec![0u8; byte_count];
            source.read_exact(&mut bytes).map_err(io_error)?;
            output_writer.write_all(&bytes).map_err(io_error)?;
            output_records = output_records
                .checked_add(segment.record_count)
                .ok_or_else(|| "training interleave output count overflow".to_string())?;
        }
        writeln!(
            ids_writer,
            "{}\t{}\t{}\t{}",
            segment.game_id, segment.segment_index, segment.first_record, segment.record_count
        )
        .map_err(io_error)?;
    }
    Ok(output_records)
}

/// Convert a verified merged format-v2 artifact into two immutable Bullet
/// record files. The source is replay-audited twice; positions are streamed.
pub fn split_artifact(
    input: &Path,
    output: &Path,
    seed: u64,
    bullet_root: &Path,
) -> Result<SplitSummary, String> {
    if output.exists() {
        return Err(format!("refusing to overwrite output {}", output.display()));
    }
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err(format!("output parent does not exist: {}", parent.display()));
    }

    let mut game_ids = Vec::new();
    let first_pass = datagen::inspect_artifact(input, |game| game_ids.push(game.game_id))?;
    if first_pass.kind != "merged-dataset" {
        return Err(format!("expected a merged dataset, found {}", first_pass.kind));
    }
    if game_ids.len() as u64 != first_pass.games {
        return Err("format-v2 decoder game count disagrees with collected IDs".into());
    }
    let validation_ids = validation_game_ids(&game_ids, seed)?;
    let source_provenance = read_source_provenance(input, &first_pass)?;
    let (bullet_root, bullet_commit) = validate_bullet_checkout(bullet_root)?;

    let output_name = output
        .file_name()
        .ok_or_else(|| "output path must have a final directory name".to_string())?
        .to_string_lossy();
    let temp = parent.join(format!(".{output_name}.tmp-{}", std::process::id()));
    fs::create_dir(&temp).map_err(|error| format!("create temporary output: {error}"))?;
    let mut temp_guard = TempDirGuard { path: temp.clone(), keep: false };

    let train_source_tmp = temp.join("train_source.bullet.tmp");
    let train_tmp = temp.join("train.bullet.tmp");
    let validation_tmp = temp.join("validation.bullet.tmp");
    let train_ids_tmp = temp.join("train_game_ids.tsv.tmp");
    let validation_ids_tmp = temp.join("validation_game_ids.tsv.tmp");
    let mut train_source_writer = BufWriter::new(File::options().write(true).create_new(true).open(&train_source_tmp).map_err(io_error)?);
    let mut validation_writer = BufWriter::new(File::options().write(true).create_new(true).open(&validation_tmp).map_err(io_error)?);
    let mut train_game_rows = Vec::new();
    let mut validation_game_rows = Vec::new();

    let mut train_records = 0u64;
    let mut validation_records = 0u64;
    let mut train_games = 0u64;
    let mut validation_games = 0u64;
    let mut game_cursor = 0usize;
    let mut callback_error: Option<String> = None;
    let second_pass = datagen::inspect_artifact(input, |game| {
        if callback_error.is_some() {
            return;
        }
        let result = (|| -> Result<(), String> {
            if game_ids.get(game_cursor) != Some(&game.game_id) {
                return Err(format!("game ID sequence changed on second pass at {}", game.game_id));
            }
            game_cursor += 1;
            let is_validation = validation_ids.contains(&game.game_id);
            let (writer, first_record) = if is_validation {
                validation_games += 1;
                (&mut validation_writer, validation_records)
            } else {
                train_games += 1;
                (&mut train_source_writer, train_records)
            };
            let mut row_count = 0u64;
            for candidate in &game.candidates {
                let board = convert_candidate(game.result_white, candidate)
                    .map_err(|e| format!("game {} ply {}: {e}", game.game_id, candidate.ply))?;
                let bytes = encode_training_record(&board)?;
                writer.write_all(&bytes).map_err(io_error)?;
                row_count += 1;
            }
            if is_validation {
                validation_records += row_count;
                validation_game_rows.push(GameRows {
                    game_id: game.game_id,
                    first_record,
                    record_count: row_count,
                });
            } else {
                train_records += row_count;
                train_game_rows.push(GameRows {
                    game_id: game.game_id,
                    first_record,
                    record_count: row_count,
                });
            }
            Ok(())
        })();
        if let Err(error) = result {
            callback_error = Some(error);
        }
    })?;
    if let Some(error) = callback_error {
        return Err(error);
    }
    if game_cursor != game_ids.len() || second_pass.games != first_pass.games || second_pass.sha256 != first_pass.sha256 {
        return Err("source artifact changed between adapter passes".into());
    }
    if train_games + validation_games != first_pass.games
        || validation_games as usize != validation_ids.len()
        || train_records + validation_records != first_pass.unique_positions
    {
        return Err("adapter output totals disagree with format-v2 source".into());
    }

    sync_writer(&mut train_source_writer)?;
    sync_writer(&mut validation_writer)?;
    drop(train_source_writer);
    drop(validation_writer);

    let train_segments = training_interleave_segments(&train_game_rows, seed, TRAIN_INTERLEAVE_RECORDS)?;
    let mut train_writer = BufWriter::new(File::options().write(true).create_new(true).open(&train_tmp).map_err(io_error)?);
    let mut train_ids_writer = BufWriter::new(File::options().write(true).create_new(true).open(&train_ids_tmp).map_err(io_error)?);
    writeln!(train_ids_writer, "game_id\tsegment_index\tfirst_record\trecord_count").map_err(io_error)?;
    let interleaved_records = write_interleaved_training_file(
        &train_source_tmp,
        &mut train_writer,
        &mut train_ids_writer,
        &train_segments,
    )?;
    if interleaved_records != train_records {
        return Err("training interleave record count disagrees with source".into());
    }
    let mut validation_ids_writer = BufWriter::new(File::options().write(true).create_new(true).open(&validation_ids_tmp).map_err(io_error)?);
    writeln!(validation_ids_writer, "game_id\tsegment_index\tfirst_record\trecord_count").map_err(io_error)?;
    for game in &validation_game_rows {
        writeln!(
            validation_ids_writer,
            "{}\t0\t{}\t{}",
            game.game_id, game.first_record, game.record_count
        )
        .map_err(io_error)?;
    }
    sync_writer(&mut train_writer)?;
    sync_writer(&mut train_ids_writer)?;
    sync_writer(&mut validation_ids_writer)?;
    drop(train_writer);
    drop(train_ids_writer);
    drop(validation_ids_writer);
    fs::remove_file(&train_source_tmp).map_err(io_error)?;
    fs::rename(&train_tmp, temp.join("train.bullet")).map_err(io_error)?;
    fs::rename(&validation_tmp, temp.join("validation.bullet")).map_err(io_error)?;
    fs::rename(&train_ids_tmp, temp.join("train_game_ids.tsv")).map_err(io_error)?;
    fs::rename(&validation_ids_tmp, temp.join("validation_game_ids.tsv")).map_err(io_error)?;

    let train_sha256 = sha256_with_shasum(&temp.join("train.bullet"))?;
    let validation_sha256 = sha256_with_shasum(&temp.join("validation.bullet"))?;
    let train_ids_sha256 = sha256_with_shasum(&temp.join("train_game_ids.tsv"))?;
    let validation_ids_sha256 = sha256_with_shasum(&temp.join("validation_game_ids.tsv"))?;
    let manifest_path = temp.join("manifest.txt");
    let mut manifest = BufWriter::new(File::options().write(true).create_new(true).open(&manifest_path).map_err(io_error)?);
    writeln!(manifest, "adapter_format=kira-m4-record-v2").map_err(io_error)?;
    writeln!(manifest, "source_format=K1ADATAS-v2").map_err(io_error)?;
    writeln!(manifest, "source_sha256={}", first_pass.sha256).map_err(io_error)?;
    writeln!(manifest, "source_config_sha256={}", first_pass.config_sha256).map_err(io_error)?;
    writeln!(manifest, "kira_source_tree_sha256={}", source_provenance.source_tree_sha256).map_err(io_error)?;
    writeln!(manifest, "kira_source_revision={}", source_provenance.source_revision).map_err(io_error)?;
    writeln!(manifest, "m3_source_manifest_sha256={}", source_provenance.manifest_sha256).map_err(io_error)?;
    writeln!(manifest, "m3_config_sha256={}", first_pass.config_sha256).map_err(io_error)?;
    writeln!(manifest, "adapter_executable_sha256={}", sha256_with_shasum(&std::env::current_exe().map_err(io_error)?)?).map_err(io_error)?;
    writeln!(manifest, "adapter_version={}", env!("CARGO_PKG_VERSION")).map_err(io_error)?;
    writeln!(manifest, "adapter_lock_sha256={}", sha256_with_shasum(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?).map_err(io_error)?;
    writeln!(manifest, "bullet_checkout={}", bullet_root.display()).map_err(io_error)?;
    writeln!(manifest, "bullet_git_commit={bullet_commit}").map_err(io_error)?;
    writeln!(manifest, "bullet_training_sources=all crates and workspace Cargo manifests clean at pinned commit; examples excluded").map_err(io_error)?;
    writeln!(manifest, "record_codec=kira-packed-training-record-v1").map_err(io_error)?;
    writeln!(manifest, "source_games={}", first_pass.games).map_err(io_error)?;
    writeln!(manifest, "source_positions={}", first_pass.unique_positions).map_err(io_error)?;
    writeln!(manifest, "split_algorithm=sort(SplitMix64Finalizer(seed XOR game_id), game_id); lowest floor(games*2/100) to validation").map_err(io_error)?;
    writeln!(manifest, "split_seed=0x{seed:016x}").map_err(io_error)?;
    writeln!(manifest, "training_row_order=round-robin interleave of train games; game order sort(SplitMix64Finalizer(seed XOR 0x{TRAIN_ORDER_DOMAIN:016x} XOR game_id), game_id); preserve ply order within each game").map_err(io_error)?;
    writeln!(manifest, "training_interleave_records={TRAIN_INTERLEAVE_RECORDS}").map_err(io_error)?;
    writeln!(manifest, "game_ids_sidecar_schema=game_id,segment_index,first_record,record_count; training games may have multiple non-contiguous output segments").map_err(io_error)?;
    writeln!(manifest, "validation_percent={VALIDATION_PERCENT}").map_err(io_error)?;
    writeln!(manifest, "train_games={train_games}").map_err(io_error)?;
    writeln!(manifest, "validation_games={validation_games}").map_err(io_error)?;
    writeln!(manifest, "train_records={train_records}").map_err(io_error)?;
    writeln!(manifest, "validation_records={validation_records}").map_err(io_error)?;
    writeln!(manifest, "record_format=KiraPackedTrainingRecord-v1;size={TRAINING_RECORD_BYTES};little-endian").map_err(io_error)?;
    writeln!(manifest, "train_sha256={train_sha256}").map_err(io_error)?;
    writeln!(manifest, "validation_sha256={validation_sha256}").map_err(io_error)?;
    writeln!(manifest, "train_game_ids_sha256={train_ids_sha256}").map_err(io_error)?;
    writeln!(manifest, "validation_game_ids_sha256={validation_ids_sha256}").map_err(io_error)?;
    writeln!(manifest, "feature_export_permutation=trainer P,N,B,R,Q,K -> engine P,N,R,B,Q,K; swap kind rows 2<->3 in each 384-row relative-color half").map_err(io_error)?;
    writeln!(manifest, "kira_score_lambda={KIRA_SCORE_LAMBDA}").map_err(io_error)?;
    writeln!(manifest, "bullet_constant_wdl={BULLET_CONSTANT_WDL}").map_err(io_error)?;
    writeln!(manifest, "eval_scale={EVAL_SCALE}").map_err(io_error)?;
    manifest.flush().map_err(io_error)?;
    manifest.get_ref().sync_all().map_err(io_error)?;
    drop(manifest);

    if output.exists() {
        return Err(format!("refusing to overwrite output {}", output.display()));
    }
    fs::rename(&temp, output).map_err(io_error)?;
    temp_guard.keep = true;

    Ok(SplitSummary {
        output: output.to_path_buf(),
        source_sha256: first_pass.sha256,
        train_games,
        validation_games,
        train_records,
        validation_records,
    })
}

/// Assemble the M6 corpus from the pinned gen2 and gen1 format-v2 artifacts.
/// First occurrence wins for exact NNUE active-feature keys, with gen2 scanned
/// before gen1. The global holdout is assigned over all composite identities,
/// including games whose rows are all removed by feature-key deduplication.
/// This path is versioned independently; `split_artifact` remains the M4 path.
pub fn split_generation_artifacts(
    gen2_input: &Path,
    gen1_input: &Path,
    output: &Path,
    seed: u64,
    bullet_root: &Path,
) -> Result<MultiSourceSplitSummary, String> {
    if output.exists() {
        return Err(format!("refusing to overwrite output {}", output.display()));
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err(format!("output parent does not exist: {}", parent.display()));
    }

    let mut composite_ids = Vec::new();
    let mut feature_owners = HashMap::<NnueFeatureKey, MultiSource>::new();
    let mut first_counts = FeatureDedupCounts::default();
    let mut callback_error = None;
    let gen2_first = datagen::inspect_artifact(gen2_input, |game| {
        composite_ids.push(CompositeGameId { source: MultiSource::Gen2, game_id: game.game_id });
        if callback_error.is_some() { return; }
        for candidate in &game.candidates {
            let result = (|| -> Result<(), String> {
                let key = nnue_feature_key(candidate)
                    .map_err(|error| format!("gen2 game {} ply {}: {error}", game.game_id, candidate.ply))?;
                let decision = claim_feature_key(MultiSource::Gen2, key, &mut feature_owners);
                first_counts.observe(MultiSource::Gen2, decision)
            })();
            if let Err(error) = result { callback_error = Some(error); return; }
        }
    })?;
    if let Some(error) = callback_error.take() { return Err(error); }
    if gen2_first.kind != "merged-dataset" {
        return Err(format!("expected gen2 merged dataset, found {}", gen2_first.kind));
    }
    if composite_ids.len() as u64 != gen2_first.games
        || first_counts.raw_records[0] != gen2_first.unique_positions
    {
        return Err("gen2 format-v2 summary disagrees with collected games or NNUE feature keys".into());
    }

    let gen2_provenance = read_generation_provenance(gen2_input, &gen2_first, "gen2")?;
    callback_error = None;
    let gen1_first = datagen::inspect_artifact(gen1_input, |game| {
        composite_ids.push(CompositeGameId { source: MultiSource::Gen1, game_id: game.game_id });
        if callback_error.is_some() { return; }
        for candidate in &game.candidates {
            let result = (|| -> Result<(), String> {
                let key = nnue_feature_key(candidate)
                    .map_err(|error| format!("gen1 game {} ply {}: {error}", game.game_id, candidate.ply))?;
                let decision = claim_feature_key(MultiSource::Gen1, key, &mut feature_owners);
                first_counts.observe(MultiSource::Gen1, decision)
            })();
            if let Err(error) = result { callback_error = Some(error); return; }
        }
    })?;
    if let Some(error) = callback_error.take() { return Err(error); }
    if gen1_first.kind != "merged-dataset" {
        return Err(format!("expected gen1 merged dataset, found {}", gen1_first.kind));
    }
    if composite_ids.len() as u64 != gen2_first.games + gen1_first.games {
        return Err("format-v2 decoder game count disagrees with collected composite IDs".into());
    }
    if gen2_first.sha256 == gen1_first.sha256 {
        return Err("gen1 and gen2 inputs resolve to the same artifact hash".into());
    }
    if first_counts.raw_records[1] != gen1_first.unique_positions
        || first_counts.total_raw() != gen2_first.unique_positions + gen1_first.unique_positions
        || first_counts.total_kept() + first_counts.total_dropped() != first_counts.total_raw()
    {
        return Err("multi-source artifact counts disagree with NNUE feature-key deduplication".into());
    }
    drop(feature_owners);
    let gen1_provenance = read_generation_provenance(gen1_input, &gen1_first, "gen1")?;
    let validation_ids = validation_composite_game_ids(&composite_ids, seed)?;
    let (bullet_root, bullet_commit) = validate_bullet_checkout(bullet_root)?;

    let output_name = output
        .file_name()
        .ok_or_else(|| "output path must have a final directory name".to_string())?
        .to_string_lossy();
    let temp = parent.join(format!(".{output_name}.tmp-{}", std::process::id()));
    fs::create_dir(&temp).map_err(|error| format!("create temporary output: {error}"))?;
    let mut temp_guard = TempDirGuard { path: temp.clone(), keep: false };

    let gen2_source_tmp = temp.join("train_gen2_source.bullet.tmp");
    let gen1_source_tmp = temp.join("train_gen1_source.bullet.tmp");
    let train_tmp = temp.join("train.bullet.tmp");
    let validation_tmp = temp.join("validation.bullet.tmp");
    let train_ids_tmp = temp.join("train_game_ids.tsv.tmp");
    let validation_ids_tmp = temp.join("validation_game_ids.tsv.tmp");
    let mut gen2_source_writer = create_new_writer(&gen2_source_tmp)?;
    let mut gen1_source_writer = create_new_writer(&gen1_source_tmp)?;
    let mut validation_writer = create_new_writer(&validation_tmp)?;
    let mut train_game_rows = Vec::new();
    let mut validation_game_rows = Vec::new();
    let mut train_records = [0u64; 2];
    let mut validation_records = 0u64;
    let mut train_games = 0u64;
    let mut validation_games = 0u64;
    let mut second_feature_owners = HashMap::<NnueFeatureKey, MultiSource>::new();
    let mut second_counts = FeatureDedupCounts::default();

    for (source, input, first) in [
        (MultiSource::Gen2, gen2_input, &gen2_first),
        (MultiSource::Gen1, gen1_input, &gen1_first),
    ] {
        let source_index = usize::from(source == MultiSource::Gen1);
        let mut cursor = 0usize;
        let mut callback_error: Option<String> = None;
        let second = datagen::inspect_artifact(input, |game| {
            if callback_error.is_some() {
                return;
            }
            let result = (|| -> Result<(), String> {
                let identity_index = cursor
                    + if source == MultiSource::Gen1 { gen2_first.games as usize } else { 0 };
                if composite_ids.get(identity_index)
                    != Some(&CompositeGameId { source, game_id: game.game_id })
                {
                    return Err(format!("{source:?} game-ID sequence changed on second pass at {}", game.game_id));
                }
                cursor += 1;
                let identity = CompositeGameId { source, game_id: game.game_id };
                let is_validation = validation_ids.contains(&identity);
                let output_first = if is_validation {
                    validation_games += 1;
                    validation_records
                } else {
                    train_games += 1;
                    train_records[source_index]
                };
                let writer: &mut BufWriter<File> = if is_validation {
                    &mut validation_writer
                } else if source == MultiSource::Gen2 {
                    &mut gen2_source_writer
                } else {
                    &mut gen1_source_writer
                };
                let mut row_count = 0u64;
                for candidate in &game.candidates {
                    let key = nnue_feature_key(candidate)
                        .map_err(|error| format!("{source:?} game {} ply {}: {error}", game.game_id, candidate.ply))?;
                    let decision = claim_feature_key(source, key, &mut second_feature_owners);
                    second_counts.observe(source, decision)?;
                    if decision != FeatureDuplicateKind::Keep {
                        continue;
                    }
                    let board = convert_candidate(game.result_white, candidate)
                        .map_err(|error| format!("{source:?} game {} ply {}: {error}", game.game_id, candidate.ply))?;
                    writer.write_all(&encode_training_record(&board)?).map_err(io_error)?;
                    row_count += 1;
                }
                if is_validation {
                    validation_records += row_count;
                    validation_game_rows.push(MultiGameRows {
                        identity,
                        first_record: output_first,
                        record_count: row_count,
                    });
                } else {
                    train_records[source_index] += row_count;
                    train_game_rows.push(MultiGameRows {
                        identity,
                        first_record: output_first,
                        record_count: row_count,
                    });
                }
                Ok(())
            })();
            if let Err(error) = result {
                callback_error = Some(error);
            }
        })?;
        if let Some(error) = callback_error {
            return Err(error);
        }
        if cursor as u64 != first.games || second.games != first.games || second.sha256 != first.sha256 {
            return Err(format!("{source:?} source artifact changed between adapter passes"));
        }
    }
    if second_counts != first_counts {
        return Err("NNUE feature-key dedup counts changed between adapter passes".into());
    }
    let gen2_kept = first_counts.kept_records[0];
    let gen1_kept = first_counts.kept_records[1];
    let kept_records = first_counts.total_kept();
    if train_games + validation_games != composite_ids.len() as u64
        || validation_games as usize != validation_ids.len()
        || train_records[0] + train_records[1] + validation_records != kept_records
    {
        return Err("multi-source adapter output totals disagree with source summaries".into());
    }

    sync_writer(&mut gen2_source_writer)?;
    sync_writer(&mut gen1_source_writer)?;
    sync_writer(&mut validation_writer)?;
    drop(gen2_source_writer);
    drop(gen1_source_writer);
    drop(validation_writer);

    let segments = multi_training_interleave_segments(&train_game_rows, seed, TRAIN_INTERLEAVE_RECORDS)?;
    let mut train_writer = create_new_writer(&train_tmp)?;
    let mut train_ids_writer = create_new_writer(&train_ids_tmp)?;
    writeln!(train_ids_writer, "source\tgame_id\tsegment_index\tfirst_record\trecord_count").map_err(io_error)?;
    let interleaved_records = write_multi_interleaved_training_file(
        &gen2_source_tmp,
        &gen1_source_tmp,
        &mut train_writer,
        &mut train_ids_writer,
        &segments,
    )?;
    if interleaved_records != train_records.iter().sum::<u64>() {
        return Err("multi-source training interleave count disagrees with staged rows".into());
    }
    let mut validation_ids_writer = create_new_writer(&validation_ids_tmp)?;
    writeln!(validation_ids_writer, "source\tgame_id\tsegment_index\tfirst_record\trecord_count").map_err(io_error)?;
    for game in &validation_game_rows {
        writeln!(
            validation_ids_writer,
            "{}\t{}\t0\t{}\t{}",
            game.identity.source.label(), game.identity.game_id, game.first_record, game.record_count
        )
        .map_err(io_error)?;
    }
    sync_writer(&mut train_writer)?;
    sync_writer(&mut train_ids_writer)?;
    sync_writer(&mut validation_ids_writer)?;
    drop(train_writer);
    drop(train_ids_writer);
    drop(validation_ids_writer);
    fs::remove_file(&gen2_source_tmp).map_err(io_error)?;
    fs::remove_file(&gen1_source_tmp).map_err(io_error)?;
    fs::rename(&train_tmp, temp.join("train.bullet")).map_err(io_error)?;
    fs::rename(&validation_tmp, temp.join("validation.bullet")).map_err(io_error)?;
    fs::rename(&train_ids_tmp, temp.join("train_game_ids.tsv")).map_err(io_error)?;
    fs::rename(&validation_ids_tmp, temp.join("validation_game_ids.tsv")).map_err(io_error)?;

    let train_sha256 = sha256_with_shasum(&temp.join("train.bullet"))?;
    let validation_sha256 = sha256_with_shasum(&temp.join("validation.bullet"))?;
    let train_ids_sha256 = sha256_with_shasum(&temp.join("train_game_ids.tsv"))?;
    let validation_ids_sha256 = sha256_with_shasum(&temp.join("validation_game_ids.tsv"))?;
    let sources_path = temp.join("sources.tsv");
    let mut sources_writer = create_new_writer(&sources_path)?;
    writeln!(sources_writer, "source\tartifact_sha256\tconfig_sha256\tsource_tree_sha256\tmanifest_sha256\tengine_sha256\tevaluator\tsearch_evaluator\tnnue_sha256")
        .map_err(io_error)?;
    for (name, artifact, provenance) in [
        ("gen2", &gen2_first, &gen2_provenance),
        ("gen1", &gen1_first, &gen1_provenance),
    ] {
        writeln!(
            sources_writer,
            "{name}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            artifact.sha256,
            artifact.config_sha256,
            provenance.source.source_tree_sha256,
            provenance.source.manifest_sha256,
            provenance.engine_sha256,
            provenance.evaluator,
            provenance.search_evaluator.as_deref().unwrap_or("not-applicable"),
            provenance.nnue_sha256.as_deref().unwrap_or("not-applicable"),
        )
        .map_err(io_error)?;
    }
    sync_writer(&mut sources_writer)?;
    drop(sources_writer);
    let source_pair_sha256 = sha256_with_shasum(&sources_path)?;
    let zero_row_games_by_source = train_game_rows.iter().chain(&validation_game_rows)
        .filter(|game| game.record_count == 0)
        .fold([0u64; 2], |mut counts, game| {
            counts[usize::from(game.identity.source == MultiSource::Gen1)] += 1;
            counts
        });
    let manifest_path = temp.join("manifest.txt");
    let mut manifest = BufWriter::new(File::options().write(true).create_new(true).open(&manifest_path).map_err(io_error)?);
    writeln!(manifest, "adapter_format=kira-m6-multisource-record-v3").map_err(io_error)?;
    writeln!(manifest, "source_format=K1ADATAS-v2").map_err(io_error)?;
    writeln!(manifest, "source_precedence=gen2>gen1;first-occurrence-within-source;duplicate_key=NNUE-active-feature-key-v1").map_err(io_error)?;
    writeln!(manifest, "nnue_feature_key_schema={NNUE_FEATURE_KEY_SCHEMA}").map_err(io_error)?;
    writeln!(manifest, "nnue_feature_key_encoding=sparse-sorted-10-bit-indices;stm-perspective-first;max-32-pieces-per-perspective").map_err(io_error)?;
    writeln!(manifest, "source_gen2_sha256={}", gen2_first.sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen1_sha256={}", gen1_first.sha256).map_err(io_error)?;
    writeln!(manifest, "source_pair_sha256={source_pair_sha256}").map_err(io_error)?;
    writeln!(manifest, "source_gen2_config_sha256={}", gen2_first.config_sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen1_config_sha256={}", gen1_first.config_sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen2_source_tree_sha256={}", gen2_provenance.source.source_tree_sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen1_source_tree_sha256={}", gen1_provenance.source.source_tree_sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen2_manifest_sha256={}", gen2_provenance.source.manifest_sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen1_manifest_sha256={}", gen1_provenance.source.manifest_sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen2_engine_sha256={}", gen2_provenance.engine_sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen1_engine_sha256={}", gen1_provenance.engine_sha256).map_err(io_error)?;
    writeln!(manifest, "source_gen2_evaluator={}", gen2_provenance.evaluator).map_err(io_error)?;
    writeln!(manifest, "source_gen1_evaluator={}", gen1_provenance.evaluator).map_err(io_error)?;
    writeln!(manifest, "source_gen2_search_evaluator={}", gen2_provenance.search_evaluator.as_deref().unwrap_or("not-applicable")).map_err(io_error)?;
    writeln!(manifest, "source_gen2_nnue_sha256={}", gen2_provenance.nnue_sha256.as_deref().unwrap_or("not-applicable")).map_err(io_error)?;
    writeln!(manifest, "source_gen2_revision={}", gen2_provenance.source.source_revision).map_err(io_error)?;
    writeln!(manifest, "source_gen1_revision={}", gen1_provenance.source.source_revision).map_err(io_error)?;
    writeln!(manifest, "adapter_executable_sha256={}", sha256_with_shasum(&std::env::current_exe().map_err(io_error)?)?).map_err(io_error)?;
    writeln!(manifest, "adapter_version={}", env!("CARGO_PKG_VERSION")).map_err(io_error)?;
    writeln!(manifest, "adapter_lock_sha256={}", sha256_with_shasum(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?).map_err(io_error)?;
    writeln!(manifest, "bullet_checkout={}", bullet_root.display()).map_err(io_error)?;
    writeln!(manifest, "bullet_git_commit={bullet_commit}").map_err(io_error)?;
    writeln!(manifest, "record_codec=kira-packed-training-record-v1").map_err(io_error)?;
    writeln!(manifest, "source_gen2_games={}", gen2_first.games).map_err(io_error)?;
    writeln!(manifest, "source_gen1_games={}", gen1_first.games).map_err(io_error)?;
    writeln!(manifest, "source_games={}", composite_ids.len()).map_err(io_error)?;
    writeln!(manifest, "source_gen2_zero_row_games={}", zero_row_games_by_source[0]).map_err(io_error)?;
    writeln!(manifest, "source_gen1_zero_row_games={}", zero_row_games_by_source[1]).map_err(io_error)?;
    writeln!(manifest, "zero_row_games={}", zero_row_games_by_source.iter().sum::<u64>()).map_err(io_error)?;
    writeln!(manifest, "source_gen2_raw_records={}", gen2_first.unique_positions).map_err(io_error)?;
    writeln!(manifest, "source_gen2_kept_records={gen2_kept}").map_err(io_error)?;
    writeln!(manifest, "source_gen2_dropped_duplicate_records={}", first_counts.source_dropped(MultiSource::Gen2)).map_err(io_error)?;
    writeln!(manifest, "source_gen2_within_source_duplicate_records={}", first_counts.within_source_dropped[0]).map_err(io_error)?;
    writeln!(manifest, "source_gen2_cross_source_duplicate_records=0").map_err(io_error)?;
    writeln!(manifest, "source_gen1_raw_records={}", gen1_first.unique_positions).map_err(io_error)?;
    writeln!(manifest, "source_gen1_kept_records={gen1_kept}").map_err(io_error)?;
    writeln!(manifest, "source_gen1_dropped_duplicate_records={}", first_counts.source_dropped(MultiSource::Gen1)).map_err(io_error)?;
    writeln!(manifest, "source_gen1_within_source_duplicate_records={}", first_counts.within_source_dropped[1]).map_err(io_error)?;
    writeln!(manifest, "source_gen1_cross_source_duplicate_records={}", first_counts.cross_source_dropped).map_err(io_error)?;
    writeln!(manifest, "raw_records={}", gen2_first.unique_positions + gen1_first.unique_positions).map_err(io_error)?;
    writeln!(manifest, "kept_records={kept_records}").map_err(io_error)?;
    writeln!(manifest, "dropped_duplicate_records={}", first_counts.total_dropped()).map_err(io_error)?;
    writeln!(manifest, "within_source_duplicate_records={}", first_counts.total_within_source_dropped()).map_err(io_error)?;
    writeln!(manifest, "cross_source_duplicate_records={}", first_counts.cross_source_dropped).map_err(io_error)?;
    writeln!(manifest, "dedup_policy=first-occurrence by game-id then ply; gen2 source fully precedes gen1; exact active NNUE feature keys; dedup before global split").map_err(io_error)?;
    writeln!(manifest, "split_algorithm=sort(SplitMix64Finalizer(seed XOR domain XOR source_tag XOR game_id), all composite source/game identities including zero-row games); lowest floor(games*2/100) to validation").map_err(io_error)?;
    writeln!(manifest, "split_domain=0x{MULTI_SPLIT_DOMAIN:016x}").map_err(io_error)?;
    writeln!(manifest, "split_seed=0x{seed:016x}").map_err(io_error)?;
    writeln!(manifest, "training_row_order=round-robin interleave across composite source/game identities; preserve ply order within each game").map_err(io_error)?;
    writeln!(manifest, "training_interleave_records={TRAIN_INTERLEAVE_RECORDS}").map_err(io_error)?;
    writeln!(manifest, "game_ids_sidecar_schema=source,game_id,segment_index,first_record,record_count").map_err(io_error)?;
    writeln!(manifest, "validation_percent={VALIDATION_PERCENT}").map_err(io_error)?;
    writeln!(manifest, "train_games={train_games}").map_err(io_error)?;
    writeln!(manifest, "validation_games={validation_games}").map_err(io_error)?;
    writeln!(manifest, "train_records={}", train_records.iter().sum::<u64>()).map_err(io_error)?;
    writeln!(manifest, "validation_records={validation_records}").map_err(io_error)?;
    writeln!(manifest, "record_format=KiraPackedTrainingRecord-v1;size={TRAINING_RECORD_BYTES};little-endian").map_err(io_error)?;
    writeln!(manifest, "train_sha256={train_sha256}").map_err(io_error)?;
    writeln!(manifest, "validation_sha256={validation_sha256}").map_err(io_error)?;
    writeln!(manifest, "train_game_ids_sha256={train_ids_sha256}").map_err(io_error)?;
    writeln!(manifest, "validation_game_ids_sha256={validation_ids_sha256}").map_err(io_error)?;
    writeln!(manifest, "kira_score_lambda={KIRA_SCORE_LAMBDA}").map_err(io_error)?;
    writeln!(manifest, "bullet_constant_wdl={BULLET_CONSTANT_WDL}").map_err(io_error)?;
    writeln!(manifest, "eval_scale={EVAL_SCALE}").map_err(io_error)?;
    manifest.flush().map_err(io_error)?;
    manifest.get_ref().sync_all().map_err(io_error)?;
    drop(manifest);

    if output.exists() {
        return Err(format!("refusing to overwrite output {}", output.display()));
    }
    fs::rename(&temp, output).map_err(io_error)?;
    temp_guard.keep = true;
    Ok(MultiSourceSplitSummary {
        output: output.to_path_buf(),
        gen2_sha256: gen2_first.sha256.clone(),
        gen1_sha256: gen1_first.sha256.clone(),
        raw_records: first_counts.total_raw(),
        kept_records,
        dropped_duplicate_records: first_counts.total_dropped(),
        gen2_raw_records: gen2_first.unique_positions,
        gen2_kept_records: gen2_kept,
        gen2_dropped_duplicate_records: first_counts.source_dropped(MultiSource::Gen2),
        gen2_cross_source_duplicate_records: 0,
        gen1_raw_records: gen1_first.unique_positions,
        gen1_kept_records: gen1_kept,
        gen1_dropped_duplicate_records: first_counts.source_dropped(MultiSource::Gen1),
        within_source_duplicate_records: first_counts.total_within_source_dropped(),
        cross_source_duplicate_records: first_counts.cross_source_dropped,
        gen2_zero_row_games: zero_row_games_by_source[0],
        gen1_zero_row_games: zero_row_games_by_source[1],
        zero_row_games: zero_row_games_by_source.iter().sum(),
        train_games,
        validation_games,
        train_records: train_records.iter().sum(),
        validation_records,
    })
}

fn create_new_writer(path: &Path) -> Result<BufWriter<File>, String> {
    File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map(BufWriter::new)
        .map_err(io_error)
}

fn write_multi_interleaved_training_file(
    gen2_source_path: &Path,
    gen1_source_path: &Path,
    output_writer: &mut BufWriter<File>,
    ids_writer: &mut BufWriter<File>,
    segments: &[MultiOutputSegment],
) -> Result<u64, String> {
    let mut gen2_source = BufReader::new(File::open(gen2_source_path).map_err(io_error)?);
    let mut gen1_source = BufReader::new(File::open(gen1_source_path).map_err(io_error)?);
    let mut output_records = 0u64;
    for segment in segments {
        if segment.first_record != output_records {
            return Err("multi-source interleave output offsets are not contiguous".into());
        }
        if segment.record_count > 0 {
            let source = if segment.identity.source == MultiSource::Gen2 {
                &mut gen2_source
            } else {
                &mut gen1_source
            };
            let byte_offset = segment
                .source_first_record
                .checked_mul(TRAINING_RECORD_BYTES as u64)
                .ok_or_else(|| "multi-source interleave byte offset overflow".to_string())?;
            source.seek(SeekFrom::Start(byte_offset)).map_err(io_error)?;
            let byte_count = usize::try_from(segment.record_count)
                .ok()
                .and_then(|count| count.checked_mul(TRAINING_RECORD_BYTES))
                .ok_or_else(|| "multi-source interleave segment size overflow".to_string())?;
            let mut bytes = vec![0u8; byte_count];
            source.read_exact(&mut bytes).map_err(io_error)?;
            output_writer.write_all(&bytes).map_err(io_error)?;
            output_records += segment.record_count;
        }
        writeln!(
            ids_writer,
            "{}\t{}\t{}\t{}\t{}",
            segment.identity.source.label(),
            segment.identity.game_id,
            segment.segment_index,
            segment.first_record,
            segment.record_count
        )
        .map_err(io_error)?;
    }
    Ok(output_records)
}

fn read_generation_provenance(
    input: &Path,
    artifact: &datagen::ArtifactSummary,
    source_name: &str,
) -> Result<MultiSourceProvenance, String> {
    let provenance = read_source_provenance(input, artifact)
        .map_err(|error| format!("{source_name} provenance: {error}"))?;
    let parent = input.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let manifest = read_key_value_manifest(&parent.join("manifest.txt"))?;
    for (key, expected) in [("format_version", "2"), ("record_version", "2")] {
        if manifest.get(key).map(String::as_str) != Some(expected) {
            return Err(format!("{source_name} source manifest is missing or mismatches {key}={expected}"));
        }
    }
    let engine_sha256 = normalized_sha256(
        manifest.get("engine_sha256").ok_or_else(|| format!("{source_name} source manifest is missing engine_sha256"))?,
        "engine_sha256",
    )?;
    let evaluator = manifest
        .get("evaluator")
        .filter(|value| {
            !value.trim().is_empty()
                && !value.chars().any(|ch| matches!(ch, '\t' | '\n' | '\r'))
        })
        .ok_or_else(|| format!("{source_name} source manifest is missing a valid evaluator"))?
        .clone();
    let search_evaluator = manifest.get("search_evaluator").cloned();
    let nnue_sha256 = manifest.get("nnue_sha256").map(|value| normalized_sha256(value, "nnue_sha256")).transpose()?;
    match source_name {
        "gen2" => {
            if evaluator != "Kira-HCE" {
                return Err("gen2 source must retain the pinned Kira-HCE label evaluator".into());
            }
            if search_evaluator.as_deref() != Some("QuantizedNnueNetwork<128>") {
                return Err("gen2 source must be labeled by QuantizedNnueNetwork<128>".into());
            }
            if nnue_sha256.as_deref() != Some(PINNED_GEN2_NNUE_SHA256) {
                return Err(format!("gen2 NNUE SHA-256 must match the promoted M5 network {PINNED_GEN2_NNUE_SHA256}"));
            }
        }
        "gen1" => {
            if evaluator != "Kira-HCE" || search_evaluator.is_some() || nnue_sha256.is_some() {
                return Err("gen1 source must retain the pinned Kira-HCE label evaluator".into());
            }
        }
        _ => return Err(format!("unsupported source label {source_name}")),
    }
    Ok(MultiSourceProvenance { source: provenance, engine_sha256, evaluator, search_evaluator, nnue_sha256 })
}

fn read_key_value_manifest(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("read source manifest {}: {error}", path.display()))?;
    let mut fields = BTreeMap::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("source manifest {} line {} is malformed", path.display(), line_number + 1));
        };
        if key.trim().is_empty() || fields.insert(key.trim().to_string(), value.trim().to_string()).is_some() {
            return Err(format!("source manifest {} has an empty or duplicate key", path.display()));
        }
    }
    Ok(fields)
}

fn io_error(error: std::io::Error) -> String {
    error.to_string()
}

fn sync_writer(writer: &mut BufWriter<File>) -> Result<(), String> {
    writer.flush().map_err(io_error)?;
    writer.get_ref().sync_all().map_err(io_error)
}

fn sha256_with_shasum(path: &Path) -> Result<String, String> {
    let output = Command::new("shasum")
        .arg("-a")
        .arg("256")
        .arg("--")
        .arg(path)
        .output()
        .map_err(|error| format!("run shasum -a 256: {error}"))?;
    if !output.status.success() {
        return Err(format!("shasum failed: {}", String::from_utf8_lossy(&output.stderr).trim()));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    let digest = stdout.split_whitespace().next().unwrap_or_default();
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("shasum returned malformed SHA-256 output: {stdout:?}"));
    }
    Ok(digest.to_ascii_lowercase())
}

fn normalized_sha256(value: &str, field: &str) -> Result<String, String> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("M3 manifest field {field} is not a 64-digit SHA-256"));
    }
    Ok(value.to_ascii_lowercase())
}

fn read_source_provenance(
    input: &Path,
    artifact: &datagen::ArtifactSummary,
) -> Result<SourceProvenance, String> {
    let parent = input
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let manifest_path = parent.join("manifest.txt");
    let text = fs::read_to_string(&manifest_path).map_err(|error| {
        format!(
            "read source M3 manifest {} beside input artifact: {error}",
            manifest_path.display()
        )
    })?;
    let mut fields = BTreeMap::<String, String>::new();
    for (line_number, line) in text.lines().enumerate() {
        let Some((key, value)) = line.split_once('=') else {
            if line.trim().is_empty() {
                continue;
            }
            return Err(format!("malformed M3 manifest line {}", line_number + 1));
        };
        let key = key.trim();
        if key.is_empty() || fields.insert(key.to_string(), value.trim().to_string()).is_some() {
            return Err(format!("empty or duplicate M3 manifest key on line {}", line_number + 1));
        }
    }
    let source_tree_sha256 = normalized_sha256(
        fields
            .get("source_sha256")
            .ok_or_else(|| "M3 manifest is missing source_sha256".to_string())?,
        "source_sha256",
    )?;
    let config_sha256 = normalized_sha256(
        fields
            .get("config_sha256")
            .ok_or_else(|| "M3 manifest is missing config_sha256".to_string())?,
        "config_sha256",
    )?;
    if config_sha256 != artifact.config_sha256.to_ascii_lowercase() {
        return Err("M3 manifest config_sha256 disagrees with K1ADATAS header".into());
    }
    if let Some(dataset_sha256) = fields.get("dataset_sha256") {
        if normalized_sha256(dataset_sha256, "dataset_sha256")? != artifact.sha256 {
            return Err("M3 manifest dataset_sha256 disagrees with input artifact".into());
        }
    }
    let expected_filename = format!("dataset-{}.k1d", artifact.sha256);
    if input.file_name().and_then(|name| name.to_str()) != Some(expected_filename.as_str()) {
        return Err(format!(
            "M3 input filename must be the content-addressed name {expected_filename}"
        ));
    }

    let revision_fields = ["source_revision", "source_commit", "source_tag"];
    let revisions: Vec<&str> = revision_fields
        .iter()
        .filter_map(|key| fields.get(*key).map(String::as_str))
        .collect();
    if revisions.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err("M3 manifest source revision fields disagree".into());
    }
    let source_revision = revisions.first().copied().unwrap_or("not-recorded-in-m3-manifest");
    if source_revision.is_empty() || source_revision.contains('\t') {
        return Err("M3 manifest source revision is empty or contains a tab".into());
    }
    Ok(SourceProvenance {
        manifest_sha256: sha256_with_shasum(&manifest_path)?,
        source_tree_sha256,
        source_revision: source_revision.to_string(),
    })
}

fn validate_bullet_checkout(root: &Path) -> Result<(PathBuf, String), String> {
    if !root.is_dir() {
        return Err(format!("Bullet checkout root is not a directory: {}", root.display()));
    }
    let root = fs::canonicalize(root).map_err(io_error)?;
    if !root.join("Cargo.toml").is_file() || !root.join("crates/bullet_lib/src/value.rs").is_file() {
        return Err(format!("path is not a Bullet source checkout root: {}", root.display()));
    }
    let top_level = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|error| format!("resolve Bullet checkout root: {error}"))?;
    if !top_level.status.success() {
        return Err(format!(
            "git rev-parse --show-toplevel failed: {}",
            String::from_utf8_lossy(&top_level.stderr).trim()
        ));
    }
    let git_root = PathBuf::from(
        String::from_utf8(top_level.stdout)
            .map_err(|error| error.to_string())?
            .trim(),
    );
    if fs::canonicalize(&git_root).map_err(io_error)? != root {
        return Err(format!("Bullet path is not the checkout root: {}", root.display()));
    }
    let commit = git_revision(&root)?;
    if commit != PINNED_BULLET_COMMIT {
        return Err(format!(
            "Bullet checkout commit mismatch: expected {PINNED_BULLET_COMMIT}, found {commit}"
        ));
    }
    let status = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["status", "--porcelain", "--untracked-files=all", "--", "crates", "Cargo.toml", "Cargo.lock"])
        .output()
        .map_err(|error| format!("check Bullet trainer source status: {error}"))?;
    if !status.status.success() {
        return Err(format!(
            "git status failed for Bullet trainer sources: {}",
            String::from_utf8_lossy(&status.stderr).trim()
        ));
    }
    if !status.stdout.is_empty() {
        return Err("Bullet crates or workspace manifests are dirty; refusing to use unpinned training code".into());
    }
    Ok((root, commit))
}

fn git_revision(root: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| format!("read git revision for {}: {error}", root.display()))?;
    if !output.status.success() {
        return Err(format!(
            "git rev-parse failed for {}: {}",
            root.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let revision = String::from_utf8(output.stdout)
        .map_err(|error| error.to_string())?
        .trim()
        .to_string();
    if revision.len() < 7 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("malformed git revision from {}", root.display()));
    }
    Ok(revision)
}

struct TempDirGuard {
    path: PathBuf,
    keep: bool,
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kira1a::eval::{Evaluator, Hce};
    use kira1a::{Board, Move, START_FEN};
    use std::collections::{BTreeMap, BTreeSet, HashSet};
    use std::sync::atomic::{AtomicU64, Ordering};

    const FIXTURE_GAME_COUNT: u64 = 100;
    const FIXTURE_ROWS_PER_GAME: usize = 5;

    #[derive(Clone)]
    struct FixtureEvent {
        fen: String,
        ply: u32,
        white_to_move: bool,
        score_stm: i32,
        best_move_uci: String,
        zobrist: u64,
    }

    #[derive(Clone)]
    struct FixtureGame {
        game_id: u64,
        seed: u64,
        opening_attempts: u32,
        accepted_opening_hce: i32,
        moves: Vec<String>,
        events: Vec<FixtureEvent>,
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "kira-m4-{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("create isolated test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn test_bullet_root() -> PathBuf {
        std::env::var_os("KIRA_M4_BULLET_ROOT")
            .map(PathBuf::from)
            .expect("focused adapter tests require explicit KIRA_M4_BULLET_ROOT")
    }

    fn next_fixture_word(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        splitmix_finalizer(*state)
    }

    fn fen_piece_at(fen: &str, square: u8) -> Option<char> {
        let placement = fen.split_whitespace().next()?;
        let wanted_file = usize::from(square & 7);
        let wanted_rank = usize::from(square >> 3);
        for (fen_rank, row) in placement.split('/').enumerate() {
            let rank = 7usize.checked_sub(fen_rank)?;
            let mut file = 0usize;
            for ch in row.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    file += empty as usize;
                } else {
                    if rank == wanted_rank && file == wanted_file {
                        return Some(ch);
                    }
                    file += 1;
                }
            }
        }
        None
    }

    fn quiet_nonchecking_moves(board: &Board) -> Vec<String> {
        if board.is_in_check(board.white_to_move) {
            return Vec::new();
        }
        let fen = board.to_fen();
        let mut legal = board.clone();
        legal
            .perft_divide(1)
            .into_iter()
            .filter_map(|(uci, _)| {
                let mv = Move::from_uci(&uci)?;
                if mv.promotion != 0
                    || fen_piece_at(&fen, mv.from).is_none_or(|piece| piece.eq_ignore_ascii_case(&'p'))
                    || fen_piece_at(&fen, mv.to).is_some()
                {
                    return None;
                }
                let mut after = board.clone();
                after.play_uci(&uci).ok()?;
                (!after.is_in_check(after.white_to_move)).then_some(uci)
            })
            .collect()
    }

    fn make_fixture_game(game_id: u64, used_zobrist: &mut HashSet<u64>) -> FixtureGame {
        for attempt in 0..10_000u32 {
            let seed = 0x4b49_5241_4d34_0000u64 ^ game_id.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let mut rng = seed ^ u64::from(attempt).wrapping_mul(0xd6e8_feb8_6659_fd93);
            let mut board = Board::startpos();
            let mut moves = Vec::new();
            for _ in 0..8 {
                let mut legal = board.clone();
                let options = legal.perft_divide(1);
                let Some((uci, _)) = options.get((next_fixture_word(&mut rng) as usize) % options.len()) else {
                    panic!("start position unexpectedly has no legal moves");
                };
                board.play_uci(uci).expect("selected legal opening move");
                moves.push(uci.clone());
            }
            let accepted_hce = Hce.evaluate(&board);
            if !(-300..=300).contains(&accepted_hce) || board.is_in_check(board.white_to_move) {
                continue;
            }

            let mut events = Vec::with_capacity(FIXTURE_ROWS_PER_GAME);
            let mut event_moves = Vec::with_capacity(FIXTURE_ROWS_PER_GAME);
            let mut candidate_hashes = Vec::with_capacity(FIXTURE_ROWS_PER_GAME);
            let mut complete = true;
            for event_index in 0..FIXTURE_ROWS_PER_GAME {
                let choices = quiet_nonchecking_moves(&board);
                if choices.is_empty() {
                    complete = false;
                    break;
                }
                let event_fen = board.to_fen();
                let event_zobrist = board.hash();
                let event = FixtureEvent {
                    fen: event_fen,
                    ply: 8 + event_index as u32,
                    white_to_move: board.white_to_move,
                    score_stm: if board.white_to_move { 1_600 } else { -1_600 },
                    best_move_uci: choices[(next_fixture_word(&mut rng) as usize) % choices.len()].clone(),
                    zobrist: event_zobrist,
                };
                board.play_uci(&event.best_move_uci).expect("selected legal quiet move");
                candidate_hashes.push(event_zobrist);
                event_moves.push(event.best_move_uci.clone());
                events.push(event);
            }
            let distinct_candidate_hashes: HashSet<u64> = candidate_hashes.iter().copied().collect();
            if !complete
                || distinct_candidate_hashes.len() != candidate_hashes.len()
                || candidate_hashes.iter().any(|hash| used_zobrist.contains(hash))
            {
                continue;
            }
            for event_move in event_moves {
                moves.push(event_move);
            }
            used_zobrist.extend(candidate_hashes);
            return FixtureGame {
                game_id,
                seed,
                opening_attempts: attempt + 1,
                accepted_opening_hce: accepted_hce,
                moves,
                events,
            };
        }
        panic!("could not construct replay-valid synthetic game {game_id}");
    }

    fn write_test_string(writer: &mut impl Write, value: &str) -> std::io::Result<()> {
        writer.write_all(&(value.len() as u32).to_le_bytes())?;
        writer.write_all(value.as_bytes())
    }

    fn write_fixture_game(writer: &mut impl Write, game: &FixtureGame) -> std::io::Result<()> {
        let mut payload = Vec::new();
        payload.push(1);
        payload.write_all(&game.game_id.to_le_bytes())?;
        payload.write_all(&game.seed.to_le_bytes())?;
        write_test_string(&mut payload, START_FEN)?;
        payload.write_all(&game.opening_attempts.to_le_bytes())?;
        payload.write_all(&game.accepted_opening_hce.to_le_bytes())?;
        payload.extend_from_slice(&[2, 5]); // White win, score-win adjudication.
        payload.write_all(&(game.events.len() as u64).to_le_bytes())?;
        payload.write_all(&(game.moves.len() as u32).to_le_bytes())?;
        for text in &game.moves {
            write_test_string(&mut payload, text)?;
        }
        payload.write_all(&(game.events.len() as u32).to_le_bytes())?;
        for (index, event) in game.events.iter().enumerate() {
            let mut record = Vec::new();
            record.push(1);
            write_test_string(&mut record, &event.fen)?;
            record.write_all(&event.ply.to_le_bytes())?;
            record.push(u8::from(!event.white_to_move));
            record.write_all(&event.score_stm.to_le_bytes())?;
            record.write_all(&(index as u32).to_le_bytes())?;
            record.write_all(&1u64.to_le_bytes())?; // fixed node limit
            record.push(0); // no depth limit
            record.push(0); // no time limit
            record.write_all(&1u64.to_le_bytes())?; // one node searched
            record.push(1); // search depth
            write_test_string(&mut record, &event.best_move_uci)?;
            record.extend_from_slice(&[0, 0]); // quiet, kept
            record.write_all(&event.zobrist.to_le_bytes())?;
            payload.write_all(&(record.len() as u32).to_le_bytes())?;
            payload.extend_from_slice(&record);
        }
        payload.write_all(&(game.events.len() as u32).to_le_bytes())?;
        for event in &game.events {
            let mut record = Vec::new();
            record.push(1);
            write_test_string(&mut record, &event.fen)?;
            record.write_all(&event.ply.to_le_bytes())?;
            record.push(u8::from(!event.white_to_move));
            record.write_all(&event.score_stm.to_le_bytes())?;
            record.push(if event.white_to_move { 2 } else { 0 }); // White-win result from STM view.
            record.push(0); // quiet best move
            record.write_all(&event.zobrist.to_le_bytes())?;
            payload.write_all(&(record.len() as u32).to_le_bytes())?;
            payload.extend_from_slice(&record);
        }
        writer.write_all(&(payload.len() as u64).to_le_bytes())?;
        writer.write_all(&payload)
    }

    fn write_synthetic_v2_merged(root: &Path) -> PathBuf {
        write_synthetic_v2_merged_ids(root, "synthetic", (1..=FIXTURE_GAME_COUNT).collect())
    }

    fn write_synthetic_v2_merged_ids(root: &Path, label: &str, ids: Vec<u64>) -> PathBuf {
        assert!(!ids.is_empty());
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
        let mut games = Vec::with_capacity(ids.len());
        let mut used_zobrist = HashSet::new();
        for game_id in ids {
            games.push(make_fixture_game(game_id, &mut used_zobrist));
        }
        let game_count = games.len() as u64;
        let positions = game_count * FIXTURE_ROWS_PER_GAME as u64;
        let temporary_input = root.join(format!("{label}.tmp.k1d"));
        let mut writer = BufWriter::new(File::create(&temporary_input).expect("create fixture artifact"));
        writer.write_all(b"K1ADATAS").unwrap();
        writer.write_all(&2u16.to_le_bytes()).unwrap();
        writer.write_all(&[0xab; 32]).unwrap(); // deterministic config SHA-256 bytes
        for total in [game_count, positions, 0, positions] {
            writer.write_all(&total.to_le_bytes()).unwrap();
        }
        for game in &games {
            write_fixture_game(&mut writer, game).unwrap();
        }
        writer.flush().unwrap();
        writer.get_ref().sync_all().unwrap();
        drop(writer);
        let config_sha256 = "ab".repeat(32);
        let source_sha256 = if label == "synthetic" { "cd".repeat(32) } else if label == "gen2" { "d2".repeat(32) } else { "d1".repeat(32) };
        let dataset_sha256 = sha256_with_shasum(&temporary_input).unwrap();
        let input = root.join(format!("dataset-{dataset_sha256}.k1d"));
        fs::rename(&temporary_input, &input).unwrap();
        let evaluator_pin = if label == "gen2" {
            format!("search_evaluator=QuantizedNnueNetwork<128>\nnnue_sha256={PINNED_GEN2_NNUE_SHA256}\n")
        } else {
            String::new()
        };
        let manifest = format!(
            "format_version=2\nrecord_version=2\nevaluator=Kira-HCE\nengine_sha256={}\n{evaluator_pin}source_sha256={source_sha256}\nconfig_sha256={config_sha256}\n",
            if label == "gen2" { "22" } else { "11" }.repeat(32),
        );
        fs::write(root.join("manifest.txt"), manifest).unwrap();
        input
    }

    fn read_records(path: &Path) -> Vec<[u8; TRAINING_RECORD_BYTES]> {
        let bytes = fs::read(path).expect("read Bullet output");
        assert_eq!(bytes.len() % TRAINING_RECORD_BYTES, 0);
        bytes
            .chunks_exact(TRAINING_RECORD_BYTES)
            .map(|record| record.try_into().unwrap())
            .collect()
    }

    fn parse_manifest_value(manifest: &str, key: &str) -> String {
        manifest
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("missing manifest key {key}"))
            .to_string()
    }

    fn candidate(fen: &str, white_to_move: bool, score_stm: i32, result_stm: u8) -> DecodedCandidate {
        DecodedCandidate {
            fen: fen.to_string(),
            ply: 8,
            white_to_move,
            score_stm,
            result_stm,
            best_move_class: BestMoveClass::Quiet,
            zobrist: 0,
        }
    }

    fn features_from_board(board: TrainingRecord) -> BTreeSet<(usize, usize)> {
        let mut features = BTreeSet::new();
        board.for_each_piece(|piece, square| {
            let color = usize::from(piece & 8 > 0);
            let kind = usize::from(piece & 7);
            let stm = [0, 384][color] + kind * 64 + usize::from(square);
            let ntm = [384, 0][color] + kind * 64 + usize::from(square ^ 56);
            features.insert((
                bullet_feature_to_kira(stm).unwrap(),
                bullet_feature_to_kira(ntm).unwrap(),
            ));
        });
        features
    }

    fn expected_features_from_fen(fen: &str) -> BTreeSet<(usize, usize)> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        let stm_white = fields[1] == "w";
        let mut expected = BTreeSet::new();
        for (fen_rank, row) in fields[0].split('/').enumerate() {
            let mut file = 0usize;
            for ch in row.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    file += empty as usize;
                    continue;
                }
                let (white, kira_kind) = match ch {
                    'P' => (true, 0),
                    'N' => (true, 1),
                    'R' => (true, 2),
                    'B' => (true, 3),
                    'Q' => (true, 4),
                    'K' => (true, 5),
                    'p' => (false, 0),
                    'n' => (false, 1),
                    'r' => (false, 2),
                    'b' => (false, 3),
                    'q' => (false, 4),
                    'k' => (false, 5),
                    _ => panic!("bad fixture piece"),
                };
                let square = (7 - fen_rank) * 8 + file;
                let stm_relative_color = usize::from(white != stm_white);
                let ntm_relative_color = 1 - stm_relative_color;
                let stm_square = if stm_white { square } else { square ^ 56 };
                let ntm_square = if stm_white { square ^ 56 } else { square };
                expected.insert((
                    stm_relative_color * 384 + kira_kind * 64 + stm_square,
                    ntm_relative_color * 384 + kira_kind * 64 + ntm_square,
                ));
                file += 1;
            }
        }
        expected
    }

    #[test]
    fn split_is_deterministic_exact_and_game_level() {
        let ids: Vec<u64> = (10_000..11_000).collect();
        let first = validation_game_ids(&ids, DEFAULT_SPLIT_SEED).unwrap();
        let second = validation_game_ids(&ids, DEFAULT_SPLIT_SEED).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 20);
        assert!(first.iter().all(|id| ids.contains(id)));
        let train: BTreeSet<_> = ids.iter().copied().filter(|id| !first.contains(id)).collect();
        assert_eq!(train.len() + first.len(), ids.len());
        assert!(train.is_disjoint(&first));
        assert_ne!(first, validation_game_ids(&ids, DEFAULT_SPLIT_SEED + 1).unwrap());
        assert!(validation_game_ids(&[7, 7], DEFAULT_SPLIT_SEED).is_err());
        assert!(validation_game_ids(&(0..49).collect::<Vec<_>>(), DEFAULT_SPLIT_SEED).is_err());
    }

    #[test]
    fn nnue_feature_key_ignores_non_nnue_fen_fields_and_canonicalizes_mirrors() {
        let castling = nnue_feature_key_from_fen(
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            true,
        ).unwrap();
        let no_castling = nnue_feature_key_from_fen(
            "r3k2r/8/8/8/8/8/8/R3K2R w - - 15 40",
            true,
        ).unwrap();
        assert_eq!(castling, no_castling, "castling rights and clocks are outside NNUE identity");

        let en_passant = nnue_feature_key_from_fen(
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            true,
        ).unwrap();
        let no_en_passant = nnue_feature_key_from_fen(
            "4k3/8/8/3pP3/8/8/8/4K3 w - - 99 21",
            true,
        ).unwrap();
        assert_eq!(en_passant, no_en_passant, "en-passant and clocks are outside NNUE identity");

        let position = nnue_feature_key_from_fen(
            "4k3/8/8/8/8/8/P7/4K3 w - - 0 1",
            true,
        ).unwrap();
        let color_rank_mirror = nnue_feature_key_from_fen(
            "4k3/p7/8/8/8/8/8/4K3 b - - 0 1",
            false,
        ).unwrap();
        assert_eq!(position, color_rank_mirror, "feature_index maps these mirrored positions identically");
    }

    #[test]
    fn multi_source_split_uses_composite_ids_gen2_dedup_and_is_reproducible() {
        let root = TestDir::new("multi-source");
        let gen2_dir = root.0.join("gen2");
        let gen1_dir = root.0.join("gen1");
        fs::create_dir(&gen2_dir).unwrap();
        fs::create_dir(&gen1_dir).unwrap();
        let gen2_ids: Vec<u64> = (1..=100).collect();
        let gen1_nonoverlap_ids: Vec<u64> = (101..200).collect();
        let (split_seed, overlapping_validation_game_id) = (0..=100u64).find_map(|seed_offset| {
            let seed = DEFAULT_SPLIT_SEED + seed_offset;
            let overlap = gen2_ids.iter().copied().find(|candidate| {
                let ids = gen2_ids.iter().copied().map(|game_id| CompositeGameId { source: MultiSource::Gen2, game_id })
                    .chain(gen1_nonoverlap_ids.iter().copied().map(|game_id| CompositeGameId { source: MultiSource::Gen1, game_id }))
                    .chain(std::iter::once(CompositeGameId { source: MultiSource::Gen1, game_id: *candidate }))
                    .collect::<Vec<_>>();
                validation_composite_game_ids(&ids, seed)
                    .is_ok_and(|validation| validation.contains(&CompositeGameId { source: MultiSource::Gen1, game_id: *candidate }))
            });
            overlap.map(|game_id| (seed, game_id))
        }).expect("fixture should include an overlapping gen1 identity in validation");
        let gen1_ids: Vec<u64> = std::iter::once(overlapping_validation_game_id).chain(gen1_nonoverlap_ids).collect();
        let gen2 = write_synthetic_v2_merged_ids(&gen2_dir, "gen2", gen2_ids.clone());
        let gen1 = write_synthetic_v2_merged_ids(&gen1_dir, "gen1", gen1_ids.clone());
        let out_a = root.0.join("m6-a");
        let out_b = root.0.join("m6-b");
        let bullet_root = test_bullet_root();

        let gen2_summary = datagen::inspect_artifact(&gen2, |_| {}).unwrap();
        let gen1_summary = datagen::inspect_artifact(&gen1, |_| {}).unwrap();
        assert_eq!(gen2_summary.unique_positions, 500);
        assert_eq!(gen1_summary.unique_positions, 500);
        let summary_a = split_generation_artifacts(
            &gen2, &gen1, &out_a, split_seed, &bullet_root,
        ).unwrap();
        assert_eq!(summary_a.raw_records, 1_000);
        assert_eq!(summary_a.kept_records + summary_a.dropped_duplicate_records, summary_a.raw_records);
        assert_eq!(summary_a.gen2_raw_records, 500);
        assert_eq!(summary_a.gen2_kept_records + summary_a.gen2_dropped_duplicate_records, 500);
        assert_eq!(summary_a.gen2_cross_source_duplicate_records, 0);
        assert_eq!(summary_a.gen1_raw_records, 500);
        assert_eq!(summary_a.gen1_kept_records + summary_a.gen1_dropped_duplicate_records, 500);
        assert_eq!(summary_a.dropped_duplicate_records,
            summary_a.within_source_duplicate_records + summary_a.cross_source_duplicate_records);
        assert!(summary_a.within_source_duplicate_records > 0, "fixture contains at least one within-source feature-key duplicate");
        assert_eq!(summary_a.train_games, 196);
        assert_eq!(summary_a.validation_games, 4);
        assert_eq!(summary_a.train_records + summary_a.validation_records, summary_a.kept_records);

        let mut expected_by_game = BTreeMap::<CompositeGameId, Vec<(NnueFeatureKey, [u8; TRAINING_RECORD_BYTES])>>::new();
        let mut seen_features = BTreeSet::new();
        let mut composite_ids = Vec::new();
        for (source, path) in [(MultiSource::Gen2, &gen2), (MultiSource::Gen1, &gen1)] {
            datagen::inspect_artifact(path, |game| {
                let identity = CompositeGameId { source, game_id: game.game_id };
                composite_ids.push(identity);
                let expected_rows = expected_by_game.entry(identity).or_default();
                for candidate in &game.candidates {
                    let key = nnue_feature_key(candidate).unwrap();
                    if !seen_features.insert(key.clone()) { continue; }
                    let board = convert_candidate(game.result_white, candidate).unwrap();
                    expected_rows.push((key, encode_training_record(&board).unwrap()));
                }
            }).unwrap();
        }
        assert_eq!(composite_ids.len(), 200);
        assert!(composite_ids.contains(&CompositeGameId { source: MultiSource::Gen2, game_id: overlapping_validation_game_id }));
        assert!(composite_ids.contains(&CompositeGameId { source: MultiSource::Gen1, game_id: overlapping_validation_game_id }));
        let gen1_zero_validation_identity = CompositeGameId { source: MultiSource::Gen1, game_id: overlapping_validation_game_id };
        assert!(expected_by_game[&gen1_zero_validation_identity].is_empty(),
            "matching gen1 game is retained as an identity after every row is deduplicated");
        let expected_validation = validation_composite_game_ids(&composite_ids, split_seed).unwrap();
        assert_eq!(expected_validation.len(), 4);
        assert!(expected_validation.contains(&gen1_zero_validation_identity), "all-deduplicated gen1 identity is assigned to validation");
        let expected_gen2_zero_rows = expected_by_game.iter().filter(|(identity, rows)| identity.source == MultiSource::Gen2 && rows.is_empty()).count();
        let expected_gen1_zero_rows = expected_by_game.iter().filter(|(identity, rows)| identity.source == MultiSource::Gen1 && rows.is_empty()).count();
        assert_eq!(summary_a.gen2_zero_row_games, expected_gen2_zero_rows as u64);
        assert_eq!(summary_a.gen1_zero_row_games, expected_gen1_zero_rows as u64);
        assert_eq!(summary_a.zero_row_games, (expected_gen2_zero_rows + expected_gen1_zero_rows) as u64);

        let mut train_composite = BTreeSet::new();
        let mut validation_composite = BTreeSet::new();
        let train_sidecar = fs::read_to_string(out_a.join("train_game_ids.tsv")).unwrap();
        assert_eq!(train_sidecar.lines().next(), Some("source\tgame_id\tsegment_index\tfirst_record\trecord_count"));
        let train_bytes = fs::read(out_a.join("train.bullet")).unwrap();
        let mut expected_train_positions = BTreeSet::new();
        for line in train_sidecar.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 5);
            let source = match fields[0] { "gen2" => MultiSource::Gen2, "gen1" => MultiSource::Gen1, _ => panic!("unexpected source label") };
            let identity = CompositeGameId { source, game_id: fields[1].parse().unwrap() };
            let segment = fields[2].parse::<usize>().unwrap();
            let first_record = fields[3].parse::<usize>().unwrap();
            let record_count = fields[4].parse::<usize>().unwrap();
            assert!(train_composite.insert(identity));
            assert!(!expected_validation.contains(&identity), "validation game leaked into train sidecar");
            let expected = &expected_by_game[&identity];
            if identity == gen1_zero_validation_identity { panic!("zero-row validation identity leaked into train sidecar"); }
            let source_offset = segment * TRAIN_INTERLEAVE_RECORDS as usize;
            let expected_count = expected.len().saturating_sub(source_offset).min(TRAIN_INTERLEAVE_RECORDS as usize);
            assert_eq!(record_count, expected_count);
            for (offset, (feature_key, bytes)) in expected[source_offset..source_offset + record_count].iter().enumerate() {
                let output_offset = (first_record + offset) * TRAINING_RECORD_BYTES;
                assert_eq!(&train_bytes[output_offset..output_offset + TRAINING_RECORD_BYTES], bytes);
                assert!(expected_train_positions.insert(feature_key.clone()), "duplicate NNUE feature key retained in train");
            }
        }
        let validation_sidecar = fs::read_to_string(out_a.join("validation_game_ids.tsv")).unwrap();
        assert_eq!(validation_sidecar.lines().next(), Some("source\tgame_id\tsegment_index\tfirst_record\trecord_count"));
        let validation_bytes = fs::read(out_a.join("validation.bullet")).unwrap();
        let mut expected_validation_positions = BTreeSet::new();
        for line in validation_sidecar.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 5);
            let source = match fields[0] { "gen2" => MultiSource::Gen2, "gen1" => MultiSource::Gen1, _ => panic!("unexpected source label") };
            let identity = CompositeGameId { source, game_id: fields[1].parse().unwrap() };
            let first_record = fields[3].parse::<usize>().unwrap();
            let record_count = fields[4].parse::<usize>().unwrap();
            assert!(validation_composite.insert(identity));
            assert!(expected_validation.contains(&identity), "train game leaked into validation sidecar");
            let expected = &expected_by_game[&identity];
            if identity == gen1_zero_validation_identity {
                assert_eq!(record_count, 0, "all-deduplicated game remains as a zero-row validation sidecar identity");
            }
            assert_eq!(record_count, expected.len());
            for (offset, (feature_key, bytes)) in expected.iter().enumerate() {
                let output_offset = (first_record + offset) * TRAINING_RECORD_BYTES;
                assert_eq!(&validation_bytes[output_offset..output_offset + TRAINING_RECORD_BYTES], bytes);
                assert!(expected_validation_positions.insert(feature_key.clone()), "duplicate NNUE feature key retained in validation");
            }
        }
        assert!(expected_train_positions.is_disjoint(&expected_validation_positions), "NNUE feature key leaked across train/validation");
        assert_eq!(expected_train_positions.len() + expected_validation_positions.len(), summary_a.kept_records as usize);
        assert!(train_composite.contains(&gen1_zero_validation_identity) ^ validation_composite.contains(&gen1_zero_validation_identity),
            "zero-row game identity appears in exactly one split sidecar");
        assert!(validation_composite.contains(&gen1_zero_validation_identity));
        assert!(train_composite.is_disjoint(&validation_composite));
        assert_eq!(train_composite.union(&validation_composite).count(), 200);
        assert_eq!(validation_composite, expected_validation);
        let gen2_overlap_identity = CompositeGameId { source: MultiSource::Gen2, game_id: overlapping_validation_game_id };
        assert!(train_composite.contains(&gen2_overlap_identity) ^ validation_composite.contains(&gen2_overlap_identity));
        assert!(train_composite.contains(&gen1_zero_validation_identity) ^ validation_composite.contains(&gen1_zero_validation_identity));

        let manifest = fs::read_to_string(out_a.join("manifest.txt")).unwrap();
        assert_eq!(parse_manifest_value(&manifest, "adapter_format"), "kira-m6-multisource-record-v3");
        assert_eq!(parse_manifest_value(&manifest, "source_precedence"), "gen2>gen1;first-occurrence-within-source;duplicate_key=NNUE-active-feature-key-v1");
        assert_eq!(parse_manifest_value(&manifest, "nnue_feature_key_schema"), NNUE_FEATURE_KEY_SCHEMA);
        assert_eq!(parse_manifest_value(&manifest, "raw_records"), "1000");
        assert_eq!(parse_manifest_value(&manifest, "kept_records"), summary_a.kept_records.to_string());
        assert_eq!(parse_manifest_value(&manifest, "dropped_duplicate_records"), summary_a.dropped_duplicate_records.to_string());
        assert_eq!(parse_manifest_value(&manifest, "source_gen2_dropped_duplicate_records"), summary_a.gen2_dropped_duplicate_records.to_string());
        assert_eq!(parse_manifest_value(&manifest, "source_gen2_cross_source_duplicate_records"), "0");
        assert_eq!(parse_manifest_value(&manifest, "source_gen1_dropped_duplicate_records"), summary_a.gen1_dropped_duplicate_records.to_string());
        assert_eq!(parse_manifest_value(&manifest, "source_gen2_zero_row_games"), expected_gen2_zero_rows.to_string());
        assert_eq!(parse_manifest_value(&manifest, "source_gen1_zero_row_games"), expected_gen1_zero_rows.to_string());
        assert_eq!(parse_manifest_value(&manifest, "zero_row_games"), (expected_gen2_zero_rows + expected_gen1_zero_rows).to_string());

        let summary_b = split_generation_artifacts(
            &gen2, &gen1, &out_b, split_seed, &bullet_root,
        ).unwrap();
        assert_eq!(summary_a.gen2_sha256, summary_b.gen2_sha256);
        assert_eq!(summary_a.gen1_sha256, summary_b.gen1_sha256);
        assert_eq!(summary_a.raw_records, summary_b.raw_records);
        assert_eq!(summary_a.kept_records, summary_b.kept_records);
        assert_eq!(summary_a.dropped_duplicate_records, summary_b.dropped_duplicate_records);
        assert_eq!(summary_a.train_games, summary_b.train_games);
        assert_eq!(summary_a.validation_games, summary_b.validation_games);
        assert_eq!(summary_a.train_records, summary_b.train_records);
        assert_eq!(summary_a.validation_records, summary_b.validation_records);
        for file in ["train.bullet", "validation.bullet", "train_game_ids.tsv", "validation_game_ids.tsv", "manifest.txt"] {
            assert_eq!(fs::read(out_a.join(file)).unwrap(), fs::read(out_b.join(file)).unwrap(), "nondeterministic {file}");
        }
    }

    #[test]
    fn multi_source_provenance_fails_closed_before_output_creation() {
        let root = TestDir::new("multi-provenance");
        let gen2_dir = root.0.join("gen2");
        let gen1_dir = root.0.join("gen1");
        fs::create_dir(&gen2_dir).unwrap();
        fs::create_dir(&gen1_dir).unwrap();
        let gen2 = write_synthetic_v2_merged_ids(&gen2_dir, "gen2", (1..=100).collect());
        let gen1 = write_synthetic_v2_merged_ids(&gen1_dir, "gen1", (101..=200).collect());
        let gen2_manifest = gen2_dir.join("manifest.txt");
        let valid_gen2_manifest = fs::read_to_string(&gen2_manifest).unwrap();
        fs::write(
            &gen2_manifest,
            valid_gen2_manifest.replace(PINNED_GEN2_NNUE_SHA256, &"00".repeat(32)),
        ).unwrap();
        let pin_output = root.0.join("wrong-network-must-not-exist");
        let pin_error = split_generation_artifacts(
            &gen2, &gen1, &pin_output, DEFAULT_SPLIT_SEED, &test_bullet_root(),
        ).unwrap_err();
        assert!(pin_error.contains("NNUE SHA-256"), "unexpected failure cause: {pin_error}");
        assert!(!pin_output.exists());
        fs::write(&gen2_manifest, valid_gen2_manifest).unwrap();
        fs::remove_file(gen1_dir.join("manifest.txt")).unwrap();
        let output = root.0.join("must-not-exist");
        let error = split_generation_artifacts(
            &gen2, &gen1, &output, DEFAULT_SPLIT_SEED, &test_bullet_root(),
        ).unwrap_err();
        assert!(error.contains("provenance") || error.contains("manifest"), "unexpected failure cause: {error}");
        assert!(!output.exists());
    }

    #[test]
    fn bullet_layout_guard_pins_little_endian_record_schema() {
        ensure_training_record_layout().unwrap();
        assert_eq!(std::mem::size_of::<TrainingRecord>(), TRAINING_RECORD_BYTES);
        assert_eq!(std::mem::align_of::<TrainingRecord>(), 8);
        assert_eq!(std::mem::offset_of!(TrainingRecord, occ), 0);
        assert_eq!(std::mem::offset_of!(TrainingRecord, pcs), 8);
        assert_eq!(std::mem::offset_of!(TrainingRecord, score), 24);
        assert_eq!(std::mem::offset_of!(TrainingRecord, result), 26);
        assert_eq!(std::mem::offset_of!(TrainingRecord, ksq), 27);
        assert_eq!(std::mem::offset_of!(TrainingRecord, opp_ksq), 28);
        assert_eq!(std::mem::offset_of!(TrainingRecord, extra), 29);
    }

    #[test]
    fn training_interleave_is_deterministic_and_preserves_each_games_row_order() {
        let games = [
            GameRows { game_id: 11, first_record: 0, record_count: 5 },
            GameRows { game_id: 12, first_record: 5, record_count: 5 },
            GameRows { game_id: 13, first_record: 10, record_count: 5 },
        ];
        let first = training_interleave_segments(&games, DEFAULT_SPLIT_SEED, 2).unwrap();
        assert_eq!(first, training_interleave_segments(&games, DEFAULT_SPLIT_SEED, 2).unwrap());
        assert_eq!(first.len(), 9);
        assert_eq!(first.iter().map(|segment| segment.record_count).sum::<u64>(), 15);
        let first_round: BTreeSet<u64> = first[..3].iter().map(|segment| segment.game_id).collect();
        assert_eq!(first_round, BTreeSet::from([11, 12, 13]));
        for game in games {
            let mut segments: Vec<_> = first.iter().filter(|segment| segment.game_id == game.game_id).collect();
            segments.sort_unstable_by_key(|segment| segment.segment_index);
            let mut source_offset = 0;
            for (expected_index, segment) in segments.iter().enumerate() {
                assert_eq!(segment.segment_index, expected_index as u64);
                assert_eq!(segment.source_first_record, game.first_record + source_offset);
                assert_eq!(segment.record_count, (game.record_count - source_offset).min(2));
                source_offset += segment.record_count;
            }
            assert_eq!(source_offset, game.record_count);
        }
        assert!(training_interleave_segments(&games, DEFAULT_SPLIT_SEED, 0).is_err());
        assert!(training_interleave_segments(&[games[0], games[0]], DEFAULT_SPLIT_SEED, 2).is_err());
    }

    #[test]
    fn synthetic_v2_artifact_conversion_hashes_atomicity_and_refusal_are_audited() {
        let root = TestDir::new("synthetic-v2");
        let input = write_synthetic_v2_merged(&root.0);
        let output_a = root.0.join("converted-a");
        let output_b = root.0.join("converted-b");

        let mut expected_by_game = BTreeMap::<u64, Vec<[u8; TRAINING_RECORD_BYTES]>>::new();
        let mut game_ids = Vec::new();
        let source_summary = datagen::inspect_artifact(&input, |game| {
            game_ids.push(game.game_id);
            let rows = game
                .candidates
                .iter()
                .map(|candidate| {
                    encode_training_record(&convert_candidate(game.result_white, candidate).unwrap()).unwrap()
                })
                .collect();
            expected_by_game.insert(game.game_id, rows);
        })
        .unwrap();
        assert_eq!(source_summary.kind, "merged-dataset");
        assert_eq!(source_summary.games, FIXTURE_GAME_COUNT);
        assert_eq!(source_summary.unique_positions, FIXTURE_GAME_COUNT * FIXTURE_ROWS_PER_GAME as u64);
        assert_eq!(
            input.file_name().unwrap().to_str().unwrap(),
            format!("dataset-{}.k1d", source_summary.sha256)
        );
        let input_manifest = input.with_file_name("manifest.txt");
        let correct_manifest = fs::read(&input_manifest).unwrap();
        let valid_manifest_with_dataset_hash = format!(
            "{}dataset_sha256={}\n",
            String::from_utf8(correct_manifest.clone()).unwrap(),
            source_summary.sha256
        );
        fs::write(&input_manifest, &valid_manifest_with_dataset_hash).unwrap();
        read_source_provenance(&input, &source_summary).unwrap();
        let mismatched_dataset_hash = valid_manifest_with_dataset_hash.replace(
            &format!("dataset_sha256={}", source_summary.sha256),
            &format!("dataset_sha256={}", "00".repeat(32)),
        );
        fs::write(&input_manifest, mismatched_dataset_hash).unwrap();
        assert!(read_source_provenance(&input, &source_summary)
            .unwrap_err()
            .contains("dataset_sha256 disagrees"));
        fs::write(&input_manifest, &correct_manifest).unwrap();

        let validation_ids = validation_game_ids(&game_ids, DEFAULT_SPLIT_SEED).unwrap();
        assert_eq!(validation_ids.len(), 2);
        let bullet_root = test_bullet_root();
        assert!(validate_bullet_checkout(&bullet_root.join("examples")).is_err());

        let mislabeled_input = root.0.join("renamed.k1d");
        fs::copy(&input, &mislabeled_input).unwrap();
        let mislabeled_output = root.0.join("mislabeled-output");
        let error = split_artifact(&mislabeled_input, &mislabeled_output, DEFAULT_SPLIT_SEED, &bullet_root).unwrap_err();
        assert!(error.contains("content-addressed name"));
        assert!(!mislabeled_output.exists());

        let bad_manifest = String::from_utf8(correct_manifest.clone())
            .unwrap()
            .replace(&format!("config_sha256={}", "ab".repeat(32)), &format!("config_sha256={}", "00".repeat(32)));
        fs::write(&input_manifest, bad_manifest).unwrap();
        let error = split_artifact(&input, &output_a, DEFAULT_SPLIT_SEED, &bullet_root).unwrap_err();
        assert!(error.contains("artifact config SHA-256 does not match adjacent manifest"));
        assert!(!output_a.exists());
        fs::write(&input_manifest, correct_manifest).unwrap();

        let summary_a = split_artifact(&input, &output_a, DEFAULT_SPLIT_SEED, &bullet_root).unwrap();
        assert_eq!(summary_a.train_games, 98);
        assert_eq!(summary_a.validation_games, 2);
        assert_eq!(summary_a.train_records, 490);
        assert_eq!(summary_a.validation_records, 10);
        assert_eq!(summary_a.source_sha256, source_summary.sha256);
        let train_rows = read_records(&output_a.join("train.bullet"));
        let validation_rows = read_records(&output_a.join("validation.bullet"));
        assert_eq!(train_rows.len(), 490);
        assert_eq!(validation_rows.len(), 10);

        let mut train_owners = BTreeSet::new();
        let train_sidecar = fs::read_to_string(output_a.join("train_game_ids.tsv")).unwrap();
        assert_eq!(train_sidecar.lines().next(), Some("game_id\tsegment_index\tfirst_record\trecord_count"));
        for line in train_sidecar.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 4);
            let game_id: u64 = fields[0].parse().unwrap();
            let segment_index: usize = fields[1].parse().unwrap();
            let first_record: usize = fields[2].parse().unwrap();
            let record_count: usize = fields[3].parse().unwrap();
            assert!(!validation_ids.contains(&game_id), "validation game leaked into train sidecar");
            assert!(train_owners.insert((game_id, segment_index)));
            let expected = expected_by_game.get(&game_id).unwrap();
            let expected_start = segment_index * TRAIN_INTERLEAVE_RECORDS as usize;
            assert_eq!(record_count, expected.len().saturating_sub(expected_start).min(TRAIN_INTERLEAVE_RECORDS as usize));
            assert_eq!(
                &train_rows[first_record..first_record + record_count],
                &expected[expected_start..expected_start + record_count]
            );
        }
        let train_games: BTreeSet<u64> = train_owners.iter().map(|(game_id, _)| *game_id).collect();
        assert_eq!(train_games.len(), 98);
        assert!(train_games.is_disjoint(&validation_ids));

        let mut validation_games = BTreeSet::new();
        let validation_sidecar = fs::read_to_string(output_a.join("validation_game_ids.tsv")).unwrap();
        assert_eq!(validation_sidecar.lines().next(), Some("game_id\tsegment_index\tfirst_record\trecord_count"));
        for line in validation_sidecar.lines().skip(1) {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 4);
            let game_id: u64 = fields[0].parse().unwrap();
            let segment_index: usize = fields[1].parse().unwrap();
            let first_record: usize = fields[2].parse().unwrap();
            let record_count: usize = fields[3].parse().unwrap();
            assert_eq!(segment_index, 0);
            assert!(validation_ids.contains(&game_id), "train game leaked into validation sidecar");
            assert!(validation_games.insert(game_id));
            let expected = expected_by_game.get(&game_id).unwrap();
            assert_eq!(record_count, expected.len());
            assert_eq!(
                &validation_rows[first_record..first_record + record_count],
                expected.as_slice()
            );
        }
        assert_eq!(validation_games, validation_ids);

        let manifest_a = fs::read_to_string(output_a.join("manifest.txt")).unwrap();
        assert_eq!(parse_manifest_value(&manifest_a, "source_sha256"), source_summary.sha256);
        assert_eq!(parse_manifest_value(&manifest_a, "kira_source_tree_sha256"), "cd".repeat(32));
        assert_eq!(parse_manifest_value(&manifest_a, "kira_source_revision"), "not-recorded-in-m3-manifest");
        assert_eq!(parse_manifest_value(&manifest_a, "m3_source_manifest_sha256"), sha256_with_shasum(&input.with_file_name("manifest.txt")).unwrap());
        assert_eq!(parse_manifest_value(&manifest_a, "bullet_git_commit"), PINNED_BULLET_COMMIT);
        assert_eq!(parse_manifest_value(&manifest_a, "bullet_checkout"), fs::canonicalize(&bullet_root).unwrap().display().to_string());
        assert_eq!(parse_manifest_value(&manifest_a, "train_sha256"), sha256_with_shasum(&output_a.join("train.bullet")).unwrap());
        assert_eq!(parse_manifest_value(&manifest_a, "validation_sha256"), sha256_with_shasum(&output_a.join("validation.bullet")).unwrap());
        assert_eq!(parse_manifest_value(&manifest_a, "train_game_ids_sha256"), sha256_with_shasum(&output_a.join("train_game_ids.tsv")).unwrap());
        assert_eq!(parse_manifest_value(&manifest_a, "validation_game_ids_sha256"), sha256_with_shasum(&output_a.join("validation_game_ids.tsv")).unwrap());
        assert_eq!(parse_manifest_value(&manifest_a, "training_interleave_records"), TRAIN_INTERLEAVE_RECORDS.to_string());

        let summary_b = split_artifact(&input, &output_b, DEFAULT_SPLIT_SEED, &bullet_root).unwrap();
        assert_eq!(summary_a.source_sha256, summary_b.source_sha256);
        for file in ["train.bullet", "validation.bullet", "train_game_ids.tsv", "validation_game_ids.tsv", "manifest.txt"] {
            assert_eq!(fs::read(output_a.join(file)).unwrap(), fs::read(output_b.join(file)).unwrap(), "nondeterministic {file}");
        }
        assert!(fs::read_dir(&root.0).unwrap().all(|entry| {
            !entry.unwrap().file_name().to_string_lossy().contains(".tmp-")
        }), "successful conversion left a temporary sibling behind");

        let existing = root.0.join("existing");
        fs::create_dir(&existing).unwrap();
        fs::write(existing.join("sentinel"), b"preserve").unwrap();
        let error = split_artifact(&input, &existing, DEFAULT_SPLIT_SEED, &bullet_root).unwrap_err();
        assert!(error.contains("refusing to overwrite"));
        assert_eq!(fs::read(existing.join("sentinel")).unwrap(), b"preserve");

        let blocked_output = root.0.join("blocked");
        let blocked_temp = root.0.join(format!(".blocked.tmp-{}", std::process::id()));
        fs::create_dir(&blocked_temp).unwrap();
        fs::write(blocked_temp.join("sentinel"), b"keep temp collision").unwrap();
        let error = split_artifact(&input, &blocked_output, DEFAULT_SPLIT_SEED, &bullet_root).unwrap_err();
        assert!(error.contains("create temporary output"));
        assert!(!blocked_output.exists());
        assert_eq!(fs::read(blocked_temp.join("sentinel")).unwrap(), b"keep temp collision");
    }

    #[test]
    fn bullet_to_kira_feature_rows_are_a_bijection_with_bishop_rook_swap() {
        let mapped: Vec<usize> = (0..768)
            .map(|feature| bullet_feature_to_kira(feature).unwrap())
            .collect();
        let distinct: BTreeSet<usize> = mapped.iter().copied().collect();
        assert_eq!(distinct.len(), 768);
        assert_eq!(bullet_feature_to_kira(2 * 64 + 9), Some(3 * 64 + 9));
        assert_eq!(bullet_feature_to_kira(3 * 64 + 9), Some(2 * 64 + 9));
        assert_eq!(bullet_feature_to_kira(384 + 2 * 64 + 9), Some(384 + 3 * 64 + 9));
        assert_eq!(bullet_feature_to_kira(384 + 3 * 64 + 9), Some(384 + 2 * 64 + 9));
        assert_eq!(bullet_feature_to_kira(768), None);
    }

    #[test]
    fn kira_chess768_features_match_kira_perspectives_for_both_turns() {
        let fixtures = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        ];
        for fen in fixtures {
            let fields: Vec<&str> = fen.split_whitespace().collect();
            let white_to_move = fields[1] == "w";
            let bbs = fen_bitboards(fen, white_to_move).unwrap();
            let board = TrainingRecord::from_bitboards(bbs, !white_to_move, 0, 1).unwrap();
            assert_eq!(features_from_board(board), expected_features_from_fen(fen));
        }
    }

    #[test]
    fn format_v2_labels_round_trip_to_kira_stm_record() {
        let white_fen = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1";
        let black_fen = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1";
        for (game_result, white_to_move) in [(0, true), (1, true), (2, true), (0, false), (1, false), (2, false)] {
            let result_stm = result_stm_from_white(game_result, white_to_move).unwrap();
            let row = candidate(
                if white_to_move { white_fen } else { black_fen },
                white_to_move,
                if white_to_move { 137 } else { -137 },
                result_stm,
            );
            let board = convert_candidate(game_result, &row).unwrap();
            assert_eq!(board.score, row.score_stm as i16);
            assert_eq!(board.result, result_stm);
        }
    }

    #[test]
    fn kira_binary_codec_round_trips_exact_32_byte_record() {
        let fen = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1";
        let bbs = fen_bitboards(fen, false).unwrap();
        let board = TrainingRecord::from_bitboards(bbs, true, 123, 2).unwrap();
        let encoded = encode_training_record(&board).unwrap();
        assert_eq!(encoded.len(), 32);
        assert_eq!(i16::from_le_bytes([encoded[24], encoded[25]]), board.score);
        assert_eq!(encoded[26], board.result);
        let decoded = decode_training_record(&encoded).unwrap();
        assert_eq!(decoded, board);
        assert_eq!(encode_training_record(&decoded).unwrap(), encoded);
        assert!(decode_training_record(&encoded[..31]).is_err());
    }

    #[test]
    fn bullet_wdl_weight_matches_kira_score_lambda_target() {
        let bullet_wdl = bullet_constant_wdl_for_kira_lambda(KIRA_SCORE_LAMBDA).unwrap();
        assert_eq!(bullet_wdl, 0.3);
        for score in [-2000i16, -400, 0, 400, 2000] {
            for result_code in 0..=2u8 {
                let result = f32::from(result_code) / 2.0;
                let expected = KIRA_SCORE_LAMBDA
                    * (1.0 / (1.0 + (-(f32::from(score) / EVAL_SCALE)).exp()))
                    + (1.0 - KIRA_SCORE_LAMBDA) * result;
                let board = TrainingRecord { score, result: result_code, ..TrainingRecord::default() };
                let actual = KIRA_SCORE_LAMBDA
                    * (1.0 / (1.0 + (-(f32::from(board.score) / EVAL_SCALE)).exp()))
                    + bullet_wdl * (f32::from(board.result) / 2.0);
                assert!((expected - actual).abs() < 1e-6, "score={score}, result={result_code}");
            }
        }
        assert!(bullet_constant_wdl_for_kira_lambda(f32::NAN).is_err());
        assert!(bullet_constant_wdl_for_kira_lambda(1.1).is_err());
    }
}
