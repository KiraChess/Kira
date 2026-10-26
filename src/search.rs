use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant};

use crate::eval::Evaluator;
use crate::{Board, Move};

pub const MATE_SCORE: i32 = 31_000;
const MATE_THRESHOLD: i32 = MATE_SCORE - 256;
const INFINITY: i32 = 32_000;
const MAX_PLY: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    pub depth: Option<u8>,
    pub nodes: Option<u64>,
    pub deadline: Option<Instant>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self { depth: Some(8), nodes: None, deadline: None }
    }
}

#[derive(Clone, Debug)]
pub struct SearchInfo {
    pub depth: u8,
    pub score: i32,
    pub nodes: u64,
    pub elapsed_ms: u128,
    pub pv: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    key: u64,
    best: Option<Move>,
    score: i16,
    depth: u8,
    bound: Bound,
    generation: u8,
}

/// Search history carries the positions relevant to repetition adjudication.
/// Its order does not matter for repetition counts, so keep a compact
/// order-independent signature alongside the hashes for constant-time TT keys.
#[derive(Clone, Default)]
struct SearchPath {
    hashes: Vec<u64>,
    sum: u64,
    xor: u64,
}

impl SearchPath {
    fn from_hashes(hashes: Vec<u64>) -> Self {
        let mut path = Self::default();
        for hash in hashes {
            path.push(hash);
        }
        path
    }

    fn push(&mut self, hash: u64) {
        self.hashes.push(hash);
        self.sum = self.sum.wrapping_add(path_mix(hash ^ 0x6a09_e667_f3bc_c909));
        self.xor ^= path_mix(hash ^ 0xbb67_ae85_84ca_a73b);
    }

    fn pop(&mut self) -> Option<u64> {
        let hash = self.hashes.pop()?;
        self.sum = self.sum.wrapping_sub(path_mix(hash ^ 0x6a09_e667_f3bc_c909));
        self.xor ^= path_mix(hash ^ 0xbb67_ae85_84ca_a73b);
        Some(hash)
    }

    fn count(&self, hash: u64) -> usize {
        self.hashes.iter().filter(|&&seen| seen == hash).count()
    }

    fn signature(&self) -> u64 {
        path_mix(self.sum ^ self.xor.rotate_left(23) ^ (self.hashes.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }
}

fn path_mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub struct Searcher {
    table: Vec<Option<Entry>>,
    mask: usize,
    generation: u8,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: Vec<i32>,
    nodes: u64,
    started: Instant,
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Self {
        let mut searcher = Self {
            table: Vec::new(),
            mask: 0,
            generation: 0,
            killers: [[None; 2]; MAX_PLY],
            history: vec![0; 2 * 64 * 64],
            nodes: 0,
            started: Instant::now(),
        };
        searcher.resize_hash(hash_mb);
        searcher
    }

    pub fn resize_hash(&mut self, hash_mb: usize) {
        let bytes = hash_mb.max(1).saturating_mul(1024 * 1024);
        let requested = (bytes / size_of::<Option<Entry>>()).max(1024);
        let exponent = usize::BITS - 1 - requested.leading_zeros();
        let entries = 1usize.checked_shl(exponent).unwrap_or(1024).max(1024);
        self.table = vec![None; entries];
        self.mask = entries - 1;
    }

    pub fn clear(&mut self) {
        self.table.fill(None);
        self.killers = [[None; 2]; MAX_PLY];
        self.history.fill(0);
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn hash_capacity_entries(&self) -> usize {
        self.table.len()
    }

    pub fn search<E, F>(
        &mut self,
        board: &Board,
        game_history: &[u64],
        evaluator: &E,
        limits: SearchLimits,
        stop: &AtomicBool,
        mut report: F,
    ) -> SearchResult
    where
        E: Evaluator,
        F: FnMut(SearchInfo),
    {
        self.nodes = 0;
        self.started = Instant::now();
        self.generation = self.generation.wrapping_add(1);
        self.killers = [[None; 2]; MAX_PLY];

        let mut root = board.clone();
        let legal = root.legal_moves();
        if legal.is_empty() {
            return SearchResult {
                best_move: None,
                score: if root.is_in_check(root.white_to_move) { -MATE_SCORE } else { 0 },
                depth: 0,
                nodes: 0,
            };
        }
        let mut best_move = legal[0];
        let mut best_score = evaluator.evaluate(&root);
        let mut completed_depth = 0;

        let mut hashes = game_history.to_vec();
        if hashes.last().copied() != Some(root.hash()) {
            hashes.push(root.hash());
        }
        let relevant_history = root.halfmove as usize + 1;
        if hashes.len() > relevant_history {
            hashes.drain(..hashes.len() - relevant_history);
        }
        let mut path = SearchPath::from_hashes(hashes);
        if root.halfmove >= 100 || repetition_count(&path, root.hash()) >= 3 {
            return SearchResult { best_move: Some(best_move), score: 0, depth: 0, nodes: 0 };
        }

        let max_depth = limits.depth.unwrap_or((MAX_PLY - 1) as u8).clamp(1, (MAX_PLY - 1) as u8);
        for depth in 1..=max_depth {
            match self.search_root(&mut root, &mut path, depth, evaluator, limits, stop) {
                Ok((score, mv)) => {
                    best_score = score;
                    best_move = mv;
                    completed_depth = depth;
                    let pv = self.extract_pv(board, &path, depth);
                    report(SearchInfo {
                        depth,
                        score,
                        nodes: self.nodes,
                        elapsed_ms: self.started.elapsed().as_millis(),
                        pv: pv.iter().map(|m| m.uci()).collect(),
                    });
                }
                Err(()) => break,
            }
            if best_score.abs() >= MATE_SCORE - depth as i32 {
                break;
            }
        }

        SearchResult { best_move: Some(best_move), score: best_score, depth: completed_depth, nodes: self.nodes }
    }

    fn search_root<E: Evaluator>(
        &mut self,
        board: &mut Board,
        path: &mut SearchPath,
        depth: u8,
        evaluator: &E,
        limits: SearchLimits,
        stop: &AtomicBool,
    ) -> Result<(i32, Move), ()> {
        self.visit_node(limits, stop)?;
        let original_alpha = -INFINITY;
        let beta = INFINITY;
        let tt_move = self.probe(transposition_key(board, path)).and_then(|entry| entry.best);
        let mut moves = board.legal_moves();
        self.order_moves(board, &mut moves, tt_move, 0);
        let mut alpha = original_alpha;
        let mut best_score = -INFINITY;
        let mut best_move = moves[0];

        for (index, mv) in moves.into_iter().enumerate() {
            let irreversible = board.squares[mv.from as usize].unsigned_abs() == 1 || is_capture(board, mv);
            let previous_path = if irreversible { Some(std::mem::take(path)) } else { None };
            let undo = board.make_move(mv);
            path.push(board.hash());
            let result = if index == 0 {
                self.negamax(board, path, depth - 1, -beta, -alpha, 1, evaluator, limits, stop).map(|s| -s)
            } else {
                let scout = self.negamax(board, path, depth - 1, -alpha - 1, -alpha, 1, evaluator, limits, stop).map(|s| -s);
                match scout {
                    Ok(score) if score > alpha && score < beta => {
                        self.negamax(board, path, depth - 1, -beta, -alpha, 1, evaluator, limits, stop).map(|s| -s)
                    }
                    other => other,
                }
            };
            if let Some(previous) = previous_path { *path = previous; } else { path.pop(); }
            board.unmake_move(mv, undo);
            let score = result?;
            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }

        let bound = if best_score <= original_alpha { Bound::Upper } else { Bound::Exact };
        self.store(transposition_key(board, path), best_move, best_score, depth, bound, 0);
        Ok((best_score, best_move))
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax<E: Evaluator>(
        &mut self,
        board: &mut Board,
        path: &mut SearchPath,
        depth: u8,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        evaluator: &E,
        limits: SearchLimits,
        stop: &AtomicBool,
    ) -> Result<i32, ()> {
        // Quiescence owns depth-zero node accounting and terminal handling.
        // Routing before visit_node prevents counting the same position once
        // in negamax and again when it enters quiescence.
        if depth == 0 {
            return self.quiescence(board, path, alpha, beta, ply, evaluator, limits, stop);
        }
        self.visit_node(limits, stop)?;
        if board.halfmove >= 100 || repetition_count(path, board.hash()) >= 3 {
            // Checkmate takes precedence over a claimable draw. Only generate
            // legal moves on these draw candidates to keep the normal search
            // path fast.
            let in_check = board.is_in_check(board.white_to_move);
            if board.legal_moves().is_empty() {
                return Ok(if in_check { -MATE_SCORE + ply as i32 } else { 0 });
            }
            return Ok(0);
        }
        if ply >= MAX_PLY - 1 {
            return Ok(evaluator.evaluate(board));
        }

        let original_alpha = alpha;
        let hash = transposition_key(board, path);
        let mut tt_move = None;
        if let Some(entry) = self.probe(hash) {
            tt_move = entry.best;
            if entry.depth >= depth {
                let score = score_from_table(entry.score as i32, ply);
                match entry.bound {
                    Bound::Exact => return Ok(score),
                    Bound::Lower if score >= beta => return Ok(score),
                    Bound::Upper if score <= alpha => return Ok(score),
                    _ => {}
                }
            }
        }

        let in_check = board.is_in_check(board.white_to_move);
        let mut moves = board.legal_moves();
        if moves.is_empty() {
            return Ok(if in_check { -MATE_SCORE + ply as i32 } else { 0 });
        }
        self.order_moves(board, &mut moves, tt_move, ply);
        let mut best_score = -INFINITY;
        let mut best_move = moves[0];

        for (index, mv) in moves.into_iter().enumerate() {
            let mover_side = usize::from(board.white_to_move);
            let quiet = !is_capture(board, mv) && mv.promotion == 0;
            let irreversible = board.squares[mv.from as usize].unsigned_abs() == 1 || is_capture(board, mv);
            let previous_path = if irreversible { Some(std::mem::take(path)) } else { None };
            let undo = board.make_move(mv);
            path.push(board.hash());
            let result = if index == 0 {
                self.negamax(board, path, depth - 1, -beta, -alpha, ply + 1, evaluator, limits, stop).map(|s| -s)
            } else {
                let scout = self.negamax(board, path, depth - 1, -alpha - 1, -alpha, ply + 1, evaluator, limits, stop).map(|s| -s);
                match scout {
                    Ok(score) if score > alpha && score < beta => {
                        self.negamax(board, path, depth - 1, -beta, -alpha, ply + 1, evaluator, limits, stop).map(|s| -s)
                    }
                    other => other,
                }
            };
            if let Some(previous) = previous_path { *path = previous; } else { path.pop(); }
            board.unmake_move(mv, undo);
            let score = result?;

            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                if quiet {
                    self.record_quiet_cutoff(mv, mover_side, ply, depth);
                }
                break;
            }
        }

        let bound = if best_score <= original_alpha { Bound::Upper } else if best_score >= beta { Bound::Lower } else { Bound::Exact };
        self.store(hash, best_move, best_score, depth, bound, ply);
        Ok(best_score)
    }

    #[allow(clippy::too_many_arguments)]
    fn quiescence<E: Evaluator>(
        &mut self,
        board: &mut Board,
        path: &mut SearchPath,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        evaluator: &E,
        limits: SearchLimits,
        stop: &AtomicBool,
    ) -> Result<i32, ()> {
        self.visit_node(limits, stop)?;
        if ply >= MAX_PLY - 1 {
            return Ok(evaluator.evaluate(board));
        }

        let in_check = board.is_in_check(board.white_to_move);
        let mut moves = board.legal_moves();
        if moves.is_empty() {
            return Ok(if in_check { -MATE_SCORE + ply as i32 } else { 0 });
        }
        if board.halfmove >= 100 || repetition_count(path, board.hash()) >= 3 {
            return Ok(0);
        }

        let mut stand_pat = -INFINITY;
        if !in_check {
            stand_pat = evaluator.evaluate(board);
            if stand_pat >= beta {
                return Ok(stand_pat);
            }
            alpha = alpha.max(stand_pat);
        }

        if !in_check {
            moves.retain(|&mv| is_capture(board, mv) || mv.promotion != 0);
            if moves.is_empty() {
                return Ok(alpha.max(stand_pat));
            }
        }
        self.order_moves(board, &mut moves, None, ply);

        for mv in moves {
            let irreversible = board.squares[mv.from as usize].unsigned_abs() == 1 || is_capture(board, mv);
            let previous_path = if irreversible { Some(std::mem::take(path)) } else { None };
            let undo = board.make_move(mv);
            path.push(board.hash());
            let result = self.quiescence(board, path, -beta, -alpha, ply + 1, evaluator, limits, stop).map(|s| -s);
            if let Some(previous) = previous_path { *path = previous; } else { path.pop(); }
            board.unmake_move(mv, undo);
            let score = result?;
            if score >= beta {
                return Ok(score);
            }
            alpha = alpha.max(score);
        }
        Ok(alpha)
    }

    fn visit_node(&mut self, limits: SearchLimits, stop: &AtomicBool) -> Result<(), ()> {
        self.nodes = self.nodes.saturating_add(1);
        if stop.load(Ordering::Relaxed)
            || limits.nodes.is_some_and(|limit| self.nodes >= limit)
            || limits.deadline.is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(());
        }
        Ok(())
    }

    fn order_moves(&self, board: &Board, moves: &mut [Move], tt_move: Option<Move>, ply: usize) {
        let side = usize::from(board.white_to_move);
        moves.sort_unstable_by_key(|&mv| {
            let score = if Some(mv) == tt_move {
                2_000_000
            } else if is_capture(board, mv) {
                let victim = captured_piece(board, mv).unsigned_abs() as i32;
                let attacker = board.squares[mv.from as usize].unsigned_abs() as i32;
                1_000_000 + piece_value(victim) * 16 - piece_value(attacker)
            } else if mv.promotion != 0 {
                900_000 + piece_value(mv.promotion.unsigned_abs() as i32)
            } else if ply < MAX_PLY && self.killers[ply][0] == Some(mv) {
                800_000
            } else if ply < MAX_PLY && self.killers[ply][1] == Some(mv) {
                799_000
            } else {
                self.history[history_index(side, mv.from, mv.to)]
            };
            std::cmp::Reverse(score)
        });
    }

    fn record_quiet_cutoff(&mut self, mv: Move, side: usize, ply: usize, depth: u8) {
        if ply < MAX_PLY && self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
        let index = history_index(side, mv.from, mv.to);
        self.history[index] = (self.history[index] + i32::from(depth) * i32::from(depth)).min(16_000);
    }

    fn probe(&self, key: u64) -> Option<Entry> {
        self.table[key as usize & self.mask].filter(|entry| entry.key == key)
    }

    fn store(&mut self, key: u64, best: Move, score: i32, depth: u8, bound: Bound, ply: usize) {
        let index = key as usize & self.mask;
        let replace = self.table[index].is_none_or(|entry| {
            entry.key == key || entry.generation != self.generation || depth >= entry.depth
        });
        if replace {
            self.table[index] = Some(Entry {
                key,
                best: Some(best),
                score: score_to_table(score, ply).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                depth,
                bound,
                generation: self.generation,
            });
        }
    }

    fn extract_pv(&self, root: &Board, initial_path: &SearchPath, max_depth: u8) -> Vec<Move> {
        let mut board = root.clone();
        let mut path = initial_path.clone();
        let mut pv = Vec::new();
        for _ in 0..max_depth {
            let Some(entry) = self.probe(transposition_key(&board, &path)) else { break; };
            let Some(mv) = entry.best else { break; };
            if !board.legal_moves().contains(&mv) {
                break;
            }
            let irreversible = board.squares[mv.from as usize].unsigned_abs() == 1 || is_capture(&board, mv);
            if irreversible { path = SearchPath::default(); }
            pv.push(mv);
            board.make_move(mv);
            path.push(board.hash());
        }
        pv
    }
}

fn history_index(side: usize, from: u8, to: u8) -> usize {
    side * 4096 + from as usize * 64 + to as usize
}

fn repetition_count(history: &SearchPath, key: u64) -> usize {
    history.count(key)
}

fn transposition_key(board: &Board, path: &SearchPath) -> u64 {
    // Board hashes intentionally omit the halfmove clock because it does not
    // affect repetition identity. TT scores do depend on the rule-50 clock,
    // so mix that context into the transposition key separately.
    let mut clock = u64::from(board.halfmove.min(100)).wrapping_add(0x9e37_79b9_7f4a_7c15);
    clock = (clock ^ (clock >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    clock = (clock ^ (clock >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    clock ^= clock >> 31;
    board.hash() ^ clock ^ path.signature()
}

fn score_to_table(score: i32, ply: usize) -> i32 {
    if score >= MATE_THRESHOLD {
        score + ply as i32
    } else if score <= -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_table(score: i32, ply: usize) -> i32 {
    if score >= MATE_THRESHOLD {
        score - ply as i32
    } else if score <= -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

fn is_capture(board: &Board, mv: Move) -> bool {
    if board.squares[mv.to as usize] != 0 {
        return true;
    }
    board.squares[mv.from as usize].unsigned_abs() == 1
        && mv.from % 8 != mv.to % 8
        && board.ep == Some(mv.to)
}

fn captured_piece(board: &Board, mv: Move) -> i8 {
    if board.squares[mv.to as usize] != 0 {
        board.squares[mv.to as usize]
    } else if board.squares[mv.from as usize] > 0 {
        board.squares[mv.to as usize - 8]
    } else {
        board.squares[mv.to as usize + 8]
    }
}

fn piece_value(kind: i32) -> i32 {
    match kind {
        1 => 100,
        2 => 320,
        3 => 500,
        4 => 335,
        5 => 900,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Hce;
    use crate::Board;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn search_returns_legal_moves_and_respects_depth_and_nodes() {
        let mut board = Board::startpos();
        let history = [board.hash()];
        let mut searcher = Searcher::new(1);
        let stop = AtomicBool::new(false);
        let result = searcher.search(
            &board,
            &history,
            &Hce,
            SearchLimits { depth: Some(4), nodes: None, deadline: None },
            &stop,
            |_| {},
        );
        assert_eq!(result.depth, 4);
        assert!(result.nodes > 0);
        assert!(board.legal_moves().contains(&result.best_move.unwrap()));

        let limited = searcher.search(
            &board,
            &history,
            &Hce,
            SearchLimits { depth: None, nodes: Some(100), deadline: None },
            &stop,
            |_| {},
        );
        assert!(limited.nodes <= 100);
        assert!(board.legal_moves().contains(&limited.best_move.unwrap()));
    }

    #[test]
    fn search_finds_mate_and_reports_terminal_positions() {
        let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let mut searcher = Searcher::new(1);
        let stop = AtomicBool::new(false);
        let result = searcher.search(
            &board,
            &[board.hash()],
            &Hce,
            SearchLimits { depth: Some(3), nodes: None, deadline: None },
            &stop,
            |_| {},
        );
        assert!(result.score >= MATE_THRESHOLD);

        let checkmate = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let terminal = searcher.search(
            &checkmate,
            &[checkmate.hash()],
            &Hce,
            SearchLimits::default(),
            &stop,
            |_| {},
        );
        assert_eq!(terminal.best_move, None);
        assert_eq!(terminal.score, -MATE_SCORE);
    }

    #[test]
    fn checkmate_takes_precedence_over_fifty_move_draw_at_quiescence() {
        let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 99 50").unwrap();
        let mut searcher = Searcher::new(1);
        let stop = AtomicBool::new(false);
        let result = searcher.search(
            &board,
            &[board.hash()],
            &Hce,
            SearchLimits { depth: Some(1), nodes: None, deadline: None },
            &stop,
            |_| {},
        );
        assert_eq!(result.best_move.unwrap().uci(), "f7g7");
        assert_eq!(result.score, MATE_SCORE - 1);
    }

    #[test]
    fn quiescence_scores_stalemate_as_draw() {
        let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut searcher = Searcher::new(1);
        let stop = AtomicBool::new(false);
        let mut position = board.clone();
        let mut path = SearchPath::from_hashes(vec![board.hash()]);
        let score = searcher.quiescence(
            &mut position,
            &mut path,
            -INFINITY,
            INFINITY,
            0,
            &Hce,
            SearchLimits::default(),
            &stop,
        ).unwrap();
        assert_eq!(score, 0);
    }

    #[test]
    fn transposition_scores_are_scoped_to_the_halfmove_clock() {
        let near_fifty = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 98 1").unwrap();
        let start_of_clock = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let stop = AtomicBool::new(false);
        let limits = SearchLimits { depth: Some(2), nodes: None, deadline: None };
        let mut reused_searcher = Searcher::new(1);
        let _ = reused_searcher.search(
            &near_fifty,
            &[near_fifty.hash()],
            &Hce,
            limits,
            &stop,
            |_| {},
        );
        let reused = reused_searcher.search(
            &start_of_clock,
            &[start_of_clock.hash()],
            &Hce,
            limits,
            &stop,
            |_| {},
        );

        let mut fresh_searcher = Searcher::new(1);
        let fresh = fresh_searcher.search(
            &start_of_clock,
            &[start_of_clock.hash()],
            &Hce,
            limits,
            &stop,
            |_| {},
        );
        assert_eq!(reused.score, fresh.score);
        assert!(reused.score > 800, "unexpected clock-0 score: {}", reused.score);
    }

    #[test]
    fn transposition_scores_are_scoped_to_repetition_history() {
        let p_fen = "k1r3n1/1nB5/1BB5/8/8/3q4/8/1N4NK w - - 0 1";
        let mut p = Board::from_fen(p_fen).unwrap();
        let mut history = vec![p.hash()];
        for text in ["g1f3", "g8f6", "b1c3", "f6g8", "c3b1", "g8f6", "f3g1", "f6g8"] {
            p.play_uci(text).unwrap();
            history.push(p.hash());
        }
        assert_eq!(p.halfmove, 8);

        let mut q = Board::from_fen("k1r3n1/2B5/1BB5/2n5/8/3q4/8/1N4NK b - - 7 4").unwrap();
        assert_eq!(q.legal_moves().iter().filter(|mv| mv.uci() == "c5b7").count(), 1);
        let stop = AtomicBool::new(false);
        let mut reused_searcher = Searcher::new(1);
        let _ = reused_searcher.search(
            &p,
            &history,
            &Hce,
            SearchLimits { depth: Some(3), nodes: None, deadline: None },
            &stop,
            |_| {},
        );
        let reused = reused_searcher.search(
            &q,
            &[q.hash()],
            &Hce,
            SearchLimits { depth: Some(4), nodes: None, deadline: None },
            &stop,
            |_| {},
        );

        let mut fresh_searcher = Searcher::new(1);
        let fresh = fresh_searcher.search(
            &q,
            &[q.hash()],
            &Hce,
            SearchLimits { depth: Some(4), nodes: None, deadline: None },
            &stop,
            |_| {},
        );
        assert_eq!(reused.score, fresh.score);
        assert_eq!(reused.score, 670);
    }

    #[test]
    fn root_history_detects_threefold_and_fifty_move_draws() {
        let mut repeated = Board::startpos();
        let mut history = vec![repeated.hash()];
        for text in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"] {
            repeated.play_uci(text).unwrap();
            history.push(repeated.hash());
        }
        let mut searcher = Searcher::new(1);
        let stop = AtomicBool::new(false);
        let repetition_result = searcher.search(
            &repeated,
            &history,
            &Hce,
            SearchLimits::default(),
            &stop,
            |_| {},
        );
        assert_eq!(repetition_result.score, 0);
        assert_eq!(repetition_result.depth, 0);

        let fifty_move = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 51").unwrap();
        let fifty_result = searcher.search(
            &fifty_move,
            &[fifty_move.hash()],
            &Hce,
            SearchLimits::default(),
            &stop,
            |_| {},
        );
        assert_eq!(fifty_result.score, 0);
        assert_eq!(fifty_result.depth, 0);
    }
}
