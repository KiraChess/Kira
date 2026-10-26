//! Float NNUE feature-transformer reference path.
//!
//! Each perspective has a 768-input feature transform. A feature identifies a
//! friendly/enemy piece kind and a square viewed from that perspective. The
//! accumulator stores the feature-transform sum before activation; output
//! concatenation puts the side-to-move perspective first.

use crate::{Board, Move};
use std::sync::atomic::{AtomicU64, Ordering};

pub const FEATURE_COUNT: usize = 768;
const TREE_LEAVES: usize = 64;
const TREE_LEAF_BASE: usize = TREE_LEAVES - 1;
const TREE_NODE_COUNT: usize = TREE_LEAVES * 2 - 1;
static NEXT_NETWORK_ID: AtomicU64 = AtomicU64::new(1);

/// A single-hidden-layer NNUE used as the float reference for training/export.
#[derive(Clone, Debug)]
pub struct NnueNetwork<const H: usize> {
    feature_weights: Vec<f32>,
    hidden_bias: [f32; H],
    output_weights: [[f32; H]; 2],
    output_bias: f32,
    output_scale: f32,
    identity: u64,
}

impl<const H: usize> NnueNetwork<H> {
    pub fn from_parts(
        feature_weights: Vec<f32>,
        hidden_bias: [f32; H],
        output_weights: [[f32; H]; 2],
        output_bias: f32,
        output_scale: f32,
    ) -> Result<Self, String> {
        if H == 0 {
            return Err("NNUE hidden size must be positive".into());
        }
        if feature_weights.len() != FEATURE_COUNT * H {
            return Err(format!(
                "expected {} feature weights, received {}",
                FEATURE_COUNT * H,
                feature_weights.len()
            ));
        }
        if !output_scale.is_finite() || output_scale <= 0.0 {
            return Err("NNUE output scale must be finite and positive".into());
        }
        if !output_bias.is_finite()
            || hidden_bias.iter().any(|x| !x.is_finite())
            || output_weights.iter().flatten().any(|x| !x.is_finite())
            || feature_weights.iter().any(|x| !x.is_finite())
        {
            return Err("NNUE parameters must all be finite".into());
        }
        Ok(Self {
            feature_weights,
            hidden_bias,
            output_weights,
            output_bias,
            output_scale,
            identity: NEXT_NETWORK_ID.fetch_add(1, Ordering::Relaxed),
        })
    }

    pub fn feature_weights(&self) -> &[f32] {
        &self.feature_weights
    }

    pub fn hidden_bias(&self) -> &[f32; H] {
        &self.hidden_bias
    }

    pub fn output_weights(&self) -> &[[f32; H]; 2] {
        &self.output_weights
    }

    pub fn output_bias(&self) -> f32 {
        self.output_bias
    }

    pub fn output_scale(&self) -> f32 {
        self.output_scale
    }

    fn feature_row(&self, feature: usize) -> &[f32] {
        let start = feature * H;
        &self.feature_weights[start..start + H]
    }
}

impl<const H: usize> crate::eval::Evaluator for NnueNetwork<H> {
    fn evaluate(&self, board: &Board) -> i32 {
        NnueAccumulator::full_recompute(board, self)
            .evaluate(board, self)
            .expect("freshly computed NNUE accumulator matches its board")
    }
}

/// The two pre-activation feature-transform sums for one board position.
#[derive(Clone, Debug)]
pub struct NnueAccumulator<const H: usize> {
    values: [[f32; H]; 2],
    // A fixed pairwise tree makes the sum independent of the order in which
    // a position was reached. A move only rebuilds paths for changed squares.
    nodes: [[[f32; H]; TREE_NODE_COUNT]; 2],
    board_hash: u64,
    network_identity: u64,
}

impl<const H: usize> NnueAccumulator<H> {
    pub fn full_recompute(board: &Board, network: &NnueNetwork<H>) -> Self {
        let mut nodes = [[[0.0; H]; TREE_NODE_COUNT]; 2];
        for (square, &piece) in board.squares.iter().enumerate() {
            for perspective in 0..2 {
                nodes[perspective][TREE_LEAF_BASE + square] =
                    feature_leaf(square, piece, perspective == 0, network);
            }
        }
        for node in (0..TREE_LEAF_BASE).rev() {
            for perspective_nodes in &mut nodes {
                let left = perspective_nodes[node * 2 + 1];
                let right = perspective_nodes[node * 2 + 2];
                for neuron in 0..H {
                    perspective_nodes[node][neuron] = left[neuron] + right[neuron];
                }
            }
        }
        let values = [nodes[0][0], nodes[1][0]];
        Self {
            values,
            nodes,
            board_hash: board.hash(),
            network_identity: network.identity,
        }
    }

    pub fn values(&self) -> &[[f32; H]; 2] {
        &self.values
    }

    pub fn board_hash(&self) -> u64 {
        self.board_hash
    }

    /// Compare the IEEE-754 bit patterns, including signed zero.
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        self.board_hash == other.board_hash
            && self.network_identity == other.network_identity
            && self.values.iter().flatten().zip(other.values.iter().flatten())
                .all(|(left, right)| left.to_bits() == right.to_bits())
            && self.nodes.iter().flatten().flatten()
                .zip(other.nodes.iter().flatten().flatten())
                .all(|(left, right)| left.to_bits() == right.to_bits())
    }

    pub fn evaluate(&self, board: &Board, network: &NnueNetwork<H>) -> Result<i32, String> {
        if self.network_identity != network.identity {
            return Err("NNUE accumulator belongs to a different network".into());
        }
        if self.board_hash != board.hash() {
            return Err("NNUE accumulator does not match board position".into());
        }
        let first = if board.white_to_move { 0 } else { 1 };
        let second = 1 - first;
        let mut output = network.output_bias;
        for neuron in 0..H {
            let first_activation = self.values[first][neuron].clamp(0.0, 1.0);
            let second_activation = self.values[second][neuron].clamp(0.0, 1.0);
            output += first_activation * network.output_weights[0][neuron];
            output += second_activation * network.output_weights[1][neuron];
        }
        Ok((output * network.output_scale).round() as i32)
    }
}

/// A reversible stack of accumulators for a search or game history.
#[derive(Clone, Debug)]
pub struct NnueAccumulatorStack<const H: usize> {
    current: NnueAccumulator<H>,
    frames: Vec<NnueFrameUndo<H>>,
}

#[derive(Clone, Debug)]
struct NnueFrameUndo<const H: usize> {
    parent_hash: u64,
    child_hash: u64,
    parent_state: Board,
    child_state: Board,
    changed_nodes: Vec<(usize, [[f32; H]; 2])>,
}

/// Opaque board undo token returned by [`NnueAccumulatorStack::make_uci_move`].
pub struct NnueMoveUndo {
    mv: Move,
    board_undo: crate::Undo,
    parent_hash: u64,
    child_hash: u64,
    parent_state: Board,
    child_state: Board,
}

impl<const H: usize> NnueAccumulatorStack<H> {
    pub fn new(board: &Board, network: &NnueNetwork<H>) -> Self {
        Self {
            current: NnueAccumulator::full_recompute(board, network),
            frames: Vec::new(),
        }
    }

    pub fn current(&self) -> &NnueAccumulator<H> {
        &self.current
    }

    pub fn depth(&self) -> usize {
        self.frames.len() + 1
    }

    /// Play a legal UCI move and update the board and both accumulators.
    pub fn make_uci_move(
        &mut self,
        board: &mut Board,
        network: &NnueNetwork<H>,
        uci: &str,
    ) -> Result<NnueMoveUndo, String> {
        if self.current.network_identity != network.identity {
            return Err("NNUE accumulator stack belongs to a different network".into());
        }
        let requested = Move::from_uci(uci).ok_or_else(|| format!("invalid UCI move: {uci}"))?;
        let mv = board.legal_moves().into_iter().find(|candidate| {
            candidate.from == requested.from
                && candidate.to == requested.to
                && candidate.promotion.abs() == requested.promotion.abs()
        }).ok_or_else(|| format!("illegal UCI move: {uci}"))?;
        let parent = board.clone();
        let board_undo = board.make_move(mv);
        if let Err(error) = self.push_transition(&parent, board, network) {
            board.unmake_move(mv, board_undo);
            return Err(error);
        }
        Ok(NnueMoveUndo {
            mv,
            board_undo,
            parent_hash: parent.hash(),
            child_hash: board.hash(),
            parent_state: parent,
            child_state: board.clone(),
        })
    }

    /// Unmake a move through the matching board and accumulator frames.
    pub fn unmake_move(&mut self, board: &mut Board, undo: NnueMoveUndo) -> Result<(), String> {
        if board.hash() != undo.child_hash || self.current().board_hash != undo.child_hash {
            return Err("board does not match NNUE move undo token".into());
        }
        if board != &undo.child_state {
            return Err("board state does not match NNUE move undo token".into());
        }
        if self.frames.last().is_none_or(|frame| frame.parent_hash != undo.parent_hash
            || frame.child_hash != undo.child_hash
            || frame.parent_state != undo.parent_state
            || frame.child_state != undo.child_state)
        {
            return Err("NNUE move undo token does not match accumulator stack".into());
        }
        board.unmake_move(undo.mv, undo.board_undo);
        self.pop(board)
    }

    /// Push a child position after the caller has made a legal board move.
    /// Only squares whose piece changed contribute feature-row deltas. This
    /// naturally covers captures, castling, en passant and promotions.
    fn push_transition(
        &mut self,
        parent: &Board,
        child: &Board,
        network: &NnueNetwork<H>,
    ) -> Result<(), String> {
        if self.current.network_identity != network.identity {
            return Err("NNUE accumulator stack belongs to a different network".into());
        }
        if self.current.board_hash != parent.hash() {
            return Err("NNUE accumulator stack is not at the parent position".into());
        }
        let mut touched = [false; TREE_NODE_COUNT];
        for square in 0..64u8 {
            let before = parent.squares[square as usize];
            let after = child.squares[square as usize];
            if before == after {
                continue;
            }
            let mut node = TREE_LEAF_BASE + square as usize;
            loop {
                touched[node] = true;
                if node == 0 {
                    break;
                }
                node = (node - 1) / 2;
            }
        }
        let mut changed_nodes = Vec::new();
        for (node, &is_touched) in touched.iter().enumerate() {
            if is_touched {
                changed_nodes.push((node, [self.current.nodes[0][node], self.current.nodes[1][node]]));
            }
        }
        for square in 0..64u8 {
            let before = parent.squares[square as usize];
            let after = child.squares[square as usize];
            if before != after {
                for perspective in 0..2 {
                    self.current.nodes[perspective][TREE_LEAF_BASE + square as usize] =
                        feature_leaf(square as usize, after, perspective == 0, network);
                }
            }
        }
        for node in (0..TREE_LEAF_BASE).rev() {
            if touched[node] {
                for perspective_nodes in &mut self.current.nodes {
                    let left = perspective_nodes[node * 2 + 1];
                    let right = perspective_nodes[node * 2 + 2];
                    for neuron in 0..H {
                        perspective_nodes[node][neuron] = left[neuron] + right[neuron];
                    }
                }
            }
        }
        self.current.values = [self.current.nodes[0][0], self.current.nodes[1][0]];
        let parent_hash = parent.hash();
        let child_hash = child.hash();
        self.current.board_hash = child_hash;
        self.frames.push(NnueFrameUndo {
            parent_hash,
            child_hash,
            parent_state: parent.clone(),
            child_state: child.clone(),
            changed_nodes,
        });
        Ok(())
    }

    /// Restore the parent frame after the caller unmakes the corresponding move.
    fn pop(&mut self, restored_parent: &Board) -> Result<(), String> {
        let Some(frame) = self.frames.last() else {
            return Err("cannot pop the root NNUE accumulator".into());
        };
        if frame.parent_hash != restored_parent.hash()
            || frame.child_hash != self.current.board_hash
            || frame.parent_state != *restored_parent
        {
            return Err("restored board does not match NNUE accumulator parent".into());
        }
        let frame = self.frames.pop().expect("checked above");
        for (node, old_values) in frame.changed_nodes {
            for perspective in 0..2 {
                self.current.nodes[perspective][node] = old_values[perspective];
            }
        }
        self.current.values = [self.current.nodes[0][0], self.current.nodes[1][0]];
        self.current.board_hash = frame.parent_hash;
        Ok(())
    }
}

fn feature_leaf<const H: usize>(
    square: usize,
    piece: i8,
    perspective_white: bool,
    network: &NnueNetwork<H>,
) -> [f32; H] {
    let mut values = if square == 0 { network.hidden_bias } else { [0.0; H] };
    if piece != 0 {
        let feature = feature_index(piece, square as u8, perspective_white)
            .expect("board contains only valid pieces and squares");
        for (value, weight) in values.iter_mut().zip(network.feature_row(feature)) {
            *value += weight;
        }
    }
    values
}

/// Map a signed board piece into the 768-input side-relative feature space.
/// `perspective_white` selects the White or Black view; Black's view flips ranks.
pub fn feature_index(piece: i8, square: u8, perspective_white: bool) -> Option<usize> {
    if piece == 0 || piece.unsigned_abs() > 6 || square >= 64 {
        return None;
    }
    let friendly = (piece > 0) == perspective_white;
    let color = if friendly { 0 } else { 1 };
    let kind = piece.unsigned_abs() as usize - 1;
    let viewed_square = if perspective_white { square } else { square ^ 56 };
    Some((color * 6 + kind) * 64 + viewed_square as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Board;
    use std::collections::HashSet;

    fn deterministic_network<const H: usize>() -> NnueNetwork<H> {
        let mut seed = 0x4b49_5241_4e4e_5545u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let feature_weights = (0..FEATURE_COUNT * H)
            .map(|_| (next() % 5) as i32 as f32 - 2.0)
            .collect();
        let hidden_bias = std::array::from_fn(|_| (next() % 3) as i32 as f32 - 1.0);
        NnueNetwork::from_parts(
            feature_weights,
            hidden_bias,
            [[0.0; H]; 2],
            0.0,
            400.0,
        )
        .unwrap()
    }

    #[test]
    fn million_random_make_unmake_sequences_match_full_recompute_bit_exactly() {
        const SEQUENCES: usize = 1_000_001;
        const PLIES_PER_SEQUENCE: usize = 5;
        const HIDDEN: usize = 128;
        assert!(SEQUENCES > 1_000_000);
        let network = deterministic_network::<HIDDEN>();
        let mut random = 0x2026_1024_4b49_5241u64;
        let mut sequences = HashSet::with_capacity(SEQUENCES);
        let mut sampled = 0usize;
        let mut transitions = 0usize;

        while sampled < SEQUENCES {
            // Sample a complete legal path, keyed by its UCI move sequence so
            // the million count means one million distinct paths, not repeated
            // trials from a small set of opening moves.
            let mut sampled_board = Board::startpos();
            let mut path = Vec::with_capacity(PLIES_PER_SEQUENCE);
            let mut path_uci = Vec::with_capacity(PLIES_PER_SEQUENCE);
            for _ in 0..PLIES_PER_SEQUENCE {
                let legal = sampled_board.legal_moves();
                if legal.is_empty() {
                    break;
                }
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let mv = legal[(random as usize) % legal.len()];
                path.push(mv);
                path_uci.push(mv.uci());
                sampled_board.make_move(mv);
            }
            if path.len() != PLIES_PER_SEQUENCE || !sequences.insert(path_uci.join(" ")) {
                continue;
            }

            let mut board = Board::startpos();
            let mut accumulators = NnueAccumulatorStack::new(&board, &network);
            let root_fen = board.to_fen();
            let root_hash = board.hash();
            let root_accumulator = accumulators.current().clone();
            let mut undos = Vec::with_capacity(PLIES_PER_SEQUENCE);
            for (ply, mv) in path.iter().copied().enumerate() {
                let parent = board.clone();
                let board_undo = board.make_move(mv);
                accumulators.push_transition(&parent, &board, &network).unwrap();
                transitions += 1;
                let recomputed = NnueAccumulator::full_recompute(&board, &network);
                assert!(
                    accumulators.current().bitwise_eq(&recomputed),
                    "incremental mismatch at sampled sequence {sampled}, ply {ply}, move {}, FEN {}",
                    mv.uci(), parent.to_fen()
                );
                undos.push((mv, board_undo));
            }

            for (ply, (mv, board_undo)) in undos.into_iter().enumerate().rev() {
                board.unmake_move(mv, board_undo);
                accumulators.pop(&board).unwrap();
                transitions += 1;
                assert_eq!(
                    board.hash(),
                    accumulators.current().board_hash(),
                    "board and accumulator hash differ at sampled sequence {sampled}, unmake ply {ply}"
                );
                let recomputed = NnueAccumulator::full_recompute(&board, &network);
                assert!(
                    accumulators.current().bitwise_eq(&recomputed),
                    "restored accumulator mismatch at sampled sequence {sampled}, unmake ply {ply}, move {}",
                    mv.uci()
                );
            }

            assert!(
                accumulators.current().bitwise_eq(&root_accumulator),
                "root drift at sampled sequence {sampled}"
            );
            assert_eq!(board.to_fen(), root_fen, "board FEN did not restore at sampled sequence {sampled}");
            assert_eq!(board.hash(), root_hash, "board hash did not restore at sampled sequence {sampled}");
            sampled += 1;
        }
        assert_eq!(sampled, SEQUENCES);
        assert_eq!(sequences.len(), SEQUENCES);
        assert_eq!(transitions, SEQUENCES * PLIES_PER_SEQUENCE * 2);
    }

    #[test]
    fn incremental_accumulator_handles_capture_castling_en_passant_and_promotions() {
        let network = deterministic_network::<128>();
        let cases = [
            ("capture", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
            ("castling", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("queen-side castling", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
            ("en-passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ];
        for (label, fen, uci) in cases {
            let mut board = Board::from_fen(fen).unwrap();
            let mut stack = NnueAccumulatorStack::new(&board, &network);
            let undo = stack.make_uci_move(&mut board, &network, uci).unwrap();
            assert!(stack.current().bitwise_eq(&NnueAccumulator::full_recompute(&board, &network)), "{label}");
            stack.unmake_move(&mut board, undo).unwrap();
            assert!(stack.current().bitwise_eq(&NnueAccumulator::full_recompute(&board, &network)), "{label} unmake");
        }

        for promotion in ["a7a8q", "a7a8r", "a7a8b", "a7a8n"] {
            let mut board = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
            let mut stack = NnueAccumulatorStack::new(&board, &network);
            let undo = stack.make_uci_move(&mut board, &network, promotion).unwrap();
            assert!(stack.current().bitwise_eq(&NnueAccumulator::full_recompute(&board, &network)), "{promotion}");
            stack.unmake_move(&mut board, undo).unwrap();
            assert!(stack.current().bitwise_eq(&NnueAccumulator::full_recompute(&board, &network)), "{promotion} unmake");
        }
    }

    #[test]
    fn tiny_network_matches_hand_computed_output_and_perspective_order() {
        let mut weights = vec![0.0f32; FEATURE_COUNT * 2];
        let mut row = |feature: usize, values: [f32; 2]| {
            weights[feature * 2] = values[0];
            weights[feature * 2 + 1] = values[1];
        };
        row(feature_index(6, 4, true).unwrap(), [0.25, 0.0]); // White king e1.
        row(feature_index(-6, 60, true).unwrap(), [0.0, 0.5]); // Black king e8.
        row(feature_index(6, 4, false).unwrap(), [0.125, 0.25]); // White king in Black view.
        row(feature_index(-6, 60, false).unwrap(), [0.125, 0.25]); // Black king in Black view.
        row(feature_index(1, 8, true).unwrap(), [0.25, 0.25]); // White pawn a2.
        row(feature_index(1, 8, false).unwrap(), [0.5, 0.0]); // Same pawn in Black view.
        let network = NnueNetwork::from_parts(
            weights,
            [0.125, 0.25],
            [[1.0 / 64.0, 1.0 / 32.0], [1.0 / 32.0, 1.0 / 16.0]],
            1.0 / 128.0,
            400.0,
        )
        .unwrap();
        let white_to_move = Board::from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1").unwrap();
        let white_acc = NnueAccumulator::full_recompute(&white_to_move, &network);
        assert_eq!(white_acc.values()[0], [0.625, 1.0]);
        assert_eq!(white_acc.values()[1], [0.875, 0.75]);
        assert_eq!(white_acc.evaluate(&white_to_move, &network).unwrap(), 49);

        let black_to_move = Board::from_fen("4k3/8/8/8/8/8/P7/4K3 b - - 0 1").unwrap();
        let black_acc = NnueAccumulator::full_recompute(&black_to_move, &network);
        assert_eq!(black_acc.evaluate(&black_to_move, &network).unwrap(), 51);

        let color_flipped = Board::from_fen("4k3/p7/8/8/8/8/8/4K3 b - - 0 1").unwrap();
        let flipped_acc = NnueAccumulator::full_recompute(&color_flipped, &network);
        assert_eq!(white_acc.values()[0], flipped_acc.values()[1]);
        assert_eq!(white_acc.values()[1], flipped_acc.values()[0]);
        assert_eq!(white_acc.evaluate(&white_to_move, &network), flipped_acc.evaluate(&color_flipped, &network));
    }

    #[test]
    fn incremental_float_tree_is_bit_exact_after_large_feature_removal() {
        let mut weights = vec![0.0f32; FEATURE_COUNT];
        weights[feature_index(3, 0, true).unwrap()] = 100_000_000.0; // White rook a1.
        weights[feature_index(6, 4, true).unwrap()] = 1.0; // White king e1.
        weights[feature_index(-6, 60, true).unwrap()] = 1.0; // Black king e8.
        let network = NnueNetwork::from_parts(weights, [0.0], [[1.0], [1.0]], 0.0, 1.0).unwrap();
        let other_network = network.clone();
        let separately_constructed = NnueNetwork::from_parts(
            network.feature_weights().to_vec(),
            *network.hidden_bias(),
            *network.output_weights(),
            network.output_bias(),
            network.output_scale(),
        ).unwrap();
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mut stack = NnueAccumulatorStack::new(&board, &network);
        assert_eq!(stack.current().values()[0][0], 100_000_000.0);
        assert!(stack.current().evaluate(&board, &separately_constructed).is_err());
        assert!(stack.push_transition(&board, &board, &separately_constructed).is_err());
        assert!(stack.current().evaluate(&board, &other_network).is_ok());

        let undo = stack.make_uci_move(&mut board, &network, "a1a2").unwrap();
        assert_eq!(stack.current().values()[0][0], 2.0);
        assert!(stack.current().bitwise_eq(&NnueAccumulator::full_recompute(&board, &network)));
        stack.unmake_move(&mut board, undo).unwrap();
        assert!(stack.current().bitwise_eq(&NnueAccumulator::full_recompute(&board, &network)));
    }

    #[test]
    fn accumulator_stack_rejects_out_of_sync_boards_and_root_pop() {
        let network = deterministic_network::<8>();
        let board = Board::startpos();
        let mut stack = NnueAccumulatorStack::new(&board, &network);
        assert!(stack.pop(&board).is_err());
        let other = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(stack.push_transition(&other, &board, &network).is_err());
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn accumulator_stack_rejects_hash_equal_board_with_different_counters() {
        let network = deterministic_network::<8>();
        let mut child = Board::startpos();
        let mut stack = NnueAccumulatorStack::new(&child, &network);
        let undo = stack.make_uci_move(&mut child, &network, "e2e4").unwrap();

        let child_fen = child.to_fen();
        let mut fields: Vec<&str> = child_fen.split_whitespace().collect();
        fields[5] = "42";
        let mut substituted = Board::from_fen(&fields.join(" ")).unwrap();
        assert_eq!(substituted.hash(), child.hash());
        assert_ne!(substituted, child);
        let unchanged_fen = substituted.to_fen();
        assert!(stack.unmake_move(&mut substituted, undo).is_err());
        assert_eq!(substituted.to_fen(), unchanged_fen);
        assert_eq!(stack.depth(), 2);
    }
}
