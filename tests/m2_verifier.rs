//! Independent, specification-driven M2 NNUE verification.
//!
//! This integration test intentionally uses only the public Board and NNUE APIs.
//! It does not inspect engine internals. The million-transition walk uses a fixed
//! seed and writes that seed and the failing transition into assertion messages.

use kira1a::Board;
use kira1a::eval::nnue::{
    FEATURE_COUNT, NnueAccumulator, NnueAccumulatorStack, NnueNetwork,
};

const H: usize = 128;
const WALK_SEED: u64 = 0x4b49_5241_4d32_2026;
const REQUIRED_TRANSITIONS: usize = 1_000_002;

fn verifier_network<const WIDTH: usize>() -> NnueNetwork<WIDTH> {
    assert_eq!(FEATURE_COUNT, 768, "M2 v1 feature count changed");
    let mut feature_weights = vec![0.0f32; FEATURE_COUNT * WIDTH];

    // Small dyadic values make every accumulator sum exactly representable in
    // f32. That lets bitwise comparison test incremental bookkeeping without
    // introducing order-dependent floating-point rounding noise.
    for feature in 0..FEATURE_COUNT {
        for hidden in 0..WIDTH {
            let mixed = (feature as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add((hidden as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9))
                .rotate_left(((feature + hidden) % 63) as u32);
            let numerator = (mixed % 15) as i32 - 7;
            feature_weights[feature * WIDTH + hidden] = numerator as f32 / 64.0;
        }
    }
    let hidden_bias = std::array::from_fn(|hidden| {
        ((hidden as i32 % 7) - 3) as f32 / 64.0
    });
    let output_weights = [
        std::array::from_fn(|hidden| ((hidden as i32 % 5) - 2) as f32 / 32.0),
        std::array::from_fn(|hidden| ((hidden as i32 % 7) - 3) as f32 / 32.0),
    ];
    NnueNetwork::from_parts(feature_weights, hidden_bias, output_weights, 0.0, 400.0)
        .expect("spec-sized verifier network should be accepted")
}

#[derive(Clone, Copy)]
struct ReplayRng(u64);

impl ReplayRng {
    fn next_u64(&mut self) -> u64 {
        // xorshift64*: deterministic, dependency-free, and sufficient for
        // choosing uniformly from each finite legal-move list.
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn index(&mut self, len: usize) -> usize {
        assert!(len > 0);
        let bound = len as u64;
        let reject_below = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= reject_below {
                return (value % bound) as usize;
            }
        }
    }
}

fn assert_same_accumulator<const WIDTH: usize>(
    actual: &NnueAccumulator<WIDTH>,
    expected: &NnueAccumulator<WIDTH>,
    context: &str,
) {
    assert!(actual.bitwise_eq(expected), "{context}: accumulator differs from full recompute");
}

fn verify_special_move<const WIDTH: usize>(
    label: &str,
    fen: &str,
    uci: &str,
    network: &NnueNetwork<WIDTH>,
) {
    let mut board = Board::from_fen(fen).unwrap_or_else(|err| panic!("{label}: invalid fixture: {err}"));
    let before_fen = board.to_fen();
    let before_hash = board.hash();
    let root_acc = NnueAccumulator::<WIDTH>::full_recompute(&board, network);
    let mut stack = NnueAccumulatorStack::new(&board, network);
    assert_same_accumulator(stack.current(), &root_acc, &format!("{label}: root"));

    let legal = board.perft_divide(1);
    assert_eq!(board.to_fen(), before_fen, "{label}: perft probe changed the board");
    assert_eq!(board.hash(), before_hash, "{label}: perft probe changed the hash");
    assert!(
        legal.iter().any(|(candidate, _)| candidate == uci),
        "{label}: expected special move {uci} absent from legal divide: {legal:?}"
    );

    let undo = stack
        .make_uci_move(&mut board, network, uci)
        .unwrap_or_else(|err| panic!("{label}: make {uci} failed: {err}"));
    let after = NnueAccumulator::<WIDTH>::full_recompute(&board, network);
    assert_same_accumulator(stack.current(), &after, &format!("{label}: after {uci}"));
    assert_ne!(
        stack.current().values(),
        root_acc.values(),
        "{label}: fixture did not change NNUE features"
    );

    stack
        .unmake_move(&mut board, undo)
        .unwrap_or_else(|err| panic!("{label}: unmake {uci} failed: {err}"));
    assert_eq!(board.to_fen(), before_fen, "{label}: FEN did not round-trip");
    assert_eq!(board.hash(), before_hash, "{label}: hash did not round-trip");
    let restored = NnueAccumulator::<WIDTH>::full_recompute(&board, network);
    assert_same_accumulator(stack.current(), &restored, &format!("{label}: after unmake"));
    assert_same_accumulator(stack.current(), &root_acc, &format!("{label}: root restoration"));
}

#[test]
fn m2_special_move_paths_match_full_recompute() {
    let network = verifier_network::<H>();

    verify_special_move(
        "ordinary capture",
        "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
        "e4d5",
        &network,
    );
    verify_special_move(
        "king-side castle",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "e1g1",
        &network,
    );
    verify_special_move(
        "queen-side castle",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "e1c1",
        &network,
    );
    verify_special_move(
        "en-passant capture",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "e5d6",
        &network,
    );
    verify_special_move(
        "queen promotion",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "a7a8q",
        &network,
    );
    verify_special_move(
        "underpromotion",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "a7a8n",
        &network,
    );
    verify_special_move(
        "Black king-side castle",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "e8g8",
        &network,
    );
    verify_special_move(
        "Black queen-side castle",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "e8c8",
        &network,
    );
    verify_special_move(
        "Black en-passant capture",
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
        "e4d3",
        &network,
    );
    verify_special_move(
        "Black queen promotion",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "a2a1q",
        &network,
    );
    verify_special_move(
        "Black underpromotion",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "a2a1n",
        &network,
    );
}

#[test]
fn m2_seeded_random_legal_walk_checks_every_make_and_unmake() {
    let network = verifier_network::<H>();
    let mut rng = ReplayRng(WALK_SEED);
    let mut transitions = 0usize;
    let mut make_count = 0usize;
    let mut unmake_count = 0usize;

    // Each branch starts from startpos, follows a seeded uniformly-selected
    // legal path of up to eight plies, and then unwinds every move. The full
    // recompute oracle is checked after every individual transition.
    while transitions < REQUIRED_TRANSITIONS {
        let mut board = Board::startpos();
        let mut stack = NnueAccumulatorStack::new(&board, &network);
        let root_fen = board.to_fen();
        let root_hash = board.hash();
        let root_acc = NnueAccumulator::<H>::full_recompute(&board, &network);
        assert_same_accumulator(stack.current(), &root_acc, "seeded walk: branch root");

        let branch_len = 1 + rng.index(8);
        let mut undo_stack = Vec::with_capacity(branch_len);
        for ply in 0..branch_len {
            let before_fen = board.to_fen();
            let before_hash = board.hash();
            let legal = board.perft_divide(1);
            assert_eq!(board.to_fen(), before_fen, "seed {WALK_SEED:#x}: perft altered FEN");
            assert_eq!(board.hash(), before_hash, "seed {WALK_SEED:#x}: perft altered hash");
            if legal.is_empty() {
                break;
            }
            let uci = &legal[rng.index(legal.len())].0;
            let undo = stack
                .make_uci_move(&mut board, &network, uci)
                .unwrap_or_else(|err| {
                    panic!(
                        "seed {WALK_SEED:#x}, transition {transitions}, ply {ply}, move {uci}, FEN {before_fen}: {err}"
                    )
                });
            transitions += 1;
            make_count += 1;
            let full = NnueAccumulator::<H>::full_recompute(&board, &network);
            assert_same_accumulator(
                stack.current(),
                &full,
                &format!(
                    "seed {WALK_SEED:#x}, make transition {transitions}, ply {ply}, move {uci}, FEN {}",
                    board.to_fen()
                ),
            );
            undo_stack.push((undo, before_fen, before_hash, uci.clone()));
        }

        while let Some((undo, prior_fen, prior_hash, uci)) = undo_stack.pop() {
            stack
                .unmake_move(&mut board, undo)
                .unwrap_or_else(|err| {
                    panic!(
                        "seed {WALK_SEED:#x}, transition {transitions}, unmake {uci}, expected FEN {prior_fen}: {err}"
                    )
                });
            transitions += 1;
            unmake_count += 1;
            assert_eq!(board.to_fen(), prior_fen, "seed {WALK_SEED:#x}: unmake FEN mismatch");
            assert_eq!(board.hash(), prior_hash, "seed {WALK_SEED:#x}: unmake hash mismatch");
            let full = NnueAccumulator::<H>::full_recompute(&board, &network);
            assert_same_accumulator(
                stack.current(),
                &full,
                &format!(
                    "seed {WALK_SEED:#x}, unmake transition {transitions}, move {uci}, FEN {}",
                    board.to_fen()
                ),
            );
        }
        assert_eq!(board.to_fen(), root_fen, "seed {WALK_SEED:#x}: branch did not return to root");
        assert_eq!(board.hash(), root_hash, "seed {WALK_SEED:#x}: branch hash did not return to root");
        assert_same_accumulator(stack.current(), &root_acc, "seeded walk: branch restored");
    }

    assert!(transitions > 1_000_000, "only {transitions} transitions checked");
    assert_eq!(make_count, unmake_count, "walk must pair every make with an unmake");
    assert_eq!(transitions, make_count + unmake_count);
    eprintln!(
        "M2 walk complete: seed={WALK_SEED:#x}, transitions={transitions}, makes={make_count}, unmakes={unmake_count}"
    );
}

#[test]
fn m2_undo_rejects_same_zobrist_position_with_different_fullmove_counter() {
    let network = verifier_network::<H>();
    let mut board = Board::startpos();
    let mut stack = NnueAccumulatorStack::new(&board, &network);
    let undo = stack
        .make_uci_move(&mut board, &network, "e2e4")
        .expect("legal start-position move");

    let child_fen = board.to_fen();
    let mut fields: Vec<&str> = child_fen.split_whitespace().collect();
    fields[5] = "42";
    let mut substituted = Board::from_fen(&fields.join(" ")).expect("valid altered counter FEN");
    assert_eq!(substituted.hash(), board.hash(), "move counters are outside the Zobrist key");
    assert_ne!(substituted.to_fen(), child_fen);
    let unchanged = substituted.to_fen();
    let expected = NnueAccumulator::<H>::full_recompute(&board, &network);
    assert!(stack.unmake_move(&mut substituted, undo).is_err());
    assert_eq!(substituted.to_fen(), unchanged, "rejected undo must not mutate the board");
    assert_eq!(stack.depth(), 2, "rejected undo must retain the valid frame");
    assert_same_accumulator(
        stack.current(),
        &expected,
        "rejected undo must retain the original child's accumulator",
    );
}

#[test]
fn m2_color_mirror_swaps_perspective_accumulators_and_preserves_score() {
    let network = verifier_network::<H>();
    let white_to_move = Board::from_fen("4k3/8/2n5/3p4/4P3/2N5/8/4K3 w - - 0 1")
        .expect("white-view fixture should parse");
    let mirrored_black_to_move =
        Board::from_fen("4k3/8/2n5/4p3/3P4/2N5/8/4K3 b - - 0 1")
            .expect("mirrored fixture should parse");

    let a = NnueAccumulator::<H>::full_recompute(&white_to_move, &network);
    let b = NnueAccumulator::<H>::full_recompute(&mirrored_black_to_move, &network);
    for hidden in 0..H {
        assert_eq!(
            a.values()[0][hidden].to_bits(),
            b.values()[1][hidden].to_bits(),
            "White-view / mirrored Black-view mismatch at hidden {hidden}"
        );
        assert_eq!(
            a.values()[1][hidden].to_bits(),
            b.values()[0][hidden].to_bits(),
            "Black-view / mirrored White-view mismatch at hidden {hidden}"
        );
    }
    assert_eq!(
        a.evaluate(&white_to_move, &network).expect("score A"),
        b.evaluate(&mirrored_black_to_move, &network).expect("score B"),
        "color-flipped equivalent positions must score identically"
    );
}

fn tiny_hand_net_with(
    output_weights: [[f32; 2]; 2],
    output_bias: f32,
) -> NnueNetwork<2> {
    assert_eq!(FEATURE_COUNT, 768);
    let mut feature_weights = vec![0.0f32; FEATURE_COUNT * 2];

    // Hand-derived from docs/CONVENTIONS.md for:
    //   4k3/8/8/8/8/8/P7/4K3 w - - 0 1
    // a1=0, files first, ranks by eight; index = rel*384+(kind-1)*64+viewed_sq.
    // White view: friendly K e1=324, friendly P a2=8, enemy K e8=764.
    // Black view: friendly K e8->e1=324, enemy K e1->e8=764,
    //             enemy P a2->a7=432.
    // Each flattened row is [hidden0, hidden1].
    feature_weights[324 * 2] = 0.25;
    feature_weights[764 * 2] = 0.25;
    feature_weights[8 * 2 + 1] = 0.5;
    feature_weights[432 * 2 + 1] = 0.25;

    NnueNetwork::from_parts(
        feature_weights,
        [0.125, 0.25],
        output_weights,
        output_bias,
        400.0,
    )
    .expect("tiny verifier network should be accepted")
}

fn tiny_hand_net() -> NnueNetwork<2> {
    tiny_hand_net_with([[0.5, 0.25], [0.125, 0.75]], -0.125)
}

#[test]
fn m2_tiny_net_matches_hand_calculated_features_and_stm_first_scores() {
    let network = tiny_hand_net();
    let white = Board::from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
        .expect("tiny-net white fixture should parse");
    let black = Board::from_fen("4k3/8/8/8/8/8/P7/4K3 b - - 0 1")
        .expect("tiny-net black fixture should parse");

    let white_acc = NnueAccumulator::<2>::full_recompute(&white, &network);
    let black_acc = NnueAccumulator::<2>::full_recompute(&black, &network);
    assert_eq!(white_acc.values()[0], [0.625, 0.75]);
    assert_eq!(white_acc.values()[1], [0.625, 0.5]);
    assert_eq!(black_acc.values()[0], [0.625, 0.75]);
    assert_eq!(black_acc.values()[1], [0.625, 0.5]);

    // White STM: (0.625*0.5 + 0.75*0.25) +
    //            (0.625*0.125 + 0.5*0.75) - 0.125 = 0.828125;
    // 400 * 0.828125 = 331.25, rounds to 331.
    assert_eq!(white_acc.evaluate(&white, &network).expect("white score"), 331);
    // Black STM swaps the accumulator order:
    // (0.625*0.5 + 0.5*0.25) +
    // (0.625*0.125 + 0.75*0.75) - 0.125 = 0.953125;
    // 400 * 0.953125 = 381.25, rounds to 381.
    assert_eq!(black_acc.evaluate(&black, &network).expect("black score"), 381);
}

#[test]
fn m2_tiny_net_rounds_half_centipawns_away_from_zero() {
    let white = Board::from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
        .expect("tiny-net white fixture should parse");
    let positive_tie =
        tiny_hand_net_with([[0.5, 0.25], [0.125, 0.75]], -0.109375);
    let negative_tie =
        tiny_hand_net_with([[-0.5, -0.25], [-0.125, -0.75]], 0.109375);

    // The hand-computed pre-scale values are +0.84375 and -0.84375, giving
    // +337.5 cp and -337.5 cp. CONVENTIONS.md requires midpoint ties away from 0.
    assert_eq!(
        NnueAccumulator::<2>::full_recompute(&white, &positive_tie)
            .evaluate(&white, &positive_tie)
            .expect("positive midpoint score"),
        338
    );
    assert_eq!(
        NnueAccumulator::<2>::full_recompute(&white, &negative_tie)
            .evaluate(&white, &negative_tie)
            .expect("negative midpoint score"),
        -338
    );
}

#[test]
fn m2_tiny_net_clipped_relu_clamps_both_ends() {
    let board = Board::startpos();
    let network = NnueNetwork::<2>::from_parts(
        vec![0.0; FEATURE_COUNT * 2],
        [2.0, -2.0],
        [[0.5, -1.0], [-0.25, 0.75]],
        0.25,
        400.0,
    )
    .expect("clipping probe network should be accepted");
    let acc = NnueAccumulator::<2>::full_recompute(&board, &network);

    // values() exposes the pre-activation sums; evaluate applies clipped ReLU.
    assert_eq!(acc.values()[0], [2.0, -2.0]);
    assert_eq!(acc.values()[1], [2.0, -2.0]);
    // 0.25 + (1.0*0.5 + 0.0*-1.0) + (1.0*-0.25 + 0.0*0.75) = 0.5.
    assert_eq!(acc.evaluate(&board, &network).expect("clipped score"), 200);
}

#[test]
fn m2_adversarial_large_finite_delta_recomputes_balanced_tree_paths() {
    let mut feature_weights = vec![0.0f32; FEATURE_COUNT];
    // Rook, friendly, a1 (square 0): (3-1)*64 + 0 = 128.
    // Friendly king e1 and enemy king e8 in White view: 324 and 764.
    feature_weights[128] = 100_000_000.0;
    feature_weights[324] = 1.0;
    feature_weights[764] = 1.0;
    let network = NnueNetwork::<1>::from_parts(
        feature_weights,
        [0.0],
        [[1.0], [1.0]],
        0.0,
        400.0,
    )
    .expect("finite cancellation-probe network should be accepted");
    let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        .expect("rook cancellation fixture should parse");
    let root_fen = board.to_fen();
    let root_hash = board.hash();
    let root_acc = NnueAccumulator::<1>::full_recompute(&board, &network);
    let mut stack = NnueAccumulatorStack::new(&board, &network);
    assert_same_accumulator(stack.current(), &root_acc, "large-delta root");

    let legal = board.perft_divide(1);
    assert!(legal.iter().any(|(uci, _)| uci == "a1a2"), "a1a2 must be legal: {legal:?}");
    let undo = stack
        .make_uci_move(&mut board, &network, "a1a2")
        .expect("rook a1a2 should update the accumulator");
    let after = NnueAccumulator::<1>::full_recompute(&board, &network);

    // In the specified 64-square balanced reduction, the original 1e8 term
    // swallows the two +1 king leaves. After Ra1-a2 the rook has weight zero;
    // the full tree sums both king leaves to 2 in the pre-activation vectors.
    // Evaluation then clips each 2 to 1.
    // A flat delta from 1e8 that removes 1e8 loses the previously swallowed
    // king contributions and yields 0, so exact comparison catches cancellation.
    assert_eq!(after.values()[0][0], 2.0);
    assert_eq!(after.values()[1][0], 2.0);
    assert_same_accumulator(stack.current(), &after, "large-delta after a1a2");
    assert_eq!(after.evaluate(&board, &network).expect("large-delta score"), 800);

    stack
        .unmake_move(&mut board, undo)
        .expect("rook unmake should restore the accumulator");
    assert_eq!(board.to_fen(), root_fen);
    assert_eq!(board.hash(), root_hash);
    let restored = NnueAccumulator::<1>::full_recompute(&board, &network);
    assert_same_accumulator(stack.current(), &restored, "large-delta after unmake");
    assert_same_accumulator(stack.current(), &root_acc, "large-delta root restored");
}

#[test]
fn m2_stack_rejects_a_distinct_network_but_accepts_its_clone() {
    let network = verifier_network::<H>();
    let clone = network.clone();
    // Construct the same parameter values in a distinct network instance. The
    // stack must key compatibility to network identity, not just equal weights.
    let distinct = verifier_network::<H>();
    let mut board = Board::startpos();
    let root_fen = board.to_fen();
    let root_hash = board.hash();
    let root = NnueAccumulator::<H>::full_recompute(&board, &network);
    let mut stack = NnueAccumulatorStack::new(&board, &network);
    let legal = board.perft_divide(1);
    let uci = &legal[0].0;

    assert!(
        stack.make_uci_move(&mut board, &distinct, uci).is_err(),
        "a distinct network instance must not be used with this accumulator stack"
    );
    assert_eq!(board.to_fen(), root_fen, "rejected network changed the board FEN");
    assert_eq!(board.hash(), root_hash, "rejected network changed the board hash");
    assert_same_accumulator(stack.current(), &root, "rejected network left accumulator unchanged");

    let undo = stack
        .make_uci_move(&mut board, &clone, uci)
        .expect("a clone of the bound network should remain compatible");
    let after = NnueAccumulator::<H>::full_recompute(&board, &clone);
    assert_same_accumulator(stack.current(), &after, "cloned network make");
    stack
        .unmake_move(&mut board, undo)
        .expect("unmake with cloned network token should succeed");
    assert_eq!(board.to_fen(), root_fen);
    assert_eq!(board.hash(), root_hash);
    assert_same_accumulator(stack.current(), &root, "cloned network unmake");
}
