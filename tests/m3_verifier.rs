//! Independent, specification-derived M3 verifier tests.
//!
//! These tests use only the public board and datagen APIs. The generation /
//! artifact integration tests are kept in this file but are deliberately not
//! run until the worker-process implementation is pinned by the operator.

use kira1a::datagen::{
    self, BestMoveClass, FilterReason, NaturalReason, NaturalResult, PostMoveOutcome,
    ScoreDecision, ScoreStreaks,
};
use kira1a::search::MATE_SCORE;
use kira1a::{Board, Move};

#[test]
fn material_fixtures_round_trip_and_match_selected_subset() {
    assert_eq!(datagen::MATERIAL_RULE_FIXTURES.len(), 24);
    for (fen, expected) in datagen::MATERIAL_RULE_FIXTURES {
        let board = Board::from_fen(fen).unwrap_or_else(|e| panic!("fixture did not parse: {fen}: {e}"));
        assert_eq!(board.to_fen(), fen, "FEN round-trip changed fixture {fen}");
        assert_eq!(datagen::has_basic_insufficient_material(&board), expected, "fixture {fen}");
    }

    for pair in datagen::MATERIAL_RULE_FIXTURES.chunks_exact(2) {
        let (white_fen, white_result) = pair[0];
        let (black_fen, black_result) = pair[1];
        assert_eq!(white_result, black_result, "paired material fixture result differs");
        let mut white_fields = white_fen.split_whitespace();
        let mut black_fields = black_fen.split_whitespace();
        assert_eq!(white_fields.next(), black_fields.next());
        assert_eq!(white_fields.next(), Some("w"));
        assert_eq!(black_fields.next(), Some("b"));
        assert_eq!(white_fields.next(), black_fields.next());
        assert_eq!(white_fields.next(), black_fields.next());
        assert_eq!(white_fields.next(), black_fields.next());
        assert_eq!(white_fields.next(), black_fields.next());
        assert_eq!(white_fields.next(), None);
        assert_eq!(black_fields.next(), None);
    }
}

#[test]
fn natural_terminal_precedence_and_replayable_history() {
    let mut mate = Board::from_fen(datagen::CHECKMATE_RULE50_PRECEDENCE_FEN).unwrap();
    assert_eq!(mate.to_fen(), datagen::CHECKMATE_RULE50_PRECEDENCE_FEN);
    assert!(mate.is_in_check(mate.white_to_move));
    assert_eq!(mate.perft(1), 0);
    let mate_hash = mate.hash();
    assert_eq!(
        datagen::classify_natural_terminal(&mut mate, &[mate_hash]),
        Some(datagen::NaturalOutcome {
            result: NaturalResult::WhiteWin,
            reason: NaturalReason::Checkmate,
        })
    );

    let mut stale = Board::from_fen(datagen::STALEMATE_MATERIAL_PRECEDENCE_FEN).unwrap();
    assert_eq!(stale.to_fen(), datagen::STALEMATE_MATERIAL_PRECEDENCE_FEN);
    assert!(!stale.is_in_check(stale.white_to_move));
    assert_eq!(stale.perft(1), 0);
    assert!(datagen::has_basic_insufficient_material(&stale));
    let stale_hash = stale.hash();
    assert_eq!(
        datagen::classify_natural_terminal(&mut stale, &[stale_hash]),
        Some(datagen::NaturalOutcome {
            result: NaturalResult::Draw,
            reason: NaturalReason::Stalemate,
        })
    );

    let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"];
    let mut board = Board::startpos();
    let mut history = vec![board.hash()];
    for (ply, mv) in cycle.iter().chain(cycle.iter()).enumerate() {
        board.play_uci(mv).unwrap_or_else(|e| panic!("cycle move {mv} failed: {e}"));
        history.push(board.hash());
        let outcome = datagen::classify_natural_terminal(&mut board, &history);
        if ply == 3 {
            assert_eq!(outcome, None, "second occurrence is not a threefold draw");
        }
        if ply == 7 {
            assert_eq!(
                outcome,
                Some(datagen::NaturalOutcome {
                    result: NaturalResult::Draw,
                    reason: NaturalReason::Threefold,
                }),
                "third occurrence must be detected from replayed history"
            );
        }
    }

    let mut at_99 = Board::from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 99 1").unwrap();
    let before_99 = at_99.clone();
    let before_99_hash = at_99.hash();
    assert_eq!(datagen::classify_natural_terminal(&mut at_99, &[before_99_hash]), None);
    at_99.play_uci("d1d2").unwrap();
    assert_eq!(at_99.to_fen(), "4k3/8/8/8/8/8/3R4/4K3 b - - 100 1");
    let after_99_hash = at_99.hash();
    assert_eq!(
        datagen::classify_natural_terminal(&mut at_99, &[before_99.hash(), after_99_hash]),
        Some(datagen::NaturalOutcome {
            result: NaturalResult::Draw,
            reason: NaturalReason::FiftyMove,
        })
    );

    // The selected material subset is lower precedence than the rule-50 rule.
    let mut overlap = Board::from_fen("4k3/8/8/8/8/8/8/2B1K3 w - - 100 1").unwrap();
    let overlap_hash = overlap.hash();
    assert_eq!(
        datagen::classify_natural_terminal(&mut overlap, &[overlap_hash]),
        Some(datagen::NaturalOutcome {
            result: NaturalResult::Draw,
            reason: NaturalReason::FiftyMove,
        })
    );
}

#[test]
fn score_streak_boundaries_use_fixed_white_perspective() {
    let fixtures = [
        (true, 1_500, ScoreDecision::WhiteWin),
        (false, -1_500, ScoreDecision::WhiteWin),
        (true, -1_500, ScoreDecision::BlackWin),
        (false, 1_500, ScoreDecision::BlackWin),
    ];
    for (white_to_move, score_stm, expected) in fixtures {
        let mut streaks = ScoreStreaks::default();
        for ply in 0..4 {
            assert_eq!(datagen::observe_score(&mut streaks, ply, score_stm, white_to_move), None);
        }
        assert_eq!(datagen::observe_score(&mut streaks, 4, score_stm, white_to_move), Some(expected));

        // A fixed-side sign reversal starts a different winner's streak.
        let reversed = -score_stm;
        assert_eq!(datagen::observe_score(&mut streaks, 5, reversed, white_to_move), None);
    }

    let mut below = ScoreStreaks::default();
    for ply in 0..4 {
        assert_eq!(datagen::observe_score(&mut below, ply, 1_500, true), None);
    }
    assert_eq!(datagen::observe_score(&mut below, 4, 1_499, true), None);
    for ply in 5..9 {
        assert_eq!(datagen::observe_score(&mut below, ply, 1_500, true), None);
    }
    assert_eq!(datagen::observe_score(&mut below, 9, 1_500, true), Some(ScoreDecision::WhiteWin));

    let mut mate_reset = ScoreStreaks::default();
    for ply in 0..4 {
        assert_eq!(datagen::observe_score(&mut mate_reset, ply, 1_500, true), None);
    }
    assert_eq!(datagen::observe_score(&mut mate_reset, 4, MATE_SCORE, true), None);
    for ply in 5..9 {
        assert_eq!(datagen::observe_score(&mut mate_reset, ply, 1_500, true), None);
    }
    assert_eq!(datagen::observe_score(&mut mate_reset, 9, 1_500, true), Some(ScoreDecision::WhiteWin));

    assert_eq!(datagen::white_relative_score(1_500, true), 1_500);
    assert_eq!(datagen::white_relative_score(-1_500, false), 1_500);
    assert_eq!(datagen::white_relative_score(-1_500, true), -1_500);
    assert_eq!(datagen::white_relative_score(1_500, false), -1_500);
}

#[test]
fn score_draw_rule_counts_only_p120_through_p159_and_defers_terminal_order() {
    let mut streaks = ScoreStreaks::default();
    assert_eq!(datagen::observe_score(&mut streaks, 119, 0, true), None);
    for ply in 120..=158 {
        assert_eq!(datagen::observe_score(&mut streaks, ply, 0, ply % 2 == 0), None);
    }
    assert!(!datagen::score_draw_ready_after_move(&streaks, 159));
    assert_eq!(datagen::observe_score(&mut streaks, 159, -5, false), None);
    assert!(datagen::score_draw_ready_after_move(&streaks, 160));

    let mut outside = ScoreStreaks::default();
    for ply in 120..=158 {
        assert_eq!(datagen::observe_score(&mut outside, ply, 0, true), None);
    }
    assert_eq!(datagen::observe_score(&mut outside, 159, 6, false), None);
    assert!(!datagen::score_draw_ready_after_move(&outside, 160));

    let mut terminal = Board::from_fen(datagen::CHECKMATE_RULE50_PRECEDENCE_FEN).unwrap();
    let terminal_history = [terminal.hash()];
    assert_eq!(
        datagen::adjudicate_after_move(
            &mut terminal,
            &terminal_history,
            Some(ScoreDecision::BlackWin),
            &streaks,
            160,
        ),
        Some(PostMoveOutcome::Natural(datagen::NaturalOutcome {
            result: NaturalResult::WhiteWin,
            reason: NaturalReason::Checkmate,
        }))
    );

    let mut live = Board::startpos();
    let live_history = [live.hash()];
    assert_eq!(
        datagen::adjudicate_after_move(&mut live, &live_history, None, &streaks, 160),
        Some(PostMoveOutcome::ScoreDraw)
    );

    // Local, replayable move-80 fixture: Black's legal g3g2 checkmate occurs
    // after the fortieth in-band observation, and natural terminal precedence
    // must override the now-ready score draw.
    let mut before_black_move_80 =
        Board::from_fen("8/8/8/8/8/6q1/5k2/7K b - - 0 80").unwrap();
    assert!(!before_black_move_80.white_to_move);
    assert!(before_black_move_80.perft(1) > 0);
    let before_hash = before_black_move_80.hash();
    before_black_move_80.play_uci("g3g2").unwrap();
    assert!(before_black_move_80.white_to_move);
    assert!(before_black_move_80.is_in_check(true));
    assert_eq!(before_black_move_80.perft(1), 0);
    let after_hash = before_black_move_80.hash();
    assert_eq!(
        datagen::adjudicate_after_move(
            &mut before_black_move_80,
            &[before_hash, after_hash],
            None,
            &streaks,
            160,
        ),
        Some(PostMoveOutcome::Natural(datagen::NaturalOutcome {
            result: NaturalResult::BlackWin,
            reason: NaturalReason::Checkmate,
        }))
    );
}

#[test]
fn opening_seed_rng_and_worker_boundaries_are_deterministic() {
    assert!(datagen::opening_score_accepted(-300));
    assert!(datagen::opening_score_accepted(300));
    assert!(!datagen::opening_score_accepted(-301));
    assert!(!datagen::opening_score_accepted(301));

    assert_eq!(datagen::derive_game_seed(0, 0), 0);
    assert_eq!(datagen::derive_game_seed(0, 1), 6_238_072_747_940_578_789);
    let mut rng = datagen::SplitMix64::seeded(0);
    assert_eq!(rng.next_u64(), 16_294_208_416_658_607_535);
    assert_eq!(rng.next_u64(), 7_960_286_522_194_355_700);

    // For bound 10 the rejection threshold is 6: 0..5 reject; 18 is accepted
    // and maps to index 8. This checks the rejection path, not modulo bias by
    // statistical sampling.
    assert_eq!(datagen::uniform_index_from_words(10, [0, 1, 5, 18]).unwrap(), 8);
    assert!(datagen::uniform_index_from_words(10, [0, 1, 5]).is_err());
    assert!(datagen::uniform_index_from_words(0, [1]).is_err());
    assert_eq!(datagen::worker_for_game(17, 4).unwrap(), 1);
    assert!(datagen::worker_for_game(17, 0).is_err());
}

#[test]
fn filter_and_best_move_classification_boundaries() {
    let quiet = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(
        datagen::classify_best_move(&quiet, Move::from_uci("e1d1").unwrap()),
        BestMoveClass::Quiet
    );

    let capture = Board::from_fen("4k3/8/8/8/8/8/p7/R3K3 w - - 0 1").unwrap();
    let mut capture_replay = capture.clone();
    capture_replay.play_uci("a1a2").unwrap();
    assert_eq!(
        datagen::classify_best_move(&capture, Move::from_uci("a1a2").unwrap()),
        BestMoveClass::Capture
    );

    let ep = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    let mut ep_replay = ep.clone();
    ep_replay.play_uci("e5d6").unwrap();
    assert_eq!(
        datagen::classify_best_move(&ep, Move::from_uci("e5d6").unwrap()),
        BestMoveClass::Capture,
        "en passant must be treated as a capture"
    );

    let promotion = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let mut promotion_replay = promotion.clone();
    promotion_replay.play_uci("a7a8q").unwrap();
    assert_eq!(
        datagen::classify_best_move(&promotion, Move::from_uci("a7a8q").unwrap()),
        BestMoveClass::Promotion
    );

    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::Quiet, 2_000), FilterReason::Kept);
    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::Quiet, -2_000), FilterReason::Kept);
    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::Quiet, 2_001), FilterReason::ScoreOutsideRange);
    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::Quiet, -2_001), FilterReason::ScoreOutsideRange);
    assert_eq!(datagen::candidate_filter_reason(true, BestMoveClass::Quiet, 0), FilterReason::InCheck);
    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::Capture, 0), FilterReason::BestMoveCapture);
    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::Promotion, 0), FilterReason::BestMovePromotion);
    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::CapturePromotion, 0), FilterReason::BestMoveCapture);
    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::Quiet, MATE_SCORE), FilterReason::MateScore);
    assert_eq!(datagen::candidate_filter_reason(false, BestMoveClass::Quiet, -MATE_SCORE), FilterReason::MateScore);

    let checked = Board::from_fen("4r1k1/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    assert!(checked.is_in_check(true));
    let mut checked_with_evasion = checked.clone();
    assert!(checked_with_evasion.perft(1) > 0, "check fixture must have a legal evasion");
}
