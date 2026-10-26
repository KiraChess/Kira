use crate::Board;

pub mod nnue;
pub mod nnue_quantized;

pub trait Evaluator: Send + Sync {
    /// Return a centipawn score from the side-to-move's point of view.
    fn evaluate(&self, board: &Board) -> i32;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Hce;

impl Evaluator for Hce {
    fn evaluate(&self, board: &Board) -> i32 {
        let (mut middle, mut ending, mut phase) = (0, 0, 0);
        for (square, &piece) in board.squares.iter().enumerate() {
            if piece == 0 {
                continue;
            }
            let color = piece.signum() as i32;
            let kind = piece.unsigned_abs() as usize;
            let oriented_square = if piece > 0 { square } else { square ^ 56 };
            let file = (oriented_square & 7) as i32;
            let rank = (oriented_square >> 3) as i32;
            let center = 14 - ((file * 2 - 7).abs() + (rank * 2 - 7).abs());
            let (mg_bonus, eg_bonus) = match kind {
                1 => (rank * 7 + center, rank * 11 + center * 2),
                2 => (center * 5, center * 3),
                3 => (rank * 2 + if rank == 6 { 12 } else { 0 }, rank * 3),
                4 => (center * 2, center * 2),
                5 => (center, center * 2),
                6 => (-center * 5, center * 5),
                _ => (0, 0),
            };
            let base = match kind {
                1 => 100,
                2 => 320,
                3 => 500,
                4 => 335,
                5 => 900,
                _ => 0,
            };
            middle += color * (base + mg_bonus);
            ending += color * (base + eg_bonus);
            phase += match kind {
                2 | 4 => 1,
                3 => 2,
                5 => 4,
                _ => 0,
            };
        }
        let phase = phase.min(24);
        let white_score = (middle * phase + ending * (24 - phase)) / 24;
        if board.white_to_move { white_score } else { -white_score }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Board, START_FEN};

    #[test]
    fn hce_uses_side_to_move_perspective_and_material() {
        let hce = Hce;
        let start = Board::startpos();
        assert_eq!(hce.evaluate(&start), 0);

        let black_to_move = Board::from_fen(&START_FEN.replace(" w ", " b ")).unwrap();
        assert_eq!(hce.evaluate(&black_to_move), 0);

        let queen_up = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let same_position_black = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1").unwrap();
        assert!(hce.evaluate(&queen_up) >= 600);
        assert_eq!(hce.evaluate(&queen_up), -hce.evaluate(&same_position_black));
    }

    #[test]
    fn hce_is_color_mirror_symmetric() {
        let white = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let black_mirror = Board::from_fen("3qk3/8/8/8/8/8/8/4K3 b - - 0 1").unwrap();
        assert_eq!(Hce.evaluate(&white), Hce.evaluate(&black_mirror));
    }
}
