use std::sync::LazyLock;

pub mod eval;
pub mod search;
pub mod datagen;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const WHITE_KING_SIDE: u8 = 1;
const WHITE_QUEEN_SIDE: u8 = 2;
const BLACK_KING_SIDE: u8 = 4;
const BLACK_QUEEN_SIDE: u8 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promotion: i8,
}

impl Move {
    pub fn from_uci(text: &str) -> Option<Self> {
        parse_uci_move(text)
    }

    pub fn uci(self) -> String {
        let mut s = format!("{}{}", square_name(self.from), square_name(self.to));
        if self.promotion != 0 {
            s.push(piece_letter(self.promotion).to_ascii_lowercase());
        }
        s
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Undo {
    mover: i8,
    captured: i8,
    captured_square: u8,
    castling: u8,
    ep: Option<u8>,
    halfmove: u16,
    fullmove: u16,
    hash: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Board {
    pub(crate) squares: [i8; 64],
    pub white_to_move: bool,
    castling: u8,
    ep: Option<u8>,
    pub(crate) halfmove: u16,
    fullmove: u16,
    hash: u64,
}

struct Zobrist {
    pieces: [[u64; 64]; 12],
    side: u64,
    castling: [u64; 16],
    ep_file: [u64; 8],
}

static ZOBRIST: LazyLock<Zobrist> = LazyLock::new(|| {
    let mut seed = 0x4b49_5241_3141_4c50u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut pieces = [[0; 64]; 12];
    for piece in &mut pieces {
        for square in piece {
            *square = next();
        }
    }
    let side = next();
    let mut castling = [0; 16];
    for key in &mut castling {
        *key = next();
    }
    let mut ep_file = [0; 8];
    for key in &mut ep_file {
        *key = next();
    }
    Zobrist { pieces, side, castling, ep_file }
});

impl Board {
    pub fn startpos() -> Self {
        Self::from_fen(START_FEN).expect("valid start position")
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() < 4 || fields.len() > 6 {
            return Err("FEN must contain four to six fields".into());
        }
        let mut squares = [0i8; 64];
        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("FEN board must contain eight ranks".into());
        }
        for (fen_rank, row) in ranks.iter().enumerate() {
            let rank = 7 - fen_rank;
            let mut file = 0usize;
            for ch in row.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    if empty == 0 || empty > 8 {
                        return Err("invalid empty-square run in FEN".into());
                    }
                    file += empty as usize;
                } else {
                    if file >= 8 {
                        return Err("too many squares in FEN rank".into());
                    }
                    let piece = letter_piece(ch).ok_or_else(|| format!("invalid FEN piece: {ch}"))?;
                    squares[rank * 8 + file] = piece;
                    file += 1;
                }
                if file > 8 {
                    return Err("too many squares in FEN rank".into());
                }
            }
            if file != 8 {
                return Err("FEN rank does not contain eight squares".into());
            }
        }
        if squares.iter().filter(|&&p| p == 6).count() != 1
            || squares.iter().filter(|&&p| p == -6).count() != 1
        {
            return Err("FEN must contain exactly one king of each color".into());
        }
        let white_to_move = match fields[1] {
            "w" => true,
            "b" => false,
            _ => return Err("invalid FEN active color".into()),
        };
        let mut castling = 0;
        if fields[2] != "-" {
            for c in fields[2].chars() {
                let bit = match c {
                    'K' => WHITE_KING_SIDE,
                    'Q' => WHITE_QUEEN_SIDE,
                    'k' => BLACK_KING_SIDE,
                    'q' => BLACK_QUEEN_SIDE,
                    _ => return Err("invalid FEN castling rights".into()),
                };
                if castling & bit != 0 {
                    return Err("duplicate FEN castling right".into());
                }
                castling |= bit;
            }
        }
        let ep = if fields[3] == "-" {
            None
        } else {
            let sq = parse_square(fields[3]).ok_or_else(|| "invalid FEN en-passant square".to_string())?;
            if rank_of(sq) != if white_to_move { 5 } else { 2 } {
                return Err("invalid FEN en-passant rank".into());
            }
            if squares[sq as usize] != 0 {
                return Err("FEN en-passant target square must be empty".into());
            }
            let captured_square = if white_to_move { sq - 8 } else { sq + 8 };
            let expected_pawn = if white_to_move { -1 } else { 1 };
            if squares[captured_square as usize] != expected_pawn {
                return Err("FEN en-passant target must follow a double pawn push".into());
            }
            Some(sq)
        };
        let halfmove = if fields.len() >= 5 {
            fields[4].parse::<u16>().map_err(|_| "invalid FEN halfmove clock".to_string())?
        } else { 0 };
        let fullmove = if fields.len() >= 6 {
            fields[5].parse::<u16>().map_err(|_| "invalid FEN fullmove number".to_string())?
        } else { 1 };
        if fullmove == 0 {
            return Err("FEN fullmove number must be positive".into());
        }
        let mut board = Self { squares, white_to_move, castling, ep, halfmove, fullmove, hash: 0 };
        board.hash = board.recompute_hash();
        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        let mut out = String::new();
        for rank in (0..8).rev() {
            if rank != 7 { out.push('/'); }
            let mut empty = 0;
            for file in 0..8 {
                let piece = self.squares[rank * 8 + file];
                if piece == 0 {
                    empty += 1;
                } else {
                    if empty != 0 { out.push(char::from_digit(empty, 10).unwrap()); empty = 0; }
                    out.push(piece_letter(piece));
                }
            }
            if empty != 0 { out.push(char::from_digit(empty, 10).unwrap()); }
        }
        out.push_str(if self.white_to_move { " w " } else { " b " });
        let mut rights = String::new();
        for (bit, c) in [(WHITE_KING_SIDE, 'K'), (WHITE_QUEEN_SIDE, 'Q'), (BLACK_KING_SIDE, 'k'), (BLACK_QUEEN_SIDE, 'q')] {
            if self.castling & bit != 0 { rights.push(c); }
        }
        out.push_str(if rights.is_empty() { "-" } else { &rights });
        out.push(' ');
        out.push_str(self.ep.map(square_name).as_deref().unwrap_or("-"));
        out.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        out
    }

    pub fn hash(&self) -> u64 { self.hash }

    pub fn recompute_hash(&self) -> u64 {
        let mut hash = ZOBRIST.castling[self.castling as usize];
        if !self.white_to_move { hash ^= ZOBRIST.side; }
        if let Some(file) = self.ep_hash_file() { hash ^= ZOBRIST.ep_file[file as usize]; }
        for (square, &piece) in self.squares.iter().enumerate() {
            if piece != 0 { hash ^= ZOBRIST.pieces[piece_index(piece)][square]; }
        }
        hash
    }

    fn ep_hash_file(&self) -> Option<u8> {
        let target = self.ep?;
        if self.squares[target as usize] != 0 { return None; }
        let captured_square = if self.white_to_move { target - 8 } else { target + 8 };
        if self.squares[captured_square as usize] != if self.white_to_move { -1 } else { 1 } {
            return None;
        }
        let source_rank = rank_of(target) as i8 + if self.white_to_move { -1 } else { 1 };
        let target_file = file_of(target) as i8;
        let capturing_pawn = if self.white_to_move { 1 } else { -1 };
        for df in [-1, 1] {
            if let Some(source) = make_square(target_file + df, source_rank) {
                if self.squares[source as usize] == capturing_pawn {
                    // An en-passant right distinguishes positions for
                    // repetition only when the capture is actually legal.
                    // Removing both pawns can reveal a rook/bishop attack on
                    // the moving side's king when the capturer was pinned.
                    let mut after_ep = self.clone();
                    after_ep.squares[source as usize] = 0;
                    after_ep.squares[captured_square as usize] = 0;
                    after_ep.squares[target as usize] = capturing_pawn;
                    if !after_ep.is_in_check(self.white_to_move) {
                        return Some(file_of(target));
                    }
                }
            }
        }
        None
    }

    pub fn is_in_check(&self, white: bool) -> bool {
        let king = if white { 6 } else { -6 };
        let Some(square) = self.squares.iter().position(|&p| p == king) else { return true; };
        self.is_attacked(square as u8, !white)
    }

    fn is_attacked(&self, target: u8, by_white: bool) -> bool {
        let file = file_of(target) as i8;
        let rank = rank_of(target) as i8;
        let pawn = if by_white { 1 } else { -1 };
        let source_rank = rank + if by_white { -1 } else { 1 };
        for df in [-1, 1] {
            let source_file = file + df;
            if let Some(source) = make_square(source_file, source_rank) {
                if self.squares[source as usize] == pawn { return true; }
            }
        }
        let knight = if by_white { 2 } else { -2 };
        for (df, dr) in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
            if let Some(source) = make_square(file + df, rank + dr) {
                if self.squares[source as usize] == knight { return true; }
            }
        }
        let king = if by_white { 6 } else { -6 };
        for df in -1..=1 {
            for dr in -1..=1 {
                if df == 0 && dr == 0 { continue; }
                if let Some(source) = make_square(file + df, rank + dr) {
                    if self.squares[source as usize] == king { return true; }
                }
            }
        }
        for (df, dr, diagonal) in [
            (1, 0, false), (-1, 0, false), (0, 1, false), (0, -1, false),
            (1, 1, true), (1, -1, true), (-1, 1, true), (-1, -1, true),
        ] {
            let mut f = file + df;
            let mut r = rank + dr;
            while let Some(source) = make_square(f, r) {
                let piece = self.squares[source as usize];
                if piece != 0 {
                    if (piece > 0) == by_white {
                        let kind = piece.abs();
                        if kind == 5 || (diagonal && kind == 4) || (!diagonal && kind == 3) {
                            return true;
                        }
                    }
                    break;
                }
                f += df;
                r += dr;
            }
        }
        false
    }

    fn pseudo_moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(64);
        let white = self.white_to_move;
        for from in 0..64u8 {
            let piece = self.squares[from as usize];
            if piece == 0 || (piece > 0) != white { continue; }
            let file = file_of(from) as i8;
            let rank = rank_of(from) as i8;
            match piece.abs() {
                1 => {
                    let direction = if white { 1 } else { -1 };
                    let start_rank = if white { 1 } else { 6 };
                    let promotion_rank = if white { 7 } else { 0 };
                    if let Some(to) = make_square(file, rank + direction) {
                        if self.squares[to as usize] == 0 {
                            self.push_pawn_move(&mut moves, from, to, promotion_rank);
                            if rank == start_rank {
                                if let Some(two) = make_square(file, rank + 2 * direction) {
                                    if self.squares[two as usize] == 0 { moves.push(Move { from, to: two, promotion: 0 }); }
                                }
                            }
                        }
                    }
                    for df in [-1, 1] {
                        if let Some(to) = make_square(file + df, rank + direction) {
                            let target = self.squares[to as usize];
                            if target != 0 && (target > 0) != white && target.abs() != 6 {
                                self.push_pawn_move(&mut moves, from, to, promotion_rank);
                            } else if self.ep == Some(to) {
                                let cap = (to as i16 - if white { 8 } else { -8 }) as usize;
                                if self.squares[cap] == if white { -1 } else { 1 } {
                                    moves.push(Move { from, to, promotion: 0 });
                                }
                            }
                        }
                    }
                }
                2 => {
                    for (df, dr) in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
                        if let Some(to) = make_square(file + df, rank + dr) { self.push_target(&mut moves, from, to, white); }
                    }
                }
                3 | 4 | 5 => {
                    for (df, dr, allowed) in [
                        (1, 0, piece.abs() != 4), (-1, 0, piece.abs() != 4), (0, 1, piece.abs() != 4), (0, -1, piece.abs() != 4),
                        (1, 1, piece.abs() != 3), (1, -1, piece.abs() != 3), (-1, 1, piece.abs() != 3), (-1, -1, piece.abs() != 3),
                    ] {
                        if !allowed { continue; }
                        let mut f = file + df;
                        let mut r = rank + dr;
                        while let Some(to) = make_square(f, r) {
                            let target = self.squares[to as usize];
                            if target == 0 { moves.push(Move { from, to, promotion: 0 }); }
                            else {
                                if (target > 0) != white && target.abs() != 6 { moves.push(Move { from, to, promotion: 0 }); }
                                break;
                            }
                            f += df;
                            r += dr;
                        }
                    }
                }
                6 => {
                    for df in -1..=1 {
                        for dr in -1..=1 {
                            if df == 0 && dr == 0 { continue; }
                            if let Some(to) = make_square(file + df, rank + dr) { self.push_target(&mut moves, from, to, white); }
                        }
                    }
                    if white && from == 4 && !self.is_attacked(4, false) {
                        if self.castling & WHITE_KING_SIDE != 0 && self.squares[5] == 0 && self.squares[6] == 0 && self.squares[7] == 3
                            && !self.is_attacked(5, false) { moves.push(Move { from, to: 6, promotion: 0 }); }
                        if self.castling & WHITE_QUEEN_SIDE != 0 && self.squares[1] == 0 && self.squares[2] == 0 && self.squares[3] == 0 && self.squares[0] == 3
                            && !self.is_attacked(3, false) { moves.push(Move { from, to: 2, promotion: 0 }); }
                    } else if !white && from == 60 && !self.is_attacked(60, true) {
                        if self.castling & BLACK_KING_SIDE != 0 && self.squares[61] == 0 && self.squares[62] == 0 && self.squares[63] == -3
                            && !self.is_attacked(61, true) { moves.push(Move { from, to: 62, promotion: 0 }); }
                        if self.castling & BLACK_QUEEN_SIDE != 0 && self.squares[57] == 0 && self.squares[58] == 0 && self.squares[59] == 0 && self.squares[56] == -3
                            && !self.is_attacked(59, true) { moves.push(Move { from, to: 58, promotion: 0 }); }
                    }
                }
                _ => unreachable!(),
            }
        }
        moves
    }

    fn push_pawn_move(&self, moves: &mut Vec<Move>, from: u8, to: u8, promotion_rank: i8) {
        if rank_of(to) as i8 == promotion_rank {
            for promotion in [5, 4, 3, 2] {
                moves.push(Move { from, to, promotion: if self.white_to_move { promotion } else { -promotion } });
            }
        } else { moves.push(Move { from, to, promotion: 0 }); }
    }

    fn push_target(&self, moves: &mut Vec<Move>, from: u8, to: u8, white: bool) {
        let target = self.squares[to as usize];
        if target == 0 || ((target > 0) != white && target.abs() != 6) {
            moves.push(Move { from, to, promotion: 0 });
        }
    }

    pub(crate) fn legal_moves(&mut self) -> Vec<Move> {
        let pseudo = self.pseudo_moves();
        let mover_white = self.white_to_move;
        let mut legal = Vec::with_capacity(pseudo.len());
        for mv in pseudo {
            let undo = self.make_move(mv);
            let okay = !self.is_in_check(mover_white);
            self.unmake_move(mv, undo);
            if okay { legal.push(mv); }
        }
        legal
    }

    pub(crate) fn make_move(&mut self, mv: Move) -> Undo {
        let mover = self.squares[mv.from as usize];
        let old_hash = self.hash;
        let old_castling = self.castling;
        let old_ep = self.ep;
        self.hash ^= ZOBRIST.castling[old_castling as usize];
        if let Some(file) = self.ep_hash_file() { self.hash ^= ZOBRIST.ep_file[file as usize]; }
        if !self.white_to_move { self.hash ^= ZOBRIST.side; }

        let en_passant = mover.abs() == 1 && old_ep == Some(mv.to) && self.squares[mv.to as usize] == 0;
        let captured_square = if en_passant { (mv.to as i16 - if mover > 0 { 8 } else { -8 }) as u8 } else { mv.to };
        let captured = self.squares[captured_square as usize];
        let undo = Undo {
            mover, captured, captured_square, castling: old_castling, ep: old_ep,
            halfmove: self.halfmove, fullmove: self.fullmove, hash: old_hash,
        };

        self.hash ^= ZOBRIST.pieces[piece_index(mover)][mv.from as usize];
        self.squares[mv.from as usize] = 0;
        if captured != 0 {
            self.hash ^= ZOBRIST.pieces[piece_index(captured)][captured_square as usize];
            self.squares[captured_square as usize] = 0;
        }
        if mover.abs() == 6 && (mv.to as i16 - mv.from as i16).abs() == 2 {
            let (rook_from, rook_to) = if mv.to > mv.from { (mv.from + 3, mv.from + 1) } else { (mv.from - 4, mv.from - 1) };
            let rook = self.squares[rook_from as usize];
            self.hash ^= ZOBRIST.pieces[piece_index(rook)][rook_from as usize];
            self.hash ^= ZOBRIST.pieces[piece_index(rook)][rook_to as usize];
            self.squares[rook_from as usize] = 0;
            self.squares[rook_to as usize] = rook;
        }
        let placed = if mv.promotion != 0 { mv.promotion } else { mover };
        self.squares[mv.to as usize] = placed;
        self.hash ^= ZOBRIST.pieces[piece_index(placed)][mv.to as usize];

        if mover.abs() == 6 {
            self.castling &= if mover > 0 { !(WHITE_KING_SIDE | WHITE_QUEEN_SIDE) } else { !(BLACK_KING_SIDE | BLACK_QUEEN_SIDE) };
        }
        for (square, bits) in [(0, WHITE_QUEEN_SIDE), (7, WHITE_KING_SIDE), (56, BLACK_QUEEN_SIDE), (63, BLACK_KING_SIDE)] {
            if mv.from == square || captured_square == square && captured.abs() == 3 { self.castling &= !bits; }
        }
        self.ep = if mover.abs() == 1 && (mv.to as i16 - mv.from as i16).abs() == 16 {
            Some(((mv.to as u16 + mv.from as u16) / 2) as u8)
        } else { None };
        self.halfmove = if mover.abs() == 1 || captured != 0 { 0 } else { self.halfmove.saturating_add(1) };
        if !self.white_to_move { self.fullmove = self.fullmove.saturating_add(1); }
        self.white_to_move = !self.white_to_move;
        self.hash ^= ZOBRIST.castling[self.castling as usize];
        if let Some(file) = self.ep_hash_file() { self.hash ^= ZOBRIST.ep_file[file as usize]; }
        if !self.white_to_move { self.hash ^= ZOBRIST.side; }
        undo
    }

    pub(crate) fn unmake_move(&mut self, mv: Move, undo: Undo) {
        self.white_to_move = !self.white_to_move;
        self.squares[mv.from as usize] = undo.mover;
        self.squares[mv.to as usize] = 0;
        if undo.captured != 0 { self.squares[undo.captured_square as usize] = undo.captured; }
        if undo.mover.abs() == 6 && (mv.to as i16 - mv.from as i16).abs() == 2 {
            let (rook_from, rook_to) = if mv.to > mv.from { (mv.from + 3, mv.from + 1) } else { (mv.from - 4, mv.from - 1) };
            self.squares[rook_from as usize] = self.squares[rook_to as usize];
            self.squares[rook_to as usize] = 0;
        }
        self.castling = undo.castling;
        self.ep = undo.ep;
        self.halfmove = undo.halfmove;
        self.fullmove = undo.fullmove;
        self.hash = undo.hash;
    }

    pub fn play_uci(&mut self, text: &str) -> Result<(), String> {
        let candidate = parse_uci_move(text).ok_or_else(|| format!("invalid UCI move: {text}"))?;
        let mv = self.legal_moves().into_iter().find(|m| {
            m.from == candidate.from && m.to == candidate.to && m.promotion.abs() == candidate.promotion.abs()
        })
            .ok_or_else(|| format!("illegal UCI move: {text}"))?;
        self.make_move(mv);
        Ok(())
    }

    pub fn perft(&mut self, depth: u8) -> u64 {
        if depth == 0 { return 1; }
        let moves = self.legal_moves();
        if depth == 1 { return moves.len() as u64; }
        let mut nodes = 0;
        for mv in moves {
            let undo = self.make_move(mv);
            nodes += self.perft(depth - 1);
            self.unmake_move(mv, undo);
        }
        nodes
    }

    pub fn perft_divide(&mut self, depth: u8) -> Vec<(String, u64)> {
        if depth == 0 { return vec![]; }
        let moves = self.legal_moves();
        let mut result = Vec::with_capacity(moves.len());
        for mv in moves {
            let undo = self.make_move(mv);
            let nodes = self.perft(depth - 1);
            self.unmake_move(mv, undo);
            result.push((mv.uci(), nodes));
        }
        result
    }
}

fn letter_piece(c: char) -> Option<i8> {
    let kind = match c.to_ascii_lowercase() { 'p' => 1, 'n' => 2, 'r' => 3, 'b' => 4, 'q' => 5, 'k' => 6, _ => return None };
    Some(if c.is_ascii_uppercase() { kind } else { -kind })
}

fn piece_letter(piece: i8) -> char {
    let c = match piece.abs() { 1 => 'p', 2 => 'n', 3 => 'r', 4 => 'b', 5 => 'q', 6 => 'k', _ => '?' };
    if piece > 0 { c.to_ascii_uppercase() } else { c }
}

fn piece_index(piece: i8) -> usize {
    if piece > 0 { (piece - 1) as usize } else { (6 - piece - 1) as usize }
}

fn square_name(square: u8) -> String {
    format!("{}{}", (b'a' + file_of(square)) as char, (b'1' + rank_of(square)) as char)
}

fn parse_square(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) { return None; }
    Some((b[0] - b'a') + 8 * (b[1] - b'1'))
}

fn parse_uci_move(s: &str) -> Option<Move> {
    if s.len() != 4 && s.len() != 5 { return None; }
    let from = parse_square(&s[0..2])?;
    let to = parse_square(&s[2..4])?;
    let promotion = if s.len() == 5 {
        let p = letter_piece(s[4..].chars().next()?)?;
        if p < 0 { p } else { -p }
    } else { 0 };
    Some(Move { from, to, promotion })
}

fn file_of(square: u8) -> u8 { square & 7 }
fn rank_of(square: u8) -> u8 { square >> 3 }
fn make_square(file: i8, rank: i8) -> Option<u8> {
    if (0..8).contains(&file) && (0..8).contains(&rank) { Some((rank * 8 + file) as u8) } else { None }
}

pub fn play_position_command(board: &mut Board, command: &str) -> Result<(), String> {
    play_position_command_with_history(board, command).map(|_| ())
}

pub fn play_position_command_with_history(board: &mut Board, command: &str) -> Result<Vec<u64>, String> {
    let mut parts = command.split_whitespace();
    if parts.next() != Some("position") { return Err("expected position command".into()); }
    let first = parts.next().ok_or_else(|| "missing position kind".to_string())?;
    let (fen, remainder) = if first == "startpos" {
        (START_FEN.to_string(), parts.collect::<Vec<_>>())
    } else if first == "fen" {
        let tokens: Vec<&str> = parts.collect();
        let moves_at = tokens.iter().position(|&t| t == "moves").unwrap_or(tokens.len());
        if moves_at < 4 { return Err("position fen needs at least four FEN fields".into()); }
        (tokens[..moves_at].join(" "), tokens[moves_at..].to_vec())
    } else { return Err("expected startpos or fen".into()); };
    *board = Board::from_fen(&fen)?;
    let mut history = vec![board.hash()];
    let moves: &[&str] = if let Some(index) = remainder.iter().position(|&t| t == "moves") { &remainder[index + 1..] } else { &[] };
    for mv in moves {
        board.play_uci(mv)?;
        history.push(board.hash());
    }
    Ok(history)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_perft_through_depth_five() {
        let mut board = Board::startpos();
        for (depth, expected) in [(1, 20), (2, 400), (3, 8_902), (4, 197_281), (5, 4_865_609)] {
            assert_eq!(board.perft(depth), expected, "depth {depth}");
        }
    }

    #[test]
    fn fen_round_trip_and_hash_are_stable() {
        let board = Board::startpos();
        assert_eq!(Board::from_fen(&board.to_fen()).unwrap().to_fen(), board.to_fen());
        assert_eq!(board.hash(), board.recompute_hash());
    }

    #[test]
    fn en_passant_hash_only_changes_when_a_capture_is_available() {
        let no_capturer = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        let no_ep = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert_eq!(no_capturer.hash(), no_ep.hash());

        let capturable = Board::from_fen("rnbqkbnr/pppppppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        let capturable_without_ep = Board::from_fen("rnbqkbnr/pppppppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 1").unwrap();
        assert_ne!(capturable.hash(), capturable_without_ep.hash());
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/4p3/PPP2PPP/RNBQKBNR b KQkq e3 0 1").is_err());

        let mut pinned_ep = Board::from_fen("k7/8/8/4KPpr/8/8/8/8 w - g6 0 1").unwrap();
        let pinned_without_ep = Board::from_fen("k7/8/8/4KPpr/8/8/8/8 w - - 0 1").unwrap();
        assert!(!pinned_ep.legal_moves().iter().any(|mv| mv.uci() == "f5g6"));
        assert_eq!(pinned_ep.hash(), pinned_without_ep.hash());
    }

    #[test]
    fn make_unmake_and_incremental_hash_survive_random_play() {
        let mut board = Board::startpos();
        let mut random = 0x2026_1024_4b49_5241u64;
        for ply in 0..100_000 {
            let before_generation = board.clone();
            let mut moves = board.legal_moves();
            assert_eq!(board, before_generation, "legal move generation restoration at random ply {ply}");
            if moves.is_empty() || board.halfmove >= 100 {
                board = Board::startpos();
                moves = board.legal_moves();
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let mv = moves[(random as usize) % moves.len()];

            let before = board.clone();
            let undo = board.make_move(mv);
            assert_eq!(board.hash(), board.recompute_hash(), "incremental hash at random ply {ply}, move {}, before {}", mv.uci(), before.to_fen());
            let after = board.clone();
            let parsed = Board::from_fen(&board.to_fen()).expect("generated FEN parses");
            assert_eq!(parsed, board, "FEN round trip at random ply {ply}");
            board.unmake_move(mv, undo);
            assert_eq!(board, before, "make/unmake restoration at random ply {ply}");
            assert_eq!(board.hash(), board.recompute_hash(), "restored hash at random ply {ply}");
            let redo = board.make_move(mv);
            assert_eq!(board, after, "repeat make at random ply {ply}");
            assert_eq!(board.hash(), board.recompute_hash(), "repeat hash at random ply {ply}");
            let _ = redo;
        }
    }

    #[test]
    fn special_moves_restore_board_and_hash() {
        let fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "rnbqkbnr/pppppppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 1",
        ];
        for fen in fens {
            let mut board = Board::from_fen(fen).unwrap();
            let before = board.clone();
            for mv in board.legal_moves() {
                let undo = board.make_move(mv);
                assert_eq!(board.hash(), board.recompute_hash(), "after {}", mv.uci());
                board.unmake_move(mv, undo);
                assert_eq!(board, before, "after {}", mv.uci());
            }
        }
        let mut ep_board = Board::from_fen("rnbqkbnr/pppppppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
        assert!(ep_board.legal_moves().iter().any(|mv| mv.uci() == "d4e3"));
    }
}
