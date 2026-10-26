//! Kira-owned fixed-width training records and Chess768 feature extraction.
//!
//! The schema is intentionally a small integer-only record.  It contains no
//! move-generation or game-rule code; source positions are validated by Kira
//! before this codec is called.

pub const TRAINING_RECORD_BYTES: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrainingRecord {
    pub occ: u64,
    pub pcs: [u8; 16],
    pub score: i16,
    pub result: u8,
    pub ksq: u8,
    pub opp_ksq: u8,
    pub extra: [u8; 3],
}

const _: () = {
    assert!(std::mem::size_of::<TrainingRecord>() == TRAINING_RECORD_BYTES);
    assert!(std::mem::align_of::<TrainingRecord>() == 8);
    assert!(std::mem::offset_of!(TrainingRecord, occ) == 0);
    assert!(std::mem::offset_of!(TrainingRecord, pcs) == 8);
    assert!(std::mem::offset_of!(TrainingRecord, score) == 24);
    assert!(std::mem::offset_of!(TrainingRecord, result) == 26);
    assert!(std::mem::offset_of!(TrainingRecord, ksq) == 27);
    assert!(std::mem::offset_of!(TrainingRecord, opp_ksq) == 28);
    assert!(std::mem::offset_of!(TrainingRecord, extra) == 29);
};

impl TrainingRecord {
    /// Build the packed row from Kira's White/Black and piece-type bitboards.
    /// Piece nibbles are stored in side-to-move perspective and ascending
    /// square order, matching the active feature mapper used during training.
    pub fn from_bitboards(
        mut bitboards: [u64; 8],
        black_to_move: bool,
        score_stm: i16,
        result_stm: u8,
    ) -> Result<Self, String> {
        if result_stm > 2 {
            return Err(format!("invalid side-to-move result code {result_stm}"));
        }
        if black_to_move {
            for bitboard in &mut bitboards {
                *bitboard = bitboard.swap_bytes();
            }
            bitboards.swap(0, 1);
        }

        let occ = bitboards[0] | bitboards[1];
        let mut pcs = [0u8; 16];
        let mut piece_count = 0usize;
        for square in 0..64u32 {
            let bit = 1u64 << square;
            if occ & bit == 0 {
                continue;
            }
            let color = if bitboards[0] & bit != 0 {
                0
            } else if bitboards[1] & bit != 0 {
                8
            } else {
                return Err("occupied square has no side-to-move color".into());
            };
            let kind = bitboards[2..]
                .iter()
                .position(|pieces| pieces & bit != 0)
                .ok_or_else(|| "occupied square has no piece type".to_string())?;
            if piece_count == 32 {
                return Err("position contains more than 32 pieces".into());
            }
            let piece = color | kind as u8;
            pcs[piece_count / 2] |= piece << (4 * (piece_count & 1));
            piece_count += 1;
        }

        if piece_count == 0 {
            return Err("position contains no pieces".into());
        }
        let own_king = bitboards[0] & bitboards[7];
        let opponent_king = bitboards[1] & bitboards[7];
        if own_king.count_ones() != 1 || opponent_king.count_ones() != 1 {
            return Err("position must contain exactly one king for each side".into());
        }

        Ok(Self {
            occ,
            pcs,
            score: score_stm,
            result: result_stm,
            ksq: own_king.trailing_zeros() as u8,
            opp_ksq: opponent_king.trailing_zeros() as u8 ^ 56,
            extra: [0; 3],
        })
    }

    pub fn result_idx(self) -> usize {
        usize::from(self.result)
    }

    /// Visit packed pieces in the same ascending-square order as the record.
    pub fn for_each_piece(self, mut visit: impl FnMut(u8, u8)) {
        let mut occupied = self.occ;
        let mut index = 0usize;
        while occupied != 0 {
            let square = occupied.trailing_zeros() as u8;
            let piece = (self.pcs[index / 2] >> (4 * (index & 1))) & 0x0f;
            visit(piece, square);
            occupied &= occupied - 1;
            index += 1;
        }
    }

    pub fn map_chess768(&self, mut visit: impl FnMut(usize, usize)) {
        self.for_each_piece(|piece, square| {
            let opponent = usize::from(piece & 8 != 0);
            let kind = usize::from(piece & 7);
            let square = usize::from(square);
            let stm = opponent * 384 + kind * 64 + square;
            let nstm = (1 - opponent) * 384 + kind * 64 + (square ^ 56);
            visit(stm, nstm);
        });
    }
}

pub fn decode_training_record(bytes: &[u8]) -> Result<TrainingRecord, String> {
    if bytes.len() != TRAINING_RECORD_BYTES {
        return Err(format!(
            "training record must be {TRAINING_RECORD_BYTES} bytes, received {}",
            bytes.len()
        ));
    }
    Ok(TrainingRecord {
        occ: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        pcs: bytes[8..24].try_into().unwrap(),
        score: i16::from_le_bytes(bytes[24..26].try_into().unwrap()),
        result: bytes[26],
        ksq: bytes[27],
        opp_ksq: bytes[28],
        extra: bytes[29..32].try_into().unwrap(),
    })
}

pub fn encode_training_record(record: &TrainingRecord) -> [u8; TRAINING_RECORD_BYTES] {
    let mut bytes = [0u8; TRAINING_RECORD_BYTES];
    bytes[0..8].copy_from_slice(&record.occ.to_le_bytes());
    bytes[8..24].copy_from_slice(&record.pcs);
    bytes[24..26].copy_from_slice(&record.score.to_le_bytes());
    bytes[26] = record.result;
    bytes[27] = record.ksq;
    bytes[28] = record.opp_ksq;
    bytes[29..32].copy_from_slice(&record.extra);
    bytes
}
