//! Versioned int16 NNUE inference format for M4 exports.
//!
//! The float [`super::nnue::NnueNetwork`] remains the reference evaluator.
//! This module stores the same 768xH CReLU transform and STM-first output
//! head as signed int16 parameters with a fixed Q14 scale.

use super::nnue::{FEATURE_COUNT, feature_index};
use crate::{Board, eval::Evaluator};
use std::{fs, path::Path};

pub const QUANTIZATION_SCALE: i32 = 16_384;
pub const SCORE_SCALE: i32 = 400;
pub const OPTIMIZER_CLIP_MIN: f32 = -1.98;
pub const OPTIMIZER_CLIP_MAX: f32 = 1.98;
const MAGIC: &[u8; 8] = b"KIRANUEQ";
const FORMAT_VERSION: u16 = 1;
pub const HEADER_BYTES: usize = 156;
const FEATURE_MAP_ID: u8 = 1;
const ACTIVATION_ID: u8 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantizationReport {
    pub feature_weights_clamped: u32,
    pub hidden_bias_clamped: u32,
    pub output_weights_clamped: u32,
    pub output_bias_clamped: u32,
    pub optimizer_clip_violations: u32,
}

impl QuantizationReport {
    pub fn clamped_total(self) -> u32 {
        self.feature_weights_clamped
            .saturating_add(self.hidden_bias_clamped)
            .saturating_add(self.output_weights_clamped)
            .saturating_add(self.output_bias_clamped)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuantizedNnueMetadata {
    pub source_weights_sha256: [u8; 32],
    pub run_manifest_sha256: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct QuantizedNnueNetwork<const H: usize> {
    feature_weights: Vec<i16>,
    hidden_bias: [i16; H],
    output_weights: [[i16; H]; 2],
    output_bias: i16,
    report: QuantizationReport,
    metadata: QuantizedNnueMetadata,
}

impl<const H: usize> QuantizedNnueNetwork<H> {
    pub fn from_float_parts(
        feature_weights: &[f32],
        hidden_bias: &[f32],
        output_weights: &[f32],
        output_bias: f32,
        output_scale: f32,
        metadata: QuantizedNnueMetadata,
    ) -> Result<Self, String> {
        if H == 0 || H > usize::from(u16::MAX) {
            return Err("NNUE hidden size must be in 1..=65535".into());
        }
        if feature_weights.len() != FEATURE_COUNT * H {
            return Err(format!(
                "expected {} feature weights, received {}",
                FEATURE_COUNT * H,
                feature_weights.len()
            ));
        }
        if hidden_bias.len() != H {
            return Err(format!("expected {H} hidden biases, received {}", hidden_bias.len()));
        }
        if output_weights.len() != 2 * H {
            return Err(format!("expected {} output weights, received {}", 2 * H, output_weights.len()));
        }
        if output_scale.to_bits() != (SCORE_SCALE as f32).to_bits() {
            return Err(format!("M4 NNUE output scale must be {SCORE_SCALE}"));
        }

        let mut report = QuantizationReport::default();
        let (feature_weights, feature_clamped, feature_violations) = quantize_slice(feature_weights)?;
        report.feature_weights_clamped = feature_clamped;
        report.optimizer_clip_violations += feature_violations;
        let (hidden_bias, bias_clamped, bias_violations) = quantize_slice(hidden_bias)?;
        report.hidden_bias_clamped = bias_clamped;
        report.optimizer_clip_violations += bias_violations;
        let hidden_bias = hidden_bias
            .try_into()
            .map_err(|_| "internal hidden-bias size mismatch".to_string())?;
        let (quantized_output, output_clamped, output_violations) = quantize_slice(output_weights)?;
        report.output_weights_clamped = output_clamped;
        report.optimizer_clip_violations += output_violations;
        let output_weights = [
            quantized_output[..H].try_into().map_err(|_| "internal first output-head size mismatch".to_string())?,
            quantized_output[H..].try_into().map_err(|_| "internal second output-head size mismatch".to_string())?,
        ];
        let (output_bias, output_bias_clamped, output_bias_violations) = quantize_one(output_bias)?;
        report.output_bias_clamped = u32::from(output_bias_clamped);
        report.optimizer_clip_violations += u32::from(output_bias_violations);

        Ok(Self { feature_weights, hidden_bias, output_weights, output_bias, report, metadata })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_BYTES {
            return Err("truncated NNUE header".into());
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.take(8)? != MAGIC {
            return Err("invalid NNUE magic".into());
        }
        let version = cursor.u16()?;
        if version != FORMAT_VERSION {
            return Err(format!("unsupported NNUE format version {version}"));
        }
        let header_bytes = usize::from(cursor.u16()?);
        if header_bytes != HEADER_BYTES {
            return Err(format!("invalid NNUE header size {header_bytes}"));
        }
        let feature_count = usize::from(cursor.u16()?);
        if feature_count != FEATURE_COUNT {
            return Err(format!("NNUE feature count must be {FEATURE_COUNT}, found {feature_count}"));
        }
        let hidden_size = usize::from(cursor.u16()?);
        if hidden_size != H {
            return Err(format!("NNUE hidden size must be {H}, found {hidden_size}"));
        }
        if H == 0 {
            return Err("NNUE hidden size must be positive".into());
        }
        let q_scale = cursor.i32()?;
        if q_scale != QUANTIZATION_SCALE {
            return Err(format!("unsupported NNUE quantization scale {q_scale}"));
        }
        let score_scale = cursor.i32()?;
        if score_scale != SCORE_SCALE {
            return Err(format!("unsupported NNUE score scale {score_scale}"));
        }
        let clip_min = f32::from_le_bytes(cursor.take(4)?.try_into().unwrap());
        let clip_max = f32::from_le_bytes(cursor.take(4)?.try_into().unwrap());
        if clip_min.to_bits() != OPTIMIZER_CLIP_MIN.to_bits() || clip_max.to_bits() != OPTIMIZER_CLIP_MAX.to_bits() {
            return Err("NNUE optimizer clipping bounds do not match M4 training".into());
        }
        if cursor.u8()? != FEATURE_MAP_ID {
            return Err("unsupported NNUE feature-map version".into());
        }
        if cursor.u8()? != ACTIVATION_ID {
            return Err("unsupported NNUE activation".into());
        }
        if cursor.take(2)? != [0, 0] {
            return Err("NNUE reserved header bytes must be zero".into());
        }
        let payload_len = usize::try_from(cursor.u32()?).map_err(|_| "NNUE payload length exceeds usize")?;
        let report = QuantizationReport {
            feature_weights_clamped: cursor.u32()?,
            hidden_bias_clamped: cursor.u32()?,
            output_weights_clamped: cursor.u32()?,
            output_bias_clamped: cursor.u32()?,
            optimizer_clip_violations: cursor.u32()?,
        };
        let payload_sha256 = cursor.take(32)?;
        let metadata = QuantizedNnueMetadata {
            source_weights_sha256: cursor.take(32)?.try_into().unwrap(),
            run_manifest_sha256: cursor.take(32)?.try_into().unwrap(),
        };
        if cursor.offset != HEADER_BYTES {
            return Err("internal NNUE header layout mismatch".into());
        }

        let expected_payload_len = payload_size::<H>()?;
        if payload_len != expected_payload_len {
            return Err(format!("NNUE payload length must be {expected_payload_len}, found {payload_len}"));
        }
        if bytes.len() != HEADER_BYTES + payload_len {
            return Err("NNUE file length does not match its declared payload".into());
        }
        let payload = &bytes[HEADER_BYTES..];
        if sha256_digest(payload).as_slice() != payload_sha256 {
            return Err("NNUE payload SHA-256 mismatch".into());
        }

        let mut values = payload.chunks_exact(2).map(|word| i16::from_le_bytes([word[0], word[1]]));
        let feature_weights = values.by_ref().take(FEATURE_COUNT * H).collect::<Vec<_>>();
        let hidden_bias: [i16; H] = values.by_ref().take(H).collect::<Vec<_>>().try_into()
            .map_err(|_| "truncated NNUE hidden bias".to_string())?;
        let first: [i16; H] = values.by_ref().take(H).collect::<Vec<_>>().try_into()
            .map_err(|_| "truncated NNUE first output head".to_string())?;
        let second: [i16; H] = values.by_ref().take(H).collect::<Vec<_>>().try_into()
            .map_err(|_| "truncated NNUE second output head".to_string())?;
        let output_bias = values.next().ok_or_else(|| "truncated NNUE output bias".to_string())?;
        if values.next().is_some() {
            return Err("NNUE payload has unexpected trailing parameters".into());
        }
        if report.feature_weights_clamped > (FEATURE_COUNT * H) as u32
            || report.hidden_bias_clamped > H as u32
            || report.output_weights_clamped > (2 * H) as u32
            || report.output_bias_clamped > 1
            || report.optimizer_clip_violations > (FEATURE_COUNT * H + 3 * H + 1) as u32
        {
            return Err("NNUE clamp or optimizer-clip count exceeds parameter count".into());
        }
        if report.optimizer_clip_violations < report.clamped_total() {
            return Err("NNUE optimizer-clip count is lower than its saturation count".into());
        }
        Ok(Self { feature_weights, hidden_bias, output_weights: [first, second], output_bias, report, metadata })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|error| error.to_string())?;
        Self::from_bytes(&bytes)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let payload_len = payload_size::<H>()?;
        let mut payload = Vec::with_capacity(payload_len);
        for value in &self.feature_weights { payload.extend_from_slice(&value.to_le_bytes()); }
        for value in &self.hidden_bias { payload.extend_from_slice(&value.to_le_bytes()); }
        for head in &self.output_weights {
            for value in head { payload.extend_from_slice(&value.to_le_bytes()); }
        }
        payload.extend_from_slice(&self.output_bias.to_le_bytes());
        if payload.len() != payload_len {
            return Err("internal NNUE payload layout mismatch".into());
        }

        let mut bytes = Vec::with_capacity(HEADER_BYTES + payload_len);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
        bytes.extend_from_slice(&(FEATURE_COUNT as u16).to_le_bytes());
        bytes.extend_from_slice(&(H as u16).to_le_bytes());
        bytes.extend_from_slice(&QUANTIZATION_SCALE.to_le_bytes());
        bytes.extend_from_slice(&SCORE_SCALE.to_le_bytes());
        bytes.extend_from_slice(&OPTIMIZER_CLIP_MIN.to_le_bytes());
        bytes.extend_from_slice(&OPTIMIZER_CLIP_MAX.to_le_bytes());
        bytes.push(FEATURE_MAP_ID);
        bytes.push(ACTIVATION_ID);
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&(payload_len as u32).to_le_bytes());
        bytes.extend_from_slice(&self.report.feature_weights_clamped.to_le_bytes());
        bytes.extend_from_slice(&self.report.hidden_bias_clamped.to_le_bytes());
        bytes.extend_from_slice(&self.report.output_weights_clamped.to_le_bytes());
        bytes.extend_from_slice(&self.report.output_bias_clamped.to_le_bytes());
        bytes.extend_from_slice(&self.report.optimizer_clip_violations.to_le_bytes());
        bytes.extend_from_slice(&sha256_digest(&payload));
        bytes.extend_from_slice(&self.metadata.source_weights_sha256);
        bytes.extend_from_slice(&self.metadata.run_manifest_sha256);
        if bytes.len() != HEADER_BYTES {
            return Err("internal NNUE header layout mismatch".into());
        }
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn feature_weights(&self) -> &[i16] { &self.feature_weights }
    pub fn hidden_bias(&self) -> &[i16; H] { &self.hidden_bias }
    pub fn output_weights(&self) -> &[[i16; H]; 2] { &self.output_weights }
    pub fn output_bias(&self) -> i16 { self.output_bias }
    pub fn quantization_report(&self) -> QuantizationReport { self.report }
    pub fn metadata(&self) -> QuantizedNnueMetadata { self.metadata }

    pub fn evaluate_board(&self, board: &Board) -> i32 {
        let mut accumulators = [
            self.hidden_bias.map(i32::from),
            self.hidden_bias.map(i32::from),
        ];
        for (square, &piece) in board.squares.iter().enumerate() {
            if piece == 0 { continue; }
            for (perspective, accumulator) in accumulators.iter_mut().enumerate() {
                let feature = feature_index(piece, square as u8, perspective == 0)
                    .expect("board contains valid pieces and squares");
                let start = feature * H;
                for (sum, &weight) in accumulator.iter_mut().zip(&self.feature_weights[start..start + H]) {
                    *sum += i32::from(weight);
                }
            }
        }

        let first = usize::from(!board.white_to_move);
        let second = 1 - first;
        let q = i64::from(QUANTIZATION_SCALE);
        let mut output = i64::from(self.output_bias) * q;
        for hidden in 0..H {
            let first_activation = accumulators[first][hidden].clamp(0, QUANTIZATION_SCALE);
            let second_activation = accumulators[second][hidden].clamp(0, QUANTIZATION_SCALE);
            output += i64::from(first_activation) * i64::from(self.output_weights[0][hidden]);
            output += i64::from(second_activation) * i64::from(self.output_weights[1][hidden]);
        }
        let denominator = f64::from(QUANTIZATION_SCALE).powi(2);
        (output as f64 * f64::from(SCORE_SCALE) / denominator).round() as i32
    }
}

impl<const H: usize> Evaluator for QuantizedNnueNetwork<H> {
    fn evaluate(&self, board: &Board) -> i32 { self.evaluate_board(board) }
}

fn quantize_slice(values: &[f32]) -> Result<(Vec<i16>, u32, u32), String> {
    let mut clamped = 0u32;
    let mut clip_violations = 0u32;
    let quantized = values.iter().map(|&value| {
        let (quantized, was_clamped, clip_violation) = quantize_one(value)?;
        clamped = clamped.saturating_add(u32::from(was_clamped));
        clip_violations = clip_violations.saturating_add(u32::from(clip_violation));
        Ok(quantized)
    }).collect::<Result<Vec<_>, String>>()?;
    Ok((quantized, clamped, clip_violations))
}

fn quantize_one(value: f32) -> Result<(i16, bool, bool), String> {
    if !value.is_finite() {
        return Err("NNUE parameters must be finite".into());
    }
    let clip_violation = !(OPTIMIZER_CLIP_MIN..=OPTIMIZER_CLIP_MAX).contains(&value);
    let scaled = (f64::from(value) * f64::from(QUANTIZATION_SCALE)).round();
    if scaled < f64::from(i16::MIN) {
        Ok((i16::MIN, true, clip_violation))
    } else if scaled > f64::from(i16::MAX) {
        Ok((i16::MAX, true, clip_violation))
    } else {
        Ok((scaled as i16, false, clip_violation))
    }
}

fn payload_size<const H: usize>() -> Result<usize, String> {
    FEATURE_COUNT.checked_mul(H)
        .and_then(|features| features.checked_add(3 * H))
        .and_then(|values| values.checked_add(1))
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| "NNUE payload size overflow".to_string())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    crate::datagen::sha256_hex(bytes)
}

fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    let hex = sha256_hex(bytes);
    let mut digest = [0u8; 32];
    for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        digest[index] = u8::from_str_radix(std::str::from_utf8(pair).expect("SHA hex is ASCII"), 16)
            .expect("SHA hex contains digits");
    }
    digest
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, offset: 0 } }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self.offset.checked_add(count).ok_or_else(|| "NNUE offset overflow".to_string())?;
        if end > self.bytes.len() { return Err("truncated NNUE file".into()); }
        let result = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16, String> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn i32(&mut self) -> Result<i32, String> { Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::nnue::NnueNetwork;
    use std::collections::HashSet;

    fn metadata() -> QuantizedNnueMetadata {
        QuantizedNnueMetadata { source_weights_sha256: [0x11; 32], run_manifest_sha256: [0x22; 32] }
    }

    fn fixture<const H: usize>() -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let mut state = 0x4b49_5241_4d34_4e4eu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let feature_weights = (0..FEATURE_COUNT * H)
            .map(|_| ((next() % 2001) as i32 - 1000) as f32 / 10_000.0)
            .collect();
        let hidden_bias = (0..H)
            .map(|_| ((next() % 1001) as i32 - 500) as f32 / 10_000.0)
            .collect();
        let output_weights = (0..2 * H)
            .map(|_| ((next() % 2001) as i32 - 1000) as f32 / 2_000.0)
            .collect();
        let output_bias = ((next() % 1001) as i32 - 500) as f32 / 10_000.0;
        (feature_weights, hidden_bias, output_weights, output_bias)
    }

    fn make_network<const H: usize>() -> (NnueNetwork<H>, QuantizedNnueNetwork<H>) {
        let (features, bias, outputs, output_bias) = fixture::<H>();
        let float_weights: [[f32; H]; 2] = [
            outputs[..H].to_vec().try_into().unwrap(),
            outputs[H..].to_vec().try_into().unwrap(),
        ];
        let float = NnueNetwork::from_parts(
            features.clone(), bias.clone().try_into().unwrap(), float_weights, output_bias, 400.0,
        ).unwrap();
        let quantized = QuantizedNnueNetwork::from_float_parts(
            &features, &bias, &outputs, output_bias, 400.0, metadata(),
        ).unwrap();
        (float, quantized)
    }

    #[test]
    fn format_round_trips_parameters_and_provenance() {
        let (_, network) = make_network::<8>();
        let bytes = network.to_bytes().unwrap();
        let loaded = QuantizedNnueNetwork::<8>::from_bytes(&bytes).unwrap();
        assert_eq!(network.feature_weights, loaded.feature_weights);
        assert_eq!(network.hidden_bias, loaded.hidden_bias);
        assert_eq!(network.output_weights, loaded.output_weights);
        assert_eq!(network.output_bias, loaded.output_bias);
        assert_eq!(network.report, loaded.report);
        assert_eq!(network.metadata, loaded.metadata);
    }

    #[test]
    fn loader_rejects_bad_magic_version_dimensions_scales_length_and_checksum() {
        let (_, network) = make_network::<8>();
        let original = network.to_bytes().unwrap();
        let mut bad = original.clone();
        bad[0] ^= 1;
        assert!(QuantizedNnueNetwork::<8>::from_bytes(&bad).unwrap_err().contains("magic"));
        let mut bad = original.clone();
        bad[8..10].copy_from_slice(&99u16.to_le_bytes());
        assert!(QuantizedNnueNetwork::<8>::from_bytes(&bad).unwrap_err().contains("version"));
        let mut bad = original.clone();
        bad[12..14].copy_from_slice(&767u16.to_le_bytes());
        assert!(QuantizedNnueNetwork::<8>::from_bytes(&bad).unwrap_err().contains("feature count"));
        let mut bad = original.clone();
        bad[16..20].copy_from_slice(&12_345i32.to_le_bytes());
        assert!(QuantizedNnueNetwork::<8>::from_bytes(&bad).unwrap_err().contains("quantization scale"));
        let mut bad = original.clone();
        bad[56..60].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(QuantizedNnueNetwork::<8>::from_bytes(&bad).unwrap_err().contains("count exceeds"));
        let mut bad = original.clone();
        bad[40..44].copy_from_slice(&1u32.to_le_bytes());
        assert!(QuantizedNnueNetwork::<8>::from_bytes(&bad).unwrap_err().contains("lower than its saturation"));
        let mut bad = original.clone();
        bad.pop();
        assert!(QuantizedNnueNetwork::<8>::from_bytes(&bad).unwrap_err().contains("length"));
        let mut bad = original;
        *bad.last_mut().unwrap() ^= 1;
        assert!(QuantizedNnueNetwork::<8>::from_bytes(&bad).unwrap_err().contains("SHA-256"));
    }

    #[test]
    fn finite_extremes_saturate_instead_of_wrapping_and_are_counted() {
        let mut features = vec![0.0; FEATURE_COUNT * 2];
        features[0] = 3.0;
        features[1] = -3.0;
        let network = QuantizedNnueNetwork::<2>::from_float_parts(
            &features, &[0.0, 0.0], &[0.0; 4], 0.0, 400.0, metadata(),
        ).unwrap();
        let report = network.quantization_report();
        assert_eq!(&network.feature_weights[..2], &[i16::MAX, i16::MIN]);
        assert_eq!(report.feature_weights_clamped, 2);
        assert_eq!(report.clamped_total(), 2);
        assert_eq!(report.optimizer_clip_violations, 2);
        let loaded = QuantizedNnueNetwork::<2>::from_bytes(&network.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded.feature_weights()[..2], [i16::MAX, i16::MIN]);
        assert_eq!(loaded.quantization_report(), report);
    }

    #[test]
    fn non_finite_parameters_are_rejected() {
        let (mut features, bias, output, output_bias) = fixture::<2>();
        features[7] = f32::NAN;
        assert!(QuantizedNnueNetwork::<2>::from_float_parts(
            &features, &bias, &output, output_bias, 400.0, metadata(),
        ).unwrap_err().contains("finite"));
    }

    #[test]
    fn q14_integer_inference_stays_close_on_one_hundred_thousand_legal_positions() {
        const H: usize = 8;
        const POSITIONS: usize = 100_000;
        let (float, quantized) = make_network::<H>();
        let mut seed = 0x2026_1025_4b49_5241u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut seen = HashSet::with_capacity(POSITIONS);
        let mut board = Board::startpos();
        let mut sampled = 0usize;
        let mut maximum_error = 0i32;
        let mut total_error = 0u64;
        while sampled < POSITIONS {
            if seen.insert(board.hash()) {
                let float_score = float.evaluate(&board);
                let integer_score = quantized.evaluate(&board);
                let error = (float_score - integer_score).abs();
                maximum_error = maximum_error.max(error);
                total_error += error as u64;
                sampled += 1;
            }
            let mut legal = board.legal_moves();
            if legal.is_empty() || board.halfmove >= 160 {
                board = Board::startpos();
                continue;
            }
            let move_index = (next() as usize) % legal.len();
            board.make_move(legal.swap_remove(move_index));
        }
        assert_eq!(sampled, POSITIONS);
        let mean_error = total_error as f64 / sampled as f64;
        assert!(maximum_error <= 10, "maximum float/int parity error {maximum_error} cp exceeds 10 cp");
        assert!(mean_error <= 2.0, "mean float/int parity error {mean_error:.5} cp exceeds 2 cp");
        eprintln!("quantized_parity_positions={sampled} mae_cp={mean_error:.5} max_abs_error_cp={maximum_error}");
    }
}
