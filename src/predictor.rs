//! Linear predictor for APE decoding.
//!
//! The predictor reconstructs audio samples from the filtered residuals.
//! It uses two adaptive filters per channel:
//!   - Filter A: 4 coefficients, uses delayed samples
//!   - Filter B: 5 coefficients, uses cross-channel correlation (stereo)
//!
//! For stereo, the X (left-ish) and Y (right-ish) channels are decoded
//! separately, then combined via inverse joint-stereo decorrelation.

const HISTORY_SIZE: usize = 512;
const PREDICTOR_ORDER: usize = 8;
const PREDICTOR_SIZE: usize = 50;

// Delay buffer offsets for the Y channel (processed first in stereo)
const YDELAYA: usize = 18 + PREDICTOR_ORDER * 4; // 50
const YDELAYB: usize = 18 + PREDICTOR_ORDER * 3; // 42
// Delay buffer offsets for the X channel
const XDELAYA: usize = 18 + PREDICTOR_ORDER * 2; // 34
const XDELAYB: usize = 18 + PREDICTOR_ORDER;     // 26

// Adaptation coefficient offsets
const YADAPTCOEFFSA: usize = 18;
const XADAPTCOEFFSA: usize = 14;
const YADAPTCOEFFSB: usize = 10;
const XADAPTCOEFFSB: usize = 5;

/// Per-channel predictor state.
#[derive(Clone)]
struct ChannelPredictor {
    /// Last prediction value (used for first-order delta).
    last_a: i32,
    /// Filter A output (inner prediction).
    filter_a: i64,
    /// Filter B output (cross-channel prediction).
    filter_b: i64,
    /// Filter A coefficients (4 taps).
    coeffs_a: [i32; 4],
    /// Filter B coefficients (5 taps).
    coeffs_b: [i32; 5],
}

impl ChannelPredictor {
    fn new() -> Self {
        ChannelPredictor {
            last_a: 0,
            filter_a: 0,
            filter_b: 0,
            coeffs_a: [0; 4],
            coeffs_b: [0; 5],
        }
    }

    fn reset(&mut self) {
        self.last_a = 0;
        self.filter_a = 0;
        self.filter_b = 0;
        self.coeffs_a = [0; 4];
        self.coeffs_b = [0; 5];
    }
}

/// The APE predictor — handles both mono and stereo.
pub struct Predictor {
    /// History buffer for delay lines. Shared across channels.
    buf: Vec<i32>,
    /// Current position in the history buffer.
    buf_pos: usize,
    /// Per-channel predictor state (up to 2 channels).
    channels: [ChannelPredictor; 2],
}

impl Predictor {
    pub fn new() -> Self {
        Predictor {
            buf: vec![0i32; HISTORY_SIZE + PREDICTOR_SIZE],
            buf_pos: HISTORY_SIZE,
            channels: [ChannelPredictor::new(), ChannelPredictor::new()],
        }
    }

    /// Reset predictor state at frame boundaries.
    pub fn reset(&mut self) {
        self.buf.fill(0);
        self.buf_pos = HISTORY_SIZE;
        for ch in &mut self.channels {
            ch.reset();
        }
    }

    /// Decode a mono sample: apply predictor inverse to get the output sample.
    pub fn decode_mono(&mut self, input: i32) -> i32 {
        let output = self.update_filter(0, input, YDELAYA, YADAPTCOEFFSA, None);

        // Wrap history buffer
        self.buf_pos += 1;
        if self.buf_pos >= HISTORY_SIZE + PREDICTOR_SIZE - 1 {
            self.wrap_history();
        }

        output
    }

    /// Decode a stereo sample pair.
    /// Takes filtered residuals for both channels, returns (left, right).
    pub fn decode_stereo(&mut self, input_y: i32, input_x: i32) -> (i32, i32) {
        // Decode Y channel (uses A filter with Y delays, B filter with Y-cross delays)
        let decoded_y = self.update_filter(
            0,
            input_y,
            YDELAYA,
            YADAPTCOEFFSA,
            Some((YDELAYB, YADAPTCOEFFSB)),
        );

        // Decode X channel (uses A filter with X delays, B filter with X-cross delays)
        let decoded_x = self.update_filter(
            1,
            input_x,
            XDELAYA,
            XADAPTCOEFFSA,
            Some((XDELAYB, XADAPTCOEFFSB)),
        );

        // Advance buffer position
        self.buf_pos += 1;
        if self.buf_pos >= HISTORY_SIZE + PREDICTOR_SIZE - 1 {
            self.wrap_history();
        }

        // Inverse channel decorrelation: Y is mid-side encoded
        // left = X - Y/2, right = left + Y  (or similar)
        let left = decoded_x - (decoded_y / 2);
        let right = left + decoded_y;

        (left, right)
    }

    /// Apply the prediction filter to one sample on the given channel.
    fn update_filter(
        &mut self,
        ch: usize,
        input: i32,
        delay_a: usize,
        _adapt_a: usize,
        b_params: Option<(usize, usize)>,
    ) -> i32 {
        let bp = self.buf_pos;
        let pred = &mut self.channels[ch];

        // Compute deltas from the delay line for filter A
        let da0 = self.buf[bp - delay_a] as i64;
        let da1 = self.buf[bp - delay_a + 1] as i64;
        let da2 = self.buf[bp - delay_a + 2] as i64;
        let da3 = self.buf[bp - delay_a + 3] as i64;

        // Filter A: prediction from own-channel history
        let prediction_a = (pred.coeffs_a[0] as i64 * da0
            + pred.coeffs_a[1] as i64 * (da0 - da1)
            + pred.coeffs_a[2] as i64 * (da1 - da2)
            + pred.coeffs_a[3] as i64 * (da2 - da3))
            >> 9;

        // Filter B: cross-channel or secondary prediction
        let prediction_b = if let Some((delay_b, _adapt_b)) = b_params {
            let db0 = self.buf[bp - delay_b] as i64;
            let db1 = self.buf[bp - delay_b + 1] as i64;
            let db2 = self.buf[bp - delay_b + 2] as i64;
            let db3 = self.buf[bp - delay_b + 3] as i64;
            let db4 = self.buf[bp - delay_b + 4] as i64;

            (pred.coeffs_b[0] as i64 * db0
                + pred.coeffs_b[1] as i64 * (db0 - db1)
                + pred.coeffs_b[2] as i64 * (db1 - db2)
                + pred.coeffs_b[3] as i64 * (db2 - db3)
                + pred.coeffs_b[4] as i64 * (db3 - db4))
                >> 9
        } else {
            0
        };

        // Combine: last_a carries the previous sample (first-order prediction)
        let current_a = pred.last_a;
        pred.filter_a = prediction_a;
        pred.filter_b = prediction_b;

        let predicted = current_a
            + ((pred.filter_a + pred.filter_b + 1) >> 1) as i32;
        let output = predicted + input;

        // Adapt filter A coefficients based on sign of error
        adapt_coefficients(
            &mut pred.coeffs_a,
            &[da0, da0 - da1, da1 - da2, da2 - da3],
            input,
        );

        // Adapt filter B coefficients
        if let Some((delay_b, _)) = b_params {
            let db0 = self.buf[bp - delay_b] as i64;
            let db1 = self.buf[bp - delay_b + 1] as i64;
            let db2 = self.buf[bp - delay_b + 2] as i64;
            let db3 = self.buf[bp - delay_b + 3] as i64;
            let db4 = self.buf[bp - delay_b + 4] as i64;

            adapt_coefficients(
                &mut pred.coeffs_b,
                &[db0, db0 - db1, db1 - db2, db2 - db3, db3 - db4],
                input,
            );
        }

        // Update state
        pred.last_a = output;

        // Store in history buffer for future predictions
        self.buf[bp] = output;

        output
    }

    /// Wrap the history buffer when it's nearly full.
    fn wrap_history(&mut self) {
        let src = PREDICTOR_SIZE;
        let count = HISTORY_SIZE;
        // Move the last HISTORY_SIZE entries to the start
        for i in 0..count {
            self.buf[i] = self.buf[i + src];
        }
        // Clear the rest
        for i in count..self.buf.len() {
            self.buf[i] = 0;
        }
        self.buf_pos = HISTORY_SIZE;
    }
}

/// Adapt filter coefficients using the sign of the error and the sign of each delay.
fn adapt_coefficients(coeffs: &mut [i32], delays: &[i64], error: i32) {
    if error > 0 {
        for (c, d) in coeffs.iter_mut().zip(delays.iter()) {
            if *d < 0 {
                *c -= 1;
            } else if *d > 0 {
                *c += 1;
            }
        }
    } else if error < 0 {
        for (c, d) in coeffs.iter_mut().zip(delays.iter()) {
            if *d < 0 {
                *c += 1;
            } else if *d > 0 {
                *c -= 1;
            }
        }
    }
}
