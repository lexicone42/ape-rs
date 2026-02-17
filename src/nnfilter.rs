//! NNFilter — Adaptive sign-LMS FIR filter for APE decoding.
//!
//! The NNFilter is the core complexity of Monkey's Audio. It uses sign-based
//! LMS (Least Mean Squares) adaptation where coefficients are updated based on
//! the sign of the error and the sign of each history sample.
//!
//! Filter parameters vary by compression level:
//!   Level 1000 (Fast):       no filter
//!   Level 2000 (Normal):     1 stage, 16 taps, fracbits=11
//!   Level 3000 (High):       1 stage, 64 taps, fracbits=11
//!   Level 4000 (Extra High): 2 stages, 32+256 taps, fracbits=10,13
//!   Level 5000 (Insane):     3 stages, 16+256+1280 taps, fracbits=11,13,15

/// Maximum number of filter stages.
pub const MAX_STAGES: usize = 3;

/// Filter orders (taps) for each compression level, per stage.
/// Index = (compression_level / 1000) - 1, so 0=Fast, 1=Normal, etc.
pub const FILTER_ORDERS: [[u16; MAX_STAGES]; 5] = [
    [0, 0, 0],       // Fast (1000) — no filter
    [16, 0, 0],      // Normal (2000)
    [64, 0, 0],      // High (3000)
    [32, 256, 0],    // Extra High (4000)
    [16, 256, 1280], // Insane (5000)
];

/// Fractional bits (right-shift for rounding) per stage per compression level.
pub const FILTER_FRACBITS: [[u8; MAX_STAGES]; 5] = [
    [0, 0, 0],
    [11, 0, 0],
    [11, 0, 0],
    [10, 13, 0],
    [11, 13, 15],
];

/// One stage of the adaptive FIR filter.
#[derive(Clone)]
pub struct NNFilterStage {
    /// Number of taps (filter order).
    order: usize,
    /// Fractional bits for rounding the dot product.
    fracbits: u8,
    /// Filter coefficients (adapted via sign-LMS).
    coeffs: Vec<i16>,
    /// Adaptation coefficients: sign(history[i]) for each tap, pre-computed.
    adapt_coeffs: Vec<i16>,
    /// Delay line / history buffer (ring buffer, 2× order for no-wrap access).
    history: Vec<i16>,
    /// Current position in the history buffer.
    delay_pos: usize,
    /// Running average for initial adaptation step size.
    avg: i32,
}

impl NNFilterStage {
    /// Create a new filter stage with the given order and fracbits.
    pub fn new(order: usize, fracbits: u8) -> Self {
        NNFilterStage {
            order,
            fracbits,
            coeffs: vec![0i16; order],
            adapt_coeffs: vec![0i16; order],
            history: vec![0i16; 2 * order],
            delay_pos: order,
            avg: 0,
        }
    }

    /// Reset the filter state (called at frame boundaries or when history wraps).
    pub fn reset(&mut self) {
        self.coeffs.fill(0);
        self.adapt_coeffs.fill(0);
        self.history.fill(0);
        self.delay_pos = self.order;
        self.avg = 0;
    }

    /// Apply the filter to one sample (decompress direction).
    ///
    /// In decompress: output = input + dot_product(coeffs, history) >> fracbits
    /// Then adapt coefficients based on sign of error.
    pub fn decompress(&mut self, input: i32) -> i32 {
        if self.order == 0 {
            return input;
        }

        let delay = self.delay_pos;
        let order = self.order;

        // Compute dot product: sum(coeffs[i] * history[delay - order + i])
        let mut sum: i64 = 0;
        let hist_start = delay - order;
        for i in 0..order {
            sum += self.coeffs[i] as i64 * self.history[hist_start + i] as i64;
        }

        // Round and add to input
        let filtered = (sum >> self.fracbits) as i32;
        let output = input + filtered;

        // Compute the sign of the residual (input) for adaptation
        let error_sign = if input > 0 {
            1i16
        } else if input < 0 {
            -1i16
        } else {
            0i16
        };

        // Adapt coefficients: coeffs[i] += error_sign * adapt_coeffs[i]
        // where adapt_coeffs[i] = sign(history[i])
        if error_sign != 0 {
            for i in 0..order {
                self.coeffs[i] = self.coeffs[i]
                    .saturating_add(error_sign.saturating_mul(self.adapt_coeffs[hist_start + i]));
            }

            // Also adapt the first coefficient based on running average
            // This is a version-specific behavior for v3.98+
            let adapt0 = if input > 0 {
                (((input as i64) << 16) >> (25 + self.fracbits as i64)) as i16
            } else {
                -((((-input as i64) << 16) >> (25 + self.fracbits as i64)) as i16)
            };
            self.coeffs[0] = self.coeffs[0].saturating_add(adapt0);
        }

        // Store output in history buffer and update adapt_coeffs
        self.history[delay] = clamp_i16(output);
        self.adapt_coeffs[delay] = if output > 0 {
            1
        } else if output < 0 {
            -1
        } else {
            0
        };
        self.delay_pos += 1;

        // Wrap history buffer when we reach the end
        if self.delay_pos >= 2 * order {
            // Copy the last `order` entries to the beginning
            let src_start = order;
            for i in 0..order {
                self.history[i] = self.history[src_start + i];
                self.adapt_coeffs[i] = self.adapt_coeffs[src_start + i];
            }
            self.delay_pos = order;
        }

        output
    }
}

/// Multi-stage NNFilter — cascades 0-3 filter stages.
pub struct NNFilter {
    stages: Vec<NNFilterStage>,
}

impl NNFilter {
    /// Create an NNFilter for the given compression level.
    /// `fset` = (compression_level / 1000) - 1, range 0..5.
    pub fn new(fset: usize) -> Self {
        let mut stages = Vec::new();
        for s in 0..MAX_STAGES {
            let order = FILTER_ORDERS[fset][s] as usize;
            if order > 0 {
                let fracbits = FILTER_FRACBITS[fset][s];
                stages.push(NNFilterStage::new(order, fracbits));
            }
        }
        NNFilter { stages }
    }

    /// Reset all filter stages.
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }

    /// Apply all filter stages to decompress one sample.
    /// Stages are applied in reverse order for decompression.
    pub fn decompress(&mut self, mut value: i32) -> i32 {
        // Apply filters in forward order (last encoded = first decoded)
        for stage in self.stages.iter_mut().rev() {
            value = stage.decompress(value);
        }
        value
    }

    /// Number of active stages.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

/// Clamp an i32 to i16 range.
fn clamp_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
