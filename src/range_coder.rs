//! Arithmetic range coder for APE entropy decoding.
//!
//! Based on the format specification for Monkey's Audio v3.99+ (version 3990).
//! The range coder uses a 32-bit range with bottom-value normalization.

// ── Range coder constants ────────────────────────────────────────────

const CODE_BITS: u32 = 32;
const TOP_VALUE: u32 = 1u32 << (CODE_BITS - 1);
const BOTTOM_VALUE: u32 = TOP_VALUE >> 8;
const EXTRA_BITS: u32 = ((CODE_BITS - 2) % 8) + 1; // = 7

/// Number of symbols in the frequency model (0..=20 + overflow).
const MODEL_ELEMENTS: usize = 22;

/// Cumulative frequency table for v3.98+ (version >= 3980).
const COUNTS_3980: [u16; MODEL_ELEMENTS] = [
    0, 19578, 36160, 48417, 56323, 60899, 63265, 64435, 64971, 65232, 65351,
    65416, 65447, 65466, 65476, 65482, 65485, 65488, 65490, 65491, 65492,
    65493,
];

/// Differential (per-symbol width) frequency table for v3.98+.
const COUNTS_DIFF_3980: [u16; MODEL_ELEMENTS - 1] = [
    19578, 16582, 12257, 7906, 4576, 2366, 1170, 536, 261, 119, 65, 31, 19,
    10, 6, 3, 3, 2, 1, 1, 1,
];

// ── Rice state ───────────────────────────────────────────────────────

/// Adaptive parameter for the Golomb-Rice–like pivot computation.
#[derive(Debug, Clone)]
pub struct RiceState {
    pub k: u32,
    pub ksum: u32,
}

impl RiceState {
    pub fn new() -> Self {
        // Initial state: k=10, ksum = (1 << k) * 16 = 16384
        // This gives an initial pivot of ksum >> 5 = 512
        RiceState {
            k: 10,
            ksum: 1 << 14, // 16384
        }
    }

    /// Update the adaptive parameters after decoding a value.
    pub fn update(&mut self, x: u32) {
        // Exponential moving average: ksum += (x+1)/2 - (ksum+16)>>5
        let add = ((x + 1) / 2).min(u32::MAX - self.ksum);
        self.ksum += add;
        let sub = (self.ksum + 16) >> 5;
        self.ksum = self.ksum.saturating_sub(sub);

        // Adapt k: ensure ksum is in [2^(k+4), 2^(k+5))
        if self.k > 0 && self.ksum < (1u32 << (self.k + 4)) {
            self.k -= 1;
        } else if self.ksum >= (1u32 << (self.k + 5)) && self.k < 24 {
            self.k += 1;
        }
    }

    /// The expected magnitude of the next value.
    pub fn pivot(&self) -> u32 {
        (self.ksum >> 5).max(1)
    }
}

// ── Range coder ──────────────────────────────────────────────────────

/// Byte-level range coder for entropy decoding.
pub struct RangeCoder<'a> {
    data: &'a [u8],
    pos: usize,
    pub low: u32,
    pub range: u32,
    help: u32,
}

impl<'a> RangeCoder<'a> {
    /// Initialize the range coder from a byte slice (compressed frame data).
    pub fn new(data: &'a [u8]) -> Self {
        let mut rc = RangeCoder {
            data,
            pos: 0,
            low: 0,
            range: 1u32 << EXTRA_BITS,
            help: 0,
        };

        // Read first byte and extract top EXTRA_BITS bits
        let first = rc.read_byte();
        rc.low = (first as u32) >> (8 - EXTRA_BITS);

        // Normalize to fill the range register
        rc.normalize();
        rc
    }

    /// Read the next byte, returning 0 on EOF.
    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    /// Renormalize: expand range by reading bytes until range > BOTTOM_VALUE.
    fn normalize(&mut self) {
        while self.range <= BOTTOM_VALUE {
            let byte = self.read_byte();
            self.low = (self.low << 8) | (byte as u32);
            self.range <<= 8;
        }
    }

    /// Decode a uniform value in [0, 2^shift) using the top `shift` bits of range.
    fn decode_culshift(&mut self, shift: u32) -> u32 {
        self.help = self.range >> shift;
        let value = self.low / self.help;
        value.min((1u32 << shift) - 1)
    }

    /// Decode a uniform value in [0, tot_f) using frequency `tot_f`.
    fn decode_culfreq(&mut self, tot_f: u32) -> u32 {
        self.help = self.range / tot_f;
        let value = self.low / self.help;
        value.min(tot_f - 1)
    }

    /// Update range coder state after decoding a symbol.
    /// `lt_f` = cumulative frequency of symbols before this one.
    /// `sy_f` = frequency of this symbol.
    fn decode_update(&mut self, lt_f: u32, sy_f: u32) {
        self.low -= self.help * lt_f;
        self.range = self.help * sy_f;
        self.normalize();
    }

    /// Decode a symbol from the frequency model (counts_3980).
    /// Returns the symbol index (0..=20 for normal symbols, 21+ for overflow).
    fn get_symbol(&mut self) -> u32 {
        let cf = self.decode_culshift(16) as u16;

        // Check for overflow escape (last bucket)
        if cf > 65492 {
            // Overflow: symbol index beyond the model
            self.decode_update(
                COUNTS_3980[MODEL_ELEMENTS - 1] as u32,
                65536 - COUNTS_3980[MODEL_ELEMENTS - 1] as u32,
            );
            return u32::MAX; // sentinel for "overflow beyond model"
        }

        // Binary search for the symbol in the cumulative frequency table
        let mut lo = 0usize;
        let mut hi = MODEL_ELEMENTS - 1;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if COUNTS_3980[mid] <= cf {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        self.decode_update(
            COUNTS_3980[lo] as u32,
            COUNTS_DIFF_3980[lo] as u32,
        );
        lo as u32
    }

    /// Decode a single signed audio value using the APE v3.99 entropy scheme.
    pub fn decode_value(&mut self, rice: &mut RiceState) -> i32 {
        let pivot = rice.pivot();

        let (base, overflow);

        if pivot < 65536 {
            // Common case: small pivot, decode base directly
            self.help = self.range / pivot;
            let b = (self.low / self.help).min(pivot - 1);
            self.low -= self.help * b;
            self.range = self.help;
            self.normalize();
            base = b;

            // Decode overflow using frequency model
            overflow = self.get_overflow();
        } else {
            // Large pivot: decode overflow first, then base in two parts
            overflow = self.get_overflow();

            // Decode base in [0, pivot) using split approach
            let pivot_bits = 32 - pivot.leading_zeros(); // ilog2(pivot) + 1
            let shift = pivot_bits - 16;

            // High 16 bits
            self.help = self.range >> 16;
            let base_high = (self.low / self.help).min(65535);
            self.low -= self.help * base_high;
            self.range = self.help;
            self.normalize();

            // Low `shift` bits
            self.help = self.range >> shift;
            let base_low = (self.low / self.help).min((1u32 << shift) - 1);
            self.low -= self.help * base_low;
            self.range = self.help;
            self.normalize();

            base = (base_high << shift) | base_low;
        }

        let x = base + overflow * pivot;
        rice.update(x);

        // Zigzag decode: odd → positive, even → non-positive
        if x & 1 != 0 {
            ((x >> 1) + 1) as i32
        } else {
            -((x >> 1) as i32)
        }
    }

    /// Decode the overflow count using the frequency model, handling escapes.
    fn get_overflow(&mut self) -> u32 {
        let sym = self.get_symbol();

        if sym == u32::MAX {
            // Escaped overflow — decode the actual count
            // The overflow is > 20; decode additional bits
            let overflow_high = self.get_symbol();
            if overflow_high == u32::MAX {
                // Double escape — very rare, decode with explicit bit count
                let bits = self.decode_culshift(5);
                self.decode_update(bits, 1);
                let value = self.decode_culshift(bits);
                self.decode_update(value, 1);
                return value + (MODEL_ELEMENTS as u32 - 1);
            }
            overflow_high + (MODEL_ELEMENTS as u32 - 1)
        } else {
            sym
        }
    }

    /// Current byte position in the data.
    pub fn position(&self) -> usize {
        self.pos
    }
}
