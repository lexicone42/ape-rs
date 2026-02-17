use ape_rs::ApeReader;
use std::path::Path;

const TEST_APE: &str = "tests/data/test.ape";
const TEST_WAV: &str = "tests/data/test_reference.wav";

#[test]
fn parse_header() {
    if !Path::new(TEST_APE).exists() {
        eprintln!("Skipping: test file not found at {TEST_APE}");
        return;
    }

    let reader = ApeReader::open(TEST_APE).expect("Failed to open APE file");
    let info = reader.info();

    eprintln!("APE info:");
    eprintln!("  Format version: {}", info.format_version);
    eprintln!("  Channels: {}", info.channels);
    eprintln!("  Sample rate: {}", info.sample_rate);
    eprintln!("  Bits per sample: {}", info.bits_per_sample);
    eprintln!("  Total samples: {}", info.total_samples);
    eprintln!("  Compression level: {}", info.compression_level);

    // Basic sanity checks
    assert!(info.sample_rate == 44100 || info.sample_rate == 48000,
        "Unexpected sample rate: {}", info.sample_rate);
    assert!(info.channels == 1 || info.channels == 2,
        "Unexpected channels: {}", info.channels);
    assert!(info.bits_per_sample == 16 || info.bits_per_sample == 24,
        "Unexpected bits per sample: {}", info.bits_per_sample);
    assert!(info.format_version >= 3990,
        "Expected v3.99+, got {}", info.format_version);
    assert!(info.total_samples > 0, "No samples?");
}

#[test]
fn decode_first_frame() {
    if !Path::new(TEST_APE).exists() {
        eprintln!("Skipping: test file not found at {TEST_APE}");
        return;
    }

    let mut reader = ApeReader::open(TEST_APE).expect("Failed to open APE file");
    let info = reader.info().clone();

    // Try to decode some samples
    let mut count = 0u64;
    let mut errors = 0u64;
    let limit = (info.sample_rate as u64) * info.channels as u64; // ~1 second

    for result in reader.samples() {
        match result {
            Ok(_sample) => count += 1,
            Err(e) => {
                if errors == 0 {
                    eprintln!("First decode error at sample {count}: {e}");
                }
                errors += 1;
                break;
            }
        }
        if count >= limit {
            break;
        }
    }

    eprintln!("Decoded {count} samples, {errors} errors");
    assert!(count > 0 || errors > 0, "Iterator returned nothing");
}

#[test]
fn full_decode_sample_count() {
    if !Path::new(TEST_APE).exists() {
        eprintln!("Skipping: test file not found at {TEST_APE}");
        return;
    }

    let mut reader = ApeReader::open(TEST_APE).expect("Failed to open APE file");
    let expected = reader.info().total_samples;

    let mut count = 0u64;
    for result in reader.samples() {
        match result {
            Ok(_) => count += 1,
            Err(e) => {
                eprintln!("Decode error at sample {count}: {e}");
                break;
            }
        }
    }

    eprintln!("Decoded {count}/{expected} samples");
    assert_eq!(count, expected, "Sample count mismatch");
}

#[test]
fn dump_first_samples() {
    if !Path::new(TEST_APE).exists() || !Path::new(TEST_WAV).exists() {
        return;
    }
    let mut reader = ApeReader::open(TEST_APE).expect("open");
    let info = reader.info().clone();
    let wav_data = std::fs::read(TEST_WAV).expect("read wav");
    let wav_samples = parse_wav_samples(&wav_data, info.bits_per_sample);

    // Decode first 441000 samples (10 seconds at 44100 mono)
    let mut ape_samples = Vec::new();
    for result in reader.samples() {
        if ape_samples.len() >= 441000 { break; }
        ape_samples.push(result.unwrap());
    }

    // Find first non-zero WAV sample
    let first_nonzero = wav_samples.iter().position(|&s| s != 0).unwrap_or(0);
    eprintln!("First non-zero WAV sample at index {first_nonzero}");

    // Show samples around that point
    let start = first_nonzero.saturating_sub(5);
    eprintln!("{:>8} {:>12} {:>12} {:>12}", "idx", "APE", "WAV", "diff");
    for i in start..start + 30 {
        if i >= ape_samples.len() { break; }
        let a = ape_samples[i];
        let w = wav_samples[i];
        eprintln!("{:>8} {:>12} {:>12} {:>12}", i, a, w, a as i64 - w as i64);
    }

    // Also show first mismatch
    let first_mismatch = (0..ape_samples.len().min(wav_samples.len()))
        .find(|&i| ape_samples[i] != wav_samples[i]);
    if let Some(idx) = first_mismatch {
        eprintln!("\nFirst mismatch at index {idx}:");
        let s = idx.saturating_sub(3);
        for i in s..s + 10 {
            if i >= ape_samples.len() { break; }
            let a = ape_samples[i];
            let w = wav_samples[i];
            let marker = if a != w { " <--" } else { "" };
            eprintln!("{:>8} {:>12} {:>12} {:>12}{}", i, a, w, a as i64 - w as i64, marker);
        }
    } else {
        eprintln!("First {} samples match perfectly!", ape_samples.len());
    }
}

/// Compare decoded APE samples against a WAV reference (ffmpeg-decoded).
/// Only checks the first N samples to keep test fast.
#[test]
fn compare_to_wav_reference() {
    if !Path::new(TEST_APE).exists() || !Path::new(TEST_WAV).exists() {
        eprintln!("Skipping: test files not found");
        return;
    }

    // Decode ALL samples from APE
    let mut reader = ApeReader::open(TEST_APE).expect("Failed to open APE file");
    let info = reader.info().clone();
    let check_samples = info.total_samples as usize;

    let mut ape_samples = Vec::with_capacity(check_samples);
    for result in reader.samples() {
        if ape_samples.len() >= check_samples {
            break;
        }
        ape_samples.push(result.expect("APE decode error"));
    }

    // Read WAV reference
    let wav_data = std::fs::read(TEST_WAV).expect("Failed to read WAV");
    let wav_samples = parse_wav_samples(&wav_data, info.bits_per_sample);

    let compare_len = check_samples.min(ape_samples.len()).min(wav_samples.len());
    eprintln!("Comparing {compare_len} samples (APE has {}, WAV has {})",
        ape_samples.len(), wav_samples.len());

    let mut max_diff = 0i64;
    let mut total_diff = 0i64;
    let mut mismatches = 0u64;

    for i in 0..compare_len {
        let diff = (ape_samples[i] as i64 - wav_samples[i] as i64).abs();
        if diff > 0 {
            mismatches += 1;
            if diff > max_diff {
                max_diff = diff;
            }
            total_diff += diff;
        }
    }

    let avg_diff = if mismatches > 0 {
        total_diff as f64 / mismatches as f64
    } else {
        0.0
    };

    eprintln!("Results: {mismatches}/{compare_len} samples differ");
    eprintln!("  Max diff: {max_diff}");
    eprintln!("  Avg diff (of mismatches): {avg_diff:.2}");

    if mismatches == 0 {
        eprintln!("PERFECT: bit-exact match!");
    }

    // Lossless codec — samples must be bit-exact
    assert_eq!(mismatches, 0,
        "Lossless decode mismatch: {mismatches} samples differ, max_diff={max_diff}");
}

/// Parse 16-bit PCM samples from a WAV file (minimal parser, assumes standard format).
fn parse_wav_samples(data: &[u8], bits_per_sample: u16) -> Vec<i32> {
    // Find "data" chunk
    let mut pos = 12; // Skip RIFF header (12 bytes)
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
        ]) as usize;
        pos += 8;

        if chunk_id == b"data" {
            let sample_data = &data[pos..pos + chunk_size];
            return match bits_per_sample {
                16 => sample_data
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
                    .collect(),
                24 => sample_data
                    .chunks_exact(3)
                    .map(|c| {
                        let raw = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
                        // Sign-extend from 24-bit
                        if raw & 0x800000 != 0 {
                            raw | !0xFFFFFF
                        } else {
                            raw
                        }
                    })
                    .collect(),
                _ => panic!("Unsupported bits_per_sample: {bits_per_sample}"),
            };
        }

        pos += chunk_size;
        // WAV chunks are word-aligned
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }
    panic!("No 'data' chunk found in WAV file");
}
