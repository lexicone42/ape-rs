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

    let mut count = 0u64;
    let mut errors = 0u64;
    let limit = (info.sample_rate as u64) * info.channels as u64;

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

/// Compare decoded APE samples against a WAV reference (ffmpeg-decoded).
#[test]
fn compare_to_wav_reference() {
    if !Path::new(TEST_APE).exists() || !Path::new(TEST_WAV).exists() {
        eprintln!("Skipping: test files not found");
        return;
    }

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

    let wav_data = std::fs::read(TEST_WAV).expect("Failed to read WAV");
    let wav_samples = parse_wav_samples(&wav_data, info.bits_per_sample);

    let compare_len = check_samples.min(ape_samples.len()).min(wav_samples.len());
    eprintln!("Comparing {compare_len} samples (APE has {}, WAV has {})",
        ape_samples.len(), wav_samples.len());

    assert_bit_exact(&ape_samples, &wav_samples, compare_len, "test.ape (mono c4000)");
}

// ── Compression level tests: mono ──────────────────────────────────

#[test]
fn mono_c1000_fast() {
    verify_ape_vs_wav(
        "tests/data/test_mono_c1000.ape",
        "tests/data/test_mono_reference.wav",
        1, 1000,
    );
}

#[test]
fn mono_c2000_normal() {
    verify_ape_vs_wav(
        "tests/data/test_mono_c2000.ape",
        "tests/data/test_mono_reference.wav",
        1, 2000,
    );
}

#[test]
fn mono_c3000_high() {
    verify_ape_vs_wav(
        "tests/data/test_mono_c3000.ape",
        "tests/data/test_mono_reference.wav",
        1, 3000,
    );
}

#[test]
fn mono_c4000_extra_high() {
    verify_ape_vs_wav(
        "tests/data/test_mono_c4000.ape",
        "tests/data/test_mono_reference.wav",
        1, 4000,
    );
}

#[test]
fn mono_c5000_insane() {
    verify_ape_vs_wav(
        "tests/data/test_mono_c5000.ape",
        "tests/data/test_mono_reference.wav",
        1, 5000,
    );
}

// ── Compression level tests: stereo ────────────────────────────────

#[test]
fn stereo_c1000_fast() {
    verify_ape_vs_wav(
        "tests/data/test_stereo_c1000.ape",
        "tests/data/test_stereo_reference.wav",
        2, 1000,
    );
}

#[test]
fn stereo_c2000_normal() {
    verify_ape_vs_wav(
        "tests/data/test_stereo_c2000.ape",
        "tests/data/test_stereo_reference.wav",
        2, 2000,
    );
}

#[test]
fn stereo_c3000_high() {
    verify_ape_vs_wav(
        "tests/data/test_stereo_c3000.ape",
        "tests/data/test_stereo_reference.wav",
        2, 3000,
    );
}

#[test]
fn stereo_c4000_extra_high() {
    verify_ape_vs_wav(
        "tests/data/test_stereo_c4000.ape",
        "tests/data/test_stereo_reference.wav",
        2, 4000,
    );
}

#[test]
fn stereo_c5000_insane() {
    verify_ape_vs_wav(
        "tests/data/test_stereo_c5000.ape",
        "tests/data/test_stereo_reference.wav",
        2, 5000,
    );
}

// ── Test helpers ───────────────────────────────────────────────────

/// Decode an APE file and verify bit-exact match against a WAV reference.
fn verify_ape_vs_wav(
    ape_path: &str,
    wav_path: &str,
    expected_channels: u16,
    expected_level: u16,
) {
    if !Path::new(ape_path).exists() || !Path::new(wav_path).exists() {
        eprintln!("Skipping: {ape_path} or {wav_path} not found");
        return;
    }

    let mut reader = ApeReader::open(ape_path).expect("Failed to open APE file");
    let info = reader.info().clone();

    assert_eq!(info.channels, expected_channels,
        "{ape_path}: expected {expected_channels} channels, got {}", info.channels);
    assert_eq!(info.compression_level, expected_level,
        "{ape_path}: expected level {expected_level}, got {}", info.compression_level);

    let expected_total = info.total_samples as usize;
    let mut ape_samples = Vec::with_capacity(expected_total);
    for result in reader.samples() {
        ape_samples.push(result.unwrap_or_else(|e| {
            panic!("{ape_path}: decode error at sample {}: {e}", ape_samples.len())
        }));
    }

    assert_eq!(ape_samples.len(), expected_total,
        "{ape_path}: decoded {} samples, expected {expected_total}", ape_samples.len());

    let wav_data = std::fs::read(wav_path).expect("Failed to read WAV");
    let wav_samples = parse_wav_samples(&wav_data, info.bits_per_sample);

    let compare_len = expected_total.min(wav_samples.len());
    let label = format!("{ape_path} ({}ch c{})", info.channels, info.compression_level);
    assert_bit_exact(&ape_samples, &wav_samples, compare_len, &label);
}

/// Assert that two sample arrays are bit-exact, with diagnostics on failure.
fn assert_bit_exact(ape: &[i32], wav: &[i32], len: usize, label: &str) {
    let mut mismatches = 0u64;
    let mut max_diff = 0i64;
    let mut first_mismatch = None;

    for i in 0..len {
        let diff = (ape[i] as i64 - wav[i] as i64).abs();
        if diff > 0 {
            if first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
            mismatches += 1;
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    if mismatches == 0 {
        eprintln!("{label}: PASS — {len} samples bit-exact");
    } else {
        let idx = first_mismatch.unwrap();
        eprintln!("{label}: FAIL — {mismatches}/{len} mismatches, max_diff={max_diff}");
        eprintln!("  First mismatch at sample {idx}: APE={}, WAV={}", ape[idx], wav[idx]);
    }

    assert_eq!(mismatches, 0,
        "{label}: {mismatches} samples differ (max_diff={max_diff})");
}

/// Parse PCM samples from a WAV file (16-bit or 24-bit).
fn parse_wav_samples(data: &[u8], bits_per_sample: u16) -> Vec<i32> {
    let mut pos = 12; // Skip RIFF header
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
                        if raw & 0x800000 != 0 { raw | !0xFFFFFF } else { raw }
                    })
                    .collect(),
                _ => panic!("Unsupported bits_per_sample: {bits_per_sample}"),
            };
        }

        pos += chunk_size;
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }
    panic!("No 'data' chunk found in WAV file");
}
