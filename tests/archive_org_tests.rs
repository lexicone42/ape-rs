//! Tests against real-world APE files downloaded from archive.org.
//!
//! These tests are skipped if the files aren't present. To run them:
//!   1. Download APE files to /tmp/ape_test/
//!   2. cargo test --release -- archive

use ape_rs::ApeReader;
use std::path::Path;
use std::process::Command;

const TEST_FILES: &[&str] = &[
    "/tmp/ape_test/breath_03.ape",
    "/tmp/ape_test/pavement_intro.ape",
    "/tmp/ape_test/pavement_cut_your_hair.ape",
];

#[test]
fn archive_org_decode_all() {
    let mut tested = 0;
    for path in TEST_FILES {
        if !Path::new(path).exists() {
            eprintln!("Skipping (not found): {path}");
            continue;
        }
        test_one_file(path);
        tested += 1;
    }
    if tested == 0 {
        eprintln!("No archive.org test files found in /tmp/ape_test/");
    } else {
        eprintln!("\nAll {tested} archive.org files decoded and verified!");
    }
}

fn test_one_file(ape_path: &str) {
    eprintln!("\n--- {ape_path} ---");

    // Step 1: Decode with ape-rs
    let mut reader = ApeReader::open(ape_path)
        .unwrap_or_else(|e| panic!("ape-rs failed to open {ape_path}: {e}"));
    let info = reader.info().clone();
    eprintln!(
        "  {}ch, {}Hz, {}bit, level {}, {} total samples",
        info.channels, info.sample_rate, info.bits_per_sample,
        info.compression_level, info.total_samples
    );

    let mut ape_samples = Vec::with_capacity(info.total_samples as usize);
    for (i, result) in reader.samples().enumerate() {
        match result {
            Ok(s) => ape_samples.push(s),
            Err(e) => panic!("ape-rs decode error at sample {i}: {e}"),
        }
    }
    assert_eq!(
        ape_samples.len() as u64, info.total_samples,
        "Sample count mismatch: got {}, expected {}",
        ape_samples.len(), info.total_samples
    );
    eprintln!("  ape-rs: decoded {} samples OK", ape_samples.len());

    // Step 2: Decode with ffmpeg to WAV for reference
    let wav_path = format!("{ape_path}.reference.wav");
    let ffmpeg = Command::new("ffmpeg")
        .args(["-y", "-i", ape_path, "-f", "wav", "-acodec"])
        .arg(match info.bits_per_sample {
            8 => "pcm_u8",
            16 => "pcm_s16le",
            24 => "pcm_s24le",
            _ => "pcm_s16le",
        })
        .arg(&wav_path)
        .output();

    match ffmpeg {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            eprintln!("  ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr));
            eprintln!("  Skipping comparison (ape-rs decode succeeded on its own)");
            return;
        }
        Err(_) => {
            eprintln!("  ffmpeg not available, skipping comparison");
            return;
        }
    }

    // Step 3: Compare
    let wav_data = std::fs::read(&wav_path).expect("read reference wav");
    let wav_samples = parse_wav_samples(&wav_data, info.bits_per_sample);
    let _ = std::fs::remove_file(&wav_path); // clean up

    let compare_len = ape_samples.len().min(wav_samples.len());
    let mut mismatches = 0u64;
    let mut max_diff = 0i64;
    let mut first_mismatch = None;

    for i in 0..compare_len {
        let diff = (ape_samples[i] as i64 - wav_samples[i] as i64).abs();
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
        eprintln!("  PASS: {compare_len} samples bit-exact vs ffmpeg");
    } else {
        let idx = first_mismatch.unwrap();
        eprintln!("  FAIL: {mismatches}/{compare_len} mismatches (max_diff={max_diff})");
        eprintln!("    First mismatch at {idx}: ape-rs={}, ffmpeg={}", ape_samples[idx], wav_samples[idx]);
    }

    assert_eq!(mismatches, 0,
        "{ape_path}: {mismatches} samples differ vs ffmpeg (max_diff={max_diff})");
}

fn parse_wav_samples(data: &[u8], bits_per_sample: u16) -> Vec<i32> {
    let mut pos = 12;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
        ]) as usize;
        pos += 8;

        if chunk_id == b"data" {
            let sample_data = &data[pos..pos + chunk_size.min(data.len() - pos)];
            return match bits_per_sample {
                16 => sample_data.chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
                    .collect(),
                24 => sample_data.chunks_exact(3)
                    .map(|c| {
                        let raw = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
                        if raw & 0x800000 != 0 { raw | !0xFFFFFF } else { raw }
                    })
                    .collect(),
                _ => panic!("Unsupported bits_per_sample: {bits_per_sample}"),
            };
        }

        pos += chunk_size;
        if chunk_size % 2 != 0 { pos += 1; }
    }
    panic!("No 'data' chunk found in WAV");
}
