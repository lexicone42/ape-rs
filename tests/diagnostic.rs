use ape_rs::ApeReader;
use std::path::Path;

const TEST_APE: &str = "tests/data/test.ape";

#[test]
fn frame_boundaries() {
    if !Path::new(TEST_APE).exists() { return; }
    let reader = ApeReader::open(TEST_APE).unwrap();
    let info = reader.info();

    eprintln!("Format version: {}", info.format_version);
    eprintln!("Channels: {}", info.channels);
    eprintln!("Sample rate: {}", info.sample_rate);
    eprintln!("Bits per sample: {}", info.bits_per_sample);
    eprintln!("Compression level: {}", info.compression_level);
    eprintln!("Total samples: {}", info.total_samples);
}
