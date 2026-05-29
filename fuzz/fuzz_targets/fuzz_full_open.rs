#![no_main]

use libfuzzer_sys::fuzz_target;
use tempfile::NamedTempFile;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Create a temporary file with the fuzzed data
    if let Ok(mut file) = NamedTempFile::new() {
        let _ = file.write_all(data);
        let _ = file.flush();
        let path = file.path();
        
        // Try to open the file as NCF
        // This tests the entire parsing pipeline:
        // - File header prefix validation
        // - CBOR header deserialization
        // - Schema block parsing
        // - Index block parsing
        // - Footer validation
        // The parser should handle all invalid inputs gracefully
        let _ = ncf_io::NcfReader::open(path);
    }
});
