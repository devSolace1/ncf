#![no_main]

use libfuzzer_sys::fuzz_target;
use ncf_core::header::FileHeaderPrefix;

fuzz_target!(|data: &[u8]| {
    // Try to decode arbitrary bytes as FileHeaderPrefix
    // The parser should handle all invalid inputs gracefully
    let _ = FileHeaderPrefix::decode(data);
});
