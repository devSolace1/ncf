#![no_main]

use libfuzzer_sys::fuzz_target;
use ncf_core::header::{FileHeaderPrefix, NcfHeader, Metadata};
use ncf_core::index::NcfIndex;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // Test index parsing with arbitrary data
    // This simulates the portion of the file where the index resides
    let _ = ciborium::de::from_reader::<NcfIndex, _>(Cursor::new(data));
});
