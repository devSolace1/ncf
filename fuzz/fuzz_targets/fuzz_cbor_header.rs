#![no_main]

use libfuzzer_sys::fuzz_target;
use ncf_core::header::NcfHeader;

fuzz_target!(|data: &[u8]| {
    // Try to decode arbitrary bytes as CBOR NcfHeader
    // The parser should handle all invalid inputs gracefully
    let _ = NcfHeader::decode_cbor(data);
});
