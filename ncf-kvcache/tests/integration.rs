use ncf_kvcache::{KvCacheConfig, KvcacheReader, KvCacheWriter};
use tempfile::tempdir;

#[test]
fn append_and_read_kvcache() {
    let temp = tempdir().expect("create temp dir");
    let path = temp.path().join("cache.ncf-kvcache");
    let config = KvCacheConfig {
        layers: 1,
        heads: 1,
        element_bytes: 4,
    };

    {
        let mut writer = KvCacheWriter::create(&path, config.clone()).expect("create writer");
        for i in 0..64u32 {
            let frame = (0..config.frame_stride() as u8)
                .map(|b: u8| b.wrapping_add(i as u8))
                .collect::<Vec<u8>>();
            writer.append_frame(&frame).expect("append frame");
        }
    }

    let reader = KvcacheReader::open(&path).expect("open reader");
    assert_eq!(reader.valid_token_count(), 64);
    let token_bytes = reader.token_bytes(0, 0, 0).expect("first token available");
    assert_eq!(token_bytes, &[0, 1, 2, 3]);
}
