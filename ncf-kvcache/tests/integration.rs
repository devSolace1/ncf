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

    let mut writer = KvCacheWriter::create(&path, config.clone()).expect("create writer");
    for i in 0..64u32 {
        let frame = (0..config.frame_stride() as u8)
            .map(|b: u8| b.wrapping_add(i as u8))
            .collect::<Vec<u8>>();
        writer.append_frame(&frame).expect("append frame");
    }

    writer.flush_pending().expect("flush pending");
    drop(writer);

    let reader = KvcacheReader::open(&path).expect("open reader");
    assert_eq!(reader.visible_token_count(), 64);
    assert_eq!(reader.header().layers, 1);
    assert_eq!(reader.header().heads, 1);
    assert_eq!(reader.header().element_bytes, 4);
    let token_bytes = reader.token_bytes(0, 0, 0).expect("first token available");
    assert_eq!(token_bytes, &[0, 1, 2, 3]);
    let block_bytes = reader.block_bytes(0, 0, 0).expect("block available");
    assert_eq!(block_bytes.len(), 64 * 4);
    assert_eq!(reader.block_token_count(0, 0, 0), Some(64));
}
