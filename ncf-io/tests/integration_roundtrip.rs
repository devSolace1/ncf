use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::NcfWriter;
use std::collections::BTreeMap;
use std::fs;

#[test]
fn roundtrip_write_and_read_tensor() {
    let tmp_dir = std::env::temp_dir();
    let path = tmp_dir.join("ncf_integration_test.ncf");
    let bytes: Vec<u8> = (0u8..64).collect();

    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let metadata = NcfHeader {
        metadata: Metadata {
            model_name: "integration_test".to_string(),
            architecture: "test-arch".to_string(),
            created_at: now,
            author: Some("tester".to_string()),
            license: None,
            quantization: None,
            custom: BTreeMap::new(),
        },
    };

    let schema = TensorSchema {
        name: "tensor0".to_string(),
        dtype: DType::U8,
        shape: vec![bytes.len() as u64],
        column_layout: Layout::RowMajor,
        compression: Compression::None,
        encoding: Encoding::Plain,
        chunks: Vec::new(),
    };

    let mut writer = NcfWriter::new(metadata, NcfFlags::empty());
    writer.add_tensor(schema, bytes.clone());
    let _ = writer.finalize(&path).expect("failed to write ncf fixture");

    // Read back using reader
    let reader = ncf_io::NcfReader::open(&path).expect("failed to open written ncf");
    let data = reader.read_tensor("tensor0").expect("read_tensor failed").expect("tensor missing");
    assert_eq!(data, bytes);

    let _ = fs::remove_file(&path);
}
