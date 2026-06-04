use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ncf_core::header::{Metadata, NcfHeader};
use ncf_core::index::NcfIndex;
use ncf_core::schema::{ChunkRef, Compression, DType, Encoding, Layout, TensorSchema};
use std::collections::BTreeMap;

fn benchmark_header_cbor_roundtrip(c: &mut Criterion) {
    let header = NcfHeader {
        metadata: Metadata {
            model_name: "ncf-core-bench".to_string(),
            architecture: "core".to_string(),
            created_at: 42,
            author: Some("bench".to_string()),
            license: None,
            quantization: None,
            custom: BTreeMap::new(),
        },
    };

    c.bench_function("ncfcore_header_cbor_roundtrip", |b| {
        b.iter(|| {
            let bytes = header.encode_cbor().expect("encode cbor");
            let decoded = NcfHeader::decode_cbor(&black_box(bytes)).expect("decode cbor");
            black_box(decoded.metadata.model_name.as_str());
        })
    });
}

fn benchmark_index_build(c: &mut Criterion) {
    let schemas: Vec<TensorSchema> = (0..16)
        .map(|i| TensorSchema {
            name: format!("layer_{:03}", i),
            dtype: DType::F32,
            shape: vec![1024, 1024],
            column_layout: Layout::RowMajor,
            compression: Compression::None,
            encoding: Encoding::Plain,
            chunks: vec![ChunkRef {
                chunk_id: i,
                byte_offset: i * 1_000_000,
                byte_len: 1_000_030,
                uncompressed_len: 1_000_000,
                checksum: [0u8; 32],
            }],
        })
        .collect();

    c.bench_function("ncfcore_index_build", |b| {
        b.iter(|| {
            let index = NcfIndex::build_from_schemas(black_box(&schemas));
            black_box(index.entry_count);
        })
    });
}

criterion_group!(benches, benchmark_header_cbor_roundtrip, benchmark_index_build);
criterion_main!(benches);
