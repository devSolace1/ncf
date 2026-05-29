use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::{NcfMmap, NcfReader, NcfWriter};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn build_sample_ncf(path: &Path) {
    if path.exists() {
        let _ = fs::remove_file(path);
    }

    let metadata = NcfHeader {
        metadata: Metadata {
            model_name: "ncf-benchmark".to_string(),
            architecture: "criterion".to_string(),
            created_at: 0,
            author: None,
            license: None,
            quantization: None,
            custom: BTreeMap::new(),
        },
    };

    let mut writer = NcfWriter::new(metadata, NcfFlags::empty());

    let payload_a = vec![0u8; 1024 * 1024];
    let schema_a = TensorSchema {
        name: "tensor_0".to_string(),
        dtype: DType::F32,
        shape: vec![256, 256, 4],
        column_layout: Layout::RowMajor,
        compression: Compression::None,
        encoding: Encoding::Plain,
        chunks: Vec::new(),
    };
    writer.add_tensor(schema_a, payload_a);

    let payload_b = vec![1u8; 512 * 1024];
    let schema_b = TensorSchema {
        name: "tensor_1".to_string(),
        dtype: DType::F16,
        shape: vec![128, 256, 2],
        column_layout: Layout::RowMajor,
        compression: Compression::None,
        encoding: Encoding::Plain,
        chunks: Vec::new(),
    };
    writer.add_tensor(schema_b, payload_b);

    writer.finalize(path).expect("failed to create benchmark NCF");
}

fn benchmark_ncf_reader_open(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_sample.ncf");
    build_sample_ncf(&sample_path);

    c.bench_function("ncf_reader_open", |b| {
        b.iter(|| {
            let reader = NcfReader::open(&sample_path).expect("open sample ncf");
            black_box(reader.metadata.metadata.model_name.clone());
        })
    });
}

fn benchmark_ncf_mmap_tensor_slice(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_sample.ncf");
    build_sample_ncf(&sample_path);
    let mmap = NcfMmap::open(&sample_path).expect("open sample ncf mmap");

    c.bench_function("ncf_mmap_tensor_slice", |b| {
        b.iter(|| {
            let slice = mmap.tensor_slice("tensor_0").expect("tensor_0 slice missing");
            black_box(slice);
        })
    });
}

criterion_group!(benches, benchmark_ncf_reader_open, benchmark_ncf_mmap_tensor_slice);
criterion_main!(benches);
