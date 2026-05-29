use criterion::{black_box, criterion_group, criterion_main, Criterion};
use memmap2::MmapOptions;
use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::{NcfMmap, NcfReader, NcfWriter};
use safetensors::tensor::{Dtype as SafeDtype, View};
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;

const REALISTIC_LAYER_COUNT: usize = 16;
const REALISTIC_TENSOR_BYTES: usize = 32 * 1024 * 1024; // 32 MiB per tensor, ~512 MiB model

struct SafeTensorView<'a> {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: &'a [u8],
}

impl<'a> View for SafeTensorView<'a> {
    fn dtype(&self) -> SafeDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

struct OwnedSafeTensor {
    name: String,
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn build_sample_ncf_with_layers(path: &Path, layer_count: usize, tensor_bytes: usize) {
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

    let elements = tensor_bytes / 4;
    let shape = vec![elements as u64, 1];

    for i in 0..layer_count {
        let name = format!("layer_{:03}", i);
        let payload = vec![i as u8; tensor_bytes];
        let schema = TensorSchema {
            name,
            dtype: DType::F32,
            shape: shape.clone(),
            column_layout: Layout::RowMajor,
            compression: Compression::None,
            encoding: Encoding::Plain,
            chunks: Vec::new(),
        };
        writer.add_tensor(schema, payload);
    }

    writer.finalize(path).expect("failed to create benchmark NCF");
}

fn build_sample_safetensors(path: &Path, layer_count: usize, tensor_bytes: usize) {
    if path.exists() {
        let _ = fs::remove_file(path);
    }

    let elements = tensor_bytes / 4;
    let shape = vec![elements, 1];

    let mut tensors = Vec::new();
    for i in 0..layer_count {
        tensors.push(OwnedSafeTensor {
            name: format!("layer_{:03}", i),
            dtype: SafeDtype::F32,
            shape: shape.clone(),
            data: vec![i as u8; tensor_bytes],
        });
    }

    safetensors::serialize_to_file(
        tensors.iter().map(|tensor| {
            (
                tensor.name.as_str(),
                SafeTensorView {
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                    data: &tensor.data,
                },
            )
        }),
        None,
        &path,
    )
    .expect("failed to serialize safetensors");
}

fn build_sample_ncf(path: &Path) {
    build_sample_ncf_with_layers(path, 4, 4 * 1024 * 1024);
}

fn build_realistic_ncf(path: &Path) {
    build_sample_ncf_with_layers(path, REALISTIC_LAYER_COUNT, REALISTIC_TENSOR_BYTES);
}

fn build_realistic_safetensors(path: &Path) {
    build_sample_safetensors(path, REALISTIC_LAYER_COUNT, REALISTIC_TENSOR_BYTES);
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
            let slice = mmap.tensor_slice("layer_000").expect("layer_000 slice missing");
            black_box(slice);
        })
    });
}

fn benchmark_safetensors_load_equivalent(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("safetensors_benchmark_sample.safetensors");
    build_sample_safetensors(&sample_path, 8, 8 * 1024 * 1024);
    let bytes = fs::read(&sample_path).expect("read safetensors sample");

    c.bench_function("safetensors_load_equivalent", |b| {
        b.iter(|| {
            let tensors = SafeTensors::deserialize(black_box(&bytes)).expect("deserialize safetensors");
            black_box(tensors.iter().count());
        })
    });
}

fn benchmark_ncf_realistic_load(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_realistic.ncf");
    build_realistic_ncf(&sample_path);

    c.bench_function("ncf_realistic_load", |b| {
        b.iter(|| {
            let reader = NcfReader::open(&sample_path).expect("open realistic ncf");
            black_box(reader.schemas.len());
        })
    });
}

fn benchmark_safetensors_realistic_load(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("safetensors_benchmark_realistic.safetensors");
    build_realistic_safetensors(&sample_path);
    let file = fs::File::open(&sample_path).expect("open realistic safetensors");
    let mmap = unsafe { MmapOptions::new().map(&file).expect("memory map safetensors file") };

    c.bench_function("safetensors_realistic_load", |b| {
        b.iter(|| {
            let tensors = SafeTensors::deserialize(black_box(&mmap[..])).expect("deserialize realistic safetensors");
            black_box(tensors.iter().count());
        })
    });
}

fn benchmark_ncf_parallel_chunk_load(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_parallel.ncf");
    build_sample_ncf_with_layers(&sample_path, 8, 4 * 1024 * 1024);
    let mmap = NcfMmap::open(&sample_path).expect("open sample ncf mmap");

    c.bench_function("ncf_parallel_chunk_load", |b| {
        b.iter(|| {
            let mmap_ref = &mmap;
            thread::scope(|s| {
                for i in 0..4 {
                    let tensor_name = format!("layer_{:03}", i);
                    s.spawn(move || {
                        let slice = mmap_ref.tensor_slice(&tensor_name).expect("slice missing");
                        black_box(slice);
                    });
                }
            });
        })
    });
}

fn benchmark_ncf_selective_layer_load(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_selective.ncf");
    build_sample_ncf_with_layers(&sample_path, 32, 4 * 1024 * 1024);
    let mmap = NcfMmap::open(&sample_path).expect("open sample ncf mmap");

    c.bench_function("ncf_selective_layer_load", |b| {
        b.iter(|| {
            let slice = mmap.tensor_slice("layer_015").expect("layer_015 slice missing");
            black_box(slice.len());
        })
    });
}

fn benchmark_ncf_streaming_chunk_verify(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_streaming.ncf");
    build_sample_ncf_with_layers(&sample_path, 4, 4 * 1024 * 1024);
    let mmap = NcfMmap::open(&sample_path).expect("open sample ncf mmap");

    c.bench_function("ncf_streaming_chunk_verify", |b| {
        b.iter(|| {
            let slice = mmap.tensor_slice("layer_000").expect("layer_000 slice missing");
            let hash = blake3::hash(black_box(slice));
            black_box(hash);
        })
    });
}

criterion_group!(
    benches,
    benchmark_ncf_reader_open,
    benchmark_ncf_mmap_tensor_slice,
    benchmark_safetensors_load_equivalent,
    benchmark_ncf_realistic_load,
    benchmark_safetensors_realistic_load,
    benchmark_ncf_parallel_chunk_load,
    benchmark_ncf_selective_layer_load,
    benchmark_ncf_streaming_chunk_verify,
);
criterion_main!(benches);
