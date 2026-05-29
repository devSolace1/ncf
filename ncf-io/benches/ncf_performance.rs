use criterion::{black_box, criterion_group, criterion_main, Criterion};
use memmap2::MmapOptions;
use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::{NcfMmap, NcfReader, NcfWriter};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use safetensors::tensor::{Dtype as SafeDtype, View};
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const REALISTIC_LAYER_COUNT: usize = 16;
const REALISTIC_TENSOR_BYTES: usize = 32 * 1024 * 1024; // 32 MiB per tensor, ~512 MiB model
const PARTIAL_LAYER_COUNT: usize = 32;
const PARTIAL_TENSOR_BYTES: usize = 4 * 1024 * 1024; // 4 MiB per tensor for partial-load test

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

struct TensorPayload {
    name: String,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn build_realistic_tensor_payload(layer_count: usize, tensor_bytes: usize) -> Vec<TensorPayload> {
    let mut rng = ChaCha20Rng::seed_from_u64(0x1f3e_2d4c_5b6a_7980);
    let elements = tensor_bytes / 4;
    let shape = vec![elements as u64, 1];

    (0..layer_count)
        .map(|i| {
            let mut data = vec![0u8; tensor_bytes];
            rng.fill_bytes(&mut data);

            TensorPayload {
                name: format!("layer_{:03}", i),
                shape: shape.clone(),
                data,
            }
        })
        .collect()
}

fn build_ncf_from_payloads(path: &Path, payloads: &[TensorPayload]) {
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

    for payload in payloads {
        let schema = TensorSchema {
            name: payload.name.clone(),
            dtype: DType::F32,
            shape: payload.shape.clone(),
            column_layout: Layout::RowMajor,
            compression: Compression::None,
            encoding: Encoding::Plain,
            chunks: Vec::new(),
        };
        writer.add_tensor(schema, payload.data.clone());
    }

    writer.finalize(path).expect("failed to create benchmark NCF");
}

fn build_safetensors_from_payloads(path: &Path, payloads: &[TensorPayload]) {
    if path.exists() {
        let _ = fs::remove_file(path);
    }

    let tensors: Vec<OwnedSafeTensor> = payloads
        .iter()
        .map(|payload| OwnedSafeTensor {
            name: payload.name.clone(),
            dtype: SafeDtype::F32,
            shape: payload.shape.iter().map(|&dim| dim as usize).collect(),
            data: payload.data.clone(),
        })
        .collect();

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
    let payloads = build_realistic_tensor_payload(4, 4 * 1024 * 1024);
    build_ncf_from_payloads(path, &payloads);
}

fn build_realistic_ncf(path: &Path) {
    let payloads = build_realistic_tensor_payload(REALISTIC_LAYER_COUNT, REALISTIC_TENSOR_BYTES);
    build_ncf_from_payloads(path, &payloads);
}

fn build_sample_ncf_with_layers(path: &Path, layer_count: usize, tensor_bytes: usize) {
    let payloads = build_realistic_tensor_payload(layer_count, tensor_bytes);
    build_ncf_from_payloads(path, &payloads);
}

fn build_sample_safetensors(path: &Path) {
    let payloads = build_realistic_tensor_payload(4, 4 * 1024 * 1024);
    build_safetensors_from_payloads(path, &payloads);
}

fn build_sample_safetensors_with_layers(path: &Path, layer_count: usize, tensor_bytes: usize) {
    let payloads = build_realistic_tensor_payload(layer_count, tensor_bytes);
    build_safetensors_from_payloads(path, &payloads);
}

fn build_realistic_safetensors(path: &Path) {
    let payloads = build_realistic_tensor_payload(REALISTIC_LAYER_COUNT, REALISTIC_TENSOR_BYTES);
    build_safetensors_from_payloads(path, &payloads);
}

fn benchmark_ncf_reader_open(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_sample.ncf");
    build_sample_ncf(&sample_path);

    // NCF open performs CBOR metadata/schema/index parsing, so this benchmark
    // measures cold-start cost. NCF dirancang untuk pola `open once, access many`.
    c.bench_function("ncf_reader_open", |b| {
        b.iter(|| {
            let reader = NcfReader::open(&sample_path).expect("open sample ncf");
            black_box(reader.metadata().metadata.model_name.as_str());
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

fn benchmark_ncf_reader_tensor_slice(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_sample.ncf");
    build_sample_ncf(&sample_path);
    let reader = NcfReader::open(&sample_path).expect("open sample ncf");

    c.bench_function("ncf_reader_tensor_slice", |b| {
        b.iter(|| {
            let slice = reader.tensor_slice("layer_000").expect("layer_000 slice missing");
            black_box(slice);
        })
    });
}

fn benchmark_ncf_sequential_layer_access(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_sequential.ncf");
    build_realistic_ncf(&sample_path);
    let mmap = NcfMmap::open(&sample_path).expect("open realistic ncf mmap");

    c.bench_function("ncf_sequential_layer_access", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for i in 0..REALISTIC_LAYER_COUNT {
                let tensor_name = format!("layer_{:03}", i);
                let slice = mmap.tensor_slice(&tensor_name).expect("slice missing");
                total = total.wrapping_add(slice.len());
            }
            black_box(total);
        })
    });
}

fn benchmark_safetensors_sequential_layer_access(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("safetensors_benchmark_sequential.safetensors");
    build_realistic_safetensors(&sample_path);
    let bytes = fs::read(&sample_path).expect("read safetensors sample");

    c.bench_function("safetensors_sequential_layer_access", |b| {
        b.iter(|| {
            let tensors = SafeTensors::deserialize(black_box(&bytes)).expect("deserialize safetensors");
            let total: usize = tensors.iter().map(|(_, view)| view.data_len()).sum();
            black_box(total);
        })
    });
}

fn benchmark_safetensors_load_equivalent(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("safetensors_benchmark_sample.safetensors");
    build_sample_safetensors(&sample_path);
    let bytes = fs::read(&sample_path).expect("read safetensors sample");

    c.bench_function("safetensors_load_equivalent", |b| {
        b.iter(|| {
            let tensors = SafeTensors::deserialize(black_box(&bytes)).expect("deserialize safetensors");
            black_box(tensors.iter().count());
        })
    });
}

fn benchmark_ncf_reader_open_realistic(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_realistic.ncf");
    build_realistic_ncf(&sample_path);

    c.bench_function("ncf_reader_open_realistic", |b| {
        b.iter(|| {
            let reader = NcfReader::open(&sample_path).expect("open realistic ncf");
            black_box(reader.metadata().metadata.model_name.as_str());
        })
    });
}

fn benchmark_ncf_realistic_load(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_realistic.ncf");
    build_realistic_ncf(&sample_path);

    c.bench_function("ncf_realistic_load", |b| {
        b.iter(|| {
            let reader = NcfReader::open(&sample_path).expect("open realistic ncf");
            black_box(reader.schemas().iter().count());
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
    let pool = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("failed to build rayon thread pool");

    c.bench_function("ncf_parallel_chunk_load", |b| {
        b.iter(|| {
            pool.install(|| {
                (0..4).into_par_iter().for_each(|i| {
                    let tensor_name = format!("layer_{:03}", i);
                    let slice = mmap.tensor_slice(&tensor_name).expect("slice missing");
                    black_box(slice);
                });
            })
        })
    });
}

fn benchmark_ncf_partial_layer_load(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("ncf_benchmark_partial.ncf");
    build_sample_ncf_with_layers(&sample_path, PARTIAL_LAYER_COUNT, PARTIAL_TENSOR_BYTES);
    let mmap = NcfMmap::open(&sample_path).expect("open sample ncf mmap");

    // This benchmark measures partial load of a single tensor from a large model
    // without accessing earlier tensor layers.
    c.bench_function("ncf_partial_layer_load", |b| {
        b.iter(|| {
            let slice = mmap.tensor_slice("layer_015").expect("layer_015 slice missing");
            black_box(slice.len());
        })
    });
}

fn benchmark_safetensors_partial_layer_access(c: &mut Criterion) {
    let sample_path = PathBuf::from(std::env::temp_dir()).join("safetensors_benchmark_partial.safetensors");
    build_sample_safetensors_with_layers(&sample_path, PARTIAL_LAYER_COUNT, PARTIAL_TENSOR_BYTES);
    let bytes = fs::read(&sample_path).expect("read safetensors sample");

    c.bench_function("safetensors_partial_layer_access", |b| {
        b.iter(|| {
            let tensors = SafeTensors::deserialize(black_box(&bytes)).expect("deserialize safetensors");
            let layer = tensors
                .iter()
                .find(|(name, _)| *name == "layer_015")
                .expect("layer_015 missing");
            black_box(layer.1.data_len());
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
    benchmark_ncf_reader_open_realistic,
    benchmark_ncf_mmap_tensor_slice,
    benchmark_ncf_reader_tensor_slice,
    benchmark_ncf_sequential_layer_access,
    benchmark_safetensors_sequential_layer_access,
    benchmark_safetensors_load_equivalent,
    benchmark_ncf_realistic_load,
    benchmark_safetensors_realistic_load,
    benchmark_ncf_parallel_chunk_load,
    benchmark_ncf_partial_layer_load,
    benchmark_safetensors_partial_layer_access,
    benchmark_ncf_streaming_chunk_verify,
);
criterion_main!(benches);
