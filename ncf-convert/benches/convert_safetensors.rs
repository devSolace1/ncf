use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ncf_convert::safetensors_to_ncf;
use safetensors::tensor::{Dtype as SafeDtype, View};
use safetensors::serialize_to_file;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

struct NamedSafeTensor<'a> {
    name: String,
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: &'a [u8],
}

impl<'a> View for NamedSafeTensor<'a> {
    fn dtype(&self) -> SafeDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl<'a> View for &'a NamedSafeTensor<'a> {
    fn dtype(&self) -> SafeDtype {
        (*self).dtype
    }

    fn shape(&self) -> &[usize] {
        &(*self).shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed((*self).data)
    }

    fn data_len(&self) -> usize {
        (*self).data.len()
    }
}

fn build_safetensors_sample(path: &PathBuf) {
    let data1 = vec![0u8; 1024 * 1024 * 4];
    let data2 = vec![1u8; 256 * 256];
    let tensors = vec![
        NamedSafeTensor {
            name: "tensor_000".to_string(),
            dtype: SafeDtype::F32,
            shape: vec![1024, 1024],
            data: &data1,
        },
        NamedSafeTensor {
            name: "tensor_001".to_string(),
            dtype: SafeDtype::I8,
            shape: vec![256, 256],
            data: &data2,
        },
    ];

    serialize_to_file(
        tensors.iter().map(|tensor| (tensor.name.as_str(), tensor)),
        None,
        path,
    )
    .expect("failed to serialize safetensors");
}

fn benchmark_convert_safetensors_to_ncf(c: &mut Criterion) {
    let temp_dir = tempdir().expect("create temp dir");
    let safetensors_path = temp_dir.path().join("sample.safetensors");
    let output_path = temp_dir.path().join("sample.ncf");
    build_safetensors_sample(&safetensors_path);

    c.bench_function("convert_safetensors_to_ncf", |b| {
        b.iter(|| {
            fs::remove_file(&output_path).ok();
            safetensors_to_ncf(&safetensors_path, &output_path, None, None)
                .expect("failed to convert safetensors");
            let metadata = fs::metadata(&output_path).expect("output metadata");
            black_box(metadata.len());
        })
    });
}

criterion_group!(benches, benchmark_convert_safetensors_to_ncf);
criterion_main!(benches);
