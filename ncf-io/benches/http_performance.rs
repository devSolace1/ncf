#![cfg(feature = "http")]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tempfile::TempDir;
use wiremock::{Mock, MockServer, ResponseTemplate};
use wiremock::matchers::{header, method};

use ncf_core::header::{Metadata, NcfFlags, NcfHeader};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::NcfHttpReader;
use ncf_io::NcfWriter;

fn build_sample_ncf(path: &std::path::Path) {
    let metadata = NcfHeader {
        metadata: Metadata {
            model_name: "http-bench".to_string(),
            architecture: "benchmark".to_string(),
            created_at: 0,
            author: None,
            license: None,
            quantization: None,
            custom: Default::default(),
        },
    };

    let mut writer = NcfWriter::new(metadata, NcfFlags::empty());
    let schema = TensorSchema {
        name: "tensor_0".to_string(),
        dtype: DType::U8,
        shape: vec![1024],
        column_layout: Layout::RowMajor,
        compression: Compression::None,
        encoding: Encoding::Plain,
        chunks: vec![],
    };

    writer.add_tensor(schema, vec![0u8; 1024]);
    writer.finalize(path).expect("write bench ncf");
}

fn benchmark_http_open(c: &mut Criterion) {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("http_bench.ncf");
    build_sample_ncf(&path);
    let bytes = std::fs::read(&path).expect("read bench ncf file");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let server = rt.block_on(async { MockServer::start().await });

    rt.block_on(async {
        Mock::given(method("GET"))
            .and(header("range", "bytes=0-47"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", format!("bytes 0-47/{len}", len = bytes.len()))
                    .set_body_bytes(bytes[0..48].to_vec()),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(header("range", "bytes=48-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", format!("bytes 48-{}/{}", bytes.len() - 1, bytes.len()))
                    .set_body_bytes(bytes[48..].to_vec()),
            )
            .mount(&server)
            .await;
    });

    c.bench_function("http_reader_open", |b| {
        b.iter(|| {
            rt.block_on(async {
                let reader = NcfHttpReader::open(&server.uri()).await.expect("open http reader");
                black_box(reader.schemas().len());
            });
        })
    });
}

criterion_group!(http_benches, benchmark_http_open);
criterion_main!(http_benches);
