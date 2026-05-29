use anyhow::Context;
use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::NcfWriter;
use safetensors::{SafeTensors, Dtype as SafeDtype};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn safetensors_to_ncf<P: AsRef<Path>>(input: P, output: P, architecture: Option<&str>, author: Option<&str>) -> anyhow::Result<()> {
    let mut file = File::open(&input).with_context(|| format!("opening safetensors file {}", input.as_ref().display()))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    let archive = SafeTensors::deserialize(&data)?;
    let arch = architecture.map(|s| s.to_string()).unwrap_or_else(|| "safetensors-converted".to_string());
    let mut writer = NcfWriter::new(
        NcfHeader {
            metadata: Metadata {
                model_name: input.as_ref().file_name().unwrap_or_default().to_string_lossy().into_owned(),
                architecture: arch,
                created_at: chrono::Utc::now().timestamp() as u64,
                author: author.map(|s| s.to_string()),
                license: None,
                quantization: None,
                custom: BTreeMap::new(),
            },
        },
        NcfFlags::empty(),
    );

    for (name, tensor) in archive.iter() {
        let shape = tensor.shape().iter().map(|v| *v as u64).collect();
        let dtype = match tensor.dtype() {
            SafeDtype::F32 => DType::F32,
            SafeDtype::F16 => DType::F16,
            SafeDtype::BF16 => DType::BF16,
            SafeDtype::I32 => DType::I32,
            SafeDtype::I16 => DType::I16,
            SafeDtype::I8 => DType::I8,
            SafeDtype::U8 => DType::U8,
            _ => DType::Custom(0),
        };
        let payload = tensor.data().to_owned();
        let schema = TensorSchema {
            name: name.to_string(),
            dtype,
            shape,
            column_layout: Layout::RowMajor,
            compression: Compression::None,
            encoding: Encoding::Plain,
            chunks: Vec::new(),
        };
        writer.add_tensor(schema, payload);
    }
    writer.finalize(output)?;
    Ok(())
}
