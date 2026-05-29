use anyhow::Context;
use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::NcfWriter;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn gguf_type_to_dtype(tensor_type: gguf::GGMLType) -> DType {
    match tensor_type {
        gguf::GGMLType::F32 => DType::F32,
        gguf::GGMLType::F16 => DType::F16,
        gguf::GGMLType::Q4_0 => DType::Q4_0,
        gguf::GGMLType::Q4_1 => DType::Q4_0,
        gguf::GGMLType::Q5_0 => DType::Custom(5),
        gguf::GGMLType::Q5_1 => DType::Custom(6),
        gguf::GGMLType::Q8_0 => DType::Q8_0,
        gguf::GGMLType::Q8_1 => DType::Custom(9),
        gguf::GGMLType::Q2K => DType::Q4K,
        gguf::GGMLType::Q3K => DType::Custom(11),
        gguf::GGMLType::Q4K => DType::Q4K,
        gguf::GGMLType::Q5K => DType::Custom(13),
        gguf::GGMLType::Q6K => DType::Custom(14),
        gguf::GGMLType::Q8K => DType::Q8_0,
        gguf::GGMLType::I8 => DType::I8,
        gguf::GGMLType::I16 => DType::I16,
        gguf::GGMLType::I32 => DType::I32,
        _ => DType::Custom(0),
    }
}

pub fn gguf_to_ncf<P: AsRef<Path>>(input: P, output: P, architecture: Option<&str>, author: Option<&str>) -> anyhow::Result<()> {
    let mut file = File::open(&input).with_context(|| format!("opening GGUF file {}", input.as_ref().display()))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    let archive = gguf::GGUFFile::read(&data)
        .map_err(|err| anyhow::anyhow!("failed to parse GGUF file: {}", err))?
        .context("failed to parse GGUF file")?;

    let arch = architecture.map(|s| s.to_string()).unwrap_or_else(|| "gguf-converted".to_string());
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

    let mut offsets: Vec<_> = archive.tensors.iter().map(|tensor| tensor.offset).collect();
    offsets.push(data.len() as u64);
    offsets.sort_unstable();

    for tensor in &archive.tensors {
        let offset_index = offsets.iter().position(|&o| o == tensor.offset).unwrap();
        let next_offset = offsets[offset_index + 1] as usize;
        let start = tensor.offset as usize;
        let end = next_offset;
        let payload = data[start..end].to_vec();
        let dtype = gguf_type_to_dtype(tensor.tensor_type);
        let shape = tensor.dimensions.iter().copied().collect();
        // Validate payload size for known element sizes
        let elem_size_opt: Option<u64> = match dtype {
            DType::F64 => Some(8),
            DType::F32 => Some(4),
            DType::F16 => Some(2),
            DType::BF16 => Some(2),
            DType::I32 => Some(4),
            DType::I16 => Some(2),
            DType::I8 => Some(1),
            DType::U8 => Some(1),
            // Quantized/custom formats may have packed representations; skip strict validation
            DType::Q4K | DType::Q4_0 | DType::Q8_0 | DType::Custom(_) => None,
        };
        if let Some(elem_size) = elem_size_opt {
            let mut elems: u64 = 1;
            for &d in &tensor.dimensions {
                elems = elems.saturating_mul(d);
            }
            let expected_bytes = elems.saturating_mul(elem_size) as usize;
            if expected_bytes != payload.len() {
                return Err(anyhow::anyhow!(
                    "payload size mismatch for tensor '{}': expected {} bytes ({} elements * {} bytes), got {} bytes",
                    tensor.name,
                    expected_bytes,
                    elems,
                    elem_size,
                    payload.len()
                ));
            }
        }
        let schema = TensorSchema {
            name: tensor.name.clone(),
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
