use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::NcfWriter;
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::fs;

#[pyfunction]
fn inspect_ncf(path: &str) -> PyResult<String> {
    let reader = ncf_io::NcfReader::open(path).map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    let mut out = String::new();
    let metadata = reader.metadata();
    out.push_str(&format!("Model: {}\n", metadata.metadata.model_name));
    out.push_str(&format!("Architecture: {}\n", metadata.metadata.architecture));
    let schemas = reader.schemas().map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    out.push_str(&format!("Tensors: {}\n", schemas.len()));
    for tensor in schemas {
        out.push_str(&format!("- {} {} {:?}\n", tensor.name, tensor.dtype, tensor.shape));
    }
    Ok(out)
}

#[pyfunction]
fn create_ncf(input: &str, output: &str, name: Option<String>) -> PyResult<()> {
    let bytes = fs::read(input).map_err(|err| PyErr::new::<pyo3::exceptions::PyIOError, _>(err.to_string()))?;
    let model_name = name.unwrap_or_else(|| input.to_string());
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let metadata = NcfHeader {
        metadata: Metadata {
            model_name: model_name.clone(),
            architecture: "python".to_string(),
            created_at: now,
            author: None,
            license: None,
            quantization: None,
            custom: BTreeMap::new(),
        },
    };
    let tensor_schema = TensorSchema {
        name: model_name,
        dtype: DType::U8,
        shape: vec![bytes.len() as u64],
        column_layout: Layout::RowMajor,
        compression: Compression::None,
        encoding: Encoding::Plain,
        chunks: Vec::new(),
    };
    let mut writer = NcfWriter::new(metadata, NcfFlags::empty());
    writer.add_tensor(tensor_schema, bytes);
    writer.finalize(output).map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    Ok(())
}

#[pymodule]
fn ncf_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inspect_ncf, m)?)?;
    m.add_function(wrap_pyfunction!(create_ncf, m)?)?;
    Ok(())
}
