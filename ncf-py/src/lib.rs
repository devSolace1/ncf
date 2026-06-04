use ncf_core::header::{Metadata, NcfHeader, NcfFlags};
use ncf_core::schema::{Compression, DType, Encoding, Layout, TensorSchema};
use ncf_io::NcfWriter;
use numpy::PyArray;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::fs;

fn tensor_data_to_numpy<'py>(py: Python<'py>, schema: &TensorSchema, data: &[u8]) -> PyResult<&'py PyAny> {
    let shape: Vec<usize> = schema.shape.iter().map(|&dim| dim as usize).collect();
    match schema.dtype {
        DType::U8 => {
            let array = PyArray::from_slice(py, data);
            array.reshape(shape.as_slice())?;
            Ok(array)
        }
        DType::I8 => {
            let values = data.iter().map(|&b| b as i8).collect::<Vec<_>>();
            let array = PyArray::from_vec(py, values);
            array.reshape(shape.as_slice())?;
            Ok(array)
        }
        DType::I16 => {
            let values = data
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap_or([0, 0])))
                .collect::<Vec<_>>();
            let array = PyArray::from_vec(py, values);
            array.reshape(shape.as_slice())?;
            Ok(array)
        }
        DType::I32 => {
            let values = data
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap_or([0, 0, 0, 0])))
                .collect::<Vec<_>>();
            let array = PyArray::from_vec(py, values);
            array.reshape(shape.as_slice())?;
            Ok(array)
        }
        DType::F32 => {
            let values = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap_or([0, 0, 0, 0])))
                .collect::<Vec<_>>();
            let array = PyArray::from_vec(py, values);
            array.reshape(shape.as_slice())?;
            Ok(array)
        }
        DType::F64 => {
            let values = data
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap_or([0, 0, 0, 0, 0, 0, 0, 0])))
                .collect::<Vec<_>>();
            let array = PyArray::from_vec(py, values);
            array.reshape(shape.as_slice())?;
            Ok(array)
        }
        _ => {
            let array = PyArray::from_slice(py, data);
            array.reshape(shape.as_slice())?;
            Ok(array)
        }
    }
}

#[pyfunction]
fn inspect_ncf(path: &str) -> PyResult<String> {
    let reader = ncf_io::NcfReader::open(path)
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;
    let mut out = String::new();
    let metadata = reader.metadata();
    out.push_str(&format!("Model: {}\n", metadata.metadata.model_name));
    out.push_str(&format!("Architecture: {}\n", metadata.metadata.architecture));
    let schemas = reader
        .schemas()
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;
    out.push_str(&format!("Tensors: {}\n", schemas.len()));
    for tensor in schemas {
        out.push_str(&format!("- {} {} {:?}\n", tensor.name, tensor.dtype, tensor.shape));
    }
    Ok(out)
}

#[pyfunction]
fn create_ncf(input: &str, output: &str, name: Option<String>) -> PyResult<()> {
    let bytes = fs::read(input).map_err(|err| PyErr::new::<PyIOError, _>(err.to_string()))?;
    let model_name = name.unwrap_or_else(|| input.to_string());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
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
    writer.finalize(output)
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;
    Ok(())
}

#[pyfunction]
fn load_tensor(path: &str, name: &str) -> PyResult<Option<Vec<u8>>> {
    let reader = ncf_io::NcfReader::open(path)
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;
    match reader.read_tensor(name) {
        Ok(Some(data)) => Ok(Some(data)),
        Ok(None) => Ok(None),
        Err(err) => Err(PyErr::new::<PyRuntimeError, _>(err.to_string())),
    }
}

#[pyfunction]
fn load_tensor_numpy(py: Python, path: &str, name: &str) -> PyResult<PyObject> {
    let reader = ncf_io::NcfReader::open(path)
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?;
    let schema = reader
        .find_schema(name)
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?
        .ok_or_else(|| PyErr::new::<PyValueError, _>("tensor not found"))?;
    let data = reader
        .read_tensor(name)
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(err.to_string()))?
        .ok_or_else(|| PyErr::new::<PyValueError, _>("tensor not found"))?;
    let array = tensor_data_to_numpy(py, schema, &data)?;
    Ok(array.into_py(py))
}

#[pyfunction]
fn load_tensor_torch(py: Python, path: &str, name: &str) -> PyResult<PyObject> {
    let array = load_tensor_numpy(py, path, name)?;
    let torch = py
        .import("torch")
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(format!("PyTorch import failed: {}", err)))?;
    let tensor = torch
        .call_method1("from_numpy", (array,))
        .map_err(|err| PyErr::new::<PyRuntimeError, _>(format!("PyTorch conversion failed: {}", err)))?;
    Ok(tensor.into_py(py))
}

#[pymodule]
fn ncf_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inspect_ncf, m)?)?;
    m.add_function(wrap_pyfunction!(create_ncf, m)?)?;
    m.add_function(wrap_pyfunction!(load_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(load_tensor_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(load_tensor_torch, m)?)?;
    Ok(())
}
