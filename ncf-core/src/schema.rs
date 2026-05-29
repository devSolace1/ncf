use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSchema {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub column_layout: Layout,
    pub compression: Compression,
    pub encoding: Encoding,
    pub chunks: Vec<ChunkRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRef {
    pub chunk_id: u64,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub uncompressed_len: u64,
    pub checksum: [u8; 32],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    U8,
    Q4K,
    Q4_0,
    Q8_0,
    Custom(u8),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Layout {
    RowMajor,
    ColMajor,
    Tiled(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    None,
    Zstd(u8),
    Lz4,
    Snappy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Encoding {
    Plain,
    DeltaRLE,
    BitPacked,
    DictionaryRLE,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F64 => write!(f, "F64"),
            DType::F32 => write!(f, "F32"),
            DType::F16 => write!(f, "F16"),
            DType::BF16 => write!(f, "BF16"),
            DType::I32 => write!(f, "I32"),
            DType::I16 => write!(f, "I16"),
            DType::I8 => write!(f, "I8"),
            DType::U8 => write!(f, "U8"),
            DType::Q4K => write!(f, "Q4K"),
            DType::Q4_0 => write!(f, "Q4_0"),
            DType::Q8_0 => write!(f, "Q8_0"),
            DType::Custom(tag) => write!(f, "Custom({})", tag),
        }
    }
}
