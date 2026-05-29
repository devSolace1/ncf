use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Schema describing a tensor stored in an NCF file.
pub struct TensorSchema {
    /// Name of the tensor (dot-separated for nested namespaces).
    pub name: String,
    /// Data type of each element.
    pub dtype: DType,
    /// Tensor shape (dimensions).
    pub shape: Vec<u64>,
    /// Memory layout of the tensor.
    pub column_layout: Layout,
    /// Compression applied to chunks for this tensor.
    pub compression: Compression,
    /// Logical encoding used for the tensor data.
    pub encoding: Encoding,
    /// References to chunks containing this tensor's data.
    pub chunks: Vec<ChunkRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Reference to a stored chunk from a tensor's schema.
pub struct ChunkRef {
    /// Identifier of the chunk.
    pub chunk_id: u64,
    /// Byte offset within the file where the chunk header begins.
    pub byte_offset: u64,
    /// Total length of the chunk (header + payload + checksum).
    pub byte_len: u64,
    /// Uncompressed payload length.
    pub uncompressed_len: u64,
    /// Blake3 checksum of the uncompressed payload.
    pub checksum: [u8; 32],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// Element data types supported by NCF.
pub enum DType {
    /// 64-bit float
    F64,
    /// 32-bit float
    F32,
    /// 16-bit float
    F16,
    /// bfloat16
    BF16,
    /// 32-bit signed integer
    I32,
    /// 16-bit signed integer
    I16,
    /// 8-bit signed integer
    I8,
    /// 8-bit unsigned integer
    U8,
    /// Quantized 4-bit (K layout)
    Q4K,
    /// Quantized 4-bit format variant 0
    Q4_0,
    /// Quantized 8-bit format variant 0
    Q8_0,
    /// Custom type tag
    Custom(u8),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// Memory layout for tensor elements.
pub enum Layout {
    /// Row-major layout.
    RowMajor,
    /// Column-major layout.
    ColMajor,
    /// Tiled layout with tile size.
    Tiled(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Compression options for stored tensor chunks.
pub enum Compression {
    /// No compression applied.
    None,
    /// Zstd with quality level.
    Zstd(u8),
    /// LZ4 compression.
    Lz4,
    /// Snappy compression.
    Snappy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
/// Logical encodings applied to tensor data.
pub enum Encoding {
    /// Plain raw bytes per element.
    Plain,
    /// Delta + RLE encoding.
    DeltaRLE,
    /// Bit-packed encoding.
    BitPacked,
    /// Dictionary RLE encoding.
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
