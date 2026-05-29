use crate::Result;
use ciborium::{de::from_reader, ser::into_writer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Magic bytes at the start of an NCF file.
pub const NCF_MAGIC: &[u8; 8] = b"NCF\0\xDE\xAD\xBE\xEF";

/// Magic bytes placed in the footer of an NCF file.
pub const FOOTER_MAGIC: &[u8; 8] = b"NCFEND!!";

bitflags::bitflags! {
    /// Flags used in the file header to indicate features of the NCF file.
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct NcfFlags: u32 {
        /// File contains compressed chunks.
        const COMPRESSED = 0x1;
        /// File is encrypted.
        const ENCRYPTED = 0x2;
        /// File is marked streaming-safe.
        const STREAMING_SAFE = 0x4;
    }
}

#[derive(Debug, thiserror::Error)]
/// Errors that can occur while parsing or handling NCF files.
pub enum NcfError {
    /// IO related error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// CBOR deserialization error.
    #[error("CBOR deserialize error: {0}")]
    Cbor(#[from] ciborium::de::Error<std::io::Error>),
    /// CBOR serialization error.
    #[error("CBOR serialize error: {0}")]
    CborSer(#[from] ciborium::ser::Error<std::io::Error>),
    /// Generic header parse error with message.
    #[error("Header parse error: {0}")]
    Header(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// High-level metadata about the model/file stored in the NCF header.
pub struct Metadata {
    /// Human-readable model name.
    pub model_name: String,
    /// Architecture identifier string.
    pub architecture: String,
    /// Creation timestamp (seconds since UNIX epoch).
    pub created_at: u64,
    /// Optional author string.
    pub author: Option<String>,
    /// Optional license string.
    pub license: Option<String>,
    /// Optional quantization metadata.
    pub quantization: Option<String>,
    /// Arbitrary custom CBOR values.
    #[serde(default)]
    pub custom: BTreeMap<String, ciborium::value::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Top-level header structure encoded into CBOR and embedded in the file.
pub struct NcfHeader {
    /// The model/file metadata.
    pub metadata: Metadata,
}

impl NcfHeader {
    /// Encode this header into CBOR bytes.
    pub fn encode_cbor(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        into_writer(self, &mut buffer)?;
        Ok(buffer)
    }
    /// Encode this header into CBOR bytes.
    pub fn decode_cbor(bytes: &[u8]) -> Result<Self> {
        let cursor = std::io::Cursor::new(bytes);
        let header: NcfHeader = from_reader(cursor)?;
        Ok(header)
    }
}

#[derive(Debug, Clone, Copy)]
/// Binary prefix that appears at the start of every NCF file describing offsets.
pub struct FileHeaderPrefix {
    /// Magic bytes identifying the file format.
    pub magic: [u8; 8],
    /// Format version.
    pub version: u32,
    /// Flags indicating file features.
    pub flags: NcfFlags,
    /// Length of the CBOR header in bytes.
    pub header_len: u64,
    /// Offset where schema block begins (from file start).
    pub schema_offset: u64,
    /// Offset where index block begins (from file start).
    pub index_offset: u64,
    /// Number of chunks stored in the file.
    pub chunk_count: u64,
}

impl FileHeaderPrefix {
    /// Create a new empty header prefix with the given flags.
    pub fn new(flags: NcfFlags) -> Self {
        Self {
            magic: *NCF_MAGIC,
            version: 0x00010000,
            flags,
            header_len: 0,
            schema_offset: 0,
            index_offset: 0,
            chunk_count: 0,
        }
    }

    /// Encode the file header prefix into its binary representation.
    pub fn encode(&self) -> [u8; 48] {
        let mut bytes = [0u8; 48];
        bytes[..8].copy_from_slice(&self.magic);
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.flags.bits().to_le_bytes());
        bytes[16..24].copy_from_slice(&self.header_len.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.schema_offset.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.index_offset.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.chunk_count.to_le_bytes());
        bytes
    }

    /// Decode a file header prefix from the provided bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 48 {
            return Err(NcfError::Header("Header prefix too short".into()));
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[..8]);
        if &magic != NCF_MAGIC {
            return Err(NcfError::Header("Invalid magic bytes".into()));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let flags = NcfFlags::from_bits_truncate(u32::from_le_bytes(bytes[12..16].try_into().unwrap()));
        let header_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let schema_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let index_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let chunk_count = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
        Ok(Self {
            magic,
            version,
            flags,
            header_len,
            schema_offset,
            index_offset,
            chunk_count,
        })
    }
}

impl fmt::Display for NcfFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.contains(NcfFlags::COMPRESSED) {
            parts.push("COMPRESSED");
        }
        if self.contains(NcfFlags::ENCRYPTED) {
            parts.push("ENCRYPTED");
        }
        if self.contains(NcfFlags::STREAMING_SAFE) {
            parts.push("STREAMING_SAFE");
        }
        write!(f, "{}", parts.join("|"))
    }
}
