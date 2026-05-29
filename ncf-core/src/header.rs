use crate::Result;
use ciborium::{de::from_reader, ser::into_writer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const NCF_MAGIC: &[u8; 8] = b"NCF\0\xDE\xAD\xBE\xEF";
pub const FOOTER_MAGIC: &[u8; 8] = b"NCFEND!!";

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct NcfFlags: u32 {
        const COMPRESSED = 0x1;
        const ENCRYPTED = 0x2;
        const STREAMING_SAFE = 0x4;
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NcfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CBOR deserialize error: {0}")]
    Cbor(#[from] ciborium::de::Error<std::io::Error>),
    #[error("CBOR serialize error: {0}")]
    CborSer(#[from] ciborium::ser::Error<std::io::Error>),
    #[error("Header parse error: {0}")]
    Header(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub model_name: String,
    pub architecture: String,
    pub created_at: u64,
    pub author: Option<String>,
    pub license: Option<String>,
    pub quantization: Option<String>,
    #[serde(default)]
    pub custom: BTreeMap<String, ciborium::value::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcfHeader {
    pub metadata: Metadata,
}

impl NcfHeader {
    pub fn encode_cbor(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        into_writer(self, &mut buffer)?;
        Ok(buffer)
    }

    pub fn decode_cbor(bytes: &[u8]) -> Result<Self> {
        let cursor = std::io::Cursor::new(bytes);
        let header: NcfHeader = from_reader(cursor)?;
        Ok(header)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FileHeaderPrefix {
    pub magic: [u8; 8],
    pub version: u32,
    pub flags: NcfFlags,
    pub header_len: u64,
    pub schema_offset: u64,
    pub index_offset: u64,
    pub chunk_count: u64,
}

impl FileHeaderPrefix {
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
