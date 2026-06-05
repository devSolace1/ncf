use crate::Result;
use ncf_core::header::Metadata;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;

/// Number of tokens contained in each micro-batch block.
pub const BLOCK_TOKEN_COUNT: usize = 64;

/// File header length in bytes.
pub const KVCACHE_HEADER_SIZE: usize = 64;

/// Magic bytes that identify an ncf-kvcache file.
pub const KVCACHE_MAGIC: &[u8; 8] = b"NCFKVCCH";

/// Magic bytes that identify the chained index trailer.
pub const KVCACHE_TRAILER_MAGIC: &[u8; 8] = b"KVCFTR!!";

/// Persistent configuration for the cache file.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of transformer layers.
    pub layers: u32,
    /// Number of attention heads per layer.
    pub heads: u32,
    /// Bytes per per-head token component.
    pub element_bytes: u32,
}

impl KvCacheConfig {
    /// Total number of columns in the cache.
    pub fn column_count(&self) -> usize {
        (self.layers as usize)
            .checked_mul(self.heads as usize)
            .expect("layer/head overflow")
    }

    /// Size of one token frame across all columns.
    pub fn frame_stride(&self) -> usize {
        self.column_count()
            .checked_mul(self.element_bytes as usize)
            .expect("frame stride overflow")
    }

    /// Number of bytes stored for one column block.
    pub fn block_bytes(&self) -> usize {
        BLOCK_TOKEN_COUNT
            .checked_mul(self.element_bytes as usize)
            .expect("block bytes overflow")
    }
}

/// Fixed header data stored in the first page of the file.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KvCacheHeader {
    /// File magic bytes.
    pub magic: [u8; 8],
    /// Version number for the file format.
    pub version: u32,
    /// Feature flags.
    pub flags: u32,
    /// Number of tokens per block.
    pub block_size: u32,
    /// Total transformer layers.
    pub layers: u32,
    /// Attention heads per layer.
    pub heads: u32,
    /// Bytes per element.
    pub element_bytes: u32,
    /// Commit epoch used to reject stale pending flushes.
    pub commit_epoch: u64,
    /// Atomic valid token count for reader bounds enforcement.
    pub valid_token_count: u64,
    /// Offset of the latest index trailer.
    pub index_head_offset: u64,
    /// Length of the CBOR metadata block.
    pub metadata_len: u32,
    /// Final padding to align the header to 64 bytes.
    pub reserved1: u32,
}

impl KvCacheHeader {
    /// Create a new header using the provided file configuration.
    pub fn new(config: &KvCacheConfig, metadata_len: u32) -> Self {
        Self {
            magic: *KVCACHE_MAGIC,
            version: 1,
            flags: 0x1, // streaming-safe
            block_size: BLOCK_TOKEN_COUNT as u32,
            layers: config.layers,
            heads: config.heads,
            element_bytes: config.element_bytes,
            commit_epoch: 0,
            valid_token_count: 0,
            index_head_offset: 0,
            metadata_len,
            reserved1: 0,
        }
    }

    /// Serialize the header to a fixed 64-byte array.
    pub fn encode(&self) -> [u8; KVCACHE_HEADER_SIZE] {
        let mut bytes = [0u8; KVCACHE_HEADER_SIZE];
        bytes[..8].copy_from_slice(&self.magic);
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.flags.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.block_size.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.layers.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.heads.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.element_bytes.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.commit_epoch.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.valid_token_count.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.index_head_offset.to_le_bytes());
        bytes[56..60].copy_from_slice(&self.metadata_len.to_le_bytes());
        bytes[60..64].copy_from_slice(&self.reserved1.to_le_bytes());
        bytes
    }

    /// Decode a header from a byte slice.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < KVCACHE_HEADER_SIZE {
            return Err(crate::error::KvcacheError::Layout(
                "file too small to contain header".into(),
            ));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[..8]);
        if &magic != KVCACHE_MAGIC {
            return Err(crate::error::KvcacheError::Layout(
                "invalid kvcache magic".into(),
            ));
        }

        let version = u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid header version".into())
        })?);
        let flags = u32::from_le_bytes(bytes[12..16].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid header flags".into())
        })?);
        let block_size = u32::from_le_bytes(bytes[16..20].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid block size".into())
        })?);
        let layers = u32::from_le_bytes(bytes[20..24].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid layer count".into())
        })?);
        let heads = u32::from_le_bytes(bytes[24..28].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid head count".into())
        })?);
        let element_bytes = u32::from_le_bytes(bytes[28..32].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid element size".into())
        })?);
        let commit_epoch = u64::from_le_bytes(bytes[32..40].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid commit epoch".into())
        })?);
        let valid_token_count = u64::from_le_bytes(bytes[40..48].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid token count".into())
        })?);
        let index_head_offset = u64::from_le_bytes(bytes[48..56].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid index offset".into())
        })?);
        let metadata_len = u32::from_le_bytes(bytes[56..60].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid metadata length".into())
        })?);
        let reserved1 = u32::from_le_bytes(bytes[60..64].try_into().map_err(|_| {
            crate::error::KvcacheError::Layout("invalid reserved field".into())
        })?);

        Ok(Self {
            magic,
            version,
            flags,
            block_size,
            layers,
            heads,
            element_bytes,
            commit_epoch,
            valid_token_count,
            index_head_offset,
            metadata_len,
            reserved1,
        })
    }
}

/// Helper container for the optional CBOR metadata map.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvCacheMetadata {
    /// Model name for this cache.
    pub model_name: String,
    /// Optional architecture identifier.
    pub architecture: Option<String>,
    /// Arbitrary custom CBOR metadata.
    #[serde(default)]
    pub custom: std::collections::BTreeMap<String, ciborium::value::Value>,
}

impl KvCacheMetadata {
    /// Convert NCF header metadata into the cache metadata structure.
    pub fn from_ncf_header(metadata: &Metadata) -> Self {
        Self {
            model_name: metadata.model_name.to_owned(),
            architecture: Some(metadata.architecture.to_owned()),
            custom: metadata.custom.clone(),
        }
    }
}
