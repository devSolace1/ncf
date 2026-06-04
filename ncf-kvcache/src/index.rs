use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One committed block payload entry in the chained index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkIndexEntry {
    /// Unique monotonic chunk identifier.
    pub chunk_id: u64,
    /// Transformer layer for this column.
    pub layer: u32,
    /// Attention head for this column.
    pub head: u32,
    /// Block index inside this column.
    pub block_idx: u64,
    /// Byte offset where the payload begins.
    pub byte_offset: u64,
    /// Total number of bytes written for this payload.
    pub byte_len: u64,
    /// Number of valid tokens in the block.
    pub token_count: u32,
}

/// An incremental chained index block appended after a payload commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexBlock {
    /// Offset of the previous index block trailer.
    pub prev_index_offset: u64,
    /// Block index being committed.
    pub block_idx: u64,
    /// Entries for every column in the current token block.
    pub entries: Vec<ChunkIndexEntry>,
}

/// In-memory index state reconstructed from the backward chain.
#[derive(Debug, Default)]
pub struct KvcacheIndex {
    /// Map by (layer, head, block_idx) for fast lookup.
    pub block_map: BTreeMap<(u32, u32, u64), ChunkIndexEntry>,
    /// Last committed chunk identifier.
    pub next_chunk_id: u64,
}

impl KvcacheIndex {
    /// Insert a set of entries from the chained index.
    pub fn insert_entries(&mut self, entries: Vec<ChunkIndexEntry>) {
        for entry in entries {
            self.next_chunk_id = self.next_chunk_id.max(entry.chunk_id.wrapping_add(1));
            self.block_map
                .insert((entry.layer, entry.head, entry.block_idx), entry);
        }
    }

    /// Find a block payload entry for a particular layer/head and block.
    pub fn get(&self, layer: u32, head: u32, block_idx: u64) -> Option<&ChunkIndexEntry> {
        self.block_map.get(&(layer, head, block_idx))
    }

    /// Build a tensor key name for diagnostics.
    pub fn tensor_key(layer: u32, head: u32) -> String {
        format!("layer.{}.head.{}.K", layer, head)
    }
}

/// Trailer appended after a CBOR index block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexTrailer {
    /// Magic bytes to validate the trailer.
    pub magic: [u8; 8],
    /// Serialized index block length in bytes.
    pub cbor_len: u64,
    /// Offset of the previous index trailer.
    pub prev_index_offset: u64,
}

impl IndexTrailer {
    /// Serialize the trailer to a fixed 24-byte buffer.
    pub fn encode(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        bytes[..8].copy_from_slice(&self.magic);
        bytes[8..16].copy_from_slice(&self.cbor_len.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.prev_index_offset.to_le_bytes());
        bytes
    }

    /// Decode a fixed 24-byte trailer buffer.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 24 {
            return Err("invalid trailer length".into());
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[..8]);
        let cbor_len = u64::from_le_bytes(
            bytes[8..16]
                .try_into()
                .map_err(|_| "invalid trailer CBOR len")?,
        );
        let prev_index_offset = u64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .map_err(|_| "invalid trailer prev offset")?,
        );
        Ok(Self {
            magic,
            cbor_len,
            prev_index_offset,
        })
    }
}
