use crate::schema::TensorSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub chunk_id: u64,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub tensor_name_hash: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcfIndex {
    pub entry_count: u64,
    pub entries: Vec<IndexEntry>,
    pub tensor_map: BTreeMap<String, u64>,
}

impl NcfIndex {
    pub fn new(entries: Vec<IndexEntry>, tensor_map: BTreeMap<String, u64>) -> Self {
        let entry_count = entries.len() as u64;
        Self {
            entry_count,
            entries,
            tensor_map,
        }
    }

    pub fn find_chunk_id(&self, name: &str) -> Option<u64> {
        self.tensor_map.get(name).copied()
    }

    pub fn build_from_schemas(schemas: &[TensorSchema]) -> Self {
        let mut entries = Vec::new();
        let mut tensor_map = BTreeMap::new();
        for schema in schemas {
            if let Some(chunk_ref) = schema.chunks.first() {
                let name_hash = xxhash_rust::xxh3::xxh3_64(schema.name.as_bytes());
                entries.push(IndexEntry {
                    chunk_id: chunk_ref.chunk_id,
                    byte_offset: chunk_ref.byte_offset,
                    byte_len: chunk_ref.byte_len,
                    tensor_name_hash: name_hash,
                });
                tensor_map.insert(schema.name.clone(), chunk_ref.chunk_id);
            }
        }
        NcfIndex::new(entries, tensor_map)
    }
}
