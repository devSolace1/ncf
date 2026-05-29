use ciborium::ser::into_writer;
use ncf_core::chunk::ChunkHeader;
use ncf_core::header::{FileHeaderPrefix, NCF_MAGIC, NcfHeader, NcfFlags};
use ncf_core::index::{IndexEntry, NcfIndex};
use ncf_core::schema::{ChunkRef, Compression, TensorSchema};
use ncf_core::Result;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub struct NcfWriter {
    pub metadata: NcfHeader,
    pub flags: NcfFlags,
    pub tensors: Vec<(TensorSchema, Vec<u8>)>,
}

impl NcfWriter {
    pub fn new(metadata: NcfHeader, flags: NcfFlags) -> Self {
        Self {
            metadata,
            flags,
            tensors: Vec::new(),
        }
    }

    pub fn add_tensor(&mut self, schema: TensorSchema, payload: Vec<u8>) {
        self.tensors.push((schema, payload));
    }

    pub fn finalize<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let header_bytes = self.metadata.encode_cbor()?;
        let header_len = header_bytes.len() as u64;

        let mut schemas: Vec<TensorSchema> = self
            .tensors
            .iter()
            .map(|(tensor, _)| {
                let mut clone = tensor.clone();
                clone.chunks = Vec::new();
                clone
            })
            .collect();

        let mut schema_bytes = Vec::new();
        into_writer(&schemas, &mut schema_bytes)?;

        let mut chunk_id = 0u64;
        let mut index_entries = Vec::new();
        let mut tensor_map = BTreeMap::new();
        let mut chunk_data = Vec::new();

        for (tensor, payload) in &self.tensors {
            let raw = payload.clone();
            let checksum = blake3::hash(&raw);
            let chunk_header = ChunkHeader {
                chunk_id,
                flags: if tensor.compression != Compression::None { 1 } else { 0 },
                uncompressed_len: raw.len() as u64,
                compressed_len: raw.len() as u64,
            };
            let chunk_offset = 48 + header_len + schema_bytes.len() as u64 + chunk_data.len() as u64;
            chunk_data.extend_from_slice(&chunk_header.encode());
            chunk_data.extend_from_slice(&raw);
            chunk_data.extend_from_slice(checksum.as_bytes());

            let chunk_ref = ChunkRef {
                chunk_id,
                byte_offset: chunk_offset,
                byte_len: chunk_header.encode().len() as u64 + raw.len() as u64 + 32,
                uncompressed_len: raw.len() as u64,
                checksum: *checksum.as_bytes(),
            };
            if let Some(schema) = schemas.iter_mut().find(|schema| schema.name == tensor.name) {
                schema.chunks.push(chunk_ref.clone());
            }
            index_entries.push(IndexEntry {
                chunk_id,
                byte_offset: chunk_offset,
                byte_len: chunk_ref.byte_len,
                tensor_name_hash: xxhash_rust::xxh3::xxh3_64(tensor.name.as_bytes()),
            });
            tensor_map.entry(tensor.name.clone()).or_insert(chunk_id);
            chunk_id += 1;
        }

        let mut final_schema_bytes = Vec::new();
        into_writer(&schemas, &mut final_schema_bytes)?;

        if final_schema_bytes.len() != schema_bytes.len() {
            schema_bytes = final_schema_bytes.clone();
            chunk_data.clear();
            index_entries.clear();
            tensor_map.clear();
            chunk_id = 0;
            for (tensor, payload) in &self.tensors {
                let raw = payload.clone();
                let checksum = blake3::hash(&raw);
                let chunk_header = ChunkHeader {
                    chunk_id,
                    flags: if tensor.compression != Compression::None { 1 } else { 0 },
                    uncompressed_len: raw.len() as u64,
                    compressed_len: raw.len() as u64,
                };
                let chunk_offset = 48 + header_len + schema_bytes.len() as u64 + chunk_data.len() as u64;
                chunk_data.extend_from_slice(&chunk_header.encode());
                chunk_data.extend_from_slice(&raw);
                chunk_data.extend_from_slice(checksum.as_bytes());
                let chunk_ref = ChunkRef {
                    chunk_id,
                    byte_offset: chunk_offset,
                    byte_len: chunk_header.encode().len() as u64 + raw.len() as u64 + 32,
                    uncompressed_len: raw.len() as u64,
                    checksum: *checksum.as_bytes(),
                };
                if let Some(schema) = schemas.iter_mut().find(|schema| schema.name == tensor.name) {
                    schema.chunks.clear();
                    schema.chunks.push(chunk_ref.clone());
                }
                index_entries.push(IndexEntry {
                    chunk_id,
                    byte_offset: chunk_offset,
                    byte_len: chunk_ref.byte_len,
                    tensor_name_hash: xxhash_rust::xxh3::xxh3_64(tensor.name.as_bytes()),
                });
                tensor_map.entry(tensor.name.clone()).or_insert(chunk_id);
                chunk_id += 1;
            }
            final_schema_bytes.clear();
            into_writer(&schemas, &mut final_schema_bytes)?;
        }

        let schema_offset = 48 + header_len;
        let index_offset = 48 + header_len + final_schema_bytes.len() as u64 + chunk_data.len() as u64;
        let index = NcfIndex::new(index_entries, tensor_map);
        let mut index_bytes = Vec::new();
        into_writer(&index, &mut index_bytes)?;
        let footer_len = (index_bytes.len() as u64).to_le_bytes();

        let mut buffer = Vec::with_capacity(
            48 + header_len as usize + final_schema_bytes.len() + chunk_data.len() + index_bytes.len() + 16,
        );
        let header_prefix = FileHeaderPrefix {
            magic: *NCF_MAGIC,
            version: 0x00010000,
            flags: self.flags,
            header_len,
            schema_offset,
            index_offset,
            chunk_count: chunk_id,
        };
        buffer.extend_from_slice(&header_prefix.encode());
        buffer.extend_from_slice(&header_bytes);
        buffer.extend_from_slice(&final_schema_bytes);
        buffer.extend_from_slice(&chunk_data);
        buffer.extend_from_slice(&index_bytes);
        buffer.extend_from_slice(b"NCFEND!!");
        buffer.extend_from_slice(&footer_len);

        let mut file = File::create(path)?;
        file.write_all(&buffer)?;
        Ok(())
    }
}
