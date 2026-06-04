use ciborium::ser::into_writer;
use lz4_flex::compress_prepend_size;
use ncf_core::chunk::{ChunkHeader, CHUNK_CHECKSUM_SIZE, CHUNK_HEADER_SIZE};
use ncf_core::constants::{FILE_HEADER_PREFIX_SIZE, MAX_HEADER_SIZE, MAX_INDEX_SIZE, MAX_SCHEMA_SIZE};
use ncf_core::header::{FileHeaderPrefix, NCF_MAGIC, NcfError, NcfFlags};
use ncf_core::index::{IndexEntry, NcfIndex};
use ncf_core::schema::{ChunkRef, Compression, TensorSchema};
use ncf_core::Result;
use snap::raw::Encoder as SnappyEncoder;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{ErrorKind, Write};
use std::path::Path;

#[derive(Debug, Clone)]
struct PreparedChunk {
    compressed_payload: Vec<u8>,
    checksum: [u8; 32],
    uncompressed_len: u64,
    compressed_len: u64,
}

fn compress_payload(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        Compression::Zstd(level) => zstd::encode_all(data, level.into()).map_err(|err| {
            NcfError::Io(std::io::Error::new(
                ErrorKind::Other,
                format!("Zstd compression failed: {}", err),
            ))
        }),
        Compression::Lz4 => Ok(compress_prepend_size(data)),
        Compression::Snappy => {
            let mut encoder = SnappyEncoder::new();
            encoder.compress_vec(data).map_err(|err| {
                NcfError::Io(std::io::Error::new(
                    ErrorKind::Other,
                    format!("Snappy compression failed: {}", err),
                ))
            })
        }
    }
}

/// Writer helper to construct and write NCF files.
pub struct NcfWriter {
    /// Header metadata to embed in the file.
    pub metadata: ncf_core::header::NcfHeader,
    /// File-level flags.
    pub flags: NcfFlags,
    /// Tensors to be written (schema + payload bytes).
    pub tensors: Vec<(TensorSchema, Vec<u8>)>,
}

impl NcfWriter {
    /// Create a new `NcfWriter` with given metadata and flags.
    pub fn new(metadata: ncf_core::header::NcfHeader, flags: NcfFlags) -> Self {
        Self {
            metadata,
            flags,
            tensors: Vec::new(),
        }
    }

    /// Add a tensor schema and its payload to be written.
    pub fn add_tensor(&mut self, schema: TensorSchema, payload: Vec<u8>) {
        self.tensors.push((schema, payload));
    }

    /// Finalize and write the NCF file to the specified path.
    pub fn finalize<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let header_bytes = self.metadata.encode_cbor()?;
        let header_len = header_bytes.len() as u64;
        if header_len > MAX_HEADER_SIZE {
            return Err(NcfError::Header(format!(
                "CBOR header size {} exceeds maximum allowed {}",
                header_len, MAX_HEADER_SIZE
            )));
        }

        let header_flags = if self
            .tensors
            .iter()
            .any(|(schema, _)| schema.compression != Compression::None)
        {
            self.flags | NcfFlags::COMPRESSED
        } else {
            self.flags
        };

        let mut schemas: Vec<TensorSchema> = self
            .tensors
            .iter()
            .map(|(tensor, _)| {
                let mut clone = tensor.clone();
                clone.chunks = Vec::new();
                clone
            })
            .collect();

        let chunk_payloads: Vec<PreparedChunk> = self
            .tensors
            .iter()
            .map(|(tensor, payload)| {
                let compressed = compress_payload(payload, tensor.compression)?;
                let checksum = blake3::hash(payload);
                Ok(PreparedChunk {
                    compressed_payload: compressed,
                    checksum: *checksum.as_bytes(),
                    uncompressed_len: payload.len() as u64,
                    compressed_len: compressed.len() as u64,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut schema_len_guess = 0u64;
        let mut index_entries = Vec::new();
        let mut tensor_map = BTreeMap::new();
        let mut chunk_data = Vec::new();
        let mut final_schema_bytes = Vec::new();

        for attempt in 0..10 {
            chunk_data.clear();
            index_entries.clear();
            tensor_map.clear();
            for schema in &mut schemas {
                schema.chunks.clear();
            }

            let schema_offset = FILE_HEADER_PREFIX_SIZE + header_len;
            let current_offset = schema_offset
                .checked_add(schema_len_guess)
                .ok_or_else(|| NcfError::Header("schema offset overflow".into()))?;

            for (chunk_id, ((tensor, _), payload)) in self
                .tensors
                .iter()
                .zip(chunk_payloads.iter())
                .enumerate()
            {
                let chunk_id = chunk_id as u64;
                let chunk_header = ChunkHeader {
                    chunk_id,
                    flags: if payload.compressed_len != payload.uncompressed_len {
                        1
                    } else {
                        0
                    },
                    uncompressed_len: payload.uncompressed_len,
                    compressed_len: payload.compressed_len,
                };
                let chunk_offset = current_offset + chunk_data.len() as u64;
                chunk_data.extend_from_slice(&chunk_header.encode());
                chunk_data.extend_from_slice(&payload.compressed_payload);
                chunk_data.extend_from_slice(&payload.checksum);

                let chunk_total_len = CHUNK_HEADER_SIZE + payload.compressed_len + CHUNK_CHECKSUM_SIZE;
                let chunk_ref = ChunkRef {
                    chunk_id,
                    byte_offset: chunk_offset,
                    byte_len: chunk_total_len,
                    uncompressed_len: payload.uncompressed_len,
                    checksum: payload.checksum,
                };
                schemas[chunk_id as usize].chunks.push(chunk_ref.clone());
                index_entries.push(IndexEntry {
                    chunk_id,
                    byte_offset: chunk_offset,
                    byte_len: chunk_total_len,
                    tensor_name_hash: xxhash_rust::xxh3::xxh3_64(tensor.name.as_bytes()),
                });
                tensor_map.entry(tensor.name.clone()).or_insert(chunk_id);
            }

            let mut candidate_schema_bytes = Vec::new();
            into_writer(&schemas, &mut candidate_schema_bytes)?;
            let candidate_schema_len = candidate_schema_bytes.len() as u64;

            if candidate_schema_len > MAX_SCHEMA_SIZE {
                return Err(NcfError::Header(format!(
                    "Schema block size {} exceeds maximum allowed {}",
                    candidate_schema_len, MAX_SCHEMA_SIZE
                )));
            }

            if candidate_schema_len == schema_len_guess {
                final_schema_bytes = candidate_schema_bytes;
                break;
            }
            schema_len_guess = candidate_schema_len;

            if attempt == 9 {
                return Err(NcfError::Header(
                    "unable to stabilize schema encoding size".into(),
                ));
            }
        }

        let schema_offset = FILE_HEADER_PREFIX_SIZE + header_len;
        let index_offset = schema_offset
            .checked_add(final_schema_bytes.len() as u64)
            .and_then(|offset| offset.checked_add(chunk_data.len() as u64))
            .ok_or_else(|| NcfError::Header("index offset overflow".into()))?;
        let index = NcfIndex::new(index_entries, tensor_map);

        let mut index_bytes = Vec::new();
        into_writer(&index, &mut index_bytes)?;
        if index_bytes.len() as u64 > MAX_INDEX_SIZE {
            return Err(NcfError::Header(format!(
                "Index block size {} exceeds maximum allowed {}",
                index_bytes.len(), MAX_INDEX_SIZE
            )));
        }

        let footer_len = (index_bytes.len() as u64).to_le_bytes();
        let mut buffer = Vec::with_capacity(
            48 + header_len as usize + final_schema_bytes.len() + chunk_data.len() + index_bytes.len() + 16,
        );
        let header_prefix = FileHeaderPrefix {
            magic: *NCF_MAGIC,
            version: 0x00010000,
            flags: header_flags,
            header_len,
            schema_offset,
            index_offset,
            chunk_count: self.tensors.len() as u64,
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
