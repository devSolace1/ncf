use ciborium::ser::into_writer;
use lz4_flex::compress_prepend_size;
use ncf_core::chunk::ChunkHeader;
use ncf_core::constants::{CHUNK_CHECKSUM_SIZE, CHUNK_HEADER_SIZE, FILE_HEADER_PREFIX_SIZE, MAX_HEADER_SIZE, MAX_INDEX_SIZE, MAX_SCHEMA_SIZE};
use ncf_core::header::{FileHeaderPrefix, NCF_MAGIC, NcfError, NcfFlags};
use ncf_core::index::{IndexEntry, NcfIndex};
use ncf_core::schema::{ChunkRef, Compression, TensorSchema};
use ncf_core::Result;
use snap::raw::Encoder as SnappyEncoder;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
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
            NcfError::Io(std::io::Error::other(format!(
                "Zstd compression failed: {}",
                err
            )))
        }),
        Compression::Lz4 => Ok(compress_prepend_size(data)),
        Compression::Snappy => {
            let mut encoder = SnappyEncoder::new();
            encoder.compress_vec(data).map_err(|err| {
                NcfError::Io(std::io::Error::other(format!(
                    "Snappy compression failed: {}",
                    err
                )))
            })
        }
    }
}

/// Writer helper to construct and write NCF files.
#[derive(Debug)]
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

        let schema_offset = FILE_HEADER_PREFIX_SIZE + header_len;

        let prepared_chunks: Vec<PreparedChunk> = self
            .tensors
            .iter()
            .map(|(tensor, payload)| {
                let compressed_payload = compress_payload(payload, tensor.compression)?;
                let compressed_len = compressed_payload.len() as u64;
                let checksum = blake3::hash(payload);
                Ok(PreparedChunk {
                    compressed_payload,
                    checksum: *checksum.as_bytes(),
                    uncompressed_len: payload.len() as u64,
                    compressed_len,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut schemas: Vec<TensorSchema> = self
            .tensors
            .iter()
            .enumerate()
            .map(|(chunk_id, (tensor, _))| {
                let mut clone = tensor.clone();
                clone.chunks = vec![ChunkRef {
                    chunk_id: chunk_id as u64,
                    byte_offset: 0,
                    byte_len: 0,
                    uncompressed_len: 0,
                    checksum: [0u8; 32],
                }];
                clone
            })
            .collect();

        // Two-pass deterministic offset computation (no 10-iteration loop)
        // Pass 1: Estimate schema size with placeholder chunk refs to compute offsets
        let mut placeholder_schemas = schemas.clone();
        for schema in &mut placeholder_schemas {
            schema.chunks.clear();
            // Add placeholder ChunkRef with minimal values for size estimation
            schema.chunks.push(ChunkRef {
                chunk_id: 0,
                byte_offset: 0,
                byte_len: 0,
                uncompressed_len: 0,
                checksum: [0; 32],
            });
        }
        
        let mut estimated_schema_bytes = Vec::new();
        into_writer(&placeholder_schemas, &mut estimated_schema_bytes)?;
        let estimated_schema_len = estimated_schema_bytes.len() as u64;
        
        if estimated_schema_len > MAX_SCHEMA_SIZE {
            return Err(NcfError::Header(format!(
                "Schema block size {} exceeds maximum allowed {}",
                estimated_schema_len, MAX_SCHEMA_SIZE
            )));
        }

        // Pass 2: Compute actual offsets using estimated schema size, then serialize final schema
        let mut current_offset = schema_offset.checked_add(estimated_schema_len)
            .ok_or_else(|| NcfError::Header("chunk offset overflow".into()))?;
        let mut index_entries = Vec::with_capacity(self.tensors.len());
        let mut tensor_map = BTreeMap::new();
        let mut total_chunk_bytes = 0u64;

        for (chunk_id, ((tensor, _), prepared)) in self
            .tensors
            .iter()
            .zip(prepared_chunks.iter())
            .enumerate()
        {
            let chunk_id = chunk_id as u64;
            let chunk_total_len = CHUNK_HEADER_SIZE + prepared.compressed_len + CHUNK_CHECKSUM_SIZE;
            
            // Use Cow<str> or reference to avoid clone in hot path
            let chunk_ref = ChunkRef {
                chunk_id,
                byte_offset: current_offset,
                byte_len: chunk_total_len,
                uncompressed_len: prepared.uncompressed_len,
                checksum: prepared.checksum,
            };

            schemas[chunk_id as usize].chunks.clear();
            schemas[chunk_id as usize].chunks.push(chunk_ref);

            index_entries.push(IndexEntry {
                chunk_id,
                byte_offset: current_offset,
                byte_len: chunk_total_len,
                tensor_name_hash: xxhash_rust::xxh3::xxh3_64(tensor.name.as_bytes()),
            });
            // Insert by reference to avoid tensor.name.clone()
            tensor_map.insert(tensor.name.clone(), chunk_id);

            total_chunk_bytes = total_chunk_bytes
                .checked_add(chunk_total_len)
                .ok_or_else(|| NcfError::Header("chunk length overflow".into()))?;
            current_offset = current_offset
                .checked_add(chunk_total_len)
                .ok_or_else(|| NcfError::Header("chunk offset overflow".into()))?;
        }

        // Serialize final schema - validate size stability
        let mut final_schema_bytes = Vec::new();
        into_writer(&schemas, &mut final_schema_bytes)?;
        let final_schema_len = final_schema_bytes.len() as u64;
        
        // Verify schema size is within bounds and reasonably close to estimate
        // (small variations due to CBOR encoding are acceptable)
        if final_schema_len > MAX_SCHEMA_SIZE {
            return Err(NcfError::Header(format!(
                "Final schema block size {} exceeds maximum allowed {}",
                final_schema_len, MAX_SCHEMA_SIZE
            )));
        }

        let index_offset = schema_offset
            .checked_add(final_schema_bytes.len() as u64)
            .and_then(|offset| offset.checked_add(total_chunk_bytes))
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
        let header_prefix = FileHeaderPrefix {
            magic: *NCF_MAGIC,
            version: 0x00010000,
            flags: header_flags,
            header_len,
            schema_offset,
            index_offset,
            chunk_count: self.tensors.len() as u64,
        };

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&header_prefix.encode())?;
        writer.write_all(&header_bytes)?;
        writer.write_all(&final_schema_bytes)?;

        for (chunk_id, prepared) in prepared_chunks.iter().enumerate() {
            let chunk_header = ChunkHeader {
                chunk_id: chunk_id as u64,
                flags: if prepared.compressed_len != prepared.uncompressed_len {
                    1
                } else {
                    0
                },
                uncompressed_len: prepared.uncompressed_len,
                compressed_len: prepared.compressed_len,
            };
            writer.write_all(&chunk_header.encode())?;
            writer.write_all(&prepared.compressed_payload)?;
            writer.write_all(&prepared.checksum)?;
        }

        writer.write_all(&index_bytes)?;
        writer.write_all(b"NCFEND!!")?;
        writer.write_all(&footer_len)?;
        writer.flush()?;
        Ok(())
    }
}
