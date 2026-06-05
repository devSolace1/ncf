use memmap2::Mmap;
use ciborium::de::from_reader;
use lz4_flex::decompress_size_prepended;
use ncf_core::chunk::ChunkHeader;
use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::IndexEntry;
use ncf_core::schema::{Compression, TensorSchema};
use ncf_core::constants::*;
use ncf_core::Result;
use self_cell::self_cell;
use serde::Deserialize;
use snap::raw::Decoder as SnappyDecoder;
use std::collections::BTreeMap;
use std::convert::TryInto;
use std::fs::File;
use std::io::{Cursor, ErrorKind};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::OnceLock;

#[derive(Debug)]
/// Borrowed view of an NCF index decoded from the file.
pub struct BorrowedNcfIndex {
    /// Number of entries in the index.
    pub entry_count: u64,
    /// Index entries.
    pub entries: Vec<IndexEntry>,
    /// Mapping from tensor name to chunk id.
    pub tensor_map: BTreeMap<String, u64>,
    /// Mapping from chunk id to index entry for fast lookup.
    pub chunk_map: BTreeMap<u64, IndexEntry>,
}

#[derive(Debug, Deserialize)]
struct RawBorrowedNcfIndex {
    pub entry_count: u64,
    pub entries: Vec<IndexEntry>,
    tensor_map: BTreeMap<String, u64>,
}

self_cell! {
    /// Reader that owns a memory map and exposes borrowed dependent data.
    pub struct NcfReader {
        owner: Mmap,
        #[covariant]
        dependent: NcfReaderData,
    }
}

#[derive(Debug)]
/// Owned dependent data stored alongside the memory map.
pub struct NcfReaderData<'this> {
    /// Parsed header metadata.
    pub metadata: NcfHeader,
    /// Lazily-initialized schema list.
    pub schemas: OnceLock<std::result::Result<Vec<TensorSchema>, String>>,
    /// Lazily-initialized schema name -> index mapping for O(log n) lookup.
    pub schema_map: OnceLock<BTreeMap<String, usize>>,
    /// Byte range of the schema block within the file.
    pub schema_range: std::ops::Range<usize>,
    /// Borrowed index data referencing the mapped memory.
    pub index: BorrowedNcfIndex,
    /// Parsed file header prefix.
    pub header_prefix: FileHeaderPrefix,
    marker: PhantomData<&'this ()>,
}

impl NcfReader {
    /// Open an NCF file and return a reader providing borrowed access.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if (mmap.len() as u64) < FILE_HEADER_PREFIX_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!("file too small: {} bytes, need at least {}", mmap.len(), FILE_HEADER_PREFIX_SIZE)
            ).into());
        }

        let reader = Self::try_new(mmap, |mmap| {
            let header_prefix = FileHeaderPrefix::decode(&mmap[..FILE_HEADER_PREFIX_SIZE as usize])
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
            
            let header_start = FILE_HEADER_PREFIX_SIZE as usize;
            let header_len = header_prefix.header_len;
            if header_len > MAX_HEADER_SIZE {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("header size {} exceeds maximum allowed {}", header_len, MAX_HEADER_SIZE),
                ));
            }
            let header_end = header_start.checked_add(header_len as usize)
                .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "header size overflow"))?;
            if header_end > mmap.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("header block out of bounds: end={}, file_size={}", header_end, mmap.len())
                ));
            }
            
            let metadata = NcfHeader::decode_cbor(&mmap[header_start..header_end])
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;

            let schema_start = header_prefix.schema_offset as usize;
            let schema_end = header_prefix.index_offset as usize;
            if schema_start > schema_end || schema_end > mmap.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("schema block out of bounds: start={}, end={}, file_size={}", 
                        schema_start, schema_end, mmap.len())
                ));
            }
            let schema_len = schema_end.saturating_sub(schema_start) as u64;
            if schema_len > MAX_SCHEMA_SIZE {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("schema block size {} exceeds maximum allowed {}", schema_len, MAX_SCHEMA_SIZE),
                ));
            }
            let schema_range = schema_start..schema_end;

            const FOOTER_SIZE: usize = 16; // 8 bytes magic + 8 bytes length
            if mmap.len() < FOOTER_SIZE {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "file too small to contain footer"
                ));
            }
            
            let footer_position = mmap.len() - FOOTER_SIZE;
            let footer_magic = &mmap[footer_position..footer_position + 8];
            if footer_magic != b"NCFEND!!" {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing or invalid footer magic"
                ));
            }

            let footer_len_bytes: [u8; 8] = mmap[footer_position + 8..footer_position + 16]
                .try_into()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid footer length"))?;
            let index_len = u64::from_le_bytes(footer_len_bytes);
            if index_len > MAX_INDEX_SIZE {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("index size {} exceeds maximum allowed {}", index_len, MAX_INDEX_SIZE),
                ));
            }
            let index_len = index_len as usize;
            let index_start = header_prefix.index_offset as usize;
            let index_end = index_start.checked_add(index_len)
                .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "index size overflow"))?;
            
            if index_end > footer_position {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("index block overlaps footer: end={}, footer_pos={}", index_end, footer_position)
                ));
            }

            let raw_index: RawBorrowedNcfIndex = from_reader(Cursor::new(&mmap[index_start..index_end]))
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
            let tensor_map: BTreeMap<String, u64> = raw_index.tensor_map;

            let mut chunk_map = BTreeMap::new();
            for entry in &raw_index.entries {
                // IndexEntry is Copy; this is now optimized to a simple copy
                chunk_map.insert(entry.chunk_id, *entry);
            }

            let index = BorrowedNcfIndex {
                entry_count: raw_index.entry_count,
                entries: raw_index.entries,
                tensor_map,
                chunk_map,
            };

            Ok(NcfReaderData {
                metadata,
                schemas: OnceLock::new(),
                schema_map: OnceLock::new(),
                schema_range,
                index,
                header_prefix,
                marker: PhantomData,
            })
        })?;

        Ok(reader)
    }

    /// Print basic info about the NCF file to stdout (for debugging).
    pub fn inspect(&self) -> Result<()> {
        let schemas = self.schemas()?;
        println!("Model: {}", self.borrow_dependent().metadata.metadata.model_name);
        println!("Architecture: {}", self.borrow_dependent().metadata.metadata.architecture);
        println!("Tensors: {}", schemas.len());
        for tensor in schemas.iter() {
            println!(" - {} {} {:?}", tensor.name, tensor.dtype, tensor.shape);
        }
        Ok(())
    }

    /// Find a tensor schema by name using O(log n) lookup.
    ///
    /// Uses the indexed schema_map for efficient lookup instead of linear scan.
    pub fn find_schema(&self, name: &str) -> Result<Option<&TensorSchema>> {
        // Ensure schemas are loaded, which also initializes schema_map
        let schemas = self.schemas()?;
        
        self.with_dependent(|_owner, data| {
            // schema_map is guaranteed to exist after schemas() call
            let schema_map = data.schema_map.get()
                .expect("schema_map should be initialized after schemas() call");
            
            match schema_map.get(name) {
                Some(&idx) => Ok(Some(&schemas[idx])),
                None => Ok(None),
            }
        })
    }

    /// Return the decoded NCF header metadata.
    pub fn metadata(&self) -> &NcfHeader {
        &self.borrow_dependent().metadata
    }

    /// Return the number of schemas/tensors in the file.
    pub fn schema_count(&self) -> Result<usize> {
        Ok(self.borrow_dependent().index.tensor_map.len())
    }

    /// Return a zero-copy slice for a tensor payload by name, if present.
    ///
    /// For compressed tensors, this returns the raw compressed payload bytes.
    /// Use `read_tensor()` to get a fully decompressed payload instead.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let chunk_id = self.borrow_dependent().index.tensor_map.get(name)?;
        let entry = self.borrow_dependent().index.chunk_map.get(chunk_id)?;
        let data = self.borrow_owner();

        let offset_start = (entry.byte_offset as usize)
            .checked_add(CHUNK_HEADER_SIZE as usize)?;
        if offset_start > data.len() {
            return None;
        }

        let chunk_total_len = entry.byte_len as usize;
        let chunk_overhead = (CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize;
        if chunk_total_len < chunk_overhead {
            return None;
        }

        let data_len = chunk_total_len - chunk_overhead;
        let offset_end = offset_start.checked_add(data_len)?;
        if offset_end > data.len() {
            return None;
        }

        Some(&data[offset_start..offset_end])
    }

    /// Verify the integrity and checksum of a single named tensor.
    pub fn verify_tensor(&self, name: &str) -> Result<bool> {
        let schema = self
            .find_schema(name)?
            .ok_or_else(|| std::io::Error::new(ErrorKind::NotFound, "tensor schema not found"))?;

        for chunk in &schema.chunks {
            if !self.verify_chunk(schema, chunk)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn verify_chunk(&self, schema: &TensorSchema, chunk: &ncf_core::schema::ChunkRef) -> Result<bool> {
        let data = self.borrow_owner();

        let header_start = chunk.byte_offset as usize;
        let header_end = header_start
            .checked_add(CHUNK_HEADER_SIZE as usize)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "chunk header overflow"))?;
        if header_end > data.len() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("chunk header out of bounds: end={}, file_size={}", header_end, data.len()),
            )
            .into());
        }

        let chunk_header = ChunkHeader::decode(&data[header_start..header_end])?;
        if chunk_header.chunk_id != chunk.chunk_id {
            return Ok(false);
        }
        if chunk_header.flags == 0 && schema.compression != Compression::None {
            return Ok(false);
        }
        if chunk_header.flags != 0 && schema.compression == Compression::None {
            return Ok(false);
        }

        let offset_start = header_end;
        let chunk_total_len = chunk.byte_len as usize;
        let chunk_overhead = (CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize;
        if chunk_total_len < chunk_overhead {
            return Ok(false);
        }

        let payload_len = chunk_total_len - chunk_overhead;
        if chunk_header.compressed_len != payload_len as u64 {
            return Ok(false);
        }

        let offset_end = offset_start.checked_add(payload_len).ok_or_else(|| {
            std::io::Error::new(ErrorKind::InvalidData, "chunk payload size overflow")
        })?;
        if offset_end > data.len() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("chunk payload out of bounds: end={}, file_size={}", offset_end, data.len()),
            )
            .into());
        }

        let payload = &data[offset_start..offset_end];
        let checksum_start = offset_end;
        let checksum_end = checksum_start.checked_add(CHUNK_CHECKSUM_SIZE as usize).ok_or_else(|| {
            std::io::Error::new(ErrorKind::InvalidData, "checksum range overflow")
        })?;
        if checksum_end > data.len() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("checksum out of bounds: end={}, file_size={}", checksum_end, data.len()),
            )
            .into());
        }

        let stored_checksum = &data[checksum_start..checksum_end];
        if stored_checksum != chunk.checksum {
            return Ok(false);
        }

        let decompressed = match schema.compression {
            Compression::None => payload.to_vec(),
            Compression::Zstd(_) => zstd::decode_all(payload).map_err(|err| {
                std::io::Error::new(ErrorKind::InvalidData, format!("Zstd decompression failed: {}", err))
            })?,
            Compression::Lz4 => decompress_size_prepended(payload).map_err(|err| {
                std::io::Error::new(ErrorKind::InvalidData, format!("LZ4 decompression failed: {}", err))
            })?,
            Compression::Snappy => {
                let mut decoder = SnappyDecoder::new();
                decoder.decompress_vec(payload).map_err(|err| {
                    std::io::Error::new(ErrorKind::InvalidData, format!("Snappy decompression failed: {}", err))
                })?
            }
        };

        if decompressed.len() as u64 != chunk.uncompressed_len {
            return Ok(false);
        }
        if chunk_header.uncompressed_len != chunk.uncompressed_len {
            return Ok(false);
        }

        let computed = blake3::hash(&decompressed);
        Ok(*computed.as_bytes() == chunk.checksum)
    }

    /// Verify every tensor in the file and return true only if all tensors pass validation.
    pub fn verify_all(&self) -> Result<bool> {
        for schema in self.schemas()? {
            if !self.verify_tensor(&schema.name)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Return the parsed file header prefix.
    pub fn header_prefix(&self) -> FileHeaderPrefix {
        self.borrow_dependent().header_prefix
    }

    /// Lazily decode and return the tensor schemas.
    pub fn schemas(&self) -> Result<&[TensorSchema]> {
        self.with_dependent(|owner, data| {
            let schemas_cell = data.schemas.get_or_init(|| {
                let schema_bytes = &owner[data.schema_range.clone()];
                from_reader(Cursor::new(schema_bytes)).map_err(|err| err.to_string())
            });

            match schemas_cell.as_ref() {
                Ok(schemas) => {
                    // Build schema_map on first access for O(log n) lookups
                    data.schema_map.get_or_init(|| {
                        let mut map = BTreeMap::new();
                        for (idx, schema) in schemas.iter().enumerate() {
                            map.insert(schema.name.clone(), idx);
                        }
                        map
                    });
                    Ok(schemas.as_slice())
                }
                Err(err) => Err(ncf_core::NcfError::Header(err.clone())),
            }
        })
    }

    /// Read and return the full tensor payload bytes for the given name.
    pub fn read_tensor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let schema = match self.find_schema(name)? {
            Some(schema) => schema,
            None => return Ok(None),
        };

        let data = self.borrow_owner();
        let mut result = Vec::new();

        for chunk in &schema.chunks {
            let header_start = chunk.byte_offset as usize;
            let header_end = header_start
                .checked_add(CHUNK_HEADER_SIZE as usize)
                .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "chunk header overflow"))?;
            if header_end > data.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("chunk header out of bounds: end={}, file_size={}", header_end, data.len()),
                )
                .into());
            }

            let chunk_header = ChunkHeader::decode(&data[header_start..header_end])?;
            if chunk_header.flags == 0 && schema.compression != Compression::None {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "schema indicates compression but chunk header is uncompressed",
                )
                .into());
            }
            if chunk_header.flags != 0 && schema.compression == Compression::None {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "chunk header indicates compression but schema is uncompressed",
                )
                .into());
            }

            let offset_start = header_end;
            let chunk_total_len = chunk.byte_len as usize;
            let chunk_overhead = (CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize;
            if chunk_total_len < chunk_overhead {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("chunk size too small: total_len={}, overhead={}", chunk_total_len, chunk_overhead),
                )
                .into());
            }

            let payload_len = chunk_total_len - chunk_overhead;
            let offset_end = offset_start.checked_add(payload_len).ok_or_else(|| {
                std::io::Error::new(ErrorKind::InvalidData, "chunk payload size overflow")
            })?;
            if offset_end > data.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("chunk payload out of bounds: end={}, file_size={}", offset_end, data.len()),
                )
                .into());
            }

            let chunk_payload = &data[offset_start..offset_end];
            let decompressed = match schema.compression {
                Compression::None => chunk_payload.to_vec(),
                Compression::Zstd(_) => zstd::decode_all(chunk_payload).map_err(|err| {
                    std::io::Error::new(ErrorKind::InvalidData, format!("Zstd decompression failed: {}", err))
                })?,
                Compression::Lz4 => decompress_size_prepended(chunk_payload).map_err(|err| {
                    std::io::Error::new(ErrorKind::InvalidData, format!("LZ4 decompression failed: {}", err))
                })?,
                Compression::Snappy => {
                    let mut decoder = SnappyDecoder::new();
                    decoder.decompress_vec(chunk_payload).map_err(|err| {
                        std::io::Error::new(ErrorKind::InvalidData, format!("Snappy decompression failed: {}", err))
                    })?
                }
            };

            if decompressed.len() as u64 != chunk_header.uncompressed_len {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "decompressed chunk size mismatch: expected {}, got {}",
                        chunk_header.uncompressed_len,
                        decompressed.len()
                    ),
                )
                .into());
            }

            result.extend_from_slice(&decompressed);
        }
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use ncf_core::header::Metadata;
    use ncf_core::schema::{DType, Encoding, Layout, Compression};
    use crate::writer::NcfWriter;

    #[test]
    fn test_find_schema_with_1000_tensors_uses_fast_lookup() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("test_1000_tensors.ncf");

        // Create 1000 tensors using NcfWriter
        let metadata = NcfHeader {
            metadata: Metadata {
                model_name: "test_model".into(),
                architecture: "test_arch".into(),
                created_at: 0,
                author: None,
                license: None,
                quantization: None,
                custom: Default::default(),
            }
        };

        let mut writer = NcfWriter::new(metadata, ncf_core::header::NcfFlags::empty());

        // Add 1000 unique tensors
        let num_tensors = 1000;
        let test_names: Vec<String> = (0..num_tensors)
            .map(|i| format!("tensor_{:04}", i))
            .collect();

        for (_i, name) in test_names.iter().enumerate() {
            let schema = TensorSchema {
                name: name.clone(),
                dtype: DType::F32,
                shape: vec![64],
                column_layout: Layout::RowMajor,
                compression: Compression::None,
                encoding: Encoding::Plain,
                chunks: vec![],
            };
            let payload = vec![0u8; 256]; // 64 * 4 bytes
            writer.add_tensor(schema, payload);
        }

        writer.finalize(&path).expect("Failed to write NCF file");

        // Now read back and test find_schema performance
        let reader = NcfReader::open(&path).expect("Failed to open NCF file");

        // Verify we have the correct number of tensors
        assert_eq!(reader.schema_count().unwrap(), num_tensors);

        // Test that all tensors can be found correctly using find_schema()
        for (_i, name) in test_names.iter().enumerate() {
            let found = reader.find_schema(name)
                .expect("Failed to find schema")
                .expect("Schema not found");
            
            assert_eq!(found.name, *name);
            assert_eq!(found.shape, vec![64]);
        }

        // Test that non-existent tensors return None
        assert!(reader.find_schema("nonexistent_tensor")
            .expect("Failed to query non-existent tensor")
            .is_none());

        // Verify schema_map was created and populated correctly
        let dependent = reader.borrow_dependent();
        assert!(dependent.schema_map.get().is_some(), "schema_map should be populated");
        
        let schema_map = dependent.schema_map.get().unwrap();
        assert_eq!(schema_map.len(), num_tensors, "schema_map should have all {} tensors", num_tensors);

        // Verify that schema_map indexing is correct
        for (i, name) in test_names.iter().enumerate() {
            let idx = schema_map.get(name)
                .expect(&format!("tensor {} should be in schema_map", name));
            assert_eq!(*idx, i, "schema_map index mismatch for {}", name);
        }

        // Verify schemas are still accessible and match
        let schemas = reader.schemas().expect("Failed to get schemas");
        assert_eq!(schemas.len(), num_tensors);
        for (i, schema) in schemas.iter().enumerate() {
            assert_eq!(schema.name, test_names[i]);
        }
    }

    #[test]
    fn test_find_schema_correctness_with_large_names() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("test_large_names.ncf");

        let metadata = NcfHeader {
            metadata: Metadata {
                model_name: "test".into(),
                architecture: "test".into(),
                created_at: 0,
                author: None,
                license: None,
                quantization: None,
                custom: Default::default(),
            }
        };

        let mut writer = NcfWriter::new(metadata, ncf_core::header::NcfFlags::empty());

        // Create tensors with various name patterns to test BTreeMap ordering
        let test_names = vec![
            "zzz_tensor",
            "aaa_tensor",
            "mmm_tensor",
            "bbb_tensor",
        ];

        for name in &test_names {
            let schema = TensorSchema {
                name: name.to_string(),
                dtype: DType::F32,
                shape: vec![32],
                column_layout: Layout::RowMajor,
                compression: Compression::None,
                encoding: Encoding::Plain,
                chunks: vec![],
            };
            let payload = vec![0u8; 128];
            writer.add_tensor(schema, payload);
        }

        writer.finalize(&path).expect("Failed to write NCF file");

        let reader = NcfReader::open(&path).expect("Failed to open NCF file");

        // Verify all tensors can be found regardless of insertion order
        for name in &test_names {
            let found = reader.find_schema(name)
                .expect("Failed to find schema")
                .expect("Schema not found");
            assert_eq!(&found.name, name);
        }

        // Verify schema_map maintains order
        let dependent = reader.borrow_dependent();
        let schema_map = dependent.schema_map.get().unwrap();
        
        let map_keys: Vec<_> = schema_map.keys().collect();
        let mut sorted_names = test_names.to_vec();
        sorted_names.sort();
        
        for (i, name) in sorted_names.iter().enumerate() {
            assert_eq!(map_keys[i], name, "schema_map should maintain sorted order");
        }
    }
}
