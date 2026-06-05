use memmap2::Mmap;
use ncf_core::header::FileHeaderPrefix;
use ncf_core::index::NcfIndex;
use ncf_core::schema::TensorSchema;
use ncf_core::constants::*;
use ncf_core::Result;
use std::fs::File;
use std::io::{Cursor, ErrorKind};
use std::path::Path;
use std::sync::OnceLock;

/// Memory-mapped view of an NCF file for zero-copy reads.
pub struct NcfMmap {
    /// Underlying memory map of the file.
    pub mmap: Mmap,
    /// Parsed file header prefix.
    pub header_prefix: FileHeaderPrefix,
    /// Decoded CBOR header metadata.
    pub metadata: ncf_core::header::NcfHeader,
    schemas: OnceLock<std::result::Result<Vec<TensorSchema>, String>>,
    schema_map: OnceLock<std::collections::BTreeMap<String, usize>>,
    schema_range: std::ops::Range<usize>,
    /// Parsed index information.
    pub index: NcfIndex,
}

impl NcfMmap {
    /// Open and memory-map the given file path as an NCF file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        // Bounds check: minimum file size
        if (mmap.len() as u64) < FILE_HEADER_PREFIX_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!("file too small: {} bytes, need at least {}", mmap.len(), FILE_HEADER_PREFIX_SIZE)
            ).into());
        }

        // Decode header prefix (first 48 bytes)
        let header_prefix = FileHeaderPrefix::decode(&mmap[..FILE_HEADER_PREFIX_SIZE as usize])?;
        
        // Bounds check: header block
        let header_len = header_prefix.header_len;
        if header_len > MAX_HEADER_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("header size {} exceeds maximum allowed {}", header_len, MAX_HEADER_SIZE),
            ).into());
        }
        let header_start = FILE_HEADER_PREFIX_SIZE as usize;
        let header_end = header_start.checked_add(header_len as usize)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "header size overflow"))?;
        if header_end > mmap.len() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("header block out of bounds: end={}, file_size={}", header_end, mmap.len())
            ).into());
        }
        
        let metadata = ncf_core::header::NcfHeader::decode_cbor(&mmap[header_start..header_end])?;
        
        // Bounds check: schema block
        let schema_start = header_prefix.schema_offset as usize;
        let schema_end = header_prefix.index_offset as usize;
        if schema_start > schema_end || schema_end > mmap.len() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("schema block out of bounds: start={}, end={}, file_size={}", 
                    schema_start, schema_end, mmap.len())
            ).into());
        }
        let schema_len = schema_end.saturating_sub(schema_start) as u64;
        if schema_len > MAX_SCHEMA_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("schema block size {} exceeds maximum allowed {}", schema_len, MAX_SCHEMA_SIZE),
            ).into());
        }
        let schema_range = schema_start..schema_end;

        // Bounds check: footer
        const FOOTER_SIZE: usize = 16; // 8 bytes magic + 8 bytes length
        if mmap.len() < FOOTER_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "file too small to contain footer"
            ).into());
        }
        
        let footer_position = mmap.len() - FOOTER_SIZE;
        let footer_magic = &mmap[footer_position..footer_position + 8];
        if footer_magic != b"NCFEND!!" {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "missing or invalid footer magic"
            ).into());
        }

        let footer_len_bytes: [u8; 8] = mmap[footer_position + 8..footer_position + 16]
            .try_into()
            .map_err(|_| std::io::Error::new(ErrorKind::InvalidData, "invalid footer length"))?;
        let index_len = u64::from_le_bytes(footer_len_bytes);
        if index_len > MAX_INDEX_SIZE {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("index size {} exceeds maximum allowed {}", index_len, MAX_INDEX_SIZE),
            ).into());
        }
        let index_len = index_len as usize;
        
        // Bounds check: index block
        let index_start = header_prefix.index_offset as usize;
        let index_end = index_start.checked_add(index_len)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "index size overflow"))?;
        if index_end > footer_position {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("index block overlaps footer: end={}, footer_pos={}", index_end, footer_position)
            ).into());
        }

        let mut index: NcfIndex = ciborium::de::from_reader(
            Cursor::new(&mmap[index_start..index_end])
        )?;
        index.chunk_map = index.entries.iter().cloned().map(|entry| (entry.chunk_id, entry)).collect();
        
        Ok(Self {
            mmap,
            header_prefix,
            metadata,
            schemas: OnceLock::new(),
            schema_map: OnceLock::new(),
            schema_range,
            index,
        })
    }

    /// Lazily decode and return the list of tensor schemas.
    pub fn schemas(&self) -> Result<&[TensorSchema]> {
        let schemas = self.schemas.get_or_init(|| {
            let slice = &self.mmap[self.schema_range.clone()];
            ciborium::de::from_reader(Cursor::new(slice)).map_err(|err| err.to_string())
        });

        match schemas.as_ref() {
            Ok(schemas) => {
                // Build schema_map on first access for O(log n) lookups
                self.schema_map.get_or_init(|| {
                    let mut map = std::collections::BTreeMap::new();
                    for (idx, schema) in schemas.iter().enumerate() {
                        map.insert(schema.name.clone(), idx);
                    }
                    map
                });
                Ok(schemas.as_slice())
            }
            Err(err) => Err(ncf_core::NcfError::Header(err.clone())),
        }
    }

    /// Return a zero-copy slice of the tensor payload for the given name.
    ///
    /// For compressed tensors, this returns the raw compressed payload bytes.
    /// Use a full read operation if you need decompressed tensor contents.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let chunk_id = self.index.tensor_map.get(name)?;
        let entry = self.index.chunk_map.get(chunk_id)?;
        
        // Bounds check: chunk offset is within file
        let offset_start = (entry.byte_offset as usize)
            .checked_add(CHUNK_HEADER_SIZE as usize)?;
        if offset_start > self.mmap.len() {
            return None;
        }

        // Calculate actual data length: total_len - header - checksum
        let chunk_total_len = entry.byte_len as usize;
        let chunk_overhead = (CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize;
        
        if chunk_total_len < chunk_overhead {
            return None;
        }
        
        let data_len = chunk_total_len - chunk_overhead;
        
        // Bounds check: slice end is within file
        let offset_end = offset_start.checked_add(data_len)?;
        if offset_end > self.mmap.len() {
            return None;
        }
        
        Some(&self.mmap[offset_start..offset_end])
    }
}
