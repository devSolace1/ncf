use memmap2::Mmap;
use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::IndexEntry;
use ncf_core::schema::ChunkRef;
use ncf_core::constants::*;
use ncf_core::Result;
use self_cell::self_cell;
use serde::Deserialize;
use serde_cbor::de::Deserializer as CborDeserializer;
use std::collections::{BTreeMap, HashMap};
use std::convert::TryInto;
use std::fs::File;
use std::io::ErrorKind;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct BorrowedTensorSchema<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    pub dtype: ncf_core::schema::DType,
    pub shape: Vec<u64>,
    pub column_layout: ncf_core::schema::Layout,
    pub compression: ncf_core::schema::Compression,
    pub encoding: ncf_core::schema::Encoding,
    pub chunks: Vec<ChunkRef>,
}

#[derive(Debug)]
pub struct BorrowedNcfIndex<'a> {
    pub entry_count: u64,
    pub entries: Vec<IndexEntry>,
    pub tensor_map: HashMap<&'a str, u64>,
}

#[derive(Debug, Deserialize)]
struct RawBorrowedNcfIndex<'a> {
    pub entry_count: u64,
    pub entries: Vec<IndexEntry>,
    #[serde(borrow)]
    tensor_map: BTreeMap<&'a str, u64>,
}

self_cell! {
    pub struct NcfReader {
        owner: Mmap,
        #[covariant]
        dependent: NcfReaderData,
    }
}

#[derive(Debug)]
pub struct NcfReaderData<'this> {
    pub metadata: NcfHeader,
    pub schemas: Vec<BorrowedTensorSchema<'this>>,
    pub index: BorrowedNcfIndex<'this>,
    pub header_prefix: FileHeaderPrefix,
}

impl NcfReader {
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
            let header_end = header_start.checked_add(header_prefix.header_len as usize)
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
            
            let mut schema_de = CborDeserializer::from_slice(&mmap[schema_start..schema_end]);
            let schemas: Vec<BorrowedTensorSchema<'_>> = Deserialize::deserialize(&mut schema_de)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;

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
            let index_len = u64::from_le_bytes(footer_len_bytes) as usize;
            let index_start = header_prefix.index_offset as usize;
            let index_end = index_start.checked_add(index_len)
                .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "index size overflow"))?;
            
            if index_end > footer_position {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("index block overlaps footer: end={}, footer_pos={}", index_end, footer_position)
                ));
            }

            let mut index_de = CborDeserializer::from_slice(&mmap[index_start..index_end]);
            let raw_index: RawBorrowedNcfIndex<'_> = Deserialize::deserialize(&mut index_de)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
            let mut tensor_map = HashMap::with_capacity(raw_index.tensor_map.len());
            for (name, chunk_id) in raw_index.tensor_map {
                tensor_map.insert(name, chunk_id);
            }

            let index = BorrowedNcfIndex {
                entry_count: raw_index.entry_count,
                entries: raw_index.entries,
                tensor_map,
            };

            Ok(NcfReaderData {
                metadata,
                schemas,
                index,
                header_prefix,
            })
        })?;

        Ok(reader)
    }

    pub fn inspect(&self) {
        self.with_dependent(|_owner, data| {
            println!("Model: {}", data.metadata.metadata.model_name);
            println!("Architecture: {}", data.metadata.metadata.architecture);
            println!("Tensors: {}", data.schemas.len());
            for tensor in data.schemas.iter() {
                println!(" - {} {} {:?}", tensor.name, tensor.dtype, tensor.shape);
            }
        })
    }

    pub fn find_schema(&self, name: &str) -> Option<&BorrowedTensorSchema<'_>> {
        self.borrow_dependent().schemas.iter().find(|schema| schema.name == name)
    }

    pub fn metadata(&self) -> &NcfHeader {
        &self.borrow_dependent().metadata
    }

    pub fn schema_count(&self) -> usize {
        self.borrow_dependent().schemas.len()
    }

    pub fn header_prefix(&self) -> FileHeaderPrefix {
        self.borrow_dependent().header_prefix
    }

    pub fn schemas(&self) -> &[BorrowedTensorSchema<'_>] {
        &self.borrow_dependent().schemas
    }

    pub fn read_tensor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let schema = match self.find_schema(name) {
            Some(schema) => schema,
            None => return Ok(None),
        };

        let data = self.borrow_owner();
        let mut result = Vec::new();
        
        for chunk in &schema.chunks {
            // Bounds check: chunk offset is within file
            let offset_start = (chunk.byte_offset as usize)
                .checked_add(CHUNK_HEADER_SIZE as usize)
                .ok_or_else(|| {
                    std::io::Error::new(ErrorKind::InvalidData, "chunk offset overflow")
                })?;
            
            if offset_start > data.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("chunk offset out of bounds: offset={}, file_size={}", offset_start, data.len())
                ).into());
            }

            // Calculate actual data length: total_len - header - checksum
            let chunk_total_len = chunk.byte_len as usize;
            let chunk_overhead = (CHUNK_HEADER_SIZE + CHUNK_CHECKSUM_SIZE) as usize;
            
            if chunk_total_len < chunk_overhead {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("chunk size too small: total_len={}, overhead={}", chunk_total_len, chunk_overhead)
                ).into());
            }
            
            let data_len = chunk_total_len - chunk_overhead;
            
            // Bounds check: slice end is within file
            let offset_end = offset_start.checked_add(data_len)
                .ok_or_else(|| {
                    std::io::Error::new(ErrorKind::InvalidData, "chunk data size overflow")
                })?;
            
            if offset_end > data.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("chunk data out of bounds: end={}, file_size={}", offset_end, data.len())
                ).into());
            }
            
            result.extend_from_slice(&data[offset_start..offset_end]);
        }
        Ok(Some(result))
    }
}
