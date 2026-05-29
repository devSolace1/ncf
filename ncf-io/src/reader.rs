use ciborium::de::from_reader;
use memmap2::Mmap;
use ncf_core::header::{FileHeaderPrefix, NcfHeader};
use ncf_core::index::IndexEntry;
use ncf_core::schema::ChunkRef;
use ncf_core::Result;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::convert::TryInto;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(borrow)]
pub struct BorrowedTensorSchema<'a> {
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
#[serde(borrow)]
struct RawBorrowedNcfIndex<'a> {
    entry_count: u64,
    entries: Vec<IndexEntry>,
    tensor_map: BTreeMap<&'a str, u64>,
}

pub struct NcfReader<'a> {
    pub metadata: NcfHeader,
    pub schemas: Vec<BorrowedTensorSchema<'a>>,
    pub index: BorrowedNcfIndex<'a>,
    pub header_prefix: FileHeaderPrefix,
    pub mmap: Mmap,
}

impl<'a> NcfReader<'a> {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 48 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file too small").into());
        }

        let header_prefix = FileHeaderPrefix::decode(&mmap[..48])?;
        let header_start = 48;
        let header_end = header_start + header_prefix.header_len as usize;
        let metadata = NcfHeader::decode_cbor(&mmap[header_start..header_end])?;

        let schema_start = header_prefix.schema_offset as usize;
        let schema_end = header_prefix.index_offset as usize;
        let schemas: Vec<BorrowedTensorSchema<'a>> = from_reader(Cursor::new(&mmap[schema_start..schema_end]))?;

        let footer_position = mmap.len() - 16;
        let footer_magic = &mmap[footer_position..footer_position + 8];
        if footer_magic != b"NCFEND!!" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Missing footer magic").into());
        }

        let footer_len_bytes: [u8; 8] = mmap[footer_position + 8..footer_position + 16]
            .try_into()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid footer length"))?;
        let index_len = u64::from_le_bytes(footer_len_bytes) as usize;
        let index_start = header_prefix.index_offset as usize;
        let index_end = index_start + index_len;

        let raw_index: RawBorrowedNcfIndex<'a> = from_reader(Cursor::new(&mmap[index_start..index_end]))?;
        let mut tensor_map = HashMap::with_capacity(raw_index.tensor_map.len());
        for (name, chunk_id) in raw_index.tensor_map {
            tensor_map.insert(name, chunk_id);
        }

        let index = BorrowedNcfIndex {
            entry_count: raw_index.entry_count,
            entries: raw_index.entries,
            tensor_map,
        };

        Ok(Self {
            metadata,
            schemas,
            index,
            header_prefix,
            mmap,
        })
    }

    pub fn inspect(&self) {
        println!("Model: {}", self.metadata.metadata.model_name);
        println!("Architecture: {}", self.metadata.metadata.architecture);
        println!("Tensors: {}", self.schemas.len());
        for tensor in &self.schemas {
            println!(" - {} {} {:?}", tensor.name, tensor.dtype, tensor.shape);
        }
    }

    pub fn find_schema(&self, name: &str) -> Option<&BorrowedTensorSchema<'a>> {
        self.schemas.iter().find(|schema| schema.name == name)
    }

    pub fn read_tensor(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let schema = match self.find_schema(name) {
            Some(schema) => schema,
            None => return Ok(None),
        };

        let mut result = Vec::new();
        for chunk in &schema.chunks {
            let offset = chunk.byte_offset as usize + 30;
            let end = offset + (chunk.byte_len as usize - 30 - 32);
            result.extend_from_slice(&self.mmap[offset..end]);
        }
        Ok(Some(result))
    }
}
